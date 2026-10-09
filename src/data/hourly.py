"""Hourly accumulations with the screen run on the 15-min frames (DECISIONS §21-§23).

The target is the 1-h accumulation in the convention of OPERA's ACRR: the hour ending at H is
0.25 x (frames H-45, H-30, H-15, H), on pixels valid in all four frames (DECISIONS §21). The
screen runs on the frames, because most error signatures exist only there, and it decides
with a whole day of frames before it repairs any one of them: a stationary artefact and a
stationary storm look alike within one hour, but not over a day (RESEARCH_NOTES §7.4e).

A processing day d is the 96 frames d 00:15 .. d+1 00:00, i.e. the 24 hours ending d 01:00 ..
d+1 00:00, plus d 00:00 and d+1 00:15 as neighbours for the temporal tests.

Chain (each step can be switched off in `cfg["rules"]`, for the per-rule ablation of
DECISIONS §22):
  per frame   clean_frame (drizzle, static clutter with hot_fallback, spikes) -> ceilings ->
              footprints of > 500 mm/h cores -> range rings >= 89 mm/h      (as v4)
  multi-frame rays: thin lines through a radar site (`cleaning.anchored_lines`) present at
              the same pixels in >= ray_min_frames of the +-ray_half_window frames
  day         local-peak persistence: n_peak = frames in which a pixel is >= peak_u and
              >= peak_ratio x its 5x5 median. A pixel is flagged when n_peak >= p1_min
              (p1_ring_min on climatological range rings, p1_prior_min on pixels of the
              `prior` mask, e.g. the per-pixel Mahalanobis mask), or when it lies on a thin
              site-anchored line of the n_peak >= p2_min map (rays inside rain). In every frame
              where a flagged pixel is a local peak it takes its 5x5 median.
  per frame   cells without temporal support (as v3/v4), judged on the frames above
  per hour    isolated single-frame bursts (`burst`, off until accepted; DECISIONS §26): one
              frame carries > 80% of the pixel-hour at >= 100 mm/h and the hour's 5x5 median
              is < 1 mm; that frame's pixel takes its 5x5 median
  hour        sum of the four frames; optionally (`advection`) also the advection-corrected
              mean over the same window, h["sum_adv"] (DECISIONS §25)

Repairs only ever lower a value. Codes are the REPAIR_* bits of `cleaning`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from src.data import cleaning as C

STEP = np.timedelta64(15, "m")
RULES = ("hot", "spike", "ceiling", "footprint", "ring", "ray", "persist", "unsupported", "burst")
DEFAULTS = {
    "hot_fallback": "keep",
    "max_size": 500,                 # v4 guard
    "ray_u": 1.0, "ray_half_window": 4, "ray_min_frames": 3,
    "peak_u": 1.0, "peak_ratio": 2.0,
    "p1_min": 24, "p1_ring_min": 12, "p2_min": 12,
    "p1_prior_min": 12,              # on pixels of a prior mask (`prior`), e.g. the Mahalanobis mask
    "rules": {**{r: True for r in RULES}, "burst": False},   # burst: off until accepted (§26)
    "burst": {"min_share": 0.8, "min_rate": 100.0, "iso_median": 1.0},
    "advection": None,               # None: plain sum only; {"mode": "blend"|"noblend",
                                     # "interp": "linear"|"cubic"|"nearest"} adds h["sum_adv"]
}


def config(q: dict | None = None) -> dict:
    """Merge the `hourly:` block of a quality config (or a plain dict) over DEFAULTS."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    h = (q or {}).get("hourly", q or {})
    for k, v in h.items():
        if k == "rules":
            cfg["rules"].update(v)
        elif k == "burst" and isinstance(v, dict):
            cfg["burst"].update(v)
        elif k in DEFAULTS:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------------------------
# advection-corrected accumulation (DECISIONS §21 status note, §25)
# ---------------------------------------------------------------------------------------------
_GRID = {}


def motion_lk(r1, r2):
    """Dense motion (u, v) in px per frame step from frame r1 to r2: pysteps Lucas-Kanade on
    log rates. Zero where there is too little rain to track."""
    from pysteps import motion
    a, b = (np.nan_to_num(r, nan=0.0) for r in (r1, r2))
    if (a >= 0.5).sum() < 50 or (b >= 0.5).sum() < 50:
        return np.zeros((2,) + a.shape, np.float32)
    x = np.stack([np.log(np.maximum(a, 0.1)), np.log(np.maximum(b, 0.1))])
    return motion.get_method("LK")(x, dense=True, verbose=False).astype(np.float32)


def warp(f, V, s, interp="linear"):
    """f moved by s x V, semi-Lagrangian with the motion taken at the destination:
    out(x) = f(x - s V(x)); zero outside the grid. cv2.remap works in fixed point (1/32 px);
    cubic can overshoot, so negative values are clipped."""
    import cv2
    H, W = f.shape
    if (H, W) not in _GRID:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _GRID[(H, W)] = (yy, xx)
    yy, xx = _GRID[(H, W)]
    flag = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}[interp]
    out = cv2.remap(f.astype(np.float32), xx - s * V[0], yy - s * V[1], flag,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.maximum(out, 0.0) if interp == "cubic" else out


def advected_mean(frames, motions, t0, t1, mode="blend", interp="linear", step_min=15.0, dt=1.0):
    """Mean rate over frame-times t0 .. t1 (minutes after frames[0]) from 15-min frames, with the
    rain moved along the motion between them (pysteps advection correction).
    frames: consecutive frames step_min apart (zeros, not NaN); motions[j]: frame j -> j+1.
    mode "blend": (1 - s) x frame j moved forward + s x frame j+1 moved backward (pysteps);
    "noblend": frame j moved forward for s < 0.5, frame j+1 moved backward otherwise, so two
    imperfectly aligned intensities are never averaged."""
    acc, n = 0.0, 0
    for m in np.arange(t0 + dt / 2, t1, dt):
        j = min(int(m // step_min), len(motions) - 1)
        s = (m - j * step_min) / step_min
        V = motions[j]
        if mode == "blend":
            f = (1 - s) * warp(frames[j], V, s, interp) + s * warp(frames[j + 1], V, -(1 - s), interp)
        else:
            f = warp(frames[j], V, s, interp) if s < 0.5 else warp(frames[j + 1], V, -(1 - s), interp)
        acc = acc + f
        n += 1
    return acc / n


def median5(z: np.ndarray, where: np.ndarray) -> np.ndarray:
    """5x5 median of `z` (NaN -> 0) at the pixels of `where` only; elsewhere 0. Same values
    as `ndimage.median_filter(z, 5)` (mode "reflect" = numpy "symmetric") at a fraction of the
    cost, because wet pixels are a few % of the grid."""
    out = np.zeros(z.shape, np.float32)
    ys, xs = np.nonzero(where)
    if ys.size:
        w = sliding_window_view(np.pad(np.nan_to_num(z, nan=0.0), 2, mode="symmetric"), (5, 5))
        out[ys, xs] = np.median(w[ys, xs].reshape(ys.size, 25), axis=1)
    return out


def local_peaks(z: np.ndarray, u: float, ratio: float):
    """(peak mask, 5x5 median map) of one frame: pixels >= u and >= ratio x their 5x5 median
    (floored at the drizzle level, so a peak in dry surroundings counts)."""
    z0 = np.nan_to_num(z, nan=0.0)
    cand = z0 >= u
    med = median5(z0, cand)
    return cand & (z0 >= ratio * np.maximum(med, C.DRIZZLE)), med


def _open(path):
    import xarray as xr
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:                                    # noqa: BLE001
        return xr.open_zarr(path, consolidated=False)


class HourlyDay:
    """One processing day. `hot(year)` / `ring(year)` return full-grid masks or None;
    `sites_rc` (N, 2) radar positions; `ceilings` a `radar_screen.CeilingTable` or None;
    `crop` (y0, y1, x0, x1) of the full grid. Call `run()`, then read:

      frames      (T,) timestamps of the processing frames (d 00:15 .. d+1 00:00 present)
      fin, code   (T, H, W) cleaned rates and repair codes of those frames
      raw         (T, H, W) raw rates (NaN = no coverage)
      hours       list of dicts {end, idx (4 frame indices), sum, raw_sum, max_rate, valid}
      signals     dict of per-pixel day maps: n_peak, n_wet, n_valid, flag_p1, flag_p2
    """

    def __init__(self, raw_dir, day, var, hot, ring=None, sites_rc=None, ceilings=None,
                 cfg=None, crop=None, prior=None):
        self.raw_dir, self.day, self.var = raw_dir, str(day), var
        self.hot_fn, self.ring_fn, self.sites, self.ceilings = hot, ring, sites_rc, ceilings
        self.prior_fn = prior          # year -> full-grid bool mask; lowers p1 there (DECISIONS §25)
        self.cfg = config(cfg)
        self.crop = crop
        self.year = int(self.day[:4])

    # ------------------------------------------------------------------ reading
    def _read_window(self):
        """Raw frames d 00:00 .. d+1 00:15 that exist, as (times, stack)."""
        d0 = pd.Timestamp(self.day)
        lo, hi = np.datetime64(d0), np.datetime64(d0 + pd.Timedelta(days=1, minutes=15))
        times, frames = [], []
        for d in (d0, d0 + pd.Timedelta(days=1)):
            p = os.path.join(self.raw_dir, d.strftime("%Y%m%d"))
            if not os.path.exists(os.path.join(p, ".zmetadata")):
                continue
            ds = _open(p)
            H, W = ds.sizes["y"], ds.sizes["x"]
            y0, y1, x0, x1 = self.crop or (0, H, 0, W)
            self.crop = (y0, y1, x0, x1)
            tv = ds.time.values
            sel = np.nonzero((tv >= lo) & (tv <= hi))[0]
            if sel.size:
                a = ds[self.var].isel(time=sel, y=slice(y0, y1), x=slice(x0, x1)).values
                times += list(tv[sel]); frames.append(a.astype(np.float32))
            ds.close()
        if not frames:
            return np.array([], "datetime64[ns]"), None
        t = np.array(times, "datetime64[ns]")
        o = np.argsort(t)
        return t[o], np.concatenate(frames)[o]

    def _sl(self, a):
        if a is None:
            return None
        y0, y1, x0, x1 = self.crop
        return a[y0:y1, x0:x1]

    # ------------------------------------------------------------------ chain
    def run(self):
        cfg, R = self.cfg, self.cfg["rules"]
        times, raw = self._read_window()
        d0 = np.datetime64(pd.Timestamp(self.day))
        self.hours, self.signals = [], {}
        if raw is None:
            self.frames = times
            return self
        T = len(times)
        y0, _, x0, _ = self.crop
        hot = self._sl(self.hot_fn(self.year)) if (R["hot"] and self.hot_fn) else None
        ring = self._sl(self.ring_fn(self.year)) if self.ring_fn else None
        S = np.empty_like(raw)
        code = np.zeros(raw.shape, np.uint16)                            # REPAIR_BURST needs bit 8
        kw = {"max_size": cfg["max_size"]}
        for t in range(T):                                               # per-frame rules
            c, info = C.clean_frame(raw[t], hot, hot_fallback=cfg["hot_fallback"],
                                    spikes=R["spike"])
            if "hot_kept" in info:
                code[t][info["hot_kept"]] |= C.REPAIR_HOT_KEPT
            if R["ceiling"] and self.ceilings is not None:
                from src.data.radar_screen import repair_ceiling
                cm = self.ceilings.mask(c, self.year, self.crop)
                repair_ceiling(c, cm)
                code[t][cm] |= C.REPAIR_CEILING
            ref = np.zeros(c.shape, bool)
            if R["footprint"]:
                z0 = np.nan_to_num(c, nan=0.0)
                core = z0 > C.UNPHYSICAL
                if core.any():
                    lab, _ = ndimage.label(z0 >= C.FOOTPRINT_MIN, structure=C._EIGHT)
                    hit = np.unique(lab[core])
                    fp = np.isin(lab, hit[hit > 0])
                    code[t][C.boundary_fill(c, fp, refused=ref, **kw)] |= C.REPAIR_FOOTPRINT
            if R["ring"] and ring is not None:
                rm = ring & (np.nan_to_num(c, nan=0.0) >= C.REPAIR_RING_MIN)
                if rm.any():
                    code[t][C.boundary_fill(c, rm, refused=ref, **kw)] |= C.REPAIR_RING
            code[t][ref] |= C.REPAIR_REFUSED
            S[t] = c

        if R["ray"] and self.sites is not None:                          # multi-frame rays
            cand = np.stack([C.anchored_lines(S[t], self.sites, u=cfg["ray_u"], row0=y0, col0=x0)
                             for t in range(T)])
            if cand.any():
                dil = np.stack([ndimage.binary_dilation(m, C._EIGHT) if m.any() else m for m in cand])
                cnt = ndimage.uniform_filter1d(dil.astype(np.float32), 2 * cfg["ray_half_window"] + 1,
                                               axis=0, mode="constant") * (2 * cfg["ray_half_window"] + 1)
                ray = cand & (np.rint(cnt) >= cfg["ray_min_frames"])
                for t in np.nonzero(ray.any(axis=(1, 2)))[0]:
                    ref = np.zeros(S[t].shape, bool)
                    code[t][C.boundary_fill(S[t], ray[t], refused=ref, **kw)] |= C.REPAIR_RAY
                    code[t][ref] |= C.REPAIR_REFUSED

        proc = (times > d0) & (times <= d0 + np.timedelta64(1, "D"))     # d 00:15 .. d+1 00:00
        valid = np.isfinite(raw)
        peaks, meds = [], []
        for t in range(T):
            p, m = local_peaks(S[t], cfg["peak_u"], cfg["peak_ratio"])
            peaks.append(p); meds.append(m)
        n_peak = np.sum([peaks[t] for t in range(T) if proc[t]], axis=0).astype(np.int16) \
            if proc.any() else np.zeros(raw.shape[1:], np.int16)
        self.signals = {"n_peak": n_peak,
                        "n_wet": (np.nan_to_num(S[proc], nan=0.0) >= C.DRIZZLE).sum(0).astype(np.int16),
                        "n_valid": valid[proc].sum(0).astype(np.int16)}
        if R["persist"]:                                                  # local-peak persistence
            thr = np.full(n_peak.shape, cfg["p1_min"], np.int16)
            if ring is not None:
                thr[ring] = cfg["p1_ring_min"]
            prior = self._sl(self.prior_fn(self.year)) if self.prior_fn else None
            if prior is not None:
                thr[prior] = np.minimum(thr[prior], cfg["p1_prior_min"])
            f1 = n_peak >= thr
            f2 = C.anchored_lines(n_peak.astype(np.float32), self.sites, u=cfg["p2_min"],
                                  row0=y0, col0=x0) if self.sites is not None else np.zeros_like(f1)
            self.signals.update(flag_p1=f1, flag_p2=f2)
            flag = f1 | f2
            if flag.any():
                for t in range(T):
                    m = flag & peaks[t] & (meds[t] < np.nan_to_num(S[t], nan=0.0))
                    if m.any():
                        S[t][m] = meds[t][m]
                        code[t][m] |= C.REPAIR_PERSIST
        del peaks, meds

        if R["unsupported"]:                                              # temporal support
            maps = [C.support_maps(S[t]) for t in range(T)]
            fin = S.copy()
            for t in range(T):
                prev = maps[t - 1] if t > 0 and times[t] - times[t - 1] == STEP else None
                nxt = maps[t + 1] if t + 1 < T and times[t + 1] - times[t] == STEP else None
                fin[t], code[t] = C.repair_unsupported(S[t], prev, nxt, code[t], max_size=cfg["max_size"])
            S = fin

        nsec = lambda t: int(np.datetime64(t, "ns").astype(np.int64))
        if R.get("burst"):                                                # isolated single-frame bursts
            bc = cfg["burst"]                                             # (DECISIONS §26)
            at = {nsec(t): j for j, t in enumerate(times)}
            n_burst = np.zeros(raw.shape[1:], np.int16)
            for e in times[proc]:
                if (e - d0) % np.timedelta64(60, "m"):
                    continue
                want = [e - k * STEP for k in (3, 2, 1, 0)]                # the hour's frames, H-45 .. H
                if not all(nsec(w) in at for w in want):
                    continue
                jj = [at[nsec(w)] for w in want]
                F4 = np.nan_to_num(S[jj], nan=0.0)
                tot, fmax, kmax = F4.sum(0), F4.max(0), F4.argmax(0)
                cand = (fmax >= bc["min_rate"]) & (fmax > bc["min_share"] * np.maximum(tot, 1e-6))
                if not cand.any():
                    continue
                iso = median5(0.25 * tot, cand) < bc["iso_median"]       # the hour has ~no rain around
                flag = cand & iso
                if not flag.any():
                    continue
                for slot in range(4):
                    m = flag & (kmax == slot)
                    if m.any():
                        t = jj[slot]
                        med = median5(S[t], m)
                        low = m & (med < np.nan_to_num(S[t], nan=0.0))
                        S[t][low] = med[low]
                        code[t][low] |= C.REPAIR_BURST
                        n_burst += low
            self.signals["n_burst"] = n_burst

        adv = {}
        if cfg["advection"]:                       # hour ending H covers frame-times H-52.5 .. H+7.5
            ac = cfg["advection"]                  # (ODYSSEY frame t covers t-10 .. t+5, centre t-2.5)
            at = {nsec(t): j for j, t in enumerate(times)}
            Z = np.nan_to_num(S, nan=0.0)
            mo = {}
            for e in times[proc]:
                if (e - d0) % np.timedelta64(60, "m"):
                    continue
                want = [e - k * STEP for k in (4, 3, 2, 1, 0, -1)]          # H-60 .. H+15
                if not all(nsec(w) in at for w in want):
                    continue
                jj = [at[nsec(w)] for w in want]
                for a_, b_ in zip(jj[:-1], jj[1:]):
                    if (a_, b_) not in mo:
                        mo[(a_, b_)] = motion_lk(Z[a_], Z[b_])
                adv[nsec(e)] = advected_mean([Z[j] for j in jj], [mo[(a_, b_)] for a_, b_ in zip(jj[:-1], jj[1:])],
                                             7.5, 67.5, ac.get("mode", "blend"), ac.get("interp", "linear"))
        keep = np.nonzero(proc)[0]
        self.frames, self.fin, self.code, self.raw = times[keep], S[keep], code[keep], raw[keep]
        pos = {t: i for i, t in enumerate(self.frames)}
        for e in range(len(self.frames)):                                 # hours, ACRR convention
            end = self.frames[e]
            if (end - d0) % np.timedelta64(60, "m"):
                continue
            want = [end - k * STEP for k in (3, 2, 1, 0)]
            if not all(w in pos for w in want):
                continue
            idx = [pos[w] for w in want]
            f, r = self.fin[idx], self.raw[idx]
            ok = np.isfinite(r).all(0)
            self.hours.append({"end": end, "idx": idx,
                               "sum": np.where(ok, 0.25 * np.nan_to_num(f, nan=0.0).sum(0), np.nan).astype(np.float32),
                               "sum_adv": (np.where(ok, adv[int(np.datetime64(end, "ns").astype(np.int64))], np.nan).astype(np.float32)
                                           if int(np.datetime64(end, "ns").astype(np.int64)) in adv else None),
                               "raw_sum": np.where(ok, 0.25 * np.nan_to_num(r, nan=0.0).sum(0), np.nan).astype(np.float32),
                               "max_rate": np.where(ok, np.nan_to_num(f, nan=0.0).max(0), np.nan).astype(np.float32),
                               "valid": ok})
        return self
