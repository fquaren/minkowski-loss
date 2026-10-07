"""v4 screen: reflectivity ceilings, repeated values and radar-wide failures (DECISIONS §19).

The v3 rules (`cleaning.py`) assume an artefact that is small, local or transient against real
rain. Three failures found on 2026-10-06 are none of these (RESEARCH_NOTES §7.4d), and this
module holds the rules that catch them. Each keys on an error signature, never on intensity
alone (DECISIONS §17 safeguard 1), and none is tuned on the tail fit (DECISIONS §20).

  repeated value  real rain on the 0.01 mm/h ODYSSEY grid almost never repeats one value
                  (per tail tile: median 1, 99.9th percentile 24 repeats of one value >= 31).
                  The count is compared with the other occupied values within +-0.5 dB, so
                  the same test works on NIMBUS, whose values sit on a ladder (ratio 1.0593
                  between levels).
  ceiling         a value a radar area produces far more often than any other value within
                  +-4 dB over a year (364.63 mm/h = 64.0 dBZ, 48.62 = 50.0 dBZ under
                  Z = 200 R^1.6). Not the +-0.5 dB median of the repeated-value test: a
                  radar-year mixes coarse ladders (steps up to ~3 dB) with sparse off-ladder
                  values, so that median is ~1 and every ladder level looks like a ceiling
                  (Phase-0 calibration, DECISIONS §19). Rain counts fall with intensity, so on
                  a ladder the next level is about as heavy; a ceiling piles up above it.
                  A ceiling pixel is censored, not measured: it is repaired like a spike, and a
                  tile-frame with >= 5 of them is rejected at split time.
  radar frame     per radar and frame, over the pixels it owns (nearest active radar within
                  250 km, the rule of `scan_flags._sites_year`): wet fraction, exceedance
                  fractions, mean, the share of log-rate variance explained by range alone
                  (R2_r = 1 - Var(z - <z>_theta(r)) / Var(z)), and the jump in log-rate across
                  the area boundary and across the maximum-range circle. A receiver or
                  calibration fault is a function of range with a sharp edge; rain is not.

Ownership is an approximation: the composite takes each pixel from the radar(s) chosen by
quality, usually but not always the nearest one (`geo.nearest_radar`).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

MP_B = 1.6                                   # Z = 200 R^1.6: dBZ = 23.01 + 16 log10(R)
WINDOW_DB = 0.5
WINDOW = 10 ** (WINDOW_DB / (10 * MP_B))     # +-0.5 dB as a rate factor, 1.0746
CEIL_WINDOW_DB = 4.0                         # wider than the coarsest ladder step seen (~3 dB)
REP_U = 31.0                                 # repeated-value test on values >= this (mm/h)
VB0, VB1 = 1000, 50000                       # histogram bins: 10.00 .. 500.00 mm/h on the 0.01 grid
RANGE_BIN_KM = 4.0
MAX_KM = 250.0
BAND_PX = 3                                  # width of the boundary / max-range bands (6 km)
FEATURES = ("n_valid", "wet", "f10", "f31", "f89", "f150", "mean", "r2_range",
            "bjump", "ejump", "n_band_in", "n_band_out")


def dbz(rate):
    """Reflectivity (dBZ) of a rain rate (mm/h) under Z = 200 R^1.6."""
    return 10 * np.log10(200.0 * np.asarray(rate, float) ** MP_B)


# --------------------------------------------------------------------------------------------
# repeated values and ceilings: a count against its occupied neighbours within +-0.5 dB
# --------------------------------------------------------------------------------------------
def excess(vb: np.ndarray, c: np.ndarray, candidates=None, window: float = WINDOW) -> np.ndarray:
    """Count of each value divided by the median count of the *other occupied* values within
    +-0.5 dB. `vb`: sorted unique value bins (value * 100, int), `c`: their counts.
    Occupied-only, so a coarse ladder (NIMBUS) is compared level to level; 1 when a value
    has no occupied neighbour. `candidates`: indices to evaluate (others get NaN)."""
    out = np.full(vb.size, np.nan)
    if vb.size == 0:
        return out
    idx = np.arange(vb.size) if candidates is None else np.asarray(candidates)
    lo = np.searchsorted(vb, vb[idx] / window, side="left")
    hi = np.searchsorted(vb, vb[idx] * window, side="right")
    for j, i in enumerate(idx):
        nb = np.r_[c[lo[j]:i], c[i + 1:hi[j]]]
        out[i] = c[i] / max(float(np.median(nb)), 1.0) if nb.size else float(c[i])
    return out


def repeated_value(t: np.ndarray, u: float = REP_U, min_count: int = 10) -> dict:
    """The most anomalous repeated value >= u in one tile.

    Returns rep_value, rep_count, rep_excess for the value with the largest excess among
    those counted >= min_count (else the most frequent one), and max_rep (the largest count
    of any single value)."""
    v = t[np.isfinite(t) & (t >= u)]
    if v.size == 0:
        return {"rep_value": np.nan, "rep_count": 0, "rep_excess": 0.0, "max_rep": 0}
    vb, c = np.unique(np.rint(v * 100).astype(np.int64), return_counts=True)
    cand = np.nonzero(c >= min_count)[0]
    if cand.size:
        ex = excess(vb, c, cand)
        i = cand[np.nanargmax(ex[cand])]
    else:
        i = int(np.argmax(c))
        ex = excess(vb, c, [i])
    return {"rep_value": vb[i] / 100.0, "rep_count": int(c[i]), "rep_excess": float(ex[i]),
            "max_rep": int(c.max())}


def peak_ratio(vb: np.ndarray, c: np.ndarray, candidates, window_db: float = CEIL_WINDOW_DB) -> np.ndarray:
    """Count of each candidate divided by the largest count of any *other* value within
    +-window_db (sorted `vb`, value * 100). The max, not the median: on a ladder the
    comparison is the adjacent level, whatever sparse off-ladder values sit between levels."""
    idx = np.asarray(candidates)
    w = 10 ** (window_db / (10 * MP_B))
    lo = np.searchsorted(vb, vb[idx] / w, side="left")
    hi = np.searchsorted(vb, vb[idx] * w, side="right")
    out = np.empty(idx.size)
    for j, i in enumerate(idx):
        nb = np.r_[c[lo[j]:i], c[i + 1:hi[j]]]
        out[j] = c[i] / max(float(nb.max()) if nb.size else 1.0, 1.0)
    return out


def ceiling_candidates(vb: np.ndarray, c: np.ndarray, min_count: int = 50,
                       min_ratio: float = 5.0, window_db: float = CEIL_WINDOW_DB):
    """Values of one radar-year histogram counted >= min_count and >= min_ratio times the
    largest other count within +-window_db. Returns (values mm/h, counts, ratios)."""
    o = np.argsort(vb)
    vb, c = vb[o], c[o]
    cand = np.nonzero(c >= min_count)[0]
    if not cand.size:
        return np.empty(0), np.empty(0, int), np.empty(0)
    r = peak_ratio(vb, c, cand, window_db)
    k = r >= min_ratio
    return vb[cand[k]] / 100.0, c[cand[k]], r[k]


class CeilingTable:
    """Ceilings per year: a value is masked within `MAX_KM` of the radar it was found for.

    `rows`: iterable of dicts with year, value, site_row, site_col (grid position of the radar).
    """

    def __init__(self, rows, shape=(2200, 1900)):
        self.shape = shape
        self.by_year = {}
        for r in rows:
            self.by_year.setdefault(int(r["year"]), []).append(
                (int(round(float(r["value"]) * 100)), float(r["site_row"]), float(r["site_col"])))
        self._disk = {}

    def _disks(self, year):
        if year not in self._disk:
            H, W = self.shape
            rr, cc = np.ogrid[0:H, 0:W]
            d = {}
            for vb, sr, sc in self.by_year.get(year, []):
                m = (rr - sr) ** 2 + (cc - sc) ** 2 <= (MAX_KM / 2.0) ** 2
                d[vb] = d[vb] | m if vb in d else m
            self._disk[year] = d
        return self._disk[year]

    def mask(self, z: np.ndarray, year: int, crop=None) -> np.ndarray:
        """Boolean map of ceiling pixels of frame `z` (or of the `crop` (y0, y1, x0, x1) of the
        full grid that `z` covers)."""
        out = np.zeros(z.shape, bool)
        disks = self._disks(year)
        if not disks:
            return out
        y0, y1, x0, x1 = crop or (0, self.shape[0], 0, self.shape[1])
        k = np.rint(np.nan_to_num(z, nan=0.0) * 100).astype(np.int64)
        for vb, disk in disks.items():
            hit = k == vb
            if hit.any():
                out |= hit & disk[y0:y1, x0:x1]
        return out


def repair_ceiling(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace ceiling pixels by the median of their valid, non-ceiling 8-neighbours (as a
    spike). A pixel with no such neighbour keeps its value: it is counted, and the tile-frame
    is rejected at split time when it holds >= 5 of them. In place; returns the changed map."""
    from src.data.quality import _neighbour_stack
    changed = np.zeros(z.shape, bool)
    if not mask.any():
        return changed
    src = np.where(mask, np.nan, z)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, z.shape[0])
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, z.shape[1])
    st = _neighbour_stack(src[y0:y1, x0:x1])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(st, axis=0)
    m = mask[y0:y1, x0:x1] & np.isfinite(med)
    zs = z[y0:y1, x0:x1]
    zs[m] = med[m]
    changed[y0:y1, x0:x1] = m
    return changed


# --------------------------------------------------------------------------------------------
# radar ownership and per-radar-frame features
# --------------------------------------------------------------------------------------------
def _year(v):
    try:
        return int(str(v).strip()[:4])
    except (TypeError, ValueError):
        return None


def active_sites(year: int, sites=None):
    """Radars active in `year` near the grid (the rule of `scan_flags._sites_year`), one row
    per key (ODIM code, or location when there is none): database duplicates merged."""
    from src.data import geo
    s = geo.load_radar_sites() if sites is None else sites
    s0, s1 = s["startyear"].map(_year), s["finishyear"].map(_year)
    ok = ((s0.isna() | (s0 <= year)) & (s1.isna() | (s1 >= year))
          & s["row"].between(-200, 2400) & s["col"].between(-200, 2100))
    act = s[ok].copy()
    act["key"] = np.where(act["odim"].str.len() > 0, act["odim"], act["location"])
    return act.drop_duplicates("key").reset_index(drop=True)


class RadarGeometry:
    """Per-year ownership (nearest active radar within 250 km, as `scan_flags._sites_year`,
    with database duplicates of one site merged by ODIM code / location) and the pixel index
    sets the per-radar-frame features need."""

    def __init__(self, year: int, shape=(2200, 1900), sites=None):
        from src.data import geo
        act = active_sites(year, sites)
        self.year, self.shape, self.sites = year, shape, act
        self.keys = act["key"].values
        H, W = shape
        rr, cc = np.mgrid[0:H, 0:W]
        x, y = geo.rowcol_to_xy(rr, cc)
        idx, dist = geo.nearest_radar(x, y, act, max_km=MAX_KM)
        self.owner = idx.astype(np.int16)
        self.R = len(act)
        flat = self.owner.ravel()
        self.own_idx = np.nonzero(flat >= 0)[0]
        self.own = flat[self.own_idx].astype(np.int64)
        nb = int(np.ceil(MAX_KM / RANGE_BIN_KM)) + 1
        rb = np.minimum((dist.ravel()[self.own_idx] // RANGE_BIN_KM).astype(np.int64), nb - 1)
        self.nb = nb
        self.key_r = self.own * nb + rb
        self._bands()

    def _bands(self):
        """Inner / outer bands, BAND_PX wide, of each radar's owned area and of its
        maximum-range circle; flat indices with the radar id."""
        H, W = self.shape
        st = np.ones((3, 3), bool)
        bi, br, bo, bro, ei, er, eo, ero = ([] for _ in range(8))
        for k in range(self.R):
            m = self.owner == k
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            y0, y1 = max(ys.min() - BAND_PX - 1, 0), min(ys.max() + BAND_PX + 2, H)
            x0, x1 = max(xs.min() - BAND_PX - 1, 0), min(xs.max() + BAND_PX + 2, W)
            mm = m[y0:y1, x0:x1]
            inner = mm & ~ndimage.binary_erosion(mm, st, iterations=BAND_PX, border_value=1)
            outer = ndimage.binary_dilation(mm, st, iterations=BAND_PX) & ~mm
            for band, li, lr in ((inner, bi, br), (outer, bo, bro)):
                yy, xx = np.nonzero(band)
                li.append((yy + y0) * W + (xx + x0)); lr.append(np.full(yy.size, k))
            rng = self.sites["maxrange_km"].iat[k]
            if np.isfinite(rng) and rng > 0:
                sr, sc = self.sites["row"].iat[k], self.sites["col"].iat[k]
                rpx = rng / 2.0
                Y0, Y1 = int(max(sr - rpx - BAND_PX - 1, 0)), int(min(sr + rpx + BAND_PX + 2, H))
                X0, X1 = int(max(sc - rpx - BAND_PX - 1, 0)), int(min(sc + rpx + BAND_PX + 2, W))
                if Y1 > Y0 and X1 > X0:
                    yy, xx = np.mgrid[Y0:Y1, X0:X1]
                    d = np.hypot(yy - sr, xx - sc)
                    for band, li, lr in (((d >= rpx - BAND_PX) & (d < rpx), ei, er),
                                         ((d >= rpx) & (d < rpx + BAND_PX), eo, ero)):
                        li.append((yy[band]) * W + xx[band]); lr.append(np.full(band.sum(), k))
        cat = lambda a: np.concatenate(a) if a else np.empty(0, np.int64)  # noqa: E731
        self.b_in, self.b_in_r = cat(bi), cat(br)
        self.b_out, self.b_out_r = cat(bo), cat(bro)
        self.e_in, self.e_in_r = cat(ei), cat(er)
        self.e_out, self.e_out_r = cat(eo), cat(ero)


def _band_mean(lz, valid, idx, rid, R, min_n=50):
    v = valid[idx]
    n = np.bincount(rid[v], minlength=R)
    s = np.bincount(rid[v], weights=lz[idx][v], minlength=R)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def radar_frame_features(z: np.ndarray, g: RadarGeometry, min_wet: float = 0.2,
                         min_valid: int = 500) -> np.ndarray:
    """Per radar features of one full frame `z` (mm/h, NaN = no coverage), shape
    (R, len(FEATURES)), float32. R2_r only where the wet fraction is >= min_wet."""
    R = g.R
    flat = z.ravel()
    vz = flat[g.own_idx]
    ok = np.isfinite(vz)
    own = g.own[ok]
    v = vz[ok]
    lz_own = np.log1p(np.maximum(v, 0.0))
    n = np.bincount(own, minlength=R).astype(float)
    nz = np.maximum(n, 1)
    out = np.full((R, len(FEATURES)), np.nan, np.float32)
    out[:, 0] = n
    for j, u in ((1, 1.0), (2, 10.0), (3, 31.0), (4, 89.0), (5, 150.0)):
        out[:, j] = np.bincount(own, weights=(v >= u).astype(float), minlength=R) / nz
    out[:, 6] = np.bincount(own, weights=v, minlength=R) / nz
    # R2_r: residual of log1p(z) about its per-(radar, range bin) mean
    kr = g.key_r[ok]
    nr = np.bincount(kr, minlength=R * g.nb)
    sr = np.bincount(kr, weights=lz_own, minlength=R * g.nb)
    mr = sr / np.maximum(nr, 1)
    m_all = np.bincount(own, weights=lz_own, minlength=R) / nz
    ss_tot = np.bincount(own, weights=(lz_own - m_all[own]) ** 2, minlength=R)
    ss_res = np.bincount(own, weights=(lz_own - mr[kr]) ** 2, minlength=R)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    out[:, 7] = np.where((out[:, 1] >= min_wet) & (n >= min_valid) & (ss_tot > 0), r2, np.nan)
    # jumps across the ownership boundary and the maximum-range circle
    lz = np.log1p(np.maximum(np.nan_to_num(flat, nan=0.0), 0.0))
    valid = np.isfinite(flat)
    bi, nbi = _band_mean(lz, valid, g.b_in, g.b_in_r, R)
    bo, nbo = _band_mean(lz, valid, g.b_out, g.b_out_r, R)
    ei, _ = _band_mean(lz, valid, g.e_in, g.e_in_r, R)
    eo, _ = _band_mean(lz, valid, g.e_out, g.e_out_r, R)
    out[:, 8] = bi - bo
    out[:, 9] = ei - eo
    out[:, 10], out[:, 11] = nbi, nbo
    out[n < min_valid, 1:] = np.nan
    return out


def value_histogram(z: np.ndarray, g: RadarGeometry):
    """Sparse per-radar counts of values 10..500 mm/h on the 0.01 grid of one frame:
    (keys = radar * (VB1 - VB0 + 1) + (bin - VB0), counts)."""
    vz = z.ravel()[g.own_idx]
    ok = np.isfinite(vz) & (vz >= VB0 / 100.0) & (vz <= VB1 / 100.0)
    if not ok.any():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    b = np.rint(vz[ok] * 100).astype(np.int64) - VB0
    key = g.own[ok] * (VB1 - VB0 + 1) + b
    return np.unique(key, return_counts=True)


def owner_shares(g: RadarGeometry, r0: int, c0: int, patch: int = 128, top: int = 3) -> str:
    """'key:frac;key:frac' of the radars owning a tile (largest first), for joining
    radar-frame flags and radar-day exclusions onto tiles at split time."""
    o = g.owner[r0:r0 + patch, c0:c0 + patch].ravel()
    o = o[o >= 0]
    if not o.size:
        return ""
    k, c = np.unique(o, return_counts=True)
    order = np.argsort(-c)[:top]
    return ";".join(f"{g.keys[k[i]]}:{c[i] / (patch * patch):.3f}" for i in order)


# --------------------------------------------------------------------------------------------
# split-time decisions (DECISIONS §19: rules decide from stored columns, so thresholds can
# change without a rescan)
# --------------------------------------------------------------------------------------------
def frame_flagged(feats: np.ndarray, q: dict) -> np.ndarray:
    """Rule 3 on a (..., len(FEATURES)) array: True where the radar-frame looks like a
    radar-wide failure. NaN features never flag."""
    f = {k: feats[..., i] for i, k in enumerate(FEATURES)}
    p = q["radar_frame"]
    ok = f["n_valid"] >= p["min_valid"]
    with np.errstate(invalid="ignore"):
        a = (f["f31"] >= p["f31_min"]) & (f["r2_range"] >= p["r2_min"])
        jump = np.fmax(f["ejump"], f["bjump"])
        b = (jump >= p["jump_min"]) & (f["f10"] >= p["f10_min"])
    return ok & (a | b)


def _owned_share(owners: str, flagged: set, ts=None) -> float:
    if not isinstance(owners, str) or not owners:
        return 0.0
    s = 0.0
    for part in owners.split(";"):
        k, v = part.rsplit(":", 1)
        if (k, ts) in flagged:
            s += float(v)
    return s


def decide(tiles, q: dict, frame_flags: set = frozenset(), day_excl: set = frozenset()):
    """Per-tile v4 rejection columns for a v4 tile table (`scan_tiles.py --quality`).

    frame_flags: {(radar key, timestamp str)} of flagged radar-frames (rule 3);
    day_excl:    {(radar key, YYYYMMDD)} of reviewed radar-day exclusions (rule 5).
    Returns a DataFrame with rej_ceiling, rej_repeat, rej_refused, rej_radar_frame,
    rej_radar_day, radar_fail_frac, radar_day_frac and rej_v4 (any)."""
    import pandas as pd
    t = tiles
    ts = t["timestamp"].astype(str).values
    day = np.array([s[:8] for s in ts])
    own = t["owners"].values
    rf = np.array([_owned_share(o, frame_flags, s) for o, s in zip(own, ts)]) if frame_flags \
        else np.zeros(len(t))
    rd = np.array([_owned_share(o, day_excl, d) for o, d in zip(own, day)]) if day_excl \
        else np.zeros(len(t))
    r = q["repeat"]
    out = pd.DataFrame({
        "rej_ceiling": t["n_ceiling"].values >= q["ceiling"]["reject_pixels"],
        "rej_repeat": (t["rep_count"].values >= r["min_count"]) & (t["rep_excess"].values >= r["min_excess"]),
        "rej_refused": t["n_refused"].values > 0,
        "radar_fail_frac": rf,
        "rej_radar_frame": rf >= q["radar_frame"]["reject_share"],
        "radar_day_frac": rd,
        "rej_radar_day": rd >= q["radar_day"]["reject_share"],
    }, index=t.index)
    out["rej_v4"] = out[["rej_ceiling", "rej_repeat", "rej_refused", "rej_radar_frame",
                         "rej_radar_day"]].any(axis=1)
    return out
