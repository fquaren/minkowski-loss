#!/usr/bin/env python
"""Is OPERA's extra fine structure over RADKLIM signal or noise? (2026-10-08; input to the
main-dataset decision, EXPERIMENTS §5; follows `radklim_geometry.py`)

Over Germany, OPERA's screened hourly sums have about twice RADKLIM's Euler characteristic at
1 mm, 25-35% more perimeter per area, and more variance below 25 km. Three explanations:
  (a) measurement noise in the 15-min OPERA frames;
  (b) beading: the hour is 4 snapshots of a cell moving 4-9 px per frame, while RADKLIM's
      hour is built from 5-min data. A sampling artefact of our accumulation, not radar noise;
  (c) real structure that RADKLIM's processing smooths away.

Per 64x64 OPERA tile >= 98% valid in RADKLIM, per hour with rain (max >= 5 mm and wet
fraction >= 5% in OPERA or RADKLIM):
  motion   tile displacement per 15 min from phase correlation of consecutive frames
  C        the screened hourly sum (0.25 x 4 frames, ACRR convention)
  Cadv     advection-corrected: each frame shifted along the motion to -6, -2, +2, +6 min
           and averaged, so a moving cell leaves a track instead of 4 beads
  K        RADKLIM, paired by hour
  geometry Minkowski functionals (area fraction, perimeter, Euler characteristic chi) at
           1, 2, 5, 10, 20 mm, and the unresolved variance U (log1p, w.r.t. 8x8 block means,
           ~16 km)
  spectra  radially binned power of log1p(C), log1p(Cadv), log1p(K) and the cross-spectra
           with K (Hann window), pooled into coherence per wavelength
  persistence  correlation of the fine band (difference of Gaussians, sigma 1 and 4 px, ~4-16
           km) of consecutive 15-min frames, at fixed pixels (Eulerian) and after shifting
           along the motion (Lagrangian); the same for a coarse band (sigma 4 and 16 px)

Reading:
  (b) chi(C) - chi(K) shrinks for Cadv and grows with motion speed;
  (a) the fine band does not persist along the motion (Lagrangian ~ Eulerian ~ 0);
  (c) the fine band persists, and the excess survives advection correction.

    python scripts/data_quality/signal_noise.py run --workers 6
    python scripts/data_quality/signal_noise.py summary
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
D = "/home/fquareng/work/data/extremes/OPERA"
CAL = f"{D}/quality_v4/calib"
RKD = f"{D}/validation/radklim/hourly"
OUT = f"{CAL}/signal_noise"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
T = 64
THR = np.array([1, 2, 5, 10, 20], np.float32)
NK = T // 2
_WIN = np.outer(np.hanning(T), np.hanning(T))
_KY, _KX = np.meshgrid(np.fft.fftfreq(T), np.fft.fftfreq(T), indexing="ij")
_KBIN = np.minimum((np.hypot(_KY, _KX) * T).round().astype(int), NK)


SPEED_CLASSES = ("slow", "mid", "fast", "nan")


def speed_class(v):
    """px per 15 min: slow < 2 (4 km), mid 2-6, fast >= 6 (12 km)."""
    if not np.isfinite(v):
        return "nan"
    return "slow" if v < 2 else ("mid" if v < 6 else "fast")


def functionals(x, thr):
    from src.data.gamma import _CROFTON_W_AXIAL as WA, _CROFTON_W_DIAG as WD
    m = x[None] >= thr[:, None, None]
    A = m.mean((1, 2))
    mi = m.astype(np.int8)
    ax = np.abs(np.diff(mi, axis=2)).sum((1, 2)) + np.abs(np.diff(mi, axis=1)).sum((1, 2))
    d1 = np.abs(mi[:, 1:, 1:] - mi[:, :-1, :-1]).sum((1, 2))
    d2 = np.abs(mi[:, 1:, :-1] - mi[:, :-1, 1:]).sum((1, 2))
    P = (WA * ax + WD * (d1 + d2)) * 2.0
    V = m.sum((1, 2))
    E = (m[:, :, 1:] & m[:, :, :-1]).sum((1, 2)) + (m[:, 1:] & m[:, :-1]).sum((1, 2))
    F_ = (m[:, 1:, 1:] & m[:, :-1, :-1] & m[:, 1:, :-1] & m[:, :-1, 1:]).sum((1, 2))
    return A, P, V - E + F_


def unresolved(x):
    lx = np.log1p(x)
    lb = np.repeat(np.repeat(lx.reshape(8, 8, 8, 8).mean((1, 3)), 8, 0), 8, 1)
    return float(((lx - lb) ** 2).mean() / max(lx.var(), 1e-9))


def spec(a):
    return np.fft.fft2((a - a.mean()) * _WIN)


def radial(z):
    return np.bincount(_KBIN.ravel(), weights=z.ravel(), minlength=NK + 1)


def shift(f, dy, dx):
    return ndimage.shift(f, (dy, dx), order=1, mode="constant", cval=0.0)


def displacement(a, b):
    """(dy, dx) in px that moves a onto b, by phase correlation of log1p fields."""
    from skimage.registration import phase_cross_correlation
    if a.max() < 0.5 or b.max() < 0.5:
        return np.nan, np.nan
    s, _, _ = phase_cross_correlation(np.log1p(b), np.log1p(a), upsample_factor=4, normalization=None)
    return float(s[0]), float(s[1])


def bands(x, s1, s2):
    lx = np.log1p(x)
    return ndimage.gaussian_filter(lx, s1) - ndimage.gaussian_filter(lx, s2)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    t0 = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (by0, by1, bx0, bx1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, None, None
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q, crop=(by0, by1, bx0, bx1)).run()
    d0 = np.datetime64(pd.Timestamp(day))
    rows = []
    keys = ("PC", "PA", "PK", "SCKr", "SCKi", "SAKr", "SAKi", "PR", "PRA", "SRKr", "SRKi", "SRAKr", "SRAKi")
    acc = {f"{k}_{c}": np.zeros(NK + 1) for k in keys for c in SPEED_CLASSES}
    acc.update({f"n_{c}": 0 for c in SPEED_CLASSES})
    valid_any = np.isfinite(K24).all(0)
    for h in hd.hours:
        i = int((h["end"] - d0) / np.timedelta64(1, "h")) - 1
        if not 0 <= i < 24:
            continue
        idx = h["idx"]
        F = np.nan_to_num(hd.fin[idx], nan=0.0)
        # RADKLIM-aligned hour: frames H-60..H-15 cover H-70..H-10 (ODYSSEY labels t-10..t+5),
        # exactly the RADKLIM file labelled H-10; needs the frame before the ACRR hour
        al = [idx[0] - 1] + list(idx[:3]) if idx[0] >= 1 and hd.frames[idx[0]] - hd.frames[idx[0] - 1] == np.timedelta64(15, "m") else None
        FA = np.nan_to_num(hd.fin[al], nan=0.0) if al is not None else None
        Kh = K24[i]
        for r0 in range(0, by1 - by0 - T + 1, T):
            for c0 in range(0, bx1 - bx0 - T + 1, T):
                sl = (slice(r0, r0 + T), slice(c0, c0 + T))
                vk = np.isfinite(Kh[sl]) & h["valid"][sl]
                if vk.mean() < 0.98:
                    continue
                f = F[:, sl[0], sl[1]] * vk
                k = np.where(vk, np.nan_to_num(Kh[sl], nan=0.0), 0.0)
                c = 0.25 * f.sum(0)
                wet = max((c >= 0.1).mean(), (k >= 0.1).mean())
                if max(c.max(), k.max()) < 5 or wet < 0.05:
                    continue
                disp = [displacement(f[j], f[j + 1]) for j in range(3)]
                dd = np.array([d for d in disp if np.isfinite(d[0])])
                if len(dd):
                    vy, vx = np.median(dd, 0)
                else:
                    vy = vx = np.nan
                speed = float(np.hypot(vy, vx)) if np.isfinite(vy) else np.nan
                if np.isfinite(speed):
                    sub = [shift(f[j], vy * o / 15.0, vx * o / 15.0) for j in range(4) for o in (-6, -2, 2, 6)]
                    cadv = np.mean(sub, 0) * vk
                else:
                    cadv = c
                d = {"day": day, "hour": i, "row": by0 + r0, "col": bx0 + c0, "speed_px15": speed,
                     "max_C": float(c.max()), "max_Cadv": float(cadv.max()), "max_K": float(k.max())}
                for name, x in (("C", c), ("Cadv", cadv), ("K", k)):
                    A, P, chi = functionals(x, THR)
                    d[f"U_{name}"] = unresolved(x)
                    for qq, u in enumerate(THR.astype(int)):
                        d[f"A{u}_{name}"], d[f"P{u}_{name}"], d[f"X{u}_{name}"] = float(A[qq]), float(P[qq]), float(chi[qq])
                # persistence of fine / coarse bands between consecutive frames
                ef, lf, ec, lc = [], [], [], []
                for j in range(3):
                    a, b = f[j], f[j + 1]
                    if a.max() < 0.5 or b.max() < 0.5:
                        continue
                    fa, fb = bands(a, 1, 4), bands(b, 1, 4)
                    ca, cb = bands(a, 4, 16), bands(b, 4, 16)
                    ef.append(corr(fa, fb)); ec.append(corr(ca, cb))
                    if np.isfinite(speed):
                        lf.append(corr(shift(fa, *disp[j]) if np.isfinite(disp[j][0]) else fa, fb))
                        lc.append(corr(shift(ca, *disp[j]) if np.isfinite(disp[j][0]) else ca, cb))
                for key, v in (("fine_eul", ef), ("fine_lag", lf), ("coarse_eul", ec), ("coarse_lag", lc)):
                    d[key] = float(np.nanmean(v)) if len(v) else np.nan
                if FA is not None:
                    fa_ = FA[:, sl[0], sl[1]] * vk
                    ca_ = 0.25 * fa_.sum(0)
                    caadv = (np.mean([shift(fa_[j], vy * o / 15.0, vx * o / 15.0) for j in range(4) for o in (-6, -2, 2, 6)], 0) * vk
                             if np.isfinite(speed) else ca_)
                    for name, x in (("R", ca_), ("Radv", caadv)):
                        A_, P_, chi_ = functionals(x, THR)
                        for qq, u in enumerate(THR.astype(int)):
                            d[f"X{u}_{name}"], d[f"P{u}_{name}"] = float(chi_[qq]), float(P_[qq])
                rows.append(d)
                cl = speed_class(speed)
                sC, sA, sK = spec(np.log1p(c)), spec(np.log1p(cadv)), spec(np.log1p(k))
                acc[f"PC_{cl}"] += radial(np.abs(sC) ** 2); acc[f"PA_{cl}"] += radial(np.abs(sA) ** 2)
                acc[f"PK_{cl}"] += radial(np.abs(sK) ** 2)
                x1, x2 = sC * np.conj(sK), sA * np.conj(sK)
                acc[f"SCKr_{cl}"] += radial(x1.real); acc[f"SCKi_{cl}"] += radial(x1.imag)
                acc[f"SAKr_{cl}"] += radial(x2.real); acc[f"SAKi_{cl}"] += radial(x2.imag)
                if FA is not None:
                    sR, sRA = spec(np.log1p(ca_)), spec(np.log1p(caadv))
                    acc[f"PR_{cl}"] += radial(np.abs(sR) ** 2); acc[f"PRA_{cl}"] += radial(np.abs(sRA) ** 2)
                    x3, x4 = sR * np.conj(sK), sRA * np.conj(sK)
                    acc[f"SRKr_{cl}"] += radial(x3.real); acc[f"SRKi_{cl}"] += radial(x3.imag)
                    acc[f"SRAKr_{cl}"] += radial(x4.real); acc[f"SRAKi_{cl}"] += radial(x4.imag)
                acc[f"n_{cl}"] += 1
    os.makedirs(os.path.join(OUT, "days"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "days", f"{day}.npz"), **{k: np.asarray(v) for k, v in acc.items()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "days", f"{day}.csv.gz"), index=False)
    return day, len(rows), time.time() - t0


def summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def _read(f):
        try:
            return pd.read_csv(f, dtype={"day": str})
        except pd.errors.EmptyDataError:                 # a day with no qualifying tile-hour
            return None
    R = pd.concat([x for x in map(_read, glob.glob(f"{OUT}/days/*.csv.gz")) if x is not None and len(x)],
                  ignore_index=True)
    S = [np.load(f) for f in glob.glob(f"{OUT}/days/*.npz")]
    acc = {k: sum(s[k] for s in S) for k in S[0].files}
    lines = [f"{R.day.nunique()} days, {len(R):,} tile-hours (64x64 px, >= 98% RADKLIM-valid)"]
    k = np.arange(NK + 1)
    lam = np.where(k > 0, T * 2.0 / np.maximum(k, 1), np.inf)       # km
    coh = lambda r, i, p1, p2: (r ** 2 + i ** 2) / np.maximum(p1 * p2, 1e-12)
    sel = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]                       # 128 .. 4 km
    curves = {}
    for cl in ("all",) + SPEED_CLASSES[:3]:
        cls = SPEED_CLASSES if cl == "all" else (cl,)
        g = lambda key: sum(acc[f"{key}_{c}"] for c in cls)
        n = sum(acc[f"n_{c}"] for c in cls)
        cC = coh(g("SCKr"), g("SCKi"), g("PC"), g("PK")); cA = coh(g("SAKr"), g("SAKi"), g("PA"), g("PK"))
        cR = coh(g("SRKr"), g("SRKi"), g("PR"), g("PK")); cRA = coh(g("SRAKr"), g("SRAKi"), g("PRA"), g("PK"))
        curves[cl] = (cC, cA, cR, cRA, g("PC") / max(n, 1), g("PA") / max(n, 1), g("PK") / max(n, 1), g("PR") / max(n, 1))
        tab = pd.DataFrame({"wavelength_km": lam[sel], "coh ACRR": cC[sel], "coh ACRR adv": cA[sel],
                            "coh aligned": cR[sel], "coh aligned adv": cRA[sel],
                            "power ACRR/K": (g("PC") / g("PK"))[sel], "power aligned/K": (g("PR") / g("PK"))[sel],
                            "power aligned adv/K": (g("PRA") / g("PK"))[sel]})
        lines.append(f"\ncoherence with RADKLIM and power ratio by wavelength (log1p), speed class {cl}, n = {n:,}:\n"
                     + tab.round(3).to_string(index=False))
    cC, cA = curves["all"][0], curves["all"][1]
    rows = []
    for u in THR.astype(int):
        b = R[(R[f"A{u}_C"] > 0) & (R[f"A{u}_K"] > 0)]
        if len(b) < 30:
            continue
        rows.append({"u_mm": u, "n": len(b), "chi C": b[f"X{u}_C"].median(), "chi Cadv": b[f"X{u}_Cadv"].median(),
                     "chi K": b[f"X{u}_K"].median(),
                     "med chi C-K": (b[f"X{u}_C"] - b[f"X{u}_K"]).median(),
                     "med chi Cadv-K": (b[f"X{u}_Cadv"] - b[f"X{u}_K"]).median(),
                     "perim C/K": (b[f"P{u}_C"] / b[f"P{u}_K"]).median(),
                     "perim Cadv/K": (b[f"P{u}_Cadv"] / b[f"P{u}_K"]).median()})
    rr = []
    for u in THR.astype(int):
        bb = R[(R[f"X{u}_R"].notna()) & (R[f"A{u}_K"] > 0)] if f"X{u}_R" in R else R.iloc[:0]
        if len(bb) >= 30:
            rr.append({"u_mm": u, "n": len(bb), "chi aligned": bb[f"X{u}_R"].median(), "chi aligned adv": bb[f"X{u}_Radv"].median(),
                       "chi K": bb[f"X{u}_K"].median(), "med chi aligned-K": (bb[f"X{u}_R"] - bb[f"X{u}_K"]).median(),
                       "med chi aligned adv-K": (bb[f"X{u}_Radv"] - bb[f"X{u}_K"]).median()})
    if rr:
        lines.append("Euler characteristic, RADKLIM-aligned OPERA hour (frames H-60..H-15) vs RADKLIM:\n" + pd.DataFrame(rr).round(3).to_string(index=False))
    lines.append("Euler characteristic and perimeter, plain vs advection-corrected OPERA hour vs RADKLIM:\n"
                 + pd.DataFrame(rows).round(3).to_string(index=False))
    lines.append(f"unresolved variance U (log1p, ~16 km blocks): C {R.U_C.median():.3f}  Cadv {R.U_Cadv.median():.3f}  K {R.U_K.median():.3f}")
    R["sbin"] = pd.cut(R.speed_px15, [0, 1, 2, 4, 6, 10, 64], right=False)
    g = R.groupby("sbin", observed=True)
    sp = pd.DataFrame({"n": g.size(), "chi2 C-K": g.apply(lambda x: (x.X2_C - x.X2_K).median()),
                       "chi2 Cadv-K": g.apply(lambda x: (x.X2_Cadv - x.X2_K).median()),
                       "fine eul": g.fine_eul.median(), "fine lag": g.fine_lag.median(),
                       "coarse eul": g.coarse_eul.median(), "coarse lag": g.coarse_lag.median()})
    lines.append("by motion speed (px per 15 min; 1 px = 2 km): chi excess at 2 mm and band persistence between 15-min frames:\n"
                 + sp.round(3).to_string())
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/summary.txt", "w").write(txt)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].semilogx(lam[1:], cC[1:], label="OPERA hour vs RADKLIM"); ax[0].semilogx(lam[1:], cA[1:], label="OPERA advection-corrected vs RADKLIM")
    ax[0].set_xlabel("wavelength (km)"); ax[0].set_ylabel("coherence"); ax[0].invert_xaxis(); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
    _, _, cR, cRA, PC, PA, PK, PR = curves["all"]
    ax[0].semilogx(lam[1:], cR[1:], "--", label="aligned hour vs RADKLIM"); ax[0].semilogx(lam[1:], cRA[1:], "--", label="aligned adv.-corrected vs RADKLIM")
    ax[1].loglog(lam[1:], PC[1:], label="OPERA ACRR hour"); ax[1].loglog(lam[1:], PA[1:], label="OPERA adv.-corrected")
    ax[1].loglog(lam[1:], PR[1:], "--", label="OPERA aligned hour"); ax[1].loglog(lam[1:], PK[1:], label="RADKLIM"); ax[1].invert_xaxis()
    ax[1].set_xlabel("wavelength (km)"); ax[1].set_ylabel("power of log1p field"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
    fig.savefig(f"{OUT}/spectra.png", dpi=90)
    print("figure:", f"{OUT}/spectra.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--days", nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    have = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{CAL}/radklim_test/days/*.npz"))
    days = [d for d in (a.days or have) if not os.path.exists(os.path.join(OUT, "days", f"{d}.npz"))]
    print(f"[signal_noise] {len(days)} days -> {OUT}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, dt in ex.map(run_day, days):
            print(f"  {day}: {n} tile-hours, {dt if dt is None else round(dt)} s", flush=True)


if __name__ == "__main__":
    main()
