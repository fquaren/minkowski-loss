#!/usr/bin/env python
"""OPERA (screened, hourly) vs RADKLIM over Germany: geometry and the heavy tail
(2026-10-08; inputs to the main-dataset decision, EXPERIMENTS §5; DECISIONS §24 result).

Rebuilds the hourly fields from the `radklim_test.py` samples. Those hold every pixel-hour
with max(raw, RADKLIM) >= 1 mm, and the screen only lowers values, so for thresholds >= 1 mm
the excursion sets of raw (A), screened (C) and RADKLIM (K) are exact. Valid pixels: RADKLIM
finite and inside a fully covered OPERA tile-hour. No 128x128 tile is fully valid in RADKLIM, so
tiles >= 98% valid are used with the invalid pixels zeroed in A, C and K alike.

  run      per day, per 128x128 tile-hour fully inside RADKLIM coverage with max(C, K) >= 5 mm:
             - Minkowski functionals (area fraction, Crofton perimeter in km, Euler
               characteristic V - E + F; as `daily_check/geom_compare.py`) of A, C, K at
               1, 2, 5, 10, 20, 30 mm;
             - peak ratio max / max(25 km block mean);
             - unresolved variance fraction U of log1p(field) w.r.t. the 25 km mean.
           And per screened pixel-hour C >= 30 mm (tile-hour kept): RADKLIM at the pixel and
           the max of RADKLIM within +-2 px and +-1 h. If RADKLIM is dry even within that
           tolerance, the peak is a residual artefact; if it is there, the peak is real and
           displaced or smoothed.
  summary  medians of C/K and A/K per functional and threshold, perimeter per area, the Euler
           characteristic chi (median, median difference to RADKLIM, Spearman rank correlation
           with RADKLIM across tile-hours), and the tolerant confirmation by C bin, for random
           and event days.

    python scripts/data_quality/radklim_geometry.py run --workers 6
    python scripts/data_quality/radklim_geometry.py summary
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
D = "/home/fquareng/work/data/extremes/OPERA"
CAL = f"{D}/quality_v4/calib"
SAMP = f"{CAL}/radklim_test/days"
RKD = f"{D}/validation/radklim/hourly"
OUT = f"{CAL}/radklim_geometry"
THR = np.array([1, 2, 5, 10, 20, 30], np.float32)


def functionals(x, thr):
    """x (N,128,128), thr (Q,) -> area fraction, perimeter (km), Euler characteristic, (N,Q)."""
    from src.data.gamma import _CROFTON_W_AXIAL as WA, _CROFTON_W_DIAG as WD
    m = x[:, None] >= thr[None, :, None, None]
    A = m.mean((2, 3))
    mi = m.astype(np.int8)
    ax = np.abs(np.diff(mi, axis=3)).sum((2, 3)) + np.abs(np.diff(mi, axis=2)).sum((2, 3))
    d1 = np.abs(mi[:, :, 1:, 1:] - mi[:, :, :-1, :-1]).sum((2, 3))
    d2 = np.abs(mi[:, :, 1:, :-1] - mi[:, :, :-1, 1:]).sum((2, 3))
    P = (WA * ax + WD * (d1 + d2)) * 2.0
    V = m.sum((2, 3))
    E = (m[:, :, :, 1:] & m[:, :, :, :-1]).sum((2, 3)) + (m[:, :, 1:] & m[:, :, :-1]).sum((2, 3))
    F_ = (m[:, :, 1:, 1:] & m[:, :, :-1, :-1] & m[:, :, 1:, :-1] & m[:, :, :-1, 1:]).sum((2, 3))
    return A, P, (V - E + F_)


def coarse_stats(x):
    """peak ratio and unresolved variance fraction of log1p(x) w.r.t. 10x10 block means (~25 km)."""
    n = len(x)
    xs = x[:, :120, :120]                                        # 128 is not a multiple of 10
    blk = xs.reshape(n, 10, 12, 10, 12).mean((2, 4))
    pr = xs.reshape(n, -1).max(1) / np.maximum(blk.reshape(n, -1).max(1), 1e-3)
    lx = np.log1p(xs)
    lb = np.repeat(np.repeat(np.log1p(xs).reshape(n, 10, 12, 10, 12).mean((2, 4)), 12, 1), 12, 2)
    U = ((lx - lb) ** 2).reshape(n, -1).mean(1) / np.maximum(lx.reshape(n, -1).var(1), 1e-9)
    return pr, U


def run_day(day):
    f = f"{SAMP}/{day}.npz"
    z = np.load(f)
    if not len(z["A"]):
        return day, None, None
    rk = np.load(f"{RKD}/{day}.npz")
    K24, (y0, y1, x0, x1) = rk["K"], rk["box"]
    H, W = y1 - y0, x1 - x0
    tiles, peaks = [], []
    hrs = z["hour"].astype(int)
    for i in np.unique(hrs):
        s = hrs == i
        r, c = z["row"][s].astype(int) - y0, z["col"][s].astype(int) - x0
        fA, fC, fK = (np.zeros((H, W), np.float32) for _ in range(3))
        fA[r, c], fC[r, c] = z["A"][s], np.where(z["rejected"][s], np.nan, z["C"][s])
        Kv = K24[i]
        fK = np.nan_to_num(Kv, nan=0.0)
        valid = np.isfinite(Kv)
        # tile-hours present in the samples define coverage; take OPERA tiles inside the box
        for r0 in range(-(-y0 // 128) * 128, y1 - 127, 128):
            for c0 in range(-(-x0 // 128) * 128, x1 - 127, 128):
                sl = (slice(r0 - y0, r0 - y0 + 128), slice(c0 - x0, c0 - x0 + 128))
                vm = valid[sl]
                if vm.mean() < 0.98:                     # no tile is fully valid in RADKLIM; the
                    continue                             # <= 2% invalid pixels are zeroed in all three
                a, cc, k = np.where(vm, fA[sl], 0.0), np.where(vm, fC[sl], 0.0), np.where(vm, fK[sl], 0.0)
                if np.isnan(cc).any() or max(np.nanmax(cc), k.max()) < 5:
                    continue
                tiles.append((day, i, r0, c0, a, cc, k))
        kmax = ndimage.maximum_filter(np.stack([np.nan_to_num(K24[j], nan=0.0) for j in
                                                range(max(i - 1, 0), min(i + 2, 24))]).max(0), size=5)
        hv = np.nan_to_num(fC, nan=0.0) >= 30
        if hv.any():
            yy, xx = np.nonzero(hv)
            peaks.append(pd.DataFrame({"day": day, "hour": i, "C": fC[yy, xx], "A": fA[yy, xx],
                                       "K": fK[yy, xx], "K_tol": kmax[yy, xx]}))
    rows = []
    if tiles:
        X = {k: np.stack([t[j] for t in tiles]) for j, k in ((4, "A"), (5, "C"), (6, "K"))}
        res = {}
        for k, x in X.items():
            Af, P, chi = functionals(x, THR)
            pr, U = coarse_stats(x)
            res[k] = (Af, P, chi, pr, U)
        for n, t in enumerate(tiles):
            d = {"day": t[0], "hour": t[1], "row": t[2], "col": t[3]}
            for k, (Af, P, chi, pr, U) in res.items():
                d[f"max_{k}"] = float(X[k][n].max())
                d[f"pr_{k}"], d[f"U_{k}"] = float(pr[n]), float(U[n])
                for q, u in enumerate(THR):
                    d[f"A{int(u)}_{k}"], d[f"P{int(u)}_{k}"], d[f"X{int(u)}_{k}"] = float(Af[n, q]), float(P[n, q]), float(chi[n, q])
            rows.append(d)
    return day, pd.DataFrame(rows), (pd.concat(peaks) if peaks else None)


def summary():
    T = pd.read_csv(f"{OUT}/tiles.csv.gz", dtype={"day": str})
    Pk = pd.read_csv(f"{OUT}/peaks.csv.gz", dtype={"day": str})
    cats = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day")["category"]
    grp = lambda d: "random" if cats.get(d) == "random" else ("event" if str(cats.get(d, "")).startswith("event") else "failure")
    T["grp"], Pk["grp"] = T.day.map(grp), Pk.day.map(grp)
    out = []
    for g in ("random", "event"):
        t = T[T.grp == g]
        out.append(f"\n=== {g}: {len(t):,} tile-hours (max(C, K) >= 5 mm) on {t.day.nunique()} days")
        rows = []
        for u in THR.astype(int):
            both = t[(t[f"A{u}_C"] > 0) & (t[f"A{u}_K"] > 0)]
            if len(both) < 20:
                continue
            ppa = lambda k: (both[f"P{u}_{k}"] / (both[f"A{u}_{k}"] * 128 * 128 * 4)).median()   # km per km2
            rows.append({"u_mm": u, "n": len(both),
                         "area C/K": (both[f"A{u}_C"] / both[f"A{u}_K"]).median(),
                         "perim C/K": (both[f"P{u}_C"] / both[f"P{u}_K"]).median(),
                         "perim/area C": ppa("C"), "perim/area K": ppa("K"), "perim/area A": ppa("A"),
                         "chi C": both[f"X{u}_C"].median(), "chi K": both[f"X{u}_K"].median(),
                         "chi A": both[f"X{u}_A"].median(),
                         "med chi C-K": (both[f"X{u}_C"] - both[f"X{u}_K"]).median(),
                         "med chi A-K": (both[f"X{u}_A"] - both[f"X{u}_K"]).median(),
                         "rho chi C,K": both[f"X{u}_C"].corr(both[f"X{u}_K"], method="spearman"),
                         "rho chi A,K": both[f"X{u}_A"].corr(both[f"X{u}_K"], method="spearman")})
        out.append("Minkowski functionals on tile-hours where both have pixels above u (medians):\n"
                   + pd.DataFrame(rows).round(3).to_string(index=False))
        out.append(f"peak ratio (max / max 25 km mean): A {t.pr_A.median():.2f}  C {t.pr_C.median():.2f}  K {t.pr_K.median():.2f}; "
                   f"unresolved variance U (log1p): A {t.U_A.median():.3f}  C {t.U_C.median():.3f}  K {t.U_K.median():.3f}")
        p = Pk[Pk.grp == g]
        rr = []
        for lo, hi in ((30, 50), (50, 75), (75, 100), (100, np.inf)):
            b = p[(p.C >= lo) & (p.C < hi)]
            if len(b):
                rr.append({"C_bin": f"{lo}-{hi}", "n": len(b), "K>=0.5C at pixel": (b.K >= 0.5 * b.C).mean(),
                           "K>=0.5C within 2 px, 1 h": (b.K_tol >= 0.5 * b.C).mean(),
                           "K_tol < 5 mm (dry)": (b.K_tol < 5).mean(), "median K_tol/C": (b.K_tol / b.C).median()})
        out.append("screened heavy pixel-hours vs RADKLIM:\n" + pd.DataFrame(rr).round(3).to_string(index=False))
    txt = "\n".join(out)
    print(txt)
    open(f"{OUT}/summary.txt", "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    os.makedirs(OUT, exist_ok=True)
    days = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{SAMP}/*.npz"))
    T, P = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for day, t, p in ex.map(run_day, days):
            if t is not None and len(t):
                T.append(t)
            if p is not None:
                P.append(p)
    pd.concat(T).to_csv(f"{OUT}/tiles.csv.gz", index=False)
    pd.concat(P).to_csv(f"{OUT}/peaks.csv.gz", index=False)
    print(f"[radklim_geometry] {sum(len(t) for t in T):,} tile-hours, {sum(len(p) for p in P):,} heavy pixel-hours")


if __name__ == "__main__":
    main()
