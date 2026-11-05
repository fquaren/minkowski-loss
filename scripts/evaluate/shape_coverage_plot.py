#!/usr/bin/env python
"""
Are the held-out extremes inside the shape space the cut-trained R4 models see? (DECISIONS §43, H1)

Pairs of peak-normalised Minkowski features (`src.evaluation.analogue.shape_features`: area, perimeter, Euler
characteristic of {x >= f * peak}) at three fractions f, one panel per pair:
    background  train tile-hours with 5 <= target max < u_c (what the cut models train on; cached features)
    points      test tile-hours with target max >= u_c (the R4 held-out extremes)
Bottom row: each feature at f = 0.3 against the target peak itself, with the cut drawn (the intensity drift).
Per panel: the share of extreme points that fall in a hexagon holding >= 5 cut-training hours.
2-D projections only: a point inside every 2-D projection can still be outside the 21-D joint support.

Usage: python scripts/evaluate/shape_coverage_plot.py --out notes/data_analysis_v4/shape_coverage_r4.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from src.evaluation.analogue import FRACTIONS, shape_features

V4 = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv"
AN = "/home/fquareng/work/data/extremes/OPERA/quality_v4/analogue"
BG_CMAP = "Blues"              # sequential, one hue: density of cut-training hours
PT = "#eb6834"                 # categorical slot 2 (reference palette): held-out extremes
INK, INK2 = "#0b0b0b", "#52514e"
NAMES = ("A", "P", "χ")
LABEL = {"A": "log1p(area / km²)", "P": "log1p(perimeter / km)", "χ": "sign · log1p|Euler χ|"}


def meta(split):
    return pd.read_csv(f"{V4}/meta/full_{split}.txt", header=None, names=["t", "y", "x", "mx", "r"])


def feat(X, name, f):
    k = FRACTIONS.index(f)
    return X[:, NAMES.index(name) * len(FRACTIONS) + k]


def coverage(xb, yb, xp, yp, gs, extent, min_count=5):
    """Share of points in hexagons holding >= min_count background points (same hexagon grid for both)."""
    fig = plt.figure()
    hb = plt.hexbin(xb, yb, gridsize=gs, extent=extent, mincnt=1)
    cnt, off = hb.get_array(), hb.get_offsets()
    plt.close(fig)
    occupied = off[cnt >= min_count]
    if len(occupied) == 0 or len(xp) == 0:
        return float("nan")
    # nearest hexagon centre as matplotlib assigns points: grid units sx = dx / nx, sy = dy / ny with
    # ny = int(nx / sqrt(3)), and distance dx'^2 + 3 dy'^2 in those units
    sx = (extent[1] - extent[0]) / gs
    sy = (extent[3] - extent[2]) / int(gs / np.sqrt(3))
    allc = off / [sx, sy]
    p = np.c_[xp, yp] / [sx, sy]
    d = (p[:, None, 0] - allc[None, :, 0]) ** 2 + 3 * (p[:, None, 1] - allc[None, :, 1]) ** 2
    nearest = np.argmin(d, axis=1)
    ok = cnt[nearest] >= min_count
    return float(ok.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--u_c", type=float, default=50.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tr = meta("train")
    c = np.load(f"{AN}/shape_train_ref5.npz")
    mx_tr = tr.set_index("r").loc[c["rows"], "mx"].values
    Xb = c["X"][mx_tr < a.u_c]
    pk_all, X_all = mx_tr, c["X"]

    te = meta("test")
    te = te[te.mx >= a.u_c]
    cache = f"{AN}/shape_test_above{a.u_c:g}.npz"
    if os.path.exists(cache) and np.array_equal(np.load(cache)["rows"], te.r.values):
        Xp = np.load(cache)["X"]
    else:
        z = zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")["test"]["original_precip"]
        Xp = np.stack([shape_features(z[int(r)]) for r in te.r.values])
        np.savez_compressed(cache, X=Xp, rows=te.r.values)
    print(f"cut-training hours {len(Xb)}, held-out extremes {len(Xp)}")

    pairs = (("A", "P"), ("A", "χ"), ("P", "χ"))
    fr = (0.1, 0.3, 0.7)
    fig, ax = plt.subplots(4, 3, figsize=(13, 16.5), constrained_layout=True)
    rows_out = []
    for i, f in enumerate(fr):
        for j, (u, v) in enumerate(pairs):
            xb, yb, xp, yp = feat(Xb, u, f), feat(Xb, v, f), feat(Xp, u, f), feat(Xp, v, f)
            lo_x, hi_x = np.percentile(np.r_[xb, xp], [0.05, 99.95]); lo_y, hi_y = np.percentile(np.r_[yb, yp], [0.05, 99.95])
            ext = (lo_x, hi_x + 1e-6, lo_y, hi_y + 1e-6)
            axx = ax[i, j]
            axx.hexbin(xb, yb, gridsize=45, extent=ext, mincnt=1, bins="log", cmap=BG_CMAP, linewidths=0)
            axx.scatter(xp, yp, s=8, c=PT, alpha=0.55, linewidths=0.3, edgecolors="white", zorder=3)
            cov = coverage(xb, yb, xp, yp, 45, ext)
            rows_out.append((f, f"{u}-{v}", cov))
            axx.set_title(f"{u} vs {v} at {f:g} × peak — {cov:.0%} of extremes in seen cells", fontsize=9.5, color=INK)
            axx.set_xlabel(LABEL[u], fontsize=8.5, color=INK2); axx.set_ylabel(LABEL[v], fontsize=8.5, color=INK2)
            axx.tick_params(labelsize=7.5, colors=INK2)
            for s in ("top", "right"):
                axx.spines[s].set_visible(False)
    # bottom row: drift with intensity at f = 0.3, all training hours >= 5 mm as background
    lp_all = np.log10(pk_all)
    lp_x = np.log10(te.mx.values)
    for j, name in enumerate(NAMES):
        axx = ax[3, j]
        yb, yp = feat(X_all, name, 0.3), feat(Xp, name, 0.3)
        lo_y, hi_y = np.percentile(np.r_[yb, yp], [0.05, 99.95])
        axx.hexbin(lp_all, yb, gridsize=45, extent=(np.log10(5), lp_x.max() + 0.02, lo_y, hi_y + 1e-6), mincnt=1,
                   bins="log", cmap=BG_CMAP, linewidths=0)
        axx.scatter(lp_x, yp, s=8, c=PT, alpha=0.55, linewidths=0.3, edgecolors="white", zorder=3)
        axx.axvline(np.log10(a.u_c), color=INK2, lw=1, ls="--")
        axx.text(np.log10(a.u_c), hi_y, f" cut {a.u_c:g} mm", fontsize=8, color=INK2, va="top")
        axx.set_title(f"{name} at 0.3 × peak vs peak (all training ≥ 5 mm)", fontsize=9.5, color=INK)
        axx.set_xlabel("log10(target peak / mm)", fontsize=8.5, color=INK2); axx.set_ylabel(LABEL[name], fontsize=8.5, color=INK2)
        axx.tick_params(labelsize=7.5, colors=INK2)
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
    fig.suptitle(f"Held-out extremes (test, peak ≥ {a.u_c:g} mm, n={len(Xp)}, orange) against the shapes the "
                 f"cut-trained models see (train, 5 ≤ peak < {a.u_c:g} mm, n={len(Xb)}, blue density, log count)",
                 fontsize=10.5, color=INK)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    pd.DataFrame(rows_out, columns=["fraction", "pair", "coverage_ge5"]).to_csv(a.out.rsplit(".", 1)[0] + ".csv",
                                                                               index=False)
    for r in rows_out:
        print(f"f={r[0]:g} {r[1]}: {r[2]:.1%} of extremes in hexagons with >= 5 cut-training hours")
