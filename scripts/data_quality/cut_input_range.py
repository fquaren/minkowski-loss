#!/usr/bin/env python
"""
How out of range are the *inputs* of the held-out extremes? (DECISIONS §43, todo A3)

For a cut at u_c (target tile maximum, hourly mm), compares the coarse-input maximum (max of
`interpolated_precip`, the upsampled 25 km field the model sees) of
    test / nimbus tile-hours with target max >= u_c           (the R4 / R5 cases)
against
    train tile-hours with target max in [lo, u_c)             (the top of what the cut model sees)
    train tile-hours with target max >= u_c                   (what only the full model sees)
and reports the share of held-out inputs above the cut-train input maximum and above its q99.9.
Train rows below `lo` are not read: their coarse maxima are bounded by their target maxima.

Usage: python scripts/data_quality/cut_input_range.py <u_c> <lo> <out_dir>
"""

import os
import sys

import numpy as np
import pandas as pd
import zarr

V4 = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv"
GROUP = {"train": "train", "test": "test", "nimbus": "nimbus"}


def coarse_max(sp, rows):
    g = zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")[GROUP[sp]]
    a, t = g["interpolated_precip"], g["original_precip"]
    cm = np.array([float(np.max(a[int(r)])) for r in rows])
    tm = np.array([float(np.max(t[int(r)])) for r in rows])
    return cm, tm


if __name__ == "__main__":
    u, lo, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    res = []
    for sp, sel in (("train", lambda m: m.mx >= lo), ("test", lambda m: m.mx >= u), ("nimbus", lambda m: m.mx >= u)):
        m = pd.read_csv(f"{V4}/meta/full_{sp}.txt", header=None, names=["t", "y", "x", "mx", "r"])
        m = m[sel(m)].copy()
        m["coarse_max"], m["target_max_store"] = coarse_max(sp, m.r.values)
        m["split"] = sp
        res.append(m)
    d = pd.concat(res)
    d.to_csv(os.path.join(out, f"coarse_max_u{u:g}.csv.gz"), index=False)
    cut = d[(d.split == "train") & (d.mx < u)].coarse_max
    full = d[(d.split == "train") & (d.mx >= u)].coarse_max
    ref_max, ref_q = cut.max(), cut.quantile(0.999)
    print(f"target max check (list vs store), max abs diff: {np.abs(d.mx - d.target_max_store).max():.4f}")
    print(f"cut-train inputs (target in [{lo}, {u})): n={len(cut)}, coarse max q50/q99/q99.9/max = "
          f"{cut.median():.1f} / {cut.quantile(.99):.1f} / {ref_q:.1f} / {ref_max:.1f} mm")
    print(f"train above the cut (full model only): n={len(full)}, coarse q50/q99/max = "
          f"{full.median():.1f} / {full.quantile(.99):.1f} / {full.max():.1f} mm")
    for sp in ("test", "nimbus"):
        h = d[d.split == sp]
        print(f"{sp} >= {u}: n={len(h)}, coarse q50/q90/max = {h.coarse_max.median():.1f} / "
              f"{h.coarse_max.quantile(.9):.1f} / {h.coarse_max.max():.1f} mm; above cut-train max "
              f"{(h.coarse_max > ref_max).mean():.1%}, above its q99.9 {(h.coarse_max > ref_q).mean():.1%}; "
              f"target/coarse ratio q50 {np.median(h.mx / h.coarse_max.clip(lower=1e-3)):.1f}")
