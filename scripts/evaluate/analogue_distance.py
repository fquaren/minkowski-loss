#!/usr/bin/env python
"""
How unseen is each held-out case? Distance to its nearest training analogue (DECISIONS §43, todo C4;
RESEARCH_NOTES §3 Q5, option O7).

Space: the coarse input the model sees, `coarse_precip` (the native 10 x 10 field at 25 km), as
log1p(mm) flattened to 100 dimensions. It is available at inference and does not depend on any model.
Euclidean distance; nearest neighbour by sklearn's KD tree.

Reference sets, one per trained model family:
    full   train rows with target max >= --ref_min            (the full-data models)
    cut    the same, restricted to target max <  --u_c          (the cut models of R4)
Train rows below --ref_min (dry or near-dry tile-hours) cannot be the analogue of a heavy-rain case
and are skipped to keep the reads affordable.

Output: <out>/analogue_<split>.csv.gz with store_row, timestamp, row, col, target max, coarse max and
the distance to the nearest analogue in each reference set (+ that analogue's store row).

Usage: python scripts/evaluate/analogue_distance.py --split test --min 20.2 --u_c 50.1 --out <dir>
"""

import argparse
import os

import numpy as np
import pandas as pd
import zarr
from sklearn.neighbors import KDTree

V4 = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv"
GROUP = {"train": "train", "val": "validation", "test": "test", "nimbus": "nimbus"}


def features(split, rows):
    a = zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")[GROUP[split]]["coarse_precip"]
    return np.log1p(np.stack([np.nan_to_num(a[int(r)]) for r in rows]).reshape(len(rows), -1)).astype(np.float32)


def meta(split):
    return pd.read_csv(f"{V4}/meta/full_{split}.txt", header=None, names=["t", "y", "x", "mx", "r"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(GROUP))
    ap.add_argument("--min", type=float, default=20.2, help="query cases: target max >= this")
    ap.add_argument("--ref_min", type=float, default=5.0, help="reference rows: target max >= this")
    ap.add_argument("--u_c", type=float, default=50.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = meta("train")
    tr = tr[tr.mx >= a.ref_min].reset_index(drop=True)
    Xtr = features("train", tr.r.values)
    q = meta(a.split)
    q = q[q.mx >= a.min].reset_index(drop=True)
    Xq = features(a.split, q.r.values)
    q["coarse_max"] = np.expm1(Xq.max(axis=1))
    for name, keep in (("full", np.ones(len(tr), bool)), ("cut", (tr.mx < a.u_c).values)):
        tree = KDTree(Xtr[keep])
        d, i = tree.query(Xq, k=1)
        q[f"dist_{name}"], q[f"analogue_row_{name}"] = d[:, 0], tr.r.values[keep][i[:, 0]]
        print(f"{name}: reference n={keep.sum()}, query n={len(q)}, distance q50/q90 = "
              f"{np.median(d):.3f} / {np.quantile(d, .9):.3f}", flush=True)
    q.rename(columns={"r": "store_row", "mx": "target_max"}).to_csv(
        os.path.join(a.out, f"analogue_{a.split}.csv.gz"), index=False)
    hi = q.target_max >= a.u_c
    print(f"above u_c={a.u_c}: n={hi.sum()}, dist_cut q50 {q.dist_cut[hi].median():.3f} vs dist_full "
          f"{q.dist_full[hi].median():.3f}; below: dist_cut q50 {q.dist_cut[~hi].median():.3f}")
