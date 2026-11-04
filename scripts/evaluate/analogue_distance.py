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

--space shape (DECISIONS §43, 2026-11-04): the target's peak-normalised Minkowski features instead
(`src.evaluation.analogue.shape_features`: area, perimeter, Euler at fractions of the patch peak), each feature
standardised with the reference rows' mean and sd. Scale-invariant, so it grades how unseen a storm's *shape* is,
which the coarse-input space cannot do for R4 (cut and full analogue distances were identical there). Evaluation
only: the target is not available at inference. Train features are cached in <out>/shape_train_ref<ref_min>.npz.

Usage: python scripts/evaluate/analogue_distance.py --split test --min 20.2 --u_c 50.1 --out <dir> [--space shape]
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


def shape(split, rows):
    from src.evaluation.analogue import shape_features
    a = zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")[GROUP[split]]["original_precip"]
    return np.stack([shape_features(a[int(r)]) for r in rows])


def meta(split):
    return pd.read_csv(f"{V4}/meta/full_{split}.txt", header=None, names=["t", "y", "x", "mx", "r"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(GROUP))
    ap.add_argument("--min", type=float, default=20.2, help="query cases: target max >= this")
    ap.add_argument("--ref_min", type=float, default=5.0, help="reference rows: target max >= this")
    ap.add_argument("--u_c", type=float, default=50.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--space", default="coarse", choices=["coarse", "shape"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = meta("train")
    tr = tr[tr.mx >= a.ref_min].reset_index(drop=True)
    q = meta(a.split)
    q = q[q.mx >= a.min].reset_index(drop=True)
    if a.space == "coarse":
        Xtr = features("train", tr.r.values)
        Xq = features(a.split, q.r.values)
        q["coarse_max"] = np.expm1(Xq.max(axis=1))
    else:
        cache = os.path.join(a.out, f"shape_train_ref{a.ref_min:g}.npz")
        if os.path.exists(cache) and np.array_equal(np.load(cache)["rows"], tr.r.values):
            Xtr = np.load(cache)["X"]
        else:
            Xtr = shape("train", tr.r.values)
            np.savez_compressed(cache, X=Xtr, rows=tr.r.values)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xq = (Xtr - mu) / sd, (shape(a.split, q.r.values) - mu) / sd
    rand = np.ones(len(tr), bool)        # control: as many rows as the cut removes, dropped at random (DECISIONS §43)
    rand[np.random.default_rng(43).choice(len(tr), int((tr.mx >= a.u_c).sum()), replace=False)] = False
    for name, keep in (("full", np.ones(len(tr), bool)), ("cut", (tr.mx < a.u_c).values), ("rand", rand)):
        tree = KDTree(Xtr[keep])
        d, i = tree.query(Xq, k=1)
        q[f"dist_{name}"], q[f"analogue_row_{name}"] = d[:, 0], tr.r.values[keep][i[:, 0]]
        print(f"{name}: reference n={keep.sum()}, query n={len(q)}, distance q50/q90 = "
              f"{np.median(d):.3f} / {np.quantile(d, .9):.3f}", flush=True)
    q["analogue_mx_full"] = tr.set_index("r").mx.reindex(q.analogue_row_full.values).values
    q = q.rename(columns={"r": "store_row", "mx": "target_max"})
    q.to_csv(os.path.join(a.out, f"analogue_{a.split}{'_shape' if a.space == 'shape' else ''}.csv.gz"), index=False)
    base = float((tr.mx >= a.u_c).mean())
    days = q.t.astype(str).str[:8].values
    rng = np.random.default_rng(0)
    print(f"H1 check ({a.space} space; DECISIONS §43), base rate of >= u_c rows in the pool {base:.1%}")
    for lab, sel in ((f">= {a.u_c}", q.target_max >= a.u_c), (f"[{a.min}, {a.u_c}) control", q.target_max < a.u_c)):
        g = q[sel.values]
        r_cut = (g.dist_cut / g.dist_full.clip(lower=1e-9)).values
        r_rand = (g.dist_rand / g.dist_full.clip(lower=1e-9)).values
        d = days[sel.values]
        ud = np.unique(d); idx = {k: np.flatnonzero(d == k) for k in ud}
        bs = []
        for _ in range(1000):
            j = np.concatenate([idx[k] for k in rng.choice(ud, len(ud))])
            bs.append(np.median(r_cut[j]) - np.median(r_rand[j]))
        lo, hi_ = np.percentile(bs, [2.5, 97.5])
        excess = np.median(r_cut) - np.median(r_rand)
        share = float((g.analogue_mx_full >= a.u_c).mean())
        if hi_ <= 0.05 and share <= 2 * base:
            verdict = "SUPPORTED"
        elif lo > 0.05 or share > 5 * base:
            verdict = "NOT SUPPORTED"
        else:
            verdict = "INCONCLUSIVE"
        print(f"  {lab}: n={len(g)}, days={len(ud)}; median d_cut/d_full {np.median(r_cut):.3f}, d_rand/d_full "
              f"{np.median(r_rand):.3f}, excess {excess:+.3f} [{lo:+.3f}, {hi_:+.3f}]; analogue >= u_c {share:.1%} "
              f"({share / base:.1f}x base) -> {verdict if '>=' in lab else '(control)'}")
