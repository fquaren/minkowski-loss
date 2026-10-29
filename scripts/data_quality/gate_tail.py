#!/usr/bin/env python
"""§20 gate criterion 3 on the hourly v4 build (DECISIONS §32): is the evaluation reference
stable when the 20 most influential non-event days are dropped?

ODYSSEY test split, unweighted (as eval_extremes.py). Statistics: exceedance rates per valid
pixel-hour at 20.2 / 31.7 / 50.1 mm; GPD above u = 20.2 (location 0; binned MLE on the 0.01 mm
grid, `pot_threshold.gpd_fit`) and its return levels at lambda = 10 / 100 / 1000 expected
exceedances (`tail_metrics.return_level`). Pass: each statistic recomputed without the 20
non-event days holding the largest share of exceedances >= 20.2 mm lies inside the 95%
day-block bootstrap interval (B = 200) of the full-sample statistic.

    python scripts/data_quality/gate_tail.py extract --workers 2
    python scripts/data_quality/gate_tail.py analyze
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.dirname(__file__)]
import pot_threshold as pt  # noqa: E402

pt.set_version("v4")
OUT = f"{pt.D}/quality_v4/gate_tail"
LEVELS = (20.2, 31.7, 50.1)
U = 20.2
LAMS = (10.0, 100.0, 1000.0)
VMIN = 5.0


def _block(rows):
    import zarr
    z = zarr.open(f"{pt.STORE}/test/original_precip", mode="r")
    nval = np.zeros(len(rows), np.int32)
    vals, pidx = [], []
    for i, r in enumerate(rows):
        a = z[r]
        f = np.isfinite(a)
        nval[i] = f.sum()
        v = a[f & (a >= VMIN)]
        if v.size:
            vals.append(np.rint(v * 100).astype(np.uint16))
            pidx.append(np.full(v.size, r, np.int32))
    cat = (lambda x, d: np.concatenate(x) if x else np.empty(0, d))
    return rows, nval, cat(vals, np.uint16), cat(pidx, np.int32)


def extract(workers):
    os.makedirs(OUT, exist_ok=True)
    info = pd.read_csv(f"{pt.META}/full_test_info.csv.gz")
    assert (info.store_row.values == np.arange(len(info))).all()
    blocks = np.array_split(np.arange(len(info)), max(1, len(info) // 2000))
    t = time.time()
    nval = np.zeros(len(info), np.int32)
    vals, pidx = [], []
    with Pool(workers) as p:
        for k, (rows, nv, v, pi) in enumerate(p.imap(_block, blocks)):
            nval[rows] = nv; vals.append(v); pidx.append(pi)
            if k % 10 == 0:
                print(f"  [extract] {k + 1}/{len(blocks)} blocks, {time.time() - t:.0f} s", flush=True)
    np.savez(f"{OUT}/test_values.npz", nval=nval, vals=np.concatenate(vals), pidx=np.concatenate(pidx))
    print(f"[extract] {len(info):,} rows, {int(nval.sum()):,} valid pixel-hours, {time.time() - t:.0f} s")


def stats(day_w, D):
    """All statistics from per-day aggregates, with day weights day_w (bootstrap multiplicities)."""
    nv = (day_w * D["nval"]).sum()
    out = {f"rate>={l:g}": (day_w * D[f"n{l:g}"]).sum() / nv for l in LEVELS}
    c = np.asarray(D["M"].T @ day_w).ravel()                  # exceedance histogram above U
    xi, sg = pt.gpd_fit(D["xv"] - U, c)
    out["xi"] = xi
    from src.evaluation.tail_metrics import return_level
    rate = c.sum() / nv
    for lam in LAMS:
        out[f"RL(lambda={lam:g})"] = return_level(xi, sg, U, rate, lam / rate) if rate > 0 else np.nan
    return out


def analyze(B=200):
    from scipy.sparse import coo_matrix
    info = pd.read_csv(f"{pt.META}/full_test_info.csv.gz", dtype={"timestamp": str, "event": str})
    z = np.load(f"{OUT}/test_values.npz")
    nval, vals, pidx = z["nval"], z["vals"] / 100.0, z["pidx"]
    ts = pd.to_datetime(info.timestamp, format="%Y%m%d%H%M%S")
    pday = (ts - pd.Timedelta(minutes=1)).dt.strftime("%Y%m%d").values    # processing day
    dcode, days = pd.factorize(pday)
    nd = len(days)
    D = {"nval": np.bincount(dcode, weights=nval, minlength=nd)}
    vday = dcode[pidx]
    for l in LEVELS:
        D[f"n{l:g}"] = np.bincount(vday[vals >= l], minlength=nd).astype(float)
    k = vals > U
    b = np.rint(vals[k] * 100).astype(np.int64)
    ub, inv = np.unique(b, return_inverse=True)
    D["xv"] = ub / 100.0
    D["M"] = coo_matrix((np.ones(k.sum()), (vday[k], inv)), shape=(nd, ub.size)).tocsr()
    full = stats(np.ones(nd), D)
    # the 20 non-event days with the largest share of exceedances >= U
    ev_day = set(pday[info.event.fillna("").values != ""])
    share = pd.Series(D[f"n{U:g}"], index=days)
    drop = share[[d not in ev_day for d in days]].sort_values(ascending=False).head(20)
    pos = {d: i for i, d in enumerate(days)}
    w = np.ones(nd); w[[pos[d] for d in drop.index]] = 0
    dropped = stats(w, D)
    rng = np.random.default_rng(0)
    boot = [stats(rng.multinomial(nd, np.full(nd, 1 / nd)).astype(float), D) for _ in range(B)]
    L = ["# Gate criterion 3 on v4: stability of the evaluation reference (DECISIONS §32)", "",
         f"ODYSSEY test split: {len(info):,} tile-hours on {nd} days, {int(nval.sum()):,} valid pixel-hours; "
         f"{int(D[f'n{U:g}'].sum()):,} pixel-hours >= {U} mm.",
         f"Dropped: the 20 non-event days with the most exceedances >= {U} mm, holding "
         f"{drop.sum() / D[f'n{U:g}'].sum():.1%} of them: {', '.join(drop.index)}.", "",
         f"| statistic | full | without the 20 days | 95% day-block interval (B = {B}) | inside |",
         "|---|---|---|---|---|"]
    ok_all = True
    for key in full:
        bs = np.array([x[key] for x in boot], float)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        ok = lo <= dropped[key] <= hi
        ok_all &= bool(ok)
        L.append(f"| {key} | {full[key]:.4g} | {dropped[key]:.4g} | [{lo:.4g}, {hi:.4g}] | {'yes' if ok else '**no**'} |")
    L += ["", f"**Criterion 3: {'PASS' if ok_all else 'FAIL'}**"]
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["extract", "analyze"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    extract(a.workers) if a.cmd == "extract" else analyze(a.B)


if __name__ == "__main__":
    main()
