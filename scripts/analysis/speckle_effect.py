#!/usr/bin/env python
"""How much of the low-threshold Minkowski target is the pre-December-2015 speckle floor?
(DECISIONS §34, data level.)

Train rows (owned list) before and after the 2015-12-12 break, matched by tile mean rain (5 bins).
Per row: area A, perimeter P and Euler characteristic chi at the three lowest thresholds, the
number of single-pixel components >= 0.1 mm, and the share of wet pixels below 0.3 mm. Pre-break
rows are also scored after zeroing values below 0.2 and below 0.3 mm, to see whether a noise
floor closes the gap to post-break rows.

    python scripts/analysis/speckle_effect.py [--per_bin 300]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

D = "/home/fquareng/work/data/extremes/OPERA"
STORE = f"{D}/patches_v4/preprocessed_dataset.zarr"
META = f"{D}/patches_v4/meta"
OUT = f"{REPO}/notes/data_analysis_v4"
BINS = [0.0, 0.005, 0.02, 0.1, 0.5, np.inf]
LAB = ["<0.005", "0.005-0.02", "0.02-0.1", "0.1-0.5", ">=0.5"]


def metrics(f, thr):
    from scipy import ndimage
    from src.data.gamma import compute_gamma_matrix
    g = compute_gamma_matrix(f, thr, 2.0, 0.0, 0.0)
    lab, n = ndimage.label(f >= 0.1)
    single = int((np.bincount(lab.ravel())[1:] == 1).sum()) if n else 0
    wet = f >= 0.1
    return {**{f"chi{k}": g[4][k] for k in range(3)}, **{f"A{k}": g[0][k] for k in range(3)},
            **{f"P{k}": g[1][k] for k in range(3)}, "singletons": single,
            "low_share": float(((f >= 0.1) & (f < 0.3)).sum() / max(wet.sum(), 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per_bin", type=int, default=300)
    a = ap.parse_args()
    import yaml
    import zarr
    thr = np.array(yaml.safe_load(open(f"{REPO}/configs/config_v4.yaml"))["PHYSICAL_THRESHOLDS"], np.float32)
    info = pd.read_csv(f"{META}/full_train_info.csv.gz", usecols=["store_row", "timestamp", "mean"], dtype={"timestamp": str})
    owned = pd.read_csv(f"{META}/full_train_owned.txt", header=None, usecols=[4])[4].values
    info = info[info.store_row.isin(owned)]
    info["era"] = np.where(info.timestamp.str[:8] < "20151212", "pre", "post")
    info["bin"] = pd.cut(info["mean"], BINS, labels=LAB, right=False)
    rng = np.random.default_rng(0)
    z = zarr.open(STORE, mode="r")["train"]["original_precip"]
    rows = []
    for (era, b), g in info.groupby(["era", "bin"], observed=True):
        pick = g.sample(min(a.per_bin, len(g)), random_state=int(rng.integers(1 << 31)))
        for r in pick.itertuples():
            f = np.nan_to_num(z[int(r.store_row)], nan=0.0)
            rows.append({"era": era, "bin": b, "variant": era, **metrics(f, thr)})
            if era == "pre":
                for s in (0.2, 0.3):
                    rows.append({"era": era, "bin": b, "variant": f"pre, < {s} zeroed",
                                 **metrics(np.where(f < s, 0.0, f), thr)})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/speckle_effect_rows.csv.gz", index=False)
    from scripts.data_quality._md import md_table
    t = df.groupby(["bin", "variant"], observed=True).median(numeric_only=True)[
        ["chi0", "chi1", "chi2", "A0", "P0", "singletons", "low_share"]].reset_index()
    t.columns = ["tile mean (mm)", "rows", "chi @0.1", "chi @0.19", "chi @0.31", "A @0.1 (km²)", "P @0.1 (km)",
                 "single-pixel components", "share of wet px < 0.3 mm"]
    n = df.groupby(["bin", "variant"], observed=True).size().values
    t.insert(2, "n", n)
    L = ["# Pre-December-2015 speckle floor: effect on the low-threshold Minkowski targets (DECISIONS §34)", "",
         f"Train rows (owned list), up to {a.per_bin} per era and tile-mean bin; medians. 'pre' = hour end before "
         "2015-12-12. Thresholds 0.1 / 0.192 / 0.314 mm.", "", md_table(t, index=False, floatfmt=".3g")]
    txt = "\n".join(L)
    open(f"{OUT}/speckle_effect.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
