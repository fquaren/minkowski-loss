#!/usr/bin/env python
"""The advection-corrected v4 store, `patches_v4_adv` (DECISIONS §35), built from `patches_v4`.

- `original_precip` is a symlink to the source `adv_precip` array; `max_rate` and `n_wet` are
  symlinks to the source arrays (zarr arrays are directories). Store rows keep the source numbering.
- `coarse_precip` / `interpolated_precip` are recomputed from the corrected field for every row.
- Metadata: the §34 owned lists, minus rows whose corrected field has any NaN (no complete
  H-60..H+15 window), with the corrected tile max, 5 columns (timestamp, row, col, max, store_row).
  Subset lists likewise. Flags carried over. Scaler = log1p(corrected train max); DEM stats copied;
  physical_thresholds.npy = the config grid.

    python scripts/dataset_v2/build_adv_store.py --workers 8
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

D = "/home/fquareng/work/data/extremes/OPERA"
SRC = f"{D}/patches_v4"
DST = f"{D}/patches_v4_adv"
GROUPS = {"train": "full_train", "validation": "full_val", "test": "full_test", "nimbus": "full_nimbus"}
SUBSETS = {"train": ["light_train", "extremes_train"], "validation": ["light_val", "extremes_val"],
           "test": ["light_test", "extremes_test", "events_test"],
           "nimbus": ["light_nimbus", "extremes_nimbus", "events_nimbus"]}
FACTOR = 12.5


def zpath(root, *p):
    return os.path.join(root, "preprocessed_dataset.zarr", *p)


def setup():
    import zarr
    os.makedirs(f"{DST}/meta", exist_ok=True)
    root = zarr.open_group(zpath(DST), mode="a")
    src = zarr.open_group(zpath(SRC), mode="r")
    for g in GROUPS:
        G = root.require_group(g)
        for name, srcname in (("original_precip", "adv_precip"), ("max_rate", "max_rate"), ("n_wet", "n_wet")):
            link = zpath(DST, g, name)
            if not os.path.islink(link):
                os.symlink(zpath(SRC, g, srcname), link)
        S = src[g]["coarse_precip"]
        for name, shape in (("coarse_precip", S.shape), ("interpolated_precip", src[g]["original_precip"].shape)):
            if name not in G:
                G.create(name, shape=shape, chunks=(1,) + shape[1:], dtype="f4", fill_value=np.nan)


def _block(args):
    g, start, end = args
    import zarr
    from src.data.preprocessing import coarsen_and_interpolate
    A = zarr.open(zpath(SRC, g, "adv_precip"), mode="r")
    Cz = zarr.open(zpath(DST, g, "coarse_precip"), mode="r+")
    Iz = zarr.open(zpath(DST, g, "interpolated_precip"), mode="r+")
    fin = np.zeros(end - start, bool)
    mx = np.full(end - start, np.nan)
    a = A[start:end]
    cs, its = [], []
    for k in range(end - start):
        t = a[k]
        fin[k] = bool(np.isfinite(t).all())
        if fin[k]:
            mx[k] = float(t.max())
        c, it = coarsen_and_interpolate(np.nan_to_num(t, nan=0.0) if not fin[k] else t, FACTOR)
        if not fin[k]:
            c, it = np.full_like(c, np.nan), np.full_like(it, np.nan)
        cs.append(c); its.append(it)
    Cz[start:end] = np.stack(cs)
    Iz[start:end] = np.stack(its)
    return g, start, fin, mx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--meta_only", action="store_true",
                    help="rewrite the metadata from the saved per-row completeness and max (e.g. after a "
                         "new exclusion in the owned lists), without recomputing the fields")
    a = ap.parse_args()
    import yaml
    import zarr
    t0 = time.time()
    fin, mx = {}, {}
    if a.meta_only:
        for g in GROUPS:
            z = np.load(f"{DST}/meta/adv_rows_{g}.npz")
            fin[g], mx[g] = z["fin"], z["mx"]
    else:
        setup()
    for g in ([] if a.meta_only else GROUPS):
        n = zarr.open(zpath(SRC, g, "adv_precip"), mode="r").shape[0]
        fin[g], mx[g] = np.zeros(n, bool), np.full(n, np.nan)
        jobs = [(g, s, min(s + a.block, n)) for s in range(0, n, a.block)]
        with ProcessPoolExecutor(a.workers) as ex:
            for k, (gg, s, f, m) in enumerate(ex.map(_block, jobs)):
                fin[g][s:s + len(f)], mx[g][s:s + len(m)] = f, m
                if k % 50 == 0:
                    print(f"  [{g}] {k + 1}/{len(jobs)} blocks, {time.time() - t0:.0f} s", flush=True)
        np.savez(f"{DST}/meta/adv_rows_{g}.npz", fin=fin[g], mx=mx[g])
        print(f"[{g}] {n:,} rows, corrected field complete on {fin[g].mean():.1%}", flush=True)
    report = []
    for g, stem in GROUPS.items():
        own = pd.read_csv(f"{SRC}/meta/{stem}_owned.txt", header=None, dtype={0: str})
        sr = own[4].values.astype(int)
        keep = fin[g][sr]
        out = own[keep].copy()
        out[3] = [f"{v:.4f}" for v in mx[g][sr[keep]]]
        out.to_csv(f"{DST}/meta/{stem}.txt", header=False, index=False)
        report.append(f"{g}: owned {len(own):,}, corrected field incomplete {int((~keep).sum()):,} "
                      f"({(~keep).mean():.1%}), kept {int(keep.sum()):,}")
        for sub in SUBSETS[g]:
            p = f"{SRC}/meta/{sub}_owned.txt"
            if not os.path.exists(p):
                continue
            s = pd.read_csv(p, header=None, dtype={0: str})
            k = fin[g][s[4].values.astype(int)]
            s = s[k].copy()
            s[3] = [f"{v:.4f}" for v in mx[g][s[4].values.astype(int)]]
            s.to_csv(f"{DST}/meta/{sub}.txt", header=False, index=False)
            report.append(f"  {sub}: {len(k):,} -> {int(k.sum()):,}")
        fl = pd.read_csv(f"{SRC}/meta/{stem}_flags.csv.gz")
        fl["adv_complete"] = fin[g][fl.store_row.values]
        fl.to_csv(f"{DST}/meta/{stem}_flags.csv.gz", index=False)
    tr = pd.read_csv(f"{DST}/meta/full_train.txt", header=None)
    np.save(f"{DST}/log_precip_max_val.npy", np.float64(np.log1p(tr[3].max())))
    shutil.copy(f"{SRC}/dem_stats.json", f"{DST}/dem_stats.json")
    cfg = yaml.safe_load(open(f"{REPO}/configs/config_v4_adv.yaml"))
    np.save(f"{DST}/physical_thresholds.npy", np.array(cfg["PHYSICAL_THRESHOLDS"], np.float32))
    report.append(f"scaler: log1p(corrected train max {tr[3].max():.3f}); {time.time() - t0:.0f} s")
    txt = "\n".join(report)
    print(txt)
    open(f"{DST}/meta/build_report.txt", "w").write("build_adv_store.py (DECISIONS §35)\n" + txt + "\n")
    print("ADV_STORE_DONE")


if __name__ == "__main__":
    main()
