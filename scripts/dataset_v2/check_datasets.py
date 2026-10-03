#!/usr/bin/env python
"""End-to-end check of the v2 dataset family through the training dataset class.

For every metadata file (full, light, extremes, events) of every split: the dataset opens,
subset rows return exactly the full-store rows they point at, the DEM channel equals the true
ground under the patch, values are finite, and a spawn-mode DataLoader produces a batch.
Also re-checks leakage on the metadata itself: no timestamp in two splits, and the minimum
day gap between train and val/test.

    python scripts/dataset_v2/check_datasets.py configs/config_v2.yaml
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    from torch.utils.data import DataLoader
    from src.utils import load_config, load_scaler_val
    from src.data.datasets import DeterministicSRDataset
    from src.data import geo
    cfg = load_config(sys.argv[1])
    d = json.load(open(cfg["DEM_STATS"])); st = (d["dem_mean"], d["dem_std"])
    sv = load_scaler_val(cfg); P = cfg["PATCH_SIZE"]
    dem = geo.load_dem_on_radar_grid(cfg["STATIC_DEM_PATH"])
    meta_dir = os.path.dirname(cfg["TRAIN_METADATA_FILE"])
    ok_all = True
    # a split other than train, so no augmentation interferes with the comparison
    group_of = {"val": "validation", "test": "test", "train": "train", "nimbus": "nimbus"}
    for split in ("val", "test", "nimbus", "train"):
        full_f = os.path.join(meta_dir, f"full_{split}.txt")
        if not os.path.exists(full_f):
            print(f"  full_{split}: absent"); continue
        kw = dict(split=group_of[split], topology_mode="euler")
        full = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], full_f, st, sv, **kw)
        full.is_train = False                         # compare without random flips
        rng = np.random.default_rng(0)
        for name in [f"full_{split}", f"light_{split}", f"extremes_{split}"] + (
                [f"events_{split}"] if split in ("test", "nimbus") else []):
            f = os.path.join(meta_dir, f"{name}.txt")
            ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], f, st, sv, **kw)
            ds.is_train = False
            if not len(ds):
                print(f"  {name:16s} empty"); continue
            ks = rng.choice(len(ds), size=min(60, len(ds)), replace=False)
            same = dem_ok = finite = 0
            for k in ks:
                x, y, g = ds[int(k)]
                r = int(ds._store_rows[k])
                xf, yf, gf = full[r]
                same += torch.equal(x, xf) and torch.equal(y, yf) and torch.equal(g, gf)
                _, yy, xx, _ = ds.metadata[int(k)]
                exp = np.clip((dem[yy:yy + P, xx:xx + P] - st[0]) / (st[1] + 1e-8), -3, 3)
                dem_ok += np.allclose(x[1].numpy(), exp, atol=1e-5)
                finite += bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(g).all())
            n = len(ks)
            good = same == n and dem_ok == n and finite == n
            ok_all &= good
            print(f"  {name:16s} {len(ds):>10,} rows | subset==store {same}/{n} | DEM ok {dem_ok}/{n} | "
                  f"finite {finite}/{n} | {'OK' if good else 'FAIL'}", flush=True)
    X, Y, G = next(iter(DataLoader(full, batch_size=64, shuffle=True, num_workers=2,
                                   multiprocessing_context="spawn")))
    print(f"  DataLoader: X {tuple(X.shape)} Y {tuple(Y.shape)} G {tuple(G.shape)}", flush=True)

    # leakage on the metadata
    days = {}
    for split in ("train", "val", "test", "nimbus"):
        import pandas as pd
        if not os.path.exists(os.path.join(meta_dir, f"full_{split}.txt")):
            days[split] = np.array([], dtype="datetime64[D]"); continue
        ts = pd.read_csv(os.path.join(meta_dir, f"full_{split}.txt"), header=None, usecols=[0],
                         dtype=str)[0] if os.path.getsize(os.path.join(meta_dir, f"full_{split}.txt")) else pd.Series([], dtype=str)
        # parse explicitly: np.datetime64("20210712") would read the whole string as a year
        days[split] = np.unique(pd.to_datetime(ts.str[:8], format="%Y%m%d").values.astype("datetime64[D]"))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test"),
                 ("train", "nimbus"), ("val", "nimbus"), ("test", "nimbus")):
        inter = np.intersect1d(days[a], days[b])
        gap = int(np.min(np.abs(days[a][:, None] - days[b][None, :])).astype(int)) if len(days[a]) and len(days[b]) else -1
        print(f"  leak {a}/{b}: shared days {len(inter)} | min gap {gap} d")
        ok_all &= len(inter) == 0 and gap >= 2
    print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
