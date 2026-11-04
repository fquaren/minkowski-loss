#!/usr/bin/env python
"""
The paired ODYSSEY / NIMBUS set for rung R2: the same tile-hours in both products (DECISIONS §43, todo B1).

The 101 days 2024-07-05 .. 2024-10-30 exist in both products; the ODYSSEY copies are in
`raw/OPERA_orig_postswitch/`. Stages, in order (each resumable):

  meta     ODYSSEY scan (`scan_hours.py --raw_dir ...` -> --hours_dir) screened with the same
           `decide_hourly` as the v4 lists (`make_splits_hourly.load_day`; the radar-frame flags and
           radar-day exclusions are the archive's, computed on NIMBUS for these days), joined with the
           NIMBUS list rows of the period: a pair exists where both products keep the tile-hour.
           -> <work>/meta/full_test.txt (ODYSSEY, 4 columns), empty train/val/nimbus, paired_index.csv
  store    `build_store_hourly.py` on that meta with the ODYSSEY originals: create, build, adv, verify
  target   the corrected target, as `build_adv_store.py`: original_precip -> adv_precip (symlink),
           coarse / interpolated recomputed; rows with an incomplete corrected field dropped.
           -> <out>/meta/full_test.txt (ODYSSEY, 5 columns) and <out>/meta/paired_nimbus.txt (the same
           tile-hours as rows of the patches_v4_adv `nimbus` group, same order, 5 columns).
           Scaler, DEM stats and thresholds are copied from patches_v4_adv, so normalisation is identical.
  gamma    `compute_gamma_targets.py` on <out> (group `test`)

Usage: python scripts/dataset_v2/build_paired_odyssey.py --stage meta|store|target|gamma
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.dirname(__file__)]
D = "/home/fquareng/work/data/extremes/OPERA"
V4 = f"{D}/patches_v4_adv"
HOURS = f"{D}/quality_v4/hourly_odyssey_paired/hours"
WORK = f"{D}/patches_v4_paired_plain"
OUT = f"{D}/patches_v4_paired"
RAW = f"{D}/raw/OPERA_orig_postswitch"
Q2 = f"{D}/quality_v2"
PERIOD = ("20240705", "20241031")
PY = [sys.executable, "-u"]


def stage_meta():
    from make_splits_hourly import _init, load_day
    q = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))
    ff = pd.read_csv(os.path.join(q["data_dir"], "frame_flags.csv.gz"), dtype=str)
    de = pd.read_csv(f"{D}/quality_v4/day_exclusions.csv", dtype=str)
    _init(q, set(zip(ff["radar"], ff["timestamp"])), set(zip(de["radar"], de["day"])))
    files = sorted(glob.glob(os.path.join(HOURS, "[0-9]" * 8 + ".csv.gz")))
    df = pd.concat([p for p in map(load_day, files) if p is not None], ignore_index=True)
    ody = df[~df.rejected][["timestamp", "row", "col", "max"]].rename(columns={"max": "ody_plain_max"})
    ody["timestamp"] = ody.timestamp.astype(str)
    nim = pd.read_csv(f"{V4}/meta/full_nimbus.txt", header=None, names=["timestamp", "row", "col", "nimbus_max", "nimbus_row"],
                      dtype={"timestamp": str})
    nim = nim[(nim.timestamp.str[:8] >= PERIOD[0]) & (nim.timestamp.str[:8] < PERIOD[1])]
    pair = nim.merge(ody, on=["timestamp", "row", "col"], how="inner").sort_values(["timestamp", "row", "col"])
    os.makedirs(f"{WORK}/meta", exist_ok=True)
    pair[["timestamp", "row", "col", "ody_plain_max"]].to_csv(f"{WORK}/meta/full_test.txt", header=False, index=False,
                                                                float_format="%.4f")
    for f in ("full_train.txt", "full_val.txt", "full_nimbus.txt"):
        open(f"{WORK}/meta/{f}", "w").close()
    pair.to_csv(f"{WORK}/meta/paired_index.csv", index=False)
    print(f"[meta] ODYSSEY scan: {len(df):,} tile-hours over {df.timestamp.str[:8].nunique()} days, rejected "
          f"{df.rejected.mean():.2%}; NIMBUS list rows in the period {len(nim):,}; pairs {len(pair):,} "
          f"({len(pair) / max(len(nim), 1):.1%} of the NIMBUS rows)")


def stage_store():
    cfg = yaml.safe_load(open(f"{REPO}/config.yaml"))
    cfg["RAW_OPERA_DATA_DIR"] = RAW
    os.makedirs(WORK, exist_ok=True)
    cpath = f"{WORK}/config_paired_build.yaml"
    yaml.safe_dump(cfg, open(cpath, "w"))
    common = ["--meta_dir", f"{WORK}/meta", "--out_dir", WORK, "--quality", f"{REPO}/configs/quality_v4.yaml",
              "--climatology", f"{Q2}/clutter_climatology.npz", "--ring_mask", f"{Q2}/ring_mask.npz", "--workers", "2"]
    for st, extra in (("create", []), ("build", ["--no_advection"]), ("adv", []), ("verify", [])):
        print(f"[store] {st}", flush=True)
        subprocess.run(PY + [f"{REPO}/scripts/dataset_v2/build_store_hourly.py", cpath, "--stage", st] + common + extra,
                       check=True)


def stage_target():
    import zarr
    from src.data.preprocessing import coarsen_and_interpolate
    src = zarr.open_group(os.path.join(WORK, "preprocessed_dataset.zarr"), mode="r")["test"]
    os.makedirs(f"{OUT}/meta", exist_ok=True)
    G = zarr.open_group(os.path.join(OUT, "preprocessed_dataset.zarr"), mode="a").require_group("test")
    for name, srcname in (("original_precip", "adv_precip"), ("max_rate", "max_rate"), ("n_wet", "n_wet")):
        link = os.path.join(OUT, "preprocessed_dataset.zarr", "test", name)
        if not os.path.islink(link):
            os.symlink(os.path.join(WORK, "preprocessed_dataset.zarr", "test", srcname), link)
    A = src["adv_precip"]
    n = A.shape[0]
    for name, shape in (("coarse_precip", src["coarse_precip"].shape), ("interpolated_precip", A.shape)):
        if name not in G:
            G.create(name, shape=shape, chunks=(1,) + shape[1:], dtype="f4", fill_value=np.nan)
    fin, mx = np.zeros(n, bool), np.full(n, np.nan)
    for k in range(n):
        t = A[k]
        fin[k] = bool(np.isfinite(t).all())
        if fin[k]:
            mx[k] = float(t.max())
            c, it = coarsen_and_interpolate(t, 12.5)
            G["coarse_precip"][k], G["interpolated_precip"][k] = c, it
    pair = pd.read_csv(f"{WORK}/meta/paired_index.csv", dtype={"timestamp": str})
    assert len(pair) == n, (len(pair), n)
    pair["ody_row"], pair["ody_max"] = np.arange(n), mx
    keep = pair[fin]
    keep[["timestamp", "row", "col", "ody_max", "ody_row"]].to_csv(f"{OUT}/meta/full_test.txt", header=False,
                                                                   index=False, float_format="%.4f")
    keep[["timestamp", "row", "col", "nimbus_max", "nimbus_row"]].to_csv(f"{OUT}/meta/paired_nimbus.txt", header=False,
                                                                         index=False, float_format="%.4f")
    keep.to_csv(f"{OUT}/meta/paired_index.csv", index=False)
    for f in ("log_precip_max_val.npy", "dem_stats.json", "physical_thresholds.npy", "persistence_thresholds.yaml"):
        if os.path.exists(f"{V4}/{f}"):
            shutil.copy(f"{V4}/{f}", f"{OUT}/{f}")
    print(f"[target] {n} rows, corrected field complete {fin.sum()} ({fin.mean():.1%}); "
          f"ODYSSEY / NIMBUS tile max, median ratio {np.nanmedian(keep.ody_max / keep.nimbus_max.clip(lower=0.1)):.2f}")


def stage_gamma():
    cfg = yaml.safe_load(open(f"{REPO}/configs/config_v4_adv.yaml"))
    cfg["PREPROCESSED_DATA_DIR"] = OUT
    cfg["TEST_METADATA_FILE"] = f"{OUT}/meta/full_test.txt"
    cpath = f"{OUT}/config_v4_paired.yaml"
    yaml.safe_dump(cfg, open(cpath, "w"))
    subprocess.run(PY + [f"{REPO}/scripts/preprocess/compute_gamma_targets.py", cpath], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["meta", "store", "target", "gamma"])
    {"meta": stage_meta, "store": stage_store, "target": stage_target, "gamma": stage_gamma}[ap.parse_args().stage]()
