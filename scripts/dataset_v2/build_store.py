#!/usr/bin/env python
"""Build the v2 patch store from the `make_splits.py` metadata.

For every row of `full_{train,val,test}.txt`, in order, writes the cleaned target
(`original_precip`), the coarse input (`coarse_precip`) and its nearest-neighbour upsampling
(`interpolated_precip`) into group `train` / `validation` / `test` of
`<out_dir>/preprocessed_dataset.zarr`, row i = metadata line i. The layout and dtypes are the
v1 ones minus `dem` (looked up by (y, x); src/data/geo.py) and the empty `quality_map`, so
the existing datasets and `compute_gamma_targets.py` read it unchanged.

The target is exactly the field the tile scan described: each frame is cleaned once with
`src.data.day_cleaner.DayCleaner` (clean_frame + the v3 repairs, or clean_frame only with
`--no_repair`) with the same per-year clutter and ring masks, and then cut. `--stage verify` checks
this: the stored tile max must equal the metadata max for every sampled row.

Work is grouped by day, so each 15-minute frame is read and cleaned once. Resumable: finished
days are listed in `<store>/.build_progress.json`, written only after a day's rows are all in.

    python scripts/dataset_v2/build_store.py config.yaml --meta_dir .../OPERA/v2 \
        --out_dir .../OPERA/patches_v2 --climatology .../quality_v2/clutter_climatology.npz \
        --stage all --workers 6
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

GROUPS = {"train": "full_train.txt", "validation": "full_val.txt", "test": "full_test.txt",
          "nimbus": "full_nimbus.txt"}   # NIMBUS product-shift split (2024-07-05 on)
_HOT = {}
_CLIM = None
_RING = None
_SITES = None


def _ring(path, year):
    global _RING
    if path is None:
        return None
    if _RING is None:
        _RING = dict(np.load(path))
    return _RING.get(f"y{year}")


def _sites():
    global _SITES
    if _SITES is None:
        from src.data import geo
        s = geo.load_radar_sites()
        _SITES = np.c_[s["row"].values, s["col"].values].astype(float)
    return _SITES


def read_meta(path):
    ts, ys, xs, mx = [], [], [], []
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 4:
                ts.append(p[0]); ys.append(int(p[1])); xs.append(int(p[2])); mx.append(float(p[3]))
    return np.array(ts), np.array(ys), np.array(xs), np.array(mx)


def store_path(out_dir):
    return os.path.join(out_dir, "preprocessed_dataset.zarr")


def _hot(clim_path, year):
    global _CLIM
    from src.data.cleaning import hot_mask_from_climatology
    if _CLIM is None:
        _CLIM = dict(np.load(clim_path))
    if year not in _HOT:
        _HOT[year] = hot_mask_from_climatology(_CLIM, year)
    return _HOT[year]


def create_store(out_dir, meta_dir, patch, coarse):
    import numcodecs
    import zarr
    root = zarr.open(store_path(out_dir), mode="a")
    comp = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    for g, f in GROUPS.items():
        n = len(read_meta(os.path.join(meta_dir, f))[0])
        grp = root.require_group(g)
        for name, shape in (("original_precip", (n, patch, patch)),
                            ("interpolated_precip", (n, patch, patch)),
                            ("coarse_precip", (n, coarse, coarse))):
            if name in grp:
                assert grp[name].shape == shape, f"{g}/{name} exists with shape {grp[name].shape}"
                continue
            grp.create_dataset(name, shape=shape, chunks=(1,) + shape[1:], dtype="float32",
                               compressor=comp, fill_value=np.nan)
        print(f"  [store] {g}: {n:,} rows", flush=True)


def build_day(args_):
    """Write every row of `rows` (all on one day) of group `g`. Returns (g, day, n, max)."""
    g, day, rows, ts, ys, xs, raw_dir, var, clim, out_dir, factor, patch, ring, repair = args_
    import torch
    import zarr
    from src.data.day_cleaner import DayCleaner
    from src.data.preprocessing import coarsen_and_interpolate
    torch.set_num_threads(1)
    dc = DayCleaner(raw_dir, day, var, hot=lambda y: _hot(clim, y), ring=lambda y: _ring(ring, y),
                    sites_rc=_sites(), repair=repair)
    stamps = {(np.datetime_as_string(t, unit="s").replace("-", "").replace("T", "")
               .replace(":", "")): k for k, t in enumerate(dc.times)}
    grp = zarr.open(store_path(out_dir), mode="r+")[g]
    order = np.argsort(ts[rows], kind="stable")
    cur_t, clean, day_max = None, None, 0.0
    for i in rows[order]:
        if ts[i] != cur_t:
            cur_t = ts[i]
            k = stamps[cur_t]
            clean = dc.clean(k)[0]
        tile = clean[ys[i]:ys[i] + patch, xs[i]:xs[i] + patch]
        tile = np.nan_to_num(tile, nan=0.0)
        c, interp = coarsen_and_interpolate(tile, factor)
        grp["original_precip"][i] = tile
        grp["interpolated_precip"][i] = interp
        grp["coarse_precip"][i] = c
        day_max = max(day_max, float(tile.max()))
    dc.close()
    return g, day, len(rows), day_max


def stage_build(cfg, a):
    prog_p = os.path.join(store_path(a.out_dir), ".build_progress.json")
    prog = json.load(open(prog_p)) if os.path.exists(prog_p) else {}
    for g, f in GROUPS.items():
        ts, ys, xs, _ = read_meta(os.path.join(a.meta_dir, f))
        days = np.array([t[:8] for t in ts])
        done = set(prog.get(g, {}).keys())
        todo = [d for d in np.unique(days) if d not in done]
        print(f"  [build] {g}: {len(ts):,} rows over {len(np.unique(days))} days, "
              f"{len(todo)} days to go", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(build_day, (g, d, np.nonzero(days == d)[0], ts, ys, xs,
                                          cfg["RAW_OPERA_DATA_DIR"], cfg["PRECIP_VAR_NAME"],
                                          a.climatology, a.out_dir, cfg["DOWNSCALING_FACTOR"],
                                          cfg["PATCH_SIZE"], a.ring_mask, not a.no_repair))
                    for d in todo]
            for k, fu in enumerate(as_completed(futs), 1):
                gg, d, n, mx = fu.result()
                prog.setdefault(gg, {})[d] = mx
                if k % 20 == 0 or k == len(futs):
                    tmp = prog_p + ".tmp"
                    json.dump(prog, open(tmp, "w")); os.replace(tmp, prog_p)
                    el = time.time() - t0
                    print(f"    {g}: {k}/{len(futs)} days, {el / k:.1f} s/day, "
                          f"~{el / k * (len(futs) - k) / 3600:.1f} h left", flush=True)
        tmp = prog_p + ".tmp"
        json.dump(prog, open(tmp, "w")); os.replace(tmp, prog_p)


def stage_verify(cfg, a, n_sample=400, seed=0):
    """Stored tile max == metadata max; coarse/interp consistent; no unwritten rows."""
    import zarr
    from src.data.preprocessing import coarsen_and_interpolate
    root = zarr.open(store_path(a.out_dir), "r")
    rng = np.random.default_rng(seed)
    ok = True
    for g, f in GROUPS.items():
        ts, ys, xs, mx = read_meta(os.path.join(a.meta_dir, f))
        if not len(ts):
            print(f"  [verify] {g}: empty", flush=True)
            continue
        idx = np.unique(np.r_[0, len(ts) - 1, rng.choice(len(ts), min(n_sample, len(ts)), replace=False)])
        bad_max = bad_ci = unwritten = 0
        for i in idx:
            t = root[g]["original_precip"][int(i)]
            if np.isnan(t).any():
                unwritten += 1; continue
            bad_max += abs(float(t.max()) - mx[i]) > 1e-3 + 1e-4 * mx[i]
            c, it = coarsen_and_interpolate(t, cfg["DOWNSCALING_FACTOR"])
            bad_ci += not (np.allclose(c, root[g]["coarse_precip"][int(i)], atol=1e-5)
                           and np.allclose(it, root[g]["interpolated_precip"][int(i)], atol=1e-5))
        print(f"  [verify] {g}: sampled {len(idx)} | unwritten {unwritten} | max != metadata "
              f"{bad_max} | coarse/interp mismatch {bad_ci}", flush=True)
        ok &= unwritten == 0 and bad_max == 0 and bad_ci == 0
    return ok


def stage_aux(cfg, a):
    """Scaler, DEM stats, thresholds files the loaders expect, next to the store."""
    from src.data.preprocessing import compute_dem_stats
    prog = json.load(open(os.path.join(store_path(a.out_dir), ".build_progress.json")))
    tmax = max(prog["train"].values())
    np.save(os.path.join(a.out_dir, "log_precip_max_val.npy"), np.array([np.log1p(tmax)]))
    print(f"  [aux] train max {tmax:.2f} mm/h -> log1p {np.log1p(tmax):.4f}", flush=True)
    compute_dem_stats(None, os.path.join(a.out_dir, "dem_stats.json"),
                      dem_path=cfg["STATIC_DEM_PATH"],
                      metadata_file=os.path.join(a.meta_dir, "full_train.txt"),
                      patch_size=cfg["PATCH_SIZE"])
    old = cfg["PREPROCESSED_DATA_DIR"]
    thr = np.asarray(cfg["PHYSICAL_THRESHOLDS"], dtype=np.float32)
    np.save(os.path.join(a.out_dir, "physical_thresholds.npy"), thr)
    print(f"  [aux] physical_thresholds.npy from config ({len(thr)} levels)", flush=True)
    src = os.path.join(old, "persistence_thresholds.yaml")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(a.out_dir, "persistence_thresholds.yaml"))
        print("  [aux] persistence_thresholds.yaml copied from v1 (b0 mode only; recompute "
              "with compute_persistence_thresholds.py if b0 is used)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--climatology", required=True)
    ap.add_argument("--ring_mask", default=None, help="ring_mask.npz (ring_climatology.py)")
    ap.add_argument("--no_repair", action="store_true", help="v2 cleaning (clean_frame only)")
    ap.add_argument("--stage", default="all", choices=["create", "build", "verify", "aux", "all"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(a.config)
    os.makedirs(a.out_dir, exist_ok=True)
    P, C = cfg["PATCH_SIZE"], int(cfg["PATCH_SIZE"] / cfg["DOWNSCALING_FACTOR"])
    stages = ["create", "build", "verify", "aux"] if a.stage == "all" else [a.stage]
    for st in stages:
        print(f"=== {st} {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        if st == "create":
            create_store(a.out_dir, a.meta_dir, P, C)
        elif st == "build":
            stage_build(cfg, a)
        elif st == "verify":
            if not stage_verify(cfg, a) and a.stage == "all":
                sys.exit("verify failed; aux files not written")
        elif st == "aux":
            stage_aux(cfg, a)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)


if __name__ == "__main__":
    main()
