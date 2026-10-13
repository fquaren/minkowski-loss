#!/usr/bin/env python
"""Build the hourly patch store from the `make_splits_hourly.py` metadata (DECISIONS §21, §25).

For every row of `full_{train,val,test,nimbus}.txt`, in order (row i = metadata line i), in
group `train` / `validation` / `test` / `nimbus` of `<out_dir>/preprocessed_dataset.zarr`:

  original_precip      (P, P) float32  the target: the plain ACRR-convention hourly sum (mm) of
                                       the four cleaned 15-min frames H-45 .. H
  coarse_precip        (C, C) float32  its coarsening (`coarsen_and_interpolate`), as v1/v2
  interpolated_precip  (P, P) float32  nearest-neighbour upsampling of the coarse field
  adv_precip           (P, P) float32  the advection-corrected hour (no-blend nearest-neighbour
                                       warping of the cleaned frames H-60 .. H+15, DECISIONS
                                       §25), stored alongside, not the target; NaN where a frame
                                       of that window is missing
  max_rate             (P, P) float32  largest cleaned 15-min rate of the four frames (mm/h)
  n_wet                (P, P) uint8    how many of the four frames are >= 0.1 mm/h

The fields are exactly what `scan_hours.py` described: the same `HourlyDay` chain with the
quality config's `hourly:` block, the same ceilings table, clutter and ring masks and radar
sites. `--stage verify` checks this (stored tile max == metadata max) and the internal
consistency (sum <= max rate, n_wet in 0..4, coarse / interpolated fields).

Work is grouped by processing day, so each day runs the chain once. The advection-corrected
field is computed only on the stored tiles; it roughly doubles the cost of a day, so
`--no_advection` builds everything else first and `--stage adv` fills `adv_precip` later.
Resumable: finished days are listed in `<store>/.build_progress.json` (key `<group>`, and
`adv:<group>` for adv_precip), written only after all of a day's rows are in.

    python scripts/dataset_v2/build_store_hourly.py config.yaml --meta_dir .../OPERA/v4_hourly \\
        --out_dir .../OPERA/patches_v4_hourly --quality configs/quality_v4.yaml \\
        --climatology .../quality_v2/clutter_climatology.npz --ring_mask .../quality_v2/ring_mask.npz \\
        --stage all --workers 6
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))
from build_store import read_meta, store_path  # noqa: E402

GROUPS = {"train": "full_train.txt", "validation": "full_val.txt", "test": "full_test.txt",
          "nimbus": "full_nimbus.txt"}
# the variant that passed (DECISIONS §25); motion interpolated on every 8th pixel (`motion_lk`):
# on 2021-07-14 12:00 it differs from the full-grid pysteps field by 0.03 mm on average over
# pixels >= 1 mm, the hour's maximum by -0.4%, at 1/35 of the cost
ADVECTION = {"mode": "noblend", "interp": "nearest", "motion_step": 8}
FIELDS = {"original_precip": "float32", "interpolated_precip": "float32", "adv_precip": "float32",
          "max_rate": "float32", "n_wet": "uint8"}


def create_store(out_dir, meta_dir, patch, coarse):
    import numcodecs
    import zarr
    root = zarr.open(store_path(out_dir), mode="a")
    comp = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    for g, f in GROUPS.items():
        n = len(read_meta(os.path.join(meta_dir, f))[0])
        grp = root.require_group(g)
        spec = {**{k: ((n, patch, patch), dt) for k, dt in FIELDS.items()},
                "coarse_precip": ((n, coarse, coarse), "float32")}
        for name, (shape, dt) in spec.items():
            if name in grp:
                assert grp[name].shape == shape, f"{g}/{name} exists with shape {grp[name].shape}"
                continue
            grp.create_dataset(name, shape=shape, chunks=(1,) + shape[1:], dtype=dt, compressor=comp,
                               fill_value=255 if dt == "uint8" else np.nan)
        print(f"  [store] {g}: {n:,} rows", flush=True)


def _ts(t):
    return np.datetime_as_string(t, unit="s").replace("-", "").replace("T", "").replace(":", "")


def _ns(t):
    return int(np.datetime64(f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}:{t[12:14]}", "ns").astype(np.int64))


def build_day(args_):
    """Write the rows `rows` (all of one processing day) of group `g`; mode "all" (every field),
    "main" (all but adv_precip) or "adv" (adv_precip only). Returns (g, day, n, max, mode)."""
    g, day, rows, ts, ys, xs, raw_dir, var, a, factor, patch, mode = args_
    import torch
    import zarr
    import scan_tiles as st
    from src.data import cleaning as C
    from src.data.hourly import HourlyDay
    from src.data.preprocessing import coarsen_and_interpolate
    torch.set_num_threads(1)
    q, ceil = st._quality(a.quality)
    cfg = {**q.get("hourly", {}), "advection": None if mode == "main" else ADVECTION}
    boxes = {}
    for i in rows:
        boxes.setdefault(_ns(ts[i]), []).append((ys[i], ys[i] + patch, xs[i], xs[i] + patch))
    hd = HourlyDay(raw_dir, day, var, hot=lambda y: st._hot(a.climatology, y),
                   ring=lambda y: st._ring(a.ring_mask, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=cfg, adv_boxes=None if mode == "main" else boxes).run()
    hours = {_ts(h["end"]): h for h in hd.hours}
    grp = zarr.open(store_path(a.out_dir), mode="r+")[g]
    day_max = 0.0
    for i in rows:
        h = hours[ts[i]]                       # KeyError = metadata not from this chain
        sl = (slice(ys[i], ys[i] + patch), slice(xs[i], xs[i] + patch))
        tile = np.nan_to_num(h["sum"][sl], nan=0.0)
        day_max = max(day_max, float(tile.max()))
        if mode != "main":
            grp["adv_precip"][i] = (h["sum_adv"][sl] if h["sum_adv"] is not None
                                    else np.full((patch, patch), np.nan, np.float32))
        if mode == "adv":
            continue
        c, interp = coarsen_and_interpolate(tile, factor)
        f4 = np.nan_to_num(hd.fin[h["idx"]][(slice(None),) + sl], nan=0.0)
        grp["original_precip"][i] = tile
        grp["interpolated_precip"][i] = interp
        grp["coarse_precip"][i] = c
        grp["max_rate"][i] = np.nan_to_num(h["max_rate"][sl], nan=0.0)
        grp["n_wet"][i] = (f4 >= C.DRIZZLE).sum(0).astype(np.uint8)
    return g, day, len(rows), day_max, mode


def stage_build(cfg, a, adv_only=False):
    prog_p = os.path.join(store_path(a.out_dir), ".build_progress.json")
    prog = json.load(open(prog_p)) if os.path.exists(prog_p) else {}
    for g, f in GROUPS.items():
        ts, ys, xs, _ = read_meta(os.path.join(a.meta_dir, f))
        # processing day of an hour ending at t: the date of t - 1 min
        days = np.array([(np.datetime64(f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}")
                          - np.timedelta64(1, "m")).astype("datetime64[D]").astype(str).replace("-", "")
                         for t in ts])
        if adv_only:                        # days whose other fields are in, adv_precip not yet
            done, mode = set(prog.get("adv:" + g, {})), "adv"
            todo = [d for d in np.unique(days) if d in prog.get(g, {}) and d not in done]
        else:
            done, mode = set(prog.get(g, {})), "main" if a.no_advection else "all"
            todo = [d for d in np.unique(days) if d not in done]
        print(f"  [build] {g}: {len(ts):,} rows over {len(np.unique(days))} days, {len(todo)} to go",
              flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(build_day, (g, d, np.nonzero(days == d)[0], ts, ys, xs,
                                          cfg["RAW_OPERA_DATA_DIR"], cfg["PRECIP_VAR_NAME"], a,
                                          cfg["DOWNSCALING_FACTOR"], cfg["PATCH_SIZE"], mode)) for d in todo]
            for k, fu in enumerate(as_completed(futs), 1):
                gg, d, n, mx, md = fu.result()
                if md != "adv":
                    prog.setdefault(gg, {})[d] = mx
                if md != "main":
                    prog.setdefault("adv:" + gg, {})[d] = mx
                if k % 20 == 0 or k == len(futs):
                    tmp = prog_p + ".tmp"
                    json.dump(prog, open(tmp, "w")); os.replace(tmp, prog_p)
                    el = time.time() - t0
                    print(f"    {g}: {k}/{len(futs)} days, {el / k:.1f} s/day, "
                          f"~{el / k * (len(futs) - k) / 3600:.1f} h left", flush=True)


def stage_verify(cfg, a, n_sample=400, seed=0):
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
        bad = {"unwritten": 0, "max != metadata": 0, "coarse/interp": 0, "sum > max_rate": 0,
               "n_wet": 0, "adv missing": 0}
        for i in idx:
            G = {k: root[g][k][int(i)] for k in (*FIELDS, "coarse_precip")}
            t = G["original_precip"]
            if np.isnan(t).any() or (G["n_wet"] == 255).any():
                bad["unwritten"] += 1
                continue
            bad["max != metadata"] += abs(float(t.max()) - mx[i]) > 1e-3 + 1e-4 * mx[i]
            c, it = coarsen_and_interpolate(t, cfg["DOWNSCALING_FACTOR"])
            bad["coarse/interp"] += not (np.allclose(c, G["coarse_precip"], atol=1e-5)
                                         and np.allclose(it, G["interpolated_precip"], atol=1e-5))
            bad["sum > max_rate"] += bool((t > G["max_rate"] + 1e-4).any())
            bad["n_wet"] += bool((G["n_wet"] > 4).any() or ((t >= 0.1) & (G["n_wet"] == 0)).any())
            bad["adv missing"] += bool(np.isnan(G["adv_precip"]).all())
        print(f"  [verify] {g}: sampled {len(idx)} | " + " | ".join(f"{k} {v}" for k, v in bad.items()),
              flush=True)
        ok &= all(v == 0 for k, v in bad.items() if k != "adv missing")
    return ok


def stage_aux(cfg, a):
    """Scaler and DEM stats next to the store (as `build_store.py`)."""
    from src.data.preprocessing import compute_dem_stats
    prog = json.load(open(os.path.join(store_path(a.out_dir), ".build_progress.json")))
    tmax = max(prog["train"].values())
    np.save(os.path.join(a.out_dir, "log_precip_max_val.npy"), np.array([np.log1p(tmax)]))
    print(f"  [aux] train max {tmax:.2f} mm -> log1p {np.log1p(tmax):.4f}", flush=True)
    compute_dem_stats(None, os.path.join(a.out_dir, "dem_stats.json"), dem_path=cfg["STATIC_DEM_PATH"],
                      metadata_file=os.path.join(a.meta_dir, "full_train.txt"), patch_size=cfg["PATCH_SIZE"])
    thr = np.asarray(cfg["PHYSICAL_THRESHOLDS"], dtype=np.float32)
    np.save(os.path.join(a.out_dir, "physical_thresholds.npy"), thr)
    print(f"  [aux] physical_thresholds.npy from config ({len(thr)} levels; 15-min levels until "
          f"the hourly thresholds are re-derived)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--quality", default="configs/quality_v4.yaml")
    ap.add_argument("--climatology", required=True)
    ap.add_argument("--ring_mask", required=True)
    ap.add_argument("--no_advection", action="store_true", help="skip adv_precip (left NaN)")
    ap.add_argument("--stage", default="all", choices=["create", "build", "adv", "verify", "aux", "all"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    a.quality = os.path.abspath(a.quality)
    from src.utils import load_config
    cfg = load_config(a.config)
    os.makedirs(a.out_dir, exist_ok=True)
    P, Cs = cfg["PATCH_SIZE"], int(cfg["PATCH_SIZE"] / cfg["DOWNSCALING_FACTOR"])
    for st in (["create", "build", "verify", "aux"] if a.stage == "all" else [a.stage]):
        print(f"=== {st} {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        if st == "create":
            create_store(a.out_dir, a.meta_dir, P, Cs)
        elif st == "build":
            stage_build(cfg, a)
        elif st == "adv":
            stage_build(cfg, a, adv_only=True)
        elif st == "verify":
            if not stage_verify(cfg, a) and a.stage == "all":
                sys.exit("verify failed; aux files not written")
        elif st == "aux":
            stage_aux(cfg, a)
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)


if __name__ == "__main__":
    main()
