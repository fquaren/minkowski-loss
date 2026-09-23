#!/usr/bin/env python
"""Per-tile quality features over the raw OPERA day stores, one CSV per day.

Reads `<raw_dir>/YYYYMMDD/` zarr stores (`TOT_PREC`, and `QIND` when present), cuts every
128x128 tile on the same stride-128 grid `generate_metadata.py` uses, and computes the
features in `src/data/quality.py` on the **raw, unfiltered** field. Rows are keyed by
`timestamp,row,col` exactly as in `*_patches_metadata.txt`, so they join onto the current
splits, and they cover the whole archive rather than only what survived preprocessing.

Output: `<out_dir>/features/YYYYMMDD.csv.gz`. Restartable with `--skip_existing`.

    python scripts/data_quality/compute_tile_features.py config.yaml \
        --out_dir /home/fquareng/work/data/extremes/OPERA/quality --workers 16

    # quick look: five days, every other time step
    python scripts/data_quality/compute_tile_features.py config.yaml \
        --out_dir $TMP/q --days 20230801 20240717 --every 2

Pass `--climatology <clutter_climatology.npz>` (from `clutter_climatology.py`) to add the
static-clutter features `clim_*`; without it they are left out.
"""

import argparse
import datetime as dt
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.quality import tile_features  # noqa: E402
from src.utils import load_config  # noqa: E402

_DEM = None
_CLIM = None


def _load_dem(path, shape):
    import xarray as xr
    with xr.open_dataset(path, engine="rasterio") as ds:
        dem = ds["band_data"].isel(band=0).values.astype(np.float32)
    if dem.shape != shape:
        raise ValueError(f"DEM shape {dem.shape} != radar grid {shape}")
    return dem


def _open_day(path):
    import xarray as xr
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def _stamp(t) -> str:
    return (np.datetime_as_string(t, unit="s")
            .replace("-", "").replace("T", "").replace(":", ""))


def process_day(day_dir, var, dem_path, clim_path, patch, min_cov, every, factor):
    global _DEM, _CLIM
    ds = _open_day(day_dir)
    H, W = ds.sizes["y"], ds.sizes["x"]
    if _DEM is None and dem_path:
        _DEM = _load_dem(dem_path, (H, W))
    if _CLIM is None and clim_path:
        _CLIM = dict(np.load(clim_path))
    has_q = "QIND" in ds
    times = ds.time.values
    origins = [(y, x) for y in range(0, H - patch + 1, patch)
               for x in range(0, W - patch + 1, patch)]

    def frame(i):
        if i < 0 or i >= len(times):
            return None, None
        r = ds[var].isel(time=i).values.astype(np.float32)
        q = ds["QIND"].isel(time=i).values.astype(np.float32) if has_q else None
        return r, q

    rows = []
    idx = list(range(0, len(times), every))
    cache = {}
    for i in idx:
        for j in (i - 1, i, i + 1):
            if j not in cache:
                cache[j] = frame(j)
        for k in [k for k in cache if k < i - 1]:
            del cache[k]
        r, q = cache[i]
        rp, _ = cache[i - 1]
        rn, _ = cache[i + 1]
        ts = _stamp(times[i])
        for y, x in origins:
            sl = (slice(y, y + patch), slice(x, x + patch))
            tile = r[sl]
            cov = float(np.isfinite(tile).mean())
            if cov < min_cov:
                continue
            f = tile_features(
                tile,
                dem=_DEM[sl] if _DEM is not None else None,
                q=q[sl] if q is not None else None,
                prev=rp[sl] if rp is not None else None,
                nxt=rn[sl] if rn is not None else None,
                factor=factor,
            )
            if _CLIM is not None and "argmax_y" in f:
                ay, ax = y + f["argmax_y"], x + f["argmax_x"]
                for key, m in _CLIM.items():
                    if key.startswith("freq_"):
                        f[f"clim_{key}_at_max"] = float(m[ay, ax])
                        f[f"clim_{key}_tile_max"] = float(np.nanmax(m[sl]))
            f.update(timestamp=ts, row=y, col=x, has_qind=int(has_q))
            rows.append(f)
    return rows


def _worker(args):
    day_dir, out_path = args[0], args[1]
    t0 = time.time()
    try:
        rows = process_day(*args[:1], *args[2:])
    except Exception as e:  # keep the pool alive; report and move on
        return day_dir, f"ERROR {e!r}"
    df = pd.DataFrame(rows)
    front = ["timestamp", "row", "col"]
    df = df[front + [c for c in df.columns if c not in front]] if len(df) else df
    tmp = out_path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.5g")
    os.replace(tmp, out_path)
    return day_dir, f"{len(df)} tiles in {time.time() - t0:.0f}s"


def select_days(raw_dir, start, end, days):
    dirs = sorted(glob.glob(os.path.join(raw_dir, "[0-9]" * 8)))
    # `.zmetadata` is written last, so its absence means the fetcher is still writing
    # that day (or was interrupted there); never read a half-written store.
    partial = [d for d in dirs if not os.path.exists(os.path.join(d, ".zmetadata"))]
    if partial:
        print(f"skipping {len(partial)} incomplete day stores: "
              f"{', '.join(os.path.basename(d) for d in partial[:5])}", flush=True)
    dirs = [d for d in dirs if d not in partial]
    if days:
        want = set(days)
        dirs = [d for d in dirs if os.path.basename(d) in want]
    if start or end:
        s = start or "00000000"
        e = end or "99999999"
        dirs = [d for d in dirs if s <= os.path.basename(d) <= e]
    return dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--raw_dir", default=None, help="default: RAW_OPERA_DATA_DIR")
    ap.add_argument("--start", default=None, help="YYYYMMDD, inclusive")
    ap.add_argument("--end", default=None, help="YYYYMMDD, inclusive")
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--min_coverage", type=float, default=1.0,
                    help="keep tiles with at least this finite fraction; 1.0 reproduces "
                         "the current patch definition (no NaN)")
    ap.add_argument("--every", type=int, default=1, help="use every n-th time step")
    ap.add_argument("--climatology", default=None, help="npz from clutter_climatology.py")
    ap.add_argument("--no_dem", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["RAW_OPERA_DATA_DIR"]
    out = os.path.join(args.out_dir, "features")
    os.makedirs(out, exist_ok=True)

    jobs = []
    for d in select_days(raw_dir, args.start, args.end, args.days):
        op = os.path.join(out, os.path.basename(d) + ".csv.gz")
        if args.skip_existing and os.path.exists(op):
            continue
        jobs.append((d, op, cfg["PRECIP_VAR_NAME"],
                     None if args.no_dem else cfg["STATIC_DEM_PATH"], args.climatology,
                     cfg["PATCH_SIZE"], args.min_coverage, args.every,
                     cfg["DOWNSCALING_FACTOR"]))
    print(f"{dt.datetime.now():%H:%M:%S} {len(jobs)} days to process -> {out}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            d, msg = f.result()
            print(f"[{n}/{len(jobs)}] {os.path.basename(d)}: {msg}", flush=True)


if __name__ == "__main__":
    main()
