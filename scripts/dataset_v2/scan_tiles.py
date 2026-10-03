#!/usr/bin/env python
"""Scan every raw day store: clean each frame and describe every fully covered tile.

One row per (timestamp, tile) where the 128x128 tile on the stride-128 grid has no NaN (the
same validity rule as `generate_metadata.py`). The frame is cleaned first with
`src.data.cleaning.clean_frame` (drizzle floor, static-clutter and spike repair; no zeroing
above 150 mm/h), so every statistic describes the field the v2 store will actually hold.

Columns: timestamp,row,col, cleaned-tile stats (max, mean, wet_frac, n_ge*, coarse_max),
raw_max, n_fixed (pixels changed by the repair), ray (RLAN ray pointing at a radar),
unphysical (cleaned max > 500), q_mean_wet / has_qind.

Output: <out_dir>/tiles/YYYYMMDD.csv.gz. Restartable with --skip_existing; a day is written
through a temporary file, so a partial file never looks complete.

    python scripts/dataset_v2/scan_tiles.py config.yaml \
        --climatology .../quality_v2/clutter_climatology.npz \
        --out_dir .../quality_v2 --workers 6 --skip_existing
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_CLIM = None
_HOT = {}
_SITES = None


def _hot(clim_path, year):
    global _CLIM
    from src.data.cleaning import hot_mask_from_climatology
    if clim_path is None:
        return None
    if _CLIM is None:
        _CLIM = dict(np.load(clim_path))
    if year not in _HOT:
        _HOT[year] = hot_mask_from_climatology(_CLIM, year)
    return _HOT[year]


def _sites():
    global _SITES
    if _SITES is None:
        from src.data import geo
        s = geo.load_radar_sites()
        _SITES = np.c_[s["row"].values, s["col"].values].astype(float)
    return _SITES


def scan_day(day_dir, var, clim_path, patch, out_path):
    import xarray as xr
    from src.data.cleaning import clean_frame, ray_flag, tile_stats, UNPHYSICAL
    try:
        ds = xr.open_zarr(day_dir, consolidated=True)
    except Exception:
        ds = xr.open_zarr(day_dir, consolidated=False)
    day = os.path.basename(day_dir.rstrip("/"))
    hot = _hot(clim_path, int(day[:4]))
    sites = _sites()
    has_q = "QIND" in ds
    rows = []
    for t in range(ds.sizes["time"]):
        raw = ds[var].isel(time=t).values.astype(np.float32)
        H, W = raw.shape
        clean, _ = clean_frame(raw, hot)
        q = ds["QIND"].isel(time=t).values if has_q else None
        ts = (np.datetime_as_string(ds.time.values[t], unit="s")
              .replace("-", "").replace("T", "").replace(":", ""))
        for r0 in range(0, H - patch + 1, patch):
            for c0 in range(0, W - patch + 1, patch):
                rt = raw[r0:r0 + patch, c0:c0 + patch]
                if not np.isfinite(rt).all():
                    continue
                ct = clean[r0:r0 + patch, c0:c0 + patch]
                st = tile_stats(ct)
                zr = np.where(rt < 0.1, 0.0, rt)
                st.update({
                    "timestamp": ts, "row": r0, "col": c0,
                    "raw_max": float(rt.max()),
                    "n_fixed": int((np.abs(ct - zr) > 1e-6).sum()),
                    "ray": int(st["n_ge1"] >= 30 and ray_flag(ct, r0, c0, sites)),
                    "unphysical": int(st["max"] > UNPHYSICAL),
                    "has_qind": int(has_q),
                })
                if has_q:
                    qt = q[r0:r0 + patch, c0:c0 + patch]
                    wet = (ct >= 0.1) & np.isfinite(qt)
                    st["q_mean_wet"] = float(qt[wet].mean()) if wet.any() else np.nan
                else:
                    st["q_mean_wet"] = np.nan
                rows.append(st)
    ds.close()
    df = pd.DataFrame(rows)
    cols = ["timestamp", "row", "col", "max", "mean", "wet_frac", "n_ge1", "n_ge10", "n_ge31",
            "n_ge53", "n_ge89", "n_ge150", "coarse_max", "raw_max", "n_fixed", "ray",
            "unphysical", "q_mean_wet", "has_qind"]
    df = df[cols] if len(df) else pd.DataFrame(columns=cols)
    tmp = out_path + ".part"
    # %.7g: float32 round-trip. %.4g lost digits (486.25 -> 486.2), so the metadata max
    # disagreed with the stored tile in build_store.py --stage verify.
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.7g")
    os.replace(tmp, out_path)
    return day, len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--climatology", default=None)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["RAW_OPERA_DATA_DIR"]
    out = os.path.join(args.out_dir, "tiles")
    os.makedirs(out, exist_ok=True)
    days = sorted(d for d in glob.glob(os.path.join(raw_dir, "[0-9]" * 8))
                  if os.path.exists(os.path.join(d, ".zmetadata")))
    if args.days:
        days = [d for d in days if os.path.basename(d) in set(args.days)]
    todo = [d for d in days if not (args.skip_existing and os.path.exists(
        os.path.join(out, os.path.basename(d) + ".csv.gz")))]
    print(f"[scan] {len(days)} complete day stores, {len(todo)} to scan -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_day, d, cfg["PRECIP_VAR_NAME"], args.climatology,
                          cfg["PATCH_SIZE"], os.path.join(out, os.path.basename(d) + ".csv.gz")): d
                for d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                day, n = f.result()
                if k % 20 == 0 or k == len(futs):
                    el = time.time() - t0
                    print(f"  {k}/{len(futs)} days ({day}: {n} tiles), "
                          f"{el / k:.0f} s/day, ~{el / k * (len(futs) - k) / 3600:.1f} h left",
                          flush=True)
            except Exception as e:
                print(f"  ! {os.path.basename(futs[f])}: {type(e).__name__}: {e}", flush=True)
    print(f"[scan] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
