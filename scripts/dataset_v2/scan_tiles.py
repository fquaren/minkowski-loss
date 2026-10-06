#!/usr/bin/env python
"""Scan every raw day store: clean each frame and describe every fully covered tile.

One row per (timestamp, tile) where the 128x128 tile on the stride-128 grid has no NaN (the
same validity rule as `generate_metadata.py`). The frame is cleaned first with
`src.data.day_cleaner.DayCleaner`: `clean_frame` (drizzle floor, static-clutter and spike
repair; no zeroing above 150 mm/h), then the v3 repairs (footprints of > 500 mm/h cores, rays,
range rings, cells without temporal support; `--no_repair` stops after `clean_frame`, i.e. the
v2 field), so every statistic describes the field the store will actually hold.

Columns: timestamp,row,col, cleaned-tile stats (max, mean, wet_frac, n_ge*, coarse_max),
raw_max, n_fixed (pixels changed by the cleaning), ray (RLAN ray pointing at a radar),
unphysical (cleaned max > 500), q_mean_wet / has_qind, n_repaired (pixels lowered by the v3
repairs; 0 with --no_repair).

v4 (`--quality configs/quality_v4.yaml`, DECISIONS §19): ceiling pixels are repaired after
`clean_frame` and every ring-median repair is guarded; extra columns n_ceiling, n_refused,
rep_value, rep_count, rep_excess, max_rep (repeated-value test on the final tile) and owners
('radar:share;...', for joining radar-frame flags and radar-day exclusions at split time).
Without --quality the output is the v3 table, unchanged.

Output: <out_dir>/tiles/YYYYMMDD.csv.gz. Restartable with --skip_existing; a day is written
through a temporary file, so a partial file never looks complete.

    python scripts/dataset_v2/scan_tiles.py config.yaml \
        --climatology .../quality_v2/clutter_climatology.npz \
        --ring_mask .../quality_v2/ring_mask.npz \
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
_RING = None
_Q = {}
_GEOM = {}


def _quality(path):
    """(quality dict, CeilingTable) from a quality_v4.yaml, cached per worker."""
    if path is None:
        return None, None
    if path not in _Q:
        import yaml
        from src.data.radar_screen import CeilingTable
        q = yaml.safe_load(open(path))
        tp = os.path.join(q["data_dir"], q["ceiling"]["table"])
        rows = pd.read_csv(tp).to_dict("records") if os.path.exists(tp) else []
        if not rows:
            print(f"[scan] warning: no ceiling table at {tp}; ceilings not repaired", flush=True)
        _Q[path] = (q, CeilingTable(rows))
    return _Q[path]


def _geometry(year):
    from src.data.radar_screen import RadarGeometry
    if year not in _GEOM:
        _GEOM.clear()
        _GEOM[year] = RadarGeometry(year)
    return _GEOM[year]


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


def scan_day(day_dir, var, clim_path, patch, out_path, ring_path=None, repair=True,
             quality=None):
    import xarray as xr
    from src.data.cleaning import ray_flag, tile_stats, UNPHYSICAL
    from src.data.day_cleaner import DayCleaner
    from src.data.cleaning import REPAIR_CEILING, REPAIR_REFUSED
    from src.data.radar_screen import owner_shares, repeated_value
    day = os.path.basename(day_dir.rstrip("/"))
    sites = _sites()
    qcfg, ceil = _quality(quality)
    v4 = qcfg is not None
    dc = DayCleaner(os.path.dirname(day_dir.rstrip("/")), day, var,
                    hot=lambda y: _hot(clim_path, y), ring=lambda y: _ring(ring_path, y),
                    sites_rc=sites, repair=repair,
                    ceilings=ceil if v4 else None,
                    max_size=qcfg["guard"]["max_size"] if v4 else None)
    g = _geometry(int(day[:4])) if v4 else None
    owners = {}
    ds = dc.ds
    has_q = "QIND" in ds
    rows = []
    for t in range(ds.sizes["time"]):
        raw = dc.raw(t)
        H, W = raw.shape
        clean, rcode = dc.clean(t)
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
                    "n_repaired": int((rcode[r0:r0 + patch, c0:c0 + patch] > 0).sum()),
                })
                if has_q:
                    qt = q[r0:r0 + patch, c0:c0 + patch]
                    wet = (ct >= 0.1) & np.isfinite(qt)
                    st["q_mean_wet"] = float(qt[wet].mean()) if wet.any() else np.nan
                else:
                    st["q_mean_wet"] = np.nan
                if v4:
                    cd = rcode[r0:r0 + patch, c0:c0 + patch]
                    st["n_ceiling"] = int(((cd & REPAIR_CEILING) > 0).sum())
                    st["n_refused"] = int(((cd & REPAIR_REFUSED) > 0).sum())
                    st.update(repeated_value(ct, qcfg["repeat"]["u"]))
                    if (r0, c0) not in owners:
                        owners[(r0, c0)] = owner_shares(g, r0, c0, patch)
                    st["owners"] = owners[(r0, c0)]
                rows.append(st)
    dc.close()
    df = pd.DataFrame(rows)
    cols = ["timestamp", "row", "col", "max", "mean", "wet_frac", "n_ge1", "n_ge10", "n_ge31",
            "n_ge53", "n_ge89", "n_ge150", "coarse_max", "raw_max", "n_fixed", "ray",
            "unphysical", "q_mean_wet", "has_qind", "n_repaired"]
    if v4:
        cols += ["n_ceiling", "n_refused", "rep_value", "rep_count", "rep_excess", "max_rep",
                 "owners"]
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
    ap.add_argument("--ring_mask", default=None, help="ring_mask.npz (ring_climatology.py)")
    ap.add_argument("--no_repair", action="store_true", help="v2 cleaning (clean_frame only)")
    ap.add_argument("--quality", default=None, help="v4: configs/quality_v4.yaml")
    ap.add_argument("--days_file", default=None, help="one YYYYMMDD per line")
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
    want = set(args.days or [])
    if args.days_file:
        want |= {l.strip() for l in open(args.days_file) if l.strip()}
    if want:
        days = [d for d in days if os.path.basename(d) in want]
    todo = [d for d in days if not (args.skip_existing and os.path.exists(
        os.path.join(out, os.path.basename(d) + ".csv.gz")))]
    print(f"[scan] {len(days)} complete day stores, {len(todo)} to scan -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_day, d, cfg["PRECIP_VAR_NAME"], args.climatology,
                          cfg["PATCH_SIZE"], os.path.join(out, os.path.basename(d) + ".csv.gz"),
                          args.ring_mask, not args.no_repair, args.quality): d
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
