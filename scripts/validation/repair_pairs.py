#!/usr/bin/env python
"""Add the v3-repaired radar values to existing gauge pairs (`gauge_vs_radar.py` output).

For every pairs file, the day is re-cleaned with `src.data.day_cleaner.DayCleaner` (the
chain the v3 store will use: clean_frame -> footprint / ray / ring repairs -> cells without
temporal support) on a crop aligned to the stride-128 tile grid, and three columns are
added at each station-frame:

  cln_r, cln3_r   repaired value at the gauge pixel / max over its 3x3 neighbourhood
  rep             OR of the repair bits that lowered the pixel (cleaning.REPAIR_*:
                  1 footprint, 2 ray, 4 ring, 8 unsupported)

The `cln` / `cln3` columns stay the v2 values, so `compare_repair.py` can compare the two
fields on identical station-frames.

Output: <out_dir>/pairs_v3/YYYYMMDD.csv.gz (restartable with --skip_existing).

    python scripts/validation/repair_pairs.py config.yaml --workers 8 --skip_existing
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

D = "/home/fquareng/work/data/extremes/OPERA"
P = 128
MARGIN = 40
_CLIM = _RING = _SITES = None
_HOT = {}


def _hot(path, year):
    global _CLIM
    from src.data.cleaning import hot_mask_from_climatology
    if _CLIM is None:
        _CLIM = dict(np.load(path))
    if year not in _HOT:
        _HOT[year] = hot_mask_from_climatology(_CLIM, year)
    return _HOT[year]


def _ring(path, year):
    global _RING
    if _RING is None:
        _RING = dict(np.load(path)) if path and os.path.exists(path) else {}
    return _RING.get(f"y{year}")


def _sites():
    global _SITES
    if _SITES is None:
        from src.data import geo
        s = geo.load_radar_sites()
        _SITES = np.c_[s["row"].values, s["col"].values].astype(float)
    return _SITES


def repair_day(pair_path, raw_dir, var, clim_path, ring_path, out_path):
    from src.data.day_cleaner import DayCleaner
    p = pd.read_csv(pair_path, dtype={"station": str})
    day = os.path.basename(pair_path)[:8]
    if not len(p):
        p.assign(cln_r=[], cln3_r=[], rep=[]).to_csv(out_path, index=False, compression="gzip")
        return day, 0
    y0 = max(((p["row"].min() - MARGIN) // P) * P, 0)
    y1 = min(-(-(p["row"].max() + MARGIN + 1) // P) * P, 2200)
    x0 = max(((p["col"].min() - MARGIN) // P) * P, 0)
    x1 = min(-(-(p["col"].max() + MARGIN + 1) // P) * P, 1900)
    dc = DayCleaner(raw_dir, day, var, hot=lambda y: _hot(clim_path, y),
                    ring=lambda y: _ring(ring_path, y), sites_rc=_sites(), crop=(y0, y1, x0, x1))
    kmap = {np.datetime_as_string(t, unit="m"): k for k, t in enumerate(dc.times)}
    cln_r = np.full(len(p), np.nan, np.float32)
    cln3_r = np.full(len(p), np.nan, np.float32)
    rep = np.zeros(len(p), np.uint8)
    nb = np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    for ts, g in p.groupby("timestamp", sort=True):
        k = kmap.get(ts)
        if k is None:
            continue
        f, code = dc.clean(k)
        ry, rx = g["row"].values - y0, g["col"].values - x0
        yy = np.clip(ry[:, None] + nb[None, :, 0], 0, f.shape[0] - 1)
        xx = np.clip(rx[:, None] + nb[None, :, 1], 0, f.shape[1] - 1)
        cln_r[g.index] = f[ry, rx]
        cln3_r[g.index] = np.nan_to_num(f[yy, xx], nan=0.0).max(1)
        rep[g.index] = code[ry, rx]
    dc.close()
    p["cln_r"], p["cln3_r"], p["rep"] = cln_r, cln3_r, rep
    tmp = out_path + ".part"
    p.to_csv(tmp, index=False, compression="gzip", float_format="%.5g")
    os.replace(tmp, out_path)
    return day, int((rep > 0).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--pairs_dir", default=f"{D}/validation/pairs")
    ap.add_argument("--climatology", default=f"{D}/quality_v2/clutter_climatology.npz")
    ap.add_argument("--ring_mask", default=f"{D}/quality_v2/ring_mask.npz")
    ap.add_argument("--out_dir", default=f"{D}/validation")
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(a.config)
    out = os.path.join(a.out_dir, "pairs_v3")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.pairs_dir, "*.csv.gz")))
    if a.days:
        files = [f for f in files if os.path.basename(f)[:8] in set(a.days)]
    todo = [f for f in files if not (a.skip_existing and os.path.exists(os.path.join(out, os.path.basename(f))))]
    print(f"[repair] {len(files)} pairs files, {len(todo)} to do -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(repair_day, f, cfg["RAW_OPERA_DATA_DIR"], cfg["PRECIP_VAR_NAME"], a.climatology,
                          a.ring_mask, os.path.join(out, os.path.basename(f))): f for f in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                day, n = fu.result()
                if k % 50 == 0 or k == len(futs):
                    el = time.time() - t0
                    print(f"  {k}/{len(futs)} ({day}: {n} repaired pairs), {el / k:.1f} s/day, "
                          f"~{el / k * (len(futs) - k) / 3600:.1f} h left", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"  ! {os.path.basename(futs[fu])}: {type(e).__name__}: {e}", flush=True)
    print(f"[repair] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
