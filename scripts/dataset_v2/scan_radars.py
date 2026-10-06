#!/usr/bin/env python
"""v4 radar pass: per radar and frame, the features of a radar-wide failure, and per radar and
day, the value histogram that ceilings are found in (DECISIONS §19, rules 2, 3 and 5).

Runs on the `clean_frame` output (drizzle floor, static clutter, spikes), *before* the v3
repairs, so a repair cannot hide a failure: on 2018-04-29 the footprint repair turned the
Madrid failure into a flat plateau.

Per day, `<out_dir>/radars/YYYYMMDD.npz`:
  times     (T,)        frame timestamps, YYYYmmddHHMMSS
  keys      (R,)        radar keys (ODIM code, or location) active that year
  feats     (T, R, F)   float32, `radar_screen.FEATURES`
  hist_key, hist_n      day totals of values 10..500 mm/h on the 0.01 grid, per radar
                        (key = radar * 49001 + bin - 1000)

Restartable with --skip_existing; a day is written through a temporary file.

    python scripts/dataset_v2/scan_radars.py config.yaml \
        --climatology .../quality_v2/clutter_climatology.npz \
        --out_dir .../quality_v4 --workers 8 --skip_existing
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_CLIM, _HOT, _GEOM = None, {}, {}


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


def geometry(year):
    from src.data.radar_screen import RadarGeometry
    if year not in _GEOM:
        _GEOM.clear()                      # one year at a time per worker (memory)
        _GEOM[year] = RadarGeometry(year)
    return _GEOM[year]


def scan_day(day_dir, var, clim_path, out_path):
    from src.data.cleaning import clean_frame
    from src.data.day_cleaner import open_day
    from src.data.radar_screen import radar_frame_features, value_histogram
    day = os.path.basename(day_dir.rstrip("/"))
    year = int(day[:4])
    g = geometry(year)
    hot = _hot(clim_path, year)
    ds = open_day(day_dir)
    T = ds.sizes["time"]
    feats = np.full((T, g.R, 12), np.nan, np.float32)
    hk, hn = [], []
    for t in range(T):
        raw = ds[var].isel(time=t).values.astype(np.float32)
        c = clean_frame(raw, hot)[0]
        feats[t] = radar_frame_features(c, g)
        k, n = value_histogram(c, g)
        hk.append(k); hn.append(n)
    times = np.array([np.datetime_as_string(x, unit="s").replace("-", "").replace("T", "")
                      .replace(":", "") for x in ds.time.values])
    ds.close()
    k = np.concatenate(hk) if hk else np.empty(0, np.int64)
    n = np.concatenate(hn) if hn else np.empty(0, np.int64)
    uk, inv = np.unique(k, return_inverse=True)
    tot = np.bincount(inv, weights=n).astype(np.int64)
    tmp = out_path + ".part.npz"
    np.savez_compressed(tmp, times=times, keys=g.keys.astype(str), feats=feats,
                        hist_key=uk, hist_n=tot)
    os.replace(tmp, out_path)
    return day, T


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--climatology", default=None)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--days_file", default=None, help="one YYYYMMDD per line")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["RAW_OPERA_DATA_DIR"]
    out = os.path.join(args.out_dir, "radars")
    os.makedirs(out, exist_ok=True)
    days = sorted(d for d in glob.glob(os.path.join(raw_dir, "[0-9]" * 8))
                  if os.path.exists(os.path.join(d, ".zmetadata")))
    want = set(args.days or [])
    if args.days_file:
        want |= {l.strip() for l in open(args.days_file) if l.strip()}
    if want:
        days = [d for d in days if os.path.basename(d) in want]
    todo = [d for d in days if not (args.skip_existing and os.path.exists(
        os.path.join(out, os.path.basename(d) + ".npz")))]
    # sort by year so each worker builds few geometries
    todo.sort()
    print(f"[radars] {len(days)} day stores, {len(todo)} to scan -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_day, d, cfg["PRECIP_VAR_NAME"], args.climatology,
                          os.path.join(out, os.path.basename(d) + ".npz")): d for d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                day, n = f.result()
                if k % 20 == 0 or k == len(futs) or k <= 3:
                    el = time.time() - t0
                    print(f"  {k}/{len(futs)} days ({day}: {n} frames), {el / k:.0f} s/day, "
                          f"~{el / k * (len(futs) - k) / 3600:.1f} h left", flush=True)
            except Exception as e:
                print(f"  ! {os.path.basename(futs[f])}: {type(e).__name__}: {e}", flush=True)
    print(f"[radars] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
