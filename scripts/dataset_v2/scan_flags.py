#!/usr/bin/env python
"""Audit flags for every covered tile, to be validated before any of them removes data.

Companion to `scan_tiles.py` (same frames, same cleaning, same stride-128 tiles, same keys),
for tiles whose cleaned max is >= 1 mm/h. One row per (timestamp, tile):

  amax_row, amax_col  grid position of the tile's cleaned maximum
  radar, radar_km     nearest OPERA radar active that year (odim code), within 250 km of it
  q_at_max            QIND at that pixel (NaN where QIND is absent); NOT comparable across the
                      2024-07-05 product switch (ODYSSEY tail ~0.2, NIMBUS ~0.85)
  unsup_max, unsup_px max value / pixel count of cells with no temporal support
                      (`cleaning.temporal_support`: >= 10 mm/h, nothing >= 1 mm/h within 30 km
                      at t +- 15 min)
  undec_max           max value of cells whose support is undecidable (missing neighbour frame
                      or no coverage around the cell)
  ring                thin arc centred on a radar in this frame (`cleaning.ring_flag`), tiles
                      with >= 30 pixels >= 1 mm/h
  on_ring, ring_px31  the tile's maximum lies on a climatological range ring of that year
                      (`ring_climatology.py`: Stevns/Sindal 2013-2017, two Finnish radars in
                      2012), and the number of ring pixels >= 31 mm/h in the tile

The neighbour frames at 00:00 and 23:45 come from the adjacent day stores when they exist;
a neighbour is used only if it is exactly 15 min away.

Output: <out_dir>/flags/YYYYMMDD.csv.gz (restartable with --skip_existing).

    python scripts/dataset_v2/scan_flags.py config.yaml \
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

_CLIM, _HOT, _SITES, _OWNER, _RINGS = None, {}, None, {}, None
STEP = np.timedelta64(15, "m")


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
    global _RINGS
    if path is None:
        return None
    if _RINGS is None:
        _RINGS = dict(np.load(path))
    return _RINGS.get(f"y{year}")


def _year(v):
    try:
        return int(str(v).strip()[:4])
    except (TypeError, ValueError):
        return None


def _sites_year(year):
    """Active radar sites in `year` (same rule as radar_attribution.active_sites), and the
    (H, W) map of the nearest one within 250 km (-1 beyond)."""
    global _SITES
    from src.data import geo
    if _SITES is None:
        _SITES = geo.load_radar_sites()
    if year not in _OWNER:
        s = _SITES
        s0, s1 = s["startyear"].map(_year), s["finishyear"].map(_year)
        ok = ((s0.isna() | (s0 <= year)) & (s1.isna() | (s1 >= year))
              & s["row"].between(-200, 2400) & s["col"].between(-200, 2100))
        act = s[ok].reset_index(drop=True)
        rr, cc = np.mgrid[0:2200, 0:1900]
        x, y = geo.rowcol_to_xy(rr, cc)
        idx, dist = geo.nearest_radar(x, y, act, max_km=250.0)
        _OWNER[year] = (act, idx.astype(np.int16), dist.astype(np.float32))
    return _OWNER[year]


def _open(path):
    import xarray as xr
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:                                  # noqa: BLE001
        return xr.open_zarr(path, consolidated=False)


def _edge_frame(raw_dir, day, var, which, hot):
    """Cleaned last (which=-1) or first (which=0) frame of an adjacent day, with its time."""
    from src.data.cleaning import clean_frame
    p = os.path.join(raw_dir, day)
    if not os.path.exists(os.path.join(p, ".zmetadata")):
        return None, None
    ds = _open(p)
    t = ds.time.values[which]
    f = clean_frame(ds[var].isel(time=which).values.astype(np.float32), hot)[0]
    ds.close()
    return f, t


def scan_day(day_dir, raw_dir, var, clim_path, patch, out_path, ring_path=None):
    from src.data.cleaning import clean_frame, ring_flag, support_maps, temporal_support
    ds = _open(day_dir)
    day = os.path.basename(day_dir.rstrip("/"))
    year = int(day[:4])
    hot = _hot(clim_path, year)
    rmask = _ring(ring_path, year)
    act, owner, odist = _sites_year(year)
    sites_rc = np.c_[act["row"].values, act["col"].values].astype(float)
    odim = act["odim"].values
    has_q = "QIND" in ds
    times = ds.time.values
    d = pd.Timestamp(day)
    prev_day = (d - pd.Timedelta(days=1)).strftime("%Y%m%d")
    next_day = (d + pd.Timedelta(days=1)).strftime("%Y%m%d")
    T = len(times)
    cache, maps = {}, {}

    def frame(k):
        if k not in cache:
            if k == -1:
                cache[k] = _edge_frame(raw_dir, prev_day, var, -1, _hot(clim_path, year - (day[4:] == "0101")))
            elif k == T:
                cache[k] = _edge_frame(raw_dir, next_day, var, 0, _hot(clim_path, year + (day[4:] == "1231")))
            else:
                raw = ds[var].isel(time=k).values.astype(np.float32)
                cache[k] = (raw, clean_frame(raw, hot)[0], times[k])
            maps[k] = None if cache[k][-2] is None else support_maps(cache[k][-2])
            for old in [j for j in cache if j < k - 2]:
                del cache[old], maps[old]
        return cache[k]

    rows = []
    for k in range(T):
        raw, cur, t = frame(k)
        p_ = frame(k - 1)
        n_ = frame(k + 1)
        pf, pt = (p_[1], p_[2]) if k > 0 else p_
        nf, nt = (n_[1], n_[2]) if k < T - 1 else n_
        prev = maps[k - 1] if pf is not None and t - pt == STEP else None
        nxt = maps[k + 1] if nf is not None and nt - t == STEP else None
        sup = temporal_support(cur, prev, nxt)
        q = None
        ts = (np.datetime_as_string(t, unit="s").replace("-", "").replace("T", "").replace(":", ""))
        H, W = cur.shape
        for r0 in range(0, H - patch + 1, patch):
            for c0 in range(0, W - patch + 1, patch):
                rt = raw[r0:r0 + patch, c0:c0 + patch]
                if not np.isfinite(rt).all():
                    continue
                ct = cur[r0:r0 + patch, c0:c0 + patch]
                mx = float(ct.max())
                if mx < 1.0:
                    continue
                ay, ax = np.unravel_index(int(np.argmax(ct)), ct.shape)
                gy, gx = r0 + ay, c0 + ax
                st = sup[r0:r0 + patch, c0:c0 + patch]
                un = st == 1
                ud = st == -1
                o = int(owner[gy, gx])
                if has_q and q is None:
                    q = ds["QIND"].isel(time=k).values
                rows.append({
                    "timestamp": ts, "row": r0, "col": c0, "amax_row": gy, "amax_col": gx,
                    "radar": odim[o] if o >= 0 else "", "radar_km": float(odist[gy, gx]),
                    "q_at_max": float(q[gy, gx]) if has_q else np.nan,
                    "unsup_max": float(ct[un].max()) if un.any() else 0.0,
                    "unsup_px": int(un.sum()),
                    "undec_max": float(ct[ud].max()) if ud.any() else 0.0,
                    "ring": int((ct >= 1.0).sum() >= 30 and ring_flag(ct, r0, c0, sites_rc)),
                    "on_ring": int(rmask[gy, gx]) if rmask is not None else 0,
                    "ring_px31": int(((ct >= 31.0) & rmask[r0:r0 + patch, c0:c0 + patch]).sum())
                                 if rmask is not None else 0,
                })
    ds.close()
    cols = ["timestamp", "row", "col", "amax_row", "amax_col", "radar", "radar_km", "q_at_max",
            "unsup_max", "unsup_px", "undec_max", "ring", "on_ring", "ring_px31"]
    df = pd.DataFrame(rows, columns=cols)
    tmp = out_path + ".part"
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.7g")
    os.replace(tmp, out_path)
    return day, len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--climatology", default=None)
    ap.add_argument("--ring_mask", default=None)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["RAW_OPERA_DATA_DIR"]
    out = os.path.join(args.out_dir, "flags")
    os.makedirs(out, exist_ok=True)
    days = sorted(d for d in glob.glob(os.path.join(raw_dir, "[0-9]" * 8))
                  if os.path.exists(os.path.join(d, ".zmetadata")))
    if args.days:
        days = [d for d in days if os.path.basename(d) in set(args.days)]
    todo = [d for d in days if not (args.skip_existing and os.path.exists(
        os.path.join(out, os.path.basename(d) + ".csv.gz")))]
    print(f"[flags] {len(days)} complete day stores, {len(todo)} to scan -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_day, d, raw_dir, cfg["PRECIP_VAR_NAME"], args.climatology,
                          cfg["PATCH_SIZE"], os.path.join(out, os.path.basename(d) + ".csv.gz"),
                          args.ring_mask): d
                for d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                day, n = f.result()
                if k % 20 == 0 or k == len(futs):
                    el = time.time() - t0
                    print(f"  {k}/{len(futs)} days ({day}: {n} tiles), "
                          f"{el / k:.0f} s/day, ~{el / k * (len(futs) - k) / 3600:.1f} h left",
                          flush=True)
            except Exception as e:                     # noqa: BLE001
                print(f"  ! {os.path.basename(futs[f])}: {type(e).__name__}: {e}", flush=True)
    print(f"[flags] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
