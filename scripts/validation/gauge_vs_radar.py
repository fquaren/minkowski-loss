#!/usr/bin/env python
"""Pair every 15-min radar frame with the rain gauges under it, before and after cleaning.

For each day store and each gauge on the radar grid with a complete record that day
(>= 140 of 144 valid 10-min values), and each frame t, one row when either the radar
(raw, 3x3 max) reaches RADAR_MIN or a gauge interval near t reaches GAUGE_MIN:

  network, station, timestamp, row, col
  raw, raw3   raw composite at the gauge pixel / max over its 3x3 neighbourhood (mm/h)
  cln, cln3   the same after `cleaning.clean_frame` (what v2 stores)
  hot         the pixel is a static-clutter hot pixel that year
  support     `cleaning.temporal_support` code at the pixel (1 unsupported, 0, -1 undecidable)
  on_ring     the pixel is on a climatological range ring that year (`ring_climatology.py`)
  q           QIND at the pixel (NaN where absent)
  g_m10, g_0, g_p10, g_p20
              gauge totals (mm per 10 min) of the intervals LABELLED t - 10, t, t + 10 and
              t + 20 min, each floored to the 10-min grid: for frames at :15 / :45 they are
              the intervals labelled t - 15, t - 5, t + 5 and t + 15. The label convention
              is left open on purpose: `validate_tail.py` picks the alignment from the lag
              correlation.

Frames are cleaned on a crop around the gauges, with a 40-px margin: wider than the cleaning
windows (3x3, 5x5), so the cleaned values equal those of the full frame. Temporal support
can differ only for a cell that reaches the crop edge (its part outside is not seen).

Output: <out_dir>/pairs/YYYYMMDD.csv.gz, restartable with --skip_existing.

    python scripts/validation/gauge_vs_radar.py config.yaml --workers 2 --skip_existing
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
RADAR_MIN = 5.0          # mm/h, raw 3x3 max
GAUGE_MIN = 0.5          # mm per 10 min (3 mm/h)
MARGIN = 40              # px around the gauges: > support radius (15) + spike/hot windows
STEP = np.timedelta64(15, "m")

_G = None                # gauges: stations frame, per-station wet series, availability
_CLIM = _RING = None


def _load_gauges(gdir):
    global _G
    if _G is not None:
        return _G
    st_all, series, avail = [], {}, {}
    for net in ("dwd", "smn"):
        p = os.path.join(gdir, f"{net}_stations.csv")
        if not os.path.exists(p):
            continue
        st = pd.read_csv(p, dtype={"station": str})
        w = np.load(os.path.join(gdir, f"{net}_wet.npz"))
        a = np.load(os.path.join(gdir, f"{net}_avail.npz"))
        order = np.lexsort((w["t"], w["sid"]))
        sid, t, rr = w["sid"][order], w["t"][order], w["rr"][order]
        bounds = np.searchsorted(sid, np.arange(len(st) + 1))
        for i, s in st.iterrows():
            if not np.isfinite(s["row"]):
                continue
            key = (net, s["station"])
            series[key] = (t[bounds[i]:bounds[i + 1]], rr[bounds[i]:bounds[i + 1]])
            avail[key] = (a["days"], a["n_valid"][i])
        st["network"] = net
        st_all.append(st[np.isfinite(st["row"])])
    st = pd.concat(st_all, ignore_index=True)
    st["row"], st["col"] = st["row"].astype(int), st["col"].astype(int)
    _G = (st, series, avail)
    return _G


def _hot(clim_path, year):
    global _CLIM
    from src.data.cleaning import hot_mask_from_climatology
    if _CLIM is None:
        _CLIM = dict(np.load(clim_path))
    return hot_mask_from_climatology(_CLIM, year)


def _ring(path, year):
    global _RING
    if _RING is None:
        _RING = dict(np.load(path)) if path and os.path.exists(path) else {}
    return _RING.get(f"y{year}")


def pair_day(day_dir, var, gdir, clim_path, ring_path, out_path):
    import xarray as xr
    from src.data.cleaning import clean_frame, support_maps, temporal_support
    st, series, avail = _load_gauges(gdir)
    day = os.path.basename(day_dir.rstrip("/"))
    dday = np.datetime64(pd.Timestamp(day).date(), "D")
    # stations with a complete record that day
    keep = []
    for k, s in st.iterrows():
        days, nv = avail[(s["network"], s["station"])]
        i = int((dday - days[0]).astype(int))
        if 0 <= i < len(days) and nv[i] >= 140:
            keep.append(k)
    cols = ["network", "station", "timestamp", "row", "col", "raw", "raw3", "cln", "cln3", "hot",
            "support", "on_ring", "q", "g_m10", "g_0", "g_p10", "g_p20"]
    if not keep:
        pd.DataFrame(columns=cols).to_csv(out_path, index=False, compression="gzip")
        return day, 0
    s = st.loc[keep].reset_index(drop=True)
    y0, y1 = max(s["row"].min() - MARGIN, 0), min(s["row"].max() + MARGIN + 1, 2200)
    x0, x1 = max(s["col"].min() - MARGIN, 0), min(s["col"].max() + MARGIN + 1, 1900)
    ry, rx = s["row"].values - y0, s["col"].values - x0
    year = int(day[:4])
    hot_full = _hot(clim_path, year)
    hot = hot_full[y0:y1, x0:x1] if hot_full is not None else None
    ring = _ring(ring_path, year)
    on_ring = ring[s["row"].values, s["col"].values].astype(int) if ring is not None else np.zeros(len(s), int)
    hot_at = hot[ry, rx].astype(int) if hot is not None else np.zeros(len(s), int)

    # gauge 10-min values on a minute grid for this day +- 30 min
    t_lo = int(dday.astype("datetime64[m]").astype(np.int64)) - 30
    t_hi = t_lo + 1440 + 60
    G = np.zeros((len(s), (t_hi - t_lo) // 10 + 1), np.float32)
    for j, r in s.iterrows():
        t, rr = series[(r["network"], r["station"])]
        m = (t >= t_lo) & (t <= t_hi) & ((t - t_lo) % 10 == 0)
        G[j, (t[m] - t_lo) // 10] = rr[m]

    ds = xr.open_zarr(day_dir, consolidated=True)
    times = ds.time.values
    has_q = "QIND" in ds
    raws, clns, maps = {}, {}, {}
    nb = np.array([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    yy = np.clip(ry[:, None] + nb[None, :, 0], 0, y1 - y0 - 1)
    xx = np.clip(rx[:, None] + nb[None, :, 1], 0, x1 - x0 - 1)

    def raw_k(k):
        if k not in raws:
            raws[k] = ds[var].isel(time=k, y=slice(y0, y1), x=slice(x0, x1)).values.astype(np.float32)
        return raws[k]

    def cln_k(k):
        if k not in clns:
            clns[k] = clean_frame(raw_k(k), hot)[0]
            maps[k] = support_maps(clns[k])
        return clns[k]

    out = []
    for k in range(len(times)):
        for old in [j for j in raws if j < k - 1]:
            raws.pop(old, None); clns.pop(old, None); maps.pop(old, None)
        raw = raw_k(k)
        raw3 = np.nan_to_num(raw[yy, xx], nan=0.0).max(1)
        tm = int(times[k].astype("datetime64[m]").astype(np.int64))
        gi = [(tm + d - t_lo) // 10 for d in (-10, 0, 10, 20)]
        g = np.stack([G[:, i] if 0 <= i < G.shape[1] else np.full(len(s), np.nan) for i in gi], 1)
        sel = (raw3 >= RADAR_MIN) | (np.nan_to_num(g, nan=0.0).max(1) >= GAUGE_MIN)
        if not sel.any():
            continue                                   # dry everywhere: nothing to clean
        cln = cln_k(k)
        prev = maps.get(k - 1) if k > 0 and times[k] - times[k - 1] == STEP else None
        if prev is None and k > 0 and times[k] - times[k - 1] == STEP:
            cln_k(k - 1); prev = maps[k - 1]
        nxt = None
        if k + 1 < len(times) and times[k + 1] - times[k] == STEP:
            cln_k(k + 1); nxt = maps[k + 1]
        sup = temporal_support(cln, prev, nxt)
        cln3 = np.nan_to_num(cln[yy, xx], nan=0.0).max(1)
        q = ds["QIND"].isel(time=k, y=slice(y0, y1), x=slice(x0, x1)).values[ry, rx] if has_q else None
        ts = np.datetime_as_string(times[k], unit="m")
        for j in np.nonzero(sel)[0]:
            out.append((s.at[j, "network"], s.at[j, "station"], ts, int(s.at[j, "row"]), int(s.at[j, "col"]),
                        float(raw[ry[j], rx[j]]), float(raw3[j]), float(cln[ry[j], rx[j]]), float(cln3[j]),
                        int(hot_at[j]), int(sup[ry[j], rx[j]]), int(on_ring[j]),
                        float(q[j]) if has_q else np.nan, *map(float, g[j])))
    ds.close()
    df = pd.DataFrame(out, columns=cols)
    tmp = out_path + ".part"
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.5g")
    os.replace(tmp, out_path)
    return day, len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--gauge_dir", default=f"{D}/validation/gauges")
    ap.add_argument("--climatology", default=f"{D}/quality_v2/clutter_climatology.npz")
    ap.add_argument("--ring_mask", default=f"{D}/quality_v2/ring_mask.npz")
    ap.add_argument("--out_dir", default=f"{D}/validation")
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(a.config)
    raw_dir = cfg["RAW_OPERA_DATA_DIR"]
    out = os.path.join(a.out_dir, "pairs")
    os.makedirs(out, exist_ok=True)
    days = sorted(d for d in glob.glob(os.path.join(raw_dir, "[0-9]" * 8))
                  if os.path.exists(os.path.join(d, ".zmetadata")))
    if a.days:
        days = [d for d in days if os.path.basename(d) in set(a.days)]
    todo = [d for d in days if not (a.skip_existing and os.path.exists(
        os.path.join(out, os.path.basename(d) + ".csv.gz")))]
    print(f"[pairs] {len(days)} day stores, {len(todo)} to pair -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(pair_day, d, cfg["PRECIP_VAR_NAME"], a.gauge_dir, a.climatology,
                          a.ring_mask, os.path.join(out, os.path.basename(d) + ".csv.gz")): d
                for d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                day, n = f.result()
                if k % 20 == 0 or k == len(futs):
                    el = time.time() - t0
                    print(f"  {k}/{len(futs)} days ({day}: {n} pairs), {el / k:.0f} s/day, "
                          f"~{el / k * (len(futs) - k) / 3600:.1f} h left", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"  ! {os.path.basename(futs[f])}: {type(e).__name__}: {e}", flush=True)
    print(f"[pairs] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
