#!/usr/bin/env python
"""
Radar-gauge agreement by intensity level on the v4 training target (DECISIONS §43, todo A1).

Question: up to which hourly level u can the radar target be trusted as the truth of a held-out
extreme? For every tile-hour of the corrected lists on the two Swiss tiles, (640, 768) and
(640, 896), the target field is read from the store (`original_precip`, mm in the hour), and each
SwissMetNet station inside the tile is paired with its pixel. Gauge sums over two windows
(as in gauge_hourly.py, DECISIONS §22):
    g_clock   10-min intervals labelled H-50 .. H      (the clock hour)
    g_shift   10-min intervals labelled H-40 .. H+10   (the empirical radar alignment)
Only hours with all 6 gauge values are kept. Per level u, both directions:
    radar >= u   n, median gauge/radar, share with gauge < u/4   (false alarms, artefacts)
    gauge >= u   n, median radar/gauge (pixel and 3x3 max), share with radar < u/4  (misses)
with day-block bootstrap 95% intervals on the medians. A point gauge against a 2 km pixel loses
some agreement at every level (spatial representativeness), so what matters is how the numbers
change with u, not their absolute value.

Usage: python scripts/data_quality/gauge_levels.py <smn_dir> <out_dir>
       python scripts/data_quality/gauge_levels.py dwd <out_dir> [min_tile_max]

DWD (German gauges, `validation/gauges`: sparse wet 10-min intervals as labelled, UTC, and per-station-day
completeness, both from scripts/validation/fetch_gauges.py): every tile holding a station, same windows; a window
counts only if every labelled day it touches is complete (144 values). To keep the store reads affordable, only
tile-hours with tile max >= min_tile_max (default 10 mm) are read, which drops only gross misses at levels >= 20 mm
(gauge >= 20 while the radar stays < 10 mm over the whole 256 km tile).
"""

import os
import sys

import numpy as np
import pandas as pd
import zarr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from src.data.geo import lonlat_to_rowcol  # noqa: E402

V4 = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv"
TILES = {(640, 768), (640, 896)}
GROUP = {"train": "train", "val": "validation", "test": "test", "nimbus": "nimbus"}
LEVELS = [5, 10, 20.2, 31.7, 50.1, 80]
B = 200


def stations(smn):
    st = pd.read_csv(os.path.join(smn, "stations.csv")).drop_duplicates("station_abbr")
    r, c = lonlat_to_rowcol(st.station_coordinates_wgs84_lon.values, st.station_coordinates_wgs84_lat.values)
    st["row"], st["col"] = r, c
    out = []
    for (ty, tx) in TILES:
        s = st[(st.row >= ty) & (st.row < ty + 128) & (st.col >= tx) & (st.col < tx + 128)].copy()
        s["ty"], s["tx"], s["py"], s["px"] = ty, tx, s.row - ty, s.col - tx
        out.append(s)
    return pd.concat(out)


def gauge_windows(smn, abbr):
    f = os.path.join(smn, f"smn_{abbr}.csv.gz")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, parse_dates=["time"])
    z = d[d.res == "10min"].set_index("time").precip_mm.sort_index()
    z = z[~z.index.duplicated()]
    full = z.reindex(pd.date_range(z.index.min().floor("h"), z.index.max().ceil("h"), freq="10min"))
    # window sums ending at a 10-min stamp: rolling over 6 values, NaN unless all 6 present
    roll = full.rolling(6, min_periods=6).sum()
    clock = roll[roll.index.minute == 0]                      # labels H-50..H -> H
    shift = roll.shift(-1)[roll.index.minute == 0]            # labels H-40..H+10 -> H
    return pd.DataFrame({"g_clock": clock, "g_shift": shift})


def pairs(smn):
    st = stations(smn)
    gw = {a: gauge_windows(smn, a) for a in st.station_abbr.unique()}
    store = zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")
    rows = []
    for sp, grp in GROUP.items():
        m = pd.read_csv(f"{V4}/meta/full_{sp}.txt", header=None, names=["t", "y", "x", "mx", "r"])
        m = m[[k in TILES for k in zip(m.y, m.x)] & (m.mx >= 1.0)]
        g = store[grp]["original_precip"]
        for t, y, x, mx, r in m.itertuples(index=False):
            f = np.asarray(g[int(r)])
            T = pd.Timestamp(str(t))
            for s in st[(st.ty == y) & (st.tx == x)].itertuples():
                w = gw.get(s.station_abbr)
                if w is None or T not in w.index:
                    continue
                py, px = int(s.py), int(s.px)
                rows.append((sp, T, s.station_abbr, float(f[py, px]),
                             float(f[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2].max()),
                             w.at[T, "g_clock"], w.at[T, "g_shift"], mx))
    p = pd.DataFrame(rows, columns=["split", "time", "abbr", "radar", "radar3", "g_clock", "g_shift", "tile_max"])
    return p.dropna(subset=["g_clock", "g_shift"])


G = "/home/fquareng/work/data/extremes/OPERA/validation/gauges"


def pairs_dwd(min_tile_max):
    st = pd.read_csv(f"{G}/dwd_stations.csv", dtype={"station": str})
    st["sid"] = np.arange(len(st))                     # index used by dwd_wet.npz / dwd_avail.npz
    st = st[st.row.notna() & (st.n_valid_days > 0)].copy()
    st["ty"], st["tx"] = (st.row // 128 * 128).astype(int), (st.col // 128 * 128).astype(int)
    st["py"], st["px"] = (st.row - st.ty).astype(int), (st.col - st.tx).astype(int)
    tiles = set(zip(st.ty, st.tx))
    by_tile = {k: g for k, g in st.groupby(["ty", "tx"])}
    store = zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")
    rows = []                                          # pass 1: radar at every station pixel
    for sp, grp in GROUP.items():
        m = pd.read_csv(f"{V4}/meta/full_{sp}.txt", header=None, names=["t", "y", "x", "mx", "r"])
        m = m[np.array([k in tiles for k in zip(m.y, m.x)]) & (m.mx >= min_tile_max).values]
        g = store[grp]["original_precip"]
        print(f"[dwd] {sp}: {len(m)} tile-hours", flush=True)
        for t, y, x, mx, r in m.itertuples(index=False):
            f = np.asarray(g[int(r)])
            T = int(np.datetime64(pd.Timestamp(str(t)), "m").astype(np.int64))
            for s in by_tile[(y, x)].itertuples():
                win = f[max(s.py - 1, 0):s.py + 2, max(s.px - 1, 0):s.px + 2]
                rows.append((sp, T, s.station, s.sid, float(f[s.py, s.px]), float(win.max()), mx))
    p = pd.DataFrame(rows, columns=["split", "tmin", "abbr", "sid", "radar", "radar3", "tile_max"])
    # pass 2: gauge window sums from the sparse wet intervals, by cumulative sums per station
    w, a = np.load(f"{G}/dwd_wet.npz"), np.load(f"{G}/dwd_avail.npz")
    order = np.lexsort((w["t"], w["sid"]))
    sid_w, t_w, rr_w = w["sid"][order], w["t"][order], w["rr"][order].astype(np.float64)
    day0 = int(a["days"][0].astype("datetime64[m]").astype(np.int64))
    ok_day = a["n_valid"] >= 144
    p["g_clock"] = np.nan
    p["g_shift"] = np.nan
    for sid, idx in p.groupby("sid").groups.items():
        lo, hi = np.searchsorted(sid_w, [sid, sid + 1])
        tt, cs = t_w[lo:hi], np.concatenate([[0.0], np.cumsum(rr_w[lo:hi])])
        T = p.loc[idx, "tmin"].values
        S = lambda a_, b_: cs[np.searchsorted(tt, b_, "right")] - cs[np.searchsorted(tt, a_, "right")]

        def complete(first, last):                     # labelled days of the first and last interval
            d1, d2 = (first - 1 - day0) // 1440, (last - 1 - day0) // 1440
            inr = (d1 >= 0) & (d2 < ok_day.shape[1])
            out = np.zeros(len(first), bool)
            out[inr] = ok_day[sid, d1[inr]] & ok_day[sid, d2[inr]]
            return out
        clock = np.where(complete(T - 50, T), S(T - 60, T), np.nan)            # labels T-50 .. T
        shift = np.where(complete(T - 40, T + 10), S(T - 50, T + 10), np.nan)   # labels T-40 .. T+10
        p.loc[idx, "g_clock"], p.loc[idx, "g_shift"] = clock, shift
    p["time"] = pd.to_datetime(p.tmin, unit="m")
    return p.dropna(subset=["g_clock", "g_shift"]).drop(columns=["tmin", "sid"])


def boot_median(x, days, rng):
    ud = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in ud}
    meds = [np.median(x[np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])]) for _ in range(B)]
    return np.percentile(meds, [2.5, 97.5])


def summary(p, g):
    rng = np.random.default_rng(0)
    out = []
    for prod, q in (("ODYSSEY", p[p.split != "nimbus"]), ("NIMBUS", p[p.split == "nimbus"])):
        days = q.time.dt.strftime("%Y%m%d").values
        for u in LEVELS:
            a = (q.radar >= u).values
            b = (q[g] >= u).values
            rec = {"product": prod, "window": g, "u": u, "n_radar>=u": int(a.sum()), "n_gauge>=u": int(b.sum())}
            if a.sum() >= 10:
                ratio = (q[g] / q.radar).values[a]
                rec["med gauge/radar"] = np.median(ratio)
                rec["ci"] = boot_median(ratio, days[a], rng).round(2).tolist()
                rec["false alarm (gauge<u/4)"] = float((q[g].values[a] < u / 4).mean())
            if b.sum() >= 10:
                rec["med radar/gauge"] = np.median((q.radar / q[g]).values[b])
                rec["med radar3x3/gauge"] = np.median((q.radar3 / q[g]).values[b])
                rec["miss (radar3x3<u/4)"] = float((q.radar3.values[b] < u / 4).mean())
            out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    smn, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    p = pairs_dwd(float(sys.argv[3]) if len(sys.argv) > 3 else 10.0) if smn == "dwd" else pairs(smn)
    p.to_csv(os.path.join(out, "pairs.csv.gz"), index=False)
    pd.set_option("display.width", 250)
    for g in ("g_clock", "g_shift"):
        s = summary(p, g)
        s.to_csv(os.path.join(out, f"levels_{g}.csv"), index=False)
        print(s.round(3).to_string(index=False))
    print(f"{len(p)} pairs, {p.abbr.nunique()} stations, splits {p.split.value_counts().to_dict()}")
