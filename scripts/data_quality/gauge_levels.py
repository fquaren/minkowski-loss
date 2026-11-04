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
    p = pairs(smn)
    p.to_csv(os.path.join(out, "pairs.csv.gz"), index=False)
    pd.set_option("display.width", 250)
    for g in ("g_clock", "g_shift"):
        s = summary(p, g)
        s.to_csv(os.path.join(out, f"levels_{g}.csv"), index=False)
        print(s.round(3).to_string(index=False))
    print(f"{len(p)} pairs, {p.abbr.nunique()} stations, splits {p.split.value_counts().to_dict()}")
