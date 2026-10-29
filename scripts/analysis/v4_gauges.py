#!/usr/bin/env python
"""The v4 dataset side by side with the DE / CH gauges.

For every gauge whose pixel lies in a stored tile-hour of a split, the gauge hour (`g_shift`
window, as `gauge_hourly.py`) is paired with the 2 km target at the gauge pixel and with the
coarse input at the same place (the coarse cell holding the pixel). Rows carry the store's
inclusion weight, so distributions are population-representative.

  pairs    -> <OUT>/gauge_pairs_<split>.csv.gz
  report   weighted quantiles, exceedance frequencies and conditional ratios per network,
           Q-Q and exceedance figures (target, coarse input, gauge)

    python scripts/analysis/v4_gauges.py pairs --split test --workers 2
    python scripts/analysis/v4_gauges.py report --split test
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "data_quality")]

D = "/home/fquareng/work/data/extremes/OPERA"
STORE = f"{D}/patches_v4/preprocessed_dataset.zarr"
META = f"{D}/patches_v4/meta"
OUT = f"{REPO}/notes/data_analysis_v4"
GROUPS = {"train": ("train", "full_train"), "validation": ("validation", "full_val"),
          "test": ("test", "full_test"), "nimbus": ("nimbus", "full_nimbus")}
FACTOR = 12.5


def _day(args):
    day, rows, split = args
    import zarr
    import gauge_hourly as gh
    st = gh.stations()
    st = st.assign(tr=(st.row.astype(int) // 128) * 128, tc=(st.col.astype(int) // 128) * 128)
    G = zarr.open(STORE, mode="r")[GROUPS[split][0]]
    t0 = (pd.Timestamp(day) - pd.Timedelta(minutes=10)).to_datetime64()
    n = 24 * 6 + 12
    base = pd.Timestamp(t0)
    out = []
    for net in ("dwd", "smn"):
        s = st[st.net == net].merge(rows, left_on=["tr", "tc"], right_on=["row", "col"], suffixes=("", "_t"))
        if not len(s):
            continue
        sids = s.sid.unique()
        Gm = gh.gauge_series(net, t0, n, sids)
        pos = {k: i for i, k in enumerate(sids)}
        for srow, grp in s.groupby("store_row"):                     # each tile-hour read once
            ts = grp.timestamp.iloc[0]
            j = int((pd.Timestamp(ts[:12]) - base) / pd.Timedelta(minutes=10))
            if j - 4 < 0 or j + 2 > n:
                continue
            g = np.array([Gm[pos[k], j - 4:j + 2].sum() for k in grp.sid])
            keep = np.isfinite(g)
            if not keep.any():
                continue
            tile = G["original_precip"][int(srow)]
            crs = G["coarse_precip"][int(srow)]
            iy = grp.row.astype(int).values - grp.tr.values                # pixels inside the tile
            ix = grp.col.astype(int).values - grp.tc.values
            cy = np.minimum((iy / FACTOR).astype(int), crs.shape[0] - 1)
            cx = np.minimum((ix / FACTOR).astype(int), crs.shape[1] - 1)
            for k in np.nonzero(keep)[0]:
                out.append((day, ts[:12], grp.station.iloc[k], net, grp.weight.iloc[k],
                            float(tile[iy[k], ix[k]]), float(crs[cy[k], cx[k]]), float(g[k])))
    return pd.DataFrame(out, columns=["day", "end", "station", "net", "weight", "target", "coarse", "gauge"])


def cmd_pairs(split, workers):
    info = pd.read_csv(f"{META}/{GROUPS[split][1]}_info.csv.gz", dtype={"timestamp": str})
    ts = pd.to_datetime(info.timestamp, format="%Y%m%d%H%M%S")
    info["pday"] = (ts - pd.Timedelta(minutes=1)).dt.strftime("%Y%m%d")
    jobs = [(d, g[["timestamp", "row", "col", "store_row", "weight"]], split) for d, g in info.groupby("pday")]
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, p in enumerate(ex.map(_day, jobs, chunksize=2)):
            parts.append(p)
            if k % 25 == 0:
                print(f"  [pairs] {k + 1}/{len(jobs)} days, {sum(len(x) for x in parts):,} pairs", flush=True)
    df = pd.concat(parts, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/gauge_pairs_{split}.csv.gz", index=False)
    print(f"[pairs] {split}: {len(df):,} gauge-hours from {df.station.nunique()} stations on {df.day.nunique()} days")


def wq(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return np.interp(qs, c, x)


def cmd_report(split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.data_quality._md import md_table
    df = pd.read_csv(f"{OUT}/gauge_pairs_{split}.csv.gz", dtype={"station": str})
    qs = np.array([0.5, 0.9, 0.99, 0.999, 0.9999])
    rows = []
    for net, g in df.groupby("net"):
        w = g.weight.values
        for fld in ("gauge", "target", "coarse"):
            q = wq(g[fld].values, w, qs)
            rows.append({"network": net, "field": fld, "gauge-hours": len(g), "mean": np.average(g[fld], weights=w),
                         "P(0)": np.average(g[fld] < 0.1, weights=w),
                         **{f"q{100 * p:g}": v for p, v in zip(qs, q)},
                         **{f"P(>={u})": np.average(g[fld] >= u, weights=w) for u in (1, 5, 20)}})
    t = pd.DataFrame(rows)
    L = [f"# v4 {split} split against DE / CH gauges", "",
         f"{len(df):,} gauge-hours, {df.station.nunique()} stations, {df.day.nunique()} days; weighted by the store's "
         "inclusion weights. 'coarse' = the 25 km input cell holding the gauge.", "", md_table(t, index=False, floatfmt=".4g"), ""]
    for net, g in df.groupby("net"):
        h = g[g.gauge >= 20]
        if len(h):
            L.append(f"- {net}: on {len(h)} gauge-hours >= 20 mm, median target / gauge {np.median(h.target / h.gauge):.2f}, "
                     f"median coarse / gauge {np.median(h.coarse / h.gauge):.2f}")
    open(f"{OUT}/gauges_{split}.md", "w").write("\n".join(L) + "\n")
    st = {"gauge": dict(color="#1baf7a", ls=":", lw=1.8), "target": dict(color="#2a78d6", ls="-", lw=1.6),
          "coarse": dict(color="#eb6834", ls="--", lw=1.6)}
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    pq = 1 - np.logspace(-1, -5, 25)
    xs = np.logspace(-1, 2.2, 150)
    for k, (net, g) in enumerate(df.groupby("net")):
        w = g.weight.values
        gq = wq(g.gauge.values, w, pq)
        a = axs[0, k]
        for fld in ("target", "coarse"):
            a.plot(gq, wq(g[fld].values, w, pq), marker="o", ms=3, label=fld, **st[fld])
        hi = max(gq.max(), 1) * 1.2
        a.plot([0.1, hi], [0.1, hi], color="#6b6b66", lw=0.8)
        a.set(xscale="log", yscale="log", xlim=(0.3, hi), ylim=(0.3, hi), xlabel="gauge quantile (mm in 1 h)",
              ylabel="dataset quantile at the gauge", title=f"{net.upper()}: Q-Q, {len(g):,} gauge-hours")
        a.legend(frameon=False); a.grid(alpha=0.3)
        a = axs[1, k]
        for fld in ("gauge", "target", "coarse"):
            v = g[fld].values
            a.plot(xs, [np.average(v >= x, weights=w) for x in xs], label=fld, **st[fld])
        a.set(xscale="log", yscale="log", xlabel="x (mm in 1 h)", ylabel="P(X >= x)", title=f"{net.upper()}: exceedance")
        a.legend(frameon=False); a.grid(alpha=0.3)
    fig.savefig(f"{OUT}/gauges_{split}.png", dpi=120); plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["pairs", "report"])
    ap.add_argument("--split", default="test", choices=list(GROUPS))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cmd_pairs(a.split, a.workers) if a.cmd == "pairs" else cmd_report(a.split)
