#!/usr/bin/env python
"""Criterion 3 of DECISIONS §25: does advection correction push heavy gauge-hours down?

On the DE/CH gauge days, the screened OPERA hour (ACRR convention, hour ending H) is formed as
the plain sum and as each advection-corrected variant of `advection_test.VARIANTS`, over the
same frame-time window (H-52.5 .. H+7.5, frames H-60 .. H+15; hours whose frames are not all in
the processing day are skipped). Each is compared with the gauge sum over the intervals
labelled H-40 .. H+10 (`g_shift`, the best-matching window, DECISIONS §22).

  run      writes one row per (hour, gauge): plain, every variant, gauge, network, on_hot
  summary  for gauge-hours >= 10 / 20 / 30 mm, the median radar/gauge ratio per variant and the
           day-block bootstrap interval of its median log-ratio difference to plain, for DE+CH
           and for CH alone (SMN gauges do not enter RADKLIM). Pass: the difference is not
           significantly negative at >= 20 mm.

    python scripts/data_quality/advection_gauges.py run --workers 2
    python scripts/data_quality/advection_gauges.py summary
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2"), os.path.dirname(__file__)]
from gauge_hourly import CLIM, D, RING, gauge_series, stations  # noqa: E402
from advection_test import VARIANTS  # noqa: E402

OUT = f"{D}/quality_v4/calib/advection_gauges"
GDAYS = f"{D}/quality_v4/calib/daily_check/gdays.txt"


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay, advected_mean, motion_lk
    t0 = time.time()
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    sdf = stations()
    y0, y1 = int(sdf.row.min()) - 48, int(sdf.row.max()) + 48
    x0, x1 = int(sdf.col.min()) - 48, int(sdf.col.max()) + 48
    r = sdf.row.values.astype(int) - y0
    c = sdf.col.values.astype(int) - x0
    hot = st._hot(CLIM, int(day[:4]))
    on_hot = hot[sdf.row.values.astype(int), sdf.col.values.astype(int)] if hot is not None else np.zeros(len(sdf), bool)
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q, crop=(y0, y1, x0, x1)).run()
    ns = lambda t: int(np.datetime64(t, "ns").astype(np.int64))
    pos = {ns(t): i for i, t in enumerate(hd.frames)}
    F = np.nan_to_num(hd.fin, nan=0.0)
    valid_all = np.isfinite(hd.fin)
    tg0 = (pd.Timestamp(day) - pd.Timedelta(minutes=10)).to_datetime64()
    n = 24 * 6 + 12
    Gm = np.full((len(sdf), n), np.nan, np.float32)
    for net in ("dwd", "smn"):
        m = (sdf.net == net).values
        Gm[m] = gauge_series(net, tg0, n, sdf.sid.values[m])
    base = pd.Timestamp(tg0)
    mcache, rows = {}, []
    for h in hd.hours:
        H = h["end"]
        want = [H - np.timedelta64(15 * k, "m") for k in (4, 3, 2, 1, 0, -1)]      # H-60 .. H+15
        if not all(ns(w) in pos for w in want):
            continue
        idx = [pos[ns(w)] for w in want]
        fr = [F[j] for j in idx]
        mo = []
        for a_, b_ in zip(idx[:-1], idx[1:]):
            if (a_, b_) not in mcache:
                mcache[(a_, b_)] = motion_lk(F[a_], F[b_])
            mo.append(mcache[(a_, b_)])
        ok = valid_all[h["idx"]].all(0)[r, c]
        d = {"day": day, "end": pd.Timestamp(H).strftime("%Y%m%d%H%M"), "station": sdf.station.values,
             "net": sdf.net.values, "on_hot": on_hot.astype(int), "plain": np.where(ok, h["sum"][r, c], np.nan)}
        for v, (mode, interp) in VARIANTS.items():
            d[v] = np.where(ok, advected_mean(fr, mo, 7.5, 67.5, mode, interp)[r, c], np.nan)
        j = int((pd.Timestamp(H) - base) / pd.Timedelta(minutes=10))
        d["gauge"] = Gm[:, j - 4:j + 2].sum(1) if j - 4 >= 0 and j + 2 <= n else np.nan
        rows.append(pd.DataFrame(d))
    os.makedirs(OUT, exist_ok=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(df):
        df = df.dropna(subset=["plain", "gauge"])
    df.to_csv(os.path.join(OUT, f"g_{day}.csv.gz"), index=False, float_format="%.4g")
    return f"{day}: {len(df)} gauge-hours, {time.time() - t0:.0f} s"


def summary(nboot=500, seed=0):
    g = pd.concat([pd.read_csv(f, dtype={"day": str, "station": str}) for f in glob.glob(f"{OUT}/g_*.csv.gz")
                   if os.path.getsize(f) > 100], ignore_index=True)
    names = ["plain"] + list(VARIANTS)
    rng = np.random.default_rng(seed)
    lines = [f"{g.day.nunique()} days, {len(g):,} gauge-hours; radar/gauge on heavy gauge-hours (window g_shift)"]
    for net_name, sub in (("DE+CH", g), ("CH only", g[g.net == "smn"])):
        for u in (10, 20, 30):
            x = sub[sub.gauge >= u]
            if len(x) < 20:
                continue
            days = x.day.unique()
            by = {d: y for d, y in x.groupby("day")}
            boots = [pd.concat([by[d] for d in rng.choice(days, len(days))]) for _ in range(nboot)]
            row = [f"{net_name} gauge >= {u} mm (n {len(x)}, {len(days)} days): plain median ratio {(x.plain / x.gauge).median():.3f}"]
            for v in names[1:]:
                lr = lambda y: np.median(np.log((y[v] + 0.1) / (y.gauge + 0.1))) - np.median(np.log((y.plain + 0.1) / (y.gauge + 0.1)))
                bs = [lr(b) for b in boots]
                lo, hi = np.quantile(bs, [0.025, 0.975])
                verdict = "FAIL (pushed down)" if hi < 0 else "pass"
                row.append(f"    {v}: ratio {(x[v] / x.gauge).median():.3f}; d median log-ratio vs plain {lr(x):+.3f} [{lo:+.3f}, {hi:+.3f}] {verdict if u >= 20 else ''}")
            lines.append("\n".join(row))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    days = [x.strip().replace("-", "") for x in open(GDAYS).read().replace(",", "\n").split() if x.strip()]
    days = [d for d in days if not os.path.exists(os.path.join(OUT, f"g_{d}.csv.gz"))]
    print(f"[advection_gauges] {len(days)} days -> {OUT}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(run_day, days):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
