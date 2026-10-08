#!/usr/bin/env python
"""Hourly radar vs DE/CH gauges, per rule of the hourly chain (DECISIONS §22, §23).

  run      for each day, run `src.data.hourly.HourlyDay` on a crop around the DWD and SMN
           gauges once per variant: `full` (every rule on), `no_<rule>` (that rule off), and
           `none` (every rule off; only the drizzle floor). Writes one row per (hour, gauge)
           with the raw hourly sum, every variant's sum and two gauge windows:
             g_clock   intervals labelled H-50 .. H      (clock hour H-60 .. H)
             g_shift   intervals labelled H-40 .. H+10   (H-50 .. H+10; the empirical radar
                       alignment, frame t <-> gauge interval labelled t+10)
           Gauges need a complete day (144 values). Gauges on a hot (static-clutter) pixel of
           that year are kept and marked `on_hot`: they are where the clutter repair acts, so
           dropping them would hide that rule's effect (`gauge_agg.py` meant to drop them but
           did not).
  summary  log-Pearson against the gauge on ONE fixed pair set (wet >= 0.1 mm in the raw sum,
           the full chain or the gauge), raw -> each variant, with a day-block bootstrap; the
           marginal gain of each rule (full minus no_<rule>); and two strata:
             false alarms  raw >= 5 mm, gauge < 0.5 mm: share brought below 1 mm
             misses        gauge >= 10 mm: median radar / gauge, raw vs variant
                           (filtering must not push heavy hours further below the gauge)

    python scripts/data_quality/gauge_hourly.py run --days_file .../gdays.txt --out .../gauge_hourly
    python scripts/data_quality/gauge_hourly.py summary --out .../gauge_hourly
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
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
D = "/home/fquareng/work/data/extremes/OPERA"
G = f"{D}/validation/gauges"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
RULES = ("hot", "spike", "ceiling", "footprint", "ring", "ray", "persist", "unsupported")
VARIANTS = ["full", "none"] + [f"no_{r}" for r in RULES]


def stations():
    out = []
    for net in ("dwd", "smn"):
        s = pd.read_csv(f"{G}/{net}_stations.csv", dtype={"station": str})
        s["net"] = net
        s["sid"] = np.arange(len(s))
        out.append(s[s.row.notna() & (s.n_valid_days > 0)])
    return pd.concat(out, ignore_index=True)


_CACHE = {}


def gauge_series(net, t0, n, sids):
    """(len(sids), n) mm per 10 min for intervals labelled t0, t0+10, ...; NaN where a gauge's
    labelled day is incomplete."""
    if net not in _CACHE:
        a = np.load(f"{G}/{net}_avail.npz")
        w = np.load(f"{G}/{net}_wet.npz")
        _CACHE[net] = ({"n_valid": a["n_valid"], "days": [str(d)[:10] for d in a["days"]]},
                       {k: w[k] for k in w.files})
    av, w = _CACHE[net]
    tm = int(np.datetime64(t0, "m").astype(np.int64))
    sel = (w["t"] >= tm) & (w["t"] < tm + 10 * n)
    M = np.zeros((len(sids), n), np.float32)
    pos = {s: i for i, s in enumerate(sids)}
    for s, tt, r in zip(w["sid"][sel], w["t"][sel], w["rr"][sel]):
        if s in pos:
            M[pos[s], (tt - tm) // 10] = r
    for k in range(n):                                   # completeness of each labelled day
        lab = np.datetime64(tm + 10 * k, "m") - np.timedelta64(1, "m")
        day = str(lab.astype("datetime64[D]"))
        if day in av["days"]:
            j = av["days"].index(day)
            M[~(av["n_valid"][sids, j] >= 144), k] = np.nan
        else:
            M[:, k] = np.nan
    return M


def run_day(args):
    day, out = args
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    import yaml
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    sdf = stations()
    y0, y1 = int(sdf.row.min()) - 48, int(sdf.row.max()) + 48
    x0, x1 = int(sdf.col.min()) - 48, int(sdf.col.max()) + 48
    r = sdf.row.values.astype(int) - y0
    c = sdf.col.values.astype(int) - x0
    hot = st._hot(CLIM, int(day[:4]))
    on_hot = hot[sdf.row.values.astype(int), sdf.col.values.astype(int)] if hot is not None else np.zeros(len(sdf), bool)
    t_start = time.time()
    res, raw, ends = {}, None, None
    for v in VARIANTS:
        rules = {k: True for k in RULES}
        if v == "none":
            rules = {k: False for k in RULES}
        elif v.startswith("no_"):
            rules[v[3:]] = False
        cfg = {**q.get("hourly", {}), "rules": rules}
        hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                       ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                       cfg=cfg, crop=(y0, y1, x0, x1)).run()
        if not hd.hours:
            return f"{day}: no complete hours"
        res[v] = np.stack([h["sum"][r, c] for h in hd.hours])            # (hours, stations)
        if raw is None:
            raw = np.stack([h["raw_sum"][r, c] for h in hd.hours])
            ends = [h["end"] for h in hd.hours]
    t0 = (pd.Timestamp(day) - pd.Timedelta(minutes=10)).to_datetime64()   # first label needed: d 00:10 - 60
    n = 24 * 6 + 12
    Gm = np.full((len(sdf), n), np.nan, np.float32)
    for net in ("dwd", "smn"):
        m = (sdf.net == net).values
        Gm[m] = gauge_series(net, t0, n, sdf.sid.values[m])
    base = pd.Timestamp(t0)
    rows = []
    for k, e in enumerate(ends):
        e = pd.Timestamp(e)
        j = int((e - base) / pd.Timedelta(minutes=10))                     # index of label H
        gc = Gm[:, j - 5:j + 1].sum(1) if j - 5 >= 0 and j + 1 <= n else np.full(len(sdf), np.nan)
        gs = Gm[:, j - 4:j + 2].sum(1) if j - 4 >= 0 and j + 2 <= n else np.full(len(sdf), np.nan)
        d = {"day": day, "end": e.strftime("%Y%m%d%H%M"), "station": sdf.station.values,
             "net": sdf.net.values, "on_hot": on_hot.astype(int), "raw": raw[k], "g_clock": gc, "g_shift": gs}
        for v in VARIANTS:
            d[v] = res[v][k]
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(df[["raw", "g_clock", "g_shift"]]).all(axis=1)]
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"g_{day}.csv.gz"), index=False, float_format="%.4g")
    return f"{day}: {len(df)} gauge-hours, {time.time() - t_start:.0f} s"


def corr(a, b):
    return np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]


def summary(out, gcol="g_shift", nboot=300, seed=0, subset="all"):
    g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out, "g_*.csv.gz")))], ignore_index=True)
    g = g.dropna(subset=["raw", gcol, *VARIANTS])
    if subset != "all":
        g = g[g.on_hot == (1 if subset == "hot" else 0)]
    w = g[(g.raw >= 0.1) | (g.full >= 0.1) | (g[gcol] >= 0.1)]
    days = w.day.unique()
    by = {d: x for d, x in w.groupby("day")}
    rng = np.random.default_rng(seed)
    boots = [pd.concat([by[d] for d in rng.choice(days, len(days))]) for _ in range(nboot)]
    print(f"{g.day.nunique()} days, {len(w):,} wet gauge-hours (fixed set), gauges: {subset}, window {gcol}")
    print(f"clock vs shifted gauge window, full chain: {corr(w.full, w.g_clock):.3f} vs {corr(w.full, w.g_shift):.3f}")
    rows = []
    base = corr(w.raw, w[gcol])
    for v in VARIANTS:
        d = [corr(b[v], b[gcol]) - corr(b.raw, b[gcol]) for b in boots]
        m = [corr(b.full, b[gcol]) - corr(b[v], b[gcol]) for b in boots] if v.startswith("no_") else None
        fa = w[(w.raw >= 5) & (w[gcol] < 0.5)]
        hv = w[w[gcol] >= 10]
        rows.append({"variant": v, "corr": corr(w[v], w[gcol]), "gain_vs_raw": corr(w[v], w[gcol]) - base,
                     "gain_lo": np.quantile(d, .025), "gain_hi": np.quantile(d, .975),
                     "rule_marginal": (corr(w.full, w[gcol]) - corr(w[v], w[gcol])) if m else np.nan,
                     "marg_lo": np.quantile(m, .025) if m else np.nan, "marg_hi": np.quantile(m, .975) if m else np.nan,
                     "false_alarms_fixed": float((fa[v] < 1).mean()) if len(fa) else np.nan,
                     "n_false_alarm": len(fa),
                     "heavy_ratio_med": float((hv[v] / hv[gcol]).median()) if len(hv) else np.nan,
                     "n_heavy": len(hv)})
    t = pd.DataFrame(rows)
    print(f"raw: corr {base:.3f}; heavy (gauge >= 10 mm) median radar/gauge {(w[w[gcol] >= 10].raw / w[w[gcol] >= 10][gcol]).median():.3f}")
    print(t.round(4).to_string(index=False))
    t.to_csv(os.path.join(out, f"summary_{gcol}_{subset}.csv"), index=False)

    # Influence: one absurd raw value at one gauge can carry the whole pooled gain (2026-10-08:
    # DWD 01346 on a clutter pixel, raw up to 2e5 mm/h, was +0.10 of a +0.108 gain).
    st_gain = {}
    for s_, x in w.groupby("station"):
        y = w[w.station != s_]
        st_gain[s_] = (corr(y.full, y[gcol]) - corr(y.raw, y[gcol]))
    sg = pd.Series(st_gain).sort_values()
    full_gain = corr(w.full, w[gcol]) - base
    print(f"\ninfluence: full-chain gain {full_gain:+.4f}; leaving out one station gives {sg.min():+.4f} .. {sg.max():+.4f} "
          f"(most influential: {sg.index[0]}, gain without it {sg.iloc[0]:+.4f})")

    # Affected pairs: where a rule changed the radar value, did it move towards the gauge?
    e = lambda col: np.abs(np.log1p(w[col]) - np.log1p(w[gcol]))
    rows = []
    for r_ in RULES:
        v = f"no_{r_}"
        aff = (w.full - w[v]).abs() > 0.01
        if not aff.any():
            rows.append({"rule": r_, "n_affected": 0}); continue
        d = (e("full") - e(v))[aff]                       # < 0: the rule moved radar towards the gauge
        rows.append({"rule": r_, "n_affected": int(aff.sum()), "n_stations": w[aff].station.nunique(),
                     "share_closer": float((d < -0.01).mean()), "share_further": float((d > 0.01).mean()),
                     "median_dlogerr": float(d.median()),
                     "gauge_wet_share": float((w[aff][gcol] >= 0.1).mean())})
    a = pd.DataFrame(rows)
    print("\naffected gauge-hours per rule (full chain vs the chain without the rule):")
    print(a.round(3).to_string(index=False))
    a.to_csv(os.path.join(out, f"affected_{gcol}_{subset}.csv"), index=False)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--days_file", default=None, help="comma- or newline-separated YYYY-MM-DD or YYYYMMDD")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gauge_window", default="g_shift", choices=["g_shift", "g_clock"])
    ap.add_argument("--subset", default="all", choices=["all", "hot", "nohot"], help="gauges on hot pixels or not")
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a.out, a.gauge_window, subset=a.subset)
        return
    days = list(a.days or [])
    if a.days_file:
        days += [x.strip() for x in open(a.days_file).read().replace(",", "\n").split() if x.strip()]
    days = [d.replace("-", "") for d in days]
    days = [d for d in days if not os.path.exists(os.path.join(a.out, f"g_{d}.csv.gz"))]
    print(f"[gauge_hourly] {len(days)} days x {len(VARIANTS)} variants -> {a.out}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(run_day, [(d, a.out) for d in days]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
