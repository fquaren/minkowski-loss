#!/usr/bin/env python
"""Readiness gate before the full-archive build (DECISIONS §29), criteria fixed 2026-10-13
before running. Reads:
  - the calibration days rescanned with the current chain (`scan_hours.py` -> calib/hourly_c),
    rejection by `decide_hourly` with the 2026-10-09 frame flags (criteria 1 and 3);
  - `gauge_hourly.py run --variants none,full` outputs (calib/gauge_gate; criterion 2). These
    are pixel values of the chain; split-time tile rejection is not applied to them.
Rule 5 is reported three ways: off, the pre-stated §28 list (day_exclusions.csv), and §28 with
the radar-year share over days with data instead of rain days.

    python scripts/data_quality/gate_prebuild.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
from src.data.radar_screen import decide_hourly  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
CAL = f"{Q4}/calib"
HOURS = f"{CAL}/hourly_c/hours"
GAUGE = f"{CAL}/gauge_gate"
OUT = f"{CAL}/gate_prebuild"
U_MAIN, U_ALL = 20.0, (10.0, 20.0, 30.0)


def rule5_sets(q):
    pre = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    r = pd.read_csv(f"{Q4}/ranking.csv.gz", dtype={"day": str}, usecols=["radar", "day", "year", "score"])
    p = q["radar_day"]
    r["e"] = r.score >= p["exclude_score"]
    y = r.groupby(["radar", "year"]).agg(n=("day", "size"), k=("e", "sum")).reset_index()
    y = y[(y.k / y.n >= p["year_share"]) & (y.k >= p["year_min_days"])]
    cor = pd.concat([r.loc[r.e, ["radar", "day"]], r.merge(y[["radar", "year"]], on=["radar", "year"])[["radar", "day"]]])
    return {"off": set(), "pre-stated": set(zip(pre.radar, pre.day)), "days-with-data": set(zip(cor.radar, cor.day))}


def load_hours():
    fs = sorted(glob.glob(f"{HOURS}/*.csv.gz"))
    t = pd.concat([pd.read_csv(f, dtype={"end": str, "frames": str, "owners": str}).assign(
        day=pd.Timestamp(os.path.basename(f)[:8])) for f in fs], ignore_index=True)
    return t, len(fs)


def owner_share(owners, key):
    for kv in str(owners).split(";"):
        if kv.startswith(key + ":"):
            return float(kv.split(":")[1])
    return 0.0


def crit1(t, rej, cal):
    out = []
    k = ~rej
    m = (t.day == pd.Timestamp("2018-04-29")).values
    sh = np.array([owner_share(o, "estjv") for o in t.owners.values])
    cand = m & (sh >= 0.10) & (t.max_rate.values >= 31)
    out.append(("madrid_20180429 (estjv >= 10%, max_rate >= 31)", int(cand.sum()), int((cand & k).sum())))
    vdays = {pd.Timestamp(d) for d in cal[cal.category.str.contains("valjevo")].day}
    m = t.day.isin(vdays).values
    cand = m & (np.abs(t.max_rate.values - 364.63) <= 0.005)
    out.append((f"valjevo_ceiling_2023 ({len(vdays)} days, max_rate = 364.63)", int(cand.sum()), int((cand & k).sum())))
    ddays = {pd.Timestamp(d) for d in ("2019-11-14", "2019-11-15", "2019-11-16", "2019-11-18", "2019-11-21")}
    m = t.day.isin(ddays).values & (t.row.values == 640) & (t.col.values == 1408)
    cand = m & (np.abs(t.rep_value.fillna(0).values - 48.62) <= 0.005) & (t.rep_count.values >= 5)
    out.append(("ceiling_4862_201911 (tile 640/1408, >= 5 px at 48.62)", int(cand.sum()), int((cand & k).sum())))
    return out


def crit3(t, d, rej):
    from make_splits import event_rows
    events = [e for e in yaml.safe_load(open(f"{REPO}/configs/prominent_events.yaml"))["events"]
              if e.get("available", True) is not False]
    ev = event_rows(t, events)
    wet = t["max"].values >= 1.0
    a = []
    for e in sorted(set(ev) - {""}):
        m = (ev == e) & wet
        if m.sum():
            a.append((e, int(m.sum()), rej[m].mean()))
    reasons = ["unphysical", "rej_ceiling", "rej_repeat", "rej_refused", "rej_radar_frame", "rej_radar_day"]
    mx = t["max"].values
    lo, hi = (mx >= 5) & (mx < 10), mx >= 20
    b = []
    for r in reasons:
        x = t["unphysical"].astype(bool).values if r == "unphysical" else d[r].values
        rl, rh = x[lo].mean(), x[hi].mean()
        b.append((r, rl, rh, rh <= max(3 * rl, 0.005)))
    return a, b, int(lo.sum()), int(hi.sum())


def block_boot(g, flag, n=1000, seed=0):
    s = g.assign(f=flag).groupby("day").agg(n=("f", "size"), k=("f", "sum"))
    if not len(s):
        return np.nan, np.nan, np.nan, 0
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    b = s.k.values[idx].sum(1) / np.maximum(s.n.values[idx].sum(1), 1)
    return s.k.sum() / s.n.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5), int(s.n.sum())


def crit2():
    fs = sorted(glob.glob(f"{GAUGE}/g_*.csv.gz"))
    g = pd.concat([pd.read_csv(f, dtype={"day": str}) for f in fs], ignore_index=True)
    g = g.dropna(subset=["raw", "full", "g_shift"])
    L = [f"{len(fs)} gauge days, {len(g):,} gauge-hours, {g.station.nunique():,} stations (g_shift window)", "",
         "| u (mm) | field | n radar >= u | false alarms (gauge < 1 mm) [95% CI] | n gauge >= u | "
         "screen-caused misses (<= 0.5 raw) | overall misses (< 0.2 u) |", "|---|---|---|---|---|---|---|"]
    res = {}
    for u in U_ALL:
        G = g[g.g_shift >= u]
        sm = ((G.full <= 0.5 * G.raw) & (G.raw > 0)).mean() if len(G) else np.nan
        for fld in ("raw", "full"):
            R = g[g[fld] >= u]
            p, lo, hi, n = block_boot(R, (R.g_shift < 1.0).values)
            om = (G[fld] < 0.2 * u).mean() if len(G) else np.nan
            L.append(f"| {u:g} | {fld} | {n:,} | {p:.1%} [{lo:.1%}, {hi:.1%}] | {len(G):,} | "
                     f"{sm:.1%} | {om:.1%} |" if fld == "full" else
                     f"| {u:g} | {fld} | {n:,} | {p:.1%} [{lo:.1%}, {hi:.1%}] | {len(G):,} | - | {om:.1%} |")
            if fld == "full":
                res[u] = (p, hi, sm)
    p, hi, sm = res[U_MAIN]
    L += ["", f"Criterion 2a (u = {U_MAIN:g} mm): false-alarm upper bound {hi:.1%} -> {'PASS' if hi <= 0.05 else 'FAIL'} (<= 5%)",
          f"Criterion 2b (u = {U_MAIN:g} mm): screen-caused misses {sm:.1%} -> {'PASS' if sm <= 0.01 else 'FAIL'} (<= 1%)"]
    return L


def main():
    q = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))
    ff = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    flags = set(zip(ff.radar, ff.timestamp))
    cal = pd.read_csv(f"{CAL}/days.csv", dtype=str)
    t, nd = load_hours()
    t["n_burst"] = t.get("n_burst", 0)
    L = ["# Pre-build readiness gate (DECISIONS §29)", "",
         f"Calibration days rescanned: {nd} ({len(t):,} tile-hours); frame flags: {len(flags):,}", ""]
    for name, de in rule5_sets(q).items():
        d = decide_hourly(t, q, frame_flags=flags, day_excl=de)
        rej = t["unphysical"].astype(bool).values | d["rej_v4"].values
        L += [f"## Rule 5: {name} ({len(de):,} excluded radar-days); tile-hours rejected {rej.mean():.2%}", "",
              "Criterion 1, known failures (candidates / kept; pass = 0 kept):", ""]
        c1 = crit1(t, rej, cal)
        L += [f"- {n}: {a} candidates, {k} kept -> {'PASS' if k == 0 else 'FAIL'}" for n, a, k in c1]
        a, b, nlo, nhi = crit3(t, d, rej)
        big = [x for x in a if x[1] >= 50]
        worst = max(big, key=lambda x: x[2]) if big else None
        L += ["", f"Criterion 3a, event tile-hours rejected (< 0.5% per event with >= 50 wet tile-hours): "
              + (f"worst {worst[0]} {worst[2]:.2%} of {worst[1]} -> {'PASS' if worst[2] < 0.005 else 'FAIL'}" if worst else "no event"),
              "", "| event | wet tile-hours | rejected |", "|---|---|---|"]
        L += [f"| {e} | {n} | {r:.2%} |" for e, n, r in sorted(a, key=lambda x: -x[2])]
        L += ["", f"Criterion 3b, rejection rate by intensity (5-10 mm: {nlo:,} tile-hours; >= 20 mm: {nhi:,}); "
              "pass: rate at >= 20 <= max(3 x rate at 5-10, 0.5%)", "",
              "| reason | 5-10 mm | >= 20 mm | pass |", "|---|---|---|---|"]
        L += [f"| {r} | {lo:.2%} | {hi:.2%} | {'PASS' if ok else 'FAIL'} |" for r, lo, hi, ok in b]
        L.append("")
    L += ["## Criterion 2, gauges (pixel values of the chain; rule 5 and tile rejection not applied)", ""] + crit2()
    txt = "\n".join(L)
    print(txt)
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
