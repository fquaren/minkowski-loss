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
    pre = pd.read_csv(f"{Q4}/day_exclusions_20261013_prestated.csv", dtype=str)   # §28 as first fixed
    r = pd.read_csv(f"{Q4}/ranking.csv.gz", dtype={"day": str}, usecols=["radar", "day", "year", "score"])
    p = q["radar_day"]
    r["e"] = r.score >= p["exclude_score"]
    y = r.groupby(["radar", "year"]).agg(n=("day", "size"), k=("e", "sum")).reset_index()
    y = y[(y.k / y.n >= p["year_share"]) & (y.k >= p["year_min_days"])]
    cor = pd.concat([r.loc[r.e, ["radar", "day"]], r.merge(y[["radar", "year"]], on=["radar", "year"])[["radar", "day"]]])
    cur = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)                      # adopted 2026-10-14
    return {"off": set(), "pre-stated": set(zip(pre.radar, pre.day)), "days-with-data": set(zip(cor.radar, cor.day)),
            "score only (adopted 2026-10-14)": set(zip(cur.radar, cur.day))}


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
    # (name, tile-hours where the failure is in the raw field (context), screened candidates, kept)
    cand = m & (sh >= 0.10) & (t.max_rate.values >= 31)
    ctx = m & (sh >= 0.10) & (t.raw_max_rate.values >= 31)
    out.append(("madrid_20180429 (estjv >= 10%, max_rate >= 31)", int(ctx.sum()), int(cand.sum()), int((cand & k).sum())))
    vdays = {pd.Timestamp(d) for d in cal[cal.category.str.contains("valjevo")].day}
    m = t.day.isin(vdays).values
    cand = m & (np.abs(t.max_rate.values - 364.63) <= 0.005)
    ctx = m & (np.abs(t.raw_max_rate.values - 364.63) <= 0.005)
    out.append((f"valjevo_ceiling_2023 ({len(vdays)} days, max_rate = 364.63)", int(ctx.sum()), int(cand.sum()), int((cand & k).sum())))
    ddays = {pd.Timestamp(d) for d in ("2019-11-14", "2019-11-15", "2019-11-16", "2019-11-18", "2019-11-21")}
    m = t.day.isin(ddays).values & (t.row.values == 640) & (t.col.values == 1408)
    cand = m & (np.abs(t.rep_value.fillna(0).values - 48.62) <= 0.005) & (t.rep_count.values >= 5)
    ctx = m & (t.n_ceiling.values > 0)          # raw value table not in the scan; ceiling pixels coded
    out.append(("ceiling_4862_201911 (tile 640/1408, >= 5 px at 48.62)", int(ctx.sum()), int(cand.sum()), int((cand & k).sum())))
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


def crit2_floor(q, flags):
    """Criterion 2a against RADKLIM (DECISIONS §29 follow-up (ii), 2026-10-14): DWD gauge-hours on
    days with RADKLIM, tile-hours rejected at split time (current day_exclusions.csv) dropped;
    false-alarm share (field >= u, gauge < 1 mm) of screened OPERA vs RADKLIM at the gauge pixel.
    Pass: upper end of the 95% day-block interval of (OPERA - RADKLIM) <= 2 points."""
    de = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    de = set(zip(de.radar, de.day))
    st = pd.read_csv(f"{D}/validation/gauges/dwd_stations.csv")[["station", "row", "col"]].dropna()
    rows = []
    for f in sorted(glob.glob(f"{GAUGE}/g_*.csv.gz")):
        day = os.path.basename(f)[2:10]
        rk, hp = f"{D}/validation/radklim/hourly/{day}.npz", None
        for h in (f"{CAL}/hourly_c/hours/{day}.csv.gz", f"{CAL}/hourly_g/hours/{day}.csv.gz"):
            hp = h if os.path.exists(h) else hp
        if not (os.path.exists(rk) and hp):
            continue
        z = np.load(rk)
        K, (y0, y1, x0, x1) = z["K"], z["box"]
        if not np.isfinite(K).any():
            continue
        g = pd.read_csv(f, dtype={"day": str, "end": str})
        g = g[(g.net == "dwd")].merge(st, on="station")
        t = pd.read_csv(hp, dtype={"end": str, "frames": str, "owners": str})
        if "n_burst" not in t:
            t["n_burst"] = 0
        d = decide_hourly(t, q, frame_flags=flags, day_excl=de)
        rej = t["unphysical"].astype(bool).values | d["rej_v4"].values
        rset = {(e[:12], r, c) for e, r, c, x in zip(t.end, t.row, t.col, rej) if x}
        r_, c_ = g.row.astype(int).values, g.col.astype(int).values
        keep = np.array([(e, (r // 128) * 128, (c // 128) * 128) not in rset for e, r, c in zip(g.end, r_, c_)])
        d1 = pd.Timestamp(day) + pd.Timedelta(hours=1)
        i = ((pd.to_datetime(g.end, format="%Y%m%d%H%M") - d1) / pd.Timedelta(hours=1)).astype(int).values
        inb = (i >= 0) & (i < 24) & (r_ >= y0) & (r_ < y1) & (c_ >= x0) & (c_ < x1)
        k = np.full(len(g), np.nan)
        k[inb] = K[i[inb], r_[inb] - y0, c_[inb] - x0]
        g = g.assign(K=k)[keep & np.isfinite(k)]
        rows.append(g[["day", "end", "station", "full", "K", "g_shift"]])
    g = pd.concat(rows, ignore_index=True).dropna()
    by = {d_: x for d_, x in g.groupby("day")}
    days = list(by)
    rng = np.random.default_rng(0)
    L = [f"DWD gauge-hours on {len(days)} days with RADKLIM, rejected tile-hours dropped: {len(g):,}", "",
         "| u (mm) | OPERA screened: n, false alarms | RADKLIM: n, false alarms | OPERA - RADKLIM [95% CI] |",
         "|---|---|---|---|"]
    out = None
    for u in U_ALL:
        def fa(x, col):
            s = x[x[col] >= u]
            return (s.g_shift < 1.0).sum() / max(len(s), 1), len(s)
        (po, no), (pr, nr) = fa(g, "full"), fa(g, "K")
        bs = []
        for _ in range(1000):
            x = pd.concat([by[d_] for d_ in rng.choice(days, len(days))])
            bs.append(fa(x, "full")[0] - fa(x, "K")[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append(f"| {u:g} | {no:,}, {po:.1%} | {nr:,}, {pr:.1%} | {po - pr:+.1%} [{lo:+.1%}, {hi:+.1%}] |")
        if u == U_MAIN:
            out = hi
    L += ["", f"Criterion 2a (revised, u = {U_MAIN:g} mm): upper bound of OPERA - RADKLIM {out:+.1%} -> "
          f"{'PASS' if out <= 0.02 else 'FAIL'} (<= +2 points)"]
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
              "Criterion 1, known failures (pass = 0 kept; 'in raw' shows the check is not vacuous):", ""]
        c1 = crit1(t, rej, cal)
        L += [f"- {n}: {c} tile-hours with it in raw / coded, {a} after the chain, {k} kept -> "
              f"{'PASS' if k == 0 else 'FAIL'}{' (vacuous: not present in raw)' if c == 0 else ''}" for n, c, a, k in c1]
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
    L += ["", "## Criterion 2a revised: false alarms against RADKLIM on the same gauge-hours", ""] + crit2_floor(q, flags)
    txt = "\n".join(L)
    print(txt)
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
