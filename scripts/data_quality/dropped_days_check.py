#!/usr/bin/env python
"""Are the 20 days that break gate criterion 3 contaminated? (DECISIONS §32, option (b))

Dropped days (from gate_tail/summary.md) against 60 random non-dropped, non-event test days.
On the test split as stored, pixel-hours with screened C >= 20.2 mm:
  RADKLIM  share RADKLIM-dry (K_tol < 5 mm), Germany
  gauges   share with gauge < 1 mm, among gauge-hours in test tile-hours with C >= 20.2 (DE + CH)
Contaminated: dropped - control > 2 points with the 95% day-block interval above 0.
Descriptive: the gauges' own exceedance frequency (>= 20.2 mm) on both sets of days.

    python scripts/data_quality/dropped_days_check.py days      # writes the day lists
    (fetch RADKLIM for all.txt; gauge_hourly.py run --out <OUT>/gauges --variants full)
    python scripts/data_quality/dropped_days_check.py analyze
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.dirname(__file__)]
from radklim_presence import k_tolerant  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
STORE = f"{D}/patches_v4/preprocessed_dataset.zarr"
META = f"{D}/patches_v4/meta"
OUT = f"{Q4}/gate_tail/dropped_check"
U = 20.2


def test_info():
    info = pd.read_csv(f"{META}/full_test_info.csv.gz", dtype={"timestamp": str, "event": str})
    ts = pd.to_datetime(info.timestamp, format="%Y%m%d%H%M%S")
    info["pday"] = (ts - pd.Timedelta(minutes=1)).dt.strftime("%Y%m%d")
    info["end"] = info.timestamp.str[:12]
    return info


def cmd_days():
    txt = open(f"{Q4}/gate_tail/summary.md").read()
    dropped = re.search(r"of them: ([0-9, ]+)\.", txt).group(1).replace(" ", "").split(",")
    info = test_info()
    ev = set(info.pday[info.event.fillna("") != ""])
    pool = sorted(set(info.pday) - ev - set(dropped))
    control = sorted(np.random.default_rng(0).choice(pool, 60, replace=False))
    os.makedirs(OUT, exist_ok=True)
    for name, d in (("dropped", dropped), ("control", control), ("all", sorted(dropped + list(control)))):
        open(f"{OUT}/{name}.txt", "w").write("\n".join(d) + "\n")
    print(f"dropped {len(dropped)}, control {len(control)} (from {len(pool)} candidate days)")


def radklim_rows(info, days):
    import zarr
    z = zarr.open(f"{STORE}/test/original_precip", mode="r")
    out = []
    for day in days:
        rk = f"{D}/validation/radklim/hourly/{day}.npz"
        if not os.path.exists(rk):
            continue
        f = np.load(rk)
        K, (y0, y1, x0, x1) = f["K"], f["box"]
        if not np.isfinite(K).any():
            continue
        Kt = k_tolerant(K)
        sub = info[(info.pday == day) & (info["max"] >= U)]
        d1 = pd.Timestamp(day) + pd.Timedelta(hours=1)
        for r in sub.itertuples():
            i = int((pd.Timestamp(r.timestamp[:12]) - d1) / pd.Timedelta(hours=1))
            if not 0 <= i < 24:
                continue
            a = z[r.store_row]
            ys, xs = np.nonzero(a >= U)
            yy, xx = ys + r.row, xs + r.col
            m = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
            if not m.any():
                continue
            kt = Kt[i, yy[m] - y0, xx[m] - x0]
            ok = np.isfinite(kt)
            out.append(pd.DataFrame({"day": day, "C": a[ys[m], xs[m]][ok], "K_tol": kt[ok]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["day", "C", "K_tol"])


def gauge_rows(info, days):
    import gauge_hourly as gh
    st = gh.stations()[["station", "net", "row", "col"]]
    keys = set(zip(info.end, info.row, info.col))
    out = []
    for day in days:
        f = f"{OUT}/gauges/g_{day}.csv.gz"
        if not os.path.exists(f):
            continue
        g = pd.read_csv(f, dtype={"end": str, "station": str}).merge(st, on=["station", "net"])
        tile = [(e, int(r) // 128 * 128, int(c) // 128 * 128) in keys for e, r, c in zip(g.end, g.row, g.col)]
        out.append(g.assign(day=day, in_test=tile)[["day", "net", "full", "g_shift", "in_test"]])
    return pd.concat(out, ignore_index=True)


def boot_diff(df, flag, groups, B=1000, seed=0):
    """Share difference (dropped - control) with a day-block bootstrap within each group."""
    rng = np.random.default_rng(seed)
    agg = df.assign(f=flag).groupby(["grp", "day"]).f.agg(["sum", "count"])
    sh = {g: agg.loc[g] for g in groups}
    pt = [sh[g]["sum"].sum() / max(sh[g]["count"].sum(), 1) for g in groups]
    bs = []
    for _ in range(B):
        v = []
        for g in groups:
            s = sh[g].iloc[rng.integers(0, len(sh[g]), len(sh[g]))]
            v.append(s["sum"].sum() / max(s["count"].sum(), 1))
        bs.append(v[0] - v[1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    n = [int(sh[g]["count"].sum()) for g in groups]
    return pt, (lo, hi), n


def cmd_analyze():
    info = test_info()
    grp = {d: "dropped" for d in open(f"{OUT}/dropped.txt").read().split()}
    grp.update({d: "control" for d in open(f"{OUT}/control.txt").read().split()})
    G = ("dropped", "control")
    L = ["# Contamination test of the days that break gate criterion 3 (DECISIONS §32)", ""]
    verdict = []
    rk = radklim_rows(info, sorted(grp)).assign(grp=lambda x: x.day.map(grp))
    if len(rk) and rk.grp.nunique() == 2:
        (pd_, pc), (lo, hi), n = boot_diff(rk, (rk.K_tol < 5).values, G)
        bad = (pd_ - pc > 0.02) and lo > 0
        verdict.append(bad)
        L += [f"**RADKLIM (Germany):** pixel-hours with C >= {U} mm that are RADKLIM-dry: dropped {pd_:.1%} "
              f"(n = {n[0]:,}, {rk[rk.grp == 'dropped'].day.nunique()} days), control {pc:.1%} (n = {n[1]:,}, "
              f"{rk[rk.grp == 'control'].day.nunique()} days); difference {pd_ - pc:+.1%} [{lo:+.1%}, {hi:+.1%}] -> "
              f"{'CONTAMINATED' if bad else 'not contaminated'}", ""]
    g = gauge_rows(info, sorted(grp)).assign(grp=lambda x: x.day.map(grp))
    t = g[g.in_test & (g.full >= U)]
    if len(t) and t.grp.nunique() == 2:
        (pd_, pc), (lo, hi), n = boot_diff(t, (t.g_shift < 1).values, G)
        bad = (pd_ - pc > 0.02) and lo > 0
        verdict.append(bad)
        L += [f"**Gauges (DE + CH):** test gauge-hours with C >= {U} mm and gauge < 1 mm: dropped {pd_:.1%} "
              f"(n = {n[0]:,}), control {pc:.1%} (n = {n[1]:,}); difference {pd_ - pc:+.1%} [{lo:+.1%}, {hi:+.1%}] -> "
              f"{'CONTAMINATED' if bad else 'not contaminated'}", ""]
    fr = g.groupby("grp").apply(lambda x: pd.Series({"gauge_hours": len(x), "gauge_exc": int((x.g_shift >= U).sum()),
                                                     "opera_exc": int((x.full >= U).sum()), "days": x.day.nunique()}))
    fr["gauge freq"] = fr.gauge_exc / fr.gauge_hours
    fr["OPERA freq at gauges"] = fr.opera_exc / fr.gauge_hours
    L += ["*Descriptive:* exceedance frequency (>= 20.2 mm per gauge-hour), all DE/CH gauges:", "",
          "| days | gauge-hours | gauge freq | OPERA freq at the gauges |", "|---|---|---|---|"]
    L += [f"| {k} ({int(r.days)}) | {int(r.gauge_hours):,} | {r['gauge freq']:.2e} | {r['OPERA freq at gauges']:.2e} |"
          for k, r in fr.iterrows()]
    if len(fr) == 2:
        L += ["", f"Ratio dropped / control: gauges {fr.loc['dropped', 'gauge freq'] / fr.loc['control', 'gauge freq']:.1f}x, "
              f"OPERA {fr.loc['dropped', 'OPERA freq at gauges'] / fr.loc['control', 'OPERA freq at gauges']:.1f}x"]
    L += ["", f"**Verdict: {'CONTAMINATED (a named failure type: see above)' if any(verdict) else 'not contaminated'}**"]
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["days", "analyze"])
    a = ap.parse_args()
    cmd_days() if a.cmd == "days" else cmd_analyze()
