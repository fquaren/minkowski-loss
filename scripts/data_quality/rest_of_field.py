#!/usr/bin/env python
"""Is the rest of an excluded radar-day's field worse than normal? (DECISIONS §28, test (i))

German radars, screened hourly field C of the current chain on the RADKLIM box, against RADKLIM.
  test     radar-days excluded with reason iso31 (score >= 1)
  control  non-excluded radar-days of the same radars on the RADKLIM days frozen before the fetch
           (`iso31_radklim/control_days.txt`)
Pixels owned by the radar (contributing geometry); pixel-hours holding an isolated pixel-frame
(`iso31_radklim/days`) are left out.

  run      per radar-day counts: C >= 5/10/20 mm and how many of them are RADKLIM-dry
           (K_tol < 5 mm); K >= 5 mm and how many have C < 1 mm; log-correlation sums
  summary  false-alarm share test vs control with a 95% day-block bootstrap of the difference;
           "worse" if > 2 points and the interval lies above 0

    python scripts/data_quality/rest_of_field.py run --workers 3
    python scripts/data_quality/rest_of_field.py summary
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
from radklim_presence import k_tolerant  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
RKD = f"{D}/validation/radklim/hourly"
ISO = f"{Q4}/calib/iso31_radklim"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{Q4}/calib/rest_of_field"
QPATH = f"{REPO}/configs/quality_v4.yaml"
US = (5, 10, 20)


def radar_days():
    r = pd.read_csv(f"{Q4}/ranking.csv.gz", dtype={"day": str}, usecols=["radar", "day", "score", "reason"])
    de = r[r.radar.str.startswith("de")]
    have = {os.path.basename(f)[:8] for f in glob.glob(f"{RKD}/*.npz")}
    test = de[(de.score >= 1) & (de.reason == "iso31") & de.day.isin(have)][["radar", "day"]]
    cdays = set(open(f"{ISO}/control_days.txt").read().split())
    control = de[(de.score < 1) & de.radar.isin(set(test.radar)) & de.day.isin(cdays)][["radar", "day"]]
    return test.assign(group="test"), control.assign(group="control")


def run_day(args):
    day, groups = args                                   # groups: {radar: "test" | "control"}
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import RadarGeometry, use_contributing
    t0 = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (y0, y1, x0, x1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, 0, 0.0
    Kt = k_tolerant(K24)
    q, ceil = st._quality(QPATH)
    rg = q["radar_geometry"]
    use_contributing(rg["nodes_table"], rg.get("valid_climatology"))
    g = RadarGeometry(int(day[:4]))
    owner = np.full(2200 * 1900, -1, np.int64)
    owner[g.own_idx] = g.own
    owner = owner.reshape(2200, 1900)[y0:y1, x0:x1]
    key_i = {k: i for i, k in enumerate(g.keys)}
    drops = {}                                           # hour -> mask of isolated pixel-hours
    f = f"{ISO}/days/{day}.csv.gz"
    if os.path.exists(f):
        x = pd.read_csv(f)
        x = x[x.row.between(y0, y1 - 1) & x.col.between(x0, x1 - 1)]
        for hh, g_ in x.groupby("hour"):
            m_ = np.zeros((y1 - y0, x1 - x0), bool)
            m_[g_.row.values - y0, g_.col.values - x0] = True
            drops[int(hh)] = m_
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q.get("hourly", {}), crop=(y0, y1, x0, x1)).run()
    d1 = np.datetime64(pd.Timestamp(day)) + np.timedelta64(1, "h")
    acc = {k: {c: 0.0 for c in ["n_C5", "fa_C5", "n_C10", "fa_C10", "n_C20", "fa_C20", "n_K5", "miss_K5",
                                 "nw", "sx", "sy", "sxx", "syy", "sxy"]} for k in groups}
    for h in hd.hours:
        i = int((h["end"] - d1) / np.timedelta64(1, "h"))
        if not 0 <= i < 24:
            continue
        C, K, T = h["sum"], K24[i], Kt[i]
        ok = np.isfinite(C) & np.isfinite(K) & np.isfinite(T)
        if i in drops:
            ok &= ~drops[i]
        for k in groups:
            if k not in key_i:
                continue
            m = ok & (owner == key_i[k])
            if not m.any():
                continue
            c, kk, t = C[m], K[m], T[m]
            a = acc[k]
            for u in US:
                a[f"n_C{u}"] += int((c >= u).sum())
                a[f"fa_C{u}"] += int(((c >= u) & (t < 5)).sum())
            a["n_K5"] += int((kk >= 5).sum())
            a["miss_K5"] += int(((kk >= 5) & (c < 1)).sum())
            w = np.maximum(c, kk) >= 0.1
            x_, y_ = np.log1p(c[w]), np.log1p(kk[w])
            a["nw"] += int(w.sum()); a["sx"] += x_.sum(); a["sy"] += y_.sum()
            a["sxx"] += (x_ * x_).sum(); a["syy"] += (y_ * y_).sum(); a["sxy"] += (x_ * y_).sum()
    rows = [{"day": day, "radar": k, "group": grp, **acc[k]} for k, grp in groups.items()]
    os.makedirs(f"{OUT}/days", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/days/{day}.csv", index=False)
    return day, len(rows), time.time() - t0


def summary():
    df = pd.concat([pd.read_csv(f, dtype={"day": str}) for f in sorted(glob.glob(f"{OUT}/days/*.csv"))])
    rng = np.random.default_rng(0)
    L = ["# Rest of the field on excluded radar-days (DECISIONS §28, test (i))", "",
         f"radar-days: test {int((df.group == 'test').sum())}, control {int((df.group == 'control').sum())}; "
         f"radars: {sorted(df[df.group == 'test'].radar.unique())}", "",
         "| measure | test | control | difference [95% CI] |", "|---|---|---|---|"]
    days = df.day.unique()
    by = {d: g for d, g in df.groupby("day")}
    verdict = None
    for num, den, lab in [(f"fa_C{u}", f"n_C{u}", f"false alarms, C >= {u} mm (K_tol < 5)") for u in US] + \
                         [("miss_K5", "n_K5", "misses, K >= 5 mm with C < 1 mm")]:
        def share(sub, grp):
            s = sub[sub.group == grp]
            return s[num].sum() / max(s[den].sum(), 1)
        pt, pc = share(df, "test"), share(df, "control")
        bs = []
        for _ in range(1000):
            sub = pd.concat([by[d] for d in rng.choice(days, len(days))])
            bs.append(share(sub, "test") - share(sub, "control"))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        nt, nc = int(df[df.group == "test"][den].sum()), int(df[df.group == "control"][den].sum())
        L.append(f"| {lab} | {pt:.1%} (n={nt:,}) | {pc:.1%} (n={nc:,}) | {pt - pc:+.1%} [{lo:+.1%}, {hi:+.1%}] |")
        if num == "fa_C5":
            verdict = (pt - pc > 0.02) and lo > 0
    def corr(r):
        n = r.nw
        if n < 50:
            return np.nan
        cov = r.sxy / n - r.sx * r.sy / n ** 2
        vx, vy = r.sxx / n - (r.sx / n) ** 2, r.syy / n - (r.sy / n) ** 2
        return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
    df["r"] = df.apply(corr, axis=1)
    L += ["", f"median per radar-day log-correlation (>= 50 wet pixel-hours): test "
          f"{df[df.group == 'test'].r.median():.2f}, control {df[df.group == 'control'].r.median():.2f}", "",
          f"**Verdict (C >= 5 mm false alarms): {'WORSE: exclusion justified' if verdict else 'NOT WORSE: exclusion is a cost'}**"]
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "summary":
        summary()
        return
    t, c = radar_days()
    rd = pd.concat([t, c])
    jobs = [(d, dict(zip(g.radar, g.group))) for d, g in rd.groupby("day")
            if not os.path.exists(f"{OUT}/days/{d}.csv")]
    print(f"[rest_of_field] {len(jobs)} days, {len(t)} test and {len(c)} control radar-days", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, s in ex.map(run_day, jobs):
            print(f"  {day}: {n} radar-days, {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
