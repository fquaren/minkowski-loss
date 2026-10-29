#!/usr/bin/env python
"""RADKLIM presence tests of the screening review (DECISIONS §30), and the German distribution
data for the slides.

On the calibration days with RADKLIM, RADKLIM box, the hourly chain runs in six variants:
  base           current quality_v4.yaml
  no_hot         static-clutter repair off
  hot_as_prior   repair off, hot pixels only lower the persistence threshold (p1_prior_min 12)
  no_unsup       no-temporal-support rule off
  p1_20, p1_16   p1_min 20 / 16 instead of 24
  run      per day, pixel-hours with max(raw A, RADKLIM K) >= 5 mm: A, K, K_tol, rejected
           (split time, current rule 5), hour, row, col and C of every variant
  summary  per test, removals (kept, C_without >= 1, C_with <= 0.5 C_without) by raw bin:
           n, RADKLIM-dry share (K_tol < 5), confirmed (K_tol >= 0.5 A); pass = dry >= 85% in
           every bin with n >= 30 (p1 tests on both halves of the days). Plus exceedance counts
           of A, screened C and K on the same pixel-hours.

    python scripts/data_quality/rule_validate.py run --workers 5
    python scripts/data_quality/rule_validate.py summary
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
CAL = f"{Q4}/calib"
RKD = f"{D}/validation/radklim/hourly"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{CAL}/rule_validate"
QPATH = f"{REPO}/configs/quality_v4.yaml"
VARIANTS = ["base", "no_hot", "hot_as_prior", "no_unsup", "p1_20", "p1_16"]
TESTS = {  # name: (without, with, split into halves)
    "hot repair": ("no_hot", "base", False),
    "hot as persistence prior (replacement)": ("no_hot", "hot_as_prior", False),
    "no temporal support": ("no_unsup", "base", False),
    "p1_min 20 (added removals)": ("base", "p1_20", True),
    "p1_min 16 (added removals)": ("base", "p1_16", True),
}
# further variant sets (--set): output directory, variants, tests
SETS = {
    "stationary": (f"{CAL}/rule_validate_stationary", ["base", "stat_12", "stat_8"], {   # DECISIONS §31
        "stationary echo, min_frames 12": ("base", "stat_12", True),
        "stationary echo, min_frames 8": ("base", "stat_8", True),
    }),
}
BINS = [10, 20, 50, 100, np.inf]
LABELS = ["10-20", "20-50", "50-100", ">=100"]


def variant_cfg(q, v):
    cfg = {**q["hourly"], "rules": dict(q["hourly"]["rules"])}
    if v in ("no_hot", "hot_as_prior"):
        cfg["rules"]["hot"] = False
    if v == "hot_as_prior":
        cfg["p1_prior_min"] = 12
    if v == "no_unsup":
        cfg["rules"]["unsupported"] = False
    if v.startswith("p1_"):
        cfg["p1_min"] = int(v[3:])
    if v.startswith("stat_"):
        cfg["rules"]["stationary"] = True
        cfg["stationary"] = {**cfg.get("stationary", {}), "min_frames": int(v[5:])}
    return cfg


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import decide_hourly
    t0 = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (y0, y1, x0, x1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, 0, 0.0
    Kt = k_tolerant(K24)
    q, ceil = st._quality(QPATH)
    tab = pd.read_csv(f"{CAL}/hourly_c/hours/{day}.csv.gz", dtype={"end": str, "frames": str, "owners": str})
    if "n_burst" not in tab:
        tab["n_burst"] = 0
    ff = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    de = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    d = decide_hourly(tab, q, frame_flags=set(zip(ff.radar, ff.timestamp)), day_excl=set(zip(de.radar, de.day)))
    rej = tab["unphysical"].astype(bool).values | d["rej_v4"].values
    rset = {(e[:12], r, c) for e, r, c, x in zip(tab.end, tab.row, tab.col, rej) if x}
    runs = {}
    for v in VARIANTS:
        prior = (lambda y: st._hot(CLIM, y)) if v == "hot_as_prior" else None
        runs[v] = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),   # off via rules
                            ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                            cfg=variant_cfg(q, v), crop=(y0, y1, x0, x1), prior=prior).run()
    d1 = np.datetime64(pd.Timestamp(day)) + np.timedelta64(1, "h")
    by = {v: {int((h["end"] - d1) / np.timedelta64(1, "h")): h for h in r.hours} for v, r in runs.items()}
    cols = {k: [] for k in ["A", "K", "K_tol", "rejected", "hour", "row", "col", *VARIANTS]}
    for i, h in by["base"].items():
        if not 0 <= i < 24:
            continue
        A, k = h["raw_sum"], K24[i]
        m = np.isfinite(A) & np.isfinite(k) & (np.maximum(A, k) >= 5.0)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        rr, cc = ys + y0, xs + x0
        end = np.datetime_as_string(h["end"], unit="m").replace("-", "").replace("T", "").replace(":", "")
        cols["A"].append(A[m]); cols["K"].append(k[m]); cols["K_tol"].append(Kt[i][m])
        cols["hour"].append(np.full(m.sum(), i, np.int8))
        cols["row"].append(rr.astype(np.int16)); cols["col"].append(cc.astype(np.int16))
        cols["rejected"].append(np.array([(end, (r // 128) * 128, (c // 128) * 128) in rset for r, c in zip(rr, cc)]))
        for v in VARIANTS:
            hv = by[v].get(i)
            cols[v].append(hv["sum"][m] if hv is not None else np.full(m.sum(), np.nan, np.float32))
    arr = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    os.makedirs(f"{OUT}/days", exist_ok=True)
    np.savez_compressed(f"{OUT}/days/{day}.npz", **arr)
    return day, len(arr["A"]), time.time() - t0


def load():
    parts = []
    for f in sorted(glob.glob(f"{OUT}/days/*.npz")):
        z = dict(np.load(f))
        if not len(z["A"]):
            continue
        parts.append(pd.DataFrame(z).assign(day=os.path.basename(f)[:8]))
    df = pd.concat(parts, ignore_index=True)
    cat = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day").category
    df["group"] = df.day.map(cat).map(lambda c: "random" if c == "random" else "event" if c.startswith("event") else "failure")
    return df


def test_table(df, wo, w):
    k = df[~df.rejected.astype(bool) & df.K_tol.notna()]
    rm = k[(k[wo] >= 1) & (k[w] <= 0.5 * k[wo])]
    b = pd.cut(rm.A, BINS, labels=LABELS, right=False)
    rows, ok = [], True
    for lab in LABELS:
        x = rm[b == lab]
        n = len(x)
        dry = (x.K_tol < 5).mean() if n else np.nan
        conf = int((x.K_tol >= 0.5 * x.A).sum())
        p = bool(dry >= 0.85) if n >= 30 else None
        ok &= p is not False
        rows.append((lab, n, dry, conf, p))
    return rows, ok


def summary():
    df = load()
    L = ["# RADKLIM presence tests of the screening review (DECISIONS §30)", "",
         f"{df.day.nunique()} days, {len(df):,} pixel-hours with max(A, K) >= 5 mm; "
         f"rejected at split time {df.rejected.astype(bool).mean():.2%}", ""]
    for name, (wo, w, halves) in TESTS.items():
        subsets = [("all days", df)]
        if halves:
            idx = {d: i for i, d in enumerate(sorted(df.day.unique()))}
            par = df.day.map(idx) % 2
            subsets = [("odd days", df[par == 1]), ("even days", df[par == 0])]
        verdict = True
        L += [f"## {name}: removals of `{w}` relative to `{wo}`", ""]
        for sname, sub in subsets:
            rows, ok = test_table(sub, wo, w)
            verdict &= ok
            L += [f"{sname}:", "", "| raw bin | removals | RADKLIM-dry | confirmed (K_tol >= 0.5 A) | pass |",
                  "|---|---|---|---|---|"]
            L += [f"| {lab} mm | {n:,} | {'-' if np.isnan(d) else f'{d:.1%}'} | {c} | "
                  f"{'n < 30' if p is None else 'PASS' if p else 'FAIL'} |" for lab, n, d, c, p in rows]
            L.append("")
        L += [f"**Verdict: {'PASS' if verdict else 'FAIL'}**", ""]
    kept = df[~df.rejected.astype(bool)]
    xs = [5, 10, 20, 30, 50, 75, 100, 150]
    L += ["## Exceedance counts on the same German pixel-hours (screened = base, kept tile-hours)", "",
          "| field | " + " | ".join(f">= {x}" for x in xs) + " |", "|---|" + "---|" * len(xs)]
    for lab, s in [("raw OPERA", df.A), ("screened OPERA", kept.base), ("RADKLIM", df.K)]:
        L.append(f"| {lab} | " + " | ".join(f"{int((s >= x).sum()):,}" for x in xs) + " |")
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--set", default="review", choices=["review", *SETS])
    a = ap.parse_args()
    if a.set != "review":                  # module globals, inherited by the forked workers
        global OUT, VARIANTS, TESTS
        OUT, VARIANTS, TESTS = SETS[a.set]
    if a.cmd == "summary":
        summary()
        return
    cal = set(pd.read_csv(f"{CAL}/days.csv", dtype=str).day)
    days = a.days or sorted(cal & {os.path.basename(f)[:8] for f in glob.glob(f"{RKD}/*.npz")})
    days = [d for d in days if not os.path.exists(f"{OUT}/days/{d}.npz")]
    print(f"[rule_validate] {len(days)} days", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, s in ex.map(run_day, days):
            print(f"  {day}: {n:,} pixel-hours, {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
