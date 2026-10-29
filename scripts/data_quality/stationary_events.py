#!/usr/bin/env python
"""Stationary-echo rule on real storms outside Germany (DECISIONS §31, criterion 2).

On the event calibration days, full grid: the current chain (base) and the chain with the rule
at min_frames 12 and 8. Counts the pixel-hours that base leaves at >= 20 mm and how many of them
the rule lowers by >= 50%. Pass: <= 1% for a candidate. Also which events are touched.

    python scripts/data_quality/stationary_events.py run --workers 3
    python scripts/data_quality/stationary_events.py summary
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
CAL = f"{D}/quality_v4/calib"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{CAL}/stationary_events"
QPATH = f"{REPO}/configs/quality_v4.yaml"
CANDS = (12, 8)
U = 20.0


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    t0 = time.time()
    q, ceil = st._quality(QPATH)
    sums = {}
    for mf in (None, *CANDS):
        cfg = {**q["hourly"], "rules": {**q["hourly"]["rules"], "stationary": mf is not None},
               "stationary": {"min_frames": mf or CANDS[0]}}
        hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                       ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil, cfg=cfg).run()
        sums[mf] = {np.datetime_as_string(h["end"], unit="m"): h["sum"] for h in hd.hours}
    row = {"day": day, "n_base": 0}
    tiles = {mf: {} for mf in CANDS}
    for end, b in sums[None].items():
        m = np.nan_to_num(b, nan=0.0) >= U
        row["n_base"] += int(m.sum())
        for mf in CANDS:
            s = sums[mf].get(end)
            low = m & (np.nan_to_num(s, nan=0.0) <= 0.5 * b) if s is not None else np.zeros_like(m)
            row[f"low_{mf}"] = row.get(f"low_{mf}", 0) + int(low.sum())
            ys, xs = np.nonzero(low)
            for y, x in zip(ys, xs):                      # where: tile of each lowered pixel-hour
                k = (end, (y // 128) * 128, (x // 128) * 128)
                tiles[mf][k] = tiles[mf].get(k, 0) + 1
    os.makedirs(f"{OUT}/days", exist_ok=True)
    pd.DataFrame([row]).to_csv(f"{OUT}/days/{day}.csv", index=False)
    pd.DataFrame([{"min_frames": mf, "end": k[0], "row": k[1], "col": k[2], "n": n}
                  for mf in CANDS for k, n in tiles[mf].items()]).to_csv(f"{OUT}/days/{day}_tiles.csv", index=False)
    return day, row, time.time() - t0


def summary():
    cat = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day").category
    df = pd.concat([pd.read_csv(f, dtype={"day": str}) for f in sorted(glob.glob(f"{OUT}/days/*[0-9].csv"))])
    df["event"] = df.day.map(cat).str.replace("event:", "", regex=False)
    L = ["# Stationary-echo rule on event days (DECISIONS §31, criterion 2)", "",
         f"{len(df)} event days; pixel-hours left >= {U:g} mm by the current chain: {int(df.n_base.sum()):,}", ""]
    for mf in CANDS:
        share = df[f"low_{mf}"].sum() / max(df.n_base.sum(), 1)
        L.append(f"- min_frames {mf}: lowered by >= 50%: {int(df[f'low_{mf}'].sum()):,} "
                 f"({share:.3%}) -> {'PASS' if share <= 0.01 else 'FAIL'} (<= 1%)")
    by = df.groupby("event")[["n_base", *[f"low_{mf}" for mf in CANDS]]].sum()
    by = by[by[[f"low_{mf}" for mf in CANDS]].sum(axis=1) > 0]
    if len(by):
        L += ["", "Events touched:", "", "| event | pixel-hours >= 20 mm | " +
              " | ".join(f"lowered (min_frames {mf})" for mf in CANDS) + " |", "|---|---|" + "---|" * len(CANDS)]
        L += [f"| {e} | {int(r.n_base):,} | " + " | ".join(f"{int(r[f'low_{mf}']):,}" for mf in CANDS) + " |"
              for e, r in by.iterrows()]
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.cmd == "summary":
        summary()
        return
    cat = pd.read_csv(f"{CAL}/days.csv", dtype=str)
    days = sorted(cat[cat.category.str.startswith("event")].day)
    days = [d for d in days if not os.path.exists(f"{OUT}/days/{d}.csv")][::-1 if a.reverse else 1]
    print(f"[stationary_events] {len(days)} days", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, row, s in ex.map(run_day, days):
            print(f"  {day}: {row}, {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
