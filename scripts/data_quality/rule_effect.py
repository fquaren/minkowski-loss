#!/usr/bin/env python
"""How much each cleaning rule removes, by raw hourly intensity (current v4 chain).

On the calibration days, full grid, current `quality_v4.yaml`. For every pixel-hour with a raw
ACRR sum A >= 1 mm, the raw mass is attributed to the stage that removed it:
  - drizzle, hot (static clutter), spike: inside `clean_frame`, which sets no code bits, so it is
    re-run stage by stage on the same frames (drizzle only -> + hot -> + spikes);
  - ceiling, footprint, ring, ray, persist, unsupported, burst: the frame's drop after
    `clean_frame` goes to the code bits set on that pixel-frame (split equally if several);
  - rejected: the screened remainder of pixel-hours whose tile-hour is rejected at split time
    (`decide_hourly`, frame flags of the radar pass, rule 5 = current `day_exclusions.csv`),
    by reason.
Per raw bin it reports the share of the bin's raw mass each stage removes and the share of
pixel-hours it lowers by >= 50% of A; separately for random, event and failure days.

    python scripts/data_quality/rule_effect.py run --workers 2
    python scripts/data_quality/rule_effect.py summary
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
Q4 = f"{D}/quality_v4"
CAL = f"{Q4}/calib"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{CAL}/rule_effect"
QPATH = f"{REPO}/configs/quality_v4.yaml"
BINS = [1, 5, 10, 20, 50, 100, np.inf]
LABELS = ["1-5", "5-10", "10-20", "20-50", "50-100", ">=100"]
BITS = {"ceiling": 16, "footprint": 1, "ring": 4, "ray": 2, "persist": 128, "unsupported": 8, "burst": 256}
REASONS = ["unphysical", "rej_ceiling", "rej_repeat", "rej_refused", "rej_radar_frame", "rej_radar_day"]
STAGES = ["drizzle", "hot", "spike", *BITS, "other", *REASONS]


def run_day(day):
    import scan_tiles as st
    from src.data import cleaning as C
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import decide_hourly
    t0 = time.time()
    q, ceil = st._quality(QPATH)
    year = int(day[:4])
    hot = st._hot(CLIM, year)
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q.get("hourly", {})).run()
    if not hd.hours:
        return day, 0, 0.0
    # split-time rejection per tile-hour, by first reason
    tab = pd.read_csv(f"{CAL}/hourly_c/hours/{day}.csv.gz", dtype={"end": str, "frames": str, "owners": str})
    ff = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    de = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    if "n_burst" not in tab:
        tab["n_burst"] = 0
    dec = decide_hourly(tab, q, frame_flags=set(zip(ff.radar, ff.timestamp)), day_excl=set(zip(de.radar, de.day)))
    dec["unphysical"] = tab["unphysical"].astype(bool).values
    first = np.full(len(tab), "", object)
    for r in REASONS[::-1]:
        first[dec[r].values] = r
    rej = {(e[:12], int(r), int(c)): f for e, r, c, f in zip(tab.end, tab.row, tab.col, first) if f}
    H, W = hd.raw.shape[1:]
    acc = []
    for h in hd.hours:
        A = h["raw_sum"]
        m = np.isfinite(A) & (A >= 1)
        if not m.any():
            continue
        b = np.digitize(A[m], BINS) - 1
        rem = {s: np.zeros(m.sum(), np.float64) for s in STAGES}
        for j in h["idx"]:
            raw = hd.raw[j]
            r0 = np.nan_to_num(raw, nan=0.0)
            zd = np.nan_to_num(C.clean_frame(raw, None, hot_fallback="keep", spikes=False)[0], nan=0.0)
            zh = np.nan_to_num(C.clean_frame(raw, hot, hot_fallback="keep", spikes=False)[0], nan=0.0)
            zs = np.nan_to_num(C.clean_frame(raw, hot, hot_fallback="keep", spikes=True)[0], nan=0.0)
            fin = np.nan_to_num(hd.fin[j], nan=0.0)
            rem["drizzle"] += 0.25 * np.maximum(r0 - zd, 0)[m]
            rem["hot"] += 0.25 * np.maximum(zd - zh, 0)[m]
            rem["spike"] += 0.25 * np.maximum(zh - zs, 0)[m]
            drop = 0.25 * np.maximum(zs - fin, 0)[m]
            code = hd.code[j][m]
            has = np.stack([(code & bit) > 0 for bit in BITS.values()])          # (rules, n)
            k = has.sum(0)
            for i, s in enumerate(BITS):
                rem[s] += np.where(k > 0, drop * has[i] / np.maximum(k, 1), 0.0)
            rem["other"] += np.where(k == 0, drop, 0.0)
        Cs = np.nan_to_num(h["sum"], nan=0.0)[m]
        end = np.datetime_as_string(h["end"], unit="m").replace("-", "").replace("T", "").replace(":", "")
        ys, xs = np.nonzero(m)
        f = np.array([rej.get((end, (y // 128) * 128, (x // 128) * 128), "") for y, x in zip(ys, xs)])
        for r in REASONS:
            rem[r] = np.where(f == r, Cs, 0.0)
        a = A[m].astype(np.float64)
        row = {"day": day, "end": end}
        for i, lab in enumerate(LABELS):
            s = b == i
            row[f"n|{lab}"] = int(s.sum())
            row[f"mass|{lab}"] = float(a[s].sum())
            for st_ in STAGES:
                row[f"{st_}|mass|{lab}"] = float(rem[st_][s].sum())
                row[f"{st_}|low|{lab}"] = int((rem[st_][s] >= 0.5 * a[s]).sum())
        acc.append(row)
    os.makedirs(f"{OUT}/days", exist_ok=True)
    df = pd.DataFrame(acc)
    num = df.drop(columns=["day", "end"]).sum().to_frame().T
    num.insert(0, "day", day)
    num.to_csv(f"{OUT}/days/{day}.csv", index=False)
    return day, len(acc), time.time() - t0


def summary():
    df = pd.concat([pd.read_csv(f, dtype={"day": str}) for f in sorted(glob.glob(f"{OUT}/days/*.csv"))])
    cat = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day").category
    df["group"] = df.day.map(cat).map(lambda c: c if c in ("random", "event") else "failure")
    L = ["# How much each rule removes, by raw hourly intensity (calibration days, current chain)", "",
         "Cell = share of the bin's raw mass the stage removes; in brackets, share of pixel-hours it "
         "lowers by >= 50%. Rejection rows: screened mass of pixel-hours in rejected tile-hours.", ""]
    for grp in ("random", "event", "failure", "all"):
        g = df if grp == "all" else df[df.group == grp]
        tot = g.drop(columns=["day", "group"]).sum()
        L += [f"## {grp} days ({len(g)})", "",
              "| stage | " + " | ".join(f"{l} mm" for l in LABELS) + " |",
              "|---|" + "---|" * len(LABELS),
              "| pixel-hours | " + " | ".join(f"{int(tot[f'n|{l}']):,}" for l in LABELS) + " |"]
        for s in STAGES:
            cells = []
            for l in LABELS:
                M, N = tot[f"mass|{l}"], tot[f"n|{l}"]
                pm, pl = tot[f"{s}|mass|{l}"] / max(M, 1e-9), tot[f"{s}|low|{l}"] / max(N, 1)
                cells.append("." if pm < 5e-5 and pl < 5e-5 else f"{pm:.2%} ({pl:.2%})")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        rest = [f"{1 - sum(tot[f'{s}|mass|{l}'] for s in STAGES) / max(tot[f'mass|{l}'], 1e-9):.1%}" for l in LABELS]
        L += ["| **kept** | " + " | ".join(rest) + " |", ""]
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--days", nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "summary":
        summary()
        return
    days = a.days or sorted(pd.read_csv(f"{CAL}/days.csv", dtype=str).day)
    days = [d for d in days if not os.path.exists(f"{OUT}/days/{d}.csv")]
    print(f"[rule_effect] {len(days)} days", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, s in ex.map(run_day, days):
            print(f"  {day}: {n} hours, {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
