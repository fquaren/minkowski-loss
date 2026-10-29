#!/usr/bin/env python
"""Raw vs screened value distributions on the full grid, and the provisional hourly tail threshold
(DECISIONS §30, item 5).

Per calibration day, current chain, full grid: log-binned histograms of
  raw15 / scr15   15-min rates (mm/h) of the raw and screened frames of complete hours
  rawH  / scrH    hourly ACRR sums (mm), raw and screened
Screened values of tile-hours rejected at split time (current rule 5) are left out of scr15 /
scrH; the raw histograms keep everything. Counts include zeros (bin 0) so frequencies are per
valid pixel.

  run      python scripts/data_quality/dist_hist.py run --workers 2 [--group random]
  u        provisional u: the hourly level whose exceedance frequency in scrH equals that of
           31 mm/h in scr15
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
OUT = f"{CAL}/dist_hist"
QPATH = f"{REPO}/configs/quality_v4.yaml"
EDGES = np.r_[0.0, np.logspace(-1, 4.5, 551)]       # bin 0 = [0, 0.1); 0.01-decade bins to ~31,600


def hist(x):
    x = x[np.isfinite(x)]
    return np.histogram(np.clip(x, 0, EDGES[-1] * 0.999), EDGES)[0].astype(np.int64)


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import decide_hourly
    t0 = time.time()
    q, ceil = st._quality(QPATH)
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q.get("hourly", {})).run()
    tab = pd.read_csv(f"{CAL}/hourly_c/hours/{day}.csv.gz", dtype={"end": str, "frames": str, "owners": str})
    if "n_burst" not in tab:
        tab["n_burst"] = 0
    ff = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    de = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    d = decide_hourly(tab, q, frame_flags=set(zip(ff.radar, ff.timestamp)), day_excl=set(zip(de.radar, de.day)))
    rej = tab["unphysical"].astype(bool).values | d["rej_v4"].values
    H, W = hd.raw.shape[1:]
    out = {k: np.zeros(len(EDGES) - 1, np.int64) for k in ("raw15", "scr15", "rawH", "scrH")}
    for h in hd.hours:
        end = np.datetime_as_string(h["end"], unit="s").replace("-", "").replace("T", "").replace(":", "")
        m = np.zeros((H, W), bool)                       # rejected tile-hours of this hour
        for r, c in zip(tab.row.values[(tab.end.values == end) & rej], tab.col.values[(tab.end.values == end) & rej]):
            m[r:r + 128, c:c + 128] = True
        ok = h["valid"] if "valid" in h else np.isfinite(h["raw_sum"])
        out["rawH"] += hist(h["raw_sum"][ok])
        out["scrH"] += hist(h["sum"][ok & ~m])
        for j in h["idx"]:
            out["raw15"] += hist(hd.raw[j][ok])
            out["scr15"] += hist(hd.fin[j][ok & ~m])
    os.makedirs(f"{OUT}/days", exist_ok=True)
    np.savez_compressed(f"{OUT}/days/{day}.npz", edges=EDGES, **out)
    return day, len(hd.hours), time.time() - t0


def exceed(h, x):
    """Frequency of values >= x from a histogram (linear within the 0.01-decade bin)."""
    c = np.r_[np.cumsum(h[::-1])[::-1], 0]
    i = np.searchsorted(EDGES, x, side="right") - 1
    f = (np.log10(x) - np.log10(EDGES[i])) / (np.log10(EDGES[i + 1]) - np.log10(EDGES[i])) if i > 0 else 0.0
    return (c[i] - f * h[i]) / c[0]


def load(group):
    cat = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day").category
    tot = None
    n = 0
    for f in sorted(glob.glob(f"{OUT}/days/*.npz")):
        c = cat.get(os.path.basename(f)[:8], "")
        g = "random" if c == "random" else "event" if c.startswith("event") else "failure"
        if group != "all" and g != group:
            continue
        z = np.load(f)
        tot = {k: z[k].copy() for k in ("raw15", "scr15", "rawH", "scrH")} if tot is None else \
            {k: tot[k] + z[k] for k in tot}
        n += 1
    return tot, n


def cmd_u():
    tot, n = load("random")
    p = exceed(tot["scr15"], 31.0)
    xs = np.logspace(0, 3, 30001)
    fh = np.array([exceed(tot["scrH"], x) for x in xs])
    u = xs[np.argmin(np.abs(fh - p))]
    txt = (f"{n} random days. P(screened 15-min rate >= 31 mm/h) = {p:.3e} per valid pixel-frame; "
           f"hourly level with the same exceedance frequency: u = {u:.1f} mm "
           f"(raw: P15 = {exceed(tot['raw15'], 31.0):.3e}, P(hour >= u) = {exceed(tot['rawH'], u):.3e})")
    print(txt)
    open(f"{OUT}/u_provisional.txt", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "u"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--group", default="random")
    ap.add_argument("--reverse", action="store_true", help="process days from the end (a second instance)")
    a = ap.parse_args()
    if a.cmd == "u":
        cmd_u()
        return
    cat = pd.read_csv(f"{CAL}/days.csv", dtype=str)
    days = sorted(cat.day if a.group == "all" else cat[cat.category.str.startswith(a.group)].day)
    days = [d for d in days if not os.path.exists(f"{OUT}/days/{d}.npz")][::-1 if a.reverse else 1]
    print(f"[dist_hist] {len(days)} days", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, s in ex.map(run_day, days):
            print(f"  {day}: {n} hours, {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
