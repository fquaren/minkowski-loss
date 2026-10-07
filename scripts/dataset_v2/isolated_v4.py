#!/usr/bin/env python
"""Isolated high values in the v4-cleaned field: pixels >= 31 mm/h with dry surroundings
(EXPERIMENTS §5, Phase-0 gallery review 2026-10-07). On the 48.62 tile (r640 c1408, Nov 2019)
v4 keeps maxima of 138-342 mm/h made of 2-9 pixels whose 5x5 median is 0. Before a rule is
proposed, this counts how often that happens on random days, event days and the known
failures, by tile max, so real isolated cells (the risk under DECISIONS §17) can be compared
with artefacts.

  scan     per Phase-0 day, on the cleaned field (what training sees), for every tile with
           max >= 31 in tiles_decided: neighbourhood statistics at the tile max and counts of
           isolated pixels. Writes <calib>/isolated/<day>.csv.gz.
  report   tables by category and tile-max bin -> <calib>/isolated/report.md.

A pixel is "isolated" when the median of its 5x5 window (10 km) is < 0.1 mm/h and at least
half of that window is covered (no-data counts as dry otherwise, which would flag coverage
edges). The window statistics are stored, so other thresholds can be read off without a rescan.
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.data_quality._md import md_table  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
CALIB = f"{D}/quality_v4/calib"
U = 31.0
DRY = 0.1
BINS = [31, 53, 89, 150, 500.01, np.inf]


def window_stats(f0, valid, ys, xs, half):
    """Median, wet fraction (>= DRY) and covered fraction of the (2*half+1)^2 window around
    each (y, x). Windows are cut from a padded copy, so pixels near the grid edge work too."""
    k = 2 * half + 1
    fp = np.pad(f0, half)
    vp = np.pad(valid, half)
    w = sliding_window_view(fp, (k, k))[ys, xs].reshape(len(ys), -1)
    v = sliding_window_view(vp, (k, k))[ys, xs].reshape(len(ys), -1)
    return np.median(w, 1), (w >= DRY).mean(1), v.mean(1)


def scan_day(day, tiles, qpath, out_path):
    import yaml
    import scan_tiles as st
    from src.data.day_cleaner import DayCleaner
    q = yaml.safe_load(open(qpath))
    _, ceil = st._quality(qpath)
    dc = DayCleaner(f"{D}/raw/OPERA", day, "TOT_PREC",
                    hot=lambda y: st._hot(f"{D}/quality_v2/clutter_climatology.npz", y),
                    ring=lambda y: st._ring(f"{D}/quality_v2/ring_mask.npz", y),
                    sites_rc=st._sites(), ceilings=ceil, max_size=q["guard"]["max_size"])
    stamps = [np.datetime_as_string(t, unit="s").replace("-", "").replace("T", "").replace(":", "")
              for t in dc.times]
    rows = []
    for ts, g in tiles.groupby("timestamp"):
        if ts not in stamps:
            continue
        f = dc.clean(stamps.index(ts))[0]
        valid = np.isfinite(f)
        f0 = np.where(valid, f, 0.0).astype(np.float32)
        for r in g.itertuples():
            t = f0[r.row:r.row + 128, r.col:r.col + 128]
            ys, xs = np.nonzero(t >= U)
            if not ys.size:
                continue
            ys, xs = ys + r.row, xs + r.col
            vals = f0[ys, xs]
            m5, w5, c5 = window_stats(f0, valid, ys, xs, 2)
            m11, w11, c11 = window_stats(f0, valid, ys, xs, 5)
            iso = (m5 < DRY) & (c5 >= 0.5)
            i = int(np.argmax(vals))
            rows.append({"timestamp": ts, "row": r.row, "col": r.col, "max": float(vals[i]),
                         "n_ge31": int(ys.size), "n_iso31": int(iso.sum()),
                         "n_iso89": int((iso & (vals >= 89)).sum()),
                         "n_iso150": int((iso & (vals >= 150)).sum()),
                         "med5_at_max": float(m5[i]), "wet5_at_max": float(w5[i]),
                         "cov5_at_max": float(c5[i]), "med11_at_max": float(m11[i]),
                         "wet11_at_max": float(w11[i]), "iso_at_max": bool(iso[i])})
    dc.close()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return day, len(rows)


def cmd_scan(workers, qpath, days=None):
    out = os.path.join(CALIB, "isolated")
    os.makedirs(out, exist_ok=True)
    t = pd.read_csv(os.path.join(CALIB, "tiles_decided.csv.gz"),
                    dtype={"timestamp": str, "day": str}, low_memory=False)
    t = t[t["max"] >= U][["timestamp", "row", "col", "day"]]
    groups = {d: g for d, g in t.groupby("day") if days is None or d in days}
    todo = {d: g for d, g in groups.items() if not os.path.exists(os.path.join(out, f"{d}.csv.gz"))}
    print(f"[iso] {len(groups)} days, {len(todo)} to scan, {sum(map(len, todo.values())):,} tiles", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_day, d, g, qpath, os.path.join(out, f"{d}.csv.gz")): d
                for d, g in todo.items()}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                day, n = fu.result()
                el = time.time() - t0
                if k % 10 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} ({day}: {n} tiles), ~{el / k * (len(futs) - k) / 60:.0f} min left",
                          flush=True)
            except Exception as e:
                print(f"  ! {futs[fu]}: {type(e).__name__}: {e}", flush=True)
    print(f"[iso] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_report():
    out = os.path.join(CALIB, "isolated")
    x = pd.concat([pd.read_csv(p, dtype={"timestamp": str}) for p in sorted(glob.glob(f"{out}/*.csv.gz"))
                   if os.path.getsize(p) > 30])
    t = pd.read_csv(os.path.join(CALIB, "tiles_decided.csv.gz"),
                    dtype={"timestamp": str, "day": str}, low_memory=False)
    x = x.merge(t[["timestamp", "row", "col", "category", "rej_v4"]], on=["timestamp", "row", "col"])
    x["set"] = np.where(x.category.str.startswith("event"), "event",
                        np.where(x.category.eq("random"), "random", x.category))
    x["bin"] = pd.cut(x["max"], BINS, right=False).astype(str)
    x.to_csv(os.path.join(out, "all.csv.gz"), index=False)
    L = ["# Isolated high values in the v4-cleaned field (Phase 0)\n",
         f"Tiles with max >= 31 mm/h: {len(x):,}. Isolated: 5x5 median < {DRY} mm/h, window "
         ">= 50% covered. Kept = not rejected by v4.\n"]
    for title, sub in (("All tiles", x), ("Tiles v4 keeps", x[~x.rej_v4])):
        g = sub.groupby(["set", "bin"])
        tab = pd.DataFrame({"tiles": g.size(), "max isolated": g.iso_at_max.mean(),
                            ">=1 iso px": g.n_iso31.apply(lambda s: (s >= 1).mean()),
                            ">=3 iso px": g.n_iso31.apply(lambda s: (s >= 3).mean()),
                            "med wet5 at max": g.wet5_at_max.median(),
                            "q01 wet11 at max": g.wet11_at_max.quantile(0.01)}).reset_index()
        L.append(f"## {title}\n")
        L.append(md_table(tab, floatfmt=".3g", index=False) + "\n")
    k = x[~x.rej_v4 & x.iso_at_max].sort_values("max", ascending=False)
    L.append("## Kept tiles whose max is isolated, strongest 25\n")
    L.append(md_table(k[["timestamp", "row", "col", "max", "set", "n_ge31", "n_iso31",
                         "wet5_at_max", "wet11_at_max", "cov5_at_max"]].head(25), floatfmt=".3g", index=False) + "\n")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L))
    print(f"[iso] wrote {out}/report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["scan", "report"])
    ap.add_argument("--quality", default=os.path.join(CALIB, "quality_calib.yaml"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--days", nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "scan":
        cmd_scan(a.workers, a.quality, set(a.days) if a.days else None)
    else:
        cmd_report()
