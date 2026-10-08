#!/usr/bin/env python
"""Does the hourly screen remove artefacts or real rain? Test against DWD RADKLIM RW over Germany
(DECISIONS §22, §24). RADKLIM shares the German radars with OPERA but has its own artefact
correction and gauge adjustment, so an artefact removed by our screen and present in RADKLIM
counts against the screen: the test is conservative.

  run      per calibration day with RADKLIM (`fetch_radklim.py`), on a crop around Germany:
             A    raw hourly sums, pixel-hours with a frame > 500 mm/h removed (the baseline)
             C    the full hourly chain, NaN where the tile-hour is rejected at split time
             Cnh  the chain without static clutter, Cns without spikes (attribution only)
             K    RADKLIM, paired by hour (`fetch_radklim.py`)
           on pixels of fully covered tile-hours where K is valid. Keeps the pixel-hours with
           max(A, K) >= 1 mm, with the OR of the repair codes over the four frames.
  summary  the test, with its criterion fixed before the results (2026-10-08):
             for raw bins 10-20, 20-50, 50-100, >= 100 mm,
             removed  C <= 0.5 A, or rejected;  kept  C >= 0.95 A, not rejected;
             RADKLIM confirms the raw value when K >= 0.5 A.
           A rule's removals count as artefacts if RADKLIM confirms them at less than half the
           rate it confirms kept values in the same bin (n >= 30); about the same rate means
           the rule removes real rain. Also: exceedance counts of A, C and K on the same
           pixel-hours, and log-correlation with K.

    python scripts/data_quality/radklim_test.py run --out .../calib/radklim_test --workers 6
    python scripts/data_quality/radklim_test.py summary --out .../calib/radklim_test
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
RKD = f"{D}/validation/radklim/hourly"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
MARGIN = 32
BITS = {"persist": 128, "footprint": 1, "ceiling": 16, "ring": 4, "ray": 2, "unsupported": 8}


def run_day(args):
    day, out = args
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import decide_hourly
    sys.path.insert(0, os.path.join(REPO, "scripts", "data_quality"))
    from extremes_effect import frame_flags
    t0 = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K, (by0, by1, bx0, bx1) = z["K"], z["box"]
    if not np.isfinite(K).any():
        return f"{day}: no RADKLIM"
    y0, y1, x0, x1 = max(by0 - MARGIN, 0), min(by1 + MARGIN, 2200), max(bx0 - MARGIN, 0), min(bx1 + MARGIN, 1900)
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    tab = pd.read_csv(f"{CAL}/hourly/hours/{day}.csv.gz", dtype={"end": str, "frames": str})
    rej = decide_hourly(tab, q, frame_flags=frame_flags(day, q))["rej_v4"].values if len(tab) else np.zeros(0, bool)
    tab = tab.assign(rejected=rej)
    runs = {}
    for name, off in (("C", ()), ("Cnh", ("hot",)), ("Cns", ("spike",))):
        cfg = {**q["hourly"], "rules": {**q["hourly"]["rules"], **{r: False for r in off}}}
        runs[name] = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                               ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                               cfg=cfg, crop=(y0, y1, x0, x1)).run()
    hd = runs["C"]
    d0 = np.datetime64(pd.Timestamp(day))
    cols = {k: [] for k in ("A", "C", "Cnh", "Cns", "K", "code", "rejected", "hour", "row", "col")}
    for j, h in enumerate(hd.hours):
        i = int((h["end"] - d0) / np.timedelta64(1, "h")) - 1                 # RADKLIM hour index
        if not 0 <= i < 24:
            continue
        end = np.datetime_as_string(h["end"], unit="s").replace("-", "").replace("T", "").replace(":", "")
        g = tab[tab.end == end]
        if not len(g):
            continue
        Kf = np.full((y1 - y0, x1 - x0), np.nan, np.float32)
        Kf[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = K[i]
        unphys = (np.nan_to_num(hd.raw[h["idx"]], nan=0.0) > 500.0).any(0)
        code = np.bitwise_or.reduce(hd.code[h["idx"]], axis=0)
        cnh = [hh for hh in runs["Cnh"].hours if hh["end"] == h["end"]][0]["sum"]
        cns = [hh for hh in runs["Cns"].hours if hh["end"] == h["end"]][0]["sum"]
        for r0, c0, rj in zip(g.row.values, g.col.values, g.rejected.values):
            if r0 + 128 <= y0 or r0 >= y1 or c0 + 128 <= x0 or c0 >= x1:
                continue
            sl = (slice(max(r0 - y0, 0), min(r0 + 128 - y0, y1 - y0)), slice(max(c0 - x0, 0), min(c0 + 128 - x0, x1 - x0)))
            a, k = h["raw_sum"][sl], Kf[sl]
            m = np.isfinite(k) & ~unphys[sl] & np.isfinite(a) & (np.maximum(a, k) >= 1.0)
            if not m.any():
                continue
            yy, xx = np.nonzero(m)
            cols["A"].append(a[m]); cols["C"].append(h["sum"][sl][m]); cols["Cnh"].append(cnh[sl][m])
            cols["Cns"].append(cns[sl][m]); cols["K"].append(k[m]); cols["code"].append(code[sl][m])
            cols["rejected"].append(np.full(m.sum(), bool(rj))); cols["hour"].append(np.full(m.sum(), i, np.int8))
            cols["row"].append((yy + sl[0].start + y0).astype(np.int16)); cols["col"].append((xx + sl[1].start + x0).astype(np.int16))
    os.makedirs(os.path.join(out, "days"), exist_ok=True)
    arr = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    np.savez_compressed(os.path.join(out, "days", f"{day}.npz"), **arr)
    return f"{day}: {len(arr['A']):,} pixel-hours, {time.time() - t0:.0f} s"


def corr(a, b):
    return float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])


def summary(out):
    cats = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day")["category"]
    group = lambda c: "random" if c == "random" else ("event" if str(c).startswith("event") else "failure")
    frames = []
    for f in sorted(glob.glob(os.path.join(out, "days", "*.npz"))):
        z = np.load(f)
        if not len(z["A"]):
            continue
        d = os.path.basename(f)[:8]
        frames.append(pd.DataFrame({k: z[k] for k in ("A", "C", "Cnh", "Cns", "K", "code", "rejected")}).assign(day=d, grp=group(cats.get(d, "random"))))
    X = pd.concat(frames, ignore_index=True)
    X["Cf"] = np.where(X.rejected, np.nan, X.C)
    lines = [f"{X.day.nunique()} days with RADKLIM, {len(X):,} pixel-hours with max(A, K) >= 1 mm"]
    bins = [(10, 20), (20, 50), (50, 100), (100, np.inf)]
    for grp in ("all", "random", "event", "failure"):
        x = X if grp == "all" else X[X.grp == grp]
        if not len(x):
            continue
        lines.append(f"\n=== {grp}: {x.day.nunique()} days")
        both = x[np.isfinite(x.Cf)]
        lines.append(f"log-corr with RADKLIM (kept tile-hours): A {corr(both.A, both.K):.3f} -> C {corr(both.Cf, both.K):.3f}")
        ex = []
        for u in (5, 10, 20, 30, 50, 75, 100):
            ex.append({"u_mm": u, "A": int((x.A >= u).sum()), "C (rejected = removed)": int((x.Cf >= u).sum()),
                       "RADKLIM": int((x.K >= u).sum())})
        lines.append("pixel-hours >= u on the same pixels:\n" + pd.DataFrame(ex).to_string(index=False))
        rows = []
        for lo, hi in bins:
            b = x[(x.A >= lo) & (x.A < hi)]
            removed = b[(b.C <= 0.5 * b.A) | b.rejected]
            kept = b[(b.C >= 0.95 * b.A) & ~b.rejected]
            conf = lambda s: float((s.K >= 0.5 * s.A).mean()) if len(s) else np.nan
            base = conf(kept)
            r = {"raw_bin": f"{lo}-{hi}", "n": len(b), "n_kept": len(kept), "confirm_kept": base,
                 "medK/A_kept": float((kept.K / kept.A).median()) if len(kept) else np.nan,
                 "n_removed": len(removed), "confirm_removed": conf(removed),
                 "medK/A_removed": float((removed.K / removed.A).median()) if len(removed) else np.nan}
            att = {"rejected": removed.rejected,
                   "hot": (~removed.rejected) & (removed.Cnh >= 0.95 * removed.A),
                   "spike": (~removed.rejected) & (removed.Cns >= 0.95 * removed.A)}
            for k_, bit in BITS.items():
                att[k_] = (~removed.rejected) & ((removed.code & bit) > 0)
            for k_, m in att.items():
                s = removed[m]
                c = conf(s)
                verdict = "" if len(s) < 30 else ("artefact" if c < 0.5 * base else ("REAL RAIN" if c >= 0.8 * base else "mixed"))
                r[f"{k_}"] = f"{len(s)}|{c:.2f}|{verdict}" if len(s) else "0"
            rows.append(r)
        t = pd.DataFrame(rows)
        lines.append("removal test (per rule: n | share confirmed by RADKLIM | verdict at the pre-stated criterion):\n"
                     + t.round(3).to_string(index=False))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "summary.txt"), "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--days", nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary(a.out)
    have = {os.path.basename(f)[:8] for f in glob.glob(f"{RKD}/*.npz")}
    cal = pd.read_csv(f"{CAL}/days.csv", dtype=str).day
    days = [d for d in (a.days or cal) if d in have and os.path.exists(f"{CAL}/hourly/hours/{d}.csv.gz")
            and not os.path.exists(os.path.join(a.out, "days", f"{d}.npz"))]
    print(f"[radklim_test] {len(days)} days -> {a.out}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(run_day, [(d, a.out) for d in days]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
