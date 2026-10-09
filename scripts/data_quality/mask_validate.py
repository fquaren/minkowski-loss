#!/usr/bin/env python
"""Acceptance tests for the per-pixel outlier mask (DECISIONS §25 and amendments).

For each RADKLIM-test day and each d^2 threshold of `pixel_outliers.py`, runs the hourly chain
over Germany with the mask as the static-clutter ("hot") mask (repair only if lower), next to
the chain without any static-clutter mask. Pixel-hours the mask lowers are tested against
RADKLIM as in the §24 removal test.
  run      writes per day the pixel-hours with max(raw, RADKLIM) >= 1 mm: raw A, no-mask Cn,
           and per threshold Ct, plus RADKLIM K
  summary  per threshold, separately on the even and odd halves of the day list:
             removed  Ct <= 0.5 Cn (the mask lowered it to at most half),
             kept     Ct >= 0.95 Cn and the pixel not masked,
           share confirmed by RADKLIM (K >= 0.5 A) in raw bins 10-20 / 20-50 / 50-100 mm.
           Pass in a bin with n >= 30: removed confirmed < 0.5 x kept. The threshold is chosen
           on the even half by the rule of DECISIONS §25 and reported on the odd half. Also
           reports the flag rate by raw intensity (safeguard 2).

    python scripts/data_quality/mask_validate.py run --workers 6
    python scripts/data_quality/mask_validate.py summary
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
from pixel_outliers import D2_THRS, OUT as PO_DIR  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
CAL = f"{D}/quality_v4/calib"
RKD = f"{D}/validation/radklim/hourly"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{CAL}/mask_validate"
_PO = None


def mask(year, thr):
    global _PO
    if _PO is None:
        _PO = np.load(f"{PO_DIR}/pixel_outliers.npz")
    k = f"y{year}_t{thr}"
    return _PO[k] if k in _PO.files else None


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    t0 = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (y0, y1, x0, x1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, 0, 0.0
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    year = int(day[:4])
    runs = {}
    for name, hot in [("Cn", None)] + [(f"C{t}", t) for t in D2_THRS]:
        if hot is not None and mask(year, hot) is None:
            continue
        cfg = dict(q.get("hourly", {}))
        cfg["rules"] = {**cfg.get("rules", {}), "hot": hot is not None}
        runs[name] = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC",
                               hot=(lambda y, t=hot: mask(y, t)) if hot is not None else (lambda y: None),
                               ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                               cfg=cfg, crop=(y0, y1, x0, x1)).run()
    base = runs["Cn"]
    d0 = np.datetime64(pd.Timestamp(day))
    cols = {k: [] for k in ["A", "K", "hour", "masked_any"] + list(runs)}
    for j, h in enumerate(base.hours):
        i = int((h["end"] - d0) / np.timedelta64(1, "h")) - 1
        if not 0 <= i < 24:
            continue
        a, k = h["raw_sum"], K24[i]
        unphys = (np.nan_to_num(base.raw[h["idx"]], nan=0.0) > 500).any(0)
        m = np.isfinite(k) & np.isfinite(a) & ~unphys & (np.maximum(a, np.nan_to_num(k)) >= 1.0)
        if not m.any():
            continue
        cols["A"].append(a[m]); cols["K"].append(k[m]); cols["hour"].append(np.full(m.sum(), i, np.int8))
        anym = np.zeros(m.sum(), bool)
        for name, hd in runs.items():
            hh = [x for x in hd.hours if x["end"] == h["end"]][0]
            cols[name].append(hh["sum"][m])
        for t in D2_THRS:
            mk = mask(year, t)
            if mk is not None:
                anym |= mk[y0:y1, x0:x1][m]
        cols["masked_any"].append(anym)
    os.makedirs(os.path.join(OUT, "days"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "days", f"{day}.npz"),
                        **{k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()})
    return day, int(sum(len(x) for x in cols["A"])), time.time() - t0


def evaluate(X, thr):
    col = f"C{thr}"
    if col not in X:
        return None
    rows, ok = [], True
    for lo, hi in ((10, 20), (20, 50), (50, 100)):
        b = X[(X.A >= lo) & (X.A < hi) & X[col].notna()]
        removed = b[b[col] <= 0.5 * b.Cn]
        kept = b[b[col] >= 0.95 * b.Cn]
        conf = lambda s: float((s.K >= 0.5 * s.A).mean()) if len(s) else np.nan
        ck, cr = conf(kept), conf(removed)
        passed = None if len(removed) < 30 else bool(cr < 0.5 * ck)
        ok &= passed is not False
        rows.append({"bin": f"{lo}-{hi}", "n": len(b), "n_removed": len(removed), "share_removed": len(removed) / max(len(b), 1),
                     "confirm_kept": ck, "confirm_removed": cr, "pass": passed})
    n10 = int(((X.A >= 10) & (X[col] <= 0.5 * X.Cn)).sum())
    return pd.DataFrame(rows), ok, n10


def summary():
    files = sorted(glob.glob(f"{OUT}/days/*.npz"))
    days = [os.path.basename(f)[:8] for f in files]
    halves = {"even": days[0::2], "odd": days[1::2]}
    lines = []
    chosen = None
    for half in ("even", "odd"):
        X = pd.concat([pd.DataFrame({k: z[k] for k in z.files}) for z in
                       (np.load(f"{OUT}/days/{d}.npz") for d in halves[half]) if len(z["A"])], ignore_index=True)
        lines.append(f"\n=== {half} half: {len(halves[half])} days, {len(X):,} pixel-hours")
        cand = []
        for t in D2_THRS:
            r = evaluate(X, t)
            if r is None:
                continue
            tab, ok, n10 = r
            lines.append(f"d2 > {t}: {'PASS' if ok else 'FAIL'}; removed pixel-hours >= 10 mm: {n10}\n" + tab.round(3).to_string(index=False))
            if ok:
                cand.append((n10, t))
        if half == "even":
            chosen = max(cand)[1] if cand else None
            lines.append(f"chosen on the even half (most removals among passing thresholds): {chosen}")
        elif chosen is not None:
            lines.append(f"reported for the chosen threshold d2 > {chosen} on the odd half (above).")
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/summary.txt", "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    have = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{CAL}/radklim_test/days/*.npz"))
    days = [d for d in have if not os.path.exists(os.path.join(OUT, "days", f"{d}.npz"))]
    print(f"[mask_validate] {len(days)} days -> {OUT}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, dt in ex.map(run_day, days):
            print(f"  {day}: {n:,} pixel-hours, {dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
