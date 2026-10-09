#!/usr/bin/env python
"""The RADKLIM test as a presence test, and the size of the remaining contamination
(2026-10-09; DECISIONS §24 result, researcher discussion of 2026-10-09).

RADKLIM's peak values are not a reliable judge (smoothing, interpolated gauge adjustment, 1->2 km
averaging, point-vs-area support), but its answer to "is there substantial rain here, give or
take 2 px and 1 h" is. This script restates the removal test that way and counts the screened
values the dataset would keep that RADKLIM sees as dry.

From the `radklim_test.py` samples (pixel-hours over Germany with max(raw A, RADKLIM K) >= 1 mm):
  K_tol    max of RADKLIM over +-2 px and the hours h-1 .. h+1 (NaN where all of it is missing)
  dry      K_tol < 1 mm; dry5: K_tol < 5 mm
  run      writes per day the samples with K_tol, and for the contaminated pixel-hours (screened,
           kept, C >= 50 mm, dry5) their signature: n_peak (frames of the day the pixel is a
           local peak, `calib/hourly/signals`), distance to the nearest radar, the largest share
           of the hour's sum carried by one 15-min frame, and the median of the screened hourly
           field over the 5x5 window (isolation)
  summary  (a) removed vs kept: share dry, by raw bin and rule; (b) kept screened values >= 20 /
           30 / 50 / 75 / 100 mm that are dry: count, share of that tail level, concentration
           in days; random / event / failure days; (c) the signature table

    python scripts/data_quality/radklim_presence.py run --workers 6
    python scripts/data_quality/radklim_presence.py summary
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
D = "/home/fquareng/work/data/extremes/OPERA"
CAL = f"{D}/quality_v4/calib"
SAMP = f"{CAL}/radklim_test/days"
RKD = f"{D}/validation/radklim/hourly"
SIG = f"{CAL}/hourly/signals"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{CAL}/radklim_presence"
BITS = {"persist": 128, "footprint": 1, "ceiling": 16, "ring": 4, "ray": 2, "unsupported": 8}


def k_tolerant(K24):
    """(24, H, W) max over +-2 px and h-1 .. h+1; NaN where the whole window is missing."""
    out = np.full(K24.shape, np.nan, np.float32)
    fin = np.isfinite(K24)
    K0 = np.where(fin, K24, -1.0)
    for i in range(24):
        sl = slice(max(i - 1, 0), min(i + 2, 24))
        m = ndimage.maximum_filter(K0[sl].max(0), size=5)
        out[i] = np.where(m < 0, np.nan, m)
    return out


def run_day(day):
    t0 = time.time()
    z = np.load(f"{SAMP}/{day}.npz")
    if not len(z["A"]):
        return day, 0, 0, 0.0
    rk = np.load(f"{RKD}/{day}.npz")
    K24, (y0, y1, x0, x1) = rk["K"], rk["box"]
    Kt = k_tolerant(K24)
    hr = z["hour"].astype(int)
    r, c = z["row"].astype(int) - y0, z["col"].astype(int) - x0
    df = pd.DataFrame({k: z[k] for k in ("A", "C", "Cnh", "Cns", "K", "code", "rejected", "hour", "row", "col")})
    df["K_tol"] = Kt[hr, r, c]
    df["day"] = day
    keep = (df.A >= 5) | (df.C >= 5) | (df.K >= 5)
    os.makedirs(os.path.join(OUT, "days"), exist_ok=True)
    df[keep].to_csv(os.path.join(OUT, "days", f"{day}.csv.gz"), index=False, float_format="%.4g")
    # signature of the contaminated pixel-hours
    bad = df[(~df.rejected.astype(bool)) & (df.C >= 50) & (df.K_tol < 5)]
    if len(bad):
        import scan_tiles as st
        from src.data.hourly import HourlyDay
        q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
        hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                       ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                       cfg=q, crop=(y0, y1, x0, x1)).run()
        d0 = np.datetime64(pd.Timestamp(day))
        byhour = {int((h["end"] - d0) / np.timedelta64(1, "h")) - 1: h for h in hd.hours}
        sig = np.load(f"{SIG}/{day}.npz") if os.path.exists(f"{SIG}/{day}.npz") else None
        sites = st._sites()
        rows = []
        for _, b in bad.iterrows():
            h = byhour.get(int(b.hour))
            rr, cc = int(b.row) - y0, int(b.col) - x0
            d = {"day": day, "hour": int(b.hour), "row": int(b.row), "col": int(b.col), "A": b.A, "C": b.C,
                 "K": b.K, "K_tol": b.K_tol, "code": int(b.code),
                 "dist_radar_km": float(2.0 * np.hypot(sites[:, 0] - b.row, sites[:, 1] - b.col).min()),
                 "n_peak": int(sig["n_peak"][int(b.row), int(b.col)]) if sig is not None else -1,
                 "n_wet": int(sig["n_wet"][int(b.row), int(b.col)]) if sig is not None else -1}
            if h is not None:
                fr = np.nan_to_num(hd.fin[h["idx"]][:, rr, cc], nan=0.0)
                d["max_frame_share"] = float(fr.max() / max(fr.sum(), 1e-6))
                d["max_frame_rate"] = float(fr.max())
                w = np.nan_to_num(h["sum"][max(rr - 2, 0):rr + 3, max(cc - 2, 0):cc + 3], nan=0.0)
                d["win5_median"] = float(np.median(w))
            rows.append(d)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "days", f"{day}_bad.csv"), index=False, float_format="%.4g")
    return day, len(df), len(bad), time.time() - t0


def summary():
    cats = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day")["category"]
    grp = lambda d: "random" if cats.get(d) == "random" else ("event" if str(cats.get(d, "")).startswith("event") else "failure")
    X = pd.concat([pd.read_csv(f, dtype={"day": str}) for f in glob.glob(f"{OUT}/days/*[0-9].csv.gz")], ignore_index=True)
    X["grp"] = X.day.map(grp)
    X["rejected"] = X.rejected.astype(bool)
    X["dry"], X["dry5"] = X.K_tol < 1, X.K_tol < 5
    lines = [f"{X.day.nunique()} days, {len(X):,} pixel-hours with A, C or K >= 5 mm (Germany, RADKLIM domain)"]
    # (a) presence version of the removal test
    rows = []
    for lo, hi in ((10, 20), (20, 50), (50, 100), (100, np.inf)):
        b = X[(X.A >= lo) & (X.A < hi) & X.K_tol.notna()]
        rem = b[(b.C <= 0.5 * b.A) | b.rejected]
        kept = b[(b.C >= 0.95 * b.A) & ~b.rejected]
        r = {"raw_bin": f"{lo}-{hi}", "n_kept": len(kept), "kept dry<1": kept.dry.mean(), "kept dry<5": kept.dry5.mean(),
             "n_removed": len(rem), "removed dry<1": rem.dry.mean(), "removed dry<5": rem.dry5.mean()}
        att = {"hot": (~rem.rejected) & (rem.Cnh >= 0.95 * rem.A), "spike": (~rem.rejected) & (rem.Cns >= 0.95 * rem.A)}
        att.update({k: (~rem.rejected) & ((rem.code & v) > 0) for k, v in BITS.items()})
        for k, m in att.items():
            s = rem[m]
            r[k] = f"{len(s)}|{s.dry5.mean():.2f}" if len(s) else "0"
        rows.append(r)
    lines.append("\n(a) presence test: share of pixel-hours where RADKLIM is dry within +-2 px / +-1 h "
                 "(per rule: n | share dry < 5 mm):\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    # (b) remaining contamination in the screened, kept values
    for g in ("all", "random", "event", "failure"):
        x = X if g == "all" else X[X.grp == g]
        kept = x[~x.rejected & x.K_tol.notna()]
        rows = []
        for u in (20, 30, 50, 75, 100):
            t = kept[kept.C >= u]
            bad = t[t.dry5]
            days = bad.day.value_counts()
            rows.append({"u_mm": u, "screened >= u": len(t), "RADKLIM dry<5": len(bad), "share": len(bad) / max(len(t), 1),
                         "RADKLIM dry<1": int(t.dry.sum()), "days": len(days),
                         "top-3 days share": days.head(3).sum() / max(len(bad), 1)})
        lines.append(f"\n(b) {g}: screened values the dataset keeps, and how many RADKLIM sees as dry:\n"
                     + pd.DataFrame(rows).round(3).to_string(index=False))
    # (c) signature
    B = [pd.read_csv(f, dtype={"day": str}) for f in glob.glob(f"{OUT}/days/*_bad.csv")]
    if B:
        B = pd.concat(B, ignore_index=True)
        B["grp"] = B.day.map(grp)
        lines.append(f"\n(c) signature of {len(B)} contaminated pixel-hours (screened >= 50 mm, RADKLIM dry < 5 mm "
                     f"within tolerance), {B.day.nunique()} days:")
        for col, q in (("n_peak", [.1, .5, .9]), ("dist_radar_km", [.1, .5, .9]), ("max_frame_share", [.1, .5, .9]),
                       ("max_frame_rate", [.1, .5, .9]), ("win5_median", [.1, .5, .9]), ("C", [.1, .5, .9])):
            if col in B:
                lines.append(f"  {col:16s} q10/50/90: " + " / ".join(f"{v:.2f}" for v in B[col].quantile(q)))
        lines.append(f"  one frame carries > 80% of the hour: {(B.max_frame_share > 0.8).mean():.0%}; "
                     f"local peak in >= 12 frames of the day: {(B.n_peak >= 12).mean():.0%}; within 30 km of a radar: "
                     f"{(B.dist_radar_km < 30).mean():.0%}; isolated (5x5 median < 5 mm): {(B.win5_median < 5).mean():.0%}")
        lines.append("  days with most: " + ", ".join(f"{d} ({n})" for d, n in B.day.value_counts().head(8).items()))
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
    days = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{SAMP}/*.npz"))
    days = [d for d in days if not os.path.exists(os.path.join(OUT, "days", f"{d}.csv.gz"))]
    print(f"[radklim_presence] {len(days)} days -> {OUT}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, nb, dt in ex.map(run_day, days):
            if nb:
                print(f"  {day}: {n:,} pixel-hours, {nb} contaminated, {dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
