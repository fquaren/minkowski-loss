#!/usr/bin/env python
"""Boundary-jump test for radars biased against their neighbours (DECISIONS §36).

From the radar pass (`quality_v4/radars/*.npz`, per radar-frame features on the clean_frame field,
contributing geometry): `bjump` = mean log1p(rate) in the 6 km band inside a radar's owned area
minus that in the 6 km band outside it. Frames with wet fraction >= 0.05 and >= 50 valid pixels
in each band; radar-day = median over >= 8 such frames; radar-year = median over >= 20 radar-days,
95% day-block bootstrap (B = 500). Flagged: top 5% of radar-years and the interval above 0.
Writes quality_v4/biased_radar_years.csv (flagged) and the full table.

    python scripts/analysis/radar_boundary.py [--workers 2]
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
Q4 = "/home/fquareng/work/data/extremes/OPERA/quality_v4"
OUT = f"{REPO}/notes/data_analysis_v4"


def _day(f):
    from src.data.radar_screen import FEATURES
    z = np.load(f)
    F, keys = z["feats"], z["keys"].astype(str)
    wet, bj = F[..., FEATURES.index("wet")], F[..., FEATURES.index("bjump")]
    nin, nout = F[..., FEATURES.index("n_band_in")], F[..., FEATURES.index("n_band_out")]
    use = (wet >= 0.05) & (nin >= 50) & (nout >= 50) & np.isfinite(bj)
    day = os.path.basename(f)[:8]
    out = []
    for j in np.nonzero(use.sum(0) >= 8)[0]:
        out.append((keys[j], day, float(np.median(bj[use[:, j], j])), int(use[:, j].sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{Q4}/radars/*.npz"))
    with ProcessPoolExecutor(a.workers) as ex:
        rd = pd.DataFrame([r for rr in ex.map(_day, files, chunksize=20) for r in rr],
                          columns=["radar", "day", "bjump", "frames"])
    rd["year"] = rd.day.str[:4]
    rng = np.random.default_rng(0)
    rows = []
    for (r, y), g in rd.groupby(["radar", "year"]):
        if len(g) < 20:
            continue
        v = g.bjump.values
        bs = np.median(v[rng.integers(0, len(v), (500, len(v)))], axis=1)
        rows.append({"radar": r, "year": y, "days": len(v), "median_bjump": float(np.median(v)),
                     "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))})
    t = pd.DataFrame(rows)
    cut = t.median_bjump.quantile(0.95)
    t["flagged"] = (t.median_bjump >= cut) & (t.lo > 0)
    t = t.sort_values("median_bjump", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(f"{OUT}/radar_boundary.csv", index=False)
    t[t.flagged][["radar", "year", "days", "median_bjump", "lo", "hi"]].to_csv(f"{Q4}/biased_radar_years.csv", index=False)
    from scripts.data_quality._md import md_table
    per = t.groupby("radar").agg(years=("year", "size"), flagged=("flagged", "sum"),
                                 median=("median_bjump", "median")).sort_values("median", ascending=False)
    focus = t[t.radar.str[:2].isin(["ro", "hu", "sk"])]
    L = ["# Boundary-jump test (DECISIONS §36)", "",
         f"{len(t)} radar-years with >= 20 radar-days; top-5% cut on the median jump = {cut:.3f} "
         f"(log1p units: a jump of 0.1 means the radar's side of its boundary reads ~10% more in 1 + rate). "
         f"Flagged radar-years: {int(t.flagged.sum())} ({t[t.flagged].radar.nunique()} radars).", "",
         "## Flagged", "", md_table(t[t.flagged], index=False, floatfmt=".3g"), "",
         "## Romania, Hungary, Slovakia (all years)", "", md_table(focus, index=False, floatfmt=".3g"), "",
         "## Radars by median jump (top 25)", "", md_table(per.head(25), floatfmt=".3g")]
    txt = "\n".join(L)
    open(f"{OUT}/radar_boundary.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
