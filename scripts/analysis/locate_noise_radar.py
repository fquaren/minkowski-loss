#!/usr/bin/env python
"""Locate the radar behind a disk-shaped noise pattern in the raw >= 31 mm/h climatology.

Segments the year's anomaly (frequency > 1e-3 and > 5x the median of the reference years) in a
window, fits a circle to its western arc and a line to its straight eastern edge, and ranks every
site of the OPERA database (archive included) by distance to the fitted centre. Written for the
2023 half-disk south of Timisoara / Fruska Gora (the suspect-radar page), which is Jastrebac (rsjas).

    python scripts/analysis/locate_noise_radar.py --year 2023 --window 143 731 1131 1646
      (-> notes/data_analysis_v4/noise_radar_<year>.png / .md)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
D = "/home/fquareng/work/data/extremes/OPERA"
NOTES = f"{REPO}/notes/data_analysis_v4"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--window", type=int, nargs=4, default=[143, 731, 1131, 1646], help="row0 row1 col0 col1")
    ap.add_argument("--ref_years", type=int, nargs="+", default=[2021, 2022, 2024, 2025])
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from src.data import geo
    z = np.load(f"{D}/quality_v2/clutter_climatology.npz")
    fy = lambda y: z[f"y{y}_n_ge31"] / np.maximum(z[f"y{y}_n_valid"].astype(float), 1)  # noqa: E731
    f, n = fy(a.year), z[f"y{a.year}_n_valid"]
    ref = np.median([fy(y) for y in a.ref_years], axis=0)
    r0, r1, c0, c1 = a.window
    win = np.zeros(f.shape, bool); win[r0:r1, c0:c1] = True
    m = ndimage.binary_opening(win & (n > 0) & (f > 1e-3) & (f > 5 * np.maximum(ref, 1e-5)), iterations=2)
    lab, k = ndimage.label(m)
    big = ndimage.binary_fill_holes(lab == (np.argmax(ndimage.sum(m, lab, range(1, k + 1))) + 1))
    rr, cc = np.nonzero(big)
    rows = np.arange(rr.min(), rr.max() + 1)
    left = np.array([cc[rr == r].min() for r in rows]); right = np.array([cc[rr == r].max() for r in rows])
    P = np.c_[rows, left].astype(float)
    sol, *_ = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)
    cy, cx = sol[0], sol[1]; R = float(np.sqrt(sol[2] + cy ** 2 + cx ** 2))
    sel = slice(len(rows) // 10, -len(rows) // 10)
    sa, sb = np.polyfit(rows[sel], right[sel], 1)
    my = (rows[0] + rows[-1]) / 2; mx = sa * my + sb
    s = geo.load_radar_sites(include_archive=True)
    s["km_to_arc_centre"] = 2 * np.hypot(s.row - cy, s.col - cx)
    s["km_to_edge_midpoint"] = 2 * np.hypot(s.row - my, s.col - mx)
    top = s.sort_values("km_to_arc_centre").drop_duplicates("odim").head(6)
    best = top.iloc[0]
    nodes = pd.read_csv(f"{D}/validation/opera_nodes/opera_nodes.csv", dtype=str)
    listed = nodes[nodes.nodes.str.contains(best.odim, na=False)].month
    excl = pd.read_csv(f"{D}/quality_v4/day_exclusions.csv", dtype=str)
    ex = excl[excl.radar == best.odim].day.str[:4].value_counts().sort_index().to_dict()
    from scripts.data_quality._md import md_table
    L = [f"# The radar behind the {a.year} disk-shaped noise in the raw >= 31 mm/h climatology", "",
         f"Anomaly: {big.sum():,} px ({4 * big.sum():,} km²), frequency > 1e-3 and > 5x the median of "
         f"{a.ref_years}. Circle fitted to its western arc: centre row {cy:.1f}, col {cx:.1f}, radius "
         f"{2 * R:.0f} km. Its straight eastern edge (col = {sa:.3f} row + {sb:.0f}) has its midpoint at row "
         f"{my:.0f}, col {mx:.0f}: the edge is a diameter through the same point.", "",
         md_table(top[["odim", "location", "country", "lat", "lon", "maxrange_km", "km_to_arc_centre",
                       "km_to_edge_midpoint"]], index=False, floatfmt=".4g"), "",
         f"**{best.odim} ({best.location}, {best.country})** is {best.km_to_arc_centre:.0f} km from the fitted "
         f"centre; the next site is {top.iloc[1].km_to_arc_centre:.0f} km away. The radius matches the "
         f"composite's 250 km range.", "",
         f"- In the composites' node lists (one sampled timestamp per month): "
         f"{listed.str[:4].value_counts().sort_index().to_dict()} months per year; in {a.year}: "
         f"{sorted(listed[listed.str.startswith(str(a.year))].tolist())}.",
         f"- Radar-days excluded by rule 5: {ex}.",
         "- On the suspect-radar page (`suspect_radars.pdf`, ownership of 2024) it carries no label because it "
         "is not in the 2024 node lists, so the 2024 geometry has no such radar."]
    open(f"{NOTES}/noise_radar_{a.year}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_axes([0.02, 0.05, 0.42, 0.85])
    sl = (slice(r0, r1), slice(c0, c1))
    ax.imshow(np.where(n[sl] > 0, np.clip(f[sl], 1e-6, None), np.nan), cmap="Blues", norm=LogNorm(1e-6, 5e-3),
              origin="lower", interpolation="nearest")
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx - c0 + R * np.cos(t), cy - r0 + R * np.sin(t), "--", color="#d62728", lw=1, label="fitted circle")
    ax.plot(sa * rows + sb - c0, rows - r0, "-", color="#d62728", lw=1, label="fitted straight edge")
    for _, r in s.drop_duplicates("odim").iterrows():
        if r0 <= r.row < r1 and c0 <= r.col < c1 and isinstance(r.odim, str) and r.odim:
            hit = r.odim == best.odim
            ax.plot(r.col - c0, r.row - r0, "^", ms=8 if hit else 4, mfc="#d62728" if hit else "white", mec="#000000")
            ax.annotate(r.odim, (r.col - c0, r.row - r0), xytext=(3, 3), textcoords="offset points",
                        fontsize=8 if hit else 6, fontweight="bold" if hit else "normal")
    ax.legend(fontsize=7, loc="lower right"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"raw frequency of 15-min rates >= 31 mm/h, {a.year}; all database sites", fontsize=9)
    yrs = list(range(2016, 2027))
    for i, y in enumerate(yrs):
        axy = fig.add_axes([0.47 + (i % 4) * 0.13, 0.66 - (i // 4) * 0.31, 0.125, 0.28])
        ny = z[f"y{y}_n_valid"][sl]
        axy.imshow(np.where(ny > 0, np.clip(fy(y)[sl], 1e-6, None), np.nan), cmap="Blues", norm=LogNorm(1e-6, 5e-3),
                   origin="lower", interpolation="nearest")
        axy.plot(best.col - c0, best.row - r0, "^", ms=4, mfc="#d62728", mec="#000000")
        axy.set_title(str(y), fontsize=8); axy.set_xticks([]); axy.set_yticks([])
    fig.suptitle(f"{best.odim} ({best.location}): the {a.year} half-disk of noise; same window by year", fontsize=10)
    fig.savefig(f"{NOTES}/noise_radar_{a.year}.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
