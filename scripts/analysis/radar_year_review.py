#!/usr/bin/env python
"""Review pages for candidate radar-years (DECISIONS §36, systematic scan): raw vs screened.

Per radar-year, two pages, 300 x 300 km around the radar:
  1. frequency of 15-min rates >= 31 mm/h: raw over the whole year (clutter climatology) and corrected
     (screened, sampled every 7th day, `rate_clim` chunks) for that year, next to the same radar's other
     years (raw, whole years);
  2. six days drawn at random (seed 0) from the radar-year's days with >= 50 pixel-frames >= 31 mm/h in
     its area (`ranking.csv.gz`): raw day maximum, screened day maximum (the full v4 hourly chain) and the
     screened frame holding the radar's maximum.

    python scripts/analysis/radar_year_review.py mtgud:2017 esclg:2025 ukham:2025 --workers 3
      (-> notes/data_analysis_v4/radar_year_review.pdf, radar_year_review_days.csv)
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2"), os.path.dirname(__file__)]
import radar_band_tail as bt  # noqa: E402
from suspect_radar_gallery import overlay  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
NOTES = f"{REPO}/notes/data_analysis_v4"
HALF = 75


def _crop(g, radar):
    k = list(g.keys).index(radar)
    r, c = int(g.sites["row"].iat[k]), int(g.sites["col"].iat[k])
    return k, (slice(max(r - HALF, 0), r + HALF), slice(max(c - HALF, 0), c + HALF))


def replay(args):
    radar, day = args
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    year = int(day[:4])
    g = bt._geometry(year)
    k, sl = _crop(g, radar)
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(f"{D}/quality_v2/clutter_climatology.npz", y),
                   ring=lambda y: st._ring(f"{D}/quality_v2/ring_mask.npz", y), sites_rc=st._sites(), ceilings=ceil,
                   cfg={**q.get("hourly", {}), "advection": None}).run()
    d0 = np.datetime64(pd.Timestamp(day))
    idx = [t for t, ts in enumerate(hd.frames) if d0 < ts <= d0 + np.timedelta64(1, "D")]
    raw = hd.raw[idx][(slice(None),) + sl]
    fin = hd.fin[idx][(slice(None),) + sl]
    mine = g.owner[sl] == k
    fm = np.where(mine[None], np.nan_to_num(fin), -1).reshape(len(idx), -1).max(1)
    tb = int(np.argmax(fm))
    return radar, day, np.nanmax(raw, 0), np.nanmax(fin, 0), fin[tb], pd.Timestamp(hd.frames[idx[tb]]).strftime("%H:%M"), float(fm[tb])


def climatology_page(pdf, radar, year):
    z = np.load(f"{D}/quality_v2/clutter_climatology.npz")
    g = bt._geometry(year)
    _, sl = _crop(g, radar)
    yrs = [y for y in range(2016, 2027) if f"y{y}_n_valid" in z.files and z[f"y{y}_n_valid"][sl].sum() > 0]
    tot = None
    for p in glob.glob(f"{D}/quality_v4/rate_clim/chunks/{year}_*.npz"):
        c = np.load(p)
        tot = {k: c[k][sl].astype(np.int64) for k in ("cor_valid", "cor_ge31")} if tot is None else \
            {k: tot[k] + c[k][sl] for k in tot}
    panels = [(f"{year} raw (whole year)", z[f"y{year}_n_ge31"][sl] / np.maximum(z[f"y{year}_n_valid"][sl], 1),
               z[f"y{year}_n_valid"][sl] > 0)]
    if tot is not None:
        panels.append((f"{year} corrected (every 7th day)", tot["cor_ge31"] / np.maximum(tot["cor_valid"], 1),
                       tot["cor_valid"] > 0))
    others = [y for y in yrs if y != year][-6:]
    panels += [(f"{y} raw", z[f"y{y}_n_ge31"][sl] / np.maximum(z[f"y{y}_n_valid"][sl], 1), z[f"y{y}_n_valid"][sl] > 0)
               for y in others]
    n = len(panels)
    cols = 4
    fig, axs = plt.subplots((n + cols - 1) // cols, cols, figsize=(15, 4 * ((n + cols - 1) // cols)), squeeze=False)
    norm = LogNorm(1e-6, 5e-3)
    for ax, (t, f, ok) in zip(axs.ravel(), panels):
        im = ax.imshow(np.where(ok, np.clip(f, 1e-6, None), np.nan), cmap="Blues", norm=norm, origin="lower",
                       interpolation="nearest")
        ax.set_facecolor("#f4f4f4")
        overlay(ax, g, sl, mark=radar)
        ax.set_title(t, fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.02, label="frequency of 15-min rates >= 31 mm/h")
    fig.suptitle(f"{radar} {year}: raw vs corrected frequency of rates >= 31 mm/h, and the radar's other years "
                 "(300 x 300 km; grey = ownership, black = national seam)", fontsize=10)
    pdf.savefig(fig); plt.close(fig)


def days_page(pdf, radar, year, res):
    g = bt._geometry(year)
    _, sl = _crop(g, radar)
    fig, axs = plt.subplots(len(res), 3, figsize=(11.5, 3.6 * len(res)), squeeze=False)
    norm = LogNorm(0.1, 100)
    for i, (_, day, rmax, fmax, frame, when, best) in enumerate(res):
        for j, (img, t) in enumerate(((rmax, f"{day} raw day max ({np.nanmax(rmax):.0f} mm/h)"),
                                      (fmax, f"screened day max ({np.nanmax(fmax):.0f} mm/h)"),
                                      (frame, f"screened {when} UTC: {radar}'s max ({best:.0f} mm/h)"))):
            ax = axs[i, j]
            im = ax.imshow(np.where(np.nan_to_num(img) >= 0.1, img, np.nan), cmap="Blues", norm=norm, origin="lower",
                           interpolation="nearest")
            ax.set_facecolor("#f4f4f4")
            overlay(ax, g, sl, mark=radar)
            ax.set_title(t, fontsize=8)
    fig.colorbar(im, ax=axs, fraction=0.02, label="mm/h (15-min rate)")
    fig.suptitle(f"{radar} {year}: 6 random days with >= 50 pixel-frames >= 31 mm/h (seed 0); raw, screened, "
                 "and the screened frame of the radar's maximum", fontsize=10)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("radar_years", nargs="+", help="radar:year")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    rk = pd.read_csv(f"{D}/quality_v4/ranking.csv.gz", dtype={"day": str}, usecols=["radar", "day", "n31"])
    rng = np.random.default_rng(0)
    picks = {}
    for ry in a.radar_years:
        radar, year = ry.split(":")
        pool = rk[(rk.radar == radar) & (rk.day.str[:4] == year) & (rk.n31 >= 50)].day.values
        picks[(radar, int(year))] = sorted(rng.choice(pool, size=min(6, len(pool)), replace=False))
        print(radar, year, len(pool), "qualifying days; drawn:", picks[(radar, int(year))], flush=True)
    jobs = [(r, d) for (r, _), ds in picks.items() for d in ds]
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(replay, jobs))
    by = {}
    for o in out:
        by.setdefault(o[0], []).append(o)
    path = f"{NOTES}/radar_year_review.pdf"
    with PdfPages(path) as pdf:
        for (radar, year), _ in picks.items():
            climatology_page(pdf, radar, year)
            days_page(pdf, radar, year, sorted(by[radar], key=lambda o: o[1]))
    pd.DataFrame([(r, y, d) for (r, y), ds in picks.items() for d in ds], columns=["radar", "year", "day"]).to_csv(
        f"{NOTES}/radar_year_review_days.csv", index=False)
    print("->", path)


if __name__ == "__main__":
    main()
