#!/usr/bin/env python
"""Images of the radars suspected of reading high (DECISIONS §36): robob, hunap, husze, skkoj.

Page 1: heavy-rate frequency maps of the region (clutter climatology: raw 15-min frames before
screening, 2016-2025), with radar ownership boundaries (grey) and national seams (black).
Then one page per radar: 6 radar-days drawn at random (seed 0) from its post-break days with a
tile-hour >= 20 mm (`radar_day_tail.csv.gz`, the days that make it stand out). Per day: the day's
maximum 15-min rate after `clean_frame`, and the frame holding the maximum inside the radar's area;
300 x 300 km around the radar.

    python scripts/analysis/suspect_radar_gallery.py   (-> notes/data_analysis_v4/suspect_radars.pdf)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))
import radar_band_tail as bt  # noqa: E402  (geometry, hot mask, paths)

SUSPECTS = ["robob", "hunap", "husze", "skkoj"]
NOTES = f"{REPO}/notes/data_analysis_v4"
HALF = 75                                     # px: 150 km either side
GREY = ListedColormap(["#8a8a8a"])
BLACK = ListedColormap(["#000000"])


def edges(lab):
    """Pixels whose right or upper neighbour has a different label (both labelled)."""
    e = np.zeros(lab.shape, bool)
    e[:, :-1] |= (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] >= 0) & (lab[:, 1:] >= 0)
    e[:-1, :] |= (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] >= 0) & (lab[1:, :] >= 0)
    return e


def overlay(ax, g, sl, mark=None):
    own = g.owner[sl]
    ctry = np.array([k[:2] for k in g.keys])
    names = sorted(set(ctry))
    cm = np.where(own >= 0, np.searchsorted(names, ctry[np.maximum(own, 0)]), -1)
    ax.imshow(np.ma.masked_where(~edges(own), own), cmap=GREY, origin="lower", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~edges(cm), cm), cmap=BLACK, origin="lower", interpolation="nearest")
    r0, c0 = sl[0].start, sl[1].start
    H, W = own.shape
    for k, r, c in zip(g.keys, g.sites["row"], g.sites["col"]):
        rr, cc = r - r0, c - c0
        if 0 <= rr < H and 0 <= cc < W:
            ax.plot(cc, rr, "^", ms=6 if k == mark else 4, mfc="white" if k != mark else "#000000",
                    mec="#000000", mew=0.8)
            ax.annotate(k, (cc, rr), xytext=(3, 3), textcoords="offset points", fontsize=6,
                        fontweight="bold" if k == mark else "normal", color="#1a1a1a")
    ax.set_xticks([]); ax.set_yticks([])


def climatology_page(pdf):
    z = np.load(bt.CLIM)
    yrs = range(2016, 2026)
    nv = sum(z[f"y{y}_n_valid"].astype(np.float64) for y in yrs)
    f31 = sum(z[f"y{y}_n_ge31"].astype(np.float64) for y in yrs) / np.maximum(nv, 1)
    f1 = sum(z[f"y{y}_n_ge1"].astype(np.float64) for y in yrs) / np.maximum(nv, 1)
    g = bt._geometry(2024)
    rc = g.sites[g.sites.key.str[:2].isin(["hu", "sk", "ro"])][["row", "col"]].values
    sl = (slice(max(int(rc[:, 0].min()) - 120, 0), int(rc[:, 0].max()) + 120),
          slice(max(int(rc[:, 1].min()) - 120, 0), int(rc[:, 1].max()) + 120))
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2))
    for ax, f, lab, lo in ((axs[0], f1, ">= 1 mm/h", 1e-3), (axs[1], f31, ">= 31 mm/h", 1e-6)):
        v = np.where(nv > 0, f, np.nan)[sl]
        im = ax.imshow(np.clip(v, lo, None), cmap="Blues", norm=LogNorm(lo, np.nanpercentile(v, 99.9)),
                       origin="lower", interpolation="nearest")
        overlay(ax, g, sl)
        ax.set_title(f"frequency of 15-min rates {lab}, 2016-2025", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04, label="fraction of valid frames")
    fig.suptitle("Hungary, Slovakia, Romania and neighbours: raw frames before screening (clutter climatology); "
                 "grey = radar ownership (2024), black = national seams", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def radar_page(pdf, radar, days):
    from src.data.cleaning import clean_frame
    from src.data.day_cleaner import open_day
    fig, axs = plt.subplots(3, 4, figsize=(14, 11))
    norm = LogNorm(0.1, 100)
    for i, day in enumerate(days):
        year = int(day[:4])
        g = bt._geometry(year)
        k = list(g.keys).index(radar)
        r, c = int(g.sites["row"].iat[k]), int(g.sites["col"].iat[k])
        sl = (slice(max(r - HALF, 0), r + HALF), slice(max(c - HALF, 0), c + HALF))
        mine = g.owner[sl] == k
        ds = open_day(f"{bt.D}/raw/OPERA/{day}")
        hot = bt._hot(year)
        best, bt_i, dmax = -1, None, None
        for t in range(ds.sizes["time"]):
            f = clean_frame(ds["TOT_PREC"].isel(time=t).values.astype(np.float32), hot)[0][sl]
            dmax = f if dmax is None else np.fmax(dmax, f)
            m = np.nanmax(np.where(mine, f, np.nan)) if np.isfinite(f[mine]).any() else -1
            if m > best:
                best, bt_i, frame = m, t, f
        when = str(ds["time"].values[bt_i])[:16].replace("T", " ")
        ds.close()
        for j, (img, ttl) in enumerate(((dmax, f"{day}: day max (max {np.nanmax(dmax):.0f} mm/h)"),
                                        (frame, f"{when} UTC: frame of {radar}'s max ({best:.0f} mm/h)"))):
            ax = axs[i // 2, 2 * (i % 2) + j]
            im = ax.imshow(np.where(img >= 0.1, img, np.nan), cmap="Blues", norm=norm, origin="lower",
                           interpolation="nearest")
            ax.set_facecolor("#f4f4f4")
            overlay(ax, g, sl, mark=radar)
            ax.set_title(ttl, fontsize=8)
    fig.colorbar(im, ax=axs, fraction=0.02, label="mm/h (15-min rate, clean_frame)")
    fig.suptitle(f"{radar}: 6 random post-break radar-days with a tile-hour >= 20 mm (seed 0); "
                 "300 x 300 km; grey = ownership, black = national seam, light grey = no data", fontsize=10)
    pdf.savefig(fig); plt.close(fig)


def main():
    rd = pd.read_csv(f"{NOTES}/radar_day_tail.csv.gz", dtype={"day": str})
    rng = np.random.default_rng(0)
    out = f"{NOTES}/suspect_radars.pdf"
    rows = []
    with PdfPages(out) as pdf:
        climatology_page(pdf)
        for radar in SUSPECTS:
            pool = rd[(rd.radar == radar) & (rd.n20 > 0)].day.values
            days = sorted(rng.choice(pool, size=min(6, len(pool)), replace=False))
            print(radar, len(pool), "days with a tile-hour >= 20 mm; drawn:", days, flush=True)
            rows += [(radar, d) for d in days]
            radar_page(pdf, radar, days)
    pd.DataFrame(rows, columns=["radar", "day"]).to_csv(f"{NOTES}/suspect_radars_days.csv", index=False)
    print("->", out)


if __name__ == "__main__":
    main()
