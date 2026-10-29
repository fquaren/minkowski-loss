#!/usr/bin/env python
"""Climatology of the v4 dataset from the full-archive hourly scan (every tile-hour, not the
stratified store), with the split-time rejections applied (current rule 5, frame flags).

  load     per tile-hour: max, mean, wet_frac, pixel exceedance counts, owning country,
           latitude, era; cached to <OUT>/tilehours.pkl
  report   tables and figures:
           - dry fractions: tile-hours with max < 0.1 mm; tile-days with no wet hour
           - tile-hour max: mean, median, quantiles; pixel exceedance frequencies at
             1 / 5 / 10 / 20 / 50 mm (counts / valid pixel-hours, assuming full tiles)
           - by latitude band, country (radar owning most of the tile), season, year, era
           - maps: mean hourly accumulation and exceedance frequency >= 20 mm per tile

    python scripts/analysis/v4_stats.py load --workers 2
    python scripts/analysis/v4_stats.py report
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
HOURS = f"{Q4}/hourly/hours"
OUT = f"{REPO}/notes/data_analysis_v4"
NIMBUS_START = "20240705"
LEVELS = {"n_ge1": 1, "n_ge5": 5, "n_ge10": 10, "n_ge20": 20, "n_ge50": 50}
LAT_BANDS = [(-90, 46, "<46N"), (46, 50, "46-50N"), (50, 55, "50-55N"), (55, 90, ">=55N")]
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
          9: "SON", 10: "SON", 11: "SON"}
_W = {}


def _init():
    from src.data.radar_screen import decide_hourly
    q = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))
    ff = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    de = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    _W.update(q=q, ff=set(zip(ff.radar, ff.timestamp)), de=set(zip(de.radar, de.day)), decide=decide_hourly)


def _day(f):
    t = pd.read_csv(f, dtype={"end": str, "frames": str, "owners": str})
    if not len(t):
        return None
    if "n_burst" not in t:
        t["n_burst"] = 0
    d = _W["decide"](t, _W["q"], frame_flags=_W["ff"], day_excl=_W["de"])
    t = t[~(t["unphysical"].astype(bool).values | d["rej_v4"].values)]
    own = t.owners.fillna("").str.split(";").str[0].str.split(":").str[0]
    return pd.DataFrame({"day": os.path.basename(f)[:8], "end": t.end.str[:12], "row": t.row.astype(np.int16),
                         "col": t.col.astype(np.int16), "max": t["max"].astype(np.float32),
                         "mean": t["mean"].astype(np.float32), "wet_frac": t.wet_frac.astype(np.float32),
                         "country": own.str[:2], **{k: t[k].astype(np.int32) for k in LEVELS}})


def cmd_load(workers):
    from src.data.geo import tile_location
    files = sorted(glob.glob(f"{HOURS}/*.csv.gz"))
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        df = pd.concat([x for x in ex.map(_day, files, chunksize=20) if x is not None], ignore_index=True)
    loc = {(r, c): tile_location(int(r), int(c)) for r, c in df[["row", "col"]].drop_duplicates().itertuples(index=False)}
    df["lat"] = np.array([loc[(r, c)]["lat"] for r, c in zip(df.row, df.col)], np.float32)
    df["lon"] = np.array([loc[(r, c)]["lon"] for r, c in zip(df.row, df.col)], np.float32)
    os.makedirs(OUT, exist_ok=True)
    df.to_pickle(f"{OUT}/tilehours.pkl")
    print(f"[load] {len(files)} days, {len(df):,} kept tile-hours")


def summarise(g):
    npx = 128 * 128 * len(g)
    q = g["max"].quantile([0.5, 0.9, 0.99, 0.999, 0.9999]).values
    out = {"tile-hours": len(g), "dry (max < 0.1)": (g["max"] < 0.1).mean(), "mean (mm/h-tile)": g["mean"].mean(),
           "median tile mean": g["mean"].median(), "wet frac": g.wet_frac.mean(),
           "max q50": q[0], "max q90": q[1], "max q99": q[2], "max q99.9": q[3], "max q99.99": q[4],
           "max (all)": g["max"].max()}
    for k, v in LEVELS.items():
        out[f"P(px >= {v})"] = g[k].sum() / npx
    return pd.Series(out)


def cmd_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_pickle(f"{OUT}/tilehours.pkl")
    df["season"] = df.day.str[4:6].astype(int).map(SEASON)
    df["year"] = df.day.str[:4]
    df["era"] = np.where(df.day >= NIMBUS_START, "NIMBUS", "ODYSSEY")
    df["lat_band"] = pd.cut(df.lat, [b[0] for b in LAT_BANDS] + [90], labels=[b[2] for b in LAT_BANDS], right=False)
    # zero-rain days per tile: a tile-day with no tile-hour >= 0.1 mm
    td = df.groupby(["row", "col", "day"])["max"].max()
    L = ["# v4 dataset climatology (full-archive hourly scan, kept tile-hours)", "",
         f"{len(df):,} tile-hours on {df.day.nunique():,} days, {df[['row', 'col']].drop_duplicates().shape[0]} tile positions. "
         f"Values in mm accumulated in 1 h. Pixel frequencies assume full 128x128 tiles.", "",
         f"Zero-rain tile-days (no hour >= 0.1 mm anywhere in the tile): {(td < 0.1).mean():.1%} of {len(td):,}.", ""]
    from scripts.data_quality._md import md_table
    fmt = lambda t: md_table(t.reset_index(), index=False, floatfmt=".4g")
    L += ["## All", "", fmt(summarise(df).to_frame("all").T), ""]
    for by, title in (("lat_band", "By latitude band"), ("country", "By country (radar owning most of the tile)"),
                      ("season", "By season"), ("era", "By product era"), ("year", "By year")):
        t = df.groupby(by, observed=True).apply(summarise)
        if by == "country":
            t = t[t["tile-hours"] >= 20000].sort_values("tile-hours", ascending=False)
        L += [f"## {title}", "", fmt(t), ""]
    open(f"{OUT}/stats.md", "w").write("\n".join(L) + "\n")
    # maps per tile position
    m = df.groupby(["row", "col"]).agg(mean=("mean", "mean"), n=("max", "size"), n20=("n_ge20", "sum")).reset_index()
    m["p20"] = m.n20 / (m.n * 128 * 128)
    H, W = 2200, 1900
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), constrained_layout=True)
    for ax, col, lab, cm in ((axs[0], "mean", "mean accumulation per hour (mm)", "Blues"),
                             (axs[1], "p20", "pixel frequency >= 20 mm in 1 h", "Blues")):
        grid = np.full((H // 128 + 1, W // 128 + 1), np.nan)
        for r, c, v in zip(m.row, m.col, m[col]):
            grid[r // 128, c // 128] = v
        im = ax.imshow(grid, origin="lower", cmap=cm)                     # stores are south-first
        ax.set_title(lab, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("v4 per-tile climatology (south at the bottom)")
    fig.savefig(f"{OUT}/maps.png", dpi=120); plt.close(fig)
    # exceedance curves of the tile-hour max by latitude band
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    xs = np.logspace(-1, 2.6, 200)
    for b, c in zip([b[2] for b in LAT_BANDS], ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):   # palette order
        s = np.sort(df.loc[df.lat_band == b, "max"].values)
        ax.plot(xs, 1 - np.searchsorted(s, xs) / len(s), lw=1.5, color=c, label=b)
    ax.set(xscale="log", yscale="log", xlabel="tile-hour maximum x (mm in 1 h)", ylabel="P(max >= x)", ylim=(1e-7, 1))
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/exceedance_by_latitude.png", dpi=120); plt.close(fig)
    print("\n".join(L[:12]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["load", "report"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cmd_load(a.workers) if a.cmd == "load" else cmd_report()
