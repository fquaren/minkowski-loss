#!/usr/bin/env python
"""Per-pixel frequency of 15-min rates >= 1 and >= 31 mm/h, raw vs corrected, on the same frames.

`clutter_climatology.py` counts raw frames only. This replays the v4 hourly chain (`HourlyDay`, as
`build_store_hourly.py`, no advection) on a sample of days and counts, per pixel and frame:
  - raw:       the composite as archived;
  - corrected: the chain's final frames (every pixel repair), with the pixels removed where the
               owning radar (contributing geometry, §27) is excluded for that frame: a flagged
               radar-frame (rule 3, `frame_flags.csv.gz`), an excluded radar-day (rule 5,
               `day_exclusions.csv`) or a flagged radar-year (§36b/c lists). The dataset removes
               whole tile-hours by owner share; owner masking is the per-pixel analogue.
Tile-hour-level rules (ownerless tiles, unphysical tile maxima) are not applied.

    python scripts/data_quality/rate_climatology.py run --step 7 --workers 4   (-> quality_v4/rate_clim/)
    python scripts/data_quality/rate_climatology.py maps                       (-> notes/data_analysis_v4/)
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2"), os.path.join(REPO, "scripts", "analysis")]
D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
OUT = f"{Q4}/rate_clim"
NOTES = f"{REPO}/notes/data_analysis_v4"
BIASED = [f"{Q4}/biased_radar_years_tail.csv", f"{Q4}/biased_radar_years_national.csv"]
KEYS = ("raw_valid", "raw_ge1", "raw_ge31", "cor_valid", "cor_ge1", "cor_ge31")
_W = {}


def _init():
    ff = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    de = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    by = set()
    for p in BIASED:
        if os.path.exists(p):
            b = pd.read_csv(p, dtype={"radar": str, "year": str})
            by |= set(zip(b.radar, b.year))
    _W.update(ff=set(zip(ff.radar, ff.timestamp)), de=set(zip(de.radar, de.day)), by=by)


def run_chunk(args):
    year, days = args
    import scan_tiles as st
    import radar_band_tail as bt
    from src.data.hourly import HourlyDay
    if not _W:
        _init()
    path = f"{OUT}/chunks/{year}_{days[0]}.npz"
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    g = bt._geometry(int(year))
    keys = np.array(list(g.keys))
    acc = {k: np.zeros(g.owner.shape, np.uint16) for k in KEYS}
    nfr = 0
    for day in days:
        hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(f"{D}/quality_v2/clutter_climatology.npz", y),
                       ring=lambda y: st._ring(f"{D}/quality_v2/ring_mask.npz", y), sites_rc=st._sites(), ceilings=ceil,
                       cfg={**q.get("hourly", {}), "advection": None}).run()
        if getattr(hd, "raw", None) is None:
            continue
        d0 = np.datetime64(pd.Timestamp(day))
        excl_day = np.array([(k, day) in _W["de"] or (k, year) in _W["by"] for k in keys])
        for t, ts in enumerate(hd.frames):
            if not (d0 < ts <= d0 + np.timedelta64(1, "D")):
                continue
            stamp = pd.Timestamp(ts).strftime("%Y%m%d%H%M%S")
            ex = excl_day | np.array([(k, stamp) in _W["ff"] for k in keys])
            bad = ex[np.maximum(g.owner, 0)] & (g.owner >= 0)
            r, f = hd.raw[t], hd.fin[t]
            rv, fv = np.isfinite(r), np.isfinite(f) & ~bad
            acc["raw_valid"] += rv
            acc["raw_ge1"] += rv & (r >= 1)
            acc["raw_ge31"] += rv & (r >= 31)
            acc["cor_valid"] += fv
            acc["cor_ge1"] += fv & (f >= 1)
            acc["cor_ge31"] += fv & (f >= 31)
            nfr += 1
        del hd
    np.savez_compressed(path, n_frames=nfr, days=np.array(days), **acc)
    return year, days[0], nfr


def cmd_run(step, workers, start, end):
    days = pd.date_range(start, end, freq=f"{step}D").strftime("%Y%m%d")
    days = [d for d in days if os.path.isdir(f"{D}/raw/OPERA/{d}")]
    chunks = []
    for y, grp in pd.Series(days).groupby(pd.Series(days).str[:4]):
        g = grp.tolist()
        chunks += [(y, g[i:i + 6]) for i in range(0, len(g), 6)]
    os.makedirs(f"{OUT}/chunks", exist_ok=True)
    todo = [c for c in chunks if not os.path.exists(f"{OUT}/chunks/{c[0]}_{c[1][0]}.npz")]
    print(f"[rate_clim] {len(days)} days (every {step}th, {start}-{end}), {len(chunks)} chunks, {len(todo)} to do",
          flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, (y, d, n) in enumerate(ex.map(run_chunk, todo)):
            print(f"  {k + 1}/{len(todo)} chunks ({y} from {d}, {n} frames)", flush=True)


def load():
    tot, nfr, ndays = None, 0, 0
    for p in sorted(glob.glob(f"{OUT}/chunks/*.npz")):
        z = np.load(p)
        if tot is None:
            tot = {k: z[k].astype(np.int32) for k in KEYS}
        else:
            for k in KEYS:
                tot[k] += z[k]
        nfr += int(z["n_frames"]); ndays += len(z["days"])
    return tot, nfr, ndays


def _overlay(ax, g, sl, sites, fs=6):
    import radar_band_tail as bt  # noqa: F401
    from suspect_radar_gallery import edges, GREY, BLACK
    own = g.owner[sl]
    ctry = np.array([k[:2] for k in g.keys])
    names = sorted(set(ctry))
    cm = np.where(own >= 0, np.searchsorted(names, ctry[np.maximum(own, 0)]), -1)
    ax.imshow(np.ma.masked_where(~edges(own), own), cmap=GREY, origin="lower", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~edges(cm), cm), cmap=BLACK, origin="lower", interpolation="nearest")
    r0, c0 = sl[0].start, sl[1].start
    H, W = own.shape
    for k, r, c in sites:
        if 0 <= r - r0 < H and 0 <= c - c0 < W:
            ax.plot(c - c0, r - r0, "^", ms=4, mfc="white", mec="#000000", mew=0.6)
            ax.annotate(k, (c - c0, r - r0), xytext=(2, 2), textcoords="offset points", fontsize=fs, color="#1a1a1a")
    ax.set_xticks([]); ax.set_yticks([])


def cmd_maps(ref_year=2021):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    import radar_band_tail as bt
    tot, nfr, ndays = load()
    g = bt._geometry(ref_year)
    sites = {}
    for y in range(2016, 2026):                       # every radar contributing in a sampled year
        gy = bt._geometry(y)
        for k, r, c in zip(gy.keys, gy.sites["row"], gy.sites["col"]):
            sites.setdefault(k, (k, int(r), int(c)))
    sites = list(sites.values())
    H, W = tot["raw_valid"].shape
    rows = [(0, 760), (720, 1480), (1440, 2200)]
    cols = [(0, 680), (610, 1290), (1220, 1900)]
    subs = [(slice(*r), slice(*c)) for r in rows[::-1] for c in cols]   # north first
    rc = np.array([(r, c) for k, r, c in sites if k[:2] in ("hu", "sk", "ro")])
    sus = (slice(max(int(rc[:, 0].min()) - 120, 0), int(rc[:, 0].max()) + 120),
           slice(max(int(rc[:, 1].min()) - 120, 0), int(rc[:, 1].max()) + 120))
    info = (f"{ndays} days (every 7th day 2016-2025), {nfr:,} 15-min frames; raw and corrected on the same "
            f"frames. Grey: radar ownership ({ref_year}); black: national seams; triangles: every radar "
            "contributing in 2016-2025.")
    for u, lo in ((31, 1e-6), (1, 1e-3)):
        fr = {s: np.where(tot[f"{s}_valid"] > 0, tot[f"{s}_ge{u}"] / np.maximum(tot[f"{s}_valid"], 1), np.nan)
              for s in ("raw", "cor")}
        hi = float(np.nanpercentile(fr["raw"], 99.9))
        norm = LogNorm(lo, hi)
        out = f"{NOTES}/rate_freq{u}_raw_vs_corrected.pdf"
        with PdfPages(out) as pdf:
            pages = [("Whole domain", (slice(0, H), slice(0, W)))]
            if u == 31:
                pages.append(("Hungary, Slovakia, Romania (the suspect-radar page)", sus))
            pages += [(f"Subdomain {i + 1} of 9 (rows {s[0].start}-{s[0].stop}, cols {s[1].start}-{s[1].stop})", s)
                      for i, s in enumerate(subs)]
            for title, sl in pages:
                if not np.isfinite(fr["raw"][sl]).any():
                    continue
                fig, axs = plt.subplots(1, 2, figsize=(16, 8.4))
                for ax, s, lab in ((axs[0], "raw", "raw composite"), (axs[1], "cor", "corrected (v4 screening)")):
                    v = fr[s][sl]
                    im = ax.imshow(np.where(np.isfinite(v), np.clip(v, lo, None), np.nan), cmap="Blues", norm=norm,
                                   origin="lower", interpolation="nearest")
                    _overlay(ax, g, sl, sites, fs=5 if sl[0].stop - sl[0].start > 1000 else 6)
                    n_r = np.nansum(tot[f"{s}_ge{u}"][sl])
                    ax.set_title(f"{lab}: frequency of 15-min rates >= {u} mm/h ({n_r:,} exceedances)", fontsize=9)
                    if title == "Whole domain":
                        for ss in subs:
                            ax.add_patch(plt.Rectangle((ss[1].start, ss[0].start), ss[1].stop - ss[1].start,
                                                       ss[0].stop - ss[0].start, fill=False, ec="#d62728", lw=0.8))
                fig.colorbar(im, ax=axs, fraction=0.025, label="fraction of valid frames")
                fig.suptitle(f"{title}\n{info}", fontsize=9)
                pdf.savefig(fig, dpi=150); plt.close(fig)
        print("->", out)
    rm = 1 - tot["cor_ge31"].sum() / max(tot["raw_ge31"].sum(), 1)
    print(f"exceedances >= 31 removed: {rm:.1%}; >= 1: {1 - tot['cor_ge1'].sum() / max(tot['raw_ge1'].sum(), 1):.1%}; "
          f"valid pixel-frames removed: {1 - tot['cor_valid'].sum() / max(tot['raw_valid'].sum(), 1):.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "maps"])
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--out", default=None, help="chunk directory (default quality_v4/rate_clim)")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    cmd_run(a.step, a.workers, a.start, a.end) if a.cmd == "run" else cmd_maps()
