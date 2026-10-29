#!/usr/bin/env python
"""Which radars carry the v4 tail, and does it look like weather or like failure?

Kept tile-hours of the full-archive scan (split-time rejections applied), attributed to the radar
owning most of the tile. Per radar (post-break, from 2015-12-12, unless --all):
  p20            pixel frequency >= 20 mm in 1 h
  top1%_share    share of the radar's >= 20 mm pixel-hours on its top 1% of days: failures
                 concentrate on a few days, weather spreads over many
  days20         days with any pixel >= 20 mm
  dry            share of tile-hours with max < 0.1 mm (a noise floor keeps tiles from being dry)
  median_mean    median tile mean (mm)

    python scripts/analysis/radar_tail.py [--workers 2]
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
OUT = f"{REPO}/notes/data_analysis_v4"
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
    t = t.assign(radar=own, day=os.path.basename(f)[:8])
    t = t[t.radar != ""]
    return t.groupby(["radar", "day"]).agg(n=("max", "size"), n20=("n_ge20", "sum"), n1=("n_ge1", "sum"),
                                           dry=("max", lambda x: (x < 0.1).sum()), mean_med=("mean", "median")).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--all", action="store_true", help="include pre-break years")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{Q4}/hourly/hours/*.csv.gz"))
    if not a.all:
        files = [f for f in files if os.path.basename(f)[:8] >= "20151212"]
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        rd = pd.concat([x for x in ex.map(_day, files, chunksize=20) if x is not None], ignore_index=True)
    rd.to_csv(f"{OUT}/radar_day_tail.csv.gz", index=False)
    rows = []
    for r, g in rd.groupby("radar"):
        tot20 = g.n20.sum()
        k = max(1, int(round(0.01 * len(g))))
        top = g.n20.nlargest(k).sum()
        rows.append({"radar": r, "country": r[:2], "radar-days": len(g), "tile-hours": int(g.n.sum()),
                     "p20": tot20 / (g.n.sum() * 128 * 128), "p1": g.n1.sum() / (g.n.sum() * 128 * 128),
                     "days20": int((g.n20 > 0).sum()), "top1%_share": top / tot20 if tot20 else np.nan,
                     "dry": g.dry.sum() / g.n.sum(), "median_mean": g.mean_med.median()})
    t = pd.DataFrame(rows)
    t = t[t["tile-hours"] >= 5000].sort_values("p20", ascending=False)
    t.to_csv(f"{OUT}/radar_tail.csv", index=False)
    from scripts.data_quality._md import md_table
    med = t[["p20", "top1%_share", "dry", "median_mean"]].median()
    L = [f"# Radars carrying the v4 tail ({'all years' if a.all else 'post-break, from 2015-12-12'})", "",
         f"{len(t)} radars with >= 5,000 tile-hours. Medians over radars: p20 {med.p20:.2e}, top-1%-days share "
         f"{med['top1%_share']:.2f}, dry {med.dry:.2f}, median tile mean {med.median_mean:.4f} mm.", "",
         "## Top 25 by p20", "", md_table(t.head(25), index=False, floatfmt=".3g"), "",
         "## Slovakia, Romania, Hungary", "", md_table(t[t.country.isin(["sk", "ro", "hu"])], index=False, floatfmt=".3g")]
    txt = "\n".join(L)
    open(f"{OUT}/radar_tail.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
