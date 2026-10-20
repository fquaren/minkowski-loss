#!/usr/bin/env python
"""Tail-focused boundary test for radars biased against their neighbours (DECISIONS §36, option b).

Per radar and day (every second day), summed over the day's frames of the clean_frame field
(contributing geometry): in the 6 km band just inside the radar's owned area and the 6 km band just
outside it, valid pixels and pixels >= 10 / >= 31 mm/h.
Per radar-year: R10 = inside / outside frequency of >= 10 mm/h, 95% day-block interval of log R10;
flagged if R10 >= 2, lower end >= 1.25 and >= 20 days with a pixel >= 10 mm/h in either band.

    python scripts/analysis/radar_band_tail.py run --workers 2      (-> quality_v4/band_tail/days/)
    python scripts/analysis/radar_band_tail.py analyze              (-> biased_radar_years_tail.csv)
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
OUT = f"{Q4}/band_tail"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
NOTES = f"{REPO}/notes/data_analysis_v4"
_G = {}


def _geometry(year):
    from src.data.radar_screen import RadarGeometry, use_contributing
    if not _G:
        rg = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))["radar_geometry"]
        use_contributing(rg["nodes_table"], rg.get("valid_climatology"))
        _G["hot"] = {}
    if year not in _G:
        _G[year] = RadarGeometry(year)
    return _G[year]


def _hot(year):
    from src.data.cleaning import hot_mask_from_climatology
    if "clim" not in _G:
        _G["clim"] = dict(np.load(CLIM))
    if year not in _G["hot"]:
        _G["hot"][year] = hot_mask_from_climatology(_G["clim"], year)
    return _G["hot"][year]


def run_day(day):
    from src.data.cleaning import clean_frame
    from src.data.day_cleaner import open_day
    year = int(day[:4])
    g = _geometry(year)
    hot = _hot(year)
    R = len(g.keys)
    acc = {k: np.zeros(R) for k in ("in_valid", "in10", "in31", "out_valid", "out10", "out31")}
    ds = open_day(f"{D}/raw/OPERA/{day}")
    for t in range(ds.sizes["time"]):
        c = clean_frame(ds["TOT_PREC"].isel(time=t).values.astype(np.float32), hot)[0].ravel()
        for side, idx, rr in (("in", g.b_in, g.b_in_r), ("out", g.b_out, g.b_out_r)):
            v = c[idx]
            ok = np.isfinite(v)
            acc[f"{side}_valid"] += np.bincount(rr[ok], minlength=R)
            acc[f"{side}10"] += np.bincount(rr[ok], weights=(v[ok] >= 10).astype(float), minlength=R)
            acc[f"{side}31"] += np.bincount(rr[ok], weights=(v[ok] >= 31).astype(float), minlength=R)
    ds.close()
    df = pd.DataFrame({"radar": g.keys, "day": day, **acc})
    df = df[(df.in_valid > 0) & (df.out_valid > 0)]
    os.makedirs(f"{OUT}/days", exist_ok=True)
    df.to_csv(f"{OUT}/days/{day}.csv.gz", index=False)
    return day, len(df)


def cmd_run(workers):
    days = sorted(os.path.basename(p) for p in glob.glob(f"{D}/raw/OPERA/20*") if os.path.isdir(p))
    days = days[::2]
    todo = [d for d in days if not os.path.exists(f"{OUT}/days/{d}.csv.gz")]
    print(f"[band_tail] {len(days)} days (every second), {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, (d, n) in enumerate(ex.map(run_day, todo, chunksize=2)):
            if k % 50 == 0:
                print(f"  {k + 1}/{len(todo)} days ({d}, {n} radars)", flush=True)


def cmd_analyze():
    from scripts.data_quality._md import md_table
    rd = pd.concat([pd.read_csv(f, dtype={"day": str}) for f in sorted(glob.glob(f"{OUT}/days/*.csv.gz"))])
    rd["year"] = rd.day.str[:4]
    rng = np.random.default_rng(0)
    rows = []
    for (r, y), g in rd.groupby(["radar", "year"]):
        sig = int(((g.in10 > 0) | (g.out10 > 0)).sum())
        if sig < 20:
            continue
        A = g[["in10", "in_valid", "out10", "out_valid", "in31", "out31"]].values
        def ratio(a):
            fi, fo = a[:, 0].sum() / a[:, 1].sum(), a[:, 2].sum() / a[:, 3].sum()
            return np.log(fi / fo) if fi > 0 and fo > 0 else np.nan
        bs = np.array([ratio(A[rng.integers(0, len(A), len(A))]) for _ in range(500)])
        r31i, r31o = A[:, 4].sum() / A[:, 1].sum(), A[:, 5].sum() / A[:, 3].sum()
        rows.append({"radar": r, "year": y, "days": len(g), "days_with_10": sig, "R10": float(np.exp(ratio(A))),
                     "lo": float(np.exp(np.nanpercentile(bs, 2.5))), "hi": float(np.exp(np.nanpercentile(bs, 97.5))),
                     "R31": float(r31i / r31o) if r31o > 0 else np.nan})
    t = pd.DataFrame(rows)
    t["flagged"] = (t.R10 >= 2) & (t.lo >= 1.25)
    t = t.sort_values("R10", ascending=False)
    os.makedirs(NOTES, exist_ok=True)
    t.to_csv(f"{NOTES}/radar_band_tail.csv", index=False)
    t[t.flagged][["radar", "year", "R10", "lo", "hi", "R31"]].to_csv(f"{Q4}/biased_radar_years_tail.csv", index=False)
    focus = t[t.radar.isin(["robob", "hunap", "husze", "skkoj", "skjav", "skkub", "sklaz", "rotim", "roora"])]
    L = ["# Tail-focused boundary test (DECISIONS §36, option b)", "",
         f"{len(t)} radar-years with >= 20 days having a pixel >= 10 mm/h in either band; every second day. "
         f"Flagged (R10 >= 2, lower 95% bound >= 1.25): {int(t.flagged.sum())} radar-years, "
         f"{t[t.flagged].radar.nunique()} radars. Median R10 over radar-years {t.R10.median():.2f}.", "",
         "## Flagged", "", md_table(t[t.flagged], index=False, floatfmt=".3g"), "",
         "## The suspects of the tail diagnostic", "", md_table(focus, index=False, floatfmt=".3g")]
    txt = "\n".join(L)
    open(f"{NOTES}/radar_band_tail.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cmd_run(a.workers) if a.cmd == "run" else cmd_analyze()
