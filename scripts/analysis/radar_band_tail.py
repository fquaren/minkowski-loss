#!/usr/bin/env python
"""Tail-focused boundary test for radars biased against their neighbours (DECISIONS §36, option b).

Per radar and day (every second day), summed over the day's frames of the clean_frame field
(contributing geometry): in the 6 km band just inside the radar's owned area and the 6 km band just
outside it, valid pixels and pixels >= 10 / >= 31 mm/h.
Per radar-year: R10 = inside / outside frequency of >= 10 mm/h, 95% day-block interval of log R10;
flagged if R10 >= 2, lower end >= 1.25 and >= 20 days with a pixel >= 10 mm/h in either band.

    python scripts/analysis/radar_band_tail.py run --workers 2      (-> quality_v4/band_tail/days/)
    python scripts/analysis/radar_band_tail.py analyze              (-> biased_radar_years_tail.csv)

National-border extension (§36c): the same counts in the 6 km bands on either side of every seam
between two countries' owned areas (country = ODIM prefix of the owning radar); a country-year is
flagged if >= 2 of its qualifying border sides have R10 >= 2 with lower bound >= 1.25.

    python scripts/analysis/radar_band_tail.py run_national --workers 3   (-> quality_v4/band_tail/national/)
    python scripts/analysis/radar_band_tail.py analyze_national           (-> biased_radar_years_national.csv)
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


def _national(year):
    """Ordered border sides (A, B): flat indices of A's pixels within BAND_PX of a B pixel, and labels."""
    from scipy import ndimage
    from src.data.radar_screen import BAND_PX
    key = ("nat", year)
    if key in _G:
        return _G[key]
    g = _geometry(year)
    ctry = np.array([k[:2] for k in g.keys])
    names = sorted(set(ctry))
    cmap = np.where(g.owner >= 0, np.searchsorted(names, ctry[np.maximum(g.owner, 0)]), -1)
    st = np.ones((3, 3), bool)
    sides, idx, lab = [], [], []
    for b, B in enumerate(names):
        near = ndimage.binary_dilation(cmap == b, st, iterations=BAND_PX)
        for a in np.unique(cmap[near & (cmap >= 0) & (cmap != b)]):
            band = near & (cmap == a)                     # A's pixels within BAND_PX of B
            idx.append(np.flatnonzero(band)); lab.append(np.full(int(band.sum()), len(sides)))
            sides.append((names[a], B))
    _G[key] = (sides, np.concatenate(idx), np.concatenate(lab))
    return _G[key]


def run_day_national(day):
    from src.data.cleaning import clean_frame
    from src.data.day_cleaner import open_day
    year = int(day[:4])
    sides, idx, lab = _national(year)
    hot = _hot(year)
    S = len(sides)
    acc = {k: np.zeros(S) for k in ("valid", "n10", "n31")}
    ds = open_day(f"{D}/raw/OPERA/{day}")
    for t in range(ds.sizes["time"]):
        v = clean_frame(ds["TOT_PREC"].isel(time=t).values.astype(np.float32), hot)[0].ravel()[idx]
        ok = np.isfinite(v)
        acc["valid"] += np.bincount(lab[ok], minlength=S)
        acc["n10"] += np.bincount(lab[ok], weights=(v[ok] >= 10).astype(float), minlength=S)
        acc["n31"] += np.bincount(lab[ok], weights=(v[ok] >= 31).astype(float), minlength=S)
    ds.close()
    df = pd.DataFrame({"side": [a for a, _ in sides], "other": [b for _, b in sides], "day": day,
                       "px": np.bincount(lab, minlength=S), **acc})
    os.makedirs(f"{OUT}/national", exist_ok=True)
    df.to_csv(f"{OUT}/national/{day}.csv.gz", index=False)
    return day, S


def cmd_run_national(workers):
    days = sorted(os.path.basename(p) for p in glob.glob(f"{D}/raw/OPERA/20*") if os.path.isdir(p))[::2]
    todo = [d for d in days if not os.path.exists(f"{OUT}/national/{d}.csv.gz")]
    print(f"[band_tail national] {len(days)} days (every second), {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, (d, n) in enumerate(ex.map(run_day_national, todo, chunksize=2)):
            if k % 50 == 0:
                print(f"  {k + 1}/{len(todo)} days ({d}, {n} border sides)", flush=True)


def cmd_analyze_national():
    from scripts.data_quality._md import md_table
    from src.data.geo import load_dem_on_radar_grid
    rd = pd.concat([pd.read_csv(f, dtype={"day": str}, keep_default_na=False)
                    for f in sorted(glob.glob(f"{OUT}/national/*.csv.gz"))])
    rd["year"] = rd.day.str[:4]
    rng = np.random.default_rng(0)
    rows = []
    for (a, b, y), g in rd.groupby(["side", "other", "year"]):
        o = rd[(rd.side == b) & (rd.other == a) & (rd.year == y)].set_index("day")
        g = g.set_index("day").join(o[["valid", "n10", "n31", "px"]], rsuffix="_o", how="inner")
        g = g[(g.valid > 0) & (g.valid_o > 0)]
        if g.empty:
            continue
        sig = int(((g.n10 > 0) | (g.n10_o > 0)).sum())
        A = g[["n10", "valid", "n10_o", "valid_o", "n31", "n31_o"]].values

        def ratio(m):
            fi, fo = m[:, 0].sum() / m[:, 1].sum(), m[:, 2].sum() / m[:, 3].sum()
            return np.log(fi / fo) if fi > 0 and fo > 0 else np.nan
        bs = np.array([ratio(A[rng.integers(0, len(A), len(A))]) for _ in range(500)])
        fin = np.isfinite(bs).any()
        r31i, r31o = A[:, 4].sum() / A[:, 1].sum(), A[:, 5].sum() / A[:, 3].sum()
        rows.append({"country": a, "against": b, "year": y, "days": len(g), "days_with_10": sig,
                     "px": int(g.px.median()), "px_other": int(g.px_o.median()), "R10": float(np.exp(ratio(A))),
                     "lo": float(np.exp(np.nanpercentile(bs, 2.5))) if fin else np.nan,
                     "hi": float(np.exp(np.nanpercentile(bs, 97.5))) if fin else np.nan,
                     "R31": float(r31i / r31o) if r31o > 0 else np.nan})
    t = pd.DataFrame(rows)
    t["qualifies"] = (t.px >= 150) & (t.px_other >= 150) & (t.days_with_10 >= 20)
    t["flagged"] = t.qualifies & (t.R10 >= 2) & (t.lo >= 1.25)
    cfg = yaml.safe_load(open(f"{REPO}/configs/config_v4_adv.yaml"))
    dem = load_dem_on_radar_grid(cfg["STATIC_DEM_PATH"]).ravel()   # mean DEM height of both bands, flagged sides
    t["dem_side"], t["dem_other"] = np.nan, np.nan
    for i in np.nonzero(t.flagged.values)[0]:
        sides, idx, lab = _national(int(t.year.iat[i]))
        for col, key in (("dem_side", (t.country.iat[i], t.against.iat[i])),
                         ("dem_other", (t.against.iat[i], t.country.iat[i]))):
            if key in sides:
                t.loc[t.index[i], col] = float(np.nanmean(dem[idx[lab == sides.index(key)]]))
    cy = (t[t.qualifies].groupby(["country", "year"])
          .agg(borders=("flagged", "size"), flagged_borders=("flagged", "sum"), median_R10=("R10", "median"))
          .reset_index())
    cy["flagged"] = cy.flagged_borders >= 2
    t = t.sort_values("R10", ascending=False)
    t.to_csv(f"{NOTES}/radar_band_national.csv", index=False)
    cy.to_csv(f"{NOTES}/radar_band_national_countries.csv", index=False)
    out = [(k, r.year) for _, r in cy[cy.flagged].iterrows() for k in _geometry(int(r.year)).keys
           if k[:2] == r.country]
    pd.DataFrame(out, columns=["radar", "year"]).to_csv(f"{Q4}/biased_radar_years_national.csv", index=False)
    q = t[t.qualifies]
    L = ["# National-border tail test (DECISIONS §36c)", "",
         f"{len(q)} qualifying border sides x years (both bands >= 150 px, >= 20 days with a pixel >= 10 mm/h); "
         f"median R10 {q.R10.median():.2f}. Flagged border sides (R10 >= 2, lower bound >= 1.25): "
         f"{int(t.flagged.sum())}. Flagged country-years (>= 2 flagged borders): {int(cy.flagged.sum())} -> "
         f"{len(out)} radar-years in `biased_radar_years_national.csv`.", "",
         "## Flagged country-years", "", md_table(cy[cy.flagged], index=False, floatfmt=".3g"), "",
         "## Flagged border sides", "",
         md_table(t[t.flagged].drop(columns=["qualifies", "flagged"]), index=False, floatfmt=".3g"), "",
         "## Hungary, Slovakia, Romania: every qualifying border side", "",
         md_table(q[q.country.isin(["hu", "sk", "ro"])].sort_values(["country", "against", "year"])
                  .drop(columns=["qualifies"]), index=False, floatfmt=".3g"), "",
         "## Country-years by median R10 over qualifying borders (top 25)", "",
         md_table(cy.sort_values("median_R10", ascending=False).head(25), index=False, floatfmt=".3g")]
    txt = "\n".join(L)
    open(f"{NOTES}/radar_band_national.md", "w").write(txt + "\n")
    print(txt)


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
    ap.add_argument("cmd", choices=["run", "analyze", "run_national", "analyze_national"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"run": lambda: cmd_run(a.workers), "analyze": cmd_analyze,
     "run_national": lambda: cmd_run_national(a.workers), "analyze_national": cmd_analyze_national}[a.cmd]()
