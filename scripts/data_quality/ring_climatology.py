#!/usr/bin/env python
"""Range rings in the per-year exceedance climatology: where a radar's coverage edge (or a
fixed range) lights up far more often than the ranges just inside and outside it.

Range rings are not thin arcs in single frames (`cleaning.ring_flag` finds none on a
7,005-tile sample); they are sparse pixels on a circle about a radar that only add up over
time, visible as circles in the >= 31 mm/h frequency map (e.g. Stevns and Sindal, DK, at
~115-118 px = 230-235 km in 2013 and 2016). So they are detected climatologically, per year:

  1. f = n_ge31 / n_valid per pixel for the year, with static-clutter hot pixels (the
     cleaning's > 1% rule) set to NaN so clutter does not pose as a ring;
  2. about each radar site in the database (current and archive): mean f per (1-px range bin, 10-degree sector), for
     ranges 16-150 px;
  3. a range r is a ring in a sector if f(r) > RATIO x the median of f over r-6..r-3 and
     r+3..r+6, and f(r) > F_MIN;
  4. it is a ring if that holds in >= MIN_SECTORS sectors (180 degrees by default).
     Shorter arcs are listed (`ring_list.csv`, `ring=False`) for inspection but not masked.

Outputs under --out_dir: ring_mask.npz (bool y<year> arrays, pixels within 1 px of a ring
range in its hit sectors) and ring_list.csv.

    python scripts/data_quality/ring_climatology.py \
        --climatology .../quality_v2/clutter_climatology.npz --out_dir .../quality_v2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

RATIO = 3.0
F_MIN = 5e-5
MIN_SECTORS = 18          # of 36: 180 degrees
LIST_SECTORS = 9          # arcs >= 90 degrees are listed
R_MIN, R_MAX = 16, 150


def main():
    from src.data import geo
    from src.data.cleaning import hot_mask_from_climatology
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--climatology", default="/home/fquareng/work/data/extremes/OPERA/quality_v2/clutter_climatology.npz")
    ap.add_argument("--out_dir", default="/home/fquareng/work/data/extremes/OPERA/quality_v2")
    a = ap.parse_args()
    clim = dict(np.load(a.climatology))
    years = sorted(int(k[1:5]) for k in clim if k.startswith("y") and k.endswith("_n_valid"))
    sites = geo.load_radar_sites()
    yy, xx = np.mgrid[0:geo.H, 0:geo.W]
    masks, rows = {}, []
    for year in years:
        nv = clim[f"y{year}_n_valid"].astype(float)
        f = np.where(nv >= 200, clim[f"y{year}_n_ge31"] / np.maximum(nv, 1), np.nan)
        hot = hot_mask_from_climatology(clim, year)
        if hot is not None:
            f[hot] = np.nan
        # Every site of the current + archive database, whatever its listed years: the
        # database's start/finish years are incomplete (Stevns is listed from 2017, but its
        # 230 km ring is in the 2013 and 2016 climatology), and a ring centred on a site
        # position is evidence of a radar there. Co-located entries are kept once.
        act = sites[sites["row"].between(0, geo.H - 1) & sites["col"].between(0, geo.W - 1)]
        act = act.loc[~act[["row", "col"]].round(0).duplicated()]
        mask = np.zeros((geo.H, geo.W), bool)
        for _, s in act.iterrows():
            r0, c0 = int(round(s.row)), int(round(s.col))
            y0, y1 = max(r0 - R_MAX - 8, 0), min(r0 + R_MAX + 9, geo.H)
            x0, x1 = max(c0 - R_MAX - 8, 0), min(c0 + R_MAX + 9, geo.W)
            fy = f[y0:y1, x0:x1]
            r = np.hypot(yy[y0:y1, x0:x1] - s.row, xx[y0:y1, x0:x1] - s.col)
            sec = ((np.arctan2(yy[y0:y1, x0:x1] - s.row, xx[y0:y1, x0:x1] - s.col) + np.pi)
                   / (2 * np.pi) * 36).astype(int) % 36
            ok = np.isfinite(fy) & (r < R_MAX + 8)
            if ok.sum() < 1000:
                continue
            rb = r[ok].astype(int)
            M = np.full((R_MAX + 8, 36), np.nan)
            g = pd.DataFrame({"r": rb, "s": sec[ok], "f": fy[ok]}).groupby(["r", "s"])["f"].mean()
            M[g.index.get_level_values(0), g.index.get_level_values(1)] = g.values
            for rr in range(R_MIN, R_MAX):
                with np.errstate(all="ignore"):
                    nb = np.nanmedian(np.r_[M[rr - 6:rr - 2], M[rr + 3:rr + 7]], axis=0)
                hit = (M[rr] > RATIO * np.maximum(nb, 1e-6)) & (M[rr] > F_MIN)
                n = int(np.nansum(hit))
                if n < LIST_SECTORS:
                    continue
                ring = n >= MIN_SECTORS
                rows.append({"year": year, "odim": s.odim, "location": s.location,
                             "range_px": rr, "range_km": 2 * rr, "sectors": n,
                             "f_ring": float(np.nanmedian(M[rr][hit])),
                             "f_around": float(np.nanmedian(nb[hit])), "ring": ring})
                if ring:
                    hs = np.nonzero(hit)[0]
                    mask[y0:y1, x0:x1] |= (np.abs(r - rr) <= 1) & np.isin(sec, hs)
        masks[f"y{year}"] = mask
        print(f"{year}: {int(mask.sum())} ring pixels, "
              f"{sum(1 for x in rows if x['year'] == year and x['ring'])} rings", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, "ring_mask.npz"), **masks)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out_dir, "ring_list.csv"), index=False)
    if len(t):
        print(t[t.ring].groupby(["odim", "location"]).agg(
            years=("year", lambda y: ",".join(map(str, sorted(set(y))))),
            range_km=("range_km", "median"), sectors=("sectors", "max")).to_string())


if __name__ == "__main__":
    main()
