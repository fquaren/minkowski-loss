#!/usr/bin/env python
"""Per-pixel outlier masks from the exceedance climatology: a robust, local Mahalanobis test
(after Franch et al. 2020; design and acceptance in DECISIONS §25 and its amendment).

Per year, per pixel with >= MIN_VALID valid frames (`quality_v2/clutter_climatology.npz`):
  features   f_k = logit((n_ge_k + 0.5) / (n_valid + 1)) for k = 1, 31, 89, 150 mm/h
  reference  the NaN-aware median of f_k over an annulus of radius R_IN..R_OUT px around the
             pixel (6-16 km). It separates 1-3 px clutter from 10-50 km orographic structure,
             and the inner hole keeps a clutter blob out of its own reference
  residual   z_k = (f_k - reference) / sqrt(sd_k^2 + se^2): sd_k = 1.4826 x MAD of the residual
             over pixels with >= 5 exceedances (floor 0.1), se = 1 / sqrt((n + 0.5)(1 - p)) the
             sampling error of the logit, so sparse counts (0-2 a year) cannot score high
  distance   Mahalanobis distance d^2 = z' R^-1 z, R the correlation of z from Spearman ranks on
             a sample (2 sin(pi rho_s / 6)); MCD is made singular by the many tied sparse-count
             pixels. Flagged where d^2 > chi2(4) quantile 1 - alpha AND
             some z_k > 0 (excess only: the repair only lowers values, so a deficit such as
             beam blockage is left to other rules)
Writes <out>/pixel_outliers.npz with y<year>_t<thr> masks (d^2 > thr, thr in D2_THRS), the
d^2 maps, and diagnostics: flags per year, overlap with the current hot mask, flag rate by
elevation decile, and years flagged per pixel.

    python scripts/data_quality/pixel_outliers.py --workers 4
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
D = "/home/fquareng/work/data/extremes/OPERA"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
OUT = f"{D}/quality_v4/calib/pixel_outliers"
THRS = (1, 31, 89, 150)
MIN_VALID = 1000
R_IN, R_OUT = 3, 8
EXPS = (3, 4, 5, 6)           # superseded: chi^2(4) does not hold (DECISIONS §25, 2nd amendment)
D2_THRS = (25, 100, 400, 1600)


def annulus_offsets(r_in=R_IN, r_out=R_OUT):
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    rr = np.hypot(yy, xx)
    m = (rr >= r_in) & (rr <= r_out)
    return np.c_[yy[m], xx[m]]


def annulus_median(f, offsets, band=64):
    """NaN-aware median of f over the annulus offsets, processed in row bands."""
    H, W = f.shape
    r = int(np.abs(offsets).max())
    pad = np.pad(f, r, mode="constant", constant_values=np.nan)
    out = np.full(f.shape, np.nan, np.float32)
    import warnings
    for y0 in range(0, H, band):
        y1 = min(y0 + band, H)
        stack = np.stack([pad[y0 + r + dy:y1 + r + dy, r + dx:r + dx + W] for dy, dx in offsets])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[y0:y1] = np.nanmedian(stack, axis=0)
    return out


def year_scores(args):
    year, seed = args
    from scipy.stats import chi2
    z = np.load(CLIM)
    nv = z[f"y{year}_n_valid"].astype(np.float64)
    ok = nv >= MIN_VALID
    if ok.sum() < 10000:
        return year, None
    offs = annulus_offsets()
    Z = []
    for u in THRS:
        n = z[f"y{year}_n_ge{u}"].astype(np.float64)
        p = (n + 0.5) / (nv + 1.0)
        f = np.where(ok, np.log(p / (1 - p)), np.nan).astype(np.float32)
        r = f - annulus_median(f, offs)
        # standardise by spatial spread AND sampling noise: the logit of a frequency from n
        # counts has standard error ~ 1/sqrt((n + 0.5)(1 - p)); with 0-2 exceedances in a year
        # (most pixels at >= 31 mm/h) the residual is noise and must not score high
        se = 1.0 / np.sqrt((n + 0.5) * (1 - p))
        live = ok & (n >= 5)
        mad = np.nanmedian(np.abs(r[live] - np.nanmedian(r[live]))) if live.sum() > 1000 else 0.0
        sd = max(1.4826 * mad, 0.1)
        Z.append((r / np.sqrt(sd ** 2 + se ** 2)).astype(np.float32))
    Z = np.stack(Z, -1)                                           # (H, W, 4)
    good = ok & np.isfinite(Z).all(-1)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(good)
    samp = Z.reshape(-1, 4)[rng.choice(idx, min(200_000, idx.size), replace=False)]
    # correlation of the (already unit-scaled) z from Spearman ranks, converted to Pearson by
    # rho = 2 sin(pi rho_s / 6): robust to outliers and, unlike MCD, not made singular by the
    # many pixels whose sparse-count features are tied
    from scipy.stats import spearmanr
    rs = spearmanr(samp).correlation
    R = 2 * np.sin(np.pi * rs / 6)
    Ri = np.linalg.inv(R)
    zz = Z.reshape(-1, 4)[idx].astype(np.float64)
    d2 = np.full(ok.shape, np.nan, np.float32)
    d2.ravel()[idx] = np.einsum("ij,jk,ik->i", zz, Ri, zz)
    excess = (Z > 0).any(-1)
    masks = {t: good & excess & (d2 > t) for t in D2_THRS}
    return year, (d2, masks, Z.max(-1).astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(CLIM)
    years = sorted({int(k[1:5]) for k in z.files if k.startswith("y") and k.endswith("_n_valid")})
    from src.data.cleaning import hot_mask_from_climatology
    from src.data import geo
    dem = geo.load_dem_on_radar_grid(geo.DEFAULT_DEM_PATH) if hasattr(geo, "DEFAULT_DEM_PATH") else None
    store, rows, nflag = {}, [], {t: None for t in D2_THRS}
    with ProcessPoolExecutor(a.workers) as ex:
        for year, res in ex.map(year_scores, [(y, 1000 + y) for y in years]):
            if res is None:
                print(f"  {year}: too few valid pixels, skipped", flush=True)
                continue
            d2, masks, zmax = res
            store[f"y{year}_d2"] = d2
            hot = hot_mask_from_climatology(dict(z), year)
            for e, m in masks.items():
                store[f"y{year}_t{e}"] = m
                nflag[e] = m.astype(np.int16) if nflag[e] is None else nflag[e] + m
                row = {"year": year, "d2_thr": e, "n_flag": int(m.sum()),
                       "share_of_valid": float(m.sum() / np.isfinite(d2).sum()),
                       "overlap_hot": int((m & hot).sum()) if hot is not None else -1,
                       "n_hot": int(hot.sum()) if hot is not None else -1}
                if dem is not None:
                    v = np.isfinite(d2) & np.isfinite(dem)
                    dec = np.digitize(dem, np.nanquantile(dem[v], np.linspace(0.1, 0.9, 9)))
                    for q in range(10):
                        sel = v & (dec == q)
                        row[f"rate_elev_d{q}"] = float(m[sel].mean()) if sel.any() else np.nan
                rows.append(row)
            print(f"  {year}: " + ", ".join(f"d2>{e}: {int(m.sum())}" for e, m in masks.items()), flush=True)
    for e, n in nflag.items():
        if n is not None:
            store[f"years_flagged_t{e}"] = n
    np.savez_compressed(f"{OUT}/pixel_outliers.npz", **store)
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/diagnostics.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.groupby("d2_thr")[["n_flag", "share_of_valid", "overlap_hot", "n_hot"]].median().round(5).to_string())
    ec = [c for c in t.columns if c.startswith("rate_elev")]
    if ec:
        print("flag rate by elevation decile (median over years):")
        print(t.groupby("d2_thr")[ec].median().map(lambda v: f"{v:.2e}").to_string())
    for e in D2_THRS:
        n = store.get(f"years_flagged_t{e}")
        if n is not None:
            print(f"d2 > {e}: pixels flagged in >= 1 year {int((n >= 1).sum())}, in >= half the years {int((n >= len(years) / 2).sum())}")


if __name__ == "__main__":
    main()
