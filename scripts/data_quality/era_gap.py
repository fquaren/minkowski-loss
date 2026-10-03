#!/usr/bin/env python
"""How different are the ODYSSEY and NIMBUS products? (OPERA switched on 2024-07-05.)

Both are on the same 2 km / 15-min grid; they differ in how a frame is made (ODYSSEY:
quality-weighted composite of the scans in a 15-min window; NIMBUS: lowest-elevation PPI at
the nominal time) and in the radar set. Two complementary measurements:

paired   the 101 days 2024-07-05..2024-10-30 exist in BOTH products: the original stores
         (ODYSSEY-type, capped at 150 mm/h upstream; kept in raw/OPERA_orig_postswitch) and the
         archive (NIMBUS, raw/OPERA). Same weather, same days, so every difference is the
         product. Compared on the common valid domain, per sampled frame:
           wet fraction, exceedance frequency at 1/10/31/89 mm/h (not above: the ODYSSEY
           copies are capped), wet-pixel intensity quantiles, log-correlation and the
           median ratio of co-located pixels, Minkowski curves (area, perimeter, Euler
           characteristic) and radially averaged power spectra on fully covered stride-128
           tiles, and frame-to-frame correlation at +15 min (the time-support difference).
months   era-level, from the tile scan tables: per calendar month, wet fraction and the
         frequency of tile maxima >= 31 / >= 89 mm/h. Each NIMBUS month is placed against the
         interannual spread of the same calendar month in ODYSSEY (z-score), so a product shift
         is separated from an unusual season.

Outputs under --out_dir: paired_frames.csv, paired_summary.md, months.csv, months_summary.md,
and figures.

    python scripts/data_quality/era_gap.py --part all \
        --tiles_dir .../quality_v2/tiles --out_dir .../quality_v2/era_gap
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

RAW = "/home/fquareng/work/data/extremes/OPERA/raw"
NIMBUS_START = "20240705"
LEVELS = (0.1, 1.0, 10.0, 31.0, 89.0)
MINK_U = (0.1, 0.5, 1.0, 3.0, 10.0, 31.0, 89.0)


def minkowski(t, levels=MINK_U):
    """Area fraction, perimeter (px edges) and Euler characteristic (4-conn) per level."""
    from skimage.measure import euler_number
    out = []
    for u in levels:
        b = t >= u
        per = np.count_nonzero(b[1:, :] != b[:-1, :]) + np.count_nonzero(b[:, 1:] != b[:, :-1])
        out.append((b.mean(), per, euler_number(b, connectivity=1) if b.any() else 0))
    return np.array(out, dtype=float)                     # (n_levels, 3)


def rapsd(t):
    f = np.abs(np.fft.fftshift(np.fft.fft2(t))) ** 2
    n = t.shape[0]; yy, xx = np.indices(t.shape); r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    return np.bincount(r.ravel(), f.ravel())[: n // 2] / np.maximum(np.bincount(r.ravel())[: n // 2], 1)


def paired_day(day, every=4, patch=128):
    import xarray as xr
    a = xr.open_zarr(f"{RAW}/OPERA_orig_postswitch/{day}", consolidated=False)
    b = xr.open_zarr(f"{RAW}/OPERA/{day}", consolidated=True)
    sh = np.intersect1d(a.time.values, b.time.values)
    rows, mk, sp = [], [], []
    for k in range(0, len(sh), every):
        t = sh[k]
        x = a.TOT_PREC.sel(time=t).values.astype(np.float64)
        y = b.TOT_PREC.sel(time=t).values.astype(np.float64)
        dom = np.isfinite(x) & np.isfinite(y)
        if dom.sum() < 1000:
            continue
        xs, ys = np.where(dom, x, 0.0), np.where(dom, y, 0.0)
        r = {"day": day, "time": str(t)[:16]}
        for u in LEVELS:
            r[f"odyssey_f{u:g}"] = float((xs[dom] >= u).mean())
            r[f"nimbus_f{u:g}"] = float((ys[dom] >= u).mean())
        wx, wy = xs[dom & (xs >= 0.1)], ys[dom & (ys >= 0.1)]
        for q in (50, 90, 99):
            r[f"odyssey_q{q}"] = float(np.percentile(wx, q)) if wx.size else np.nan
            r[f"nimbus_q{q}"] = float(np.percentile(wy, q)) if wy.size else np.nan
        both = dom & (xs >= 0.1) & (ys >= 0.1) & (xs <= 149)
        if both.sum() > 100:
            r["logcorr"] = float(np.corrcoef(np.log(xs[both]), np.log(ys[both]))[0, 1])
            r["median_ratio"] = float(np.median(ys[both] / xs[both]))
        # +15 min persistence in each product
        if k + 1 < len(sh):
            for name, arr in (("odyssey", a), ("nimbus", b)):
                p0 = np.where(dom, arr.TOT_PREC.sel(time=t).values, 0.0)
                p1 = arr.TOT_PREC.sel(time=sh[k + 1]).values
                m = dom & np.isfinite(p1) & ((p0 >= 0.1) | (np.nan_to_num(p1) >= 0.1))
                if m.sum() > 100:
                    r[f"{name}_persist15"] = float(np.corrcoef(p0[m], np.nan_to_num(p1)[m])[0, 1])
        rows.append(r)
        H, W = xs.shape
        for r0 in range(0, H - patch + 1, patch):
            for c0 in range(0, W - patch + 1, patch):
                if not dom[r0:r0 + patch, c0:c0 + patch].all():
                    continue
                tx, ty = xs[r0:r0 + patch, c0:c0 + patch], ys[r0:r0 + patch, c0:c0 + patch]
                if max(tx.max(), ty.max()) < 1.0:
                    continue
                mk.append(np.stack([minkowski(tx), minkowski(ty)]))
                sp.append(np.stack([rapsd(tx), rapsd(ty)]))
    return rows, mk, sp


def part_paired(out_dir, workers):
    from concurrent.futures import ProcessPoolExecutor
    days = sorted(os.listdir(f"{RAW}/OPERA_orig_postswitch"))
    rows, mk, sp = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, m, s in ex.map(paired_day, days):
            rows += r; mk += m; sp += s
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out_dir, "paired_frames.csv"), index=False)
    mk, sp = np.array(mk), np.array(sp)              # (n, 2, L, 3), (n, 2, K)
    np.savez_compressed(os.path.join(out_dir, "paired_tiles.npz"), minkowski=mk, rapsd=sp,
                        levels=np.array(MINK_U))
    L = ["# ODYSSEY vs NIMBUS on the same days (paired)", "",
         f"{df['day'].nunique()} days, {len(df)} frames, {len(mk)} wet fully-covered tiles. "
         "ODYSSEY copies are capped at 150 mm/h upstream, so nothing above 89 mm/h is compared.", "",
         "| quantity | ODYSSEY | NIMBUS | NIMBUS / ODYSSEY |", "|---|---|---|---|"]
    for u in LEVELS:
        o, n = df[f"odyssey_f{u:g}"].mean(), df[f"nimbus_f{u:g}"].mean()
        L.append(f"| frequency >= {u:g} mm/h | {o:.5f} | {n:.5f} | {n / o:.3f} |")
    for q in (50, 90, 99):
        o, n = df[f"odyssey_q{q}"].median(), df[f"nimbus_q{q}"].median()
        L.append(f"| wet-pixel p{q} (mm/h) | {o:.2f} | {n:.2f} | {n / o:.3f} |")
    for name in ("persist15",):
        L.append(f"| frame-to-frame corr (+15 min) | {df['odyssey_' + name].median():.3f} | "
                 f"{df['nimbus_' + name].median():.3f} | |")
    L += ["", f"co-located wet pixels: log-correlation median {df['logcorr'].median():.3f}, "
          f"median ratio NIMBUS/ODYSSEY {df['median_ratio'].median():.3f}", "",
          "Minkowski curves, tile mean (NIMBUS / ODYSSEY):", "",
          "| u (mm/h) | area | perimeter | Euler char. (mean ODY / NIM) |", "|---|---|---|---|"]
    if len(mk):
        mo, mn = mk[:, 0].mean(0), mk[:, 1].mean(0)
        for i, u in enumerate(MINK_U):
            L.append(f"| {u:g} | {mn[i, 0] / max(mo[i, 0], 1e-12):.3f} | "
                     f"{mn[i, 1] / max(mo[i, 1], 1e-12):.3f} | {mo[i, 2]:.2f} / {mn[i, 2]:.2f} |")
        so, sn = sp[:, 0].mean(0), sp[:, 1].mean(0)
        k = np.array([2, 4, 8, 16, 32, 63])
        L += ["", "Power spectrum ratio NIMBUS/ODYSSEY by wavelength: " + ", ".join(
            f"{256 / kk:.0f} km {sn[kk] / so[kk]:.2f}" for kk in k)]
    open(os.path.join(out_dir, "paired_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def part_months(tiles_dir, out_dir):
    parts = []
    for f in sorted(glob.glob(os.path.join(tiles_dir, "*.csv.gz"))):
        d = pd.read_csv(f, usecols=["max", "wet_frac", "unphysical", "ray"])
        d = d[(d.unphysical == 0) & (d.ray == 0)]
        day = os.path.basename(f)[:8]
        parts.append({"day": day, "wet": d.wet_frac.mean(), "f31": (d["max"] >= 31).mean(),
                      "f89": (d["max"] >= 89).mean(), "n": len(d)})
    t = pd.DataFrame(parts)
    t["year"], t["month"] = t.day.str[:4].astype(int), t.day.str[4:6].astype(int)
    t["era"] = np.where(t.day >= NIMBUS_START, "NIMBUS", "ODYSSEY")
    ym = t.groupby(["era", "year", "month"])[["wet", "f31", "f89"]].mean().reset_index()
    ym.to_csv(os.path.join(out_dir, "months.csv"), index=False)
    ody = ym[ym.era == "ODYSSEY"]; nim = ym[ym.era == "NIMBUS"]
    L = ["# Month-matched era comparison (tile scan tables)", "",
         "NIMBUS month value vs the ODYSSEY interannual distribution of the same calendar month "
         "(z = (NIMBUS - mean) / sd over ODYSSEY years).", "",
         "| year-month | wet frac z | P(max>=31) z | P(max>=89) z |", "|---|---|---|---|"]
    for _, r in nim.sort_values(["year", "month"]).iterrows():
        ref = ody[ody.month == r.month]
        z = [(r[c] - ref[c].mean()) / ref[c].std() if len(ref) > 2 and ref[c].std() > 0 else np.nan
             for c in ("wet", "f31", "f89")]
        L.append(f"| {r.year}-{r.month:02d} | {z[0]:+.2f} | {z[1]:+.2f} | {z[2]:+.2f} |")
    open(os.path.join(out_dir, "months_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", default="all", choices=["paired", "months", "all"])
    ap.add_argument("--tiles_dir", default="/home/fquareng/work/data/extremes/OPERA/quality_v2/tiles")
    ap.add_argument("--out_dir", default="/home/fquareng/work/data/extremes/OPERA/quality_v2/era_gap")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.part in ("paired", "all"):
        part_paired(a.out_dir, a.workers)
    if a.part in ("months", "all"):
        part_months(a.tiles_dir, a.out_dir)


if __name__ == "__main__":
    main()
