#!/usr/bin/env python
"""Which POT threshold u does the v3 data support? (EXPERIMENTS §5, "Re-derive the POT threshold".)

u = 31 mm/h was inherited from the loss grid and never tested. This script measures, on the
v3 store (cleaned, repaired, ODYSSEY train/val/test plus the NIMBUS split):

  mrl       mean residual life e(u) = E[X - u | X > u]. Above a valid threshold it is linear in u
            with slope xi / (1 - xi).
  stability GPD fits over a grid of u: xi and the modified scale sigma* = sigma - xi u. Both are
            constant in u above a valid threshold (Coles 2001, §4.3).
  declust   the same fits on space-time cluster maxima (one value per tile and storm run,
            runs split at gaps > 1 h). Same xi in theory if the pixel-level fit is in its
            asymptotic regime; they differ when dependence or sub-asymptotic shape dominates.
  subsets   per latitude band and per season.

Uncertainty is a day-block bootstrap: pixels on the same day share storms, so the
independent-pixel standard error would be far too small.

Weights. The splits are stratified by tile max (dry < 0.1 <= light < 1 <= moderate < 10 <=
heavy < 31 <= tail), each row carries its inclusion weight N_stratum / n_stratum. Above
31 mm/h only tail tiles contribute, so in train the weighted and unweighted fits coincide;
below 31 heavy and tail tiles mix with different weights and only the weighted fit estimates
the population. In test, event tiles have weight 1 and the rest of their stratum ~2.7, so the
unweighted fit -- which is what eval_extremes.py computes as gpd_xi_obs -- over-represents
events. Both are reported.

    python scripts/data_quality/pot_threshold.py --stage extract   # ~10 min, 8 workers
    python scripts/data_quality/pot_threshold.py --stage analyze   # fits, figures, report
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_quality._md import md_table  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
STORE = f"{D}/patches_v3/preprocessed_dataset.zarr"
META = f"{D}/v3"
OUT = f"{D}/quality_v3/pot_threshold"
SPLITS = {"train": "full_train", "validation": "full_val", "test": "full_test",
          "nimbus": "full_nimbus"}
U_MIN = 10.0                      # values stored above this; 0.01 mm/h grid -> uint16 * 100
U_GRID = [10, 11, 13, 15, 17, 19, 22, 25, 28, 31, 35, 40, 45, 53, 60, 70, 80, 89, 100, 115,
          130, 150]
LAT_BANDS = [("south <46N", -90, 46), ("46-50N", 46, 50), ("50-55N", 50, 55), (">=55N", 55, 90)]
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


# --------------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------------
def _read_block(args):
    split, rows = args
    import zarr
    z = zarr.open(f"{STORE}/{split}/original_precip", mode="r")
    vals, pidx = [], []
    for r in rows:
        a = z[r]
        v = a[a > U_MIN]
        if v.size:
            vals.append(np.rint(v * 100).astype(np.uint16))
            pidx.append(np.full(v.size, r, dtype=np.int32))
    if not vals:
        return np.empty(0, np.uint16), np.empty(0, np.int32)
    return np.concatenate(vals), np.concatenate(pidx)


def extract(workers):
    os.makedirs(OUT, exist_ok=True)
    for split, stem in SPLITS.items():
        out = f"{OUT}/exceed_{split}.npz"
        if os.path.exists(out):
            print(f"[extract] {split}: exists, skipping"); continue
        info = pd.read_csv(f"{META}/{stem}_info.csv.gz")
        assert (info.store_row.values == np.arange(len(info))).all()
        rows = np.where(info["max"].values > U_MIN)[0]
        blocks = [(split, b) for b in np.array_split(rows, max(1, len(rows) // 2000))]
        t = time.time()
        with Pool(workers) as p:
            res = p.map(_read_block, blocks, chunksize=1)
        vals = np.concatenate([r[0] for r in res]); pidx = np.concatenate([r[1] for r in res])
        # the stored tile max must equal the metadata max (to the 0.01 grid)
        mx = np.zeros(len(info)); np.maximum.at(mx, pidx, vals / 100.0)
        bad = np.abs(mx[rows] - info["max"].values[rows]) > 0.011
        print(f"[extract] {split}: {len(rows):,} tiles > {U_MIN}, {vals.size:,} pixels, "
              f"max mismatches {bad.sum()}, {time.time() - t:.0f} s")
        np.savez(out, vals=vals, pidx=pidx, n_tiles=len(info))


# --------------------------------------------------------------------------------------------
# GPD on binned data (values live on a 0.01 mm/h grid, so binning is exact)
# --------------------------------------------------------------------------------------------
def gpd_fit(x, w):
    """Weighted MLE of GPD(xi, sigma) for exceedances x > 0 with weights w. Returns (xi, sigma)."""
    from scipy.optimize import minimize
    W = w.sum()
    if W <= 0 or x.size < 3:
        return np.nan, np.nan
    m = (w * x).sum() / W
    s2 = (w * (x - m) ** 2).sum() / W
    xi0 = float(np.clip(0.5 * (1 - m * m / s2), -0.4, 0.8))
    sg0 = 0.5 * m * (m * m / s2 + 1)
    xmax = x.max()

    def nll(p):
        ls, xi = p
        sg = np.exp(ls)
        if xi < 0 and xmax >= -sg / xi:
            return 1e300
        if abs(xi) < 1e-7:
            return W * ls + (w * x).sum() / sg
        return W * ls + (1 + 1 / xi) * (w * np.log1p(xi * x / sg)).sum()

    r = minimize(nll, [np.log(sg0), xi0], method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-9 * W, "maxiter": 2000})
    return float(r.x[1]), float(np.exp(r.x[0]))


def fit_grid(xv, counts, grid=U_GRID, min_n=50):
    """xv: bin values (mm/h), counts: weights per bin (1-D). One fit per u in grid."""
    out = []
    for u in grid:
        k = (xv > u) & (counts > 0)
        n = counts[k].sum()
        if n < min_n:
            out.append((np.nan, np.nan, np.nan)); continue
        xi, sg = gpd_fit(xv[k] - u, counts[k])
        out.append((xi, sg, sg - xi * u))
    return np.array(out)                       # [len(grid), 3]: xi, sigma, sigma*


def mean_excess(xv, counts, us):
    out = np.full(len(us), np.nan)
    cs = np.cumsum((counts * xv)[::-1])[::-1]; cn = np.cumsum(counts[::-1])[::-1]
    for i, u in enumerate(us):
        j = np.searchsorted(xv, u, side="right")
        if j < len(xv) and cn[j] > 0:
            out[i] = cs[j] / cn[j] - u
    return out


# --------------------------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------------------------
class Sample:
    """Exceedances of one subset as a sparse day x bin matrix, for day-block bootstrapping."""

    def __init__(self, name, vals_mmph, day_codes, weights, n_days):
        from scipy.sparse import coo_matrix
        b = np.rint(vals_mmph * 100).astype(np.int64)
        self.b0 = b.min() if b.size else 0
        ub, inv = np.unique(b, return_inverse=True)
        self.xv = ub / 100.0
        self.name, self.n_days = name, n_days
        self.M = coo_matrix((weights, (day_codes, inv)), shape=(n_days, ub.size)).tocsr()
        self.n_raw = vals_mmph.size

    def counts(self, mult=None):
        if mult is None:
            return np.asarray(self.M.sum(0)).ravel()
        return np.asarray(self.M.T @ mult).ravel()


def _boot_job(args):
    sample, seeds, grid, us_mrl = args
    res, mrl = [], []
    for s in seeds:
        mult = np.random.default_rng(s).multinomial(sample.n_days, np.full(sample.n_days,
                                                                          1 / sample.n_days))
        c = sample.counts(mult.astype(float))
        res.append(fit_grid(sample.xv, c, grid))
        mrl.append(mean_excess(sample.xv, c, us_mrl) if us_mrl is not None else None)
    return res, mrl


def bootstrap(sample, B, workers, grid=U_GRID, us_mrl=None, seed=0):
    seeds = np.random.default_rng(seed).integers(0, 2**31, B)
    jobs = [(sample, ch, grid, us_mrl) for ch in np.array_split(seeds, workers) if ch.size]
    with Pool(workers) as p:
        parts = p.map(_boot_job, jobs)
    fits = np.stack([f for r, _ in parts for f in r])            # [B, U, 3]
    mrl = np.stack([m for _, ms in parts for m in ms]) if us_mrl is not None else None
    return fits, mrl


def load_split(split):
    info = pd.read_csv(f"{META}/{SPLITS[split]}_info.csv.gz")
    z = np.load(f"{OUT}/exceed_{split}.npz")
    vals, pidx = z["vals"] / 100.0, z["pidx"]
    info["day"] = info.timestamp.astype(str).str[:8]
    info["month"] = info.timestamp.astype(str).str[4:6].astype(int)
    return info, vals, pidx


def tile_lat(info):
    from src.data.geo import tile_location
    pos = info[["row", "col"]].drop_duplicates()
    lat = {(r, c): tile_location(int(r), int(c))["lat"] for r, c in pos.itertuples(index=False)}
    return np.array([lat[(r, c)] for r, c in zip(info.row.values, info.col.values)])


def pixel_sample(name, info, vals, pidx, weighted, tile_mask=None):
    day_codes_t, days = pd.factorize(info.day)
    keep = np.ones(vals.size, bool) if tile_mask is None else tile_mask[pidx]
    w = info.weight.values[pidx[keep]] if weighted else np.ones(keep.sum())
    return Sample(name, vals[keep], day_codes_t[pidx[keep]], w, len(days))


def cluster_maxima(info, gap_min=60):
    """One value per tile position and storm run: tiles at the same (row, col) whose
    timestamps are <= gap_min apart belong to one run. Only tail tiles (max >= 31), so the
    sampling is uniform within the stratum and no weights are needed (u >= 31 only)."""
    t = info[info["max"] >= 31].copy()
    t["ts"] = pd.to_datetime(t.timestamp.astype(str), format="%Y%m%d%H%M%S")
    t = t.sort_values(["row", "col", "ts"])
    new = (t.row.diff() != 0) | (t.col.diff() != 0) | (t.ts.diff() > pd.Timedelta(minutes=gap_min))
    t["cl"] = new.cumsum()
    g = t.groupby("cl").agg(mx=("max", "max"), day=("day", "first"))
    return g


def analyze(workers, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(f"{OUT}/fig", exist_ok=True)
    us_mrl = np.arange(10, 301, 2.0)
    rows, mrl_rows, boots = [], {}, {}

    def run(sample, label, kind, b=B, grid=U_GRID, with_mrl=False):
        c = sample.counts()
        point = fit_grid(sample.xv, c, grid)
        bf, bm = bootstrap(sample, b, workers, grid, us_mrl if with_mrl else None)
        lo, hi = np.nanpercentile(bf, 2.5, 0), np.nanpercentile(bf, 97.5, 0)
        for i, u in enumerate(grid):
            k = sample.xv > u
            nd = (np.asarray((sample.M[:, k] > 0).sum(1)).ravel() > 0).sum()
            rows.append(dict(subset=label, kind=kind, u=u, n_exc=int(round(c[k].sum())),
                             days=int(nd), xi=point[i, 0], xi_lo=lo[i, 0], xi_hi=hi[i, 0],
                             sigma=point[i, 1], sstar=point[i, 2], sstar_lo=lo[i, 2],
                             sstar_hi=hi[i, 2]))
        if with_mrl:
            mrl_rows[label] = (mean_excess(sample.xv, c, us_mrl),
                               np.nanpercentile(bm, 2.5, 0), np.nanpercentile(bm, 97.5, 0))
        boots[label] = bf
        print(f"[analyze] {label} ({kind}): done", flush=True)

    data = {s: load_split(s) for s in SPLITS}

    # --- the eval-relevant and population fits ------------------------------------------
    info, vals, pidx = data["train"]
    run(pixel_sample("train", info, vals, pidx, True), "train (weighted)", "pixel", with_mrl=True)
    for s in ("validation", "test", "nimbus"):
        info, vals, pidx = data[s]
        run(pixel_sample(s, info, vals, pidx, True), f"{s} (weighted)", "pixel", with_mrl=True)
    info, vals, pidx = data["test"]
    run(pixel_sample("test", info, vals, pidx, False), "test (unweighted = eval)", "pixel")
    ev = info.event.notna().values
    run(pixel_sample("test", info, vals, pidx, False, tile_mask=~ev),
        "test, no event tiles", "pixel")

    # --- declustered: cluster maxima, u >= 31 -------------------------------------------
    g31 = [u for u in U_GRID if u >= 31]
    for s in ("train", "test"):
        info = data[s][0]
        g = cluster_maxima(info if s == "train" else info[info.event.isna()])
        dc, nd = pd.factorize(g.day)
        run(Sample(s, g.mx.values, dc, np.ones(len(g)), len(nd)),
            f"{s} cluster maxima" + ("" if s == "train" else ", no events"), "cluster",
            grid=g31)

    # --- subsets of train: latitude band and season -------------------------------------
    info, vals, pidx = data["train"]
    lat = tile_lat(info)
    for name, lo_, hi_ in LAT_BANDS:
        m = (lat >= lo_) & (lat < hi_)
        run(pixel_sample("train", info, vals, pidx, True, tile_mask=m), f"train {name}",
            "region", b=max(50, B // 2))
    for name, months in SEASONS.items():
        m = info.month.isin(months).values
        run(pixel_sample("train", info, vals, pidx, True, tile_mask=m), f"train {name}",
            "season", b=max(50, B // 2))

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/fits.csv", index=False)
    pd.DataFrame({"u": us_mrl, **{f"{k}|{q}": v[i] for k, v in mrl_rows.items()
                                  for i, q in enumerate(("e", "lo", "hi"))}}).to_csv(
        f"{OUT}/mrl.csv", index=False)

    # --- figures ------------------------------------------------------------------------
    def stab_plot(labels, fname, title):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for lab in labels:
            d = res[res.subset == lab]
            for a, (c, lo_, hi_) in zip(ax, (("xi", "xi_lo", "xi_hi"),
                                            ("sstar", "sstar_lo", "sstar_hi"))):
                l, = a.plot(d.u, d[c], "o-", ms=3, label=lab)
                a.fill_between(d.u, d[lo_], d[hi_], color=l.get_color(), alpha=0.15)
        for a, yl in zip(ax, ("shape xi", "modified scale sigma* = sigma - xi u (mm/h)")):
            a.set_xscale("log"); a.set_xlabel("threshold u (mm/h)"); a.set_ylabel(yl)
            for u in (31, 53, 89):
                a.axvline(u, color="0.7", lw=0.8, ls=":")
            a.grid(alpha=0.3)
        ax[0].axhline(0, color="k", lw=0.6)
        ax[0].legend(fontsize=7)
        fig.suptitle(title + "  (95% day-block bootstrap bands)")
        fig.savefig(f"{OUT}/fig/{fname}", dpi=130); plt.close(fig)

    stab_plot(["train (weighted)", "validation (weighted)", "test (weighted)"],
              "stability_splits.png", "GPD threshold stability, ODYSSEY splits")
    stab_plot(["test (weighted)", "test (unweighted = eval)", "test, no event tiles"],
              "stability_test_weighting.png", "test: weighting and event tiles")
    stab_plot(["train (weighted)", "train cluster maxima", "test cluster maxima, no events",
               "nimbus (weighted)"], "stability_declustered.png",
              "pixel-level vs cluster maxima; NIMBUS")
    stab_plot([f"train {n}" for n, _, _ in LAT_BANDS], "stability_regions.png",
              "train by latitude band")
    stab_plot([f"train {n}" for n in SEASONS], "stability_seasons.png", "train by season")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for lab, (e, lo_, hi_) in mrl_rows.items():
        l, = ax.plot(us_mrl, e, lw=1.2, label=lab)
        ax.fill_between(us_mrl, lo_, hi_, color=l.get_color(), alpha=0.15)
    for u in (31, 53, 89):
        ax.axvline(u, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("threshold u (mm/h)"); ax.set_ylabel("mean excess E[X-u | X>u] (mm/h)")
    ax.set_title("mean residual life (95% day-block bootstrap)"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/fig/mrl.png", dpi=130); plt.close(fig)

    # --- report -------------------------------------------------------------------------
    show = [10, 15, 19, 25, 31, 40, 53, 70, 89, 115, 150]
    with open(f"{OUT}/report.md", "w") as f:
        f.write("# POT threshold on v3\n\n`scripts/data_quality/pot_threshold.py`, "
                f"B = {B} day-block bootstrap replicates (B/2 for regions and seasons).\n\n")
        for kind in ("pixel", "cluster", "region", "season"):
            d = res[(res.kind == kind) & res.u.isin(show)].copy()
            d["xi [95%]"] = [f"{a:+.3f} [{b:+.3f}, {c:+.3f}]" for a, b, c in
                             zip(d.xi, d.xi_lo, d.xi_hi)]
            d["sigma* [95%]"] = [f"{a:.1f} [{b:.1f}, {c:.1f}]" for a, b, c in
                                 zip(d.sstar, d.sstar_lo, d.sstar_hi)]
            f.write(f"## {kind}\n\n" + md_table(
                d[["subset", "u", "n_exc", "days", "xi [95%]", "sigma* [95%]"]], index=False,
                floatfmt=".4g") + "\n\n")
    print(f"[analyze] wrote {OUT}/report.md, fits.csv, mrl.csv, fig/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "analyze", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    if a.stage in ("extract", "all"):
        extract(a.workers)
    if a.stage in ("analyze", "all"):
        analyze(a.workers, a.B)
