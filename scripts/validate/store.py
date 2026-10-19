"""Stage validator: the patch store (v4 hourly; any store with the same layout).

  structure       groups, arrays, shapes against the metadata row counts
  metadata        sampled rows: no NaN, stored max = metadata max, sum <= max frame rate,
                  n_wet in 0..4
  coarse          coarse / interpolated fields = coarsen_and_interpolate(target)
  reproduction    random tile-hours recomputed from raw through the hourly chain (full grid,
                  as build_store_hourly.py) equal the stored target, max_rate and n_wet
  dem             DEM orientation by geography: Mont Blanc's pixel is high, a Dutch tile is low
                  (catches the N-S mirroring of 2026-09-28)
  scaler          log_precip_max_val = log1p(train max)
  adv             advection-corrected field: coverage and non-negativity (INFO + check)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import REPO, guarded, info, ok

sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]

D = "/home/fquareng/work/data/extremes/OPERA"
Q2 = f"{D}/quality_v2"
FIELDS = ["original_precip", "coarse_precip", "interpolated_precip", "max_rate", "n_wet"]


def groups(cfg):
    meta = os.path.dirname(cfg["TRAIN_METADATA_FILE"])
    owned = cfg["TRAIN_METADATA_FILE"].endswith("_owned.txt")        # DECISIONS §34 filtered lists
    return {"train": cfg["TRAIN_METADATA_FILE"], "validation": cfg["VAL_METADATA_FILE"],
            "test": cfg["TEST_METADATA_FILE"],
            "nimbus": os.path.join(meta, "full_nimbus_owned.txt" if owned else "full_nimbus.txt")}


def read_meta(path):
    """(timestamps, rows, cols, max, store_rows). A 5th column is the store row (filtered and
    subset lists); without it the line position is."""
    m = pd.read_csv(path, header=None, dtype={0: str})
    sr = m.iloc[:, 4].values.astype(int) if m.shape[1] >= 5 else np.arange(len(m))
    return (m.iloc[:, 0].values, m.iloc[:, 1].values.astype(int), m.iloc[:, 2].values.astype(int),
            m.iloc[:, 3].values.astype(float), sr)


def store(cfg, mode="r"):
    import zarr
    return zarr.open(os.path.join(cfg["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"), mode=mode)


def c_structure(cfg):
    z = store(cfg)
    out = []
    for g, mp in groups(cfg).items():
        if not os.path.exists(mp):
            out.append(ok(f"structure/{g}", g not in z, "no metadata file", path=mp)); continue
        _, _, _, _, sr = read_meta(mp)
        miss = [f for f in FIELDS if g not in z or f not in z[g]]
        if miss:
            out.append(ok(f"structure/{g}", False, f"missing arrays {miss}")); continue
        G = z[g]
        N = G["original_precip"].shape[0]
        same = all(G[f].shape[0] == N for f in FIELDS) and G["original_precip"].shape[1:] == (128, 128)
        refs = len(np.unique(sr)) == len(sr) and sr.min() >= 0 and sr.max() < N
        out.append(ok(f"structure/{g}", same and refs, "arrays agree, 128x128, metadata store rows unique and in range",
                      metadata_rows=len(sr), store_rows=N, coarse=str(G["coarse_precip"].shape[1:])))
    return out


def c_metadata(cfg, n_sample, seed=0):
    z = store(cfg)
    rng = np.random.default_rng(seed)
    out = []
    for g, mp in groups(cfg).items():
        if g not in z or not os.path.exists(mp):
            continue
        ts, ys, xs, mx, sr = read_meta(mp)
        idx = np.unique(np.r_[0, len(ts) - 1, rng.choice(len(ts), min(n_sample, len(ts)), replace=False)])
        G = z[g]
        bad = {"nan": 0, "max != metadata": 0, "sum > max_rate": 0, "n_wet out of 0..4": 0}
        for i in idx:
            t, r, w = G["original_precip"][sr[i]], G["max_rate"][sr[i]], G["n_wet"][sr[i]]
            bad["nan"] += bool(np.isnan(t).any())
            bad["max != metadata"] += abs(float(np.nanmax(t)) - mx[i]) > 1e-3 + 1e-4 * mx[i]
            bad["sum > max_rate"] += bool((t > r + 1e-3).any())
            bad["n_wet out of 0..4"] += bool((w > 4).any())
        out.append(ok(f"metadata/{g}", sum(bad.values()) == 0, f"{len(idx)} sampled rows", **bad))
    return out


def c_coarse(cfg, n_sample, seed=1):
    from src.data.preprocessing import coarsen_and_interpolate
    z = store(cfg)
    f = cfg["DOWNSCALING_FACTOR"]
    rng = np.random.default_rng(seed)
    out = []
    for g in groups(cfg):
        if g not in z:
            continue
        G = z[g]
        n = G["original_precip"].shape[0]
        idx = rng.choice(n, min(n_sample, n), replace=False)
        bad = 0
        for i in idx:
            c, it = coarsen_and_interpolate(G["original_precip"][i], f)
            bad += not (np.allclose(c, G["coarse_precip"][i], atol=1e-5) and np.allclose(it, G["interpolated_precip"][i], atol=1e-5))
        out.append(ok(f"coarse/{g}", bad == 0, f"{len(idx)} sampled rows", mismatches=bad))
    return out


def _reproduce(args):
    day, rows, quality = args
    import scan_tiles as st
    from src.data import cleaning as C
    from src.data.hourly import HourlyDay
    q, ceil = st._quality(quality)
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(f"{Q2}/clutter_climatology.npz", y),
                   ring=lambda y: st._ring(f"{Q2}/ring_mask.npz", y), sites_rc=st._sites(), ceilings=ceil,
                   cfg={**q.get("hourly", {}), "advection": None}).run()
    hours = {np.datetime_as_string(h["end"], unit="s").replace("-", "").replace("T", "").replace(":", ""): h for h in hd.hours}
    res = []
    for g, i, t, y, x, tgt, mr, nw in rows:
        h = hours.get(t)
        if h is None:
            res.append((g, i, np.inf, np.inf, 99)); continue
        sl = (slice(y, y + 128), slice(x, x + 128))
        tile = np.nan_to_num(h["sum"][sl], nan=0.0)
        f4 = np.nan_to_num(hd.fin[h["idx"]][(slice(None),) + sl], nan=0.0)
        res.append((g, i, float(np.abs(tile - tgt).max()),
                    float(np.abs(np.nan_to_num(h["max_rate"][sl], nan=0.0) - mr).max()),
                    int(np.abs((f4 >= C.DRIZZLE).sum(0).astype(np.uint8).astype(int) - nw.astype(int)).max())))
    return res


def c_reproduction(cfg, n_days, workers, quality, seed=2):
    z = store(cfg)
    rng = np.random.default_rng(seed)
    picks = []
    for g, mp in groups(cfg).items():
        if g not in z or not os.path.exists(mp):
            continue
        ts, ys, xs, mx, sr = read_meta(mp)
        for i in rng.choice(len(ts), max(1, n_days // 4), replace=False):
            picks.append((g, int(sr[i]), ts[i], int(ys[i]), int(xs[i])))
    jobs = {}
    for g, i, t, y, x in picks:
        T = pd.Timestamp(f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}")
        day = (T - pd.Timedelta(minutes=1)).strftime("%Y%m%d")
        G = z[g]
        jobs.setdefault(day, []).append((g, i, t, y, x, G["original_precip"][i], G["max_rate"][i], G["n_wet"][i]))
    with ProcessPoolExecutor(workers) as ex:
        res = [r for rr in ex.map(_reproduce, [(d, r, quality) for d, r in jobs.items()]) for r in rr]
    d_t = max(r[2] for r in res); d_m = max(r[3] for r in res); d_w = max(r[4] for r in res)
    return [ok("reproduction", d_t <= 1e-4 and d_m <= 1e-3 and d_w == 0,
               f"{len(res)} tile-hours on {len(jobs)} days recomputed from raw (full grid)",
               max_abs_target=d_t, max_abs_max_rate=d_m, max_abs_n_wet=d_w)]


def c_dem():
    from src.data import geo
    def pix(lon, lat):
        r, c = geo.lonlat_to_rowcol(lon, lat)
        r, c = int(np.round(r)), int(np.round(c))
        t = geo.dem_patch((r // 128) * 128, (c // 128) * 128)
        return float(t[r % 128, c % 128]), float(np.nanmean(t))
    mb, mb_tile = pix(6.865, 45.833)                 # Mont Blanc
    nl, nl_tile = pix(5.0, 52.3)                     # IJsselmeer area
    return [ok("dem/orientation", mb > 2000 and nl < 100 and mb_tile > nl_tile + 500,
               "Mont Blanc pixel high, Dutch pixel low, Alpine tile above Dutch tile",
               mont_blanc_m=mb, alpine_tile_mean_m=mb_tile, dutch_m=nl, dutch_tile_mean_m=nl_tile)]


def c_scaler(cfg):
    meta = cfg["TRAIN_METADATA_FILE"].replace("_owned.txt", ".txt")   # the scaler uses the full split
    _, _, _, mx, _ = read_meta(meta)
    v = float(np.load(os.path.join(cfg["PREPROCESSED_DATA_DIR"], "log_precip_max_val.npy")))
    return [ok("scaler", abs(v - np.log1p(mx.max())) < 1e-4, "log_precip_max_val = log1p(train max)",
               stored=v, expected=float(np.log1p(mx.max())), train_max=float(mx.max()))]


def c_adv(cfg, n_sample, seed=3):
    z = store(cfg)
    rng = np.random.default_rng(seed)
    out = []
    for g in groups(cfg):
        if g not in z or "adv_precip" not in z[g]:
            continue
        A = z[g]["adv_precip"]
        idx = rng.choice(A.shape[0], min(n_sample, A.shape[0]), replace=False)
        nan_rows = neg = 0
        for i in idx:
            a = A[i]
            nan_rows += bool(np.isnan(a).all())
            neg += bool((a[np.isfinite(a)] < -1e-6).any())
        out.append(ok(f"adv/{g}", neg == 0, f"{len(idx)} sampled rows; all-NaN rows are hours without a full "
                      "H-60..H+15 window or not yet filled", all_nan_share=nan_rows / len(idx), negative_rows=neg))
    return out


def run(cfg, a):
    checks = []
    checks += guarded("structure", c_structure, cfg)
    checks += guarded("metadata", c_metadata, cfg, a.n_sample)
    checks += guarded("coarse", c_coarse, cfg, max(50, a.n_sample // 10))
    checks += guarded("dem/orientation", c_dem)
    checks += guarded("scaler", c_scaler, cfg)
    checks += guarded("adv", c_adv, cfg, max(50, a.n_sample // 10))
    if a.n_days:
        checks += guarded("reproduction", c_reproduction, cfg, a.n_days, a.workers, a.quality)
    return checks
