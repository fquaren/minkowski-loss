#!/usr/bin/env python
"""v4 radar screen from the radar pass (`scan_radars.py`): ceilings, radar-frame flags,
radar-day ranking and the review gallery (DECISIONS §19, rules 2, 3 and 5).

  ceilings  per radar-year value histograms -> values counted >= min_count and >= min_ratio x
            the largest other count within +-window_db. Writes ceilings.csv (the table the v4 scan
            repairs) and ceilings_daily.csv (per radar, day and value).
  frames    rule 3 on every radar-frame with the thresholds of the quality config. Writes
            frame_flags.csv.gz (flagged radar-frames, joined onto tiles at split time) and
            radar_day.csv.gz (per radar-day aggregates). --keep_frames also writes every wet
            radar-frame (calibration only; large on the full archive).
  rank      radar-days scored against the radar's own 99th percentile in that season and
            its 5 nearest radars on the same day, per signal (share of owned pixel-frames
            >= 150 mm/h, ceiling pixels, flagged frames):
                score_s = log10((x + floor) / (max(q99_own, median_neighbours) + floor))
            i.e. decades above both references. Writes review.csv (top N, for the
            researcher's exclude / keep decision) and ranking.csv.gz (all).
  gallery   one image per reviewed radar-day: three frames of the clean_frame field around
            the radar, with its maximum-range circle and owned area. Writes review.md.

    python scripts/data_quality/radar_screen_v4.py all --quality configs/quality_v4.yaml
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_quality._md import md_table  # noqa: E402
from src.data.radar_screen import (FEATURES, VB0, VB1, active_sites, ceiling_candidates,  # noqa: E402
                                   dbz, frame_flagged)

NB = VB1 - VB0 + 1
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
FLOORS = {"f150": 1e-5, "ceil_px": 1.0, "n_flag": 0.5}


def day_files(d):
    return sorted(glob.glob(os.path.join(d, "radars", "[0-9]" * 8 + ".npz")))


# --------------------------------------------------------------------------------------------
def cmd_ceilings(q, d):
    c = q["ceiling"]
    files = day_files(d)
    by_year = {}
    for f in files:
        by_year.setdefault(int(os.path.basename(f)[:4]), []).append(f)
    rows, daily = [], []
    for year, fs in sorted(by_year.items()):
        act = active_sites(year)
        acc = np.zeros((len(act), NB), np.int64)
        keys = None
        for f in fs:
            z = np.load(f)
            keys = z["keys"]
            np.add.at(acc, (z["hist_key"] // NB, z["hist_key"] % NB), z["hist_n"])
        assert keys is not None and len(keys) == len(act) and (keys == act["key"].values.astype(str)).all()
        found = []
        for ri in np.nonzero(acc.sum(1))[0]:
            nz = np.nonzero(acc[ri])[0]
            vals, cnt, ratio = ceiling_candidates(nz + VB0, acc[ri, nz], c["min_count"], c["min_ratio"],
                                                   c["window_db"])
            for v, n, r in zip(vals, cnt, ratio):
                found.append((ri, int(round(v * 100)) - VB0))
                rows.append({"year": year, "radar": keys[ri], "location": act["location"].iat[ri],
                             "country": act["country"].iat[ri], "site_row": act["row"].iat[ri],
                             "site_col": act["col"].iat[ri], "value": v, "dbz": round(float(dbz(v)), 2),
                             "count": int(n), "ratio": round(float(r), 1)})
        if found:
            fk = {ri * NB + b: (ri, b) for ri, b in found}
            for f in fs:
                z = np.load(f)
                m = np.isin(z["hist_key"], list(fk))
                for k, n in zip(z["hist_key"][m], z["hist_n"][m]):
                    ri, b = fk[int(k)]
                    daily.append({"radar": keys[ri], "day": os.path.basename(f)[:8],
                                  "value": (b + VB0) / 100.0, "count": int(n)})
        print(f"[ceilings] {year}: {len(fs)} days, {len(found)} ceilings", flush=True)
    t = pd.DataFrame(rows, columns=["year", "radar", "location", "country", "site_row", "site_col",
                                    "value", "dbz", "count", "ratio"])
    dd = pd.DataFrame(daily, columns=["radar", "day", "value", "count"])
    if len(t):
        g = dd.assign(year=dd.day.str[:4].astype(int)).groupby(["year", "radar", "value"]).agg(
            n_days=("day", "nunique"), first=("day", "min"), last=("day", "max")).reset_index()
        t = t.merge(g, on=["year", "radar", "value"], how="left")
    t.to_csv(os.path.join(d, "ceilings.csv"), index=False)
    dd.to_csv(os.path.join(d, "ceilings_daily.csv"), index=False)
    print(f"[ceilings] {len(t)} radar-year ceilings -> {d}/ceilings.csv")


# --------------------------------------------------------------------------------------------
def _frames_day(args):
    f, q, keep = args
    z = np.load(f)
    feats, times, keys = z["feats"], z["times"].astype(str), z["keys"].astype(str)
    day = os.path.basename(f)[:8]
    fl = frame_flagged(feats, q)                                   # (T, R)
    valid = feats[..., 0] >= q["radar_frame"]["min_valid"]
    rows = []
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        for r in np.nonzero(valid.any(0))[0]:
            v = valid[:, r]
            F = feats[v, r]
            rows.append({"radar": keys[r], "day": day, "n_frames": int(v.sum()),
                         "n_flag": int(fl[:, r].sum()),
                         "f150": float(np.nanmean(F[:, 5])), "max_f150": float(np.nanmax(F[:, 5])),
                         "max_f31": float(np.nanmax(F[:, 3])), "max_wet": float(np.nanmax(F[:, 1])),
                         "max_mean": float(np.nanmax(F[:, 6])), "max_r2": float(np.nanmax(F[:, 7])),
                         "max_ejump": float(np.nanmax(F[:, 9])), "max_bjump": float(np.nanmax(F[:, 8]))})
    ti, ri = np.nonzero(fl)
    flags = pd.DataFrame({"radar": keys[ri], "timestamp": times[ti]})
    kept = None
    if keep:
        ti, ri = np.nonzero(valid & (feats[..., 1] >= 0.2))
        kept = pd.DataFrame(feats[ti, ri], columns=FEATURES)
        kept.insert(0, "timestamp", times[ti]); kept.insert(0, "radar", keys[ri])
    return pd.DataFrame(rows), flags, kept


def cmd_frames(q, d, workers, keep=False):
    files = day_files(d)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_frames_day, [(f, q, keep) for f in files], chunksize=8))
    rd = pd.concat([r[0] for r in res], ignore_index=True)
    ff = pd.concat([r[1] for r in res], ignore_index=True)
    rd.to_csv(os.path.join(d, "radar_day.csv.gz"), index=False)
    ff.to_csv(os.path.join(d, "frame_flags.csv.gz"), index=False)
    if keep:
        pd.concat([r[2] for r in res], ignore_index=True).to_csv(
            os.path.join(d, "frames_wet.csv.gz"), index=False)
    print(f"[frames] {len(files)} days, {len(rd):,} radar-days, {len(ff):,} flagged radar-frames "
          f"on {ff.assign(day=ff.timestamp.str[:8])[['radar', 'day']].drop_duplicates().shape[0]:,} radar-days")


# --------------------------------------------------------------------------------------------
def _neighbours(year, k=5):
    act = active_sites(year)
    xy = np.c_[act["x"].values, act["y"].values]
    from scipy.spatial import cKDTree
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    keys = act["key"].values
    return {keys[i]: list(keys[idx[i, 1:]]) for i in range(len(keys))}, act.set_index("key")


def cmd_rank(q, d, events_path="configs/prominent_events.yaml"):
    import yaml
    rd = pd.read_csv(os.path.join(d, "radar_day.csv.gz"), dtype={"day": str})
    cd = pd.read_csv(os.path.join(d, "ceilings_daily.csv"), dtype={"day": str})
    cpx = cd.groupby(["radar", "day"])["count"].sum().rename("ceil_px")
    rd = rd.merge(cpx, on=["radar", "day"], how="left").fillna({"ceil_px": 0})
    rd["year"] = rd.day.str[:4].astype(int)
    rd["season"] = rd.day.str[4:6].astype(int).map(SEASON)
    sig = list(FLOORS)
    for s in sig:
        q99 = rd.groupby(["radar", "season"])[s].quantile(0.99).rename(f"{s}_q99")
        rd = rd.merge(q99, on=["radar", "season"], how="left")
    # median of the 5 nearest radars on the same day (radars without data that day ignored)
    meta = {}
    for s in sig:
        rd[f"{s}_nb"] = 0.0
    for year, g in rd.groupby("year"):
        nb, act = _neighbours(year)
        meta[year] = act
        radars = sorted(set(g.radar) | {n for v in nb.values() for n in v})
        ri = {r: i for i, r in enumerate(radars)}
        days = sorted(g.day.unique())
        di = {x: i for i, x in enumerate(days)}
        gi, gr = g.day.map(di).values, g.radar.map(ri).values
        nbm = np.array([[ri[n] for n in nb.get(r, [])] + [len(radars)] * (5 - len(nb.get(r, [])))
                        for r in radars])                       # padded with an all-NaN column
        for s in sig:
            M = np.full((len(days), len(radars) + 1), np.nan)
            M[gi, gr] = g[s].values
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(M[:, nbm], axis=2)            # (days, radars)
            rd.loc[g.index, f"{s}_nb"] = np.nan_to_num(med[gi, gr], nan=0.0)
    for s in sig:
        ref = np.maximum(rd[f"{s}_q99"].fillna(0), rd[f"{s}_nb"].fillna(0))
        rd[f"score_{s}"] = np.log10((rd[s] + FLOORS[s]) / (ref + FLOORS[s]))
    sc = rd[[f"score_{s}" for s in sig]]
    rd["score"] = sc.max(axis=1)
    rd["reason"] = sc.idxmax(axis=1).str.replace("score_", "")
    ev = yaml.safe_load(open(events_path))["events"]
    def event_on(day):
        t = pd.Timestamp(day)
        return ",".join(e["id"] for e in ev if e.get("available", True) is not False
                        and pd.Timestamp(e["start"]) <= t <= pd.Timestamp(e["end"]))
    rd = rd.sort_values("score", ascending=False).reset_index(drop=True)
    rd.to_csv(os.path.join(d, "ranking.csv.gz"), index=False)
    N = q["radar_day"]["review_top"]
    top = rd[rd.score > 0].head(N).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["location"] = [meta[y].loc[r, "location"] if r in meta[y].index else "" for y, r in zip(top.year, top.radar)]
    top["country"] = [meta[y].loc[r, "country"] if r in meta[y].index else "" for y, r in zip(top.year, top.radar)]
    top["event_day"] = [event_on(x) for x in top.day]
    top["image"] = [f"gallery/{r:03d}_{x}_{k}.png" for r, x, k in zip(top["rank"], top.day, top.radar)]
    top["decision"] = ""
    top["note"] = ""
    cols = ["rank", "radar", "location", "country", "day", "score", "reason", "f150", "ceil_px",
            "n_flag", "max_f31", "max_r2", "max_ejump", "max_bjump", "event_day", "image",
            "decision", "note"]
    top[cols].to_csv(os.path.join(d, "review.csv"), index=False)
    n1 = int((rd.score >= 1).sum())
    cut = float(top.score.iloc[-1]) if len(top) else np.nan
    print(f"[rank] {len(rd):,} radar-days; score >= 1 (a decade above both references): {n1:,}; "
          f"review list: top {len(top)}, cut-off score {cut:.2f}")
    with open(os.path.join(d, "rank_summary.txt"), "w") as f:
        f.write(f"radar_days {len(rd)}\nscore_ge_1 {n1}\nscore_gt_0 {int((rd.score > 0).sum())}\n"
                f"review_top {len(top)}\ncutoff {cut}\n")


# --------------------------------------------------------------------------------------------
def _gallery_one(args):
    row, d, raw_dir, var, clim = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from src.data.cleaning import clean_frame, hot_mask_from_climatology
    from src.data.day_cleaner import open_day
    from src.data.radar_screen import RadarGeometry, radar_frame_features
    day, key = str(row["day"]), row["radar"]
    year = int(day[:4])
    g = RadarGeometry(year)
    k = list(g.keys).index(key)
    hot = hot_mask_from_climatology(dict(np.load(clim)), year) if clim else None
    ds = open_day(os.path.join(raw_dir, day))
    T = ds.sizes["time"]
    # pick the frame with the largest f150 (then f31) from a cheap pass over every 4th frame
    best, bt = -1.0, 0
    for t in range(0, T, 4):
        c = clean_frame(ds[var].isel(time=t).values.astype(np.float32), hot)[0]
        f = radar_frame_features(c, g)[k]
        s = np.nan_to_num(f[5]) * 10 + np.nan_to_num(f[3])
        if s > best:
            best, bt = s, t
    frames = sorted({max(bt - 8, 0), bt, min(bt + 8, T - 1)})
    sr, sc = g.sites["row"].iat[k], g.sites["col"].iat[k]
    rng = g.sites["maxrange_km"].iat[k]
    rpx = (rng if np.isfinite(rng) else 250.0) / 2.0
    h = int(rpx + 25)
    y0, y1 = int(max(sr - h, 0)), int(min(sr + h, g.shape[0]))
    x0, x1 = int(max(sc - h, 0)), int(min(sc + h, g.shape[1]))
    fig, ax = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 4.8), constrained_layout=True)
    ax = np.atleast_1d(ax)
    for a, t in zip(ax, frames):
        c = clean_frame(ds[var].isel(time=t).values.astype(np.float32), hot)[0]
        f = radar_frame_features(c, g)[k]
        im = a.imshow(np.where(c[y0:y1, x0:x1] >= 0.1, c[y0:y1, x0:x1], np.nan), origin="lower",
                      cmap="turbo", norm=LogNorm(0.1, 300), extent=(x0, x1, y0, y1))
        a.contour(np.arange(x0, x1), np.arange(y0, y1), (g.owner[y0:y1, x0:x1] == k).astype(float),
                  levels=[0.5], colors="w", linewidths=0.8)
        th = np.linspace(0, 2 * np.pi, 200)
        a.plot(sc + rpx * np.cos(th), sr + rpx * np.sin(th), "w--", lw=0.7)
        a.plot(sc, sr, "w^", ms=6)
        a.set_xlim(x0, x1); a.set_ylim(y0, y1)
        ts = str(ds.time.values[t])[11:16]
        a.set_title(f"{ts} UTC  wet {f[1]:.2f}  f31 {f[3]:.3f}  f150 {f[5]:.4f}\n"
                    f"R2_range {f[7]:.2f}  edge jump {f[9]:.2f}  boundary jump {f[8]:.2f}", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    ds.close()
    fig.colorbar(im, ax=ax, shrink=0.8, label="mm/h (clean_frame, before repairs)")
    fig.suptitle(f"#{row['rank']}  {key} {row['location']} ({row['country']})  {day}  score {row['score']:.2f} "
                 f"[{row['reason']}]  ceiling px {int(row['ceil_px'])}  flagged frames {int(row['n_flag'])}"
                 + (f"  EVENT DAY: {row['event_day']}" if isinstance(row["event_day"], str) and row["event_day"] else ""),
                 fontsize=10)
    out = os.path.join(d, row["image"])
    fig.savefig(out, dpi=90); plt.close(fig)
    return out


def cmd_gallery(q, d, cfg, workers, clim):
    rv = pd.read_csv(os.path.join(d, "review.csv"), dtype={"day": str})
    os.makedirs(os.path.join(d, "gallery"), exist_ok=True)
    jobs = [(r, d, cfg["RAW_OPERA_DATA_DIR"], cfg["PRECIP_VAR_NAME"], clim) for _, r in rv.iterrows()]
    with ProcessPoolExecutor(workers) as ex:
        for k, out in enumerate(ex.map(_gallery_one, jobs), 1):
            if k % 10 == 0:
                print(f"[gallery] {k}/{len(jobs)}", flush=True)
    s = open(os.path.join(d, "rank_summary.txt")).read() if os.path.exists(os.path.join(d, "rank_summary.txt")) else ""
    with open(os.path.join(d, "review.md"), "w") as f:
        f.write("# v4 radar-day review list\n\nMark each row `exclude` or `keep` in `review.csv` "
                "(column `decision`). Scores are decades above both the radar's own 99th "
                "percentile in that season and its 5 nearest radars that day. The field shown is "
                "the clean_frame output, before any repair. Dashed circle: maximum range; white "
                "contour: the area the radar owns.\n\n```\n" + s + "```\n\n")
        show = rv[["rank", "radar", "location", "day", "score", "reason", "ceil_px", "n_flag",
                   "event_day"]].copy()
        f.write(md_table(show, index=False, floatfmt=".2f") + "\n\n")
        for _, r in rv.iterrows():
            f.write(f"### #{r['rank']} {r['radar']} {r['location']} {r['day']}\n\n![]({r['image']})\n\n")
    print(f"[gallery] {len(jobs)} images -> {d}/gallery, {d}/review.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["ceilings", "frames", "rank", "gallery", "all"])
    ap.add_argument("--quality", default="configs/quality_v4.yaml")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default=None, help="override the quality config's data_dir")
    ap.add_argument("--climatology", default="/home/fquareng/work/data/extremes/OPERA/quality_v2/clutter_climatology.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep_frames", action="store_true")
    a = ap.parse_args()
    import yaml
    from src.utils import load_config
    q = yaml.safe_load(open(a.quality))
    d = a.data_dir or q["data_dir"]
    if a.cmd in ("ceilings", "all"):
        cmd_ceilings(q, d)
    if a.cmd in ("frames", "all"):
        cmd_frames(q, d, a.workers, a.keep_frames)
    if a.cmd in ("rank", "all"):
        cmd_rank(q, d)
    if a.cmd in ("gallery", "all"):
        cmd_gallery(q, d, load_config(a.config), a.workers, a.climatology)


if __name__ == "__main__":
    main()
