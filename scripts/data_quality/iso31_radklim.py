#!/usr/bin/env python
"""RADKLIM test of rule 3 of the radar-day exclusion (DECISIONS §28 addendum): are the isolated
>= 31 mm/h pixels that make a German radar-day score >= 1 (`iso31`) artefacts?

  run      per day: every isolated pixel-frame counted by `iso31` (owned by the radar under the
           contributing geometry, >= 31 mm/h in `clean_frame`, `radar_screen.isolated_mask`) for
           the radars of that day in the test or control set; its hour (ACRR window H-45..H),
           RADKLIM K and K_tol (+-1 h, +-2 px) and, for test radar-days, the screened hourly sum C
           of the current chain (`quality_v4.yaml`, burst rule on) at that pixel.
             test     German radar-days that §28 excludes with reason iso31
             control  German radar-days with iso31 > 0 and score < 1 on RADKLIM days already on disk
                      before the test (fixed list `control_days.txt`, written on first run)
  summary  pooled RADKLIM-dry share (K_tol < 5 mm) of the test pixel-frames with a 95% day-block
           bootstrap interval; pass >= 85%. Per radar, per radar-day, control, and the share left
           >= 5 mm by the hourly chain.

    python scripts/data_quality/iso31_radklim.py run --workers 3
    python scripts/data_quality/iso31_radklim.py summary
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2"), os.path.dirname(__file__)]
from radklim_presence import k_tolerant  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
RKD = f"{D}/validation/radklim/hourly"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{Q4}/calib/iso31_radklim"
QPATH = f"{REPO}/configs/quality_v4.yaml"


def radar_days():
    """(test, control) DataFrames of radar, day."""
    r = pd.read_csv(f"{Q4}/ranking.csv.gz", dtype={"day": str}, usecols=["radar", "day", "score", "reason", "iso31"])
    de = r[r.radar.str.startswith("de")]
    test = de[(de.score >= 1) & (de.reason == "iso31")][["radar", "day"]]
    cpath = f"{OUT}/control_days.txt"
    if not os.path.exists(cpath):
        os.makedirs(OUT, exist_ok=True)
        have = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{RKD}/*.npz"))
        open(cpath, "w").write("\n".join(have) + "\n")
    cdays = set(open(cpath).read().split())
    control = de[(de.score < 1) & (de.iso31 > 0) & de.day.isin(cdays)][["radar", "day"]]
    return test, control


def run_day(args):
    day, test_radars, control_radars = args
    import scan_tiles as st
    from src.data.cleaning import clean_frame
    from src.data.day_cleaner import open_day
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import ISO_U, RadarGeometry, isolated_mask, use_contributing
    t0 = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (y0, y1, x0, x1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, 0, 0.0
    Kt = k_tolerant(K24)
    q, ceil = st._quality(QPATH)
    rg = q["radar_geometry"]
    use_contributing(rg["nodes_table"], rg.get("valid_climatology"))
    year = int(day[:4])
    g = RadarGeometry(year)
    key_i = {k: i for i, k in enumerate(g.keys)}
    want = {key_i[k]: "test" for k in test_radars if k in key_i}
    want.update({key_i[k]: "control" for k in control_radars if k in key_i})
    sel = np.isin(g.own, list(want))
    pix, pown = g.own_idx[sel], g.own[sel]
    hot = st._hot(CLIM, year)
    ds = open_day(f"{D}/raw/OPERA/{day}")
    d1 = np.datetime64(pd.Timestamp(day)) + np.timedelta64(1, "h")          # end of hour index 0
    rows = []
    for t in range(ds.sizes["time"]):
        tt = ds.time.values[t].astype("datetime64[m]")
        H = tt + ((-tt.astype(np.int64)) % 60).astype("timedelta64[m]")       # hour end holding frame t
        i = int((H - d1) / np.timedelta64(1, "h"))
        if not 0 <= i < 24:
            continue
        c = clean_frame(ds["TOT_PREC"].isel(time=t).values.astype(np.float32), hot)[0]
        v = c.ravel()[pix]
        hi = np.isfinite(v) & (v >= ISO_U)
        if not hi.any():
            continue
        ys, xs = np.divmod(pix[hi], c.shape[1])
        iso = isolated_mask(c, ys, xs)
        if not iso.any():
            continue
        ys, xs, vv, oo = ys[iso], xs[iso], v[hi][iso], pown[hi][iso]
        inb = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
        k = np.full(len(ys), np.nan, np.float32)
        kt = np.full(len(ys), np.nan, np.float32)
        k[inb], kt[inb] = K24[i][ys[inb] - y0, xs[inb] - x0], Kt[i][ys[inb] - y0, xs[inb] - x0]
        rows.append(pd.DataFrame({"day": day, "radar": g.keys[oo], "group": [want[o] for o in oo],
                                  "time": str(tt), "hour": i, "row": ys, "col": xs, "value": vv,
                                  "K": k, "K_tol": kt}))
    ds.close()
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df["C"] = np.nan
    if len(df) and (df.group == "test").any():
        hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                       ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                       cfg=q.get("hourly", {}), crop=(y0, y1, x0, x1)).run()
        sums = {int((h["end"] - d1) / np.timedelta64(1, "h")): h["sum"] for h in hd.hours}
        m = (df.group == "test") & df.row.between(y0, y1 - 1) & df.col.between(x0, x1 - 1)
        df.loc[m, "C"] = [sums[h][r - y0, c - x0] if h in sums else np.nan
                          for h, r, c in zip(df.hour[m], df.row[m], df.col[m])]
    os.makedirs(f"{OUT}/days", exist_ok=True)
    df.to_csv(f"{OUT}/days/{day}.csv.gz", index=False, float_format="%.4g")
    return day, len(df), time.time() - t0


def boot(df, n=1000, seed=0):
    """Pooled dry share and its 95% day-block bootstrap interval."""
    g = df.groupby("day").agg(n=("dry", "size"), k=("dry", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    s = g.k.values[idx].sum(1) / g.n.values[idx].sum(1)
    return g.k.sum() / g.n.sum(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def summary():
    fs = sorted(glob.glob(f"{OUT}/days/*.csv.gz"))
    df = pd.concat([pd.read_csv(f, dtype={"day": str}) for f in fs], ignore_index=True)
    test, control = radar_days()
    keys = set(zip(test.radar, test.day)) | set(zip(control.radar, control.day))
    df = df[[k in keys for k in zip(df.radar, df.day)]]
    df = df[df.K_tol.notna()].assign(dry=lambda x: x.K_tol < 5.0)
    L = [f"# RADKLIM test of rule 3 (DECISIONS §28 addendum)", "",
         f"{len(fs)} days processed; pixel-frames inside RADKLIM coverage: "
         f"test {int((df.group == 'test').sum()):,}, control {int((df.group == 'control').sum()):,}", ""]
    for grp in ("test", "control"):
        x = df[df.group == grp]
        if not len(x):
            continue
        p, lo, hi = boot(x)
        L.append(f"## {grp}: {x.day.nunique()} days, {x[['radar', 'day']].drop_duplicates().shape[0]} radar-days, "
                 f"{len(x):,} pixel-frames")
        L.append(f"RADKLIM-dry (K_tol < 5 mm): {p:.1%} [{lo:.1%}, {hi:.1%}]"
                 + (f"  -> {'PASS' if p >= 0.85 else 'FAIL'} (pass >= 85%)" if grp == "test" else ""))
        L.append(f"K_tol quantiles q10/q50/q90 (mm): {x.K_tol.quantile([.1, .5, .9]).round(1).tolist()}")
        if grp == "test":
            c = x.C.dropna()
            L.append(f"left >= 5 mm by the hourly chain: {(c >= 5).mean():.1%} of {len(c):,} "
                     f"(median screened hour {c.median():.1f} mm)")
            dd = x[x.C.notna()]
            L.append(f"RADKLIM-dry share among those left >= 5 mm: {dd[dd.C >= 5].dry.mean():.1%}")
        pr = x.groupby("radar").agg(radar_days=("day", "nunique"), n=("dry", "size"), dry=("dry", "mean"))
        L += ["", "| radar | radar-days | pixel-frames | dry |", "|---|---|---|---|"]
        L += [f"| {r} | {a} | {b:,} | {c:.1%} |" for r, (a, b, c) in pr.sort_values("n", ascending=False).iterrows()]
        rd = x.groupby(["radar", "day"]).dry.mean()
        L += ["", f"per radar-day dry share: q10/q50/q90 = {rd.quantile([.1, .5, .9]).round(2).tolist()}; "
              f"radar-days with < 50% dry: {int((rd < 0.5).sum())} of {len(rd)}", ""]
    txt = "\n".join(L)
    print(txt)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary", "days"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    test, control = radar_days()
    if a.cmd == "days":                       # the RADKLIM days the fetcher needs
        print("\n".join(sorted(set(test.day))))
        return
    if a.cmd == "summary":
        summary()
        return
    tg, cg = test.groupby("day").radar.apply(list), control.groupby("day").radar.apply(list)
    days = sorted((set(tg.index) | set(cg.index)) & {os.path.basename(f)[:8] for f in glob.glob(f"{RKD}/*.npz")})
    days = [d for d in days if not os.path.exists(f"{OUT}/days/{d}.csv.gz")]
    print(f"[iso31_radklim] {len(days)} days -> {OUT}", flush=True)
    jobs = [(d, tg.get(d, []), cg.get(d, [])) for d in days]
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, s in ex.map(run_day, jobs):
            print(f"  {day}: {n} pixel-frames, {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
