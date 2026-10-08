# Copied 2026-10-08 from the calibration run of 2026-10-07 (quality_v4/calib/daily_check/),
# unchanged, to keep the method in the repo (DECISIONS §21-22). Paths are hard-coded.
"""Radar (v4-cleaned, crop over DE + CH) vs 10-min gauges at three aggregations.
Alignment from validation/alignment.csv: radar frame t <-> gauge interval labelled t + 10.
  15min: rate at t vs 6 x gauge(t+10)                   (mm/h)
  1h   : 0.25 x sum of frames h:00..h:45 vs gauge intervals labelled h:10..h+1:00   (mm)
  24h  : 0.25 x sum of 96 frames vs 144 gauge intervals labelled 00:10..24:00        (mm)
Only gauges with a complete day (144 values) and radar valid in all frames used; gauges on a
hot pixel of that year are dropped."""
import sys, os, importlib.util, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

REPO = "/work/fquareng/ch2/minkowski-loss"
G = "/home/fquareng/work/data/extremes/OPERA/validation/gauges"
OUT = sys.argv[1]
DAYS = sys.argv[2].split(",")


def load_rg():
    sys.argv = ["x"]
    sys.path[:0] = [REPO, f"{REPO}/scripts/data_quality"]
    spec = importlib.util.spec_from_file_location("rg", f"{REPO}/scripts/data_quality/rule_gallery.py")
    rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
    return rg


def stations():
    out = []
    for net in ("dwd", "smn"):
        s = pd.read_csv(f"{G}/{net}_stations.csv", dtype={"station": str})
        s["net"] = net; s["sid"] = np.arange(len(s))
        out.append(s[s.row.notna() & (s.n_valid_days > 0)])
    return pd.concat(out, ignore_index=True)


_CACHE = {}


def gauge_day(net, day, sids):
    """(n_sid, 144) mm per 10 min for intervals labelled day 00:10 .. day+1 00:00; NaN if incomplete."""
    if net not in _CACHE:
        a = np.load(f"{G}/{net}_avail.npz"); w = np.load(f"{G}/{net}_wet.npz")
        _CACHE[net] = ({"n_valid": a["n_valid"], "days": a["days"]}, {k: w[k] for k in w.files})
    av, w = _CACHE[net]
    days = [str(d)[:10] for d in av["days"]]
    j = days.index(day) if day in days else None
    t0 = int(np.datetime64(day + "T00:10").astype("datetime64[m]").astype(np.int64))
    sel = (w["t"] >= t0) & (w["t"] <= t0 + 143 * 10)
    sid, t, rr = w["sid"][sel], w["t"][sel], w["rr"][sel]
    M = np.zeros((len(sids), 144), np.float32)
    pos = {s: i for i, s in enumerate(sids)}
    for s, tt, r in zip(sid, t, rr):
        if s in pos:
            M[pos[s], (tt - t0) // 10] = r
    if j is not None:                                    # completeness of the labelled day
        ok = av["n_valid"][sids, j] >= 144
        M[~ok] = np.nan
    else:
        M[:] = np.nan
    return M


def run_day(day):
    rg = load_rg()
    import scan_tiles as st
    st_df = stations()
    y0, y1 = int(st_df.row.min()) - 48, int(st_df.row.max()) + 48
    x0, x1 = int(st_df.col.min()) - 48, int(st_df.col.max()) + 48
    t_start = time.time()
    d = day.replace("-", "")
    import yaml
    from src.data.day_cleaner import DayCleaner
    q = yaml.safe_load(open(rg.QPATH)); _, ceil = st._quality(rg.QPATH)
    dc = DayCleaner(f"{rg.D}/raw/OPERA", d, "TOT_PREC", hot=lambda y: st._hot(rg.CLIM, y),
                    ring=lambda y: st._ring(rg.RING, y), sites_rc=st._sites(), ceilings=ceil,
                    max_size=q["guard"]["max_size"], crop=(y0, y1, x0, x1))
    if dc.T != 96:
        return f"{day}: {dc.T} frames, skipped"
    r = (st_df.row.values.astype(int) - y0); c = (st_df.col.values.astype(int) - x0)
    hot = st._hot(rg.CLIM, int(d[:4]))[y0:y1, x0:x1][r, c]
    R = np.full((len(st_df), 96), np.nan, np.float32)
    RAW = np.full((len(st_df), 96), np.nan, np.float32)
    REJ = np.zeros((len(st_df), 96), bool)
    gt = (st_df.row.values.astype(int) // 128) * 1000 + st_df.col.values.astype(int) // 128
    utile = np.unique(gt)
    for k in range(96):
        fin, code = dc.clean(k)
        R[:, k] = fin[r, c]
        RAW[:, k] = dc._read(dc.ds, k)[r, c]
        # v4 tile rejection by the frame-level code bits: refused repair, or >= 5 ceiling pixels
        for tid in utile:
            ty, tx = divmod(tid, 1000)
            a0, a1 = max(ty * 128 - y0, 0), min(ty * 128 + 128 - y0, y1 - y0)
            b0, b1 = max(tx * 128 - x0, 0), min(tx * 128 + 128 - x0, x1 - x0)
            cd = code[a0:a1, b0:b1]
            if ((cd & 32) > 0).any() or ((cd & 16) > 0).sum() >= 5:
                REJ[gt == tid, k] = True
    Gm = np.full((len(st_df), 144), np.nan, np.float32)
    for net in ("dwd", "smn"):
        m = (st_df.net == net).values
        Gm[m] = gauge_day(net, day, st_df.sid.values[m])
    keep = np.isfinite(R).all(1) & np.isfinite(Gm).all(1)
    R, Gm, sdf, RAW, REJ = R[keep], Gm[keep], st_df[keep], RAW[keep], REJ[keep]
    RAW = np.where(np.isfinite(RAW), RAW, 0.0)
    rows = []
    # gauge interval i (labelled 00:10 + 10 i) ; radar frame k at 15 k min <-> label 15k + 10 -> i = 1.5 k
    for k in range(96):
        i = (15 * k) // 10                                 # interval labelled t+10 (floored grid)
        rows.append(pd.DataFrame(dict(day=day, agg="15min", win=k, station=sdf.station.values, net=sdf.net.values,
                                      radar=R[:, k], raw=RAW[:, k], rej=REJ[:, k], gauge=6 * Gm[:, i])))
    for h in range(24):
        rows.append(pd.DataFrame(dict(day=day, agg="1h", win=h, station=sdf.station.values, net=sdf.net.values,
                                      radar=0.25 * R[:, 4 * h:4 * h + 4].sum(1), raw=0.25 * RAW[:, 4 * h:4 * h + 4].sum(1),
                                      rej=REJ[:, 4 * h:4 * h + 4].any(1), gauge=Gm[:, 6 * h:6 * h + 6].sum(1))))
    rows.append(pd.DataFrame(dict(day=day, agg="24h", win=0, station=sdf.station.values, net=sdf.net.values,
                                  radar=0.25 * R.sum(1), raw=0.25 * RAW.sum(1), rej=REJ.any(1), gauge=Gm.sum(1))))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"{OUT}/g_{d}.csv.gz", index=False)
    return f"{day}: {keep.sum()} gauges, {time.time() - t_start:.0f} s"


if __name__ == "__main__":
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for msg in ex.map(run_day, DAYS):
            print(msg, flush=True)
