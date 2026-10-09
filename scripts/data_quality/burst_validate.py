#!/usr/bin/env python
"""Acceptance test of the isolated single-frame burst rule (DECISIONS §26), with RADKLIM as a
presence reference (dry = < 5 mm within +-2 px and +-1 h; DECISIONS §25).

For each RADKLIM-test day, over Germany, the hourly chain runs three times: without the rule (C),
with it as pre-stated (Cb: isolation median < 1 mm) and with the 5 mm sensitivity variant (Cb5).
Kept = tile-hour not rejected at split time (`decide_hourly`, as `radklim_test.py`).
  run      per day, the pixel-hours with max(raw A, RADKLIM K) >= 5 mm: A, C, Cb, Cb5, K, K_tol,
           rejected, hour, row, col
  summary  1. removals by the rule (C >= 1 mm and Cb <= 0.5 C, kept; C >= 1 excludes pixel-hours
              already removed by earlier rules): RADKLIM-dry share per raw bin (pass:
              >= 85% in every bin with n >= 30)
           2. RADKLIM-dry share of kept screened values >= 20 / 30 / 50 / 75 / 100 mm, without and
              with the rule (and the 5 mm variant)
           3. removals where RADKLIM confirms the raw value (K_tol >= 0.5 A): count and days
           4. flag rate by raw intensity; random / event / failure days

    python scripts/data_quality/burst_validate.py run --workers 8
    python scripts/data_quality/burst_validate.py summary
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
CAL = f"{D}/quality_v4/calib"
RKD = f"{D}/validation/radklim/hourly"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
OUT = f"{CAL}/burst_validate"


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import decide_hourly
    from extremes_effect import frame_flags
    t0 = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (y0, y1, x0, x1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, 0, 0, 0.0
    Kt = k_tolerant(K24)
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    tab = pd.read_csv(f"{CAL}/hourly/hours/{day}.csv.gz", dtype={"end": str, "frames": str})
    rej = decide_hourly(tab, q, frame_flags=frame_flags(day, q))["rej_v4"].values if len(tab) else np.zeros(0, bool)
    rejected_tiles = {(e, r, c) for e, r, c, x in zip(tab.end, tab.row, tab.col, rej) if x}
    runs = {}
    for name, burst, iso in (("C", False, 1.0), ("Cb", True, 1.0), ("Cb5", True, 5.0)):
        cfg = dict(q.get("hourly", {}))
        cfg["rules"] = {**cfg.get("rules", {}), "burst": burst}
        cfg["burst"] = {"min_share": 0.8, "min_rate": 100.0, "iso_median": iso}
        runs[name] = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                               ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                               cfg=cfg, crop=(y0, y1, x0, x1)).run()
    base = runs["C"]
    d0 = np.datetime64(pd.Timestamp(day))
    cols = {k: [] for k in ("A", "C", "Cb", "Cb5", "K", "K_tol", "rejected", "hour", "row", "col")}
    nb = 0
    for j, h in enumerate(base.hours):
        i = int((h["end"] - d0) / np.timedelta64(1, "h")) - 1
        if not 0 <= i < 24:
            continue
        end = np.datetime_as_string(h["end"], unit="s").replace("-", "").replace("T", "").replace(":", "")
        a, k = h["raw_sum"], K24[i]
        unphys = (np.nan_to_num(base.raw[h["idx"]], nan=0.0) > 500).any(0)
        m = np.isfinite(k) & np.isfinite(a) & ~unphys & (np.maximum(a, np.nan_to_num(k)) >= 5.0)
        if not m.any():
            continue
        yy, xx = np.nonzero(m)
        rr, cc = yy + y0, xx + x0
        cols["A"].append(a[m]); cols["K"].append(k[m]); cols["K_tol"].append(Kt[i][m])
        cols["hour"].append(np.full(m.sum(), i, np.int8)); cols["row"].append(rr.astype(np.int16)); cols["col"].append(cc.astype(np.int16))
        cols["rejected"].append(np.array([(end, (r // 128) * 128, (c // 128) * 128) in rejected_tiles for r, c in zip(rr, cc)]))
        for name, hd in runs.items():
            hh = [x for x in hd.hours if x["end"] == h["end"]][0]
            cols[name].append(hh["sum"][m])
    for name in ("Cb", "Cb5"):
        nb += int((runs[name].code & 256).astype(bool).sum()) if name == "Cb" else 0
    os.makedirs(os.path.join(OUT, "days"), exist_ok=True)
    arr = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    np.savez_compressed(os.path.join(OUT, "days", f"{day}.npz"), **arr)
    return day, len(arr["A"]), nb, time.time() - t0


def summary():
    cats = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day")["category"]
    grp = lambda d: "random" if cats.get(d) == "random" else ("event" if str(cats.get(d, "")).startswith("event") else "failure")
    frames = []
    for f in sorted(glob.glob(f"{OUT}/days/*.npz")):
        z = np.load(f)
        if len(z["A"]):
            d = os.path.basename(f)[:8]
            frames.append(pd.DataFrame({k: z[k] for k in z.files}).assign(day=d, grp=grp(d)))
    X = pd.concat(frames, ignore_index=True)
    X["rejected"] = X.rejected.astype(bool)
    X["dry5"] = X.K_tol < 5
    kept = X[~X.rejected & X.K_tol.notna()]
    lines = [f"{X.day.nunique()} days, {len(X):,} pixel-hours with max(A, K) >= 5 mm (Germany)"]
    for v in ("Cb", "Cb5"):
        rem = kept[(kept.C >= 1.0) & (kept[v] <= 0.5 * kept.C)]   # C >= 1: the rule lowered something
                                                                  # the earlier rules had kept
        rows, ok = [], True
        for lo, hi in ((10, 20), (20, 50), (50, 100), (100, np.inf)):
            b = rem[(rem.A >= lo) & (rem.A < hi)]
            allb = kept[(kept.A >= lo) & (kept.A < hi)]
            dry = float(b.dry5.mean()) if len(b) else np.nan
            p = None if len(b) < 30 else bool(dry >= 0.85)
            ok &= p is not False
            rows.append({"raw_bin": f"{lo}-{hi}", "n_pixel_hours": len(allb), "n_removed": len(b),
                         "flag_rate": len(b) / max(len(allb), 1), "removed RADKLIM-dry": dry, "pass (>= 0.85)": p})
        lines.append(f"\n1. removals by the rule, variant {v} ({'pre-stated, iso < 1 mm' if v == 'Cb' else 'sensitivity, iso < 5 mm'}): "
                     f"{'PASS' if ok else 'FAIL'}\n" + pd.DataFrame(rows).round(4).to_string(index=False))
        conf = rem[rem.K_tol >= 0.5 * rem.A]
        lines.append(f"3. removals RADKLIM confirms (K_tol >= 0.5 A): {len(conf)} of {len(rem)}"
                     + (f"; days: {', '.join(f'{d} ({n})' for d, n in conf.day.value_counts().head(8).items())}" if len(conf) else ""))
    for g in ("all", "random", "event", "failure"):
        x = kept if g == "all" else kept[kept.grp == g]
        rows = []
        for u in (20, 30, 50, 75, 100):
            r = {"u_mm": u}
            for v in ("C", "Cb", "Cb5"):
                t = x[x[v] >= u]
                r[f"{v} n"] = len(t); r[f"{v} dry"] = float(t.dry5.mean()) if len(t) else np.nan
            rows.append(r)
        lines.append(f"\n2. {g}: kept screened values >= u and their RADKLIM-dry share, without (C) and with the rule (Cb, Cb5):\n"
                     + pd.DataFrame(rows).round(3).to_string(index=False))
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/summary.txt", "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    days = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{CAL}/radklim_test/days/*.npz"))
    days = [d for d in days if not os.path.exists(os.path.join(OUT, "days", f"{d}.npz"))]
    print(f"[burst_validate] {len(days)} days -> {OUT}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, nb, dt in ex.map(run_day, days):
            print(f"  {day}: {n:,} pixel-hours, {nb} burst pixel-frames, {dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
