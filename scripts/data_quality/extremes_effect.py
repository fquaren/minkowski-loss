#!/usr/bin/env python
"""Once the physically impossible values are gone, how much does the rest of the screen change
the distribution of extreme hourly precipitation? (2026-10-08; DECISIONS §21-§24)

On the pixels of fully covered tile-hours (the unit the dataset stores), per day:
  A  raw hourly sums, with every pixel-hour that has a frame > 500 mm/h removed: the baseline
     "only the unphysical values are removed"
  B  the hourly chain's repairs, on the same pixel-hours as A
  C  B without the tile-hours rejected at split time (`radar_screen.decide_hourly`, radar-disk
     flags with the temporal check, no radar-day exclusions): the dataset as it would be built
  D  A without those tile-hours: separates the effect of rejection (D vs A) from that of the
     repairs (C vs D)
Accumulated as histograms on fine log bins, plus the tile-hour maxima of A and B with each
tile-hour's rejection flag.

    python scripts/data_quality/extremes_effect.py run --out .../calib/extremes_effect --workers 6
    python scripts/data_quality/extremes_effect.py report --out .../calib/extremes_effect
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
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
D = "/home/fquareng/work/data/extremes/OPERA"
Q4 = f"{D}/quality_v4"
CAL = f"{Q4}/calib"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
EDGES = np.r_[0.0, np.logspace(-1, 3, 161)]          # mm per hour, 40 bins per decade to 1000 mm
UNPHYS = 500.0
VERSIONS = ("A", "B", "C", "D")


def frame_flags(day, q):
    """{(radar key, timestamp)} flagged by rule 3 with the temporal check, for the stores of
    `day` and the next day (the processing day ends at d+1 00:00)."""
    from src.data.radar_screen import frame_flagged_temporal
    out = set()
    for d in (day, (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y%m%d")):
        f = f"{Q4}/radars/{d}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        fl = frame_flagged_temporal(z["feats"], q)
        ti, ri = np.nonzero(fl)
        out |= {(str(z["keys"][r]), str(z["times"][t])) for t, r in zip(ti, ri)}
    return out


def run_day(args):
    day, out, off = (*args, ()) if len(args) == 2 else args
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    from src.data.radar_screen import decide_hourly
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    q = {**q, "hourly": {**q.get("hourly", {}), "rules": {**q["hourly"]["rules"], **{r: False for r in off}}}}
    t0 = time.time()
    tab = pd.read_csv(f"{CAL}/hourly/hours/{day}.csv.gz", dtype={"end": str, "frames": str})
    rej = decide_hourly(tab, q, frame_flags=frame_flags(day, q)) if len(tab) else pd.DataFrame()
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil, cfg=q).run()
    nb = len(EDGES) - 1
    H = {v: np.zeros(nb, np.int64) for v in VERSIONS}
    tiles = []
    by_end = {e: g for e, g in tab.assign(rejected=rej["rej_v4"].values if len(rej) else False).groupby("end")}
    for h in hd.hours:
        end = np.datetime_as_string(h["end"], unit="s").replace("-", "").replace("T", "").replace(":", "")
        g = by_end.get(end)
        if g is None:
            continue
        unphys = (np.nan_to_num(hd.raw[h["idx"]], nan=0.0) > UNPHYS).any(0)
        for r0, c0, rj in zip(g.row.values, g.col.values, g.rejected.values):
            sl = (slice(r0, r0 + 128), slice(c0, c0 + 128))
            keep = ~unphys[sl]
            a, b = h["raw_sum"][sl][keep], h["sum"][sl][keep]
            ia = np.clip(np.searchsorted(EDGES, a, side="right") - 1, 0, nb - 1)
            ib = np.clip(np.searchsorted(EDGES, b, side="right") - 1, 0, nb - 1)
            H["A"] += np.bincount(ia, minlength=nb)
            H["B"] += np.bincount(ib, minlength=nb)
            if not rj:
                H["C"] += np.bincount(ib, minlength=nb)
                H["D"] += np.bincount(ia, minlength=nb)
            tiles.append((end, r0, c0, float(a.max()) if a.size else np.nan,
                          float(b.max()) if b.size else np.nan, bool(rj), int(unphys[sl].sum())))
    os.makedirs(os.path.join(out, "days"), exist_ok=True)
    reasons = rej[[c for c in rej.columns if c.startswith("rej_") and c != "rej_v4"]].sum().to_dict() if len(rej) else {}
    np.savez_compressed(os.path.join(out, "days", f"{day}.npz"), **{f"h{v}": H[v] for v in VERSIONS},
                        tiles=np.array(tiles, dtype=object), reasons=np.array([reasons], dtype=object))
    return f"{day}: {len(tiles)} tile-hours, {int(sum(t[5] for t in tiles))} rejected, {time.time() - t0:.0f} s"


def _quantile(h, p):
    """p-quantile of the values >= 0.1 mm from a histogram (log-linear within the bin)."""
    w = h[1:].astype(float)
    c = np.cumsum(w) / w.sum()
    j = int(np.searchsorted(c, p))
    lo, hi = EDGES[1 + j], EDGES[2 + j]
    c0 = c[j - 1] if j else 0.0
    f = (p - c0) / max(c[j] - c0, 1e-12)
    return float(lo * (hi / lo) ** f)


def report(out):
    from scipy.stats import genpareto
    cats = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day")["category"]
    group = lambda c: "random" if c == "random" else ("event" if c.startswith("event") else "failure")
    Z = {os.path.basename(f)[:8]: np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(out, "days", "*.npz")))}
    us = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300)
    lines = []
    for grp in ("random", "event", "failure"):
        days = [d for d in Z if group(cats.get(d, "random")) == grp]
        if not days:
            continue
        H = {v: sum(Z[d][f"h{v}"] for d in days) for v in VERSIONS}
        cc = {v: H[v][::-1].cumsum()[::-1] for v in VERSIONS}
        T = pd.DataFrame([t for d in days for t in Z[d]["tiles"]],
                         columns=["end", "row", "col", "maxA", "maxB", "rejected", "n_unphys"])
        lines.append(f"\n=== {grp}: {len(days)} days, {len(T):,} tile-hours, {int(T.rejected.sum()):,} rejected "
                     f"({T.rejected.mean():.2%}); pixel-hours with a frame > 500 mm/h set aside: {int(T.n_unphys.sum()):,}")
        rows = []
        for u in us:
            j = np.searchsorted(EDGES, u)
            rows.append({"u_mm": u, "A_raw_no_unphys": int(cc["A"][j]), "B_repaired": int(cc["B"][j]),
                         "C_dataset": int(cc["C"][j]), "B/A": cc["B"][j] / max(cc["A"][j], 1),
                         "D/A rejection": cc["D"][j] / max(cc["A"][j], 1), "C/A total": cc["C"][j] / max(cc["A"][j], 1)})
        lines.append("pixel-hours >= u (fully covered tiles):\n" + pd.DataFrame(rows).round(3).to_string(index=False))
        qs = [0.99, 0.999, 0.9999, 0.99999]
        lines.append("quantiles of wet pixel-hours (>= 0.1 mm), mm:  " + "  ".join(
            f"q{p}: A {_quantile(H['A'], p):.1f} / C {_quantile(H['C'], p):.1f}" for p in qs))
        kept = T[~T.rejected]
        for lab, x in (("A, all tile-hours", T.maxA), ("C, kept tile-hours", kept.maxB)):
            x = x.dropna().values
            ex = {u: int((x >= u).sum()) for u in (20, 50, 100, 200)}
            u0 = 20.0
            e = x[x > u0] - u0
            xi = genpareto.fit(e, floc=0)[0] if len(e) > 50 else np.nan
            lines.append(f"tile-hour max, {lab}: n {len(x):,}; >= 20/50/100/200 mm: {ex}; GPD xi over 20 mm {xi:+.3f} (n {len(e)})")
        reasons = pd.DataFrame([dict(Z[d]["reasons"][0]) for d in days]).sum()
        lines.append("rejections by rule (tile-hours): " + ", ".join(f"{k[4:]} {int(v):,}" for k, v in reasons.items()))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "report.txt"), "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--off", nargs="*", default=[], help="hourly-chain rules switched off (attribution)")
    ap.add_argument("--group", default=None, choices=["random", "event", "failure"])
    a = ap.parse_args()
    if a.cmd == "report":
        return report(a.out)
    cats = pd.read_csv(f"{CAL}/days.csv", dtype=str)
    order = {"random": 0}
    cats["o"] = cats.category.map(lambda c: 0 if c == "random" else (1 if c.startswith("event") else 2))
    if a.group:
        cats = cats[cats.o == {"random": 0, "event": 1, "failure": 2}[a.group]]
    days = [d for d in cats.sort_values(["o", "day"]).day if not os.path.exists(os.path.join(a.out, "days", f"{d}.npz"))]
    print(f"[extremes_effect] {len(days)} days (random first) -> {a.out}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(run_day, [(d, a.out, tuple(a.off)) for d in days]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
