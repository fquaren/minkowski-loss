#!/usr/bin/env python
"""How the hourly chain changes the precipitation it keeps: distributions and time series,
raw vs filtered, and which rule removed what at which intensity (DECISIONS §17 safeguard 2:
report every rule's rate by intensity bin; §23).

  run      per day, run `src.data.hourly.HourlyDay` on the full grid and accumulate, over valid
           pixel-hours:
             - histograms of the raw and filtered hourly sums on log bins (0.1 .. 1e5 mm),
               and the 2-D histogram raw x filtered;
             - per rule, the pixel-hours it lowered (by >= 5% and >= 0.05 mm) and the amount it
               removed, by raw-hour bin (a pixel-hour lowered by several rules counts for each);
             - per hour, the domain mean, the 99.9th percentile and the counts >= 10 / 20 / 50 mm,
               raw and filtered.
           Writes <out>/days/YYYYMMDD.npz.
  report   pools the days: exceedance curves raw vs filtered, the share of pixel-hours each
           rule lowered per raw bin, the mass removed, and hourly time series for chosen days.
           Writes <out>/hourly_effect.png and summary tables.

    python scripts/data_quality/hourly_effect.py run --days_file .../calib/days.txt --out .../calib/hourly_effect
    python scripts/data_quality/hourly_effect.py report --out .../calib/hourly_effect --ts_days 20210714 20160529
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
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
EDGES = np.r_[0.0, np.logspace(-1, 5, 61)]            # mm per hour; bin 0 = [0, 0.1)
BITS = {"footprint": 1, "ray": 2, "ring": 4, "unsupported": 8, "ceiling": 16, "persist": 128,
        "clean_frame": 0}                               # hot, spike, drizzle floor: no code, by difference


def run_day(args):
    day, out = args
    import scan_tiles as st
    from src.data import cleaning as C
    from src.data.hourly import HourlyDay
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    t0 = time.time()
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q).run()
    nb = len(EDGES) - 1
    h_raw = np.zeros(nb, np.int64); h_fin = np.zeros(nb, np.int64)
    h2 = np.zeros((nb, nb), np.int64)
    n_low = {k: np.zeros(nb, np.int64) for k in BITS}
    m_low = {k: np.zeros(nb) for k in BITS}
    ts = []
    for h in hd.hours:
        ok = h["valid"]
        r, f = h["raw_sum"][ok], h["sum"][ok]
        br = np.clip(np.searchsorted(EDGES, r, side="right") - 1, 0, nb - 1)
        bf = np.clip(np.searchsorted(EDGES, f, side="right") - 1, 0, nb - 1)
        h_raw += np.bincount(br, minlength=nb); h_fin += np.bincount(bf, minlength=nb)
        np.add.at(h2, (br, bf), 1)
        code = np.bitwise_or.reduce(hd.code[h["idx"]], axis=0)[ok]
        low = (r - f) > np.maximum(0.05, 0.05 * r)        # lowered by >= 5% and >= 0.05 mm, so the
                                                          # drizzle floor's hundredths do not count
        coded = np.zeros(r.shape, bool)
        for k, b in BITS.items():
            if not b:
                continue
            m = low & ((code & b) > 0)
            coded |= m
            n_low[k] += np.bincount(br[m], minlength=nb)
            m_low[k] += np.bincount(br[m], weights=(r - f)[m], minlength=nb)
        m = low & ~coded                                  # lowered by clean_frame only
        n_low["clean_frame"] += np.bincount(br[m], minlength=nb)
        m_low["clean_frame"] += np.bincount(br[m], weights=(r - f)[m], minlength=nb)
        ts.append([str(h["end"])[:16], ok.sum(), r.mean(), f.mean(), np.quantile(r, .999), np.quantile(f, .999),
                   *[(r >= u).sum() for u in (10, 20, 50)], *[(f >= u).sum() for u in (10, 20, 50)],
                   r.max(), f.max()])
    os.makedirs(os.path.join(out, "days"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "days", f"{day}.npz"), h_raw=h_raw, h_fin=h_fin, h2=h2,
                        rules=np.array(list(BITS)), n_low=np.stack([n_low[k] for k in BITS]),
                        m_low=np.stack([m_low[k] for k in BITS]),
                        ts=np.array(ts, dtype=object) if ts else np.zeros((0, 14), object))
    return f"{day}: {len(hd.hours)} hours, {time.time() - t0:.0f} s"


TS_COLS = ["end", "n_valid", "mean_raw", "mean_fin", "q999_raw", "q999_fin", "n10_raw", "n20_raw",
           "n50_raw", "n10_fin", "n20_fin", "n50_fin", "max_raw", "max_fin"]


def report(out, ts_days, events=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(out, "days", "*.npz")))
    Z = [np.load(f, allow_pickle=True) for f in files]
    days = [os.path.basename(f)[:8] for f in files]
    rules = list(Z[0]["rules"])
    h_raw = sum(z["h_raw"] for z in Z); h_fin = sum(z["h_fin"] for z in Z)
    n_low = sum(z["n_low"] for z in Z); m_low = sum(z["m_low"] for z in Z)
    ctr = np.sqrt(np.maximum(EDGES[:-1], 0.05) * EDGES[1:])
    lo = EDGES[:-1]
    ccdf = lambda h: h[::-1].cumsum()[::-1]

    # tables
    tab = pd.DataFrame({"raw_bin_lo": lo, "n_raw": h_raw, "n_fin": h_fin})
    for i, r in enumerate(rules):
        tab[f"share_lowered_{r}"] = n_low[i] / np.maximum(h_raw, 1)
        tab[f"mm_removed_{r}"] = m_low[i]
    tab.to_csv(os.path.join(out, "by_intensity.csv"), index=False)
    for u in (0.1, 1, 10, 20, 50, 100, 200, 500):
        j = np.searchsorted(EDGES, u)
        print(f"pixel-hours >= {u:>5g} mm: raw {ccdf(h_raw)[j]:>12,}  filtered {ccdf(h_fin)[j]:>12,}  "
              f"ratio {ccdf(h_fin)[j] / max(ccdf(h_raw)[j], 1):.3f}")
    mass_raw = (h_raw * ctr).sum()
    print("mass removed by rule (share of the raw total, approx.):",
          {r: f"{m_low[i].sum() / mass_raw:.2e}" for i, r in enumerate(rules)})

    ts = pd.concat([pd.DataFrame(list(z["ts"]), columns=TS_COLS).assign(day=d) for z, d in zip(Z, days) if len(z["ts"])],
                   ignore_index=True)
    for c in TS_COLS[1:]:
        ts[c] = ts[c].astype(float)
    ts.to_csv(os.path.join(out, "hourly_series.csv"), index=False)

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    a = fig.add_subplot(gs[0, 0])
    a.loglog(lo[1:], ccdf(h_raw)[1:], label="raw", color="0.4")
    a.loglog(lo[1:], ccdf(h_fin)[1:], label="filtered", color="C0")
    a.set_xlabel("hourly accumulation u (mm)"); a.set_ylabel("pixel-hours >= u"); a.legend()
    a.set_title(f"exceedance, {len(days)} days", fontsize=10); a.grid(alpha=.3, which="both")
    b = fig.add_subplot(gs[0, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        b.semilogx(lo[1:], ccdf(h_fin)[1:] / ccdf(h_raw)[1:], color="C0")
    b.axhline(1, color="0.6", lw=.8); b.set_ylim(0, 1.1)
    b.set_xlabel("u (mm)"); b.set_ylabel("filtered / raw exceedances"); b.grid(alpha=.3, which="both")
    b.set_title("how much of the tail the screen keeps", fontsize=10)
    c = fig.add_subplot(gs[1, :])
    for i, r in enumerate(rules):
        sh = n_low[i] / np.maximum(h_raw, 1)
        if sh.max() > 0:
            c.loglog(ctr[1:], np.where(sh[1:] > 0, sh[1:], np.nan), marker=".", label=r)
    c.set_xlabel("raw hourly accumulation (mm)"); c.set_ylabel("share of pixel-hours lowered")
    c.set_title("per rule: share of pixel-hours it lowered, by raw intensity (safeguard 2: a steep rise needs a look)", fontsize=10)
    c.legend(ncol=4, fontsize=8); c.grid(alpha=.3, which="both")
    for k, d in enumerate(ts_days[:2]):
        x = ts[ts.day == d]
        ax = fig.add_subplot(gs[2, k])
        if not len(x):
            ax.set_title(f"{d}: not in the run"); continue
        t = pd.to_datetime(x["end"])
        ax.plot(t, x.max_raw, color="0.4", lw=1, label="max raw")
        ax.plot(t, x.max_fin, color="C0", lw=1, label="max filtered")
        ax.plot(t, x.q999_raw, color="0.4", lw=1, ls="--", label="q99.9 raw")
        ax.plot(t, x.q999_fin, color="C0", lw=1, ls="--", label="q99.9 filtered")
        ax.set_yscale("log"); ax.set_ylabel("mm per hour"); ax.grid(alpha=.3)
        ax2 = ax.twinx(); ax2.plot(t, x.n20_raw, color="0.4", lw=.6, alpha=.6); ax2.plot(t, x.n20_fin, color="C0", lw=.6, alpha=.6)
        ax2.set_ylabel("pixel-hours >= 20 mm (thin)")
        ax.set_title(f"{d}: hourly max, q99.9 and count >= 20 mm", fontsize=10); ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(axis="x", labelsize=7)
    fig.savefig(os.path.join(out, "hourly_effect.png"), dpi=90)
    print("figure:", os.path.join(out, "hourly_effect.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--days_file", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ts_days", nargs="*", default=["20210714", "20160529"])
    a = ap.parse_args()
    if a.cmd == "report":
        return report(a.out, a.ts_days)
    days = list(a.days or [])
    if a.days_file:
        days += [x.strip() for x in open(a.days_file).read().replace(",", "\n").split() if x.strip()]
    days = [d.replace("-", "") for d in days]
    days = [d for d in days if not os.path.exists(os.path.join(a.out, "days", f"{d}.npz"))]
    print(f"[hourly_effect] {len(days)} days -> {a.out}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(run_day, [(d, a.out) for d in days]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
