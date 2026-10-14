#!/usr/bin/env python
"""Figures for the 2026-10-14 screening slides (notes/slides/2026-10-14_screening/figs).

  gauges    Q-Q and exceedance of OPERA (raw, screened) against DE and CH gauges at the gauge pixel
            (calib/gauge_gate, 246 days)
  radklim   tail Q-Q and exceedance of OPERA (raw, screened) against RADKLIM on the same German
            pixel-hours (calib/rule_validate)
  europe    exceedance frequency of raw vs screened, 15-min and hourly, 100 random days, full grid
            (calib/dist_hist), with the provisional hourly u

    python scripts/data_quality/slides_figs.py gauges radklim europe
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
CAL = f"{D}/quality_v4/calib"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = f"{REPO}/notes/slides/2026-10-14_screening/figs"

# fixed categorical order (dataviz reference palette, slots 1-3); line style is a second channel
SCREENED = dict(color="#2a78d6", ls="-", lw=1.6, label="OPERA screened")
RAW = dict(color="#eb6834", ls="--", lw=1.6, label="OPERA raw")
REF = dict(color="#1baf7a", ls=":", lw=1.8, label="reference")
INK, MUTED, GRID = "#0b0b0b", "#6b6b66", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=110)      # preview only
    plt.close(fig)


def exceed_curve(ax, x, style, xs):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = len(x)
    s = np.sort(x)
    ax.plot(xs, (n - np.searchsorted(s, xs, side="left")) / n, **style)


def qq(ax, ref, fld, style, ps):
    ax.plot(np.quantile(ref, ps), np.quantile(fld, ps), marker="o", ms=3, **style)


def gauges():
    fs = sorted(glob.glob(f"{CAL}/gauge_gate/g_*.csv.gz"))
    g = pd.concat([pd.read_csv(f, usecols=["net", "raw", "full", "g_shift"]) for f in fs], ignore_index=True).dropna()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    xs = np.logspace(-1, 2.3, 200)
    ps = 1 - np.logspace(-1, -5, 17)
    for ax, net, name in zip(axs, ("dwd", "smn"), ("Germany (DWD)", "Switzerland (MeteoSwiss)")):
        s = g[g.net == net]
        lo, hi = 0.1, min(300, max(np.quantile(s.raw, ps[-1]), np.quantile(s.g_shift, ps[-1])) * 1.3)
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        qq(ax, s.g_shift, s.raw, RAW, ps)
        qq(ax, s.g_shift, s.full, SCREENED, ps)
        ax.set(xscale="log", yscale="log", xlim=(0.3, hi), ylim=(0.3, hi),
               xlabel="gauge quantile (mm in 1 h)", ylabel="OPERA quantile at the gauge pixel (mm in 1 h)")
        same = np.allclose(np.quantile(s.raw, ps), np.quantile(s.full, ps), rtol=0.02)
        ax.set_title(f"{name}: {len(s) / 1e6:.1f} M gauge-hours" + (" (raw = screened)" if same else ""),
                     fontsize=9, color=INK, loc="left")
        if np.quantile(s.raw, ps[-1]) > hi:
            ax.annotate(f"raw reaches {np.quantile(s.raw, ps[-1]):,.0f} mm at q99.999\n(a gauge on a clutter pixel)",
                        (0.04, 0.58), xycoords="axes fraction", fontsize=7, color=RAW["color"])
        for p, lab in ((0.999, "q99.9"), (0.9999, "q99.99"), (0.99999, "q99.999")):
            ax.annotate(lab, (np.quantile(s.g_shift, p), np.quantile(s.full, p)), fontsize=7, color=MUTED,
                        xytext=(4, -10), textcoords="offset points")
        if net == "dwd":
            ax.legend(loc="upper left")
    save(fig, "qq_gauges")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, net, name in zip(axs, ("dwd", "smn"), ("Germany", "Switzerland")):
        s = g[g.net == net]
        exceed_curve(ax, s.raw, RAW, xs)
        exceed_curve(ax, s.full, SCREENED, xs)
        exceed_curve(ax, s.g_shift, {**REF, "label": "gauges"}, xs)
        ax.set(xscale="log", yscale="log", xlabel="hourly accumulation x (mm)", ylabel="P(X >= x) per gauge-hour",
               ylim=(1e-7, 1))
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        if net == "dwd":
            ax.legend(loc="lower left")
    save(fig, "exceed_gauges")
    print("gauges: done")


def radklim():
    parts = []
    for f in sorted(glob.glob(f"{CAL}/rule_validate/days/*.npz")):
        z = np.load(f)
        if len(z["A"]):
            parts.append(pd.DataFrame({k: z[k] for k in ("A", "K", "base", "rejected")}))
    df = pd.concat(parts, ignore_index=True)
    k = df[~df.rejected.astype(bool)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    xs = np.logspace(np.log10(5), 2.6, 200)
    n = len(k)
    for fld, st in ((k.A, RAW), (k.base, SCREENED), (k.K, {**REF, "label": "RADKLIM"})):
        s = np.sort(fld.values)
        axs[0].plot(xs, n - np.searchsorted(s, xs, side="left"), **st)
    axs[0].set(xscale="log", yscale="log", xlabel="hourly accumulation x (mm)", ylabel="pixel-hours >= x")
    axs[0].set_title("Germany, same pixel-hours (196 days)", fontsize=9, color=INK, loc="left")
    axs[0].legend(loc="lower left")
    top = 20000
    ra, rc, rk = (np.sort(v.values)[::-1][:top] for v in (k.A, k.base, k.K))
    hi = max(ra[0], rk[0]) * 1.2
    axs[1].plot([5, hi], [5, hi], color=MUTED, lw=0.8)
    axs[1].plot(rk, ra, **{**RAW, "lw": 1.4})
    axs[1].plot(rk, rc, **{**SCREENED, "lw": 1.4})
    for r in (10, 100, 1000, 10000):
        axs[1].annotate(f"rank {r:,}", (rk[r - 1], rc[r - 1]), fontsize=7, color=MUTED, xytext=(4, -10),
                        textcoords="offset points")
    axs[1].set(xscale="log", yscale="log", xlim=(5, hi), ylim=(5, hi), xlabel="RADKLIM order statistic (mm)",
               ylabel="OPERA order statistic (mm)")
    axs[1].set_title("tail Q-Q (k-th largest against k-th largest)", fontsize=9, color=INK, loc="left")
    save(fig, "radklim")
    print("radklim: done")


def europe():
    tot = None
    cat = pd.read_csv(f"{CAL}/days.csv", dtype=str).set_index("day").category
    for f in sorted(glob.glob(f"{CAL}/dist_hist/days/*.npz")):
        if cat.get(os.path.basename(f)[:8]) != "random":
            continue
        z = np.load(f)
        edges = z["edges"]
        h = {k: z[k] for k in ("raw15", "scr15", "rawH", "scrH")}
        tot = h if tot is None else {k: tot[k] + h[k] for k in tot}
    u = open(f"{CAL}/dist_hist/u_provisional.txt").read().split("u = ")[1].split(" mm")[0]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, (kr, ks, xl, mark) in zip(axs, (("raw15", "scr15", "15-min rate x (mm/h)", 31.0),
                                            ("rawH", "scrH", "hourly accumulation x (mm)", float(u)))):
        for k, st in ((kr, RAW), (ks, SCREENED)):
            c = np.cumsum(tot[k][::-1])[::-1] / tot[k].sum()
            ax.plot(edges[1:-1], c[1:], **st)
        ax.axvline(mark, color=MUTED, lw=0.8)
        ax.annotate(f"{'31 mm/h' if kr == 'raw15' else f'u = {u} mm'}", (mark, 0.3), fontsize=7, color=MUTED,
                    xytext=(3, 0), textcoords="offset points")
        ax.set(xscale="log", yscale="log", xlim=(0.1, 3000), ylim=(1e-10, 1), xlabel=xl, ylabel="P(X >= x) per valid pixel")
        ax.set_title("15-min frames" if kr == "raw15" else "hourly sums", fontsize=9, color=INK, loc="left")
    axs[0].legend(loc="lower left")
    save(fig, "europe")
    print("europe: done")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for c in sys.argv[1:]:
        globals()[c]()
