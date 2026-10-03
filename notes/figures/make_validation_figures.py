#!/usr/bin/env python
"""Gauge-validation figure for notes/data_quality_assessment.tex (Part II).

Reads the outputs of `scripts/validation/validate_tail.py` and writes to
notes/figures/data_quality/:

  fig_gauge_corroboration.png  per product era, the share of radar pixels the gauge
                               corroborates (>= 1 mm/h, top row) and confirms as heavy
                               (>= 10 mm/h, bottom row), by raw radar intensity, for untouched
                               pixels and each cleaning class (`corroboration.csv`);
  fig_gauge_values.png         tail calibration: exceedance ratio N(radar >= u) / N(gauge >= u)
                               with its day-block bootstrap band, and the QQ of the two tails
                               (`exceedance.csv`, `qq.csv`, from --values_dir).

    python notes/figures/make_validation_figures.py [--validation_dir ...]
"""

import argparse
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_quality")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150,
                     "savefig.bbox": "tight"})

BINS = ["[10, 31)", "[31, 89)", "[89, 150)", "[150, 500)", "[500, inf)"]
LABELS = ["10–31", "31–89", "89–150", "150–500", "≥ 500"]
# fixed categorical order (dataviz reference palette, slots 1-5) + a marker per class,
# so identity never rests on colour alone
CLASSES = [("untouched", "untouched", "#2a78d6", "o"),
           ("no temporal support", "no temporal support", "#eb6834", "s"),
           ("on range ring", "on range ring", "#1baf7a", "^"),
           ("spike repaired", "spike repaired", "#eda100", "D"),
           ("tile rejected: unphysical", "in tile rejected as unphysical", "#e87ba4", "v")]
MIN_N = 20     # bins with fewer pixel-gauge pairs are not drawn


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validation_dir",
                    default="/home/fquareng/work/data/extremes/OPERA/validation/values_20261003")
    ap.add_argument("--values_dir",
                    default="/home/fquareng/work/data/extremes/OPERA/validation/values_20261003")
    a = ap.parse_args()
    c = pd.read_csv(os.path.join(a.validation_dir, "corroboration.csv"))
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    rows = [("corroborated", "gauge ≥ 1 mm/h"), ("gauge >= 10", "gauge ≥ 10 mm/h")]
    for j, era in enumerate(["ODYSSEY", "NIMBUS"]):
        for i, (col, ylab) in enumerate(rows):
            ax = axes[i, j]
            for cls, label, colour, marker in CLASSES:
                d = c[(c["class"] == cls) & (c["era"] == era)].set_index("raw bin")
                d = d.reindex(BINS)
                y = d[col].where(d["n"] >= MIN_N)
                if y.notna().sum() == 0:
                    continue
                ax.plot(range(len(BINS)), y.values, color=colour, marker=marker, ms=5,
                        lw=1.6, label=label, markeredgecolor="white", markeredgewidth=0.8)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(f"{era} (2012 – 2024-07-04)" if era == "ODYSSEY"
                             else f"{era} (2024-07-05 →)")
            if j == 0:
                ax.set_ylabel(f"share with {ylab}")
            if i == 1:
                ax.set_xticks(range(len(BINS)))
                ax.set_xticklabels(LABELS)
                ax.set_xlabel("raw radar rate at the gauge (mm/h)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    path = os.path.join(OUT, "fig_gauge_corroboration.png")
    fig.savefig(path)
    print(path)
    if os.path.exists(os.path.join(a.values_dir, "exceedance.csv")):
        values_figure(a.values_dir)


ERA_COLOUR = {"ODYSSEY": "#2a78d6", "NIMBUS": "#eb6834"}          # palette slots 1-2
# the 3x3-max variant only brackets the point-vs-area offset; it is in exceedance.csv
VARIANT_STYLE = {"v2 pixel": ("-", "o"), "v2 pixel, flags excluded": ("-.", "D"),
                 "raw pixel": ("--", "s")}


def values_figure(values_dir):
    exc = pd.read_csv(os.path.join(values_dir, "exceedance.csv"))
    q = pd.read_csv(os.path.join(values_dir, "qq.csv"))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for era, colour in ERA_COLOUR.items():
        for var, (ls, mk) in VARIANT_STYLE.items():
            e = exc[(exc["era"] == era) & (exc["radar"] == var) & (exc["network"] == "all")
                    & (exc["gauge"] == "best interval") & (exc["n_gauge"] >= 20)]
            if not len(e):
                continue
            a0.plot(e["u"], e["ratio"], ls=ls, marker=mk, ms=4, color=colour, lw=1.5,
                    markeredgecolor="white", markeredgewidth=0.6, label=f"{era}, {var}")
            if var == "v2 pixel":
                a0.fill_between(e["u"], e["q05"], e["q95"], color=colour, alpha=0.15, lw=0)
            d = q[(q["era"] == era) & (q["radar"] == var) & (q["gauge_k"] >= 5) & (q["radar_k"] > 0)]
            if len(d):
                a1.plot(d["gauge_k"], d["radar_k"], ls=ls, marker=mk, ms=3.5, color=colour, lw=1.5,
                        markeredgecolor="white", markeredgewidth=0.5)
    a0.axhline(1, color="0.5", lw=0.8)
    a0.set_xscale("log"); a0.set_yscale("log")
    a0.set_xticks([10, 20, 31, 53, 89, 150]); a0.set_xticklabels(["10", "20", "31", "53", "89", "150"])
    a0.set_xlabel("threshold u (mm/h)")
    a0.set_ylabel("N(radar ≥ u) / N(gauge ≥ u)")
    a0.set_title("(a) exceedance ratio, same station-frames")
    lim = [5, 400]
    a1.plot(lim, lim, color="0.5", lw=0.8)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(lim); a1.set_ylim(3, 3000)
    a1.set_xlabel("k-th largest gauge rate (mm/h)")
    a1.set_ylabel("k-th largest radar rate (mm/h)")
    a1.set_title("(b) QQ of the two tails")
    for ax in (a0, a1):
        ax.grid(color="0.92", lw=0.6, which="major"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, lab = a0.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12),
               fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_gauge_values.png")
    fig.savefig(path)
    print(path)


if __name__ == "__main__":
    main()
