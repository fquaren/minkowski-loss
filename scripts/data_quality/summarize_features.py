#!/usr/bin/env python
"""Summarise the tile-feature tables: what is in the data, and where the artefacts sit.

Reads `<features_dir>/*.csv.gz` (from `compute_tile_features.py`), tags each tile with its
current split (train / val / test / none — "none" = never made it into the patch set, e.g.
the newly fetched archive), applies the candidate rules in `rules.py`, and writes to
`<out_dir>`:

    summary.md             the tables below, readable on their own
    flag_rates_by_stratum.csv, rule_overlap_tail.csv, flag_rates_by_{year,month,hour}.csv
    tile_flag_rate_tail.csv    per tile location: share of tail tiles flagged
    flagged_tail.csv       every flagged tile with max >= --tail, for patch_gallery.py
    figures/*.png          feature histograms by intensity stratum, max vs wet-area scatter,
                           flag rate by hour / month / tile location

Strata are on the **raw** tile maximum, so the tail is defined before our declutter step
removes anything. `--labels` (a CSV with timestamp,row,col,label as written by
`patch_gallery.py`, label in {artefact, real, unsure}) adds per-rule precision / recall.

    python scripts/data_quality/summarize_features.py config.yaml \
        --features_dir /home/fquareng/work/data/extremes/OPERA/quality/features \
        --out_dir /home/fquareng/work/data/extremes/OPERA/quality/summary
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
from src.utils import load_config  # noqa: E402
from _md import md_table  # noqa: E402
from rules import RULES, INFORMATIONAL, apply_rules  # noqa: E402

STRATA = [0, 0.1, 1, 10, 31, 53, 89, 150, 500, np.inf]
HIST_FEATURES = ["conc", "isolated_frac_ge1", "peak_block_ratio", "peak_nbr_ratio",
                 "max_abs_grad", "largest_comp_ge1", "spoke_len", "persist_prev",
                 "corr_prev", "wet_over_sea_frac", "q_at_max", "clim_freq_ge31_at_max"]


def load_features(features_dir, sample_frac=None, seed=0):
    files = sorted(glob.glob(os.path.join(features_dir, "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"no feature tables in {features_dir}")
    parts = []
    for f in files:
        d = pd.read_csv(f, dtype={"timestamp": str})
        if sample_frac:
            d = d.sample(frac=sample_frac, random_state=seed)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S")
    return df, len(files)


def tag_split(df, cfg):
    df["split"] = "none"
    key = ["timestamp", "row", "col"]
    for s, k in (("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")):
        m = pd.read_csv(cfg[f"{k}_METADATA_FILE"], header=None, usecols=[0, 1, 2],
                        names=key, dtype={"timestamp": str})
        m["_in"] = True
        hit = df[key].merge(m, on=key, how="left")["_in"].notna().values
        df.loc[hit, "split"] = s
    return df


def stratum(df):
    labels = [f"[{a:g},{b:g})" for a, b in zip(STRATA[:-1], STRATA[1:])]
    return pd.cut(df["max"], STRATA, right=False, labels=labels)


def rates(flags, by):
    """Share flagged per group, NaN-aware; plus group size."""
    g = flags.groupby(by, observed=True)
    out = g.mean()
    out.insert(0, "n", g.size())
    return out


def figures(df, flags, tail, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    feats = [f for f in HIST_FEATURES if f in df and df[f].notna().any()]
    wet = df[df["max"] >= 1]
    groups = [("1-10", (wet["max"] < 10)), ("10-31", (wet["max"] >= 10) & (wet["max"] < 31)),
              ("31-89", (wet["max"] >= 31) & (wet["max"] < 89)), (">=89", wet["max"] >= 89)]
    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, f in zip(np.atleast_1d(axes).ravel(), feats):
        v = wet[f]
        pos = v[v > 0]
        logx = len(pos) and pos.max() / max(pos.min(), 1e-9) > 100
        bins = (np.logspace(np.log10(max(pos.min(), 1e-6)), np.log10(pos.max()), 50)
                if logx else 50)
        for name, m in groups:
            x = v[m].dropna()
            if len(x):
                ax.hist(x[x > 0] if logx else x, bins=bins, histtype="step", density=True,
                        label=f"max {name} ({len(x):,})")
        if logx:
            ax.set_xscale("log")
        ax.set_title(f)
    for ax in np.atleast_1d(axes).ravel()[len(feats):]:
        ax.set_axis_off()
    np.atleast_1d(axes).ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_hist_by_stratum.png"), dpi=130)
    plt.close(fig)

    # max vs largest wet component: storms grow with intensity, clutter does not
    fig, ax = plt.subplots(figsize=(7, 6))
    t = df[df["max"] >= 10]
    af = flags.loc[t.index, "any_flag"].fillna(0) > 0
    ax.scatter(t.loc[~af, "max"], t.loc[~af, "largest_comp_ge1"] + 1, s=2, alpha=0.3,
               label="no flag", c="tab:blue")
    ax.scatter(t.loc[af, "max"], t.loc[af, "largest_comp_ge1"] + 1, s=2, alpha=0.3,
               label="any flag", c="tab:red")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("raw tile max (mm/h)"); ax.set_ylabel("largest component >=1 mm/h (px) + 1")
    ax.axvline(150, ls=":", c="k"); ax.legend(markerscale=5)
    fig.savefig(os.path.join(fig_dir, "max_vs_wet_area.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    ft = flags.loc[tail.index]
    for key, fname in ((tail["time"].dt.hour, "hour"), (tail["time"].dt.month, "month")):
        r = ft.groupby(key.values).mean()
        fig, ax = plt.subplots(figsize=(9, 4))
        for c in [c for c in r.columns if r[c].notna().any()]:
            ax.plot(r.index, r[c], marker="o", ms=3, label=c, lw=2 if c == "any_flag" else 1)
        ax.set_xlabel(fname); ax.set_ylabel("share of tail tiles flagged")
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(os.path.join(fig_dir, f"flag_rate_by_{fname}.png"), dpi=130,
                    bbox_inches="tight")
        plt.close(fig)

    loc = ft.assign(row=tail["row"], col=tail["col"]).groupby(["row", "col"])["any_flag"]
    loc = loc.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(loc["col"] + 64, loc["row"] + 64, c=loc["any_flag"], s=160, marker="s",
                    cmap="Reds", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label="share of tail tiles flagged")
    ax.set_aspect("equal"); ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("Flag rate per tile location (tail tiles)")
    fig.savefig(os.path.join(fig_dir, "flag_rate_by_tile.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tail", type=float, default=31.0, help="tail = raw max >= this")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--sample_frac", type=float, default=None,
                    help="subsample each day's table (for a quick pass)")
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    df, n_days = load_features(args.features_dir, args.sample_frac)
    df = tag_split(df, cfg)
    df["stratum"] = stratum(df)
    rules = {**RULES, **INFORMATIONAL}
    flags = apply_rules(df, rules)
    tail = df[df["max"] >= args.tail]

    L = ["# Dataset quality summary", "",
         f"{len(df):,} tiles from {n_days} days, {df.time.min():%Y-%m-%d} to "
         f"{df.time.max():%Y-%m-%d}. Tail = raw tile max >= {args.tail:g} mm/h "
         f"({len(tail):,} tiles, {len(tail) / len(df):.2%}).", "",
         "**Rule thresholds are initial guesses** (see `rules.py`); read the rates as "
         "prevalence of a signature, not as an artefact count, until checked against labels.",
         "", "## Rules", "",
         md_table(pd.DataFrame({"rule": list(rules), "meaning": [v[1] for v in rules.values()]}),
                  index=False), ""]

    L += ["## Tiles per split", "", md_table(df.groupby("split").size().to_frame("tiles")), ""]
    L += ["## Content by raw-max stratum", "",
          md_table(df.groupby("stratum", observed=True).agg(
              tiles=("max", "size"), wet_frac=("wet_frac", "median"),
              largest_comp_ge1=("largest_comp_ge1", "median"), conc=("conc", "median"),
              isolated_frac_ge1=("isolated_frac_ge1", "median"))), ""]

    by_stratum = rates(flags, df["stratum"])
    by_stratum.to_csv(os.path.join(args.out_dir, "flag_rates_by_stratum.csv"))
    L += ["## Flag rate by raw-max stratum", "",
          "Share of tiles in each stratum that each rule flags. A rule that is an artefact "
          "signature should *not* rise towards the tail if the tail were clean.", "",
          md_table(by_stratum.drop(columns=[c for c in by_stratum if by_stratum[c].isna().all()]),
                   floatfmt=".3f"), ""]

    ft = flags.loc[tail.index]
    avail = [c for c in RULES if ft[c].notna().any()]
    b = ft[avail].fillna(0) > 0
    inter = b.values.T.astype(int) @ b.values.astype(int)
    size = np.diag(inter)
    jac = inter / np.clip(size[:, None] + size[None, :] - inter, 1, None)
    jac = pd.DataFrame(jac, index=avail, columns=avail)
    jac.to_csv(os.path.join(args.out_dir, "rule_overlap_tail.csv"))
    L += ["## Rule overlap in the tail (Jaccard)", "",
          "Near-zero off-diagonals mean the rules catch different things and a union is "
          "needed; high ones mean redundancy.", "", md_table(jac, floatfmt=".2f"), ""]

    for key, name in ((tail["time"].dt.year, "year"), (tail["time"].dt.month, "month"),
                      (tail["time"].dt.hour, "hour"), (tail["split"], "split")):
        r = rates(ft[avail + ["any_flag"]], key.rename(name))
        r.to_csv(os.path.join(args.out_dir, f"flag_rates_by_{name}.csv"))
        if name in ("year", "split"):
            L += [f"## Tail flag rate by {name}", "", md_table(r, floatfmt=".3f"), ""]

    loc = ft.assign(row=tail["row"], col=tail["col"]).groupby(["row", "col"])
    loc = loc["any_flag"].agg(["size", "mean"]).rename(columns={"size": "n_tail",
                                                                 "mean": "any_flag"})
    loc.sort_values("any_flag", ascending=False).to_csv(
        os.path.join(args.out_dir, "tile_flag_rate_tail.csv"))
    L += ["## Tile locations with the highest tail flag rate", "",
          md_table(loc[loc["n_tail"] >= 20].sort_values("any_flag", ascending=False).head(15),
                   floatfmt=".3f"), ""]

    dq = df.loc[tail.index]
    L += ["## Declutter step (what the current pipeline does to the tail)", "",
          f"- tail tiles with any pixel > 150 mm/h: {(dq.n_over_declutter > 0).mean():.2%}",
          f"- ... of which some of those pixels sit inside a cell (would become holes): "
          f"{(dq.n_over_declutter_in_cell > 0).sum():,} tiles",
          f"- tail tiles with raw max > 500 mm/h: {(dq['max'] > 500).mean():.2%}", ""]

    flagged = pd.concat([tail[["timestamp", "row", "col", "max", "split"]],
                         ft[avail + ["any_flag"]]], axis=1)
    flagged = flagged[flagged["any_flag"] > 0].sort_values("max", ascending=False)
    flagged.to_csv(os.path.join(args.out_dir, "flagged_tail.csv"), index=False)

    if args.labels:
        lab = pd.read_csv(args.labels, dtype={"timestamp": str}).dropna(subset=["label"])
        lab = lab[lab["label"].isin(["artefact", "real"])]
        j = lab.merge(pd.concat([df[["timestamp", "row", "col"]], flags], axis=1),
                      on=["timestamp", "row", "col"])
        y = j["label"] == "artefact"
        rows = []
        for c in list(RULES) + ["any_flag"]:
            p = j[c].fillna(0) > 0
            tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
            rows.append({"rule": c, "flagged": int(p.sum()),
                         "precision": tp / (tp + fp) if tp + fp else np.nan,
                         "recall": tp / (tp + fn) if tp + fn else np.nan})
        L += [f"## Against labels ({len(j)} labelled tiles, {int(y.sum())} artefacts)", "",
              md_table(pd.DataFrame(rows), index=False, floatfmt=".2f"), ""]

    if not args.no_figures:
        figures(df, flags, tail, os.path.join(args.out_dir, "figures"))
        L += ["## Figures", "", "`figures/feature_hist_by_stratum.png`, "
              "`figures/max_vs_wet_area.png`, `figures/flag_rate_by_{hour,month,tile}.png`", ""]

    path = os.path.join(args.out_dir, "summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
