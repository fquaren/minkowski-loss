#!/usr/bin/env python
"""Figures for notes/data_quality_assessment.tex.

Reads the audit outputs (default /home/fquareng/work/data/extremes/OPERA/quality) and the
raw day stores, and writes PNGs to notes/figures/data_quality/. Synthetic examples use the
same feature code as the audit (src/data/quality.py), so the values printed on them are the
values the audit would compute.

    python notes/figures/make_data_quality_figures.py [--quality_dir ...]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_quality"))
import src  # noqa: E402,F401  (node limits)
from src.data.quality import tile_features, pipeline_filter, adaptive_block_means  # noqa: E402
from rules import apply_rules  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle, Ellipse  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
RAIN = plt.get_cmap("turbo").copy()
RAIN.set_bad("white")
NORM = LogNorm(vmin=0.1, vmax=200)
OUT = os.path.join(HERE, "data_quality")
RULE_ORDER = ["spike", "speckle", "isolated_peak", "flicker", "sea_clutter", "spoke",
              "static_clutter", "qind_low", "declutter_isolated", "unphysical"]


def show(ax, a, title=None):
    im = ax.imshow(np.where(np.asarray(a) >= 0.1, a, np.nan), origin="lower", cmap=RAIN,
                   norm=NORM, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return im


# ------------------------------------------------------------------------ synthetic
def blob(cy, cx, peak, sy, sx=None, N=128):
    yy, xx = np.indices((N, N))
    sx = sy if sx is None else sx
    return peak * np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2)))


def fig_signatures():
    rng = np.random.default_rng(3)
    N = 128
    noise = lambda: np.exp(0.25 * rng.standard_normal((N, N)))  # noqa: E731
    base = blob(60, 50, 45, 9, 14) + blob(85, 95, 20, 6) + 0.6 * blob(40, 90, 3, 25)
    storm = base * noise()
    moved = np.roll(base, 4, axis=1) * noise()                  # 8 km of advection
    spike = 0.4 * blob(50, 50, 1.0, 18) * noise()
    spike[90, 95] = 85.0
    speck = np.where(rng.random((N, N)) < 0.03, rng.gamma(1.2, 3.0, (N, N)), 0.0)

    def line(a, y0, x0, y1, x1, v=4.0):
        for t in np.linspace(0, 1, 600):
            a[int(round(y0 + (y1 - y0) * t)), int(round(x0 + (x1 - x0) * t))] = v
        return a
    spoke = line(0.8 * blob(30, 30, 3, 10) * noise(), 50, 20, 125, 120)
    merged = line(blob(40, 40, 8, 14) * noise(), 30, 35, 125, 120)
    flick = 0.4 * blob(50, 50, 1.0, 18) * noise()
    flick[88:93, 93:98] = 40.0 * noise()[88:93, 93:98]
    empty = np.zeros((N, N))
    hole = blob(64, 64, 230, 7) * np.exp(0.1 * rng.standard_normal((N, N)))

    cases = [
        ("(a) convective storm", storm, dict(prev=moved, nxt=moved)),
        ("(b) isolated spike", spike, dict(prev=spike, nxt=spike)),
        ("(c) speckle", speck, {}),
        ("(d) spoke, detached", spoke, {}),
        ("(e) spoke, merged", merged, {}),
        ("(f) one-frame flicker", flick, dict(prev=empty, nxt=empty)),
        ("(g) >150 mm/h core", hole, {}),
    ]
    keys = [("isolated_frac_ge1", "iso"), ("peak_block_ratio", "PBR"),
            ("spoke_len", "spoke"), ("persist_prev", "pers"),
            ("n_over_declutter_in_cell", "in-cell")]
    fig, axes = plt.subplots(1, len(cases), figsize=(15, 3.3))
    for ax, (name, a, kw) in zip(axes, cases):
        f = tile_features(a, **kw)
        im = show(ax, a, name)
        if name.startswith("(g)"):
            ax.contour(a, levels=[150], colors="k", linewidths=0.8)
        lab = []
        for k, short in keys:
            v = f.get(k, np.nan)
            lab.append(f"{short} = {v:.3g}" if v is not None and np.isfinite(v) else f"{short} = n/a")
        ax.set_xlabel("\n".join(lab), family="monospace", fontsize=7, loc="left")
    cb = fig.colorbar(im, ax=list(axes), shrink=0.75, pad=0.01)
    cb.set_label("mm/h")
    fig.savefig(os.path.join(OUT, "fig_signatures.png"))
    plt.close(fig)


def fig_pooling():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    n, out = 128, 10
    for i in range(out):
        a, b = int(np.floor(i * n / out)), int(np.ceil((i + 1) * n / out))
        ax.add_patch(Rectangle((a, i % 2 * 0.6), b - a, 0.5, fc=plt.cm.tab10(i % 10), alpha=0.6))
        ax.text((a + b) / 2, i % 2 * 0.6 + 0.25, f"{b - a}", ha="center", va="center",
                fontsize=7)
    ax.set_xlim(-2, 130); ax.set_ylim(-0.2, 1.3); ax.set_yticks([])
    ax.set_xlabel("pixel index along one axis (128 px = 256 km)")
    ax.set_title(r"adaptive_avg_pool2d bins, 128 $\to$ 10: widths 13 or 14 px, overlapping")
    rng = np.random.default_rng(1)
    t = blob(70, 40, 30, 10) + blob(30, 95, 6, 15)
    t[95, 100] = 90
    t *= np.exp(0.2 * rng.standard_normal(t.shape))
    ax = axes[1]
    show(ax, t, "peak_block_ratio: peak / its coarse block mean")
    f = pipeline_filter(t)
    c = adaptive_block_means(f, 10)
    by, bx = 7, 7
    y0, y1 = int(np.floor(by * 12.8)), int(np.ceil((by + 1) * 12.8))
    x0, x1 = int(np.floor(bx * 12.8)), int(np.ceil((bx + 1) * 12.8))
    ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, ec="k", lw=1))
    ax.text(x0, y1 + 2, f"block mean {c[by, bx]:.2f} mm/h, peak 90 mm/h\n"
                        f"ratio = 90/({c[by, bx]:.2f}+0.1) = {90 / (c[by, bx] + 0.1):.0f}",
            fontsize=7, va="bottom")
    fig.savefig(os.path.join(OUT, "fig_pooling.png"))
    plt.close(fig)


# ------------------------------------------------------------------------ real data
def load_features(qdir):
    files = sorted(glob.glob(os.path.join(qdir, "features", "*.csv.gz")))
    df = pd.concat([pd.read_csv(f, dtype={"timestamp": str}) for f in files], ignore_index=True)
    df["year"] = df["timestamp"].str[:4].astype(int)
    df["hour"] = df["timestamp"].str[8:10].astype(int)
    df["source"] = np.where(df["has_qind"] == 1, "fetched archive", "original 2023-24 stores")
    return df


def fig_domain(qdir, df):
    clim = np.load(os.path.join(qdir, "clutter_climatology.npz"))
    nv = clim["n_valid"].astype(float)
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    im = ax.imshow(np.where(nv > 0, nv / nv.max(), np.nan), origin="lower", cmap="Greys",
                   vmin=0, vmax=1)
    orig = df[df["source"].str.startswith("original")][["row", "col"]].drop_duplicates()
    for r, c in orig.itertuples(index=False):
        ax.add_patch(Rectangle((c, r), 128, 128, fill=False, ec="tab:red", lw=0.7))
    ax.set_title(f"OPERA 2 km LAEA grid (2200 x 1900). Red: the {len(orig)} tile\n"
                 "locations fully covered in the 2023-24 stores")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of time steps with valid data")
    fig.savefig(os.path.join(OUT, "fig_domain.png"))
    plt.close(fig)


def fig_climatology(qdir):
    clim = np.load(os.path.join(qdir, "clutter_climatology.npz"))
    f = clim["freq_ge31"]
    from scipy import ndimage
    ex = f - ndimage.median_filter(f, size=15)
    hs = pd.read_csv(os.path.join(qdir, "clutter_hotspots.csv"))
    zooms = [(int(hs.y[0]), int(hs.x[0]), 90, "(b) top hot spot"),
             (1230, 1000, 220, "(c) range rings"),
             (520, 1420, 150, "(d) spoke fan")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    im = axes[0].imshow(np.where(f > 0, f, np.nan), origin="lower", cmap="magma",
                        norm=LogNorm(1e-5, 0.3))
    axes[0].set_title("(a) $\\hat p_{31}$ over the domain")
    fig.colorbar(im, ax=axes[0], shrink=0.7, pad=0.02)
    for ax, (cy, cx, w, t) in zip(axes[1:], zooms):
        sl = (slice(max(cy - w, 0), cy + w), slice(max(cx - w, 0), cx + w))
        im2 = ax.imshow(np.where(ex[sl] > 0, ex[sl], np.nan), origin="lower", cmap="viridis",
                        norm=LogNorm(1e-5, 0.3),
                        extent=(sl[1].start, sl[1].stop, sl[0].start, sl[0].stop))
        ax.set_title(t)
        axes[0].add_patch(Rectangle((sl[1].start, sl[0].start), sl[1].stop - sl[1].start,
                                    sl[0].stop - sl[0].start, fill=False, ec="c", lw=1))
        axes[0].text(sl[1].start, sl[0].stop + 15, t[:3], color="c", fontsize=7)
    cb = fig.colorbar(im2, ax=list(axes[1:]), shrink=0.7, pad=0.01)
    cb.set_label("local excess $\\hat p_{31} - \\mathrm{med}_{15}(\\hat p_{31})$")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(OUT, "fig_climatology.png"))
    plt.close(fig)


def fig_value_range(df):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.logspace(-1, 5.3, 120)
    for s, c in (("original 2023-24 stores", "tab:blue"), ("fetched archive", "tab:orange")):
        v = df.loc[(df["source"] == s) & (df["max"] > 0.1), "max"]
        ax.hist(v, bins=bins, histtype="step", color=c, density=True,
                label=f"{s} ({len(v):,} tiles)")
    ax.axvline(150, ls=":", c="k")
    ax.text(160, ax.get_ylim()[1] * 0.3 if ax.get_ylim()[1] > 0 else 1, "150 mm/h", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("raw tile maximum (mm/h)"); ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("The original stores stop at 150 mm/h; the archive does not")
    fig.savefig(os.path.join(OUT, "fig_value_range.png"))
    plt.close(fig)


def fig_features(df):
    wet = df[df["max"] >= 1]
    groups = [("1-10", (wet["max"] < 10)), ("10-31", (wet["max"] >= 10) & (wet["max"] < 31)),
              ("31-89", (wet["max"] >= 31) & (wet["max"] < 89)), (r"$\geq$89", wet["max"] >= 89)]
    feats = [("isolated_frac_ge1", "isolated fraction (>= 1 mm/h)", np.logspace(-4, 0, 60)),
             ("peak_block_ratio", "peak / coarse-block mean", np.logspace(0, 4, 60)),
             ("largest_comp_ge1", "largest component >= 1 mm/h (px)", np.logspace(0, 4.2, 60)),
             ("persist_prev", "persistence into t-15", np.logspace(-3, 1, 60))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    for ax, (k, lab, bins) in zip(axes, feats):
        for name, m in groups:
            v = wet.loc[m, k].dropna()
            ax.hist(v[v > 0], bins=bins, histtype="step", density=True, label=f"max {name}")
        ax.set_xscale("log"); ax.set_title(lab)
    axes[0].legend(fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_features.png"))
    plt.close(fig)


def fig_rates(qdir, df, flags):
    st = pd.read_csv(os.path.join(qdir, "summary", "flag_rates_by_stratum.csv"), index_col=0)
    st = st[st["n"] > 0].iloc[2:]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    ax = axes[0]
    x = np.arange(len(st))
    for r in RULE_ORDER:
        if r in st and st[r].notna().any():
            ax.plot(x, st[r], marker=".", lw=1, label=r)
    ax.plot(x, st["any_flag"], marker="o", lw=2.4, c="k", label="any flag")
    ax.set_xticks(x); ax.set_xticklabels(st.index, rotation=35, fontsize=7)
    ax.set_xlabel("raw tile max stratum (mm/h)"); ax.set_ylabel("share flagged")
    ax.set_title("(a) flag rate rises with intensity")
    ax.legend(fontsize=6, ncol=2)
    tail = df["max"] >= 31
    y = flags.loc[tail].groupby(df.loc[tail, "year"])["any_flag"].mean()
    axes[1].bar(y.index.astype(str), y.values, color="tab:red")
    axes[1].set_title("(b) tail tiles flagged, by year"); axes[1].set_ylim(0, 1)
    h = flags.loc[tail].groupby(df.loc[tail, "hour"])[["any_flag", "isolated_peak",
                                                        "flicker", "qind_low"]].mean()
    for c in h:
        axes[2].plot(h.index, h[c], marker=".", lw=2.2 if c == "any_flag" else 1, label=c)
    axes[2].set_xlabel("hour (UTC)"); axes[2].set_title("(c) diurnal cycle of the tail flag rate")
    axes[2].legend(fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_rates.png"))
    plt.close(fig)


HOLE_TILES = [("20230707151500", 512, 512, "a real convective core"),
              ("20130814163000", 640, 1280, "an extended sector artefact (also 'in-cell')")]


def fig_hole(raw_dir):
    import xarray as xr
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for row, (ts, r0, c0, what) in zip(axes, HOLE_TILES):
        ds = xr.open_zarr(os.path.join(raw_dir, ts[:8]))
        t = np.datetime64(pd.to_datetime(ts, format="%Y%m%d%H%M%S"))
        i = int(np.nonzero(ds.time.values == t)[0][0])
        a = ds["TOT_PREC"].isel(time=i, y=slice(r0, r0 + 128), x=slice(c0, c0 + 128)).values
        f = pipeline_filter(a)
        iy, ix = np.unravel_index(np.nanargmax(a), a.shape)
        show(row[0], a, f"raw: {what}\nmax {np.nanmax(a):.0f} mm/h, {ts} ({r0},{c0})")
        show(row[1], f, "after filter_precip_bounds (> 150 -> 0)")
        for ax in row[:2]:
            ax.axhline(iy, c="k", lw=0.5, ls=":")
        xs = np.arange(128)
        row[2].plot(xs, np.nan_to_num(a[iy]), label="raw", c="tab:blue")
        row[2].plot(xs, f[iy], label="filtered", c="tab:red", ls="--")
        row[2].axhline(150, c="k", ls=":", lw=0.8)
        row[2].set_xlim(max(ix - 25, 0), min(ix + 25, 127))
        row[2].set_ylabel("mm/h"); row[2].legend(fontsize=7)
        row[2].set_title("profile along the dotted row")
    axes[1, 2].set_xlabel("x (px)")
    fig.savefig(os.path.join(OUT, "fig_hole.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality_dir", default="/home/fquareng/work/data/extremes/OPERA/quality")
    ap.add_argument("--raw_dir", default="/home/fquareng/work/data/extremes/OPERA/raw/OPERA")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fig_signatures(); print("signatures")
    fig_pooling(); print("pooling")
    df = load_features(args.quality_dir); print("features", len(df))
    flags = apply_rules(df)
    df = df.join(flags[["unphysical"]]) if "unphysical" in flags else df
    fig_domain(args.quality_dir, df); print("domain")
    fig_climatology(args.quality_dir); print("climatology")
    fig_value_range(df); print("value range")
    fig_features(df); print("features fig")
    fig_rates(args.quality_dir, df, flags); print("rates")
    fig_hole(args.raw_dir); print("hole")


if __name__ == "__main__":
    main()
