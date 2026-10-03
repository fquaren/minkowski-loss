#!/usr/bin/env python
"""Which radars are consistently bad? Attribute artefact signatures to OPERA radar sites.

The composite has no per-pixel "source radar" field, so each pixel is attributed to the
**nearest radar active that year** within `--max_km`. That is an approximation (the composite
picks radars by quality or elevation, usually but not always the nearest), good enough to find
sites whose surroundings are persistently dirty.

Two independent signals, per radar and per year:

  clutter    from `clutter_climatology.npz` (per-pixel counts per year):
             - hot pixels: rain >= 31 mm/h in more than `--hot_freq` of valid time steps.
               Real rain at 31 mm/h occurs in ~1e-4 of steps, so 1e-2 is 100x climatology.
             - pixels that ever exceeded 500 mm/h.
  tail       from the per-tile feature tables: tail tiles (raw max >= `--tail`) whose peak
             pixel is attributed to the radar, and the share of them flagged by the
             artefact rules (`rules.py`), excluding `sea_clutter` while its DEM features are
             being recomputed (see `src/data/geo.py`).

"Consistently bad" = in the worst `--worst_q` of radars on the **same** signal in at least
`--min_years` of the well-sampled years (those with >= `--min_days` audited days; a year with
a handful of days is too small to rank on).

    python scripts/data_quality/radar_attribution.py config.yaml \
        --quality_dir /home/fquareng/work/data/extremes/OPERA/quality \
        --out_dir /home/fquareng/work/data/extremes/OPERA/quality/radars
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
from src.data import geo  # noqa: E402
from rules import RULES, apply_rules  # noqa: E402

FEATURE_COLS = ["timestamp", "row", "col", "max", "argmax_y", "argmax_x", "coverage",
                "n_spikes", "spike_mass_frac", "isolated_frac_ge1", "n_ge1", "peak_block_ratio",
                "persist_prev", "persist_next", "argmax_over_sea", "peak_block_mean",
                "spoke_len", "clim_freq_ge31_at_max", "q_at_max", "n_over_declutter",
                "n_over_declutter_in_cell"]


def _year(v):
    try:
        return int(str(v).strip()[:4])
    except (TypeError, ValueError):
        return None


def active_sites(sites, year):
    """Sites plausibly operating in `year` (start <= year <= finish, blanks = open)."""
    s0 = sites["startyear"].map(_year)
    s1 = sites["finishyear"].map(_year) if "finishyear" in sites else pd.Series(None, index=sites.index)
    ok = (s0.isna() | (s0 <= year)) & (s1.isna() | (s1 >= year))
    on_grid = sites["row"].between(-200, geo.H + 200) & sites["col"].between(-200, geo.W + 200)
    return sites[ok & on_grid]


def pixel_owner(sites, max_km):
    """(H, W) array: index into `sites` of the nearest radar within max_km, -1 elsewhere."""
    rr, cc = np.mgrid[0:geo.H, 0:geo.W]
    x, y = geo.rowcol_to_xy(rr, cc)
    idx, _ = geo.nearest_radar(x, y, sites, max_km=max_km)
    return idx


def clutter_stats(clim, year, owner, sites, hot_freq, min_valid):
    key = f"y{year}_"
    nv = clim[key + "n_valid"].astype(float)
    n31 = clim[key + "n_ge31"].astype(float)
    mx = clim[key + "max"]
    ok = nv >= min_valid
    freq = np.where(ok, n31 / np.maximum(nv, 1), 0.0)
    hot = ok & (freq > hot_freq)
    ge500 = ok & (mx >= 500)
    rows = []
    for k in range(len(sites)):
        m = owner == k
        area = int((m & ok).sum())
        if area == 0:
            continue
        rows.append({"site": k, "year": year, "valid_px": area,
                     "hot_px": int((m & hot).sum()),
                     "ge500_px": int((m & ge500).sum()),
                     "hot_per_10k_px": 1e4 * (m & hot).sum() / area})
    return pd.DataFrame(rows)


def tail_stats(df, flags, year, owner, sites):
    d = df[df["year"] == year]
    if d.empty:
        return pd.DataFrame()
    pr = (d["row"] + d["argmax_y"]).clip(0, geo.H - 1).astype(int).values
    pc = (d["col"] + d["argmax_x"]).clip(0, geo.W - 1).astype(int).values
    who = owner[pr, pc]
    f = flags.loc[d.index]
    t = pd.DataFrame({"site": who, "any": f["any_flag_nosea"].values,
                      "unphysical": f["unphysical"].values, "static": f["static_clutter"].values,
                      "spike": f["spike"].values, "spoke": f["spoke"].values,
                      "ge500": (d["max"] >= 500).values})
    t = t[t["site"] >= 0]
    g = t.groupby("site")
    out = pd.DataFrame({
        "n_tail": g.size(),
        "tail_flag_rate": g["any"].mean(),
        "n_unphysical": g["unphysical"].sum(),
        "static_rate": g["static"].mean(),
        "spike_rate": g["spike"].mean(),
        "spoke_rate": g["spoke"].mean(),
        "n_tail_ge500": g["ge500"].sum(),
    }).reset_index()
    out["year"] = year
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--quality_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--radar_db", default=None)
    ap.add_argument("--max_km", type=float, default=250.0)
    ap.add_argument("--tail", type=float, default=31.0)
    ap.add_argument("--hot_freq", type=float, default=0.01)
    ap.add_argument("--min_valid", type=int, default=200)
    ap.add_argument("--min_tail", type=int, default=30,
                    help="radars with fewer tail tiles in a year are not ranked on the tail")
    ap.add_argument("--worst_q", type=float, default=0.10)
    ap.add_argument("--min_years", type=int, default=3)
    ap.add_argument("--min_days", type=int, default=30,
                    help="only years with at least this many audited days are ranked")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sites_all = geo.load_radar_sites(args.radar_db)
    clim = np.load(os.path.join(args.quality_dir, "clutter_climatology.npz"))
    years = sorted({int(k[1:5]) for k in clim.files if k.startswith("y") and k[1:5].isdigit()})
    print(f"[radars] {len(sites_all)} sites in the database; climatology years {years}")

    files = sorted(glob.glob(os.path.join(args.quality_dir, "features", "*.csv.gz")))
    days_per_year = pd.Series([os.path.basename(f)[:4] for f in files]).astype(int).value_counts()
    ranked = sorted(y for y in years if days_per_year.get(y, 0) >= args.min_days)
    print(f"[radars] audited days per year: {dict(days_per_year.sort_index())}; "
          f"ranking on {ranked}")
    years = ranked
    parts = []
    for f in files:
        d = pd.read_csv(f, dtype={"timestamp": str},
                        usecols=lambda c: c in FEATURE_COLS)
        parts.append(d[d["max"] >= args.tail])
    df = pd.concat(parts, ignore_index=True)
    df["year"] = df["timestamp"].str[:4].astype(int)
    flags = apply_rules(df)
    flags["any_flag_nosea"] = (flags[[k for k in RULES if k != "sea_clutter"]]
                               .fillna(0) > 0).any(axis=1).astype(float)
    print(f"[radars] {len(df):,} tail tiles (max >= {args.tail:g}) from {len(files)} days")

    clut, tail, site_rows = [], [], []
    for y in years:
        s = active_sites(sites_all, y).reset_index(drop=True)
        owner = pixel_owner(s, args.max_km)
        c = clutter_stats(clim, y, owner, s, args.hot_freq, args.min_valid)
        t = tail_stats(df, flags, y, owner, s)
        for tab in (c, t):
            if not tab.empty:
                tab["odim"] = s.loc[tab["site"], "odim"].values
                tab["location"] = s.loc[tab["site"], "location"].values
                tab["country"] = s.loc[tab["site"], "country"].values
                tab["band"] = s.loc[tab["site"], "band"].values
        clut.append(c); tail.append(t)
        print(f"  {y}: {len(s)} active sites; {int(c['hot_px'].sum()) if len(c) else 0} hot px;"
              f" {int(t['n_tail'].sum()) if len(t) else 0} tail tiles attributed")
    clut = pd.concat(clut, ignore_index=True)
    tail = pd.concat([t for t in tail if len(t)], ignore_index=True)
    key = ["odim", "location", "country", "band", "year"]
    per = clut.drop(columns="site").merge(tail.drop(columns="site"), on=key, how="outer")

    # rank within each year; a radar is "worst" on a signal if it is in the top worst_q
    def worst(col, cond=None):
        z = per.copy()
        if cond is not None:
            z.loc[~cond(z), col] = np.nan
        thr = z.groupby("year")[col].transform(lambda v: v.quantile(1 - args.worst_q))
        return (z[col] >= thr) & z[col].notna() & (z[col] > 0)
    per["worst_hot"] = worst("hot_per_10k_px")
    per["worst_tail"] = worst("tail_flag_rate", lambda z: z["n_tail"] >= args.min_tail)
    per["worst_ge500"] = worst("ge500_px")

    agg = per.groupby(["odim", "location", "country", "band"]).agg(
        years=("year", "nunique"),
        years_worst_hot=("worst_hot", "sum"),
        years_worst_tail=("worst_tail", "sum"),
        years_worst_ge500=("worst_ge500", "sum"),
        hot_px_mean=("hot_px", "mean"),
        hot_per_10k_px_mean=("hot_per_10k_px", "mean"),
        ge500_px_mean=("ge500_px", "mean"),
        n_tail=("n_tail", "sum"),
        tail_flag_rate_mean=("tail_flag_rate", "mean"),
        n_unphysical=("n_unphysical", "sum"),
    ).reset_index()
    agg["years_worst_any"] = per.assign(w=per[["worst_hot", "worst_tail", "worst_ge500"]]
                                        .any(axis=1)).groupby(
        ["odim", "location", "country", "band"])["w"].sum().values
    sig = ["years_worst_hot", "years_worst_tail", "years_worst_ge500"]
    agg["years_worst_same_signal"] = agg[sig].max(axis=1)
    agg["bad_on"] = agg[sig].apply(
        lambda r: ",".join(k.replace("years_worst_", "") for k in sig
                           if r[k] >= args.min_years), axis=1)
    agg["consistently_bad"] = agg["years_worst_same_signal"] >= args.min_years
    agg = agg.sort_values(["consistently_bad", "years_worst_same_signal",
                           "years_worst_any", "ge500_px_mean"], ascending=False)

    per.to_csv(os.path.join(args.out_dir, "radar_by_year.csv"), index=False)
    agg.to_csv(os.path.join(args.out_dir, "radar_summary.csv"), index=False)

    bad = agg[agg["consistently_bad"]]
    lines = [f"# Radar attribution ({len(files)} days, years {years})", "",
             f"Nearest active radar within {args.max_km:g} km. Hot pixel = >= 31 mm/h in "
             f"> {args.hot_freq:g} of valid steps. Tail = raw max >= {args.tail:g} mm/h. "
             f"'Worst' = top {args.worst_q:.0%} of radars in that year. Consistently bad = "
             f"worst on the same signal in >= {args.min_years} of the ranked years {years} "
             f"(>= {args.min_days} audited days each).", "",
             f"**{len(bad)} of {len(agg)} radars are consistently bad.**", ""]
    cols = ["odim", "location", "country", "band", "years", "bad_on", "years_worst_hot",
            "years_worst_tail", "years_worst_ge500", "hot_px_mean", "ge500_px_mean", "n_tail",
            "tail_flag_rate_mean", "n_unphysical"]
    show = bad[cols].head(40).copy()
    for c in ("hot_px_mean", "ge500_px_mean", "tail_flag_rate_mean"):
        show[c] = show[c].round(2)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in show.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    open(os.path.join(args.out_dir, "radar_summary.md"), "w").write("\n".join(lines) + "\n")

    # map: all-years hot-pixel frequency, with the consistently bad radars labelled
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    nv = clim["n_valid"].astype(float)
    fr = np.where(nv >= args.min_valid, clim["n_ge31"] / np.maximum(nv, 1), np.nan)
    fig, ax = plt.subplots(figsize=(10, 11))
    im = ax.imshow(np.where(fr > 0, fr, np.nan), origin="lower", cmap="magma_r",
                   norm=LogNorm(1e-5, 0.3), interpolation="nearest")
    cur = sites_all[sites_all["source"] == "current"]
    ax.plot(cur["col"], cur["row"], "+", ms=4, color="tab:blue", label="OPERA radar")
    for _, r in bad.head(25).iterrows():
        # match on code, name and country: several sites have an empty ODIM code
        hit = sites_all[(sites_all["odim"].fillna("") == (r["odim"] if isinstance(r["odim"], str) else ""))
                        & (sites_all["location"] == r["location"])
                        & (sites_all["country"] == r["country"])]
        if hit.empty:
            continue
        s = hit.iloc[0]
        ax.plot(s["col"], s["row"], "o", mfc="none", mec="red", ms=9)
        name = r["odim"] if isinstance(r["odim"], str) and r["odim"] else r["location"]
        ax.annotate(name, (s["col"], s["row"]), fontsize=7,
                    color="red", xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0, geo.W); ax.set_ylim(0, geo.H)
    ax.set_title("Frequency of >= 31 mm/h per pixel (all audited days); "
                 "red = consistently bad radars", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, label="fraction of valid 15-min steps >= 31 mm/h")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(os.path.join(args.out_dir, "radar_map.png"), dpi=130, bbox_inches="tight")
    print(f"[radars] {len(bad)} consistently bad of {len(agg)}; wrote {args.out_dir}")


if __name__ == "__main__":
    main()
