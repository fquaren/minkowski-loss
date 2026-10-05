#!/usr/bin/env python
"""Do the v3 repairs fix the tail without removing rain? v2 vs v3 on the same gauge pairs.

Input: `repair_pairs.py` output (`pairs_v3/`, which holds the v2 `cln` and the v3 `cln_r`
side by side). Gauge sanity mask, alignment and the matched gauge rate are those of
`validate_tail.py` (single best-correlated interval).

  1. Exceedance ratio rho(u) = N(radar >= u) / N(gauge >= u), v2 vs v3, per era, with a 90%
     day-block bootstrap. v2 counts only the tiles v2 keeps; v3 keeps every covered tile
     (rays and > 500 mm/h cores are repaired, not rejected).
  2. What each repair lowered: gauge rate under the repaired pixels (q50, q90), AUC against
     untouched pixels of the same v2 intensity bin (1 = only dry pixels repaired), and how
     often the gauge saw >= 10 mm/h there (rain removed).
  3. Regained tiles: pixels in tiles v2 rejected and v3 keeps, against untouched pixels.

Writes <out_dir>/repair_comparison.md and CSV tables.

    python scripts/validation/compare_repair.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_tail as vt  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
REPAIRS = {1: "footprint", 2: "ray", 4: "ring", 8: "unsupported"}


def rho(p, g, variants, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    day = pd.factorize(p["time"].dt.floor("D"))[0]
    nd = day.max() + 1
    w = np.stack([np.bincount(b, minlength=nd) for b in rng.integers(0, nd, size=(n_boot, nd))])
    rows = []
    for name, (rad, sel0) in variants.items():
        for era in ("ODYSSEY", "NIMBUS"):
            sel = sel0 & (p["era"] == era).values & np.isfinite(rad) & np.isfinite(g)
            for u in vt.U_EXC:
                nr = np.bincount(day[sel & (rad >= u)], minlength=nd)
                ng = np.bincount(day[sel & (g >= u)], minlength=nd)
                with np.errstate(divide="ignore", invalid="ignore"):
                    br = np.where(w @ ng > 0, (w @ nr) / (w @ ng), np.nan)
                rows.append({"field": name, "era": era, "u": u, "n_radar": int(nr.sum()),
                             "n_gauge": int(ng.sum()),
                             "ratio": nr.sum() / ng.sum() if ng.sum() else np.nan,
                             "q05": np.nanquantile(br, 0.05), "q95": np.nanquantile(br, 0.95)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs_dir", default=f"{D}/validation/pairs_v3")
    ap.add_argument("--tiles_dir", default=f"{D}/quality_v2/tiles")
    ap.add_argument("--gauge_qc", default=f"{D}/validation/gauges/gauge_qc.csv")
    ap.add_argument("--out_dir", default=f"{D}/validation/repair_v3")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    p = vt.load_pairs(a.pairs_dir, a.tiles_dir)
    n_qc = vt.apply_gauge_qc(p, a.gauge_qc)
    al = vt.alignment(p)
    g = vt.best_single(p, al)
    covered = p["unphysical"].notna().values                  # fully covered (in the scan table)
    kept_v2 = covered & (p["unphysical"] == 0).values & (p["ray"] == 0).values
    variants = {"v2": (p["cln"].values, kept_v2), "v3": (p["cln_r"].values, covered)}
    r = rho(p, g, variants)
    r.to_csv(os.path.join(a.out_dir, "rho_v2_v3.csv"), index=False)

    L = ["# v3 repairs against the gauges", "",
         f"{len(p):,} station-frames ({covered.sum():,} in fully covered tiles); gauge sanity "
         f"mask: {n_qc:,} intervals. Gauge = single best-correlated interval (`validate_tail.py`).", "",
         "## 1. Exceedance ratio N(radar >= u) / N(gauge >= u)", ""]
    for era in ("ODYSSEY", "NIMBUS"):
        e = r[r["era"] == era]
        tab = e.pivot(index="field", columns="u", values="ratio").round(2).astype(str)
        lo = e.pivot(index="field", columns="u", values="q05").round(2).astype(str)
        hi = e.pivot(index="field", columns="u", values="q95").round(2).astype(str)
        tab = tab + " [" + lo + ", " + hi + "]"
        n = e.pivot(index="field", columns="u", values="n_radar")
        L += [f"**{era}** (radar exceedances v2 / v3: " + ", ".join(
            f"u={u}: {int(n.loc['v2', u]):,} / {int(n.loc['v3', u]):,}" for u in n.columns) +
            f"; gauge: " + ", ".join(f"{int(e[(e.field == 'v3') & (e.u == u)].n_gauge.iloc[0]):,}"
                                     for u in n.columns) + ")", "", vt.md(tab), ""]

    # 2. per repair
    lab = [f"[{vt.BINS[i]:g}, {vt.BINS[i + 1]:g})" for i in range(len(vt.BINS) - 1)]
    b = pd.cut(p["cln"], vt.BINS, right=False, labels=lab)
    rep = p["rep"].fillna(0).astype(int).values
    lowered = (p["cln"] - p["cln_r"]).values > 1e-3
    untouched = covered & ~lowered & (rep == 0)
    rows = []
    for era in ("ODYSSEY", "NIMBUS", "all"):
        e = np.isfinite(g) & ((p["era"] == era).values if era != "all" else True)
        for bl in lab:
            eb = e & (b == bl).values
            ref = g[eb & untouched]
            for bit, name in REPAIRS.items():
                s = eb & ((rep & bit) > 0) & lowered
                if s.sum() < 5:
                    continue
                q = np.quantile(g[s], [0.5, 0.9])
                rows.append({"repair": name, "era": era, "v2 bin": bl, "n": int(s.sum()),
                             "gauge q50": round(q[0], 2), "gauge q90": round(q[1], 2),
                             "gauge >= 10": round(float((g[s] >= 10).mean()), 3),
                             "untouched q50": round(float(np.median(ref)), 2) if len(ref) else np.nan,
                             "AUC untouched > repaired": round(vt.auc(ref, g[s]), 2)})
    rv = pd.DataFrame(rows)
    rv.to_csv(os.path.join(a.out_dir, "repaired_pixels.csv"), index=False)
    L += ["## 2. What each repair lowered (gauge mm/h under the pixel, by its v2 value)", "",
          vt.md(rv[rv["era"] == "all"].drop(columns="era"), index=False), "",
          "Per era in `repaired_pixels.csv`.", ""]
    tot = lowered & np.isfinite(g)
    L += [f"All repaired pixels: {int(tot.sum()):,}; the gauge saw >= 10 mm/h at "
          f"{(g[tot] >= 10).mean():.1%} of them and >= 1 mm/h at {(g[tot] >= 1).mean():.1%} "
          f"(untouched pixels >= 10 mm/h in v2: >= 10 at "
          f"{(g[untouched & (p['cln'].values >= 10) & np.isfinite(g)] >= 10).mean():.1%}).", ""]

    # 3. regained tiles
    regained = covered & ~kept_v2
    rows = []
    for bl in lab:
        eb = np.isfinite(g) & (pd.cut(p["cln_r"], vt.BINS, right=False, labels=lab) == bl).values
        ref = g[eb & kept_v2 & untouched]
        s = eb & regained
        if s.sum() < 5:
            continue
        rows.append({"v3 bin": bl, "n regained": int(s.sum()),
                     "corroborated (>= 1)": round(float((g[s] >= 1).mean()), 2),
                     "untouched corroborated": round(float((ref >= 1).mean()), 2) if len(ref) else np.nan,
                     "gauge q50": round(float(np.median(g[s])), 2),
                     "untouched q50": round(float(np.median(ref)), 2) if len(ref) else np.nan,
                     "AUC untouched > regained": round(vt.auc(ref, g[s]), 2)})
    rg = pd.DataFrame(rows)
    rg.to_csv(os.path.join(a.out_dir, "regained_tiles.csv"), index=False)
    L += ["## 3. Pixels in tiles v2 rejected and v3 keeps", "",
          f"{int(regained.sum()):,} station-frames in {p.loc[regained, ['timestamp', 'trow', 'tcol']].drop_duplicates().shape[0]:,} regained tile-frames.", "",
          vt.md(rg, index=False) if len(rg) else "(none with >= 5 pairs in any bin)", ""]
    open(os.path.join(a.out_dir, "repair_comparison.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
