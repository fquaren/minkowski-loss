#!/usr/bin/env python
"""Is the radar tail real? Corroboration of radar intensities by independent rain gauges,
for every cleaning action and audit flag (from `gauge_vs_radar.py` pairs).

A radar pixel is **corroborated** when the gauge under it records >= GAUGE_WET mm/h in the
matched interval (`--wet`), i.e. it rained there at all. Artefacts (clutter, spikes, rings,
interference) leave the gauge dry; a real cell rarely does, even with the point-vs-2-km and
timing mismatches. A rule is informative when the pixels it flags are corroborated much less
often than the pixels it leaves alone, *in the same intensity bin*.

Steps:
1. Alignment. Correlation of log1p(radar) with log1p(gauge) for the gauge intervals
   labelled t-10, t, t+10, t+20 min, per network and product era (ODYSSEY composites a
   15-min window, NIMBUS is instantaneous). The two best-correlated adjacent intervals
   define the matched gauge rate (their max, in mm/h).
2. Corroboration by raw intensity bin, for each class: untouched, hot-pixel repaired,
   spike repaired, tile rejected (unphysical / ray, from the tile tables), no temporal
   support, on a range ring.
3. Wrongly removed rain: pixels the cleaning lowered by > 50% from >= 31 mm/h while the
   gauge saw >= 10 mm/h.
4. Underestimation: for gauge rates >= 30 mm/h, the share where the cleaned 3x3 max is
   >= 10 mm/h, and the median radar/gauge ratio.
5. QIND: AUC of QIND for corroborated vs not, among raw >= 31 mm/h, per era (QIND scales
   differ between products).

Gauge values flagged by `gauge_qc.py` (impossible, stuck, isolated bursts) are masked first.

Writes <out_dir>/validation_summary.md and the tables as CSV.

    python scripts/validation/validate_tail.py
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

D = "/home/fquareng/work/data/extremes/OPERA"
BINS = [10, 31, 89, 150, 500, np.inf]
NIMBUS_START = "2024-07-05"
GCOLS = ["g_m10", "g_0", "g_p10", "g_p20"]


def md(df, index=True):
    """Markdown table without the optional `tabulate` dependency."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def load_pairs(pdir, tiles_dir):
    parts = []
    for f in sorted(glob.glob(os.path.join(pdir, "*.csv.gz"))):
        d = pd.read_csv(f, dtype={"station": str})
        if not len(d):
            continue
        day = os.path.basename(f)[:8]
        # tile-level rejection from the scan table of that day
        tp = os.path.join(tiles_dir, f"{day}.csv.gz")
        if os.path.exists(tp):
            t = pd.read_csv(tp, usecols=["timestamp", "row", "col", "unphysical", "ray"],
                            dtype={"timestamp": str})
            d["ts14"] = pd.to_datetime(d["timestamp"]).dt.strftime("%Y%m%d%H%M%S")
            d["trow"], d["tcol"] = (d["row"] // 128) * 128, (d["col"] // 128) * 128
            d = d.merge(t.rename(columns={"timestamp": "ts14", "row": "trow", "col": "tcol"}),
                        on=["ts14", "trow", "tcol"], how="left")
        parts.append(d)
    p = pd.concat(parts, ignore_index=True)
    p["time"] = pd.to_datetime(p["timestamp"])
    p["era"] = np.where(p["time"] >= NIMBUS_START, "NIMBUS", "ODYSSEY")
    return p


def apply_gauge_qc(p, qc_path):
    """Set gauge intervals flagged by `gauge_qc.py` to NaN. Returns the number masked."""
    if not qc_path or not os.path.exists(qc_path):
        return 0
    qc = pd.read_csv(qc_path, dtype={"station": str}, usecols=["network", "station", "t"])
    qc["bad"] = True
    tmin = p["time"].values.astype("datetime64[m]").astype(np.int64)
    n = 0
    for col, off in zip(GCOLS, (-10, 0, 10, 20)):
        k = pd.DataFrame({"network": p["network"].values, "station": p["station"].values, "t": tmin + off})
        bad = k.merge(qc, on=["network", "station", "t"], how="left")["bad"].notna().values
        p.loc[bad, col] = np.nan
        n += int(bad.sum())
    return n


def alignment(p):
    rows = []
    for (net, era), g in p.groupby(["network", "era"]):
        both = g[(g["raw"] > 0.1)]
        r = {"network": net, "era": era, "n": len(both)}
        for c in GCOLS:
            x, y = np.log1p(both["cln"]), np.log1p(6 * both[c])
            ok = np.isfinite(x) & np.isfinite(y)
            r[c] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 100 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def matched_rate(p, al):
    """Gauge rate (mm/h): max of the two adjacent intervals with the best summed correlation."""
    g = np.full(len(p), np.nan)
    choice = {}
    for _, r in al.iterrows():
        cors = [r[c] for c in GCOLS]
        k = int(np.nanargmax([cors[i] + cors[i + 1] for i in range(3)]))
        choice[(r["network"], r["era"])] = (GCOLS[k], GCOLS[k + 1])
        m = ((p["network"] == r["network"]) & (p["era"] == r["era"])).values
        g[m] = 6 * np.nanmax(p.loc[m, [GCOLS[k], GCOLS[k + 1]]].values, axis=1)
    return g, choice


def classes(p):
    lowered = (p["raw"] - p["cln"]) > 1e-3
    c = {
        "untouched": (~lowered) & (p["unphysical"].fillna(0) == 0) & (p["ray"].fillna(0) == 0)
                     & (p["support"] != 1) & (p["on_ring"] == 0),
        "hot repaired": lowered & (p["hot"] == 1),
        "spike repaired": lowered & (p["hot"] == 0),
        "tile rejected: unphysical": p["unphysical"].fillna(0) == 1,
        "tile rejected: ray": p["ray"].fillna(0) == 1,
        "no temporal support": p["support"] == 1,
        "on range ring": p["on_ring"] == 1,
    }
    return c


def corroboration(p, g, wet):
    out = []
    lab = [f"[{BINS[i]:g}, {BINS[i + 1]:g})" for i in range(len(BINS) - 1)]
    b = pd.cut(p["raw"], BINS, right=False, labels=lab)
    for name, m in classes(p).items():
        for era in ("ODYSSEY", "NIMBUS", "all"):
            me = m & ((p["era"] == era) if era != "all" else True)
            for bl in lab:
                s = me & (b == bl) & np.isfinite(g)
                n = int(s.sum())
                if not n:
                    continue
                out.append({"class": name, "era": era, "raw bin": bl, "n": n,
                            "corroborated": float((g[s] >= wet).mean()),
                            "gauge >= 10": float((g[s] >= 10).mean()),
                            "median gauge (mm/h)": float(np.median(g[s]))})
    return pd.DataFrame(out)


def auc(pos, neg):
    if len(pos) < 20 or len(neg) < 20:
        return np.nan
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs_dir", default=f"{D}/validation/pairs")
    ap.add_argument("--tiles_dir", default=f"{D}/quality_v2/tiles")
    ap.add_argument("--out_dir", default=f"{D}/validation")
    ap.add_argument("--wet", type=float, default=1.0, help="gauge rate (mm/h) that counts as rain")
    ap.add_argument("--gauge_qc", default=f"{D}/validation/gauges/gauge_qc.csv",
                    help="flags from gauge_qc.py ('' to disable)")
    a = ap.parse_args()
    p = load_pairs(a.pairs_dir, a.tiles_dir)
    n_qc = apply_gauge_qc(p, a.gauge_qc)
    al = alignment(p)
    g, choice = matched_rate(p, al)
    cor = corroboration(p, g, a.wet)
    al.to_csv(os.path.join(a.out_dir, "alignment.csv"), index=False)
    cor.to_csv(os.path.join(a.out_dir, "corroboration.csv"), index=False)

    L = ["# Radar tail vs rain gauges", "",
         f"{len(p):,} station-frames, {p['station'].nunique()} gauges "
         f"({', '.join(f'{k}: {v}' for k, v in p.groupby('network')['station'].nunique().items())}), "
         f"{p['time'].dt.date.nunique()} days. Corroborated = gauge >= {a.wet:g} mm/h in the "
         f"matched interval. Gauge sanity check (`gauge_qc.py`): {n_qc:,} gauge intervals in "
         "these pairs masked.", "", "## 1. Alignment (correlation of log rates)", "",
         md(al.round(3), index=False), "",
         "Matched intervals: " + "; ".join(f"{k[0]}/{k[1]}: {v[0]}+{v[1]}" for k, v in choice.items()),
         "", "## 2. Corroboration by class and raw intensity", ""]
    piv = cor[cor["era"] == "all"].pivot(index="class", columns="raw bin", values="corroborated")
    cnt = cor[cor["era"] == "all"].pivot(index="class", columns="raw bin", values="n")
    order = [c for c in (f"[{BINS[i]:g}, {BINS[i + 1]:g})" for i in range(len(BINS) - 1)) if c in piv]
    piv, cnt = piv[order], cnt[order]
    tab = piv.round(2).astype(str) + " (" + cnt.fillna(0).astype(int).astype(str) + ")"
    L += ["Share corroborated (n):", "", md(tab), ""]
    for era in ("ODYSSEY", "NIMBUS"):
        e = cor[(cor["era"] == era) & (cor["class"] == "untouched")]
        if len(e):
            L += [f"{era}, untouched: " + ", ".join(f"{r['raw bin']} {r['corroborated']:.2f} (n={r['n']})"
                                                     for _, r in e.iterrows()), ""]
    # 3. wrongly removed rain
    low = (p["raw"] >= 31) & (p["cln"] < 0.5 * p["raw"])
    wr = low & (g >= 10)
    L += ["## 3. Rain removed by the cleaning", "",
          f"Pixels lowered by > 50% from >= 31 mm/h: {int(low.sum()):,}; of these the gauge saw "
          f">= 10 mm/h at {int(wr.sum()):,} ({wr.sum() / max(low.sum(), 1):.1%}).", ""]
    # 4. underestimation
    hg = g >= 30
    if hg.sum():
        ratio = (p.loc[hg, "cln3"] / g[hg]).replace([np.inf, -np.inf], np.nan)
        L += ["## 4. Heavy gauge rain seen by the radar", "",
              f"Gauge >= 30 mm/h: {int(hg.sum()):,} station-frames; cleaned 3x3 max >= 10 mm/h at "
              f"{(p.loc[hg, 'cln3'] >= 10).mean():.1%}; median radar(3x3)/gauge {ratio.median():.2f}.", ""]
    # 5. QIND
    L += ["## 5. QIND as a discriminator (raw >= 31 mm/h)", ""]
    for era in ("ODYSSEY", "NIMBUS"):
        m = (p["era"] == era) & (p["raw"] >= 31) & np.isfinite(p["q"]) & np.isfinite(g)
        if m.sum():
            pos, neg = p.loc[m & (g >= a.wet), "q"].values, p.loc[m & (g < a.wet), "q"].values
            L.append(f"- {era}: AUC {auc(pos, neg):.2f} (corroborated {len(pos):,}, not {len(neg):,}; "
                     f"median QIND {np.median(pos) if len(pos) else np.nan:.2f} vs "
                     f"{np.median(neg) if len(neg) else np.nan:.2f})")
    # per year, untouched tail
    m = classes(p)["untouched"] & (p["raw"] >= 31) & np.isfinite(g)
    yr = pd.DataFrame({"year": p.loc[m, "time"].dt.year, "c": g[m] >= a.wet}).groupby("year")["c"].agg(["mean", "size"])
    L += ["", "## 6. Untouched tail (raw >= 31) by year", "", md(yr.round(2)), ""]
    open(os.path.join(a.out_dir, "validation_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
