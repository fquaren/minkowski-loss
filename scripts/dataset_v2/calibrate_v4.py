#!/usr/bin/env python
"""Phase 0 of the v4 build: does the v4 screen catch the known failures and spare real
extremes? (DECISIONS §19, EXPERIMENTS §5 "v4 screen and rebuild".)

  days    the calibration set: the known failures (Madrid 2018-04-28..30, ten Valjevo 2023
          ceiling days, the seven 48.62 mm/h days of Nov 2019), every day of every available
          event in configs/prominent_events.yaml, and 100 random days stratified by season
          (seed 20261006). Writes <calib>/days.csv and days.txt.
  report  after scan_radars.py, radar_screen_v4.py ceilings/frames and scan_tiles.py
          --quality on those days: the Phase-0 gates and the evidence for the thresholds.
          Writes <calib>/report.md and figures.

Gates: every known failure caught; event-tile rejection < 0.5% per event (wet tiles, max >=
10 mm/h); the strongest flags of each rule looked at. A failed gate stops the build.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.data_quality._md import md_table  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
RAW = f"{D}/raw/OPERA"
VALJEVO_DAYS = ["20230213", "20230214", "20230224", "20230310", "20230611", "20230622",
                "20230712", "20230722", "20230729", "20230813"]
MADRID_DAYS = ["20180428", "20180429", "20180430"]
BINS = [0, 1, 10, 31, 89, 150, 500.01]


def make_days(calib, events_path, seed=20261006):
    import yaml
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RAW, "[0-9]" * 8))
                  if os.path.exists(os.path.join(p, ".zmetadata")))
    rows = [(d, "madrid") for d in MADRID_DAYS] + [(d, "valjevo_ceiling") for d in VALJEVO_DAYS]
    f = pd.read_csv(f"{D}/quality_v3/pot_threshold/multiplicity_flags_train.csv")
    rows += [(str(d), "ceiling_4862") for d in sorted(f[f.val == 48.62].day.unique())]
    for e in yaml.safe_load(open(events_path))["events"]:
        if e.get("available", True) is False:
            continue
        for t in pd.date_range(e["start"], e["end"]):
            rows.append((t.strftime("%Y%m%d"), f"event:{e['id']}"))
    known = {d for d, _ in rows}
    rng = np.random.default_rng(seed)
    pool = pd.DataFrame({"day": [d for d in have if d not in known]})
    pool["season"] = pool.day.str[4:6].astype(int).map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    for s, g in pool.groupby("season"):
        rows += [(d, "random") for d in rng.choice(g.day.values, 25, replace=False)]
    df = pd.DataFrame(rows, columns=["day", "category"])
    df = df[df.day.isin(have)].drop_duplicates()
    os.makedirs(calib, exist_ok=True)
    df.to_csv(os.path.join(calib, "days.csv"), index=False)
    with open(os.path.join(calib, "days.txt"), "w") as fh:
        fh.write("\n".join(sorted(df.day.unique())) + "\n")
    print(df.category.str.split(":").str[0].value_counts().to_string())
    print(f"{df.day.nunique()} distinct days -> {calib}/days.txt")


def load_tiles(calib):
    fs = sorted(glob.glob(os.path.join(calib, "tiles", "*.csv.gz")))
    t = pd.concat([pd.read_csv(f, dtype={"timestamp": str}) for f in fs], ignore_index=True)
    t["day"] = t.timestamp.str[:8]
    return t


def report(calib, quality, events_path):
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_splits import event_rows
    from src.data.radar_screen import decide
    q = yaml.safe_load(open(quality))
    days = pd.read_csv(os.path.join(calib, "days.csv"), dtype={"day": str})
    cat = days.groupby("day").category.agg(lambda s: ";".join(sorted(set(s))))
    t = load_tiles(calib)
    t["category"] = t.day.map(cat).fillna("")
    ff = pd.read_csv(os.path.join(calib, "frame_flags.csv.gz"), dtype={"timestamp": str})
    flags = set(zip(ff.radar, ff.timestamp))
    dec = decide(t, q, flags)
    t = pd.concat([t, dec], axis=1)
    ev = [e for e in yaml.safe_load(open(events_path))["events"] if e.get("available", True) is not False]
    t["event"] = event_rows(t.assign(day=pd.to_datetime(t.day)), ev)
    rules = ["rej_ceiling", "rej_repeat", "rej_refused", "rej_radar_frame"]
    os.makedirs(os.path.join(calib, "fig"), exist_ok=True)
    L = [f"# v4 Phase-0 calibration\n\nQuality config: `{quality}`. Tiles: {len(t):,} on "
         f"{t.day.nunique()} days. Flagged radar-frames: {len(ff):,}.\n"]

    # ---- gate 1: known failures
    L.append("## Gate 1: known failures\n")
    kf = []
    m = (t.day == "20180429") & t.owners.fillna("").str.contains("estjv:") & (t["max"] >= 31)
    sh = t.owners.fillna("").str.extract(r"estjv:([0-9.]+)")[0].astype(float).fillna(0)
    m &= sh >= 0.10
    kf.append(("Madrid 2018-04-29: tiles >= 10% Madrid-owned, max >= 31", int(m.sum()),
               int(t.loc[m, "rej_v4"].sum())))
    ceil = pd.read_csv(os.path.join(calib, "ceilings.csv")) if os.path.exists(os.path.join(calib, "ceilings.csv")) else pd.DataFrame()
    for name, yr, val in (("Valjevo 364.63 (2023)", 2023, 364.63), ("48.62 (2019)", 2019, 48.62)):
        hit = ceil[(ceil.year == yr) & (np.abs(ceil.value - val) < 0.005)] if len(ceil) else ceil
        kf.append((f"{name} in the ceiling table", 1, int(len(hit) > 0)))
    m = t.category.str.contains("valjevo") & (np.abs(t.rep_value - 364.63) < 0.005) & (t.rep_count >= 5)
    kf.append(("Valjevo days: tiles still holding >= 5 px at 364.63 after v4 -> rejected", int(m.sum()),
               int(t.loc[m, "rej_v4"].sum())))
    m = t.category.str.contains("ceiling_4862") & (t.row == 640) & (t.col == 1408) & (t["max"] >= 31)
    kf.append(("48.62 days, tile r640 c1408, max >= 31: rejected", int(m.sum()), int(t.loc[m, "rej_v4"].sum())))
    L.append(md_table(pd.DataFrame(kf, columns=["check", "tiles", "caught"]), index=False) + "\n")
    m = t.category.str.contains("valjevo") & (t.n_ceiling > 0)
    L.append(f"Valjevo-day tiles with ceiling pixels: {int(m.sum())}; of which >= 5 (rejected): "
             f"{int((t.n_ceiling[m] >= 5).sum())}; repaired (1-4): {int((t.n_ceiling[m] < 5).sum())}.\n")

    # ---- gate 2: events
    L.append("## Gate 2: event tiles (wet: max >= 10 mm/h)\n")
    w = t[(t.event != "") & (t["max"] >= 10)]
    g = w.groupby("event").agg(tiles=("rej_v4", "size"), rejected=("rej_v4", "sum"),
                               **{r: (r, "sum") for r in rules})
    g["rejected %"] = 100 * g.rejected / g.tiles
    L.append(md_table(g.reset_index(), index=False, floatfmt=".3g") + "\n")
    worst = g["rejected %"].max() if len(g) else 0
    L.append(f"**Worst event: {worst:.2f}% rejected (gate: < 0.5%).**\n")

    # ---- rejection by intensity bin, random days
    L.append("## Rejection by tile max, random days\n")
    r = t[t.category == "random"].copy()
    r["bin"] = pd.cut(r["max"], BINS, right=False).astype(str)
    gb = r.groupby("bin").agg(tiles=("rej_v4", "size"), **{x: (x, "mean") for x in rules + ["rej_v4"]})
    L.append(md_table(gb.reset_index(), index=False, floatfmt=".3g") + "\n")

    # ---- radar-frame features: random vs failure days
    fw = os.path.join(calib, "frames_wet.csv.gz")
    if os.path.exists(fw):
        fr = pd.read_csv(fw, dtype={"timestamp": str})
        fr["day"] = fr.timestamp.str[:8]
        fr["category"] = fr.day.map(cat).fillna("")
        L.append("## Radar-frame features (wet frames: wet fraction >= 0.2)\n")
        qs = []
        for c, gg in (("random", fr[fr.category == "random"]),
                      ("event", fr[fr.category.str.startswith("event")]),
                      ("Madrid estjv 04-29", fr[(fr.day == "20180429") & (fr.radar == "estjv")])):
            for k in ("f31", "f150", "r2_range", "ejump", "bjump"):
                v = gg[k].dropna()
                if len(v):
                    qs.append({"set": c, "feature": k, "n": len(v), "q50": v.quantile(.5),
                               "q99": v.quantile(.99), "q999": v.quantile(.999), "max": v.max()})
        L.append(md_table(pd.DataFrame(qs), index=False, floatfmt=".3g") + "\n")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for c, gg, col in (("random", fr[fr.category == "random"], "0.6"),
                           ("event", fr[fr.category.str.startswith("event")], "tab:blue"),
                           ("Madrid 04-29", fr[(fr.day == "20180429") & (fr.radar == "estjv")], "tab:red")):
            ax[0].scatter(gg.f31, gg.r2_range, s=3, c=col, label=c, alpha=0.5)
            ax[1].scatter(gg.f10, np.fmax(gg.ejump, gg.bjump), s=3, c=col, label=c, alpha=0.5)
        p = q["radar_frame"]
        ax[0].axvline(p["f31_min"], c="k", lw=0.7); ax[0].axhline(p["r2_min"], c="k", lw=0.7)
        ax[1].axvline(p["f10_min"], c="k", lw=0.7); ax[1].axhline(p["jump_min"], c="k", lw=0.7)
        ax[0].set_xscale("symlog", linthresh=1e-3); ax[0].set_xlabel("f31 (share of owned pixels >= 31)")
        ax[0].set_ylabel("R2_range"); ax[1].set_xlabel("f10"); ax[1].set_ylabel("max(edge, boundary) jump")
        ax[0].legend(markerscale=4, fontsize=8)
        fig.savefig(os.path.join(calib, "fig", "radar_frames.png"), dpi=110); plt.close(fig)
        L.append("![](fig/radar_frames.png)\n")

    # ---- flagged radar-frames outside the known failures (candidate false positives)
    ff["day"] = ff.timestamp.str[:8]
    ff["category"] = ff.day.map(cat).fillna("")
    other = ff[~((ff.day == "20180429") & (ff.radar == "estjv"))]
    L.append(f"## Flagged radar-frames other than Madrid 04-29: {len(other):,} on "
             f"{other[['radar', 'day']].drop_duplicates().shape[0]} radar-days\n")
    if len(other):
        L.append(md_table(other.groupby(["radar", "day", "category"]).size().rename("frames")
                          .reset_index().sort_values("frames", ascending=False).head(30), index=False) + "\n")

    # ---- repeated values and refused repairs on random / event days
    for rule, cols in (("rej_repeat", ["rep_value", "rep_count", "rep_excess"]),
                       ("rej_refused", ["n_refused"]), ("rej_ceiling", ["n_ceiling"])):
        x = t[t[rule]]
        L.append(f"## {rule}: {len(x):,} tiles; by category\n")
        L.append(md_table(x.category.str.split(":").str[0].value_counts().rename("tiles").reset_index(),
                          index=False) + "\n")
        y = x[x.category.isin(["random"]) | x.category.str.startswith("event")]
        if len(y):
            L.append(md_table(y.sort_values(cols[-1], ascending=False).head(20)[
                ["timestamp", "row", "col", "max", "category"] + cols], index=False) + "\n")
    rp = t[t["max"] >= 31]
    L.append("## Repeated-value statistics, tiles with max >= 31\n")
    L.append(md_table(pd.DataFrame({"set": ["random", "event"],
                                    "max_rep q99": [rp[rp.category == "random"].max_rep.quantile(.99),
                                                    rp[rp.category.str.startswith("event")].max_rep.quantile(.99)],
                                    "max_rep q999": [rp[rp.category == "random"].max_rep.quantile(.999),
                                                     rp[rp.category.str.startswith("event")].max_rep.quantile(.999)],
                                    "rep_excess q999": [rp[rp.category == "random"].rep_excess.quantile(.999),
                                                        rp[rp.category.str.startswith("event")].rep_excess.quantile(.999)]}),
                      index=False, floatfmt=".3g") + "\n")
    t.to_csv(os.path.join(calib, "tiles_decided.csv.gz"), index=False)
    open(os.path.join(calib, "report.md"), "w").write("\n".join(L))
    print(f"[calib] wrote {calib}/report.md")


def gallery(calib, n=12):
    """Top flags per rule on random and event days (candidate false positives) and on the
    known-failure days (should be true positives), from the stored v4 tiles."""
    import zarr  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import yaml
    from src.data.day_cleaner import DayCleaner
    import scan_tiles as st
    t = pd.read_csv(os.path.join(calib, "tiles_decided.csv.gz"), dtype={"timestamp": str, "day": str})
    q = yaml.safe_load(open(os.path.join(calib, "quality_calib.yaml")))
    qpath = os.path.join(calib, "quality_calib.yaml")
    for rule in ("rej_repeat", "rej_refused", "rej_ceiling", "rej_radar_frame"):
        for which, sel in (("fp", t.category.eq("random") | t.category.str.startswith("event")),
                           ("tp", ~(t.category.eq("random") | t.category.str.startswith("event")))):
            x = t[t[rule] & sel].sort_values("max", ascending=False)
            x = x.drop_duplicates(["day", "row", "col"]).head(n)
            if not len(x):
                continue
            fig, ax = plt.subplots(2, (len(x) + 1) // 2, figsize=(3 * ((len(x) + 1) // 2), 6.4),
                                   constrained_layout=True, squeeze=False)
            for a in ax.ravel():
                a.axis("off")
            for a, (_, r) in zip(ax.ravel(), x.iterrows()):
                _, ceil = st._quality(qpath)
                dc = DayCleaner(RAW, r.day, "TOT_PREC", hot=lambda y: st._hot(
                    f"{D}/quality_v2/clutter_climatology.npz", y), ring=lambda y: st._ring(
                    f"{D}/quality_v2/ring_mask.npz", y), sites_rc=st._sites(),
                    ceilings=ceil, max_size=q["guard"]["max_size"])
                k = list(np.datetime_as_string(dc.times, unit="s")).index(
                    f"{r.timestamp[:4]}-{r.timestamp[4:6]}-{r.timestamp[6:8]}T{r.timestamp[8:10]}:{r.timestamp[10:12]}:{r.timestamp[12:14]}")
                f = dc.clean(k)[0][r.row:r.row + 128, r.col:r.col + 128]
                dc.close()
                a.imshow(np.where(f >= 0.1, f, np.nan), origin="lower", cmap="turbo", norm=LogNorm(0.1, 300))
                a.set_title(f"{r.timestamp[:12]} r{r.row} c{r.col}\nmax {r['max']:.0f} {r.category[:18]}", fontsize=7)
            fig.suptitle(f"{rule}: strongest flags, {'random / event days (candidate false positives)' if which == 'fp' else 'known-failure days'}")
            fig.savefig(os.path.join(calib, "fig", f"gallery_{rule}_{which}.png"), dpi=80); plt.close(fig)
            print(f"[calib] gallery {rule} {which}: {len(x)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["days", "report", "gallery"])
    ap.add_argument("--calib", default=f"{D}/quality_v4/calib")
    ap.add_argument("--quality", default=None)
    ap.add_argument("--events", default="configs/prominent_events.yaml")
    a = ap.parse_args()
    if a.cmd == "days":
        make_days(a.calib, a.events)
    elif a.cmd == "report":
        report(a.calib, a.quality or os.path.join(a.calib, "quality_calib.yaml"), a.events)
    else:
        gallery(a.calib)
