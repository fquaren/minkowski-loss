#!/usr/bin/env python
"""Per-rule review gallery of the v4 screen on the Phase-0 calibration days (DECISIONS §17
safeguard 3: look before trusting). For every rule, >= 10 examples: the strongest flags and a
random draw among all flags, so typical cases and false positives are seen, not only the
clearest artefacts (2026-10-07: the strongest RLAN-ray flag shown to MeteoSwiss was a front).

  census   re-clean a sample of frames (all candidate tiles in them) and count, per tile, the
           pixels each rule changed: hot (static clutter), spike, footprint, ray, ring,
           unsupported, ceiling, refused. Writes <out>/census.csv.gz.
  render   per rule, 4 strongest + 8 random examples (<= 2 per day), each as t-15 | t with the
           rule's pixels outlined | t after the screen | t+15 (raw, no-data grey). Tile rules
           (repeated value, refused, ceiling >= 5 px) and radar rules (radar-disk frames,
           iso31 radar-days) from the Phase-0 tables. Writes <out>/rule_gallery.pdf and
           <out>/index.csv (one row per example, with an empty `verdict` column).

    python scripts/data_quality/rule_gallery.py census --workers 2
    python scripts/data_quality/rule_gallery.py render
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]

D = "/home/fquareng/work/data/extremes/OPERA"
CALIB = f"{D}/quality_v4/calib"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
QPATH = f"{CALIB}/quality_calib.yaml"
FRAME_RULES = ["hot", "spike", "footprint", "ray", "ring", "unsupported"]
CODE_BIT = {"footprint": 1, "ray": 2, "ring": 4, "unsupported": 8, "ceiling": 16, "refused": 32}


def stamps_of(dc):
    return [np.datetime_as_string(x, unit="s").replace("-", "").replace("T", "").replace(":", "")
            for x in dc.times]


def make_cleaner(day, v4=True):
    import yaml
    import scan_tiles as st
    from src.data.day_cleaner import DayCleaner
    q = yaml.safe_load(open(QPATH))
    _, ceil = st._quality(QPATH)
    return DayCleaner(f"{D}/raw/OPERA", day, "TOT_PREC",
                      hot=lambda y: st._hot(CLIM, y), ring=lambda y: st._ring(RING, y),
                      sites_rc=st._sites(), ceilings=ceil if v4 else None,
                      max_size=q["guard"]["max_size"] if v4 else None)


def rule_masks(raw, s1, code, hot):
    """Full-frame boolean mask of the pixels each rule changed."""
    r0 = np.nan_to_num(raw)
    ch = np.abs(np.nan_to_num(s1) - np.where(r0 < 0.1, 0.0, r0)) > 1e-6
    hm = hot if hot is not None else np.zeros(raw.shape, bool)
    m = {"hot": ch & hm, "spike": ch & ~hm & (r0 >= 10)}
    for k, b in CODE_BIT.items():
        m[k] = (code & b) > 0
    return m


# ------------------------------------------------------------------------------------ census
def _census_day(args):
    day, rows = args
    import scan_tiles as st
    from src.data import cleaning
    dc = make_cleaner(day)
    hot = st._hot(CLIM, int(day[:4]))
    stamps = stamps_of(dc)
    out = []
    for ts, g in rows.groupby("timestamp"):
        if ts not in stamps:
            continue
        k = stamps.index(ts)
        raw = dc._read(dc.ds, k)
        s1 = cleaning.clean_frame(raw, hot)[0]
        fin, code = dc.clean(k)
        m = rule_masks(raw, s1, code, hot)
        for r in g.itertuples():
            s = (slice(r.row, r.row + 128), slice(r.col, r.col + 128))
            d = {"day": day, "timestamp": ts, "row": r.row, "col": r.col, "category": r.category,
                 "raw_max": float(np.nanmax(raw[s])), "max": float(np.nanmax(fin[s]))}
            d.update({k: int(v[s].sum()) for k, v in m.items()})
            out.append(d)
    dc.close()
    return pd.DataFrame(out)


def cmd_census(out, workers, seed=20261007):
    t = pd.read_csv(f"{CALIB}/tiles_decided.csv.gz", dtype={"timestamp": str, "day": str},
                    low_memory=False)
    cand = t[(t.n_repaired > 0) | (t.n_fixed > t.n_repaired)]
    rng = np.random.default_rng(seed)
    jobs = []
    for day, g in cand.groupby("day"):
        ts = g.timestamp.unique()
        n = 8 if day < "20180101" else 4                     # rings exist only to 2017
        pick = rng.choice(ts, min(n, len(ts)), replace=False)
        jobs.append((day, g[g.timestamp.isin(pick)]))
    print(f"[census] {len(jobs)} days, {sum(len(j[1]) for j in jobs):,} candidate tile-frames", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        res = []
        for i, r in enumerate(ex.map(_census_day, jobs), 1):
            res.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} days", flush=True)
    c = pd.concat(res, ignore_index=True)
    c.to_csv(os.path.join(out, "census.csv.gz"), index=False)
    print("[census] tiles with >= 1 px per rule:",
          {k: int((c[k] > 0).sum()) for k in FRAME_RULES + ["ceiling", "refused"]}, flush=True)


# ------------------------------------------------------------------------------------ render
def pick(df, key, n_top=4, n_rand=8, min_px=1, seed=1):
    x = df[df[key] >= min_px].copy()
    top = x.sort_values(key, ascending=False).drop_duplicates("day").head(n_top)
    rest = x.drop(top.index)
    rest = rest.groupby("day", group_keys=False).apply(lambda g: g.sample(min(len(g), 2), random_state=seed))
    rnd = rest.sample(min(n_rand, len(rest)), random_state=seed)
    return pd.concat([top.assign(draw="strongest"), rnd.assign(draw="random")])


def cmd_render(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    import scan_tiles as st
    from src.data import cleaning
    from src.data.radar_screen import FEATURES, RadarGeometry, isolated_mask

    norm = LogNorm(0.1, 300)
    cmap = plt.get_cmap("turbo").copy(); cmap.set_bad("white")
    cache, geoms = {}, {}

    def cl(day, v4=True):
        if (day, v4) not in cache:
            if len(cache) > 6:
                for kk in list(cache)[:3]:
                    cache.pop(kk).close()
            cache[(day, v4)] = make_cleaner(day, v4)
        return cache[(day, v4)]

    def get(ts, dmin=0):
        t2 = (pd.Timestamp(ts) + pd.Timedelta(minutes=dmin)).strftime("%Y%m%d%H%M%S")
        day = t2[:8]
        if not os.path.exists(f"{D}/raw/OPERA/{day}/.zmetadata"):
            return None
        dc = cl(day)
        s = stamps_of(dc)
        if t2 not in s:
            return None
        k = s.index(t2)
        raw = dc._read(dc.ds, k)
        hot = st._hot(CLIM, int(day[:4]))
        s1 = cleaning.clean_frame(raw, hot)[0]
        fin, code = dc.clean(k)
        return raw, s1, fin, code, hot

    def panel(ax, z, title, box, mask=None, color="k"):
        a = np.asarray(z, float)[box]
        img = np.where(np.isfinite(a) & (a >= 0.1), a, np.nan)
        ax.imshow(img, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        nod = ~np.isfinite(a)
        if nod.any():
            ax.imshow(np.where(nod, 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=4,
                      interpolation="nearest")
        if mask is not None and mask[box].any():
            ax.contour(mask[box].astype(float), levels=[0.5], colors=color, linewidths=0.7, origin="lower")
        ax.set_title(title, fontsize=6.5); ax.set_xticks([]); ax.set_yticks([])

    index = []
    pdf = PdfPages(os.path.join(out, "rule_gallery.pdf"))

    def page(title, examples, draw_one):
        for i0 in range(0, len(examples), 3):
            fig, axs = plt.subplots(3, 4, figsize=(11, 8.3), constrained_layout=True, squeeze=False)
            for ax in axs.ravel():
                ax.axis("off")
            for j, ex in enumerate(examples[i0:i0 + 3]):
                for ax in axs[j]:
                    ax.axis("on")
                lab = draw_one(axs[j], ex)
                index.append({"rule": title, "n": i0 + j + 1, **lab, "verdict": "", "note": ""})
                axs[j, 0].set_ylabel(f"#{i0 + j + 1} {lab['draw']}", fontsize=7)
            fig.suptitle(f"{title} ({i0 // 3 + 1}/{(len(examples) + 2) // 3})", fontsize=10)
            pdf.savefig(fig, dpi=110); plt.close(fig)
            print(f"[render] {title} {min(i0 + 3, len(examples))}/{len(examples)}", flush=True)

    def tile_example(rule, mask_fn, extra=""):
        def draw(axs, r):
            ts = r.timestamp
            box = (slice(r.row, r.row + 128), slice(r.col, r.col + 128))
            cur = get(ts)
            raw, s1, fin, code, hot = cur
            m = mask_fn(raw, s1, fin, code, hot, r)
            prev, nxt = get(ts, -15), get(ts, 15)
            if prev is not None:
                panel(axs[0], prev[0], "raw t-15", box, m)
            panel(axs[1], raw, f"raw {ts[:8]} {ts[8:12]} r{r.row} c{r.col}\nmax {np.nanmax(raw[box]):.0f}, "
                               f"{int(m[box].sum())} px outlined ({r.category[:22]})", box, m)
            rej = "" if "rej" not in extra else f" {extra}"
            panel(axs[2], fin, f"after the screen: max {np.nanmax(fin[box]):.0f}{rej}", box, m, "0.4")
            if nxt is not None:
                panel(axs[3], nxt[0], "raw t+15", box, m)
            return {"draw": r.draw, "day": r.day, "timestamp": ts, "row": r.row, "col": r.col,
                    "category": r.category, "px": int(m[box].sum())}
        return draw

    c = pd.read_csv(os.path.join(out, "census.csv.gz"), dtype={"timestamp": str, "day": str})
    titles = {"hot": "static clutter (hot pixels)", "spike": "isolated spike",
              "footprint": "unphysical core and footprint (> 500 mm/h)", "ray": "RLAN ray",
              "ring": "range ring (>= 89 mm/h)", "unsupported": "no temporal support"}
    shown = pd.DataFrame([{"day": "20160530", "timestamp": "20160530123000", "row": 896, "col": 1152,
                           "category": "event:elvira_friederike_201605", "draw": "shown to MCH"}])
    for rule in FRAME_RULES:
        ex = pick(c, rule, min_px=1 if rule in ("spike", "ring") else 3)
        if rule == "ray":                                # the example the researcher found to be a front
            ex = pd.concat([shown, ex], ignore_index=True)
        ex = list(ex.itertuples())
        page(titles[rule], ex, tile_example(rule, lambda raw, s1, fin, code, hot, r, rule=rule:
                                            rule_masks(raw, s1, code, hot)[rule]))

    t = pd.read_csv(f"{CALIB}/tiles_decided.csv.gz", dtype={"timestamp": str, "day": str},
                    low_memory=False)
    # ceiling: tiles with ceiling pixels, both repaired (1-4) and rejected (>= 5)
    ce = t[t.n_ceiling > 0].assign(**{"ceiling": lambda d: d.n_ceiling})
    ex = pick(ce, "ceiling", n_top=4, n_rand=8)
    small = ce[ce.n_ceiling < 5]
    ex = pd.concat([ex, small.sample(min(3, len(small)), random_state=2).assign(draw="random, 1-4 px")])
    page("reflectivity ceiling (outlined: ceiling pixels; >= 5 rejects the tile)", list(ex.itertuples()),
         tile_example("ceiling", lambda raw, s1, fin, code, hot, r: (code & 16) > 0))
    # repeated value (tile rejected)
    rp = t[t.rej_repeat].assign(rep=lambda d: d.rep_count)
    page("repeated value (tile rejected)", list(pick(rp, "rep").itertuples()),
         tile_example("repeat", lambda raw, s1, fin, code, hot, r:
                      np.abs(np.nan_to_num(fin) - float(r.rep_value)) < 0.006, "rej"))
    # refused repair (tile rejected)
    rf = t[t.rej_refused].assign(refused=lambda d: d.n_refused)
    page("refused repair (component > 500 px; tile rejected)", list(pick(rf, "refused").itertuples()),
         tile_example("refused", lambda raw, s1, fin, code, hot, r: (code & 32) > 0, "rej"))

    # radar-disk failure: flagged radar-frames, crop around the radar
    ff = pd.read_csv(f"{CALIB}/frame_flags.csv.gz", dtype=str)
    ff["day"] = ff.timestamp.str[:8]
    ff = ff.groupby(["radar", "day"], group_keys=False).apply(lambda g: g.sample(min(len(g), 2), random_state=3))
    ff = ff.assign(draw="flagged")

    def radar_box(year, key, half=160):
        if year not in geoms:
            geoms.clear(); geoms[year] = RadarGeometry(year)
        g = geoms[year]
        k = list(g.keys).index(key)
        sr, sc = int(g.sites["row"].iat[k]), int(g.sites["col"].iat[k])
        box = (slice(max(sr - half, 0), sr + half), slice(max(sc - half, 0), sc + half))
        return g, k, box

    def draw_frame(axs, r):
        ts = r.timestamp
        g, k, box = radar_box(int(ts[:4]), r.radar)
        own = g.owner == k
        cur = get(ts)
        raw, s1, fin, code, hot = cur
        prev, nxt = get(ts, -15), get(ts, 15)
        if prev is not None:
            panel(axs[0], prev[1], "t-15 (clean_frame)", box, own)
        panel(axs[1], s1, f"{r.radar} {ts[:8]} {ts[8:12]} (clean_frame)\nblack: owned area", box, own)
        panel(axs[2], fin, "after the screen (tiles >= 10% owned are rejected)", box, own, "0.4")
        if nxt is not None:
            panel(axs[3], nxt[1], "t+15 (clean_frame)", box, own)
        return {"draw": r.draw, "day": r.day, "timestamp": ts, "row": -1, "col": -1,
                "category": r.radar, "px": int(own.sum())}
    page("radar-disk failure (flagged radar-frames)", list(ff.itertuples()), draw_frame)

    # iso31: top radar-days by score_iso31, frame with the most isolated pixels
    rk = pd.read_csv(f"{CALIB}/ranking.csv.gz", dtype={"day": str})
    top = rk[rk.score_iso31 > 0].sort_values("score_iso31", ascending=False)
    ora = rk[(rk.radar == "roopa") & rk.day.str.startswith("201911")].head(2)
    top = pd.concat([top.head(10), ora]).drop_duplicates(["radar", "day"]).assign(draw="top iso31")
    iiso = FEATURES.index("n_iso31")

    def draw_iso(axs, r):
        z = np.load(f"{CALIB}/radars/{r.day}.npz")
        keys = list(z["keys"].astype(str))
        kk = keys.index(r.radar)
        f = np.nan_to_num(z["feats"][:, kk, iiso])
        ts = str(z["times"].astype(str)[int(np.argmax(f))])
        g, k, box = radar_box(int(ts[:4]), r.radar)
        own = g.owner == k
        raw, s1, fin, code, hot = get(ts)
        ys, xs = np.nonzero(own & (np.nan_to_num(s1) >= 31))
        iso = np.zeros(own.shape, bool)
        if ys.size:
            iso[ys[isolated_mask(s1, ys, xs)], xs[isolated_mask(s1, ys, xs)]] = True
        from scipy import ndimage
        isod = ndimage.binary_dilation(iso, iterations=2)
        prev, nxt = get(ts, -15), get(ts, 15)
        if prev is not None:
            panel(axs[0], prev[1], "t-15 (clean_frame)", box, isod)
        panel(axs[1], s1, f"{r.radar} {ts[:8]} {ts[8:12]}: {int(iso.sum())} isolated px >= 31\n"
                          f"day {r.iso31:.0f} vs neighbours {r.iso31_nb:.0f}", box, isod)
        panel(axs[2], fin, "after the screen (no automatic action)", box, own, "0.4")
        if nxt is not None:
            panel(axs[3], nxt[1], "t+15 (clean_frame)", box, isod)
        return {"draw": r.draw, "day": r.day, "timestamp": ts, "row": -1, "col": -1,
                "category": r.radar, "px": int(iso.sum())}
    page("radar-day ranking: isolated high pixels (top radar-days)", list(top.itertuples()), draw_iso)

    pdf.close()
    pd.DataFrame(index).to_csv(os.path.join(out, "index.csv"), index=False)
    print(f"[render] {len(index)} examples -> {out}/rule_gallery.pdf, index.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["census", "render"])
    ap.add_argument("--out", default=f"{CALIB}/rule_gallery")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "census":
        cmd_census(a.out, a.workers)
    else:
        cmd_render(a.out)
