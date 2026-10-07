"""One figure per screening rule, from real failing radars (notes/mch_slides.tex, "Rules by
example"). Raw field, cleaned field and the pixels the rule changed; for the radar-level
rules, the statistic the rule computes.

    python notes/figures/make_rule_examples.py notes/figures/mch

Reads the raw archive, quality_v2 climatologies and quality_v4/calib (Phase 0). The example
frames were picked by per-rule pixel counts on Phase-0 tile-frames (2026-10-07).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
import scan_tiles as st  # noqa: E402
from src.data import cleaning  # noqa: E402
from src.data.day_cleaner import DayCleaner  # noqa: E402
from src.data.radar_screen import VB0, VB1, RadarGeometry, dbz  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
CALIB = f"{D}/quality_v4/calib"
CLIM = f"{D}/quality_v2/clutter_climatology.npz"
RING = f"{D}/quality_v2/ring_mask.npz"
QPATH = f"{CALIB}/quality_calib.yaml"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
Q = yaml.safe_load(open(QPATH))
_, CEIL = st._quality(QPATH)
NORM = LogNorm(0.1, 300)
CMAP = plt.get_cmap("turbo").copy()
CMAP.set_bad("0.85")                                   # no coverage: grey; dry: white (masked below)
_CL = {}


def cleaner(day, v4=True):
    key = (day, v4)
    if key not in _CL:
        _CL[key] = DayCleaner(f"{D}/raw/OPERA", day, "TOT_PREC",
                              hot=lambda y: st._hot(CLIM, y), ring=lambda y: st._ring(RING, y),
                              sites_rc=st._sites(), ceilings=CEIL if v4 else None,
                              max_size=Q["guard"]["max_size"] if v4 else None)
    return _CL[key]


def frame(ts, v4=True, dk=0):
    """(raw, stage-1 clean_frame, final, repair code) of the frame `ts` (+dk steps)."""
    day = ts[:8]
    dc = cleaner(day, v4)
    stamps = [np.datetime_as_string(x, unit="s").replace("-", "").replace("T", "").replace(":", "")
              for x in dc.times]
    k = stamps.index(ts) + dk
    raw = dc._read(dc.ds, k)
    s1 = cleaning.clean_frame(raw, st._hot(CLIM, int(day[:4])))[0]
    fin, code = dc.clean(k)
    return raw, s1, fin, code


def show(ax, z, title, box=None):
    z = np.asarray(z, float)
    if box is not None:
        z = z[box]
    img = np.where(np.isfinite(z) & (z < 0.1), np.nan, z)
    img = np.ma.masked_where(np.isfinite(z) & (z < 0.1), img)
    cm = CMAP.copy()
    cm.set_bad("white")
    ax.imshow(np.where(np.isfinite(z), img, np.nan), origin="lower", cmap=cm, norm=NORM)
    nod = ~np.isfinite(z)
    if nod.any():
        ax.imshow(np.where(nod, 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=4)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def mark(ax, m, color="k", box=None):
    if box is not None:
        m = m[box]
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=color, linewidths=0.8, origin="lower")


def tile(r, c, n=128):
    return (slice(r, r + n), slice(c, c + n))


def colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("mm/h", fontsize=8); cb.ax.tick_params(labelsize=7)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("wrote", name, flush=True)


def changed(raw, s1):
    r = np.where(np.nan_to_num(raw) < 0.1, 0.0, np.nan_to_num(raw))
    return np.abs(np.nan_to_num(s1) - r) > 1e-6


# ---------------------------------------------------------------------------- frame rules
def fig_static_clutter():
    ts, r, c = "20160518060000", 768, 1152
    raw, s1, fin, code = frame(ts)
    hot = st._hot(CLIM, 2016)
    clim = np.load(CLIM)
    f31 = clim["y2016_n_ge31"] / np.maximum(clim["y2016_n_valid"], 1)
    b = tile(r, c)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    ax[0].imshow(np.where(f31[b] > 0, f31[b], np.nan), origin="lower", cmap="magma_r",
                 norm=LogNorm(1e-4, 0.3))
    mark(ax[0], hot, "tab:blue", b)
    ax[0].set_title("share of 2016 frames >= 31 mm/h\n(blue: > 1%, the hot mask)", fontsize=8)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    show(ax[1], raw, f"raw {ts[:8]} {ts[8:12]}, max {np.nanmax(raw[b]):.0f} mm/h", b)
    mark(ax[1], changed(raw, s1) & hot, "k", b)
    show(ax[2], fin, f"cleaned, max {np.nanmax(fin[b]):.0f} mm/h", b)
    colorbar(fig, ax[2])
    save(fig, "ex_static_clutter.png")


def fig_spike():
    ts, r, c = "20140929160000", 512, 768
    raw, s1, fin, code = frame(ts)
    hot = st._hot(CLIM, 2014)
    sp = changed(raw, s1) & ~hot & (np.nan_to_num(raw) >= 10)
    ys, xs = np.nonzero(sp[tile(r, c)])
    i = np.argmax(np.nan_to_num(raw[tile(r, c)])[ys, xs])          # the strongest spike
    cy, cx = int(ys[i]) + r, int(xs[i]) + c
    b = (slice(cy - 16, cy + 16), slice(cx - 16, cx + 16))
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6), constrained_layout=True)
    show(ax[0], raw, f"raw {ts[:8]} {ts[8:12]} (64 km crop), max {np.nanmax(raw[b]):.0f}", b)
    mark(ax[0], sp, "k", b)
    show(ax[1], s1, f"after the spike rule, max {np.nanmax(s1[b]):.0f}", b)
    colorbar(fig, ax[1])
    save(fig, "ex_spike.png")


def fig_code_rule(ts, r, c, bit, name, label):
    raw, s1, fin, code = frame(ts)
    b = tile(r, c)
    m = (code & bit) > 0
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6), constrained_layout=True)
    show(ax[0], raw, f"raw {ts[:8]} {ts[8:12]}, max {np.nanmax(raw[b]):.0f} mm/h", b)
    mark(ax[0], m, "k", b)
    show(ax[1], fin, f"cleaned, max {np.nanmax(fin[b]):.0f} mm/h ({label})", b)
    colorbar(fig, ax[1])
    save(fig, name)


def fig_ring():
    ts, r, c = "20140920080000", 1024, 1024
    raw, s1, fin, code = frame(ts)
    clim = np.load(CLIM)
    f31 = clim["y2014_n_ge31"] / np.maximum(clim["y2014_n_valid"], 1)
    ring = np.load(RING)["y2014"]
    b = (slice(r - 128, r + 256), slice(c - 128, c + 256))
    bt = tile(r, c)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    ax[0].imshow(np.where(f31[b] > 0, f31[b], np.nan), origin="lower", cmap="magma_r",
                 norm=LogNorm(1e-5, 1e-2))
    mark(ax[0], ring, "tab:blue", b)
    ax[0].add_patch(plt.Rectangle((128, 128), 128, 128, fill=False, ec="k", lw=0.8))
    ax[0].set_title("share of 2014 frames >= 31 mm/h, 768 km\n(blue: ring mask; box: tile)", fontsize=8)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    show(ax[1], raw, f"raw {ts[:8]} {ts[8:12]}, max {np.nanmax(raw[bt]):.0f} mm/h", bt)
    mark(ax[1], (code & cleaning.REPAIR_RING) > 0, "k", bt)
    show(ax[2], fin, f"cleaned, max {np.nanmax(fin[bt]):.0f} mm/h", bt)
    colorbar(fig, ax[2])
    save(fig, "ex_ring.png")


def fig_unsupported():
    ts, r, c = "20250717234500", 512, 1152
    b = tile(r, c)
    raw, s1, fin, code = frame(ts)
    m = (code & cleaning.REPAIR_UNSUPPORTED) > 0
    step = lambda m: (pd.Timestamp(ts) + pd.Timedelta(minutes=m)).strftime("%Y%m%d%H%M%S")  # noqa: E731
    prev = frame(step(-15))[1]                          # by timestamp: t+15 may be the next day
    nxt = frame(step(+15))[1]
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.4), constrained_layout=True)
    show(ax[0], prev, "t - 15 min", b); mark(ax[0], m, "k", b)
    show(ax[1], s1, f"t = {ts[8:10]}:{ts[10:12]}, max {np.nanmax(s1[b]):.0f} mm/h", b); mark(ax[1], m, "k", b)
    show(ax[2], nxt, "t + 15 min", b); mark(ax[2], m, "k", b)
    show(ax[3], fin, f"cleaned t, max {np.nanmax(fin[b]):.0f} mm/h", b)
    colorbar(fig, ax[3])
    save(fig, "ex_unsupported.png")


# ---------------------------------------------------------------------------- radar rules
def radar_year_hist(key, year):
    """Summed value histogram (value*100 -> count) of one radar over the calibration days of
    one year (radar pass, before repairs)."""
    acc = {}
    for f in sorted(glob.glob(f"{CALIB}/radars/{year}*.npz")):
        z = np.load(f)
        keys = list(z["keys"].astype(str))
        if key not in keys:
            continue
        i = keys.index(key)
        hk, hn = z["hist_key"], z["hist_n"]
        nb = VB1 - VB0 + 1
        sel = hk // nb == i
        for b, n in zip(hk[sel] % nb + VB0, hn[sel]):
            acc[int(b)] = acc.get(int(b), 0) + int(n)
    v = np.array(sorted(acc)); return v / 100.0, np.array([acc[x] for x in v])


def fig_ceiling():
    ts, r, c = "20230709190000", 512, 1280
    raw = frame(ts)[0]
    b = tile(r, c)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    show(ax[0], raw, f"raw {ts[:8]} {ts[8:12]}, Serbia\n(black: pixels at exactly 364.63 mm/h)", b)
    mark(ax[0], np.abs(np.nan_to_num(raw) - 364.63) < 0.006, "k", b)
    w = 10 ** (4 / 16)                                   # +-4 dB as a rate factor
    for a, (key, year, val, lo, hi, lab) in zip(ax[1:], (
            ("rsval", 2023, 364.63, 150, 500, "Valjevo 2023: a ceiling"),
            ("esahr", 2024, 10.28, 6, 18, "Malaga 2024: a value ladder, not a ceiling"))):
        v, n = radar_year_hist(key, year)
        s = (v >= lo) & (v <= hi)
        a.vlines(v[s], 0.8, n[s], color="0.4", lw=0.6)
        k = np.argmin(np.abs(v - val))
        a.vlines([v[k]], 0.8, [n[k]], color="tab:red", lw=1.5)
        a.axvspan(val / w, val * w, color="tab:blue", alpha=0.08)
        nb = n[(v >= val / w) & (v <= val * w) & (np.abs(v - val) > 0.005)]
        ratio = n[k] / max(nb.max() if nb.size else 1, 1)
        a.set_yscale("log"); a.set_xscale("log"); a.set_ylim(0.8, None)
        a.set_xlabel("rain rate (mm/h), 0.01 mm/h bins"); a.set_ylabel("count, calibration days")
        a.set_title(f"{lab}: {val} mm/h = {dbz(val):.1f} dBZ\ncount / largest other count "
                    f"within +-4 dB (shaded) = {ratio:.1f}", fontsize=8)
    save(fig, "ex_ceiling.png")


def fig_repeat():
    ts, r, c = "20230803191500", 512, 1408
    raw, s1, fin, code = frame(ts)
    t = pd.read_csv(f"{CALIB}/tiles_decided.csv.gz", dtype={"timestamp": str, "day": str},
                    low_memory=False)
    row = t[(t.timestamp == ts) & (t.row == r) & (t.col == c)].iloc[0]
    b = tile(r, c)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1.5]})
    show(ax[0], fin, f"{ts[:8]} {ts[8:12]}: {int(row.rep_count)} pixels at exactly "
                     f"{row.rep_value:.2f} mm/h", b)
    mark(ax[0], np.abs(np.nan_to_num(fin) - row.rep_value) < 0.006, "k", b)
    tail = t[t["max"] >= 31]
    for lab, sel, col in (("random and event days", tail.category.eq("random") | tail.category.str.startswith("event"), "0.4"),
                          ("known-failure days", ~(tail.category.eq("random") | tail.category.str.startswith("event")), "tab:red")):
        x = tail.loc[sel, "max_rep"].dropna()
        ax[1].hist(x, bins=np.arange(0.5, 400, 1), histtype="step", color=col, label=f"{lab} (n={len(x):,})")
    ax[1].axvline(50, color="k", ls="--", lw=0.8); ax[1].text(52, 3e4, "reject >= 50", fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("largest number of pixels sharing one value >= 31 mm/h, per tile")
    ax[1].set_ylabel("tiles with max >= 31 mm/h"); ax[1].legend(fontsize=7)
    save(fig, "ex_repeat.png")


def fig_radar_disk():
    def geom(year):
        g = RadarGeometry(year)
        k = list(g.keys).index("estjv")
        sr, sc = g.sites["row"].iat[k], g.sites["col"].iat[k]
        rr, cc = np.indices(g.shape)
        return g.owner == k, np.hypot(rr - sr, cc - sc) * 2.0, g.sites["maxrange_km"].iat[k], int(sr), int(sc)
    cases = [("20180429054500", "2018-04-29 05:45, Madrid (640 km)", "failure", "tab:red"),
             ("20241029220000", "2024-10-29 22:00, same radar, rain", "rain", "0.4")]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    for a, (ts, lab, short, col) in zip(ax[:2], cases):
        own, rng_km, maxr, sr, sc = geom(int(ts[:4]))
        b = (slice(sr - 160, sr + 160), slice(sc - 160, sc + 160))
        z = frame(ts)[1]
        show(a, z, lab, b)
        mark(a, own, "k", b)
        mark(a, rng_km <= maxr, "tab:blue", b)
        m = own & np.isfinite(z) & (z >= 0.1)
        x, y = rng_km[m], np.log1p(z[m])
        bins = np.arange(0, 252, 4)
        idx = np.digitize(x, bins)
        mean = np.array([y[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) + 1)])
        r2 = 1 - np.nanvar(y - mean[idx]) / np.var(y)
        sub = np.random.default_rng(0).choice(x.size, min(x.size, 4000), replace=False)
        ax[2].scatter(x[sub], y[sub], s=1, color=col, alpha=0.3)
        ax[2].plot(bins, mean[1:], color=col, lw=1.5, label=f"{short}: R2 of range alone = {r2:.2f}")
    ax[2].set_xlabel("range from the radar (km)"); ax[2].set_ylabel("log(1 + rate)")
    ax[2].set_title("owned pixels: rate against range", fontsize=8); ax[2].legend(fontsize=7)
    save(fig, "ex_radar_disk.png")


def fig_guard():
    ts, r, c = "20180429061500", 256, 256
    b = tile(r, c)
    raw = frame(ts, v4=False)[0]
    v3 = frame(ts, v4=False)[2]
    raw4, _, v4, code = frame(ts, v4=True)
    vals, cnt = np.unique(np.round(v3[b][np.isfinite(v3[b])], 2), return_counts=True)
    top = vals[np.argmax(cnt * (vals >= 31))]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    show(ax[0], raw, f"raw {ts[:8]} {ts[8:12]}, Madrid, max {np.nanmax(raw[b]):.0f}", b)
    show(ax[1], v3, f"unguarded repair: {cnt.max():,} pixels at {top:.2f} mm/h", b)
    mark(ax[1], np.abs(np.nan_to_num(v3) - top) < 0.006, "k", b)
    show(ax[2], v4, "guarded: repair refused (black), tile rejected", b)
    mark(ax[2], (code & cleaning.REPAIR_REFUSED) > 0, "k", b)
    colorbar(fig, ax[2])
    save(fig, "ex_guard.png")


def fig_iso31():
    r = pd.read_csv(f"{CALIB}/ranking.csv.gz", dtype={"day": str})
    days = pd.read_csv(f"{CALIB}/days.csv", dtype={"day": str})
    r = r.merge(days, on="day", how="left")
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    cat = r.category.fillna("")
    for lab, sel, col in (("random days", cat.eq("random"), "0.6"),
                          ("event days", cat.str.startswith("event"), "tab:blue"),
                          ("Valjevo 2023 and Madrid failure days", cat.isin(["valjevo_ceiling", "madrid"]), "tab:orange")):
        ax.scatter(r.iso31_nb[sel] + 1, r.iso31[sel] + 1, s=2, color=col, alpha=0.4, label=lab)
    o = (r.radar == "roopa") & r.day.str.startswith("201911")
    ax.scatter(r.iso31_nb[o] + 1, r.iso31[o] + 1, s=25, color="tab:red", label="Oradea, Nov 2019")
    lim = [0.7, 1e5]
    ax.plot(lim, lim, color="k", lw=0.6); ax.plot(lim, [10 * x for x in lim], color="k", lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("median of the 5 nearest radars, same day (+1)")
    ax.set_ylabel("isolated pixels >= 31 mm/h, this radar-day (+1)")
    ax.set_title("isolated high pixels per radar-day (calibration days; dashed: 10x)", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "ex_iso31.png")


if __name__ == "__main__":
    which = sys.argv[2:] or ["static", "spike", "footprint", "ray", "ring", "unsupported",
                             "ceiling", "repeat", "disk", "guard", "iso"]
    jobs = {"static": fig_static_clutter, "spike": fig_spike,
            "footprint": lambda: fig_code_rule("20140608064500", 1152, 1024, cleaning.REPAIR_FOOTPRINT,
                                               "ex_footprint.png", "footprint lowered to its ring median"),
            "ray": lambda: fig_code_rule("20160530123000", 896, 1152, cleaning.REPAIR_RAY,
                                         "ex_ray.png", "ray lowered to its ring median"),
            "ring": fig_ring, "unsupported": fig_unsupported, "ceiling": fig_ceiling,
            "repeat": fig_repeat, "disk": fig_radar_disk, "guard": fig_guard, "iso": fig_iso31}
    for w in which:
        jobs[w]()
