#!/usr/bin/env python
"""
Clear errors among the held-out tile-hours above the cut (DECISIONS §43, todo B5; screening policy §17).

For every test / nimbus tile-hour with target max >= u_c, from the hourly scan of the same tile:
    isolated   target max >= u_c while the tile's previous and next hours are both < 10 mm
               (the single-hour spike signature; 2% of >= 50 mm tile-hours on a random sample)
    prev, next the neighbouring hours' maxima on the same tile
Then gallery pages of the target field (adv-corrected, as trained on) for: every isolated one, the
48 largest, and 48 random ones. Nothing is removed here: the pages are for looking (§17: clear errors
only), and the flags go into the event table and the R4/R5 reporting.

Usage: python scripts/data_quality/heldout_tail_check.py 50.1 <out_dir>
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402

D = "/home/fquareng/work/data/extremes/OPERA"
V4 = f"{D}/patches_v4_adv"
HOURS = f"{D}/quality_v4/hourly/hours"
GROUP = {"test": "test", "nimbus": "nimbus"}


def neighbours(rows):
    """prev / next hour max of the same tile, from the scan tables of the day before, of and after."""
    days = sorted({str(t)[:8] for t in rows.t})
    need = sorted({d for day in days for d in (pd.Timestamp(day) + pd.Timedelta(days=k) for k in (-1, 0, 1))})
    h = pd.concat([pd.read_csv(f"{HOURS}/{d:%Y%m%d}.csv.gz", usecols=["end", "row", "col", "max"])
                   for d in need if os.path.exists(f"{HOURS}/{d:%Y%m%d}.csv.gz")])
    h = h.drop_duplicates(["end", "row", "col"]).set_index(["end", "row", "col"])["max"]
    t = pd.to_datetime(rows.t.astype(str), format="%Y%m%d%H%M%S")
    key = lambda dt: [(int(x.strftime("%Y%m%d%H%M%S")), y, c) for x, y, c in zip(t + dt, rows.y, rows.x)]
    return h.reindex(key(pd.Timedelta(hours=-1))).values, h.reindex(key(pd.Timedelta(hours=1))).values


def page(sel, title, path):
    n = len(sel)
    if n == 0:
        return
    cols = 8
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(2.1 * cols, 2.3 * np.ceil(n / cols)), squeeze=False)
    stores = {sp: zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")[GROUP[sp]]["original_precip"]
              for sp in sel.split.unique()}
    for ax, r in zip(axs.flat, sel.itertuples()):
        f = np.asarray(stores[r.split][int(r.r)])
        ax.imshow(np.log1p(f), origin="lower", cmap="turbo", vmin=0, vmax=np.log1p(150))
        ax.set_title(f"{r.split[:3]} {str(r.t)[:10]} ({r.y},{r.x})\nmax {r.mx:.0f}  prev {r.prev:.0f} next {r.next:.0f}",
                     fontsize=6)
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + "  (log1p mm, colour cap 150 mm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    u, out = float(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    parts = []
    for sp in GROUP:
        m = pd.read_csv(f"{V4}/meta/full_{sp}.txt", header=None, names=["t", "y", "x", "mx", "r"])
        parts.append(m[m.mx >= u].assign(split=sp))
    d = pd.concat(parts, ignore_index=True)
    d["prev"], d["next"] = neighbours(d)
    d["isolated"] = (d.prev < 10) & (d.next < 10)
    d.to_csv(os.path.join(out, "heldout_flags.csv.gz"), index=False)
    for sp, g in d.groupby("split"):
        print(f"{sp}: {len(g)} tile-hours >= {u}; isolated {g.isolated.sum()} ({g.isolated.mean():.1%}); "
              f"by max bin isolated share: "
              + ", ".join(f"[{a},{b}) {g[(g.mx >= a) & (g.mx < b)].isolated.mean():.0%}"
                          for a, b in ((u, 80), (80, 100), (100, 150), (150, 1e9))))
    rng = np.random.default_rng(0)
    iso = d[d.isolated].sort_values("mx", ascending=False)
    for k in range(0, len(iso), 48):
        page(iso.iloc[k:k + 48], f"isolated tile-hours >= {u} mm, {k + 1}-{min(k + 48, len(iso))} of {len(iso)}",
             os.path.join(out, f"isolated_{k // 48:02d}.png"))
    page(d.nlargest(48, "mx"), f"48 largest held-out tile-hours", os.path.join(out, "largest.png"))
    page(d.iloc[rng.choice(len(d), 48, replace=False)].sort_values("mx"), "48 random held-out tile-hours >= u_c",
         os.path.join(out, "random.png"))
