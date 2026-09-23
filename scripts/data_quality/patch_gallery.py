#!/usr/bin/env python
"""Look at the tiles: image pages for a selection, plus a label sheet to fill in.

One row per tile, columns:

    t-15 raw | t raw | t+15 raw | target as the model sees it | coarse input | DEM | QIND

"raw" is the stored field before any filtering; "target as the model sees it" applies the
current drizzle/declutter step (`pipeline_filter`), so declutter holes show up as white
pixels inside a storm. The row title carries the tile key, raw max and any rules that fire.

Selections (`--select`):

    top_max            highest raw max first
    rule:<name>        tiles a rule in rules.py flags, highest max first (e.g. rule:flicker)
    unflagged          tiles no rule flags — the "clean" tail, to check for misses
    random             uniform over the stratum
    list:<csv>         explicit timestamp,row,col list (e.g. an earlier label sheet)

all restricted to `--min_max <= raw max < --max_max`. Writes `page_XXX.png` and
`labels.csv` (timestamp,row,col,page,slot,rules,label,notes) under
`<out_dir>/<selection>/`. Fill `label` with artefact / real / unsure and pass the sheet to
`summarize_features.py --labels` for per-rule precision and recall.

    python scripts/data_quality/patch_gallery.py config.yaml \
        --features_dir .../quality/features --out_dir .../quality/gallery \
        --select random --min_max 31 --n 64
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
from src.data.quality import adaptive_block_means, pipeline_filter  # noqa: E402
from src.utils import load_config  # noqa: E402
from rules import RULES, apply_rules  # noqa: E402
from summarize_features import load_features  # noqa: E402

_DAY_CACHE = {}


def open_day(raw_dir, day):
    import xarray as xr
    if day not in _DAY_CACHE:
        p = os.path.join(raw_dir, day)
        try:
            _DAY_CACHE[day] = xr.open_zarr(p, consolidated=True)
        except Exception:
            _DAY_CACHE[day] = xr.open_zarr(p, consolidated=False)
    return _DAY_CACHE[day]


def read_tile(raw_dir, var, ts, row, col, patch):
    """(prev, cur, next, qind) for one tile; None where unavailable."""
    t = np.datetime64(pd.to_datetime(ts, format="%Y%m%d%H%M%S"))
    out = []
    for dt_min in (-15, 0, 15):
        tt = t + np.timedelta64(dt_min, "m")
        day = str(tt.astype("datetime64[D]")).replace("-", "")
        try:
            ds = open_day(raw_dir, day)
        except Exception:
            out.append(None)
            continue
        hit = np.nonzero(ds.time.values == tt)[0]
        if not len(hit):
            out.append(None)
            continue
        sl = dict(time=int(hit[0]), y=slice(row, row + patch), x=slice(col, col + patch))
        out.append(ds[var].isel(**sl).values)
        if dt_min == 0:
            q = ds["QIND"].isel(**sl).values if "QIND" in ds else None
    return out[0], out[1], out[2], q


def select(df, flags, how, n, lo, hi, seed):
    m = (df["max"] >= lo) & (df["max"] < hi)
    if how == "top_max":
        pick = df[m].sort_values("max", ascending=False)
    elif how.startswith("rule:"):
        rule = how.split(":", 1)[1]
        if rule not in flags:
            raise KeyError(f"unknown rule {rule!r}; have {list(RULES)}")
        pick = df[m & (flags[rule] > 0)].sort_values("max", ascending=False)
    elif how == "unflagged":
        pick = df[m & (flags["any_flag"] == 0)].sample(frac=1, random_state=seed)
    elif how == "random":
        pick = df[m].sample(frac=1, random_state=seed)
    else:
        raise ValueError(how)
    return pick.head(n)


def render(rows, flags, raw_dir, var, dem, patch, factor, page_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    norm = LogNorm(vmin=0.1, vmax=200)
    cols = ["t-15 raw", "t raw", "t+15 raw", "target (filtered)", "coarse input", "DEM", "QIND"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.1 * len(cols), 2.3 * len(rows)),
                             squeeze=False)
    im = None
    for i, (_, r) in enumerate(rows.iterrows()):
        prev, cur, nxt, q = read_tile(raw_dir, var, r["timestamp"], int(r["row"]),
                                      int(r["col"]), patch)
        target = pipeline_filter(cur) if cur is not None else None
        coarse = None
        if target is not None:
            c = adaptive_block_means(target, int(patch / factor))
            coarse = np.kron(c, np.ones((int(np.ceil(patch / c.shape[0])),) * 2))[:patch, :patch]
        panels = [prev, cur, nxt, target, coarse]
        for j, a in enumerate(panels):
            ax = axes[i, j]
            if a is not None:
                shown = np.where(a >= 0.1, a, np.nan)
                im = ax.imshow(shown, origin="lower", cmap=cmap, norm=norm,
                               interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        sl = (slice(int(r["row"]), int(r["row"]) + patch),
              slice(int(r["col"]), int(r["col"]) + patch))
        axes[i, 5].imshow(dem[sl], origin="lower", cmap="terrain", vmin=-100, vmax=2500)
        if q is not None:
            axes[i, 6].imshow(q, origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
        else:
            axes[i, 6].text(0.5, 0.5, "no QIND", ha="center", va="center",
                            transform=axes[i, 6].transAxes, fontsize=8)
        fired = [k for k in RULES if flags.at[r.name, k] > 0]
        axes[i, 0].set_ylabel(f"#{i}  {r['timestamp']}\n({int(r['row'])},{int(r['col'])})",
                              fontsize=7)
        axes[i, 0].text(0.0, 1.03, f"max {r['max']:.1f} mm/h   " + ", ".join(fired),
                        transform=axes[i, 0].transAxes, fontsize=7, va="bottom")
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=8, pad=14)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(right=0.9, hspace=0.35, wspace=0.08, top=0.95)
    if im is not None:
        cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
        fig.colorbar(im, cax=cax, label="mm/h")
    fig.suptitle(title, fontsize=10, y=0.99)
    fig.savefig(page_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--select", default="top_max")
    ap.add_argument("--min_max", type=float, default=31.0)
    ap.add_argument("--max_max", type=float, default=np.inf)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--per_page", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["RAW_OPERA_DATA_DIR"]
    patch, factor = cfg["PATCH_SIZE"], cfg["DOWNSCALING_FACTOR"]

    df, _ = load_features(args.features_dir)
    flags = apply_rules(df)
    if args.select.startswith("list:"):
        want = pd.read_csv(args.select.split(":", 1)[1], dtype={"timestamp": str})
        key = ["timestamp", "row", "col"]
        idx = df.reset_index().merge(want[key], on=key)["index"]
        rows = df.loc[idx].head(args.n)
        name = "list_" + os.path.splitext(os.path.basename(args.select.split(":", 1)[1]))[0]
    else:
        rows = select(df, flags, args.select, args.n, args.min_max, args.max_max, args.seed)
        name = f"{args.select.replace(':', '_')}_max{args.min_max:g}-{args.max_max:g}"
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    print(f"{len(rows)} tiles -> {out}", flush=True)

    import xarray as xr
    with xr.open_dataset(cfg["STATIC_DEM_PATH"], engine="rasterio") as ds:
        dem = ds["band_data"].isel(band=0).values

    sheet = []
    for p, start in enumerate(range(0, len(rows), args.per_page)):
        chunk = rows.iloc[start:start + args.per_page]
        path = os.path.join(out, f"page_{p:03d}.png")
        render(chunk, flags, raw_dir, cfg["PRECIP_VAR_NAME"], dem, patch, factor, path,
               f"{name} — page {p}")
        for slot, (i, r) in enumerate(chunk.iterrows()):
            sheet.append({"timestamp": r["timestamp"], "row": int(r["row"]),
                          "col": int(r["col"]), "max": round(float(r["max"]), 2),
                          "page": p, "slot": slot,
                          "rules": ";".join(k for k in RULES if flags.at[i, k] > 0),
                          "label": "", "notes": ""})
        print(f"  {path}", flush=True)
        _DAY_CACHE.clear()
    pd.DataFrame(sheet).to_csv(os.path.join(out, "labels.csv"), index=False)
    print(f"label sheet: {os.path.join(out, 'labels.csv')}")


if __name__ == "__main__":
    main()
