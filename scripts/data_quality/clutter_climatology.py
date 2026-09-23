#!/usr/bin/env python
"""Per-pixel exceedance climatology over the raw archive: where does "rain" recur?

A real storm crosses a pixel a handful of times a year at high intensity; ground clutter,
wind farms, sea clutter and RLAN spokes light up the *same* pixels over and over. Counting,
per pixel, how often each threshold is exceeded across every 15-minute composite makes
those locations stand out as hot spots and rays on a map, which no per-tile feature can do.

Writes `<out_dir>/clutter_climatology.npz` with, for the whole period and per year:

    n_valid          time steps with a finite value
    n_ge<u>          exceedance counts for u in THRESHOLDS
    freq_ge<u>       n_ge<u> / n_valid          (whole period only)
    max              pixel maximum
    q_mean           mean QIND where present    (whole period only)

and one PNG map per threshold under `<out_dir>/figures/`. `freq_ge*` feeds the `clim_*`
features of `compute_tile_features.py --climatology`.

    python scripts/data_quality/clutter_climatology.py config.yaml \
        --out_dir /home/fquareng/work/data/extremes/OPERA/quality --workers 6
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_config  # noqa: E402

THRESHOLDS = (1.0, 31.0, 89.0, 150.0, 500.0)


def day_counts(day_dir, var):
    import xarray as xr
    try:
        ds = xr.open_zarr(day_dir, consolidated=True)
    except Exception:
        ds = xr.open_zarr(day_dir, consolidated=False)
    H, W = ds.sizes["y"], ds.sizes["x"]
    acc = {"n_valid": np.zeros((H, W), np.int32), "max": np.full((H, W), -np.inf, np.float32)}
    for u in THRESHOLDS:
        acc[f"n_ge{u:g}"] = np.zeros((H, W), np.int32)
    q_sum, q_n = np.zeros((H, W)), np.zeros((H, W), np.int32)
    for i in range(ds.sizes["time"]):
        r = ds[var].isel(time=i).values
        fin = np.isfinite(r)
        acc["n_valid"] += fin
        z = np.where(fin, r, 0.0)
        np.maximum(acc["max"], z, out=acc["max"], where=fin)
        for u in THRESHOLDS:
            acc[f"n_ge{u:g}"] += z >= u
        if "QIND" in ds:
            q = ds["QIND"].isel(time=i).values
            fq = np.isfinite(q) & fin
            q_sum[fq] += q[fq]
            q_n += fq
    acc["q_sum"], acc["q_n"] = q_sum, q_n
    return os.path.basename(day_dir), acc


def _merge(into, acc):
    for k, v in acc.items():
        if k not in into:
            into[k] = v.copy()
        elif k == "max":
            np.maximum(into[k], v, out=into[k])
        else:
            into[k] += v


def plot_maps(res, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    os.makedirs(fig_dir, exist_ok=True)
    for u in THRESHOLDS:
        f = res[f"freq_ge{u:g}"]
        fig, ax = plt.subplots(figsize=(8, 9))
        pos = f[f > 0]
        if pos.size:
            im = ax.imshow(np.where(f > 0, f, np.nan), origin="lower", cmap="magma",
                           norm=LogNorm(vmin=max(pos.min(), 1e-6), vmax=pos.max()))
            fig.colorbar(im, ax=ax, shrink=0.7, label=f"fraction of time steps >= {u:g} mm/h")
        ax.set_title(f"Exceedance frequency >= {u:g} mm/h "
                     f"({int(res['n_valid'].max())} time steps max)")
        ax.set_axis_off()
        fig.savefig(os.path.join(fig_dir, f"clim_freq_ge{u:g}.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    # Hot-spot ranking: pixels whose >=31 frequency is far above their neighbourhood's.
    from scipy import ndimage
    f = res["freq_ge31"]
    bg = ndimage.median_filter(f, size=15)
    excess = f - bg
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(np.where(excess > 0, excess, np.nan), origin="lower", cmap="viridis",
                   norm=LogNorm(vmin=1e-5, vmax=max(float(np.nanmax(excess)), 2e-5)))
    fig.colorbar(im, ax=ax, shrink=0.7, label="freq>=31 minus 15-px median")
    ax.set_title("Local excess of >=31 mm/h frequency (static clutter candidates)")
    ax.set_axis_off()
    fig.savefig(os.path.join(fig_dir, "clim_excess_ge31.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return excess


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=6,
                    help="<= 6 alongside the 2-core fetcher: 8-core budget on node34")
    ap.add_argument("--top", type=int, default=50, help="hot-spot pixels to list")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["RAW_OPERA_DATA_DIR"]
    days = sorted(glob.glob(os.path.join(raw_dir, "[0-9]" * 8)))
    # skip stores the fetcher has not finished (`.zmetadata` is written last)
    days = [d for d in days if os.path.exists(os.path.join(d, ".zmetadata"))]
    if args.start or args.end:
        s, e = args.start or "0", args.end or "9"
        days = [d for d in days if s <= os.path.basename(d) <= e]
    print(f"{len(days)} days", flush=True)

    total, by_year = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(day_counts, d, cfg["PRECIP_VAR_NAME"]) for d in days]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                day, acc = f.result()
            except Exception as e:
                print(f"[{n}/{len(days)}] ERROR {e!r}", flush=True)
                continue
            _merge(total, acc)
            _merge(by_year.setdefault(day[:4], {}), {k: v for k, v in acc.items()
                                                     if not k.startswith("q_")})
            if n % 20 == 0:
                print(f"[{n}/{len(days)}]", flush=True)

    nv = np.maximum(total["n_valid"], 1)
    res = {k: v for k, v in total.items() if not k.startswith("q_")}
    for u in THRESHOLDS:
        res[f"freq_ge{u:g}"] = (total[f"n_ge{u:g}"] / nv).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        res["q_mean"] = np.where(total["q_n"] > 0, total["q_sum"] / total["q_n"],
                                 np.nan).astype(np.float32)
    for y, acc in by_year.items():
        for k, v in acc.items():
            res[f"y{y}_{k}"] = v
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "clutter_climatology.npz")
    np.savez_compressed(path, **res)
    print(f"wrote {path}")

    excess = plot_maps(res, os.path.join(args.out_dir, "figures"))
    flat = np.argsort(np.nan_to_num(excess, nan=-1).ravel())[::-1][:args.top]
    lines = ["y,x,freq_ge31,local_excess,freq_ge150,max"]
    for i in flat:
        y, x = np.unravel_index(i, excess.shape)
        lines.append(f"{y},{x},{res['freq_ge31'][y, x]:.3g},{excess[y, x]:.3g},"
                     f"{res['freq_ge150'][y, x]:.3g},{res['max'][y, x]:.4g}")
    with open(os.path.join(args.out_dir, "clutter_hotspots.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:11]))


if __name__ == "__main__":
    main()
