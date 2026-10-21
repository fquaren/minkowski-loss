#!/usr/bin/env python
"""Every correction the hourly chain made to one tile-hour, frame by frame.

Replays the day exactly as `build_store_hourly.py` does (full grid, `configs/quality_v4.yaml`, the
store validator's construction), then for each of the hour's four 15-min frames (H-45 .. H) and the
tile's 128 x 128 pixels reports:
  - `clean_frame` (per-frame, uncoded): pixels changed between raw and cleaned, split into hot-mask
    pixels (static clutter filled from neighbours) and other pixels (isolated spikes);
  - each coded repair (`code` bits: footprint, ray, ring, unsupported, ceiling, persistence, burst,
    stationary) and the uncorrected marks (refused, hot kept): pixels, rate before / after;
  - the frame before and after: max, wet pixels, valid pixels;
and for the hour: raw sum, screened plain sum, advection-corrected sum.

    python scripts/data_quality/tile_hour_corrections.py --end 202409132300 --row 896 --col 1024
      (-> notes/data_analysis_v4/tile_corrections/<end>_<row>_<col>.md / .png)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2")]
D = "/home/fquareng/work/data/extremes/OPERA"
Q2 = f"{D}/quality_v2"
OUT = f"{REPO}/notes/data_analysis_v4/tile_corrections"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--end", required=True, help="hour end, YYYYMMDDHHMM (UTC)")
    ap.add_argument("--row", type=int, required=True, help="tile's south row")
    ap.add_argument("--col", type=int, required=True, help="tile's west column")
    ap.add_argument("--quality", default=f"{REPO}/configs/quality_v4.yaml")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import scan_tiles as st
    from build_store_hourly import ADVECTION
    from src.data import cleaning as C
    from src.data import geo
    from src.data.hourly import HourlyDay

    end = pd.Timestamp(f"{a.end[:4]}-{a.end[4:6]}-{a.end[6:8]} {a.end[8:10]}:{a.end[10:12]}")
    day = (end - pd.Timedelta(minutes=1)).strftime("%Y%m%d")
    e_ns = int(np.datetime64(end, "ns").astype(np.int64))
    sl = (slice(a.row, a.row + 128), slice(a.col, a.col + 128))
    q, ceil = st._quality(a.quality)
    hot_fn = lambda y: st._hot(f"{Q2}/clutter_climatology.npz", y)  # noqa: E731
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=hot_fn,
                   ring=lambda y: st._ring(f"{Q2}/ring_mask.npz", y), sites_rc=st._sites(), ceilings=ceil,
                   cfg={**q.get("hourly", {}), "advection": ADVECTION},
                   adv_boxes={e_ns: [(a.row, a.row + 128, a.col, a.col + 128)]}).run()
    h = next(x for x in hd.hours if int(np.datetime64(x["end"], "ns").astype(np.int64)) == e_ns)
    hot = hot_fn(int(day[:4]))
    hot_t = hot[sl] if hot is not None else np.zeros((128, 128), bool)
    cfg = {**q.get("hourly", {})}
    bits = [("footprint", C.REPAIR_FOOTPRINT), ("ceiling", C.REPAIR_CEILING), ("ring", C.REPAIR_RING),
            ("ray", C.REPAIR_RAY), ("persistence", C.REPAIR_PERSIST), ("stationary", C.REPAIR_STATIONARY),
            ("unsupported", C.REPAIR_UNSUPPORTED), ("burst", C.REPAIR_BURST),
            ("refused (left as is)", C.REPAIR_REFUSED), ("hot kept (left as is)", C.REPAIR_HOT_KEPT)]
    rows, frames = [], []
    for t in h["idx"]:
        ts = pd.Timestamp(hd.frames[t]).strftime("%Y-%m-%d %H:%M")
        raw = hd.raw[t][sl]
        cf, _ = C.clean_frame(hd.raw[t], hot, hot_fallback=cfg.get("hot_fallback", "zero"),
                              spikes=cfg.get("rules", {}).get("spike", True))
        cf = cf[sl]
        fin = hd.fin[t][sl]
        code = hd.code[t][sl]
        frames.append((ts, raw, fin, code))
        ch = ~np.isclose(np.nan_to_num(raw, nan=-1), np.nan_to_num(cf, nan=-1))
        drz = ch & (np.nan_to_num(raw) < C.DRIZZLE)
        for name, m in (("clean_frame: drizzle floor (< 0.1 mm/h set to 0)", drz),
                        ("clean_frame: hot-mask pixel filled", ch & ~drz & hot_t),
                        ("clean_frame: isolated spike (>= 10 mm/h, neighbours < 10%)", ch & ~drz & ~hot_t)):
            if m.any():
                rows.append({"frame": ts, "correction": name, "pixels": int(m.sum()),
                             "max before": float(np.nanmax(raw[m])), "max after": float(np.nanmax(np.nan_to_num(cf[m]))),
                             "mean change": float(np.nanmean(np.nan_to_num(cf[m]) - np.nan_to_num(raw[m])))})
        for name, b in bits:
            m = (code & b) > 0
            if m.any():
                rows.append({"frame": ts, "correction": name, "pixels": int(m.sum()),
                             "max before": float(np.nanmax(raw[m])), "max after": float(np.nanmax(np.nan_to_num(fin[m]))),
                             "mean change": float(np.nanmean(np.nan_to_num(fin[m]) - np.nan_to_num(raw[m])))})
        rows.append({"frame": ts, "correction": "== frame total (raw -> final)",
                     "pixels": int((~np.isclose(np.nan_to_num(raw, nan=-1), np.nan_to_num(fin, nan=-1))).sum()),
                     "max before": float(np.nanmax(raw)) if np.isfinite(raw).any() else np.nan,
                     "max after": float(np.nanmax(fin)) if np.isfinite(fin).any() else np.nan,
                     "mean change": float(np.nanmean(np.nan_to_num(fin) - np.nan_to_num(raw))),
                     "valid": int(np.isfinite(raw).sum()), "wet >= 0.1 before": int((np.nan_to_num(raw) >= 0.1).sum()),
                     "wet >= 0.1 after": int((np.nan_to_num(fin) >= 0.1).sum())})
    t = pd.DataFrame(rows)
    raw_sum = np.nansum([0.25 * f[1] for f in frames], axis=0)
    plain = np.nan_to_num(h["sum"][sl], nan=0.0)
    adv = h["sum_adv"][sl]
    loc = geo.tile_location(a.row, a.col)
    from scripts.data_quality._md import md_table
    os.makedirs(OUT, exist_ok=True)
    stem = f"{OUT}/{a.end}_{a.row}_{a.col}"
    L = [f"# Corrections to tile-hour {end:%Y-%m-%d %H:%M} UTC, tile ({a.row}, {a.col}) {a.label}", "",
         f"Tile centre {loc['lat']:.2f}N {loc['lon']:.2f}E; frames {', '.join(f[0][11:] for f in frames)} "
         f"(the hour H-45 .. H); replay of `HourlyDay` with `{os.path.relpath(a.quality, REPO)}`.", "",
         "| hour | max (mm in 1 h) | mean | pixels >= 1 mm |", "|---|---|---|---|",
         f"| raw sum | {raw_sum.max():.2f} | {raw_sum.mean():.4f} | {(raw_sum >= 1).sum()} |",
         f"| screened, plain sum | {plain.max():.2f} | {plain.mean():.4f} | {(plain >= 1).sum()} |",
         f"| screened, advection-corrected (the v4 target) | {np.nanmax(adv):.2f} | {np.nanmean(adv):.4f} | "
         f"{(np.nan_to_num(adv) >= 1).sum()} |", "",
         "## Per frame and correction (rates in mm/h)", "", md_table(t, index=False, floatfmt=".3g"), "",
         "Rules that act on whole tile-hours (radar-day exclusion, ownerless, flagged radar-years) remove a "
         "tile-hour instead of changing pixels; this one is in the store, so none of them applied."]
    open(f"{stem}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    _figure(frames, raw_sum, plain, adv, f"{stem}.png", f"{end:%Y-%m-%d %H:%M} UTC, tile ({a.row}, {a.col}) {a.label}")


def _figure(frames, raw_sum, plain, adv, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    fig, ax = plt.subplots(3, 5, figsize=(17, 10.5))
    norm = LogNorm(0.1, 100)
    show = lambda z: np.where(np.nan_to_num(z) >= 0.1, z, np.nan)  # noqa: E731
    for j, (ts, raw, fin, code) in enumerate(frames):
        ax[0, j].imshow(show(raw), cmap="Blues", norm=norm, origin="lower", interpolation="nearest")
        ax[0, j].set_title(f"raw {ts[11:]} (max {np.nanmax(raw):.1f} mm/h)", fontsize=8)
        im = ax[1, j].imshow(show(fin), cmap="Blues", norm=norm, origin="lower", interpolation="nearest")
        ax[1, j].set_title(f"screened {ts[11:]} (max {np.nanmax(fin):.1f})", fontsize=8)
        ch = ~np.isclose(np.nan_to_num(raw, nan=-1), np.nan_to_num(fin, nan=-1))
        drz = ch & (np.nan_to_num(raw) < 0.1)
        ax[2, j].imshow(np.isfinite(raw), cmap=ListedColormap(["#d9d9d9", "#ffffff"]), vmin=0, vmax=1,
                        origin="lower", interpolation="nearest")
        yy, xx = np.nonzero(drz)
        ax[2, j].scatter(xx, yy, s=2, c="#b0c4de", marker="s", linewidths=0, label="drizzle floor")
        yy, xx = np.nonzero(ch & ~drz)
        ax[2, j].scatter(xx, yy, s=14, facecolors="none", edgecolors="#000000", linewidths=0.8,
                         label="repaired (persistence, spike, ...)")
        ax[2, j].set_title(f"changed {ts[11:]}: {int(drz.sum())} drizzle, {int((ch & ~drz).sum())} repaired",
                           fontsize=8)
        if j == 0:
            ax[2, j].legend(fontsize=6, loc="lower right")
    hn = LogNorm(0.1, 50)
    for i, (z, ttl) in enumerate(((raw_sum, "raw sum"), (plain, "screened plain sum"))):
        ax[i, 4].imshow(show(z), cmap="Blues", norm=hn, origin="lower", interpolation="nearest")
        ax[i, 4].set_title(f"{ttl} (max {np.nanmax(z):.1f} mm)", fontsize=8)
    im2 = ax[2, 4].imshow(show(adv), cmap="Blues", norm=hn, origin="lower", interpolation="nearest")
    ax[2, 4].set_title(f"advection-corrected sum, target (max {np.nanmax(adv):.1f} mm)", fontsize=8)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.colorbar(im, ax=ax[:2, :4], fraction=0.02, label="15-min rate (mm/h)")
    fig.colorbar(im2, ax=ax[:, 4], fraction=0.04, label="mm in 1 h")
    fig.suptitle(title + "  (north up, 256 x 256 km)", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
