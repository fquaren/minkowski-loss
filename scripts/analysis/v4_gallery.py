#!/usr/bin/env python
"""Gallery of v4 samples: coarse input, 2 km target and the Minkowski targets the loss compares.

  random  100 tile-hours drawn from all four splits in proportion to their inclusion weights,
          i.e. a fair sample of real hours (the store itself over-represents heavy rain by design)
  events  100 tile-hours drawn uniformly from events_test + events_nimbus

Per sample: the coarse field (12.5x coarser, as the model sees it), the target, and the
Minkowski functionals of the target and of the interpolated input at the hourly
PHYSICAL_THRESHOLDS: area A, Crofton perimeter P and Euler characteristic chi (4-connected), the
three channels of TOPOLOGY_MODE "euler". Writes multi-page PDFs and a CSV of the samples.

    python scripts/analysis/v4_gallery.py [--n 100] [--seed 0]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

D = "/home/fquareng/work/data/extremes/OPERA"
STORE = f"{D}/patches_v4/preprocessed_dataset.zarr"
META = f"{D}/patches_v4/meta"
OUT = f"{REPO}/notes/data_analysis_v4"
GROUPS = {"train": "full_train", "validation": "full_val", "test": "full_test", "nimbus": "full_nimbus"}
TARGET_C, INPUT_C = "#2a78d6", "#eb6834"         # dataviz palette slots 1-2, fixed order
MUTED = "#6b6b66"


def draw_samples(n, seed):
    rng = np.random.default_rng(seed)
    infos = []
    for g, stem in GROUPS.items():
        i = pd.read_csv(f"{META}/{stem}_info.csv.gz", dtype={"timestamp": str, "event": str})
        infos.append(i.assign(group=g))
    allr = pd.concat(infos, ignore_index=True)
    nonev = allr[allr.event.fillna("") == ""]
    p = nonev.weight.values / nonev.weight.sum()
    rnd = nonev.iloc[rng.choice(len(nonev), n, replace=False, p=p)].assign(kind="random")
    ev = allr[allr.event.fillna("") != ""]
    evs = ev.iloc[rng.choice(len(ev), n, replace=False)].assign(kind="events")
    return rnd, evs


def minkowski(field, thr, pixel_km=2.0):
    from src.data.gamma import compute_gamma_matrix
    g = compute_gamma_matrix(np.nan_to_num(field, nan=0.0), thr, pixel_km, 0.0, 0.0)
    return g[0], g[1], g[4]                          # A, P_crofton, chi_exact


def gallery(rows, thr, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    import zarr
    from src.data.geo import tile_location
    z = zarr.open(STORE, mode="r")
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("white"); cmap.set_bad("#d8d8d4")
    norm = LogNorm(vmin=0.1, vmax=80)
    rec = []
    with PdfPages(path) as pdf:
        for p0 in range(0, len(rows), 4):
            fig, axs = plt.subplots(4, 5, figsize=(13, 11.5), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1, 1, 1.1, 1.1, 1.1]})
            for r, (_, s) in enumerate(rows.iloc[p0:p0 + 4].iterrows()):
                G = z[s.group]
                tgt = G["original_precip"][int(s.store_row)]
                crs = G["coarse_precip"][int(s.store_row)]
                itp = G["interpolated_precip"][int(s.store_row)]
                At, Pt, Ct = minkowski(tgt, thr)
                Ai, Pi, Ci = minkowski(itp, thr)
                loc = tile_location(int(s.row), int(s.col))
                ts = pd.Timestamp(s.timestamp[:12])
                ax = axs[r]
                # zeros must render as "under" (white), not as LogNorm's invalid (grey = no data)
                show = lambda f: np.where(np.isnan(f), np.nan, np.maximum(f, 1e-6))
                ax[0].imshow(show(crs), cmap=cmap, norm=norm, interpolation="nearest")
                im = ax[1].imshow(show(tgt), cmap=cmap, norm=norm, interpolation="nearest")
                for a in ax[:2]:
                    a.set_xticks([]); a.set_yticks([])
                ax[0].set_ylabel(f"{ts:%Y-%m-%d %H:%M} UTC\n{loc['lat']:.1f}N {loc['lon']:.1f}E\n"
                                 f"{s.group}{', ' + s.event if isinstance(s.event, str) and s.event else ''}",
                                 fontsize=7)
                ax[0].set_title(f"coarse input (max {np.nanmax(crs):.1f} mm)", fontsize=7)
                ax[1].set_title(f"target 2 km (max {np.nanmax(tgt):.1f} mm)", fontsize=7)
                for a, (yt, yi), lab in ((ax[2], (At, Ai), "area A (km²)"), (ax[3], (Pt, Pi), "perimeter P (km)"),
                                         (ax[4], (Ct, Ci), "Euler characteristic χ")):
                    a.plot(thr, yt, color=TARGET_C, lw=1.5, marker="o", ms=2.5, label="target")
                    a.plot(thr, yi, color=INPUT_C, lw=1.5, ls="--", marker="o", ms=2.5, label="interpolated input")
                    a.set_xscale("log"); a.set_title(lab, fontsize=7); a.tick_params(labelsize=6)
                    a.grid(alpha=0.3); a.set_xlabel("threshold (mm in 1 h)", fontsize=6)
                    if lab.startswith("area"):
                        a.set_yscale("symlog", linthresh=4); a.set_ylim(bottom=0)
                if r == 0:
                    ax[2].legend(fontsize=6, frameon=False)
                rec.append({"kind": s.kind, "group": s.group, "store_row": int(s.store_row), "timestamp": s.timestamp,
                            "row": int(s.row), "col": int(s.col), "lat": round(loc["lat"], 2), "lon": round(loc["lon"], 2),
                            "event": s.event if isinstance(s.event, str) else "", "max": float(np.nanmax(tgt)),
                            "mean": float(np.nanmean(tgt)), "wet_frac": float(np.nanmean(tgt >= 0.1)),
                            "chi_target_at_1.35": float(Ct[5]), "chi_input_at_1.35": float(Ci[5])})
            fig.colorbar(im, ax=axs[:, 1], shrink=0.5, location="right",
                         label="mm in 1 h (log; white = 0, grey = no data)")
            fig.suptitle(f"{title} — samples {p0 + 1}-{min(p0 + 4, len(rows))} of {len(rows)}", fontsize=10)
            pdf.savefig(fig); plt.close(fig)
    return pd.DataFrame(rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    thr = np.load(f"{D}/patches_v4/physical_thresholds.npy")
    import yaml
    cfg_thr = np.array(yaml.safe_load(open(f"{REPO}/configs/config_v4.yaml"))["PHYSICAL_THRESHOLDS"], float)
    if not np.allclose(thr, cfg_thr):                # the store still has the 15-min levels until gamma runs
        thr = cfg_thr
    rnd, evs = draw_samples(a.n, a.seed)
    r1 = gallery(rnd.sort_values("timestamp"), thr, f"{OUT}/gallery_random_hours.pdf",
                 "Random hours (weighted by inclusion, all splits)")
    r2 = gallery(evs.sort_values("timestamp"), thr, f"{OUT}/gallery_event_hours.pdf", "Event hours (events_test + events_nimbus)")
    pd.concat([r1, r2]).to_csv(f"{OUT}/gallery_samples.csv", index=False)
    print(pd.concat([r1, r2]).groupby("kind")[["max", "mean", "wet_frac"]].describe().round(2).T)


if __name__ == "__main__":
    main()
