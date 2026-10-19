#!/usr/bin/env python
"""Plain against advection-corrected hourly field, same hours (DECISIONS §35; methods §6.5 check).

Samples from the advection store's metadata (rows with a complete corrected field):
  random      40 tile-hours, weighted by the store's inclusion weights (fair sample of hours)
  convective  40 summer (JJA) tile-hours with corrected max >= 20.2 mm: where fast growth and
              decay would break the correction's assumption
  events      20 tile-hours from events_test / events_nimbus
Per sample: plain sum | corrected field | corrected - plain | Euler characteristic of both against
the hourly thresholds. Writes notes/data_analysis_v4/gallery_plain_vs_adv.pdf.

    python scripts/analysis/v4_gallery_compare.py
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
D = "/home/fquareng/work/data/extremes/OPERA"
PLAIN, ADV = f"{D}/patches_v4/preprocessed_dataset.zarr", f"{D}/patches_v4_adv/preprocessed_dataset.zarr"
META = f"{D}/patches_v4_adv/meta"
OUT = f"{REPO}/notes/data_analysis_v4"
GROUPS = {"train": "full_train", "validation": "full_val", "test": "full_test", "nimbus": "full_nimbus"}


def samples(seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for g, stem in GROUPS.items():
        m = pd.read_csv(f"{META}/{stem}.txt", header=None, names=["ts", "row", "col", "max", "sr"], dtype={"ts": str})
        w = pd.read_csv(f"{D}/patches_v4/meta/{stem}_info.csv.gz", usecols=["store_row", "weight", "event"],
                        dtype={"event": str}).set_index("store_row")
        parts.append(m.assign(group=g, weight=w.weight.reindex(m.sr).values, event=w.event.reindex(m.sr).fillna("").values))
    a = pd.concat(parts, ignore_index=True)
    ne = a[a.event == ""]
    rnd = ne.iloc[rng.choice(len(ne), 40, replace=False, p=ne.weight / ne.weight.sum())].assign(kind="random")
    jja = ne[ne.ts.str[4:6].isin(["06", "07", "08"]) & (ne["max"] >= 20.2)]
    con = jja.iloc[rng.choice(len(jja), 40, replace=False)].assign(kind="convective")
    ev = a[a.event != ""]
    evs = ev.iloc[rng.choice(len(ev), 20, replace=False)].assign(kind="event")
    return pd.concat([rnd, con, evs], ignore_index=True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    import zarr
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from src.data.gamma import compute_gamma_matrix
    from src.data.geo import tile_location
    thr = np.array(yaml.safe_load(open(f"{REPO}/configs/config_v4_adv.yaml"))["PHYSICAL_THRESHOLDS"], np.float32)
    zp, za = zarr.open(PLAIN, mode="r"), zarr.open(ADV, mode="r")
    S = samples()
    cmap = plt.get_cmap("Blues").copy(); cmap.set_under("white"); cmap.set_bad("#d8d8d4")
    norm = LogNorm(vmin=0.1, vmax=80)
    show = lambda f: np.where(np.isnan(f), np.nan, np.maximum(f, 1e-6))
    rec = []
    with PdfPages(f"{OUT}/gallery_plain_vs_adv.pdf") as pdf:
        for p0 in range(0, len(S), 4):
            fig, axs = plt.subplots(4, 4, figsize=(12, 11.5), constrained_layout=True)
            for r, (_, s) in enumerate(S.iloc[p0:p0 + 4].iterrows()):
                pl = zp[s.group]["original_precip"][int(s.sr)]
                ad = za[s.group]["original_precip"][int(s.sr)]
                cp = compute_gamma_matrix(pl, thr, 2.0, 0.0, 0.0)[4]
                ca = compute_gamma_matrix(ad, thr, 2.0, 0.0, 0.0)[4]
                loc = tile_location(int(s.row), int(s.col))
                ax = axs[r]
                ax[0].imshow(show(pl), cmap=cmap, norm=norm, interpolation="nearest")
                im = ax[1].imshow(show(ad), cmap=cmap, norm=norm, interpolation="nearest")
                d = ad - pl
                lim = max(1.0, float(np.nanpercentile(np.abs(d), 99.9)))
                imd = ax[2].imshow(d, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
                for a_ in ax[:3]:
                    a_.set_xticks([]); a_.set_yticks([])
                ax[0].set_ylabel(f"{s.kind}\n{pd.Timestamp(s.ts[:12]):%Y-%m-%d %H:%M}\n{loc['lat']:.1f}N {loc['lon']:.1f}E"
                                 + (f"\n{s.event}" if s.event else ""), fontsize=7)
                ax[0].set_title(f"plain sum (max {np.nanmax(pl):.1f} mm)", fontsize=7)
                ax[1].set_title(f"advection-corrected (max {np.nanmax(ad):.1f} mm)", fontsize=7)
                ax[2].set_title("corrected - plain (mm)", fontsize=7)
                ax[3].plot(thr, cp, color="#eb6834", ls="--", lw=1.5, marker="o", ms=2.5, label="plain")
                ax[3].plot(thr, ca, color="#2a78d6", lw=1.5, marker="o", ms=2.5, label="corrected")
                ax[3].set_xscale("log"); ax[3].set_title("Euler characteristic χ", fontsize=7)
                ax[3].tick_params(labelsize=6); ax[3].grid(alpha=0.3); ax[3].set_xlabel("threshold (mm in 1 h)", fontsize=6)
                if r == 0:
                    ax[3].legend(fontsize=6, frameon=False)
                rec.append({"kind": s.kind, "group": s.group, "store_row": int(s.sr), "ts": s.ts, "max_plain": float(np.nanmax(pl)),
                            "max_adv": float(np.nanmax(ad)), "chi_2mm_plain": float(cp[6]), "chi_2mm_adv": float(ca[6])})
            fig.colorbar(im, ax=axs[:, 1], shrink=0.4, label="mm in 1 h (log)")
            fig.colorbar(imd, ax=axs[:, 2], shrink=0.4, label="difference (mm)")
            fig.suptitle(f"Plain vs advection-corrected — samples {p0 + 1}-{min(p0 + 4, len(S))} of {len(S)}", fontsize=10)
            pdf.savefig(fig); plt.close(fig)
    t = pd.DataFrame(rec)
    t.to_csv(f"{OUT}/gallery_plain_vs_adv.csv", index=False)
    print(t.groupby("kind")[["max_plain", "max_adv", "chi_2mm_plain", "chi_2mm_adv"]].median().round(2))


if __name__ == "__main__":
    main()
