"""R4 scorecard v2: the deterministic losses and the FM rows of EXPERIMENTS G4, full and cut training lists, one subset.

Cells print the metric; colour is the % difference of the metric's error against the reference row (MSE, full
list, seed 42), blue = better. Errors: the metric itself for lower-is-better columns; |ln x| for ratios whose
ideal is 1 (exceedance ratio, anisotropy), so over- and under-prediction by the same factor score alike; for
FSS (higher is better) the colour is -(FSS / FSS_ref - 1).
    python scripts/evaluate/r4_scorecard.py --subset ext --out figures/r4_scorecard_valext
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap

BASE = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/eval_results/extremes"

REF = ("MSE", {"full": "r4_mse_full_s42_v100", "cut": "r4_mse_cut_s42"})
# Optical flow is left out (2026-11-16): with co-located input and target its teacher is the block-upsampled
# coarse input, so the loss trains the U-Net toward the input blocks (EXPERIMENTS §3b).
# Scorecard v2 rows (EXPERIMENTS G4, 2026-11-21). SSIM, wet area and unwindowed spectral are not reported (G0).
LOSSES = [
    ("MSE + QM", {"full": "r4_mse_qm_full_s42", "cut": "r4_mse_qm_cut_s42"}),
    ("MSE random-drop", {"cut": "r4_mse_randdrop_s42"}),
    ("wmse", {"full": "r4_wmse_full_s42", "cut": "r4_wmse_cut_s42"}),
    ("quantile", {"full": "r4_quantile_full_s42", "cut": "r4_quantile_cut_s42"}),
    ("fss", {"full": "r4_fss_full_s42", "cut": "r4_fss_cut_s42"}),
    ("windowed spectral 0.51", {"full": "r4_spectral_hann_full_s42", "cut": "r4_spectral_hann_cut_s42"}),
    ("Minkowski area-only", {"full": "r4_area_full_s42", "cut": "r4_area_cut_s42"}),
    ("Minkowski", {"full": "r4_mink_full_s42", "cut": "r4_mink_cut_s42"}),
]
# FM area-only (w 41) aborted after epoch 1 in both arms: left out until its bracket picks a weight (E3).
FMS = [
    ("FM, clean", {"full": "fm_r4_clean_full", "cut": "fm_r4_clean_cut"}),
    ("FM, control (w = 0)", {"full": "fm_energy_ctl_cond", "cut": "fm_r4_ctl_cut_s42"}),
    ("FM + Minkowski", {"full": "fm_energy_qm_cond", "cut": "fm_r4_mink_cut_s42"}),
]


def _cond(s, k):
    return s["conditioned"]["obs_selected"][k] if "conditioned" in s else float("nan")


# (title, subtitle, getter, error kind, format)
METRICS = [
    ("RMSE", "all pixels [mm]", lambda s: s["rmse_overall"], "low", "{:.3f}"),
    ("twCRPS", "extreme, threshold-weighted", lambda s: s.get("twcrps_extreme", float("nan")), "low", "{:.4f}"),
    ("FSS", "u = 50.1 mm, window 11", lambda s: s["fss_thr50.1_win11"], "high", "{:.3f}"),
    ("Exceedance ratio", "obs. max >= u (ideal 1)", lambda s: _cond(s, "exceedance_ratio"), "ln", "{:.3f}"),
    ("Peak ratio", "pred/obs max, extremes (ideal 1)", lambda s: s["peak_ratio_extreme_median"], "ln", "{:.2f}"),
    ("Hit rate", "pred >= u | obs >= u", lambda s: s["conditioned"]["hit_rate"] if "conditioned" in s
     else float("nan"), "high", "{:.3f}"),
    ("False alarms", "pred >= u | obs < u",
     lambda s: s["conditioned"]["false_alarm_rate"] if "conditioned" in s else float("nan"), "fa", "{:.1e}"),
    ("RAPSD", "log-spectral distance", lambda s: s["rapsd_log_distance"], "low", "{:.2f}"),
    ("Anisotropy", "axis / diagonal (ideal 1)", lambda s: s["anisotropy_axis_over_diag"], "ln", "{:.2f}"),
]
ARMS = ("full", "cut")
EDGES = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]


def _err(v, kind):
    if kind == "ln":
        return abs(math.log(v)) if v > 0 else float("inf")
    return v


def pct(v, ref, kind, fa_floor=0.0):
    if not (np.isfinite(v) and np.isfinite(ref)):
        return float("nan")
    if kind == "fa":            # the reference may have no false alarm at all: floor it at one case
        ref, v, kind = max(ref, fa_floor), max(v, fa_floor), "low"
    if kind == "high":
        return -100.0 * (v / ref - 1.0)
    e, er = _err(v, kind), _err(ref, kind)
    if er == 0:
        return float("nan")
    return 100.0 * (e / er - 1.0) if np.isfinite(e) else 1e9


def load(subset, tag):
    p = f"{BASE}/v4adv_val{subset}/{tag}/extremes_summary.yaml"
    return yaml.safe_load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["ext", "sub"], default="ext")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [REF] + LOSSES + FMS
    S = {(r, a): load(args.subset, t) for r, tags in rows for a, t in tags.items()}
    ref_full = S[(REF[0], "full")]
    if ref_full is None:
        raise SystemExit("reference row missing")
    xi = {round(s["gpd_xi_obs"], 4) for s in S.values() if s}
    if len(xi) != 1:
        raise SystemExit(f"gpd_xi_obs differs across rows: {xi}")   # POT level / patch set not shared
    n = ref_full["n_samples"]
    c = ref_full["conditioned"]
    fa_floor = 1.0 / (c["n_cases"] - c["n_obs_selected"])

    cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.02, 0.98, len(EDGES) + 1)))
    norm = BoundaryNorm(EDGES, cmap.N, extend="both")
    nm, nr = len(METRICS), len(rows)
    cw, ch, gap = 0.62, 0.42, 0.35
    fig_w = 2.6 + nm * (2 * cw + gap)
    H = 1.95 + nr * ch + 1.0
    fig = plt.figure(figsize=(fig_w, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, fig_w); ax.set_ylim(0, H); ax.axis("off")
    top = 1.95 + nr * ch
    x0 = 2.6
    groups = [(0, 1, "#eef1f8", ""), (1, 1 + len(LOSSES), "#fbe5e5", "deterministic U-Net,\nauxiliary losses"),
              (nr - len(FMS), nr, "#e9f3e9", "flow matching,\nreward fine-tunes")]
    for a, b, col, lab in groups:
        y1, y0 = top - a * ch, top - b * ch
        ax.add_patch(plt.Rectangle((0.05, y0 + 0.03), x0 - 0.1, y1 - y0 - 0.06, color=col, ec="#e3b3b3" if lab else col))
        if lab:
            ax.text(0.18, (y0 + y1) / 2, lab, rotation=90, ha="center", va="center", fontsize=8)
    for i, (rname, _) in enumerate(rows):
        ax.text(x0 - 0.1, top - (i + 0.5) * ch, rname, ha="right", va="center", fontsize=10)
    for j, (title, sub, get, kind, fmt) in enumerate(METRICS):
        gx = x0 + j * (2 * cw + gap)
        ax.text(gx + cw, top + 0.62, title, ha="center", va="bottom", fontsize=11)
        ax.text(gx + cw, top + 0.42, sub, ha="center", va="bottom", fontsize=6.5)
        refv = get(ref_full)
        for k, arm in enumerate(ARMS):
            cx = gx + k * cw
            ax.text(cx + cw / 2, top - nr * ch - 0.12, arm, ha="center", va="top", fontsize=9)
            for i, (rname, tags) in enumerate(rows):
                y = top - (i + 1) * ch
                if arm not in tags:
                    continue
                s = S[(rname, arm)]
                if s is None:
                    ax.add_patch(plt.Rectangle((cx + 0.02, y + 0.02), cw - 0.04, ch - 0.04, fc="white",
                                               ec="#bbbbbb", hatch="///", lw=0.6))
                    ax.text(cx + cw / 2, y + ch / 2, "pending", ha="center", va="center", fontsize=6, color="#777")
                    continue
                v = get(s)
                p = 0.0 if (rname, arm) == (REF[0], "full") else pct(v, refv, kind, fa_floor)
                fc = "#f2f2f2" if not np.isfinite(p) else cmap(norm(np.clip(p, -1e3, 1e3)))
                ax.add_patch(plt.Rectangle((cx + 0.02, y + 0.02), cw - 0.04, ch - 0.04, fc=fc, ec="#bbbbbb", lw=0.6))
                txt = fmt.format(v) if np.isfinite(v) else "n/a"
                dark = np.isfinite(p) and abs(p) >= 20
                ax.text(cx + cw / 2, y + ch / 2, txt, ha="center", va="center", fontsize=7.2,
                        color="white" if dark else "black")
        ax.text(gx + cw, top - nr * ch - 0.38, "training list", ha="center", va="top", fontsize=8)
    cax = fig.add_axes([0.32, 0.9 / H, 0.4, 0.13 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=EDGES, extend="both")
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Better  ←  % difference in error vs MSE (full list)  →  Worse", fontsize=9)
    name = {"ext": "val_extremes1024", "sub": "val_subset4096"}[args.subset]
    ax.text(0.1, 0.06, f"{name}, n = {n}; POT u = 50.1 mm; seed 42 for every row; U-Nets unet_latest, FM 16 members "
            f"heun-16. Error for the colour: the metric (low is better), |ln x| for ratios with ideal 1, FSS "
            f"inverted; false alarms floored at one case.\nSingle seed: across 3 seeds (full val) the exceedance ratio spreads by ~0.012 (MSE) and "
            f"~0.018 (Minkowski), hit rate by up to 0.08. Cut = no training tile-hour >= 50.1 mm. twCRPS of a single field is its threshold-weighted MAE.",
            fontsize=7, va="bottom")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".png", dpi=160)

    with open(args.out + ".md", "w") as f:
        f.write(f"# R4 scorecard, {name} (n = {n}, u = 50.1 mm, seed 42)\n\n")
        hdr = ["model"] + [f"{m[0]} {a}" for m in METRICS for a in ARMS]
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for rname, tags in rows:
            cells = [rname]
            for title, sub, get, kind, fmt in METRICS:
                for arm in ARMS:
                    s = S.get((rname, arm)) if arm in tags else None
                    cells.append(fmt.format(get(s)) if s else ("pending" if arm in tags else ""))
            f.write("| " + " | ".join(cells) + " |\n")
    print("wrote", args.out + ".png", args.out + ".md")


if __name__ == "__main__":
    main()
