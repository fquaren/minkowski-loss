#!/usr/bin/env python
"""Main result table (DECISIONS §43, §47) from eval_extremes outputs on the FM subsets, u = 50.1.

Table 1, one row per model (seed rows averaged):
  bulk R0 (val_subset4096): member MAE and its change against the block baseline, CRPS, anisotropy
  R1 full arm / R4 cut arm (val_extremes1024): exceedance ratio, median peak ratio, q90 ratio of patch maxima,
      false-alarm ratio, twCRPS per observed tail pixel
  degradation interaction: [log X(cut) - log X(full)] - the same for the block baseline (MSE for U-Nets, FM control
      for FM), with a 95% paired day-block bootstrap interval; reported as a percentage, exp(.) - 1.
      Every R4 model is scored on the same patches as its full-arm twin and as the baseline, so one day resample
      is applied to all four columns at once.
Table 2 (appendix): FSS, hit rate, input-selected exceedance, q50 ratio, FAR on valsub, exceedance by
  distance above the cut, and the random-drop control.

FM tail scores are averages over the 16 members (`table` block of the summary). A row is filled from
<out>/v4adv_val{ext,sub}_tab/<tag>, else from the old directory of the same name (the G3 queue writes there).

  python scripts/analysis/main_table.py --out <dir> [--B 2000]
"""
import argparse
import os

import numpy as np
import yaml

from src.evaluation.day_block import processing_days

OUTE = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/eval_results/extremes"
META = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta"
U = 50.1

# (row name, block, [(full tag, cut tag) per seed])
DET = [("MSE", ["mse_s42", "mse_s43", "mse_s44"]), ("MSE + QM", ["mse_qm_s42"]), ("wMSE", ["wmse_s42"]),
       ("quantile", ["quantile_s42"]), ("FSS", ["fss_s42"]), ("spectral (Hann) †", ["spectralhann0.051_s42"]),
       ("area only", ["area_s42"]), ("Minkowski", ["mink_s42", "mink_s43", "mink_s44"])]
GEN = [("FM clean", ["fm_clean"]), ("FM control", ["fm_ctl_s42", "fm_ctl_s43", "fm_ctl_s44"]),
       ("FM + pixel ES", ["fm_g3_pixel@s42"]), ("FM + tw-pixel ES", ["fm_g3_pixeltw@s42"]),
       ("FM + FSS", ["fm_g3_fss@s42"]), ("FM + spectral (Hann) †", ["fm_g3_spectralhann@s42"]),
       ("FM + area only", ["fm_g3_area@s42"]), ("FM + Minkowski", ["fm_mink_s42", "fm_mink_s43", "fm_mink_s44"])]
BLOCKS = [("Deterministic (U-Net)", DET, "MSE"), ("Generative (flow matching)", GEN, "FM control")]
TAIL = [("exc", "Exc"), ("peak", "Peak"), ("q90", "q90"), ("far", "FAR"), ("tw", "twCRPS")]
DEGR = [("exc", "ΔExc"), ("peak", "ΔPeak"), ("q90", "Δq90")]


def tags(t, arm):
    if "@" in t:                                  # G3: fm_g3_<row>_<arm>_<seed>
        r, s = t.split("@"); return f"{r}_{arm}_{s}"
    return f"{t}_{arm}"


def find(tag, sub):
    for d in (f"{OUTE}/v4adv_val{sub}_tab/{tag}", f"{OUTE}/v4adv_val{sub}/{tag}"):
        p = f"{d}/extremes_summary.yaml"
        if os.path.isfile(p):
            s = yaml.safe_load(open(p))
            if "table" in s:
                return s, dict(np.load(f"{d}/extremes_arrays.npz"))
    return None


def scores(a, idx):
    """Tail scores on the cases idx (with repeats), averaged over members."""
    tmax, nobs = a["tmax"][idx], a["n_exc_obs"][idx].astype(float)
    pm, ne = a["pred_max_members"][:, idx], a["n_exc_pred_members"][:, idx].astype(float)
    ev = tmax >= U
    alarm = pm >= U
    far = np.where(alarm.sum(1) > 0, (alarm & ~ev).sum(1) / np.maximum(alarm.sum(1), 1), np.nan)
    return {"exc": float(np.mean(ne[:, ev].sum(1) / max(nobs[ev].sum(), 1.0))),
            "peak": float(np.mean(np.median(pm[:, ev] / tmax[ev], axis=1))),
            "q90": float(np.mean(np.quantile(pm[:, ev], 0.9, axis=1) / np.quantile(tmax[ev], 0.9))),
            "far": float(np.nanmean(far)) if np.isfinite(far).any() else np.nan}


def load_row(seeds):
    """Per seed: (full ext, cut ext, full sub, cut sub) or None when any part is missing."""
    out = []
    for t in seeds:
        parts = [find(tags(t, arm), sub) for sub in ("ext", "sub") for arm in ("full", "cut")]
        out.append(parts if all(parts) else None)
    return out


def fmt(x, f="{:.3f}"):
    return "–" if x is None or not np.isfinite(x) else f.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    days = processing_days(f"{META}/val_extremes1024.txt", 1024)
    dcodes = np.unique(days, return_inverse=True)[1]
    D = dcodes.max() + 1
    by_day = [np.flatnonzero(dcodes == d) for d in range(D)]
    rng = np.random.default_rng(0)
    resamples = [np.concatenate([by_day[d] for d in rng.integers(0, D, D)]) for _ in range(args.B)]
    full_idx = np.arange(1024)

    def log_degr(row, idx):               # {metric: mean over complete seeds of log X(cut) - log X(full)}
        sc = [(scores(p[0][1], idx), scores(p[1][1], idx)) for p in row if p]
        with np.errstate(divide="ignore", invalid="ignore"):
            return {k: (np.mean([np.log(c[k]) - np.log(f[k]) for f, c in sc]) if sc else np.nan) for k, _ in DEGR}

    def degr_boot(row):                   # point {k: float}, bootstrap {k: array[B]}
        pt = log_degr(row, full_idx)
        bs = [log_degr(row, i) for i in resamples]
        return pt, {k: np.array([b[k] for b in bs]) for k, _ in DEGR}

    lines, csv = [], []
    head1 = ("| model | seeds | MAE | ΔMAE | CRPS | aniso | " + " | ".join(f"{n} R1" for _, n in TAIL) + " | "
             + " | ".join(f"{n} R4" for _, n in TAIL) + " | " + " | ".join(n for _, n in DEGR) + " |")
    lines += ["# Main table on validation (u = 50.1 mm/h; valsub for bulk, valext for the tail)", "", head1,
              "|" + "---|" * (head1.count("|") - 1)]
    app = ["", "# Appendix table", "",
           "| model | arm | FSS@50.1 w11 | hit | Exc (input-sel.) | q50 | FAR valsub | Exc 50-80 | Exc 80-120 | Exc >120 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for bname, rows, base in BLOCKS:
        lines.append(f"| **{bname}** |" + " |" * (head1.count("|") - 2))
        loaded = {name: load_row(seeds) for name, seeds in rows}
        brow = loaded[base]
        bmae = np.mean([p[2][0]["table"]["mae_member"] for p in brow if p]) if any(brow) else np.nan
        bd_point, bd_boot = degr_boot(brow)
        for name, seeds in rows:
            row = loaded[name]
            ok = [p for p in row if p]
            if not ok:
                lines.append(f"| {name} | 0/{len(seeds)} |" + " – |" * (head1.count("|") - 3))
                continue
            T = lambda i, k: np.mean([p[i][0]["table"][k] for p in ok])
            mae = T(2, "mae_member")
            crps = np.mean([p[2][0].get("crps_mean", p[2][0]["table"]["mae_member"]) for p in ok])
            cells = [name, f"{len(ok)}/{len(seeds)}", fmt(mae, "{:.4f}"),
                     fmt(100 * (mae / bmae - 1), "{:+.1f}%"), fmt(crps, "{:.4f}"), fmt(T(2, "anisotropy_member"), "{:.2f}")]
            for arm in (0, 1):
                s = {k: np.mean([scores(p[arm][1], full_idx)[k] for p in ok]) for k in ("exc", "peak", "q90", "far")}
                s["tw"] = T(arm, "twcrps_per_obs_tail_pixel")
                cells += [fmt(s["exc"]), fmt(s["peak"], "{:.2f}"), fmt(s["q90"], "{:.2f}"), fmt(s["far"], "{:.2f}"),
                          fmt(s["tw"], "{:.1f}")]
            rpt, rbs = degr_boot(row) if name != base else (None, None)
            for k, _ in DEGR:
                if name == base:
                    cells.append(f"{fmt(100 * (np.exp(bd_point[k]) - 1), '{:+.0f}%')} (base)")
                    continue
                pt = rpt[k] - bd_point[k]
                bs = rbs[k] - bd_boot[k]
                # a ratio that is 0 in a resample gives log = -inf; inverted_cdf keeps +-inf instead of
                # interpolating inf - inf into nan, and nan (both rows 0) is dropped
                lo, hi = np.nanquantile(bs, [0.025, 0.975], method="inverted_cdf")
                pc = lambda x: 100 * (np.exp(x) - 1)
                pf = lambda x: ("+inf" if x > 0 else "-100") if np.isinf(x) else f"{pc(x):+.0f}"
                cells.append(f"{pf(pt)}% [{pf(lo)}, {pf(hi)}]")
                csv.append((name, k, pc(pt), pc(lo), pc(hi), len(ok)))
            lines.append("| " + " | ".join(cells) + " |")
            for arm, an in ((0, "full"), (1, "cut")):
                sm, a = ok[0][arm]
                bins = []
                for lo_, hi_ in ((50.1, 80), (80, 120), (120, np.inf)):
                    sel = (a["tmax"] >= lo_) & (a["tmax"] < hi_)
                    bins.append(np.mean([p[arm][1]["n_exc_pred_members"][:, sel].sum(1).mean()
                                         / max(p[arm][1]["n_exc_obs"][sel].sum(), 1) for p in ok]))
                A = lambda k: np.mean([p[arm][0]["table"][k] for p in ok])
                farsub = np.mean([p[arm + 2][0]["table"]["false_alarm_ratio"] for p in ok])
                app.append(f"| {name} | {an} | {fmt(A('fss_thr50.1_win11_member'))} | {fmt(A('hit_rate'), '{:.2f}')} | "
                           f"{fmt(A('exceedance_ratio_input'))} | {fmt(A('maxq_ratio_q50'), '{:.2f}')} | "
                           f"{fmt(farsub, '{:.2f}')} | " + " | ".join(fmt(b) for b in bins) + " |")
    rd = find("mse_randdrop_s42", "ext")
    if rd:
        s = scores(rd[1], full_idx)
        app.append(f"| MSE random drop (size-matched control) | full | {fmt(rd[0]['table']['fss_thr50.1_win11_member'])} | "
                   f"{fmt(rd[0]['table']['hit_rate'], '{:.2f}')} | {fmt(rd[0]['table']['exceedance_ratio_input'])} | "
                   f"{fmt(rd[0]['table']['maxq_ratio_q50'], '{:.2f}')} | – | exc {s['exc']:.3f} | | |")
    n_ev = int((find("mse_s42_full", "ext")[1]["tmax"] >= U).sum()) if find("mse_s42_full", "ext") else -1
    notes = ["", f"Validation, not test (F2 not run). valext = val_extremes1024 ({n_ev} patches with max >= {U}), "
             f"{D} processing days, B = {args.B} day resamples.",
             "Exc = predicted / observed pixels >= u on observed-event patches; Peak = median predicted / observed patch "
             "maximum there; q90 = 90th percentile of predicted over observed patch maxima (unpaired); FAR = P(obs max < u "
             "| pred max >= u); twCRPS = threshold-weighted CRPS (weight 1{z >= u}) summed over all pixels per observed "
             "pixel >= u (mm/h). FM: averages over 16 members; twCRPS and CRPS use the ensemble. U-Net CRPS = MAE.",
             "Δ columns: change from R1 to R4 relative to the block baseline's change, exp(Δlog) - 1; positive = degrades "
             "less than the baseline. Brackets: 95% paired day-block bootstrap. A ratio already above 1 in R1 that falls "
             "towards 1 in R4 counts as degradation here; read the R1/R4 values with it.",
             "† the windowed spectral loss is gamed (centre spikes, DECISIONS §45)."]
    md = "\n".join(lines + notes + app) + "\n"
    open(f"{args.out}/main_table_val.md", "w").write(md)
    with open(f"{args.out}/main_table_val_interaction.csv", "w") as f:
        f.write("model,metric,point_pct,lo95_pct,hi95_pct,n_seeds\n")
        for r in csv:
            f.write(",".join(str(x) for x in r) + "\n")
    print(md)


if __name__ == "__main__":
    main()
