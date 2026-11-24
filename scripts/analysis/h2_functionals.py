#!/usr/bin/env python
"""H2 test on the stored functionals (DECISIONS §50, pre-stated).

Per model, channel and level: S = mean over the 1,024 valext patches of |g_pred - g_target| in the loss's space
(log1p A, log1p P, signed log1p chi). Primary: D = log S_Minkowski - log S_area-only for perimeter and Euler at
50.1 mm in the cut arm, 95% day-block bootstrap. Known-answer check on the area channel first.

    python scripts/analysis/h2_functionals.py --out <dir> [--B 2000]
"""
import argparse
import os

import numpy as np

from src.evaluation.day_block import processing_days

OUTE = "/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/eval_results/extremes/v4adv_valext_tab"
META = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta"
ROWS = {"MSE": ["mse_s42", "mse_s43", "mse_s44"], "MSE + QM": ["mse_qm_s42"], "wMSE": ["wmse_s42"],
        "quantile": ["quantile_s42"], "FSS": ["fss_s42"], "area only": ["area_s42"],
        "Minkowski": ["mink_s42", "mink_s43", "mink_s44"]}
CH = ["area", "perimeter", "euler"]
LEVELS = [20.2, 31.7, 50.1, 79.0]
PIX_KM2 = 4.0


def to_loss_space(g):
    return np.stack([np.log1p(g[:, 0]), np.log1p(g[:, 1]), np.sign(g[:, 2]) * np.log1p(np.abs(g[:, 2]))], 1)


def load(tag):
    a = np.load(f"{OUTE}/{tag}/extremes_arrays.npz")
    return {k: a[k] for k in ("gamma_hat", "gamma_target", "thresholds", "n_exc_pred_members", "n_exc_obs")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    err = {}           # model -> {arm: array [seeds, N, 3, Q]}
    th = None
    for name, seeds in ROWS.items():
        err[name] = {}
        for arm in ("full", "cut"):
            es = []
            for t in seeds:
                a = load(f"{t}_{arm}")
                th = a["thresholds"]
                j = int(np.argmin(np.abs(th - 50.1)))
                # known answer: exp(area) - 1 at 50.1 equals 4 km^2 x the stored exceedance counts
                ap_ = np.expm1(to_loss_space(a["gamma_hat"])[:, 0, j])
                at_ = np.expm1(a["gamma_target"][:, 0, j])
                e_p = np.abs(ap_ - PIX_KM2 * a["n_exc_pred_members"][0]).max()
                e_t = np.abs(at_ - PIX_KM2 * a["n_exc_obs"]).max()
                if e_p > 1e-2 * PIX_KM2 or e_t > 1e-2 * PIX_KM2:
                    raise SystemExit(f"KNOWN-ANSWER CHECK FAILED {t}_{arm}: pred {e_p:.3g}, target {e_t:.3g} km^2")
                es.append(np.abs(to_loss_space(a["gamma_hat"]) - a["gamma_target"]))
            err[name][arm] = np.stack(es)
    print(f"known-answer check passed (thresholds {np.round(th, 3).tolist()})")
    li = [int(np.argmin(np.abs(th - u))) for u in LEVELS]

    days = processing_days(f"{META}/val_extremes1024.txt", 1024)
    dc = np.unique(days, return_inverse=True)[1]
    D = dc.max() + 1
    by_day = [np.flatnonzero(dc == d) for d in range(D)]
    rng = np.random.default_rng(0)
    res = [np.concatenate([by_day[d] for d in rng.integers(0, D, D)]) for _ in range(args.B)]

    def S(e, idx):                      # e [seeds,N,3,Q] -> [3,Q], seeds averaged
        return e[:, idx].mean(axis=(0, 1))

    full = np.arange(1024)
    pt = {m: {arm: S(err[m][arm], full) for arm in err[m]} for m in err}
    bs = {m: {arm: np.stack([S(err[m][arm], i) for i in res]) for arm in err[m]} for m in err}

    def contrast(m, ref, arm, c, j):
        p = np.log(pt[m][arm][c, j]) - np.log(pt[ref][arm][c, j])
        b = np.log(bs[m][arm][:, c, j]) - np.log(bs[ref][arm][:, c, j])
        lo, hi = np.quantile(b, [0.025, 0.975])
        return p, lo, hi

    pc = lambda x: 100 * (np.exp(x) - 1)
    L = ["# H2 test on the stored functionals (DECISIONS §50), valext, validation", ""]
    # primary
    L += ["## Primary: log S_Minkowski - log S_area-only, cut arm, 50.1 mm (as % change of the mean residual)", ""]
    verdict = []
    for c in (1, 2):
        p, lo, hi = contrast("Minkowski", "area only", "cut", c, li[2])
        L.append(f"- {CH[c]}: {pc(p):+.1f}% [{pc(lo):+.1f}, {pc(hi):+.1f}]")
        verdict.append("sup" if hi < 0 else ("not" if lo > 0 else "inc"))
    v = "supported" if all(x == "sup" for x in verdict) else ("not supported" if "not" in verdict else "inconclusive")
    L += ["", f"**Verdict by the pre-stated rule: {v}.**", ""]
    # S table
    L += ["## Mean residual S per model, channel, level", ""]
    for arm in ("full", "cut"):
        L += [f"### {arm} arm", "", "| model | " + " | ".join(f"{CH[c][:5]} {u:g}" for c in range(3) for u in LEVELS)
              + " |", "|---|" + "---|" * (3 * len(LEVELS))]
        for m in err:
            L.append(f"| {m} | " + " | ".join(f"{pt[m][arm][c, j]:.3f}" for c in range(3) for j in li) + " |")
        L.append("")
    # contrasts
    for ref, models in (("area only", ["Minkowski"]), ("MSE", ["MSE + QM", "wMSE", "quantile", "FSS", "area only",
                                                                 "Minkowski"])):
        L += [f"## log S_model - log S_{ref} (% change of the mean residual; negative = closer to the target)", ""]
        for arm in ("full", "cut"):
            L += [f"### {arm} arm", "", "| model | channel | " + " | ".join(f"{u:g} mm" for u in LEVELS) + " |",
                  "|---|---|" + "---|" * len(LEVELS)]
            for m in models:
                for c in range(3):
                    cells = []
                    for j in li:
                        p, lo, hi = contrast(m, ref, arm, c, j)
                        cells.append(f"{pc(p):+.0f} [{pc(lo):+.0f}, {pc(hi):+.0f}]")
                    L.append(f"| {m} | {CH[c]} | " + " | ".join(cells) + " |")
            L.append("")
    L.append("Levels 20.2 and 31.7 mm are in every loss grid; 50.1 and 79.0 mm in none. Seeds: MSE and Minkowski 3, "
             f"others 1. Bootstrap: 95%, B = {args.B} day resamples over {D} days.")
    md = "\n".join(L) + "\n"
    open(f"{args.out}/h2_functionals_val.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
