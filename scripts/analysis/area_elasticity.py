"""How does the fine-scale excursion-set area respond to the coarse state, compared with the
intensity percentiles? (Tom's "area as anchor" claim, reading (a).)

Coarse state c = mean of the 10x10 coarse field (mm in 1 h). Patches are binned in log c.
Per bin b, S_b(u) = mean fraction of the 128x128 patch at or above u (from gamma_targets
channel 0, which is 4 km^2 per pixel). Then, across bins:

  a_A(u) = d log S_b(u) / d log c      (area elasticity at fixed level u)
  a_q(s) = d log q_b(s) / d log c      (elasticity of the level exceeded by a fraction s)

and the two-mechanism fit a_A(u) = alpha + beta * kappa(u), kappa(u) = -d log S / d log u:
alpha = how much wet/convective area grows with c (frequency), beta = how much the intensity
scale grows with c. Pure area change -> (1, 0)-like; pure intensity rescaling -> (0, 1)-like.
For fixed s, a_q(s) = beta + alpha / kappa, so percentile sensitivity falls toward beta in the tail
if alpha > 0.

  extract: python3 scripts/analysis/area_elasticity.py extract config.yaml --split train --out X.npz
  analyse: python3 scripts/analysis/area_elasticity.py analyse X.npz --out_dir DIR
"""
import argparse
import json
import os

import numpy as np

N_PIX = 128 * 128


def extract(config, split, out, block=20000):
    import yaml
    import zarr

    cfg = yaml.safe_load(open(config))
    root = cfg["PREPROCESSED_DATA_DIR"]
    meta_key = {"train": "TRAIN", "validation": "VAL", "test": "TEST"}[split]
    rows, days = [], []
    for line in open(cfg[f"{meta_key}_METADATA_FILE"]):
        p = line.strip().split(",")
        rows.append(int(p[4]))
        days.append(int(p[0][:8]))
    rows, days = np.asarray(rows), np.asarray(days)
    g = zarr.open(os.path.join(root, "preprocessed_dataset.zarr"), "r")[split]
    # Listed rows only: the list carries the exclusions (DECISIONS §34, §36).
    order = np.argsort(rows)
    rows_s = rows[order]
    c = np.empty(len(rows), np.float32)
    area = np.empty((len(rows), g["gamma_targets"].shape[-1]), np.float32)
    for lo in range(0, rows_s[-1] + 1, block):
        sel = (rows_s >= lo) & (rows_s < lo + block)
        if not sel.any():
            continue
        hi = rows_s[sel].max() + 1
        r = rows_s[sel] - lo
        c[order[sel]] = g["coarse_precip"][lo:hi][r].mean(axis=(1, 2))
        area[order[sel]] = g["gamma_targets"][lo:hi, 0][r] / 4.0  # km^2 -> pixels
        print(f"{lo + block}/{rows_s[-1] + 1}", flush=True)
    u = np.load(os.path.join(root, "physical_thresholds.npy"))
    np.savez(out, c=c, area=area, day=days, u=u)


def bin_stats(c, area, groups, edges):
    """Per-bin mean patch fraction above each u, summed by group for the bootstrap."""
    b = np.digitize(c, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    nb = len(edges) - 1
    idx = groups[ok] * nb + b[ok]
    ng = groups.max() + 1
    n = np.bincount(idx, minlength=ng * nb).reshape(ng, nb)
    cs = np.bincount(idx, weights=np.log(c[ok]), minlength=ng * nb).reshape(ng, nb)
    a = np.stack([np.bincount(idx, weights=area[ok, j], minlength=ng * nb)
                  for j in range(area.shape[1])], -1).reshape(ng, nb, -1)
    return n, cs, a


def elasticities(n, cs, a, u, s_levels, min_n=50, min_pix=100):
    """From per-bin counts: a_A(u), a_q(s), kappa(u) and the (alpha, beta) fit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = cs / n
        S = a / (n[:, None] * N_PIX)
    good_bin = n >= min_n
    aA = np.full(len(u), np.nan)
    for j in range(len(u)):
        m = good_bin & (a[:, j] >= min_pix)
        if m.sum() >= 3:
            aA[j] = np.polyfit(logc[m], np.log(S[m, j]), 1)[0]
    # Level exceeded by fraction s in each bin: log-log interpolation of S_b(u).
    lu = np.log(u)
    aq = np.full(len(s_levels), np.nan)
    for k, s in enumerate(s_levels):
        x, y = [], []
        for b in np.flatnonzero(good_bin):
            m = a[b] >= min_pix
            ls = np.log(S[b, m])
            if m.sum() >= 2 and ls[0] >= np.log(s) >= ls[-1]:
                x.append(logc[b])
                y.append(np.interp(np.log(s), ls[::-1], lu[m][::-1]))
        if len(x) >= 3:
            aq[k] = np.polyfit(x, y, 1)[0]
    # kappa from the pooled curve, the shape all bins are compared against.
    Sp = a[good_bin].sum(0) / (n[good_bin].sum() * N_PIX)
    with np.errstate(divide="ignore"):
        kappa = -np.gradient(np.log(Sp), lu)
    m = np.isfinite(aA) & np.isfinite(kappa)
    X = np.stack([np.ones(m.sum()), kappa[m]], 1)
    coef, *_ = np.linalg.lstsq(X, aA[m], rcond=None)
    resid = aA[m] - X @ coef
    return dict(aA=aA, aq=aq, kappa=kappa, alpha=coef[0], beta=coef[1],
                fit_rmse=float(np.sqrt(np.mean(resid ** 2))), S=S, logc=logc, n=n)


def analyse(npz, out_dir, n_boot=200, c_min=0.01, n_bins=16, seed=0):
    d = np.load(npz)
    c, area, day, u = d["c"], d["area"], d["day"], d["u"]
    keep = c >= c_min
    c, area, day = c[keep], area[keep], day[keep]
    edges = np.logspace(np.log10(c_min), np.log10(np.quantile(c, 0.9999)), n_bins + 1)
    _, g = np.unique(day, return_inverse=True)
    n, cs, a = bin_stats(c, area, g, edges)
    s_levels = np.logspace(-1, -5, 9)
    est = elasticities(n.sum(0), cs.sum(0), a.sum(0), u, s_levels)
    # Day-block bootstrap: patches of one day share the weather, so days are the unit.
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in ("aA", "aq", "alpha", "beta")}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n.shape[0], n.shape[0]), minlength=n.shape[0])
        e = elasticities(np.tensordot(w, n, 1), np.tensordot(w, cs, 1),
                         np.tensordot(w, a, 1), u, s_levels)
        for k in boot:
            boot[k].append(e[k])
    ci = {k: np.nanpercentile(np.asarray(v, float), [2.5, 97.5], axis=0) for k, v in boot.items()}

    os.makedirs(out_dir, exist_ok=True)
    res = dict(n_patches=int(len(c)), n_days=int(n.shape[0]), c_min=c_min,
               u=u.tolist(), aA=est["aA"].tolist(), aA_ci=ci["aA"].tolist(),
               s_levels=s_levels.tolist(), aq=est["aq"].tolist(), aq_ci=ci["aq"].tolist(),
               kappa=est["kappa"].tolist(), alpha=float(est["alpha"]), alpha_ci=ci["alpha"].tolist(),
               beta=float(est["beta"]), beta_ci=ci["beta"].tolist(), fit_rmse=est["fit_rmse"])
    json.dump(res, open(os.path.join(out_dir, "elasticities.json"), "w"), indent=1)
    plot(est, ci, u, s_levels, edges, out_dir)
    return res


def plot(est, ci, u, s_levels, edges, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    good = est["n"] >= 50
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(np.flatnonzero(good)):
        ax[0].loglog(u, est["S"][b], color=cmap(i / good.sum()), lw=1,
                     label=f"c={np.exp(est['logc'][b]):.2g}" if i % 3 == 0 else None)
    ax[0].set(xlabel="u (mm in 1 h)", ylabel="S_b(u): patch fraction >= u",
              title="area curves per coarse-mean bin")
    ax[0].legend(fontsize=7)

    ax[1].semilogx(u, est["aA"], "o-", label="a_A(u) = dlog S / dlog c")
    ax[1].fill_between(u, *ci["aA"], alpha=0.3)
    ax[1].semilogx(u, est["alpha"] + est["beta"] * est["kappa"], "k--",
                   label=f"alpha + beta kappa(u): alpha={est['alpha']:.2f}, beta={est['beta']:.2f}")
    ax[1].semilogx(u, est["kappa"], ":", color="grey", label="kappa(u) (pure scale shift, beta=1)")
    ax[1].axhline(1, color="grey", lw=0.5)
    ax[1].set(xlabel="u (mm in 1 h)", ylabel="elasticity", title="area elasticity vs level")
    ax[1].legend(fontsize=7)

    ax[2].semilogx(s_levels, est["aq"], "o-", color="C1", label="a_q(s) = dlog q_s / dlog c")
    ax[2].fill_between(s_levels, *ci["aq"], color="C1", alpha=0.3)
    ax[2].invert_xaxis()
    ax[2].set(xlabel="exceeded patch fraction s (tail to the right)", ylabel="elasticity",
              title="percentile elasticity vs level")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "area_elasticity.png"), dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("config")
    e.add_argument("--split", default="train")
    e.add_argument("--out", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("npz")
    a.add_argument("--out_dir", required=True)
    a.add_argument("--c_min", type=float, default=0.01)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.config, args.split, args.out)
    else:
        print(json.dumps(analyse(args.npz, args.out_dir, c_min=args.c_min), indent=1)[:3000])
