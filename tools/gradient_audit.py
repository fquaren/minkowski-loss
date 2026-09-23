#!/usr/bin/env python
"""Gradient audit of auxiliary losses: how much does each term actually steer training?

The math behind every number printed here is in `notes/gradient_audit.pdf`. In short, for a
base loss L0 (the MSE) and an auxiliary loss Lk used with weight lam:

    share      r = lam * ||grad Lk|| / ||grad L0||       (r << 1: the term barely moves theta)
    parity     lam* = ||grad L0|| / ||grad Lk||,  so r = lam / lam*
    cos        cos(grad L0, grad Lk)
    P-share    the same ratio after Adam's diagonal preconditioner P = 1/(sqrt(v_hat)+eps),
               taken from the checkpoint's optimizer state. This ratio is what shapes the
               actual AdamW update.

Norms are of the **mean** gradient over many minibatches, with the minibatch-noise bias
removed:
    E||g_B||^2 = ||mu||^2 + tr(Sigma)/B  =>  ||mu||^2 ~ (N ||g_bar||^2 - mean ||g_i||^2) / (N-1).
At a converged checkpoint mu_0 ~ 0 and per-batch norms are mostly noise, which is why the
first version of this audit (tools/gradient_audit_v0/) is only a rough guide. Error bars come
from a jackknife over K groups of batches.

Two modes (both run by default):

  gradients   at each checkpoint in --grad_at, measure every loss in --losses against the
              MSE term. `init` = freshly initialised weights (seeded), no optimizer state.
  evaluate    for every run in --runs, evaluate all losses on the same batches (no grad), and
              report each model's own objective against vanilla, as a paired difference with
              a bootstrap interval.

Runs on GPU 1 only (enforced by `import src`, see src/node_limits.py). Keep --num_workers at 4
or less: the whole job shares the 8-core budget with anything else that is running.

    python tools/gradient_audit.py config.yaml --n_batches 256 \
        --out_dir eval_results/gradient_audit/20260923
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.utils import load_config, load_scaler_val, set_seed  # noqa: E402  (applies node limits)
from src.data.datasets import DeterministicSRDataset  # noqa: E402
from src.losses.competing import build_structural_loss  # noqa: E402
from src.models.unet import LogSpaceResidualUNet  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

# runs/ is gitignored, so a worktree has none: fall back to the main checkout's.
RUN_DIR = next(d for d in (os.path.join(ROOT, "runs", "sr_analytical"),
                           "/work/fquareng/ch2/minkowski-loss/runs/sr_analytical")
               if os.path.isdir(d))
LOSSES = ["minkowski", "spectral", "ssim", "wetarea", "opticalflow"]
# Weights the Study-1 runs were trained with (scripts/hpc/losses_wrapper.sh; Minkowski 1e-4).
WEIGHTS = {"minkowski": 1e-4, "spectral": 5e-4, "ssim": 1.6e-4, "wetarea": 2e-2,
           "opticalflow": 4e-5}
RUNS = {
    "vanilla": ("UNet_Ana_20260623_144958", None),
    "minkowski": ("UNet_Ana_20260731_100128", "minkowski"),
    "spectral": ("UNet_Ana_20260829_033325", "spectral"),
    "ssim": ("UNet_Ana_20260824_155753", "ssim"),
    "wetarea": ("UNet_Ana_20260826_041128", "wetarea"),
    "opticalflow": ("UNet_Ana_20260827_151319", "opticalflow"),
    "vanilla_34ep": ("UNet_Ana_20260722_081033", None),
}


def to_phys(t, mv):
    return F.relu(torch.expm1(torch.clamp(t * mv, max=7.0)))


def make_loader(cfg, args, sv):
    dem = json.load(open(cfg["DEM_STATS"]))
    ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], cfg["TRAIN_METADATA_FILE"],
                                (dem["dem_mean"], dem["dem_std"]), sv, split="train",
                                data_percentage=args.data_percentage,
                                topology_mode=cfg.get("TOPOLOGY_MODE", "euler"))
    g = torch.Generator().manual_seed(args.seed)   # same batches on every pass
    sampler = WeightedRandomSampler(ds.sample_weights, args.n_batches * args.batch_size,
                                    replacement=True, generator=g)
    return DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=False)


def load_model(tag, dev, seed, which):
    m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
    if tag == "init":
        return m, None, 0
    run = RUNS[tag][0]
    ck = torch.load(os.path.join(RUN_DIR, run, f"unet_{which}.pth"), map_location=dev,
                    weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    return m, ck.get("optimizer_state_dict"), ck.get("epoch")


def adam_preconditioner(model, opt_state):
    """Flat diag(1/(sqrt(v_hat)+eps)) in model.parameters() order, or None."""
    if not opt_state or not opt_state.get("state"):
        return None
    grp = opt_state["param_groups"][0]
    b2, eps = grp["betas"][1], grp["eps"]
    parts = []
    for i, p in enumerate(model.parameters()):
        st = opt_state["state"].get(i)
        if st is None:
            return None
        step = float(st["step"])
        v_hat = st["exp_avg_sq"].to(p.device).float() / (1 - b2 ** step)
        parts.append((1.0 / (v_hat.sqrt() + eps)).reshape(-1))
    return torch.cat(parts)


def flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      .float() for p in model.parameters()])


def jackknife(fn, K):
    """Leave-one-group-out estimates of a statistic -> (full, se)."""
    full = fn(None)
    loo = np.array([fn(k) for k in range(K)])
    se = np.sqrt((K - 1) / K * np.sum((loo - loo.mean()) ** 2))
    return float(full), float(se)


def gradient_pass(tag, which, cfg, args, losses, loader, mv, dev):
    model, opt_state, epoch = load_model(tag, dev, args.seed, which)
    model.train()
    P = adam_preconditioner(model, opt_state) if args.adam else None
    terms = ["mse"] + list(losses)
    D = sum(p.numel() for p in model.parameters())
    K = args.groups
    gsum = {t: torch.zeros(K, D, device=dev) for t in terms}     # per-group gradient sums
    sq = {t: np.zeros(K) for t in terms}                          # per-group sum ||g_i||^2
    cnt = np.zeros(K)
    per_batch = {t: [] for t in terms}
    values = {t: [] for t in terms}
    clip = {t: [] for t in losses}      # per-batch ||g_mse + lam g_aux|| (what gets clipped)
    cross = {t: np.zeros(K) for t in losses}   # per-group sum <g_mse,i , g_aux,i> (same batch)

    t0 = time.time()
    for i, (X, Y, G) in enumerate(loader):
        k = i % K
        X, Y, G = X.to(dev), Y.to(dev), G.to(dev)
        pred = model(X)
        pp, tp, rp = to_phys(pred[:, 0:1], mv), to_phys(Y[:, 0:1], mv), to_phys(X[:, 0:1], mv)
        grads = {}
        lm = F.mse_loss(pred, Y)
        model.zero_grad()
        lm.backward(retain_graph=True)
        grads["mse"] = flat_grad(model)
        values["mse"].append(lm.item())
        for n, fn in losses.items():
            la = fn(pred[:, 0:1], Y[:, 0:1], pp, tp, gamma_target=G, reference_phys=rp,
                    anneal_factor=args.anneal)
            model.zero_grad()
            la.backward(retain_graph=True)
            grads[n] = flat_grad(model)
            values[n].append(la.item())
            clip[n].append((grads["mse"] + WEIGHTS[n] * grads[n]).norm().item())
            cross[n][k] += float((grads["mse"] * grads[n]).sum())
        for t, g in grads.items():
            gsum[t][k] += g
            nrm2 = float(g.square().sum())
            sq[t][k] += nrm2
            per_batch[t].append(np.sqrt(nrm2))
        cnt[k] += 1
        del grads, pred
        if (i + 1) % 25 == 0:
            print(f"  [{tag}] {i + 1}/{args.n_batches} batches, {time.time() - t0:.0f}s",
                  flush=True)

    def sums(t, drop):
        keep = [j for j in range(K) if j != drop]
        return gsum[t][keep].sum(0), sq[t][keep].sum(), cnt[keep].sum()

    def mean_norm2(t, drop, M=None):
        s, s2, n = sums(t, drop)
        g_bar = s / n
        if M is not None:
            g_bar = g_bar * M
        raw = float(g_bar.square().sum())
        if M is not None:          # noise correction needs per-batch P-norms: not tracked
            return raw, raw
        corr = max((n * raw - s2 / n) / (n - 1), 0.0)
        return raw, corr

    res = {"checkpoint": tag, "which": which if tag != "init" else "init", "epoch": epoch,
           "n_batches": int(cnt.sum()), "batch_size": args.batch_size,
           "anneal": args.anneal, "adam_state": P is not None, "terms": {}}
    for t in terms:
        res["terms"][t] = {
            "value_mean": float(np.mean(values[t])),
            "per_batch_norm_median": float(np.median(per_batch[t])),
            "per_batch_norm_iqr": [float(np.percentile(per_batch[t], 25)),
                                   float(np.percentile(per_batch[t], 75))],
        }
        full_raw, _ = jackknife(lambda d: np.sqrt(mean_norm2(t, d)[0]), K)
        full_c, se_c = jackknife(lambda d: np.sqrt(mean_norm2(t, d)[1]), K)
        res["terms"][t].update(mean_grad_norm=full_raw, signal_norm=full_c,
                               signal_norm_se=se_c,
                               noise_fraction=float(1 - (full_c / np.median(per_batch[t])) ** 2)
                               if np.median(per_batch[t]) > 0 else None)
    for n in losses:
        lam = WEIGHTS[n]

        def parity(d, n=n):
            a, b = np.sqrt(mean_norm2("mse", d)[1]), np.sqrt(mean_norm2(n, d)[1])
            return a / b if b > 0 else np.nan

        def cos(d, n=n):
            a = sums("mse", d)[0]
            b = sums(n, d)[0]
            return float(F.cosine_similarity(a, b, dim=0))

        def cos_corr(d, n=n):
            """Cosine of the mean gradients with the same-batch noise covariance removed."""
            a, a2, m = sums("mse", d)
            b, _, _ = sums(n, d)
            c2 = cross[n][[j for j in range(K) if j != d]].sum()
            inner = (m * float((a / m * b / m).sum()) - c2 / m) / (m - 1)
            na, nb = np.sqrt(mean_norm2("mse", d)[1]), np.sqrt(mean_norm2(n, d)[1])
            return inner / (na * nb) if na > 0 and nb > 0 else np.nan

        lam_star, lam_se = jackknife(parity, K)
        cc, cc_se = jackknife(cos_corr, K)
        c, c_se = jackknife(cos, K)
        # At a converged checkpoint the MSE signal can be indistinguishable from noise
        # (corrected norm 0): parity is then undefined, i.e. any lam dominates on average.
        def safe(x, y):
            return x / y if y not in (0, 0.0) and np.isfinite(y) else float("inf")

        pb = np.median(np.array(per_batch["mse"]) / np.maximum(np.array(per_batch[n]), 1e-30))
        out = {"lambda_used": lam, "parity_lambda": lam_star, "parity_lambda_se": lam_se,
               "share_r": safe(lam, lam_star), "share_r_rel_se": safe(lam_se, lam_star),
               "mse_signal_below_noise": bool(lam_star == 0),
               "cos_mean_grads": c, "cos_se": c_se,
               "cos_noise_corrected": cc, "cos_noise_corrected_se": cc_se,
               "parity_lambda_per_batch_median": float(pb),
               "clip_engaged_frac": float(np.mean(np.array(clip[n]) > 1.0))}
        if P is not None:
            ps, _ = jackknife(lambda d, n=n: safe(lam * np.sqrt(mean_norm2(n, d, P)[0]),
                                                  np.sqrt(mean_norm2("mse", d, P)[0])), K)
            out["adam_share"] = ps
        res["terms"][n].update(out)
    del gsum
    torch.cuda.empty_cache()
    return res


def evaluate_pass(cfg, args, losses, loader, mv, dev):
    """Every loss for every run, on identical batches; paired against vanilla."""
    vals = {}
    for tag in args.runs:
        model, _, epoch = load_model(tag, dev, args.seed, args.which)
        model.eval()
        per = {t: [] for t in ["mse", "mae_phys"] + list(losses)}
        with torch.no_grad():
            for X, Y, G in loader:
                X, Y, G = X.to(dev), Y.to(dev), G.to(dev)
                pred = model(X)
                pp, tp = to_phys(pred[:, 0:1], mv), to_phys(Y[:, 0:1], mv)
                rp = to_phys(X[:, 0:1], mv)
                per["mse"].append(F.mse_loss(pred, Y).item())
                per["mae_phys"].append((pp - tp).abs().mean().item())
                for n, fn in losses.items():
                    per[n].append(fn(pred[:, 0:1], Y[:, 0:1], pp, tp, gamma_target=G,
                                     reference_phys=rp, anneal_factor=args.anneal).item())
        vals[tag] = {k: np.array(v) for k, v in per.items()}
        print(f"  evaluated {tag} (epoch {epoch})", flush=True)

    rng = np.random.default_rng(args.seed)
    out = {}
    base = vals.get("vanilla")
    for tag, v in vals.items():
        row = {k: float(a.mean()) for k, a in v.items()}
        own = RUNS[tag][1]
        if base is not None and own is not None:
            d = v[own] - base[own]                   # paired, per batch
            boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            row["own_loss"] = own
            row["own_vs_vanilla_rel"] = float(d.mean() / base[own].mean())
            row["own_vs_vanilla_ci95"] = [float(np.percentile(boot, 2.5) / base[own].mean()),
                                          float(np.percentile(boot, 97.5) / base[own].mean())]
        out[tag] = row
    return out


def to_markdown(grad_results, eval_results):
    L = ["# Gradient audit", ""]
    for r in grad_results:
        L += [f"## Gradients at `{r['checkpoint']}` ({r['which']}, epoch {r['epoch']}), "
              f"{r['n_batches']} x {r['batch_size']} patches, anneal {r['anneal']}", "",
              f"MSE: value {r['terms']['mse']['value_mean']:.3e}, signal ||mu|| "
              f"{r['terms']['mse']['signal_norm']:.3e} ± {r['terms']['mse']['signal_norm_se']:.1e}, "
              f"per-batch median {r['terms']['mse']['per_batch_norm_median']:.3e} "
              f"(noise fraction {r['terms']['mse']['noise_fraction']:.2f})", "",
              "| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | "
              "Adam share | noise frac | clip engaged |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for n, t in r["terms"].items():
            if n == "mse":
                continue
            L.append(f"| {n} | {t['lambda_used']:.1e} | {t['parity_lambda']:.2e} "
                     f"(± {t['parity_lambda_se']:.1e}) | {t['share_r']:.3g} | "
                     f"{t['parity_lambda_per_batch_median']:.2e} | {t['cos_mean_grads']:+.3f} | "
                     f"{t['cos_noise_corrected']:+.3f} | "
                     f"{t.get('adam_share', float('nan')):.3g} | "
                     f"{t['noise_fraction']:.2f} | {t['clip_engaged_frac']:.0%} |")
        L.append("")
    if eval_results:
        cols = ["mse", "mae_phys"] + LOSSES
        L += ["## Every loss for every model (same batches, eval mode)", "",
              "| model | " + " | ".join(cols) + " | own vs vanilla (95% CI) |",
              "|" + "---|" * (len(cols) + 2)]
        for tag, row in eval_results.items():
            own = (f"{row['own_vs_vanilla_rel']:+.1%} [{row['own_vs_vanilla_ci95'][0]:+.1%}, "
                   f"{row['own_vs_vanilla_ci95'][1]:+.1%}]" if "own_loss" in row else "")
            L.append(f"| {tag} | " + " | ".join(f"{row.get(c, float('nan')):.4g}" for c in cols)
                     + f" | {own} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_batches", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--groups", type=int, default=8, help="jackknife groups")
    ap.add_argument("--grad_at", nargs="*", default=["init", "vanilla", "minkowski"],
                    help="checkpoints to take gradients at ('init' = fresh weights)")
    ap.add_argument("--losses", nargs="*", default=LOSSES)
    ap.add_argument("--runs", nargs="*", default=list(RUNS), help="for the evaluate pass")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--anneal", type=float, default=0.1)
    ap.add_argument("--data_percentage", type=float, default=10.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_adam", dest="adam", action="store_false")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()
    if args.num_workers > 4:
        sys.exit("--num_workers > 4 would break the 8-core node budget alongside other jobs")

    set_seed(args.seed)
    cfg = load_config(args.config)
    dev = torch.device("cuda")
    sv = load_scaler_val(cfg)
    mv = torch.tensor(sv, device=dev)
    losses = {n: build_structural_loss(n, cfg, dev) for n in args.losses}
    loader = make_loader(cfg, args, sv)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).uuid}), CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
          f"cores {sorted(os.sched_getaffinity(0))}", flush=True)

    grad_results = []
    for tag in args.grad_at:
        print(f"gradients at {tag}", flush=True)
        set_seed(args.seed)
        grad_results.append(gradient_pass(tag, args.which, cfg, args, losses, loader, mv, dev))
        json.dump(grad_results, open(os.path.join(args.out_dir, "gradients.json"), "w"),
                  indent=1)
    eval_results = None
    if not args.skip_eval:
        print("evaluate pass", flush=True)
        eval_results = evaluate_pass(cfg, args, losses, loader, mv, dev)
        json.dump(eval_results, open(os.path.join(args.out_dir, "evaluate.json"), "w"),
                  indent=1)
    md = to_markdown(grad_results, eval_results)
    open(os.path.join(args.out_dir, "gradient_audit.md"), "w").write(md)
    json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
