#!/usr/bin/env python
"""Is the Minkowski/flow-matching coupling implemented correctly?

Phase 3 produced a null-to-negative result. That has two possible causes, and they call
for opposite responses: either the reward is a poor objective for this problem, or the
gradient never reaches the weights in a usable form. This script tests the second, on a
single batch, in minutes rather than days.

Five tests, in decreasing order of decisiveness:

  1. OVERFIT A SINGLE BATCH. Take one batch and optimise the reward alone for N steps.
     A correctly wired reward must be drivable down substantially on data it is allowed
     to memorise. If it barely moves, the coupling is broken and no hyperparameter will
     save it. Everything else is diagnosis of *why*.

  2. GRADIENT CONCENTRATION. The reward is computed on expm1(z * max_val), so
     d(phys)/dz = max_val * exp(z * max_val): with max_val ~ 5 a pixel at z=0.9 carries
     ~700x the gradient of one at z=0.1. If a handful of pixels own the gradient norm,
     clip_grad_norm_ scales everything else toward zero and only the brightest pixels
     ever move.

  3. CLIPPING RATE. What fraction of steps hit the clip, and by how much. A norm that is
     routinely orders of magnitude above the threshold means the effective learning rate
     is set by the clip, not by REWARD_LR.

  4. PER-THRESHOLD SIGNAL. Reward and gradient contribution per threshold. Deep
     thresholds (89, 150 mm/h) have empty excursion sets in most members, and the
     annealed indicator has width tau = anneal * u, so they may contribute nothing.

  5. K SENSITIVITY. Cosine similarity between the reward gradient at K=1 and at larger K.
     If they disagree, one differentiable solver step is not representative of the
     trajectory and K=1 is shaping the wrong thing.

Usage:
  python scripts/evaluate/diagnose_coupling.py config.yaml \
      --fm_checkpoint runs/sr_flow_matching/<run>/fm_best.pth \
      --backbone runs/sr_analytical/<run>/unet_best.pth
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import (
    load_config, load_scaler_val, load_physical_thresholds, load_persistence_thresholds,
)
from src.models.unet import LogSpaceResidualUNet
from src.models.flow_matching import FlowMatching
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.trainers.reward_finetune import (
    draft_k_sample, minkowski_reward, _log_gamma, _load_sigma, _make_fm,
)


def draft_k_sample_x0(fm, cond, n_steps, method, K, x0=None):
    """draft_k_sample with an optional caller-supplied base draw (for reproducibility)."""
    if x0 is None:
        return draft_k_sample(fm, cond, n_steps, method, K, 1, False)
    x = x0
    dt = 1.0 / n_steps
    n_free = max(n_steps - K, 0)

    def _step(xx, i):
        t0 = i * dt
        t = torch.full((xx.shape[0],), t0, device=xx.device, dtype=xx.dtype)
        v = fm.velocity(xx, t, cond)
        if method == "euler":
            return xx + dt * v
        xe = xx + dt * v
        return xx + 0.5 * dt * (v + fm.velocity(xe, torch.full_like(t, t0 + dt), cond))

    with torch.no_grad():
        for i in range(n_free):
            x = _step(x, i)
    for i in range(n_free, n_steps):
        x = _step(x, i)
    return x


def _flat_grad(params):
    return torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fm_checkpoint", required=True)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--mode", default=None, help="override REWARD_MODE")
    ap.add_argument("--steps", type=int, default=40, help="overfit steps for test 1")
    ap.add_argument("--lr", type=float, default=None, help="override REWARD_LR")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--ensemble", type=int, default=None)
    ap.add_argument("--fixed_noise", action="store_true", default=True,
                    help="reuse one base-noise draw across steps so the reward is a "
                         "deterministic function of the parameters (default on)")
    ap.add_argument("--resample_noise", dest="fixed_noise", action="store_false",
                    help="re-draw base noise every step (the original, un-interpretable form)")
    ap.add_argument("--extreme_batch", action="store_true",
                    help="scan for a batch containing actual extremes instead of taking the first")
    ap.add_argument("--scan_batches", type=int, default=200)
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_val = load_scaler_val(config)
    max_val = torch.tensor(scaler_val, device=device, dtype=torch.float32)
    topo = config.get("TOPOLOGY_MODE", "euler")
    u_phys = load_physical_thresholds(config)
    mode = args.mode or config.get("REWARD_MODE", "distributional")
    M = args.ensemble or int(config.get("REWARD_ENSEMBLE", 8))
    K = int(config.get("REWARD_K", 1))
    n_steps = int(config.get("FM_SAMPLE_STEPS", 16))
    sampler = config.get("FM_SAMPLER", "heun")
    anneal = float(config.get("REWARD_ANNEAL", 0.05))
    w_spread = float(config.get("REWARD_SPREAD_WEIGHT", 0.5))
    lr = args.lr or float(config.get("REWARD_LR", 1e-5))

    print(f"[diag] mode={mode} M={M} K={K} sampler={sampler}-{n_steps} lr={lr:g} "
          f"anneal={anneal} max_val={float(max_val):.4f}")

    # ---- models ------------------------------------------------------------------
    bpath = args.backbone or config["BACKBONE_CHECKPOINT"]
    backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    backbone.load_state_dict(torch.load(bpath, map_location=device)["model_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    sigma, sigma_src = _load_sigma(config, device)
    print(f"[diag] backbone={os.path.basename(bpath)}  sigma_r: {sigma_src}")

    cond_on_mean = config.get("FM_CONDITION_ON_MEAN", True)
    fm = _make_fm(config, 3 if cond_on_mean else 2, device)
    fm.load_state_dict(torch.load(args.fm_checkpoint, map_location=device)["model_state_dict"])
    fm.train()

    thresh_b0 = load_persistence_thresholds(config)[0] if topo == "b0" else 0.0
    geom_fn = AnalyticalMinkowskiLoss(
        physical_thresholds=u_phys, pixel_size_km=config.get("PIXEL_SIZE_KM", 2.0),
        topology_mode=topo, area_mode="ste", persistence_thresh_b0=thresh_b0,
    ).to(device)

    with open(config["DEM_STATS"]) as f:
        d = json.load(f)
    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config["VAL_METADATA_FILE"],
                                (float(d["dem_mean"]), float(d["dem_std"])), scaler_val,
                                split="validation", topology_mode=topo)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        multiprocessing_context="spawn")
    if args.extreme_batch:
        best, best_max = None, -1.0
        for bi, (Xc, Yc, Gc) in enumerate(loader):
            if bi >= args.scan_batches:
                break
            m = float(torch.expm1(Yc[:, 0:1].clamp(0, 1) * float(scaler_val)).amax())
            if m > best_max:
                best, best_max = (Xc, Yc, Gc), m
        X, Y, Ygamma = best
        print(f"[diag] selected batch with max intensity {best_max:.1f} mm/h "
              f"(scanned {args.scan_batches} batches)")
    else:
        X, Y, Ygamma = next(iter(loader))
    X, Y, Ygamma = X.to(device), Y.to(device), Ygamma.to(device)
    with torch.no_grad():
        mu = backbone(X)
    cond = torch.cat([X, mu[:, 0:1]], dim=1) if cond_on_mean else X

    def decode(z):
        return F.relu(torch.expm1(torch.clamp(z, 0.0, 1.0) * max_val))

    # A reward that re-draws its base noise every call is a different random variable at
    # every step, so an optimisation trajectory over it cannot separate "not descending"
    # from "too noisy to see". Fixing the noise makes the reward a deterministic function
    # of the parameters, which is what the overfit test needs.
    gen = torch.Generator(device=device.type).manual_seed(1234)
    shape = (cond.shape[0], 1, cond.shape[2], cond.shape[3])
    FIXED_X0 = [torch.randn(shape, device=device, generator=gen) for _ in range(M)]

    def sample_fields(k=K, fixed=None):
        fixed = args.fixed_noise if fixed is None else fixed
        out = []
        for m in range(M):
            x0 = FIXED_X0[m] if fixed else None
            r = draft_k_sample_x0(fm, cond, n_steps, sampler, k, x0)
            out.append(decode(mu[:, 0:1] + sigma * r))
        return torch.stack(out, 0)

    params = [p for p in fm.parameters() if p.requires_grad]

    # ---- TEST 2/3: gradient concentration and clipping ---------------------------
    print("\n=== TEST 2/3: gradient concentration and clipping ===")
    fields = sample_fields()
    fields.retain_grad()
    reward = minkowski_reward(geom_fn, fields, Ygamma, mode, anneal, w_spread)
    fm.zero_grad(set_to_none=True)
    reward.backward(retain_graph=False)
    gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
    fg = fields.grad.abs().reshape(-1)
    k1 = max(1, int(0.01 * fg.numel()))
    top1 = fg.topk(k1).values.sum() / fg.sum().clamp_min(1e-30)
    nz = (fg > 0).float().mean()
    print(f"  reward value              : {float(reward):.5f}")
    print(f"  parameter grad norm       : {gnorm:.3e}   (clip threshold 1.0 -> "
          f"{'CLIPPED x%.0f' % (gnorm) if gnorm > 1 else 'not clipped'})")
    print(f"  field-grad mass in top 1%% : {float(top1)*100:.1f}%%  "
          f"(uniform would be 1%%; >50%% means a few pixels own the update)")
    print(f"  fraction of pixels with nonzero grad: {float(nz)*100:.1f}%")

    # ---- TEST 4: per-threshold signal --------------------------------------------
    print("\n=== TEST 4: per-threshold reward and gradient contribution ===")
    f0 = fields[0].detach().clone().requires_grad_(True)
    lg = _log_gamma(geom_fn, f0, anneal)
    resid = (lg - Ygamma).abs().mean(dim=(0, 1))          # [Q]
    print(f"  {'u (mm/h)':>10s} {'|resid|':>10s} {'grad norm':>12s} {'wet px':>9s}")
    for q, uq in enumerate(u_phys):
        g = torch.autograd.grad(lg[:, :, q].sum(), f0, retain_graph=True)[0]
        wet = (f0 >= float(uq)).float().mean()
        print(f"  {float(uq):10.2f} {float(resid[q]):10.4f} {float(g.abs().sum()):12.4e} "
              f"{float(wet)*100:8.3f}%")

    # ---- TEST 5: K sensitivity ---------------------------------------------------
    print("\n=== TEST 5: gradient direction vs K ===")
    def grad_at(k):
        fm.zero_grad(set_to_none=True)
        r = minkowski_reward(geom_fn, sample_fields(k), Ygamma, mode, anneal, w_spread)
        r.backward()
        return _flat_grad(params).detach().clone()
    g1 = grad_at(1)
    for k in (2, 4):
        if k > n_steps:
            continue
        gk = grad_at(k)
        cos = F.cosine_similarity(g1[None], gk[None]).item()
        print(f"  cos(grad K=1, grad K={k}) = {cos:+.4f}  "
              f"({'consistent' if cos > 0.5 else 'DIVERGENT -- K=1 is not representative'})")

    # ---- TEST 1: overfit a single batch (decisive) --------------------------------
    print("\n=== TEST 0: reward noise floor at FIXED parameters ===")
    with torch.no_grad():
        rr = [float(minkowski_reward(geom_fn, sample_fields(fixed=False), Ygamma,
                                     mode, anneal, w_spread)) for _ in range(12)]
        rf = [float(minkowski_reward(geom_fn, sample_fields(fixed=True), Ygamma,
                                     mode, anneal, w_spread)) for _ in range(3)]
    print(f"  resampled noise: mean {np.mean(rr):.5f}  sd {np.std(rr):.5f}  "
          f"({100*np.std(rr)/max(np.mean(rr),1e-9):.1f}% of the mean)")
    print(f"  fixed noise    : {rf[0]:.5f} (repeat sd {np.std(rf):.2e} -- should be ~0)")
    print(f"  => any change smaller than ~{2*np.std(rr):.4f} in TEST 1 is indistinguishable "
          f"from sampling noise")

    print(f"\n=== TEST 1: overfit one batch, reward only, {args.steps} steps (DECISIVE) ===")
    print(f"  base noise: {'FIXED' if args.fixed_noise else 'RESAMPLED each step'}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    hist, clipped = [], 0
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        r = minkowski_reward(geom_fn, sample_fields(), Ygamma, mode, anneal, w_spread)
        r.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        clipped += int(gn > 1.0)
        opt.step()
        hist.append(float(r))
        if i % max(1, args.steps // 10) == 0:
            print(f"  step {i:3d}  reward {float(r):.5f}  grad norm {gn:.3e}")
    r0, r1 = float(np.mean(hist[:3])), float(np.mean(hist[-3:]))
    print(f"\n  reward {r0:.5f} -> {r1:.5f}  ({(r1/r0-1)*100:+.1f}%)")
    print(f"  steps clipped: {clipped}/{args.steps}")
    if r1 < 0.8 * r0:
        print("  VERDICT: the reward is drivable. The coupling is wired correctly and the")
        print("           null result is about the objective or the budget, not the plumbing.")
    elif r1 < 0.97 * r0:
        print("  VERDICT: the reward moves only weakly. Suspect gradient starvation --")
        print("           check the concentration and clipping numbers above.")
    else:
        print("  VERDICT: the reward barely moves on data it is allowed to memorise.")
        print("           The coupling is broken; fix the plumbing before tuning anything.")


if __name__ == "__main__":
    main()
