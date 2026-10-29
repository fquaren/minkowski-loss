#!/usr/bin/env python
"""Gradient norm of each term of the reward fine-tuning loss, on a few training batches.

The fine-tune minimises  w_reward * R + w_prox * L_prox + w_retain * L_FM.  Loss values say
nothing about which term moves the weights; the gradient norms do. If the reward's share of
the total gradient is tiny, a fine-tune is mostly whatever the other terms are (e.g. more
flow-matching training), and its gains cannot be credited to the reward.

Each term is built exactly as in src/trainers/reward_finetune.py (same sampler, K, M,
anchor surrogate and retention loss) and back-propagated on its own, before clipping.

Usage:
  python scripts/evaluate/grad_norm_terms.py configs/config_v4_adv_fm_energy.yaml \
      --fm_checkpoint runs/sr_flow_matching/<run>/fm_best.pth --backbone <unet.pth>
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.utils import load_config, load_scaler_val, load_physical_thresholds, set_seed
from src.models.unet import LogSpaceResidualUNet
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.trainers.reward_finetune import (
    draft_k_sample, minkowski_reward, _base_sample, _load_sigma, _make_fm,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fm_checkpoint", required=True)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_val = load_scaler_val(config)
    max_val = torch.tensor(scaler_val, device=device, dtype=torch.float32)

    mode = config.get("REWARD_MODE", "distributional")
    K = int(config.get("REWARD_K", 1)); M = int(config.get("REWARD_ENSEMBLE", 4))
    w_reward = float(config.get("REWARD_WEIGHT", 1e-3))
    w_prox = float(config.get("REWARD_PROXIMAL_WEIGHT", 1.0))
    w_retain = float(config.get("REWARD_FM_RETENTION_WEIGHT", 1.0))
    w_spread = float(config.get("REWARD_SPREAD_WEIGHT", 0.5))
    anneal = float(config.get("REWARD_ANNEAL", 0.05))
    form = config.get("REWARD_FORM", "threshold")
    n_steps = int(config.get("FM_SAMPLE_STEPS", 16)); sampler = config.get("FM_SAMPLER", "heun")
    print(f"[grad] mode={mode} form={form} K={K} M={M} w_reward={w_reward:g} w_prox={w_prox:g} "
          f"w_retain={w_retain:g} w_spread={w_spread:g} anneal={anneal}")

    bpath = args.backbone or config["BACKBONE_CHECKPOINT"]
    backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    backbone.load_state_dict(torch.load(bpath, map_location=device)["model_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    sigma, sigma_src = _load_sigma(config, device)
    print(f"[grad] sigma_r: {sigma_src}")

    cond_on_mean = config.get("FM_CONDITION_ON_MEAN", True)
    ckpt = torch.load(args.fm_checkpoint, map_location=device)["model_state_dict"]
    fm = _make_fm(config, 3 if cond_on_mean else 2, device); fm.load_state_dict(ckpt); fm.train()
    fm_ref = _make_fm(config, 3 if cond_on_mean else 2, device); fm_ref.load_state_dict(ckpt)
    fm_ref.eval()
    params = [p for p in fm.parameters() if p.requires_grad]

    geom_fn = AnalyticalMinkowskiLoss(
        physical_thresholds=load_physical_thresholds(config),
        pixel_size_km=float(config.get("PIXEL_SIZE_KM", 2.0)),
        topology_mode=config.get("TOPOLOGY_MODE", "euler"), area_mode="ste",
    ).to(device)

    with open(config["DEM_STATS"]) as f:
        d = json.load(f)
    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config["TRAIN_METADATA_FILE"],
                                (float(d["dem_mean"]), float(d["dem_std"])), scaler_val,
                                split="train", topology_mode=config.get("TOPOLOGY_MODE", "euler"))
    bs = int(config.get("REWARD_BATCH_SIZE", 8))
    idx = np.random.default_rng(args.seed).choice(len(ds), bs * args.n_batches, replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=bs, shuffle=False, num_workers=0)

    def decode(y):
        return torch.relu(torch.expm1(torch.clamp(y, 0.0, 1.0) * max_val))

    def grad_of(loss):
        fm.zero_grad(set_to_none=True)
        loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None]).clone()

    rows = []
    for bi, (X, Y, Ygamma) in enumerate(loader):
        X, Y, Ygamma = X.to(device), Y.to(device), Ygamma.to(device)
        with torch.no_grad():
            mu = backbone(X)
        cond = torch.cat([X, mu[:, 0:1]], dim=1) if cond_on_mean else X

        fields = torch.stack([decode(mu[:, 0:1] + sigma * draft_k_sample(
            fm, cond, n_steps, sampler, K, 1, True)) for _ in range(M)], 0)
        reward = minkowski_reward(geom_fn, fields, Ygamma, mode, anneal, w_spread,
                                  form, decode(Y[:, 0:1]))
        g_rew = grad_of(w_reward * reward)

        # anchor, as in the trainer (same surrogate x1)
        with torch.no_grad():
            x0 = _base_sample(fm, fields.shape[1:], device, X.dtype)
            x1 = (fields[0].detach() / max_val).clamp(0, 1)
            tt = torch.rand(x0.shape[0], device=device)
            xt = (1 - tt.view(-1, 1, 1, 1)) * x0 + tt.view(-1, 1, 1, 1) * x1
            v_ref = fm_ref.velocity(xt, tt, cond)
        prox = ((fm.velocity(xt, tt, cond) - v_ref) ** 2).mean()
        g_prox = grad_of(w_prox * prox)

        # retention is measured even when its weight is 0, to show what it would contribute
        loss_fm = fm.training_loss((Y[:, 0:1] - mu[:, 0:1]) / sigma, cond)
        g_fm = grad_of(w_retain * loss_fm) if w_retain > 0 else grad_of(0.0 * loss_fm)

        g_tot = g_rew + g_prox + g_fm
        n = lambda g: float(g.norm())
        cos = lambda a, b: float((a @ b) / (a.norm() * b.norm() + 1e-30))
        rows.append([n(g_rew), n(g_prox), n(g_fm), n(g_tot), cos(g_rew, g_tot)])
        print(f"batch {bi}: R={reward.item():.4f} prox={prox.item():.2e} L_FM={loss_fm.item():.4f} | "
              f"|g| reward={rows[-1][0]:.3e} prox={rows[-1][1]:.3e} fm={rows[-1][2]:.3e} "
              f"total={rows[-1][3]:.3e} cos(reward,total)={rows[-1][4]:+.3f}")

    r = np.array(rows)
    print(f"\nmean |g|: reward {r[:,0].mean():.3e}  prox {r[:,1].mean():.3e}  "
          f"fm {r[:,2].mean():.3e}  total {r[:,3].mean():.3e}")
    print(f"reward share of the total norm: {np.mean(r[:,0] / r[:,3]):.3%}   "
          f"mean cos(reward, total) {r[:,4].mean():+.3f}   (clip threshold 1.0)")


if __name__ == "__main__":
    main()
