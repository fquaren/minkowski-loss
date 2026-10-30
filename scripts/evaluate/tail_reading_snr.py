#!/usr/bin/env python
"""Signal and noise of each reward feature in the tail, on a real FM ensemble.

For every condition, M members are drawn from the FM model and each reward feature is computed
per threshold u_j, only where the target has n_j = #{y >= u_j} >= 1:

  count  log1p A(u_j)       the threshold form's area channel (vertical reading)
  rank   log1p f_(n_j)      the quantile_mass area channel (horizontal reading)
  perim  log1p P(u_j)
  euler  signed log1p chi(u_j)

Per patch: d = mean over members - target, s = sd over members. Per threshold, over patches:
  bias/spread  mean(d) / rms(s): systematic mismatch in units of member noise
  E|d|/spread  mean|d| / rms(s): ~0.82 for a calibrated Gaussian ensemble of 16
  spread       rms(s), in the feature's own units (what the reward's L1 sums)
plus the share of members whose maximum is below u_j ("out of reach").

The two area readings are the same survival curve read along two axes; to first order their
bias/spread ratios agree, so a difference measures the nonlinearity (log1p floor at zero
count, out-of-reach members) rather than noise alone.

Usage:
  python scripts/evaluate/tail_reading_snr.py configs/config_v4_adv_fm_evalext.yaml \
      --fm_checkpoint <fm.pth>
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
from src.trainers.reward_finetune import reward_features, _load_sigma, _make_fm

CH = ["count", "rank", "perim", "euler"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fm_checkpoint", required=True)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--ensemble", type=int, default=16)
    ap.add_argument("--n_batches", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv = float(load_scaler_val(config))
    u = load_physical_thresholds(config)
    anneal = float(config.get("REWARD_ANNEAL", 0.05))
    n_steps = int(config.get("FM_SAMPLE_STEPS", 16)); sampler = config.get("FM_SAMPLER", "heun")

    backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    backbone.load_state_dict(torch.load(args.backbone or config["BACKBONE_CHECKPOINT"],
                                        map_location=device)["model_state_dict"])
    backbone.eval()
    sigma, sigma_src = _load_sigma(config, device)
    fm = _make_fm(config, 3, device)
    fm.load_state_dict(torch.load(args.fm_checkpoint, map_location=device)["model_state_dict"])
    fm.eval()
    geom = AnalyticalMinkowskiLoss(physical_thresholds=u,
                                   pixel_size_km=float(config.get("PIXEL_SIZE_KM", 2.0)),
                                   topology_mode="euler", area_mode="ste").to(device)
    print(f"[snr] M={args.ensemble} anneal={anneal} {sampler}-{n_steps} sigma_r: {sigma_src}")

    with open(config["DEM_STATS"]) as f:
        d = json.load(f)
    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config["VAL_METADATA_FILE"],
                                (float(d["dem_mean"]), float(d["dem_std"])), mv,
                                split="validation", topology_mode="euler")
    idx = np.random.default_rng(args.seed).choice(len(ds), args.batch_size * args.n_batches,
                                                  replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, num_workers=0)
    print(f"[snr] {config['VAL_METADATA_FILE']}: {len(idx)} patches")

    def decode(z):
        return torch.relu(torch.expm1(torch.clamp(z, 0.0, 1.0) * mv))

    D, S, V, R = [], [], [], []          # per batch: [B,4,Q] mean diff, sd, valid; [B,Q] reach
    uu = torch.as_tensor(u, device=device, dtype=torch.float32)
    with torch.no_grad():
        for X, Y, Yg in loader:
            X, Y, Yg = X.to(device), Y.to(device), Yg.to(device)
            mu = backbone(X)
            cond = torch.cat([X, mu[:, 0:1]], 1)
            f = torch.stack([decode(mu[:, 0:1] + sigma * fm.sample(cond, n_steps=n_steps,
                                                                  method=sampler, in_channels=1))
                             for _ in range(args.ensemble)], 0)          # [M,B,1,H,W]
            y = decode(Y[:, 0:1])
            lg, tg, _ = reward_features(geom, f, Yg, anneal, "threshold")
            lq, tq, W = reward_features(geom, f, Yg, anneal, "quantile_mass", y)
            feat = torch.cat([lg[:, :, :1], lq[:, :, :1], lg[:, :, 1:]], 2)   # [M,B,4,Q]
            targ = torch.cat([tg[:, :1], tq[:, :1], tg[:, 1:]], 1)            # [B,4,Q]
            valid = W[:, 0] > 0                                               # n_j >= 1
            D.append((feat.mean(0) - targ).cpu())
            S.append(feat.std(0, unbiased=True).cpu())
            V.append(valid.cpu())
            fmax = f.flatten(2).amax(2)                                       # [M,B]
            R.append((fmax.unsqueeze(-1) < uu).float().mean(0).cpu())         # [B,Q]

    D, S, V, R = (torch.cat(a, 0).numpy() for a in (D, S, V, R))
    print(f"\n{'u mm':>6s} {'n_pat':>5s} {'reach0%':>7s} | " +
          " | ".join(f"{c:>5s} b/s  E|d|/s  sprd" for c in CH))
    for j in range(len(u)):
        m = V[:, j]
        if m.sum() < 5:
            continue
        cells = []
        for c in range(4):
            dd, ss = D[m, c, j], S[m, c, j]
            sp = np.sqrt((ss ** 2).mean())
            cells.append(f"{dd.mean()/sp:+9.2f} {np.abs(dd).mean()/sp:7.2f} {sp:5.3f}")
        print(f"{float(u[j]):6.3g} {int(m.sum()):5d} {100*R[m, j].mean():7.1f} | " + " | ".join(cells))
    print("b/s = bias/spread; E|d|/s ~0.82 if calibrated (Gaussian, M=16); "
          "reach0% = members whose maximum is below u")


if __name__ == "__main__":
    main()
