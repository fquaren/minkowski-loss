#!/usr/bin/env python
"""Where does the flow-matching velocity loss put its gradient, by target intensity.

The companion of reward_threshold_share.py for the base objective. The FM loss is a mean over
pixels of (v - u)^2, so it splits exactly into per-bin terms L_b (pixels whose *target* value
falls in bin b). For each bin, four shares:

  pixels    share of pixels in the bin (what a mass-proportional loss would give)
  loss      share of the loss value
  |dL/dv|   share of |dL/dv| on the network output, the analogue of |dR/dz| for the reward
  param     <G_b, G> / |G|^2, with G_b the parameter gradient of L_b summed over all batches
            and G = sum_b G_b; sums to 1 (bins can be negative: they pull against the rest)

plus |G_b| / |G|. Bins are on the target field, whereas reward_threshold_share.py bins the
reward by the generated field; the loss has no generated field.

Usage:
  python scripts/evaluate/fm_loss_intensity_share.py configs/config_v4_adv_fm_evalext.yaml \
      --fm_checkpoint <fm.pth>
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.utils import load_config, load_scaler_val, set_seed
from src.models.unet import LogSpaceResidualUNet
from src.data.datasets import DeterministicSRDataset
from src.trainers.reward_finetune import _load_sigma, _make_fm

BINS = [0.0, 1.0, 10.0, 20.0, 50.0, np.inf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fm_checkpoint", required=True)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--n_batches", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--draws", type=int, default=2, help="(t, x0) draws per batch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv = float(load_scaler_val(config))

    backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    backbone.load_state_dict(torch.load(args.backbone or config["BACKBONE_CHECKPOINT"],
                                        map_location=device)["model_state_dict"])
    backbone.eval()
    sigma, sigma_src = _load_sigma(config, device)
    fm = _make_fm(config, 3, device)
    fm.load_state_dict(torch.load(args.fm_checkpoint, map_location=device)["model_state_dict"])
    fm.eval()   # no dropout noise in the gradient; the loss itself is unchanged
    params = [p for p in fm.parameters() if p.requires_grad]
    print(f"[fm-share] sigma_r: {sigma_src}  params={sum(p.numel() for p in params):,}")

    with open(config["DEM_STATS"]) as f:
        d = json.load(f)
    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config["VAL_METADATA_FILE"],
                                (float(d["dem_mean"]), float(d["dem_std"])), mv,
                                split="validation", topology_mode="euler")
    idx = np.random.default_rng(args.seed).choice(len(ds), args.batch_size * args.n_batches,
                                                  replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, num_workers=0)
    print(f"[fm-share] {config['VAL_METADATA_FILE']}: {len(idx)} patches x {args.draws} draws")

    nb = len(BINS) - 1
    G = [None] * nb
    acc = {k: np.zeros(nb) for k in ("pixels", "loss", "dv")}
    for X, Y, _ in loader:
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            mu = backbone(X)
            cond = torch.cat([X, mu[:, 0:1]], 1)
            r_true = (Y[:, 0:1] - mu[:, 0:1]) / sigma
            y = torch.relu(torch.expm1(torch.clamp(Y[:, 0:1], 0.0, 1.0) * mv))
        masks = [(y >= BINS[b]) & (y < BINS[b + 1]) for b in range(nb)]
        for _ in range(args.draws):
            # same construction as FlowMatching.training_loss, kept per pixel
            x0 = torch.randn_like(r_true)
            t = torch.rand(r_true.shape[0], device=device)
            tb = t.view(-1, 1, 1, 1)
            xt = (1 - tb) * x0 + tb * r_true
            v = fm.velocity(xt, t, cond)
            err2 = (v - (r_true - x0)) ** 2
            for b in range(nb):
                Lb = (err2 * masks[b]).sum() / err2.numel()
                gb = torch.autograd.grad(Lb, params, retain_graph=b < nb - 1, allow_unused=True)
                gb = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                for g, p in zip(gb, params)])
                G[b] = gb if G[b] is None else G[b] + gb
                e = err2.detach()
                acc["pixels"][b] += float(masks[b].sum())
                acc["loss"][b] += float((e * masks[b]).sum())
                acc["dv"][b] += float((e.sqrt() * masks[b]).sum())

    Gt = sum(G)
    nt = float(Gt.norm()) ** 2
    proj = np.array([float(G[b] @ Gt) / nt for b in range(nb)])
    norm = np.array([float(G[b].norm()) for b in range(nb)]) / nt ** 0.5
    lab = [f"[{BINS[b]:g},{BINS[b+1]:g})" for b in range(nb)]
    print(f"\n{'target mm':>10s} {'pixels%':>9s} {'loss%':>8s} {'|dL/dv|%':>9s} "
          f"{'param%':>8s} {'|G_b|/|G|':>10s} {'loss/px':>8s}")
    for b in range(nb):
        s = lambda k: 100 * acc[k][b] / acc[k].sum()
        lpp = acc["loss"][b] / max(acc["pixels"][b], 1) / (acc["loss"].sum() / acc["pixels"].sum())
        print(f"{lab[b]:>10s} {s('pixels'):9.4f} {s('loss'):8.3f} {s('dv'):9.3f} "
              f"{100*proj[b]:8.2f} {norm[b]:10.3f} {lpp:8.2f}")
    print("loss/px: mean squared velocity error per pixel in the bin, relative to the overall mean")


if __name__ == "__main__":
    main()
