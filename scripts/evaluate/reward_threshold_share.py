#!/usr/bin/env python
"""Where does the reward's gradient go: by threshold, by channel, by pixel intensity.

Samples M members from a flow-matching checkpoint on a few batches, then differentiates the
reward with respect to the model-space value z of every pixel (field = expm1(z * max_val)),
which is where the gradient enters the network. Two tables per reward form:

  * share of the total |dR/dz| contributed by each (channel, threshold) term, isolated by
    letting the gradient through that single entry of the curves;
  * share of |dR/dz| landing on pixels in each intensity bin of the generated field.

A reward that "reaches heavy rain" puts a non-negligible share on the high thresholds and on
pixels above ~20 mm. The original threshold form puts ~all of it on light rain (DECISIONS §5).

Usage:
  python scripts/evaluate/reward_threshold_share.py configs/config_v4_adv_fm_evalext.yaml \
      --fm_checkpoint <fm.pth> --backbone <unet.pth> --forms threshold quantile_mass
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
    reward_features, reward_from_features, _load_sigma, _make_fm,
)

BINS = [0.0, 1.0, 10.0, 20.0, 50.0, np.inf]
CH = ["area", "perim", "euler"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fm_checkpoint", required=True)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--mode", default="energy")
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--forms", nargs="+", default=["threshold", "quantile_mass"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0],
                    help="mass-weight exponents tried for quantile_mass")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv = float(load_scaler_val(config))
    u = load_physical_thresholds(config)
    anneal = float(config.get("REWARD_ANNEAL", 0.05))
    w_spread = float(config.get("REWARD_SPREAD_WEIGHT", 0.5))
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
    print(f"[share] mode={args.mode} M={args.ensemble} anneal={anneal} sigma_r: {sigma_src}")

    with open(config["DEM_STATS"]) as f:
        d = json.load(f)
    key = "VAL_METADATA_FILE" if args.split == "val" else "TRAIN_METADATA_FILE"
    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config[key],
                                (float(d["dem_mean"]), float(d["dem_std"])), mv,
                                split="validation" if args.split == "val" else "train",
                                topology_mode="euler")
    idx = np.random.default_rng(args.seed).choice(len(ds), args.batch_size * args.n_batches,
                                                  replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, num_workers=0)

    def decode(z):
        return torch.relu(torch.expm1(torch.clamp(z, 0.0, 1.0) * mv))

    variants = [("threshold", None)] if "threshold" in args.forms else []
    if "quantile_mass" in args.forms:
        variants += [("quantile_mass", a) for a in args.alphas]
    Q = len(u)
    acc = {v: {"term": np.zeros((3, Q)), "bin": np.zeros(len(BINS) - 1)} for v in variants}

    for X, Y, Yg in loader:
        X, Y, Yg = X.to(device), Y.to(device), Yg.to(device)
        with torch.no_grad():
            mu = backbone(X)
            cond = torch.cat([X, mu[:, 0:1]], 1)
            z0 = torch.stack([mu[:, 0:1] + sigma * fm.sample(cond, n_steps=n_steps, method=sampler,
                                                             in_channels=1)
                              for _ in range(args.ensemble)], 0)
        y = decode(Y[:, 0:1])
        for (form, alpha) in variants:
            z = z0.clone().requires_grad_(True)
            f = decode(z)
            kw = {} if alpha is None else {"alpha": alpha}
            lg, tg, W = reward_features(geom, f, Yg, anneal, form, y, **kw)
            live = lg - lg.detach()
            tot = None
            for c in range(3):
                for j in range(Q):
                    m = torch.zeros_like(lg); m[:, :, c, j] = 1.0
                    r = reward_from_features(geom, lg.detach() + m * live, tg, W, args.mode, w_spread)
                    g, = torch.autograd.grad(r, z, retain_graph=True, allow_unused=True)
                    if g is None:
                        continue
                    acc[(form, alpha)]["term"][c, j] += float(g.abs().sum())
                    tot = g.abs() if tot is None else tot + g.abs()
            fv = f.detach()
            for b in range(len(BINS) - 1):
                sel = (fv >= BINS[b]) & (fv < BINS[b + 1])
                acc[(form, alpha)]["bin"][b] += float(tot[sel].sum())

    for v in variants:
        t = acc[v]["term"]; t = t / t.sum()
        bn = acc[v]["bin"]; bn = bn / bn.sum()
        name = v[0] if v[1] is None else f"{v[0]} (alpha={v[1]:g})"
        print(f"\n=== {name}: share of |dR/dz| per threshold (%) ===")
        print(f"{'u (mm)':>8s} " + " ".join(f"{c:>7s}" for c in CH) + f" {'all':>7s}")
        for j in range(Q):
            print(f"{float(u[j]):8.3g} " + " ".join(f"{100*t[c, j]:7.2f}" for c in range(3))
                  + f" {100*t[:, j].sum():7.2f}")
        print(f"{'channel':>8s} " + " ".join(f"{100*t[c].sum():7.2f}" for c in range(3)))
        print("share by pixel intensity: " + "  ".join(
            f"[{BINS[b]:g},{BINS[b+1]:g}) {100*bn[b]:.2f}%" for b in range(len(BINS) - 1)))


if __name__ == "__main__":
    main()
