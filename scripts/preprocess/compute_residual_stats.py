#!/usr/bin/env python
"""Compute residual statistics of the frozen stage-1 mean for the flow-matching stage.

Corrective (CorrDiff-style) decomposition: the frozen backbone predicts the conditional
mean mu(c); the flow-matching model will learn the residual r = x1 - mu(c) in
log-normalised [0,1] field space. This script forms r over the train split and reports/saves:

  - per-pixel residual mean (should be ~0 if the backbone is a clean conditional mean);
  - the global standard deviation sigma_r (scalar);
  - the per-pixel std map, plus its spatial coefficient of variation, to decide whether the
    flow-matching trainer should standardise the residual by a scalar or a per-pixel map.

Outputs to PREPROCESSED_DATA_DIR:
  residual_std.npy      per-pixel std map, shape [1, H, W]
  residual_stats.json   sigma_r scalar, residual-mean diagnostics, map-structure summary

Usage:
  python scripts/preprocess/compute_residual_stats.py config.yaml \
      --backbone <out_dir>/unet_best.pth --split train
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, load_scaler_val
from src.models.unet import LogSpaceResidualUNet
from src.data.datasets import DeterministicSRDataset

_META = {"train": "TRAIN_METADATA_FILE", "validation": "VAL_METADATA_FILE",
         "val": "VAL_METADATA_FILE", "test": "TEST_METADATA_FILE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--backbone", required=True, help="frozen stage-1 backbone checkpoint")
    ap.add_argument("--split", default="train", choices=list(_META))
    ap.add_argument("--data_percentage", type=float, default=100.0)
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_val = load_scaler_val(config)
    topo_mode = config.get("TOPOLOGY_MODE", "euler")
    with open(config["DEM_STATS"]) as f:
        dstats = json.load(f)
    dem_stats = (float(dstats["dem_mean"]), float(dstats["dem_std"]))

    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.backbone, map_location=device)["model_state_dict"])
    model.eval()

    ds = DeterministicSRDataset(
        config["PREPROCESSED_DATA_DIR"], config[_META[args.split]], dem_stats, scaler_val,
        split=args.split, data_percentage=args.data_percentage, topology_mode=topo_mode,
    )
    loader = DataLoader(
        ds, batch_size=config.get("BATCH_SIZE", 128), shuffle=False,
        num_workers=config.get("NUM_WORKERS", 4), pin_memory=False,
        multiprocessing_context="spawn",
    )

    # Streaming per-pixel sum and sum of squares of r = x1 - mu (both in [0,1]).
    s = ssq = None
    n = 0
    with torch.no_grad():
        for X, Y, _ in tqdm(loader, desc="residual stats"):
            X, Y = X.to(device), Y.to(device)
            mu = model(X)                       # [B,1,H,W] in [0,1]
            r = Y[:, 0:1] - mu[:, 0:1]          # residual in [0,1] space
            rs, rss = r.sum(dim=0), (r ** 2).sum(dim=0)   # [1,H,W]
            s = rs if s is None else s + rs
            ssq = rss if ssq is None else ssq + rss
            n += r.shape[0]

    mean_map = s / n                            # [1,H,W]
    std_map = torch.sqrt(torch.clamp(ssq / n - mean_map ** 2, min=0.0))  # per-pixel sigma_r
    P = std_map.numel()
    g_mean = s.sum() / (n * P)
    sigma_scalar = float(torch.sqrt(torch.clamp(ssq.sum() / (n * P) - g_mean ** 2, min=0.0)))
    cv = float(std_map.std() / (std_map.mean() + 1e-12))   # spatial variation of sigma_r

    out_dir = config["PREPROCESSED_DATA_DIR"]
    np.save(os.path.join(out_dir, "residual_std.npy"), std_map.cpu().numpy())
    stats = {
        "backbone": args.backbone,
        "split": args.split,
        "n_samples": int(n),
        "sigma_r_scalar": sigma_scalar,
        "residual_mean_abs_max": float(mean_map.abs().max()),
        "residual_mean_spatial_mean": float(mean_map.mean()),
        "std_map_min": float(std_map.min()),
        "std_map_max": float(std_map.max()),
        "std_map_mean": float(std_map.mean()),
        "std_map_cv": cv,
        "recommend_per_pixel_map": bool(cv > 0.25),
    }
    with open(os.path.join(out_dir, "residual_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nSaved residual_std.npy and residual_stats.json to {out_dir}")
    print("Interpretation: residual_mean_abs_max should be small (clean conditional mean); "
          "if recommend_per_pixel_map is true, set FM_USE_RESIDUAL_MAP: true in config, "
          "otherwise use the scalar sigma_r_scalar as FM_RESIDUAL_SCALE.")


if __name__ == "__main__":
    main()