#!/usr/bin/env python
"""Evaluate the deterministic SR backbone (LogSpaceResidualUNet) on a held-out split.

Study-1 control. Field-space super-resolution metrics:
  - MAE / RMSE: overall, by target-intensity group, and on extreme patches;
  - RAPSD log-distance (spectral fidelity);
  - SAL (structure-amplitude-location);
  - analytical Minkowski distance (predicted-field gamma-hat vs offline target gamma);
  - isoperimetric-violation rate of the predicted gamma.

CRPS is intentionally omitted: for a single deterministic field it equals the MAE; it
enters with the generative ensemble in Phase 2 (see src/evaluation/extreme_metrics.py).

Usage:
    python scripts/evaluate/eval_backbone.py config.yaml \
        --checkpoint <out_dir>/unet_best.pth --split test
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import (
    load_config,
    load_scaler_val,
    load_physical_thresholds,
    load_persistence_thresholds,
)
from src.models.unet import LogSpaceResidualUNet
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.evaluation.metrics import compute_isoperimetric_violation
from src.evaluation.training_metrics import compute_radial_power_spectrum
from src.evaluation.extreme_metrics import sal

_META_KEY = {"test": "TEST_METADATA_FILE", "validation": "VAL_METADATA_FILE",
             "val": "VAL_METADATA_FILE", "train": "TRAIN_METADATA_FILE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--checkpoint", required=True, help="backbone checkpoint (unet_best.pth)")
    ap.add_argument("--split", default="test", choices=list(_META_KEY))
    ap.add_argument("--output_dir", default="eval_results/backbone")
    ap.add_argument("--extreme_pct", type=float, default=99.0,
                    help="target patch-max percentile defining extreme patches")
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    scaler_val = load_scaler_val(config)
    max_val = torch.tensor(scaler_val, device=device, dtype=torch.float32)
    pixel_km = float(config.get("PIXEL_SIZE_KM", 2.0))
    topo_mode = config.get("TOPOLOGY_MODE", "euler")
    quantiles = np.asarray(config["QUANTILE_LEVELS"], np.float32)
    u_phys = load_physical_thresholds(config)
    with open(config["DEM_STATS"]) as f:
        dstats = json.load(f)
    dem_stats = (float(dstats["dem_mean"]), float(dstats["dem_std"]))

    # Model
    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Analytical Minkowski functionals (same configuration as training)
    thresh_b0 = load_persistence_thresholds(config)[0] if topo_mode == "b0" else 0.0
    geom_fn = AnalyticalMinkowskiLoss(
        physical_thresholds=u_phys, quantile_levels=quantiles, pixel_size_km=pixel_km,
        topology_mode=topo_mode, area_mode="ste", persistence_thresh_b0=thresh_b0,
    ).to(device)

    # Data
    test_ds = DeterministicSRDataset(
        config["PREPROCESSED_DATA_DIR"],
        config[_META_KEY[args.split]],
        dem_stats,
        scaler_val,
        split=args.split,
        topology_mode=topo_mode,
    )
    loader = DataLoader(
        test_ds, batch_size=config.get("BATCH_SIZE", 32), shuffle=False,
        num_workers=config.get("NUM_WORKERS", 4), pin_memory=False,
        multiprocessing_context="spawn",
    )

    mae_i, rmse_i, tmean_i, tmax_i = [], [], [], []
    gamma_tgt = []
    S_i, A_i, L_i, gamma_hat = [], [], [], []
    mink_sum, n_batches = 0.0, 0
    spec_i = []
    rapsd_p = rapsd_t = None

    with torch.no_grad():
        for X, Y, Ygamma in tqdm(loader, desc="eval backbone"):
            X, Y, Ygamma = X.to(device), Y.to(device), Ygamma.to(device)
            Y_pred = model(X)
            # log-normalised [0,1] -> physical mm/h (identical to the trainer's conversion)
            pred_phys = torch.relu(torch.expm1(torch.clamp(Y_pred[:, 0:1] * max_val, max=7.0)))
            target_phys = torch.relu(torch.expm1(Y[:, 0:1] * max_val))

            err = pred_phys - target_phys
            mae_i.append(err.abs().mean(dim=(1, 2, 3)).cpu().numpy())
            rmse_i.append(torch.sqrt((err ** 2).mean(dim=(1, 2, 3))).cpu().numpy())
            tmean_i.append(target_phys.mean(dim=(1, 2, 3)).cpu().numpy())
            tmax_i.append(target_phys.amax(dim=(1, 2, 3)).cpu().numpy())

            mink_sum += geom_fn(pred_phys, Ygamma, anneal_factor=0.05).item()
            n_batches += 1
            a, p, t = geom_fn._functionals(pred_phys, anneal_factor=0.05)
            gamma_hat.append(torch.stack([a, p, t], dim=1).cpu().numpy())  # [B,3,Q]
            gamma_tgt.append(Ygamma.cpu().numpy())   # log-space target, for the gamma-curve plots

            rp_b = compute_radial_power_spectrum(pred_phys)     # [B,K]
            rt_b = compute_radial_power_spectrum(target_phys)
            # Per-sample log-spectral distance, kept for the perception-distortion cloud.
            # The batch means below are unchanged, so rapsd_log_distance is untouched.
            spec_i.append((torch.log(rp_b + 1e-12) - torch.log(rt_b + 1e-12))
                          .abs().mean(dim=-1).cpu().numpy())
            rp, rt = rp_b.mean(dim=0), rt_b.mean(dim=0)
            rapsd_p = rp if rapsd_p is None else rapsd_p + rp
            rapsd_t = rt if rapsd_t is None else rapsd_t + rt

            pp = pred_phys.squeeze(1).cpu().numpy()
            tt = target_phys.squeeze(1).cpu().numpy()
            for i in range(pp.shape[0]):
                s, acomp, l = sal(pp[i], tt[i])
                S_i.append(s); A_i.append(acomp); L_i.append(l)

    mae, rmse = np.concatenate(mae_i), np.concatenate(rmse_i)
    tmean, tmax = np.concatenate(tmean_i), np.concatenate(tmax_i)
    gamma_hat = np.concatenate(gamma_hat)
    gamma_tgt = np.concatenate(gamma_tgt)
    S, A, L = np.array(S_i), np.array(A_i), np.array(L_i)
    spectral_dist = np.concatenate(spec_i)
    rapsd_p = (rapsd_p / n_batches).cpu().numpy()
    rapsd_t = (rapsd_t / n_batches).cpu().numpy()

    thr = float(np.percentile(tmax, args.extreme_pct))
    ext = tmax >= thr

    def grp(mask):
        mask = np.asarray(mask, bool)
        if not mask.any():
            return {"n": 0}
        return {"n": int(mask.sum()), "mae": float(mae[mask].mean()),
                "rmse": float(rmse[mask].mean())}

    q1, q2, q3 = np.percentile(tmean, [25, 50, 75])
    groups = {
        "all": grp(np.ones_like(mae, bool)),
        "q1_low": grp(tmean <= q1),
        "q2": grp((tmean > q1) & (tmean <= q2)),
        "q3": grp((tmean > q2) & (tmean <= q3)),
        "q4_high": grp(tmean > q3),
        f"extreme_top{100 - args.extreme_pct:.0f}pct": grp(ext),
    }

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": int(mae.size),
        "topology_mode": topo_mode,
        "mae_overall": float(mae.mean()),
        "rmse_overall": float(rmse.mean()),
        "mae_extreme": float(mae[ext].mean()),
        "rmse_extreme": float(rmse[ext].mean()),
        "extreme_threshold_mmph": thr,
        "minkowski_distance": float(mink_sum / max(n_batches, 1)),
        "isoperimetric_violation_pct": float(compute_isoperimetric_violation(gamma_hat)),
        "rapsd_log_distance": float(np.mean(np.abs(
            np.log(rapsd_p + 1e-12) - np.log(rapsd_t + 1e-12)))),
        "SAL_median": [float(np.median(S)), float(np.median(A)), float(np.median(L))],
        "SAL_mean": [float(S.mean()), float(A.mean()), float(L.mean())],
        "groups_mae_rmse": groups,
        "note_crps": "omitted: deterministic CRPS == MAE; CRPS enters with the Phase-2 ensemble",
    }
    with open(os.path.join(args.output_dir, "backbone_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    np.savez_compressed(
        os.path.join(args.output_dir, "backbone_arrays.npz"),
        mae=mae, rmse=rmse, tmean=tmean, tmax=tmax, S=S, A=A, L=L,
        spectral_dist=spectral_dist,
        rapsd_pred=rapsd_p, rapsd_target=rapsd_t, gamma_hat=gamma_hat,
        gamma_target=gamma_tgt, thresholds=np.asarray(u_phys, dtype=np.float32),
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
