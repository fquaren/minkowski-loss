#!/usr/bin/env python
"""Evaluate the clean flow-matching baseline (Phase 2) by sampling.

Unlike the deterministic backbone, a generative model must be judged on draws:
  - structural metrics (SAL, RAPSD, Minkowski, isoperimetric) are computed on a SINGLE
    ensemble member, because the ensemble mean re-smooths and would hide the very structure
    the generative model is meant to recover;
  - accuracy is judged by CRPS (proper probabilistic score over the ensemble) and by the
    ensemble-mean MAE/RMSE (the best point estimate);
  - the dry-tail bias -- the ratio of generated to target intensity at high quantiles
    (q = 0.99, 0.999) -- is measured explicitly; it is the Phase-3 coupling baseline.

Rows are directly comparable to eval_backbone.py (same conversions, same anneal).

Usage:
  python scripts/evaluate/eval_fm_baseline.py config.yaml \
      --fm_checkpoint runs/sr_flow_matching/<run>/fm_best.pth \
      --backbone runs/sr_analytical/<run>/unet_best.pth \
      --split test --ensemble 16 --max_batches 100
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
    load_config, load_scaler_val, load_physical_thresholds, load_persistence_thresholds,
)
from src.models.unet import LogSpaceResidualUNet
from src.models.flow_matching import FlowMatching
from src.data.datasets import DeterministicSRDataset
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.evaluation.metrics import compute_isoperimetric_violation
from src.evaluation.training_metrics import compute_radial_power_spectrum
from src.evaluation.extreme_metrics import sal, crps_ensemble

_META = {"test": "TEST_METADATA_FILE", "validation": "VAL_METADATA_FILE",
         "val": "VAL_METADATA_FILE", "train": "TRAIN_METADATA_FILE"}


def _load_sigma(config, device):
    """Resolve sigma_r, returning (sigma, provenance). Must match the trainer exactly."""
    if config.get("FM_USE_RESIDUAL_MAP", False):
        path = config.get("FM_RESIDUAL_STD_PATH",
                          os.path.join(config["PREPROCESSED_DATA_DIR"], "residual_std.npy"))
        t = torch.from_numpy(np.load(path).astype(np.float32)).to(device).clamp_min(1e-3)
        return t.unsqueeze(0), f"per-pixel map {path}"
    if "FM_RESIDUAL_SCALE" in config and config["FM_RESIDUAL_SCALE"] is not None:
        return float(config["FM_RESIDUAL_SCALE"]), f"config FM_RESIDUAL_SCALE={float(config['FM_RESIDUAL_SCALE']):.4g}"
    stats_path = os.path.join(config["PREPROCESSED_DATA_DIR"], "residual_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            sigma = float(json.load(f)["sigma_r_scalar"])
        return sigma, f"residual_stats.json sigma_r_scalar={sigma:.4g}"
    return 1.0, "DEFAULT 1.0 (no standardisation)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fm_checkpoint", required=True)
    ap.add_argument("--backbone", default=None, help="overrides BACKBONE_CHECKPOINT")
    ap.add_argument("--split", default="test", choices=list(_META))
    ap.add_argument("--ensemble", type=int, default=16, help="ensemble size for CRPS")
    ap.add_argument("--steps", type=int, default=None, help="ODE steps (default FM_SAMPLE_STEPS)")
    ap.add_argument("--sampler", default=None, help="euler|heun (default FM_SAMPLER)")
    ap.add_argument("--extreme_pct", type=float, default=99.0)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="cap batches (sampling is expensive; None = full split)")
    ap.add_argument("--output_dir", default="eval_results/fm_baseline")
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
    sigma, sigma_src = _load_sigma(config, device)
    print(f"[eval] residual scale sigma_r: {sigma_src}")
    condition_on_mean = config.get("FM_CONDITION_ON_MEAN", True)
    steps = args.steps or config.get("FM_SAMPLE_STEPS", 16)
    sampler = args.sampler or config.get("FM_SAMPLER", "heun")
    with open(config["DEM_STATS"]) as f:
        dstats = json.load(f)
    dem_stats = (float(dstats["dem_mean"]), float(dstats["dem_std"]))

    # models
    backbone = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    bpath = args.backbone or config["BACKBONE_CHECKPOINT"]
    backbone.load_state_dict(torch.load(bpath, map_location=device)["model_state_dict"])
    backbone.eval()
    fm = FlowMatching(in_channels=1, c_in_condition=3 if condition_on_mean else 2,
                      time_scale=config.get("FM_TIME_SCALE", 1000.0), device=str(device)).to(device)
    fm.load_state_dict(torch.load(args.fm_checkpoint, map_location=device)["model_state_dict"])
    fm.eval()

    thresh_b0 = load_persistence_thresholds(config)[0] if topo_mode == "b0" else 0.0
    geom_fn = AnalyticalMinkowskiLoss(
        physical_thresholds=u_phys, quantile_levels=quantiles, pixel_size_km=pixel_km,
        topology_mode=topo_mode, area_mode="ste", persistence_thresh_b0=thresh_b0,
    ).to(device)

    ds = DeterministicSRDataset(config["PREPROCESSED_DATA_DIR"], config[_META[args.split]],
                                dem_stats, scaler_val, split=args.split, topology_mode=topo_mode)
    loader = DataLoader(ds, batch_size=config.get("BATCH_SIZE", 128), shuffle=False,
                        num_workers=config.get("NUM_WORKERS", 4), pin_memory=False,
                        multiprocessing_context="spawn")

    M = args.ensemble
    mae_mean_i, rmse_mean_i, mae_m0_i, crps_i = [], [], [], []
    tmean_i, tmax_i, S_i, A_i, L_i = [], [], [], [], []
    dt_ratio = {0.99: [], 0.999: []}
    gamma_hat, mink_sum, nb = [], 0.0, 0
    rapsd_m0 = rapsd_mean = rapsd_tgt = None

    def _com_sample(cond):
        return sigma * fm.sample(cond, n_steps=steps, method=sampler, in_channels=1)

    with torch.no_grad():
        for bi, (X, Y, Ygamma) in enumerate(tqdm(loader, desc="fm baseline")):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            X, Y, Ygamma = X.to(device), Y.to(device), Ygamma.to(device)
            mu = backbone(X)
            cond = torch.cat([X, mu[:, 0:1]], dim=1) if condition_on_mean else X
            ens = torch.stack([
                torch.relu(torch.expm1(torch.clamp(mu[:, 0:1] + _com_sample(cond), 0.0, 1.0) * max_val))
                for _ in range(M)
            ], dim=0)                                   # [M,B,1,H,W]
            target_phys = torch.relu(torch.expm1(Y[:, 0:1] * max_val))
            mean_pred, m0 = ens.mean(0), ens[0]

            em = (mean_pred - target_phys)
            mae_mean_i.append(em.abs().mean(dim=(1, 2, 3)).cpu().numpy())
            rmse_mean_i.append(torch.sqrt((em ** 2).mean(dim=(1, 2, 3))).cpu().numpy())
            mae_m0_i.append((m0 - target_phys).abs().mean(dim=(1, 2, 3)).cpu().numpy())
            tmean_i.append(target_phys.mean(dim=(1, 2, 3)).cpu().numpy())
            tmax_i.append(target_phys.amax(dim=(1, 2, 3)).cpu().numpy())

            # CRPS over the ensemble (per pixel -> per sample)
            crps = crps_ensemble(ens[:, :, 0].cpu().numpy(), target_phys[:, 0].cpu().numpy())
            crps_i.append(crps.mean(axis=(1, 2)))

            # structural metrics on a single member (m0)
            mink_sum += geom_fn(m0, Ygamma, anneal_factor=0.05).item(); nb += 1
            a, p, t = geom_fn._functionals(m0, anneal_factor=0.05)
            gamma_hat.append(torch.stack([a, p, t], dim=1).cpu().numpy())
            rm0 = compute_radial_power_spectrum(m0).mean(0)
            rmn = compute_radial_power_spectrum(mean_pred).mean(0)
            rtg = compute_radial_power_spectrum(target_phys).mean(0)
            rapsd_m0 = rm0 if rapsd_m0 is None else rapsd_m0 + rm0
            rapsd_mean = rmn if rapsd_mean is None else rapsd_mean + rmn
            rapsd_tgt = rtg if rapsd_tgt is None else rapsd_tgt + rtg

            pp, tt = m0.squeeze(1).cpu().numpy(), target_phys.squeeze(1).cpu().numpy()
            for i in range(pp.shape[0]):
                s, ac, l = sal(pp[i], tt[i]); S_i.append(s); A_i.append(ac); L_i.append(l)

            # dry-tail bias: per-patch high-quantile ratio (generated / target)
            for q in (0.99, 0.999):
                pq = torch.quantile(m0.flatten(1), q, dim=1)
                tq = torch.quantile(target_phys.flatten(1), q, dim=1)
                r = (pq / tq.clamp_min(1e-6)).cpu().numpy()
                r[tq.cpu().numpy() < 1e-3] = np.nan       # ignore ~dry patches
                dt_ratio[q].append(r)

    mae_mean = np.concatenate(mae_mean_i); rmse_mean = np.concatenate(rmse_mean_i)
    mae_m0 = np.concatenate(mae_m0_i); crps = np.concatenate(crps_i)
    tmean = np.concatenate(tmean_i); tmax = np.concatenate(tmax_i)
    S, A, L = np.array(S_i), np.array(A_i), np.array(L_i)
    gamma_hat = np.concatenate(gamma_hat)
    rapsd_m0 = (rapsd_m0 / nb).cpu().numpy(); rapsd_mean = (rapsd_mean / nb).cpu().numpy()
    rapsd_tgt = (rapsd_tgt / nb).cpu().numpy()
    thr = float(np.percentile(tmax, args.extreme_pct)); ext = tmax >= thr

    def _rlog(a, b): return float(np.mean(np.abs(np.log(a + 1e-12) - np.log(b + 1e-12))))
    def _nanmed(x): return float(np.nanmedian(np.concatenate(x)))
    def _nanmed_ext(x):
        v = np.concatenate(x)
        return float(np.nanmedian(v[ext])) if ext.any() else float("nan")

    summary = {
        "fm_checkpoint": args.fm_checkpoint, "backbone": bpath, "split": args.split,
        "ensemble_size": M, "ode_steps": steps, "sampler": sampler,
        "n_samples": int(mae_mean.size), "topology_mode": topo_mode,
        "crps_mean": float(crps.mean()), "crps_extreme": float(crps[ext].mean()),
        "mae_ensemble_mean": float(mae_mean.mean()), "rmse_ensemble_mean": float(rmse_mean.mean()),
        "mae_single_member": float(mae_m0.mean()),
        "mae_extreme_ensemble_mean": float(mae_mean[ext].mean()),
        "extreme_threshold_mmph": thr,
        "minkowski_distance_member": float(mink_sum / max(nb, 1)),
        "isoperimetric_violation_pct_member": float(compute_isoperimetric_violation(gamma_hat)),
        "rapsd_logdist_member": _rlog(rapsd_m0, rapsd_tgt),
        "rapsd_logdist_ensemble_mean": _rlog(rapsd_mean, rapsd_tgt),
        "SAL_median_member": [float(np.median(S)), float(np.median(A)), float(np.median(L))],
        "SAL_mean_member": [float(S.mean()), float(A.mean()), float(L.mean())],
        "dry_tail_ratio_q99_median": _nanmed(dt_ratio[0.99]),
        "dry_tail_ratio_q999_median": _nanmed(dt_ratio[0.999]),
        "dry_tail_ratio_q99_extreme_median": _nanmed_ext(dt_ratio[0.99]),
        "dry_tail_ratio_q999_extreme_median": _nanmed_ext(dt_ratio[0.999]),
        "note": "structural metrics on a single member; CRPS on the ensemble; "
                "dry_tail_ratio<1 => generated tail weaker than target (dry-tail bias).",
    }
    with open(os.path.join(args.output_dir, "fm_baseline_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    np.savez_compressed(
        os.path.join(args.output_dir, "fm_baseline_arrays.npz"),
        crps=crps, mae_ensemble_mean=mae_mean, mae_single_member=mae_m0,
        tmean=tmean, tmax=tmax, S=S, A=A, L=L,
        rapsd_member=rapsd_m0, rapsd_ensemble_mean=rapsd_mean, rapsd_target=rapsd_tgt,
        gamma_hat=gamma_hat,
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()