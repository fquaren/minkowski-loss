#!/usr/bin/env python
"""Evaluate a trained emulator on the test set."""

import argparse
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils import load_config, load_scaler_val, load_emulator
from src.losses.minkowski import MinkowskiLoss
from src.data.datasets import ZarrMixupDataset
from src.evaluation.metrics import (
    create_metrics_dataframe, calculate_grouped_metrics,
    calculate_per_feature_metrics,
)
from src.evaluation.jacobian import compute_jacobian_norms, generate_saliency_maps


def run_prediction_loop(model, loader, criterion, device, topology_mode):
    model.eval()
    all_preds, all_targets, all_images = [], [], []
    all_total_losses, all_geom_losses = [], []

    with torch.no_grad():
        for inputs, log_targets in tqdm(loader, desc="Inference"):
            inputs = inputs.to(device)
            log_targets = log_targets.to(device)

            pred_phys = model(inputs)
            if topology_mode == "b0":
                pred_log = torch.log1p(pred_phys)
            else:
                pred_log = torch.sign(pred_phys) * torch.log1p(torch.abs(pred_phys))

            total, d_a, d_p, d_t = criterion(pred_log, log_targets)
            geom_batch = torch.stack([d_a, d_p, d_t], dim=1)

            # Denormalize input for visualisation
            input_phys = torch.expm1(inputs * loader.dataset.scaler_val)

            all_total_losses.append(total.cpu().numpy())
            all_geom_losses.append(geom_batch.cpu().numpy())
            all_preds.append(pred_phys.cpu().numpy())
            all_targets.append(
                torch.sign(log_targets) * torch.expm1(torch.abs(log_targets))
            .cpu().numpy())
            all_images.append(input_phys.squeeze(1).cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_images),
        np.concatenate(all_total_losses),
        np.concatenate(all_geom_losses),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to emulator checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_results/emulator")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_val = load_scaler_val(config)
    topology_mode = config.get("TOPOLOGY_MODE", "b0")
    quantiles = config["QUANTILE_LEVELS"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_emulator(args.checkpoint, config, str(device))
    criterion = MinkowskiLoss(quantile_levels=quantiles).to(device)

    # Load test data
    zarr_path = os.path.join(config["PREPROCESSED_DATA_DIR"],
                             "preprocessed_dataset.zarr")
    test_ds = ZarrMixupDataset(
        zarr_path, split="test", scaler_val=scaler_val,
        augment=False, include_original=True, include_mixup=False,
        topology_mode=topology_mode,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.get("BATCH_SIZE", 32),
        shuffle=False, num_workers=config.get("NUM_WORKERS", 4),
        pin_memory=True,
    )

    # Predictions
    preds, targets, images, total_losses, geom_losses = run_prediction_loop(
        model, test_loader, criterion, device, topology_mode,
    )

    # Metrics
    df = create_metrics_dataframe(preds, targets, images, total_losses, geom_losses)
    grouped = calculate_grouped_metrics(df)
    per_feature = calculate_per_feature_metrics(preds, targets, np.array(quantiles))

    print("\n=== Grouped Metrics ===")
    print(grouped.to_string(float_format="%.4f"))
    print("\n=== Per-Feature Mean ===")
    print(per_feature["mean_by_component"].to_string(float_format="%.4e"))

    # Jacobian analysis
    print("\nComputing Jacobian norms...")
    jac_norms = compute_jacobian_norms(model, test_loader, device, n_samples=200)
    for name, vals in jac_norms.items():
        print(f"  {name}: mean={np.mean(vals):.4f}, max={np.max(vals):.4f}")

    # Save results
    grouped.to_csv(os.path.join(args.output_dir, "grouped_metrics.csv"))
    per_feature["r2_matrix"].to_csv(os.path.join(args.output_dir, "r2_matrix.csv"))
    np.savez_compressed(
        os.path.join(args.output_dir, "predictions.npz"),
        preds=preds, targets=targets,
    )
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
