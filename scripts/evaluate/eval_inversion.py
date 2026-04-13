#!/usr/bin/env python
"""Run feature inversion test on trained emulators.

Optimises a noise vector through the frozen emulator to match a target
gamma, then verifies the result with exact TDA computation.
"""

import argparse
import os
import yaml
import numpy as np
import torch

from src.utils import (
    load_config, load_scaler_val, load_emulator,
    load_persistence_thresholds, load_physical_thresholds,
    set_seed,
)
from src.evaluation.inversion import (
    create_synthetic_target, run_inversion, verify_inversion,
    plot_inversion_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results/inversion")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler_val = load_scaler_val(config)
    phys_thresh = load_physical_thresholds(config)
    thresh_b0, thresh_b1 = load_persistence_thresholds(config)
    topology_mode = config.get("TOPOLOGY_MODE", "b0")
    arch = config.get("ARCHITECTURE", "Constrained")
    pixel_km = config.get("PIXEL_SIZE_KM", 2.0)
    patch_size = config["PATCH_SIZE"]
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_emulator(args.checkpoint, config, device)
    quantiles = config["QUANTILE_LEVELS"]

    modes = ["large_storm", "small_storm", "no_storm"]
    all_results = {}

    for i, mode in enumerate(modes):
        set_seed(args.seed + i)
        print(f"\n--- Inversion: {mode} ---")

        log_target = create_synthetic_target(mode, quantiles, device=device)

        result = run_inversion(
            model, log_target, scaler_val,
            patch_size=patch_size, steps=args.steps, device=device,
        )

        # Quantitative verification via exact TDA
        verification = verify_inversion(
            result["final_phys"], log_target, phys_thresh,
            pixel_km, thresh_b0, thresh_b1,
            topology_mode=topology_mode,
        )

        print(f"  Inversion RMSE (log-space): {verification['rmse']:.4f}")

        # Plot
        plot_inversion_results(
            result["initial_phys"], result["final_phys"],
            arch, mode, args.output_dir,
        )

        # Save raw data
        np.savez_compressed(
            os.path.join(args.output_dir, f"inversion_{mode}_{arch}.npz"),
            initial=result["initial_phys"],
            final=result["final_phys"],
            loss_total=result["loss_history"]["total"],
            loss_mse=result["loss_history"]["mse"],
            target=log_target.cpu().numpy(),
            exact_gamma=verification["exact_gamma_log"],
            residual=verification["residual"],
        )

        all_results[mode] = {
            "rmse": verification["rmse"],
            "final_mse": result["loss_history"]["mse"][-1],
        }

    # Summary
    print("\n=== Inversion Summary ===")
    for mode, r in all_results.items():
        print(f"  {mode}: RMSE={r['rmse']:.4f}, final_MSE={r['final_mse']:.6f}")

    with open(os.path.join(args.output_dir, "inversion_summary.yaml"), "w") as f:
        yaml.dump(all_results, f)


if __name__ == "__main__":
    main()
