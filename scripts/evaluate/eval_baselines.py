#!/usr/bin/env python
"""Evaluate the analytical approximation baseline on the test set."""

import argparse
import os
import yaml
import numpy as np

from src.utils import load_config, load_scaler_val, load_physical_thresholds
from src.data.datasets import ZarrMixupDataset
from src.evaluation.baselines import evaluate_analytical_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--output_dir", type=str, default="eval_results/baselines")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of test set to evaluate (for speed)")
    args = parser.parse_args()

    config = load_config(args.config)
    scaler_val = load_scaler_val(config)
    phys_thresh = load_physical_thresholds(config)
    quantiles = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)
    topology_mode = config.get("TOPOLOGY_MODE", "euler")
    os.makedirs(args.output_dir, exist_ok=True)

    zarr_path = os.path.join(config["PREPROCESSED_DATA_DIR"],
                             "preprocessed_dataset.zarr")
    test_ds = ZarrMixupDataset(
        zarr_path, split="test", scaler_val=scaler_val,
        augment=False, include_original=True, include_mixup=False,
        subset_fraction=args.subset_fraction,
        topology_mode=topology_mode,
    )

    print(f"Evaluating analytical baseline on {len(test_ds)} samples...")
    metrics = evaluate_analytical_baseline(
        test_ds, phys_thresh, quantiles, scaler_val,
        pixel_size_km=config.get("PIXEL_SIZE_KM", 2.0),
        topology_mode=topology_mode,
    )

    print("\n=== Analytical Approximation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out_path = os.path.join(args.output_dir, "analytical_metrics.yaml")
    with open(out_path, "w") as f:
        yaml.dump(metrics, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
