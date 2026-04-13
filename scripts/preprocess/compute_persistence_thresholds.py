#!/usr/bin/env python
"""Compute empirical persistence noise thresholds from training data."""

import argparse
import os
import yaml
from src.utils import load_config
from src.data.gamma import compute_persistence_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    config = load_config(args.config)
    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    max_workers = config.get("MAX_WORKERS", 4)

    result = compute_persistence_thresholds(
        zarr_path,
        num_samples=args.samples,
        target_percentile=args.percentile,
        max_workers=max_workers,
    )

    out = {
        "PERSISTENCE_THRESHOLD_B0": result["thresh_b0"],
        "PERSISTENCE_THRESHOLD_B1": result["thresh_b1"],
        "PERSISTENCE_THRESHOLD_UNIFIED": result["unified"],
    }

    out_path = os.path.join(config["PREPROCESSED_DATA_DIR"],
                            "persistence_thresholds.yaml")
    with open(out_path, "w") as f:
        yaml.dump(out, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
