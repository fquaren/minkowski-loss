#!/usr/bin/env python
"""Compute 4-channel gamma targets [A, P, B0, B1] and append to Zarr."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zarr
from tqdm import tqdm

from src.data.gamma import compute_climatological_thresholds, compute_gamma_matrix
from src.utils import load_config, load_persistence_thresholds


def _worker(args):
    start, end, zarr_path, split, thresholds, pixel_km, tb0, tb1 = args
    store = zarr.open(zarr_path, mode="r+")
    group = store[split]
    precip = group["original_precip"][start:end]
    gamma = np.zeros((precip.shape[0], 4, len(thresholds)), dtype=np.float32)
    for i in range(precip.shape[0]):
        gamma[i] = compute_gamma_matrix(
            precip[i],
            thresholds,
            pixel_km,
            tb0,
            tb1,
        )
    group["gamma_targets"][start:end] = gamma
    return f"{split} {start}:{end}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    quantiles = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)
    pixel_km = config.get("PIXEL_SIZE_KM", 2.0)
    thresh_b0, thresh_b1 = load_persistence_thresholds(config)
    chunk_size = config.get("WORKER_CHUNK_SIZE", 500)
    max_workers = config.get("MAX_WORKERS", 4)

    # Compute physical thresholds from training CDF
    phys_thresh = compute_climatological_thresholds(zarr_path, quantiles)
    thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"
    )
    np.save(thresh_path, phys_thresh)
    print(f"Physical thresholds saved to {thresh_path}")

    store = zarr.open(zarr_path, mode="r+")

    for split in ["train", "validation", "test"]:
        if split not in store:
            continue

        group = store[split]
        n = group["original_precip"].shape[0]
        n_q = len(phys_thresh)

        # Store persistence thresholds as attributes
        group.attrs["persistence_threshold_b0"] = float(thresh_b0)
        group.attrs["persistence_threshold_b1"] = float(thresh_b1)

        if "gamma_targets" not in group:
            group.create_dataset(
                "gamma_targets",
                shape=(n, 4, n_q),
                chunks=(chunk_size, 4, n_q),
                dtype="float32",
            )

        tasks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            tasks.append(
                (
                    start,
                    end,
                    zarr_path,
                    split,
                    phys_thresh,
                    pixel_km,
                    thresh_b0,
                    thresh_b1,
                )
            )

        print(f"\n--- Computing gamma targets: {split} ({n} samples) ---")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(tasks)):
                f.result()

    print("\nGamma targets complete.")


if __name__ == "__main__":
    main()
