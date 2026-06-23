#!/usr/bin/env python
"""Compute 5-channel gamma targets [A, P_crofton, B0, B1, chi_exact] and append to Zarr.

Updated for the rewritten src/data/gamma.py:
  - perimeter (row 1) is now the isotropic Cauchy-Crofton estimator, matching the
    differentiable loss estimator (was marching squares);
  - a new row 4, chi_exact, is the exact 4-connected Euler characteristic
    (skimage.euler_number, connectivity=1), the loss-consistent target for
    TOPOLOGY_MODE="euler" (B0 - B1 from GUDHI is 8-connected + persistence-filtered
    and is NOT used as the euler target).

Channel layout (axis 1):
    0: A          area (km^2)
    1: P_crofton  perimeter (km), Cauchy-Crofton
    2: B0         persistence-filtered connected components  -> TOPOLOGY_MODE="b0"
    3: B1         persistence-filtered holes (diagnostics)
    4: chi_exact  exact 4-connected Euler characteristic      -> TOPOLOGY_MODE="euler"

A pre-existing 4-channel `gamma_targets` is detected and regenerated (use --force to
regenerate even when the channel count already matches).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zarr
from tqdm import tqdm

from src.data.gamma import compute_climatological_thresholds, compute_gamma_matrix
from src.utils import load_config, load_persistence_thresholds

N_CHANNELS = 5  # [A, P_crofton, B0, B1, chi_exact]
CHANNEL_LAYOUT = "A,P_crofton,B0,B1,chi_exact"


def _worker(args):
    start, end, zarr_path, split, thresholds, pixel_km, tb0, tb1 = args
    store = zarr.open(zarr_path, mode="r+")
    group = store[split]
    precip = group["original_precip"][start:end]
    gamma = np.zeros((precip.shape[0], N_CHANNELS, len(thresholds)), dtype=np.float32)
    for i in range(precip.shape[0]):
        # compute_gamma_matrix returns (5, Q); thresholds are PHYSICAL mm/h.
        gamma[i] = compute_gamma_matrix(
            precip[i],
            thresholds,
            pixel_km,
            tb0,
            tb1,
        )
    group["gamma_targets"][start:end] = gamma
    return f"{split} {start}:{end}"


def _ensure_dataset(group, n, n_q, chunk_size, force):
    """Create gamma_targets, regenerating if it is absent, stale-shaped, or forced."""
    if "gamma_targets" in group:
        existing = group["gamma_targets"]
        stale = (existing.ndim != 3 or existing.shape[1] != N_CHANNELS
                 or existing.shape[0] != n or existing.shape[2] != n_q)
        if force or stale:
            reason = "forced" if force and not stale else (
                f"stale shape {existing.shape} != {(n, N_CHANNELS, n_q)}")
            print(f"  regenerating gamma_targets ({reason})")
            del group["gamma_targets"]
        else:
            print("  gamma_targets already present with correct shape; "
                  "pass --force to recompute")
            return False
    group.create_dataset(
        "gamma_targets",
        shape=(n, N_CHANNELS, n_q),
        chunks=(chunk_size, N_CHANNELS, n_q),
        dtype="float32",
    )
    group["gamma_targets"].attrs["channels"] = CHANNEL_LAYOUT
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--force", action="store_true",
                        help="recompute even if gamma_targets already has the right shape")
    args = parser.parse_args()

    config = load_config(args.config)
    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    quantiles = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)
    pixel_km = config.get("PIXEL_SIZE_KM", 2.0)
    drizzle = config.get("DRIZZLE_THRESHOLD", 0.1)
    thresh_b0, thresh_b1 = load_persistence_thresholds(config)
    chunk_size = config.get("WORKER_CHUNK_SIZE", 500)
    max_workers = config.get("MAX_WORKERS", 4)

    # Physical climatological thresholds from the training CDF (same drizzle floor
    # as the rest of the pipeline). These are the excursion-set thresholds and must
    # match what the loss loads via load_physical_thresholds(config).
    phys_thresh = compute_climatological_thresholds(
        zarr_path, quantiles, drizzle_threshold=drizzle
    )
    thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"
    )
    np.save(thresh_path, phys_thresh)
    print(f"Physical thresholds saved to {thresh_path}")

    store = zarr.open(zarr_path, mode="r+")
    n_q = len(phys_thresh)

    for split in ["train", "validation", "test"]:
        if split not in store:
            continue

        group = store[split]
        n = group["original_precip"].shape[0]

        # Persistence thresholds recorded for provenance (used for B0/B1 rows).
        group.attrs["persistence_threshold_b0"] = float(thresh_b0)
        group.attrs["persistence_threshold_b1"] = float(thresh_b1)

        print(f"\n--- Computing gamma targets: {split} ({n} samples) ---")
        if not _ensure_dataset(group, n, n_q, chunk_size, args.force):
            continue

        tasks = [
            (start, min(start + chunk_size, n), zarr_path, split,
             phys_thresh, pixel_km, thresh_b0, thresh_b1)
            for start in range(0, n, chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(tasks)):
                f.result()

    print("\nGamma targets complete. Layout (axis 1): " + CHANNEL_LAYOUT)


if __name__ == "__main__":
    main()