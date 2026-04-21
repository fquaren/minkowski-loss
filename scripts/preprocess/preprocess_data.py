#!/usr/bin/env python
"""Create the preprocessed Zarr store from raw OPERA data and metadata.

This orchestration script distributes tasks to a process pool while
enforcing strict threading constraints for safety across C-extensions.
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numcodecs
import yaml
import zarr
from tqdm import tqdm

from src.data.preprocessing import (
    compute_dem_stats,
    compute_global_scaler,
    process_batch,
)


def load_config(config_path: str) -> dict:
    """Load configuration variables from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess OPERA patches into Zarr store."
    )
    parser.add_argument("config", type=str, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Prevent thread deadlocks during multiprocessing fork/spawn
    numcodecs.blosc.use_threads = False

    map_path = os.path.join(config["METADATA_DIR"], "timestamp_map.json")

    metadata_paths = {
        "train": os.path.join(config["METADATA_DIR"], "train_patches_metadata.txt"),
        "validation": os.path.join(config["METADATA_DIR"], "val_patches_metadata.txt"),
        "test": os.path.join(config["METADATA_DIR"], "test_patches_metadata.txt"),
    }

    output_zarr = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    print(f"Creating Zarr store at: {output_zarr}")
    root = zarr.open(output_zarr, mode="w")

    patch_size = config["PATCH_SIZE"]
    coarse_size = int(patch_size / config["DOWNSCALING_FACTOR"])
    batch_size = config.get("PREPROCESS_BATCH_SIZE", 1000)
    dem_path = config["STATIC_DEM_PATH"]

    all_payloads = {}

    for group_name, path in metadata_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {group_name}.")
            continue

        with open(path, "r") as f:
            lines = f.readlines()

        n = len(lines)
        if n == 0:
            continue
        print(f"Found {n} patches for '{group_name}'.")

        group = root.create_group(group_name)
        compressor = numcodecs.Blosc(
            cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
        )

        # Initialize lock-free chunking structure (one patch per disk file)
        for name, shape, chunks in tqdm(
            [
                (
                    "original_precip",
                    (n, patch_size, patch_size),
                    (1, patch_size, patch_size),
                ),
                (
                    "interpolated_precip",
                    (n, patch_size, patch_size),
                    (1, patch_size, patch_size),
                ),
                (
                    "coarse_precip",
                    (n, coarse_size, coarse_size),
                    (1, coarse_size, coarse_size),
                ),
                ("dem", (n, patch_size, patch_size), (1, patch_size, patch_size)),
                (
                    "quality_map",
                    (n, patch_size, patch_size),
                    (1, patch_size, patch_size),
                ),
            ],
            desc="Initializing Zarr datasets",
        ):
            group.create_dataset(
                name, shape=shape, chunks=chunks, dtype="float32", compressor=compressor
            )

        # Transmit map_path rather than the loaded JSON mapping
        static_args = (output_zarr, dem_path, group_name, config, map_path)

        tasks = []
        for i, line in enumerate(lines):
            timestamp_str, y_str, x_str, _ = line.strip().split(",")
            patch_meta = (timestamp_str, int(y_str), int(x_str))
            tasks.append((i, patch_meta))

        payloads = []
        for j in range(0, len(tasks), batch_size):
            batch = tasks[j : j + batch_size]
            payloads.append({"static": static_args, "tasks": batch})

        all_payloads[group_name] = payloads

    max_workers = config.get("MAX_WORKERS", 4)

    for group_name, payloads in all_payloads.items():
        total = sum(len(p["tasks"]) for p in payloads)
        print(f"\n--- Processing '{group_name}' ({total} patches) ---")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, p) for p in payloads]
            for future in tqdm(
                as_completed(futures), total=len(payloads), desc=group_name
            ):
                results = future.result()
                if results:
                    for msg in results:
                        print(msg)

    print("\nZarr store creation complete.")

    # Compute global scaler and DEM stats
    compute_global_scaler(output_zarr, config["PREPROCESSED_DATA_DIR"])

    dem_stats_path = config.get(
        "DEM_STATS", os.path.join(config["PREPROCESSED_DATA_DIR"], "dem_stats.json")
    )
    compute_dem_stats(output_zarr, dem_stats_path)

    print("\nPreprocessing pipeline finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
