#!/usr/bin/env python
"""Generate patch metadata and timestamp map in parallel."""

import os
import glob
import argparse
import shutil
import json
import numpy as np
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils import load_config
from src.data.metadata import find_valid_patches_numba


def scan_zarr_folder_for_patches(folder_path: str, precip_var_name: str, patch_size: int):
    local_coords_lines = []
    local_timestamp_map = {}
    try:
        with xr.open_zarr(folder_path, consolidated=True) as ds:
            data = ds[precip_var_name].load().values
            times = ds.time.values

            for t_idx in range(len(times)):
                frame = data[t_idx]
                timestamp_dt_numpy = times[t_idx]

                timestamp_str = (
                    np.datetime_as_string(timestamp_dt_numpy, unit="s")
                    .replace("-", "")
                    .replace("T", "")
                    .replace(":", "")
                )
                local_timestamp_map[timestamp_str] = (folder_path, t_idx)

                y_coords, x_coords, max_vals = find_valid_patches_numba(
                    frame, patch_size, patch_size
                )

                for i in range(len(y_coords)):
                    local_coords_lines.append(
                        f"{timestamp_str},{y_coords[i]},{x_coords[i]},{max_vals[i]:.4f}\n"
                    )
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
    return local_coords_lines, local_timestamp_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    config = load_config(args.config)

    raw_dir = config["RAW_OPERA_DATA_DIR"]
    meta_dir = config["METADATA_DIR"]
    patch_size = config["PATCH_SIZE"]
    var_name = config["PRECIP_VAR_NAME"]
    max_workers = config.get("MAX_WORKERS", 4)

    temp_meta_dir = os.path.join(meta_dir, "temp_metadata")
    if os.path.exists(temp_meta_dir):
        shutil.rmtree(temp_meta_dir)
    os.makedirs(temp_meta_dir)

    zarr_folders = sorted(glob.glob(os.path.join(raw_dir, "[0-9]" * 8)))
    if not zarr_folders:
        raise FileNotFoundError(f"No Zarr folders found in {raw_dir}")

    timestamp_map = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                scan_zarr_folder_for_patches, folder, var_name, patch_size
            ): os.path.join(temp_meta_dir, f"{os.path.basename(folder)}.txt")
            for folder in zarr_folders
        }
        for future in tqdm(as_completed(future_to_path), total=len(zarr_folders), desc="Scanning data"):
            output_path = future_to_path[future]
            result_lines, local_map = future.result()
            if result_lines:
                with open(output_path, "w") as f:
                    f.writelines(result_lines)
            if local_map:
                timestamp_map.update(local_map)

    map_path = os.path.join(meta_dir, "timestamp_map.json")
    with open(map_path, "w") as f:
        json.dump(timestamp_map, f)


if __name__ == "__main__":
    main()