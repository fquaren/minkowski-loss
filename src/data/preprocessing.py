"""
Data preprocessing: physical filtering, coarsening, and Zarr store creation.

All operations run on CPU to avoid CUDA context initialization in
multiprocessing workers. IPC overhead is minimized by localizing 
metadata loading to the worker process memory.
"""

import os
import json
import numpy as np
import xarray as xr
import zarr
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Module-level cache for worker processes
_WORKER_TIMESTAMP_MAP = None


def filter_precip_bounds(arr: np.ndarray, min_thresh: float, max_thresh: float) -> np.ndarray:
    """Zero out precipitation outside [min_thresh, max_thresh].

    Parameters
    ----------
    arr : np.ndarray
        Precipitation field (modified in-place).
    min_thresh : float
        Drizzle threshold (mm/h).
    max_thresh : float
        Declutter threshold (mm/h).
    """
    mask = (arr < min_thresh) | (arr > max_thresh)
    arr[mask] = 0.0
    return arr


def coarsen_and_interpolate(arr: np.ndarray, factor: float):
    """Coarsen via adaptive average pooling, upsample via nearest-neighbor.

    NOTE: When patch_size is not evenly divisible by factor,
    adaptive_avg_pool2d uses overlapping windows with non-uniform
    weights. This is the case for the OPERA setup (128 / 12.5 = 10.24,
    truncated to 10). The approximation is acceptable for the
    publication's downscaling task.

    Parameters
    ----------
    arr : np.ndarray, shape (H, W)
        Single precipitation field.
    factor : float
        Downscaling factor.

    Returns
    -------
    coarse : np.ndarray, shape (H', W')
    interpolated : np.ndarray, shape (H, W)
    """
    H, W = arr.shape
    H_new, W_new = int(H / factor), int(W / factor)

    # Use torch on CPU only
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    coarse_t = F.adaptive_avg_pool2d(t, output_size=(H_new, W_new))
    interp_t = F.interpolate(coarse_t, size=(H, W), mode="nearest")

    return coarse_t.squeeze().numpy(), interp_t.squeeze().numpy()


def process_batch(batch_payload: dict) -> list:
    """Worker function for parallel Zarr writing.

    Reads precipitation and DEM patches, applies filtering and
    coarsening, and writes results to the shared Zarr store.
    """
    global _WORKER_TIMESTAMP_MAP

    static_args = batch_payload["static"]
    tasks = batch_payload["tasks"]
    
    # Receive map_path instead of the loaded dictionary
    output_zarr_path, dem_path, group_name, config, map_path = static_args
    
    patch_size = config["PATCH_SIZE"]
    drizzle = config.get("DRIZZLE_THRESHOLD", 0.1)
    declutter = config.get("DECLUTTER_THRESHOLD", 150.0)
    factor = config["DOWNSCALING_FACTOR"]

    # Load into memory strictly once per worker process
    if _WORKER_TIMESTAMP_MAP is None:
        with open(map_path, "r") as f:
            _WORKER_TIMESTAMP_MAP = json.load(f)

    results = []
    target_zarr = zarr.open(output_zarr_path, mode="r+")

    try:
        with xr.open_dataset(dem_path, engine="rasterio") as ds:
            dem_memory = (
                ds["band_data"].isel(band=0).drop_vars("band", errors="ignore").load()
            )
    except Exception as e:
        return [f"FATAL: Failed to load DEM: {e}"]

    source_cache = {}
    valid_indices = []
    precip_list = []
    dem_list = []

    for idx, patch_meta in tasks:
        timestamp_str, y_start, x_start = patch_meta
        try:
            # Query the localized cache
            source_folder, time_idx = _WORKER_TIMESTAMP_MAP[timestamp_str]

            if source_folder not in source_cache:
                try:
                    ds = xr.open_zarr(source_folder, consolidated=True)
                    if not ds.dims:
                        ds = xr.open_zarr(source_folder, consolidated=False)
                except Exception:
                    ds = xr.open_zarr(source_folder, consolidated=False)
                source_cache[source_folder] = ds

            ds = source_cache[source_folder]
            precip = (
                ds[config["PRECIP_VAR_NAME"]]
                .isel(time=time_idx,
                      y=slice(y_start, y_start + patch_size),
                      x=slice(x_start, x_start + patch_size))
                .load().values
            )

            if precip.shape != (patch_size, patch_size):
                results.append(f"Skipped {idx}: boundary truncation.")
                continue

            dem_patch = dem_memory.isel(
                y=slice(y_start, y_start + patch_size),
                x=slice(x_start, x_start + patch_size),
            ).values

            if dem_patch.shape != (patch_size, patch_size):
                results.append(f"Skipped {idx}: DEM truncation.")
                continue

            precip = np.nan_to_num(precip, nan=0.0)
            precip = filter_precip_bounds(precip, drizzle, declutter)

            valid_indices.append(idx)
            precip_list.append(precip)
            dem_list.append(dem_patch)

        except Exception as e:
            results.append(f"Error at {idx}: {e}")

    for i, (idx, precip) in enumerate(zip(valid_indices, precip_list)):
        coarse, interpolated = coarsen_and_interpolate(precip, factor)
        target_zarr[f"{group_name}/original_precip"][idx] = precip
        target_zarr[f"{group_name}/interpolated_precip"][idx] = interpolated
        target_zarr[f"{group_name}/coarse_precip"][idx] = coarse
        target_zarr[f"{group_name}/dem"][idx] = dem_list[i]

    return results if results else []


def compute_global_scaler(zarr_path: str, output_dir: str):
    """Compute global max of log1p(precip) from training data."""
    print("\n--- Computing global training scaler ---")
    store = zarr.open(zarr_path, mode="r")
    data = store["train/original_precip"]
    chunk_size = 5000
    global_max = 0.0

    for i in tqdm(range(0, data.shape[0], chunk_size), desc="Scanning"):
        chunk_log = np.log1p(data[i : i + chunk_size])
        chunk_max = float(np.max(chunk_log))
        global_max = max(global_max, chunk_max)

    path = os.path.join(output_dir, "log_precip_max_val.npy")
    np.save(path, np.array([global_max]))
    print(f"Global log1p max: {global_max:.4f} → {path}")


def compute_dem_stats(zarr_path: str, output_path: str):
    """Compute mean and std of DEM from training data."""
    print("\n--- Computing DEM statistics ---")
    store = zarr.open(zarr_path, mode="r")

    if "train/dem" not in store:
        print("Warning: train/dem not found.")
        return

    data = store["train/dem"]
    chunk_size = 5000
    count = 0
    sum_val = np.float64(0.0)
    sum_sq = np.float64(0.0)

    for i in tqdm(range(0, data.shape[0], chunk_size), desc="DEM stats"):
        chunk = data[i : i + chunk_size]
        valid = chunk[~np.isnan(chunk)]
        count += valid.size
        sum_val += np.sum(valid, dtype=np.float64)
        sum_sq += np.sum(valid ** 2, dtype=np.float64)

    if count == 0:
        print("Error: No valid DEM data.")
        return

    mean = sum_val / count
    std = np.sqrt(sum_sq / count - mean ** 2)

    stats = {"dem_mean": float(mean), "dem_std": float(std)}
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"DEM stats: mean={mean:.2f}, std={std:.2f} → {output_path}")