"""
Offline mixup augmentation for emulator training data.

Generates synthetic precipitation fields as convex combinations of
ground truth and noise-injected interpolated fields:

    x_mix = λ * x_real + (1 - λ) * (x_interp + ε)

where λ ~ Beta(α, α) and ε ~ N(0, σ²I).

The corresponding gamma targets are recomputed exactly on the mixed
fields (not interpolated from the component gammas, since Minkowski
functionals are nonlinear). Gamma now has 5 channels
[A, P_crofton, B0, B1, chi_exact]; see src/data/gamma.compute_gamma_matrix.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zarr
from tqdm import tqdm

from src.data.gamma import compute_gamma_matrix
from src.utils import load_config, load_persistence_thresholds, load_physical_thresholds

N_CHANNELS = 5  # [A, P_crofton, B0, B1, chi_exact]
CHANNEL_LAYOUT = "A,P_crofton,B0,B1,chi_exact"


def apply_mixup_numpy(
    real_chunk: np.ndarray,
    interp_chunk: np.ndarray,
    alpha: float = 0.2,
    noise_std: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Apply mixup augmentation to a batch of fields.

    Parameters
    ----------
    real_chunk : np.ndarray, shape (N, H, W)
    interp_chunk : np.ndarray, shape (N, H, W)
    alpha : float
        Beta distribution parameter.
    noise_std : float
        Gaussian noise standard deviation.
    rng : np.random.Generator

    Returns
    -------
    mixed : np.ndarray, shape (N, H, W)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = real_chunk.shape[0]
    noise = noise_std * rng.standard_normal(interp_chunk.shape).astype(np.float32)
    interp_noisy = np.clip(interp_chunk + noise, 0.0, None)

    lam = rng.beta(alpha, alpha, size=(N, 1, 1)).astype(np.float32)
    return lam * real_chunk + (1 - lam) * interp_noisy


def worker_mixup_chunk(args):
    """Process a chunk: mixup + gamma recomputation."""
    (
        start_idx,
        end_idx,
        zarr_path,
        physical_thresholds,
        config,
        thresh_b0,
        thresh_b1,
        global_seed,
    ) = args

    store = zarr.open(zarr_path, mode="r+")
    group = store["train"]

    real = group["original_precip"][start_idx:end_idx]
    interp = group["interpolated_precip"][start_idx:end_idx]

    N = real.shape[0]
    pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)
    alpha = config.get("MIXUP_ALPHA", 0.2)
    noise_std = config.get("NOISE_STD", 0.05)

    # Deterministic per-chunk seed derived from global seed
    rng = np.random.default_rng(seed=global_seed + start_idx)
    mixed = apply_mixup_numpy(real, interp, alpha, noise_std, rng)

    # mixed is a convex combination of physical precip fields -> still mm/h,
    # so compute_gamma_matrix (which thresholds in physical space) applies directly.
    gamma_chunk = np.zeros((N, N_CHANNELS, len(physical_thresholds)), dtype=np.float32)
    for i in range(N):
        gamma_chunk[i] = compute_gamma_matrix(
            mixed[i], physical_thresholds, pixel_size_km, thresh_b0, thresh_b1
        )

    group["mixup_precip"][start_idx:end_idx] = mixed
    group["mixup_gamma_targets"][start_idx:end_idx] = gamma_chunk

    return f"Processed {start_idx}:{end_idx}"


def run_mixup_pipeline(config_path: str, seed: int = 42):
    """Main entry point for offline mixup augmentation."""
    config = load_config(config_path)
    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    physical_thresholds = load_physical_thresholds(config)
    thresh_b0, thresh_b1 = load_persistence_thresholds(config)

    store = zarr.open(zarr_path, mode="r+")
    group = store["train"]
    num_samples = group["original_precip"].shape[0]
    _, H, W = group["original_precip"].shape
    n_q = len(physical_thresholds)
    chunk_size = config.get("WORKER_CHUNK_SIZE", 500)

    # Create or overwrite datasets (recreate to guarantee the 5-channel layout,
    # in case a stale 4-channel mixup_gamma_targets exists from a previous run)
    for name, shape, chunks in [
        ("mixup_precip", (num_samples, H, W), (chunk_size, H, W)),
        ("mixup_gamma_targets", (num_samples, N_CHANNELS, n_q), (chunk_size, N_CHANNELS, n_q)),
    ]:
        if name in group:
            del group[name]
        group.create_dataset(name, shape=shape, chunks=chunks, dtype="float32")
    group["mixup_gamma_targets"].attrs["channels"] = CHANNEL_LAYOUT

    tasks = []
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        tasks.append(
            (
                start,
                end,
                zarr_path,
                physical_thresholds,
                config,
                thresh_b0,
                thresh_b1,
                seed,
            )
        )

    max_workers = config.get("MAX_WORKERS", max(1, os.cpu_count() // 4))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_mixup_chunk, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(tasks), desc="Mixup"):
            f.result()

    print("Mixup augmentation complete.")