"""
PyTorch datasets for the Mink-DDPM project.

All datasets read from the preprocessed Zarr store and return
log-transformed data ready for training.

ZarrMixupDataset:       Emulator training (precip → gamma)
DeterministicSRDataset: UNet super-resolution (low-res+DEM → high-res)
DiffusionSRDataset:     DDPM super-resolution (low-res+DEM → high-res)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import zarr

from src.utils import signed_log1p


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _gamma_to_log(gamma_phys: np.ndarray, topology_mode: str) -> np.ndarray:
    """Convert 4-channel physical gamma to 3-channel log-transformed target.

    Parameters
    ----------
    gamma_phys : np.ndarray, shape (..., 4, Q)
        Raw gamma matrix [A, P, B0, B1] in physical units.
    topology_mode : str
        "euler" → chi = B0 - B1 (signed, use signed_log1p)
        "b0"   → B0 only (non-negative, use log1p)

    Returns
    -------
    np.ndarray, shape (..., 3, Q)
        Log-transformed gamma [log1p(A), log1p(P), f(topology)].
    """
    log_A = np.log1p(gamma_phys[..., 0, :])
    log_P = np.log1p(gamma_phys[..., 1, :])

    if topology_mode == "euler":
        chi = gamma_phys[..., 2, :] - gamma_phys[..., 3, :]
        log_T = signed_log1p(chi)
    elif topology_mode == "b0":
        log_T = np.log1p(gamma_phys[..., 2, :])
    else:
        raise ValueError(f"Unknown topology_mode: {topology_mode!r}")

    return np.stack([log_A, log_P, log_T], axis=-2)


# ---------------------------------------------------------------
# 1. Emulator dataset
# ---------------------------------------------------------------


class ZarrMixupDataset(Dataset):
    """Dataset for emulator training, supporting original + mixup data.

    Returns
    -------
    input_tensor : Tensor [1, H, W]
        Log-normalized precipitation in [0, 1].
    log_target_gamma : Tensor [3, Q]
        Log-transformed gamma targets.

    Notes
    -----
    Augmentations (flip, 90° rotation) are invariant for Minkowski
    functionals: area, perimeter, and topology are all preserved
    under rigid motions of the plane.
    """

    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        scaler_val: float = 1.0,
        augment: bool = True,
        include_original: bool = True,
        include_mixup: bool = True,
        subset_fraction: float = 1.0,
        topology_mode: str = "b0",
        load_in_ram: bool = False,
    ):
        self.scaler_val = float(scaler_val)
        self.augment = augment
        self.topology_mode = topology_mode
        self.load_in_ram = load_in_ram

        store = zarr.open(zarr_path, mode="r")
        if split not in store:
            raise KeyError(f"Split '{split}' not found in zarr store.")
        self.group = store[split]

        # Build list of (data_key, target_key) pairs
        self.data_keys = []
        self.target_keys = []
        self.lengths = []

        if include_original and "original_precip" in self.group:
            self.data_keys.append("original_precip")
            self.target_keys.append("gamma_targets")
            self.lengths.append(self.group["original_precip"].shape[0])

        if include_mixup and "mixup_precip" in self.group:
            self.data_keys.append("mixup_precip")
            self.target_keys.append("mixup_gamma_targets")
            self.lengths.append(self.group["mixup_precip"].shape[0])

        if not self.data_keys:
            raise ValueError("No data arrays found for the given include flags.")

        self.cumulative_sizes = np.cumsum(self.lengths)
        self.indices_map = np.arange(self.cumulative_sizes[-1])

        # Subset
        if 0.0 < subset_fraction < 1.0:
            rng = np.random.default_rng(seed=42)
            n = max(1, int(len(self.indices_map) * subset_fraction))
            self.indices_map = np.sort(
                rng.choice(self.indices_map, size=n, replace=False)
            )

        # RAM loading
        if self.load_in_ram:
            self._load_into_ram()

        if self.augment:
            self.transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomChoice(
                        [v2.RandomRotation([d, d]) for d in [0, 90, 180, 270]]
                    ),
                ]
            )

    def _load_into_ram(self):
        self.data_arrays = []
        self.target_arrays = []
        for i in range(len(self.data_keys)):
            start = 0 if i == 0 else self.cumulative_sizes[i - 1]
            end = self.cumulative_sizes[i]
            mask = (self.indices_map >= start) & (self.indices_map < end)
            local_idx = self.indices_map[mask] - start
            self.data_arrays.append(self.group[self.data_keys[i]].oindex[local_idx])
            self.target_arrays.append(self.group[self.target_keys[i]].oindex[local_idx])
        self.indices_map = np.arange(len(self.indices_map))
        self.lengths = [a.shape[0] for a in self.data_arrays]
        self.cumulative_sizes = np.cumsum(self.lengths)

    def _resolve_index(self, idx):
        real_idx = self.indices_map[idx]
        src = int(np.searchsorted(self.cumulative_sizes, real_idx, side="right"))
        local = real_idx if src == 0 else real_idx - self.cumulative_sizes[src - 1]
        return src, int(local)

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, idx):
        src, local = self._resolve_index(idx)

        if self.load_in_ram:
            patch = self.data_arrays[src][local].copy()
            gamma_phys = self.target_arrays[src][local].copy()
        else:
            patch = self.group[self.data_keys[src]][local].copy()
            gamma_phys = self.group[self.target_keys[src]][local].copy()

        # Normalize: log1p → scale to [0, 1]
        patch_norm = np.clip(np.log1p(patch) / self.scaler_val, 0.0, 1.0)
        input_tensor = torch.from_numpy(patch_norm).float().unsqueeze(0)

        if self.augment:
            input_tensor = self.transform(input_tensor)

        # Convert 4ch gamma to 3ch log target
        log_gamma = _gamma_to_log(gamma_phys, self.topology_mode)
        log_gamma_tensor = torch.from_numpy(log_gamma).float()

        return input_tensor, log_gamma_tensor


# ---------------------------------------------------------------
# 2. Deterministic SR dataset
# ---------------------------------------------------------------


class DeterministicSRDataset(Dataset):
    """Super-resolution dataset for UNet training.

    Returns
    -------
    input_stack : Tensor [3, H, W]
        Channel 0: log-normalized interpolated precip [0, 1].
        Channel 1: z-scored DEM.
    target_tensor : Tensor [1, H, W]
        Log-normalized target precip [0, 1].
    target_gamma : Tensor [3, Q]
        Log-transformed gamma targets.
    """

    def __init__(
        self,
        preprocessed_data_dir: str,
        metadata_file: str,
        dem_stats: tuple,
        scaler_max_val: float,
        split: str = "train",
        data_percentage: float = 100.0,
        wet_dry_ratio: float = 1.0,
        topology_mode: str = "b0",
        load_in_ram: bool = False,
        dem_clip_sigma: float = 3.0,
    ):
        self.dem_mean, self.dem_std = dem_stats
        self.scaler_max_val = float(scaler_max_val)
        self.split = split
        self.is_train = split == "train"
        self.topology_mode = topology_mode
        self.load_in_ram = load_in_ram
        self.dem_clip_sigma = dem_clip_sigma

        self.zarr_path = os.path.join(
            preprocessed_data_dir, "preprocessed_dataset.zarr"
        )

        # Validate split exists in Zarr store at init time (not lazily in workers)
        _store = zarr.open(self.zarr_path, mode="r")
        if split not in _store:
            available = list(_store.keys())
            # Handle common val/validation mismatch
            alt = {"validation": "val", "val": "validation"}
            if split in alt and alt[split] in _store:
                self.split = alt[split]
                print(
                    f"Warning: split '{split}' not found, using '{self.split}' "
                    f"(available: {available})"
                )
            else:
                raise KeyError(
                    f"Split '{split}' not found in {self.zarr_path}. "
                    f"Available groups: {available}"
                )
        else:
            self.split = split

        self.metadata = []
        is_wet = []
        with open(metadata_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    ts, y, x, p_max = parts
                    self.metadata.append((ts, int(y), int(x), float(p_max)))
                    is_wet.append(float(p_max) > 0.1)

        self.valid_indices = np.arange(len(self.metadata))
        is_wet = np.array(is_wet)

        if 0.0 < data_percentage < 100.0:
            n = max(1, int(len(self.valid_indices) * data_percentage / 100.0))
            rng = np.random.default_rng(seed=42)
            self.valid_indices = rng.choice(self.valid_indices, size=n, replace=False)
            is_wet = is_wet[self.valid_indices]

        self.sample_weights = None
        if self.is_train and wet_dry_ratio is not None:
            n_wet = np.sum(is_wet)
            n_dry = len(is_wet) - n_wet
            w_wet = 1.0 / n_wet if n_wet > 0 else 0.0
            w_dry = (1.0 / n_dry) * wet_dry_ratio if n_dry > 0 else 0.0
            self.sample_weights = np.where(is_wet, w_wet, w_dry).astype(np.float64)

        if self.load_in_ram:
            store = zarr.open(self.zarr_path, mode="r")
            g = store[split]
            self.ram_original = g["original_precip"][:]
            self.ram_interp = g["interpolated_precip"][:]
            self.ram_gamma = g["gamma_targets"][:]
            self.ram_dem = g["dem"][:]
        else:
            self._store = None
            self._group = None

        self.geom_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )

    def _init_zarr(self):
        if self._store is None:
            self._store = zarr.open(self.zarr_path, mode="r")
            self._group = self._store[self.split]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        if self.load_in_ram:
            target_phys = self.ram_original[real_idx]
            interp_phys = self.ram_interp[real_idx]
            gamma_phys = self.ram_gamma[real_idx]
            dem_patch = self.ram_dem[real_idx]
        else:
            self._init_zarr()
            target_phys = self._group["original_precip"][real_idx]
            interp_phys = self._group["interpolated_precip"][real_idx]
            gamma_phys = self._group["gamma_targets"][real_idx]
            dem_patch = self._group["dem"][real_idx]

        target_norm = np.clip(np.log1p(target_phys) / self.scaler_max_val, 0.0, 1.0)
        interp_norm = np.clip(np.log1p(interp_phys) / self.scaler_max_val, 0.0, 1.0)

        target_t = torch.from_numpy(target_norm).float().unsqueeze(0)
        interp_t = torch.from_numpy(interp_norm).float().unsqueeze(0)

        dem_z = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        if self.dem_clip_sigma > 0:
            dem_z = np.clip(dem_z, -self.dem_clip_sigma, self.dem_clip_sigma)
        dem_t = torch.from_numpy(dem_z).float().unsqueeze(0)

        input_stack = torch.cat([interp_t, dem_t], dim=0)

        if self.is_train:
            input_stack, target_t = self.geom_transform(input_stack, target_t)

        log_gamma = _gamma_to_log(gamma_phys, self.topology_mode)
        gamma_t = torch.from_numpy(log_gamma).float()

        return input_stack, target_t, gamma_t


# ---------------------------------------------------------------
# 3. Diffusion SR dataset
# ---------------------------------------------------------------


class DiffusionSRDataset(Dataset):
    """Super-resolution dataset for DDPM training.

    Returns
    -------
    input_stack : Tensor [3, H, W]
        Channel 0: precip in [-1, 1]
        Channel 1: z-scored DEM.
    target_tensor : Tensor [1, H, W]
        Target precip in [-1, 1].
    target_gamma : Tensor [3, Q]
        Log-transformed gamma targets.
    """

    def __init__(
        self,
        preprocessed_data_dir: str,
        metadata_file: str,
        dem_stats: tuple,
        scaler_max_val: float,
        split: str = "train",
        data_percentage: float = 100.0,
        topology_mode: str = "b0",
        load_in_ram: bool = False,
        dem_clip_sigma: float = 3.0,
    ):
        self.dem_mean, self.dem_std = dem_stats
        self.scaler_max_val = float(scaler_max_val)
        self.split = split
        self.is_train = split == "train"
        self.topology_mode = topology_mode
        self.load_in_ram = load_in_ram
        self.dem_clip_sigma = dem_clip_sigma

        self.zarr_path = os.path.join(
            preprocessed_data_dir, "preprocessed_dataset.zarr"
        )

        # Validate split exists in Zarr store at init time (not lazily in workers)
        _store = zarr.open(self.zarr_path, mode="r")
        if split not in _store:
            available = list(_store.keys())
            # Handle common val/validation mismatch
            alt = {"validation": "val", "val": "validation"}
            if split in alt and alt[split] in _store:
                self.split = alt[split]
                print(
                    f"Warning: split '{split}' not found, using '{self.split}' "
                    f"(available: {available})"
                )
            else:
                raise KeyError(
                    f"Split '{split}' not found in {self.zarr_path}. "
                    f"Available groups: {available}"
                )
        else:
            self.split = split

        self.metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    ts, y, x, p_max = parts
                    self.metadata.append((ts, int(y), int(x), float(p_max)))

        self.valid_indices = np.arange(len(self.metadata))

        if 0.0 < data_percentage < 100.0:
            n = max(1, int(len(self.valid_indices) * data_percentage / 100.0))
            rng = np.random.default_rng(seed=42)
            self.valid_indices = np.sort(
                rng.choice(self.valid_indices, size=n, replace=False)
            )

        if self.load_in_ram:
            store = zarr.open(self.zarr_path, mode="r")
            g = store[self.split]
            self.ram_original = g["original_precip"].oindex[self.valid_indices]
            self.ram_interp = g["interpolated_precip"].oindex[self.valid_indices]
            self.ram_gamma = g["gamma_targets"].oindex[self.valid_indices]
            self.ram_dem = g["dem"].oindex[self.valid_indices]
            self.valid_indices = np.arange(len(self.valid_indices))
        else:
            self._store = None
            self._group = None

        self.geom_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        if self.load_in_ram:
            target_phys = self.ram_original[real_idx]
            interp_phys = self.ram_interp[real_idx]
            gamma_phys = self.ram_gamma[real_idx]
            dem_patch = self.ram_dem[real_idx]
        else:
            if self._store is None:
                self._store = zarr.open(self.zarr_path, mode="r")
                self._group = self._store[self.split]
            target_phys = self._group["original_precip"][real_idx]
            interp_phys = self._group["interpolated_precip"][real_idx]
            gamma_phys = self._group["gamma_targets"][real_idx]
            dem_patch = self._group["dem"][real_idx]

        target_norm = np.clip(np.log1p(target_phys) / self.scaler_max_val, 0.0, 1.0)
        interp_norm = np.clip(np.log1p(interp_phys) / self.scaler_max_val, 0.0, 1.0)

        target_t = torch.from_numpy(target_norm).float().unsqueeze(0) * 2.0 - 1.0
        interp_t = torch.from_numpy(interp_norm).float().unsqueeze(0) * 2.0 - 1.0

        dem_z = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        if self.dem_clip_sigma > 0:
            dem_z = np.clip(dem_z, -self.dem_clip_sigma, self.dem_clip_sigma)
        dem_t = torch.from_numpy(dem_z).float().unsqueeze(0)

        input_stack = torch.cat([interp_t, dem_t], dim=0)

        if self.is_train:
            input_stack, target_t = self.geom_transform(input_stack, target_t)

        log_gamma = _gamma_to_log(gamma_phys, self.topology_mode)
        gamma_t = torch.from_numpy(log_gamma).float()

        return input_stack, target_t, gamma_t
