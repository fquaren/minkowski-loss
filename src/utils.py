"""
Shared utilities for the Mink-DDPM project.

Centralises config validation, reproducibility, model loading, and
data (de)normalization so that no training or evaluation script
re-implements these independently.
"""

import os
import random
import logging
import sys
from contextlib import contextmanager

import yaml
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {
    "PATCH_SIZE": int,
    "DOWNSCALING_FACTOR": (int, float),
    "QUANTILE_LEVELS": list,
    "PREPROCESSED_DATA_DIR": str,
    "PIXEL_SIZE_KM": (int, float),
    "DRIZZLE_THRESHOLD": (int, float),
}


def load_config(path: str) -> dict:
    """Load and validate the project configuration YAML.

    Raises ``KeyError`` or ``TypeError`` if required keys are missing
    or have the wrong type, rather than silently falling back to
    scattered defaults.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    for key, expected in REQUIRED_CONFIG_KEYS.items():
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
        if not isinstance(config[key], expected):
            raise TypeError(
                f"Config key {key}: expected {expected}, got {type(config[key])}"
            )

    return config


def load_persistence_thresholds(config: dict) -> tuple:
    """Load or fall back to persistence thresholds.

    Returns (thresh_b0, thresh_b1).  Raises FileNotFoundError if the
    thresholds file doesn't exist and no fallback is configured.
    """
    emp_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "persistence_thresholds.yaml"
    )
    if os.path.exists(emp_path):
        with open(emp_path, "r") as f:
            emp = yaml.safe_load(f)
        return (
            float(emp["PERSISTENCE_THRESHOLD_B0"]),
            float(emp["PERSISTENCE_THRESHOLD_B1"]),
        )

    fallback = config.get("PERSISTENCE_THRESHOLD")
    if fallback is not None:
        return (float(fallback), float(fallback))

    raise FileNotFoundError(
        f"Persistence thresholds not found at {emp_path} and no "
        f"PERSISTENCE_THRESHOLD fallback in config."
    )


def load_scaler_val(config: dict) -> float:
    """Load the global log-precip max scaling value."""
    path = os.path.join(config["PREPROCESSED_DATA_DIR"], "log_precip_max_val.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return float(np.load(path).item())


def load_physical_thresholds(config: dict) -> np.ndarray:
    """Load precomputed physical precipitation thresholds."""
    path = os.path.join(config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Physical thresholds not found: {path}")
    return np.load(path)


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 3. Logging
# ---------------------------------------------------------------------------


def setup_logger(
    name: str,
    log_dir: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create an isolated logger with at most one stream + one file handler.

    Prevents the handler-leak pattern that causes process hangs during
    long Optuna runs.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def cleanup_logger(logger: logging.Logger):
    """Flush, close, and remove all handlers."""
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


@contextmanager
def managed_logger(name: str, log_dir: str = None):
    """Context manager ensuring logger cleanup even on exceptions."""
    logger = setup_logger(name, log_dir=log_dir)
    try:
        yield logger
    finally:
        cleanup_logger(logger)


# ---------------------------------------------------------------------------
# 4. Data denormalization
# ---------------------------------------------------------------------------


class DataDenormalizer:
    """Converts between normalized log-space [0, 1] and physical mm/h.

    The forward transform (applied in datasets) is:
        x_norm = clip(log1p(x_phys) / max_val, 0, 1)

    This class provides the inverse.
    """

    def __init__(self, max_val: float):
        self.max_val = float(max_val)

    def to_physical_np(self, x_norm: np.ndarray) -> np.ndarray:
        """Numpy: [0,1] normalized → physical mm/h."""
        return np.maximum(np.expm1(x_norm * self.max_val), 0.0)

    def to_physical_torch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Torch: [0,1] normalized → physical mm/h (differentiable)."""
        x_scaled = torch.clamp(x_norm * self.max_val, max=50.0)
        return F.relu(torch.expm1(x_scaled))

    def to_physical_from_diffusion(self, x_diff: torch.Tensor) -> torch.Tensor:
        """Torch: [-1,1] diffusion domain → physical mm/h."""
        x_norm = (x_diff + 1.0) / 2.0
        return self.to_physical_torch(x_norm)


# ---------------------------------------------------------------------------
# 5. Model loading
# ---------------------------------------------------------------------------


def load_emulator(checkpoint_path: str, config: dict, device: str):
    """Load a frozen emulator from a checkpoint.

    Instantiates the correct architecture from config["ARCHITECTURE"]
    and loads the saved state dict.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth checkpoint file.
    config : dict
        Project configuration (needs ARCHITECTURE, QUANTILE_LEVELS, etc.).
    device : str
        Target device ("cuda" or "cpu").

    Returns
    -------
    nn.Module
        Frozen emulator in eval mode.
    """
    from src.models.emulators import (
        BaselineCNN,
        LipschitzCNN,
        ConstrainedLipschitzCNN,
    )

    arch = config["ARCHITECTURE"]
    n_quantiles = len(config["QUANTILE_LEVELS"])
    patch_size = config["PATCH_SIZE"]
    input_shape = (1, patch_size, patch_size)
    pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)

    if arch == "Baseline":
        model = BaselineCNN(n_quantiles=n_quantiles, input_shape=input_shape)
    elif arch == "Lipschitz":
        model = LipschitzCNN(n_quantiles=n_quantiles, input_shape=input_shape)
    elif arch == "Constrained":
        model = ConstrainedLipschitzCNN(
            n_quantiles=n_quantiles,
            input_shape=input_shape,
            quantile_levels=config["QUANTILE_LEVELS"],
            pixel_area_km2=pixel_size_km**2,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Emulator checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip "module." prefix from DataParallel checkpoints
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


# ---------------------------------------------------------------------------
# 6. Signed log transform (for Euler characteristic)
# ---------------------------------------------------------------------------


def signed_log1p(x):
    """sign(x) * log1p(|x|), applicable to numpy arrays or torch tensors.

    Used for the Euler characteristic which can be negative.
    """
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x):
    """Inverse of signed_log1p: sign(x) * expm1(|x|)."""
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.expm1(torch.abs(x))
    return np.sign(x) * np.expm1(np.abs(x))
