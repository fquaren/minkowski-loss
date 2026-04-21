"""
JIT-compiled function to compute local anomaly scores.
"""

import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def compute_anomaly_score_numba(
    precip: np.ndarray, size: int = 5, epsilon: float = 0.1
) -> np.ndarray:
    """Computes a continuous quality map using JIT-compiled loops."""
    h, w = precip.shape
    out = np.zeros_like(precip, dtype=np.float32)
    pad = size // 2

    for i in range(h):
        for j in range(w):
            # Define local window boundaries with clipping
            y_min = max(0, i - pad)
            y_max = min(h, i + pad + 1)
            x_min = max(0, j - pad)
            x_max = min(w, j + pad + 1)

            # Extract window and compute moments
            window = precip[y_min:y_max, x_min:x_max]
            mu = np.mean(window)
            std = np.std(window)

            val = precip[i, j]
            if val > mu:
                out[i, j] = (val - mu) / (std + epsilon)

    return out
