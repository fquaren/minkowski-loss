"""
Structural metrics for SR training-time validation.

All functions operate on physical-space precipitation fields [B, 1, H, W]
in mm/h. The validation loop accumulates gamma tensors and calls
``PerFeatureR2Accumulator.compute()`` once per epoch.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch


# ---------------------------------------------------------------
# Radial-averaged power spectrum (RAPS)
# ---------------------------------------------------------------


def compute_radial_power_spectrum(
    field: torch.Tensor,
    n_bins: int | None = None,
) -> torch.Tensor:
    """Radial-averaged power spectrum of a 2D field.

    Parameters
    ----------
    field : Tensor [B, 1, H, W]
        Physical-space field (mm/h or any units).
    n_bins : int or None
        Number of radial bins. Defaults to ``min(H, W) // 2``.

    Returns
    -------
    Tensor [B, n_bins]
        Mean spectral power per radial wavenumber bin.
    """
    B, _, H, W = field.shape
    device = field.device

    if n_bins is None:
        n_bins = min(H, W) // 2

    # 2D real FFT, orthonormal scaling
    fft = torch.fft.rfft2(field[:, 0], norm="ortho")  # [B, H, W//2+1]
    power = fft.abs().pow(2)  # [B, H, W//2+1]

    # Radial wavenumber grid (cycles per pixel)
    ky = torch.fft.fftfreq(H, device=device)
    kx = torch.fft.rfftfreq(W, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    k_mag = torch.sqrt(KX.pow(2) + KY.pow(2))  # [H, W//2+1]

    k_max = k_mag.max().clamp(min=1e-6)
    bin_idx = (
        (k_mag / k_max).mul(n_bins - 1).long().clamp(0, n_bins - 1)
    )  # [H, W//2+1]

    # Pixel counts per bin (shared across batch)
    counts = torch.zeros(n_bins, device=device, dtype=field.dtype)
    counts.scatter_add_(
        0,
        bin_idx.flatten(),
        torch.ones_like(bin_idx.flatten(), dtype=field.dtype),
    )
    counts = counts.clamp(min=1.0)

    # Scatter sum power into bins, per sample
    bin_idx_b = bin_idx.flatten().unsqueeze(0).expand(B, -1)  # view, no copy
    power_flat = power.reshape(B, -1)
    raps = torch.zeros(B, n_bins, device=device, dtype=field.dtype)
    raps.scatter_add_(1, bin_idx_b, power_flat)
    raps.div_(counts.unsqueeze(0))

    return raps


def compute_raps_error(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    n_bins: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE between log-RAPS curves of prediction and target.

    Computed in log-power space to handle the wide dynamic range
    of precipitation spectra (small wavenumbers dominate raw power).

    Returns scalar Tensor.
    """
    raps_p = compute_radial_power_spectrum(pred_phys, n_bins)
    raps_t = compute_radial_power_spectrum(target_phys, n_bins)
    return (raps_p.add(eps).log() - raps_t.add(eps).log()).pow(2).mean()


# ---------------------------------------------------------------
# Isoperimetric violation rate
# ---------------------------------------------------------------


def compute_isoperimetric_violation_rate(
    pred_phys: torch.Tensor,
    thresholds: List[float],
    pixel_size_km: float = 2.0,
) -> tuple[float, float]:
    """Fraction of (sample, threshold) pairs where P² < 4πA.

    Operates directly on the physical field using hard thresholding +
    edge counting. Discrete pixelated boundaries always satisfy the
    inequality for connected sets, so violations indicate either highly
    fragmented predictions or numerical inconsistencies.

    Parameters
    ----------
    pred_phys : Tensor [B, 1, H, W] in mm/h
    thresholds : list of float, intensity thresholds in mm/h
    pixel_size_km : float, spatial resolution

    Returns
    -------
    (mean_rate, max_rate) : float pair in [0, 1]
        ``mean_rate`` averages over thresholds; ``max_rate`` returns
        the worst-performing threshold.
    """
    pixel_area = pixel_size_km ** 2
    rates: List[float] = []

    with torch.no_grad():
        for thresh in thresholds:
            mask = (pred_phys >= thresh).float()  # [B, 1, H, W]

            # Area
            area = mask.sum(dim=(-3, -2, -1)) * pixel_area  # [B]

            # Perimeter via 4-connected edge counting
            # Interior edges between wet and dry pixels
            diff_h = (
                mask[..., :, 1:] - mask[..., :, :-1]
            ).abs().sum(dim=(-3, -2, -1))
            diff_v = (
                mask[..., 1:, :] - mask[..., :-1, :]
            ).abs().sum(dim=(-3, -2, -1))
            # Image-border edges for pixels at the boundary
            border = (
                mask[..., :, 0].sum(dim=(-2, -1))
                + mask[..., :, -1].sum(dim=(-2, -1))
                + mask[..., 0, :].sum(dim=(-2, -1))
                + mask[..., -1, :].sum(dim=(-2, -1))
            )
            perimeter = (diff_h + diff_v + border) * pixel_size_km  # [B]

            # Only meaningful where there is precipitation
            has_precip = area > pixel_area
            violated = (perimeter.pow(2) < 4.0 * math.pi * area) & has_precip

            denom = has_precip.sum().item()
            if denom == 0:
                continue
            rates.append(violated.sum().item() / denom)

    if not rates:
        return 0.0, 0.0
    return float(sum(rates) / len(rates)), float(max(rates))


# ---------------------------------------------------------------
# Per-feature gamma R² (accumulator)
# ---------------------------------------------------------------


class PerFeatureR2Accumulator:
    """Accumulate predicted and target gamma tensors across validation.

    R² is a population statistic; computing it per-batch and averaging
    introduces bias when sample variance differs across batches. This
    accumulator stores tensors on CPU to avoid GPU memory pressure
    and computes R² in one shot at the end of validation.
    """

    FEATURE_NAMES = ("A", "P", "T")

    def __init__(self) -> None:
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(
        self, pred_gamma_log: torch.Tensor, target_gamma_log: torch.Tensor
    ) -> None:
        """Add a batch of gamma tensors.

        Parameters
        ----------
        pred_gamma_log, target_gamma_log : Tensor [B, 3, Q]
            Log-space gamma values.
        """
        self._preds.append(pred_gamma_log.detach().float().cpu())
        self._targets.append(target_gamma_log.detach().float().cpu())

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()

    def compute(self) -> Dict[str, float]:
        """Return per-feature R² across the accumulated samples.

        Returns
        -------
        dict
            Keys: ``"R2_A"``, ``"R2_P"``, ``"R2_T"``.
        """
        if not self._preds:
            return {f"R2_{n}": float("nan") for n in self.FEATURE_NAMES}

        preds = torch.cat(self._preds, dim=0)  # [N, 3, Q]
        targets = torch.cat(self._targets, dim=0)

        out: Dict[str, float] = {}
        for k, name in enumerate(self.FEATURE_NAMES):
            y_pred = preds[:, k, :].flatten()
            y_true = targets[:, k, :].flatten()
            ss_res = (y_true - y_pred).pow(2).sum()
            ss_tot = (y_true - y_true.mean()).pow(2).sum()
            r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-9)
            out[f"R2_{name}"] = float(r2.item())
        return out