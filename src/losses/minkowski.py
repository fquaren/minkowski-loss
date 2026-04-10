"""
Differentiable loss functions for Minkowski functional emulation.

Contains:
  - MinkowskiLoss: W1-like distance on quantile curves (emulator training)
  - HomoscedasticMinkowskiLoss: learned multi-task weighting variant
  - AnalyticalMinkowskiLoss: differentiable relaxation via sigmoid/Goedel
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinkowskiLoss(nn.Module):
    """L1 Wasserstein-like distance between log-transformed quantile curves.

    Integrates |pred - target| over the quantile/threshold axis using
    the trapezoidal rule. The integration variable is the physical
    threshold, so loss contributions are weighted by threshold spacing.

    Parameters
    ----------
    quantile_levels : list or np.ndarray
        Physical thresholds (mm/h) or quantile levels used as the
        integration variable.
    """

    def __init__(self, quantile_levels):
        super().__init__()
        self.register_buffer(
            "quantiles",
            torch.tensor(quantile_levels, dtype=torch.float32),
        )

    def forward(self, pred_log, target_log, w_a=1.0, w_p=1.0, w_t=1.0):
        """
        Parameters
        ----------
        pred_log : Tensor [B, 3, Q]
        target_log : Tensor [B, 3, Q]
        w_a, w_p, w_t : float
            Per-functional weights (area, perimeter, topology).

        Returns
        -------
        total : Tensor [B]
        dist_a, dist_p, dist_t : Tensor [B] each
        """
        abs_diff = torch.abs(pred_log.float() - target_log.float())
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)  # [B, 3]

        total = w_a * dist[:, 0] + w_p * dist[:, 1] + w_t * dist[:, 2]
        return total, dist[:, 0], dist[:, 1], dist[:, 2]


class HomoscedasticMinkowskiLoss(nn.Module):
    """Learned multi-task weighting via homoscedastic uncertainty.

    Implements Kendall, Gal & Cipolla (2018): each task gets a learnable
    log-variance parameter that balances its contribution to the total
    loss. The regularization term prevents any task from being zeroed out.

    IMPORTANT: the parameters of this module must be included in the
    optimizer alongside the model parameters:
        optimizer = Adam(list(model.parameters()) + list(criterion.parameters()))

    Reference:
        Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
        Losses for Scene Geometry and Semantics", CVPR 2018.
        DOI: 10.1109/CVPR.2018.00781

    Parameters
    ----------
    quantile_levels : list or np.ndarray
        Integration variable for the trapezoidal rule.
    n_tasks : int
        Number of Minkowski functionals (default 3: A, P, topology).
    """

    def __init__(self, quantile_levels, n_tasks=3):
        super().__init__()
        self.register_buffer(
            "quantiles",
            torch.tensor(quantile_levels, dtype=torch.float32),
        )
        # log(sigma^2), initialized to 0 → sigma^2 = 1 → equal weighting
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, pred_log, target_log):
        """
        Parameters
        ----------
        pred_log : Tensor [B, 3, Q]
        target_log : Tensor [B, 3, Q]

        Returns
        -------
        total_loss : scalar
        dist_a, dist_p, dist_t : scalar each (unweighted, for logging)
        learned_weights : Tensor [3] (exp(-log_var), for monitoring)
        """
        abs_diff = torch.abs(pred_log.float() - target_log.float())
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)  # [B, 3]
        dist_mean = dist.mean(dim=0)  # [3]

        # precision_i = 1 / sigma_i^2 = exp(-log_var_i)
        precision = torch.exp(-self.log_vars)
        # L = sum_i [ 0.5 * precision_i * L_i + 0.5 * log_var_i ]
        total = (0.5 * precision * dist_mean + 0.5 * self.log_vars).sum()

        return (
            total,
            dist_mean[0].detach(),
            dist_mean[1].detach(),
            dist_mean[2].detach(),
            precision.detach(),
        )


class AnalyticalMinkowskiLoss(nn.Module):
    """Differentiable Minkowski functional approximation via continuous relaxation.

    Replaces discrete thresholding with temperature-controlled sigmoids.
    Area and perimeter are computed from the soft excursion sets; the
    Euler characteristic uses the Goedel t-norm (min) on a cubical grid.

    Morphological pre-processing (close → open) and a persistence mask
    stabilize the topological computation against noise.

    Parameters
    ----------
    thresholds : array-like
        Physical intensity thresholds in mm/h.
    pixel_size_km : float
        Pixel edge length.
    init_factor : float
        Initial sigmoid temperature = threshold * init_factor.
    min_temp : float
        Minimum temperature to prevent numerical issues.
    persistence_thresh : float
        Minimum local prominence for a feature to contribute to Euler.
    """

    def __init__(
        self,
        thresholds,
        pixel_size_km=2.0,
        init_factor=0.1,
        min_temp=1e-3,
        persistence_thresh=1.87,
    ):
        super().__init__()
        self.pixel_size_km = pixel_size_km
        self.pixel_area = pixel_size_km ** 2
        self.persistence_thresh = persistence_thresh

        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds, dtype=torch.float32).view(1, -1, 1, 1),
        )
        base_temps = np.maximum(np.array(thresholds) * init_factor, min_temp)
        self.register_buffer(
            "base_temps",
            torch.tensor(base_temps, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def forward(self, pred_phys, target_gamma_log, anneal_factor=1.0):
        """
        Parameters
        ----------
        pred_phys : Tensor [B, 1, H, W]
            Predicted field in physical space (mm/h).
        target_gamma_log : Tensor [B, 4, Q] or [B, 3, Q]
            Log1p-transformed targets. If 4 channels, Euler is computed
            as channel 2 - channel 3.
        anneal_factor : float
            Multiplier for sigmoid temperature (decays during training).

        Returns
        -------
        loss : scalar
        """
        temps = self.base_temps * anneal_factor

        # --- Morphological pre-processing for topology ---
        dilated = F.max_pool2d(pred_phys, kernel_size=3, stride=1, padding=1)
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-closed, kernel_size=3, stride=1, padding=1)
        field_topo = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(field_topo, kernel_size=15, stride=1, padding=7)

        # --- Area and perimeter (from raw field, not morphologically filtered) ---
        p_raw = torch.sigmoid((pred_phys - self.thresholds) / temps)  # [B, Q, H, W]
        area = torch.sum(p_raw, dim=(2, 3)) * self.pixel_area

        p_pad = F.pad(p_raw, (1, 1, 1, 1), mode="replicate")
        dx = (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]) / 2.0
        dy = (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1]) / 2.0
        perimeter = torch.sum(
            torch.sqrt(dx ** 2 + dy ** 2 + 1e-8), dim=(2, 3)
        ) * self.pixel_size_km

        # --- Euler characteristic via Goedel t-norm ---
        p_base = torch.sigmoid((field_topo - self.thresholds) / temps)
        pers_mask = torch.sigmoid(
            (local_max - (self.thresholds + self.persistence_thresh)) / temps
        )
        p_topo = torch.min(p_base, pers_mask)

        V = torch.sum(p_topo, dim=(2, 3))
        E_x = torch.sum(torch.min(p_topo[:, :, :, :-1], p_topo[:, :, :, 1:]), dim=(2, 3))
        E_y = torch.sum(torch.min(p_topo[:, :, :-1, :], p_topo[:, :, 1:, :]), dim=(2, 3))
        F_faces = torch.sum(
            torch.min(
                torch.min(p_topo[:, :, :-1, :-1], p_topo[:, :, :-1, 1:]),
                torch.min(p_topo[:, :, 1:, :-1], p_topo[:, :, 1:, 1:]),
            ),
            dim=(2, 3),
        )
        euler = V - E_x - E_y + F_faces  # [B, Q]

        # --- Assemble predicted gamma and compute loss ---
        pred_gamma_phys = torch.stack([area, perimeter, euler], dim=1)  # [B, 3, Q]

        from src.utils import signed_log1p, signed_expm1

        pred_gamma_log = signed_log1p(pred_gamma_phys)

        # Process targets: combine B0-B1 if 4 channels provided
        if target_gamma_log.shape[1] == 4:
            target_raw = signed_expm1(target_gamma_log)
            target_euler = target_raw[:, 2, :] - target_raw[:, 3, :]
            target_processed = torch.stack(
                [target_raw[:, 0, :], target_raw[:, 1, :], target_euler], dim=1
            )
            target_log = signed_log1p(target_processed)
        else:
            target_log = target_gamma_log.float()

        abs_diff = torch.abs(pred_gamma_log - target_log)
        dist = torch.trapezoid(abs_diff, self.thresholds.view(-1), dim=2)

        return dist.sum(dim=1).mean()
