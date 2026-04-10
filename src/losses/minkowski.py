"""
Minkowski image loss functions for training emulators and SR models.

Implements the W1-like distance over quantile curves (Eq. 2 in paper)
with optional homoscedastic multi-task uncertainty weighting
(Kendall, Gal & Cipolla, CVPR 2018).
"""

import torch
import torch.nn as nn


class MinkowskiLoss(nn.Module):
    """
    Fixed-weight Minkowski image loss.

    Computes the trapezoidal-rule L1 integral of |pred - target| over
    the quantile/threshold axis for each functional (A, P, topo),
    then returns the weighted sum.

    The integration variable is the quantile level vector, so the loss
    is implicitly weighted toward regions where quantiles are spaced
    further apart. Verify your QUANTILE_LEVELS spacing is intentional.
    """

    def __init__(self, quantile_levels):
        super().__init__()
        self.register_buffer(
            "quantiles",
            torch.tensor(quantile_levels, dtype=torch.float32),
        )

    def forward(
        self,
        pred_log: torch.Tensor,
        target_log: torch.Tensor,
        w_a: float = 1.0,
        w_p: float = 1.0,
        w_topo: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pred_log : Tensor [B, 3, Q]
            log1p-transformed predicted gamma vector.
        target_log : Tensor [B, 3, Q]
            log1p-transformed target gamma vector.
        w_a, w_p, w_topo : float
            Per-functional weights.

        Returns
        -------
        total : Tensor [B], weighted sum of per-functional distances.
        dist_a, dist_p, dist_topo : Tensor [B] each.
        """
        pred_log = pred_log.float()
        target_log = target_log.float()

        abs_diff = torch.abs(pred_log - target_log)  # [B, 3, Q]
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)  # [B, 3]

        total = w_a * dist[:, 0] + w_p * dist[:, 1] + w_topo * dist[:, 2]
        return total, dist[:, 0], dist[:, 1], dist[:, 2]


class HomoscedasticMinkowskiLoss(nn.Module):
    """
    Learned-weight Minkowski image loss via homoscedastic uncertainty.

    Instead of fixed weights (w_a, w_p, w_topo), learns one log-variance
    parameter per functional. The loss for functional i is:

        L_i = 0.5 * exp(-s_i) * dist_i + 0.5 * s_i

    where s_i = log(sigma_i^2) is a learnable parameter. The log-variance
    regulariser prevents any task from being suppressed entirely.

    IMPORTANT: include self.parameters() in the optimizer, e.g.:
        optimizer = Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=...
        )

    Reference: Kendall, Gal & Cipolla (2018), DOI: 10.1109/CVPR.2018.00781
    """

    def __init__(self, quantile_levels, n_tasks: int = 3):
        super().__init__()
        self.register_buffer(
            "quantiles",
            torch.tensor(quantile_levels, dtype=torch.float32),
        )
        # Initialise log(sigma^2) = 0 => sigma^2 = 1 => equal weighting
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(
        self,
        pred_log: torch.Tensor,
        target_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pred_log : Tensor [B, 3, Q]
        target_log : Tensor [B, 3, Q]

        Returns
        -------
        total : scalar, mean over batch of the uncertainty-weighted loss.
        dist_a, dist_p, dist_topo : Tensor [B] each (unweighted, for logging).
        """
        pred_log = pred_log.float()
        target_log = target_log.float()

        abs_diff = torch.abs(pred_log - target_log)
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)  # [B, 3]

        # Kendall et al. weighting: precision * loss + log-variance penalty
        precision = torch.exp(-self.log_vars)  # [3]
        # dist.mean(dim=0) is [3], one mean-distance per functional
        weighted = 0.5 * precision * dist.mean(dim=0) + 0.5 * self.log_vars
        total = weighted.sum()

        return total, dist[:, 0], dist[:, 1], dist[:, 2]

    def get_learned_weights(self) -> dict:
        """Return the current effective weights for logging."""
        with torch.no_grad():
            sigma_sq = torch.exp(self.log_vars)
            precision = 1.0 / sigma_sq
        return {
            "sigma2_A": sigma_sq[0].item(),
            "sigma2_P": sigma_sq[1].item(),
            "sigma2_topo": sigma_sq[2].item(),
            "weight_A": precision[0].item(),
            "weight_P": precision[1].item(),
            "weight_topo": precision[2].item(),
        }
