"""
Differentiable loss functions for Minkowski functional emulation.

Contains:
  - MinkowskiLoss: W1-like distance on quantile curves (emulator training)
  - HomoscedasticMinkowskiLoss: learned multi-task weighting variant
  - AnalyticalMinkowskiLoss: differentiable relaxation; topology via either the
      local Euler characteristic chi = beta_0 - beta_1 ("euler") OR differentiable
      persistent homology for the pure connected-component count beta_0 ("b0").

Drop-in replacement for src/losses/minkowski.py. MinkowskiLoss and
HomoscedasticMinkowskiLoss are unchanged.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class MinkowskiLoss(nn.Module):
    """L1 Wasserstein-like distance between log-transformed quantile curves."""

    def __init__(self, quantile_levels):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantile_levels, dtype=torch.float32))

    def forward(self, pred_log, target_log, w_a=1.0, w_p=1.0, w_t=1.0):
        abs_diff = torch.abs(pred_log.float() - target_log.float())
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)  # [B, 3]
        total = w_a * dist[:, 0] + w_p * dist[:, 1] + w_t * dist[:, 2]
        return total, dist[:, 0], dist[:, 1], dist[:, 2]


class HomoscedasticMinkowskiLoss(nn.Module):
    """Learned multi-task weighting via homoscedastic uncertainty.

    Reference: Kendall et al., CVPR 2018, DOI: 10.1109/CVPR.2018.00781.
    """

    def __init__(self, quantile_levels, n_tasks=3):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantile_levels, dtype=torch.float32))
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, pred_log, target_log):
        abs_diff = torch.abs(pred_log.float() - target_log.float())
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)  # [B, 3]
        dist_mean = dist.mean(dim=0)
        precision = torch.exp(-self.log_vars)
        total = (0.5 * precision * dist_mean + 0.5 * self.log_vars).sum()
        return (total, dist_mean[0].detach(), dist_mean[1].detach(),
                dist_mean[2].detach(), precision.detach())


# ---------------------------------------------------------------------------
# Differentiable persistent homology for beta_0 (connected components)
# ---------------------------------------------------------------------------

class _Dim0PersistencePairs(torch.autograd.Function):
    """Superlevel-set dim-0 persistence pairs of a 2D field, with gradients.

    Forward: returns (births, deaths) in ORIGINAL (superlevel) coordinates for
    every connected component, computed with GUDHI on the negated field. The
    essential component (global max, never dies) gets a finite sentinel death far
    below the data so its death-gate saturates to 1 with ~zero gradient.

    Backward: each birth/death value equals the field at a specific critical pixel
    (the GUDHI coface), so the gradient of a birth/death w.r.t. the field is a
    one-hot at that pixel. The gradient is therefore exact but supported only on
    the O(#components) critical pixels -- the known sparsity of PH gradients.

    Reference: Carriere et al., "Optimizing persistent homology based functions",
    ICML 2021, arXiv:2010.08356.
    """

    @staticmethod
    def forward(ctx, field: torch.Tensor):
        import gudhi as gd  # lazy: only needed for topology_mode="b0"
        f = field.detach().cpu().numpy().astype(np.float64)
        H, W = f.shape
        flat = f.reshape(-1)
        cc = gd.CubicalComplex(dimensions=(H, W), top_dimensional_cells=(-flat))
        cc.persistence()
        fin, ess = cc.cofaces_of_persistence_pairs()
        sentinel = float(flat.min() - 1.0e4)
        bcells, dcells, births, deaths = [], [], [], []
        if len(fin) > 0 and fin[0].size > 0:
            for bc, dc in fin[0]:
                bcells.append(int(bc)); dcells.append(int(dc))
                births.append(flat[bc]); deaths.append(flat[dc])
        if len(ess) > 0 and ess[0].size > 0:
            for bc in ess[0]:
                bcells.append(int(bc)); dcells.append(-1)   # -1 => no death pixel
                births.append(flat[bc]); deaths.append(sentinel)
        ctx.bcells, ctx.dcells, ctx.shape = bcells, dcells, (H, W)
        ctx.dtype, ctx.device = field.dtype, field.device
        b = torch.tensor(births, dtype=field.dtype, device=field.device)
        d = torch.tensor(deaths, dtype=field.dtype, device=field.device)
        return b, d

    @staticmethod
    def backward(ctx, grad_b, grad_d):
        H, W = ctx.shape
        g = torch.zeros(H * W, dtype=ctx.dtype, device=ctx.device)
        for i, bc in enumerate(ctx.bcells):
            g[bc] = g[bc] + grad_b[i]
        for i, dc in enumerate(ctx.dcells):
            if dc >= 0:
                g[dc] = g[dc] + grad_d[i]
        return g.view(H, W)


class AnalyticalMinkowskiLoss(nn.Module):
    r"""Differentiable Minkowski functional approximation.

    Area and perimeter are always differentiable and dense. The topology channel
    is selected by ``topology_mode``:

      "euler"  chi = V - Ex - Ey + F on the cubical complex (4-connected, Goedel-min
               t-norm). DENSE gradients on the near-threshold band. The hard limit
               equals skimage.measure.euler_number(connectivity=1); the matching
               offline target is gamma.compute_euler_characteristic_exact (channel 4).
               chi = beta_0 - beta_1: in hole-rich fields it is beta_1-dominated.

      "b0"     pure connected-component count beta_0 via differentiable persistent
               homology (GUDHI superlevel filtration + _Dim0PersistencePairs). A soft
               count of dim-0 bars alive at each threshold, persistence-gated by
               ``persistence_thresh_b0`` to match the offline target gamma._count_b0
               (channel 2). SPARSE gradients (critical pixels only) and per-image CPU
               cost -- correct quantity for hole-rich fields, at a price.

    Integration of |log(gamma_hat) - log(gamma)| is over the QUANTILE axis. Soft
    excursion sets use the PHYSICAL climatological thresholds (mm/h).

    Parameters
    ----------
    physical_thresholds : array-like, shape (Q,)
        Climatological thresholds u^(i) in mm/h (load_physical_thresholds(config)).
    quantile_levels : array-like, shape (Q,)
        Strictly increasing quantile levels in [0, 1]; the integration variable.
    pixel_size_km : float
        Pixel edge length (km). Pixel area = pixel_size_km**2.
    topology_mode : {"euler", "b0"}
        Topology channel. Must match config["TOPOLOGY_MODE"] and the offline target
        channel selected by gamma.select_topology_target / datasets._gamma_to_log.
    tau_factor, tau_min : float
        Per-threshold sigmoid temperature tau = max(u * tau_factor, tau_min) (mm/h).
    area_mode : {"ste", "soft"}
        "ste" = hard-count-forward / soft-gradient-backward (default).
        "soft" = plain soft sum (use for finite-difference gradient checks).
    crofton_w_axial, crofton_w_diag : float
        Calibrated Cauchy-Crofton weights (pixel units; * pixel_size_km).
    persistence_thresh_b0 : float
        Persistence noise floor (mm/h) for the "b0" soft count. Pass
        load_persistence_thresholds(config)[0] to match the offline target.
    tau_persistence : float
        Temperature (mm/h) of the persistence gate in "b0" mode.
    """

    def __init__(
        self,
        physical_thresholds,
        quantile_levels,
        pixel_size_km: float = 2.0,
        topology_mode: str = "euler",
        tau_factor: float = 0.1,
        tau_min: float = 1e-3,
        area_mode: str = "ste",
        crofton_w_axial: float = 0.473215,
        crofton_w_diag: float = 0.217716,
        persistence_thresh_b0: float = 0.0,
        tau_persistence: float = 0.1,
    ):
        super().__init__()
        if area_mode not in ("ste", "soft"):
            raise ValueError(f"area_mode must be 'ste' or 'soft', got {area_mode!r}")
        if topology_mode not in ("euler", "b0"):
            raise ValueError(f"topology_mode must be 'euler' or 'b0', got {topology_mode!r}")
        self.pixel_size_km = float(pixel_size_km)
        self.pixel_area = float(pixel_size_km) ** 2
        self.topology_mode = topology_mode
        self.area_mode = area_mode
        self.tau_min = float(tau_min)
        self.w_ax = float(crofton_w_axial)
        self.w_di = float(crofton_w_diag)
        self.persistence_thresh_b0 = float(persistence_thresh_b0)
        self.tau_persistence = float(tau_persistence)

        u = np.asarray(physical_thresholds, dtype=np.float32)
        q = np.asarray(quantile_levels, dtype=np.float32)
        if u.shape != q.shape:
            raise ValueError("physical_thresholds and quantile_levels must align 1:1.")
        if np.any(np.diff(q) <= 0):
            raise ValueError("quantile_levels must be strictly increasing.")

        self.register_buffer("u", torch.tensor(u, dtype=torch.float32).view(1, -1, 1, 1))
        base = np.maximum(u * tau_factor, tau_min)
        self.register_buffer("tau_base", torch.tensor(base, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("q", torch.tensor(q, dtype=torch.float32))

    # -- topology: dense Euler --------------------------------------------------
    def _euler(self, s: torch.Tensor) -> torch.Tensor:
        """chi = V - Ex - Ey + F via Goedel-min on the soft mask s [B,Q,H,W]."""
        V = s.sum(dim=(2, 3))
        Ex = torch.minimum(s[:, :, :, :-1], s[:, :, :, 1:]).sum(dim=(2, 3))
        Ey = torch.minimum(s[:, :, :-1, :], s[:, :, 1:, :]).sum(dim=(2, 3))
        F = torch.minimum(
            torch.minimum(s[:, :, :-1, :-1], s[:, :, :-1, 1:]),
            torch.minimum(s[:, :, 1:, :-1], s[:, :, 1:, 1:]),
        ).sum(dim=(2, 3))
        return V - Ex - Ey + F  # [B, Q]

    # -- topology: differentiable persistent-homology beta_0 --------------------
    def _betti0(self, pred_phys: torch.Tensor, tau_row: torch.Tensor) -> torch.Tensor:
        """Soft connected-component count via PH. pred_phys [B,1,H,W] -> [B,Q]."""
        B = pred_phys.shape[0]
        u = self.u.view(-1)                       # [Q]
        tau_p = max(self.tau_persistence, self.tau_min)
        out = []
        for b in range(B):
            births, deaths = _Dim0PersistencePairs.apply(pred_phys[b, 0])
            if births.numel() == 0:
                out.append(torch.zeros_like(u))
                continue
            bb = births.view(-1, 1)               # [K,1]
            dd = deaths.view(-1, 1)
            tr = tau_row.view(1, -1)              # [1,Q]
            alive = torch.sigmoid((bb - u.view(1, -1)) / tr) * \
                    torch.sigmoid((u.view(1, -1) - dd) / tr)        # [K,Q]
            gate = torch.sigmoid(((bb - dd) - self.persistence_thresh_b0) / tau_p)  # [K,1]
            out.append((gate * alive).sum(0))     # [Q]
        return torch.stack(out, 0)                # [B,Q]

    def _functionals(self, pred_phys: torch.Tensor, anneal_factor: float):
        """Return (area, perimeter, topo) each [B,Q] from a physical field [B,1,H,W]."""
        tau = (self.tau_base * float(anneal_factor)).clamp_min(self.tau_min)  # [1,Q,1,1]
        s = torch.sigmoid((pred_phys - self.u) / tau)                          # [B,Q,H,W]

        # AREA: straight-through (hard forward, soft backward)
        if self.area_mode == "ste":
            hard = (pred_phys >= self.u).to(s.dtype)
            s_area = hard + (s - s.detach())
        else:
            s_area = s
        area = s_area.sum(dim=(2, 3)) * self.pixel_area

        # PERIMETER: isotropic Cauchy-Crofton on the soft mask
        ax = ((s[:, :, :, 1:] - s[:, :, :, :-1]).abs().sum(dim=(2, 3))
              + (s[:, :, 1:, :] - s[:, :, :-1, :]).abs().sum(dim=(2, 3)))
        di = ((s[:, :, 1:, 1:] - s[:, :, :-1, :-1]).abs().sum(dim=(2, 3))
              + (s[:, :, 1:, :-1] - s[:, :, :-1, 1:]).abs().sum(dim=(2, 3)))
        perimeter = (self.w_ax * ax + self.w_di * di) * self.pixel_size_km

        # TOPOLOGY: chi (dense) or beta_0 (sparse, PH)
        if self.topology_mode == "euler":
            topo = self._euler(s)
        else:
            topo = self._betti0(pred_phys, tau.view(-1))
        return area, perimeter, topo

    def forward(self, pred_phys: torch.Tensor, target_gamma_log: torch.Tensor,
                anneal_factor: float = 1.0) -> torch.Tensor:
        """
        pred_phys : Tensor [B,1,H,W] in physical mm/h.
        target_gamma_log : Tensor [B,3,Q] signed_log1p targets [logA, logP, topo],
            where topo is signed_log1p(chi) for "euler" or log1p(beta_0) for "b0".
            A legacy [B,4,Q] = [A,P,B0,B1] is accepted ONLY for "euler" (uses B0-B1,
            an 8-connected, persistence-filtered quantity inconsistent with the
            analytical chi -- regenerate targets with gamma.compute_gamma_matrix).
        """
        from src.utils import signed_log1p, signed_expm1

        area, perimeter, topo = self._functionals(pred_phys, anneal_factor)
        pred_log = signed_log1p(torch.stack([area, perimeter, topo], dim=1))  # [B,3,Q]

        if target_gamma_log.shape[1] == 4:
            if self.topology_mode != "euler":
                raise ValueError("4-channel legacy target is only valid for topology_mode='euler'.")
            tg = signed_expm1(target_gamma_log.float())
            chi_t = tg[:, 2, :] - tg[:, 3, :]
            target_log = signed_log1p(torch.stack([tg[:, 0, :], tg[:, 1, :], chi_t], dim=1))
        else:
            target_log = target_gamma_log.float()

        abs_diff = torch.abs(pred_log - target_log)               # [B,3,Q]
        dist = torch.trapezoid(abs_diff, self.q, dim=2)           # integrate over q -> [B,3]
        return dist.sum(dim=1).mean()