"""
Feature inversion test for emulator gradient quality.

Optimises a noise vector through the frozen emulator to match a
target gamma vector, then computes the exact Minkowski functionals
of the resulting field to assess whether the emulator's gradients
are physically meaningful.

Critical fix from review: optimisation now operates in normalized
log-space (matching the emulator's training domain), not in raw
physical space.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.gamma import compute_gamma_matrix, select_topology_target
from src.utils import signed_log1p


def create_synthetic_target(
    mode: str,
    quantile_levels: list,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a synthetic gamma target in log-space.

    Parameters
    ----------
    mode : str
        "large_storm", "small_storm", or "no_storm"
    quantile_levels : list of float
    device : str

    Returns
    -------
    Tensor [1, 3, Q] in signed-log space
    """
    Q = len(quantile_levels)
    target = torch.zeros(1, 3, Q, device=device)
    max_precip = 150.0
    thresh = torch.tensor(quantile_levels, device=device)
    cutoff = thresh > max_precip

    if mode == "large_storm":
        target[0, 0] = torch.logspace(4.0, 3.4, Q, device=device)  # area (km²)
        target[0, 1] = torch.logspace(3.5, 3.0, Q, device=device)  # perimeter (km)
        target[0, 2] = torch.linspace(1, 20, Q, device=device)      # topology
    elif mode == "small_storm":
        target[0, 0] = torch.linspace(2000, 10, Q, device=device)
        target[0, 1] = torch.linspace(600, 10, Q, device=device)
        target[0, 2, 0] = 1
    elif mode == "no_storm":
        pass  # all zeros
    else:
        raise ValueError(f"Unknown mode: {mode}")

    target[:, :, cutoff] = 0.0
    return signed_log1p(target)


def run_inversion(
    model,
    log_target: torch.Tensor,
    scaler_val: float,
    patch_size: int = 128,
    steps: int = 200,
    lr: float = 0.1,
    tv_weight: float = 1e-5,
    l2_weight: float = 1e-6,
    device: str = "cpu",
) -> dict:
    """Optimise a normalized input to match a target gamma via the emulator.

    The optimisation is in log-normalised space [0, 1], matching the
    emulator's training domain. The field is constrained to [0, 1]
    via clamping after each step.

    Parameters
    ----------
    model : nn.Module (frozen emulator)
    log_target : Tensor [1, 3, Q]
    scaler_val : float
    patch_size : int
    steps : int
    lr : float
    tv_weight, l2_weight : float
    device : str

    Returns
    -------
    dict with keys: initial_phys, final_phys, loss_history, final_gamma_pred
    """
    model.eval()

    # Initialise with smooth noise in normalised space
    low_res = torch.randn(1, 1, patch_size // 4, patch_size // 4, device=device)
    x_norm = F.interpolate(low_res, size=patch_size, mode="bilinear",
                            align_corners=False)
    x_norm = torch.clamp(x_norm * 0.1 + 0.3, 0.0, 1.0)

    initial_phys = np.expm1(x_norm.detach().cpu().numpy()[0, 0] * scaler_val)
    initial_phys = np.maximum(initial_phys, 0.0)

    x_param = x_norm.detach().clone().requires_grad_(True)
    optimizer = optim.Adam([x_param], lr=lr)

    loss_history = {"total": [], "mse": [], "tv": [], "l2": []}

    for step in tqdm(range(steps), desc="Inversion", leave=False):
        optimizer.zero_grad()

        pred_phys = model(x_param)
        pred_log = signed_log1p(pred_phys)

        mse = F.mse_loss(pred_log, log_target)
        tv = (torch.sum(torch.abs(x_param[:, :, :, :-1] - x_param[:, :, :, 1:])) +
              torch.sum(torch.abs(x_param[:, :, :-1, :] - x_param[:, :, 1:, :])))
        l2 = torch.mean(x_param ** 2)

        total = mse + tv_weight * tv + l2_weight * l2
        total.backward()
        optimizer.step()

        with torch.no_grad():
            x_param.clamp_(0.0, 1.0)

        loss_history["total"].append(total.item())
        loss_history["mse"].append(mse.item())
        loss_history["tv"].append((tv * tv_weight).item())
        loss_history["l2"].append((l2 * l2_weight).item())

    # Final field in physical space
    final_norm = x_param.detach().cpu().numpy()[0, 0]
    final_phys = np.maximum(np.expm1(final_norm * scaler_val), 0.0)

    # Final emulator prediction
    with torch.no_grad():
        final_pred = model(x_param)
    final_gamma_pred = signed_log1p(final_pred).cpu().numpy()[0]

    return {
        "initial_phys": initial_phys,
        "final_phys": final_phys,
        "loss_history": loss_history,
        "final_gamma_pred": final_gamma_pred,
    }


def verify_inversion(
    final_phys: np.ndarray,
    log_target: torch.Tensor,
    physical_thresholds: np.ndarray,
    pixel_size_km: float,
    thresh_b0: float,
    thresh_b1: float,
    topology_mode: str = "euler",
) -> dict:
    """Compute exact Minkowski functionals on the dreamt field.

    This is the critical verification step: does the emulator-optimised
    field actually match the target in ground-truth metric space?

    Parameters
    ----------
    final_phys : (H, W) dreamt precipitation field
    log_target : Tensor [1, 3, Q] target in signed-log space
    physical_thresholds : (Q,) thresholds
    pixel_size_km : float
    thresh_b0, thresh_b1 : float
    topology_mode : str

    Returns
    -------
    dict with exact_gamma, target_gamma (both physical), and residuals
    """
    gamma_4ch = compute_gamma_matrix(
        final_phys, physical_thresholds, pixel_size_km, thresh_b0, thresh_b1,
    )
    gamma_3ch = select_topology_target(gamma_4ch, mode=topology_mode)
    exact_log = signed_log1p(gamma_3ch)

    target_np = log_target.cpu().numpy()[0]  # [3, Q]
    residual = exact_log - target_np
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    return {
        "exact_gamma_log": exact_log,
        "target_gamma_log": target_np,
        "residual": residual,
        "rmse": rmse,
    }


def plot_inversion_results(
    initial_phys: np.ndarray,
    final_phys: np.ndarray,
    arch_name: str,
    mode: str,
    save_dir: str,
    gt_phys: np.ndarray = None,
):
    """Visualise initial noise, dreamt field, and optional ground truth."""
    os.makedirs(save_dir, exist_ok=True)
    has_gt = gt_phys is not None
    ncols = 3 if has_gt else 2

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    cmap = plt.get_cmap("cividis_r").copy()
    cmap.set_bad("lightgrey")

    vmax = np.max(final_phys)
    if has_gt:
        vmax = max(vmax, np.max(gt_phys))
    vmax = max(vmax, 1.0)

    axes[0].imshow(initial_phys, cmap=cmap, origin="lower",
                   vmin=0, vmax=max(np.max(initial_phys), 1e-3))
    axes[0].set_title("Initial noise")
    axes[0].axis("off")

    im = axes[1].imshow(final_phys, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
    axes[1].set_title(f"Dreamt ({arch_name})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="mm/h")

    if has_gt:
        im2 = axes[2].imshow(gt_phys, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
        axes[2].set_title("Ground truth")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="mm/h")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"inversion_{mode}_{arch_name}.pdf"), dpi=300)
    plt.close()
