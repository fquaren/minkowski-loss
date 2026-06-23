"""
Baseline evaluation: analytical approximation of Minkowski functionals.

Uses the same differentiable relaxation as AnalyticalMinkowskiLoss
but applied at inference time for comparison against neural emulators.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils import signed_log1p


def compute_analytical_gamma(
    field_phys: torch.Tensor,
    thresholds: np.ndarray,
    pixel_size_km: float = 2.0,
    init_factor: float = 1e-5,
    min_temp: float = 1e-6,
    persistence_thresh: float = 1.87,
) -> torch.Tensor:
    """Compute differentiable Minkowski functionals via sigmoid relaxation.

    Parameters
    ----------
    field_phys : Tensor [B, 1, H, W]
        Physical precipitation field.
    thresholds : array-like, shape (Q,)
        Physical intensity thresholds.
    pixel_size_km : float
    init_factor, min_temp : float
        Sigmoid temperature control.
    persistence_thresh : float
        Minimum local prominence for Euler topology features.

    Returns
    -------
    Tensor [B, 3, Q]
        Log-transformed [area, perimeter, euler] via signed_log1p.
    """
    device = field_phys.device
    thresholds_t = torch.tensor(thresholds, dtype=torch.float32, device=device)
    temps = torch.tensor(
        np.maximum(np.array(thresholds) * init_factor, min_temp),
        dtype=torch.float32,
        device=device,
    )

    pixel_area = pixel_size_km**2
    areas, perimeters, eulers = [], [], []

    # Morphological pre-processing
    dilated = F.max_pool2d(field_phys, kernel_size=3, stride=1, padding=1)
    closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-closed, kernel_size=3, stride=1, padding=1)
    field_topo = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
    local_max = F.max_pool2d(field_topo, kernel_size=15, stride=1, padding=7)

    for q_idx in range(len(thresholds_t)):
        thresh = thresholds_t[q_idx]
        temp = temps[q_idx]

        # Area and perimeter from raw field
        p_raw = torch.sigmoid((field_phys - thresh) / temp)
        area = torch.sum(p_raw, dim=(1, 2, 3)) * pixel_area

        p_pad = F.pad(p_raw, (1, 1, 1, 1), mode="replicate")
        dx = (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]) / 2.0
        dy = (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1]) / 2.0
        perimeter = (
            torch.sum(torch.sqrt(dx**2 + dy**2 + 1e-8), dim=(2, 3)) * pixel_size_km
        )

        # Euler characteristic from morphologically filtered field
        p_base = torch.sigmoid((field_topo - thresh) / temp)
        pers_mask = torch.sigmoid((local_max - (thresh + persistence_thresh)) / temp)
        p_topo = torch.min(p_base, pers_mask)

        V = torch.sum(p_topo, dim=(1, 2, 3))
        E_x = torch.sum(
            torch.min(p_topo[:, :, :, :-1], p_topo[:, :, :, 1:]), dim=(1, 2, 3)
        )
        E_y = torch.sum(
            torch.min(p_topo[:, :, :-1, :], p_topo[:, :, 1:, :]), dim=(1, 2, 3)
        )
        F_faces = torch.sum(
            torch.min(
                torch.min(p_topo[:, :, :-1, :-1], p_topo[:, :, :-1, 1:]),
                torch.min(p_topo[:, :, 1:, :-1], p_topo[:, :, 1:, 1:]),
            ),
            dim=(1, 2, 3),
        )
        euler = V - E_x - E_y + F_faces

        areas.append(area)
        perimeters.append(perimeter)
        eulers.append(euler)

    gamma_phys = torch.stack(
        [
            torch.stack(areas, dim=1).unsqueeze(-1),
            torch.stack(perimeters, dim=1),
            torch.stack(eulers, dim=1).unsqueeze(-1),
        ],
        dim=1,
    )  # [B, 3, Q]

    return signed_log1p(gamma_phys)


def evaluate_analytical_baseline(
    test_dataset,
    physical_thresholds: np.ndarray,
    quantiles: np.ndarray,
    scaler_val: float,
    pixel_size_km: float = 2.0,
    batch_size: int = 200,
    topology_mode: str = "euler",
) -> dict:
    """Run analytical approximation on a test set and compute metrics.

    Parameters
    ----------
    test_dataset : ZarrMixupDataset
    physical_thresholds : (Q,) physical thresholds in mm/h
    quantiles : (Q,) for the integration variable
    scaler_val : float
    pixel_size_km : float
    batch_size : int
    topology_mode : str

    Returns
    -------
    dict of metrics
    """
    from src.evaluation.metrics import evaluate_predictions
    from src.utils import signed_expm1

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, multiprocessing_context="spawn",)

    y_true_list = []
    y_pred_list = []

    for inputs, log_targets in tqdm(loader, desc="Analytical baseline"):
        # Denormalize to physical space
        scaled = torch.clamp(inputs[:, 0:1, :, :] * scaler_val, max=7.0)
        phys = F.relu(torch.expm1(scaled))

        with torch.no_grad():
            gamma_ana = compute_analytical_gamma(
                phys,
                physical_thresholds,
                pixel_size_km=pixel_size_km,
            )

        y_pred_list.append(gamma_ana.numpy())
        y_true_list.append(log_targets.numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    # Analytical outputs Euler; if targets are already 3-channel, use directly
    # If targets are 4-channel, we need to combine B0-B1
    if y_true.shape[1] == 4:
        t_raw = np.sign(y_true) * np.expm1(np.abs(y_true))
        euler_raw = t_raw[:, 2, :] - t_raw[:, 3, :]
        euler_log = np.sign(euler_raw) * np.log1p(np.abs(euler_raw))
        y_true = np.stack([y_true[:, 0, :], y_true[:, 1, :], euler_log], axis=1)

    feature_names = ["Area", "Perimeter", "Euler"]
    return evaluate_predictions(y_true, y_pred, quantiles, feature_names)
