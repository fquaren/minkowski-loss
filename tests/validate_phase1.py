"""
Phase-1 validation: run on the cluster, where data and offline gamma targets exist.

Validates the rewritten AnalyticalMinkowskiLoss *as an estimator*, decoupled from any
generator: it reads the high-resolution fields (original_precip, physical mm/h) and the
offline targets (gamma_targets) directly from the zarr store -- the exact pairing produced
by compute_gamma_targets.py -- and compares the analytical functionals to those targets.

This deliberately does NOT use DeterministicSRDataset: that dataset returns the
(low-resolution conditioning input, log-gamma) pair, not the high-resolution field the
targets were computed from, and applies a log transform. Reading the raw arrays is both
simpler and the conceptually correct estimator check.

PREREQUISITE. Regenerate gamma_targets with the new gamma.py first: they must be
(N, 5, Q) = [A, P_crofton, B0, B1, chi_exact]. Targets are stored as RAW physical values
(no log), so they compare directly to AnalyticalMinkowskiLoss._functionals output.

Checks:
  1. Per-functional R^2 of [A_hat, P_hat, topo_hat] vs the offline targets (held-out split).
  2. Finite-difference vs autodiff gradient agreement on the soft functionals.
  3. Gradient localisation rho: fraction of |d topo / d x| mass on the near-threshold band.

Usage:
    python validate_phase1.py --config config.yaml [--split validation] [--n 256]
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import torch
import zarr

from src.utils import load_config, load_physical_thresholds, load_persistence_thresholds
from src.losses.minkowski import AnalyticalMinkowskiLoss

# gamma_targets channel layout (axis 1): [A, P_crofton, B0, B1, chi_exact]
CH_AREA, CH_PERIM, CH_B0, CH_CHI = 0, 1, 2, 4


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Coefficient of determination, flattened over (sample, threshold)."""
    p, t = pred.ravel(), true.ravel()
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def _r2_per_q(pred, true):           # pred, true: (n, Q)
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)   # (Q,)




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=256, help="number of held-out samples for R^2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    u_phys = load_physical_thresholds(config)            # (Q,) mm/h
    q_lev = np.asarray(config["QUANTILE_LEVELS"], np.float32)
    px = float(config.get("PIXEL_SIZE_KM", 2.0))
    topo_mode = config.get("TOPOLOGY_MODE", "euler")

    thresh_b0 = 0.0
    if topo_mode == "b0":
        thresh_b0, _ = load_persistence_thresholds(config)  # match offline gamma._count_b0

    loss = AnalyticalMinkowskiLoss(
        physical_thresholds=u_phys,
        quantile_levels=q_lev,
        pixel_size_km=px,
        topology_mode=topo_mode,
        area_mode="ste",
        persistence_thresh_b0=thresh_b0,
    ).to(device)

    # ---- (1) R^2 vs offline targets, read straight from zarr ----
    zarr_path = os.path.join(config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr")
    group = zarr.open(zarr_path, mode="r")[args.split]
    n = min(args.n, group["original_precip"].shape[0])

    field = np.asarray(group["original_precip"][:n], dtype=np.float32)   # (n, H, W) mm/h
    gamma = np.asarray(group["gamma_targets"][:n], dtype=np.float32)     # (n, 5, Q) raw
    if gamma.shape[1] < 5:
        raise RuntimeError(
            f"gamma_targets has {gamma.shape[1]} channels; expected 5. "
            "Regenerate with the updated compute_gamma_targets.py before validating."
        )

    x = torch.from_numpy(field).unsqueeze(1).to(device)   # (n, 1, H, W)
    with torch.no_grad():
        a, p, t = loss._functionals(x, anneal_factor=0.05)  # near-hard; each (n, Q)
    a, p, t = (z.cpu().numpy() for z in (a, p, t))

    ch_topo = CH_CHI if topo_mode == "euler" else CH_B0
    topo_name = "chi (4-conn)" if topo_mode == "euler" else "beta_0 (PH)"
    print(f"Per-functional R^2 vs offline targets (split={args.split}, n={n}, anneal=0.05):")
    print(f"  R2_A    = {_r2(a, gamma[:, CH_AREA]):.4f}   (expect ~0.99)")
    print(f"  R2_P    = {_r2(p, gamma[:, CH_PERIM]):.4f}   (Crofton vs Crofton)")
    print(f"  R2_topo = {_r2(t, gamma[:, ch_topo]):.4f}   ({topo_name} vs same)")

    for name, pr, tg in [("A", a, gamma[:, CH_AREA]),
                     ("P", p, gamma[:, CH_PERIM]),
                     ("topo", t, gamma[:, ch_topo])]:
        r2q = _r2_per_q(pr, tg)
        print(name, "tail R^2 (q>=0.9):", np.round(r2q[q_lev >= 0.9], 4))

    # ---- (2) finite-difference vs autodiff on the SOFT functionals ----
    loss_soft = AnalyticalMinkowskiLoss(
        u_phys, q_lev, pixel_size_km=px, topology_mode=topo_mode,
        area_mode="soft", persistence_thresh_b0=thresh_b0,
    ).to(device).double()
    xg = (torch.rand(1, 1, 16, 16, dtype=torch.float64, device=device)
          * float(u_phys.max())).requires_grad_(True)

    def scalar(xin: torch.Tensor) -> torch.Tensor:
        aa, pp, tt = loss_soft._functionals(xin, anneal_factor=1.0)
        return aa.sum() + pp.sum() + tt.sum()

    scalar(xg).backward()
    ana = xg.grad.clone()
    eps, num = 1e-4, torch.zeros_like(xg)
    with torch.no_grad():
        for i in range(xg.numel()):
            d = torch.zeros_like(xg).view(-1); d[i] = eps; d = d.view_as(xg)
            num.view(-1)[i] = (scalar((xg + d).detach()) - scalar((xg - d).detach())) / (2 * eps)
    rel = ((ana - num).norm() / (num.norm() + 1e-12)).item()
    print(f"\nFD vs autodiff rel error (soft A+P+topo): {rel:.2e}   (expect < 1e-4)")

    # ---- (3) gradient localisation rho on the topology channel ----
    x2 = (torch.rand(1, 1, 32, 32, device=device) * float(u_phys.max())).requires_grad_(True)
    _, _, tt = loss.double()._functionals(x2.double(), anneal_factor=1.0)
    tt.sum().backward()
    g = x2.grad.abs().squeeze()
    tau = torch.as_tensor(np.maximum(u_phys * 0.1, 1e-3)).to(device)
    band = ((x2.detach().squeeze()[None]
             - torch.as_tensor(u_phys, device=device)[:, None, None]).abs()
            <= (3.0 * tau[:, None, None])).any(0)
    rho = (g[band].sum() / (g.sum() + 1e-12)).item()
    print(f"Gradient localisation rho (|d topo/dx| on +/-3 tau band): {rho:.3f}   (expect -> 1)")

    # ---- (4) diagnostic add-on, b0 mode ----
    g = x2.grad.abs().squeeze()
    nz = (g > 1e-9).float().mean().item()          # fraction of pixels with gradient
    for k in (3.0, 5.0, 8.0):
        band = ((x2.detach().squeeze()[None]
                - torch.as_tensor(u_phys, device=device)[:, None, None]).abs()
                <= (k * tau[:, None, None])).any(0)
        print(f"k={k}: rho={(g[band].sum()/(g.sum()+1e-12)).item():.3f}, nz_frac={nz:.4f}")


if __name__ == "__main__":
    main()