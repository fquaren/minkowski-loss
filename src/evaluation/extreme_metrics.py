"""
Structure metrics for precipitation super-resolution evaluation.

Contains SAL (structure-amplitude-location; Wernli et al. 2008, doi:10.1175/2008MWR2415.1)
and the ensemble CRPS estimator (Gneiting & Raftery 2007, doi:10.1198/016214506000001437).

SAL is meaningful for a deterministic (point) prediction and is the structural score for
the Study-1 backbone comparison. CRPS is a probabilistic score: for a single deterministic
field it reduces exactly to the mean absolute error, so it is informative only once the
generative model (Phase 2) produces an ensemble; `crps_ensemble` is provided for that stage.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage


def _objects(field: np.ndarray, f: float, p: float):
    """Threshold R* = f * percentile_p(wet pixels); label contiguous objects (4-conn)."""
    wet = field[field > 0]
    if wet.size == 0:
        return np.zeros(field.shape, dtype=int), 0
    r_star = f * float(np.percentile(wet, p))
    labels, n = ndimage.label(field >= r_star)
    return labels, n


def _scaled_volume(field: np.ndarray, labels: np.ndarray, n: int) -> float:
    """Precip-weighted mean of per-object scaled volume V_n / max_n (the S ingredient)."""
    if n == 0:
        return 0.0
    num = den = 0.0
    for k in range(1, n + 1):
        obj = field[labels == k]
        vn, rmax = obj.sum(), obj.max()
        if rmax <= 0:
            continue
        num += vn * (vn / rmax)
        den += vn
    return num / den if den > 0 else 0.0


def _com(field: np.ndarray):
    """Precipitation-weighted centre of mass (row, col)."""
    tot = field.sum()
    H, W = field.shape
    if tot <= 0:
        return np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    yy, xx = np.mgrid[0:H, 0:W]
    return np.array([(yy * field).sum() / tot, (xx * field).sum() / tot])


def _scatter(field: np.ndarray, labels: np.ndarray, n: int, c_total: np.ndarray) -> float:
    """Precip-weighted mean distance of object centres of mass from the total COM (L2)."""
    if n == 0:
        return 0.0
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W]
    num = den = 0.0
    for k in range(1, n + 1):
        m = labels == k
        rn = field[m].sum()
        if rn <= 0:
            continue
        cy = (yy[m] * field[m]).sum() / rn
        cx = (xx[m] * field[m]).sum() / rn
        num += rn * np.hypot(cy - c_total[0], cx - c_total[1])
        den += rn
    return num / den if den > 0 else 0.0


def sal(pred: np.ndarray, obs: np.ndarray, f: float = 1.0 / 15.0, p: float = 95.0):
    """SAL components for one (pred, obs) field pair, physical units, non-negative.

    Returns (S, A, L). S, A in [-2, 2]; L = L1 + L2 in [0, 2]. All zero for identical fields.
    """
    eps = 1e-12
    d_pred, d_obs = float(pred.mean()), float(obs.mean())
    A = (d_pred - d_obs) / (0.5 * (d_pred + d_obs) + eps)

    lp, npred = _objects(pred, f, p)
    lo, nobs = _objects(obs, f, p)
    v_pred = _scaled_volume(pred, lp, npred)
    v_obs = _scaled_volume(obs, lo, nobs)
    S = (v_pred - v_obs) / (0.5 * (v_pred + v_obs) + eps)

    cp, co = _com(pred), _com(obs)
    diag = float(np.hypot(*obs.shape))
    L1 = float(np.hypot(*(cp - co)) / diag)
    L2 = 2.0 * abs(_scatter(pred, lp, npred, cp) - _scatter(obs, lo, nobs, co)) / diag
    return float(S), float(A), float(L1 + L2)


def crps_ensemble(ensemble: np.ndarray, obs: np.ndarray, fair: bool = True) -> np.ndarray:
    """CRPS estimator from an ensemble.

    ensemble : (M, ...) ensemble members; obs : (...) observation.
    Returns the per-element CRPS (same shape as obs). The 'fair' (unbiased) estimator uses
    1/(M(M-1)) for the spread term; for M=1 this reduces to |x - obs| (= the absolute error).
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(obs, dtype=np.float64)
    m = ens.shape[0]
    skill = np.abs(ens - y[None, ...]).mean(axis=0)
    if m == 1:
        return skill
    diff = np.abs(ens[:, None, ...] - ens[None, :, ...]).sum(axis=(0, 1))
    denom = m * (m - 1) if fair else m * m
    spread = diff / denom
    return skill - 0.5 * spread