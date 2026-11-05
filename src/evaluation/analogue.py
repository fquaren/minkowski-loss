"""
Peak-normalised Minkowski features: how unseen is a storm's *shape*? (DECISIONS §43, todo C4;
RESEARCH_NOTES §1 prediction 4, §3 Q5.)

The thresholds are fractions of the patch's own peak, u_k = f_k * max(field), so the features describe the
geometry of the excursion sets independently of intensity: a field and the same field scaled by any factor
have identical features. With fixed physical thresholds every case above the R4 cut would be trivially unseen
(nothing in cut training reaches 50.1 mm), which says nothing about geometry.

Per fraction f_k: area (km^2), Cauchy-Crofton perimeter (km) and the exact 4-connected Euler characteristic,
the hard-mask versions of the loss's own estimators (src/data/gamma.py), with the loss's transforms:
log1p, log1p, signed log1p.
"""

from __future__ import annotations

import numpy as np

from src.data.gamma import _compute_area, _compute_perimeter_crofton, compute_euler_characteristic_exact

FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def shape_features(field: np.ndarray, pixel_km: float = 2.0, fractions=FRACTIONS) -> np.ndarray:
    """(3 * len(fractions),) features [log1p A..., log1p P..., signed log1p chi...]; zeros for a dry field."""
    f = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
    peak = f.max()
    if peak <= 0:
        return np.zeros(3 * len(fractions), np.float32)
    u = peak * np.asarray(fractions, dtype=np.float64)
    masks = f[:, :, None] >= u[None, None, :]
    area = _compute_area(masks, pixel_km ** 2)
    per = _compute_perimeter_crofton(masks, pixel_km)
    chi = compute_euler_characteristic_exact(f, u.astype(np.float32))
    return np.concatenate([np.log1p(area), np.log1p(per), np.sign(chi) * np.log1p(np.abs(chi))]).astype(np.float32)
