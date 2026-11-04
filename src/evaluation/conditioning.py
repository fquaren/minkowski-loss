"""
Scores that guard against the forecaster's dilemma (DECISIONS §43).

Scoring only cases selected because the *observation* was extreme rewards models that over-predict.
Two remedies, used side by side with the observation-conditioned scores:

1. The same scores on cases selected by the *coarse input*, which is available at inference and does
   not depend on the model. To add no free threshold, the input-selected set has as many cases as the
   observation-selected one: the cases with the largest coarse-input maxima.
2. Proper scores over all cases with weight on the tail:
   - the threshold-weighted CRPS with weight 1{z >= t}. With the chaining function v(z) = max(z, t),
     twCRPS(F, y) = CRPS(F o v^-1, v(y)), i.e. the ordinary CRPS after replacing every member x by
     max(x, t) and the observation y by max(y, t). One member gives the threshold-weighted MAE.
   - the Brier score of the event "patch maximum >= u", with the ensemble's member fraction as the
     forecast probability (0 or 1 for a deterministic model).
"""

from __future__ import annotations

import numpy as np

from src.evaluation.extreme_metrics import crps_ensemble


def twcrps_ensemble(ensemble: np.ndarray, obs: np.ndarray, t: float, fair: bool = True) -> np.ndarray:
    """Per-element threshold-weighted CRPS, weight 1{z >= t}. ensemble (M, ...), obs (...)."""
    return crps_ensemble(np.maximum(ensemble, t), np.maximum(obs, t), fair=fair)


def case_scores(tmax, cmax, pmax, pexc, nexc_pred, nexc_obs, u):
    """Case-level scores at level u under observation- and input-based selection.

    tmax, cmax, pmax : per-case target, coarse-input and predicted (one member) patch maxima
    pexc             : per-case forecast probability that the patch maximum is >= u
    nexc_pred/_obs   : per-case number of pixels >= u in the prediction (one member) / target
    """
    tmax, cmax, pmax, pexc = map(np.asarray, (tmax, cmax, pmax, pexc))
    nexc_pred, nexc_obs = np.asarray(nexc_pred, float), np.asarray(nexc_obs, float)
    event = tmax >= u
    n_sel = int(event.sum())
    out = {"u": float(u), "n_cases": int(tmax.size), "n_obs_selected": n_sel,
           "hit_rate": float((pmax[event] >= u).mean()) if n_sel else float("nan"),
           "false_alarm_rate": float((pmax[~event] >= u).mean()) if (~event).any() else float("nan"),
           "brier_max_ge_u": float(np.mean((pexc - event) ** 2))}
    by_input = np.zeros(tmax.size, bool)
    if n_sel:
        by_input[np.argsort(-cmax, kind="stable")[:n_sel]] = True
    for name, sel in (("obs_selected", event), ("input_selected", by_input)):
        if not sel.any():
            continue
        out[name] = {
            "share_obs_events": float(event[sel].mean()),
            "exceedance_ratio": float(nexc_pred[sel].sum() / max(nexc_obs[sel].sum(), 1.0)),
            "peak_ratio_median": float(np.median(pmax[sel] / np.maximum(tmax[sel], 1e-6))),
            "pred_max_ge_u_share": float((pmax[sel] >= u).mean()),
        }
    return out
