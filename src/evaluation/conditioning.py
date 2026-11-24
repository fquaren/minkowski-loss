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


def member_case_scores(tmax, cmax, pmax, nexc_pred, nexc_obs, u, qs=(0.5, 0.9)):
    """The main table's case-level tail scores, per ensemble member, then averaged over members.

    pmax, nexc_pred : (M, N) per-member patch maxima and pixel counts >= u (a deterministic model: M = 1)
    tmax, cmax, nexc_obs : (N,) as in case_scores

    Every score is a function of one member against the observation, so averaging over members gives the
    expected score of a single draw; scoring member 0 alone (case_scores) adds member noise to an FM row.
    false_alarm_ratio = P(observed max < u | predicted max >= u): unlike the false-alarm rate it does not
    depend on how many non-events the subset holds. maxq_ratio_q* = q-quantile of predicted over observed
    patch maxima on the observed-event cases (unpaired), the stand-in for the GPD return-level bias,
    which cannot be fitted for weak-tail models.
    """
    tmax, cmax = np.asarray(tmax, float), np.asarray(cmax, float)
    pmax, nexc_pred = np.atleast_2d(np.asarray(pmax, float)), np.atleast_2d(np.asarray(nexc_pred, float))
    nexc_obs = np.asarray(nexc_obs, float)
    event = tmax >= u
    n_sel = int(event.sum())
    by_input = np.zeros(tmax.size, bool)
    if n_sel:
        by_input[np.argsort(-cmax, kind="stable")[:n_sel]] = True
    per = []
    for p, n in zip(pmax, nexc_pred):
        alarm = p >= u
        s = {"hit_rate": (p[event] >= u).mean() if n_sel else np.nan,
             "false_alarm_ratio": (alarm & ~event).sum() / alarm.sum() if alarm.any() else np.nan,
             "exceedance_ratio_obs": n[event].sum() / max(nexc_obs[event].sum(), 1.0),
             "exceedance_ratio_input": n[by_input].sum() / max(nexc_obs[by_input].sum(), 1.0),
             "peak_ratio_median_obs": np.median(p[event] / np.maximum(tmax[event], 1e-6)) if n_sel else np.nan}
        for q in qs:
            s[f"maxq_ratio_q{round(100 * q)}"] = (np.quantile(p[event], q) / np.quantile(tmax[event], q)
                                                   if n_sel else np.nan)
        per.append(s)
    out = {"u": float(u), "n_members": len(per), "n_obs_selected": n_sel}
    for k in per[0]:
        v = np.array([s[k] for s in per], float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out
