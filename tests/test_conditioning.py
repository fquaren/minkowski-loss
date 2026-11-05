"""Tests for src/evaluation/conditioning.py: known answers for twCRPS and the case scores."""

import numpy as np
from scipy import integrate, stats

from src.evaluation.conditioning import case_scores, twcrps_ensemble
from src.evaluation.extreme_metrics import crps_ensemble


def test_twcrps_below_all_data_is_crps():
    rng = np.random.default_rng(0)
    ens, y = rng.gamma(2.0, 3.0, (8, 50)), rng.gamma(2.0, 3.0, 50)
    np.testing.assert_allclose(twcrps_ensemble(ens, y, t=-1.0), crps_ensemble(ens, y))


def test_twcrps_one_member_is_weighted_mae():
    x, y = np.array([[1.0, 5.0, 12.0, 30.0]]), np.array([20.0, 2.0, 15.0, 31.0])
    t = 10.0
    np.testing.assert_allclose(twcrps_ensemble(x, y, t)[...], np.abs(np.maximum(x[0], t) - np.maximum(y, t)))


def test_twcrps_matches_gaussian_quadrature():
    # twCRPS(F, y) = int_t^inf (F(z) - 1{y <= z})^2 dz for weight 1{z >= t}
    mu, sd, t = 10.0, 4.0, 13.0
    m = 2000                                     # quantile grid: deterministic, error O(1/m)
    ens = stats.norm(mu, sd).ppf((np.arange(m) + 0.5) / m)[:, None]
    for y in (5.0, 14.0, 22.0):
        f = lambda z: (stats.norm(mu, sd).cdf(z) - (y <= z)) ** 2
        ref = sum(integrate.quad(f, a, b, limit=200)[0] for a, b in ((t, max(t, y)), (max(t, y), 60.0)))
        got = twcrps_ensemble(ens, np.array([y]), t)[0]
        np.testing.assert_allclose(got, ref, rtol=0.01, atol=1e-3)


def test_identity_model_is_perfect():
    rng = np.random.default_rng(2)
    tmax = rng.gamma(1.5, 15.0, 400)
    cmax = tmax / 4 + rng.normal(0, 2, 400)
    nexc = rng.integers(0, 30, 400) * (tmax >= 50)
    s = case_scores(tmax, cmax, tmax, (tmax >= 50).astype(float), nexc, nexc, 50.0)
    assert s["hit_rate"] == 1.0 and s["false_alarm_rate"] == 0.0 and s["brier_max_ge_u"] == 0.0
    for k in ("obs_selected", "input_selected"):
        assert s[k]["peak_ratio_median"] == 1.0
    assert s["obs_selected"]["exceedance_ratio"] == 1.0
    assert 0.0 < s["input_selected"]["share_obs_events"] <= 1.0


def test_input_selection_same_size_and_independent_of_prediction():
    tmax = np.array([60, 55, 10, 5, 70, 1.0])
    cmax = np.array([3, 20, 25, 1, 15, 2.0])
    a = case_scores(tmax, cmax, tmax, np.zeros(6), np.zeros(6), np.ones(6), 50.0)
    b = case_scores(tmax, cmax, tmax * 3, np.ones(6), np.zeros(6), np.ones(6), 50.0)
    assert a["n_obs_selected"] == 3
    # selected by cmax: cases 2, 1, 4 -> obs events among them: 1 and 4 -> 2/3
    assert np.isclose(a["input_selected"]["share_obs_events"], 2 / 3)
    assert a["input_selected"]["share_obs_events"] == b["input_selected"]["share_obs_events"]


def test_overprediction_is_caught_by_false_alarms():
    tmax = np.array([60.0, 10.0, 12.0, 8.0])
    over = case_scores(tmax, tmax, tmax * 7, np.ones(4), np.ones(4), np.ones(4), 50.0)
    assert over["hit_rate"] == 1.0 and over["false_alarm_rate"] == 1.0
