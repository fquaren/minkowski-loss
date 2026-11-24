"""Tests for src/evaluation/conditioning.py: known answers for twCRPS and the case scores."""

import numpy as np
from scipy import integrate, stats

from src.evaluation.conditioning import case_scores, member_case_scores, twcrps_ensemble
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


def _cases(seed=0, n=400):
    rng = np.random.default_rng(seed)
    tmax = rng.gamma(2.0, 20.0, n)
    cmax = 0.2 * tmax + rng.normal(0, 2, n)
    nexc_o = np.where(tmax >= 50, rng.integers(1, 30, n), 0)
    return tmax, cmax, nexc_o


def test_member_scores_one_member_matches_case_scores():
    tmax, cmax, nexc_o = _cases()
    rng = np.random.default_rng(1)
    pmax = tmax * rng.uniform(0.5, 1.3, tmax.size)
    nexc_p = np.where(pmax >= 50, rng.integers(0, 30, tmax.size), 0)
    c = case_scores(tmax, cmax, pmax, np.zeros_like(pmax), nexc_p, nexc_o, 50.0)
    m = member_case_scores(tmax, cmax, pmax[None], nexc_p[None], nexc_o, 50.0)
    assert m["hit_rate"] == c["hit_rate"]
    assert m["exceedance_ratio_obs"] == c["obs_selected"]["exceedance_ratio"]
    assert m["exceedance_ratio_input"] == c["input_selected"]["exceedance_ratio"]
    assert m["peak_ratio_median_obs"] == c["obs_selected"]["peak_ratio_median"]


def test_member_scores_average_over_members():
    tmax, cmax, nexc_o = _cases()
    a, b = 0.6 * tmax, 1.2 * tmax                      # two members with known peak ratios
    na, nb = np.zeros(tmax.size), 2.0 * nexc_o          # exceedance ratios 0 and 2
    m = member_case_scores(tmax, cmax, np.stack([a, b]), np.stack([na, nb]), nexc_o, 50.0)
    assert np.isclose(m["peak_ratio_median_obs"], 0.9)
    assert np.isclose(m["exceedance_ratio_obs"], 1.0)
    assert np.isclose(m["maxq_ratio_q50"], 0.9) and np.isclose(m["maxq_ratio_q90"], 0.9)


def test_false_alarm_ratio_known_answer_and_independent_of_non_event_count():
    # 4 events all hit, 2 false alarms among 10 non-events -> FAR = 2/6
    tmax = np.r_[np.full(4, 80.0), np.full(10, 10.0)]
    pmax = np.r_[np.full(4, 70.0), 60.0, 60.0, np.full(8, 5.0)]
    z = np.zeros_like(tmax)
    m = member_case_scores(tmax, tmax, pmax[None], z[None], z, 50.0)
    assert np.isclose(m["false_alarm_ratio"], 2 / 6) and m["hit_rate"] == 1.0
    # add 100 correctly negative non-events: the ratio is unchanged, the rate is not
    t2, p2 = np.r_[tmax, np.full(100, 10.0)], np.r_[pmax, np.full(100, 5.0)]
    z2 = np.zeros_like(t2)
    m2 = member_case_scores(t2, t2, p2[None], z2[None], z2, 50.0)
    assert np.isclose(m2["false_alarm_ratio"], 2 / 6)
    c, c2 = (case_scores(t, t, p, z_, z_, z_, 50.0) for t, p, z_ in ((tmax, pmax, z), (t2, p2, z2)))
    assert c["false_alarm_rate"] != c2["false_alarm_rate"]


def test_no_alarm_gives_nan_far_not_zero():
    tmax = np.r_[80.0, 10.0]; z = np.zeros(2)
    m = member_case_scores(tmax, tmax, np.array([[5.0, 5.0]]), z[None], z, 50.0)
    assert np.isnan(m["false_alarm_ratio"]) and m["hit_rate"] == 0.0
