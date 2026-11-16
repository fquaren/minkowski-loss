"""Quantile mapping (Baselines v2 G1): identity on its own marginal, in-sample marginal match."""
import numpy as np
import torch

from src.evaluation.quantile_map import LogHistogram, QuantileMap, fit_table, default_levels


def _obs(n=400_000, seed=0):
    r = np.random.default_rng(seed)
    x = r.lognormal(0.0, 1.5, n)
    x[r.random(n) < 0.7] = 0.0                     # mostly dry, heavy tail
    return x


def _fit(pred, obs):
    hp, ho = LogHistogram(), LogHistogram()
    hp.update(pred); ho.update(obs)
    t = fit_table(hp, ho)
    return QuantileMap(t["q_pred"], t["q_obs"]), t


def test_qm_of_target_is_identity():
    y = _obs()
    qm, _ = _fit(y, y)
    x = np.array([0.0, 0.01, 0.3, 1.0, 7.0, 40.0, 1e3])
    assert np.allclose(qm(x), x, rtol=1e-9, atol=0)


def test_in_sample_marginal_matches_target():
    y = _obs(seed=1)
    pred = 0.5 * y ** 0.8 + 0.02                   # too wet at the bottom, too weak at the top
    qm, t = _fit(pred, y)
    mapped = qm(pred)
    for p in (0.75, 0.9, 0.99, 0.999):
        assert np.isclose(np.quantile(mapped, p), np.quantile(y, p), rtol=2e-3), p
    assert np.mean(mapped == 0) > 0.69             # the drizzle is mapped back to dry


def test_multiplicative_extrapolation_and_torch():
    qm = QuantileMap([0.0, 1.0, 10.0], [0.0, 2.0, 30.0])
    assert np.isclose(qm(np.array([20.0]))[0], 60.0)
    x = torch.tensor([[0.0, 1.0, 10.0, 20.0]])
    out = qm(x)
    assert isinstance(out, torch.Tensor) and torch.allclose(out, torch.tensor([[0.0, 2.0, 30.0, 60.0]]))


def test_levels_reach_the_tail():
    lv = default_levels()
    assert lv[0] == 0.0 and np.isclose(1 - lv[-1], 1e-6) and np.all(np.diff(lv) > 0)


def test_top_level_keeps_min_pixels_above():
    y = _obs(n=100_000)
    _, t = _fit(y, y)
    assert (1 - t["levels"][-1]) * t["n_pixels"] >= 100
