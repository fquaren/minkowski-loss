"""
Tests for the reward forms in src/trainers/reward_finetune.py (DECISIONS §39).

The decisive case: a prediction whose heavy core (16-17 mm) lies far from every threshold
(13.5 and 20.2 mm are > 20 tau away) while the target exceeds 20.2, 31.7 and 50.1 mm. The
original threshold-form reward gives that core no gradient; the quantile_mass form must
pull it up. No data or GPU needed.
"""
import numpy as np
import pytest
import torch

from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.trainers.reward_finetune import (
    minkowski_reward, reward_features, reward_from_features, _log_gamma,
)

U = [0.1, 0.192, 0.314, 0.509, 0.828, 1.35, 2.11, 3.43, 5.36, 8.61, 13.5, 20.2, 31.7, 50.1, 79.0]
ANNEAL = 0.05


@pytest.fixture(scope="module")
def geom():
    return AnalyticalMinkowskiLoss(physical_thresholds=np.array(U, np.float32),
                                   pixel_size_km=2.0, topology_mode="euler", area_mode="ste")


def _cell(peak, n=48, c=(24, 24), r=12, seed=0):
    """A rain cell: light background ramping to `peak` at the centre, plus light noise."""
    y, x = np.mgrid[:n, :n]
    d = np.sqrt((x - c[1]) ** 2 + (y - c[0]) ** 2)
    f = np.where(d <= r, peak * np.exp(-(d / (0.4 * r)) ** 2), 0.0)
    f = f + np.where(d <= r + 6, 0.3, 0.0)
    f = f + 0.05 * np.random.default_rng(seed).random((n, n))
    return torch.tensor(f, dtype=torch.float32).view(1, 1, n, n)


def _target(geom, y):
    return _log_gamma(geom, y, ANNEAL).detach()


def _peak_grad(geom, mode, form, M=2):
    y = _cell(80.0)
    pred = torch.cat([_cell(17.0, seed=s) for s in range(M)], 0).unsqueeze(1)  # [M,1,1,H,W]
    pred = pred.clone().requires_grad_(True)
    r = minkowski_reward(geom, pred, _target(geom, y), mode, ANNEAL, 0.5, form, y)
    r.backward()
    g = pred.grad[:, 0, 0]                                   # [M,H,W]
    top = pred.detach()[:, 0, 0] >= 16.0                     # the heavy core of each member
    return g, top


@pytest.mark.parametrize("mode", ["per_sample", "energy", "distributional"])
def test_threshold_form_gives_the_heavy_core_no_gradient(geom, mode):
    # documents the failure being fixed: the 20.2 / 31.7 / 50.1 thresholds sit above the
    # 17 mm prediction, so nothing pulls the core up
    g, top = _peak_grad(geom, mode, "threshold")
    assert g[top].abs().max() < 1e-6


@pytest.mark.parametrize("mode", ["per_sample", "energy", "distributional"])
def test_quantile_form_pulls_the_heavy_core_up(geom, mode):
    g, top = _peak_grad(geom, mode, "quantile_mass")
    core = g[top]
    assert core.abs().max() > 1e-4
    # gradient descent on the reward raises the core: d reward / d value < 0 where nonzero
    nz = core[core.abs() > 1e-8]
    assert (nz < 0).all()


@pytest.mark.parametrize("mode", ["per_sample", "energy", "distributional"])
def test_unit_weights_reproduce_the_original_reward(geom, mode):
    y = _cell(80.0)
    pred = torch.cat([_cell(30.0, seed=s) for s in range(3)], 0).unsqueeze(1)
    lg, tg, W = reward_features(geom, pred, _target(geom, torch.cat([y, y])[:1]), ANNEAL)
    assert W is None
    r0 = reward_from_features(geom, lg, tg, None, mode, 0.5)
    r1 = reward_from_features(geom, lg, tg, torch.ones_like(tg), mode, 0.5)
    assert torch.allclose(r0, r1, rtol=1e-6, atol=1e-8)
    assert torch.allclose(r0, minkowski_reward(geom, pred, tg, mode, ANNEAL, 0.5))


def test_area_channel_is_the_rank_matched_quantile(geom):
    y = _cell(80.0)
    pred = _cell(30.0).unsqueeze(0)
    lg, tg, W = reward_features(geom, pred, _target(geom, y), ANNEAL, "quantile_mass", y)
    ys = torch.sort(y.flatten(), descending=True).values
    ps = torch.sort(pred.flatten(), descending=True).values
    for j, u in enumerate(U):
        n = int((y.flatten() >= u).sum())
        if n == 0:
            assert W[0, 0, j] == 0 and tg[0, 0, j] == 0 and lg[0, 0, 0, j] == 0
        else:
            assert W[0, 0, j] == 1
            assert torch.isclose(tg[0, 0, j], torch.log1p(ys[n - 1]))
            assert torch.isclose(lg[0, 0, 0, j], torch.log1p(ps[n - 1]))


def test_identical_prediction_scores_zero(geom):
    y = _cell(80.0)
    pred = y.unsqueeze(0).repeat(3, 1, 1, 1, 1)
    for mode in ("per_sample", "energy"):
        r = minkowski_reward(geom, pred, _target(geom, y), mode, ANNEAL, 0.5, "quantile_mass", y)
        assert abs(float(r)) < 1e-6


def test_weights_depend_on_the_target_only(geom):
    y = _cell(80.0)
    a = reward_features(geom, _cell(30.0).unsqueeze(0), _target(geom, y), ANNEAL, "quantile_mass", y)[2]
    b = reward_features(geom, _cell(5.0, seed=3).unsqueeze(0), _target(geom, y), ANNEAL,
                        "quantile_mass", y)[2]
    assert torch.equal(a, b)


def test_mass_weights_fall_with_pixel_count_and_cap_above_the_maximum(geom):
    y = _cell(40.0)                                   # max ~40 mm: 50.1 and 79 are empty
    W = reward_features(geom, y.unsqueeze(0), _target(geom, y), ANNEAL, "quantile_mass", y)[2]
    w = W[0, 1]
    assert torch.isclose(w.mean(), torch.tensor(1.0))
    assert torch.equal(W[0, 1], W[0, 2])              # perimeter and Euler share the weights
    n = (y.flatten().unsqueeze(0) >= torch.tensor(U).unsqueeze(1)).sum(1)
    pop = n >= 1
    assert (torch.diff(w[pop]) >= 0).all()            # fewer pixels -> larger weight
    top = int(pop.sum()) - 1
    assert torch.allclose(w[~pop], w[top].expand(int((~pop).sum())))
