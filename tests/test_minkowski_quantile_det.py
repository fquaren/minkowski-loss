"""
The quantile_mass form of the deterministic (Study 1) Minkowski loss (DECISIONS §41).

Same decisive case as tests/test_reward_form.py: a 17 mm core under a target above 50 mm.
The threshold form gives the core no gradient; the quantile_mass form must pull it up.
An unset MINKOWSKI_FORM must build exactly the old loss. No data or GPU needed.
"""
import numpy as np
import pytest
import torch

from src.losses.competing import QuantileMassMinkowskiLoss, build_structural_loss
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.trainers.reward_finetune import _log_gamma

U = [0.1, 0.192, 0.314, 0.509, 0.828, 1.35, 2.11, 3.43, 5.36, 8.61, 13.5, 20.2, 31.7, 50.1, 79.0]
ANNEAL = 0.05


@pytest.fixture(scope="module")
def geom():
    return AnalyticalMinkowskiLoss(physical_thresholds=np.array(U, np.float32),
                                   pixel_size_km=2.0, topology_mode="euler", area_mode="ste")


def _cell(peak, n=48, c=(24, 24), r=12, seed=0):
    y, x = np.mgrid[:n, :n]
    d = np.sqrt((x - c[1]) ** 2 + (y - c[0]) ** 2)
    f = np.where(d <= r, peak * np.exp(-(d / (0.4 * r)) ** 2), 0.0)
    f = f + np.where(d <= r + 6, 0.3, 0.0)
    f = f + 0.05 * np.random.default_rng(seed).random((n, n))
    return torch.tensor(f, dtype=torch.float32).view(1, 1, n, n)


def _core_grad(loss_fn, geom, y, pred):
    pred = pred.clone().requires_grad_(True)
    tg = _log_gamma(geom, y, ANNEAL).detach()
    loss_fn(pred, tg, y, ANNEAL).backward()
    top = pred.detach()[0, 0] >= 16.0
    return pred.grad[0, 0][top]


def test_exact_prediction_scores_zero(geom):
    y = _cell(80.0)
    loss = QuantileMassMinkowskiLoss(geom)(y, _log_gamma(geom, y, ANNEAL), y, ANNEAL)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_threshold_form_gives_the_core_no_gradient(geom):
    y, pred = _cell(80.0), _cell(17.0, seed=1)
    g = _core_grad(lambda p, tg, yy, a: geom(p, tg, anneal_factor=a), geom, y, pred)
    assert g.abs().max() < 1e-6


def test_quantile_form_pulls_the_core_up(geom):
    y, pred = _cell(80.0), _cell(17.0, seed=1)
    g = _core_grad(QuantileMassMinkowskiLoss(geom), geom, y, pred)
    assert g.abs().max() > 1e-4
    nz = g[g.abs() > 1e-8]
    assert (nz < 0).all()          # descent raises the core


def _config(tmp_path, **kw):
    np.save(tmp_path / "physical_thresholds.npy", np.array(U, np.float32))
    return {"PREPROCESSED_DATA_DIR": str(tmp_path), "TOPOLOGY_MODE": "euler", **kw}


def test_unset_form_builds_the_old_loss(tmp_path, geom):
    sl = build_structural_loss("minkowski", _config(tmp_path), "cpu")
    assert sl.name == "minkowski" and not sl.needs_target_field
    assert isinstance(sl.fn, AnalyticalMinkowskiLoss)
    y, pred = _cell(80.0), _cell(17.0, seed=1)
    tg = _log_gamma(geom, y, ANNEAL)
    assert float(sl(None, None, pred, y, gamma_target=tg, anneal_factor=ANNEAL)) == \
        pytest.approx(float(geom(pred, tg, anneal_factor=ANNEAL)))


def test_quantile_form_is_selected_and_receives_the_target(tmp_path, geom):
    sl = build_structural_loss("minkowski", _config(tmp_path, MINKOWSKI_FORM="quantile_mass"), "cpu")
    assert sl.name == "minkowski (quantile_mass)" and sl.needs_target_field
    y, pred = _cell(80.0), _cell(17.0, seed=1)
    tg = _log_gamma(geom, y, ANNEAL)
    assert float(sl(None, None, pred, y, gamma_target=tg, anneal_factor=ANNEAL)) == \
        pytest.approx(float(QuantileMassMinkowskiLoss(geom)(pred, tg, y, ANNEAL)))


def test_unknown_form_raises(tmp_path):
    with pytest.raises(ValueError):
        build_structural_loss("minkowski", _config(tmp_path, MINKOWSKI_FORM="quantile"), "cpu")
