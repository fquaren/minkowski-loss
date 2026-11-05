"""
Per-channel weights of the Minkowski loss and the area-only ablation (DECISIONS §43, todo D2).

Known answers: unit weights reproduce the loss exactly, in both forms; and the area channel depends
only on the distribution of pixel values, so shuffling the prediction's pixels leaves the area-only
loss unchanged while it changes the full loss (perimeter and Euler see where the pixels are).
"""
import numpy as np
import pytest
import torch

from src.losses.competing import QuantileMassMinkowskiLoss
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.trainers.reward_finetune import _log_gamma

U = [0.1, 0.192, 0.314, 0.509, 0.828, 1.35, 2.11, 3.43, 5.36, 8.61, 13.5, 20.2, 31.7, 50.1, 79.0]
ANNEAL = 0.05


def _geom(w=(1.0, 1.0, 1.0)):
    return AnalyticalMinkowskiLoss(physical_thresholds=np.array(U, np.float32), pixel_size_km=2.0,
                                   topology_mode="euler", area_mode="ste", channel_weights=w)


def _field(peak, seed, n=48):
    y, x = np.mgrid[:n, :n]
    rng = np.random.default_rng(seed)
    f = sum(rng.uniform(0.3, 1) * peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))
            for cx, cy, s in rng.uniform([8, 8, 2], [40, 40, 7], (4, 3)))
    return torch.tensor(f + 0.05 * rng.random((n, n)), dtype=torch.float32).view(1, 1, n, n)


def _shuffled(t, seed=3):
    flat = t.flatten()
    return flat[torch.from_numpy(np.random.default_rng(seed).permutation(flat.numel()))].view_as(t)


def _losses(w, pred, y):
    g = _geom(w)
    tg = _log_gamma(g, y, ANNEAL).detach()
    return float(g(pred, tg, ANNEAL)), float(QuantileMassMinkowskiLoss(g)(pred, tg, y, ANNEAL))


def test_unit_weights_reproduce_the_loss():
    y, pred = _field(60.0, 0), _field(30.0, 1)
    old = AnalyticalMinkowskiLoss(physical_thresholds=np.array(U, np.float32), pixel_size_km=2.0,
                                  topology_mode="euler", area_mode="ste")      # no channel_weights
    tg = _log_gamma(old, y, ANNEAL).detach()
    ref = (float(old(pred, tg, ANNEAL)), float(QuantileMassMinkowskiLoss(old)(pred, tg, y, ANNEAL)))
    assert _losses((1.0, 1.0, 1.0), pred, y) == pytest.approx(ref, rel=1e-6)


def test_channels_add_up():
    y, pred = _field(60.0, 0), _field(30.0, 1)
    parts = [_losses(w, pred, y) for w in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    full = _losses((1.0, 1.0, 1.0), pred, y)
    for k in range(2):
        assert sum(p[k] for p in parts) == pytest.approx(full[k], rel=1e-5)


@pytest.mark.parametrize("form", [0, 1])           # 0 threshold, 1 quantile_mass
def test_area_only_ignores_where_the_pixels_are(form):
    y, pred = _field(60.0, 0), _field(30.0, 1)
    area = _losses((1.0, 0.0, 0.0), pred, y)[form]
    area_shuf = _losses((1.0, 0.0, 0.0), _shuffled(pred), y)[form]
    full = _losses((1.0, 1.0, 1.0), pred, y)[form]
    full_shuf = _losses((1.0, 1.0, 1.0), _shuffled(pred), y)[form]
    assert area_shuf == pytest.approx(area, rel=1e-5, abs=1e-7)
    assert abs(full_shuf - full) > 0.05 * full


def _trunc(u_max, w=(1.0, 1.0, 1.0)):
    u = np.array(U, np.float32)
    return AnalyticalMinkowskiLoss(physical_thresholds=u[u < u_max], pixel_size_km=2.0, topology_mode="euler",
                                   area_mode="ste", channel_weights=w, target_prefix=True)


def test_truncated_grid_equals_prefix_grid():
    # the full-grid gamma target, cut to its first Q columns, scores like a loss built on that grid alone
    y, pred = _field(60.0, 0), _field(30.0, 1)
    full = _geom()
    tg_full = _log_gamma(full, y, ANNEAL).detach()
    tr = _trunc(50.1)
    q = tr.xi.numel()
    assert q == 13
    explicit = AnalyticalMinkowskiLoss(physical_thresholds=np.array(U[:q], np.float32), pixel_size_km=2.0,
                                       topology_mode="euler", area_mode="ste")
    assert float(tr(pred, tg_full, ANNEAL)) == pytest.approx(float(explicit(pred, tg_full[..., :q], ANNEAL)), rel=1e-6)


def test_truncated_grid_is_blind_above_the_cut():
    # pixels at 60 or at 200 mm lie above every remaining threshold: the truncated loss cannot tell them apart
    y = _field(40.0, 0)
    a = _field(30.0, 1)
    a[0, 0, 20:24, 20:24] = 60.0
    b = a.clone()
    b[0, 0, 20:24, 20:24] = 200.0
    full = _geom()
    tg = _log_gamma(full, y, ANNEAL).detach()
    tr = _trunc(50.1)
    assert float(tr(a, tg, ANNEAL)) == pytest.approx(float(tr(b, tg, ANNEAL)), rel=1e-5, abs=1e-7)
    assert abs(float(full(a, tg, ANNEAL)) - float(full(b, tg, ANNEAL))) > 1e-3
