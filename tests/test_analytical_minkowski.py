"""
Tests for the rewritten AnalyticalMinkowskiLoss in src/losses/minkowski.py.
Mirror of tests/test_gamma.py style. Run anywhere torch is installed
(no data, no GUDHI required).
"""
import numpy as np
import torch
import pytest
from skimage import measure
from src.losses.minkowski import AnalyticalMinkowskiLoss


def _disk(n=64, c=(32, 32), r=10, val=5.0):
    y, x = np.mgrid[:n, :n]
    f = np.zeros((n, n), np.float32)
    f[np.sqrt((x - c[1]) ** 2 + (y - c[0]) ** 2) <= r] = val
    return f


def _annulus(n=64, c=(32, 32), ri=6, ro=14, val=5.0):
    y, x = np.mgrid[:n, :n]
    d = np.sqrt((x - c[1]) ** 2 + (y - c[0]) ** 2)
    f = np.zeros((n, n), np.float32)
    f[(d <= ro) & (d > ri)] = val
    return f


def _loss(thr=(1.0,), q=(0.5,), area_mode="soft", **kw):
    return AnalyticalMinkowskiLoss(
        np.array(thr, np.float32), np.array(q, np.float32),
        pixel_size_km=2.0, area_mode=area_mode, **kw,
    )


def _chi(field, anneal=0.05):
    x = torch.from_numpy(field).view(1, 1, *field.shape)
    _, _, c = _loss()._functionals(x, anneal_factor=anneal)
    return c.item()


@pytest.mark.parametrize("field,expected", [
    (_disk(), 1),
    (_disk(c=(20, 20), r=8) + _disk(c=(45, 45), r=8), 2),
    (_annulus(), 0),
])
def test_chi_matches_exact_euler(field, expected):
    mask = field >= 1.0
    assert measure.euler_number(mask, connectivity=1) == expected
    assert abs(_chi(field) - expected) < 1e-2


def test_soft_converges_to_hard_as_tau_shrinks():
    field = _disk(c=(20, 20), r=8) + _disk(c=(45, 45), r=8)
    vals = [_chi(field, anneal=a) for a in (1.0, 0.2, 0.05)]
    assert all(abs(v - 2.0) < 1e-2 for v in vals)


def test_ste_forward_is_hard_count():
    loss = _loss(area_mode="ste")
    x = (torch.rand(1, 1, 16, 16) * 5).requires_grad_(True)
    area, _, _ = loss._functionals(x, anneal_factor=1.0)
    hard = (x.detach() >= loss.u).float().sum((2, 3)) * loss.pixel_area
    assert torch.allclose(area.detach(), hard, atol=1e-4)


def test_ste_backward_is_soft_nonzero():
    loss = _loss(area_mode="ste")
    x = (torch.rand(1, 1, 16, 16) * 5).requires_grad_(True)
    area, _, _ = loss._functionals(x, anneal_factor=1.0)
    area.sum().backward()
    assert x.grad.abs().sum().item() > 0.0


def test_finite_difference_matches_autodiff():
    loss = _loss(thr=(0.2, 1.0, 4.0), q=(0.1, 0.5, 0.9), area_mode="soft").double()
    x = torch.from_numpy(np.random.RandomState(1).rand(8, 8)).view(1, 1, 8, 8).requires_grad_(True)

    def scalar(xin):
        a, p, c = loss._functionals(xin, 1.0)
        return a.sum() + p.sum() + c.sum()

    scalar(x).backward()
    ana, eps, num = x.grad.clone(), 1e-4, torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.numel()):
            d = torch.zeros_like(x).view(-1); d[i] = eps; d = d.view_as(x)
            num.view(-1)[i] = (scalar((x + d).detach()) - scalar((x - d).detach())) / (2 * eps)
    assert ((ana - num).norm() / num.norm()).item() < 1e-4


def test_integration_is_over_quantile_axis():
    loss = _loss(thr=(0.2, 4.0), q=(0.1, 0.9))
    assert torch.allclose(loss.q, torch.tensor([0.1, 0.9]))


# --- topology_mode="b0": differentiable persistent-homology beta_0 -------------
# Requires GUDHI. Skipped automatically if not installed.
gudhi = pytest.importorskip("gudhi")


def _b0_loss(thr=(1.0,), q=(0.5,)):
    return AnalyticalMinkowskiLoss(
        np.array(thr, np.float32), np.array(q, np.float32),
        pixel_size_km=2.0, topology_mode="b0", area_mode="soft",
    )


@pytest.mark.parametrize("field,expected", [
    (_disk(), 1),
    (_disk(c=(20, 20), r=8) + _disk(c=(45, 45), r=8), 2),
    (_annulus(), 1),   # annulus is ONE connected component (beta_0=1), unlike chi=0
])
def test_betti0_counts_components(field, expected):
    x = torch.from_numpy(field).view(1, 1, *field.shape)
    _, _, b0 = _b0_loss()._functionals(x, anneal_factor=0.02)
    assert abs(b0.item() - expected) < 1e-2


def test_betti0_gradient_flows_and_is_sparse():
    field = _disk(c=(20, 20), r=8) + _disk(c=(45, 45), r=8)
    x = torch.from_numpy(field).view(1, 1, *field.shape).requires_grad_(True)
    _, _, b0 = _b0_loss(thr=(1.0, 3.0), q=(0.3, 0.7))._functionals(x, anneal_factor=1.0)
    b0.sum().backward()
    assert torch.isfinite(x.grad).all()
    # PH gradient is supported on O(#components) critical pixels only
    assert 0 < (x.grad.abs() > 1e-9).sum().item() < field.size // 4