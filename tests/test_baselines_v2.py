"""Baselines v2 losses (EXPERIMENTS "Baselines v2" G1): known answers for each."""
import numpy as np
import pytest
import torch

from src.losses.competing import (FFT2DKernelCRPSLoss, PersistentHomologyLoss, PinballLoss, SoftFSSLoss,
                                  WeightedMSELoss, _diagram, train_exceedance_fractions)
from src.evaluation.tail_metrics import FSSAccumulator

U = [0.1, 1.0, 10.0, 50.0]


def _rain(seed, b=4, h=48, w=48, scale=8.0):
    """Smooth, mostly dry, heavy-tailed fields in mm."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 1, h, w, generator=g)
    k = torch.ones(1, 1, 7, 7) / 49.0
    x = torch.nn.functional.conv2d(x, k, padding=3) * 7.0
    return (torch.relu(x - 0.5) ** 2 * scale)


# ---- weighted MSE -------------------------------------------------------------------------
def test_wmse_constant_weights_is_mse():
    y, p = _rain(0), _rain(1)
    fn = WeightedMSELoss(U, [1.0] * len(U))
    yn, pn = torch.log1p(y) / 7, torch.log1p(p) / 7
    assert torch.allclose(fn(pn, yn, y), torch.mean((pn - yn) ** 2))


def test_wmse_zero_at_target_and_bin_weights():
    fn = WeightedMSELoss(U, [0.5, 0.1, 0.01, 0.001])
    y = torch.tensor([0.0, 0.1, 0.5, 1.0, 9.99, 10.0, 60.0]).view(1, 1, 1, -1)
    assert float(fn(y, y, y)) == 0.0
    # one unit of squared error per pixel: the loss is the mean of the bin weights
    w = torch.tensor([1.0, 2.0, 2.0, 10.0, 10.0, 100.0, 1000.0])
    assert torch.isclose(fn(y + 1, y, y), w.mean())


def test_exceedance_fractions_from_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    n, nq = 4, 2
    g = zarr.open(str(tmp_path / "preprocessed_dataset.zarr"), mode="w").create_group("train")
    a = np.zeros((n, 5, nq), dtype=np.float32)
    area = 16 * 16 * 4.0                                   # patch 16 px, 2 km pixels
    a[:, 0, 0] = [area, 0.5 * area, 0.0, 0.0]             # rows 0, 1 wet; rows 2, 3 dry
    a[:, 0, 1] = [0.25 * area, 0.0, 0.0, 0.0]
    g.create_dataset("gamma_targets", data=a)
    g["gamma_targets"].attrs["channels"] = "A,P_crofton,B0,B1,chi_exact"
    lst = tmp_path / "train.txt"
    lst.write_text("t,0,0,5.0,0\nt,0,0,3.0,1\nt,0,0,0.0,2\nt,0,0,0.05,3\n")
    cfg = {"TRAIN_METADATA_FILE": str(lst), "PREPROCESSED_DATA_DIR": str(tmp_path),
           "PATCH_SIZE": 16, "PIXEL_SIZE_KM": 2.0}
    p = train_exceedance_fractions(cfg, [0.1, 1.0])
    # half the draws are wet patches (mean fraction 0.75, 0.125), half dry (0, 0)
    assert np.allclose(p, [0.375, 0.0625])


# ---- pinball ------------------------------------------------------------------------------
def test_pinball_minimiser_is_the_quantile():
    y = torch.from_numpy(np.random.default_rng(0).exponential(size=20000)).float()
    c = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([c], lr=0.05)
    fn = PinballLoss(0.95)
    for _ in range(1500):
        opt.zero_grad(); fn(c.expand_as(y), y).backward(); opt.step()
    assert abs(float(c.detach()) - np.quantile(y.numpy(), 0.95)) < 0.05


# ---- windowed spectral --------------------------------------------------------------------
def test_spectral_window_ignores_edge_spike():
    g = torch.Generator().manual_seed(3)
    smooth = torch.nn.functional.avg_pool2d(torch.rand(2, 1, 64, 64, generator=g), 9, 1, 4)
    target = smooth + 0.2 * torch.rand(2, 1, 64, 64, generator=g)     # textured
    pred = smooth + 0.1                                               # smooth: lacks HF power
    spiked = pred.clone()                                             # random spikes on two edges
    spiked[..., 0, :] += 0.3 * torch.rand(64, generator=g)
    spiked[..., :, 0] += 0.3 * torch.rand(64, generator=g)
    raw, win = FFT2DKernelCRPSLoss(), FFT2DKernelCRPSLoss(window="hann")
    assert float(raw(target, target)) < 1e-6 and float(win(target, target)) < 1e-6
    assert float(raw(spiked, target)) < float(raw(pred, target))      # the hack works unwindowed
    assert torch.allclose(win(spiked, target), win(pred, target))     # and is invisible windowed


def test_spectral_default_unchanged():
    a, b = torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
    assert torch.equal(FFT2DKernelCRPSLoss()(a, b), FFT2DKernelCRPSLoss(window="none")(a, b))


# ---- soft FSS ------------------------------------------------------------------------------
def test_fss_zero_at_target():
    y = _rain(4)
    assert float(SoftFSSLoss(U, temperature=1e-3)(y, y)) < 1e-3


def test_fss_tends_to_evaluation_fss():
    y, p = _rain(5), _rain(6)
    wins = (5, 11)
    acc = FSSAccumulator(U[:3], wins)
    acc.update(p[:, 0].numpy(), y[:, 0].numpy())
    ref = 1.0 - np.mean(list(acc.result().values()))
    got = float(SoftFSSLoss(U[:3], wins, temperature=1e-5)(p, y))
    assert abs(got - ref) < 1e-4


def test_fss_gradient_finite():
    y, p = _rain(7), _rain(8).requires_grad_()
    SoftFSSLoss(U)(p, y).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


# ---- persistent homology -------------------------------------------------------------------
def test_ph_diagram_indices_are_c_order():
    f = np.zeros((5, 7)); f[1, 5] = 3.0; f[3, 2] = 2.0
    h0, _ = _diagram(f, 0.0)
    births = {int(i) for i in h0[:, 0]}
    assert births == {np.ravel_multi_index((1, 5), f.shape), np.ravel_multi_index((3, 2), f.shape)}


def test_ph_zero_for_identical_and_finite_gradient():
    y = _rain(9, b=2, h=32, w=32)
    fn = PersistentHomologyLoss()
    assert float(fn(y, y)) == 0.0
    p = (y * 0.5 + _rain(10, b=2, h=32, w=32) * 0.5).requires_grad_()
    loss = fn(p, y)
    loss.backward()
    assert float(loss) > 0 and torch.isfinite(p.grad).all()
    assert 0 < int((p.grad != 0).sum()) < p.numel() // 4      # only critical pixels move


def test_ph_missing_peak_costs_and_descends():
    t = torch.zeros(1, 1, 16, 16); t[0, 0, 4, 4] = 20.0; t[0, 0, 11, 11] = 10.0
    p = torch.zeros(1, 1, 16, 16); p[0, 0, 4, 4] = 20.0; p[0, 0, 11, 11] = 1.0
    p.requires_grad_()
    fn = PersistentHomologyLoss(min_persistence=0.0)
    l0 = fn(p, t)
    l0.backward()
    assert float(p.grad[0, 0, 11, 11]) < 0                    # raise the missing peak
    with torch.no_grad():
        p2 = p - 1.0 * p.grad
    assert float(fn(p2, t)) < float(l0)
