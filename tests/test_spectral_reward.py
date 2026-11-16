"""
Tests for the spectral energy-score reward (REWARD_MODE: spectral) in
src/trainers/reward_finetune.py. No data or GPU needed.
"""
import torch

from src.losses.competing import FFT2DKernelCRPSLoss
from src.trainers.reward_finetune import minkowski_reward, spectral_energy_reward

N = 32


def _cell(n=N):
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    d2 = (x - n / 2) ** 2 + (y - n / 2) ** 2
    return 20.0 * torch.exp(-d2 / (2 * 5.0 ** 2))                       # mm/h


def _draw(s, g, n=N):
    """Smooth cell plus multiplicative noise of amplitude s: one member of the law L_s."""
    return (_cell(n) * torch.exp(s * torch.randn(n, n, generator=g))).view(1, 1, n, n)


def test_single_member_equals_competing_loss_in_log1p_space():
    g = torch.Generator().manual_seed(0)
    p, t = _draw(0.5, g), _draw(0.5, g)
    ref = FFT2DKernelCRPSLoss()(torch.log1p(p), torch.log1p(t))
    assert torch.allclose(spectral_energy_reward(p.unsqueeze(0), t), ref, atol=1e-6)


def test_fair_normalisation_against_class_m2():
    # the class divides the spread term by M^2, the reward by M(M-1)
    g = torch.Generator().manual_seed(1)
    M = 4
    ens = torch.cat([_draw(0.5, g) for _ in range(M)], 0).unsqueeze(1)   # [M,1,1,H,W]
    t = _draw(0.5, g)
    lp = torch.log1p(ens).permute(1, 0, 2, 3, 4)                          # [B=1,M,1,H,W]
    biased = FFT2DKernelCRPSLoss()(lp, torch.log1p(t))
    S = torch.abs(torch.fft.rfft2(torch.log1p(ens)[:, 0, 0], norm="ortho")).flatten(1)
    T = torch.abs(torch.fft.rfft2(torch.log1p(t)[0, 0], norm="ortho")).flatten()
    t1 = (S - T).abs().mean(-1).mean()
    pair_sum = (S[:, None] - S[None]).abs().mean(-1).sum()
    assert torch.allclose(biased, t1 - 0.5 * pair_sum / M**2, atol=1e-5)
    assert torch.allclose(spectral_energy_reward(ens, t), t1 - 0.5 * pair_sum / (M * (M - 1)),
                          atol=1e-5)


def _expected_score(s, s0, M, reps=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(reps):
        t = _draw(s0, g)
        ens = torch.cat([_draw(s, g) for _ in range(M)], 0).unsqueeze(1)
        tot += float(spectral_energy_reward(ens, t))
    return tot / reps


def test_ensemble_score_is_minimised_by_the_true_law():
    # propriety on a synthetic law: the ensemble drawn with the target's own noise level
    # scores lower than both a too-smooth and a too-noisy ensemble
    s0 = 0.5
    sc = {s: _expected_score(s, s0, M=4) for s in (0.0, s0, 2 * s0)}
    assert sc[s0] < sc[0.0] and sc[s0] < sc[2 * s0], sc


def test_dry_patch_gives_finite_gradient():
    ens = torch.zeros(2, 1, 1, N, N, requires_grad=True)
    t = _draw(0.5, torch.Generator().manual_seed(2))
    spectral_energy_reward(ens, t).backward()
    assert torch.isfinite(ens.grad).all()


def test_dispatch_through_minkowski_reward():
    g = torch.Generator().manual_seed(3)
    ens = torch.cat([_draw(0.5, g) for _ in range(2)], 0).unsqueeze(1)
    t = _draw(0.5, g)
    r = minkowski_reward(None, ens, None, "spectral", 0.05, 0.5, "threshold", t)
    assert torch.allclose(r, spectral_energy_reward(ens, t))


# ---- the other competing losses as ensemble rewards ------------------------------------
import numpy as np
import pytest
from src.losses.competing import ImageSSIMLoss, OpticalFlowConsistencyLoss, WeightedSoftWetAreaLoss
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.trainers import reward_finetune as rf

MAXV = 6.4205


def _ens(M, s=0.5, seed=4):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([_draw(s, g) for _ in range(M)], 0).unsqueeze(1), _draw(s, g)


def test_wetarea_single_member_is_the_l1_soft_mask_distance():
    ens, t = _ens(1)
    s = lambda x: torch.sigmoid((x - 0.1) / 0.05)
    ref = (s(ens[0]) - s(t)).abs().mean()
    assert torch.allclose(rf.wetarea_energy_reward(ens, t), ref, atol=1e-6)
    # same mask as the U-Net loss, which squares the difference where the reward takes |.|
    u = WeightedSoftWetAreaLoss(threshold=0.1, temperature=0.05)(ens[0], t)
    assert torch.allclose(((s(ens[0]) - s(t)) ** 2).mean(), u, atol=1e-7)


def test_wetarea_ensemble_score_is_minimised_by_the_true_law():
    def score(s, s0=1.0, M=4, reps=100):
        g = torch.Generator().manual_seed(0)
        tot = 0.0
        for _ in range(reps):
            # small cell, so the 0.1 mm/h contour moves with the noise
            t = (_draw(s0, g) * 0.02)
            ens = torch.cat([_draw(s, g) * 0.02 for _ in range(M)], 0).unsqueeze(1)
            tot += float(rf.wetarea_energy_reward(ens, t))
        return tot / reps
    sc = {s: score(s) for s in (0.0, 1.0, 2.0)}
    assert sc[1.0] < sc[0.0] and sc[1.0] < sc[2.0], sc


def test_ssim_single_member_equals_imagessimloss_in_norm_space():
    ens, t = _ens(1)
    ref = ImageSSIMLoss()(torch.log1p(ens[0]) / MAXV, torch.log1p(t) / MAXV)
    assert torch.allclose(rf.ssim_energy_reward(ens, t, max_val=MAXV), ref, atol=1e-6)


def test_ssim_spread_term_uses_fair_normalisation():
    M = 3
    ens, t = _ens(M)
    f, tt = torch.log1p(ens) / MAXV, torch.log1p(t) / MAXV
    L = ImageSSIMLoss()
    t1 = sum(L(f[m], tt) for m in range(M)) / M
    ordered = sum(L(f[m], f[k]) for m in range(M) for k in range(M) if m != k)
    assert torch.allclose(rf.ssim_energy_reward(ens, t, max_val=MAXV),
                          t1 - 0.5 * ordered / (M * (M - 1)), atol=1e-6)


def test_opticalflow_single_member_equals_l1_consistency_loss():
    ens, t = _ens(1)
    ref_field = torch.roll(t, shifts=(3, -2), dims=(-2, -1))       # a displaced "coarse input"
    u = OpticalFlowConsistencyLoss(x_dim=N, y_dim=N, loss_type="l1")(ens[0], t, reference=ref_field)
    r = rf.opticalflow_energy_reward(ens, t, reference=ref_field)
    assert torch.allclose(r, u, atol=1e-6)


@pytest.mark.parametrize("mode", ["spectral", "wetarea", "ssim", "opticalflow"])
def test_pixel_modes_dispatch_and_give_finite_gradients(mode):
    ens, t = _ens(3)
    ens = ens.clone().requires_grad_(True)
    ref_field = torch.roll(t, shifts=(3, -2), dims=(-2, -1))
    r = minkowski_reward(None, ens, None, mode, 0.05, 0.5, "threshold", t, MAXV, ref_field)
    r.backward()
    assert torch.isfinite(r) and torch.isfinite(ens.grad).all() and ens.grad.abs().sum() > 0


def test_area_only_channel_weights_keep_only_the_area_channel():
    U = np.array([0.1, 0.5, 2.0, 8.0, 31.7], np.float32)
    g_all = AnalyticalMinkowskiLoss(physical_thresholds=U, area_mode="ste")
    g_area = AnalyticalMinkowskiLoss(physical_thresholds=U, area_mode="ste", channel_weights=[1.0, 0.0, 0.0])
    resid = torch.rand(4, 3, len(U), generator=torch.Generator().manual_seed(5))
    only_area = resid.clone(); only_area[:, 1:] = 0.0
    assert torch.allclose(rf._integrate_xi(g_area, resid), rf._integrate_xi(g_all, only_area), atol=1e-7)
    assert not torch.allclose(rf._integrate_xi(g_area, resid), rf._integrate_xi(g_all, resid))
