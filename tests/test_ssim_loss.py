"""SSIM loss known answers (DECISIONS §16 bug: a perfect prediction scored 0.465)."""
import torch

from src.losses.competing import ImageSSIMLoss


def _field(seed, wet=0.2):
    g = torch.Generator().manual_seed(seed)
    f = torch.rand(2, 1, 64, 64, generator=g)
    return f * (torch.rand(2, 1, 64, 64, generator=g) < wet)      # mostly dry, values in [0, 1)


def test_perfect_prediction_scores_zero():
    y = _field(0)
    assert float(ImageSSIMLoss()(y, y)) < 1e-6


def test_all_dry_pair_scores_zero():
    z = torch.zeros(2, 1, 64, 64)
    assert float(ImageSSIMLoss()(z, z)) < 1e-6


def test_missing_rain_is_penalised_and_ordered():
    y = _field(1)
    loss = ImageSSIMLoss()
    zero, half = float(loss(torch.zeros_like(y), y)), float(loss(0.5 * y, y))
    assert 0.0 < half < zero
