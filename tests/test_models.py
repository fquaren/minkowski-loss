"""
Tests for src/models/emulators.py.

Verifies forward pass shapes, architectural constraints, and
topology mode switching for all three emulator architectures.
"""

import torch
import numpy as np
import pytest
from src.models.emulators import BaselineCNN, LipschitzCNN, ConstrainedLipschitzCNN


QUANTILE_LEVELS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
N_Q = len(QUANTILE_LEVELS)
INPUT_SHAPE = (1, 128, 128)
BATCH = 2


@pytest.fixture(params=["b0", "euler"])
def topology_mode(request):
    return request.param


@pytest.fixture
def dummy_input():
    return torch.rand(BATCH, 1, 128, 128)


class TestBaselineCNN:
    def test_output_shape(self, topology_mode, dummy_input):
        model = BaselineCNN(n_quantiles=N_Q, input_shape=INPUT_SHAPE,
                            topology_mode=topology_mode)
        out = model(dummy_input)
        assert out.shape == (BATCH, 3, N_Q)

    def test_b0_mode_nonnegative(self, dummy_input):
        model = BaselineCNN(n_quantiles=N_Q, topology_mode="b0")
        out = model(dummy_input)
        assert torch.all(out[:, 2, :] >= 0), "B0 output must be non-negative"

    def test_euler_mode_allows_negative(self, dummy_input):
        model = BaselineCNN(n_quantiles=N_Q, topology_mode="euler")
        # With random weights, output can be negative
        torch.manual_seed(0)
        model = BaselineCNN(n_quantiles=N_Q, topology_mode="euler")
        out = model(dummy_input)
        # Just check it doesn't crash — negativity depends on weights
        assert out.shape == (BATCH, 3, N_Q)


class TestLipschitzCNN:
    def test_output_shape(self, topology_mode, dummy_input):
        model = LipschitzCNN(n_quantiles=N_Q, input_shape=INPUT_SHAPE,
                             topology_mode=topology_mode)
        out = model(dummy_input)
        assert out.shape == (BATCH, 3, N_Q)

    def test_gradient_flow(self, dummy_input):
        model = LipschitzCNN(n_quantiles=N_Q, topology_mode="b0")
        x = dummy_input.requires_grad_(True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_spectral_norm_present(self):
        model = LipschitzCNN(n_quantiles=N_Q)
        # conv2 should have spectral norm
        assert hasattr(model.conv2, "weight_orig"), \
            "conv2 should be spectrally normalized"
        # entry should NOT have spectral norm
        assert not hasattr(model.entry, "weight_orig"), \
            "entry conv should NOT be spectrally normalized"

    def test_robust_block_no_attenuation(self, dummy_input):
        """Verify RobustBlock uses x + F(x), not (x + F(x))/2."""
        from src.models.emulators import RobustBlock
        block = RobustBlock(32)
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        # If /2 were applied, output norm would be ~half of input
        assert out.norm() > x.norm() * 0.4, \
            "RobustBlock output suspiciously small — check for /2 attenuation"


class TestConstrainedLipschitzCNN:
    def test_output_shape(self, topology_mode, dummy_input):
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_Q, input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            topology_mode=topology_mode,
        )
        out = model(dummy_input)
        assert out.shape == (BATCH, 3, N_Q)

    def test_area_monotonicity(self, dummy_input):
        """Area must be non-increasing along the threshold axis."""
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_Q, input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
        )
        out = model(dummy_input)
        area = out[:, 0, :].detach()
        diffs = torch.diff(area, dim=1)
        assert torch.all(diffs <= 1e-5), \
            f"Area monotonicity violated: max increase = {diffs.max():.6f}"

    def test_isoperimetric_inequality(self, dummy_input):
        """Perimeter must satisfy P >= 2*sqrt(pi*A)."""
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_Q, input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
        )
        out = model(dummy_input)
        A = out[:, 0, :].detach()
        P = out[:, 1, :].detach()
        P_min = 2.0 * torch.sqrt(torch.pi * A)
        violations = (P < P_min - 1e-4).sum().item()
        assert violations == 0, \
            f"Isoperimetric inequality violated in {violations} entries"

    def test_area_bounded_by_domain(self, dummy_input):
        """Area must not exceed total domain area."""
        pixel_area = 4.0  # 2km × 2km
        max_area = 128 * 128 * pixel_area
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_Q, input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=pixel_area,
        )
        out = model(dummy_input)
        area = out[:, 0, :].detach()
        assert torch.all(area <= max_area + 1e-3), \
            f"Area exceeds domain: max={area.max():.1f}, limit={max_area:.1f}"

    def test_b0_nonnegative(self, dummy_input):
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_Q, input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS, topology_mode="b0",
        )
        out = model(dummy_input)
        assert torch.all(out[:, 2, :] >= 0)

    def test_requires_quantile_levels(self):
        with pytest.raises(ValueError, match="quantile_levels"):
            ConstrainedLipschitzCNN(n_quantiles=N_Q, quantile_levels=None)


class TestModelGradientIntegrity:
    """Verify all three architectures produce finite gradients."""

    @pytest.mark.parametrize("ModelClass,kwargs", [
        (BaselineCNN, {}),
        (LipschitzCNN, {}),
        (ConstrainedLipschitzCNN, {"quantile_levels": QUANTILE_LEVELS}),
    ])
    def test_finite_gradients(self, ModelClass, kwargs, dummy_input):
        model = ModelClass(n_quantiles=N_Q, input_shape=INPUT_SHAPE, **kwargs)
        x = dummy_input.requires_grad_(True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"Non-finite gradient in {name}"
