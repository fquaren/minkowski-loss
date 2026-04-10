"""
Tests for src/losses/minkowski.py
"""

import torch
import pytest
from src.losses.minkowski import MinkowskiLoss, HomoscedasticMinkowskiLoss


@pytest.fixture
def quantile_levels():
    return [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


class TestMinkowskiLoss:
    def test_output_shapes(self, quantile_levels):
        criterion = MinkowskiLoss(quantile_levels)
        B, Q = 4, len(quantile_levels)
        pred = torch.randn(B, 3, Q)
        target = torch.randn(B, 3, Q)
        total, d_a, d_p, d_topo = criterion(pred, target)
        assert total.shape == (B,)
        assert d_a.shape == (B,)

    def test_zero_loss_for_identical_inputs(self, quantile_levels):
        criterion = MinkowskiLoss(quantile_levels)
        x = torch.randn(2, 3, len(quantile_levels))
        total, _, _, _ = criterion(x, x)
        assert torch.allclose(total, torch.zeros(2), atol=1e-6)

    def test_gradient_flows(self, quantile_levels):
        criterion = MinkowskiLoss(quantile_levels)
        pred = torch.randn(2, 3, len(quantile_levels), requires_grad=True)
        target = torch.randn(2, 3, len(quantile_levels))
        total, _, _, _ = criterion(pred, target)
        total.sum().backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_symmetry(self, quantile_levels):
        """L(a, b) == L(b, a) since it's an L1 distance."""
        criterion = MinkowskiLoss(quantile_levels)
        a = torch.randn(3, 3, len(quantile_levels))
        b = torch.randn(3, 3, len(quantile_levels))
        t1, _, _, _ = criterion(a, b)
        t2, _, _, _ = criterion(b, a)
        assert torch.allclose(t1, t2, atol=1e-5)


class TestHomoscedasticMinkowskiLoss:
    def test_output_is_scalar(self, quantile_levels):
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        pred = torch.randn(4, 3, len(quantile_levels))
        target = torch.randn(4, 3, len(quantile_levels))
        total, _, _, _ = criterion(pred, target)
        assert total.dim() == 0  # scalar

    def test_log_vars_receive_gradient(self, quantile_levels):
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        pred = torch.randn(4, 3, len(quantile_levels), requires_grad=True)
        target = torch.randn(4, 3, len(quantile_levels))
        total, _, _, _ = criterion(pred, target)
        total.backward()
        assert criterion.log_vars.grad is not None
        assert not torch.all(criterion.log_vars.grad == 0)

    def test_initial_weights_are_equal(self, quantile_levels):
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        weights = criterion.get_learned_weights()
        assert abs(weights["weight_A"] - weights["weight_P"]) < 1e-6
        assert abs(weights["weight_A"] - weights["weight_topo"]) < 1e-6

    def test_parameters_included_in_optimizer(self, quantile_levels):
        """Verify log_vars shows up in criterion.parameters()."""
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        param_list = list(criterion.parameters())
        assert len(param_list) == 1
        assert param_list[0] is criterion.log_vars

    def test_gradient_flows_through_model_and_loss(self, quantile_levels):
        """Simulate the optimizer setup: model + criterion params."""
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        # Dummy "model"
        model_param = torch.randn(3, 3, len(quantile_levels), requires_grad=True)
        target = torch.randn(3, 3, len(quantile_levels))
        total, _, _, _ = criterion(model_param, target)
        total.backward()
        assert model_param.grad is not None
        assert criterion.log_vars.grad is not None
