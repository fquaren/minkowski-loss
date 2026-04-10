"""
Tests for src/losses/minkowski.py.

Verifies gradient flow, shape consistency, and mathematical properties
of the Minkowski loss variants.
"""

import torch
import numpy as np
import pytest
from src.losses.minkowski import (
    MinkowskiLoss,
    HomoscedasticMinkowskiLoss,
    AnalyticalMinkowskiLoss,
)


@pytest.fixture
def quantile_levels():
    return [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


@pytest.fixture
def batch_data(quantile_levels):
    B, Q = 4, len(quantile_levels)
    pred = torch.randn(B, 3, Q, requires_grad=True)
    target = torch.randn(B, 3, Q)
    return pred, target


class TestMinkowskiLoss:
    def test_output_shapes(self, quantile_levels, batch_data):
        criterion = MinkowskiLoss(quantile_levels)
        pred, target = batch_data
        total, d_a, d_p, d_t = criterion(pred, target)
        assert total.shape == (4,)
        assert d_a.shape == (4,)

    def test_zero_loss_on_identical(self, quantile_levels):
        criterion = MinkowskiLoss(quantile_levels)
        x = torch.randn(2, 3, len(quantile_levels))
        total, _, _, _ = criterion(x, x)
        assert torch.allclose(total, torch.zeros(2), atol=1e-6)

    def test_gradient_flow(self, quantile_levels, batch_data):
        criterion = MinkowskiLoss(quantile_levels)
        pred, target = batch_data
        total, _, _, _ = criterion(pred, target)
        total.sum().backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_symmetric(self, quantile_levels):
        criterion = MinkowskiLoss(quantile_levels)
        a = torch.randn(2, 3, len(quantile_levels))
        b = torch.randn(2, 3, len(quantile_levels))
        t1, _, _, _ = criterion(a, b)
        t2, _, _, _ = criterion(b, a)
        assert torch.allclose(t1, t2, atol=1e-5)

    def test_non_negative(self, quantile_levels, batch_data):
        criterion = MinkowskiLoss(quantile_levels)
        pred, target = batch_data
        total, d_a, d_p, d_t = criterion(pred, target)
        assert torch.all(total >= 0)


class TestHomoscedasticMinkowskiLoss:
    def test_output_shapes(self, quantile_levels, batch_data):
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        pred, target = batch_data
        total, d_a, d_p, d_t, weights = criterion(pred, target)
        assert total.shape == ()  # scalar
        assert weights.shape == (3,)

    def test_initial_equal_weighting(self, quantile_levels):
        """At init, log_vars=0 → all precisions = 1 → equal weights."""
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        assert torch.allclose(
            torch.exp(-criterion.log_vars),
            torch.ones(3),
        )

    def test_log_vars_receive_gradients(self, quantile_levels, batch_data):
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        pred, target = batch_data
        total, _, _, _, _ = criterion(pred, target)
        total.backward()
        assert criterion.log_vars.grad is not None
        assert not torch.all(criterion.log_vars.grad == 0)

    def test_gradient_flows_to_pred(self, quantile_levels, batch_data):
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        pred, target = batch_data
        total, _, _, _, _ = criterion(pred, target)
        total.backward()
        assert pred.grad is not None

    def test_optimizer_integration(self, quantile_levels, batch_data):
        """Ensure log_vars can be included in the optimizer."""
        model = torch.nn.Linear(10, 18)  # dummy
        criterion = HomoscedasticMinkowskiLoss(quantile_levels)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=1e-3
        )
        pred, target = batch_data
        total, _, _, _, _ = criterion(pred.detach().requires_grad_(True), target)
        total.backward()
        optimizer.step()
        # log_vars should have changed from zero
        assert not torch.allclose(criterion.log_vars, torch.zeros(3))


class TestAnalyticalMinkowskiLoss:
    def test_output_is_scalar(self, quantile_levels):
        criterion = AnalyticalMinkowskiLoss(thresholds=quantile_levels)
        pred_phys = torch.rand(2, 1, 32, 32) * 10.0
        target = torch.rand(2, 4, len(quantile_levels))
        loss = criterion(pred_phys, target)
        assert loss.shape == ()

    def test_gradient_flows(self, quantile_levels):
        criterion = AnalyticalMinkowskiLoss(thresholds=quantile_levels)
        pred_phys = torch.rand(2, 1, 32, 32, requires_grad=True) * 10.0
        target = torch.rand(2, 4, len(quantile_levels))
        loss = criterion(pred_phys, target)
        loss.backward()
        assert pred_phys.grad is not None
        assert not torch.all(pred_phys.grad == 0)

    def test_handles_3ch_targets(self, quantile_levels):
        """Should work with pre-combined 3-channel targets."""
        criterion = AnalyticalMinkowskiLoss(thresholds=quantile_levels)
        pred_phys = torch.rand(2, 1, 32, 32) * 10.0
        target = torch.rand(2, 3, len(quantile_levels))
        loss = criterion(pred_phys, target)
        assert loss.shape == ()

    def test_zero_field(self, quantile_levels):
        """Zero input should produce near-zero area/perimeter."""
        criterion = AnalyticalMinkowskiLoss(thresholds=quantile_levels)
        pred_phys = torch.zeros(1, 1, 32, 32)
        target = torch.zeros(1, 3, len(quantile_levels))
        loss = criterion(pred_phys, target)
        assert loss.item() < 1.0  # should be small but not exactly 0 due to sigmoid
