"""
Tests for src/data/preprocessing.py and src/trainers/base.py.
"""

import numpy as np
import torch
import pytest
from src.data.preprocessing import filter_precip_bounds, coarsen_and_interpolate
from src.trainers.base import EarlyStopping, cosine_warmup_weight


class TestFilterPrecipBounds:
    def test_zeros_below_drizzle(self):
        arr = np.array([0.05, 0.1, 0.5, 1.0])
        result = filter_precip_bounds(arr.copy(), 0.1, 150.0)
        assert result[0] == 0.0
        assert result[1] == 0.1  # at threshold, kept

    def test_zeros_above_declutter(self):
        arr = np.array([1.0, 100.0, 160.0])
        result = filter_precip_bounds(arr.copy(), 0.1, 150.0)
        assert result[2] == 0.0
        assert result[1] == 100.0

    def test_preserves_valid(self):
        arr = np.array([0.5, 1.0, 50.0, 149.9])
        result = filter_precip_bounds(arr.copy(), 0.1, 150.0)
        np.testing.assert_array_equal(result, arr)


class TestCoarsenAndInterpolate:
    def test_output_shapes(self):
        arr = np.random.rand(128, 128).astype(np.float32)
        coarse, interp = coarsen_and_interpolate(arr, 12.5)
        assert coarse.shape == (10, 10)   # int(128/12.5) = 10
        assert interp.shape == (128, 128)

    def test_nearest_neighbor_preserves_nonnegativity(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        _, interp = coarsen_and_interpolate(arr, 4.0)
        assert np.all(interp >= 0)

    def test_integer_factor_mass_conservation(self):
        """With integer factor, block mean should conserve total sum."""
        arr = np.random.rand(64, 64).astype(np.float32) * 10
        factor = 4.0
        coarse, interp = coarsen_and_interpolate(arr, factor)
        # Interpolated field (nearest-neighbor from block means)
        # should have approximately the same total mass
        np.testing.assert_allclose(
            arr.sum(), interp.sum(), rtol=0.01,
            err_msg="Mass not conserved for integer downscaling factor"
        )


class TestEarlyStopping:
    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=3)
        es(1.0)  # best
        es(1.1)  # worse
        es(1.2)  # worse
        assert es(1.3) is True  # 3rd worse → stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)
        es(0.9)  # improvement resets counter
        assert es(1.0) is False  # only 1 worse after reset
        assert es.counter == 1

    def test_state_dict_roundtrip(self):
        es = EarlyStopping(patience=5)
        es(1.0)
        es(1.1)
        state = es.state_dict()
        es2 = EarlyStopping(patience=1)
        es2.load_state_dict(state)
        assert es2.counter == es.counter
        assert es2.best_score == es.best_score


class TestCosineWarmupWeight:
    def test_zero_during_warmup(self):
        for epoch in range(5):
            w = cosine_warmup_weight(epoch, total_epochs=20, w_max=1.0,
                                      warmup_epochs=5)
            assert w == 0.0, f"Expected 0 during warmup, got {w} at epoch {epoch}"

    def test_reaches_wmax(self):
        w = cosine_warmup_weight(19, total_epochs=20, w_max=1.0, warmup_epochs=0)
        assert w > 0.95, f"Expected near w_max at final epoch, got {w}"

    def test_monotonically_increasing(self):
        weights = [
            cosine_warmup_weight(e, total_epochs=50, w_max=2.0, warmup_epochs=5)
            for e in range(50)
        ]
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1] - 1e-10, \
                f"Non-monotonic at epoch {i}: {weights[i-1]:.4f} → {weights[i]:.4f}"
