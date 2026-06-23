"""
Tests for src/data/gamma.py.

Verifies topological computations against fields with analytically
known Minkowski functionals.
"""

import numpy as np
import pytest
from src.data.gamma import (
    compute_persistence_diagram,
    compute_gamma_matrix,
    select_topology_target,
)


def _make_disk(size=64, center=(32, 32), radius=10, intensity=5.0):
    """Create a field with a single circular region above background."""
    y, x = np.mgrid[:size, :size]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    field = np.zeros((size, size), dtype=np.float64)
    field[dist <= radius] = intensity
    return field


def _make_annulus(size=64, center=(32, 32), r_inner=5, r_outer=15, intensity=5.0):
    """Create a field with a ring (one component, one hole)."""
    y, x = np.mgrid[:size, :size]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    field = np.zeros((size, size), dtype=np.float64)
    field[(dist >= r_inner) & (dist <= r_outer)] = intensity
    return field


class TestPersistenceDiagram:
    """Basic sanity checks for GUDHI integration."""

    def test_empty_field(self):
        field = np.zeros((32, 32))
        pairs = compute_persistence_diagram(field)
        # Should have at least the essential H0 feature
        assert len(pairs) >= 1

    def test_single_peak(self):
        field = _make_disk(size=32, center=(16, 16), radius=5, intensity=10.0)
        pairs = compute_persistence_diagram(field)
        # The peak creates one significant H0 feature
        h0_pairs = [p for p in pairs if p[0] == 0]
        assert len(h0_pairs) >= 1

    def test_nan_handling(self):
        """NaN values should be treated as zero (background)."""
        field = _make_disk(size=32, center=(16, 16), radius=5, intensity=10.0)
        field_nan = field.copy()
        field_nan[0, 0] = np.nan
        pairs_clean = compute_persistence_diagram(field)
        pairs_nan = compute_persistence_diagram(field_nan)
        # Should produce identical results since field[0,0] was already 0
        assert len(pairs_clean) == len(pairs_nan)


class TestGammaMatrix:
    """Verify gamma matrix computation on fields with known topology."""

    def test_single_disk_b0(self):
        """Single disk above threshold → B0=1, B1=0."""
        field = _make_disk(size=64, center=(32, 32), radius=10, intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0, thresh_b0=0.01, thresh_b1=0.01
        )

        assert gamma.shape == (5, 1)
        assert gamma[0, 0] > 0, "Area should be positive"
        assert gamma[1, 0] > 0, "Perimeter should be positive"
        assert gamma[2, 0] == 1, f"Expected B0=1, got {gamma[2, 0]}"
        assert gamma[3, 0] == 0, f"Expected B1=0, got {gamma[3, 0]}"
        assert gamma[4, 0] == 1, f"Expected chi_exact=0, got {gamma[4, 0]}"

    def test_two_disks_b0(self):
        """Two separated disks → B0=2."""
        field = np.zeros((64, 64), dtype=np.float64)
        y, x = np.mgrid[:64, :64]
        field[((x - 16) ** 2 + (y - 32) ** 2) <= 64] = 5.0
        field[((x - 48) ** 2 + (y - 32) ** 2) <= 64] = 5.0

        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0, thresh_b0=0.01, thresh_b1=0.01
        )
        assert gamma[2, 0] == 2, f"Expected B0=2, got {gamma[2, 0]}"

    def test_annulus_b1(self):
        """Annulus → B0=1, B1=1 (one component with one hole)."""
        field = _make_annulus(
            size=64, center=(32, 32), r_inner=5, r_outer=15, intensity=5.0
        )
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0, thresh_b0=0.01, thresh_b1=0.01
        )
        assert gamma[2, 0] == 1, f"Expected B0=1, got {gamma[2, 0]}"
        assert gamma[3, 0] == 1, f"Expected B1=1, got {gamma[3, 0]}"
        assert gamma[4, 0] == 0, f"Expected chi_exact=0, got {gamma[4, 0]}"

    def test_empty_field_all_zero(self):
        """Zero field → all zeros except possibly background at low threshold."""
        field = np.zeros((32, 32))
        thresholds = np.array([0.5, 1.0, 5.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=2.0, thresh_b0=0.01, thresh_b1=0.01
        )
        np.testing.assert_array_equal(gamma[0, :], 0.0)  # no area
        np.testing.assert_array_equal(gamma[1, :], 0.0)  # no perimeter
        np.testing.assert_array_equal(gamma[4, :], 0.0)  # no chi_exact
    def test_area_monotonicity(self):
        """Area must be non-increasing with increasing threshold."""
        field = np.random.default_rng(42).exponential(2.0, size=(64, 64))
        thresholds = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=2.0, thresh_b0=0.1, thresh_b1=0.1
        )
        area = gamma[0, :]
        assert np.all(np.diff(area) <= 0), "Area must be non-increasing"

    def test_area_physical_units(self):
        """Verify area is in km² given pixel_size_km."""
        field = np.ones((10, 10)) * 5.0
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=2.0, thresh_b0=0.01, thresh_b1=0.01
        )
        expected_area = 10 * 10 * 4.0  # 100 pixels × 4 km²/pixel
        assert gamma[0, 0] == expected_area

    def test_persistence_filtering(self):
        """High persistence threshold should filter out small features."""
        field = np.zeros((64, 64))
        field[10:12, 10:12] = 0.5  # small, low-intensity feature
        field[30:50, 30:50] = 10.0  # large, high-intensity feature

        thresholds = np.array([0.1], dtype=np.float32)
        gamma_lax = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0, thresh_b0=0.01, thresh_b1=0.01
        )
        gamma_strict = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0, thresh_b0=5.0, thresh_b1=5.0
        )

        assert (
            gamma_lax[2, 0] >= gamma_strict[2, 0]
        ), "Stricter threshold should yield fewer components"

    def test_output_shape_and_dtype(self):
        field = np.random.default_rng(0).uniform(0, 10, (64, 64))
        thresholds = np.linspace(0.5, 5.0, 20).astype(np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=2.0, thresh_b0=0.1, thresh_b1=0.1
        )
        assert gamma.shape == (5, 20)
        assert gamma.dtype == np.float32


class TestSelectTopologyTarget:
    """Verify the 4→3 channel conversion."""

    def test_euler_mode(self):
        gamma_4 = np.array(
            [
                [100, 50, 10],  # A
                [40, 20, 5],  # P
                [3, 2, 1],  # B0
                [1, 0, 0],  # B1
                [2, 2, 1],  # chi_exact
            ],
            dtype=np.float32,
        )
        gamma_3 = select_topology_target(gamma_4, mode="euler")
        assert gamma_3.shape == (3, 3)
        np.testing.assert_array_equal(gamma_3[2, :], [2, 2, 1])  # B0-B1

    def test_b0_mode(self):
        gamma_4 = np.array(
            [
                [100, 50, 10],
                [40, 20, 5],
                [3, 2, 1],
                [1, 0, 0],
                [2, 2, 1],
            ],
            dtype=np.float32,
        )
        gamma_3 = select_topology_target(gamma_4, mode="b0")
        assert gamma_3.shape == (3, 3)
        np.testing.assert_array_equal(gamma_3[2, :], [3, 2, 1])  # B0 only

    def test_batched(self):
        gamma_4 = np.random.rand(10, 4, 21).astype(np.float32)
        gamma_3 = select_topology_target(gamma_4, mode="euler")
        assert gamma_3.shape == (10, 3, 21)

    def test_invalid_mode(self):
        gamma_4 = np.zeros((4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown topology mode"):
            select_topology_target(gamma_4, mode="invalid")


class TestIsoperimetricInequality:
    """Verify that computed perimeters satisfy P >= 2*sqrt(pi*A)."""

    def test_disk_isoperimetric(self):
        """A disk should be close to the isoperimetric bound."""
        field = _make_disk(size=128, center=(64, 64), radius=30, intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0, thresh_b0=0.01, thresh_b1=0.01
        )
        A, P = gamma[0, 0], gamma[1, 0]
        P_min = 2.0 * np.sqrt(np.pi * A)
        assert (
            P >= P_min * 0.95
        ), f"Perimeter {P:.1f} violates isoperimetric bound {P_min:.1f}"
