"""
Tests for src/data/gamma.py

Uses synthetic fields with analytically known Minkowski functionals
to verify correctness of the TDA pipeline.
"""

import numpy as np
import pytest

from src.data.gamma import (
    compute_gamma_matrix,
    compute_gamma_matrix_4ch,
    compute_persistence_diagram,
    extract_persistence_pairs,
    count_betti_at_thresholds,
    estimate_persistence_thresholds,
    compute_area_perimeter,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic fields with known topology
# ---------------------------------------------------------------------------

def make_disk(size=64, cx=32, cy=32, radius=10, intensity=5.0):
    """Single disk: 1 CC, 0 holes at any threshold < intensity."""
    y, x = np.mgrid[:size, :size]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    field = np.zeros((size, size), dtype=np.float64)
    field[mask] = intensity
    return field


def make_annulus(size=64, cx=32, cy=32, r_outer=20, r_inner=8, intensity=5.0):
    """Annulus (ring): 1 CC, 1 hole at any threshold < intensity."""
    y, x = np.mgrid[:size, :size]
    outer = (x - cx)**2 + (y - cy)**2 <= r_outer**2
    inner = (x - cx)**2 + (y - cy)**2 <= r_inner**2
    field = np.zeros((size, size), dtype=np.float64)
    field[outer & ~inner] = intensity
    return field


def make_two_disks(size=64, intensity=5.0):
    """Two separated disks: 2 CCs, 0 holes."""
    y, x = np.mgrid[:size, :size]
    d1 = (x - 16)**2 + (y - 32)**2 <= 8**2
    d2 = (x - 48)**2 + (y - 32)**2 <= 8**2
    field = np.zeros((size, size), dtype=np.float64)
    field[d1 | d2] = intensity
    return field


def make_empty():
    """Zero field: 0 CC, 0 holes, 0 area."""
    return np.zeros((64, 64), dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests: persistence diagram extraction
# ---------------------------------------------------------------------------

class TestPersistenceDiagram:
    def test_empty_field(self):
        diagram = compute_persistence_diagram(make_empty())
        pairs_b0 = extract_persistence_pairs(diagram, dim=0)
        # Only the essential (background) feature should exist
        assert pairs_b0.shape[0] <= 1

    def test_single_disk_has_one_finite_b0(self):
        field = make_disk(intensity=5.0)
        diagram = compute_persistence_diagram(field)
        pairs = extract_persistence_pairs(diagram, dim=0)
        # Filter finite pairs with nonzero persistence
        finite = pairs[pairs[:, 3] == 0]
        significant = finite[finite[:, 2] > 0.01]
        # One connected component (the disk) plus possibly the background
        assert significant.shape[0] >= 1

    def test_annulus_has_b1_feature(self):
        field = make_annulus(intensity=5.0)
        diagram = compute_persistence_diagram(field)
        pairs_b1 = extract_persistence_pairs(diagram, dim=1)
        significant = pairs_b1[pairs_b1[:, 2] > 0.1]
        assert significant.shape[0] >= 1, "Annulus should have at least one H1 feature"


# ---------------------------------------------------------------------------
# Tests: gamma matrix computation
# ---------------------------------------------------------------------------

class TestGammaMatrix:
    def test_empty_field_all_zeros(self):
        thresholds = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            make_empty(), thresholds, pixel_size_km=1.0,
            topology_target="b0"
        )
        assert gamma.shape == (3, 3)
        np.testing.assert_array_equal(gamma, 0.0)

    def test_disk_area_is_correct(self):
        """Area should equal number of pixels in disk * pixel_area."""
        radius = 10
        field = make_disk(size=64, radius=radius, intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0,
            topology_target="b0"
        )
        # Area should be approximately pi*r^2 (discrete approximation)
        expected_area = np.sum(field > 1.0) * 1.0  # pixel_size=1
        assert abs(gamma[0, 0] - expected_area) < 1.0

    def test_disk_b0_equals_one(self):
        field = make_disk(intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0,
            thresh_b0=0.01, topology_target="b0"
        )
        assert gamma[2, 0] == 1.0, f"Single disk should have B0=1, got {gamma[2, 0]}"

    def test_two_disks_b0_equals_two(self):
        field = make_two_disks(intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0,
            thresh_b0=0.01, topology_target="b0"
        )
        assert gamma[2, 0] == 2.0, f"Two disks should have B0=2, got {gamma[2, 0]}"

    def test_annulus_euler_is_zero(self):
        """Annulus: 1 CC - 1 hole = 0 Euler."""
        field = make_annulus(intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0,
            thresh_b0=0.01, thresh_b1=0.01,
            topology_target="euler"
        )
        assert gamma[2, 0] == 0.0, (
            f"Annulus should have chi=0 (1 CC - 1 hole), got {gamma[2, 0]}"
        )

    def test_disk_euler_is_one(self):
        """Disk: 1 CC - 0 holes = 1 Euler."""
        field = make_disk(intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0,
            thresh_b0=0.01, thresh_b1=0.01,
            topology_target="euler"
        )
        assert gamma[2, 0] == 1.0, (
            f"Disk should have chi=1 (1 CC - 0 holes), got {gamma[2, 0]}"
        )

    def test_4ch_stores_b0_and_b1_separately(self):
        field = make_annulus(intensity=5.0)
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix_4ch(
            field, thresholds, pixel_size_km=1.0,
            thresh_b0=0.01, thresh_b1=0.01,
        )
        assert gamma.shape == (4, 1)
        # B0 = 1 (one ring), B1 = 1 (one hole)
        assert gamma[2, 0] == 1.0, f"Expected B0=1, got {gamma[2, 0]}"
        assert gamma[3, 0] == 1.0, f"Expected B1=1, got {gamma[3, 0]}"

    def test_area_monotonically_decreasing(self):
        """Area must be non-increasing with increasing threshold."""
        field = make_disk(intensity=5.0)
        thresholds = np.array([0.5, 1.0, 2.0, 4.0, 6.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, pixel_size_km=1.0,
            topology_target="b0"
        )
        area = gamma[0, :]
        assert np.all(np.diff(area) <= 0), (
            f"Area should be non-increasing, got {area}"
        )

    def test_output_shape(self):
        thresholds = np.linspace(0.1, 5.0, 21).astype(np.float32)
        gamma = compute_gamma_matrix(
            make_disk(), thresholds, topology_target="euler"
        )
        assert gamma.shape == (3, 21)

    def test_nan_handling(self):
        """NaN pixels should be treated as background (0.0)."""
        field = make_disk(intensity=5.0)
        field[0:10, 0:10] = np.nan
        thresholds = np.array([1.0], dtype=np.float32)
        gamma = compute_gamma_matrix(
            field, thresholds, topology_target="b0"
        )
        # Should still detect the disk
        assert gamma[2, 0] >= 1.0


# ---------------------------------------------------------------------------
# Tests: persistence threshold estimation
# ---------------------------------------------------------------------------

class TestPersistenceThresholds:
    def test_returns_expected_keys(self):
        fields = np.stack([make_disk(intensity=5.0) for _ in range(5)])
        result = estimate_persistence_thresholds(fields, percentile=90.0)
        assert "thresh_b0" in result
        assert "thresh_b1" in result
        assert "thresh_unified" in result

    def test_unified_is_max(self):
        fields = np.stack([make_annulus(intensity=5.0) for _ in range(10)])
        result = estimate_persistence_thresholds(fields, percentile=90.0)
        assert result["thresh_unified"] == max(
            result["thresh_b0"], result["thresh_b1"]
        )


# ---------------------------------------------------------------------------
# Tests: area and perimeter
# ---------------------------------------------------------------------------

class TestAreaPerimeter:
    def test_perimeter_satisfies_isoperimetric_inequality(self):
        """P^2 >= 4*pi*A for any shape."""
        field = make_disk(intensity=5.0, radius=15)
        thresholds = np.array([1.0], dtype=np.float32)
        area, perimeter = compute_area_perimeter(
            field, thresholds, pixel_size_km=1.0
        )
        # Discrete approximation: allow small tolerance
        assert perimeter[0]**2 >= 4 * np.pi * area[0] * 0.9, (
            f"Isoperimetric violation: P={perimeter[0]:.2f}, A={area[0]:.2f}"
        )
