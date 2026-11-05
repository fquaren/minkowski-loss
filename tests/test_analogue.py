"""Peak-normalised shape features (src/evaluation/analogue.py): known answers."""
import numpy as np

from src.evaluation.analogue import shape_features


def _cells(centres, n=128, s=4.0, peak=40.0):
    y, x = np.mgrid[:n, :n]
    return sum(peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2)) for cx, cy in centres)


def test_scaling_the_field_leaves_the_features_unchanged():
    f = _cells([(40, 50), (80, 70), (60, 100)])
    for c in (0.1, 3.0, 25.0):
        np.testing.assert_allclose(shape_features(c * f), shape_features(f), atol=1e-5)


def test_one_core_is_far_from_many_small_cells():
    one = _cells([(64, 64)], s=12.0)
    rng = np.random.default_rng(0)
    many = _cells(rng.uniform(10, 118, (25, 2)), s=2.5)
    near = _cells([(66, 61)], s=11.0)                       # a similar single core, shifted
    d = lambda a, b: float(np.linalg.norm(shape_features(a) - shape_features(b)))
    assert d(one, near) < 0.25 * d(one, many)


def test_dry_field_gives_zeros():
    assert not shape_features(np.zeros((128, 128))).any()
