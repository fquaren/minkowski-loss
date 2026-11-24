"""Known-answer checks for scripts/analysis/area_elasticity.py.

Pure area change (wet fraction proportional to c, fixed intensity law): a_A(u) = 1 at every u,
alpha = 1, beta = 0. Pure intensity rescaling (fixed wet fraction, intensities scaled by c):
every percentile scales with c, so a_q(s) = 1, and the fit gives alpha ~ 0, beta ~ 1.
"""
import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "area_elasticity",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis", "area_elasticity.py"))
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)

U = np.array([0.1, 0.192, 0.314, 0.509, 0.828, 1.35, 2.11, 3.43, 5.36, 8.61, 13.5, 20.2, 31.7,
              50.1, 79.0])
S_LEVELS = np.logspace(-2, -4, 5)


def _lognormal_sf(x, sigma=1.2):
    from math import erfc, sqrt
    return np.array([0.5 * erfc(np.log(v) / (sigma * sqrt(2))) for v in x])


def _run(w_of_c, s_of_c):
    rng = np.random.default_rng(0)
    c = np.exp(rng.uniform(np.log(0.5), np.log(1.5), 20000))  # narrow: local slopes
    area = np.stack([ae.N_PIX * w_of_c(ci) * _lognormal_sf(U / s_of_c(ci)) for ci in c])
    groups = rng.integers(0, 50, len(c))
    edges = np.logspace(np.log10(0.5), np.log10(1.5), 9)
    n, cs, a = ae.bin_stats(c, area, groups, edges)
    return ae.elasticities(n.sum(0), cs.sum(0), a.sum(0), U, S_LEVELS, min_pix=1)


def test_pure_area_change():
    e = _run(lambda c: 0.1 * c, lambda c: 3.0)
    m = np.isfinite(e["aA"])
    assert m.sum() >= 10
    np.testing.assert_allclose(e["aA"][m], 1.0, atol=0.02)
    assert abs(e["alpha"] - 1) < 0.02 and abs(e["beta"]) < 0.02


def test_pure_intensity_rescaling():
    e = _run(lambda c: 0.1, lambda c: 3.0 * c)
    m = np.isfinite(e["aq"])
    assert m.sum() >= 3
    np.testing.assert_allclose(e["aq"][m], 1.0, atol=0.03)
    assert abs(e["alpha"]) < 0.1 and abs(e["beta"] - 1) < 0.1
