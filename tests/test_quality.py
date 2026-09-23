"""
Tests for src/data/quality.py: each artefact feature must separate its synthetic artefact
from a synthetic storm.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.quality import (
    adaptive_block_means, tile_features, pipeline_filter,
)

N = 128


def storm(peak=60.0, sigma=8.0, cy=64, cx=64):
    yy, xx = np.indices((N, N))
    return peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def spike(value=80.0):
    r = np.zeros((N, N))
    r[40, 90] = value
    return r


def spoke():
    r = np.zeros((N, N))
    r[20:110, 60] = 5.0
    return r


class TestSeparation:
    def test_spike_vs_storm(self):
        s, k = tile_features(storm()), tile_features(spike())
        assert k["isolated_frac_ge1"] == 1.0 and s["isolated_frac_ge1"] == 0.0
        assert k["n_spikes"] == 1 and s["n_spikes"] == 0
        assert k["peak_block_ratio"] > 50 > s["peak_block_ratio"]
        assert k["peak_nbr_ratio"] > 100 * s["peak_nbr_ratio"]

    def test_spoke_is_long_and_thin(self):
        f = tile_features(spoke())
        assert f["max_elong"] > 0.95
        assert f["spoke_len"] == pytest.approx(90, rel=0.05)
        assert tile_features(storm())["spoke_len"] == 0.0

    def test_flicker_vs_advection(self):
        r = storm()
        moved = storm(cx=68)                          # 8 km of advection
        f_storm = tile_features(r, prev=moved, nxt=moved)
        f_flick = tile_features(spike(), prev=np.zeros((N, N)), nxt=np.zeros((N, N)))
        assert f_storm["persist_prev"] > 0.8
        assert f_flick["persist_prev"] == 0.0 and f_flick["persist_next"] == 0.0
        assert f_storm["corr_prev"] > 0.8

    def test_declutter_in_cell(self):
        r = storm(peak=200.0, sigma=6.0)
        f = tile_features(r)
        assert f["n_over_declutter"] > 0
        assert f["n_over_declutter_in_cell"] == f["n_over_declutter"]
        assert tile_features(spike(400.0))["n_over_declutter_in_cell"] == 0
        # and the current pipeline really does punch the hole
        assert pipeline_filter(r)[64, 64] == 0.0


class TestPlumbing:
    def test_pool_matches_torch(self):
        a = np.random.default_rng(0).gamma(0.3, 5.0, (N, N))
        t = F.adaptive_avg_pool2d(torch.from_numpy(a)[None, None], (10, 10))[0, 0].numpy()
        np.testing.assert_allclose(adaptive_block_means(a, 10), t, rtol=1e-6)

    def test_nan_coverage(self):
        r = storm()
        r[:, :64] = np.nan
        f = tile_features(r)
        assert f["coverage"] == pytest.approx(0.5)
        assert np.isfinite(f["max"])

    def test_all_nan_tile(self):
        f = tile_features(np.full((N, N), np.nan))
        assert f == {"coverage": 0.0}

    def test_dem_and_qind(self):
        dem = np.full((N, N), 500.0)
        dem[:, 80:] = -1.0
        q = np.full((N, N), 0.9)
        q[40, 90] = 0.1
        f = tile_features(spike(), dem=dem, q=q)
        assert f["argmax_over_sea"] == 1
        assert f["q_at_max"] == pytest.approx(0.1)
        assert f["q_frac_low_ge31"] == 1.0
