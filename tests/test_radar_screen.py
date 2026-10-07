"""src/data/radar_screen.py: ceilings, repeated values and radar-wide failures are caught;
real-looking rain is not (DECISIONS §19)."""

import numpy as np
import pandas as pd

from src.data import geo
from src.data.cleaning import boundary_fill
from src.data.radar_screen import (FEATURES, CeilingTable, RadarGeometry, ceiling_candidates, dbz,
                                   isolated_mask, radar_frame_features, repair_ceiling,
                                   repeated_value)

LADDER = 1.0593                              # NIMBUS: constant ratio between value levels


def rain_tile(seed=0, n=128):
    """A heavy convective tile on the 0.01 mm/h grid: smooth cells, no repeated values."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((n, n))
    z = np.zeros((n, n))
    for _ in range(6):
        cy, cx, s, p = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(4, 12), rng.uniform(40, 200)
        z += p * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    z *= np.exp(rng.normal(0, 0.05, z.shape))
    return np.round(z, 2).astype(np.float32)


def to_ladder(z):
    k = np.round(np.log(np.maximum(z, 0.1) / 0.1) / np.log(LADDER))
    return np.where(z >= 0.1, np.round(0.1 * LADDER ** k, 2), 0.0).astype(np.float32)


def test_dbz_of_the_two_ceilings():
    assert abs(dbz(364.63) - 64.0) < 0.01 and abs(dbz(48.62) - 50.0) < 0.01


def test_real_rain_has_no_repeated_value():
    r = repeated_value(rain_tile())
    assert r["max_rep"] <= 5 and r["rep_count"] < 50


def test_repair_plateau_is_caught():
    z = rain_tile(); z[10:60, 10:60] = 88.27            # a ring-median repair of a big region
    r = repeated_value(z)
    assert r["rep_value"] == 88.27 and r["rep_count"] >= 2500 and r["rep_excess"] > 100


def test_ladder_rain_is_not_a_repeat_but_a_ladder_ceiling_is():
    z = to_ladder(rain_tile(1))
    r = repeated_value(z)
    assert r["rep_count"] >= 20 and r["rep_excess"] < 10       # many per level, like neighbours
    z2 = z.copy(); m = z2 >= 31; z2[m & (np.indices(z.shape).sum(0) % 3 == 0)] = 364.63
    assert repeated_value(z2)["rep_value"] == 364.63 and repeated_value(z2)["rep_excess"] > 10


def test_ceiling_candidates_fine_grid_and_ladder():
    rng = np.random.default_rng(0)
    v = np.round(31 + rng.pareto(2.5, 400_000) * 20, 2); v = v[v <= 500]
    vb, c = np.unique(np.rint(v * 100).astype(np.int64), return_counts=True)
    assert ceiling_candidates(vb, c)[0].size == 0
    c2 = c.copy(); i = np.searchsorted(vb, 36463)
    vb2, c2 = np.insert(vb, i, 36463), np.insert(c2, i, 5000)
    vals, cnt, ratio = ceiling_candidates(vb2, c2)
    assert list(vals) == [364.63] and ratio[0] > 1000
    lv = to_ladder(v)
    lb, lc = np.unique(np.rint(lv * 100).astype(np.int64), return_counts=True)
    assert ceiling_candidates(lb, lc)[0].size == 0


def test_ceiling_candidates_mixed_and_coarse_ladders():
    """A radar-year mixes ladder levels with sparse off-ladder values, and some ladders step
    ~3 dB: neither is a ceiling (the Phase-0 false positives); a ceiling on top of them is."""
    rng = np.random.default_rng(1)
    v = np.round(10 + rng.pareto(2.0, 200_000) * 8, 2); v = v[v <= 500]
    mixed = np.r_[to_ladder(v), np.round(10 + rng.pareto(2.0, 3_000) * 8, 2)]
    coarse = np.round(10 * 1.5 ** np.round(np.log(v / 10) / np.log(1.5)), 2)   # ~2.8 dB steps
    for x in (mixed, coarse[coarse <= 500]):
        vb, c = np.unique(np.rint(x * 100).astype(np.int64), return_counts=True)
        assert ceiling_candidates(vb, c)[0].size == 0
        i = np.searchsorted(vb, 36463)
        vb2, c2 = np.insert(vb, i, 36463), np.insert(c, i, 2000)
        assert list(ceiling_candidates(vb2, c2)[0]) == [364.63]


def test_ceiling_table_masks_only_near_its_radar():
    t = CeilingTable([{"year": 2023, "value": 364.63, "site_row": 100, "site_col": 100}],
                     shape=(600, 600))
    z = np.zeros((600, 600), np.float32); z[110, 110] = 364.63; z[500, 500] = 364.63
    m = t.mask(z, 2023)
    assert m[110, 110] and not m[500, 500] and m.sum() == 1
    assert not t.mask(z, 2022).any()
    assert t.mask(z[100:200, 100:200], 2023, crop=(100, 200, 100, 200))[10, 10]


def test_repair_ceiling_takes_neighbour_median():
    z = np.full((20, 20), 40.0, np.float32); z[10, 10] = 364.63
    m = np.zeros(z.shape, bool); m[10, 10] = True
    ch = repair_ceiling(z, m)
    assert ch[10, 10] and z[10, 10] == 40.0


def test_guard_refuses_large_components_only():
    z = np.full((80, 80), 5.0, np.float32)
    z[5:35, 5:35] = 200.0                                     # 900 px region
    z[60:64, 60:64] = 200.0                                   # 16 px region
    mask = z >= 150
    ref = np.zeros(z.shape, bool)
    ch = boundary_fill(z, mask, max_size=500, refused=ref)
    assert ref[5:35, 5:35].all() and not ref[60:64, 60:64].any()
    assert (z[5:35, 5:35] == 200.0).all() and (z[60:64, 60:64] == 5.0).all()
    assert ch[60:64, 60:64].all() and not ch[5:35, 5:35].any()
    z2 = np.full((80, 80), 5.0, np.float32); z2[5:35, 5:35] = 200.0
    boundary_fill(z2, z2 >= 150)                              # default: v3, repaired
    assert (z2[5:35, 5:35] == 5.0).all()


def _two_radar_geometry(shape=(400, 400)):
    rows = []
    for k, (r, c) in enumerate(((150, 120), (250, 300))):
        x, y = geo.rowcol_to_xy(r, c)
        rows.append({"odim": f"r{k}", "location": f"site{k}", "startyear": None, "finishyear": None,
                     "row": r, "col": c, "x": float(x), "y": float(y), "maxrange_km": 200.0})
    return RadarGeometry(2020, shape=shape, sites=pd.DataFrame(rows))


def test_radar_disk_failure_vs_rain():
    g = _two_radar_geometry()
    rr, cc = np.indices(g.shape)
    sr, sc = g.sites["row"].iat[0], g.sites["col"].iat[0]
    d = np.hypot(rr - sr, cc - sc)
    fail = np.where((g.owner == 0) & (d < 100), 300 * np.exp(-d / 60), 0.0).astype(np.float32)
    f = radar_frame_features(fail, g)
    assert f[0, 7] > 0.9 and f[0, 9] > 1.5                    # range-only, edge at max range
    rain = np.zeros(g.shape, np.float32)
    rng = np.random.default_rng(3)
    for _ in range(60):                                      # widespread cells, any azimuth
        cy, cx, s = rng.uniform(0, 400), rng.uniform(0, 400), rng.uniform(5, 25)
        rain += rng.uniform(5, 60) * np.exp(-((rr - cy) ** 2 + (cc - cx) ** 2) / (2 * s * s))
    f2 = radar_frame_features(rain, g)
    assert f2[0, 1] >= 0.2 and f2[0, 7] < 0.4 and abs(f2[0, 9]) < 1.0 and abs(f2[0, 8]) < 1.0


def test_isolated_mask_cluster_vs_cell_vs_edge():
    """A 3-pixel cluster in dry air is isolated; a real cell's core is not (its window is
    wet), and neither is a pixel whose window is mostly outside coverage."""
    z = np.zeros((60, 60), np.float32)
    z[10, 10:13] = [200.0, 340.0, 150.0]                     # cluster, 5x5 median 0
    yy, xx = np.indices(z.shape)
    z += (60 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * 4.0 ** 2))).astype(np.float32)
    z[:, 55:] = np.nan                                       # no coverage
    z[0, 54] = 80.0                                          # grid corner + no data: 36% covered
    ys, xs = np.nonzero(np.nan_to_num(z) >= 31)
    iso = isolated_mask(z, ys, xs)
    got = {(int(y), int(x)) for y, x in zip(ys[iso], xs[iso])}
    assert got == {(10, 10), (10, 11), (10, 12)}


def test_radar_frame_counts_isolated_pixels_per_radar():
    g = _two_radar_geometry()
    z = np.zeros(g.shape, np.float32)
    ys, xs = np.nonzero(g.owner == 0)
    y0, x0 = int(np.median(ys)), int(np.median(xs))
    z[y0, x0:x0 + 2] = [300.0, 280.0]
    f = radar_frame_features(z, g)
    i31, iiso = FEATURES.index("n31"), FEATURES.index("n_iso31")
    assert f[0, i31] == 2 and f[0, iiso] == 2
    assert np.nan_to_num(f[1, i31]) == 0
