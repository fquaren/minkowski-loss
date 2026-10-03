"""src/data/cleaning.py: clear errors are repaired or flagged; real storms are left alone."""

import numpy as np

from src.data.cleaning import clean_frame, ray_flag, ring_flag, temporal_support, tile_stats


def blob(h=128, w=128, cy=64, cx=64, peak=100.0, sigma=6.0):
    yy, xx = np.indices((h, w))
    return (peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def test_isolated_spike_is_repaired():
    f = np.zeros((128, 128), np.float32); f[40, 40] = 80.0
    c, info = clean_frame(f)
    assert info["n_spikes_fixed"] == 1 and c[40, 40] < 1.0


def test_real_cell_untouched():
    f = blob(peak=120.0)
    c, info = clean_frame(f)
    assert info["n_spikes_fixed"] == 0
    np.testing.assert_allclose(c[f >= 0.1], f[f >= 0.1])


def test_no_zeroing_above_150():
    f = blob(peak=300.0)
    c, _ = clean_frame(f)
    assert abs(c.max() - 300.0) < 1e-3


def test_static_clutter_pixel_repaired_and_nearby_cell_kept():
    f = np.full((64, 64), 2.0, np.float32) + blob(64, 64, cy=30, cx=45, peak=60.0, sigma=3.0)
    f[30, 15] = 60.0                                   # clutter pixel on a known hot spot
    hot = np.zeros_like(f, bool); hot[30, 15] = True
    c, info = clean_frame(f, hot)
    assert info["n_hot_fixed"] == 1 and c[30, 15] < 5.0
    np.testing.assert_allclose(c[25:36, 40:51], f[25:36, 40:51])   # the real cell untouched


def test_nan_coverage_preserved():
    f = blob(); f[:10] = np.nan
    c, _ = clean_frame(f)
    assert np.isnan(c[:10]).all() and np.isfinite(c[10:]).all()


def test_drizzle_floor():
    f = np.full((32, 32), 0.05, np.float32)
    assert clean_frame(f)[0].max() == 0.0


def test_ray_pointing_at_radar_is_flagged():
    t = np.zeros((128, 128), np.float32); t[64, 10:120] = 5.0; t[63, 10:120] = 5.0   # E-W line
    site_on_axis = np.array([[64.0, 128 + 60.0]])      # radar due east, on the line's axis
    site_off_axis = np.array([[64.0 + 60.0, 64.0]])    # radar due north: line is tangential
    assert ray_flag(t, 0, 0, site_on_axis)
    assert not ray_flag(t, 0, 0, site_off_axis)


def test_tile_stats_keys():
    s = tile_stats(blob(peak=60.0))
    assert s["max"] > 59 and s["n_ge31"] > 0 and s["coarse_max"] > 0


def _arc(shape, cy, cx, r, th0, th1, width=1.0, val=5.0):
    yy, xx = np.indices(shape)
    rr = np.hypot(yy - cy, xx - cx)
    th = np.arctan2(yy - cy, xx - cx)
    return np.where((np.abs(rr - r) <= width) & (th >= th0) & (th <= th1), val, 0.0).astype(np.float32)


def test_ring_about_radar_is_flagged_and_tangential_line_is_not():
    site = np.array([[64.0, -40.0]])                  # radar 80 km west of the tile
    ring = _arc((128, 128), 64, -40, 100, -0.6, 0.6)  # 100 px arc, ~120 px long
    assert ring_flag(ring, 0, 0, site)
    line = np.zeros((128, 128), np.float32); line[10:118, 60:62] = 5.0   # N-S band at x=60
    assert not ring_flag(line, 0, 0, site)            # tangential to the radar, but straight
    assert not ring_flag(blob(peak=20.0, sigma=15.0), 0, 0, site)


def test_temporal_support():
    cur = np.zeros((200, 200), np.float32); cur[100:104, 100:104] = 40.0
    empty = np.zeros_like(cur)
    near = np.zeros_like(cur); near[110:113, 108:111] = 3.0      # ~20 km away: support
    far = np.zeros_like(cur); far[160:163, 160:163] = 3.0        # ~120 km away: none
    assert (temporal_support(cur, empty, empty)[100:104, 100:104] == 1).all()
    assert (temporal_support(cur, near, empty)[100:104, 100:104] == 0).all()
    assert (temporal_support(cur, far, far)[100:104, 100:104] == 1).all()
    assert (temporal_support(cur, None, empty)[100:104, 100:104] == -1).all()
    gap = empty.copy(); gap[90:95, 90:95] = np.nan                # no coverage near the cell
    assert (temporal_support(cur, gap, empty)[100:104, 100:104] == -1).all()
    assert temporal_support(cur, empty, empty)[0, 0] == 0
