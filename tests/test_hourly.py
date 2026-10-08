"""src/data/hourly.py and the hourly-chain rules (DECISIONS §21-§23): the hour follows OPERA's
ACRR convention, persistent artefacts are lowered while moving rain is kept, and every new rule
leaves the v3/v4 behaviour unchanged when it is off."""

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from src.data import cleaning as C
from src.data.hourly import HourlyDay, local_peaks, median5
from src.data.radar_screen import FEATURES, frame_flagged, frame_flagged_temporal, repeat_context


def blob(h, w, cy, cx, peak, sigma):
    yy, xx = np.indices((h, w))
    return (peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))).astype(np.float32)


# ---------------------------------------------------------------- frame-level pieces
def test_median5_matches_ndimage():
    rng = np.random.default_rng(0)
    z = np.where(rng.random((60, 50)) < 0.4, rng.gamma(0.6, 5, (60, 50)), 0).astype(np.float32)
    m = z >= 1
    np.testing.assert_allclose(median5(z, m)[m], ndimage.median_filter(z, size=5)[m])


def test_local_peak_point_vs_smooth_cell():
    z = blob(64, 64, 32, 32, 40.0, 6.0)
    z[10, 10] = 30.0
    p, _ = local_peaks(z, 1.0, 2.0)
    assert p[10, 10] and not p[28:37, 28:37].any()


def test_hot_fallback_keep_leaves_wide_band():
    z = np.full((40, 40), 20.0, np.float32)
    hot = np.zeros(z.shape, bool); hot[:, 10:20] = True          # 10 px wide hot band
    zero, _ = C.clean_frame(z, hot)                                # v3: centre of the band -> 0
    keep, info = C.clean_frame(z, hot, hot_fallback="keep")
    assert zero[20, 15] == 0.0
    assert keep[20, 15] == 20.0 and info["hot_kept"][20, 15]
    assert keep[20, 11] == 20.0                                    # edge: median of clean neighbours


def test_clean_frame_default_unchanged():
    rng = np.random.default_rng(1)
    z = rng.gamma(0.5, 6, (50, 50)).astype(np.float32)
    hot = rng.random(z.shape) < 0.05
    a, _ = C.clean_frame(z, hot)
    b, _ = C.clean_frame(z, hot, hot_fallback="zero", spikes=True)
    np.testing.assert_array_equal(a, b)


def test_anchored_lines_ray_vs_front_vs_offset_line():
    H = W = 200
    site = np.array([[100.0, 0.0]])                                 # radar on the west edge
    ray = np.zeros((H, W), np.float32); ray[99:101, 20:180] = 5.0   # 2 px wide, through the site
    front = np.zeros((H, W), np.float32); front[90:110, 20:180] = 5.0   # 20 px wide band, same axis
    off = np.zeros((H, W), np.float32); off[159:161, 20:180] = 5.0  # thin, parallel, 60 px off the site
    assert C.anchored_lines(ray, site)[100, 100]
    assert not C.anchored_lines(front, site).any()
    assert not C.anchored_lines(off, site).any()


def test_repeat_context_plateau_vs_speckle():
    t = np.full((32, 32), 40.0, np.float32); t[10:14, 10:14] = 44.12    # plateau in rain at 40
    assert repeat_context(t, 44.12) > 0.5
    s = np.full((32, 32), 3.0, np.float32); s[5, 5] = s[20, 9] = s[25, 25] = 364.63   # speckle
    assert repeat_context(s, 364.63) < 0.1


def _feats(wet_series, f31=0.3, r2=0.7):
    T = len(wet_series)
    f = np.zeros((T, 1, len(FEATURES)), np.float32)
    f[:, 0, FEATURES.index("n_valid")] = 5000
    f[:, 0, FEATURES.index("wet")] = wet_series
    f[:, 0, FEATURES.index("f10")] = 0.3
    f[:, 0, FEATURES.index("f31")] = f31
    f[:, 0, FEATURES.index("r2_range")] = r2
    return f


def test_frame_flag_temporal_keeps_switch_on_and_lasting_failures_drops_storm():
    q = {"radar_frame": {"min_valid": 500, "f31_min": 0.1, "r2_min": 0.5, "jump_min": 2.0,
                         "f10_min": 0.2, "temporal": {"min_wet_jump": 0.15}}}
    flash = _feats([0.0, 0.35, 0.0])                     # a failure switching on for one frame
    storm = _feats([0.5, 0.6, 0.62])                     # rain matching its neighbours, flagged
    storm[[0, 2], 0, FEATURES.index("f31")] = 0.05       # in one frame only (all 4 gallery storms)
    assert frame_flagged_temporal(flash, q)[1, 0]
    assert frame_flagged(storm, q)[1, 0] and not frame_flagged_temporal(storm, q)[1, 0]
    lasting = _feats([0.99, 0.99, 0.99])                 # Madrid: flagged in a run
    assert frame_flagged_temporal(lasting, q)[:, 0].all()
    q2 = {"radar_frame": {k: v for k, v in q["radar_frame"].items() if k != "temporal"}}
    np.testing.assert_array_equal(frame_flagged_temporal(storm, q2), frame_flagged(storm, q2))


# ---------------------------------------------------------------- end to end on a synthetic store
H, W = 96, 96
SITE = np.array([[48.0, 0.0]])


def _write_day(root, day, frames):
    import xarray as xr
    t0 = pd.Timestamp(day)
    times = pd.date_range(t0, periods=len(frames), freq="15min")
    ds = xr.Dataset({"TOT_PREC": (("time", "y", "x"), np.stack(frames).astype(np.float32))},
                    coords={"time": times, "y": np.arange(H), "x": np.arange(W)})
    ds.to_zarr(str(root / t0.strftime("%Y%m%d")), mode="w", consolidated=True)


def _synthetic_days(tmp_path, ray=True):
    """Two days of 96 frames: background drizzle-free rain band moving east, a stationary
    clutter point at (20, 70), a moving convective cell, and (optionally) a 2-px ray from the
    radar at (48, 0) present in every frame."""
    out = {}
    for d, day in enumerate(("20200101", "20200102")):
        frames = []
        for k in range(96):
            g = d * 96 + k
            f = np.zeros((H, W), np.float32)
            f += blob(H, W, 75, (g * 3) % W, 8.0, 5.0)              # moving cell
            f[20, 70] = 30.0; f[20, 71] = 6.0                        # stationary clutter point
            if ray:
                f[47:49, 5:90] = np.maximum(f[47:49, 5:90], 4.0)    # persistent ray
            frames.append(f)
        _write_day(tmp_path, day, frames)
        out[day] = frames
    return out


def _run(tmp_path, rules=None):
    cfg = {"rules": rules or {}}
    return HourlyDay(str(tmp_path), "20200101", "TOT_PREC", hot=lambda y: None, sites_rc=SITE,
                     cfg=cfg).run()


def test_hours_follow_acrr_convention(tmp_path):
    days = _synthetic_days(tmp_path)
    off = {r: False for r in ("hot", "spike", "ceiling", "footprint", "ring", "ray", "persist", "unsupported")}
    hd = _run(tmp_path, off)
    assert len(hd.hours) == 24
    ends = [pd.Timestamp(h["end"]) for h in hd.hours]
    assert ends[0] == pd.Timestamp("2020-01-01 01:00") and ends[-1] == pd.Timestamp("2020-01-02 00:00")
    # hour ending 01:00 = 0.25 x (00:15 + 00:30 + 00:45 + 01:00)
    f = days["20200101"]
    want = 0.25 * (f[1] + f[2] + f[3] + f[4])
    floor = lambda a: np.where(a < C.DRIZZLE, 0.0, a)              # clean_frame's drizzle floor stays on
    np.testing.assert_allclose(hd.hours[0]["sum"], 0.25 * sum(floor(f[k]) for k in (1, 2, 3, 4)), rtol=1e-6)
    np.testing.assert_allclose(hd.hours[0]["raw_sum"], want, rtol=1e-6)
    # the last hour uses the next day's 00:00 frame
    nxt = days["20200102"][0]
    want_last = 0.25 * (f[93] + f[94] + f[95] + nxt)
    np.testing.assert_allclose(hd.hours[-1]["raw_sum"], want_last, rtol=1e-6)


def test_persistent_point_lowered_moving_cell_kept(tmp_path):
    _synthetic_days(tmp_path, ray=False)
    on = _run(tmp_path, {"ray": False})
    off = _run(tmp_path, {"ray": False, "persist": False})
    assert on.signals["n_peak"][20, 70] >= 90 and on.signals["flag_p1"][20, 70]
    assert (on.code[:, 20, 70] & C.REPAIR_PERSIST).all()
    assert on.hours[5]["sum"][20, 70] < 0.2 * off.hours[5]["sum"][20, 70]
    cell = on.signals["n_peak"][60:90, :]
    assert cell.max() < 24                                      # the moving cell is never flagged
    np.testing.assert_allclose(on.hours[5]["sum"][60:90], off.hours[5]["sum"][60:90])


def test_persistent_ray_repaired(tmp_path):
    _synthetic_days(tmp_path, ray=True)
    on = _run(tmp_path, {"persist": False})
    off = _run(tmp_path, {"persist": False, "ray": False})
    assert (on.code[:, 47, 40] & C.REPAIR_RAY).all()
    assert on.hours[3]["sum"][47, 40] < 0.25 * off.hours[3]["sum"][47, 40]


def test_missing_frame_drops_only_its_hour(tmp_path):
    import shutil
    import xarray as xr
    _synthetic_days(tmp_path, ray=False)
    p = tmp_path / "20200101"
    ds = xr.open_zarr(str(p)).load()
    ds = ds.drop_sel(time=pd.Timestamp("2020-01-01 05:30"))
    shutil.rmtree(p)
    ds.to_zarr(str(p), mode="w", consolidated=True)
    hd = _run(tmp_path)
    ends = {pd.Timestamp(h["end"]) for h in hd.hours}
    assert len(hd.hours) == 23 and pd.Timestamp("2020-01-01 06:00") not in ends
