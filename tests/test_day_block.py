"""Day-block recomputation of the extreme-table columns (src/evaluation/day_block.py)."""

import numpy as np
import pytest
from scipy.stats import genpareto

from src.evaluation.day_block import DayRecord, gpd_fit_weighted, parse_leave_out
from src.evaluation.tail_metrics import FSSAccumulator


def _fields(rng, B=12, H=32):
    obs = rng.gamma(0.4, 8.0, (B, H, H))
    pred = obs * rng.lognormal(0.0, 0.3, obs.shape)
    return pred, obs


def test_fss_per_sample_sums_equal_totals():
    rng = np.random.default_rng(0)
    pred, obs = _fields(rng)
    a, b = FSSAccumulator([5, 20], [1, 5]), FSSAccumulator([5, 20], [1, 5])
    per = a.update(pred, obs, per_sample=True)
    assert b.update(pred, obs) is None                      # default unchanged
    for k, (num, den) in per.items():
        assert np.isclose(num.sum(), a._num[k]) and np.isclose(den.sum(), a._den[k])
    assert a.result() == b.result()


def _record(rng, n_batches=5, B=12):
    fss = FSSAccumulator([5, 20], [1, 5])
    rec = DayRecord(list(fss._num), pot_u=20.0)
    days = []
    for i in range(n_batches):
        pred, obs = _fields(rng, B)
        per = fss.update(pred, obs, per_sample=True)
        mse = ((pred - obs) ** 2).mean(axis=(1, 2))
        rec.update(per, mse, obs.max(axis=(1, 2)), pred, obs)
        days += [f"2021060{i + 1}"] * B
    rec.finalize(np.array(days), ext_thr=40.0, lam_targets=(10.0, 100.0))
    return fss, rec


def test_unit_weights_reproduce_fss_and_counts():
    fss, rec = _record(np.random.default_rng(1))
    t = rec.table(np.ones(rec.D))
    for k, v in fss.result().items():
        assert np.isclose(t[k], v)
    n_p, n_o = len(rec.exc["pred"][1]), len(rec.exc["obs"][1])
    assert np.isclose(t["gpd_exc_count_ratio"], n_p / n_o)


def test_leave_out_nothing_equals_full_and_dropping_changes_counts():
    _, rec = _record(np.random.default_rng(2))
    full = rec.table(np.ones(rec.D))
    same = rec.leave_out(set())
    assert same["n_days_dropped"] == 0
    assert all(np.isclose(same[k], full[k], equal_nan=True) for k in full)
    one = rec.leave_out({"20210601"})
    assert one["n_days_dropped"] == 1


def test_intervals_bracket_a_stable_statistic():
    _, rec = _record(np.random.default_rng(3), n_batches=8)
    full = rec.table(np.ones(rec.D))
    iv = rec.intervals(60)
    lo, hi = iv["rmse_overall"]
    assert lo <= full["rmse_overall"] <= hi


def test_weighted_gpd_recovers_shape():
    x = genpareto.rvs(0.2, scale=5.0, size=40_000, random_state=np.random.default_rng(4))
    xi, sg = gpd_fit_weighted(x, np.ones_like(x))
    assert abs(xi - 0.2) < 0.03 and abs(sg - 5.0) < 0.3


def test_parse_leave_out_range_and_file(tmp_path):
    f = tmp_path / "days.txt"
    f.write_text("20210621\n20210622\n")
    out = parse_leave_out(["20210616-20210618", f"@{f}"])
    assert out["20210616-20210618"] == {"20210616", "20210617", "20210618"}
    assert out[str(f)] == {"20210621", "20210622"}
