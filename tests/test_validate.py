"""The stage validators (scripts/validate): pass on a consistent synthetic store, and catch each
kind of corruption they exist for."""

import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "scripts", "validate"), REPO]

import gamma as vg  # noqa: E402
import store as vs  # noqa: E402
from common import write_report  # noqa: E402

THR = [0.1, 1.0, 5.0, 20.0]


def _make(tmp_path, n=6, seed=0):
    import zarr
    from src.data.gamma import compute_gamma_matrix
    from src.data.preprocessing import coarsen_and_interpolate
    rng = np.random.default_rng(seed)
    root = tmp_path / "store"
    z = zarr.open(str(root / "preprocessed_dataset.zarr"), mode="w")
    meta = tmp_path / "meta"
    meta.mkdir()
    cfg = {"PREPROCESSED_DATA_DIR": str(root), "DOWNSCALING_FACTOR": 12.5, "PIXEL_SIZE_KM": 2.0,
           "PHYSICAL_THRESHOLDS": THR, "PERSISTENCE_THRESHOLD": 0.5}
    train_max = 0.0
    for g, key, stem in (("train", "TRAIN_METADATA_FILE", "full_train"), ("validation", "VAL_METADATA_FILE", "full_val"),
                         ("test", "TEST_METADATA_FILE", "full_test")):
        t = np.maximum(rng.gamma(0.5, 4.0, (n, 128, 128)).astype(np.float32) - 1.0, 0)
        cs, its = zip(*[coarsen_and_interpolate(x, 12.5) for x in t])
        G = z.create_group(g)
        G["original_precip"] = t
        G["coarse_precip"] = np.stack(cs)
        G["interpolated_precip"] = np.stack(its)
        G["max_rate"] = t * 1.0
        G["n_wet"] = rng.integers(0, 5, t.shape).astype(np.uint8)
        G["gamma_targets"] = np.stack([compute_gamma_matrix(x, np.array(THR, np.float32), 2.0, 0.5, 0.5) for x in t])
        lines = [f"20200101{h:02d}0000,{128 * h},256,{t[h].max():.4f}" for h in range(n)]
        (meta / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        cfg[key] = str(meta / f"{stem}.txt")
        if g == "train":
            train_max = float(np.round(t.max(), 4))
    np.save(root / "log_precip_max_val.npy", np.float32(np.log1p(train_max)))
    np.save(root / "physical_thresholds.npy", np.array(THR, np.float32))
    return cfg, z


class A:
    n_sample, n_days, workers, quality = 50, 0, 1, None


def _status(checks, prefix):
    return {c.status for c in checks if c.name.startswith(prefix)}


def test_store_checks_pass_on_a_consistent_store(tmp_path):
    cfg, _ = _make(tmp_path)
    for fn in (vs.c_structure, vs.c_scaler):
        assert {c.status for c in fn(cfg)} == {"PASS"}
    assert {c.status for c in vs.c_metadata(cfg, 50)} == {"PASS"}
    assert {c.status for c in vs.c_coarse(cfg, 50)} == {"PASS"}


def test_store_checks_catch_corruption(tmp_path):
    cfg, z = _make(tmp_path)
    t = z["test"]["original_precip"][2]
    t[5, 5] = t.max() + 10.0                         # a value the metadata and coarse field don't know
    z["test"]["original_precip"][2] = t
    assert "FAIL" in _status(vs.c_metadata(cfg, 50), "metadata/test")
    assert "FAIL" in _status(vs.c_coarse(cfg, 50), "coarse/test")
    np.save(os.path.join(cfg["PREPROCESSED_DATA_DIR"], "log_precip_max_val.npy"), np.float32(1.0))
    assert {c.status for c in vs.c_scaler(cfg)} == {"FAIL"}


def test_gamma_checks_pass_then_catch_corruption(tmp_path):
    cfg, z = _make(tmp_path)
    assert {c.status for c in vg.run(cfg, A())} == {"PASS"}
    g = z["train"]["gamma_targets"][1]
    z["train"]["gamma_targets"][1] = np.zeros_like(g)          # an unwritten row on a wet patch
    assert "FAIL" in _status(vg.c_invariants(cfg, 50), "invariants/train")
    assert "FAIL" in _status(vg.c_recompute(cfg, 50), "recompute/train")
    np.save(os.path.join(cfg["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"), np.array([0.1, 0.17, 0.28, 0.48], np.float32))
    assert "FAIL" in _status(vg.c_setup(cfg), "setup/thresholds")


def test_missing_gamma_is_a_fail_not_a_skip(tmp_path):
    cfg, z = _make(tmp_path)
    del z["validation"]["gamma_targets"]
    assert "FAIL" in _status(vg.c_shape(cfg), "shape/validation")


@pytest.mark.skipif(not os.path.exists("/home/fquareng/work/data/extremes/OPERA/europe_dem_laea.tif"), reason="DEM not on disk")
def test_dem_orientation_on_the_real_dem():
    assert {c.status for c in vs.c_dem()} == {"PASS"}


def test_report_marks_failure(tmp_path):
    from common import Check
    path, passed = write_report("demo", [Check("a", "PASS"), Check("b", "FAIL", "broken", {"x": 1.5})], {"k": "v"},
                                out_root=str(tmp_path))
    assert not passed and "**FAIL**" in open(path).read()
