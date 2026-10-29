"""The training validator (scripts/validate/training.py) on a synthetic batch: it passes when
training is wired correctly, and fails on an inactive structural loss, a threshold mismatch and a
bad run log."""

import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "scripts", "validate"), REPO]

import training as vt  # noqa: E402

THR = [0.1, 1.0, 5.0, 20.0]


def _cfg(tmp_path, w=1e-3):
    np.save(tmp_path / "log_precip_max_val.npy", np.float32(np.log1p(100.0)))
    np.save(tmp_path / "physical_thresholds.npy", np.array(THR, np.float32))
    return {"PREPROCESSED_DATA_DIR": str(tmp_path), "PHYSICAL_THRESHOLDS": THR, "TOPOLOGY_MODE": "euler",
            "PIXEL_SIZE_KM": 2.0, "NUM_EPOCHS": 3, "MINKOWSKI_WARMUP_EPOCHS": 5,
            "STRUCTURAL_LOSS": "minkowski", "STRUCTURAL_LOSS_WEIGHTS": {"minkowski": w}}


def _batch(B=2, H=64, seed=0):
    import torch
    from scipy import ndimage
    from src.data.datasets import _gamma_to_log
    from src.data.gamma import compute_gamma_matrix
    rng = np.random.default_rng(seed)
    mv = np.log1p(100.0)
    t = np.stack([np.maximum(ndimage.gaussian_filter(rng.gamma(0.5, 20.0, (H, H)), 2) - 3.0, 0) for _ in range(B)])
    itp = np.stack([ndimage.uniform_filter(x, 12) for x in t])
    X = np.stack([np.log1p(itp) / mv, np.zeros_like(itp)], 1).astype(np.float32)
    Y = (np.log1p(t) / mv)[:, None].astype(np.float32)
    G = np.stack([_gamma_to_log(compute_gamma_matrix(x, np.array(THR, np.float32), 2.0, 0.0, 0.0), "euler") for x in t])
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(G.astype(np.float32))


class A:
    batch, steps, device, log = 2, 80, "cpu", None

    def __init__(self, batch_override=None, log=None):
        self.batch_override, self.log = batch_override, log


def _by(checks):
    return {c.name: c for c in checks}


@pytest.fixture(scope="module")
def batch():
    return _batch()


def test_training_checks_pass_on_a_wired_setup(tmp_path, batch):
    c = _by(vt.run(_cfg(tmp_path), A(batch)))
    for name in ("thresholds", "forward", "gradients", "overfit", "determinism", "checkpoint"):
        assert c[name].status == "PASS", (name, c[name].numbers, c[name].detail)
    assert c["warmup"].status == "INFO" and "NO epoch at full weight" in c["warmup"].detail   # 3 epochs, warm-up 5


def test_inactive_structural_loss_fails_the_gradient_check(tmp_path, batch):
    assert _by(vt.run(_cfg(tmp_path, w=0.0), A(batch)))["gradients"].status == "FAIL"


def test_threshold_mismatch_fails(tmp_path, batch):
    cfg = _cfg(tmp_path)
    np.save(tmp_path / "physical_thresholds.npy", np.array([0.1, 0.17, 0.28, 0.48], np.float32))
    assert vt.c_thresholds(cfg)[0].status == "FAIL"


def test_log_check(tmp_path):
    good = tmp_path / "good.log"
    good.write_text("auxiliary structural loss: minkowski (space=phys)\nstructural weight w_max=0.0001 (from STRUCTURAL_LOSS_WEIGHTS)\n"
                    "Epoch 1 train 0.0123 val 0.0119\n")
    bad = tmp_path / "bad.log"
    bad.write_text(good.read_text() + "Epoch 2 train nan val nan\n")
    assert vt.c_log(str(good))[0].status == "PASS"
    assert vt.c_log(str(bad))[0].status == "FAIL"
