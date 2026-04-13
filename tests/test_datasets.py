"""
Tests for src/data/datasets.py helpers and src/utils.py.
"""

import numpy as np
import torch
import pytest

from src.data.datasets import _gamma_to_log
from src.utils import (
    signed_log1p, signed_expm1, DataDenormalizer, load_config,
)


class TestGammaToLog:
    """Test the 4ch → 3ch gamma conversion."""

    def test_b0_mode(self):
        gamma = np.array([
            [100, 50, 10],   # A
            [40, 20, 5],     # P
            [3, 2, 1],       # B0
            [1, 0, 0],       # B1
        ], dtype=np.float32)
        result = _gamma_to_log(gamma, "b0")
        assert result.shape == (3, 3)
        # Third channel should be log1p(B0)
        np.testing.assert_allclose(result[2], np.log1p([3, 2, 1]), rtol=1e-5)

    def test_euler_mode(self):
        gamma = np.array([
            [100, 50, 10],
            [40, 20, 5],
            [3, 2, 1],   # B0
            [1, 0, 0],   # B1
        ], dtype=np.float32)
        result = _gamma_to_log(gamma, "euler")
        assert result.shape == (3, 3)
        # chi = B0 - B1 = [2, 2, 1]
        chi = np.array([2, 2, 1], dtype=np.float32)
        expected = np.sign(chi) * np.log1p(np.abs(chi))
        np.testing.assert_allclose(result[2], expected, rtol=1e-5)

    def test_euler_negative(self):
        """Euler can be negative when B1 > B0."""
        gamma = np.array([
            [100, 50],
            [40, 20],
            [1, 0],     # B0
            [3, 2],     # B1
        ], dtype=np.float32)
        result = _gamma_to_log(gamma, "euler")
        chi = np.array([1 - 3, 0 - 2], dtype=np.float32)  # [-2, -2]
        expected = np.sign(chi) * np.log1p(np.abs(chi))
        np.testing.assert_allclose(result[2], expected, rtol=1e-5)

    def test_batched(self):
        gamma = np.random.rand(8, 4, 21).astype(np.float32)
        result = _gamma_to_log(gamma, "euler")
        assert result.shape == (8, 3, 21)

    def test_invalid_mode(self):
        gamma = np.zeros((4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown topology_mode"):
            _gamma_to_log(gamma, "invalid")


class TestSignedLog1p:
    """Test round-trip of signed_log1p / signed_expm1."""

    def test_numpy_roundtrip(self):
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0, 100.0])
        recovered = signed_expm1(signed_log1p(x))
        np.testing.assert_allclose(recovered, x, rtol=1e-6)

    def test_torch_roundtrip(self):
        x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0, 100.0])
        recovered = signed_expm1(signed_log1p(x))
        torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-6)

    def test_zero(self):
        assert signed_log1p(0.0) == 0.0
        assert signed_log1p(np.float32(0.0)) == 0.0

    def test_sign_preservation(self):
        assert signed_log1p(-3.0) < 0
        assert signed_log1p(3.0) > 0

    def test_torch_gradient(self):
        x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        y = signed_log1p(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestDataDenormalizer:
    def test_roundtrip_numpy(self):
        """log1p → normalize → denormalize should recover original."""
        max_val = 5.0
        d = DataDenormalizer(max_val)
        x_phys = np.array([0.0, 0.1, 1.0, 10.0, 50.0])
        x_norm = np.clip(np.log1p(x_phys) / max_val, 0.0, 1.0)
        recovered = d.to_physical_np(x_norm)
        np.testing.assert_allclose(recovered, x_phys, rtol=1e-4)

    def test_roundtrip_torch(self):
        max_val = 5.0
        d = DataDenormalizer(max_val)
        x_phys = torch.tensor([0.0, 0.1, 1.0, 10.0])
        x_norm = torch.clamp(torch.log1p(x_phys) / max_val, 0.0, 1.0)
        recovered = d.to_physical_torch(x_norm)
        torch.testing.assert_close(recovered, x_phys, rtol=1e-4, atol=1e-5)

    def test_nonnegative(self):
        d = DataDenormalizer(5.0)
        x = np.array([-0.1, 0.0, 0.5])
        result = d.to_physical_np(x)
        assert np.all(result >= 0)

    def test_diffusion_domain(self):
        """[-1, 1] → physical should work."""
        d = DataDenormalizer(5.0)
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = d.to_physical_from_diffusion(x)
        assert result.shape == (3,)
        assert torch.all(result >= 0)


class TestConfigValidation:
    def test_missing_key_raises(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("PATCH_SIZE: 128\n")
        with pytest.raises(KeyError, match="DOWNSCALING_FACTOR"):
            load_config(str(config_file))

    def test_wrong_type_raises(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(
            "PATCH_SIZE: 'not_an_int'\n"
            "DOWNSCALING_FACTOR: 12.5\n"
            "QUANTILE_LEVELS: [0.1]\n"
            "PREPROCESSED_DATA_DIR: '/tmp'\n"
            "PIXEL_SIZE_KM: 2.0\n"
            "DRIZZLE_THRESHOLD: 0.1\n"
        )
        with pytest.raises(TypeError, match="PATCH_SIZE"):
            load_config(str(config_file))

    def test_valid_config(self, tmp_path):
        config_file = tmp_path / "good.yaml"
        config_file.write_text(
            "PATCH_SIZE: 128\n"
            "DOWNSCALING_FACTOR: 12.5\n"
            "QUANTILE_LEVELS: [0.1, 0.5, 0.9]\n"
            "PREPROCESSED_DATA_DIR: '/tmp'\n"
            "PIXEL_SIZE_KM: 2.0\n"
            "DRIZZLE_THRESHOLD: 0.1\n"
        )
        config = load_config(str(config_file))
        assert config["PATCH_SIZE"] == 128
