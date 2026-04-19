"""
Unit tests for preprocessing utilities.
"""

import numpy as np
import pytest

from src.data.preprocessing import (
    apply_window,
    clip_hu,
    hu_to_3channel,
    normalize_image,
)


class TestApplyWindow:
    def test_output_range(self):
        hu = np.linspace(-1000, 3000, 1000)
        result = apply_window(hu, center=40, width=80)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_clipping(self):
        hu = np.array([-9999.0, 40.0, 9999.0])
        result = apply_window(hu, center=40, width=80)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)

    def test_output_dtype(self):
        hu = np.zeros((512, 512), dtype=np.float32)
        result = apply_window(hu, center=40, width=80)
        assert result.dtype == np.float64 or result.dtype == np.float32


class TestHuTo3Channel:
    def test_output_shape(self):
        hu = np.zeros((512, 512), dtype=np.float32)
        result = hu_to_3channel(hu)
        assert result.shape == (512, 512, 3)

    def test_output_range(self):
        hu = np.random.uniform(-1000, 3000, (64, 64)).astype(np.float32)
        result = hu_to_3channel(hu)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        hu = np.zeros((64, 64), dtype=np.float32)
        result = hu_to_3channel(hu)
        assert result.dtype == np.float32


class TestClipHU:
    def test_clip_bounds(self):
        hu = np.array([-2000.0, 0.0, 5000.0])
        result = clip_hu(hu, hu_min=-1000.0, hu_max=3000.0)
        assert result[0] == -1000.0
        assert result[1] == 0.0
        assert result[2] == 3000.0


class TestNormalizeImage:
    def test_shape_preserved(self):
        img = np.random.rand(224, 224, 3).astype(np.float32)
        result = normalize_image(img)
        assert result.shape == img.shape

    def test_normalised_mean_near_zero(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img = np.stack(
            [np.full((64, 64), m, dtype=np.float32) for m in mean], axis=-1
        )
        result = normalize_image(img, mean=mean, std=std)
        assert np.allclose(result, 0.0, atol=1e-5)
