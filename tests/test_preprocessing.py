from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_recoloring.preprocessing import (
    bgr_to_rgb,
    load_lab_pair,
    model_output_to_rgb,
    read_rgb_image,
    rgb_to_bgr,
    rgb_to_model_inputs,
)


def test_lab_shapes_ranges_and_round_trip() -> None:
    generator = np.random.default_rng(7)
    rgb = generator.uniform(0.05, 0.95, size=(16, 24, 3)).astype(np.float32)
    lightness, chroma = rgb_to_model_inputs(rgb)
    restored = model_output_to_rgb(lightness, chroma)
    assert lightness.shape == (16, 24, 1)
    assert chroma.shape == (16, 24, 2)
    assert lightness.dtype == np.float32
    assert chroma.dtype == np.float32
    assert 0.0 <= float(lightness.min()) <= float(lightness.max()) <= 100.0
    assert float(np.max(np.abs(chroma))) <= 1.0
    np.testing.assert_allclose(restored, rgb, atol=2e-4)


def test_grayscale_rgba_and_resize(tmp_path: Path) -> None:
    grayscale = np.arange(20 * 10, dtype=np.uint8).reshape(20, 10)
    gray_path = tmp_path / "gray.png"
    Image.fromarray(grayscale).save(gray_path)
    loaded_gray = read_rgb_image(gray_path)
    assert loaded_gray.shape == (20, 10, 3)
    np.testing.assert_allclose(loaded_gray[..., 0], loaded_gray[..., 1])

    rgba = np.zeros((9, 11, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 128
    rgba_path = tmp_path / "rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(rgba_path)
    loaded_rgba = read_rgb_image(rgba_path)
    assert loaded_rgba.shape == (9, 11, 3)
    assert loaded_rgba.min() >= 0.0 and loaded_rgba.max() <= 1.0

    lightness, chroma = load_lab_pair(rgba_path, (32, 40))
    assert lightness.shape == (32, 40, 1)
    assert chroma.shape == (32, 40, 2)


def test_rgb_bgr_convention_is_explicit() -> None:
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    expected_bgr = np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)
    np.testing.assert_array_equal(rgb_to_bgr(rgb), expected_bgr)
    np.testing.assert_array_equal(bgr_to_rgb(expected_bgr), rgb)


def test_floating_input_outside_unit_range_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rgb_to_model_inputs(np.full((2, 2, 3), 255.0, dtype=np.float32))
