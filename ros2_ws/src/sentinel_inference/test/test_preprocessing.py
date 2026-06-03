"""Hardware/torch-free tests for 4-band assembly. Run anywhere with NumPy."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinel_inference.preprocessing import assemble_four_band, resize_hw


def test_resize_shape_2d_and_3d():
    assert resize_hw(np.zeros((100, 50)), 224).shape == (224, 224)
    assert resize_hw(np.zeros((100, 50, 3)), 224).shape == (224, 224, 3)


def test_resize_identity_when_same_size():
    img = np.random.rand(224, 224, 3).astype(np.float32)
    assert np.allclose(resize_hw(img, 224), img)


def test_resize_preserves_constant_value():
    img = np.full((640, 480, 3), 0.42, dtype=np.float32)
    out = resize_hw(img, 224)
    assert np.allclose(out, 0.42, atol=1e-5)


def test_assemble_shape_and_order_rgb_only():
    rgb = np.zeros((648, 1152, 3), dtype=np.uint8)
    rgb[..., 0] = 10   # B
    rgb[..., 1] = 20   # G
    rgb[..., 2] = 30   # R
    chw, present = assemble_four_band(rgb, None)
    assert chw.shape == (4, 224, 224)
    assert chw.dtype == np.float32
    assert present is False
    # channel means follow [B, G, R, NIR] order, scaled /255
    assert abs(chw[0].mean() - 10 / 255) < 1e-3
    assert abs(chw[1].mean() - 20 / 255) < 1e-3
    assert abs(chw[2].mean() - 30 / 255) < 1e-3
    assert chw[3].mean() == 0.0  # NIR zero-filled


def test_assemble_with_nir_present():
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    nir = np.full((480, 640), 200, dtype=np.uint8)
    chw, present = assemble_four_band(rgb, nir)
    assert present is True
    assert abs(chw[3].mean() - 200 / 255) < 1e-3


def test_assemble_nir_2d_or_3d_equivalent():
    rgb = np.full((100, 100, 3), 50, dtype=np.uint8)
    nir2 = np.full((100, 100), 90, dtype=np.uint8)
    nir3 = nir2[..., None]
    a, _ = assemble_four_band(rgb, nir2)
    b, _ = assemble_four_band(rgb, nir3)
    assert np.allclose(a, b)


def test_input_scale_applies():
    rgb = np.full((50, 50, 3), 255, dtype=np.uint8)
    chw, _ = assemble_four_band(rgb, None, input_scale=2.0)
    assert abs(chw[0].max() - 2.0) < 1e-4


def test_bad_rgb_shape_raises():
    try:
        assemble_four_band(np.zeros((10, 10)), None)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-3-channel RGB")
