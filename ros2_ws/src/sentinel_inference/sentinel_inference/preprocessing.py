"""Build WaterDroneNet input from camera frames. Pure NumPy — no rclpy, no torch.

WaterDroneNet wants ``[4, 224, 224]`` float32 in **[Blue, Green, Red, NIR]**
channel order. OpenCV/cv_bridge give BGR, so RGB channels map straight through;
NIR is appended as channel 3 (zero-filled and flagged when absent).

Note the domain gap: the model trained on Sentinel-2 surface reflectance, not
8-bit drone pixels. ``input_scale`` is the knob; real drone→satellite
radiometric calibration is a separate task.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DRONE_IMG_SIZE = 224


def resize_hw(img: np.ndarray, size: int = DRONE_IMG_SIZE) -> np.ndarray:
    """Bilinear resize on the first two axes to ``(size, size)``.

    Works for 2-D (H, W) and 3-D (H, W, C) arrays. No OpenCV dependency so this
    module imports and tests with NumPy alone.
    """
    in_h, in_w = img.shape[:2]
    img = img.astype(np.float32)
    if in_h == size and in_w == size:
        return img
    ys = np.clip((np.arange(size) + 0.5) * in_h / size - 0.5, 0, in_h - 1)
    xs = np.clip((np.arange(size) + 0.5) * in_w / size - 0.5, 0, in_w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)
    wx = (xs - x0)
    if img.ndim == 3:
        wy = wy[:, None, None]
        wx = wx[None, :, None]
    else:
        wy = wy[:, None]
        wx = wx[None, :]
    Ia = img[y0][:, x0]
    Ib = img[y0][:, x1]
    Ic = img[y1][:, x0]
    Id = img[y1][:, x1]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy


def assemble_four_band(
    rgb_bgr: np.ndarray,
    nir: Optional[np.ndarray] = None,
    size: int = DRONE_IMG_SIZE,
    input_scale: float = 1.0,
) -> Tuple[np.ndarray, bool]:
    """Assemble a model-ready 4-band tensor from camera frames.

    Args:
        rgb_bgr: ``(H, W, 3)`` BGR frame (uint8 or float).
        nir: ``(H, W)`` or ``(H, W, 1)`` NIR frame, or None.
        size: output spatial size (224).
        input_scale: multiplies the [0, 1]-scaled pixels (domain-gap knob).

    Returns:
        ``(chw, nir_present)`` where ``chw`` is float32 ``(4, size, size)`` in
        [B, G, R, NIR] order, scaled to roughly [0, input_scale].
    """
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError(f"rgb_bgr must be (H, W, 3); got {rgb_bgr.shape}")

    rgb = resize_hw(rgb_bgr, size)  # (size, size, 3) = B, G, R

    if nir is None:
        nir_r = np.zeros((size, size), dtype=np.float32)
        present = False
    else:
        nir2 = nir if nir.ndim == 2 else nir[..., 0]
        nir_r = resize_hw(nir2, size)
        present = True

    four = np.concatenate([rgb, nir_r[..., None]], axis=-1)  # (size, size, 4)
    four = four.astype(np.float32) / 255.0 * float(input_scale)
    chw = np.transpose(four, (2, 0, 1))  # (4, size, size)
    return np.ascontiguousarray(chw, dtype=np.float32), present
