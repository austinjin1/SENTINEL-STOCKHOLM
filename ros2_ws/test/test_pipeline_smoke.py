"""End-to-end data-contract smoke: drone capture -> computer assembly.

No ROS, no torch, no camera. Wires sentinel_camera's synthetic backend straight
into sentinel_inference's 4-band assembly to prove the two packages agree on
shape, dtype, and channel order — the contract DDS carries between machines.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "sentinel_camera"))
sys.path.insert(0, os.path.join(ROOT, "src", "sentinel_inference"))

from sentinel_camera.backends import SyntheticBackend
from sentinel_inference.preprocessing import assemble_four_band


def test_synthetic_frame_assembles_to_model_input():
    cam = SyntheticBackend(width=1152, height=648, fps=30.0)
    frame = cam.render(3)                      # (648, 1152, 3) BGR uint8
    chw, nir_present = assemble_four_band(frame, None)
    assert chw.shape == (4, 224, 224)
    assert chw.dtype == np.float32
    assert nir_present is False                # RGB-only today
    assert 0.0 <= chw.min() and chw.max() <= 1.0


def test_channel_order_survives_capture_to_tensor():
    # Paint distinct B/G/R so we can confirm order end to end.
    cam = SyntheticBackend(8, 8, 30.0)
    frame = cam.render(0).copy()
    frame[..., 0] = 40    # B
    frame[..., 1] = 80    # G
    frame[..., 2] = 120   # R
    chw, _ = assemble_four_band(frame, None)
    assert abs(chw[0].mean() - 40 / 255) < 1e-3   # Blue  -> ch0
    assert abs(chw[1].mean() - 80 / 255) < 1e-3   # Green -> ch1
    assert abs(chw[2].mean() - 120 / 255) < 1e-3  # Red   -> ch2
    assert chw[3].mean() == 0.0                   # NIR   -> ch3 (absent)


def test_streaming_frames_each_assemble():
    cam = SyntheticBackend(320, 240, 30.0, max_frames=4)
    for frame in cam.frames():
        chw, _ = assemble_four_band(frame, None)
        assert chw.shape == (4, 224, 224)
