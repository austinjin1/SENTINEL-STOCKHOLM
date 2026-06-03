"""Hardware-free tests for the camera HAL. Run anywhere: pytest, no ROS, no cam.

    python -m pytest ros2_ws/src/sentinel_camera/test -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinel_camera.backends import SyntheticBackend, make_backend


def test_synthetic_frame_shape_and_dtype():
    be = SyntheticBackend(width=1152, height=648, fps=30.0)
    f = be.render(0)
    assert f.shape == (648, 1152, 3)
    assert f.dtype == np.uint8
    assert 0 <= f.min() and f.max() <= 255


def test_synthetic_is_deterministic():
    a = SyntheticBackend(64, 48, 30.0).render(7)
    b = SyntheticBackend(64, 48, 30.0).render(7)
    assert np.array_equal(a, b)


def test_synthetic_frames_advance():
    a = SyntheticBackend(64, 48, 30.0).render(0)
    b = SyntheticBackend(64, 48, 30.0).render(1)
    assert not np.array_equal(a, b)


def test_frames_iterator_yields_requested_count():
    be = SyntheticBackend(32, 24, 30.0, max_frames=5)
    frames = list(be.frames())
    assert len(frames) == 5
    assert all(f.shape == (24, 32, 3) for f in frames)


def test_make_backend_synthetic_explicit():
    be = make_backend("synthetic", width=320, height=240, fps=15.0)
    assert isinstance(be, SyntheticBackend)
    assert next(be.frames()).shape == (240, 320, 3)


def test_make_backend_unknown_raises():
    try:
        make_backend("nope", width=1, height=1, fps=1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown backend")
