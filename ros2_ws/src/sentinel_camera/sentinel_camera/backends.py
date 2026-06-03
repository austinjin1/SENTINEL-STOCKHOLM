"""Camera backends (HAL) for the SENTINEL drone payload.

Pure capture layer — no ``rclpy`` here, so it tests on any laptop. Each backend
yields BGR ``uint8`` frames of shape ``(H, W, 3)``. Selection order on the Pi is
libcamera → opencv; ``synthetic`` is the no-hardware path for dev/CI.

Mirrors CINE-Sensing's camera_publisher capture, but keeps colour (the water
model needs RGB+NIR, not the grayscale CINE uses for VIO).
"""

from __future__ import annotations

import subprocess
from typing import Iterator, Optional

import numpy as np


class CameraBackend:
    """Interface: a backend produces BGR uint8 frames."""

    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class SyntheticBackend(CameraBackend):
    """Deterministic moving gradient + noise. No hardware, no deps beyond numpy.

    Lets the whole pipeline run and be tested on a box with no camera. Frame
    content is intentionally water-ish (low, slowly varying) so downstream
    assembly/inference see plausible input.
    """

    def __init__(self, width: int, height: int, fps: float, max_frames: Optional[int] = None):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.max_frames = max_frames
        self._n = 0
        yy, xx = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        self._gx = xx / max(self.width - 1, 1)
        self._gy = yy / max(self.height - 1, 1)

    def render(self, n: int) -> np.ndarray:
        """One deterministic frame for step ``n`` (exposed for tests)."""
        phase = (n % 60) / 60.0
        rng = np.random.RandomState(n)  # deterministic per frame
        base = 0.25 + 0.15 * self._gx + 0.10 * np.sin(2 * np.pi * (self._gy + phase))
        noise = rng.normal(0.0, 0.02, size=base.shape).astype(np.float32)
        chan = np.clip(base + noise, 0.0, 1.0)
        # BGR: greenish water, slightly brighter green channel
        bgr = np.stack([chan * 0.7, chan * 1.0, chan * 0.6], axis=-1)
        return (bgr * 255.0).astype(np.uint8)

    def frames(self) -> Iterator[np.ndarray]:
        while self.max_frames is None or self._n < self.max_frames:
            yield self.render(self._n)
            self._n += 1


class OpenCVBackend(CameraBackend):
    """V4L2 / USB camera via OpenCV. Used when libcamera is unavailable."""

    def __init__(self, camera_id: int, width: int, height: int, fps: float):
        import cv2  # imported lazily so the module loads without OpenCV

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera id {camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                yield frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class LibcameraBackend(CameraBackend):
    """Raspberry Pi camera via ``rpicam-vid`` MJPEG stream (CINE-Sensing path)."""

    def __init__(self, width: int, height: int, fps: float, shutter_us: int, gain: float):
        import cv2

        self._cv2 = cv2
        self.proc = subprocess.Popen(
            [
                "rpicam-vid", "--codec", "mjpeg",
                "--width", str(int(width)), "--height", str(int(height)),
                "--framerate", str(int(fps)), "--timeout", "0", "--nopreview",
                "--shutter", str(int(shutter_us)), "--gain", str(float(gain)),
                "--awb", "auto", "--quality", "90", "--output", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8,
        )

    def frames(self) -> Iterator[np.ndarray]:
        cv2 = self._cv2
        buf = b""
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                soi = buf.find(b"\xff\xd8")
                eoi = buf.find(b"\xff\xd9", soi + 2)
                if soi == -1 or eoi == -1:
                    break
                jpg = buf[soi:eoi + 2]
                buf = buf[eoi + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

    def close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def detect_libcamera() -> bool:
    """True if a Pi camera is reachable via libcamera (``rpicam-hello``)."""
    try:
        r = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and "Available cameras" in r.stdout
    except Exception:
        return False


def make_backend(kind: str, *, width: int, height: int, fps: float,
                 camera_id: int = 0, shutter_us: int = 10000,
                 gain: float = 1.0) -> CameraBackend:
    """Build a backend. ``kind`` is auto|libcamera|opencv|synthetic.

    ``auto`` tries libcamera, then opencv, then falls back to synthetic so the
    node always comes up (and logs which path it took).
    """
    kind = (kind or "auto").lower()
    if kind == "synthetic":
        return SyntheticBackend(width, height, fps)
    if kind == "libcamera":
        return LibcameraBackend(width, height, fps, shutter_us, gain)
    if kind == "opencv":
        return OpenCVBackend(camera_id, width, height, fps)
    if kind == "auto":
        if detect_libcamera():
            return LibcameraBackend(width, height, fps, shutter_us, gain)
        try:
            return OpenCVBackend(camera_id, width, height, fps)
        except Exception:
            return SyntheticBackend(width, height, fps)
    raise ValueError(f"unknown backend kind: {kind!r}")
