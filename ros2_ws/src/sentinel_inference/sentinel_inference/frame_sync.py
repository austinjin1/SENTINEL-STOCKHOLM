"""Match a NIR frame to an RGB frame by timestamp. Pure Python — no rclpy.

The two cameras publish independently, so the inference node pairs them by
nearest stamp within a slop window. With one camera (NIR absent) the node skips
this entirely and zero-fills NIR.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional, Tuple


class ApproxTimeSync:
    """Ring buffer of stamped NIR frames; query the nearest to an RGB stamp.

    Stamps are floats in seconds (ROS time -> ``stamp.sec + stamp.nanosec*1e-9``).
    """

    def __init__(self, slop_sec: float = 0.05, max_buffer: int = 30):
        self.slop_sec = float(slop_sec)
        self._buf: Deque[Tuple[float, Any]] = deque(maxlen=max_buffer)

    def add(self, stamp: float, frame: Any) -> None:
        self._buf.append((float(stamp), frame))

    def match(self, stamp: float) -> Optional[Any]:
        """Nearest buffered frame within ``slop_sec`` of ``stamp``, else None."""
        if not self._buf:
            return None
        best = None
        best_dt = self.slop_sec
        for s, frame in self._buf:
            dt = abs(s - stamp)
            if dt <= best_dt:
                best_dt = dt
                best = frame
        return best

    def __len__(self) -> int:
        return len(self._buf)
