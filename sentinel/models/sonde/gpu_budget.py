# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M1 GPU budget = the lead's cap file (lead ruling 2026-11-10, ``docs/plans/sota_campaign.md`` "GPU rule update").

``runner.M1_GPUS`` is ``(0,)`` in the frozen ``runner.py`` (0444, hashed by FREEZE_ew/FREEZE_hydromet), so it is not
edited. :func:`widen` replaces it, in this process only, with the cap from ``sentinel_runs/.gpu_locks/ALLOWED`` (now
0–9), and raises the physically-free-memory threshold so that a GPU in use by another user's process is never taken.
GPUs are still locked one at a time through ``sentinel.core.gpu`` (flock + free-memory check); call before
``runner.setup_gpu`` and before CUDA initialises.
"""

from __future__ import annotations

import os

DEFAULT_MIN_FREE_MB = 10000   # 2080 Ti = 11019 MiB: only an essentially idle GPU qualifies


def widen(min_free_mb: int | None = None) -> tuple[int, ...]:
    from sentinel.core import gpu as gpu_mod

    from . import runner

    env = os.environ.get("SENTINEL_GPUS")
    cap = tuple(gpu_mod._CAP)
    gpus = tuple(g for g in (int(x) for x in env.split(",") if x.strip()) if g in cap) if env else cap
    runner.M1_GPUS = gpus or cap
    gpu_mod.ALLOWED = list(runner.M1_GPUS)
    gpu_mod.MIN_FREE_MB = int(min_free_mb if min_free_mb is not None
                              else os.environ.get("SENTINEL_GPU_MIN_FREE_MB", DEFAULT_MIN_FREE_MB))
    return runner.M1_GPUS
