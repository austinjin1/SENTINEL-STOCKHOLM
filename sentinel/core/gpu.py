# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GPU budget enforcement: at most the GPUs listed in SENTINEL_GPUS (default 0-4) are used.

Each job takes an exclusive ``flock`` on one GPU's lock file and keeps it until the process
exits, so concurrent jobs from different workstreams can never oversubscribe the budget.

Usage as a wrapper (preferred for long training runs)::

    python -m sentinel.core.gpu --name sensor_pretrain -- python scripts/train_sensor.py ...

or in-process::

    from sentinel.core.gpu import acquire_gpu
    gpu = acquire_gpu("sensor_pretrain")   # sets CUDA_VISIBLE_DEVICES

The host is shared with processes outside SENTINEL that do not take these locks, so a GPU is
only handed out when it also has at least ``min_free_mb`` of physically free memory
(``--min-free-mb``; default ``SENTINEL_GPU_MIN_FREE_MB`` or 3000). Otherwise the lock is
released again and the job keeps waiting.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from . import paths

_LOCK_DIR = paths.RUNS_ROOT / ".gpu_locks"


def _parse(txt: str) -> list[int]:
    return [int(g) for g in txt.replace(" ", "").split(",") if g.strip()]


# Hard cap set by the project lead (file .gpu_locks/ALLOWED, e.g. "0,1"); a job's own
# SENTINEL_GPUS can only narrow it. If the job's list has no GPU inside the cap, the cap is used.
_CAP_FILE = _LOCK_DIR / "ALLOWED"
_CAP = _parse(_CAP_FILE.read_text()) if _CAP_FILE.exists() else [0, 1, 2, 3, 4]
ALLOWED = [g for g in _parse(os.environ.get("SENTINEL_GPUS", ",".join(map(str, _CAP)))) if g in _CAP] or _CAP
MIN_FREE_MB = int(os.environ.get("SENTINEL_GPU_MIN_FREE_MB", "3000"))
_held = []  # keep file objects alive for the life of the process


def _try_lock(gpu: int, name: str):
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(_LOCK_DIR / f"gpu{gpu}.lock", "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{name} pid={os.getpid()} since={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fh.flush()
    return fh


def free_mb() -> dict[int, int]:
    """Physically free memory per GPU (MiB) from nvidia-smi; {} if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    res = {}
    for line in out.strip().splitlines():
        i, f = (x.strip() for x in line.split(","))
        res[int(i)] = int(f)
    return res


def acquire_gpu(name: str, *, wait: bool = True, poll_s: float = 30.0, prefer: int | None = None,
                min_free_mb: int | None = None) -> int:
    """Lock one allowed GPU with enough free memory, set CUDA_VISIBLE_DEVICES, return its index."""
    need = MIN_FREE_MB if min_free_mb is None else min_free_mb
    order = ALLOWED if prefer not in ALLOWED else [prefer] + [g for g in ALLOWED if g != prefer]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match nvidia-smi indices
    while True:
        free = free_mb() if need > 0 else {}
        for g in order:
            if need > 0 and free and free.get(g, 0) < need:
                continue
            fh = _try_lock(g, name)
            if fh is not None:
                _held.append(fh)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(g)
                return g
        if not wait:
            raise RuntimeError(f"no free GPU among {ALLOWED}")
        time.sleep(poll_s)


def status() -> dict[int, str]:
    out = {}
    free = free_mb()
    for g in ALLOWED:
        fh = _try_lock(g, "status-probe")
        if fh is None:
            p = _LOCK_DIR / f"gpu{g}.lock"
            out[g] = "busy: " + (p.read_text().strip() if p.exists() else "?")
        else:
            out[g] = "free"
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
        if g in free:
            out[g] += f" ({free[g]} MiB physically free)"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="job")
    ap.add_argument("--prefer", type=int, default=None)
    ap.add_argument("--min-free-mb", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.status:
        for g, s in status().items():
            print(f"GPU {g}: {s}")
        return 0
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given")
    g = acquire_gpu(a.name, prefer=a.prefer, min_free_mb=a.min_free_mb)
    print(f"[sentinel.gpu] {a.name} -> GPU {g}", file=sys.stderr, flush=True)
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
