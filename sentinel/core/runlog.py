# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Automatic training-run logging (TRAINING_LOG.md + logs/training_runs.jsonl).

Wrap every training run in :class:`TrainingRun`; the entry is written when the run ends,
*including* failed and interrupted runs, so the log cannot silently omit failures::

    with TrainingRun("sensor", "S1-pretrain", config=cfg, data=["usgs_iv_v1"]) as run:
        ...
        run.metric("val_nse_DO", 0.71)
        run.note("early-stopped at epoch 31")
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import socket
import time
import traceback
from pathlib import Path

from . import paths
from .registry import code_state, config_hash

TRAINING_LOG_MD = paths.REPO_ROOT / "TRAINING_LOG.md"
TRAINING_LOG_JSONL = paths.REPO_ROOT / "logs" / "training_runs.jsonl"


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '')})"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _locked_append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(text)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class TrainingRun:
    def __init__(self, workstream: str, experiment: str, *, config: dict | None = None,
                 data: list[str] | None = None, plan_ref: str = "", iteration: str = "base",
                 artifacts_dir: str | None = None):
        self.workstream = workstream
        self.experiment = experiment
        self.config = config or {}
        self.data = data or []
        self.plan_ref = plan_ref
        self.iteration = iteration
        self.artifacts_dir = artifacts_dir
        self.metrics: dict[str, float] = {}
        self.notes: list[str] = []
        self.status = "running"

    def metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def __enter__(self):
        self.t0 = time.time()
        self.start = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        return self

    def __exit__(self, exc_type, exc, tb):
        dur_min = (time.time() - self.t0) / 60
        if exc_type is None:
            self.status = "✅ Completed"
        elif exc_type is KeyboardInterrupt:
            self.status = "⚠️ Interrupted"
        else:
            self.status = "❌ Failed"
            self.notes.append(f"{exc_type.__name__}: {exc}")
            self.notes.append("traceback tail: " + " | ".join(
                traceback.format_exception(exc_type, exc, tb)[-3:]).replace("\n", " ")[:800])
        code = code_state()
        rec = {
            "workstream": self.workstream, "experiment": self.experiment, "start": self.start,
            "duration_min": round(dur_min, 2), "status": self.status, "plan_ref": self.plan_ref,
            "iteration": self.iteration, "config": self.config, "config_hash": config_hash(self.config),
            "data": self.data, "metrics": self.metrics, "notes": self.notes, "host": socket.gethostname(),
            "hardware": _gpu_name(), "artifacts_dir": self.artifacts_dir, "code": code,
        }
        _locked_append(TRAINING_LOG_JSONL, json.dumps(rec, default=str) + "\n")
        cfg_lines = "\n".join(f"  - {k}: {v}" for k, v in sorted(self.config.items()))
        met_lines = "\n".join(f"  - {k}: {v:.6g}" for k, v in sorted(self.metrics.items())) or "  - (none)"
        md = (
            f"\n## {self.workstream} / {self.experiment} — {self.start}\n"
            f"- **Plan ref**: {self.plan_ref or '—'} | **Iteration**: {self.iteration}\n"
            f"- **Status**: {self.status} | **Duration**: {dur_min:.1f} min | **Hardware**: {rec['hardware']}\n"
            f"- **Code**: {code['git_branch']}@{code['git_commit'][:8]}{' (dirty)' if code['git_dirty'] else ''}"
            f", package {code['package_hash']}, script `{Path(code['script'] or '-').name}`\n"
            f"- **Data**: {', '.join(self.data) or '—'}\n"
            f"- **Config** (hash {rec['config_hash']}):\n{cfg_lines or '  - (none)'}\n"
            f"- **Final metrics**:\n{met_lines}\n"
            f"- **Notes**: {' / '.join(self.notes) or '—'}\n"
            f"- **Artifacts**: {self.artifacts_dir or '—'}\n"
        )
        _locked_append(TRAINING_LOG_MD, md)
        return False  # never swallow exceptions
