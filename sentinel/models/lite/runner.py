# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing for the SENTINEL-Lite drivers: configs, GPU (prefer GPU 2), run dirs, TrainingRun,
registry logging (non-smoke, val only in this phase)."""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths

WORKSTREAM = "M7-lite"
PLAN_REF = "RESEARCH_PLAN.md §10 (G6 SENTINEL-Lite); docs/preregistration/lite_protocol.md"
GPU_PREFER = 2
DATA_MANIFESTS = ("naip_matchups", "lite_labels")


def cpu_threads(n: int = 6) -> None:
    """Shared-host rule (CONTRIBUTING_V2, 2026-10-24 22:30): CPU jobs <= 6 threads, GPU jobs 2."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


def _ancestors(max_depth: int = 8) -> list[int]:
    """This process and its ancestors (Linux /proc), so a GPU locked by a wrapping
    ``python -m sentinel.core.gpu -- bash -c '...'`` queue is reused rather than a second GPU taken."""
    out, pid = [os.getpid()], os.getpid()
    for _ in range(max_depth):
        try:
            stat = open(f"/proc/{pid}/stat").read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
        out.append(pid)
    return out


def setup_gpu(name: str) -> int:
    """Lock one GPU (prefer GPU 2). If this process was started by ``python -m sentinel.core.gpu`` (the parent
    holds the lock of the GPU in CUDA_VISIBLE_DEVICES), reuse that GPU instead of taking a second one."""
    from ...core import gpu as gpu_mod

    if gpu_mod._held:
        raise RuntimeError("a GPU is already held by this process")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.isdigit():
        lock = gpu_mod._LOCK_DIR / f"gpu{cvd}.lock"
        try:
            holder = lock.read_text()
        except OSError:
            holder = ""
        if any(f"pid={a} " in holder for a in _ancestors()):
            return int(cvd)
    return gpu_mod.acquire_gpu(name, prefer=GPU_PREFER if GPU_PREFER in gpu_mod.ALLOWED else None)


def load_config(path: str | Path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text()) or {}
    cfg["_config_file"] = str(path)
    return cfg


def run_directory(name: str, smoke: bool = False) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return paths.run_dir("lite", f"{'smoke/' if smoke else ''}{name}_{stamp}")


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_") or k == "_config_file"}


def training_run(experiment: str, cfg: dict, run_dir: Path, smoke: bool = False):
    from ...core.runlog import TrainingRun

    return TrainingRun(WORKSTREAM, f"SMOKE/{experiment}" if smoke else experiment, config=public_config(cfg),
                       data=list(DATA_MANIFESTS), plan_ref=PLAN_REF,
                       iteration="smoke" if smoke else str(cfg.get("iteration", "base")), artifacts_dir=str(run_dir))


def log_scores(scores: dict, *, model: str, experiment: str, split: str, cfg: dict, artifacts, smoke: bool,
               is_baseline: bool, notes: str = "", extra: dict | None = None) -> list[dict]:
    """Log each target's primary metric (+ secondary metrics in ``extra``) to the registry."""
    if smoke:
        return []
    if split not in ("val",):
        raise ValueError("this phase registers validation results only")
    from ...core.registry import log_result

    out = []
    for target, s in scores.items():
        if not np.isfinite(s.get("value", np.nan)):
            continue
        sec = {k: v for k, v in s.items() if k not in ("value", "ci_low", "ci_high")}
        out.append(log_result(workstream=WORKSTREAM, experiment=experiment, model=model, metric=s["primary"],
                              value=s["value"], split=split, n=s["n"], ci_low=s["ci_low"], ci_high=s["ci_high"],
                              ci_method=f"cluster bootstrap over water bodies (B=2000, {s['n_groups']} groups)",
                              target=target, is_baseline=is_baseline, config=public_config(cfg),
                              data_manifests=list(DATA_MANIFESTS), artifacts=[str(a) for a in artifacts],
                              notes=notes, extra={"secondary": sec, **(extra or {})}))
    return out


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def save_predictions(path, lab: pd.DataFrame, preds: dict[str, np.ndarray], sigmas: dict | None = None) -> None:
    df = lab[["naip_chip_id", "water_body", "site_id", "role", "site_class", "naip_year"]
             + [c for c in lab.columns if c.startswith("y_") and c != "y_chla_class"]].copy()
    for t, p in preds.items():
        df[f"p_{t}"] = p
    for t, s in (sigmas or {}).items():
        df[f"s_{t}"] = s
    df.to_parquet(path, index=False)
