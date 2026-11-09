# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Canonical filesystem locations.

Large data and run artifacts live OUTSIDE the git repository. Override with
environment variables:

    SENTINEL_DATA   raw / interim / processed datasets  (default ~/STOCKHOLM/sentinel_data)
    SENTINEL_RUNS   checkpoints, predictions, run logs   (default ~/STOCKHOLM/sentinel_runs)
    SENTINEL_LEGACY read-only v1 Hugging Face mirror     (default ~/STOCKHOLM/SENTINEL-HF)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_HOME_STOCKHOLM = Path.home() / "STOCKHOLM"

DATA_ROOT = Path(os.environ.get("SENTINEL_DATA", _HOME_STOCKHOLM / "sentinel_data"))
RUNS_ROOT = Path(os.environ.get("SENTINEL_RUNS", _HOME_STOCKHOLM / "sentinel_runs"))
LEGACY_ROOT = Path(os.environ.get("SENTINEL_LEGACY", _HOME_STOCKHOLM / "SENTINEL-HF"))

RAW = DATA_ROOT / "raw"
INTERIM = DATA_ROOT / "interim"
PROCESSED = DATA_ROOT / "processed"
MANIFESTS = DATA_ROOT / "manifests"

RESULTS_DIR = REPO_ROOT / "results"
REGISTRY_PATH = RESULTS_DIR / "registry.jsonl"
FIGURES_DIR = REPO_ROOT / "figures"


def ensure(path: Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw(source: str) -> Path:
    return ensure(RAW / source)


def interim(source: str) -> Path:
    return ensure(INTERIM / source)


def processed(name: str) -> Path:
    return ensure(PROCESSED / name)


def run_dir(workstream: str, run_name: str) -> Path:
    return ensure(RUNS_ROOT / workstream / run_name)
