# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Runtime settings for the SENTINEL Console (API + web app).

Data locations come only from :mod:`sentinel.core.paths` and are read at *call* time (not
import time), so a process that changes ``SENTINEL_DATA`` / ``SENTINEL_RUNS`` and reloads
``sentinel.core.paths`` sees the new locations. The Console never creates or writes anything
under the data or runs roots (they are mounted read-only in Docker), so it uses the path
constants (``paths.PROCESSED`` ...) and never ``paths.processed()``, which creates directories.

Resources that live in the repository (results registry, model cards, plugin registry,
RESULTS.md, RESEARCH_PLAN.md, pre-registration docs) default to their repository locations;
tests and deployments may override them with :func:`set_overrides`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from ..core import paths

MODALITIES = ("sonde", "orbit", "lite", "microbe", "chemtox", "bioassess")

# Upload size limits in MiB (configuration constants, not results).
UPLOAD_LIMITS_MIB = {"sonde": 25, "orbit": 64, "lite": 64, "microbe": 64, "chemtox": 2, "bioassess": 16}

USGS_OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v1"
USER_AGENT = "SENTINEL-v2 research (public-data download)"

_OVERRIDES: dict[str, Any] = {}


def set_overrides(**kwargs: Any) -> None:
    """Override Settings fields for every later :meth:`Settings.from_env` call (tests, deployments)."""
    valid = {f.name for f in fields(Settings)}
    bad = set(kwargs) - valid
    if bad:
        raise TypeError(f"unknown Settings fields: {sorted(bad)}")
    _OVERRIDES.update(kwargs)


def clear_overrides() -> None:
    _OVERRIDES.clear()


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Settings:
    data_root: Path
    processed: Path
    raw: Path
    manifests: Path
    runs_root: Path
    registry_path: Path
    repo_root: Path
    model_cards_dir: Path
    models_yaml: Path
    results_md: Path
    research_plan: Path
    prereg_dir: Path
    keys_file: Path
    live_enabled: bool = True
    usgs_api_base: str = USGS_OGC_BASE
    api_url: str = "http://127.0.0.1:8000"
    upload_limits: dict = field(default_factory=lambda: {k: v * 2**20 for k, v in UPLOAD_LIMITS_MIB.items()})
    max_observation_rows: int = 200_000
    default_observation_rows: int = 50_000

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        repo = paths.REPO_ROOT
        s = cls(
            data_root=paths.DATA_ROOT,
            processed=paths.PROCESSED,
            raw=paths.RAW,
            manifests=paths.MANIFESTS,
            runs_root=paths.RUNS_ROOT,
            registry_path=paths.REGISTRY_PATH,
            repo_root=repo,
            model_cards_dir=repo / "docs" / "model_cards",
            models_yaml=Path(__file__).resolve().with_name("models.yaml"),
            results_md=repo / "RESULTS.md",
            research_plan=repo / "RESEARCH_PLAN.md",
            prereg_dir=repo / "docs" / "preregistration",
            keys_file=Path.home() / ".config" / "sentinel" / "keys.env",
            live_enabled=_flag("SENTINEL_LIVE", True),
            api_url=os.environ.get("SENTINEL_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        )
        merged = {**_OVERRIDES, **overrides}
        return replace(s, **merged) if merged else s

    @property
    def live_dir(self) -> Path:
        """Where the live prospective-trial pipeline writes forecast files."""
        return self.runs_root / "live"

    def rel(self, p: str | os.PathLike) -> str:
        """Path relative to the data root (or repo root) for display; absolute otherwise."""
        p = Path(p)
        for root, prefix in (
            (self.data_root, "$SENTINEL_DATA"),
            (self.runs_root, "$SENTINEL_RUNS"),
            (self.repo_root, ""),
        ):
            try:
                r = p.resolve().relative_to(Path(root).resolve())
            except (ValueError, OSError):
                continue
            return f"{prefix}/{r}" if prefix else str(r)
        return str(p)
