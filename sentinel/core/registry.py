# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Append-only results registry: the single source of truth for every reported number.

Rules (see RESEARCH_PLAN.md, "Integrity rules"):
  * Every metric that appears in RESULTS.md, a figure, the paper, the README or the
    platform must be written here by the code that computed it.
  * Entries are never edited or deleted. A corrected result is a new entry whose
    ``supersedes`` field names the old entry's ``id``.
  * Each entry records the code state (git commit, dirty flag, hash of the
    ``sentinel`` package and of the calling script) and the data manifests used.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path
from typing import Any, Iterable

from . import paths


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True,
                             text=True, timeout=20, check=False)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — git missing is not fatal
        return ""


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def package_hash() -> str:
    """Hash of every .py file in the sentinel package (order-independent of mtime)."""
    h = hashlib.sha256()
    pkg = paths.REPO_ROOT / "sentinel"
    for f in sorted(pkg.rglob("*.py")):
        h.update(str(f.relative_to(pkg)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def code_state() -> dict[str, Any]:
    script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    state = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "package_hash": package_hash(),
        "script": str(script) if script else None,
        "script_sha256": file_sha256(script) if script and script.is_file() else None,
        "argv": sys.argv,
    }
    return state


def config_hash(config: dict | None) -> str | None:
    if config is None:
        return None
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def snapshot_code(dest_dir: str | os.PathLike) -> Path:
    """Archive the sentinel package + calling script next to a run's artifacts."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "code_snapshot.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(paths.REPO_ROOT / "sentinel", arcname="sentinel",
                filter=lambda ti: None if "__pycache__" in ti.name else ti)
        script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
        if script and script.is_file():
            tar.add(script, arcname=f"script/{script.name}")
    return out


def log_result(
    *,
    workstream: str,
    experiment: str,
    model: str,
    metric: str,
    value: float,
    split: str,
    n: int | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    ci_method: str | None = None,
    target: str | None = None,
    is_baseline: bool = False,
    config: dict | None = None,
    data_manifests: Iterable[str] | None = None,
    artifacts: Iterable[str] | None = None,
    supersedes: str | None = None,
    notes: str = "",
    extra: dict | None = None,
    registry_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Append one metric to the registry and return the stored entry."""
    if value is None or (isinstance(value, float) and value != value):
        raise ValueError(f"refusing to log missing/NaN value for {experiment}/{metric}")
    entry = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "workstream": workstream,
        "experiment": experiment,
        "model": model,
        "is_baseline": bool(is_baseline),
        "target": target,
        "metric": metric,
        "value": float(value),
        "ci_low": None if ci_low is None else float(ci_low),
        "ci_high": None if ci_high is None else float(ci_high),
        "ci_method": ci_method,
        "split": split,
        "n": None if n is None else int(n),
        "config_hash": config_hash(config),
        "config": config,
        "data_manifests": list(data_manifests or []),
        "artifacts": [str(a) for a in (artifacts or [])],
        "supersedes": supersedes,
        "notes": notes,
        "extra": extra or {},
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "code": code_state(),
    }
    path = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return entry


def read_registry(registry_path: str | os.PathLike | None = None) -> list[dict[str, Any]]:
    path = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def registry_frame(registry_path: str | os.PathLike | None = None, *, current_only: bool = True):
    """Registry as a pandas DataFrame; drops superseded entries by default."""
    import pandas as pd

    rows = read_registry(registry_path)
    df = pd.DataFrame(rows)
    if df.empty or not current_only:
        return df
    superseded = set(df["supersedes"].dropna())
    return df[~df["id"].isin(superseded)].reset_index(drop=True)


def lookup(experiment: str, model: str, metric: str, split: str, *, target: str | None = None,
           registry_path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Fetch exactly one current registry entry; raise if missing or ambiguous.

    Figure and table code must obtain numbers through this function (never literals).
    """
    df = registry_frame(registry_path)
    if df.empty:
        raise KeyError("registry is empty")
    m = (df["experiment"] == experiment) & (df["model"] == model) & (df["metric"] == metric) \
        & (df["split"] == split)
    if target is not None:
        m &= df["target"] == target
    hits = df[m]
    if len(hits) == 0:
        raise KeyError(f"no registry entry for {experiment}/{model}/{metric}/{split}/{target}")
    if len(hits) > 1:
        hits = hits.sort_values("timestamp")
        latest = hits.iloc[-1]
        if not (hits["value"] == latest["value"]).all():
            raise KeyError(f"ambiguous: {len(hits)} differing entries for "
                           f"{experiment}/{model}/{metric}/{split}/{target}; supersede old ones")
        return latest.to_dict()
    return hits.iloc[0].to_dict()
