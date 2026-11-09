# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Provenance blocks attached to every Console response that carries data.

A block names the dataset manifest(s) (``$SENTINEL_DATA/manifests/<name>.json``, written by
``sentinel.core.manifest.write_manifest``), their creation dates, sources, licence, the
``synthetic`` flag, upstream manifests (``manifest:<name>`` sources, followed recursively),
the local files read (with modification times) and, for observations, the approval status of
the values returned. A missing manifest is reported as missing — never filled in.
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
from pathlib import Path
from typing import Any, Iterable

from .config import Settings

_lock = threading.Lock()
_cache: dict[str, tuple[tuple, dict]] = {}

# Which manifest documents each local data product the Console reads.
DATASET_MANIFESTS: dict[str, str | None] = {
    "sonde_daily": "sonde_daily",
    "sonde_hourly": "sonde_hourly",
    "sites": "sites",
    "usgs_dv_catalog": "usgs_dv_raw",
    "usgs_iv_core_sites": None,  # site list used by the IV downloader; no manifest covers it
}

APPROVAL_RULE = (
    "provisional = USGS qualifier code beginning with 'P' when the data were retrieved; "
    "USGS may have revised or approved these values since retrieval"
)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _mtime_iso(st_mtime: float) -> str:
    return _dt.datetime.fromtimestamp(st_mtime, _dt.timezone.utc).isoformat(timespec="seconds")


def _load_manifest(path: Path) -> dict | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    key = str(path)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        m = {"_error": f"unreadable manifest: {type(e).__name__}"}
    with _lock:
        _cache[key] = (stamp, m)
    return m


def _retrieval_fields(stats: dict) -> dict:
    """Any stats keys that record a retrieval date (naming varies between downloaders)."""
    out = {}
    for k, v in (stats or {}).items():
        lk = k.lower()
        if "retriev" in lk or lk in {"downloaded", "download_date", "accessed"}:
            if isinstance(v, (str, int, float)) or v is None:
                out[k] = v
    return out


def manifest_summary(settings: Settings, name: str, *, _seen: set[str] | None = None) -> dict[str, Any]:
    seen = set() if _seen is None else _seen
    path = settings.manifests / f"{name}.json"
    m = _load_manifest(path)
    if m is None:
        return {
            "name": name,
            "status": "missing",
            "path": settings.rel(path),
            "note": "manifest not written yet (the dataset build may still be in progress); "
            "creation date, checksums and sources are unknown",
        }
    if "_error" in m:
        return {"name": name, "status": "unreadable", "path": settings.rel(path), "note": m["_error"]}
    code = m.get("code") or {}
    stats = m.get("stats") or {}
    out = {
        "name": m.get("name", name),
        "status": "present",
        "path": settings.rel(path),
        "created_utc": m.get("created"),
        "description": m.get("description"),
        "license": m.get("license"),
        "synthetic": bool(m.get("synthetic", False)),
        "sources": list(m.get("sources") or []),
        "n_files": m.get("n_files"),
        "total_bytes": m.get("total_bytes"),
        "retrieval": _retrieval_fields(stats),
        "git_commit": code.get("git_commit"),
        "git_dirty": code.get("git_dirty"),
    }
    seen.add(name)
    upstream = []
    for src in out["sources"]:
        if isinstance(src, str) and src.startswith("manifest:"):
            up = src.split(":", 1)[1]
            if up not in seen:
                upstream.append(manifest_summary(settings, up, _seen=seen))
    if upstream:
        out["upstream"] = upstream
    return out


def any_synthetic(summaries: Iterable[dict]) -> bool | None:
    """True if any manifest (or upstream) is synthetic; None if none of them is present."""
    present = False
    for s in summaries:
        if s.get("status") == "present":
            present = True
            if s.get("synthetic"):
                return True
        r = any_synthetic(s.get("upstream") or [])
        if r:
            return True
        if r is False:
            present = True
    return False if present else None


def file_record(settings: Settings, path: Path) -> dict:
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"path": settings.rel(path), "exists": False}
    return {
        "path": settings.rel(path),
        "exists": True,
        "bytes": st.st_size,
        "modified_utc": _mtime_iso(st.st_mtime),
    }


def block(
    settings: Settings,
    datasets: Iterable[str],
    files: Iterable[Path] = (),
    *,
    approval: dict | None = None,
    notes: Iterable[str] = (),
) -> dict:
    """Assemble the provenance block for a response."""
    manifests = []
    for d in datasets:
        name = DATASET_MANIFESTS.get(d, d)
        if name is None:
            manifests.append(
                {
                    "dataset": d,
                    "name": None,
                    "status": "no manifest",
                    "note": "this file is not covered by any dataset manifest",
                }
            )
        else:
            manifests.append({"dataset": d, **manifest_summary(settings, name)})
    out = {
        "generated_utc": utc_now(),
        "datasets": list(datasets),
        "manifests": manifests,
        "synthetic": any_synthetic(manifests),
        "files": [file_record(settings, Path(f)) for f in files],
        "notes": list(notes),
    }
    if approval is not None:
        out["approval"] = {**approval, "rule": APPROVAL_RULE}
    return out
