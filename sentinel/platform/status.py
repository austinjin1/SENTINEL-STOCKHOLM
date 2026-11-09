# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Project status shown by the Console, read from the repository and run directories.

Nothing here is typed in: the RESULTS.md headline lines are quoted verbatim, plan sections
are extracted from RESEARCH_PLAN.md at render time, registry counts are computed from the
registry, and the live trial counts as started only once a forecast file exists under
``$SENTINEL_RUNS/live/``.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

from .config import Settings

HEADLINE_KEYS = ("Last updated", "Status", "Threshold status")
LIVE_SUFFIXES = (".csv", ".csv.gz", ".parquet", ".nc", ".json", ".jsonl")


def results_headline(s: Settings) -> dict:
    p = Path(s.results_md)
    if not p.exists():
        return {"exists": False, "path": s.rel(p)}
    out: dict = {"exists": True, "path": s.rel(p)}
    for line in p.read_text(encoding="utf-8").splitlines()[:40]:
        for k in HEADLINE_KEYS:
            m = re.match(rf"^\*\*{re.escape(k)}\*\*\s*:\s*(.+)$", line.strip())
            if m:
                out[k] = m.group(1).strip()
    return out


def plan_section(s: Settings, number: int) -> str | None:
    """Markdown of ``## <number>. ...`` from RESEARCH_PLAN.md, up to the next level-2 heading."""
    p = Path(s.research_plan)
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^##\s+{number}\.\s", ln):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    body = "\n".join(lines[start:end]).rstrip()
    return re.sub(r"\n-{3,}\s*$", "", body)


def prereg_docs(s: Settings) -> list[dict]:
    d = Path(s.prereg_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"^\*\*Status:\*\*\s*(.+)$", text, re.M)
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), p.stem)
        out.append(
            {"file": s.rel(p), "title": title, "status": m.group(1).strip() if m else None, "text": text}
        )
    return out


def live_trial(s: Settings, max_files: int = 50) -> dict:
    d = s.live_dir
    files = []
    if d.is_dir():
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and p.name.lower().endswith(LIVE_SUFFIXES):
                st = p.stat()
                files.append(
                    {
                        "path": s.rel(p),
                        "bytes": st.st_size,
                        "modified_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                    }
                )
    files.sort(key=lambda f: f["modified_utc"], reverse=True)
    return {
        "live_dir": s.rel(d),
        "exists": d.is_dir(),
        "started": bool(files),
        "n_files": len(files),
        "latest": files[:max_files],
    }
