# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Read-only views of the results registry (``results/registry.jsonl``) for the Console.

All reads go through :func:`sentinel.core.registry.registry_frame` (current entries only by
default). Every metric that leaves this module for display is resolved through
:mod:`sentinel.platform.guard`.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..core import registry as _registry
from . import guard
from .config import Settings

FILTER_FIELDS = ("workstream", "experiment", "model", "split", "metric", "target")


def is_test_split(split: Any) -> bool:
    """True if any '/'-, ':'- or '|'-separated component of the split name starts with "test".

    Matches the conventions in use: ``test``, ``test_a``, ``test_b`` (sonde) and
    ``t-F2F/occurrence/test`` (chemtox); ``val``, ``cv``, ``train`` and ``t-F2F/all`` are not tests.
    """
    if not isinstance(split, str):
        return False
    return any(part.strip().lower().startswith("test") for part in re.split(r"[/:|]", split))


def frame(s: Settings, *, current_only: bool = True) -> pd.DataFrame:
    path = Path(s.registry_path)
    if not path.exists():
        return pd.DataFrame()
    for attempt in range(3):  # tolerate a concurrent append exposing a partial line
        try:
            return _registry.registry_frame(path, current_only=current_only)
        except ValueError:
            if attempt == 2:
                raise
            time.sleep(0.05)
    return pd.DataFrame()


def filtered(s: Settings, **filters: Any) -> pd.DataFrame:
    df = frame(s)
    if df.empty:
        return df
    m = pd.Series(True, index=df.index)
    for k in FILTER_FIELDS:
        v = filters.get(k)
        if v is not None and v != "":
            m &= df[k].astype(str) == str(v) if k in df else False
    b = filters.get("is_baseline")
    if b is not None and "is_baseline" in df:
        m &= df["is_baseline"].astype(bool) == bool(b)
    return df[m]


def _json_safe(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    try:
        if pd.isna(v) is True:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return v
    return v


def entries(s: Settings, df: pd.DataFrame, *, include_config: bool = False) -> list[dict]:
    """Registry rows for output. Each row is verified by the guard and carries a ``display``."""
    out = []
    for _, row in df.iterrows():
        rec = {k: _json_safe(v) for k, v in row.to_dict().items()}
        if not include_config:
            rec.pop("config", None)
        rec["guarded"] = guard.guarded_metric(rec, registry_path=s.registry_path).as_dict()
        out.append(rec)
    return out


def status(s: Settings) -> dict:
    """Counts used by the Home page banner (computed from the registry, never typed in)."""
    path = Path(s.registry_path)
    if not path.exists():
        return {
            "registry_path": s.rel(path),
            "exists": False,
            "n_entries": 0,
            "n_current": 0,
            "n_test_non_baseline": 0,
            "has_validated_results": False,
            "by_workstream": [],
        }
    all_df = frame(s, current_only=False)
    cur = frame(s)
    if cur.empty:
        n_test = 0
        by_ws: list[dict] = []
    else:
        test = cur["split"].map(is_test_split)
        base = cur["is_baseline"].astype(bool)
        n_test = int((test & ~base).sum())
        by_ws = []
        for ws, g in cur.groupby("workstream", dropna=False):
            gt = g["split"].map(is_test_split)
            gb = g["is_baseline"].astype(bool)
            by_ws.append(
                {
                    "workstream": None if pd.isna(ws) else ws,
                    "current_entries": int(len(g)),
                    "baseline_entries": int(gb.sum()),
                    "test_split_non_baseline": int((gt & ~gb).sum()),
                    "experiments": int(g["experiment"].nunique()),
                }
            )
    return {
        "registry_path": s.rel(path),
        "exists": True,
        "n_entries": int(len(all_df)),
        "n_current": int(len(cur)),
        "n_test_non_baseline": n_test,
        "has_validated_results": n_test > 0,
        "by_workstream": by_ws,
    }
