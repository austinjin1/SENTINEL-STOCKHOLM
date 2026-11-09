# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared helpers of the event-layer label-feasibility scripts (``scripts/eval/events_feasibility_*.py``;
docs/plans/event_layer_plan.md §3).

**Never-read-test-labels rule, enforced in code.** Every script (1) reads id / role / date columns only,
(2) builds a keep-mask that excludes every protected role and block, (3) passes the kept roles / dates
through :func:`assert_allowed` / :func:`assert_dates_within` (which raise :class:`ProtectedRowsError`),
and only then (4) reads label columns with :func:`read_labels`, which returns kept rows only. Loaders of
other workstreams that have their own guard (sonde ``prepare_data``, ``load_nrsa``, ``load_uk``) are used
with their protective defaults.

Registration: ``--register`` appends one registry row per count to ``results/registry.jsonl``
(workstream ``events``, experiment ``events/feasibility``, model ``labels:<workstream>``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sentinel.core import paths

WORKSTREAM = "events"
EXPERIMENT = "events/feasibility"
PROTECTED_TOKENS = ("test", "holdout", "reserved", "protected")
FUTURE_CUTOFF = pd.Timestamp("2023-09-30")   # later external records stay unseen (future event-track tests)


class ProtectedRowsError(RuntimeError):
    """Raised before any label column is read if a protected row survived the id/role/date mask."""


def assert_allowed(values, allowed, what: str) -> None:
    v = pd.Series(np.asarray(values, dtype=object)).dropna().astype(str)
    bad = sorted(set(v) - set(map(str, allowed)))
    if bad or any(any(t in x.lower() for t in PROTECTED_TOKENS) for x in set(v)):
        raise ProtectedRowsError(f"{what}: protected or unexpected values kept {bad[:5]}; "
                                 "labels are never read for these rows")


def assert_dates_within(dates, lo=None, hi=FUTURE_CUTOFF, what: str = "dates") -> None:
    d = pd.to_datetime(pd.Series(np.asarray(dates)), errors="coerce").dropna()
    if hi is not None and (d > pd.Timestamp(hi)).any():
        raise ProtectedRowsError(f"{what}: rows after {pd.Timestamp(hi).date()} kept")
    if lo is not None and (d < pd.Timestamp(lo)).any():
        raise ProtectedRowsError(f"{what}: rows before {pd.Timestamp(lo).date()} kept")


def read_labels(path, columns, keep: np.ndarray, *, filters=None) -> pd.DataFrame:
    """Label columns of ``path`` for the rows of ``keep`` (a mask computed from id/role/date columns read
    with the same ``filters``); protected rows are dropped before the frame is returned."""
    t = pq.read_table(path, columns=list(columns), filters=filters).to_pandas()
    keep = np.asarray(keep, bool)
    if len(t) != keep.size:
        raise ValueError(f"{path}: {len(t)} label rows vs mask of {keep.size}")
    return t[keep].reset_index(drop=True)


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true", help="append the counts to the results registry")
    ap.add_argument("--out", default=None, help="output directory (default: $SENTINEL_RUNS/events/feasibility_<name>)")
    return ap


def out_dir(args, name: str) -> Path:
    p = Path(args.out) if args.out else paths.RUNS_ROOT / "events" / f"feasibility_{name}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write(df: pd.DataFrame, out: Path, name: str, meta: dict | None = None) -> Path:
    f = out / f"{name}.csv"
    df.to_csv(f, index=False)
    if meta is not None:
        (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return f


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def register(rows: list[dict], *, script: str, artifacts, data_manifests=(), notes: str = "") -> int:
    """rows: dicts with model (workstream), target (event), split (cell label, never a test cell), metric,
    value, n and optional extra. NaN values are skipped (the registry refuses them). A re-run supersedes the
    current entry with the same (model, target, split, metric). Returns rows logged."""
    from sentinel.core.registry import log_result, registry_frame

    prev = {}
    cur = registry_frame()
    if not cur.empty:
        cur = cur[cur["experiment"] == EXPERIMENT].sort_values("timestamp")
        prev = {(m, t, s, mt): i for m, t, s, mt, i in
                cur[["model", "target", "split", "metric", "id"]].itertuples(index=False)}
    k = 0
    for r in rows:
        v = r["value"]
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if any(t in str(r["split"]).lower() for t in ("test",)):
            raise ProtectedRowsError(f"refusing to register a count under split {r['split']!r}")
        key = (f"labels:{r['model']}", str(r["target"]), str(r["split"]), r["metric"])
        log_result(workstream=WORKSTREAM, experiment=EXPERIMENT, model=key[0], metric=r["metric"],
                   supersedes=prev.get(key),
                   value=float(v), split=str(r["split"]), n=None if r.get("n") is None else int(r["n"]),
                   target=str(r["target"]), config={"script": script, "version": "events_feasibility v1"},
                   data_manifests=list(data_manifests), artifacts=[str(a) for a in artifacts],
                   notes=notes or "event-layer label feasibility (plan §3); train/val roles only",
                   extra={kk: _clean(vv) for kk, vv in (r.get("extra") or {}).items()})
        k += 1
    return k


def stamp() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
