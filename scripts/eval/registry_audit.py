#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Read-only integrity audit of the append-only results registry.

Written after A5 delta review A5.3-a, which found two M4 rows whose ``supersedes`` pointed at row
ids that existed nowhere (ids that had been guessed rather than read). Such a row retires nothing,
so the row it was meant to retire stays live — and nothing flagged it. Its **first run found twelve,
not two**: the other ten were M4's own `prereg_snapshot_v7`/`v8`/`v9` rows, which put a version
*name* (``"prereg_snapshot_v6"``) into a field that must hold a row *id*. All twelve were M4 errors;
no other workstream's row was affected.

Checks, over the whole registry (every workstream):

* **dangling ``supersedes``** — a row whose ``supersedes`` names an id that no row carries;
  it FAILS the audit unless an explicit acknowledgement row names it (see below);
* **duplicate ids** — two rows sharing one ``id``;
* **self-supersession** — a row that supersedes itself.

**Acknowledgement.** The registry is append-only, so a dangling ``supersedes`` field can never be
edited away; an audit that simply failed on it would fail forever. An acknowledgement row — metric
``dangling_supersedes_acknowledged`` with ``extra.acknowledges_dangling`` = the dangling row's id
and a stated reason — records a deliberate, reasoned decision. It does **not** use ``supersedes``,
so it retires nothing: acknowledging a malformed field must never hide the row's content. The
audit lists acknowledged rows but fails only on **unacknowledged** ones, so it cannot be satisfied
silently.

It **only reads**. It never writes, never edits a row and never supersedes anything; exit status 1
if any check fails, so it can gate a script or a CI step.

    python -B scripts/eval/registry_audit.py                 # whole registry
    python -B scripts/eval/registry_audit.py --workstream chemtox
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:          # pragma: no cover - corrupt line
                raise SystemExit(f"{path}:{n}: unparseable registry line ({e})") from e
    return rows


def audit(rows: list[dict], workstream: str | None = None) -> dict:
    ids = Counter(r.get("id") for r in rows)
    known = set(ids)
    scope = [r for r in rows if workstream is None or r.get("workstream") == workstream]
    acked = {(r.get("extra") or {}).get("acknowledges_dangling") for r in rows
             if r.get("metric") == "dangling_supersedes_acknowledged"}
    all_dangling = [r for r in scope if r.get("supersedes") and r["supersedes"] not in known]
    dangling = [r for r in all_dangling if r["id"] not in acked]
    acknowledged = [r for r in all_dangling if r["id"] in acked]
    selfsup = [r for r in scope if r.get("supersedes") and r["supersedes"] == r.get("id")]
    dup = sorted(i for i, c in ids.items() if c > 1)
    return {"n_rows": len(rows), "n_in_scope": len(scope), "dangling": dangling,
            "acknowledged": acknowledged,
            "self_supersession": selfsup, "duplicate_ids": dup}


def main(argv=None) -> int:
    from sentinel.core import paths

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(paths.REPO_ROOT / "results" / "registry.jsonl"))
    ap.add_argument("--workstream", default=None)
    a = ap.parse_args(argv)

    res = audit(load(Path(a.registry)), a.workstream)
    print(f"registry rows {res['n_rows']}, in scope {res['n_in_scope']}"
          + (f" (workstream={a.workstream})" if a.workstream else ""))
    for r in res["dangling"]:
        print(f"  DANGLING  {r['id']}  {r.get('workstream')}/{r.get('experiment')}/{r.get('model')}"
              f"  supersedes {r['supersedes']} -> no such row")
    for r in res["acknowledged"]:
        print(f"  acknowledged-dangling  {r['id']}  supersedes {r['supersedes']} (reasoned, not failing)")
    for r in res["self_supersession"]:
        print(f"  SELF-SUPERSEDES  {r['id']}")
    for i in res["duplicate_ids"]:
        print(f"  DUPLICATE-ID  {i}")
    bad = len(res["dangling"]) + len(res["self_supersession"]) + len(res["duplicate_ids"])
    print("OK" if not bad else f"FAIL: {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
