# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared, hash-pinned exclusion helper for RESERVED sites (lead ruling 2026-11-10; EW-v3-N / M1-N new-site block).

Every NON-frozen loader that reads the shared sonde tiers (``processed/sonde_daily``, ``sonde_hourly``,
``raw/usgs_iv``) by index, glob or explicit list calls one of these on its site list:
  * :func:`exclude_reserved` — drop reserved sites (index- or glob-driven candidate builders);
  * :func:`assert_no_reserved` — refuse (split-gated loaders, where a reserved site would be a bug).
The reserved list is ``docs/preregistration/sonde_newsites_list.json``, pinned by SHA-256 here; a changed list raises.
Frozen scorers are never edited (their pinned code stays as frozen; they are split-gated or content-pinned).
Site IDs may be given bare (``01646500``) or prefixed (``USGS-01646500``, ``WQP:USGS-01646500``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re

from ..core import paths

NEWSITE_LIST = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_newsites_list.json"
NEWSITE_LIST_SHA256 = "6200a77cbdb0bb7963e6373a962374a764a6577bd7b945a42530a2b2e4e8b720"
_ID = re.compile(r"(\d{8,15})$")


class ReservedSiteError(PermissionError):
    pass


@functools.lru_cache(maxsize=1)
def reserved_ids() -> frozenset[str]:
    raw = NEWSITE_LIST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NEWSITE_LIST_SHA256:
        raise ReservedSiteError("reserved new-site list differs from its pinned SHA-256")
    return frozenset(s["site_no"] for s in json.loads(raw)["sites"])


def _bare(site) -> str:
    m = _ID.search(str(site))
    return m.group(1) if m else str(site)


def is_reserved(site) -> bool:
    return _bare(site) in reserved_ids()


def exclude_reserved(sites) -> list:
    """Return ``sites`` without reserved ones (order and original ID form kept)."""
    res = reserved_ids()
    return [s for s in sites if _bare(s) not in res]


def assert_no_reserved(sites, where: str = "") -> None:
    bad = sorted({_bare(s) for s in sites} & reserved_ids())
    if bad:
        raise ReservedSiteError(f"{len(bad)} reserved new-site-block sites requested{' in ' + where if where else ''} "
                                f"(e.g. {bad[:3]})")


# --------------------------------------------------------------------------- pre-claim audit (lead ruling 2026-11-10)
AUDIT_SINCE = "2026-11-09"
_ANY_ID = re.compile(r"(?<!\d)(\d{8,15})(?!\d)")


def preclaim_audit(since: str = AUDIT_SINCE, *, registry_path=None, runs_root=None,
                   exempt_dirs: tuple = ()) -> dict:
    """Scan every registry row with timestamp >= ``since`` and every run record (``data_info.json``,
    ``*run*.json`` / ``*manifest*.json`` under ``sentinel_runs``) modified since then for reserved site IDs.
    Returns {"registry_hits": [...], "run_record_hits": [...]} — the EW-v3 / M1-N scorers refuse to claim if either
    is non-empty (``exempt_dirs``: the look's own directory)."""
    import datetime as _dt
    import json as _json
    import os

    res = reserved_ids()
    rp = registry_path or paths.REGISTRY_PATH
    reg_hits = []
    with open(rp) as fh:
        for ln in fh:
            try:
                r = _json.loads(ln)
            except ValueError:
                continue
            if str(r.get("timestamp", ""))[:10] < since:
                continue
            ids = set(_ANY_ID.findall(ln)) & res
            if ids:
                reg_hits.append({"id": r.get("id"), "experiment": r.get("experiment"), "sites": sorted(ids)[:5]})
    t0 = _dt.datetime.fromisoformat(since).timestamp()
    run_hits = []
    root = runs_root or paths.RUNS_ROOT
    for dirpath, _, files in os.walk(root):
        if any(str(dirpath).startswith(str(e)) for e in exempt_dirs):
            continue
        for f in files:
            if not (f == "data_info.json" or (f.endswith(".json") and ("run" in f or "manifest" in f))):
                continue
            p = os.path.join(dirpath, f)
            try:
                if os.stat(p).st_mtime < t0:
                    continue
                ids = set(_ANY_ID.findall(open(p, errors="ignore").read())) & res
            except OSError:
                continue
            if ids:
                run_hits.append({"file": p, "sites": sorted(ids)[:5]})
    return {"since": since, "registry_hits": reg_hits, "run_record_hits": run_hits}


def assert_preclaim_clean(**kw) -> dict:
    a = preclaim_audit(**kw)
    if a["registry_hits"] or a["run_record_hits"]:
        raise ReservedSiteError(f"pre-claim audit: reserved sites appear in {len(a['registry_hits'])} registry rows and "
                                f"{len(a['run_record_hits'])} run records since {a['since']}")
    return a
