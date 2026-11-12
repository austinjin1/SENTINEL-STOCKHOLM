# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""ENFORCING access guard for the PROTECTED USGS new-site block (EW-v3-N / M1-N unseen test; review C-N3).

Data: ``processed/sonde_newsites`` (own tiers) — locked at mode 000 after its manifest is built
(``scripts/data/lock_newsites.py``). Every read of new-site values goes through :func:`open_site`, which enforces,
in this order (no caller-supplied callbacks):
  1. ``purpose == "score"``;
  2. a lead-written freeze for the look (``FREEZE_ew_v3.json`` / ``FREEZE_m1n.json``): not a candidate, no
     placeholders, pinning this dataset's ``content_sha256`` and the site-list sha256 (verified once per process);
  3. an OPEN single-look claim of THIS process: the look's ``CLAIM.json`` exists, records this pid and host, and
     neither ``SCORED.json`` nor ``ABORTED.json`` exists (read from disk here, never from the caller);
  4. the file mode is readable only because the look's scorer unlocked it under that claim.
:func:`assert_not_newsite` / :func:`exclude_reserved` (block-list, sha-pinned list) must be called by every
non-frozen loader that reads the SHARED tiers (``processed/sonde_daily``, ``sonde_hourly``, ``raw/usgs_iv``) and by
every EW-v3 / M1-N training or development entry point; they refuse / drop reserved sites.
Integrity tooling reads only ``integrity.parquet`` (counts and coverage).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

from ..core import paths

ROOT = paths.PROCESSED / "sonde_newsites"
LIST = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_newsites_list.json"
LIST_SHA256 = "6200a77cbdb0bb7963e6373a962374a764a6577bd7b945a42530a2b2e4e8b720"
FREEZES = {"ew_v3": paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_ew_v3.json",
           "m1n": paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_m1n.json"}
LOOK_DIRS = {"ew_v3": paths.RUNS_ROOT / "sonde" / "looks" / "ew_v3_newsite",
             "m1n": paths.RUNS_ROOT / "sonde" / "looks" / "m1n_newsite"}
TIERS = ("dv", "daily_iv", "hourly")
_VERIFIED: dict = {}


class ProtectedNewSiteError(PermissionError):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def content_sha256(root: Path = ROOT) -> str:
    files = sorted(p for t in TIERS for p in (root / t / "sites").glob("*.parquet"))
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}\t{_sha(f)}\n".encode())
    return h.hexdigest()


def newsite_ids() -> set[str]:
    if _sha(LIST) != LIST_SHA256:
        raise ProtectedNewSiteError("new-site list differs from its pinned sha256")
    return {s["site_no"] for s in json.loads(LIST.read_text())["sites"]}


def assert_not_newsite(sites) -> None:
    """Block-list check for every non-frozen loader of the shared tiers / every EW-v3 & M1-N dev entry point."""
    bad = sorted(set(map(str, sites)) & newsite_ids())
    if bad:
        raise ProtectedNewSiteError(f"{len(bad)} reserved new-site-block sites requested (e.g. {bad[:3]})")


def exclude_reserved(sites) -> list[str]:
    """Drop reserved new-site-block sites (for index-driven candidate builders); returns the kept list."""
    res = newsite_ids()
    return [s for s in map(str, sites) if s not in res]


def integrity_summary():
    import pandas as pd

    return pd.read_parquet(ROOT / "integrity.parquet")


def open_claim(look: str) -> dict:
    """The look's claim, verified from disk: exists, this pid + host, not sealed / aborted."""
    d = LOOK_DIRS.get(look)
    if d is None or not (d / "CLAIM.json").is_file():
        raise ProtectedNewSiteError(f"no single-look claim for {look!r}")
    if (d / "SCORED.json").exists() or (d / "ABORTED.json").exists():
        raise ProtectedNewSiteError(f"look {look!r} is sealed or aborted: access closed")
    c = json.loads((d / "CLAIM.json").read_text())
    if c.get("pid") != os.getpid() or c.get("host") != socket.gethostname():
        raise ProtectedNewSiteError("the open claim belongs to another process / host")
    return c


def _enforce(look: str, purpose: str) -> dict:
    if purpose != "score":
        raise ProtectedNewSiteError(f"new-site block: purpose {purpose!r} refused (score only)")
    if look not in FREEZES or not FREEZES[look].is_file():
        raise ProtectedNewSiteError(f"no lead-written freeze for look {look!r}")
    fr = json.loads(FREEZES[look].read_text())
    if any(m in json.dumps(fr.get("status", "")) for m in ("candidate", "NOT_A_FREEZE")) or "<" in json.dumps(fr):
        raise ProtectedNewSiteError("freeze record is a candidate or has placeholders")
    pins = fr.get("newsite_data") or {}
    if pins.get("site_list_sha256") != LIST_SHA256 or _sha(LIST) != LIST_SHA256:
        raise ProtectedNewSiteError("new-site list differs from the freeze pin")
    claim = open_claim(look)
    if _VERIFIED.get(look) != pins.get("content_sha256"):
        if pins.get("content_sha256") != content_sha256():
            raise ProtectedNewSiteError("new-site data differ from the freeze pin")
        _VERIFIED[look] = pins.get("content_sha256")
    return claim


def open_site(site: str, tier: str, *, look: str, purpose: str):
    import pandas as pd

    _enforce(look, purpose)
    if tier not in TIERS:
        raise ValueError(tier)
    if str(site) not in newsite_ids():
        raise ProtectedNewSiteError("open_site serves new-site-block sites only")
    p = ROOT / tier / "sites" / f"{site}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def preclaim_checks(look: str) -> dict:
    """MUST be called by the EW-v3-N / M1-N scorers BEFORE claiming (lead ruling 2026-11-10): the pre-claim audit of
    registry rows and run records since 2026-11-09 (``sentinel.data.reserved.assert_preclaim_clean``)."""
    from .reserved import assert_preclaim_clean

    return assert_preclaim_clean(exempt_dirs=(LOOK_DIRS[look],))
