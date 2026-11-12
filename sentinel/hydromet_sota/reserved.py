# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Sites reserved by other workstreams that M6 must never train, tune or select on.

* ``docs/preregistration/sonde_newsites_list.json`` (M1 EW-v3 / M1-N unseen test block, 729 USGS sites; lead notice
  2026-11-09). Its sha256 is pinned here; a changed file is refused until the pin is updated with a dated note.

Every M6 SOTA builder calls :func:`exclude_reserved` on its candidate site list (and logs the count) and
:func:`assert_no_reserved` on the final training/tuning site set.
"""

from __future__ import annotations

import hashlib
import json

from ..core import paths

NEWSITES_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_newsites_list.json"
NEWSITES_SHA256 = "6200a77cbdb0bb7963e6373a962374a764a6577bd7b945a42530a2b2e4e8b720"


class ReservedSiteError(RuntimeError):
    pass


def reserved_usgs_sites(path=None, sha256: str | None = NEWSITES_SHA256) -> set[str]:
    p = path or NEWSITES_FILE
    raw = p.read_bytes()
    if sha256 and hashlib.sha256(raw).hexdigest() != sha256:
        raise ReservedSiteError(f"{p} changed (sha256 != pinned {sha256[:12]}); update the pin with a dated note")
    body = json.loads(raw)
    return {str(s["site_no"]) for s in body["sites"]}


def _norm(site: str) -> str:
    s = str(site)
    return s.split("-", 1)[1] if s.upper().startswith("USGS-") else s


def exclude_reserved(sites, *, log=print, **kw) -> list[str]:
    """Drop reserved sites (USGS numbers or ``USGS-<no>`` WQP ids); logs how many were removed."""
    res = reserved_usgs_sites(**kw)
    keep = [s for s in sites if _norm(s) not in res]
    log(f"[reserved] excluded {len(sites) - len(keep)} of {len(sites)} sites listed in {NEWSITES_FILE.name}")
    return keep


def assert_no_reserved(sites, **kw) -> None:
    """Raise if any reserved site is present in an M6 training/tuning/selection set."""
    res = reserved_usgs_sites(**kw)
    bad = sorted({_norm(s) for s in sites} & res)
    if bad:
        raise ReservedSiteError(f"{len(bad)} reserved M1 test sites in M6 training data, e.g. {bad[:5]}")
