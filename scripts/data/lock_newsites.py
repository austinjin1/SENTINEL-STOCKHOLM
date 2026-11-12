#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lock / unlock the PROTECTED new-site directory ``processed/sonde_newsites`` (review C-N3).

``--lock``   (after ``download_newsites.py manifest``): verify the manifest's content pin, then chmod every data file
             under ``dv/``, ``daily_iv/``, ``hourly/`` to 000 and those directories to 0300 (no listing, no reading —
             an accidental read fails loudly, also for the owner). ``manifest.json`` and ``integrity.parquet`` (counts
             and coverage only) stay readable. Contents are unchanged, so the content pin stays valid.
UNLOCK IS SCORER-ONLY (review warning): the look's scorer imports this module and calls
             ``unlock_in_process(look)`` after claiming; it checks ``newsite_guard.open_claim`` (claim of THIS process,
             not sealed/aborted) and sets files 0400 / dirs 0500; the scorer calls ``relock()`` in its ``finally``.
             The command line ``--unlock`` always refuses (a CLI process can never hold the scorer's claim).
             Every lock/unlock is appended to ``lock_log.jsonl``.
Not run while the pull is in progress.

    python -B scripts/data/lock_newsites.py --lock
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket

from sentinel.data import newsite_guard as G

TIER_DIRS = ("dv", "daily_iv", "hourly")


def _log(rec: dict) -> None:
    with open(G.ROOT / "lock_log.jsonl", "a") as fh:
        fh.write(json.dumps({**rec, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "pid": os.getpid(),
                             "host": socket.gethostname()}) + "\n")


def set_mode(file_mode: int, dir_mode: int) -> int:
    n = 0
    for t in TIER_DIRS:
        d = G.ROOT / t / "sites"
        if not d.exists():
            continue
        os.chmod(d, 0o700)                      # make sure we can list while changing
        for f in d.iterdir():
            os.chmod(f, file_mode)
            n += 1
        os.chmod(d, dir_mode)
    return n


def unlock_in_process(look: str) -> int:
    G.open_claim(look)                          # refuses unless THIS process holds the open claim
    n = set_mode(0o400, 0o500)
    _log({"action": "unlock", "look": look, "n_files": n})
    return n


def relock(look: str) -> int:
    n = set_mode(0o000, 0o300)
    _log({"action": "relock", "look": look, "n_files": n})
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--unlock", action="store_true")
    ap.add_argument("--look", choices=tuple(G.LOOK_DIRS))
    a = ap.parse_args()
    if a.lock:
        man = json.loads((G.ROOT / "manifest.json").read_text())
        if man["content_sha256"] != G.content_sha256():
            raise SystemExit("content differs from the manifest pin: not locking")
        n = set_mode(0o000, 0o300)
        _log({"action": "lock", "n_files": n, "content_sha256": man["content_sha256"]})
        print(f"locked {n} files")
    elif a.unlock:
        raise SystemExit("unlock is scorer-only: the look's scorer calls unlock_in_process(look) inside its own claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
