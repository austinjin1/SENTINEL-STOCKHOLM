#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""PRINT (never write) the draft of ``docs/preregistration/FREEZE_ew_live.json`` for the lead (amendment §11).
The lead reviews it, writes the file himself (read-only), commits and pushes before 2026-12-01.

    python -B scripts/live/freeze_draft.py
"""

from __future__ import annotations

import json

import live_common as LC


def main() -> int:
    cfg, cfg_sha = LC.load_config()
    smap = LC.REPO / "configs" / "live" / "ew_v2_live_series_map.json"
    amend = LC.REPO / "docs" / "preregistration" / "ew_v2_live_amendment.md"
    draft = {
        "created": "<date, lead>",
        "status": "FROZEN — EW-v2 live prospective trial (Test-C), amendment L1",
        "authorised_by": "<lead>",
        "amendment": str(amend.relative_to(LC.REPO)),
        "amendment_sha256": LC.sha256_file(amend) if amend.is_file() else None,
        "config": str(LC.CONFIG.relative_to(LC.REPO)),
        "config_sha256": cfg_sha,
        "series_map": str(smap.relative_to(LC.REPO)),
        "series_map_sha256": LC.sha256_file(smap) if smap.is_file() else "<run build_series_map.py --out frozen>",
        "code_sha256": LC.code_sha256(),
        "code_files": LC.code_files(),
        "calibration_sha256": LC.verify_calibrations(cfg),
        "freeze_ew_sha256": LC.sha256_file(LC.REPO / "docs/preregistration/FREEZE_ew.json"),
        "env_lock_sha256": LC.env_lock_sha256(),
        "static_pins": LC.verify_static_pins(cfg),
        "python_env": "/home/bcheng/miniconda3/envs/sentinel2-live (dedicated; run_daily.sh)",
        "env_lockfiles_sha256": {f: LC.sha256_file(LC.REPO / "configs/live" / f) for f in
                                 ("sentinel2-live.conda-explicit.txt", "sentinel2-live.pip-freeze.txt")},
        "timestamp_backends": ["ots"],
        "auto_push": {"enabled": False, "remote": "origin", "branch": "v2-rebuild",
                      "note": "set enabled=true in the record at the freeze only if the owner approves push access"},
        "timestamp_backends_note": "set at the freeze, after the owner's approval of hash-only OTS submission and a "
                                   "successful ots_selftest.py; the record is never edited afterwards (amendment §11)",
        "trial": cfg["trial"],
        "hashlog": str(LC.HASHLOG.relative_to(LC.REPO)),
        "deviations_file": "docs/preregistration/ew_v2_live_deviations.md",
    }
    print(json.dumps(draft, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
