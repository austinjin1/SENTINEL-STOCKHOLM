#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Prepare the SENTINEL-Orbit freeze CANDIDATE (orbit_protocol.md §9, §9b, §12). Writes
docs/preregistration/orbit_freeze_candidate.json — NEVER FREEZE_orbit.json; the guard refuses every path except
FREEZE_orbit.json and every status except FROZEN, so the candidate cannot unlock protected rows.

Lead step (not done by this script): copy the candidate to FREEZE_orbit.json, set "status": "FROZEN", add
"authorised_by", "authorised" (ISO date) and "candidate_sha256" (sha256 of the candidate file). Every other key must
stay equal to the candidate's (the guard checks it), and scoring_code_sha256 / data_pins must still match. Then
scripts/eval/orbit_test.py runs once per look (M2/test, M2U/test).

    python -B scripts/eval/orbit_prepare_freeze.py --version dev_v5 --satwq satwq_v5_X_s0 ... \
        --baselines baselines_dev_v5 --comparators-json .../comparators_val_selected.json \
        [--blend-with gbm_imagery_context] [--track-u-satwq ... --track-u-baselines ...] --selection-note "..."
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import guard  # noqa: E402
from sentinel.models.orbit import split as SP  # noqa: E402
from sentinel.models.orbit import testlook as TL  # noqa: E402

CANDIDATE = guard.CANDIDATE_FILE


def sha(p: Path) -> str:
    return guard.artifact_sha256(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--satwq", nargs="+", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--comparators-json", required=True)
    ap.add_argument("--blend-with", default=None)
    ap.add_argument("--track-u-satwq", nargs="*", default=None)
    ap.add_argument("--track-u-baselines", default=None)
    ap.add_argument("--events-artifact", required=True, help="events_orbit_<version>.pkl (scripts/eval/events_orbit.py)")
    ap.add_argument("--track-u-events-artifact", default=None)
    ap.add_argument("--event-selection-note", required=True)
    ap.add_argument("--dryrun-dir", default=None, help="test_looks_dryrun/<label> of the val dry run (recorded)")
    ap.add_argument("--selection-note", required=True, help="which val result selected this configuration")
    ap.add_argument("--configs-tried", type=int, required=True)
    a = ap.parse_args()
    R = paths.RUNS_ROOT / "orbit"
    man = paths.MANIFESTS / f"orbit_{a.version}.json"
    ck = [R / r / "model.pt" for r in a.satwq]
    fb = R / a.baselines / "fitted_baselines.pkl"
    weights = {p.parent.name: sha(p) for p in ck}
    weights["fitted_baselines"] = sha(fb)
    weights["events_orbit"] = sha(Path(a.events_artifact))
    body = {"created": dt.datetime.now().isoformat(timespec="seconds"), "status": "CANDIDATE",
            "split_sha256": SP.split_sha256(), "data_version": a.version,
            "data_manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(),
            "satwq_checkpoints": [str(p) for p in ck], "fitted_baselines": str(fb), "blend_with": a.blend_with,
            "events_artifact": str(Path(a.events_artifact).resolve()),
            "event_track": {"protocol": "orbit_protocol.md §12", "primary_event": "E4a", "tiers": ["E4a_t12", "E4a_t50"],
                            "descriptive": ["E3a_tss", "E3a_turb"], "primary_route": "derived",
                            "challenger_route": "direct", "note": a.event_selection_note,
                            "not_compared": {"CyAN_CI_cyano": "needs an Earthdata login the user has not provided"}},
            "looks": {"M2/test": {"experiment": TL.experiment("M2/test", a.version),
                                  "cells": ["headline", "testB_2026", "pooled_2024_2026"]},
                      "M2U/test": {"experiment": TL.experiment("M2U/test", a.version), "cells": ["trackU_test"]}},
            "dryrun_dir": a.dryrun_dir,
            "comparators": json.loads(Path(a.comparators_json).read_text()),
            "selection": {"rule": "lowest val MdSA (chl-a, valxval_lake) among configurations scored on val, "
                                   "orbit_protocol.md §6", "note": a.selection_note,
                          "configs_tried": a.configs_tried}}
    if a.track_u_satwq:
        cu = [R / r / "model.pt" for r in a.track_u_satwq]
        fu = R / a.track_u_baselines / "fitted_baselines.pkl"
        for p in cu:
            weights[p.parent.name] = sha(p)
        weights["fitted_baselines_track_u"] = sha(fu)
        if a.track_u_events_artifact:
            weights["events_orbit_track_u"] = sha(Path(a.track_u_events_artifact))
        tf = paths.REPO_ROOT / "docs" / "preregistration" / "orbit_track_u.json"
        body["track_u"] = {"satwq_checkpoints": [str(p) for p in cu], "fitted_baselines": str(fu),
                           "events_artifact": (str(Path(a.track_u_events_artifact).resolve())
                                               if a.track_u_events_artifact else None),
                           "track_file_sha256": hashlib.sha256(tf.read_bytes()).hexdigest()}
    body["weights_sha256"] = weights
    print("[freeze] hashing source data for content pins (a few minutes)...", flush=True)
    body["data_pins"] = TL.make_data_pins()
    body["scoring_code_sha256"] = guard.scoring_code_sha256()
    body["frozen_files"] = guard.frozen_files()
    CANDIDATE.write_text(json.dumps(body, indent=1) + "\n")
    print(f"wrote {CANDIDATE} sha256 {hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
