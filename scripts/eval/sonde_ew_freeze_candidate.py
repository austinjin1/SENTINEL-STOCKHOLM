#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Assemble the EW-v2 FREEZE CANDIDATE from one calibrate run (val only). Writes
``docs/preregistration/ew_v2_freeze_candidate.json``; it never writes a FREEZE_*.json file and never
loads test data (the lead creates the freeze record after an independent pre-test review).

Pre-specified selection rule (protocol §5): per primary hazard, the SENTINEL method is the
candidate (non-baseline) method with the highest VAL out-of-fold (by water year) BSS vs the
identically recalibrated climatology at L = 3 d. Baselines (never selectable): climatology,
persistence, threshold, logistic, event_gbm (the protocol's GBM baseline, identical inputs).
Quantile-derived methods (q_*) are excluded from selection because their test inputs would need
frozen M1 forecasting checkpoints (reported as secondary descriptives only).

    python -B scripts/eval/sonde_ew_freeze_candidate.py --run <sentinel_runs/sonde/ew_...>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from sentinel.core import paths
from sentinel.models.sonde import guard

BASELINES = ("climatology", "persistence", "threshold", "logistic", "event_gbm")
PRIMARY = ("H1_hypoxia", "H2_chloride", "H4_nitrate")
SECONDARY = ("H1_hypoxia_t4", "H1_hypoxia_t2", "H2_chloride_acute", "H5b_chl_bloom")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lead", type=int, default=3)
    ap.add_argument("--out", default=str(paths.REPO_ROOT / "docs" / "preregistration" / "ew_v2_freeze_candidate.json"))
    ap.add_argument("--configs-tried", type=int, required=True, help="EW calibrate configurations tried on val")
    ap.add_argument("--snapshot", action="append", default=[],
                    help="role=manifest_name pinning a data snapshot (default sonde_daily, sonde_hourly, "
                         "forcings_daily, chloride_derived)")
    a = ap.parse_args()
    run = Path(a.run)
    cf = run / "event_calibration.json"
    cal = json.loads(cf.read_text())
    res = pd.read_csv(run / "event_scores.csv")
    if set(res["block"]) != {"val_oof_by_wy"}:
        raise SystemExit("calibrate runs only (val_oof_by_wy)")
    selected, evidence = {}, {}
    for hz in PRIMARY + SECONDARY:
        r = res[(res["hazard"] == hz) & (res["lead"] == a.lead) & res["bss"].notna()]
        cand = r[~r["method"].isin(BASELINES) & ~r["method"].str.startswith("q_")]
        if cand.empty:
            continue
        best = cand.sort_values("bss", ascending=False).iloc[0]
        selected[hz] = str(best["method"])
        keep = ["method", "n", "n_pos", "bss", "bss_ci_low", "bss_ci_high", "bss_sm_ci_low", "bss_sm_ci_high",
                "n_basin_groups_with_onsets", "auroc", "b1_pooled_alarm_days_per_30d", "b1_pod_real", "b1_pod_placebo"]
        evidence[hz] = r[[c for c in keep if c in r.columns]].round(5).to_dict("records")
    weights = {}
    for key, ent in cal["entries"].items():
        for name, art in ent["artifacts"].items():
            f = run / art["file"]
            if sha(f) != art["sha256"]:
                raise SystemExit(f"{f} changed since calibration")
            weights[f"{key}|{name}"] = art["sha256"]
    files = list(guard.EW_SCORING_CODE)
    snaps = {"sonde_daily": "sonde_daily", "sonde_hourly": "sonde_hourly", "forcings_daily": "forcings_daily",
             "chloride_derived": "chloride_derived"}
    snaps.update(dict(x.split("=", 1) for x in a.snapshot))
    import importlib.util

    spec = importlib.util.spec_from_file_location("se", paths.REPO_ROOT / "scripts" / "train" / "sonde_events.py")
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    pins = {}
    for role, name in snaps.items():
        mf = paths.MANIFESTS / f"{name}.json"
        man = json.loads(mf.read_text())
        # W1: content pin over the files the loader reads (verified strictly before the single-pass marker)
        pins[role] = {"manifest": name, "manifest_sha256": sha(mf),
                      "created": man.get("created") or man.get("timestamp"), "n_files": len(man.get("files", [])),
                      "loader_dir": f"processed/{se.LOADER_DIRS[role]}", **se.snapshot_content_sha256(role)}
        print(f"pinned {role}: {name} content {pins[role]['content_sha256'][:12]} ({pins[role]['n_content_files']} files)")
    cl_man = json.loads((paths.MANIFESTS / f"{snaps['chloride_derived']}.json").read_text())
    sd_dir = str((paths.PROCESSED / "sonde_daily").resolve())
    if (cl_man.get("stats") or {}).get("sonde_daily_dir") != sd_dir:   # C1: H2 series from the pinned daily tier
        raise SystemExit(f"chloride_derived was not built from the loader's sonde_daily ({sd_dir})")
    cfg = se.runner.load_config("configs/sonde/events.yaml")
    chash = se.frozen_config_hash(cfg)
    if chash != cal.get("config_hash"):
        raise SystemExit(f"current configs/sonde/events.yaml hash {chash} != calibration {cal.get('config_hash')}")
    split = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json"
    out = {
        "status": "CANDIDATE — not frozen; the lead writes docs/preregistration/FREEZE_ew.json (or FREEZE_sonde.json) "
                  "after the independent pre-test review. Test data have not been loaded.",
        "prepared": _dt.datetime.now().isoformat(timespec="seconds"),
        "selection_rule": __doc__.split("Pre-specified selection rule")[1].split("python -B")[0].strip(),
        "selected_method": selected,
        "primary_hazards": list(PRIMARY), "secondary_hazards": list(SECONDARY), "primary_lead_days": a.lead,
        "val_oof_evidence_lead3": evidence,
        "configs_tried_ew_val": a.configs_tried,
        "freeze_record_draft": {
            "created": "[PENDING — set by the lead at freeze]",
            "split_sha256": sha(split),
            "scoring_code_files": files,
            "scoring_code_sha256": guard.scoring_code_sha256(files=files),
            "thresholds_sha256": sha(cf),
            "thresholds_file": str(cf),
            "weights_sha256": weights,
            "calibration_config_hash": cal.get("config_hash"),
            "selected_method": selected,
            "invocation": {"config": "configs/sonde/events.yaml",
                           "config_sha256": sha(paths.REPO_ROOT / "configs" / "sonde" / "events.yaml"),
                           "config_hash": chash, "quantile_models": {}, "lstm_checkpoint": None,
                           "include_test_sites": True, "passes": {"test": ["test"], "test_ab": ["test", "test_b"]},
                           "freeze_file": "docs/preregistration/FREEZE_ew.json"},
            "primary": {"population": "test-role sites x Test-A (WY2024-25), Approved-value labels",
                        "family": list(se.PRIMARY_FAMILY), "lead_days": se.PRIMARY_LEAD, "alpha": 0.05,
                        "multiplicity": "Holm over the fixed family; underpowered hazards enter with p = 1",
                        "min_evaluable_onsets": se.MIN_ONSETS, "min_basin_groups_with_onsets": se.MIN_ONSET_BASIN_GROUPS,
                        "evaluable_onset": "not censored, onset in (first issue, last issue + 3 d], >= 1 usable issue "
                                           "row in [onset - 3 d, onset) (ew_scoring.evaluable_onsets; gate = POD = controls)",
                        "ci": "basin-group > site-month two-stage block bootstrap, B = 2000",
                        "p_value": "one-sided (1 + #{BSS* <= 0}) / (1 + B)"},
            "pooled_secondary": {"hazard": "H1_hypoxia", "blocks": ["test", "test_b"], "approval_cutoff": "2027-06-30",
                                 "rule": "WY2026 label days still provisional at retrieval are missing; fraction reported",
                                 "data_pin": f"{se.POOLED_DATA_FILE} (dated record written on/after the cut-off: "
                                             "data_snapshots for all four loader roles + identity_check of rows "
                                             "<= 2023-09-30 vs these pins; Test-A rows re-derived from the new "
                                             "snapshot; changed WY2024-25 label days reported). This freeze record "
                                             "is never rewritten for the pooled pass."},
            "data_snapshots": pins,
            "data_snapshots_note": "Each promoted loader directory is described by two manifests (e.g. sonde_daily.json "
                                   "and sonde_daily_refresh_20261108.json, same files); this record pins the dated "
                                   "refresh manifest AND a content hash over the files the loader reads "
                                   "(sonde_events.SNAPSHOT_FILES), which is what the score phase verifies.",
            "single_pass": {"marker": "sentinel_runs/sonde/.ew_scored_<pass>.json (fixed name, independent of this "
                                      "record)", "registry_rows": "ew_v2_<pass>_score (pass_claimed row logged right "
                                                                  "after the marker)"},
            "git_commit": "[PENDING — tree is uncommitted (user instruction: no commits); hashes above are content hashes]",
        },
        "score_command": ("python -B scripts/train/sonde_events.py --phase score --config configs/sonde/events.yaml "
                          f"--calibration {cf} --eval-blocks test --allow-test --include-test-sites "
                          "--freeze-file docs/preregistration/FREEZE_ew.json --run-name ew_v2_testA_score"),
        "configs_tried_detail": f"{a.configs_tried} calibrate configurations + 5 offline stack variants (val only; "
                                "registry ew_v2_val_oof, ew_v2_stack_explore)",
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out}: selected {selected}; scoring_code_sha256 {out['freeze_record_draft']['scoring_code_sha256'][:12]}; "
          f"thresholds {out['freeze_record_draft']['thresholds_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
