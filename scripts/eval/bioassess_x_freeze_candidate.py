#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write the M5-X freeze-CANDIDATE record ``docs/preregistration/bioassess_x_freeze_candidate.json`` (protocol §9).

Built from the development selection (``x_track/dev/selection_x_v1.json``). It fixes, for the single pass:
candidates per block (C-X + ≤ 2 secondaries) and per-target comparators as executable specs (model, variant per
target, ensemble members); seeds, B (2,000 marginal / 10,000 confirmatory), α and the 12-test Holm family; the
decision table; configuration counts and the selection note; sha256 of every code/config/protocol/split/pool/fold/
frozen-split file, of the data files, of the TabPFN weights; Python and R package versions. It never creates the
FREEZE file (the lead does, after the independent pre-test review). No model is fitted here.

    python -B scripts/eval/bioassess_x_freeze_candidate.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sentinel.core import paths
from sentinel.core.registry import file_sha256
from sentinel.models.bioassess import xdev as XD
from sentinel.models.bioassess import xtrack as X

REPO = paths.REPO_ROOT
OUT = REPO / "docs/preregistration/bioassess_x_freeze_candidate.json"
SEL = XD.DEV_DIR / "selection_x_v1.json"
CODE = ["docs/preregistration/bioassess_x_protocol.md", "configs/bioassess/bioset.yaml", "configs/bioassess/baselines.yaml",
        "sentinel/models/bioassess/xtrack.py", "sentinel/models/bioassess/xdata.py", "sentinel/models/bioassess/xdev.py",
        "sentinel/models/bioassess/xbioset.py", "sentinel/models/bioassess/xtestlook.py",
        "sentinel/models/bioassess/data.py", "sentinel/models/bioassess/uk.py",
        "sentinel/models/bioassess/features.py", "sentinel/models/bioassess/baselines.py",
        "sentinel/models/bioassess/bioset.py", "sentinel/models/bioassess/cv.py", "sentinel/models/bioassess/splits.py",
        "sentinel/models/bioassess/pretrain.py", "sentinel/models/bioassess/evaluate.py",
        "sentinel/models/bioassess/__init__.py", "sentinel/models/orbit/split.py",
        "sentinel/eval/splits.py", "sentinel/eval/bootstrap.py", "sentinel/eval/metrics.py", "sentinel/core/seed.py",
        "sentinel/core/registry.py", "sentinel/core/paths.py", "sentinel/core/gpu.py", "sentinel/core/runlog.py",
        "sentinel/models/chemtox/tabpfn_model.py", "scripts/eval/bioassess_x_final.py", "scripts/eval/bioassess_x_dev.py",
        "scripts/eval/bioassess_x_bioinfer.R", "scripts/eval/bioassess_x_gjam.R", "scripts/train/bioassess_x_bioset.py",
        "scripts/eval/bioassess_x_select.py", "scripts/eval/bioassess_x_freeze_candidate.py",
        "tests/test_bioassess_models.py", "tests/test_bioassess_x_dev.py", "sentinel/events/bioassess_x.py",
        "scripts/eval/bioassess_x_events.py", *X.FROZEN_SPLIT_FILES.values()]
RUN_FILES = [XD.XDIR / "splits_x_v1.parquet", XD.XDIR / "splits_x_v1.json", XD.XDIR / "pools_x_v1.json",
             XD.XDIR / "power_x_v1.json", XD.DEV_DIR / "folds_dev_v1.json", SEL]
DATA = ["processed/nars_streams/visits.parquet", "processed/nars_streams/benthic_counts.parquet",
        "processed/nars_streams/chemistry_wide.parquet", "processed/nars_streams/chemistry_long.parquet",
        "processed/nars_lakes/visits.parquet", "processed/nars_lakes/benthic_counts.parquet",
        "processed/nars_lakes/chemistry_wide.parquet", "processed/nars_lakes/chemistry_long.parquet",
        "processed/taxonomy_harmonised/taxon_map.parquet", "processed/uk_ea_ecology/sites.parquet",
        "processed/uk_ea_ecology/samples_invertebrate.parquet", "processed/uk_ea_ecology/counts_invertebrate.parquet",
        "processed/uk_ea_ecology/chemistry_matched_wide_pm30d_invertebrate.parquet",
        "processed/fuse_tier_c/rows.parquet", "processed/s2_wqp_matchups_hist/samples.parquet"]
PACKAGES = ["tabpfn", "torch", "scikit-learn", "lightgbm", "numpy", "pandas", "scipy", "joblib", "pyarrow", "PyYAML"]
TABPFN_CKPT = Path.home() / ".cache/tabpfn/tabpfn-v2-regressor.ckpt"
BASE_ENS = {"bens_rfg_tabpfn": ("rf_community_geography", "tabpfn_community"),
            "bens_rfg_gbm_tabpfn": ("rf_community_geography", "gbm_community", "tabpfn_community"),
            "bens_rfctx_tabpfn": ("rf_community_context", "tabpfn_community")}


def comparator_spec(name: str, choice: dict) -> dict:
    if name in BASE_ENS:
        return {"kind": "mean", "members": [comparator_spec(m, choice) for m in BASE_ENS[name]]}
    if name in ("bioinfer", "gjam"):
        return {"kind": "r", "method": name}
    return {"kind": "baseline", "name": name, "variant": dict(choice.get(name, {}))}


def sentinel_spec(name: str, blk: dict) -> dict:
    choice = blk["variant_choice"]
    if name.startswith("bsx_"):
        return {"kind": "bioset", "config": name}
    best = {"kind": "bioset", "config": blk["best_bioset"]}
    if name == "ens_c_form":
        return {"kind": "mean", "members": [best, comparator_spec("rf_community_geography", choice),
                                            comparator_spec("tabpfn_community", choice)]}
    if name == "ens_top2":
        return {"kind": "mean", "members": [best] + [comparator_spec(m, choice) for m in blk["ens_top2_members"]]}
    raise ValueError(name)


def _versions():
    import importlib.util

    spec = importlib.util.spec_from_file_location("x_final", REPO / "scripts/eval/bioassess_x_final.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.package_versions(PACKAGES), mod.r_package_versions()


def event_track(sel: dict) -> dict:
    """Secondary event track (amendment v8): specs, dev power, test-block capacity, frozen calibration data."""
    from sentinel.events import bioassess_x as XE

    evm = REPO / "sentinel/eval/events.py"
    devf, calf = XD.DEV_DIR / "events" / "events_dev_v1.json", XD.DEV_DIR / "events" / "events_calibration_v1.parquet"
    et = {"amendment": "v8", "role": "SECONDARY (never in the 12-test confirmatory family); scored in the same single "
                                    "M5-X look; original M5 test blocks are spent and unused",
          "events": {f"{b}|{c}": {"target": t, "shared_event_code": XE.EVENT_CODE[c],
                                  "direct_head": (b, c) in XE.POWERED,
                                  "status": "descriptive only" if (b, c) in XE.DESCRIPTIVE_ONLY else "secondary"}
                     for (b, c), (t, _, _) in XE.TRACK.items()},
          "descriptive_only_note": "Power rule (>= 30 positives and >= 10 clusters on the development pseudo-test) "
                                   "governs (amendment v8a: the lead corrected an earlier instruction that conflicted "
                                   "with it, before any event output existed). Powered: X-L E5b, X-L E6, X-U E6 (35/25). "
                                   "Descriptive only: X-U E2a (19/7), X-L E2a (11/11), X-L E3 (11/11).",
          "threshold_baselines": {b: {m: comparator_spec(m, sel["blocks"][b]["variant_choice"]) for m in XE.COMPARATORS}
                                  for b in XD.BLOCKS}}
    if not evm.is_file() or not devf.is_file() or not calf.is_file():
        et["status"] = ("BLOCKED: " + ("shared sentinel/eval/events.py not built yet; " if not evm.is_file() else "")
                        + ("development event run not done (waits for the lead's events.py v1 release notice)"
                           if not devf.is_file() else ""))
        return et
    dev = json.loads(devf.read_text())
    et.update({"status": "active", "events_module_sha256": file_sha256(evm), "calibration_file": str(calf),
               "calibration_sha256": file_sha256(calf), "dev_results_file": str(devf),
               "dev_results_sha256": file_sha256(devf)})
    cap = []
    for k, v in dev["events"].items():
        et["events"][k].update({"power_dev": v["power_dev"], "projection": v["projection"]})
        pr = v["projection"]["projected_test_positives"]
        cap.append(f"{k}: dev {v['power_dev']['n_pos']} positives / {v['power_dev']['n_clusters_pos']} clusters "
                   f"({v['power_dev']['flag']}); projected test positives ≈ {pr:.0f} of {v['projection']['n_test_split']} "
                   f"rows -> {'can carry it (secondary)' if (pr >= 30 and et['events'][k]['status'] == 'secondary') else 'descriptive only'}")
    et["test_block_capacity"] = ("Test labels unread; projection = development prevalence x development label "
                                 "availability x test rows. The plan's power rule is re-applied on the test rows inside "
                                 "the look. " + " | ".join(cap))
    return et


def build() -> dict:
    S, frozen, _ = XD.load_context()
    for b, fl in XD.read_folds(XD.DEV_DIR / "folds_dev_v1.json", S, frozen).items():
        for fo in fl:                                   # re-verify the development folds the selection used
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
    sel = json.loads(SEL.read_text())
    power = json.loads((XD.XDIR / "power_x_v1.json").read_text())
    gj = {}
    for b in XD.BLOCKS:
        f = XD.DEV_DIR / "r_out" / "gjam" / b / "fold00_info.txt"
        gj[b] = f.read_text() if f.exists() else "missing"
    rec = {"status": "candidate", "prepared": dt.datetime.now().isoformat(timespec="seconds"),
           "DISCLOSURE_GJAM_FIDELITY_AMENDMENT_V6": (
               "PROMINENT DISCLOSURE: the GJAM fidelity check pre-declared in amendment v4.8 was run first and FAILED "
               "(X-L fold 0: r = 0.43 TP, 0.52 TN, 0.54 chloride; NA for conductivity, turbidity, pH; "
               "x_track/dev/r_out/gjam/X-L/fold00_v4check_info.txt). Amendment v6, which replaces the comparison by "
               "our routine run in gjam's in-sample scheme, was written AFTER that failure was seen (but before any "
               "GJAM development prediction was scored). It can only add a comparator. Full-size v6 outcome per "
               "block is attached below; GJAM is a comparator only if v6 passed in both blocks."),
           "gjam_v6_fidelity_info": gj,
           "protocol": "docs/preregistration/bioassess_x_protocol.md", "seeds": [0, 1, 2], "n_boot": 2000,
           "n_boot_confirmatory": 10000, "alpha": 0.05,
           "family": "C-X vs per-target comparator on r2_anomaly, 2 blocks x 6 targets = 12 tests, Holm (alpha 0.05); "
                     "p = two-sided paired location-cluster bootstrap p if delta > 0 else 1",
           "decision_table": ["1 Holm-rejected & per-seed delta > 0 (seeds 0,1,2) & region-unit paired CI > 0 -> SOTA "
                              "(M5-X, block, within-region)", "2 Holm-rejected (row 1 not met) -> Above, not robust",
                              "3 CI upper < 0 -> Below", "4 CI includes 0 & realised Holm-worst MDE "
                              "(3.71 * CI width / 3.92) > 0.02 -> Inconclusive (underpowered)", "5 otherwise -> Competitive"],
           "power_x": power, "candidates": {}, "comparators": {}, "dev_evidence": {}}
    n_conf = {}
    for b in XD.BLOCKS:
        blk = sel["blocks"][b]
        sel_name, sec = blk["selected"][0], blk["secondary"]
        rec["candidates"][b] = {"confirmatory": {"name": sel_name, "spec": sentinel_spec(sel_name, blk)},
                                "secondary": [{"name": m, "spec": sentinel_spec(m, blk)} for m in sec]}
        rec["comparators"][b] = {t: {"name": v["model"], "spec": comparator_spec(v["model"], blk["variant_choice"]),
                                     "dev_r2_anomaly": v["dev_r2_anomaly"]}
                                 for t, v in blk["per_target_comparator"].items()}
        rec["dev_evidence"][b] = {k: blk[k] for k in ("sentinel_mean_r2_anomaly", "comparator_mean_r2_anomaly",
                                                       "paired_dev_vs_comparator", "n_pseudo_test")}
        n_conf[b] = blk["n_sentinel_configurations"]
    rec["event_track"] = event_track(sel)
    rec["n_configurations_tried"] = {**n_conf, "total": sum(n_conf.values()), "cap": 12,
                                     "rule": "counted per (configuration, block); amendment v4.9"}
    rec["selection_note"] = ("Selected by the pre-registered rule (highest mean development r2_anomaly over the block's 6 "
                             "targets, seed 0) among the SENTINEL configurations of amendment v4.9; development = "
                             "NLA 2017 LOEO (X-L) and UK 2016-20 leave-one-area-out (X-U) pseudo-tests on training-pool "
                             "rows only. Comparators: strongest of all §6 comparators per (block, target) by development "
                             "r2_anomaly, each in its chosen variant.")
    files = {rel: file_sha256(REPO / rel) for rel in CODE}
    files.update({str(f): file_sha256(f) for f in RUN_FILES})
    rec["sha256"] = files
    rec["data_sha256"] = {str(paths.DATA_ROOT / f): file_sha256(paths.DATA_ROOT / f) for f in DATA}
    rec["tabpfn_checkpoint"] = {"path": str(TABPFN_CKPT), "sha256": file_sha256(TABPFN_CKPT)}
    rec["sha256"][str(TABPFN_CKPT)] = rec["tabpfn_checkpoint"]["sha256"]
    rec["package_versions"], rec["r_package_versions"] = _versions()
    from sentinel.models.bioassess import xtestlook as XL

    rec["scoring_code_sha256"] = XL.scoring_code_sha256()
    rec["scoring_code_files"] = list(XL.SCORING_CODE)
    rec["data_snapshots"] = XL.make_data_pins()          # content-hash pins (manifest sha + exact loader files)
    rec["freeze_procedure"] = ("After the independent pre-test review, the LEAD writes "
                               "docs/preregistration/bioassess_x_freeze.json = this record with status 'frozen', "
                               "frozen_at and candidate_sha256 = sha256 of this file. The scorer accepts only that "
                               "file (xtestlook.load_freeze) and never this candidate.")
    rec["resume_rule"] = ("one resume after an infrastructure failure before the first bioassess_x_test registry row: "
                          "same record, same authorisation, same run directory; sha256-verified cached units are never "
                          "refitted (protocol §9)")
    rec["one_time_guards"] = {"marker": str(X.paths.RUNS_ROOT / "bioassess/x_track/x_final_scored.json"),
                              "claim_experiment": "bioassess_x_claim", "result_experiment": "bioassess_x_test",
                              "outputs": "sealed mode 0444 + SHA256SUMS"}
    rec["test_rows_note"] = ("test rows are loaded with the design rules of xdata (NLA: first visit with benthic taxa; "
                             "UK: >= 3 harmonised taxa); the numbers loaded vs in the split table are recorded in "
                             "pass_info.json")
    return rec


if __name__ == "__main__":
    r = build()
    OUT.write_text(json.dumps(r, indent=1, default=float))
    print(f"wrote {OUT}: {len(r['sha256'])} files, {len(r['data_sha256'])} data files; "
          f"configurations {r['n_configurations_tried']}")
