#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the M5 freeze CANDIDATE record (does not freeze; the lead freezes after review).

Writes ``docs/preregistration/bioassess_freeze_candidate.json`` with everything the single pass reads:

* ``candidates``; ``sota_comparators`` — per target, the strongest baseline on development LOEO (pooled
  R², current ``bioassess_phase1_cv`` entries) in the approved comparator set; ``h_m5_reference`` and
  ``h_m5_family_candidate``; ``controls_for`` (candidate C: leave-BioSet-out and equal-size baseline
  ensembles); ``models`` (everything the pass fits);
* statistical settings: ``tests`` (all blocks, one pass), ``confirmatory_test``, ``n_boot`` (2,000,
  marginal CIs and secondary Δs), ``n_boot_confirmatory`` (10,000), ``alpha``;
* provenance: ``sha256`` of every code file the predictions or scores depend on (M5 code, shared
  ``sentinel/eval`` + ``sentinel/core`` modules, the TabPFN shim, configs, protocol, scorer),
  ``data_sha256`` (NRSA, taxonomy and UK EA files, relative to ``$SENTINEL_DATA``),
  ``tabpfn_checkpoint`` (path + sha256), ``package_versions``;
* ``n_configurations_tried`` and ``selection_note`` (attached to every test registry row).

``scripts/eval/bioassess_final.py`` runs only from ``bioassess_freeze.json``, which the lead creates
from this reviewed record.

    python -B scripts/eval/bioassess_freeze_candidate.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256, registry_frame  # noqa: E402

CANDIDATES = ["BioSet[bioset_st_flat]",
              "Ensemble[BioSet[bioset_st_flat]+TabPFNv2[community]]",
              "Ensemble[BioSet[bioset_st_flat]+geography+RandomForest[community+geography]+TabPFNv2[community]]"]
COMPARATOR_SET = ["RandomForest[community]", "LightGBM[community]", "TabPFNv2[community]",
                  "RandomForest[community+geography]",
                  "Ensemble[RandomForest[community+geography]+TabPFNv2[community]]",
                  "Ensemble[LightGBM[community]+TabPFNv2[community]]",
                  # cycle 2: comparator ensembles as large as the candidate ensembles; context variants
                  "Ensemble[RandomForest[community+geography]+LightGBM[community]+TabPFNv2[community]]",
                  "RandomForest[community+context]",
                  "Ensemble[RandomForest[community+context]+TabPFNv2[community]]",
                  "Ensemble[RandomForest[community+context]+LightGBM[community]+TabPFNv2[community]]"]
CONTROLS = ["RandomForest[geography]", "RandomForest[context]", "EcoregionMean", "NearestReferenceSite",
            "kNN-MAT[community]", "WA-PLS2[community]",
            "BioSet[bioset_st_flat]+geography", "Ensemble[BioSet[bioset_st_flat]+geography+TabPFNv2[community]]"]
CONTROLS_FOR = {CANDIDATES[2]: {
    "leave_bioset_out": "Ensemble[RandomForest[community+geography]+TabPFNv2[community]]",
    "equal_size_baseline": "Ensemble[RandomForest[community+geography]+LightGBM[community]+TabPFNv2[community]]"}}
TARGETS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l", "log10_turbidity_ntu",
           "ph_lab")
HASHED = ["docs/preregistration/bioassess_protocol.md",
          "configs/bioassess/common.yaml", "configs/bioassess/bioset.yaml", "configs/bioassess/baselines.yaml",
          "sentinel/models/bioassess/__init__.py", "sentinel/models/bioassess/data.py",
          "sentinel/models/bioassess/splits.py", "sentinel/models/bioassess/features.py",
          "sentinel/models/bioassess/baselines.py", "sentinel/models/bioassess/bioset.py",
          "sentinel/models/bioassess/cv.py", "sentinel/models/bioassess/evaluate.py",
          "sentinel/models/bioassess/transfer.py", "sentinel/models/bioassess/uk.py",
          "sentinel/eval/splits.py", "sentinel/eval/bootstrap.py", "sentinel/eval/metrics.py",
          "sentinel/core/seed.py", "sentinel/core/registry.py", "sentinel/core/paths.py", "sentinel/core/gpu.py",
          "sentinel/core/runlog.py", "sentinel/models/chemtox/tabpfn_model.py",
          # R2 (delta re-review 2026-11-08): the rest of the scorer's import chain (M4's chemtox package
          # is imported via the TabPFN shim; trivial package __init__ files)
          "sentinel/models/chemtox/__init__.py", "sentinel/models/chemtox/evaluate.py",
          "sentinel/models/chemtox/features.py", "sentinel/__init__.py", "sentinel/core/__init__.py",
          "sentinel/eval/__init__.py", "sentinel/models/__init__.py",
          "scripts/eval/bioassess_final.py"]
DATA_FILES = ["processed/nars_streams/visits.parquet", "processed/nars_streams/benthic_counts.parquet",
              "processed/nars_streams/chemistry_wide.parquet", "processed/nars_streams/chemistry_long.parquet",
              "processed/taxonomy_harmonised/taxon_map.parquet",
              "processed/uk_ea_ecology/sites.parquet", "processed/uk_ea_ecology/samples_invertebrate.parquet",
              "processed/uk_ea_ecology/counts_invertebrate.parquet",
              "processed/uk_ea_ecology/chemistry_matched_wide_pm30d_invertebrate.parquet"]
PACKAGES = ["tabpfn", "torch", "scikit-learn", "lightgbm", "numpy", "pandas", "scipy", "joblib", "pyarrow", "PyYAML"]
TABPFN_CKPT = Path.home() / ".cache/tabpfn/tabpfn-v2-regressor.ckpt"


def main() -> int:
    from importlib.metadata import version

    df = registry_frame()
    df = df[(df["experiment"] == "bioassess_phase1_cv") & (df["split"] == "nrsa0814/loeo_cv") & (df["metric"] == "r2")]
    df = df.sort_values("timestamp").groupby(["model", "target"], as_index=False).last()
    missing = [m for m in COMPARATOR_SET + CANDIDATES if m not in set(df["model"])]
    if missing:
        raise SystemExit(f"no development LOEO entries for {missing}")
    comp, evidence = {}, {}
    for t in TARGETS:
        sub = df[(df["target"] == t) & df["model"].isin(COMPARATOR_SET)].sort_values("value", ascending=False)
        comp[t] = sub.iloc[0]["model"]
        evidence[t] = {r.model: {"r2": float(r.value), "registry_id": r.id} for r in sub.itertuples()}
    models = list(dict.fromkeys(CANDIDATES + COMPARATOR_SET + CONTROLS))
    rec = {
        "status": "CANDIDATE — not frozen; the lead creates bioassess_freeze.json after review",
        "prepared": dt.datetime.now().isoformat(timespec="seconds"),
        "candidates": CANDIDATES, "sota_comparators": comp, "comparator_evidence_dev_loeo": evidence,
        "h_m5_reference": "RandomForest[community]", "h_m5_family_candidate": CANDIDATES[0],
        "controls_for": CONTROLS_FOR, "models": models,
        "tests": ["A", "B", "C", "T"], "confirmatory_test": "A",
        "n_boot": 2000, "n_boot_confirmatory": 10000, "alpha": 0.05,
        "families": {"sota": f"Holm over {len(CANDIDATES)} candidates x 6 targets = {len(CANDIDATES) * 6} tests "
                             "(Test A only)",
                     "h_m5": "Holm over model A x 6 targets = 6 tests (Test A only); B and C vs RF[community] "
                             "are reported as secondary"},
        "decision_rule": ("one-sided: p = two-sided paired-bootstrap p if delta > 0 else 1 (one-sided at alpha/2); "
                          "SOTA label for (candidate, target) only if Holm-rejected in the SOTA family AND the "
                          "per-seed delta is > 0 for seeds 0, 1 and 2; candidate C's 'adds signal' framing only "
                          "where delta > 0 against both its leave-BioSet-out and equal-size controls (secondary)"),
        "n_configurations_tried": {"sentinel_side": 13, "baseline_side_additions": 7,
                                   "detail": "protocol §10 (cycles 1-2)"},
        "selection_note": ("candidates selected on development LOEO (NRSA 2008-14) after 13 SENTINEL "
                           "configurations; development scores are selection-biased"),
        "sha256": {f: file_sha256(REPO / f) for f in HASHED},
        "data_sha256": {f: file_sha256(paths.DATA_ROOT / f) for f in DATA_FILES},
        "tabpfn_checkpoint": {"path": str(TABPFN_CKPT), "sha256": file_sha256(TABPFN_CKPT)},
        "package_versions": {p: version(p) for p in PACKAGES},
        "lead_decisions_2026_10_24": {
            "admitted_candidates": CANDIDATES,
            "claim_framing": {
                CANDIDATES[0]: "H-M5: above the blocked RF[community] (Holm, 6 tests); SOTA only where it beats "
                               "the per-target strongest baseline (Holm, 18 tests) with all seeds positive",
                CANDIDATES[1]: "SOTA only where it beats the per-target strongest baseline (Holm, 18 tests) with "
                               "all seeds positive",
                CANDIDATES[2]: "BioSet adds signal beyond the strongest baselines (contains RF[community+geography] "
                               "as a member); equal-size baseline-ensemble control stays in the comparator set"}},
        "exposure_note": ("NRSA 2018-19 (Tests A/B, T1) was used before the pass (i) by the research-phase RF "
                          "baseline (rf_blocked_baseline) and (ii) from 2026-10-25 03:43 UTC as the validation cycle "
                          "of D5 BioIntegrity (benthic condition labels from the same samples; one oracle, "
                          "oracle_gbm_chem, takes 2018-19 chemistry as input). No M5 setting was chosen on it; M5's "
                          "candidates, comparators and settings were fixed on NRSA 2008-14 before (ii). Test C "
                          "(2023-24) was not scored by any workstream at freeze."),
        "resume_rule": ("one resume after an infrastructure failure before the first registry row: same record, "
                        "same authorisation, from the sha256-verified per-unit prediction cache; cached units are "
                        "never refitted (protocol §9.1)"),
        "expected_runtime_h": "6-9 (one GPU)",
    }
    out = REPO / "docs/preregistration/bioassess_freeze_candidate.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("candidates", "sota_comparators", "families")}, indent=1))
    print(f"{len(rec['sha256'])} code files, {len(rec['data_sha256'])} data files hashed; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
