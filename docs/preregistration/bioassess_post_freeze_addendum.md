# M5 BioAssess — post-freeze addendum (not part of the hashed freeze record)

## A1 (2026-11-09, lead) — correction to the freeze record's exposure note
`bioassess_freeze.json` (frozen 2026-11-08 22:02 EST) states: "Test C (2023-24) was not scored by any
workstream at freeze." **This is incorrect.** D5 BioIntegrity's one-time test look scored NRSA 2023-24
(the same survey as M5 Test C) at 2026-11-09 01:28–01:32 UTC (2026-11-08 20:28–20:32 EST), about 1.5 h
before the M5 freeze. The D5 look loaded 2023-24 chemistry (M5's targets) into memory but did not use it as
a feature or save it (downstream post-results review, REVIEW_REPORT_POST_RESULTS.md).

Impact assessment: M5's candidates, comparators and settings were fixed on NRSA 2008-14 before any D5 test
activity, and the M5 freeze record was regenerated only for the reviewer's R1/R2/W-a/W-b items (no selection
change; record diff = hashes, seven newly hashed files, exposure_note, resume_rule, expected_runtime_h).
No M5 decision used D5's test output. The error is one of disclosure, not of selection; it is reported here
and must be reported alongside every Test C result and in the M5 post-results review. The frozen file is
left unchanged (editing it would invalidate the running single pass).

## A2 (2026-11-09, M5) — registry notes carrying the corrected sentence; Test C rows without a note
Following A1 and the post-results review (REVIEW_REPORT_POST_RESULTS.md, W-M5P-5):
- **1,749** of the 2,180 `bioassess_test` registry rows (every `nrsa1819/*` split of Tests A/B and the
  `ukpool->nrsa1819/transfer_test` T1 rows) carry the frozen `EXPOSURE_NOTE`, which ends "Test C (2023-24)
  was not scored by any workstream at freeze." That sentence is **incorrect** (see A1). The rows are **not**
  rewritten (registry rows are append-only and match the frozen scorer exactly); this entry and A1 are the
  correction and are referenced wherever M5 test numbers are reported.
- The **426** Test C rows (`nrsa2324/loeo_x_cycle_test`) carry **no** exposure note (the frozen scorer
  attaches the note only to 2018-19 rows). The A1 text applies to all of them: D5 BioIntegrity scored NRSA
  2023-24 about 1.5 h before the M5 freeze; no M5 decision used it; Test C is a secondary replication.
- Counts verified on `results/registry.jsonl` on 2026-11-09 (experiment `bioassess_test`).

## A3 (2026-11-09, M5) — POST-HOC, non-confirmatory comparison with bio.infer-style ML inference
Requested by the lead after the post-results review (W-M5P-4: the published EPA taxon–environment ML
inference, Yuan 2007 / R `bio.infer`, was not in the frozen comparator set). Not part of the frozen pass;
not Holm-adjusted; the frozen registry rows are unchanged.
- **Method:** re-implementation `sentinel/models/bioassess/bioinfer.py` (per-taxon logistic regression with
  linear + quadratic terms of the standardised variable, OTUs with ≥ 30 training occurrences, ML estimate
  on a 200-point grid over the training range; settings fixed before running, nothing tuned), fitted per
  ecoregion on the development rows (Test A protocol) and scored on the saved Test A rows
  (`pred_testA.parquet`, sha256 recorded). Script `scripts/eval/bioassess_posthoc_bioinfer.py`; outputs
  `$SENTINEL_RUNS/bioassess/posthoc_bioinfer_20261109T052447/posthoc_bioinfer_comparisons.json`; 18 registry
  rows, experiment `bioassess_test_posthoc` (6 × `r2`, 12 × `delta_r2_vs_bioinfer` for C and A).
- **Label rule (stated before running):** a SOTA label of candidate C stands on a target only if C's Δ vs
  bio.infer has a location-cluster CI above 0; otherwise it becomes "competitive".
- **Result:** bio.infer Test A R² −0.17 (conductivity) to −1.27 (TP) — raw ML inferences are over-dispersed
  (prediction SD 1.3–1.7× observed; correlation 0.46–0.59); the development-only LOEO diagnostic is
  equally negative (−0.29 to −1.60), so this is the method as implemented, not a test artefact. Even an
  oracle linear recalibration on the test rows would give at most R² 0.22–0.35, below C (0.41–0.57).
  C − bio.infer: Δ R² +0.74 to +1.68, all CIs (location cluster, 50-km block, ecoregion) > 0, all seeds
  positive → **the SOTA labels stand** (chloride, pH, conductivity; TN's rule outcome unchanged).
  Configuration count: +1 post-hoc comparator (bio.infer), one setting, run once.

## A4 (2026-11-09, lead + M5) — A3's bio.infer comparison is NOT evidence
The development LOEO sanity check (NRSA 2008-14): R² −0.29 to −1.60, far below WA-PLS2 (0.18–0.30; registry e.g. `9ec21e854822`, `62b4cb574e36`) and kNN-MAT (0.13–0.31). The raw ML inferences are over-dispersed on development data as on Test A, so the
re-implementation is not a faithful stand-in for the published method. A3's result ("C beats bio.infer 6/6,
labels unchanged") is **withdrawn as support**: our bio.infer-style re-implementation performed poorly; it may not reflect the EPA software; the SOTA label rests on the pre-registered comparators (WA-PLS2, kNN-MAT, RF, TabPFN, ensembles). The 18 `bioassess_test_posthoc` rows stay in the
registry (append-only) and are to be read with this entry. Published-method coverage: bio.infer is listed
as *not compared* (W-M5P-4).
