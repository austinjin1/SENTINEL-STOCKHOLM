# Handoff — M5 SENTINEL-BioAssess (updated 2026-11-09 ~06:30 EST)

## State: TEST PASS DONE (2026-11-08 22:30 – 11-09 04:45, lead-run) and WRITTEN UP; nothing running
- Run dir `$SENTINEL_RUNS/bioassess/final_20261108T220242/` (253 units fitted, no resume); 2,180 rows
  `bioassess_test`; post-results review 🟡 (REVIEW_REPORT_POST_RESULTS.md, "M5 BioAssess — post-results review").
- **Final labels (Test A = leave-one-ecoregion-out × cycle-out, NRSA 2018-19; belong to ensemble C =
  mean(BioSet+geography, RF[community+geography], TabPFNv2)):**
  - SOTA (as ensemble C, LOEO scope): chloride +0.021 [+0.013, +0.028], pH +0.024 [+0.015, +0.032],
    conductivity +0.013 [+0.006, +0.019]; RMSE −2.0 / −2.2 / −1.5 %; replicated on Test C (2023-24).
  - TN: competitive (SOTA by the pre-registered rule, not robust: ecoregion CI [−0.013, +0.030], 4/9 + , WMT-driven).
  - TP, turbidity: competitive.
  - BioSet alone significantly WORSE than the comparator on TN (−0.059), cond (−0.038), Cl (−0.031), pH (−0.036).
  - Test B (cycle-out): C's gain ≈ 0; RF[geography] best of 21 on TN/cond/Cl → community signal helps out of region.
  - H-M5 met 6/6 (Holm, seed-averaged); 4/6 all seeds +; Test C 5/6 (chloride all seeds negative).
  - T1/T2 transfer negative.
  - Post hoc (A3/A4): our bio.infer-style re-implementation performed poorly; it may not reflect the EPA software; the SOTA label rests on the pre-registered comparators (WA-PLS2, kNN-MAT, RF, TabPFN, ensembles); dev LOEO −0.29 to −1.60 vs WA-PLS2 0.18–0.30 → not evidence; bio.infer = not compared.
- Write-up done: RESULTS.md (2026-11-09 entry), model card (v0.2.0), test-results file docs/preregistration/bioassess_test_results.md (formerly appended as protocol §11, moved by the lead 2026-11-09; changes the
  protocol's hash vs the frozen record — frozen version is in the freeze archive), addendum A2 (1,749 stale
  "Test C not scored" notes; 426 Test C rows without a note; rows not rewritten) and A3 (bio.infer),
  `docs/novelty_checks/bioassess_2026-11-09.md`, TRAINING_LOG manual entry. Tables:
  `scripts/eval/bioassess_writeup_tables.py` → `$SENTINEL_RUNS/bioassess/writeup_20261109/`.
- Freeze record and scorer untouched after the pass. Never re-run the test pass.
- **M5-X follow-on track — DEVELOPMENT STARTED (final confirmation PASSED 2026-11-09).** Protocol
  `docs/preregistration/bioassess_x_protocol.md` v3, snapshot `1a0c7f4b8331`; split `x_track/splits_x_v1.parquet`
  (sha256 18a35ef3…, unchanged). **Configurations tried on track X: 0 (no model fitted yet).**
  Done in development:
  - Lead's pre-fit condition: static test `test_m5x_static_entry_points_use_fit_guard_and_guarded_encoder_loader`
    (every `scripts/**/bioassess_x_*.py` must call `assert_fit_ids(`, except the allowlisted non-fitting
    `bioassess_x_prereg.py`/`bioassess_x_power.py`; no M5-X file may use `PretrainedEncoder(`, `torch.load(`,
    `resolve_pretrained`, `uk_pretrain_latest`, `init_encoder=`; `pretrained=` needs `require_guarded(`); verified it
    fails on a bad probe script. `xtrack.load_pretrained_encoder` marks encoders; `xtrack.require_guarded` checks.
  - bio.infer fidelity PASSED (`scripts/eval/bioassess_x_bioinfer_fidelity.R` →
    `x_track/fidelity/bioinfer_fidelity.txt`): bundled `makess` exact (max diff 0), documented `mlsolve`/`taxon.env`
    examples run headless, 99.6 % consistent solutions, inferred vs observed on the Oregon example r = 0.63 (fine
    sediment), 0.68 (temperature), n = 245. → bio.infer stays an eligible comparator.
  - GJAM 2.7 installed in `rbioinfer` (needs `PATH=$E/bin:$PATH` for compilation; env 1.2 GB). GJAM fidelity
    check (our out-of-sample inverse routine vs gjam's in-sample `xpredMu`) NOT yet done.
  - Unified loader `sentinel/models/bioassess/xdata.py::load_x_design(S, frozen)`: all allowed pool rows
    (204,026: UK 198,484 [~48k with chemistry], NRSA 2008–14 3,946, NLA 2007–17 1,596); protected ids dropped
    before chemistry is read (parquet row filters); `assert_fit_ids` on load; BioDesign-compatible (ag_eco9 =
    region; elevation_m = UK site altitude; UK context/indices/analysis_method columns). Load ≈ 5 min CPU.
  Next (in order): (1) dev-design fold builder (X-L pseudo-test NLA 2017 LOEO; X-U pseudo-test UK 2016–20
  pool_fit leave-area-out; frozen/region/1-km rules) in a `scripts/eval/bioassess_x_dev.py` that calls
  `assert_fit_ids`; (2) baselines on dev folds (RF/LGBM ≤ 60k rows, TabPFN ≤ 6k, WA-PLS2/kNN pool choice, index
  baselines X-U, region mean, RF[geography]) → registry experiment `bioassess_x_dev`; (3) bio.infer on dev folds
  (R, its own OTU pipeline; ITIS coverage), GJAM fidelity + dev; (4) ≤ 12 BioSet configs (GPU 0/1 ≤ 2 h chunks);
  (5) selection → per-target comparators → M5-X scorer (bioassess_final pattern) → dev dry run → freeze candidate →
  report for pre-test review.
- **Literature:** extended publisher-restricted web pass done (novelty check §"Literature pass 2"); Scopus not
  accessible to agents — a human Scopus/Scholar check is still required before SOTA wording leaves the repo.

## Pre-test review fixes (REVIEW_REPORT_PRE_TRAINING.md, "M5 BioAssess — pre-test gate")

### Done in code (32 unit tests pass)
- **C1 per-seed results.** All stochastic models keep seeds 0–2. An ensemble's seed-s prediction is
  the mean of its members' seed-s fits. Per-seed R², per-seed Δ and `all_seeds_positive` are logged.
  `pred_test*.parquet` holds the per-seed columns. `--score-from-predictions RUN_DIR` re-scores
  A, B and C (W8).
- **C2 Holm families on Test A only, logged to the registry.** SOTA family: 3 candidates × 6 targets
  = 18 tests. H-M5 family: A × 6 = 6 tests. Metrics `holm_rejected_*` carry the raw p, the adjusted p
  and α. The other blocks are labelled "secondary, unadjusted". The claim table is in protocol §9.1,
  and the stale "12 tests" text is removed.
- **C3 hashing.** The candidate record now hashes:
  - 24 code files, including shared `sentinel/eval/{splits,bootstrap,metrics}`,
    `sentinel/core/{seed,registry,paths,gpu,runlog}` and `chemtox/tabpfn_model.py`;
  - 9 data files (NRSA, taxonomy, UK EA);
  - the TabPFN checkpoint;
  - 10 package versions.

  `verify_record`/`verify_freeze` check all of them. **Still open:** the M5 tree is untracked in git.
  The lead should commit it or archive it at freeze.
- **C4.**
  - The marker is written only after imports, the GPU lock and the data load.
  - NaN metrics are recorded as not estimable (`SafeLog`).
  - The scorer refuses if the registry already holds a `bioassess_test` row (W9).
  - Unit tests cover the second-pass, changed-hash and changed-package refusals and the synthetic
    scoring path.
  - The `--dry-run-dev` mode exists. **Pending:** the dry-run and reproduction runs above.
- **C5.** The test list, B (2,000 marginal; 10,000 confirmatory) and α come from the record only; there
  are no CLI overrides; all blocks run in one pass.
- **W1** the vocabulary is built from development tokens only.
- **W2** C is compared with its leave-BioSet-out and equal-size controls (`controls_for`).
- **W3** B = 10,000 for confirmatory tests.
- **W4** 50-km-block and ecoregion paired-Δ sensitivity, logged in `extra`.
- **W5** the T1/T2 deviation is documented in §10; T1/T2 use 3 seeds; T1 is compared with
  RF[geography].
- **W6** an exposure note is attached to every 2018-19 test row; `baseline_bioassess_rf.py` now needs
  `--i-know-2018-19-is-a-test-cycle`.
- **W7** configuration counts and the selection note are attached to every test row.
- **W10** `is_baseline` is false for SENTINEL controls.
- **W11** the R10 process deviation is disclosed in RESULTS.md and protocol §10.
- **W12** the two `python -c` rows were re-logged via `scripts/eval/bioassess_pair_cv.py`
  (`2bda4e42645a` and `ac57eaee0a78` supersede `dc08e7e83e36` and `2f821dd3afc4`).
- `sentinel/models/bioassess/data.py` was **not** changed. The D5 coupling is intact.

## Changed files since the review (for the diff summary)
- `scripts/eval/bioassess_final.py` (rewritten)
- `scripts/eval/bioassess_freeze_candidate.py` (expanded hashing and settings)
- `scripts/eval/bioassess_pair_cv.py` (new)
- `scripts/eval/baseline_bioassess_rf.py` (guard flag only)
- `tests/test_bioassess_models.py` (3 new scorer tests)
- `docs/preregistration/bioassess_protocol.md` (§9.1 claim table and scorer description; §10
  deviations)
- `sentinel/models/bioassess/cv.py` (2026-11-08: `_glob` fold-unobserved geography fix + warning)
- `docs/preregistration/bioassess_freeze_candidate.json` (regenerated 2026-11-08 after the dry run)
- `RESULTS.md` (M5 line: R10 disclosure)

## Results status (development CV only; the test is untouched)
- Registry experiments: `bioassess_phase1_cv`, `bioassess_phase1_selection`,
  `bioassess_phase1_transfer_dev`.
- TabPFN block50 finished at 01:54 and is logged.
- Candidates: A `BioSet[bioset_st_flat]`, B `Ensemble[BioSet+TabPFN]`, C
  `Ensemble[BioSet+geo + RF+geo + TabPFN]`. The lead admitted all three.

## Open issues / next steps
- Saturation (§15.5 items 5–7): NOT saturated by the ceiling test (best dev LOEO R² 0.36–0.58), but margins over the
  strongest baselines are small (+0.02–0.03 for ensembles; A alone below them). Proposed follow-on track, separately
  pre-registered AFTER the test pass, never replacing it: larger data (UK EA 280,992 samples; NRSA 2023-24 + NLA
  after their test use), within-ecoregion skill (r2_anomaly) as the a-priori hard stratum; own protocol section,
  frozen split sha256 and power check before any selection; size estimate to lead first if > 10 GB.
- Data request to the data workstream: StreamCat covariates at NRSA sites. 2008-09 has no COMIDs, so
  NLDI lookups come first. `prg_bmmi0809` must stay excluded.
- After re-review and the lead's freeze, the lead runs or authorises the single test pass.
- GPU: lock-helper cap (GPUs 0/1 since 2026-11-08; test-critical jobs first). CPU: one heavy job, ≤ 6 threads.

## 2026-11-08 22:02 — lead freeze (R3 done without git)
- `docs/preregistration/bioassess_freeze.json` written (read-only; copy of the candidate + status/frozen_at/candidate_sha256).
- Archive: `$SENTINEL_RUNS/freeze_archives/bioassess_freeze_20261108T220227.tar.gz` (34 files),
  sha256 `80af567e1c219265334661c5ba3c3848104f52061ed4d1e6f9858113ccbeca18`; registry row `bioassess/prereg_snapshot`.
- Single test pass launched 22:03 (PID 1125635), log `$SENTINEL_RUNS/bioassess/logs/final_test_pass_20261108.log`,
  authorised-by "lead (Claude) 2026-11-08, freeze archive 80af567e". Waits for a GPU 0/1 lock.
- Do NOT edit any hashed file (incl. M4 `chemtox/{__init__,evaluate,features}.py`) until the pass has finished.

## 2026-11-09 07:00–09:30 — M5-X development (follow-on engineer); status: DEV RUNNING, nothing frozen, no test touched
**Protocol:** amendments v4 (implementation details, written before any fit), v5 (implementation notes), v6 (GJAM
fidelity correction, disclosed as written after the first check failed) appended to
`docs/preregistration/bioassess_x_protocol.md` §10. Ordering evidence: registry `bioassess_x_dev/folds`
(`9be524896dd5`, 11:45 UTC, before the first fit) and `bioassess_x/prereg_snapshot` "amendment_snapshot" (13 files,
13:36 UTC; the v6 disclosure sentence was added after it — the freeze candidate hashes the final text).
**Configurations tried on track X: 0 scored** (4 BioSet configs × 2 blocks defined in v4.9; 1 partly fitted, see below).

### New code (frozen M5 files untouched; static test + 17 new tests green)
- `sentinel/models/bioassess/xdev.py` — context cache (`load_context`:97, design pickle + frozen-key cache),
  dev folds `dev_folds`:154 (v4.1: dev rule ∩ test-design pool − cycle/time-out − 1-km/cluster buffer;
  `assert_fit_ids` on every fold), multi-source labels `labels`:203, sparse `XFeaturizer`:274 (= M5
  CommunityFeaturizer semantics, unit-tested equal), all §6 tabular comparators `fit_predict`:384 (M5 fixed
  hyperparameters, row budgets), test-design loaders `load_test_rows`:451 / `test_folds`:485 (scorer only).
- `sentinel/models/bioassess/xbioset.py` — BioSet configs `CONFIGS`, source/method tokens, union labels,
  `pretrain_fold`:145 (per-fold UK pretraining, sidecar from ids used), `fit_fold`:179 (encoder only via
  `load_pretrained_encoder` + `require_guarded`), `_chunk_nograd`:98 (frozen `fit_bioset` validates in one batch →
  CUDA OOM/invalid-argument on X-U; no-grad forwards now chunked, unit-tested identical).
- `scripts/eval/bioassess_x_dev.py` — `prepare`:76, `baselines`/`run_model`:99, `export-r`:147, `ingest-r`:181,
  `score`:206 (registry `bioassess_x_dev`, logged once per prediction sha), `snapshot`:256.
- `scripts/train/bioassess_x_bioset.py` — per-(block, config) GPU chunks ≤ 105 min, per-fold parts.
- `scripts/eval/bioassess_x_bioinfer.R` — bio.infer run itself: `headless_defaults`:18 (assignInNamespace: no
  manual correction, first ITIS candidate), `taxon.env.fast`:27 (two loops vectorised; **bit-identical** on the
  Oregon example and a dev fold: `x_track/dev/r_out/bioinfer/bioinfer_taxon_env_fast_equivalence.txt`); ITIS matched
  2,358/2,609 names.
- `scripts/eval/bioassess_x_gjam.R` — gjam 2.7 fit (PA, 100 taxa, ≤ 5,000 rows, ng 2000/burnin 500) + our Gibbs
  inverse `inverse`:68 (out-of-sample and gjam's in-sample scheme). Fidelity: v4.8 check as executed FAILED
  (X-L fold 0: r 0.43/0.52/0.54, NA ×3; `r_out/gjam/X-L/fold00_v4check_info.txt`); cause = gjam's `xpredMu`
  conditions w on the observed x. v6 check (in-sample scheme) on a 400-row smoke: r 0.992–0.9999 → routine faithful;
  the full-size v6 check runs on fold 0 of both blocks in the R chain. GJAM fit ≈ 87 min/fold (≤ 2 CPU-h rule met).
- `scripts/eval/bioassess_x_select.py` — variants per (block, target), baseline ensembles, SENTINEL ensembles
  (C-form, top-2), per-target comparator, selection rule, paired dev Δ (B = 2,000) → `x_track/dev/selection_x_v1.json`.
- `scripts/eval/bioassess_x_final.py` — ONE-TIME scorer: fixed marker `x_track/x_final_scored.json` (O_EXCL,
  :413), resume marker :417, registry refusal `verify_freeze`:125, record/package (Python+R) verification
  `verify_record`:104, **claim-time registry row** `bioassess_x_claim` :420 (written after setup, before test rows
  load), per-unit sha256 cache, scoring `score_blocks`:286 (Holm-12, per-seed, region-unit and 50-km CIs, sign
  counts, realised MDE, decision table `label`:270), secondary :348, single registry flush, `seal`:383 (0444 +
  SHA256SUMS). `--dry-run-dev [--max-folds N] [--cpu]`. **Smoke dry run passed** (1 fold/block, cheap provisional
  record incl. bio.infer R path; `x_track/dev/smoke_x_final_20261109T071901`, scratch registry; main registry has 0
  `bioassess_x_claim/_test` rows).
- `scripts/eval/bioassess_x_freeze_candidate.py` — writes `docs/preregistration/bioassess_x_freeze_candidate.json`
  (executable specs, 12-test family, decision table, config counts, hashes of 41 code/run files, 15 data files,
  TabPFN weights, Python + R versions). **Never writes a FREEZE file.**
- `tests/test_bioassess_x_dev.py` — 11 tests (featurizer equivalence, subsample, labels, union mapping, decision
  table, Holm, scorer refusals, record verification, cache tamper + seal, chunked forward).

### Development folds (`x_track/dev/folds_dev_v1.json`)
X-L: 9 ecoregion folds, 415 NLA-2017 pseudo-test visits (allowed after frozen exclusions), training 202,946–203,218
rows. X-U: 19 area folds, 4,788 UK 2016–20 pseudo-test samples, training 173,896–188,366 rows. Buffer removed 3 / 57.

### Running now (all resumable; my processes only; ≤ 6 CPU threads; GPU via the lock on GPU 0/1)
- GPU chain `x_track/dev/gpu_chain.sh` (PID 1316101) then retry chain `gpu_chain2.sh` (1334143): TabPFN X-L →
  BioSet X-L flat_geo → TabPFN X-U → rloss/same/ukpt × 2 blocks; X-U `bsx_flat_geo` folds 0–7 done, folds 8–18 rerun
  by chain 2 (chunked validation fix). Each chunk ≤ 105 min, 2-min pause between chunks.
- CPU chain `cpu_chain.sh` (1320062, 5 threads): X-U RF same → X-L multi → X-U multi. X-U RF ≈ 8 min/fold.
- R chain `r_chain3.sh` (1352269, 1 thread): bio.infer all folds, then GJAM pool (1 worker while the CPU chain runs,
  6 after), fidelity on fold 0 of both blocks.
- `finish.sh` (1354172) waits for all three, checks completeness (stops if anything is missing), ingests R outputs
  (GJAM only if the v6 fidelity passed in both blocks), `score --log`, `select --log`, logs ensembles, writes the
  freeze-candidate record, then the scorer's dev dry run (`--max-folds 1`, 3 seeds, GPU lock). Logs:
  `x_track/dev/logs/{finish,dryrun_x_final}.log`.

### Next steps (successor)
1. Check `x_track/dev/logs/gpu_chain.log` / `finish.log`. If a chain item FAILED, rerun it (scripts are resumable).
2. After `finish.sh`: read `selection_x_v1.json`, `dev_scores.csv`, the candidate record and the dry-run sealed dir;
   append the configuration counts (cap 12, counted per block) and dev CIs here; report for the independent
   pre-test review. Do NOT create `bioassess_x_freeze.json`; do NOT run the scorer without `--dry-run-dev`.
3. Open: GJAM v6 full-size fidelity outcome; TRPI excluded (not obtained); human Scopus/Scholar check before any
   SOTA wording; record snapshot again after v6 disclosure edit (freeze candidate hashes the final protocol).

### Interim development results (09:40; seed 0; the 135 finished (model, variant, target) rows were logged to registry `bioassess_x_dev` at 09:56, r2_anomaly with 95 % location-cluster CI, B = 2,000; `x_track/dev/dev_scores.csv`)
- **X-L (415 NLA-2017 pseudo-test visits, 9 folds), mean over 6 targets:** TabPFNv2[community] (same-continent)
  **0.367** (per target 0.30–0.47; e.g. TP 0.402 [0.294, 0.494], cond 0.472 [0.372, 0.563]) > LightGBM 0.310 >
  RF[community] 0.277 > **BioSet `bsx_flat_geo` 0.262** (TP 0.270 [0.154, 0.366], TN 0.175 [0.017, 0.307], cond 0.427
  [0.316, 0.524], Cl 0.304 [0.193, 0.400], turbidity 0.089 [−0.061, 0.203], pH 0.309 [0.201, 0.419]) ≈ RF[c+g] 0.262 >
  WA-PLS2 0.11–0.14 > kNN-MAT 0.06–0.09 > RF[geography] 0.044 > region mean 0 > nearest reference −0.41.
  **bio.infer (run itself): compared, weak** — −0.30 to −2.34 (all CIs < 0). Multi-source variants are worse than
  same-continent for TabPFN (0.05–0.25) and similar/worse for WA-PLS2/kNN.
- **X-U (4,788 UK 2016–20 samples, 19 folds), so far:** UK indices RF+geography 0.295, LightGBM 0.286, indices
  linear+geography 0.275, WA-PLS2 0.170, RF[geography] 0.171; RF[community] variants, TabPFN, BioSet (0/4 complete:
  `bsx_flat_geo` 8/19 folds), bio.infer and GJAM still running.
- **Configurations tried (counted per block, cap 12): X-L 1 (bsx_flat_geo), X-U 1 (bsx_flat_geo, partial) = 2 of
  ≤ 12 defined (4 BioSet + 2 ensembles per block).** Baseline side: 16 comparator models (+ variants) per block.
- **ETA:** CPU chain ≈ 15 h (X-U RF ≈ 8 min/fold), GPU chain ≈ 12–14 h, GJAM ≈ 27 folds × ~1.5 h (1 worker during the
  CPU chain, 6 after) → development complete ≈ 2026-11-10 morning; `finish.sh` then selects, writes the freeze
  candidate and runs the dry run (~1–2 h). Freeze candidate ready for the pre-test review ≈ 2026-11-10 midday.

### 2026-11-09 ~10:30 — lead follow-up: single-look guard (M7 lesson) implemented
- `sentinel/models/bioassess/xtestlook.py` (new; hydromet testlook pattern): `load_freeze` accepts ONLY
  `docs/preregistration/bioassess_x_freeze.json` (refuses the candidate path, status ≠ frozen, candidate-sha or
  `scoring_code_sha256` mismatch); `make_data_pins`/`verify_data_pins` (content hash over the exact loader files of 6
  datasets + manifest sha); `claim` (registry refusal + O_EXCL `TEST_LOOK.json` + claim row); `require_open_claim`
  (called by `xdev.load_test_rows`); `claim_resume` (single, before sealing); `seal_predictions`/`check_sealed`;
  `mark_scored` (O_EXCL `SCORED.json`).
- `scripts/eval/bioassess_x_final.py` rewired (`verify_freeze`, `setup`, `design_and_folds`, `predict_all`,
  `score_saved`, `run_pass`, `main`; adds `--rescore-from-saved`). Dry run uses the candidate with its own look dir +
  scratch registry. CPU smoke of the new path passed (`x_track/dev/smoke_x_final_v2/look/{TEST_LOOK,PREDICTIONS_SAVED,
  SCORED}.json`); real look dir absent; main registry 0 `bioassess_x_test` rows.
- Freeze-candidate builder now writes `scoring_code_sha256`, `data_snapshots` (pins), `freeze_procedure`, and a
  top-level `DISCLOSURE_GJAM_FIDELITY_AMENDMENT_V6` with the per-block v6 outcome attached.
- Tests: `tests/test_bioassess_x_dev.py` 14 pass (5 new guard tests); M5-X static test green. Protocol v7 note added.

### 2026-11-09 ~13:00 — lead follow-up: secondary EVENT TRACK folded into M5-X (amendment v8; no event result computed)
- Pre-registered as protocol amendment v8 before any event prediction/metric (only dev label counts looked at).
  Events: E2a Cl > 230 mg/L, E5b TP > 0.1 mg/L, E6 pH ∉ [6.5, 9], E3 turbidity > 50 NTU (X-L); E2a, E6 (X-U).
- M5-X dev power (≥ 30 pos & ≥ 10 clusters): X-L E5b 105/105 F, E6 46/46 F, E2a 11/11 U, E3 11/11 U; X-U E2a 19/7 U,
  E6 35/25 (passes on M5-X dev, **descriptive only by lead ruling** — plan counts 4 and 10). Direct head only for
  X-L E5b and E6.
- Routes: derived = Gaussian on the target (dev OOF residual SD) → `events.exceedance_probability` → Platt via
  `events.recalibrate` (dev only; cross-fitted for dev numbers, frozen for test); direct = LightGBM classifier.
  Comparators: threshold-on-baseline RF[geography] and TabPFNv2, prevalence, month climatology. Metrics:
  `events.state_metrics`, `events.compare_routes` (location-cluster B = 2,000).
- Code: `sentinel/events/bioassess_x.py` (adapter; imports `sentinel.eval.events`, copies nothing),
  `scripts/eval/bioassess_x_events.py` (dev run → `x_track/dev/events/{events_dev_v1.json,
  events_calibration_v1.parquet}`), scorer `predict_events` / `score_events` (direct + climatology units sealed with
  the other predictions; events scored in the same single look, registry metrics `event_*`, labelled SECONDARY),
  candidate `event_track` block with the **test-block capacity** statement (projection from dev prevalence; test
  labels unread). Scoring-code hash now covers the adapter and `sentinel/eval/events.py`.
- **BLOCKER:** `sentinel/eval/events.py` does not exist yet (being built). The adapter is written against the plan
  §1.3 API (`EVENTS`, `event_state`, `exceedance_probability(pred, ev)` with a Gaussian/log10 spec dict,
  `recalibrate`, `state_metrics`, `compare_routes`, `power_flag`); check the call shapes when it lands
  (`test_x_event_derived_probability_monotone_with_shared_module` is skipped until then). `finish.sh` (PID 1421199)
  runs the event dev step only if the module exists; otherwise the candidate's `event_track.status` = BLOCKED and
  `bioassess_x_events.py --log` + `bioassess_x_freeze_candidate.py` must be re-run before the review.
- Tests: `tests/test_bioassess_x_dev.py` 16 pass + 1 skipped; M5-X static test green.

### 2026-11-09 12:36 — amendment v8a (lead correction): X-U E6 powered
- The pre-registered power rule governs; the lead's earlier "UK pH descriptive only" conflicted with it and was
  corrected before any event output existed (no `x_track/dev/events/`, 0 `bioassess_x_events_dev` rows).
  Powered (direct head + derived-vs-direct Holm): X-L E5b, X-L E6, **X-U E6** (35/25). Descriptive only: X-U E2a,
  X-L E2a, X-L E3. Code: `sentinel/events/bioassess_x.py` `POWERED`/`DESCRIPTIVE_ONLY`; candidate note updated.
- Adapter aligned with the draft `sentinel/eval/events.py` v1 (snapshot codes `E2a_lab`, `E5b`, `E6_snap`,
  `E3a_turb`; `Gaussian(mu, sigma, space)`; metrics as {point, ci_low, ci_high}; flags "feasible"/"underpowered").
  The event tests now run against the draft: 17 pass. Re-verify when the lead announces the v1 release.
- `finish.sh` (PID 1422173) runs the event dev step only if `x_track/dev/events_v1_release_ack.txt` exists. Write
  it on the lead's release notice; if it comes after `finish.sh`, run `bioassess_x_events.py --log`, then
  `bioassess_x_freeze_candidate.py`, then the dry run.

### 2026-11-09 13:37 — events.py v1.0 released
- Verified `sentinel/eval/events.py` sha256 `4279d242…54ae`, mode 0444 (= the draft the adapter was aligned to);
  the adapter now refuses any other version (`EVENTS_V1_SHA256`). Wrote `x_track/dev/events_v1_release_ack.txt`,
  so `finish.sh` (PID 1422173) runs the event dev step.
- Lead caveat: log10 back-transform in `to_event_units` is rounded to 6 decimals (boundary 230 mg/L / 0.1 mg/L
  now label 0, unit-tested). The derived route already passes `space="log10"` (threshold transformed).
- Rule check: all 31 M5-freeze-hashed files match `bioassess_freeze.json` (none edited by M5-X work). M5-X
  amendments go only in `bioassess_x_protocol.md`. Tests: 17/17 pass.

### 2026-11-09 ~14:00 — single-look hardening (M7 lessons; protocol v7a)
- `xtestlook.py`: `code_closure`/`scoring_code_files` (static import closure, 46 files incl. events.py → hazards /
  ew_scoring / early_warning, frozen M5 modules, R scripts); `verify_artifacts`; sealing registry row
  (`predictions_sealed`, every file's sha) + `check_sealed` against it (unlisted/missing/changed refused; refused once
  any scoring row exists); `require_open_claim` closed from sealing on (registry-checked, marker deletion can't
  reopen).
- Scorer: `write_truth` puts labels + scoring metadata into the sealed folder; `score_saved`/`score_events` read only
  the sealed folder; `--rescore-from-saved` loads no test rows; event errors → `event_track_status.json` + registry
  `event_track_not_run`, never fatal; artifacts verified before the claim (test + dry run).
- Candidate builder: `artifacts` (paths → sha) and `scoring_code_files` = the closure.
- Tests: `tests/test_bioassess_x_dev.py` 21 pass (closure subprocess test exercising the event path; hardening;
  artifacts). CPU smoke v3 of the hardened path passed (`x_track/dev/smoke_x_final_v3`, scratch registry,
  identical numbers to v2).

### 2026-11-09 ~18:00 — SOTA campaign (docs/plans/sota_campaign.md): M5-X expectation + SOTA plan
**Guard (campaign rule 7) completed:** claim tied to pid/host (`require_open_claim`; the single resume transfers
ownership), any BaseException after the claim → O_EXCL `ABORTED_*.json` + registry `look_aborted` (non-scoring, so
the pre-registered resume rule still applies), plus the v7a items. Tests 22 pass.

**Per-target expectation in the candidate** (`bioassess_x_select.py` `sota_expectation_cx` → candidate
`sota_expectation`): C-X vs the strongest PUBLISHED method (WA-PLS2, kNN-MAT, bio.infer, GJAM if v6 passes), the
strongest SIMPLE/practitioner baseline (region mean, RF[geography], nearest reference, UK-index regressions) and
the per-target comparator; paired dev Δ r2_anomaly, location-cluster CI, seed 0.
**Provisional dev reading (17:35 code check on partial outputs; deleted afterwards so `finish.sh` regenerates it;
X-U `bsx_flat_geo`/`ukpt`, some RF multi variants, X-U bio.infer ingest and GJAM still missing):** C-X = the
C-form ensemble in both blocks (mean dev r2_anomaly X-L 0.395, X-U 0.346). vs published: SOTA-expected on 12/12
(Δ +0.09 to +0.31). vs simple: SOTA-expected 10/12 (X-U chloride and ammonia only competitive vs UK indices +
geography). vs the per-target comparator (mostly baseline ensembles containing TabPFN): **SOTA-expected only X-U
conductivity (+0.031 [0.016, 0.044]), orthophosphate (+0.017 [0.009, 0.025]), nitrate (+0.032 [0.018, 0.046]);
competitive on the other 9** (Δ −0.02 to +0.03, CIs include 0). Final numbers come from `finish.sh`.

**SOTA plan (for later tracks on UNSEEN blocks; for lead review before any compute):**
- *Diagnosis.* BioSet alone (dev 0.26–0.31) is below TabPFN/LightGBM on the same community inputs; its gain is
  realised only in ensembles (variance reduction), and against baseline ensembles that already contain TabPFN the
  margin is ≈ 0.01–0.03 — below the X-L MDE (0.023–0.036). Rule 5: BioSet has no information the GBM lacks (same
  taxa; taxonomy hierarchy and cross-continent labels add little), so more capacity is unlikely to help.
- *Levers, ranked by expected gain per GPU-hour:*
  1. **Catchment/landscape context for every model (parity):** StreamCat/LakeCat (US) and CEH land cover /
     catchment descriptors (UK). CPU only. Information the community does not carry (between- and within-region
     chemistry drivers); requires the pending NLDI/COMID lookups (data request already filed).
  2. **More labelled data from new sources:** Scotland (SEPA), Ireland (EPA), Canada (CABIN) invertebrates with
     matched chemistry — also gives new-geography test blocks. Download/QC cost, no GPU.
  3. **Taxonomic-resolution harmonisation** (aggregate to the coarsest common level per family across ANLA/ANLE
     and across continents) for all models — targets the 2016–20 → 2021+ resolution shift (W-MX-8). CPU.
  4. **Large-corpus pretraining** (all UK + NRSA + NLA unlabelled samples, plus GBIF occurrence co-occurrence) for
     the encoder — only if the per-fold UK-pretraining config (`bsx_flat_geo_ukpt`, running) beats `bsx_flat_geo`
     on dev; ~1–2 GPU-h per fold set.
  5. **Within-region objective** (region-anomaly loss already tried as `bsx_flat_geo_rloss`: X-L 0.269 vs 0.262)
     — low expected gain.
- *Track definitions (each pre-registered before any result; unseen blocks only):* (a) **prospective UK EA**
  invertebrate samples dated after the data snapshot at `pool_fit` sites (≈ 1,500–2,500 per year; X-U-type
  leave-area-out; power ≈ X-U at 1 year); (b) **new geography** Scotland/Ireland (whole-country block); (c) NLA
  2027 / NRSA 2028-29 when released. M5/M5-X spent blocks are never reused.
- *Caps and ETA:* ≤ 8 configurations per new track, all counted; levers 1 + 3 ≈ 3–4 days CPU after the
  COMID/catchment data arrive; lever 2 depends on data access (1–2 weeks).
- **GJAM incident (17:40):** X-U fold-0 v6 fidelity crashed after its fit because `bioassess_x_gjam.R` saved the
  fit before creating the output directory (fixed: `dir.create` first). Fold 1 (same bug) was killed. R chain
  restarted as `r_chain4.sh` (PID 1584897; bio.infer complete for all 28 folds); `finish.sh` re-armed (PID 1585375).
  GJAM v6 fidelity on fold 0 of both blocks ≈ 21:00 tonight; the remaining GJAM folds run 1 worker until the CPU
  chain ends, then 6 → development complete ≈ 2026-11-10 afternoon (≈ +4 h vs the earlier ETA).

### 2026-11-10 06:30 — GJAM v6 fidelity (full size): FAILED in both blocks → GJAM "not compared, with reason"
- Criterion (v6, fixed before any GJAM dev score): our routine in gjam's in-sample scheme vs `xpredMu`, r ≥ 0.90 for
  every target. **X-L fold 0** (4,474 rows, fit 99 min): TP 0.731, TN 0.776, cond 0.884, Cl 0.840, turbidity 0.778,
  pH 0.816 → FAIL. **X-U fold 0** (5,000 rows, fit 114 min): pH 0.912, cond 0.886, Cl 0.827, orthoP 0.923,
  NH4 0.911, nitrate 0.851 → FAIL. (The 400-row smoke gave ≥ 0.99; at full size the posterior-mean coefficients /
  correlation used by our routine do not reproduce gjam's per-iteration sampling closely enough.) Out-of-sample-mode
  r 0.37–0.66 (informational). Files: `x_track/dev/r_out/gjam/{X-L,X-U}/fold00_info.txt`, logs `gjam_*_fold00.log`.
- Per the lead's instruction: reported as failed, **no further amendment**. `finish.sh` does not ingest GJAM (its
  gate requires `fidelity_pass TRUE` in both blocks); the candidate's v6 disclosure attaches both info files. The
  GJAM pool was stopped (X-U 19/19 and X-L 2/9 folds had been fitted; outputs kept, not used).
- Development otherwise complete: GPU chains (all 8 BioSet configs + TabPFN) finished 21:42; CPU chain 00:28;
  bio.infer 28/28. `finish.sh` (PID 1585375) is now running ingest → score → select → events → candidate → dry run.

### 2026-11-10 ~12:00 — GPU rule update (GPUs 0–9) applied; scorer GPU use restructured (protocol v7b)
- No M5-X file hard-codes `SENTINEL_GPUS` (checked); the cap file now reads 0–9 and all access goes through
  `sentinel.core.gpu` with `--min-free-mb`.
- The first dev dry run (started 07:03 under finish.sh, waited for a lock until ~10:40, then held GPU 0 while running
  3-seed CPU baselines) was stopped at ~70 GPU-min: it would have exceeded 2 h, and the real pass would need
  ≈ 14 GPU-h in one process. Fix (`scripts/eval/bioassess_x_final.py`): the pass process holds **no** GPU;
  `gpu_units` → `write_gpu_bundle` (label-free, sha-pinned test inputs written under the open claim) →
  `run_gpu_workers` launches `--gpu-worker` chunks of ≤ 100 min, one at a time, via the lock helper with
  `--min-free-mb 5000` and `SENTINEL_GPUS` narrowed by `unoccupied_gpus()` (no process of another user);
  workers refuse after sealing and fill the sha-verified unit cache; then CPU units, sealing, scoring. SIGTERM of
  the pass = recorded abort. Tests 26 pass; CPU smoke v4 passed.
- Candidate regenerated (code hashes changed); dev dry run relaunched 11:58 (`candidate_dryrun.sh`, PID 2134296;
  first worker on GPU 8, 12 GPU units). Log `x_track/dev/logs/dryrun_x_final.log`.

### 2026-11-10 14:49 EST — M5-X: scorer fits every listed comparator (protocol v9 + v10); candidate regenerated
- **Earlier dry run (scorer before this change): passed.** PID 2134875 exited 0 at 14:16 EST. Run folder
  `sentinel_runs/bioassess/x_final_dryrun_dev_20261110T115830` (candidate a4b077c8…0ff7, now replaced). Look
  markers sealed, `results/SHA256SUMS` verifies, 12 confirmatory rows (10 "Inconclusive (underpowered)", 1
  "Competitive", 1 "Not estimable"). GPU worker 61 min on GPU 8 via the lock helper. No test value read.
- **Lead ruling A, applied in the approved order** (change-log entry → scorer → tests → candidate → dry run):
  - `bioassess_x_protocol.md` **v9** (2026-11-10 14:33 EST, before any scorer edit): the scorer fits every listed
    §6 comparator at test, 14 for X-L and 17 for X-U (bio.infer, the UK index baselines and the 3 baseline
    ensembles included). Paired CI (B = 10,000) of C-X minus each. Row 1 needs every lower bound > 0. Row 2 names
    each comparator not beaten. A comparator with no finite estimate counts as not beaten.
  - **v10** (2026-11-10 14:37 EST, before the label function was edited and before the dry run; lead ruling on my
    row-3 question): "Below" also fires when any listed comparator is significantly above C-X (paired CI upper
    bound < 0). It can only make a label harsher. The trigger uses unadjusted 95 % intervals against up to 17
    comparators; v10 says so.
  - Protocol sha256 after v10: `02832564abb67863f483545024bff63af76ab6869b5157c56188c5ea55b09edc` (511 lines).
    Reported to the lead. **Do not edit the file again without a new dated entry and a new report.**
- **Not compared, with reason** (in the candidate under `not_compared`, and in `results/not_compared.json`):
  - GJAM: failed the pre-declared fidelity check (v4.8 item 8 as corrected in v6; `fidelity_pass` FALSE in both
    blocks). The 2-CPU-hour ground is **not** the ground used: the fits took 99.3 min (X-L fold 0) and 114.4 min
    (X-U fold 0). The record holds, per block and target, the in-sample-scheme r, the out-of-sample-mode r, n rows,
    and the path and sha256 of the info files and logs. No GJAM prediction is scored on test rows. No further
    GJAM tuning. Claim wording must not say "beats all joint-species models".
  - TRPI: licence (pre-declared).
- **Code** (new or changed; frozen M5 files untouched):
  - `scripts/eval/bioassess_x_final.py`: `LISTED`, `check_listed` (refuses a record whose comparator list differs,
    before setup and before any claim), `score_listed` (parallel paired bootstraps, 5 workers × 1 thread; equal to
    the serial result bit for bit), new `label`, outputs `results/listed_comparators.json` and
    `results/not_compared.json`, registry rows `delta_r2_anomaly_listed_comparator` and `comparator_not_compared`.
  - `scripts/eval/bioassess_x_freeze_candidate.py`: `comparator_set` per block (spec, class, dev scores),
    `not_compared`, GJAM fidelity files added to the hashed artifacts (the generator stops if the fidelity is not
    a recorded failure).
  - `tests/test_bioassess_x_dev.py`: 4 new tests (the lead's 3 label cases, list refusal, every listed comparator
    scored on synthetic rows, GJAM log parsing). 69 tests pass (M5-X + M5 model files).
- **Candidate** `docs/preregistration/bioassess_x_freeze_candidate.json`: sha256
  `20a4d2e3d433af778e39fbf87f32ba59864b8043213063766912b40212bf4680` (98,527 bytes, written 14:45 EST).
  Candidates and per-target comparators are unchanged (12 of 12 configurations). **Any edit to a hashed file
  (protocol, configs, `sentinel/models/bioassess/*.py`, scorer, generator, select script, the two test files,
  events files) means: regenerate the candidate, repeat the dry run, report the new sha256.**
- **New dry run** started 14:45:44 EST, PID 2276010, log `x_track/dev/logs/dryrun_x_final_v10.log`, run folder
  `sentinel_runs/bioassess/x_final_dryrun_dev_20261110T144544`. Outcome: see the next section.
- Still not done, by instruction: no FREEZE file, no test scoring, no test target read.

### 2026-11-10 18:40 EST — M5-X: dry run of the v9/v10 scorer finished (development rows only)

- Command: `bioassess_x_final.py --dry-run-dev --max-folds 1`. Started 14:45:44 EST, exit 0 at 18:36:13 EST.
  Run folder `sentinel_runs/bioassess/x_final_dryrun_dev_20261110T144544` (`results/` 8 JSON files + `SHA256SUMS`,
  all verified; `look/` holds `PREDICTIONS_SAVED` (129 files), `SCORED`, `TEST_LOOK` of the dry-run folder only).
  Log `sentinel_runs/bioassess/x_track/dev/logs/dryrun_x_final_v10.log`.
- **No test value was read. No FREEZE file exists. The test blocks are unscored.**
- Candidate `docs/preregistration/bioassess_x_freeze_candidate.json` sha256
  `20a4d2e3d433af778e39fbf87f32ba59864b8043213063766912b40212bf4680` (unchanged); protocol sha256
  `02832564abb67863f483545024bff63af76ab6869b5157c56188c5ea55b09edc` (unchanged); scoring-code hash
  `2ea4315ab148c6122b37a3bd7acdd922ae0e77d3246162a214f69b18fd324f16`. The 63 hashed files were re-checked after
  the run: 0 changed.
- Outcome (one development fold per track, so this checks the code path, not the result): 12 confirmatory rows;
  10 "Inconclusive (underpowered)", 1 "Competitive" (X-U NH4-N, delta 0.0638 [0.0085, 0.1531]), 1 "Not
  estimable" (X-U chloride: 1 cluster in the single fold; the full development design had 197). Every listed
  comparator is fitted and scored: 14 per target on X-L, 17 on X-U (186 rows in `listed_comparators.json`).
  Not compared: GJAM only (failed fidelity check).
- Agreement with the earlier dry run (`x_final_dryrun_dev_20261110T115830`): all 12 labels identical, same
  comparator per target, deltas equal to 4 decimals (largest change 0.0001 in one CI end; 4 X-U rows).
- Time: GPU units 14:46-15:46; predictions saved 18:28; scored 18:36. Per fold: X-L about 0.4 GPU-h + 1.1 h CPU
  (5 threads); X-U about 0.6 GPU-h + 1.9 h CPU. Projection for the real pass (9 + 19 folds): about 15 GPU-h and
  about 45 h of 5-thread CPU (about 2 days). **Open with the lead:** the candidate allows one resume only; a
  pass of this length needs a ruling on chunking before FREEZE.
- Risk: `sentinel/models/bioassess/__init__.py` and the M5-X modules are hashed. Any edit there means: regenerate
  the candidate, repeat the dry run, report the new sha256. M5-F work touches only unhashed files.
- Waiting on the lead: pre-test review and the FREEZE decision. The engineer does not write FREEZE and does not
  score test.

### 2026-11-10 18:54 EST — M5-F: stamped work started (French biology pull, comparator table, UK development design)

Stamp: registry row `f87159c7dc1d` (protocol sha256 `480c765f…099e`, rulings sha256 `8d96e3c0…b3bf`; both 0444,
never edited; later rule changes go to `docs/preregistration/bioassess_f_deviations.md`, stamped by the lead first).

**Lead rulings of 2026-11-10 (evening) logged here**
- M5-X: candidate `20a4d2e3…4680` and protocol `02832564…9edc` are under independent pre-test review. No edit
  of either file, of the 48 pinned files or of any file in `scoring_code_files` until the lead rules. No ruling
  yet on chunking of the real pass. The lead writes the freeze; the engineer writes no look, freeze or claim row.
- Comparator table: (a) TabPFN cap 300 + 150 with the 500-column rule, (b) B2 literal input list, (d) multi-output
  RF, (e) MEMLM deadline — accepted as written. (c) a second region-offset variant with α chosen by
  leave-one-pseudo-region-out CV over {0.1, 1, 10, 100, 1000} is added; for G7 the comparator is the variant with
  the higher mean development pooled R²; both reported.
- Compute: GPUs 3–9 when idle (GPU 1 when free, never GPU 2), one job per GPU through the lock helper, ≤ 2 h;
  CPU ≤ 6 threads per job. No setting changes for speed.
- Gate rows G2 (`bioassess_f/gate_g2`) and G4 (`bioassess_f/gate_g4`) are written only by the lead.

**Done**
- Code (all outside the M5-X pinned list): stamp guard, `locked_availability()` (counts only), fetcher with
  bounded retry and unsplit-block record in `ftrack.py` (sha256 `3468551a…5732`); pull driver
  `scripts/data/bioassess_f_pull.py` (sha256 `1de26486…0e2d`); `tests/test_bioassess_f.py` (64 pass).
- Comparator table `docs/preregistration/bioassess_f_comparators.md`: first written 2026-11-10T23:31Z
  (sha256 `c7917682…ba29`), before any French request and any fit. Two dated lines appended in §9 at
  2026-11-10T23:51Z, before any fit: the lead's region-offset variant; the UK development design details
  (folds, B2 member rows, index definitions, UK reference set). Current sha256
  `9622754b1c699de2936e2ac9a66bad77e123bb3b940979dc37330d9d06ae3048`, reported to the lead.
- French biology pull (step B): first request 2026-11-10T23:32Z. Probe on department 01 checked by counts only:
  indices, 40 blocks, 14,914 records = stored lines minus headers (support 13: 11,938; support 10: 2,976); no
  unsplit block. Full pull runs in chunks of ≤ 100 min; chunk 1 started 18:41 EST. Registry row `09ffb9f65ff0`.
  `docs/licenses.md` row updated. No record is parsed; no chemistry request is sent before the lead logs G2.
- UK development design (§9.4): `sentinel/models/bioassess/fdev.py`, runner
  `scripts/eval/bioassess_f_ukdev.py` (stages `design`, `refs`, `ma`, `table`), tests
  `tests/test_bioassess_f_ukdev.py` (4 pass: fold invariants on the real design, guard refusals, featurizer
  rules). Design: 212,503 allowed UK rows (198,484 invertebrate, 14,019 diatom-only), 19 `agency_area` folds,
  4,788 pseudo-test rows (33–687 per fold), training 180,860–195,410 rows per fold. Files in
  `sentinel_runs/bioassess/f_track/ukdev/` (`design.pkl`, `folds.json`, `pred/<model>/seed0/<area>.parquet`).
- First UK development fits started 18:52–18:53 EST (lead notified): cheap references on CPU (5 threads);
  TabPFN [community] / coords / nocoords on GPUs 5 / 6 / 7; #7 `ma_zs` timing run (3 epochs, no validation
  number) on GPU 8.

**Rules kept in the code**
- Every fit passes `UKDev.assert_fit_ids` (→ `ftrack.assert_fit_ids`, then the fold pool). Checkpoints and resume
  states carry track `F-Z`, fold `ukdev:<area>`, design `dev`.
- Gate G5 / §9.5: #7 and #8 predictions are stored unscored. The `table` stage scores them only with
  `--candidates` and only when every model of `fdev.UK_REFERENCES` is stored; this is not run before the lead's
  G4 stamp.

**Open**
- The Hub'eau count requests take 7–9 s each. The biology pull has about 15,500 count requests plus pages:
  about 40 h in total at one request at a time (about 25 chunks).
- CPU cost: RF `nocoords` (reference) and the RF member of #8 are each 114 fits on ≤ 60,000 rows; one heavy CPU
  job at a time. Timings are added below when measured.
- Owed: MEMLM line (before the first French development fit); G2 files; G3 counts; G4 package; `ftestlook.py`;
  French design builder.

## 2026-11-11 13:08 EST — M5-X fix package for the pre-test review (engineer); status: REHEARSAL RUNNING, nothing frozen, no test row or label touched

**Lead rulings X-B1..X-B4 (2026-11-10 ~19:15 EST, verdict `freeze_after_fixes`)** are implemented as protocol
amendment **v11** (`docs/preregistration/bioassess_x_protocol.md`, sha256
`d5ba296dd1cc6c088212327ee789592b162d0edd8bf6dbab2ece564f33e7d4d1`, entry dated 2026-11-10 19:24 EST, written
before any code it names was changed or run and before any test value was read).

- **X-B3 (rule comparators, no new fit).** `scripts/eval/bioassess_x_select.py` now derives, per block, from the
  selected C-X: `cxr_nobioset` (equal-weight mean of C-X's baseline members) and `cxr_swap` (those plus the
  listed single baseline with the highest mean development `r2_anomaly`, bio.infer/GJAM and C-X members
  excluded). Both are means of predictions that already existed; no model was fitted. X-L: members
  RF[community+geography] + TabPFN[community], swap `gbm_community` (0.3102). X-U: members
  RF[community+context] + RF[index+geography], swap `rf_community_geography` (0.2934). Listed sets 14 → 16
  (X-L) and 17 → 19 (X-U). Both compete for the per-target comparator by the unchanged §6 rule; after the
  re-run **no per-target comparator changed** (the X-L rule comparators have the same members as
  `bens_rfg_tabpfn` / `bens_rfg_gbm_tabpfn`, so their frames are reused bit for bit and the tie stays with the
  ensemble's own listed name; recorded in the candidate).
- **X-B1/X-B2 (scorer restructure).** `scripts/eval/bioassess_x_final.py` runs the look as four commands —
  claim, chunk (any number, any order, in parallel, resumable), seal, score — with a label-free prediction
  phase, write-once units per (block, leaf model, fold, seed) and per event fit, a blind between chunks, one
  seal and single-shot scoring with one locked registry append; the look runs from a read-only code snapshot
  (`sentinel_runs/bioassess_x/code_<tree16>/`, files 0444, dirs 0555), and every command verifies the snapshot,
  not the live tree. `tests/test_bioassess_x_dev.py` is 41 tests (all pass, 2026-11-11 13:00 EST), one per
  five-point item plus parallel chunks, the live-tree-edit / snapshot-edit subprocess test and the
  cap-refused-in-real-mode tests.
- **Record errors fixed at regeneration** (`scripts/eval/bioassess_x_freeze_candidate.py`): `one_time_guards`
  now names the markers and metrics the code uses; `resume_rule` is the v11 item 2 text; the 395 MB training
  design cache and the frozen-key cache are pinned in `artifacts`; every file of the scoring import closure is
  listed singly (64 files); `reproducibility_tolerance` and the two v11 disclosures (optimistic power table;
  not bit-reproducible) are in the record; the X-L test-row note says 459 (439 Fuse tier-C + 20 D2n-only of
  974), not 549.
- **Candidate (final, 2026-11-11 16:31 EST):** `docs/preregistration/bioassess_x_freeze_candidate.json`
  sha256 `a4b506b6711527c1db976e709e65b7609c0193957a5fc534ce7379251024bebb`; code snapshot
  `sentinel_runs/bioassess_x/code_179746238432ac31` (57 files, 0444/0555); it pins the rehearsal record
  `sentinel_runs/bioassess_x/rehearsal_v11.json` (sha256
  `300a9757c2097b94408024a566508e8bc9e29579372a06ddfa6a643f7b3a1256`). No FREEZE file exists; 0
  `bioassess_x_test` registry rows; no test row and no test label was read.
- **X-B4 rehearsal** (development only, cap 300 training rows per source and fold, reduced ensembles and epochs,
  R methods at 3x the cap; "numbers not usable"): run dir
  `sentinel_runs/bioassess/x_final_dryrun_dev_20261111T141649`, 1,046 units over all 9 X-L and 19 X-U folds.
  Chunks: a GPU chunk queued on the shared GPU lock (GPU 1), a CPU chunk killed by PID and started again (no unit
  refitted), and a `--cpu-smoke` chunk fitting GPU-kind units on the CPU beside it. Prediction phase 28 min; seal
  141 files; `--score` killed 12 s in (abort recorded); `--rescore-from-saved` recomputed everything in 1 h 40 min
  and wrote 12 confirmatory rows and 569 registry rows in one locked append. An earlier rehearsal refused to seal
  while 190 units were missing (v11 item 2.4 holds). `--verify-scored` was started and stopped by hand to free the
  CPU; it is not part of X-B4 and its logic is unit-tested.
- **Two defects the rehearsal found, both fixed and unit-tested** (42 tests pass): (1) a unit whose fit raised
  killed the chunk and every later chunk failed on the same unit, so the look could never seal - the chunk now
  records the failing unit, writes nothing for it, goes on and ends with exit 6 naming it; (2) the rehearsal cap
  starved bio.infer's R code (`mlsolve` could not build its coefficient matrix) - the R methods now get
  `R_CAP_FACTOR = 3` times the cap in the dry run, and the cap is still refused in real mode.
- **For scheduling the real pass:** the fits are not the problem, the scoring is. GPU units 2-9 s each at the cap
  (the one TabPFN unit fitted on CPU took 1,004 s); bio.infer is the slowest CPU unit (64 s per X-L fold at 900
  rows per source, 150-330 s at full size); the scoring step alone took 1 h 40 min on 5 CPU workers
  (B = 10,000 against 16 + 19 listed comparators x 12 targets). Per-unit times: `results/pass_info.json`.

## 2026-11-11 16:35 EST — M5-F state (engineer)

- **UK references complete.** All 15 models of `fdev.UK_REFERENCES` are stored for all 19 `agency_area` folds
  (`sentinel_runs/bioassess/f_track/ukdev/pred/<model>/seed0/`): pool mean, month climatology, RF[geography],
  nearest station, WA-PLS2, kNN-MAT, the four index regressions, TabPFN [community] / coords / nocoords,
  RF[nocoords], LightGBM[nocoords]. Gate G5 condition "every UK reference stored" is met; nothing is scored —
  `table --candidates` waits for the lead's G4 stamp.
- **#7 / #8 complete (2026-11-11 19:31 EST).** Every model of `fdev.UK_REFERENCES`, both `MEMBERS_B2`
  (`b2_rf_zs`, `b2_tabpfn_zs`) and `ma_zs` are stored for all 19 `agency_area` folds, seed 0;
  `ma_zs_noregion` stored as well. Nothing is scored: the `table --candidates` stage waits for the lead's G4
  stamp (gate G5 / §9.5). Earlier state, for the record: `b2_tabpfn_zs` had two TabPFN
  segfaults inside `tabpfn/architectures/base/transformer.py`, each resumable per area (~480 s per area on
  one GPU); `b2_rf_zs` ran on CPU at ~300 s per area. The GPU jobs now queue on the shared sentinel GPU lock
  (`run_gpu_wait.sh`) instead of giving up when every permitted GPU is busy.
- **French biology pull:** step B `stations_hydrobio` complete (96 departments, 21,567 records, no unsplit
  block); `indices` and `taxons` by department x support x year are running in 100-minute chunks (chunks 1-4
  done: 1,160 + 1,194 + 1,087 + 1,173 requests; chunk 5 running). One request at a time, counts only, nothing
  parsed. M5-F data on disk 0.002 GB of the 5 GB cap; disk 166 GB free.
- **MEMLM (owed §9 line).** The paper was fetched (Sun, Holden & Birks 2024, Clim. Past 20:2373-2398,
  doi:10.5194/cp-20-2373-2024). Its MEMLM is a two-layer "Multi Ensemble Machine Learning Model": layer 1 =
  random forest + extremely randomised trees + LightGBM, 1,000 trees each, on taxon abundances; layer 2 =
  multiple linear regression that integrates the three reconstructions, its weights fitted on the three models'
  internal 5-fold cross-validation predictions (not an average). The embedding variants MEMLMe / MEMLMc (GloVe,
  256 dimensions, 1,000 epochs) are separate models and are not proposed. The line is NOT yet written into
  `bioassess_f_comparators.md`: the lead asked to be told before that file is edited again, and the cost
  question is with the lead.
