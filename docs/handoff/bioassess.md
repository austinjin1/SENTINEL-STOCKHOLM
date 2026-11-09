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
- Write-up done: RESULTS.md (2026-11-09 entry), model card (v0.2.0), protocol §11 (appended; changes the
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

### Interim development results (09:40; unlogged progress scoring, seed 0, r2_anomaly with 95 % location-cluster CI, B = 2,000; `x_track/dev/dev_scores.csv`)
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
