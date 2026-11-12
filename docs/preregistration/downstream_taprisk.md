# D4 TapRisk — pre-registration protocol (drinking-water nitrate / DBP violation risk of community water systems)

**Status:** v1.0, written 2026-11-08 **before the split was frozen, before any label count was inspected and
before any model was fitted** (validation only; RESEARCH_PLAN §8 D4, §14, §15; lead direction 2026-11-08).
**Code:** `sentinel/models/downstream/taprisk.py`, `scripts/train/downstream_taprisk.py`,
`scripts/data/build_taprisk_links.py`. **Data:** `labels_sdwis` (EPA SDWIS/ECHO bulk 2026-10-14, public
domain; datasheet `docs/datasheets/labels_sdwis.md`), `labels_cws_boundaries` (EPA SAB v3, public domain),
`taprisk_catchment` (new: NLDI COMID snap of service-area points + curated StreamCat; datasheet
`docs/datasheets/taprisk_catchment.md`), `wqp_discrete` (ambient nitrate, observed), `wbd_huc12`.
**No upstream-model output is used** (observed covariates only), so the §8 cross-fitting rule is not triggered;
`crossfit.phase1_no_upstream` is applied.

## 1. Question, incumbent and claim target
Can the risk that a community water system (CWS) incurs a **health-based nitrate MCL violation** in the next
three years be predicted for systems in **unseen states** and future periods better than the incumbent
**Pennino et al. 2020** (*Sci. Total Environ.* 722:137661; PMC8204728)?
- Incumbent (published): R `randomForest` classification of "catchment had ≥ 1 nitrate MCL violation
  2013–2017", NHDPlusV2 catchments located via SDWIS facility locations, ≈ 210 StreamCat metrics (+ N surplus,
  census), majority class under-sampled to balance (×10), importance-based variable selection, at risk if
  p > 0.5; GW and SW modelled separately; 10-fold CV on pre-balanced data: GW sensitivity 75.8 %, specificity
  81.5 %; SW PCC 90.3 % (sens 91.9 %, spec 88.6 %); 57,096 GW catchments (748 violating), 4,692 SW (62).
- **Re-implemented incumbent on our units (lead guidance (1)):** the same model form (balanced
  under-sampled RF ×10 on StreamCat catchment + watershed metrics with importance-based variable selection),
  tuned with parity. For H-G4 the Pennino-form re-implementation **is the D4 incumbent**; the published
  numbers are context and are used for the reproduction check (§5).
- Primary task **D4a: GW nitrate** (Pennino's powered task). Secondary track **D4b: surface-water (SW + GU)
  DBP MCL (TTHM/HAA5)** with the same design (a Pennino-form RF is its incumbent; no published DBP model with
  comparable evaluation was found — dated novelty check `docs/novelty_checks/downstream_taprisk.md`).
  SW nitrate (Pennino: 62 violating catchments) is reported descriptively only.

## 2. Units and labels
- Unit: CWS with an own source (`primary_source_code` GW for D4a; SW or GU for D4b), active at some time
  2010–2024, with an SAB v3 service-area polygon; purchased-water systems excluded (their source is the
  seller's). Location = polygon representative point; NHDPlusV2 COMID by NLDI position snap (proxy for
  Pennino's facility catchment — SDWIS facility coordinates are not public; disclosed).
- Origins T (decision date 1 January T+1): label window = calendar years T+1 … T+3.
  **Label = 1** if a health-based violation of the task label (`labels_sdwis` `label` = `nitrate_mcl` /
  `dbp_mcl`) has a compliance period overlapping the window. A system is a candidate at T if active on
  1 January T+1 (not deactivated before it, first reported before it).
- Origins: train **T = 2010, 2013**; validation **T = 2016** (window 2017–2019); test **T = 2019** (window
  2020–2022); T = 2022 (2023–2025) is a reserve test origin (its window ends inside the reporting lag of the
  2026-10 snapshot).

## 3. Decision time and features (observed only)
Everything a feature uses must be known on 1 January T+1:
- violation history uses violations with `viol_first_reported_date` ≤ 31 December T (not just begin date):
  counts of health-based violations of the task label in the 3 and 10 years before, years since the last,
  any other health-based violation, monitoring/reporting violations of the rule group (3 years);
- system attributes from SDWIS (log population, service connections, owner type, wholesaler flag, primary
  source, DBP schedule category for D4b) — values of the 2026 snapshot, which may postdate T for
  slowly-changing attributes (disclosed);
- StreamCat curated catchment + watershed metrics of the snapped COMID (static; land cover 2019 etc. —
  postdate early origins, as in Pennino; disclosed);
- ambient nitrate (D4a) / DOC (D4b): HUC8 median of WQP surface-water results in [T − 9, T];
- location (lat/lon of the point) — learners only.

## 4. Split (frozen before any label count; `--freeze-split`)
States (SDWIS `state_code`; tribal/territory primacies grouped into their own state code) are dealt to roles
from **system counts only**: largest first to the role furthest below its target (test 0.20, val 0.15);
ties by sha1(seed 20261108, state). Rows: `train` = train states × T ∈ {2010, 2013}; `val_known` = train
states × T = 2016; `val_new` = val states × T = 2016; `test_known` = train + val states × T = 2019;
`test_new` = test states × T = 2019. Val-state rows at train origins and test-state rows at non-test origins
are dropped (unseen states start cold). Bootstrap unit: HUC8 of the point (nearest WBD HUC12 centroid).

## 5. Incumbent reproduction (before any challenger fit)
Pennino design on our units: GW, unit = snapped catchment (COMID) with ≥ 1 GW CWS, label = any nitrate MCL
violation compliance period overlapping 2013–2017 (all states, no split — their random design), RF ×10 on
balanced under-samples, 10-fold CV on the pre-balanced data, sensitivity and specificity at p > 0.5.
**Tolerance: within ±0.05 of 75.8 % / 81.5 %**; outside it, diagnose before any claim (differences expected
from the proxy location and the StreamCat subset are disclosed).

## 6. Models and comparators (fit on train rows; same 4-configuration budget per learned model)
- Simple baselines: prevalence; state rate (shrunk; location-only); persistence (any task-label violation
  first reported ≤ T in the 3 years before, tie-break by count); ambient-nitrate rule (HUC8 median); system
  size/source/state logistic (research/07 baseline).
- **Incumbent `pennino_rf`:** balanced RF ×10 (500 trees each; sklearn RandomForestClassifier), StreamCat
  features only; configurations = number of top-importance features kept ∈ {all, 50, 25, 10} (importance from
  a balanced RF on the inner-fit rows).
- **Incumbent+ `pennino_rf_hist`:** the same plus the violation-history and system features (so the
  challenger's information advantage is not only history; reported, and part of the claim family).
- Challengers: L2 logistic (C ∈ {0.01, 0.1, 1, 10}); LightGBM (num_leaves {7, 15} × min_child {20, 100},
  early stopping, 3 seeds) on all features.
- Selection: inner split of training rows by origin (fit T = 2010, score T = 2013), criterion AUROC.

## 7. Endpoints and statistics
- Primary: **AUROC** on `val_known` and `val_new` (claim at test on both). Secondary: AUPRC, Brier, Pennino's
  metrics (sensitivity/specificity/PCC at p > 0.5 on the natural and on a balanced subsample), D4b, SW
  nitrate (descriptive).
- Paired HUC8-cluster bootstrap (B = 2,000). **Claim rule (§15.1, single test look):** the frozen challenger
  beats `pennino_rf`, `pennino_rf_hist` and the strongest simple baseline (chosen on validation) on both test
  sets, paired CI excluding 0, Holm over all these comparisons, and the per-seed paired point estimate > 0 in
  each of 3 seeds. Otherwise "competitive"/"negative".

## 8. Iteration, freezing, test
Validation only; one change at a time; every configuration logged (`configs_tried.jsonl`). Single test look
after a FREEZE record (test path with `testlook` guard, data pins, saved predictions, as for the other apps).
No git (user instruction): ordering evidence = registry timestamps + `prereg_snapshot`.

## 9. Frozen split, counts, power (filled from `--freeze-split` / `--build-only`, before any fit)
- **Split frozen 2026-11-08:** `downstream_taprisk_split.json`, **sha256 `a4a1fc7bb5f997d0a656c527c884ed46…5b3298`**
  (full: a4a1fc7bb5f997d0a656c527c884ed4691c59da341eff81d0d424bbd3e5b3298); states: 24 train / 13 val /
  19 test; 35,803 systems with points (GW 31,546, SW 3,782, GU 475).
- **Counts and power (`--build-only`, labels only, no StreamCat, no fit):**
  | task | train (n / pos) | val_known (n / pos) | val_new (n / pos) | paired MDE known / new |
  |---|---|---|---|---|
  | D4a GW nitrate | 39,672 / 312 | 20,055 / 177 | 4,520 / 41 | 0.053 / 0.111 |
  | D4b SW+GU DBP | 5,816 / 866 | 2,915 / 434 | 779 / 108 | 0.035 / 0.071 |
  | SW nitrate (descriptive) | 5,816 / 10 | 2,915 / 5 | 779 / 1 | — |
  D4a `val_new` is **underpowered** (41 violators; MDE ≈ 0.11 AUROC, not inflated for clustering); the test
  origin (test states, ~20 % of systems) will be of similar size — a D4a claim on new states is unlikely
  unless the gain is large; D4b is adequately powered. Stated before any fit.
- **Deviation (2026-11-08, before the reproduction ran):** the §5 reproduction uses **train + val states
  only** (test states' labels are never read outside the single test look), unlike Pennino's all-CONUS
  design; variable selection is not applied in the reproduction (all curated StreamCat columns; the top-k
  selection is part of the phase-1 incumbent tuning). Simple-baseline list fixed as: location kNN (25 nearest
  training systems), state rate (shrunk), persistence, ambient rule, size/source logistic.

## 10. Incumbent reproduction and iteration log (validation only)
- **Reproduction (run `taprisk/reproduction`, 2026-11-09):** Pennino design on our units (GW catchments with ≥ 1 GW
  CWS in train + val states, any nitrate MCL violation overlapping 2013–2017; 20,041 catchments, 253 violating;
  balanced RF ×10, 10-fold CV on pre-balanced data): **sensitivity 0.813, specificity 0.815**, PCC 0.814, AUROC
  0.883 vs published 0.758 / 0.815. Specificity reproduces exactly; sensitivity is **+0.055, outside the ±0.05
  tolerance by 0.005, in the incumbent's favour**. Diagnosis: (i) sampling — 253 positives give a per-rep
  sensitivity SE ≈ 0.025, so +0.055 ≈ 2 SE; (ii) design — fewer catchments (train + val states only, 20k vs 57k;
  253 vs 748 violators) and the service-area-point proxy for facility locations; (iii) no variable selection in
  the reproduction. The re-implementation is therefore not weaker than the published model on its own design;
  phase 1 proceeds with this disclosure (no change to the incumbent).
- **Phase 1 (runs `taprisk/phase1_<task>`, 2026-11-09; 4 configurations per learned model, 16 per task).** AUROC
  (HUC8 paired CIs):
  | task / set (n / pos) | pennino_rf | pennino_rf_hist | logistic | GBM | strongest simple | GBM − rf | GBM − rf_hist |
  |---|---|---|---|---|---|---|---|
  | D4a known (20,055 / 177) | 0.939 | 0.952 | 0.948 | **0.961** | kNN 0.865 | +0.022 [0.014, 0.033] | +0.009 [0.003, 0.016] |
  | D4a new (4,520 / 41) | 0.803 | **0.891** | 0.874 | 0.865 | persistence 0.762 | +0.062 [0.014, 0.114] | −0.026 [−0.058, 0.006] |
  | D4b known (2,915 / 434) | 0.831 | **0.843** | 0.822 | 0.838 | persistence 0.791 | +0.007 [−0.005, 0.020] | −0.005 [−0.014, 0.004] |
  | D4b new (779 / 108) | 0.628 | 0.773 | **0.785** | 0.783 | persistence 0.746 | +0.155 [0.086, 0.219] | +0.010 [−0.016, 0.043] |
  Pennino-form RF selected top-50 features (D4a) / all (D4b). The plain Pennino-form RF is beaten clearly in new
  states (both tasks) and in known states (D4a); the history-augmented RF is not beaten on all sets → §7 rule
  not met on validation (competitive).
- **it1_ens (pre-registered 2026-11-09, before computing):** one change — challenger = equal-weight **rank**
  average of the phase-1 GBM and `pennino_rf_hist` validation scores (balanced-RF probabilities are on a different
  scale), from the saved validation predictions (no refit, no weight tuning). Same comparators and claim rule; the
  ensemble must beat `pennino_rf`, `pennino_rf_hist` (its own component) and the strongest simple baseline on both
  sets. Keep if it does; otherwise D4 stays competitive and the GBM is the freeze candidate for a "competitive" look.
- **it1_ens result (run `taprisk/it1_ens`, 2026-11-09): rejected.** Ensemble − pennino_rf_hist: D4a known +0.006
  [0.004, 0.010], D4a new −0.010 [−0.026, 0.006]; D4b known +0.001 [−0.003, 0.006], D4b new +0.014 [0.000, 0.031]
  (p 0.044). It beats the plain Pennino-form RF on all four sets (+0.014 … +0.159) but not its own history-RF
  component on all sets. **Status after 2 iterations (34 configurations): competitive** — the pre-registered
  family includes `pennino_rf_hist`, so beating only the published-form RF does not meet §7 (as for D5, the
  stricter pre-registered rule governs). Freeze candidate if a look is wanted: phase-1 GBM (D4a), expected
  label "competitive" unless the test differs.
- **Write-up separation (lead rule, 2026-11-09):** once a D4 FREEZE record exists, this protocol is never edited;
  test-look results and any later write-up go to `docs/preregistration/downstream_taprisk_test_results.md`.
- **it2 — bounded final iteration (pre-registered 2026-11-09, before any it2 code ran; lead cap: 1.5 days, ≤ 8
  configurations; this spec uses 6 → running total 40).** One coordinated change (lead decision), all features
  known on 1 January T+1:
  - *Richer violation history* (violations first reported ≤ 31 Dec T): task-label health-based counts in 1/3/5/10
    years; years since last; **lagged severity** = max over the prior 5 years of measured value ÷ MCL (nitrate
    10 mg/L; TTHM 0.080, HAA5 0.060 mg/L) for task-label violations; max public-notification tier (5 y); formal
    enforcement actions (5 y); monitoring/reporting counts of the rule group in 1/3/5 years; any other
    health-based violation (3/10 y).
  - *Treatment / source proxies* (SDWIS source facilities active by T; 2026-snapshot attributes, disclosed): number
    of sources by type (well, intake, reservoir, spring, consecutive connection), share of sources treated
    (`is_source_treated_ind`), any unfiltered surface source (`filtration_status_code`), share purchased
    (consecutive connections) and any treated purchase (`seller_treatment_code`), availability mix; plus log
    population. **No SDWIS treatment-process table is in our extract** (disclosed; source-level treatment flags
    are the available proxy).
  - Configurations per task (fixed hyperparameters, no search): (1) **`gbm_rich`** — phase-1 selected GBM
    hyperparameters on phase-1 + rich features (3 seeds); (2) **`pennino_rf_hist_rich`** — the history-augmented
    balanced RF ×10 on the same rich features (parity comparator: the incumbent+ gets the same information; it
    joins the claim family); (3) **`stack_rf_gbm`** — level-0 = `gbm_rich` and `pennino_rf_hist_rich` fitted on
    inner folds (training rows, 5 state-grouped folds) to give out-of-fold scores; level-1 = L2 logistic (C = 1)
    on the logits of the two out-of-fold scores; at prediction time level-0 models refit on all training rows.
    3 configurations × 2 tasks = 6.
  - **Selection rule (validation only):** candidates {phase-1 GBM, `gbm_rich`, `stack_rf_gbm`}; a candidate
    qualifies if, on both D4a validation sets, it beats `pennino_rf`, `pennino_rf_hist`, `pennino_rf_hist_rich` and
    the strongest simple baseline (paired HUC8 CI > 0, Holm over all, per-seed Δ > 0). Among qualifiers the one
    with the largest minimum margin is frozen; if none qualifies, the candidate with the highest mean D4a validation
    AUROC over the two sets is frozen and its test label is expected to be "competitive". The same model family /
    configuration is used for D4b (secondary). The test look scores D4a (primary claim) and D4b (secondary) once.
- **Test-look specification (written 2026-11-09, before the D4 candidate record):** frozen challenger = the it2
  rule's choice (D4a: `stack_rf_gbm`, no qualifier → expected "competitive"; same configuration for D4b).
  Test fit = **train + val rows** (origins 2010, 2013, 2016; train + val states) for the challenger and every
  comparator, frozen hyperparameters (Pennino-form RFs: top-k as selected in phase 1 recomputed by importance on the
  fit rows; stack meta-learner on 5 state-grouped inner folds of the fit rows). Test sets: `test_known` (train + val
  states × T = 2019) and `test_new` (test states × T = 2019), window 2020–2022. Primary (D4a): stack vs
  `pennino_rf`, `pennino_rf_hist`, `pennino_rf_hist_rich` and the strongest simple baseline on validation (known:
  location kNN; new: persistence), paired HUC8 AUROC, Holm over the 8 comparisons, per-seed Δ > 0. D4b is scored in
  the same look as a secondary result (its strongest simple baseline: persistence). Pennino's own metrics
  (sensitivity/specificity at p > 0.5) reported. Data pinned in `downstream_taprisk_data_pins.json`; the FREEZE
  record also carries `taprisk_registry.scoring_code_sha256()` (D4 files outside `common.scoring_files`), verified
  by the test script before the claim. Results go only to `downstream_taprisk_test_results.md`.
- **it2 results (runs `taprisk/it2_<task>`, 2026-11-09; 6 configurations → total 40).** D4a AUROC known / new:
  stack 0.960 / 0.883, gbm_rich 0.962 / 0.871, pennino_rf_hist_rich 0.953 / 0.883, pennino_rf_hist 0.952 / 0.891,
  pennino_rf 0.939 / 0.803. Stack − pennino_rf_hist: +0.008 [0.004, 0.012] known, −0.008 [−0.032, 0.015] new;
  stack − pennino_rf_hist_rich +0.008 [0.004, 0.012] / +0.000 [−0.010, 0.009]; stack − pennino_rf +0.022 / +0.080
  [0.040, 0.129]. D4b: stack 0.845 / 0.787 vs pennino_rf_hist 0.843 / 0.773 (Δ +0.002 [−0.003, 0.007] / +0.014
  [−0.004, 0.037]); vs pennino_rf +0.014 / +0.159. **No candidate qualifies → by the pre-registered rule the frozen
  challenger is `stack_rf_gbm` (highest mean D4a validation AUROC); expected test label: competitive.**

## 11. Pre-test review corrections (pre-registered 2026-11-10, before any code change; review "D4 TapRisk — pre-test review (2026-11-09)", FAIL)
- **C5 (enforcement look-ahead), correction to it2 (no new configuration):** `rh_task_enf_formal_5y` is rebuilt from the
  violation × enforcement scan (`interim/sdwis/violations_enforcement_scan.parquet`): number of enforcement actions with
  `ENF_ACTION_CATEGORY` = Formal, `ENFORCEMENT_DATE` ≤ 31 Dec T **and** `ENF_FIRST_REPORTED_DATE` ≤ 31 Dec T (when
  present), on task-label health-based violations first reported ≤ 31 Dec T with begin year ≥ T − 4.
  `public_notification_tier` and `viol_measure_mgl` are per-violation static fields (reviewer: look static) and are kept.
- **C6 (fit-row label look-ahead):** a violation counts toward a **fit-row** label only if `viol_first_reported_date` ≤
  31 Dec of the **scored origin** S (S = 2016 for validation, 2019 for the test look); evaluation-row labels stay
  uncensored (they are outcomes). Applies to every model and baseline (all comparators are refit under the censored
  labels with their frozen hyperparameters; logged as a correction, not new configurations).
- **it2 re-run (correction):** both tasks re-run on validation with C5 + C6 and the parity change below; the written
  selection rule is re-applied unchanged; the D4 configuration count stays 40 (+ the numeric track's own budget below).
- **W3 (parity):** `pennino_rf_hist_rich` (and therefore the stack's RF level-0) also receives lat/lon, as the stack's GBM does.
- **W4 (seed rule reading):** only the GBM seed varies across the 3 seeds; the RF level-0 and the meta-learner use seed 0
  in every seed. The per-seed rule is read on these 3 stack variants.
- **W1 (cluster unit):** in addition to HUC8, the D4a `test_new` primary comparisons are reported with a **state-cluster**
  bootstrap; the claim additionally requires Holm rejection under the state-cluster bootstrap for the `test_new`
  comparisons (stricter; pre-declared here).
- **W2 (disclosure):** units are defined by the 2026 SDWIS `primary_source_code` and SAB v3 presence (survivorship: systems
  that switched to purchased water or were consolidated after violations drop out of every origin); source proxies and
  system attributes are 2026-snapshot values.
- **W6:** pinned WQP partitions must contain only the pinned files (any extra file → refusal before the claim).
- **Hardened single look (C1–C4, C7):** D4-only module `sentinel/models/downstream/taprisk_look.py` wrapping the shared
  primitives (shared `testlook.py` / `common.py` unchanged): candidate-copy refused (status FROZEN + authorised_by/at +
  candidate sha + text ≠ candidate); claim registry row (pid/host) right after the claim; `BaseException` before sealing →
  `ABORTED.json` + registry row; sealed predictions (0444, `SHA256SUMS`, registry seal row with sha) before any scoring
  row; test access only inside the claiming process and closed at sealing; rescore only from the registry-sealed file and
  only while no scoring row exists; unlisted files in the look folder refused; `SENTINEL_RUNS`/`SENTINEL_DATA` resolved and
  pinned; full import-closure hash (incl. `sentinel/core/{paths,registry}.py`, package `__init__`s, `toxinwatch.py`);
  GBM `n_estimators` hard-coded (D4a 88, D4b 75) instead of reading `phase1_results.json` after the claim.
- **Numeric track (user requirement 2026-11-10; secondary, descriptive, no claim):** target = **log10 system-mean HAA5
  (µg/L)** from UCMR4 distribution-system samples 2018–2020 (`labels_ucmr`), plus raw-water TOC and bromide where sampled;
  unit = UCMR4 system in the D4 universe; decision date 1 Jan 2018 (features as of T = 2017, D4b feature set + rich
  features); states as frozen (train → fit, val → validation, test → scored once in the D4 look). Models: quantile
  LightGBM (q 0.05 / 0.5 / 0.95; median = point prediction), RF regression on the Pennino-form features, state-median and
  history (prior DBP-MCL violation count) baselines; budget 4 configurations. Metrics: RMSE and R² on log10, 90 % PI
  coverage and mean width (HUC8 bootstrap). **No measured nitrate concentrations exist in our SDWIS extract** (violation
  values only) → no nitrate regression track (disclosed).
- **Numeric-track correction (2026-11-10, after the first validation run, before any test look):** the first
  implementation averaged log10 values (geometric mean) and floored zeros at 10⁻³ µg/L — 3.3 % of HAA5 results are
  reported as 0 with no MRL, producing log10 = −3 outliers (validation R² of the train median −0.28). Corrected to the
  pre-registered definition: **log10 of the system mean of raw values** (censored → ½ MRL, reported zeros kept), floored
  at ~½ MRL (HAA5 1 µg/L, TOC 500 µg/L, bromide 10 µg/L). Validation re-run; the first run is kept in the registry
  (`numeric/*/val_states`) and superseded by `numeric_v2`. No other change.
- **it2c results (corrected re-run, runs `taprisk/it2c_<task>`, 2026-11-10; correction, configuration count unchanged
  at 40 + 2 numeric = 42).** Fit labels censored at 2016 (D4a training positives 312 → 265; D4b 866 → 730). AUROC
  known / new (D4a): stack 0.959 / 0.891, gbm_rich 0.961 / 0.886, gbm_phase1 0.961 / 0.875, pennino_rf 0.935 / 0.808,
  pennino_rf_hist 0.949 / 0.894, pennino_rf_hist_rich (+lat/lon) 0.949 / 0.886. Stack − pennino_rf_hist: +0.010
  [0.006, 0.016] known, −0.004 [−0.028, 0.020] new; − pennino_rf +0.025 / +0.082 [0.041, 0.132]. D4b: stack 0.820 /
  0.788; − pennino_rf_hist +0.005 [0.000, 0.009] / +0.008 [−0.004, 0.025]; − pennino_rf_hist_rich +0.003 / −0.003.
  **Re-applied rule: no qualifier → `stack_rf_gbm` (highest mean D4a validation AUROC 0.925); expected test label:
  competitive.**
- **Numeric track, validation (run `taprisk/numeric_val_v2`; train states → val states, n ≈ 480 UCMR4 systems).**
  R² (RMSE, log10): HAA5 quantile-GBM median **0.222** (0.487) vs Pennino-form RF regression 0.040, history median
  −0.064, state median −0.306; 90 % PI coverage **0.919** [0.888, 0.945], mean width 1.48 log10. TOC 0.356 vs RF 0.124;
  coverage **0.526** [0.459, 0.595] (under-covered; heavily censored target, 59 % < MRL). Bromide 0.053 vs RF 0.115;
  coverage 0.945. Secondary/descriptive; no recalibration (would be a new pre-registered change).
- **Configuration accounting (2026-11-10):** `configs_tried.jsonl` holds 50 D4 entries = 40 distinct pre-review
  configurations + 6 it2c **correction** re-runs of the same 6 it2 configurations (logged with status "correction") +
  2 numeric configurations logged twice (first run and the corrected `numeric_v2` run). Distinct configurations: **42**
  (40 classification + 2 numeric).
- **Delta-review corrections (2026-11-10, before the freeze; review "D4 TapRisk — delta review (2026-11-10)", PASS WITH
  CONCERNS).** E3: the `test_claim` registry row is written inside an abort handler, so an interrupt between the
  marker and the row writes `ABORTED.json` + a `test_aborted` row (never a marker-only state). E4: a rescore re-runs
  the whole frozen-state check (lead freeze, split, import-closure hash, pinned paths, data pins, environment pin,
  pinned-folder contents), not only the common hash. E2: the environment pin is the sha of
  `pip freeze --exclude-editable` (the editable `-e git+…@HEAD` line moved on every repo commit); it is checked by a
  D4-specific check in `taprisk_look`, and `expected.pip_freeze_sha256` is removed from the D4 pins. W-a: pinned
  folders are checked with `rglob` and any sub-folder is refused (hive discovery would read it). W-b: the numeric
  target aggregation filters to in-scope systems before any averaging. E1: the candidate record is regenerated last.
- **Numeric-track deviations (recorded verbatim, W-b):** (i) the §11 spec says "D4b feature set + rich features", but
  the implemented `feature_cols` omits the `rh_`/`sp_` rich features (StreamCat + base history + system + ambient +
  lat/lon only); (ii) HAA5 collection dates run to **2021-12-13**, not "2018–2020" as written; (iii) the log floors
  (HAA5 1, TOC 500, bromide 10 µg/L) were chosen **after** the first validation run; (iv) TOC 90 % PI coverage is
  **0.526** [0.459, 0.595] — under-covered, reported as such; (v) UCMR values are now filtered to in-scope systems
  before aggregation (W-b fix). The track is secondary and descriptive and never enters `claim_met` or H-G4.
- **Dry-run reproducibility after the delta-review fixes (2026-11-10 18:27 UTC, validation rows only, before the
  freeze).** The lead required the post-fix `--dry-run-on-val` predictions to be byte-identical to the pre-fix ones.
  **They are not, and cannot be with this code:** two post-fix runs of identical code differ from each other in the
  same columns and by the same size as each differs from the pre-fix run, so the difference is run-to-run noise and
  not an effect of the fixes. Cause: scikit-learn 1.9.1 sums the trees of a forest in thread-arrival order when it
  predicts with `n_jobs` > 1 (prediction with `n_jobs=1` is bit-exact in an isolated diagnostic). Of the 35 prediction
  columns, the 27 label, split, baseline, LightGBM and numeric-quantile columns are bit-identical in all three files;
  the 8 forest-derived columns (`s_pennino_rf`, `s_pennino_rf_hist`, `s_pennino_rf_hist_rich`, `s_stack_rf_gbm` and its
  3 seed columns, `pennino_rf_reg`) differ by at most 2.7e-15. Between the two same-code runs the three incumbent-RF
  columns have tie-break rank changes (109, 1 and 4 of 28,269 rows); the challenger columns have none. In the two
  `results.json` files all 176 AUROC values are bit-identical and 8 of 1,580 values differ by at most 4.4e-16. The
  sealed look is deterministic (a rescore reads the sealed file); a from-scratch re-run would reproduce the sealed
  predictions only to about 3e-15 in the forest-derived columns. Not changed here: making forest prediction
  single-threaded would be a hashed-code change and is left to the lead's decision.
- **Known wart on the E3 path (2026-11-10, fail-closed, not changed).** If the claim-row write itself fails, `claim()`
  aborts and the script's outer handler then calls `abort()` again: the second call raises `FileExistsError` (which
  masks the original exception in the traceback) and a second `test_aborted` registry row is written. The look stays
  closed and a second attempt is refused (`test_e3_script_level_interrupt_on_claim_row`).
- **Single-threaded forest prediction (lead ruling 2026-11-10 14:25 EST = 19:25 UTC; code changed 19:27 UTC; this note
  20:20 UTC; validation only, no test access).** This supersedes the last sentence of the reproducibility note above;
  that note is otherwise left as written. **Change, prediction only:** `taprisk_models.BalancedRF.predict` and the
  `pennino_rf_reg` forest in `taprisk_numeric.fit_predict` set `n_jobs = 1` on the fitted forests before predicting, so
  the trees are summed in a fixed order. **Fitting is unchanged** (threaded): a library-behaviour diagnostic on random
  arrays (no project data) showed that threaded fits are identical tree by tree, and identical to a single-threaded fit,
  for both the classifier and the regressor, so fitting was not a source of the differences. No configuration, feature,
  seed, label, split or selection rule changed, and nothing was re-selected. **Evidence — two dry runs on validation
  with the changed code** (run 3, 19:37–19:58 UTC; run 4, 19:59–20:18 UTC): the two `predictions.parquet` files are
  **byte-identical** (sha256 `804e7a4b0f945dc7095c351f17fac6a8669f20d8ef02371fe7572dc51f18764f`), and so are the two
  `results.json` files (sha256 `c4add9e73dc34833097f525406baf29cb35839962cd2adcc5dd6d09565ce3c8e`) and the printed logs.
  **Against the last run before the change** (run 2, threaded prediction): all 176 AUROC values (points, interval
  bounds, valid-bootstrap counts) are bit-identical, as are every AUPRC value, paired difference, interval, p-value,
  Holm decision, per-seed difference and claim flag. 9 of 1,580 result values differ, by at most 4.4e-16: 8 are RMSE/R²
  values or interval bounds of the incumbent numeric forest `pennino_rf_reg` (numeric track, secondary), 1 is an
  interval bound of the `pennino_rf` Brier score; this is the same family of values that already differed between the
  two runs before the change. In the prediction files 30 of 35 columns are bit-identical; 5 forest-derived columns
  differ by at most 2.2e-15 (`s_pennino_rf` 328, `s_pennino_rf_hist` 11, `s_pennino_rf_hist_rich` 1 and
  `s_stack_rf_gbm_seed2` 1 of 28,269 rows; `pennino_rf_reg` 708 of 1,432 rows), because the trees are now summed in tree
  order instead of thread-arrival order. A from-scratch re-run now reproduces the predictions exactly. Tests: 85 pass (two new:
  `test_forest_prediction_is_single_threaded_and_bit_reproducible`, `test_numeric_forest_prediction_is_single_threaded`).
