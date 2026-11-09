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
