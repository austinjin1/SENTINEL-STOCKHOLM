# D2 ToxinWatch — pre-registration protocol (microcystin > 8 µg/L at Washington lakes)

**Status:** v1.0, 2026-10-24, written and split frozen **before any ToxinWatch model was fitted** (phase A).
Phase 1 = validation only; the test role is never loaded by phase-1 code (`common.drop_protected` right
after role assignment; `common.log_metric` refuses split names containing "test").
**Implements:** RESEARCH_PLAN.md §8 D2, §14, §15.1, §15.5 (amendment 2026-10-24); docs/research/07 §d.2.
**Code:** `sentinel/models/downstream/toxinwatch.py`, `scripts/train/downstream_toxinwatch.py`,
`tests/test_downstream_toxinwatch.py`.

## 1. Labels and unit
- Source: WA Toxic Algae database (`processed/labels_wa_toxic_algae`, vintage 2026-10-22; datasheet
  `docs/datasheets/labels_wa_toxic_algae.md`). Licence restricted (King County terms): only aggregate
  metrics leave the repo.
- Unit: **site-day with a microcystin result**; label `exceed_epa_8ugL_any` (any result that day > 8 µg/L,
  EPA 2019 recreational value; censored results resolved only when the bound decides). 4 undetermined
  site-days are dropped.
- **Lake cluster** (unit of split and bootstrap): site IDs are merged by (a) normalised waterbody name
  (parenthetical waterbody > text before "," / " - " > name; "lake" removed) within a county (rivers:
  across counties), (b) a curated alias table for King County swim beaches on Lake Washington / Lake
  Sammamish (`WATERBODY_ALIASES`), and (c) coordinates ≤ 2.0 km (union-find). 525 sites → 345 clusters.
  Residual risk: un-aliased beaches of large lakes could form separate clusters (reviewer to spot-check).

## 2. Decision time and inputs (observed covariates only; no upstream-model features)
- Decision at the morning of sampling day *d*. Lab toxin results take days: every prior-result feature
  uses only cluster-days dated **≤ d − 7** (sensitivity: ≤ d − 3). Weather windows end at d − 1
  (gridMET, nearest available cell, distance recorded as `wx_cell_km`, median 21 km).
- Features (30): DOY harmonics, lat/lon, river flag; lake history ≤ d−7 (n samples, n exceedances,
  EB-smoothed rate with training prevalence, days since last result, last exceedance, last quantified
  log value, 365-d max log value / exceedances / n, same-year exceedances); antecedent weather
  (precip 3/7/30 d, Tmax 3/7/30 d, Tmin 7 d, Tmax anomaly 7/30 d vs a train-years DOY climatology,
  srad, wind, VPD, dry days).
- `scum_info` (an additional-sample-information form exists) is a property of the sample, not known
  before sampling, and marks targeted scum sampling → **not a feature**; results are reported stratified
  by it. County/WRIA enter only through coordinates.

## 3. Splits (frozen; v2 after the pre-training review)
- **Current split: v2**, file `docs/preregistration/downstream_toxinwatch_split_v2.json`, sha256
  **`13852e4a97490da97ddafeefeb757074c2b832c7cb361ab9bf85c077e11bef85`** (`toxinwatch.SPLIT_SHA`;
  assignments `lake_role`, `site_cluster`; meta holds the rules). v1
  (`downstream_toxinwatch_split.json`, sha256 `04e14820…cdb383`) is **superseded** (see §7) and kept
  unchanged on disk.
- v2 lake roles (rule chosen from row counts only): clusters holding > 10 % of all site-days take the
  train lake role (only Lake Washington, 43.8 %); the remaining clusters, per stratum (≥ 200 site-days /
  other), are dealt largest-first to the role furthest below its site-day target (test 0.2, val 0.2,
  train 0.6; ties by sha1). Outcome: 206 train, 69 val, 69 test clusters.
- Row rule: test = test lake **or** year ≥ 2022; `val_known` = train lake, 2019–2021 ("known lakes,
  future"); `val_new` = val lake, 2019–2021 ("new lakes", primary, starts after training ends);
  `val_new_pre2019` = val lake, 2007–2018 (exploratory only: same dates and gridMET cells as training);
  `train` = train lake, 2007–2018.
- Phase-1 counts (site-days / exceedances / clusters): train 6,586 / 409 / 132; val_known 2,052 / 215 /
  101; val_new 213 / 51 / 31 (2019–2021); val_new_pre2019 979 / 156 / 45. Test: 5,799 site-days,
  247 clusters (row counts only).
- `val_known` primary = clusters with ≥ 1 training row; clusters without training rows are reported
  separately (exploratory).

## 4. Models and comparators (all fit on train rows only)
| Row | Model |
|---|---|
| prevalence | training exceedance rate |
| climatology | monthly exceedance rate (train) |
| location-only | inverse-distance k-NN (k = 5) smoothed rate of train lake clusters |
| persistence | last exceedance known at d − 7 (tie-break by its value) |
| lake history | EB-smoothed lake rate ≤ d − 7 |
| **incumbent (re-implemented)** | **INLA-class spatiotemporal model** (Schaeffer et al. 2024 / Myer et al. 2020 class): logistic model with a partially pooled (EB-shrunk) lake effect, an L2-penalised lake one-hot (ridge-shrunk lake random intercept), a spatial-neighbourhood effect (k-NN train-lake rate excluding the own lake; discrete stand-in for the SPDE field), 2 seasonal harmonics, an AR(1)-type state term (last exceedance × age) and the same antecedent-weather windows the GBM sees. The published EPA forecast uses satellite WHO-alert labels and cannot be re-run on toxin labels; this re-implementation of its model class is the incumbent (choice documented in `docs/novelty_checks/downstream_toxinwatch.md`). |
| logistic | all features, median imputation + indicators, standardised |
| GBM challenger (phase 1; not yet a "SENTINEL" model — it uses observed covariates only) | LightGBM, 3 seeds (0, 1, 2) averaged; n_estimators by early stopping on an inner time split of training rows (fit ≤ 2015, stop on 2016–2018) |

## 5. Endpoints and statistics
- Primary metric: **AUROC** (threshold-free); co-primary reported: AUPRC (with prevalence), Brier for
  probabilistic rows; stratified by scum flag.
- Primary evaluation sets: `val_known_2019_2021` (clusters with training rows) and
  `val_new_lakes_2019_2021`; exploratory: `val_known_2019_2021_no_train_rows_EXPL`,
  `val_new_lakes_2007_2018_EXPL`. Test analogues: known lakes 2022–2026, test lakes 2022–2026 (pre-2022
  test-lake rows exploratory).
- CIs: cluster bootstrap over **lake clusters**, B = 2,000; paired AUROC/AUPRC differences (SENTINEL −
  each comparator) with the same resampling. Per-seed scores in registry `extra`.
- **Claim rule (§15.1), on the single test look after freezing:** "SOTA" iff on BOTH test sets
  ("known lakes, 2022–2026" and "test lakes") the paired AUROC difference vs (i) the INLA-class incumbent
  and (ii) the strongest simple baseline (the best of persistence / location-only / climatology / lake
  history, chosen on validation) has a 95 % CI excluding 0 (Holm over the 2 sets), holding in ≥ 3
  seeds. Otherwise "competitive" / "negative". No "first-in-class" claim is sought (prior cyanotoxin ML
  exists).
- Secondary, exploratory (underpowered): NLA microcystin > 8 µg/L as an unbiased cross-sectional
  check — only 19 / 18 / 13 / 28 exceeding visits in 2007 / 2012 / 2017 / 2022; reported as
  "preliminary" with Clopper–Pearson intervals if run. WQP microcystin is not used in phase 1.

## 6. Tuning parity, configurations, iteration
- Phase-1 run: **4 configurations per learned model** (incumbent: L2 C ∈ {0.01, 0.1, 1, 10}; logistic:
  C ∈ {0.01, 0.1, 1, 10}; GBM: num_leaves ∈ {7, 15} × min_child_samples ∈ {20, 60}, n_estimators by
  early stopping), each selected by log-loss on the inner time split of training rows (fit ≤ 2015,
  score 2016–2018); validation data never used for selection.
- Any hyperparameter search uses **inner CV on training rows only** (time-ordered inner folds), with
  the **same number of configurations for the incumbent, logistic and GBM**; baselines are never
  weakened or removed. Every configuration is logged with `common.log_config_tried`, and the count is
  reported next to any claim.
- Iteration plan (validation only, one change at a time): diagnose by evaluation set, lake and
  scum stratum → literature search for the current best microcystin-forecast method → one targeted
  change (e.g. lake random effects in the GBM, embargo-aware recency weighting, cell-weather
  representativeness filter) → re-run on validation. The test block is scored once per frozen
  configuration (§15.5 amendment); the first test result stays reported.

## 7. Deviations log
All entries below: **2026-10-24, pre-training review, before any fit.**
- **C1 split re-frozen as v2.** v1 put Lake Washington (43.8 % of site-days) in the test-lake role and
  left 3,396 training site-days. New rule from row counts only (§3); v1 sha 04e14820…cdb383 superseded.
  Disclosure: the v1 build printed per-role counts **including test-role exceedance counts**; the v2
  rule was chosen from site-day counts only, and test exceedance counts are no longer written.
- **C2** alias `moses lake south` → `moses` (Moses Lake South had been a separate cluster); other
  same-name clusters > 2 km apart were spot-checked (all in different counties / different lakes).
- **C3** primary new-lake evaluation restricted to 2019–2021 (after training ends); pre-2019 new-lake rows
  exploratory (41.7 % shared an exact gridMET cell-date with a training row).
- **W1** incumbent strengthened (lake one-hot L2 intercept, same weather windows) and given the same
  tuning budget as the challengers (§6). **W2** `val_known` restricted to clusters with training rows.
  **W3** no test positive counts written. **W7** GBM renamed "GBM challenger". Weather from cells
  > 25 km away set to missing.

## 8. Iteration log (validation only; §6)
- **base** (2026-10-24 22:19, run `toxinwatch/val_emb7`): 12 configurations tried (4 per learned model).
  Diagnosis: on known lakes the GBM challenger beats the incumbent but not persistence (CI includes 0);
  scum-flagged samples are the hard stratum (AUROC 0.76 vs 0.97 without scum); on new lakes the GBM does
  not beat the lake-history baseline. All numbers in the registry (`downstream`, `toxinwatch/mc_exceed8`).
- **it1** (2026-10-24 23:00): one change — add sampling-SITE history (beach-level, ≤ d − 7) and lake
  sampling intensity (cluster-days in 30 / 90 d ending d − 7; reactive sampling follows blooms) to the
  logistic and GBM feature sets; incumbent design unchanged; same 4-config budgets.
- **it2** (2026-10-25 01:05): one change to the comparator set — a closer INLA analogue
  `st_glmm_incumbent_v2` (= v1 + AR(1)-type exponentially decayed last result / last exceedance at τ = 14
  and 60 d, respecting d − 7, + 365-d max, + an additive cubic-spline spatial smooth in lat/lon in place
  of the SPDE field), same 4-config L2 grid. Challenger features as in it1. For any claim the stronger of
  the two incumbent variants on validation is "the incumbent".
  Result (run `toxinwatch/val_emb7_it2`, 2026-10-25 01:40–02:05): incumbent v2 is stronger on known lakes
  (AUROC 0.925 vs 0.916) but weaker on new lakes (0.628 vs 0.706; the spatial spline extrapolates poorly).
  Against v2 the GBM challenger's known-lake gain is +0.010 [−0.001, +0.055] (CI includes 0). Verdict after
  3 runs (40 configurations): **competitive** — no §15.1 label on validation; persistence (known lakes) and
  lake-history rate (new lakes) are not beaten with CI excluding 0.
- **it3 — national training rows (pre-registered 2026-11-08, before any it3 code was run; lead approval
  2026-11-08).** One change vs it2: the learned models' training set = WA training rows (unchanged) **+ the
  D2n national training rows** (WQP microcystin, `d2n_microcystin` split sha 0d9eb684…, train-role HUC4 ×
  2006–2017 only; WA excluded there; no row of any frozen test block). National rows are mapped onto the
  it1 feature names: DOY harmonics, lat/lon, `is_river` = 0, lake history (n, exceedances, smoothed rate
  recomputed with the WA a = 2 and WA training prevalence, days since last, last exceedance, last log value,
  365-d max log / exceedances / n, same-year exceedances), `tw_lake_n_30d` (= cluster-days in the 30 d
  ending d − 7), the 15 WA weather windows (same gridMET code; national DOY climatology on 2006–2017);
  WA-only features (`tw_site_*`, `tw_lake_n_90d`, `tw_lake_exceed_90d`) are missing for national rows;
  plus a source flag `src_national` for the learners.
  - **Parity:** both INLA-class incumbent variants are ALSO refit on the pooled rows (`*_pooled`, same
    4-config grids, same inner split); the it2 WA-only incumbents are kept unchanged. **The incumbent for
    the claim is the strongest of the four variants on each validation set.** Simple baselines are fit on
    WA training rows only (identical to it2).
  - Evaluation unchanged: WA `val_known_2019_2021` and `val_new_lakes_2019_2021`, lake-cluster paired
    bootstrap, 3 GBM seeds. Keep it3 if the GBM (or logistic) beats the strongest incumbent on BOTH primary
    sets with CI excluding 0; otherwise report and stop D2 iteration (D2 stays "competitive").
  - **it3 result (run `toxinwatch/val_emb7_it3`, 2026-11-08): kept.** 12,357 national training rows (751
    exceedances) added. Pooling also strengthens the incumbents: v2 known 0.925 → 0.930 (pooled), new lakes
    0.628 → 0.747. AUROC known / new: GBM **0.939 / 0.807** (seeds 0.939/0.938/0.939; 0.810/0.803/0.806),
    logistic 0.935 / 0.819. Strongest incumbent on both sets = `st_glmm_incumbent_v2_pooled`.
    GBM − strongest incumbent: known **+0.009 [0.001, 0.045]** (p 0.022), new **+0.060 [0.006, 0.163]**
    (p 0.030) → Holm over the two sets rejects both. GBM − strongest simple baseline: known (persistence)
    +0.034 [0.008, 0.060]; new (lake history) +0.051 [0.002, 0.126]. Logistic − v2_pooled: +0.006
    [−0.006, 0.037] known (not significant), +0.072 [0.024, 0.153] new.
    Caveat: the known-lake margin is thin (lower bound 0.001); 72 configurations tried in D2 so far.
    **Freeze candidate (D2): GBM challenger it3** (WA + D2n national training rows, it1 features +
    `src_national`, num_leaves 7, min_child_samples 60, early-stopped on the inner 2016–2018 split, 3 seeds),
    test-time fit = the same rows (WA train lakes 2007–2018 + D2n train rows), frozen hyperparameters;
    incumbent for the claim = strongest of the four INLA-class variants. Awaiting lead review.
- **it3 test-look multiplicity rule (lead decision 2026-11-08, written before any test look).** The claim is
  decided by the pre-registered §5 rule: Holm over the two test sets, applied separately to (i) GBM −
  strongest incumbent (`st_glmm_incumbent_v2_pooled`) and (ii) GBM − strongest simple baseline (known:
  persistence; new lakes: lake history rate), each CI excluding 0. The Holm across all four primary
  comparisons is reported alongside as a stricter sensitivity (`claim_met_all_four_holm` in
  `test_look/results.json`); **if the §5 rule passes but the all-four Holm fails, the D2 claim is labelled
  "not robust to all-comparison correction"** in RESULTS.md and the paper. (On validation the dry run gives
  exactly this case: §5 rule passes, all-four Holm fails.)
- **Pre-test gate deviations (2026-11-08, before any test look):**
- **Shared (all five candidates), pre-test gate 2026-11-08 (lead decisions; written before any test look):**
  (a) **seed rule**: a claim additionally requires the paired gain to be > 0 in each of ≥ 3 seeds (per-seed
  point estimates on the test block, vs every primary comparator); per-seed predictions are saved;
  deterministic challengers (L2 logistic) meet it trivially. (b) every input data file is sha256-pinned in
  `downstream_<app>_data_pins.json` and verified before the claim. (c) predictions (incl. per-seed) are saved
  once before any registry write; after a crash only re-scoring of the saved predictions is allowed
  (`--rescore-from-saved`), never re-fitting. (d) **No git (user instruction):** pre-registration order
  cannot be proven from version control; evidence = registry timestamps, file mtimes and a
  `prereg_snapshot` registry row (sha256 + mtime of protocol, record and test script) appended when the
  FREEZE file is written.
  - **Strongest simple baseline at known lakes = `location_knn`** (validation AUROC 0.9104 > persistence
    0.9050 > lake history 0.8829; the §5 rule "best of persistence / location-only / climatology / lake
    history, chosen on validation"); it3 text and the earlier record wrongly named persistence. New lakes:
    lake history rate (0.7559 > persistence 0.7374), unchanged. Validation verdict unchanged (GBM −
    location_knn +0.029 [0.006, 0.134], p 0.004).
  - **Configurations disclosed: 88** (72 D2 + 16 D2n phase-1 configurations seen before it3 was designed).
  - The pre-registered **scum-flag breakdown** (§5) is computed at test (scum / no-scum AUROC, reported).
  - **Order of events:** the multiplicity decision text above was written AFTER the validation dry run had
    shown the all-four Holm failing (test script 16:29, dry run 16:35, protocol 16:37); the governing rule
    is the original §5 text of 2026-10-24.
  - **Dominant clusters:** Lake Washington ≈ 60 % of known-lake test rows and `wa_lc0026` ≈ 39 % of new-lake
    rows, so these clusters dominate the lake-bootstrap CIs. Pooled-incumbent parity: learners see
    `src_national`; the pooled StGlmm has no source offset and the pooled v2 spline spans CONUS (as
    pre-registered; WA-only variants remain in the family).
