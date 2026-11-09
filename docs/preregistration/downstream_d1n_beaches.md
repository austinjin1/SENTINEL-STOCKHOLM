# D1n — national freshwater-beach E. coli track (enlarged track of D1 SafeSwim)

**Status:** v1.0, written 2026-11-08 **before the split was frozen and before any model was fitted** on this
track. **Why:** RESEARCH_PLAN §15.5 amendment 2026-11-08 (items 5–7) — on the Chicago track (D1) the
strong methods tie within CIs (61 validation exceedances; logistic − feature-beach GM +0.036
[−0.001, 0.078]). This track is **added**; the Chicago track, its split and its results are unchanged and
always reported. Claims name the track ("D1n national").
**Code:** `sentinel/models/downstream/wqp_track.py` (labels, clusters, features, models, scoring),
`sentinel/models/downstream/d1n_beaches.py` (configuration), `scripts/train/downstream_d1n_beaches.py`.
**Data:** `processed/wqp_discrete` (manifest `wqp_discrete`, datasheet `docs/datasheets/wqp_discrete.md`;
WQP is public domain / US-government data), gridMET `forcings_daily`, WBD `wbd_huc12`. No new download.

## 1. Labels and unit
- WQP `ecoli` results (CFU or MPN/100 mL), `status_class == accepted`, QC activity types removed, sites
  with `site_class == lake` **and** "BEACH" in the station name. Chicago lakefront box
  (41.60–42.10 °N, −87.80 to −87.45 °E) excluded entirely (keeps every Chicago D1a/D1b test row out).
- Beach-day label: geometric mean of the day's results (floored at 1) > **235**; censored results kept
  only if the bound decides ('<DL' with DL ≤ 235 → below; '>DL' with DL ≥ 235 → above; '<' value = DL/2
  for the log feature).
- Beach cluster = union-find of stations ≤ 1.0 km; unit of rows = cluster-day (any station exceeding →
  1). Spatial group = HUC4 of the nearest WBD HUC12 centroid; **bootstrap unit = HUC8** (same way).

## 2. Split (frozen with `--freeze-split`; sha256 recorded in §8 and in `d1n_beaches.SPLIT_SHA`)
- Years: train 2004–2015, validation 2016–2019, test 2021–2025; 2020 excluded (COVID sampling gap);
  2026 (partial) is a reserve test season.
- HUC4 roles dealt from **site-day counts only** (no labels): groups > 10 % of all site-days → train;
  others largest-first to the role furthest below its target (test 0.20, val 0.15); ties by sha1(seed).
- Rows: `train` = train-HUC4 × train years; `val_known` = train-HUC4 × val years, primary rows = clusters
  with ≥ 1 training row; `val_new` = val-HUC4 × val years (beaches never seen; their earlier rows are
  dropped, so they start cold in 2016); test analogues `test_known` (train+val HUC4 × 2021–2025),
  `test_new` (test HUC4 × 2021–2025). Test rows are dropped before any feature is built.

## 3. Decision time and features (observed only; no upstream-model outputs)
Morning forecast for day *d*: culture results from samples dated ≤ *d* − 1 (USGS previous-day
convention, as D1a). Own-cluster history (n, EB rate, last value/exceedance/age, 30/365-d means, 365-d
max, same-year exceedances), network (other clusters ≤ 25 km: results in [d−3, d−1]; count, mean/max log,
any exceedance, nearest distance), gridMET windows 1/2/3/7/14/30 d ending d − 1 at the nearest SENTINEL
cell ≤ 30 km (anomalies vs a 2004–2015 DOY climatology), lat/lon, DOY harmonics, weekday.

## 4. Models and comparators (same 4-configuration budget for every learned model)
- Simple baselines: climatology (cluster × month, shrunk), location-only (cluster rate shrunk to an
  inverse-distance 5-NN of training clusters), history rate (EB, ≤ d − 1), persistence (previous result,
  scored only where it is ≤ 3 days old).
- **Incumbent (re-implemented): USGS nowcast / Virtual-Beach-style MLR** of log10 E. coli — one ridge MLR
  per beach cluster with ≥ 50 training beach-days, a pooled MLR otherwise; predictors: rain 1/2/3/7 d,
  Tmax 3 d, wind 1 d, previous result and log-age, 365-d mean, DOY harmonics. Turbidity, wave height and
  stream discharge used by many site models are not nationally available (disclosed limitation).
  Ridge α ∈ {0.1, 1, 10, 100}. Context only: USGS goal ≥ 80 % correct and ≥ 5 points over persistence.
- Challengers: L2 logistic (C ∈ {0.01, 0.1, 1, 10}); LightGBM (num_leaves {7, 15} × min_child {20, 100},
  n_estimators by early stopping, 3 seeds averaged).
- Selection for every learned model: inner split of training years (fit 2004–2012, score 2013–2015),
  criterion AUROC; then refit on all training rows. Validation never used for selection.

## 5. Endpoints and statistics
- Primary: **AUROC** on `val_known` and on `val_new` (each reported; claim needs both at the test look).
  Secondary: AUPRC, Brier, persistence-eligible rows.
- Paired HUC8-cluster bootstrap (B = 2,000) of challenger − comparator; Holm across comparators per
  challenger × set.
- **Claim rule (§15.1, single test look after freezing):** "SOTA (D1n national)" iff on both test sets the
  frozen challenger beats (i) the VB-MLR incumbent and (ii) the strongest simple baseline (best of
  climatology / location-only / history rate on validation) with paired CIs excluding 0 (Holm), in ≥ 3
  seeds; otherwise "competitive"/"negative".
- **A-priori difficulty strata** (reported for every model, no separate claim): rain-event days (2-d
  rain > training P90), cold-start beaches (< 10 prior beach-days), no own result within 3 days.

## 6. Power check (from `--build-only`, before any fit) — §8
Hanley–McNeil SE at AUROC 0.75 and the paired minimum detectable difference (80 % power, ρ = 0.6;
not inflated for clustering).

## 7. Iteration
As D1 §5b (validation only, one change at a time, every configuration logged). Test scored once per frozen
configuration; 2026 is the reserve season for a post-test configuration.

## 8. Frozen split, counts and power check
- **Split frozen 2026-11-08:** `downstream_d1n_beaches_split.json`, **sha256 `aa601b50…e6f96861`**
  (full: aa601b505530f2d93bf90881e106d882ffc46efdf38ac9cda2db1816e6f96861); HUC4 roles train 30 / val 27 /
  test 28 groups; 1,419 beach stations → 858 clusters (1 km); 263,368 eligible beach-days 2000–2026
  before year/role rules.
- **Phase-1 frame (train + val only; test rows dropped before features):** train 56,987 cluster-days /
  8,827 exceedances / 271 clusters / 21 HUC4; `val_known` (clusters with training rows) 22,672 / 2,609 /
  247 clusters / 67 HUC8; `val_new` 5,769 / 613 / 151 clusters / 53 HUC8. gridMET missing (> 30 km) 28.8 %.
- **Power check:** AUROC SE at 0.75 = 0.006 (`val_known`), 0.012 (`val_new`); paired MDE ≈ 0.014 / 0.029
  (not inflated for HUC8 clustering). Adequate (Chicago D1b validation: 61 exceedances).
- **Smoke test disclosure:** before the phase-1 run the code was exercised once on TRAINING years only
  (fit 2004–2012, score 2013–2015 = the inner selection block) to catch errors; no validation row was
  scored and nothing was changed afterwards except bug fixes (none needed).

## 9. Iteration log (validation only)
- **phase 1** (run `d1n_beaches/phase1`, 2026-11-08; 12 configurations, 4 per learned model). Selected:
  logistic C 1; GBM 15 leaves / min_child 20; VB-MLR ridge α 100. AUROC (HUC8 paired CIs, Holm):
  | set (n / pos) | climatology | location | history rate | persistence* | **VB-MLR incumbent** | logistic | GBM |
  |---|---|---|---|---|---|---|---|
  | val_known (21,817 / 2,548) | 0.709 | 0.707 | 0.722 | 0.679 | 0.750 | 0.789 | **0.800** |
  | val_new (5,769 / 613) | 0.497 | 0.622 | 0.712 | 0.640 | 0.767 | **0.783** | 0.781 |
  *persistence scored only where the previous result is ≤ 3 days old.
  - GBM − incumbent: known **+0.050 [0.029, 0.074]** (Holm); new +0.013 [−0.005, 0.042].
  - logistic − incumbent: known **+0.039 [0.018, 0.063]**; new **+0.016 [0.002, 0.040]** (both Holm).
  - Both challengers − strongest simple baseline (history rate): known +0.067 / +0.078, new +0.071 /
    +0.069, all CIs > 0 (Holm).
  - A-priori strata (val_known): rain days (2-d rain > train P90; 1,700 / 446) GBM 0.756 vs incumbent
    0.711; no own result within 3 d (7,580 / 661) GBM 0.795 vs 0.743. val_new cold-start beaches
    (1,306 / 89): logistic 0.791, incumbent 0.787.
  **Status:** the §5 rule (both sets, vs incumbent and strongest simple baseline) holds on validation for
  the **logistic**; the GBM meets it on known beaches only. **Freeze candidate (D1n): L2 logistic, C = 1,
  phase-1 features** (selection between the two pre-specified challengers made on validation —
  disclosed). Awaiting lead go-ahead.

## 10. Pre-test gate deviations (2026-11-08, before any test look)
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
- **Test-time fit = train + val rows for the claimed logistic AND every comparator** (VB-MLR incumbent,
  simple baselines, GBM with 503 trees), frozen hyperparameters; "known" = clusters with ≥ 1 fit row. (§4
  said "refit on all training rows"; the train+val refit was coded but not written — recorded here.)
- **Incumbent re-tuning (§15.5 item 4):** phase 1 selected ridge α = 100 at the top of its grid with inner
  AUROC still rising. The VB-MLR grid is widened to α ∈ {0.1, 1, 10, 100, 10³, 10⁴, 10⁵, 10⁶} and re-selected
  on the **inner training-year split only** (fit 2004–2012, score 2013–2015); validation numbers are then
  re-reported with the re-selected α; the claimed logistic is unchanged.
- **Holm = all four primary comparisons (2 comparators × 2 sets, m = 4)** — stricter than §5's per-set Holm.
- Strata thresholds (rain P90) use training-role rows only. **"New beach" = beach in a never-fitted HUC4,
  not a new water body:** 63 of 160 test-new clusters lie on Great-Lakes HUC4s that share a lake with fitted
  HUC4s (nearest fitted cluster median 97 km, min 2.6 km) — disclosed. Logistic was chosen over the GBM on
  validation (two pre-specified challengers).
- **Incumbent re-tuning result (run `d1n_beaches/alpha_wide`, 2026-11-08):** inner AUROC by α: 0.1–100 rising
  to 0.7734 at α = 100, then 0.7579 (10³), 0.7321 (10⁴), 0.7237 (10⁵), 0.7209 (10⁶) → **α = 100 re-selected,
  now interior**. Validation unchanged: logistic − VB-MLR known +0.039 [0.018, 0.063], new +0.016
  [0.002, 0.040] (p 0.032). 8 more configurations (D1n total 20).
