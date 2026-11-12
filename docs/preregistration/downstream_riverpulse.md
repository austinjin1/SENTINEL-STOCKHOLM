# D8 RiverPulse — pre-registration protocol (daily GPP / ER at reaches without sensors)

**Status:** v1.0, 2026-10-24, written and split frozen **before any RiverPulse model was fitted** (phase A).
Phase 1 = validation sites only; test-role sites are dropped right after role assignment.
**Implements:** RESEARCH_PLAN.md §8 D8, §14, §15.1, §15.5 (amendment 2026-10-24); docs/research/07 §d.8.
**Code:** `sentinel/models/downstream/riverpulse.py`, `scripts/train/downstream_riverpulse.py`,
`tests/test_downstream_riverpulse.py`.

## 1. Labels
- Powell Center daily metabolism, 356 rivers 2007–2016 (Appling et al. 2018, doi:10.5066/F70864KX;
  `processed/labels_powell_metabolism`). Targets: daily **GPP** (primary) and **ER** (secondary),
  g O2 m⁻² d⁻¹. Filter: model confidence H or M and day-level plausibility (GPP ≥ −0.5, ER ≤ 0.5);
  sensitivity: H only. These are model-derived estimates (inversions of diel DO), not measurements.
- Also reported: site-year mean GPP / ER (site-years with ≥ 180 labelled days), the "metabolic regime"
  level.

## 2. Inputs (observed only; circularity control)
- **Forbidden**: every daily predictor column of the label table (DO, DO saturation/amplitude, water
  temperature, depth, discharge, velocity, modelled shortwave, day length as published, turnover
  distance), K600, MCMC/CI columns and hydraulic-geometry coefficients (`riverpulse.FORBIDDEN`,
  enforced in `build_features` and unit-tested). No upstream-model features in phase 1.
- Allowed: DOY harmonics, latitude/longitude, gauge elevation, astronomical day length (computed),
  observed gridMET weather at the gauge's cell (same-day included — this is estimation of the label
  day at an unsensed reach, not a forecast; cells > 25 km away → missing), and StreamCat catchment /
  watershed attributes (179 columns) from `processed/downstream_catchment` (Powell published COMID;
  primary for every site) with the registry table as fallback. Model-derived columns are excluded
  (`iwi`, `ici`, `bankfulldepth`, `bankfullwidth`, `wettedwidth`, `thalwegdepth`). The driver refuses to
  train until `manifests/downstream_catchment.json` exists (flag `--allow-missing-catchment` for
  diagnostics only, recorded in the config, never registered); per-site StreamCat source counts are
  stored in registry `extra`.

## 3. Split (frozen; v2 after the pre-training review)
- **Current split: v2**, file `docs/preregistration/downstream_riverpulse_split_v2.json`, sha256
  **`20c550f9ab42ac343fdbdda1bd2eb845ad7e2081a06904037bbd19881b66b0d8`** (`riverpulse.SPLIT_SHA`).
  v1 (`downstream_riverpulse_split.json`, sha256 `4e4d064e…cd486`) is superseded (§7), kept on disk.
- Site roles are copied from the FINAL sonde split (`sonde_split.json`, sha256 fd4bb14f…; HUC4 basin
  groups with flow-connected sites together), so later M1/M6 predictions at val/test sites can be
  cross-fit: 314 sites by site, 37 by their HUC4's role in the sonde split; **flow override**: the 5
  Powell sites in HUC4 1307 (directly downstream of HUC4 1306 sonde-train sites) take the train role.
- Phase-1 counts (site-days / sites / HUC4 groups): train 204,695 / 158 / 49; val 37,941 / 31 / 12;
  test 156,836 / 115 (untouched). All days of a site share its role (spatial hold-out; the label period
  lies inside the upstream "train" time block).

## 4. Models and comparators (fit on train sites only)
| Row | Model |
|---|---|
| climatology | global DOY mean of train sites (±15-d smoothing) |
| location-only | mean DOY climatology of the 5 nearest train sites |
| light + season | ridge on quadratic expansion of srad (1 d, 7 d), day length, DOY harmonics |
| **incumbent (re-implemented)** | **Segatto et al. 2021-style random forest** on StreamCat + weather + season (300 trees, all training rows with per-tree bootstrap of 50 % of rows, native missing-value handling). No published held-out number exists to reproduce (Appling regime classes are descriptive), so this re-implementation is the incumbent (`docs/novelty_checks/downstream_riverpulse.md`). |
| **incumbent 2 (re-implemented)** | **Maavara et al. 2025-style RF** (*Science* 390:622–627, doi:10.1126/science.adu9843; Crossref-verified 2026-10-25): scikit-learn RandomForestRegressor trained on Appling et al. 2018 metabolism with 25 features of the classes listed on its Dryad page (slope, upstream area, temperature, precipitation, land-cover proportions; plus stream order, elevation, season). Exact feature list and modelled width/discharge inputs unavailable (Dryad files require login; width and discharge excluded by our rules) — closest public analogue (`riverpulse.MAAVARA_FEATURES`), same 4-config grid. The stronger of the two RF incumbents on validation is "the incumbent" for claims. |
| GBM challenger (phase 1; not a "SENTINEL" model — observed covariates only) | LightGBM, 3 seeds averaged; n_estimators by early stopping on the shared inner group split |

## 5. Endpoints and statistics
- Primary: **daily GPP R² (NSE)** at unseen sites; also RMSE, Spearman; ER and site-year means secondary.
- CIs: cluster bootstrap over **HUC4 basin groups** (val: 31 sites in 12 groups), B = 2,000; paired R²
  differences (challenger − comparator) with the same resampling; per-seed scores in registry `extra`.
- **Claim rule (§15.1), single test look after freezing:** "SOTA" iff on test sites the paired daily-GPP
  R² difference vs (i) the Segatto-style RF incumbent and (ii) the strongest simple baseline
  (climatology / location-only / light + season, chosen on validation) has a 95 % CI excluding 0, in ≥ 3
  seeds; ER is tested the same way, Holm over {GPP, ER}. Otherwise "competitive" / "negative".
- Secondary external check (optional, if time): StreamPULSE daily metabolism (`labels_streampulse`;
  de-duplicate gauges 05406457 / 05406469 on site and date).

## 6. Tuning parity, configurations, iteration
- Phase-1 run: **4 configurations per learned model** — light + season: ridge α ∈ {0.1, 1, 10, 100};
  RF: min_samples_leaf ∈ {5, 20} × max_features ∈ {0.33, 0.6} (100 trees during selection, 300 final);
  GBM: num_leaves ∈ {15, 31} × min_child_samples ∈ {50, 200} (early stopping) — all selected by daily-GPP
  MSE on ONE shared inner hold-out of 20 % of train HUC4 groups (`riverpulse.inner_groups`, seed 0).
- Any hyperparameter search uses **inner grouped CV on train sites only** (HUC4 groups), with the
  **same number of configurations for the RF incumbent and the GBM**; baselines are never weakened or
  removed; configurations are logged with `common.log_config_tried` and counted in any claim.
- Iteration plan (validation sites only, one change at a time): diagnose by site, region, season and
  confidence class → literature search for the current best unsensed-reach metabolism method → one
  targeted change (e.g. log/asinh target, light attenuation proxies from canopy/width, site-level
  regime model + daily anomaly) → re-run on validation. Test scored once per frozen configuration.

## 6b. Iteration log
- **base** (run 2026-10-25 01:19–01:40, `riverpulse/val_confHM`; 16 configurations): adds incumbent 2
  (Maavara-style RF) on lead guidance before the first fit. Result: daily GPP at unseen sites is not
  predictable above climatology by any model (all R² CIs span 0; best point R² 0.10, Maavara-style RF);
  ER: both RFs and the GBM beat the simple baselines, GBM does not beat the RFs. Diagnosis: one val site
  (USGS 03025500, mean GPP 20.8 g O2 m⁻² d⁻¹) dominates squared error; site-level magnitude does not
  transfer, while within-site temporal correlation is higher for learned models (median r 0.52 vs 0.42).
- **it1** (2026-10-25 01:50): one change — every learned model (ridge, both RFs, GBM) is fitted on
  asinh(target) and back-transformed; selection criterion = raw-unit inner MSE; same 4-config grids.

## 7. Deviations log
All entries: **2026-10-24, pre-training review, before any fit.**
- **W6 split re-frozen as v2**: Powell sites in HUC4 1307 (val in v1, directly downstream of HUC4 1306
  sonde-train sites) moved to the train role; v1 sha 4e4d064e…cd486 superseded. v1 build printed
  per-role site-day counts only (regression labels; no outcome statistics of the test role used).
- **C4** StreamCat from `downstream_catchment` (Powell COMIDs) with registry fallback; refuse to train
  without its manifest; model-derived StreamCat columns excluded; `downstream_catchment` added to the
  data manifests.
- **W4** RF incumbent on all training rows with native NaN handling (no median fill, no subsampling).
- **W5** bootstrap unit changed from site to HUC4 basin group.
- **W7** GBM renamed "GBM challenger"; tuning parity (§6) made explicit with equal 4-config budgets.
- **it2 — site-regime + daily-anomaly decomposition (pre-registered 2026-11-09, before any it2 code ran; lead
  direction: finish D8 by decomposition or a final negative report).** One change to the challenger (incumbents,
  baselines, split, asinh target of it1 unchanged): ŷ(site, day) = sinh( R(site) + A(site, day) ), where
  **R** = random forest (500 trees, min_samples_leaf 3, max_features 0.33; fixed, no search) trained on ONE row per
  training site — static features (StreamCat `sc_*`, lat, lon) → that site's mean asinh(target) over its training
  days; **A** = LightGBM with the it1-selected GBM hyperparameters on daily rows, target asinh(y) − site mean,
  features = the full it1 feature set. Rationale (diagnosis §6b): site-level magnitude does not transfer while
  within-site temporal correlation is higher for learned models, so separating the two removes the dominance of
  a few high-GPP sites from the daily learner. 2 configurations (R, A); 3 seeds for A. Scored like it1 (daily
  R² at val sites, HUC4 paired bootstrap) against both RF incumbents and the strongest simple baseline, for GPP
  and ER. Keep if it beats the stronger RF incumbent and the strongest simple baseline on GPP with CI > 0;
  otherwise D8 is closed with a final negative/competitive report (no further iteration).
- **it2 result (run `riverpulse/val_confHM_it2_decomp`, 2026-11-09): not kept → D8 closed (final report).** Daily
  R² at validation sites: GPP decomposition 0.132 vs Maavara-style RF 0.176 (Δ −0.043 [−0.097, 0.433]), vs DOY
  climatology +0.096 [0.048, 0.433]; ER 0.133 vs Maavara-style RF 0.207 (Δ −0.073 [−0.154, 0.069]), vs
  light + season +0.100 [0.012, 0.215]. **Final D8 verdict (validation only; no test look spent): negative for the
  SENTINEL challenger** — no learned challenger (GBM base / it1 asinh / it2 decomposition) beats the re-implemented
  Maavara et al. 2025-style RF incumbent for GPP or ER; all learned models beat the simple baselines for ER, and
  GPP at unseen sites remains barely predictable (best R² 0.18, CIs spanning 0 at the HUC4 level). Configurations:
  32 + 2 = 34. D8 counts toward H-G4 "implemented" only.
