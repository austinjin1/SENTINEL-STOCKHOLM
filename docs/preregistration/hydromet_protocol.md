# SENTINEL-HydroMet (M6) — Pre-registration protocol

**Status:** DRAFT v0.11 (2026-11-09; v0.2 of 2026-10-22 plus the §9 deviations; harder tracks in §10–§11). It incorporates the M6 pre-training review
(`REVIEW_REPORT_PRE_TRAINING.md`, M6 section: C1–C3, W1–W6) and the coordinator's decisions.

- The protocol is frozen when `docs/preregistration/FREEZE_hydromet.json` is written. That happens only
  after re-review; the file is not created in this phase.
- This file is part of the hashed frozen code (`sentinel/models/hydromet/guard.py::frozen_files`), so any
  edit after the freeze invalidates the freeze.

**Authors:** Bryan Cheng (AI-assisted). **Implements:** RESEARCH_PLAN.md §5.6, §14 and §15 (the M6 primary
endpoint of H-G1).

## 1. Objective and hypotheses
We estimate daily stream water quality at reaches **without any in-situ sensor input**. The only inputs
are observed daily weather (reanalysis/nowcast), static catchment attributes and the calendar. We test
whether SENTINEL's EA-LSTM beats **both** the strongest GBM baseline **and** the strongest
physics/process baseline at **unseen basin groups**.

- **H1 (primary, per model × endpoint).** The SENTINEL model's median per-site KGE exceeds that of the
  best comparator in *each* of the two comparator families (§6). This is an intersection–union test with
  basin-group resampling and Holm correction (§7).
- **H2 (secondary).** Secondary metrics, secondary targets and ablations (§7). These are reported
  descriptively, without multiplicity control.

## 2. Data, split and eligibility
- **Data.**
  - Targets: USGS NWIS daily statistics (`processed/sonde_daily`, manifest `sonde_daily`).
  - Forcings: gridMET daily (`processed/forcings_daily`, manifest `forcings_daily`).
  - Static attributes: EPA StreamCat (`processed/catchment/streamcat_sites.parquet`, manifest `streamcat`)
    plus lat/lon/elevation from `processed/sites`.
- **Split.** The frozen FINAL split `docs/preregistration/sonde_split.json`.
  - File and canonical sha256: **`fd4bb14fe7304d05310e286e9aced22675fd6d07aa4dcac6415ed039a1e06a80`**.
  - Roles are assigned to **basin groups**: HUC4 units merged across NLDI flow-connected stations
    (`sonde_split_report.json`). It is used as-is.
  - **No other split is ever loaded by M6 code** (`data.load_split` refuses non-FINAL splits, review C3a).
    This includes smoke tests.
- **Smoke tests** use at most 50 FINAL-split **train-role** sites only, scored on their inner groups. They
  never use val- or test-role sites and are never registered.
- **Time blocks** (`sentinel/data/sonde_splits.py::TIME_BLOCKS`): train WY2008–2021, val WY2022–2023,
  test WY2024–2025, test_b WY2026. Non-train blocks are scored only from 42 days after their start
  (embargo).
- **Site eligibility** (`data.eligible_sites`). A site is eligible if all of these hold:
  - it is in the split;
  - its USGS site type is ST, ST-TS, ST-CA or ST-DCH;
  - its forcing source is gridMET;
  - StreamCat attributes are present (CONUS, with an NHDPlus COMID).

  Sites are excluded only by these rules, never by performance. Counts are written to each run's
  `data_info.json`.
- **Scoring eligibility.** A site is scored for a target in a block only if it has ≥ 60 approved
  (non-provisional) daily values there, on cells where every compared model has a finite prediction.
  Dropped cells are counted per model (`rows_*.json`).
- **Fitting data.** Fitting uses train-role sites × the train block only. Every normalisation statistic,
  climatology, DO ratio and 7-DADMax offset is computed from this cell.
- **Inner groups.** Early stopping, LSTM grid selection, GBM grid selection, the air→water window choice
  and air2stream donors use train-role **inner** basin groups only:
  - `inner_frac = 0.15`;
  - `inner_seed = 0`, identical for every model seed and every baseline;
  - groups are taken in a seeded random order and each whole group is added while the inner share stays
    ≤ `inner_frac` (greedy fill, `data.inner_split`). With the FINAL split this gives 256 of 1,712
    eligible train-role sites in 13 groups.
- **Use of val-role sites.** Val-role sites are used only for evaluation and for selecting comparators
  (§6).

## 3. Inputs (enforced in code)
- **Forcing.**
  - Variables: gridMET precipitation, Tmax/Tmin/Tmean, shortwave radiation, wind, RH max/min, specific
    humidity, VPD and reference ET.
  - Primary **nowcast** mode: the model for day t uses days ≤ t.
  - The `lagged` mode (≤ t − L) is secondary.
  - From v0.10 (`upstream_forcing: append`): the same gridMET variables averaged over the site's total upstream
    drainage basin (NLDI basin of the StreamCat COMID; area-weighted fractional gridMET cells;
    `processed/hydromet_upstream_forcing`, datasheet `docs/datasheets/hydromet_upstream_forcing.md`) are appended
    as 11 further channels (`ws_*`). Observed weather only; the basin polygon comes from NHDPlus V2 geometry, not
    from any gauge record.
  - Observed weather is never called a forecast. An NWP mode (`nwp_archive`) is out of scope.
- **Static.** lat, lon, elevation and StreamCat attributes. Model-predicted stream temperature and NRSA
  condition are already excluded at the datasheet level.
- **Calendar.** DOY harmonics.
- **Never used:**
  - in-situ measurements, including gauged discharge and stage;
  - monitoring-network metadata;
  - NWIS drainage area.
- **Gauge-derived static attributes (review W4).**
  - StreamCat `bficat/bfiws` (Wolock BFI, interpolated from gauge records) and `runoffcat/runoffws`
    (water-balance runoff) are used in the primary analysis. So is the NWIS gauge-datum elevation
    `elev_m`.
  - None of these is a water-quality measurement.
  - Their influence is quantified by the pre-registered ablation in §7.
  - IWI/ICI and predicted channel dimensions are landscape or geometry model products.
- **Enforcement.** `data.assert_allowed_inputs`, and tests showing that:
  - inputs are invariant to targets;
  - predictions of every baseline AND the EA-LSTM are invariant to targets outside train-role × train
    block;
  - forcing windows are causal;
  - normalisation is train-only.

## 4. Targets
| Class | Target | Definition | Model scale |
|---|---|---|---|
| **Primary** | `temp_c_mean` | daily mean water temperature (°C) | linear |
| **Primary** | `temp_c_7dadm` | trailing 7-day mean of daily max temperature (all 7 days present) | linear |
| **Primary** | `do_mgl_mean` | daily mean DO (mg/L) | linear |
| **Primary** | `do_mgl_min` | daily minimum DO (mg/L) | linear |
| Secondary | `spcond_uscm_mean` | daily mean specific conductance (µS/cm) | log(x + 1) |
| Secondary | `turb_fnu_mean` | daily mean turbidity (FNU only) | log(x + 0.1) |
| Secondary | `no3no2_mgl_mean` | daily mean nitrate+nitrite (mg/L as N) | log(x + 0.01) |

The coordinator confirmed the classes on 2026-10-22: turbidity, nitrate and SC are secondary.

**H-G1 mapping.** For RESEARCH_PLAN H-G1 (M6: "≥ 2 of 5 targets"), a target variable counts as won if at
least one of its primary endpoints is **won** (§7). The M6 claim therefore requires **both** temperature
**and** DO to be won (confirmed). Otherwise the result is reported as "competitive" or "negative" (§15.1).

## 5. Metrics and resampling unit
- **Primary metric.** The median across sites of per-site KGE (`sentinel.eval.metrics.kge`, original
  units) on the scored cells.
- **Resampling unit (review C1).**
  - Sites cluster strongly in few basin groups. The val cell has about 12 groups, and one group holds
    about 39% of the sites.
  - The primary resampling unit for every CI and p-value is therefore the **basin group** (the split's
    spatial unit). A group is drawn with all of its scored sites, using percentile bootstrap with
    B = 2,000.
  - Site resampling is reported as a labelled sensitivity analysis (`*_site` columns).
  - `n_groups` is reported next to `n_sites` everywhere.
- **Secondary metrics.**
  - Per-site NSE and RMSE (plus MAE and bias); for log targets, also NSE and RMSE on the model scale.
  - Medians come with group-bootstrap CIs.
- **Pooled cross-site skill.**
  - NSE, KGE and RMSE over all scored cells of all scored sites at once (plus log-NSE for log targets).
  - CIs come from a group bootstrap computed exactly from per-group sufficient statistics
    (`evaluate.pooled_metrics`).
- **Probabilistic outputs.** EA-LSTM μ/σ are stored. Their CRPS and coverage are secondary.

## 6. Models and comparators
- **SENTINEL models.**
  - `ealstm` (`configs/hydromet/lstm.yaml`).
  - The physics-guided DO variant `ealstm_physdo` (`configs/hydromet/lstm_physics_do.yaml`), which uses
    the hyperparameters selected for `ealstm` (`--hparams-from`).
  - From v0.7–v0.10 also `ealstm_gbmblend`, `ealstm_full`, `ealstm_compact` and `ealstm_ws` (§9).
  - Each is trained with seeds 0, 1 and 2.
  - The evaluated model is the **3-seed ensemble**: the mean of the seed point estimates in model space
    (`pipeline.seed_ensembles`).
  - Checkpoints, freeze entries and prediction files are keyed `<model>_s<seed>` (review C2).
- **Comparator families** (`evaluate.COMPARATOR_FAMILIES`; reviews W1 and W2):

  | Endpoints | GBM family | Physics/process family |
  |---|---|---|
  | temperature (mean, 7-DADMax) | `forcing_gbm`, `doy_location_gbm`, `forcing_gbm_bag` (v0.6), `forcing_gbm_compact` (v0.9) | `air2stream_regional`, `air2water_linear`, `air2water_mohseni` |
  | DO (mean, min) | `forcing_gbm`, `doy_location_gbm`, `forcing_gbm_bag` (v0.6), `forcing_gbm_compact` (v0.9) | `dosat_ratio[air2stream_regional]`, `dosat_ratio[doy_location_gbm]` |
  | both (v0.10 additions) | `forcing_gbm_ws_bag` (3-seed bag, upstream channels appended) | temperature: `air2stream_regional_ws`, `air2water_linear_ws`, `air2water_mohseni_ws`; DO: `dosat_ratio[air2stream_regional]_ws` (all driven by basin-mean forcings) |

- **"Best of" within each family.** For each primary endpoint, the comparator is the family member with
  the highest median per-site KGE on **val-role sites × val block** (`evaluate.select_comparators`).
  - The choice is written to `FREEZE_hydromet.json` `comparators` before any test scoring.
  - On protected data, comparators are read only from the freeze file. They are never re-selected on
    test data.
- **Same cells and data.** All models are scored on identical cells, and all are fitted on the same
  train-role × train-block data.

## 7. Statistical analysis
- **Primary test (per model m ∈ {ealstm, ealstm_physdo, ealstm_gbmblend, ealstm_full, ealstm_compact, ealstm_ws, ealstm_ws_full} and endpoint v; 28 hypotheses from v0.11; 24 in v0.10; 20 in v0.9, 16 in v0.8, 12 in v0.7, 8 before).**
  - For each family f ∈ {gbm, physics}: Δ_f = median_site KGE(m) − median_site KGE(comparator_f(v)), on
    the sites scored by both.
  - The paired bootstrap resamples **basin groups** (`sentinel.eval.bootstrap.paired_bootstrap` with
    groups; B = 2,000; seed 0). It gives a 95% CI and a two-sided p_f.
  - Intersection–union: p_IUT = max(p_gbm, p_physics).
  - Holm–Bonferroni at family-wise α = 0.05 across the 28 p_IUT values (24 in v0.10, 20 in v0.9, 16 in v0.8, 12 in v0.7, 8 before).
- **Decision rule (implemented in `evaluate.primary_comparison`, column `won`).** An endpoint is **won**
  only if all of the following hold:
  1. Holm rejects;
  2. Δ_gbm > 0 and Δ_physics > 0, each with a CI excluding 0;
  3. **every-seed rule (review W3):** the point Δ_f is > 0 for each of the 3 individual seeds against both
     comparators.
- **Secondary analyses** (descriptive, no multiplicity control):
  - site-resampling sensitivity of all of the above;
  - secondary metrics, pooled skill and secondary targets;
  - train-role sites × val/test block (temporal-only extrapolation);
  - the `lagged` forcing mode;
  - skill against distance to the nearest train site;
  - physics-DO head vs plain EA-LSTM on DO.
- **Pre-registered static-input ablation (review W4)** (`configs/hydromet/{baselines,lstm}_nogauge.yaml`,
  `static_ablation: no_gauge_derived`).
  - Changes: drop StreamCat BFI and runoff, and replace the NWIS elevation with StreamCat `elevcat`.
  - Models re-fitted: `forcing_gbm_nogauge`, `air2stream_regional_nogauge`,
    `dosat_ratio[air2stream_regional]_nogauge` and `ealstm_nogauge_s0` (seed 0, inherited
    hyperparameters).
  - Report Δ median per-site KGE vs the matching full-input model (group-bootstrap CI).
- **Context only.** Published unseen-basin numbers are context and are never compared head-to-head.

## 8. Tuning parity, freezing and the single test look
- **Tuning parity (review W5).** The EA-LSTM and `forcing_gbm` each get **3** configurations, selected on
  the inner train-role groups only. Val-role sites are never used, and non-selected configurations are
  never applied to val.
  - EA-LSTM grid (seed 0, as run; v0.4): static-input noise σ ∈ {0, 0.5, 1.0} at hidden 128 and dropout
    0.4, with lr 5e-4, weight decay 1e-4 and 8 epochs of 200 steps.
    - Criterion: lowest inner point MSE (standardised, target-balanced).
    - Record: `$SENTINEL_RUNS/hydromet/DIAG_ealstm_inner/selection.json`.
    - Seed runs of the selected point early-stop on inner point MSE (patience 6, ≤ 60 epochs of 200
      steps). From v0.5 there is no time cap (v0.4 runs were capped at 18 min per seed; that capped
      ensemble, `ealstm_v3`, is kept as a disclosed earlier iteration and is not a freeze candidate).
  - GBM grid: `max_rows_per_site` ∈ {1000, 3000, all train days}, chosen per target by inner RMSE on the
    same uncapped inner rows, then refitted on fit + inner sites. A target with < 50 inner rows uses
    500 trees (no early stopping).
  - All configurations tried are logged (`$SENTINEL_RUNS/hydromet/configs_tried.jsonl`) and both counts
    are reported.
  - Further iteration follows RESEARCH_PLAN §15.5, with the same configuration count for both families.
- **Freeze candidate (written 2026-11-08 after the registered val look `M6/val_v2`; NOT frozen — no FREEZE file).**
  - Rule (fixed before reading `M6/val_v2`, handoff of 2026-10-25): among the standalone EA-LSTM variants (excluding
    `ealstm_gbmblend`, which contains the GBM), the one with the highest mean over the 4 primary endpoints of
    min(Δ_gbm, Δ_physics); the blend is reported separately.
  - Cycles 1–4 (`M6/val_v2`): `ealstm_full` −0.011 (best), `ealstm` −0.014, `ealstm_compact` −0.016,
    `ealstm_physdo` −0.017; `ealstm_gbmblend` −0.0003 (reported separately). **Candidate: `ealstm_full`
    (seeds 0–2, `configs/hydromet/lstm_full.yaml`).** No endpoint is won; the label would be "competitive".
  - Cycle 5 (`M6/val_v3`, 2026-11-09): `ealstm_ws` −0.0037 > `ealstm_full` −0.011 → **candidate: `ealstm_ws`
    (seeds 0–2, `configs/hydromet/lstm_ws.yaml`)**; `ealstm_gbmblend` −0.0003 reported separately.
  - Cycle 6 (`M6/val_v4`, 2026-11-09): `ealstm_ws_full` −0.0015 > `ealstm_ws` −0.0037 → **current candidate:
    `ealstm_ws_full` (seeds 0–2; `configs/hydromet/lstm_ws_full.yaml` refitted from `ealstm_ws_v1_s<k>`)**; blend
    −0.0003 reported separately. Comparators to freeze (`M6/val_v4` `comparators.json`): temp mean `forcing_gbm_bag` /
    `air2stream_regional_ws`; 7-DADMax `forcing_gbm` / `air2stream_regional_ws`; DO mean `forcing_gbm` /
    `dosat_ratio[air2stream_regional]_ws`; DO min `forcing_gbm_bag` / `dosat_ratio[air2stream_regional]`. Expected label
    on test: "competitive" (no val endpoint won). The lead authorises the freeze and the single test look.
  - **Candidate record** `docs/preregistration/hydromet_freeze_candidate.json` (`scripts/eval/hydromet_freeze_candidate.py
    write|verify`; NOT a freeze): split sha256, `scoring_code_sha256`, weights sha256 of the candidate, blend members and
    selected comparators (14 artifacts), and **data snapshot pins** (manifest sha256 + content sha256 over every file the
    loaders read): `sonde_daily` → the promoted `sonde_daily_refresh_20261108` (manifest `dcd359ea…`, content
    `65d86584…`, 5,501 files), `forcings_daily`, `hydromet_upstream_forcing`, `streamcat`, `sites`. The 2026-11-08
    promotion left every value ≤ 2023-09-30 identical (`compare_vs_sonde_daily.json`), so artifacts trained before or
    after it saw the same data. `verify` must PASS immediately before the test look.
  - **No test data in selection** (audit in the record): the val scores `M6/val_v2…v4` and every candidate/comparator
    run loaded data ending 2023-09-30 with blocks {train, val} only. Track-N eligibility counts of test-block sample days
    (counts, no values) were used only for the §12 power check.
- **Label, confirmatory family, power and comparators (2026-11-09, pre-test review C2/W2/W3).**
  - The M6 label (§15.1, H-G1) is carried by the §8 candidate **`ealstm_ws_full`** alone. The confirmatory family of
    `M6/test` is the 28 hypotheses of §7 (7 models × 4 primary endpoints; all 7 models' artifacts are frozen, so none
    enters at p = 1 unless its predictions fail). `ealstm_gbmblend` is reported separately (it contains the GBM). A
    Holm-significant, won endpoint of a non-candidate model is reported as a secondary finding and does not change
    the label. Track N (§12, Holm 4) and track R (§10) are separate secondary families.
  - **Power of the test cell** (record spans; group SE scaled from the candidate's val CI): ≈ 34 / 33 / 22 / 22 basin
    groups (temp mean / 7-DADMax / DO mean / DO min); minimal detectable Δ median KGE at 80 % power ≈ 0.014 / 0.022 /
    0.07 / 0.06 at α = 0.05 and ≈ 0.020 / 0.032 / 0.10 / 0.08 under worst-case Holm-28. The candidate's val Δgbm is
    +0.002 / −0.002 / −0.016 / +0.011, so a GBM-beating result is essentially unreachable: the expected
    "competitive" label is largely a power outcome, and the test look is expected to be underpowered for the gaps
    seen on val.
  - **Published methods (§15.1(i)).** The strongest published deep models for these targets at unmonitored sites are
    LSTMs driven by meteorological forcings and static catchment attributes: Rahmani et al. 2021 (stream
    temperature, unmonitored and data-scarce basins; their best variants add observed streamflow, which M6 excludes
    as in-situ input) and Zhi et al. 2021 (continental DO, CAMELS-Chem); process-guided RNNs (Jia et al. 2021) need
    site-specific calibration data. `ealstm` (forcings + static attributes, multi-target) is the M6 re-implementation
    of that class on our split, not an exact re-run of either code base. Therefore even a won endpoint would be
    labelled "beats GBM and physics baselines (H-G1)", **not** "state of the art".
  - **Single test look (pre-test review C1).** Protected data are reachable only through
    `scripts/eval/hydromet_test_look.py` (`sentinel/models/hydromet/testlook.py`): freeze verification (split, code
    hash, data pins) before an atomic O_EXCL claim marker per look (`M6/test`, `M6N/test`, `M6R/test`, `M6/test_b`),
    refusal if the registry already holds the look's experiment, a claim-time registry row, predictions sealed
    0444 with `SHA256SUMS`, and a SCORED marker; `guard.check_access` refuses protected data without an open claim of
    the current freeze. `hydromet_score.py`, `hydromet_predict.py` and `hydromet_track_n.py` refuse protected data.
    Models, Holm family size and comparators come only from `freeze["looks"][<look>]`. A val rehearsal of the
    whole path (`--rehearse-on-val`) must reproduce `M6/val_v4` before the freeze.
  - **test_b** (WY2026) is 47 % provisional in the 2026-11-08 snapshot: it gets its own later freeze
    (`FREEZE_hydromet_test_b.json`) with a later `sonde_daily` pin, same models and family.
- **Freeze contents.** Before test scoring, `FREEZE_hydromet.json` records:
  - `created` and `split_sha256` (above);
  - `scoring_code_sha256` (`guard.scoring_code_sha256()`). It covers **all** `sentinel/models/hydromet/*.py`,
    `configs/hydromet/*.yaml`, `scripts/{train,eval}/hydromet_*.py`, this protocol and the shared code
    they import (review C3b);
  - `weights_sha256` for every compared artifact, keyed `<model>` / `<model>_s<seed>`;
  - `comparators` (§6).
- **The test look.** Test predictions come only from `scripts/eval/hydromet_predict.py`.
  - It requires `--allow-test` plus the freeze file, and verifies every artifact against `weights_sha256`.
  - The scorer accepts a protected prediction file only if its `artifact_sha256` equals the frozen entry
    and its `freeze_sha256` equals the current freeze (review C3c).
  - Training scripts refuse test blocks and test-role sites unconditionally.
  - Test scoring (`hydromet_score.py --primary --allow-test`) is run **once**. Test-A (WY2024–2025) is
    primary; test_b is a later secondary look under the same freeze.
- **Registry.** Results reach the registry only through `scripts/eval/hydromet_score.py --register`, in one
  consolidated experiment per block. Smoke runs and non-FINAL splits are refused.

## 9. Deviations log
| Date | Change | Reason |
|---|---|---|
| 2026-10-22 | v0.1: target classes, pooled skill, primary metric and comparison | Coordinator decisions (1)–(3) |
| 2026-10-22 | v0.2 (before any full run): basin-group resampling; two-family IUT comparator rule with air→water regressions in the physics family; every-seed rule in code; `<model>_s<seed>` artifact keys; non-FINAL splits refused, smoke = train-role only; freeze hash covers all model code, configs and protocol; prediction provenance checked; static-input ablation; matched 3-configuration tuning for GBM and EA-LSTM; GBM may use all train days; SC secondary; H-G1 needs both T and DO; Holm over 8 | Pre-training review C1–C3, W1–W6; coordinator confirmations |
| 2026-10-22 | v0.3 (first full runs aborted within minutes; no val prediction made): inner split changed from a prefix rule (which gave 45 inner sites and 0 nitrate inner sites) to a greedy whole-group fill; EA-LSTM grid changed to capacity × dropout with lr 5e-4, weight decay 1e-4 and 200-step epochs; GBM fallback of 500 trees when a target has no inner rows | Inner-split coverage, and inner-loss rise after epoch 0 (train-role inner data only) |
| 2026-10-24 | v0.4 (before any EA-LSTM val prediction): EA-LSTM early stopping on inner point MSE instead of NLL; static-input noise added; the EA-LSTM grid became the inner-only static-noise sweep {0, 0.5, 1.0} (3 configurations, matching the GBM's 3; the v0.3 hidden/dropout grid was aborted after 3 epochs of its first point and is counted as tried); 18-min cap per seed; physics-DO and no-gauge EA-LSTM runs deferred | Inner MSE and NLL rose from epoch 0 (site memorisation; train-role data only); coordinator time budget |
| 2026-10-24 | v0.5 (RESEARCH_PLAN §15.5 improvement cycle 1, after the capped v0.4 val look `M6/val_v1`): the 18-min cap is removed; seed runs of the same selected point (hidden 128, dropout 0.4, static noise 0.5; the inner-only selection is unchanged) early-stop on inner train-role point MSE (patience 6, ≤ 60 epochs). The deferred pre-registered runs (`ealstm_physdo` seeds 0–2, `ealstm_nogauge_s0`) are run with the same rule. No hyperparameter, input, comparator or decision-rule change; the EA-LSTM and `forcing_gbm` keep 3 grid configurations each (the GBM was never time-capped; its trees already early-stop on the same inner rows). Re-scored on val as `M6/val_v2`; the v0.4 numbers stay in the registry under `M6/val_v1`. MTS-LSTM, CMAL heads and graph upstream context (RESEARCH_PLAN §5.6 options) are not in this protocol and are not run | Seeds stopped by the cap at epochs 7–8 while inner MSE was still falling (seed 0 best at its last epoch) — under-training, diagnosed on train-role inner groups |
| 2026-10-24 | v0.6 (before any v0.5 val prediction was scored): GBM-side ensemble parity. `forcing_gbm_bag` = model-space mean of `forcing_gbm_bag_s{0,1,2}` (identical grid, inner selection and LightGBM settings; seeds change row sampling and bagging; `configs/hydromet/baselines_gbmbag.yaml`) joins the GBM comparator family of every primary endpoint (`evaluate.COMPARATOR_FAMILIES`). The comparator can only become stronger. Secondary paired analyses of §7 (static-input ablation; physics-DO head vs EA-LSTM) are now computed and registered by the scorer (`evaluate.secondary_comparisons`) | RESEARCH_PLAN §15.5 amendment (iterate until a §15.1 label; tuning parity, baselines never weakened): the SENTINEL model is a 3-seed ensemble, so the GBM gets the same ensembling |
| 2026-10-24 | v0.7 (cycle 2; written before any v0.5/v0.6 val score): third SENTINEL model `ealstm_gbmblend`. Per seed k, `ealstm_gbmblend_s<k>` = equal-weight model-space mean of `ealstm_s<k>` and `forcing_gbm_bag_s<k>` (`pipeline.seed_blends`); the evaluated model is its 3-seed ensemble (= mean of the `ealstm` and `forcing_gbm_bag` ensembles). No weight is fitted on any data. It is compared under the same two-family IUT and every-seed rule; the Holm family grows from 8 to 12 hypotheses (3 models × 4 endpoints). Disclosure: the blend contains the GBM comparator family's own model, so a win means "the EA-LSTM adds skill on top of the GBM", not "the EA-LSTM alone beats the GBM" | RESEARCH_PLAN §15.5 amendment, lever "ensembling"; GBM-side parity is `forcing_gbm_bag` (v0.6) |
| 2026-10-24 | v0.8 (cycle 3; written before any v0.5–v0.7 val score): fourth SENTINEL model `ealstm_full`. For each seed k, the early-stopped `ealstm_s<k>` run (v0.5) fixes the epoch count E_k = argmin inner MSE + 1; a fresh model with the same seed and hyperparameters is then trained on fit + inner train-role sites (all 1,712) for E_k epochs, with no validation and the last epoch kept (`hydromet_lstm.py --refit-full-from`, `configs/hydromet/lstm_full.yaml`). This mirrors `forcing_gbm`, which is refitted on fit + inner rows after inner selection (review O3). Holm family: 16 hypotheses (4 models × 4 endpoints) | Parity: the EA-LSTM was the only model not refitted on the inner groups (13 basin groups, 15% of train-role sites never seen by its weights) |
| 2026-10-24 | v0.9 (cycle 4; written before any v0.5–v0.8 val score was registered; only the unregistered interim look at the v0.5 `ealstm` ensemble had been made): fifth SENTINEL model `ealstm_compact` (seeds 0–2, same selected hyperparameters, early stopping as v0.5) on an a-priori compact static set of 45 attributes (`data.COMPACT_STATIC`: location/topography, 1991–2020 climate normals, groundwater/hydrology incl. BFI/runoff, watershed and 100-m riparian land cover, dams, soils, channel geometry, N sources) instead of all 188. GBM parity: `forcing_gbm_compact` (same set, same 3-configuration grid; `configs/hydromet/baselines_compact.yaml`) joins the GBM comparator family. Holm family: 20 hypotheses (5 models × 4 endpoints) | Diagnosis on train-role data: EA-LSTM train loss falls to ≈ 0.06 while inner MSE plateaus at ≈ 0.21, and static-input noise lowered inner MSE (0.246 → 0.210) — site memorisation through 188 static inputs with 1,456 fit sites |
| 2026-11-08 | v0.10 (cycle 5; written after `M6/val_v2` was registered and before any upstream-forcing model was trained or scored): sixth SENTINEL model `ealstm_ws` (seeds 0–2, hyperparameters of the v0.4 inner selection, early stopping as v0.5) with upstream-basin-averaged gridMET forcings appended to the site-cell forcings (`configs/hydromet/lstm_ws.yaml`). Parity: `forcing_gbm_ws_bag_s{0,1,2}` (the full `forcing_gbm` feature set also computed on the basin-mean channels, `f_ws_*`; same 3-configuration grid; `baselines_ws_gbmbag.yaml`) joins the GBM family, and the physics family gains `air2stream_regional_ws`, `air2water_{linear,mohseni}_ws` and `dosat_ratio[air2stream_regional]_ws` driven by basin-mean forcings (`baselines_ws_physics.yaml`). Comparators can only get stronger. Holm family: 24 hypotheses (6 models × 4 endpoints). Secondary: Δ of each `_ws` model vs its site-cell counterpart. Re-scored as `M6/val_v3` (all v0.5–v0.10 prediction sets) | Literature lever (Siddik et al. 2025, J. Hydrology: upstream-area-averaged meteorology improves unseen-basin stream temperature); site-cell weather misrepresents snowmelt/upstream climate of large basins (median StreamCat watershed 630 km², max 2.9 M km²) |
| 2026-11-08 | Harder tracks §10–§12 added (§12 track N approved by the lead 13:00) (RESEARCH_PLAN §15.5 amendment of 2026-11-08, items 5–7), written before any model was fitted or scored on them; the original benchmark (§1–§8) is unchanged and always reported | M6 is saturated: all strong models reach median temperature KGE ≈ 0.90 and val has 7–12 basin groups per endpoint |
| 2026-11-08 | §10 amendment (GPU budget): track-R `ealstm_full` refits are optional and scored only if all 18 finish; otherwise Holm over 8 | User instruction: SENTINEL GPUs reduced to 0/1, shared, ≤ 2 h per job |
| 2026-11-09 | v0.11 (cycle 6; written after `M6/val_v3` was registered, before any `ealstm_ws_full` run): seventh SENTINEL model `ealstm_ws_full` = the v0.8 refit rule (fit + inner sites, E_k epochs from the early-stopped run, last epoch kept) applied to `ealstm_ws_s<k>` (`configs/hydromet/lstm_ws_full.yaml`). No new comparator is needed (all GBMs are already refitted on fit + inner; `forcing_gbm_ws_bag` and the `_ws` physics members exist). Holm family: 28 hypotheses (7 models × 4 endpoints). Re-scored as `M6/val_v4` | Combination of the two levers that each helped the EA-LSTM in `M6/val_v3` (refit: cycle 3; upstream forcings: cycle 5) |
| 2026-11-09 | Pre-declarations requested by the lead (before any further track selection): §8 candidate record with data snapshot pins and no-test audit; §10 track-R comparators fixed and own test look; §12 track-N candidates (`ealstm_ws_full`, `ealstm_gbmblend`), fixed comparators and own test look | Lead instruction 2026-11-09; promoted sonde snapshot (2026-11-08) |

## 10. Harder track R — leave-region-out (added 2026-11-08, before any fit or score on it)
- **Why.** The original benchmark is saturated (§9, 2026-11-08 row). Track R is stricter extrapolation (unseen
  macro-regions instead of unseen HUC4 groups) and has ~5× more scored basin groups. It is reported **in
  addition to** §1–§8; claims name the track.
- **Folds (frozen).** `docs/preregistration/hydromet_track_r_folds.json`, sha256
  **`142341a9d0379f2b50c90bdc72508cce8aa24d78d79db855aba9f1065f92a6c8`** (built from the FINAL split
  `fd4bb14f…`; `sentinel/models/hydromet/tracks.py`, `scripts/eval/hydromet_make_tracks.py`).
  - Six macro-regions, each a union of whole HUC2 regions fixed by geography: R1 {01, 02}, R2 {03, 06, 08},
    R3 {04, 05, 07, 09}, R4 {10, 11, 12}, R5 {13–16}, R6 {17, 18}.
  - A basin group goes whole to the region holding most of its sites (3 of 108 groups span two HUC2s).
- **Cells.** For fold k: fit on FINAL train- and val-role sites outside region k × train block (inner early-stopping
  groups chosen among them by the §2 rule, `inner_seed = 0`); score train/val-role sites of region k × val block
  (WY2022–2023, 42-day embargo). The CV cell is the union over the six folds. FINAL test-role sites are never fitted
  on; in the single test look each is scored on the test block by the model of its own region's fold.
- **Models.** Per fold, the EA-LSTM recipe of the original-track freeze candidate (§8; `ealstm_full` as of
  2026-11-08: the early-stopped `ealstm_s<k>` run fixes E_k, then the refit on all fold-train sites), 3 seeds, with the
  hyperparameters of the original inner selection (no new tuning). SENTINEL models scored: `ealstm`, `ealstm_full`
  and `ealstm_gbmblend` (= `ealstm` + `forcing_gbm_bag`). If the §8 candidate changes to an upstream-forcing (`_ws`)
  recipe before any track-R fit, the `_ws` recipe and the `_ws` comparator members are used instead.
- **Amendment 2026-11-08 16:10 (before any track-R score; user cut the SENTINEL GPU budget to two shared GPUs).**
  The early-stopped `ealstm` seeds of all six folds run first and the `ealstm_full` refits after them. If the refits
  are not all finished when track R is scored, `ealstm_full` is dropped from the track-R family (Holm over 8:
  `ealstm`, `ealstm_gbmblend`) and the omission is reported; no partial-fold `ealstm_full` is ever scored.
- **Pre-declared 2026-11-09 (before any track-R score):** comparators are FIXED to the original-track selection among
  the members track R refits (`M6/val_v2` `comparators.json`, copied to `docs/preregistration/hydromet_track_r_comparators.json`): temp mean `forcing_gbm_bag` / `air2stream_regional`;
  7-DADMax `forcing_gbm` / `air2stream_regional`; DO mean `forcing_gbm` / `dosat_ratio[air2stream_regional]`; DO min
  `forcing_gbm_bag` / `dosat_ratio[air2stream_regional]` (each refitted per fold). No comparator is selected on the
  track-R CV cell (this replaces the "best of family on the CV cell" sentence below). Track-R candidates:
  `ealstm_gbmblend` and `ealstm` (+ `ealstm_full` under the GPU amendment). **Own single test look:** after all folds
  finish, the fold artifacts' sha256 are added to a track-R freeze entry (lead); each FINAL test-role site is then
  predicted once on the test block by its region's fold models and scored once (`M6R/test`). Track R never replaces
  the primary (§1–§8) claim.
- **Comparators** (all refitted per fold with the original configs): GBM family `forcing_gbm`, `doy_location_gbm`,
  `forcing_gbm_bag` (3 seeds); physics family `air2stream_regional`, `air2water_linear`, `air2water_mohseni`
  (temperature) and `dosat_ratio[air2stream_regional]`, `dosat_ratio[doy_location_gbm]` (DO). (`forcing_gbm_compact`
  is omitted: it never was the selected GBM in `M6/val_v2`, and the cost is one more suite per fold.) Comparators
  are the fixed ones of the pre-declaration above; nothing is selected on the track-R CV cell.
- **Statistics.** As in §5–§7: median per-site KGE, basin-group paired bootstrap (B = 2,000), IUT, every-seed rule,
  Holm over 12 hypotheses (3 models × 4 primary endpoints) if all `ealstm_full` refits finish, else over 8 (GPU
  amendment). Fold-level results are secondary.
- **Clarifications 2026-11-09 (pre-test review W4).** (i) Fixing the comparators to the original val selection
  instead of best-of-family on the CV cell is a deliberate departure from §6: the CV cell contains the val-role ×
  val cell already used for six cycles of selection, so selecting there again would add a selection step on
  non-confirmatory data; the fixed members were the strongest refitted members on val. It can make the comparator
  weaker than a CV-selected one (§15.5(4)); this is disclosed with every track-R result. (ii) The R1 fold logs
  printed descriptive val-cell KGEs of `ealstm_s0/s1` and `ealstm_full_s0/s1` (no comparator Δ) before this
  pre-declaration. (iii) The CV cell is not confirmatory. The only track-R claim cell is `M6R/test`, and it scores
  the SAME FINAL test-role sites × test block as the primary look, after it: track R is **secondary/descriptive**,
  its own Holm family, never combined with or replacing the primary. Its power check covers the CV cell only; it
  adds no test-cell power. (iv) Test path (implemented and hashed with the primary freeze): separate freeze file
  `FREEZE_hydromet_trackR.json` with the same `scoring_code_sha256`, fold-unique artifact keys
  `trackr_<fold>/<model>_s<seed>`, its own `looks["M6R/test"]` block (models, Holm size, the fixed comparators,
  artifacts, data pins) and the single-pass guard; `scripts/eval/hydromet_test_look.py --look M6R/test`.
- **Power check** (`docs/preregistration/hydromet_track_r_power.json`; from group counts and the original-track SE):
  upper-bound CV groups 67 / 63 / 44 / 41 (temp mean / 7-DADMax / DO mean / DO min) vs 12 / 10 / 9 / 7 in the
  original val cell; minimal detectable Δ median KGE (80% power, worst-case Holm α/12) ≈ 0.025 / 0.030 / 0.062 / 0.060
  (original cell: 0.058 / 0.076 / 0.138 / 0.144). Observed original-track Δs are ~0.01, so track R may still be
  underpowered for temperature unless regional extrapolation widens the gaps; this is stated with any result.
- **Registry.** Experiments `M6R/cv_v1` (CV cell) and later `M6R/test` (single look after freeze).

## 11. Harder track S — a-priori difficulty strata (added 2026-11-08, before any stratified score)
- Applied to the original val cell (`M6/val_*`) and to track R. Thresholds come from FINAL train-role sites × train
  block only (`tracks.train_thresholds`, `tracks.heatwave_mask`).
- **Strata.**
  - `warm_p90` (temperature endpoints): observed value > pooled train P90 of that target.
  - `heatwave` (temperature endpoints): runs of ≥ 3 consecutive days with site gridMET Tmax above the site's own
    train-block P95, plus the 3 following days (forcing-defined; no target used).
  - `low_do_summer` (DO endpoints): July–September days with observed value < pooled train P10.
- **Metrics.** Per-site RMSE and bias on stratum days (sites with ≥ 10 stratum days), median over sites with
  basin-group bootstrap CIs; exceedance CSI on all scored days for the P90 temperature thresholds and for
  `do_mgl_min` < 5 mg/L. Paired Δ RMSE vs the selected comparator of each family (as in §6) with group bootstrap.
- **Status.** Descriptive (secondary; no Holm claim) on the original track. A stratum becomes a claim only on a
  track where it is listed here before scoring; currently none is a primary claim.

## 12. Harder track N — national unsensed reaches, WQP discrete samples (added 2026-11-08, approved by the lead; written before any prediction at a track-N station)
- **Why.** M6's purpose is water quality at reaches **without** a sensor; the original benchmark scores sensor
  sites only (12 / 9 val groups). Track N scores the same models at ~10⁴ WQP stations without a continuous sensor.
  Reported as its own track alongside §1–§8; claims name the track.
- **Data** (`processed/hydromet_wqp_national`, datasheet `docs/datasheets/hydromet_wqp_national.md`,
  `scripts/data/build_hydromet_wqp_national.py`). From `processed/wqp_discrete` (discrete rows only; no logger or
  continuous series exist there): WQP `site_class = stream`, CONUS; `temp` (°C) and `do` (mg/L); `status_class =
  accepted`; uncensored; activity types other than time-series composites; depth NaN or ≤ 2 m. Daily value = mean of
  the station-date's samples, compared with the model's daily-mean prediction (`temp_c_mean`, `do_mgl_mean`; the
  7-DADMax and DO minimum cannot be observed from grab samples). Only WY2021-10-01 onward is stored (val and test
  blocks; nothing is ever a training row).
- **Unsensed.** Excluded: WQP `USGS-<no>` stations with any USGS daily-value water-quality series, and every station
  whose NHDPlus V2 COMID (point in the simplified catchment) carries such a site or a FINAL-split site.
- **Inputs** exactly as for sonde sites: site-cell gridMET (nearest valid cell ≤ 3 cells), StreamCat of the COMID,
  lat/lon; `elev_m` = StreamCat `elevcat` (no gauge datum exists; disclosed distribution shift).
- **Roles (frozen)** `docs/preregistration/hydromet_track_n_stations.json` (`sentinel/models/hydromet/track_n.py`):
  candidate roles = roles of FINAL basin groups owning the station's HUC4 plus roles of flow-connected FINAL sites
  (NHDPlus V2 downstream main path ≤ 500 km, either direction). {test} → test; test with any other role → excluded;
  {val} → val; {train} → train (seen region; secondary in-region analysis only); {train, val} → excluded. No
  candidate (region without any sonde site) → new-region group = HUC4s merged over station flow connections; the
  whole group is val or test by sha1 of its smallest HUC4 (even → val). Resampling unit = basin group (FINAL group, or
  `N:<HUC4…>`).
- **Models.** No model is fitted on track N. All artifacts are the original-track ones (FINAL train-role sites ×
  train block), applied unchanged (`scripts/eval/hydromet_track_n.py predict`): the SENTINEL models of §7 that
  exist when the track is scored and the comparator families of §6. Comparators are fixed from the original-track val
  selection (`comparators.json` of the registered `M6/val_*` score), never re-selected on track N. `_ws` models need
  station upstream forcings and are scored on track N only if those are built with the same method (disclosed).
- **Endpoints and statistics.** Primary metric: median over stations (≥ 8 sample days in the block) of per-station
  RMSE (lower is better). Δ_f = median RMSE(model) − median RMSE(best comparator of family f); paired basin-group
  bootstrap (B = 2,000, seed 0); IUT over the two families; Holm over (models × 2 endpoints); won = Holm rejects,
  both Δ < 0 with CI below 0, and every seed better than both comparators. Secondary: median per-station KGE and bias,
  train-role (in-region) stations × val block, new-region stations alone.
- **Frozen roles:** sha256 **`ffb41471c9a75dcbebedf083f273cfa41aae3ffeb7dabe7fd420c5f23120e2d0`** (12,768 stations
  with StreamCat + gridMET: 6,244 train-region, 1,365 val, 5,046 test, 113 excluded — 93 touch test-role sites, 20 touch
  both train and val; 1,444 stations in regions without any sonde site).
- **Power check** (`docs/preregistration/hydromet_track_n_power.json`): val cell 1,000 / 938 stations in 12 / 12 basin
  groups (temp / DO; median 14 / 12 sample days); test cell 3,014 / 2,724 stations in 38 / 36 groups; train-region
  (secondary) 4,605 / 4,199 stations in 52 / 50 groups. The val cell has as many groups as the original val cell, so
  track N adds stations (≈ 6×) but not groups for selection; the test cell has ≈ 3× the val cell's groups. Stated
  with every track-N result.
- **Cells.** Selection/descriptive: val-role stations × val block (`M6N/val_*`). Test: test-role stations × test
  block, once, after the freeze, via the same guard as `hydromet_predict.py` (`--allow-test` + FREEZE file).
- **Pre-declared 2026-11-09 (after `M6N/val_v1` and `M6N/inregion_val_v1`, before any further track-N selection).**
  - Track-N candidates (both tested; Holm over 2 models × 2 endpoints = 4; untestable hypotheses enter at p = 1):
    the §8 candidate **`ealstm_ws_full`** (seed ensemble; station upstream forcings from
    `processed/hydromet_upstream_forcing/*_trackn`, same builder, but the "NLDI polygon if the area ratio vs StreamCat
    is outside [0.5, 2]" rule uses cached NLDI polygons only — no new requests on the shared quota — so flagged
    stations without a cached polygon keep the local basin; count disclosed in the manifest) and **`ealstm_gbmblend`** (= `ealstm` + `forcing_gbm_bag`, chosen for track N because the
    blend was the only model better than the GBM in the powered in-region look; disclosed as selected after
    `M6N/inregion_val_v1`).
  - Comparators FIXED from the original-track selection `M6/val_v4`: temp mean `forcing_gbm_bag` /
    `air2stream_regional_ws`; DO mean `forcing_gbm` / `dosat_ratio[air2stream_regional]_ws` (the `_ws` physics
    members run with station basin-mean forcings, `upstream_forcing: replace`).
  - Split: track-N roles sha256 `ffb41471…`; FINAL split `fd4bb14f…`. Power: §12 power check above.
  - **Own single test look** (`M6N/test`, `scripts/eval/hydromet_test_look.py --look M6N/test`): models, family size,
    comparators, artifacts and data pins (`hydromet_wqp_national`, `hydromet_upstream_forcing_trackn`, `streamcat`)
    come from `FREEZE_hydromet.json` `looks["M6N/test"]`; test-role stations × test block, once. Track N never replaces the primary
    (§1–§8) claim; the val looks `M6N/val_*` stay descriptive.
