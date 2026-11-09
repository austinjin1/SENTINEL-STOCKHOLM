---
model_id: hydromet_ealstm
name: SENTINEL-HydroMet
modality: hydromet
registry_models: [hydromet_ealstm, hydromet_ealstm_physdo, hydromet_ealstm_gbmblend, hydromet_ealstm_full,
                  hydromet_ealstm_compact, hydromet_ealstm_nogauge_s0, hydromet_forcing_gbm, hydromet_forcing_gbm_bag,
                  hydromet_forcing_gbm_compact, hydromet_doy_location_gbm, hydromet_air2water_linear,
                  hydromet_air2water_mohseni, hydromet_air2stream_regional,
                  "hydromet_dosat_ratio[doy_location_gbm]", "hydromet_dosat_ratio[air2stream_regional]"]
registry_experiments: ["M6/val_v1", "M6/val_v2"]
workstream: M6
version: 0.2.0
status: validation (no test result; freeze not yet written)
license: MIT
intended_use: >
  Daily water-quality estimates (water temperature, 7-day average daily maximum temperature, dissolved
  oxygen mean/min, specific conductance, turbidity, nitrate+nitrite) at stream reaches WITHOUT any
  in-situ sensor, from observed daily weather (gridMET), StreamCat catchment attributes and the
  calendar. Reanalysis/nowcast setting; a context token for fusion. Not a forecast.
limitations: >
  Validation results only (val-role basin groups x WY2022-2023); these are selection-affected because
  several iterations were evaluated on the same val cell (configurations tried are listed below and in
  configs_tried.jsonl). Few val basin groups per endpoint (7-12; one group holds a large share of
  sites), so CIs are wide. CONUS stream sites with gridMET forcings and StreamCat attributes only.
  Observed weather is an oracle for future days, so nothing here may be called a forecast. Specific
  conductance, turbidity and nitrate have near-zero or negative median KGE at unseen sites for every
  model; nitrate has very few val-role sites.
---

# SENTINEL-HydroMet (M6)

**Status: validation.** Full-scale models trained and scored on val-role sites × the val block
(registry experiments `M6/val_v1`, capped v0.4 EA-LSTM, and `M6/val_v2`, protocol v0.5–v0.9). No
test-role site or test block has been predicted; `FREEZE_hydromet.json` has not been written. This card
contains no performance numbers. All metrics come from `results/registry.jsonl`
(RESEARCH_PLAN §3 R1/R2).

## Model details
- **Main model:** an entity-aware LSTM (EA-LSTM; Kratzert et al. 2019). Static attributes gate the
  input. It predicts all targets at once with a Gaussian head (quantile head optional). The loss is
  masked for missing targets and balanced per target. Code: `sentinel/models/hydromet/lstm.py`.
  Selected configuration (inner train-role groups only, `DIAG_ealstm_inner/selection.json`): hidden 128,
  dropout 0.4, static-input noise σ 0.5, lr 5e-4, weight decay 1e-4, 365-day windows with loss on the last
  90 days, early stopping on inner point MSE (patience 6). The evaluated model is the 3-seed ensemble.
- **Variants (protocol §6, §9):**
  - `ealstm_physdo`: physics-guided DO head. DO = DO_sat(T̂_w, P(elev)) × a learned ratio, where DO_sat is
    Benson–Krause and T̂_w is the model's own temperature output (`configs/hydromet/lstm_physics_do.yaml`).
  - `ealstm_full` (v0.8): refitted on all train-role sites (fit + inner) for each seed's inner-selected
    epoch count, mirroring the GBM refit (`configs/hydromet/lstm_full.yaml`).
  - `ealstm_compact` (v0.9): an a-priori 45-attribute static set (`data.COMPACT_STATIC`).
  - `ealstm_gbmblend` (v0.7): an unfitted equal-weight model-space blend of the EA-LSTM and the
    seed-bagged forcing GBM, per seed. It contains the GBM comparator's own model.
  - `ealstm_nogauge_s0`: the pre-registered static-input ablation (no BFI/runoff; DEM elevation).
- **Baseline ladder** (`sentinel/models/hydromet/baselines.py`):
  - DOY + location climatology: LightGBM on lat/lon/elevation and DOY harmonics.
  - Pooled air→water temperature regressions: linear and Mohseni logistic.
  - An air2stream-type hybrid. It has no discharge term, is calibrated per train site, and is
    transferred to unseen sites by k-nearest-donor output averaging.
  - DO_sat(T̂_w) × the monthly median DO/DO_sat ratio from train sites.
  - LightGBM on causal forcing windows + StreamCat + calendar (`forcing_gbm`; 3-configuration grid
    selected on inner groups, refitted on fit + inner sites), its 3-seed bag `forcing_gbm_bag` and its
    compact-static variant `forcing_gbm_compact` (GBM-side parity for the EA-LSTM ensemble and static set).
- Checkpoints go to `$SENTINEL_RUNS/hydromet/<run>/g<k>_s<seed>/<model>_best.pt`, with their sha256 in
  `checkpoints.json` (keys `<model>_s<seed>`).

## Inputs (enforced in code)
Only three column groups exist: forcing, static and calendar
(`sentinel.models.hydromet.data.assert_allowed_inputs`).
- Names that look like sensor data are refused: in-situ variables, discharge/stage,
  monitoring-network metadata (`has_*`, numbers of upstream sensors) and NWIS drainage area.
- `tests/test_hydromet.py` checks that inputs do not change when targets change, and that
  predictions do not change when target values outside train-role sites × train block change.
- Forcing windows are causal: the features for day t use days ≤ t − lag.

## Training and evaluation data
- **Manifests:** `sonde_daily`, `forcings_daily`, `streamcat`.
- **Split:** the frozen FINAL split `docs/preregistration/sonde_split.json`. It groups basins as
  HUC4 + NLDI flow connectivity. Its sha256 is stored in every run's `data_info.json` and
  `split_report.json`.
- **Fitting data:** train-role sites × WY2008–2021 only. All normalisation statistics come from
  the same cell.
- **Early stopping and window selection:** a held-out set of train-role basin groups ("inner"
  sites). Val-role sites are never used for these choices.
- **Evaluation cell:** val-role sites (unseen basin groups) × WY2022–2023, with a 42-day embargo.
  Only approved values are used, and every model is scored on identical cells.
- **Test-role sites and test blocks:** only through `scripts/eval/hydromet_predict.py`. This needs
  `--allow-test` plus `docs/preregistration/FREEZE_hydromet.json` with matching split, scoring-code
  and artifact hashes.

## Evaluation
- **Per-site metrics:** NSE, KGE and RMSE (plus log-scale NSE for conductance, turbidity and
  nitrate) from `sentinel.eval.metrics`.
- **Summary:** the median across sites, with a percentile bootstrap CI resampling basin groups (site
  resampling as sensitivity; `sentinel.eval.bootstrap`); pooled cross-site NSE/KGE/RMSE.
- **Pre-registration:** `docs/preregistration/hydromet_protocol.md` (v0.9; deviations in §9). Primary
  endpoints are median per-site KGE for temperature (mean, 7-DADMax) and DO (mean, min). Each SENTINEL
  seed ensemble must beat the best GBM-family comparator AND the best physics/process comparator
  (intersection–union) with basin-group paired bootstrap, Holm correction over all (model × endpoint)
  hypotheses and an every-seed rule (`evaluate.primary_comparison`, column `won`).
- **Secondary:** physics-DO head vs EA-LSTM on DO and the no-gauge-derived static ablation as paired
  basin-group-bootstrap differences (`evaluate.secondary_comparisons`), pooled skill.
- **Registry:** experiments `M6/val_v1` (v0.4, 18-min-capped EA-LSTM; superseded as a candidate but kept)
  and `M6/val_v2` (v0.5–v0.9: uncapped EA-LSTM, physics-DO, refit, compact, blend, ablation and the
  GBM bag/compact comparators). Configurations tried: `$SENTINEL_RUNS/hydromet/configs_tried.jsonl`.

## Uncertainty and calibration
The Gaussian head writes model-space mu/sigma into the prediction files. CRPS and coverage are
secondary endpoints and have not been evaluated yet.

## Limitations, risks and known failure modes
- See the front matter.
- Per-site NSE for level-dominated variables (specific conductance) can be strongly negative at
  unseen sites even when the seasonal shape is right, because the site mean is not known without a
  sensor.

## Licences
- gridMET: public domain.
- StreamCat and USGS NWIS: US public domain.
- Dependencies: LightGBM (MIT) and PyTorch (BSD).

## Provenance
Each run directory holds `config.json`, `data_info.json`, `split_report.json`, `inputs.json`,
`checkpoints.json` and the TrainingRun entry (`logs/training_runs.jsonl`).
