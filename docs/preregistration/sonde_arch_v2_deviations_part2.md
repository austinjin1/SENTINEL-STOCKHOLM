# M1 SENTINEL-Sonde — architecture round v2: deviations log, part 2 (implementation record)

**Written:** Tue 10 Nov 2026 13:04 EST (18:04 UTC), by M1. AI-assisted. Status: DRAFT for the lead's stamp, which covers
part 1 and part 2 together.

**Supplements** `sonde_arch_v2_deviations.md` (part 1, D1–D5, sha `031ddddc…fcd92`, under independent review; not
edited). Stamped texts: `sonde_arch_v2_amendment.md` (sha `6c7b169d…`) and `sonde_arch_v2_lead_rulings.md`
(sha `7cbf3987…`).

**State at writing.**
- **No validation number exists on any new arm or track. No step-1 number exists.** I checked:
  - the registry has no `sonde_arch_v2/step1` or `sonde_arch_v2/cv` metric row;
  - there are no `sentinel_runs/sonde/v2_cv/*/fold*` OOF forecasts;
  - `sentinel_runs/sonde/v2_step1` has no predictions.
- Only train-role × train-block pipeline smokes (no score) and a CPU timing run (wall time only) have been run.
- The binding conditions of the lead's ruling of 2026-11-10 (~18:00 UTC) are written into each entry.

Code references are file:line at the time of writing.

## Conformance note

**Item 1 — per-product NWP features for the GBM-type references (conforms to §2c).**
- §2c: "GBM-type models get per-product features and availability flags."
- **Code:** `sentinel/models/sonde_v2/gbm_nwp.py:70` (`_nwp_blocks`). For each product p (p0 = bias-corrected GEFS c00;
  p1 = bias-corrected Open-Meteo GFS), the features from `nwp.gbm_nwp_blocks` (`nwp.py`) carry the prefix `p{p}_`:
  - per-lead Tmean/Tmax/Tmin for leads 1..min(h, 7);
  - mean Tmean over leads 1..h;
  - cumulative degree-days above 0 °C;
  - exponentially weighted Tmean (decay 0.7);
  - P sum and P max;
  - SW mean;
  - target-day Tmean;
  - availability fractions for temperature and precipitation over leads 1..h.
- **GBM-NWP and PF-GBM get exactly the same per-product features and flags**: PF-GBM = GBM-NWP plus the physics
  block. Its physics block reads the masked mean of the products' raw fields (`gbm_nwp.py:80`).
- In the step-1 contrast, the single perfect-prog product is p0.

**Item 15 — 2-year reference-window climatology (conformance; added Tue 10 Nov 2026 13:25 EST).**
- **Stamped (`m1n_protocol.md` v0.5 §3, Training data):** "All models are trained with 2-year-window climatologies
  (regime parity with new sites)." Reference window (m1n §1/§2b): "the first 730 calendar days from the site's first
  observation of the variable (≥ 365 observed days required)".
- **Before:** the v2 code used the frozen training-years climatology that `common.prepare_data` builds (all
  train-block years ≤ 2021-09-30, ±15 d, min_years = 3). **No number was produced with it**; only train-only smokes
  (no score) and a timing run (wall time only) ran.
- **Now** (`sentinel/models/sonde_v2/prep.py:106` `reference_window_climatology`, applied in `prepare_v2` at line 48
  through `apply_reference_window_climatology`, line 155, before any baseline, row or tensor is built):
  - window = the first 730 days from the site's first observation of the variable;
  - ≥ 365 observed days required;
  - the same estimator as the frozen `data.climatology`: circular ±15-day smoothing, with fallback to the window's
    overall mean/variance where fewer than min_years = 2 distinct years support a day of year.
- **Scope:** every v2 model and every reference (SONDE-HF, R-NH, GBM-NWP, PF-GBM, Chronos-2-cov, and the naive
  references damped anomaly, climatology and persistence that the v2 scorer builds from `data.clim_mean`). It covers
  the anomaly targets, the inputs and the scoring baselines alike. The frozen original-track code is untouched.
- **Sites with < 365 reference-window days** for a variable get no climatology (NaN). That site-variable then has no
  anomaly target (excluded from every model's training loss), no naive baseline, and no common scored row. This is
  one rule for all models.

## Group A — deviations from the stamped text

**A6 (item 6) — targets of the multi-task models.**
- **Stamped (§2d):** "T max and min (and 7-day average of daily max); DO max, DO diel amplitude and DO % saturation
  …; SC daily max and turbidity daily max; log discharge mean and gage height … The loss weights are fixed per-target
  inverse-scale normalisations, set before training." The plan §4.1 list also includes `gage`.
- **Code** (`sentinel/models/sonde_v2/prep.py`):
  - `V2_VARS` = the 12 frozen TARGETS (incl. ph_mean, chl, fdom, discharge) + T max, T min, DO max, SC max,
    turbidity max;
  - `add_derived_targets` (prep.py:163) appends three trained heads, computed from the daily tier:
    - `do_amp_mgl` = DO max − DO min (same day);
    - `do_pctsat` = 100 · DO mean / DO_sat(T mean, P(elev)) (Benson–Krause, M6 routine);
    - `temp_c_7dadm` = mean of T max over days t−6..t, all 7 present.
  - That is 20 targets in all. They are **trained heads, not derived at prediction.**
  - **Gage height is dropped**: no gage-height series exists in the daily tier on disk.
  - Loss weights: equal weights on standardised anomalies (each target divided by its own reference-window
    climatological sd; `train_v2.py` `loss_fn`, `nets.pinball`). This is the implementation of "fixed per-target
    inverse-scale normalisations".
- **Parity (§2b):**
  - §2b says: "Every model that can take an input gets the same: … auxiliary targets, where the model can take them."
    §2b names no stacking route. So:
    - SONDE-HF (all configs) trains on all 20 heads; C3 = primary-only loss weights.
    - R-NH (one target per model, by design), GBM-NWP and PF-GBM (per-variable models) fit a separate model for each
      auxiliary target, for the §2d numeric report. They cannot take multi-task supervision, so they get none.
    - Chronos-2-cov forecasts every cube variable present at a site as a target row (multi mode).
  - Auxiliary targets never enter or re-weight the §4 decision.

**A8 (item 8) — bias map.**
- **Stamped (§2c):** "a per-site linear map from forecast to gridMET is fitted on train-block overlap only … per
  month-of-year where the site has ≥ 60 overlap days in that month, else a pooled HUC2 map."
- **Code** (`sentinel/models/sonde_v2/nwp.py:124`): the fitted slope b is clipped to [0.5, 1.5]. Where neither a
  site-month nor a pooled HUC2-month map exists, the site's all-month map is used (`nwp.py:166`, source code 2).
- **Reason:** to guard against degenerate fits on short or odd overlaps.
- **Parity:** one map per product, applied to every model alike.
- **Counts of site-months per source** (train-block weather only): PENDING. They are computed from `BiasMap.source`
  when the full WY2008–2019 reforecast extraction completes, and added here as a dated addendum before the stamp.

**A11 (item 11) — CV early-stopping set.**
- **Stamped (§2h):** "Early stopping on an inner 15 % of train-role basin groups, never on the val block."
- **Code** (`scripts/train/sonde_v2_cv.py:102`): in CV, early stopping uses 15 % of the in-fold basin groups (seed
  20261110 + fold). These include val-role sites' **train-block** data, and never a val-block row.
- **Approved for CV** by the lead (L3 fixes CV fits on in-fold sites × train block). The same rule applies to every
  learned model (SONDE-HF, R-NH, GBM-NWP and PF-GBM early stopping).
- **Final fits** use 15 % of **train-role** basin groups, as §2h says. Step 1 already does
  (`scripts/train/sonde_v2_step1.py`, train-role sites only).

**A12 (item 12) — GBM row cap.**
- **Stamped (§2b):** "GBM-NWP: the current GBM-w plus per-lead NWP features". The frozen GBM-w config has
  `max_fit_rows: null` (`configs/sonde/gbm.yaml:21`).
- **Code:** step 1 caps at 1.5 M fit rows per (variable, h) (`sonde_v2_step1.py:147`); CV at 1.5 M
  (`sonde_v2_cv.py:221`). Early stopping uses 300k sampled inner rows (`sonde_v2_cv.py:227`).
- **Condition:**
  - A **full-rows configuration (max_fit_rows = null)** is one of the ≤ 5 parity configurations of GBM-NWP and of
    PF-GBM, unless one fit of it exceeds 2 h at 6 threads. In that case the largest cap that fits in 2 h is
    declared in the parity log before any CV run.
  - Step 1 states its cap and the total row count in the parity log (`n_rows_total_before_cap`).
- **Timing** (temp_c_mean, fold 0, h = 3, 6 threads): PENDING. Reported to the lead and added here as a dated
  addendum with the declared caps, before any CV run.

**A13 (item 13) — PF-GBM physics features.**
- **Stamped (§2b):** "PF-GBM: GBM-NWP plus uncalibrated physics features: … Raymond K600 from Q and slope, and a
  two-end-member SC mixing estimate."
- **Slope was found on disk:** column `nhd_slope` in `processed/catchment/streamcat_sites.parquet` (6,040 of 6,516
  registry sites).
- **Code** (`prep.py:199` `add_slope_static`):
  - raw `slope` feeds the PF-GBM Raymond K600 feature (`gbm_nwp.py:99`);
  - `sc_log10_nhd_slope` is a **static given to every model alike**. The frozen feature table and the frozen tensors
    pick it up through the `sc_` prefix: SONDE-HF, R-NH, GBM-NWP and PF-GBM.
  - Sites with no slope get NaN, and the K600 feature is NaN there.
  - The SC mixing estimate uses the **last observed discharge**, because no discharge forecast exists in the inputs.
    This is a deviation in that the stated "Q" is the last observed Q.
- So the case is "**slope found on disk, K600 built**"; nothing was downloaded.

## Group B — implementation specification where the stamped text is silent or cites plan §4.1

**B2 (item 2) — SONDE-HF hourly branch.**
- **Stamped:** §2a cites plan §4.1 ("the last 14 d × 24 h of T, DO, SC, pH and turbidity (+ masks, solar
  elevation)").
- **Code** (`data_v2.py:146` `load_hourly_cube`): per hour, a site-standardised level and an observed mask for
  T/DO/SC/pH/turbidity, plus **hour-of-day sin/cos**.
  - **No solar elevation.**
  - Approved values only; local standard time; reserved sites excluded.
  - Branch initialised from the daily LSTM state at d − 14 (`nets.py`, `SondeHF.forward`).
- **R-NH has no hourly branch, by design** (§2b: "no hourly branch").
- **Parity:** the hourly branch is SONDE-HF's own architectural element (C1 vs C2 tests it). The hourly references of
  M1-H get hourly inputs under the M1-H protocol.

**B3 (item 3) — daily hindcast inputs and statics.**
- **Stamped:** §2a cites plan §4.1.
- **Code** (`data_v2.py:66`): the frozen `sentinel.models.sonde.lstm.SondeTensors` (imported, not edited):
  - per variable: standardised anomaly vs the 2-year reference-window climatology, site-standardised level and
    observed flag;
  - DOY sin/cos;
  - 6 lagged gridMET channels + mask.
  - Statics:
    - lat, lon, elevation, log drainage area;
    - StreamCat `sc_*` attributes, including the new `sc_log10_nhd_slope`;
    - per-variable site mean and log-sd, each with availability;
    - HUC2 embedding → MLP to 32.
  - **There is no site-type input and no NHDPlus attribute other than slope** (StreamCat holds the rest).
- **Condition (parity):**
  - an identical static set and an identical training-site set (all site types) for SONDE-HF, R-NH, GBM-NWP and
    PF-GBM;
  - written per run to `sentinel_runs/sonde/v2_cv/PARITY_LOG.jsonl` (`sentinel/models/sonde_v2/parity.py`: row
    count, site count, site-types mix, train-site hash, feature-column hash, static columns).

**B4 (item 4) — forecast-branch inputs.**
- **Stamped:** plan §4.1 lists the NWP embedding, clear-sky shortwave, day length and lead one-hot.
- **Code:** per lead, `data_v2.py:110` (`fdet`) and `nets.py` (`SondeHF.forward`, `RNH.forward`):
  - masked-mean NWP embedding;
  - clear-sky SW;
  - day length;
  - DOY sin/cos;
  - lead one-hot;
  - static embedding.
- **Fixed before step 1:**
  - R-NH previously received only DOY sin/cos from these. It now receives the **same four deterministic inputs**
    (`nets.py:160`; `train_v2.py` forward).
  - GBM-NWP / PF-GBM now receive target-day clear-sky SW and day length as features (`gbm_nwp.py:108`
    `_det_blocks`). The target-day DOY sin/cos is already in the frozen feature table.

**B5 (item 5) — heads.**
- **Stamped:** plan §4.1 ("one head per target t and lead k").
- **Code:**
  - `MonotoneQuantileHead` is one linear map from [forecast-LSTM state ‖ static embedding] at each lead to all
    targets × 19 monotone quantiles. Lead identity comes from the lead one-hot input (`nets.py:210`).
  - **R-NH uses the same head form** on [state ‖ static embedding] (`nets.py:162`): CMAL for R1/R2, monotone
    quantile otherwise.

**B7 (item 7) — stage-1 error model.**
- **Stamped (§2c):** "stage 1 is perfect-prog with a lead- and season-dependent error model (GEFS
  reforecast-minus-gridMET residuals at train-role sites, WY2008–2019)".
- **Code** (`nwp.py:173` `ErrorModel`, `fit_error_model`; `products.py` `build`):
  - Gaussian errors per lead × field × season (mean, sd), with AR(1) correlation across leads per field;
  - fitted on bias-corrected GEFS reforecast minus gridMET at train-role sites, issue dates ≤ 2019-12-31 (train-block
    weather only, no target);
  - **one fixed-seed sample** (seed 20261110) per run.
- **Precipitation:** forecast = max(observed + Gaussian error, 0) (`nwp.py:199`). The zero-inflated field is
  clipped, not modelled; zero days can become positive and wet days can become 0.
- **Condition:**
  - both sides of step 1 and every learned comparator see the **same** noise sample: the same seed, the same site
    and date axes (`prepare_v2(roles=("train","val"))` in both drivers), and one array for both models in step 1;
  - the error model uses train-block weather only.

**B9 (item 9) — training budget.**
- **Code** (`train_v2.py:37–39`; `sonde_v2_cv.py:163`):
  - stage 1: max 30 epochs × 800 steps, batch 256, patience 4;
  - stage 2: lr/3, max 10 epochs, patience 3;
  - AdamW lr 1e-3, grad clip 1, fp16 autocast.
- **R-NH gets the same budget:** the same `TrainConfig` defaults and the same CV stage 1/2 code path.

**B10 (item 10) — site types.**
- **Code:** `prep.py` `prepare_v2(site_types=None)`, i.e. all site types (streams, lakes, estuaries), because the
  M1-N primary is all classes. The frozen M1 configs used streams only.
- **Parity:** the same site set for every model, logged per run (B3).

**B14 (item 14) — step-1 product slots.**
- **Code:** step 1 uses one NWP product slot, the perfect-prog GEFS-error product (`sonde_v2_step1.py:110`), for
  both models. CV uses two slots (GEFS, GFS).
- This is consistent with §2f: "perfect-prog gridMET for the target days plus the stage-1 NWP error model".

## D1–D5 preliminary conditions noted (part 1 unchanged)
- **D2:** the literal 19-level variant **Q19** exists as a parity configuration (`gbm_nwp.py` `quantile_levels=19`;
  `sonde_v2_cv.py` CONFIGS `"Q19"`, for both GBM-NWP and PF-GBM). The step-1 default is named **Q** in the parity log.
- **D4:** `LOG_VARS` is registered only in `sentinel/models/sonde_v2/prep.py`, which only v2 processes import. No
  frozen original-track script imports `sentinel.models.sonde_v2`.

## Log
| Date | Change | Reason |
|---|---|---|
| 2026-11-10 18:04 UTC | v0.1: items 1–14 recorded; code fixed for items 4, 5 and 13 before step 1 | lead ruling of ~18:00 UTC |
| 2026-11-10 18:25 UTC (13:25 EST) | v0.2: item 15 (reference-window climatology conformance) added; body closed for review — later A8/A12 numbers only as append-only dated addendum lines below | lead message of ~13:24 EST |
