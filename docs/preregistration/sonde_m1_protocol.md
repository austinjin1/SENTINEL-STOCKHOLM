# M1 SENTINEL-Sonde — pre-registration protocol (forecasting, Test-A / Test-B)

**Status:** DRAFT v0.1 (2026-11-08). Implements RESEARCH_PLAN.md §5.1, §14, §15.2 (M1 primary).
Frozen together with, or after, the EW-v2 record (`ew_v2_protocol.md`); test blocks are scored once,
only after the lead authorises it. **Authors:** Austin Jin, Bryan Cheng (AI-assisted).

## 1. Benchmark (original track — always reported)
- Split `docs/preregistration/sonde_split.json` (sha256 `fd4bb14fe7304d05310e286e9aced22675fd6d07aa4dcac6415ed039a1e06a80`); stream sites (ST, ST-TS, ST-CA, ST-DCH); blocks Train WY2008–21, Val WY2022–23, **Test-A WY2024–25**, Test-B WY2026 (Approved values only), 42-day embargo.
- Cells: seen × Test-A, unseen (test-role) × train-period, **unseen × Test-A (headline)**, NEON × Test-A/B (if accessible).
- Variables: temperature, DO mean, DO min, SC, pH, turbidity (mean, median), NO3+NO2-N; horizons 1, 2, 3, 5, 7, 10, 14 d; 19 quantiles clipped to `PLAUSIBLE`.
- **Primary (§15.2):** median per-site CRPS skill of the frozen SENTINEL model vs the best of the plan references {damped anomaly, climatology, persistence, GBM (with weather), LSTM (with weather), zero-shot Chronos-2 uni/multi, zero-shot TimesFM-2.5} on unseen × Test-A, basin-group cluster bootstrap (B = 2,000), lower 95% bound > 0 at every h ∈ {1, 2, 3, 5, 7} for ≥ 3 of {turbidity, SC, nitrate, DO, temperature}, in each of 3 seeds. Holm across the five variables.
- **Strict control (reported next to the primary):** the same test against equal-weight ensembles of baselines only (GBM-w + LSTM(-w) + zero-shot Chronos-2 multi), so that an ensembling gain is not credited to SENTINEL.
- N-b virtual sensing: KGE improvement over GBM for ≥ 2 variables (basin-group CI > 0).

## 2. Freeze candidate
Candidate prepared 2026-11-08 (`sonde_m1_freeze_candidate.json`, NOT frozen): `ens3w` = equal-weight Chronos-2-FT + LSTM-weather + zero-shot Chronos-2 multi, seeds s0–s2, member checkpoint SHA-256s listed. On val the bar is **not** met on the unseen cell (SC 4–5/5 horizons, turbidity mean 4–5/5, nitrate 2–3/5, DO mean 2/5, DO min 0/5, temperature 0/5; 8–13 basin groups per variable). Freeze for Test-A is a lead decision.

## 3. Harder track M1-X (§15.5 amendment items 5–7; added 2026-11-08, before any result on it)
Saturation note: on familiar (train-role) sites, temperature and DO at h = 1 d are near ceiling (N-b temperature KGE ≈ 0.89 for both GBM and LSTM; all strong methods within CIs). The unseen-station cell is not saturated. M1-X adds, without replacing the original track:
- **X1 storm windows:** issue days whose trailing 3-day observed precipitation (gridMET, lagged as in `common.load_weather`) ≥ the site's training-block P95 of that quantity.
- **X2 heat windows:** issue days whose trailing 3-day mean tmax ≥ the site's training-block P95.
- **X3 extremes:** threshold-weighted CRPS with weight 1{target beyond the site × DOY training climatology by > 1.645 SD} (DO: lower tail). This is a proper score, not CRPS conditioned on the outcome.
- **X4 longer horizons (21, 30 d):** requires retraining, so it is planned only after the Test-A freeze.
- Same frozen split and blocks (the strata are row subsets defined from training-block forcings or climatology; definition code `scripts/eval/sonde_strata_power.py`, hashed at freeze). Power check on val **before** any model is scored on a stratum (2026-11-08, `sentinel_runs/sonde/m1x_strata_power_val/strata_power_val.csv`, h = 3 d; counts only): unseen (val-role) sites / basin groups in X1 storm — DO min 40/8, SC 97/12, temperature 166/13, turbidity mean 16/9, nitrate 7/3; X2 heat similar; X3 extremes — DO min 35/8, SC 65/12, temperature 157/13, turbidity mean 15/8, nitrate 0/0. Pre-specified minimum for a stratum claim: ≥ 5 basin groups with stratum rows at unseen sites → nitrate (all strata) and turbidity median are descriptive only. No model has been scored on any stratum. Claims name the track.

## 4. Original-track Test-A look: report-only, hardened single look (added 2026-11-10; lead approval in principle)
- **Status.** The original M1 track (test-role × Test-A) is scored **once, report-only**. It is the frozen original
  benchmark; no new-track or SOTA claim rests on it (the SOTA attempt is M1-N, `m1n_protocol.md`). The candidate stays
  as prepared on 2026-11-08 (`sonde_m1_freeze_candidate.json`). On val the bar is not met.
- **Disclosure: the DO labels are exposed.** The EW-v2 Test-A pass (2026-11-09) loaded and scored DO-min labels at
  test-role sites in WY2024–25 (H1 onsets: 43, 11 basin groups). The DO-min and DO-mean forecasting targets of this
  look are therefore **not blind in spirit**, and this is stated wherever the result is reported. The other variables
  (temperature, SC, pH, turbidity, nitrate) were loaded by that pass as model inputs only, never scored, also disclosed.
- **Plan.** `docs/preregistration/sonde_m1_testA_plan.json`, built by `scripts/eval/sonde_m1_testlook.py --phase plan`.
  It holds every predict job (model, config, overrides, checkpoint + SHA-256, cross-checked against the candidate), the
  ensembles (the candidate ens3w s0–s2 plus the strict-control ensembles; equal weights), and the scoring call
  (9 references, 3 strict controls, 3 candidate seeds; B = 2,000).
- **Val rehearsal.** `--phase rehearsal` re-runs every job on the VAL block and requires reproduction of the stored val
  forecasts. It is streaming: outputs are deleted after the comparison, to respect the disk cap.
- **Rehearsal findings and the weather-input rule (2026-11-10, before any freeze; no test data touched).**
  - The forcings product was rebuilt on 2026-10-27 (`forcings_daily/combine.log`). The rebuild added ERA5 forcings
    at 79 M1 sites (57 train-role, 22 val-role) that had none when the weather models (`gbm_weather`,
    `lstm_weather_s0–s2`, trained and scored on 2026-10-24) were built: 1,826 vs now 1,905 of 1,906 sites with
    forcings. Re-predicting with the current forcings changes val forecasts **only** at those ERA5-forced sites
    (gbm_weather: max |Δ| 1.47 °C on temperature, every other site Δ = 0).
  - **Rule.** Every weather job runs with `sonde_predict.py --weather-exclude-sources era5_seamless`. This sets NaN
    weather at every ERA5-forced site, which reproduces exactly the inputs of the candidate's training and val
    evidence (gbm_weather re-predicted on val with the rule: Δ = 0 in all 8 files). Test-role sites forced by ERA5
    get NaN weather, as such sites did in training. The rule is a reproduction rule fixed before the look; it was not
    chosen on any score. **Lead ruling 2026-11-10:** this fix is correct. The look reproduces the pre-rebuild
    (2026-10-24) weather inputs the candidate was trained and validated on, so the candidate's val evidence stands.
    Disclosure, wherever the result is reported: the forcings product was rebuilt on 2026-10-27, and the look
    deliberately uses the pre-rebuild inputs (ERA5-forced sites without weather).
  - **Comparison bug, fixed.** The first rehearsal compared `chronos2_zs_uni` with the *zs_multi* file stored in the
    same directory (glob without the model prefix). It now compares the exact file name, and the job is re-run.
- **Execution rules (2026-11-10, lead GPU rule update; no change to any model or forecast).**
  - GPU jobs may use GPUs 0–9 through `sentinel/core/gpu.py`, idle GPUs only (`--min-free-mb` 10000 of 11019 MiB, so
    a GPU used by another user is never taken). The frozen `runner.M1_GPUS = (0,)` is widened per process by
    `sentinel/models/sonde/gpu_budget.py`; the frozen file is not edited. At most 3 GPU jobs run at once
    (2 CPU threads each = the 6-thread rule), and CPU jobs run alone.
  - **≤ 2 h per GPU process.** Multivariate Chronos (FT s0–s2 and zero-shot multi) took about 2.7 h per seed on val in
    one process. The first rehearsal pass exceeded the cap; this is disclosed. These jobs now run with
    `--max-minutes 100`:
    - after the first batch flush past 100 min, the process saves a resume state and exits (code 75), releasing its
      GPU;
    - the orchestrator re-runs the same command, which resumes at the next site;
    - stops happen only at flush boundaries, where the pending batch is empty, so every batch, and hence every
      forecast, equals an uninterrupted run.

    The val rehearsal on the final plan verifies this (Δ = 0 required). Zero-shot univariate Chronos and TimesFM
    take about 1 h on val, roughly 1.3 h expected on test, and run unchunked.
  - The rehearsal re-runs **all 12 jobs** with exactly the final plan's commands and records the plan SHA-256 in each
    identity entry.
- **Hardened single look** (`--phase look`). It needs a lead-written `FREEZE_sonde.json` that pins the plan SHA-256
  and the full import-closure code hash (`testlook_plan_sha256`, `testlook_code_sha256`). Before the claim it verifies
  every checkpoint, the registry (no `sonde_m1_testA_score` row) and the absence of any marker. The look then:
  1. claims with an O_EXCL `CLAIM.json` holding pid and host, plus a `look_claimed` row;
  2. predicts;
  3. seals the predictions (`PREDICTIONS_SHA256SUMS`, 0444, plus a row);
  4. scores with `--register`;
  5. writes the numeric-forecast report (below);
  6. writes `SCORED.json` plus a `look_sealed` row.

  Any `BaseException` writes `ABORTED.json` and a registry row. A rescore is refused.
- **Numeric forecasts (required; user requirement relayed by the lead 2026-11-10, added before any freeze).** Every
  model reports specific numeric values, not only skill scores or events. `scripts/eval/sonde_pi_report.py` (in the
  look's import-closure code hash) reads the sealed predictions and the frozen common rows and writes, for every model,
  every reference (including persistence and climatology), every variable and every horizon h ∈ {1, 2, 3, 5, 7}:
  - the point forecast, defined as the predictive median, in original units;
  - prediction intervals 50 % (q25–q75), 80 % (q10–q90) and 90 % (q05–q95), with empirical coverage and mean width;
  - MAE, RMSE and bias of the median, and MAE/RMSE skill vs persistence and vs climatology (1 − metric/reference,
    pooled over the scored rows).

  Per-row numeric forecasts (site, issue date, h, observation, q05/q10/q25/q50/q75/q90/q95) are written for the three
  candidate seeds (`numeric/numeric_forecasts_test.parquet`). Pooled metrics for all labels go to
  `numeric/numeric_metrics_test.csv`. They include:
  - the median over sites (≥ 30 scored rows) of coverage, MAE, RMSE and MAE skill vs persistence, with per-site values
    in `numeric/numeric_site_metrics_test.csv`;
  - basin-group cluster percentile bootstrap 95 % CIs (B = 2,000, seed 0; the same draws as the frozen
    `bootstrap_ci`) for pooled MAE, RMSE, coverage and the MAE/RMSE skills, for every model, reference and strict control.
    The naive references' own rows have no CI. These are descriptive companions to the pre-registered CRPS-skill endpoint, and
  no decision rests on them. Quantiles are sorted before use (the frozen evaluator's crossing guard). The same report
  was run on val as a check (handoff).
- **Disk (lead ruling 2026-11-10).** Approved up to **15 GB** for the sealed forecasts and the numeric tables, above
  M1's 10 GB cap. Before the claim the look refuses to start if free space is below 60 GB + 15 GB.
- **Disk (original estimate).** The look writes about 13 GB of test forecasts (about 0.75 GB per model or ensemble), which needs the
  lead's approval over M1's 10 GB new-data cap, or a prior clean-up of superseded M1 runs (`sentinel_runs/sonde` holds
  52 GB).
