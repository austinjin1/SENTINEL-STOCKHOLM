# SENTINEL-HydroMet (M6) — test results (single looks of 2026-11-09)

Written after both looks from the sealed outputs and their registry rows only. It does not amend
`hydromet_protocol.md` (frozen; track R shares its hash). Sources:
- `M6/test`: `$SENTINEL_RUNS/hydromet/test_looks/M6_test/` (`TEST_LOOK.json` claimed 2026-11-09 21:14 UTC,
  sealed `predictions/` + `SHA256SUMS`, `SCORED.json` 16:32 EST), `score/primary_test_test.csv`,
  `score/summary_test_test.csv`; registry experiment `M6/test` (1,785 rows incl. the claim row).
- `M6N/test`: `.../test_looks/M6N_test/` (claimed 21:33 UTC, `SCORED.json` 17:20 EST), `score/primary_test.csv`,
  `score/summary_test.csv`, `score/sensitivity/`; registry `M6N/test` (117 rows) and `M6N/test_sens427` (116 rows).
- Freeze `docs/preregistration/FREEZE_hydromet.json` (written by the lead). Cells: FINAL test-role sites/stations ×
  Test-A (WY2024–2025, 42-day embargo).

## 1. Primary look `M6/test` (protocol §5–§8)
Metric: median over sites of per-site KGE; Δ = SENTINEL − frozen comparator on the sites scored by both;
paired basin-group bootstrap (B = 2,000), two-sided p; intersection–union p_IUT = max(p_gbm, p_phys); Holm over
the 28 hypotheses (7 models × 4 endpoints, all tested); won = Holm rejects ∧ both Δ > 0 with CI > 0 ∧ every seed
Δ > 0 against both comparators.

**Label model `ealstm_ws_full` (3-seed ensemble):**

| Endpoint | sites / groups | KGE model | Δ vs best GBM [95 % CI] (comparator) | Δ vs best physics [95 % CI] (comparator) | every seed > GBM / > physics | p_IUT | Holm | won |
|---|---|---|---|---|---|---|---|---|
| temp mean | 333 / 33 | 0.902 | −0.006 [−0.020, 0.003] (`forcing_gbm_bag` 0.907) | +0.017 [0.006, 0.033] (`air2stream_regional_ws` 0.885) | no / yes | 0.193 | no | no |
| temp 7-DADMax | 330 / 32 | 0.890 | +0.004 [−0.007, 0.018] (`forcing_gbm` 0.886) | +0.014 [0.004, 0.031] (`air2stream_regional_ws` 0.876) | no / yes | 0.357 | no | no |
| DO mean | 178 / 22 | 0.789 | −0.000 [−0.020, 0.035] (`forcing_gbm` 0.790) | +0.090 [0.054, 0.142] (`dosat_ratio[air2stream_regional]_ws` 0.699) | no / yes | 0.951 | no | no |
| DO min | 185 / 22 | 0.800 | +0.013 [−0.018, 0.046] (`forcing_gbm_bag` 0.788) | +0.071 [0.038, 0.104] (`dosat_ratio[air2stream_regional]` 0.730) | no / yes | 0.404 | no | no |

Per-seed Δ vs GBM (temp mean, 7-DADMax, DO mean, DO min): [−0.001, −0.013, −0.016], [+0.003, −0.004, +0.001],
[−0.000, +0.003, −0.007], [+0.006, +0.010, −0.002]. Median per-site RMSE, temp mean: 1.44 °C [1.37, 1.62]
(`ealstm_ws_full`) vs 1.33 °C [1.24, 1.49] (`forcing_gbm_bag`); DO mean: 1.00 vs 0.98 mg/L (`forcing_gbm`, the frozen DO-mean comparator).

**§15.1 label for `ealstm_ws_full`: "competitive"** — its CI overlaps the best GBM on all four endpoints; it beats
the best physics/process comparator with CI excluding 0 on all four (and every seed beats physics), but no
hypothesis is won and Holm rejects none. The M6 H-G1 criterion ("beats GBM and physics with CI excluding 0 for
≥ 2 of 5 targets") is **not met**. As pre-declared (§8), the test cell was expected to be underpowered for the val-sized gaps, and a "SOTA" label was excluded in advance. More power would not be expected to change the label: on temperature mean, the candidate's point estimate favours the GBM, and four other standalone EA-LSTMs are significantly worse than it. The val-era gains did not replicate. On test, upstream forcings (`ealstm_ws` vs `ealstm`) give Δ −0.001 [−0.009, 0.010] / +0.001 [−0.014, 0.015] / −0.003 [−0.020, 0.014] / −0.015 [−0.031, 0.007], against +0.012 / +0.019 on val temperature. The candidate's val Δgbm (+0.002 / −0.002 / −0.016 / +0.011) became −0.006 / +0.004 / −0.000 / +0.013. The candidate was selected over 6 val cycles: 8 val-scored EA-LSTM variants plus the capped `ealstm_v3`, a 4-point EA-LSTM inner grid, and the GBM 3-point inner grid × 5 variants (RESULTS.md, cycle-6 line). "Physics" here means the pre-registered simple physics/process baselines: an air2stream-type model without discharge, transferred by k-nearest donors, and DO_sat × the monthly ratio.

**Other models in the Holm-28 family (secondary; none won, none Holm-rejected):** on temperature mean, four
standalone EA-LSTMs are below the best GBM with CI excluding 0 — `ealstm` −0.014 [−0.030, −0.002], `ealstm_physdo`
−0.018 [−0.036, −0.005], `ealstm_full` −0.011 [−0.027, −0.001], `ealstm_compact` −0.026 [−0.048, −0.013];
`ealstm_ws` −0.015 [−0.031, 0.001]. `ealstm_gbmblend` (reported separately; contains the GBM): Δgbm −0.002
[−0.013, 0.005] / +0.004 [−0.012, 0.019] / +0.005 [−0.009, 0.020] / −0.003 [−0.022, 0.026]; Δphysics CI > 0 on temp
mean, DO mean and DO min. Every SENTINEL model beats the DO physics references with CI excluding 0.

## 2. Track N look `M6N/test` (protocol §12; secondary family, Holm over 4)
Unsensed WQP stream stations (test role), daily mean of grab samples vs the daily-mean prediction. Metric: median
over stations (≥ 8 sample days) of per-station RMSE; Δ < 0 = better. Comparators fixed from `M6/val_v4`.

Power (pre-declared, protocol §12): the test cell holds 3,014 / 2,724 stations in 38 / 36 basin groups, about 3× the val cell's 12 groups. Track N adds stations, not groups, for selection.

| Model | Endpoint | stations / groups | Δ RMSE vs best GBM [95 % CI] | Δ RMSE vs best physics [95 % CI] | every seed better (GBM / physics) | p_IUT | Holm | won |
|---|---|---|---|---|---|---|---|---|
| `ealstm_ws_full` | temp mean (°C) | 3,014 / 38 | **+0.161 [0.095, 0.246]** (worse) | −0.135 [−0.225, −0.067] | no / yes | < 0.001 | rejects (direction: worse than GBM) | no |
| `ealstm_ws_full` | DO mean (mg/L) | 2,724 / 36 | −0.012 [−0.063, 0.051] | −0.097 [−0.182, −0.046] | no / yes | 0.476 | no | no |
| `ealstm_gbmblend` | temp mean | 3,014 / 38 | +0.047 [0.005, 0.079] (worse) | −0.249 [−0.325, −0.217] | no / yes | 0.026 | no | no |
| `ealstm_gbmblend` | DO mean | 2,724 / 36 | **−0.035 [−0.068, −0.002]** | −0.121 [−0.229, −0.053] | yes / yes | 0.038 | no | no |

Median per-station RMSE: temperature `ealstm_ws_full` 2.01 °C [1.86, 2.18] vs `forcing_gbm_bag` 1.84 [1.74, 2.01] (KGE 0.812 vs 0.839); DO `ealstm_gbmblend` 1.32 mg/L vs `forcing_gbm` 1.35 (KGE 0.635 vs 0.615). Median per-station bias: temperature `ealstm_ws_full` +0.19 °C [−0.19, 0.42], `forcing_gbm_bag` +0.22 [0.03, 0.34]; DO `ealstm_ws_full` −0.31 mg/L [−0.50, −0.08], `ealstm_gbmblend` −0.37, `forcing_gbm` −0.37.

**Track-N temperature is a §15.1 "negative" result for `ealstm_ws_full`:** it is worse than the GBM with CI excluding 0, and every seed is worse (+0.19, +0.26, +0.21 °C). This is the only Holm rejection in this family, and it is in the unfavourable direction. It arises because the frozen intersection–union test uses two-sided p values. The registry rows of this hypothesis carry `holm_reject: true` and must not be read without the sign of Δ. Using two-sided p changed no decision.

The blend's DO result is narrow:
- Its median RMSE is lower than that of the frozen comparator `forcing_gbm` (CI below 0; every seed lower, but seed 0 by only 0.0006 mg/L).
- p_IUT = 0.038 does not survive Holm. After the first rejection (p < 0.001 ≤ 0.0125), the next p_IUT, 0.026 (blend temperature), exceeds 0.05/3 = 0.0167, so Holm stops. The result is not won.
- It is not lower than the bagged GBM the blend contains (`forcing_gbm_bag` 1.336 vs blend 1.316 mg/L; not a frozen comparison).
- Its every-seed margin fails in the sensitivity re-score.

The new-region-only secondary of §12 was not produced by the look. It is not reported here (4 basin groups).

**Pre-declared sensitivity (`M6N/test_sens427`, descriptive).** This re-score excludes the 87 test-role stations whose basin fell back to the local delineation (of the 427 overall), which removes 62 scored stations per endpoint. Every Δ and CI moves by ≤ 0.005, and the Holm decisions are unchanged: `ealstm_ws_full` temp Δgbm +0.157 [0.092, 0.240], DO −0.011 [−0.061, 0.045]; blend temp +0.047 [0.004, 0.077], DO −0.033 [−0.066, −0.003]. One thing does change: the blend's DO every-seed rule vs `forcing_gbm` is no longer met (seed 0: +0.002 mg/L).

## 3. Summary as registered
- M6 primary (`ealstm_ws_full`): **competitive** — level with the GBM (all CIs include 0), better than the
  physics/process baselines on all four primary endpoints (CIs exclude 0). 0 of 28 hypotheses won.
- Track N: at unsensed reaches the candidate is **worse than the GBM on temperature (§15.1 "negative" on track N;** +0.16 °C median RMSE, CI excludes 0, every seed worse) and level on DO. The GBM blend has a lower DO median RMSE than the frozen GBM comparator (−0.035 mg/L, CI excludes 0), but it is not Holm-significant, not robust in the sensitivity re-score, and not lower than its own bagged GBM. 0 of 4 won.
- Low-DO detection was **not evaluated on test**, and it was never evaluated for `ealstm_ws_full`. On validation (track S, `M6S/val_v2`; cycle 1–4 models and baselines), pooled CSI for daily-minimum DO < 5 mg/L was ≤ 0.033 for every model, including the GBMs: no evaluated model detects low DO.
- §7 secondaries on test (registered, descriptive):
  - physics-DO head vs `ealstm`: +0.009 [−0.022, 0.027] (DO mean) and −0.003 [−0.024, 0.012] (DO min);
  - upstream forcings, `ealstm_ws` vs `ealstm`: all CIs include 0 (see §1).

  The pre-registered no-gauge static ablation was not carried to test, because its artifacts are not in the freeze. It is val-only (`M6/val_v2`).
- Pending: track R (`M6R/test`, secondary, separate freeze) and the event track (`M6E/test`).
