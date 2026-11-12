# SENTINEL-Sonde architecture round v2 — GBM time→cap rule and timing benchmark

Written 2026-11-10 23:46 UTC by the M1 engineer under lead ruling E, BEFORE the timing run. The benchmark record
(`sentinel_runs/sonde/v2_smoke/gbm_timing.json`) carries the sha256 of this file. The lead rules on the GBM settings
after seeing the numbers; the ruling is entered in `sentinel/models/sonde_v2/declared.py` (`GBM_SETTINGS`) before the
code list of the consolidated deviations file is written. Nothing here is a result.

## 1. Fit unit

One unit = one scored variable, one horizon, all boosters of one configuration of one GBM arm:
Q19 = 19 quantile boosters; Q = 7 quantile boosters; LS = 2 boosters (L1 location, L2 log-scale).
Recipe (frozen `GBMConfig` values): learning rate 0.05, at most 2000 rounds, early stopping after 100 rounds without
improvement on the early-stopping rows, num_leaves 63, min_data_in_leaf 200, feature_fraction 0.8,
bagging_fraction 0.8 (every round), lambda_l2 1.0, max_bin 255, LightGBM 4.7.0, 6 CPU threads.

## 2. Cap ladder

Fit rows per variable and horizon, descending: full rows, 1,500,000, 1,000,000, 500,000, 250,000.
A cap is applied after the row masks (frozen training rows ∩ reference-window issue rule ∩ finite target) by
`gbm_nwp.stratified_cap`: every site keeps a share proportional to its row count (largest-remainder rounding), rows
drawn at random within the site, generator seed `[seed, crc32(variable), horizon, 0]`.
Early-stopping rows: the same masks on the early-stopping sites, the same subsampling with seed element 1, at most
400,000 rows. The same rule in step 1 and in CV, for both GBM arms.

## 3. Time of a unit

T(cap) = t_build(cap) + n_boosters × 2000 × s(cap), with t_build = feature binning time and s = seconds per boosting
round of one booster (mean over the timed boosters of that objective type). This is an upper bound: it gives no
credit for early stopping.

## 4. Cap of a configuration

The first cap on the ladder with T(cap) ≤ 2 h, evaluated for the more expensive arm (PF-GBM, which has the physics
features) on the scored variable and horizon with the most labelled fit rows. The cap is used by BOTH GBM arms, in
step 1 and in CV. LS runs at full rows if its unit passes. If the lead rules LightGBM `device=gpu`, T is taken from
the GPU timings and the same rule applies.

## 5. Benchmark (timing only)

* Data: train-role sites × train block only (`prepare_v2(blocks=("train",), roles=("train",))`); fit sites = the
  step-1 fit sites (folds 1–4 of the train-role basin-group split minus the 15 % early-stopping groups). No
  val-block row and no val-role site is loaded.
* Features: PF-GBM (physics features on). NWP input: observed gridMET of the target days (perfect-prog without the
  error-model noise; the feature count and row count are those of a real fit).
* Unit timed: temp_c_mean, h = 1 (the labelled fit-row count of every scored variable at h = 1 is written to the
  record; counts only).
* Fits: exactly 50 boosting rounds per booster, no validation set, no early stopping. Boosters: quantile α = 0.05,
  0.50, 0.95; L1; L2. Rows: full and 500,000. Devices: CPU with 6 threads; LightGBM `device=gpu` with max_bin 255 on
  one idle GPU out of 7, 8, 9 taken through the lock helper.
* s(cap) for caps that are not timed: linear interpolation in the row count between 500,000 and full rows;
  250,000 rows: s(500,000) / 2.
* Kept: wall times; the training loss after 50 rounds of the CPU and of the GPU booster on a fixed seeded sample of
  200,000 fit rows (equality check of the two devices only); the 1-minute load average at start and end.
  No skill, no early-stopping loss and no held-out loss is computed. No forecast file and no registry row is written.

## 6. Synthetic equality check of the two devices

Seeded synthetic regression, 200,000 fit rows × 40 features, 50,000 synthetic test rows, quantile α = 0.1, 0.5, 0.9,
200 rounds, the recipe of §1. Passes if the pinball loss of the GPU fit is within 1 % of the CPU fit for every α.

## 7. Load

The benchmark starts only if the 1-minute load average is ≤ 36. A timing taken under load is not used to justify a
smaller GBM setting: if the numbers would lead to a reduction, the benchmark is repeated after the GEFS extraction
has ended and that record is used.

## 8. Projections sent to the lead

At learning rate 0.05 / 2000 rounds (upper bound): step 1 (2 variables × 3 horizons, Q19) and CV per fold and arm
(8 variables × 7 horizons; Q19, Q, LS), scaled by the labelled-row count of each variable and, for CV, by the ratio
of train+val-role to train-role split sites (approximate).
