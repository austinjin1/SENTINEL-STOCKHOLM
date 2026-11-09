# Datasheet — EW-v2 H2 derived chloride (`chloride_derived`)

| | |
|---|---|
| **Inputs** | `wqp_discrete` (variable `chloride`, mg/L, USGS stations `USGS-<site_no>`; rebuilt 2026-10-25, 93.9 M processed rows) and `sonde_daily` (`spcond_uscm_mean`, `spcond_uscm_max`) |
| **Code** | `sentinel/eval/chloride.py` (rules), `scripts/data/build_chloride_derived.py` (build), `tests/test_chloride.py` |
| **Licence** | US public domain (derived from USGS/WQP data) |
| **Outputs** | `$SENTINEL_DATA/processed/chloride_derived/{regressions.parquet, pairs.parquet, sites/<site>.parquet}`; manifest `chloride_derived` |
| **Built** | 2026-10-25 |

## Method (pre-specified, EW-v2 protocol §3)
Same-day pairs of lab chloride (censored values dropped; replicates → median) and USGS daily-mean SC. Per site, OLS
`Cl = a + b·SC` on **training-block pairs only (≤ 2021-09-30)**. Eligible: ≥ 20 pairs, R² ≥ 0.9, b > 0, SC at
230 mg/L ≤ 1.5 × max paired SC. Derived series: `cl_mgl_mean`, `cl_mgl_4dmean` (4-day trailing mean, all days
observed), `cl_mgl_max` (from daily-max SC). Hazards `H2_chloride` (4-day mean > 230, `pre_gap` 3) and
`H2_chloride_acute` (> 860) in `sentinel/eval/hazards.py`.

## Coverage
102,133 chloride rows at 1,172 USGS stations with daily SC → 45,099 pairs. **52 eligible sites** (split roles:
27 train, 9 val, 13 test, 3 not in split). Rejections: < 20 pairs 809, R² < 0.9 273, extrapolation 33, slope ≤ 0 5.
Eligible fits: median R² 0.95, median 66 pairs. Sensitivity (not primary): R² ≥ 0.8 → 80 sites; R² ≥ 0.7 → 96.

## Onset counts (training and validation blocks only; test block not inspected)
| Hazard | Block | train-role | val-role | test-role sites |
|---|---|---|---|---|
| H2 chronic | WY2008–21 | 115 (15 sites) | 19 (4) | 110 (12) |
| H2 chronic | WY2022–23 | 5 (3) | 1 (1) | 6 (5) |
| H2 acute | WY2008–21 | 89 (10) | 6 (1) | 118 (10) |
| H2 acute | WY2022–23 | 6 (2) | 0 | 10 (5) |

## Known issues
* H2 is **underpowered** for calibration (1 chronic onset at val-role sites in WY2022–23) and very likely for the
  test block (≥ 50 onsets required); recent winters have far fewer onsets than WY2008–21 at these sites.
* A single linear relation ignores seasonality/hysteresis; SC from non-chloride ions (e.g. sulfate) can inflate
  derived chloride at some sites (R² ≥ 0.9 screens most).
* Daily-max SC is a proxy for the 1-h acute criterion.

## Rebuild
`python -B scripts/data/build_chloride_derived.py`

## Rebuild from the promoted daily tier (promoted 2026-11-08; EW-v2 pre-test re-review C1)
The 2026-10-25 build read `sonde_daily` when it was the 2026-10-22 USGS pull. After the 2026-11-08 approval-status refresh was promoted, the series were rebuilt from the promoted daily tier (`build_chloride_derived.py --out-name chloride_derived_refresh_20261108 --compare-with chloride_derived`). `processed/chloride_derived` -> `chloride_derived_refresh_20261108` (symlink; old build kept as `chloride_derived_20261025`). Identity check (`compare_vs_chloride_derived.json`): all 1172 candidate-site regressions identical (a, b, R², n_pairs, eligibility; 52 eligible) and all 52 derived site series identical on every date <= 2023-09-30 (max |Δ| = 0, NaN pattern identical), so no train/val value changed. Pairs: 45095 (fit pairs unchanged; only post-2021 'later' pairs can differ, they are unused). Manifest now hashes every file.
