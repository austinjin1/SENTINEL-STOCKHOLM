# OSF preregistration text — EW-v2 (secondary data analysis template)

Paste-ready text for the OSF "Preregistration for secondary data analysis" form. The binding
document is `ew_v2_protocol.md` (v0.2); the archive `ew_v2_osf.tar.gz` holds it together with the
hashed code, the frozen calibration file and the model artifacts (`MANIFEST.sha256.json`).

## Part 1 — Study information
**Title.** SENTINEL EW-v2: pre-registered test of freshwater hazard-onset early warning at unseen USGS stations (Test-A, water years 2024–2025).

**Authors.** Austin Jin, Bryan Cheng (AI-assisted analysis; disclosed).

**Research questions.** Do daily probabilistic forecasts of hazard onset within 3 days carry skill over a site × season climatology at stream stations never used in training or calibration? Do alarms at a frozen false-alarm budget precede real onsets more often than season-matched placebo dates?

**Hypotheses.** H-a (primary, per hazard in the fixed family {H1 hypoxia, H2 chloride, H4 nitrate}): the Brier skill score of the pre-selected method vs the identically recalibrated climatology at h = 3 d is > 0 (one-sided, Holm over m = 3). H-b and H-c (secondary): paired Brier differences vs probabilistic baselines; POD before real vs placebo onsets. The validation evidence is weak and H1 is projected to have < 50 evaluable onsets, so the primary test is **expected to fail or be underpowered**. The result is reported whatever it is.

## Part 2 — Data description
**Datasets.** USGS continuous water-quality daily values and 15-min aggregates (public domain), gridMET daily weather, and Water Quality Portal chloride samples for the SC→Cl regressions. Snapshots are pinned in the freeze record (`data_snapshots`) by manifest SHA-256 and by a content hash over every file the loaders read; the score phase refuses any difference. The H2 derived-chloride series were rebuilt from the promoted daily tier (regressions and train/val series identical to the earlier build); the per-site regressions (`regressions.parquet`, the H2 label definition) are in the registration package. Test-A uses the daily-value snapshot rebuilt from a 2026-11-08 re-pull of WY2024–26, which captures approval status; its train/val rows are verified identical to the calibration snapshot.

**Access.** Public USGS NWIS / Water Data APIs, gridMET (THREDDS) and WQP. Retrieval dates are in the data manifests.

**Splits.** Frozen 2026-10-22 (`sonde_split.json`, SHA-256 `fd4bb14f…`). 2,678 stream sites in 120 basin groups (HUC4 merged across flow-connected stations), split 1,780 / 334 / 564 sites train / val / test. Time blocks: train WY2008–21, calibration WY2022–23, Test-A WY2024–25, Test-B WY2026.

**Prior knowledge statement.**
1. No model output was ever computed on test-role sites or on any WY2024+ date. All sonde run records (72 at the 2026-11-08 re-review, plus the later val-only rehearsal runs) show data ending 2023-09-30 and no protected access. The guard refuses protected data without the freeze record.
2. **Split eligibility (disclosed).** Site eligibility required ≥ 730 days of record over the full period, including WY2024–26. The split is therefore conditioned on test-period data *availability*, not on values.
3. **Chloride datasheet (disclosed).** The chloride-regression builder read all sites and years. Its datasheet reports H2 onset counts at test-role sites for WY2008–21 and WY2022–23 (no Test-A/B values).
4. **Power projection.** The H1 projection (≈ 44 Test-A onsets; ≈ 28 test-role basin groups with DO sites, ≈ 10 expected to carry onsets) used train/val-block onset rates and site metadata only.
5. **Validation results known to the authors.** H1 stack val out-of-fold BSS +0.022 (basin-group CI −0.015 to 0.043). Only persistence had a CI above 0. Configurations tried (val only): 8 calibration configurations and 5 offline stack variants. A 9th calibration run (`ew_live_variant_cal_nosubdaily`, val only) calibrates the live-trial variant without sub-daily features and was not used for selection.
6. **Approval-status descriptor.** The pooled share of provisional flags by water year after 2023-09-30 was computed across all sites, not stratified by role, when the refreshed snapshot was verified (daily tier: WY2024 4.4%, WY2025 8.9%, WY2026 47%, WY2027 99%); the independent re-review also computed pooled hourly shares on 20 sites (0 / 0 / 28% / 95%). No values, labels or model outputs were inspected.
7. **H2 projection.** The a-priori H2 Test-A projection (≈ 2 chronic onsets, upper scenario ≈ 7; protocol §9) uses train/val-role onset rates and per-role eligible-site counts only; `pairs.parquet` contains post-2021 WQP chloride pairs (incl. Test-A dates) and `regressions.parquet` lists `n_pairs_all`; neither enters the fit or the scoring.

## Part 3 — Variables
Hazard definitions, label validity rules and features: protocol §2–§4 (exactly as `sentinel/eval/hazards.py`, `sentinel/models/sonde/features.py`, `subdaily.py`). Predictors are available at issue time only. Labels use Approved values only.

## Part 4 — Analysis plan
- **Primary population:** test-role sites × Test-A.
- **Statistic:** pooled BSS at h = 3 d vs the climatology recalibrated by the same frozen Platt step.
- **Uncertainty:** basin-group > site-month two-stage bootstrap (B = 2,000); one-sided p-value.
- **Multiplicity:** Holm over {H1, H2, H4}.
- **Power rule:** a hazard with < 50 evaluable onsets (not censored, onset in (first issue, last issue + 3 d], ≥ 1 usable issue row in the 3 days before) or < 2 basin groups with such onsets at test-role sites is "underpowered": it enters Holm with p = 1 (still m = 3) and cannot be claimed; its bootstrap p is descriptive. The same onset set defines POD, placebo and all controls. The number of basin groups with onsets is reported next to every p (CI approximate with < 10 groups).
- **Selected methods (fixed):** H1, H2 and H4 use the logistic stack of the statistical baselines plus the monotone GBM. The method contains no M1 forecasting or foundation-model component.
- **Secondary analyses and controls:** protocol §7–§8 (placebo with cluster bootstrap, random alarms, circular-shift surrogates, label permutation, negative-control sites, covariate shuffle, input-truncation audit, Molchan, reliability, REV, Diebold–Mariano).
- **Pooled secondary H1:** Test-A ∪ Test-B, scored once on Approved data retrieved on or after 2027-06-30, from a new snapshot pinned by a dated record (`FREEZE_ew_test_ab_data.json`; rows ≤ 2023-09-30 must be identical to the Test-A pins; Test-A rows re-derived from the new snapshot, changed label days reported).
- **Single pass:** one frozen invocation per pass, enforced by an atomic fixed-name marker (independent of the freeze record) and a registry row written when scoring starts; every check runs before the marker is claimed.
- **Deviations:** listed in protocol §12. Any later deviation is dated and reported next to the result.

## Part 5 — Other
The live prospective trial is registered separately (amendment, before 2026-12-01). The v1 SENTINEL early-warning claims were retracted (placebo-failing detector) and are not re-scored.
