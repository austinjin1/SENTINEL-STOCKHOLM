# M3 architecture round v2 — deviations file 01

**Date:** 2026-11-10 (written 20:23 UTC, after the lead's stamp of 19:41 UTC, registry row `421854a9d225`)
**Author:** M3 (SENTINEL-Microbe)
**Status:** proposed. Not in force until the lead stamps it with a registry row of experiment
`microbe/prereg_snapshot` whose `extra.sha256` maps
`docs/preregistration/microbe_arch_v2_deviations_01.md` to this file's sha256 (rulings L3). Once stamped it is
never edited; a correction is file 02.

**Releases:** A0, XGB_V4CLR_MO, CALIBRATION, NICHE_S_QMP, GEO_POWER

## 1. Why this file exists

The stamped amendment (`microbe_arch_v2_amendment.md`) and the rulings file (`microbe_arch_v2_lead_rulings.md`)
order five things without fixing every detail needed to compute them: ablation A0 and the calibration
diagnostic (ruling A1), the masked 17-dimension `xgb_v4clr_mo` (ruling B5), the +QMP twin of NicheInvert-S
(possible G10 entry, amendment §5 and ruling B8), the QMP confound rule (amendment §5) and the GEO power file
(amendment §7.3, ruling B9). Ruling OQ11 says anything that fixes a design choice needs a stamped deviations
file. The readings below are those choices. They were written before any development number of this round
existed: at the time of writing no arch-v2 development run has started and the registry holds no row of
experiment `microbe_arch_v2_cv`.

The code refuses each affected run until this file is stamped (`sentinel/models/microbe/stamp.py`,
`require_release`; tested in `tests/test_microbe_archv2.py`). Nothing else waits for this file: D1 (`niche_s`),
A1 (`niche_s_tax`), `xgb_nichez_context`, `xgb_v4mr` and its twins, RF[V4-CLR], TabPFNv2[V4-CLR], WA-tol, WA-PLS,
`xgb_clr_stack` and the baselines on the GEO family run under the amendment stamp alone.

No reading changes a hypothesis, a family, a Holm denominator, the cap of 8, a split or a scorer.

## 2. Readings

### R15 — ablation A0, `niche_s_a0` (key `A0`)

Ruling A1: "NicheInvert-S with no context prior, κ = 0 and no deshrinking clip". Everything not listed is D1
unchanged (same vocabulary, sequence prior, intercept, Newton rules, inner folds).

- (a) **No context prior** = a flat prior on z. The context term is dropped, the search starts at the training
  mean (z = 0), and Σ⁻¹ is replaced by 1e-6 · I. That ridge has no statistical role; it only keeps the Newton
  system solvable on directions the reads do not identify. A0 reads no context column.
- (b) **κ = 0**: λ = 1 for every unit with a labelled training event, so its optimum is its own weighted average
  and its tolerance its own corrected tolerance, floored as in D1. A unit with N2 ≤ 1 has no tolerance estimate
  and takes the pooled tolerance. A unit with no labelled event on a dimension keeps reading R2 (prior optimum,
  pooled tolerance).
- (c) **No deshrinking clip**: the deshrinking line is fitted as in reading R10; its slope is not clipped.
- (d) With a flat prior the posterior mode does not depend on N_eff, so the N_eff grid is the single value 100.
  It only scales the reported Laplace SD.

A0 is counted, runs on `lodo` only, is never selectable and carries no claim (rulings §2.1).

### V1 — `xgb_v4clr_mo` (key `XGB_V4CLR_MO`)

Ruling B5: "fitted on all 17 master dimensions with missing labels masked (not mean-filled)".

- One XGBoost `multi_output_tree` model per fold and seed on all 17 master dimensions, whatever target family is
  scored. Families P and G read the same fitted model.
- Inputs and hyperparameters are those of `xgb_v4clr` (V4-CLR cap 500 + community descriptors;
  `baselines.DEFAULTS["xgb"]`, `tree_method="hist"`).
- Each dimension is standardised with the training mean and SD of its labelled rows. `base_score` is 0, which is
  the training mean after standardisation.
- Masking: the squared-error objective is supplied as a custom function whose gradient and Hessian are 0 where
  the label is missing. A missing label is not replaced by any value.
- A training row with no label on any dimension is dropped. A dimension with fewer than 10 labelled training
  rows or zero variance is left out of the model (prediction NaN), as for the baselines.
- `field_temp_c` is a training dimension only; it is never scored (ruling OQ2).

It is a comparator, not counted against the cap (ruling B5).

### C1 — calibration diagnostic (key `CALIBRATION`)

Ruling A1: "within-domain slope and predicted/observed SD ratio per target on D1's development predictions".
Per target, on the `lodo` out-of-fold predictions, over the rows where the observed and the predicted value are
both finite (no statistic if fewer than 10 rows):

- `slope_within_domain`: both variables have their domain mean subtracted (the domain is the held-out fold);
  the statistic is the least-squares slope of observed on predicted, Σ p̃ỹ / Σ p̃². Below 1 means the predictions
  are over-dispersed within domains, above 1 under-dispersed.
- `sd_ratio`: SD(predicted) / SD(observed), pooled over the same rows.
- `sd_ratio_within_domain`: the same ratio after subtracting domain means.
- `slope_pooled`: the slope of observed on predicted without centring by domain (reported, not a gate input).

The diagnostic is run for `niche_s` on both families and, for reference, for A0, WA-tol and `xgb_v4clr`. It is
logged to the registry (metrics `calib_*`), uses no bootstrap and selects nothing. It is not a gate condition;
the lead reads it with the gate.

### R14 — `niche_s_qmp`, the +QMP twin of NicheInvert-S (key `NICHE_S_QMP`)

Fitted only if G1 is the entry that the mechanical G10 rule picks (ruling B8). The 4 QMP features, filled with
the training median, and their 4 missing flags are appended to the tier-0 context columns of the context-prior
ridge. Nothing else changes; the QMP block never enters the read likelihood.

### R17 — the QMP confound rule (no key: the check is label-free and runs under the amendment stamp)

Amendment §5: "If laboratory alone explains more than 50% of the variance of the load index, +QMP results are
descriptive only."

- Share of variance = one-way between-group sum of squares / total sum of squares, over the events that have the
  feature and the grouping. It is computed for each of the 4 QMP features, by laboratory and by batch.
- Load index = the feature `qmp_load`.
- The rule fires if the laboratory share of `qmp_load` exceeds 0.5 on the development events **or** on the
  events of all four blocks pooled. Either one is enough.
- The script (`scripts/eval/microbe_qmp_confound.py`) opens the two QMP tables and the id lists only. No target
  table is opened. It is run once, before any +QMP fit, and its output is reported with every +QMP result.

This reading decides only the wording "descriptive only". It does not decide whether a +QMP model runs.

### P1 — the GEO power file (key `GEO_POWER`)

Amendment §7.3 asks for `docs/preregistration/microbe_geo_power.json` "from the comparators' development
out-of-fold predictions (method of `microbe_split_lodo_x001_report.json`)", with a target labelled
"underpowered" when its minimum detectable ΔR² exceeds 0.30. Ruling B9 says the label uses the holdout-scaled
figure at the family's first Holm step (0.05/8 for H-GEO).

- **Pairs.** Every finished entry among G1, G3, G7, G8, G9, each against `rf_clr` (the comparator of the
  reference method) and against `tier0_best`, on the family-G `lodo` out-of-fold predictions. Stochastic models
  use seed 0.
- **SE.** Paired site-cluster bootstrap SE of ΔR² per target, B = 1,000, bootstrap seed 0, as in the reference
  report. The file stores every pair's SE. The power figure of a target uses the **largest** SE over its pairs.
- **Development MDD.** (z₁₋α/₂ + z₀.₈) · SE with α = 0.05/8. Stored for reference.
- **Holdout scaling.** SE_holdout = SE · √(n_dev,t / 214), where n_dev,t is the number of development rows scored
  for target t and 214 is the holdout event count. This is the upper end of the amendment's range ("if rows
  dominate"). The event count is used for every target because per-target counts of protected blocks are not
  computed (ruling I-04); it understates the factor by at most about 1%.
- **Label.** A target is "underpowered" if its holdout-scaled MDD exceeds 0.30. The target stays in the family
  and the Holm denominator stays 8.
- The file is written once, after G1, G3, G7, G8 and G9 are finished and before G10 is chosen, and is pinned in
  the freeze candidate. G5 and G10 do not enter it.

## 3. State when this file was written

- No arch-v2 development number exists. The affected runs (`niche_s_a0`, `xgb_v4clr_mo`, the calibration
  diagnostic, `niche_s_qmp`, the GEO power file) have not run.
- The readings R15, V1, C1 and R14 are implemented and unit-tested on synthetic data only
  (`tests/test_microbe_archv2.py`). The GEO power script is not written yet; it will call
  `require_release(…, "GEO_POWER")` and get a refusal test.
- The D1 gate inputs that wait for this file are A0 and the calibration diagnostic (ruling B6).
