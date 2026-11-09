# M3 SENTINEL-Microbe — secondary event track (B output): pre-registration

**Status:** pre-registered 2026-11-09, **before any event-level analysis of any M3 model** and before the M3 freeze.
No event metric, probability or recalibration has been computed. Follows `docs/plans/event_layer_plan.md` (§1.1
definitions, §1.2 metrics, §2.2 routes, §2.3 comparators, §4 file rules) and the lead's direction of 2026-11-09.
Parent protocol: `docs/preregistration/microbe_protocol.md`. Config: `configs/events/microbe.yaml`.
**Role: secondary.** The M3 primary endpoint (H-M3, water quality) is unchanged.

## 1. Framing
A 16S community integrates weeks to months of environmental history. The event output is therefore **chronic
state classification at the sample date**: "is this water nutrient-enriched / in a bloom state / ionically
stressed when sampled?" It is **not onset detection, not lead time, and not early warning.** Every event is a
criterion-anchored excursion (plan §1.1), not a regulatory finding.

## 2. Events, and power status (development counts; development labels only)
Counts use only `development_ids` with collect date < 2024-01-01 (plan "test-label hygiene"). Power rule (plan
§1.2): inferential only if ≥ 30 positives **and** ≥ 10 clusters (NEON domains = the M3 resampling unit) carry
positives.

| Code | Definition (target, sample-date value) | n | Positives | Prevalence | Domains with positives | Sites with positives | Status |
|---|---|---|---|---|---|---|---|
| **E5b** | TP > 0.1 mg/L (lab grab) | 1,183 | 226 | 0.191 | 19 | 33 | **inferential (secondary family)** |
| **E4a** | chl-a > 20 µg/L (sonde daily mean, µg/L) | 394 | 35 | 0.089 | 14 | 16 | **inferential (secondary family)** |
| E4a_t12 | chl-a > 12 µg/L (WHO alert 1; sensitivity) | 394 | 66 | 0.168 | 16 | 18 | descriptive sensitivity tier |
| **E2c** | SC > 300 µS/cm (sonde daily mean) | 730 | 268 | 0.367 | 10 | 12 | **borderline** (see below) |
| E1 | DO (daily mean) < 3 mg/L; also < 5 | 683 | 20 / 48 | 0.029 / 0.070 | 5 / 7 | 5 / 8 | descriptive only |
| E3 | turbidity > 50 FNU (FNU vs NTU noted) | 655 | 25 | 0.038 | 8 | 9 | descriptive only |
| E5a | NO₃+NO₂-N > 10 mg/L | 1,180 | 0 | 0 | 0 | 0 | n/a |

- **E2c rule, fixed now.** E2c is inferential only if the development OOF rows on which every compared method
  has a prediction contain ≥ 30 positives in ≥ 10 domains. The current count is exactly 10, so the rule decides.
  Otherwise E2c is descriptive. Either way it is secondary: the 300 µS/cm benchmark comes from Central
  Appalachia (EPA/600/R-10/023F) and is not national.
- The E4a direct head is fitted only if ≥ 5 training positives exist in every LODO training fold; otherwise
  E4a is derived-only.
- Events are computed from the raw (untransformed) values. Thresholds are applied in the model's log space as
  log10(threshold), using the same floors as `design.TARGETS`.

## 3. Model under evaluation
The frozen M3 model M\* chosen by `microbe_protocol.md` §6/§10 (selection rule applied to the full candidate set
at freeze). Its development-LODO OOF predictions are used for every event. The track (primary `lodo` or enlarged
`lodo_x001`) is the one M\* is frozen on. The event layer adds **no** new WQ model and changes no M3 code.

## 4. Routes (plan §2.2)

### 4.1 Derived — primary
- **Distribution.** p_raw = P(y > thr) under Gaussian(μ, σ) on the model's target scale: log10 for every event
  target except DO, which is modelled in mg/L. μ is the OOF prediction. σ is the residual SD of that target.
- **σ for domain g** is computed from the OOF residuals of the other domains (leave-one-domain-out residual
  SD), so no row's own residual sets its σ.
- **Recalibration:** a single Platt recalibration (logistic on logit p_raw), cross-fitted the same way. For
  development reporting, the Platt map applied to domain g is fitted on the OOF rows of domains ≠ g.
- **Frozen for the test:** σ and the Platt map are fitted once on all development OOF rows (the M3
  validation set) and frozen.
- The reliability table and ECE are reported before and after recalibration.
- Implemented through `sentinel.eval.events.exceedance_probability` / `recalibrate` once the shared module
  exists. Until then nothing is run.

### 4.2 Direct head — challenger, powered events only (E5b, E4a, and E2c if powered)
- **Model:** a binary classifier on exactly the inputs of M\*. If M\* is an XGB model, it uses M\*'s
  hyperparameters with objective `binary:logistic`. If M\* is a Ridge or fusion model, it is an L2-logistic
  regression with C chosen by site-grouped inner CV, on the training rows of each LODO fold.
- **Tuning:** no extra budget beyond the parity rule (§15.5 item 4).
- **Recalibration:** cross-fitted Platt, as in the derived route.
- **Seeds:** the mean of seeds 0–2.

### 4.3 Route comparison (pre-registered)
- Paired ΔBrier and ΔAUPRC (derived − direct), cluster bootstrap B = 2,000, with domain-unit and site-unit
  CIs, Holm across the inferential events.
- **Primary = derived**, whatever the comparison shows. The direct head is reported as a challenger.

## 5. Comparators (plan §2.3)
1. **Threshold-on-baseline, derived the same way:**
   - (a) the H-M3 tier-0 reference `tier0_best` (season + location under domain hold-out);
   - (b) RF[genus-CLR], the plan's named M3 comparator;
   - (c) the strongest community baseline per target used for H-M3 (stronger of tuned/untuned RF/XGB).
   Each uses its own LODO residual SD and cross-fitted Platt map.
2. **Event climatology (season + location):**
   - L2-logistic on DOY harmonics, latitude, longitude, elevation and aquatic type, fitted on the training rows
     of each fold;
   - the constant training-fold prevalence.

   Site × DOY climatology is undefined for a held-out domain, so it is not used. Persistence does not apply:
   the modality is snapshot sampling with no prior state at prediction time.

## 6. Metrics (plan §1.2; development LODO OOF now, test later)
- **Per event:**
  - AUPRC, with prevalence as the chance level;
  - Brier score;
  - BSS vs the event climatology;
  - ECE and the reliability table;
  - n, positives, and domains with positives.
- **Paired comparisons:** M\* (derived) vs each comparator in §5, as ΔBrier and ΔAUPRC with domain- and
  site-cluster bootstrap CIs (B = 2,000).
- **Holm:** across the inferential events (E5b, E4a, plus E2c if powered) within each comparator. This is a
  secondary family, separate from the 16 H-M3 hypotheses.
- **Claim wording:** M\* "classifies <event> state better than <comparator>" only if the paired ΔBrier < 0 with
  both units' CIs excluding 0 after Holm. AUPRC is supporting evidence.
- **Descriptive events:** E1, E3 and E4a_t12 get point metrics and CIs only, with no comparison claims.
- Every number is logged through `sentinel.core.registry.log_result`, workstream `M3-microbe`, experiment
  `microbe_events_dev` (development) or `microbe_events_test` (test).

## 7. Test (not run in this phase)
- **Primary holdout (2024).** Event metrics are scored in the **same single pass** as the H-M3 endpoint after
  the freeze, with the frozen σ and Platt maps.
- **Test-B (2025, `microbe_protocol.md` §10.5).** Scored the same way when it exists.
- **No holdout label counts** are inspected before scoring. Plan hygiene: test power is not pre-checked from
  test labels; the development power above is stated with the result.
- **Freeze record.** The event track gets its own freeze record `FREEZE_events_microbe.json`. It pins the M3
  freeze record by sha256 and hashes:
  - `configs/events/microbe.yaml`;
  - `sentinel/events/microbe.py`;
  - `scripts/eval/events_microbe.py`;
  - `sentinel/eval/events.py`;
  - this file.

## 8. Files
New paths only (plan §4.1):
- `sentinel/events/microbe.py` — adapter: OOF → event labels, routes, comparators;
- `scripts/eval/events_microbe.py` — development run;
- `configs/events/microbe.yaml`.

No M3 model file and no frozen file is edited.

## 9. Amendments
- 2026-11-09, after a code smoke test (no logging; B = 50) and before any logged event result. The smoke test
  showed that near-constant predictors (GlobalMean as tier-0 for TP and turbidity) give degenerate
  cross-fitted Platt maps: logistic separation on per-domain σ differences makes the comparator's Brier
  explode. That would weaken comparators artificially.
  - **Rule (all methods identically):** if a fold's Platt fit has |slope| > 20 or non-finite parameters, that
    fold's rows keep the raw derived probability.
  - The frozen test maps follow the same rule.
  - The change can only strengthen comparators.
  - The smoke run's numbers are not reported.
