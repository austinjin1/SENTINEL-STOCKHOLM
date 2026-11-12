# D7 TroutRefuge — pre-registration protocol (coldwater fish occupancy at unsampled reaches)

**Status:** v1.0, written 2026-11-09 **before any D7 code, split or model** (RESEARCH_PLAN §8 D7, §14, §15; lead
direction 2026-11-09). Validation only until a FREEZE record; **test-look write-ups go only to
`downstream_troutrefuge_test_results.md`** — this file is never edited after its FREEZE.
**Code (to be written after this protocol):** `sentinel/models/downstream/troutrefuge.py`,
`sentinel/models/downstream/troutrefuge_guard.py` (D7-only hardened single-look guard),
`scripts/train/downstream_troutrefuge*.py`, `scripts/data/build_troutrefuge_streamtemp.py`.
**Data:** `nars_streams` (NRSA fish counts + NRSA fish autecology `src_temp_nrsa`; public domain), `downstream_catchment`
(StreamCat + COMID links of NRSA sites), new `troutrefuge_streamtemp` (EPA StreamCat modelled stream temperature
MAST/MSST/MWST 2008/2009/2013/2014 for the same COMIDs; public domain), `forcings_daily` (gridMET), `sonde_daily`
(E7 overlay only). **Observed / external covariates only — no SENTINEL upstream output** (cross-fitting rule not
triggered; `crossfit.phase1_no_upstream` applied).

## 1. Question, labels, claim target
Predict **coldwater fish occupancy** at stream reaches without a fish survey: label = 1 if an NRSA fish sample
(first visit; sample types of the NRSA fish protocol) contains ≥ 1 taxon whose NRSA autecology temperature class
is **coldwater (`CD`)** (trait table of the visit's own survey cycle; taxa without a class are ignored); 0 if the
visit's fish sample contains only cool/warm/unclassified taxa. Visits without any collected fish are not in the
NRSA fish-count table and are excluded (disclosed: zero-catch reaches are not labelled).
Secondary label: salmonid (family SALMONIDAE) presence.

## 2. Split (frozen with `--freeze-split` before any fit)
Survey roles as D5 (independent task, separate split file): **train = NRSA 2008-09 + 2013-14, validation =
2018-19, test = 2023-24** (no D7 label of 2023-24 has been read by any workstream; D5's test look scored benthic
condition only). Location clusters = the frozen 1-km `visit_location` map of D5 (reused read-only). Primary
evaluation = **new locations** (cluster with no first visit in any fitting cycle); all first visits secondary.
Test-time fit = train + val cycles (as D5). Bootstrap unit HUC4 (HUC8 secondary).

## 3. Comparators (fixed before any fit; same configuration budget per learned family)
- Simple: prevalence; ecoregion rate; location kNN (15 nearest training reaches in other clusters); logistic on
  geography (lat, lon, elevation, ecoregion).
- **Thermal-threshold incumbent (published model output):** score = −MAST, EPA StreamCat modelled mean August stream
  temperature (Hill et al. 2013, *Freshwater Science* 32:39–55, RF on 2008–2014 temperature records), year matched
  to the visit (2008 → 0809, 2013 → 1314, 2014 → later cycles; disclosed); also reported as the binary rule
  MAST ≤ 20 °C (EPA 910-B-03-002 salmonid criterion).
- **RF occupancy incumbent:** random forest (500 trees, sqrt mtry; Hill-2017-style, as D5) on StreamCat + MAST/MSST/MWST.
- Feature-matched RF on the challenger's features (parity).
- Challengers: LightGBM (3 seeds) and L2 logistic on geography + StreamCat + MAST/MSST/MWST + gridMET summer climate
  of the 3 calendar years before the visit (observed, ends before the visit).
- **Configuration cap: ≤ 40** configurations for D7 in total (all families, all iterations; 8 per learned family in
  phase 1, inner leave-one-ecoregion-out CV on the training cycles, criterion AUROC).

## 4. Endpoints and claim rule (§15.1)
Primary: **AUROC, validation new locations** (claim on test new locations). Paired HUC4 cluster bootstrap. A claim
requires the frozen challenger to beat the RF occupancy incumbent, the thermal-threshold incumbent, the
feature-matched RF and the strongest simple baseline, all with CI > 0, Holm over these four, **and** a per-seed paired
point estimate > 0 for each of 3 seeds. Otherwise "competitive"/"negative". Power check before any fit; a (model,
event) cell with < 30 positives or < 10 HUC4 clusters with positives is **underpowered** (events power rule).

## 5. Harmful-event angle — E7 thermal stress (secondary, exploratory)
Shared `sentinel/eval/events.py` v1.0 (imported, never copied): **E7 = 7DADMax > 20 °C** at **coldwater-eligible**
sonde sites (`events.coldwater_eligible`, train-block rule; series `temp_c_max` → `temp_c_7dadm` via
`events.derive_series`, state via `events.event_state`). Overlay: NRSA fish sites within 5 km (same HUC8) of a
coldwater-eligible USGS sonde site with ≥ 30 observed Jul–Aug days in the visit year: report the association between
the D7 occupancy probability and the observed E7 state-days of that summer (`events.state_metrics`, with the
occupancy probability as the score of "no E7"). Power-flagged; no claim. M6 owns the E7 forecasting track; D7 does
not duplicate it.

## 6. Single-look guard (lead hardening, 2026-11-09)
D7-only module `troutrefuge_guard.py` (the shared downstream `testlook.py` is FREEZE-hashed and is not edited):
pinned data + environment + every frozen artifact verified **before** the claim; claim = O_EXCL marker **and** a
`look_claimed` registry row, refused if any registry row of the look's experiment exists; protected rows loadable only
while an unscored claim of this freeze is open; predictions written once and **sealed** (0444 + `SHA256SUMS`, O_EXCL
marker) before any registry write; **access closed at sealing**; `--rescore-from-saved` scores only the sealed,
sha-checked predictions, and is refused after `SCORED.json`; any exception (`BaseException`, incl. interrupts) before
sealing writes `ABORTED.json` + a registry row (claim closed); the FREEZE records the **full import-closure hash**
(sha256 of every `sentinel/` module imported by the test path, computed at freeze time and re-checked before the claim).

## 7. Iteration, freezing
Validation only; one change at a time; every configuration logged; cap §3. No git (user instruction): ordering
evidence = registry timestamps + prereg snapshot + tar.gz archive of hashed files at freeze.

## 8. Frozen split, counts and power (filled from `--freeze-split` / `--build-only`, before any fit)
- **Amendment (2026-11-09, before any D7 code ran beyond a refused data request):** the project StreamCat client
  excludes MAST/MSST/MWST on purpose (modelled stream temperature overlaps SENTINEL temperature targets;
  `docs/datasheets/streamcat.md`). Therefore: (i) **no D7 challenger or RF incumbent uses MAST/MSST/MWST**; feature
  sets are geography + curated StreamCat (incl. PRISM 1991–2020 tmax/tmean normals) + gridMET summer climate;
  (ii) the **thermal-threshold incumbent** is MAST (Hill et al. 2013) **only if the lead approves using the excluded
  metric as a comparator score** (never a feature); otherwise (default) it is an observed-input thermal rule:
  predicted August stream temperature from a linear air–water regression on the PRISM tmax normal of the watershed
  (Mohseni-type, fitted on training reaches with NRSA field water temperature measured in Jul–Aug), score =
  −prediction, plus the binary rule ≤ 20 °C. The new data source `troutrefuge_streamtemp` is withdrawn unless approved.
- **Trait harmonisation (2026-11-09, before labelling):** NRSA fish temperature classes (`src_temp_nrsa`) exist only
  in the 2013-14 and 2023-24 fish-taxa tables. Class per taxon = by upper-case taxon name, **2013-14 first, then
  2023-24** (taxonomic metadata, not labels); `CD` = coldwater; hybrids/uncertain codes (`CL?`, `WM X CL`) are not
  coldwater. Taxa unmatched in both tables are unclassified. The share of fish individuals with a class is reported
  per cycle.
- **Split frozen 2026-11-09:** `downstream_troutrefuge_split.json`, **sha256 `587caeca6c634e944d8268b001becbcca8ba650d3244f68f4c2bdad3e6f84a3c`**
  (visit roles by survey; D5 1-km location clusters, else own site id). Disclosure: building the unit table reads
  all cycles' fish counts into memory (as D5's `load_units`); test-cycle labels are not stored in the split, are
  dropped (`common.drop_protected`) before any count, and no test-cycle number was printed.
- **Counts / power (`--build-only`, train + val only, no fit):**
  | set | n | coldwater-occupied (y_cold) | prevalence | HUC4 with positives | paired MDE | salmonid positives |
  |---|---|---|---|---|---|---|
  | train (0809 + 1314) | 3,813 | 910 | 0.239 | 114 | 0.023 | 841 |
  | val all (1819) | 1,594 | 379 | 0.238 | 89 | 0.036 | 359 |
  | **val new locations** | 806 | 201 | 0.249 | 70 | 0.050 | 188 |
  Median share of classified fish per visit 1.0. Not underpowered (≥ 30 positives, ≥ 10 HUC4).
- **Lead decision (2026-11-09, recorded before the fetch):** MAST (Hill et al. 2013) is approved **as a comparator
  score only** (the published thermal incumbent; any overlap of its training data with our test favours the
  incumbent → conservative). Conditions: dedicated fetch `scripts/data/fetch_mast_d7.py` into its own dataset
  `d7_mast` (manifest, datasheet, content pin in the D7 data pins); the shared StreamCat client is not modified;
  MAST/MSST/MWST can never enter any D7 challenger feature matrix (enforced in code, unit-tested); no other
  workstream may read `d7_mast`. **Both thermal incumbents are kept** — MAST rule (score −MAST, plus binary
  MAST ≤ 20 °C) and the observed air–water rule — and the claim must beat both (Holm family grows to five:
  RF occupancy incumbent, MAST thermal incumbent, air–water thermal incumbent, feature-matched RF, strongest simple).
- **Phase-1 budget and comparator details (pre-registered 2026-11-10, before any D7 fit):** learned families tuned by
  inner leave-one-ecoregion-out CV on the training cycles (criterion AUROC), 8 configurations each: RF occupancy
  incumbent (StreamCat only; max_features × min_samples_leaf grid), feature-matched RF (challenger features), LightGBM
  challenger (num_leaves × min_child × colsample grid; 3 seeds averaged), L2 logistic challenger (8 C values); the
  air–water rule is one fitted configuration → 33 of the 40-configuration cap. Air–water rule (thermal incumbent 2):
  ordinary least squares of NRSA field water temperature on the watershed PRISM 1991–2020 tmax normal (`tmax9120ws`),
  fitted on training-cycle first visits sampled 1 Jul – 31 Aug; score = −prediction; binary rule prediction ≤ 20 °C.
  Challenger features: lat, lon, elevation, ecoregion one-hot, curated StreamCat (incl. PRISM normals), gridMET mean
  June–August Tmax and precipitation of the three calendar years before the visit (nearest cell ≤ 30 km); visit
  day-of-year is **not** a feature for occupancy (it is a sampling-design variable).
- **Numeric track (user requirement 2026-11-10; secondary, descriptive, no claim):** target = **NRSA field water
  temperature (°C) at first visits sampled 1 Jul – 31 Aug** (instantaneous measurement; noisy). Model: quantile
  LightGBM (q 0.05 / 0.5 / 0.95; fixed num_leaves 15, min_child 20, lr 0.03, 400 trees) on the challenger features plus
  visit day-of-year and gridMET mean Tmax of the 7 days ending the day before the visit. Comparators: MAST (Hill et al.
  2013, year-matched, the published August-mean model — a different target definition, disclosed), the air–water rule,
  the training ecoregion mean. Metrics: RMSE, R², 90 % PI coverage and mean width on validation new locations (HUC4
  bootstrap); scored once at the D7 test look as a secondary descriptive result. 1 configuration.
- **Primary challenger (pre-registered 2026-11-10, before any fit):** the challenger family (LightGBM or L2 logistic)
  with the higher **inner-CV AUROC on the training cycles** is the primary challenger (validation is never used for this
  choice); the other is reported as secondary. Strongest simple baseline = best of {ecoregion rate, location kNN,
  logistic-geo} on validation new locations (prevalence is not a candidate). MAST and air–water comparisons use the rows
  where the comparator is defined (MAST covers ≈ 90 % of units).
- **Phase-1 validation result (2026-11-10; run `downstream/troutrefuge/phase1`, 33 configurations):** primary
  challenger by inner CV = LightGBM (inner LOEO AUROC 0.913 vs logistic 0.909; RF incumbent 0.894, feature-matched RF
  0.903). **Validation new locations (n 806, 201 positives, 172 HUC4):** LightGBM 0.9515 [0.930, 0.969]; RF occupancy
  incumbent 0.9441; feature-matched RF 0.9475; logistic challenger 0.9410; location kNN 0.9006 (strongest simple);
  logistic-geo 0.888; ecoregion rate 0.866; MAST rule 0.883 (n 746); air–water rule 0.850. Paired Δ AUROC of the
  primary challenger: vs RF incumbent **+0.0074 [−0.0027, +0.0188]** (p 0.16, not significant; 3/3 seeds > 0);
  vs feature-matched RF +0.0040 [−0.0056, +0.0148]; vs MAST rule +0.066 [+0.035, +0.099]; vs air–water rule +0.102
  [+0.063, +0.146]; vs location kNN +0.051 [+0.032, +0.075] (last three Holm-rejected). All first visits (n 1,594):
  vs RF incumbent +0.0067 [+0.0004, +0.0133], p 0.038, **not** Holm-rejected. **Reading: beats both pre-registered
  thermal rules and the simple baselines; ties the RF occupancy incumbent (Δ far below the 0.050 paired MDE). Claim
  rule not met on validation → "competitive" as it stands.** Numeric track (Jul–Aug field temperature, val new
  locations, n 434): quantile LightGBM median RMSE 2.96 °C, R² 0.690 [0.606, 0.756]; air–water rule RMSE 4.00, R² 0.427;
  ecoregion mean RMSE 3.99, R² 0.435; MAST RMSE 10.26, R² −2.98 (see correction below); 90 % PI coverage **0.797**
  [0.753, 0.839] (under-covers), mean width 8.69 °C.
- **Comparator correction and amendment (2026-11-10 17:35 UTC — after the phase-1 validation result above, before any
  phase-1b computation, before any test access; the only MSST numbers seen so far are label-free column summaries):**
  (i) **Naming error corrected.** In Hill et al. 2013 / StreamCat, **MAST = mean *annual* stream temperature**, MSST =
  mean *summer* (July–August) and MWST = mean winter stream temperature. §3 and the datasheet called MAST a "mean
  August" temperature — wrong (my error). Consequences seen in phase 1: the binary rule MAST ≤ 20 °C is nearly vacuous
  (validation sensitivity 0.905, specificity 0.007; median MAST 11.8 °C), and MAST is ≈ 10 °C below Jul–Aug field
  temperature (hence R² −2.98). The pre-registered MAST rule is therefore a **weak form** of the published thermal
  incumbent; its ranking score (−MAST) is still a legitimate comparator and stays.
  (ii) **Thermal incumbent 3 added: `msst_rule`** = −MSST (modelled mean summer stream temperature, same year map as
  MAST), plus the binary rule MSST ≤ 20 °C — the form that matches the 20 °C salmonid summer criterion. Same restricted
  dataset `d7_mast` (fetched under the 2026-11-09 decision; comparator score only, never a feature — the guard regex
  already covers `msst`). Scored by `--phase1b`; no learned family is re-tuned (the frozen phase-1 configurations and
  saved validation predictions are reused; the LightGBM challenger is refitted at its selected configuration only to
  obtain per-seed predictions).
  (iii) **Claim rule now requires beating six comparators** (Holm over six): RF occupancy incumbent, MAST rule, MSST
  rule, air–water rule, feature-matched RF, strongest simple; each CI > 0 and 3/3 seeds > 0. This change was made
  after seeing phase-1 validation results; it can only make the claim harder.
  (iv) **Numeric track:** MSST added as the like-for-like published comparator (modelled Jul–Aug mean vs Jul–Aug field
  temperature). The MAST numeric row stays, labelled "annual mean — level not comparable".
  (v) **Prediction-interval calibration (1 configuration):** conformalised quantile regression (Romano, Patterson &
  Candès 2019). Conformity score E = max(lo − y, y − hi) with lo = min(q05, q50), hi = max(q95, q50), from **inner
  leave-one-ecoregion-out out-of-fold** predictions on the training cycles (same fixed quantile-LightGBM settings);
  calibrated interval = [lo − Q, hi + Q], Q = the ⌈(n + 1)·0.9⌉ / n empirical quantile of E. Validation is not used
  to calibrate. Uncalibrated coverage stays reported next to it.
  (vi) **Configuration count:** 33 + 1 (MSST rule) + 1 (CQR) = **35 of 40**.
  (vii) **Lead confirmation requested:** the 2026-11-09 approval names MAST; MSST is used here under the same
  conditions (comparator score only). If the lead declines, the `msst_rule` rows are withdrawn and reported as withdrawn.
- **Phase-1b validation result (2026-11-10; run `downstream/troutrefuge/phase1b`):** LightGBM refit reproduces the saved
  phase-1 predictions exactly (max abs diff 0.0). **MSST rule AUROC 0.9219 [0.896, 0.947]** on validation new locations
  (n 780 with MSST, 198 positives) — far stronger than the MAST rule (0.883) and the air–water rule (0.850); binary
  MSST ≤ 20 °C: sensitivity 0.864, specificity 0.783 (n 1,519). Primary challenger vs MSST rule: **+0.0295
  [+0.0123, +0.0469]**, p < 0.001, 3/3 seeds > 0 (all first visits +0.0487 [+0.034, +0.064]). **Holm over six
  (validation new locations): rejected for MAST, MSST, air–water and location kNN; not rejected for the RF occupancy
  incumbent (p 0.16) or the feature-matched RF (p 0.39).** Reading unchanged: D7 beats every thermal incumbent and
  simple baseline, ties the RF occupancy incumbent → "competitive" on validation; not a "beats incumbent" result.
  Numeric: MSST RMSE 3.23 °C, R² 0.618 (n 419) vs quantile LightGBM 2.96 °C, R² 0.690; paired RMSE difference on MSST
  rows −0.38 °C [−0.54, −0.22] (all first visits −0.54 [−0.66, −0.42]). CQR margin 1.62 °C (n_cal 2,019; OOF
  uncalibrated coverage 0.772): 90 % PI coverage 0.797 → **0.935 [0.905, 0.962]**, mean width 8.69 → 11.94 °C (all first
  visits 0.946 [0.928, 0.962]) — now slightly over-covering (leave-one-ecoregion-out calibration is harder than the
  new-location task). **Count correction:** the configuration log holds **36 of 40** (32 grid + air–water + numeric
  quantile model + MSST rule + CQR); item (vi) above said 35 because it omitted the numeric quantile model.
- **Lead decisions (2026-11-10 14:25 EST = 19:25 UTC; recorded 19:40 UTC, before any further D7 computation):**
  (a) **MSST is approved as an added comparator, score only**, under the same `d7_mast` restrictions as MAST (never a
  feature; shared StreamCat client not modified; no other workstream reads `d7_mast`). This answers item (vii) of the
  17:35 UTC amendment: the `msst_rule` rows stay. (b) **MAST stays**, with the corrected description (mean *annual*
  stream temperature); the "mean August" wording in §3 is superseded by item (i) of that amendment. (c) **The claim
  family is six, Holm-corrected, and the challenger must beat every listed comparator** (RF occupancy incumbent, MAST
  rule, MSST rule, air–water rule, feature-matched RF, strongest simple). (d) **No freeze candidate is built yet.** A
  pre-registered **ecoregion-transfer track** is being designed; both tracks are to be frozen together and looked at
  in one event. Counts-only design note (no model, no score, no test value):
  `docs/preregistration/downstream_troutrefuge_ecoregion_transfer_design.md`. It is a proposal until the lead approves
  it; on approval its design is copied here as a dated amendment before the first transfer fit.
