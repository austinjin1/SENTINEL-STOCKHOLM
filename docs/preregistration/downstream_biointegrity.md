# D5 BioIntegrity — pre-registration protocol (NRSA benthic condition at unsampled reaches)

**Status:** v1.0, written 2026-10-24 **before any D5 model was fitted**. Phase 1 = development CV and
validation (NRSA 2018-19) only; the test cycle (NRSA 2023-24) is not scored until the configuration is
frozen in `FREEZE_downstream_biointegrity.json` (scored once). Implements RESEARCH_PLAN §8 D5, §14, §15
(H-G4); docs/research/07 §d.5. Author: Bryan Cheng (AI-assisted).

## 1. Question and endpoints
Can benthic-macroinvertebrate condition (NRSA `BENT_MMI_COND`) be predicted at reaches **without a
biological sample** from observed landscape/climate covariates, better than the incumbent
(Hill et al. 2017, random forest on 198 StreamCat predictors, NRSA 2008-09 Good vs Poor: OOB accuracy
75%, sensitivity 73%, specificity 77%, OOB AUC 0.84; random 71-site holdout AUC 0.81 — all random,
not blocked, designs; docs/research/04 B1)?

- **Primary endpoint:** AUROC for Poor (1) vs Good (0) (Fair excluded) on **validation = NRSA 2018-19
  first visits at new locations** (location cluster with no 2008-09/2013-14 first visit), model trained on
  NRSA 2008-09 + 2013-14. Paired cluster-bootstrap difference vs the incumbent re-run under the identical
  split (`hill2017_rf_streamcat`). Same endpoint on the test cycle after freezing.
- **Secondary:** accuracy at p = 0.5 (Hill's metric), AUPRC (prevalence shown), Brier; all 2018-19 first
  visits; revisited locations (with the persistence baseline); 3-class ordinal (later phase); fish MMI
  (later phase); design-weighted accuracy (later phase).
- **Development CV (spatial):** leave-one-aggregated-ecoregion-out (9 folds) on the training cycles;
  pooled AUROC. Reported, not used to select between model families.

## 2. Units, labels and split (frozen)
- Unit: NRSA site visit with `visit_no == '1'` and a Good/Fair/Poor benthic MMI class
  (`processed/nars_streams/condition_wide.parquet`). Within-cycle revisits are excluded.
- Location clusters: 1.0-km connected components of all NRSA site coordinates over the four cycles
  (`sentinel.models.bioassess.data.all_site_locations`, coordinates only).
- Roles by cycle: 2008-09 + 2013-14 → train (3,942 visits; 1,274 Good / 935 Fair / 1,733 Poor);
  2018-19 → val (1,918; 624 / 473 / 821; 956 at new locations); 2023-24 → test (1,944; 606 / 455 / 883).
- **Split file** `docs/preregistration/downstream_biointegrity_split.json`,
  **sha256 `0d9de79a17b33e06e57703e15b2f4309bcabd0dbc55e8e21f3eb2a9558f1601c`** (canonical JSON; visit →
  role, visit → location cluster, val/test new-location flag). Report: `downstream_biointegrity_split_report.json`.
- Test-role rows are dropped (`common.drop_protected`) immediately after role assignment.

## 3. Inputs (observed covariates only; phase 1 uses no upstream-model outputs)
| Set | Content | Status |
|---|---|---|
| `geo` | lat, lon, elevation, ag_eco9 one-hot | deployable |
| `streamcat` | curated StreamCat cat/ws/riparian metrics (land cover 2019, nutrients, infrastructure, lithology, geochemistry, soils, PRISM 1991–2020 climate, IWI/ICI) + NHDPlus VAA; COMID published (2013-14, 2018-19) or NLDI position snap (2008-09, part of 2023-24); `processed/downstream_catchment` | deployable |
| `wx` | gridMET antecedent precipitation/Tmax sums, means and anomalies over 7/30/90 d **ending the day before the visit**; nearest SENTINEL gridMET cell (NaN if > 30 km); climatology from 2000–2014 only | deployable |
| `chem`, `habitat` | NRSA co-measured chemistry (log TP, TN, NH4, NO3, Cl, SO4, conductivity, turbidity, DOC, TSS; pH, ANC) and physical habitat | **oracle** (same visit) — upper bound only |

StreamCat's own predicted condition (`prg_bmmi0809`, Hill 2017 output) is excluded at download.
Caveat: MMI benchmarks use reference sites screened on chemistry/habitat, so oracle rows are partly
circular by construction; they are never the claimed model.

## 4. Models and baseline ladder (fixed a priori; every row reported)
Prevalence (constant) → ecoregion Poor rate (train) → location-only kNN (k = 15 nearest training reaches,
same location cluster excluded) → persistence (previous-cycle class at the same location; revisits only)
→ logistic on `geo` → **incumbent RF on StreamCat** (500 trees, sqrt mtry; Hill 2017 style) → RF on
geo+StreamCat → LightGBM on geo+StreamCat (400 trees, lr 0.03, 15 leaves, colsample 0.5) → +wx →
**spatial-context rows** (`nbr`: Poor share among the 5/15/50 nearest training-cycle reaches in other
location clusters — prior survey labels, available at decision time) for RF (baseline, parity) and GBM,
with and without wx → oracle rows. The primary comparisons (Holm) are `gbm_geo_streamcat_wx_nbr` and
`gbm_geo_streamcat` vs the incumbent. Tree models: 3 seeds averaged; seed spread reported.

**Incumbent reproduction:** RF on StreamCat, NRSA 2008-09 only, out-of-bag accuracy (Hill's random/OOB
design) compared with the published 75% (and OOB AUC 0.84 quoted in docs/research/04). Tolerance ±0.05;
outside it, diagnose before claiming anything.

**Tuning parity (improvement cycles, RESEARCH_PLAN §15.5 amendment):** any hyperparameter search uses
inner leave-one-ecoregion-out CV on the training cycles only, with the same budget for the incumbent RF
and for each challenger; the number of configurations tried is logged (`configs_tried.jsonl`).

## 5. Statistics
Cluster bootstrap (B = 2,000) over **HUC4 basins** (resampling unit; HUC8 secondary — see §8) for every CI; paired bootstrap for
model − incumbent differences; Holm across the two primary comparisons at the test look. Claim labels per RESEARCH_PLAN §15.1.

## 6. Test-time definitions (pre-registered before any fit; review W7)
- The frozen test model is **re-fitted on 2008-09 + 2013-14 + 2018-19** with the frozen configuration.
- Test "new location" = location cluster with no first visit in any of those three cycles, derived
  deterministically from the frozen `visit_location` map (the frozen `val_test_new_location` flag for
  test rows is defined against the training cycles only and is reported as a secondary subset).
- The `nbr` pool at test time = labels of all three fitting cycles (other location clusters only).

## 6a. Relation to M5 BioAssess and the cross-fitting rule
- D5 **imports** (never copies or edits) `sentinel.models.bioassess.data.all_site_locations` for the 1-km
  location clusters; that file is in D5's scoring-code hash (`common.scoring_files("biointegrity")`), so any
  change to it invalidates a D5 freeze.
- D5 uses NRSA 2018-19 as validation and 2023-24 as test **for its own task** (condition from landscape
  covariates). Phase 1 uses no M5 output.
- **Cross-fitting rule (RESEARCH_PLAN §8):** if a later D5 version uses any M5 BioAssess prediction or
  embedding as a feature, the M5 model that produces it for 2018-19 rows must have been fitted without any
  2018-19 (and 2023-24) data, and the one used for 2023-24 rows without any 2023-24 data; for training-cycle
  rows the M5 features must be out-of-fold (location-cluster folds). `crossfit.check_crossfit` enforces the
  site/time contract; such a version needs a new protocol version and a new frozen split file.

## 7. Disclosures (R6)
- Writing the split report printed the test-cycle class counts (606 Good / 455 Fair / 883 Poor). No model
  or feature choice uses them.

## 8. Deviations log
- 2026-10-24, pre-training review (before any fit):
  - **C1:** StreamCat model-derived columns excluded from every feature set: Doyle et al. 2023 predicted
    bankfull depth/width, wetted width, thalweg depth (random forest possibly trained on NRSA field data)
    and the IWI/ICI composite indices (`biointegrity.EXCLUDED_STREAMCAT`).
  - **W6:** resampling unit for every CI and paired difference changed from HUC8 (≈ 1.4 validation visits
    per HUC8 — nearly row-level) to **HUC4**; HUC8 intervals are still computed and stored (`*_huc8`). A
    claim needs the HUC4 interval to exclude 0.
  - **W8:** the claim family is extended: each primary challenger must beat **both** the incumbent
    (`hill2017_rf_streamcat`) and the feature-matched RF (`rf_geo_streamcat` for `gbm_geo_streamcat`;
    `rf_geo_streamcat_nbr` for `gbm_geo_streamcat_wx_nbr`); Holm over these four comparisons.
  - **W1:** the persistence baseline is scored and registered on revisited locations only.
  - `wx_cell_km` (distance to the gridMET proxy cell) is no longer a feature; `nani`/`nsurp` are divided by
    AOI area like the National Nutrient Inventory totals.
  - The split sha256 is enforced by the driver (`biointegrity.SPLIT_SHA`).

## 9. Results so far and iteration log (validation only; RESEARCH_PLAN §15.5 amendment)
- **phase 1** (run `phase1_20261024-233247`, 28 fits): incumbent reproduced under its own design (RF on
  StreamCat, NRSA 2008-09 OOB accuracy 0.774 vs published 0.75; OOB AUROC 0.849 vs 0.84). On validation
  new locations the GBM on geo+StreamCat beats the incumbent by ΔAUROC +0.014 (HUC4 CI just above 0,
  p = 0.047) but not the feature-matched RF (+0.013, CI includes 0); none of the four pre-registered
  primary comparisons survives Holm. `wx` adds nothing; `nbr` adds ≤ 0.003. Oracle co-measured
  chemistry + habitat: +0.076 (upper bound). Verdict after phase 1: **competitive**.
- **it1_mmi_regression** (one change): learners trained on the continuous benthic MMI score of all
  training first visits (Fair included; more information per visit); Poor score = −predicted MMI. Parity:
  RF regression on identical features; the incumbent row is unchanged (published method).
  **Result (run `it1_mmi_regression_*`): rejected.** New-location AUROC GBM-regression 0.807 vs
  GBM-classification 0.858 (Δ −0.037 vs incumbent, CI below 0); RF-regression 0.790. Likely cause: MMI
  condition benchmarks are ecoregion-specific, so the raw score ranks visits across regions differently
  from the class. Classification stays.
- **it2_tuned** (one change): hyperparameters by inner leave-one-ecoregion-out CV on the
  training cycles, criterion pooled Poor-vs-Good AUROC, 8 configurations for **every** learned family
  including the incumbent RF (tuned incumbent reported next to the published-default incumbent; a claim
  must beat both).
  **it2 result (run `it2_tuned_20261025-012347`; recorded 2026-11-08).** Val new locations, AUROC diffs
  (HUC4 paired CI): tuned GBM − published-default incumbent **+0.014 [0.001, 0.028]** (p 0.038); − tuned
  incumbent **+0.022 [0.007, 0.037]** (p 0.003); − feature-matched RF +0.013 [−0.000, 0.026]; − tuned
  feature-matched RF +0.012 [−0.000, 0.024]. All val visits: tuned GBM beats every RF row (e.g. − tuned RF
  +0.009 [0.002, 0.016]). The tuned incumbent is *weaker* than its published default on new locations
  (inner-LOEO tuning did not transfer). Verdict: beats the incumbent on validation (both variants, CI > 0);
  vs the feature-matched RF still within CI → competitive under the protocol's strict reading.
- **it3_ens** (pre-registered 2026-11-08, before computing): one change — challenger = equal-weight mean of
  the predicted Poor probabilities of `tuned_gbm_geo_streamcat` and `tuned_rf_geo_streamcat` (both from
  it2; **no new fit, no weight tuning**). Same comparators, subsets and HUC4/HUC8 paired bootstrap as it2.
  Keep if it beats all four RF rows on new locations with CI excluding 0; otherwise the it2 tuned GBM
  stays the freeze candidate.
  **it3_ens result (2026-11-08, `biointegrity/it3_ens`): kept.** New-location AUROC 0.858 (= tuned GBM
  0.858; ensemble − tuned GBM −0.001 [−0.006, 0.005]) with lower variance: ensemble − published incumbent
  **+0.014 [0.005, 0.023]** (HUC8 [0.006, 0.022], p 0.003); − tuned incumbent +0.021 [0.010, 0.033];
  − feature-matched RF **+0.013 [0.004, 0.021]** (p 0.005); − tuned feature-matched RF **+0.011
  [0.004, 0.019]** (p 0.003); Holm over the four RF comparisons rejects all. All visits: same pattern.
  **Freeze candidate (D5): `ens_tuned_gbm_rf`** (it2 tuned GBM + it2 tuned RF on geo+StreamCat, equal
  weights), refit on 2008-09 + 2013-14 + 2018-19 per §6. Selection disclosure: 4 iterations (phase 1, it1,
  it2, it3) on the same validation cycle; configurations logged in `configs_tried.jsonl` (count reported
  with any claim). Freezing awaits the lead's go-ahead.

## 10. Pre-test gate deviations (2026-11-08, before any test look)
- **Shared (all five candidates), pre-test gate 2026-11-08 (lead decisions; written before any test look):**
  (a) **seed rule**: a claim additionally requires the paired gain to be > 0 in each of ≥ 3 seeds (per-seed
  point estimates on the test block, vs every primary comparator); per-seed predictions are saved;
  deterministic challengers (L2 logistic) meet it trivially. (b) every input data file is sha256-pinned in
  `downstream_<app>_data_pins.json` and verified before the claim. (c) predictions (incl. per-seed) are saved
  once before any registry write; after a crash only re-scoring of the saved predictions is allowed
  (`--rescore-from-saved`), never re-fitting. (d) **No git (user instruction):** pre-registration order
  cannot be proven from version control; evidence = registry timestamps, file mtimes and a
  `prereg_snapshot` registry row (sha256 + mtime of protocol, record and test script) appended when the
  FREEZE file is written.
- **Claim family = all four RF rows** (published and tuned Hill-2017 incumbent, default and tuned
  feature-matched RF), Holm over four on HUC4 — as pre-registered in §8 W8 and the it3 keep rule; the
  record's earlier two-comparison family was a weakening and is withdrawn.
- **Secondary results (reported, outside the Holm family):** accuracy at p = 0.5 (Hill's metric); all 2023-24
  first visits; revisited locations with the persistence baseline; the frozen `val_test_new_location`
  subset; logistic on geo. Predictions saved for every 2023-24 unit.
- **Disclosures:** it3 was registered after the it2 validation results existed; the "before computing"
  statement cannot be proven from files: the it3 pre-registration text was appended in the same shell
  command immediately before the ens script was created and run (ens script 12:40:27, registry 12:40:31);
  the protocol's mtime (12:41:34) reflects the later append of the it3 result, which overwrote the earlier
  mtime — without git the order rests on this account and the registry timestamps. Validation CIs are selection-biased (4 iterations,
  92 configurations on one cycle). The tuned incumbent's selection (max_features 0.5) is at its grid edge;
  the published-default incumbent (stronger on validation) is in the family. Tuning parity is per learner
  family (8 configs each); the challenger combines two tuned families + an equal-weight rule.

## 11. Single test look — result (2026-11-08 20:25–20:32; FREEZE_downstream_biointegrity.json; registry `biointegrity/test_look`)
NRSA 2023-24 first visits at new locations (n = 979, 600 Poor). AUROC: ensemble 0.861 [0.836, 0.884]; Hill-2017
RF (published) 0.849, tuned 0.848; feature-matched RF 0.853, tuned 0.852; tuned GBM 0.860; location kNN 0.727.
Paired (HUC4): ensemble − Hill published **+0.012 [0.003, 0.021]** (p 0.005), − Hill tuned **+0.013 [0.005,
0.021]** (p 0.002) — Holm-rejected; − feature-matched RF +0.008 [−0.001, 0.017] (p 0.062), − tuned RF +0.009
[0.001, 0.018] (p 0.027) — not Holm-rejected. Seed rule holds (all per-seed Δ > 0). **Pre-registered claim
(all four RF rows) not met → label: competitive.** Against the §8 incumbent (Hill et al. 2017 RF, both
variants) the gain is Holm-significant. Secondary: all first visits ens 0.885 vs Hill 0.875; revisited
locations ens 0.919 vs Hill 0.918 vs persistence 0.880.
H-G4 (post-results review): the pre-registered 2026-10-24 rule governs → D5 does **not** count toward
"beats its incumbent".
