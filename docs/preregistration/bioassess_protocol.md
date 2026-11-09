# M5 SENTINEL-BioAssess — pre-registration protocol (community → water chemistry)

**Status:** v1.0, written 2026-10-22 **before any SENTINEL-BioAssess model was trained** (phase 1).
Phase 1 = implementation + cross-validation on development data only. The external tests (§7)
are **not scored in phase 1**; they are scored once, after the configuration is frozen (§9).
**Authors:** Bryan Cheng (AI-assisted). **Implements:** RESEARCH_PLAN.md §5.5, §14, §15.2 (H-G1 M5
primary), docs/research/04 §A.4, §C.3, §D.2, §E.3.

## 1. Question and primary endpoint

Task B2 ("biology → stressor"): predict co-measured water chemistry from the benthic
macroinvertebrate assemblage of the same NRSA site visit.

- **Primary endpoint (H-M5, RESEARCH_PLAN §15.2):** leave-one-ecoregion-out (LOEO) R² of the frozen
  SENTINEL-BioAssess model above the random-forest baseline **run under the identical protocol**,
  for ≥ 3 of the 6 targets, with the paired cluster-bootstrap 95% CI of the R² difference excluding
  0, Holm-corrected across the 6 targets — scored on the external test of §7.1.
- **Within-region skill is co-primary for interpretation.** Pooled LOEO R² mixes between-region
  and within-region skill (research-phase finding: within single held-out ecoregions TP R² ranged
  −0.34 to 0.36). Every table reports, for every model and baseline, the pooled R², the
  per-held-out-ecoregion R² (all 9, plus median/min/max) and the anomaly R² (§5). A pooled gain that
  is not accompanied by a within-region gain will be reported as "between-region skill only".

## 2. Data

| Role | Data | Rows |
|---|---|---|
| **Development** (all fitting, CV, model selection) | NRSA 2008-09 + 2013-14 (`processed/nars_streams`), first visits (`visit_no == "1"`) with a benthic sample and ≥ 1 target | ≈ 3,950 visits |
| **External test A/B** (untouched in phase 1) | NRSA 2018-19 first visits | ≈ 1,900 |
| **External test C** (untouched; replication) | NRSA 2023-24 first visits | ≈ 1,900 |
| **Pretraining pool** | UK EA freshwater invertebrate analyses (`processed/uk_ea_ecology`), all years, sites outside the reserved UK test blocks | ≈ 225k analyses |
| **UK reserved test** (untouched) | EA invertebrate sites in 20% of 50-km blocks (seed 0, `spatial_block_split`) | ≈ 20% of sites |

Prior exposure (disclosed): NRSA 2018-19 was used during research for the fixed, untuned RF
baseline (`rf_blocked_baseline`) and for diagnostics by the data engineer (per-ecoregion R² of that
RF). No SENTINEL-BioAssess configuration, hyperparameter, feature or architectural choice is made
on 2018-19 or 2023-24 data. Phase-1 code loads NRSA through `sentinel.models.bioassess.data.load_nrsa`,
which raises unless only development surveys are requested (unit-tested).

**Location clusters.** NRSA re-samples locations across cycles under new site IDs (no shared IDs;
measured: 933 of 1,919 2018-19 sites lie within 50 m of a 2013-14 site, 460 of a 2008-09 site).
All NRSA sites of all four cycles are linked into **location clusters**: connected components of
the graph "two site coordinates ≤ 1.0 km apart" (coordinates only, no labels). Location clusters are
the unit of every split and of every bootstrap. Within-cycle revisits share `site_id` and therefore
a cluster. Every cluster lies in one aggregated ecoregion (checked).

## 3. Targets (identical to the RF baseline)

`log10 TP` (µg/L, floor 1), `log10 TN` (mg/L, floor 0.01), `log10 conductivity` (µS/cm, floor 1),
`log10 chloride` (mg/L, floor 0.01), `log10 turbidity` (NTU, floor 0.01), lab `pH`. Censored
(non-detect) results — ≤ 11 per analyte in development data — are set to their canonical detection
limit and then floored (documented substitution at modelling time, never at ingest). Missing targets
are masked (multi-target models) or the row is dropped for that target (single-target models).

## 4. Inputs

- **Community (primary feature set):** 300-organism fixed-count subsample (`count_300`) per source
  taxon, mapped through `processed/taxonomy_harmonised` (macroinvertebrates). Tabular models use
  Hellinger-transformed relative abundances of harmonised `mixed_otu` taxa present in ≥ 10 samples
  **of the training fold**; the set model uses every present taxon with its harmonised lineage.
  Sample descriptors allowed with the community: log10 total count_300, log10 number of taxa,
  boatable-vs-wadeable sample type (`BELGB`/`BETB` vs `BERW`/`BERWW`).
- **Geography (controls only):** latitude, longitude, elevation (median-filled within the training
  fold). Used only by the geography-only control and the labelled "community+geography" variants.
- **Never inputs:** any chemistry (including field chemistry and the other targets), physical
  habitat, condition classes/MMI/O-E, reference class (except to define the nearest-reference-site
  baseline's reference set), survey-cycle indicator, StreamCat special metrics.
- All scalers, vocabularies, prevalence filters, target standardisers and imputers are fit on the
  training fold only (unit-tested).

## 5. Metrics and uncertainty

Computed with `sentinel.eval.metrics` on pooled out-of-fold predictions:

- **pooled R²** (`metrics.r2`), RMSE, Spearman ρ;
- **per-region R²**: R² within each held-out ecoregion g, using g's own observed mean as reference;
  summarised by median, min and max over the 9 ecoregions (all 9 values stored);
- **anomaly R²** (within-region skill net of region-level bias):
  `1 − Σ_g Σ_{i∈g} ((y_i − ȳ_g) − (p_i − p̄_g))² / Σ_g Σ_{i∈g} (y_i − ȳ_g)²`, where ȳ_g and p̄_g are
  the observed and predicted means of region g in the evaluated rows; region = held-out ecoregion
  (LOEO) or the row's `ag_eco9` (other schemes). A constant-per-region predictor scores 0.
- **between-region R²**: R² of the 9 region means p̄_g against ȳ_g (diagnostic, stored in `extra`).

CIs (`sentinel.eval.bootstrap`): percentile cluster bootstrap, B = 2,000, **resampling unit =
location cluster** (primary); a second CI resampling **whole ecoregions** (9 clusters; coarse,
reported in `extra`). Model-vs-RF comparisons: `paired_bootstrap` on the same rows, location-cluster
unit; Holm across the 6 targets for the primary endpoint. Seeds: every stochastic model (BioSet, RF,
GBM, TabPFN) is run with seeds 0, 1, 2 and its evaluated prediction is the **mean over the 3 seeded
fits** (pre-specified, identical rule for models and baselines); per-seed R² is stored to show
whether a gain holds per seed (§15.1 iii). Deterministic baselines are fit once.

## 6. Cross-validation on development data (phase 1)

| Scheme | Registry split | Definition | Role |
|---|---|---|---|
| **LOEO** | `nrsa0814/loeo_cv` | 9 folds, one per `ag_eco9`, whole location clusters | **primary; model selection** |
| 50-km blocks | `nrsa0814/block50_cv` | 10 folds; location clusters (centroid) → 50-km blocks (`spatial_blocks`) → folds (`spatial_kfold`, seed 0) | secondary |
| random | `nrsa0814/random5_cv` | 5 folds over location clusters, seed 0 | leakage-gap context only |

Early stopping / inner validation (neural models only): 15% of the **training** fold's location
clusters, held out as whole 50-km blocks (`spatial_block_split`, seed 0); never the held-out fold.
Splits are saved with `split_report()` next to the results.

### Models (all under identical folds and rows)

Baselines (fixed hyperparameters, no tuning):
1. `RF[community]` — RandomForest 300 trees, `min_samples_leaf=3`, `max_features=0.3` (the
   research baseline's settings) on the community feature set.
2. `RF[community-taxa-only]` — same, Hellinger taxa only (the research baseline's exact input set).
3. `RF[geography]` — lat/lon/elevation only (geography-only control).
4. `GBM[community]` — LightGBM, 600 trees, learning rate 0.03, 31 leaves, `min_child_samples=20`,
   `colsample_bytree=0.3`, `subsample=0.8` (bagging every iteration), no early stopping.
5. `ecoregion-mean` — training-fold mean of the row's ecoregion; under LOEO the held-out
   ecoregion is absent from training, so this reduces to the training-fold global mean.
6. `nearest-reference-site` — chemistry of the geographically nearest training-fold reference
   site (`reference_class == "R"`, 2008-09/2013-14 screens).
7. `kNN-community` (modern-analogue technique) — mean of the 10 nearest training samples in chord
   (Hellinger-Euclidean) distance, inverse-distance weighted.
8. `WA-PLS[2]` — weighted-averaging partial least squares, 2 components, on relative abundances.
9. `TabPFN-v2[community]` — `tabpfn==2.2.1` regressor (Prior Labs Licence 1.1), the 300 most
   prevalent training-fold taxa (Hellinger) + sample descriptors, ≤ 6,000 training rows,
   8 estimators, GPU.

SENTINEL-BioAssess **BioSet** candidates (taxon-embedding set models, `sentinel/models/bioassess`):

| id | encoder | taxon embedding | extra |
|---|---|---|---|
| `bioset_st` | set transformer, 2 SAB layers, d = 128, 4 heads, PMA + abundance-weighted mean pooling | hierarchical (phylum+class+order+family+genus+name, initialised coarse-to-fine) | — |
| `bioset_ds` | DeepSets (no attention), same pooling | hierarchical | — |
| `bioset_st_flat` | as `bioset_st` | name-level only (one-hot-equivalent; ablation) | — |
| `bioset_st_resdrop` | as `bioset_st` | hierarchical | resolution dropout 0.3 (genus/name levels dropped per token) |
| `bioset_st_ukpt` | as `bioset_st` | hierarchical | resolution dropout 0.3 + encoder pretrained on the UK EA pool (masked-taxon + community→chemistry), fine-tuned per fold; the pretraining effect is read against `bioset_st_resdrop` |

Shared training settings: masked MSE on training-fold-standardised targets (all 6 jointly),
AdamW (lr 1e-3, weight decay 1e-2), batch 128, ≤ 200 epochs, early stopping on inner-validation
loss (patience 25), token dropout 0.1, dropout 0.1. Exact values in `configs/bioassess/bioset.yaml`.

**Selection rule (fixed):** the BioSet candidate with the highest mean pooled LOEO R² across the
6 targets on development data is frozen as SENTINEL-BioAssess; ties within 0.005 → the candidate
without pretraining, then fewer parameters. The number of configurations tried is reported; up to
3 improvement cycles (RESEARCH_PLAN §15.5) may add candidates, each logged and counted. The
selected model's development LOEO score is optimistically biased by selection; only §7 scores are
claims.

## 7. External / final evaluation (phase 2 — NOT run in phase 1)

1. **Test A (primary endpoint): LOEO × cycle-out on NRSA 2018-19.** For each ecoregion g: fit the
   frozen configuration on development rows outside g; predict 2018-19 first visits in g. Training
   never contains a 2018-19 row and never contains a location of ecoregion g, so the cross-cycle
   re-sampled locations cannot leak. RF[community] and all baselines are re-run identically. Pooled,
   per-region and anomaly R² with location-cluster CIs; paired bootstrap vs RF[community]; Holm.
2. **Test B: cycle-out.** Fit on all development rows; predict all 2018-19 first visits. Reported
   overall and separately for (i) new locations (cluster absent from development data) and
   (ii) re-sampled locations.
3. **Test C: replication on NRSA 2023-24**, protocol of Test A.
4. **Transfer UK → NRSA:** encoder + chemistry heads trained on the UK pretraining pool only,
   applied zero-shot to NRSA 2018-19 for the shared targets (pH, log conductivity, log chloride,
   log TP, log turbidity [UK NTU determinand 6396 only]); compared with NRSA-trained models and the
   geography-only control (§15.3 soft threshold). **Transfer NRSA → UK:** models fit on all
   development rows applied to the UK reserved test blocks for the same shared targets. UK
   chemistry = ±30-day median at the paired WQ point (`chemistry_matched_wide_pm30d_invertebrate`).

## 8. Integrity checks (unit tests, `tests/test_bioassess_models.py`)

No location cluster, site or visit in both training and held-out folds of any scheme; within-cycle
revisits and cross-cycle re-sampled locations grouped; test folds partition the rows; a target's
chemistry is never an input (features are unchanged when targets are perturbed; no chemistry column
names in any feature set); normalisers fit on training rows only; the development loader refuses
2018-19/2023-24.

## 9. Freezing

Before phase 2: the selected configuration and this file are frozen; sha256 of this file,
`configs/bioassess/*.yaml` and the final scoring script are recorded below; the test is scored once.

### 9.1 Freeze CANDIDATE (prepared 2026-10-24, updated after improvement cycles 1–2; NOT frozen, test NOT scored)

- **Model A — pre-registered selection:** `bioset_st_flat` = `configs/bioassess/bioset.yaml`
  defaults + `{embedding: flat}` (set transformer, 2 SAB layers, d = 128, 4 heads, PMA +
  abundance-weighted mean pooling, name-level taxon embedding, token dropout 0.1, no resolution
  dropout, no pretraining, no auxiliary outputs; MSE on the 6 standardised targets; AdamW 1e-3 /
  wd 1e-2; batch 128; ≤ 200 epochs, patience 25 on the inner 15% block-held-out validation; mean
  of seeds 0, 1, 2). Community + 3 sample descriptors, no geography. Endpoint: H-M5 vs RF[community].
- **Model B — co-primary (admitted by the lead):**
  `Ensemble[BioSet[bioset_st_flat]+TabPFNv2[community]]`, equal weights fixed a priori.
- **Model C — co-primary (admitted by the lead, 2026-10-24):**
  `Ensemble[BioSet[bioset_st_flat]+geography+RandomForest[community+geography]+TabPFNv2[community]]`,
  equal weights fixed a priori. Its claim is framed as **"BioSet adds signal beyond the strongest
  baselines"** (C contains the baseline RF[community+geography] as a member); the equal-size
  baseline-ensemble control `Ensemble[RF[community+geography]+LightGBM+TabPFNv2]` stays in the
  comparator set.

- **Geography control (reported with both):** `BioSet[bioset_st_flat]+geography` vs
  RF[community+geography], and `Ensemble[BioSet+geography+TabPFNv2]` vs
  `Ensemble[RF[community+geography]+TabPFNv2]`.
- **Lead decisions (2026-10-24):** (1) Models B and C are admitted as **co-primary** candidates.
  (2) Multiplicity: see the claim table below. (3) **SOTA comparator** = for each target, the strongest baseline on development LOEO
  (pooled R²) among {RF[community], LightGBM[community], TabPFNv2[community],
  RF[community+geography], Ensemble[RF[community+geography]+TabPFNv2], Ensemble[LightGBM+TabPFNv2]}
  plus, from cycle 2, {Ensemble[RF[community+geography]+LightGBM+TabPFNv2], RF[community+context],
  Ensemble[RF[community+context]+TabPFNv2], Ensemble[RF[community+context]+LightGBM+TabPFNv2]};
  "SOTA" is claimed only for targets where a candidate beats that baseline on the test with the
  Holm-adjusted test rejecting (claim table below). The number of configurations tried (§10) is
  attached to every test registry row. Improvement cycle 2 is complete (§10).

#### Claim table (confirmatory = Test A only; everything else secondary, unadjusted)

| Candidate | Family | Comparator per target | Family size | Decision rule |
|---|---|---|---|---|
| A `BioSet[bioset_st_flat]` | **H-M5** (registered primary endpoint, RESEARCH_PLAN §15.2) | RF[community] | 6 (A × 6 targets) | Holm-rejected for ≥ 3 of 6 targets |
| A, B, C | **SOTA** (§15.1) | per-target strongest development baseline (`sota_comparators` in the record) | 18 (3 × 6) | Holm-rejected **and** per-seed Δ > 0 for seeds 0, 1, 2 (§15.1 iii) → "SOTA" for that target |
| C | "adds signal beyond the strongest baselines" (secondary) | leave-BioSet-out `Ensemble[RF[community+geography]+TabPFNv2]` and equal-size `Ensemble[RF[community+geography]+LightGBM+TabPFNv2]` | — | stated only where Δ > 0 against **both**; unadjusted |

- **Test:** paired location-cluster bootstrap of Δ pooled R² on the same rows, B = **10,000** for
  the confirmatory (SOTA, H-M5) tests on Test A, B = 2,000 elsewhere. **One-sided rule:** p = the
  two-sided bootstrap p if Δ > 0, else 1 (equivalent to a one-sided test at α/2; conservative).
  α = 0.05, Holm step-down; raw p, Holm-adjusted p, α and the rejection flag are logged to the
  registry (`holm_rejected_sota`, `holm_rejected_h_m5`).
- **Sensitivity (pre-registered, reported in `extra`):** the same paired Δ with resampling units
  50-km block and ecoregion (Test A predictions come from 9 per-ecoregion fits; location clusters
  alone may be anti-conservative).
- **Per seed:** every stochastic model keeps seeds 0, 1, 2; an ensemble's seed-s prediction is the
  mean of its members' seed-s fits; per-seed R² and Δ are logged; saved prediction files are the
  canonical record (`--score-from-predictions` re-scores them exactly).
- **Exposure:** NRSA 2018-19 (Tests A, B, T1) was used before this pass (i) by the research-phase
  RF baseline (`rf_blocked_baseline`, fitted and scored within 2018-19) and (ii) from 2026-10-25
  03:43 UTC as the **validation cycle of D5 BioIntegrity** (benthic condition labels derived from
  the same benthic samples; one oracle, `oracle_gbm_chem`, takes the 2018-19 chemistry — M5's
  targets — as input; iterative model choice on it). No M5 setting was chosen on 2018-19: M5's
  candidates, comparators and settings were fixed on NRSA 2008-14 before (ii), and nothing has fed
  back since. Test C (2023-24) was not scored by any workstream at freeze (D5's 2023-24 test is
  pending; the post-results report records which pass ran first). This note is attached to every
  2018-19 test row (not to the UK-reserved T2 rows).

- **Comparators re-run identically in §7:** RF[community], LightGBM[community],
  TabPFNv2[community], RF[community+geography], RF[geography], ecoregion mean, nearest reference
  site, kNN-MAT, WA-PLS2, and the baseline ensembles above.
- **Phase-1 basis:** registry experiments `bioassess_phase1_cv` and `bioassess_phase1_selection`;
  selection record `$SENTINEL_RUNS/bioassess/phase1_selection.json`.
- **Scoring script (not run on test data):** `scripts/eval/bioassess_final.py` — Tests A, B (all /
  new / re-sampled locations), C, T1/T2 (`sentinel/models/bioassess/transfer.py`). Test blocks,
  B and α come only from the freeze record (no CLI overrides; all blocks in one pass). It refuses
  unless every hash in `bioassess_freeze.json` matches (code incl. shared `sentinel/eval` and
  `sentinel/core` modules and the TabPFN shim; NRSA/taxonomy/UK data files; the TabPFN checkpoint;
  package versions), `--authorised-by` is given, no one-time marker exists and the registry holds
  no `bioassess_test` row; the marker is written only after imports (incl. TabPFN, LightGBM,
  sklearn), the GPU lock, the data load and a TabPFN warm-up on the locked GPU succeed; NaN metrics
  are recorded as not estimable. The pass has two phases: (1) every fit and prediction of every
  block (A, C, B, T1/T2) is produced and cached per unit (block | ecoregion fold | base model, or
  transfer seed; sha256 manifest) and the prediction tables are saved with their sha256; (2) only
  then is anything scored, and registry rows are written only after all blocks are scored.
  Expected runtime of the real pass: ~6–9 h on one GPU (the dry run took 2.4 h on half the
  training rows, with Test C reusing Test A's fits).
- **Resume rule (pre-registered 2026-11-08):** if the single pass fails for an infrastructure
  reason (crash, OOM, kill, host loss) **before the first `bioassess_test` registry row exists**, it
  may be resumed **once**, under the same freeze record and the same authorisation, from the
  cached predictions in the same run directory (`--resume RUN_DIR --crash-note …`). Cached units
  whose sha256 matches the manifest are reused and never refitted; only units without a manifest
  entry (the unit in progress at the crash and later ones) are fitted. Because no score exists
  before all predictions exist, and scoring from saved predictions is deterministic, a resume
  carries no outcome information. The crash and the resume are logged (marker, `resume.json`,
  TrainingRun, §10). A second failure, or any failure after a registry row exists, needs a new
  lead decision. A `--dry-run-dev` mode runs the identical path
  on development data (dev = 2008-09, pseudo-test = 2013-14, UK pool validation blocks) into a
  scratch registry. The taxon vocabulary is built from development tokens only. Refusals and the
  scoring path are unit-tested.
  The candidate record `docs/preregistration/bioassess_freeze_candidate.json`
  (`scripts/eval/bioassess_freeze_candidate.py`) lists candidates, the per-target SOTA comparators
  chosen from development LOEO, and the sha256 of every frozen file including this protocol.
- **Still required before the test pass:** R10 pre-test gate done (2026-10-25: FAIL, C1–C5/W1–W12
  fixed; delta re-review 2026-11-08: PASS WITH CONCERNS, R1/R2/W-a/W-b addressed). The lead
  archives or commits the frozen tree (R3) and copies the reviewed candidate record to
  `bioassess_freeze.json` (any later edit to a hashed file — incl. the shared `sentinel/core`,
  `sentinel/eval` and M4 `chemtox` modules on the import path — makes the scorer refuse).
- **Code-change disclosure:** `cv.py`, `bioset.py`, `data.py`, `features.py` and `baselines.py` were edited after the phase-1
  CV runs (cycle 2 added the context features, the within-region loss and a behaviour-preserving
  refactor of the fold runner into `bioset_rows`, unit-tested for equivalence); earlier edits (training-fold median fill of missing elevation for the geography variant; an explicit
  error when validation loss is never finite; optional auxiliary outputs, off by default). None of
  these changes the computation of model A (`aux_targets: false`, `region_loss_weight: 0`, no
  geography); the refactor moved code without changing it (equivalence test in §8's test file).

Hashes: see `docs/preregistration/bioassess_freeze_candidate.json` (`sha256`).

## 10. Amendments and phase-1 execution log (dated; §1–§8 unchanged since CV results existed except the dated pre-CV amendments below; §9.1 is the evolving freeze candidate)

- **2026-10-22, before any CV run:** §5 seed rule made identical for all stochastic models (mean
  of seeds 0, 1, 2); §6 table states that `bioset_st_ukpt` also uses resolution dropout 0.3, so the
  pretraining effect is read against `bioset_st_resdrop`.
- **2026-10-22, during CV (execution only):** jobs were stopped and restarted under the new
  CONTRIBUTING_V2 CPU cap (≤ 8 threads per job). Only (model, scheme) pairs without registry entries
  were re-run; no configuration changed.
- **2026-10-22, after LOEO results (additional comparisons, selection rule unchanged):** BioSet
  candidates are additionally paired against `LightGBM[community]` and `TabPFNv2[community]` (the
  strongest community-only baselines in CV, stronger than the RF named in H-M5), and the
  community+geography BioSet variant is paired against `RandomForest[community+geography]`. The
  final evaluation (§7) will report these comparisons next to the pre-registered RF comparison.
- **2026-10-24, improvement cycle 1 (RESEARCH_PLAN §15.5 amendment; development LOEO only):**
  (1) `bioset_st_flat_aux` — co-measured chemistry (`data.AUX_TARGETS`) as training-only extra
  outputs: neutral (inside the 0.005 tie margin; the tie-breaker keeps `bioset_st_flat`).
  (2) Equal-weight OOF ensembles (no fitted weights): `Ensemble[BioSet[bioset_st_flat]+TabPFNv2]`
  and, as baseline-ensemble controls, `Ensemble[LightGBM+TabPFNv2]` and
  `Ensemble[RF[community+geography]+TabPFNv2]`; plus the geography variant
  `Ensemble[BioSet[bioset_st_flat]+geography+TabPFNv2]`. Configurations tried so far: 6 BioSet
  candidates, 1 BioSet geography control, 4 ensembles (2 SENTINEL, 2 baseline controls).
  Literature check (2026-10-24): published community→chemistry comparators are WA-PLS/MAT, tree
  ensembles (RF/boosted trees) and spatial regression; the first two families are reproduced here.
- **2026-10-24, improvement cycle 2 (lead directive; development LOEO only; one change at a time):**
  (1) `bioset_st_flat_rloss` — extra within-batch, within-ecoregion anomaly loss on training rows:
  pooled tie with `bioset_st_flat`, small within-region gain (mean anomaly R² +0.008).
  (2) Sample context = geography + sampling day-of-year (collection metadata) given to both sides:
  `BioSet[bioset_st_flat]+context` vs `RandomForest[community+context]`, plus the trivial-context
  baseline `RandomForest[context]` (RESEARCH_PLAN §14.3).
  (3) Equal-weight 3-member ensembles: `Ensemble[BioSet[bioset_st_flat]+geography+RF[community+geography]+TabPFNv2]`
  and its `+context` twin (tie: mean pooled R² within 0.005 → the geography version, fewer inputs),
  with equal-size baseline controls `Ensemble[RF[community+geography]+LightGBM+TabPFNv2]` and
  `Ensemble[RF[community+context]+(LightGBM+)TabPFNv2]`, added to the SOTA comparator set so the
  comparator ensembles are as large as the candidate ensembles.
  (4) TabPFN v2 on 50-km blocks (secondary scheme). (5) UK↔US transfer diagnostic on development
  rows only (`bioassess_phase1_transfer_dev`; NRSA 2018-19 and the reserved UK blocks untouched).
  Not feasible here and documented: StreamCat covariates at NRSA sites (not built for NRSA; 2008-09
  has no COMIDs — a data-workstream dependency; the predicted-condition column `prg_bmmi0809` must
  stay excluded), and ecoregion-conditional heads (the held-out ecoregion has no training rows under
  LOEO, so its head would be undefined).
  Selection tie-break extended: "then the simpler training objective" (needed when
  `bioset_st_flat_rloss` tied `bioset_st_flat` with identical parameter counts; disclosed).
  **Configurations tried (SENTINEL side, development LOEO):** 7 BioSet candidates, 2 BioSet input
  variants (+geography, +context), 4 equal-weight ensembles = 13; baseline-side additions: 2 context
  RFs, 5 baseline ensembles.
- **2026-10-25, pre-test gate (independent review, FAIL → fixes):** scorer and freeze builder
  rewritten as listed in §9.1 (per-seed results, H-M5 and SOTA Holm families on Test A logged to
  the registry, full provenance hashing, development dry run, marker after setup, record-only
  settings, development-only vocabulary, C-vs-control comparisons, B = 10,000, block/ecoregion
  sensitivity, configuration counts and exposure note on every test row, `is_baseline` false for
  SENTINEL controls). `scripts/eval/baseline_bioassess_rf.py` now refuses to run without an explicit
  acknowledgement flag (it fits and scores within 2018-19). Two development comparison rows written
  from an interactive `python -c` were re-logged from `scripts/eval/bioassess_pair_cv.py`
  (superseding the originals; values unchanged).
- **Deviation — transfer tests T1/T2:** §7.4 names "encoder + chemistry heads trained on the UK
  pool"; the pass uses `RandomForest[family-log-class]` on a shared family-level representation
  (`transfer.py`) because the pretrained checkpoint keeps no chemistry heads and the encoder
  candidate (`bioset_st_ukpt`) was not selected. Seeds 0, 1, 2; T1 is compared with
  RF[geography] (NRSA-trained, Test B fit) on the same rows (§15.3 soft threshold); T2 has no
  comparator. Development diagnostic: `bioassess_phase1_transfer_dev`.
- **Process deviation — R10:** M5 had no pre-training review before its first training run
  (2026-10-22); the 2026-10-25 pre-test gate found no leakage retroactively. Disclosed in
  RESULTS.md and to be disclosed in the paper.
- **Selection outcome (rule of §6, development LOEO):** `bioset_st_flat`. Configurations tried: 5
  BioSet candidates (+ 1 community+geography control run of the selected candidate, not a
  selection candidate). The hierarchical-embedding and UK-pretraining candidates did not win
  (negative ablation results, reported as such).
- **Scorer verification (2026-11-08, development only):** the reproduction rerun of candidate A's
  development LOEO score with the frozen-candidate code matches the registered 2026-10-22 values to
  < 1e-7 R² on all six targets (per-seed R² identical to 4 d.p.). `--dry-run-dev` attempts (all on
  development data, scratch registry; nothing logged to `results/registry.jsonl`):
  1. 2026-10-25 03:01 — **failed** after 6.4 min: `elevation_m` is missing for every NRSA 2008-09
     row, so a 2008-09-only fit fold had a NaN fill for the `+geography` BioSet. Fix (`cv._glob`): a
     geography column with no observed value in the fit rows is set to 0 for *all* rows of that fold
     (train and test). It never triggers when the fit rows include 2013-14 (all registered CV runs
     and the real pass), so registered results and the frozen procedure are unchanged.
  2. 2026-11-08 12:23 — **failed** after 0.9 min: CUDA OOM in the first TabPFN fit on GPU 4, which a
     non-SENTINEL process occupied (9 GB of 11 GB). Its run directory (holding only a copy of the
     record) was deleted. Consequences: the scorer pins no GPU (lock-helper cap), requires ≥ 5 GB
     physically free at lock time, and warms TabPFN up before the marker.
  3. 2026-11-08 13:38 — stopped by M5 while still waiting for a GPU lock (before any fit), when the
     GPU cap changed to GPUs 0/1; its run directory (record copy only) was deleted.
  4. 2026-11-08 15:53 — **completed** end to end (GPU 0, 2 h 26 min; 2,179 scratch rows).
  After the delta re-review the scorer was restructured into the two-phase form with the per-unit
  cache and the resume rule (W-a), and re-run on development data:
  5. 2026-11-08 19:18 — restructured scorer, **with a deliberate kill to exercise the resume rule**:
     SIGTERM at 19:52 after 20 cached units (a hard kill leaves no TrainingRun entry; the crash note
     is in `resume.json`), then `--resume` (19:53–21:56, GPU 0): 20 units reused without refitting,
     233 fitted; phase 1 finished 21:27, phase 2 (scoring + one flush) took 29 min; 2,180 scratch rows
     (the 2,179 of attempt 4 + the new `h_m5_n_targets_rejected` row); `not_estimable.json` empty.
     Against attempt 4: all prediction tables agree to ≤ 3.5e-6 (BioSet GPU float32 nondeterminism;
     TabPFN/RF/T1/T2 ≤ 5e-15), registry values to ≤ 6.2e-8, and all 24 Holm decisions are identical.
     Run directory `$SENTINEL_RUNS/bioassess/final_dryrun_dev_20261108T191804/`.
