---
model_id: sentinel_bioassess
name: SENTINEL-BioAssess
modality: bioassess
registry_models: ["BioSet[bioset_st_flat]", "Ensemble[BioSet[bioset_st_flat]+geography+RandomForest[community+geography]+TabPFNv2[community]]", "BioSet[bioset_st_flat]+context", "BioSet[bioset_st_flat_rloss]", "Ensemble[BioSet[bioset_st_flat]+TabPFNv2[community]]", "Ensemble[BioSet[bioset_st_flat]+geography+TabPFNv2[community]]", "BioSet[bioset_st_flat_aux]", "BioSet[bioset_st_flat]+geography", "BioSet[bioset_st]", "BioSet[bioset_ds]", "BioSet[bioset_st_resdrop]", "BioSet[bioset_st_ukpt]"]
registry_experiments: [bioassess_phase1_cv, bioassess_phase1_selection, bioassess_phase1_transfer_dev, bioassess_test, bioassess_test_posthoc]
workstream: M5
version: 0.2.0
status: externally tested (single pre-registered pass 2026-11-09)
license: MIT
intended_use: >
  Research model inferring co-measured stream water chemistry (log TP, log TN, log conductivity,
  log chloride, log turbidity, lab pH) from a benthic macroinvertebrate assemblage (NRSA-style
  300-organism fixed count, harmonised taxonomy), for condition/stressor inference at sites with
  biological but no chemical sampling, and as a token producer for SENTINEL-Fuse.
limitations: >
  Tested once (NRSA 2018-19, replication 2023-24, UK reserved). The SOTA scope is narrow: only as
  the ensemble with the strongest baselines, only under leave-one-ecoregion-out, only for chloride, pH,
  conductivity, with 1-2 % RMSE gains; BioSet alone is below the strongest baseline on 4 targets, and
  where the region was sampled before a geography-only RF is better for TN, conductivity, chloride.
  Pooled skill mixes between-region and within-region signal; read the anomaly and per-ecoregion
  R² before using predictions inside one ecoregion. One visit per site: no temporal / early-warning
  use. Not validated outside CONUS wadeable/boatable NRSA protocols; UK<->US transfer failed (negative).
---

# Model card — SENTINEL-BioAssess (M5)

**Status: externally tested once (2026-11-09).** Development results are registry entries
(`bioassess_phase1_cv`, `bioassess_phase1_selection`); the single test pass is registry experiment
`bioassess_test` (2,180 rows), the post-hoc bio.infer comparison `bioassess_test_posthoc`. The headline
test numbers below are copied from those rows (RESULTS.md, 2026-11-09; docs/preregistration/bioassess_test_results.md, formerly protocol §11).

## Model details

**BioSet** — a taxon-embedding set model (`sentinel/models/bioassess/bioset.py`):

- tokens = present taxa; each token = Σ of learned embeddings of its harmonised lineage
  (phylum, class, order, family, genus, finest name; coarse-to-fine initial scales so related taxa
  start close) + a projection of four abundance features (log count, relative abundance, its square
  root, log10-abundance class). Coarser identifications carry fewer lineage levels (mixed-resolution
  handling); optional *resolution dropout* removes genus/name ids during training;
- encoder: 2 pre-LN self-attention blocks (set transformer, no positional encoding) or DeepSets;
  pooling = attention pooling (PMA) ⊕ abundance-weighted mean;
- sample descriptors (log10 total count = total-count offset, log10 number of taxa, boatable
  protocol) concatenated after pooling;
- multi-target linear heads, masked MSE on training-fold-standardised targets;
- optional encoder pretraining on the UK Environment Agency invertebrate archive (masked-taxon
  family/order prediction + community → ±30-day chemistry; `sentinel/models/bioassess/pretrain.py`,
  `scripts/train/bioassess_pretrain_uk.py`);
- the evaluated prediction is the mean of 3 seeded networks.

Candidates and fixed settings: `configs/bioassess/bioset.yaml`. **Selected by the pre-registered
LOEO rule: `bioset_st_flat`** (set transformer with a flat, name-level taxon embedding); the
hierarchical-embedding, DeepSets, resolution-dropout and UK-pretrained candidates are kept as
reported ablations (selection record: `$SENTINEL_RUNS/bioassess/phase1_selection.json`). Parameter counts are recorded per
fold in `diag_<model>_<scheme>.csv` next to the out-of-fold predictions.

## Training and evaluation data

`nars_streams` (NRSA benthic 300-counts + chemistry), `taxonomy_harmonised`, `uk_ea_ecology`
(pretraining pool only). Development = NRSA 2008-09 + 2013-14 first visits; 1-km location clusters
over all NRSA cycles are the unit of every split and bootstrap (NRSA re-samples locations under new
site ids). See `docs/preregistration/bioassess_protocol.md` §2–§4.

## Evaluation

Pre-registration: `docs/preregistration/bioassess_protocol.md`. Phase-1 splits:
`nrsa0814/loeo_cv` (primary, selection), `nrsa0814/block50_cv`, `nrsa0814/random5_cv` (leakage
gap). Baselines under identical folds: RF (community; taxa-only; geography-only;
community+geography), LightGBM, ecoregion mean, nearest reference site, kNN modern analogues,
WA-PLS (2 components), TabPFN v2. Metrics: pooled R², anomaly R² (within-region), per-ecoregion R²
(median/min/max), between-region R², RMSE, Spearman; location-cluster bootstrap CIs (B = 2000) and
ecoregion-unit CIs. Selection and paired differences vs RF[community]:
`scripts/eval/bioassess_cv_report.py`.

### Phase-1 status (qualitative; numbers only in the registry)

Development LOEO CV only (selection-biased; configuration counts in the protocol §10). A single
BioSet network is above the pre-registered blocked-RF reference for every target and above
like-for-like geography/context-augmented RFs for a minority-to-majority of targets, but not above
TabPFN v2 for most targets. An equal-weight ensemble that adds BioSet+geography to the strongest
geography-augmented baselines is above the per-target strongest baseline (including equal-size
baseline ensembles) for every target, pooled and within-region. Within-ecoregion skill remains much
lower than pooled skill. UK↔US zero-shot transfer is weak (development diagnostic,
`bioassess_phase1_transfer_dev`). No external test has been scored; candidates and comparators for
the single test pass are listed in `docs/preregistration/bioassess_protocol.md` §9.1.

### External test (single pre-registered pass, 2026-11-09)

Run `$SENTINEL_RUNS/bioassess/final_20261108T220242/`; post-results review 🟡 (REVIEW_REPORT_POST_RESULTS.md);
labels scoped as the review requires. **Scope: Test A = leave-one-ecoregion-out × cycle-out on NRSA 2018-19
(prediction in an unseen ecoregion), and the label belongs to ensemble C** = mean(BioSet+geography,
RF[community+geography], TabPFNv2) — BioSet adds signal to the strongest baselines; BioSet is not SOTA alone.

| Target | C − comparator, Test A (Δ R² [95 % CI]) | RMSE change | Label | BioSet alone − comparator |
|---|---|---|---|---|
| Chloride | +0.021 [+0.013, +0.028] | −2.0 % | SOTA (ensemble C, LOEO) | −0.031 [−0.056, −0.009] |
| pH | +0.024 [+0.015, +0.032] | −2.2 % | SOTA (ensemble C, LOEO) | −0.036 [−0.071, −0.003] |
| Conductivity | +0.013 [+0.006, +0.019] | −1.5 % | SOTA (ensemble C, LOEO) | −0.038 [−0.062, −0.016] |
| TN | +0.011 [+0.004, +0.017] | −1.1 % | competitive (SOTA by rule; not robust to ecoregion resampling, 4/9 ecoregions +) | −0.059 [−0.081, −0.037] |
| TP | +0.005 [−0.003, +0.013] | −0.4 % | competitive | −0.008 [−0.030, +0.013] |
| Turbidity | +0.012 [+0.002, +0.022] | −1.0 % | competitive (Holm-not rejected) | −0.014 [−0.034, +0.006] |

- **Test B (cycle-out; regions sampled before):** C − comparator ≈ 0 on every target (CIs include 0); a
  geography-only RF is the best model for TN (R² 0.680 vs C 0.619), conductivity (0.740 vs 0.712) and
  chloride (0.706 vs 0.642). Use the community signal for out-of-region inference.
- **H-M5 met:** BioSet > blocked RF[community] on 6/6 targets (Test A, Holm over 6; Δ +0.045 to +0.133),
  4/6 with every seed positive; Test C 5/6 (chloride not rejected, all seeds negative).
- **Test C (NRSA 2023-24, secondary replication):** chloride, pH, conductivity and TN gains replicate.
  Exposure: D5 scored 2023-24 ~1.5 h before the M5 freeze (addendum A1/A2); no M5 decision used it.
- **Transfer UK↔US (T1/T2): negative.**
- **Post hoc (non-confirmatory, addenda A3/A4):** our bio.infer-style re-implementation performed poorly; it may not reflect the EPA software; the SOTA label rests on the pre-registered comparators (WA-PLS2, kNN-MAT, RF, TabPFN, ensembles). (Its development LOEO R² was −0.29 to −1.60,
  far below WA-PLS2; bio.infer is therefore listed as not compared.)
- Absolute skill is moderate (pooled R² 0.41–0.57; median per-ecoregion R² 0.16–0.34): the gains are
  statistically solid for chloride, pH, conductivity but practically marginal.

## Uncertainty and calibration

Phase 1 produces point predictions; CIs quantify sampling uncertainty of the metrics only.
Predictive distributions (seed spread / heteroscedastic heads) are not yet calibrated.

## Limitations, risks and known failure modes

- Index circularity avoided by design: no MMI/O-E/condition class is predicted from counts.
- Chemistry is never an input (unit-tested); reference class is used only to define the
  nearest-reference-site baseline.
- Taxonomic resolution and lab practice differ by cycle; the harmonised lineage absorbs part of this,
  and no survey-cycle indicator is an input.

## Licences

NRSA: U.S. Government work (EPA). UK EA: Open Government Licence v3.0. GBIF Backbone (taxonomy
harmonisation): CC BY 4.0. TabPFN 2.2.1 (baseline only): Prior Labs Licence 1.1.

## Provenance

Every CV entry stores git state, config hash, input sha256 and the protocol sha256; every training
run is in `TRAINING_LOG.md` / `logs/training_runs.jsonl`.
