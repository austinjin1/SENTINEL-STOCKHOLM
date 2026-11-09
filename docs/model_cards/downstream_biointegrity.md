---
model_id: ens_tuned_gbm_rf
name: BioIntegrity ensemble (D5)
modality: downstream
registry_models: [ens_tuned_gbm_rf]
registry_experiments: ["biointegrity/test_look"]
workstream: Downstream
version: 1.0.0
status: frozen (pre-registered)
license: MIT
intended_use: >
  Probability that a wadeable stream reach is in Poor benthic condition (NRSA MMI) from landscape covariates.
limitations: >
  NRSA wadeable/boatable CONUS only; Fair class excluded from the binary target.
---

# Model card — BioIntegrity ensemble (D5)

## Model details
Equal-weight mean of a tuned LightGBM and a tuned random forest on geography + StreamCat. Code: `scripts/train/downstream_biointegrity_test.py` (frozen test path), module(s) under
`sentinel/models/downstream/`. Frozen configuration and every hashed file: `docs/preregistration/FREEZE_downstream_biointegrity.json`
(its `prereg_archive` holds a tar.gz of all hashed files with sha256).

## Intended use and out-of-scope use
Research decision support only; not a regulatory or public-health advisory tool.

## Training and evaluation data
Protocol `docs/preregistration/downstream_biointegrity.md`; split file and sha256 in the FREEZE record; data files sha256-pinned in
`docs/preregistration/downstream_biointegrity_data_pins.json`.

## Evaluation
Single pre-registered test look (registry experiment `biointegrity/test_look`); baselines and incumbents scored on identical rows;
the §15.1 label is stated in the protocol's test-result section and in RESULTS.md. No numbers here.

**Evaluation summary:** Competitive (beats the Hill-2017 RF incumbent but not the feature-matched RFs; pre-registered rule not met). Primary registry rows: 4f672ea74c88, 2a9c4c56315c, 391f046a404e, 59b50f578b64.

## Limitations, risks and known failure modes
Uses NRSA 2018-19 for fitting at test time, the same cycle M5 BioAssess reserves; no M5 output is used. See also the front matter and the protocol's deviation and disclosure sections.

## Licences
US public-domain / open data as listed in the datasheets (`docs/datasheets/`).

## Provenance
No git (user instruction); provenance = FREEZE record, prereg archive sha256 and the `prereg_snapshot` registry row.
