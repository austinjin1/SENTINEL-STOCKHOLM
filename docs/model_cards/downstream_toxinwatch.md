---
model_id: toxinwatch_gbm_it3
name: ToxinWatch GBM it3 (D2)
modality: downstream
registry_models: [gbm_challenger]
registry_experiments: ["toxinwatch/mc_exceed8@it3/test_look"]
workstream: Downstream
version: 1.0.0
status: frozen (pre-registered)
license: MIT
intended_use: >
  Probability that a Washington lake sample exceeds 8 ug/L microcystin, at decision time d-7.
limitations: >
  Lake Washington dominates; reactive sampling; national rows added for training only.
---

# Model card — ToxinWatch GBM it3 (D2)

## Model details
LightGBM (3 seeds) trained on Washington + national WQP microcystin rows. Code: `scripts/train/downstream_toxinwatch_test.py` (frozen test path), module(s) under
`sentinel/models/downstream/`. Frozen configuration and every hashed file: `docs/preregistration/FREEZE_downstream_toxinwatch.json`
(its `prereg_archive` holds a tar.gz of all hashed files with sha256).

## Intended use and out-of-scope use
Research decision support only; not a regulatory or public-health advisory tool.

## Training and evaluation data
Protocol `docs/preregistration/downstream_toxinwatch.md`; split file and sha256 in the FREEZE record; data files sha256-pinned in
`docs/preregistration/downstream_toxinwatch_data_pins.json`.

## Evaluation
Single pre-registered test look (registry experiment `toxinwatch/mc_exceed8@it3/test_look`); baselines and incumbents scored on identical rows;
the §15.1 label is stated in the protocol's test-result section and in RESULTS.md. No numbers here.

**Evaluation summary:** Competitive (does not beat the strongest INLA-style incumbent after Holm). Primary registry rows: 1d857c230b72, 25348c4309b1, 2b069af464a5, 19784d2940ed.

## Limitations, risks and known failure modes
Incumbent is an INLA-style re-implementation (regularised GLMM-style logistic with splines), not INLA; Lake Washington dominates the known-lake bootstrap. See also the front matter and the protocol's deviation and disclosure sections.

## Licences
US public-domain / open data as listed in the datasheets (`docs/datasheets/`).

## Provenance
No git (user instruction); provenance = FREEZE record, prereg archive sha256 and the `prereg_snapshot` registry row.
