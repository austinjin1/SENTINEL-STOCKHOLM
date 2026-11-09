---
model_id: pressure_gbm
name: MusselGuard pressure GBM (D6)
modality: downstream
registry_models: [pressure_gbm]
registry_experiments: ["musselguard/test_look"]
workstream: Downstream
version: 1.0.0
status: frozen (pre-registered)
license: MIT
intended_use: >
  Three-year-ahead risk that an uninfested CONUS HUC12 records a first dreissenid detection.
limitations: >
  NAS label completeness for recent years unknown; reporting-effort driven.
---

# Model card — MusselGuard pressure GBM (D6)

## Model details
LightGBM on propagule-pressure, network and location features (3 seeds). Code: `scripts/train/downstream_musselguard_test.py` (frozen test path), module(s) under
`sentinel/models/downstream/`. Frozen configuration and every hashed file: `docs/preregistration/FREEZE_downstream_musselguard.json`
(its `prereg_archive` holds a tar.gz of all hashed files with sha256).

## Intended use and out-of-scope use
Research decision support only; not a regulatory or public-health advisory tool.

## Training and evaluation data
Protocol `docs/preregistration/downstream_musselguard.md`; split file and sha256 in the FREEZE record; data files sha256-pinned in
`docs/preregistration/downstream_musselguard_data_pins.json`.

## Evaluation
Single pre-registered test look (registry experiment `musselguard/test_look`); baselines and incumbents scored on identical rows;
the §15.1 label is stated in `docs/preregistration/downstream_musselguard_test_results.md` (the frozen protocol itself is read-only) and in RESULTS.md. No numbers here.

**Evaluation summary:** Competitive (beats all baselines at origin 2018, not at 2021; both origins were required). Primary registry rows: fc866dd478fc, bf1446da18cc, d22f0087f0b3, 39a3640ae5f1.

## Limitations, risks and known failure modes
Incumbent (Kanankege 2018) is a regression-kriging re-implementation with a census-population proxy; T = 2021 has few positives and possibly incomplete labels. See also the front matter and the protocol's deviation and disclosure sections.

## Licences
US public-domain / open data as listed in the datasheets (`docs/datasheets/`).

## Provenance
No git (user instruction); provenance = FREEZE record, prereg archive sha256 and the `prereg_snapshot` registry row.
