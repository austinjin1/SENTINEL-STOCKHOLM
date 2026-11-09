---
model_id: safeswim_d1b_logistic
name: SafeSwim D1b nowcast (Chicago)
modality: downstream
registry_models: [logistic_l2]
registry_experiments: ["safeswim/d1b_nowcast/test_look"]
workstream: Downstream
version: 1.0.0
status: frozen (pre-registered)
license: MIT
intended_use: >
  Same-afternoon nowcast of qPCR Enterococci > 1,000 CCE/100 mL at Chicago non-input beaches.
limitations: >
  Twelve beaches, one city; relies on same-day qPCR at five input beaches; trained 2015-2021.
---

# Model card — SafeSwim D1b nowcast (Chicago)

## Model details
L2 logistic regression on 26 features (compact_each+q). Code: `scripts/train/downstream_safeswim_test.py` (frozen test path), module(s) under
`sentinel/models/downstream/`. Frozen configuration and every hashed file: `docs/preregistration/FREEZE_downstream_safeswim.json`
(its `prereg_archive` holds a tar.gz of all hashed files with sha256).

## Intended use and out-of-scope use
Research decision support only; not a regulatory or public-health advisory tool.

## Training and evaluation data
Protocol `docs/preregistration/downstream_safeswim.md`; split file and sha256 in the FREEZE record; data files sha256-pinned in
`docs/preregistration/downstream_safeswim_data_pins.json`.

## Evaluation
Single pre-registered test look (registry experiment `safeswim/d1b_nowcast/test_look`); baselines and incumbents scored on identical rows;
the §15.1 label is stated in the protocol's test-result section and in RESULTS.md. No numbers here.

**Evaluation summary:** Competitive (single test look; does not beat the official City model or the feature-beach GM after Holm). Primary registry rows: 351537383783, e5e785365380, db7cef7c21a2, 306e86856d31.

## Limitations, risks and known failure modes
Incumbent is the published City of Chicago model (used as published). Only 12 beach clusters, so beach-cluster CIs are likely too narrow. See also the front matter and the protocol's deviation and disclosure sections.

## Licences
US public-domain / open data as listed in the datasheets (`docs/datasheets/`).

## Provenance
No git (user instruction); provenance = FREEZE record, prereg archive sha256 and the `prereg_snapshot` registry row.
