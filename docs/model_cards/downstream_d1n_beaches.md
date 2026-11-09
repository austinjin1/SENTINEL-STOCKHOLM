---
model_id: d1n_logistic
name: SafeSwim national beaches (D1n)
modality: downstream
registry_models: [logistic]
registry_experiments: ["d1n_beaches/test_look"]
workstream: Downstream
version: 1.0.0
status: frozen (pre-registered)
license: MIT
intended_use: >
  Morning forecast of beach-day E. coli > 235 CFU/100 mL at US freshwater beaches monitored in WQP.
limitations: >
  Chicago excluded; new = new HUC4, not new water body; WQP coverage dominated by Great Lakes states.
---

# Model card — SafeSwim national beaches (D1n)

## Model details
L2 logistic regression on own-beach history, neighbour results and gridMET windows. Code: `scripts/train/downstream_d1n_beaches_test.py` (frozen test path), module(s) under
`sentinel/models/downstream/`. Frozen configuration and every hashed file: `docs/preregistration/FREEZE_downstream_d1n_beaches.json`
(its `prereg_archive` holds a tar.gz of all hashed files with sha256).

## Intended use and out-of-scope use
Research decision support only; not a regulatory or public-health advisory tool.

## Training and evaluation data
Protocol `docs/preregistration/downstream_d1n_beaches.md`; split file and sha256 in the FREEZE record; data files sha256-pinned in
`docs/preregistration/downstream_d1n_beaches_data_pins.json`.

## Evaluation
Single pre-registered test look (registry experiment `d1n_beaches/test_look`); baselines and incumbents scored on identical rows;
the §15.1 label is stated in the protocol's test-result section and in RESULTS.md. No numbers here.

**Evaluation summary:** Beats its re-implemented Virtual-Beach incumbent and the strongest simple baseline on the D1n track; NOT SOTA (a Brooks-style RF was not compared). Primary registry rows: bd1ea64b30fe, 5e2298a8f07e, 76bde9f699fd, a713b5934069.

## Limitations, risks and known failure modes
Incumbent (VB-MLR) is a simplified re-implementation; 'new beach' = beach in a never-fitted HUC4, not a new water body. See also the front matter and the protocol's deviation and disclosure sections.

## Licences
US public-domain / open data as listed in the datasheets (`docs/datasheets/`).

## Provenance
No git (user instruction); provenance = FREEZE record, prereg archive sha256 and the `prereg_snapshot` registry row.
