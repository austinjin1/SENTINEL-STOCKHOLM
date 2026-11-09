# Model cards

One Markdown file per SENTINEL v2 model, named `<model_id>.md`, beginning with YAML front matter.
The SENTINEL Console (`GET /models`, Models page) reads the front matter and attaches every
**current** entry of `results/registry.jsonl` whose `model` is listed in `registry_models`
(optionally restricted to `registry_experiments`). This README is the template and is not itself
treated as a card; neither is any file whose name starts with `_`.

**Never type a metric into a card.** Performance numbers exist only in the results registry,
written by the code that computed them (`sentinel.core.registry.log_result`). The Console shows
them next to the card through the metric guard (`sentinel/platform/guard.py`), which refuses any
number that is not a current registry entry. A card says what the model is, what it was trained
on, how it was evaluated and where its results are recorded. The platform test suite scans this
template for typed-in metric claims.

A model can be served by `POST /predict/{modality}` only if its entry in
`sentinel/platform/models.yaml` points to an existing card and the registry holds current entries
for the entry's `registry_experiment` and `model_id` (see `docs/platform.md`, "Plugin interface").

## Template

Copy everything between the outer fences into `docs/model_cards/<model_id>.md`.

````markdown
---
model_id: sentinel_sonde            # registry "model" name of the documented model (required)
name: SENTINEL-Sonde                # display name (required)
modality: sonde                     # sonde | orbit | lite | microbe | chemtox | bioassess | hydromet | fuse (required)
registry_models: [sentinel_sonde]   # registry "model" values this card covers (required)
registry_experiments: []            # optional: only these registry experiments
workstream: M1
version: 0.1.0
status: in development              # in development | frozen (pre-registered) | retired
license: MIT
intended_use: >
  What the model is for, for whom, at what spatial and temporal scale.
limitations: >
  Known failure modes and conditions outside the training distribution.
---

# SENTINEL-Sonde

## Model details
Architecture; parameter count (computed from the checkpoint and logged to the registry);
training code path; checkpoint location under `$SENTINEL_RUNS` and its sha256.

## Intended use and out-of-scope use

## Training and evaluation data
Manifest names (`$SENTINEL_DATA/manifests/<name>.json`), the split file and its hash, time and
spatial blocks, known data issues.

## Evaluation
Registry experiment ids, the pre-registration document, and the baselines compared under the
identical split. No numbers here: the Console renders them from the registry.

## Uncertainty and calibration
How predictive distributions are produced and how calibration was checked (registry experiment ids).

## Limitations, risks and known failure modes

## Licences
Data sources and dependencies (see `docs/licenses.md`).

## Provenance
Git commit, config hash and the registry entry ids of the frozen evaluation.
````
