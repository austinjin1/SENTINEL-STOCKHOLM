# Datasheet — ChemTox prospective test set (ECOTOX records ADORE does not contain)

| | |
|---|---|
| **Source** | EPA ECOTOX knowledgebase, release **2026-09-15** (`$SENTINEL_DATA/processed/ecotox/ecotox_aquatic.parquet`, manifest `ecotox`), curated with **ADORE's own rules**; ADORE (DOI [10.25678/0008C9](https://doi.org/10.25678/0008C9), manifest `adore_raw`) is used **only** to define what must be excluded, plus its per-species Add-my-Pet block; chemical identities from `$SENTINEL_DATA/processed/chemicals/chem_ids.parquet` (manifest `chemicals`) |
| **License** | ECOTOX: US EPA, public domain. ADORE: CC BY 4.0 (exclusion reference + species traits). |
| **Built** | 2026-10-24, `scripts/data/chemtox_build_prospective.py` (no network access; no model is fitted and no target value printed) |
| **Manifest** | `chemtox_prospective` |
| **Local paths** | `$SENTINEL_DATA/processed/chemtox_prospective/{prospective.parquet, report.json, build.log}` |
| **Rows / columns** | 11,500 rows × 55 columns (ADORE's column names, so the ADORE feature pipeline consumes it unchanged) |
| **sha256** | `ea078629fb084e6fad0b721ca2c76fbfbca5a0cd838e5373639251f4b288db7e` (byte-reproducible in `envs/sentinel2`; verified by rebuilding) |
| **Loader / module** | `sentinel/models/chemtox/prospective.py` — `build()`, `load(track)`, `augmentation_rows()`, `assert_prospective_chemicals_absent()` |
| **Pre-registration** | `docs/preregistration/chemtox_prospective_plan.md` (stage 1 frozen before any model was scored) |

## Why it exists

The single pre-registered ADORE test pass is spent (`docs/preregistration/chemtox_test_plan.md`), and
RESEARCH_PLAN §15.5 forbids re-claiming on a spent test block. ADORE's core *is* ECOTOX, so the natural
unseen test set is the ECOTOX records ADORE does **not** contain, curated identically.

## Curation (ADORE's rules, recovered from the release on 2026-10-24)

`effect_norm = MOR`; `endpoint_norm ∈ {LC50, EC50}`; `duration_h ∈ {24, 48, 72, 96}`;
`taxon_group ∈ {fish, crustaceans, algae}`; `media_type_norm ∈ {FW, SW}`; `conc1_type_norm ∈ {A, F, T}`;
not left/right censored (`~` kept, as in ADORE); a molar target (`log10_conc_mol_L`); a parseable SMILES.
No filter on exposure type or test location (ADORE has all levels). One row per ECOTOX `result_id`
(ADORE does not aggregate replicates either).

Funnel: 471,097 aquatic rows → 153,346 (MOR) → 120,230 (endpoint) → 100,706 (duration) → 81,064 (taxa) →
80,800 (media) → 77,523 (concentration type) → 71,301 (uncensored) → 60,834 (molar target) →
**60,590 rows / 3,020 chemicals**, of which 27,781 rows are ADORE rows (matched by `result_id`).

**Two documented deviations from ADORE** (also in the pre-registration §1.1): rows whose value ADORE
reconstructed as the average of a reported min and max are not included (2.0% of ADORE's t-F2F rows), and
`chem_ws` / `chem_mp` (experimental water solubility and melting point, which ADORE compiled per chemical
from PubChem) do not exist for chemicals outside ADORE and are stored as missing.

## Exclusion logic (what makes it a test set)

A row enters a **novel-chemical** track only if its `result_id` is absent from all 158,330 ADORE rows
inspected (the filtered table plus every challenge file) **and** its chemical is absent from ADORE by CAS
*and* by InChIKey (2,408 CAS / 2,407 InChIKeys) **and** its species is one of ADORE's 203 species (so the
Add-my-Pet block is defined, copied per species after asserting it is species-constant).
`assert_disjoint_from_adore` raises on any overlap and runs for every track inside the builder;
`tests/test_chemtox_prospective.py` re-checks it. The **temporal** track instead holds out *studies*
(publications after ADORE's latest reference year, 2021) and may share chemicals with ADORE.

## Tracks

| Track | Rows | Chemicals | Species | References | median pub. year | NN Tanimoto to ADORE train (median) |
|---|---|---|---|---|---|---|
| `fish_novel` ∩ `in_adore_domain` (**primary endpoint**) | 1,912 | 301 | 64 | 310 | 1985 | 0.425 (all `fish_novel`) |
| `fish_novel` (all carbon-containing) | 2,313 | 363 | 71 | 364 | 1985 | 0.425 |
| `crust_novel` ∩ `in_adore_domain` | 230 | 112 | 9 | 98 | 1992 | 0.476 (all `crust_novel`) |
| `crust_novel` (all carbon-containing) | 295 | 136 | 10 | 114 | 1992 | 0.476 |
| `algae_novel` | 3 | 2 | 2 | 2 | — | 0.371 |
| `noncarbon` (metal salts / inorganics; out of ADORE's domain) | 8,774 | 225 | 114 | 1,136 | 1984 | 0.200 |
| `temporal` (published after 2021; **underpowered**) | 115 | 29 | 13 | 44 | 2024 | 0.571 |

`in_adore_domain` is a stored boolean: `structure_class ∈ {organic, organic_multicomponent}`, the two classes
that make up 99.6% of ADORE's 2,408 chemicals (2,191 + 207; 10 inorganic, no organometallics). The primary
endpoint is therefore fixed by the file, not by a decision taken after seeing scores.

## Target and difficulty

`result_conc1_mean_mol_log` = log10(LC50/EC50 in mol/L). Primary track: mean −4.58, SD **1.730**,
replicate noise floor measured inside the set (pooled within (chemical, species, duration) SD over 361
groups / 1,180 rows) **0.452** log units — above ADORE's 0.383, i.e. these records are noisier.
Endpoints 2,297 LC50 / 16 EC50 (all `fish_novel`); durations 96 h 1,010, 24 h 605, 48 h 555, 72 h 143.

## Known issues

* The novel-chemical pool is biased towards chemicals ADORE *could not* featurise: of 12,894 novel-chemical
  rows on ADORE species, 10,216 are inorganic and 1,087 multicomponent. That is why the primary endpoint is
  restricted to ADORE's own chemical classes and the rest is reported as labelled extension tracks.
* The novel-chemical tracks are **not** temporally new (median publication year 1985): their novelty is
  chemical, not chronological. The temporal track is the chronological one and it is underpowered
  (29 chemicals), so no claim rests on it.
* Taxonomic coverage is narrower than ADORE's (64 fish species in the primary track vs 140 in ADORE t-F2F)
  and 8 rows have missing Add-my-Pet values, which the feature pipeline imputes to the training mean.
* `chem_pcp_fp` (PubChem) and `chem_ToxPrint_fp` cannot be computed locally, so feature specs using those
  fingerprints cannot be scored here; MACCS and Morgan are bit-identical to ADORE's own columns
  (300/300 and 200/200 chemicals verified), and `chem_rdkit_clogp` matches to 2e-15.

## How to rebuild

```
python -B scripts/data/chemtox_build_prospective.py          # writes the parquet, report.json, manifest
python -B -m pytest tests/test_chemtox_prospective.py -q     # 16 integrity tests, incl. the sha256 check
```
