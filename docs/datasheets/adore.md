# Datasheet — ADORE (A benchmark Dataset for machine learning in ecotOxicology)

| | |
|---|---|
| **Source** | Schür, Gasser, Perez-Cruz, Schirmer & Baity-Jesi (2023), *A benchmark dataset for machine learning in ecotoxicology*, Scientific Data 10:718. Data: Eawag ERIC open data, DOI [10.25678/0008C9](https://doi.org/10.25678/0008C9) |
| **Access** | CKAN API `https://opendata.eawag.ch/api/3/action/package_show?id=adore` → resources `readme.md`, `license`, `adore_dataset.zip` (404,308,404 B). The HTML page sits behind a bot check; the API and resource URLs do not. |
| **Version / integrity** | CKAN `metadata_modified` 2024-01-18T11:22:36; zip resource `last_modified` 2023-09-14. sha256 of all three resources verified against the CKAN `hash` field (zip `2f31028c5e517416…`). |
| **License** | CC BY 4.0 (the `LICENSE` file in the release; CKAN leaves `license_id` empty) |
| **Retrieved** | 2026-10-22 (manifest `adore_raw`) |
| **Manifests** | `adore_raw` (converted release, 35 members), `adore` (statistics + official split assignments) |
| **Local paths** | `$SENTINEL_DATA/raw/adore/release/<member path>.parquet` (every CSV/TSV member), `$SENTINEL_DATA/processed/adore/` |
| **Loader** | `sentinel/data/sources/adore.py` — `load_challenge()`, `split_assignments()`, `train_test()`, `cv_folds()`, `chemical_overlap()` |

## What it is
Acute mortality (LC50/EC50) of fish, crustaceans and algae curated from EPA ECOTOX, expanded with chemical
descriptors/fingerprints (MACCS, PubChem, Morgan, ToxPrint bit strings; mol2vec; Mordred), species traits,
phylogenetic distances, and **ready-made split columns**. 33,448 points, 2,408 chemicals, 203 species
(`a-FCA2FCA`). Target used by SENTINEL M4: `result_conc1_mean_mol_log` = log10(L(E)C50 in mol/L) (verified equal to
log10(`result_conc1_mean_mol`) within 4e-13 in every challenge file); `result_conc1_mean_log` is the mass-unit
(log10 mg/L) robustness target.

## Challenges (computed by `scripts/data/build_adore.py`; `challenge_summary.json`, `split_overlap.csv`)

| Challenge | Rows | Chemicals | Species | `split_occurrence` train / test | Chem. overlap occurrence | Chem. overlap totally-random |
|---|---|---|---|---|---|---|
| t-F2F (fish) | 26,114 | 1,905 | 140 | 21,981 / 4,133 | 0 | **821** |
| t-C2C (crustaceans) | 6,630 | 1,444 | 17 | 5,592 / 1,038 | 0 | 401 |
| t-A2A (algae) | 704 | 174 | 46 | 619 / 85 | 0 | 49 |
| s-F2F-1 / -2 / -3 | 4,641 / 4,219 / 3,945 | 782 / 936 / 718 | 1 each | 3,928/713 · 3,550/669 · 3,320/625 | 0 | 281 / 295 / 242 |
| s-C2C | 5,196 | 1,356 | 1 | 4,373 / 823 | 0 | 370 |
| s-A2A (paper Table 3: "s-A2C", a typo) | 163 | 116 | 1 | 139 / 24 | 0 | 11 |
| a-CA2F-same | 28,912 | 991 | 198 | 6,374 / 22,538 | **991 (by design)** | 991 |
| a-CA2F-diff | 10,910 | 2,408 | 134 | 7,334 / 3,576 | 0 | 0 |
| a-FCA2FCA (all data) | 33,448 | 2,408 | 203 | 28,127 / 5,321 | 0 | 1,043 |

* t-*/s-*/a-FCA2FCA files carry 11 split columns: `split_totallyrandom`, `split_random`, `split_occurrence`,
  `split_scaffold-murcko`, `split_scaffold-generic`, and for each scaffold type `-loo-0`, `-loo-1`, `-llo`. Values:
  `0`–`4` (CV fold of a training row), `test`, and `trainvalid` (training rows without a fold, loo/llo variants).
  Train↔test chemical overlap is **0 for every column except `split_totallyrandom`** (t-F2F: 821 shared chemicals).
* a-CA2F-* files carry only `split_totallyrandom`, `split_random`, `split_occurrence`; in all three the **test set
  is every fish row** and training rows are crustaceans + algae (the columns differ only in fold assignment). In
  a-CA2F-same the 991 test chemicals are, by construction, all present in training (cross-taxon transfer for known
  chemicals); in a-CA2F-diff they are disjoint. Test species never occur in training (species overlap 0).
* SENTINEL M4 uses `split_occurrence` (primary) and `split_scaffold-murcko` (secondary, t-* only), folds 0–4 for
  selection, test rows scored once. `split_assignments.parquet` (long form: challenge, result_id, split,
  assignment, role, fold) is the only split source; nothing is re-split.

## Trivial baselines and noise floor (registry experiment `adore_trivial_baselines`, `is_baseline=True`)
Computed by `scripts/data/build_adore_baselines.py` (`sentinel/data/chemtox_baselines.py`), fit on official training
rows only; test = official test rows; CV = mean over the 5 official folds. Numbers live in
`results/registry.jsonl` (149 entries; ids in `processed/adore/trivial_baselines_registry_ids.json`) — they reproduce
docs/research/04 §A.2 exactly. Replicate noise floor (pooled within-experiment SD, keys = CAS, species, duration,
concentration type, media type, exposure type): t-F2F 0.383 log10 units (12,165 dof; 13,949 keys, 4,083 with
replicates). These are reference points, not SENTINEL results.

## QC applied
* sha256 of every downloaded resource checked against CKAN.
* Lossless conversion: every one of 31 tables re-read from the zip and compared cell by cell with its Parquet
  (233,035,335 cells, 0 mismatches; `raw/adore/conversion_check.json`). Numbers are parsed only when every non-empty
  value round-trips; ID/CAS/DTXSID/InChI/SMILES/split/bit-string columns stay strings; `True`/`False` → boolean.
* `pubchem_bits.csv` is tab-separated despite its extension (separator sniffed from the header).
* No duplicate `result_id`s and no missing targets in any challenge file.

## Known issues
* `test_cas_name` contains embedded newlines (quoted in the CSV; preserved).
* The a-CA2F "occurrence" naming does not imply chemical disjointness (see above) — report a-CA2F-same results as
  cross-taxon transfer, never as unseen-chemical generalisation.
* t-A2A has only 85 test rows (occurrence) — algae-only claims carry very wide intervals.
* ADORE is derived from an older ECOTOX release; it is not refreshed by the SENTINEL ECOTOX pipeline (by design).

## Rebuild
```bash
python -B scripts/data/download_adore.py            # CKAN -> verify sha256 -> Parquet + cell check -> manifest adore_raw (zip deleted)
python -B scripts/data/build_adore.py               # challenge stats + split assignments -> manifest adore
python -B scripts/data/build_adore_baselines.py     # trivial baselines + noise floor -> results registry (idempotent)
```
