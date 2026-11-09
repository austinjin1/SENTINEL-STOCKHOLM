# Datasheet — EPA ECOTOX Knowledgebase, aquatic endpoint table (`ecotox`)

| | |
|---|---|
| **Source** | US EPA ECOTOX Knowledgebase, quarterly ASCII release <https://gaftp.epa.gov/ecotox/ecotox_ascii_09_15_2026.zip> (newest on the index page on 2026-10-22; previous: 06_11_2026) |
| **Release / integrity** | 2026-09-15 release; zip 143,255,418 B (= Content-Length), sha256 `6aba4def7d6e405e…` (EPA publishes no checksum; recorded in manifest `ecotox_raw`), HTTP Last-Modified 2026-09-01 |
| **License** | US Government work; EPA states no explicit license ("public domain" is not stated explicitly) |
| **Retrieved** | 2026-10-22 |
| **Manifests** | `ecotox_raw` (50 tables as all-string Parquet + PDFs), `ecotox` (tidy aquatic table), `ecotox_pull_subset` (EPA standardised columns, QC only) |
| **Code** | `sentinel/data/sources/ecotox.py` (release parsing), `sentinel/data/chemtox_ecotox.py` (filtering, units, QC), `scripts/data/{download_ecotox,download_ecotox_pull,build_ecotox}.py` |

## Raw release
All 50 pipe-delimited tables (tests, results, media_characteristics, doses, dose_responses, species, chemicals,
references, … and 37 lookup tables) are stored verbatim as all-string Parquet in
`$SENTINEL_DATA/raw/ecotox/release_2026-09-15/` (121 MB); the zip is deleted after conversion. **Every table's row and
column count equals the release's own `release_notes_09_15_26.txt`** (e.g. tests 725,635 × 131, results 1,250,611 ×
137, species 29,651 × 16, chemicals 18,551 × 4, references 132,423 × 8). One results record contains `|` inside a
quoted comment (`"UNCHARACTERISED PROTEIN (TR|M7BW30|…)"`); it is re-joined verbatim (`repaired_lines = 1`), no line
is dropped. The data dictionary (`ECOTOX_ASCII_dd.pdf`) and `AsciiDownloadHelp5.pdf` are kept beside the tables.

## Tidy table `processed/ecotox/ecotox_aquatic.parquet`
One row per ECOTOX result with `tests.organism_habitat == 'Water'`, taxa **fish, crustaceans, algae, insects
(class Insecta), molluscs, amphibians** (from `species.ecotox_group` + class), endpoints **LC50/EC50/IC50**,
**EC/LC/IC 05–25**, **NOEC/LOEC/NOEL/LOEL/MATC** (asterisked and `/`-annotated codes normalised; log-transformed
endpoints excluded). Not a re-derivation of ADORE (ADORE keeps its own curated subset and splits).

**Counts** (2026-09-15 release; `processed/ecotox/summary.json`): 392,697 aquatic tests in the release → **471,097
result rows** (441,146 with a numeric value), **6,941 chemicals** (6,001 with a resolved structure), **3,955 species**,
18,120 references.

| Taxon group | Rows | Species |
|---|---|---|
| fish | 253,432 | 809 |
| crustaceans | 86,058 | 780 |
| algae | 50,207 | 875 |
| molluscs | 31,295 | 499 |
| insects | 26,341 | 806 |
| amphibians | 23,764 | 186 |

Endpoints: NOEC 156,210 · LC50 118,349 · LOEC 115,684 · EC50 39,807 · NOEL 12,824 · LOEL 7,982 · EC10 4,567 ·
MATC 3,816 · IC50 3,253 · LC10 2,682 · EC20 1,746 · others < 1,000 each. Exposure class: chronic 309,688 · acute
141,056 · prolonged 16,515 · unknown 3,838. Water type: freshwater 363,806 · saltwater 71,325 · culture 33,257 ·
unknown 2,709. Of the rows with a value, 91.4% have `conc_mg_L` and 86.0% `conc_mol_L` (mass + molar 376,628 ·
mass only 26,679 · molar only 2,941 · not convertible 34,898). `mw_source` over all rows: chemical MW 348,037 ·
element atomic weight via `ion1` 87,834 · none 35,226. 21,527 values are censored (16,650 right, 4,877 left);
2,619 are approximate.

Columns (main): `result_id`, `test_id`, `reference_number`, `cas` (validated, hyphenated), `chemical_name`,
`ecotox_dtxsid`, `chem_dtxsid`/`chem_smiles`/`chem_inchikey`/`chem_structure_class` (from `chemicals`), taxonomy
(`taxon_group`, `latin_name`, `common_name`, `kingdom`, `phylum_division`, `class`, `tax_order`, `family`, `genus`,
`species`, `ncbi_taxid`), `organism_lifestage`, `endpoint` (raw) / `endpoint_norm` / `endpoint_group` /
`endpoint_flag`, `effect` / `effect_norm`, `measurement`, `exposure_class`, `obs_duration_mean` + unit and
`duration_h`, `exposure_duration_h`, `exposure_type`, `media_type` / `water_type`, `test_location`, `conc1_type`,
`ion1`, `conc1_mean_op` / `conc1_mean` / `conc1_min` / `conc1_max` / `conc1_unit` (raw text), `conc_value` +
`conc_value_flag`, `censored` + `censor_kind`, `conc_basis`, `conc_unit_kind`, **`conc_mg_L`, `conc_mol_L`**
(+ `_min`/`_max` variants, `log10_conc_mg_L`, `log10_conc_mol_L`), `conc_conversion`, `mw_used_g_mol` + `mw_source`,
media pH / temperature / hardness / DO / salinity (raw + numeric), `publication_year`, `reference_doi`.
`references_aquatic.parquet` holds the cited references; `unit_inventory.csv` lists every concentration unit with
its row count and handling.

## Standardisation rules (unit-tested)
* **Concentration** — unit = optional basis prefix (`AI` active ingredient, `ae` acid equivalent) + core. Mass/volume
  cores → mg/L with exact SI factors (g, mg, ug, ng, pg over L, ml, ul, dm3, cm3, m3); `ppm` = mg/L and `ppb` = ug/L
  (w/v, dilute aqueous media); molar cores (M, mM, uM, nM, pM, mol/L, mmol/L, umol/L, nmol/L, mmol/m3, …) → mol/L.
  mg/L ↔ mol/L uses the MW of the substance *as expressed*: the element's standard atomic weight when `ion1` is a
  single element (e.g. `Cu` → mg Cu/L), otherwise the RDKit MW of the tested chemical (incl. counter-ions) from the
  `chemicals` table; formulation (`conc1_type = F`) values are converted like ADORE does — filter on `conc1_type` if
  needed. Not converted (kept, `conc_conversion = not_convertible`): v/v (`ul/L`, `ml/L`), `%`, per-kg, per-area,
  per-organism, diet units, `0/00`, units with trailing qualifiers (`mg/L CaCO3`), and **`ppt`**, which the ECOTOX
  code table defines as parts per *trillion* but which is also used for parts per thousand (e.g. a 42 ppt LC50 for
  *Macrobrachium rosenbergii*), so it is ambiguous.
* **Values** — `conc1_mean` only; ranges are never averaged into a value; a trailing `*` is stripped and flagged;
  `NR`/`NC`/`--` → NaN (`missing_code`).
* **Censoring** — `<`, `<=` left-censored; `>`, `>=` right-censored; `~`, `ca` approximate (not censored); values
  are never substituted.
* **Duration** — clock units (s, mi, h, d, wk, mo = 30.4375 d, yr = 365.25 d) → hours; developmental time units
  (hpf, dph, dpf, …) keep NaN hours.
* **Exposure class** — acute = L(E)C50/IC50 ≤ 96 h; prolonged = L(E)C50/IC50 > 96 h; chronic = NOEC/LOEC/NOEL/LOEL/
  MATC/ECx/LCx (x ≤ 25); unknown = median endpoint without duration.
* **Codes** — ECOTOX appends `/` when a comment exists (`FW/`) and `~` to approximate effects (`~MOR`); `*_norm`
  columns strip these; `water_type` from `media_type` (FW, SW, CUL, AQU, SED, …).

## QC — independent cross-check against EPA's own standardisation
`download_ecotox_pull.py` keeps only EPA's standardised columns from `data-pulls/aquatic_all.zip` (data version
20250911, 694,662 rows; QC reference only, never used as data). 451,330 tidy rows share a result id with it:
* **Concentration**: 383,545 rows where EPA gives mg/L and we have mg/L → **375,053 agree within 0.1% (97.8%)**.
  Of 8,492 differences, 5,825 are molar values expressed as an ion (we use the element's atomic weight, e.g. mg Zn/L;
  EPA uses the parent salt's MW), 2,591 molar values converted with a different MW, and only **76** are plain mass
  units — there EPA's value itself differs (e.g. 75.4 mg/L vs EPA 7.54 AI mg/L: active-ingredient/purity
  adjustments or revisions between the 2025-09 pull and the 2026-09 release). EPA has mg/L where we do not for 3,160
  rows (2,240 molar values without a known MW, 918 units we deliberately leave unconverted, e.g. mg/kg).
* **Duration**: 423,629 comparable rows, 98.9% agree (differences are mostly EPA mapping developmental time units).

## Known issues
* ECOTOX mixes study designs (lab/field, static/flow-through, measured/nominal); filter on `test_location`,
  `exposure_type`, `chem_analysis_method`, `conc1_type` as the application requires.
* Many results are censored or missing values; NOEC/LOEC dominate the chronic rows.
* `water_type` is missing where `media_type` is NR/NC.
* Molar values for ion-expressed concentrations are per mole of the element (e.g. mol Cu/L).

## Rebuild
```bash
python -B scripts/data/download_ecotox.py          # newest release -> verified Parquet tables (zip deleted)
python -B scripts/data/download_ecotox_pull.py     # optional QC reference (EPA standardised columns only)
python -B scripts/data/build_chemicals.py          # CAS -> DTXSID -> structure / MW (needs ADORE + NRC built)
python -B scripts/data/build_ecotox.py             # tidy aquatic table (+ EPA cross-check if the pull subset exists)
```
