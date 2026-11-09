# Datasheet — USCG National Response Center incident reports (`nrc_incidents`)

| | |
|---|---|
| **Source** | US Coast Guard National Response Center, FOIA files <https://nrc.uscg.mil/FOIAFiles/CY{YY}.xlsx> (one workbook per calendar year of report receipt, CY90 … CY26) + `DataDictionary.xlsx` |
| **License** | US Government work; no license stated. The NRC homepage states the files contain **initial, unvalidated** reports. |
| **Retrieved** | 2026-10-22 (37 workbooks, 1990–2026; CY26 Last-Modified 2026-10-20; sha256 + Last-Modified per file in manifest `nrc_raw` and `raw/nrc/downloads.json`) |
| **Manifests** | `nrc_raw` (workbooks, kept: each < 20 MB), `nrc_incidents` (tidy tables + sheet archive) |
| **Code** | `sentinel/data/sources/nrc.py` (parsing, cleaning), `sentinel/data/chemtox_nrc.py` (joins, coverage), `scripts/data/{download,build}_nrc.py` |

## Structure
Each workbook has ten sheets keyed by `SEQNOS` (NRC report number): CALLS, INCIDENT_COMMONS, INCIDENT_DETAILS,
INCIDENTS, MATERIAL_INVOLVED, MATERIAL_INV0LVED_CR (sic), TRAINS_DETAIL, DERAILED_UNITS, VESSELS_DETAIL,
MOBILE_DETAILS (columns identical across years; CY23 INCIDENT_COMMONS carries 50 empty `ColumnN` fields, dropped).
All sheets are archived as strings in `$SENTINEL_DATA/interim/nrc/sheets/<SHEET>.parquet` (with `source_file`).

Outputs in `$SENTINEL_DATA/processed/nrc_incidents/`:

| File | Grain | Content |
|---|---|---|
| `reports.parquet` | 1 row / report (1,033,999) | report id, file year, receipt time, incident time (local), incident type, cause, description, location (city/county/state/ZIP), `lat`/`lon` + flags, medium, body of water, tributary, water-supply-contaminated, waterway closed, evacuations, injuries, hospitalisations, fatalities, damages, release duration/rate, sheen, report-level material flags, `is_drill` |
| `materials.parquet` | 1 row / (report, material) (1,067,348) | CHRIS code, CAS (raw + validated), UN number, material name, amount released + unit (+ litres/kg), if-reached-water, amount in water + unit (+ litres/kg), qualifiers |
| `nrc_incidents.parquet` | 1 row / (report, material); reports without a material row once (1,149,079) | flat join of the two + CAS → chemical identity (`chem_dtxsid`, `chem_name`, `chem_smiles`, `chem_inchikey`, `chem_mw_g_mol`, `chem_structure_class`) |
| `coverage_by_year.csv`, `summary.json` | per file year / overall | coverage statistics below |

## Cleaning rules (all unit-tested in `tests/test_chemtox_data.py`)
* **CAS**: `000000-00-0` placeholder (695,916 material rows) → missing; zero-padded forms normalised (`000071-43-2` →
  `71-43-2`); check digit verified (28 malformed strings such as `10699-0`, `D007` → missing, raw kept in `cas_raw`).
* **Amounts**: unit `UNKNOWN AMOUNT`/`UNKNOWN`/`NONE` → amount missing, unit kept as qualifier (1990s files pair
  `NONE` with 0). Volumes → litres and masses → kg with exact US-customary factors (gallon 3.785411784 L,
  barrel 158.987294928 L, pound 0.45359237 kg, short ton 907.18474 kg, …); `OUNCE(S)` read as US fluid ounce. Gas
  volumes (MCF/MMCF), counts (EACH), PPM/PPB and radioactivity units are kept unconverted.
* **Flags**: Y/N/U (and YES/NO/UNKNOWN/UNK) → nullable boolean, U → missing; raw code kept in `*_raw`.
* **Coordinates**: deg/min/sec + quadrant → signed decimal degrees (blank min/sec = 0). Blank quadrant → state's
  hemisphere (N/W; S for AS; E for GU/MP), flagged `quadrant_inferred`; wrong quadrant letter (`N` on a longitude)
  → `bad_quadrant`, no value; 0/0/0 → `zero`; min/sec ≥ 60 or deg out of range → `out_of_range`. Decoded points
  outside the US + territories extent (5 boxes in `US_EXTENT_BOXES`) keep `lat_decoded`/`lon_decoded` but get
  `lat`/`lon` = NaN (`coord_flag = outside_us_extent`). `has_coordinates` = valid and inside the extent.
  Overall `coord_flag`: missing 903,420 · ok 125,038 · quadrant_inferred 3,846 · outside_us_extent 992 · invalid 651.
* **Times**: wall-clock local times without a zone → naive `*_local` timestamps (not UTC); raw strings kept; 55
  reports have unparseable incident times (e.g. `6/11/205`).
* **Duplicates**: no SEQNOS appears in two workbooks; 13 repeated SEQNOS inside INCIDENT_DETAILS (CY17: 9, CY26: 4)
  → last row kept.
* **Privacy**: responsible company names, caller fields and street-address fields are not carried into the tidy
  tables (they remain only in the raw workbooks/sheet archive).

## Coverage (computed by `build_nrc.py`; `summary.json`, `coverage_by_year.csv`)
"Water-reaching" is reported two ways: `medium_desc == WATER` (incident level) and ≥ 1 material with
`IF_REACHED_WATER = YES`.

| Scope | Water-reaching reports | with coordinates given | with valid coordinates (in US extent) | with a valid CAS |
|---|---|---|---|---|
| CY24, medium = WATER | 12,249 | 2,828 (23.1%) | 2,804 (22.9%) | 249 (2.0%) |
| CY24, material reached water | 11,739 | 2,666 (22.7%) | 2,645 (22.5%) | 233 (2.0%) |
| 1990–2026, medium = WATER | 522,588 | 111,112 (21.3%) | 109,807 (21.0%) | 7,416 (1.4%) |
| 1990–2026, material reached water | 637,941 | 108,562 (17.0%) | 107,308 (16.8%) | 7,190 (1.1%) |

The research figure "~23% of 2024 water incidents have coordinates" is confirmed (the research counted reports with
a latitude degree present: 2,830; both latitude and longitude present: 2,828; valid and plausible: 2,804). Only 11
CY24 water reports have both a CAS number and coordinates. Water incidents are dominated by petroleum products
(unknown oil, diesel, crude, hydraulic oil, gasoline …) that carry no CAS; the most frequent CAS numbers among
water-reaching materials are ethylene glycol (107-21-1), sulfuric acid, sodium hydroxide, benzene, ammonia.

### CAS → chemical identity join (`chemicals` dataset)
80,723 material rows carry a valid CAS (417 distinct numbers); all are in the `chemicals` table and 80,576 rows
(360 distinct CAS) resolve to a structure. Among water-reaching material rows, 7,394 have a valid CAS and 7,352 a
structure (organic 4,276 · inorganic 3,048 · organic multicomponent 27 · organometallic 1), 42 are `no_structure`
(petroleum products/mixtures with a CAS). In CY24, 247 of the 249 water reports with a CAS resolve to a structure.
The spill-severity module can therefore score only ~1–2% of water-reaching reports from structure; oil/petroleum
reports (the majority) need a separate, product-class route.

## Known issues
* Initial, unvalidated reports; duplicates of the same event under different SEQNOS are possible; quantities are
  often "UNKNOWN AMOUNT" (amount in water known for 226,907 of 681,313 reached-water material rows).
* **1990–1999 files contain no CAS numbers**, and their `IF_REACHED_WATER` is `YES` for most materials even when
  the medium is air or land (e.g. 1990: 25,218 reached-water reports vs 13,740 with medium WATER) — use
  `medium_desc` for those years; CHRIS codes (present throughout) are the route to identities before 2000.
* `file_year` is the year of report receipt; incident dates can precede it (and a few are corrupt, e.g. 1920).
* 1,632 drill calls (`is_drill`) are kept and included in the coverage counts (they are not separable from
  incidents in the research comparison).
* Coordinates are self-reported; geocoding of `incident_location`/`location_nearest_city` is needed for most
  reports (≈ 77–79% of water reports have none).

## Rebuild
```bash
python -B scripts/data/download_nrc.py      # 37 workbooks + dictionary; re-downloads only changed files
python -B scripts/data/build_nrc.py         # tidy tables (run again after build_chemicals.py for the CAS join)
```
