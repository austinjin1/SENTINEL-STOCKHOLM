# Datasheet: `wqp_discrete` — Water Quality Portal discrete samples (WQX 3.0)

| | |
|---|---|
| **Name** | `wqp_discrete` (processed) / `wqp_raw` (raw parsed) |
| **Source** | Water Quality Portal (USGS + EPA + National Water Quality Monitoring Council), https://www.waterqualitydata.us — WQX 3.0 profiles (`/wqx3/`) |
| **Endpoints** | `POST https://www.waterqualitydata.us/wqx3/Result/search?mimeType=csv&zip=yes&sorted=no` (JSON body, `dataProfile=fullPhysChem`, `sampleMedia=Water`); `POST .../wqx3/Station/search`; planning only: legacy `GET .../data/summary/monitoringLocation/search?dataProfile=periodOfRecord` |
| **License** | US public domain / open government data. Data are owned by the ~1,000 submitting organizations; cite the WQP (Read et al. 2017, *WRR* 53:1735, doi:10.1002/2016WR019993) and the organizations (`organization` column). |
| **Retrieval** | 2026-10-22 06:12 EDT to 2026-10-23 20:59 EDT (targets 10-22/23, physchem 10-23, stations 10-23) |
| **Spatial scope** | 52 jurisdictions: 48 contiguous states + DC, Alaska, Hawaii, Puerto Rico (`statecode` list in every request) |
| **Temporal scope** | Activity start date 2000-01-01 → 2026-10-23 (inclusive) |
| **Manifests** | `$SENTINEL_DATA/manifests/wqp_raw.json`, `$SENTINEL_DATA/manifests/wqp_discrete.json` (sha256 per file, request parameters, counts) |
| **Code** | `sentinel/data/sources/wqp.py` (client + parser), `sentinel/data/wqp_harmonize.py` (variables, units), `sentinel/data/wqp_qc.py` (QC), `scripts/data/download_wqp.py`, `scripts/data/build_wqp_discrete.py`, tests `tests/test_wqp.py` |
| **Synthetic data** | None |

## Files

```
$SENTINEL_DATA/raw/wqp/
  results/group=<g>/<g>_<start>_<end>.parquet   tidy parsed rows, one file per (group x date window), all 52 states
  physchem/physchem_<start>_<end>.parquet       DO/pH/temperature rows at site-dates with a target result
  locations/<chunk>.parquet                     site fields embedded in result rows
  summary/<ST>.parquet                          legacy (WQX 2.2) per site/year/characteristic counts (reference only)
  stations.parquet                              Station/search + result-embedded locations
  ledger.jsonl, download.log                    request ledger (resume) and log
$SENTINEL_DATA/processed/wqp_discrete/
  variable=<name>/data.parquet                  analysis-ready rows (hive partition by variable)
  _stations.parquet                             sites referenced by processed rows
  _unmapped.parquet, _unmapped_summary.csv      rows that could not be mapped/converted (+ reason)
  _qc_rejects.parquet                           rows removed by QC (+ first failing step)
  _qc_report.csv/.json                          per-variable counts before/after every QC step
  _coverage_by_variable.csv / _by_state.csv / _by_year.csv, _completeness_by_group_year.csv
```

Read with e.g. `pd.read_parquet(paths.processed("wqp_discrete"), filters=[("variable", "=", "tp")])`
(files and folders starting with `_` are ignored by the hive-partition reader).

### Processed columns

`site_id` (WQX MonitoringLocationIdentifier, string), `state` (postal), `huc8` (string), `lat`, `lon`,
`site_type`, `site_class` (stream / lake / estuary_ocean / wetland / groundwater / facility / other /
unknown), `activity_start_date` (local date), `activity_start_time`, `activity_tz`, `time_utc`
(only when time and a known zone code are present), `depth_m`, `activity_relative_depth`,
`value` (canonical unit; NaN for censored rows), `unit`, `censored`, `censor_type` (`<` / `>`),
`detection_limit` (canonical unit), `detection_limit_type`, `value_original`, `unit_original`,
`characteristic`, `sample_fraction`, `fraction_class`, `species_used`, `speciation_assumed`,
`method_id`, `method_context`, `activity_type`, `media_subdivision`, `hydrologic_event`,
`hydrologic_condition`, `provider` (USGS / STORET / STEWARDS), `organization`, `project_id`,
`activity_id`, `result_id`, `status`, `status_class` (accepted / preliminary / unknown),
`qualifier`, `value_type`, `statistical_base`, `result_comment`, `last_change_date`, `usgs_pcode`.

## How it was collected

* **Chunking.** One request per (characteristic group x date window) covering all 52 jurisdictions
  as a `statecode` list: monthly windows for optical, phosphorus, nitrogen, ions, bacteria;
  quarterly for carbon; half-years for cyanotoxins; semi-monthly for physchem (1,779 target +
  644 physchem requests, <= 2 concurrent). *Deviation from the brief's "state x group x
  year-range":* measured 2026-10-22, state-filtered requests over >= 1-year ranges take 20-45 s to
  the first row and the server truncates any query whose first row takes > ~60 s (appending
  `ERROR: INCOMPLETE DATA ...`; e.g. Ohio nitrogen 2021, Alaska optical 2015-24 failed), whereas
  short windows over the 52-state list start streaming in 3-10 s.
* **Truncation handling.** Responses ending with the INCOMPLETE trailer are split in half (dates,
  then names) recursively — 342 splits; no partial response is stored.
* **Transient errors.** Exponential backoff, `Retry-After` honoured. A WQP code-list outage
  (2026-10-22 07:03-07:06 EDT) returned bogus HTTP 400s for 152 chunks; they were re-fetched in a
  second pass and the client now retries that error. Final ledger: 0 failed chunks.
* **Profile.** `fullPhysChem` (181 columns, incl. `LastChangeDate`, hydrologic event, temperature
  basis); 53 columns are kept in the raw table.
* **Physchem (DO, pH, water temperature).** Downloaded for all states, stored only for (site,
  local date) pairs that also have a target result; the processed table additionally requires a
  QC-passed target result. Requesting by site list was tested and rejected (POST bodies above
  ~100 site IDs are refused; ~5,000 extra requests to save ~25 % of rows).
* **Completeness reference.** Legacy WQX 2.2 period-of-record summary (site x year x
  characteristic counts) per state.

## Characteristic groups requested and variable mapping

Exact requested names: `sentinel/data/sources/wqp.py::CHARACTERISTIC_GROUPS`; routing by
characteristic, sample fraction, unit, speciation and method: `sentinel/data/wqp_harmonize.py`.
Not requested: probe/sensor names (`Chlorophyll a (probe...)`, `Phycocyanin (probe...)`, FDOM),
periphyton chlorophyll/pheophytin, gene targets. Fraction classes: *total* = Total / Unfiltered /
Total Recoverable; *dissolved* = Dissolved / Filtered (field, lab, field and/or lab);
*particulate* = Suspended / Non-Filterable; *unspecified* = blank / None.

| variable | unit | plausible range | WQX CharacteristicName(s) |
|---|---|---|---|
| `chla` | ug/L | 0 – 10000 | Chlorophyll a; Chlorophyll A; Chlorophyll a - Phytoplankton (suspended) |
| `chla_corrected` | ug/L | 0 – 10000 | Chlorophyll a, corrected for pheophytin; Chlorophyll a, free of pheophytin |
| `chla_uncorrected` | ug/L | 0 – 10000 | Chlorophyll a, uncorrected for pheophytin |
| `pheophytin` | ug/L | 0 – 5000 | Pheophytin a; Pheophytin; Phaeophytin - Phytoplankton (suspended); Pheopigments of Chlorophyll; Phaeopigment |
| `tss` | mg/L | 0 – 100000 | Total suspended solids; Total Suspended Particulate Matter |
| `ssc` | mg/L | 0 – 500000 | Suspended Sediment Concentration (SSC); Suspended sediment concentration (SSC) |
| `turbidity_ntu` | NTU | 0 – 10000 | (via unit routing, see below) |
| `turbidity_fnu` | FNU | 0 – 10000 | (via unit routing, see below) |
| `turbidity_ntru` | NTRU | 0 – 10000 | (via unit routing, see below) |
| `secchi` | m | 0.01 – 50 | Depth, Secchi disk depth; Secchi depth; Water transparency, Secchi disc |
| `transparency_tube` | cm | 0.5 – 125 | Transparency, tube with disk; Transparency, Secchi tube with disk |
| `acdom440` | 1/m | 0 – 100 | Absorbance at 440 nm |
| `cdom_abs` | 1/m | 0 – 200 | Colored dissolved organic matter (CDOM) |
| `phycocyanin` | ug/L | 0 – 10000 | Phycocyanin; Phycocyanins; Algae-Phycocyanin |
| `tp` | mg/L as P | 0 – 50 | Phosphorus (fraction total); Total Phosphorus, mixed forms (fraction total); Phosphorus as P (fraction total); Phosphorus (fraction unspecified); Total Phosphorus, mixed forms (fraction unspecified); Phosphorus as P (fraction unspecified) |
| `tdp` | mg/L as P | 0 – 50 | Phosphorus (fraction dissolved); Total Phosphorus, mixed forms (fraction dissolved); Phosphorus as P (fraction dissolved) |
| `orthop` | mg/L as P | 0 – 50 | Orthophosphate; Orthophosphate as P; Orthophosphate as PO4; Ortho-Phosphate-Phosphorus; Soluble Reactive Phosphorus (SRP); Phosphate-phosphorus (fraction dissolved); Phosphate-phosphorus as P (fraction dissolved); Phosphate-phosphorus as PO4 (fraction dissolved); Phosphate (fraction dissolved) |
| `tn` | mg/L as N | 0 – 100 | Nitrogen (fraction total); Total Nitrogen, mixed forms (fraction total); Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3) (fraction total); Total Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3) (fraction total); Nitrogen (fraction unspecified); Total Nitrogen, mixed forms (fraction unspecified); Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3) (fraction unspecified); Total Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3) (fraction unspecified) |
| `tdn` | mg/L as N | 0 – 100 | Nitrogen (fraction dissolved); Total Nitrogen, mixed forms (fraction dissolved); Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3) (fraction dissolved); Total Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3) (fraction dissolved) |
| `no3` | mg/L as N | 0 – 100 | Nitrate; Nitrate as N; Nitrate-N; Nitrate-nitrogen; Nitrate-Nitrogen |
| `no2` | mg/L as N | 0 – 20 | Nitrite; Nitrite as N; Nitrite-N |
| `no3no2` | mg/L as N | 0 – 100 | Inorganic nitrogen (nitrate and nitrite); Inorganic nitrogen (nitrate and nitrite) as N; Nitrate + Nitrite; Nitrate-Nitrite; Inorganic nitrogen (nitrate and nitrite) ***retired***use Nitrate + Nitrite |
| `nh3` | mg/L as N | 0 – 100 | Ammonia; Ammonia and ammonium; Ammonia as NH3; Ammonia-nitrogen; Ammonia-nitrogen as N; Ammonium; Ammonium as N; Ammonium as NH4; Ammonium-nitrogen; Ammonium-Nitrogen; Nitrogen, ammonia (NH3) as NH4 |
| `tkn` | mg/L as N | 0 – 200 | Kjeldahl nitrogen (fraction total); Total Kjeldahl nitrogen (fraction total); Total Kjeldahl nitrogen (Organic N & NH3) (fraction total); Nitrogen Kjeldahl (fraction total); Kjeldahl nitrogen (fraction unspecified); Total Kjeldahl nitrogen (fraction unspecified); Total Kjeldahl nitrogen (Organic N & NH3) (fraction unspecified); Nitrogen Kjeldahl (fraction unspecified) |
| `doc` | mg/L as C | 0 – 500 | Organic carbon (fraction dissolved); Non-purgeable Organic Carbon (NPOC) (fraction dissolved) |
| `toc` | mg/L as C | 0 – 500 | Organic carbon (fraction total); Non-purgeable Organic Carbon (NPOC) (fraction total) |
| `chloride` | mg/L | 0 – 250000 | Chloride |
| `sulfate` | mg/L as SO4 | 0 – 100000 | Sulfate; Sulfate as SO4; Sulfate as S; Total Sulfate |
| `spcond` | uS/cm | 0 – 250000 | Specific conductance; Specific conductivity; Specific conductivity***retired***use Specific conductance |
| `ecoli` | CFU or MPN/100mL | 0 – 1e+08 | (via unit routing, see below) |
| `ecoli_qpcr` | CCE/100mL | 0 – 1e+09 | (via unit routing, see below) |
| `enterococcus` | CFU or MPN/100mL | 0 – 1e+08 | (via unit routing, see below) |
| `enterococcus_qpcr` | CCE/100mL | 0 – 1e+09 | (via unit routing, see below) |
| `fecal_coliform` | CFU or MPN/100mL | 0 – 1e+08 | Fecal Coliform |
| `microcystins` | ug/L | 0 – 100000 | Microcystin; Microcystins and nodularins; Total microcystins plus nodularins; Microcystins/nodularin |
| `microcystins_dissolved` | ug/L | 0 – 100000 | Microcystin (fraction dissolved); Microcystins and nodularins (fraction dissolved); Total microcystins plus nodularins (fraction dissolved); Microcystins/nodularin (fraction dissolved); Dissolved microcystins plus nodularins |
| `microcystins_particulate` | ug/L | 0 – 100000 | Microcystin (fraction particulate); Microcystins and nodularins (fraction particulate); Total microcystins plus nodularins (fraction particulate); Microcystins/nodularin (fraction particulate); Particulate microcystins/nodularin |
| `microcystin_lr` | ug/L | 0 – 100000 | Microcystin LR |
| `microcystin_rr` | ug/L | 0 – 100000 | Microcystin RR |
| `microcystin_yr` | ug/L | 0 – 100000 | Microcystin YR |
| `microcystin_la` | ug/L | 0 – 100000 | Microcystin LA |
| `microcystin_other_congener` | ug/L | 0 – 100000 | Microcystin LF; Microcystin LW; Microcystin LY; Microcystin WR; Microcystin HilR; Microcystin HiLR; Microcystin HtyR; Microcystin HtYR; Microcystin dLR; Microcystin dRR; Microcystin LR [DAsp3]; Microcystin LR, Desmethyl; Microcystin LR [Dha7]; Microcystin LR [D-Leu1]; Microcystin RR [DAsp3] |
| `cylindrospermopsin` | ug/L | 0 – 10000 | Cylindrospermopsin; Cylindrospermopsins |
| `anatoxin_a` | ug/L | 0 – 10000 | Anatoxin-A; Anatoxin a; Anatoxin |
| `saxitoxin` | ug/L | 0 – 10000 | Saxitoxin; Saxitoxins |
| `do` | mg/L | 0 – 30 | (via unit routing, see below) |
| `do_pctsat` | % | 0 – 300 | (via unit routing, see below) |
| `ph` | pH | 1 – 13 | pH; PH |
| `temp` | deg C | -2 – 50 | Temperature, water |

* **(unmapped: ambiguous)**: Chlorophyll; Escherichia
* **turbidity_ntu / _fnu / _ntru (by unit)**: Turbidity; Turbidity Field
* **(flag: Secchi right-censored)**: Depth, Secchi disk visible at bottom (Y/N) (choice list)
* **spcond (only if 25 °C basis)**: Conductivity
* **ecoli / ecoli_qpcr (by unit + method)**: Escherichia coli
* **enterococcus / enterococcus_qpcr**: Enterococcus; Enterococci
* **do / do_pctsat (by unit)**: Dissolved oxygen (DO); Dissolved oxygen; Dissolved Oxygen***retired***use Dissolved Oxygen (DO)

Notes: TP/TN/TKN accept an unstated fraction (treated as total, `fraction_class="unspecified"` kept
so it can be filtered); DOC/TOC do not (unstated → unmapped). "Phosphate-phosphorus" with a *Total*
fraction is mapped to `tp` (legacy STORET "PHOSPHATE, TOTAL"; in the year-2000 pull its values
match TP — median 0.08–0.13 mg/L vs 0.02–0.03 for orthophosphate, 1.8× orthophosphate in 3,464
same-sample pairs); filtered/unstated "Phosphate-phosphorus" is ambiguous (TDP vs ortho-P) and
unmapped. `Chlorophyll` (no "a") is ambiguous and unmapped.

## Raw download totals (final)

| group | WQX3 rows (raw, all QC classes) |
|---|---|
| optical | 17,931,173 |
| phosphorus | 10,321,124 |
| nitrogen | 19,488,836 |
| carbon | 2,168,922 |
| ions | 19,394,088 |
| bacteria | 8,861,256 |
| cyanotoxins | 291,187 |
| **targets total** | **78,456,586** |
| physchem downloaded / kept (co-sampled site-dates) | 83,071,416 / 47,951,964 |

Raw Parquet: 6.1 GB (`results` 2.9 GB, `physchem` 1.4 GB, `locations` 1.7 GB); 560,888 stations.
Processed: ~2.5 GB. Total for this dataset well under the 25 GB budget.

Build note: WQX3 leaves `Location_StatePostalCode` blank for ~303k stations (e.g. FL, AZ, MI, TN
STORET sites). The build recovers the state from the per-state Station/search response; rows whose
state is still unknown are kept (`state = ""`) because every request was restricted to the 52
statecodes.

## Units (canonical unit per variable; explicit factors, all tested in `tests/test_wqp.py`)

| conversion | factor |
|---|---|
| ug/L, ppb, mg/m3, ng/mL → ug/L | 1; mg/L, ppm → ug/L ×1000; ng/L ×0.001 |
| nitrate/nitrite+nitrate/TN as NO3 → as N | ×0.22590 (14.0067/62.0049) |
| nitrite as NO2 → as N | ×0.30446 |
| ammonium as NH4 → as N; ammonia as NH3 → as N | ×0.77649; ×0.82245 |
| (ortho)phosphate as PO4 → as P | ×0.32614 |
| sulfate as S → as SO4 | ×2.99587 |
| µmol/L → mg/L (as N, P, C, Cl, SO4) | × molar mass/1000; meq/L via charge (Cl 1, SO4 2, NO3/NO2/NH4 1) |
| Secchi ft / cm / in → m | ×0.3048 / 0.01 / 0.0254 |
| transparency tube in / m → cm | ×2.54 / 100 |
| temperature °F → °C; K → °C | (F−32)×5/9; K−273.15 |
| conductance mS/cm, mmho/cm → µS/cm; S/m | ×1000; ×10,000 (µmho/cm = µS/cm) |
| bacteria per mL / per L / per dL → per 100 mL | ×100 / ×0.1 / ×1 |
| decadic absorbance per cm at 440 nm → a_CDOM(440) m⁻¹ | ×230.26 (= ln10 × 100) |
| turbidity | **never converted**: NTU, FNU, NTRU are separate variables; JTU, FTU, FNRU, FAU, etc. unmapped |

Speciation is taken from the unit string, `Result_MethodSpeciation` or the characteristic name;
conflicting sources → unmapped. Where none is given, nitrate/nitrite/ammonia/NO3+NO2 are assumed
"as N", orthophosphate "as P", sulfate "as SO4" (`speciation_assumed=True`). E. coli / enterococci
culture (CFU, MPN, # per 100 mL) and qPCR (CCE/100 mL) are separate variables; a qPCR method
(EPA 1609/1611/1696, "qPCR") with a culture-looking unit is unmapped, never put into the culture
variable. Counts without a volume ("MPN", "CFU") are unmapped. Sensor units (RFU, QSE) are unmapped.

## QC (in order; counts per step and variable in `_qc_report.csv` and below)

1. non-water media; 2. QC activity types (blanks, spikes, replicates, duplicates, reference
samples — flagged at parse time in `qc_class`); 3. data-logger activity types, time-aggregated
results (`Result_TimeBasis` set), summary statistics (daily/weekly min/max/median/geometric
mean…), and logger-like series posted as field measurements (> 4 distinct values of one
characteristic at the same site/date/time/depth, or > 12 distinct times per day for
DO/pH/temperature); 4. status Rejected; 5. sentinel codes (−9,999,999 … −99, 999,999+ masked to
NaN at parse time, original kept in `value_text`); 6. no numeric value and not censored (or
censored without any limit); 7. plausible range (table above; censored rows are checked on their
limit); 8. speciation duplicates (same activity reported in two speciations, e.g. USGS P/PO4 —
converted copy dropped); 9. duplicates (same variable, characteristic, site, local date+time,
depth, value to 6 significant figures, censoring and limit; rows without a time must also share
the activity; keep accepted > preliminary, then latest `LastChangeDate`); 10. physchem only:
no QC-passed target result at the same site on the same date.

Censoring: `<` from detection conditions (Not Detected, Below Reporting Limit, Present Below
Quantification Limit…), leading `<` in the value, or qualifier U/ND; `>` from "Present Above
Quantification Limit", leading `>` (e.g. E. coli > 2419.6 MPN), Secchi/tube `StatisticalBase =
Minimum` (disk visible on bottom / tube full) or a "Secchi disk visible at bottom = Y" record in the
same activity. Censored rows have `value = NaN` and the limit in `detection_limit` (no DL/2
substitution).

## Completeness check (WQX3 rows downloaded vs legacy WQX 2.2 summary)

2000–2023: 0.95–0.99 of the legacy count for every group (the legacy summary counts all media;
requests here use `sampleMedia=Water`; WQX3 also re-keys some records). Physchem 1.00. 2024–2026:
1.10–1.35 (USGS data added after 2024-03-11 exist only in WQX3), confirming the post-2024 USGS
gap is filled. Per group × year: `_completeness_by_group_year.csv`.

## Known issues / limits

* **WQX3 is labelled beta** by WQP; the server truncates slow queries (handled) and had a short
  code-list outage (handled). Results are a snapshot of 2026-10-22/23; WQP data are revised
  continuously (`last_change_date`).
* **Cross-organization duplicates are not removed**: states re-upload USGS or partner data under
  their own site IDs; same-site duplicates only are removed. A spatial (site-distance + same
  date/value) dedup is needed before pooling across organizations.
* **Uneven sampling**: programs are event- and region-driven (e.g. Florida dominates
  cyanotoxins; Minnesota and Wisconsin dominate Secchi/transparency tube; beach programs dominate
  enterococci). Weight or cap by state/organization for national models.
* `time_utc` is missing when the time or zone code is missing; the local `activity_start_date` is
  always present.
* Discrete chlorophyll includes rare unit-mislabelled values that pass the 0–10,000 µg/L range;
  plausible-range checks are broad by design (unit-error hunting per site is left to modelling).
* Unmapped rows (counts below) include "Conductivity" without a 25 °C basis, bacteria counts
  without volume, JTU/FTU/FNRU turbidity, DOC/TOC with an unstated fraction, and blank units.
* Site types include wells and facilities (`site_class`); filter to stream/lake/estuary for
  ambient surface-water work.
* Physchem rows are co-sampled at the day level (same site and local date), not necessarily the
  same activity/time.

## Rebuild

```bash
cd SENTINEL-STOCKHOLM
PY=/home/bcheng/miniconda3/envs/sentinel2/bin/python
$PY -B scripts/data/download_wqp.py summary  --workers 2   # ~12 min
$PY -B scripts/data/download_wqp.py results  --workers 2   # ~20 h (rerun to retry failed chunks)
$PY -B scripts/data/download_wqp.py physchem --workers 2   # ~16 h
$PY -B scripts/data/download_wqp.py stations --workers 2   # ~12 min
$PY -B scripts/data/download_wqp.py manifest
$PY -B scripts/data/build_wqp_discrete.py                   # ~3.2 h; writes processed + manifest
$PY -B -m pytest -q -p no:cacheprovider tests/test_wqp.py   # offline unit tests
```
(`download_wqp.py all` runs the five download phases in order; every phase is resume-safe.)


## Processed coverage, QC counts and unmapped rows

## Coverage by variable

| variable | unit | rows | sites | states | first_year | last_year | censored_frac | usgs_frac | median_value |
|---|---|---|---|---|---|---|---|---|---|
| acdom440 | 1/m | 5,996 | 1,716 | 11 | 2009 | 2026 | 0.0554 | 0.83 | 1.06 |
| anatoxin_a | ug/L | 20,042 | 5,604 | 34 | 2004 | 2026 | 0.368 | 0.159 | 0.25 |
| chla | ug/L | 1,274,507 | 65,604 | 52 | 2000 | 2026 | 0.0337 | 0.0504 | 5.64 |
| chla_corrected | ug/L | 1,024,268 | 78,299 | 49 | 2000 | 2026 | 0.0393 | 0.0101 | 5.1 |
| chla_uncorrected | ug/L | 539,839 | 43,870 | 43 | 2000 | 2026 | 0.0253 | 0.033 | 6 |
| chloride | mg/L | 2,410,780 | 211,419 | 52 | 2000 | 2026 | 0.0295 | 0.211 | 21.1 |
| cylindrospermopsin | ug/L | 31,818 | 6,263 | 41 | 2006 | 2026 | 0.575 | 0.116 | 0.1 |
| do | mg/L | 9,858,656 | 351,975 | 52 | 2000 | 2026 | 0.001 | 0.0907 | 7.7 |
| do_pctsat | % | 175,180 | 6,352 | 35 | 2000 | 2026 | 0.0021 | 0 | 92.3 |
| doc | mg/L as C | 673,329 | 60,886 | 52 | 2000 | 2026 | 0.0139 | 0.288 | 4.1 |
| ecoli | CFU or MPN/100mL | 2,191,937 | 100,474 | 52 | 2000 | 2026 | 0.0585 | 0.0824 | 62 |
| ecoli_qpcr | CCE/100mL | 3,357 | 12 | 3 | 2012 | 2025 | 0.0015 | 0.127 | 32 |
| enterococcus | CFU or MPN/100mL | 2,542,252 | 39,954 | 51 | 2000 | 2026 | 0.142 | 0.0119 | 10 |
| enterococcus_qpcr | CCE/100mL | 6,218 | 3,102 | 48 | 2008 | 2025 | 0.162 | 0 | 234 |
| fecal_coliform | CFU or MPN/100mL | 2,243,584 | 81,660 | 51 | 2000 | 2026 | 0.0962 | 0.0474 | 26 |
| microcystin_la | ug/L | 11,099 | 4,044 | 16 | 2008 | 2026 | 0.111 | 0.102 | 0.1 |
| microcystin_lr | ug/L | 14,491 | 4,535 | 21 | 2005 | 2026 | 0.176 | 0.093 | 0.25 |
| microcystin_other_congener | ug/L | 72,747 | 4,108 | 17 | 2008 | 2026 | 0.136 | 0.127 | 0.25 |
| microcystin_rr | ug/L | 11,836 | 4,149 | 17 | 2005 | 2026 | 0.134 | 0.112 | 0.1 |
| microcystin_yr | ug/L | 11,164 | 4,033 | 16 | 2008 | 2026 | 0.13 | 0.101 | 0.25 |
| microcystins | ug/L | 70,892 | 6,030 | 49 | 2003 | 2026 | 0.545 | 0.0896 | 0.36 |
| microcystins_dissolved | ug/L | 1,872 | 247 | 14 | 2005 | 2025 | 0.309 | 0.618 | 0.19 |
| nh3 | mg/L as N | 3,806,307 | 254,368 | 52 | 2000 | 2026 | 0.309 | 0.164 | 0.046 |
| no2 | mg/L as N | 1,363,272 | 106,037 | 52 | 2000 | 2026 | 0.294 | 0.269 | 0.0061 |
| no3 | mg/L as N | 1,456,969 | 119,352 | 52 | 2000 | 2026 | 0.119 | 0.289 | 0.39 |
| no3no2 | mg/L as N | 3,995,723 | 263,211 | 52 | 2000 | 2026 | 0.123 | 0.145 | 0.293 |
| orthop | mg/L as P | 2,701,358 | 162,923 | 52 | 2000 | 2026 | 0.137 | 0.198 | 0.0264 |
| ph | pH | 9,965,076 | 390,395 | 52 | 2000 | 2026 | 0 | 0.131 | 7.7 |
| pheophytin | ug/L | 795,345 | 69,963 | 49 | 2000 | 2026 | 0.111 | 0.0563 | 2 |
| phycocyanin | ug/L | 19,814 | 1,237 | 27 | 2010 | 2026 | 0.0378 | 0.708 | 0.67 |
| saxitoxin | ug/L | 14,529 | 3,642 | 34 | 2006 | 2026 | 0.448 | 0.229 | 0.5 |
| secchi | m | 2,415,901 | 121,440 | 52 | 2000 | 2026 | 0.007 | 0 | 1.4 |
| spcond | uS/cm | 9,155,073 | 345,284 | 52 | 2000 | 2026 | 0.0004 | 0.182 | 420 |
| ssc | mg/L | 674,441 | 12,697 | 52 | 2000 | 2026 | 0.0149 | 0.892 | 54 |
| sulfate | mg/L as SO4 | 1,785,862 | 187,272 | 52 | 2000 | 2026 | 0.0247 | 0.216 | 24.4 |
| tdn | mg/L as N | 617,229 | 40,193 | 52 | 2000 | 2026 | 0.0629 | 0.388 | 0.67 |
| tdp | mg/L as P | 858,438 | 46,045 | 52 | 2000 | 2026 | 0.132 | 0.277 | 0.026 |
| temp | deg C | 11,467,908 | 389,427 | 52 | 2000 | 2026 | 0.0001 | 0.108 | 18.4 |
| tkn | mg/L as N | 2,920,160 | 184,757 | 52 | 2000 | 2026 | 0.0718 | 0.111 | 0.7 |
| tn | mg/L as N | 1,844,020 | 94,892 | 52 | 2000 | 2026 | 0.0564 | 0.302 | 0.782 |
| toc | mg/L as C | 1,171,730 | 90,340 | 52 | 2000 | 2026 | 0.0182 | 0.0853 | 5.04 |
| tp | mg/L as P | 4,952,986 | 268,291 | 52 | 2000 | 2026 | 0.0482 | 0.115 | 0.062 |
| transparency_tube | cm | 632,927 | 27,597 | 34 | 2000 | 2026 | 0.278 | 0.0059 | 46 |
| tss | mg/L | 3,281,222 | 200,575 | 52 | 2000 | 2026 | 0.109 | 0.0684 | 10 |
| turbidity_fnu | FNU | 496,902 | 12,531 | 52 | 2000 | 2026 | 0.0179 | 0.398 | 3.9 |
| turbidity_ntru | NTRU | 217,572 | 18,034 | 52 | 2000 | 2026 | 0.0354 | 0.728 | 7 |
| turbidity_ntu | NTU | 4,100,182 | 193,939 | 52 | 2000 | 2026 | 0.0078 | 0.0269 | 4.9 |


## Coverage by state (rows; key variables)

| state | rows | max_sites_any_variable | chla_corrected | secchi | tss | turbidity_ntu | tp | tn | no3no2 | chloride | spcond | ecoli | microcystins | temp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AK | 431,422 | 7,110 | 1,191 | 569 | 4,068 | 136,788 | 4,630 | 3,876 | 14,472 | 6,722 | 92,492 | 1,385 | 32 | 47,391 |
| AL | 879,254 | 3,810 | 640 | 40,012 | 56,342 | 68,798 | 55,592 | 3,851 | 57,106 | 46,523 | 32,672 | 61,286 | 28 | 86,171 |
| AR | 1,498,523 | 2,945 | 54 | 1,342 | 96,103 | 101,633 | 105,818 | 47,972 | 94,688 | 98,633 | 95,894 | 11,696 | 41 | 115,499 |
| AZ | 459,899 | 6,903 | 2,104 | 6,609 | 11,342 | 26,703 | 16,160 | 5,704 | 18,108 | 17,067 | 58,964 | 27,615 | 176 | 51,890 |
| CA | 3,342,594 | 15,541 | 970 | 31,068 | 71,412 | 72,932 | 61,547 | 44,713 | 61,484 | 71,432 | 257,973 | 153,374 | 2,187 | 315,979 |
| CO | 1,990,602 | 10,329 | 1,357 | 2,937 | 35,789 | 32,228 | 136,271 | 94,479 | 109,773 | 65,634 | 212,585 | 41,209 | 517 | 212,399 |
| CT | 607,895 | 1,759 | 0 | 1,893 | 4,853 | 17,141 | 14,948 | 18,998 | 15,556 | 9,581 | 32,114 | 19,072 | 50 | 38,413 |
| DC | 222,963 | 253 | 25 | 10,141 | 11,760 | 13,631 | 3,819 | 3,513 | 5,418 | 1,623 | 14,252 | 10,305 | 0 | 31,071 |
| DE | 674,850 | 1,496 | 46 | 22,810 | 36,850 | 34,898 | 35,465 | 5,883 | 42,310 | 40,782 | 38,136 | 2,701 | 24 | 46,363 |
| FL | 18,851,984 | 93,648 | 557,828 | 835,874 | 591,351 | 973,426 | 1,116,462 | 273,337 | 808,680 | 443,560 | 1,753,221 | 117,373 | 256 | 2,016,959 |
| GA | 2,127,303 | 4,809 | 1,259 | 10,605 | 91,975 | 115,899 | 102,990 | 25,109 | 105,602 | 16,429 | 110,547 | 37,372 | 87 | 312,115 |
| HI | 764,830 | 1,760 | 1,530 | 6,330 | 16,741 | 93,954 | 11,489 | 13,802 | 25,861 | 3,419 | 7,239 | 2 | 0 | 107,490 |
| IA | 1,844,127 | 11,153 | 32,182 | 14,872 | 88,378 | 81,956 | 84,801 | 13,332 | 92,440 | 98,942 | 79,644 | 91,216 | 14,432 | 155,048 |
| ID | 791,731 | 5,358 | 571 | 3,629 | 27,233 | 51,396 | 50,613 | 17,114 | 50,743 | 22,247 | 65,422 | 28,496 | 77 | 83,017 |
| IL | 1,346,893 | 4,826 | 36,094 | 24,966 | 76,714 | 51,132 | 80,847 | 16,482 | 77,462 | 52,487 | 116,587 | 54,632 | 2,868 | 119,150 |
| IN | 1,775,158 | 14,572 | 89 | 15,451 | 85,914 | 128,236 | 103,223 | 16,054 | 97,099 | 83,334 | 142,302 | 115,864 | 1,486 | 176,704 |
| KS | 571,937 | 2,723 | 4,202 | 1,144 | 30,040 | 28,536 | 29,457 | 7,751 | 9,304 | 30,876 | 46,726 | 24,884 | 1,656 | 44,424 |
| KY | 544,671 | 2,143 | 135 | 2,266 | 20,733 | 22,996 | 28,351 | 3,203 | 24,320 | 23,715 | 69,683 | 5,539 | 57 | 69,446 |
| LA | 852,902 | 3,331 | 867 | 9,040 | 38,244 | 54,617 | 55,621 | 6,469 | 51,667 | 46,130 | 88,299 | 2,261 | 251 | 85,675 |
| MA | 790,716 | 5,323 | 456 | 3,877 | 9,793 | 30,426 | 27,976 | 21,817 | 16,229 | 11,045 | 57,821 | 31,306 | 267 | 83,006 |
| MD | 5,131,023 | 6,027 | 18,996 | 82,993 | 213,229 | 88,551 | 150,386 | 150,146 | 299,516 | 26,115 | 576,700 | 9,205 | 35 | 592,405 |
| ME | 286,827 | 2,987 | 933 | 4,317 | 1,591 | 5,373 | 8,093 | 2,571 | 4,897 | 3,546 | 37,257 | 2,211 | 62 | 72,113 |
| MI | 1,312,439 | 6,200 | 3,689 | 153,630 | 30,116 | 38,780 | 52,793 | 24,574 | 36,703 | 29,643 | 104,228 | 118,149 | 1,279 | 208,332 |
| MN | 7,980,964 | 27,512 | 205,368 | 534,783 | 282,954 | 109,616 | 518,243 | 25,360 | 250,699 | 133,835 | 908,541 | 113,691 | 1,408 | 1,293,939 |
| MO | 1,099,145 | 5,822 | 5,242 | 28,230 | 49,600 | 20,632 | 81,685 | 76,365 | 43,552 | 30,591 | 77,309 | 44,069 | 8,428 | 105,768 |
| MS | 797,107 | 2,613 | 11 | 7,513 | 27,209 | 45,989 | 30,813 | 10,381 | 31,012 | 22,581 | 93,961 | 9,532 | 74 | 104,641 |
| MT | 1,059,880 | 10,020 | 1,512 | 3,809 | 38,113 | 49,466 | 49,908 | 33,400 | 51,741 | 44,862 | 183,268 | 7,753 | 391 | 155,456 |
| NC | 2,716,028 | 4,529 | 6,102 | 27,174 | 80,788 | 179,972 | 139,533 | 16,470 | 137,280 | 25,053 | 384,433 | 4,496 | 874 | 396,068 |
| ND | 1,224,243 | 3,249 | 0 | 5,478 | 65,459 | 22,272 | 87,292 | 22,346 | 89,477 | 41,227 | 122,685 | 31,951 | 244 | 156,328 |
| NE | 971,040 | 3,339 | 14 | 7,086 | 39,191 | 68,004 | 35,361 | 7,034 | 44,194 | 37,261 | 134,929 | 47,846 | 15,418 | 127,938 |
| NH | 663,133 | 4,207 | 26 | 17,823 | 1,476 | 73,735 | 57,633 | 1,454 | 8,303 | 29,086 | 88,409 | 55,134 | 190 | 111,595 |
| NJ | 1,901,399 | 9,064 | 846 | 28,740 | 71,953 | 69,826 | 70,085 | 45,646 | 57,428 | 35,004 | 118,705 | 39,502 | 270 | 160,158 |
| NM | 469,905 | 3,655 | 47 | 3,084 | 8,684 | 30,504 | 15,254 | 4,313 | 13,450 | 11,326 | 86,379 | 10,424 | 60 | 57,275 |
| NV | 372,351 | 3,639 | 51 | 72 | 11,704 | 16,753 | 17,665 | 13,274 | 17,550 | 17,016 | 39,923 | 10,438 | 53 | 41,154 |
| NY | 1,680,654 | 6,561 | 293 | 42,196 | 25,327 | 53,986 | 52,819 | 45,583 | 35,470 | 76,275 | 135,534 | 25,661 | 643 | 151,286 |
| OH | 1,247,198 | 9,448 | 7,348 | 9,685 | 62,173 | 8,842 | 91,352 | 27,373 | 91,358 | 71,797 | 119,844 | 86,622 | 1,164 | 95,127 |
| OK | 1,300,164 | 6,997 | 24,729 | 14,153 | 27,958 | 118,662 | 90,503 | 28,414 | 39,142 | 59,351 | 63,328 | 28,469 | 88 | 130,309 |
| OR | 2,104,305 | 11,458 | 5,681 | 13,975 | 86,106 | 218,935 | 99,460 | 41,373 | 107,125 | 41,657 | 230,806 | 110,469 | 8,905 | 347,302 |
| PA | 2,003,921 | 10,330 | 361 | 5,210 | 93,288 | 32,031 | 96,021 | 85,051 | 75,146 | 86,507 | 225,699 | 43,644 | 46 | 165,300 |
| PR | 187,172 | 1,343 | 0 | 647 | 1,907 | 2,156 | 7,547 | 7,912 | 7,524 | 3,461 | 13,302 | 162 | 0 | 25,185 |
| RI | 117,132 | 727 | 111 | 346 | 2,990 | 4,350 | 4,449 | 6,930 | 4,930 | 4,357 | 6,847 | 122 | 25 | 5,540 |
| SC | 1,529,959 | 5,034 | 22,170 | 11,771 | 19,156 | 136,103 | 109,157 | 69,773 | 114,649 | 7,302 | 16,962 | 60,828 | 4,581 | 197,129 |
| SD | 945,139 | 4,782 | 8,543 | 7,888 | 54,957 | 45,389 | 48,655 | 2,995 | 41,372 | 8,678 | 143,339 | 28,956 | 227 | 148,552 |
| TN | 1,231,095 | 7,291 | 95 | 6,698 | 49,399 | 41,897 | 67,994 | 2,958 | 75,365 | 14,877 | 122,674 | 60,158 | 34 | 176,849 |
| TX | 1,333,122 | 5,107 | 400 | 30,987 | 38,996 | 4,435 | 46,751 | 53,298 | 48,416 | 60,957 | 188,188 | 5,525 | 692 | 84,737 |
| UT | 1,471,407 | 7,727 | 4,926 | 9,330 | 90,636 | 43,475 | 93,206 | 29,100 | 102,847 | 73,960 | 185,620 | 49,212 | 91 | 157,142 |
| VA | 5,423,852 | 14,061 | 53,134 | 65,648 | 211,767 | 202,956 | 240,531 | 238,250 | 175,129 | 45,031 | 717,250 | 154,199 | 94 | 735,874 |
| VT | 890,234 | 4,454 | 728 | 36,569 | 10,151 | 30,713 | 86,358 | 48,330 | 8,298 | 32,354 | 132,791 | 14,613 | 24 | 139,921 |
| WA | 1,240,314 | 6,602 | 491 | 3,296 | 34,546 | 123,061 | 44,770 | 13,303 | 41,467 | 23,456 | 96,925 | 31,241 | 705 | 174,599 |
| WI | 4,272,240 | 11,549 | 10,396 | 205,544 | 119,203 | 63,890 | 310,226 | 44,620 | 135,109 | 98,381 | 352,768 | 142,246 | 218 | 890,201 |
| WV | 291,793 | 5,655 | 368 | 63 | 18,097 | 3,484 | 14,604 | 15,145 | 16,485 | 10,026 | 31,948 | 3,317 | 34 | 29,057 |
| WY | 281,371 | 3,716 | 57 | 1,728 | 6,863 | 9,018 | 7,759 | 6,819 | 11,767 | 15,026 | 41,948 | 4,504 | 43 | 32,418 |


## QC counts per step (rows removed)

| variable | input | non_water_media | qc_activity | logger_or_aggregate | rejected_status | sentinel_code | no_value | out_of_range | speciation_duplicate | duplicate | not_co_sampled | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acdom440 | 6,194 | 0 | 193 | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 5,996 |
| anatoxin_a | 21,543 | 0 | 924 | 25 | 0 | 0 | 148 | 1 | 0 | 403 | 0 | 20,042 |
| chla | 1,369,941 | 0 | 37,098 | 34,572 | 28 | 2 | 2,939 | 1,652 | 0 | 19,143 | 0 | 1,274,507 |
| chla_corrected | 1,077,604 | 0 | 37,142 | 3,470 | 1 | 1 | 1,469 | 1,494 | 0 | 9,759 | 0 | 1,024,268 |
| chla_uncorrected | 566,477 | 0 | 15,403 | 2,085 | 262 | 3 | 5,432 | 155 | 0 | 3,298 | 0 | 539,839 |
| chloride | 2,542,013 | 0 | 78,078 | 6,422 | 226 | 0 | 2,510 | 72 | 0 | 43,925 | 0 | 2,410,780 |
| cylindrospermopsin | 35,098 | 0 | 869 | 0 | 0 | 0 | 1,323 | 1 | 0 | 1,087 | 0 | 31,818 |
| do | 13,381,698 | 0 | 129,960 | 2,914,366 | 2,123 | 115 | 53,946 | 20,028 | 0 | 192,856 | 209,648 | 9,858,656 |
| do_pctsat | 438,326 | 0 | 1,305 | 256,184 | 11 | 0 | 506 | 187 | 0 | 2,492 | 2,461 | 175,180 |
| doc | 723,629 | 0 | 28,985 | 1,250 | 23 | 0 | 1,007 | 104 | 0 | 18,931 | 0 | 673,329 |
| ecoli | 2,507,133 | 0 | 96,221 | 171,084 | 248 | 682 | 10,630 | 391 | 0 | 35,940 | 0 | 2,191,937 |
| ecoli_qpcr | 3,361 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 3,357 |
| enterococcus | 2,716,715 | 0 | 33,246 | 68,395 | 72 | 2,827 | 30,985 | 9 | 0 | 38,929 | 0 | 2,542,252 |
| enterococcus_qpcr | 12,468 | 0 | 0 | 6,250 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6,218 |
| fecal_coliform | 2,434,442 | 0 | 29,897 | 121,451 | 29 | 3,090 | 4,903 | 341 | 0 | 31,147 | 0 | 2,243,584 |
| microcystin_la | 11,270 | 0 | 108 | 0 | 0 | 0 | 3 | 26 | 0 | 34 | 0 | 11,099 |
| microcystin_lr | 16,291 | 0 | 1,158 | 0 | 0 | 0 | 4 | 12 | 0 | 626 | 0 | 14,491 |
| microcystin_other_congener | 75,151 | 0 | 1,996 | 0 | 0 | 0 | 0 | 0 | 0 | 408 | 0 | 72,747 |
| microcystin_rr | 12,790 | 0 | 761 | 0 | 0 | 0 | 2 | 26 | 0 | 165 | 0 | 11,836 |
| microcystin_yr | 11,408 | 0 | 161 | 0 | 0 | 0 | 6 | 27 | 0 | 50 | 0 | 11,164 |
| microcystins | 78,544 | 0 | 2,522 | 1,701 | 0 | 0 | 3,057 | 4 | 0 | 368 | 0 | 70,892 |
| microcystins_dissolved | 3,236 | 0 | 72 | 1,192 | 0 | 0 | 28 | 0 | 0 | 72 | 0 | 1,872 |
| nh3 | 4,754,882 | 0 | 138,718 | 36,007 | 172 | 215 | 32,550 | 13,015 | 622,047 | 105,851 | 0 | 3,806,307 |
| no2 | 1,799,315 | 0 | 40,658 | 7,955 | 151 | 189 | 6,279 | 2,189 | 335,993 | 42,629 | 0 | 1,363,272 |
| no3 | 1,887,866 | 0 | 39,447 | 16,047 | 71 | 125 | 3,127 | 6,865 | 339,847 | 25,368 | 0 | 1,456,969 |
| no3no2 | 4,363,259 | 0 | 142,982 | 107,238 | 143 | 2 | 25,250 | 5,984 | 0 | 85,937 | 0 | 3,995,723 |
| orthop | 3,382,193 | 0 | 83,289 | 17,256 | 55 | 0 | 21,286 | 3,973 | 498,133 | 56,843 | 0 | 2,701,358 |
| ph | 13,157,091 | 0 | 139,952 | 2,608,122 | 2,100 | 178 | 64,960 | 12,068 | 0 | 220,454 | 144,181 | 9,965,076 |
| pheophytin | 847,473 | 0 | 33,083 | 417 | 1 | 4 | 765 | 11,997 | 0 | 5,861 | 0 | 795,345 |
| phycocyanin | 23,671 | 0 | 371 | 2,622 | 0 | 0 | 5 | 230 | 0 | 629 | 0 | 19,814 |
| saxitoxin | 15,432 | 0 | 612 | 0 | 0 | 0 | 97 | 0 | 0 | 194 | 0 | 14,529 |
| secchi | 2,498,721 | 0 | 8,349 | 26,160 | 13 | 220 | 6,340 | 8,451 | 0 | 33,287 | 0 | 2,415,901 |
| spcond | 11,622,833 | 0 | 128,575 | 2,126,167 | 1,461 | 21 | 6,656 | 24,093 | 0 | 180,787 | 0 | 9,155,073 |
| ssc | 695,127 | 0 | 3,145 | 11,863 | 0 | 0 | 631 | 361 | 0 | 4,686 | 0 | 674,441 |
| sulfate | 1,879,678 | 0 | 54,676 | 1,184 | 15 | 0 | 1,413 | 228 | 0 | 36,300 | 0 | 1,785,862 |
| tdn | 658,853 | 0 | 25,354 | 3,623 | 0 | 131 | 4,738 | 270 | 0 | 7,508 | 0 | 617,229 |
| tdp | 929,147 | 0 | 29,818 | 11,291 | 9 | 88 | 12,117 | 46 | 0 | 17,340 | 0 | 858,438 |
| temp | 20,542,295 | 0 | 143,045 | 8,304,736 | 2,355 | 7 | 50,958 | 19,408 | 0 | 260,746 | 293,132 | 11,467,908 |
| tkn | 3,082,596 | 0 | 93,611 | 9,001 | 483 | 0 | 12,148 | 2,243 | 0 | 44,950 | 0 | 2,920,160 |
| tn | 1,988,908 | 0 | 57,552 | 6,278 | 9 | 0 | 1,860 | 2,973 | 53,268 | 22,948 | 0 | 1,844,020 |
| toc | 1,245,400 | 0 | 46,317 | 383 | 26 | 1 | 2,396 | 432 | 0 | 24,115 | 0 | 1,171,730 |
| tp | 5,245,244 | 0 | 165,936 | 29,209 | 461 | 0 | 26,770 | 3,882 | 433 | 65,567 | 0 | 4,952,986 |
| transparency_tube | 647,465 | 0 | 1,374 | 959 | 0 | 0 | 1,860 | 5,807 | 0 | 4,538 | 0 | 632,927 |
| tss | 3,582,837 | 0 | 125,263 | 90,816 | 661 | 10 | 17,982 | 2,635 | 0 | 64,248 | 0 | 3,281,222 |
| turbidity_fnu | 619,170 | 0 | 12,024 | 96,897 | 0 | 0 | 48 | 5,003 | 0 | 8,296 | 0 | 496,902 |
| turbidity_ntru | 221,891 | 0 | 3,223 | 692 | 0 | 0 | 1 | 253 | 0 | 150 | 0 | 217,572 |
| turbidity_ntu | 4,817,266 | 0 | 96,471 | 502,501 | 2,903 | 3,256 | 26,718 | 18,169 | 0 | 67,066 | 0 | 4,100,182 |


## Unmapped rows by reason

| unmapped_reason | rows |
|---|---|
| conductivity_not_temperature_compensated | 3,035,254 |
| unit | 1,920,354 |
| no_value_and_no_unit | 740,557 |
| fraction_dissolved | 453,771 |
| ambiguous_characteristic | 446,901 |
| unit_missing | 433,440 |
| fraction_unspecified | 234,656 |
| fraction_particulate | 212,235 |
| turbidity_unit_not_ntu_fnu_ntru | 152,718 |
| speciation_conflict | 86,754 |
| fraction_other | 80,063 |
| aux_secchi_visible_at_bottom | 23,104 |
| speciation_unrecognised | 15,971 |
| sensor_fluorescence_unit | 6,662 |
| speciation_NO3_invalid | 2,878 |
| speciation_P_invalid | 2,801 |
| qpcr_method_with_culture_unit | 1,819 |
| speciation_N_invalid | 1,615 |
| speciation_NO2_invalid | 1,483 |
| speciation_Cl_invalid | 471 |
| speciation_NH3_invalid | 430 |
| speciation_NH4_invalid | 303 |
| speciation_S_invalid | 222 |
| qpcr_gene_copies_unit | 142 |
| speciation_PO4_invalid | 1 |


## Top unmapped (reason x characteristic x unit)

| unmapped_reason | characteristic | unit | rows |
|---|---|---|---|
| conductivity_not_temperature_compensated | Conductivity | uS/cm | 2,061,158 |
| conductivity_not_temperature_compensated | Conductivity | umho/cm | 652,701 |
| ambiguous_characteristic | Chlorophyll | ug/L | 395,501 |
| unit | Enterococcus | MPN | 332,472 |
| unit | Specific conductance | mho/cm | 265,242 |
| unit | Fecal Coliform | MPN | 204,060 |
| fraction_dissolved | Kjeldahl nitrogen | mg/L | 170,770 |
| conductivity_not_temperature_compensated | Conductivity | mS/cm | 161,864 |
| fraction_dissolved | Phosphate-phosphorus | mg/L | 145,094 |
| unit | pH | mV | 135,241 |
| no_value_and_no_unit | Fecal Coliform |  | 132,372 |
| unit | Escherichia coli | MPN | 117,922 |
| fraction_particulate | Organic carbon | mg/L | 105,252 |
| fraction_unspecified | Phosphate-phosphorus as P | ug/l | 104,289 |
| no_value_and_no_unit | Enterococcus |  | 100,407 |
| unit | pH | nu | 89,819 |
| unit | Colored dissolved organic matter (CDOM) | ug/L | 86,814 |
| unit | pH | units/cm | 84,409 |
| no_value_and_no_unit | Inorganic nitrogen (nitrate and nitrite) ***retired***use Nitrate + Nitrite |  | 78,367 |
| conductivity_not_temperature_compensated | Conductivity | S/m | 77,204 |
| turbidity_unit_not_ntu_fnu_ntru | Turbidity | JTU | 75,021 |
| fraction_particulate | Phosphorus | mg/L | 71,385 |
| unit_missing | Inorganic nitrogen (nitrate and nitrite) as N |  | 69,761 |
| unit_missing | Fecal Coliform |  | 69,003 |
| no_value_and_no_unit | Orthophosphate |  | 64,519 |
| speciation_conflict | Phosphate-phosphorus | mg/L | 63,596 |
| conductivity_not_temperature_compensated | Conductivity | deg C | 58,225 |
| fraction_unspecified | Organic carbon | mg/L | 54,803 |
| no_value_and_no_unit | Ammonia-nitrogen |  | 48,488 |
| fraction_unspecified | Phosphate-phosphorus | mg/L | 48,355 |
| unit | Suspended Sediment Concentration (SSC) | #/mL | 48,348 |
| no_value_and_no_unit | Total suspended solids |  | 46,311 |
| fraction_dissolved | Phosphate-phosphorus as P | mg/l | 45,800 |
| fraction_other | Phosphorus | mg/L | 45,409 |
| unit_missing | Total suspended solids |  | 45,363 |
| unit_missing | Ammonia-nitrogen as N |  | 45,259 |
| fraction_dissolved | Kjeldahl nitrogen | mg/l | 42,454 |
| unit | Escherichia coli | CFU | 40,279 |
| no_value_and_no_unit | Nitrite |  | 39,094 |
| unit | Fecal Coliform | CFU | 37,736 |


## Completeness: WQX3 rows vs legacy WQX 2.2 summary

| group | period | wqx3_rows | legacy_total | ratio |
|---|---|---|---|---|
| bacteria | 2000-2023 | 8,232,484 | 8,418,417 | 0.978 |
| bacteria | 2024-present | 628,772 | 632,171 | 0.995 |
| carbon | 2000-2023 | 2,023,842 | 2,116,191 | 0.956 |
| carbon | 2024-present | 145,080 | 124,141 | 1.17 |
| cyanotoxins | 2000-2023 | 213,700 | 196,585 | 1.09 |
| cyanotoxins | 2024-present | 77,487 | 67,322 | 1.15 |
| ions | 2000-2023 | 18,141,043 | 18,855,827 | 0.962 |
| ions | 2024-present | 1,253,045 | 1,118,980 | 1.12 |
| nitrogen | 2000-2023 | 18,138,059 | 18,546,213 | 0.978 |
| nitrogen | 2024-present | 1,350,777 | 1,001,989 | 1.35 |
| optical | 2000-2023 | 16,669,515 | 17,414,540 | 0.957 |
| optical | 2024-present | 1,261,658 | 1,150,537 | 1.1 |
| phosphorus | 2000-2023 | 9,611,469 | 9,866,064 | 0.974 |
| phosphorus | 2024-present | 709,655 | 554,637 | 1.28 |
| physchem | 2000-2023 | 78,147,104 | 77,788,816 | 1 |
| physchem | 2024-present | 4,924,312 | 4,697,388 | 1.05 |
