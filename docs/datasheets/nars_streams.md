# Datasheet — `nars_streams` (EPA NRSA rivers & streams, 2008-09 … 2023-24)

| | |
|---|---|
| **Location** | `$SENTINEL_DATA/processed/nars_streams/` (Parquet, zstd) · manifest `manifests/nars_streams.json` · raw `raw/nars/<survey>/` (manifest `nars_raw`) |
| **Source** | U.S. EPA National Aquatic Resource Surveys data hub, <https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys> (one HTML table; the parsed catalog of all 380 rows is saved as `raw/nars/_index/nars_catalog_all.csv`, the page itself as `nars_data_page_2026-10-22.html`) |
| **License** | U.S. Government work (EPA). The hub page carries no license statement (unverified); treated as public domain, cite EPA NARS. |
| **Retrieved** | 2026-10-22 (351 data + metadata files incl. 2 tables extracted from the NLA 2007 bundle, 0 failures; 10 files — NRSA 2013-14 and 2023-24 site information, seven NRSA 2018-19 files, the NLA 2007–2022 population-estimate file — were copied from the 2026-10-22 research cache only after their name and byte size matched the official `Content-Length`; the parsed hub catalog has 378 file rows, 174 selected) |
| **Builder** | `sentinel/data/sources/nars.py` (catalog, file roles, tidy parsers) · `sentinel/data/bio_nars.py` (tables) |
| **Synthetic data** | none |

## Coverage

| Survey | Cycle | Visits | Sites | Within-cycle revisits | Dates | Benthic visits / taxa | Fish visits / taxa |
|---|---|---|---|---|---|---|---|
| `nrsa0809` | 2008-09 | 2,320 | 2,123 | 181 | 2008-06-03 … 2009-11-30 | 2,066 / 969 | 2,100 / 623 |
| `nrsa1314` | 2013-14 | 2,261 | 2,069 | 192 | 2013-04-30 … 2014-10-14 | 2,255 / 891 | 2,058 / 626 |
| `nrsa1819` | 2018-19 | 2,112 | 1,919 | 191 | 2018-05-07 … 2019-10-08 | 2,106 / 894 | 1,773 / 615 |
| `nrsa2324` | 2023-24 | 2,143 | 1,916 | 186 | 2023-04-23 … 2024-10-03 | 2,133 / 907 | 1,812 / 591 |
| **total** | | **8,836** | | | | 355,870 count rows (primary samples), 1,174 distinct benthic names | 91,722 rows, 845 names |

All nine NARS aggregated ecoregions (`ag_eco9`: CPL, NAP, NPL, SAP, SPL, TPL, UMW, WMT, XER) occur in every cycle. NRSA 2023-24 (released June–Sept 2026) is included. **No NRSA cycle publishes periphyton or diatom taxonomic counts** (confirmed on the hub): only periphyton biomass (AFDM) and chlorophyll-a, which are in `periphyton.parquet`.

## Tables

| File | Grain | Key columns |
|---|---|---|
| `visits.parquet` (8,836) | one row per site visit | `visit_id` (= `<survey>-<UID>`), `uid`, `site_id`, `unique_id` (cross-cycle site id, 2018-19 and 2023-24 only), `visit_no`, `date` (local), `year`, `lat`/`lon` (NAD83 dd), `state`, `ag_eco9`, `ag_eco3`, `us_l3code`, `us_l4code`, `huc8` (8-digit **string**), `huc2` (from HUC8), `huc2_nars` (NARS code, e.g. `H03W`), `huc10`/`huc12` (zero-padded strings, 2023-24), `comid` (string) + `comid_source`, `reachcode` (14-digit string or null) + `reachcode_mangled`, `site_type` (PROB/HAND), `reference_class` (R least / S intermediate / T most disturbed; raw value in `reference_class_raw`, 2008-09 and 2013-14 only), `within_cycle_revisit`, `index_visit`, `n_cycles_unique_id`, `design_weight`, `elevation_m`, `has_benthic`/`has_fish`/`has_chemistry`, `benthic_n_taxa`, `fish_n_taxa` |
| `benthic_counts.parquet` | visit × sample type × taxon (primary samples only) | `taxa_id`, `taxon_name` (= NARS `TARGET_TAXON`), `count` (full lab count, `TOTAL`), `count_300` (300-organism fixed-count subsample, `TOTAL300`; NaN when the taxon is not in the subsample — never 0-filled at ingest), `is_distinct`, `is_distinct_300`, `sample_type` (BERW/BERWW reach-wide, BETB/BELGB transect/boatable), `src_ptv`, `src_ffg`, `src_habit` |
| `benthic_counts_qc_duplicates.parquet` (7,518) | same schema | 2008-09 field-duplicate benthic samples (`sample_category` D, 171 visits; BD, 1 visit) moved out of the main table so they are never summed with the primary sample — usable as a replicate noise floor |
| `fish_counts.parquet` | visit × taxon | `taxon_name` (NARS common name, `FINAL_NAME`), `count`, `anomaly_count`, `is_distinct`, `non_native` |
| `taxa_source.parquet` | (survey, assemblage, taxa_id, name) | published hierarchy and traits (`src_phylum` … `src_genus`, `src_ffg`, `src_habit`, `src_ptv`, fish `src_family/genus/species`, tolerance/trophic/habitat codes). Harmonised taxonomy: `processed/taxonomy_harmonised` |
| `chemistry_long.parquet` (≈219k) | visit × analyte | `analyte` (canonical), `source_code`, `value` (as published), `unit_source`, `value_canonical` + `unit` (canonical), `unit_status` (ok / converted / assumed_unit / unknown_unit), `censored`, `detection_limit` (+ canonical), `mdl`, `rl`, `below_mdl`, `flag` (NARS/QA/lab flags joined by `;`), `qualifier`, `source_file`, `source_column` |
| `chemistry_wide.parquet` | one row per visit | `<analyte>_<unit>` in canonical units, e.g. `tp_ug_l`, `tn_mg_l`, `nh4_n_mg_l`, `no3_n_mg_l`, `chloride_mg_l`, `sulfate_mg_l`, `cond_us_cm`, `ph_lab_su`, `turbidity_ntu`, `doc_mg_l`, `tss_mg_l`, `anc_ueq_l`, cations, `silica_mg_l`, `chla_ug_l`, `microcystin_ug_l`, `cylindrospermopsin_ug_l`, field `do_field_mg_l`, `temp_field_degc`, `ph_field_su`, `cond_field_us_cm`; `cond_unit_source`; `n_censored_analytes`. Non-detects and unknown units are NaN here (details stay in the long table) |
| `periphyton.parquet` | visit × analyte | periphyton AFDM and chlorophyll-a (2013-14, 2018-19, 2023-24) with source units (area- and volume-based units differ by cycle; `unit_status` tells which rows are in the canonical unit) |
| `condition_long.parquet` / `condition_wide.parquet` | visit × indicator | benthic MMI score + class (Good/Fair/Poor), O/E score + class, fish MMI score + class, TP/TN/salinity/acidity condition, bed sediment, in-stream cover, riparian vegetation & disturbance, microcystin & cylindrospermopsin (detect / EPA benchmark), enterococci, fish-tissue Hg. `source_file` and `source_column` are kept; `condition_wide` prefers the cycle's MMI file for `bent_mmi_cond` |
| `phab_long.parquet` / `phab_key_wide.parquet` | visit × metric | every numeric metric of the cycle's "larger set" physical-habitat file (long); a key subset wide (`XFC_NAT`, `PCT_FN`, `PCT_SAFN`, `LRBS_USE`, `W1_HALL`, `XCMGW`, `XCDENBK`, `XWIDTH`, `XDEPTH_CM`, `XSLOPE`, `RP100`, …) |
| `uid_alignment.csv`, `build_stats.json`, `analyte_dictionary.json` | | build diagnostics (below) |

### Chemistry coverage (visits with a value in `chemistry_wide`)

| | 2008-09 | 2013-14 | 2018-19 | 2023-24 |
|---|---|---|---|---|
| TP / TN | 2,312 / 2,312 | 2,261 / 2,261 | 2,109 / 2,109 | 2,137 / 2,138 |
| NH4-N / NO3-N | 2,305 / 2,254 | 2,261 / 2,240 | 2,038 / 2,035 | 2,092 / 1,902 |
| Cl / SO4 | 2,312 / 2,312 | 2,261 / 2,260 | 2,110 / 2,109 | 2,137 / 2,136 |
| conductivity / lab pH | 2,257 / 2,257 | 2,260 / 2,207 | 2,109 / 2,109 | 2,136 / 2,138 |
| turbidity / DOC / TSS | 2,257 / 2,312 / 2,312 | 2,260 / 2,261 / 2,261 | 2,109 / 2,107 / 2,095 | 2,090 / 2,138 / 2,030 |
| microcystin (detected values) | — | 915 | 234 | 670 |
| field DO / temp / pH / cond | 2,250 / 2,286 / 2,274 / 2,195 | 2,211 / 2,228 / 2,217 / 2,225 | 2,091 / 2,100 / 2,079 / 2,100 | not published |

## Processing and QC (what was done, exactly)

1. **Catalog** parsed from the hub table; shapefile/bundle zips excluded (watershed shapefiles are not needed for tabular work; the 2008-09 one is 128 MB). File roles per cycle are fixed in `nars.FILE_ROLES` (verified names, 2026-10-22).
2. **Visits** = site-information rows of sampled visits (`SITESAMP`/`SAMPLED` = Y where present). `visit_id` is survey-scoped (`nrsa1819-2014737`).
3. **UID reconciliation.** Two 2018-19 files use a superseded UID numbering: the fish count file (22,049 of 22,057 rows) and periphyton chlorophyll (2,111 rows). Rows whose UID is absent from the site information are re-keyed through an *unambiguous* (SITE_ID, collection date) match, falling back to an unambiguous (SITE_ID, VISIT_NO) match; 22,049 and 2,109 rows were remapped, the rest left unresolved (`uid_alignment.csv`). Ambiguous keys are never guessed (unit-tested).
4. **Rows outside the design dropped** (counted in `build_stats.json`): 821 chemistry rows / 130 visits — field rows of special-study "AOI" sites (2018-19), 33 chemistry rows with blank SITE_ID (2023-24), and 2–5 rows per file whose UID could not be matched.
5. **Primary samples only** for chemistry: `SAM_CODE` REGULAR (lab duplicates/splits dropped), `SAMPLE_CAT`/`SAMPLE_CATEGORY` P (duplicates/blanks dropped), `MATRIX` WATER. Community counts carry `sample_category`; non-primary samples (2008-09 benthic D/BD) go to `*_counts_qc_duplicates.parquet`. Blank-category 2008-09 rows are "sample lost during shipping" notes without taxa and are dropped.
6. **Missing codes** (blank, −9999, −999999, `NA`, non-numeric tokens) → NaN at parse time; tokens kept in `qualifier` (unit-tested).
7. **Censoring.** NARS flag legend (2023-24 metadata): ND = non-detect, L = estimated (between MDL and RL), Q = quality issue, S = shipping-time exceedance, H = holding-time exceedance, NR = no lab result. `ND` → `censored=True`, `value` NaN (NARS leaves RESULT blank), `detection_limit` = MDL (else RL). `L`, `S`, `H`, `Q` stay as flags. 2008-09 `_ALERT` code R ("below reporting limit", value reported) is a flag, not censoring. Values reported below the MDL without an ND flag (e.g. 2012-style estimates) are kept and flagged `below_mdl`. No substitution (DL/2 etc.) anywhere.
8. **Units.** Each analyte has one canonical unit (`ANALYTES` in `nars.py`): TP ug/L as P; TN, NH4-N, NO3-N, NO2-N, NO3+NO2-N, TKN mg/L as N; major ions, DOC, TSS mg/L; conductivity uS/cm; pH SU; turbidity NTU; ANC ueq/L; colour PCU; chl-a, toxins ug/L. The only conversion is mg/L ↔ ug/L (×1000; e.g. 2008-09 NTL is published in ug/L → ×0.001), unit-tested. Unit spellings are normalised (`mg N/L`, `MG N/L`, `mgN/L` → mg/L; `Std. Units`/blank → SU for pH). A blank unit on a non-detect row is filled only when every labelled row of that analyte in the same file maps to one canonical unit (`unit_status = assumed_unit`). Files without any unit column (2008-09 `chem.csv`, NLA 2007 files, NRSA 2013-14 microcystin) use the unit documented in their metadata file (`NRSA0809_CHEM_COLUMNS`, `NLA07_WQ_COLUMNS`, "Result for Microcystin - UG/L"). Unknown units (e.g. 2023-24 NO3+NO2 has no unit in file or metadata; 2018-19 periphyton AFDM in "g") are never converted (`unknown_unit`, NaN in the wide table).
9. **Conductivity label check.** 2018-19 lab conductivity is labelled "uS/cm at 25 C" (WRS, NCDEQ labs) or "uS/cm" (WSLOH, ODEQ; 153 visits). Both are treated as specific conductance: the lab/field-specific-conductance ratio has median 1.010 for "uS/cm" vs 1.005 for "at 25 C" (10–90%: 0.90–1.36 vs 0.91–1.19). The source label is kept in `unit_source` / `cond_unit_source`.
10. **IDs as strings.** HUC8 `H11130301` → `11130301`, 7-digit HUC8 zero-padded; HUC10/HUC12 (2023-24 lost leading zeros) zero-padded to fixed width; COMID/UID/TAXA_ID strings.
11. **COMID.** Present in site information for 2013-14, 2018-19 and 866 of 2,143 2023-24 visits; **absent for all 2008-09 visits** (no COMID column in any 2008-09 site file; the "External Data" file covers non-NRSA sites). 33 2023-24 visits receive the COMID of the same `UNIQUE_ID` in 2018-19 (`comid_source = other_cycle_same_unique_id`, only when that UNIQUE_ID maps to exactly one COMID). Use coordinates + NLDI for the rest.
12. **Field measurements** (DO, temperature, pH, conductivity): calibration rows removed; median over the visit's measurement rows (several transects in 2008-09; `n_field_rows` kept).

## Known issues and traps

- **Index circularity:** benthic/fish MMI and O/E classes are deterministic functions of the counts in `*_counts.parquet`; never predict them from those counts (docs/research/04 §E.3).
- **Reference classes** use chemistry/habitat screens; exclude screening variables or run sensitivity analyses when predicting condition.
- `REACHCODE` and `NHDPLUSID` in the 2023-24 site file went through spreadsheet scientific notation (e.g. `3.1402E+12`) and are unrecoverable (`reachcode_mangled`, `reachcode` null).
- 2018-19 and 2023-24 have no published reference (RT) designations; `reference_class` is null there.
- 2023-24 has no field-chemistry file and no population-estimate (condition) file yet: only benthic and fish MMI classes.
- Two 2018-19 sites carry `VISIT_NO = 1` in both years (1,918 first-visit rows / 1,916 sites).
- Values are as published, without range clipping. Saline/brine-affected streams give extreme but real values (`cond_us_cm` max 118,000; `chloride_mg_l` max 47,500; `sulfate_mg_l` max 33,549; `turbidity_ntu` max 47,654); one blank-corrected TP is negative (−2 ug/L). Models that take logs must apply an explicit, documented floor (the RF baseline uses floors of 1 ug/L TP, 0.01 mg/L TN/Cl, 1 uS/cm, 0.01 NTU).
- Taxonomic resolution and labs differ by cycle (NRSA genus-level IDs dominate; 2008-09 has 969 benthic names vs 891–907 later) — use `processed/taxonomy_harmonised` for cross-cycle work.
- Protocol differences: wadeable (BERW/BERWW) vs boatable (BETB/BELGB) benthic sample types; never pool without the `sample_type` indicator.

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_nars.py     # raw + manifest nars_raw (resume-safe)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/build_nars.py        # processed nars_streams + nars_lakes
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_bio_data.py
```
