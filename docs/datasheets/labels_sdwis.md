# Datasheet — `labels_sdwis` (SDWIS drinking-water systems, violations and PWS-quarter labels)

**Use:** TapRisk (D4). The labels are health-based violations of rules tied to source-water quality, for surface-water-sourced community water systems (CWS):
- nitrate MCL;
- Stage 1/2 DBP (TTHM/HAA5) MCL;
- DBP-precursor (TOC) removal TT;
- surface-water-treatment-rule turbidity TT.

These are **administrative labels** (compliance determinations). They are secondary to measured concentrations; see `labels_ucmr` for raw-water TOC and bromide.

## Source

| Item | Value |
|---|---|
| Publisher | US EPA, SDWIS/Fed via the ECHO "SDWA dataset" bulk download (refreshed quarterly) |
| URL (verified 2026-10-22) | https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip — 558,010,511 bytes, Last-Modified 2026-10-14 01:15:30 GMT, ETag `"2142908f-65dc2aace8981"`, sha256 `4d92d5f4…f41dccb0`. The snapshot has `SUBMISSIONYEARQUARTER` = **2026Q3** (a fiscal-year quarter) |
| Data dictionary | https://echo.epa.gov/tools/data-downloads/sdwa-download-summary (saved as `raw/sdwis/sdwa_download_summary.txt`). Code meanings come from `SDWA_REF_CODE_VALUES.csv` in the zip, saved as `sdwis_ref_codes.parquet` (2,376 codes) |
| Access | Direct download, no key or login |
| License / terms | US federal government data (EPA), public domain |
| Retrieved | 2026-10-22. The zip was downloaded twice that day with identical sha256 |
| Cross-check (fallback API) | Envirofacts `https://data.epa.gov/dmapservice/sdwis.violation/pwsid/equals/<PWSID>/json` was queried for 4 systems. Every Envirofacts violation ID was present in the ECHO bulk. ECHO had 26–52% **more** violations (older Archived/Resolved ones, 1993–2015), so ECHO bulk is the more complete source. See `build_stats.json → envirofacts_crosscheck` |

**How the 4.1 GB violations table was read.** `SDWA_VIOLATIONS_ENFORCEMENT.csv` was stream-scanned inside the zip without extraction: 15,550,611 rows read, 2,271,943 kept. Kept rows match RULE_CODE ∈ {121, 122, 123, 130, 210, 220, 230, 331} or a nitrate/DBP/SWTR/TOC contaminant code. The superset is cached as `interim/sdwis/violations_enforcement_scan.parquet` (59 MB). The zip was deleted after the build, under the "raw not kept unless < 100 MB" rule; its sha256 and headers are in the manifest.

## Files (`$SENTINEL_DATA/processed/labels_sdwis/`)

| File | Rows | Content |
|---|---|---|
| `sdwis_systems.parquet` | 434,757 | All PWS (CWS 97,156, of which 49,342 active; NTNCWS 49,634; TNCWS 287,941). **Contact fields dropped** |
| `sdwis_violations.parquet` | 806,618 | One row per (PWSID, VIOLATION_ID) in the scanned rule superset, all systems. 174,203 are health-based. Each row carries a label, enforcement summary and system attributes |
| `sdwis_pws_quarter_labels.parquet` | 1,070,774 | PWS × calendar-quarter label panel: 15,070 surface-water-sourced CWS, 2005Q1–2026Q2 |
| `sdwis_source_facilities.parquet` | 692,138 | Source facilities (`IS_SOURCE_IND = Y`): intakes `IN`, wells `WL`, reservoirs `RS`, springs, consecutive connections `CC` with `seller_pwsid`. **No coordinates** |
| `sdwis_areas_served.parquet` | 579,080 | Counties, cities, ZIP codes and tribal areas served |
| `sdwis_ref_codes.parquet` | 2,376 | SDWA code → description |
| `build_stats.json` | — | Every count in this datasheet |

## Surface-water-sourced systems

`PRIMARY_SOURCE_CODE` (2026Q3 snapshot value) codes:

| Code | Meaning |
|---|---|
| `SW` | surface water |
| `SWP` | surface water purchased |
| `GU` | groundwater under the direct influence of surface water (GWUDI), regulated under the surface-water treatment rules |
| `GUP` | purchased GWUDI |
| `GW` / `GWP` | groundwater / groundwater purchased: **excluded** from the panel but kept in the violations table |

**Counts:**
- 20,022 SW-sourced CWS (SWP 12,579; SW 6,524; GU 760; GUP 159).
- 11,886 of them are active.
- 15,070 were active at some point in 2005Q1–2026Q2 and appear in the panel.

**Purchased-water systems (`SWP`/`GUP`)** buy treated water. Turbidity and SWTR treatment techniques apply to the treating wholesaler, so `hb_turbidity_tt` and `hb_swtr_tt` are structurally rare for them. DBP rules do apply to consecutive systems. Stratify by `primary_source_code`.

## Label definitions (violation level → `label`)

| `label` | Definition (SDWA codes) |
|---|---|
| `nitrate_mcl` | Health-based MCL violation (VIOLATION_CODE 01 single sample / 02 average), Nitrate Rule (RULE_CODE 331), CONTAMINANT_CODE 1040 nitrate, 1041 nitrite or 1038 nitrate-nitrite |
| `dbp_mcl` | Health-based MCL violation, RULE_CODE 210 / 220 / 230 (Stage 1 running annual average, Stage 2 locational running annual average (LRAA), former TTHM rule), CONTAMINANT_CODE 2950 TTHM or 2456 HAA5 |
| `dbp_other_mcl` | Health-based MCL violation, RULE_CODE 210, 1011 bromate or 1009 chlorite |
| `dbp_precursor_tt` | Health-based TT violation 46 "Treatment Technique Precursor Removal" (Stage 1 DBPR TOC removal), CONTAMINANT_CODE 2920 |
| `turbidity_tt` | Health-based TT violation 43 "Single Turbidity Exceed (Enhanced SWTR)" or 44 "Monthly Turbidity Exceed (Enhanced SWTR)" (RULE_CODE 122/123) |
| `swtr_tt_other` | Health-based TT violation 41 "Treatment Technique (SWTR and GWR)" or 42 "Failure to Filter (SWTR)" under RULE_CODE 121/122/123. Under the original SWTR, code 41 lumps turbidity-performance and disinfection failures; SDWIS does not separate them |
| `""` (no label) | Monitoring/reporting violations (`is_monitoring_reporting`). Also health-based violations outside these definitions: MRDL, uncertified operator (210/12), failure to address a deficiency (123/45), no prior approval, uncovered reservoir, report failures |

`label_group` (`nitrate`, `dbp`, `dbp_precursor`, `swtr`, `other`) groups violations by rule family, whether or not they are health-based.

**Labeled health-based violations:**

| `label` | All systems | SW-sourced CWS |
|---|---|---|
| `dbp_mcl` | 74,533 | 50,058 |
| `nitrate_mcl` | 41,051 | 2,170 |
| `swtr_tt_other` | 38,793 | 25,242 |
| `turbidity_tt` | 9,532 | 7,357 |
| `dbp_precursor_tt` | 5,825 | 5,452 |
| `dbp_other_mcl` | 201 | 140 |

The SW-sourced CWS labels come from 6,607 systems.

## PWS × quarter panel (`sdwis_pws_quarter_labels.parquet`)

**Rows.** One row per SW-sourced CWS per calendar quarter. Each system's panel runs from max(2005Q1, quarter of `first_reported_date`) to min(2026Q2, quarter of `pws_deactivation_date` for inactive systems). It starts in 2005Q1 because by then the Stage 1 DBPR (2002/2004) and the IESWTR/LT1ESWTR turbidity rules (2002/2005) applied to all SW/GU systems; earlier zeros are not comparable.

**Flag columns** (int8 0/1):
- `hb_<label>` = 1 if a health-based violation of that label has a **compliance period** (`compl_per_begin_date`..`compl_per_end_date`) overlapping the quarter. A missing end date counts as the begin quarter only.
- `hb_<label>_onset` = 1 if the compliance period **begins** in the quarter.
- `hb_swtr_tt` = `turbidity_tt` OR `swtr_tt_other`.
- `hb_any` = any label.
- `mr_nitrate`, `mr_dbp`, `mr_swtr` = a monitoring/reporting violation of that rule group overlaps the quarter. A 0 label in such a quarter is *unverified*, not clean.
- `recent_reporting_lag` = the quarter ends less than 365 days before the 2026-10-22 snapshot. These quarters are still being reported.

**System attributes:** `state_code`, `primacy_agency_code`, `primary_source_code`, `population_served`, `pop_cat_5_code`, `owner_type_code`, `pws_activity_code`.

**Panel prevalence** (share of 1,070,774 quarters):

| Flag | Share |
|---|---|
| `hb_dbp_mcl` | 3.39% (36,338 quarters) |
| `hb_swtr_tt` | 0.77% |
| `hb_dbp_precursor_tt` | 0.44% |
| `hb_turbidity_tt` | 0.37% |
| `hb_nitrate_mcl` | 0.07% (744) |
| `hb_dbp_other_mcl` | 0.01% |
| `hb_any` | 4.33% |

## Schema (`sdwis_violations.parquet`, main columns)

**Keys and codes:** `pwsid`, `violation_id`, `facility_id` (strings); `rule_code`, `rule_name`, `violation_code`, `violation_name`, `violation_category_code`, `contaminant_code`, `contaminant_name`, `is_health_based`, `label_group`, `label`, `is_monitoring_reporting`.

**Dates:** `compl_per_begin_date`, `compl_per_end_date`, `compl_per_days`, `non_compl_per_begin_date`, `non_compl_per_end_date`, `calculated_rtc_date`, `viol_first_reported_date`, `viol_last_reported_date`. Each has a matching `*_flag` column.

**Measures:** `viol_measure` (as reported), `unit_of_measure` (as reported), `viol_measure_mgl`, `viol_measure_qc`, `federal_mcl` (text), `state_mcl`.

**Status and notification:** `violation_status` (Resolved, Archived, Addressed, Unaddressed), `public_notification_tier`, `severity_ind_cnt`.

**Enforcement summary:** `n_enf_actions`, `n_enf_formal`, `n_enf_informal`, `n_enf_resolving`, `first_enf_date`, `last_enf_date`, `has_boil_water_order` (enforcement codes SFH/EFH), `enf_action_types`.

**System attributes:** `pws_type_code`, `primary_source_code`, `is_surface_source`, `state_code`, `population_served`.

## QC applied

- **De-duplication.** The violations file has one row per violation × enforcement action, and `VIOLATION_ID` alone is not unique (490,388 distinct IDs across 806,618 violations). Rows are therefore de-duplicated on (PWSID, VIOLATION_ID). Enforcement is summarised per violation.
- **Dates** are parsed from MM/DD/YYYY, and impossible values become NaT with a flag. The `<NA>` flag means OK.
  - Dates before 1975 are calendar placeholders → `out_of_range`. Affected: 168 compliance-begin dates (years 1900 and 1902), and 1,002 non-compliance-end / return-to-compliance dates (for example 683 in 1950).
  - `--->` → `open_ended` (1,355 non-compliance end dates).
  - A compliance end before its begin → `before_begin` (4).
  - Report dates must be on or before the snapshot. Compliance-end and return-to-compliance dates must be on or before the snapshot + 5 years.
- **Units.** `viol_measure` is kept together with `unit_of_measure`. `viol_measure_mgl` is derived only for explicit MG/L, UG/L or NG/L units (factors 1, 1e-3, 1e-6). Blank units are never guessed: for example 13,979 nitrate violations have no unit.
- **Measure plausibility.** `viol_measure_qc` flags MCL-violation measures in two ways (values are never changed):
  - `gt_100x_federal_mcl` (406): likely unit-entry errors, for example TTHM 92 entered as MG/L.
  - `lt_federal_mcl` (321): below the federal MCL. Causes include stricter state MCLs, unit errors, and averaging.
- **Privacy.** These `SDWA_PUB_WATER_SYSTEMS` columns are dropped at parse time and never written: ORG_NAME, ADMIN_NAME, EMAIL_ADDR, PHONE_NUMBER, PHONE_EXT_NUMBER, FAX_NUMBER, ALT_PHONE_NUMBER, ADDRESS_LINE1, ADDRESS_LINE2, CITY_NAME, ZIP_CODE, COUNTRY_CODE. They hold administrative contacts, usually a person. Test fixtures replace them with `REDACTED`.
- **IDs** are strings with leading zeros kept. Examples: tribal `070000002`, alphanumeric Washington IDs such as `WA53AA220`, violation ID `0920641`.

## Known biases and caveats

1. **Violations are not exceedance events.**
   - Compliance periods are monitoring periods: monthly for SWTR, quarterly for LRAA, monthly to annual for nitrate (median 91 days, 90th percentile 365 days).
   - A Stage 2 DBP violation is a locational running annual average over four quarters. It lags and smooths the underlying water quality.
   - Use `*_onset` for timing, and never compute lead times against compliance dates.
2. **"No violation" ≠ compliance.** A system that missed monitoring gets a monitoring/reporting violation, not an MCL violation. Use `mr_*` and treat those quarters as uncertain.
3. **Small-system under-reporting and primacy-agency differences.** States (and EPA regions for tribes) report to SDWIS/Fed with different practices and completeness. Evaluate leave-one-state-out, and never let one system span train and test.
4. **Reporting lag.** For health-based violations since 2015, the gap from compliance-period begin to first report has median 148 days, 75th percentile 226 days and 90th percentile 419 days. 2026Q2 already shows about 20% fewer violations than earlier quarters, so drop or down-weight quarters flagged `recent_reporting_lag`.
5. **Treatment confounds DBPs.** Disinfectant type, chloramine conversions and precursor removal change DBPs independently of source water. Raw-water TOC and bromide (`labels_ucmr`) are the treatment-free target.
6. **Snapshot attributes.** Source type and population are 2026Q3 values; a system that switched source is labelled with its current source throughout.
7. **Effort and status artefacts.** "Archived" status means more than 5 years past the end of non-compliance, or the system became inactive. It is not a resolution.

## Joins

**No coordinates.**
- `sdwis_source_facilities` lists intakes and wells without coordinates. Locations are withheld for security, so the source-water location is not public.
- Link a PWS spatially through `labels_cws_boundaries` on `pwsid` (service-area polygons; 43% are modeled). Treat the result as a proxy (see that datasheet).
- For purchased water, follow `facility_type_code == "CC"` → `seller_pwsid` to the treating wholesaler. 99.6% of active purchased-surface-water CWS have a seller link.
- 91.8% of self-supplied SW/GU CWS have an active intake or reservoir facility record.

**Other keys:**
- `sdwis_areas_served`: county, city and ZIP served.
- `labels_ucmr`: same PWSID.

## Rebuild

```
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_sdwis.py
```

Options: `--keep-zip`, `--panel-start 2005Q1`, `--panel-end 2026Q2`, `--crosscheck 4`, `--rebuild` (forces a re-scan).

The script downloads the zip (558 MB, about 25 s), reuses or creates the interim scan (85 s), writes `processed/labels_sdwis/` and `manifests/labels_sdwis.json`, and then deletes the zip. Peak disk use is about 0.7 GB. Tests: `python -B -m pytest tests/test_labels_drinking_water.py`.
