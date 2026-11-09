# Datasheet — `labels_ucmr` (UCMR 4 source-water DBP precursors, HAAs, cyanotoxins)

Use: TapRisk (D4). The cleanest drinking-water target in this set is raw-water total organic carbon (TOC) and bromide, the DBP precursors. They are measured before treatment, so treatment does not confound them, unlike finished-water DBPs or violations.

## Source

| Item | Value |
|---|---|
| Publisher | US EPA Office of Water, Fourth Unregulated Contaminant Monitoring Rule (UCMR 4) |
| Dataset | UCMR 4 occurrence data text files (standardized March 2024) |
| URL (verified 2026-10-22) | https://www.epa.gov/system/files/other-files/2024-04/ucmr4-occurrence-data.zip (10,628,817 bytes; Last-Modified 2024-04-04; sha256 `932dd461…d17b255`) |
| Landing page | https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule |
| Access | Direct download, no key or login |
| License / terms | US federal government data (EPA), public domain; the page states no dataset-specific license |
| Retrieved | 2026-10-22 |
| Data dictionary | `UCMR4_DataSummary_508.pdf` inside the zip (Tables 5, 7, 8). `Instructions for Accessing UCMR Results_508.pdf` explains how censoring is encoded |

**UCMR 5 (2023–2025) was checked and not used.** The zip (`ucmr5-occurrence-data.zip`, Last-Modified 2026-10-15) holds only 29 PFAS and lithium. All samples are entry-point (`EP`) samples. It has no TOC or bromide. `--check-ucmr5` re-verifies this and records the result in the manifest.

## Files (`$SENTINEL_DATA/processed/labels_ucmr/`)

| File | Rows | Content |
|---|---|---|
| `ucmr4_source_water.parquet` | 49,454 | Raw/source water (`SR`): TOC 24,721 (3,438 PWS) and bromide 24,733 (3,443 PWS). **Total 3,444 PWS (matches the research figure).** |
| `ucmr4_haa.parquet` | 190,359 | HAA5 63,484 / HAA6Br 63,443 / HAA9 63,432 at distribution-system points (`DS`), 4,924 PWS |
| `ucmr4_cyanotoxins.parquet` | 105,865 | Entry-point (`EP`) samples at 3,487 PWS: total microcystins 35,000 (8 detections), anatoxin-a 35,405 (132), cylindrospermopsin 35,425 (13), plus 7 congeners × 5 rows (4 detections). **157 detections in total** |
| `ucmr4_haa_ade.parquet` | 257,656 | Disinfectant type and residual, plus treatment information, per HAA sample event |
| `ucmr4_cyanotoxin_ade.parquet` | 249,943 | Bloom and toxin questionnaire, disinfectant type and treatment, per cyanotoxin sample event |
| `ucmr4_zipcodes.parquet` | 19,323 | ZIP codes served per PWS (5-character strings) |
| `build_stats.json` | — | All counts quoted here |

Raw: `$SENTINEL_DATA/raw/ucmr/ucmr4-occurrence-data.zip` is kept, because it is under the 100 MB rule. `ucmr4_zip_info.json` holds its headers, sha256 and member list.

## Coverage

- **Sample dates:**
  - Source water: 2018-01-02 to 2021-11-09.
  - HAA: 2018-01-02 to 2021-12-13.
  - Cyanotoxins: 2018-03-05 to 2021-09-30.
- **Places:** 61–62 states, territories and tribal regions.
- **Design:** all PWS serving more than 10,000 people, plus a nationally representative random sample of 800 small PWS (≤ 10,000).
- **Source-water PWS by facility water type:** SW 1,467; GU 103; MX 45; GW 1,871. A PWS can have several facilities. GW facilities are **included**, so filter on `facility_water_type ∈ {SW, GU, MX}` for surface-water work.

## Label definitions

| Target | Definition | Notes |
|---|---|---|
| Raw-water TOC (µg/L) | `analyte == "total organic carbon"`, `sample_point_type == "SR"` | MRL 1,000 µg/L (= 1 mg/L). 59.1% censored. Continuous or censored-regression target |
| Raw-water bromide (µg/L) | `analyte == "bromide"`, `SR` | MRL 20 µg/L. 29.2% censored |
| HAA5 high | HAA5 > 60 µg/L (MCL value) | The MCL applies to a locational running annual average, so **a single sample above 60 is not a violation**. Research count: 2.1% of single results |
| Cyanotoxin detection | `censored == False` for a cyanotoxin | Reference values: EPA 2015 10-day health advisory for total microcystins, 0.3 µg/L (bottle-fed infants, young children) and 1.6 µg/L (school-age and adults). Cylindrospermopsin: 0.7 / 3 µg/L |

## Schema (results tables)

The three results tables share one schema:

- **IDs:** `pwsid`, `facility_id`, `sample_point_id`, `sample_id`, all strings. Leading zeros are kept (for example the tribal PWSID `010106001` and facility `00007`).
- **System and sample:** `pws_name`, `pws_size` (S ≤ 10,000; L > 10,000; per SDWIS/FED 2015-12-31), `state`, `epa_region`, `facility_water_type` (SW, GW, GU, MX), `sample_point_type` (SR, EP, DS), `sample_event_code` (SE1–SE4; SE5–SE8 for cyanotoxins), `collection_date` and `collection_date_flag`.
- **Result:** `analyte`, `method_id`, `result_sign` (`<` or `=`), `censored`, `value`, `detection_limit`, `mrl`, `unit`.
  - `value` is NaN when censored.
  - `detection_limit` equals the MRL when censored.
  - `unit` is `ug/L`; the source has `µg/L`. This is a relabel only; no conversion is applied.

Additional-data-element (ADE) tables have these columns: `pwsid`, `facility_id`, `sample_point_id`, `sample_event_code`, `element`, `response`, `response_desc` and `if_other_text`. `response_desc` is decoded from Data Summary Tables 7–8. Join to results on (`pwsid`, `facility_id`, `sample_point_id`, `sample_event_code`).

## QC applied

- **Censoring kept** as `censored` plus `detection_limit`. There is no DL/2 substitution.
- **HAA groups:** values are reported sums. They have no MRL and never carry `<`. 7,473 values are exactly 0 and are kept as 0.
- **Dates:** parsed from M/D/YYYY. Anything outside 2017-01-01 to 2021-12-31 would be set to NaT and flagged; none occurred.
- **Missing values:** unknown result signs would set the value to NaN and be flagged; none occurred.
- **IDs:** read as strings.
- **Undecoded ADE codes:** 19 responses use a code that belongs to another element (for example `CL2` under TreatmentInformation). They are kept as reported, with `response_desc` empty.

## Known biases and caveats

1. **Sampling design.**
   - HAA monitoring applied only to systems subject to the D/DBPR HAA5 requirements.
   - Cyanotoxin monitoring applied only to SW/GU systems, with four sample events in one season. It does not follow blooms.
   - Source-water TOC and bromide were sampled only alongside HAA sample events (SE1–SE4: two events for GW, four for SW/GU).
2. **Censoring is heavy.** TOC is 59% below its 1 mg/L MRL. Cyanotoxins are more than 99.6% non-detects.
3. **One snapshot.** Each PWS was sampled within a single 12-month window between 2018 and 2020, so this is not a time series.
4. **Small systems are a random sample** of 800, while all large systems are included. Weight accordingly in national summaries.
5. **The bloom questionnaire** (`ucmr4_cyanotoxin_ade`) is self-reported by the system. For example, "DK" (don't know) answered 3,691 bloom-occurrence questions.

## Joins

- **Systems and violations:** join on `pwsid` to `labels_sdwis` (systems, violations) and to `labels_cws_boundaries` (service-area polygon).
- **ZIP codes:** `ucmr4_zipcodes` gives the ZIP codes served.
- **No coordinates.** The `SR` sample point is the untreated water at a facility. Intake locations are not public, so any link to SENTINEL river or lake predictions goes through the service-area or upstream-catchment *proxy* described in `labels_cws_boundaries.md`.

## Rebuild

```
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_ucmr.py --check-ucmr5
```

This reuses `raw/ucmr/ucmr4-occurrence-data.zip` if present, rewrites `processed/labels_ucmr/`, and writes `manifests/labels_ucmr.json`. Tests: `python -B -m pytest tests/test_labels_drinking_water.py`.
