# Datasheet — `labels_noaa_lake_erie_hab` (NOAA western Lake Erie cyanoHAB bloom Severity Index, all vintages)

| Item | Value |
|---|---|
| Publisher | NOAA National Centers for Coastal Ocean Science (NCCOS), HAB Monitoring & Forecasting program |
| Product page | https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/lake-erie/ (verified 2026-10-22; snapshot kept in raw) |
| Retrieval date | 2026-10-22 |
| Access | Plain HTTPS downloads, no login, no key |
| License / terms | US Government work (public domain). NOAA asks for the attribution: *"The Lake Erie Severity Index (SI) is derived from NOAA's National Centers for Coastal Ocean Science (NCCOS) HAB Monitoring Program."* |
| Client / downloader | `sentinel/data/sources/events_noaa_lake_erie.py` / `scripts/data/download_events_noaa_lake_erie.py` |
| Tests | `tests/test_labels_noaa_lake_erie.py` (fixtures in `tests/fixtures/labels_noaa_lake_erie_hab/`: both real spreadsheets and real `pdftotext` outputs of bulletins) |
| Manifest | `$SENTINEL_DATA/manifests/labels_noaa_lake_erie_hab.json` (25 files, 36 MB incl. raw PDFs; sha256 per file) |

## What the label is

The **bloom Severity Index (SI)** is one number per year for the western basin of Lake Erie. It comes from satellite Cyanobacteria Index (CI) composites: CI is summed over the three highest 10-day composites (≈ 30 days of peak bloom biomass, `total_CI_avg_30d`), and SI = 10·(log10 CI − log10 CI₂₀₀₅)/(log10 CI₂₀₁₁ − log10 CI₂₀₀₅). NOAA floors SI at 0.3 ("If the calculated SI<0.3, then the SI was set to 0.3"), so 0.3 values are left-censored (`si_floor_censored`). It is a **basin-scale annual label (n = 26 years)**. It suits seasonal-severity skill tests and context; it has far too few rows to train a model on.

## The series was revised: label vintages are kept side by side

After the 2024 season NOAA reprocessed the full MERIS (2002–2011) / MODIS (2000–2018) / OLCI (2016–) record. They used new MODIS↔MERIS/OLCI machine-learning alignment, NASA CyAN v6 OLCI calibration and atmospheric correction, and a recompiled MERIS set. Sources: `HAB_LE_TimeseriesUpdate_2025.pdf` (Last-Modified 2025-05-07) and the forecast recalibration note dated 2026-04-22. Early-season forecast models were recalibrated to the new series before the 2026 season.

| `label_vintage` | File (official URL) | Server Last-Modified | Years | Columns |
|---|---|---|---|---|
| `curated_2000to2025` (current) | `nccospublicstor.blob.core.windows.net/hab-data/bulletins/lake-erie/2025/NOAA_NCCOS_2000to2025_Curated_LE_Annual_Severity.xlsx` | 2025-12-04 | 2000–2025 (26) | CI max, area max, CI 30-d, area 30-d, SI, date of CI max |
| `revised_2000to2024` (first reprocessed release) | `cdn.coastalscience.noaa.gov/hab-data/bulletins/lake-erie/2025/NOAA_NCCOS_LE_Bloom_SeverityIndex_Revised2000to2024.xlsx` | 2025-07-03 | 2000–2024 (25) | CI 30-d, SI only |
| as-issued | Season-end bulletins 2018–2025 (PDF) | per bulletin | primary statements 2018–2025; quoted values for 2011, 2015, 2016, 2017 | SI as published that season |

Selected years (from `severity_vintage_comparison.parquet`; all values were extracted by code from the files):

| Year | As issued (bulletin) | Revised 2000–2024 | Curated 2000–2025 |
|---|---|---|---|
| 2011 | 10 (quoted in 2018–2020 bulletins) | 10.0 | 10.0 |
| 2015 | 10.5 (quoted in 2018–2020 bulletins) | 8.3 | 8.4 |
| 2017 | 8.0 (quoted in 2018/2019 bulletins) | 6.8 | 7.4 |
| 2019 | 7.3 | 5.7 | 5.9 |
| 2021 | 6.0 | 3.0 | 3.4 |
| 2023 | 5.3 | 2.5 | 2.9 |
| 2024 | 6.6 | 3.9 | 4.2 |
| 2025 | 2.4 (already on the new scale) | — | 2.4 |

The two reprocessed files also differ from each other (e.g. 2017: 6.8 vs 7.4; 2006: 2.1 vs 2.5). This is why **every row carries `label_vintage`, `vintage_last_modified` and `file_sha256`.** A model or evaluation must name its vintage. A comparison of skill against pre-2025 published forecasts must use the as-issued scale, because those forecasts targeted it.

Raw spreadsheets are stored with the server Last-Modified date in the file name (`severity/2025-12-04_…xlsx`). An in-place update by NOAA, such as a 2026 row being added, therefore creates a new file instead of overwriting the frozen vintage.

## Files

Processed (`$SENTINEL_DATA/processed/labels_noaa_lake_erie_hab/`):

| File | Rows | Content |
|---|---|---|
| `severity_index.parquet` | 51 (26 + 25) | One row per vintage × year: `label_vintage, year, total_CI_max, total_area_km2_max, total_CI_avg_30d, total_area_km2_avg_30d, severity_index, date_ci_max (+flag), si_floor_censored, vintage_last_modified, file_url, file_sha256`. Columns a vintage lacks are NaN; for `date_ci_max` the flag is `absent_in_vintage` |
| `severity_as_issued.parquet` | 25 | `season` (bulletin), `year`, `si`, `statement_type` (`primary` = the bulletin's own season; `mentioned` = value quoted for another year), `context` (matched text), `doc_url`, `retrieved_via` |
| `seasonal_forecasts.parquet` | 6 | Seasons 2018, 2020, 2021, 2023, 2025, 2026: `issue_date, forecast_si, forecast_si_lo, forecast_si_hi, parse_ok, context`. This is the operational seasonal baseline; it targeted the as-issued scale until 2025 |
| `severity_vintage_comparison.parquet` | 26 | year × {curated, revised, as-issued primary} |
| `severity_metadata_sheets.json` | — | The "Metadata" sheets of both spreadsheets (definitions, formula, attribution) |

Raw (`$SENTINEL_DATA/raw/noaa_lake_erie_hab/`): both spreadsheets, 16 bulletin PDFs (final assessments 2018–2025, seasonal forecasts 2018/2020/2021/2023/2025/2026, the time-series update and the recalibration note), their `pdftotext` outputs, the product-page snapshot and `retrieval_2026-10-22.json`. The retrieval file records URL, HTTP Last-Modified, ETag, bytes, sha256 and retrieval route for each file.

## QC and parsing rules

- The spreadsheets are read with the stdlib XLSX reader (`events_common.read_xlsx`). Excel's 17-digit floats are shortened (1.1000000000000001 → 1.1). Date-formatted cells are converted to ISO dates.
- Bulletin numbers come from `pdftotext -layout`, matched on lower-cased, **whitespace-stripped** text. The 2018–2021 PDFs have kerning artefacts ("sev erityind ex of 4. 5").
- Primary-statement patterns: "bloom in YYYY had a severity index (SI) of X" and "the YYYY [western] Lake Erie cyanobacterial bloom had a severity index (SI) of X".
- Forecast pattern: "severity [index] of [about|~] X, [but this could|with a potential] range between/of A–B" (2018: "measure 6 on the severity index, but could range between 5 and 7.5").
- All 8 final assessments and all 6 forecasts parsed. The tests pin the values on fixtures.

**Documents no longer at their official URL.** Two were fetched from Internet Archive captures of the official files and are flagged in `retrieved_via`:
- 2025 season-end assessment (`cdn…/2025/SeasonalAssessment_2025_08.pdf`, capture 2025-12-05)
- June 2025 seasonal forecast (`nccospublicstor…/2025/SeasonalForecast_2025_06.pdf`, capture 2025-07-05)

No final-assessment PDF was found for seasons before 2018. As-issued values for 2011/2015/2016/2017 exist only as quotations inside the 2018–2020 bulletins.

## Known biases and caveats

- **n = 26 years** at one location (western basin). Report seasonal skill with leave-one-year-out and say "underpowered" (EW-v2 §12.6 requires ≥ 50 test onsets per hazard).
- **Sensor changes** (MERIS → MODIS → OLCI) are absorbed by inter-calibration. Before 2016 the values depend on MODIS/MERIS alignment and changed the most between vintages.
- The metadata of `revised_2000to2024` says the scale is anchored to "maximum observed bloom biomass (2015)", but its formula uses 2011. The curated file's formula uses 2011. This is recorded as published.
- **2026:** the seasonal forecast (3.5, range 3–4.5, issued 2026-06-25) is stored. The 2026 season-end SI had not been published by 2026-10-22.
- The SI measures biomass, not toxin. Toxin labels for Lake Erie come from GLERL weekly monitoring or WQP (other workstreams).
- The NCCOS satellite products feed the SI. A SENTINEL model that also ingests CyAN/OLCI CI for the western basin is partly circular with this label.

## Join to sites

- **Spatial unit:** the western basin of Lake Erie, a single unit. Join on `year` only, for example to Maumee River (USGS 04193500 Waterville) spring TBP/discharge loads, which are the official forecast predictors, or to basin-aggregated SENTINEL predictions.
- **Lake identifiers:** use the Lake Erie western-basin polygon from NHDPlus or GLAHF if a polygon is needed. No COMID applies.

## Rebuild

```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -I scripts/data/download_events_noaa_lake_erie.py
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_noaa_lake_erie.py
```

The rebuild needs poppler `pdftotext` (`/usr/bin/pdftotext` on this host). It takes about 1 minute.

## Related: EPA/NASA CyAN lake-week bloom labels

These were not built here; see `labels_cyan_availability.md`.
