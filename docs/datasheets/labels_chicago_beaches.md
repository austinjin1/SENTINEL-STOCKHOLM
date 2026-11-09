# Datasheet: `labels_chicago_beaches`: Chicago beach FIB labels, official predictions, beach sensors

| | |
|---|---|
| **Use in SENTINEL** | SafeSwim (RESEARCH_PLAN §8 D1, `docs/research/07` §d.1): primary lab labels (culture *E. coli*; qPCR *Enterococci*), the incumbent operational model for re-scoring, and in-lake/weather covariates. Also EW-v2 H6 (beach FIB) context. |
| **Publisher / data owner** | Chicago Park District (attribution on every lab and sensor dataset). The predictions dataset is attributed to the City of Chicago. Published on the City of Chicago Data Portal (Socrata). |
| **Retrieved** | 2026-10-22 (UTC), full pulls, no key, no login |
| **Manifest** | `$SENTINEL_DATA/manifests/labels_chicago_beaches.json` (23 files, sha256 per file; request parameters and all counts in `stats`) |
| **Client / downloader** | `sentinel/data/sources/downstream_chicago_beaches.py` · `scripts/data/download_labels_chicago_beaches.py` |
| **Tests** | `tests/test_labels_chicago_beaches.py` (19 tests). Fixtures in `tests/fixtures/labels_chicago_beaches/` are real rows cut from the 2026-10-22 downloads. |

## 1. Sources (verified 2026-10-22)

Each dataset was pulled through the SODA 2.1 CSV endpoint `https://data.cityofchicago.org/resource/<id>.csv` (`$order=:id`, pages of 50,000). The row count was checked against `count(*)` from the API. The `/api/views/<id>.json` metadata is stored next to each raw file.

| Key | Dataset (portal page) | Rows | Source "rows updated" | Licence field |
|---|---|---|---|---|
| lab | Beach Lab Data, https://data.cityofchicago.org/d/2ivx-z93u | 34,967 | 2026-10-08 | none |
| predictions | Beach E. coli Predictions, https://data.cityofchicago.org/d/xvsz-3xcj | 11,385 | 2025-09-02 | none |
| advisories | Beach Swim Advisories (historical, 2016), https://data.cityofchicago.org/d/t62e-8nvc | 972 | 2026-10-21 | none |
| water_sensors | Beach Water Quality – Automated Sensors, https://data.cityofchicago.org/d/qmqz-2xku | 49,151 | 2026-10-22 | "See Terms of Use" |
| weather_sensors | Beach Weather Stations – Automated Sensors, https://data.cityofchicago.org/d/k7hf-8y75 | 210,568 | 2026-10-22 | "See Terms of Use" |
| sensor_locations | Beach Water and Weather Sensor Locations, https://data.cityofchicago.org/d/g3ip-u8rb | 9 | 2015-06-04 | none |

**Superset check.** The lab table `2ivx-z93u` is the parent of three filtered views:
- `hmqm-anjq`: DNA tests, 17,594 rows
- `hh4t-tnq8`: culture tests, 18,002 rows
- `9wd3-tqr2`: both tests, 629 rows

17,594 + 18,002 − 629 = 34,967, which equals the parent row count. Only the parent is downloaded; the view counts are stored in `raw/chicago_beaches/lab_view_counts.json`.

**Licence and terms.** The datasets declare no open licence. The licence field is empty, or says "See Terms of Use". The City of Chicago Data Terms of Use apply (https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html):
- The data are provided "as is", with no warranty.
- The City may require use to stop.
- Any derivative application must display the City's disclaimer: *"This site provides applications using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago. The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at this site. The data provided at this site is subject to change at any time. It is understood that the data provided at this site is being used at one's own risk."*

Research use is fine. **The SENTINEL Console must show this disclaimer if it displays these data.**

**Model documentation used.** The predictions come from the City's "Clear Water" project:
- MIT-licensed code: https://github.com/Chicago/clear-water (`Master.R`) and https://github.com/Chicago/clear-water-app (`app.R`; last commit 2019-05-24).
- `app.R` shows the production "DNA Model". It is a `randomForest` fed with the **same-day** qPCR geometric means at **5 input beaches** (Calumet, 63rd Street, Rainbow, Montrose, South Shore). It runs only once all five results are posted, and predicts 15 other beaches.

## 2. Files

**Raw** (`$SENTINEL_DATA/raw/chicago_beaches/`, 6.2 MB): `<key>_<id>.csv.gz` (all-string CSV exactly as served), `<key>_<id>_metadata.json`, `lab_view_counts.json`, `download.log`.

**Processed** (`$SENTINEL_DATA/processed/labels_chicago_beaches/`, 9.0 MB, Parquet/zstd):

| Table | Rows | Content |
|---|---|---|
| `culture_ecoli.parquet` | 18,002 | One row per culture test (*E. coli*, CFU/100 mL), 2006-05-23 → 2016-09-29 |
| `qpcr_enterococci.parquet` | 17,592 | One row per DNA/qPCR test (*Enterococci*, CCE/100 mL), 2015-05-26 → 2026-09-07 |
| `predictions_official.parquet` | 11,385 | Official beach-day predictions, 2017-05-26 → 2025-09-01 (the incumbent) |
| `advisories_2016.parquet` | 972 | 2016 culture-era model: predicted level, P(≥ 235 CFU), advisory Y/N |
| `water_sensors_long.parquet` | 255,295 | Hourly in-lake sensors, long format (49,143 measurements × variables), 2013–2026 |
| `weather_sensors_long.parquet` | 2,530,976 | Hourly beach weather stations, long format (210,568 measurements), 2015–2026 |
| `sensor_locations.parquet` | 9 | Sensor lat/lon (6 water, 3 weather) |
| `beaches.parquet` | 27 | Beach lookup: key, name variants, lat/lon, per-dataset counts, model roles |
| `predictions_vs_qpcr_matched.parquet` | 9,549 | Official prediction × same-day observed qPCR at the same beach (+ previous qPCR) |
| `build_stats.json` | — | All counts quoted here |

## 3. Coverage

| Table | Units / sites | Time | Labels |
|---|---|---|---|
| culture | 27 beach keys, plus 376 tests whose beach is literally "NA" | 11 seasons 2006–2016 (May–Sep; one March record), 1,446–1,734 tests/yr | 17,941 with a result; **2,508 (14.0 %) > 235 CFU** |
| qPCR | 20 beaches | 2015 (270, pilot), 2016 (628), 2017–2019 (~2,000/yr), **2020 (47; 6–7 July only)**, 2021–2026 (1,710–1,898/yr) | **1,227 (7.0 %) > 1,000 CCE** |
| predictions | 15 beaches (the city's predicted set) | 2017–2025, 1,350–1,470/yr; **2020: 30 rows**; none yet for 2026 (dataset last updated 2025-09-02) | source "DNA Model" only |
| advisories | 9 beaches | 2016-06-04 → 2016-09-22 | 44 advisories ("Y") |
| water sensors | 6 beaches (63rd St, Calumet, Montrose, Ohio St incl. an "Old – Ohio Street" position, Osterman, Rainbow) | 2013 (5), 2014–2016 (7.6k–14.3k/yr), 2017–2026 (0.3k–3.3k/yr), **no 2022 data** | covariates |
| weather | 3 stations (63rd St, Foster, Oak St) | 2015–2026, 12.7k–25.2k/yr | covariates |

**Re-scoring check.** I matched the official predictions to same-day qPCR at the same beach, with qPCR aggregated to a beach-day geometric mean:
- All matched pairs: **9,549 beach-days at 14 beaches, 2017–2025, 494 > 1,000 CCE.** Juneway is predicted but never qPCR-tested.
- Restricted to beach-days whose previous qPCR at that beach is ≤ 3 days earlier (the persistence-eligibility rule of `docs/research/07` §b.2): **9,436 beach-days, 491 exceedances**.

The restricted counts reproduce the research figures exactly. No model was scored here, and nothing was written to the registry.

## 4. Label definitions

| Column | Table | Definition |
|---|---|---|
| `result` | culture, qPCR | Test-level concentration: the **geometric mean of the replicate readings present** (1–2 per test), recomputed here. A zero replicate gives 0. NaN when there is no reading. |
| `exceeds_235_cfu` | culture | `result > 235` CFU/100 mL. This is the EPA 2012 RWQC *E. coli* BAV and the City's culture-era advisory trigger (dataset description). Nullable boolean: NA when there is no result. |
| `exceeds_1000_cce` | qPCR | `result > 1000` CCE/100 mL, the City's qPCR advisory level (dataset description: "exceed USEPA recommended levels, which is 1000* CCE"). |
| `qpcr_gmean_cce` | matched | Beach-day geometric mean of the day's qPCR results (`daily_qpcr()`); `n_tests` = tests that day |
| `predicted_level` | predictions | Official model output, as published. See the unit caveat in §7. |
| `swim_advisory` | advisories | Advisory issued (Y/N), 2016 culture-era model. **Policy decision, not a measurement.** |

**Decision-time rules** (enforce in every feature builder):
- **Culture** results arrive 18–24 h after sampling. Never use them as a same-day input.
- **qPCR** results arrive about 4–5 h after sampling. Sampling is at a median of 11:00 local (5th–95th percentile 11:00–12:00), so results arrive mid-afternoon. A same-day neighbour qPCR is a legitimate input only for an afternoon nowcast.
- **The official prediction** exists only after the 5 input beaches' same-day qPCR is posted. It is a same-day nowcast, not a morning forecast.
- **Persistence** baseline: the previous qPCR day at the same beach (`prev_qpcr_gmean_cce`, `prev_gap_days`).

## 5. Schema (main columns)

**`culture_ecoli` / `qpcr_enterococci`**

| Column(s) | Meaning |
|---|---|
| `culture_test_id` / `dna_test_id` | Test ID (string) |
| `beach`, `beach_key` | Raw name, and the normalised key (§8) |
| `sample_timestamp_raw` | Timestamp exactly as published |
| `sample_time_local` | Naive local wall-clock time |
| `time_utc` | UTC instant; NaT when the time is unknown or falls in a DST gap or overlap |
| `date` | Local date |
| `date_flag` | `''`, `year_repaired`, `missing`, `unparseable`, `out_of_range`, `dst_nonexistent`, `dst_ambiguous` |
| `time_known` | False if no sample time was recorded (published as 00:00) |
| `indicator`, `unit` | Organism and unit (one value per table) |
| `reading_1`, `reading_2`, `reading_mean_published`, `result` | Replicate readings, the published mean, and the recomputed result |
| `mean_check` | `ok`, `published_sqrt_of_single_replicate`, `published_mean_without_replicates`, `published_mean_missing`, `published_mean_mismatch`, `no_result` |
| `n_replicates` | Number of replicate readings present |
| `censored`, `censor_side`, `detection_limit` | Censoring (§6); `censor_side` is `left`, `right` or `both`; `detection_limit` = 2419.6 for right-censored culture |
| `qualifier` | Original text of anything masked |
| `latitude`, `longitude`, `coord_flag` | Coordinates; `coord_flag` is `''`, `from_beach_lookup` or `missing` |
| `exceeds_235_cfu` / `exceeds_1000_cce` | The label (§4) |

Culture only adds `sample_2_time_utc`, `sample_2_date_flag`, `sample_interval_min` and `culture_note` (always empty in this release).

**`predictions_official`:** `record_id, beach_name, beach_key, date, date_flag, prediction_source, predicted_level, qualifier, predicted_level_unit_as_published, latitude, longitude, coord_flag`.

**`advisories_2016`:** `record_id, beach_name, beach_key, date, date_flag, predicted_level_cfu, prob_ge_235_cfu, swim_advisory, swim_advisory_raw, qualifier`.

**`*_sensors_long`:** `sensor_name, beach_key, measurement_id, time_local, time_utc, date, date_flag, variable, value, unit, qualifier`. Use `sensors_wide()` for a wide table.

Units:
- Water: `water_temperature` degC; `turbidity` NTU; `transducer_depth`, `wave_height` m; `wave_period` s; `battery_life` V.
- Weather: degC, %, mm/h, mm, deg, m/s, hPa, W/m², V.
- `precipitation_type` is a code: 0 none, 60 liquid, 70 solid, 40 unspecified. One record has code 5, which is undocumented.

**`beaches`:** `beach_key`, `names`, `latitude`/`longitude` (median of published coordinates), `n_distinct_coords`, `n_culture`, `n_dna`, `n_predictions`, `n_advisories`, `n_water_sensor_hours`, `is_predicted_beach`, `is_model_input_beach`, `water_sensor_latitude`/`_longitude`.

## 6. QC applied at ingest

- **Year repair.** 31 timestamps had two-digit-year errors and were repaired from `00YY` to `20YY`, flagged `year_repaired`, with the raw text kept:
  - 19 qPCR tests dated `0023-06-08` / `0023-06-23`. Neighbouring test IDs are dated 2023-06-06 … 2023-06-23.
  - 12 Humboldt culture tests dated `0016-09-…`. Neighbouring IDs are dated 2016-09.

  Any other impossible date becomes NaT and is flagged. There were none.
- **Unknown sample times.**
  - All culture tests before 2016 lack a sample time (the source says so).
  - 3,469 qPCR tests are published at 00:00, mostly in 2017 (1,883) and 2018 (1,251).

  For these, `time_known=False` and `time_utc` is NaT; the date is kept. DST gaps and overlaps become NaT with a flag and are never guessed.
- **Published-mean artifacts.** Labels use the recomputed `result`, and the published mean is kept. Recomputing changes 11 culture labels; it changes no qPCR labels.
  - 90 single-replicate culture tests: the published mean equals √x, as if the missing replicate were 1. Example: a single reading of 2420 is published as 49.2.
  - 61 culture rows: published mean = 1 but no readings at all, so `result` is NaN.
  - 39 culture rows: readings present (some containing a 0) but no published mean.
  - 8 qPCR tests: readings (x, 1) are published as x, flagged `published_mean_mismatch`.
- **Censoring.** Nothing is substituted.
  - Culture readings cluster at 2419.6–2420.6. This is consistent with the IDEXX Quanti-Tray/2000 upper limit of 2419.6 MPN/100 mL. That is an inference: the City documents CFU, and the method is unverified. 402 tests with a replicate in [2419, 2421] are flagged right-censored with `detection_limit` 2419.6. One reading of 6488 suggests a diluted sample and is not flagged.
  - Zero readings (35 culture, 22 qPCR tests) are flagged left-censored. The reporting limit is not published.
- **Duplicates.** DNA tests 69 and 541 each appear on two parent rows (two culture tests that day) and are de-duplicated. Repeat tests on the same beach-day are kept: 44 culture and 40 qPCR beach-days have more than one test (at most 3). Aggregate to beach-day for modelling.
- **Sensors.**
  - Values ≤ −9999 (−100000, −99999.992) and wind 999.9 → NaN, qualifier `missing_code:<raw>`: 6,944 water and 25 weather values.
  - Outside the physical plausibility ranges in `WATER_VARS` / `WEATHER_VARS` → NaN, qualifier `out_of_range:<raw>`: 4 water values (water temperature 36.6–46.6 °C) and 3 weather values (pressure 3098.5 hPa, maximum wind 445.2 m/s, interval rain −0.9 mm).
  - 48 all-zero placeholder records (42 water, 6 weather, all March 2025; battery also 0) → NaN, qualifier `all_zero_record`.
  - Turbidity exactly 0 is kept but flagged `zero_reading` (2,818 values).
- **Advisories.** 54 rows carry predicted level −1000 or −1002 together with probability −1, presumably meaning no prediction was issued. Both values are set to NaN with a qualifier.
- **Coordinates.** Checked against a Chicago lakefront bounding box. Culture-era rows have no coordinates in the source, and 2,355 of them were filled from the beach lookup (`coord_flag = from_beach_lookup`). 662 remain missing: 376 beach "NA" and 286 "Columbia", which has no coordinates anywhere.
- **IDs** are strings with no float artifacts.

## 7. Known issues, biases and caveats

1. **Method change: never pool.** The labels change organism, unit and threshold over time:
   - 2006–2016: culture *E. coli*, CFU/100 mL, threshold 235.
   - 2015 onward: qPCR *Enterococci*, CCE/100 mL, threshold 1,000.

   The 2015–2016 overlap (629 parent rows with both tests) supports a cross-walk study only.
2. **Resampling and effort.** Sampling is seasonal (late May to early September), with roughly daily qPCR at about 20 beaches. Repeat samples cluster on bad days, so weight by beach-day, not by sample. 2020 is nearly empty (COVID): 47 qPCR tests and 30 predictions. Drop 2020 from temporal splits or report it separately.
3. **The incumbent's unit is ambiguous.**
   - The column description says "Predicted E. coli level in CFU/100 mL", but the model inputs are qPCR CCE.
   - Values range 11.6–743.3 and never reach 1,000.
   - `Master.R` evaluates at 235 (culture era). The 2017+ advisory threshold is not documented in the data.

   Re-score with threshold-free metrics (AUROC, AUPRC), or with a threshold calibrated on training years only. The official model needs **same-day qPCR from 5 input beaches**, so compare it with SafeSwim nowcasts that respect the same information time.
4. **Policy-circular labels.** `advisories_2016` and any "advisory issued" target come from the City's model, so they are not measurements. Use them only as incumbent context.
5. **Coordinates are approximate.**
   - Several culture-era beaches carry a neighbouring beach's coordinates in the source: Fargo = Marion Mahony Griffin (Jarvis), Loyola = Leone, North Shore = Hartigan (Albion), Lane = Osterman. Don't trust joins at < 1 km for these.
   - **Humboldt is an inland lagoon beach in Humboldt Park** (41.906, −87.704), not on Lake Michigan. Exclude it from lake-pixel joins.
6. **Censored and odd values.** See §6 (402 right-censored culture tests; zeros).
7. **Sensor artifacts that are not screened here.** Apply flatline/spike QC (PyHydroQC-style) before use:
   - Ohio Street wave height is a constant 0.187 m for the entire 2017 season (2,714 records).
   - The Foster weather station reports wind speed exactly 3.3 m/s in 5,296–8,626 records per year from 2021, and wind direction 0 in 46,276 records.
   - Sensor positions move (the "Old – Ohio Street Beach" position). There is no water-sensor data in 2022. `battery_life` is telemetry, not water quality.
8. **Revisions.** The portal is live, so rows can be added or edited (the lab table was updated 2026-10-08). The manifest's sha256 values freeze this vintage. Report results against this vintage.

## 8. Joining to sites and upstream predictions

- **Beach identity.** Use `beach_key`, the stable key across renames:
  - lower-case alphanumerics;
  - parenthetical aliases dropped: "Hartigan (Albion)" → `hartigan`, "Margaret T Burroughs (31st)" → `margarettburroughs`, "Marion Mahony Griffin (Jarvis)" → `marionmahonygriffin`;
  - "beach" / "weather station" stripped;
  - "Ohio" and "Old – Ohio Street" → `ohiostreet`;
  - "Montrose Dog" stays a separate beach.

  `beaches.parquet` gives one coordinate per key, plus water-sensor positions.
- **Upstream lake predictions.** Join each Lake Michigan beach's lat/lon to the nearest nearshore lake pixel (Sentinel-2/Landsat chips; record the distance). Point → COMID snapping via NLDI is not meaningful for open-lake shoreline points. For river influence, use distance to the Chicago River / Calumet mouths and upstream NLDI navigation from those mouths.
- **Covariates.** Water sensors join on `beach_key` (6 beaches). Weather joins to the nearest of 3 stations (`sensor_locations`). External NOAA/USGS Lake Michigan stations should be resolved by lat/lon through their own station APIs; station IDs were not verified here.
- **Cross-jurisdiction.** For leave-one-jurisdiction-out tests, Chicago's qPCR/CCE labels are not comparable in unit with WQP/BEACON culture *E. coli* labels. Pair Chicago culture years (2006–2016) with other programmes' culture labels.

## 9. How to rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_chicago_beaches.py            # re-uses raw files if present
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_chicago_beaches.py --refresh  # re-download everything (~30 s)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_chicago_beaches.py
```

`--no-weather` skips the weather-station dataset. The whole build takes about 1 minute and uses 15 MB on disk.
