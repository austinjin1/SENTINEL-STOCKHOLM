# Datasheet — `uk_ea_ecology` (Environment Agency, England: invertebrates + diatoms with paired water chemistry)

| | |
|---|---|
| **Location** | `$SENTINEL_DATA/processed/uk_ea_ecology/` (Parquet, zstd) · manifest `manifests/uk_ea_ecology.json` · raw `raw/uk_ea/ecology/` (manifest `uk_ea_ecology_raw`), `raw/uk_ea/wq/` (manifests `uk_ea_wq_points_raw`, `uk_ea_wq_observations_raw`) |
| **Sources** | Ecology & Fish Data Explorer bulk downloads <https://environment.data.gov.uk/ecology/explorer/downloads/> (`INV_OPEN_DATA.zip` 62,294,157 B, `DIAT_OPEN_DATA.zip` 8,706,349 B, `OPEN_DATA_TAXON_INFO.zip` 825,711 B; all Last-Modified 2026-10-20 08:45 GMT, ETags in the manifest). Water Quality Archive (WQA) API v1.3.1 <https://environment.data.gov.uk/water-quality/> (OpenAPI spec saved as `raw/uk_ea/wq/wqa_openapi_2026-10-22.json`) |
| **License** | Open Government Licence v3.0 — "Contains Environment Agency information © Environment Agency and/or database right" |
| **Retrieved** | 2026-10-22. The INV and DIAT zips were copied from the 2026-10-22 research cache after their size matched the official `Content-Length` (Last-Modified predates the cache); zip CRCs verified |
| **Builder** | `sentinel/data/sources/uk_ea.py` · `sentinel/data/bio_uk_ea.py` · `scripts/data/download_uk_ea.py` (stages `ecology`, `points`, `observations`) · `scripts/data/build_uk_ea.py` |
| **Synthetic data** | none |

## Coverage

| Assemblage | Analyses | Samples | Sites | Taxa (keys) | Taxon records | Dates | Since 2000 |
|---|---|---|---|---|---|---|---|
| Freshwater macroinvertebrates | 280,992 | 280,992 | 30,735 | 3,069 | 7,581,200 | 1965-06-08 … 2026-09-29 | 135,399 samples |
| Diatoms | 28,148 (19,227 light microscopy `LABP`, 8,921 DNA metabarcoding `DNAP`) | 27,391 | 7,238 | 1,669 | 1,199,202 | 1993-06-14 … 2025-10-16 | 27,664 analyses |

Invertebrate samples per decade: 1960s 19, 1970s 1,335, 1980s 22,889, 1990s 121,350, 2000s 68,353, 2010s 47,197, 2020s 19,849. Sample reasons: National Monitoring 133,027, local monitoring 68,687, historic unknown 46,515, local investigative 24,739, RSN 4,380. 1,231 invertebrate analyses are third-party data (`is_third_party_data`). Site types: 29,245 rivers (WBRV), 427 canals, 338/325/221 ditches/drains, 138 springs, 41 estuarine (excluded from pairing). 80% of diatom site IDs are also invertebrate sites.

## Tables

| File | Grain | Notes |
|---|---|---|
| `sites.parquet` | (site, assemblage) | `site_id` (string), `assemblage`, `waterbody_type`, `water_body`, `wfd_waterbody_id` (64% of invertebrate, 99% of diatom sites), `easting_bng`/`northing_bng` (EPSG:27700), `lon`/`lat` (WGS84 via pyproj), RIVPACS-style site descriptors with units in the column names (`site_altitude_m`, `site_slope_m_per_km`, `site_dist_from_source_km`, `site_width_m`, `site_depth_cm`, substrate %, `site_alkalinity_mg_per_l_caco3`, …), sample-date range, `ecn_site` |
| `samples_invertebrate.parquet`, `samples_diatom.parquet` | one row per analysis | `sample_id`, `analysis_id`, `sample_date` (local date), sample type/method/reason, `analysis_type` (LABP / DNAP), `analysis_method` (+ description), `replicate_code`, `is_third_party_data`; published indices as `idx_*` (BMWP, WHPT, WHPT-NW, LIFE family/species, PSI, EPSI, DEHLI, CCI; TDI4, TDI5LM, TDI5NGS and their % guilds) — **these are functions of the same analysis' taxa; never use them as targets for models fed those taxa** |
| `counts_invertebrate.parquet` (7,581,200) | analysis × taxon | `taxon_list_item_key` (NBN key), `abundance` (`TOTAL_ABUNDANCE` as published), `abundance_category` (A–E), `log_abundance_class` (1–5), `analysis_method` |
| `counts_diatom.parquet` (1,199,202) | analysis × taxon | `abundance` = valves counted (LM) or sequence reads (DNA), `analysis_type` |
| `taxon_dictionary.parquet` (28,556) | taxon key | names, preferred name, rank, parent key, taxon type; harmonised lineage in `processed/taxonomy_harmonised` |
| `wq_points.parquet` (23,264) | WQA sampling point | all freshwater types (F6 rivers 18,449, FZ 2,281, FA lakes 1,693, FJ canals 385, …), open (15,192) and closed (8,072) |
| `site_wq_pairs.parquet` | (site, assemblage) | paired WQ point, `pair_distance_m`, second-nearest point and distance, point type/status/label, `pair_rule` |
| `wq_observations.parquet` | WQA observation | `wq_point`, `sample_notation`, `observation_notation`, `sample_time_local`, `time_utc`, `date`, `tz_source`, `sampling_purpose`, `sample_material`, `determinand` (4-digit code), `determinand_label`, `short_name`, `result_raw`, `value`, `censored`, `censor_op`, `detection_limit`, `qualifier`, `unit` |
| `chemistry_matched_long.parquet` | (sample, determinand, unit, window) | `n_obs`, `median`, `mean`, `min`, `max`, `frac_censored`, `nearest_value`, `nearest_lag_days`, `assemblage`, `wq_point`, `pair_distance_m` |
| `chemistry_matched_wide_pm30d_{invertebrate,diatom}.parquet` | one row per sample | `<short>__<unit>__pm30d__median` and `__n` (unit is part of the column name) |

## Join rule (chemistry ↔ biology), fixed before any modelling

1. **Space.** Each ecology site (invertebrate and diatom; all waterbody types except estuarine `WBES`) is paired with the **nearest WQA point of type F6 (rivers), FJ (canals), FK (land drains) or FZ (unspecified freshwater), any status, within 1,000 m** straight-line distance in British National Grid. Lake (FA), bathing (FL), augmentation/transfer/comparative-inlet (FD/FB/FC) points are not used. The WQA does not publish WFD water-body IDs for its points, so "same water body" cannot be enforced — use `pair_distance_m` to tighten (e.g. ≤ 250 m).
   - Invertebrates: 23,263 of 30,694 eligible sites paired (13,112 distinct WQ points); distance quantiles 10/25/50/75/90% = 0 / 27 / 109 / 413 / 724 m; 3,289 at 0 m (co-located), 11,255 ≤ 100 m, 15,017 ≤ 250 m. 120,313 of the 135,399 post-2000 samples are at paired sites.
   - Diatoms: 6,565 of 7,238 sites paired (5,718 points); 0 / 0 / 41 / 151 / 488 m; 1,714 at 0 m, 4,464 ≤ 100 m, 5,349 ≤ 250 m.
2. **Time.** For each sample (date *d*, 2000 onwards — the WQA starts in 2000) and each core determinand, the paired point's observations are summarised in two windows: **`pm30d` = [d − 30, d + 30] days** and **`prior365d` = [d − 365, d]**: n, median, mean, min, max, fraction censored, value nearest in time and its lag (days, observation − sample).
3. **What is summarised.** Only routine samples of the water itself: sample material "RIVER / RUNNING SURFACE WATER", "CANAL WATER" or "ANY WATER", excluding "UNPLANNED REACTIVE MONITORING" (pollution-incident) purposes. Effluent, precipitation, estuarine and incident samples recorded at the same points stay in `wq_observations` but never enter the windows. Censored results enter summaries at their reported limit and `frac_censored` says how often; units are never mixed (one summary per determinand × unit).
4. **Determinands** (EA codes): pH 0061; conductivity at 25 °C 0077 and at 20 °C 0062 (kept separate); water temperature 0076; ammoniacal N 0111; total oxidised N 0116; nitrate-N 0117; nitrite-N 0118; reactive orthophosphate-P 0180; total P 0348; DO mg/L 9924 and % saturation 9901; BOD5 ATU 0085; suspended solids 0135; chloride 0172; alkalinity to pH 4.5 0162; sulphate 0183; turbidity 0068 (FTU) and 6396 (NTU) kept separate; total hardness 0158.

### Chemistry coverage (build 2026-10-22)

- `wq_observations`: **7,651,377** observations at 10,100 WQ points, 2000-01-01 … 2026-10-15; 7,601,972 eligible for window summaries (routine river/canal water). Fraction censored ("<x"): ammoniacal N 0.35, BOD 0.25, suspended solids 0.18, nitrite 0.14, orthophosphate 0.11, sulphate 0.11, total P 0.06, nitrate 0.02; pH/temperature/DO/conductivity ~0.
- Invertebrates: 135,399 samples since 2000; 120,313 at paired sites; **77,112 with any ±30-day chemistry** (pH 75,154; ammoniacal N 74,744; orthophosphate 73,323; nitrate 67,315; conductivity@25 °C 43,018; chloride 37,249; total P only 10,820) and 84,512 with any prior-365-day chemistry.
- Diatoms: 26,907 samples since 2000; 25,031 at paired sites; **16,554 with ±30-day chemistry** (orthophosphate 15,705; total P 4,658), 18,217 prior-365-day.
- Download: 3,381 + 265 request jobs (25 points × 1 year), 0 final failures; 81 diatom jobs first failed with transient HTTP 403 and succeeded on a re-run with 2 workers.

## WQA access route (verified 2026-10-22) and its quirks

The old WQA API was replaced by the "Water Quality Explorer" API (OpenAPI 3.1, v1.3.1). Used here: `GET /water-quality/sampling-point` (JSON-LD, ≤ 250 per page, BNG by default; `Accept: application/ld+json` is required — without an Accept header the request falls through to the web app and returns HTML 404) and `POST /water-quality/data/observation` with **query-string filters and no body** (a JSON body `{}` is rejected with "Invalid query parameters"), `Accept: text/csv`, ≤ 100 points and ≤ 1 calendar year per multi-point call, ≤ 2,500 rows per page. Paging (`skip > 0`) of multi-point calls returned HTTP 502 above ~40 points (observed), so requests use 25 points × 1 year with a recursive halving fallback. Empty results are HTTP 204. The Explorer's bulk "area × year" CSV download (`/water-quality/api/proxy/data/area/download`, form fields `body`, `format`, `_type=file-download`) works but streams all determinands at ~150 KB/s and was not used. Observation times are published without a UTC offset: stored as `sample_time_local` + `date`, converted to `time_utc` assuming Europe/London (`tz_source`). Censored results are strings such as `<0.03` (→ `censored`, `detection_limit`); coded results (e.g. "No flow/discharge at sampling point") become `value` NaN with the text in `qualifier`.

Request plan: WQ points paired with invertebrate sites × the calendar years covered by their samples' windows (84,210 point-years, 11,094 points), plus the extra point-years needed for diatom samples (6,269 point-years, 2,366 additional points); ≤ 4 concurrent requests, retries with back-off.

## Known issues and traps

- **Abundance is not comparable across EA analysis methods.** `ANAA` = actual counts (33,803 analyses); `ANLA` (log abundance, BT001 scale; 160,630) and `ANLE` (local abundance estimate; 85,532) mostly store class representative values (3, 33, 333, 3,333 …). `log_abundance_class` (A–E → 1–5) is the cross-method measure; `ANPA` analyses are presence-only. Method use shifts over time (ANLE dominates before 2000, ANLA after).
- **Mixed taxonomic resolution:** EA invertebrate records are ~50% family-level, ~29% species, ~10% genus, ~6% class; use `processed/taxonomy_harmonised` (`analysis_taxon` = family).
- **Non-invertebrates in invertebrate samples:** the EA taxa table also lists macrophytes, algae, fish and other vertebrates observed in the net (`taxonomy_harmonised.group`).
- **Index circularity:** BMWP/WHPT/LIFE/PSI/TDI are computed from the same taxa lists.
- **Spatial pairing ≠ same reach:** a paired point can sit on a tributary or across a confluence; there is no flow-network check. Distances are stored; tighten them for strict analyses.
- **Chemistry starts in 2000;** earlier samples have no matched chemistry. Sampling frequency varies by point (monthly at most GQA/WFD sites), so the ±30-day window is often empty for small or closed points.
- Site-level `ALKALINITY`, `CONDUCTIVITY`, `TOTAL_HARDNESS`, `CALCIUM` in the site file are long-term site descriptors (RIVPACS inputs), not sample-date chemistry.

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
PY=/home/bcheng/miniconda3/envs/sentinel2/bin/python
$PY -B scripts/data/download_uk_ea.py ecology            # bulk zips (+ --reuse-dir DIR to reuse verified copies)
$PY -B scripts/data/download_uk_ea.py points             # WQA freshwater sampling points + determinand codes
$PY -B scripts/data/download_uk_ea.py observations --workers 4                       # ~2 h, resume-safe
$PY -B scripts/data/download_uk_ea.py observations --workers 4 --assemblage diatom   # extra point-years
$PY -B scripts/data/build_uk_ea.py
```
