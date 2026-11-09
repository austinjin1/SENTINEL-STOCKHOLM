# Datasheet — `labels_wa_toxic_algae` (Washington State Toxic Algae database)

Lab cyanotoxin results (microcystin, anatoxin-a, cylindrospermopsin, saxitoxin) for Washington
freshwater sites, 2007-05-22 → 2026-10-14, as primary measured labels for **D2 ToxinWatch** and
H5c-style sample-based event context (EW-v2).

## Source and publisher
- **Publisher:** WA Department of Ecology, Freshwater Algae Control Program (state-funded toxin
  testing for lake managers, local jurisdictions and residents). Ecology's program page
  (`https://ecology.wa.gov/water-shorelines/water-quality/freshwater-algae/algae-testing-and-control`)
  names this website as the program's results database.
- **Host:** King County (website and database, `https://www.nwtoxicalgae.org/`).
- **Retrieved:** 2026-10-22 09:24 UTC (vintage `2026-10-22`).

## Verified URLs and access method
| What | URL | Method |
|---|---|---|
| Toxin results (whole DB) | `https://www.nwtoxicalgae.org/Data.aspx` | GET page → POST the ASP.NET form with **all search fields blank** and the "Export Toxin Data" button → attachment `toxin2026_10_22.csv` (Windows-1252, 2,804,783 bytes, 31,055 rows). One request; the site says a blank search returns "all results in the Toxic Algae Database". |
| Site coordinates | `https://www.nwtoxicalgae.org/LocationExtractor.aspx` | POST "Get EIM Locations" → `EIMLocation.csv` (525 sites). Identical lat/lon (max difference 0) to the map layer embedded in Data.aspx (`c_SiteInfoHiddenField`, 507 sites). |
| Published counts | `YearSummary.aspx`, `StateSummary.aspx` | GET (cross-check). |
| Guideline values | `About.aspx` | GET; "Microcystin: >= 8 µg/L, Anatoxin-a: >= 1 µg/L, Cylindrospermopsin: >= 15 µg/L, Saxitoxin: >= 75 µg/L" (WA DOH provisional recreational guidance). Parsed on every build and compared with the constants in code. |
| Verification only | `Data.aspx?SiteID=<id>` | 4 per-site pages (sites 202, 15, 45, 603: 712/260/124/14 rows) — identical row multisets to the export. |

- `robots.txt`: HTTP 404 (no crawl rules). No login, key or CAPTCHA. 10 sequential requests per
  build, ≥ 1.5 s apart, generic User-Agent; no personal data sent.
- Not used: `Scum.aspx?LabSampleNum=` ("Additional Sample Information": counts of boats/swimmers/
  dogs, scum colour, weather; one page per sample, 7,131 samples have one); "Export Phyto Data"
  (phytoplankton IDs). WA Ecology **EIM** (`apps.ecology.wa.gov/eim/search/`) redirects downloads to
  `DownloadLogin.aspx` (login) → not used.

## License / terms
**Restricted.** Every page binds visitors to the King County website terms of use
(`https://www.kingcounty.gov/About/termsOfUse.aspx` → kingcounty.gov/en/shared-topics/about-king-county/website/terms-of-use):
site content "including all text, graphics, compilations, data … is the property of King County"
and may not be published, displayed or distributed "without the prior written permission of King
County". Scraping/automated access is not prohibited. **Use for internal research; do not publish
or upload raw or row-level derived tables (e.g. to SENTINEL-HF) without written permission;
aggregate metrics and figures citing the source are the intended use.** No open licence is stated.

## Files
Raw — `$SENTINEL_DATA/raw/wa_toxic_algae/2026-10-22/` (3.0 MB): `toxin_export.csv` (as received),
`eim_locations.csv`, `data_page.html.gz`, `year_summary.html.gz`, `state_summary.html.gz`,
`about.html.gz`, `site_pages/site_{15,45,202,603}.html.gz`, `request_log.json`.

Processed — `$SENTINEL_DATA/processed/labels_wa_toxic_algae/` (780 KB, Parquet zstd):

| File | Rows | Grain |
|---|---|---|
| `wa_toxin_results.parquet` | 31,055 | lab sample × toxin (16,973 lab samples, 15,962 site-days) |
| `wa_toxin_siteday.parquet` | 29,311 | site × date × toxin (replicates and sampling points pooled) |
| `wa_sites.parquet` | 525 | site (`site_id` = program SiteID) |
| `wa_year_summary_crosscheck.parquet` | 74 | year × toxin, export vs published YearSummary |

Manifest: `$SENTINEL_DATA/manifests/labels_wa_toxic_algae.json` (sha256 of all files, request
parameters, retrieval time, coverage, cross-check).

## Coverage
| Toxin | Results | Sites | Years | Censored (<MDL etc.) | ≥ WA guideline (computed) | Published StateSummary (samples / exceedings) |
|---|---|---|---|---|---|---|
| Microcystin | 16,566 | 495 | 2007–2026 | 11,089 | 1,603 (≥ 8 µg/L); 1,597 > 8 µg/L (EPA) | 16,575 / 1,595 |
| Anatoxin-a | 12,696 | 466 | 2009–2026 | 10,612 | 419 (≥ 1 µg/L) | 12,702 / 418 |
| Cylindrospermopsin | 846 | 290 | 2009–2026 | 807 | 0 (≥ 15 µg/L) | 846 / 0 |
| Saxitoxin | 947 | 297 | 2009–2026 | 753 | 1 (≥ 75 µg/L) | 947 / 1 |

- 503 sites with results (525 listed); 41 counties; 62 WRIAs named. King County = 59.6% of rows.
- Season: June–October = 89.5% of rows (winter sampling exists, e.g. American Lake, Anderson Lake).
- Samples per site: median 8, 90th percentile 152, max 712; 213 sites have ≤ 5 results. The six
  most-sampled sites (~700 results each) are King County swim beaches (Gene Coulon, Matthews
  Beach, Green Lake West, Seward Park, Lake Sammamish, Magnuson).
- Site-day labels: microcystin 15,633 site-days (1,510 ≥ 8 µg/L; 1,504 > 8 µg/L); anatoxin-a 12,008
  (371 ≥ 1 µg/L).
- Coordinates: 498 of 525 sites (482 of 503 result sites; 73 result rows lack site coordinates).

Per year (results / exceedances; "published" = YearSummary "Number of Exceedings"):

| Year | MC results (export / YearSummary) | MC ≥ 8 µg/L (computed) | MC exceedings (published) | ANA results (export / YearSummary) | ANA ≥ 1 µg/L (computed) | ANA exceedings (published) |
|---|---|---|---|---|---|---|
| 2007 | 71 / 75 | 18 | 17 | – / – | – | – |
| 2008 | 285 / 285 | 80 | 80 | – / – | – | – |
| 2009 | 1042 / 1042 | 94 | 94 | 458 / 458 | 17 | 16 |
| 2010 | 1014 / 1014 | 74 | 74 | 706 / 706 | 15 | 15 |
| 2011 | 1121 / 1121 | 81 | 81 | 766 / 766 | 32 | 32 |
| 2012 | 891 / 891 | 65 | 65 | 580 / 580 | 40 | 40 |
| 2013 | 811 / 811 | 83 | 83 | 465 / 465 | 25 | 25 |
| 2014 | 822 / 822 | 94 | 94 | 480 / 480 | 15 | 15 |
| 2015 | 994 / 995 | 147 | 147 | 492 / 493 | 27 | 27 |
| 2016 | 774 / 776 | 44 | 43 | 434 / 436 | 19 | 19 |
| 2017 | 786 / 788 | 31 | 31 | 756 / 759 | 39 | 39 |
| 2018 | 753 / 753 | 60 | 60 | 713 / 713 | 22 | 22 |
| 2019 | 845 / 845 | 100 | 100 | 819 / 819 | 24 | 24 |
| 2020 | 919 / 919 | 104 | 104 | 820 / 820 | 23 | 23 |
| 2021 | 933 / 933 | 98 | 98 | 945 / 945 | 24 | 24 |
| 2022 | 1018 / 1018 | 67 | 66 | 927 / 927 | 19 | 19 |
| 2023 | 879 / 879 | 103 | 101 | 837 / 837 | 19 | 19 |
| 2024 | 904 / 904 | 105 | 102 | 848 / 848 | 18 | 18 |
| 2025 | 907 / 907 | 88 | 88 | 876 / 876 | 25 | 25 |
| 2026 | 797 / 797 | 67 | 67 | 774 / 774 | 16 | 16 |

## Label definitions
- **`exceed_wa_guideline`** (primary, all toxins): result ≥ WA DOH provisional recreational
  guidance — microcystin 8, anatoxin-a 1, cylindrospermopsin 15, saxitoxin 75 µg/L. Censoring is
  resolved only when the bound decides: `<L` with L ≤ guideline → False; `>L` with L ≥ guideline →
  True; otherwise (`<RDL` with unknown limit, blank MDL, missing result) → `<NA>`.
- **`exceed_epa_8ugL`** (microcystin only; `<NA>` for other toxins): result > 8 µg/L (EPA 2019
  recreational swimming advisory; strict inequality). Same censoring rule.
- **`exceed_published`**: the site's `ExceedInd` ("Above State Guideline") as published. It follows
  the same ≥ rule and the *current* guideline for all years (1 disagreement in 16,566 microcystin
  rows), i.e. history was re-flagged under today's values. It disagrees with the computed label on
  4 rows: three right-censored `>100` microcystin results (2007) flagged "No", and one 73 µg/L
  microcystin result (2016-10-03, SiteID 39) flagged "No". Use the computed label.
- Site-day table: `*_any` = three-valued OR over the day's results (True if any True; `<NA>` if none
  True but some undetermined; else False); `max_value` over quantified results only.
- Values are never imputed: censored results keep `value = NaN`, `censor_type` ('<' or '>'),
  `censor_limit` (MDL for `<MDL`, the stated number for `<0.05`/`>100`, NaN for `<RDL`) and
  `detection_limit` (the MDL column).

## Schema (`wa_toxin_results.parquet`)
`source`, `record_id` (`site:date:lab_sample:toxin`, unique), `site_id` (string), `site_name`,
`county`, `wria` (string), `wria_name`, `date` (local sample date; no time-of-day is published),
`year`, `date_flag`, `toxin_raw`, `toxin` (microcystin | anatoxin-a | cylindrospermopsin |
saxitoxin), `value` (µg/L), `unit` (`ug/L`, from the export header "(µg/L)"), `value_raw`,
`censored`, `censor_type`, `censor_limit`, `detection_limit`, `result_missing`, `qualifier`,
`exceed_published`, `guideline_wa_ugL`, `exceed_wa_guideline`, `exceed_epa_8ugL`, `scum_info`
(`ScumInd`: an "Additional Sample Information" field record — scum colour, users, weather — exists
for the sample; **not** a validated "sample taken from scum" flag), `location_text` (free-text
sampling point), `ecology_tracking_number` (`ClientLocator`; absent 2007–2008, present on 15–51% of
rows per year since 2009), `lab_sample_num`,
`lab_sample_num_norm` (upper-case, whitespace removed).

`wa_sites.parquet`: `site_id`, `eim_location_id` (truncated code; **not unique** — 32 duplicates),
`county`, `latitude`, `longitude` (decimal degrees; EIM datum code 1, accuracy code 99 =
unspecified), `coord_flag`, `coord_source`, `site_name`, `map_status_snapshot` (Within / Exceeded /
Unknown as of retrieval), `map_last_sample_date`, `in_map_layer`, `name_has_river_or_creek`
(heuristic, 70 sites), `n_results`, `n_lab_samples`, `first_date`, `last_date`, `toxins_tested`,
`wria`, `n_exceed_wa`, `in_results`.

## QC applied
- Header checked against the expected 13 columns (build fails on a schema change). The export is
  **not quoted**: 244 rows have commas inside `Location` and 4 contain literal `"`; rows are split
  with an anchored regex (strict WRIA/date/toxin/flag fields, comma-free tail). 31,055 of 31,055
  lines parsed; unparsed lines would be written to `unparsed_lines.json` (build fails if > 0.1%).
- Toxin names normalised (73 lower-case rows in 2013–2015). Dates parsed `%m/%d/%Y` and range-checked
  (all valid). IDs kept as strings. Lab numbers stripped (trailing blanks, lower-case `l`).
- Non-numeric results: `<MDL` 23,254; `NA`/`N/A`/blank 27 (→ `result_missing`); `<0.05` 3; `>100` 3
  (right-censored); `<RDL` 1. Blank MDL on 8 rows.
- Coordinates validated against a WA bounding box (all inside); 27 sites without coordinates are
  flagged `missing`.
- Cross-checks: (i) **YearSummary**: result counts match in 67 of 74 year × toxin cells; the export
  has 15 fewer results (2007 MC −4; 2015 −2; 2016 −4; 2017 −5; whole-DB 31,055 vs 31,070 published).
  The published "Number of Exceedings" equals "`ExceedInd` = Yes AND value strictly > guideline" in
  73 of 74 cells (exception: 2007 microcystin, 15 vs 17, the year with 4 missing results); the
  differences from the ≥ rule are ties at exactly 8.000 / 1.000 µg/L. (ii) **Per-site pages**: 4
  sites, identical row multisets. (iii) **StateSummary** totals above.

## Known biases and caveats
- **Reactive, effort-driven sampling.** Samples are submitted after a bloom is reported or by lake
  programs that choose to monitor; Ecology states it does not have results for all waterbodies.
  "No sample" ≠ "no toxin"; site-years without samples are not negatives. Exceedance rates are
  far higher than in probability surveys (use NLA for unbiased rates).
- **Two sampling regimes mixed:** routine King County swim-beach monitoring (e.g. Gene Coulon:
  weekly, median gap 7 days, May–Oct; 22 King County sites with ≥ 300 results hold 47% of all rows;
  King County overall 59.6%) vs. complaint/bloom-triggered samples elsewhere. Report performance
  separately or weight by site/lake; a pooled model can be dominated by a few King County lakes.
- **Scum samples.** Rows with `scum_info` (23%) exceed the guideline 3.7× as often (14.9% vs 4.0%):
  samples often target visible scum, so concentrations are not open-water averages.
- **Analytical method / lab not reported** in the export (MDL varies by toxin and era: microcystin
  MDL 0.01–0.8 µg/L, median 0.15). Method changes over time (ELISA vs LC-MS/MS) cannot be
  identified from the published fields — treat as total microcystins as reported.
- Location points are site-level (beach/access point or lake point) with unspecified accuracy; 39
  rows (9 sites) carry sample-point coordinates inside `location_text` (not parsed). Several
  site_ids belong to the same lake (e.g. multiple Lake Washington and Moses Lake beaches).
- Guideline history: the site applies today's WA DOH values to all years. Earlier WA guidance
  values (if different) are not documented on the site and were not verified this session; any
  "advisory at the time" analysis needs the historical DOH values.
- 2007–2008 contain microcystin only; anatoxin-a, cylindrospermopsin, saxitoxin start 2009.
- The database is live (rows up to 2026-10-14): counts change between vintages; freeze a vintage.

## Join to sites / upstream predictions
- Key: `site_id` (program SiteID, string). Results → `wa_sites` for coordinates, county, WRIA.
- **Lakes (most sites):** point-in-polygon of the site point against NHD/NHDPlus HR waterbody polygons
  (or LAGOS-US / NLA lake IDs) to get the waterbody COMID/permanent ID; then group site_ids per
  waterbody. Use satellite lake pixels (M2) of that polygon, and lake-level upstream predictions.
- **River sites** (`name_has_river_or_creek`, 70 sites, e.g. Columbia/Yakima River parks): snap via
  NLDI `https://api.water.usgs.gov/nldi/linked-data/comid/position?coords=POINT(lon lat)`; record the
  snap distance (09 §a.3 rule: drop > 500 m unless named waterbody).
- Fallback for the 27 sites without coordinates: lake name + county (+ WRIA) geocoding to GNIS/NHD names.
- HUC12: WBD point query on the site point. Spatial blocks: WRIA or HUC8; keep all site_ids of one
  lake in the same fold.
- Decision-time rule: toxin results take days; never use same-day results as inputs.

## How to rebuild
```bash
# download (10 requests, ~30 s) + build; writes raw/<today>/, processed/, manifest
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_events_wa_toxic_algae.py
# offline rebuild from the latest raw vintage (byte-identical Parquet verified 2026-10-22)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_events_wa_toxic_algae.py --from-raw
# tests (fixtures only, no network)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_wa_toxic_algae.py
```
Client: `sentinel/data/sources/events_wa_toxic_algae.py`. Freeze label vintages (raw sub-directory
per retrieval date) — the database is live and history can be re-flagged or edited.
