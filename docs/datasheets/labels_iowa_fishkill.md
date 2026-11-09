# Datasheet — `labels_iowa_fishkill` (Iowa DNR Fish Kill Database)

| Item | Value |
|---|---|
| Use in SENTINEL | D3 **KillAlert** labels (RESEARCH_PLAN §8; `docs/research/07` §d.3) and EW-v2 event context (`docs/research/09` §a) |
| Publisher | Iowa Department of Natural Resources (Environmental Services Division field offices + Fisheries Bureau) |
| URLs (verified 2026-10-22) | API `https://programs.iowadnr.gov/fishkill/api/v1/events?f=strict` (documented REST API v1, released 2023-09-05; spec `…/fishkill/swagger/docs/V1`). CSV export `…/fishkill/Events/Download/Events`. Detail pages `…/fishkill/events/{id}`. ADBNet segments `https://programs.iowadnr.gov/adbnet/api/v1/segments/{id}`. EPA ATTAINS layers `https://gispub.epa.gov/arcgis/rest/services/OW/ATTAINS_Assessment/MapServer/{0,1,2}`. NLDI `https://api.water.usgs.gov/nldi/linked-data/comid/position` |
| Retrieval date | 2026-10-22 (snapshot tag `20261022`) |
| Access | Public, no login. The DNR FAQ says the login is for administrators only and "other users can view all available public information without logging into the database". `robots.txt` returns 404 (no crawl rules). Detail pages were fetched sequentially at 1 request/s |
| License / terms | No explicit license. The API spec's `termsOfService` is `https://www.iowa.gov/policies`. Requested citation: *"Iowa Dept. of Natural Resources, 2013, Fish Kill Database, online at https://programs.iowadnr.gov/fishkill/, accessed on (date of access)."* DNR disclaimer: locations of some kills are estimates; counts are estimates |
| Client / downloader | `sentinel/data/sources/events_iowa_fishkill.py` / `scripts/data/download_events_iowa_fishkill.py` |
| Tests | `tests/test_labels_iowa_fishkill.py`, using real recorded API rows, 3 detail pages, an ADBNet segment, and NLDI and ATTAINS responses in `tests/fixtures/labels_iowa_fishkill/` |
| Manifest | `$SENTINEL_DATA/manifests/labels_iowa_fishkill.json` |

## Label definition

One row per **investigated fish-kill event with release status "Final"**. An event has a calendar date (local), a point of origin, a waterbody, a cause class and a magnitude class.

The API timestamp `T12:00:00` is a placeholder, so only the **date** is meaningful. Use it as "kill observed/reported around day d", never as an onset time. The DNR notes that many kills are reported days late, and that the location is the approximate start of the kill, not its extent.

| Column | Definition |
|---|---|
| `cause_origin` | Published origin: `Natural` 296, `Anthropogenic` 484, `Unknown` 255 |
| `cause` | Published cause, one of 19 classes (DNR Codex "Fish Kill Causes"): Animal Waste 244, Environmental 243, Unknown 153, Ammonia/Fertilizer 69, Unknown (Natural Suspected) 52, Unknown (Anthro Suspected) 50, Disease/Parasite/Stress 46, Pesticides/Herbicides 39, Other Chemical/Pollutant 27, Wastewater 26, Chlorinated Water 17, Organic Material 17, Pollutant – Unknown/Unspecified 17, Gas Bubble Disease 11, Petroleum Products 8, Cyanobacteria Bloom 7, Storm Water 6, Chlorides/Road Salt 2, Mechanical/Electrical 1 |
| `is_natural` / `is_anthropogenic` / `is_unknown_origin` | Derived from `cause_origin` only |
| **`is_env_wq`** | `cause ∈ {Environmental, Cyanobacteria Bloom}` (**250 events**; 172 since 2000). Codex: Environmental = "temperature extremes, summer/winter kill, water stagnation from lack of flow, low dissolved oxygen, and general algal blooms". This is the class a water-quality forecast could plausibly anticipate, and the **recommended primary positive class for KillAlert**. Spills, manure and fertilizer events are better treated as a separate class or excluded (07 §d.3) |
| `is_natural_suspected` | `cause = Unknown (Natural Suspected)` (52). Use for sensitivity analyses |
| `magnitude_id`, `magnitude_desc`, `magnitude_lo/hi` | Magnitude class 0–10 (0 = Unknown/Unspecified, 224 events). The class range is parsed from the description, e.g. "10,001 - 25,000" → 10001/25000, and "1,000,000 or more" → 1e6/inf |
| `est_fish_killed` | Estimated fish killed "if enumerated" (690 events). Null → NaN, never 0. The raw text is kept |
| `kill_length_mi`, `valuation_usd`, `investigation_expenses_usd` | From the detail pages, when present. "Kill Length" appears on 755 pages: 441 give a number in miles and 314 say "not enumerated", which becomes NaN |
| `comments_text` | Investigators' onsite comments (narrative; often contains field DO, ammonia and temperature readings and species lists) |
| `species_counts.parquet` | Rows parsed from **SPECIES / NUMBER KILLED / MONETARY VALUE tables** embedded in the comments. Column alignment honours `colspan`. The "Number"/"Total #" column wins over "FISH KILLED" captions. An unnamed numeric row is the total. Species named only in narrative text are left in `comments_text` |
| `species_table_total`, `species_table_check` | Per event: the table's total row (else the sum of its rows) compared with the API `est_fish_killed`. Values: `match` (within 1 % or 1 fish), `mismatch`, `no_api_count`, `no_table_count`. A mismatch is a source inconsistency, e.g. 3,314 in the table vs 3,134 in the API |

## Coverage (measured 2026-10-22)

- **Events:** 1,035 released events. Event IDs run to 1084, so 49 IDs are not released (draft or withdrawn; the API serves only "Final").
- **Dates:** 1981-02-20 to 2026-10-07; 602 since 2000, 243 since 2010. By decade: 1980s 206, 1990s 227, 2000s 359, 2010s 150, 2020s 93.
- **Seasonality:** all events: Jul 170, Aug 283, Sep 148. `is_env_wq` events are bimodal: winterkill Feb–Mar (60) and summerkill Jul–Aug (106).
- **Waterbody types:** River/Stream 755, Lake 143, Pond 114, Quarry/Pit 14, Unspecified 6, Wetland 3. Env/WQ kills are 98 river, 80 lake, 59 pond. Unlike the WI/MN lake studies, this is mostly a stream dataset.
- **Space:** 98 Iowa counties. **999 events have coordinates** (36 missing). **82 are flagged estimated location** (`est_location`), and some estimates say they "may have no correlation with actual site of kill, other than right county, right river". All points fall inside Iowa ± 0.2°.
- **Detail pages:** 1035 of 1,035 pages parsed; kill length for 441 events, valuation for 336, onsite comments for 1035; 67 events carry a species table (486 species rows, excluding totals); page date ≠ API date for 0 events.
- **305(b) segments:** 895 event–segment links on 594 distinct ADBNet segments (800 events have ≥ 1 segment; 235 have none). 593 of 594 segments match an ATTAINS assessment unit ("IA " + ADB code); 65 of those list 'Cause Unknown – Fish Kills' as a cause of impairment.
- **NLDI:** **COMIDs not computed in this build** (see Join to sites): the 999 geocoded points are ready for the NLDI snap (`--nldi`, about 1,000 requests).

## Files

Processed (`$SENTINEL_DATA/processed/labels_iowa_fishkill/`):

| File | Rows | Content |
|---|---|---|
| `events.parquet` | 1,035 | One row per event: published API fields, validated date/coordinates (`date_flag`, `coord_flag`), derived labels, detail-page fields (`kill_length_mi`, `valuation_usd`, `investigation_expenses_usd`, `fisheries_office`, `location_desc`, `page_created`, `page_last_update`, `comments_text`, `page_date_mismatch`), `species_table_total`, `species_table_check`; with `--nldi` also `comid`, `comid_method`, `comid_context` |
| `event_segments.parquet` | 895 | event × impacted 305(b) assessment segment, with `is_origin_segment` |
| `segments.parquet` | 594 | ADBNet segment metadata (`adb_code`, name, type, size, `huc8` zero-padded string, designated uses) plus the ATTAINS cross-check (`attains_au_id`, `in_attains`, `attains_reporting_cycle`, `attains_ir_category`, `attains_overall_status`, `attains_cause_unknown_fish_kills`) |
| `species_counts.parquet` | 542 | event × species-table row: `species_text`, `number_killed`, `value_usd`, `is_total_row` |

Raw (`$SENTINEL_DATA/raw/iowa_fishkill/`):
- API snapshot `api/events_strict_20261022.json`, CSV export, swagger spec and Codex/doc pages;
- `event_pages_20261022.jsonl.gz` (all detail pages), `adbnet_segments_20261022.jsonl.gz`, `attains_au_20261022.json` (plus `nldi_position_<stamp>.jsonl.gz` when built with `--nldi`);
- `download.log`.

## QC applied

- IDs are strings. HUC8 codes are zero-padded strings. COMIDs are strings.
- **Dates:** only the calendar date is kept. Each date is validated against 1950-01-01 … tomorrow (no failures). The detail page's "Date of Kill" is cross-checked against the API date (`page_date_mismatch`).
- **Coordinates:** checked against the Iowa extent (−96.85…−89.94 E, 40.17…43.71 N), with swapped lat/lon detected; missing → NaN + `coord_flag`.
- **Counts:** null counts and magnitudes stay NaN. The magnitude class 0 "Unknown/Unspecified" is never treated as zero.
- **Cross-checks:** the CSV export and the API agree on row count and IDs (the downloader warns otherwise). Each detail page's own Event ID is checked against the requested ID (`page_id_mismatch`).

## Known biases and label noise

- **Reporting effort.** Kills enter the database only when someone reports them and DNR investigates. That depends on population, stream access, media attention and staffing; the 2010s have fewer events than the 2000s. **Never treat "no record" as "no kill".**
  - Use a case-crossover design: the same reach, other years, same season (07 §d.3).
  - Or use an effort covariate such as road density or population.
- **Onset and location uncertainty.** First report ≠ onset; the date is the date of the kill as recorded. Points are approximate origins, and 82 are explicitly estimated. Snap with a distance tolerance and keep `est_location` as a filter.
- **Cause attribution.**
  - Causes are investigators' judgements; 255 are "Unknown*".
  - Environmental kills (low DO, temperature) are often recorded without measurements; the comments sometimes contain field DO.
  - Anthropogenic spills (manure, ammonia) are not predictable from ambient WQ forecasts. Keep them out of the positive class, or model them separately.
- **Magnitude.** Counts are estimates (AFS Special Publication 30 / Iowa Administrative Code 113 procedures); natural and late-reported kills are least accurate.
- **Circularity with ATTAINS.** Iowa uses fish kills as evidence in 305(b) assessments (ATTAINS cause "Cause Unknown – Fish Kills"). **Never use ATTAINS impairment causes or IR categories as features for KillAlert.** They are kept in `segments.parquet` only so the circularity can be audited.

## Join to sites

| Key | How |
|---|---|
| lat/lon | `latitude`/`longitude` (WGS84 per the DNR page), for 999 events |
| **COMID** (NHDPlus V2) | **Not computed in this build.** `--nldi` adds `comid` from NLDI `comid/position` (point-in-catchment; `comid_method`, `comid_context` = `stream_reach` / `catchment_of_standing_water`). On 2026-10-22 the `api.water.usgs.gov` gateway limited this host to 3,600 requests/h (`x-ratelimit-limit: 3600`; HTTP 429 with `Retry-After: 258`). The quota was in use by the site-linking workstream (`scripts/data/download_nldi.py`), so the ~1,000 snaps were deferred to that join step (`sentinel/data/sources/nldi.py`). For River/Stream events the snap gives the kill reach. **For Lake/Pond events the catchment COMID is not the waterbody**: use NHD waterbody polygons or LAGOS-US lake ids |
| 305(b) segment / HUC8 | `event_segments` → `segments.adb_code` and `segments.huc8` |
| **ATTAINS AU** | `attains_au_id = "IA " + adb_code`. Verified 2026-10-22 on the ATTAINS geospatial layers (e.g. ADB `01-YEL-447` = AU `IA 01-YEL-447`, Hecker Creek, cycle 2024). `in_attains` gives the match rate |
| Sensors | From the COMID, navigate downstream/upstream with NLDI (`…/comid/{id}/navigation/DM/nwissite?distance=50`) to find USGS DO/temperature sensors (09 §a.3). Use the reach × week as the unit for KillAlert |

## Rebuild

```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -I scripts/data/download_events_iowa_fishkill.py          # as built 2026-10-22
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -I scripts/data/download_events_iowa_fishkill.py --nldi   # + COMIDs (needs NLDI quota)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_iowa_fishkill.py
```

A first run takes about 25 minutes: 1,035 detail pages at 1 request/s, 594 segments and 18 ATTAINS queries. `--nldi` adds about 1,000 rate-limited NLDI lookups. Re-runs reuse the cached snapshot bundles for the same `--stamp`. Use `--refresh` to re-download the API snapshot, and `--no-pages`, `--no-segments` or `--no-attains` to skip steps.
