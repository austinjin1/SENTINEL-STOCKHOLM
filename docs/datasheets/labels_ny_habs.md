# Datasheet — `labels_ny_habs` (NY DEC harmful algal blooms: waterbody-years 2012–2025 + NYHABS 2026 reports)

**Role in SENTINEL v2:** secondary, positive-only HAB labels for D2 ToxinWatch and EW-v2 context
(H5). Each row says "DEC documented a cyanobacteria HAB at this waterbody in this year (or on this
date)". Absence of a row is not evidence of no bloom.

## Source and access

| Item | Value |
|---|---|
| Publisher | NYS Department of Environmental Conservation (DEC), Division of Water, HABs Program |
| 2012–2018 summary | "Harmful Algal Bloom Statewide Occurrence Summary: 2012-2018", data.ny.gov `qtg2-hjth` — https://data.ny.gov/d/qtg2-hjth (rows last updated 2019-05-31) |
| 2019+ summary | "Harmful Algal Blooms by Waterbody Summary: Beginning 2019", data.ny.gov `95my-wijm` — https://data.ny.gov/d/95my-wijm (rows last updated 2026-01-14; posted annually, so 2026 is not yet included) |
| API used | SODA 2.x CSV `https://data.ny.gov/resource/<id>.csv?$select=*,:id&$order=:id` (paged 50,000) + `/api/views/<id>.json` metadata |
| NYHABS per-report views | DEC-owned public ArcGIS Online feature views (org `DZHaqZm9cxOD4CWM`): `Archived_HAB_Reports` (item 41a2a043…), `Archived_HAB_Reports_DIL` (19b4069e…), `Current_HAB_Reports_7_3_19` (302fac4b…), `Current_HAB_Reports_DIL` (9492e4e6…), layer 0 `HAB_Master_table`; map app https://nysdec.maps.arcgis.com/apps/webappviewer/index.html?id=ae91142c812a4ab997ba739ed9723e6e |
| Retrieval date (vintage) | 2026-10-22 |
| Access | Anonymous, no key. data.ny.gov answered normally on 2026-10-22 (the research session saw TLS resets). |
| Licence / terms | data.ny.gov metadata carry no licence field; the portal's OPEN-NY Terms of Use (2013) allow free reuse for lawful purposes with no attribution, share-alike or pre-approval requirement. The ArcGIS items are public but state no licence (`licenseInfo` empty). |

### What is and is not obtainable (documented 2026-10-22)

- **Obtained:** both waterbody-year summaries (880 + 1,371 rows) and every public NYHABS report of the
  2026 season (1,406 archived + 45 current = 1,451 confirmed reports).
- **Not public:** per-report NYHABS records of earlier seasons. The views are filtered
  (`viewDefinitionQuery`) to `HAB_STATUS IN ('Confirmed','Confirmed_with_High_Toxins')` — suspicious or
  non-HAB reports are never exposed — and "archived" means "reported within the current season"
  (DEC notifications page); the master layer is not shared. Status changes from the last 27 h
  (`STATUS_DATE NOT BETWEEN now − 1.125 d AND now`) are hidden. Historical per-report data would need
  a request to DEC (`HABsInfo@dec.ny.gov`, the dataset contact).
- **Not parsed:** DEC's annual "HABs Archive Summary" PDFs (2020–2025, e.g.
  https://dec.ny.gov/sites/default/files/2025-12/2025nyhabssummary.pdf) list the same waterbody,
  county, first/last report and number of reports as `95my-wijm`.
- **Out of scope by DEC definition:** Lake Champlain blooms (reported by Vermont), marine HABs
  (red/brown tides), other algae, beach closures and drinking-water notices.

## Files

- Raw (kept, 1.9 MB): `$SENTINEL_DATA/raw/ny_habs/2026-10-22/` — `qtg2-hjth.csv`, `95my-wijm.csv`
  (with Socrata `:id`), `*_metadata.json`, `nyhabs_{archived,archived_dil,current,current_dil}.json`
  (Esri JSON features with point geometry) and `*_layer.json` (layer definitions incl. view filters),
  `fetch_info.json`.
- Processed (zstd Parquet): `$SENTINEL_DATA/processed/labels_ny_habs/`
  - `waterbody_years.parquet` — 2,251 rows: published grain, both eras.
  - `lake_years.parquet` — 2,148 rows: one row per lake-year (recommended label table).
  - `nyhabs_reports.parquet` — 1,451 rows: per-report confirmed blooms, accumulated over all raw
    vintages (deduplicated by `globalid`; values from the latest snapshot; `first_seen`/`last_seen`).
- Manifest: `$SENTINEL_DATA/manifests/labels_ny_habs.json`.

## Coverage (vintage 2026-10-22)

| Table | Rows | Years | Space |
|---|---|---|---|
| `waterbody_years` | 2,251 (880 + 1,371) | 2012–2025 | 587 distinct waterbody names; all rows have a lake-centroid lat/lon inside NY |
| `lake_years` | 2,148 (777 + 1,371) | 2012–2025 | as above |
| `nyhabs_reports` | 1,451 | 2026-04-23 … 2026-10-20 | 249 waterbodies, 54 counties; report points |

Lake-years per year: 2012 49 · 2013 75 · 2014 88 · 2015 115 · 2016 149 · 2017 143 · 2018 158 · 2019 98 ·
2020 182 · 2021 194 · 2022 221 · 2023 206 · 2024 218 · 2025 252. Status (published rows 2012–18):
suspicious 230 · confirmed 414 · confirmed with high toxins 236; collapsed lake-years 185 / 386 / 206.
2019+: all 1,371 are "confirmed". First-report month (lake-years): Jun 428 · Jul 568 · Aug 515 ·
Sep 341 · Oct 118, May 101, rest ≤ 37. 2019+ reports per lake-year: median 2, max 376; 650 lake-years
rest on a single report. NYHABS 2026 extents: small localized 614 · large localized 534 · widespread/
lakewide 234 · open water 13 · not reported 56.

## Label definitions

| Column | Definition |
|---|---|
| `bloom_status` | Ordered `suspicious` < `confirmed` < `confirmed_high_toxin` (`status_rank` 1–3); the MAXIMUM status DEC documented for the waterbody that year. Codes: S, C, HT (2012–18); "confirmed" (2019+); NYHABS `HAB_STATUS` Confirmed / Confirmed_with_High_Toxins. |
| `status_basis` | **Definitions differ by era.** 2012–18: C/HT = "water sampling results have confirmed" (S = visual/photo only) → `water_sampling` / `visual_or_photo`. 2019+: confirmed = "visual observations and/or digital photographs **or** water sampling" → `visual_photo_or_sampling`. A 2019+ "confirmed" is therefore closer to a 2012–18 S-or-C. |
| `label_confirmed` | `bloom_status` ≥ confirmed. Recommended positive label (era-aware; see above). |
| `label_high_toxin` | HT ("toxins present in quantities to potentially cause health effects"); 2012–18 only, `<NA>` for 2019+ (not reported in the summary). |
| `first_report_date`, `last_report_date` | 2012–18: first/last week on the DEC notification list (`date_basis = dec_notification_listing`); 2019+: first/last NYHABS report (`nyhabs_report`). Not bloom onset/end. |
| `n_reports` (2019+), `n_weeks_listed`, `n_weeks_with_updates` (2012–18) | Effort/persistence indicators, not severity. |
| NYHABS `date`, `report_time_utc` | Observation time (ArcGIS date fields are UTC per the layer's `dateFieldsTimeReference`); `date` = America/New_York calendar date. `status_time_utc` = status date (DIL views only). |
| `extent_class` / `extent_rank` | Observer's rough extent: small_localized 1 < large_localized 2 < widespread_lakewide 3; `open_water` (sample near lake centre) and `not_reported` have no rank. |

## QC applied

1. Status codes mapped by a closed vocabulary (unknown codes raise an error).
2. Dates parsed with calendar validation (2012-01-01 … retrieval + 1 day); a first/last date whose
   year differs from `year` gets `year_mismatch` (none in this vintage). All 2,251 rows passed.
3. Coordinates validated against a NY box (W −80.0, S 40.4, E −71.7, N 45.1) with zero/invalid
   flags (all passed).
4. `record_id` = SHA-1 of all published data fields (stable across Socrata re-uploads, unlike
   `:id`). Three verbatim duplicate rows in `qtg2-hjth` (Waneta Lake 2015 Schuyler and 2017 Steuben,
   Tully Lake 2016) get a `#2` suffix and `is_exact_duplicate = True`; they are dropped before the
   lake-year collapse.
5. 2012–18 multi-county lakes are listed once per county (e.g. Canandaigua: Ontario + Yates), sometimes
   with different dates or status. `lake_years` collapses rows sharing (era, normalised name, year,
   lat/lon rounded to 0.001°) → max status, earliest first date, latest last date, max weeks, joined
   counties (60 lake-years built from > 1 published row). Same-name different lakes (e.g. Brooks Lake, Orange vs Richmond)
   stay separate because their coordinates differ.
6. NYHABS: the four views are merged on `globalid` (lake sections come only from the non-DIL view,
   status dates only from the DIL views); `window` = `archived` or `current_14d`.

## Known issues and biases

- **Effort- and complaint-driven.** Reports come from the public, DEC programmes (LCI, CSLAP
  volunteers) and partner programmes; NYHABS 2026 data providers: PUB 620, CAY 196, SBK 152, OPR 102,
  CAN 81, DOH 59, SEN 57, NYC 49, … (codes are not documented; PUB appears to mean public reports —
  unverified). Busy lakes with active monitors (Cayuga 279 reports in 2025) dominate counts.
- **Positive-only.** Only blooms are listed; from 2019 only confirmed blooms. Waterbody-years with no
  row are unknown, not negative — use these rows as positives near sensors and build negatives from
  sensor/satellite definitions, never from absence.
- **Era break in 2019:** reporting system (NYHABS) and the confirmed definition changed, so counts
  and the confirmed fraction are not comparable across 2018/2019 (185 → 98 rows).
- **Lake centroids.** Summary coordinates are one point per waterbody (DEC centroid); blooms may be
  shoreline-local. One 2018 row (Brooks Lake, Orange County) carries the Staten Island Brooks Lake
  coordinates — county/coordinate mismatches exist.
- **First report ≠ onset;** last report ≠ end (no end dates exist). Use "bloom documented in year y /
  week w".
- **Current-season view filters** hide suspicious reports and status changes from the last 27 h; the
  2026 report set will still change until DEC posts the 2026 summary (expected early 2027).
- NYHABS county text is sometimes truncated (`New` for New York County).

## Join notes

- Lake centroid / report point → NHD (or NHDPlus HR) waterbody polygon by point-in-polygon (fallback:
  nearest waterbody within 200 m, record distance), then the waterbody COMID → CyAN lake / EPA
  cyanoHAB-forecast lake id and LAGOS-US `lagoslakeid` via their COMID crosswalks.
- Across eras and with NYHABS points use `waterbody_key` (normalised name) **plus** distance
  (≤ 2 km between centroid/report point) — names repeat across the state (13 "Long Pond" lake-years).
- Not pulled here but public: DEC `Lakes_Locations_Portal_View` feature service (same ArcGIS org) for
  DEC lake locations.
- No HUC/COMID is published; `county` is text (multi-county strings in 2019+, e.g. "Ontario, Yates").

## Rebuild

```bash
# new vintage (raw/ny_habs/<today>/), rebuild processed tables + manifest (accumulates NYHABS snapshots)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_events_ny_habs.py
# rebuild 2026-10-22 offline from raw
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_events_ny_habs.py --vintage 2026-10-22 --skip-download
# tests (fixtures: tests/fixtures/labels_ny_habs/, real recorded rows/features)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_ny_habs.py
```

To archive NYHABS per-report data for future seasons, run the downloader weekly from May to
November; every vintage folder is kept and merged by `globalid`.

Code: `sentinel/data/sources/events_ny_habs.py`, `scripts/data/download_events_ny_habs.py`.
