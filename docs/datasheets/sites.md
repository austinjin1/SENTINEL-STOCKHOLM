# Datasheet — `sites` (SENTINEL site registry)

**File:** `$SENTINEL_DATA/processed/sites/sites.parquet` (manifest `manifests/sites.json`)
**Builder:** `sentinel/data/sites.py` · `scripts/data/build_sites.py`
**Retrieved / built:** 2026-10-22

## Sources
| Part | Source | Access |
|---|---|---|
| USGS sites (6,482) | `raw/usgs_dv/dv_series_catalog.parquet` — NWIS series catalog of every site with a daily-value (DV) water-quality series (`usgs_legacy.WQ_PARAMS`) | built by `download_usgs_dv.py` |
| USGS time zone, altitude datum, drainage area | NWIS site service `https://waterservices.usgs.gov/nwis/site/?siteOutput=expanded` (100 sites/request) | no key |
| NEON aquatic sites (34) | `https://data.neonscience.org/api/v0/sites` — sites whose product list contains DP1.20288.001 (water quality); `/locations/<site>` → water body (`STREAM…`/`LAKE…`/`RIVER…`) → sensor station | no token (metadata endpoints) |
| HUC for NEON and for 27 USGS sites with an empty `huc_cd` | USGS WBD MapServer layer 6 (HUC12) point query | no key |

Licence: USGS data US public domain; NEON data CC BY 4.0.

## Schema (one row per site)
`site_key` (`USGS-<site_no>` / `NEON-<code>`), `network`, `site_id`, `name`, `lat`, `lon`,
`coord_datum`, `coord_source`, `site_type` (USGS `site_tp_cd`; NEON wadeable streams and
non-wadeable rivers → `ST`, lakes → `LK`), `site_type_detail` (NEON habitat text), `state`,
`huc2`, `huc4`, `huc8`, `huc12` (12-digit only where NWIS gives a 12-digit code or WBD was queried),
`huc_source` (`nwis_huc_cd` / `wbd_point_huc12`), `elev_m` (NWIS `alt_va` ft × 0.3048, or NEON
`locationElevation`), `elev_datum` (NAVD88, NGVD29, LMSL, …), `elev_source`, `tz_cd`,
`utc_offset_h` (standard time, no DST), `drain_area_km2` (NWIS `drain_area_va` mi² × 2.589988),
`in_conus` (bounding box 24.4–49.5 N, 125.0–66.8 W; routing only), `wq_parm_cds`, `n_wq_series`,
`wq_begin`, `wq_end`, `has_temp/has_do/has_sc/has_ph/has_turb/has_no3`; after NLDI has run, also
`comid` (NHDPlus V2, string) and `comid_method`. All IDs (site numbers, HUCs, COMIDs) are strings.

## Coverage (2026-10-22 build)
- 6,516 sites: 6,482 USGS + 34 NEON. Site types: ST 5,280, GW 438, LK 211, ES 168, ST-TS 118,
  ST-CA 80, SP 70, ST-DCH 49, others < 30 each.
- 6,230 inside the CONUS box, 286 outside (AK 158, HI 119, PR 5, GU 4).
- HUC: 6,455 from NWIS `huc_cd`, 61 from the WBD point query; no site lacks a HUC8.
- Time zones: EST 2,531, CST 1,507, PST 1,339, MST 853, AKST 158, HST 119, AST 5, GST (Guam) 4.
- Elevation for 5,291 sites; drainage area for 4,556.
- WQ series flags (USGS): temperature 6,174, SC 3,459, DO 1,654, pH 1,412, turbidity 1,195,
  nitrate 210; DO + SC + temperature at the same site (any years): 1,463.
- NEON coordinates: wadeable streams use the downstream sensor station S2 (the water-quality
  sonde); lakes and non-wadeable rivers use the buoy. The site-level NEON point has a stated
  1,000 m uncertainty and is kept only as a fallback.

## QC applied
- **Row-alignment guard.** Rows are re-checked against the catalog: name, latitude, longitude,
  state and type all match for 100% of the rows. NWIS expanded `tz_cd` and `huc_cd` also match
  for 100% of the rows; `site_tp_cd` matches for 99.98%. A unit test runs the builder on a
  deliberately unsorted catalog. An early build had scrambled USGS rows because `site_id` was
  assigned by position against label-aligned columns. That build was found, discarded and
  rebuilt before any product used it.
- `site_key` uniqueness is enforced (the build fails on duplicates).
- HUC strings are zero-padded and taken verbatim; codes of odd length or non-digits are refused,
  not guessed.

## Known issues
- NWIS `alt_va` contains 459 values ≤ 0.5 m (189 < 0 m). Most are estuarine or tidal sites; a
  few look like placeholders. The values are kept as reported, with their datum.
- The NAD27/NAD83 distinction is not applied to coordinates. NWIS decimal coordinates are
  reported in NAD83 here; NEON uses WGS84. The differences (< 2 m) do not matter at the
  catchment/4-km scale.
- NEON standard UTC offsets come from a state table. All 34 sites lie in single-zone parts of
  their states; Arizona (SYCA) correctly gets MST.
- The registry lists every site with a DV water-quality series ever recorded. 33 non-CONUS sites
  have no data after 2000, so the 2000–present forcings are of little use for them.

## Rebuild
```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/build_sites.py
```
HTTP responses are cached under `raw/site_registry/` (NWIS batches, NEON JSON, WBD JSON), so a
rebuild is offline. After `download_nldi.py`, re-run the build to attach COMIDs.
