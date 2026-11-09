# Datasheet — `labels_cws_boundaries` (EPA community water system service-area boundaries v3)

**Use:** the spatial key for TapRisk (D4). It links a public water system (PWSID) to a place, so SDWIS violations (`labels_sdwis`) and UCMR results (`labels_ucmr`) can be related to SENTINEL's river and lake predictions.

> **Join limitation (read first).** A service area is where a system **delivers** water. It is not where the water is **drawn**. The locations of drinking-water intakes and source-water protection areas are not public, because EPA and the states withhold them for security reasons.
>
> Any aggregation of SENTINEL predictions to a PWS is therefore a **proxy**. Examples are predictions inside the service area, in the HUC12s it overlaps, or on the NHDPlus network upstream of it. The proxy can be wrong: a city may draw from a reservoir 50 km away, or across a basin divide.
>
> For purchased-water systems (`SWP`/`GUP`), follow the seller PWSID in `labels_sdwis/sdwis_source_facilities.parquet`. Facility type `CC` (consecutive connection) names the wholesaler that actually treats the source water. 99.6% of active purchased-surface-water CWS have such a link. For self-supplied SW/GU systems, 91.8% have an active intake or reservoir facility record, but those records carry **no coordinates**.

## Source

| Item | Value |
|---|---|
| Publisher | US EPA Office of Water / Office of Research and Development. Public Water System Service Area Boundaries, **version 3** (EPA page: "Version 3 now available! March 2026"; the GitHub Version_History README says v3.1 was released 2026-03-15; the zip folder is `3_0`) |
| Official bulk file ("Download Map Data" on the EPA page) | https://github.com/USEPA/ORD_SAB_Model/raw/refs/heads/main/Version_History/PWS_Boundaries_Latest.zip → Git LFS `media.githubusercontent.com` (570,438,381 bytes). The ETag equals the sha256 `4bc410d6…a162b6e77b`; **the download's sha256 matches the ETag**. The LFS server's Last-Modified is the request time, so the content version is identified by the sha256. The GeoPackage inside is dated 2026-03-27 |
| Landing page (verified 2026-10-22) | https://www.epa.gov/ground-water-and-drinking-water/community-water-system-service-area-boundaries (resources tab). Repository: https://github.com/USEPA/ORD_SAB_Model |
| Hosted service (cross-check only) | https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/Water_System_Boundaries/FeatureServer/0 (layer "Final"; last data edit 2026-05-29) |
| Access | No key or login |
| License / terms | US Government work (EPA). The repository's `license.md` is MIT ("U.S. Federal Government (in countries where recognized)"). EPA states that modeled boundaries are estimates and that it cannot verify non-EPA boundaries |
| Retrieved | 2026-10-22 |
| Documentation kept in `raw/cws_boundaries/` | `SAB_v3_README.md`, `SAB_v3_Census_Tables_README.md`, `SAB_Documentation.pdf` (4 MB), `zip_info.json` (headers, sha256, member list) |

## Files (`$SENTINEL_DATA/processed/labels_cws_boundaries/`)

| File | Rows | Content |
|---|---|---|
| `cws_boundaries.parquet` | 44,656 | GeoParquet, **EPSG:4326** (reprojected from EPSG:4269 NAD83), MultiPolygon, not simplified. One row per CWS polygon; 44,642 unique PWSIDs |
| `cws_unmapped.parquet` | 5,082 | CWS that SAB v3 could **not** map (`Missing_CWS_V_3_0.csv`; NY 460, GA 349, PR 338, CO 316, …) |
| `cws_tract_crosswalk.parquet` | 165,121 | PWS ↔ 2020 census tract links (44,473 PWS): area weight, building-footprint weight, population estimates |
| `cws_block_group_crosswalk.parquet` | 337,332 | PWS ↔ 2020 block group links (same weights) |
| `build_stats.json` | — | All counts quoted here, including the FeatureServer comparison |

**Not kept:**
- The `T_NTNC` layer (77,469 non-community systems such as schools and campgrounds).
- The census-block crosswalk (a 403 MB CSV).
- The zip (570 MB) and the extracted GeoPackage (686 MB). Both were deleted after the build; their sha256 is in the manifest. Rerunning the build script re-downloads them.

## Coverage

- 44,656 polygons for 44,642 CWS PWSIDs, covering all 50 states, DC, territories and tribal systems.
- EPA states this covers about 99% of the population served by CWS.
- **Provenance** (`boundary_source`, from the SAB `Symbology_Field`):
  - 25,409 polygons are "System Sourced" (state or utility data).
  - 19,247 polygons (**43.1%**) are "Modeled".
- `model_method` values: Random Forest 16,330; Decision Tree 1,578; Parcel 1,086; OSM 253; blank or None for most system-sourced polygons.
- 7 PWSIDs appear more than once, in 21 rows (multi-part systems delivered as separate rows; VA5019052 has 8). They are flagged with `pwsid_duplicated`.
- 1 polygon (MO1010714) is invalid (self-intersection). It is kept unrepaired and flagged `geometry_valid = False`; its `area_km2` is measured on a `make_valid()` copy.
- **Coverage of SDWIS (2026Q3):** 89.2% of active CWS and 90.0% of the 11,886 active surface-water-sourced CWS have a boundary polygon.
- **Hosted service vs bulk file:**
  - The FeatureServer has 44,615 features (44,613 PWSIDs).
  - 44,612 PWSIDs are common to both.
  - 30 PWSIDs are only in the bulk GeoPackage; 1 is only in the service.
  - The bulk file is the reference.

## Schema (`cws_boundaries.parquet`)

**IDs and system attributes:**
- `pwsid` (string)
- `pws_name`
- `primacy_agency`
- `state` (first two characters of the PWSID)
- `pop_cat_5`
- `population_served`, `service_connections` (Int64)
- `service_area_type`

**Boundary provenance:**
- `boundary_source`: System Sourced or Modeled.
- `is_modeled` (bool).
- `model_method`.
- Supplier detail: `original_data_provider`, `data_provider_type`, `data_source` (URL), `modification_method`, `feature_type`, `method_details`.
- Verification: `verification_status`, `confirmed`, `date_created`.

**Geometry:**
- `pwsid_duplicated`.
- `geometry_valid`: the output of shapely `is_valid` (1 invalid polygon). Geometries are not repaired.
- `area_km2`: computed on EPSG:6933 (equal area); invalid polygons are measured on a `make_valid()` copy.
- `geometry`.

## QC applied

- **CRS:** reprojected from EPSG:4269 to EPSG:4326. The datum shift between them is at most a few metres. No simplification or precision rounding is applied.
- **IDs:** strings; whitespace and non-breaking spaces are stripped.
- **Blanks:** set to `<NA>`.
- **Duplicates:** kept and flagged, never dropped.
- **Encodings:** `Missing_CWS_V_3_0.csv` is cp1252 (for example "ACUED. RURAL DOÑA ELENA"); the tract and block-group CSVs are ASCII.
- **Census keys:** census `GEOID20` values are zero-padded strings (11 characters for tracts, 12 for block groups).

## Known biases and caveats

1. **Modeled boundaries are estimates.** 43% are machine-learning or parcel models built from SDWIS attributes, census places, parcels and OSM. Stratify any analysis by `is_modeled`.
2. **Incomplete coverage.** 5,082 CWS are unmapped, concentrated in the Northeast, Puerto Rico and tribal or territorial systems. Small systems are over-represented in both the modeled set and the unmapped set.
3. **Mixed vintages.**
   - State sources date from 2004 to 2025, and some `method_details` say "may be out of date".
   - The v3 README states that the CWS polygons are identical to v2.1.
   - The SDWIS attributes stored with the boundaries come from the SAB build, not from the 2026Q3 `labels_sdwis` snapshot. v1.2 used SDWIS 2024Q4; the v2.x/v3 vintage is not stated. Take population and system type from `labels_sdwis`.
4. **Service area ≠ source water** (see the box above). Consecutive systems can be far from their wholesaler's intake.
5. **Overlaps:** polygons can overlap (wholesalers vs retail systems; duplicated systems).

## Joins

1. **To SDWIS and UCMR:** join on `pwsid`.
2. **To hydrography (proxy):** intersect polygons with WBD HUC12 or NHDPlus V2 catchments, or seed NLDI upstream navigation from flowlines that touch the polygon. Record the method, and treat the result as a proxy for the source-water catchment.
3. **To census, CDC SVI and EJ layers:** use `geoid20` from the tract or block-group crosswalk. Prefer `bldg_weight` over `area_weight`.

## Rebuild

```
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_cws_boundaries.py --check-service
```

The script re-downloads the 570 MB zip, verifies sha256 against the ETag, extracts the GeoPackage to `interim/`, writes `processed/labels_cws_boundaries/` and `manifests/labels_cws_boundaries.json`, and then deletes the zip and the GeoPackage. Use `--keep-raw` to keep them. Peak disk use is about 1.3 GB. Tests: `python -B -m pytest tests/test_labels_drinking_water.py`.
