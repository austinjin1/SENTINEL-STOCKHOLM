# Datasheet — `taprisk_catchment` (D4 TapRisk: CWS service-area COMID links + StreamCat)

**Use:** D4 TapRisk covariates (protocol `docs/preregistration/downstream_taprisk.md`). Built by
`scripts/data/build_taprisk_links.py` (stages points → links → streamcat → manifest). No labels are read.

| Item | Value |
|---|---|
| Units | CWS (SDWIS `pws_type_code` CWS) with own source (`primary_source_code` SW, GU, GW), active at some time 2010–2024, with an EPA SAB v3 service-area polygon (`labels_cws_boundaries`): 35,803 systems (GW 31,546, SW 3,782, GU 475) |
| Location | polygon representative point (`shapely.point_on_surface`, inside the polygon); 43 % of SAB polygons are modelled (`is_modeled` kept) |
| COMID | USGS NLDI `linked-data/comid/position` snap of the point (NHDPlus V2), cached in `raw/taprisk_links/nldi_cache` |
| Attributes | EPA StreamCat curated metric groups (same list as `processed/catchment` / `downstream_catchment`), catchment + watershed |
| Licence | US public domain (EPA SDWIS / SAB / StreamCat; USGS NLDI / NHDPlus V2) |
| Retrieved | 2026-11-08/09 |

## Known limitations
- SDWIS source-facility (well/intake) coordinates are not public; the catchment of the service-area point is a
  proxy for the source catchment (Pennino et al. 2020 used facility locations). GW wells may lie outside the
  service-area catchment; SW intakes are often upstream of it (watershed metrics partly compensate).
- StreamCat metrics are static (e.g. NLCD 2019) and postdate early origins.
- Purchased-water systems are excluded.

## Files and counts
| File | Rows | Content |
|---|---|---|
| `system_points.parquet` | 35,803 | pwsid, source code, state, representative point (lat/lon), `is_modeled`, polygon area |
| `system_links.parquet` | 35,803 | NLDI snap: 35,151 `ok` (GW 31,017, SW 3,674, GU 460), 652 `http_404` (no NHDPlus catchment at the point; mostly coastal) → 29,856 unique COMIDs (NLDI run 2026-11-08 18:33 → 2026-11-09 07:01, 0 errors, 0 HTTP 429) |
| `streamcat_comid.parquet` | 29,856 | 179 curated StreamCat columns (10 metric groups); 29,696 COMIDs with ≥ 1 value |
| `streamcat_dictionary.csv` | 179 | column → StreamCat metric, AOI, description |
Manifest: `manifests/taprisk_catchment.json` (2026-11-09T19:29Z, 26.9 MB).
