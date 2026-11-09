# Datasheet — `hydromet_upstream_forcing` (upstream-basin-averaged daily gridMET forcings for M6 sites)

**OBSERVED weather (gridMET analysis).** Same caveat as `forcings_daily`: for future days this is an oracle, never
a forecast covariate. Built for SENTINEL-HydroMet (M6) protocol v0.10 (upstream-area forcings, Siddik et al. 2025
style). Contains no in-situ measurement of any kind.

**Files** (`$SENTINEL_DATA/processed/hydromet_upstream_forcing/`, manifest `manifests/hydromet_upstream_forcing.json`)
- `upstream_forcing.parquet`: one row per (site_key, date); columns `site_key`, `date` (gridMET day), the 10
  harmonised gridMET variables of `forcings_daily` (`precip_mm`, `tmax_c`, `tmin_c`, `srad_wm2`, `wind_ms`,
  `rhmax_pct`, `rhmin_pct`, `sph_kgkg`, `vpd_kpa`, `pet_mm`; same units) as basin means, and `status`
  (`permanent`/`provisional`/`early`, most provisional over variables). One row group per 100 sites.
- `basins.parquet`: per site `site_no`, `site_key`, `lat`, `lon`, `comid` (StreamCat COMID), `comid_method`,
  `streamcat_wsareasqkm`, `basin_area_km2` (from the covered sub-cells), `n_catchments`, `n_cells`, `cover_frac`,
  `fallback`, `method`, `nldi_basin_area_km2` (QC, cached NLDI polygon where available).
- `coverage.parquet`: `basins.parquet` without geometry, plus QC columns `missing_frac`, `tmax_mean_diff_c`
  (basin − site-cell mean Tmax, WY2008–2021), `precip_mean_ratio`, `area_ratio_vs_streamcat`.
- Interim (deletable after build): `$SENTINEL_DATA/interim/hydromet_upstream_forcing/` — NLDI cache (QC), `catchment_cells.npz`
  (catchment × cell sub-cell counts), `weights.npz`
  (CSR site × gridMET-cell weights), per (variable, year) `agg/*.npz`, `build.log`. gridMET netCDFs are downloaded one
  at a time to `interim/.../nc/` and deleted after use.

**Code:** `sentinel/data/hydromet_upstream.py` · `scripts/data/build_hydromet_upstream_forcing.py {basins_local,aggregate,combine}`
· tests `tests/test_hydromet_upstream.py`. Model use: `sentinel.models.hydromet.data.attach_upstream_forcing`
(config key `upstream_forcing: append|replace`).

## Sources and licence
- NHDPlus V2 network: VAA `hydroseq`/`dnhydroseq` (`raw/nhdplus_vaa`, HydroShare 6092c8a62fac45be97a09bfd0b0bf726; public
  domain).
- NHDPlus V2 **simplified catchments** (`NHDPlusV21_<DA>_<VPU>_NHDPlusCatchment2_03.7z`, 21 CONUS VPUs, 836 MB, EPA S3
  `https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/Extensions/SimplifiedCatchments/`; public domain),
  stored as per-VPU GeoParquet in `interim/hydromet_wqp_national/catchments/` (shared with the track-N build).
- gridMET yearly netCDF `https://www.northwestknowledge.net/metdata/data/<var>_<year>.nc`, 2005–2026 (CC0 / public
  domain; Abatzoglou 2013).
- USGS NLDI `comid/{comid}/basin` polygons: cached for ~900 sites before the switch (QC), and used for the sites flagged by
  the area rule below.

**Method change (2026-11-08, before any use).** The first build fetched NLDI basin polygons, but the shared NLDI
quota (3,600 requests/h per IP, all SENTINEL jobs) was exhausted after ~800 sites (HTTP 429 waits). To keep the quota
for other workstreams, basins are delineated locally from the same NHDPlus V2 data NLDI uses; the cached NLDI
polygons are only used to check the local areas.

## Sites
All M6-eligible sites of the FINAL sonde split, every role (2,559: stream site types, gridMET forcing source,
StreamCat attributes present). Only inputs are built; no target value is read.

## Method
1. Upstream set of the site's StreamCat COMID: all VAA flowlines whose `dnhydroseq` chain reaches it (breadth-first
   search on the reversed main-path graph), including the site's own catchment. Minor divergences follow their
   `dnhydroseq`, so a divergent branch counts toward the basin it rejoins.
2. Each catchment is rasterised once on a 1/120° grid (5 × 5 sub-cells per gridMET cell; sub-cell centre in polygon).
   Catchments covering no sub-cell centre get their representative-point cell with one sub-cell. A basin's weight for
   a gridMET cell = its covered sub-cells × cos(latitude), over valid land cells, normalised to 1. `cover_frac` = share
   on valid cells. Sites whose COMID is not on the VAA network use their own gridMET cell (`fallback = True`).
3. Daily basin mean = weighted mean over cells finite that day (weights renormalised; NaN if none). Kelvin → °C for
   Tmax/Tmin (exact offset), other variables in file units (as `forcings_daily`).
4. QC rule (fixed before use): the main-path graph misses area upstream of divergences and diversions (first pass:
   58 sites < 0.5 × and 16 > 2 × the StreamCat watershed area, e.g. Santee–Cooper, Pee Dee). For every site outside
   [0.5, 2] × StreamCat `wsareasqkm`, the NLDI basin polygon (which follows divergences) replaces the local basin and is
   rasterised with the same sub-cell rule (`method = nldi_polygon`; a few dozen requests).
5. Basin parts outside CONUS (Canada, Mexico) are not covered: NHDPlus V2 has no catchments there, which lowers
   `basin_area_km2` for, e.g., the Columbia, Red River of the North and St. Lawrence tributaries.

## Coverage, QC and known issues (build of 2026-11-08 19:57 UTC)
- 2,559 sites × 2005-10-01 … 2026-11-06 = 19.7 M rows; gridMET status: 7,527 permanent, 162 provisional, 18 early days.
- Basins: 2,485 local, 74 NLDI polygons (area rule), 3 fall back to the site cell (COMID off the VAA network).
  gridMET cells per basin: median 58 (quartiles 13–326, max 264,089 — Mississippi).
- Area checks: local ÷ NLDI on 961 sites with a cached NLDI polygon: 1st/10th/50th/90th/99th percentiles
  0.88 / 1.00 / 1.05 / 1.12 / 1.33 (the ~5 % median excess comes from rasterising the simplified catchment
  polygons). Basin ÷ StreamCat `wsareasqkm`: 0.69 / 1.00 / 1.04 / 1.10 / 1.40;
  13 sites stay outside [0.5, 2] after the rule (StreamCat and NLDI disagree there too) — kept and flagged by
  `area_ratio_vs_streamcat` in `coverage.parquet`.
- 32 basins have < 95 % of their area on gridMET land cells (Canada/Mexico parts, coastal water).
- Consistency: for single-cell basins the basin Tmax equals the site-cell Tmax exactly (max |Δ| 0.0 °C); across all
  sites the basin is a median 0.67 °C colder than the site cell (1st percentile −8.4 °C: mountain headwaters).
- No synthetic data; no in-situ value is used.

## Rebuild
```
python -B scripts/data/build_hydromet_wqp_national.py catchments   # once (shared simplified catchments)
python -B scripts/data/build_hydromet_upstream_forcing.py all
```
