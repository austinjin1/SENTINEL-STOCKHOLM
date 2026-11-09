# Datasheet — `downstream_catchment` (COMID links + StreamCat for downstream label units)

**Files** (`$SENTINEL_DATA/processed/downstream_catchment/`, manifest `manifests/downstream_catchment.json`)
- `unit_links.parquet`: one row per label unit — `unit_source` (`nrsa`, `iowa_fishkill`, `powell`,
  `streampulse`), `unit_id` (NRSA `<survey>:<site_id>`, Iowa `event_id`, Powell/StreamPULSE `site_id`),
  `lat`, `lon`, `comid_published`, `comid_published_source`, `comid_snap`, `snap_status`, `comid` (final,
  string), `comid_method` (`published:<source>` / `nldi_position` / `unlinked`).
- `streamcat_comid.parquet`: one row per unique COMID — the curated StreamCat metric groups of
  `sentinel.data.sources.streamcat.CURATED` (same list as `processed/catchment`, minus metrics the API
  rejected there), `catareasqkm`/`wsareasqkm`, NHDPlus V2 VAA `nhd_slope`/`nhd_streamorde`/
  `nhd_totdasqkm`/`nhd_lengthkm`, `streamcat_found`.
- `streamcat_dictionary.csv`: column → metric, AOI, year, units, description.
- Raw/cache: `raw/downstream_links/{nldi_cache,streamcat_cache,build.log,nldi_stats.json}`.

**Code:** `sentinel/data/downstream_links.py` · `scripts/data/build_downstream_links.py`
**Built:** 2026-10-24 (EDT). **Synthetic data:** none.

## Sources and licence
- USGS NLDI `https://api.water.usgs.gov/nldi/linked-data/comid/position` (point-in-catchment, NHDPlus V2),
  ≤ 1 request / 1.25 s, cached; public domain.
- EPA StreamCat REST API `https://api.epa.gov/StreamCat/streams/metrics` (POST, 500 COMIDs per request,
  4 s spacing, cached); public domain; cite Hill et al. 2016 (JAWRA 52:120–128).
- NHDPlus V2 VAA (local `raw/nhdplus_vaa`).

## Linking rules
- **Published COMID kept** where the label source provides one: NRSA 2013-14, 2018-19 and part of
  2023-24 site files; Powell `nhdplus_id` (except the 12 sites whose COMID disagrees with Bernhardt et
  al. 2022, which are re-snapped); StreamPULSE `comid`.
- **Published COMIDs are validated** (`validate_published`): values that are not digit strings (NRSA
  2023-24 site files went through spreadsheet scientific notation, e.g. `1.03E+11`) or are absent from the
  NHDPlus V2 VAA are discarded (raw value kept in `comid_published_invalid`) and the unit is snapped instead
  (2026-10-24 build: 143 NRSA units, 2 Powell sites).
- Otherwise NLDI position snap of the published coordinates (NRSA 2008-09 has no COMID column at all;
  Iowa fish-kill points; remaining StreamPULSE sites). Position snaps can land on a neighbouring small
  catchment (lakes, ponds, large rivers): treat `ws` metrics for such units with care.
- StreamCat's predicted biological condition (`prg_bmmi0809`, Hill 2017) and modelled stream temperature
  are excluded (they are labels/targets elsewhere).

## Coverage
See `manifests/downstream_catchment.json` → `stats` (`links` counts per source × method,
`n_comids`, `n_streamcat_found`).

## Known issues
- National Nutrient Inventory `n_*`/`p_*` columns are totals per AOI (divide by area for intensity).
- `nani`/`nsurp` scale with AOI area despite kg/ha/yr units (see `streamcat.md`).
- Iowa lake/pond kill points snap to a stream catchment, not the waterbody.

## Rebuild
```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/build_downstream_links.py --stage all
```
