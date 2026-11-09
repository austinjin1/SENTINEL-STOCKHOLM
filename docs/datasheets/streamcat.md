# Datasheet — `streamcat` (catchment and watershed attributes per site)

**Files**
- `$SENTINEL_DATA/processed/catchment/streamcat_sites.parquet`: one row per SENTINEL site.
- `streamcat_dictionary.csv`: column → metric, AOI, year, units, description, category, dataset,
  source.
- `streamcat_pctfull.parquet`: per-dataset `*_catpctfull/*_wspctfull` coverage QC.
- Raw: `raw/streamcat/{variable_info.csv,changelog.json,cache/}`.
- Manifest: `manifests/streamcat.json`.

**Code:** `sentinel/data/sources/streamcat.py` · `scripts/data/download_streamcat.py`
**Retrieved:** 2026-10-22. The StreamCat database version (`/changelog`, latest entry) is recorded
in the manifest `stats.streamcat_db_version`.

## Source and download location
EPA StreamCat (Hill et al. 2016, JAWRA 52:120–128): NHDPlus V2 catchment (`cat`), upstream watershed
(`ws`) and 100-m riparian-buffer (`catrp100`, `wsrp100`) summaries for ~2.65 M CONUS stream segments.
Licence: US public domain.

Download location, checked 2026-10-22:
- The bulk HydroRegion/State CSV zips no longer resolve:
  - `gaftp.epa.gov/epadatacommons/ORD/NHDPlusLandscapeAttributes/StreamCat/HydroRegions/` (and the
    `EPADataCommons` spelling) returns 404;
  - the older `newftp.epa.gov` links are still on the EPA updates page;
  - the ORD data-commons S3 bucket returns 403.
- The EPA StreamCat page routes bulk access to its web tool and to the **REST API**
  `https://api.epa.gov/StreamCat/streams` (Swagger: `usepa.github.io/StreamCatWebServices_Public`).
  The API is **keyless**; the response header gives `x-ratelimit-limit: 1000` per hour.
- We use `POST /metrics` (form fields `name`, `aoi`, `comid`, `showareasqkm`, `showpctfull`) in
  batches of 500 COMIDs × one metric group, sequentially, with 4 s between requests and a gzip-JSON
  cache (resume-safe).
- Units and descriptions come from `GET /variable_info`.

## Join key
NHDPlus V2 COMID from `site_network` (`comid_method` ∈ {`nldi_nwissite`, `nldi_position`}).
StreamCat covers CONUS only, so AK/HI/PR/GU sites and sites without a COMID have no attributes
(`streamcat_found = False`).

## Curated attribute set (`sentinel.data.sources.streamcat.CURATED`)
| Group | AOI | Metrics |
|---|---|---|
| landcover | cat, ws | NLCD 2019 classes: open water, developed open/low/medium/high, barren, deciduous, conifer, mixed forest, shrub, grassland, hay, crop, woody and herbaceous wetland; impervious 2019 |
| riparian100m | catrp100, wsrp100 | impervious, crop, deciduous, conifer and mixed forest 2019 in the 100-m buffer |
| agri_nutrients | cat, ws | tile drainage, fertiliser / manure / cultivated-BNF N (AgriculturalNitrogen), NANI, N surplus, pesticide use 1997, inorganic N wet deposition 2008 |
| nutrient_inventory | cat, ws | National Nutrient Inventory 2017 N/P totals (total inputs, farm fertiliser, livestock manure, human waste), USGS point-source N/P loads 2012 |
| infrastructure | cat, ws | population and housing density 2010, road density, road–stream crossings, dam density and NID/normal storage, NPDES / Superfund / TRI densities, WWTP (all, major) density, septic, mine and coal-mine density, canal density |
| lithology | ws | 18 surficial lithology classes (carbonate residual … water) |
| geochem | ws | Al₂O₃, CaO, Fe₂O₃, K₂O, MgO, Na₂O, P₂O₅, S, SiO₂, N, compressive strength, hydraulic conductivity, rock N |
| soils | cat, ws | STATSGO clay, sand, organic matter, permeability, depth to bedrock, water-table depth, K-factor |
| climate_hydro_topo | cat, ws | PRISM 1991–2020 precipitation and Tmax/Tmean/Tmin, baseflow index, runoff, precipitation − ET, wetness index, elevation |
| indices_channel | other | Index of Watershed Integrity, Index of Catchment Integrity, predicted bankfull depth/width, wetted width, thalweg depth |

Further columns:
- `catareasqkm` and `wsareasqkm` (catchment and watershed area).
- NHDPlus V2 VAA reach attributes `nhd_slope` (m/m; −9998 → NaN), `nhd_streamorde`,
  `nhd_totdasqkm` and `nhd_lengthkm`. StreamCat has no slope metric.

Deliberately excluded:
- Model-predicted stream temperature (`mast/msst/mwst`). These overlap SENTINEL targets.
- Predicted NRSA biological condition (`prg_bmmi0809`, Hill et al. 2017). It is a downstream label
  (docs/research/07).

Notes on specific metrics:
- `rddens` uses the populated (non-year) road-density table; `rddens2010` returns no values.
- National Nutrient Inventory `n_*/p_*` columns are total kg per year in the AOI. Divide by
  `catareasqkm`/`wsareasqkm` for an intensity.

## Coverage
See the manifest `stats` for: sites with a COMID, sites found in StreamCat, unique COMIDs, the
number of attribute columns, any requested columns that came back missing, and HTTP statistics.

## Known issues
- Watershed (`ws`) metrics depend on the COMID chosen. A position-linked COMID can sit on a small
  tributary next to the gauge (e.g. lakes and estuaries). Check `comid_method` and compare
  `wsareasqkm` with the registry `drain_area_km2` before trusting `ws` metrics for such sites.
- API quirks seen on 2026-10-22, handled in code (`streamcat.wants_flags`):
  - riparian AOIs fail with HTTP 403 when `showareasqkm`/`showpctfull` are sent;
  - National Nutrient Inventory metrics fail with HTTP 555 (ORA-00923) with the same flags;
  - so those requests omit the flags, and `other`-AOI requests omit them too.
- A metric the API still rejects with an SQL error is dropped and listed in `stats.failed_metrics`.
  Transient 502/503 errors are retried; if they persist the run stops, and a re-run resumes from
  the cache.
- `nani` and `nsurp` (dataset Nsurp_NANI): `/variable_info` gives kg/ha/yr, but the returned values
  scale with AOI area (e.g. ws ≈ 6.6 × 10⁷ for the Illinois River at Florence). Treat them as totals
  until EPA clarifies; their `pctfull` is null.
- Land cover is a single 2019 snapshot; no time-varying land use.

## Rebuild
```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_streamcat.py --batch 500
```
It needs `raw/nldi/comid_links.parquet` or `processed/site_network/nodes.parquet` from `download_nldi.py`.
