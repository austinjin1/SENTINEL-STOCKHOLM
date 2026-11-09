# Datasheet — HydroLAKES v1.0 polygons (`hydrolakes`)

| | |
|---|---|
| **Source** | Messager, M.L. et al. (2016) *Estimating the volume and age of water stored in global lakes using a geo-statistical approach*, Nat. Commun. 7:13603 (doi:10.1038/ncomms13603). HydroSHEDS. |
| **URL** | https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip (820,295,132 B) |
| **Licence** | CC BY 4.0 |
| **Retrieved** | 2026-10-25 01:38 UTC. The sha256 of the zip is in manifest `hydrolakes` (`stats.raw_zip_sha256`). The zip was deleted after parsing. |
| **Local path** | `$SENTINEL_DATA/processed/hydrolakes/hydrolakes_na_eu.parquet` (GeoParquet, EPSG:4326, 818 MB) |
| **Code** | `sentinel/data/sources/hydrolakes.py` and `scripts/data/build_hydrolakes.py` |
| **Used by** | SENTINEL-Orbit split rule (`sentinel/models/orbit/split.py`): water-body identity and lake pour-point anchors |

## Contents
- Lakes of at least 10 ha inside two bounding boxes:
  - North America (lon −170…−50, lat 14…72): 974,363 lakes.
  - Europe (lon −25…45, lat 34…72): 101,283 lakes.
- Columns: `Hylak_id`, `Lake_name`, `Country`, `Continent`, `Lake_area` (km²), `Shore_len`,
  `Depth_avg`, `Vol_total`, `Elevation`, `Wshd_area`, `Pour_long`, `Pour_lat`, plus the polygon and
  `region`.

## Point assignment (`assign_points`)
- A point is assigned to the lake polygon that contains it.
- Otherwise it goes to the nearest polygon within 100 m. This covers shore stations and coarse
  polygons. Distances are computed in EPSG:5070 for North America and EPSG:3035 for Europe.
- Ties go to the smallest `Hylak_id`.

## Known issues
- The minimum lake size is 10 ha, so smaller ponds are unassigned. Reservoirs inside rivers are
  only partly covered. Unassigned points become their own water body (see orbit_protocol.md §2.2).
- Of the 19,433 Orbit development rows, 52% fall in a HydroLAKES lake.

## Rebuild
```bash
curl -A "SENTINEL-v2 research (public-data download)" -o $SENTINEL_DATA/raw/hydrolakes/HydroLAKES_polys_v10_shp.zip \
  https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip
python -B scripts/data/build_hydrolakes.py
```
