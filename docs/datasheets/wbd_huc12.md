# Datasheet — `wbd_huc12` (CONUS HUC12 universe from the USGS Watershed Boundary Dataset)

| | |
|---|---|
| **Use in SENTINEL** | Candidate universe and flow connectivity (`tohuc`) for D6 MusselGuard (`docs/preregistration/downstream_musselguard.md`) |
| **Source** | USGS WBD MapServer layer 6 "12-digit HU (Subwatershed)", `https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query` (ArcGIS 11.3, maxRecordCount 2000, no key) |
| **Retrieved** | 2026-10-24 (UTC 01:45–01:50 on 2026-10-25), 54 sequential requests ≥ 2 s apart, generic User-Agent |
| **Licence** | US Government work, public domain (cite USGS WBD) |
| **Files** | `$SENTINEL_DATA/processed/wbd_huc12/huc12.parquet` (87,028 rows, 3.1 MB); raw page cache `raw/wbd_huc12/cache/` (gzip JSON); manifest `manifests/wbd_huc12.json` (sha256, per-HUC2 counts, query parameters) |
| **Code** | `sentinel/data/sources/wbd_huc12.py` (request builder, parser, centroid, tidy) · `scripts/data/download_wbd_huc12.py` · tests in `tests/test_downstream_musselguard.py` |
| **Synthetic data** | none |

## Coverage
HUC2 regions 01–18 (CONUS): 87,028 HUC12s (service total incl. AK/HI/PR/islands: 103,068). Per-HUC2
counts are in the manifest (largest: 10 Missouri 13,725; smallest: 06 Tennessee 1,075).

## Columns
`huc12` (12-digit string), `name`, `areasqkm`, `states` (comma list of postal codes), `tohuc` (downstream
HUC12, or a special code such as OCEAN / CANADA / CLOSED BASIN), `tohuc_kind` (`huc12` 85,433 /
`special` 1,595), `hutype`, `humod`, `lon`, `lat` (approximate centroid), `huc2`, `huc4`, `huc8`.

## Processing / QC
- Query per HUC2 prefix (`huc12 LIKE 'NN%'`), ordered by `huc12`, paged by `resultOffset`.
- Geometry requested with `maxAllowableOffset=0.02` (degrees), 3 decimals, WGS84 — only to compute a
  representative point: area-weighted centroid of the largest ring (bbox centre if degenerate). Centroids
  are approximate (≈ 1–2 km) and may fall outside very concave units; use for ≥ 10-km distances only.
- Codes validated as 12 digits; duplicates dropped (none found); no centroid is missing.

## Known issues
- `tohuc` defines the WBD drainage tree; closed basins and coastal units terminate. Great-Lakes open-water
  HUC12s exist as units and are connected through `tohuc`.
- WBD is revised over time; this snapshot is frozen by its sha256 in the manifest and in the MusselGuard
  split meta.

## Rebuild
```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_wbd_huc12.py   # ~5 min, cached/resumable
```
