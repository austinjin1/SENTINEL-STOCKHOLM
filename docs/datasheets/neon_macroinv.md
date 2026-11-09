# Datasheet — neon_macroinv (Macroinvertebrate collection)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20120.001` — Macroinvertebrate collection (https://data.neonscience.org/data-products/DP1.20120.001; RELEASE-2026 https://doi.org/10.48443/hp56-s582)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_macroinv/`

All basic-package tables of DP1.20120.001 for the 34 aquatic sites, typed with the product variables file.

## Files

| file | content |
|---|---|
| `inv_fieldData.parquet` | stacked NEON table |
| `inv_persample.parquet` | stacked NEON table |
| `inv_taxonomyProcessed.parquet` | stacked NEON table |
| `density.parquet` | inv_taxonomyProcessed + field benthicArea → density_per_m2 (estimatedTotalCount / benthicArea) |

## Coverage

```json
{
 "inv_fieldData": {
  "rows": 8625,
  "first": "2014-07-01",
  "last": "2026-09-21",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 7201,
   "PROVISIONAL": 1424
  }
 },
 "inv_persample": {
  "rows": 7532,
  "first": "2014-07-01",
  "last": "2026-08-10",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 6446,
   "PROVISIONAL": 1086
  }
 },
 "inv_taxonomyProcessed": {
  "rows": 368654,
  "first": "2014-07-01",
  "last": "2026-08-10",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 320240,
   "PROVISIONAL": 48414
  }
 },
 "density": {
  "rows": 368654,
  "with_density": 368337
 }
}
```

## QC applied

- Typed by variables file; IDs as strings.

## Known issues

- Staff identifiers (`*By`) dropped.
- Release/provisional flagged per row.

## Rebuild

```bash
python -B scripts/data/download_neon.py --products macroinv
python -B scripts/data/build_neon.py macroinv
```
