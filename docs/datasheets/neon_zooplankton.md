# Datasheet — neon_zooplankton (Zooplankton collection)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20219.001` — Zooplankton collection (https://data.neonscience.org/data-products/DP1.20219.001; RELEASE-2026 https://doi.org/10.48443/6p6g-d296)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_zooplankton/`

All basic-package tables of DP1.20219.001 for the 34 aquatic sites, typed with the product variables file.

## Files

| file | content |
|---|---|
| `zoo_fieldData.parquet` | stacked NEON table |
| `zoo_perSample.parquet` | stacked NEON table |
| `zoo_taxonomyProcessed.parquet` | stacked NEON table |

## Coverage

```json
{
 "zoo_fieldData": {
  "rows": 737,
  "first": "2014-07-02",
  "last": "2026-09-14",
  "sites": 7,
  "releases": {
   "RELEASE-2026": 626,
   "PROVISIONAL": 111
  }
 },
 "zoo_perSample": {
  "rows": 735,
  "first": "2014-07-02",
  "last": "2026-07-22",
  "sites": 7,
  "releases": {
   "RELEASE-2026": 646,
   "PROVISIONAL": 89
  }
 },
 "zoo_taxonomyProcessed": {
  "rows": 8431,
  "first": "2014-07-02",
  "last": "2026-07-22",
  "sites": 7,
  "releases": {
   "RELEASE-2026": 7642,
   "PROVISIONAL": 789
  }
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
python -B scripts/data/download_neon.py --products zooplankton
python -B scripts/data/build_neon.py zooplankton
```
