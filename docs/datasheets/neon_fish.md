# Datasheet — neon_fish (Fish electrofishing / gill / fyke netting counts)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20107.001` — Fish electrofishing, gill netting, and fyke netting counts (https://data.neonscience.org/data-products/DP1.20107.001; RELEASE-2026 https://doi.org/10.48443/s1e6-df79)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_fish/`

All basic-package tables of DP1.20107.001 for the 34 aquatic sites, typed with the product variables file.

## Files

| file | content |
|---|---|
| `fsh_bulkCount.parquet` | stacked NEON table |
| `fsh_fieldData.parquet` | stacked NEON table |
| `fsh_invertBycatch.parquet` | stacked NEON table |
| `fsh_perFish.parquet` | stacked NEON table |
| `fsh_perPass.parquet` | stacked NEON table |
| `fsh_summary.parquet` | stacked NEON table |

## Coverage

```json
{
 "fsh_bulkCount": {
  "rows": 2051,
  "first": "2016-08-30",
  "last": "2026-06-04",
  "sites": 25,
  "releases": {
   "RELEASE-2026": 1686,
   "PROVISIONAL": 365
  }
 },
 "fsh_fieldData": {
  "rows": 2825,
  "first": "2016-03-29",
  "last": "2026-09-13",
  "sites": 28,
  "releases": {
   "RELEASE-2026": 2344,
   "PROVISIONAL": 481
  }
 },
 "fsh_invertBycatch": {
  "rows": 559,
  "first": "2019-12-02",
  "last": "2026-03-18",
  "sites": 2,
  "releases": {
   "RELEASE-2026": 446,
   "PROVISIONAL": 113
  }
 },
 "fsh_perFish": {
  "rows": 118865,
  "first": "2016-03-29",
  "last": "2026-09-12",
  "sites": 28,
  "releases": {
   "RELEASE-2026": 102421,
   "PROVISIONAL": 16444
  }
 },
 "fsh_perPass": {
  "rows": 4681,
  "first": "2016-03-29",
  "last": "2026-09-13",
  "sites": 28,
  "releases": {
   "RELEASE-2026": 3956,
   "PROVISIONAL": 725
  }
 },
 "fsh_summary": {
  "rows": 1805,
  "first": "2025-02-10",
  "last": "2026-09-12",
  "sites": 27,
  "releases": {
   "PROVISIONAL": 1805
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
python -B scripts/data/download_neon.py --products fish
python -B scripts/data/build_neon.py fish
```
