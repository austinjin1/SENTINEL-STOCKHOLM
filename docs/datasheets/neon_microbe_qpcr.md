# Datasheet — neon_microbe_qpcr (Surface-water microbe group abundances (qPCR))

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20278.001` — Surface water microbe group abundances (https://data.neonscience.org/data-products/DP1.20278.001; RELEASE-2026 https://doi.org/10.48443/vht7-gy53)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_microbe_qpcr/`

All basic-package tables of DP1.20278.001 for the 34 aquatic sites, typed with the product variables file.

## Files

| file | content |
|---|---|
| `mga_swGroupAbundances.parquet` | stacked NEON table |
| `mic_swDnaPooling.parquet` | stacked NEON table |

## Coverage

```json
{
 "mga_swGroupAbundances": {
  "rows": 5278,
  "first": "2014-08-05",
  "last": "2025-12-17",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 4698,
   "PROVISIONAL": 580
  }
 },
 "mic_swDnaPooling": {
  "rows": 54,
  "first": "2024-07-01",
  "last": "2025-09-29",
  "sites": 31,
  "releases": {
   "PROVISIONAL": 29,
   "RELEASE-2026": 25
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
python -B scripts/data/download_neon.py --products microbe_qpcr
python -B scripts/data/build_neon.py qpcr
```
