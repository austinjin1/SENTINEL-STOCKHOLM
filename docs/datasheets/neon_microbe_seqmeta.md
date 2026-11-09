# Datasheet — neon_microbe_seqmeta (NEON surface-water marker-gene sequencing metadata)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20282.001` — Surface water microbe marker gene sequences (https://data.neonscience.org/data-products/DP1.20282.001; RELEASE-2026 https://doi.org/10.48443/dx8c-sy17)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_microbe_seqmeta/`

All DP1.20282.001 (expanded) tables stacked across sites and months.

## Files

| file | content |
|---|---|
| `amc_fieldGenetic.parquet` | stacked table |
| `amc_fieldSuperParent.parquet` | stacked table |
| `mic_swDnaPooling.parquet` | stacked table |
| `mmg_swDnaExtraction.parquet` | stacked table |
| `mmg_swMarkerGeneSequencing_16S.parquet` | stacked table |
| `mmg_swMarkerGeneSequencing_ITS.parquet` | stacked table |
| `mmg_swPcrAmplification_16S.parquet` | stacked table |
| `mmg_swPcrAmplification_ITS.parquet` | stacked table |
| `mmg_swRawDataFiles.parquet` | stacked table |

## Coverage

```json
{
 "amc_fieldGenetic": {
  "rows": 3541,
  "first": "2014-08-05",
  "last": "2026-10-07",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 2437,
   "PROVISIONAL": 1104
  }
 },
 "amc_fieldSuperParent": {
  "rows": 4126,
  "first": "2014-08-05",
  "last": "2026-10-07",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 2547,
   "PROVISIONAL": 1579
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
 },
 "mmg_swDnaExtraction": {
  "rows": 3420,
  "first": "2014-08-05",
  "last": "2026-06-29",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 2471,
   "PROVISIONAL": 949
  }
 },
 "mmg_swMarkerGeneSequencing_16S": {
  "rows": 2411,
  "first": "2014-08-05",
  "last": "2024-11-12",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 2411
  }
 },
 "mmg_swMarkerGeneSequencing_ITS": {
  "rows": 2246,
  "first": "2014-08-05",
  "last": "2024-11-12",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 2246
  }
 },
 "mmg_swPcrAmplification_16S": {
  "rows": 2514,
  "first": "2014-08-05",
  "last": "2024-11-12",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 2514
  }
 },
 "mmg_swPcrAmplification_ITS": {
  "rows": 2453,
  "first": "2014-08-05",
  "last": "2024-11-12",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 2453
  }
 },
 "mmg_swRawDataFiles": {
  "rows": 9942,
  "first": "2014-08-05",
  "last": "2024-11-12",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 9942
  }
 }
}
```

## QC applied

- Typed by the product variables file; staff identifiers dropped.

## Known issues

- Raw FASTQ files are referenced (mmg_swRawDataFiles) but not downloaded.

## Rebuild

```bash
python -B scripts/data/download_neon.py --products microbe_seq
python -B scripts/data/build_neon.py microbe
```
