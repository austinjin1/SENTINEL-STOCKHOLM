# Datasheet — neon_metagenome_meta (Surface-water metagenome metadata)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20281.001` — Surface water microbe metagenome sequences (https://data.neonscience.org/data-products/DP1.20281.001; RELEASE-2026 https://doi.org/10.48443/c2at-7y65)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_metagenome_meta/`

All expanded-package tables of DP1.20281.001 for the 34 aquatic sites, typed with the product variables file.

## Files

| file | content |
|---|---|
| `amc_fieldGenetic.parquet` | stacked NEON table |
| `amc_fieldSuperParent.parquet` | stacked NEON table |
| `mic_swDnaPooling.parquet` | stacked NEON table |
| `mms_swMetagenomeDnaExtraction.parquet` | stacked NEON table |
| `mms_swMetagenomeSequencing.parquet` | stacked NEON table |
| `mms_swRawDataFiles.parquet` | stacked NEON table |

## Coverage

```json
{
 "amc_fieldGenetic": {
  "rows": 2540,
  "first": "2014-08-05",
  "last": "2026-09-09",
  "sites": 34,
  "releases": {
   "PROVISIONAL": 2241,
   "RELEASE-2026": 299
  }
 },
 "amc_fieldSuperParent": {
  "rows": 3125,
  "first": "2014-08-05",
  "last": "2026-09-09",
  "sites": 34,
  "releases": {
   "PROVISIONAL": 2812,
   "RELEASE-2026": 313
  }
 },
 "mic_swDnaPooling": {
  "rows": 54,
  "first": "2024-07-01",
  "last": "2025-09-29",
  "sites": 31,
  "releases": {
   "PROVISIONAL": 54
  }
 },
 "mms_swMetagenomeDnaExtraction": {
  "rows": 2519,
  "first": "2014-08-05",
  "last": "2026-06-29",
  "sites": 34,
  "releases": {
   "PROVISIONAL": 2223,
   "RELEASE-2026": 296
  }
 },
 "mms_swMetagenomeSequencing": {
  "rows": 276,
  "first": "2014-08-05",
  "last": "2023-10-31",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 276
  }
 },
 "mms_swRawDataFiles": {
  "rows": 517,
  "first": "2014-08-05",
  "last": "2023-10-31",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 517
  }
 }
}
```

## QC applied

- Typed by variables file; IDs as strings.

## Known issues

- Staff identifiers (`*By`) dropped.
- Release/provisional flagged per row.
- Sequence data not downloaded; use NMDC (nmdc:sty-11-hht5sb92, PRJNA406977) via the raw-file identifiers here.

## Rebuild

```bash
python -B scripts/data/download_neon.py --products metagenome
python -B scripts/data/build_neon.py metagenome
```
