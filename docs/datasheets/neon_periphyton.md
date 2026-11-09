# Datasheet — neon_periphyton (Periphyton, seston, phytoplankton chemistry)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20163.001` — Periphyton, seston, and phytoplankton chemical properties (https://data.neonscience.org/data-products/DP1.20163.001; RELEASE-2026 https://doi.org/10.48443/szsc-hh32)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_periphyton/`

All basic-package tables of DP1.20163.001 for the 34 aquatic sites, typed with the product variables file.

## Files

| file | content |
|---|---|
| `alg_algaeDataPerSampleCompCNPS.parquet` | stacked NEON table |
| `alg_algaeDataPerSampleCompChl.parquet` | stacked NEON table |
| `alg_algaeExternalLabDataPerSample.parquet` | stacked NEON table |
| `alg_domainLabChemistry.parquet` | stacked NEON table |
| `alg_domainLabChemistryComp.parquet` | stacked NEON table |
| `alg_fieldData.parquet` | stacked NEON table |

## Coverage

```json
{
 "alg_algaeDataPerSampleCompCNPS": {
  "rows": 3843,
  "first": "2024-01-23",
  "last": "2026-06-02",
  "sites": 34,
  "releases": {
   "PROVISIONAL": 3843
  }
 },
 "alg_algaeDataPerSampleCompChl": {
  "rows": 1611,
  "first": "2024-01-23",
  "last": "2026-08-25",
  "sites": 34,
  "releases": {
   "PROVISIONAL": 1611
  }
 },
 "alg_algaeExternalLabDataPerSample": {
  "rows": 62478,
  "first": "2014-05-15",
  "last": "2023-11-21",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 62478
  }
 },
 "alg_domainLabChemistry": {
  "rows": 29135,
  "first": "2014-05-15",
  "last": "2023-11-21",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 29135
  }
 },
 "alg_domainLabChemistryComp": {
  "rows": 3434,
  "first": "2024-01-23",
  "last": "2026-09-02",
  "sites": 34,
  "releases": {
   "PROVISIONAL": 3434
  }
 },
 "alg_fieldData": {
  "rows": 9279,
  "first": "2014-05-15",
  "last": "2026-09-02",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 6910,
   "PROVISIONAL": 2369
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
python -B scripts/data/download_neon.py --products periphyton
python -B scripts/data/build_neon.py periphyton
```
