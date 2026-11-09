# Datasheet — neon_sites (NEON aquatic site table)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.30006.001` — DP1.30006.001 (https://data.neonscience.org/data-products/DP1.30006.001; RELEASE-2026 https://doi.org/10.48443/jhbd-n155)
  - `DP3.30006.001` — DP3.30006.001 (https://data.neonscience.org/data-products/DP3.30006.001; RELEASE-2026 https://doi.org/10.48443/8srr-nv51)
  - `DP1.30010.001` — DP1.30010.001 (https://data.neonscience.org/data-products/DP1.30010.001; RELEASE-2026 https://doi.org/10.48443/xv9v-1r95)
  - `DP3.30010.001` — DP3.30010.001 (https://data.neonscience.org/data-products/DP3.30010.001; RELEASE-2026 https://doi.org/10.48443/8vq2-s021)
  - `DP1.30003.001` — DP1.30003.001 (https://data.neonscience.org/data-products/DP1.30003.001; RELEASE-2026 https://doi.org/10.48443/5ts2-rc92)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_sites/`

One row per NEON aquatic site. Use `domain` for leave-one-domain-out splits. `station_lat/lon` are the downstream S2 station (streams) or buoy (lakes/rivers) from the SENTINEL site registry; `huc*`/`comid` come from the registry (WBD point query / NLDI position).

## Files

| file | content |
|---|---|
| `sites.parquet` | site, domain, aquatic_type/class, coordinates, tz, HUC2–12, COMID, n_months_/first_/last_<product> |
| `aop_availability.parquet` | AOP site-months per product (availability only) |

## Coverage

```json
{
 "sites": 34,
 "domains": 19,
 "aquatic_class": {
  "stream": 24,
  "lake": 7,
  "river": 3
 },
 "huc12_missing": 0
}
```

## QC applied

- HUC and COMID strings (leading zeros kept).

## Known issues

- HUC/COMID from point queries; lakes may sit on a reach boundary.
- AOP imagery not downloaded (availability only).

## Rebuild

```bash
python -B scripts/data/download_neon.py --site-meta --aop
python -B scripts/data/build_neon.py sites
```
