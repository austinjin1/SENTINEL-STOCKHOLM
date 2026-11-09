# Datasheet — neon_eos (NEON elevation of surface water (DP1.20016.001))

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20016.001` — Elevation of surface water (https://data.neonscience.org/data-products/DP1.20016.001; RELEASE-2026 https://doi.org/10.48443/aja5-bs82)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_eos/`

HOR codes: 101 = upstream sensor set (S1), 102 = downstream (S2) in wadeable streams; 103 = buoy (lakes/rivers); 110/111/112 = lake inlet/outlet/littoral where present (see sensor_positions). Raw native-resolution values with all basic QF columns are kept as zstd Parquet in raw/neon.

## Files

| file | content |
|---|---|
| `hourly/<SITE>.parquet` | site, time_utc (UTC hour start), hor, ver, <var>_mean/_min/_max/_n, release, provisional |
| `index.parquet` | per site × hor × ver × var: first, last, n_hours, prov_hours |
| `sensor_positions.parquet` | NEON sensor_positions (HOR/VER → location, offsets) |
| `units.json / qc_counts.json` | units per variable; values seen / masked by QF |
| `daily/<SITE>.parquet` | site, date (local standard time), hor, ver, <var>_mean/_min/_max/_hours, provisional |

## Coverage

```json
{
 "wse_m": {
  "sites": 34,
  "series": 113,
  "hours": 4026085,
  "prov_hours": 629233,
  "first": "2016-06-29",
  "last": "2026-09-30"
 }
}
```

## QC applied

- Values with the variable's final QF == 1 (or science-review flag == 2) set to NaN before aggregation. QC counts: {"wse_m": {"n": 8821155, "masked_qf": 781007}}
- Then the SENTINEL plausible-range screen of the USGS daily tier (sonde_daily.PLAUSIBLE via the equivalent USGS code, e.g. pH 3–11.5, DO 0–25 mg/L) masks outliers and small negative turbidity/chlorophyll readings are clipped to 0 (counts in qc_counts.json).
- Hourly: mean/min/max/count of valid native values in each UTC hour.
- Daily: local standard time (fixed UTC offset, no DST; `utc_offset_h` in neon_sites); count-weighted mean of hourly means, min/max of hourly extremes; NaN when < 18 hours have data (`<var>_hours` kept).
- Units: {"wse_m": "meter"} — RFU and µg/L chlorophyll are separate variables; turbidity is FNU (YSI EXO).

## Known issues

- Sensor drift/fouling corrections are NEON's; residual spikes remain (apply SENTINEL plausibility/spike screens downstream).
- Provisional months (2025-07 onward for most sites) may change.
- Lake buoys are removed in winter (ice) — seasonal gaps by design.

## Rebuild

```bash
python -B scripts/data/download_neon.py --products eos
python -B scripts/data/build_neon.py eos --workers 4
```
