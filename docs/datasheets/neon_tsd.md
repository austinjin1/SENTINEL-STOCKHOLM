# Datasheet — neon_tsd (NEON temperature at specific depth (lakes/rivers, DP1.20264.001))

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20264.001` — Temperature at specific depth in surface water (https://data.neonscience.org/data-products/DP1.20264.001; RELEASE-2026 https://doi.org/10.48443/vyqt-c157)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_tsd/`

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
 "temp_c": {
  "sites": 10,
  "series": 83,
  "hours": 2669181,
  "prov_hours": 559816,
  "first": "2017-08-19",
  "last": "2026-09-30"
 },
 "thermistor_depth_m": {
  "sites": 10,
  "series": 83,
  "hours": 2669386,
  "prov_hours": 560011,
  "first": "2017-08-19",
  "last": "2026-09-30"
 }
}
```

## QC applied

- Values with the variable's final QF == 1 (or science-review flag == 2) set to NaN before aggregation. QC counts: {"temp_c": {"n": 5594669, "masked_qf": 373760, "masked_implausible": 0, "clipped_to_zero": 0}, "thermistor_depth_m": {"n": 6285840, "masked_qf": 1064522}}
- Then the SENTINEL plausible-range screen of the USGS daily tier (sonde_daily.PLAUSIBLE via the equivalent USGS code, e.g. pH 3–11.5, DO 0–25 mg/L) masks outliers and small negative turbidity/chlorophyll readings are clipped to 0 (counts in qc_counts.json).
- Hourly: mean/min/max/count of valid native values in each UTC hour.
- Daily: local standard time (fixed UTC offset, no DST; `utc_offset_h` in neon_sites); count-weighted mean of hourly means, min/max of hourly extremes; NaN when < 18 hours have data (`<var>_hours` kept).
- Units: {"temp_c": "celsius", "thermistor_depth_m": "meter"} — RFU and µg/L chlorophyll are separate variables; turbidity is FNU (YSI EXO).

## Known issues

- Sensor drift/fouling corrections are NEON's; residual spikes remain (apply SENTINEL plausibility/spike screens downstream).
- Provisional months (2025-07 onward for most sites) may change.
- Lake buoys are removed in winter (ice) — seasonal gaps by design.

## Rebuild

```bash
python -B scripts/data/download_neon.py --products tsd
python -B scripts/data/build_neon.py tsd --workers 4
```
