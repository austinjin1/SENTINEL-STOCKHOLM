# Datasheet — neon_sonde_daily (NEON water-quality sonde (DP1.20288.001) — daily)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20288.001` — Water quality (multisonde) (https://data.neonscience.org/data-products/DP1.20288.001; RELEASE-2026 https://doi.org/10.48443/15k3-aj25)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_sonde_daily/`

HOR codes: 101 = upstream sensor set (S1), 102 = downstream (S2) in wadeable streams; 103 = buoy (lakes/rivers); 110/111/112 = lake inlet/outlet/littoral where present (see sensor_positions). Raw native-resolution values with all basic QF columns are kept as zstd Parquet in raw/neon.

## Files

| file | content |
|---|---|
| `sites/<SITE>.parquet` | site, date (local standard time), hor, ver, <var>_mean/_min/_max/_hours, provisional |

## Coverage

```json
{
 "chl_rfu": {
  "sites": 10,
  "series": 10,
  "hours": 259180,
  "prov_hours": 79836,
  "first": "2021-09-22",
  "last": "2026-09-30"
 },
 "chl_ugl": {
  "sites": 34,
  "series": 62,
  "hours": 1388942,
  "prov_hours": 79723,
  "first": "2016-07-29",
  "last": "2026-09-30"
 },
 "do_mgl": {
  "sites": 34,
  "series": 68,
  "hours": 2965326,
  "prov_hours": 468120,
  "first": "2016-07-29",
  "last": "2026-09-30"
 },
 "do_sat_local_pct": {
  "sites": 34,
  "series": 61,
  "hours": 1863369,
  "prov_hours": 468151,
  "first": "2021-09-01",
  "last": "2026-09-30"
 },
 "do_sat_sealevel_pct": {
  "sites": 34,
  "series": 68,
  "hours": 3134715,
  "prov_hours": 468431,
  "first": "2016-07-29",
  "last": "2026-09-30"
 },
 "fdom_qsu": {
  "sites": 34,
  "series": 41,
  "hours": 1301764,
  "prov_hours": 245839,
  "first": "2017-10-12",
  "last": "2026-09-30"
 },
 "ph": {
  "sites": 34,
  "series": 68,
  "hours": 3067362,
  "prov_hours": 476972,
  "first": "2016-07-29",
  "last": "2026-09-30"
 },
 "sensor_depth_m": {
  "sites": 10,
  "series": 10,
  "hours": 367358,
  "prov_hours": 75136,
  "first": "2017-10-20",
  "last": "2026-09-30"
 },
 "spcond_uscm": {
  "sites": 34,
  "series": 68,
  "hours": 3085289,
  "prov_hours": 478164,
  "first": "2016-07-29",
  "last": "2026-09-30"
 },
 "turb_fnu": {
  "sites": 34,
  "series": 68,
  "hours": 2872031,
  "prov_hours": 478099,
  "first": "2016-07-29",
  "last": "2026-09-30"
 }
}
```

## QC applied

- Values with the variable's final QF == 1 (or science-review flag == 2) set to NaN before aggregation. QC counts: {"spcond_uscm": {"n": 181552051, "masked_qf": 17332080, "masked_implausible": 0, "clipped_to_zero": 0}, "do_mgl": {"n": 174125668, "masked_qf": 16562435, "masked_implausible": 5825, "clipped_to_zero": 0}, "do_sat_sealevel_pct": {"n": 174095220, "masked_qf": 8380369, "masked_implausible": 436, "clipped_to_zero": 0}, "do_sat_local_pct": {"n": 102748548, "masked_qf": 5822988, "masked_implausible": 650, "clipped_to_zero": 0}, "ph": {"n": 174573411, "masked_qf": 12815123, "masked_implausible": 71667, "clipped_to_zero": 0}, "chl_ugl": {"n": 75160150, "masked_qf": 14230401, "masked_implausible": 0, "clipped_to_zero": 0}, "chl_rfu": {"n": 3205451, "masked_qf": 196839, "masked_implausible": 0, "clipped_to_zero": 0}, "turb_fnu": {"n": 181062208, "masked_qf": 33282659, "masked_implausible": 0, "clipped_to_zero": 0}, "fdom_qsu": {"n": 90676108, "masked_qf": 29151399, "masked_implausible": 0, "clipped_to_zero": 0}, "sensor_depth_m": {"n": 4667387, "masked_qf": 345536}}
- Then the SENTINEL plausible-range screen of the USGS daily tier (sonde_daily.PLAUSIBLE via the equivalent USGS code, e.g. pH 3–11.5, DO 0–25 mg/L) masks outliers and small negative turbidity/chlorophyll readings are clipped to 0 (counts in qc_counts.json).
- Hourly: mean/min/max/count of valid native values in each UTC hour.
- Daily: local standard time (fixed UTC offset, no DST; `utc_offset_h` in neon_sites); count-weighted mean of hourly means, min/max of hourly extremes; NaN when < 18 hours have data (`<var>_hours` kept).
- Units: {"spcond_uscm": "microsiemensPerCentimeter", "do_mgl": "milligramsPerLiter", "do_sat_sealevel_pct": "percent", "do_sat_local_pct": "percent", "ph": "pH", "chl_ugl": "microgramsPerLiter", "chl_rfu": "relativeFluorescenceUnits", "turb_fnu": "formazinNephelometricUnit", "fdom_qsu": "quinineSulfateUnit", "sensor_depth_m": "meter"} — RFU and µg/L chlorophyll are separate variables; turbidity is FNU (YSI EXO).

## Known issues

- Sensor drift/fouling corrections are NEON's; residual spikes remain (apply SENTINEL plausibility/spike screens downstream).
- Provisional months (2025-07 onward for most sites) may change.
- Lake buoys are removed in winter (ice) — seasonal gaps by design.
- fDOM is often fully QF-failed for long periods at some sites (e.g. ARIK 2023-07).
- Early months (2016) at several sites are published with all values empty.
- `chl_ugl` (µg/L) and `chl_rfu` (RFU) come from the same sensor in different periods/sites; they are never merged.

## Rebuild

```bash
python -B scripts/data/download_neon.py --products sonde
python -B scripts/data/build_neon.py sonde --workers 4
```
