# Datasheet — `nwp_archive` (archived operational NWP forecasts, Open-Meteo Previous Runs)

**Use:** covariates for the **Operational** forecast mode only (docs/research/01 d.1). Never mix these
with observed or reanalysis weather (`forcings_daily` = oracle when used for future days).
**Files:** `$SENTINEL_DATA/processed/nwp_archive/{nwp_daily,coverage}_<tag>.parquet` where `<tag>` is
`pilot` (default) or `core` with `--sites-file` (manifest `manifests/nwp_archive_<tag>.json`); hourly chunks in `raw/openmeteo_nwp/hourly/<model>/<site_key>/`;
gzip-JSON response cache in `raw/openmeteo_nwp/cache/`; probes in `raw/openmeteo_nwp/{availability,lead_semantics}.json`.
**Code:** `sentinel/data/sources/openmeteo.py` · `scripts/data/download_nwp_archive.py`
**Retrieved:** from 2026-10-22, still running (see *Status*).

## Source and licence
`https://previous-runs-api.open-meteo.com/v1/forecast` (free tier, non-commercial, no key; data CC BY 4.0;
GFS = NOAA public domain; ECMWF IFS open data CC BY 4.0). Requested hourly variables:
`temperature_2m`, `precipitation` and `shortwave_radiation` at `_previous_day1` … `_previous_day7`, `timezone=GMT`.
Models: `gfs_global` (NCEP GFS 0.11°/0.25°; **not** `gfs_seamless`, which blends HRRR for the first
two days) and `ecmwf_ifs025` (ECMWF IFS 0.25° open data).

## What `previous_dayN` means (verified)
- Open-Meteo documents `<var>_previous_dayN` as "the value predicted 24·N hours before valid time".
- We checked this against the Single Runs API for valid day 2026-10-18 (Illinois River at Florence,
  00/06/12/18 UTC). At all 8 valid hours (both models), `previous_day1` equals the run initialised
  exactly 24 h earlier. The runs initialised 18, 30 and 36 h earlier differ, apart from one GFS tie.
  See `lead_semantics.json`.
- The Single Runs API keeps only recent runs (a 2025-06 run → "not available"). Older periods
  therefore rely on the documented construction.

## Daily aggregation and the issue-time bound
- Hourly values are grouped into **local standard-time days** using the site's fixed `utc_offset_h`
  (no DST, matching USGS DV days). Aggregates: `tmax_c`, `tmin_c`, `tmean_c` (2 m temperature),
  `precip_mm` (sum), `srad_wm2` (mean shortwave flux, W/m²).
- A day is aggregated only if all 24 hourly values exist; otherwise the value is NaN.
  `n_hours_t/p/s` are kept.
- Open-Meteo stamps precipitation and radiation at the end of the preceding hour. Their daily
  window is therefore [23:00 of L−1, 23:00 of L) local, one hour earlier than the calendar day.
  This keeps the bound below strict.
- `max_init_utc` is an upper bound on the initialisation time of every run that contributed to the
  row: `floor_6h(last valid hour − 24·N h)`.
  - With an issue time at local midnight starting day L − N + 1, every row satisfies
    `max_init_utc < issue` (unit-tested for UTC−5, −8, −10 and 0).
  - For the 00 UTC issue time of the SENTINEL-Sonde protocol, use a row only if
    `max_init_utc + latency ≤ issue`. Typical publication latency is ~4–5 h for GFS and ~7–8 h for
    IFS open data (not measured here; choose a conservative value).
  - Lead 1 issued at 00 UTC is therefore not available for US local days; use the next lead.

## Archive availability (first day with ≥ 20 of 24 hourly values at USGS-05586300; `availability.json`)
| Model | temperature_2m | precipitation | shortwave_radiation |
|---|---|---|---|
| gfs_global | 2021-03-24 (lead 7 from 2021-03-30) | 2024-01-20 (lead 7: 01-26) | 2024-01-20 (lead 7: 01-26) |
| ecmwf_ifs025 | 2024-02-04 (lead 7: 02-10) | 2024-02-04 (lead 7: 02-10) | 2024-03-07 (lead 7: 03-13) |

Consequences for the benchmark blocks (Train WY2008–2021, Val WY2022–2023, Test-A WY2024–2025, Test-B WY2026):
- **No operational precipitation or radiation forecast exists before 2024-01-20.** Validation
  (WY2022–2023) has GFS temperature only.
- Test-A lacks operational precipitation for 2023-10-01 … 2024-01-19 (GFS) and 2024-02-04 (IFS).
- Training years need another source: the GEFS v12 reforecast on AWS (2000–2019), NOAA GFS archives
  on AWS/NCEI, or the neon4cast GEFS drivers for NEON sites. None of these is downloaded here.
- Isolated gaps exist; for example, GFS temperature is null on 2023-12-30 … 2024-01-01 at the
  probe site. The per-site, per-lead `coverage.parquet` reports the real completeness.

## Pilot sites (`raw/openmeteo_nwp/sites_pilot.csv`)
50 CONUS USGS stream sites (`site_tp_cd = ST`) were chosen deterministically. Each has DO (00300),
SC (00095) and temperature (00010) DV series overlapping for ≥ 5 years and still active in 2026;
324 sites qualified. Selection was round-robin over HUC2 regions (16 regions, 2–4 sites each),
longest overlap first. For the later core list use `--sites-file <csv|parquet with site_key>`.

## Cost and rate limits (measured 2026-10-22)
- Free-tier limits per IP: 600 calls/min, 5,000/h, 10,000/day.
- One request costs `max(1, max(n, t·n)) · locations` calls, with `n = variables·models/10` and
  `t = days/14`. This is the formula of the Open-Meteo pricing-page calculator and reproduces its
  examples (unit-tested).
- Hourly and daily variables cost the same per day.
- A site-year with 21 variables (3 × 7 leads) costs 2.1 × 365/14 ≈ 55 calls.
- Plan for the pilot: 750 calendar-year chunk requests, **17,462 calls in total = 349 per site**.
  - GFS: 202 per site (temperature from 2021-03-24; precipitation/radiation from 2024-01-20).
  - IFS: about 147 per site.
- Our jobs share a cross-process budget ledger (`raw/openmeteo/budget_ledger.jsonl`). It caps them
  at 8,000 calls per rolling 24 h, 4,000 per hour and 400 per minute, leaving headroom for other
  SENTINEL jobs on the same IP.
- Measured throughput: about 20 chunk requests (≈ 440 calls) per minute until the minute cap binds.
  The daily cap is used up in about 2 h; the job then sleeps until the rolling window frees.
- Feasibility:
  - 50 pilot sites × 2 models ≈ 2.2 days at 8,000/day.
  - A ~600-site core list ≈ 210k calls, i.e. about 26 days at 8,000/day or 21 days at the full
    10,000/day.
  - Faster options: the paid customer API (no daily limit), or self-hosting the Open-Meteo archive
    from its AWS open-data bucket.
- 429 responses: on a "Minutely/Hourly" limit the client sleeps 65 s / 1 h; on "Daily" it sleeps
  until the ledger window has room. Every response is cached, so the job resumes at the next
  missing chunk.

## Status
2026-10-22 12:50 UTC: 246/750 chunks downloaded (GFS for 28 sites). The 8,000-call ledger window is
used up, and the job sleeps and resumes automatically.
- On data already in hand: precipitation/radiation have ~1,004 days per lead since 2024-01-20 with
  99.9% completeness; temperature has ~2,014 days since 2021-03 with 98.9% completeness.
- Expected completion: ~2–3 days with the IFS model, sharing the budget with ERA5.

See `raw/openmeteo_nwp/stats_pilot.json` and `raw/openmeteo/chain_stdout.log`. The background
chain is: GFS pilot → ERA5 priority-0 forcings → IFS pilot → aggregate → remaining ERA5 → forcings
combine. The aggregate step can be re-run at any time on the chunks downloaded so far.

## Rebuild
```bash
# probe availability + semantics (≈ 160 calls), then download, then aggregate (all resumable)
python -B scripts/data/download_nwp_archive.py all --pilot 50 --models gfs_global,ecmwf_ifs025
# core list later:
python -B scripts/data/download_nwp_archive.py all --sites-file core_sites.csv --models gfs_global,ecmwf_ifs025
python -B scripts/data/download_nwp_archive.py aggregate --pilot 50   # re-aggregate cached chunks
```
