# Datasheet — `safeswim_drivers` (hydrometeorological drivers for D1 SafeSwim, Chicago)

| | |
|---|---|
| **Use** | D1 SafeSwim iteration features (`sentinel/models/downstream/safeswim.py::driver_features`) |
| **Location** | `$SENTINEL_DATA/processed/safeswim_drivers/` (Parquet/zstd); raw responses `raw/safeswim_drivers/` (gzip JSON / NDBC .txt.gz, < 20 MB); manifest `manifests/safeswim_drivers.json` |
| **Code** | client `sentinel/data/sources/safeswim_drivers.py` · downloader `scripts/data/download_safeswim_drivers.py` · tests `tests/test_safeswim_drivers.py` |
| **Retrieved** | 2026-10-25 (UTC; per-run timestamp in manifest `stats.retrieved_utc`), years 2005–2026 |
| **Licence** | US public domain (USGS, NOAA) |
| **Synthetic data** | none |

## Sources (verified 2026-10-25)

| Table | Source | Request | Rows / coverage |
|---|---|---|---|
| `usgs_discharge_daily.parquet` | USGS NWIS WaterServices daily values `https://waterservices.usgs.gov/nwis/dv/` (JSON) | `parameterCd=00060`, `statCd=00003` (daily mean, ft³/s), sites below | 05536000 NB Chicago R at Niles 7,966 d (2005-01-01 → 2026-10-23); 05536105 NB Chicago R at Albany Ave 4,856 d (→ 2018-04-18, **not used** as a feature: ends before validation); 05536290 Little Calumet R at South Holland 7,966 d; 05536890 Chicago Sanitary & Ship Canal near Lemont 7,962 d |
| `coops_water_level_daily.parquet` | NOAA CO-OPS data API `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter` | `product=daily_mean`, station 9087044 Calumet Harbor, `datum=IGLD`, metric, LST, one request per year | 7,942 days, 2005–2026 |
| `ndbc_daily.parquet` | NOAA NDBC historical stdmet `https://www.ndbc.noaa.gov/data/historical/stdmet/<id>h<year>.txt.gz` | buoys 45007 (southern Lake Michigan, offshore; 2005–2025, seasonal ~Apr–Dec, 2020–21 from July only) and 45174 (Wilmette nearshore; 2015-08 → 2025, seasonal ~May–Nov) | 6,293 station-days (local-date daily summaries of hourly data) |

Discovery: the Chicago-area discharge gauges were listed with the NWIS site service
(`bBox=-88.0,41.6,-87.5,42.2&parameterCd=00060&hasDataTypeCd=dv`); 05536101 North Shore Channel at
Wilmette, 05536123 Chicago R at Columbus Dr, 05536118 and 05536358 have no daily discharge after
2010 and were not downloaded. No SENTINEL dataset held discharge for these gauges (`usgs_dv`
catalog = water-quality series only).

## Processing and QC

- **USGS:** WaterML JSON → `site_no` (string, leading zero), `date`, `discharge_cfs`, `qualifiers` (A,
  A e, P, Ice, …). The no-data value (−999999) → NaN. Qualifier counts: A 26,579, A,e 1,844, P 327.
- **CO-OPS:** `water_level_m_igld` (IGLD85 m) with the published flags string; empty values → NaN.
- **NDBC:** NDBC missing codes per column (WDIR/MWD 999, WSPD/GST/WVHT/DPD/APD/VIS/TIDE 99, PRES 9999,
  ATMP/WTMP/DEWP 999) → NaN (counts per station-year in manifest `stats.ndbc.n_masked_codes`; recent
  years have many masked wave rows — 10-min wind records without waves). Pre-2007 headers (`WD`, `BAR`,
  no minute column) are aliased. Daily summaries over the **local (America/Chicago) calendar day**:
  mean/max wave height (m), mean wind speed (m/s), mean onshore wind component `WSPD·sin(WDIR)`
  (m/s, positive = wind from the east, i.e. onshore at Chicago's west shore), mean water temperature
  (°C), number of hourly records.

## Decision-time rule (SafeSwim)

All drivers enter SafeSwim as values of day **d − 1** (or windows ending d − 1), for the morning
forecast and for the afternoon nowcast alike. Real-time versions of all three sources exist operationally
(NWIS instantaneous values, CO-OPS 6-min water levels, NDBC real-time feeds), so a d − 1 daily summary
is available by the morning of d; this is conservative for the nowcast. Approved vs provisional USGS values
are not distinguished in v1 (retrospective values are mostly approved; the operational provisional value
may differ slightly — a known optimism of a few percent in discharge, documented here).

## Known issues

- Buoys are seasonal: 45007 is missing in May–June 2020–2021 and 2015 starts in May; 45174 starts 2015.
  Missing days stay NaN (HistGradientBoosting handles NaN; the logistic learner imputes the training
  median with an indicator).
- 45007 is ~70 km offshore; nearshore waves at a beach differ. 45174 is nearshore at Wilmette (north).
- Discharge does not capture lock openings / CSO reversals to the lake directly (MWRD/USACE event logs are
  not in this dataset); canal flow at Lemont is a storm-diversion proxy only.

## Rebuild

```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_safeswim_drivers.py --start-year 2005 --end-year 2026
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_safeswim_drivers.py
```
About 70 requests (4 USGS, 22 CO-OPS, ~44 NDBC), ≥ 1 s apart; ~3 minutes.
