# Datasheet — `forcings_daily` (daily meteorological forcings per site, 2000-01-01 → present)

**OBSERVED weather (gridMET analysis / ERA5 reanalysis).** For future days this is an **oracle**:
use it for past-weather inputs or for declared oracle upper bounds, never as "forecast" covariates.
Operational forecasts come from `nwp_archive`.

**Files**
- `$SENTINEL_DATA/processed/forcings_daily/forcings_daily.parquet` (one row group per site, sorted
  by `site_key, date`), `coverage.parquet`, `dictionary.csv` (manifest `manifests/forcings_daily.json`).
- Per-site raw files: `raw/gridmet/sites/<site_key>.parquet` (gridMET short names, file units;
  manifest `gridmet_sites`) and `raw/openmeteo_era5/sites/<site_key>.parquet` (harmonised + `om_*`
  source columns).
- Site → cell maps: `raw/gridmet/site_cells.parquet`, `raw/openmeteo_era5/era5_sites.parquet`.

**Code:** `sentinel/data/sources/{gridmet,openmeteo}.py` · `scripts/data/download_forcings.py {gridmet,era5,combine}`

## Sources
| Source | Sites | Access (verified 2026-10-22) | Licence |
|---|---|---|---|
| gridMET (Abatzoglou 2013), 1/24° ≈ 4 km, CONUS | sites whose nearest valid cell lies within 3 cells | bulk yearly netCDF `https://www.northwestknowledge.net/metdata/data/<var>_<year>.nc` (10 vars × 2000–2026 = 270 files, ~45–60 MB each, ~20 MB/s); download → extract → delete | public domain / CC0 |
| Open-Meteo archive API, `models=era5_seamless` (ERA5-Land 0.1° where available, else ERA5 0.25°) | AK, HI, PR, GU and coastal sites outside the gridMET land mask | `https://archive-api.open-meteo.com/v1/archive`, daily, `timezone` = site standard offset (`Etc/GMT±N`), `cell_selection=land` | CC BY 4.0 (Copernicus ERA5 / Open-Meteo) |

Rejected method: the gridMET THREDDS NCSS point service. Its CSV output returns **packed
integers without `scale_factor`** (precipitation `215` for 21.5 mm on 2024-01-08 at the Illinois
River). One request per site and variable would also make the server decompress about 160 chunks.

## Variables and units (harmonised table; see `dictionary.csv`)
| column | unit | gridMET | ERA5 (Open-Meteo) |
|---|---|---|---|
| precip_mm | mm/day | `pr` | `precipitation_sum` |
| tmax_c, tmin_c | °C | `tmmx`, `tmmn` (K − 273.15) | `temperature_2m_max/min` |
| srad_wm2 | W/m², daily mean | `srad` | `shortwave_radiation_sum` (MJ/m²) × 10⁶/86400 |
| wind_ms | m/s, 10 m, daily mean | `vs` | `wind_speed_10m_mean` (`wind_speed_unit=ms`) |
| rhmax_pct, rhmin_pct | % | `rmax`, `rmin` | `relative_humidity_2m_max/min` |
| sph_kgkg | kg/kg | `sph` | **derived**: 0.622e/(p − 0.378e); e = e°(dew point); p = FAO-56 eq. 7 at the ERA5 cell elevation |
| vpd_kpa | kPa, daily mean | `vpd` | **derived** FAO-56: (e°(Tmax) + e°(Tmin))/2 − e°(Tdew), floored at 0 |
| pet_mm | mm/day | `pet` (ASCE grass reference ET) | `et0_fao_evapotranspiration` (FAO-56 grass reference) |

Further columns:
- `source` ∈ {`gridmet`, `era5_seamless`}. Sources are never mixed within a site.
- `status`: gridMET `permanent` / `provisional` / `early`, taken from the file attributes
  `last_permanent_slice` / `last_provisional_slice`. For a day, the most provisional status of
  any variable is used. ERA5 rows have `era5` (the final days are ERA5T and may be revised).

Day definitions:
- gridMET: "calendar days ending at midnight MST (07 UTC the next day)" (file `note5`).
- ERA5: the site's local-standard-time calendar day.
- Open-Meteo stamps hourly sums at the end of each interval, so ERA5 daily sums cover about
  [23:00, 23:00) local.

## Site → cell
- gridMET: the nearest cell centre. If that cell is masked (water/outside), the nearest valid cell
  within a 7 × 7 window (r = 3 cells ≈ 12 km); `cell_dist_km` is stored.
  - 6,226 of 6,516 sites map to 5,101 unique cells.
  - Median offset 1.6 km, max 3.02 km; only one site is more than 3 km away.
- ERA5: 290 sites (AK 158, HI 119, PR 5, GU 4, FL coastal/ocean 4) are grouped into 178 request
  groups: a 0.1° box plus the time zone, with one representative point. The grid point Open-Meteo
  actually used (`om_lat/om_lon/om_elev_m`) is stored.

## Coverage (build of 2026-10-22 10:25 UTC)
- `forcings_daily.parquet`: **60,952,540 rows** (1.05 GB), 6,226 sites × 2000-01-01 … 2026-10-20
  (9,790 days), complete for every gridMET site (median completeness 1.0).
- gridMET status in 2026: 233 permanent days (to 2026-08-21), 54 provisional, 6 early.
- After the 12:56 UTC refresh: 61,004,643 rows. 97 of the 290 ERA5 sites have 2025-01-01 …
  2026-10-15 so far (115 of 1,463 priority-0 box-years). The rest arrive over about 14 days at
  8,000 calls/day; the job pauses itself at the budget cap.
- The 290 ERA5 sites had no rows in the first build. They fill in as the budgeted ERA5 job progresses (newest
  benchmark years first). Re-run `combine` to refresh, or let the ERA5 chain run it at the end.
  `coverage.parquet` lists the per-site span and completeness.
- Plausibility screen: 0 negative precipitation, RH, radiation, wind or PET values; **233 rows with
  Tmax < Tmin**, a known rare gridMET artefact (kept, not altered).

## Open-Meteo budget (why ERA5 takes days)
- A 9-variable daily request costs 0.9 × days/14 calls, i.e. ~23.5 calls per box-year and ~630
  per box for 2000–2026.
- The 178 groups need ~112k calls. The free tier allows 10,000 per day per IP; our ledger caps us
  at 8,000 per rolling 24 h, shared with `nwp_archive`.
- Requests run in priority order:
  1. benchmark years (≥ 2007) overlapping the group's WQ data: ~34k calls;
  2. other years ≥ 2007: ~48.7k;
  3. pre-2007 years overlapping WQ data: ~8.9k;
  4. everything else: ~20.4k.
- Per-site ERA5 files are refreshed before every budget sleep, so `combine` can run at any time
  and reports the partial coverage in `coverage.parquet`.
- Alternatives considered:
  - Daymet single-pixel covers AK/HI/PR at 1 km but only through 2025 (2026 arrives in 2027).
    That would change the source inside Test-B.
  - NASA POWER is too coarse (0.5°).
  - CDS ERA5 requires an account.

## QC
- gridMET packing (`scale_factor`, `add_offset`, `_FillValue`, `_Unsigned`) is applied by netCDF4.
  The extraction refuses a file whose units differ from the expected ones. Units seen are stored
  in `raw/gridmet/gridmet_stats.json`.
- `combine` counts implausible values without altering them: precipitation < 0, Tmax < Tmin,
  RH > 100.5 %, RHmin > RHmax, negative radiation/wind/PET, |T| outliers. Counts are in
  `manifests/forcings_daily.json` → `stats.plausibility_counts`.
- Unit-tested (offline, with real fixtures): packed 215 → 21.5 mm, fill → NaN, masked-cell search,
  status flags, K → °C, MJ → W/m², FAO-56 helper values.

## Known issues
- gridMET's last ~2 months are provisional/early and will change on re-download. Use `status`;
  benchmark tests should use permanent values or re-pull after finalisation.
- A 4-km cell (gridMET) or a 9–28 km cell (ERA5) is not the stream. Steep or coastal sites
  (Hawaii especially) inherit large representativeness errors. ERA5-Land has no precipitation,
  radiation or wind at some island cells, so `era5_seamless` fills those from 0.25° ERA5 (seen
  at Honolulu).
- `sph_kgkg` and `vpd_kpa` for ERA5 rows are derived quantities (see the table), not reanalysis
  output.

## Rebuild
```bash
PY="/home/bcheng/miniconda3/envs/sentinel2/bin/python -B"
$PY scripts/data/download_forcings.py gridmet --start-year 2000   # ~35 min; resumable per variable-year
$PY scripts/data/download_forcings.py era5 --day-cap 8000          # multi-day; resumable; sleeps on budget
$PY scripts/data/download_forcings.py combine                      # harmonised table + coverage + manifest
```
