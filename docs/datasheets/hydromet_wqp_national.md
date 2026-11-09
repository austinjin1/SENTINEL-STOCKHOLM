# Datasheet — `hydromet_wqp_national` (M6 track N: national unsensed reaches, WQP discrete samples)

**Purpose.** Evaluation-only set for SENTINEL-HydroMet track N (protocol §12): stream water temperature and
dissolved oxygen grab samples at stations **without a continuous sensor**, plus the inputs M6 models need there.
No row is ever used for training.

**Files** (`$SENTINEL_DATA/processed/hydromet_wqp_national/`, manifest `manifests/hydromet_wqp_national.json`)
- `stations.parquet`: WQP `site_id`, `station_key` (`WQP:<site_id>`), organization, provider, state, `huc8`, `lat`/`lon`
  (WQP standardised, NAD83), site type/class, `comid` (NHDPlus V2 catchment containing the point) and the exclusion
  flags used.
- `obs_daily.parquet`: `site_id`, `date` (local activity date), `target` (`temp_c_mean` °C / `do_mgl_mean` mg/L),
  `value` (mean of the station-date's accepted samples), `n_samples`. Dates from 2021-10-01 only.
- `eligibility.parquet`: station × target × block with ≥ 8 sample days (`n_days`).
- `streamcat_comid.parquet`: curated StreamCat metrics of the station COMIDs (same groups as `processed/catchment`)
  + NHDPlus V2 VAA slope/stream order/drainage area/length; `streamcat_found`.
- `station_cells.parquet`: nearest valid gridMET cell (≤ 3 cells) per station; `forcings.parquet`: daily gridMET at
  that cell, 2005-10-01 → 2025-12-31, `forcings_daily` units.
- Interim: `$SENTINEL_DATA/interim/hydromet_wqp_national/` (simplified catchments GeoParquet per VPU — also used by
  `hydromet_upstream_forcing` —, StreamCat cache, per-year forcing arrays, logs).

**Code:** `scripts/data/build_hydromet_wqp_national.py {catchments,stations,streamcat,forcing,manifest}`; roles and
scoring `sentinel/models/hydromet/track_n.py`, `scripts/eval/hydromet_track_n.py`.

## Sources and licence
- `processed/wqp_discrete` (Water Quality Portal, WQX 3.0, retrieved 2026-10-22/23; public domain; cite Read et al.
  2017 and the submitting organisations — `organization` column). Discrete rows only (that build stores no logger or
  continuous series).
- NHDPlus V2 simplified catchments (EPA S3 `NHDPlusV21/Data/Extensions/SimplifiedCatchments/`, 21 CONUS VPUs; public
  domain) for the point-in-catchment COMID.
- EPA StreamCat REST API (POST, 500 COMIDs per request, 4 s spacing, cached; public domain; Hill et al. 2016).
- gridMET yearly netCDF (CC0).

## Selection and QC (fixed before any model output on these stations)
- `site_class = stream`; states other than AK, HI, PR, GU, VI, AS, MP; variables `temp` (unit `deg C`) and `do`
  (`mg/L`); `status_class = accepted`; not censored; finite; activity type not a time-series composite or QC sample;
  depth NaN or ≤ 2 m.
- Block eligibility: ≥ 8 sample days in val (2021-11-12 … 2023-09-30) or test (2023-11-12 … 2025-09-30), i.e. after
  the 42-day embargo.
- Unsensed: excluded WQP `USGS-<no>` stations with any USGS DV water-quality series (573 of 16,220 candidates) and
  stations whose COMID holds a USGS DV water-quality site or a FINAL sonde-split site (1,725); 1,692 points fell in no
  CONUS catchment. Kept: 12,771 stations.
- Known issues: a grab sample is an instant, compared with a daily-mean prediction (diurnal range adds equal noise to
  every model); simplified catchment boundaries can assign a bank-side point to the neighbouring catchment; WQP
  coordinates of some stations are approximate (`horz_accuracy` not filtered); `elev_m` input uses StreamCat
  `elevcat` because WQP stations have no gauge datum.

## Coverage
See the manifest `stats` (station counts by target × block, providers, StreamCat match, gridMET match).

## Rebuild
```
python -B scripts/data/build_hydromet_wqp_national.py catchments
python -B scripts/data/build_hydromet_wqp_national.py stations
python -B scripts/data/build_hydromet_wqp_national.py streamcat
python -B scripts/data/build_hydromet_wqp_national.py forcing
python -B scripts/data/build_hydromet_wqp_national.py manifest
```
