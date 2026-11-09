# Datasheet — sonde_daily (SENTINEL-Sonde daily tier)

- **Source:** USGS NWIS daily values (legacy WaterServices DV; statistics min 00002, mean 00003, max 00001, median 00008), raw manifests `usgs_dv_raw`, `usgs_dv_raw_batches_median`.
- **Retrieved:** 2026-10-22 (requests covered 2000-10-01 … 2026-10-20). **License:** US public domain.
- **Sites:** every USGS site with a DV water-quality series (catalog of 6,482 sites; 5,500 with at least one valid value).
- **Build:** `python scripts/data/build_sonde_daily.py --rebuild` → `$SENTINEL_DATA/processed/sonde_daily/` (per-site parquet, index, splice log, QC counts).
- **QC rules** (`sentinel/data/sonde_daily.py`): estimated (`e`), incomplete-day (`[4]`) and censored (`<`, `>`) values masked; non-numeric codes (Ice, Eqp, Dis, Bkw, Dry, Ssn, Pmp, Mnt, ***) and −999999 → NaN at ingest; plausible-range masking per parameter; small negative fluorescence/turbidity readings clipped to 0; provisional flag kept per value (`<var>_prov`).
- **QC counts:** {'rows_in': 123231215, 'masked_qualifier': 2575640, 'masked_implausible': 153457, 'clipped_to_zero': 14521, 'rows_valid': 120378931}
- **Series splicing:** 3105 secondary-series decisions logged; 2373 used, 732 rejected for inconsistency on ≥ 30 overlapping days (source series id kept per value in `<var>_ts`).
- **Known issues:** daily statistics cannot resolve sub-daily events (H1 hypoxia ≥ 2 h requires the hourly/15-min tier); pH and many turbidity sites publish only min/max/median (mean and median are kept as separate variables); USGS `SuppressData` thresholds on some DO series may censor low values (research report 01); RFU vs µg/L and FNU vs NTU are separate variables.

## Coverage by variable

| var                   |   sites |   sites_2y |   median_days | first      | last       |   prov |
|:----------------------|--------:|-----------:|--------------:|:-----------|:-----------|-------:|
| temp_c_mean           |    4304 |       2891 |        1476   | 2000-10-01 | 2026-10-20 |  0.075 |
| temp_c_max            |    4038 |       2765 |        1626   | 2000-10-01 | 2026-10-20 |  0.076 |
| temp_c_min            |    4026 |       2758 |        1626   | 2000-10-01 | 2026-10-20 |  0.076 |
| discharge_cfs_mean    |    3297 |       3059 |        7493   | 2000-10-01 | 2026-10-20 |  0.038 |
| spcond_uscm_mean      |    2441 |       1696 |        1649   | 2000-10-01 | 2026-10-20 |  0.059 |
| spcond_uscm_min       |    2253 |       1602 |        1756   | 2000-10-01 | 2026-10-20 |  0.063 |
| spcond_uscm_max       |    2250 |       1599 |        1761.5 | 2000-10-01 | 2026-10-20 |  0.063 |
| do_mgl_mean           |    1379 |        857 |        1285   | 2000-10-01 | 2026-10-20 |  0.053 |
| do_mgl_max            |    1342 |        854 |        1395.5 | 2000-10-01 | 2026-10-20 |  0.056 |
| do_mgl_min            |    1341 |        855 |        1401   | 2000-10-01 | 2026-10-20 |  0.055 |
| gage_ft_mean          |    1257 |       1130 |        6502   | 2000-10-01 | 2026-10-20 |  0.04  |
| ph_max                |    1126 |        736 |        1435.5 | 2000-10-01 | 2026-10-20 |  0.044 |
| ph_min                |    1123 |        735 |        1445   | 2000-10-01 | 2026-10-20 |  0.043 |
| ph_median             |    1027 |        700 |        1585   | 2000-10-01 | 2026-10-20 |  0.049 |
| turb_fnu_max          |     958 |        660 |        1342.5 | 2000-10-01 | 2026-10-20 |  0.082 |
| turb_fnu_min          |     956 |        646 |        1303.5 | 2000-10-01 | 2026-10-20 |  0.082 |
| turb_fnu_mean         |     726 |        479 |        1263   | 2000-10-01 | 2026-10-20 |  0.079 |
| gage_ft_max           |     548 |        477 |        5347   | 2000-10-01 | 2026-10-20 |  0.017 |
| gage_ft_min           |     545 |        475 |        5356   | 2000-10-01 | 2026-10-20 |  0.016 |
| turb_fnu_median       |     463 |        311 |        1259   | 2000-10-01 | 2026-10-20 |  0.091 |
| temp_c_median         |     259 |        169 |        1484   | 2001-03-02 | 2026-10-20 |  0.075 |
| salinity_ppt_max      |     235 |        202 |        3096   | 2000-10-01 | 2026-10-20 |  0.046 |
| salinity_ppt_min      |     233 |        201 |        3089   | 2000-10-01 | 2026-10-20 |  0.044 |
| discharge_cfs_max     |     188 |        162 |        5391.5 | 2000-10-01 | 2026-10-20 |  0.008 |
| discharge_cfs_min     |     185 |        159 |        5391   | 2000-10-01 | 2026-10-20 |  0.007 |
| no3no2_mgl_mean       |     178 |        126 |        1238   | 2008-03-27 | 2026-10-20 |  0.156 |
| salinity_ppt_mean     |     173 |        149 |        3272   | 2000-10-01 | 2026-10-20 |  0.076 |
| ph_mean               |     161 |         56 |         367   | 2000-10-01 | 2026-10-20 |  0.015 |
| spcond_uscm_median    |     153 |        115 |        1515   | 2001-03-02 | 2026-10-20 |  0.092 |
| do_pctsat_mean        |     151 |         29 |          13   | 2000-10-01 | 2026-10-20 |  0.053 |
| no3no2_mgl_max        |     151 |        111 |        1254   | 2009-07-30 | 2026-10-20 |  0.171 |
| no3no2_mgl_min        |     149 |        109 |        1254   | 2009-07-30 | 2026-10-20 |  0.177 |
| do_pctsat_min         |     133 |         28 |          14   | 2000-10-01 | 2026-10-20 |  0.045 |
| do_pctsat_max         |     133 |         28 |          14   | 2000-10-01 | 2026-10-20 |  0.045 |
| discharge_tf_cfs_mean |     100 |         94 |        3513.5 | 2000-10-01 | 2026-10-20 |  0.109 |
| do_mgl_median         |      97 |         68 |        1350   | 2003-12-10 | 2026-10-20 |  0.114 |
| pc_rfu_mean           |      61 |         34 |         830   | 2011-05-18 | 2026-10-20 |  0.552 |
| pc_rfu_min            |      57 |         35 |         848   | 2011-05-18 | 2026-10-20 |  0.59  |
| pc_rfu_max            |      57 |         34 |         848   | 2011-05-18 | 2026-10-20 |  0.59  |
| chl_ugl_mean          |      47 |         19 |         621   | 2001-04-27 | 2026-10-20 |  0.465 |
| chl_rfu_mean          |      44 |         25 |         801   | 2017-11-09 | 2026-10-20 |  0.744 |
| chl_rfu_max           |      44 |         26 |         826.5 | 2016-10-04 | 2026-10-20 |  0.744 |
| chl_rfu_min           |      43 |         26 |         828   | 2016-10-04 | 2026-10-20 |  0.738 |
| fdom_qse_mean         |      42 |         29 |        1031   | 2010-09-27 | 2026-10-20 |  0.546 |
| pc_ugl_mean           |      41 |         11 |         389   | 2013-06-27 | 2026-10-20 |  0.349 |
| fdom_qse_max          |      41 |         29 |        1032   | 2010-09-27 | 2026-10-20 |  0.619 |
| fdom_qse_min          |      40 |         29 |        1039   | 2010-09-27 | 2026-10-20 |  0.61  |
| chl_ugl_max           |      40 |         16 |         597.5 | 2001-04-26 | 2026-10-20 |  0.491 |
| chl_ugl_min           |      39 |         16 |         621   | 2001-04-26 | 2026-10-20 |  0.504 |
| pc_ugl_max            |      37 |         11 |         390   | 2013-06-26 | 2026-10-20 |  0.387 |
| pc_ugl_min            |      37 |         11 |         390   | 2013-06-27 | 2026-10-20 |  0.387 |
| no3no2_mgl_median     |      29 |         23 |        1117   | 2012-03-22 | 2026-10-20 |  0.065 |
| salinity_ppt_median   |      16 |         15 |        3748   | 2003-09-11 | 2026-10-20 |  0.099 |
| turb_ntu_mean         |      13 |          1 |         407   | 2000-10-05 | 2013-09-29 |  0     |
| turb_ntu_min          |      13 |          3 |         604   | 2000-10-05 | 2007-04-29 |  0     |
| turb_ntu_max          |      12 |          3 |         605   | 2000-10-05 | 2007-04-29 |  0     |
| fdom_qse_median       |      11 |          8 |        1044   | 2019-10-08 | 2026-10-20 |  0.263 |
| gage_ft_median        |       8 |          1 |         198   | 2007-07-12 | 2026-10-20 |  0.018 |
| discharge_tf_cfs_min  |       7 |          6 |        1989   | 2005-10-01 | 2026-10-19 |  0.03  |
| discharge_tf_cfs_max  |       7 |          6 |        1985   | 2005-10-01 | 2026-10-19 |  0.03  |
| no3_mgl_min           |       4 |          4 |        2079.5 | 2011-03-18 | 2026-10-20 |  0.017 |
| no3_mgl_mean          |       4 |          4 |        2428.5 | 2011-03-18 | 2026-10-20 |  0.017 |
| no3_mgl_max           |       4 |          4 |        2428.5 | 2011-03-18 | 2026-10-20 |  0.017 |
| chl_rfu_median        |       3 |          2 |         990   | 2016-10-04 | 2026-10-20 |  0.333 |
| discharge_cfs_median  |       1 |          1 |        1116   | 2023-09-30 | 2026-10-20 |  0.145 |
| pc_ugl_median         |       1 |          0 |          74   | 2019-07-30 | 2019-11-17 |  0     |
| pc_rfu_median         |       1 |          0 |         195   | 2020-03-13 | 2020-10-14 |  0     |
| turb_ntu_median       |       1 |          0 |         659   | 2002-11-05 | 2005-09-15 |  0     |

## Approval-status refresh (promoted 2026-11-08)
WY2024-26 re-pulled from USGS (daily values 2023-09-30 < date <= 2026-11-08; 15-min for the core set) before the legacy WaterServices slow-down, built with refresh precedence inside the window only and pre-window rows copied verbatim. `processed/sonde_daily` -> `sonde_daily_refresh_20261108`, `processed/sonde_hourly` -> `sonde_hourly_refresh_20261108` (symlinks; old snapshots kept as `sonde_daily_20261022`, `sonde_hourly_20261024`; manifests list the superseded manifest hash). Identity check: 5500 daily and 749 hourly site tables identical on every date <= 2023-09-30 (`compare_vs_sonde_daily.json`, `compare_vs_sonde_hourly.json`), so no train/val value changes.
