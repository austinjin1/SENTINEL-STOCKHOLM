# Datasheet — v1 microcystin benchmark with re-extracted Sentinel-2 features (`mc_benchmark_v2`)

| | |
|---|---|
| **Labels (v1, read-only)** | `$SENTINEL_LEGACY/data/processed/biology/noaa_habs/mc_fusion_dataset.parquet` (3,649 rows, 227 sites, 2016-04-26 → 2025-10-22; WQP microcystin, µg/L) joined positionally to `mc_download_targets.parquet` (`targets.reset_index(drop=True).iloc[idx]`, exactly as v1); the join is verified row by row (site id, date and value all equal) |
| **Imagery** | S2 L2A chips (`s2_chips.md`), store `$SENTINEL_DATA/processed/s2_chips/mc_benchmark/` (5,924 chips, 215 MB) |
| **Licences** | WQP public; Copernicus Sentinel data free, full and open; JRC GSW without restriction |
| **Built** | 2026-10-22 (manifest `mc_benchmark_v2`) |
| **Code** | `scripts/data/build_mc_benchmark_v2.py`, `sentinel/data/matchups.py`, `sentinel/data/chips.py` (`v1_definition_features`) |
| **Outputs** | `$SENTINEL_DATA/processed/mc_benchmark_v2/mc_benchmark_v2.parquet` (one row per v1 row, 286 columns), `candidates.parquet` (every row × acquisition considered), `qc_report.json` |

## What was fixed relative to v1 (`download_microcystin_s2.py`, docs/research/10 §0)
| v1 defect | v2 |
|---|---|
| BOA offset (baseline ≥ 04.00, from 2022-01-25) never applied — two radiometric eras mixed | offset decided per item (`offset_dn`, `offset_source`, XML-verified on MPC); harmonised DN |
| 20 m bands read as native 64 × 64 windows (1.28 km) vs 10 m bands (0.64 km) | all bands on one 10 m grid (640 m), nearest replication |
| no cloud / water mask; only scene cloud % | SCL + JRC layers and centre-window statistics |
| nominal nearest scene, no scene id / true overpass time | every acquisition ±7 d enumerated, item id + granule sensing time kept, nearest usable chosen |
| MPC only | Earth Search C1 first, MPC fallback |

## Method
1. All S2 L2A acquisitions covering each site within ±7 days of the sample date (v1's window) — 18,466
   (site, acquisition) candidates, 4,257 scenes.
2. SCL screening (centre pixel has data, ≥ 5 SCL-valid pixels in the centre 7 × 7); for each row the 2 nearest
   usable acquisitions (water-rich centre preferred, then |Δt|, then scene cloud) get full chips → 5,924 chips.
3. Per candidate: `qc_strict` = chip ok & ≥ 5 SCL-water pixels in the centre 3 × 3 & JRC water test & no radiometric
   flag; `qc_lenient` = chip ok & ≥ 5 SCL-water pixels in the centre 7 × 7 & no cloud/shadow/cirrus in the centre
   3 × 3 & no radiometric flag. Each v1 row keeps its best candidate (strict, else lenient, else any chip; then |Δt|).
4. Features: all chip index columns (water-masked centre medians `rw3_*`, `rw7_*`, valid-pixel medians `rv*`, SCL
   fractions, JRC statistics, AOT, geometry) plus the 26 v1 features recomputed on the corrected chip
   (`v2def_*`; same definitions — chip mean, centre-20 × 20 mean, std per band; NDVI/NDCI/NDWI/G:R) and a
   water-only variant (`v2def_water_*`). The v1 features (`v1_sf0..25`) are kept for comparison only.
5. Kept from v1 for traceability only: `v1_CHLA_unharmonised`, `v1_PC_unharmonised` (mixed µg/L, mg/m³ and RFU with
   sentinel codes — **never use as features**), `v1_cloud`. Label: `mc_ugL` (µg/L), `label_gt8` (> 8 µg/L, v1's
   threshold; US-EPA-style, not WHO).

## QC survival (`qc_report.json`)
| Stage | Rows (of 3,649) |
|---|---|
| ≥ 1 S2 acquisition within ±7 d | 3,649 |
| usable after SCL screening (chip extracted) | 3,512 (96.2 %) |
| **qc_lenient** | **2,395 (65.6 %)** |
| **qc_strict** | **2,114 (57.9 %)**, 124 sites |
| qc_strict within same day / ±1 d / ±3 d / ±7 d | 411 / 1,107 / 1,698 / 2,114 |
| qc_lenient within same day / ±1 d / ±3 d / ±7 d | 443 / 1,213 / 1,882 / 2,395 |
| positives (> 8 µg/L): all / strict / lenient | 286 / 160 / 171 |
| post-2022-01-25 rows: all / strict / lenient | 1,166 / 688 / 767 |

Chosen chips by source: Earth Search C1 2,035, MPC 1,477; by baseline 05.00 1,816, 02.12 535, 04.00 499,
05.10 416, 05.09 155, 05.11 54, 03.00 37. Radiometric flags among chosen chips: 123 `bright_water`.

## Radiometric checks
* **Era confound (data-QC diagnostic, site-grouped 5-fold CV, gradient-boosting classifier predicting pre/post
  2022-01-25 from the 26 features, 2,395 qc_lenient rows / 167 sites):** AUROC **0.990 with the v1 features**
  (reproducing docs/research/10's 0.989) vs **0.661 with the same 26 features recomputed on the corrected chips**.
  The residual 0.66 can reflect genuine differences between the periods (sites, seasons, sensors S2A/B, processing
  baselines) and is not by itself evidence of an offset error.
* Per (source, baseline) median clear-water B11 reflectance (offset check, `offset_group_check`): 0.005–0.014 for
  ES 05.00/05.09 and MPC 02.12/03.00/04.00/05.10 (n = 62–2,555 chips per group), 0.032–0.053 for the small ES
  05.10/05.11 groups (n = 17–69); no group shows the ±0.1 shift of a missing or doubled offset.

## Known issues
* Rows are site-dates from v1's target table (Florida and California programmes dominate; 40 % of rows share a
  same-day sample within 5 km — v1 audit W1); use spatial blocking (≥ 5 km) for any evaluation.
* Many sites are canals, small ponds or shore stations: only 124 of 227 sites have a strict-QC row.
* ±7 d is v1's window; same-day pairs are few (411 strict). Report the window used.

## Rebuild
```bash
python -B scripts/data/build_mc_benchmark_v2.py --read-workers 2   # planning/screening/chips cached; resume-safe
```
