# Datasheet — AquaCorr-Global v1.0 (`aquacorr_global`) + SENTINEL S2 L2A chips (`s2_chips_aquacorr`)

| | |
|---|---|
| **Source** | Ghazi Zadeh Hashemi, M.S., Bolten, J., Kumar, S., Seegers, B. (2026) *AquaCorr-Global: A Multi-Sensor, Multi-Decadal Water Quality Matchup Dataset for Inland and Coastal Waters*, v1.0, Zenodo, published 2026-08-14. Record doi:10.5281/zenodo.21927415 (concept doi:10.5281/zenodo.21927414). **Not peer-reviewed.** |
| **URL** | https://zenodo.org/records/21927415 (API `https://zenodo.org/api/records/21927415`) |
| **Licence** | CC BY 4.0 (Zenodo record) |
| **Retrieved** | 2026-10-22; md5 of every file verified against the Zenodo API (manifest `aquacorr_global_raw`) |
| **Files kept** | `aquacorr_matchups.parquet` (377,032,221 B), `aquacorr_metadata.json`, `aquacorr_gloria_validation.parquet`, `aquacorr_rrs_spectra.zarr.zip` (88.6 MB). The 505 MB `csv.gz` duplicates the parquet and was not downloaded. |
| **Local paths** | raw `$SENTINEL_DATA/raw/aquacorr_global/`; processed `$SENTINEL_DATA/processed/aquacorr_global/`; chips `$SENTINEL_DATA/processed/s2_chips/aquacorr/` |
| **Code** | `sentinel/data/sources/aquacorr.py`; `scripts/data/download_aquacorr.py`, `scripts/data/build_aquacorr.py`, `scripts/data/build_aquacorr_s2_chips.py` |
| **Manifests** | `aquacorr_global_raw`, `aquacorr_global`, `s2_chips_aquacorr` |

## What it is
1,984,387 same-calendar-day matchups (UTC date of the overpass) between harmonised in-situ records and ACOLITE-DSF
"Rrs" (3 × 3 window at the site; 10 m for S2, 30 m Landsat) for Landsat 5/7/8/9 and Sentinel-2A/B, 1984–2024,
202,029 sites. Parameters: chl-a, TSS, Secchi (`sdd`), turbidity, aCDOM(440), DOC, TOC. Sources: AquaMatch (WQP
harmonised), ChemLotUS, Caravan-Qual, AquaSat v1, GLORIA, SeaBASS. S2 rows carry the ACOLITE scene id with the
**MGRS tile and granule sensing time** (`S2B_MSI_2019_06_14_15_41_22_T19TCH_L2R`), which we use to fetch the
same acquisition from L2A.

## Tidy table (`processed/aquacorr_global/aquacorr_matchups.parquet`, 254 columns)
All original columns except the WKB geometry (lat/lon kept), plus: decoded source flags `<p>_method_class`,
`<p>_quality_tier`, `<p>_fraction`, `<p>_depth_type`, `<p>_outlier`, `<p>_recommended` for chla/sdd/tss/turbidity/toc
(bit layout from the record: bits 0-1 method, 2-3 tier, 4-5 fraction, 6-7 depth type, 8 outlier, 9 recommended;
tested); `<p>_unit` (**assumed** units — see issues); `region` (coarse bounding boxes: US_conus / US_alaska /
US_hawaii / US_puerto_rico / Europe / other; CONUS box includes border strips of Canada/Mexico); `mgrs_tile`,
`scene_sensing_time` (S2, parsed from the scene id); duplicate structure `acq_group_n_scenes`,
`colocated_group_n_rows`; spectral plausibility `rrs_blue`, `rrs_swir1`, `flag_swir_gt_0p01`, `flag_blue_gt_0p1`;
`matchup_window_days = 0`.

## Audit findings (computed by `build_aquacorr.py` → `audit.json`)
**Composition.** Sensors: L7 663,844; L5 467,730; L8 292,038; S2A 280,036; S2B 242,464; L9 38,275. Sources: AquaMatch
622,894; ChemLotUS 603,484; Caravan-Qual 367,129; AquaSat_v1 300,276; AquaMatch+AquaSat_v1 85,825; AquaMatch+ChemLotUS
3,129; GLORIA 1,535; SeaBASS 115. The undocumented provenance letter `quality_tier` maps one-to-one onto sources
(A = GLORIA, B = SeaBASS, C/D = AquaMatch, D = AquaSat_v1, E = ChemLotUS/Caravan-Qual + some AquaMatch).
**Geography:** US (CONUS box) 1,622,857 rows / 171,258 sites; Europe 323,732 / 25,107; elsewhere 37,526 / 5,606
(so ≈ 82 % US by rows). S2 rows: US 372,807, Europe 135,880, other 13,813.

**Duplicates.** No exact duplicate rows and no duplicate (site, date, sensor, scene). But **503,749 rows (25 %) repeat
an in-situ sample for another scene of the same overpass** (overlapping MGRS tiles / WRS rows; 976,682 rows belong to
such multi-scene acquisitions) — these are not independent matchups and must be collapsed or kept in one split
group. **187,497 rows share coordinates (1e-5°), date and scene with a row of another `site_id`** (cross-source
duplicates, mostly AquaSat_v1/AquaMatch/ChemLotUS); only 541 chl-a / 230 TSS / 862 Secchi / 17 turbidity of these
also share the value, i.e. most are the same station under two ids carrying different variables.

**Matchup window.** `date_insitu == date_satellite` for 100 % of rows and `date_satellite` is the UTC date of
`overpass_utc` (100 %); no in-situ time of day is given, so "same day" means up to ±~12 h. For sites east of 140°E
(3,618 rows) the overpass is ~22–23 UTC, i.e. the UTC date is the local date − 1. S2 scene-id times equal
`overpass_utc` exactly.

**Chl-a method.** 616,327 chl-a values: method class unknown 485,553 (79 %), extraction/lab 128,458 (21 %),
in vivo/sensor 2,316. `recommended` = 138,398 (all lab rows + 9,940 unknown-method rows; in vivo never). The counts in
`aquacorr_metadata.json` do **not** reproduce from the published flags (metadata: extraction/lab 137,815, recommended
147,929, "method_labeled" 551,602; flags: 128,458 / 138,398 / 130,774 non-unknown). TSS: 377,163 lab (= recommended);
Secchi: method always unknown (recommended 162); turbidity: in vivo/sensor 296,866, never recommended. Replicates:
4,258 chl-a values are same-day medians.

**QC flags.** Overall `passed_qc` 47.2 % (S2 60.6 %): `flag_insufficient_pixels` 32.9 % (S2 15.5 %), `flag_acolite`
31.7 % (S2 15.0 %), `flag_high_cv` 20.4 % (S2 23.8 %), `flag_negative_rrs` 0.3 %, `flag_high_sza` never set (SZA up to
70° passes). **`passed_qc` does not screen land, cloud or glint:** among 316,877 S2 rows with `passed_qc`, the median
"Rrs" is 0.065 (492 nm), 0.090 (665 nm), 0.277 (864 nm) and 0.181 sr⁻¹ (SWIR-1); 88.9 % have SWIR-1 > 0.01 sr⁻¹ and
41.5 % blue > 0.1 sr⁻¹ — values impossible for clean water-leaving reflectance (SWIR > 0.01 by source: ChemLotUS
98.9 %, Caravan-Qual 97.8 %, AquaMatch 77.0 %, GLORIA 31.6 %). Example: Lake Okeechobee centre, passed-QC rows with
Rrs(492) ≈ 0.38 and SWIR ≈ 0.23 sr⁻¹ (cloud). The record itself warns that river windows include shoreline pixels.

**Values / units.** Units are **not stated** in the published files (the column dictionary `aquacorr_variables.xlsx`
listed in `aquacorr_metadata.json` is not in the record, nor are `aquacorr_insitu_master.parquet` and
`aquacorr_landsr_comparison.parquet`; the metadata `doi` field is the placeholder `10.5281/zenodo.XXXXXXX`). Assumed
from source conventions and distributions: chl-a µg/L (median 5.6, max 5,000), TSS mg/L (median 8.5, max 50,000),
Secchi m (median 1.52, max 50), turbidity NTU-or-FNU (median 5.7, max 49,418; 322 zeros), aCDOM m⁻¹, DOC/TOC mg/L.
Coordinates: 98,165 rows use LAGOS lake centroids (`coord_centroid`, Landsat rows only; none in S2).

## SENTINEL L2A chips for AquaCorr S2 matchups
**Selection** (`s2_chip_requests.parquet`, seed 20261022): S2 rows with `passed_qc`, no centroid coordinates, US or
Europe, at least one of chl-a/TSS/Secchi/turbidity; one row per physical acquisition (site, date, sensor; the tile in
the site's own UTM zone kept, then most valid pixels); priority lab-recommended chl-a > Secchi > other chl-a >
TSS/turbidity with a Europe quota of 20,000; budget 120,000 rows → 119,079 chip requests (34,323 scenes, 2,718 dates;
values: chl-a 80,983 incl. 17,325 lab, Secchi 56,997, TSS 33,686, turbidity 20,346).

**Scene resolution** (`s2_chip_plan.parquet`): 117,068 of 119,079 requests resolved — Earth Search C1 72,042, MPC
45,026; 109,798 on the AquaCorr tile, **7,270 (6.1 %) whose point lies outside the AquaCorr-named tile's raster**
(an AquaCorr defect; another tile of the same acquisition — same platform, date and relative orbit — is used,
`match_kind = alternate_tile`), 2,011 (1.7 %) with no L2A item in either catalogue.

**Extraction status (2026-10-24, interim):** 47,278 chips extracted (16,225 sites, 7,771 scenes, all US — the
selection is processed in priority order, US before Europe), 7,317 requests failed with read timeouts (retryable),
65,375 not yet extracted. Resume with the rebuild command below (`--retry-failed` redoes the timeouts).

**L2A cross-check of AquaCorr `passed_qc` rows (same acquisition, n = 47,278 extracted):**
* Sen2Cor SCL marks cloud/cirrus in the centre 3 × 3 for **43.6 %** (cloud over the majority of the 3 × 3 for
  42.6 %); cloud shadow 3.7 %.
* ≥ 5 SCL-water pixels in the centre 3 × 3 (`qc_water_c3`): 38.1 %; JRC water test: 70.3 %; **`qc_pass` (both +
  no radiometric flag): 17,216 (36.4 %)** — chl-a 14,938 (3,499 lab-recommended), Secchi 13,594, TSS 1,586,
  turbidity 1,196.
* AquaCorr's SWIR-1 "Rrs" > 0.01 sr⁻¹ occurs in 99.0 % of rows that L2A marks cloudy, 98.4 % of rows failing our
  QC and 46.6 % of rows passing it: the implausible ACOLITE spectra are largely cloud/land contamination.
* 2,729 extracted rows use an alternate tile; 99.94 % are the same acquisition (platform, UTC date, orbit).
* Radiometric flags: 1,047 `bright_water`, 28 `double`. **Offset group check** (median clear-water B11):
  ES 05.00 0.0127 (n = 8,886), ES 05.09 0.0130, ES 05.10 0.0100, MPC 02.12 0.0102 (n = 3,951), MPC 03.00 0.0050,
  MPC 04.00 0.0135, MPC 05.10 0.0152 — no group shows a missing/double offset.
* Chips: 1.8 GB (≈ 39 kB/chip).

## Known issues / guidance
* Treat AquaCorr matchups as **candidate** pairs: screen with our L2A SCL/JRC columns (`qc_water_c3`,
  `qc_jrc_water`, `qc_pass`) or SWIR plausibility before using ACOLITE Rrs as a baseline input.
* Collapse multi-scene repeats and cross-source duplicates before splitting; keep multiple stations of one lake and
  date in one group.
* ES C1 times agree with ACOLITE scene times to < 1 s; MPC baseline-02.xx granule `SENSING_TIME` precedes them by
  up to ~10 min (metadata convention) — the acquisition identity is platform + UTC date + relative orbit + tile.

## Rebuild
```bash
python -B scripts/data/download_aquacorr.py                      # Zenodo files, md5-verified
python -B scripts/data/build_aquacorr.py --budget 120000 --europe-quota 20000   # tidy + audit + selection
RW=8 SW=6 processed/aquacorr_global/run_chips_loop.sh   # restart loop around:
#   build_aquacorr_s2_chips.py --stage all --read-workers 8 --scene-workers 6 [--retry-failed]  (resume-safe)
python -B scripts/data/build_aquacorr_s2_chips.py --stage join
```
