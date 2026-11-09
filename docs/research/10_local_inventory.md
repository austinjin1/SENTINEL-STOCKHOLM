# 10 — Local asset inventory: what is reusable for the v2 rebuild

**Date:** 2026-10-22
**Scope:** every dataset, script and checkpoint that exists *on this machine* for SENTINEL, with a verdict for v2. Sources: `/home/bcheng/STOCKHOLM/SENTINEL-HF` (HF mirror at `e977248`: code + `data/` + `checkpoints/` + `results/` + `logs/`), `/home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM` (GitHub clone, branch `v2-rebuild` = `main` @ `e18e312`), the audit `/home/bcheng/STOCKHOLM/REVIEW_REPORT.md` and its evidence in `/home/bcheng/STOCKHOLM/review_artifacts/`.
**Method:** read-only. Every dataset was opened with pandas / pyarrow / numpy (npz headers read from the zip without loading; the 4.2 GB chip archive was streamed chip by chip). All 76 `.pt` files were opened with `torch.load(weights_only=True)`; 19 of them need a numpy allowlist (`torch.serialization.safe_globals([numpy scalar, dtype, ndarray reconstruct])`, still `weights_only=True`, no arbitrary code). Tests ran in a scratch copy of the repo. Numbers marked **(measured)** were computed in this session; numbers marked **(audit)** come from `REVIEW_REPORT.md`.

**Verdict vocabulary.** **REUSE AS-IS**: usable now without changes. **REUSE RAW BUT REPROCESS**: the underlying observations are real and useful, but the stored form has a defect (bad normalization, missing provenance, mixed units, and so on), so rebuild the derived product. **DISCARD**: synthetic, circular, invalid, superseded, or so small or derived that re-pulling is cheaper than repairing it.

---

## 0. Summary

1. **There is almost no raw data on disk.** HF `data/` holds only *processed* products. Every `data/raw/...` path that the code reads is absent: the USGS IV parquet files (383 stations, 51.6 M records), the NEON DP1.20288.001 consolidated parquet (51.4 M rows), the EMP BIOM and mapping file, the ECOTOX ASCII dump, the GEO series matrices, the S2 tile caches (`drone_tiles/`, `mc_tiles/*.npz`, 78,734 "massive" tiles), the NHDPlus site cache and the WQP driver CSVs. `/home/bcheng/SENTINEL`, the target of the 51 hard-coded paths, is an **unrelated project on this machine** (TD-SSM, a cfDNA model). Nothing water-related lives there, and no copy exists anywhere else under `/home/bcheng` or `/shared`.
2. **The most valuable reusable assets:**
   - The microcystin fusion table and its row-aligned driver cache (`mc_fusion_dataset.parquet` + `_mc_dg_featurecache.parquet`; 3,649 rows, 227 sites). They reproduce the audit's "solid" result: satellite arm 0.650 **(measured)**.
   - The raw WQP pulls behind it: microcystin/cyanotoxins (30,097 rows), chlorophyll/phycocyanin (754,709 rows, 43,278 sites) and the expanded microcystin set (27,835 rows, 40 states, including 583 NARS NLA rows).
   - The 12,000 real Sentinel-2 chips with USGS targets (`drone_wq_pairs.npz`, 379 stations).
   - The raw FIB tables: E. coli for OH/MI/WI (182,578 rows) plus 1,346 GHCN-Daily precipitation files.
   - USGS BioData invertebrates and fish (384,885 + 16,242 rows).
   - The EMP taxonomic matrices (26,181 samples × 5 ranks).
   - KEGG zebrafish pathways.
   - The NHDPlus site→COMID map (561 sites).
   - The Track-A event manifests: 34 NEON sites, 39 USGS DO sites and 41 salinization sites, with onset dates.
3. **New defects found in this inventory (not in the audit):**
   - **The microcystin satellite features mix two radiometric baselines.** The S2 L2A BOA offset (PB ≥ 04.00, from 2022-01-25) is uncorrected in `download_microcystin_s2.py` / `expand_02_colocate_s2.py`. The median B02 mean is 0.049 before the switch and 0.146 after.
     - The era can be predicted from the 26 features with AUROC **0.989** (site-grouped CV).
     - Positives are more common before 2022 (9.5% vs 4.3% at > 8 µg/L).
     - Offset-correcting the 18 band statistics moves the satellite arm from 0.637 to 0.626 **(measured)**. So the confound is small but real, and the v2 features must be re-extracted.
   - **Band footprints are mismatched in the same downloader.** Each band is read as a 64×64 window on its *native* grid, with no resampling. The 20-m bands B05 and B11 therefore cover 1.28 km while B02/B03/B04/B08 cover 0.64 km, so per-pixel NDCI = (B05 − B04)/(B05 + B04) combines different areas (code inspection of `download_microcystin_s2.py` / `expand_02_colocate_s2.py`, lines 85–93).
   - **The HydroViT chips carry the same offset** in 11,814 of 12,000 chips (median per-chip minimum reflectance 0.105).
     - Only 1,899 chips (16%) have water (NDWI > 0) in the 5×5 pixels at the gauge, and only 3,881 (32%) have more than 5% water pixels anywhere in the 2.24-km chip **(measured)**.
     - The acquisition date, item ID and SCL mask were never stored.
     - Targets are the nearest daily USGS mean within ±5 d of a *nominal* 5/15/25-of-month date, not of the actual overpass.
   - **The pigment features mix units.** CHLA/PC merge lab µg/L, mg/m³ and probe RFU, and include sentinel codes (CHLA max 999,999; PC max 2,000,000). The WQP indicator table has 298 values ≥ 99,999 and 597 negative values.
   - **The FIB label filter is loose.** The `unit.contains("100")` test admits `CCE/100mL` (qPCR, 1,521 WI rows), and `"Water" in ActivityMediaName` admits groundwater rows.
   - **`wqp_biological.parquet` is silently truncated** to exactly 300,000 rows by `df.head()`, which leaves HUC02 with 0 rows.
   - **The checkpoints have extra problems.** The AquaSSM anomaly heads in `aquassm_anomaly_best.pt` and `aquassm_v2_anomaly_best.pt` contain NaN weights. The shipped `waterdronenet_pretrain_best.pt` (ViT-S/16, 21.9 M params, 224 px) does not match its log, which describes an inline model with `img_size=64` and 4,978,778 params.
4. **Checkpoints:** nothing on disk is a good initializer. Public weights (timm ImageNet ViT-S/16, which the only honest satellite result used, or a remote-sensing / time-series foundation model) dominate everything local. Two checkpoints are worth keeping as *reference baselines*: `microbiomenet_geographic_holdout_full_emp_best.pt` and `species_health_leakagefree_best.pt`. The real final models the audit names (`hydrovit_wq_real.pt`, `aquassm_neon_ft.pt`) are **not on disk anywhere**.
5. **Code:** about 25 scripts are worth carrying forward as templates. These are the clean downloaders, the FIB / microcystin / zebrafish LOSO / EMP geographic / species leak-free / twin climatology / PCMCI+ protocols, the Track-A event builders and `weighted_conformal.py`. They are listed in §3.1, together with the review's own re-analysis scripts (placebo, DOY+lat baseline, 5-km blocking). The rest (~150 k LOC) should be archived.
   - Tests: **33 passed, 1 failed** (`TestEJOverlay::test_import`, missing `h3`) under Python 3.13.2 / torch 2.7.1+cu118. The same failure occurs under the new `sentinel2` env (Python 3.11.17).
   - 17 third-party imports are declared in none of the three dependency files.
6. **Disk:** the STOCKHOLM tree uses 24.2 GB. Half of that is the HF `.git` LFS store, which duplicates the working tree. Free space was 369 GB at the start of this session and **359 GB at the end** (other processes are writing).

---

## 1. Disk footprint

| Location | Size | Notes |
|---|---|---|
| `SENTINEL-HF/` total | **24 GB** | ≈ 12 GB working tree + 12 GB `.git` (11 GB `.git/lfs` objects = duplicate of the LFS files in the tree) |
| `SENTINEL-HF/data/` | 4.6 GB | 4.2 GB is one file (`processed/satellite/drone_wq_pairs.npz`); everything else is 0.4 GB |
| `SENTINEL-HF/checkpoints/` | 6.0 GB | 64 `.pt` + 36 small results JSONs; molecular 2.8 GB, microbial 1.8 GB, waterdronenet 0.63 GB, satellite 0.31 GB, sensor 0.18 GB, biomotion 0.11 GB, foundation 0.10 GB, fusion 0.09 GB, other 0.07 GB |
| `SENTINEL-HF/*.pt` (12 root files) | 1.3 GB | 11 are md5-duplicates of files in `checkpoints/`; only `hydrovit.pt` (160 MB) is unique |
| `SENTINEL-HF/results/` | 55 MB | ~240 files: mostly JSON plus 6 `.npz` caches (twin cache 34 MB, Track-A residual caches 17 MB) |
| `SENTINEL-HF/figures/`, `paper/`, `demo/`, `logs/` | 26 / 2.7 / 2.9 / 1.9 MB | `figures/` show retracted numbers (audit C10); `logs/` (155 files) are useful provenance |
| `SENTINEL-STOCKHOLM/` | 172 MB | 149 MB `.git`; working tree ~23 MB (8.8 MB of it is the paper .docx/.pdf) |
| `review_artifacts/` + `REVIEW_REPORT.md` | 4.8 MB | audit evidence scripts (several reusable, §3.1 E) |
| Duplicated checkpoint bytes | 1.64 GB | 76 `.pt` files → 63 unique md5s (12 duplicate groups, §4) |
| **Disk free (`/`)** | **359 GB** | 369 GB at the start of this session; other jobs are consuming space |

**Storage planning for re-downloads.** These are estimates from local evidence, not quotes.
- **Sentinel-2 chips, as previously stored** (4 bands × 224² float32, savez_compressed): **0.37 MB per chip** (4.46 GB / 12,000).
  - The earlier 78,734-tile "massive" pull would be ≈ 29 GB in that format.
  - Storing uint16 DN instead halves the raw size. Keeping 10–13 bands roughly triples it.
  - Budget **≈ 30–100 GB** for a 100 k-chip, 10-band v2 matchup set.
- **USGS IV:** the 383-station 2022–2025 pull held 51.6 M records. A v2 pull of 1–3 k stations over 10 years is likely tens of GB as parquet.
- **NEON DP1.20288.001:** 51.4 M rows were consolidated before; plan for a few GB.
- **Easily affordable within 359 GB:** EMP release-1 deblur BIOM + mapping, ECOTOX ASCII, the GEO series matrices and WQP pulls are each ≤ a few GB.
- **Optional cleanup:** pruning `SENTINEL-HF/.git/lfs` would recover ~11 GB (a later, deliberate decision; not done).

---

## 2. Data inventory

### 2.1 Master table

Paths are relative to `SENTINEL-HF/data/processed/` unless noted. "Script" means the producing script in the repo.

| # | Dataset | Size / format | Contents (verified) | Provenance (script → public source) | Known problems | Verdict |
|---|---|---|---|---|---|---|
| D1 | `sensor/full/*.npz` + `preprocessing_stats.json`, `normalization_stats.json` | 3.7 MB; 877 npz + 2 json | **Only 877 of 127,421 windows (0.7%), from 2 stations** (01302020: 354, 01303152: 523). Each npz: `values` (512, 6) f32 z-scored [DO, pH, SpCond, Temp, Turb, ORP]; `delta_ts` (512,) seconds (900 s = 15 min); `labels` (512,) all 0; `mask` (512, 6), with ORP never observed. The stats JSON lists all **381 stations** (sequences per station 6–1,090, median 348); 512-step windows, 25% overlap, clamp ±5 | `scripts/data_acquisition/download_sensor.py` (dataretrieval `nwis` IV, pcodes 00300/00400/00095/00010/63680, 2022-01-01→2025-12-31, ≥3 params) → USGS NWIS IV (`waterservices.usgs.gov/nwis/iv/`) | **Normalization stats include the −999999 code** (DO μ −1,848, σ 43,051; pH μ −293; Temp μ −166; Turb μ −248). Normalized DO spans only 0.0429–0.0434 (σ 6e-5) **(measured)**, so 4 of 5 real channels are constant. Raw values *are* recoverable (x·σ+μ, ~1e-4 precision: DO 4.4–14.8 mg/L), but only for these 2 tidal stations. Legacy threshold labels | **DISCARD** the windows; **REUSE AS-IS** the 381-station list in `preprocessing_stats.json` as a seed manifest. **Re-download raw USGS IV** |
| D2 | `satellite/drone_wq_pairs.npz` | **4.46 GB** (9.6 GB raw); npz | `images` (12000, 4, 224, 224) f32 = L2A DN/10000, bands **B02, B03, B04, B08** (10 m → 2.24 km chip). `targets` (12000, 5) f32 [DO, pH, Turb, Temp, SpCond]: finite 11,318 / 10,859 / 8,657 / 11,954 / 11,752; all five finite in 7,682. `metadata`: {site_id, date, lat, lon}. **379 USGS stations** (378 overlap D1), 1–81 chips each. Nominal dates 2022-01-05 → 2026-01-05, **only the 5th, 15th or 25th** of a month; lat 25.8–47.1, lon −123.1 to −72.3; dominated by the Northeast/Mid-Atlantic (site prefixes 01/02 = 194 stations) | `download_drone_s2_pairs.py` → Planetary Computer STAC `sentinel-2-l2a` (scene cloud < 30%, nearest scene within ±5 d of nominal date) + USGS IV parquet daily mean (nearest day within ±5 d, physical-range filtered) | **BOA offset uncorrected** in 11,814 chips (186 pre-2022-01-25). Median per-chip min reflectance 0.105 **(measured)**. No SCL / cloud / water mask; 628 chips bright (B02 mean > 0.3), 231 with > 10% nodata. Water at the gauge pixel in only 16% of chips. **Actual acquisition datetime, item ID and processing baseline not stored.** In-situ match is relative to the nominal date (up to ~10 d from the overpass). Random-station split proximity (audit C6). Docstring says "same-day", code does ±5 d | **REUSE RAW BUT REPROCESS** (subtract 0.1 for post-PB04 chips; add water masks; use as an interim benchmark). **Plan a provenance-complete re-download** for v2 |
| D3 | `satellite/sentinel3/*.json` | 124 KB; 4 json | CDSE product catalog (300 OLCI WFR/WRR products discovered, **0 downloaded**), CMEMS dataset list, auth guide | `download_sentinel3.py` | catalog only, no data | **DISCARD** |
| D4 | `trackA/` (`extract_summary.json`, `expanded_manifest.json`, `neon_satellite_coverage.json`, `s2_anomaly_run.log`, `s2_anomaly_tiles/*.fail` × 105) | 452 KB | NEON hypoxia event manifests (DO < 4 mg/L, 15-min bins, 7-d baseline). **34 NEON aquatic sites**: 15 DO-hypoxia episodes with onset UTC, 17 controls, 2 dropped (Appendix A). Spans mostly 2024-05 → 2026-04; per-site S2 scene counts and NDCI baselines. The `.fail` files are empty failure markers | `scripts/experiments/trackA_leadtime/01_extract_episodes.py`, `06_extract_expanded.py` → NEON DP1.20288.001 (`data.neonscience.org/api/v0`), QC'd on `*FinalQF == 0` | Raw series absent; onsets are threshold crossings of the input channel (fine as event labels, not for "lead time" claims). Placebo failure of the detector (audit C2) does not invalidate the event list | **REUSE AS-IS** (event catalog). Rebuild series from the NEON API |
| D5 | `trackA_usgs/` (`usgs_manifest.json`, `_build_results.json`, `_iv_series_catalog.csv`, 4 candidate CSVs) | 824 KB | **17 DO-hypoxia episode + 22 control USGS sites** (Appendix A), 2022-06 → 2025-07, 15-min bins, artifact screening. `_iv_series_catalog.csv`: **6,318 IV series at 973 sites** (parm, begin, end, count; 1961–2026) | `trackA_leadtime/usgs/build_usgs_npz.py`, `make_manifest.py` → NWIS IV + site service | 11 of 17 episode and 12 of 22 control stations overlap AquaSSM pretraining (audit C2). Per-site npz absent | **REUSE AS-IS** (manifests + IV catalog for site selection) |
| D6 | `trackA_salt/` (`salt_manifest.json`, `_build_results.json`, 3 CSVs) | 148 KB | **30 salinization (SpCond excursion) episodes + 11 controls**; site-relative threshold max(median + 4·IQR/1.349, p99.5) sustained ≥ 6 h; 594 candidate conductance sites | `trackA_leadtime/salinization/build_salt_npz.py`, `make_salt_manifest.py` → NWIS IV | per-site npz absent | **REUSE AS-IS** (event catalog) |
| D7 | `hydrology/nhdplus_graph.json`, `comid_mapping.json`, `nhdplus_graph_stats.json` | 300 KB | 561 USGS sites → NHDPlusV2 COMID, reachcode, measure, drainage area, stream order. Graph: 338 edges (48 are same-COMID twins), **247 connected components, largest 22, 146 isolated** **(measured)**. Travel time 0.08–232 h. 322 sites overlap D1 | `download_nhdplus.py` → USGS NLDI (`api.water.usgs.gov/nldi/linked-data`) + Fabric pygeoapi `nhdflowline_network` | Edges only between monitored sites, so the graph is fragmented; label circularity is in the GNN script, not the data | **REUSE AS-IS** the COMID map; **REPROCESS** the edges (full NHDPlus topology / NLDI navigation) |
| D8 | `biology/noaa_habs/mc_fusion_dataset.parquet` + `_mc_dg_featurecache.parquet` | 0.58 + 0.02 MB | **The audit's "solid" microcystin table.** 3,649 site-dates, 227 sites, 2016-04-26 → 2025-10-22. Columns: `mc` (µg/L), `cloud`, **`sf0..sf25`** (6 bands B02/B03/B04/B05/B08/B11 × [tile mean, centre-20×20 mean, std] + NDVI/NDCI/NDWI/G:R × [mean, centre]), `CHLA`, `PC`, `label` (> 8). Positives: 286 (> 8), 633 (> 1.6), 1,286 (> 0.3). Driver cache, row-aligned: `drv_temp` (2,922 finite), `drv_phosphorus` (1,977), `drv_nitrogen` (205). Coordinates recoverable exactly via `idx` → `mc_download_targets` row **(measured)** | `trackB_fusion/analyze_microcystin_fusion.py` (features; pigment merged ±3 d) ← `download_microcystin_s2.py` (64×64, ±7 d, cloud < 40%). Drivers: `08_driver_arm.py` (WQP ±14 d) cached by `11_domaingen_ood.py`. Source: WQP + Planetary Computer S2 L2A | **BOA-offset era mix** (§0 item 3). **B05/B11 windows cover 4× the area of the 10-m bands** (no resampling). CHLA/PC unit mixing (lab µg/L, mg/m³, RFU) + sentinel codes. Dominated by FL (21FL* orgs) and CA tribal programs. 40% of rows have a same-day sample within 5 km (audit W1). P units inconsistent (max 2,180). Thresholds are US-EPA, not WHO | **REUSE AS-IS** as the frozen benchmark to reproduce 0.650 / 0.684 / 0.626 / 0.776 / 0.823. **REPROCESS** for v2: re-extract offset-corrected, water-masked features; harmonize pigment; 5-km blocking |
| D9 | `biology/noaa_habs/mc_fusion_expanded.parquet` + `_manifest.json` | 0.73 MB | 4,196 rows, 637 sites, 18 HUC2, 2015-07 → 2025-10; +796 "expanded" rows (NARS NLA/NCCA, state programs); adds `huc2`, `state`, `source`, `temp_C`, `TP` (1,950 finite), `TN` (185). Only 8 regions have ≥ 30 samples and ≥ 6 positives | `expand_01_pull_microcystin.py` → `expand_01b_make_targets.py` → `expand_02_colocate_s2.py` → `expand_03_build_dataset.py` | same as D8; 53% of rows in HUC2 03 (Southeast) and 28% in HUC2 18 (California) | **REUSE RAW BUT REPROCESS** |
| D10 | `biology/noaa_habs/mc_download_targets.parquet`, `mc_expand_raw.parquet`, `mc_expand_targets.parquet` | 0.10 / 0.38 / 0.16 MB | Microcystin site-date targets with lat/lon: 7,034 (2016-04 → 2025-10); raw expanded **27,835 results, 40 states, 18 HUC2** (incl. **583 NARS NLA rows from 324 lakes**); 11,293 breadth-capped targets | same as D8/D9 → WQP (`waterqualitydata.us`, dataretrieval `wqp.get_results`) | Lab method / analyte heterogeneity (MC-LR vs total MC) not harmonized | **REUSE AS-IS** (raw toxin labels) |
| D11 | `biology/noaa_habs/wqp_hab_cyanotoxins.parquet` | 2.2 MB | 30,097 WQX3.0 results, 5,438 sites, 2015-04 → 2025-12: MC-LR 10,930, cylindrospermopsin 8,820, saxitoxin 5,328, microcystin 4,379, cyanobacteria qPCR 639. Units mostly µg/L. Includes HUC8 / HUC12 codes | `download_habs_wqp_cyanotoxins.py` → WQP WQX3.0 API | 11 states (FL 69%); HUC codes stored as floats (source of the `10_region_transfer.py` bug) | **REUSE AS-IS** |
| D12 | `biology/noaa_habs/wqp_hab_indicators.parquet` (+ `_columns.csv`) | 31.8 MB | **754,709 pigment results, 43,278 sites**, 2015-01 → 2025-12: chl-a corrected 374,520 / uncorrected 227,804 / plain 68,371 / free 44,782 / probe-RFU 32,024; phycocyanin 5,142 + 2,066 RFU | `download_habs_wqp.py` → WQP Result service | FL ≈ 73% of rows (21FL*), MN PCA, WI DNR. 20,665 non-numeric, 298 ≥ 99,999, 597 negative values; RFU is not µg/L | **REUSE RAW BUT REPROCESS** (harmonize characteristic + units) |
| D13 | `biology/noaa_habs/erddap_*.parquet` (6), `erddap_hab_catalog.json`, `download_metadata.json`, `mc_tiles/*.fail` (523), `mc_expand_tiles/*.fail` (872) | 0.4 MB + markers | Coarse ocean-colour chlorophyll (MODIS 8-day/monthly, VIIRS weekly/monthly) on a subsampled ~0.75–0.83° lattice over the Great Lakes / Chesapeake / Florida (2018–2026). `.fail` files hold only "no_items" / "edge" / "empty" | `download_noaa_habs.py` → CoastWatch ERDDAP; HABSOS returned 0 records | far too coarse for inland water; tiles absent | **DISCARD** (re-pull ERDDAP / CyAN if needed) |
| D14 | `biology/usgs_biodata/` (`biodata_invertebrates`, `biodata_fish`, `biodata_sites`, `nwis_bio_sites`, `wqp_biological` parquet; `download_metadata.json`) | 18 MB | HUC2 02/04/05 only, 2015–2025. **Invertebrates 384,885 counts / 8,019 sites / 4,208 taxa**; **fish 16,242 counts / 1,562 sites / 68 taxa**; 24,321 site records (drainage area, HUC8, state); 1,086 NWIS site-parameter rows (736 sites) with pigment params (MN/MI/WI); `wqp_biological` 300,000 rows (E. coli 186,995, chl-a 62,919, coliform 20,267, …) | `download_usgs_biodata.py` → WQP biological results (BioData via WQP; `aquatic.biodata.usgs.gov` referenced) | **`wqp_biological` truncated by `df.head(300000)`**, leaving HUC04 278,721 / HUC05 21,279 / HUC02 0. Survey effort confounds presence/absence: 80% of sites never fish-surveyed (audit W2) | **REUSE RAW BUT REPROCESS** inverts / fish / sites; **DISCARD** `wqp_biological` (re-pull) |
| D15 | `biology/wq_samples/expanded_wq_samples.csv`, `great_lakes_wq_samples.csv` | 51.7 + 7.1 MB | USGS discrete samples (WQX3.0 layout): 84,507 rows / 19 sites (59 characteristics: stage, flow, SpCond, temp, nutrients); 10,693 rows / 6 Great Lakes sites (nutrients). 2015–2025 | **no producing script in the repo** (format matches `download_habs_wqp_cyanotoxins.py`) | provenance gap; few sites | **DISCARD** (re-pull WQP for the v2 site list) |
| D16 | `fib_nowcast/` (`wqp_ecoli_ohio.csv`, `wqp_stations_ohio.csv`, `ghcn_prcp/` 125 files, `ghcnd_stations_ohbbox.csv`, `ghcnd_stations_all.txt`, `dataset_meta.json`; `region_MI/`, `region_WI/` same layout) | 136 MB | E. coli results: **OH 29,780 rows / 452 sites; MI 89,989 / 1,799; WI 62,809 / 1,029**, 2015-01 → 2023-12. GHCN-Daily PRCP: OH 125 files (72,256 rows), MI 467 (448,901), WI 754 (686,658); 489 files are header-only. Global GHCN station list (11 MB). Built tables were in memory only; their meta: OH 23,982 rows / 90 sites / 20.1% positive, MI 21,919 / 274 / 7.2%, WI 49,166 / 443 / 13.9% | `scripts/data/fetch_fib_nowcast_data.py`, `fetch_fib_nowcast_region.py` → WQP Result + NOAA NCEI (`ncei.noaa.gov/access/services/data/v1`, `ghcnd-stations.txt`) | Unit filter admits `CCE/100mL` qPCR (WI 1,521); media filter admits groundwater / "street sweepings" rows (MI 325); MPN/g sediment rows present. Leakage caveats are in the evaluation, not the data (audit W3) | **REUSE AS-IS** the raw tables; **REPROCESS** labels (culture units, surface water only) |
| D17 | `microbial/emp_taxonomic_features.npz` | 33 MB | **26,181 EMP samples × relative abundance** at phylum 86 / class 231 / order 372 / family 470 / genus 1,247 (rows sum to 1; 77–95% zeros); 97 EMP studies; `sample_ids` = EMP `#SampleID`. **No labels, no coordinates** | `scripts/domaingen/extract_taxonomic_features.py` ← EMP release-1 deblur 90-bp BIOM (317,314 ASVs, Greengenes taxonomy) | Docstring says CLR but stored values are relative abundance. Labels and lat/lon need the EMP mapping file (absent). The 8-class labels are EMP habitat types with ~63% coin-flip "impacted" labels (audit C5/W6) | **REUSE AS-IS** (features); **re-download** mapping + BIOM |
| D18 | `microbial/emp_16s/metadata.json`, `emp_16s_full/metadata.json` | 8 KB each | Metadata of the absent per-sample npz sets (1,390-sample subset; 20,288-sample full, top-5,000 ASVs, 8 classes) | `download_emp_microbiome.py`, `process_emp_full_release.py` | data absent | **DISCARD** (keep only as documentation) |
| D19 | `molecular/realgeo/genes.json`, `samples.json` | 87 + 97 KB | **Manifest of the clean 12-series zebrafish dataset**: 279 GSMs in GSE90457 (37), GSE110543 (36), GSE93227 (34), GSE116641 (32), GSE121338 (28), GSE89653 (27), GSE89756 (22), GSE84394 (18), GSE129812 (15), GSE93532 (14), GSE112016 (12), GSE110340 (4); 94 controls / 185 exposed, with `label_reason`, title, source, characteristics; 7,902 shared gene symbols | `scripts/preprocessing/build_toxigene_realgeo.py` → NCBI GEO series matrices | **`X.npy` / `y.npy` / `study.npy` absent.** Labels come from control-token heuristics; tissues differ by series (liver, testes, larvae, gill, eye, brain) | **REUSE AS-IS** (manifest); rebuild X from GEO |
| D20 | `molecular/real/GSE73661.npz`, `GSE83514.npz`, `geo_summary.json`; `molecular/geo_sample_metadata.json`, `gene_names.json` | 26 MB | GSE73661 = human ulcerative-colitis biopsies (33,252 × 178); GSE83514 = FMDV-infected cattle (45,220 × 40); metadata also lists GSE54800 / GSE104776 (cancer lncRNA). `gene_names.json` holds placeholders `gene_0000…` | `download_geo_transcriptomics.py`, `prepare_geo_for_toxigene.py` | **Not aquatic toxicology** (audit W6) | **DISCARD** |
| D21 | `behavioral/ecotox_daphnia_conc_response.csv`, `ecotox_daphnia_curves.csv`; `behavioral_real/metadata.json`, `behavioral_fullreal/metadata.json` | 52 KB + 16 KB | 1,901 rows (effect, endpoint, conc_mgL, effect_pct) and 174 rows (chemical, conc, effect); no CAS / test IDs. The metadata JSONs describe the absent 17,074- / 29,421-record pseudo-trajectory sets | `process_ecotox_behavioral.py`, `expand_biomotion_data.py` → EPA ECOTOX ASCII (`cfpub.epa.gov/ecotox/files/ecotox_ascii_*.zip`) | No behaviour data exist (audit C5); too small and identifier-free for chemical-grouped splits | **DISCARD**; **re-download ECOTOX ASCII** |
| D22 | `trackB/embeddings.npz`, `embeddings_v2.npz`, `sensor_noleak.npz`, `pairs_summary.json`, `hab_s2_coverage.json` | 68 MB | 11,185 (11,183) chip-indexed embeddings: `sat` 384-d (WaterDroneNet), `sensor` 256-d (AquaSSM), `spectral` 32-d. `hab_s2_coverage.json`: S2 coverage for 3 HAB-advisory lakes (Detroit L. OR 2024, Seneca L. NY 2019, Milford L. KS 2016) | `trackB_fusion/02_embed.py`, `02b_add_spectral.py`, `taskB_rebuild_sensor_noleak.py` | Embeddings come from retired encoders (AquaSSM with corrupted normalization); `pairs_*.npz` absent | **DISCARD** (keep `hab_s2_coverage.json` as an event note) |
| D23 | `../external/kegg_dre/` (`pathways.txt`, `pathway_gene_links.txt`, `gene_list.txt`, `pathway_symbol_sets.json`) | 4.5 MB | KEGG *Danio rerio*: 196 pathways, 41,508 genes, 38,960 pathway–gene links, symbol sets | `scripts/benchmarks/toxigene_pathway_batch.py` → KEGG REST (`rest.kegg.jp`) | KEGG licence: academic use; snapshot date unknown | **REUSE AS-IS** (refresh at release) |
| D24 | `../tierA/site_metrics.csv`, `stations_all.csv`, `us-states.geojson` | 1.6 MB | 7,878 sites with taxon richness, EPT taxa, EPT / tolerant presence (HUC2 02 / 04 / 05); 8,022 WQP stations; US states polygon | `scripts/figures/build_tierA_figures.py` → WQP biological + Station | derived; EPT flag from name matching | **REUSE RAW BUT REPROCESS** (rebuild from D14 with taxonomy tables) |
| D25 | `SENTINEL-HF/results/benchmarks/twin_eval_cache.npz` | 34 MB | preds / lower / upper / targets / masks (6 horizons × 51,549 site-held-out real USGS windows × 10 vars) + physics + init states. Observed fraction per var: [0.44, 0, 0, 0, 0, 0.97, 0.13, 0.85, 0, 0] **(measured)** | `scripts/evaluation/evaluate_twin.py` (synthetic-trained twin) | predictions from a synthetic-trained model; 6 of 10 vars never observed; dates need raw USGS | **DISCARD** (rebuild windows from raw) |
| D26 | `SENTINEL-HF/results/trackA_leadtime/**/residual_cache*.npz` (4), `results/exp5_explainability/temporal_attention.npz`, `results/exp6_propagation/dan/*.npy` | 17 MB | Per-site AquaSSM reconstruction residuals (22 NEON, 39 USGS, 41 salinization sites; window ends, timestamps, coverage, 4-channel error) | `aquassm_specificity/cache_residuals*.py` | derived from the degenerate AquaSSM (audit C2) | **DISCARD** (keep only to re-run the placebo audit) |

### 2.2 Notes per modality (what v2 can lean on)

- **Sensor time series (USGS / NEON).** Nothing usable at full resolution is on disk. These are worth keeping:
  - The 381-station seed list (D1), extendable with the 973-site, 6,318-series IV catalog (D5).
  - The event catalogs: 15 NEON + 17 USGS DO episodes, 39 controls, 30 salinization episodes (D4–D6).
  - The QC logic in the event builders: NEON `*FinalQF == 0` gating; USGS qualifier exclusion `{"Eqp","Mnt","Dis","***","Bad","Ssn","Fld","ZFl","Rat","Pr"}` plus flatline/spike artifact screening in `usgs/build_usgs_npz.py::screen_artifacts`.
  - Neither `download_sensor.py` nor `sentinel/data/sensor/preprocessing.py` masks −999999. Use the range tables in `sentinel/data/sentinel_db/quality.py::PlausibleRange` (or `PARAM_RANGES` in `download_drone_s2_pairs.py`) when re-ingesting.
- **Satellite.** D2 is the only real chip set: 4 bands only, no masks, offset uncorrected, nominal dates. It is fine for reproducing the HydroViT / climatology comparison:
  - the audit DOY + lat baseline gives R² 0.878 for Temp (`review_artifacts/satellite/doy_baseline_results.json`);
  - the ViT-S/16 test R² is 0.7286 (`logs/train_hydrovit_real.log`).

  A SOTA satellite model needs a fresh, provenance-complete pull: item ID, datetime, processing baseline, SCL, all 10 m / 20 m bands as uint16. The Sentinel-3 and ERDDAP holdings are catalogs or coarse grids only.
- **Microcystin / HAB (the best result).** D8 is self-contained and reproduces the result today. D9–D12 hold the raw labels and pigments needed to rebuild it properly. The NLA lake microcystin present locally is only the 583 WQP NARS rows. The full NLA 2007/2012/2017/2022 files (chemistry, chl-a, toxins, landscape) are **not on disk** (re-download from EPA NARS).
- **FIB / beaches.** D16 is complete raw material for three states. Weather covariates on disk are limited to GHCN-Daily precipitation for those regions and the WQP driver cache (temp / TP / TN) for microcystin rows.
  - No ERA5, Daymet, PRISM, gridMET, NLDAS, NLCD, StreamCat or HydroATLAS data are on disk.
  - Downloaders exist for StreamCat (`scripts/training/species_mechanism_download_streamcat.py`) and HydroLAKES / LakeATLAS (`download_hydrolakes.py`).
- **Species / biodiversity.** D14 (inverts + fish) is real and sizeable. Effort-aware occupancy needs the activity / sampling metadata it contains. Re-pull `wqp_biological`.
- **Microbial.** D17 is the only matrix. To use it, re-download the EMP mapping file (`emp_qiime_mapping_release1.tsv`) for env labels and lat/lon, plus the release-1 BIOM if ASV-level features are needed.
- **Chemical / biological (toxicology).** Nothing usable for expression except the D19 manifest. ECOTOX must be re-downloaded (the code expects `data/raw/ecotox/ecotox_ascii_03_12_2026`). KEGG (D23) is ready.
- **Drone RGB + NIR.** No drone imagery exists anywhere locally. The only proxy is the S2 4-band chip set (D2). The ROS2 camera workspace on `origin/camera-model` is hardware scaffolding (§3.3).

### 2.3 Referenced by code but absent: must re-download or rebuild

| Missing path (as in code) | Code refs (approx.) | What it was | Rebuild with | Public source |
|---|---|---|---|---|
| `data/raw/sensor/full/*.parquet` | 23 | 383 USGS IV station parquets, 2022–2025, 51.6 M records | `download_sensor.py` (add −999999 / qualifier masking) | USGS NWIS IV via `dataretrieval` (or the new `api.waterdata.usgs.gov` OGC API) |
| `data/raw/neon_aquatic/neon_DP1.20288.001_consolidated.parquet` | 16 | NEON water quality, 51.4 M rows (1-min) | `trackA_leadtime/01_extract_episodes.py` inputs; `compress_neon_streaming.py` | NEON API `data.neonscience.org/api/v0` DP1.20288.001 (+ DP1.20264.001 temperature) |
| `data/raw/emp/emp_qiime_mapping_release1.tsv` + `emp_deblur_90bp.release1.biom` | 8 | EMP release-1 metadata + 317,314-ASV table (26,181 samples) | `process_emp_full_release.py`, `extract_taxonomic_features.py` | Earth Microbiome Project release 1 (Thompson et al. 2017, EMP FTP / Qiita) |
| `data/raw/ecotox/ecotox_ascii_03_12_2026/` | 5 | ECOTOX quarterly ASCII dump | `biomotion_gbm.py`, `sentinel/data/ecotox/download.py` | `cfpub.epa.gov/ecotox/` (bulk ASCII) |
| `data/processed/molecular/realgeo/{X,y,study}.npy` | 5 | 279 × 7,902 zebrafish expression | `build_toxigene_realgeo.py` (series list in D19) | NCBI GEO series matrices |
| `data/processed/microbial/emp_16s(_full)/*.npz` | 9 | per-sample top-5,000-ASV npz | `process_emp_full_release.py` | EMP (above) |
| `data/processed/biology/noaa_habs/mc_tiles/*.npz`, `mc_expand_tiles/*.npz` | 2+ | 64×64 6-band S2 tiles for microcystin rows | `download_microcystin_s2.py`, `expand_02_colocate_s2.py` (fix offset, store item ID / SCL) | Planetary Computer S2 L2A |
| `data/processed/satellite/drone_tiles/`, `crisis_tiles/`, S2 "massive" cache | 6 | 4-band 224² tile caches (78,734 tiles) | `download_drone_s2_pairs.py`, `download_s2_massive.py` | Planetary Computer S2 L2A |
| `data/raw/hydrology/nhdplus/cache/site_info/*.json`, `fabric_flowline_attrs.json` | 10 | site coordinates + flowline attributes | `download_nhdplus.py` | USGS NLDI + Fabric pygeoapi |
| `data/raw/wqp_drivers/*.csv` | 9 | WQP temp / P / N at microcystin sites | `08_driver_arm.py::wqp_download` (D8 already holds the co-located values) | WQP |
| `data/processed/trackA/*.npz`, `trackA_usgs/*.npz`, `trackA_salt/*.npz` | 17 | per-site 15-min series for events / controls | `01_extract_episodes.py`, `usgs/build_usgs_npz.py`, `salinization/build_salt_npz.py` | NEON / NWIS |
| `data/processed/trackB/pairs_chl.npz`, `pairs_forecast.npz` | 16 (dir) | Track-B co-located labels | `trackB_fusion/01_build_pairs.py` | n/a (legacy; not needed) |
| `data/processed/biology/species_mechanism/` | 3 | site → COMID table + StreamCat metrics (~993 sites) | `species_mechanism_build_sites.py`, `species_mechanism_download_streamcat.py` | EPA WATERS PointIndexing + EPA StreamCat |
| `data/sentinel_db/grqa` | 4 | GRQA v1.3 | `download_grqa.py`, `ingest_grqa.py` | Zenodo 10.5281/zenodo.7056647 |
| `data/processed/satellite/paired_wq_v5.npz` | — | **fabricated HydroViT set (quarantined)** | do **not** rebuild (`expand_hydrovit_data.py` injected ±10% target noise) | — |

---

## 3. Code inventory (`SENTINEL-STOCKHOLM`, 419 .py files, ~172 k LOC)

HF code is identical except for one extra file (`scripts/evaluation/reeval_case_studies_31.py`). "Runs now?" means the inputs are on disk today.

### 3.1 Keep: v2 templates

**A. Clean data downloaders** (all need: replace hard-coded paths with a config; add a provenance manifest, i.e. source URL, query, timestamp, row counts and sha256; remove synthetic fallbacks; add rate limiting / caching).

| Path | What / key functions | Why keep | Fix before reuse |
|---|---|---|---|
| `scripts/data_acquisition/download_sensor.py` | `discover_stations_smart`, `download_station_data` (dataretrieval NWIS IV; 5 pcodes; per-station parquet) | simple, real | **mask −999999 + qualifiers** before stats; fit normalization on train stations only; drop the placeholder ORP channel |
| `scripts/data_acquisition/discover_usgs_stations.py`, `download_sensor_fast.py` | NWIS site service discovery (3,620 sites found); daily-value variant | site discovery | path config |
| `sentinel/data/sensor/download.py` | `discover_stations`, `_download_iv_waterdata`, `download_bulk`, `save_station_catalog` | supports the newer waterdata endpoint | same masking fix |
| `scripts/data_acquisition/download_drone_s2_pairs.py`, `download_nwis_s2_pairs.py`, `download_s2_massive.py`, `sentinel/data/satellite/download.py` (`PlanetaryComputerDownloader`) | STAC search → signed COG window read at a gauge | working Planetary Computer pattern | **apply `BOA_ADD_OFFSET` per item (PB ≥ 04.00)**; store item ID, datetime, PB, cloud, SCL; keep uint16; match in-situ to the *actual* overpass; add water masks |
| `scripts/experiments/trackB_fusion/download_microcystin_s2.py`, `expand_02_colocate_s2.py` | 6-band (B02/B03/B04/B05/B08/B11) 64² tiles ±7 d; stores `actual_date`, `cloud` | stores the actual date | same offset / SCL fix; **resample B05/B11 (20 m) to the 10-m grid**. Today they are read as native 64×64 windows covering 1.28 km vs 0.64 km |
| `scripts/data_acquisition/download_habs_wqp.py`, `download_habs_wqp_cyanotoxins.py`, `scripts/experiments/trackB_fusion/expand_01_pull_microcystin.py`, `08_driver_arm.py::wqp_download` | WQP legacy + WQX3.0 pulls for pigments, cyanotoxins, drivers | sources of the best result | path config; keep HUC codes as zero-padded strings |
| `scripts/data/fetch_fib_nowcast_data.py`, `fetch_fib_nowcast_region.py` | WQP E. coli + NCEI GHCN-Daily PRCP; nearest-gauge antecedent rain | real, label independent of predictors | tighten the unit filter (exclude CCE / qPCR, MPN/g) and media (surface water only) |
| `scripts/data_acquisition/download_usgs_biodata.py` | WQP biological / BioData, fish, inverts, sites | real | **remove the `df.head(max_records…)` truncation** |
| `scripts/data_acquisition/download_nhdplus.py` | NLDI site → COMID, Fabric flowline attributes, graph build | real APIs | build the full network topology, not only site–site edges |
| `scripts/data_acquisition/process_emp_full_release.py`, `scripts/domaingen/extract_taxonomic_features.py` | memory-safe BIOM → npz; rank aggregation | real | write CLR if the docstring says CLR |
| `scripts/preprocessing/build_toxigene_realgeo.py` | `parse_series_matrix`, `assign_label`, `collapse_to_genes`; the 12-series list | the audit-endorsed dataset | make label assignment reviewable (manual table rather than token heuristics) |
| `sentinel/data/ecotox/download.py`, `preprocessing.py` | `download_ecotox_bulk`, `filter_freshwater_aquatic`, `standardize_concentration`, `split_by_chemical` | real ECOTOX handling | update the release date |
| `scripts/training/species_mechanism_build_sites.py`, `species_mechanism_download_streamcat.py` | EPA WATERS COMID indexing + StreamCat catchment metrics | real catchment covariates | finish COMID resolution (stopped at 1,000) |
| `scripts/data_acquisition/download_grqa.py`, `ingest_grqa.py`, `download_hydrolakes.py`, `download_epa_wqp.py`, `download_eu_waterbase.py`, `download_gems_water.py` | GRQA (Zenodo), HydroLAKES / LakeATLAS, WQP by HUC2, EU Waterbase, GEMStat | real global sources (never used for results) | verify endpoints; GEMStat has a fallback path |
| `trackA_leadtime/01_extract_episodes.py`, `06_extract_expanded.py`, `usgs/build_usgs_npz.py` (+ `run_build_parallel.py`, `make_manifest.py`), `salinization/build_salt_npz.py`, `make_salt_manifest.py`, `conductivity/define_conductance_events.py` | event catalogs from raw values with QC + artifact screening | best existing event-definition code | keep only the event logic; drop residual/detector coupling |

**B. Clean evaluation protocols the audit called sound**

| Path | Key functions | Protocol | Runs now? | Fix |
|---|---|---|---|---|
| `scripts/training/train_hydrovit_real.py` | `station_holdout_split`, `masked_r2`, `baseline_r2_const`, `run` | station-disjoint split asserted; train-only target / band stats; val-only selection; timm ViT-S/16 ImageNet, 4 bands | **Yes** (D2 + timm weights download) | add the DOY + lat baseline (`review_artifacts/satellite/doy_baseline.py`), spatial blocking and the offset fix; pinned `cuda:0` |
| `scripts/experiments/trackB_fusion/analyze_microcystin_fusion.py` (`spectral_features`, `load_dataset`), `07_microcystin_fusion.py` (`hgb`, `oof`, `site_bootstrap_delta`), `09_driver_robustness.py`, `08_mc_fusion_robustness.py`, `fusion_cycle2/modality_contribution.py`, `10_region_transfer.py`, `11_domaingen_ood.py` | GBM fusion, station-grouped CV, site-cluster bootstrap, shuffle control, leave-region-out | the project's best result | **Yes** for 07 and 08 (D8 only). 09, 10, `modality_contribution.py` and 11 read `data/raw/wqp_drivers` (absent). Script 11 uses the driver cache only if its mtime is *newer* than the source, and in the HF clone the times are equal. The 3-way 0.823 is reproducible by joining `_mc_dg_featurecache.parquet` row-wise to D8, as `review_artifacts/bio_fusion/scripts/mc_check.py` does | **HUC bug:** `10_region_transfer.py` lines 124–126 do `.astype(str).str.zfill(8).str[:2]` on float codes, so `3090201.0` becomes "30". Make **5-km spatial-block CV** the headline (`review_artifacts/bio_fusion/scripts/mc_spatial3.py`). Harmonize CHLA / PC; credit the GBM |
| `scripts/benchmarks/benchmark_toxigene_realgeo.py` | `load_data`, `run_logreg`, `run_within_study_cv`, `_study_metrics`, `pooled_metrics` | 12-series LOSO; feature selection inside each fold; study-block bootstrap | No (needs D19 → X / y rebuild) | report "best-of-N on test folds" correctly (audit W4) |
| `scripts/domaingen/run_domaingen_experiment.py` (+ `multiseed_sensitivity.py`), `scripts/training/train_microbiomenet.py::load_and_split_data_geographic`, `sentinel/data/splits.py::geographic_holdout_split` | `is_coinflip_label`, `clr`, `bootstrap_ci`, `train_logreg` | KMeans(20) region-disjoint holdout; coin-flip labels excluded | No (needs EMP mapping) | match feature budgets across arms (audit W5) |
| `scripts/experiments/biomotion_gbm.py`, `biomotion_transfer.py` | `build_table`, `feature_matrix` (guardrail asserts on banned features), `loco_class`, `cross_species` | CAS-grouped split with overlap assert | No (needs ECOTOX) | reframe as ECOTOX test-outcome modelling, not "behaviour" |
| `scripts/evaluation/eval_fib_nowcast.py`, `eval_fib_crossregion.py`, `verify_fib_nogeo.py` | `temporal_holdout`, `site_grouped_holdout`, `persistence_baseline`, `cross_region`, `pooled_loro`, `bootstrap_auroc_ci` | labels independent of predictors; persistence baselines | **Yes** (D16) | strictly antecedent features only; no best-of-4 selection on held-out regions (audit W3) |
| `scripts/training/train_species_health_leakagefree.py` | `split_of`, leak-free feature set | target counts removed from inputs | partial (needs raw sensor for 54 sites; otherwise median-imputed) | replace the hash split with a HUC8-blocked split; model survey effort |
| `scripts/evaluation/twin_reformulate.py` | `site_split`, `build_climatology`, `auroc` | site-disjoint; leak-free DOY climatology (AUROC 0.844, audit) | No (raw USGS + absolute path) | rebuild from raw; remove `/home/bcheng/SENTINEL` |
| `scripts/experiments/causal_lowperf/causal-real.py` + `sentinel/evaluation/causal_chains.py::KNOWN_CAUSAL_CHAINS` | PCMCI+ (tigramite, ParCorr, BH-FDR) on NEON BARC / LIRO / LEWI | genuine negative result (0 of 3) | No (NEON raw) | declare `tigramite` |
| `scripts/analysis/transfer_gap_analysis.py` + `results/TRANSFER_GAP_MEMO.md` | ID vs OOD table | the transfer-gap finding | Yes (reads results JSONs) | drop the "fixes" framing (audit W5); one absolute path |
| `sentinel/uncertainty/weighted_conformal.py` | `domain_classifier_weights`, `weighted_quantile`, `conformal_classification_sets`, `conformal_regression_intervals` | Tibshirani et al. 2019, clean split-conformal | Yes | use instead of `sentinel/models/theory/conformal.py` (in-sample scorer) |
| `tests/test_fusion_bugfixes.py::test_input_derived_anomaly_label_is_banned` + `scripts/training/train_fusion.py::_derive_anomaly_from_input` | guard against input-derived labels | keep the idea | Yes | importing `train_fusion.py` **creates `results/` and `checkpoints/` directories** as a side effect; move the guard into a library module |

**C. Small reusable utilities**
- `sentinel/data/sentinel_db/ontology.py`: `harmonize_unit`, `resolve_parameter`, `build_ontology`. It maps raw parameter names to canonical ones.
- `sentinel/data/sentinel_db/quality.py`: `PlausibleRange`, `validate_physical_plausibility`. Use these as the −999999 / range mask.
- `sentinel/data/sentinel_db/schema.py`: pydantic records.
- `sentinel/data/satellite/preprocessing.py`: `compute_ndci`, `compute_fai`, `compute_ndti`, `compute_mndwi`.
- `sentinel/data/alignment/geographic.py`: `find_colocated_sites`, `_haversine_km`. Needs the undeclared `h3`.
- `sentinel/utils/config.py`, `sentinel/utils/logging.py`.
- `sentinel/models/satellite_encoder/hydrovit_backbone.py`: the timm wrapper used by `train_hydrovit_real.py`.

**D. Reference baselines, not templates.**
- `sentinel/models/fusion/` (fixed Perceiver IO, covered by 3 regression tests) and `scripts/experiments/trackB_fusion/mc_neural_fusion_fixed.py`. The Perceiver scores 0.754 on microcystin (audit).
- `scripts/experiments/weak_benchmarks/hydrodensenet/honest_hydrodensenet.py`: GBM on 4-band statistics.
- `scripts/deployment/prospective_validation.py`: a pre-registration hashing idea only.

**E. The review's own re-analysis scripts, worth promoting into the v2 eval suite** (paths in `/home/bcheng/STOCKHOLM/review_artifacts/`; they point at old scratch paths):
- `trackA/placebo.py`, `placebo_perSite_season.py`, `placebo_salt.py`: random and season-matched pseudo-onset controls.
- `satellite/doy_baseline.py`: day-of-year / latitude / longitude GBM on the exact HydroViT split.
- `satellite/marginal_imagery.py`: imagery added on top of climatology.
- `bio_fusion/scripts/mc_spatial*.py`: 5- and 20-km connected-component spatial blocking.
- `downstream/species_disease/fib_*.py`: FIB rebuild, date and leave-one-region-out checks.
- `engineering/static/deps.py`: dependency scan.

### 3.2 Archive (legacy; move to `legacy/` and do not import)

| Group | Paths | Reason |
|---|---|---|
| Figure / paper literals | `scripts/figures/generate_submission_figures.py`, `make_modality_samples.py`, `make_poster_data_samples.py`; `paper_figures_v3.py` (git history only); `sentinel/evaluation/figures.py` | hard-coded values / synthetic panels (audit C1) |
| Legacy sensor heads & experiments | `scripts/preprocessing/process_usgs_to_training.py` (threshold labels), `expand_aquassm_data.py`, `preprocess_sensor.py`; `scripts/training/train_aquassm.py`, `train_phase2_anomaly.py`, `train_frozen_backbone.py`; `scripts/experiments/exp1_case_studies*.py` (literal lead times), `exp_discovery_scan.py`, `exp_pollution_fingerprint.py`, `exp_false_positive_rate.py`, `exp_temporal_persistence.py`, `exp_earlywarning_roc.py`, `exp_case_attribution.py`, `exp3_epa_violation_correlation.py`, `exp18_seasonal_analysis.py`; `scripts/evaluation/predictability_audit.py` (−999999 unmasked), `neon_anomaly_scan.py`; `scripts/deployment/*`; `sentinel/models/sensor_encoder/` | circular / mis-fed heads, wrong channel order (audit C2, C3) |
| Track-A detector stack | `trackA_leadtime/02_detect.py` (single-run fallback), `03_report.py`, `05_finetune_neon.py` (trains on eval windows), `07_satellite_anomaly.py`, `08_loso_generalization.py`, `_detector.py`, `aquassm_specificity/**` (detector chosen on outcomes), `combined/**`, `usgs/cache_residuals_usgs.py`, `detect_csd_ews_*.py` | fails the placebo test; keep only the event builders (§3.1 A) |
| Satellite legacy | `scripts/training/train_hydrovit.py`, `scripts/preprocessing/expand_hydrovit_data.py` (fabricated targets), `expand_satellite_wq_pairs.py`, `scripts/experiments/architecture/hydrovit-spectral.py` (`SHIP_VIT_CHLA = 0.34` literal), `exp4_satellite_imagery.py`, `scripts/training/train_hydrodensenet.py`, `sentinel/models/satellite_encoder/` (except `hydrovit_backbone.py`), `sentinel/models/waterdronenet/` | fabricated data / misattributed numbers (audit C4, C6) |
| BioMotion | `scripts/preprocessing/expand_biomotion_data.py`, `process_ecotox_behavioral.py`; `scripts/training/train_biomotion.py`; `scripts/benchmarks/benchmark_biomotion.py`; `scripts/experiments/exp_behavioral_case_studies.py`, `exp19_behavioral_profile.py`; `sentinel/data/behavioral/**` (52 synthetic hits); `sentinel/models/biomotion/` | ECOTOX records rendered as fake trajectories (audit C5) |
| Microbial / molecular legacy | `scripts/data_acquisition/download_microbial_data.py` (51 synthetic hits), `scripts/preprocessing/expand_emp_dataset.py`, `expand_microbiomenet_data.py`, `expand_toxigene_data.py`, `prepare_geo_for_toxigene.py`, `process_geo_zebrafish.py`; `scripts/training/train_toxigene.py`; `scripts/benchmarks/benchmark_toxigene.py`, `benchmark_toxigene_ml.py`, `benchmark_microbiomenet.py`; `scripts/experiments/exp_microbial_case_studies.py`, `exp_molecular_case_studies.py`; `sentinel/models/microbial_encoder/`, `molecular_encoder/` | leaky random splits, synthetic matrices, circular case studies |
| Fusion / foundation / contrastive | `scripts/training/train_fusion.py` (after extracting the guard), `train_contrastive.py`; `sentinel/training/*`; `scripts/experiments/exp12_*`, `exp15_contrastive_alignment.py`, `exp16_*`, `exp7_*`, `exp5_*`, `exp10_mc_dropout.py` (double sigmoid), `exp11_label_noise_sensitivity.py` (algebraic identity), `exp9_bootstrap_ci.py` (`n_bootstrap = 0`), `exp14_*`, `exp17_*`, `exp20_*`; `trackB_fusion/01–06`, `07_taskC_*`, `07b–07d`, `_fusion.py`, `taskB_*`, `diag_*`, `bloom_ew/*`, `12_bloom_anomaly_ood.py`, `ood_watertype/*` | random embeddings, circular labels, WaterDroneNet / AquaSSM embeddings |
| Twin / GNN / cascade / theory | `scripts/training/train_twin.py`, `scripts/evaluation/evaluate_twin.py`, `validate_twin_realworld.py`, `conformal_calibrate_twin.py`; `sentinel/models/twin/`; `scripts/training/train_stream_gnn.py` (label `anomaly_labels[dst] = float(prob > 0.3)` from the model input, line 460), `sentinel/models/graph/`, `exp6_propagation.py`; `sentinel/models/escalation/`, `sentinel/training/train_escalation.py`; `sentinel/models/theory/conformal.py` (scorer fit on its calibration set), `sensor_placement.py`, `sentinel/evaluation/sensor_placement_eval.py` (arbitrary costs) | synthetic / circular (audit C5) |
| Biology legacy | `scripts/training/train_species_health.py` (circular), `train_disease_forecast.py`, `scripts/evaluation/evaluate_disease_forecast.py`; `sentinel/models/biology/arg_surveillance.py` (pseudo-labels), `disease_forecast.py`, `bioremediation.py`, `inverse_aop.py`, `metatranscriptomic.py`, `field_dose_response.py` | circular / never run on real data |
| Evaluation framework | `sentinel/evaluation/case_study.py` (simulated streams), `ablation.py`, `global_hotspots.py` (40 synthetic hits), `information_analysis.py`, `missing_modality.py`, `conformal_eval.py`, `run_causal_discovery.py` (simulated); `sentinel/data/case_studies/*`; `scripts/evaluation/run_evaluation.py`, `rerun_holdout_experiments.py`, `source_attribution_real.py`, `conformal_real_eval.py`; `scripts/pipeline/*` | built on simulated streams |
| Platform / demo | `sentinel/platform/*` (8 of 11 endpoints are placeholders; CORS `*`), `sentinel/dashboard/app.py` (crashes on load at line 268), `demo/` (React; `Math.random` series labelled OBSERVED), `docker-compose.yml` (starts a missing module), `Dockerfile` | misrepresents results (audit C10, W9) |
| Misc | `scripts/data_acquisition/download_all.py` (fails on import), `download_satellite_real.py`, `coregister_satellite_wq.py`, `download_modis_ocean_color.py`, `download_sentinel3.py` (0 products), `download_noaa_habs.py`, `download_gbif.py`, `download_who_jmp.py`, `build_microbiomenet_full_emp_report.py`; `scripts/_archive/**`; `scripts/sentinel_system.py`; `scripts/preprocessing/extract_real_embeddings.py`, `prepare_phase2_data.py`, `expand_neon_and_others.py` (crashed), `preprocess_biodata.py`, `process_ecotox.py` | superseded or unused |
| Branch `origin/camera-model` | `ros2_ws/src/sentinel_camera` (RGB + NoIR backends, publisher), `sentinel_inference` (frame sync, preprocessing, inference node), `sentinel_perception_msgs/WaterQuality.msg` | **refactor-only** for the drone: keep the camera / frame-sync / msg scaffolding. The inference node loads random weights by default, uses the wrong class, falls back to synthetic frames, and reports success after loading 0 of 182 tensors (audit W12) |

### 3.3 Package map

| Directory | .py files | LOC | Verdict |
|---|---|---|---|
| `sentinel/data/sensor` | 3 | 1,520 | refactor (mask codes; train-only normalization) |
| `sentinel/data/satellite` | 3 | 1,573 | refactor (offset, SCL) |
| `sentinel/data/microbial` · `molecular` · `ecotox` | 3 · 3 · 3 | 1,716 · 2,230 · 805 | refactor / keep downloaders |
| `sentinel/data/sentinel_db` | 7 | 2,750 | keep (ontology, quality, schema); add `h3` |
| `sentinel/data/alignment` · `splits.py` | 2 · 2 | 683 · 509 | keep after review |
| `sentinel/data/behavioral` · `case_studies` | 4 · 3 | 2,731 · 1,145 | archive |
| `sentinel/evaluation` | 12 | 7,698 | archive (keep `KNOWN_CAUSAL_CHAINS`) |
| `sentinel/models/*` (13 subpackages) | 89 | 36,271 | archive (keep `hydrovit_backbone.py`; `fusion/` as a reference) |
| `sentinel/training` | 12 | 10,841 | archive |
| `sentinel/platform` · `dashboard` | 8 · 1 | 5,295 · 895 | archive |
| `sentinel/uncertainty` · `utils` | 2 · 3 | 367 · 299 | keep |
| `scripts/data_acquisition` · `scripts/data` | 30 · 2 | 10,402 · 637 | keep ~20 (§3.1 A) / archive rest |
| `scripts/preprocessing` | 18 | 7,689 | keep `build_toxigene_realgeo.py`; archive rest |
| `scripts/training` | 20 | 12,922 | keep `train_hydrovit_real.py`, `train_species_health_leakagefree.py`, `species_mechanism_*`; archive rest |
| `scripts/experiments/trackB_fusion` (+ subdirs) | 43 | 8,664 | keep the microcystin chain (~12 files); archive rest |
| `scripts/experiments/trackA_leadtime` (+ subdirs) | 41 | 9,636 | keep event builders (~9 files); archive detectors |
| `scripts/experiments` (top level) | 36 | 19,864 | keep `biomotion_gbm.py`, `biomotion_transfer.py`, `species_health_honest.py`, `weighted_conformal_deployment.py` (review first); archive rest |
| `scripts/experiments/{architecture, causal_lowperf, weak_benchmarks}` | 5 · 3 · 11 | 2,297 · 1,307 · 4,014 | keep `causal-real.py`; reference `fusion-perceiver.py`, `microbiome-taxpool.py`, `honest_hydrodensenet.py`; archive rest |
| `scripts/evaluation` | 15 | 6,651 | keep FIB ×3, `twin_reformulate.py`, `causal_real_eval.py` (GRQA PCMCI); archive rest |
| `scripts/benchmarks` | 7 | 2,760 | keep `benchmark_toxigene_realgeo.py`, `toxigene_pathway_batch.py`; archive rest |
| `scripts/domaingen` · `scripts/analysis` | 6 · 1 | 1,342 · 298 | keep |
| `scripts/figures` · `pipeline` · `deployment` · `_archive` | 4 · 2 · 4 · 4 | 1,603 · 564 · 2,772 · 404 | archive (keep `build_tierA_figures.py` data functions) |
| `tests` | 2 | — | keep `test_fusion_bugfixes.py`; replace `test_models.py` with split / leakage / data tests |

### 3.4 Tests (measured)

- The run used a scratch copy of the repo (never the repo itself) with `python3 -B -m pytest -p no:cacheprovider`.
  - Base miniconda Python **3.13.2**, torch **2.7.1+cu118**, pytest 9.0.2: **33 passed, 1 failed in 8 s.** The failure is `tests/test_models.py::TestEJOverlay::test_import` (`ModuleNotFoundError: h3`).
  - Without `scripts/`, the label-ban test is skipped instead (32 passed, 1 skipped, 1 failed). With `scripts/` present it passes, but importing `train_fusion.py` creates `checkpoints/fusion/` and `results/benchmarks/` directories.
  - The `sentinel2` env (Python **3.11.17**, created 2026-10-22 03:55, apparently for v2) gives the same failure.
- **Coverage:** the tests check only imports, shapes and three fusion regressions. Nothing tests splits, leakage, checkpoint loading or reported numbers (audit W10).

### 3.5 Dependencies & hygiene (measured)

- **The three dependency files disagree.**
  - `requirements.txt` (17 packages) lacks timm, transformers, rasterio, pystac-client, planetary-computer, GEOparse and biom-format.
  - `environment.yml` (env name `sentinel`, py 3.11) lacks pyarrow, requests, h5py, fastapi and streamlit.
  - `pyproject.toml` (py ≥ 3.10) lacks pyarrow, requests, fastapi and streamlit. The README names the env `physiformer`.
- **17 of 46 third-party top-level imports are declared nowhere:** pyproj, lightgbm, h3, xgboost, pydantic, PIL (pillow), gymnasium, torchdiffeq, statsmodels, tigramite, rapidfuzz, redbiom, combat, torch_geometric, joblib, cartopy, Bio (biopython).
- **Hard-coded paths:** 51 occurrences of `/home/bcheng/SENTINEL` in 44 files. On this machine that directory is an unrelated project, so all of them must become config.
- **Unsafe loads:** 73 `torch.load(..., weights_only=False)` calls in 49 files.
- **Pinned GPUs:** 27 `cuda:N` / `CUDA_VISIBLE_DEVICES` uses.
- **Unused config:** `configs/default.yaml` is read by no training script (audit).

### 3.6 `results/` files that keep-scripts need as inputs vs pure outputs

- **Inputs:** `results/benchmarks/*.json` (read by `transfer_gap_analysis.py`). `results/benchmarks/twin_eval_cache.npz` is read by `twin_reformulate.py` but is legacy (D25). `results/trackA_leadtime/**/residual_cache*.npz` are needed only to re-run the placebo audit.
- **Reference metrics for v2 regression tests** (pure outputs worth pinning):
  - `results/trackB_fusion/mc_fusion.json`, `mc_driver_fusion.json` (0.650 / 0.684 / 0.626 / 0.776 / 0.823), `mc_region_transfer.json` (≈ 0.49), `fusion_cycle2/modality_contribution.json`.
  - `results/benchmarks/hydrovit_real.json` (Temp test R² 0.7286), `microbiomenet_geographic_holdout_full_emp.json`, `toxigene_realgeo.json`, `species_health_leakagefree_retrained.json`, `twin_reformulate.json`, `disease_fib_*.json`, `biomotion_transfer.json`.
  - `results/causal/real_causal_results.json`, `results/audit_corrections_20260531.json`, `results/TRANSFER_GAP_MEMO.md`.
  - `review_artifacts/trackA/placebo_report.json`, `review_artifacts/satellite/doy_baseline_results.json`.
- **Everything else in `results/`** (case studies, exp1–exp20, conformal, robustness, prospective, compiled) is legacy output. Archive it.

---

## 4. Checkpoints (`SENTINEL-HF/checkpoints/` + 12 root `.pt`)

All 76 `.pt` files load with `weights_only=True`. 19 need the numpy allowlist: `aquassm.pt`, `hydrodensenet.pt`, `toxigene.pt`, sensor `aquassm_{expanded,final,full,real}`, waterdronenet `{best,ph,v4}`, toxigene `v6–v9b`, biomotion `phase1` / `phase2`, `foundation_best`, `mome_best`. The audit's statement that "all 12 published checkpoints load with weights_only=True" holds only with that allowlist.

| File(s) | Size | What it is (params; inputs) | Trained on / provenance | Audit status | v2 verdict |
|---|---|---|---|---|---|
| `aquassm.pt` = `sensor/aquassm_full_best.pt` = `sensor/aquassm_real_best.pt` (md5-identical) | 17.8 MB each | AquaSSM, 13.52 M; `ssm.input_proj` (256, 6): 6 channels; heads: mpp, anomaly_detector, anomaly_classifier, sensor_health; stored `head`, `val_auroc` 0.942 | 127 k USGS windows with corrupted normalization; threshold-rule labels (`results_full.json`, n_train 233,646) | legacy supervised head; normalization bug; reconstruction head untrained (C2, C3) | **DISCARD** |
| `sensor/aquassm_{pretrained,v2_mpp,expanded,final,v3,v4,anomaly,v2_anomaly}_best.pt` | 17.6–17.8 MB | same architecture | same data; **`anomaly` and `v2_anomaly` anomaly-head weights are NaN** **(measured)** | legacy | **DISCARD** |
| `aquassm_neon_ft.pt` | — | NEON self-supervised fine-tune used by Track-A | **not on disk anywhere** | trained on evaluation windows (C2) | n/a |
| `hydrovit.pt` (root, unique) | 160.5 MB | HydroViT v9: band attention (16, 10) + CNN + ViT-S/16 with 13-band patch embed (384, 13, 16, 16), 42.0 M | `paired_wq_v5.npz` (**fabricated**, quarantined) | retracted (C4, C10) | **DISCARD** |
| `satellite/hydrovit_mae_best.pt`, `satellite/hydrovit_wq_best.pt` | 155.8 MB each | ViT-S/16 with 13-band patch embed + multi-res / temporal stacks + WQ head, 40.8 M | MAE on **2,139** real S2 tiles (`results_real_mae.json`); WQ fine-tune mean R² 0.12 (`results_wq_finetuned.json`) | superseded | **DISCARD** (public S2 foundation weights are strictly better inits) |
| `hydrovit_wq_real.pt` | — | the honest 0.729 model (timm ImageNet ViT-S/16, 4 bands) | **not on disk anywhere**; `train_hydrovit_real.py` re-creates it in ~10 min (578 s on an A100 per the log) | genuine, but credited to the wrong model (C4) | re-train if needed |
| `hydrodensenet.pt` = `waterdronenet/waterdronenet_v4_best.pt` | 32.5 MB | HydroDenseNet V4, 8.45 M; 4-band stem + **pH encoder / FiLM** (needs sonde pH) | S2 4-band chips (offset uncorrected); training script not in repo | Lite misdescribed (C6) | **DISCARD** |
| `waterdronenet/waterdronenet_{best,ph,pretrain}_best.pt` | 251 / 84 / 266 MB | ViT-S/16 (4-band patch embed (384, 4, 16, 16)) + mu / sigma heads, 21.9 M; `pretrain` = MAE phase | training script absent; **log (`train_waterdronenet.log`) describes img_size 64 / 4.98 M params, not the shipped 224 px / 21.9 M model** | old WaterDroneNet (W12) | **DISCARD** (provenance broken) |
| `microbiomenet.pt` = `microbial/microbiomenet_v3_best.pt` | 44.6 MB | MicroBiomeNetV5-style sparse-gate transformer, 11.67 M; input 5,000 ASVs | random 70 / 15 / 15 split (`results_v5.json`: macro-F1 0.90) | retracted (C10) | **DISCARD** |
| `microbial/microbiomenet_{v2,v4,v5}_best.pt` | 44.6–44.8 MB | same family, 11.67–11.74 M | random splits | leaky | **DISCARD** |
| `microbial/microbiomenet_{best,emp,real}_best.pt` (emp = real, md5) | 496.9 MB each | 130.2 M; DNABERT-style sequence encoder + Aitchison layers + simplex ODE + zero gate | random split (`results_real.json` macro-F1 0.91) | evaluated only on the leaky split (C9) | **DISCARD** |
| `microbial/microbiomenet_geographic_holdout_best.pt`, `…_full_emp_best.pt` | 44.6 MB each | 11.67 M, 5,000-ASV input | KMeans geographic holdout (network macro-F1 0.30–0.40; LR on taxonomy beats it) | genuine but weak | **REFERENCE-ONLY baseline** (keep `…_full_emp_best.pt`) |
| `toxigene.pt` = `molecular/toxigene_v9b_best.pt`; `toxigene_v9_best.pt` | 979.9 MB each | 256.9 M dense MLP; `input_norm` (61,479 features); thresholds for 7 classes | matrix partly synthetic; lookup-table labels | retracted (C5, C9) | **DISCARD** (1.96 GB to prune) |
| `molecular/toxigene_v2…v8_best.pt`, `simpleMLP_v2…v4_best.pt`, `toxigene_{best,expanded,fullreal}_best.pt` | 2.4 MB – 318 MB | transformers / MLPs on 61,479- or 5,000-gene inputs; 0.62–83 M | synthetic / invalid-GEO matrices | legacy | **DISCARD** |
| `biomotion.pt` = `biomotion/biomotion_v2_best.pt`; `biomotion_expanded`, `*_phase1`, `phase1`, `phase2` | 11.4–34 MB | TrajectoryDiffusionEncoder + classifier, 2.98–3.01 M; 16-d × 200-step "trajectories" | ECOTOX rendered as pseudo-trajectories (`results.json`: 500 trajectories) | no behaviour data (C5) | **DISCARD** |
| `sentinel_fusion.pt` = `fusion/fusion_real_best.pt`; `sentinel_fusion_heads.pt` = `fusion/fusion_head_best.pt` | 11.6 / 0.4 MB | Perceiver-style fusion, 3.02 M (+ 99 k head); projection bank (satellite 384 → 256, …) | circular \|z\| > 3 label; no longer loads into the current class (audit C10) | retracted | **DISCARD** |
| `fusion/mome_best.pt`, `foundation/foundation_best.pt`, `foundation/foundation_pretrain_best.pt` | 82 / 33 / 62 MB | MoME 7.84 M; foundation 3.05 M / 5.65 M | hash-seeded random embeddings (audit Observations); val AUROC 0.54 / 0.65 | uninformative | **DISCARD** |
| `contrastive/*.pt` (5) | 0.4–1.2 MB | 128-d projection pairs, 99.8 k | `val_recall_at_1 = 1.0` at epoch 1 (trivial) | retracted (Observations) | **DISCARD** |
| `stream_gnn.pt` = `graph/stream_gnn_best.pt` | 25.2 MB | GNN 1.32 M (+ propagator) on `nhdplus_graph.json` | label = threshold of its own input; val AUROC 1.0 | circular (C5) | **DISCARD** |
| `digital_twin.pt` = `twin/twin_best.pt`; `twin/twin_phase1_best.pt` | 1.3 MB each | twin ODE + assimilator + corrector, 342 k | `_generate_minimal` synthetic fallback (300 / 60 / 60 samples) | synthetic-trained (C5) | **DISCARD** |
| `species_health.pt` = `biology/species_health_best.pt` | 3.1 MB | SentinelSpeciesHealthIndex, 798 k; 6 species embeddings | circular inputs (health R² 0.9995) | retracted (C10) | **DISCARD** |
| `biology/species_health_leakagefree_best.pt` | 3.1 MB | same network; stores `feature_names` (lat, lon, abslat, elev, log_drainage, non-target community, effort, huc2, wq summaries); best val macro-AUROC 0.951 | leak-free features, hash split (91.7% of test sites within 10 km of train, W2) | genuine but leaky by proximity | **REFERENCE-ONLY baseline** |
| `waterborne_disease.pt` = `biology/disease_forecast_best.pt` | 1.8 MB | 4 pathogen heads (cyanotoxin / vibrio / naegleria / schistosomiasis), 470 k; env input 262-d | old 4-pathogen model | retracted (C10) | **DISCARD** |
| `biology/arg_surveillance_best.pt` | 26.2 MB | 6.87 M; 5,000-OTU community mapper; epoch 1, val R² −0.01 | pseudo-labels (W: "Failed to load EMP data") | invalid (C5) | **DISCARD** |
| 36 `checkpoints/**/results*.json` | < 0.1 MB | per-run metric dumps | — | mostly retracted | archive with the checkpoints |

**Missing but referenced.** 32 of the 82 distinct `.pt` name literals in code (two are f-string fragments) do not exist anywhere under `/home/bcheng/STOCKHOLM`, including:
- `hydrovit_wq_real.pt`, `aquassm_neon_ft.pt`;
- `hydrovit_wq_v3` / `v6` / `v7` / `v8` / `v9.pt`, `hydrovit_real_mae.pt`, `hydrovit_v2_best.pt`;
- `aquassm_v2*.pt`, `aquassm_best.pt`, `aquassm_final.pt`, `aquassm_real_pretrained.pt`;
- `fusion_real_chl.pt`, `fusion_v2_best.pt`, `twin_phase2_best.pt`, `final_calibrated.pt`;
- the five `*_embeddings.pt` caches, `best_contrastive.pt`, `cached_embeddings_v2.pt`.

**Initialization candidates.** None recommended.
- The only honest satellite result came from **public timm ImageNet ViT-S/16** weights. The local MAE backbones were pretrained on 2,139 tiles or have broken provenance.
- No local time-series, microbial or molecular checkpoint was trained on a valid split with clean inputs.
- For v2, start from public foundation weights. For satellite, use multispectral S2-pretrained models. For drone RGB + NIR, use an ImageNet / RS backbone with a 4-channel stem. For sensors, use public time-series foundation models.
- Keep the two REFERENCE-ONLY checkpoints to report "previous model" rows.
- **Pruning:** archiving everything except those two would free ~7.4 GB of checkpoint bytes (and the matching ~7 GB in `.git/lfs`). Deduplicating alone frees 1.64 GB.

---

## 5. Suggested local layout for v2 (no action taken)

- **`data/raw/<source>/`**: write-once raw pulls, each with `MANIFEST.json` (URL, query, timestamp, sha256, row counts). USGS IV, NEON, S2 (uint16 + item metadata), WQP, ECOTOX, EMP, GEO, GHCN, StreamCat / NHDPlus.
- **`data/interim/`**: copies of the REUSE items above, frozen with sha256:
  - D8 (microcystin benchmark) and D9–D12 (raw WQP);
  - D2 (S2 chips, plus an offset-correction flag);
  - D4–D6 (event catalogs);
  - D7 (COMID map);
  - D14 (BioData);
  - D16 (FIB raw);
  - D17 (EMP taxonomy);
  - D19 (GEO manifest);
  - D23 (KEGG).
- **`legacy/`**: archived code and checkpoints. The HF mirror can stay as the archive of record.

---

## Appendix A. Site lists (from the manifests)

- **NEON (34)**
  - Episodes (15): ARIK, BARC, BIGC, BLWA, COMO, CRAM, GUIL, KING, LECO, LIRO, MAYF, MCDI, PRIN, PRLA, SUGG.
  - Controls (17): BLDE, BLUE, CARI, CUPE, FLNT, HOPB, LEWI, MART, MCRA, OKSR, POSE, PRPO, REDB, TECR, TOMB, WALK, WLOU.
  - Dropped: SYCA, TOOK.
- **USGS DO episodes (17):** 01649500, 01656903, 040851385, 04087170, 07288500, 073802512, 07381600, 08049500, 08062500, 08065350, 08068500, 08181500, 08181800, 08211200, 14206950, 453004122510301, 453030122560101.
- **USGS DO controls (22):** 01632900, 01673000, 02035000, 03512000, 04024000, 04027000, 04067500, 05406479, 06610505, 07374000, 11447650, 11455315, 11455508, 11465240, 12098700, 12100490, 12101100, 12101102, 14202980, 14209710, 14210000, 453040123065201.
- **Salinization:** 30 episodes + 11 controls in `trackA_salt/salt_manifest.json` (e.g. 01302020, 01389005, 01400500, …, 06893620).
- **AquaSSM stations (381):** `sensor/full/preprocessing_stats.json → stations` (site-number prefixes: 01: 95, 02: 99, 03: 35, 04: 50, 08: 38, 11: 35, others 29). The chip set (D2) covers 379 stations, 378 of them shared.

## Appendix B. Scratch scripts used for the measured numbers

These are in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/inventory/` (session scratch, not versioned):
- `npz_headers.py`: npz shapes without loading.
- `sat_stream.py`, `sat_meta.py`: chip statistics, offset and water fraction.
- `sensor_npz.py`: sensor sample.
- `mc_offset_check.py`: era confound.
- `pq_summary.py`: table profiles.
- `ckpt_scan.py`, `ckpt_scan2.py`, `ckpt_helpers.py`: checkpoint scan; outputs in `ckpt_scan.json`.
- `deps_scan.py`: dependency scan.
- `repo_copy/`: the test run.
