# Datasheet — Sentinel-2 L2A chip stores (`s2_chips`) and the STAC imagery layer

| | |
|---|---|
| **What** | 64 × 64 chips at 10 m (640 m × 640 m) around in-situ points: 12 L2A bands + AOT + SCL + JRC water occurrence, one store per campaign |
| **Imagery sources** | Element 84 Earth Search STAC `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-c1-l2a` (primary); Microsoft Planetary Computer STAC `https://planetarycomputer.microsoft.com/api/stac/v1`, collection `sentinel-2-l2a` (fallback, anonymous SAS signing); JRC Global Surface Water v1.3 occurrence (`jrc-gsw`, Planetary Computer) |
| **Licences** | Copernicus Sentinel data — free, full and open (attribution "contains modified Copernicus Sentinel data [year]"); JRC GSW — free of charge, without restriction of use (cite Pekel et al. 2016, doi:10.1038/nature20584) |
| **Retrieved** | 2026-10-22 onward (per-chip retrieval is implicit in the run logs; scene catalogues cached in `$SENTINEL_DATA/interim/s2_scene_catalog/<campaign>/`) |
| **Code** | `sentinel/data/sources/stac.py` (search, offset logic, chip reads, JRC, Landsat, NAIP), `sentinel/data/chips.py` (store + features), `sentinel/data/matchups.py` (planning, SCL screening, extraction engine) |
| **Stores** | `$SENTINEL_DATA/processed/s2_chips/{aquacorr, mc_benchmark, wqp, naip_teacher}/` |
| **Manifests** | `s2_chips_aquacorr`, `mc_benchmark_v2` (includes its store index), `s2_wqp_matchups`, `naip_matchups` |
| **Tests** | `tests/test_stac_chips.py` (no network; fixtures are trimmed real STAC items / metadata XML) |

## Store format
```
<store>/layers.json            layer names, dtype (int16) and value semantics
<store>/shards/shard-NNNNN.npz one zip member per chip (key = chip_id), deflate; <= 2048 chips and << 2 GB per shard
<store>/index/part-NNNNN.parquet one row per *request* (ok and failed); written after its shard (atomic renames)
<store>/index.parquet          consolidated index (latest row per request_key)
```
Read with `sentinel.data.chips.ChipStore(root).load(chip_id)`; reflectance with `stac.to_reflectance(chip[i])`.
Resume: `ChipWriter` skips request keys already in the index (failures too, unless `--retry-failed`); shards without
an index part (crash) are deleted on restart and their chips re-extracted.

## Layers (int16, `chips.S2_STORE_LAYERS`)
| # | Layer | Native | Value |
|---|---|---|---|
| 0–11 | B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 | 60/10/10/10/20/20/20/10/20/60/20/20 m | **harmonised DN**: raw DN + `offset_dn`; reflectance = value × 1e-4; nodata = −32768 |
| 12 | AOT | 20 m (ES C1) / 10 m (MPC) | raw DN, AOT(550 nm) = value × 1e-3; nodata = −32768 |
| 13 | SCL | 20 m | Sen2Cor class 0–11 (0 no data, 3 cloud shadow, 6 water, 8/9 cloud, 10 cirrus, 11 snow) |
| 14 | JRC | 0.00025° | JRC GSW occurrence 0–100 % (1984–2020) reprojected (nearest) onto the chip grid; 255 = no data |

**Footprint rule (fixes v1 defect).** All layers are on one 10 m grid anchored on the MGRS tile origin. 20 m and
60 m bands are upsampled by nearest-neighbour pixel replication on the nested grids (a 20 m pixel → exactly 2 × 2
10 m pixels, a 60 m pixel → 6 × 6), so every layer covers the same 640 m footprint and keeps its original values
(no interpolation across shorelines). The point lies in chip pixel (32, 32); its sub-pixel position is stored
(`point_subpix_y/x`). v1 read 20 m bands as native 64 × 64 windows (1.28 km, 4× the area of the 10 m bands).
Tested: `test_replication_maps_each_10m_pixel_to_its_native_pixel`, `test_all_layers_share_the_chip_footprint`.

## Radiometry: BOA offset (fixes v1 defect) — `stac.s2_offset_decision`
* Processing baseline ≥ 04.00 (operational from 2022-01-25) stores reflectance as (DN + BOA_ADD_OFFSET)/10000 with
  BOA_ADD_OFFSET = −1000. The offset is decided **per item**, never by date:
  1. `earthsearch:boa_offset_applied = true` (Earth Search legacy collection) → offset 0: the DN were already
     corrected although their `raster:bands` still advertise offset −0.1 (double-correction trap).
  2. Earth Search C1: `raster:bands` offset/scale (−0.1/1e-4 = −1000), cross-checked with the baseline rule.
  3. Planetary Computer: no offset metadata; BOA_ADD_OFFSET is read from each item's product XML
     (MTD_MSIL2A.xml) and must agree with the baseline rule.
  4. Unknown/implausible baselines (e.g. Earth Search legacy "00.01") or disagreeing evidence → item skipped
     (`status = offset_inconsistent`).
* Verified on real pixels (2023-08-27, T17TKG): Earth Search C1 (05.09, −1000 applied) and the legacy item with
  `boa_offset_applied=true` (0 applied) give identical harmonised DN (median |Δ| = 0 in all 12 bands); Planetary
  Computer 05.09 / 05.10 products of the same acquisition differ by 0–6 DN (processing runs).
* Per-chip sanity statistics: `qc_refl_p01_min` (minimum over bands of the 1st-percentile reflectance of SCL-valid
  pixels; ≈ −0.1 after a double correction) and `qc_water_b11_median`; `offset_suspect` ∈ {'', 'double', 'missing'}.
  Tested both cases (`test_offset_*`, `test_harmonise_raw_and_precorrected_agree`, `test_offset_sanity_*`).

## Source traps found 2026-10-22 (measured)
* **Earth Search C1 gaps:** at Lake Erie no `sentinel-2-c1-l2a` items for 2015-06 → 2018-01 and 2022-01 → 2022-11
  (3 items in 2022-03); C1 baselines 05.00 (2018–2021), 05.09–05.13 (2022-12 →). These dates fall back to MPC.
* **MPC `datetime` is the datatake start**, not the granule sensing time (13 min earlier on T17TKG 2023-08-27). The
  exact time is read from each MPC item's granule XML (`SENSING_TIME`). For MPC products of baseline 02.xx this
  `SENSING_TIME` itself precedes the tile overpass by up to ~10 min (compared with ACOLITE/C1 times); ES C1 item
  times equal the granule `SENSING_TIME` (< 1 s).
* **MPC duplicates:** 2022–2023 hold original (04.00 / 05.09) and reprocessed (05.10) items of the same acquisition
  and tile; the highest baseline is kept (`matchups.dedupe_mpc_reprocessing`).
* **Mixed baselines across years are unavoidable** (02.12/03.00 pre-2018 and 2021 on MPC; 05.00 C1; 04.00/05.10 in
  2022): every chip records `stac_source`, `collection`, `processing_baseline`; models should test for residual
  dependence on baseline/year.
* Earth Search legacy `sentinel-2-l2a` is not used (mixed `boa_offset_applied`, duplicate `_0_`/`_1_` items,
  baseline "00.01" items of unclear provenance).

## Per-chip metadata (index columns)
Scene: `item_id`, `stac_source`, `collection`, `platform`, `mgrs_tile`, `sensing_time_utc` (exact granule time),
`item_datetime`, `processing_baseline`, `offset_dn`, `offset_source`, `offset_already_applied`, `offset_consistent`,
`offset_xml_verified`, `scene_cloud_cover`, `sun_zenith`, `sun_azimuth`, `view_zenith`, `view_azimuth` (ES C1: item
mean incidence angles; MPC: mean of per-band granule angles), `relative_orbit`, `product_uri`, `epsg`.
Chip: `chip_crs`, `chip_left`, `chip_top`, `chip_res_m`, `point_row/col`, `point_subpix_y/x`, `chip_nodata_frac`.
SCL: class fractions for the centre 3 × 3 (`scl_c3_*`), centre 7 × 7 (`scl_c7_*`) and whole chip (`scl_chip_*`),
plus `*_n_water` (SCL 6) and `*_n_valid` (not 0/1/3/8/9/10/11). Centre windows are on the 10 m grid (3 × 3 = 30 m,
7 × 7 = 70 m ≈ 3 × 3 native 20 m SCL pixels).
Water: `jrc_occ_point`, `jrc_occ_max_60m`, `jrc_dist_water_m` (distance to the nearest JRC pixel with occurrence
≥ 50 %; 0 on water), `jrc_dist_land_m` (distance to the nearest pixel < 50 % = distance to shore for water points),
`jrc_water_ok` (occurrence ≥ 50 % at the point or within 60 m), `jrc_c3_median`, `jrc_chip_water_frac`.
Spectra: `rw{3,7}_<band>` median reflectance over SCL-water pixels, `rv{3,7}_<band>` over SCL-valid pixels,
`aot_c{3,7}`.

## Extraction protocol
1. Planning per UTC date: one STAC query per date and region (MultiPoint `intersects` for ≤ 40 points, else bbox),
   Earth Search C1 first, then MPC for points C1 does not cover that day; footprint containment via the item's data
   geometry; result cached per date.
2. SCL screening (optional, used for WQP / microcystin / NAIP-teacher candidates): one SCL read per scene; a
   request is *usable* if its centre pixel has data and the centre 7 × 7 has ≥ 5 SCL-valid pixels.
3. Full chips scene by scene: each COG opened once per scene, all windows for that scene read from it.
   Concurrency: ≤ 8 COG/XML/JRC reads in total across all running jobs (one shared I/O pool per job; job pools sized
   so the sum is 8), ≤ 4 concurrent STAC API requests per host per job, GDAL HTTP retries (5) + job-level retries
   with backoff; User-Agent `SENTINEL-v2 research (public-data download)`.
4. Measured throughput (this server, 6 read workers): ≈ 2,600 scene tasks/h; ≈ 9,000 chips/h when scenes hold
   several points. A 14-layer chip costs ~14 COG reads (~0.6 s each, latency-bound: ~100 ms RTT to S3 us-west-2 and
   Azure West Europe). Compressed size ≈ 40–60 kB per chip.

## Robustness (incident 2026-10-22)
The first AquaCorr run deadlocked at 07:40 (all threads in futex waits, CLOSE-WAIT sockets to Azure blob storage
after HTTP 503s). Since then: curl low-speed abort (`GDAL_HTTP_LOW_SPEED_TIME=60`, `GDAL_HTTP_CONNECTTIMEOUT=30`),
per-read future timeouts (900 s), a progress watchdog that flushes and exits (code 3) after 20 min without progress,
SIGTERM = flush-then-exit, and a shell restart loop (`processed/aquacorr_global/run_chips_loop.sh`). Restarts are
lossless for written shards; at most the in-memory buffer (< 2,048 chips) is re-extracted. A consistency check after
the restarts found every indexed chip present in its shard and no duplicate request keys.

## Other readers (infrastructure, no products yet)
* **Landsat C2 L2** (`stac.read_landsat_chips`): 32 × 32 at 30 m, SR B1–B7 + ST (lwir11) + QA_PIXEL + QA_AEROSOL as
  raw uint16 with `raster:bands` scale/offset recorded (SR: 2.75e-5, −0.2; ST: 0.00341802, 149 K); QA_PIXEL bits
  decoded by `stac.qa_pixel_flags` (tested). Smoke-tested 2026-10-22 on LC09 012/030 2023-08-26.
* **NAIP** (`stac.read_naip_chips`, `stac.naip_scene_stats`): see `naip_matchups.md`.

## Rebuild
See the per-campaign datasheets (`aquacorr_global.md`, `mc_benchmark_v2.md`, `s2_wqp_matchups.md`,
`naip_matchups.md`). Unit tests: `python -B -m pytest tests/test_stac_chips.py -q`.
