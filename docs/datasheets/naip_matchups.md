# Datasheet — NAIP × WQP matchups with Sentinel-2 teacher chips (`naip_matchups`) and M7 hazard label sets

| | |
|---|---|
| **Imagery** | USDA NAIP 4-band (R, G, B, NIR) Cloud-Optimized GeoTIFFs on Microsoft Planetary Computer (`naip`, 2010–2023; GSD 1.0 m ≤ 2017, 0.6 m 2018+, 0.3 m in some 2023 states); S2 L2A teacher chips (`s2_chips.md`) |
| **In situ** | WQP surface Secchi, turbidity, TSS (+ SSC), chl-a samples (`wqp_surface_samples`, same rules as `s2_wqp_matchups.md`) |
| **Licences** | NAIP: public domain (USDA FSA; the Planetary Computer STAC licence field reads "proprietary" — public-domain status per USGS/LILA; confirm with FSA before redistributing imagery); WQP public; Copernicus Sentinel data free, full and open |
| **Code** | `sentinel/data/naip.py`, `sentinel/data/sources/stac.py` (`search_naip`, `read_naip_chips`, `naip_scene_stats`), `scripts/data/build_naip_matchups.py`; hazard sets `scripts/data/download_naip_hazard_labels.py` |
| **Outputs** | `$SENTINEL_DATA/processed/naip_matchups/{matchups.parquet, naip_scene_stats.parquet, coverage.json}`; NAIP chips `$SENTINEL_DATA/processed/naip_chips/wqp/`; S2 teacher chips `$SENTINEL_DATA/processed/s2_chips/naip_teacher/` |
| **Manifests** | `naip_matchups`; hazard sets `hazard_labels_{marida,mados,dekerf,chesapeake_landcover,floodnet}` |

## Matchup rules
* States: Minnesota + FL, WI, MI — the 3 states with the most lake/reservoir Secchi + chl-a results 2010–2023 in the WQP
  summary service (MN 391,855; FL 298,429; WI 135,334; MI 96,089), passed as `--states MN FL WI MI`;
  `--national` for all states.
* NAIP items found per site (MultiPoint STAC queries, cached); site ↔ DOQQ by footprint containment; a sample matches
  a DOQQ when |sample date − NAIP acquisition date| ≤ 1 day (NAIP STAC `datetime` = acquisition date; no time of
  day; a DOQQ may be a mosaic of flight lines and only one date is recorded).
* Sites must pass the JRC water test (occurrence ≥ 50 % at or within 60 m of the site).

## NAIP chips and radiometry
* 256 × 256 pixels at native GSD (77 m at 0.3 m, 154 m at 0.6 m, 256 m at 1 m) centred on the site, uint8 DN in
  NAIP band order R, G, B, NIR (`layers.json`); the DOQQ CRS/transform of each chip is in the index.
* **NAIP is 8-bit, vendor colour-balanced per state-year and not reflectance.** DN are stored unchanged; per-DOQQ
  statistics (`naip_scene_stats.parquet`: per-band mean, std, percentiles 1–99 over all non-zero pixels of a
  16×-decimated read, valid and dark-NIR fractions, dark-NIR (NIR < G, water-like) band medians) support per-scene
  normalisation or S2-anchored calibration later.
* Chip features: centre 9 × 9 / 31 × 31 per-band median DN, dark-NIR fractions, fill and saturation fractions.

## S2 teacher chips
For each (site, DOQQ) matchup with a NAIP date ≥ 2015-06-23, every S2 acquisition within ±1 day of the NAIP date
is SCL-screened; the nearest usable one (≥ 5 SCL-valid pixels in the centre 7 × 7) gets a full S2 chip
(`s2_chip_id`, `dt_days_s2_naip`, `s2_sensing_time_utc`).

## Coverage (run 2026-10-24 on the raw WQP WQX3 download `raw('wqp')`, manifest `wqp_raw`; `processed('wqp_discrete')` did not exist yet)
* MN, FL, WI, MI; NAIP years 2010–2023: **10,672 samples** within ±1 d of a NAIP acquisition at **6,195 (site, DOQQ)
  pairs** — Secchi 5,547, chl-a 2,770, turbidity 1,296, TSS 1,037, SSC 22 (lakes/reservoirs: Secchi 4,843, chl-a
  1,817, turbidity 486, TSS 385).
* By state: MN 4,585 (Secchi 2,839, chl-a 1,168), FL 3,996 (chl-a 1,414, Secchi 1,141), WI 1,586 (Secchi 1,128),
  MI 505 (Secchi 439). By NAIP year: 2010 1,618; 2013 1,337; 2015 1,167; 2017 1,148; 2019 1,053; 2021 1,663;
  2023 1,442; even years (partial state cycles) 80–353.
* 6,195 NAIP chips (all ok); 3,907 samples have an S2 teacher chip within ±1 d of the NAIP date (S2 exists only from
  mid-2015).
* Input: 2,561,062 surface samples of the 4 targets at 66,566 sites in these states (40,139 pass the JRC water test).
* Not yet run nationally (`--national`); NAIP chips 776 MB (≈ 125 kB each), S2 teacher chips 77 MB.

## Hazard / segmentation label sets (raw, `$SENTINEL_DATA/raw/hazard_labels/`)
| Set | Content | Licence | Status |
|---|---|---|---|
| MARIDA (Zenodo 5151941) | S2 patches 256 × 256, 15 classes incl. sediment-laden water, foam, marine debris; no oil | CC BY 4.0 | downloaded: MARIDA.zip 1.16 GB (md5 ok), 4,528 members, 4.7 GB uncompressed: `patches/<scene>/*.tif` + `_cl` class + `_conf` masks, shapefiles |
| MADOS (Zenodo 10664073) | S2 marine pollution incl. oil spills / look-alikes, 174 scenes | CC BY 4.0 | downloaded: MADOS.zip 4.04 GB (md5 ok), 56,760 members (5.8 GB): per-scene band TIFFs + class masks (ACOLITE L2R) |
| Oil Spill Drone, De Kerf et al. 2024 (Zenodo 10555314) | drone RGB, 1,268 images, oil / water / other (CamVid format) | CC BY 4.0 | downloaded: dataset.zip 1.14 GB (md5 ok): 1,268 JPG images + 1,268 PNG masks, train/val/test |
| Chesapeake Land Cover (LILA) | 1 m land-cover labels (class 1 = water) + NLCD for 732 tiles; NAIP 'naip-new' for val/test tiles | CDLA-Permissive-1.0 | downloaded (15 GB): 732 `_lc.tif` + 732 `_nlcd.tif` label tiles, 150 `_naip-new.tif` (val/test); GCS md5 in manifest |
| FloodNet supervised v1.0 | UAV RGB, 2,343 images, 10 classes incl. water / flooded road / building | CDLA-Permissive-1.0 | downloaded: FloodNet.zip 13.06 GB (Dropbox, no checksum published), 7,046 members (`FloodNet-Supervised_v1.0`, `ColorMasks-FloodNetv1.0`) |
| LADOS | drone RGB oil/emulsion/sheen | article CC BY 4.0; data only on Roboflow Universe (account) | not downloaded |

## Rebuild
```bash
python -B scripts/data/build_naip_matchups.py --source raw --states MN FL WI MI   # as run; --source processed once wqp_discrete exists
python -B scripts/data/download_naip_hazard_labels.py --sets marida mados dekerf chesapeake floodnet
```

## 2026-11-08: national extension (M7)
The national build (`naip_matchups_national`, `naip_matchups_national.md`) appended chips to the stores
`processed/naip_chips/wqp/` and `processed/s2_chips/naip_teacher/`. All 6,195 NAIP and 2,130 S2 chips of this
4-state build keep their chip ids and shards (checked by request key). The 4-state `matchups.parquet`,
`coverage.json` and `naip_scene_stats.parquet` are byte-identical to the original (sha256 `7bd16bf0…`, `e342d13a…`,
`d506db02…`). The manifest `naip_matchups` was rewritten to list the current superset NAIP index together with the
pre-national indices, kept in `processed/naip_matchups/snapshot_4state_20261024/` (NAIP index sha256 `535dca20…`, as
in the original manifest). The original manifest's file list is kept under `stats.pre_national_manifest`.
