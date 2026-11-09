# Datasheet — `nars_lakes` (EPA National Lakes Assessment 2007, 2012, 2017, 2022)

| | |
|---|---|
| **Location** | `$SENTINEL_DATA/processed/nars_lakes/` (Parquet, zstd) · manifest `manifests/nars_lakes.json` · raw `raw/nars/nla07|nla12|nla17|nla22/` (manifest `nars_raw`) |
| **Source** | U.S. EPA NARS data hub, <https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys> |
| **License** | U.S. Government work (EPA); no license statement on the hub page (unverified) — cite EPA NARS |
| **Retrieved** | 2026-10-22 (same download as `nars_streams`; the 2007–2022 population-estimate file was copied from the research cache after its size matched `Content-Length`; NLA 2007 water quality extracted from the official `nla2007_alldata.zip`) |
| **Builder** | `sentinel/data/sources/nars.py`, `sentinel/data/bio_nars.py` (`build_resource(..., "lakes")`) |
| **Synthetic data** | none |

## Coverage

| Survey | Visits | Lakes (SITE_ID) | Revisits | Dates | Benthic visits / taxa | Zooplankton visits / taxa | Phytoplankton visits / taxa | Profiles (visits) |
|---|---|---|---|---|---|---|---|---|
| `nla07` | 1,252 | 1,157 | 95 | 2007-05-08 … 2007-10-18 | 1,132 / 558 | 1,243 / 131 | 1,249 / 1,543 | 1,245 |
| `nla12` | 1,231 | 1,131 | 100 | 2012-05-02 … 2012-09-27 | 1,200 / 579 | 1,229 / 499 | 1,230 / 884 | 1,200 |
| `nla17` | 1,210 | 1,113 | 97 | 2017-05-02 … 2017-10-10 | 1,191 / 597 | 1,210 / 378 | 1,187 / 386 | 1,207 |
| `nla22` | 1,225 | 1,129 | 96 | 2022-05-09 … 2022-09-29 | 1,071 / 528 | 1,076 / 398 | 1,225 / 561 | 1,221 |
| **total** | **4,918** | | | | 177,583 rows / 829 names | 99,664 rows / 662 names | 124,662 rows / 2,406 names | 44,108 depth rows |

Visits are the sampled rows of each cycle's site-information file (`SITESAMP`/`SAMPLED` = Y/YES; the 2012–2022 site files also list ~1,400–4,500 unsampled frame lakes, which are excluded). HUC8 and COMID are present for every visit; NLA 2007 has no UID, so its `visit_id` is `nla07-<SITE_ID>-<VISIT_NO>`.

## Tables

Same schema as `nars_streams` (see that datasheet for column definitions): `visits`, `benthic_counts`, `zooplankton_counts`, `phytoplankton_counts`, `taxa_source`, `chemistry_long`, `chemistry_wide`, `condition_long`, `condition_wide`, `phab_long`, `phab_key_wide`, plus:

| File | Grain | Notes |
|---|---|---|
| `zooplankton_counts.parquet` | visit × sample type × taxon | `count`, `count_300`, `biomass`, `biomass_300`, `density`, `tow_volume`; sample types differ by cycle (2007 `ZOCN/ZOFN` coarse/fine nets; later ZOCN/ZOFN/ZONW …) |
| `phytoplankton_counts.parquet` | visit × analysis × taxon | `count` (natural units), `biovolume`, `density`, `src_algal_group`; **NLA 2007 soft-algae and diatom counts are separate analyses** (`sample_type` = `PHYT_SOFTALGAE` / `PHYT_DIATOM`) — never sum them |
| `zooplankton_counts_qc_duplicates.parquet` (866), `phytoplankton_counts_qc_duplicates.parquet` (2,466) | as above | NLA 2007 field-duplicate samples (`SAMPLE_CATEGORY` D, 73–74 visits; X) kept out of the main tables |
| `profiles.parquet` | visit × depth | `depth_m`, `temperature_c`, `do_mgl`, `ph`, `conductivity_uscm`, `metalimnion` flag, `flag` — the DO profiles for hypoxia work |

### Chemistry (visits with a value in `chemistry_wide`)

| | 2007 | 2012 | 2017 | 2022 |
|---|---|---|---|---|
| TP / TN | 1,252 / 1,252 | 1,230 / 1,230 | 1,199 / 1,203 | 1,221 / 1,224 |
| NH4-N / NO3-N | 1,252 / 1,252 | 1,229 / 1,178 | 1,138 / 1,131 | 1,188 / 771 |
| Cl / SO4 | 1,252 / 1,252 | 1,230 / 1,230 | 1,193 / 1,187 | 1,224 / 1,205 |
| conductivity / pH | 1,252 / 1,252 | 1,230 / 1,230 | 1,186 / 1,187 | 1,191 / 1,191 |
| turbidity / DOC / chl-a | 1,252 / 1,252 / 1,247 | 1,189 / 1,230 / 1,225 | 1,178 / 1,199 / 1,210 | 1,224 / 1,224 / 1,220 |
| atrazine | — | 1,228 | 473 | 545 |
| Secchi depth (m) | 1,183 | 1,155 | 1,070 | 1,209 |

**Algal toxins** (`chemistry_long`, analyte / visits / detected / censored): microcystin 2007 1,252 / 391 / 861; 2012 1,230 / 1,230 / 0 (710 flagged `below_mdl`); 2017 1,186 / 319 / 867; 2022 1,225 / 601 / 624. Cylindrospermopsin 2017 1,186 / 97 / 1,089; 2022 1,225 / 116 / 1,109. 2012 also has littoral microcystin and chl-a (`microcystin_littoral`, `chla_littoral`); 2017 has the bottle-comparison analyte `microcystin_legacy_bottle` (MICZ = "microcystin in legacy bottle type").

**NLA 2007 full water chemistry** is not listed on the hub as a separate file: it is `NLA2007_WaterQuality_20091123.csv` inside the official "NLA 2007 All Data" zip, which the downloader fetches and extracts (`nars.INCLUDE_ARCHIVES`). Its mg/L columns are used for ions (`*_PPM`); NH4 and NO3 are mg/L **as N** (verified: ueq/L × 14.007/1000 reproduces `NH4N_PPM` and `NO3N_PPM`). Flags such as `<RL (0.02)` mark values below the reporting limit that are still reported (not censored). Field duplicates (`SAMPLE_CATEGORY` D, 74 rows) are dropped. **Secchi depth** (`secchi_m`) = mean of disappearance and reappearance depths (given as SECMEAN/SECCHI in 2007/2012, computed in 2017/2022); `CLEAR_TO_BOTTOM` is kept as the flag (a lower bound).

### Condition labels

`condition_long` holds the cycle-specific MMI/condition files **and** the harmonised 2007–2022 population-estimate file (`nla2007-2022_data_forpopestimates_indexvisits_probsites_0.csv`, index visits of probability lakes), which applies the 2017 benthic and zooplankton MMIs to every cycle (`BENT_MMI_COND_2017`, `ZOOP_MMI_COND_2017`) plus trophic state, TP/TN/chl-a condition, DO class, microcystin/cylindrospermopsin benchmarks, atrazine, drawdown and lake-habitat classes. `source_file`/`source_column` say which version a label is; compare cycles only within one version.

## Processing and QC

As for `nars_streams` (missing codes, primary samples only, censoring rules, canonical units, string IDs, UID reconciliation, orphan removal), plus:

- **`ppb` = ug/L** for ELISA toxin and atrazine results (aqueous ppb w/v ≡ ug/L in dilute water; NARS metadata documents these as ppb).
- **NLA 2007 microcystin substitution undone.** Lakes below the 0.1 ug/L reporting limit were published as 0.05 ug/L (RL/2) with `MCYST_PRES = "1:LESS THAN RL (0.1 ug/L)"`. These 861 rows are stored as `censored = True`, `value` NaN, `detection_limit` 0.1, with the published 0.05 recorded in `qualifier`.
- **NLA 2012 values below the MDL** (e.g. microcystin < 0.1 ug/L reported as numbers, including 288 zeros) are kept as published and flagged `below_mdl`; they carry no non-detect flag in the source, so they are not converted to censored values.
- `nla2017_zoopMMI.csv` ships with a corrupted first header (`7`) holding the UIDs; the parser detects it.
- Dropped orphan rows (visit not in the sampled site information): 129 phytoplankton rows (5 visits, 2012), 38 atrazine rows (2017), 19,028 phab rows (38 visits, 2012).
- The NLA 2017 physical-habitat file is per station (not per visit) and is not melted into `phab_long`.

## Known issues and traps

- **HAB circularity:** cyanobacterial taxa in `phytoplankton_counts` produce the microcystin/cylindrospermopsin they would be used to predict — exclude producing taxa or predict toxins from environment only (docs/research/04 §E.3).
- **Index circularity:** benthic/zooplankton MMI classes are functions of the corresponding counts.
- Zooplankton and phytoplankton taxonomic resolution and lab methods changed strongly between 2007 and 2012 (e.g. 131 vs 499 zooplankton names); use `processed/taxonomy_harmonised` and a cycle indicator.
- 2022 benthic sampling covers fewer visits (1,071) than other assemblages.
- Values are as published (no range clipping): saline lakes reach `cond_us_cm` 64,810 and `sulfate_mg_l` 47,325; microcystin max 355 ug/L.
- Population weights (`design_weight`) are for design-based estimates; model-based work should not treat them as precision weights.

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_nars.py
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/build_nars.py --resource lakes
```
