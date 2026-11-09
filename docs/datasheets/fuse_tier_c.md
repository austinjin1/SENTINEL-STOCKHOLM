# Datasheet — `fuse_tier_c` (SENTINEL-Fuse Tier C: microcystin rows with co-located modalities)

| | |
|---|---|
| **Location** | `$SENTINEL_DATA/processed/fuse_tier_c/` · manifest `manifests/fuse_tier_c.json` |
| **Built** | 2026-10-25 by `scripts/data/build_fuse_tiers.py c` (`sentinel/data/fuse_tier_c.py`) |
| **Inputs** | manifests `wqp_discrete` (WQX 3.0), `nars_lakes` (NLA 2007/2012/2017/2022), `mc_benchmark_v2`, `s2_wqp_matchups`, `forcings_daily` |
| **License** | US public domain (WQP — cite providers and Read et al. 2017; EPA NARS); Copernicus Sentinel data free and open |
| **Synthetic data** | none |

## Files
| File | Content |
|---|---|
| `rows.parquet` | one row per (site, local sample date): ids (`row_id`, `site_id`, `source` wqp/nla, `organization`, `state`, `huc8/huc4/huc2` strings, `lat/lon`, `site_class`), labels (`mc_max_unc`, `mc_dl_cens`, `mc_cens`, `mc_log10`, `y_gt8`, `y_gt1p6`, `y_gt0p3`, `mc_basis`), modality blocks `<modality>__*` + `<modality>__present`, weather proxy (`wx_site_key`, `wx_cell_km`) |
| `columns.json` | modality → feature columns |
| `availability_all.csv`, `availability_labelled_gt8.csv` | availability-pattern counts |

## Coverage — rebuild 2026-11-08 (current; review W3 satellite dt ≤ 0, W4 WQP/NLA duplicates removed)
* 73,160 site-days at 12,562 sites (NLA 4,892 rows; WQP 68,268); label basis: total 59,941, congener sum 13,219.
* 8 µg/L: 73,039 labelled, 2,862 positive at 925 sites; satellite present on 2,278 rows (imagery on/before the sample
  date only). Frozen split roles unchanged; every labelled site has a role (`make_fuse_splits.py check`). Pre-rebuild
  rows backed up in the Fuse scratchpad until the comparison is reported.

## Coverage — original build 2026-10-25 (superseded)
* 73,364 site-days at 12,731 sites, 2003-01-21 … 2026-09-17; 19 HUC2 regions.
  * WQP total microcystins: 55,252 site-days (4,196 sites; NARS_WQX submissions, `facility` and
    `estuary_ocean` sites dropped); WQP congener sums (site-days with congeners but no total): 13,219;
    NLA visits: 4,893 (4,506 lakes).
* Labels (labelled / positives / sites with a positive): **> 8 µg/L 73,243 / 2,863 / 926** (NLA 78
  positives; WQP total 2,232; congener sums 553); > 1.6 µg/L 59,796 / 7,392 / 1,498; > 0.3 µg/L
  54,587 / 15,827 / 2,693 (lower thresholds and the regression target exclude congener-sum rows).
* Rows with modality present: pigment 36,125; nutrient 36,649; physchem 43,963; satellite 2,243;
  history 59,490; profile 4,849; zoop 4,734; benthic 4,570; phyto 4,890; wx 68,716 (nearest forcing
  cell ≤ 50 km; median distance 17 km); static/time all.
* Largest patterns among labelled rows: history only 23,808 (966 sites); pigment+nutrient+physchem+history
  20,587 (1,809); physchem+history 7,088; pigment+nutrient+physchem 5,168 (4,660); NLA 7-modality
  pigment+nutrient+physchem+profile+zoop+benthic+phyto 4,141 (4,136 lakes); 5-modality with satellite
  1,223 (104 sites). Rows with ≥ 5 non-hub modalities: ≈ 6,100 (NLA ≈ 4,900; WQP with satellite ≈ 1,250).

## Rules
* Labels: max uncensored result per site-day; censored results keep the bound (`mc_dl_cens`); a label is
  NaN if the bound cannot decide it. Thresholds 8 (EPA 2019 recreational, primary), 1.6 (v1), 0.3 µg/L
  (EPA 10-day health advisory, children).
* **Congener sums:** several programmes (Florida DEP, SFWMD, …) report non-detect congeners at the
  reporting limit with no censor flag; summing ~10 congeners adds ~2–3 µg/L. Congener-sum rows therefore
  carry only the 8 µg/L label; `y_gt1p6`, `y_gt0p3`, `mc_log10` are NaN for them (`mc_basis`).
* Same-day co-located modalities (WQP: same site_id and local date, median of results; censored → bound);
  NLA values from `chemistry_wide` (TP µg/L → mg/L exactly ×10⁻³), profiles, community summaries.
* `history`: prior results at the same site dated ≤ d − 7 (lab latency), 365-day window.
* `wx`: nearest SENTINEL gridMET/ERA5 cell (`sentinel.models.downstream.weather`), windows ending d − 1;
  absent when the cell is > 50 km away.
* Other cyanotoxins (anatoxin-a, cylindrospermopsin, saxitoxin) are never inputs.

## Known issues
* Satellite coverage is small: `mc_benchmark_v2` (v1 sites) and the Minnesota pilot of `s2_wqp_matchups`;
  the national S2 × WQP matchup run (Orbit workstream) will extend it (re-run this build).
* `phyto` includes cyanobacterial biovolume — causally proximal to toxin production; P2 is reported with
  and without it. HAB-circularity warning in `nars_lakes.md`.
* 40 % of v1 benchmark rows share a same-day sample within 5 km — spatial blocking and the 5-km buffer of
  the frozen split handle this.
* Programme mix differs by region (Nebraska, Iowa, Oregon dominate WQP); sampling is often targeted at
  suspected blooms (prevalence is not population prevalence).

## Rebuild
```bash
python -B scripts/data/build_fuse_tiers.py c
```
