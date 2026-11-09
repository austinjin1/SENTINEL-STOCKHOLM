# Datasheet — `fuse_tier_w` (SENTINEL-Fuse Tier W: WQP lab samples at USGS sonde sites)

| | |
|---|---|
| **Location** | `$SENTINEL_DATA/processed/fuse_tier_w/` · manifest `manifests/fuse_tier_w.json` |
| **Built** | 2026-10-25 by `scripts/data/build_fuse_tiers.py w` (`sentinel/data/fuse_tier_w.py`) |
| **Inputs** | manifests `wqp_discrete`, `sonde_daily`, `s2_wqp_matchups`, `forcings_daily`, `streamcat`, `sites`; site roles from `docs/preregistration/sonde_split.json` (sha256 `fd4bb14f…`) |
| **License** | US public domain (USGS NWIS, WQP, gridMET, StreamCat); Copernicus open (S2) |
| **Synthetic data** | none |

## Content
`rows.parquet`: one row per (USGS site of the M1 split, local date) with ≥ 1 WQP lab result among TP, TN,
NO3+NO2, NH3, ortho-P, TSS, SSC, chl-a, *E. coli*, DOC, chloride (water years 2009–2025). Columns:
`site_id`, `role` (M1 train/val/test), `basin_group`, modality blocks `sonde__*` (USGS daily sonde
features on or before d — same engineered features as Tier D), `labhist__<analyte>__{value,age}`
(previous sample strictly before d, ≤ 90 d), `satellite__*`, hub `wx__*`, `static__*`, `time__*`,
targets `ylab__<analyte>` (log10(x + offset)) with `__cens`.

## Coverage
* 179,918 rows at 1,343 sites; roles: train 113,434, val 17,118, **test 49,366 (never used by Fuse
  development code)**.
* Present: sonde 170,189; labhist 171,673; satellite **0** (S2 × WQP matchups are Minnesota-only so far
  and these are river gauges); wx 177,229.
* Target rows: SSC 116,282; TP 88,128; NH3 85,658; NO3NO2 83,604; ortho-P 82,359; TN 79,525; chloride
  69,982; TSS 33,883; DOC 33,608; *E. coli* 31,290; chl-a 5,141.
* Patterns: sonde+labhist 162,848 (1,190 sites); labhist only 8,825; sonde only 7,341; hub only 904.

## Known issues
* Only two non-hub modalities until satellite matchups exist at these sites → Tier W is for pretraining
  and the nutrient/FIB nowcast tasks, not for ≥ 5-modality claims.
* Lab-history `age` uses calendar days; lab latency is not modelled (sensitivity analysis in the protocol).

## Rebuild
```bash
python -B scripts/data/build_fuse_tiers.py w
```
