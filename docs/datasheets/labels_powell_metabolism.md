# Datasheet: `labels_powell_metabolism` — Powell Center daily stream metabolism (356 US rivers)

Labels for **RiverPulse (D8)**: daily gross primary productivity (GPP), ecosystem respiration (ER) and gas-exchange rate (K600) at USGS gauges. Built 2026-10-22 by `scripts/data/download_labels_powell_metabolism.py` with client `sentinel/data/sources/downstream_metabolism.py`. Manifest: `$SENTINEL_DATA/manifests/labels_powell_metabolism.json`.

## Source

| Item | Value |
|---|---|
| Publisher | U.S. Geological Survey, John Wesley Powell Center working group |
| Data release | Appling AP, Read JS, Winslow LA, Arroita M, Bernhardt ES, Griffiths NA, Hall RO Jr, Harvey JW, Heffernan JB, Stanley EH, Stets EG, Yackulic CB (2018). *Metabolism estimates for 356 U.S. rivers (2007-2017)*. USGS data release, doi:10.5066/F70864KX |
| Paper | Appling et al. (2018). The metabolic regimes of 356 rivers in the United States. *Scientific Data* 5:180292, doi:10.1038/sdata.2018.292 |
| Landing page / API | `https://www.sciencebase.gov/catalog/item/59bff507e4b091459a5e0982?format=json`; children: `https://www.sciencebase.gov/catalog/items?parentId=59bff507e4b091459a5e0982&format=json` |
| Verified | 2026-10-22 (UTC), anonymous HTTPS, no key. ScienceBase JSON answers can take ~30 s. |
| Release state | Parent `lastUpdated` 2026-10-13T13:44:55Z. Provenance note: "2026-03-04: The structure of this data release has been updated to support migration to a new USGS data system. No changes were made to the data during this update." Label files (items 1, 5, 7, 8) were last updated 2020-08-20. |
| License / terms | US Government work, public domain. FGDC metadata: access constraints "none"; use constraints "None, but see Distribution Liability below. Users are advised to read the associated publication thoroughly to understand appropriate use and data limitations." |
| Extra column | `sites.comid_bernhardt2022` is from the Bernhardt et al. 2022 figshare table `lotic_site_info_full` (doi:10.6084/m9.figshare.19074299, MIT licence), used only as a second COMID assignment. |

ScienceBase publishes no checksums for these files (the `checksum` field is empty). The sha256 values in the manifest were computed locally at download time. Download URLs are resolved at run time from each item's JSON by file name. They are not hard-coded, because the 2026 restructure changed the URL scheme for large files.

## Files

**Children of the release.** Sizes are as listed by ScienceBase on 2026-10-22.

| # | Item ID | Content | Size | Used |
|---|---|---|---|---|
| 1 | 59bff64be4b091459a5e098b | `site_data.tsv` (+ FGDC xml) | 120,159 B | yes (kept raw) |
| 2 | 5a5e3791e4b06e28e9be48c1 | Spatial data parent | — | **HTTP 404** "Item not found … or the item is secured and requires authentication" since the restructure. Its children stay public. |
| 2a | 5a5e37bde4b06e28e9be48ca | Site points shapefile | 134,541 B | no (lat/lon is in site data) |
| 2b | 5a5e37bbe4b06e28e9be48c6 | Catchment polygons shapefile | 132,426,299 B | no (available if catchment features are needed) |
| 3 | 59c03b72e4b091459a5e0b64 | `timeseries.zip` | 7,465,739,701 B | no |
| 4 | 59eb9b9de4b0026a55ffe37c | `inputs.zip` | 1,226,831,078 B | no |
| 5 | 59eb9b86e4b0026a55ffe379 | `config.zip` (model configurations) | 5,273 B | yes (kept raw) |
| 6 | 59eb9ba6e4b0026a55ffe37f | `fits.zip` (full model outputs) | 176,204,952 B | no |
| 7 | 59eb9bafe4b0026a55ffe382 | `diagnostics.zip` (model diagnostics and confidence) | 31,869 B | yes (kept raw) |
| 8 | 59eb9c0ae4b0026a55ffe389 | `daily_predictions.zip` | 100,995,905 B | parsed, then **deleted** under the > 100 MB raw-retention rule. Its sha256 `9b3bc13035ce1d5c89547714ad76f3e4e010992c02d24e92174f2de6f7ac54b8` is in the manifest; `--keep-raw` keeps the file. |

**Raw directory.** `$SENTINEL_DATA/raw/powell_metabolism/` (0.8 MB) holds:
- the small files above;
- FGDC xml for items 1, 5, 7 and 8;
- `lotic_site_info_full.csv.zip`;
- `metadata/*.json` (ScienceBase item JSON for every child, which provides the provenance).

**Processed directory.** `$SENTINEL_DATA/processed/labels_powell_metabolism/` (83 MB):

| File | Rows | Content |
|---|---|---|
| `daily_metabolism.parquet` | 490,907 | One row per site-day: estimates, CIs, MCMC diagnostics, daily predictors, flags, model confidence |
| `sites.parquet` | 602 | All candidate sites (356 modelled): IDs, COMIDs, coordinates, hydraulic-geometry coefficients, infrastructure flags, coverage, site confidence |
| `models.parquet` | 433 | One row per model (site × input resolution): diagnostics, confidence rating, configuration |

## Coverage (measured)

- **Sites and days.** 356 modelled sites (of 602 candidates in the site table) and 490,907 site-days. Both numbers equal those in the paper.
- **Dates.** 2007-10-01 to 2016-12-30.
  - Inputs were assembled for October 2007 to January 2017; the last estimate date is 2016-12-30. The FGDC date range agrees.
  - Days per site: 61 to 3,296 (median 1,180), matching the paper. There are 2,190 site-years.
- **Rows by year:**

  | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 |
  |---|---|---|---|---|---|---|---|---|---|
  | 6,560 | 29,447 | 34,380 | 36,936 | 46,044 | 55,849 | 59,614 | 69,184 | 72,064 | 80,829 |
- **Models.** 433 models.
  - Sites by model count: 286 have one model, 63 have two, 7 have three (one per DO logging resolution).
  - Site-days by resolution: 15 min 330,625; 60 min 83,710; 30 min 59,484; 5 min 15,042; 12 min 2,046.
- **Space.**
  - Modelled sites lie at latitude 27.94–60.20: CONUS plus one Alaska site.
  - The site table also lists unmodelled candidates in Puerto Rico (2 sites) and Alaska (1).
  - NWIS site types of the modelled sites: ST 344, SP 7, ST-TS 2, ST-CA 2, ST-DCH 1.

## Label definition

- **`gpp`** (`g O2 m-2 d-1`): mean daily gross primary productivity. It is the median of the post-warm-up MCMC posterior; `gpp_lower` and `gpp_upper` are its 2.5% and 97.5% quantiles. `gpp_n_eff` and `gpp_rhat` are the MCMC diagnostics.
- **`er`** (`g O2 m-2 d-1`): ecosystem respiration. More negative values mean more respiration. Its CI and diagnostic columns follow the same pattern.
- **`k600`** (`d-1`): reaeration rate coefficient scaled to Schmidt number 600.
- **`date`**: the solar day the estimate applies to, from 04:00 to 03:59 the next day (streamMetabolizer solar time, `calcLon`).
- **Model.** Every model is streamMetabolizer `b_Kb_oipi_tr_plrckm.stan`: Bayesian state-space, with K600 partially pooled on discharge. Inputs are NWIS DO, water temperature, discharge-derived depth (hydraulic geometry) and modelled light.
- **These are model-derived estimates, not measurements.** They are inversions of diel dissolved-oxygen curves.
- **Unit columns.** `gpp_er_unit` and `k600_unit` store the units explicitly.
- **Daily predictors** carry their units in the column name: `do_obs_mgl`, `do_sat_mgl`, `do_amp_mgl`, `do_psat_pct`, `depth_m`, `temp_water_c`, `day_length_h`, `discharge_m3s`, `shortwave_wm2`, `velocity_ms`, `do_tdist80_m`.
  - The FGDC text gives the averaging window of the DO predictors as "4am to 3:59am". For depth, temperature, day length, discharge, shortwave, velocity and turnover distance it says "4am to 3:59pm". This looks like a metadata typo (unverified); treat all columns as the same solar day.

## Quality filters (authors' guidance, quoted from the paper's Technical Validation)

The authors give a model-level rating, not a day-level filter. Their algorithm labels "each model as likely deserving of Low, Medium, or High confidence"; it is "intended only as guidance".

The rating uses these criteria:
- **Convergence.** Problems are inferred "when Rhat > 1.2" for SD(K600) or SD(process error).
- **K600 spread.** The 10th–90th quantile range of daily K600 is "unrealistic when > 50"; models with a range "< 15" get the highest ratings.
- **Implausible values.** The authors compute the "percentages of GPP_d estimates below −0.5 and ER_d values above 0.5". A model is Low when more than 50% of days fall beyond these thresholds, Medium at 25–50%, and High below 25%.
- **Positive ER.** "positive ER_d often reflects a miscalibrated oxygen sensor", while GPP may still be reliable.

The Usage Notes add that "most data users will filter this complete dataset down to the estimates that are appropriate for their analysis".

**Site-day counts** (every row has GPP and ER):

| Filter | Site-days |
|---|---|
| All | 490,907 |
| `model_confidence` H / M / L | 363,275 / 68,236 / 59,396 |
| H or M | 431,511 |
| **H or M, and day-level plausible (GPP ≥ −0.5 and ER ≤ 0.5)**, the proposed SENTINEL default | **399,472** |
| H, and plausible | 344,961 |
| Plausible, any confidence | 445,154 |
| All three daily Rhat ≤ 1.05 | 470,472 |

- **Models and sites by rating.** Models: H 299, M 63, L 71. Sites by `site_min_confidence`: H 245, M 48, L 63.
- **Flag columns.**
  - `flag_gpp_negative` (gpp < −0.5): 15,218 days.
  - `flag_er_positive` (er > 0.5): 34,132 days.
  - `flag_rhat` (any daily Rhat > 1.05, the threshold the metadata gives for convergence): 20,435 days.

The SENTINEL default is a proposal, not the authors' rule. Report H-only as a sensitivity check.

## QC applied at ingest

- **IDs are strings.**
  - `site_id` is `nwis_<number>`. `nwis_id` keeps its leading zeros, and 15-digit IDs are left untouched.
  - `nwis_id` must equal the suffix of `site_name`; otherwise the build fails.
  - `comid` is stored as a string.
- **Missing values and codes.**
  - R `NA` becomes NaN with an empty qualifier.
  - Sentinel codes (−999999, −99999, −9999) and non-numeric text become NaN, and `qualifier` records `col=code`. No such values exist in this release (0 rows).
  - The only column with NaN is `do_tdist80_m`: 1,301 rows, all at nwis_01648010.
- **Dates.** Parsed with `%Y-%m-%d` and bounded to 2007-01-01..2017-12-31; none were invalid. There are no duplicate `(site_id, date)` rows; a duplicate fails the build.
- **Coordinates.** Checked against the US bounding box; all 602 pass. The datum is NAD83, as published.
- **Model confidence.** Attached to every daily row by `(site_id, resolution_min)`. Every row matched a published model.

## Known biases and caveats

- **Site selection.** Sites are NWIS gauges with at least hourly DO (at least 100 observations) and all model inputs available at the same time. The paper: "limited to monitoring locations chosen for the purposes of other projects, resulting in non-uniform spatial and hydrologic coverage". Larger, gauged, monitored, eastern and midwestern rivers are over-represented.
- **Inverse-model artefacts.**
  - ER and K600 trade off against each other (equifinality).
  - Days with input gaps longer than 3 h were dropped.
  - Light is modelled, and depth and velocity come from hydraulic geometry.
  - Sites with low confidence ratings or positive ER often have DO-sensor calibration problems.
- **Infrastructure.** `struct_{canal,dam,npdes}_flag` takes values 0/50/80/95. A value of 0 means the nearest canal, dam or NPDES outfall is closer than the 50th-percentile daily 80% O2-turnover distance. NA occurs at 269 of 602 sites (23 of them modelled); the metadata does not explain it.
- **Circularity (mandatory control).**
  - GPP and ER were inverted from these sites' own DO series. An upstream DO model trained on the same site-days leaks into RiverPulse.
  - Either exclude these sites' DO from upstream training, or restrict RiverPulse inputs to non-DO upstream outputs (RESEARCH_PLAN §8 D8; 07 §d.8).
  - The daily predictor columns are the model's own inputs. Never use them as features in an "unsensed reach" evaluation.
- **Time window.** The series ends 2016-12-30, so it does not overlap the EW-v2 test years.

## Joins

- **`nwis_id`.** Use it with NWIS and NLDI: `https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{nwis_id}`. Prefer this route when other SENTINEL tables were snapped through NLDI, because the same site can get different COMIDs from different methods (07 §c.1).
- **`comid` (NHDPlusV2).** Taken from the published `nhdplus_id`, which is the reach whose centre point is nearest the site (paper).
  - Present for 583 of 602 sites, including 355 of the 356 modelled sites; nwis_15298040 has none.
  - **Cross-check against Bernhardt et al. 2022** (`comid_bernhardt2022`, `comid_agree`). 354 modelled sites appear in both tables. The COMIDs agree for 342 and differ for 12:
    - nwis_01104430, 01196500, 01658000, 02208450, 03298200, 03321500, 03538830, 04136500, 05515500, 05579630, 08155500, 08180700.
    - Snap these 12 sites with NLDI before any reach-level join.
- **Coordinates and HUC.** `latitude`/`longitude` (NAD83) are present for every site. The release has no HUC; derive it from the NWIS site service `huc_cd` or a WBD point query.
- **Overlap with `labels_streampulse`.** Gauges 05406457 and 05406469 also appear in the StreamPULSE daily results. When pooling the two datasets, de-duplicate on (USGS site number, date).

## Schema: `daily_metabolism.parquet`

| Column | Type | Notes |
|---|---|---|
| `site_id`, `nwis_id` | string | `nwis_01021050`, `01021050` |
| `date` | datetime64 (day) | solar day 04:00–03:59 |
| `resolution_min` | Int64 | DO logging interval of the fitted model |
| `gpp`, `gpp_lower`, `gpp_upper`, `gpp_n_eff`, `gpp_rhat` | float | g O2 m-2 d-1 |
| `er`, `er_lower`, `er_upper`, `er_n_eff`, `er_rhat` | float | g O2 m-2 d-1 |
| `k600`, `k600_lower`, `k600_upper`, `k600_n_eff`, `k600_rhat` | float | d-1 |
| `do_obs_mgl` … `do_tdist80_m` | float | daily predictors (units in the name) |
| `gpp_er_unit`, `k600_unit` | category | explicit units |
| `qualifier` | str | `col=code` for masked values |
| `flag_gpp_negative`, `flag_er_positive`, `flag_rhat` | bool | day-level flags (above) |
| `model_confidence`, `site_min_confidence` | category | authors' H/M/L ratings |

- **`sites.parquet`:** `site_id`, `nwis_id`, `comid`, `comid_method`, `site_name`, `latitude`, `longitude`, `coord_flag`, `coord_datum`, `altitude_ft`, `altitude_datum`, `nwis_site_type`, `hydgeom_{c,f,a,b,k,m}` (depth = cQ^f, width = aQ^b, velocity = kQ^m; SI units), `struct_{canal,dam,npdes}_flag`, `qualifier`, `n_days`, `date_min`, `date_max`, `n_models`, `site_min_confidence`, `site_confidence`, `modeled`, `comid_bernhardt2022`, `comid_agree`.
- **`models.parquet`:**
  - Keys and diagnostics: `site_id`, `nwis_id`, `resolution_min`, `k600_daily_sigma_rhat`, `err_obs_iid_sigma_rhat`, `err_proc_iid_sigma_rhat`, `k_median_d`, `k_range_d`, `neg_gpp_pct`, `pos_er_pct`, `run_hrs`.
  - Ratings: `model_confidence`, `site_min_confidence`, `site_confidence`, `qualifier`.
  - Configuration: `required_timestep_d`, `model_name`, `k600_lnq_nodes_centers`, `k600_daily_sigma_sigma`, `burnin_steps`, `saved_steps`, `depth_src`.

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_powell_metabolism.py              # ~2–5 min (101 MB download)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_powell_metabolism.py --keep-raw   # keep daily_predictions.zip
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_metabolism.py
```

The downloader skips files that already exist with the size ScienceBase lists. If `daily_predictions.zip` was deleted, it is downloaded again.
