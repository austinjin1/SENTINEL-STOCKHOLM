# Datasheet: `labels_streampulse` — StreamPULSE public daily metabolism estimates

These are secondary labels for **RiverPulse (D8)**. The table holds daily GPP, ER and K600 at StreamPULSE research sites, and it extends the Powell Center labels in time (2006–2024) and to non-USGS sites.

- **Built:** 2026-10-22 by `scripts/data/download_labels_streampulse.py`.
- **Client:** `sentinel/data/sources/downstream_metabolism.py`.
- **Manifest:** `$SENTINEL_DATA/manifests/labels_streampulse.json`.

## Source and access

| Item | Value |
|---|---|
| Publisher | StreamPULSE network (Duke University; NSF Macrosystems EF-1442439), https://data.streampulse.org |
| Bulk page | https://data.streampulse.org/download_bulk. The page says datasets are "updated daily at 12:00 AM EST"; the served files had `Last-Modified` dates of 2026-10-19. |
| Endpoints (verified 2026-10-22) | Plain HTML **POST** forms with an empty body. **No login, token or key.**<br>• `https://data.streampulse.org/_alldailyres_download` → `all_daily_model_results.csv.zip` (3,219,329 B)<br>• `/_allmodelsumm_download` → `all_model_summary_data.csv.zip` (34,658 B)<br>• `/_allbasicsite_download` → `all_basic_site_data.csv.zip` (33,586 B) |
| COMID table | Bernhardt et al. 2022 (*PNAS* 119:e2121976119), figshare "CSV data for Bernhardt et al. 2022 (PNAS)", doi:10.6084/m9.figshare.19074299.v3 (modified 2023-06-01). File `lotic_site_info_full.csv.zip` (25,492 B) from `https://ndownloader.figshare.com/files/33900755`. |
| Licence and terms | **StreamPULSE data policy** (https://data.streampulse.org/policy; snapshot saved as raw `streampulse_policy.html`): "The data are available under the Open Data Commons Attribution License" (ODC-By 1.0).<br>• Cite Appling, Hall, Yackulic & Arroita (2018), *JGR Biogeosciences* 123:624–645, until StreamPULSE DOIs exist.<br>• Acknowledge the StreamPULSE network and NSF Macrosystems.<br>• Notify data contributors of derivative works.<br>• Data from newly contributed sites are embargoed for one year by default and are not in the public files.<br>The Bernhardt figshare table is licensed MIT. |
| Personal data | The portal's site table includes contributor names and e-mails (`contact`, `contactEmail`). **Both columns are dropped at parse time.** They remain only inside the raw zip, as published. Test fixtures were redacted. |

The portal also serves other bulk products, which are **not downloaded** here. They are documented for completeness:

| Product | Size |
|---|---|
| Public sensor data | 508.1 MB |
| NEON data housed in StreamPULSE | 2.0 GB |
| Powell Center data housed in StreamPULSE | 1.8 GB |
| Public model objects (RDS) | 852.7 MB |
| Grab samples | 236 kB |
| Reach characterisation | 261 kB |
| Supplementary metadata | 286 kB |

The bulk page also links three external products:
- **Dryad**, follow-up metabolism for 59 Powell sites, 2008–2021: doi:10.5061/dryad.bcc2fqzj2.
- **Figshare 28074452**, NEON metabolism estimates.
- **The Bernhardt 2022 figshare CSVs**, all open, licensed MIT, and not built here:
  - `lotic_standardized_full` (56 MB zip)
  - `lotic_gap_filled` (53 MB)
  - `fluxnet_filtered_metabolism` (14 MB)
  - the site-info tables

  These overlap Powell and StreamPULSE.

## Files

**Raw** (`$SENTINEL_DATA/raw/streampulse/`, 3.3 MB, kept):
- the three StreamPULSE zips, each with a `.headers.json` sidecar that records Content-Disposition, Content-Length and Last-Modified at download time;
- `lotic_site_info_full.csv.zip`;
- `figshare_19074299.json` (the record metadata, including the licence);
- `streampulse_policy.html`.

**Processed** (`$SENTINEL_DATA/processed/labels_streampulse/`, 4.2 MB):

| File | Rows | Content |
|---|---|---|
| `daily_metabolism.parquet` | 62,383 | One row per published site-day: estimates, CIs, `valid_day`, error text, flags, joined site attributes |
| `models.parquet` | 541 | One row per model run (site × year × run), with fit diagnostics and `current_best` |
| `sites.parquet` | 738 | Public site table, de-duplicated, without contact columns, plus Bernhardt COMIDs |
| `bernhardt2022_site_info.parquet` | 435 | Repaired Bernhardt site table: 354 Powell sites and 81 StreamPULSE sites |

## Coverage (measured)

**Daily table.** 62,383 site-days at 134 published site IDs, 2006-12-07 to 2024-06-30. There are 288 site-years. Rows per year:

| 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 398 | 779 | 737 | 700 | 728 | 1,134 | 6,729 | 6,442 | 6,810 | 10,443 | 8,052 | 10,795 | 5,077 | 2,441 | 184 | 532 | 207 | 182 |

Coverage after 2020 is thin.

**Estimates.**
- `valid_day` is True for 48,690 rows. All of them have GPP and ER, and they cover all 134 site IDs.
- The other 13,693 rows have no estimate. Their `errors` text gives the reason, for example "NAs in DO.obs; NAs in DO.sat; …" (5,804 rows) or "NAs in depth; NAs in discharge" (1,829 rows).

**US versus non-US sites.**
- `in_us` is True for 102 daily sites, which hold 38,623 valid rows dated 2008-01-08 to 2023-11-01.
- The non-US regions are Austria (AT), Australia (AU), Costa Rica (CR), Ecuador (EC), Greenland (GL), New Zealand (NZ) and Sweden (SE).
- 4 daily sites have no public site row, so `in_us` is `<NA>` for them; `region_is_us_code` is always available.

**Data source of the daily sites** (from the site table): StreamPULSE-Leveraged 104, StreamPULSE-Core 23, NEON 3 (`AL_MAYF-up`, `AZ_SYCA-up`, `PR_GUIL-up`), not in the site table 4.

**Site table.**
- 738 sites after de-duplication: 356 Powell Center sites, 271 Leveraged, 68 NEON and 43 Core. Only some of these have public daily model results.
- 68 sites are currently embargoed (`embargo_days_remaining` > 0).

## Label definition

- **`gpp` and `er`** (`g O2 m-2 d-1`) and **`k600`** (`d-1`) are daily estimates with 95% intervals (`*_lower`, `*_upper`).
- **Model.** All 541 runs use streamMetabolizer, Bayesian, with the Stan engine. Pooling is `binned` for 531 runs and `none` for 10. The models are fitted by StreamPULSE to the contributors' sensor DO.
- **These are model-derived estimates, not measurements.**
- **Primary label rows:** `valid_day == True` and both plausibility flags False. That gives 43,973 rows, of which 34,602 are US rows (`in_us`).
- **Plausibility flags** use the Appling et al. 2018 thresholds:
  - `flag_gpp_negative` (GPP < −0.5): 2,477 valid rows.
  - `flag_er_positive` (ER > 0.5): 2,801 valid rows.
- **Model-level quality** is in `models.parquet`:
  - Fit statistics: `o2_gof`, `gpp_95ci`, `er_95ci`, `prop_pos_er`, `prop_neg_gpp`, `er_k600_cor`, `coverage`, `kmax`.
  - `current_best`: 287 of 541 runs, at most one per site-year.

  These columns can be used to tighten the filter, for example by excluding runs with a high ER–K600 correlation (equifinality).
- **Extreme K600.** 2,890 valid days have K600 > 100 d-1. Treat them with caution.

## QC applied at ingest

- **Site IDs** are `region_site`, kept exactly as published.
  - The portal is not case-consistent: daily `co_Cameo2` and `co_WindyGap2` correspond to site rows `CO_Cameo2` and `CO_WindyGap2`. Daily rows join to the site table by exact match first, then by a unique case-insensitive match. `site_match` records the result (exact 127, case_insensitive 3, none 4), and `site_table_id` holds the matched key.
  - `MI_Tobacco` and `MI_TOBACCO` are two different model runs of one site, with overlapping dates and different estimates. Both are kept, and their 74 overlapping site-days are flagged in `dup_site_date_ci`. Pick one before pooling.
- **Duplicate site registrations.** `CO_Wfresdown1`, `OH_EastChagrin` and `IA_WalnutCreek12` appear twice, seconds apart, with identical coordinates. The earliest registration is kept; `n_registrations` and `all_site_names` record the rest. The build fails if the duplicated rows disagree on coordinates.
- **Values.** Blank estimates become NaN. Sentinel codes and non-numeric text would become NaN with a `qualifier` entry; none occur (0 rows).
- **Dates.** Validated against 1990-01-01 up to tomorrow; none are flagged. `(site_id, date)` is unique, and a duplicate fails the build.
- **USGS gauge numbers** are strings. All-digit IDs shorter than 8 digits are zero-padded to 8. The value `0` or a blank becomes NA.
- **Contact columns** are dropped.

## Known biases and caveats

- **Who contributes.** The network is research-driven and opportunistic. Sites are clustered in NH, NC, FL, WI, AZ, IN, PR and Austria, and most data fall in 2013–2018.
- **Embargoes.** Embargoed sites are not in the public files, so the public set is not stable over time. Pin the vintage through the manifest sha256 hashes.
- **Contributor choices.** Model configuration and the `current_best` selection are made by contributors, so model choices are heterogeneous across sites.
- **Circularity.** As with Powell, the labels are inversions of each site's DO. Do not let an upstream DO model see these site-days if its outputs become RiverPulse inputs.
- **Overlap with Powell.** Gauges 05406457 and 05406469 have daily results here and in `labels_powell_metabolism`. De-duplicate on (USGS site number, date). The 356 Powell sites in the site table have no daily rows in this public product.

## Joins

| Key | Daily sites covered | Notes |
|---|---|---|
| `usgs_site_no` | 39 | Leads to NWIS and NLDI (`nwissite/USGS-{no}`) |
| `comid` | 77 | NHDPlusV2, from Bernhardt et al. 2022; recorded in `comid_source` |
| `latitude`, `longitude` | 130 | Use NLDI `comid/position` for sites without a COMID |
| NEON site code | 3 | `MAYF`, `SYCA`, `GUIL` from the `site` field |

## Schema: `daily_metabolism.parquet`

- **IDs:** `region`, `site`, `site_id`.
- **Time:** `date` (day), `date_flag`, `year`.
- **Estimates:**
  - `gpp`, `gpp_lower`, `gpp_upper`
  - `er`, `er_lower`, `er_upper`
  - `k600`, `k600_lower`, `k600_upper`
  - units in `gpp_er_unit` and `k600_unit`
- **Flags and diagnostics:** `region_is_us_code`, `valid_day`, `warnings`, `errors`, `qualifier`, `flag_gpp_negative`, `flag_er_positive`, `dup_site_date_ci`.
- **Join columns:** `site_table_id`, `site_match`, `data_source`, `usgs_site_no`, `latitude`, `longitude`, `in_us`, `comid`, `in_site_table`.

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_streampulse.py            # uses cached zips if present
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_streampulse.py --refresh  # re-download (portal content changes)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_metabolism.py
```
