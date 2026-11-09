# 09: Event catalogues, early-warning evaluation, and the operational EWS landscape (design input for SENTINEL "Early Warning v2")

*Compiled 2026-10-22. Sources were checked through official pages, APIs, bulk downloads and Crossref/arXiv metadata during this session. Small samples are in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/` (157 MB in total). Nothing else in the repository was modified.*

**Verification tags used throughout**

| Tag | Meaning |
|---|---|
| **[V-data]** | I downloaded or queried the dataset or API this session. Row counts and field names are what I observed. |
| **[V-doc]** | I checked the statement against the official document or web page text this session. |
| **[V-ref]** | I checked the bibliographic metadata (title, authors, venue, year, pages, DOI or arXiv ID) against Crossref, doi.org or the arXiv API this session. |
| **[P]** | Partly verified. The statement comes from search-result snippets or secondary pages, not the primary document. |
| **[U]** | Not verified this session. Do not quote it without checking. |

The WebSearch quota was exhausted twice during the session, and several sites (orsanco.org, icpdr.org, data.ny.gov via curl) were unreachable from the server. Items that could not be re-checked are tagged [P] or [U].

---

## 0. Decisions that matter (one-page summary)

1. **Why v1 failed, in arithmetic.** v1 counted a "detection" if any alarm fell in the 45 days before onset. Assume false alarms arrive as a Poisson process at rate λ per day and the lookback window is W days. Then a random date is "detected" with probability P₀ = 1 − exp(−λW).
   - The review measured a median of 11.2 alarms per 30 clean days. That gives λW ≈ 16.8, so P₀ ≈ 1. The placebo result of 98–99% follows directly.
   - Even a well-behaved detector with 1 false alarm a month would "detect" 78% of random dates under a 45-day window.
   - **The v2 design must therefore fix three things before any test data is opened:** (a) a short lookback equal to the forecast horizon (1–7 d), (b) an event-level false-alarm budget (false-alarm episodes per site-month), and (c) report every detection rate next to its chance rate P₀ or a placebo rate.
2. **Recast early warning as probabilistic forecasting of event onsets.**
   - Issue a forecast at fixed times, for example daily, giving P(onset in (t, t+h]) for h = 1, 3, 7 days.
   - Score every issue time with the Brier score and Brier skill score (BSS) against site × day-of-year climatology and against persistence. Add reliability diagrams.
   - Report event-level POD, false alarms per site-month and lead times at one alarm threshold frozen in advance.
   - This is the standard in weather forecast verification (Jolliffe & Stephenson 2012; Wilks 2019). EPA's own detection-system evaluation (EDS Challenge, EPA 817-R-13-002) used this logic: percent of events detected against invalid alerts per week. That report explicitly rejects per-time-step ROC curves.
3. **Ground truth should come mainly from public sensors, not incident lists.** Of the seven landmark incidents I checked, only one is clearly visible in public continuous sensor data: the Des Moines/Raccoon River nitrate exceedances (2013–2025, including 2015).
   - Five were not measured at the time and one (Trinseo 2023) is unclear. Either the analyte is not sensed (MCHM, microcystin, latex), or no sonde was deployed: the Animas/San Juan continuous WQ records start in 2016, after Gold King.
   - Incident catalogues (NRC spills, SSO reports, fish kills, HAB advisories) are complaint- or effort-driven and have poor location data. Use them as secondary labels and as context layers, never as the main denominator.
4. **Recommended v2 event classes, in order of observability:**
   - DO hypoxia onsets: DO < 3.0 mg/L sustained ≥ 2 h, anchored to the EPA 1986 warmwater 1-day minimum. Tiers at 4.0 and 2.0 mg/L.
   - Road-salt salinization: chloride estimated from specific conductance (SC) exceeding the EPA chronic (230 mg/L, 4-day mean) or acute (860 mg/L, 1-h mean) criterion.
   - Turbidity excursions.
   - Nitrate above the MCL (10 mg/L as N) upstream of drinking-water intakes.
   - Satellite lake-week cyanobacteria blooms, using EPA's forecast definition: median chl-a ≥ 12 µg/L with cyanobacteria dominance.
   - Beach E. coli above the EPA Beach Action Value (235 cfu/100 mL).

   Supply is large. There are about 1,011 active real-time USGS surface-water DO sites, about 1,595 SC sites, about 976 turbidity sites and about 200 nitrate sites [V-data].
5. **Baselines are mandatory, and some are operational.** Every result must be compared against:
   - climatology, persistence, a threshold-on-current-value rule, logistic regression on lags, and a GBM;
   - where they exist, the operational forecasts: EPA's experimental cyanoHAB forecast for 2,192 lakes, NOAA's Lake Erie products, and USGS-style beach-nowcast persistence benchmarks (Francy et al. 2013).
6. **Run the prospective trial now, pre-registered and externally timestamped.**
   - Register on OSF.
   - Write every forecast to an append-only public log (git commits plus OpenTimestamps or Zenodo snapshots) before the outcome is observed.
   - Score after USGS data approval. Also submit the relevant variables to the running **NEON Ecological Forecasting Challenge** (oxygen, temperature, chla; CRPS against climatology and persistenceRW) and the **VERA challenge** (it includes `Bloom_binary_mean` = chl-a > 20 µg/L). Both are scored independently and prospectively.
   - Winter 2026–27 suits salinization and turbidity. Summer 2027 suits DO, HAB and beaches.
7. **Hard infrastructure deadline.** USGS legacy WaterServices will be permanently shut down on **22 February 2027**. Intentional slowdowns begin on 16 November 2026 [V-doc]. Build the live pipeline only on `api.waterdata.usgs.gov/ogcapi/v1`. It is rate-limited without a key; signup is at https://api.waterdata.usgs.gov/signup/ [V-data].

---

## (a) Event catalogues

### a.1 Master table: real, public catalogues (access checked)

Counts are what I observed in this session's downloads or API calls unless marked otherwise.

| # | Source (hazard) | Access URL(s) | Status | Key fields | Coverage / counts observed | Licence / terms | Joinability to USGS/NEON sensors | Label noise / reporting bias |
|---|---|---|---|---|---|---|---|---|
| 1 | **NY DEC "Harmful Algal Blooms by Waterbody Summary: Beginning 2019"** (HAB) | https://data.ny.gov/d/95my-wijm (SODA: `/resource/95my-wijm.json`); 2012–2018 companion: https://data.ny.gov/d/qtg2-hjth | [V-data] via WebFetch (data.ny.gov resets TLS from this server; use a browser or another host) | `waterbody_name, county, year, bloom_type, date_first_report, date_last_report, number_of_reports, latitude, longitude` | 1,371 waterbody-year rows, 2019–2025; `bloom_type` is "confirmed" in all 1,371. The 2012–18 set holds each waterbody's maximum annual status. | Catalog says "no license info" [P] | Lake centroid lat/lon only, so joining needs a lake polygon (NHDPlus waterbody). First/last report dates bound each episode. | Effort-driven (volunteer and agency visits). Only confirmed blooms are listed, so non-bloom years are implicit. First report ≠ onset. |
| 2 | **California Water Boards FHAB** (bloom reports, cases, advisories) | CKAN: `https://data.ca.gov/api/3/action/package_show?id=surface-water-freshwater-harmful-algal-blooms` | [V-data] | Bloom reports have 55 columns, including `Observation_Date, Bloom_Latitude/Longitude, Advisory_Recommended, AdvisoryStartDate, AdvisoryEndDate, Water_Body_Type, Case_ID` | **3,728 bloom reports** (observations 2016–2026, peak 737 in 2022). **1,484 cases** (2016–2026). 2,080 "results" rows, mostly blank analytes, so toxin values are sparse here. Advisories: Caution 902, Warning 88, Danger 166. **End date present for only 238 of 3,297 with a start date.** | CKAN licence "Other (Public Domain)" | Lat/lon for 3,727 of 3,728 reports. Many sites are reservoirs and wadeable streams. | Voluntary reports, assumed "suspected" until confirmed (official disclaimer) [V-doc]. Malformed years appear (1976, 2555, 2626). Lift dates are mostly missing. |
| 3 | **Oregon OHA recreational cyanoHAB advisory archive** | https://www.oregon.gov/oha/ph/healthyenvironments/recreation/harmfulalgaeblooms/pages/archive.aspx; downloads via the ArcGIS map https://experience.arcgis.com/experience/01f798892ecb48d8ac99ef9b4ebe37c8/ | [V-doc] | Waterbody, status, date, county, toxin values. Field sets change by era (start/end dates in 2007–2015). | 2007–2025. 32 entries in 2025, 19 in 2024 (including a permanent Umpqua advisory). | State web content [U] | Waterbody names only, so geocoding is needed. | OHA states that only a fraction of waterbodies are monitored [P]. |
| 4 | **Washington toxic algae (Ecology / King County)** | https://www.nwtoxicalgae.org/ | [V-doc] | Toxin results by lake and date; county/WRIA/year summaries | Results run into Oct 2026. No bulk export link found. | [U] | Lake names; scraping or a request would be needed. | Sampling is triggered by reports. |
| 5 | **Ohio EPA public-water-system cyanotoxin monitoring** | Strategy: https://dam.assets.ohio.gov/image/upload/epa.ohio.gov/Portals/28/documents/habs/2020-PWS-HAB-Strategy.pdf | [V-doc] for the rules. Bulk data **not found** [U]. | Raw and finished microcystin results by intake | Weekly raw/finished sampling May–Oct; raw > 5 µg/L triggers more sampling [V-doc]. Results are on an Ohio EPA dashboard [P]. | — | Intake locations are good anchors for HAB-at-intake events. | Sampling frequency depends on prior detections, which biases event detection. |
| 6 | **EPA/NASA CyAN satellite cyanobacteria index** (HAB) | NASA: https://www.earthdata.nasa.gov/data/projects/cyan; file search https://oceandata.sci.gsfc.nasa.gov/api/cyan_file_search/; EPA: https://www.epa.gov/water-research/cyanobacteria-assessment-network-cyan; CyANWeb https://qed.epa.gov/cyanweb/ | [V-doc]; file download not tested (OB.DAAC normally requires a free Earthdata login [U]) | Daily and 7-day max GeoTIFF tiles, DN → CI: `CI = 10^(DN×0.011714 − 4.1870866)`. DN 0 = below detection, 254 = land, 255 = no data. Range about 10⁴–7×10⁶ cells/mL. | 300 m. MERIS 2002–2012; OLCI S3A 2016–, S3B 2018–. Product v6 (DOI 10.5067/MERGED-S3/OLCI/L3M/CYAN/CI/6.0). | Citation requested; no explicit licence on the NASA page [V-doc] | Lake polygons of satellite-resolvable lakes, about 2,000+. Can be joined to NHD lakes. | Clouds, ice, mixed shoreline pixels, small lakes excluded. EPA: the original project ended in 2023 and moved to Sentinel-2 in 2024; the S3 portion "will conclude" but data stay public and are reprocessed annually [V-doc]. CyANWeb had an outage notice on 25 Mar 2026 [P]. |
| 7 | **EPA experimental cyanoHAB forecast** (an operational *baseline*, not ground truth) | https://www.epa.gov/habs/hab-forecasts; method page https://www.epa.gov/water-research/cyanobacterial-harmful-algal-blooms-forecasting-research | [V-doc] | Weekly bloom probability per lake. Bloom = **median lake chl-a ≥ 12 µg/L with cyanobacteria dominance**. Week runs Sunday–Saturday. | 2,192 S3-resolvable CONUS lakes; issued April–November. EPA claims "overall prediction accuracy of 90 percent" and notes the model "tends to overpredict". Dashboards offer downloadable data. | — | Same lakes as CyAN | Trained and validated against CyAN satellite labels, not in-situ toxin data [V-doc]. Accuracy is not skill. |
| 8 | **NOAA NCCOS Lake Erie bloom severity index (SI), curated 2000–2025** | https://nccospublicstor.blob.core.windows.net/hab-data/bulletins/lake-erie/2025/NOAA_NCCOS_2000to2025_Curated_LE_Annual_Severity.xlsx; product page https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/lake-erie/ | [V-data] | `year, total_CI_max, total_area_km2_max, total_CI_avg_30d, total_area_km2_avg_30d, severity_index, Date_max` | 26 years. Examples: 2011 = 10, 2015 = 8.4, 2017 = 7.4, 2019 = 5.9, 2023 = 2.9, 2024 = 4.2, 2025 = 2.4. | Attribution statement in file | Basin-scale annual label, n = 26. Useful only for seasonal skill. | **The labels were revised.** The series was reprocessed after 2024. 2015 is now 8.4, though press releases said 10.5; 2023 is now 2.9 against a previously published 5.3 [P for the old values]. Freeze the label vintage. |
| 9 | **NOAA HAB-OFS Lake Erie bulletins/imagery archive** | https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:NOS-HABOFS-LakeErie | [V-doc] | Seasonal bulletin PDFs and GeoTIFFs | From 2017-06-25, with annual accessions | NOAA | Western basin | Bulletin content is itself a forecast. Use it as a baseline. |
| 10 | **GLERL/CIGLR western Lake Erie weekly monitoring** (microcystin, pigments, nutrients) | Boegehold et al. 2023 (ESSD) + NCEI DOI 10.25921/11da-3x54; current provisional data https://www.glerl.noaa.gov/res/HABs_and_Hypoxia/wle-weekly-current | [V-ref] / [P] | Station, depth, date, microcystins, chl-a, phycocyanin, nutrients | 2012–2021 in the paper (accessions 0187718 for 2012–18, 0292222 for 2022 [P]). Weekly, May–Oct. | NOAA public | About 8 fixed western-basin stations [P] | Fixed schedule, so less effort bias. Good for toxin labels. |
| 11 | **Water Quality Portal (WQP)** discrete toxin and FIB results | `https://www.waterqualitydata.us/data/Result/search?characteristicName=...` | [V-data] (HEAD counts, 2015–2025) | Standard WQX result fields | `Microcystin`: 52,425 results at 4,753 sites. `Microcystin LR`: 12,603 / 4,197. `Cylindrospermopsin`: 29,820 / 6,546. `Escherichia coli`: **1,442,087 results at 57,828 sites**. Names must be valid enumerations ("Microcystins" returns HTTP 400). | Public | Site lat/lon. WQP sites are linked in NLDI. | Sampling is event-driven and heterogeneous across agencies, with detection limits and units mixed. |
| 12 | **Chicago Park District "Beach Lab Data"** (FIB) | https://data.cityofchicago.org/resource/2ivx-z93u.json (also hmqm-anjq DNA, hh4t-tnq8 culture, 9wd3-tqr2 both) | [V-data] | `beach, dna_sample_timestamp, dna_reading_mean, culture_sample_1_timestamp, culture_reading_mean, latitude, longitude` | 34,967 rows. Culture tests end in 2016; DNA (qPCR) readings > 0 in 17,572 rows; 1,227 rows have DNA mean > 1,000. Updated 2026-10-08. | City open data | Fixed beaches, near NOAA/USGS Great Lakes stations | Malformed timestamps (min "0023-06-08"). The threshold for the DNA test must come from the Park District's method documentation [U]. EPA's qPCR BAV of 1,000 cce/100 mL applies to *Enterococcus* [V-doc]. |
| 13 | **EPA BEACON 2.0** (beach advisories/closures) | https://beacon.epa.gov/ords/beacon2/r/beacon_apex/beacon2_beta/about-page | [V-doc] | Beach actions (PRAWN), monitoring (WQX), RAD geography | National coastal and Great Lakes beaches | EPA | Beach geometries | Jurisdictions submit after the season ends. EPA lists known issues: duplicate rows, inconsistent beach counts. Export format not confirmed [U]. |
| 14 | **USCG National Response Center (NRC)** (chemical/oil spills) | https://nrc.uscg.mil/ → `FOIAFiles/CY90.xlsx … CY26.xlsx` | [V-data] (CY24.xlsx, 15 MB) | 10 sheets: CALLS, INCIDENT_COMMONS (lat/lon in deg/min/sec), INCIDENT_DETAILS (`MEDIUM_DESC, BODY_OF_WATER, WATER_SUPPLY_CONTAMINATED`), MATERIAL_INVOLVED (`IF_REACHED_WATER, AMOUNT_IN_WATER`), TRAINS/VESSELS… | **2024: 22,966 reports; 12,249 with medium = WATER; only 2,830 of those (23%) have coordinates.** 112 flagged water-supply-contaminated. Top materials: unknown oil 4,356, diesel 1,659. CY26 updated 2026-10-20. | US Government | Coordinates are often missing, so geocoding (address/waterbody) is needed. Mostly coastal or vessel incidents. | **Initial, unvalidated reports** (homepage disclaimer) [V-doc]. Duplicates and "UNKNOWN SHEEN" are common; quantities often unknown. |
| 15 | **EPA ECHO sewer-overflow/bypass event tables** (SSO/CSO) | https://echo.epa.gov/files/echodownloads/all_sewer_overflow_and_collection_systems_tables.zip | [V-data] | `sewer_overflow_bypass_start/end_datetime, duration_hours, latitude_measure, longitude_measure, receiving water`, type and cause tables | **6,501 events**: SSO 5,416, CSO 567, bypass 518. 2019–2026, with 84% in 2025–26. Lat/lon for 6,359. | US Government | Good coordinates | **Coverage reflects electronic-reporting adoption, not occurrence.** IL 2,073, GA 1,195 and PR 990 dominate. |
| 16 | **ECHO NPDES DMR / effluent violations** (discharger side) | https://echo.epa.gov/tools/data-downloads (`npdes_dmrs_fyYYYY.zip`, `npdes_eff_downloads.zip`) | [V-doc] (file list) | Permit, outfall, parameter, limit, DMR value, violation | FY2009–FY2026 | US Government | Outfall layer (`npdes_outfalls_layer.zip`) gives locations. | Monthly self-reports, often averaged. An exceedance is not an ambient event. |
| 17 | **EPA SDWIS / SDWA violations** (drinking water) | https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip (558 MB, 2026-10-14) | [V-doc] (fields); not downloaded (size) | `VIOLATION_CODE, CONTAMINANT_CODE, VIOLATION_CATEGORY_CODE, IS_HEALTH_BASED_IND, NON_COMPL_PER_BEGIN/END_DATE, PUBLIC_NOTIFICATION_TIER` | Quarterly refresh | US Government | Water system, not a point. Service areas are coarse. | Compliance periods (monthly/quarterly), not event times. |
| 18 | **Boil-water advisories** | No national US database found | [U] | — | — | — | — | SDWIS Tier-1 public-notification violations are a partial, lagged proxy. |
| 19 | **Florida FWC Fish Kill Hotline** | ArcGIS REST: https://gis.myfwc.com/mapping/rest/services/Projects_FWC/FishKillHotline_forDEP/MapServer (layer 1 = 10 years) | [V-data] | `DateOfCall, KillCounty, WaterBodyName, LatDD, LonDD, Species, SuspectedCause` | **25,764 reports, 2011-01-02 to 2026-10-15.** Causes: Red Tide 7,489; Unknown 6,849; **Low DO 2,699**; Algae Bloom 2,421; Low Temperature 789. | State | Lat/lon present, but the service says points are a "general area" [P] | Public calls, so duplicates come with media attention. Cause is "suspected". |
| 20 | **Delaware fish kills** (tidal/non-tidal) | https://data.delaware.gov/resource/ycvg-4ys9.json; `8pfi-yck7` | [V-data] (WebFetch) | `location, county, watershed, basin, year, date, species, no_killed, probable_cause` | Tidal: 234 rows, 1981–2026 | State open data | No coordinates | Small n |
| 21 | **Iowa DNR fish kill database** | https://programs.iowadnr.gov/fishkill/ | [V-doc] | Waterbody, date, cause (animal waste, ammonia/fertilizer, …) | About 1,084 event IDs (an inference from the latest ID); 2024–26 recent | — | Map/table only; no export | Investigated kills only |
| 22 | **NY "Sewage Pollution Right to Know" (SPRTK)** (sewage discharges) | https://www.dec.ny.gov/fs/data/SPRTK_Data.xlsx (→ extapps.dec.ny.gov), 2021–present; https://www.dec.ny.gov/fs/data/Sewage_Discharge_Reports.xlsx for 2013–2021; page https://dec.ny.gov/environmental-protection/water/water-quality/sewage-pollution-right-to-know | [V-data] (2021+ workbook, 11.5 MB) | From Aug 2021: `notificationId, facilityName, latitude, longitude, sprtkStartDischargeDatetime, sprtkEndDischargeDatetime, dischargeDuration, receivingWaterBody, quantity, treatedState, dischargeReason…` | **56,058 notification rows in 27 tabs, Jan 2021–Aug 2026.** Schema changes between tabs. Recent tabs have lat/lon on every row. | State | Good coordinates (recent). Useful for Hudson, Long Island Sound and Great Lakes USGS sites. | DEC does not quality-check the data. Volumes are estimated; durations can be negative or zero-volume; the same discharge appears in daily and termination notices [V-doc]. Impossible dates occur (a start date of 2026-12-30). |
| 23 | NY CSO outfalls (counts per outfall) | https://data.ny.gov/d/ephi-ffu6 | [V-data] (catalog) | Outfall, receiving water, number of overflow events, lat/lon | 2013–2020 | — | Outfall points | Counts, not event times |
| 24 | **Michigan EGLE CSO/SSO/RTB** | Annual reports, e.g. https://www.michigan.gov/documents/egle/wrd-2020-CSO-SSO-RTB_Annual-Report_729069_7.pdf | [P] | The annual-report appendix has an event table (start/end, waterbody, outfall, cause, volume) [P] | Series since July 2000 [P] | State | PDF tables, so parsing is needed. | — |
| 25 | **England EA Event Duration Monitoring (EDM) annual returns** | https://www.data.gov.uk/dataset/19f6064d-7356-466f-844e-d20ea10ae9fd/ (zips via environment.data.gov.uk, 2020–2025) | [V-doc] | Per overflow per year: spill count and hours, EDM operability %, reasons | About 14,000 overflows [P] | EA attribution | Not event-level; England only | **Revised after publication.** Anglian Water data were revised in Aug 2026 [V-doc]. |
| 26 | **England National Storm Overflows Hub** (near-real-time start/stop) | Launched Nov 2024 by Water UK with Stream (https://www.streamwaterdata.co.uk/) | [P] | Overflow active status, start/stop | Near real time, with an API used by Rivers Trust and Surfers Against Sewage [P] | [U] | — | **No history**, so you must archive it yourself. |
| 27 | **NOAA Storm Events Database** (floods etc.) | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ (`StormEvents_details-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz`, plus locations and fatalities) | [V-data] (2024 file) | 51 columns incl. `EVENT_TYPE, BEGIN_DATE_TIME, BEGIN_LAT/LON, FLOOD_CAUSE, EVENT_NARRATIVE` | 2024: 69,801 events; **7,133 Flood + Flash Flood, 100% with BEGIN_LAT**. FLOOD_CAUSE includes "Heavy Rain / Burn Area" (47). | US Government | Points or zones. Good for covariates and for excluding flood-driven sensor artifacts. | NWS reporting practices vary by office. |
| 28 | **MTBS burned areas** (wildfire) | Points: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/fod_pt_shapefile/mtbs_fod_pts_data.zip; perimeters (390 MB): `.../burned_area_extent_shapefile/mtbs_perimeter_data.zip`; portal https://burnseverity.cr.usgs.gov/ | [V-data] (points) | `event_id, incid_name, incid_type, ig_date, burnbndac, burnbndlat/lon, …` | **30,932 fires, 1984–2026. Only 69 in 2025 and 21 in 2026**, so latency is about 1–2 years. Mapping thresholds: > 1,000 ac (West) / > 500 ac (East) [V-doc metadata]. | US Government | Intersect perimeters with NLDI upstream basins to get the burned fraction upstream of each gauge. | Large fires only. For recent fires use NIFC/WFIGS [U]. |
| 29 | **NWS NWPS flood categories** (stage thresholds) | `https://api.water.noaa.gov/nwps/v1/gauges/{LID}` | [V-data] | `flood.categories.{action,minor,moderate,major}.stage/flow`, `usgsId` | Example: BRKM2 (USGS 01646500) action 5 ft, minor 10, moderate 12, major 14 | US Government | Direct USGS ID | Flood categories are regulatory-style thresholds already joined to gauges. |
| 30 | **USGS continuous sensors → derived events** (DO, SC, turbidity, nitrate, chl, phycocyanin) | New OGC API: `https://api.waterdata.usgs.gov/ogcapi/v0` (v1 since Sep 2026) collections `continuous`, `daily`, `time-series-metadata`, `time-series-revisions`; legacy `waterservices.usgs.gov` until 22 Feb 2027 | [V-data] | Every value carries `approval_status` (Provisional/Approved), `qualifier` and `last_modified`. The metadata include a `thresholds` field. | Active real-time surface-water sites (legacy site service, summed over HUC2): **DO 1,011; SC 1,595; turbidity 976; nitrate (99133) 200** | Public domain | The best anchor: NLDI links them natively. | Fouling, drift, ice and burial artifacts. Provisional values get revised, and the API has rate limits (HTTP 429 observed). |
| 31 | **NEON aquatic water quality** (DP1.20288.001) | `https://data.neonscience.org/api/v0/products/DP1.20288.001` | [V-data] | DO, SC, pH, turbidity, chl-a, fDOM with QF flags | **34 sites, 2016-03 → 2026-09** (about 1-month latency); RELEASE-2021…2025 | CC0 [U] | Site coordinates; also targets of the NEON forecasting challenge | Releases are re-processed, so pin the release version. |

### a.2 Landmark incidents: can public sensors see them?

I checked the continuous-parameter coverage of each site through USGS `time-series-metadata` [V-data].

| Incident | Date | What happened | USGS sites checked (continuous WQ at the time?) | Visible in public sensor data? |
|---|---|---|---|---|
| Toledo, OH do-not-drink advisory (Lake Erie microcystin) | Aug 2014 (about 2–4 Aug; affected population about 400–500k) [U for details] | Toxin in finished water at the Lake Erie intake | 04193500 Maumee R. at Waterville: DO/SC/pH/turbidity only from 2021-11; chl from 2022. NOAA SI for 2014 = 5.1 (reprocessed) [V-data]. | **No.** The toxin is not sensed and WQ sensors came later. Satellite CI can see the bloom, not the toxin. |
| Elk River MCHM spill, Charleston, WV | 9 Jan 2014 | About 7,500–10,000 gal of crude MCHM, about 1 mile upstream of the WV American Water intake; up to 300,000 residents under a do-not-use order [P, Wikipedia citing the WV After Action Review] | 03198000 Kanawha R. at Charleston: Q/GH only; 03197000 Elk R. at Queen Shoals is upstream | **No** |
| East Palestine, OH derailment | 3 Feb 2023 | Vinyl chloride, butyl acrylate and others reached Sulphur Run → Leslie Run → Bull Ck → N. Fork Little Beaver Ck → Little Beaver Ck → Ohio R. ODNR estimated about 43,000 aquatic animals killed [P]. | 03109500 Little Beaver Ck: Q/GH only | **No** (grab sampling only) |
| Gold King Mine, CO | 5 Aug 2015 | About 3 million gal of acid mine drainage into Cement Ck → Animas → San Juan [P] | 09358550, 09358000, 09359020, 09361500, 09363500, 09365000, 09368000, 09371010: **SC/pH/turbidity series all begin in 2016** | **No.** The continuous monitors were installed after the event. |
| Dan River coal ash, NC | 2 Feb 2014 | About 39,000 t of ash and 27 million gal of wastewater [P] | 02071000 Dan R. nr Wentworth: Q/GH only | **No** |
| Trinseo latex spill, Bristol, PA (Delaware R.) | Mar 2023 [U] | Latex emulsion to Otter Ck → Delaware R.; Philadelphia advisory [U] | 01463500 Trenton (upstream, DO/SC/pH/turbidity); 01467200 Penn's Landing | **Unclear.** Sensors exist, but the acrylates are not sensed. |
| **Des Moines / Raccoon River nitrate** | 2013–2025 | NO₃+NO₂-N above the 10 mg/L MCL upstream of the Des Moines Water Works intakes | **05484500 Raccoon R. at Van Meter: in-situ nitrate since 2012** | **Yes.** 2015 had **243 of 345 days > 10 mg/L** (max 18.7, approved data). 2012–2025 had **14 onsets** (crossing > 10 after ≥ 14 d < 9) in 4,649 days [V-data]. |

**Takeaway:** headline disasters are mostly invisible to public sondes. Early-warning claims should be made about **sensor-observable event classes**, with incidents used as illustrative case studies only after a coverage check.

### a.3 Joining events to sensors (recipe, with working calls)

The USGS Network-Linked Data Index (NLDI) is at **`https://api.water.usgs.gov/nldi/linked-data/`**. The old `labs.waterdata.usgs.gov/api/nldi` path returns 404 [V-data].

1. **Snap an event point to an NHDPlus flowline:** `GET .../comid/position?coords=POINT(lon%20lat)`. Example: `POINT(-81.6326 38.3498)` → COMID 6925247.
2. **Downstream sensors within D km:** `GET .../comid/{COMID}/navigation/DM/nwissite?distance=50`. The example returned 12 NWIS sites, with reach `measure` values (03198000 Kanawha at Charleston, Winfield Dam, …). Use `DD` to include diversions, and `UM`/`UT` for upstream mainstem or tributaries.
3. **Upstream drivers:** `GET .../nwissite/USGS-05484500/basin` returns the upstream basin polygon, to intersect with MTBS perimeters, SSO points, NRC spills and so on. `GET .../nwissite/USGS-05484500/navigation/UM/nwissite?distance=100` returned 5 upstream sites.
4. **Rule for "event → sensor" labels:** a sensor is eligible for an upstream incident only if:
   - it is downstream on the network within D km (default 50 km); and
   - the expected travel time (distance ÷ velocity, with velocity from NHDPlus EROM or a conservative 0.3–1 m/s range) falls inside the forecast horizon; and
   - no major confluence dilutes the flow more than k-fold (drainage-area ratio < 5, using NWIS `drainage_area`).

   Lakes need polygon containment, not a point snap. For lake HABs, use NHD waterbody polygons and CyAN lake IDs.
5. Record the snap distance. Drop events snapped more than 500 m from a flowline unless they were geocoded to a named waterbody.

### a.4 Label noise and reporting bias: cross-cutting rules

- **Effort and complaint bias.** HAB reports (CA FHAB, NY, OR) and fish-kill hotlines scale with visitors, media attention and staffing. CA reports peaked at 737 in 2022 and dropped to 296 in 2024. Never treat "no report" as "no event". Use these sources only for *positive* labels near sensors, and estimate negatives from sensor-derived definitions.
- **Onset uncertainty.** First report ≠ onset, and lift dates are mostly missing (238 of 3,297 in CA). Never compute lead time against a report date. Compute it against a sensor-defined onset, or use report-based events only as "occurred within week w".
- **Revisions.** The NOAA severity index was reprocessed (2015: 10.5 → 8.4). EA EDM returns were revised in Aug 2026. USGS provisional data get approved or edited, and NEON publishes annual re-releases. **Freeze and hash each label vintage**, and report results per vintage.
- **Location error.** Only 23% of NRC water-medium reports have coordinates. FWC points mark a general area. NY HAB rows are lake centroids.
- **Coverage-driven artifacts.** The ECHO SSO table reflects which states e-report (IL, GA, PR), not where overflows occur. The IWAP Rhine report states explicitly that message counts cannot be read as water-quality trends because monitoring keeps improving [V-doc].
- **Duplicates.** NY SPRTK repeats each discharge in daily and termination notices; NRC has duplicate calls. De-duplicate on (facility, receiving water, start time ± 6 h).
- **Garbage values.** Impossible dates appeared in CA FHAB (2555, 2626), NY SPRTK (a 2026-12-30 start) and Chicago (year 0023). The v1 review also found USGS −999999 missing codes inside the normalization statistics. Validate every field against physical and calendar ranges.

### a.5 Which catalogues to use (ranking for v2)

1. **USGS continuous sensors** (DO/SC/turbidity/nitrate), with sensor-derived, regulation-anchored events. These give the largest n, objective onsets and a live feed.
2. **CyAN lake-week bloom labels**, plus EPA's weekly forecast as the operational baseline.
3. **NEON aquatic sites**, plus the NEON/VERA challenges for independent prospective scoring.
4. **Chicago beach lab data** (daily FIB) for a beach-nowcast task with a persistence benchmark.
5. **CA FHAB, NY HABs, FL FWC (low-DO kills), NY SPRTK, ECHO SSO and NRC** as secondary positive labels and context layers, each with its known bias stated.

---

## (b) Evaluation methodology and recommended metrics

### b.1 Diagnosis of v1 in standard terms

- **Unit mismatch.** The false-positive rate was measured per 2-hour window, but "detection" meant any alarm within 45 days. An alarm rate of λ per day makes P₀ = 1 − e^(−λW) for a random date.
  - With λ = 11.2/30 d⁻¹ and W = 45 d, P₀ ≈ 1 − e^(−16.8) ≈ 1.
  - Even at λ = 1/30 d⁻¹, P₀ = 1 − e^(−1.5) = 0.78.
  - At λ = 1/30 d⁻¹ and W = 3 d, P₀ = 0.095.

  The placebo rates the review measured (98–99%) are exactly this chance level. *(Derived arithmetic from the numbers in `/home/bcheng/STOCKHOLM/REVIEW_REPORT.md`, assuming Poisson alarms. Autocorrelated alarms lower P₀ somewhat, but not to near zero.)*
- **Conditioning on events.** v1 looked for precursors only before known events: the "prosecutor's fallacy" of Boettiger & Hastings (2012). The remedy is to evaluate on *all* issue times, or equivalently against matched placebo onsets.
- **In-sample calibration.** The threshold was calibrated on the same windows used to estimate the false-positive rate. The calibration period must be disjoint in time and must precede the test period.
- **Look-ahead.** The z-scores used whole-record statistics, and the fine-tune overlapped the evaluation windows. This is leakage in the sense of Kapoor & Narayanan (2023).
- **Selection on outcomes.** The detector was chosen from at least 5 candidates using test outcomes. Pre-register one model, or choose on the validation period and test once. Correct for the selection if not (White 2000; Hansen 2005).

### b.2 Forecast framing that avoids these traps

- **Forecast object.** At each issue time *t* (for example daily at 06:00 local, or every 6 h), the system emits p̂(s,t,h) = P(onset of event E at site *s* in (t, t+h]). It may use only data whose *arrival time* is ≤ t: respect telemetry latency of about 1 h, and never use backfilled values. USGS WaterAlert notes stations usually transmit about hourly [V-doc].
- **Scoring population.** Score every (s, t) with valid inputs, not just pre-event windows. Exclude issue times while an event is ongoing, because an onset needs a preceding non-event state.
- **Deterministic alarms** come from a frozen threshold on p̂ or on a detector score. An *alarm episode* is a maximal run of consecutive alarmed issue times.
  - An episode is a **hit** if an onset occurs in [episode start, episode start + h]. Otherwise it is a **false alarm**.
  - An onset with no alarm episode starting in [onset − h, onset) is a **miss**.
  - **Warning time** = onset − start of the hit episode. It lies in [0, h] by construction. Report alarms after onset separately as "nowcast detections" with negative lead.

### b.3 Metric menu

| Family | Metric | Why / caveat | Key refs |
|---|---|---|---|
| Probabilistic | **Brier score (BS)**, **BSS = 1 − BS/BS_ref** against (i) site × day-of-year climatology and (ii) persistence | A proper score that rewards calibration and resolution. Use the Murphy (1973) decomposition (reliability / resolution / uncertainty). For a Bernoulli forecast, CRPS reduces to BS, which is what NEON/VERA compute. | Brier 1950; Murphy 1973; Gneiting & Raftery 2007; Hersbach 2000 |
| Probabilistic | **Reliability diagram with consistency bars**; log score | Shows whether "30%" means 30% | Bröcker & Smith 2007 |
| Discrimination | ROC-AUC, **PR-AUC (state the prevalence)** | ROC is insensitive to the base rate; PR is more informative for rare events | Saito & Rehmsmeier 2015; Davis & Goadrich 2006; Mason & Graham 2002 (significance of ROC area) |
| 2×2 at a frozen threshold | POD (hit rate), **FAR (false-alarm ratio)**, POFD, CSI, ETS/GSS, **Peirce (H−F)**, **SEDI** | ETS is only asymptotically equitable. SEDI stays informative as the base rate → 0. | Hogan et al. 2010; Stephenson et al. 2008; Ferro & Stephenson 2011; JWGFVR site |
| Event level | **POD at a fixed false-alarm budget** (false-alarm episodes per site-month), **false alarms per site-month realized in test**, time-in-alarm fraction τ, warning-time distribution | This is what operators feel. EPA lists "number of invalid alerts per month" as a data-quality objective [V-doc, EPA 817-B-16-003]. | EPA 817-R-13-002 (EDS Challenge); Fawcett & Provost 1999 (AMOC) |
| Chance-corrected event level | **Molchan error diagram** (miss rate ν against alarm fraction τ; random ν = 1 − τ), area skill score; **placebo POD** and real − placebo difference | Directly tests "better than alarming at random with the same τ" | Molchan 1991; Zechar & Jordan 2008 |
| Decision value | **Relative economic value V(C/L)** curve | Lets a utility read value at its own cost/loss ratio (for example PAC dosing against a do-not-drink event). V_max occurs at C/L = base rate and equals the Peirce score (a standard result attributed to Richardson 2000; from memory [U]). | Richardson 2000; Murphy 1977 |
| Detector-only (unsupervised scores) | VUS-PR (threshold-free, buffer-tolerant) and affiliation precision/recall (compared with random) | **Never use point-adjusted F1.** Report these as secondary metrics only, because they do not encode lead time. | Paparrizos et al. 2022a,b; Huet et al. 2022; Kim et al. 2022; Liu & Paparrizos 2024 |

### b.4 What the anomaly-detection scoring literature says (and what to avoid)

- **NAB** (Lavin & Ahmad 2015) scores detections inside labeled windows with a function that rewards earliness, plus application profiles. Singh & Olinsky (2017) and Wu & Keogh (2021/2023) criticise NAB-style benchmarks for ambiguous windows, unrealistic anomaly density, mislabeled data and trivially solvable series. **Do not use NAB scores as headline numbers.**
- **Range-based precision/recall** (Tatbul et al. 2018) adds existence, overlap, cardinality and position-bias terms. It is flexible, but has many tunable knobs, so pre-specify them.
- **Point adjustment (PA).** Kim et al. (AAAI 2022) show that PA counts a whole anomaly segment as detected if any point in it is flagged. That lets even random scores reach "state-of-the-art" F1, and an untrained model rivals published methods. **This is the anomaly-detection twin of SENTINEL's 45-day "any alarm" rule.**
- **VUS** (Paparrizos et al., PVLDB 2022) integrates ROC/PR over buffer lengths, removing threshold and lag sensitivity. TSB-AD (Liu & Paparrizos, NeurIPS 2024 D&B) finds VUS-PR the most reliable of the measures it examined, and finds simple statistical methods often beat deep models [P].
- **Affiliation metrics** (Huet et al., KDD 2022) are local, parameter-free, and **normalised against a random predictor**. They show adversarial predictors can game earlier event-based metrics.
- **Other metrics.** eTaPR (Hwang et al., CIKM 2019) and composite F-scores (Garg et al., IEEE TNNLS 2022) exist. Sørbø & Ruocco (DMKD 2024) survey roughly 20 metrics and recommend choosing them by application needs.
- **Operational precedent.** EPA's EDS Challenge rejects per-time-step ROC for alarm systems: "an EDS that produced a two hour alert once a month would appear equivalent to an EDS that produced a two minute alert twice a day!" Instead it plots **% events detected against invalid alerts per week** over thresholds [V-doc].

### b.5 Lead time without look-ahead (checklist)

1. **No centered filters.** Use causal filters only (trailing windows, one-sided EWMA). Fit no smoother on the full series.
2. **Normalisation and climatology** must come from the training period only, per site. Recompute them for each rolling origin.
3. **Respect data latency.** Store each observation's arrival timestamp, or the API's `last_modified`, and feed the model only what was available at *t*.
4. **Provisional vs approved data.** Retrospective tests on approved data are likely optimistic, since approval typically edits or removes sensor artifacts (an inference, not measured here). Report both, if you capture provisional vintages prospectively. The new USGS API returns `approval_status` and `last_modified` per value [V-data].
5. **Onset definition must be causal-free:** the onset is defined from observations alone, and lead time is measured only to that onset.
6. **Lead time comes only from hit episodes inside [onset − h, onset).** Never quote the earliest alarm in a long lookback.

### b.6 Placebo, random-onset and surrogate controls (mandatory)

- **Season-matched placebo onsets.**
  - For each real onset at site *s* on day-of-year *d*, draw K = 20 pseudo-onsets at the same site in other years within ±30 days of *d*.
  - Each must sit in clean periods (no event within ±14 d) with the same data-coverage filters.
  - Apply the *identical* scoring rule. Report POD_real, POD_placebo, and their difference with a 95% bootstrap CI (resampling events).
  - Get a permutation p-value by shuffling the real/placebo labels within site-season strata.
- **Circular-shift alarm surrogates.** Shift each site's alarm series by a random offset of at least 30 days, which preserves τ and the autocorrelation. Recompute POD to form a null distribution.
  - Event coincidence analysis (Donges et al. 2016; CoinCalc, Siegmund et al. 2017) formalises the precursor coincidence rate with analytic Poisson and surrogate tests.
- **"Random alarm at the same τ"** is the Molchan diagonal. Report the area skill score against it (Zechar & Jordan 2008).
- **Sensor-fault control.** Report the share of false alarms that coincide with data later flagged, edited or deleted during USGS approval. These are equipment artifacts, not water-quality events.

### b.7 Matched false-alarm comparisons

- Compare every method at **equal false-alarm budgets**. Plot POD (warning time ≥ L) against false-alarm episodes per site-month: an AMOC/FROC-style curve, the form used in the EDS Challenge.
- Report partial AUC over the operationally relevant range, for example 0–2 false alarms per site-month.
- Report the test-period false-alarm rate realised at the threshold frozen during calibration. A mismatch is itself a finding.

### b.8 Blocking, rolling origin, and spatial generalisation

- **Temporal.** Train ≤ 2021; calibrate/validate 2022–2023; test once on 2024–2025; then prospective.
  - Within training, use rolling-origin or blocked CV with gaps of at least the horizon plus autocorrelation length (Tashman 2000; Bergmeir & Benítez 2012; Bergmeir, Hyndman & Koo 2018).
  - For non-stationary series, out-of-sample holdouts estimate error better than shuffled CV (Cerqueira, Torgo & Mozetič 2020).
- **Spatial.** Hold out whole HUC4 basins, with sites never seen in training (Roberts et al. 2017). Keep NEON and USGS separate, and state any pretraining overlap. The v1 review found 11 of 17 "zero-shot" USGS episode stations were pretraining stations.
- **Pseudo-prospective replay.** For each month of 2024–2025, refit only on data before the month start, freeze, and forecast the month. This is the CSEP "pseudo-prospective" idea (Schorlemmer et al. 2018).

### b.9 Calibrating alarm thresholds on held-out time

- Pick the threshold on the **calibration years only**, to meet a pre-registered budget (default **≤ 1 false-alarm episode per site per 30 days**). Pool across sites, or use site-specific thresholds shrunk toward the pooled value. Freeze it.
- For CUSUM, set the decision interval for an in-control **ARL₀ ≥ 30 days** on calibration clean periods, and report ARL₁ (detection delay). CUSUM is minimax-optimal for delay at a given ARL₀ (Page 1954; Lorden 1971).
- **Conformal options.**
  - Inductive conformal anomaly detection bounds the false-alarm rate at ε under exchangeability (Laxhammar & Falkman 2015). Conformal p-values support multiple-testing control (Bates et al. 2023).
  - Under drift and seasonality, use adaptive conformal inference (Gibbs & Candès 2021) or conformal PID control (Angelopoulos, Candès & Tibshirani 2023). State that guarantees are marginal and long-run, not per site-month. The v1 review flagged a misinterpreted conformal guarantee.

### b.10 Statistical testing and multiplicity

- **Uncertainty.** Use block or stationary bootstrap over site-months (Künsch 1989; Politis & Romano 1994) for all CIs. Compare models with paired score differences: the Diebold–Mariano test (1995), Ferro (2007) for Brier scores, and DelSole & Tippett (2014).
- **Multiplicity.** Use one primary endpoint per hazard, with Holm across hazards (Holm 1979). Use Benjamini–Hochberg FDR (1995) for site, horizon and secondary analyses. Wilks (2016) shows how field significance is routinely overstated across many sites. Report every analysis, including null results (Gelman & Loken 2014, forking paths).
- **Model search.** If more than one candidate model is evaluated on test, correct with White's Reality Check (2000) or Hansen's SPA (2005), or simply do not do it.

### b.11 Prospective evaluation and pre-registration precedents

- **CSEP** (earthquake forecasting) runs models frozen in code, with forecasts registered before the target period and fixed tests (Jordan 2006; Schorlemmer et al. 2007, 2018).
- **NEON Ecological Forecasting Challenge** (Thomas et al. 2023) [V-doc]:
  - Still active: 110,838 forecasts submitted; page updated 2026-10-21.
  - Aquatics variables are `oxygen`, `temperature` and `chla` (daily, 30-day horizon, about 3-day latency, 34 sites).
  - CRPS via `scoringRules`; baselines `climatology` and `persistenceRW`.
  - Forecast format is the EFI standard (Dietze et al. 2023).
  - The v1 review notes this challenge predates SENTINEL's "first pre-registered" claim.
- **VERA challenge** (Virginia Tech; FCR/BVR reservoirs and TUBR inflow) [V-doc]:
  - 22,752 forecasts submitted.
  - Variables include `DO_mgL_mean`, `Chla_ugL_mean` and **`Bloom_binary_mean` (chl-a > 20 µg/L; bernoulli family)**.
  - Scored with CRPS and log score; baselines climatology and persistenceRW.
- **USGS beach nowcasts** (Francy et al. 2013, SIR 2013-5166) [V-doc] set performance goals *before* the season:
  - ≥ 80% correct, sensitivity ≥ 50%, specificity ≥ 85%, and at least 5 percentage points better than persistence (the previous day's E. coli).
  - Only 24 of 42 models beat persistence by that margin.
  - This is a direct template for SENTINEL's beach task.
- **Pre-registration venues:** OSF Registries (https://osf.io/registries), AsPredicted (https://aspredicted.org/). Rationale: Nosek et al. 2018.

---

## (c) Operational systems and platforms: landscape and gaps

### c.1 Comparison table

| System | Operator | Hazard | Method | How performance is reported | Status (2026) | Open? |
|---|---|---|---|---|---|---|
| **CANARY-EDS** | US EPA / Sandia | Drinking-water distribution contamination | Residual-based event detection on multi-sensor streams (algorithm details in the manual on cfpub.epa.gov) | Evaluated in the EDS Challenge (below). No performance claims in the repo. | **Archived and deprecated 24 Jun 2026**; "should not be used in production"; last release v4.3.3 [V-doc] | Yes (MATLAB LGPL-2.1; Java Apache-2.0) |
| **EPA EDS Challenge** (evaluation, not a system) | EPA OGWDW, report **EPA 817-R-13-002** (Apr 2013) | Distribution-system events | 5 EDSs: CANARY, ana::tool (s::can), OptiEDS (OptiWater), BlueBox (Whitewater Security), Event Monitor (Hach). Run on utility baselines from 6 stations, with **576 simulated events (96 per station)** superimposed by EDDIES. | **% events detected vs invalid alerts per week (0–3.5/week shown)**; event-level. Rejects per-time-step ROC. | Findings: no "best" EDS; detectability depends on background variability; fewer invalid alerts means lower sensitivity; simulated events were detected more reliably than real baseline events [V-doc] | Report public |
| **EPA online WQ monitoring guidance** | EPA, **817-B-16-003** (source water, Sep 2016, 2019 version); **817-B-18-001** (distribution, Apr 2018) | Source and distribution water | Design guidance | Recommends data-quality objectives including **"number of invalid alerts per month"** [V-doc] | Current | Public |
| **SRBC Early Warning System** | Susquehanna River Basin Commission | Source-water contamination | 55 stations measuring at least pH, temperature and turbidity; time-of-travel tool | Not reported as skill [V-doc, EPA 817-B-16-003 case study] | Running [U for 2026] | Data partly public |
| **Delaware Valley EWS** | Philadelphia Water Dept. | Spills to Schuylkill/Delaware | 88 locations (4 intake stations + 84 USGS); hotline/web incident reports; spill-trajectory and time-of-travel tools; > 300 users from 50 organisations | Incident list only (e.g., 2012 vinyl chloride derailment) [V-doc] | [U] | Private (members) |
| **RAIN** (River Alert Information Network) | 51 utilities, PA DEP, WV DHHR, universities | Monongahela/Allegheny/Ohio | 29 stations (DO, NH₃, pH, SC, temperature, turbidity); threshold notifications | Example incident: 2010 bromide rise [V-doc] | [U] | Public data via USGS site [P] |
| **ORSANCO Organics Detection System** | ORSANCO | Ohio River spills (VOCs) | GC units at utilities; 16 stations, 13 on the Ohio River from Pittsburgh to Paducah (2017 annual report); founded 1978 after a major chemical release [P] | Detections and incident tracking (e.g., 2006 Beaver River ethanol derailment) [P] | orsanco.org unreachable from this server [U] | Partly |
| **Rhine International Warning and Alarm Plan (IWAP)** | ICPR (7 main warning centres R1–R7) | Sudden pollution in Rhine, Neckar, Main | Monitoring stations + reports; web-IWAP messages (warning / information / search) | Annual message statistics. **2025: 20 events (2 warnings, 18 information) + 4 search messages; one information message withdrawn after repeat measurements did not confirm it.** States that counts cannot be read as WQ trends [V-doc, ICPR report 320] | Active since 1985 [V-doc] | Reports public |
| **Rhine Alarm Model** | ICPR + CHR (post-Sandoz 1986) | Spill-wave travel | Flow-time model from Lake Constance to the North Sea plus tributaries; tracer-calibrated | "About 98% reliability" for wave development [V-doc] | New version expected by 2027 [V-doc] | Restricted |
| **Danube AEWS** | ICPDR (PIACs; expert units use the Danube Basin Alarm Model) | Transboundary accidents | Alert messaging + DBAM | 61 accidents May 1997–Dec 2006 [P] | icpdr.org unreachable [U] | Manual public [P] |
| **USGS WaterAlert** | USGS | Any real-time parameter | User-set thresholds → e-mail or text | None (pure threshold) | Active (accounts.waterdata.usgs.gov/wateralert) [V-doc] | Free |
| **USGS Water Data APIs** | USGS | Data | OGC API (v1 Sep 2026) | — | **Legacy WaterServices off on 22 Feb 2027**; slowdowns from 16 Nov 2026; Observations API decommissioned 19 Oct 2026 [V-doc] | Public; API key advised |
| **EPA CyAN / CyANWeb** | EPA, NASA, NOAA, USGS | Lake cyanobacteria | Satellite CI (OLCI 300 m) | Validation papers (e.g., Coffer et al. 2020; Seegers et al. 2021) | S3 portion concluding; data kept and reprocessed; Android app retired end-2024 [V-doc]/[P] | Public data |
| **EPA cyanoHAB forecast** | US EPA (water research program) | Lake-week bloom probability | Model trained on CyAN (Schaeffer et al. 2024) | "Overall prediction accuracy of 90 percent"; tends to overpredict; low false omission rate [V-doc] | Seasonal Apr–Nov | Dashboards and downloads |
| **NOAA Lake Erie HAB forecast** | NOAA NCCOS (+GLERL) | Western Lake Erie bloom (seasonal SI; daily/weekly position and mixing) | Statistical seasonal models on Maumee loads; satellite + hydrodynamic transport (≥ 96 h) | No skill statement on the page. SI series reprocessed [V-doc] | Active (bulletin dated 20 Oct 2026). **GLERL's experimental HAB Tracker retired and merged into the operational forecast** [V-doc] | Public |
| **State HAB dashboards** (NY, CA FHAB, OR, WA, OH) | States | HAB advisories | Reports, sampling, advisory tiers | Counts only | Active | Public (varying) |
| **USGS Great Lakes beach Nowcasts / EPA Virtual Beach** | USGS; EPA (Virtual Beach v3.0.7, Feb 2019; MLR/PLS/GBM) | Beach E. coli | Regression/GBM on turbidity, rain, waves… | **Sensitivity/specificity/accuracy against persistence, with pre-set goals** (Francy et al. 2013) | Virtual Beach "actively supported" (page updated 30 Mar 2026) [V-doc] | Free software |
| **FLARE** | Virginia Tech (Thomas et al. 2020) | Reservoir temperature and WQ | Iterative data-assimilation forecasts | Skill and uncertainty partitioning are reported in the paper (comparison details not re-checked [U]) | Research-operational | Open |
| **NEON / VERA challenges** | EFI RCN / Virginia Tech | Aquatic DO, chl-a, temperature, bloom | Community forecasts | **CRPS/log score against climatology and persistenceRW, prospective, public** [V-doc] | Active Oct 2026 | Open |
| Commercial: Hach Event Monitor, s::can ana::tool, Whitewater BlueBox, OptiWater OptiEDS | Vendors | Utility EDS | Proprietary | Only via the 2013 EDS Challenge [V-doc] | [U] | No |
| Commercial: Xylem/YSI, In-Situ, KISTERS, Aquatic Informatics, NexSens, LG Sonic, bbe, Fluidion, Proteus | Vendors | Sensors and cloud platforms | Thresholds/analytics | I found no public, independent detection or false-alarm statistics [U] | — | No |

### c.2 Observations on how existing systems report performance

- **Notification networks report incident counts, not skill.** This covers IWAP, AEWS, DV-EWS, RAIN, SRBC and WaterAlert. Their "early warning" value comes from spatial coverage, time-of-travel tools and fast human communication, not from prediction.
- **The two places with rigorous, operator-relevant scoring** are EPA's EDS Challenge (event-level detection against invalid alerts per week, on simulated events) and the USGS beach nowcasts (pre-set goals against persistence).
- **The newest public forecast** (EPA's cyanoHAB forecast) headlines "accuracy 90%". Accuracy can be high for a trivial model when the base rate is near 0 or 1. Skill against climatology or persistence is what a judge should ask for.
- **The infrastructure is in flux.** CANARY is archived, USGS legacy services shut down in Feb 2027, CyAN's Sentinel-3 project is concluding, and GLERL's HAB Tracker is retired.

### c.3 Gap analysis: what an open SENTINEL platform could credibly add

1. **A public, timestamped forecast ledger with verified skill against baselines.** No US WQ early-warning system I found publishes BSS, Molchan curves or false alarms per month against climatology or persistence. A daily scorecard of POD at the frozen false-alarm budget, realised false alarms per site-month and BSS would be new and verifiable.
2. **Regulation-native, multi-hazard alerts.** Event definitions would tie directly to EPA/state criteria: DO, chloride, E. coli BAV, microcystin, nitrate MCL, NWS flood stage. Each alert would state the criterion, its averaging rule and that it is "advisory, not a regulatory determination". WaterAlert offers only raw thresholds.
3. **Separating sensor faults from real events.** The EDS Challenge found background variability drives invalid alerts. Combining cross-sensor consistency, NLDI upstream/downstream corroboration and provisional-to-approved revision history (now exposed per value in the USGS API) addresses the most common operator complaint.
4. **Open upstream → downstream propagation for any US gauge or intake.** Rhine-Alarm-Model-style time-of-travel is available today only in closed regional systems (DV-EWS, SRBC). NLDI navigation plus NHDPlus velocities makes an open version possible.
5. **Honest context layers.** Overlay NRC spills, SPRTK/ECHO overflows, FWC fish kills, HAB advisories, MTBS burn scars and Storm Events, each with a bias badge (complaint-driven, effort-biased, revised, location-imprecise).
6. **Decision support for a user's own cost/loss ratio.** A utility enters C/L; the platform shows the threshold that maximises relative economic value for that user, and the expected alarms per month.
7. **Continuity.** Native support for the new USGS OGC API, and daily archiving of provisional vintages. I found no public USGS archive of pre-approval provisional values; the `time-series-revisions` collection documents post-approval revisions [V-doc], but whether earlier vintages exist elsewhere is unverified [U]. Such an archive would be a novel benchmark of provisional-data artifacts.
8. **Low-resource users.** Free, explainable alerts for small utilities and lake associations that cannot buy commercial EDS.

*What judges and water managers are likely to value:* real lead time measured honestly, a stated false-alarm rate per month, clear thresholds, an explanation of why an alert fired, visible uncertainty, and evidence it beats "just look at the current reading". These points are my inference from the EPA guidance and EDS Challenge findings above; I did not find a dedicated survey this session [U].

---

## (d) Recommended "Early Warning v2" protocol for SENTINEL

### d.1 Scope and principles

- Make **forecasting claims only for sensor-observable event classes** with large n.
- Treat landmark incidents as **narrative case studies only**, after a coverage check (a.2).
- Every number reported must have: a pre-registered definition, a frozen threshold, a chance or placebo comparator, the best simple baseline, and a CI from block bootstrap.

### d.2 Event definitions (pre-register exactly these)

| Hazard | Variable / source | Event (binary) | Duration / onset rule | Regulatory anchor (verified) |
|---|---|---|---|---|
| **H1 Hypoxia (rivers/streams/lakes)** | USGS DO (00300), NEON DO | **Primary:** DO < **3.0 mg/L**. Tiers at < 4.0 and < 2.0 mg/L. Coldwater/trout sites use < 4.0 (primary) and < 5.0. | ≥ 2 h sustained (≥ 8 consecutive 15-min values, ≥ 90% coverage). **Onset** = first qualifying time after ≥ 72 h with DO ≥ threshold + 1.0 mg/L and ≥ 80% coverage. Exclude: USGS equipment/ice qualifiers, values < 0 or > 25 mg/L, −999999 codes, and any run below threshold > 30 d (likely burial or fouling; flag for review). | EPA 1986 freshwater DO: warmwater other-life-stage **1-day minimum 3.0**, 7-day mean minimum 4.0, 30-day mean 5.5. Coldwater 1-day minimum 4.0. **"All minima should be considered as instantaneous concentrations to be achieved at all times"** (Gold Book, EPA 440/5-86-001, Table 1) [V-doc]. NY 6 NYCRR 703.3: non-trout daily average ≥ 5.0 and never < 4.0; trout ≥ 6.0 / ≥ 5.0; class D ≥ 3.0 [V-doc]. 2 mg/L is the conventional coastal "hypoxia" severity tier; I have not verified a specific source's unit convention [U]. |
| **H2 Salinization (road salt)** | USGS SC (00095) → chloride via a **site-specific regression** fitted on paired WQP chloride–SC samples (require ≥ 20 pairs, R² ≥ 0.9; sites without one use the statistical tier) | **Chronic:** estimated Cl⁻ 4-day mean > **230 mg/L**. **Acute:** 1-h mean > **860 mg/L**. | Onset after ≥ 7 d below 80% of the threshold | EPA 1988 chloride criteria (EPA 440/5-88-001): 4-day average 230 / 1-hour average 860 mg/L, each "not more than once every three years on the average", for chloride associated with sodium [V-doc]. Background: Corsi et al. 2010; Kaushal et al. 2018. |
| **H3 Turbidity excursion** | USGS turbidity (63680) | **Treatment-relevant excursion:** hourly median > background + 5 NTU when background ≤ 50, or > 1.10 × background when background > 50 (background = trailing 7-day median), for ≥ 6 h. **Statistical tier:** > training-period site P99 for ≥ 3 h. | Onset after ≥ 72 h non-event | No federal ambient numeric criterion; EPA refers to the Gold Book narrative [V-doc]. Washington WAC 173-201A-200(1)(e) uses the "5 NTU over background (≤ 50) / 10% (> 50)" form [V-doc]. That rule is borrowed here as an adaptation: Washington's "background" is a reference condition, e.g. one outside the discharge's influence (definition not re-checked [U]), whereas this protocol substitutes a trailing 7-day median at the same site. Drinking-water filtered-water limits are ≤ 0.3 NTU in ≥ 95% of monthly samples and never > 1 NTU (40 CFR 141.173(a)) [V-doc]. These are context, not ambient standards. |
| **H4 Nitrate at intakes** | USGS in-situ NO₃+NO₂ (99133) at sites upstream of DW intakes (NLDI DM ≤ 50 km) | Daily mean > **10 mg/L as N** | Onset after ≥ 14 consecutive days < 9 mg/L. At Van Meter this gives 14 onsets in 2012–2025, about 1 per site-year [V-data]. | MCL nitrate 10 mg/L as N; nitrite 1; total 10 (40 CFR 141.62) [V-doc] |
| **H5 Lake cyanoHAB (satellite)** | CyAN 300 m CI (lake-week), or EPA forecast labels | Lake-week bloom = EPA definition: **median lake chl-a ≥ 12 µg/L with cyanobacteria dominance** | Onset = first bloom week after ≥ 3 valid non-bloom weeks | WHO 2020 alert-level framework: chl-a 1 / **12** µg/L for drinking-water AL1/AL2 and **24 µg/L** for recreational use; biovolume 0.3 / 4 / 8 mm³/L (WHO/HEP/ECH/WSH/2020.6, Table 7.1) [V-doc] |
| H5b Lake bloom (in situ) | NEON lake chl-a; VERA | Daily mean chl-a > 20 µg/L (VERA `Bloom_binary`) | Onset after ≥ 7 non-bloom days | VERA target definition [V-doc] |
| H5c Toxin (retrospective, sparse) | WQP, CA FHAB, GLERL | Microcystins > **8 µg/L** (recreational); > 0.3 / 1.6 µg/L in finished water | Sample-based | EPA 822-R-19-001 (May 2019): **8 µg/L** microcystins, **15 µg/L** cylindrospermopsin, not to be exceeded on any single day as an advisory (≤ 3 excursions per season for assessment). EPA 820R15100 (2015) 10-day HAs **0.3 / 1.6 µg/L**. WHO 2020 provisional GVs for MC-LR: 1 (lifetime DW) / 12 (short-term DW) / 24 (recreational) µg/L. CA triggers 0.8 / 6 / 20 µg/L [all V-doc] |
| **H6 Beach FIB** | Chicago Beach Lab Data (and BEACON/WQP) | Single sample **E. coli > 235 cfu/100 mL** (culture). DNA-test threshold per the Park District [U]. | Daily issue at 08:00, before lab results | EPA 2012 RWQC (820-F-12-058): GM 126 / STV 410 (36 per 1,000) or GM 100 / STV 320 (32 per 1,000); 30-day GM, ≤ 10% STV excursion; **BAV 235** (or 190) cfu/100 mL. Enterococcus BAV 70 / 60; qPCR BAV 1,000 / 640 cce [V-doc]. |
| H7 Flood stage (context or optional) | NWPS categories + USGS stage | Stage ≥ NWS *minor* flood | — | NWPS API thresholds per gauge [V-data] |

Other anchors (for feature engineering or secondary tiers): pH 6.5–9 (EPA 1986) [V-doc]; ammonia 2013 freshwater acute 17 / chronic 1.9 mg TAN/L at pH 7 and 20 °C (EPA 822-R-13-001) [V-doc].
*Caveat:* the criteria are assessment rules with averaging periods and allowed frequencies. SENTINEL's events are **"criterion-anchored excursions"**, not regulatory violations. Say so on every figure.

### d.3 Forecast tasks

- **Issue times.** Daily at 06:00 local for H1–H4 and H6; weekly (Monday) for H5.
- **Horizons.** h = 1, 3, 7 d for H1–H4; 1 week for H5; same-day for H6 (lead about 18–24 h relative to lab results, as in USGS nowcasts).
- **Inputs at issue time *t*.** Only observations with arrival ≤ *t*. Weather forecasts (e.g., NWS) are allowed only as issued at *t*.
- **Eligibility.** A site enters the scoring set for a day if ≥ 80% of the previous 72 h are present.

### d.4 Baselines (all required, same inputs and splits)

1. **Climatology:** site × day-of-year onset frequency, smoothed by logistic regression on harmonics and fitted on training years.
2. **Persistence / current state:** P(onset | current value bin), for example DO now in [3, 4) → empirical P. NEON/VERA use `persistenceRW` for continuous targets.
3. **Threshold on current value:** an alarm if the current value is within δ of the criterion (DO < 4.0 for H1), with δ tuned on calibration.
4. **Logistic regression on lags:** last 24 h min/mean/slope, diel amplitude, temperature, discharge and its change, day-of-year harmonics, recent rain.
5. **GBM** (LightGBM/XGBoost) on the same features.
6. **SENTINEL deep model(s)** and the **v1 CUSUM detector**, the latter re-scored under v2 rules.
7. **Operational baselines** where they exist:
   - H5: EPA cyanoHAB weekly forecast; NOAA Lake Erie products for the western basin.
   - H6: persistence (previous day's result), with the Francy et al. goals as an external yardstick.
   - H7: NWS official forecasts.

### d.5 Splits, calibration, freezing

- **Splits.** Train ≤ 2021-12-31. Calibrate/validate 2022-01-01 to 2023-12-31. **Test once** on 2024-01-01 to 2025-12-31 (approved data). Spatial holdout: about 20% of sites by HUC4, never used in training or calibration.
- **Normalisation.** Training-period statistics only. Mask missing-value codes before computing statistics, to fix the v1 −999999 bug.
- **Thresholds.**
  - Each method's alarm threshold is set on the calibration period to **≤ 1 false-alarm episode per site per 30 days**. Pre-register a second budget of 2 per 30 days for sensitivity.
  - For CUSUM: ARL₀ ≥ 30 d.
  - Thresholds are frozen and their hashes recorded before test data are loaded. A script must refuse to read 2024+ data until the freeze file exists.
- **Model selection** uses calibration data only. At most one SENTINEL model per hazard goes to test. Any additional candidates are reported with a Reality-Check correction.

### d.6 Endpoints and success criteria

- **Primary (per hazard H1, H2, H4; H5 for the summer trial):** BSS against climatology at h = 3 d (H5: 1 week), pooled over test site-days.
  - **Success** = BSS > 0 with the lower 95% block-bootstrap bound (site-month blocks) > 0. Holm-adjust across hazards.
- **H3 (turbidity) and H6 (beach) are pre-registered as secondary.** H6 is judged against the Francy et al. (2013) goals: overall correct ≥ 80% and at least 5 points above persistence, sensitivity ≥ 50%, specificity ≥ 85%.
- **Key secondary:**
  - ΔBS against the persistence/threshold baseline and against GBM (paired block bootstrap; Diebold–Mariano).
  - POD (warning ≥ 24 h) at the frozen budget, with the **realised** test false-alarm rate per site-month.
  - **POD_real − POD_placebo** with a permutation p-value.
  - Molchan area skill score.
  - Reliability diagram and REV curve.
- **Report, never hide:** results on held-out basins, provisional vs approved scoring, and failure cases.
- **Claims allowed:** "skill at h days" only where the primary test passes. Lead time is quoted as the warning-time distribution of hit episodes at the frozen budget, never as the maximum lookback.

### d.7 Controls (all pre-registered)

- Season-matched placebo onsets (K = 20 per real onset).
- Circular-shift alarm surrogates (1,000 shifts ≥ 30 d).
- Random alarms at the same τ (Molchan diagonal).
- **Negative-control hazard:** run the H1 model to "predict" H1 at sites where DO never fell below 6 mg/L in training. It should yield almost no alarms; this checks false alarms on clean sites with real labels.
- **Sensor-fault audit:** share of false alarms matching periods later edited during USGS approval.

### d.8 Sample size and power (plan, then check)

- **Precision of POD.** The 95% CI half-width for POD ≈ 1.96·√(p(1−p)/n). That is ±0.18 at n = 30 and ±0.10 at n = 100 (p = 0.5).
- **Targets.** Aim for ≥ 50 onsets per hazard in test, and ≥ 30 in each prospective season, before making a claim. Below that, report the hazard as "underpowered".
- **Feasibility.** At one site, nitrate onsets run about 1 per site-year. With about 1,000 real-time DO sites and about 1,600 SC sites, pooled counts should be ample. **Count historical onsets per hazard on 2022–2023 data before registration** and put those counts in the registration.

### d.9 Pre-registration and live prospective trial (start now)

1. **By about 15 Nov 2026.** Freeze code, models, thresholds and event definitions. Register on OSF with:
   - this protocol, the site list rules, git commit hashes and model-weight SHA-256 hashes;
   - the analysis scripts;
   - the primary and secondary endpoints and the multiplicity plan.
2. **Pipeline.**
   - Daily cron on the server, using `api.waterdata.usgs.gov/ogcapi/v1` with an API key.
   - Write EFI-standard forecast files (`reference_datetime, datetime, site_id, family = bernoulli, parameter = prob, variable = event_id, prediction`).
   - Commit to a public repo, stamp each day's file hash with OpenTimestamps, and push a weekly Zenodo snapshot.
   - **Also snapshot the raw provisional inputs**, to build the provisional-vintage archive.
3. **External, independent scoring.** Submit daily `oxygen`, `chla` and `temperature` forecasts to the NEON challenge and `Bloom_binary_mean` / `DO_mgL_mean` to VERA.
4. **Seasons.**
   - Winter trial: 1 Dec 2026 – 31 Mar 2027 (H2, H3, H1 where relevant).
   - Summer trial: 1 Apr – 30 Sep 2027 (H1, H4, H5; H6 over the Chicago beach season).
   - Each season has one **pre-specified analysis date**, 14 Apr 2027 and 15 Oct 2027, scored on provisional data. A final re-score on approved data follows when USGS approves the data. No other efficacy peeking.
5. **Reporting template.**
   - Number of sites, site-days and onsets.
   - Calibration, test and prospective results side by side.
   - Placebo and surrogate nulls.
   - Every deviation from the registration, with its date.

### d.10 What to say about v1 now

Use the review's suggested sentence: *"…the label-free detector … raised a pre-onset alarm before random clean dates as often as before real events; we therefore report no validated discrimination or lead time."* Then re-score the v1 detector under v2 rules as one more baseline.

---

## (e) Risks

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| USGS WaterServices shutdown (22 Feb 2027); v0 → v1 redirects (mid-2027); API rate limits (429 observed) | Certain / high | Use v1 from day one with an API key. Cache locally, retry with backoff, monitor pipeline health. |
| Federal program changes: CyAN's Sentinel-3 portion concluding and CANARY archived [V-doc]; possible further agency budget or reorganisation changes [U] | Medium / medium | Mirror needed data. Make CyAN-dependent tasks optional. Pin data vintages. |
| Too few events in prospective seasons | Medium / high | Pool sites, pre-register minimum-n rules, use multiple hazards, run seasonal trials. |
| Label revisions (NOAA SI, EA EDM, USGS approvals, NEON releases) | High / medium | Hash and freeze each vintage; score both provisional and approved. |
| Sensor artifacts counted as events | High / high | Apply qualifier and range filters, minimum duration and hysteresis, manual-review flags, sensor-fault audit. |
| Reporting-bias labels used as negatives | Medium / high | Use incident catalogues only as positives and context. Derive negatives from sensor definitions. |
| Multiplicity and forking paths | High / high | One primary endpoint per hazard, Holm, frozen analysis code, OSF registration. |
| Baselines win (GBM or persistence beats the deep model) | Plausible | Report it. "Simple baselines match deep models" is already the paper's most defensible finding, and TSB-AD reports the same pattern. |
| Public-alert misinterpretation | Low / high | Label everything "research forecast, not an official advisory". Link to the state or EPA source. Never publish health advice. |
| Unverified items in this report | — | Re-check every [P]/[U] entry before citing it in the paper. |

---

## (f) References

All entries below were metadata-verified [V-ref] via Crossref/doi.org or the arXiv API unless otherwise tagged.

**Forecast verification**
- Jolliffe, I. T., & Stephenson, D. B. (Eds.) (2011/2012). *Forecast Verification: A Practitioner's Guide in Atmospheric Science* (2nd ed.). Wiley. doi:10.1002/9781119960003
- Wilks, D. S. (2019). *Statistical Methods in the Atmospheric Sciences* (4th ed.). Elsevier. doi:10.1016/C2017-0-03921-6
- WWRP/WGNE JWGFVR. *Forecast Verification: Issues, Methods and FAQ.* https://jwgfvr.github.io/forecastverification/ (redirect from https://www.cawcr.gov.au/projects/verification/) [V-doc]
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
- Murphy, A. H. (1973). A new vector partition of the probability score. *J. Appl. Meteorol.*, 12(4), 595–600. doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2
- Murphy, A. H. (1977). The value of climatological, categorical and probabilistic forecasts in the cost-loss ratio situation. *MWR*, 105(7), 803–816. doi:10.1175/1520-0493(1977)105<0803:TVOCCA>2.0.CO;2
- Murphy, A. H. (1993). What is a good forecast? An essay on the nature of goodness in weather forecasting. *Weather and Forecasting*, 8(2), 281–293. doi:10.1175/1520-0434(1993)008<0281:WIAGFA>2.0.CO;2
- Richardson, D. S. (2000). Skill and relative economic value of the ECMWF ensemble prediction system. *QJRMS*, 126(563), 649–667. doi:10.1002/qj.49712656313
- Stephenson, D. B., Casati, B., Ferro, C. A. T., & Wilson, C. A. (2008). The extreme dependency score: a non-vanishing measure for forecasts of rare events. *Meteorol. Appl.*, 15(1), 41–50. doi:10.1002/met.53
- Ferro, C. A. T., & Stephenson, D. B. (2011). Extremal dependence indices: improved verification measures for deterministic forecasts of rare binary events. *Weather and Forecasting*, 26(5), 699–713. doi:10.1175/WAF-D-10-05030.1
- Hogan, R. J., Ferro, C. A. T., Jolliffe, I. T., & Stephenson, D. B. (2010). Equitability revisited: why the "equitable threat score" is not equitable. *Weather and Forecasting*, 25(2), 710–726. doi:10.1175/2009WAF2222350.1
- Bröcker, J., & Smith, L. A. (2007). Increasing the reliability of reliability diagrams. *Weather and Forecasting*, 22(3), 651–661. doi:10.1175/WAF993.1
- Ferro, C. A. T. (2007). Comparing probabilistic forecasting systems with the Brier score. *Weather and Forecasting*, 22(5), 1076–1088. doi:10.1175/WAF1034.1
- DelSole, T., & Tippett, M. K. (2014). Comparing forecast skill. *MWR*, 142(12), 4658–4678. doi:10.1175/MWR-D-14-00045.1
- Mason, S. J., & Graham, N. E. (2002). Areas beneath the relative operating characteristics (ROC) and relative operating levels (ROL) curves. *QJRMS*, 128(584), 2145–2166. doi:10.1256/003590002320603584
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378. doi:10.1198/016214506000001437
- Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559–570. doi:10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. doi:10.1371/journal.pone.0118432
- Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proc. ICML 2006*, 233–240. doi:10.1145/1143844.1143874
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *IJF*, 22(4), 679–688. doi:10.1016/j.ijforecast.2006.03.001

**Event-based anomaly-detection evaluation**
- Lavin, A., & Ahmad, S. (2015). Evaluating real-time anomaly detection algorithms: the Numenta Anomaly Benchmark. *IEEE ICMLA 2015*, 38–44. doi:10.1109/ICMLA.2015.141; arXiv:1510.03336
- Singh, N., & Olinsky, C. (2017). Demystifying Numenta anomaly benchmark. *IJCNN 2017*, 1570–1577. doi:10.1109/IJCNN.2017.7966038
- Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., & Gottschlich, J. (2018). Precision and recall for time series. *NeurIPS 2018*. arXiv:1803.03639
- Kim, S., Choi, K., Choi, H.-S., Lee, B., & Yoon, S. (2022). Towards a rigorous evaluation of time-series anomaly detection. *AAAI*, 36(7), 7194–7201. doi:10.1609/aaai.v36i7.20680; arXiv:2109.05257
- Paparrizos, J., Kang, Y., Boniol, P., Tsay, R. S., Palpanas, T., & Franklin, M. J. (2022). TSB-UAD: an end-to-end benchmark suite for univariate time-series anomaly detection. *PVLDB*, 15(8), 1697–1711. doi:10.14778/3529337.3529354
- Paparrizos, J., Boniol, P., Palpanas, T., Tsay, R. S., Elmore, A., & Franklin, M. J. (2022). Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. *PVLDB*, 15(11), 2774–2787. doi:10.14778/3551793.3551830
- Wu, R., & Keogh, E. J. (2021, early access; journal version 2023). Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. *IEEE TKDE*. doi:10.1109/TKDE.2021.3112126; arXiv:2009.13807
- Huet, A., Navarro, J. M., & Rossi, D. (2022). Local evaluation of time series anomaly detection algorithms. *KDD 2022*, 635–645. doi:10.1145/3534678.3539339; arXiv:2206.13167
- Hwang, W.-S., Yun, J.-H., Kim, J., & Kim, H. C. (2019). Time-series aware precision and recall for anomaly detection. *CIKM 2019*, 2241–2244. doi:10.1145/3357384.3358118
- Garg, A., Zhang, W., Samaran, J., Savitha, R., & Foo, C.-S. (2022). An evaluation of anomaly detection and diagnosis in multivariate time series. *IEEE TNNLS*, 33(6), 2508–2517. doi:10.1109/TNNLS.2021.3105827; arXiv:2109.11428
- Sørbø, S., & Ruocco, M. (2024). Navigating the metric maze: a taxonomy of evaluation metrics for anomaly detection in time series. *Data Min. Knowl. Discov.*, 38(3), 1027–1068. doi:10.1007/s10618-023-00988-8
- Liu, Q., & Paparrizos, J. (2024). The elephant in the room: towards a reliable time-series anomaly detection benchmark. *NeurIPS 2024 Datasets & Benchmarks* (NeurIPS 37, pp. 108231–108261). https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf [P]
- Fawcett, T., & Provost, F. (1999). Activity monitoring: noticing interesting changes in behavior. *KDD '99*, 53–62. doi:10.1145/312129.312195
- Hill, D. J., & Minsker, B. S. (2010). Anomaly detection in streaming environmental sensor data: a data-driven modeling approach. *Environ. Model. Softw.*, 25(9), 1014–1022. doi:10.1016/j.envsoft.2009.08.010
- Leigh, C., Alsibai, O., Hyndman, R. J., Kandanaarachchi, S., et al. (2019). A framework for automated anomaly detection in high frequency water-quality data from in situ sensors. *Sci. Total Environ.*, 664, 885–898. doi:10.1016/j.scitotenv.2019.02.085

**Chance controls, alarm-based prediction, prospective testing**
- Boettiger, C., & Hastings, A. (2012). Early warning signals and the prosecutor's fallacy. *Proc. R. Soc. B*, 279(1748), 4734–4739. doi:10.1098/rspb.2012.2085
- Boettiger, C., & Hastings, A. (2012). Quantifying limits to detection of early warning for critical transitions. *J. R. Soc. Interface*, 9(75), 2527–2539. doi:10.1098/rsif.2012.0125
- Molchan, G. M. (1991). Structure of optimal strategies in earthquake prediction. *Tectonophysics*, 193(4), 267–276. doi:10.1016/0040-1951(91)90336-Q
- Zechar, J. D., & Jordan, T. H. (2008). Testing alarm-based earthquake predictions. *Geophys. J. Int.*, 172(2), 715–724. doi:10.1111/j.1365-246X.2007.03676.x
- Jordan, T. H. (2006). Earthquake predictability, brick by brick. *Seismol. Res. Lett.*, 77(1), 3–6. doi:10.1785/gssrl.77.1.3
- Schorlemmer, D., Gerstenberger, M. C., Wiemer, S., Jackson, D. D., & Rhoades, D. A. (2007). Earthquake likelihood model testing. *SRL*, 78(1), 17–29. doi:10.1785/gssrl.78.1.17
- Schorlemmer, D., Werner, M. J., Marzocchi, W., Jordan, T. H., et al. (2018). The Collaboratory for the Study of Earthquake Predictability: achievements and priorities. *SRL*, 89(4), 1305–1313. doi:10.1785/0220180053
- Donges, J. F., Schleussner, C.-F., Siegmund, J. F., & Donner, R. V. (2016). Event coincidence analysis for quantifying statistical interrelationships between event time series. *Eur. Phys. J. Spec. Top.*, 225(3), 471–487. doi:10.1140/epjst/e2015-50233-y
- Siegmund, J. F., Siegmund, N., & Donner, R. V. (2017). CoinCalc: a new R package for quantifying simultaneities of event series. *Comput. Geosci.*, 98, 64–72. doi:10.1016/j.cageo.2016.10.004
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1–2), 100–115. doi:10.1093/biomet/41.1-2.100
- Lorden, G. (1971). Procedures for reacting to a change in distribution. *Ann. Math. Stat.*, 42(6), 1897–1908. doi:10.1214/aoms/1177693055

**Validation design, leakage, multiplicity**
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *IJF*, 16(4), 437–450. doi:10.1016/S0169-2070(00)00065-0
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Inf. Sci.*, 191, 192–213. doi:10.1016/j.ins.2011.12.028
- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *CSDA*, 120, 70–83. doi:10.1016/j.csda.2017.11.003
- Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: an empirical study on performance estimation methods. *Mach. Learn.*, 109(11), 1997–2028. doi:10.1007/s10994-020-05910-7
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. doi:10.1111/ecog.02881
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. doi:10.1016/j.patter.2023.100804
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), §5.10 time-series cross-validation. https://otexts.com/fpp3/tscv.html [V-doc]
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289–300. doi:10.1111/j.2517-6161.1995.tb02031.x
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Stat.*, 6(2), 65–70. https://www.jstor.org/stable/4615733 [U, not re-checked]
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263. doi:10.1080/07350015.1995.10524599
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126. doi:10.1111/1468-0262.00152
- Hansen, P. R. (2005). A test for superior predictive ability. *JBES*, 23(4), 365–380. doi:10.1198/073500105000000063
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Ann. Stat.*, 17(3), 1217–1241. doi:10.1214/aos/1176347265
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303–1313. doi:10.1080/01621459.1994.10476870
- Wilks, D. S. (2016). "The stippling shows statistically significant grid points": how research results are routinely overstated and overinterpreted, and what to do about it. *BAMS*, 97(12), 2263–2273. doi:10.1175/BAMS-D-15-00267.1
- Gelman, A., & Loken, E. (2014). The statistical crisis in science. *American Scientist*, 102(6), 460. doi:10.1511/2014.111.460
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606. doi:10.1073/pnas.1708274114

**Threshold calibration / conformal**
- Laxhammar, R., & Falkman, G. (2015). Inductive conformal anomaly detection for sequential detection of anomalous sub-trajectories. *Ann. Math. Artif. Intell.*, 74(1–2), 67–94. doi:10.1007/s10472-013-9381-7
- Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023). Testing for outliers with conformal p-values. *Ann. Stat.*, 51(1), 149–178. doi:10.1214/22-AOS2244
- Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 2021*. arXiv:2106.00170
- Angelopoulos, A. N., Candès, E. J., & Tibshirani, R. J. (2023). Conformal PID control for time series prediction. *NeurIPS 2023*. arXiv:2307.16895

**Ecological/WQ forecasting practice, surveillance evaluation**
- Dietze, M. C., Fox, A., Beck-Johnson, L. M., Betancourt, J. L., et al. (2018). Iterative near-term ecological forecasting: needs, opportunities, and challenges. *PNAS*, 115(7), 1424–1432. doi:10.1073/pnas.1710231115
- Lewis, A. S. L., Woelmer, W. M., Wander, H. L., Howard, D. W., et al. (2022). Increased adoption of best practices in ecological forecasting enables comparisons of forecastability. *Ecol. Appl.*, 32(2), e02500. doi:10.1002/eap.2500
- Thomas, R. Q., Boettiger, C., Carey, C. C., Dietze, M. C., et al. (2023). The NEON Ecological Forecasting Challenge. *Front. Ecol. Environ.*, 21(3), 112–113. doi:10.1002/fee.2616
- Dietze, M. C., Thomas, R. Q., Peters, J., Boettiger, C., et al. (2023). A community convention for ecological forecasting: output files and metadata version 1.0. *Ecosphere*, 14(11), e4686. doi:10.1002/ecs2.4686
- Thomas, R. Q., Figueiredo, R. J., Daneshmand, V., Bookout, B. J., et al. (2020). A near-term iterative forecasting system successfully predicts reservoir hydrodynamics and partitions uncertainty in real time. *WRR*, 56(11), e2019WR026138. doi:10.1029/2019WR026138
- Carey, C. C., Woelmer, W. M., Lofton, M. E., Figueiredo, R. J., et al. (2022). Advancing lake and reservoir water quality management with near-term, iterative ecological forecasting. *Inland Waters*, 12(1), 107–120. doi:10.1080/20442041.2020.1816421
- Rousso, B. Z., Bertone, E., Stewart, R., & Hamilton, D. P. (2020). A systematic literature review of forecasting and predictive models for cyanobacteria blooms in freshwater lakes. *Water Res.*, 182, 115959. doi:10.1016/j.watres.2020.115959
- Buehler, J. W., et al. (2004). Framework for evaluating public health surveillance systems for early detection of outbreaks. *MMWR Recomm. Rep.*, 53(RR-5). https://www.cdc.gov/mmwr/preview/mmwrhtml/rr5305a1.htm [V-doc]
- Kleinman, K. P., & Abrams, A. M. (2006). Assessing surveillance using sensitivity, specificity and timeliness. *Stat. Methods Med. Res.*, 15(5), 445–464. doi:10.1177/0962280206071641
- Francy, D. S., Brady, A. M. G., Carvin, R. B., Corsi, S. R., Fuller, L. M., Harrison, J. H., Hayhurst, B. A., Lant, J., Nevers, M. B., Terrio, P. J., & Zimmerman, T. M. (2013). *Developing and implementing the use of predictive models for estimating water quality at Great Lakes beaches.* USGS SIR 2013-5166. doi:10.3133/sir20135166 [V-doc]

**HAB / WQ domain and data papers**
- Stumpf, R. P., Wynne, T. T., Baker, D. B., & Fahnenstiel, G. L. (2012). Interannual variability of cyanobacterial blooms in Lake Erie. *PLOS ONE*, 7(8), e42444. doi:10.1371/journal.pone.0042444
- Stumpf, R. P., Johnson, L. T., Wynne, T. T., & Baker, D. B. (2016). Forecasting annual cyanobacterial bloom biomass to inform management decisions in Lake Erie. *J. Great Lakes Res.*, 42(6), 1174–1183. doi:10.1016/j.jglr.2016.08.006
- Boegehold, A. G., Burtner, A. M., Camilleri, A. C., Carter, G., et al. (2023). Routine monitoring of western Lake Erie to track water quality changes associated with cyanobacterial harmful algal blooms. *ESSD*, 15(8), 3853–3868. doi:10.5194/essd-15-3853-2023 (data: doi:10.25921/11da-3x54 [P])
- Schaeffer, B. A., Bailey, S. W., Conmy, R. N., Galvin, M., et al. (2018). Mobile device application for monitoring cyanobacteria harmful algal blooms using Sentinel-3 satellite Ocean and Land Colour Instruments. *Environ. Model. Softw.*, 109, 93–103. doi:10.1016/j.envsoft.2018.08.015
- Schaeffer, B. A., Reynolds, N., Ferriby, H., Salls, W., et al. (2024). Forecasting freshwater cyanobacterial harmful algal blooms for Sentinel-3 satellite resolved U.S. lakes and reservoirs. *J. Environ. Manage.*, 349, 119518. doi:10.1016/j.jenvman.2023.119518
- Myer, M. H., Urquhart, E., Schaeffer, B. A., & Johnston, J. M. (2020). Spatio-temporal modeling for forecasting high-risk freshwater cyanobacterial harmful algal blooms in Florida. *Front. Environ. Sci.*, 8, 581091. doi:10.3389/fenvs.2020.581091
- Coffer, M. M., Schaeffer, B. A., Darling, J. A., Urquhart, E. A., et al. (2020). Quantifying national and regional cyanobacterial occurrence in US lakes using satellite remote sensing. *Ecol. Indic.*, 111, 105976. doi:10.1016/j.ecolind.2019.105976
- Seegers, B. N., Werdell, P. J., Vandermeulen, R. A., Salls, W., et al. (2021). Satellites for long-term monitoring of inland U.S. lakes: the MERIS time series and application for chlorophyll-a. *Remote Sens. Environ.*, 266, 112685. doi:10.1016/j.rse.2021.112685
- Chorus, I., & Welker, M. (Eds.) (2021). *Toxic Cyanobacteria in Water* (2nd ed.). CRC Press. doi:10.1201/9781003081449
- Corsi, S. R., Graczyk, D. J., Geis, S. W., Booth, N. L., et al. (2010). A fresh look at road salt: aquatic toxicity and water-quality impacts on local, regional, and national scales. *ES&T*, 44(19), 7376–7382. doi:10.1021/es101333u
- Kaushal, S. S., Likens, G. E., Pace, M. L., Utz, R. M., et al. (2018). Freshwater salinization syndrome on a continental scale. *PNAS*, 115(4), E574–E583. doi:10.1073/pnas.1711234115 (page range from memory [U])
- Diaz, R. J., & Rosenberg, R. (2008). Spreading dead zones and consequences for marine ecosystems. *Science*, 321(5891), 926–929. doi:10.1126/science.1156401

**Regulatory / guidance documents (all [V-doc])**
- US EPA (1986). *Quality Criteria for Water 1986* ("Gold Book"), EPA 440/5-86-001, DO Table 1. https://www.epa.gov/sites/default/files/2018-10/documents/quality-criteria-water-1986.pdf
- US EPA (1988). *Ambient Water Quality Criteria for Chloride – 1988*, EPA 440/5-88-001. https://www.epa.gov/sites/default/files/2018-08/documents/chloride-aquatic-life-criteria-1988.pdf
- US EPA (2012). *Recreational Water Quality Criteria*, 820-F-12-058. https://www.epa.gov/sites/default/files/2015-10/documents/rwqc2012.pdf
- US EPA (2013). *Aquatic Life Ambient Water Quality Criteria for Ammonia – Freshwater 2013*, EPA-822-R-13-001. https://www.epa.gov/sites/default/files/2015-08/documents/aquatic-life-ambient-water-quality-criteria-for-ammonia-freshwater-2013.pdf
- US EPA (2015). *Drinking Water Health Advisory for the Cyanobacterial Microcystin Toxins*, EPA-820R15100. https://www.epa.gov/sites/default/files/2017-06/documents/microcystins-report-2015.pdf
- US EPA (2019). *Recommended Human Health Recreational Ambient Water Quality Criteria or Swimming Advisories for Microcystins and Cylindrospermopsin*, EPA 822-R-19-001. https://www.epa.gov/sites/default/files/2019-05/documents/hh-rec-criteria-habs-document-2019.pdf
- US EPA. National Recommended Aquatic Life Criteria table. https://www.epa.gov/wqc/national-recommended-water-quality-criteria-aquatic-life-criteria-table
- US EPA. State & Tribal cyanotoxin thresholds table. https://www.epa.gov/cyanohabs/state-habs-monitoring-programs-and-resources
- WHO (2020). *Cyanobacterial toxins: microcystins*, WHO/HEP/ECH/WSH/2020.6. https://cdn.who.int/media/docs/default-source/wash-documents/wash-chemicals/microcystins-background-201223.pdf
- 40 CFR 141.62 (nitrate MCL) and 141.173(a) (turbidity), via https://www.law.cornell.edu/cfr/text/40/141.62 and /141.173
- 6 NYCRR 703.3 (NY DO standards), via https://www.law.cornell.edu/regulations/new-york/6-NYCRR-703.3
- WAC 173-201A-200(1)(e) (WA turbidity), https://app.leg.wa.gov/WAC/default.aspx?cite=173-201A-200
- Ohio EPA (2020). *Public Water System Harmful Algal Bloom Response Strategy.* https://dam.assets.ohio.gov/image/upload/epa.ohio.gov/Portals/28/documents/habs/2020-PWS-HAB-Strategy.pdf
- US EPA (2013). *Water Quality Event Detection System Challenge: Methodology and Findings*, EPA 817-R-13-002. https://www.epa.gov/sites/default/files/2015-07/documents/water_quality_event_detection_system_challenge_methodology_and_findings.pdf
- US EPA (2016/2019). *Online Water Quality Monitoring in Source Water*, EPA 817-B-16-003. https://www.epa.gov/sites/default/files/2016-09/documents/online_source_water_monitoring_guidance.pdf
- US EPA (2018). *Online Water Quality Monitoring in Distribution Systems*, EPA 817-B-18-001. https://www.epa.gov/sites/default/files/2018-05/documents/owqm-ds_guidance_042018.pdf
- ICPR (2026). *International Warning and Alarm Plan Rhine (IWAP) – messages 2025*, Report No. 320 (DE/FR/NL). https://www.iksr.org/fileadmin/user_upload/DKDM/Dokumente/Fachberichte/DE/rp_De_0320.pdf
- USGS Water Data blog: "Intentional degradations to WaterServices ahead of final decommissioning" (18 Sep 2026) https://waterdata.usgs.gov/blog/wdfn-waterservices-degradation/; "Announcing V1 of the WDFN APIs" (4 Sep 2026) https://waterdata.usgs.gov/blog/api-v1-release/

---

### Appendix: sample files saved this session (all under `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/`)

- CA FHAB: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/hab/cafhab/bloom-report.csv`, `hab-cases.csv`, `hab-results.csv`, data dictionary and disclaimer PDFs (same folder)
- NOAA Lake Erie SI: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/hab/noaa_le_si/le_si.xlsx`
- NRC 2024: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/spills_fishkills_dw/nrc/CY24.xlsx`
- ECHO SSO tables: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/spills_fishkills_dw/echo_sso/x/sewer_overflow_bypass_event.csv` (plus companion tables)
- NY SPRTK: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/sewage_storm_fire_salt_nldi/nysprtk/SPRTK_Data.xlsx`
- Storm Events 2024: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/sewage_storm_fire_salt_nldi/stormevents/StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz`
- MTBS fire points: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/sewage_storm_fire_salt_nldi/mtbs/fod/mtbs_FODpoints_DD.dbf`
- Raccoon R. nitrate (daily, 2012–2025): `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/sewage_storm_fire_salt_nldi/vanmeter_no3_dv_legacy.json`
- Criteria PDFs and text extracts: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/thresholds/`
- EPA EDS Challenge and OWQM guidance, ICPR report 320: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/operational/`
- Crossref verification dumps: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/events/methodology/crossref*.json`
- Helper scripts: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/scripts_events/`
