# 07 — Downstream applications with real, public labels (SENTINEL rebuild, Goal 4)

**Date:** 2026-10-22 · **Scope:** downstream targets that SENTINEL's upstream water-quality (WQ) models can feed: human-health risk, species/ecosystem health, and regulatory or socio-ecological endpoints. Every target must have real labels, be joinable to upstream predictions, and allow honest held-out evaluation.
**Inputs assumed from upstream:** predicted DO, temperature, pH, specific conductance (SC), turbidity, nutrients, chl-a, cyanobacteria/microcystin and *E. coli*, plus embeddings, at sites (USGS/WQP stations, NEON, lake pixels) over time.

**How this was verified.** Data access was checked live on 2026-10-22 with `curl` against the actual APIs and downloads. Row counts, field names and class balances below were *measured here* unless a source is given. Papers were checked against Crossref, PubMed or OpenAlex metadata (title, authors, venue, DOI), and metrics are quoted from abstracts or tables. The session's shared web-search budget ran out twice during the work. Where a fact could not be confirmed it is marked **(unverified)**. Small samples (122 MB) were saved in the scratchpad; see Appendix A.

Related notes: `04_ecotox_bioassessment.md` (bioassessment SOTA and the blocked-RF baselines; reused here, not duplicated), `05_multimodal_fusion.md` (cyanoHAB forecasting SOTA) and `10_local_inventory.md` (local FIB and microcystin tables).

---

## 0. Executive summary

**The decisions that matter**

1. **Lead with eight downstream applications** (ranked in §d), each benchmarked against the *incumbent operational method*, not a strawman:
   1. **SafeSwim:** beach FIB exceedance nowcast/forecast.
   2. **ToxinWatch:** cyanotoxin exceedance and HAB-advisory forecast.
   3. **KillAlert:** fish-kill early warning.
   4. **TapRisk:** drinking-water nitrate/DBP violation risk.
   5. **BioIntegrity:** NRSA/NLA biological condition.
   6. **MusselGuard:** dreissenid invasion forecast.
   7. **TroutRefuge:** cold-water fish occupancy under predicted thermal regimes.
   8. **RiverPulse:** GPP/ER at reaches without sensors.

   Three stretch targets follow: lake bottom-water hypoxia (NLA profiles), ATTAINS impaired-water screening, and avian botulism (WHISPers).

2. **Every application must beat the operational baseline, not only persistence.** Concrete incumbents exist:
   - **Beaches.** USGS beach models were validated against a persistence model; only 24 of 42 beat it by ≥5 points. The City of Chicago publishes its operational predictions. In my quick check (§b) they reach **AUROC 0.70** on 9,436 beach-days (2017–2025). The same-day qPCR mean at the 6 "feature" beaches alone reaches **0.74**.
   - **CyanoHABs.** EPA's INLA forecast scored 90% accuracy, 88% sensitivity, 91% specificity and 49% precision on a 2021 holdout.
   - **Bioassessment.** Hill et al. 2017 (RF + StreamCat) predicted 75% of condition classes correctly.
   - **Drinking water.** Pennino et al. 2020's random-forest nitrate-violation model correctly classified 90% of catchments for surface-water systems (79% of violating catchments for groundwater).
   - **Invasives.** A co-kriging model (Kanankege et al. 2018) reached AUC 0.78 for zebra mussel risk.

   The previous FIB model only tied persistence (0.62). The new design has to show skill above these incumbents.

3. **Primary labels must be measurements or field observations, never thresholds of SENTINEL's own inputs.** Examples are lab FIB counts, lab toxin results, fish-kill investigations, MMI condition classes, infestation confirmations and GPP/ER. Administrative labels (advisories, 303(d) listings, violations) are secondary targets. Each needs a "policy-circularity" check, because NY State Parks closures include "Preemptive: rainfall" and "Predicted exceedance (model)" reasons, and BEACON actions include rain advisories.

4. **Cross-fit the upstream predictions.** For any downstream test fold, the upstream WQ model must not have seen that fold's sites or later times. This applies most to *E. coli* and microcystin, which are both upstream outputs and downstream labels. Otherwise the downstream score inherits upstream leakage. The protocol is in §d.0.

5. **Data plumbing changed in 2024–2026, and the code must reflect it:**
   - The legacy WQP (WQX 2.2) has **no USGS data after 2024-03-11**; use WQX 3.0.
   - The ATTAINS REST API now **requires an api.data.gov key**. Its ArcGIS geospatial layers stay open and include NHDPlus HR catchment links.
   - NLDI moved to `api.water.usgs.gov/nldi`; the old `labs.waterdata.usgs.gov` URL returns 404.
   - EJScreen and CEJST are off federal servers (mirrors only). CDC SVI and EJI pages are live.
   - EPA NRSA 2023–24 data are **not yet posted**.

6. **Joins are feasible for every primary target:**
   - Beaches: BEACON beach ID = WQP `ProjectIdentifier` (verified for Ohio, org 21OHBCH: 53,069 *E. coli* results at 96 sites).
   - NARS and the Powell Center metabolism data ship COMIDs.
   - Iowa fish kills ship lat/lon; NAS records ship lat/lon and HUC12.
   - Drinking water joins through the EPA service-area boundaries v3 (>44,000 CWS).
   - WHISPers is county-level.

---

## (a) Candidate applications and label sources

**Legend:**
- **Feas.** = feasibility 1–5.
- **Imp.** = societal impact 1–5.
- **Verified** = access tested here.
- License "US Gov" = federal data, public domain unless the dataset says otherwise.

### a.1 Human health

| ID | Application | Label source (verified URL / API) | Label definition | Size (measured unless noted) | Coverage · resolution | License | Join keys | Feas./Imp. |
|---|---|---|---|---|---|---|---|---|
| H1 | **Beach FIB exceedance** (nowcast/forecast) | **WQP BEACH projects**: `https://www.waterqualitydata.us/data/Result/search?organization=21OHBCH&characteristicName=Escherichia%20coli`. **BEACON 2.0** monitoring report, e.g. `https://beacon.epa.gov/ords/beacon2/r/beacon_apex/beacon2/monitoring-wqx?state=VA&year=2023` (APEX UI; CSV/XLS export; 100k-row cap). | Single sample > BAV: *E. coli* 235 (or 190) cfu/100 mL; enterococci 70 (or 60) cfu; *Enterococcus* qPCR 1,000 (or 640) CCE (EPA 2012 RWQC Table 5) | Ohio Dept of Health alone: 53,069 *E. coli* results at 96 sites. National WQP *E. coli*: 2,965,368 results / 117,221 sites (not all beaches). BEACON 2020: 6,349 coastal and Great Lakes beaches; 5,055 program beaches; 3,427 monitored | US coastal + Great Lakes (+ inland via WQP); daily to weekly in season; 2000–present | US Gov | WQP `MonitoringLocationIdentifier`; BEACON beach ID = WQP `ProjectIdentifier` (guide: "Query WQP using the Beach ID as the Project ID"); beach lat/lon | 5 / 5 |
| H1b | Beach FIB (city/state portals with richer context) | **Chicago** (Socrata, no key): Beach Lab Data `2ivx-z93u`; official model predictions `xvsz-3xcj`; advisories `t62e-8nvc`; beach sensors `qmqz-2xku`; weather `k7hf-8y75`. **NY State Parks** `data.ny.gov` `wwwd-za77`. **King County WA** `data.kingcounty.gov` `mbzm-4r9y`. **Ramsey Co. MN** `2yt3-vdb6`. **NYC** `2xir-kwzz` | As H1. Chicago qPCR advisory at >1,000 CCE | Chicago: 31 beaches; culture *E. coli* 2006–2016 (17,963 samples, 13.9% >235); qPCR 2015–2026 (17,594, 7.0% >1,000 CCE); 11,385 official predictions (2017–2025). NYS Parks: 17,687 rows, 2015–2025, 74–85 beaches, 45 waterbodies, 10.1% of *E. coli* >235. King Co.: 4,786 rows, 30 lake beaches, 2019–2026 | Daily (Chicago, King Co.) / weekly | Portal terms (open) | Beach name + lat/lon | 5 / 5 |
| H1c | Bathing-water class (Europe, transfer test) | **EEA** "Bathing Water Directive – Status of bathing water" 2025 v1.0: `https://www.eea.europa.eu/en/datahub/datahubitem-view/c3858959-90da-4c1b-b9ca-492db0e514df` (xlsx 43 MB) | Seasonal class Excellent / Good / Sufficient / Poor (4-season percentile rule on *E. coli* + intestinal enterococci) | 694,918 site-seasons; 27,759 sites; 31 countries; 1990–2025; 9,905 inland sites. 2025 inland: 5,818 Excellent, 840 Good, 212 Sufficient, 198 Poor | Europe; annual | **CC BY 4.0** | lat/lon | 4 / 4 |
| H2 | Beach notification actions (advisories/closures) | BEACON "Beach Actions (Advisories and Closures)" report (type, start/end, duration, indicator); NYS Parks `beach_status`/`status_reason`; Chicago `t62e-8nvc` | Action on day *d* (contamination advisory or closure); **exclude rain/preemptive/model-based** | 2020: 7,562 actions; 60,115 of 726,175 beach-days (8%) under action; 20% of actions lasted 1 day (EPA 2020 report). NYS Parks: 790 "Closed: Exceedance", 225 "Preemptive: rainfall", 79 "Predicted exceedance (model)" | As H1 | US Gov | Beach ID | 4 / 5 |
| H3 | **Cyanotoxin exceedance** (lab) | **WQP** (`characteristicName=Microcystin`, `Microcystin LR`, `Cylindrospermopsin`, `Anatoxin-A`, `Saxitoxin`, …). **WA Toxic Algae** `https://www.nwtoxicalgae.org/` (per-lake HTML tables). **EPA NLA** 2007/12/17/22 and NRSA toxin files | Microcystins > 8 µg/L, cylindrospermopsin > 15 µg/L (EPA 2019 swimming advisory); state values for anatoxin-a (WA DOH guidance; values **unverified**) | WQP: microcystin 73,928 results / 6,365 sites; MC-LR 14,729 / 4,525; cylindrospermopsin 31,092 / 6,691; anatoxin-A 15,584 / 5,193; saxitoxin 11,823 / 3,098. WA 2007–2026: microcystin 16,575 samples (1,595 above guideline); anatoxin-a 12,702 (418). NLA 2022: 1,225 microcystin + 1,225 cylindrospermopsin results; 981 lakes; 2% of lakes exceeded the EPA criterion (per 05 note). NRSA 2013–14 microcystin: only 1 of 1,853 visits >8 µg/L | National, biased toward blooming lakes (WQP/state) vs probability survey (NLA) | US Gov; WA terms not stated | WQP site; NLA `SITE_ID` → COMID; WA lake name + county | 5 / 5 |
| H4 | **CyanoHAB advisories/closures** | **California FHAB** (data.ca.gov "Surface Water – Freshwater Harmful Algal Blooms"; CKAN datastore API). **NY DEC** `95my-wijm` (2019+) and `qtg2-hjth` (2012–18). **Oregon OHA** archive `https://www.oregon.gov/oha/PH/HEALTHYENVIRONMENTS/RECREATION/HARMFULALGAEBLOOMS/Pages/archive.aspx`. NYS Parks HAB closures | Tiered advisory (CA: Caution / Warning / Danger; OR: Advisory / Precaution; NY: suspicious / confirmed / high toxin) on a waterbody-date or waterbody-year | CA: 14,437 advisory records 2016–2026 (Caution 4,403 · Warning 478 · Danger 1,215 · None 6,242); 3,728 bloom reports with lat/lon. NY: 880 waterbody-years 2012–18 (confirmed 414, high-toxin 236, suspicious 230); 1,371 confirmed waterbody-years 2019–25. OR: ~245 advisories 2007–2025. NYS Parks: 257 "Harmful Algal Bloom" + 165 "Blue-green algae" closures | CA, NY, OR, WA; event/weekly | CA Public Domain; others open portals | lat/lon / waterbody | 4 / 5 |
| H5 | HAB illness (human / animal) | CDC OHHABS `https://www.cdc.gov/ohhabs/data/index.html` (annual summaries only). Backer et al. 2013 (dog poisonings review). WHISPers "Toxicosis (cyanobacteria)" | Illness case counts | OHHABS 2022: 15 states, 372 HAB events, 95 human and 102,071+ animal illnesses; 2021: 16 states, 368 events, 117 human, ≥2,715 animal. **No public event-level data.** WHISPers: 27 cyanobacteria-toxicosis events | State-year | US Gov | state / county | 1 / 5 (validation only) |
| H6 | **Drinking-water violations** | **SDWIS** via Envirofacts `https://data.epa.gov/dmapservice/sdwis.violation/primacy_agency_code/equals/OH/1:2/json` (works; the old `efservice` syntax fails). **ECHO bulk** `https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip` (558 MB, quarterly; last modified 2026-10-14) | Health-based violation (`is_health_based_ind=Y`) by PWS-quarter for nitrate MCL, Stage 1/2 DBP (TTHM/HAA5), SWTR turbidity TT, RTCR *E. coli* | ~10% of CWS have a health-based violation (Scanlon 2022); 17,900 CWS panel 1982–2015 (Allaire 2018) | National; quarterly | US Gov | PWSID → service-area polygon (EPA SAB v3); `primary_source_code` (SW/GW) | 4 / 5 |
| H7 | Drinking-water **concentrations** (DBP, precursors, cyanotoxins, nitrate) | **UCMR 4** `https://www.epa.gov/system/files/other-files/2024-04/ucmr4-occurrence-data.zip` (10.6 MB); UCMR 5 zip (13.5 MB, updated 2026-10-15). **SYR4** compliance data `https://www.epa.gov/dwsixyearreview/six-year-review-4-compliance-monitoring-data-2012-2019` (zips incl. "DBP_Related Parameters" and nitrate groups) | HAA5 > 60 µg/L; raw-water TOC / bromide (DBP precursors); cyanotoxin detection | UCMR4: 932,854 results / 5,833 PWS. HAA5: 63,484 results / 4,924 PWS (2.1% single results >60). Raw-water TOC 24,721 + bromide 24,733 / 3,444 PWS. Cyanotoxins: 105,865 results / 3,487 PWS (2018-03 to 2021-09), only 157 detections / 67 PWS | National; sample events | US Gov | PWSID, facility, sample point type (SR = source) | 3 / 4 |
| H8 | Waterborne outbreaks | **CDC NORS** (BEAM) `https://data.cdc.gov/resource/5xkq-dg7x.json` (the old NORS dashboard URL now 301-redirects) | Outbreak (year, month, state, etiology, setting, illnesses, deaths, water exposure/type) | 66,713 outbreaks 1971–2023; 3,106 waterborne; 282 untreated lake/reservoir; 680 community drinking water | **State-month only** | US Gov | state | 1 / 5 (validation / case studies) |
| H9 | *Vibrio* (estuarine; optional) | Archer et al. 2023 30-yr case DB (ecological niche model); NOAA Chesapeake model (Jacobs et al. 2014); CDC COVIS (public granularity **unverified**) | Cases / presence | Eastern-US wound infections rose from 10 to 80 per year (1988–2018) (Archer 2023) | Coastal | — | state/county | 2 / 4 |
| H10 | *Naegleria fowleri* PAM (optional) | Gharpure et al. 2021 (EID); NORS etiologies | Case reports | Tiny n (2 deaths in the 140 untreated-recreational-water outbreaks of 2000–2014; Graciaa 2018) | State | US Gov | state | 1 / 3 |

### a.2 Species and ecosystem health

| ID | Application | Label source (verified) | Label definition | Size | Coverage · resolution | License | Join keys | Feas./Imp. |
|---|---|---|---|---|---|---|---|---|
| E1 | **Fish kills** | **Iowa DNR Fish Kill DB**: JSON API `https://programs.iowadnr.gov/fishkill/api/v1/events?f=strict`; CSV `/fishkill/Events/Download/Events`. **Delaware** `data.delaware.gov` `ycvg-4ys9` (tidal), `8pfi-yck7` (non-tidal). **WI/MN** via Till 2019 and Tye 2022 (access **unverified**; likely by request). **Florida FWC hotline** (since 1995, >40,000 reports; no public bulk download found) | Investigated kill event with date, location, cause class (natural/environmental vs anthropogenic), magnitude | Iowa: 1,035 events 1981–2026 (602 since 2000); 999 geocoded; River/Stream 755, Lake 143, Pond 114; origin Anthropogenic 484 / Natural 296 / Unknown 255; causes Animal Waste 244, Environmental 243, Ammonia/Fertilizer 69, … Delaware: 234 + 290. WI+MN: 526 events in lakes 2003–2013 (press coverage of Tye 2022; **unverified** in paper) | IA, DE (+WI, MN, FL on request); event-day | IA: not stated; DE open | lat/lon → COMID/HUC12; Iowa `impactedSegs` → state assessment segments | 4 / 5 |
| E2 | **Wildlife die-offs** (avian botulism, algal toxicosis) | **USGS WHISPers** API `https://whispers.usgs.gov/api/eventsummaries/?format=json&diagnosis=33` (Django REST, public) | Mortality event with diagnosis | 12,474 public events; Botulism Type C 1,300; Botulism (unspecified) 329; Type E 95; cyanobacteria toxicosis 27 | US; event start/end; county | US Gov | county (admin level 2), state | 3 / 4 |
| E3 | Mussel die-offs | No public event database found. Case literature: Richard et al. 2020 (Clinch River densovirus) | — | — | — | — | — | 1 / 3 |
| E4 | **Biological condition** (macroinvertebrate / fish MMI) | **EPA NARS** `https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys`: NRSA 2008–09, 2013–14, 2018–19 (MMI files posted 2025-03); NLA 2007/12/17/22; **NRSA 2023–24 not yet posted** | `BENT_MMI_COND`, `FISH_MMI_COND` ∈ {Good, Fair, Poor} | NRSA 2018–19 benthic: 2,110 visits / 1,919 sites (Poor 914, Good 673, Fair 519). Fish: 1,777 (Poor 758, Good 561, Fair 412). NRSA 2013–14: 1,853 visits (benthic Poor 805, Good 597, Fair 446). NLA 2022 site info: 3,880 rows | CONUS probability survey; ~5-yr cycles | US Gov | **COMID, HUC8, lat/lon in site files**; design weights | 5 / 4 |
| E5 | **Cold-water fish occupancy / thermal habitat** | NRSA fish counts (true absences at fish-sampled sites); state surveys (**EBTJV** access **unverified**); EcoSHEDS NE brook-trout occupancy *model* (`https://usgs.gov/apps/ecosheds/` — a model output, **not a label**); NorWeST temperatures (doi:10.2737/RDS-2016-0033) | Presence/absence of salmonids at surveyed reaches | DeWeber & Wagner 2015: 7,327 training / 1,832 validation samples. GBIF brook trout (US): 56,236 presence-only records, 0 absences | Eastern + western US | US Gov / mixed | COMID | 3 / 3 |
| E6 | Amphibian disease (Bd/Bsal) | Amphibian Disease Portal `https://amphibiandisease.org/` (GitHub BNHM; counts/API **unverified**) | Infection status | **unverified** | Global | — | lat/lon | 2 / 2 |
| E7 | **Invasive dreissenid mussels** | **USGS NAS** API `https://nas.er.usgs.gov/api/v2/occurrence/search?species_ID=5` (zebra = 5, quagga = 95); **MN DNR infested waters** `https://files.dnr.state.mn.us/eco/invasives/infested-waters.xlsx` (updated 2026-10-21) | Established occurrence; MN listing (confirmed vs connected) with year | NAS zebra 11,034 records 1986–2026 (7,521 established; 1,788 HUC12; 39 states); quagga 1,523 (1,216 established; 204 HUC12; 23 states). MN list 1,526 rows; 398 confirmed ZM lakes/wetlands + 302 "connected"; MN has 11,842 lakes | US; year | US Gov / MN DNR | lat/lon, **HUC8/10/12 in NAS**; MN DOW number | 4 / 4 |
| E8 | Biodiversity occurrence | GBIF API `https://api.gbif.org/v1/occurrence/search` (search keyless; bulk download needs login); RivFishTIME (Comte et al. 2021) | Presence (and abundance for time series) | US zebra mussel 8,515; brook trout 56,236; `occurrenceStatus=ABSENT` → 0 | Global | per-dataset CC licenses | lat/lon | 2 / 3 |
| E9 | **Stream metabolism (GPP/ER)** | Powell Center: Appling et al. 2018; ScienceBase doi:10.5066/F70864KX (child items 1–8; `daily_predictions.zip` 101 MB) | Daily GPP, ER, K600 estimates (model-derived from DO) | 356 rivers, Oct 2007–Jan 2017, 490,907 site-days; site table 602 rows with `nwis_id`, `nhdplus_id` | CONUS USGS gauges; daily | US Gov | **NWIS ID + COMID given** | 4 / 3 |
| E10 | Hypoxia ("dead zones") | Gulf shelf hypoxic area (LUMCON/LSU; `https://gulfhypoxia.net/research/shelfwide-cruises/`; 1985–present, missing 2016; values in images only). **NLA DO profiles** (e.g. `nla2022_profile_wide.csv`, 10,617 depth rows); Jane et al. 2021 lake data (EDI; EDI API needs a free account since 2026-07 per 01 note) | Annual hypoxic area; **lake bottom DO < 2 mg/L** at an NLA visit | Gulf: ~38 annual values (tiny n). NLA: ~1,000 lakes per cycle | Gulf / CONUS lakes | US Gov | COMID (NLA) | 3 / 4 |

### a.3 Regulatory and socio-ecological endpoints

| ID | Application | Label source (verified) | Label definition | Size | Notes |
|---|---|---|---|---|---|
| O1 | **Impaired waters (ATTAINS)** | **Geospatial (no key):** `https://gispub.epa.gov/arcgis/rest/services/OW/ATTAINS_Assessment/MapServer` (layers 0 points, 1 lines, 2 areas, 3 catchment associations). **REST API** `attains.epa.gov/attains-public/api/...` now returns 401: "send API requests through api.data.gov using a valid API Key" | Per-AU use support (`ecological_use`, `recreation_use`, `drinkingwater_use`, `fishconsumption_use`: Fully / Not Supporting / Insufficient / Not Assessed); IR category; cause groups (`pathogens`, `nutrients`, `oxygen_depletion`, `temperature`, `algal_growth`, `biotoxins`, `cause_unknown_fish_kills`, …) | 534,145 line features, 92,653 areas, 5,935 points; **6,816,110 catchment associations (NHDPlus HR `nhdplusid` + `huc12`)**. Ecological use over line features: Fully 196,839 / Not 84,323 / Insufficient 133,454 / Not Assessed 97,291 | State-by-state methods; cycle lag; circular with monitoring data. Feas. 4 / Imp. 4 |
| O2 | Nutrient-criteria exceedance | EPA 2021 lake TN/TP criteria (EPA-822-R-21-005; stressor–response to chl-a, microcystin, hypoxia); state numeric criteria (state-progress table **unverified**) | Observed TN/TP/chl-a > criterion | WQP | **A translation layer, not an independent endpoint** |
| O3 | Salinization | EPA chloride criteria (acute 860 / chronic 230 mg/L, 1988, verified in EPA table); EPA 2011 conductivity benchmark 300 µS/cm (Central Appalachia); NRSA `SAL_COND`; Dugan et al. 2017 (371 lakes) | Observed Cl > 230 mg/L; benthic response to SC | WQP + NRSA | Use as a BioIntegrity sub-analysis and a regulatory translation of upstream SC |
| O4 | Wildfire-impacted WQ | MTBS `https://www.mtbs.gov/direct-download` (200) + WQP before/after; CA post-fire monitoring (data.ca.gov, 2017–18); Paul et al. 2022; Raoelison et al. 2023 | Post-fire exceedance (turbidity, nutrients) | — | Best used as an **out-of-distribution stress test** of upstream models, not a separate label |
| O5 | EJ overlays (stratification, not labels) | **CDC SVI** (2022 listed) `https://www.atsdr.cdc.gov/place-health/php/svi/index.html` (200); **CDC EJI** `https://www.atsdr.cdc.gov/place-health/php/eji/index.html` (200; page dated 2024-12-02). **EJScreen**: `epa.gov/ejscreen` → 404; mirrors at PEDP `https://pedp-ejscreen.azurewebsites.net/` (200) and UChicago data rescue `https://knowledge.uchicago.edu/records/z7nbk-5bk41` (2024 data, gdb + CSV). **CEJST**: official site unreachable; PEDP mirror `https://public-environmental-data-partners.github.io/j40-cejst-2/` (200) | Tract / block-group indices | — | Use for fairness-stratified metrics and impact framing; cite the mirror provenance |

---

## (b) SOTA and baselines per application

### b.1 Published comparators

Metrics are as reported. "Design" is the held-out design stated by the source, where known.

| App | Paper | Data | Method | Held-out design | Reported metric(s) | Baseline beaten |
|---|---|---|---|---|---|---|
| H1 | **Francy et al. 2013**, USGS SIR 2013-5166, doi:10.3133/sir20135166 | 42 Great Lakes beaches, 2010–12 | MLR / PLS (turbidity most common predictor) | Fit on prior years, **validated on 2012** | Goals: ≥80% correct, ≥50% sensitivity, ≥85% specificity, ≥5 pts over persistence. **24/42 models beat persistence by ≥5 pts.** Goals met (model vs persistence): correctness 28 vs 17, sensitivity 17 vs 4, specificity 34 vs 25 | Persistence (previous-day *E. coli*) |
| H1 | **Brooks et al. 2016**, *Environ. Model. Softw.* 76:81–94, doi:10.1016/j.envsoft.2015.10.012 | 7 Wisconsin beaches | 14 regression methods | (abstract) | **Random forest most accurate**, then adaptive-lasso MLR | Other methods |
| H1 | **Lucius et al. 2019**, *Water Res. X* 2:100016, doi:10.1016/j.wroa.2018.100016 | Chicago 2006–2016 + 2017 live pilot (15 beaches) | Hybrid: qPCR at 5 feature beaches + covariates | Leave-one-year-out, 2016 holdout, live 2017 | 2017 hybrid: sensitivity 0.112, specificity 0.980, **AUC 0.753** | Prior-day RF model: 2016 sensitivity 0.037 / AUC 0.644; 2015 sensitivity 0.032 / AUC 0.655 |
| H1 | **Thoe et al. 2014**, *Water Res.* 67:105–117, doi:10.1016/j.watres.2014.09.001 | Santa Monica Beach | MLR, logistic, PLS, ANN, classification tree | Validation period | CT correctly predicted **42%** of fecal-coliform postings vs **28%** for the current method | Current method (day-old FIB) |
| H1 | **Searcy & Boehm 2021**, *ES&T* 55(3):1908–1918, doi:10.1021/acs.est.0c06742 | 3 CA marine beaches; 24–48 h high-frequency sampling | 126 models; RF best | Withheld data | RF consistently top | — |
| H1 | **Searcy & Boehm 2023**, *ES&T* 57(46):17930–17939, doi:10.1021/acs.est.2c05972 | 2 CA sites | ~400 forecast models, up to 3-day lead | Tested against persistence and a nowcast | Sensitivity/specificity **similar to persistence** but significantly higher AUC | Persistence, nowcast |
| H1 | **Zhang et al. 2018**, *ES&T* 52(15):8446–8455, doi:10.1021/acs.est.8b01022 | 4 Lake Michigan beaches | NARX / WA-NAR neural nets | 3 months of test data | R² ≈ 0.8 (beaches) | Prior models at the same sites |
| H3/H4 | **Schaeffer et al. 2024**, *J. Environ. Manage.* 349:119518, doi:10.1016/j.jenvman.2023.119518 | 2,192 OLCI-resolved US lakes | INLA spatiotemporal vs SVC, RF, DNN, LSTM, RNN | Train 2017–2020 → **test 2021** | Accuracy 90%, sensitivity 88%, specificity 91%, **precision 49%** | All ML/NN comparators. **Label = satellite WHO Alert Level 1 (>12 µg/L chl-a with cyano dominance), not toxin** |
| H3/H4 | **Myer et al. 2020**, *Front. Environ. Sci.* 8:581091, doi:10.3389/fenvs.2020.581091 | 103 Florida lakes, OLCI, 2016–2019 | INLA (SPDE + AR1) | 80/20 random split + 1-week-ahead holdout | Validation AUC 0.95 (sensitivity/specificity 0.88); forecast week **AUC 0.89** (sensitivity/specificity 0.82) | — (target: >100,000 cells/mL) |
| H3 | Hollister & Kreakie 2016, *F1000Research* 5:151, doi:10.12688/f1000research.7955.2 | NLA 2007 | chl-a vs microcystin advisory association | — | (metrics not extracted) | — |
| H3 | Prior SENTINEL GBM (audit) | WQP microcystin + S2 + drivers | GBM fusion | 5-km / 20-km blocking; leave-one-HUC-out | 0.755 (5-km), 0.602 (20-km), ~0.5 cross-region | Single modality ≈0.65 |
| H6 | **Pennino et al. 2020**, *Sci. Total Environ.* 722:137661, doi:10.1016/j.scitotenv.2020.137661 | SDWIS nitrate MCL violations | RF on land cover, N inputs, soils, climate | (design not in abstract) | SW: **90%** of catchments correctly classified, R² 0.52; GW: 79% of violating catchments, R² 0.43 | — |
| H6 | Scanlon et al. 2022, *ES&T* 56(18):12965–12974, doi:10.1021/acs.est.1c08697 | SDWIS | RF driver analysis | — | ~10% of CWS have health-based violations; small rural systems dominate | — |
| H6 | Allaire et al. 2018, *PNAS* 115(9):2078–2083, doi:10.1073/pnas.1719805115 | 17,900 CWS, 1982–2015 | Probit | — | ~21 M people on violating CWS in 2015 | — |
| E1 | **Tye et al. 2022**, *Limnol. Oceanogr. Lett.* 7(6):510–519, doi:10.1002/lol2.10274 | Fish mortality events + 1.2 M temperature profiles, 8,891 lakes | Temperature-based models | (abstract) | "Accurately predicted contemporary summerkills" (**no number in abstract**); projected ~6- to 34-fold increases by 2100 | — |
| E1 | Till et al. 2019, *Nat. Clim. Change* 9(8):637–641, doi:10.1038/s41558-019-0520-y | Wisconsin kills + modeled lake temperatures | — | — | **Metrics unverified** (abstract not retrievable) | — |
| E1 | — | — | — | — | **No AUC-reported fish-kill ML benchmark found** (searched) → open niche | — |
| E4 | **Hill et al. 2017**, *Ecol. Appl.* 27(8):2397–2415, doi:10.1002/eap.1617 | NRSA 2008–09 benthic MMI | RF + StreamCat | Random (OOB + 71-site holdout; per 04) | Abstract: 75% of sites correctly classified. Per 04: OOB AUC 0.84, holdout AUC 0.81 | — (**predictions published in StreamCat — never use as a feature**) |
| E4 | Blocked RF baselines (computed in 04) | NRSA 2018–19 | RF | Leave-one-ecoregion-out | Community → chemistry R² 0.29–0.39 | Geography-only (TP 0.06) |
| E5 | **DeWeber & Wagner 2015**, *TAFS* 144(1):11–24, doi:10.1080/00028487.2014.963256 | Eastern brook trout reaches | Hierarchical Bayesian logistic | 7,327 train / 1,832 validation | **AUC ≈ 0.78, κ = 0.44** | — |
| E7 | **Kanankege et al. 2018**, *Front. Vet. Sci.* 4:231, doi:10.3389/fvets.2017.00231 | MN waterbodies (125/18,411 infested) | Probability co-kriging + boater network | (abstract) | Zebra mussel **AUC 0.78** (distance to infested + boater traffic + road access) | — |
| E7 | Whittier et al. 2008, *Front. Ecol. Environ.* 6(4):180–184, doi:10.1890/070073 | >3,000 stream sites | Calcium-risk classes | — | Very low <12, low 12–20, moderate 20–28, high >28 mg/L Ca | Rule baseline |
| E9 | Segatto et al. 2021, *Ecosystems* 24(7):1792–1809, doi:10.1007/s10021-021-00618-8 | 12 sensor reaches → 292 reaches | RF | (abstract) | Network-scale regimes (no single metric in abstract) | — |
| E10 | Scavia et al. 2017, *PNAS* 114(33):8823–8828, doi:10.1073/pnas.1705293114 | Gulf hypoxia | Multi-model ensemble | — | Policy-scenario probabilities | — |

### b.2 New baseline computed here (Chicago operational model)

**Data and setup.** I joined the City of Chicago's official "Beach E. coli Predictions" (`xvsz-3xcj`, prediction source "DNA Model") to same-day qPCR results at the same beach (`2ivx-z93u`). That gives n = 9,436 beach-days at the predicted beaches (15 in the prediction file; 14 have matching lab data), 2017–2025, with 491 exceedances (5.2%) above 1,000 CCE. The persistence baseline required the previous sample within 3 days.

| Predictor | AUROC | AUPRC |
|---|---|---|
| Official Chicago model (`predicted_level`) | **0.702** | 0.128 |
| Persistence (previous qPCR at the same beach) | 0.645 | 0.095 |
| Same-day geometric mean of qPCR at the 6 "feature" beaches (63rd St, Calumet, Humboldt, Montrose, Rainbow, South Shore) | **0.740** | 0.166 |

Per-year AUROC of the official model ranges from 0.56 (2023) to 0.75 (2017).

**Caveats.** This is an exploratory, unofficial check. I assumed `predicted_level` is issued before the same-day result, and the 2020 season is nearly empty. Feature-beach qPCR is legitimately available at decision time in Chicago's design (about 4–5 h turnaround). Culture results (18–24 h) are not available at decision time.

**Lesson for SafeSwim.** The baseline to beat in Chicago is ~0.74 (network same-day qPCR), not 0.64. Script: `scripts_downstream/chicago_eval2.py` (Appendix A).

### b.3 Strongest baseline to report for each recommended application

| App | Must-report baselines (all under the identical split) | Strongest expected |
|---|---|---|
| SafeSwim | Persistence; rainfall-threshold rule; site × day-of-year climatology; Virtual-Beach-style MLR/PLS (USGS archive doi:10.5066/P9Y9O1YJ); RF (Brooks 2016); Chicago official predictions; feature-beach same-day GM | Feature-beach GM (Chicago); RF on turbidity + rain + wave (Great Lakes) |
| ToxinWatch | Last toxin result at the lake (persistence); chl-a or CyAN-CI threshold; INLA spatiotemporal (Schaeffer 2024 style); prior SENTINEL GBM | INLA / GBM with lake random effects |
| KillAlert | Month × region climatology; air/water-temperature-anomaly logistic (Till/Tye style); observed DO-threshold rule where sensors exist | Temperature-anomaly logistic |
| TapRisk | System size + source type + state (Allaire/Scanlon drivers); Pennino-style RF on StreamCat; previous-violation persistence | Persistence + RF |
| BioIntegrity | Hill 2017 RF on StreamCat (re-run blocked); geography-only; ecoregion prevalence | RF on StreamCat |
| MusselGuard | Distance-to-nearest-infested; Ca-risk rule; co-kriging (Kanankege 2018) | Distance + connectivity |
| TroutRefuge | Range-only prior; NorWeST/EcoSHEDS temperature + logistic (DeWeber & Wagner) | Hierarchical logistic with modeled temperature |
| RiverPulse | Site climatology; light (PAR) + season; RF (Segatto 2021 style) | RF on light + flow |

---

## (c) Join strategy to upstream models

### c.1 Keys and tested services

| Step | Service (tested 2026-10-22) | Example that worked | Field(s) to keep | Gotcha |
|---|---|---|---|---|
| Station → COMID | **NLDI** `https://api.water.usgs.gov/nldi/linked-data/{wqp\|nwissite}/{id}?f=json` | `nwissite/USGS-05427850` → comid **13294288** (reachcode 07090002008236, measure 4.34); `wqp/USGS-05427850` → comid **13293576** | `comid`, `reachcode`, `measure`, `mainstem` | **The same site gets different COMIDs by source** (point-in-catchment vs hydrolocation). For river gauges use `nwissite` (reach-indexed) and record which method was used. Old `labs.waterdata.usgs.gov/api/nldi` → 404 |
| Lat/lon → COMID | NLDI `.../linked-data/comid/position?f=json&coords=POINT(lon lat)` | `POINT(-89.35 43.0864)` → 13294318 | `comid` | Snaps to the nearest flowline catchment; lakes need waterbody COMID or LAGOS IDs |
| Upstream/downstream networks | NLDI `.../comid/{id}/navigation/UM/wqp?distance=10`; `.../comid/{id}/basin` | 12 upstream-mainstem WQP sites within 10 km; basin polygon | site lists, basin | Put flow-connected sites in the **same CV fold** |
| Lat/lon → HUC12 | WBD `https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?geometry=lon,lat&geometryType=esriGeometryPoint&inSR=4326&outFields=huc12,name&f=json` | → `070900020504` (Cherokee Lake–Yahara River) | `huc12` | Store HUCs as **zero-padded strings** (the old float-HUC bug split Florida into "30"/"31") |
| COMID → landscape | **StreamCat** `https://api.epa.gov/StreamCat/streams/metrics?name=pcturbcrop2019&comid=179` (no key; rate limit 1,000) | 200 | metrics | NHDPlus **V2** COMIDs. StreamCat also carries **predicted biotic condition (Hill 2017): exclude it** |
| AU / impairment | ATTAINS MapServer layer 3 (catchment associations) | `state='OH'` → `nhdplusid` 24000200022878…, `huc12` 050902010802 | `assessmentunitidentifier`, `nhdplusid`, `huc12`, use fields | ATTAINS uses **NHDPlus HR** IDs, StreamCat/NLDI use **V2** → join via HUC12 or spatial overlay |
| Beach ↔ monitoring | WQP `organization=21OHBCH` | `ProjectIdentifier` = BEACON beach ID (e.g. OH136995 Wagar Beach); 53,069 *E. coli* / 96 sites | `ProjectIdentifier`, `MonitoringLocationIdentifier`, `ActivityStartDate/Time` | BEACON station IDs before 2015 are not in PRAWN; use the project ID |
| PWS ↔ place | EPA SAB v3 FeatureServer `https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/Water_System_Boundaries/FeatureServer` | (page verified; >44,000 CWS) | `PWSID`, polygon | ~40% of boundaries are EPA-modeled. **Intakes are not public**, so the source-water location is a proxy |
| Lakes | NLA `SITE_ID`/`COMID`; LAGOS-US LOCUS (Cheruvelil et al. 2021); MN DOW numbers; CA FHAB / NY DEC lat/lon | — | IDs | LAGOS/EDI downloads now need a free EDI account (per 01) |
| WQ observations after 2024-03-11 | WQP **WQX 3.0** `https://waterqualitydata.us/beta/` (wqx3 endpoints) | Legacy USGS-05427850 results after 2024-04-01 = **0**; 2022–Mar 2024 = 629 | — | Any temporal test set from 2024 on **must** use WQX 3.0 or USGS Water Data APIs |

### c.2 Alignment rules

1. **Decision time, not sample time.** Each label row carries a decision timestamp *t₀*: for example 08:00 on beach day *d*, or Monday of the forecast week for a fish kill. Features may only use upstream outputs **issued** before *t₀* minus the lead time.
   - Satellite inputs must respect acquisition plus processing latency.
   - Gap-filled or smoothed sensor series must be produced causally, with no two-sided filters. The audit found two-sided windows in the bloom-ahead analysis.
2. **Availability by data type:**
   - Culture FIB (18–24 h) is never available the same day.
   - qPCR (~4–5 h) is available the same day only if the program runs it in the morning (Chicago).
   - Sensors are near real-time.
   - Lab toxins take days.

   Same-day neighbor exceedances are allowed **only** when this rule permits.
3. **Spatial support.**
   - *Point labels* (beaches, kills, NAS records): use the nearest upstream prediction location on the same waterbody. Pick lake pixels for lake beaches, or the downstream-most reach within *x* km for river beaches. Record the distance and use it as a feature or a filter.
   - *Polygon labels* (HUC12, county for WHISPers, PWS service area, ATTAINS AU): area-weight or flow-weight upstream predictions over the contained or upstream reaches.
   - *Survey labels* (NARS): aggregate upstream predictions over the 1–3 years before the visit (summer means, minima of DO, maxima of temperature, SC exceedance days), plus the visit-window values.
4. **Embeddings.** Use the same windows and pooling as the variables. Report **variables-only, embeddings-only, both, and an oracle with observed WQ** wherever co-measured data exist (NARS, beaches with turbidity, Chicago sensors). This quantifies what the upstream model contributes.
5. **Cross-fitting (non-negotiable).** Build upstream predictions for each downstream fold with an upstream model trained without that fold's sites and without times after the fold's start. Use the same spatial blocks (e.g. HUC8 or 50-km) and the same temporal cut. Freeze and hash the fold assignments before any downstream training.

---

## (d) Prioritized recommendations and evaluation designs

### d.0 Common protocol (applies to all)

- **Splits, fixed and hashed before training.** Each application uses both:
  - **Temporal (forward-chaining):** e.g. train ≤ 2019, validate 2020–2021, test 2022–2025. Drop 2020 when operations were disrupted (Chicago 2020 has ~30 predictions).
  - **Spatial:** leave-one-HUC4 or 50-km block out, and **leave-one-jurisdiction (state/agency)-out** for administrative labels.

  Report random-CV only to show the leakage gap.
- **Metrics.**
  - Rare events: AUROC, AUPRC (with prevalence shown), Brier score, a reliability curve, and sensitivity at the manager's operating point. For beaches that is the USGS goals (specificity ≥ 85%); for HABs it is precision at fixed recall.
  - Forecasts: lead-time curves.
  - CIs: **block bootstrap over sites and years**.
- **Controls.** Label-permutation and site-shuffle placebos. Every table includes "location-only" and "climatology-only" rows (the old species model was a range prior; the old FIB model tied lat/lon).
- **Ablations.** Upstream variables vs embeddings vs oracle observed WQ vs none.
- **Reporting.** Pre-register the operating thresholds and the single primary metric per task. Show all regions, including failures, as the Track-A placebo lesson requires.
- **Compute.** These are tabular or small sequence models (GBM, small GRU/transformer, hierarchical GLM). One RTX 2080 Ti suffices. Data engineering is the bottleneck.

### d.1 — SafeSwim: beach FIB exceedance nowcast (0-day) and forecast (1–3 days) — **Priority 1**

- **Why.** It is a daily public-health decision with an incumbent operational method to beat. It is the clearest judge-facing story ("swimmers were told the beach was safe on days it wasn't").
- **Labels (primary).** Lab-measured exceedance at the BAV (*E. coli* > 235 cfu/100 mL culture; *Enterococcus* qPCR > 1,000 CCE). Sources:
  - WQP BEACH projects (BEACON beach ID = `ProjectIdentifier`), e.g. Ohio (53,069 results / 96 sites).
  - Chicago (culture 2006–16, qPCR 2015–26).
  - NY State Parks (2015–25).
  - King County (2019–26).
  - Local OH/MI/WI *E. coli* tables (`10_local_inventory.md` D16; fix the unit and media filters).
- **Labels (secondary).** BEACON contamination advisories/closures and NYS Parks "Closed: Exceedance". **Exclude** "Preemptive: rainfall", "Preemptive: turbidity" and "Predicted (model)" rows from the targets. Report them separately as policy actions.
- **Inputs.** Upstream turbidity, SC, temperature, flow-proxy and *E. coli* predictions at the beach and its tributary mouths, plus embeddings. Antecedent rain (GHCN). Wave/wind (Chicago sensors, or NOAA buoys **(unverified)**). Network same-day qPCR only where operationally available.
- **Evaluation.**
  - (i) Forward-chaining by season.
  - (ii) Leave-one-jurisdiction-out (Ohio ↔ Chicago ↔ NY ↔ King County).
  - (iii) Within a jurisdiction, 20-km beach-cluster blocks.
  - (iv) Optional Europe transfer: predict EEA seasonal class from SENTINEL-style satellite and climate features for inland sites (CC BY 4.0). Label = class at season *s*, inputs ≤ season *s* − 1.
- **Baselines.** See b.3. The Chicago feature-beach GM (0.74) and the official model (0.70) are the bars in Chicago. USGS performance goals and the persistence comparison apply in the Great Lakes.
- **Pitfalls specific to this task:**
  - Resampling after an exceedance (more samples during bad periods): weight by beach-day, not by sample.
  - Advisories lifted only after clean resamples (persistence leakage in secondary labels).
  - The method change from culture to qPCR in Chicago in 2017: never pool the units.
  - Single-sample variance: report the noise ceiling from duplicate samples (`dna_sample_1/2`, `samplea/b/c`).
- **Success claim to aim for.** "At the same false-alarm rate as the City of Chicago model, SafeSwim flags X% more exceedance days (2022–2025, held-out years), and it transfers to Ohio beaches with AUROC Y under leave-one-jurisdiction-out." Report it even if X ≤ 0.

### d.2 — ToxinWatch: cyanotoxin exceedance and HAB-advisory forecast — **Priority 2**

- **Why.** Dog deaths, closed lakes and drinking-water intakes. This extends the project's one audited "solid" result (microcystin, 0.755 under 5-km blocking) into a decision tool. It must fix the cross-region collapse honestly, or report it.
- **Labels (primary).** Lab microcystin > 8 µg/L and cylindrospermopsin > 15 µg/L (EPA 2019). Anatoxin-a uses the state guidance (WA values to be confirmed). Sources:
  - WQP (74k microcystin results / 6.4k sites).
  - WA database (16,575 microcystin and 12,702 anatoxin-a samples, 2007–2026).
  - NLA 2007/12/17/22 (probability sample; exceedance is rare, ~2% of lakes in 2022).
- **Labels (secondary).** CA FHAB advisory tier (Caution/Warning/Danger; 14,437 records), NY DEC confirmed and high-toxin waterbody-years, OR advisories and NYS Parks HAB closures. All are triggered partly by visual biomass, so a model fed predicted chl-a can look good on them circularly. Treat them as secondary targets and report "chl-a-only" as a baseline.
- **Inputs.** Upstream chl-a, cyano/microcystin, temperature, nutrients, SC and turbidity. Satellite CI (CyAN/OLCI) where available. Lake morphometry (LakeCat / LAGOS). Weather.
- **Evaluation.**
  - (i) 20-km blocks plus leave-one-EPA-region-out (headline: generalization to new regions).
  - (ii) Forward-chaining (train ≤ 2020, test 2022–2025).
  - (iii) "New lakes" vs "known lakes, future weeks", reported separately.
  - (iv) NLA as an **unbiased external test** (design-weighted). This answers "does it work on a random lake?", which WQP's bloom-chasing samples cannot.
- **Baselines.** Persistence; chl-a/CI threshold; INLA (Schaeffer 2024 style, re-run on toxin labels); the prior GBM.
- **Pitfalls:**
  - Preferential sampling: report performance stratified by "routine" vs "complaint-driven" sampling where metadata allow (CA `Case_Class`: Event Response vs Monitoring_*).
  - Analyte heterogeneity (MC-LR vs total MC; ELISA vs LC-MS).
  - Florida dominance in WQP (69% of the local cyanotoxin pull): cap rows per state or weight by state.

### d.3 — KillAlert: fish-kill early warning — **Priority 3**

- **Why.** A visible ecological disaster, a direct consumer of predicted DO and temperature, and **no AUC-reported ML benchmark exists** (searched). That makes it a credible "first benchmarked model" claim.
- **Labels.** Iowa DNR events (1,035; use 2000+; split **natural/environmental** from anthropogenic causes, since manure or fertilizer spills are not predictable from WQ forecasts and should be a separate class or excluded). Delaware (524). WI/MN events via Till 2019 / Tye 2022 if the authors or agencies share them **(access unverified)**. Florida by records request.
- **Unit and negatives.** Reach (COMID) × week. Two designs:
  - (a) **Case-crossover:** compare the event week to control weeks at the same reach in other years, same season. This removes reach-level reporting propensity, the biggest bias in kill data.
  - (b) A population design with all reach-weeks as candidates, with an effort covariate (road density, population within 5 km).
- **Inputs.** Upstream DO minima, temperature maxima and anomalies, chl-a, turbidity spikes, nitrate/ammonia proxies, and embeddings over the preceding 1–14 days. Weather.
- **Evaluation.** Leave-one-year-out (heatwave years held out) plus leave-HUC8-out. Report lead-time curves (days before the event when risk first crosses the threshold).
- **Baselines.** Climatology; temperature-anomaly logistic (Till/Tye style); observed-DO threshold where sensors exist (oracle).
- **Pitfalls.** Cause misattribution; multi-day events counted once; Iowa events are mostly streams while WI/MN are lakes, so report them separately.

### d.4 — TapRisk: drinking-water nitrate and DBP risk for surface-water systems — **Priority 4**

- **Why.** Health impact for millions (Allaire 2018: ~21 M people on violating systems in 2015) and a strong environmental-justice angle (rural and small systems). It links SENTINEL's nutrient, turbidity and temperature predictions to tap-water outcomes.
- **Labels:**
  - SDWIS health-based violations (nitrate MCL; Stage 2 DBP TTHM/HAA5 LRAA; SWTR turbidity TT) by PWS-quarter, from ECHO bulk or Envirofacts.
  - Concentrations: UCMR4 HAA5 (63,484 results / 4,924 PWS), raw-water TOC and bromide (49,454 results / 3,444 PWS), and SYR4 nitrate/DBP data (2012–2019).
- **Join.** PWSID → SAB v3 polygon → HUC12s → upstream-network aggregation of predictions (proxy intake catchment). Restrict to `primary_source_code` SW. Flag the ~40% of modeled boundaries.
- **Evaluation.** Leave-one-state-out (primacy agencies differ) plus temporal (train ≤ 2018, test 2019–2024). Do not let one system's quarters span both train and test.
- **Baselines.** Persistence (prior violation), system-size/source/state logistic, Pennino-style RF on StreamCat.
- **Pitfalls.** Under-reporting by small systems; LRAA compliance lags; treatment confounds DBPs (report raw-water TOC/bromide prediction as the cleaner, treatment-free target); intake locations unavailable. Use CDC SVI 2022 for stratified fairness metrics.

### d.5 — BioIntegrity: biological condition from predicted WQ regimes — **Priority 5**

- **Why.** Clean Water Act "biological integrity" is the canonical ecosystem-health endpoint, and the national, randomized, COMID-joined labels are the cleanest in this report.
- **Coordination with 04.** `04_ecotox_bioassessment.md` designs the community ↔ chemistry models. Here the question is narrower: **do SENTINEL's *predicted* WQ regimes add skill beyond StreamCat landscape metrics for predicting condition class at unsampled reaches?**
- **Labels.** NRSA `BENT_MMI_COND` and `FISH_MMI_COND` (2008–09, 2013–14, 2018–19); NLA benthic/zooplankton MMI. Use Good vs Poor as the primary binary task and ordinal Good/Fair/Poor as secondary.
- **Evaluation.** Train on the 2008–09 and 2013–14 cycles → test on 2018–19 (temporal). Add leave-one-AG_ECO9-out (spatial). Report design-weighted and unweighted results. Revisit sites must not straddle folds.
- **Baselines.** Hill 2017 RF re-run under the identical blocked protocol; geography-only; ecoregion prevalence. Report an "oracle" with NRSA's own co-measured chemistry as an upper bound, never as an input to the "predicted" model.
- **Pitfalls.** StreamCat's predicted-condition columns; MMI regional benchmarks; NRSA 2023–24 not yet released (a future external test).

### d.6 — MusselGuard: dreissenid invasion forecast — **Priority 6**

- **Why.** A true **forecast** evaluation is possible (predict which lakes become infested next) with real confirmations, and it uses upstream SC and pH as calcium proxies.
- **Labels.** MN infested-waters list (year listed; confirmed vs "connected") and NAS established records (HUC12, year). Negatives: MN lakes not listed by the end of the test window, treated as **presumed negatives** with documented label noise.
- **Evaluation.** Rolling origin: train on infestation status as of year *T*, predict new confirmations in *T*+1…*T*+5 among uninfested lakes, for *T* = 2012, 2015, 2018. Add leave-one-state-out for NAS-based suitability.
- **Baselines.** Distance to the nearest infested water plus connectivity; Ca-risk rule (Whittier 2008, with measured WQP Ca where available); co-kriging (Kanankege 2018: AUC 0.78).
- **Pitfalls.** Detection effort and boater traffic dominate spread (boater-network data are not public); presence-only NAS; MN "connected" listings are administrative.

### d.7 — TroutRefuge: cold-water fish occupancy under predicted thermal regimes — **Priority 7**

- **Labels.** Salmonid presence/absence at **fish-surveyed** sites only: NRSA fish counts, plus state surveys if obtainable. This is the fix for the old "80% never surveyed but labeled absent" error.
- **Inputs.** Upstream-predicted July mean temperature, 7-day max, exceedance-days above species thresholds (Wehrly et al. 2007), DO minima; StreamCat.
- **Evaluation.** Within native range only. Leave-one-HUC6-out. Compare with modeled temperature from NorWeST/EcoSHEDS as inputs (the published approach).
- **Baselines.** Range-only prior; DeWeber & Wagner-style hierarchical logistic (AUC ≈ 0.78).
- **Pitfalls.** Stocking; survey-method differences; EcoSHEDS occupancy outputs are model predictions and must not be used as labels.

### d.8 — RiverPulse: GPP/ER at reaches without sensors — **Priority 8**

- **Labels.** Powell Center daily GPP/ER (356 rivers, 490,907 site-days; COMIDs given).
- **Evaluation.** Leave-site-out grouped by HUC4, plus leave-years-out.
- **Circularity control.** GPP/ER were inverted from each site's DO series. Either exclude these sites' DO from upstream training, or restrict inputs to non-DO upstream outputs (temperature, turbidity, light proxies, SC, nutrients).
- **Baselines.** Site climatology; light + season; RF (Segatto 2021 style).
- **Value.** Scientific novelty in ecosystem function; lower lay-judge appeal.

### d.9 Stretch targets (only after d.1–d.4 are solid)

- **Lake bottom-water hypoxia** (NLA DO profiles 2007–2022): predict bottom DO < 2 mg/L from surface-level upstream predictions plus morphometry. Evaluate leave-one-ecoregion-out and on the 2022 cycle. This is a real "dead zone" label without Gulf-style tiny n.
- **ATTAINS screening:** predict aquatic-life "Not Supporting" for assessed AUs, and rank the ~230k "Insufficient Information / Not Assessed" line features (several per AU) for monitoring. Evaluate leave-one-state-out and forward on reporting cycles. **Caution:** impairment decisions use the same monitoring data the upstream may have trained on, so cross-fitting is mandatory.
- **Avian botulism** (WHISPers, county-week): leave-one-year-out; type C wetlands vs type E Great Lakes reported separately.
- **Gulf/Chesapeake hypoxia:** case study only (n ≈ 38 years).

### d.10 Ranking rationale

| Rank | App | Label quality | n | Join | Held-out rigor | Incumbent to beat | Judge appeal |
|---|---|---|---|---|---|---|---|
| 1 | SafeSwim | High (lab) | 10⁴–10⁵ | Direct | Year + jurisdiction | Yes (Chicago, USGS) | Very high |
| 2 | ToxinWatch | High (lab) / medium (advisory) | 10⁴ | Direct | 20-km + region + NLA | Yes (INLA) | Very high |
| 3 | KillAlert | Medium (reporting bias) | 10³ | lat/lon | Case-crossover + year | None published (opportunity) | High |
| 4 | TapRisk | Medium (administrative) | 10⁴–10⁵ PWS-quarters | Proxy (no intakes) | State + time | Yes (Pennino) | High |
| 5 | BioIntegrity | High (survey) | ~2×10³ per cycle | COMID | Cycle + ecoregion | Yes (Hill 2017) | Medium-high |
| 6 | MusselGuard | Medium (detection) | 10³ | HUC12 / DOW | Rolling origin | Yes (co-kriging) | Medium |
| 7 | TroutRefuge | Medium-high | 10³ | COMID | HUC6 | Yes (DeWeber & Wagner) | Medium |
| 8 | RiverPulse | High (but model-derived) | 4.9×10⁵ site-days | COMID | Site / HUC4 | Weak | Medium-low |

---

## (e) Risks

1. **Reporting and effort bias.**
   - Fish kills, HAB advisories, NAS records and WHISPers depend on who looks.
   - *Mitigations:* case-crossover designs, effort covariates, unbiased external tests (NLA, NRSA), and reporting performance by observation channel (CA `Case_Class`).
2. **Policy-circular labels.**
   - Preemptive rainfall closures; advisories driven by visual biomass (chl-a); MN "connected" listings; 303(d) listings derived from monitoring data.
   - *Mitigations:* measured labels as primary, administrative labels as secondary, and explicit "rule-only" baselines.
3. **Upstream-to-downstream leakage.**
   - The upstream *E. coli* and microcystin models trained on the same samples used as downstream labels.
   - Same-day neighbor results not available at decision time; non-causal smoothing.
   - *Mitigations:* cross-fitting, decision-time availability rules, and causal filters (§c.2).
4. **Spatial autocorrelation and pseudo-replication.**
   - Microcystin: 40% of rows had a same-day sample within 5 km. Beaches cluster on one shoreline. Revisit sites in NARS.
   - *Mitigations:* block splits, cluster bootstrap, flow-connected sites in the same fold.
5. **Distribution shift and data-plumbing drift.**
   - Legacy WQP lacks USGS data after 2024-03-11.
   - ATTAINS REST needs an API key.
   - NLDI moved; EDI downloads need an account; CA FHAB CSV links redirect (use the CKAN datastore API).
   - BEACON reports are APEX pages that reset plain `curl` (use UI export or the WQP project route).
   - *Mitigation:* pin dataset versions and download dates in a manifest.
6. **Political and availability risk.**
   - EJScreen and CEJST were removed from federal servers in early 2025.
   - Cite mirrors with provenance (PEDP, UChicago data rescue) and prefer CDC SVI 2022, which is live.
   - Keep local snapshots of every label set used.
7. **Small n for the most dramatic events.**
   - Gulf hypoxia (~38 years), *Naegleria*, outbreaks (state-month).
   - Use them only as case studies or validation, never as training targets.
8. **Threshold heterogeneity.**
   - BAV 235 vs 190; state toxin guidance values; MMI regional benchmarks.
   - *Mitigation:* fix one threshold per task and run sensitivity analyses.
9. **Over-claiming risk** (the previous version's failure mode).
   - Every table needs persistence/climatology/location-only rows, all regions including failures, and CIs.
   - Never quote a random-split number next to a blocked one.

---

## (f) References

### Papers (metadata verified via Crossref / PubMed / OpenAlex unless marked)

- Allaire M, Wu H, Lall U. National trends in drinking water quality violations. *PNAS* 115(9):2078–2083, 2018. doi:10.1073/pnas.1719805115
- Appling AP, Read JS, Winslow LA, Arroita M, Bernhardt ES, Griffiths NA, Hall RO Jr, Harvey JW, Heffernan JB, Stanley EH, Stets EG, Yackulic CB. The metabolic regimes of 356 rivers in the United States. *Scientific Data* 5:180292, 2018. doi:10.1038/sdata.2018.292 (data: doi:10.5066/F70864KX)
- Archer EJ, Baker-Austin C, Osborn TJ, Jones NR, Martínez-Urtaza J, Trinanes J, Oliver JD, González FJC, Lake IR. Climate warming and increasing *Vibrio vulnificus* infections in North America. *Scientific Reports* 13:3893, 2023. doi:10.1038/s41598-023-28247-2
- Backer LC, Landsberg JH, Miller M, Keel K, Taylor TK. Canine cyanotoxin poisonings in the United States (1920s–2012): review of suspected and confirmed cases from three data sources. *Toxins* 5(9):1597–1628, 2013. doi:10.3390/toxins5091597
- Baker-Austin C, Trinanes JA, Taylor NGH, Hartnell R, Siitonen A, Martinez-Urtaza J. Emerging *Vibrio* risk at high latitudes in response to ocean warming. *Nature Climate Change* 3:73–77, 2013 (online 2012). doi:10.1038/nclimate1628
- Bernhardt ES, Savoy P, Vlah MJ, Appling AP, Koenig LE, Hall RO Jr, Arroita M, et al. Light and flow regimes regulate the metabolism of rivers. *PNAS* 119(8):e2121976119, 2022. doi:10.1073/pnas.2121976119
- Brooks WR, Corsi SR, Fienen MN, Carvin RB. Predicting recreational water quality advisories: a comparison of statistical methods. *Environmental Modelling & Software* 76:81–94, 2016. doi:10.1016/j.envsoft.2015.10.012
- Cheruvelil KS, Soranno PA, McCullough IM, Webster KE, Rodriguez LK, Smith NJ. LAGOS-US LOCUS v1.0. *Limnology and Oceanography Letters* 6(5):270–292, 2021. doi:10.1002/lol2.10203
- Collier SA, Deng L, Adam EA, Benedict KM, Beshearse EM, Blackstock AJ, Bruce BB, et al. Estimate of burden and direct healthcare cost of infectious waterborne disease in the United States. *Emerging Infectious Diseases* 27(1):140–149, 2021. doi:10.3201/eid2701.190676
- Comte L, Carvajal-Quintero J, Tedesco PA, et al. RivFishTIME: a global database of fish time-series to study global change ecology in riverine systems. *Global Ecology and Biogeography* 30(1):38–50, 2021. doi:10.1111/geb.13210
- DeWeber JT, Wagner T. Predicting brook trout occurrence in stream reaches throughout their native range in the eastern United States. *Transactions of the American Fisheries Society* 144(1):11–24, 2015. doi:10.1080/00028487.2014.963256
- Dugan HA, Bartlett SL, Burke SM, Doubek JP, Krivak-Tetley FE, Skaff NK, Summers JC, et al. Salting our freshwater lakes. *PNAS* 114(17):4453–4458, 2017. doi:10.1073/pnas.1620211114
- Fox EW, Hill RA, Leibowitz SG, Olsen AR, Thornbrugh DJ, Weber MH. Assessing the accuracy and stability of variable selection methods for random forest modeling in ecology. *Environmental Monitoring and Assessment* 189:316, 2017. doi:10.1007/s10661-017-6025-0
- Francy DS, Brady AMG, Carvin RB, Corsi SR, Fuller LM, Harrison JH, Hayhurst BA, Lant J, Nevers MB, Terrio PJ, Zimmerman TM. Developing and implementing the use of predictive models for estimating water quality at Great Lakes beaches. USGS Scientific Investigations Report 2013-5166, 2013. doi:10.3133/sir20135166
- Francy DS, Brady AMG, Zimmerman TM. Data for multiple linear regression models for estimating *E. coli* … Great Lakes NowCast, 2019. USGS data release, 2021. doi:10.5066/P9Y9O1YJ
- Gharpure R, Gleason M, Salah Z, Blackstock AJ, Hess-Homeier D, Yoder JS, Ali IKM, et al. Geographic range of recreational water–associated primary amebic meningoencephalitis, United States, 1978–2018. *Emerging Infectious Diseases* 27(1), 2021. doi:10.3201/eid2701.202119
- Graciaa DS, Cope JR, Roberts VA, Cikesh BL, Kahler AM, Vigar M, Hilborn ED, et al. Outbreaks associated with untreated recreational water — United States, 2000–2014. *MMWR* 67(25):701–706, 2018. doi:10.15585/mmwr.mm6725a1
- Hendriksen RS, Munk P, Njage P, et al. Global monitoring of antimicrobial resistance based on metagenomics analyses of urban sewage. *Nature Communications* 10:1124, 2019. doi:10.1038/s41467-019-08853-3 (sewage, not surface water; listed because the earlier ARG module lacked real labels)
- Hill RA, Fox EW, Leibowitz SG, Olsen AR, Thornbrugh DJ, Weber MH. Predictive mapping of the biotic condition of conterminous U.S. rivers and streams. *Ecological Applications* 27(8):2397–2415, 2017. doi:10.1002/eap.1617
- Hill RA, Weber MH, Leibowitz SG, Olsen AR, Thornbrugh DJ. The Stream-Catchment (StreamCat) dataset. *JAWRA* 52(1):120–128, 2016. doi:10.1111/1752-1688.12372
- Hollister JW, Kreakie BJ. Associations between chlorophyll *a* and various microcystin health advisory concentrations. *F1000Research* 5:151, 2016. doi:10.12688/f1000research.7955.2
- Isaak DJ, Young MK, Nagel DE, Horan DL, Groce MC. The cold-water climate shield. *Global Change Biology* 21(7):2540–2553, 2015. doi:10.1111/gcb.12879
- Isaak DJ, Wenger SJ, Peterson EE, Ver Hoef JM, Nagel DE, Luce CH, Hostetler SW, et al. The NorWeST summer stream temperature model and scenarios for the western U.S. *Water Resources Research* 53(11):9181–9205, 2017. doi:10.1002/2017WR020969 (data: doi:10.2737/RDS-2016-0033)
- Jacobs JM, Rhodes M, Brown CW, Hood RR, Leight A, Long W, Wood R. Modeling and forecasting the distribution of *Vibrio vulnificus* in Chesapeake Bay. *Journal of Applied Microbiology* 117(5):1312–1327, 2014. doi:10.1111/jam.12624
- Jane SF, Hansen GJA, Kraemer BM, et al. Widespread deoxygenation of temperate lakes. *Nature* 594:66–70, 2021. doi:10.1038/s41586-021-03550-y
- Kanankege KST, Alkhamis MA, Phelps NBD, Perez AM. A probability co-kriging model to account for reporting bias and recognize areas at high risk for zebra mussels and Eurasian watermilfoil invasions in Minnesota. *Frontiers in Veterinary Science* 4:231, 2018. doi:10.3389/fvets.2017.00231
- Kaushal SS, Likens GE, Pace ML, Utz RM, Haq S, Gorman J, Grese M. Freshwater salinization syndrome on a continental scale. *PNAS* 115(4), 2018. doi:10.1073/pnas.1711234115
- La VT, Cooke SJ. Advancing the science and practice of fish kill investigations. *Reviews in Fisheries Science* 19(1):21–33, 2011. doi:10.1080/10641262.2010.531793
- Lafrancois BM, Riley SC, Blehert DS, Ballmann AE. Links between type E botulism outbreaks, lake levels, and surface water temperatures in Lake Michigan, 1963–2008. *Journal of Great Lakes Research* 37(1):86–91, 2011. doi:10.1016/j.jglr.2010.10.003
- Loftin KA, Graham JL, Hilborn ED, Lehmann SC, Meyer MT, Dietze JE, Griffith CB. Cyanotoxins in inland lakes of the United States: occurrence and potential recreational health risks in the EPA National Lakes Assessment 2007. *Harmful Algae* 56:77–90, 2016. doi:10.1016/j.hal.2016.04.001
- Lucius N, Rose K, Osborn C, Sweeney ME, Chesak R, Beslow S, Schenk T Jr. Predicting *E. coli* concentrations using limited qPCR deployments at Chicago beaches. *Water Research X* 2:100016, 2019. doi:10.1016/j.wroa.2018.100016
- Mazor RD, Rehn AC, Ode PR, Engeln M, Schiff KC, Stein ED, Gillett DJ, et al. Bioassessment in complex environments: designing an index for consistent meaning in different settings. *Freshwater Science* 35(1):249–271, 2016. doi:10.1086/684130
- McDonald YJ, Jones NE. Drinking water violations and environmental justice in the United States, 2011–2015. *American Journal of Public Health* 108(10):1401–1407, 2018. doi:10.2105/AJPH.2018.304621
- Mishra S, Stumpf RP, Schaeffer BA, Werdell PJ, Loftin KA, Meredith A. Measurement of cyanobacterial bloom magnitude using satellite remote sensing. *Scientific Reports* 9:18310, 2019. doi:10.1038/s41598-019-54453-y
- Myer MH, Urquhart E, Schaeffer BA, Johnston JM. Spatio-temporal modeling for forecasting high-risk freshwater cyanobacterial harmful algal blooms in Florida. *Frontiers in Environmental Science* 8:581091, 2020. doi:10.3389/fenvs.2020.581091
- Paul MJ, LeDuc SD, Lassiter MG, Moorhead LC, Noyes PD, Leibowitz SG. Wildfire induces changes in receiving waters: a review with considerations for water quality management. *Water Resources Research* 58(9):e2021WR030699, 2022. doi:10.1029/2021WR030699
- Pennino MJ, Leibowitz SG, Compton JE, Hill RA, Sabo RD. Patterns and predictions of drinking water nitrate violations across the conterminous United States. *Science of the Total Environment* 722:137661, 2020. doi:10.1016/j.scitotenv.2020.137661
- Raoelison OD, Valenca R, Lee A, Karim S, Webster JP, Poulin BA, Mohanty SK. Wildfire impacts on surface water quality parameters: cause of data variability and reporting needs. *Environmental Pollution* 317:120713, 2023. doi:10.1016/j.envpol.2022.120713
- Read JS, Jia X, Willard J, Appling AP, Zwart JA, Oliver SK, Karpatne A, et al. Process-guided deep learning predictions of lake water temperature. *Water Resources Research* 55(11):9173–9190, 2019. doi:10.1029/2019WR024922
- Richard JC, Leis E, Dunn CD, Agbalog R, Waller D, Knowles S, Putnam J, Goldberg TL. Mass mortality in freshwater mussels (*Actinonaias pectorosa*) in the Clinch River, USA, linked to a novel densovirus. *Scientific Reports* 10:14498, 2020. doi:10.1038/s41598-020-71459-z
- Rousso BZ, Bertone E, Stewart R, Hamilton DP. A systematic literature review of forecasting and predictive models for cyanobacteria blooms in freshwater lakes. *Water Research* 182:115959, 2020. doi:10.1016/j.watres.2020.115959
- Savoy P, Appling AP, Heffernan JB, Stets EG, Read JS, Harvey JW, Bernhardt ES. Metabolic rhythms in flowing waters: an approach for classifying river productivity regimes. *Limnology and Oceanography* 64(5):1835–1851, 2019. doi:10.1002/lno.11154
- Scanlon BR, Fakhreddine S, Reedy RC, Yang Q, Malito JG. Drivers of spatiotemporal variability in drinking water quality in the United States. *ES&T* 56(18):12965–12974, 2022. doi:10.1021/acs.est.1c08697
- Scavia D, Bertani I, Obenour DR, Turner RE, Forrest DR, Katin A. Ensemble modeling informs hypoxia management in the northern Gulf of Mexico. *PNAS* 114(33):8823–8828, 2017. doi:10.1073/pnas.1705293114
- Schaeffer BA, Bailey SW, Conmy RN, Galvin M, Ignatius AR, Johnston JM, Keith DJ, et al. Mobile device application for monitoring cyanobacteria harmful algal blooms using Sentinel-3 satellite Ocean and Land Colour Instruments. *Environmental Modelling & Software* 109:93–103, 2018. doi:10.1016/j.envsoft.2018.08.015
- Schaeffer BA, Reynolds N, Ferriby H, Salls W, Smith D, Johnston JM, Myer M. Forecasting freshwater cyanobacterial harmful algal blooms for Sentinel-3 satellite resolved U.S. lakes and reservoirs. *Journal of Environmental Management* 349:119518, 2024. doi:10.1016/j.jenvman.2023.119518
- Searcy RT, Boehm AB. A day at the beach: enabling coastal water quality prediction with high-frequency sampling and data-driven models. *ES&T* 55(3):1908–1918, 2021. doi:10.1021/acs.est.0c06742
- Searcy RT, Boehm AB. Know before you go: data-driven beach water quality forecasting. *ES&T* 57(46):17930–17939, 2023 (online 2022). doi:10.1021/acs.est.2c05972
- Segatto PL, Battin TJ, Bertuzzo E. The metabolic regimes at the scale of an entire stream network unveiled through sensor data and machine learning. *Ecosystems* 24(7):1792–1809, 2021. doi:10.1007/s10021-021-00618-8
- Thoe W, Gold M, Griesbach A, Grimmer M, Taggart ML, Boehm AB. Predicting water quality at Santa Monica Beach: evaluation of five different models for public notification of unsafe swimming conditions. *Water Research* 67:105–117, 2014. doi:10.1016/j.watres.2014.09.001
- Thronson A, Quigg A. Fifty-five years of fish kills in coastal Texas. *Estuaries and Coasts* 31(4):802–813, 2008. doi:10.1007/s12237-008-9056-5
- Till A, Rypel AL, Bray A, Fey SB. Fish die-offs are concurrent with thermal extremes in north temperate lakes. *Nature Climate Change* 9(8):637–641, 2019. doi:10.1038/s41558-019-0520-y (metrics unverified)
- Tye SP, Siepielski AM, Bray A, Rypel AL, Phelps NBD, Fey SB. Climate warming amplifies the frequency of fish mass mortality events across north temperate lakes. *Limnology and Oceanography Letters* 7(6):510–519, 2022. doi:10.1002/lol2.10274
- Wehrly KE, Wang L, Mitro M. Field-based estimates of thermal tolerance limits for trout: incorporating exposure time and temperature fluctuation. *Transactions of the American Fisheries Society* 136(2):365–374, 2007. doi:10.1577/T06-163.1
- Whittier TR, Ringold PL, Herlihy AT, Pierson SM. A calcium-based invasion risk assessment for zebra and quagga mussels (*Dreissena* spp). *Frontiers in Ecology and the Environment* 6(4):180–184, 2008. doi:10.1890/070073
- Willard JD, Read JS, Appling AP, Oliver SK, Jia X, Kumar V. Predicting water temperature dynamics of unmonitored lakes with meta-transfer learning. *Water Resources Research* 57(7):e2021WR029579, 2021. doi:10.1029/2021WR029579
- Zhang J, Qiu H, Li X, Niu J, Nevers MB, Hu X, Phanikumar MS. Real-time nowcasting of microbiological water quality at recreational beaches: a wavelet and artificial neural network-based hybrid modeling approach. *ES&T* 52(15):8446–8455, 2018. doi:10.1021/acs.est.8b01022
- Zhou Y, Obenour DR, Scavia D, Johengen TH, Michalak AM. Spatial and temporal trends in Lake Erie hypoxia, 1987–2007. *ES&T* 47(2):899–905, 2013. doi:10.1021/es303401b

### Agency documents and datasets (URLs verified 2026-10-22)

**EPA**
- BEACON 2.0 User's Guide v4.1 (Dec 2025): `https://beacon.epa.gov/ords/beacon2/r/beacon_apex/files/static/v63/beacon_usersguide.pdf`
- EPA 2020 swimming-season report: `https://www.epa.gov/system/files/documents/2021-08/beach-swimming-season-report-2020.pdf`
- 2012 RWQC (EPA 820-F-12-058): `https://www.epa.gov/sites/default/files/2015-10/documents/rwqc2012.pdf`
- 2019 cyanotoxin swimming-advisory fact sheet: `https://www.epa.gov/sites/default/files/2019-05/documents/hh-rec-criteria-habs-factsheet-2019.pdf`
- 2021 lake nutrient criteria (EPA-822-R-21-005): `https://www.epa.gov/system/files/documents/2021-08/nutrient-lakes-reservoirs-report-final.pdf`
- Aquatic-life criteria table (chloride): `https://www.epa.gov/wqc/national-recommended-water-quality-criteria-aquatic-life-criteria-table`
- Conductivity benchmark: `https://cfpub.epa.gov/ncea/risk/recordisplay.cfm?deid=233809`
- Predictive beach models / Virtual Beach: `https://www.epa.gov/beaches/models-predicting-beach-water-quality`
- NARS data: `https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys`
- UCMR occurrence data: `https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule`
- SYR4: `https://www.epa.gov/dwsixyearreview/six-year-review-4-compliance-monitoring-data-2012-2019`
- CWS service-area boundaries v3: `https://www.epa.gov/ground-water-and-drinking-water/community-water-system-service-area-boundaries`
- ECHO SDWA bulk: `https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip`
- Envirofacts API: `https://www.epa.gov/enviro/envirofacts-data-service-api`
- ATTAINS geospatial: `https://gispub.epa.gov/arcgis/rest/services/OW/ATTAINS_Assessment/MapServer`
- StreamCat API: `https://api.epa.gov/StreamCat/streams/metrics`

**CDC**
- OHHABS: `https://www.cdc.gov/ohhabs/data/index.html`
- NORS/BEAM: `https://data.cdc.gov/resource/5xkq-dg7x.json`
- SVI: `https://www.atsdr.cdc.gov/place-health/php/svi/index.html`
- EJI: `https://www.atsdr.cdc.gov/place-health/php/eji/index.html`

**USGS**
- NAS API: `https://nas.er.usgs.gov/api/v2/`
- WHISPers API: `https://whispers.usgs.gov/api/`
- NLDI: `https://api.water.usgs.gov/nldi/`
- WBD service: `https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer`
- Powell Center metabolism (ScienceBase): `https://www.sciencebase.gov/catalog/item/59bff507e4b091459a5e0982`
- EcoSHEDS: `https://usgs.gov/apps/ecosheds/`

**Water Quality Portal:** `https://www.waterqualitydata.us/` (WQX 3.0 at `/beta/`)

**State and city**
- Iowa DNR fish kills: `https://programs.iowadnr.gov/fishkill/`
- Delaware fish kills: `https://data.delaware.gov/resource/ycvg-4ys9.json`, `.../8pfi-yck7.json`
- California FHAB: data.ca.gov dataset "surface-water-freshwater-harmful-algal-blooms"
- NY DEC HABs: `https://data.ny.gov/resource/95my-wijm.json`, `.../qtg2-hjth.json`
- NY State Parks beaches: `https://data.ny.gov/resource/wwwd-za77.json`
- Oregon OHA archive (URL in §a.1)
- Washington Toxic Algae: `https://www.nwtoxicalgae.org/`
- Chicago: `https://data.cityofchicago.org/resource/{2ivx-z93u,xvsz-3xcj,t62e-8nvc,qmqz-2xku,k7hf-8y75}.json`
- King County: `https://data.kingcounty.gov/resource/mbzm-4r9y.json`
- MN infested waters: `https://files.dnr.state.mn.us/eco/invasives/infested-waters.xlsx`
- Florida FWC hotline page: `https://myfwc.com/research/saltwater/health/fish-kills-hotline/`

**Other**
- EEA bathing water (CC BY 4.0): datahub item `c3858959-90da-4c1b-b9ca-492db0e514df`
- Gulf hypoxia: `https://gulfhypoxia.net/research/shelfwide-cruises/`
- MTBS: `https://www.mtbs.gov/direct-download`
- GBIF API: `https://api.gbif.org/v1/`
- EJScreen mirror (PEDP): `https://pedp-ejscreen.azurewebsites.net/`
- EJScreen 2024 data rescue (UChicago): `https://knowledge.uchicago.edu/records/z7nbk-5bk41`
- CEJST mirror: `https://public-environmental-data-partners.github.io/j40-cejst-2/`

### Not verified or not found (do not cite without checking)

- Kansas KDHE HAB archive (page returned 403).
- Vermont, Utah, Ohio ODH and Florida DEP HAB archives (not checked).
- WA DOH numeric toxin guidance values.
- Wisconsin/Minnesota fish-kill data access (Till 2019 / Tye 2022 data availability).
- Till 2019 metrics.
- Amphibian Disease Portal counts and API.
- EBTJV data access.
- CDC COVIS public granularity.
- Montana "hoot-owl" closure archive.
- USGS BioData host `aquatic.biodata.usgs.gov` did not resolve on 2026-10-22; use WQP biological profiles.
- A national ML benchmark for ATTAINS/303(d) prediction: none found.
- A public freshwater-mussel die-off database: none found.

---

## Appendix A. Local samples and scripts (scratchpad; temporary)

Root: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/downstream/` (122 MB total). These files are untrusted downloads; read them with `python3 -B -I`.

- **`fib_beach/`**
  - `chicago/beach_lab_data.csv` (4.4 MB), `beach_ecoli_predictions.csv` (1.4 MB), `beach_swim_advisories.csv`
  - `nys_parks/beach_water_testing_2015plus.csv` (2.0 MB)
  - `kingcounty/freshwater_swim_beach.csv`
  - `ramsey/beach_wq.csv`
  - `eea_bwd/bw_assessment_1990_2025.xlsx` (42 MB)
- **`drinking/ucmr4/`:** `ucmr4.zip` plus extracted cyanotoxin and HAA-indicator tables. The 154 MB `UCMR4_All.txt` was deleted after summarizing.
- **`nars/`:**
  - NRSA 2018–19 site info, benthic MMI, fish MMI, enterococci
  - NRSA 2013–14 all-condition table
  - NLA 2022 site info, algal toxins, profiles, condition estimates
- **`invasive/`:** `nas/nas_species_5.json`, `nas/nas_species_95.json`, `mn_infested/infested-waters.xlsx`
- **`fishkill/iowa/events.json`**
- **`metabolism/site_data.tsv`**
- **`txt/`:** extracted text of the BEACON guide, EPA reports and fact sheets; the WQP Ohio project list

Scripts: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/scripts_downstream/`
- `summ_beach.py`, `summ_eea2.py`, `summ_ucmr4.py`, `summ_ucmr4_haa.py`, `summ_nars.py`, `nas_fetch.py`, `mn_inf.py`
- `chicago_eval.py` and `chicago_eval2.py` (§b.2 numbers)
- `crossref_batch.py`, `pmabs.sh`, `oa_abs.py` (reference checks)
