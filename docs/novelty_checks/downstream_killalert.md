# Novelty check — D3 KillAlert (fish-kill risk prediction with held-out discrimination metrics)

**Date of search:** 2026-10-24 (UTC). **Searcher:** Downstream workstream (AI-assisted; Bryan Cheng).
**Question:** Is there a published fish-kill / fish mass-mortality *prediction* model evaluated on
held-out events with a discrimination metric (AUC/ROC, or an equivalent hit/false-alarm skill score)?
If not, "first-in-class" is available for KillAlert under RESEARCH_PLAN §15.1 — provided the model
also beats every applicable baseline (season climatology, temperature-anomaly conditional logistic
in the Till/Tye style, GBM) with paired-bootstrap CIs excluding 0.

## Queries run (web search tool, US results, 2026-10-24)

| # | Query | Relevant hits |
|---|---|---|
| 1 | `fish kill prediction model machine learning AUC weather` | Chen, Nicholson & Cheng 2020 (Sci. Rep.) |
| 2 | `predicting fish kill events lakes logistic regression ROC temperature summerkill winterkill` | Tye et al. 2022; Till et al. 2019; Chen 2020 |
| 3 | `fish mortality event forecast random forest dissolved oxygen river "fish kill" risk model validation` | DO-forecast papers (Korea urban streams, WRR 2021; Mississippi MLP-RF 2024) — predict DO, not kills |
| 4 | `"fish kill" OR "fish die-off" "AUC" classification model held-out validation` | none relevant (aquaculture mortality classifier, NZ fish *distribution* models) |
| 5 | `Tye 2022 "Fish die-offs are concurrent with thermal extremes" summerkill prediction accuracy boosted regression` | Tye 2022 / Till 2019 abstracts and press only |
| 6 | `Tye 2022 lol2.10274 summerkill logistic model "area under" OR AUC OR "true skill" fish mortality events Minnesota Wisconsin` | Phelps et al. 2019 (J. Aquat. Anim. Health) |
| 7 | `Phelps 2019 "fish kill" Minnesota ecological niche model land surface temperature partial ROC Journal of Aquatic Animal Health` | Phelps 2019 abstract only |

Prior search in `docs/research/07_downstream_applications.md` (2026-10-22, §b.1 E1): "No AUC-reported
fish-kill ML benchmark found (searched)".

## Closest prior work and why it is not a held-out discrimination benchmark

| Work | What it does | Held-out discrimination metric? |
|---|---|---|
| Chen, Nicholson & Cheng 2020, *Sci. Rep.* 10:17003, doi:10.1038/s41598-020-73922-3 (full text read on PMC) | Self-organizing map on hourly weather around 19 grey-mullet kills (Danshui River, Taiwan, 2010–2018); ±1 SD warning thresholds | **No** — no train/test split, no AUC/accuracy; thresholds derived on the same data |
| Till et al. 2019, *Nat. Clim. Change* 9:637–641, doi:10.1038/s41558-019-0520-y | 502 Wisconsin die-offs vs simulated lake temperatures; summerkills linked to heat extremes | Not found in abstract (full text not accessed — paywalled); described as association + projection |
| Tye et al. 2022, *Limnol. Oceanogr. Lett.* 7:510–519, doi:10.1002/lol2.10274 | Temperature-based models of summerkill across 8,891 lakes (MN, WI) | Abstract: models "accurately predicted contemporary summerkills" — **no number in abstract**; full text returned HTTP 403 (Wiley) and DOAJ 403, so the metric (if any) is **unverified** |
| Phelps et al. 2019, *J. Aquat. Anim. Health* 31:61–70, doi:10.1002/aah.10054 | 225 MN fish kills 2003–2013; environmental-factor models (maximum night-time LST most important) | Not in abstract; full text not accessed — **unverified** |
| DO forecasting (e.g. WRR 2021 Korean urban streams, LSTM) | Forecast DO (a kill driver), not kill events | No (different target) |

## Verdict (provisional)

No published fish-kill **event** model with a held-out AUC (or equivalent) was found. Two candidates
(Tye 2022, Phelps 2019) could contain an in-sample or cross-validated skill metric in their full text,
which could not be retrieved this session. Before any "first-in-class" claim leaves the repo:

1. Obtain and read the full texts of Tye 2022 and Phelps 2019 (library access or author copies);
   record whether they report AUC/TSS and under what validation design. If either reports a held-out
   AUC, KillAlert becomes a "SOTA" comparison instead and must reproduce/beat that design where the
   data allow (their data are lake summerkills in MN/WI — not Iowa streams; a like-for-like comparison
   would need their event data).
2. Repeat the search (Google Scholar / Web of Science) for "fish kill" + "case-crossover" and
   "mass mortality event" + "prediction" with dates after 2024.
3. The claim, if earned, is narrow: *first held-out, case-crossover-evaluated fish-kill risk model for
   Iowa (stream-dominated) kills from observed antecedent weather/flow*, beating every baseline in
   `docs/preregistration/downstream_killalert.md` §6 with CI excluding 0.

## Label search (additional public geocoded fish-kill records), 2026-10-24/25
| Source | Access checked | Verdict |
|---|---|---|
| Delaware DNREC non-tidal fish kills, data.delaware.gov `8pfi-yck7` | Socrata API: 290 rows, lat/long, 2004–2025, licence "Public Domain" | **adopted** (`labels_delaware_fishkill`, 184 events, 142 geocoded) |
| Delaware tidal fish kills `ycvg-4ys9` | 234 rows, **no coordinates** | not used (tidal; would need geocoding) |
| Florida FWC Fish Kill Hotline (since 1995, > 40,000 reports) | web search; FWC ArcGIS `Projects_FWC/FishKills_forDEP_WQDashboard` now returns "service not found"; an ArcGIS Online "Fish_Kill_Report_Hot_Spot_06_17" item exists (hot-spot summary, not events) | no public bulk event export found |
| North Carolina DWR "Fish Kills and Algal Blooms Public Reports" (ArcGIS Online) | 256 features, all 2026 (rolling) | inadequate (only the test-period year) |
| Oklahoma DEQ fish-kill layers (ArcGIS Online) | 63 rows (2021) | too small |
| Minnesota (UMN `Fish_Kill_Lakes`, MN DNR sightings) | lake polygons / citizen sightings, not dated investigated events | inadequate; Phelps et al. 2019 MN data by request only |
| Maryland MDE, Missouri MDC | annual PDF reports only | not machine-readable at scale |
| Wisconsin (Till 2019) | not public (by request) | not used |

ArcGIS Online search used: `title:"fish kill" AND (type:"Feature Service" OR type:"Map Service")`
(42 items, 2026-10-25).

### Label search, round 2 (2026-11-08; lead guidance + §15.5 amendment 2026-11-08)
Searched: Socrata catalog, ~30 ArcGIS Online queries, Dryad, ScienceBase, GitHub, state agency pages
(VA, SC, PA, NJ, TX, KS, KY, OH, IN, IL, MI, NY, WA, OR, CA, LA, GA, AL, FL, NC, UT, MA, CT, NE), plus
Ireland / England / NSW. Zenodo (403) and EDI (search unauthorised) could not be queried.
| Source | Finding | Verdict |
|---|---|---|
| MN DNR (Phelps et al. 2019) + WI DNR (Till et al. 2019) compilations, GitHub `simontye/2020_MME_Temp` raw.zip | ~225 MN (lat/lon) + ~500 WI (WBIC, needs lake join) events 2003–2014, good cause coding | **not adopted**: no licence stated, ends 2014 (adds nothing to the 2018+ test block) |
| Georgia DNR WRD `Fish_Kill` FeatureServer | 211 rows 2002–2016, lat/lon, cause; only 13 non-pollution events since 2010; no licence | inadequate |
| Virginia DEQ PReP export | 117k pollution reports, no coordinates before 2019; server 404 on 2026-11-08 | unverified / inadequate |
| Ireland IFI fish kills (CC BY 4.0) | year-only dates | inadequate (no day) |
| NC DWR historic | PDFs, no coordinates; current layer has no cause field | inadequate |
| Florida FWC | services now token-only | inadequate |
| Utah DEQ incidents, LA LDWF, misc. ArcGIS items | pollution-only / ≤ 14 rows / Iowa copies | inadequate |
Conclusion: Iowa + Delaware remain the only adequate public, licensed, day-dated, geocoded sources;
D3 stays **underpowered** against the temperature-anomaly conditional logistic (v2: wx_clogit − temp_anom
+0.046 [−0.063, 0.148], 52 val sets).
