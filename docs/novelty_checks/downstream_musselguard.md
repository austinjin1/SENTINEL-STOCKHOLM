# Novelty check — D6 MusselGuard (dreissenid invasion forecasting)

**Date of search:** 2026-10-24 (UTC). **Searcher:** Downstream workstream (AI-assisted; Bryan Cheng).
**Question:** Do published forecasts of new zebra/quagga mussel invasions exist that were validated on
later (held-out) invasions? If yes, "first-in-class" is not available and MusselGuard must be framed
as a comparison (SOTA / competitive / negative) against reproducible analogues of those methods.

## Searches and reads
| # | Query / source | Result |
|---|---|---|
| 1 | Full text of Kanankege et al. 2018, *Front. Vet. Sci.* 4:231, doi:10.3389/fvets.2017.00231 (frontiersin.org, read 2026-10-24) | Design recorded below |
| 2 | `zebra mussel spread forecast gravity model boater validation Bossenbroek Leung predicted new invasions lakes AUC` | Leung, Drake & Lodge 2004 (*Ecology*, "Predicting invasions: propagule pressure and the gravity of Allee effects"); Leung et al. 2006 (*Biol. Invasions*, gravity models of boater movement); Bossenbroek, Lodge & Finnoff 2007 (ESA abstract, continental gravity forecast over 210 watersheds); a Texas gravity + habitat model; McGarrity & McMahon 2024 (*MBI*) critique |

Bibliographic details of the gravity-model papers come from search snippets and must pass the
bibliography triple check before citation (Bossenbroek, Kraft & Nekola 2001, *Ecol. Appl.*, gravity
models for inland-lake invasion, is known to the author but was **not re-verified** this session).

## Prior held-out forecasts (they exist)
| Work | Unit / data | Validation | Reported skill |
|---|---|---|---|
| **Kanankege et al. 2018** (full text read) | 18,411 MN waterbody centroids; 125 ZM-infested by end of 2015 | Probability co-kriging of invasion status + boater-traffic eigenvector centrality (2013 survey, 1,376 waterbodies) + inverse distance to major road; **fit on status by 2010, validated on invasions reported 2011–2015** ("true validation"); 5-fold CV also reported | ZM true-validation **AUC 0.78** (sens 0.78 / spec 0.72 at risk rank 3); CV AUC 0.73. Variables were chosen by the highest validation AUC, and the 2013 boater survey post-dates the 2010 origin, so 0.78 is optimistic |
| **Leung, Drake & Lodge 2004** (per search snippet) | 1,589 lakes (Great Lakes region), gravity-model boater pressure + Allee effect | Fit on 1992–1996 presence/absence, **validated on 1997–2001 invasions** | "up to nine times as many invasions as a null (random) model" (no AUC in snippet) |
| Bossenbroek et al. (2001; 2007 abstract) | inland lakes / 210 CONUS watersheds | gravity-model forecasts of spread | not verified |

## Verdict
- **"First-in-class" is NOT available** for dreissenid invasion forecasting: rolling/temporal-holdout
  forecasts exist (Kanankege 2018; Leung 2004). Re-scoping to "first national HUC12 forecast" only to
  escape these would violate RESEARCH_PLAN §15.5 amendment rule 3, so we do not claim it.
- **Like-for-like incumbent reproduction is not possible** with public data: Kanankege's unit (MN
  waterbodies), MN DNR presence history and boater-movement network are not available to us; Leung's
  gravity model needs boater survey data. We implement **analogues** under our identical split instead:
  (i) a kernel spatial smoother of infestation status at T (indicator-kriging / co-kriging analogue
  without the boater covariate), (ii) a gravity-style source-pressure baseline Σ max(d, 5 km)^-α over
  NAS record locations dated ≤ T (α tuned on train origins), (iii) distance-to-nearest-source.
- The strongest honest label MusselGuard can earn is **"SOTA vs reproducible analogues of the
  published methods"** (§15.1 requires the strongest published method re-run on our split; we can re-run
  only analogues, so the label must say so) or "competitive"/"negative".
- **H-G4:** D6 counts toward "≥ 6 of D1–D8 implemented with real labels and blocked evaluation", but it
  should **not** be counted toward "≥ 4 beat their incumbent", because the incumbent (AUC 0.78, MN
  waterbodies) cannot be compared like-for-like. The MN HUC12 stratum is reported as context only.
