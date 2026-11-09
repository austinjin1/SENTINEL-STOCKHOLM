# Novelty / incumbent check — D2 ToxinWatch (microcystin > 8 µg/L at lakes)

**Date:** 2026-10-24 (WebSearch "standard" mode). **Purpose:** choose the incumbent for D2
(protocol `docs/preregistration/downstream_toxinwatch.md`).

## Queries and hits
1. "microcystin exceedance forecast lake machine learning lab toxin samples spatiotemporal model" — the
   search tool returned an error (no results); not retried with the same wording.
2. "cyanobacteria harmful algal bloom toxin advisory prediction lakes gradient boosting Washington state"
   - Cappelletti et al., doi:10.1002/wer.70473 — gradient boosting (4 algorithms × 12 resampling schemes) for
     near-real-time cyanobacterial **alert levels** from routine water measurements (study area not in snippet).
   - Llewellyn, WWU MSc thesis (cedar.wwu.edu/wwuet/82) — bloom status of 50 NW Washington lakes from water
     chemistry (2007–2009; multivariate statistics, not toxin forecasting).
   - Heinlen et al. (UW Tacoma, 2024 poster) — Sentinel-2 likelihood of toxin-producing blooms in WA lakes.
   - NALMS 2025 article — national cyanoHAB forecasting; Bayesian (INLA) weekly forecast of the WHO 12 µg/L
     chl-a alert level for 2,192 satellite-resolved lakes (Schaeffer et al. 2024, already in docs/research/07).
   - WA DOH guidance values (8 µg/L microcystins, 1 anatoxin-a, 15 cylindrospermopsin, 75 saxitoxin).

## Conclusion
- ML/statistical forecasting of cyanobacterial alerts and blooms is published (gradient boosting, INLA
  spatiotemporal models); "first-in-class" is **not** sought.
- The strongest published operational class for lake-level HAB forecasting is the INLA spatiotemporal model
  (Schaeffer et al. 2024), whose labels are satellite chl-a/cyano alerts, so the published model cannot be
  re-run on lab toxin labels. Its model class is **re-implemented under our split** (`st_glmm_incumbent`:
  ridge-shrunk lake intercept + EB lake effect + spatial-neighbourhood term + seasonal harmonics + AR(1)-type
  state + the same weather windows), tuned with the same budget as the challengers. A gradient-boosting
  challenger (the Cappelletti-style class) is the other comparator.
- Open: verify Cappelletti et al. metadata and study area; a direct INLA (R-INLA SPDE) fit on our split would
  be a stronger reproduction if R-INLA can be installed.
