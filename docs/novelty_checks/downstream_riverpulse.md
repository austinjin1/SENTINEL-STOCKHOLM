# Novelty / incumbent check — D8 RiverPulse (daily GPP/ER at unsensed reaches)

**Date:** 2026-10-24 (searches run by the downstream workstream fork, WebSearch "standard" mode).
**Purpose:** choose the incumbent for D8 (RESEARCH_PLAN §15.1; protocol `docs/preregistration/downstream_riverpulse.md`).

## Queries and hits
1. "predicting stream gross primary production unmonitored rivers machine learning random forest Appling 356 rivers"
   - Dryad doi:10.5061/dryad.c2fqz61kx — "River metabolism in the contiguous United States: Random forest model code, inputs and outputs" (inputs largely Appling et al. 2018 + MERIT Hydro reach variables; GPP for > 175,000 rivers).
   - Segatto, Battin & Bertuzzo 2021, *Ecosystems* 24:1792–1809, doi:10.1007/s10021-021-00618-8 (RF from 12 sensor reaches → 292 reaches, Ybbs network).
2. "continental scale river metabolism random forest GPP prediction contiguous United States Dryad c2fqz61kx paper"
   - The Dryad package accompanies **Maavara et al. 2025, *Science*, "River metabolism in the contiguous United States: A West of extremes"** (per search-result summaries; bibliographic details **not yet verified** against Crossref).
   - Appling et al. 2018, *Sci. Data* 5:180292 (the label set).

## Conclusion
- A published method exists for exactly this task class: **random forests trained on Appling et al. 2018 site estimates to predict metabolism at unmonitored reaches** (Maavara et al. 2025; Segatto et al. 2021 at network scale). "First-in-class" is therefore **not** available for D8.
- No held-out daily-GPP skill number under a basin-blocked split was found in the search snippets (unverified; the Maavara notebooks/paper were not opened). The incumbent is therefore **re-implemented under our split**: a random forest on catchment attributes + weather + season (`rf_segatto_incumbent`; StreamCat attributes in place of MERIT Hydro reach variables), tuned with the same 4-configuration budget as the challenger. Target claim: "SOTA" per §15.1 against this re-implementation and the strongest simple baseline.
- **Update 2026-10-25.** Crossref (api.crossref.org/works/10.1126/science.adu9843): Maavara, Yuan, Johnson,
  Zhang, Aho, Brinkerhoff, Logozzo, Raymond, "River metabolism in the contiguous United States: A West of
  extremes", *Science* 390:622–627 (2025-11-06). Dryad (doi:10.5061/dryad.c2fqz61kx, version 395397) lists
  `random_forest_CONUS_GPP.ipynb`, `appling_data_06202024.csv`, MERIT Hydro catchments; the Dryad page states
  25 features (7 in a simplified model), scikit-learn 1.4.2 RandomForestRegressor with tuning. File downloads
  returned HTTP 401/403 without a Dryad login, so the exact features, validation design and skill numbers
  were **not** obtained; no published held-out number is quoted. A Maavara-style RF (25 features of the
  stated classes) is added as a second re-implemented incumbent (`rf_maavara_incumbent`).
