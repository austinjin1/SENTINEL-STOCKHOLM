# Novelty / incumbent check — D4 TapRisk (2026-11-08)

Searches (2026-11-08, web search): "machine learning predict disinfection byproduct TTHM HAA5 MCL violations
SDWIS community water systems national"; "SDWIS health-based violations prediction random forest
disinfection byproduct violations United States"; Pennino et al. 2020 full text (PMC8204728).

| Work | Task / data | Evaluation | Role for D4 |
|---|---|---|---|
| **Pennino et al. 2020**, *Sci. Total Environ.* 722:137661 (PMC8204728) | Nitrate MCL violation (any, 2013–2017) per NHDPlusV2 catchment (systems located by facility), GW and SW separately; RF on ≈ 210 StreamCat + N surplus + census variables; balanced under-sampling ×10; variable selection by importance | 10-fold CV on pre-balanced data: GW sens 75.8 % / spec 81.5 %; SW PCC 90.3 % (sens 91.9 %, spec 88.6 %); holdout of unbalanced data SW PCC 85.7 %; random (not spatially blocked) | **D4a incumbent** (re-implemented on our units; reproduction check) |
| Scanlon et al. 2022, *Environ. Sci. Technol.* (BEG UT Austin) | Drivers of SDWIS health-based violations 2018–2020 (dominated by DBPR violations), RF on environmental, operational and sociodemographic variables | explanatory driver analysis; no blocked out-of-sample forecast reported in the material found | context for D4b; its variable classes are covered by `pennino_rf_hist` (environmental + system + history) |
| ML THM/HAA studies (Ontario 113 systems 2025; treatment-plant LSBoost 2024; Han River ANN 2018; 2002 utility THM susceptibility) | DBP concentrations at plants / distribution systems from operational data | plant-level R² | not comparable (operational inputs, no national violation labels) |

Conclusion: no published model forecasts SDWIS DBP MCL violations for unseen states/periods with blocked
evaluation; D4b would still not be "first-in-class" without beating every applicable baseline, and Scanlon
2022 shows prior national RF work on these labels, so D4b claims at most "beats Pennino-form / history RF".
D4a's incumbent is Pennino 2020 (re-implemented, with a reproduction check).
