# M5 BioAssess — results of the single test pass

This text was appended to `bioassess_protocol.md` as §11 on 2026-11-09 05:33 EST, after the pass. It is moved here verbatim on 2026-11-09 by the lead, so the frozen protocol stays byte-identical to its hash in `bioassess_freeze.json` (710a5956…). See addendum A5.


## 11. Results of the single test pass (appended 2026-11-09; append-only — §1–§10 as frozen)
*This section was appended after the pass. The frozen protocol (sha256 in `bioassess_freeze.json`) is the
version in the freeze archive `bioassess_freeze_20261108T220227.tar.gz`; this appendix changes no rule.
Corrections to the frozen record are in `bioassess_post_freeze_addendum.md` (A1–A3).*

- **Pass:** one authorised pass, 2026-11-08 22:30 – 2026-11-09 04:45 EST, run dir
  `$SENTINEL_RUNS/bioassess/final_20261108T220242/` (no resume; 253 units fitted); 2,180 registry rows,
  experiment `bioassess_test`; independent post-results review 🟡 PASS WITH CONCERNS
  (REVIEW_REPORT_POST_RESULTS.md, "M5 BioAssess — post-results review"), exact re-score 2,155/2,155 A/B/C rows.
- **Confirmatory (Test A = leave-one-ecoregion-out × cycle-out, NRSA 2018-19), labels as scoped by the review:**

  | Target | Ensemble C − per-target comparator (Δ R², 95 % location-cluster CI) | Holm p (18) | Seeds | Ecoregion-unit CI | RMSE | Label |
  |---|---|---|---|---|---|---|
  | Chloride | +0.021 [+0.013, +0.028] | < 1e-4 | 3/3 + | [+0.001, +0.036] | −2.0 % | **SOTA as ensemble C, LOEO scope** |
  | pH | +0.024 [+0.015, +0.032] | < 1e-4 | 3/3 + | [+0.014, +0.033] | −2.2 % | **SOTA as ensemble C, LOEO scope** |
  | Conductivity | +0.013 [+0.006, +0.019] | 0.006 | 3/3 + | [+0.004, +0.022] | −1.5 % | **SOTA as ensemble C, LOEO scope** |
  | TN | +0.011 [+0.004, +0.017] | 0.021 | 3/3 + | [−0.013, +0.030] | −1.1 % | **competitive** (SOTA by the pre-registered rule; not robust: 4/9 ecoregions positive, WMT +0.18) |
  | TP | +0.005 [−0.003, +0.013] | 1 | mixed | [−0.013, +0.025] | −0.4 % | competitive |
  | Turbidity | +0.012 [+0.002, +0.022] | 0.26 | mixed | [−0.009, +0.037] | −1.0 % | competitive |

  C = mean(BioSet+geography, RF[community+geography], TabPFNv2[community]). **BioSet alone (candidate A) is
  significantly worse than the comparator on TN (−0.059 [−0.081, −0.037]), conductivity (−0.038), chloride
  (−0.031) and pH (−0.036).** C beats its leave-BioSet-out control on 6/6 targets and its equal-size control
  on 5/6 (TP CI includes 0): BioSet adds signal to the strongest baselines.
- **Test B (cycle-out, same regions, NRSA 2018-19; secondary):** C − comparator ≈ 0 (TN −0.003, conductivity
  +0.001, chloride −0.001, pH +0.006; all CIs include 0); `RandomForest[geography]` (lat/lon/elevation only) is
  the best of all 21 models on TN (0.680 vs C 0.619), conductivity (0.740 vs 0.712) and chloride (0.706 vs
  0.642). The community signal helps most **out of region**.
- **Test C (LOEO × cycle-out, NRSA 2023-24; secondary replication; see A1/A2 — D5 scored 2023-24 ~1.5 h
  before the M5 freeze, no M5 decision used it; Test C shares 598 locations with Test A):** C − comparator
  chloride +0.018, pH +0.024, conductivity +0.012, TN +0.012 (CIs > 0).
- **H-M5 (BioSet A vs blocked RF[community], Test A, Holm over 6): met, 6/6 rejected** (Δ +0.045 to +0.133,
  seed-averaged prediction); only **4/6** have every seed positive (TP, turbidity mixed); Test C (secondary,
  Holm over 6) 5/6, chloride not rejected with all three seeds negative.
- **T1/T2 transfer (secondary, deviation §10):** negative (T1 R² −0.32 to +0.07; T2 only TP > 0).
- **Post hoc (A3, non-confirmatory):** bio.infer-style ML inference Test A R² −0.17 to −1.27; C beats it on
  6/6 targets (all CIs > 0) → labels unchanged.
- p-values reported as 0 by the bootstrap are p < 1e-4 (B = 10,000).
- **Correction (2026-11-09, appended):** the post-hoc bio.infer bullet above is withdrawn as support (addendum A4): our bio.infer-style re-implementation performed poorly; it may not reflect the EPA software; the SOTA label rests on the pre-registered comparators (WA-PLS2, kNN-MAT, RF, TabPFN, ensembles); its development LOEO sanity check (NRSA 2008-14): R² −0.29 to −1.60, far below WA-PLS2 (0.18–0.30; registry e.g. `9ec21e854822`, `62b4cb574e36`) and kNN-MAT (0.13–0.31).
