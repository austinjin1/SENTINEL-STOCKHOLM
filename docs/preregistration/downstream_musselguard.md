# D6 MusselGuard — pre-registration protocol (rolling-origin dreissenid HUC12 forecast)

**Status:** v1.0, 2026-10-24, split frozen **before any MusselGuard model was fit** (phase 1 =
validation origin only). Implements RESEARCH_PLAN §8 D6, §14, §15; docs/research/07 §d.6. Code:
`sentinel/models/downstream/musselguard.py`, `scripts/train/downstream_musselguard.py`,
`sentinel/data/sources/wbd_huc12.py` + `scripts/data/download_wbd_huc12.py` (universe), tests
`tests/test_downstream_musselguard.py`.

## 1. Question, incumbent and claim target
Which not-yet-infested HUC12 subwatersheds will record their first zebra/quagga mussel within the next
3 years? **Incumbent:** Kanankege et al. 2018 (co-kriging + boater network, MN waterbodies, AUC 0.78).
**An honest like-for-like comparison is NOT possible:** their unit is the MN waterbody, their
inputs include boater-movement networks that are not public, and their validation design is not
reproducible from the abstract. We therefore (a) report a Minnesota HUC12 stratum for context only,
explicitly *not* as a reproduction, and (b) target **first-in-class** for a national, rolling-origin,
HUC12-level forecast — but the novelty check (`docs/novelty_checks/downstream_musselguard.md`, 2026-10-24) found published
held-out invasion forecasts, so first-in-class is withdrawn (see §10).

## 2. Data and labels
- **Labels:** USGS NAS dreissenid occurrences (`labels_nas_dreissena`, retrieved 2026-10-22), group
  `dreissena_any` (zebra + quagga + *Dreissena* sp.), `first_year_any` per HUC12 (earliest record of any
  status; sensitivity: `first_year_actual`, established-only).
- **Universe:** 87,028 CONUS WBD HUC12s (HUC2 01–18; `processed/wbd_huc12`, pulled 2026-10-24 from WBD
  MapServer layer 6; manifest `wbd_huc12`). Centroids are from generalised geometry (≈ 1–2 km).
- **Presumed negatives:** candidates without a record by T + 3. NAS is presence-only and
  effort-driven; undetected infestations are label noise and depress precision.
- The MN DNR infested-waters list is not used as a label here (no coordinates and no lake universe
  locally; it is a current snapshot without history).

## 3. Units, origins and decision time
- Candidate = universe HUC12 not infested at origin T (first_year_any ≤ T). Positive = first record in
  (T, T + 3].
- **Origins (frozen):** train T = 2006, 2009, 2012 (label windows end ≤ 2015 = the val origin); **val
  T = 2015** (2016–2018); **test T = 2018 (2019–2021) and T = 2021 (2022–2024)** — the last window ends
  2 years before the 2026 NAS vintage (ingestion lag rule).
- Decision time = 31 Dec of T. Features use only records dated ≤ T (unit-tested: moving a post-T record
  leaves every feature unchanged). Caveat: the NAS snapshot is from 2026, so records dated ≤ T may have
  been *entered* after T (retrospective literature); this vintage effect cannot be removed and is
  stated with every result.

## 4. Features (observed only; no upstream-model outputs in phase 1)
Distance to nearest NAS record location dated ≤ T (all countries, so Canadian Great-Lakes records act
as sources); counts of source locations within 25/50/100/200 km; nearest/recent (T−3, T] spread;
WBD `tohuc` flow connectivity (hops from an infested HUC12 upstream; hops to one downstream; ≤ 50
hops); centroid lat/lon and area; optional WQP specific-conductance / pH medians per HUC8 over
[T−9, T] (calcium proxies; the harmonised WQP table has no calcium variable) — used only after the
`wqp_discrete` rebuild manifest is rewritten.

## 5. Splits (frozen) and resampling unit
- Frozen file `docs/preregistration/downstream_musselguard_split.json`,
  **sha256 `04daee8a36e240443a8f711d8b8286c4db2f0e9cd2b9377f0b5d963eeb66dd46`** (origin roles +
  HUC2 spatial folds + universe sha256).
- Spatial secondary: leave-one-HUC2-out over the train origins.
- Counts after the review fixes (C1, US-state filter), candidates / positives (front ≤ 100 km
  candidates / positives; MN candidates / positives): T=2006 train 82,468 / 190 (22,626 / 120; 2,441 / 12);
  T=2009 train 82,278 / 178 (29,054 / 171; 2,429 / 24); T=2012 train 82,100 / 172 (31,115 / 156;
  2,405 / 47); **T=2015 val 81,928 / 184 (33,637 / 171; 2,358 / 78)**; source
  `$SENTINEL_RUNS/downstream/musselguard/build/build_counts.json`. Test origins are not built in phase 1.
- Bootstrap unit: HUC8 (cluster), B = 2,000.

## 6. Endpoints, models and comparators
- **Primary endpoint:** AUPRC on val/test candidates in the **invasion-front stratum** (nearest record
  ≤ 100 km at T; 63% of T=2006 and 89–93% of later-origin positives) — distance alone makes all-CONUS AUROC trivially high. Secondary:
  AUROC front; AUROC/AUPRC all-CONUS; MN stratum (context, not an incumbent reproduction).
- **Baselines:** distance-only (−distance to nearest record ≤ T); kernel spatial smoother of status at T
  (co-kriging analogue); gravity-style source pressure; location-only GBM (lat, lon, area);
  SC-rule (Whittier-2008 analogue on HUC8 specific conductance classes; Ca itself unavailable — stated
  as a substitution) when WQP is available; propagule-pressure logistic.
- **Challenger:** LightGBM on pressure + connectivity + location (+ WQP proxies when available).
- **Exact comparison for the label (amended at review, §10):** on the test origins (scored once after
  freeze), front-stratum AUPRC of the selected challenger exceeds each baseline (distance-only, kernel
  smoother, gravity, location-only GBM, pressure logistic, SC-rule if available) with paired HUC8-cluster
  bootstrap 95% CIs excluding 0 (Holm across comparisons) at both test origins → "SOTA vs reproducible
  analogues of the published methods" (never "first-in-class"; the incumbent itself is not re-run).

## 7. Tuning parity and configurations
Each learned model gets the same budget: 6 configurations (logistic C ∈ {0.001 … 100}; GBM
num_leaves ∈ {7, 15, 31} × min_child_samples ∈ {50, 300}), selected by 5-fold HUC2-grouped inner CV on
the pooled **train origins only** (criterion AUPRC). Baselines are never weakened; every configuration
is logged via `common.log_config_tried`.

## 8. Phase gate, freezing and test look
`--phase1` runs only after the pre-training review. Test origins are never constructed in phase 1
(`common.check_roles` before building; `common.drop_protected`; `log_metric` refuses test splits). Test
is scored once after `FREEZE_downstream_musselguard.json`.

## 9. Iteration plan
Diagnose (residuals by HUC2, by distance band, MN vs Great Lakes; attribution) → literature search
(gravity/boater models, e.g. Kanankege 2018, Leung/Bossenbroek network models; calcium suitability) →
one change at a time (e.g. lake-density / reservoir features from NHD, road-network proxies for boater
pressure, WQP Ca from the raw WQP table, upstream M2/M6 predicted SC/pH via cross-fit providers) →
validation only. All iterations logged; number of configurations reported with any claim.

## 10. Deviations log
All entries below: **2026-10-24, pre-training review, before any fit** (split file unchanged, sha above;
the universe file is unchanged — the US-state filter is applied in code).
- C1 (critical) leakage fix: source-location bins were averaged over all record years, so post-T records
  moved bin positions (log distance changed in 1,484 rows at T=2015). `load_records()` now returns
  record-level points and `sources_at(records, T)` builds bins only from records dated ≤ T inside
  `build_origin`; regression test with a pre-T and a post-T record in the same bin.
- W8 HUC12s whose `states` list has no US state (CN/MX-only units) are dropped from the universe.
- W9 inner-CV criterion = the primary endpoint (front-stratum AUPRC); GBMs use 3 seeds (0, 1, 2),
  seed-averaged, per-seed values reported.
- W10 Kanankege 2018 full text read (design in `docs/novelty_checks/downstream_musselguard.md`: MN
  waterbodies, fit on status by 2010, validated on 2011–2015, co-kriging with boater centrality and road
  distance, AUC 0.78 chosen on the validation set). Added two baselines with the same 6-configuration
  budget, parameter chosen on the train origins: **kernel spatial smoother** of infestation status at T
  (co-kriging/indicator-kriging analogue, Gaussian bandwidth ∈ {10, 20, 40, 80, 160, 320} km) and a
  **gravity-style** source pressure Σ max(d, 5 km)^-α (α ∈ {1, 1.5, 2, 2.5, 3, 4}). Their log-transformed
  values are also inputs to the logistic and GBM models (same features for both).
- Claim framing: published held-out invasion forecasts exist (Kanankege 2018; Leung et al. 2004), so
  **"first-in-class" is not claimed**; the target label is SOTA-vs-reproducible-analogues or
  competitive/negative. D6 counts toward H-G4 "implemented" but **not** toward "beats its incumbent".
- Final test model: refit on train + val origins (2006–2015) with the frozen hyperparameters/rule
  parameters, scored once on T = 2018 and T = 2021.

## 11. Iteration it2 — re-implemented incumbent (pre-registered 2026-10-25, before any it2 fit)
Lead direction: treat a faithful-as-possible re-implementation of Kanankege et al. 2018 on our HUC12
units as **the incumbent**; the published AUC 0.78 (MN waterbodies) is context only.
- **Model form (theirs → ours):** probability co-kriging of invasion status with secondary variables
  (boater-traffic eigenvector centrality; inverse distance to major road) → **regression-kriging
  analogue**: stage 1 logistic regression of status at T (1 = HUC12 infested by T) on the secondary
  variable; stage 2 local ordinary kriging (exponential covariance, k = 32 neighbours, leave-self-out,
  nugget fixed at 0.1 × residual sill) of the residual; score = stage-1 probability + kriged residual.
  Like theirs, it is fitted to status at T and validated on later invasions.
- **Secondary variable (public proxy):** log10(1 + 2010 census population of counties whose internal
  point is within 50 km) (`census_county`, public domain) — a proxy for both boater origin and road access;
  no public boater-movement or national road-proximity layer at HUC12 is available locally (stated
  limitation). Population is a 2010 snapshot (after the 2006/2009 origins; Kanankege likewise used a 2013
  survey for a 2010 origin).
- **Tuning parity:** range a ∈ {10, 20, 40, 80, 160, 320} km (6 configurations = the challenger's budget),
  selected by front-stratum AUPRC on the train origins.
- **Challengers:** `pressure_gbm` (unchanged from phase 1) and `pressure_pop_gbm` (+ the same population
  features, log_pop 25/50/100 km — so the challenger has the incumbent's information).
- **Primary comparison for the claim:** front-stratum AUPRC, challenger vs `kanankege_reimpl`, paired
  HUC8-cluster bootstrap 95% CI excluding 0, Holm across all baselines (distance, gravity, smoother,
  location GBM, pressure logistic, incumbent). Registry experiments carry `@it2_kanankege`.

### 11a. it2 result (run `phase1_it2_kanankege`, 2026-10-25 02:25–03:15; recorded 2026-11-08)
Front-stratum (≤ 100 km) val AUPRC (prevalence 0.005; HUC8-cluster CIs): `kanankege_reimpl` (selected
a = 320 km) 0.022 [0.014, 0.042]; distance-only 0.021; gravity 0.021; smoother 0.017; location GBM 0.020;
pressure logistic 0.033; **pressure_gbm 0.093 [0.047, 0.156]**; pressure_pop_gbm 0.087. Paired
pressure_gbm − kanankege_reimpl: **+0.071 [0.025, 0.129]**, Holm-significant (all six baselines rejected);
pressure_pop_gbm − incumbent +0.066 [0.019, 0.121]. MN stratum AUROC: incumbent 0.724 (published 0.78 on MN
waterbodies, context only), pressure_gbm 0.851. Population features do not help the challenger.
**Caveat:** the incumbent's train-origin AUPRC rose monotonically with the range and the selection hit
the top of the grid (320 km) → it2b below.

### 11b. it2b — incumbent-tuning sensitivity (pre-registered 2026-11-08, before any it2b fit)
One change vs it2: the incumbent's grid is widened to range a ∈ {10 … 2560 km} (9 values) × neighbourhood
k ∈ {32, 128} × nugget fraction ∈ {0.1, 0.5} = **36 configurations** (6× the challenger's budget; same
selection criterion: front-stratum AUPRC on train origins). Challengers and other baselines unchanged.
The stronger of the it2 / it2b incumbent on validation is the incumbent for the claim. Val predictions are
saved (`val_predictions.parquet`). Registry experiments carry `@it2b_kanankege_grid`.
**Deviation (2026-11-08, before any it2b output existed):** the first it2b launch ran 1 h 40 min without
finishing the kriging stage (k = 128 local systems cost ~60× per solve; estimated > 12 h) and was stopped
by PID; nothing was scored. Grid amended to range a ∈ {10 … 2560} × nugget {0.1, 0.5} at k = 32 (18) +
a ∈ {160 … 2560} at k = 64, nugget 0.1 (5) = **23 configurations** (≈ 4× the challenger's budget).
**it2b result (run `phase1_it2b_kanankege_grid`, 2026-11-08 14:19–15:25):** with 23 incumbent
configurations the train-origin selection moves to an interior point (a = 80 km, k = 32, nugget 0.5;
train front AUPRC 0.0207 vs 0.0199 for it2's a = 320 km). Validation front AUPRC of the incumbent is
unchanged (0.022 [0.014, 0.042]); **pressure_gbm − kanankege_reimpl = +0.071 [0.025, 0.128]**, Holm-
significant against all six baselines (same as it2). The grid-edge caveat is resolved: the incumbent is
insensitive to its kriging hyperparameters here. **Freeze candidate (D6): `pressure_gbm`** (phase-1
configuration); incumbent for the claim = it2b-selected Kanankege re-implementation. Awaiting lead go-ahead.

## 12. Pre-test gate deviations (2026-11-08, before any test look)
- **Shared (all five candidates), pre-test gate 2026-11-08 (lead decisions; written before any test look):**
  (a) **seed rule**: a claim additionally requires the paired gain to be > 0 in each of ≥ 3 seeds (per-seed
  point estimates on the test block, vs every primary comparator); per-seed predictions are saved;
  deterministic challengers (L2 logistic) meet it trivially. (b) every input data file is sha256-pinned in
  `downstream_<app>_data_pins.json` and verified before the claim. (c) predictions (incl. per-seed) are saved
  once before any registry write; after a crash only re-scoring of the saved predictions is allowed
  (`--rescore-from-saved`), never re-fitting. (d) **No git (user instruction):** pre-registration order
  cannot be proven from version control; evidence = registry timestamps, file mtimes and a
  `prereg_snapshot` registry row (sha256 + mtime of protocol, record and test script) appended when the
  FREEZE file is written.
- **Per-origin claim (as §6/§10 state):** T = 2018 and T = 2021 are scored separately (front / CONUS / MN);
  the claim needs CI > 0 at **both** origins; Holm across all baselines × 2 origins; pooled origins secondary.
- **Incumbent = it2 configuration (a = 320 km, k = 32, nugget 0.1)**, by the written §11b rule ("stronger of
  it2/it2b on validation": it2 0.0215797 vs it2b 0.0215759 front AUPRC). it2b's "interior optimum" is partial
  (nugget 0.5 is the top of a 2-value grid; k = 64 tried only with nugget 0.1); validation AUPRC is flat.
- **SC-rule baseline (Whittier-2008 analogue, §6):** its condition (rewritten `wqp_discrete` manifest) was met
  on 2026-10-25, but every build ran without WQP — an unlogged omission. It is run on validation now
  (`downstream_musselguard_scrule.py`, origin 2015, HUC8 specific-conductance median from WQP ≤ T) and added
  to the test Holm family; the pressure GBM is unchanged (WQP features are not among its inputs).
- **H-G4 counting resolved by the original §10 text:** D6 counts toward H-G4 "implemented" but **not** toward
  "beats its incumbent" (the Kanankege model is a re-implementation on our units; §11's lead direction makes
  it the comparator for the D6 claim but does not override §10's H-G4 rule).
- `census_county` added to `DATA_MANIFESTS`. **Label completeness:** the NAS 2026-10-22 snapshot has no
  entry dates, so completeness of 2022–2024 first detections (labels of T = 2021) cannot be checked without
  peeking; under-reporting lowers power and, with the both-origins rule, raises the chance of "competitive".
- **SC-rule result (run `musselguard/it2_scrule`, 2026-11-08):** validation origin 2015, front stratum: SC rule
  AUPRC 0.0057 (coverage 98.4 % of front candidates; HUC8s without WQP conductance are unscored), i.e. at
  prevalence; pressure_gbm − SC rule +0.088 [0.044, 0.149] (SC-covered rows). The SC rule joins the test Holm
  family (7 baselines × 2 origins = 14 comparisons).
- **H-G4 consequence (stated 2026-11-08, before any test look; lead step 2):** from the five pending looks,
  H-G4 "beats its incumbent" can count only **D1 (with D1n as its national track), D2 and D5** — at most
  three applications — so **H-G4 cannot be met by these looks alone**; a fourth needs D4 TapRisk (started,
  `downstream_taprisk.md`) or D7. A D6 test result is reported with its own label but is not counted toward
  "beats its incumbent".

## 13. Single test look — result (2026-11-08 20:40–22:12; FREEZE_downstream_musselguard.json; registry `musselguard/test_look`)
Front-stratum AUPRC. **T = 2018** (n = 36,450, 102 positives): pressure_gbm 0.061 [0.030, 0.109] vs incumbent
(Kanankege re-impl) 0.017; Δ +0.044 [0.014, 0.085]; all 7 baselines Holm-rejected (Δ +0.037 … +0.058).
**T = 2021** (n = 38,390, 66 positives): pressure_gbm 0.018 [0.008, 0.044] vs incumbent 0.007; Δ +0.011
[0.001, 0.035] (p 0.020) — only the SC rule is Holm-rejected (m = 14); vs distance-only +0.005 [−0.032, 0.032],
vs pressure logistic +0.006 [−0.008, 0.026]. Seed rule holds. **Both-origins rule fails → label:
competitive** (strong at T = 2018, not at T = 2021, where labels for 2022–2024 may be incomplete — §12 W3).
Pooled origins (secondary): 0.034 vs incumbent 0.011. Not counted toward H-G4 "beats incumbent" (§12).
Caveat (post-results review): T = 2021 has only 66 front positives in 38,390 rows.
