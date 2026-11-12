# M5-F — comparator table (settings fixed before the fits they govern)

- **First written:** 2026-11-10T23:31Z (18:31 EST) by the M5 engineer (AI-assisted), under lead rulings B1 and B2
  of `bioassess_f_lead_rulings.md` (sha256 `8d96e3c0…b3bf`, stamp row `f87159c7dc1d`).
- **State of the data when written:** no French request sent; no French file on disk; no M5-F fit of any
  reference or arm run on any rows, UK rows included.
- **Rule of this file (B1):** a line is written before the first fit it governs and is not changed afterwards.
  New lines are appended in §9 with their own date. The path and sha256 of this file are reported to the lead
  at the first writing and each time a line is added. This file changes no rule of the stamped protocol or of
  the rulings; where it differs from them, they win.
- **Sources of the settings:** `configs/bioassess/baselines.yaml`
  (sha256 `bfc4e86b99f045540cdaf5c5af18cac6db3fb3cd1288a3534df3559f8059011a`) and the M5-X feature rules of
  `sentinel/models/bioassess/xdev.py` (`XFeaturizer`, `stratified_subsample`, `_fit_learner`). No setting is
  tuned on French rows.

## 1. Shared rules (every comparator of §8 items 1, 2, 5–9)

| Item | Setting |
|---|---|
| Taxon unit | Harmonised taxa (`mixed_otu`), per modality |
| Taxon features | Hellinger values of every taxon present in ≥ 10 labelled training rows of the fold (per modality). No cap for RF, LightGBM, WA-PLS2, kNN-MAT. TabPFN: §4 |
| Count descriptors (community block) | log10 total count and log10 number of taxa of the sample, per modality (M5-X rule) |
| [community] | Invertebrate taxon features + invertebrate count descriptors |
| [community + context] | [community] + the closed context list of §9.1: day of year (sine, cosine), latitude, longitude, altitude with its missing flag, source one-hot, field-protocol one-hot per modality, diatom offset in days. F-Z: the two named input sets of lead ruling (a) (`ftrack.FZ_INPUT_SETS`) |
| Missing numeric inputs | Median of the training rows of the fold |
| Second assemblage (variant) | Diatom taxon features + diatom count descriptors. Rows without an attached diatom sample: missing (NaN) for TabPFN and LightGBM; 0 plus one indicator column for RF and the linear models |
| Diatom-only training rows (variant, item 6 only) | Invertebrate block missing by the same rule |
| RF | `n_estimators` 300, `min_samples_leaf` 3, `max_features` 0.3 (yaml `rf`) |
| LightGBM | 600 trees, learning rate 0.03, 31 leaves, `min_child_samples` 20, `colsample_bytree` 0.3, `subsample` 0.8, `subsample_freq` 1 (yaml `gbm`) |
| TabPFN v2 | Package `tabpfn` 2.2.1, v2 weights, `n_estimators` 8, `ignore_pretraining_limits=True`, prediction in chunks of 1,000 rows (yaml `tabpfn`) |
| Row budgets | RF, LightGBM, multi-output RF, MEMLM: ≤ 60,000 training rows. TabPFN: ≤ 6,000. Seeded subsample stratified by source × region (`xdev.stratified_subsample`; region = French fold of §4, UK `agency_area`, NRSA region of M5-X), seed = model seed |
| Seeds | Development: seed 0. Test: seeds 0, 1, 2 averaged where the method is stochastic |
| Labels | One fit per target and window on the rows that carry that label. Predictions in label units of §6 |

## 2. Simple baselines (item 1)

| Comparator | Setting |
|---|---|
| Pool mean | Mean of the training labels of the fold |
| Month climatology | Mean of the training labels of the calendar month of the operation; a month with < 10 labelled training rows takes the pool mean |
| Geography-only RF | yaml `rf` settings on (latitude, longitude, altitude); missing altitude = training median |
| Nearest training station | The training station nearest by great-circle distance that has ≥ 1 labelled training row for the target; prediction = mean of that station's training labels |

## 3. Published transfer functions (item 2) and bio.infer (item 3)

| Comparator | Setting |
|---|---|
| WA-PLS2 | 2 components on squared Hellinger values (relative abundance), `baselines.wapls_fit_predict` as in M5-X |
| kNN-MAT | k = 10, distance-weighted, on Hellinger values (yaml `knn_community`) |
| Assemblage variants of both (rulings §2.2) | Invertebrate; diatom-trained; two-assemblage (the two Hellinger blocks side by side, each transformed on its own). A row without the needed diatom sample takes the invertebrate variant's prediction |
| MEMLM (reimplemented) | **Owed line** (§9): members and stacking rule from the paper. Deadline: before the first French development fit of any model. If the paper's description is not sufficient: "not compared, with reason" in §8 |
| bio.infer 1.3.6 | Its own OTU pipeline, as in M5-X (fidelity check of M5-X stands); ITIS coverage of French taxa reported |

## 4. ML baselines (item 6) and the TabPFN taxon cap

| Comparator | Setting |
|---|---|
| RF, LightGBM, TabPFN v2 | §1, each on [community] and [community + context] |
| TabPFN taxon cap, one assemblage | The 300 most prevalent invertebrate taxa among the training rows (yaml `max_taxa`), each in ≥ 10 |
| TabPFN taxon cap, two assemblages | 300 invertebrate + 150 diatom taxa, the most prevalent of each modality, each in ≥ 10 training rows. (Not below the hybrid's 200 + 100.) If the column count of an input set would exceed 500, diatom taxa are dropped from the least prevalent until it is 500; the number dropped is recorded |
| Multi-output RF (Kocev et al. 2013) | One `sklearn` `RandomForestRegressor` with the yaml `rf` settings on the target matrix = the label columns the candidates train on in that pool that have ≥ 50 labelled training rows in the fold, each z-scored with training-row statistics. A missing label is set to 0 (the training mean) for the split search only. Leaf value of a target in a tree = mean of the observed z-scored labels of that target over the training rows of the fit that fall in the leaf; the tree's own leaf value where the leaf holds none. Prediction = mean over trees, back-transformed |
| Auxiliary out-of-fold features (variant of the per-target learners) | Predictions of the auxiliary targets of §6.4 by the same learner and settings: for training rows from 5 folds grouped by station (seed 0), for the rows to predict from the fit on all training rows |
| Pool-prevalence variant | As the parity table of §8. TabPFN takes only the form "out-of-pool taxa set to missing" (the extra-column form would exceed 500 columns) |

## 5. Index regressions (item 5)

| Comparator | Setting |
|---|---|
| Index columns, F-A / F-T | Delivered indices of the operation: 7613; 1000, 5910, 6951 where delivered; 5856 and 1022 of the attached diatom operation. A missing index = training median plus one indicator column per index. France-only training pool (the UK has no such columns) |
| Index columns, F-Z | Computed by our code from harmonised taxa on both countries: richness, EPT richness, EPT share, Shannon |
| Linear inverse calibration | Ordinary least squares on the index columns |
| RF | yaml `rf` settings on the index columns |
| With geography | The same two models with latitude, longitude and altitude added |

## 6. Ensembles (item 7), hybrid controls (item 8), region controls (item 9), events

| Comparator | Setting |
|---|---|
| Baseline ensembles | Unweighted mean of the members' predictions, per target. "Three strongest single baselines": the three item-6 baselines (in their chosen variants) with the highest pooled R² on the development pseudo-test for that target, seed 0 |
| Hybrid controls | `pfn_hybrid` with retrieval `random` and `knn`; PCA(32) of the Hellinger features (the hybrid's 200 + 100 taxa) fitted on training rows in place of the learned embedding; the query grouping, budget and TabPFN call of §9.2 |
| Region descriptors | §8 item 9 and the mirror rule of §9.1, unchanged. If there are fewer than 21 training pseudo-regions, the PCA has (number of pseudo-regions − 1) components and the remaining descriptor columns are 0 |
| Region-offset model | Ridge regression (`sklearn` `Ridge`, α = 10, intercept not penalised) of the pseudo-region mean residual of the strongest baseline ensemble on the 20 descriptors, standardised with training pseudo-region statistics. The residuals of the training rows are out-of-fold: 5 folds grouped by pseudo-region, seed 0. One model per target |
| Event comparators | Threshold-on-baseline with a normal predictive distribution whose spread is the SD of that baseline's development residuals for the target; prevalence and month climatology of the training rows (`fevents.py`) |

## 7. Fixed baseline members of #6 `ens_f` and #8 `ens_zs` (ruling B2)

One recipe for every target. Not chosen on development. Not the comparator variants of §1–§6.

| Item | Setting |
|---|---|
| Inputs (both members) | Invertebrate Hellinger features + attached-diatom Hellinger features + the context columns of §9.1 (#8: without latitude and longitude). **Nothing else:** no count descriptors, no region descriptors, no pool-prevalence columns, no auxiliary out-of-fold features |
| Training rows | #6: the candidate's own pool for the track. #8: the UK allowed rows |
| RF member | yaml `rf`; every taxon in ≥ 10 training rows; ≤ 60,000 rows (§1 subsample); no diatom sample: 0 plus one indicator column; missing numeric context: training median |
| TabPFN v2 member | yaml `tabpfn`; taxon cap 300 + 150 (§4, with the 500-column rule); ≤ 6,000 context rows (§1 subsample); one plain fit per target and fold (no retrieval, no query grouping); no diatom sample: missing |
| Ensemble | Unweighted mean of the three members, per target |

## 8. Not compared (technical inability only; written before any candidate validation number)

| Comparator | Reason | Recorded |
|---|---|---|
| GJAM | Failed the pre-declared fidelity check in M5-X (protocol §8 item 4); not re-attempted | Protocol v1, 2026-11-10 |

## 9. Lines added after the first writing (dated; nothing above is edited)

| Date (UTC) | Line | Deadline it met |
|---|---|---|
| — | MEMLM members and stacking rule: **owed** | Before the first French development fit of any model |
| 2026-11-10T23:51Z | **Region-offset model, second variant (lead ruling of 2026-11-10, evening, point (c)).** As the α = 10 line of §6 (same descriptors, standardisation, out-of-fold residuals, one model per target), but α is chosen per target, inside each fold, by leave-one-pseudo-region-out cross-validation on the training pseudo-regions of that fold over the grid {0.1, 1, 10, 100, 1000}; the α with the lowest mean squared error of the left-out pseudo-region mean residual is taken (a tie takes the larger α). The α = 10 line stays. For gate G7 and for every comparison that names the region-offset baseline ensemble, the comparator is the variant with the higher mean development pooled R²; both are reported | Before any fit of any M5-F reference or arm (none had run) |
| 2026-11-10T23:51Z | **UK development design (§9.4): what is fitted and how (engineer).** (1) Folds: the 19 UK `agency_area` values that have pseudo-test rows (`pool_fit` invertebrate rows of 2016–2020 with ≥ 1 F-Z primary label). Training rows of a fold: allowed UK rows dated ≤ 2015-12-31, outside the area, more than 1 km from every pseudo-test station of the area and sharing no `location_id` with one. Targets: the six F-Z primary targets (`__pm30d`). Seed 0. (2) Invertebrate comparators train on the rows that carry an invertebrate list. ML comparators: [community] + the numeric columns of the named input set (`coords`: day-of-year sine and cosine, latitude, longitude, altitude with its missing flag; `nocoords`: the same without latitude and longitude); the source and field-protocol one-hots are not used in this single-source design. Subsample strata: source × `agency_area`. (3) Computed indices (§5, F-Z): richness = number of harmonised taxa of the sample; EPT richness = number of those in the orders Ephemeroptera, Plecoptera, Trichoptera; EPT share = EPT richness / richness; Shannon = natural-log Shannon index of the relative abundances. Missing value = training median. (4) B2 members of #8 (§7): trained on the rows of #7 in the fold (invertebrate rows and diatom-only rows; a diatom-only row has its invertebrate block missing by the rule of §1); inputs = the literal list of §7 with the source and field-protocol one-hots and the diatom offset. (5) #7 `ma_zs`: `bioset_ma.recipe("ma_zs")` unchanged; inner validation = 15 % of the fold's training rows in 50-km blocks; chunk length = a fixed number of epochs set from one timing run on UK training rows that prints no validation number. Region context of a pseudo-test row: the mirror rule of §9.1 over the allowed invertebrate rows of the held-out area dated 2016–2020 (inputs only). No-region ablation: the same weights with the region slot masked. (6) #8 `ens_zs` = unweighted mean of `ma_zs`, the RF member and the TabPFN member, per target. (7) A prediction that is not finite takes the fold's training mean for scoring; the count is reported. (8) **UK reference set** (stored for every fold before any UK pseudo-test number of #7 or #8 is computed): pool mean, month climatology, geography-only RF, nearest station, WA-PLS2 and kNN-MAT (invertebrate variants), the four index regressions, TabPFN v2 on [community], on `coords` and on `nocoords`, RF on `nocoords`. Run afterwards if CPU time allows, and reported as such: RF on [community] and on `coords`, LightGBM on the three input sets, the baseline ensembles of these. Not run on the UK design (they belong to the French development design): MEMLM, bio.infer, multi-output RF, auxiliary and pool-prevalence variants, hybrid controls, region controls, event comparators | Before the first UK development fit of any reference, of #7, of #8 or of a member of #8 (none had run) |
| 2026-11-11T00:00Z | **UK development design: the "if CPU time allows" items are fixed (lead ruling of 2026-11-10, evening, second message, point 2).** IN, added to the UK reference set of the line above: LightGBM on `nocoords`; the two baseline ensembles on `nocoords` (RF + TabPFN, and RF + LightGBM + TabPFN: unweighted means of stored predictions, no new fit). OUT, not run on the UK design: RF on [community]; RF on `coords`; LightGBM on [community] and on `coords`; the baseline ensembles on `coords`. Reason: CPU is the limiting resource (one heavy CPU job of ≤ 6 threads) and these comparators are chosen on the French development design, not here. This is not revisited | Before any UK reference score was computed or read (the `table` stage had not been run; no stored prediction had been compared with a label) |
