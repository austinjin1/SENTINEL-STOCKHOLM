# M5-X — SENTINEL-BioAssess larger / harder follow-on track (pre-registration)

**Status:** PRE-REGISTERED, **v3 (2026-11-09)**. Revised after the pre-registration review ("M5-X —
pre-registration review": C-MX-1/2, W-MX-1..10), the re-review ("M5-X — re-review": C-MXR-1/2, W-MXR-1..3) and the
lead's item-7 and M2 rulings. **No model has been fitted or selected on track-X data.** v1 (snapshot
`ced6d0538314`) and v2 (snapshot `47130a305f69`) are superseded; the split *file* is unchanged. Approved by the lead 2026-11-09 under RESEARCH_PLAN
§15.5 items 5–7.

**Relation to M5:** this track **never replaces** the frozen M5 result (`bioassess_protocol.md`, test pass
2026-11-09, RESULTS.md). The frozen M5 numbers are always reported next to track-X numbers, and every
track-X claim names its block ("M5-X, X-L" or "M5-X, X-U").

## 1. Why a harder track
M5 is not saturated by the ceiling test (best Test A R² 0.41–0.57), but the frozen SOTA margins are small
(+0.011 to +0.024 R², 1–2 % RMSE). They hold only for an ensemble and only under leave-one-ecoregion-out, and a
geography-only RF wins once a region has been sampled before. Track X asks whether the community signal is
real and transferable on **larger, previously unscored data**:
- **X-L:** lakes (a different ecosystem).
- **X-U:** a different continent's national network (England).

Track X scores within-region skill as the a-priori hard stratum.

## 2. Data, prior exposure and the item-7 ruling
There is no new download. Sources are under existing manifests `nars_streams`, `nars_lakes` and `uk_ea_ecology`
(licences/QC there). One software environment has been added: `rbioinfer` (R 4.3 + CRAN `bio.infer` 1.3.6,
1.2 GB; loads headless — Tk unavailable, a warning only).

**Item-7 ruling (lead, 2026-11-09).** RESEARCH_PLAN §15.5 item 7 applies as written: **no row of any frozen test
block of any workstream enters any M5-X training pool.** It is implemented by `xtrack.frozen_test_keys()` from
the frozen split files and applied by `xtrack.training_rows()`.

| Excluded from every training pool | Rows | Source of the rule |
|---|---|---|
| NRSA 2018-19 and 2023-24 (M5 Tests A/B/C; D5 validation/test cycles) | 3,866 | `nrsa1819`, `nrsa2324` |
| UK samples at M5 T2 `reserved_test` sites (all years) | 67,279 | `uk.py::uk_site_roles` |
| NLA visits at Fuse tier-C `test_region` / `test_spatial` sites | 2,211 | `fuse_tier_c_split.json` (sha256 `e2fd5de9…`) |
| NLA visits at D2n NLA sites in D2n test HUC4s | 149 | `downstream_d2n_microcystin_split.json` |
| NLA 2017 visits that are the same sampling events (unique_id, date) as M2-orbit test-role rows (lead ruling: item 7 covers any workstream's registered test rows; turbidity is a shared target) | 102 not already covered (218 visits in total) | M2's own `orbit.split.assign` on `s2_wqp_matchups_hist` NARS_WQX-NLA rows (metadata only, no cache written) |
| **Either M5-X block's test or excluded rows** in the other block's pools (C-MXR-1; e.g. the 974 X-L test visits and 1 excluded NLA 2022 visit were in every X-U pool in v1/v2) | all | `xtrack.training_rows` (cross-block rule) |

Other frozen splits were checked: D5 (NRSA 2018-19 / 2023-24, covered above), D1n, M7-Lite (`test_n` chips share no
NLA sampling event with the pools), M6, Fuse tiers D/W, Sim (descriptive thresholds only). None has further test
rows on NRSA, NLA or EA invertebrate samples. No other workstream's model code reads `uk_ea_ecology`.

**Prior exposure that remains (disclosed, W-MX-10).**
- **X-L test rows (NLA 2022):** 549 of them lie at Fuse/D2n test sites (corrected in amendment v11 item 4: the count is 459). They are used only as M5-X *test* rows,
  never for training. Fuse used NLA chemistry and benthic summaries as microcystin *inputs*; no workstream has
  scored lake chemistry as a target.
- **X-U test rows (2021–26 `pool_fit` samples):** their labels were training targets in M5's
  `bioassess_phase1_transfer_dev` and in the M5 UK pretraining (`uk_pretrain_20261022T091937`). Neither was ever
  scored on them. That pretraining informed M5's "transfer is negative" conclusion.
- **Candidate families:** chosen from the M5 *test* outcome (the ensemble form that won).
- **Split builder:** it loads target values through the shared loaders but uses only availability masks
  (`notna()`).

## 3. Test blocks (frozen split table)
**Split table:** `$SENTINEL_RUNS/bioassess/x_track/splits_x_v1.parquet`. **Unchanged from v1.**
- 583,766 rows; sha256 `18a35ef3ba9ddd87a5be2e8c4ff1a18d29bea8869318f940b150b1306feee459`; content sha256
  `59ab19bfcf0b3d4d4d7c6eb977d4d31c8620da269630d8d82923ae39d52cb3fa`.
- Built by `sentinel/models/bioassess/xtrack.py::make_splits`; the rebuild is bit-identical, and
  `validate_splits()` returns `[]`.

**X-L — lakes (NLA 2022; leave-one-ecoregion-out × cycle-out).**
- **Test:** 974 NLA 2022 first visits with a benthic sample and at least 1 target. 973 location clusters (1 km);
  folds are the 9 aggregated ecoregions (67–165 visits each).
- **Cycle-out:** training contains no NLA 2022 row. NRSA rows after 2022 are frozen-test rows and excluded anyway.

**X-U — England (EA; leave-one-area-out × time-out).**
- **Test:** 8,121 invertebrate samples dated 2021–2026 at `pool_fit` sites with at least 1 primary target and a
  known area. 2,149 location clusters (3.78 samples per cluster, temporal repeats); folds are the 19 EA agency
  areas (80–949 samples each).
- **Excluded by the time-out rule (W-MX-2):** all other UK samples dated 2021–2026, 10,315 in total:
  - 4,361 at `reserved_test` sites;
  - 771 at `pool_val` sites;
  - 5,183 `pool_fit` samples without any primary target or area.

  `pool_val` samples dated ≤ 2020 (development-scored by M5, not a frozen test) remain in training.

**Training pool of fold g (`xtrack.training_rows`; review C-MX-2):**
1. Pool rows of the block. Same-continent sources (NLA + NRSA for X-L; UK for X-U) are taken outside region g;
   the other continent's rows are region-free.
2. Minus every test/excluded row of **either** M5-X block (cross-block rule) and every frozen-test row (§2).
3. **Minus every row within 1 km of any fold-g test location, any source and any region.** It also removes any
   row in a 1-km cluster that contains a fold-g test location. 1-km clusters span EA areas, and the same site
   can appear under two ids (minimum 17 m apart).

Removed by rule 3: X-U 0–1,882 rows per fold (2,234 in total); X-L 1 row.

**Resulting per-fold training pools** (`$SENTINEL_RUNS/bioassess/x_track/pools_x_v1.json`, recomputed for v3;
`validate_training_pools()` = `[]`, which also checks the cross-block rule):

| Block | Total | NLA | NRSA | UK |
|---|---|---|---|---|
| X-L | 203,318–203,621 | 1,310–1,503 | 3,372–3,638 | 198,484 |
| X-U | 184,830–200,835 | 1,597 | 3,958 | 179,275–195,280 |

## 4. Targets
- **X-L (as NRSA):** log10 TP, log10 TN, log10 conductivity, log10 chloride, log10 turbidity, lab pH.
- **X-U (EA matched ±30 d medians, `uk.py::UK_TARGETS`):** pH, log10 conductivity at 25 °C, log10 chloride,
  log10 orthophosphate-P, log10 ammoniacal-N, log10 nitrate-N. TP and TON are secondary.

## 5. Metrics; hard stratum
- **Primary: within-region skill `r2_anomaly`** (anomalies from region means; region = ag_eco9 / EA agency
  area). Fixed a priori.
- **Secondary:** pooled R², median per-region R², RMSE, Spearman.
- **CIs:** percentile location-cluster bootstrap; B = 2,000 marginal and B = 10,000 confirmatory paired.
- **Sensitivity:** 50-km-block and region-unit paired CIs. With 9 and 19 units these are coarse, so the per-region
  sign counts are reported alongside.

## 6. Comparators (mandatory; identical training rows per fold)
1. **Simple:**
   - region mean;
   - geography-only RF (X-L: lat, lon, elevation; X-U: lat, lon, site altitude);
   - nearest reference site (X-L only).
2. **Published transfer functions:** WA-PLS2 and kNN-MAT. Each chooses, on development with the same budget,
   between the multi-source pool and the same-continent pool (W-MX-7 ii).
3. **bio.infer** (EPA CRAN package 1.3.6, run itself in `rbioinfer`; `get.taxonomic` → `get.otu` →
   `taxon.env` (linear + quadratic) → `mlsolve`). For X-U the ITIS OTU coverage of UK taxa is reported.
   - **Fidelity check (W-MX-5):** reproduce the package's own documented example on its bundled EMAP-West data
     (`bcnt.emapw`, `envdata.emapw`) headless, and confirm that `taxon.env`/`mlsolve` run without our code paths.
     It must reproduce the example's documented inferences (and, where the documentation gives no number, the
     inference–observation agreement reported in Yuan 2007, *J. Stat. Softw.* 22(3)).
   - If the software runs faithfully, its predictions **stay eligible** for the per-target strongest comparator
     whatever their rank. Weak results are reported as "compared, weak", never "not compared".
   - It is excluded only if the software itself cannot be run faithfully, and the reason is recorded.
4. **GJAM-style joint-species inverse prediction** (Clark et al. 2017, CRAN `gjam`; W-MX-6).
   - Fitted per fold on ≤ 5,000 training rows × the 100 most prevalent taxa (fold-training prevalence).
   - Out-of-sample inverse prediction of each target maximises the joint likelihood under the fitted
     coefficients and residual covariance.
   - Fidelity check: on training rows our routine must reproduce `gjam`'s own in-sample inverse predictions
     (`xpredMu`).
   - Reported as "not compared, with reason" only if it cannot run within 2 CPU-hours per fold.
5. **UK pressure-index baselines (X-U only; W-MX-6).**
   - Regression of each target on the EA-computed indices of the same sample: WHPT-ASPT, WHPT-NTAXA,
     PSI (family score), E-PSI (family score), LIFE (family index), BMWP-ASPT. These are 99–100 % available
     and derived from the taxa only.
   - Fitted as linear inverse calibration and as RF, each with and without geography.
   - TRPI is **not in the EA dataset**. It is added only if its published taxon scores are obtained under
     licence before the freeze; otherwise it is excluded for that reason.
6. **ML baselines:** RF[community], LightGBM[community], TabPFNv2[community], RF[community+geography],
   RF[community+context].
7. **Baseline ensembles:** RF[c+g]+TabPFN, RF[c+g]+LGBM+TabPFN, RF[context]+TabPFN.

**Row budgets (W-MX-7 iii):** TabPFN ≤ 6,000 training rows. RF/LightGBM ≤ 60,000. Both use a seeded subsample
stratified by source × region. WA-PLS2/kNN-MAT/bio.infer use all pool rows of their chosen pool. BioSet uses
all rows.

**Context features (W-MX-7 i), explicitly:**
- **X-L:** lat, lon, elevation, sampling day-of-year (sin, cos).
- **X-U:** lat, lon, `site_altitude_m`, `site_slope_m_per_km`, `site_dist_from_source_km`, `site_width_m`,
  `site_depth_cm`, `site_discharge_category`, substrate % (`site_boulders_cobbles_pct`, `site_pebbles_gravel_pct`,
  `site_sand_pct`, `site_silt_clay_pct`), sampling day-of-year.
- **Excluded as chemistry:** `site_alkalinity_mg_per_l_caco3`, `site_conductivity_us_per_cm`,
  `site_total_hardness_mg_per_l_caco3`, `site_calcium_mg_per_l` and any other chemistry column.

**Per-target comparator:** the strongest of all the above by development `r2_anomaly` (§7). A comparator that is
strongest on any target is never dropped.

## 7. SENTINEL candidates and development (selection only on training-pool rows)
**Development designs.** Every M5-X fit calls `xtrack.assert_fit_ids` on its fit and validation ids. It refuses
any test/excluded row of either block and any frozen-test row (unit test
`test_xtrack_fit_id_guard_refuses_protected_rows`). Development pseudo-tests are drawn only from pool rows, so
frozen-test rows (incl. the M2 events in NLA 2017) are neither fitted nor scored in development.
- **X-L:** pseudo-test NLA 2017 under LOEO × cycle-out. Training: NLA 2007/12 + NRSA + UK fold pool rows.
- **X-U:** pseudo-test UK 2016–2020 `pool_fit` samples under leave-one-area-out × time-out. Training: UK ≤ 2015
  + NRSA + NLA.
- The same frozen-test, region and 1-km rules apply. The UK 2016–20 pseudo-test is the stated proxy for the
  taxonomic-resolution shift (W-MX-8).

**No reuse of any M5 checkpoint (C-MX-1, C-MXR-2).**
- **Single choke point:** every M5-X checkpoint load goes through `xtrack.load_pretrained_encoder`, which:
  - refuses `"auto"` and the M5 pointer `uk_pretrain_latest.json`;
  - refuses the M5 UK encoder (sha256 `91453a22…`, pretrained on all `pool_fit` samples incl. the X-U test labels);
  - refuses any checkpoint without a `.train_ids.json` sidecar;
  - refuses any checkpoint whose recorded training ids are **not a subset of that fold's training pool**
    (`training_rows(S, block, g, frozen)`). That excludes area-g, 1-km-buffered, frozen-test and other-block
    rows, so e.g. an all-UK-≤ 2020 encoder is refused;
  - only then loads via `cv.PretrainedEncoder`.
- **`resolve_pretrained`:** importing `xtrack` marks the process `SENTINEL_TRACK=M5-X`, and
  `scripts/train/bioassess_cv.py::resolve_pretrained` then refuses (so `"auto"` can never resolve to the M5
  encoder in an M5-X process).
- **Sidecar provenance:** the per-fold pretraining code writes the sidecar from the ids it actually used
  (`xtrack.write_train_ids_sidecar`).
- **Unit test through the real loaders:**
  `test_xtrack_encoder_loader_refuses_auto_m5_and_out_of_pool_encoders`.
- UK pretraining, if used, is redone **per fold**:
  - **X-U:** on UK ≤ 2020 rows outside area g, with the 1-km and frozen rules;
  - **X-L:** on the X-L pool's UK rows;
  - its vocabulary comes from the same rows, and early stopping uses training-pool rows only.

**Per-fold fitting (W-MX-9).** Everything below is fitted on fold-g training rows only, in both the development
and test designs (M5's W1 rule):
- taxon vocabularies, `min_name_count` filters, community feature selection;
- feature and target scalers;
- bio.infer `taxon.env` / OTU tables and the GJAM fits.

**Taxonomic-resolution shift (W-MX-8).** The UK `analysis_method` share of ANLA rises from 51 % (≤ 2010) to 67 %
(2011–15), 83 % (2016–20) and 93 % (2021+ test). ANLE is about 39 % ≤ 2010 and about 0 % after. Handling:
- `analysis_method` (ANLA / ANLE / ANAA / other) is an allowed input for every UK model: a global token for
  BioSet, one-hot for baselines;
- X-U results are also reported separately for ANLA and non-ANLA samples.

**Candidate families.** BioSet (set transformer over harmonised taxa) on the multi-source pool with a source
token, with or without per-fold UK pretraining and with or without geography/context. Plus the ensemble form that
won M5: BioSet(+geography) averaged with the strongest baselines.

**Budget and selection.**
- At most **12 SENTINEL configurations** across both blocks, all counted.
- Baselines get the same data and the same tuning budget (never weakened).
- Selection rule: highest mean development `r2_anomaly` over the 6 targets per block.
- One confirmatory candidate per block (C-X); up to 2 more are secondary.

**Compute.** GPU jobs ≤ 2 h on GPUs 0/1 via the lock helper, chunked per fold. CPU ≤ 6 threads.

## 8. Confirmatory family, decision table, power
**Family.**
- C-X vs the per-target comparator on `r2_anomaly`, 2 blocks × 6 targets = **12 tests**, Holm at α = 0.05.
- One-sided rule: p = two-sided paired location-cluster bootstrap p (B = 10,000) if Δ > 0, else 1.
- "CI" below means the 95 % paired location-cluster CI of Δ.
- **MDE_X** is the block's pre-registered Holm-worst MDE for that target (table below; for the X-U targets
  without an NRSA proxy, the X-U maximum 0.024).

**Decision table** (per block × target; the first matching row applies — exhaustive and non-overlapping,
W-MX-3):

| # | Condition | Label |
|---|---|---|
| 1 | Holm-rejected ∧ per-seed Δ > 0 for seeds 0, 1, 2 ∧ region-unit paired CI > 0 | **SOTA (M5-X, block, within-region)** |
| 2 | Holm-rejected (row 1 not met) | **Above, not robust** (which condition failed is stated) |
| 3 | CI upper bound < 0 | **Below** |
| 4 | CI includes 0 ∧ realised Holm-worst MDE > 0.02 | **Inconclusive (underpowered)** |
| 5 | otherwise (not Holm-rejected, and row 4 not met) | **Competitive** |

**Row 4 terms (W-MXR-2):**
- Realised Holm-worst MDE = 3.71 × SE_realised, where SE_realised = (paired CI width) / 3.92.
- 0.02 is the pre-declared smallest effect of interest: about the size of M5's robust chloride/pH gains.
- A null is therefore "Inconclusive" when the realised data could not have detected a 0.02 within-region gain, and
  "Competitive" when they could.

**Secondary (unadjusted, labelled as such):** pooled R²; other candidates; SENTINEL vs geography-only RF;
cross-ecosystem X-L (streams-only training → lakes); X-U by `analysis_method`.

**Power check** (`scripts/eval/bioassess_x_power.py`; `power_x_v1.json`).
- Method: the empirical SE of the paired Δ `r2_anomaly` (ensemble C − comparator) on frozen M5 Test A (1,901
  clusters), scaled by √(1,901 / n_clusters).
- The test sets are unchanged by the v2 pool rules, so the MDEs are unchanged (re-run gives identical output).
- MDE at 80 % power:

  | Block | α = 0.05 | Holm worst case (α/12) |
  |---|---|---|
  | X-L | 0.017–0.027 | 0.023–0.036 |
  | X-U | 0.012–0.018 | 0.015–0.024 |

- **Wording (W-MX-4).** X-U is powered for within-region effects ≥ about 0.015–0.024, i.e. effects the size of
  M5's chloride/pH/turbidity gains (+0.025 to +0.029). It is **not** powered for effects the size of M5's TN, TP
  or conductivity gains (+0.005, +0.010, +0.012). X-L is powered only for effects ≥ about 0.023–0.036.
- **Caveats:**
  - The SE proxy comes from NRSA streams, with about 1 sample per cluster. X-U has 3.78 samples per cluster
    (temporal repeats).
  - The X-U targets orthophosphate, ammonia and nitrate have no proxy.
  - The realised CI widths are reported and feed row 4 of the decision table.
  - The v3 pool changes do not touch any test row, so the MDEs are unchanged.

## 9. Freeze and single pass
Same machinery as M5 (`bioassess_final.py` pattern):
- a freeze-candidate record hashing code, configs, this protocol, the split table, `pools_x_v1.json`, the frozen
  split files of §2 and the data;
- an independent pre-test review; a lead-created `bioassess_x_freeze.json`;
- one authorised pass with per-unit prediction caching, scoring only after all predictions exist, and one
  registry flush (experiment `bioassess_x_test`);
- the pre-registered single-resume rule.

## 10. Amendments
(dated, append-only)
- **2026-11-09 v2 (before any fit):**
  - C-MX-1: per-fold UK pretraining only, plus the encoder guard and its test.
  - C-MX-2: 1-km buffer and cluster rule in `training_rows`, `validate_training_pools`, and the test.
  - Item-7 ruling: frozen-test rows of M5, D5, Fuse and D2n removed from every pool; pools recomputed.
  - W-MX-1..10: exclusion description, exhaustive decision table, power wording, bio.infer fidelity check,
    GJAM, UK index baselines, context features, `analysis_method`, per-fold fitting, disclosures.

  Registry snapshot `bioassess_x/prereg_snapshot` v2 supersedes `ced6d0538314`.
- **2026-11-09 v3 (before any fit):**
  - C-MXR-1: cross-block rule (no X-L test/excluded row in any X-U pool and vice versa) in `training_rows`, plus
    the `validate_training_pools` check and a unit test.
  - C-MXR-2: `load_pretrained_encoder` choke point, the subset rule, the `resolve_pretrained` refusal under M5-X,
    sidecar writer, and unit tests through the real loaders; the `assert_fit_ids` loader guard is built.
  - Lead M2 ruling: 102 NLA 2017 visits added to the frozen-test exclusions.
  - W-MXR-2: decision-table row 4 uses the realised MDE vs a 0.02 smallest effect of interest.
  - W-MXR-3: pools recomputed.

  Snapshot v3 supersedes `47130a305f69`.
- **2026-11-09 v4 — implementation details fixed before any track-X fit (no model fitted, no development
  result seen; written by the M5-X follow-on engineer).** These close points the protocol leaves open; none
  touches a test row, the split file or the confirmatory family.
  1. **Development designs (§7) — exact rules.** Dev fold g uses the same regions as the test design.
     - X-L: pseudo-test = allowed NLA 2017 first visits in ecoregion g with ≥ 1 X-L target. Training =
       `training_rows(S, "X-L", g, frozen)` (the fold-g *test-design* pool) minus every NLA 2017 row (cycle-out)
       minus every row within 1 km of, or in the 1-km cluster of, a fold-g pseudo-test location.
     - X-U: pseudo-test = allowed UK `pool_fit` samples dated 2016–2020 in area g with ≥ 1 X-U primary target.
       Training = `training_rows(S, "X-U", g, frozen)` minus every UK row dated ≥ 2016 (time-out) minus the same
       1-km / cluster rule.
     - Intersecting with the test-design pool makes every development fit (incl. per-fold pretraining) legal
       under the test design, so the encoder guard applies unchanged. Fold id lists are written once
       (`x_track/dev/folds_dev_v1.json`, sha256 logged).
  2. **Multi-source labels.** Three targets are measured on both continents and are harmonised:
     pH (`ph_lab` ↔ `uk_ph`), log10 conductivity (`log10_cond_us_cm` ↔ `uk_log10_cond25_us_cm`), log10 chloride
     (`log10_chloride_mg_l` ↔ `uk_log10_chloride_mg_l`). For these, a model's training rows are either
     *same-continent* (rows of the block's continent) or *multi-source* (all pool rows with the harmonised label,
     plus a source indicator). Other targets are same-continent only. Every pooled community comparator
     (WA-PLS2, kNN-MAT, RF/LightGBM/TabPFN variants) is run in both variants for the shared targets, and the
     variant is chosen per (block, target) by development `r2_anomaly` (§6 item 2 extended to all of them, so no
     comparator is weakened). bio.infer, GJAM and the UK index baselines are same-continent regional
     calibrations (single variant).
  3. **Seeds in development.** Development fits use seed 0 for every stochastic model (baselines and SENTINEL);
     selection and per-target comparators are decided on seed 0. The test pass uses seeds 0, 1, 2 (§8 unchanged).
  4. **Tabular features** (fit on the fold's labelled training rows of the variant): Hellinger relative
     abundances of harmonised taxa (`mixed_otu`) present in ≥ 10 such rows (M5 setting; TabPFN: the 300 most
     prevalent), the two count descriptors, a source one-hot (multi-source variant) and an `analysis_method`
     one-hot (ANLA / ANLE / ANAA / other; 0 for US rows). Geography = lat, lon, elevation (UK: site altitude).
     Context = §6 lists. Missing numeric inputs are filled with training-row medians. Hyperparameters = M5's
     fixed `configs/bioassess/baselines.yaml`. Row budgets (§6): seeded subsample stratified by source × region,
     proportional allocation, per target over its labelled rows.
  5. **Region mean** under leave-region-out is the same-continent training mean (r2_anomaly = 0 by
     construction). **Nearest reference** (X-L) = value at the nearest US training row with reference class R.
  6. **UK index baselines (X-U):** the six EA indices + `analysis_method` one-hot; missing index values filled
     with training medians; ordinary least squares ("linear inverse calibration") and RF (M5 RF settings), each
     with and without geography (lat, lon, site altitude). TRPI: not obtained → excluded for that reason.
  7. **bio.infer (run itself).** `get.taxonomic` once on the list of raw taxon names (name-only ITIS lookup;
     no sample or label). Headless, its two interactive prompts are answered by fixed defaults through
     `assignInNamespace`: no manual name correction (`correct.taxanames` returns the names unchanged) and the
     first listed ITIS candidate for each ambiguous name (`get.dupe.sel`). Per fold: `get.otu` → `taxon.env(~x +
     I(x^2))` per target on the fold's same-continent labelled training rows → for the pseudo-test rows `get.otu`
     → `makess` → `mlsolve`. A sample with no modelled taxon gets the training mean (counted and reported).
  8. **GJAM.** Per fold: ≤ 5,000 same-continent labelled training rows (seeded, stratified by region) × the 100
     most prevalent `mixed_otu` taxa (presence/absence, `typeNames = "PA"`); covariates = the block's 6 targets
     standardised with training statistics (missing values imputed by gjam); `ng = 2000`, `burnin = 500`.
     Out-of-sample inverse prediction: Gibbs sampler with the posterior-mean coefficients and correlation matrix,
     alternating latent w | y, x (truncated normal) and x | w (Gaussian, truncated to the training range), 1,000
     iterations after 200 burn-in; prediction = posterior mean of x. **Fidelity criterion (fixed now):** on the
     training rows of the first fold of each block, our routine vs gjam's in-sample `xpredMu` must give Pearson
     r ≥ 0.90 for every target with a non-degenerate in-sample prediction; otherwise GJAM is "not compared,
     with reason".
  9. **SENTINEL configurations (§7 budget; counted per block, ≤ 6 per block, ≤ 12 in total).** BioSet =
     M5's set transformer with flat embedding (the M5 winner) trained multi-task on all multi-source labelled rows
     of the fold (9 output columns: the union of both blocks' targets, shared ones merged), a source token and an
     `analysis_method` token added as set elements; inner early-stopping validation from training locations (M5
     rule). Per block: (a) `bsx_flat_geo`; (b) `bsx_flat_geo_rloss` (+ within-region anomaly loss, weight 1);
     (c) `bsx_flat_geo_ukpt` (+ per-fold UK pretraining, M5 pretraining objective and settings; X-U: UK training
     rows of the fold; X-L: the fold pool's UK rows; checkpoint loaded only through
     `xtrack.load_pretrained_encoder`); (d) `bsx_flat_geo_same` (same-continent rows only); (e) Ensemble
     C-form = mean(best of a–d, RF[community+geography], TabPFNv2[community]); (f) Ensemble = mean(best of a–d,
     the two strongest baselines by mean development `r2_anomaly`). Selection (§7): highest mean development
     `r2_anomaly` over the block's 6 targets among (a)–(f).
  10. **Development statistics.** `r2_anomaly` with region = ag_eco9 (X-L) / EA area (X-U) on the pooled
      out-of-fold predictions; 95 % percentile location-cluster bootstrap, B = 2,000. Experiment
      `bioassess_x_dev`.
- **2026-11-09 v5 — implementation notes (before any development score was computed).**
  1. bio.infer `taxon.env` builds its site × taxon matrix with a per-site `match` loop and its ROC area with an
     n_present × n_absent matrix; on X-U folds (~40,000 rows) that is infeasible (days per fold; GBs per taxon).
     `scripts/eval/bioassess_x_bioinfer.R::taxon.env.fast` is the package function with those two loops vectorised
     (`rowsum`; `findInterval` count of absent < present). Verified **bit-identical** (taxa, coefficients max |Δ| = 0,
     ROC, x-limits) on the package's Oregon example and on a development fold
     (`x_track/dev/r_out/bioinfer/bioinfer_taxon_env_fast_equivalence.txt`). ITIS lookup: 2,358 of 2,609 names
     matched; unmatched names drop out, as in interactive use without manual correction.
  2. The frozen M5 inner-validation splitter requires one region per 1-km location cluster; 24 UK clusters span EA
     areas. For BioSet's inner early-stopping split, the region-anomaly loss and per-fold pretraining only, each
     cluster takes its first region (`xbioset.samples_view`); folds and scoring keep the real region.
  3. Test rows are loaded with the design rules of `xdata` (NLA: first visit with benthic taxa; UK: ≥ 3 harmonised
     taxa); loaded vs split-table counts are written to `pass_info.json` of the single pass.
- **2026-11-09 v6 — GJAM fidelity check: executed result and corrected comparison (before any GJAM development
  prediction was scored).** The v4.8 check as executed (X-L fold 0, 4,474 training rows, gjam fit 86.7 min) compared
  our *out-of-sample* routine with gjam's in-sample `xpredMu`: r = 0.43 (TP), 0.52 (TN), 0.54 (chloride), NA for
  conductivity, turbidity and pH → **failed** (`x_track/dev/r_out/gjam/X-L/fold00_v4check_info.txt`). Diagnosis
  from the gjam 2.7 source: inside its Gibbs loop `Y <- w`, i.e. `xpredMu` = E[x | w] with the latent w sampled
  given the observed y **and the observed x**; an out-of-sample routine cannot condition on x, so the two are not
  expected to agree. Corrected check (fixed now): our routine run in gjam's in-sample scheme on the same training
  rows (w | y, observed x; then x | w) must give r ≥ 0.90 vs `xpredMu` for every target with finite `xpredMu`
  (pairwise-complete rows); the out-of-sample-mode correlations are reported alongside. Run on fold 0 of both
  blocks. If it fails, GJAM is "not compared, with reason". This change can only add a comparator (it never favours
  SENTINEL); it is disclosed as made after the first check's outcome was seen. (For completeness: an unlogged
  progress scoring of the finished X-L same-continent baselines and five X-U baselines was run at 07:38; no GJAM,
  bio.infer or SENTINEL prediction existed then, and nothing in v4–v6 depends on it.)
- **2026-11-09 v7 — single-look enforcement (lead instruction after the M7 review; no science change).** The M5-X
  scorer follows `sentinel/models/hydromet/testlook.py` (`sentinel/models/bioassess/xtestlook.py`): it accepts only
  the lead-written `bioassess_x_freeze.json` (never the candidate), checks `scoring_code_sha256`, file/package hashes
  and content-hash data pins (manifest sha + exact loader files) before the claim; registry refusal; O_EXCL
  `TEST_LOOK.json` + claim-time registry row; test rows loadable only under that open claim; predictions sealed once
  (0444, `SHA256SUMS`, O_EXCL `PREDICTIONS_SAVED.json`); O_EXCL `SCORED.json` after scoring. The single resume
  (§9) is allowed only before sealing; `--rescore-from-saved` after sealing.
- **2026-11-09 v8 — secondary EVENT TRACK (lead instruction: every model outputs (A) water quality and (B)
  harmful-event detection; `docs/plans/event_layer_plan.md` M5 rows and §4). Written before any event prediction or
  event metric was computed; only event label counts of the development pseudo-tests were looked at (below).**
  M5-X is a chronic-stressor diagnosis (invertebrates integrate weeks–months); the track is **secondary**, never part
  of the 12-test confirmatory family, scored inside the same single M5-X look. The original M5 test blocks are spent
  and are not used.
  1. **Events** (criterion-anchored, definitions from the shared `sentinel/eval/events.py`, imported, never copied):
     E2a chloride > 230 mg/L; E5b TP > 0.1 mg/L; E6 pH < 6.5 or > 9.0 (lab pH); E3 (snapshot variant of E3a)
     turbidity > 50 NTU. X-L: E2a, E5b, E6, E3. X-U: E2a, E6 (TP and turbidity are not X-U targets → n/a).
  2. **Power (plan §1.2: ≥ 30 positives and ≥ 10 clusters with positives) on the development pseudo-tests:**
     X-L E5b 105 / 105 (F), E6 46 / 46 (F), E2a 11 / 11 (U), E3 11 / 11 (U); X-U E2a 19 / 7 (U), E6 35 / 25 (would
     pass on M5-X dev, but **descriptive only by lead ruling**: the plan's UK validation counts are 4 and 10
     positives). U events and both X-U events are **descriptive only** (derived route, no direct head, no tests).
  3. **Derived route (primary):** P(threshold exceeded) from the M5-X predictive distribution = Gaussian on the
     model's (log-)target with mean = the model's seed-mean prediction and SD = its out-of-fold residual SD on the
     development folds (per block × target); one Platt recalibration per (block, model, event) fitted on development
     out-of-fold probabilities only (cross-fitted by development fold for the development numbers; fitted on all
     development folds and frozen for the test look). Applied to the confirmatory candidate C-X of the block.
  4. **Direct route (challenger), only for powered events (X-L E5b, E6):** LightGBM binary classifier on the
     tabular community features (M5 fixed GBM hyperparameters, same-continent training rows of the fold, seeds 0–2
     at test), recalibrated like every route.
  5. **Comparators:** threshold-on-baseline = derived route applied to RF[geography] and TabPFNv2[community]
     (their own residual SD + recalibration); prevalence (fold training prevalence) and climatology (fold training
     prevalence by sampling month, add-one smoothing, fallback prevalence).
  6. **Metrics:** `events.state_metrics` (AUPRC with prevalence as chance level, Brier, BSS vs climatology, ECE)
     and paired ΔBrier / ΔAUPRC by `events.compare_routes`, location-cluster bootstrap B = 2,000; contrasts:
     derived(C-X) − each comparator, derived − direct. Holm across the powered events of a block for derived vs
     direct (plan §2.2); everything else unadjusted, labelled secondary.
  7. **Test-block capacity:** test labels are not read before the look; the candidate reports projected positives
     (development prevalence × test n). Inside the look, the plan's power rule is re-applied on the test rows and any
     event below it is reported descriptively.
  8. **Code:** adapter `sentinel/events/bioassess_x.py` (plan §4.1 location), development runner
     `scripts/eval/bioassess_x_events.py`; frozen development calibration data
     `x_track/dev/events/events_calibration_v1.parquet` (hashed in the candidate).
- **2026-11-09 12:36 v8a — correction to v8.2 (lead correcting an earlier instruction that conflicted with the
  pre-registered power rule; made before any event prediction, probability or metric existed — no
  `x_track/dev/events/` output and no `bioassess_x_events_dev` registry row at the time of writing).** The lead's
  earlier "UK pH underpowered, descriptive only" was based on the plan's preliminary UK validation counts and was not a
  ruling meant to override the rule. The pre-registered rule (≥ 30 positives and ≥ 10 clusters with positives on the
  development pseudo-test) governs: **X-U E6 (35 positives / 25 clusters) is powered** — secondary, with the direct
  LightGBM head, exactly like X-L E5b and E6, and included in the derived-vs-direct Holm family of its block. X-U E2a
  (19 / 7), X-L E2a (11 / 11) and X-L E3 (11 / 11) remain descriptive only by the rule. Everything else in v8 is
  unchanged (approved by the lead).
- **2026-11-09 v7a — single-look hardening (lead, M7-review lessons; no science change).** (1) `scoring_code_sha256`
  hashes the full static `sentinel` import closure of the scorer, the M5-X dev/event code and `events.py` (incl.
  `hazards.py`, `ew_scoring.py`, `early_warning.py` and the frozen M5 modules) plus the R scripts; a subprocess test
  runs the event path and fails if any loaded module is unhashed. (2) Sealing writes a registry row with every sealed
  file's sha256 and `check_sealed` verifies against it, refusing unlisted/missing/changed files; `--rescore-from-saved`
  is refused once any scoring row exists; test labels needed for scoring are written into the sealed folder and
  test-row loading is refused from sealing on (a deleted marker cannot reopen it: the registry's sealing/scoring rows
  are checked); every listed artifact (split, pools, power, folds, selection, event calibration/dev results, TabPFN
  weights) is verified before the claim; any event-track error is recorded as "not run" (file + registry row) and
  never stops the primary look.
- **2026-11-10 v7b — GPU use of the single pass (lead GPU rule update: GPUs 0–9, lock helper with --min-free-mb, never
  a GPU another user occupies, jobs ≤ 2 h; no science change).** The pass process holds no GPU. GPU units (BioSet
  fits, TabPFN) run in worker subprocesses of ≤ 100 min each, launched one at a time through `sentinel.core.gpu`
  with `--min-free-mb 5000` and `SENTINEL_GPUS` narrowed to GPUs with no other user's process; workers read a
  label-free, sha-verified bundle of the test inputs written under the open claim, refuse once the pass is sealed,
  and write into the same sha-verified unit cache. SIGTERM of the pass is a recorded abort.
- **2026-11-10 14:33 EST v9 — every listed comparator is fitted and compared at the test look (lead ruling of
  2026-11-10; written before any test data are seen: no freeze file, no claim, no `bioassess_x_test` registry row
  exists, and no test label has been read).** Until now the scorer fitted at test only the per-target comparator
  (the strongest by development `r2_anomaly`) and RF[geography]. From this entry on:
  1. **Comparators fitted at test.** Every §6 comparator of the block, each in its development-chosen variant per
     (block, target) (v4.2), with seeds 0–2 where the method is stochastic. X-L (14): region mean, RF[geography],
     nearest reference site, WA-PLS2, kNN-MAT, bio.infer, RF[community], LightGBM[community], TabPFNv2[community],
     RF[community+geography], RF[community+context], and the three baseline ensembles of §6 item 7. X-U (17): the
     same without the nearest reference site, plus the four UK-index models (linear and RF, each with and without
     geography). bio.infer is included whatever its development rank (§6 item 3). The record lists each comparator
     with its executable specification; the scorer refuses a record whose list differs from this one.
  2. **Paired interval against each.** For every (block, target) and every listed comparator: Δ = `r2_anomaly`(C-X)
     − `r2_anomaly`(comparator) on the seed-mean predictions, with the 95 % percentile CI of the paired
     location-cluster bootstrap (B = 10,000, the same routine and bootstrap seed as the family test, so the same
     resampled clusters whenever the row sets are equal). Rows: finite label and finite predictions of both models.
  3. **Decision table.** The 12-test Holm family is unchanged (C-X vs the per-target comparator). **Row 1 needs in
     addition: paired CI lower bound > 0 against every listed comparator of the block.** A listed comparator with
     no finite paired estimate counts as not beaten. **Row 2 "Above, not robust" names each listed comparator that
     is not beaten.** Rows 3–5 are unchanged. The added condition needs no multiplicity adjustment: all bounds must
     hold together, so it can remove a SOTA label and can never add one.
  4. **Reporting.** `results/listed_comparators.json` holds every paired estimate (Δ, CI, p, n). A listed comparator
     whose CI lies entirely below 0 (it is above C-X) is named in the reason text of the row, whatever the label.
  5. **Not compared, with reason — GJAM (§6 item 4).** §6 gives one ground ("cannot run within 2 CPU-hours per
     fold"). Amendment v4 item 8, as corrected in v6, added a second ground before any GJAM development prediction
     was scored: the fidelity check (our routine in gjam's in-sample scheme vs gjam's own `xpredMu`, Pearson
     r ≥ 0.90 for every target, fold 0 of both blocks). **The time ground is not the ground used:** the measured
     gjam fits took 99.3 min (X-L fold 0, 4,474 rows) and 114 min (X-U fold 0, 5,000 rows). **The ground used is
     the failed fidelity check:** X-L r = 0.73 (TP), 0.78 (TN), 0.88 (conductivity), 0.84 (chloride), 0.78
     (turbidity), 0.82 (pH); X-U r = 0.91 (pH), 0.89 (conductivity), 0.83 (chloride), 0.92 (ortho-P), 0.91 (NH4-N),
     0.85 (nitrate); `fidelity_pass` FALSE in both blocks (`x_track/dev/r_out/gjam/<block>/fold00_info.txt`). As
     v6 discloses, the corrected check was written after the first check (v4 item 8 as executed) had failed. The
     check was run once per block and is not repeated; gjam is not tuned further. No GJAM prediction is fitted or
     scored on test rows. **Limitation, stated in every report:** one published joint-species method could not be
     reproduced faithfully and is not in the comparison; no claim may say that C-X beats all joint-species models.
     The record and the scorer output carry, per block and target, the in-sample-scheme r, the out-of-sample-mode
     r, the number of rows, and the path and sha256 of the fidelity files.
  6. **Not compared, with reason — TRPI (§6 item 5).** Its published taxon scores were not obtained under licence
     before the freeze. No other comparator is left out.
  7. **What does not change.** The development selection (C-X, secondaries, comparator variants, per-target
     comparators) and the configuration count (12 of 12). No new model is fitted on development. The freeze
     candidate with sha256 a4b077c8…0ff7 is superseded: the development dry run is repeated with the changed
     scorer and a new candidate is written for the pre-test review.
- **2026-11-10 14:37 EST v10 — row 3 "Below" also fires when any listed comparator is significantly above C-X
  (lead ruling of 2026-11-10; written before the label function was edited, before the development dry run of
  the v9 scorer, and before any test value was read: no freeze file, no claim, no `bioassess_x_test` registry row
  exists).** This entry changes only the label function.
  1. **Row 3.** A target is labelled "Below" when the paired 95 % CI of C-X minus the comparator (v9 item 2) has
     its upper bound < 0 against the per-target comparator (as before) **or against any listed comparator of the
     block** (v9 item 1). This is checked before rows 2, 4 and 5 and before the "Not estimable" label: a target is
     labelled "Above, not robust", "Inconclusive (underpowered)" or "Competitive" only if no listed comparator is
     significantly above C-X. Row 1 cannot meet the new trigger, because row 1 needs every lower bound > 0.
  2. **No finite estimate.** A listed comparator with no finite paired estimate still counts as not beaten for
     row 1 (v9 item 3). It cannot fire the new row-3 trigger (it has no CI); the reason text of the row names it
     as "not estimable".
  3. **Reason.** A win needs every listed comparator beaten; a "Competitive" label next to a comparator that is
     significantly better would mislead. The change can only make a label harsher: every label it changes becomes
     "Below". The new trigger uses unadjusted 95 % intervals against up to 17 comparators, so "Below" can arise
     by chance more often than for one comparator; the error is on the cautious side for any claim.
  4. **What does not change.** The 12-test Holm family, rows 1, 2, 4 and 5, the comparator lists, the paired
     estimates, the naming of every listed comparator that is not beaten in the reason text and in
     `results/listed_comparators.json` (v9 items 3–4), the development selection and the configuration count.
     In v9 item 3, "Rows 3–5 are unchanged" now holds for rows 4–5 only.
- **2026-11-10 19:24 EST v11 — pre-test review fixes: rule comparators, chunked label-free prediction phase with
  write-once units, pinned code snapshot, record corrections, two disclosures (lead rulings X-B1..X-B4 of
  2026-11-10 on the pre-test review verdict `freeze_after_fixes`; written before any code named here was changed
  or run, and before any test value was read: no freeze file, no claim, no `bioassess_x_test` registry row
  exists, and no test label has been read).** Earlier text stays in the history; where it differs, this entry
  governs.
  1. **Two comparators added by rule, both blocks, no new fit (X-B3).** For each block, with C-X as selected:
     (i) `cxr_nobioset` = C-X with BioSet removed, i.e. the equal-weight mean of C-X's baseline members;
     (ii) `cxr_swap` = C-X with BioSet replaced by the listed single baseline (not an ensemble, not a member of
     C-X, bio.infer and GJAM excluded as in §7 step 4) that has the highest mean development `r2_anomaly` over the
     block's six targets on seed 0, i.e. the equal-weight mean of C-X's baseline members and that baseline.
     Both are means of predictions that already exist on development. Both join the listed set of v9 item 1
     (X-L 14 → 16, X-U 17 → 19) and both compete for the per-target comparator by the unchanged §6 rule
     (strongest by development `r2_anomaly`). The scorer checks that the two specifications in the record follow
     this rule from the record's own C-X and refuses otherwise. Where a rule comparator has the same members as
     an already listed ensemble (X-L) it is still listed under its rule name.
     **Reason, disclosed:** the reviewer found that no listed X-U comparator was "C-X without BioSet", and
     checked three unlisted alternatives on development (seed 0): mean(RF[community+context], RF[index+geography])
     0.3160, the same plus TabPFN 0.3314, the same plus LightGBM 0.3375, against C-X 0.3469 and the best listed
     comparator 0.3247. The rule above was fixed by the lead after that check. No choice among alternatives is
     made from development deltas: the rule names one swap member by its single-model development score, and that
     member need not be the alternative that looked strongest in the reviewer's check. This is stated in the
     record and in every report.
     **Families.** The confirmatory family stays one test per block × target, C-X vs the per-target comparator
     (12 tests, Holm); the per-target comparator is re-chosen among the enlarged list by the unchanged rule. Row 1
     needs the paired CI lower bound > 0 against every listed comparator, now 16 and 19; row 3 fires against any of
     them. Any Holm family formed over listed comparators (events: derived vs each comparator) grows with the
     list. No denominator is held at its earlier size.
  2. **Prediction phase: label-free, chunked, write-once (X-B1; replaces the single-resume rule of §9, v5 and
     v7b).** The look has four separate commands: claim, chunk (any number), seal, score.
     (1) *Label-free prediction phase.* The claim loads the test rows once, drops every test label inside the
     loader and writes a hashed inputs bundle. Chunks read only that bundle and the training pools; each chunk
     asserts, before any fit, that no test row of its block carries a finite label. No truth file exists before
     sealing. The label loader refuses unless the seal verifies and scoring has begun.
     (2) *Write-once units.* One unit per (block, leaf model, fold, seed) — and per event fit — each in its own
     file with its own sha256, written under a temporary name and moved into place; an existing unit is never
     replaced, and no whole-run manifest is rewritten. Chunks may run in any order, in parallel, on CPU or on
     idle GPUs through the lock helper (≤ 100 min each), and may be started again any number of times before
     sealing. Every chunk re-verifies the freeze record, the code snapshot, the data pins and the artifacts. A
     chunk that fails or is killed leaves finished units in place and does not end the look.
     (3) *Blind between chunks.* A chunk prints and stores unit identities, row counts and wall time only: no
     prediction value or summary. No prediction file is opened by a person between chunks. Code and record cannot
     change between chunks (item 3).
     (4) *Seal once*, only when every planned unit exists and verifies: per-model prediction files are assembled
     from the units, hashed (`SHA256SUMS`, 0444), and one `predictions_sealed` registry row is written.
     (5) *Single-shot scoring.* Labels are loaded only after the seal verifies; all `bioassess_x_test` scoring
     rows are appended to the registry in one locked write after the results are sealed. Scoring is not resumed:
     if it stops before that write, `--rescore-from-saved` recomputes everything from the sealed predictions from
     the start (recorded); once the rows exist, no scoring run can write again.
     Each of the five points has a unit test. An event-track error is still recorded as "not run" (v7).
  3. **Pinned code snapshot (X-B2).** Before the freeze, every pinned code and config file is copied to a
     read-only snapshot under `sentinel_runs/bioassess_x/` (files 0444, directories 0555). The freeze record
     lists each file with its sha256. The look runs from the snapshot; every command verifies the snapshot (exact
     file set and every hash), not the live tree, and refuses in real mode when started from any other tree. The
     freeze file, the candidate record and the registry stay in the live repository. Later edits of the live
     tree neither change nor stop the look; an edit of the snapshot stops it.
  4. **Record corrections.** (a) §2 said 549 X-L test rows lie at Fuse/D2n test sites. The count is **459** (439
     at Fuse tier-C test sites and 20 more at D2n-only test sites; 974 X-L test rows in all); "549" was a digit
     transposition. Counts from sample keys only. (b) `one_time_guards` and `resume_rule` in the record now name
     the markers and registry metrics the code uses (`TEST_LOOK.json`, `PREDICTIONS_SAVED.json`,
     `SCORING_STARTED.json`, `SCORED.json`; `look_claimed`, `predictions_sealed`). (c) The cached training design
     (`x_track/dev/design_v1.pkl`) and frozen-key cache (`x_track/dev/frozen_keys.json`) are pinned by sha256 and
     verified by every command. (d) Every file of the scoring code closure is listed singly in the record.
  5. **Disclosure — the power table of §8 is optimistic.** It is a proxy from NRSA Test A. The reviewer scaled the
     development paired CIs of C-X vs comparator to the test cluster counts: projected test SE 0.008–0.020 on X-L
     (registered 0.006–0.010) and 0.006–0.026 on X-U (registered 0.004–0.0065); projected Holm-worst detectable
     difference above 0.02 for all 12 targets (X-L 0.031–0.073, X-U 0.022–0.098). Most rows whose CI spans 0 are
     therefore expected to read "Inconclusive (underpowered)". The decision table and the 0.02 threshold are not
     changed; row 4 uses the realised CI as before.
  6. **Disclosure and rule — runs are not bit-reproducible.** Two development dry runs of the same record
     differed by up to 4.4e-5 in Δ, 9.7e-5 in CI ends, 1.5e-4 in per-seed Δ and 0.0016 in p (GPU fits). Rule:
     (a) the sealed predictions of the one look are the record; they are never refitted. (b) Any rescore from the
     same sealed predictions must reproduce every stored number within 1e-9 (absolute) and every label exactly;
     if it does not, both result sets are kept and reported, and no label is used until the lead has ruled. (c) A
     later refit from the snapshot (reproduction, never a second look) counts as reproducing the result when Δ and
     CI ends agree within 2e-4, per-seed Δ within 3e-4 and p within 3e-3. (d) The result tables flag every
     confirmatory row in which a quantity that decides the label lies within these tolerances of its threshold
     (CI end or per-seed Δ of 0; p of its Holm threshold); the label stands and the flag is reported with it.
     (e) A bootstrap p of 0 is reported as "< 2/B".
  7. **Rehearsal (X-B4).** Before the freeze, one full development rehearsal runs through the new commands from
     a snapshot: every fold of both blocks, at least two GPU chunks, a chunk killed and started again, the seal,
     the scoring, and a rescore from saved. A dry-run-only cap on training rows and ensemble size keeps it cheap;
     the cap is refused in real mode, and the rehearsal record says its numbers are not usable.
  8. **What does not change.** C-X, the secondaries, every model, feature set, seed, split and fold; the
     configuration count (12 of 12); the decision table apart from the larger listed sets; B, the bootstrap seed
     and the metrics. No new model is fitted on development. The candidate with sha256 20a4d2e3…4680 is
     superseded.
