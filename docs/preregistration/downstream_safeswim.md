# D1 SafeSwim — pre-registration protocol (Chicago beach fecal-indicator exceedance)

**Status:** v1.0, written 2026-10-24 **before any SafeSwim model was fitted or scored** (phase 1 =
validation years only; the test years are never loaded by phase-1 code). **Implements:** RESEARCH_PLAN
§8 D1, §14; docs/research/07 §b.2, §d.1; datasheet `docs/datasheets/labels_chicago_beaches.md`.
**Code:** `sentinel/models/downstream/safeswim.py`, `scripts/train/downstream_safeswim.py`,
`tests/test_downstream_safeswim.py`.

## 1. Frozen split

`docs/preregistration/downstream_safeswim_split.json` —
**sha256 `faabfff9bfbe6d5d72cca8c30f26c0ff79faeb8d1771626bbfd8d3dc854b25b9`** (canonical JSON, frozen
with `common.freeze_split` before any model selection; it also records the sha256 of every label
Parquet of the 2026-10-22 vintage). Unit = beach-day; role = role of the calendar year.

| Task | Train | Validation (phase 1) | Test (untouched) | Excluded |
|---|---|---|---|---|
| **D1a** culture *E. coli* > 235 CFU/100 mL | 2006–2013 | 2014–2015 | 2016 | — |
| **D1b** qPCR *Enterococci* > 1,000 CCE/100 mL | 2015–2019, 2021 | 2022–2023 | 2024–2025 (+2026 qPCR, no official predictions) | 2020 (COVID: 47 tests, 30 predictions) |

Units are never pooled across D1a/D1b (organism, unit and threshold differ). Humboldt (inland lagoon)
and Columbia (no coordinates) are not label units. **D1b label units are the 15 non-input beaches the
City predicts** (14 with qPCR); the 5 City model-input beaches (Calumet, 63rd St, Rainbow, Montrose,
South Shore) only provide features. Secondary spatial check: leave-one-latitude-band-out (south < 41.85°N
≤ central < 41.95°N ≤ north; band per beach in the split file), fit on train years of the other bands,
scored on validation years of the held-out band.

## 2. Labels

Beach-day geometric mean of that day's tests (replicate-level `result` floored at 1 before logging);
label = gmean **>** threshold (strict). Repeat tests on one day form one unit (no per-sample weighting).
Censored culture values (2419.6 cap) are kept as published (they exceed 235 either way).

## 3. Decision time (enforced in code, unit-tested)

- **D1a morning forecast** for day *d*: culture results from samples taken on day ≤ *d*−1 (18–24 h lab
  time; this matches the USGS "previous-day" persistence convention). Same-day culture is refused
  (`network_features` raises).
- **D1b nowcast** (afternoon; same information time as the City's "DNA Model"): same-day qPCR at the 5
  input beaches is allowed (`inp_sd_*`); nothing from the label beach on day *d*.
- **D1b morning forecast:** no same-day qPCR anywhere (`inp_prev_*`, `net_prev_*` = most recent day
  strictly before *d*).
- **Persistence:** previous beach-day at the same beach (strictly earlier); the persistence *baseline*
  is scored only where the gap ≤ 3 days (research/07 §b.2 rule).
- **Weather:** observed gridMET at the nearest available 4-km cell (0.8–23 km from the beaches; distance
  stored), windows ending *d*−1 for every mode (the gridMET day ends 07 UTC the next day, so day *d* is
  never complete at decision time); anomalies vs a day-of-year climatology fitted on the task's training
  years only. Beach weather stations are not used in v1.0.
- **Lake sensors:** lake-wide *d*−1 means with flatline (≥ 12 identical hourly values) and zero-turbidity
  masking — **ablation only** (`--with-sensors`): no water-sensor data exist in 2022 (validation) and
  < 5 records before 2014, so they are not in the primary feature set.
- No upstream-model (M1/M2/M6) features in phase 1 (`crossfit.phase1_no_upstream`).

## 4. Endpoints

- **Primary (per task): validation AUROC** (threshold-free, because the official prediction's unit is
  ambiguous and its 2017+ advisory threshold undocumented).
- Secondary: AUPRC (with prevalence), Brier (probabilistic models and rate baselines), ROC-point
  sensitivity at specificity 0.85 (USGS specificity goal), leave-band-out AUROC.
- **Primary comparisons** (paired, identical rows): D1b nowcast best learner vs the **official City
  model** (incumbent; research/07 b.2: AUROC 0.70 over 2017–2025) and vs the same-day **feature-beach
  GM** (strongest simple baseline, 0.74); D1b forecast and D1a vs **persistence**. Holm across the
  primary comparisons.

## 4a. Claim rule (RESEARCH_PLAN §15.1, H-G4) — the exact comparison that would earn "SOTA"

- **D1b (headline):** SafeSwim nowcast (frozen best learner) vs the **official City "DNA Model"**
  (`predicted_level`, the incumbent) on identical non-input-beach beach-days of the **test years
  2024–2025**: paired AUROC difference > 0 with the 95% paired cluster-bootstrap CI excluding 0 under
  **both** beach clusters and ISO-week clusters; **and** the same holds against the same-day
  **feature-beach GM** (strongest simple baseline; research 0.74). Holm across the D1a/D1b primary
  comparisons. Only then "SOTA" (incumbent re-scored on our split, strongest simple baseline included,
  single frozen test look); otherwise "competitive" (CI overlaps) or "negative".
- **D1b forecast:** vs persistence (≤ 3-day eligible rows) — reported, not an incumbent claim.
- **D1a:** vs persistence (USGS convention). The 2016 culture-era City model (`advisories_2016`,
  `prob_ge_235_cfu`, 9 beaches, 2016-06-04 → 09-22) exists **only in the D1a test year**, so it can be
  compared only at the single frozen test look (paired rows at its 9 beaches); it is not used in phase 1.
- Incumbent-reproduction check (§5) must agree with the research value within ±0.03 AUROC on
  overlapping years, or we stop and debug the join before any claim.

## 5. Baseline ladder and learners (one pre-specified configuration each)

Prevalence; location-only (beach rate, shrunk to the overall rate, k = 20); climatology (beach × month
rate, shrunk to the beach rate); persistence; rainfall rule (score = d−1 gridMET precipitation);
official City model (D1b; `predicted_level`); feature-beach GM (D1b nowcast; mean log10 same-day qPCR of
the 5 input beaches). Learners: L2 logistic (C = 1; train-median imputation + missingness indicators +
standardisation fitted on train) and sklearn HistGradientBoosting (300 iterations, lr 0.05, 15 leaves,
min leaf 40, L2 1.0; deterministic — seeds do not change it, so a single seed is reported). Features:
day of year, weekday, beach latitude, persistence (`prev_*`), network features (§3), gridMET windows
1/2/3/7/14 d. Every configuration tried is appended to `$SENTINEL_RUNS/downstream/configs_tried.jsonl`.

**Incumbent reproduction:** the official model's AUROC is recomputed on development years only
(2017–2019, 2021–2023) with the ≤ 3-day persistence-eligibility restriction and compared with the
research value (0.702, 2017–2025, n = 9,436); the full-period number is not recomputed in phase 1
(it includes test years).

## 5a. Tuning parity

- v1.0 uses one pre-specified configuration per learner (above). Any later hyperparameter search uses
  **inner forward-chaining CV on training years only** (folds = training years, validate on the next
  training year), never validation or test data, with the **same search budget (number of
  configurations, same search space style) for every learned baseline/incumbent-style model** (L2
  logistic ≈ Virtual-Beach MLR, GBM/RF ≈ Brooks 2016) **and every challenger**.
- Rate baselines (location, climatology) get their shrinkage k tuned with the same inner CV if any
  challenger is tuned. The official City model and persistence are used as published (no tuning
  possible) — they are never weakened, subsetted or recalibrated to the challenger's advantage.
- Every configuration tried (including failures) is logged via `common.log_config_tried`; the count is
  reported next to any claim.

## 5b. Iteration plan (RESEARCH_PLAN §15.5 amendment; validation only)

Each cycle: (1) **diagnose** on validation — per-beach/per-year AUROC, reliability, error analysis of
missed exceedances (rain-driven vs dry-weather, river-mouth proximity); (2) **literature search** for the
current best beach-FIB method (e.g. Brooks 2016 RF, Searcy & Boehm 2021/2023, Lucius 2019 hybrid,
Zhang 2018 NARX) and log what is adopted; (3) **one change at a time** (e.g. beach weather stations /
NOAA buoy waves, river-outflow or CSO proxies, rainfall intensity timing, beach-specific calibration,
TabPFN), re-run on validation only, keep/reject by the pre-specified validation AUROC with paired CI;
(4) log every run (TrainingRun, registry, configs_tried). Test years are scored once per frozen
configuration; a new configuration after a test look needs a new, unseen test period (2026 qPCR season
is the reserve for D1b). Iteration continues until the §4a claim is met on validation or the avenues are
documented as exhausted in RESULTS.md.

## 6. Uncertainty

Cluster bootstrap (B = 2,000) with **calendar ISO week** as the primary resampling unit (all beaches of a week
together — beaches share weather and lake state, so beach-level resampling would be anti-conservative);
paired bootstrap for model − baseline AUROC differences on identical rows; the claim-relevant comparisons
(§4a) are additionally bootstrapped with **beach** clusters, and a claim needs both CIs to exclude 0. Validation D1b has only 64
exceedances (2.7%) over ~30 week-clusters: CIs will be wide and are reported as such.

## 7. Phase-1 descriptive counts (from `--build-only`, no model fitted)

| Task | Train beach-days / positives | Val beach-days / positives | Val prevalence |
|---|---|---|---|
| D1a forecast | 12,414 / 1,758 (25 beaches) | 3,303 / 455 (21 beaches) | 13.8% |
| D1b nowcast & forecast | 5,649 / 374 (14 beaches) | 2,411 / 64 (12 beaches) | 2.7% |

D1b official predictions available on 4,938 train / 2,288 val beach-days; persistence-eligible
(gap ≤ 3 d) 5,582 / 2,387.

## 8. Freezing and the single test look

Test years are scored once, after this protocol, the scoring code and the chosen configuration are
frozen in `docs/preregistration/FREEZE_downstream_safeswim.json` (`common.check_roles` verifies split and
scoring-code sha256). Phase-1 registry entries use split names `val_2014_2015` (D1a) and
`val_2022_2023` (D1b) only.

## 9. Deviations log

- 2026-10-24 (before any model fit): split re-frozen once because `common.freeze_split` now validates
  only maps named `*role`; the year maps were renamed `d1a_year_role` / `d1b_year_role` (so their roles
  are validated) and the claim comparators were added to `meta`. Roles are unchanged. Superseded sha:
  `49e4a839a03548e6c398a46c05f558d9015c2ebf1ccd2c34a98ea2ba1a585b49`.
- 2026-10-24, **pre-training review, before any fit** (no critical findings; warnings fixed):
  1. *Identical rows.* Every registered ladder row is scored on an explicit row set, named in the
     experiment (`<task>/rows=all|official_rows|persist_rows`): within a row set only scorers defined on
     every row are reported, so learners are re-scored on the comparator's rows. The **D1b nowcast primary
     comparison uses `rows=official_rows`** (beach-days with an official prediction); persistence
     comparisons use `rows=persist_rows` (gap ≤ 3 d).
  2. HistGradientBoosting `early_stopping=False` (sklearn's `auto` would hold out a random 10% when
     n > 10,000, i.e. for D1a).
  3. `safeswim.SPLIT_SHA` = the §1 sha; `load_frozen_split()` passes it to `common.load_split(expect_sha=)`
     and re-hashes every label Parquet against the sha256 values stored in the split meta (raises on
     mismatch).
  4. *R6 disclosure.* The research values used as reference points (official model AUROC 0.702,
     feature-beach GM 0.740; research/07 §b.2) were computed over 2017–2025, i.e. **including the test
     years 2024–2025**. No model, feature or threshold choice in this protocol used them; they serve only
     as the reproduction target (§4a), which is re-checked on development years only.
  5. Shared code updates acknowledged (threadpoolctl thread cap; `check_roles` requires `split_sha` with
     `allow_test`; scoring hash = `common.scoring_files('safeswim')`, incl. the Chicago parser); tests re-run.
  Paired AUROC differences use a rank-based AUROC (`safeswim.fast_auroc`, identical to sklearn's) for speed.

## 10. Iteration log (validation only)

- **Cycle 0 (phase 1, v1.0 config, run `phase1_base`, 2026-10-24):** incumbent reproduced on development
  years (official AUROC 0.703 vs research 0.702). D1b nowcast on `official_rows` (2022–23, n = 2,288,
  61 exceedances): official 0.614, feature-beach GM 0.699, logistic 0.661, GBM 0.659 → learners do not beat
  the strongest simple baseline (GM). **Diagnosis:** low validation prevalence (2.7% vs 6.6% in training)
  and a wide feature set (61 columns, mostly weather windows) relative to 374 training positives; the GM
  is a strong low-variance signal.
- **Cycle 1 (queued, one change each vs cycle 0, same learners/configs for all models = tuning parity):**
  `inp_each` (5 individual same-day input-beach values + nearest input beach — the City "DNA Model"
  design), `compact` (12 parsimonious features), `compact_each`, and — after a literature check
  (Lucius et al. 2019, *Water Res. X* 2:100016: beaches are selectively correlated) — `base_corr` /
  `compact_corr` (same-day value of the input beach best correlated with each label beach on
  **training rows only**). Scored on `rows=official_rows`, experiment suffix `/it1_<set>`.
- **Cycle 2 (queued after cycle 1; lead guidance 2026-10-25: hydrometeorological drivers):** new source
  `safeswim_drivers` (datasheet `docs/datasheets/safeswim_drivers.md`, manifest `safeswim_drivers`):
  USGS daily discharge (05536000, 05536290, 05536890), CO-OPS Calumet Harbor lake level, NDBC 45007/45174
  waves/wind/water temperature, Chicago beach weather-station hourly rain (intensity), nearest-neighbour
  beach result on exactly d−1. **All values end d−1** (conservative for the nowcast). One driver group
  added at a time to the base set (`base+q|lake|waves|stn|nbr`), D1b nowcast on `official_rows` and D1a
  on `persist_rows`; driver columns missing on > 50% of training rows are dropped (train-only rule, e.g.
  45174/weather stations for D1a). Same learners and configurations as cycle 0 (tuning parity); only the
  primary comparators are re-scored (they are identical across runs), via `--baselines`.
  Not obtained: CSO / lock-opening event logs (MWRD/USACE; no bulk public download found in this cycle).
- **Cycles 1–2 results (runs `it12_d1b`, `it12_d1a`, 2026-10-25 02:56–03:58; recorded 2026-11-08).** D1b nowcast,
  `official_rows` (n = 2,288, 61 exceedances), AUROC; paired diff vs official City model / vs feature-beach GM
  (ISO-week clusters; beach clusters in the registry):
  | set | logistic | GBM | logistic − official | logistic − GM |
  |---|---|---|---|---|
  | base (cycle 0) | 0.661 | 0.659 | +0.047 [−0.046, 0.140] | −0.039 [−0.092, 0.011] |
  | inp_each | 0.679 | 0.678 | | |
  | compact | 0.721 | 0.680 | +0.107 [0.035, 0.181] | +0.021 [−0.021, 0.069] |
  | **compact_each** | **0.736** | 0.698 | **+0.122 [0.044, 0.198]** (beach [0.045, 0.211]) | +0.036 [−0.001, 0.078] (beach [−0.000, 0.059]) |
  | base_corr / compact_corr | 0.678 / 0.729 | 0.646 / 0.691 | | |
  | base + q / lake / waves / stn / nbr | 0.665 / 0.648 / 0.661 / 0.646 / 0.661 | 0.688 / 0.666 / 0.659 / 0.680 / 0.676 | | |
  Diagnosis: parsimony is what helps (12–17 features vs 61); driver groups added to the wide base set
  give GBM +0.007–0.029 (q, stn, nbr) and nothing for the logistic. D1a (persist_rows, 441 exceedances):
  GBM base+nbr 0.692 (vs persistence +0.120 [0.073, 0.174]; vs climatology +0.028 [−0.008, 0.067]).
  Status: on validation compact_each logistic beats the incumbent with CI excluding 0 under both cluster
  units; vs the feature-beach GM the CI touches 0 → §4a not met on validation. Configurations so far:
  cycle 0 = 1 per learner; cycles 1–2 = 15 feature sets × 2 learners (feature-set selection on validation:
  selection effect disclosed; the test look is the arbiter).
- **Cycle 3 (pre-registered 2026-11-08, before any fit):** one change per run vs `compact_each` — add one
  driver group (`compact_each+q`, `+stn`, `+nbr`, `+lake`, `+waves`), D1b nowcast on `official_rows`,
  same learners/configurations, experiment suffix `/it3_<set>`. Keep a group only if the logistic or GBM
  AUROC rises by ≥ 0.005 on validation; the freeze candidate is the best of {compact_each, compact_each +
  kept groups} by validation logistic/GBM AUROC.
- **Cycle 3 result (run `it3_d1b`, 2026-11-08).** Logistic / GBM AUROC on `official_rows`: compact_each+q
  **0.743** / 0.698; +stn 0.732 / 0.690; +nbr 0.736 / 0.695; +lake 0.735 / 0.690; +waves 0.736 / 0.698
  (compact_each 0.736 / 0.698). Only `q` (USGS discharge d−1) passes the ≥ 0.005 keep rule; the others are
  rejected. **compact_each+q logistic − official City model +0.129 [0.046, 0.209] (week) / [0.053, 0.220]
  (beach); − feature-beach GM +0.043 [0.014, 0.080] (week) / [0.015, 0.063] (beach)** → the §4a rule holds
  on validation under both cluster units. **Freeze candidate (D1b): L2 logistic (C = 1), feature set
  `compact_each+q`**, refit on training years only (2015–2019, 2021) as specified, scored once on
  2024–2025. Selection disclosure: 21 feature sets × 2 learners screened on the same validation years
  (cycles 0–3); validation optimism is expected; the single test look decides. D1a stays a secondary
  task (GBM base+nbr beats persistence; not climatology). Freezing awaits the lead's go-ahead.

## 11. Pre-test gate deviations (2026-11-08, before any test look)
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
- **Test rows = 2024–2025 only.** The 2026 qPCR reserve season is not loaded in the test look (no labels or
  scores written); it stays the reserve for a post-look configuration. 2026 has no official City
  predictions, so a reserve look could re-test only vs the feature-beach GM.
- **Holm family = the four pre-registered D1a/D1b primary comparisons (m = 4):** D1b nowcast vs official and
  vs GM (scored), D1b forecast vs persistence and D1a vs persistence (not frozen, counted with p = 1).
  Applied separately under ISO-week and beach clusters; both must reject.
- The frozen 26 feature columns are asserted in the test script (no silent loss of the `q` drivers);
  `safeswim_drivers` added to `DATA_MANIFESTS`. USGS discharge for 2024–2025 is now approved data, whereas
  only provisional values exist on a real decision day (true on validation too).
- **Correction:** the screening count in §10 cycle 3 ("21 feature sets") is wrong — 16 D1b-nowcast feature
  sets were run, **15 distinct** (`+waves` equals `compact_each`); 50 config-log entries / 40 unique hashes
  incl. one crashed run. Power caveat: ~2,300 test beach-days at 12 beaches; the bootstrap over 12 beach
  clusters may be anti-conservative. Claim names the task: D1b qPCR *Enterococci* > 1,000 CCE/100 mL nowcast.
- Protocol modification time vs fit time for cycle 3 is not provable without git (see (d)).

## 12. Single test look — result (2026-11-08 20:32–20:33; FREEZE_downstream_safeswim.json; registry `safeswim/d1b_nowcast/test_look`)
D1b qPCR *Enterococci* > 1,000 CCE nowcast, 2024–2025 official rows (n = 2,300, 101 exceedances; 2026 never
loaded). AUROC: frozen logistic 0.729 [0.679, 0.785]; official City model 0.686 [0.626, 0.744]; feature-beach
GM 0.730 [0.689, 0.772]; GBM 0.713; persistence 0.680. Paired: logistic − official **+0.043** [−0.009, 0.102]
(week, p 0.101) / [0.003, 0.105] (beach, p 0.038); logistic − GM **−0.000** [−0.036, 0.037] (week) /
[−0.030, 0.025] (beach). Holm (m = 4) rejects nothing under either cluster unit → **§4a not met.
Label: competitive** (CI overlaps the incumbent under ISO-week clustering; no gain over the same-day
feature-beach GM). The validation gain over GM (+0.043) did not replicate (selection over 15 feature sets).
Caveat (post-results review): the beach unit has only 12 clusters (week unit 32), so beach-cluster percentile
CIs are likely too narrow.
