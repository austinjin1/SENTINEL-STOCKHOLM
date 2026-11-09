# D3 KillAlert — pre-registration protocol (Iowa fish-kill risk, case-crossover)

**Status:** v1.0, 2026-10-24, written and split frozen **before any KillAlert model was fit** (phase 1 =
validation only; the test block is not scored in phase 1). Implements RESEARCH_PLAN §8 D3, §14, §15;
docs/research/07 §d.3. Code: `sentinel/models/downstream/killalert.py`,
`scripts/train/downstream_killalert.py`, tests `tests/test_downstream_killalert.py`.

## 1. Question and claim target
Do observed antecedent weather and stream-flow anomalies discriminate fish-kill days from same-place,
same-season control days? No published fish-kill model with a held-out AUC was found
(`docs/novelty_checks/downstream_killalert.md`, 2026-10-24, provisional), so the claim target is
**first-in-class**, which requires beating every baseline of §6 with paired CI excluding 0.

## 2. Data and labels
- `labels_iowa_fishkill` (Iowa DNR, vintage 20261022): events with release status Final, coordinates
  present, date ≥ 2000-01-01 (gridMET starts 2000): **591 events**.
- **Primary positive class:** `is_env_wq` (cause Environmental or Cyanobacteria Bloom).
  Secondary: all-cause; sensitivity: `is_env_wq` ∪ `Unknown (Natural Suspected)`.
- Cause task: Natural vs Anthropogenic among events with a known origin (Unknown excluded).
- Known biases: reporting effort (handled by the same-location design), approximate dates/locations
  (82 estimated locations, kept; sensitivity excluding them), cause attribution by investigators.

## 3. Unit, decision time and design
- **Matched set** = one event (case day d at its point) + referent days at the same point: the same
  calendar date ±3 days in every *other* year of the event's own year block, excluding referent days
  within 5 km and ±30 days of any recorded kill (any cause, any year 1981–2026).
- **Decision time:** end of day d − 1. All features use windows ending **d − 1** (unit-tested); no
  same-day or later observation enters. Sensitivity (pre-specified): windows ending d − 3 (reporting lag).
- Weather is observed (gridMET analysis) — this is a *nowcast/risk* model with observed inputs, not a
  forecast with NWP inputs.

## 4. Features (observed covariates only; no upstream-model outputs in phase 1)
- gridMET at the nearest SENTINEL-site cell (median distance 22.7 km train / 21.8 km val — a
  representativeness limitation): precipitation sums 1/3/7/14/30 d, max 1-d precip in 7 d, dry days
  in 14 d, Tmax/Tmin means, shortwave, VPD, wind; Tmax/Tmin/precip anomalies vs a ±15-d smoothed
  day-of-year climatology fitted on **train-block years 2000–2009 only**.
- Nearest USGS gauge ≤ 30 km (straight line; not necessarily the same stream): 7-d mean log-discharge
  anomaly (DOY climatology on 2000–2009) and 3-d change; ~55% missing (missing indicators added).
  Water-temperature anomaly is built but ~99% missing in Iowa and is not a model input.
- Cause task adds season, waterbody type and (if `manifests/downstream_catchment.json` exists)
  StreamCat watershed land use / manure / fertiliser at the snapped COMID.

## 5. Splits (frozen) and resampling unit
- Forward-chaining by event year: **train 2000–2009, val 2010–2017, test 2018–2026**; referent days
  never leave their event's block.
- Spatial secondary: 50-km block 5-fold CV over train+val events (`spatial_kfold`, seed 0).
- Frozen file `docs/preregistration/downstream_killalert_split.json`,
  **sha256 `4c7301c3144af503ab1ca72841ddfbbdda53e5ab30014150c28b092b72d0cf1e`**.
- Counts after building (train / val): all-cause strata 349 / 121; `is_env_wq` strata 107 / **24**;
  env ∪ natural-suspected 122 / 25; referents 21,351 / 5,905 (median 63 / 49 per set); cause-known
  274 / 98 (natural 123 / 31). Test block (2018–2026, ~121 events, ~38 env_wq) is untouched.
- **Power flag:** the primary class has only **24 validation events** (< 30 → "preliminary",
  RESEARCH_PLAN §14.4). Validation results for `is_env_wq` are reported as preliminary; all-cause
  (121 sets) is reported alongside. No claim is made from validation.
- Bootstrap unit: matched set (event), B = 2,000; sensitivity: event-year clusters (shared weather).

## 6. Endpoints, models and comparators
- **Primary endpoint:** mean matched-set concordance (P(case score > referent score), ties ½) on the
  primary class. Secondary: pooled AUROC/AUPRC over all case+referent days; lead-time curve (later).
- **Placebo row:** year-only conditional logistic (secular reporting trend; review W1).
- **Baselines (all under the identical split):** (B1) season climatology — month case rate (≈ constant
  within a matched set, expected ≈ 0.5); (B2) location-only logistic on lat/lon (constant within a set);
  (B3) **temperature-anomaly conditional logistic** (Till/Tye-style: Tmax anomaly 7/30 d × summer and
  × winter indicators); (B4) GBM is both a baseline class and a challenger.
- **Challengers:** conditional logistic on all weather + gauge features; LightGBM on the same + season.
- **Cause task:** prior, season + waterbody logistic, GBM (+ weather, gauge, StreamCat); AUROC, Brier.
- **Exact comparison that would earn "first-in-class":** on Test (2018–2026, scored once after freeze),
  the selected challenger's matched-set concordance for `is_env_wq` exceeds **each** of B1–B3 and the
  GBM/conditional-logistic alternative not selected, with paired bootstrap (unit = matched set) 95% CIs
  excluding 0, Holm-corrected across those comparisons, and absolute concordance > 0.5 with CI
  excluding 0.5 — plus the novelty check finalised (§1).

## 7. Tuning parity and configurations
- Every learned model is tuned by **inner 5-fold CV over matched sets of the training block only**,
  with the same budget (6 configurations each: clogit L2 ∈ {0.01, 0.1, 1, 3, 10, 30}; GBM num_leaves ∈
  {7, 15, 31} × min_child_samples ∈ {50, 200}); criterion = inner-CV matched concordance. Baselines are
  never weakened. Every configuration is logged with `common.log_config_tried`.
- Validation (2010–2017) is used to compare and select, never to fit.

## 8. Phase gate, freezing and test look
Phase 1 runs `--phase1` only after the downstream pre-training review. Test events are dropped right
after role assignment (`common.drop_protected`); `common.log_metric` refuses test split names. The test
block is scored once after `FREEZE_downstream_killalert.json` (split sha + scoring-code sha) exists.

## 9. Iteration plan (RESEARCH_PLAN §15.5 amendment)
Diagnose (per-season, per-waterbody, per-year residual concordance; feature attributions) → literature
search for the current best method (e.g. lake-temperature models of Tye/Till; DO-driven thresholds) →
one change at a time (e.g. lake vs stream sub-models, longer antecedent windows, upstream M1/M6
predicted DO/temperature once cross-fit providers exist) → re-run on validation/CV only. Every
iteration is logged; the number of configurations tried is reported with any claim.

## 10. Deviations log
All entries below: **2026-10-24, pre-training review, before any fit** (split unchanged, sha above).
- W1 secular trend: a score favouring earlier years reaches concordance 0.608 on val env_wq (reporting
  declined over time). Added a **year-only conditional-logistic placebo row** (`year_only_clogit`) to the
  ladder and a linear year term (`year_c` = (year − 2005)/10) to every learned model (temperature-anomaly
  clogit, weather clogit, GBM). Every challenger must beat the placebo too.
- W2 gauge structural missingness: gauge series begin 2000-10-01 and have gaps, so value-missingness could
  separate cases from referents. Gauge anomalies are imputed to 0 (climatological) for all rows; the
  only missingness feature is the location-level "no gauge ≤ 30 km" flag (constant within a set). The
  value-level `_isna` indicators were removed.
- W3 placeholder dates: events dated 1 January (events 4 and 5, 2000-01-01, page created 2000-03-27) are
  excluded from the primary analysis (`date_suspect`) and included only in a sensitivity run; events
  entered > 365 d after the kill date (`late_entry`, 21 train / 49 val) form a second sensitivity. The
  primary env_wq train count is therefore 105 (was 107); val unchanged (24).
- W4 inner CV is grouped by **event year** (5 folds of whole years) instead of by matched set.
- W5 pre-registered secondary analyses are run in phase 1: 50-km spatial CV (train+val events), d − 3
  windows, excluding estimated locations; paired bootstrap of each challenger vs **each** baseline
  (season, location, year-only placebo, temperature-anomaly clogit) with Holm; GBM uses 3 seeds
  (0, 1, 2; seed-averaged scores, per-seed concordance reported).
- W6 final test model: the configuration selected on validation is **refit on train + val (2000–2017)
  with the frozen hyperparameters** and scored once on test; the anomaly/gauge climatologies stay fitted
  on 2000–2009 (part of the frozen feature definition).
- W7 referent windows are clipped to the block's calendar years (unit-tested).
- The primary challenger (weather clogit vs GBM) is chosen by **train inner-CV concordance**, not by
  validation.

## 11. v2 — Iowa + Delaware pooled analysis (pre-registered 2026-10-25, before any v2 fit)
**Why.** Iowa alone gives only 24 validation `is_env_wq` events (preliminary). A dated search for further
public, geocoded fish-kill records (2026-10-24/25; see `docs/novelty_checks/downstream_killalert.md`
§"Label search") found one adequate source: **Delaware DNREC non-tidal fish-kill reports**
(`labels_delaware_fishkill`, 142 geocoded events 2004–2025; public domain). Florida FWC (no public bulk
export found; ArcGIS dashboard service no longer exists), Maryland/Missouri (PDF reports only), North
Carolina DWR (public layer holds 2026 only), Oklahoma DEQ (63 rows, 2021) and Minnesota (UMN layer holds
lake polygons, not events) were inadequate.
- **Split (frozen):** `docs/preregistration/downstream_killalert_v2_split.json`,
  **sha256 `0edd117dd15ad02ca2203d8f36c079d7c00ff11856ed954b15ad8fb8337cc7ef`**. Same year blocks for both
  states (train 2000–2009, val 2010–2017, test 2018–2026); Iowa roles identical to v1; 50-km spatial folds
  recomputed over pooled development events (seed 0).
- **Counts (events with coordinates):** IA train/val/test 349/121/121 (env_wq 107/24/38); DE 24/55/63
  (env_wq 9/28/35). Pooled validation env_wq ≈ 52 (primary class, before the date-suspect filter).
- **Cause harmonisation (DE):** keyword rules in `sentinel/data/sources/events_delaware_fishkill.py`
  (anthropogenic wins over environmental wording; disease → Natural non-env; low DO / winterkill / ice /
  algal bloom / temperature / turnover / low water → `is_env_wq`). Iowa classes unchanged. There is no DE
  analogue of "Unknown (Natural Suspected)".
- **Analysis:** **pooled is primary** (one model, matched sets from both states; referents never cross
  states or blocks); per-state matched concordance of the same pooled models is secondary
  (experiments `.../state_IA`, `.../state_DE`). Bootstrap unit = matched set.
- **Model specification for v2:** the ladder of §6 (incl. year-only placebo) with the challengers'
  feature set of iteration it1 (long 60/90-d windows added). Disclosure: it1 was chosen after viewing
  Iowa validation results (iteration 1 of §9), so v2 validation on the Iowa part is not independent of
  that choice; the Delaware part is. Same tuning budget for every learned model (§7).
- Weather: gridMET nearest SENTINEL cell (Delaware median 5.6 km, max 19.9 km); gauges from IA/MN/WI/IL/
  MO/NE/SD/DE/MD/PA/NJ USGS sites ≤ 30 km.
- Registry experiments carry the suffix `@v2`.
