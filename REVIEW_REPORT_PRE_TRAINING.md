# SENTINEL v2 — Pre-Training Review Reports (RESEARCH_PLAN.md §3 R10)

**Review mode:** Pre-Training (review skill, sections 1–5), one section per workstream.
**Reviewer:** adversarial research-integrity review (AI-assisted), run at each workstream's first-training gate.
**Background:** v1 was retracted for data leakage, a placebo-failing early-warning detector, misattributed
baselines and synthetic data. Every workstream must pass this gate before its first real (non-smoke) training run.
Critical issues block training or freezing until they are fixed and re-reviewed.

| Workstream | Date | Status |
|---|---|---|
| M1 SENTINEL-Sonde (+ EW-v2 daily tier) | 2026-10-22 | 🔴 FAIL (EW calibration / test path); forecast-model training may proceed on the FINAL split — see scope note |
| M6 SENTINEL-HydroMet | 2026-10-22 | 🔴 FAIL (statistics unit / test path / guard); full training on the FINAL split with val scoring may proceed — see scope note |
| Downstream (D1, D2, D3, D5, D6, D8) | 2026-10-24 | 🔴 FAIL at review (6 criticals: D2 ×3, D5, D6, D8); all 6 fixed before any fit of the affected app (resolution table) → training gate passed per app |
| M2 SENTINEL-Orbit | 2026-10-24 | 🔴 FAIL at first review (C1 split / C2 comparator selection / C3 stale cache) → fixed → 🟡 PASS WITH CONCERNS at re-review; N1 fixed (split v3) before any real training. Open before freeze: W4 (power, national WQP 2024–25 test data) |
| M5 SENTINEL-BioAssess (pre-test gate) | 2026-10-25 | 🔴 FAIL — no contamination or leakage; 5 scorer/freeze criticals (C1 per-seed, C2 H-M5/Holm in registry, C3 hash coverage, C4 untested single-shot, C5 run-time params) must be fixed before freeze |
| M5 SENTINEL-BioAssess (delta re-review) | 2026-11-08 | 🟡 PASS WITH CONCERNS — C1–C5, W1–W12 resolved; dry run + reproduction + exact re-score verified; before freeze: R1 exposure wording (D5 used 2018-19), R2 hash/pre-import chemtox chain, R3 commit/archive tree; then regenerate candidate |

---

## M1 SENTINEL-Sonde — 2026-10-22

**Status: 🔴 FAIL — DO NOT freeze or run EW calibration/selection until C1–C3 are fixed.**

**Scope of the verdict.** The forecasting stack (data cube, guard, splits, common rows, baselines, GBM, LSTM,
Chronos-2 zero-shot / fine-tune, CRPS scoring) passed every leakage check: no feature, normaliser, climatology,
residual quantile or training window reaches past WY2021 or into val/test sites. Its full training runs on the
FINAL split, scored on the **val** block, may proceed (the runner already refuses non-smoke runs on a non-final
split, and the FINAL split doesn't exist yet). Three things block the gate. (C1) The EW-v2 false-alarm budget
in the code is not the one in the protocol. (C2) The EW script fits on the labels of whichever block it scores,
including test. (C3) There's no way to score test with frozen models. C1 has to be fixed before any EW threshold,
placebo result or val-based EW model selection is produced. C2 and C3 have to be fixed before the freeze.

**Reviewed:** RESEARCH_PLAN.md §3, §5.1, §12, §14–15; docs/preregistration/ew_v2_protocol.md (v0.1 incl. the
2026-10-22 deviation); docs/CONTRIBUTING_V2.md; `sentinel/models/sonde/*.py`,
`sentinel/models/baselines/sonde_baselines.py`, `sentinel/eval/{metrics,splits,bootstrap,early_warning,hazards,ew_scoring}.py`,
`sentinel/data/{sonde_daily,sonde_splits}.py`, `scripts/train/sonde_*.py`, `scripts/eval/sonde_*.py`,
`configs/sonde/*.yaml`, `tests/test_{sonde_models,eval_core,hazards}.py`, the installed `chronos` 2.3.2
`Chronos2Dataset`/`fit()` source, and the smoke outputs in `/home/bcheng/STOCKHOLM/sentinel_runs/sonde/smoke/`.

**Tests:** full suite `python -B -m pytest`: **548 passed**, 0 failed (1 unrelated rasterio deprecation warning);
M1 targeted files: 43 passed. Probes (outside the repo):
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/review_m1/probe_ew.py`.

### Critical issues (must fix)

**C1. The alarm-day false-alarm budget in the protocol is not implemented. The standing-alarm loophole is
still open.**
`sentinel/eval/ew_scoring.py:161-180` (`calibrate_episode_threshold`), `scripts/train/sonde_events.py:164-183`,
`configs/sonde/events.yaml:24`.
The protocol (§6, deviation of 2026-10-22) says *"alarm-days on valid non-onset issue days ≤ 1 per site per 30
days (pooled POFD ≤ 1/30) … every alarmed day counts"*. The code still counts **episodes** per 30 *usable* days
(the denominator includes onset-positive days). Variant `b1` uses maximal runs. Variants `b1c`/`b2c` split a
standing alarm into chunks of L days, so at L = 7 a standing alarm costs only 1/7 per day.
- Probe P1: 30 sites × 300 days, no onsets, 7 alarm-days in every 30 (time-in-alarm 23.3%). This passes the
  1/30 budget under `b1` for every L, and under `b1c` for L = 7 (`fa_per_site_30d = 1.00`). That is 7× the
  protocol's alarm-day POFD of 0.033.
- Smoke `dev_events2/event_scores.csv` at the "1/30" `b1c` threshold: time-in-alarm is 3–9% at L = 3 and
  5–21% at L = 7. For example, climatology H1 at L = 7 has τ = 0.211, POD_real 0.67 and POD_placebo 0.59.
  Placebo POD is 0.5–1.0 across methods. This is the v1 placebo-failure pattern, and it is being produced at a
  budget the protocol does not allow.
- **Fix:** calibrate on val with `early_warning.calibrate_threshold(score, y, valid, target_pofd=budget/30)`
  over the valid issue rows. Valid rows already exclude in-event days, so the y == 0 rows are the "valid
  non-onset issue days". Use 1/30 as primary and 2/30 as sensitivity. Report the realised per-site and pooled
  alarm-day POFD and time-in-alarm. Keep the episode variants only as secondary, clearly labelled descriptives,
  and fix the `events.yaml` comment ("episode"). Add a unit test asserting that a 7-in-30 standing alarm fails
  the primary budget for L ∈ {1, 3, 7}.

**C2. `sonde_events.py` fits on the labels of the block it scores, so `--eval-blocks test` would fit on test
labels.**
`scripts/train/sonde_events.py:110` loops over `runner.eval_blocks(a)`, and inside that loop:
- `:135` `EventGBM.fit(X_tr, tr.y, X_ev, ev.y)` early-stops on the scored rows `ev`.
- `:160` `crossfit_platt(p, ev.y, wy)` fits Platt recalibration on the scored block's labels. It is out of fold
  by water year, but still on that block.
- `:176` `calibrate_episode_threshold(frames, …)` calibrates alarm thresholds on the scored block.
- `:181` runs the placebo comparison at those in-sample thresholds.

On the val block this only biases model selection: event_gbm is early-stopped on the rows it is ranked on, and
the other methods aren't. On the test block it would be direct test-label leakage. Nothing writes or reads a
frozen threshold/recalibration file, although protocol §10 requires a hashed "Threshold file".
- **Fix:** split the script into two phases.
  - *calibrate (val only):* early-stop on an inner slice that is never scored (e.g. WY2020–21 of the train
    block, as GBM does, or WY2022 for early stopping and WY2023 for selection). Fit Platt on all val rows.
    Calibrate thresholds per C1. Write `thresholds_<hazard>_L<L>.json` (threshold, Platt a/b, best_iteration,
    model hashes) and its SHA-256.
  - *score:* load that file, apply it unchanged, and never refit.

  Make `sonde_events.py` refuse any protected block unless the frozen calibration file is given and its hash
  matches `FREEZE_sonde.json`.

**C3. There is no predict-only path from frozen weights. Test forecasts can only come from a retrain with test
data loaded.**
`scripts/train/sonde_gbm.py:30-52`, `scripts/train/sonde_lstm.py:95-115`, `scripts/train/sonde_chronos_ft.py:163-182`.
Each script retrains, then scores `eval_blocks` in the same process. With `--eval-blocks test --allow-test`
the test cube is in memory during training. The test forecasts would also come from a new, non-deterministic
retrain, not from the validated checkpoint whose hash the freeze record (§10, "Model weights per hazard") must
hold. `SondeGBM.load`, `SondeLSTM.load_state` and the Chronos fine-tuned checkpoint exist, but no script uses
them for scoring.
- **Fix:** add `scripts/eval/sonde_predict.py` (or a `--from-checkpoint` flag) that loads the frozen weights,
  checks their SHA-256 against `FREEZE_sonde.json`, and only predicts. Make the training scripts refuse
  `test`/`test_b` in `--eval-blocks` and `include_test_sites: true`.

### Warnings (should fix; all before the freeze unless noted)

- **W1. Issue days where an onset is structurally impossible are scored as valid negatives.**
  `sentinel/eval/hazards.py:100` excludes only the issue day itself when it is an event day. In the refractory
  window after an event, and on the days after a non-onset hypoxic run, every row is `valid=True, y=0` even
  though the refractory rule makes y = 1 impossible (probe P3). Good forecasters are penalised in BS and in
  false alarms on near-event days, and the effect differs by method. This is not quite what protocol §4 says
  ("ongoing event excluded"). Fix: mark issue day d invalid (or score it as "continuation") when (d, d+L]
  lies entirely inside the refractory window of an event day ≤ d.
- **W2. Placebo dates aren't restricted to "clean periods".** `sentinel/eval/early_warning.py:207` excludes only
  windows around the `events` passed in. `sonde_events.site_frames` (`scripts/train/sonde_events.py:57-58`)
  passes only onset-anchored events (`ew_scoring.daily_events`). So event days without a clean pre-window
  (after gaps, or continuations) and near-threshold periods remain sampleable (probe P2: 2 of 200 placebo
  dates fell on event days, 4 inside or just after a non-onset hypoxic run). The bias is conservative (placebo
  POD goes up), but it isn't the pre-registered control. Fix: exclude all event days ± (h + exclusion), and
  require the placebo day to meet the same clean pre-window rule as a real onset (e.g. H1: the 3 prior days
  ≥ 4 mg/L).
- **W3. Approved-only test scoring isn't implemented.** `Cube.prov` is loaded (`sentinel/models/sonde/data.py:92-93`)
  but never used in rows, labels or scores. Protocol §6 requires Approved values only for test. Add a
  `score_mask &= ~prov` path for protected blocks and for EW labels.
- **W4. Quantile clipping to `PLAUSIBLE` isn't implemented anywhere**, although protocol §6 says it is. No
  model is clipped, so all are currently treated the same, but the protocol text and the code disagree.
  Implement one shared clip in `common.save_forecast`/`score_rows` and test it.
- **W5. The freeze file's contents aren't validated.** `sentinel/models/sonde/guard.py:70-73` accepts any
  non-empty JSON object, and `split_sha256` is optional. Probe P4: `{"anything": 1}` plus `allow_test=True`
  unlocks `test` and `role:test`. The default paths are safe (flag and file both required, and the file is
  currently absent). Still, require `split_sha256`, `scoring_code_sha256`, `thresholds_sha256` and
  `weights_sha256`, and verify each one against what is in use.
- **W6. Validation selection bias in the forecasting models.** GBM (`sonde_gbm.py:41`, `es_rows=data.rows(v,"val")`),
  the LSTM (`lstm.py:323`) and Chronos-FT all early-stop on the val block, including val-role sites. Their val
  scores are then also used to rank models and to weight the planned ensemble. This is acceptable for
  selection, but report it, and keep val numbers out of any headline.
- **W7. The val role is too thin for hazard variables.** In the DEV split report, val has chl 1 site, nitrate
  5, pH 2 and fDOM 14. Test has chl 3 and nitrate 25. With a 70/10/20 split by site count, the val-based EW
  calibration for H4/H5b, and the H5b claims, would rest on 1–5 sites. Stratify the FINAL split at the
  basin-group level by hazard-variable availability, or pre-register H5b as "underpowered/descriptive".
- **W8. M1 primary metric vs CI.** §15.2 defines the primary as the **median** per-site CRPS skill with a CI.
  `scripts/eval/sonde_score.py:30-46` bootstraps only the **pooled** skill, and `skill_summary` has no CI.
  Add a site-bootstrap CI for the median per-site skill and a Diebold–Mariano test (§14.4). Neither is
  implemented yet.
- **W9. Lead-time and issue-time convention.** Issue date d uses day d's complete daily statistics (e.g. the DO
  daily minimum), so the earliest real issue is the end of day d (with DV publication lag, later). Plan §12.3
  says 06:00 issue. A lead-1 "hit" for an onset on d+1 can therefore come with well under 24 h of warning (DO
  minima occur around dawn). Write the convention down, and count "≥ 24 h warning" POD only for alarms at
  d ≤ onset − 2, or define the 15-min tier as the lead-time source.
- **W10. Primary hazard H2 (chloride) isn't implemented** in the daily tier (`events.yaml` hazards: H1 tiers,
  H4, H5b). Protocol §7 lists H2 as primary. Either implement the SC→Cl site-regression series or amend the
  protocol before registration.
- **W11. Climatology reference for test-role sites.** `HarmonicClimatology` per-site deviations come from
  `tr_all` with roles ("train", "val") (`sonde_events.py:109`, `events.py:97` default roles). When test-role
  sites are scored, their reference would be pooled-only, which is weaker and would inflate BSS for unseen
  sites. Fit deviations from each scored site's own training-year rows, whatever its role, as the docstring
  promises.
- **W12. Site eligibility uses the full record.** `scripts/data/build_sonde_split.py:35` keeps sites with
  ≥ 730 days counted over all years, including WY2024–26. That conditions inclusion on test-period
  availability (not on values). Count eligibility days in WY2008–2023 only.

### Observations (nice to fix)

- O1. `prepare_data` drops sites with no finite climatology (`common.py:224-228`) without recording them.
  Protocol §6 asks for them to be "reported as unscorable". Write them to `data_info.json`.
- O2. Baseline residual/change quantiles (`sonde_baselines.py:66-84`) and the GBM WY2020–21 residual slice
  are computed against an in-sample climatology (fitted on the same years). Expect slight under-coverage out
  of sample, the same for all models. Report coverage.
- O3. The LSTM event head is fitted on embeddings of the LSTM's own training rows, so its train fit is
  optimistic. It is evaluated on val/test only, so this is not leakage.
- O4. Chronos-2 contamination. The comment in `sonde_chronos_zeroshot.py` relies on disclosed corpora. Chronos-FT
  is a candidate SENTINEL model (A), so its Test-A results should carry the same caveat, with Test-B as the
  clean check.
- O5. A stale comment says peft is not installed (`chronos_ft.py:4-5`, `configs/sonde/chronos_ft.yaml:11`).
  peft 0.21.2 is installed and pinned. Note too that `fit()` overrides `eval_steps`/`save_steps` via
  extra kwargs (fine; documented here).
- O6. `common.yaml` `site_types: null` says "open question". The protocol now fixes a streams-only headline,
  and the DEV split is already streams-only. Set it explicitly for the FINAL runs.
- O7. `runner.score_and_summarize` summarises roles all/train/val only. Add a "test" role row for the
  unseen-site cell.
- O8. The quantile→onset probability uses only horizons {1,2,3,5,7}, skipping days 4 and 6 for L = 7. It is
  the same for every quantile model and is calibrated by Platt. Document it.

### What was verified as correct (no action)

- **Temporal:** `SondeSplit.time_mask` gives issue d ≥ block start + 42 d (non-train) and d + h ≤ block end.
  Climatology, m_site/s_site, LSTM floors and static statistics, GBM scale floor and baseline φ/quantiles all
  use dates ≤ 2021-09-30 and train-role sites only. Features are strictly ≤ d (`test_features_causal` has
  power). GBM fit rows have target ≤ 2019-09-30 and the residual slice is WY2020–21 with a 42-d embargo,
  disjoint from fit rows. LSTM training samples come from `issue_idx("train", 14)`, so all targets are
  ≤ WY2021. Chronos-FT train inputs are truncated at 2021-09-30 per train-role site, and the installed
  `Chronos2Dataset` (TRAIN mode) slices targets inside each series, so it can't reach WY2022+. Validation
  windows end inside val after the embargo. The loaded cube ends at the last authorised block (`data_end_date`
  plus `assert_cube_within`).
- **Spatial:** roles are disjoint (`assert_disjoint`). Basin groups never span roles. Test-role sites need
  `include_test_sites` plus the flag plus the freeze file. Every smoke run's `data_info.json` shows roles
  {train, val}, protected [] and cube_end 2023-09-30. No test-role site or test-year value was loaded.
- **Identical rows:** `common_rows` plus `score_forecast` (raises on non-finite values in scorable rows),
  `align_forecast` (raises on missing rows), and `pooled_skill_ci` (raises on row mismatch). Naive-baseline-
  undefined rows are dropped for every model. All models use the same 3-day staleness rule.
- **Scores:** CRPS is 2 × mean pinball over 19 levels, in original units, after sorting. Back-transforms are
  monotone, so quantiles map correctly. LSTM quantiles are monotone by construction, GBM's by sorted residual
  quantiles × s > 0, Chronos's by sorting. BSS, AUROC/AUPRC and contingency maths are correct. Bootstrap units
  are site (CRPS) and site-month (BSS).
- **EW labels:** onset rules match protocol §3 for H1 (+ tiers), H4 and H5b. In-event issue days are
  excluded. The detection window is strictly before onset, [o − h, o). Placebo sampling is month-matched with
  K = 20 per onset.
- **Registry:** smoke runs and non-final splits can't be registered (`assert_registrable`), and non-smoke runs
  refuse non-final splits.

### Checklist summary

| Category | Items checked | Passed | Failed | N/A | Notes |
|---|---|---|---|---|---|
| 1 Code correctness (architecture, loss, data pipeline, training loop) | 18 | 17 | 1 | 0 | W1 label validity; LSTM/GBM/Chronos loops correct (train/eval modes, no_grad, zero_grad, clip, best-ckpt restore) |
| 2 Data leakage (temporal, spatial, preprocessing, group, test guard) | 12 | 9 | 3 | 0 | C2 (fits on scored block's labels), C3 (test data in memory during retrain), W5 (freeze contents) |
| 3 Experimental design (baselines, fair rows, metrics, power, selection) | 12 | 6 | 6 | 0 | C1 budget, W2 placebo, W7 power, W8 primary-metric CI, W10 H2 missing, W11 reference |
| 4 Reproducibility (seeds, pinned deps, configs, run logs) | 6 | 6 | 0 | 0 | seeds, pins (`requirements-v2.txt`), config hash, configs_tried.jsonl, TrainingRun |
| 5 Code quality | 5 | 4 | 1 | 0 | no TODO/FIXME; stale comments (O5, `events.yaml`) |
| Protocol conformance (ew_v2_protocol §3–§10) | 8 | 3 | 5 | 0 | alarm-day budget, approved-only, clipping, threshold file, H2 |

**Re-review required** for: `ew_scoring.py`/`sonde_events.py` after C1 and C2, the new predict-only path (C3),
and the guard (W5), before `FREEZE_sonde.json` is written.

---

## M6 SENTINEL-HydroMet — 2026-10-22

**Status: 🔴 FAIL. Do not write `FREEZE_hydromet.json` or touch test data until C1–C3 are fixed and re-reviewed.**

**Scope of the verdict.** The training side passed every leakage check:
- inputs are forcing/static/calendar only;
- normaliser, early stopping, air2stream donor calibration, DO/DO_sat ratios and 7-DADMax offsets are all
  fitted on train-role × train block;
- forcing windows are causal and the embargo is applied;
- every model is scored on identical approved cells.

Full (non-smoke) training runs of the baselines and both EA-LSTMs on the FINAL split, scored on the **val**
block, may proceed. What fails is the inference and test layer:
- (C1) primary CIs and p-values resample sites, although the val cell has only 12 basin groups;
- (C2) the pre-registered 3-seed ensemble cannot pass through the frozen test path;
- (C3) the guard has three gaps.

C3a must be fixed before any further smoke run on a non-final split. C1–C3 must be fixed before the freeze.

**Reviewed (sha256 prefixes at 10:20 EDT; several files changed during the review):**
- `sentinel/models/hydromet/{data 438ea52a, features 26c37d03, physics 6eab5a08, baselines ebade577,
  lstm f3684941, evaluate cc4b5d9a, guard 29641527, runner 3c415522}.py`
- `scripts/train/hydromet_{baselines c7475c00, lstm 68220d15}.py`
- `scripts/eval/hydromet_{predict 10da452e, score 98796c81}.py`
- `configs/hydromet/*.yaml`, `tests/test_hydromet.py`
- `docs/preregistration/hydromet_protocol.md` (DRAFT v0.1, `deed2668`; it appeared during the review)
- shared code: `sentinel/models/sonde/{data,guard}.py`, `sentinel/data/sonde_splits.py`,
  `sentinel/eval/{bootstrap,metrics,splits}.py`
- `docs/datasheets/streamcat.md`
- smoke outputs in `/home/bcheng/STOCKHOLM/sentinel_runs/hydromet/smoke/`. All of them used the FINAL split.

**Tests:** `python -B -m pytest tests/test_hydromet.py` gave **36 passed** (run twice, before and after the
mid-review edits).

**Probes** (outside the repo) are in
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/review_m6/`:
- `probe_split.py`: 2,559 eligible sites; 0 COMIDs and 0 identical StreamCat vectors are shared across roles;
  the nearest val→train distance is 4.3 km, with 1 val site within 5 km.
- `probe_groups.py`: number of basin groups per role and target.
- `probe_lstm_invariance.py`: EA-LSTM predictions, plain and physics-DO, are bit-identical when every target
  outside train-role × train block is perturbed. Predictions up to day t do not change when forcings after t
  are perturbed.
- `probe_guard.py`: C2 and C3.

### Critical issues (must fix)

**C1. Primary inference uses site as the resampling unit, but sites are clustered in a handful of basin groups.**
Code: `sentinel/models/hydromet/evaluate.py:246-247` (`paired_bootstrap` without `groups`), `:98`, `:111`
(`summarize`) and `:205` (`pooled_metrics`, `groups=si`). Protocol: §5 (line 98) and §7 (line 135).
- The 300 eligible val-role sites fall in **12** basin groups. Group 0501 alone holds 118 (39%).
- For DO, about 87 val sites fall in 9 groups, and the largest group holds 57%.
- Test has 547 sites in 36 groups.
- Basin groups exist because neighbouring sites share weather, geology and land use. Their per-site KGEs are
  therefore not independent. Resampling sites gives too-narrow CIs and too-small p-values. This is the
  pseudo-replication failure v1 was retracted for (R7, §14.4 "cluster bootstrap").
- **Fix:**
  - For the primary H1, resample **basin groups**, each with all of its scored sites, and recompute
    Δ = median_site(model) − median_site(comparator) on each resample. Use the same scheme for the pooled
    metrics and the summary CIs.
  - Keep site resampling only as a labelled sensitivity analysis.
  - Report n_groups next to n_sites everywhere.
  - Update protocol §5 and §7 and add a unit test.

**C2. The frozen test path cannot score the pre-registered primary model (3-seed ensemble).**
Code: `scripts/eval/hydromet_predict.py:53`, `:56`, `:78`; `scripts/eval/hydromet_score.py:57-59`.
- `hydromet_predict` keys every EA-LSTM checkpoint as `ealstm` (or `ealstm_physdo`) and writes no `seed`
  into the prediction header.
- So `verify_weights` can match only one sha256 per name, and 2 of the 3 seeds fail.
- Even with a workaround, `hydromet_score` then names all three files `ealstm` and raises "duplicate model
  name". `seed_ensembles` never forms the evaluated model on test.
- The val path works only because `hydromet_lstm.py:62` writes the seed. This is the same class of defect
  as M1 C3.
- **Fix:**
  - Key artifacts and freeze entries by `<model>_s<seed>`, taking the seed from the checkpoint cfg.
  - Write `seed` into the predict header.
  - Add an end-to-end test (temporary freeze, 2 seeds, predict → score `--primary`).

**C3. Guard gaps.**
- **(a) A smoke run on a non-final split loads FINAL test-role sites.**
  - Location: `runner.py:128-131` and `data.py:132-142`.
  - `--smoke --split docs/preregistration/sonde_split_dev.json` passes the guard, because roles are taken
    from the dev split.
  - 30 of the 110 smoke sites are FINAL test-role sites, and 415 of the 564 FINAL test sites are
    train/val-role in the dev split.
  - Smoke metrics are printed, so this is a test-site peek even though nothing is registered. No existing
    run did this.
  - **Fix:** M6 has no need for non-final splits, so refuse them outright. Otherwise, drop or refuse every
    site whose FINAL role is `test`.
- **(b) The freeze does not cover the model code.**
  - `SCORING_CODE` (`guard.py:27-31`) omits `features.py`, `baselines.py`, `physics.py`, `lstm.py` and
    `runner.py`.
  - Pickled baselines run the *current* class code, so test predictions can change after the freeze
    without being detected.
  - `runner.score_and_write` and `runner.eval_cells` are scoring logic.
  - **Fix:** hash these files, either in `SCORING_CODE` or under a separate required `model_code_sha256` key.
- **(c) The scorer does not check where protected prediction files came from.**
  - Location: `hydromet_score.py:49-60`.
  - On protected data the scorer should require `header.artifact_sha256 == freeze.weights_sha256[name]`
    and `header.freeze_sha256 ==` the current freeze. Otherwise any npz carrying the right split hash is
    scored as test.

### Warnings (should fix before the freeze)

- **W1 (high). "Best of" comparator vs the plan's wording.**
  - H-G1 for M6 says "above GBM **and** physics-hybrid baselines". The code compares against one
    comparator selected on val (`evaluate.py:212-227`). On test, the non-selected baseline may be the
    stronger one.
  - The val selection is also dominated by single groups (C1).
  - **Fix (preferred):** an intersection–union rule. An endpoint is won only if Δ > 0 with CI excluding 0
    against **both** `forcing_gbm` and the physics hybrid (`air2stream_regional` /
    `dosat_ratio[air2stream_regional]`). This needs no extra multiplicity correction.
  - **Alternative:** log the reduction to a single comparator as a deviation in protocol §9.
- **W2. The comparator pool leaves out baselines.** `COMPARATOR_POOL` (`evaluate.py:172-179`) omits
  `air2water_linear` and `air2water_mohseni`. In smoke val they match `forcing_gbm` on 7-DADMax.
  Including them can only make the comparator stronger.
- **W3. The seed condition of protocol §7 is not implemented.** The rule "Δ > 0 for each of the 3 seeds"
  is not in `primary_comparison` or `hydromet_score.py:69-80`, so the "won" decision would be made by hand.
  Implement it in the frozen code and register it.
- **W4. Some static inputs are derived from gauges.**
  - `bficat`/`bfiws`: Wolock (2003) BFI grid, interpolated from USGS gauge streamflow records. At a gauge
    it partly encodes that gauge's own discharge.
  - `runoffcat`/`runoffws`: water-balance runoff.
  - `elev_m`: NWIS gauge datum. An unsensed reach would use a DEM value.
  - None of these is a WQ measurement, and MSST/MAST/MWST and prG_BMMI are correctly excluded. But the
    claim is "no sensor". **Fix:** pre-register a secondary ablation (drop BFI/runoff; use StreamCat
    `elevcat` for elevation) and report Δ.
  - IWI/ICI and the predicted channel dimensions are landscape or NRSA-geometry model products. That is
    acceptable; document it in the datasheet.
- **W5. Tuning-effort parity.**
  - `forcing_gbm` uses fixed hyperparameters, while the EA-LSTM may iterate on val (§15.5, up to 3 × 20 runs).
  - `max_rows_per_site: 1000` (`baselines.py:111`) gives the GBM about 20% of each site's train-block days,
    while the LSTM sees all of them.
  - **Fix:**
    - Before the freeze, run a small GBM grid on the **inner** groups that includes `max_rows_per_site`
      ∈ {1000, 3000, all}.
    - Match the number of val-evaluated configurations between GBM and LSTM, and report both counts.
- **W6. Missing tests.**
  - EA-LSTM end-to-end target invariance (the probe passes; add it to the suite).
  - The predict → score test path (it would have caught C2).
  - A guard test for a non-final split (C3a) and for prediction provenance (C3c).
- **W7. Code changed during the review.** `runner.py`, `evaluate.py`, `guard.py` and the protocol were edited
  at 10:15–10:19. The `inner_seed` fix (`runner.py:142-144`, the same inner groups for every seed and
  baseline) landed mid-review and was verified with smoke runs s0/s1 (identical inner sites). Re-review the
  final hashes at the freeze.

### Observations

- O1. `physics._ffill` back-fills leading NaNs with a later value. This only affects the 2005 spin-up and
  is harmless.
- O2. The 7-DADMax offset is fitted on fit sites for `air2water` and on fit + inner sites for
  `air2stream`. Harmless, but inconsistent.
- O3. The GBM refits on fit + inner sites, while the LSTM trains on fit sites only. This favours the
  baseline, which is acceptable.
- O4. The StreamCat land-cover (2019) and nutrient-input (2017) snapshots are used for 2008–2021. They are
  static context, not leakage.
- O5. Smoke EA-LSTM inner-val loss rises from epoch 0 (43 fit sites). This is expected at smoke scale, but
  check `patience`/`lr` at full scale.
- O6. The physics checks pass:
  - the Benson–Krause coefficients match APHA 4500-O;
  - DO_sat at 1 atm is 14.62/11.29/9.09/7.56 mg/L at 0/10/20/30 °C;
  - at 20 °C and 634 mmHg it is 7.55 mg/L;
  - pressure is (1 − 2.25577e-5·z)^5.25588, which gives 0.8345 atm at 1,500 m;
  - the torch and numpy paths agree;
  - the clamp to [0, 40] °C is applied before use.

### Checklist

| Section | Checked | Passed | Failed | N/A | Notes |
|---|---|---|---|---|---|
| 1.1 Architecture | 6 | 6 | 0 | 0 | EA-LSTM input gate static-only; physics-DO head wired; shapes OK |
| 1.2 Loss/optim | 5 | 5 | 0 | 0 | masked per-target Gaussian NLL/pinball; AdamW; grad clip |
| 1.3 Data pipeline | 7 | 7 | 0 | 0 | whitelist + regex refusal; raw → model-space transforms; 7-DADMax strict |
| 1.4 Training loop | 6 | 6 | 0 | 0 | train/eval modes, no_grad, best-state checkpoint, inner early stop |
| 2 Leakage | 5 | 5 | 0 | 0 | targets/normaliser/calibration train-only (tests + probe); causal; embargo; groups disjoint (W4 caveat) |
| 3 Experimental design | 6 | 2 | 4 | 0 | C1 unit, W1 comparator rule, W2 pool, W5 tuning parity |
| 4 Reproducibility | 5 | 4 | 1 | 0 | seeds/configs logged; freeze coverage incomplete (C3b) |
| 5 Code quality | 5 | 4 | 1 | 0 | missing end-to-end tests (W6) |
| Test guard / freeze path | 6 | 2 | 4 | 0 | refusals OK for training; C2, C3a–c |

**Re-review required** of `evaluate.py`, `hydromet_predict.py`, `hydromet_score.py`, `guard.py`, `runner.py`
and protocol §5–§7 after C1–C3 and W1–W3, before `FREEZE_hydromet.json` is written.

## M7 SENTINEL-Lite — 2026-10-24

**Review mode:** pre-training (before any non-smoke fit). **Reviewer:** an independent agent with a fresh context,
read-only, which ran its own checks on real data (scripts in the session scratchpad), plus fixes and a re-check by the
M7 workstream. **Scope:** `sentinel/models/lite/*.py`, `scripts/train/lite_*.py`, `scripts/eval/lite_*.py`,
`configs/lite/*.yaml`, `tests/test_lite.py`, `docs/preregistration/lite_protocol.md`, `lite_split.json`.

**Overall status: 🟡 PASS WITH CONCERNS → fixed, cleared for training.** The reviewer found no critical issues.
The workstream treated W1 as blocking, because it contradicts the headline "unseen water bodies" claim. W1 and all
other warnings were fixed before any real training. Two things stay open by design: W8 (teacher not yet available)
and the low power for turbidity/TSS.

### Critical
None.

### Warnings (status)
| # | Finding | Status |
|---|---|---|
| W1 | The 2-km water-body proxy split large lakes and bays across roles (≈ 60 test chips, e.g. Biscayne Bay, Lake Winnebago, Tampa Bay) | **Fixed.** Union of four merge rules: 2 km; same normalised name ≤ 40 km; same HUC8 for river/estuary sites (R4); same HydroLAKES lake. The split was rebuilt and re-frozen (v3 sha256 `46ed6e1f…`). Re-check: no HydroLAKES lake is in more than one of train/val/test; the 11 test chips that still share a crude name with train/val ≤ 40 km are distinct water bodies (e.g. "Mouth of Hillsborough River" vs "Mouth of Loxahatchee River"). |
| W2 | Holm ignored the direction of the difference | **Fixed.** One-sided "student better" p-value; a rejection counts only if the student is better (`lite_compare.py`). |
| W3 | The Secchi band-ratio baseline (linear Kloiber) is barely better than a constant | **Fixed.** A pre-registered family of forms (log, water-pixel); the best form on val is the comparator, which favours the baseline. |
| W4 | No model-selection score was pre-specified; val is noisy | **Fixed.** Protocol §5: selection score S = mean MdSA ratio vs GBM-RGBN; every configuration is counted (the §15.5 amendment lifts the cap). |
| W5 | `lite_compare` dropped unmatched rows silently and did not check split hashes | **Fixed.** It now asserts identical chip sets and the same frozen split sha256. |
| W6 | Labels were not covered by the freeze | **Fixed.** `labels_sha256` is part of the split hash, labels are written only after the check, and `role_frame` verifies them. |
| W7 | Distillation: `k_max` overflow crash; trust stored per cell instead of per target | **Fixed and tested** (`test_distill_kmax_and_per_target_trust`). |
| W8 | The teacher may have trained on Lite test lakes | **Open (teacher not ready).** Protocol §5 now requires a disjointness check when teacher targets are generated; teacher targets are only computed on train-role chips. |
| W9 | Smoke checkpoints could be registered | **Fixed.** `smoke` is stored in the config; export, robustness and compare refuse to register smoke runs. |
| W10 | `ExportWrapper` ignored `rgb_only` | **Fixed and tested** (`test_export_wrapper_rgb_only_matches_training_path`). |
| W11 | Censored rows were miscounted | **Fixed.** Counted before the finite filter: Secchi 46, TSS 8, chl-a 17, turbidity 1. The truncation of the clear-water end is stated. |
| W12 | The primary population differed from the plan | **Fixed in the protocol.** Secchi and chl-a are scored on lakes/reservoirs. Turbidity and TSS use all water bodies (lake-only test n = 58 and 62); this is a stated deviation, and lake-only results are secondary. |

### Observations addressed
- GBM baselines now average 3 seeds, matching the 3-seed student ensemble.
- `gloria_init` without the spectral branch now gives a clear error.
- `n_groups` KeyError fixed.
- The 0.5-pixel pooling offset and cuDNN non-determinism are accepted and noted.
- R4 flow-connection: HUC8 merging for flowing and estuarine sites.

### Verified OK (reviewer evidence)
- Split hash reproduces; labels reproduce from the matchups.
- Water bodies and sites are disjoint; the year cut holds.
- Scene statistics are label-free.
- Chips line up with their labels: centre medians within 3.5 DN of the extraction-time values.
- Normalisation statistics are fit on train only.
- FP32 ONNX matches torch (6e-8).
- Rotation/flip leaves the pooled output invariant.
- β-NLL, LR schedule and GradScaler order are correct.
- MdSA formula is correct; the bootstrap resamples water bodies.
- `pytest tests/test_lite.py`: 19 passed after the fixes (15 before).

### Checklist summary
| Area | Checked | Passed | Failed → fixed | Open |
|---|---|---|---|---|
| Code correctness | 14 | 11 | 3 (W7, W10, W11) | 0 |
| Data leakage | 10 | 8 | 1 (W1) | 1 (W8, teacher pending) |
| Experimental design | 10 | 5 | 5 (W2–W5, W12) | 0 |
| Reproducibility | 6 | 4 | 2 (W6, W9) | 0 |

---

## Downstream (Goal 4: D1 SafeSwim, D2 ToxinWatch, D3 KillAlert, D5 BioIntegrity, D6 MusselGuard, D8 RiverPulse) — 2026-10-24

**Status: 🔴 FAIL at review time — 6 critical issues in 4 applications; D1 SafeSwim and D3 KillAlert had none.**
Each application may start its first real training run only after its criticals are fixed and logged in its
protocol's deviations log (resolution table below). Nothing had been fitted, scored or registered for any
downstream application when the review ran (verified: 0 `downstream` rows in `results/registry.jsonl`,
empty `configs_tried.jsonl`, only `--build-only` artifacts on disk).

**Reviewed:** RESEARCH_PLAN §3, §8, §14, §15 (incl. the 2026-10-24 §15.5 amendment); docs/CONTRIBUTING_V2.md;
label datasheets; `sentinel/models/downstream/{common,weather,crossfit,safeswim,toxinwatch,killalert,biointegrity,musselguard,riverpulse}.py`;
`scripts/train/downstream_*.py`; `sentinel/data/downstream_links.py`, `scripts/data/build_downstream_links.py`;
`sentinel/data/sources/wbd_huc12.py`, `scripts/data/download_wbd_huc12.py`; `docs/preregistration/downstream_*.md`
and `*_split.json`; `docs/novelty_checks/downstream_killalert.md`; `tests/test_downstream_*.py`.
Method: three independent read-only reviewers (fresh context; D1+D5+shared, D2+D8, D3+D6) with probes in
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/review_downstream/`.
**Tests:** `tests/test_downstream_*.py` — common+safeswim 18 passed, toxinwatch+riverpulse 21 passed,
killalert+musselguard 16 passed.

### Critical issues (must fix before training)
| ID | App | Finding | Fix |
|---|---|---|---|
| C-D5-1 | D5 | StreamCat `bankfulldepth/bankfullwidth/wettedwidth/thalwegdepth` are Doyle et al. 2023 RF predictions (possibly trained on NRSA field data incl. 2018-19 sites) — an upstream prediction that may have seen evaluation sites (§8 cross-fitting, R4); IWI/ICI composites have the same risk | Excluded from every D5 feature set (`biointegrity.EXCLUDED_STREAMCAT`); also passed to D3/D8 |
| C-D2-1 | D2 | Lake Washington (one cluster, 43.8 % of site-days) is 80.6 % of test-lake rows; test holds 67 % of labelled site-days, train only 3,396 | Re-freeze as a new split with a rule chosen from row counts only (very large clusters → train role for lake assignment, scored as known-lake/future); old sha superseded and disclosed |
| C-D2-2 | D2 | Moses Lake South (val) and Moses Lake (test) are the same lake (name normalisation + > 2 km) | Alias + spot-check of large multi-site lakes before re-freezing |
| C-D2-3 | D2 | "New-lake" val/test rows share dates and gridMET cells with training rows (41.7 % of val-new rows share an exact cell × date) | Primary new-lake evaluation restricted to years after training ends (val 2019–2021; test 2022+) |
| C-D8-1 | D8 | Missing StreamCat (downstream_catchment not yet built) silently NaN for 24 sites incl. the 4 highest-GPP val sites; same config hash would later give different features | Driver refuses to train without `manifests/downstream_catchment.json` (or an explicit, recorded flag); manifest in DATA_MANIFESTS; per-site source counts in registry `extra` |
| C-D6-1 | D6 | `load_sources()` sets each 0.01° source bin's coordinates to the mean over all years, so features at origin T move with post-T records (≤ 0.54 km; 1,484 rows at T = 2015) — breaks R4 | Filter `year <= T` before binning; test with pre-/post-T records in one bin |

### Warnings (should fix; fixed before first fit unless noted)
- **Shared:** BLAS ignored the 8-thread cap (numpy imported before env vars) → `limit_threads` now uses
  threadpoolctl (verified: openblas 8). Split sha not enforced (`load_split` without `expect_sha`;
  `check_roles` skipped the split check when `split_sha` was None) → per-app `SPLIT_SHA` constants, and
  `allow_test` now requires `split_sha`. Scoring-code hash omitted label/cluster definitions and included
  other apps' modules → explicit per-app list `common.scoring_files(app)`. `dry_days_14d` counted missing
  days as wet → missing stays missing. `binary_scores` now exposes `n_boot_valid`.
- **D1:** registered baselines/models scored on different rows (official 2,288 vs model 2,411) → identical
  rows for claims + rescoring on comparator rows; sklearn HGB `early_stopping='auto'` silently on for D1a
  (> 10k rows) → `early_stopping=False`; research reference values (0.702/0.740) span test years → disclosed.
- **D5:** HUC8 bootstrap ≈ row-level (1.4 val visits per HUC8) → HUC4 claim unit, HUC8 secondary;
  primary comparison mixed model and feature gains → feature-matched RF added to the Holm family;
  persistence registered on revisited locations only; test-time training set / new-location / `nbr` pool
  pre-registered; test-cycle class counts disclosed; `wx_cell_km` dropped as a feature; `nani/nsurp`
  area-normalised.
- **D2:** INLA-style incumbent weaker than the published method class (no weather, fixed lake rate, untuned)
  → weather terms, lake random intercept / penalised lake effect, equal tuning budget; `val_known` included
  35 clusters without training rows → restricted; build report printed test positive counts → row counts only.
- **D8:** RF incumbent on a 120k subsample with median fill vs GBM on all rows with native NaN → parity;
  bootstrap by site although 36 val sites sit in 13 HUC4s → HUC4 unit; HUC4 1307 (val) directly downstream of
  1306 (sonde train sites) → flow-connected assignment (train role) for cross-fitting; "SENTINEL" label on a
  plain GBM (R5) → renamed.
- **D3:** long-term trend gives spurious matched skill (a year-only score reaches concordance 0.608 on val
  env_wq) → year-only placebo row + year term / detrended anomalies; gauge data start 2000-10-01 (missingness
  marks year 2000) → WY2001 start or structural-missing handling; placeholder 1 January dates (ice-out
  winterkills) → sensitivity only; inner CV not grouped by year → grouped; pre-registered 50-km CV, d−3 and
  estimated-location sensitivities, Holm vs each baseline, ≥ 3 seeds; referent windows not clipped to the
  block (latent spill into test) → clipped + test. **Power:** 24 val / ~38 test env_wq events (preliminary).
- **D6:** 3,669 HUC12s entirely in Canada/Mexico in the universe → dropped; inner-CV criterion ≠ primary
  endpoint (front-stratum AUPRC) → aligned; Kanankege 2018 full text unread and co-kriging/gravity
  comparators missing → added; novelty check needed; the Kanankege AUC 0.78 is not reproducible on our units
  (MN waterbodies, private boater data), so D6 cannot claim to beat that incumbent.

### Observations
- No test-label leakage through D3 referent exclusion (recomputed with pre-2018 events only: identical 27,256
  referents); the guard will still filter test events explicitly.
- Operational latency: gridMET/USGS daily values are provisional and late, so d−1 inputs are optimistic;
  d−3 sensitivity runs cover it (D2, D3).
- Some NRSA 2018-19 published COMIDs have implausible digit counts; validate against StreamCat coverage.
- No tests yet for `biointegrity.py` beyond the baselines or for `downstream_links.py`.
- Cause-of-kill attribution ("Environmental") is partly based on investigators' reading of the weather —
  circular for interpretation of the cause model.

### Verified correct (no action)
Split sha256 values match file bytes and protocols for all six apps; test-role rows are dropped before
feature building in every driver and `log_metric` refuses test splits; weather windows end d−1 (600-row probe,
0 mismatches) and climatologies use training years only; D1 persistence/network features strictly earlier,
same-day culture refused, official predictions joined on the right day (lag-0 Spearman 0.574 vs ≤ 0.23 at
±1 d); D2 d−7 embargo strict on sample date over the lake cluster; D8 has no own-site DO/temperature/discharge/
light predictors; D5 `nbr` uses training labels only with own-cluster exclusion; D3 conditional-logistic
likelihood/gradient and tie-aware concordance correct; D6 labels in (T, T+3], candidates exclude units
infested by T, no training feature or label dated ≥ 2016; AUPRC = average precision; paired bootstrap uses
the same resampled groups; `cpu_slot` locking correct.

### Resolution table (updated as fixes land)
| ID | Status | Evidence |
|---|---|---|
| C-D5-1 | ✅ fixed 2026-10-24 (before any D5 fit) | `biointegrity.EXCLUDED_STREAMCAT`; protocol §8 deviations |
| C-D2-1, C-D2-2, C-D2-3 | ✅ fixed before any D2 fit | new split `downstream_toxinwatch_split_v2.json` sha `13852e4a…ef85` (Lake Washington → train role; balanced by site-days; v1 `04e14820…` superseded, test-count disclosure); Moses Lake alias; primary new-lake val 2019–2021; protocol §7 |
| C-D8-1 | ✅ fixed before any D8 fit | driver refuses without `manifests/downstream_catchment.json` (`downstream_riverpulse.py:183`); manifest built 2026-10-24 23:30; split v2 `20c550f9…b0d8` (HUC4 1307 → train, W6) |
| C-D6-1 | ✅ fixed before any D6 fit | `musselguard.py:78-79` bins only records with `year <= T`; regression test added |
| Shared warnings | ✅ fixed | `common.py`, `weather.py`; `tests/test_downstream_common.py` 8 passed |

### Checklist summary
| Category | Checked | Passed | Failed (at review) | N/A |
|---|---|---|---|---|
| Code correctness | 18 | 14 | 4 (HGB early stopping, thread cap, split sha, scoring hash) | 0 |
| Data leakage | 14 | 9 | 5 (C-D5-1, C-D2-2, C-D2-3, C-D6-1, D8 flow connection) | 0 |
| Experimental design | 16 | 8 | 8 (C-D2-1, row parity, bootstrap units ×2, incumbent parity ×2, trend placebo, claim family) | 0 |
| Reproducibility | 8 | 6 | 2 (C-D8-1 silent feature change, missing sensitivities) | 0 |
| Code quality | 6 | 5 | 1 (missing tests D5/links) | 0 |


---

## M3 SENTINEL-Microbe — 2026-10-24

**Review mode:** Pre-Training. **Reviewer:** independent review sub-agent (read-only, adversarial) + workstream fixes.
**Scope:** `sentinel/models/microbe/`, `sentinel/data/microbe_wq.py`, `scripts/data/build_microbe_wq.py`,
`scripts/train/microbe_{cv,controls,compare,embed,freeze_split}.py`, `scripts/eval/microbe_dnabert_parity.py`,
`configs/microbe/`, `tests/test_microbe*.py`, `docs/preregistration/microbe_protocol.md`, frozen split
`docs/preregistration/microbe_split.json` (sha256 at review `4fac5438…`; re-frozen after the upstream data change to
`aaeb05305afd8df9a1cc677f974d443b5417b1acdbd95ef1160368dd120626f8`, protocol §10).

**Initial verdict:** 🟡 PASS WITH CONCERNS (2 critical, 9 warnings, 9 observations). **After fixes:** 🟢 PASS for the
primary LODO path, conditional on the join audit (C2b) being run and inspected after the chemistry/sonde joins and
before the first training run.

### Critical issues
- **C1 — the cross-primer/cross-release alternative primary had no untouched test set** (all 707 .001 events were
  development). **Fixed:** .001 events before 2017-01-01 (190 events, 15 domains, 24 sites) reserved as a cross-release
  final test (`configs/microbe/common.yaml: release_holdout`, enforced in `design.load_design(with_release_v1=True)`);
  Holm over 8 targets × 2 primary routes (16) for H-M3 (protocol §10).
- **C2 — chemistry/sonde joins untested and never run on real schemas.** **Fixed (a):** `tests/test_microbe_wq.py`
  (synthetic frames in the NEON processed schemas: replicate summing, local-date, median primer rule incl. chloroplast
  reads, parentSampleID match, duplicate-parent choice, censoring → detection limit, same-station ±24 h fallback,
  mg/L-only chemistry variables, sonde HOR/local date/±1 d ablation). **Open (b):** after the targeted build, write and
  inspect the join audit (chem_match counts, Δt, censored per analyte, sonde non-missing per site × HOR, `.AOS.reach`
  events) — `scripts/train/microbe_freeze_split.py --dry-run --audit-out …`; confirm `load_design()` still passes the
  frozen-split guard.
  **Done 2026-10-24 ~23:20:** the audit found that `neon_chem/chem_event_wide` keeps per-bottle rows
  (`*.RAW.NUT`/`*.FIL.NUT`/`.2`), so post-2019 TN/TP/NO3 were missed by an exact parentSampleID join → fixed by
  collapsing to the field visit (`microbe_wq.visit_key`, tested); sonde targets restricted to S2/buoy events; the
  upstream rebuild of `neon_microbe_16s` changed the event set, so the split was re-frozen under the unchanged rule
  (sha256 `aaeb0530…`, protocol §10). Audit: `docs/preregistration/microbe_join_audit.json`.

### Warnings (all addressed before training)
- W1 permutation null permuted over all rows and used 1 seed vs 3 → now within each fold's training rows, same seed rule,
  `--tag` respected (`cv.permuted_training_targets`, `microbe_controls.py`; test added).
- W2 `microbe_compare.py` crashed on release schemes → loads the v1 design per scheme.
- W3 primer length rule mislabelled 4 BARC events (long chloroplast reads) → read-weighted **median** length.
- W4 ±24 h chemistry fallback could take inlet/outlet chemistry → requires same named location.
- W5 site-unit CI only, 6 domains with 1 site → H-M3 needs positive, Holm-significant differences with **both** site- and
  domain-cluster bootstraps (`microbe_compare.py`).
- W6 tuning asymmetry → pre-registered RF/XGB grids with site-grouped inner CV (`rf_clr_tuned`, `xgb_clr_tuned`), run
  before any claim; asymmetry reported.
- W7 vacuous feature-leak test and missing integrity tests → rewritten; tests added for id-vocabulary-from-fit-rows,
  within-fold permutation, release folds, frozen-split guard.
- W8 DNA-encoder numbers in the protocol without computing code → `scripts/eval/microbe_dnabert_parity.py` (registry
  experiment `microbe_encoder_checks`); unsupported fp16 sentence removed.
- W9 comparator wording → "season + location of an unseen domain"; `SiteMean+DOY` reported as a context row in the
  time-out test.

### Observations
O1 fp16 default → fp32 default + weights sha256 verified at load. O3 TabPFN now pools genera beyond 300 into "other".
O5 SeqSet early stopping on whole training domains under domain hold-out. O2 (guard checks ids only), O4 (no RDP genus
confidence threshold), O6 (manifest lists), O7 (`.AOS.reach` events), O8 (LECO time-zone label, data workstream),
O9 (2024-only holdout; `scripts/eval/microbe_final.py` to be written at freeze) — noted; O4/O7 reported in the join audit.

### Verified correct (reviewer)
DNABERT-S re-implementation line-by-line vs reference (Wqkv layout, scaling, ALiBi slopes incl. 12 heads, −10000 mask,
GEGLU, post-LN, no position embeddings, padding disabled, masked mean pooling); k-mer profile; SeqSet masking/pooling/
masked MSE/early stopping/eval modes; train-fold-only standardisers and id vocabulary; GenusCLR, context fill, grouped
Ridge, site_mean_doy; anomaly/per-domain R² and paired closure; LODO/LOSO site-disjoint, release folds domain-disjoint;
.001 and .002 share no dnaSampleID/parentSampleID/site×date; field DO/SC/T never carried into the design; holdout only
via `allow_holdout` in the freeze script; split frozen on a target-free build, cut moved 2023 → 2024 per §2.3;
seeds and pinned requirements.

### Checklist summary
| Category | Checked | Passed | Failed → fixed | Open |
|---|---|---|---|---|
| Code correctness | 14 | 12 | 2 (W3, O3) | 0 |
| Data leakage | 9 | 8 | 1 (W1) | 0 |
| Experimental design | 10 | 5 | 5 (C1, W5, W6, W9, O5) | 0 |
| Data pipeline / joins | 6 | 3 | 2 (W4, C2a) | 1 (C2b join audit) |
| Reproducibility / provenance | 6 | 4 | 2 (W8, O1) | 0 |
| Tests | 5 | 2 | 3 (W7) | 0 |
---

## M4 SENTINEL-ChemTox v2 (prospective test set + improvement cycle 2) — 2026-10-24

**Status: 🔴 FAIL — do NOT run `scripts/eval/chemtox_score_prospective.py` (and do not write stage 2) until C1–C3 are fixed and re-reviewed.**

**Scope of the verdict.** The prospective file itself is sound. It is a genuine held-out set: there is zero
CAS, InChIKey or `result_id` overlap with all 158,330 ADORE release rows, and every published number in §1.3 was
reproduced from the parquet. The cycle-2 CV machinery is also leak-free: every transformer is fit on training
rows only, extra rows lose the validation fold's chemicals, and scores come from ADORE validation rows only.
CV searches may therefore continue.

Three things block the one-shot scoring run:
- **C1.** The scorer crashes *after* every RMSE has been written to the log. This is the failure pattern of the
  ADORE pass (`chemtox_test_plan.md` §6b).
- **C2.** The §4.3 feature-spec rule compares the wrong thing on CV. At test time it systematically handicaps
  whichever models (including the frozen v1 baselines) use `chem_ws`/`chem_mp`.
- **C3.** About 29% of primary-track rows are salts, stereoisomers or metal complexes of ADORE chemicals. The plan
  does not disclose this and has no pre-registered sensitivity subset. It has to be pre-registered *before*
  scoring.

**Reviewed:** RESEARCH_PLAN.md §3, §15.1, §15.5; docs/CONTRIBUTING_V2.md;
docs/preregistration/chemtox_prospective_plan.md, chemtox_test_plan.md;
`sentinel/models/chemtox/{prospective,cycle,gbdt,gpboost_model,readacross,chemeleon_fp,data,features,xgb,evaluate,aux,ensemble}.py`,
the relevant parts of `chemprop_model.py`; `scripts/data/chemtox_build_prospective.py`, `scripts/train/chemtox_cycle.py`,
`scripts/eval/chemtox_cycle_select.py`, `scripts/eval/chemtox_score_prospective.py`;
`sentinel/core/registry.py`, `sentinel/eval/bootstrap.py`; `tests/test_chemtox_prospective.py`; the frozen parquet,
`report.json`, `build.log` and `$SENTINEL_DATA/manifests/chemtox_prospective.json`; and the cycle-2 run directories
(searches in progress: xgboost, lightgbm, gpboost and readacross, all `extra-none`).

**Files changed while the review was running:** `chemtox_score_prospective.py` (22:30, sha256 `ace5c65e…`) and
`cycle.py` (22:32). Findings refer to those versions. One defect I found in the earlier scorer version was fixed by
that edit (R1 below).

**Tests:** `pytest tests/test_chemtox_prospective.py` gives **16 passed**. No test exercises the scoring script.
Probes (light, no fits, `OMP_NUM_THREADS=2`) are in
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/`:
- `recompute.py` (track statistics)
- `salts.py` and `metal.py` (parent-structure overlap)
- `aug.py` (augmentation overlap)
- `clip.py` (descriptor z-scores)
- `cutoff.py` (ADORE year cut-off, Tanimoto)
- `crash.py` (`already_scored`)

### Critical issues (must fix before the scoring run)

**C1. `already_scored` crashes with `TypeError`, and it is reached only after every model's prospective RMSE has
been logged.**
`scripts/eval/chemtox_score_prospective.py:119-122`, called at `:394`; the metrics are logged at `:378`.

`registry.lookup(experiment, model, metric, split, *, target, registry_path)` takes no `workstream` keyword, and
`metric`/`split` are required. Probe `crash.py` gives
`TypeError: lookup() got an unexpected keyword argument 'workstream'`. Even with the arguments fixed, `lookup`
raises `KeyError` when no entry exists (`registry.py:186-194`), which is exactly the first-time case.

The sequence on a real run would be:
1. Every model is fit.
2. Every track's RMSE and CI is printed to `score.log` and stdout (`:378-380`).
3. The first `already_scored` call raises.
4. Nothing reaches the registry, and `prospective_results.json` is never written.

So partial scores are seen and are not recorded. That is the §6b failure mode a third time. The refusal check is
also in the wrong place: it should prevent a re-score, not run after the scores exist.

- **Fix:**
  - Implement it as a filter over `registry_frame()`, e.g. `df[(experiment==EXPERIMENT)&(model==m)&(split==want)]`.
  - Run it for every (model, track) **right after the seals, before any data are loaded or any model is fit**, and
    exit non-zero if any is already scored without `--allow-rescore`.
  - Add an end-to-end test of `main()` on a temporary registry, with a tiny fake prospective file and two cheap
    models, in both dry-run and real mode.

**C2. The §4.3 "feature-spec question" is not answered by the CV that is supposed to answer it. The resulting test
condition is biased between models.**
`sentinel/models/chemtox/cycle.py:45-51` (`chemical: ["four", "mw_logp"]` as a free search dimension),
`cycle.py:337-343` (`run_cv_extra`), `features.py:145-153` (NaN → training mean), plan §1.1 deviation 2 and §4.3.

On ADORE CV, every validation row has its **real** experimental `chem_ws`/`chem_mp`. On the prospective set they
are **100% missing** and imputed to the training mean (`prospective.py:331-332`). The CV comparison of `four`
against `mw_logp` therefore measures the value of *having* WS/MP, not the cost of *imputing* them. It is biased
towards `four`, which is exactly the spec that degrades at test.

The plan's wording "(i) the frozen phase-1 spec with `chem_ws`/`chem_mp` imputed to the training mean" is not
what the code does: nothing imputes them on CV. A `grep` for any masking of `chem_ws`/`chem_mp` in
`sentinel/models/chemtox` and `scripts` finds none.

The consequence is not only lower accuracy. It biases the **comparison**:
- The frozen phase-1 chemprop (`select/t-F2F_split_occurrence/selection.json`: `features: {fingerprint: null,
  published: true}`, i.e. `chemical="four"`) and any baseline frozen with `four` see an out-of-distribution
  covariate for every prospective row.
- A cycle-2 model that happened to draw `mw_logp` does not.

This breaks RESEARCH_PLAN §15.5 guard 4 (baselines must not be weakened) in a way that depends on random search
draws.

- **Fix (before any further freeze):** in `run_cv_extra`, and in any CV used to choose a spec, set
  `chem_ws`/`chem_mp` to NaN on the **validation** rows whenever the spec uses them. That reproduces the
  prospective condition: the training mean via `PublishedFeatures`. Then either:
  - (a) apply the §4.3 rule per family, **including** `xgboost`, `rf` and `chemprop_v1frozen`, using these masked
    CV scores; or
  - (b) pre-register `mw_logp` for every model.

  Report both specs' CV numbers. Update §1.1 deviation 2 and §4.3 to describe what the code actually does.

**C3. 65 of 301 primary-track chemicals (552 of 1,912 rows, 28.9%) are salts, stereoisomers or metal complexes of
ADORE chemicals. The plan calls the track "genuinely novel chemistry".**
`prospective.py:357-358` (novelty = CAS ∪ InChIKey), plan §1.2 and §1.3 ("InChIKey-level overlap with ADORE is 0
by construction … genuinely novel chemistry, not a re-labelling of ADORE's").

Probe `salts.py` compares each chemical's parent (RDKit largest fragment, uncharged) against ADORE's 2,408
chemicals:

| Match type | Primary chemicals |
|---|---|
| Full parent InChIKey | 48 |
| Parent connectivity block | 65 |
| Raw InChIKey connectivity block | 16 |

Examples:
- aniline hydrochloride (142-04-1) ↔ aniline (62-53-3)
- the four single stereoisomers of difenoconazole (1161016-80-3/-82-5/-84-7/-86-9) ↔ racemic difenoconazole
  (119446-68-3)
- (R)-propranolol HCl ↔ propranolol HCl
- the (S)-cypermethrin ester ↔ cypermethrin
- Zn/Cu/Fe dimethyldithiocarbamates ↔ Na/K dimethyldithiocarbamate
- picric acid ammonium salt ↔ picric acid
- NTA/DTPA sodium salts ↔ the free acids

Against the actual training set (all t-F2F, 1,905 chemicals), **18 primary chemicals (6.0%) have Morgan Tanimoto
1.0** to a training chemical (`cutoff.py`).

This is not row leakage, and the CAS/InChIKey rule is what ADORE itself uses. But for these chemicals the "novel
chemical" claim is not true in any toxicological sense, and a post-hoc "parent-novel" subset would look like
choosing the endpoint after the fact.

- **Fix (cheap, pre-scoring):**
  - Add a frozen `parent_inchikey` / `parent_novel` flag (largest-fragment, uncharged, connectivity-block key; the
    function in the module, the key list hashed) and pre-register `fish_novel ∩ in_adore_domain ∩ parent_novel`
    (≈236 chemicals / ≈1,360 rows) as a **co-primary or mandatory sensitivity** track.
  - Extend `assert_prospective_chemicals_absent` / `augmentation_rows` to parent keys (see W13).
  - Correct the §1.2 and §1.3 wording.

### Resolved during the review

**R1 (would have been critical).** In the scorer version read first (sha256 `48978b3c…`), `paired_delta_ci` used
`bootstrap_ci`, which returns no `p_value`. Every `vs_xgboost.get("p_value")` was therefore `None`, and the Holm
correction was **silently skipped**, so the pre-registered decision rule could not be evaluated. The 22:30 version
uses `paired_bootstrap(..., higher_is_better=False)` and `holm_bonferroni`. Both are verified correct below.

### Warnings (should fix; all before stage 2 unless noted)

**W1. Plan §1.4 row count is wrong:** it says 11,497, but the file and `report.json` have **11,500** rows
(2,313 + 295 + 3 + 8,774 + 115).

**W2. The stage-1 hash table and manifest path are stale.**
- The plan (§1.4, §6) records `prospective.py` sha256 `29d6c5b3…`. The current module is `082e7d3f…`, modified
  22:23, *after* the 22:00 freeze.
- The manifest lives at `$SENTINEL_DATA/manifests/chemtox_prospective.json`, not `manifests/` in the repo.
- `build.log` shows five builds with four different hashes. The last two are identical, consistent with "verified
  twice".

The data hash still matches, so the frozen rows are unaffected, but the scorer now uses the *edited* module
(`prospective_keys`). **Fix:** re-run `P.build()` into a scratch directory without `write_manifest`, confirm it
reproduces `ea078629…`, then record the new module hash and a one-line description of the edit.

**W3. The domain-of-applicability numbers describe the wrong subset and reference set.** The 0.425 / 0.218 /
0.800 / 43% figures are for all 363 `fish_novel` chemicals against the 1,588 *train-partition* chemicals. Recomputed:

| Primary chemicals (301) vs | Median | p10 | p90 | Below 0.4 | At 1.0 |
|---|---|---|---|---|---|
| Train partition | 0.444 | — | — | 40.2% | — |
| Real training set (all t-F2F, 1,905) | 0.476 | 0.229 | 0.800 | 38.9% | 6.0% |

Report these numbers and drop "genuinely novel".

**W4. Pre-registered secondary tracks are never scored.** `track_mask` (`score:125-130`) always intersects
`fish_novel` and `crust_novel` with `in_adore_domain`. The §1.3 tracks "`fish_novel` (all carbon-containing)"
(2,313 rows) and "`crust_novel` (all)" (295 rows) therefore produce no numbers. **Fix:** add explicit track names,
e.g. `fish_novel_all` and `crust_novel_all`.

**W5. The model names and baseline definitions in plan §3 do not exist in the code.**
- The frozen keys are family names: `xgboost`, `rf`, `chemprop`, `ensemble_mean`, `ensemble_stack_nnls`. They are
  not `xgboost_maccs`, `rf_maccs`, `chemprop_v1frozen` or `ensemble_nnls_v1frozen`.
- The frozen `xgboost` is the winner of a 32-config search over `TABULAR_FEATURES` (MACCS *or Morgan*, ± RDKit
  descriptors, `four` or `mw_logp`). It is therefore not necessarily "MACCS + published block", i.e. no longer the
  reproduced published baseline.
- The cycle-2 stack is rebuilt from whatever members are passed. It is not "the v1 stack".

**Fix:** freeze, under distinct names, (a) the literal published-spec baselines (MACCS + published37 with the
phase-1 frozen params) and (b) the equal-budget-tuned variants. State which one is the Holm comparator and the
§4.4 reference.

**W6. Budget accounting can under-report configurations.** `chemtox_cycle_select.py:116-127` picks the best
across *all* `extra-*` variants of a family, but `:139` records `n_configurations = args.budget` (32). A family run
under k variants actually gets 32k configurations. Also, ~10 cycle-2 families × 32 configs compete against
XGBoost's 32. That is allowed on a fresh test set, but §15.5 guard 2 requires the total to be stated.
**Fix:** record the real count per family and the cycle total. Run the baselines under the same extra variants,
or select within a fixed variant.

**W7. Several CLI options sit outside the seals, and the sensitivity entry collides with the main entry.**
`score:80-92`.
- `--seeds`, `--models`, `--tracks`, `--n-boot` and `--train-partition-only` are not covered by the config hash.
  `--models` also changes the Holm family.
- The `--train-partition-only` sensitivity pass logs to the same experiment/model/split, so once C1 is fixed it is
  refused; with `--allow-rescore` it creates a duplicate.
- `log_result` is never passed `supersedes` (`:399-417`), although the docstring says it is recorded.

**Fix:** refuse overrides that differ from the frozen block. Log the sensitivity pass under a distinct split, e.g.
`prospective/<track>/test@train_partition`. Pass `supersedes`.

**W8. The Holm family and the "all 3 seeds" rule do not match §4.2/§4.4.**
- Holm runs over every scored entry, including trivial baselines, `rf` and the ensembles (`:382-388`). That is
  conservative, but it is not "the per-model primary comparisons".
- The §4.4 / §15.1 requirement that the gain holds in all 3 seeds is never computed. Only the **seed-averaged**
  prediction (itself a 3-model ensemble) is compared, and `seed_rmse` carries no paired CI.

**Fix:** pre-register the Holm family explicitly in stage 2. Compute `paired_bootstrap` per seed against
`xgboost` and report it.

**W9. The dry run does not exercise `fit`/`predict`.** All 11,500 rows are predicted, including 8,774 `noncarbon`
rows with single-ion SMILES, through chemprop (which raises on unparseable SMILES, `chemprop_model.py`
`_datapoints`), CheMeleon and read-across. A predict-time crash happens before any target is read, so it is not an
integrity breach, but it burns the pass. **Fix:** in `--dry-run`, fit each frozen model on a small ADORE subset
and predict every `pro` row, without reading `result_conc1_mean_mol_log`.

**W10. A pre-registered ensemble can silently disappear.** `nnls_weights` returns `None` and the stack is skipped
with a warning if any member's OOF file is missing (`score:338-340`, `:445-446`). An ensemble with fewer than two
available components is also skipped (`:331-333`). **Fix:** check that every frozen component and OOF file exists
**before** fitting, and exit if any is missing. Also document that NNLS has no intercept and its weights are
renormalised to sum to 1 (`:452-453`), which changes the fitted combination slightly.

**W11. The CheMeleon cache can be corrupted by parallel shards.** `chemeleon_fp.py:56-61` writes the fixed path
`chemeleon_fp.tmp.npz`, so concurrent shards can interleave writes or lose updates. The cache also stores no hash
of the encoder weights. **Fix:** use a per-process temp name plus a file lock, and store and check the
`chemeleon_mp.pt` sha256 in the npz.

**W12. The `CLIP_SD=50` docstring claim ("no ADORE row is affected") is false for `rdkit_descriptors_full=True`.**
`data.py:165-176`. Probe `clip.py` (t-F2F, 26,114 rows):

| Descriptor list | Fit | Max \|z\| (descriptor) |
|---|---|---|
| Short | training partition | 12.3 |
| Short | CV folds | ≤ 14.8 |
| Full | training partition | **148** (`fr_benzodiazepine`) |
| Full | CV fold 2 | 298 (`Kappa3`) |
| Full | CV fold 3 | **399** (`fr_HOCCN`) |

The full list also has **8,928 non-finite raw values**, which become 0. No registry entry used the full list, so
no registered number changes. **Fix:** correct the docstring. Consider clipping raw values, or a robust scaler for
the `fr_*` count descriptors.

**W13. Augmentation rows carry parents of primary chemicals.** Probe `aug.py`: 23 augmentation chemicals
(134 rows) share a parent with a primary chemical. Examples are the difenoconazole racemate, glyphosate plus its
isopropylamine salt, PFOS plus its K salt, and pentachlorophenol Na salt. If any `extra=ecotox` configuration is
frozen, C3 gets worse. **Fix:** tie this to C3 by excluding parent keys in `augmentation_rows`.

**W14. The scoring script has no tests.** Nothing covers the seals, `already_scored`, Holm, the paired CI,
`track_mask`, `nnls_weights` or the trivial/ensemble block. That is the class of bug that crashed the ADORE pass
twice, and C1 would have been caught by one test. **Fix:** see C1.

**W15. GPBoost with `sample_weight` is unverified.** `gpboost_model.py:88-91` passes `weight=` to `gpb.Dataset`
together with a `GPModel`. Check that gpboost 1.8.0 honours weights in the GPBoost algorithm rather than raising
or ignoring them. This path is reached only when `--extra-weight ≠ 1`. **Fix:** add a tiny test.

**W16. DataFrame-valued chemprop params are stripped at freeze.** `chemtox_cycle_select.py:158-159` drops
`aux_rows`/`ecotox_rows`. A phase-1 configuration with `multitask=True` would then be scored as a different model
(no auxiliary rows), and one with `pretrain='ecotox'` would raise mid-pass. The prospective guard is also never
applied to rows passed through `params`. This is latent: the frozen phase-1 chemprop uses neither. **Fix:** refuse
to freeze such configurations, or rebuild those rows through `load_extra_rows`, which applies the guard.

### Observations

- **O1.** `report.json` `track_overlap_rows = 0` is computed on duplicates of (`result_id`, track), which are zero
  by construction (`prospective.py:391`). In fact **63 rows** sit in two tracks (`fish_novel` ∩ `temporal`).
  Report the cross-track count.
- **O2.** The ADORE cut-off is confirmed: 32,609 ADORE rows matched in ECOTOX, max publication year 2021, 0 rows
  after 2021. The plan's "2,432 references, p99 2019" uses a different denominator: row-level gives 2,395
  references and p99 2016. State the denominator.
- **O3.** 839 ADORE `result_id`s (2.5%) are absent from the 2026-09-15 ECOTOX release, so the `result_id` check
  cannot see an ADORE row re-issued under a new id.
  - The novel tracks are protected by the chemical exclusion, and the temporal track by the year.
  - Re-issued ADORE rows could appear among the augmentation rows. That is harmless: they are CV-safe through the
    per-fold chemical drop, and only duplicate training data in the prospective pass.
- **O4.** Primary-track composition: 958 of 1,912 rows (50%) are `organic_multicomponent`, against 8.6% of ADORE's
  chemicals. 596 rows (82 chemicals) are `has_metal`, including Cu/Zn/Mn dithiocarbamates, Paris green and copper
  8-quinolinolate. These are in-domain by the stated rule (ADORE has 96 metal-bearing multicomponent chemicals),
  but the §4.1 breakdown should split `has_metal`.
- **O5.** Read-across recasts the whole training pool to float32 on every query (`readacross.py:160-168, 226`),
  roughly 11.5k × 26k × 2048 per pass. Precompute it once. `sample_weight` is silently ignored.
- **O6.** The scorer's `EXTRA_COLUMNS` (`:65-66`) lacks `reference_number`, so the temporal-reference part of the
  guard is skipped on ADORE frames. That is vacuous here (ADORE ≤ 2021 < temporal), but the guard reports less
  than it claims. The cycle runner does include the column.
- **O7.** The temporal track (29 chemicals) is labelled underpowered in plan §1.3 and §4.4 and in the registry
  `notes` (`score:412-414`). That labelling is adequate.

### What was verified as correct (no action)

- **Frozen file, recomputed from the parquet (`recompute.py`):**
  - Totals: 11,500 rows; sha256 `ea078629…` matches the plan, `report.json` and the manifest; builder sha256
    `80a617ae…` matches.
  - Primary track: **1,912 rows / 301 chemicals** (CAS ↔ InChIKey 1:1, no nulls) **/ 64 species / 310
    references**; **y_sd 1.7304**; **replicate floor 0.4519** (361 groups / 1,180 rows); classes 954 organic +
    958 organic_multicomponent; no missing MW, logP, MACCS or Morgan.
  - Every §1.3 track size matches (2,313/363/71/364; 230/112/9/98; 295/136/10/114; 3/2/2/2; 8,774/225/114/1,136;
    115/29/13/44), and so does the curation funnel.
  - `in_adore_domain` is a stored boolean column, so the primary rows are fixed by the file.
- **Exclusion:**
  - The novel tracks have 0 CAS, 0 InChIKey and 0 `result_id` overlap with the full ADORE release (filtered table
    plus all 11 challenge files); asserted per track at build time and tested by
    `test_novel_tracks_share_no_chemical_or_row_with_adore`.
  - The temporal track's minimum year is 2022 (`test_temporal_track_is_after_the_adore_cutoff_and_not_an_adore_row`).
  - Species are restricted to ADORE's 203, and taxonomy is asserted species-constant.
- **Guard wiring:**
  - `prospective_keys` is track-aware: novel-track CAS/InChIKey, every prospective `result_id`, temporal
    references.
  - `assert_prospective_chemicals_absent` runs in `augmentation_rows`, in both `load_extra_rows` paths and in
    `build_training_frame`, before any fit.
  - Tests: `test_guard_raises_on_a_prospective_chemical`, `test_guard_passes_on_adore_training_rows`,
    `test_augmentation_rows_are_clean_and_not_adore_rows`.
- **CV hygiene:**
  - `run_cv_extra` drops validation-fold chemicals (CAS and InChIKey, positional) from the extra rows.
  - The `FeatureBuilder` is fit on training (+ extra) rows only, and scores come from ADORE validation rows only.
  - Selection uses full 5-fold records only (`full_records`, `min_folds=5`).
- **Transformers:** `InformativeBits`, `PublishedFeatures`, the RDKit block, `StandardizedColumns` and
  `CheMeleonBlock` (standardiser and PCA) are all fit inside `FeatureBuilder.fit` on training rows.
  - In the scorer, every builder is fit on `train_rows`, and the prospective frame is only transformed.
  - Trivial baselines are fit on training rows.
  - NNLS weights are fit on training-partition OOF arrays; prospective rows cannot reach them, and a missing OOF
    file returns `None` without crashing.
- **Models:**
  - GPBoost: `response_mean` is the right field with `pred_latent=False`. Unseen levels become a new level with a
    zero random effect (`test_gpboost_handles_a_taxon_it_never_saw`). The groups are taxonomy-only, so the random
    effects cannot absorb chemical identity.
  - LightGBM, CatBoost, XGBoost and RF plumb `sample_weight` and set seeds.
  - Read-across cannot match its own row on either path, and ragged or missing bit strings fall back cleanly
    (tests `test_readacross_*`, `test_bits_matrix_and_tanimoto`).
  - CheMeleon uses `.eval()` and `no_grad`, the v2 multi-hot featuriser, a deterministic per-SMILES cache and
    ±50 clipping (`test_chemeleon_block_is_deterministic_and_standardised`).
- **Search:**
  - `sample()` is deterministic with seed 20261024 and deduplicates draws
    (`test_cycle_search_spaces_are_deduplicated_and_reproducible`).
  - Every family gets 32 draws.
  - The XGBoost space covers the published hyper-parameters (minus γ = 0) at least as finely as phase 1's
    `XGB_GRID`, and the RF space contains phase 1's `RF_GRID`. The new families' spaces are larger, but the draw
    budget is equal.
- **Seals and statistics (22:30 version):**
  - Approval is checked before anything is read, and the data hash before the parquet is read.
  - The frozen hash is computed identically in the selection and scoring scripts.
  - Targets are read only after all fits (`:354`), and the dry run never reads them.
  - The paired CI resamples chemical clusters on identical rows.
  - Holm step-down (threshold α/(m−i), stop at the first non-rejection) is correct.
- **Provenance:** `gpboost==1.8.0` is pinned in `requirements-v2.txt:44` and `pyproject.toml:47`; installed
  versions are gpboost 1.8.0, lightgbm 4.7.0 and catboost 1.2.10. The registry has no `chemtox_prospective_test`
  entry yet, so nothing has been scored. Phase-1 frozen chemprop uses no ECOTOX pretraining, multitask or
  foundation init, so the `ecotox_mp_*.pt` encoders cannot reach the prospective pass.

### Checklist summary

| Category | Items checked | Passed | Failed | N/A | Notes |
|---|---|---|---|---|---|
| 1 Code correctness (scorer, models, CV loop) | 16 | 12 | 4 | 0 | C1 crash; W9 dry-run coverage, W10 silent ensemble skip, W15 GPBoost weights |
| 2 Data leakage (rows, chemicals, transformers, stack weights) | 12 | 10 | 2 | 0 | C3 parent-structure overlap (not row leakage), W13 augmentation parents |
| 3 Experimental design (fair baselines, budgets, metrics, statistics) | 12 | 6 | 6 | 0 | C2 WS/MP condition mismatch, W4 tracks, W5 baseline identity, W6 budget count, W7 unsealed CLI, W8 Holm/seed rule |
| 4 Reproducibility (seeds, pins, hashes, manifests) | 8 | 6 | 2 | 0 | W2 stale module hash and manifest path, W11 cache race |
| 5 Code quality and tests | 4 | 2 | 2 | 0 | W14 untested scorer, W12 false docstring |
| Documentation honesty (pre-registration vs code) | 10 | 6 | 4 | 0 | W1 11,497 vs 11,500, W3 DoA subset, C3 "genuinely novel", §4.3 wording (C2) |

**Re-review required** for the scorer (C1, W7–W10, W14), for the §4.3 masked-CV implementation and its results
(C2), and for the parent-key track definition and its pre-registration text (C3), before stage 2 is appended and
before `--i-have-approval` is used.

### Round 2 re-review (fixes to C1-C3 and W1-W16) — 2026-10-24

**Verdict: 🔴 FAIL. Do not freeze stage 2 and do not use `--i-have-approval` yet.**

Two of the three round-1 blockers are now correctly implemented at the mechanism level:
- **C1 (crash after scoring) is resolved.**
- **C2's deployment masking is correct** inside the cycle-2 CV.

The fixes did introduce new defects, and parts of C2, C3 and W10 are not done:
- The scorer now decides "comparator vs candidate" by exact model name. That breaks for the `*_mwlogp` and
  `*_aug`/`*_multitaxa` names the new freeze writes (N1).
- Pre-registered ensembles can still silently lose members (N2).
- The strict parent-novel endpoint that C3's fix pre-registers is never computed by the sealed pass (N3).
- The §4.3 masked-CV rule is not applied to the published baselines or to `chemprop`, and unmasked phase-1
  out-of-fold files can still enter the ensembles (N4).

All four are cheap to fix. None is a data leak, and nothing has been scored: the registry has 0
`chemtox_prospective_test` entries.

**Versions reviewed (sha256[:16]).** The scorer, select, cycle, train and plan files were edited *during* this
review (latest edits 23:03–23:05). Findings refer to these versions:

| File | sha256[:16] |
|---|---|
| `chemtox_score_prospective.py` | `0f663893169845d4` |
| `chemtox_cycle_select.py` | `4806a6c4c68b01e5` |
| `scripts/train/chemtox_cycle.py` | `a7410611992f663e` |
| `cycle.py` | `1e9b01f0bdef0bac` |
| `prospective.py` | `74765aea3243aa1e` |
| plan | `05183cd5ed9185f8` |

Two defects I found in intermediate versions were fixed by those edits (R2, R3 below).

**Tests and probes.**
- `pytest tests/test_chemtox_prospective.py`: **26 passed** (98 s, `OMP_NUM_THREADS=2`).
- Probes are in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/r2/`:
  - `probe_scorer.py`: name resolution, extra kinds, and the guard on empty, missing and superseded registries.
  - `recompute2.py`: every §1.3/§1.3a/§1.3b number.
  - `altkey.py`: the parent key against RDKit `rdMolStandardize`.
  - `nnall.py`: NN Tanimoto against the *actual* training set.
  - `aug2.py`: augmentation parents.
  - `counts.py`: §2 row counts.
  - `cp_ions.py`: chemprop featurisation of every prospective SMILES.
  - `e2e_mock.py`: a **mocked end-to-end `main()`**.
    - The mock replaces `fit_predict` with random predictions (so no model is fitted).
    - It uses a copy of the prospective file with the target columns **dropped** and replaced by random numbers
      (so no real target is read).
    - Registry, run directory and training log are redirected to scratch.
    - The real registry, run directory and `TRAINING_LOG` were verified untouched afterwards.
  - `sel/`: a `--freeze` of the select script into scratch.

#### Round-1 findings: status

| ID | Status | Evidence |
|---|---|---|
| **C1** | **RESOLVED** | See note C1 below. |
| **C2** | **PARTIALLY RESOLVED** | The masking is correct and reaches every model (note C2). Not done: §4.3 says "the same rule applies to the baselines and to `chemprop`" (plan l.297), but there is no masked CV of `xgboost_published`, `rf_published` or the phase-1 chemprop configuration, and phase-1 out-of-fold files (unmasked) still enter the ensembles via `--phase1-families` (select l.226-235) → N4, N1. |
| **C3** | **PARTIALLY RESOLVED** | The data side is fully verified (note C3). Not done: the strict endpoint has no track in the scorer → N3. |
| W1 | RESOLVED | 11,500 rows (2,313 + 295 + 3 + 8,774 + 115). |
| W2 | PARTIALLY | Data, module and builder hashes match. §1.4 still gives the manifest as `manifests/chemtox_prospective.json`; it is `$SENTINEL_DATA/manifests/chemtox_prospective.json` (sha matches `3b2fc439…`). §6 code hashes are stale again (Y6). |
| W3 | PARTIALLY | "Genuinely novel" is gone and the train-partition figures are right. The figures for the set the models are actually trained on (all t-F2F, 1,905 chemicals) are still not reported (Y7). |
| W4 | RESOLVED | `*_all` tracks exist; masks 1,912 / 2,313 / 230 / 295 are tested. |
| W5 | PARTIALLY | `xgboost_published` / `rf_published` = phase-1 frozen MACCS + published37[`four`], and `model_family` resolves every suffix. But the role (comparator/candidate) is still assigned by exact name → N1. |
| W6 | RESOLVED | Per-variant counts, `n_configurations_across_variants` and `cv_rmse_by_chemical_block` are recorded. The frozen `n_configurations` still says 32 (the per-variant budget); acceptable now that the real count sits beside it. |
| W7 | PARTIALLY | Overrides are recorded in the registry config, not refused (Y8). `_trainpart` names work (mocked: 15 `*_trainpart` entries). `supersedes` is passed (mocked: 15/15 entries of an `--allow-rescore` pass carry it). New: the sensitivity pass overwrites the primary pass's artifacts (Y3). |
| W8 | PARTIALLY | The Holm family is "candidates only" as written, but `*_mwlogp` leak into it (N1). Direction is not checked (Y4). The per-seed rule is degenerate for ensembles (Y5). |
| W9 | PARTIALLY | The dry run now fits on a subsample and predicts all 11,500 rows. It still exits 0 when a pre-flight fit fails, and it skips the trivial/ensemble/NNLS/metrics block entirely (Y1). |
| **W10** | **NOT RESOLVED** | The `len(comps) < 2 → warning, skip` path is still there (scorer l.444-446). Components are filtered silently (N2). A missing OOF file raises only *after* every model has been fitted (l.452). NNLS without intercept and with renormalisation is still undocumented in the plan. |
| W11 | PARTIALLY | A unique temp file plus `os.replace` prevents corruption. The docstring claim "a concurrent writer can only ever replace it with a superset" is false (lost update; harmless, the next call recomputes). No encoder-weight hash is stored. |
| W12 | RESOLVED | `CLIP_SD = 1e6`, and the docstring now matches the measured values. |
| W13 | RESOLVED | `augmentation_rows()`: 1,101 → **944 rows** (251 chemicals), 157 dropped as prospective parents. **0** rows share a parent key with the primary track or any novel track (`aug2.py`; also asserted in `augmentation_rows` and by `test_augmentation_excludes_prospective_parents`). The 23:02 dry run shows lightgbm `ecotox` = 26,114 + 944 = 27,058. |
| W14 | PARTIALLY | 10 new tests. Still no end-to-end test of `main()` beyond the seals. `existing_entries` cannot be tested without monkeypatching (the registry path is not injectable). N1–N3 would have been caught by the mocked run above. |
| W15 | RESOLVED | `test_gpboost_rejects_or_honours_sample_weight` passes. |
| W16 | RESOLVED | The freeze raises on DataFrame-valued params (latent; phase-1 chemprop has none). |
| O1, O2, O3, O6 | RESOLVED | 63 cross-track rows (recomputed, = `report.json`); both cut-off denominators stated; re-issue risk disclosed (§1.3b); `reference_number` in `EXTRA_COLUMNS`. |
| O4 | RESOLVED | Breakdowns by `structure_class`, `has_metal`, `duration_h` and `parent_novel` appear in the mocked output. |
| O5 | RESOLVED | Bits and popcount are precomputed. `sample_weight ≠ 1` raises. The per-query fancy-index copy remains (performance only). |
| O7 | unchanged | Adequate. |

**C1 evidence.**
- `existing_entries` / `supersede_id` (scorer l.131-156) filter `registry_frame()`. They return `[]` / `None` on a
  missing and on an empty registry (`probe_scorer.py`).
- The guard runs at l.330, after the seals and before any dataset load or fit, and exits **6**.
- Mocked `main()`:
  - 1st pass: rc 0, 105 entries (15 models × 7 tracks).
  - Identical 2nd pass: rc 6, "84 model × track combinations already have prospective entries".
  - `--models xgboost_published`: rc 6.
  - `--train-partition-only`: rc 0, with `*_trainpart` names that do not collide.
  - `--allow-rescore --tracks temporal`: rc 0, 15/15 new entries carry `supersedes`.
- The only remaining raise after the first printed RMSE is `log_result`'s NaN refusal, which needs all-NaN
  predictions. The missing-OOF `RuntimeError` fires before any metric is printed.
- Residual: trivial baselines are not covered by the guard (Y2).

**C2 evidence.**
- `run_cv_extra` (cycle.py l.446-447) blanks `chem_ws`/`chem_mp` on a *copy* of the validation rows only.
- Trace: `builder.transform(val_rows)` → `PublishedFeatures.transform` → NaN → `nan_to_num(..., 0.0)` = the
  training mean after standardisation. That is exactly the prospective condition: the frozen file has 0 non-null
  `chem_ws`/`chem_mp`.
- Training rows keep their values. The `mean_`/`sd_` are fitted on them.
- The **same masked frame** is passed as `valid_rows` to `fit` and as `rows` to `predict`. So read-across
  (fingerprints only), GPBoost (taxonomy groups only) and chemprop (`x_d` from X) all see the deployment condition.
  No other model reads the two columns (`grep`).
- Run directories are labelled `_dm` / `_nodm`. `cycle_dirs(require_masking=True)` keeps `_dm` only.
- `DO_NOT_USE_FOR_SELECTION.txt` sits in all 4 old cycle-2 directories (gpboost, lightgbm, readacross, xgboost).
- The only masked runs so far are lightgbm screening runs (14 of 32). No selection-eligible masked record exists
  yet.

**C3 evidence.**
- The parquet `3b2fc439…` carries `parent_key`, `parent_novel` and `nn_tanimoto_adore_train`. `parent_novel` is
  constant within each chemical. Two rebuilds in `build.log` give the same hash. The manifest matches.
- Recomputed **65 of 301 primary chemicals / 550 of 1,912 rows** not parent-novel. Round 1's 552 used a slightly
  different key, and 550 is what the frozen key gives.
- The parent key is sound (see 🟢 G1).
- `augmentation_rows` now drops prospective parents and asserts at the parent level (W13).
- The plan has §1.3a/§1.3b. Every number in them reproduces (table below).

**Resolved during this review.**
- **R2.** The 22:52 scorer's `model_family` did not strip `_mwlogp`. Every freeze writes `xgboost_published_mwlogp`
  and `rf_published_mwlogp` (select l.331; confirmed by a scratch `--freeze`), so the real pass would have raised
  `KeyError` in `CYC.make_model`, after fitting every other model. Fixed at 23:04 with `NAME_SUFFIXES` and a
  `family` field.
- **R3.** The 22:59 select wrote `extra = best_variant.split("extra-")[-1]`, i.e. `"ecotox_dm"`, and
  `load_extra_rows("ecotox_dm")` silently returned 0 rows. Any augmented winner would have been scored *without* its
  augmentation. Fixed at 23:04 (`removesuffix("_dm")`).

#### Numbers recomputed from the parquet and `report.json` (plan §1.3, §1.3a, §1.3b, §1.4, §2)

All match the plan except where marked.

**§1.3 tracks.** All seven rows are unchanged: 2,313/363/71/364; 230/112/9/98; 295/136/10/114; 3/2/2/2;
8,774/225/114/1,136; 115/29/13/44; primary 1,912/301/64/310.

**§1.3a strict-subset table.**

| Endpoint | Rows | Chemicals | Species | Refs | Target SD | Replicate floor | Replicate groups | NN Tanimoto per chemical (median / % < 0.4 / % = 1.0) |
|---|---|---|---|---|---|---|---|---|
| Primary | 1,912 | 301 | 64 | 310 | 1.7304 | 0.4519 | 361 / 1,180 rows | 0.444 / 40.2% / 5.0% |
| Strict | 1,362 | 236 | 54 | 238 | 1.7638 | 0.4208 | 237 / 819 rows | 0.391 / 50.8% / 0.8% |
| `crust_novel` strict | 154 | 74 | 8 | 75 | 1.7475 | 0.6733 | — | — |

**Other §1.3a figures.**
- Temporal: 59 of 115 rows are parent-novel.
- Primary track per row: median 0.500, 1.6% at 1.0.
- `fish_novel_all`: 0.425 / 43.0%, p10 0.218, p90 0.800.
- `crust_novel` 0.476; `noncarbon` 0.200 (98.7% below 0.4); `temporal` 0.571.

**Not in the plan (W3).** Against the **actual training set** (all t-F2F, 1,905 chemicals):

| Set | Median | p10 | p90 | Below 0.4 | At 1.0 |
|---|---|---|---|---|---|
| Primary | 0.476 | 0.229 | 0.800 | 38.9% | 6.0% (18 chemicals) |
| Strict | 0.400 | 0.210 | 0.689 | 49.2% | 0.8% (2 chemicals) |

**§1.3b.** 958 `organic_multicomponent` / 954 `organic` rows; 596 rows / 82 chemicals with `has_metal`; 63 rows in
two tracks.

**§1.4.** 11,500 rows; sha256 `3b2fc439…`; builder `80a617ae…`; module `74765aea…`. All match. The manifest path is
wrong (W2).

**§2.**
- The training partition has 21,981 rows (of 26,114; test 4,133).
- `t-C2C` has 6,630 rows and `t-A2A` 704 (7,334 total). The guard counts 0 for every key on both.

**§6.**
- Match: data `3b2fc4398db940a2`, module `74765aea3243aa1e`, builder `80a617aecd89610a`.
- **Stale:** `cycle.py` is now `1e9b01f0bdef0bac` (plan `4e1b0ac7…`), the scorer is `0f663893169845d4` (plan
  `f01de5d0…`), and the package hash has changed (plan `1e71b908…`).

#### New findings

**🔴 N1. Comparator/candidate role is decided by exact name, so baseline variants become candidates.**
Scorer l.70-76 (`BASELINES`, `COMPARATORS`) and l.474-476 (`comparator_names`, `candidate_models`); select l.305
(`name = family + EXTRA_SUFFIX`) and l.331 (`*_mwlogp`).

What happens:
- Every freeze adds `xgboost_published_mwlogp` and `rf_published_mwlogp`. The scorer classifies both as
  **candidates**.
- The mocked run (`e2e_mock.py`) gives `candidate_models = [lightgbm, gpboost, chemprop, xgboost_published_mwlogp,
  rf_published_mwlogp, ensemble_mean, ensemble_stack_nnls]` and the same 7 in `holm_family_vs_xgboost_published`.
  Their registry entries get `is_baseline=False`.
- Any re-tuned baseline whose winning variant used extra rows is renamed `xgboost_aug`, `rf_multitaxa` or
  `readacross_aug`. It then also becomes a candidate, and `vs_xgboost` / `vs_rf` / `vs_readacross` silently
  disappear.

Consequences:
1. **Weakened baselines.** The un-weakened published baseline (WS/MP not imputed) is never a comparator, and is
   excluded from `best_baseline`. Meanwhile the Holm reference (a) is the variant that imputes WS/MP for every
   prospective row. That is the guard-4 problem C2 was about, and it contradicts the select comment ("comparing
   against the better one keeps the baseline at full strength").
2. **A larger Holm family.**
3. **A misleading claim:** a reproduced-baseline variant can be listed as "a candidate that beats the published
   method".

**Fix:**
- Write an explicit `role: comparator|candidate` into every frozen entry, and derive `comparator_names` from it
  (or from `model_family(name) ∈ {xgboost, rf, readacross}` plus `*_published*` plus the trivial baselines).
- Add the `*_mwlogp` entries to the comparators and to `best_baseline`.
- List them in plan §3.

**🔴 N2. Pre-registered ensembles can silently change composition or disappear (W10 not fixed).**
Select l.189/235 builds ensembles from **family** keys (`oof[family]`), so `components` holds family names.
l.305 freezes members as `family + suffix` (e.g. `lightgbm_aug`). Scorer l.444 keeps only components that are in
`preds`, and l.445 skips the ensemble with a warning if fewer than 2 remain.

Mocked scenario B (`lightgbm` frozen as `lightgbm_aug`):
- Both ensembles were scored over `[chemprop, xgboost]` only.
- rc 0, no error, and the component list is not in the registry entry.
- The NNLS weights are refitted on that subset too.

**Fix:**
- In select, write components as the frozen names.
- In the scorer, **before the first fit**, assert that every ensemble component is in `names`, and call
  `nnls_weights` once to prove every OOF file exists and aligns. Exit non-zero otherwise.
- Delete the `< 2 → skip` branch.
- Record `components` and `weights` in the registry `extra`.

**🔴 N3. The pre-registered strict parent-novel endpoint is not produced by the sealed pass.**
Scorer l.64 `TRACKS` has no strict track. The only output is a `breakdowns.parent_novel` cell: point RMSE per value,
with no CI, no paired difference and no Holm.

Plan §1.3a calls it a "mandatory co-reported endpoint". §4.4(d) makes it part of the claim rule ("whose CI is
reported next to the primary one"). Since the pass is one-shot, it would have to be computed later outside the seal.

**Fix:**
- Add `fish_novel_strict` (= `fish_novel ∩ in_adore_domain ∩ parent_novel`, 1,362 rows / 236 chemicals) and
  `crust_novel_strict` (154 / 74) to `TRACKS` and `track_mask`, with a test asserting 1,362 and 154.
- Name the splits in §1.3a/§5.
- State in §4.4(d) which comparisons the sign rule uses: vs `xgboost_published` and vs `best_baseline`, with
  `best_baseline` re-resolved on the strict rows.

**🔴 N4. The §4.3 masked-CV rule is pre-registered for baselines and chemprop but not implemented. Unmasked runs
can still enter selection.**
Plan l.297: "the same rule applies to the baselines and to `chemprop`". In the code:
- No masked CV exists for `xgboost_published`/`rf_published` (`four` vs `mw_logp`). Select freezes both variants
  instead of choosing one.
- `chemprop` is taken from the phase-1 `search/` directory, whose CV and OOF are **unmasked** (`cv.py` l.85 passes
  raw validation rows). Its spec is `chemical='four'`.
- Its OOF goes into `oof` and therefore into **both cycle-2 ensembles and the NNLS weights** (select l.226-235,
  scorer `nnls_weights` l.603-604). That mixes masked and unmasked runs in exactly the path the C2 fix says cannot
  happen ("an unmasked run cannot enter selection", §4.3). The stack is biased towards the member whose CV used the
  feature it will not have.

**Fix:**
- Run the two published configurations in both chemical blocks, and the phase-1 chemprop configuration, through
  `chemtox_cycle.py --config-json … ` (masked, `_dm`). The script already supports this.
- Choose each by masked CV rule, and freeze the chosen published variant as Holm reference (a), decided before the
  test.
- Make select refuse `--phase1-families` OOF files as ensemble members unless a `_dm` counterpart exists.
- Alternatively, amend §4.3 to say exactly what is and is not masked.

**🟡 Warnings**

- **Y1. The dry run is weaker than it looks.**
  - It returns 0 even when a pre-flight fit fails or predictions are non-finite (l.403-421, l.469): failures only
    go to the log.
  - It skips the trivial / ensemble / NNLS / metrics / registry block entirely (l.436). So N2 and a missing OOF file
    pass the dry run.
  - The engineer's 23:00 dry run did not include chemprop. Chemprop *featurises and runs a forward pass* on all
    670 unique prospective SMILES, including 79 bond-less ions (`cp_ions.py`), but the real fit path is unproven.
  - **Fix:** count the failures and exit 7; in the dry run, also run the component, OOF and NNLS checks; pre-flight
    chemprop on the GPU once.
- **Y2. Trivial baselines are outside the single-pass guard.** `global_mean`, `species_mean` and
  `species_duration_mean` are logged but not checked (l.330): the mock refused 84 combinations against 105 logged
  entries. A follow-up `--models X` pass re-logs them as non-superseding duplicates. **Fix:** include the trivial
  names in the guard list.
- **Y3. Each pass overwrites the previous pass's artifacts.** `out_dir = run_dir("chemtox", "prospective_test")`
  is a fixed path (l.309), and `pred_{name}.npy` has no suffix (l.433). So the `--train-partition-only` (or
  `--allow-rescore`) pass overwrites the primary pass's predictions and `prospective_results.json`, while the
  primary registry `artifacts` still point at those paths. **Fix:** use one directory per pass (suffix +
  timestamp).
- **Y4. Holm ignores direction.**
  - `paired_bootstrap` p-values are two-sided (bootstrap.py l.83), so `holm_rejected=True` also for a candidate
    that is significantly *worse*.
  - The CI is unadjusted, so "CI excludes 0 after Holm" (§4.4a) is ambiguous.
  - **Fix:** define the claim as `holm_rejected ∧ diff < 0 ∧ ci_high < 0` in the plan and compute it in code (e.g.
    `claim_ok`).
- **Y5. The per-seed rule is degenerate for the ensembles.**
  - `seed_predictions` is the single averaged prediction (l.458), so `all_seeds_better` for an ensemble just means
    "the mean is better".
  - Chemprop's "seed" is itself a 3-model ensemble (`n_seeds: 3`).
  - **Fix:** build per-seed ensembles from the components' seed-s predictions, and document chemprop's internal
    seeds. (Comparing a single candidate seed against the seed-*averaged* comparator is conservative, which is fine.)
- **Y6. Stale hashes and path.** §6 code hashes are stale (see the table above), and so is the §1.4 manifest path.
  **Fix:** re-record them at stage 2, as the plan says it will.
- **Y7. The training-set novelty figures are not reported (W3 residual).** §1.3a l.121 says "5.0% … to a training
  chemical", but `nn_tanimoto_adore_train` is computed against the training *partition* (1,588 chemicals), not the
  set the prospective models are trained on. **Fix:** add 0.476 / 38.9% / 6.0% (primary) and 0.400 / 49.2% / 0.8%
  (strict) against all t-F2F.
- **Y8. CLI overrides are recorded, not refused (W7).** `--seeds`, `--models`, `--tracks` and `--n-boot` change
  the pre-registered statistics. **Fix:** at least refuse `--seeds` and `--n-boot` values that differ from the
  frozen block without `--allow-rescore`.
- **Y9. NNLS behaviour is undocumented (W10 residual).** The plan does not say that NNLS has no intercept and that
  its weights are renormalised to sum to 1.
- **Y10. `--extra-weight` is lost at freeze.** It is in neither the run-directory label (train l.115-116) nor the
  frozen block (select drops `_`-prefixed params), and the scorer fits with `sample_weight=None`. Two runs with
  different weights would share a directory, and a weighted winner would be scored unweighted. This is latent;
  only `--extra-weight ≠ 1` triggers it.
- **Y11. The strict-subset decision rule is weak and ambiguous.**
  - It is sign-only.
  - "The weaker is the headline" names no comparator.
  - With 236 chemicals, a sign flip within noise passes.
  - **Fix:** together with N3, also require the strict-subset paired CI against `xgboost_published` to be reported,
    and state the downgrade wording if it includes 0.

**🟢 Observations**

- **G1. The parent key is sound.**
  - It agrees with an RDKit `rdMolStandardize` key (`LargestFragmentChooser(preferOrganic)` + `Uncharger` + stereo
    removal + InChIKey block) on **299 of 301** primary chemicals (`altkey.py`).
  - The 2 disagreements:
    - One under-merge: sodium dichloroisocyanurate 51580-86-0, 3 rows that should leave the strict subset.
    - One `None`: the Mn/Zn ethylenebis(dithiocarbamate)/fosetyl mixture 66523-56-6, conservatively excluded.
  - 40 of 301 keys are fallback SMILES rather than InChIKey blocks. Setting formal charges to 0 breaks sanitisation
    for nitro groups, quaternary N and aromatic `[nH]`. It works in practice (same function on both sides; picric
    acid NH4 salt ↔ picric acid still matches), but switching to `rdMolStandardize.Uncharger` would be more robust.
  - Metal complexes merge with their ligand (Zn/Cu/Mn dithiocarbamates ↔ Na salt). That is toxicologically coarse
    but conservative: it only shrinks the strict subset.
  - The strict subset is defined against **all** ADORE chemicals, which is the right, conservative reference.
- **G2. `best_baseline` chosen on the scored rows is acceptable.** Taking the minimum over comparators is
  conservative for a claim, and it is pre-registered (§3.6). Its p-value is not selection-adjusted, but the bias
  favours the baseline.
- **G3. A misleading docstring.** `existing_entries` says "a re-run recorded with `supersedes` does not block the
  next one". In fact the superseding entry is current and does block (probe), which is the correct behaviour.
  Correct the docstring.
- **G4. The masking design is scientifically right for selection.** Training rows keep WS/MP and validation rows
  get the training mean, which reproduces exactly what a deployed model sees. A further improvement would be
  training with random WS/MP dropout or a missing-indicator, so the model learns the imputed state. That is not
  required.

#### Checklist (round 2)

| Item | Result |
|---|---|
| C1 crash / guard | ✅ verified by mocked end-to-end runs (rc 0 → rc 6 → `--allow-rescore` supersedes) |
| C2 masking mechanism | ✅ |
| C2 rule coverage (baselines, chemprop, ensembles) | ❌ N4, N1 |
| C3 data, numbers, augmentation | ✅ |
| C3 strict endpoint in the sealed pass | ❌ N3 |
| Scorer completes for the requested frozen block | ✅ (mocked; xgboost_published, rf_published, both `_mwlogp`, xgboost, rf, lightgbm, gpboost, chemprop, readacross, ensemble_mean, ensemble_stack_nnls; 7 tracks, NNLS on real OOF files) |
| Scorer correctness under suffixed names | ❌ N1, N2 |
| Pre-registration ↔ code | ❌ §3 roles, §4.3 rule, §4.4(d), §1.4 manifest, §6 hashes |

**Re-review required** for N1–N4. Each is a few lines of code plus the plan text; the masked CV runs in N4 are a
small amount of compute. After that comes a mocked `main()` test committed to `tests/` (`e2e_mock.py` can be
adapted directly), then the stage-2 freeze.

### Round 3 re-review (fixes to N1-N4) — 2026-10-24

**Verdict: 🔴 FAIL. The single prospective scoring pass may NOT proceed. Do not freeze stage 2 and do not
use `--i-have-approval`.**

Most of the round-2 fixes are real and verified by a mocked end-to-end run:
- the `role` field;
- comparators first, and Holm over the candidates only;
- the strict tracks;
- the unmasked-member guard;
- trivial baselines inside the guard;
- the dry run returning 8.

Two new blockers remain, and both are in code paths the committed tests do not reach:
- **N5.** The NNLS stack cannot be scored once any member's frozen name carries a suffix, and the
  freeze silently drops `chemeleon_*` members.
- **N6.** `published_reference` falls back to the *weakest* published variant, which contradicts
  §3/§4.2/§4.3/§4.4.

Both are small fixes. Nothing has been scored: the registry has **0** `chemtox_prospective_test`
entries.

**Versions reviewed.** All match plan §6 except the package hash.

| File | sha256[:16] | Plan §6 |
|---|---|---|
| `chemtox_score_prospective.py` | `5515402fca2f9d5e` | ✓ |
| `chemtox_cycle_select.py` | `ad55dddd40c8d66e` | ✓ |
| `scripts/train/chemtox_cycle.py` | `a1dc897cfeb85dcc` | ✓ |
| `cycle.py` | `920d58c258e9cc30` | ✓ |
| `prospective.py` | `74765aea3243aa1e` | ✓ |
| builder | `80a617aecd89610a` | ✓ |
| data | `3b2fc4398db940a2` | ✓ (full sha256 `3b2fc439…54d9` recomputed) |
| `sentinel` package | **`e7729f45dce5db8c`** | ✗ plan says `7de259975a2e52f2` |

The package hash differs because other workstreams edited `sentinel/` after 23:24 (e.g.
`sentinel/data/microbe_wq.py`). See Z5.

**Tests and probes.**
- `pytest tests/test_chemtox_prospective.py`: **27 passed** (99 s, `OMP_NUM_THREADS=2`).
- Probes are in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/r3/`.
  - `build_fake.py` builds a **fake** `$SENTINEL_RUNS` tree: synthetic shard records and random OOF
    files, plus a read-only copy of the phase-1 `selection.json`.
  - The real `chemtox_cycle_select.py --freeze` was run against that tree with `--out` set to scratch.
  - `e2e3.py` is a mocked `main()` of the scorer:
    - `fit_predict` is replaced by random predictions after `CYC.make_model` resolves the family.
    - It uses the round-2 copy of the prospective file with the target columns dropped.
    - The registry, run directory and training log are redirected to scratch.
  - Afterwards, the real registry still has 0 prospective entries. No line of the real registry,
    training log or `TRAINING_LOG.md` mentions the scratch directory. The engineer's three earlier
    dry runs are the only `chemtox_prospective_test:scoring` lines.
- Scenarios:
  - **A**: the brief's block (`xgboost`, `rf`, `readacross`, `lightgbm` winning as `_aug`, `gpboost` as
    `_multitaxa`, `chemprop`, both ensembles over lightgbm/gpboost/chemprop, plus the 4 published
    entries).
  - **C**: `xgboost` and `readacross` winners augmented, `chemeleon_lightgbm` present.
  - **P**: a phase-1 OOF file.

#### Round-2 findings: status

| ID | Status | Evidence |
|---|---|---|
| **N1** comparator vs candidate | **RESOLVED** (role logic). The reference choice is broken → N6. | See note N1 below. |
| **N2** ensemble composition | **PARTIALLY RESOLVED** | The component-name and missing-component checks work. The OOF resolution for the stack is broken, and members can still be dropped silently at freeze → N5. |
| **N3** strict endpoint | **RESOLVED** (computation). §4.4(d) still has no operational definition → Z2. | See note N3 below. |
| **N4** masked CV everywhere | **PARTIALLY RESOLVED** | See note N4 below. |

**N1 evidence (scenario A, frozen block written by the real freeze).**
- Every frozen entry carries `role`. `xgboost_published[_mwlogp]` and `rf_published[_mwlogp]` get
  `role=comparator` and `family=xgboost|rf`. `lightgbm_aug` and `gpboost_multitaxa` get
  `candidate`, and so do both ensembles.
- The scorer logs:
  - `comparators: [xgboost, rf, readacross, xgboost_published, xgboost_published_mwlogp, rf_published, rf_published_mwlogp]`
  - `candidates: [lightgbm_aug, gpboost_multitaxa, chemprop]`
  - `candidate_models` = those three + `ensemble_mean` + `ensemble_stack_nnls`.
- In all 9 tracks, `holm_family_vs_published_reference` and `holm_family_vs_best_baseline` are exactly
  those 5 candidates.
- Registry `is_baseline` is True for all 7 comparators and 3 trivial baselines, and False for the 3
  candidates and 2 ensembles (135 entries = 15 models × 9 tracks).
- `best_baseline` / `published_reference` are created per track, skipped in the model loop and popped
  before `log_result`. They never appear as a registry `model`. They appear only as `vs_*` keys in
  `extra`.
- Role edge cases (scorer `role_of`, l.343-348):
  - An entry with no `role` and no `family` falls back to `model_family()`. `readacross_aug` →
    `readacross` → comparator. `rf_aug_multitaxa` and `xgboost_published_mwlogp` → comparator.
  - `xgboost_aug` (scenario C) is frozen as `comparator`.
  - A candidate family whose name only *contains* a baseline name (`chemeleon_xgboost`) stays a
    candidate in both scripts, because both use exact family matching. That is defensible: it is a
    new representation.
  - An unknown name with no role defaults to candidate. That only enlarges the Holm family
    (conservative) and cannot happen through `--freeze`.

**N3 evidence.**
- `fish_novel_strict` gives **1,362 rows / 236 chemicals**. `crust_novel_strict` gives
  **154 / 74**.
- Each strict track has its own `rmse_ci`, every `vs_*` paired CI, a Holm table for both references,
  `best_baseline` re-resolved on the strict rows, and registry notes "CO-PRIMARY (strict
  parent-novel)".

**N4 evidence.**
- The four masked-CV numbers are real and match the run directories:
  - XGB `four` **1.4600** (record 910); XGB `mw_logp` **0.9945** (911).
  - RF `four` **1.2561** (912); RF `mw_logp` **0.9909** (913).
  - All four are 5-fold, `_deployment_masking: True`, under
    `cycle2/{xgboost,rf}_…_extra-none_dm/shardpub_*`.
- The unmasked-member guard (select l.276-284) works:
  - It raises for a phase-1 family with an OOF file (scenario P).
  - It also raises when that family is excluded via `--ensemble-members`. That is stricter than
    necessary and harmless.
  - It does not trigger for `_dm` members.
- `--allow-unmasked` is the only escape. It appears in no queue script: `grep` of `master_queue.sh`,
  `cpu_queue*.sh`, `gpu_queue*.sh`, the RUNBOOK and the repo. Nothing in the repo or the queues uses
  `--no-deployment-masking` or `--extra-weight` either.
- The phase-1 configurations are re-running masked as cycle-2 directories:
  - `gpu_queue3.sh`: embeddings (running), then tabpfn, then chemprop `four`.
  - `gpu_queue4.sh`: chemprop `mw_logp`.
  - Their configuration JSONs carry an explicit `device`/`accelerator`, so `attach_device` keeps
    them.
  - They are **not finished**, so §4.3's "each of those is re-run" is not yet true.
- Not done:
  - The masked published numbers never reach the frozen block → N6.
  - `--allow-unmasked` is not recorded → Z4.

**Warnings from round 2.**

| ID | Status | Note |
|---|---|---|
| Y1 dry run | PARTIALLY | Returns 8 on a failed fit. Non-finite pre-flight predictions are only logged, not counted (l.466-467). The trivial/ensemble/NNLS block is still skipped in the dry run (l.488), so the dry run does **not** catch N5. |
| Y2 trivial guard | RESOLVED | `guarded` includes `TRIVIAL` (l.378). |
| Y3 artifacts | PARTIALLY | `pred_{name}{suffix}.npy` no longer collide. `prospective_results.json` is still a fixed path, so a `--train-partition-only` pass overwrites the primary pass's report (the per-track `best_baseline` model name and the Holm tables live only there). Trivial and ensemble `artifacts` point at files that are never written. |
| Y4 direction | RESOLVED | `holm_rejected` requires `diff < 0` (l.619). Mock: candidates with `diff>0` never get `True`. |
| Y5 seeds | RESOLVED as specified | Ensembles get `all_seeds_better: null` + `seed_rule`. But §4.4(c) then makes an ensemble claim-ineligible, or waives (c), and the plan does not say which → Z3. |
| Y6 hashes | PARTIALLY | Code hashes match. The package hash is stale (Z5). The §1.4 manifest path still says `manifests/…` rather than `$SENTINEL_DATA/manifests/…`. |
| Y8 overrides | unchanged | Recorded, not refused. |
| Y10 extra_weight | PARTIALLY | Frozen (select l.349), but the scorer always fits with `sample_weight=None` and does not refuse `extra_weight ≠ 1`. Latent: no queue uses `--extra-weight`. |
| Y11 strict rule | NOT RESOLVED | → Z2. |
| Accelerator fix | RESOLVED | `attach_device` (cycle.py l.238-251) only fills a missing `device`/`accelerator`. The cycle-2 `chemeleon_tabpfn` screen (32/32 failed with the CPU error before the fix) has **not** been re-queued. That is acceptable under §3 item 8 ("whichever exist"), but it is not "fixed" in the runs. |

#### New findings

**🔴 N5. The NNLS stack cannot be scored when a component's frozen name differs from its family, and the
freeze silently drops `chemeleon_*` ensemble members.**

*Scorer.* `nnls_weights` (scorer l.668-694) looks up `payload["families"][c]` (l.676) with `c` = the
**frozen** name now written into `components` (e.g. `lightgbm_aug`). `families` is keyed by family
(`lightgbm`), so the lookup is empty and the function returns `None` (l.685-686). Even with the
lookup fixed, l.684 builds the file name from the component name (`oof_lightgbm_aug_003.npy`), which
does not exist either. The frozen `oof_sources` (keyed by family) are written by select l.374 and
read by nobody.

Mocked scenario A:
1. All 30 fits run.
2. `ensemble_mean` is computed.
3. Then `RuntimeError: ensemble_stack_nnls: … out-of-fold prediction matrix is missing …` (l.507-511).
4. rc ≠ 0, nothing logged.

So with `lightgbm_aug` or `gpboost_multitaxa` in the stack, which is the brief's case and the
master queue's items 7 and 8, the one-shot pass **cannot complete**. It fails after every model has
been fitted. No metric has been printed and no target has been read at that point, so this is not a
leak, but stage 2 would have to be re-frozen. The dry run skips this block (l.488), so it would pass.

*Select.* The runner saves `oof_{args.model}_{idx}.npy` (train l.195), e.g.
`oof_chemeleon_lightgbm_006.npy`. Select looks for `oof_{MODEL_ALIAS[family]}_…` (l.214, and
`oof_source` l.166-167), i.e. `oof_lightgbm_006.npy`. That file is absent, and `if p.exists()`
(l.215) skips the family **silently**.

Scenario C, with `--ensemble-members lightgbm chemprop chemeleon_lightgbm`, froze both ensembles over
`[chemprop, lightgbm_aug]` only, with no warning. Every `chemeleon_*` family is affected
(`chemeleon_lightgbm`, `chemeleon_gpboost`, `chemeleon_xgboost`, `chemeleon_tabpfn`). That is exactly
the "ensemble silently loses members" defect N2 was about, moved from the scorer into the freeze.

A related loud failure: an ensemble member from a comparator family whose winner is suffixed (e.g.
`xgboost` → `xgboost_aug`) makes the freeze raise "component 'xgboost' has no frozen entry"
(scenario C2). The cause is that `family_to_frozen` (l.356-357) maps candidate families only. It is
loud, but it would block the runbook's membership rule if a re-tuned tree qualifies.

**Fix:**
- In select, build OOF paths with the runner's naming (`oof_{family}_{idx:03d}.npy`, no
  `MODEL_ALIAS`).
- Raise if any `--ensemble-members` family has no OOF file.
- Key `oof_sources` by **frozen** name.
- Map comparator families in `family_to_frozen` too.
- In the scorer, make `nnls_weights` read `frozen[ename]["oof_sources"]`, which is inside the hashed
  block, instead of `payload["families"]`.
- Before the first fit **and in the dry run**, check that every component is in `names` and run
  `nnls_weights` once (exit non-zero on failure).
- Add a test whose frozen block has a suffixed NNLS member and real OOF files in a temp runs root.

**🔴 N6. `published_reference` resolves to the weakest published variant, contradicting the plan.**

`published_baselines()` (select l.81-96) copies only `features`, `params`, `n_configurations` and
`source`. The freeze adds `seeds`, `extra`, `cycle`, `family`, `role` and `note` (l.377-390). **No
`cv_rmse`** is written, and the masked records 910-913 are never linked to these entries.

The scorer (l.363-368) then finds `with_cv` empty and falls back to `"xgboost_published"`, the `four`
variant whose masked CV is **1.4600** (scenario A log: `published reference … xgboost_published
(masked CV None of {…: None})`). The plan says the opposite:
- §3 2b: decided by masked CV.
- §4.3: "the `mw_logp` variant … becomes the `published_reference`".

So Holm reference (a) and the headline "vs the published method" number would be computed against a
baseline the plan itself describes as half a log unit weaker.

The committed e2e test hides this, because it hand-writes `cv_rmse` into the published entries (test
l.498-499), which the real freeze never does.

The plan is also internally inconsistent about *which* set the reference is chosen from:
- §4.2 says "the better of the **two** published variants" (XGB `four` vs `mw_logp`).
- §4.3 names XGB `mw_logp`.
- §4.4(a) and the RUNBOOK still say `xgboost_published`.
- The code (once `cv_rmse` is present) takes the minimum over **all four** `*_published*` entries.
  With the measured numbers that gives `rf_published_mwlogp` (0.9909 < 0.9945): probe `selA3.json`.

The overall claim is still bounded by (b) `best_baseline`, which contains every published variant
re-resolved on the track, so the SOTA verdict itself is not made easier. Criterion (a), and the
§15.1(i) number that will be quoted, are wrong.

**Fix:**
- Write each published entry's masked `cv_rmse` (and the source record index) into the frozen block.
- Make the scorer **exit** if any `*_published*` entry lacks `cv_rmse`, instead of falling back.
- Better still, have the freeze write an explicit `published_reference` key (chosen by the stated
  rule) into the hashed block, and record its name in every registry entry's `config`. §3 2b claims
  this already ("in every registry entry"), but today the name is only in the log and
  `prospective_results.json`.
- State one rule in §3/§4.2/§4.3/§4.4(a)/RUNBOOK. Either "the better XGB variant" or "the better of
  all four published variants".

**🟡 Warnings**

- **Z1. The published masked runs live inside the re-tuned families' search directories.** Records
  910-913 sit in `cycle2/{xgboost,rf}_…_extra-none_dm/`. Select reads every `shard*_results.json`
  there, so:
  - the re-tuned `xgboost`/`rf` pool is 32 + 2 configurations, while the frozen `n_configurations`
    says 32;
  - the "re-tuned" winner can be literally the published `mw_logp` configuration (in scenario A it
    was record 911);
  - `finalists.py` ranks these 5-fold scores together with 3-fold screened scores. Indices 910-913
    passed to `--indices` are silently ignored, so the finals get fewer than 5 configurations.

  Acceptable if disclosed. Cleaner to run them under a separate label.
- **Z2. §4.4(d) has no operational definition, and the code computes no decision.**
  - The plan does not say which fields decide "keeps its sign". The natural reading is
    `tracks.fish_novel_strict.models[<cand>].vs_published_reference.diff < 0` **and**
    `.vs_best_baseline.diff < 0`, with `best_baseline` re-resolved on the strict rows.
  - The plan does not say whether the strict track's own Holm/CI matter.
  - "The weaker is the headline" names no comparator.
  - **Fix:** state those fields in §4.4(d), and emit a `claim` block per candidate in
    `prospective_results.json` that evaluates (a)-(d) mechanically.
- **Z3. §4.4(c) versus ensembles.** `all_seeds_better` is `null` for both ensembles (and for any
  single-prediction model). (c) requires it to be *true*, so as written an ensemble can never be
  called "improved". **Fix:** before the pass, either declare ensembles exempt from (c) (and say so
  in the claim), or build per-seed ensembles from the components' seed-s predictions.
- **Z4. `--allow-unmasked` leaves no trace in `selection.json`** (scenario P3: the string "unmasked"
  does not occur). It also re-admits `_nodm` and the four discarded pre-fix directories. **Fix:**
  record it in `payload` and in every frozen entry, and make the scorer refuse a frozen block built
  with it.
- **Z5. The package hash in §6 cannot stay valid.** `package_hash()` covers all of `sentinel/`, which
  M3/M6/M7/downstream edit continuously (`e7729f45…` now, against `7de25997…` in the plan). Record a
  ChemTox-scoped hash at stage 2 (`sentinel/models/chemtox/`, `sentinel/eval/bootstrap.py`,
  `sentinel/core/registry.py`, `sentinel/data/sources/adore.py`), or snapshot the package
  (`snapshot_code`) at scoring time.
- **Z6. Plan text lags the code in small ways.**
  - §3 item 7 still describes `chemprop` as "the configuration frozen for the ADORE pass". It is now
    the masked re-run, with its chemical block chosen by masked CV (`gpu_queue4.sh`).
  - §4.2's comparison list omits `*_mwlogp`, `global_mean`, `species_mean` and
    `published_reference`. The code compares against all of them.
  - §4.3 states the phase-1 re-runs as done; they are queued.
  - The RUNBOOK's ensemble-membership rule is not in the plan yet. It must be in stage 2 before the
    pass.

**🟢 Observations**

- **G5.** A `--train-partition-only --models xgboost_published <ensemble>` sensitivity pass (RUNBOOK
  step 4) raises "components … were not scored". That is correct, but the runbook should list the
  components explicitly.
- **G6.** `vs_best_baseline` is also computed for the comparator that *is* the best baseline (diff 0,
  p ≈ 1). It is harmless (not in Holm), but it is noise in the registry `extra`.
- **G7.** `best_baseline` chosen per track on that track's RMSE is stated in §3 item 6 and is
  conservative for (b): the minimum over 10 comparators on the scored rows, with a selection bias
  that favours the baseline. `published_reference` is fixed across tracks and chosen pre-test, which
  is correct once N6 is fixed.

#### Checklist (round 3)

| Item | Result |
|---|---|
| Role assignment, aliases, `is_baseline`, Holm family | ✅ (mocked scenario A, 9 tracks, 135 entries, no alias leak) |
| Strict tracks with full statistics | ✅ 1,362/236 and 154/74 |
| Ensemble components by frozen name | ✅ mean. ❌ NNLS (N5) |
| No silent member loss at freeze | ❌ `chemeleon_*` (N5) |
| Unmasked guard / `--allow-unmasked` not used | ✅ / ✅ (not recorded: Z4) |
| `published_reference` chosen by masked CV | ❌ falls back to the `four` variant (N6) |
| Plan §3/§4.2/§4.3/§4.4 ↔ code | ❌ N6 rule mismatch, Z2, Z3, Z6 |
| §4.3 masked numbers | ✅ 1.4600 / 0.9945 (and RF 1.2561 / 0.9909) |
| §6 hashes | ✅ code, data, builder. ❌ package (Z5) |
| Tests | ✅ 27 passed. The e2e test does not cover NNLS or a real freeze |

**Verdict: 🔴 FAIL. The single prospective scoring pass may NOT proceed until N5 and N6 are fixed:**
- the OOF resolution for the NNLS stack;
- no silent member loss at freeze;
- a dry-run NNLS check;
- `published_reference` written from masked CV, plus one consistent rule in the plan.

**Z2/Z3 must also be decided in the plan before stage 2.** Re-review can be a short targeted check:
re-run `r3/build_fake.py` + a real `--freeze` + `r3/e2e3.py` **without** `PATCH_NNLS`.


---

## M2 SENTINEL-Orbit — 2026-10-24

**Status: 🟡 PASS WITH CONCERNS** after two review rounds. The first round **FAILED** (3 criticals). All
critical items were fixed, together with the re-review's N1 (split v3), **before any real training run**; only
smoke runs (≤ 500 fit rows, 2 epochs, never registered) had been executed. Do not write `FREEZE_orbit.json` until
W4 is resolved.

**Reviewer:** an independent adversarial review agent, read-only (it read all code and ran checks on the dev
cache), plus the workstream's own empirical leakage checks.

**Reviewed:**
- `sentinel/models/orbit/{split,guard,data,features,baselines,mdn,satwq,pipeline,evaluate,teacher}.py`
- `sentinel/data/sources/hydrolakes.py` and `scripts/data/build_hydrolakes.py`
- `scripts/train/orbit_{build_split,baselines,satwq,compare,placebo,gloria_trackr}.py`
- `tests/test_orbit.py` and `configs/orbit/satwq.yaml`
- `docs/preregistration/orbit_protocol.md`, the `orbit_split*.json` files and the split reports

### Round 1: FAIL

**Checked and correct:**
- The guard: the dev cache has 0 protected rows, and every training and compare entry goes through
  `load_dev(training=True)`.
- Normalisation, kNN, index calibration and MDN scalers are fitted on fit rows only.
- GLORIA pretraining uses train role and years ≤ 2021 only.
- No lake or site_id spans two roles.
- SatWQ-Net masking and `key_padding_mask` polarity were verified by perturbation.
- The OneCycle step count, best-weight restore and equal-weight seed mixtures are correct.
- The metrics match `sentinel.eval.metrics`.
- No chip, scene, point-date or spectrum overlap between fit, inner and val (our own check).

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| C1 | 🔴 | Non-lake points (estuaries, rivers) had 50-km roles. Chesapeake had 697 train / 431 val rows, Tampa Bay was split too, and 27.6% of non-lake val rows lay within 10 km of a fit row. Shore stations of test lakes were in training. | **Split v2:** lake capture radius 500 m; non-lake roles from 200-km blocks; 1-km buffer near lakes of another role; **headline = in-lake rows only**; vendored `spatial_blocks`; `verify_split_file` raises if the rule code changes. Chesapeake non-lake rows are now 1,177 train / 3 val. |
| C2 | 🔴 | `mdsa_log10` returned NaN when one prediction was NaN. That made MDN_B always use 7 bands, which under-tunes the H1(a) comparator. | `mdsa_log10` now **raises** on non-finite values. The MDN inner selection masks unpredictable rows and counts them; every caller was traced. |
| C3 | 🔴 | The cache was out of date with the split code; non-lake unit names spanned blocks and roles. | Rebuilt as dev_v2, then dev_v3. `load_dev` recomputes role, block, tblock, waterbody, buffer and protected from the frozen rule and asserts equality. The lake cache name carries the HydroLAKES sha. |
| W1 | 🟡 | The CI unit (water body) did not match §15.1 (block bootstrap). | Primary CI = paired **role-block** bootstrap; water-body bootstrap is a sensitivity. |
| W2 | 🟡 | Tuning asymmetry: MDN had 2 configs, GBM 8 configs with 1 seed, and baselines saw only inner rows while SatWQ-Net was selected on val. | MDN_B grid of 12 configs (bands × iterations × input scaling); GBM grid of 16 configs plus a 5-seed final ensemble; kNN k ∈ {4, 8, 16, 32}. Each H1 comparator is the **val-selected** member of its family. Official MDN-STREAM weights added as a further baseline. |
| W3 | 🟡 | The seen-lake cell included inner blocks. | Excluded. |
| W4 | 🟡 | **Open.** Dev is 100% AquaCorr while the test cell is mostly WQP-MN. TSS/turbidity val counts are tiny, and one block holds 28.5% of primary-cell rows. | TSS/turbidity use the pooled cell `valxany_lake`. **Before the freeze:** the national WQP 2024–25 rebuild and a power analysis. |
| W5 | 🟡 | Rounding-based de-duplication missed pairs. | Distance-based (20 m AquaCorr–AquaCorr, 100 m WQP–AquaCorr, same date). |
| W6 | 🟡 | NTU, FNU and NTRU were merged. | Documented as a forced deviation (AquaCorr gives no unit). The WQP unit is kept, and the test is to be stratified by unit. **To be coded in the eval script.** |
| W7 | 🟡 | Grid error in mixture quantiles (up to 0.07 log10). | Exact bisection; error < 1e-12 against brentq. |
| W8 | 🟡 | Silent fallbacks. | `run_epochs` raises when early stopping is impossible; the index fallback rate is logged; GLORIA NaN coordinates and inner-block rows are dropped. |
| W9 | 🟡 | The freeze did not pin the data, and teacher code sat inside the frozen globs. | `data_version` and `data_manifest_sha256` are required and verified against the manifest and its files. teacher.py and trackr are excluded; the placebo stays frozen. |
| self | 🔴 | The vectorised MDN averaged the loss over 10 inits, so each member's NLL gradient was 1/10 against an unscaled weight decay. | Loss is now summed over inits, with `weight_decay = 2·l2` (Keras-equivalent coupled L2). |

### Round 2 (re-review): PASS WITH CONCERNS
- C1, C2, C3, W1–W3, W5, W7 and W8 are resolved. W9 is now fully resolved after this round.
- **N1 (🟡):** the 500 m capture radius pulled estuary stations into tiny HydroLAKES polygons of another role.
  For example, 69 Hillsborough Bay rows sat in a 0.76 km² polygon with train role inside a test estuary block.
  **Fixed by split v3** (`orbit_split_v3.json`, sha256 `ffe288f2…5e32`): a lake row outside its polygon is
  buffer-excluded when the surrounding non-lake block has a different role. The fix uses geometry only, and
  `test_v3_out_of_polygon_lake_rows_buffered_when_estuary_role_differs` covers it. Lake roles are unchanged.
- **N2 (obs.):** the buffer checks only the nearest lake. The non-lake track has 7–11 blocks, so it is
  descriptive only.
- **N3 (obs.):** val selection of comparators is optimistic for the baselines, as documented. The chosen
  members are saved (`comparators_val_selected.json`) and go into FREEZE.
- **Residual near-neighbour proximity:** in-lake val points have a train point within 10 km in about 35% of
  cases, which is inherent to 50-km blocks. A pre-registered sensitivity analysis scores headline rows more than
  10 km from any train point.
- Tests: `tests/test_orbit.py`, 22 passing.

**Gate decision:** real training and validation-only scoring may start on dev_v3. `FREEZE_orbit.json` and any
test access remain blocked until W4 (the national WQP 2024–25 rebuild, power analysis) and W6 (unit-stratified
test scoring code) are done and re-reviewed.

---

## M5 BioAssess — pre-test gate (2026-10-25)

**Review mode:** pre-test gate (review skill §1–5 plus §6.2/§7). Question: may the frozen M5 configuration be
scored once on the held-out data (NRSA 2018-19 Tests A/B, NRSA 2023-24 Test C, UK reserved blocks T1/T2)?
**Scope:** M5 only. Read: RESEARCH_PLAN §3, §5.5, §14, §15 (incl. §15.5 amendment); CONTRIBUTING_V2;
`docs/preregistration/bioassess_protocol.md` (state of 2026-10-24 23:34, sha256 `35c24cd1…`);
`bioassess_freeze_candidate.json` (prepared 23:34:14); model card; `sentinel/models/bioassess/*`;
`scripts/{train,eval}/bioassess_*.py`; `configs/bioassess/*`; `tests/test_bioassess_models.py`; the registry
(1,474 `bioassess` rows); TRAINING_LOG M5 entries; RESULTS.md M5 line.
**Lead decisions taken as given:** candidates A, B and C admitted; Holm over 18 tests; per-target comparator =
strongest development-LOEO baseline; C framed as "BioSet adds signal beyond the strongest baselines".
**Reviewer actions:** read-only, plus `pytest tests/test_bioassess*.py` (29 passed, CPU) and two metadata-only
checks on `visits.parquet` (coordinates/survey/ecoregion only; no chemistry, no test-survey taxa). The scorer
was not run, `bioassess_freeze.json` was not created, and no M5 file was changed.

### Summary

**Overall status: 🔴 FAIL — do not freeze or score yet.**

No contamination was found. No phase-1 code path loaded 2018-19, 2023-24 or reserved-UK labels; there is no
test-split registry entry and no scoring marker; the location-cluster logic and the Test A/B design are correct.
The frozen-code hashes currently match.

The scorer that would be frozen, `scripts/eval/bioassess_final.py`, has five blocking defects:

- it discards the per-seed predictions that the SOTA rule (§15.1 iii) needs;
- it never computes the pre-registered H-M5 endpoint with its Holm correction;
- it keeps the confirmatory Holm decision outside the registry;
- the freeze hashes leave out shared code, the data and the TabPFN weights that the predictions depend on;
- it has never been run end to end, yet it burns the one-time marker before it does any work.

All five can only be fixed **before** the freeze, because the script and its hash become immutable afterwards.
Each fix is small and none needs new development results. Re-review is needed only for the diff.

### Critical issues (must fix before `bioassess_freeze.json` is created)

**C1 — The per-seed results needed for the §15.1 (iii) "gain holds in ≥ 3 seeds" criterion are thrown away.**
- `bioassess_final.py:160` keeps only the seed mean (`p, _, _ = CV.bioset_rows(...)`), and `:166-168` averages
  the baseline seeds without storing them.
- Protocol §5 says "per-seed R² is stored to show whether a gain holds per seed (§15.1 iii)".
- GPU training is not bit-reproducible, so after the single pass the per-seed numbers can never be recovered.
  The "SOTA" label would then be unawardable, or awarded without its third condition.
- **Fix:**
  1. Keep the per-seed predictions for every stochastic model and save them in the `pred_test*.parquet` files.
  2. Define the per-seed ensemble a priori as the mean of the members' seed-s fits.
  3. Log `r2` and `delta_r2_vs_sota` per seed to the registry (`extra.per_seed` or a metric suffix).
  4. State in §9.1 that a SOTA label needs the per-seed Δ > 0 for seeds 0, 1 and 2, in addition to the Holm
     criterion.

**C2 — The pre-registered primary endpoint is not computed as registered, and the confirmatory decision is not
in the registry.**
- Protocol §1 and RESEARCH_PLAN §15.2 (H-G1, M5) define the hard-threshold endpoint as LOEO R² above the
  blocked RF[community] for ≥ 3 of 6 targets, with paired CIs excluding 0, **Holm-corrected across the 6
  targets**.
- The scorer logs `delta_r2_vs_h_m5` (`bioassess_final.py:196-210`) but feeds only the `sota` p-values into
  Holm (`:203-204`). H-M5 therefore gets no multiplicity correction at all.
- The Holm result itself goes only to `run_dir/holm_<test>.json` (`:211-213`). That breaks R1/R2: the number the
  claim rests on is not a registry entry.
- Holm is also applied to every block (A, C, B_all, B_new, B_resampled). The pre-registration names only Test A
  as confirmatory, so the other files invite selective reading.
- §9.1 still contains the superseded sentence "Holm across both models × 6 targets (12 tests)" next to the new
  18-test rule.
- **Fix:**
  1. In §9.1, write one explicit claim table: for each candidate, the comparator, the family (SOTA: 18 tests;
     H-M5: model A × 6 targets, or 3 × 6 if H-M5 is claimed for B and C too) and the decision rule.
  2. Delete the stale 12-test text.
  3. In the scorer, compute both families **on Test A only**.
  4. Log each test's raw p, Holm-adjusted p, adjusted α and rejected flag through `log_result`
     (e.g. metric `holm_rejected_sota`, `holm_rejected_h_m5`).
  5. Label the B/C/T comparisons "secondary, unadjusted".

**C3 — The freeze does not pin everything the predictions and scores depend on.**
- `bioassess_freeze_candidate.py:47-53` hashes only M5's own files. It leaves out:
  - `sentinel/eval/splits.py` (`spatial_block_split`: the inner early-stopping split, `splits.py:85`);
  - `sentinel/eval/bootstrap.py` (all CIs, p-values, Holm);
  - `sentinel/eval/metrics.py`;
  - `sentinel/core/seed.py` (`bioset.py:363`);
  - `sentinel/core/registry.py` (`file_sha256`, `log_result`);
  - `sentinel/models/chemtox/tabpfn_model.py`, the sklearn shim without which `tabpfn` 2.2.1 does not import
    (verified: a bare `import tabpfn` raises `ImportError: _is_pandas_df`);
  - the TabPFN weights (`~/.cache/tabpfn/tabpfn-v2-regressor.ckpt`);
  - the input data (`data.input_hashes()` already exists, `data.py:220-227`, but is not used by the freeze);
  - the UK EA processed files;
  - package versions (`tabpfn 2.2.1`, `torch 2.7.1+cu118`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`,
    `numpy 2.2.6`, `pandas 2.3.3`).
- The `sentinel/eval/*` modules are shared by nine concurrently edited workstreams. An unrelated edit between
  freeze and scoring would silently change M5's test computation.
- **Fix:**
  1. Add those files to `HASHED`, plus `sentinel/models/bioassess/__init__.py`, `sentinel/core/paths.py` and
     `sentinel/core/gpu.py`.
  2. Add `data.input_hashes()`, the sha256 of the UK parquet files and of the TabPFN checkpoint, and a
     `pip freeze` digest.
  3. Make `verify_freeze()` check all of them.
  4. Snapshot the frozen tree. All M5 files are **untracked in git** (`git status`: `??`), so the lead should
     commit it, or archive it with its sha256 in the freeze record.

**C4 — The single-shot scorer has never been run, and it spends the test before it does any work.**
- The marker is written (`bioassess_final.py:110-114`) before the torch import, the GPU lock (`:128`, which can
  wait for hours on the shared GPU 4) and any data load. A kill or crash at that point burns the pass under the
  protocol's "needs a new lead decision" rule.
- No test runs the A/B/C/T paths, the memoised ensemble assembly, the `+geography`/`+context` BioSet dispatch or
  the UK transfer calls. The only scorer test (`test_final_scorer_refuses_without_freeze_and_parses_ensembles`)
  covers the missing-freeze refusal and `ensemble_members()`. The "refuses a second pass (unit-tested)" claim in
  §9.1 is not backed by a test.
- `log_result` raises on NaN values (`registry.py:118-119`), so any NaN metric in a small stratum aborts the run
  mid-way.
- **Fix:**
  1. Add a `--dry-run-dev` mode that runs the **identical** code path with a pseudo-test from development data
     only: dev = nrsa0809 and pseudo-test = nrsa1314 for A/B; UK `pool_val` as the pseudo-reserved set for T.
     It must write no marker and log to a scratch registry (`registry_path`).
  2. Run it end to end once on GPU 4.
  3. Confirm on development data that the frozen code reproduces candidate A's development LOEO pooled R²
     within seed noise. The registered value was produced on 2026-10-22 by an older code version (package
     `dc6479c4`); the refactor-equivalence test uses synthetic data only.
  4. Write the marker only after the GPU lock, imports and the data load succeed.
  5. Make NaN metrics log as "not estimable" notes rather than raising.
  6. Add a unit test for the marker refusal.

**C5 — Statistical settings can still be changed at run time.**
- `--tests` (`bioassess_final.py:105`) lets a subset run, after which the marker blocks the rest forever. It also
  lets the lead run A first and decide on B, C and T after seeing A.
- `--n-boot` (`:106`) can change B away from the registered 2,000.
- **Fix:** read the test list and B from the freeze record and drop both CLI options (or refuse any value that
  differs from the frozen one).

### Warnings (should fix, preferably in the same pre-freeze diff)

- **W1 — Taxon vocabulary is built from test inputs.**
  - Where: `bioassess_final.py:159` (and `cv.py:141` in CV) runs `TaxonVocab.build(design.tokens)` over every
    loaded row, including the 2018-19 / 2023-24 test samples.
  - No label information leaks: unseen names are mapped to UNK by `restrict`, `cv.py:107-114`. But the embedding
    table size, and with it the RNG draws for **all** subsequent parameter initialisations, depends on the
    test-set taxon list. That contradicts protocol §4 ("vocabularies … fit on the training fold only").
  - Fix: build the vocabulary from development tokens only (the same set CV used).
- **W2 — "BioSet adds signal" for C is not tested directly.**
  - The scorer tests C only against the single per-target comparator.
  - For cond, chloride and pH that comparator is the 2-member `Ensemble[RF+geo+TabPFN]` (C without BioSet). For
    TP it is the 3-member RF+geo+LGBM+TabPFN; for TN, RF[context]+TabPFN; for turbidity, TabPFN.
  - Fix: log, for C and every target, the paired Δ against (a) the leave-BioSet-out ensemble
    `Ensemble[RF[community+geography]+TabPFNv2]` and (b) the equal-size control
    `Ensemble[RF[community+geography]+LightGBM+TabPFNv2]`. Pre-register that the framing is used only where
    both Δ > 0 (secondary, unadjusted) in addition to the confirmatory Holm test.
- **W3 — Monte-Carlo resolution near the Holm threshold.**
  - The smallest Holm threshold is 0.05/18 ≈ 0.0028. With B = 2,000 and p = 2·min(tail), a test passes only if
    ≤ 2 replicates fall on the wrong side, so the decision is sensitive to bootstrap noise.
  - Fix: pre-register B = 10,000 (or more) for the confirmatory paired tests only.
- **W4 — The location-cluster paired bootstrap is anti-conservative for Test A.**
  - Test A predictions come from 9 per-ecoregion fits, and errors are spatially correlated beyond 1 km.
  - Fix: pre-register a sensitivity for the confirmatory Δs (paired 50-km-block bootstrap, or the coarse
    ecoregion unit) and log it in `extra`. Today the ecoregion-unit CI exists only for the marginal metrics
    (`evaluate.py:116-118`), not for the paired Δ.
- **W5 — T1/T2 deviate from the protocol and have no comparator.**
  - Protocol §7.4 says "encoder + chemistry heads trained on the UK pool". The scorer instead uses
    `RandomForest[family-log-class]` (`transfer.py`, `bioassess_final.py:260-287`). This is not recorded as a
    deviation in protocol §10.
  - It is single-seed (`seed=0`, against the 3-seed rule of §5).
  - There is no geography-only comparison, so the soft threshold of §15.3 ("UK↔US transfer above
    geography-only") cannot be evaluated from the T entries.
  - Fix: document the deviation; use seeds 0–2; log the paired Δ against `RandomForest[geography]` on the same
    NRSA 2018-19 rows for T1. For T2, state that no comparator exists.
- **W6 — Prior exposure of 2018-19 (disclosed, but understate it nowhere).**
  - `rf_blocked_baseline` (108 registry rows, `scripts/eval/baseline_bioassess_rf.py:55`) fitted and scored RF
    variants *within* NRSA 2018-19. The research-phase RF numbers in RESEARCH_PLAN §5.5 come from 2018-19.
  - No BioSet setting was chosen on it, so Tests A/B stay valid cross-cycle tests. Only Test C (2023-24) is fully
    unexposed.
  - Fix: say so next to every A/B claim. Move the script to `legacy/`, or add the `HoldoutAccessError` guard, so
    it cannot be re-run casually.
- **W7 — Configuration counts and selection effects are not attached to the test entries.**
  - Fix: add `n_configurations_tried` (13 SENTINEL + 7 baseline-side) and the selection note from §10 to the
    `config`/`extra` of every `bioassess_test` row, as §9.1 promises.
- **W8 — GPU nondeterminism.**
  - BioSet and TabPFN on CUDA are not bit-reproducible: embedding backward and attention use atomics, and no
    `torch.use_deterministic_algorithms` is set.
  - Fix: keep the saved prediction parquets as the canonical record and add a `--score-from-predictions` path, so
    the scoring can be repeated exactly. Record the GPU model and driver.
- **W9 — Re-run guard is a deletable file.**
  - Fix: `verify_freeze()` should also refuse when the registry already holds any `experiment ==
    "bioassess_test"` row.
- **W10 — `is_baseline` is mis-set for SENTINEL controls.**
  - `bioassess_final.py:189` marks `BioSet[bioset_st_flat]+geography` and
    `Ensemble[BioSet+geography+TabPFNv2]` as baselines.
  - Fix: use `is_baseline = not model.startswith(("BioSet[", "Ensemble[BioSet"))`.
- **W11 — Process: M5 had no R10 pre-training review before its first training run.**
  - M5 is absent from the status table above, and phase-1 CV started 2026-10-22.
  - The review found no leakage retroactively, but disclose the deviation in RESULTS.md and the paper.
- **W12 — Provenance gaps in development rows.**
  - Two `bioassess_phase1_selection` rows (candidate C vs the TN comparator, 03:32/03:33 UTC) were written from
    `python -c` (`script_sha256: null`).
  - Fix: re-log them from a script, superseding the originals.

### Observations

- O1: The comparator set is a superset of the lead's list. It adds RF[community+context] and its ensembles; the
  TN comparator is `Ensemble[RF[community+context]+TabPFNv2]`, the strongest one on development data. This
  choice is conservative.
- O2: Test C has only 99 re-sampled locations (vs 967 of 1,918 in 2018-19), which makes it the cleanest
  replication.
- O3: `sentinel/models/downstream/common.py:118` hashes `sentinel/models/bioassess/data.py` for D5's freeze.
  D5 also uses 2018-19 as validation and 2023-24 as test.
  - The fixes above touch only the scorer, the freeze builder and possibly `cv.py`; `data.py` must not change, or
    D5's freeze breaks.
  - No D5 registry rows on 2018-19/2023-24 exist yet. D5 results on these cycles must not feed back into M5.
- O4: The selection tie-break "then the simpler training objective" was added after
  `bioset_st_flat_rloss` tied. It is disclosed, and it selects the simpler model, so it is not optimistic.
- O5: The p-value is two-sided, and the scorer sets p = 1 when Δ ≤ 0. In effect this is a one-sided test at
  α/2, which is conservative and fine, but state it in §9.1.

### Verified correct (no action)

- **Holdout guard.**
  - `load_nrsa` raises `HoldoutAccessError` unless `allow_holdout=True`. The only callers that pass the flag are
    in `bioassess_final.py` (`:222,237,257`); `grep` finds no other M5 path.
  - `load_uk(include_reserved=False)` is used by pretraining and by the transfer diagnostic, and the latter
    asserts that no reserved rows are present.
  - The registry has no `bioassess` row on 2018-19, 2023-24 or UK-reserved splits except the disclosed
    `rf_blocked_baseline`.
  - No `final_scored.json`, `final_*` run directory or `bioassess_freeze.json` exists.
  - The `nars_streams` files are unchanged since 2026-10-22 10:09, and their sha256 matches the CV runs'
    `input_sha256`.
- **Location clusters** are 1-km connected components over all four cycles, built from coordinates only.
  Metadata check:
  - 0 clusters span more than one ecoregion in dev+2018-19 or dev+2023-24;
  - 0 violations of the Test A train/test location assert, so the run will not crash there;
  - dev cluster count (3,121 among first visits) is identical with or without test-cycle sites, so test sites do
    not chain development clusters;
  - Test B strata: 951 new and 967 re-sampled of 1,918 first visits with a benthic sample. The definition
    (cluster absent from development rows) is correct.
- **Test A design:** for every ecoregion g, the training rows are exactly the development LOEO training fold
  (same `location_id`s and inner 50-km early-stopping split, seed 0). The held-out ecoregion and 2018-19 never
  enter training.
- **Training-fold fitting:**
  - `CommunityFeaturizer` (taxa, prevalence filter), the geography median fill, the BioSet target and glob
    standardisers, and the allowed-name mask are all fit on training/fit rows only;
  - TabPFN context = training rows only (≤ 6,000; no subsampling at these sizes);
  - no chemistry, MMI, condition or cycle column is an input (unit-tested);
  - geography and sampling DOY are legitimate collection metadata.
- **Comparators** run through the same `predict()` on identical rows with seeds 0–2; ensembles are equal-weight,
  with members memoised per fold. All targets are present in every cycle (aggregate analyte counts
  ≈ 8.7–8.8k of 8,831 visits), so no all-NaN target will abort the run.
- **Holm** (`bootstrap.holm_bonferroni`) is a correct step-down. The paired bootstrap resamples location clusters
  jointly for both models on the same rows (`evaluate.paired_vs`).
- The 14 hashed files match the candidate record as of 2026-10-24 23:34. `pytest tests/test_bioassess*.py`:
  29 passed.

### Checklist summary

| Category | Checked | Passed | Failed | N/A |
|---|---|---|---|---|
| Test data untouched | 6 | 6 | 0 | 0 |
| Leakage (clusters, Test B strata, fold fitting, TabPFN context, features) | 8 | 7 | 1 (W1, inputs only) | 0 |
| Freeze integrity (hash coverage, refusal, marker, frozen params) | 6 | 2 | 4 (C3, C4, C5, W9) | 0 |
| Statistics (Holm, paired bootstrap, seeds, claim per target) | 7 | 3 | 4 (C1, C2, W3, W4) | 0 |
| Repeatability (determinism, dry run, provenance) | 5 | 1 | 4 (C4, W8, W12, untracked code) | 0 |

### Gate decision

**Not yet.** Fix C1–C5 (and preferably W1–W5, W7, W9, W10) in `bioassess_final.py`,
`bioassess_freeze_candidate.py` and §9.1 of the protocol. Then:

1. Run `--dry-run-dev` end to end.
2. Regenerate the candidate record.
3. Submit the diff for re-review.

Once the re-review passes, the lead may copy the record to `bioassess_freeze.json` and run the single pass. No
development result has to be redone.

---

## Sim — SENTINEL-Sim scenario engine (Goal 3) — pre-training gate (2026-10-25)

**Review mode:** pre-training (review skill §1–5), before the first real Sim training run.
**Scope:** `sentinel/models/sim/{physics,data,guard,model,train,baselines,hindcast,dsst,scenario,checks,latent,runner}.py`,
`scripts/train/sim_train.py`, `scripts/eval/{sim_hindcast,sim_checks,sim_dsst}.py`, `tests/test_sim.py`,
`docs/preregistration/sim_protocol.md` (v1.1). Reused, not edited: `sentinel/models/hydromet/physics.py`,
`sentinel/models/baselines/sonde_baselines.py`, `sentinel/models/sonde/data.py`, `sentinel/data/sonde_splits.py`,
`sentinel/models/downstream/riverpulse.py::load_labels`.
**Reviewer:** independent adversarial reviewer agent (read-only; ran `pytest tests/test_sim.py` and inspection snippets on the
cached validation dataset). The causal-ladder and threshold code (`sentinel/analysis/`) is **not** in this gate; it gets its
own addendum before its first real-data run.

### Summary
**Overall status: 🟡 PASS WITH CONCERNS → 🟢 PASS after the fixes below (re-verified by the workstream: tests 22/22 pass).**
No leakage found. Verified: split sha256 fd4bb14f…; cache holds 1,170 sites (train 889 / cal 148 / val 133), zero test-role
sites, dates ≤ 2023-09-30; every cal site is train-role; no basin group spans roles; normalisation from train-role non-cal ×
train block (cool years only for DSST); the loss mask excludes warm-up and non-train-block days; no water-quality observation is
a network input; the error correction uses only residuals ≤ 3 days old at or before the issue day; φ/quantiles from train-block
(issue, target) pairs; same rows for all models; the "hindcast with observed forcings" label is correct. Physics verified: exact
hourly affine step and Hillis–Steele composition order, units (areal g O₂ m⁻² d⁻¹ ÷ depth = mg L⁻¹ d⁻¹), Σ light weights = 1,
Schmidt polynomial, Raymond 2012 constants, fp32 vs fp64 integration ≤ 1.5e-3 mg/L over 545 days.

### Critical issues (all fixed before training)
| # | Finding | Fix |
|---|---|---|
| C1 | Freeze hash omitted scoring code (`sonde_splits.py`, `sonde/data.py`, `runner.py`, `checks.py`, `latent.py`, `dsst.py`, `scenario.py`, `sim_checks.py`); checkpoint hashes never verified | `guard.FROZEN_CODE` extended (+ `sim_cal_sites.json`); `guard.verify_weights()` checks every test-scored checkpoint against `FREEZE_sim.json` |
| C2 | Inclusive inequalities let a non-responsive model pass M1/M3/M4/M7 | Strict criteria with minimum effects: M1 ΔT_w ≥ 0.2 °C; M3 ΔDO_mean < 0 and ≥ ½ the solubility-only change; M4 strict; M7 ΔGPP ≤ −5%, ΔDO_max < 0 (protocol v1.2 §5.2) |
| C3 | DSST pass rule rested on 6–7 cal basin groups | DSST evaluated on cal + val-role sites × hot train-block years (19 groups); ≥ 10 groups required (`sim_dsst.py`) |

### Warnings (resolution)
1. Extreme initial K600/depth; DO clamp blocked gradients → K600 floor 0.3 d⁻¹; loss uses the unclamped DO state (reported DO stays ≥ 0).
2. Discharge `interpolate(limit=7)` partially filled long gaps; fallback median could include val-period Q → only gaps ≤ 7 d interpolated (unit test); fallback = train-period median when ≥ 30 days (discharge is an input, documented).
3. M5 C–Q slope used filled discharge → observed (unfilled) discharge only; ΔSC over all block days stated.
4. Scenario range ends untested → new M9 checks at −3/+4 °C, Q × 1.5, SW × 1.1; precipitation-only = "model association", no check.
5. Holm/seed variance (R7) → intersection–union rule and per-seed CRPSS range documented as deviations (protocol §4.5).
6. Wrong Powell counts in protocol → corrected (val 21 eligible sites / 5 groups; test 104).
7. Cache key / cal list drift → cal list frozen in `docs/preregistration/sim_cal_sites.json`; cache key hashes all data code + processed-data manifests + cal file.
8. Comparators used WY2006–07 → masked; all comparator statistics now train-block only.
9. "Never in memory" wording → guard docstring corrected (files read whole by the shared M1 loader, cut immediately, never retained).
10. DSST windows' warm-up inputs may include hot years; sampler ignored `valid_days` → sampler counts only loss-eligible days; input overlap documented (§5.4).
11. cuDNN determinism; NaN cal loss crash → `cudnn.deterministic=True`, `benchmark=False`; explicit failure if no finite cal loss.

### Observations
- M2 is a code-consistency identity, not physical evidence (stated in protocol).
- In the DSST, climatology fitted on cool years is cold-biased in hot years (favours the Sim vs climatology); the persistence comparison and the random-years control offset this — interpret accordingly.
- The physics-only DO baseline uses observed water temperature at the target day (a stronger oracle than the Sim gets).

### Checklist
| Category | Checked | Passed | Failed (fixed) |
|---|---|---|---|
| Code correctness | 14 | 13 | 1 (W1, fixed) |
| Data leakage | 12 | 12 | 0 |
| Experimental design | 10 | 5 | 5 (C2, C3, W4, W5, W6 — fixed/documented) |
| Reproducibility | 6 | 3 | 3 (C1, W7, W11 — fixed) |
| Code quality | 6 | 5 | 1 (W11 — fixed) |

**Gate decision:** validation-only Sim training (registry workstream `sim`) may start. Test-A stays locked until `FREEZE_sim.json`
exists and a post-results review has passed.

---

## Sim — analysis addendum (causal ladder, conductivity thresholds, E1/E3) — 2026-10-25 review, recorded 2026-11-08

**Review mode:** pre-run addendum to the Sim gate above, before any real-data causal, threshold, E1 or E3 run.
**Scope:** `sentinel/analysis/**` (`causal/{prep,discovery,controls,ladder,step4,synthetic}.py`, `thresholds.py`,
`insights/{e1,e3}.py`), `scripts/eval/sim_{causal_step1,causal_real,thresholds,e1,e3}.py`, `tests/test_sim_analysis.py`.
**Reviewer:** independent adversarial reviewer agent (2026-10-25; the verdict arrived after the pause and was relayed by
the lead in `docs/handoff/sim.md`). Recorded here by the Sim workstream on 2026-11-08, on resuming.

### Summary
**Overall status: 🔴 FAIL** — no real-data causal / conductivity-threshold / E1 / E3 run until the five blocking items
are fixed or declared in `docs/preregistration/sim_protocol.md`.
**Passed:** no leakage; no test-role sites and no Test-A dates in any analysis; synthetic data kept separate
(`split="synthetic"`, manifest `synthetic=True`); synthetic ground truth correct (11 direct links).

### Blocking issues
| # | Finding | Required fix |
|---|---|---|
| A1 | FDR correction never reaches the graphs. tigramite's `fdr_method` adjusts `p_matrix` only, so links are accepted at the uncorrected `pc_alpha` 0.01 | Prune graph links whose BH-adjusted p exceeds alpha, in every replicate |
| A2 | Time-reversal negative control N2 is invalid. It fired in 5/5 seeds on a correct model, because reversed autocorrelated data produce weather→river links by construction | Redefine as a forward-vs-reversed comparison, or drop it from the pooled FPR |
| A3 | Distant-weather control N3 runs constrained, but the protocol says unconstrained | `constrained=False` in `scripts/eval/sim_causal_real.py` |
| A4 | The conductivity-threshold bootstrap treats revisits as independent sites (4,633 samples vs 1,928 `unique_id`). The TITAN bootstrap resamples samples, not sites | Resample physical sites in every bootstrap, TITAN included |
| A5 | The Appalachian 300 µS/cm positive control is not testable: 102 eligible samples and 3 reference samples, and the reference-taxa and size filters are skipped silently | Add a pre-declared minimum-data rule or an "underpowered / not testable" label |

**Disclosure:** the reviewer computed Appalachian HC05 ≈ 787 µS/cm (95% CI 617–964) on real data. Any later change
to that stratum's analysis is **outcome-informed** and must be logged in `sim_protocol.md` §12.

### Warnings
W1 causal step 4 has no CLI, and the hourly (diel) controls are missing · W2 the causal real-data script has no
`TrainingRun` and no step-1 gate check · W3 TITAN implementation differs from TITAN2 (undocumented) · W4 E3 misses
some Hare et al. 2021 data rules and uses an undeclared dam-storage proxy · W5 E1 Newey–West errors ignore the gaps
between summers (and within them) · W6 some registry `split` labels are not listed in protocol §11 · W7 the E1/E3
scripts have no thread limits.

### Resolution (Sim workstream, 2026-11-08; before any real-data run)
| # | Fix |
|---|---|
| A1 | `discovery.fdr_prune()`: links whose BH-adjusted p > alpha are removed from the graph of the full fit and of every bootstrap replicate (tigramite returns adjusted p per replicate). Unit test `test_fdr_prune_removes_links`. Step 1 was re-run with the fix. |
| A2 | N2 was dropped from the pooled FPR. It is replaced by the diagnostic **N2-asym**: for each weather→river pair, the detection frequency in reversed data minus that in forward data, both unconstrained. It is reported per class with no gate. Protocol §7 / §12. |
| A3 | `constrained=False` for the N3 run. |
| A4 | Bootstrap unit = **physical site cluster**: NRSA `unique_id` (2018-19 and 2023-24) merged with coordinate clusters ≤ 1 km (`bioassess.data.location_clusters`), because 2008-09 and 2013-14 have no `unique_id`. XC95/HC05, the TITAN purity/reliability bootstrap and the community change-point CI all resample clusters. The number of clusters is reported. |
| A5 | Pre-declared testability rule (protocol §8, logged in §12 as **outcome-informed**): a stratum is testable only with ≥ 300 samples, ≥ 150 site clusters, ≥ 19 genera passing the occurrence filter (so HC05 interpolates rather than clamps at the lowest rank) and ≥ 10 reference samples for the reference-taxa filter. Otherwise it is labelled "not testable (underpowered)", the positive control is neither "replicated" nor "not replicated", and the estimate is still reported, labelled. Every skipped filter is recorded in the output. |
| W1–W7 | See protocol §12 (2026-11-08 entry): step-4 phase + diel phase added to `sim_causal_real.py`; `TrainingRun` + step-1 gate check; TITAN2 differences documented; Hare rules (T_w < 1 °C excluded, no ≥ 30-day gaps, Δφ < −10 d dropped, −10…0 → 0, ≥ 2 years) + the dam-storage proxy declared; calendar-lag HAC; split labels added; thread limits. |

## Fuse — SENTINEL-Fuse multimodal fusion (Goal 2) — pre-training gate (review 2026-10-25; fixes verified 2026-11-08)

**Review mode:** pre-training, before the first real (non-smoke) deep-model run.
**Scope:** `sentinel/data/fuse_tiers.py`, `fuse_tier_c.py`, `fuse_tier_w.py`; `sentinel/models/fuse/{splits,data,baselines,stage0,model,train,plugins}.py`;
`scripts/data/build_fuse_tiers.py`, `scripts/train/fuse_stage0.py`, `scripts/train/fuse_train.py`, `scripts/eval/fuse_compare.py`;
`tests/test_fuse.py`; `docs/preregistration/fuse_protocol.md` and the frozen splits `fuse_tier_d_split_v2.json` (`20adbe59…`),
`fuse_tier_c_split.json` (`e2fd5de9…`).
**Reviewer:** independent adversarial reviewer agent (read-only), 2026-10-25. The full findings were carried in
`docs/handoff/fuse.md` (the section was not appended before the 2026-10-25 pause); this entry records them with their
resolution. Fix verification by the workstream, 2026-11-08: `pytest tests/test_fuse.py` 26/26 pass (7 new tests for the fixes);
`scripts/data/make_fuse_splits.py check` regenerates both frozen split files byte-identically.

### Summary
**Overall status: 🟡 PASS WITH CONCERNS → 🟢 PASS for development-CV training after the fixes below.** No test-row leakage found:
test-role rows are removed right after loading (`stage0.drop_protected`, asserted), freshness rules (chemistry strictly before d,
16S ≤ d, sensor windows ending at d, weather analyses ending at d) are implemented as pre-registered, fold-wise fitting (16S
CLR/PCA, scalers, target standardisation) uses training rows only, and the plug-and-play bit-identity test passes. Open items
that do not block development-CV training are listed at the end.

### Critical issues
| # | Finding | Resolution |
|---|---|---|
| C1 | Left-censored lab targets used the published sub-DL value as the Tobit bound | Fixed (`data.tier_d_target_matrix`: bound = detection limit; censored rows without a DL dropped); test `test_censored_lab_target_bound_is_detection_limit` |
| C2 | Decision rules ambiguous and deviations from the plan undocumented (P1 all-vs-any target after Holm; which modalities count for P3; P2 "5-km" scheme; P4 margin; P3 normalisation) | Fixed: protocol v1.1 §8 decision rules + §11 deviations log (P1 ≥ 1 target vs both comparators after Holm + leave-one-domain-out sign check; P3 non-hub only, ΔCRPS / CRPS_hub, eligibility ≥ 200 rows & ≥ 3 test sites; P2 5-km = buffer + distance strata; P4 2 % margin) |

### Warnings
| # | Finding | Resolution |
|---|---|---|
| W1 | Censored `mc_log10` scored in Stage-0 | Fixed (censored rows trained on as bounds, never scored); GBM censoring strategy documented (protocol §6) |
| W2 | Tobit term −log Φ had vanishing gradients for z ≪ 0 | Fixed (`torch.special.log_ndtr`) |
| W3 | Tier C satellite features could postdate the sample | Fixed in code (`dt_days ≤ 0`); Tier C rebuild queued 2026-11-08 (after Stage-0 v3) |
| W4 | WQP/NLA duplicate samples | Fixed in code (`drop_nla_duplicates`); same rebuild; split roles unchanged (counts change slightly; protocol §2.2) |
| W5 | Congener-sum labels mixed with total microcystin | `y_gt8_total` is the P2 primary label; congener rows only in the `y_gt8` sensitivity (protocol §2.2) |
| W6 | 22–24 smoke registry entries under `val_cv_lodgo` | Superseded by 24 entries with split `smoke` (2026-11-08) |
| W7 | Paired evaluator joined OOF files by positional row only | `fuse_compare.py` asserts (site, date) equality of Stage-0, Fuse and the frame |
| W8 | Bootstrap units (Tier C should cluster by 20-km block; Tier D domain sensitivity) | Tier C CIs resample 20-km blocks; Tier D site clusters + leave-one-domain-out range (protocol §10) |
| W9 | Few clusters (8 Tier D test sites) → percentile CIs too narrow; P3 feasibility unknown | CI widened by sqrt(G/(G−1)) (`stats.cluster_ci`); P3 feasibility counted on presence only (all 8 modalities at all 8 test sites after the NEON P2 build; protocol §8.2) |
| W10 | TabPFN comparator missing; tuning parity for searches; phyto on/off | Phyto switch added (`--exclude-mods phyto`); tuning-parity rule in protocol §6; TabPFN comparator `scripts/train/fuse_tabpfn.py` (shared scikit-learn shim of `chemtox/tabpfn_model.py`, quantile CRPS, anchored like `concat@a`). **Open:** its GPU run (queued on GPU 4; CPU is too slow) |
| W11 | Split generator not in the repo | `scripts/data/make_fuse_splits.py` (regenerate + sha check; write refuses to overwrite) |

### Observations
* Lab targets TN/TP/DOC logged; Tier W verifies M1's split sha — done.
* cuDNN determinism set in `fuse_train.py`; OneCycle schedule with early stopping kept (best-epoch checkpoint), documented.
* Post-review additions (2026-11-08, covered by the same tests): persistence-anchored comparators and Fuse option (anchor removed
  whenever its owner modality is absent or dropped — `test_anchor_removed_when_owner_modality_dropped`); hydro (single HOR,
  past-only anomaly — `test_single_hor_daily_and_trailing_anomaly_are_causal`), macroinvertebrate and periphyton modalities
  (latest event ≤ d, ≤ 120 d — `test_bio_block_latest_event_on_or_before_and_age_cap`); a-priori hard strata
  (`test_hard_strata_use_training_thresholds_and_tails`).

### Open (non-blocking for development CV; blocking for test scoring)
TabPFN comparator run; equal-count random search for `concat` whenever Fuse hyper-parameters are searched; M3 fold-aware OOF
(cross-fitting rule §5) before any M3 output enters Fuse; `fuse_freeze.json` and the Holm p-value implementation in the final
scoring script; post-results review before any number leaves the repo.

### Round 2 re-review of the addendum fixes — 2026-11-08
**Reviewer:** an independent adversarial reviewer agent (read-only). It ran the Sim test suites (37/37 passed)
and inspection snippets on synthetic arrays only. **Verdict: FAIL → fixed by the workstream the same day (see
below); no real-data causal / threshold / E1 / E3 run had started.**
- **Status of round-1 items:**
  - A1, A2, A3, A4, W1, W3, W4, W5, W6, W7: fixed.
  - A5 and W2: partially fixed (see C1, W-b, W-c).
  - Verified correct: the BH family vs tigramite 5.2.10.1 semantics (symmetric contemporaneous p, bootstrap
    replicates return p_matrix); site-cluster bootstraps in XC95/HC05, TITAN and the CP CI; calendar-time HAC
    equals statsmodels HAC on contiguous data (1e-15); no leakage (validation guard, train-block thresholds,
    region renormalisation, diel solar time).
- **C1 (blocking):** `scripts/eval/sim_thresholds.py` had a SyntaxError, an `else:` detached by a line that
  had been moved. **Fixed**; new test `test_all_sim_scripts_parse`.
- **Warnings, all fixed:**
  - **W-a:** J-PCMCI+ is now FDR-pruned; LPCMCI is labelled uncorrected.
  - **W-b:** testability counts the taxa that enter the *primary* (reference-taxa) HC05.
  - **W-c:** step-1 outputs go to `<tag>/<site_set>/`. `gate_config` re-derives the dev winner
    (`ladder.step1_selection`), requires exactly one confirmed tag and complete runs.
  - **W-d:** the step-2 cross-correlation baseline uses the gated FDR level.
  - **W-e:** a taxon absent from a bootstrap replicate gives NaN, not index 0.
  - **W-f:** region runs are checked against their `region_holdout.json`, and per-fold scores are added.
  - **W-g:** frozen "underpowered" cells are labelled in the verdicts.
  - **W-h:** the causal_real cache guard now includes the step-1 tag and a code hash.
  - **W-i:** the step-4 docstring is corrected, and the step 3 / step 4 site-overlap limitation is added to
    protocol §7.
  - **W-j:** tests added for the selection rule, the Hare lag clip and classes, and the gap rule.
  - **Minor:** lagged LPCMCI `o->` now counts as adjacency, and E1 trait imputation happens inside the CV folds.

---

## M4 ChemTox — round-4 pre-test gate (2026-11-08)

**Verdict: 🟡 PASS WITH CONCERNS. The frozen configuration (`630b43e905d9b852`) may be scored once on the
sealed prospective set**, after the lead has authorised it and after the pre-pass items W1–W3 below are
done. W1–W3 are documentation and runbook steps only: no M4 code change, no re-freeze, no re-review.
Do them before the pass, because once scores exist they would be post hoc.

There are no critical issues. Both round-3 blockers are fixed and verified on the **real** frozen block:
- **N5:** the NNLS stack resolves its renamed member `gpboost_multitaxa` through the frozen `oof_sources`.
  The weights refit cleanly, and the stack and mean CV scores recompute exactly.
- **N6:** `__published_reference__ = rf_published_mwlogp` sits inside the hashed block. The scorer exits if
  it is absent.

The prospective set has not been read by any model-scoring code. The registry has 0
`chemtox_prospective_test` rows.

**Reviewer constraints honoured.** I did not run the pre-flight or the scorer's `main()` and did not use
`--i-have-approval`. I modified no M4 file, used no GPU, used ≤ 6 CPU threads and made no git commit. The
prospective parquet was opened only for identifier, flag and feature columns. The target columns
`result_conc1_mean*` were never read by any probe. Probes are in
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/r4/`:

| Probe | What it checks |
|---|---|
| `dirs.py` | every cycle-2 run directory |
| `overlap.py`, `overlap2.py` | identifier overlap with ADORE |
| `adore_stage.py`, `why*.py` | at which ADORE pipeline stage the prospective rows and chemicals existed |
| `n5.py` | the real `nnls_weights()` on the real frozen block, OOF integrity, recomputed ensemble CV |
| `tracks.py` | all 9 tracks through the scorer's metric helpers with **random** targets |

**Versions reviewed.**

| Item | sha256[:16] | Matches |
|---|---|---|
| `chemtox_cycle_select.py` | `f5f70abc16821d19` | ✓ freeze `code_hashes`, round-4 note |
| `chemtox_score_prospective.py` | `a0cb0385226e9aaf` | ✓ |
| `scripts/train/chemtox_cycle.py` | `258b00023693e4da` | ✓ |
| `prospective.py` | `74765aea3243aa1e` | ✓ |
| `cycle.py` | `920d58c258e9cc30` | ✓ (§6) |
| builder | `80a617aecd89610a` | ✓ |
| data (full sha256 recomputed) | `3b2fc4398db940a2…dc54d9` | ✓ |
| `selection.json` | `bf2ed61c…237328` | ✓ |
| `frozen` block (recomputed) | `630b43e905d9b852` | ✓ |
| `sentinel` package | `0be577835889f982` | ✗ freeze says `ca39f9ea257fcb1a` (other workstreams; see W6) |

No file under `sentinel/models/chemtox/`, `scripts/{train,eval}/chemtox_*` or `configs/chemtox/` is newer
than `selection.json`. `sentinel/eval/bootstrap.py`, `sentinel/core/registry.py` and
`sentinel/data/sources/adore.py` are dated 2026-10-22.

**Tests:** `pytest -q -p no:cacheprovider tests/test_chemtox*.py` (6 threads, CPU only): **129 passed** (exit 0; 3 library warnings). The real registry, TRAINING_LOG and prospective run directory are unchanged afterwards (0 prospective rows; still 3 dry-run log entries from 2026-10-24).

### What was verified (by check requested)

**(1) The prospective set is still unread by model-scoring code; there is zero overlap with ADORE.**
- **Registry.** `results/registry.jsonl` has 0 rows with experiment `chemtox_prospective_test`, or with any
  split starting `prospective/`. The 613 `chemtox` rows are ADORE test/CV/search rows plus 114
  `adore_chemtox_cycle2` rows. The 2026-11-08 freeze logged 14 models (12 single + 2 `cycle2_ensemble_*`) ×
  their metrics. `is_baseline` is True exactly for `xgboost`, `rf` and `readacross` (round-4 change 6 ✓).
- **File access.** `prospective_path()` is called only by:
  - the scorer;
  - `prospective_keys()`, the exclusion guard (identifier sets only);
  - the builder;
  - tests.

  No `scripts/train` or `scripts/eval` path other than the sealed scorer opens it. The run directory
  `$SENTINEL_RUNS/chemtox/prospective_test/` holds only the 2026-10-24 dry-run `score.log`/`dry_run.json`.
  Those are three pre-flight runs: features and fits on training rows only, "no target value was read",
  frozen hashes `ef40897…`/`04e6265…` from scratch selections (see W4).
- **Overlap** against the 158,330 ADORE curated rows (12 challenge files + `ecotox_mortality_filtered`;
  2,408 CAS / 2,407 InChIKeys / 33,448 `result_id`s):

  | Track | Rows | CAS overlap | InChIKey overlap | `result_id` overlap |
  |---|---|---|---|---|
  | `fish_novel` | 2,313 | 0 | 0 | 0 |
  | `crust_novel` | 295 | 0 | 0 | 0 |
  | `algae_novel` | 3 | 0 | 0 | 0 |
  | `noncarbon` | 8,774 | 0 | 0 | 0 |
  | `temporal` | 115 | 52 (ADORE chemicals, by design §1.2) | — | 0 |

  Track sizes reproduce §1.3/§1.3a exactly: primary 1,912 / 301, strict 1,362 / 236, crust 230 / 112,
  crust strict 154 / 74.
- **Guards.** `assert_prospective_chemicals_absent` runs on:
  - every training frame of the pass (`build_training_frame`);
  - every extra-row source (`load_extra_rows`, `adore_other_taxa_rows`).

  It checks CAS, InChIKey, `result_id` and temporal `reference_number` (and parent key for augmentation).
  `ds.assert_no_chemical_leakage` runs per dataset.

**(2) The stage-2 block matches `selection.json`; tuning parity; ensembles; N5 with a renamed member.**
- **Stage 2 ↔ JSON.** Plan lines 402–459 were regenerated with the M4 renderer (`stage2_md.py`) from
  `selection.json`. They are **identical** to the appended text, and identical to the scratch `stage2.md`
  the finalize script wrote. The renderer prints all non-device parameters, every hash and the published
  CV scores. Its feature column is a summary (e.g. it omits `published=True`); the hashed JSON is
  authoritative.
- **Parity (probe `dirs.py`).** Every `*_dm` search directory has exactly 32 distinct sampled indices
  (0–31), 32 screens and 5 finalists × 5 folds, all `_deployment_masking=True`. The exceptions:
  - `gpboost` (none): 3 finalists, 2 failed with GPBoost CG NaN.
  - `chemeleon_tabpfn`: config 0 OOM plus a retry; see (6).

  No entry is the best of more than 32 configurations:
  - `lightgbm` / `lightgbm_aug` and `gpboost` / `gpboost_multitaxa` are separate entries, one directory each.
  - The `_dm_cfg-pub_*` runs are excluded from every search by the `endswith("_dm")` filter, so Z1 is
    resolved.
  - The unmasked legacy directories (`*_extra-none` without `_dm`) are excluded.
  - Comparators `xgboost`, `rf`, `readacross` are re-tuned with the same 32-configuration budget, the same
    screen protocol and the same caps policy (`SCREEN_CAPS`).
  - `chemprop` = the phase-1 32-config winner, re-run masked in two explicit variants; the better is frozen
    (O3).
- **Frozen entries ↔ the records they came from.** For every component:
  - `params` / `features` equal the 5-fold record's;
  - `cv_rmse` equals the record's `cv_mean`;
  - the OOF file recomputes the record's pooled RMSE to 4 dp. Pooled values: catboost 0.9515,
    chemeleon_gpboost 0.9512, chemeleon_lightgbm 0.9610, chemprop 0.9547, gpboost_multitaxa 0.9724.

  Each OOF file has length 26,114, finite on exactly the 21,981 training-fold rows, and NaN on all 4,133
  ADORE-test rows. So the NNLS refit cannot see a held-out row.
- **Ensemble membership** was applied as pre-registered. The rule ("every candidate with masked CV RMSE
  below the strongest re-tuned comparator") is coded in `m4_queue/finalize.sh` (12:31, before the
  `chemeleon_tabpfn` result existed). It is stated in stage 2. It computes the bar over `xgboost`, `rf`
  and `readacross` only (0.9733) and selects catboost, chemeleon_gpboost, chemprop, chemeleon_lightgbm and
  gpboost_multitaxa. It correctly excludes:
  - `lightgbm_aug` 0.9797;
  - `lightgbm` 0.9865;
  - `gpboost` 1.0158;
  - `chemeleon_tabpfn` 1.0427.
- **N5 path, renamed member.**
  - The frozen ensembles list `components` by frozen name, with `component_families` (`gpboost` for
    `gpboost_multitaxa`).
  - `oof_sources` is keyed by frozen name and points at
    `gpboost_t-F2F_split_occurrence_extra-adore_taxa_all_dm/oof_gpboost_000.npy`, the runner's naming
    `oof_{family}_{idx}`.
  - The real scorer `nnls_weights(frozen, "ensemble_stack_nnls", …)` returns catboost 0.1945,
    chemeleon_gpboost 0.2575, chemeleon_lightgbm 0.0423, chemprop 0.3421, gpboost_multitaxa 0.1636.
  - `EN.stacked_ensemble` / `average_ensemble` on the same files recompute CV **0.921166 / 0.916383** =
    frozen.
  - The scorer runs `nnls_weights` and the component-presence check **before any fit and before the
    dry-run branch**, so the pre-flight now exercises N5 (round-3 Y1 resolved).
  - `build_training_frame` handles both frozen `extra` values: `ecotox` goes through
    `PRO.augmentation_rows`; `adore_taxa_all` takes all 7,334 t-C2C/t-A2A rows, minus prospective chemicals.
    Columns are aligned exactly as in CV. `extra_weight` is 1.0 everywhere.
- **Fit-path parity CV ↔ scorer.** Both call `CYC.make_model(MODEL_ALIAS…)` and an equivalent
  `FeatureBuilder`. `valid_rows` is used in CV only to exclude the fold's chemicals from chemprop's
  auxiliary data, never for early stopping, so fitting with `valid_rows=None` at test time is the same
  model. Seeds are [0, 1, 2] for every entry, with per-seed ensembles (Z3 resolved).

**(3) The masking protocol is identical at test time.** The prospective file has `chem_ws` and `chem_mp`
**100% NaN** (0 of 11,500 non-null). `chem_mw` and `chem_rdkit_clogp` are 100% present. Every model
receives the same `pro` frame.
- In CV, `mask_deployment_missing` blanks exactly those two columns on the validation rows.
- At test, they are already missing.
- Both paths impute through the same `PublishedFeatures` training-row mean.

All candidates and re-tuned comparators use `chemical=mw_logp`, so masking affects only `xgboost_published`
and `rf_published` (`four`). Those two are scored exactly as their masked CV runs (1.4600 / 1.2561) were.
The reference is `rf_published_mwlogp`, which is unaffected.

**(4) Holm family, comparators and bootstrap unit.**
- **Roles.** The frozen block has 9 single candidates + 2 ensembles = **11** candidates, and 7 frozen
  comparators. The scorer's `candidate_models` = predictions not in comparators ∪ trivial, giving the same
  11.
- **Holm.** Holm is applied separately for `published_reference` (= `rf_published_mwlogp`, fixed pre-test)
  and `best_baseline` (per-track minimum over all 10 comparators, including the re-tuned XGBoost bar).
  A rejection counts only if `diff < 0`.
- **Claims.** A claim needs both comparisons, so the two families need no cross-correction. Requiring
  `best_baseline` is at least as strict as the re-tuned-XGBoost bar the lead named. `vs_xgboost` and every
  other `vs_*` are reported uncorrected.
- **Bootstrap.** Paired cluster bootstrap over `test_cas` (chemical), 2,000 resamples, seed 0. This is the
  same function as the ADORE pass. The p-value is two-sided. `holm_bonferroni` is a correct step-down.
- **Probe `tracks.py`.** All 9 tracks run through `track_mask` → `E.report` → `E.bootstrap_report` →
  `paired_delta_ci` → Holm with random targets, without error, including `algae_novel` (3 rows / 2
  chemicals; see O5).

**(5) The scorer refuses a second pass, verifies hashes and writes only to the registry.** The order in
`main()` is:
1. Approval flag.
2. Data sha256 (exit 3).
3. Frozen-block hash (5).
4. `__published_reference__` present (7).
5. Registry check over every (model, track) including trivial baselines and ensembles, before any data
   load or fit (6).
6. Ensemble components and NNLS (9).
7. Fits.
8. The target column, read only after all fits.

Registry writes go through `log_result` only, with `supersedes` only under `--allow-rescore`. The other
writes are artifacts in the run directory (`pred_*.npy`, `score.log`, `prospective_results.json`) and the
`TrainingRun` log line. The mocked end-to-end test asserts rc 0 → second pass rc 6 → bad component rc 9.

**(6) Deviations and disclosure.**
- **`chemeleon_tabpfn` config 0.**
  - It was lost to an external CUDA OOM in the 11-08 screen and re-screened alone (`--indices 0 --tag
    screen0retry`, same sampled configuration, same folds 0–2), after the finals.
  - Its score of 1.3485 ranks 26/32; the 5th finalist is 1.0658, so the finalist set would not have
    changed.
  - The OOM error record is kept and counted in `n_failed`; `full_records` dedups by index.
  - Disclosed in the round-4 note, but **not** in `RERUN_NOTE.txt`, the handoff or stage 2 (O1).
- **`chemeleon_gpboost` screen/final gap.** This is a fold-subset effect, not a screening artefact. On the
  same folds 0–2, the uncapped finalist config 0 scores 0.9248 (screen 0.9222) and config 6 scores 0.9322
  (screen 0.9194). Folds 3–4 are harder (5-fold 0.9504 / 0.9636). The screen's top 5 are exactly the
  finalists. No re-screen was run, which keeps the 32 budget. Disclosed in the handoff.
- **`gpboost` (none).** 2 of 5 finalists failed (CG NaN) and the family was frozen from 3; this is
  disclosed.

### Critical issues

None.

### Warnings

**W1 (do before the pass). Half of the primary track consists of chemicals that ADORE's own pipeline had
and filtered out. The plan does not say so.** Plan §0/§1 says "curated with ADORE's own rules … only the
chemicals are new". The probes `adore_stage.py` / `why*.py` read only the ADORE release and identifier
columns.

**977 of the 1,912 primary rows (51%; 117 of 301 chemicals)** are rows of ADORE's own
`processed/ecotox_mortality_processed.parquet`:
- They have already passed ADORE's row-level rules (effect, endpoint, duration, concentration type, media).
- They were dropped at ADORE's **chemical-level** filter (processed → filtered: 3,295 → 2,408 chemicals).
- For the strict track the figure is 553 of 1,362 rows.

Every one of ADORE's 2,408 chemicals has `chem_ws`, `chem_mp` and `chem_mol2vec_allowed=True`. Of the 136
primary chemicals in ADORE's property table:
- 47 lack WS/MP;
- 71 of the remaining 89 have `chem_mol2vec_allowed=False`.

So these chemicals are new to ADORE **because ADORE could not featurise them**, not because they were
tested later. This is not leakage (no ADORE pre-filter file is training data), and every model is scored
on the same rows. It is, however:
- an undisclosed selection mechanism that probably explains much of the composition shift in §1.3b
  (637 of the 977 rows are `organic_multicomponent`);
- a stratum readers will ask about.

**Fix (documentation, before the pass):**
1. Add a §1.3c to the plan stating these numbers.
2. Correct the "only the chemicals are new" wording.
3. Pre-register a reported breakdown `adore_prefilter_seen` (true/false) with a frozen, sha256-hashed
   side file of the 977 `result_id`s (or 117 CAS), computed from the ADORE release only.

The breakdown can be computed after the pass from the saved `pred_*.npy` files, so the scorer does not
change. Doing this after seeing scores would be post hoc.

**W2 (do before the pass). The bootstrap unit is CAS, but 49 of the 301 primary CAS fall into 18
multi-CAS parent-structure groups** (salts and stereoisomers of each other; strict track 16 CAS in 7
groups). Resampling them independently makes the CIs and Holm p-values mildly anti-conservative. The plan
pre-registers "chemical", so the primary analysis stays as it is.

**Fix:** pre-register a sensitivity CI clustered by `parent_key` for the primary comparisons. It can be
computed post-pass from `pred_*.npy`. State that a claim reported with only the CAS-clustered CI is
qualified if the parent-clustered CI includes 0.

**W3 (runbook, before the pass). The `_trainpart` sensitivity pass overwrites the primary pass's
`prospective_results.json`** (round-3 Y3, still open: fixed path `out_dir / "prospective_results.json"`).
§4.4 defines the decision fields (`holm_rejected_vs_*`, `holm_family_vs_*`, per-track `best_baseline`)
**in that file**. The registry has the per-model `holm_rejected` flags but not the `best_baseline`
identity or the family list.

**Fix:** immediately after the primary pass and before any other invocation:
1. Copy `prospective_results.json` (and `score.log`) to `prospective_results_primary.json`.
2. Record its sha256 in the handoff and RESULTS.md.
3. Make it read-only.

Alternatively, fix the filename in the scorer, which needs a hash update and a short re-check.

**W4. The round-4 note says "`--i-have-approval` has never been used". That is inaccurate.**
`TRAINING_LOG.md` lines 4667/5044/5267 record three `chemtox_prospective_test:scoring` runs on 2026-10-24
(22:28, 22:50, 23:00). The scorer cannot start a `TrainingRun` without the flag. They were `--dry-run`
pre-flights against scratch selections:
- the first was on the previous data file `ea078629…`;
- the run directory log shows features and fits on training rows only, "no target value was read";
- the registry has no entry from them.

The test suite also passes the flag (rc 2/3/5 checks and the mocked e2e on a target-stripped copy).
Round 3 already noted these runs. No leak, but the statement should be corrected to say "never used for a
non-dry run; three dry runs on 2026-10-24".

**W5. The scorer has no finiteness guard on the real predictions.** The dry run checks finiteness on a
3,000-row fit only. In the real pass, `preds[name] = P.mean(0)` is unchecked. A NaN model would:
- get NaN metrics in the registry;
- drop silently out of the Holm family (`pvals` filters non-finite p-values, which enlarges the others'
  power);
- corrupt `best_baseline`'s `min`.

GPBoost's CG-NaN failures in CV raised exceptions rather than returning NaN, so a GPBoost failure would
crash before the target is read. That is safe and re-runnable. Other families give no such guarantee.

**Mitigation without a code change:** the lead checks the pre-flight `dry_run.json`
(`predictions_finite` for all 16 single models). **Better (small change, re-hash):** before reading `y_all`,
exit non-zero if any `preds[...]` is non-finite.

**W6. The package hash is already stale** (`0be57783…` now against `ca39f9ea…` at freeze, 14:20 today),
because of edits in `analysis/causal`, `models/{downstream,lite,orbit,sonde}`. No ChemTox dependency
changed. Z5 is still open. **Fix:** at scoring time, record a ChemTox-scoped hash in the handoff. For
reference, the reviewer's concatenation
`cat sentinel/models/chemtox/*.py sentinel/eval/bootstrap.py sentinel/core/registry.py sentinel/data/sources/adore.py scripts/eval/chemtox_score_prospective.py | sha256sum`
gives `8c5cd6b523a95fed` now. Also confirm the scorer is still `a0cb0385226e9aaf` immediately before the
pass, because the scorer does not check its own hash.

### Observations

- **O1.** Disclose the `chemeleon_tabpfn` config-0 OOM and re-screen in `RERUN_NOTE.txt` and the handoff,
  not only in the round-4 note. `finalize_now.sh` uses `tee` without `-a`, so it overwrote the
  `retry rc=` and "top-5 not yet run" lines that `retry_then_finalize.sh` had written to `finalize.log`.
  `cycle_tabpfn_retry0.out` survives. Note this in the audit trail.
- **O2.** Plan text still lags in places (round-3 Z6):
  - §4.2 says `published_reference` is "the better of the **two** published variants". The rule in §3 2b,
    §4.3 and the code is the best of **four**.
  - §3 item 7 still describes `chemprop` as "the configuration frozen for the ADORE pass". It is the masked
    `mw_logp` variant (stage 2 is correct).
  - The scorer's warning "`xgboost_published` is not in this pass" is stale wording. It is harmless.
- **O3.** `chemprop` shows `n_configurations: 32`. Its actual selection is the phase-1 search (32
  configurations, unmasked CV) plus a masked choice between `v1frozen` (0.9564) and `v1_chemprop_mwlogp`
  (0.9527), i.e. one configuration beyond 32. That is the §4.3 rule applied as written. Report it as
  "32 + 1 chemical-block variant".
- **O4.** The masked `embeddings` (1.1579) and `tabpfn` (1.2397) v1 re-runs exist but are correctly
  absent from the freeze, because §3 names only `chemprop` among phase-1 models. The handoff table labels
  them "candidate"; relabel them "not frozen (not in §3)".
- **O5.** `algae_novel` (3 rows / 2 chemicals) gives degenerate bootstrap p-values. In the random-target
  probe p = 0.0, so a "Holm rejection" is possible on 2 clusters. It is labelled secondary. Also mark it
  "descriptive only, no inference" in the report.
- **O6.** CPU policy: CONTRIBUTING_V2 caps CPU-heavy jobs at 6 threads and GPU jobs at
  `torch.set_num_threads(2)`. The scorer defaults to `--threads 8` (and sets torch threads to it). Run with
  `--threads 6`, or `--threads 2` with `--accelerator gpu`.
- **O7.** GPBoost (`gpboost`, `gpboost_multitaxa`, `chemeleon_gpboost`) can raise a CG-NaN error on a new
  fit. That would crash the pass during fitting, before any target is read and before any registry write,
  so a re-run is allowed. Keep `score.log` either way.
- **O8.** `prospective_keys()` reads the whole parquet (including targets) into memory to build identifier
  sets. It uses only identifiers. Restricting `columns=` would make "no code outside the scorer loads the
  targets" literally true.
- **O9.** `__allow_unmasked__` is recorded (False) but the scorer does not refuse a block with True
  (round-3 Z4 asked for that). It is moot for this freeze.

### Checklist (round 4)

| Item | Result |
|---|---|
| 0 prospective registry rows; scorer is the only reader; guards on every training frame | ✅ |
| Zero CAS / InChIKey / `result_id` overlap with ADORE curated rows (novel tracks) | ✅ (temporal shares CAS by design) |
| Track sizes = plan §1.3/§1.3a | ✅ |
| Stage-2 text = `selection.json`; frozen hash and file sha256 recomputed | ✅ |
| Each variant its own 32-config entry; no entry > 32; comparators re-tuned with the same budget | ✅ (chemprop 32+1, O3) |
| Ensemble-membership rule as pre-registered | ✅ 5 members, bar 0.9733 |
| N5 renamed member + NNLS OOF lookup (real block) | ✅ weights refit; CV recomputed exactly |
| OOF files masked, 5-fold, NaN on ADORE test rows, params = frozen | ✅ |
| Masking identical CV ↔ test | ✅ WS/MP 100% NaN in the prospective file |
| Holm family 11; references fixed; direction rule; chemical clusters | ✅ (W2 sensitivity) |
| Single-pass refusal before fits; hash seals; registry-only result writes | ✅ (W3 report overwrite, W5 finiteness) |
| Deviations disclosed | 🟡 W1 (new), W4 (statement), O1 |
| Tests | ✅ 129 passed |

**Gate decision.** The scientific and integrity conditions for the single pass are met. The lead may
authorise M4 to run:
1. W1–W3 (documentation and runbook, recorded in the plan as a dated addendum **before** the pass);
2. the dry-run pre-flight (rc must be 0, and `predictions_finite` must be true for all 16 single models;
   see W5);
3. the single pass with `--expect-config-hash 630b43e905d9b852` and `--expect-data-hash
   3b2fc439…dc54d9`, then the `_trainpart` pass only after the W3 copy.

W4–W6 and O1–O9 can be fixed with the post-results write-up. The post-results review gate still applies
before any number leaves the repository.

---

## Downstream — pre-test gate (2026-11-08)

**Scope:** whether each of the five freeze candidates (D1 SafeSwim Chicago D1b, D1n national beaches,
D5 BioIntegrity, D6 MusselGuard, D2 ToxinWatch it3) may be scored **once** on its test block.
**Mode:** pre-test gate. No test look was run, no FREEZE file was written, no downstream file was changed.
**Method:** the lead reviewer checked the shared guard, records, hashes, test scripts, statistics, pytest
and dry runs. Three independent read-only reviewers (fresh context) covered D1+D1n, D5+D6 and D2. Their
probes read test-role **keys and row counts only, never test labels**, and are in
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/gate_ds/{d1,d5d6,d2}/`.
**Read:** RESEARCH_PLAN §3, §8, §14, §15 (incl. §15.5 amendments 2026-10-24 / 2026-11-08);
docs/CONTRIBUTING_V2.md; docs/handoff/downstream.md; the 2026-10-24 Downstream section above; protocols
`downstream_{safeswim,d1n_beaches,biointegrity,musselguard,toxinwatch,d2n_microcystin}.md` + splits; the five
`FREEZE_CANDIDATE_downstream_*.json`; `sentinel/models/downstream/*` (common, testlook, wqp_track, weather,
crossfit, app modules); `scripts/train/downstream_*` (training, iteration and `_test.py`);
`tests/test_downstream*.py`; registry rows (workstream `downstream`); `configs_tried.jsonl`.

### Summary

| App | Verdict | Blocking items (fix → re-hash record → re-run dry run → short delta review) |
|---|---|---|
| D1 SafeSwim (Chicago D1b qPCR) | 🟡 may score after fixes | C-D1-1 2026 reserve written to the test-look predictions; C-D1-2 Holm family smaller than pre-registered |
| D1n national beaches | 🟡 may score after fixes | C-D1n-1 train+val refit not pre-registered; C-D1n-2 incumbent ridge α at grid edge (§15.5 item 4); Holm m = 4 vs protocol m = 2 to be recorded |
| D5 BioIntegrity | 🟡 may score after fixes | C-D5-T1 claim family drops the two feature-matched RFs; C-D5-T2 pre-registered secondary sets not scored/saved; S-C1 |
| D6 MusselGuard | 🟡 may score after fixes | C-D6-T1 test origins pooled (protocol: CI > 0 at **both** origins); C-D6-T2 incumbent choice ≠ written "stronger of it2/it2b" rule; C-D6-T3 SC-rule baseline silently absent; S-C1 |
| D2 ToxinWatch (it3) | 🟡 may score after fixes | C-D2-T1 wrong "strongest simple baseline" frozen for known lakes (location_knn, not persistence); S-C1 (thin margin makes the seed rule decisive) |

**No app may be scored with its current candidate.** None of the problems is leakage or an earlier test peek.
They are claim-rule and implementation deviations from the pre-registered protocols. Every fix is cheap and
can be made before any test row is read. Because each fix edits a hashed file, the candidate records must
be regenerated afterwards (nothing is frozen yet, so this costs nothing).

**Integrity status:**
- Test data untouched for all five apps.
- No data leakage found.
- National tracks exclude the original test blocks: the Chicago box removes 40,871 lakefront rows, and
  Washington is excluded from D2n (0 WA rows; nearest national site to any WA site is 18.5 km away).
- D5 uses no M5 outputs.
- D2 pooled incumbents get the same rows as the GBM.

### Verified by the lead reviewer (evidence)
- **pytest** (`tests/test_downstream*.py`): **102 passed**. Afterwards: registry rows unchanged for
  `downstream`, no `TEST_LOOK.json`, all hashes unchanged.
- **Registry:** 2,340 `downstream` rows, **0** with a split containing "test" and **0** experiments containing
  "test". No `FREEZE_downstream_*.json` exists. No `$SENTINEL_RUNS/downstream/*/TEST_LOOK.json` exists.
- **Records:** for all five, the following equal the current files:
  - `scoring_code_sha256` and every per-file sha256
  - the file set (`common.scoring_files(app)`)
  - the split sha256
  - the dry-run `results.json` sha256 and the artifact sha256s
  
  `configs_tried` equals `n_configs_tried(app)`: D1 50, D1n 12, D5 92, D6 131, D2 72.
- **Dry runs reproduced independently** (`--dry-run-on-val` with `SENTINEL_RUNS` redirected to the scratchpad,
  so no recorded artifact was overwritten; one job at a time, ≤ 6 threads, no GPU). Each `results.json` is
  **identical value for value** (0 differences) to the recorded one.
  - D1: logistic 0.7428, +0.129 vs official, +0.043 vs GM.
  - D1n: claim_met true.
  - D5: claim_met true under the *current* (too small) family.
  - D2: claim_met true, all-four Holm false.
  - D6: claim_met true.
  
  Registry `downstream` rows were unchanged by the dry runs. Rows added meanwhile belong to M1/M3.
- **Single-pass guard** (`testlook.py`):
  - `authorise` runs `check_roles` (FREEZE + split sha + scoring-code sha) then the O_EXCL claim.
  - Every library builder calls `require_authorised`: `safeswim.build`, `wqp_track.build_frame`,
    `toxinwatch.build_features`, `biointegrity.build_features`, `musselguard.build_origin`.
  - Registry writes go through `log_test_metric` only. `common.log_metric` still refuses test splits.
  - All five test scripts call `TL.authorise` before any test row is built, and `mode_from_args` enforces
    exactly one mode.
- **Holm/p-value mechanics:** `paired_bootstrap` p = 2·min(P(Δ ≤ 0), P(Δ ≥ 0)), so a Holm rejection plus
  Δ > 0 is at least as strict as "CI excludes 0". A NaN p never rejects.

### Shared findings (all apps)
- **S-C1 (critical for D2, D5, D6; cheap).** The "holds in ≥ 3 seeds" condition (RESEARCH_PLAN §15.1(iii), R7;
  each protocol's claim rule) is not part of any `claim_met`. Only the seed-averaged prediction is tested,
  and per-seed values are only reported (e.g. `downstream_toxinwatch_test.py:129`). Per-seed predictions
  are not saved (`:141-142`), so the condition cannot be evaluated after the look.
  **Fix:** before freezing, write the exact reading into each FREEZE record. Two options:
  (a) the seed-averaged model decides, per-seed paired results reported; or
  (b) every seed's paired Δ > 0 and Holm-rejected.
  Compute per-seed paired differences against the governing comparators, and save per-seed predictions.
  D1 and D1n claim deterministic logistics (trivially met; state it).
- **S-W1. Input data are neither hashed nor verified at look time.** `check_roles` verifies the split and code
  only. Files that change predictions and test-set membership include:
  - `processed/safeswim_drivers/*.parquet` (and `safeswim_drivers` missing from `safeswim.DATA_MANIFESTS`)
  - `forcings_daily`, `raw/gridmet/site_cells.parquet`
  - `wbd_huc12/huc12.parquet` (D1n/D2n HUC assignment of clusters)
  - WQP `ecoli`/`microcystins` parquet (the shas are in the split meta but not re-checked)
  - the NAS Dreissena files and the universe
  - `census_county` (feeds the D6 incumbent; missing from `musselguard.DATA_MANIFESTS`)
  - NARS/StreamCat
  
  All of these match their manifests today (reviewers' checks).
  **Fix:** pin per-file sha256 (and the expected national row count for D2) in each FREEZE record, and assert
  them in the test script **before** `authorise` so a mismatch cannot burn the look.
- **S-W2. Mislabelled registry metric.** `testlook.py:152` names every paired difference `auroc_diff_paired_<unit>`
  even when `metric=fast_auprc` (D6), so all D6 test differences would be registered as AUROC.
  **Fix:** derive the label from the metric. This is a shared hashed file, so all five records must be
  re-hashed.
- **S-W3. The test-mode write path has never been executed.** `score_set(mode="test")` → `log_test_metric` →
  `require_authorised` per metric is never run by a dry run, and a crash after the claim burns the look (by
  design). **Fix:** add a synthetic end-to-end test using the `isolated` fixture of
  `tests/test_downstream_testpaths.py` plus a monkeypatched registry writer. It should exercise
  `score_set(mode="test")` and `holm_primary` for each script's set layout. Alternatively, run each script's
  test path on synthetic frames.
- **S-W4. Pre-registration ordering cannot be proven from files.** Protocols and scripts have no git history,
  so ordering rests on mtimes. Affected:
  - D1 cycle 3 (protocol modified after the fits)
  - D5 it3 (ens script 12:40:27, registry 12:40:31, protocol 12:41:34)
  - D2 it3 and the D2 multiplicity text (test script modified 16:29, dry run 16:35, protocol 16:37: the
    rule was written after the validation dry run showed the all-four Holm failing)
  
  The test looks remain valid because selection used validation only. The lead should attest the order and
  the protocols should state it. Recommendation: commit (or archive with sha256) protocols at each
  pre-registration step.
- **S-O1.** The scoring hash does not cover `sentinel/core/{paths,registry,runlog}.py`, package `__init__`s or
  library versions. Record `pip freeze` sha in the FREEZE. Run each look with the default `SENTINEL_RUNS`
  (the variable relocates the TEST_LOOK marker, although the registry test-row check still applies).
  `wqp_track.py` is shared by D1n, D2n and D2, so an edit for one invalidates the others.
  `APP_SCORING_FILES["d2n_microcystin"]` lacks `toxinwatch.py`, which `wqp_track.py:565` imports for the
  D2n incumbent (not freezing now).

### D1 SafeSwim (Chicago, D1b qPCR *Enterococci* nowcast) — 🟡
**Criticals**
- **C-D1-1. 2026 reserve season loaded and written.**
  - `safeswim.py:41-42` gives 2026 the role `test`.
  - `build(include_test=True)` keeps it (`safeswim.py:388-392`).
  - `downstream_safeswim_test.py:55` takes every test row.
  - `:74-77` writes `y` plus every model score for all of them (1,286 2026 eval beach-days, by key count)
    to `predictions_test_2024_2025_official_rows.parquet`.
  
  The scored rows exclude 2026 (no official prediction), but the protocol (§5b) reserves 2026 for a
  post-look configuration, and writing its labels next to the frozen model's scores would spend it.
  **Fix:** filter `ev` to 2024–2025 before scoring and writing (better: build with test years 2024–2025 only),
  and state this in the record.
- **C-D1-2. Holm family.** Protocol §4/§4a pre-register "Holm across the D1a/D1b primary comparisons". That is
  four comparisons: D1b nowcast vs official and vs GM, D1b forecast vs persistence, D1a vs persistence.
  The script uses only the two nowcast comparisons (`downstream_safeswim_test.py:12, 35, 70-73`), which is
  less strict. **Fix (either):**
  (a) Holm with m = 4, counting the two unscored comparisons as p = 1. On validation this still rejects,
      with p 0 / 0.006 under week and beach.
  (b) Add a dated protocol deviation, before the freeze, that withdraws D1a and D1b-forecast from the claim
      family permanently.
**Warnings**
- **W1. Silent feature loss.** Driver columns are added only if the discharge parquet exists
  (`safeswim.py:414-415`) and filtered by missingness (`:469-474`), so `compact_each+q` could silently become
  `compact_each`. **Fix:** assert the frozen 26 feature names in the test script.
- **W2. Data hashing.** See S-W1 (`safeswim_drivers` is not in `DATA_MANIFESTS`, `safeswim.py:33`).
  Discharge for 2024–2025 is now USGS-approved, whereas only provisional values exist on a real decision
  day. This held on validation too; disclose it.
- **W3. Screening count.** Protocol §10 says "21 feature sets". The logs show 16 D1b-nowcast feature sets
  (15 distinct; `+waves` = `compact_each`), and 50 log entries / 40 unique hashes including one crashed run.
  Correct the count.
- **W4. Power.** The test block has about 2,300 official-row beach-days at 12 beaches over 32 ISO weeks (key
  counts). A percentile bootstrap over 12 beach clusters may be anti-conservative. The validation margin
  over GM (+0.043 [0.014, 0.080]) is thin after 4 cycles of screening.
- **W5. 2026 cannot re-test the main claim.** 2026 has no official City predictions, so the reserve can
  re-test only vs GM. State this.
**Observations**
- The claim must name the task: D1b qPCR *Enterococci* > 1,000 CCE.
- The research reference values (0.702 / 0.740) span the test years; this is already disclosed and was
  not used for selection.
- The GM baseline is the strongest simple rule on validation (0.700 vs ≤ 0.676) and on training, so it was
  not weakened.
- The official model is used as published and joined on the same local date (reproduction 0.703 vs 0.702).
- Decision-time rules verified:
  - same-day culture refused (`safeswim.py:178-179`)
  - previous-value features strictly earlier, with assertions (`:166, :201-210`)
  - same-day qPCR from input beaches only
  - drivers and gridMET end d−1, with climatology from training years
  - imputer and scaler fit on training rows only (`:543-556`)
  - test refit on training years only, as the protocol states

### D1n national beaches — 🟡
**Criticals**
- **C-D1n-1. Test-time fit set not pre-registered.** The script refits on train + val and defines "known"
  as clusters in that fit set (`downstream_d1n_beaches_test.py:13, 53, 55-57`). Protocol §4 says only
  "refit on all training rows", and this path has never run (the dry run fits on train only).
  **Fix:** before the freeze, add a dated protocol deviation that registers the train+val refit (GBM fixed
  at 503 trees, every comparator refit on the same rows, as coded), or revert to train-only.
- **C-D1n-2. Incumbent under-tuning risk (§15.5 item 4).** VB-MLR ridge α selected 100, the largest value
  in its grid, with inner AUROC still rising (0.7700 / 0.7704 / 0.7724 / 0.7734; `phase1_results.json`).
  This is the same situation D6 resolved with it2b. **Fix:** widen the incumbent grid (e.g. α to 10⁴)
  using the **inner training-year split only**, freeze the selected α, and report the validation numbers
  again.
- **Holm.** Protocol §5 says Holm per challenger × set (m = 2). The script uses m = 4 across both sets
  (`:92-93`). This is stricter and acceptable, but must be written into the protocol and record before the
  look.
**Warnings**
- **W1. Unverified membership data.** Test-set membership is recomputed from the unhashed WBD file
  (`wqp_track.py:148-155`); see S-W1.
- **W2. Stratum threshold.** The rain stratum uses a P90 from train+val (`:83-86`) rather than the
  protocol's "training P90". It is descriptive only; disclose.
- **W3. "New beach" = new HUC4, not a new water body.** Nearest fit cluster to `test_new`: median 97 km,
  5th percentile 15.7 km, minimum 2.6 km. 63 of 160 test-new clusters lie on Great Lakes HUC4s that share a
  lake with training HUC4s. Disclose.
- **W4.** Choosing logistic over GBM was done on validation (disclosed; two candidates).
**Verified**
- Chicago box excludes 40,871 lakefront rows before processing (`wqp_track.py:83-84`). The 8 out-of-box
  stations with Chicago-like names are elsewhere (CA/MI/MN/ND/OH/WI).
- 2020 and 2026 have no role.
- Phase 1 drops test rows before features. Artifacts hold 2016–2019 dates only.
- History and network inputs ≤ d−1; weather ends d−1 with a 2004–2015 climatology.
- Comparators are scored on identical rows. HUC8 bootstrap as registered.
- Test size about 33k known / 10k new cluster-days (keys).

### D5 BioIntegrity — 🟡
**Criticals**
- **C-D5-T1. Claim family.** The test script's primary family is only the two Hill rows: `PRIMARY` =
  {`hill2017_rf_streamcat`, `tuned_hill2017_rf_streamcat`} (`downstream_biointegrity_test.py:43`).
  The feature-matched RFs are secondary (`:44-45`), and `claim_met`/Holm cover two comparisons (`:92-93`).
  But the protocol pre-registers more:
  - §8 W8 (2026-10-24): each challenger must beat both the incumbent and the feature-matched RF, with Holm
    over four.
  - The it3 rule (§9): keep only if the ensemble beats **all four RF rows**, and its validation result was
    "Holm over the four RF comparisons rejects all".
  
  The record's `claim_rule` repeats the weakening.
  **Fix:** `PRIMARY = {"ens_tuned_gbm_rf": [hill2017_rf_streamcat, tuned_hill2017_rf_streamcat, rf_geo_streamcat,
  tuned_rf_geo_streamcat]}`, Holm over the four on HUC4; correct the record. On validation all four reject
  (it3), so the fix carries no incentive problem.
- **C-D5-T2. Pre-registered secondary results would be unrecoverable.** The script scores only test new
  locations (`:68`) and saves predictions only for them. Missing are:
  - accuracy at p = 0.5 (Hill's metric)
  - all 2023-24 first visits
  - revisits with the persistence baseline
  - the frozen `val_test_new_location` subset (§6 says it "is reported as a secondary subset")
  - logistic-on-geo (listed among comparators)
  
  **Fix:** add these as secondary sets outside the Holm family, and save predictions for every 2023-24 unit.
- S-C1 applies (3 seeds per member; the seed rule is not part of the claim).
**Warnings**
- **W1. Selection timing.** it3 was registered after the it2 validation results existed. Its "before
  computing" claim rests on a 4-second mtime gap (S-W4). Validation CIs are selection-biased (4 iterations
  on one cycle); disclose with the 92 configurations.
- **W2. Tuned incumbent at grid edge.** It selected `max_features 0.5`, the largest value in its grid, with
  inner AUROC rising (0.662 → 0.672). The published-default incumbent (stronger on validation) is also in
  the family, so the risk is limited. Widen the grid on inner LOEO, or disclose.
- **W3. Tuning budget.** Parity is per learner family (8 configs each). The challenger combines two tuned
  families plus an ensemble rule; disclose.
- **W4.** `build_features` runs after the claim. A crash on 2023-24 data would burn the look (S-W3).
**Verified**
- Builders refuse test rows (`biointegrity.py:174-177`). The script authorises (`:55`) before `load_units`
  (`:56`).
- Phase-1, iter and ens scripts drop test rows before features. `load_units` reads test-cycle classes into
  memory and drops them; the split-report class counts are disclosed in §7.
- **No M5 output is used:** the only `bioassess` import is `all_site_locations` (coordinates; hashed), and
  `phase1_no_upstream` is applied. C-D5-1 exclusions hold (`ici`, `iwi`, `bankfull*`, `wettedwidth`,
  `thalwegdepth` filtered).
- RF imputation is fit on fit rows. The test refit (train + 2018-19) uses the it2 `chosen` parameters exactly.
- The new-location definition matches §6 (recomputed 1-km clusters = frozen `visit_location`, 0 mismatches
  over 7,804 visits).
- HUC4 primary / HUC8 secondary bootstrap. Test size 1,280 new-location units / 190 HUC4 (keys).

### D6 MusselGuard — 🟡
**Criticals**
- **C-D6-T1. Origins pooled.** Protocol §6 requires "CIs excluding 0 (Holm across comparisons) at both test
  origins" and §10 says "scored once on T = 2018 and T = 2021". The script builds one pooled set
  `test_origins_2018_2021_*` (`downstream_musselguard_test.py:93-108`). No amendment authorises pooling,
  and the record's `claim_rule` omits "both origins".
  **Fix:** score each origin separately (front / CONUS / MN), Holm across 6 baselines × 2 origins, and
  require both. Pooled results may be reported as secondary.
- **C-D6-T2. Incumbent selection rule.** §11b says "the stronger of the it2 / it2b incumbent on validation
  is the incumbent for the claim". Validation front AUPRC is it2 `kanankege_r320` 0.0215797 vs it2b
  `r80_k32_n0.5` 0.0215759 (registry). The script uses it2b (`:37`).
  **Fix:** use the it2 configuration as written, or (conservative) put both in the Holm family.
  The difference is negligible; the point is to follow the written rule.
- **C-D6-T3. A pre-registered baseline is missing without a logged deviation.** The SC-rule baseline
  (Whittier-2008 analogue; §6 "if available") was conditional on the rewritten `wqp_discrete` manifest
  (§4). That condition was met on 2026-10-25 04:31Z, before it2 and it2b (the manifest has spcond/pH/huc8),
  yet every build ran with `wqp: false`. `BASELINES` (`:42`) omits it. Under §15.5 item 4 (baselines never
  removed), **either** run it on validation and add it to the family, **or** log a dated deviation with the
  reason, before the freeze.
- S-C1 applies (3 GBM seeds).
**Warnings**
- **W1.** S-W2 (AUPRC differences registered as `auroc_diff_paired_huc8`) affects D6 directly.
- **W2. "Interior optimum" is partial.** The range (80 km) is interior, but the nugget 0.5 is the top of a
  two-value grid and k = 64 was tried only with nugget 0.1. Validation AUPRC is flat across settings;
  disclose.
- **W3. Label completeness.** The NAS 2026-10-22 snapshot has no entry dates, so completeness of the
  2022-2024 labels for T = 2021 cannot be checked without peeking; only the 2-year lag rule protects it.
  Under-reporting lowers power, and with a both-origins rule raises the chance of "competitive". State
  this before the look.
- **W4. Provenance.** `census_county` is not in `DATA_MANIFESTS` (`musselguard.py:42`). Pin the NAS and
  universe sha (S-W1).
- **W5. Framing conflict.** §10 says D6 "counts toward H-G4 'implemented' but **not** toward 'beats its
  incumbent'". §11 (lead direction) makes the re-implementation "the incumbent", and the handoff counts D6
  toward the four needed. Resolve this in the protocol before the look.
**Verified**
- `build_origin` refuses test origins without an authorised claim (`musselguard.py:188-191`). The script
  authorises (`:70`) before `frame()` (`:76`).
- C-D6-1 fix intact: `sources_at` bins only records with year ≤ T. Post-T records are loaded but never
  enter features.
- Candidates and infested-by-T use `first_year_any ≤ T`; labels fall in (T, T+3].
- Training labels end 2018; test labels 2019-2024.
- Frozen parameters match the phase-1 selections. Comparisons are on identical rows (`ok` mask).
- The primary endpoint is front-stratum AUPRC on a HUC8 bootstrap.
- Budgets: pressure GBM 6 configurations, incumbent 23. The aborted first it2b launch scored nothing.

### D2 ToxinWatch (it3) — 🟡
**Criticals**
- **C-D2-T1. Strongest simple baseline (known lakes).** §5 defines it as "the best of persistence /
  location-only / climatology / lake history, chosen on validation". On `val_known_2019_2021` (it3
  registry rows) the scores are:
  - location_knn **0.9104**
  - persistence 0.9050
  - lake history 0.8829
  - climatology 0.5765
  
  The driver's own selection records `best_simple_baseline = "location_knn"`
  (`downstream_toxinwatch.py:159`). Yet the test script hard-codes `"known": "persistence"`
  (`downstream_toxinwatch_test.py:51`), and protocol §8 and the record do the same.
  **Fix:** governing known-lake comparator = `location_knn` (or require beating both). Log it in §7 and
  re-hash. On validation GBM − location_knn = +0.029 [0.006, 0.134], p 0.004, so the validation verdict is
  unchanged and there is no incentive problem. New lakes: lake history 0.7559 > persistence 0.7374
  (correct as frozen).
- **S-C1** applies with force: the known-lake validation lower bound is 0.001, so how the seed condition is
  read could decide the label. Save the per-seed GBM predictions.
**Warnings**
- **W1. Multiplicity.** The code (`:131-140`) correctly makes Holm over the 2 sets per comparator family
  governing (= original §5 text, 2026-10-24) and reports the all-four Holm as a sensitivity, consistent with
  the lead decision. But the decision text was written after the validation dry run (S-W4), and the script
  docstring still says "Holm across the four primary comparisons" (`:9-14`). Fix the docstring and state
  the order of events. The "not robust to all-comparison correction" label must be applied if the case
  recurs at test.
- **W2. Data hashing (S-W1).** The 12,357 national training rows are rebuilt at run time from WQP files.
  Pin the file shas and the expected row count.
- **W3. Configuration count.** The D2 lineage also saw D2n phase 1 (16 configs) before it3 was designed;
  disclose 72 + 16 = 88.
- **W4. Pooled-incumbent parity.** Learners get `src_national`; the pooled StGlmm design has no source
  offset (`toxinwatch.py:461-493`), and the pooled v2 spline uses 6 knots over CONUS. Both were
  pre-registered and the WA-only variants remain candidates; disclose.
- **W5. Scum stratification missing.** The pre-registered scum-flag stratification (§5) is absent from the
  test path; add the scum flag and site_id to `predictions.parquet`.
- **W6. Test-block description.** The record should state:
  - "known" = train-role clusters with ≥ 1 training row
  - 2026 runs through 2026-10-14 (vintage 2026-10-22)
  - key counts: known 3,190 rows / 78 clusters, new 568 / 40
  - Lake Washington is about 60% of known rows and `wa_lc0026` about 39% of new rows, so these clusters
    dominate the lake-bootstrap CIs
**Observations**
- The dry run reproduces it3 exactly for GBM, logistic and baselines. v2 incumbents on new lakes differ in
  the 4th decimal (0.74679 vs 0.74703) because `knn_lake_rate` averages eval-row coordinates per cluster
  (`toxinwatch.py:371`). This is label-free, but "reproduces exactly" in the handoff is slightly overstated.
- `common.py` and `wqp_track.py` were edited after the it3 run; the identical dry run shows predictions
  were unaffected.
- National rows include NLA/NARS records from other states; keep any NLA secondary check disjoint from
  training. The §5 sentence "WQP microcystin is not used in phase 1" is out of date.
**Verified**
- **No earlier contact with WA test data:** lake roles from counts only; test rows dropped before features;
  builders guarded; the script claims (`:81`) before reading WA rows (`:83`).
- **National rows:** D2n train role only (`it3.py:49-53`, `assert_no_test`); WA excluded (0 WA or
  null-state rows); nearest national site to any WA site 18.5 km (none within the 2-km cluster radius).
- **Same rows for every model:** all four incumbents are refit at test on the same pooled rows with C = 1.0
  (matches selections), and logistic/GBM parameters match.
- **Train-only fitting:** d−7 embargo and d−1 weather verified; climatologies, scalers, imputers, p0 and the
  kNN term are fit on train; the inner early-stopping split is train-only.
- **Split and selection:** split v2 was frozen before the first fit; `site_cluster` equals a recomputation
  (525 sites); the strongest incumbent (v2_pooled) was chosen on validation on both sets.

### Checklist summary (this gate)
| Category | Checked | Passed | Failed | N/A |
|---|---|---|---|---|
| Test data untouched (registry, builders, earlier paths, markers) | 15 | 15 | 0 | 0 |
| Leakage (blocking, decision time, train-only fits, national exclusions, cross-fit, pooled rows) | 20 | 20 | 0 | 0 |
| Incumbent fidelity / tuning parity | 7 | 4 | 3 (D1n α edge, D6 it2-vs-it2b rule, D6 SC-rule) | 0 |
| Records (hashes, selection vs protocol, configs counts, file coverage) | 15 | 11 | 4 (data unhashed, D2 simple baseline, D5 claim_rule, D6 claim_rule) | 0 |
| Test scripts (guards, modes, claim-before-read, writer, dry runs) | 15 | 12 | 3 (D1 2026 reserve, D1n refit, AUPRC label) | 0 |
| Statistics (Holm families, units, claim rules, seeds) | 15 | 9 | 6 (D1 m, D1n m (stricter), D5 family, D6 pooling, seed rule ×3 apps counted once, D5 secondaries) | 0 |
| Tests | 1 | 1 (102 passed) | 0 | 0 |

### Gate decision
**No test look is authorised yet.** For each app:
1. Apply its criticals and S-C1 (seed-based apps), plus S-W1 (data pins), S-W2 (AUPRC label) and S-W3
   (test-mode write-path test). Where the fix is a decision rather than code, record it as a dated protocol
   deviation.
2. Regenerate the FREEZE_CANDIDATE record (code sha, file sha256s, data shas, exact claim rule incl. the
   seed reading).
3. Re-run `--dry-run-on-val`. Expected validation verdicts: D1 unchanged under m = 4; D5 unchanged with the
   4-RF family; D2 unchanged vs location_knn; D6 to be recomputed per origin, which on validation has only
   one origin, so the dry-run check is structural.
4. A short delta review of the diff confirms the above.

Then the lead writes the FREEZE file and scores the test block once. Order of readiness (least work first):
D2 and D5 (comparator/family edits), then D1 (filter + Holm m), D1n (α grid on the inner split + deviation
note), D6 (per-origin scoring + incumbent rule + SC-rule decision). The post-results review (R10) still
applies before any test number leaves the repository.

---

## EW-v2 — pre-test / pre-registration gate (2026-11-08)

**Status: 🔴 FAIL — do NOT write `FREEZE_ew.json`, register on OSF, or score Test-A yet.** There is no
contamination: no test-role site and no Test-A/Test-B date was ever loaded by model-scoring code. The calibration
is sound: train-only fits, val-only Platt, stack and thresholds, alarm-day budget, identical rows, and an
identically recalibrated climatology reference. The selection rule was applied as written, and the dry run
reproduces the calibration exactly. What is not ready is the *registered decision procedure*. The primary analysis
population is undefined: the code pools all site roles, while the power statement uses test-role sites. The Holm
step and the CI named in the success rule are neither consistent nor implemented. Run-time parameters and re-runs
of the test scoring are not locked. Most §8 controls (and parts of §5/§7) are promised but not implemented, and no
deviation records that. §11 cannot be executed as written. All of these can be fixed in days without touching data
or results. Fixing them changes `scoring_code_sha256`, so the candidate must be regenerated afterwards
(`scripts/eval/sonde_ew_freeze_candidate.py --run … --configs-tried 8`).

**Reviewed:** `docs/preregistration/ew_v2_protocol.md` (incl. the 18:00 edit adding the pooled Test-A ∪ Test-B
secondary), `ew_v2_freeze_candidate.json`, `scripts/eval/sonde_ew_freeze_candidate.py`, `scripts/eval/sonde_onset_counts.py`,
RESEARCH_PLAN §12/§14/§15, CONTRIBUTING_V2, `docs/handoff/M1-sonde.md`, the M1 section above (C1–C3, W1–W12),
`sentinel/eval/{hazards,early_warning,ew_scoring,chloride,bootstrap}.py`, `sentinel/models/sonde/{events,guard,subdaily,common,data,runner}.py`,
`scripts/train/sonde_events.py`, `scripts/eval/sonde_predict.py`, `scripts/data/build_{chloride_derived,sonde_split}.py`,
`configs/sonde/{events,common}.yaml`, `docs/datasheets/chloride_derived.md`, the split file, the registry
(`ew_v2_val_oof`, `ew_v2_onset_counts`), `sentinel_runs/sonde/configs_tried.jsonl`, and every sonde run's `data_info.json`.

**Tests:** `pytest tests/test_hazards.py tests/test_chloride.py tests/test_sonde_models.py`: **57 passed**, 0 failed.

**Independent checks (all reproduced here):**
- `sha256(sonde_split.json)` = `fd4bb14f…` ✓.
- `guard.scoring_code_sha256(files=scoring_code_files)` = `72e93858…` ✓ (current tree).
- `sha256(event_calibration.json)` = `3c243c1b…` ✓.
- All 90 artifacts re-hashed; each matches both the calibration file and `weights_sha256` ✓.
- Calibration split hash `fd4bb14fe730`, `smoke: false`, budgets 1/30 and 2/30 ✓.
- Dry run (`ew_v2_cand_dryrun_val_insample`) vs calibration (H1/H2/H4, L = 3): same rows (166,303 / 8,013 / 15,290). Labels are identical, max |Δp| = 0.0 for the stack and for Platt(event_gbm), and the in-sample realised alarm-days are ≤ 1/30 for every method ✓.
- All **69** `sentinel_runs/sonde/**/data_info.json` files: `cube_end 2023-09-30`, `protected []`, roles {train, val} only ✓.
- Selection re-derived from `event_scores.csv`. H1: stack 0.0222 > event_gbm_mono 0.0133. H2: 0.2377 > 0.2221. H4: 0.0035 > −0.0054. Secondary: t4 → mono, t2 → stack, H2-acute → mono ✓.
- `configs_tried.jsonl` lists **8** non-smoke `events_calibrate` runs ✓. Registry `ew_v2_val_oof` holds 7 of them (W9).

### Critical issues (must fix before the freeze / OSF)

**C1. The primary decision procedure is not fully specified, and the code does not implement it.**
- *(a) Population.* `scripts/train/sonde_events.py:405,429-454`: the score phase builds `roles_all` (train + val + test
  roles with `--include-test-sites`). `evaluate_scores` (`:100-137`) pools all rows, with no role filter or role rows.
  Test-A would therefore be dominated by **train-role** sites (temporal-only generalisation). In contrast, §9's
  power projection (≈ 44 onsets), the new pooled secondary ("same … test-role sites") and the M1 primary
  ("unseen × Test-A") all refer to test-role sites. The choice moves H1 from "underpowered" (≈ 44) to roughly
  130+ onsets, so it has to be fixed before unblinding.
- *(b) CI unit.* §7 says success is the "lower 95% **site-month** block-bootstrap bound". The 2026-10-24 deviation, the
  §9 pooled text and the code's primary `bss_ci_low` (`ew_scoring.py:144-149`) use the **basin-group > site-month**
  two-stage bootstrap.
- *(c) Holm.* Holm is not implemented in EW. `bootstrap.holm_bonferroni` exists but is never called, and no p-values
  or adjusted bounds are produced. The family is also undefined: are H2/H4, expected underpowered, members (m = 3)?
  What about H5b, which is "conditional"?
- *(d)* The onset count behind the ≥ 50 rule is undefined (onsets at which sites? evaluable at L = 3?).
- **Fix:**
  - Protocol §7: primary population = **test-role sites × Test-A**. Report the all-roles, train-role and val-role
    × Test-A cells as labelled secondary cells.
  - Primary CI = basin-group > site-month two-stage.
  - Holm family fixed at {H1, H2, H4} (m = 3, whatever the onset counts), with one-sided bootstrap
    p = P*(BSS* ≤ 0) and Holm-adjusted rejection. "Underpowered" (< 50 evaluable onsets at test-role sites,
    L = 3) blocks the claim but not the computation.
  - Implement role-stratified rows plus Holm in `evaluate_scores`/`score()`, with a unit test.

**C2. Run-time parameters are not locked for protected scoring.**
`scripts/train/sonde_events.py:472-477` and `runner.apply_overrides` (`runner.py:58-71`) accept `--config <any path>`
(with `common.yaml` taken from that path's directory), `--set k=v`, `--seed` and `--quantile-model`. Any of them
changes test behaviour (bootstrap seed, `n_boot`, `k_placebo`, feature blocks, hazards) without touching a hashed
file. `score()` never compares `cal["config_hash"]` (`c68560d0…`) with the running config. Precedent: M5 C5.
- **Fix:** when `data.access["protected"]`:
  - refuse `--set`, `--seed`, `--smoke` and `--temp-split`;
  - require `--config` == repo `configs/sonde/events.yaml`;
  - require `config_hash(public_config(cfg) minus include_test_sites)` == the calibration `config_hash`, also recorded in `FREEZE_ew.json`;
  - write argv to `score_manifest.json`;
  - unit-test all of the above.

**C3. There is no single-pass guard on test scoring.** `score()` (`sonde_events.py:380-459`) can be re-run on `test`
any number of times. Other workstreams have one: `scripts/eval/orbit_test.py:133`, `bioassess_final.py:122`,
`sentinel/models/downstream/testlook.py:174`.
- **Fix:** before any protected load, atomically create a marker, e.g.
  `sentinel_runs/sonde/.ew_scored_<block>_<freeze_sha12>.json` (`O_EXCL`), and refuse if it exists. Give the
  pre-declared pooled Test-A ∪ Test-B pass its own named marker so that it is not a "re-score". Unit-test both.

**C4. §8 controls and parts of §5/§7 are promised ("all reported", "all mandatory" in RESEARCH_PLAN §12.6), but
neither implemented nor logged as deviations.**
- Implemented: the season-matched placebo (`ew_scoring.py:364-406`) and POD ≥ 24 h.
- Absent from the EW code: circular-shift surrogates, random alarms at matched rate, negative-control hazard,
  label permutation within site, station swap, covariate shuffle, input-truncation audit (only an M1 unit test
  exists), sensor-fault audit, v1 CUSUM re-scoring (§5, RESEARCH_PLAN §12.1), Molchan, reliability, relative
  economic value, and Diebold–Mariano for EW.
- H-b asks for ΔBS vs the **GBM** baseline, but `BASELINES = ("climatology", "persistence", "logistic")`
  (`sonde_events.py:63`). ΔBS vs "threshold" is impossible, because threshold is `NOT_PROB`.
- Code written after unblinding for these analyses would be unregistered analytic flexibility.
- **Fix:**
  - Implement in hashed code before the freeze at least the controls that consume test predictions: random
    alarms at matched rate, circular-shift surrogates, label permutation, and the negative-control hazard.
  - Add `event_gbm` to `BASELINES`.
  - Amend H-b: threshold is compared on POD at budget only.
  - Log a dated deviation for each remaining item, with its exact definition, or drop it (e.g. the sensor-fault
    audit needs provisional-vs-approved snapshots; station swap needs a defined pairing).

**C5. §11 (live trial) cannot be executed as written, so it must not be registered in this form.**
- No EFI forecast writer, daily pipeline or file hashing exists. `opentimestamps` is not installed.
  `sentinel/data/sources/efi.py` only reads targets.
- H3 is scheduled for winter but not implemented (`hazards.py:56`).
- H5b is scheduled for summer but has no calibrated model (0 val onsets, so no `event_calibration.json` entry).
- NEON/VERA/usgsrc4cast scoring is not implemented.
- Live inputs are all provisional: `subdaily.py:42-46` drops provisional hourly values, so the frozen DO models'
  `sd_*` features would always be NaN live, i.e. off-distribution.
- Unspecified: site list, issue time (RESEARCH_PLAN §12.3 says 06:00 local; day-d statistics need DV
  publication), EFI `variable`/`datetime` semantics for "onset within h", the public log location, and the
  expected live onset counts (H2 val: ≈ 6 chronic onsets per winter across all 52 eligible sites, so ≥ 30 per
  season is implausible).
- **Fix:** register §11 as a separate, dated amendment before 2026-12-01, after a working dry run. The amendment
  must cover: frozen model/feature variant for provisional inputs, sites, issue time, file schema, hash/timestamp
  log, analysis dates, and the expected count with a power statement. The Test-A freeze need not wait for this if
  §11 is split off.

### Warnings (should fix before the freeze)

- **W1. Protocol text ≠ code** (edit the protocol; no code change needed):
  - H4 pre-window: the protocol says "< 9 mg/L", the code uses ≤ 9 (`hazards.py:43,75`).
  - Not stated in the protocol: H1-tier pre-windows (≥ 5.0 / ≥ 3.0), the H2-acute onset rule (7 d ≤ 688 mg/L,
    `pre_gap` 0) and the refractory periods (H4 14 d, H5b 7 d).
  - H2 `pre_gap = 3`: state the clean window explicitly as [d−10, d−4].
  - The negative-label coverage rule (≥ 80% of (d, d+h] observed; positives exempt) and refractory-continuation
    invalidation (`hazards.py:91-121`) are not in §2/§4.
  - The §3 "15-min tier" column is not operational in Test-A scoring.
- **W2. §7 says the climatology gets "the same out-of-fold (by WY) Platt recalibration".** The frozen Platt, stack
  and thresholds are fitted on **all** val rows (`sonde_events.py:311-325`). OOF-by-WY is used only for the val
  descriptives. The implementation is correct; fix the wording.
- **W3. The freeze record has no data pins.** H2 test labels depend on `processed/chloride_derived/regressions.parquet`
  and `sites/`. Inputs depend on the `sonde_daily`, `sonde_hourly` and `forcings_daily` snapshots, and the
  handoff plans a WY2024–26 re-pull. Add the manifest SHA-256 and retrieval date for each to `FREEZE_ew.json`, and
  pre-specify which snapshot Test-A uses.
- **W4. Gaps in hash coverage.** `sentinel/models/sonde/evaluate.py` is used at score time:
  `events.onset_prob_from_quantiles` calls `clip_plausible` for `q_damped_anomaly`. `sentinel/eval/splits.py` is
  also imported on the scoring path. Neither is in `guard.EW_SCORING_CODE` (`guard.py:42-49`).
- **W5. The protected path has never been exercised.** The dry run was val in-sample with `protected=False`. Not yet
  run: `verify_thresholds`, `--include-test-sites`, and `fit_site_deviation` for unseen sites (val-role sites
  already carry deltas from calibration). Add a unit test with a scratch freeze record. Also add a val dry run in
  which val-role sites are treated as unseen (their deltas dropped).
- **W6. The pooled Test-A ∪ Test-B secondary is not implemented.** `score()` evaluates blocks separately
  (`:429-456`). Also missing: a pre-specified retrieval cut-off for WY2026 approvals, a rule for sites or days
  still provisional at that date (not missing at random), and the 42-d embargo at the start of Test-B. Implement
  pooled scoring in hashed code and state the cut-off date.
- **W7. H5b in §7 primary ("if ≥ 50 test onsets") cannot be scored**, because no H5b artifacts exist. §9 says
  lake/challenge only. Remove it from §7.
- **W8. The H-c test is anti-conservative.** The Fisher exact test (`ew_scoring.py:400`) treats the 20 placebos per
  onset as independent, but they share sites and alarm regimes. RESEARCH_PLAN §12.6 says "permutation p". Use a
  site-cluster permutation, or a basin-bootstrap of POD_real − POD_placebo.
- **W9. Configurations-tried count.** There are 8 calibrate runs; `full_events_calibrate` (raw-climatology
  reference) is not in the registry. The 5 offline stack variants (scratchpad `m1/stack_explore.py`: feat,
  feat+qda, feat+allq, gbm+clim, feat+qft) are unregistered and only mentioned in a note. Register or list them,
  and put "8 calibrate configurations + 5 stack variants, val only" in protocol §7/§12 next to the selection rule.
- **W10. Prior M1 W12 is still open.** Split eligibility counts ≥ 730 days over the full record, including
  WY2024–26 (`build_sonde_split.py:35,69`). The frozen split is therefore conditioned on test-period
  availability, though not on values. It cannot be changed now; disclose it in §12.
- **W11. Test-role label statistics were computed outside the guard.** `build_chloride_derived.py` reads all sites
  and all years. The chloride datasheet reports H2 onset counts at **test-role** sites for WY2008–21 and
  WY2022–23 (`docs/datasheets/chloride_derived.md:24-29`), and §9's "12 across all roles" uses them. No model
  output and no Test-A/B values were involved. Disclose it in §12 (prior-knowledge statement for OSF).
- **W12. Power in basin groups.** The projected ≈ 44 H1 onsets at test-role sites will fall in few basin groups
  (val: 17 groups across two roles), so the two-stage CI will be wide. State the expected number of test-role
  basin groups with DO sites and with onsets.

### Observations

- O1. The selected "SENTINEL" EW model is a logistic stack of baselines plus a monotone GBM. It has no M1
  forecasting or foundation component (q_* methods are excluded), so claims must describe it as such.
- O2. The val evidence is weak. H1 stack OOF BSS is +0.022, basin CI [−0.015, 0.043]; only persistence has a CI
  > 0. H2 has 18 positive rows in 3 groups, H4 12 in 2. With H1 ≈ 44 projected and H2/H4 underpowered, the
  RESEARCH_PLAN §15.2 EW criterion (≥ 2 hazards pass on Test-A) is expected to fail. Say so in the registration.
- O3. Out-of-fold realised alarm-days are 1.03–1.05 per 30 d for the stack, slightly over budget out of sample.
  Expect the same on test. Report it; never re-tune.
- O4. Ties in discrete scorers (persistence/threshold) push realised alarm rates far below budget (H1-t4
  persistence: 0 alarm-days, POD 0). Budget-matched POD comparisons disadvantage them. Consider randomised
  tie-breaking, or note it.
- O5. `common.attach_chloride` takes the `cl_mgl_max` provisional flag from **daily-mean** SC, not daily-max SC
  (`common.py:422-426`).
- O6. The guard docstring (`guard.py:17-19`) says protected values are "never in memory". In fact site files are
  read whole and then cut (`data.py:83-88`, `subdaily.py:73`, `attach_chloride`). Correct the wording as was done
  for Sim W9.
- O7. `brier_skill` silently drops non-finite p per method (`ew_scoring.py:127-128`). Assert finite p on identical
  rows for every registered method.
- O8. The §10 row "Scoring code (`sentinel/eval/*.py`)" is really 19 files. The git commit is pending because of the
  no-commit rule. Archive the hashed files, the calibration file and the 90 artifacts (they live outside the
  repo) as a tarball with SHA-256 for OSF.
- O9. H1 climatology Platt slope ≈ 0.21, so the recalibrated reference is close to a constant base rate. Keep this
  in mind when interpreting BSS.
- O10. There are no unit tests for the H4/H5b onset rules or for the protected score path (see W5).

### Verified correct (no action)
- **Integrity.** No test-role site or WY2024+ date was loaded by any model-scoring run (69/69 run records; registry
  rows are val/train only). Test-role sites are protected in every block (`guard.py:50-51`), and training scripts
  refuse them (`runner.py:220`).
- **Leakage.** Event GBM is fitted on d + L ≤ 2019-09-30 and early-stopped on WY2020–21 train-role rows.
  Logistic, persistence and pooled climatology use train-role × train block. Per-site climatology deviations use
  each site's own training-year rows. Platt, stack and thresholds use val only. Chloride regressions use
  training-block pairs only (≤ 2021-09-30).
- **Hazard rules.** H1 (< 3, 3 d ≥ 4, 7-d refractory), H2 chronic (OLS, ≥ 20 pairs, R² ≥ 0.9, b > 0,
  extrapolation ≤ 1.5×, 4-day mean all observed, ≤ 184 mg/L, `pre_gap` 3), H4 (> 10, 14 d) and H5b (> 20,
  7 d ≤ 15) match the table, except for W1. Labels cover (d, d+h]. In-event and refractory-continuation days are
  invalid. Labels and placebo candidates use approved values only.
- **Scoring.** The budget is alarm-days on valid non-onset rows, pooled ≤ 1/30, sensitivity 2/30. Every method
  is scored on identical rows (H2 restricted to SC rows). Quantile clipping is shared. The BSS reference is the
  identically Platt-recalibrated climatology. CIs are two-stage basin > site-month with B = 2,000. Placebos are
  K = 20, month-matched, clean-candidate days only, away from all event runs.
- **Freeze record.** The freeze-record draft has the required keys and covers `EW_SCORING_CODE`. `load_freeze`
  rejects a mismatched code hash or file list. The score phase verifies the calibration SHA against the freeze
  record, the artifact SHAs against the calibration file and the split SHA, and requires a predict manifest for
  q-models.

### Checklist summary
| Category | Checked | Passed | Failed | Notes |
|---|---|---|---|---|
| Test isolation (guards, run logs, registry) | 6 | 5 | 1 | W11 (test-role label counts outside guard; not model scoring) |
| Hazard definitions / labels / coverage | 9 | 7 | 2 | W1 (H4 ≤/<, unstated rules), W7 |
| Calibration discipline | 7 | 6 | 1 | W2 wording |
| Selection + configs tried | 4 | 3 | 1 | W9 |
| Controls (§8) | 10 | 1 | 9 | C4 |
| Statistics (primary, Holm, power, pooled) | 7 | 2 | 5 | C1, W6, W8, W12 |
| Freeze completeness / reproduction / single pass | 8 | 5 | 3 | C2, C3, W3–W5 |
| Live trial (§11) | 6 | 1 | 5 | C5 |

**Re-review required** for C1–C4 and W3–W6 (code and protocol diff, plus the regenerated candidate hashes) before
`FREEZE_ew.json` is written. C5 can be re-reviewed separately before 2026-12-01.

---

## M5 BioAssess — delta re-review (2026-11-08)

**Review mode:** delta re-review at the pre-test gate (review skill §1–5, §6.2, §7), limited to the fixes for
C1–C5 and W1–W12 of "M5 BioAssess — pre-test gate (2026-10-25)".
**Read:**
- `docs/handoff/bioassess.md`;
- protocol §1, §5, §7, §9.1 and §10 (sha256 `71524d38…`);
- `bioassess_freeze_candidate.json` (prepared 2026-11-08 18:55:12);
- `scripts/eval/bioassess_final.py` (`dfb3f8a5…`) and `scripts/eval/bioassess_freeze_candidate.py`;
- `sentinel/models/bioassess/{cv,baselines,evaluate,transfer,uk,data}.py`, `sentinel/core/{gpu,registry,runlog}.py`
  and `sentinel/models/chemtox/__init__.py`;
- `tests/test_bioassess_models.py`;
- the dry run `final_dryrun_dev_20261108T155353/` and the reproduction `phase1_cv_20261025T025126/`;
- `results/registry.jsonl`, `TRAINING_LOG.md` / `logs/training_runs.jsonl`, and the M5 line of RESULTS.md.

**Reviewer actions:** read-only, plus the following.
- `pytest -q tests/test_bioassess*.py` on CPU: **33 passed**.
- `verify_record(candidate, check_data=True)` returned `[]`. An independent sha256 recomputation of all 24 code
  files, all 9 data files and the TabPFN checkpoint also gave 0 mismatches.
- Re-scored the dry run's `pred_testA.parquet` with the frozen `score_frame` into the scratchpad (CPU, 5 workers):
  **exact match**. All 450 block-A registry rows (value, CI and `extra`), all 48 comparisons and all 24 Holm decisions are identical to the dry run's.
- Metadata-only check of development rows (`load_nrsa(DEV_SURVEYS)`): `elevation_m` coverage by survey and
  ecoregion.

The test pass was not run, `bioassess_freeze.json` was not created, and no M5 file was changed.

### Summary

**Overall status: 🟡 PASS WITH CONCERNS. The lead may freeze after the three small pre-freeze items R1–R3
below.** Once R1–R3 are done, the candidate record has been regenerated and `verify_record` returns `[]`, no further
re-review is needed. A hash/diff check of that edit is enough.

C1–C5 and W1–W12 are resolved in code, record and protocol. The scorer has now run end to end on development data.
Candidate A's registered development LOEO score reproduces to < 2e-8. All 24 + 9 + 1 hashes match the files on
disk. The fold fix in `cv.py` cannot affect any registered result or the real pass. The dry run wrote nothing to
`results/registry.jsonl`, and no M5 code read 2018-19, 2023-24 or reserved-UK labels.

Three gaps remain. All three are frozen into hashed files or need an action at freeze time, so they must be closed
before `bioassess_freeze.json` is written:
- the 2018-19 exposure statement is now incomplete, because D5 has since used 2018-19 as its validation cycle;
- three unhashed M4 modules are imported for the first time after the one-time marker;
- the frozen tree is still untracked in git.

### Resolution of the 2026-10-25 findings

| ID | Status | Evidence |
|---|---|---|
| C1 per-seed | ✅ | `Predictor` keeps seeds 0–2 for every stochastic model (`STOCHASTIC`; deterministic baselines are replicated). An ensemble's seed-s prediction = mean of its members' seed-s fits (`_fit`, memoised per fold). `pred_test*.parquet` stores `pred_seed{0,1,2}` + seed-mean columns. Every Δ row carries `extra.per_seed_delta` and `all_seeds_positive`; every marginal row carries `seed_r2`. `sota_label` = Holm-rejected ∧ all three per-seed Δ > 0, as in the §9.1 claim table. |
| C2 H-M5 / Holm | ✅ | Families are built only when `block == rec["confirmatory_test"]` (= "A"): SOTA = 3 × 6 = 18 and H-M5 = model A × 6 = 6. The dry run has 18 `holm_rejected_sota` and 6 `holm_rejected_h_m5` rows, all in block A. Each row's `extra` holds `p_raw`, `p_holm_adjusted`, `alpha`, `family_size` and the per-seed flag. B/C/T rows are tagged "secondary, unadjusted"; the synthetic test asserts that there is no Holm outside A. The stale 12-test sentence is gone, and the claim table in §9.1 matches the record's `families`/`decision_rule`. `holm_adjust` agrees with `bootstrap.holm_bonferroni` (unit test) and is a correct step-down. |
| C3 hash coverage | ✅ (gap → R2) | 24 code files, including shared `sentinel/eval/{splits,bootstrap,metrics}`, `sentinel/core/{seed,registry,paths,gpu,runlog}` and `chemtox/tabpfn_model.py`. 9 data files: every file that `load_nrsa`/`load_uk` read (checked against `data.py`/`uk.py`). The TabPFN checkpoint (path + sha256). 10 package versions. `verify_record`/`verify_freeze` check all of these, and the unit test covers the code-change and package-change refusals. **Gap:** see R2. **Still open:** the git snapshot (R3). |
| C4 single shot | ✅ | `after_setup()` (the marker) runs after the imports, `acquire_gpu`, both `load_nrsa` calls, `load_uk`, the config read and the vocabulary build. `SafeLog` turns non-finite values into `not_estimable.json`. A NaN in `extra` cannot abort the run (`json.dumps` default). Refusal tests: second pass, existing `bioassess_test` rows, changed hash, changed package. The `--dry-run-dev` run completed (145.8 min, GPU 0; all blocks; `not_estimable.json` = `[]`). Reproduction: 12/12 registered `BioSet[bioset_st_flat]` `nrsa0814/loeo_cv` r2/r2_anomaly values reproduced to ≤ 1.4e-8; per-seed R² agree to ~1e-8. |
| C5 run-time params | ✅ | The CLI has only `--authorised-by`, `--dry-run-dev` and `--score-from-predictions`. `tests`, `n_boot`, `n_boot_confirmatory`, `alpha`, candidates, comparators and controls are read from the record, and all blocks run in one pass. |
| W1 vocab | ✅ | `TaxonVocab.build(dev tokens)` in `Predictor`. |
| W2 C controls | ✅ | `controls_for[C]` = leave-BioSet-out `Ensemble[RF[c+g]+TabPFN]` and equal-size `Ensemble[RF[c+g]+LGBM+TabPFN]`; 6 + 6 Δ rows per block. The framing rule "Δ > 0 vs both" is in §9.1 and in `decision_rule`. |
| W3 B | ✅ | B = 10,000 for SOTA/H-M5 Δ on Test A (`ci_method` of all 36 rows); 2,000 elsewhere. |
| W4 sensitivity | ✅ | `extra.sensitivity_ci` holds `block50` and `ecoregion` paired-Δ CIs on all 240 candidate Δ rows. |
| W5 transfer | ✅ | The deviation is documented in §10. T1/T2 use seeds 0–2. T1 is compared with RF[geography] (Test-B fit) on the same rows. T2 is stated to have no comparator. |
| W6 exposure | ⚠️ → R1 | The note is attached to every `nrsa1819` row, and `baseline_bioassess_rf.py` needs `--i-know-2018-19-is-a-test-cycle`. The **content** is now incomplete (R1). |
| W7 config counts | ✅ | `n_configurations_tried` + `selection_note` are in the `config` of all 2,179 dry-run rows. |
| W8 nondeterminism | ✅ | Per-seed parquets are canonical, and GPU name, index and driver are in `config.hardware`. `--score-from-predictions` re-scores A/B/C. The re-score check above gave an exact match. |
| W9 re-run guard | ✅ | `verify_freeze` also refuses on existing `bioassess_test` registry rows (unit-tested). |
| W10 is_baseline | ✅ | `is_sentinel()`. Dry run: all `BioSet[…]` / `Ensemble[BioSet…]` rows, incl. the geography controls, have `is_baseline = false`, and all 16 baselines have `true`. |
| W11 R10 deviation | ✅ | Disclosed in protocol §10 and RESULTS.md. |
| W12 provenance | ✅ | `2bda4e42645a` / `ac57eaee0a78` (script `bioassess_pair_cv.py`, sha `d69b215e…`) supersede `dc08e7e83e36` / `2f821dd3afc4` with identical values. |

### cv.py fold fix — cannot change a registered result or the real pass

The fix is the `_glob` change: a geography column with a non-finite training-fold median is set to 0 for every row
of that fold.

- **Model A:** it is never reached. `geography=False` skips the whole fill block, and the reproduction run
  confirms this.
- **`+geography` variants (C's BioSet member, the geography controls):** the dead-column branch needs a column with
  zero observed values among the *fit* rows.
  - `lat`/`lon` are 100 % observed.
  - `elevation_m` is 0 % observed in 2008-09 but 89.5 % in 2013-14, and present in every ecoregion (133–265 rows).
  - Every LOEO training fold therefore has 1,583–1,715 observed elevations. The 85 % fit subset cannot be empty
    of them, and neither can the block50/random5 folds or the all-development fit of Test B.
  - Test A uses exactly the LOEO training folds.
- **Finite case:** the new code reduces to `fillna(fill)`, which the pre-existing training-median test covers.
  The branch can fire only when the fit rows come from 2008-09 alone, i.e. in the dry run.

### Test-data isolation (re-checked)

- **Registry:** 0 `bioassess_test` rows. No `bioassess` row on 2018-19, 2023-24 or UK-reserved splits apart from
  the disclosed `rf_blocked_baseline`. No row references a `final_dryrun` artifact. The 20 `bioassess` rows added
  since the last review are the 2 W12 re-logs and 18 TabPFN `block50_cv` development rows.
- **Not present:** `final_scored.json`, `bioassess_freeze.json` and any `final_<stamp>` run directory.
- **Code paths:** `allow_holdout=True` / `include_reserved=True` appear only in the non-dry branch of
  `run_pass`. Both dry-run attempts of 2026-11-08 (12:23, 16:00) ran the dry branch.
- **Run log:** the M5 entries in `logs/training_runs.jsonl` since 2026-10-24 are development CV, the reproduction
  and three dry-run attempts only.

### Required before `bioassess_freeze.json` (R1–R3; small, but frozen afterwards)

**R1 — The 2018-19 exposure statement is incomplete.**
- `EXPOSURE_NOTE` (in the hashed scorer), protocol §9.1 "Exposure", and the record's `exposure_note` say that 2018-19
  was exposed "once … to the research-phase RF baseline".
- Since 2026-10-25 03:43 UTC, D5 BioIntegrity has used NRSA 2018-19 as its **validation** cycle: 276 registry rows
  (`biointegrity/phase1_val`, `iter/it1_mmi_regression`, `iter/it2_tuned`), with iterative model choice.
  - The labels are benthic MMI condition, derived from the same benthic samples that are M5's input.
  - One model, `oracle_gbm_chem`, takes the 2018-19 chemistry (M5's targets) as input.
- This does not invalidate Test A/B. M5's candidates, comparators and settings were all fixed on 2008-14 before
  the first D5 2018-19 row, and nothing has fed back since. But the disclosure, as frozen, would be wrong.
- **Fix:**
  1. Reword the note in the scorer, §9.1 and the builder. For example: "NRSA 2018-19 was used before this pass
     (i) by the research-phase RF baseline (`rf_blocked_baseline`) and (ii) as the validation cycle of D5
     BioIntegrity (benthic condition labels; one chemistry-input oracle) from 2026-10-25. No M5 setting was chosen
     on it. M5 was fixed on 2008-14 before (ii)."
  2. Change "Test C (2023-24) is fully unexposed" to "… unscored by any workstream at freeze". D5's 2023-24 test is
     still pending; record in the post-results report which pass ran first.
  3. Optionally drop the note from the T2 rows (UK reserved), where it currently also appears.
  4. Bring RESULTS.md ("2018-19 test untouched") into line.

**R2 — Unhashed code that is imported only after the marker.**
- `baselines.fit_predict("tabpfn_community")` imports `sentinel.models.chemtox.tabpfn_model` lazily, at the first
  TabPFN fit, after `after_setup()`.
- That executes `sentinel/models/chemtox/__init__.py`, which imports `chemtox/evaluate.py` and `chemtox/features.py`.
  All three are M4 files, edited by an active workstream, and none is hashed.
- They do not enter any computation. But an M4 edit that breaks their import (or adds a heavy dependency) would
  crash the real pass *after* the test is spent, and `verify_freeze` would not notice.
- **Fix:** do both of the following:
  1. Add those three files, plus the trivial `sentinel/__init__.py`, `sentinel/core/__init__.py`,
     `sentinel/eval/__init__.py` and `sentinel/models/__init__.py`, to `HASHED`. These were the only unhashed
     `sentinel` modules loaded on the scorer path in an import trace.
  2. In `run_pass`, call `load_tabpfn()` (and import `lightgbm`/`sklearn.ensemble`) before `after_setup()`.

**R3 — Snapshot the frozen tree.**
- Every M5 file and every shared `sentinel/*` file is still untracked in git (`git status`: `??`).
- At freeze, the lead commits the tree, or archives it with `registry.snapshot_code` / a tarball of the 24 files
  and records its sha256 in `bioassess_freeze.json`.
- Otherwise the hashes prove *that* the code changed but cannot show *what* ran.

After R1/R2, rerun `scripts/eval/bioassess_freeze_candidate.py` **last**. If the protocol is edited at freeze (e.g.
§9.1's "independent R10 pre-test review (running)"), make that edit **before** regenerating, then copy the record.

### Warnings (recommended; decide before freezing, because the scorer is immutable afterwards)

- **W-a — Crash after the marker burns the pass, and the observed failure rate is high.**
  - 2 of 3 dry-run attempts failed. The 2026-11-08 12:23 attempt hit a CUDA OOM on GPU 4 within 0.9 min, caused by a
    foreign 9 GB process. It is not mentioned in the handoff or §10, and its run directory
    `final_dryrun_dev_20261108T122205` no longer exists.
  - The real pass trains on 3,946 instead of 1,882 rows and fits Test C separately: in the dry run C reused A's memo.
    A realistic estimate is roughly 6–9 h, not the handoff's "~2.5 h".
  - `SENTINEL_GPU_MIN_FREE_MB` is checked only at lock time.
  - Block A is scored and logged before C/B/T are fitted. A crash later therefore leaves `bioassess_test` rows, and
    the guard then refuses any completion.
  - **Recommended:**
    1. Fit and save **all** predictions (A, C, B, T) before the first `score_frame`.
    2. Cache each (model, fold) prediction to `run_dir`.
    3. Pre-register in §9.1: "an infrastructure failure before the first registry row may be resumed once from the
       cached predictions under the same record and authorisation; the crash is logged".
  - With no scores visible before all predictions exist, a resume carries no outcome information.
- **W-b — `os.environ["SENTINEL_GPUS"] = "4"` is still set** (`bioassess_final.py:47`).
  - Under the current cap file (`0,1`) it has no effect.
  - If the cap file is removed, the scorer is pinned to GPU 4 again. This contradicts the handoff ("lock-helper
    cap").
  - Remove it or make it explicit.
- **W-c — Paths the dry run could not exercise:**
  - the separate `design_c` load on 2023-24;
  - `load_uk(include_reserved=True)` with role `reserved_test` (the role name matches `uk.py`);
  - the 2018-19 exposure branch.

  Each is a few lines and was checked by reading. A unit test of `run_pass` with `test_c != test_a` on development
  surveys (e.g. pseudo-A = 2013-14 subset) would close this.

### Observations

- O1: The bootstrap p can be exactly 0, which the dry run shows. Report it as p < 1/B (1e-4 at B = 10,000) in the
  paper. The Holm decision is unaffected.
- O2: The "≥ 3 of 6" H-M5 verdict is not a registry row of its own; it is derivable from the 6
  `holm_rejected_h_m5` rows. Consider logging the count.
- O3: The per-seed criterion is strict for BioSet-containing candidates.
  - Single-seed BioSet fits are much weaker than the 3-seed mean. Development: seed R² ≈ 0.30–0.33 against 0.357
    pooled for TP.
  - The comparators' seed-s fits are close to their means.
  - The SOTA label will therefore often fail on (iii) even where Holm rejects. This is conservative and
    pre-registered; state it when reporting.
- O4: `--score-from-predictions` re-scores A/B/C but not T1/T2 (their predictions are saved).
- O5: Stale text to fix: the handoff ("GPU: M5 uses only GPU 4"); RESULTS.md ("blocking scorer issues are being
  fixed before freeze"); §10, which lists two dry-run attempts instead of three. Changing the protocol means
  regenerating the record.

### Checklist summary

| Category | Checked | Passed | Failed | N/A |
|---|---|---|---|---|
| C1–C5 resolution | 5 | 5 | 0 | 0 |
| W1–W12 resolution | 12 | 11 | 1 (W6 content → R1) | 0 |
| Hash integrity (record = disk; coverage of the import path) | 4 | 3 | 1 (R2) | 0 |
| cv.py fix inert for registered results / real pass | 3 | 3 | 0 | 0 |
| Dry run: end to end, scratch-only, families, B, sensitivity | 6 | 6 | 0 | 0 |
| Test data untouched (registry, markers, code paths, run log) | 5 | 5 | 0 | 0 |
| Robustness of the single pass | 3 | 1 | 2 (W-a, W-c) | 0 |
| Tests (`pytest tests/test_bioassess*.py`) | 1 | 1 (33 passed) | 0 | 0 |

### Gate decision

**Conditional go.** Do R1 (exposure wording), R2 (hash the chemtox import chain and pre-import TabPFN before the
marker) and preferably W-a/W-b. Then regenerate `bioassess_freeze_candidate.json` last, confirm
`verify_record == []`, and commit or archive the tree (R3). After that the lead may copy the record to
`bioassess_freeze.json` and authorise the single pass. No development result needs to be redone.

---

## Downstream — delta re-review (2026-11-08)

**Scope:** delta re-review of the fixes to the five downstream freeze candidates after the pre-test gate above
(D1 SafeSwim, D1n national beaches, D5 BioIntegrity, D6 MusselGuard, D2 ToxinWatch it3). Checked against the
lead's decisions on that gate's criticals and warnings.
**Mode:** pre-test gate (delta). No test look was run. No FREEZE file was written. No downstream file was changed.
**Method:** one independent reviewer (fresh context, no subagents).
- Read: the five test scripts; `testlook.py`, `common.py`, the changed parts of `safeswim.py` and `musselguard.py`;
  `downstream_d1n_beaches_alpha.py` and `downstream_musselguard_scrule.py`; the new protocol deviation sections;
  the records, data pins and `tests/test_downstream_testpaths.py`.
- Recomputed every hash.
- Re-ran all five `--dry-run-on-val` with `SENTINEL_RUNS` redirected to
  `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/ds_rereview/runs`.
  The jobs ran one at a time, with ≤ 6 threads and no GPU.
- Ran pytest.

There are no git diffs: the downstream files are untracked (no-git instruction). So the review read the
current files against the previous gate's findings.

### Summary

| App | Verdict | Gate items | Remaining |
|---|---|---|---|
| D1 SafeSwim (D1b qPCR nowcast) | 🟢 | C-D1-1, C-D1-2, W1–W5 fixed | — |
| D1n national beaches | 🟢 | C-D1n-1, C-D1n-2, Holm m = 4, W2, W3 fixed | — |
| D5 BioIntegrity | 🟢 | C-D5-T1, C-D5-T2, S-C1 fixed; W1–W3 disclosed | — |
| D6 MusselGuard | 🟢 | C-D6-T1, C-D6-T2, C-D6-T3, S-C1, W1–W5 fixed | — |
| D2 ToxinWatch it3 | 🟢 | C-D2-T1, S-C1, W1–W3, W5, W6 fixed | — |

**No critical issue remains.** One new shared warning (**W-N1**) is recommended before any FREEZE file is
written. W-N1 means a degenerate secondary stratum can crash scoring after the claim, and
`--rescore-from-saved` cannot recover from that crash.
- The fix is one guard in shared `testlook.py`. That changes the scoring sha of all five apps, so if the lead
  takes the fix: regenerate the five records and re-run the dry runs. No number is expected to change.
- If the lead does not take it, the residual risk is low (see W-N1) and the looks may proceed.

**Integrity:**
- Test data untouched: registry `downstream` rows were 2,353 before and after the dry runs, with 0 test splits.
- No `TEST_LOOK.json` and no `FREEZE_downstream_*.json` exist.
- The dry-run predictions contain validation keys only:
  - D1: 2022–23
  - D1n: 2016–19
  - D5: 1,918 val-role visits
  - D2: 2019–21, train/val lake roles
  - D6: origin 2015 only
- I read no test labels. My probes read split assignments (keys) and validation outputs only.

### Verified (evidence)
- **Hashes.** For all five records, the following equal the files on disk:
  - `scoring_code_sha256` and every per-file sha256
  - the file set (`common.scoring_files(app)`, now including the data-pins file and the new
    `_alpha` / `_scrule` scripts through the glob)
  - the split sha
  - the data-pins sha
  - the test-script sha
  - the dry-run `results.json` sha and the artifact shas

  `configs_tried` equals `n_configs_tried(app)`: D1 50, D1n 20 (12 + 8 α grid), D5 92, D6 132 (131 + SC rule),
  D2 72 (the record discloses 88 = 72 + 16 D2n).
  Every `evidence_registry_rows` id exists in the registry with the recorded value, and every one is a
  validation split.
  `pip freeze` sha = record (`c3ba836b…`).
- **Dry runs reproduced** (value-for-value diff against the recorded `results.json`):
  - **D1:** 0 differences, byte-identical.
    - logistic 0.7428; − official +0.129, − GM +0.043.
    - Holm m = 4, with the two unscored rows at p = 1. Week p: 0.006 / 0.000. Beach p: 0.000 / 0.006.
      All reject. `years_scored` = [2022, 2023].
  - **D1n:** 0 differences, byte-identical.
    - Holm m = 4: known −VB +0.039 (p 0), −history +0.067 (p 0); new −VB +0.016 (p 0.032), −history +0.071 (p 0).
      All reject (0.032 is the last Holm step, threshold 0.05 — thin).
  - **D5:** 0 differences at 1e-12. One Brier-CI bound differs in the 17th digit (thread summation order).
    - The four RF rows: +0.0137 / +0.0209 / +0.0126 / +0.0113, p 0.003 / 0 / 0.005 / 0.003, Holm m = 4, all reject.
    - Per-seed differences are all > 0 (min 0.0094).
  - **D2:** 0 differences, byte-identical.
    - Incumbent family: known +0.0092 (p 0.022), new +0.0598 (p 0.029).
    - Simple family: known − location_knn +0.0285 (p 0.004), new − lake history +0.0507 (p 0.048).
    - Per-family Holm (m = 2) rejects all four. The seed rule holds (min per-seed 0.0082).
    - The all-four Holm fails, so the label is "beats both, not robust to all-comparison correction".
  - **D6:** 0 differences, byte-identical (`3ee9f033…`, 30.6 min). Incumbent column `kanankege_r320` (front AUPRC 0.0216 = it2); 7 primary comparisons on the single validation origin (Holm m = 7; at test m = 14), all p 0, diffs +0.060…+0.089, per-seed all > 0 (min 0.056); SC rule front AUPRC 0.0057, coverage 98.4 %. Per-origin / both-origins logic is structural on validation (one origin) and is exercised with two origins by the synthetic end-to-end test.
- **Shared fixes (testlook.py):**
  - (a) Seed rule in `claim_met`. Every scored primary comparison needs a Holm rejection with Δ > 0 **and**
    each of ≥ 3 per-seed point estimates > 0. Per-seed values are computed on the same rows as the paired
    comparison (`y` and comparator finite). Deterministic logistic challengers (D1, D1n) are exempt, and this
    is stated.
  - (b) `verify_data` runs before `cpu_slot` / `authorise` in all five scripts.
    The pins cover every file the modules read:
    - D1: Chicago labels + `safeswim_drivers` + `forcings_daily` + `site_cells`
    - D1n / D2: WQP variable + `_stations` + `huc12` + forcings
    - D5: NARS + `downstream_catchment` + forcings
    - D6: NAS + `huc12` + `census_county` + WQP spcond/pH
    
    D6's partitioned WQP read only touches the pinned partitions; pyarrow ignores `_stations`.
    The pins file is part of the scoring hash.
  - (c) `save_predictions` runs once (O_EXCL marker with the file sha) before any registry write. It saves the
    per-seed columns.
    - `--rescore-from-saved` needs the claim, refuses after `SCORED.json`, and re-checks the file sha.
    - A second save is refused, so re-fitting is impossible.
    - The synthetic end-to-end test (S-W3) runs `score(…, "test")` for each script's set layout through the
      rescore path, with a monkeypatched registry.
  - (d) Metric label derived from the metric (`auprc_diff_paired_huc8` for D6; the test asserts it).
  - (e) `prereg_snapshot` registers sha256 + mtime of the given files plus the FREEZE file
    (split `prereg`, which does not count as a test row).
- **D1:**
  - `safeswim.RESERVE_YEARS = {"d1b": ("2026",)}` removes 2026 from `dev_years` **before** the labels are
    assigned, so no 2026 row enters `X`. The script then restricts scoring to 2024–25 and raises otherwise.
  - The 26 frozen columns are asserted.
  - `safeswim_drivers` is in `DATA_MANIFESTS`.
  - Protocol §11 records: the m = 4 Holm family, the 21 → 15-distinct correction, the power caveat
    (12 beach clusters), the fact that 2026 can only re-test against GM, and approved-vs-provisional discharge.
- **D1n:**
  - Train+val refit for every model, written as a dated deviation (§10). "Known" = clusters with ≥ 1 fit row.
    This matches `wqp_track.assign`: test-year rows of val-role HUC4s are `test_known`.
  - α grid {0.1 … 10⁶} on the inner training-year split (`_inner`: fit 2004–12, score 2013–15; val not used).
    α = 100 is re-selected and is now interior (0.7734, then 0.7579 at 10³).
  - Strata use training-role rows. The HUC4 ≠ water-body caveat is disclosed.
- **D5:**
  - `PRIMARY` = the four RF rows; Holm over four on HUC4.
  - Secondary sets are scored and saved:
    - all first visits
    - revisited locations with persistence
    - the frozen `val_test_new_location` flag (the split holds it for 1,944 test visits, 1,861 True)
    - accuracy at 0.5
    - logistic-geo, ecoregion and location-kNN
  - Predictions are saved for every evaluated unit, including Fair.
  - The ensemble's per-seed predictions = the mean of the two members' seed-s predictions.
  - Disclosures (it3 ordering, 92 configs, grid-edge tuned incumbent, per-family parity) are in §10.
- **D6:**
  - Each test origin is its own set (front / CONUS / MN); pooled is secondary.
  - The primary family = front sets of both origins × 7 baselines → Holm m = 14. `claim_met` requires every one,
    i.e. both origins.
  - Incumbent = it2 (320 km, k 32, nugget 0.1) → `kanankege_r320`, as the written §11b rule requires.
  - SC rule:
    - Runs on validation (`it2_scrule`) and is in `BASELINES`.
    - Monotone in SC, matching the Whittier direction (val front AUROC 0.557).
    - Uses only WQP samples dated ≤ T.
    - Its paired comparison is on SC-covered rows (98.4 % front coverage; disclosed).
  - `census_county` is in the manifests.
  - H-G4 counts as "implemented" only, per the original §10 text. The record says the same.
- **D2:**
  - Known-lake simple comparator = `location_knn`, in the script, protocol §7 and record. The family split
    (incumbent / simple) and the label rule are implemented as decided.
  - Docstring corrected. 88 configurations disclosed.
  - Scum breakdown (`scum_info_any`) and `site_id` are in the predictions and results.
  - National row count pinned (12,357).
  - Dominant clusters and test-block definition are in the record.
- **pytest** `tests/test_downstream*.py`: **115 passed**, 0 failed, rc 0 (102 at the previous gate; new tests include the 5 synthetic test-mode end-to-end runs, tamper / seed-rule / Holm-unscored tests). The registry `downstream` rows were 2,353 afterwards, no `TEST_LOOK.json` or FREEZE file was created, and all record hashes were unchanged after the tests and dry runs.

### New findings
- **W-N1 (warning, shared; recommended before FREEZE).** In test/rescore mode, `score_set` registers every
  paired difference via `log_test_metric`, which raises `ValueError` on a non-finite value
  (`testlook.py:214-215`).
  - `C.paired` returns `diff = NaN` when a set or stratum has a single class, is empty, or the comparator is
    all-NaN on it (checked synthetically).
  - The crash would come after `save_predictions`. `--rescore-from-saved` re-runs the same code and crashes
    again. Fixing the code changes the scoring sha, so `require_authorised` would refuse.
  - The pre-registered path cannot then complete. The saved predictions survive, but any scoring becomes an
    off-protocol deviation.
  - Exposure is low but non-zero. The most exposed sets are the D6 per-origin MN strata (secondary; MN positives
    per origin rose 12 → 24 → 47 → 78 for T = 2006–2015) and the D5 secondary subsets. D1, D1n and D2 scored sets
    are large.
  - **Fix:** in `score_set`, register a paired row only if `np.isfinite(d["diff"])`; otherwise store
    `"not_estimable": true` in the results.
    Also make `holm_primary` / `claim_met` fail closed when no primary comparison is scored. Currently
    `claim_met` on an empty `scored` dict returns True (vacuous); no current script can reach that case.
    Then re-hash all five records and re-run the dry runs.
- **O-N1.** D2 `check_expected("n_national_train_rows")` runs after the claim (`predict`). The row count is
  deterministic given the pinned WQP/HUC12 files and the hashed code, so this is a formality. It could be
  moved before `authorise` (the national frame needs no WA test rows).
- **O-N2.** The record's `pip_freeze_sha256` is not checked by the scripts. The lead should check it right
  before each look (it matches today).
- **O-N3.** `docs/handoff/downstream.md` step 1 still says "D1, D2, D5, D6 beat their incumbent … (4 needed)".
  The bottom section says D6 no longer counts → **H-G4 "beats incumbent" has only 3 candidates (D1, D2, D5;
  D1n counts with D1)**. The plan-level consequence (H-G4 cannot be met by these looks alone) should be stated
  before the looks, not after.
- **O-N4.** D6 §6 lists the challenger as "+ WQP proxies when available". The frozen `pressure_gbm` has no
  WQP inputs (§12 states this). The deviation goes against the challenger, so it is conservative.
- **O-N5.** `downstream_musselguard_scrule.py` ran at 17:25:52, before the last `musselguard.py` edit (17:28:27).
  The dry run recomputes the SC rule with the current module: front AUPRC 0.00571 and pressure_gbm − SC +0.0885, identical to `it2_scrule` → the edit did not change the rule.
- Carried over (accepted, disclosed): thin margins (D2 new-lake simple p 0.048; D1n new −VB p 0.032), and
  selection-biased validation CIs.

### Gate decision
All five may be frozen. The order is the lead's choice. Recommended first: fix W-N1 in `testlook.py`, then
regenerate the five candidate records and re-run the five dry runs. These are expected to be value-identical;
a quick hash re-check is enough, with no further review.

At freeze:
- write each `FREEZE_downstream_<app>.json` from its record
- call `testlook.prereg_snapshot` with the protocol, record and test script
- check `pip freeze` against the recorded sha
- run each look once with the default `SENTINEL_RUNS` / `SENTINEL_DATA`

Post-results review (R10) applies before any test number leaves the repository.

---

## EW-v2 — pre-test re-review (post final chain, 2026-11-08)

**Review mode:** pre-training (gates the single Test-A look and the OSF registration) · **Reviewer:** independent review skill. I did not do this work.
**Status: 🔴 FAIL — do NOT write `FREEZE_ew.json`, register on OSF or score Test-A yet.**

Most of the earlier gate is fixed, and the data promotion is clean. No model has touched test data: 72/72 sonde run records show train/val only, and no test marker exists. Three defects remain in the registered decision procedure and in the H2 labels:

1. H2 Test-A labels would come from the **superseded** USGS pull.
2. The ≥ 50-onset claim gate counts onsets that the protocol does not call evaluable.
3. Holm would take invalid p-values from underpowered hazards.

There is also one single-pass loophole. All of these are code or protocol fixes of hours. None needs new data or new results. Each fix changes `scoring_code_sha256`, so afterwards the candidate, the OSF tarball and the rehearsal must be regenerated, and this section re-checked (delta review).

**Reviewed:**
- Protocol v0.2, `ew_v2_osf_preregistration.md`, and `ew_v2_freeze_candidate.json` (prepared 22:08).
- The final chain (`m1/final_chain.sh`, `after_iv_chain.done`, `promote.out`, `compare*.out`, `candidate_final.out`, `osf_package.out`, `ew_rehearsal_final.out`).
- `scripts/train/sonde_events.py`.
- `sentinel/eval/{ew_controls,ew_scoring,early_warning,hazards,chloride}.py`.
- `sentinel/models/sonde/{guard,common,subdaily,runner}.py`.
- `scripts/data/{compare_sonde_daily_snapshots,promote_sonde_snapshots,build_chloride_derived}.py`.
- The handoff note, `configs_tried.jsonl`, the registry, and the OSF tarball (extracted to scratch).

**Tests:** `pytest tests/test_ew_controls.py tests/test_hazards.py tests/test_chloride.py`: 13 passed.

### Independent checks (reproduced here)
- 🟢 **Candidate hashes match the current files.**
  - `scoring_code_sha256` over the 22 listed files = `ff835a0c…` ✓.
  - `thresholds_sha256` = sha of `ew_v2_cand_bag5_wxrunoff/event_calibration.json` (`3c243c1b…`) ✓.
  - All 90 artifacts re-hashed: they match both the calibration file and `weights_sha256` ✓.
  - `configs/sonde/events.yaml` sha `dcd935ef…` ✓; calibration `config_hash` `c68560d0…` ✓.
  - `split_sha256` `fd4bb14f…` ✓.
  - Pinned manifests `sonde_daily_refresh_20261108.json` = `46d69209…` and `sonde_hourly_refresh_20261108.json` = `435c74cb…` ✓. Note that `promote.out`'s `dcd359ea…`/`74bdc126…` are the regenerated generic `sonde_daily.json`/`sonde_hourly.json`, which are different files.
- 🟢 **Selection used val only.** I re-derived it from `event_scores.csv` (block = `val_oof_by_wy` only), excluding baselines and q_* methods:
  - H1: stack 0.0222.
  - H2: stack 0.2377 > mono 0.2221.
  - H4: stack 0.0035. The baseline logistic is higher (0.0134) but is not selectable, as registered.
  - Secondary: t4 → mono, t2 → stack.
  - Configurations: 8 non-smoke `events_calibrate` runs before the candidate. The 9th (`ew_live_variant_cal_nosubdaily`, 19:25) is the live-trial variant and was not used for selection. It should be disclosed (O2).
- 🟢 **Promotion did not change train/val data.**
  - My own sample check, strict and NaN-aware on the union of indices: 80 random daily sites (1.39 M finite values) and 25 random hourly sites (55.9 M values), old vs refreshed, ≤ 2023-09-30. Values and approval flags are identical; site sets are equal (5,500 / 749).
  - End to end: `ew_v2_cand_dryrun_val_insample` (old snapshot, 17:00) vs `ew_v2_rehearsal_val_final` (promoted snapshot) give identical rows and labels for all 18 hazard × lead files, with max |Δp| < 1e-8 for every method.
- 🟢 **Approval flags in the refreshed tiers behave as expected.**
  - Daily pooled provisional share: WY2024 4.4%, WY2025 8.9%, WY2026 47%, WY2027 99%.
  - Hourly (20 sites): 0 / 0 / 28% / 95%.
  - Labels and placebo candidates mask provisional values (`events.py:68-69`, `:563-564`). Hourly features drop provisional values (`subdaily.py:42-46`).
  - The exception is H2 (C1).
- 🟢 **The H2 regressions have no leakage.** I recomputed all 1,172 candidate-site OLS fits from `pairs.parquet` rows dated ≤ 2021-09-30 (maximum fit-pair date exactly 2021-09-30):
  - Coefficients, n_pairs and eligibility match for every site; 52 are eligible.
  - The ≥ 20 pairs, R² ≥ 0.9, b > 0 and extrapolation ≤ 1.5× rules are implemented as written (`chloride.py:50-75`).
  - `pre_gap = 3` gives the clean window [d−10, d−4] (`hazards.py:80`).
  - The derived series ≤ 2023-09-30 equal `a + b·SC` of the promoted snapshot exactly (6 sites; max |Δ| = 0, NaN pattern identical).
- 🟢 **Primary statistic implemented as written.**
  - BSS vs the frozen-Platt climatology on identical rows.
  - Two-stage basin > site-month bootstrap, B = 2,000.
  - p = (1 + #{BSS* ≤ 0}) / (1 + B) (`ew_controls.py:39-54`).
  - Holm step-down with adjusted p, NaN counted in m and never rejected (`:57-68`).
  - Role cells: the primary is the test-role cell (`sonde_events.py:690-713`).
  - Budget: val in-sample pooled alarm-days ≤ 1/30 for every hazard (0.9968–0.99997).
  - Placebo: K = 20, month-matched, clean candidates, plus a basin-cluster bootstrap (`ew_scoring.py:366-412`, `ew_controls.py:71-96`).
  - §8 controls all run in the final rehearsal. The truncation audit gives max feature diff 0.0 for every hazard.
- 🟢 **The OSF tarball matches.**
  - sha `223f3672…` ✓; 121 files, each matches its `MANIFEST.sha256.json` entry.
  - The 26 code files are byte-identical to the repo. The calibration file and 90 models are identical to the run. The protocol, candidate, split and OSF text are identical to `docs/preregistration`.
  - Secret scan of the extracted package (key/token/secret/password/bearer/`hf_`/`ghp_`/`AKIA`/private-key/`keys.env`/e-mail patterns, long base64 runs): **no hits**. The only `environ` uses are `SENTINEL_GPUS`/thread variables in `runner.py`. Two files contain `/home/bcheng/...` absolute paths (O4). I did not read `keys.env`.
- 🟢 **Frozen invocation (earlier C2) is resolved.** `--set`, `--seed`, `--smoke`, `--temp-split`, other configs, quantile models and LSTM checkpoints are refused. The config hash is checked against both the calibration and the freeze record. argv is written to the marker and to `score_manifest.json` (`sonde_events.py:406-436`). The unit test passes.

### Status of the earlier gate's C1–C5
| Item | Status | Note |
|---|---|---|
| C1 population / CI / Holm / onset count | 🟡 partly | (a) and (b) are resolved. (c) Holm is implemented, but its validity is broken by underpowered members (new C3). (d) The onset count is defined in the protocol, but the code counts differently (new C2). |
| C2 run-time parameter lock | 🟢 resolved | `--split` is not refused before the marker, but it fails after it (W5). |
| C3 single pass | 🟡 partly | The marker is keyed by the freeze-file sha, and the registry clause is dead code (new C4). |
| C4 §8 controls | 🟢 resolved | Implemented, with deviations logged in §8/§12. `event_gbm` is now in `BASELINES`. H-b threshold is compared on POD only. |
| C5 live trial | 🟢 resolved | Split off into a separate amendment before 2026-12-01. Nothing in §1–§10 depends on it. |

### 🔴 Critical issues (must fix before `FREEZE_ew.json` / OSF)

**C1. H2 Test-A labels and inputs come from the superseded USGS pull, with approval flags from the new one. This breaks "Approved values only".**
- `common.attach_chloride` (`sentinel/models/sonde/common.py:410-418`) reads precomputed series from `processed/chloride_derived/sites/<site>.parquet`.
- These were built on 2026-10-25 (manifest `chloride_derived` created 2026-10-25T05:49Z, series to 2026-10-20) from `sonde_daily`, which at that time was the 2026-10-22 snapshot (`build_chloride_derived.py:37,43`).
- The provisional flags, however, come from SC in the promoted cube (`common.py:422-426`).
- For WY2024–25, a day that was provisional (or missing) in the October pull and has since been approved or revised will therefore be labelled with the **old** value as if it were approved. Newly approved SC days stay NaN, which is missingness tied to approval.
- `verify_snapshots` does not check `chloride_derived` or `forcings_daily` (`sonde_events.py:535`: `LOADER_DIRS` holds only daily/hourly). The pin in `data_snapshots` is decorative.
- Train/val are not affected: I verified the series ≤ 2023-09-30 equal `a + b·SC_new`.
- **Fix (preferred):** compute `cl_mgl_mean`/`4dmean`/`max` inside `attach_chloride` from the cube's SC and `regressions.parquet`. That code is hashed, so a stale file is impossible. Assert identity with the stored series ≤ 2023-09-30.
- **Fix (alternative):** rebuild `chloride_derived/sites` from the promoted snapshot, with an identity check ≤ 2023-09-30, and re-pin it.
- **In either case:**
  - add `chloride_derived` and `forcings_daily` to the strict snapshot verification;
  - put `regressions.parquet` (it is the H2 label definition) in the OSF tarball.

**C2. The ≥ 50 evaluable-onset claim gate does not implement the protocol's definition.**
- Protocol §7 defines evaluable onsets as "not censored, ≥ 1 usable issue day in the 3-day window".
- The code counts every non-censored onset in `events` (`sonde_events.py:709`). `daily_events` hard-codes `onset_censored=False` (`ew_scoring.py:330`), so this counts **all** onsets in [first issue date, last issue + L]. That includes onsets with no usable issue row before them.
- POD and placebo use yet another set: `ew_scoring.py:378` drops onsets after the last issue date. The controls use a third set (`ew_controls.py:105`).
- This already shows in the final rehearsal. For H1-t4, `primary_endpoint.json` has `n_evaluable_onsets = 4`, while `b1_events_evaluable = 3`. `random_pod_obs` is 0.25 while `b1_pod_real` is 0.0.
- H1 is projected at about 44 onsets, so a few miscounted onsets can flip "underpowered", and with it the claim.
- **Fix:**
  - write one function for evaluable onsets: onsets in (first issue, last issue + L] that are not censored and have ≥ 1 usable issue row in [onset − L, onset);
  - use it for the gate, POD, placebo and all controls;
  - unit-test an onset at the block start, one after the last issue day, and one with no usable rows.

**C3. Holm over {H1, H2, H4} uses p-values that are invalid for the underpowered members, which inflates the familywise error for H1.**
- In the final rehearsal, H2 (1 onset, 1 basin group) gets p = 0.0055 and `reject_holm = True`, with BSS CI [0.35, 0.95].
- With ≤ 2 basin groups the outer bootstrap stage is degenerate. Replicates that miss the few positive site-months give BSS* driven only by negatives, so P*(BSS* ≤ 0) is not a test of onset skill.
- H2 and H4 are both registered as expected-underpowered. Their spurious "rejections" then step H1's Holm threshold down from 0.0167 to 0.025, or to 0.05.
- My earlier gate recommended "underpowered blocks the claim but not the computation or the Holm step". This is the flaw in that recommendation.
- **Fix (protocol §7 and code `sonde_events.py:749-754`):** a hazard that is underpowered (< 50 evaluable onsets, using C2's definition) enters Holm with p = 1. It is still counted in m = 3. Its bootstrap p is reported only as descriptive. Optionally, also require ≥ 5 basin groups with onsets.

**C4. The single-pass guard allows a second Test-A look under any re-written freeze record.**
- The marker is `.ew_scored_<pass>_<freeze_sha12>.json` (`sonde_events.py:449`). Any edit to `FREEZE_ew.json` (a new `created`, or new hashes after a code change) gives a new marker name.
- The fallback registry check (`:451`) can never fire, because the score phase writes no registry rows (`:817`, "descriptive, not registered").
- So "a second pass is refused (also if registry rows of the pass exist)" (§6) holds only for an unchanged freeze file.
- **Fix:**
  - name the marker `.ew_scored_<pass>.json`, and refuse if any `.ew_scored_<pass>*` exists;
  - log the primary endpoint and its control rows to the registry (`ew_v2_<pass>_score`, required anyway by the shared rule "log every metric") right after scoring;
  - unit-test both "re-written freeze record → refused" and "registry rows present → refused".

### 🟡 Warnings (should fix with the above)
- **W1. Data pins are names and byte sizes, not content.** The manifests list only path + bytes, and `verify_snapshots` compares names and sizes (`sonde_events.py:550-553`). Add a sha256 over the sorted (name, file-sha256) list of each loader directory to `FREEZE_ew.json` and check it strictly. This is cheap: 5.5k + 750 files.
- **W2. The pooled Test-A ∪ Test-B pass cannot run as registered.**
  - In 2027 the WY2026 data need a new snapshot. `verify_snapshots(strict=True)` will then refuse it against the 2026-11-08 pin.
  - A new freeze record would, under C4, also reopen the `test` pass.
  - **Fix:** pre-specify now how the `test_ab` snapshot is pinned: a dated amendment, identity of rows ≤ 2025-09-30 with the Test-A snapshot or else Test-A labels re-derived, and its own snapshot key.
- **W3. Power is honest for H2/H4, but the protocol gives no a-priori Test-A onset projection for H2 at test-role sites.** It gives "1 at val-role sites, 12 across all roles" for the val block. Add the scaled projection, as was done for H1/H4, and state that H2 is evaluated only descriptively on Test-A, with its confirmatory use in the live winter trial.
- **W4. `n_basin_groups_with_onsets` and the CI with about 10 groups.** Report the number of basin groups with onsets next to every primary p. State that with < 10 groups the two-stage CI is approximate.
- **W5.** `--split` is not refused by `check_frozen_invocation`. It fails later (`sonde_events.py:620`), after the marker is claimed, so a typo burns the pass. Add it to the pre-marker refusals.
- **W6. Prior-knowledge statement 1 in the OSF text** says "All 69 sonde run records". There are now 72, all train/val. Also disclose the val-only `ew_live_variant_cal_nosubdaily` calibration and the pooled hourly flag shares computed here.

### 🟢 Observations
- O1. Hourly identity is compared up to 2023-09-30T23:59Z. Local-standard-time day 2023-09-30 ends 05–10 h later, and those hours come from the refresh (`--refresh-start` is UTC). No val issue row uses them, and the rehearsal reproduces the probabilities, so there is no effect. Note it in the datasheet.
- O2. Two manifests describe each promoted directory (`sonde_daily.json` `dcd359ea…` and `sonde_daily_refresh_20261108.json` `46d69209…`). The freeze record pins the latter. State this in the record so other workstreams re-pin consistently.
- O3. `pairs.parquet` holds post-2021 "later" WQP chloride pairs, including Test-A dates, and `regressions.parquet` has `n_pairs_all`. Neither is used by the fit or the scoring. Add this to the W11 disclosure.
- O4. `ew_v2_freeze_candidate.json` and `event_calibration.json` contain `/home/bcheng/...` absolute paths. This is harmless, but consider relativising them for OSF.
- O5. The H4 val-role in-sample realised alarm rate is 1.87/30 (2 onsets, 3 sites). Out-of-cell rates above budget are expected. Never re-tune.

### Checklist summary
| Category | Checked | Passed | Failed | Notes |
|---|---|---|---|---|
| Test isolation / run records | 4 | 4 | 0 | 72/72 train/val; no test marker; no `FREEZE_ew.json` |
| Promotion identity (train/val) | 4 | 4 | 0 | sample identity + end-to-end Δp |
| Approved-only Test-A labels | 4 | 3 | 1 | C1 (H2 stale series) |
| H2 derivation / leakage | 5 | 5 | 0 | regressions recomputed exactly |
| Candidate hashes / selection | 7 | 7 | 0 | |
| Primary statistic / Holm / power rule | 6 | 3 | 3 | C2, C3, W3 |
| Controls (§8) / placebo / bootstrap | 9 | 9 | 0 | |
| Freeze enforcement / single pass | 6 | 4 | 2 | C4, W1 (W2, W5 are design gaps) |
| OSF package / secrets | 4 | 4 | 0 | add regressions.parquet (C1) |

**Gate:** after fixing C1–C4, plus W1, W2 and W5, which are cheap:
1. regenerate the candidate (`--configs-tried 8`), the OSF tarball and the val rehearsal;
2. run a delta re-review of the changed code, the H2 identity ≤ 2023-09-30 and the new hashes;
3. only then may the lead write `FREEZE_ew.json`.

The 2026-11-15 target is still reachable.

---

## EW-v2 — delta re-review (2026-11-09)

**Review mode:** pre-training delta (gates `FREEZE_ew.json` and OSF registration). **Reviewer:** independent review skill.
**Scope:** fixes for the 2026-11-08 re-review findings (C1–C4, W1–W6), protocol v0.3, the regenerated candidate, the OSF tarball and the rehearsal `ew_v2_rehearsal_val_delta`. I verified every claim myself; none was accepted from the handoff note.

**Status: 🟢 PASS WITH CONCERNS. The lead may write `FREEZE_ew.json`, subject to the one condition under W-a below.**

There are no open critical issues. No test data were scored or loaded by model code:
- no `.ew_scored_*` marker exists under `sentinel_runs/sonde`;
- no `ew_v2_test*_score` rows are in `results/registry.jsonl`;
- no `FREEZE_ew*.json` exists.

### Independent checks (all reproduced here)
- **Hashes.**
  - Candidate file `5fe35736…` ✓. Protocol `0d5c8f2e…` ✓.
  - `scoring_code_sha256` recomputed over the 22 listed files = `c110bcff…` ✓.
  - Thresholds `3c243c1b…` are unchanged. All 90 artifacts match both the calibration file and `weights_sha256` ✓.
  - The selection is unchanged (H1, H2 and H4 all use the stack). It was re-derived in the previous section.
- **Content pins (W1).**
  - I recomputed the content hash of all four loader directories with my own hashlib code: daily 5,501 files, hourly 749, forcings 1, chloride 53. All four equal the candidate's `content_sha256`.
  - Manifest pins are unchanged.
  - Refusals tested against the current code:
    - tampered chloride pin → refused;
    - missing forcings pin → refused;
    - changed `scoring_code_sha256` → refused;
    - shortened file list → refused;
    - intact record → accepted.
- **C1 (H2 labels) is resolved.**
  - `chloride_derived` is now a symlink to `chloride_derived_refresh_20261108`; the old build is kept as `chloride_derived_20261025`.
  - Regressions: a, b, r², n_pairs, sc_max, eligibility and reason are identical for all 1,172 sites (52 eligible). Only `n_pairs_all`, the count of post-2021 pairs that is not used, differs, at 4 sites.
  - All 52 series are identical to the old build on every date ≤ 2023-09-30 (max |Δ| 0, NaN pattern identical).
  - Over the full record, the new series equal clip(a + b·SC) of the **promoted** snapshot:
    - `cl_mgl_mean`: max relative difference 2.8e-5 (float32), NaN pattern identical;
    - `cl_mgl_max` (43 sites with daily-max SC): the same;
    - `cl_mgl_4dmean` equals the 4-day rolling mean of `cl_mgl_mean` exactly.
    I report these as aggregate differences only; no Test-A outcome was looked at.
  - The score-time check is `check_chloride_consistency` (`sonde_events.py:232-267`), called at `:276`.
  - `regressions.parquet` is in the tarball.
- **C2 (evaluable onsets) is resolved.**
  - There is one definition, `ew_scoring.evaluable_onsets` (`:334-361`): the onset is not censored, lies in (first issue, last issue + h], and has ≥ 1 usable row in [onset − h, onset).
  - It is used by `placebo_comparison` (`:414`), `ew_controls._site_arrays` (`:142`) and the gate (`sonde_events.py:896`, via `count_evaluable_onsets`).
  - In the delta rehearsal, gate = POD = random = circular-shift counts for every hazard (H1 3, H1-t4 4, H2 1, H4 2), and `gate_equals_pod_set` is true.
- **C3 (Holm and power) is resolved.** `ew_controls.is_underpowered` / `primary_decision` (`:72-104`) work as follows:
  - a hazard is underpowered if it has < 50 evaluable onsets **or** < 2 basin groups with onsets;
  - an underpowered hazard, or one without a result, enters Holm with p = 1 and is still counted in m = 3;
  - its bootstrap p is kept only as `p_one_sided_descriptive`;
  - a claim requires Holm rejection and not underpowered.
  In the rehearsal, H2's p = 0.0055 (1 onset, 1 group) now enters as 1 and is not rejected. Protocol §7 matches the code.
- **C4 (single pass) is resolved.** I wrote my own second-look test (scratch marker directory and scratch registry; `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/rev_ew2/second_look.py`):
  1. first claim → `.ew_scored_test.json`;
  2. second claim under a **re-written freeze record** → refused (marker);
  3. marker deleted, `pass_claimed` registry row present → refused (parsed registry check);
  4. legacy-named marker `.ew_scored_test_<sha12>.json` present → refused;
  5. `test_ab` → separate marker, allowed.
  The `pass_claimed` row is written right after the marker (`sonde_events.py:779`). Primary and control rows are written at the end (`:951`; 210 rows in the rehearsal's scratch registry). The default registry path (`paths.REGISTRY_PATH`) equals the path the claim checks.
- **W2 pooled pass.** `pooled_data_pins` (`:721-740`) requires a dated `FREEZE_ew_test_ab_data.json` created on or after 2027-06-30, with all four pins and a passed identity check ≤ 2023-09-30. The freeze record is never rewritten. ✓
- **W3 H2 projection.** §9 projects ≈ 2 chronic onsets (upper ≈ 7) at the 13 test-role sites, from train/val-role rates. H2 is underpowered a priori and stated as such. ✓
- **W5.** `--split` and the split-file sha are refused before the marker (`:476-481`). ✓
- **W6.** The OSF text and the §12 entry are updated, including the 9th (live-variant) calibrate run. ✓
- **Rehearsal reproduces calibration.** `ew_v2_rehearsal_val_delta` vs `ew_v2_cand_dryrun_val_insample` (pre-promotion): identical rows and labels in all 18 files, max |Δp| = 2e-17 across all methods. All four snapshots match. Truncation audit diff 0.0 everywhere. All three primary hazards are underpowered on val, so no claim is made, as expected.
- **Tests.** `pytest tests/test_ew_controls.py tests/test_hazards.py tests/test_chloride.py tests/test_sonde_daily.py tests/test_sonde_models.py`: **72 passed**.
- **OSF tarball `02ad146b…`.**
  - The `.sha256` sidecar matches. All 126 manifest entries verify.
  - All 29 code files are byte-identical to the repo. Calibration, 90 models, protocol, candidate, split, OSF text, `regressions.parquet` and the chloride manifest are identical to their sources.
  - Secret scan of the extracted package (key/token/secret/password/bearer/authorization/`hf_`/`ghp_`/`github_pat_`/`AKIA`/private-key/`keys.env`/X-Api-Key/NEON_API/USGS_API, e-mail addresses, long base64 runs, binary files included): **no hits**. Four JSON files carry `/home/bcheng/...` paths (harmless). I did not read `keys.env`.

### Rulings on the engineer's self-reported open items
- 🟡 **W-a. Static inputs are not pinned:** `processed/sites/sites.parquet` and `processed/catchment/streamcat_sites.parquet`. The loader reads them at `common.py:43,53,363,377`.
  - Both have mtime 2026-10-22, earlier than every EW calibration, and the rehearsal reproduces calibration exactly, so they are the calibration-time files.
  - They are shared registry files on a shared host, so a silent rebuild before the test run is a real risk: it could change site scope, offsets or static features.
  - **Does not block, on condition that** the lead records them in `FREEZE_ew.json`, e.g. `"static_inputs": {"processed/sites/sites.parquet": "0e2800716c4447ff53094f7100f045d3d0f1d81046da7accf216ca3978fd3a44", "processed/catchment/streamcat_sites.parquet": "b6886581a7564163fec000edc7326ac80514079c20df38ac9a0656989b789d6d"}` (computed here 2026-11-09).
  - The operator then re-checks both with `sha256sum` immediately before the single Test-A invocation and logs the result next to `primary_endpoint.json`.
  - A mismatch means do not run; restore the file first.
  - A code check (adding both to `SNAPSHOT_FILES`) is better, but it would force another candidate/OSF/rehearsal cycle. It is not required.
- 🟢 **The score-time chloride check covers the daily mean only.** It does not block:
  - `cl_mgl_max` is used only by H2-acute, which is secondary;
  - it is covered by the strict content pin;
  - I verified it independently against the promoted daily-max SC (43 sites, float32 agreement, NaN pattern identical);
  - `cl_mgl_4dmean`, the primary H2 variable, is exactly the rolling mean of the checked `cl_mgl_mean`.
- 🟢 **The candidate's val POD/placebo figures use the v0.2 onset set.** This does not block: they are descriptive, selection is by val-OOF BSS only, and it is disclosed in §12. Label them "v0.2 event set" wherever they are quoted.

### 🟢 Observations
- O1. `--allow-test` is not checked before the marker. Omitting it makes `runner.prepare` raise after the pass is claimed. This is conservative (no result, the pass is burnt), but the operator must use the registered `score_command` verbatim. Consider adding the check the next time hashed code changes anyway.
- O2. `min_basin_groups = 2` only removes the fully degenerate case. With 2–9 groups the CI is still approximate; this is reported per W4. With the projected ≈ 10 test-role groups for H1, it is acceptable as registered.
- O3. When writing `FREEZE_ew.json`, copy `freeze_record_draft` verbatim, set `created`, and add `static_inputs` (W-a). Then re-run `guard.load_freeze` on the written file (this checks the code hash) and `verify_snapshots(strict=True)` **before** the OSF upload. Neither touches test data. Add the sha256 of `FREEZE_ew.json` to the OSF registration text.

| Category | Checked | Passed | Failed | Notes |
|---|---|---|---|---|
| Earlier re-review C1–C4 | 4 | 4 | 0 | each verified independently |
| Warnings W1–W6 | 6 | 6 | 0 | |
| Hashes / content pins / refusals | 9 | 9 | 0 | own hash code; tamper tests |
| Train/val unchanged (data + probabilities) | 3 | 3 | 0 | max \|Δp\| 2e-17 |
| Second-look guard | 5 | 5 | 0 | own test |
| Tests | 72 | 72 | 0 | |
| OSF package / secrets | 4 | 4 | 0 | no secrets |
| Self-reported open items | 3 | 2 | 0 | W-a: conditional |

**Gate:**
1. The lead may write `FREEZE_ew.json`, with `static_inputs` (W-a), then verify it as in O3.
2. Then the OSF registration.
3. The single Test-A pass runs only after the lead's separate authorisation, using the registered `score_command` verbatim.
4. Post-results review (R10) applies before any Test-A number leaves the repository.

---

## M4 ChemTox unseen-taxa — pre-registration review (2026-11-09)

**Review Mode**: Pre-Training (stage 1: data + rules frozen, nothing fitted)
**Reviewer**: independent (review skill, PRE-TRAINING mode). I did not do this work.
**Scope:** `docs/preregistration/chemtox_unseen_taxa_plan.md` (sha256 `fe15581f…`, verified),
`sentinel/models/chemtox/unseen_taxa.py` (`1b2b0617…`, verified),
`scripts/data/chemtox_build_unseen_taxa.py` (`7f3e4eff…`, verified),
`$SENTINEL_DATA/processed/chemtox_unseen_taxa/{test_blocks,training_pool}.parquet`
(`649dc7d1…` / `7e915fbe…`, both verified against the plan, the manifest and the registry),
manifest `chemtox_unseen_taxa`, registry `chemtox_unseen_taxa/prereg_snapshot` (3 rows),
`tests/test_chemtox_prospective.py:599-628`, `docs/handoff/chemtox.md:97-128`,
RESEARCH_PLAN.md §15.1 / §15.5, and the M4 section of REVIEW_REPORT_POST_RESULTS.md.
No model was fitted, no test block scored, no frozen or M5-hashed file touched.

### Summary

**Overall Status: 🔴 FAIL — stage 2 (fitting / selection) must not start yet.**

The **data freeze is clean**: every leakage check I ran independently on the two parquet files
passes, and every number in the plan's power table reproduces exactly from the frozen data. The
**design**, however, has six blocking problems, and they all bear on selection rather than only on
scoring, so they cannot be deferred: the headline feature decision (§1.1) is not implementable in
the current code and its silent default does the opposite of what the plan says; the comparator set
contains **no method that can actually transfer to an unseen class**, so the track as written
cannot satisfy §15.1(i) and risks being a contest of constant offsets; two candidates have an
undocumented, taxonomy-dependent prediction path; the plan's own leakage guard is a silent no-op on
the very frames the pipeline uses; and there is no one-time scoring mechanism although the repo
already contains one.

### What I verified independently (and it holds)

Recomputed from the frozen parquet files with my own code (pandas 2.3.3, ≤ 4 threads, no GPU):

| Check | Result |
|---|---|
| Rows of a held-out class in the pool (by ECOTOX `class`) | **0** |
| Row identity (`result_id`) pool ∩ test | **0** |
| `species_number` / `latin_name` / `genus` / `family` / `tax_order` overlap pool ∩ test | **0 / 0 / 0 / 0 / 0** |
| Duplicate `result_id` within test / within pool | **0 / 0** |
| `test_id` overlap pool ∩ test | **0** |
| Pool `class` inventory | Actinopterygii 42,533; Malacostraca 9,386; Branchiopoda 7,085; Maxillopoda 1,191; Cephalaspidomorphi 70; Ostracoda 143; Chondrichthyes 5; 7 algal classes 97 — **no misclassified insect / mollusc / amphibian** |
| Misclassified taxa with a missing `class` | 65 pool rows have `class` = NA; all are `Osteichthyes` (54), `Crustacea` (8), `Algae`/`Cafeteria` (3) — high-rank ECOTOX entries of ADORE groups, **not** hidden held-out taxa (but see 🔴 C5) |
| Curated rows that are neither an ADORE taxon nor a held-out class | **0** — the upstream `ecotox_aquatic.parquet` contains exactly the 6 groups, so the pool discards nothing and `taxon_group` ↔ `class` is consistent |
| `chem_novel` recomputed (CAS ∪ InChIKey, upper-cased, stripped) | **0 mismatches** (995 rows); 0 NA CAS or NA InChIKey on either side; **0** `chem_novel` rows whose canonical SMILES is nevertheless in the pool |
| `parent_novel` recomputed with `prospective.parent_key` over the pool | **0 mismatches** (831 rows); `parent_novel ⊆ chem_novel` holds |
| Plan §2 power table (rows / primary / chemicals / species / refs / SD / replicate floor / groups / strata) | **every cell reproduces exactly**, incl. insects' floor 0.649 and molluscs' `chem_novel` SD 0.738 |
| Hashes: plan, module, builder, both parquet files vs plan §3, manifest and registry `prereg_snapshot` | **all five match**; the snapshot's plan hash equals the current file, so the protocol has not moved since the freeze |
| `pytest -k unseen_taxa_blocks` | 1 passed |

**The "same chemical measured in fish is fine" stance is defensible** and the `chem_novel` stratum
is correctly computed. 763 of 1,179 test chemicals also occur in the pool; that is not leakage of
test rows (identity, species and lineage are all disjoint) and it is what a deployed hazard model
faces. Two caveats to state in the results, not to fix now: the primary endpoint therefore measures
*taxon* extrapolation almost alone, and the joint "new chemical × new class" cell is the
underpowered stratum the plan already labels as such.

### 🔴 Critical issues (must be fixed before stage 2)

1. **🔴 C-M4U-1 — The §1.1 trait-block decision is not implementable in the current code, and the
   default it will fall back to does the opposite of what the plan claims.**
   *Where:* plan:42-47 ("the trait block is dropped for all models on this track");
   `sentinel/models/chemtox/features.py:64-65,118-119,137-141,168` (`TAX_MULTIHOT`,
   `TAX_CONTINUOUS` hardcoded into `PublishedFeatures.fit/transform/_names/_continuous`);
   `sentinel/models/chemtox/data.py:47-57,104-107` (`FeatureSpec.published` is all-or-nothing; only
   `chemical_columns` is configurable).
   - There is **no switch** that drops the Add-my-Pet block while keeping the exposure block. The
     only existing options are `published=True` (traits in) or `published=False` (traits *and*
     duration / media / exposure / conc-type out) — the latter contradicts "structure, chemistry and
     exposure features only".
   - `features.py` is hashed by M5's running pass and must not be edited, so the drop needs new code
     (a subclass or a new block in a new module) that **does not exist and has not been reviewed**.
   - The dangerous default: `prospective.to_adore_schema` (`prospective.py:395-399`) merges the
     per-species Add-my-Pet table, so for every test species all `tax_eco_*` / `tax_lh_*` /
     `tax_ps_*` values are NaN. `PublishedFeatures.transform` (`features.py:149-156`) then imputes
     the **training mean** (0 after standardisation) and emits an **all-zero** multi-hot vector.
     The traits are not dropped: every test row receives one identical, fish-derived surrogate
     trait vector, and the models have split on real fish/crustacean trait values. That is a
     systematic train→test feature shift plus a constant pattern unique to the test blocks — the
     opposite of "weakens every model equally".
   - *Fix:* implement the drop explicitly (new module, not `features.py`), freeze the exact feature
     list in stage 2, and add a test asserting that no `tax_*` column reaches any model matrix. If
     the drop is implemented, say so in the plan with the module and function name.

2. **🔴 C-M4U-2 — No comparator in §5 can transfer to an unseen class; the declared baselines
   degenerate to constants, so §15.1(i)'s "strongest simple baseline" is not met.**
   *Where:* plan:132-148 (comparators), plan:144-146 (trivial baselines).
   - Because the hold-out is at class level, pool ∩ test is empty at **order, family, genus and
     species** level (verified above). Consequently: `species_mean` = `species_duration_mean` =
     `global_mean` (the plan admits this at :144-146); `readacross` can never match species or
     `tax_group`, so every test row falls to the `"global"` level and, when the similarity gate
     bites, to the global mean (`readacross.py:111-140`); a GPBoost hierarchy of
     species ⊃ genus ⊃ group is **entirely unseen**, so it returns the grand mean of the random
     effect by construction (`gpboost_model.py:26-28,38-39`). With the trait block dropped (C1), the
     *only* taxon-aware channel left in the whole design is the per-block intercept.
   - The obvious strong, simple, published comparator is missing: a **chemical-matched surrogate
     baseline** — predict the test row from the pool's fish value(s) for the same chemical — and its
     published form, **interspecies correlation estimation (ICE / EPA Web-ICE)**, which exists
     precisely to extrapolate acute LC50 from a surrogate species to an untested one and publishes
     models for insects, molluscs and amphibians. I measured its applicability on the frozen blocks:
     **91 % of insect, 90 % of mollusc and 82 % of amphibian primary rows** have a same-chemical
     fish measurement in the pool (366 / 273 / 316 chemicals). It is therefore applicable to the
     large majority of the primary endpoint, not a corner case.
   - Without it, "beats the best baseline on unseen taxa" would again be a structural artefact in
     which only the candidate has a usable signal — the same shape as 🔴 C-M4P-1
     (REVIEW_REPORT_POST_RESULTS.md:284), which this very plan was written to avoid.
   - *Fix (before selection, so tuning parity holds):* add (a) a chemical-matched fish-surrogate
     baseline on the full primary endpoint with an explicit rule for rows without a match, and
     (b) ICE/Web-ICE (published regressions, or re-fitted on the pool with the method's own form) on
     the surrogate-available subset, pre-registered as a labelled subset comparison. Then
     `best_baseline` (plan:147-148) must include them.

3. **🔴 C-M4U-3 — The strongest *published* method for this exact task is relegated to an optional
   candidate, so a SOTA claim is unreachable as written.**
   *Where:* plan:150-157 ("GPBoost only if its reproducibility is fixed first");
   RESEARCH_PLAN.md:362 §15.1(i); `gpboost_model.py:9-24`.
   - The repo's own citations make the point: Zubrod et al., *Environment International* 186:108607
     (2024) is a published cross-chemical **and cross-species** species-sensitivity method, and
     Yuan et al., *ET&C* 45(9):2466-2482 (2026) benchmark it as the best 2D-structure-based model
     for cross-species prediction. The ADORE `rf`/`xgboost` reference (plan:134-137) is a
     *within-taxa* benchmark; beating it on an unseen class is not "the strongest published method
     for the task".
   - So the plan's current state is self-defeating: if GPBoost's D1/D2 reproducibility is not fixed,
     the strongest published method is absent and **no §15.1 "SOTA" label can be earned on this
     track** (only "competitive" / "negative"). The reproducibility precondition is right as a
     *candidate* rule; it cannot be a reason to omit the published comparator.
   - *Fix:* make the published GPBoost bio-QSAR configuration a **mandatory comparator** (fixed
     `GPModel` seed, fixed `PYTHONHASHSEED`, fixed thread budget, repeated-fit determinism check
     recorded in stage 2 — exactly the preconditions the plan already lists), or state in the plan
     that this track can yield at most "competitive". Also: there is no
     `docs/novelty_checks/chemtox_unseen_taxa*.md`; §15.1/§15.5(3) require a dated novelty search
     before any "first-in-class" wording, and it must cover ICE/Web-ICE and the two papers above.

4. **🔴 C-M4U-4 — Two candidates have an undocumented, taxonomy-dependent prediction path for
   unseen classes; the choice must be pre-registered, not left to a code default.**
   *Where:* `chemprop_model.py:57,371-377`; `readacross.py:36-37,111-140`; plan:150-157.
   - `ChempropTox._task_index` maps any unmatched `tax_group` to **index 0** (`idx =
     np.zeros(...)`, `TASK_GROUPS = ("fish", "crusta", "algae")`). A multitask chemprop — the plan's
     *first* candidate — would therefore silently predict every insect, mollusc and amphibian row
     with its **fish head**, with nothing in the plan or the code saying so. It is an undeclared
     modelling decision inside a frozen-configuration claim.
   - `readacross` silently drops to the `"global"` neighbour pool for 100 % of test rows, so the
     "distance-gated k-NN read-across" comparator is, on this track, a plain chemical-similarity
     kNN over fish/crustacean rows. That should be stated like `species_mean` is.
   - *Fix:* in stage 2, pre-register for each candidate how an unseen taxon is handled (single-task,
     or an explicit, justified head/averaging rule), and report the readacross widening counters
     (`self.widening_`) with the results.

5. **🔴 C-M4U-5 — The plan's primary leakage guard is a silent no-op on the frames the pipeline
   actually builds, and it is NA-permissive.**
   *Where:* `unseen_taxa.py:108-117` and `:120-127` (both `return` when the frame is empty or the
   column is missing); `unseen_taxa.py:84` (`~isin(HELD_OUT_CLASSES)` keeps `class`-NA rows);
   plan:32-33 ("must be called on every training frame of the pass").
   - `assert_no_test_classes` returns **silently** if `"class"` is not a column. `OUTPUT_COLUMNS`
     of `prospective.to_adore_schema` (`prospective.py:106-117`) carries no `class` column — only
     `tax_class` from the Add-my-Pet merge, which is NaN for unseen species — and
     `aux.ecotox_auxiliary` (`aux.py:219-232`) returns a frame with neither `class` nor `result_id`.
     Every training frame the existing ChemTox pipeline consumes therefore passes **vacuously**,
     including the auxiliary/pretraining path the plan explicitly wants guarded. The same applies to
     `assert_no_test_rows` (no `result_id` ⇒ silent pass).
   - The pool filter and the guard both treat `class` = NA as "not held out" (65 such rows; I
     verified all 65 are ADORE-group entries, so **today there is no leak**) — but the rule is
     wrong in general: an ECOTOX row of an insect larva with a missing `class` would pass both.
   - Today the aux path happens to be safe only because `aux.ECOTOX_TAXON_GROUPS` defaults to
     `("fish","crustaceans","algae")` (`aux.py:52,184,209`) — a default, not a guard.
   - *Fix:* make both functions **raise** when the keying column is absent (or take the column
     explicitly), treat `class`-NA as held out unless `taxon_group` is an ADORE group, and add a
     `taxon_group`-based guard for ADORE-schema frames. Add a test that a frame without `class`
     raises.

6. **🔴 C-M4U-6 — There is no one-time scoring guard, although the repo already contains the
   mechanism; the frozen files are unprotected and the freeze test does not pin their hashes.**
   *Where:* plan:190-196 (§7, prose only: "a **single** scored pass per block after the lead
   authorises it explicitly"); `tests/test_chemtox_prospective.py:599-628`;
   `sentinel/models/downstream/testlook.py:52-198`.
   - `sentinel.models.downstream.testlook` already provides exactly what §7 describes:
     `make_pins`/`verify_data` (input + pip-freeze pins), `authorise` (freeze sha + split sha +
     scoring-code sha, exclusive-write marker), `require_authorised` called *inside* the row loader,
     `save_predictions` (write-once), `authorise_rescore` (crash recovery) and `mark_scored`. The
     unseen-taxa plan references none of it, and `unseen_taxa.py` has **no loader at all** — stage 2
     would read `test_blocks.parquet` directly, so there is no choke point to guard.
   - The frozen artefacts are mode 0664 with no `SHA256SUMS` sidecar; the closed track's
     W-M4P-5 (REVIEW_REPORT_POST_RESULTS.md:383) was exactly this.
   - The freeze test skips silently if the files are absent and asserts sizes against `report.json`
     only — it never asserts the pre-registered **sha256**, so a silent rebuild of either parquet
     would leave 29/29 green.
   - *Fix:* add a `load_test_block()` in a new module that calls a `require_authorised`-style check
     and verifies both parquet sha256 against the plan; pin the five hashes in the test
     (no skip when the data root exists); `chmod 0444` both parquet files and write `SHA256SUMS`;
     name the authorisation record, the marker path and the write-once prediction rule in §7.

### 🟡 Warnings (should fix; none alone blocks stage 2 once the criticals are closed)

1. **🟡 W-M4U-1 — Study/reference sharing across the hold-out is unmentioned and unmodelled.**
   359 `reference_number` values occur in both files: **3,757 of 15,726 test rows (24 %)** come from
   a publication that also contributed 15,001 pool rows. Within blocks the primary endpoint is also
   reference-concentrated (molluscs: one reference is 394 of 2,172 rows = 18 %; amphibians top-10
   references 33 %). This is not row leakage, but same-lab/same-protocol/same-batch structure plus a
   same-chemical fish value is an optimism channel, and a chemical-clustered bootstrap
   (plan:167-169) does not absorb it. Add a **reference-clustered sensitivity CI** (as the plan
   already does for parent structure) and report a "no shared reference" stratum.

2. **🟡 W-M4U-2 — The power section asserts adequacy instead of computing a detectable effect.**
   plan:83-94 claims adequacy "at the effect sizes seen in the closed track (0.03-0.3 log units)",
   with no MDE. For scale, the closed track's chemical-clustered paired CI on 301 chemicals had a
   half-width of ≈ 0.21 log units (`crust_novel` −0.313 [−0.540, −0.123],
   REVIEW_REPORT_POST_RESULTS.md:464). With 363-571 chemicals and SD 1.36-1.73, detectable paired
   differences are of order 0.1-0.2, i.e. the lower end of the quoted range (0.03) is far out of
   reach, and on insects anything below ~0.1 is inside the replicate floor (0.649) anyway. Compute
   and pre-register an MDE per block (model-free is fine: resample the frozen blocks with a
   constant-offset pair), and state that no claim may rest on a difference below it.

3. **🟡 W-M4U-3 — The validation splits de-duplicate chemicals by CAS and InChIKey only, not by
   `parent_key`.** plan:119-120. The track's own test strata distinguish `chem_novel` from
   `parent_novel` precisely because a salt, counter-ion or stereoisomer shares neither CAS nor
   InChIKey; leaving those in the selection training rows makes the selection splits easier than the
   test blocks they are supposed to mirror. This is the W-M4P-2 / §4.5 lesson
   (REVIEW_REPORT_POST_RESULTS.md:356). Use `prospective.parent_key` in the split as well.

4. **🟡 W-M4U-4 — Two validation scores are a thin basis for selecting among 32 configurations ×
   ~8 families, and one direction is effectively single-taxon.** plan:117-126. Algae are **100 of
   60,575 pool rows (0.17 %)**, so "crustaceans + algae" is crustaceans; the design is a single
   fish ↔ crustacean pair. The pool contains four classes with ≥ 1,000 rows (Actinopterygii,
   Malacostraca, Branchiopoda, Maxillopoda): a **leave-one-class-out** rotation inside the pool
   would mirror the class-level hold-out far more faithfully and reduce selection variance at no
   extra data cost. At minimum, report the two per-split scores for every selected configuration
   and the spread of the top-5, so the selection noise is visible.

5. **🟡 W-M4U-5 — The feature matrix and the schema adapter are not frozen, and the obvious default
   crashes.** The frozen files carry neither `chem_ws` nor `chem_mp` (ADORE's experimental
   solubility/melting point; `prospective.py:427-428` sets them NaN), so `FeatureSpec.chemical =
   "four"` — the default (`data.py:51`) — would raise on `chem_ws`. They also use ECOTOX names
   (`duration_h`, `media_type_norm`, …) and `chem_structure_class`, while the pipeline expects
   ADORE names and `structure_class`. Stage 2 must pre-register `chemical = "mw_logp"` (as the
   prospective track did), name the adapter function, and hash it.

6. **🟡 W-M4U-6 — `best_baseline` is chosen after the scores are seen** (plan:147-148). Taking the
   minimum-RMSE comparator is conservative for the candidate, but the resulting paired CI is not
   nominal; label it descriptive, and state in advance that no claim rests on beating a trivial
   baseline that is structurally the global mean (plan:144-146 already says the latter — good).

7. **🟡 W-M4U-7 — "Molluscs" is 4 classes with a 96 % / 4 % split and a metal-heavy block.**
   Polyplacophora contributes 6 rows and Cephalopoda 2 of 4,430; the primary (in-domain) endpoint is
   only 2,172 of 4,430 rows (49 %), against 83 % for insects and 78 % for amphibians. Report the
   per-class macro breakdown and word any mollusc claim as Gastropoda + Bivalvia.

8. **🟡 W-M4U-8 — Deeper design risk, worth stating in the plan before the fact.** With the trait
   block dropped (C1), no shared lineage at any rank, and no phylogeny or species embedding, *no*
   model on this track has a channel through which taxon information can transfer; the best
   achievable behaviour is a taxon-blind chemistry model plus a per-block offset it cannot know. A
   "SOTA on unseen taxa" claim is then close to unreachable by construction, and the honest result
   is a quantified statement of how far chemistry alone carries. If the track is meant to be
   winnable, give at least one candidate a legitimate, pre-registered cross-taxon channel that is
   defined for new species — class-level traits derivable from ECOTOX, body size, or the
   cytochrome-*b* / taxonomy species embedding already contemplated in `docs/handoff/chemtox.md:163`
   — and declare it before any fit. (This is a design observation, not an integrity finding.)

### 🟢 Observations

- O1. Freeze hygiene is good: the `prereg_snapshot` rows carry all five hashes, the manifest records
  `git_commit`/`package_hash`/`script_sha256`, and the plan hash in the registry equals the current
  file — credible ordering evidence that nothing was written after the freeze.
- O2. `power_summary` does read test-block targets (`unseen_taxa.py:138-155`), which the module
  docstring and plan:72 disclose; the disclosed quantities (SD, replicate floor, stratum sizes) are
  model-free, so this is acceptable pre-registration, and I reproduced all of them.
- O3. The class-level hold-out rationale (plan:35-40) is the right call and is more conservative
  than the species-level alternative; I confirmed it leaves no lineage path (order/family/genus all
  disjoint).
- O4. Restricting the pool to ADORE's taxa costs nothing: the upstream curated table contains
  exactly the 6 groups, so 0 rows are discarded by that rule.
- O5. The three blocks are not independent samples of "unseen taxa" difficulty: insects are the
  arthropod neighbours of the crustacean pool and amphibians the vertebrate neighbours of the fish
  pool. Expect a block-ordering effect and do not average RMSE across blocks (plan:174-176 already
  forbids pooled claims — keep that).

### Checklist summary

| Category | Items checked | Passed | Failed | Notes |
|---|---|---|---|---|
| Frozen-artefact integrity (5 hashes, manifest, registry) | 8 | 8 | 0 | recomputed myself |
| Data leakage (class / species / lineage / row id / dup / study) | 10 | 9 | 1 | W-M4U-1 (shared references) |
| Stratum correctness (`chem_novel`, `parent_novel`, `in_adore_domain`) | 4 | 4 | 0 | 0 mismatches |
| Power table reproduction | 30 cells | 30 | 0 | exact |
| Guards (`assert_no_test_classes`, `assert_no_test_rows`, pool filter) | 3 | 0 | 3 | C-M4U-5 |
| Feature design (trait drop, undefined-for-new-species features) | 4 | 1 | 3 | C-M4U-1, W-M4U-5 |
| Selection design (splits, budget, no test information) | 4 | 2 | 2 | W-M4U-3, W-M4U-4 |
| Comparators / §15.1 reachability | 4 | 0 | 4 | C-M4U-2, C-M4U-3 |
| Statistics (Holm, seeds, CIs, replicate floors) | 6 | 4 | 2 | W-M4U-1, W-M4U-2 |
| Reproducibility (seeds ≥ 3, GPBoost admissibility, env pin) | 4 | 3 | 1 | C-M4U-3 interaction |
| One-time scoring guard | 4 | 0 | 4 | C-M4U-6 |
| Tests | 2 | 1 | 1 | passes, but no hash pin / silent skip |

### Gate

1. **Stage 2 (fitting and hyper-parameter selection) may not start.** C-M4U-1 to C-M4U-5 all change
   the feature matrix, the comparator set that must be tuned with parity, or the prediction path,
   so starting selection now would have to be thrown away.
2. Fix C-M4U-1, C-M4U-2, C-M4U-3, C-M4U-4, C-M4U-5 in the protocol (and, for C1/C5, in new code
   outside the M5-hashed files), re-freeze the plan with a new sha256 and a new `prereg_snapshot`
   row, and request a short delta review of the diff. C-M4U-6 must be closed before the scored
   pass; closing it now is cheaper.
3. The frozen **data** needs no rebuild: the blocks, the pool and the strata are correct as frozen,
   and re-running the builder would only churn the hashes. Protect the two parquet files
   (`0444` + `SHA256SUMS`) instead.
4. The warnings should be answered in the re-frozen plan; W-M4U-8 at least as an explicit,
   pre-registered statement of what this track can and cannot show.
5. A post-results review applies before any number from these blocks leaves the repository.

---

## M4 ChemTox unseen-taxa — delta review (2026-11-09)

**Review Mode**: Pre-Training, delta (two revisions since the FAIL of
"M4 ChemTox unseen-taxa — pre-registration review"). **Reviewer**: independent; I did not do this
work. Nothing was fitted, no test row was read: `unseen_taxa_guard` reports **0 claims, 0 registry
rows and no block directory** for all three blocks, and the `chemtox_unseen_taxa_test` experiment
does not exist.

**Hashes verified on disk:** plan `c4891d7266a8a114`, `unseen_taxa.py` `f161c71f9b48d03b`,
`unseen_taxa_guard.py` `f8b4037a8a40db61`, pins `6937c82b8a7d1662`, novelty check
`1c8f222e70b4f473`, tests `174d8295cedf9599` — all as reported. **Data unchanged**
(`649dc7d1…` / `7e915fbe…`, now mode 0444 with a `SHA256SUMS` sidecar). The four M5-hashed files
(`__init__`, `evaluate`, `features`, `tabpfn_model`) are byte-identical to the pins in
`$SENTINEL_RUNS/bioassess/final_20261108T220242/freeze_record.json` and have 10-22/10-24 mtimes:
**untouched**. `verify_freeze()` passes. 11 new tests pass (+29 in the prospective file = the 40
reported).

### Summary

**Overall Status: 🟡 PASS WITH CONCERNS — stage 2 may start once D-1, D-2 and D-3 are applied.**

All six criticals and the warnings of the first review are genuinely closed, and I re-verified the
substantive ones by running the new code rather than reading it. The §1.4 channel amendment is
well constructed and, unusually, raises the bar it created: Channel A is provably pool-only, and
the channel-matched comparator rule (d2) is the right answer to C-M4P-1. Three defects remain, all
of them small text/code fixes that need **no rebuild and no re-run**; two of them are in the new
channel code, which is why they must be closed before any fitting rather than before scoring.

### Verified closed (by execution, not by reading)

| Item | Evidence |
|---|---|
| C-M4U-1 trait drop | `TaxonBlindBuilder` + `TRAIT_PREFIXES` + `assert_no_trait_columns`, applied to every model (plan:51-61, 402-403); `to_model_schema` injects constant placeholders that the wrapper then deletes; `features.py` untouched |
| C-M4U-2 surrogate/ICE | `FishSurrogateBaseline`, `ICERegressionBaseline` mandatory (plan:347-367) with a frozen fallback rule and dual reporting (full endpoint + surrogate-available subset) |
| C-M4U-3 published comparator | `gpboost_published` mandatory, `SeededGPBoost` (seeded `GPModel`, pinned threads, Cholesky instead of the CG solver that produced D2's NaNs), two-process determinism check, and a pre-registered **failure⇒downgrade** rule (plan:327-346); novelty check filed |
| C-M4U-4 taxon routing | `assert_taxon_agnostic` makes `chemprop multitask=True` inadmissible (verified it raises); `readacross` degeneracy and its `widening_` counters pre-registered (plan:368-373) |
| C-M4U-5 guards | Both guards now **raise** on a missing keying column and treat `class`=NA as held out; I called them on a column-less frame and on NA rows and both raised |
| C-M4U-6 one-time guard | `unseen_taxa_guard` mirrors `testlook`: pins, `verify_freeze`, `O_CREAT\|O_EXCL` 0444 marker, registry refusal, `load_test_block` choke point, write-once predictions, `mark_scored` + `seal_dir`; tests **pin all five hashes and fail rather than skip** |
| W-M4U-1/2/4/5/6/7 | reference-clustered **co-primary** CI (plan:242-248); MDE computed (`unseen_taxa.mde`); leave-one-class-out rotation — fold sizes reproduce exactly (838/42,533; 9,724/9,386; 12,750/7,085; 26,806/1,191); `chemical="mw_logp"` + `to_model_schema` frozen; `best_baseline` labelled descriptive; mollusc wording and no cross-block averaging |
| **Channel A is pool-only** | Overwriting **every** test target with −999 leaves all seven features **bit-identical**; fitting on pool+test raises; `transform` is a keyed lookup |
| **Per-fold refit is honest** | Refitting inside `validation_splits`: Actinopterygii fold → 6 fish chemicals and **0** ICE pairs; Malacostraca 1,853/959; Branchiopoda 1,846/1,460; Maxillopoda 2,351/5,354 — a fold's surrogate coverage really is what test-time coverage will be |
| Channel count | registry `prereg_snapshot_v3` `n_channel_features` = 20 = 7 surrogate + 13 higher-rank one-hots ✔ |

### 🔴 Must fix before stage 2

1. **🔴 D-1 — The parent-key blocking of the validation folds is a silent no-op: `parent_key` is not
   in `training_pool.parquet`.**
   *Where:* `unseen_taxa.py:565-580` (`pk = pool["parent_key"] if "parent_key" in pool.columns else
   None`); plan:297-301 and :284-296.
   - I confirmed the frozen pool has **no `parent_key` column** (the test blocks do). So `bad_pk`
     is empty, the fold tables in plan:290-295 were computed with CAS/InChIKey blocking only, and
     the W-M4U-3 fix the plan credits is **not in force**. `SurrogateFeatures.parent_geo_` is empty
     for the same reason, so the surrogate's parent-level fallback never fires.
   - This is the exact silent-fallback pattern C-M4U-5 was raised about, reintroduced one function
     later.
   - *Fix:* add `parent_key` to the pool at selection time (`to_model_schema`, via
     `prospective.parent_keys`) and **raise** if it is absent, then recompute and re-freeze the
     §4 fold table.

2. **🔴 D-2 — Four Channel-B levels are identically zero in every training frame and non-zero only
   on test rows, and each is a one-to-one proxy for a held-out class.**
   *Where:* `unseen_taxa.py:811-836` (`HIGHER_RANKS`, `HIGHER_RANK_LEVELS`); plan:148-152 and the
   guard `assert_no_class_or_lower` (`:874-892`); test `tests/test_chemtox_unseen_taxa.py:195-221`.
   - Measured on the frozen files: `rank_phylum=Mollusca` pool 0 / test 4,430;
     `rank_subphylum=Hexapoda` pool 0 / test 8,619; `rank_subphylum=Conchifera` pool 0 / test
     4,424; `rank_subphylum=Aculifera` pool 0 / test 6. Within this dataset **Hexapoda ≡ Insecta,
     Aculifera ≡ Polyplacophora, Mollusca ≡ the mollusc block**, so plan:150-152's "by construction
     it cannot encode anything at class rank or below" is **false as written**. The guard compares
     name *strings*, and the test only checks that a level is not spelled like a class name, so
     neither can catch an information-equivalent proxy.
   - **No target information leaks** — a column that is constant in training carries nothing that
     can be fitted — but two real hazards follow. (a) For a neural candidate (chemprop's `x_d`), a
     weight on an input that is always 0 in training receives **zero gradient** and keeps its
     random initial value, then multiplies 1 on test rows only: an arbitrary, seed-dependent offset
     applied exclusively to the test block, invisible in validation. (b) Those columns are
     perfectly collinear with "is a test row", so any model that puts weight on them produces a
     test-only shift — structurally the same defect as the all-zero trait multi-hot the plan itself
     rejects in §1.1.
   - Related interpretation point, worth stating in advance: the amphibian Channel-B vector is
     **bit-identical** to the fish vector, so Channel B gives amphibians a relabelling, not
     information; for molluscs only `rank_clade=Protostomia` is shared with the pool.
   - *Fix (data-independent, so it applies identically in folds and at test):* pre-register that
     one-hot levels which are constant in the **fitting** frame are dropped at fit time — or map
     test-only levels to `Other` — and add a test asserting every retained level is non-zero in the
     training frame. Correct plan:150-152 to say what the mapping does and does not guarantee.

3. **🔴 D-3 — The claim-gating "surrogate-available subset" has two different definitions in the
   frozen artefacts.**
   *Where:* plan:361-367 vs `SurrogateFeatures.transform` (`unseen_taxa.py:767-792`); §6(c)
   (plan:428-429) makes a SOTA claim depend on this subset.
   - plan:361-363 quotes **90.8 % / 89.6 % / 82.2 %** (insects/molluscs/amphibians), which I
     reproduce exactly as *fish-only* same-chemical coverage. The implementation's `surr_available`
     — any pool record, including the parent-key fallback — gives **91.7 % / 92.3 % / 85.3 %** on
     the same primary rows (molluscs differ by 2.7 pp).
   - A pre-registered stratum that gates a claim must have exactly one frozen rule, or the subset
     can be chosen after the scores are seen.
   - *Fix:* name the rule in the plan (fish-only, or any-pool-record), make the baseline, the
     feature and the stratum use that one rule, and re-state the three coverage numbers for it.

### 🟡 Warnings (fix before the scored pass)

1. **🟡 W-D-1 — `verify_freeze` never checks the module and guard pins it writes.** `make_pins`
   records `module_sha256` and `guard_sha256` (`unseen_taxa_guard.py:60-73`) but `verify_freeze`
   (`:76-91`) compares only the two parquet files and the plan, so `unseen_taxa.py` — which now
   holds the channels, the surrogate block and the fold rotation — could change after the freeze
   undetected. Two lines.
2. **🟡 W-D-2 — The hardened guard raises on legitimate ADORE-schema frames.** Its docstring says
   such frames are checked through `tax_group`, but the default is `group_col="taxon_group"` and
   `TRAINING_GROUPS` lacks ADORE's `crusta` code; I passed a `tax_group` frame of
   fish/crusta/algae and it raised "neither 'class' nor 'taxon_group' is present". It fails closed
   (safe) but will block the aux/ADORE path at stage 2 and invites loosening under time pressure.
   Accept both column names and both vocabularies.
3. **🟡 W-D-3 — The configuration accounting is internally inconsistent.** 608 = 8×32 + 11×32 is
   arithmetically right (plan:195-201), but the inventory behind it is not: §1.4.4 says **11**
   mandatory comparators, registry `prereg_snapshot_v3` says **19** (`v2` said 12), and the
   "**+5** fixed-configuration baselines" clause lists **six** items (2 surrogate + ICE + 3 trivial
   means), with §5.1(4)'s `_both` variant a seventh; the published `rf`/`xgboost` configurations
   (plus their `mw_logp` and `_chan` variants) are counted in neither bucket. §15.5(2) requires this
   count to be reported next to the claim, so it must be reconciled now.
4. **🟡 W-D-4 — The ICE fit is described differently from how it is implemented, and its
   small-sample fallback is unregistered.** plan:358 says "fish → crustacean; 16,998 pairs"; the
   frozen code regresses fish → **crustaceans + algae** and I reproduce **17,081** pairs (16,998 is
   exactly the crustacean-only count). Separately, when `ice_pairs_ < 30` the code silently keeps
   `a=0, b=1`, i.e. the ICE feature becomes a copy of the base — which is what happens in the
   Actinopterygii fold (0 pairs). Both need one sentence each in the plan.
5. **🟡 W-D-5 — `surr_n_pool_taxa`: the rename is what makes the feature pass the guard.** I
   verified that `surr_n_species` raises `assert_no_class_or_lower` and `surr_n_pool_taxa` does
   not. The feature itself is admissible — it is a pool-side count and says nothing about the test
   row's taxonomy — but this shows the guard is a string check a rename defeats, so plan:167-169's
   claim that a test proves no class-or-lower information "as a feature name **or value**"
   overstates what a name guard can prove. Document the rename's rationale and add a content-based
   check (e.g. every feature column is a function of pool data only — the test-target-overwrite
   invariant already used for Channel A generalises).
6. **🟡 W-D-6 — Two over-strong statements about the guard.** plan:313-315 says stage-2 code
   "cannot" reach test rows except through `load_test_block`; nothing technically prevents a direct
   `read_parquet`, so the choke point is a convention plus a 0444 file. Also `mark_scored` does not
   call `require_authorised`, and `registry_rows` depends on the split string containing
   `/<block>/`. Soften the wording and pin the scorer's sha in stage 2.
7. **🟡 W-D-7 — Claim condition (g) is stricter than §15.1 and conflates two different noise
   scales.** plan:436 requires the RMSE difference to exceed **the larger of the MDE and the
   replicate floor**. The replicate floor (0.649 on insects) bounds the *absolute* RMSE any model
   can attain; it is not the precision of a *paired difference*, which the MDE (0.011-0.053)
   measures. Requiring a candidate to beat a strong published comparator by **> 0.649 log units**
   makes a SOTA claim on insects effectively unreachable — the over-correction of my own W-M4U-2.
   Recommend: gate claims on the MDE together with the two co-primary CIs, report the replicate
   floor as context for absolute error, and keep "inside the measurement noise" for differences
   below the MDE.

### 🟢 Observations

- O-D1. The §1.4 amendment is correctly ordered: it predates every number, states the expected
  ordering (amphibians > insects > molluscs) **and** the reason molluscs should gain least, and
  hands the channels to the comparators including `gpboost_published_chan`. The "if the observed
  ordering differs, that is a finding about the channel, not a licence to re-interpret the
  endpoint" sentence (plan:506-508) is exactly the right pre-commitment.
- O-D2. `ResidualSurrogateModel` computes its base from pool rows only and adds it back at predict
  time; the test that the base is restored passes. The residual variants are the cleanest way to
  measure "what a learner adds on top of ICE", which is the honest framing of this track.
- O-D3. The novelty check is filed and the plan refuses "first-in-class" wording until the Web-ICE
  metadata marked *to verify* is actually verified (plan:443-445) — honest, and it must be done.
- O-D4. §15.1 is now **reachable**: (i) the strongest published method for the task is a mandatory
  comparator (`gpboost_published`, Zubrod 2024 / Yuan 2026) alongside ICE in Web-ICE's own form and
  the ADORE reference, with channel-matched variants; (ii) chemical- **and** reference-clustered
  co-primary CIs; (iii) seeds 0-2 all-better; (iv) a single look enforced in code. The binding
  constraint on a claim is now the plan's own (g), not §15.1 — see W-D-7.

### Gate

1. **Stage 2 (selection) may start as soon as D-1, D-2 and D-3 are applied** and the plan is
   re-frozen (new sha256, new pins, `prereg_snapshot_v4`). D-1 changes the fold construction and
   D-2 the feature matrix definition, so both must precede any fit; D-3 is one paragraph. No
   rebuild of the frozen data and no re-run of anything is required.
2. W-D-1 to W-D-7 before the scored pass; W-D-3 and W-D-4 are cheapest to fix now, while the plan
   is open.
3. Scoring still requires: stage 2 appended (frozen configurations, feature list, determinism-check
   result, hashes), a dry-run pre-flight, the lead's explicit authorisation, and the §7 guard.
4. A post-results review applies before any number from these blocks leaves the repository.

## M6 HydroMet — pre-test review (freeze candidate, 2026-11-09)

**Mode:** pre-test gate for the single Test-A look. **Reviewer:** independent agent with a fresh context. It was
read-only except for this section. It did not score test data, write a FREEZE file, or use git.
**Scope:** protocol DRAFT v0.11 §1–§12 (`hydromet_protocol.md` sha `fa02da1e`, which was edited during the review,
last at 05:06:55 EST); `hydromet_freeze_candidate.json` (`ed49bc5d`) and its script; `guard.py`, `pipeline.py`,
`evaluate.py`, `lstm.py`, `data.py`, `scripts/eval/hydromet_{score,predict,track_n}.py`, `scripts/train/hydromet_lstm.py`,
`configs/hydromet/lstm_ws{,_full}.yaml`, `scripts/data/build_hydromet_upstream_forcing.py`; the run dirs and
`score_val_v{2,3,4}`; `configs_tried.jsonl`; the registry; the RESULTS.md M6 lines; RESEARCH_PLAN §5.6 and §15.
`scoring_code_sha256` at review time: **`e95283ee…`** (42 files).
**Tests:** `tests/test_hydromet{,_tracks,_upstream}.py` all pass in the `sentinel2` env.

**Status: 🔴 FAIL. Do not freeze yet.** The science of the candidate is clean:
- the §8 rule reproduces from val only;
- the refit leaks nothing;
- the upstream forcings are causal, inputs-only and pinned;
- the expected "competitive" label is stated honestly.

What fails is the test machinery. Three gaps (C1–C3) leave forking paths at the single look, or make the
pre-declared track looks unimplementable under one freeze. Each is a code or protocol fix of hours. No new
training or new val results are needed.

### Verified (passes)
1. **§8 selection on validation only.** I recomputed the rule from `score_val_v{2,3,4}/primary_val_val.csv`: mean over
   the 4 endpoints of min(Δgbm, Δphys).
   - val_v4: `ealstm_ws_full` −0.0015 > `ealstm_ws` −0.0037 > `ealstm_full` −0.0111 > `ealstm` −0.0138 >
     `ealstm_compact` −0.0160 > `ealstm_physdo` −0.0167.
   - The blend (−0.0003) is reported separately.
   - This matches the protocol (§8, lines 209–223) and RESULTS. The comparators in the candidate record equal
     `score_val_v4/comparators.json`. `hydromet_track_r_comparators.json` equals `score_val_v2/comparators.json`.
   - Every val score and every artifact run in the record has `date_end` 2023-09-30 and blocks {train, val}
     (no-test audit, 0 violations).
2. **Refit (cycle 6).**
   - `ealstm_ws_full_v1_s{0,1,2}` were trained on 1,712 train-role sites with 0 inner sites.
   - E_k = 17 / 3 / 4, equal to argmin+1 of `ealstm_ws_v1_s<k>` inner MSE; same grid point. The last epoch is
     kept: NaN val loss saves every epoch, `lstm.py:336-339`.
   - The normaliser is fitted on train-role × train block only (`data.py:514-536`). The training code refuses
     non-train roles (`lstm.py:224`).
   - Timeline is consistent with the v0.11 row. Times are EST: val_v3 was registered at 00:49 and ws_full s0
     started at 01:12.
3. **Upstream-basin forcings.**
   - gridMET only, averaged over NHDPlusV2 or NLDI basins.
   - The NLDI swap is triggered by StreamCat `wsareasqkm`, not by NWIS drainage area.
   - No target is read. Same-day nowcast, as for site-cell forcings.
   - The file is unchanged since 2026-11-08 14:57, which is before any `_ws` training.
   - It is content-pinned in the record.
4. **Data snapshot (the promotion of 2026-11-08 22:08).**
   - M6's run records (`data_manifests`) name only `sonde_daily`, `forcings_daily` and `streamcat`, with no sha. So
     no pin points at the old snapshot.
   - The candidate record pins the **promoted** `sonde_daily_refresh_20261108`: manifest `dcd359ea…`, content
     `65d86584…`, 5,501 files.
   - It also pins `forcings_daily`, `hydromet_upstream_forcing`, `streamcat` and `sites`. The patterns match the
     loader reads (`data.py:59-63, 352`; `sonde/data.py:27`).
   - `verify` → **PASS**, data and 14 artifacts, run by me.
   - Two `ealstm_ws` seeds trained across the promotion. The identity check (5,500 sites identical ≤ 2023-09-30)
     makes this harmless.
   - The test look reads WY2024–25 labels from the refreshed snapshot, as it should.
5. **Guard basics and statistics.**
   - Test blocks and roles need `--allow-test` plus a freeze with split, code and weights hashes.
   - Protected prediction files must carry the frozen artifact sha and the current freeze sha.
   - On protected data, comparators are read **only** from the freeze; `--comparators` is ignored there
     (`pipeline.py:217-220`), so the new flag does not change primary scoring on test.
   - 3 seeds and the every-seed rule are in code.
   - The basin-group paired bootstrap uses B = 2,000.
   - The Holm family keeps untestable hypotheses at p = 1 (`evaluate.py:261-316`), so models missing on test
     cannot shrink it.
6. **Label honesty.** No val endpoint is won. The protocol (line 223) and RESULTS say the expected label is
   "competitive", and that tracks R and N never replace the primary claim (lines 299–300 and 384).

### Critical (must fix before `FREEZE_hydromet.json`)
**C1. No single-pass guard on the test look.**
- Problem:
  - `hydromet_score.py` (and `hydromet_track_n.py score`) can be re-run on `--block test --allow-test` any
    number of times.
  - "Run **once**" (protocol line 246) is enforced only by convention.
  - This is the same defect as EW-v2 C3/C4.
- Fix:
  - Add a fixed-name marker, keyed by block and track (M6/test, M6/test_b, M6R/test, M6N/test). Write it
    atomically before any protected prediction is loaded, and refuse when it already exists.
  - Also refuse when the registry already holds the experiment.
  - Add a second-look test.

**C2. The primary family and the claim model are CLI-selectable at the look.**
- Problem:
  - The new `--models` flag (`hydromet_score.py:34`, `pipeline.py:229`) is honoured on protected data. It can
    shrink Holm from the pre-registered 28 to, for example, 4 (candidate only).
  - The protocol also never says which model carries the M6 label. §7 tests 7 models, while §8 names one
    candidate.
  - The record freezes artifacts for only 3 SENTINEL models (`ealstm_ws_full`, `ealstm`, `ealstm_gbmblend`).
    The other 4 therefore enter Holm-28 at p = 1. This is valid but undisclosed.
- Fix:
  - Write `primary_models` (all 7) and `n_hypotheses` = 28 into the freeze, and make the scorer refuse a
    different `--models` on protected data.
  - State in §8 that the M6 label is the candidate's (`ealstm_ws_full`) result under Holm-28, that the blend
    is reported separately, and how a Holm-significant non-candidate would be reported.

**C3. The pre-declared track R and track N test looks cannot run under this freeze as specified.**
- **Track R** (§10, lines 292–300: "fold artifacts' sha256 are added to a track-R freeze entry"):
  - `weights_sha256` is keyed `<model>_s<seed>`, so `ealstm_s0` of six folds collide.
  - `load_prediction_set` accepts fold files only with `--track-r cv` (`pipeline.py:116-120`).
  - `--track-r cv` restricts eligibility to train/val roles (`runner.py:174`).
  - On protected data the scorer takes the **original-track** freeze comparators, which include
    `_ws` members that track R never refits.
  - So `M6R/test` needs new code. New code changes `scoring_code_sha256`, which invalidates the primary freeze.
  - Its test cell is the **same** FINAL test-role sites × test block as the primary. Writing it after the primary
    look means writing it after seeing those cells.
- **Track N** (§12, lines 371–384):
  - The test score takes comparators from the CLI, not from the freeze (`hydromet_track_n.py:211-212`).
  - The model set comes from `--models` and from which files are present (`:228-229`).
  - Hypotheses with < 3 paired stations are **dropped** from Holm instead of entering at p = 1 (`:255-259`).
  - Track-N data are not pinned: `hydromet_wqp_national` and the `*_trackn` upstream forcings, which are still
    being built.
  - The `_ws` track-N path has not yet run even on val (`M6N/val_v2` is pending), so a bug fix after the freeze
    would break it.
- Fix (before the primary freeze), one of:
  - **(a)** Implement and rehearse both test paths now:
    - fold-qualified keys `trackr_<fold>/<model>_s<seed>`;
    - per-track `comparators` and `primary_models` blocks in the freeze, enforced on protected data;
    - track-N comparators read from the freeze;
    - p = 1 for untestable track-N hypotheses;
    - track-N data pins in the candidate record;
    - run `M6N/val_v2` first.

    Then freeze once with all three tracks' code.
  - **(b)** Amend §10 and §12 to say that tracks R and N get a **second freeze**, with code written after the
    primary look, and label their test results exploratory.

  (a) is preferred.

### Warnings
- **W1 (high). The candidate record is stale, and `verify` does not check code or protocol.**
  - Record `scoring_code_sha256` `b8c20a01…` ≠ current `e95283ee…`. Record `protocol_sha256` `ea0c0e4a…` ≠
    current `fa02da1e…`. Both changed after the record was written: `hydromet_score.py`, `pipeline.py` and
    `hydromet_track_n.py` at 05:05:52, the protocol at 05:06:55.
  - Make `verify` also compare `scoring_code_sha256` and `protocol_sha256`, and regenerate the record after
    the C1–C3 edits.
  - Make the data pins part of the FREEZE schema, checked by `hydromet_predict.py` and `hydromet_score.py` on
    protected data (as EW-v2 does), instead of a manual pre-step.
- **W2. Power of the primary test cell is not stated.**
  - The test cell has ~36 groups for temperature and fewer for DO.
  - Scaling the val SE (protocol §10) gives an MDE of about 0.034 (temperature mean) at α = 0.05. Under
    Holm-28 it is larger still.
  - The observed val Δgbm for the candidate is +0.002 / −0.002 / −0.016 / +0.011.
  - A GBM-beating result is therefore essentially unreachable, and "competitive" is largely a power outcome.
    Say so in §8, next to the expected label.
- **W3. §15.1(i): no published deep stream-temperature or DO model is named as a re-run comparator.**
  - Examples: Rahmani et al. 2021 unmonitored-basin LSTM; Zhi et al. 2021 CONUS DO LSTM.
  - The plain `ealstm` is effectively that reproduction, but the protocol does not say so. §7 line 190 makes
    published numbers context only.
  - State that, even if an endpoint were won, the label would be "beats GBM and physics baselines (H-G1
    wording)", not "SOTA", unless `ealstm` is documented as the published-method reproduction.
  - The strongest simple baselines are present: `doy_location_gbm` and `air2water_linear`.
- **W4. Track R inconsistencies.**
  - Line 305 still says comparators are "chosen on the track-R CV cell", and line 307 says "Holm over 12".
    Both contradict the 2026-11-09 pre-declaration (fixed comparators; Holm 8 or 12).
  - Fixing comparators from the original val instead of best-of-family on the CV cell is a weakening under
    §15.5(4) and needs a stated reason.
  - "Before any track-R score" is not literally true. The R1 fold logs print descriptive val-cell KGEs for
    `ealstm_s0/s1` and `full_s0/s1`, though no comparator Δ was seen. Disclose this.
  - The CV cell contains the val-role × val cell used for six cycles of selection, so it is not confirmatory.
    State that the only track-R claim cell is `M6R/test`.
  - Its power check covers only the CV cell. The test cell is the original 36-group cell, so track R adds no
    test-cell power.
- **W5. Track N.**
  - The `*_trackn` station basins are built cache-only, without new NLDI requests. So the "NLDI polygon if area
    ratio > 2" rule used for sonde sites can fall back to local delineation. The basins are therefore not quite
    "same builder and rule"; disclose this.
  - §12 line 352 still says "six SENTINEL models".
- **W6. Configuration count (RESULTS line 44).**
  - Val-scored EA-LSTM variants: `ealstm`, `physdo`, `gbmblend`, `full`, `compact`, `nogauge`, `ws`, `ws_full`
    (8), plus capped `ealstm_v3`, plus the unregistered interim look at v0.5 `ealstm` (protocol v0.9 row).
  - The line says "7 variants" and omits the interim look.
  - List the val looks: `M6/val_v1`…`v4`, interim, `M6S/val_v2`, `M6N/val_v1`, `M6N/inregion_val_v1`.
- **W7. Code and protocol were edited during this review (05:04–05:07).** Re-hash at the freeze.

### Observations
- O1. Refit seeds 1 and 2 train for only 3–4 epochs (600–800 steps), against 17 for seed 0. This reflects
  seed-level instability of the early-stopping point. Per-seed val Δgbm on temperature mean is negative for all
  three seeds (−0.005, −0.008, −0.0004). The +0.002 comes from the ensemble.
- O2. The M6 handoff was rewritten at 01:37 on 11-09 and no longer carries the snapshot-promotion notice. The
  candidate record now documents it.
- O3. test_b (WY2026) is 47% provisional in the refreshed snapshot. A test_b look "under the same freeze" will need
  a later snapshot and its own data pin; pre-declare this.
- O4. The track-N power check used test-block sample-day **counts**, not values. This is disclosed (line 230)
  and acceptable.
- O5. `sentinel/core/registry.py` is not in `SHARED_CODE`. This is minor: it writes and does not score.

### Checklist
| Area | Checked | Pass | Fail | Notes |
|---|---|---|---|---|
| §8 selection on val only, counts | 4 | 3 | 1 | W6 count wording |
| Refit / leakage (sites, periods, normaliser) | 5 | 5 | 0 | |
| Upstream forcings (causal, no targets, pinned) | 4 | 4 | 0 | W5 (track N builder) |
| Comparators (§15.1) | 3 | 2 | 1 | W3 |
| Guard: hashes, provenance, comparators on test | 5 | 4 | 1 | W1 |
| Single look / family pinned | 3 | 0 | 3 | C1, C2 |
| Tracks R/N separate looks, splits, power | 5 | 2 | 3 | C3, W4 |
| Statistics (group bootstrap, Holm, seeds, power) | 4 | 3 | 1 | W2 |
| Data snapshot (promotion) | 3 | 3 | 0 | verify PASS |

**Gate.** Fix C1–C3 and W1. Then regenerate the candidate record, re-run `verify`, and request a delta
re-review. After that, the lead may write `FREEZE_hydromet.json` and authorise the single Test-A look. No
experimental result needs to be redone.

---

## M4 unseen-taxa — confirmation (2026-11-09)

Confirmation check of the D-1/D-2/D-3 + W-D fixes from the delta review. Independent; verified by
execution. **Nothing fitted, no test row read**: `unseen_taxa_guard` reports 0 claims, 0 registry
rows and no block directory for all three blocks.

**Hashes** all as reported: plan `36fe088f665d0484`, `unseen_taxa.py` `33b0c7211051c475`, guard
`e3721450bb97727c`, pins `fff1ba54ac4c4ac7`, tests `f70ea16607147915`. Data unchanged
(`649dc7d1…`/`7e915fbe…`). The four M5-hashed files still match the bioassess `freeze_record.json`
pins. `verify_freeze()` passes. **46 tests pass** (11+29 → 46).

| Claim | Confirmed by execution |
|---|---|
| **D-1** | `validation_splits` **raises** without `parent_key`; `add_parent_key` derives it from the pool's own canonical SMILES (16 of 60,575 null, RDKit misses); folds **676 / 8,378 / 11,523 / 24,505** — exactly as reported, 15-19 % smaller than before, so parent blocking is now really in force ✔ |
| **D-2** | `HigherRankFeatures` keeps **9 of 13** levels; **no kept level is constant-0 in the fitting frame and none is non-zero on test while 0 in the pool** (I checked every column). Per-fold dropping works (Actinopterygii fold keeps 6). Vectors: amphibians = Chordata+Vertebrata+Deuterostomia (bit-identical to fish), insects = Arthropoda+Protostomia (`Hexapoda` dropped), molluscs = Protostomia only — matching plan:192-194 ✔ |
| **D-3** | One rule (any pool record by CAS or `parent_key`); I reproduce the table exactly: 6,776/7,165 = 94.57 %, 2,021/2,172 = 93.05 %, 1,826/2,084 = 87.62 %, chemicals 405/297/342 ✔ (plan rounds molluscs to 93.1 %; 93.05 → 93.0 is the nearer value — cosmetic) |
| W-D-1 | `verify_freeze` now also checks `module_sha256` and `guard_sha256` (`:89-98`) ✔ |
| W-D-2 | An ADORE-schema `tax_group` frame of fish/crusta/algae passes; `tax_group="insects"` still raises ✔ |
| W-D-3 | One inventory table: 19 searched entries × 32 = **608**, **12** fixed, **31** entries; registry `prereg_snapshot_v4` carries the same three numbers (plus `n_channel_features` 16 = 7+9, levels kept 9 / dropped 4) ✔ |
| W-D-4 | ICE fits fish → **every non-fish** pool row, **17,081** pairs (reproduced), `MIN_PAIRS = 30` with a declared identity fallback ✔ |
| W-D-7 | Claim condition (g) is now the **MDE only**; plan:518 states explicitly that the replicate floor is not a difference threshold and is reported descriptively ✔ |
| Extra fixes | `FishSurrogateBaseline.fit` no longer silently re-fits on non-fish rows when the surrogate taxa are absent (records `groups_present_` and degenerates to the pool mean), and the fallback level is always the **full** training frame's mean, not the subset's ✔ |
| Channel A invariant | Still holds after the rewrite: overwriting every test target leaves all seven features bit-identical; fitting on pool+test raises ✔ |

### 🟢 Two cosmetic leftovers (no gate)

- The paragraph after the inventory table (plan:242-247) says "**19** are mandatory comparators"
  and then counts 17 without the surrogate/ICE baselines and **21** with them; the table and the
  registry are consistent, only this sentence is garbled. Use 21 (or 17 + 4) and drop the 19.
- `ICERegressionBaseline`'s docstring still says the pairs are "fish → crustacean"; the code and
  plan:426 say fish → non-fish. One line, in a pinned module, so fix it at the next re-freeze.

### Gate

**Stage 2 (selection) may start.** The three blocking items are closed in code, verified by
execution, and the plan, pins and `prereg_snapshot_v4` agree with what the code does. Scoring still
requires stage 2 appended (frozen configurations, feature list, determinism-check result, hashes),
a dry-run pre-flight, the lead's explicit authorisation and the §7 guard; a post-results review
applies before any number leaves the repository.

---

## M5-X — pre-registration review (2026-11-09)

**Review mode:** Pre-training (pre-registration, before any track-X fit or selection).
**Reviewer:** independent review skill (did not author the work). Adversarial.
**Scope:** `docs/preregistration/bioassess_x_protocol.md` (sha256 `c41f193d…`), split table
`$SENTINEL_RUNS/bioassess/x_track/splits_x_v1.parquet` (sha256 `18a35ef3…`, content `59ab19bf…`), its builder
`sentinel/models/bioassess/xtrack.py`, `scripts/eval/bioassess_x_power.py` + `power_x_v1.json`, test
`test_xtrack_training_rows_and_split_invariants`, registry row `ced6d0538314`, `docs/handoff/bioassess.md`, frozen
M5 result and its post-results review, addenda A1–A4, `docs/novelty_checks/bioassess_2026-11-09.md`,
RESEARCH_PLAN §15.1/§15.5.

### What was verified (by execution, no target values read)
- The 7 file hashes in `ced6d0538314` match the files on disk now. Registry holds exactly one `bioassess_x*` row
  (the snapshot); no M5-X fit or run directory exists.
- The split table reproduces the protocol counts: X-L test 974 NLA-2022 visits / 973 clusters / 9 folds (67–165);
  X-U test 8,121 samples / 2,149 clusters / 19 areas (80–949); X-U UK training pool years 1965–2020, no NaN year or
  area. `validate_splits` and the xtrack unit test pass.
- Power recomputed from `power_x_v1.json`: MDE = (z₀.₉₇₅+z₀.₈)·SE = 2.80·SE and Holm-worst = (z₁₋₀.₀₅/₂₄+z₀.₈)·SE
  = 3.71·SE; SE scaled by √(1,901/n). Protocol table values (X-L 0.017–0.027 / 0.023–0.036; X-U 0.012–0.018 /
  0.015–0.024) are correct.
- Fuse/NLA: no model code outside `bioassess` reads `nars_lakes`; Fuse's use of NLA chemistry was as *input*
  to microcystin. No M5-X selection step uses a Fuse artefact. UK EA labels appear in the registry only in
  `bioassess_phase1_transfer_dev`, `bioassess_test` and Sim's descriptive `conductivity_thresholds`.
- `bio.infer` 1.3-6 (2020-09-02, Yuan) is installed in conda env `rbioinfer`.

### Summary
**Overall status: 🔴 FAIL — development may not start until C1 and C2 are fixed** (both are small, code-level
fixes; no re-design and no re-freeze of the split *file* is needed if C2 is implemented in `training_rows`).
The design is otherwise sound: test blocks are genuinely unscored, the split was built from availability
masks, the hard stratum and robustness condition are declared a priori, and the power numbers are right.

### Critical issues (must fix before the first fit)
1. **🔴 C-MX-1: The M5 UK-pretrained encoder was trained on the X-U test labels and is auto-loaded.**
   `uk_pretrain_20261022T091937/encoder.pt` (pointer `uk_pretrain_latest.json`) was pretrained with a
   chemistry loss (`chem_loss_weight` 1.0) on **all 199,128 `pool_fit` samples of all years**
   (`sentinel/models/bioassess/pretrain.py:58,73`: `Yf = uk.Y.loc[fit_ids]`), i.e. including the 2021–26
   `pool_fit` samples that are now the X-U test, and its vocabulary comes from the same rows (`pretrain.py:63`).
   `scripts/train/bioassess_cv.py:67-68` loads it silently whenever a config has `"pretrained": "auto"` (the M5
   `bioset_st_ukpt` default). Protocol §7 (`bioassess_x_protocol.md:103-104`) lists "with or without UK
   pretraining" without forbidding reuse. Any reuse = direct X-U test-label leakage (and, for X-U, leakage of
   area-g ≤2020 labels from the same sites under leave-one-area-out).
   **Fix:** (a) protocol amendment: M5-X never loads any M5 checkpoint; UK pretraining for X-U is redone
   **per fold** on that fold's training rows only (UK ≤2020 outside area g; vocab from the same rows), and for
   X-L on X-L training-pool UK rows only; early stopping on training-pool rows, never on 2021+ rows;
   (b) the planned loader guard must also refuse the old encoder (sha256 `91453a22…`) and any checkpoint whose
   recorded training ids intersect X-U test/excluded ids; (c) unit test for both refusals.
2. **🔴 C-MX-2: "Location clusters never cross a fold" (`bioassess_x_protocol.md:54`) is false for X-U.**
   `location_clusters` is computed over all UK coordinates irrespective of area (`xtrack.py:87`), and
   `training_rows` drops by region only (`xtrack.py:124-130`). Measured on the frozen split: **170** UK 1-km
   clusters span >1 EA area; **2,242** UK ≤2020 training rows share a cluster with **63** X-U test locations
   (220 test samples, 57 clusters) in a *different* area; **77** training/test pairs are < 50 m apart (min 17 m —
   effectively the same site under a different site id). X-L: 1 NLA training row 73 m from a test lake in another
   ecoregion. **Fix:** in `training_rows`, also drop every row whose `cluster` is in the test-cluster set of fold
   g (or a 1-km buffer); add the invariant to `validate_splits` and the unit test; re-snapshot. Report the number
   of rows removed. The split parquet itself need not change.

### Warnings (fix before the freeze unless stated)
1. **🟡 W-MX-1 (high, before first fit): §15.5 item 7 conflict needs an explicit lead ruling.** Item 7 says
   "additions to a training set never include rows of any frozen test block". The X-L/X-U training pools include
   M5 Test A (NRSA 2018-19), Test C (2023-24, also D5's scored test) and 62,918 ≤2020 samples at M5 T2
   `reserved_test` sites (`bioassess_x_protocol.md:22,50`). Flagging them `previously_seen_at_test` is honest, but
   the protocol must state the lead's interpretation (spent blocks may be training data for a *different* track;
   the M5 numbers are not re-reported from them) as a dated §10 amendment, and the lead should confirm no other
   workstream has a frozen-but-unscored block on NRSA 2018-24, NLA or UK EA.
2. **🟡 W-MX-2 (high): the "10,315 excluded" description is wrong.** Protocol l.51 and the xtrack docstring say they
   are 2021–26 samples at `pool_val`/`reserved_test` sites. Measured: 4,361 `reserved_test` + 771 `pool_val` +
   **5,183 `pool_fit`** (2021+ `pool_fit` samples with no primary target or no area). Correct the text (and the
   handoff). Also note that `pool_val`/`reserved_test` ≤2020 samples (also M5-scored) *are* in training — the
   exclusion is a time-out rule, not a "previously scored" rule; say so.
3. **🟡 W-MX-3 (high): label rules are non-exhaustive and overlap (`bioassess_x_protocol.md:117-127`).** (i) Holm-
   rejected but a seed or the region-unit CI fails, with |Δ| ≥ MDE → no label; (ii) CI includes 0 and |Δ| ≥ MDE →
   no label; (iii) "Competitive" and X-L "Inconclusive" both apply to the same case — no precedence; (iv) "the
   MDE" is not specified (α = 0.05 or Holm-worst; pre-registered proxy or realised SE); (v) "the CI" in "Below" is
   not specified (location-cluster paired, presumably). Write one ordered, exhaustive decision table per
   (block, target) before the freeze.
4. **🟡 W-MX-4 (medium): power wording overstated.** l.142-143 "X-U is adequately powered for effects of that
   size" — M5 within-region effects were TN +0.005, TP +0.010, cond +0.012, below the X-U Holm-worst MDE
   (0.017–0.018). Say: X-U is powered for effects ≥ ~0.015–0.024 (chloride/pH/turbidity-size effects), not for
   TN/TP/cond-size effects. Also the SE proxy comes from NRSA targets; the X-U targets (orthophosphate,
   ammonia, nitrate) have no proxy, and X-U has ~3.8 samples per cluster (temporal repeats) vs ~1 in Test A.
5. **🟡 W-MX-5 (medium): the bio.infer gate conflates fidelity with competitiveness and is asymmetric
   (l.88-90).** "Dev R² below WA-PLS2 on a majority of targets ⇒ not reproduced faithfully" would label a faithful
   run of a weak method as unfaithful, and dropping it can remove the strongest comparator on a minority target
   where it beats WA-PLS2. **Fix:** (a) fidelity = reproduce the package's own documented example on its bundled
   EMAP-West data (`bcnt.emapw`/`envdata.emapw`) and confirm `taxon.env`/`mlsolve` run without our code paths;
   (b) if the software runs, its predictions stay eligible for the per-target max comparator regardless of
   ranking; report "compared, weak" rather than "not compared"; (c) the gate metric must be stated
   (`r2_anomaly` or pooled R²) and fixed. Note `bio.infer` needs `tcltk`; record how it is run headless.
6. **🟡 W-MX-6 (medium): practitioner/index comparators missing for X-U; one method class missing in the
   literature check.** For England the natural practitioner baseline is chemistry regressed on standard EA
   pressure indices (WHPT-ASPT/NTAXA for organic/nutrient pressure; PSI/E-PSI; TRPI for phosphorus; SPEAR; LIFE)
   ± geography — cheap, interpretable, and the comparison a UK reviewer will ask for. RIVPACS/RICT is correctly
   identified as the forward direction, but its O/E outputs can enter that index regression. For lakes,
   chironomid/invertebrate TP transfer functions are WA/WA-PLS-class and covered by WA-PLS2. The literature check
   omits joint-species-distribution **inverse prediction** (e.g. GJAM, Clark et al. 2017, which predicts X from
   composition); search and either add or justify exclusion. Citations above to be verified by the
   bibliography pass. A human Scopus/Scholar pass is still required before any SOTA wording (handoff l.31-32).
7. **🟡 W-MX-7 (medium): comparator parity / feature definitions.** (i) "context" features for UK
   (`RF[community+context]`) are undefined; RICT-style predictors include **alkalinity** (chemistry, strongly
   tied to cond/pH) — list allowed context columns and exclude any chemistry. (ii) Classical transfer functions
   are normally calibrated on a regional set; let WA-PLS2/kNN-MAT choose (on dev, same budget) between the
   multi-source pool and the same-continent pool, otherwise they are handicapped by the cross-continent
   vocabulary. (iii) TabPFN capped at 6,000 rows of a ~270k pool; state RF/LGBM row budgets explicitly.
8. **🟡 W-MX-8 (medium): UK taxonomic-resolution shift across the time-out.** `analysis_method` share of ANLA:
   51 % (≤2010), 67 % (2011-15), 83 % (2016-20), **93 % (2021+ test)**; ANLE ~39 % ≤2010, ~0 % after. Pre-declare
   `analysis_method` as an allowed input/token (or restrict training to ANLA/ANAA) and report X-U by method.
   Dev pseudo-test (2016-20) is a reasonable proxy for this shift; state that.
9. **🟡 W-MX-9 (medium): per-fold vocabulary and scalers are not pre-declared.** §7 must state that taxon
   vocabularies, `min_name_count` filters, feature/target scalers and any OTU tables (incl. bio.infer's
   `taxon.env`) are fitted on fold-g training rows only (M5's W1 rule), for both dev and test designs.
10. **🟡 W-MX-10 (low): prior exposure to X-U labels via M5 development.** The 2021-26 `pool_fit` labels were
    *training targets* in M5's `bioassess_phase1_transfer_dev` and the UK pretraining that informed M5's
    "transfer is negative" conclusion, and M5-X's candidate families are chosen from the M5 *test* outcome
    ("the ensemble form that won M5"). Neither scores X-U, but disclose both in §2/§7.

### Observations
- The builder loads target *values* into memory (`uk.Y` via `load_uk`, `chemistry_wide` in `nla_universe`) and uses
  only `notna()`; "never target values" should read "uses only availability masks (values loaded, not used)" —
  same honesty standard as A1.
- `ced6d0538314` was logged from a stdin script (`argv ["-"]`, no script hash); log future snapshots from a file.
- Region-unit CIs over 9 (X-L) / 19 (X-U) units are coarse; report sign counts alongside.
- X-L training includes NRSA 2023-24 (after the 2022 test year); different ecosystem and out-of-region only —
  disclose under "cycle-out".
- Cross-area hydrological connectivity (same river across EA areas) is not captured by 1-km clusters; the 50-km
  sensitivity CI partly covers it.
- 353 NLA-2022 test lakes have earlier-cycle visits; they are correctly excluded with region g.
- `bioassess_x_power.py` defaults `--family-size 36`; the stored run used 12 — set the default to 12.

### Checklist summary
| Category | Checked | Passed | Failed | Warn |
|---|---|---|---|---|
| Test-block integrity (unscored, availability-only split, X-U excl., temporal) | 8 | 5 | 1 (C-MX-2) | 2 (W-MX-2, W-MX-10) |
| Data leakage (pretrained weights, vocab/scalers, spatial adjacency, §15.5-7) | 6 | 2 | 2 (C-MX-1, C-MX-2) | 2 (W-MX-1, W-MX-9) |
| Endpoint / multiplicity / labels | 5 | 3 | 0 | 2 (W-MX-3, W-MX-4) |
| Comparators (§15.1(i), gate fairness, parity) | 6 | 3 | 0 | 3 (W-MX-5/6/7) |
| Power recomputation | 3 | 3 | 0 | 0 |
| Single-pass / guard design | 4 | 3 | 0 | 1 (guard not yet implemented — required before first fit) |
| Literature check | 2 | 1 | 0 | 1 (W-MX-6) |

### Gate
**Development may start once:** C-MX-1 and C-MX-2 are fixed in code with unit tests; the loader guard (§7/§9,
refusing X-L/X-U test ids *and* the M5 UK encoder) exists and is unit-tested; W-MX-1 (lead ruling) and W-MX-9
(per-fold vocab/scalers) are written into protocol §10 as dated amendments; and a new `bioassess_x/prereg_snapshot`
row records the updated hashes. A short delta review of those items suffices. W-MX-2…8 must be closed before
the freeze candidate.

---

## M5-X — re-review (2026-11-09, v2 / snapshot `47130a305f69`)

**Mode:** pre-training delta re-review (no fitting, no target values read). Verified by execution with the
project env (`sentinel2`, Python 3.11.17), CPU ≤ 6 threads. Another workstream's cache was not written:
the M2 role computation used `assign_lakes(use_cache=False)`.

### Verified (claims that hold)
- Hashes of all 11 files in `47130a305f69` match disk. The split parquet is unchanged (`18a35ef3…`). The snapshot was
  logged from a script file (`bioassess_x_prereg.py`, sha `fdcbcf66…`).
- `tests/test_bioassess_models.py`: **38 passed**.
- `frozen_test_keys()` reproduces the claimed counts: NRSA 3,866, UK `reserved_test` 67,279, Fuse tier C 2,211,
  D2n 149. `validate_training_pools()` returns `[]`. Buffer removal matches: X-U 2,234, X-L 1.
  `pools_x_v1.json` matches a fresh recomputation for 28/28 folds.
- C-MX-2 is fixed for within-block test rows: the 1-km buffer applies to all sources and areas, and the cluster rule
  is applied too.
- W-MX-2, -3, -4, -5, -6, -7, -8, -9 and -10 are written into the protocol as claimed. The decision table is
  ordered and exhaustive; the power wording is corrected; the bio.infer fidelity check now reproduces the bundled
  example and the package's predictions stay eligible; GJAM and the EA indices are added; context excludes
  chemistry; `analysis_method` is used as an input and a stratum; vocab and scalers are fitted per fold.

### Critical issues (new or still open)
1. **🔴 C-MXR-1: the X-L test block (NLA 2022) is inside every X-U training pool.** `make_splits` adds
   `nla.assign(block="X-U", role="train_pool")` for **all** NLA rows (`xtrack.py:113`), so the X-U pool's 4,221
   NLA rows include all **974 X-L test visits plus the 1 excluded visit**.
   - Verified by execution: `training_rows(S, "X-U", "ANGLIAN - CENTRAL", …)` returns 975 NLA-2022 ids.
   - `frozen_test_keys()` does not list M5-X's own X-L test. `validate_training_pools` checks only the same block's
     test ids.
   - This contradicts protocol §2 l.43 ("used only as M5-X *test* rows, never for training"), l.63 and the item-7
     ruling as written.
   - It is also a live leak path. The X-U development design trains on "UK ≤ 2015 + NRSA + NLA" (§7), so the
     X-L test chemistry would be loaded and fitted during development, before the X-L pass. The family, budget
     and settings are shared across blocks, so those fits can steer X-L selection.
   - D2n `test_known` rows (train/val HUC4 × 2021–25) touch only NLA 2022, so this one fix closes them too.
   - The flaw was already in v1 and I missed it in my first review.
   - **Fix:** in `training_rows` (or `frozen_test_keys`), drop the other block's test and excluded rows. Add to
     `validate_training_pools` a cross-block check: no X-L test/excluded id in any X-U pool, and the reverse.
     Add a unit test, recompute `pools_x_v1.json` and re-snapshot.
2. **🔴 C-MXR-2: the encoder guard is not invoked on any load path.** `guard_encoder` has **no call site** outside
   its unit test.
   - The only checkpoint loader, `sentinel/models/bioassess/cv.py:73` (`PretrainedEncoder`, `torch.load`), and
     `scripts/train/bioassess_cv.py:63-76` (`resolve_pretrained`) never call it.
   - `"pretrained": "auto"` still resolves through `uk_pretrain_latest.json` to the forbidden encoder `91453a22…`.
   - No code writes the `.train_ids.json` sidecar yet.
   - The guard checks only overlap with test/excluded ids. An encoder pretrained on **all** UK ≤ 2020 rows would
     pass, even though it includes area-g rows, `reserved_test` frozen rows and buffered rows, all of which the
     fold's pool excludes.
   - **Fix:**
     - (a) Call `guard_encoder` inside `PretrainedEncoder.__init__`, or in a single M5-X loader that every M5-X
       script must use, and make `resolve_pretrained` refuse `"auto"` in M5-X.
     - (b) Strengthen the check to `train_ids ⊆ ids(training_rows(S, block, g, frozen))`.
     - (c) Have the per-fold pretraining script write the sidecar from the ids it actually used.
     - (d) Add unit tests in which the `"auto"` path and a full-UK encoder are both refused through the real loader.
   - The §7 test-id loader guard ("refuses test and excluded ids … before the first fit") also does not exist yet.
     It is the same gate condition and should be built at the same choke point.

### Warnings
1. **🟡 W-MXR-1 (high; decide before the first fit): the item-7 audit is incomplete for M2-Orbit.** The lead's
   list says "M2 none". The M2 split (`orbit_split_v3.json`; `orbit_protocol.md` §6) **protects every test-role
   row in any year**. Its national matchups `s2_wqp_matchups_hist` hold 5,972 `NARS_WQX-NLA` rows (2017:
   turbidity, chl-a, Secchi, microcystin), and I recomputed M2 roles for them with M2's own `split.assign`
   (metadata only).
   - 1,084 rows are M2 **test-role**.
   - They are the same sampling events (same `unique_id` and date) as **218 NLA 2017 visits**. 199 of those are
     in the X-L pool, and **102 are not covered by `frozen_test_keys()`**: they remain in X-L pools (768
     fold-rows in total) and in every X-U pool.
   - Turbidity is a target in both tasks.
   - These rows are protected but are not in M2's scored cells (test years 2024–25), so the lead may rule them
     out of scope. Either add them to `frozen_test_keys()` (≈ 1 % of NLA training rows; NLA 2017 is also the X-L
     development pseudo-test) or record the ruling and its reason in §2.
   - Other checks: M7-Lite `test_n` chips share no NLA sampling event with the pools. M6, Fuse D/W and the HAB
     label sets hold no NARS rows. No other workstream's model code reads `uk_ea_ecology`.
2. **🟡 W-MXR-2 (low): decision-table row 4 rarely fires.** The CI half-width (≈ 1.96·SE) is compared with MDE_X
   (≈ 3.71·SE at Holm-worst), so "Inconclusive" needs a realised SE about 1.9× the proxy SE. Underpowered X-L
   nulls will mostly read "Competitive". That is acceptable under §15.1, but state it, or compare the realised MDE
   (3.71·SE_realised) with the observed |Δ|.
3. **🟡 W-MXR-3 (low):** `pools_x_v1.json` and the §3 pool table must be regenerated after C-MXR-1 (and W-MXR-1).
   The X-U NLA count per fold (2,209) will drop by up to 975 (the NLA-2022 rows not already removed as frozen-test rows).

### Gate
**Development may NOT start yet.** C-MXR-1 is a real test-label leak path into development. C-MXR-2 means the
C-MX-1 protection exists only on paper. Both are small code fixes.
- Once both are fixed, the W-MXR-1 ruling is recorded, the guards have unit tests that go through the real
  loaders, the pools are recomputed and there is a new snapshot, a short delta check of those items suffices.
- No other item from the first review is open.

---

## M5-X — final confirmation (2026-11-09, v3 / snapshot `1a0c7f4b8331`)

**Mode:** pre-training confirmation. No fitting and no target values read. Run in env `sentinel2`, Python 3.11.17,
with ≤ 6 threads. M2's cache was not written.

### Verified by execution
- **Snapshot.** All files in `1a0c7f4b8331` (supersedes `47130a305f69`) match disk. The split parquet is
  unchanged (`18a35ef3…`). The M5 frozen files `cv.py` (`4acd3010…`) and `bioset.py` (`909c8ca3…`) still match
  `bioassess_freeze.json`. `scripts/train/bioassess_cv.py`, which was edited, is not in the M5 freeze record.
- **Tests:** `tests/test_bioassess_models.py` passes **40/40**.
- **C-MXR-1 closed.**
  - `training_rows` drops the test and excluded rows of **both** blocks.
  - `validate_training_pools` has a cross-block check and returns `[]`.
  - An independent recount found **0** protected or frozen ids in any of the 28 fold pools.
  - X-U pools now hold 1,597 NLA rows (was 2,209); X-L pools hold 1,310–1,503.
  - `pools_x_v1.json` (`per_fold`) matches a fresh recomputation 28/28.
- **M2 ruling applied.** `frozen_test_keys()` now gives NRSA 3,866, UK 67,279, Fuse 2,211, D2n 149 and
  **M2 102**. The 102 matches my independent count (218 same-event visits, of which 102 were not already
  covered). The roles come from M2's own `split.assign` with `use_cache=False`. The NaN-cast warnings come from
  the non-lake pour-point branch; none of the NARS rows has a missing coordinate.
- **C-MXR-2 closed for the code that exists.**
  - `xtrack.load_pretrained_encoder` refuses `"auto"`, the M5 pointer, the M5 encoder `91453a22…`, checkpoints
    with no id record, and any id outside the fold pool (subset check). The tests exercise all of these through
    the real loader, and a clean per-fold checkpoint loads through `cv.PretrainedEncoder`.
  - Importing `xtrack` sets `SENTINEL_TRACK=M5-X`, and `bioassess_cv.resolve_pretrained` then exits (tested).
  - `assert_fit_ids` refuses test and excluded rows of either block, and frozen-test rows (tested).
- **Decision table.** Row 4 is now "CI includes 0 ∧ realised Holm-worst MDE (3.71·CI width/3.92) > 0.02
  pre-declared SESOI". It is well defined, and it was declared before any M5-X result. With the pre-registered
  MDEs (0.023–0.036), most X-L nulls will read "Inconclusive". That is the honest outcome.

### Not verifiable yet (no M5-X fit entry point exists)
- **"Every fit entry point calls `assert_fit_ids`"** holds only because no M5-X fit or pretraining script exists
  yet. `grep` finds no caller outside the tests.
- **"No other code path can load a pretrained encoder"** is not true at module level.
  - `cv.PretrainedEncoder(path)` (`cv.py:73`, `torch.load`) can be called directly without a guard.
  - `bioset.fit(..., init_encoder=…)` (`bioset.py:355,391`) accepts any state dict.
  - Both files are M5-frozen and must not be edited.
  - Today no M5-X code calls either one except through `load_pretrained_encoder`.

### Condition (🟡 W-MXF-1, must hold before the first fit is *run*)
Add a static (AST) unit test over every M5-X module and script: anything that imports `xtrack` or names
`bioassess_x` in `scripts/`. The test asserts that:
- (a) each fit, pretrain or score entry point calls `xtrack.assert_fit_ids` on its fit, validation and pretraining
  ids before fitting;
- (b) none of them references `PretrainedEncoder`, `torch.load`, `resolve_pretrained`, `uk_pretrain_latest` or
  `init_encoder=` except through `xtrack.load_pretrained_encoder`;
- (c) per-fold pretraining writes its sidecar with `write_train_ids_sidecar` from the ids it actually used.

Re-run it whenever a new M5-X script is added, and record it in the next snapshot. The pre-test review will check
it.

### Gate
**Development may start:** writing code, baselines, the bio.infer/GJAM fidelity checks and development CV on
training-pool rows only.
- No critical issue is open.
- **W-MXF-1** must be in place before the first M5-X fit or pretraining run.
- The freeze still requires: the freeze-candidate record (including `pools_x_v1.json` and the frozen split files);
  an independent pre-test review; the lead's freeze and authorisation; and a human Scopus/Scholar check before any
  SOTA wording.
