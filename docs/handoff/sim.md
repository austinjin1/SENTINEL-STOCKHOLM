# Handoff — SENTINEL-Sim workstream (Goal 3), paused 2026-10-25

Paused on user instruction. Nothing was killed. Nothing was committed. **No Sim result has been registered yet.**

## Running processes (left running)
| PID | What | Output |
|---|---|---|
| 1764019 | `scripts/eval/sim_causal_step1.py --n-sites 20 --workers 6 --boot 50` — causal ladder step 1, simulation recovery on DECLARED SYNTHETIC data. Development run, no `--register`. 6 CPU workers. | `$SENTINEL_RUNS/sim/causal_synthetic/` (`synthetic_site*.parquet`, `graphs_bagged_site*.npz`, `step1_results.json`, manifest `sim_causal_synthetic` with synthetic=True); log `$SENTINEL_RUNS/sim/logs/causal_step1_dev.log` |
| 1770249 | `sentinel.core.gpu --name sim_base_s0` wrapper (SENTINEL_GPUS=4), waiting for the GPU-4 lock (held by microbe_seqset, pid 1532162); then runs `scripts/train/sim_train.py --run-name base_s0 --seed 0` | `$SENTINEL_RUNS/sim/base_s0/{model.pt,history.json,sim_eval.npz}`; log `$SENTINEL_RUNS/sim/logs/base_s0.log`; TrainingRun entry (workstream `sim`) |

Background waiter shells were also started (they only wait/print; harmless).
The queue script `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/sim_queue.sh` was written but **never launched**.

## When they finish
1. **Step 1:** read `step1_results.json` → `summary.gate_pass`. The gate needs Bagged-PCMCI+ recall ≥ 0.8 and FPR ≤ 0.05.
   - If it passed: re-run with `--register` (or register from the json).
   - If it failed: log the result, change the configuration (protocol §7 step 1) and re-run before any real-data causal run.
2. **base_s0:** check `base_s0.log` for epoch losses and NaNs. Then queue the remaining runs on GPU 4, one job at a time, in this order (e.g. `sim_queue.sh "base_s1:--seed 1" "base_s2:--seed 2" "dsst_s0:--dsst --seed 0" …`):
   - base_s1, base_s2;
   - dsst_s0–2 (`--dsst`);
   - dsstctl_s0–2 (`--dsst-control`).
3. **Evaluate on the validation cell** (val-role sites × WY2022–23):
   - `scripts/eval/sim_hindcast.py --runs base_s0 base_s1 base_s2 --tag base --register`
   - `scripts/eval/sim_checks.py --runs … --tag base --register`
   - `scripts/eval/sim_dsst.py --dsst-runs … --control-runs … --tag dsst --register`
   - Also score each seed alone and report the per-seed CRPSS range (protocol §4.5).

## Next planned steps (after resuming)
- **Pending addendum review** of `sentinel/analysis/**` and `scripts/eval/sim_{causal_step1,causal_real,thresholds,e1,e3}.py`. A reviewer agent was launched; its result was not yet received. Re-run it if it is lost.
- Fix its criticals and append "Sim — analysis addendum" to `REVIEW_REPORT_PRE_TRAINING.md` BEFORE any of the real-data runs below.
- Run the real-data analyses one CPU-heavy job at a time, each with ≤ 6 threads:
  - `sim_causal_real.py --phase controls`, then `--phase discovery`, then `--phase summarise --register`;
  - step 4 (`sentinel/analysis/causal/step4.py`; no CLI yet);
  - `sim_thresholds.py --register`;
  - `sim_e1.py --register`, then `sim_e3.py --register`.
- Write the model card `docs/model_cards/sim.md` (not written yet). Use the registry model names from `sim_hindcast.py` (`SENTINEL-Sim v1 (<tag>, N-seed ensemble):sim_ec_site`).
- Iterate on validation only (§15.5 amendment). Then write `FREEZE_sim.json` (split sha, `guard.frozen_code_sha256()`, weights sha per seed). Score Test-A once (`--allow-test`, not yet implemented in the eval scripts). Then the post-results review.

## Key files
- **Protocol:** `docs/preregistration/sim_protocol.md` (v1.2). Frozen cal list: `docs/preregistration/sim_cal_sites.json`.
- **Novelty checks:** `docs/novelty_checks/sim_*_2026-10-25.md`.
- **Code:** `sentinel/models/sim/*`, `sentinel/analysis/{causal,insights}/*`, `sentinel/analysis/thresholds.py`.
- **Tests:** `tests/test_sim.py` (22 pass), `tests/test_sim_analysis.py` (8 pass).
- **Data cache:** `$SENTINEL_RUNS/sim/cache/simdata_validation_*.pkl` (~0.6 GB; the cache key changes whenever the data code changes).

## Open issues
- The step-1 smoke run (2 sites, 2 bootstraps) gave recall 0.77 and FPR 0.13, so the gate may fail.
- LPCMCI rejects our link assumptions, so it runs unconstrained.
- J-PCMCI+ ran in the smoke run, but its use is unreviewed.
- **Thresholds:** the Appalachia (L3 69/70) NRSA positive control has only 102 eligible samples. A dev check gave HC05 ≈ 790 µS/cm (20 bootstraps), so a likely non-replication. The national bootstrap CI excluded its own point estimate (only 5 bootstrap replicates); recheck with B = 1000.
- **Validation cells are small:** val DO has 28 sites in 9 groups; Powell val has 21 sites in 5 groups.
- **Interfaces we depend on:** Sim imports M6 `hydromet/physics.do_sat_mgl` and D8 `riverpulse.load_labels` without editing them. Tell those workstreams before changing either signature.
- Turbidity–discharge hysteresis class and the forecastability score are not implemented (`derived.py`).

## Addendum (lead, 2026-10-25): analysis-code review verdict — FAIL (arrived after the pause)
No real-data causal / conductivity-threshold / E1 / E3 run until these are fixed or declared in the protocol:
1. FDR correction never reaches the graphs (links accepted at uncorrected alpha 0.01).
2. Time-reversal negative control N2 invalid (fired 5/5 seeds on a correct model) — redefine (forward vs reversed) or drop from pooled FPR.
3. Distant-weather control N3 runs constrained; protocol says unconstrained → `constrained=False` in scripts/eval/sim_causal_real.py.
4. Conductivity-threshold bootstrap treats revisits as sites (4,633 vs 1,928 unique_id); TITAN bootstrap resamples samples not sites.
5. Appalachian 300 µS/cm control not testable (102 samples, 3 reference samples; filters silently skipped) — pre-declared minimum-data rule or "underpowered" label.
Disclosure: the reviewer computed Appalachian HC05 ≈ 787 µS/cm (95% CI 617–964) on real data; any later change to that stratum is outcome-informed and must be logged in sim_protocol.md §12.
Warnings: step 4 + hourly controls missing; causal script lacks TrainingRun and step-1 gate check; TITAN vs TITAN2 differences; E3 misses some Hare et al. rules and uses an undeclared dam-storage proxy; E1 errors ignore gaps between summers; some registry split labels not in protocol; E1/E3 scripts lack thread limits.
Passed: no leakage, no test-role sites/Test-A dates, synthetic data kept separate, synthetic ground truth correct (11 links).
To do on resume: append "Sim — analysis addendum" to REVIEW_REPORT_PRE_TRAINING.md and fix 1–5 first.

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.


## Note from M1 (2026-11-10, lead ruling): reserved USGS new-site block — shared exclusion helper
- 729 USGS sites (`docs/preregistration/sonde_newsites_list.json`, sha `6200a77c…`) are RESERVED as the EW-v3-N / M1-N
  unseen test block. Never train, tune, select, plot or compute target statistics on them.
- Copies exist in the shared tiers `processed/sonde_daily` (422 files), `sonde_hourly` (27) and `raw/usgs_iv{,_refresh_20261108}`.
  - The 27 + 21 raw IV directories are now **mode 000** (lock record `raw/usgs_iv*/RESERVED_NEWSITES_LOCK.jsonl`).
  - `sonde_daily` / `sonde_hourly` are untouched: frozen scorers content-hash them.
- Shared helper `sentinel/data/reserved.py`:
  - `exclude_reserved` (drop) and `assert_no_reserved` (refuse), hash-pinned list;
  - `assert_preclaim_clean` (pre-claim audit).
- Tests: `tests/test_reserved_loaders.py`.
- **Edited by M1 (no frozen file; verified against every FREEZE_*/pin record):** `sentinel/models/sim/data.py` (`build`: `assert_no_reserved(cand)`; split-gated, so no behaviour change) and `sentinel/analysis/causal/diel.py` (`has_hourly` returns False for reserved sites). `sim_causal_real.py` / `sim_e1` / `sim_e3` / `causal/ladder.py` load sites through `sim.data.build` (split-gated) and were not edited. Any new Sim loader that reads the sonde tiers by index or glob must call the helper.
