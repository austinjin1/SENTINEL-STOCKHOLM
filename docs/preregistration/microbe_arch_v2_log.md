# M3 architecture round v2 — execution log

Execution records only: what ran, when, registry row ids, counts (lead ruling OQ11). Nothing here fixes a design
choice; those go in `microbe_arch_v2_deviations_NN.md`. Times are UTC from `date`. Entries are appended, never
rewritten.

## 2026-11-10

### E1 — stamp
- Lead stamp of the amendment and rulings: registry row `421854a9d225`, 19:41:37 UTC. Released on `lodo` only;
  D2, the EA family, H4 and GAS-3 closed.

### E2 — Step 0 (code and tests only; no development number)
- 23:27 UTC: `pytest tests/test_microbe.py tests/test_microbe_testlook.py tests/test_microbe_niche.py
  tests/test_microbe_archv2.py tests/test_microbe_wq.py` → 130 passed, 0 failed, 0 skipped (first full pass
  20:27 UTC; rerun after the slot-#8 gate change asked by the lead, see E5).
- Two defects found by the new tests and fixed before any run: the feature cache key of `v4mr.features` did not
  cover the QMP table or the samples table; the fold cache key of `niche._fold` did not cover the held-out
  context. No run had used either cache.
- `scripts/data/microbe_geo_counts.py` retired: mode 0444, sha256 unchanged
  (`692744722be6f0317d826d96bd445287a278ee842e2295a0c9164a731092f987`), not run again.
- `processed/microbe_wq_geo/protected_labels.parquet`: mode 0440. `counts.json` left as built (lead, 23:30 UTC).
- Deviations file 01 sent to the lead for review and stamp: sha256
  `a3fdc1ed74468123c96e744b88a25958762487e2f5106178b50d6f9a96245a58`. Not stamped yet; its five keys are closed.

### E3 — record owed under ruling L7 (a)
- Grouped EA runs with ENA `collection_date` on or after 2023-10-01: **98** (50 in EA-dev, 48 in EA-test).
  Counted from `microbe_arch_v2_evidence/ena_PRJEB90117_samples.tsv` and `ea_partition.json` (titles matching
  the partition script's pattern; recount 23:28 UTC, same numbers).
- These runs stay in M3's EA partition under lead ruling 4b (`LEAD_DECISIONS_20261110.md`): M3's EA track uses
  sample-based biofilm data with its own partition; M6 reserves Hydrology-API sonde data; the biofilm chemistry
  is outside M6's sonde reservation; no overlap.

### E4 — record owed under ruling L7 (b): source of "1,620 of 1,649 linked by id" (`microbe_ea_protocol.md` §2.5)
- Source: one command run by M3 on 2026-11-10 at 15:41:08 UTC, before the rulings file existed. It streamed the
  authors' file `Example_data/filtered_metadata_normal.csv` (GitHub `Micro-Biology/Warren_et_al_-_microbial_indicators`,
  `HEAD`) from `curl` into a Python process run with `-I`. The file was not written to disk.
- The process kept the first three fields of each row (`ANON_ID`, year, date), printed the row count (1,620), the
  first four (id, year, date) triples, and the number of `ANON_ID` values found among the 1,649 ENA
  `sample_title` values (1,620 of 1,620).
- **Disclosure.** That file also holds the label columns (3-month-mean TON / nitrate). Its bytes passed through
  the process once. No label value was printed, stored, summarised or used, then or since. This was a read of
  the label file's id columns before ruling L4 made the header check the only exception; since the rulings no
  EA label file has been downloaded or opened.
- "1,649" is the run count of the ENA sample report (`n_runs` in `ea_partition.json`).

### E5 — slot #8 gate form (agreed with the lead, 23:30 UTC)
- The lead writes the D1-gate evaluation as a registry row: experiment `microbe/d1_gate_evaluation`,
  `extra.d1_gate` = "pass" or "fail", `extra.written_by` = "lead". `--slot8-row` requires all three (tested).

### E6 — header check of `water_chem.csv` (ruling L4)
- Script written, **not run**: `scripts/data/microbe_ea_header_check.py`. It waits for the lead's written
  approval of its sha256. Licence row for Zenodo record 18668846 `water_chem.csv` added to `docs/licenses.md`
  (CC BY 4.0, column names only).

### E7 — correction to E2 (times)
- The full five-file test run after the slot-#8 gate change ran 23:29:21–23:31:57 UTC (130 passed), not at 23:27.
  Two tests of the GEO power script were added afterwards (E9); the full set is rerun before the D1 run (see E11).

### E8 — header check run (ruling L4)
- Lead's written approval of sha256 `90afcfd9b4f8cb7298c770161e9f4adbf9b0c0b4f6e494450d50e9f2a48c0705` received.
  Run once at 23:33:24 UTC as `python -I -B scripts/data/microbe_ea_header_check.py`; sha verified immediately
  before; exit code 0.
- Output `microbe_arch_v2_evidence/ea_water_chem_header.json`, sha256
  `920f90fe7962e4cc9e58ea1ea48c1ea74c0a5160d4b706ba66d576843c2cdb03`, 19 column names: one empty name (first
  column), then `Alkalinity_combined`, `Alkalinity_grans`, `Alkalinity`, `Ammonia_N_unionised`, `Ammoniacal_N`,
  `DOC`, `Chloride`, `Conductivity`, `Nitrate_N`, `Nitrite_N`, `TN_oxidised`, `Orthophosphate`, `DO`, `DO_sat`,
  `TP`, `SiO2`, `Water_temp`, `pH`. Script and output set to mode 0444. No data row was kept.
- Lead (same message): the E4 disclosure is repeated in the EA-family deviations file when that file is written;
  no stream of a file that carries label columns from now on.

### E9 — GEO power script (code only; key `GEO_POWER` closed)
- `scripts/eval/microbe_geo_power.py` written with two tests. Its SE reproduces the eight `paired_se_site` and
  MDD values of `microbe_split_lodo_x001_report.json` exactly (largest absolute difference 0) from the
  existing-track family-P development out-of-fold files, which are the files that report was made from. The
  reference method is the 95% percentile-interval width of the paired site bootstrap divided by 2·z₀.₉₇₅.
  Not run on family G: it refuses until deviations file 01 is stamped.

### E10 — first development run of the round; file hashes at that moment (23:34:47 UTC)
- First run: tier-0 baselines on the GEO family (`global_mean`, `context_ridge`, `context_rf`, then
  `--tier0-best`), `lodo`, family G, CPU, started 23:34:3x UTC. They run first because `context_rf` is the
  MSE-skill reference of every later family-G score.
- sha256 at that moment:
  - `scripts/train/microbe_arch_v2_cv.py` 30b291305908b816b7cf842b47ad403525e64d7fd50a2b2ec85e59ad36d54dca
  - `scripts/train/microbe_cv.py` bc3e974e1644ccdd5d755917aebe280c3da33af0fc97cba7b36fa737d0e8b381
  - `sentinel/models/microbe/archv2.py` a60da848d325b137ac3ba04b9c8c649006d8c674bdf77e05b7a6bc7213b471fb
  - `sentinel/models/microbe/niche.py` 055ad4dab8d74e5b9e15d6dfe2fe3a077ec7fb510354407f2ccdee7b01c1b762
  - `sentinel/models/microbe/v4mr.py` 3638553f6ef813880286066999498120144ff6bf8b643a9c5fbcbae519777891
  - `sentinel/models/microbe/wa.py` 77ebca08614f344e61751f71281bc038b881df2d83770eef3119e1a3ac1dd59c
  - `sentinel/models/microbe/baselines.py` 8279d7d0b7fe8588662fa7036512c6c8233bf0230a573b75f919f5ec1820da04
  - `sentinel/models/microbe/stamp.py` 279783d42d048eabc91995fb1fea43b85419d31654ee17cae926529a46298c6f
  - `sentinel/models/microbe/design.py` 3902fa796b69f7bc8ce526c5f90d2288f782820e6d302161cead8c2fdf768109
  - `sentinel/models/microbe/splits.py` 4ffa0ee9c10d4bbf219a6a00f72a829215c1baee8934c1e027df46fe58bc7ec7
  - `sentinel/models/microbe/evaluate.py` 701a97127823c5ddf50acdded952fda013d2959890258d961eb3f02b3613c295
  - `sentinel/models/microbe/features.py` 22200785ccdc9103c3570b5423f88dcd41537280ae8de479f09ea825f1741365
  - `sentinel/models/microbe/cv.py` 5c34c33aa9da0f8e93e85be8cd28bfa88a9f85797704949cf1ca2cd3bb92ede4
  - `sentinel/data/microbe_geo.py` 4140ac21951879c0d5f6fc84bb98d0190a9266e06c911e88996a94456719bea3
  - `scripts/eval/microbe_qmp_confound.py` 28cd176182a19a261cd581b10c3becde741e1f8050b8db6d208707cb154d3fd4
  - amendment 022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a; rulings
    e148a09def685f6fa730650e24ba384389daa1fe44d29c07151568da028ecd95

### E11 — full M3 test set rerun (23:35:30–23:38:28 UTC)
- 132 passed (five M3 test files; `tests/test_microbe_archv2.py` sha256
  7c6c25038e6764ee332237515aa88d1424b603ec5fb20626b13e61cddad34629, 61 tests). Code changes after this entry
  (E15 onwards) get their own full rerun before the first D1 development run.

### E12 — deviations file 01 stamped
- Lead stamp: registry row `ccbf6d0e2fb5` (experiment `microbe/prereg_snapshot`), 2026-11-10T23:39:24Z, for
  `microbe_arch_v2_deviations_01.md` (sha256 a3fdc1ed…5a58). The amendment's stamp stays row `421854a9d225`
  (19:41:37Z). `microbe_arch_v2_dev01_lead_rulings.md` (sha256
  25b00774eba005c2bd73abb6fd838f886d3cbd531ec09925db95ff71acbcb86c) wins where it differs from deviations 01;
  read in full. Released on `lodo`: `niche_s_a0`, `xgb_v4clr_mo`, the calibration diagnostic.
- No M3 code compares a stamp row id with a fixed value.

### E13 — D1 train-only smoke test (amendment §2.5); D1 held
- `sentinel_runs/microbe/cv_archv2/niche_s_smoke_train_only.json` (fold 1 training events only; no held-out row,
  no metric): convergence rule passed (largest first-step failure share 0.008576 ≤ 0.05); 66,006 units; fit
  0.0135 CPU-h; inversion of 1,166 training rows 0.081–0.092 CPU-h per N_eff value; projected 9.105 CPU-h for
  the run that serves both families, against 6 CPU-h in §2.5 and in the runner.
- D1 held; two questions sent to the lead (reading of "per track"; fold shards). No development number of
  `niche_s` exists.

### E14 — lead ruling of 2026-11-10 on the `niche_s` CPU budget (§2.5) and fold shards (received 23:44 UTC)
- A. Budget: the stricter reading. A track is a development scheme (§1.2, §2.2: `lodo`, `lodo_x001`). One
  `niche_s` run on one scheme has 6 CPU-h, whatever number of target families it serves; `lodo_x001` later has its
  own 6 CPU-h. The runner's 6 CPU-h constant stays.
- Order, all train-only, each logged before the next: (1) finish the 1–2 BLAS-thread timing; thread count is not a
  model setting; fix one value and log it. (2) A pure implementation speed-up (same objective, solver, stopping
  rule, vocabulary) is allowed before the first development run; verify it on the train-only smoke output and log
  the largest absolute difference of the posterior modes against the current smoke file, and that the first-step
  failure share is unchanged. (3) If the projected CPU time of the full-vocabulary `lodo` run is then ≤ 6 CPU-h:
  run as specified, full vocabulary. (4) If still > 6 CPU-h: apply the pre-declared restriction (units in ≥ 3
  training events), declared in the log before the run; repeat the train-only smoke test on the restricted
  vocabulary (convergence rule ≤ 5% and projection ≤ 6 CPU-h must hold); then run; same configuration, because no
  development number has been read. The choice between 3 and 4 is made from the train-only timing alone; not both.
- B. Fold shards approved: one process per outer fold or group of folds, same code path per fold, the thread
  count fixed in step 1, at most 6 threads in total across the `niche_s` shards running at one time. A shard
  writes predictions and its own CPU time only; it computes, prints and logs no metric. One merge-and-score step,
  once, after all 19 folds are present; it sums the CPU time over the shards and refuses to score if the sum is
  > 6 CPU-h (then step 4 applies; the rerun is the same configuration). A unit test on synthetic data shows that
  sharded output equals unsharded output (exact equality of the merged prediction table) before the run. The same
  pattern may be used for `niche_s_a0` and the other inversion-based entries, each with its own budget.
- Owed: the notice when the first real D1 shard is live (PIDs, thread count, full or restricted vocabulary).

### E15 — ruling E14 step 1: thread count of the inversion-based models fixed at 1 (logged 00:10 UTC, 2026-11-11)
- Train-only timing (fold 1 training events, 292 of them, N_eff 100, code at niche.py sha256 055ad4da…c1b762):
  1 BLAS thread 58.55 CPU-s / 58.50 s wall; 2 threads 65.47 / 57.26; 6 threads 91.22 / 56.63. Largest absolute
  difference of the posterior modes between 1 and 6 threads: 7.52e-8. No held-out row, no label comparison.
- Fixed: 1 thread for every inversion-based model (`niche_s`, its variants, `xgb_nichez_context`), sharded or
  not (`NICHE_THREADS = 1` in `scripts/train/microbe_arch_v2_cv.py`). Other models keep 6.

### E16 — ruling E14 steps 2 and 3: implementation speed-up verified; full vocabulary (logged 00:10 UTC)
- Change (`sentinel/models/microbe/niche.py`, sha256 055ad4da…c1b762 → 32d01a52205836827f3ba645c115ac284fca25596
  c1c66bed31084e650a0f07c): the per-event Hessian of the inversion is assembled for a chunk of events at once from
  precomputed second-moment features of the units (`moment_features`, `cov_from_moments`) instead of one event at
  a time (`cov_rowwise`, kept for the unit test). Objective, solver, step rule, stopping rule and vocabulary
  unchanged. Unit test: batched = row-wise Hessian on synthetic data (`tests/test_microbe_niche.py`).
- Verification, train-only (fold 1 = domain D01, 1,166 training events, 66,006 units, 1 thread, N_eff 10 / 30 /
  100 / 300 / 1000; old code = copy at sha 055ad4da…, new = repository file). No held-out row; no prediction
  compared with a label.
  - Largest absolute difference of the posterior modes (17 dimensions; SD of the modes 0.54–0.71):
    7.82e-8 / 1.28e-7 / 1.80e-7 / 5.15e-8 / 3.47e-8. Largest over all: 1.80e-7. No event differs by more than 1e-6.
  - First-step failures: 10 / 1 / 2 / 1 / 8 of 1,166 with both codes, the same events (shares 0.008576 / 0.000858 /
    0.001715 / 0.000858 / 0.006861, as in the smoke file of E13). Exact-Hessian flag identical event by event.
  - Number of Newton steps differs for 224 / 94 / 11 / 8 / 6 events (round-off at the stopping test); their mode
    differences are within the figures above.
  - Largest absolute difference of the posterior SDs: 2.04e-7.
- CPU time at 1 thread: inversion of the 1,166 events 207–222 CPU-s per N_eff with the old code, 71–79 CPU-s with
  the new; fit 23.4 / 27.3 CPU-s. Projection for the full-vocabulary `lodo` run (formula of the smoke test):
  old code 6.195 CPU-h, new code 2.567 CPU-h.
- Step 3 applies (2.567 ≤ 6 CPU-h): `niche_s` runs as specified, full vocabulary. Step 4 (restricted vocabulary)
  is not used and is not run.
- Evidence files: `sentinel_runs/microbe/cv_archv2/niche_s_speedup_check/` (modes of both codes, the two scripts,
  the old module).

### E17 — family-G tier-0 tables finished; TabPFN comparators started (logged 00:10 UTC)
- `cv_archv2/lodo/G`: `global_mean`, `context_ridge`, `context_rf` (seeds 0–2), `tier0_best` finished
  23:34–23:48 UTC, 2026-11-10 (registry rows of experiment `microbe_arch_v2_cv`).
- Started through the GPU lock helper, settings and seeds as specified: `tabpfn_v4clr` family P (GPU 1, 23:49:49
  UTC), `tabpfn_v4clr` family G (GPU 3, 23:50:12 UTC), `tabpfn_clr` family G (GPU 4, 23:50:36 UTC). GPU 2 not
  used.

### E18 — code changes after E11 (dev01 rulings D1–D8, D11; fold shards of E14); full test rerun (logged 00:20 UTC, 2026-11-11)
- Changes, all before any `niche_s` development number; none alters a prediction of a model that has a development
  number (tier-0 family-G tables; the three TabPFN runs of E17 were started before these edits and use the runner as
  loaded at their start):
  - `sentinel/models/microbe/stamp.py` (D11): only registry rows with `extra.written_by` = "lead" count as a stamp
    of the amendment or of a deviations file. On the real registry: amendment stamped (rows `421854a9d225`,
    `ccbf6d0e2fb5` both carry the mark), keys A0, XGB_V4CLR_MO, CALIBRATION, NICHE_S_QMP, GEO_POWER released by
    `ccbf6d0e2fb5`; D2, EA, H4, GAS not released.
  - `sentinel/models/microbe/archv2.py` (D1, D8): `gate` applies the release key and the slot-#8 row check of
    `<base>` to `<base>_qmp`; `niche_s_qmp` needs the id of the single registry row of experiment
    `microbe/g10_choice` whose `extra.g10_entry` is G1.
  - `scripts/train/microbe_arch_v2_cv.py`: `NICHE_THREADS = 1` (E15); `niche_s_qmp` in COUNTED (D8); `--g10-row`;
    `--shard-folds` (predictions, label-free diagnostics and CPU time of the listed outer folds; no metric, no
    registry row except `run_failed`) and `--merge-shards` (all folds present, one configuration and one code
    state, CPU time summed over the shards; over budget → `run_failed` row, no prediction file read, no metric).
    Each fold goes through `archv2.oof`, the code path of the unsharded run.
  - `scripts/eval/microbe_qmp_confound.py` (D5–D7): a non-finite laboratory share fires the rule; the script
    refuses if its output exists; the output is created exclusively. Not run yet.
  - `scripts/eval/microbe_geo_power.py` (D2–D4): pairs = each finished entry against every §7.4 comparator (plus
    the two context comparators for G1 and G8); SE from the 95% interval width; scaling sqrt(1,215/214) for every
    target; `--not-run ENTRY=REF`. Not run.
- Tests added (`tests/test_microbe_archv2.py`, 61 → 74 tests): lead mark on stamp rows; D1 twins
  (`xgb_v4clr_mo_qmp`, `xgb_v4mr_stack_qmp`); D8 G10-choice row; D6 non-finite share (three cases) and D7 run-once;
  D2–D4 power file (24 pairs with G3 reported as not run, 29 with all five entries); shards: the merged prediction
  tables of `niche_s` and `xgb_nichez_context` (both families, seed files included) equal the unsharded tables bit
  for bit on the synthetic design, scores and registry values equal, a shard writes no registry row and no score
  file, the merge refuses a missing fold, another configuration, another code state and a CPU sum over the budget.
- Full M3 test set 00:14:01–00:20:21 UTC: 146 passed, 0 failed, 0 skipped (five files; 1 thread).
- sha256 now:
  - `sentinel/models/microbe/niche.py` 32d01a52205836827f3ba645c115ac284fca25596c1c66bed31084e650a0f07c
  - `sentinel/models/microbe/stamp.py` 505343e35abf54d25a241af8d9fe15931c8d63493ce76b58fd1c95bcdd62bd61
  - `sentinel/models/microbe/archv2.py` a92bbf5529795592ca36f54a2520a2d089f72e1c89c53fce87edf9ce649893ce
  - `sentinel/models/microbe/v4mr.py` 3638553f6ef813880286066999498120144ff6bf8b643a9c5fbcbae519777891 (unchanged)
  - `scripts/train/microbe_arch_v2_cv.py` eef84f4a002360805ebe90d6432345ce877f4ed64134008e0462c83a97b4a8db
  - `scripts/eval/microbe_qmp_confound.py` 26c1dae5b8bd62015102df33e5ac7ac6d050576aa92fb925b7bd58d76d47cc04
  - `scripts/eval/microbe_geo_power.py` 2eeabb69d5cc63df9d3db0d8b2ef3d05e42dd2d86c66a3ad6354c8e37c5719fc
  - `tests/test_microbe_archv2.py` d48b4280fc7ff24d3ff40f7bef58f6bf7abb9d6eb907dc11b4a49941e568613a
  - `tests/test_microbe_niche.py` 5f4de56342c0ea070fa659b591fa2baed6cc67c30c9ac57d8261b54671c44edd

### E19 — QMP confound check run once (00:21:14–00:21:17 UTC, 2026-11-11); D1 shards queued (logged 00:21 UTC)
- `scripts/eval/microbe_qmp_confound.py` (sha256 26c1dae5…cc04), after the D1/D6/D7/D11 changes passed their tests
  (E18). Label-free: `qmp_features.parquet`, `qmp_meta.parquet` and ids only. Output
  `sentinel_runs/microbe/cv_archv2/qmp_confound.json` (sha256
  a338b58c60614efea40c0aaa944b8dd0f443547b048363c377725abb43dfad19); 16 registry rows of experiment
  `microbe_arch_v2_qmp_confound` (last id `01ea6ca3de74`). The first output stands.
- Share of the variance of `qmp_load` explained by the qPCR laboratory (2 laboratories): development events
  0.2102 (n = 1,159), all four blocks pooled 0.3881 (n = 1,966). Both finite and ≤ 0.5: the rule has NOT fired;
  +QMP results are not restricted to descriptive by this rule.
- Reported with every +QMP result, as §5 requires: the qPCR batch explains 0.4892 (development, 153 batches) and
  0.7434 (all blocks, 251 batches) of the variance of `qmp_load`; laboratory explains 0.2818 / 0.4841 of
  `qmp_log_copies`, 0.0327 / 0.0934 of `qmp_log_dna`, 0.0082 / 0.0011 of `qmp_log_volume`.
- D1 fold shards (`niche_s` + `xgb_nichez_context`, 6 processes at 1 thread, folds 0-6-12-18 / 1-7-13 / 2-8-14 /
  3-9-15 / 4-10-16 / 5-11-17, full vocabulary) are queued behind the load rule of the lead's 23:45 UTC instruction
  (start when the 1-minute load is ≤ 40; it is 50–57 from other workstreams' jobs). Launcher
  `sentinel_runs/microbe/logs/archv2_niche_shards.sh`, status file `archv2_d1_shards.status`.

### E20 — lead rulings A–C (received about 00:24 UTC, 2026-11-11); first D1 shards live 00:25:35 UTC (logged 00:29 UTC)
- The lead accepted E15–E19 as reported and ruled on three questions of M3.
- **A (load figure).** The six one-thread D1 shards start regardless of the 1-minute load: the "≤ 40" figure is a
  hygiene figure of the launcher, not a stamped rule; rule 5 of the round says ≤ 6 threads per job and CPU jobs
  sequential per workstream, and six one-thread shards are one 6-thread job. Conditions: (1) the 1-minute load and
  the time are logged at launch and at the end; (2) the D1 shard set is the only CPU job of M3 while it runs
  (TabPFN on GPUs continues); (3) CPU time is the sum over shards, measured as before; if a stamped CPU cap is
  reached M3 stops and reports — no restart and no setting change without a ruling.
- **B (long comparators).** The fold-shard pattern may be used for `rf_clr_tuned`, `xgb_clr_tuned`,
  `xgb_clr_stack`, `xgb_v4mr` and `xgb_v4clr_3000`, under the E14 conditions without exception: same code path per
  fold; every seed and every inner tuning split a function of the fold only, never of the shard or of the order;
  shards write predictions and CPU time only and compute no score; one merge-and-score step; the synthetic
  bit-for-bit equality test (sharded = unsharded tables) passes for EACH of these models before its real run;
  ≤ 6 threads per model in total; one model at a time; CPU time = sum over shards, counted against that model's
  stamped budget; a model stays a counted model (a re-run is refused). The "≤ 2 h" figure of rule 5 is the GPU-lock
  limit; a CPU-only shard that runs longer is not a breach; a shard is restarted only by the stamped rerun rule
  (infrastructure failure, logged).
- **C (`xgb_v4mr` budget).** Train-only timing of `xgb_v4mr` first, after the D1 shards and not beside them; the
  projection goes to the lead; if it is > 6 CPU-h per track `xgb_v4mr` is held until the lead rules. M3 invents no
  fallback.
- The lead noted the QMP confound result of E19 (rule not fired; first output stands; batch share reported with
  every +QMP result). Unchanged: nothing not released is touched; no git commit.
- **Done on ruling A.** The waiting launcher of E19, the load-gated comparator queue A and the block-model timing
  job were stopped at about 00:25 UTC; none had started a process (status file: one `waiting` line only; no
  `_shards` folder existed). The launcher `sentinel_runs/microbe/logs/archv2_niche_shards.sh` got a `MAX_LOAD`
  setting (default 40; `none` starts at once) and writes time and load at the start and at the end. It is not one
  of the three hashed code files; no runner setting and no seed changed.
- **First D1 `niche_s` lodo run on real data live: 2026-11-11T00:25:35Z, 1-minute load at launch 50.88.** Six CPU
  processes, 1 thread each, full vocabulary (E16), `--models niche_s xgb_nichez_context --shard-folds …`:
  PID 2433462 folds 0 6 12 18; 2433463 folds 1 7 13; 2433464 folds 2 8 14; 2433465 folds 3 9 15; 2433466 folds
  4 10 16; 2433467 folds 5 11 17. All six alive at 00:28:12Z at 0.93–0.96 of one core each (1-minute load then
  61.68). Status file `sentinel_runs/microbe/logs/archv2_d1_shards.status`, logs `archv2_d1_shard{0..5}.log`.
  Notice sent to the lead at 00:28 UTC.
- Code state the merge will require (unchanged since E18; not edited until both merges are done): `niche.py`
  32d01a52205836827f3ba645c115ac284fca25596c1c66bed31084e650a0f07c, `archv2.py`
  a92bbf5529795592ca36f54a2520a2d089f72e1c89c53fce87edf9ce649893ce, `scripts/train/microbe_arch_v2_cv.py`
  eef84f4a002360805ebe90d6432345ce877f4ed64134008e0462c83a97b4a8db.
- M3 jobs running at this entry: the six D1 shards (CPU) and three TabPFN runs on GPUs 1, 3, 4 (`tabpfn_v4clr` on
  P and on G, `tabpfn_clr` on G). Order after the shards: merge `niche_s` and `xgb_nichez_context` once each; the
  `xgb_v4mr` train-only timing; `niche_s_a0` and `niche_s_tax` shards; the comparators, one CPU job at a time.

### E21 — D1 shards finished 00:57:44 UTC; `niche_s` and `xgb_nichez_context` merged and scored once (logged 01:02 UTC, 2026-11-11)
- Shards: all six processes ended with rc 0 at 2026-11-11T00:57:44Z (wall 32 min 9 s); 1-minute load 50.88 at
  launch, 53.78 at the end (it was 61.68 at 00:28 and about 97 at 00:31, from other workstreams' jobs). 19 fold
  files per model, one code state (the E20 hashes), 1 thread, 6 process ids. No restart, no setting change.
- CPU time, summed over the shards: `niche_s` 1.985 CPU-h (largest fold 0.112; budget 6 CPU-h; projection of E16
  2.567); `xgb_nichez_context` 0.304 CPU-h (it reuses the fold fit of `niche_s`).
- Merge-and-score, once each, 2,000 bootstrap draws: `niche_s` 00:57:58–00:59:15 UTC (load 53.94 / 56.61),
  `xgb_nichez_context` 00:59:15–01:00:32 UTC (load 56.61 / 69.24); both rc 0. Logs
  `sentinel_runs/microbe/logs/archv2_merge_{niche_s,xgb_nichez_context}.log`.
- Development LODO, 19 folds, pooled R² (registry experiment `microbe_arch_v2_cv`, split `neon_dev/lodo_cv`):

  | model | family | mean R² | per target | registry rows (first .. last) |
  |---|---|---|---|---|
  | `niche_s` (D1, slot #6) | P | 0.1857 | log_tn 0.079, log_tp 0.096, log_doc 0.272, log_no3 0.145, log_sc 0.566, log_turb 0.034, do 0.228, log_chla 0.066 | 48 rows `b231341e010e` .. `579e6bb5f9a9` |
  | `niche_s` (G1) | G | 0.1989 | ph 0.381, log_cl 0.032, log_so4 −0.010, log_ca 0.228, log_dic 0.714, do_sat 0.009, log_nh4 0.122, log_uv254 0.115 | 48 rows `1d53da190d35` .. `72730265ff70` |
  | `xgb_nichez_context` (control) | P | 0.1392 | log_tn −0.030, log_tp 0.080, log_doc 0.195, log_no3 0.176, log_sc 0.520, log_turb −0.198, do 0.307, log_chla 0.064 | 48 rows `907cd3503cc5` .. `3397218f5250` |
  | `xgb_nichez_context` (control) | G | 0.1775 | ph 0.200, log_cl 0.099, log_so4 0.120, log_ca 0.310, log_dic 0.521, do_sat −0.132, log_nh4 0.171, log_uv254 0.129 | 48 rows `f3047d633814` .. `a052d85dcc90` |

- For orientation only (existing track, family P, same split, registry experiment `microbe_phase1_cv`): mean R²
  0.3250 `ens_ridge_v4clr+xgb_v4clr`, 0.3171 `xgb_v4clr_context`, 0.3143 `xgb_v4clr`, 0.2893 `xgb_clr`, 0.2845
  `ridge_clr`. **On family P the D1 point estimate is well below the existing V4 tables (0.186 against
  0.314–0.325): a negative result on its face.** The paired comparisons of §6 and the D1 gate are not computed
  here; the gate is evaluated once by the lead. Family G has only the tier-0 tables so far (mean R² −0.035
  `tier0_best`); its comparators are not run yet.
- Label-free diagnostics (`lodo/{P,G}/niche_s_diag.json`): the inner choice of N_eff is 10 — the smallest grid
  value — in 17 of 19 folds and 30 in 2; first-step failure share of the held-out inversions 0.0016 pooled
  (0 to 0.027 per fold; kill rule not reached); 62,679–66,643 units per fold (full vocabulary).
- sha256 of the merged tables: P/niche_s 6ad4f768cd23744dd2167f4c1e3d2c3158cccfc1e6ea557511a01ba23d97ed6e;
  G/niche_s a7372c75ae3fad4d50d622582dbda0b1e8c9ea61064bd14f7b4ea88cdb58fca6; P/xgb_nichez_context
  15769ecf4f1f978700c70240c9fbc3b2d8e1ec0617739f0df1b80598be20d01c; G/xgb_nichez_context
  ab066be5dd8628b3d6d5628441a12075edfdb058cc89da99969eba4f971a91bb.
- Configurations counted so far in this round: `niche_s` (slot #6) and `xgb_nichez_context` (control) each have a
  development number now and are never rerun.

### E22 — lead capacity and order ruling; ruling-B shard change in the runner, tests green (logged 01:23 UTC, 2026-11-11)

- **Lead message (received about 01:05 UTC), recorded as given.** (a) The D1 row ids of E21 are received; no gate
  outcome is stated by anyone yet; the lead evaluates the gate once, from registry rows, when everything ruling
  B6 lists is finished on `lodo`. (b) Capacity: rule 5 limits threads per job (at most 6), not the number of
  jobs. M3 may run CPU jobs in parallel with at most 12 threads in total across its CPU jobs at any moment. The
  `xgb_v4mr` one-thread timing run is the exception: it runs with no other M3 CPU job alive and its record states
  the load average. Same settings, same seeds, ruling-B models at 1 thread, bit-for-bit tests green before the
  shard change is used on real rows. (c) Order: (1) ruling-B shard change + five tests; (2) `xgb_v4mr` timing,
  projection to the lead; (3) the gate prerequisites of B6 — `niche_s_a0`, the calibration diagnostic on D1's
  development predictions, `xgb_v4mr`, every unconditional §6 comparator on `lodo` (WA-tol included); (4) then
  `niche_s_tax` and the remaining comparators and tables (family G comparators, tuned, stack, `xgb_v4clr_mo`).
  When (3) is complete M3 sends one message with the registry row ids per model and family and the statement
  "gate prerequisites complete"; the lead logs the evaluation under `microbe/d1_gate_evaluation` and names slot
  #8. Neither slot-#8 arm starts before that. (d) The N_eff-at-grid-minimum diagnostic stays as logged; no grid
  change. (e) The three TabPFN GPU jobs continue; idle GPUs may be used for approved GPU work through the lock
  helper, never GPU 2, at most 2 h per job.
- **Runner change (ruling B), made after both D1 merges and with no shard set in progress.**
  `scripts/train/microbe_arch_v2_cv.py` sha256 eef84f4a…a8db → d85d90a7d32d89a7a5dbbb4e9d0648cada355d452703ce1ce7742dea647e8cbd.
  `--shard-folds` / `--merge-shards` now also accept `rf_clr_tuned`, `xgb_clr_tuned`, `xgb_clr_stack`, `xgb_v4mr`
  and `xgb_v4clr_3000`: same `archv2.oof` code path per fold; every seed comes from `--seeds` and every inner
  split is a function of the fold's training rows only; a shard writes predictions and its CPU time and
  computes no score; one model per shard invocation; one thread per shard process; `--overwrite` refused in
  both modes; CPU time is the sum over shards and the merge refuses above the stamped budget. These five models
  run at one thread whether sharded or not (none had a table from this runner). No prediction of a model that
  already has a development number is touched: `niche.py` (32d01a52…0f07c) and `archv2.py` (a92bbf55…93ce) are
  unchanged, and the merge code path of `niche_s` / `xgb_nichez_context` is not reused for a second merge.
- **Tests.** `tests/test_microbe_archv2.py` sha256 231619e092dec852ec7f5f21a11462f4fd0001058aff33c75bfe9787c53ef61c:
  five new parametrised cases `test_comparator_fold_shards_merge_to_exactly_the_unsharded_table[model]`
  (synthetic design; unsharded table against shards in fold groups [3,5], [0], [4,2,1] + merge: bit-for-bit
  equal prediction tables per family and seed, equal scores and registry values; refusals for two models in one
  shard call, `--overwrite`, a missing fold and a second merge). Run 01:01:58–01:20:32 UTC: 79 of 79 passed,
  rc 0 (slowest: `rf_clr_tuned` 529 s, `xgb_clr_tuned` 127 s).
- **Reading of order item (3) that M3 follows, stated before any run.** On family P (the gate uses the 8 primary
  targets only) the unconditional §6 rows without a `lodo` table are: `rf_v4clr`, `tabpfn_v4clr` (GPU job
  running), `xgb_v4clr_3000`, `xgb_gfo`, `wa_tol`, `wa_pls` and `xgb_v4clr_mo`. The lead's order puts
  `xgb_v4clr_mo` in item (4); §6 lists it without a condition. M3 runs it on P inside item (3) so that the
  B6 list is complete under either reading, and says so in the message. `xgb_v4mr` is one counted run that
  serves P and G together (one 6 CPU-h budget on `lodo`).
- **Timing script.** `v4mr_train_timing_v2.py` (scratch) now wraps the one-thread models in
  `threadpool_limits(1)`, as `run_shard` does. Not run yet; it runs next, alone.

### E23 — `xgb_v4mr` train-only timing: projection 41.6 CPU-h against a 6 CPU-h budget, model HELD; step-(3) lanes started 01:34:30 UTC (logged 01:36 UTC, 2026-11-11)

- **Correction of E22's header.** E22 was appended at 01:21 UTC, not 01:23 UTC.
- **Timing run (lead ruling C; order item 2).** `v4mr_train_timing_v2.py` (scratch) for `xgb_v4mr`, one thread
  (`threadpool_limits(1)` as in a shard), seed 0, fold `domain=D01`, the 1,166 training events only; the "test"
  rows are 20 training events; no held-out row, no label comparison, no score, no registry row. Run alone:
  01:21:14–01:33:51 UTC with no other M3 CPU job alive (the three TabPFN GPU jobs ran; one ended at 01:29 UTC).
  1-minute load 49.84 at the start of the timed calls, 50.17 at the end (51.73 at process start).
  Record: `sentinel_runs/microbe/cv_archv2/xgb_v4mr_train_only_timing.json`, sha256
  3bd3851ab28a7c257947398d06c0c45402e782d7b5f83f320f54d06835f79a7a; log `logs/archv2_v4mr_timing.log`, rc 0.
  - one call with 1 target 209.7 CPU-s; one call with 3 targets 531.9 CPU-s; per target 161.1 CPU-s; per-call
    overhead (features) 48.5 CPU-s.
  - **Projection for `lodo`, 3 seeds, 19 folds: family P 21.2 CPU-h, family G 21.2 CPU-h, P + G 41.6 CPU-h.**
    The stamped budget is 6 CPU-h per track. The projection is about 7 times the budget (3.5 times for one
    family alone).
  - **`xgb_v4mr` is held until the lead rules** (ruling C). M3 proposes no fallback and starts no shard of it.
- **Calibration diagnostic (key `CALIBRATION`; ruling A1 / deviations 01 C1), run once on D1's `lodo`
  out-of-fold tables, 01:34:30–01:34:58 UTC, rc 0** (`logs/archv2_calib_niche_s.log`;
  `lodo/{P,G}/niche_s_calibration.json`). Registry rows (experiment `microbe_arch_v2_cv`, `config.diagnostic =
  calibration`): P `39d1d60f676e` .. `9d186bf1d4d7` (32 rows), G `48f3f298b838` .. `54d2dae73f1a` (32 rows).
  (within-domain slope of observed on predicted, SD(predicted)/SD(observed)):
  P — log_tn (0.94, 0.37), log_tp (1.27, 0.27), log_doc (0.52, 0.50), log_no3 (1.43, 0.36), log_sc (0.66, 0.89),
  log_turb (1.44, 0.25), do (1.46, 0.44), log_chla (−0.22, 0.38);
  G — ph (1.21, 0.45), log_cl (0.77, 0.25), log_so4 (0.19, 0.34), log_ca (0.69, 0.48), log_dic (0.62, 0.89),
  do_sat (1.25, 0.23), log_nh4 (1.43, 0.34), log_uv254 (0.52, 0.33).
  No calibration statistic is a gate condition (dev01 ruling D9); the figures are reported as they are.
- **`tabpfn_v4clr` family P finished** (GPU 1, lock helper; 01:29:15 UTC, rc 0; 2.43 CPU-h): development LODO mean
  pooled R² 0.3050; rows `5cf52cbbac97` .. `7a7e6e231950`. The G jobs of `tabpfn_v4clr` and `tabpfn_clr` continue.
- **Step-(3) lanes (capacity ruling of E22: at most 12 threads in total).** New launcher
  `sentinel_runs/microbe/logs/archv2_cpu_lane.sh` (sha256 fc7e47df…2962): one lane runs its jobs strictly one
  after the other, each job at most 6 threads; two lanes are alive. No load gate.
  - lane a (`archv2_lane_a.jobs`, begun 01:34:30 UTC, load 48.36): calibration (done) → `niche_s_a0` six
    one-thread fold shards (live 01:34:58 UTC, load 48.66, PIDs 2483877–2483882, fold groups as for D1) and one
    merge → `xgb_v4clr_3000` family P, six one-thread fold shards and one merge (ruling B).
  - lane b (`archv2_lane_b.jobs`, begun 01:34:32 UTC): `wa_tol` + `wa_pls` (P, G) → `rf_v4clr` P → `xgb_gfo` P →
    `xgb_v4clr_mo` P.
  - `xgb_v4clr_3000` is sharded on family P only now so that the gate does not wait for G. Its G table is a
    step-(4) item. Estimated from the `xgb_v4mr` per-target cost scaled by the number of features (an estimate,
    not a measurement): about 14 CPU-h for P, so about 2.5 h of wall time at six shards.

### E24 — `wa_tol`, `wa_pls` and ablation A0 (`niche_s_a0`) finished on `lodo` (logged 01:47 UTC, 2026-11-11)

All numbers are development LODO (1,215 events, 19 folds), mean pooled R² over the 8 targets of the family;
experiment `microbe_arch_v2_cv`, split `neon_dev/lodo_cv`. No model here is stochastic (one table each).

| Model | Family | Mean R² | Registry rows (first .. last, 48 each) | sha256 of the table |
|---|---|---|---|---|
| `wa_tol` | P | 0.1935 | `41788011f885` .. `e5ed04f329ed` | 474ed982f32637c81021500a1b8bbd8045837f7a73ed23f8946c6c4e66b8d4bc |
| `wa_tol` | G | 0.3044 | `0598e9c49de3` .. `ed0522fe55f1` | 43e5194ab4ac4207de98e21e157d58a9a5dfb29a54446b8ae5fb10d7d7160f5b |
| `wa_pls` | P | 0.2813 | `de19a374e99a` .. `cfa015d07663` | 3aa5e2c73655d4540a8703ab9bed648429663a002b4153329bdd9e53b923af0c |
| `wa_pls` | G | 0.3379 | `0a6e710ff964` .. `fbaa0eb2f19f` | 6116941f0de41ee0a791c3eacfc137a356c6111dc4e865745d54531d8e005d9e |
| `niche_s_a0` (A0) | P | 0.1042 | `5b5f1b540c4b` .. `2a7af16ffbc2` | e1d865689695d6a16e5bd15399964e9b646a1cf7f6cb74653e9727030b6ddddc |
| `niche_s_a0` (A0) | G | 0.1453 | `7afd5a0dd518` .. `53324b393463` | 2e9c9876aa3200796d775bb6086e67b0d76af2134c88a9cbf091a303165daa5b |

- `wa_tol` + `wa_pls`: one job, 6 threads, 01:34:32–01:42:59 UTC, rc 0 (`logs/archv2_wa_PG.log`).
  Per target, `wa_tol` P: log_tn 0.143, log_tp 0.187, log_doc 0.225, log_no3 −0.135, log_sc 0.632, log_turb 0.060,
  do 0.251, log_chla 0.184; `wa_pls` P: 0.074, 0.181, 0.341, 0.177, 0.743, 0.235, 0.319, 0.180.
- `niche_s_a0` (counted ablation; key `A0`): six one-thread fold shards 01:34:58–01:43:37 UTC (PIDs
  2483877–2483882, every rc 0; load 48.66 → 52.08), CPU time summed over the shards 0.66 CPU-h (budget 6);
  one merge-and-score step 01:43:37–01:44:51 UTC, rc 0 (`logs/archv2_merge_a0.log`). Per target P: log_tn −0.044,
  log_tp 0.105, log_doc 0.158, log_no3 0.029, log_sc 0.438, log_turb 0.025, do 0.010, log_chla 0.113.
  Label-free diagnostics (`lodo/{P,G}/niche_s_a0_diag.json`): N_eff grid is the single value 100 (R15 d);
  first-step failure share of the held-out inversions 0.027 pooled (0 to 0.327 per fold: 0.327 in fold 9,
  0.122 in fold 14, 0.064 in fold 10, below 0.03 elsewhere). The 5% kill rule of §2.5 is worded on the
  train-only smoke test of D1, not on A0; the figure is reported as it is and nothing is changed.
- **Honest reading, no gate statement:** on both families the classical comparators are above D1 on the mean
  (`wa_tol` 0.194 and `wa_pls` 0.281 against `niche_s` 0.186 on P; 0.304 and 0.338 against 0.199 on G), and D1
  is above its ablation A0 (0.186 against 0.104 on P; 0.199 against 0.145 on G). Paired comparisons are not
  computed here.
- **Configurations counted so far in this round:** `niche_s` (slot #6), `xgb_nichez_context` (control),
  `niche_s_a0` (A0). `xgb_v4mr` (slot #7) is held (E23) and has no development number.
- **Calibration diagnostic, reference models (deviations 01 C1: "for reference, for A0, WA-tol and
  `xgb_v4clr`").** Appended to lane b at 01:45:30 UTC as its last two jobs: `--calibration niche_s_a0 wa_tol` on
  P and G, and `--calibration xgb_v4clr` on P (its G table does not exist yet; that reference row follows in
  step 4).
- Lane a moved on to the `xgb_v4clr_3000` family-P fold shards at 01:44:51 UTC (PIDs 2492300–2492305, load 54.97);
  lane b to `rf_v4clr` P at 01:42:59 UTC.

### E25 — `tabpfn_v4clr` family G finished; two corrections of E24 (logged 2026-11-11 01:52 UTC)

- **Correction of E24 (time).** E24 was appended at 01:46 UTC, not at 01:47 UTC as its header says.
- **Addition to E24 (queue).** `xgb_clr_stack` (ruling B5: runs unconditionally on `lodo`; one-thread fold
  shards for P and G in one shard set, then one merge-and-score step) was appended to lane b as its last job at
  01:46:22 UTC. M3 treats it as part of item (3) of the lead's order unless the lead rules otherwise (question
  sent, no reply yet).
- **`tabpfn_v4clr` family G** (comparator, GPU through the lock helper, seeds 0–2): rc 0 at 01:51:04 UTC
  (`logs/archv2_tabpfn_v4clr_G.log`), 3.27 CPU-h. Mean pooled R² 0.3379 (ph 0.549, log_cl 0.047, log_so4 0.110,
  log_ca 0.417, log_dic 0.815, do_sat 0.188, log_nh4 0.244, log_uv254 0.333). Registry rows `c5423c8dc270` ..
  `4ad059a63cea` (48). Table sha256 `ab4e6fe1876c7b30af6542b834cb043bf22d1994abcb6c71680b717b1d66faa7`.
  The mean equals that of `wa_pls` G (0.3379) to four decimals; the two tables were checked to be different
  files with different per-target values (`wa_pls` G: ph 0.679, log_dic 0.871, log_uv254 0.202), so this is a
  coincidence of the rounded mean, not a copied table.
- Load average at 01:51 UTC: 94 (1 min), 77 (5 min). M3 has 12 one-thread-equivalent CPU threads alive (lane a:
  six `xgb_v4clr_3000` P shards; lane b: `rf_v4clr` P at up to 6 threads) plus one GPU job (`tabpfn_clr` G,
  fold 18 of 19 done). The rest of the load belongs to other workstreams. Nothing of M3 is changed for it.

### E26 — `tabpfn_clr` family G finished (logged 2026-11-11 01:54 UTC)

- `tabpfn_clr` family G (comparator, GPU through the lock helper, seeds 0–2): rc 0 at 01:54:04 UTC
  (`logs/archv2_tabpfn_clr_G.log`), 3.30 CPU-h. Mean pooled R² 0.3534 (ph 0.599, log_cl 0.209, log_so4 0.126,
  log_ca 0.418, log_dic 0.804, do_sat 0.172, log_nh4 0.296, log_uv254 0.204). Registry rows `d02198e7096d` ..
  `1a05679873dd` (48). Table sha256 `5d1014fe402e704e42f1f19b722752d6282f73b94a35a88391494a95f0b71c6f`.
- All three TabPFN GPU jobs of this round are finished (`tabpfn_v4clr` P and G, `tabpfn_clr` G). M3 holds no GPU.
- Family G so far, mean pooled R² on development LODO: `tabpfn_clr` 0.3534, `wa_pls` 0.3379, `tabpfn_v4clr`
  0.3379, `wa_tol` 0.3044, `niche_s` 0.1989, `xgb_nichez_context` 0.1775, `niche_s_a0` 0.1453. No paired
  comparison is computed here.


### E27 — lead ruling on `xgb_v4mr` logged; the V-4 runner edit; overnight comparators (logged 2026-11-11 18:32 UTC)

**1. The lead's ruling on the `xgb_v4mr` CPU figure.** File
`docs/preregistration/microbe_arch_v2_v4mr_lead_ruling_20261110.md` (mode 0444, sha256
`1a65d47ad65398827ad64d8e784d7a819c556ec17e65c8fa6d50d9a95625469e`), registry row `fa48ff4c72bf`
(M3-microbe / `microbe/lead_v4mr_ruling`, `extra.written_by == "lead"`). It is not a stamp and not a
deviations file; where it and a message of the lead differ, the file wins. Read in full before acting.
The hold of ruling C (E23) is ended by it. In short:

- **V-1** `xgb_v4mr` runs on `lodo` exactly as amendment §3.1: the §3.1 features and caps, the `xgb_v4clr`
  hyperparameters, seeds 0–2, 19 folds, both target families, all targets. Nothing is reduced.
- **V-2** The CPU figure on `lodo` is at most 30 CPU-h for each target family (P; G), 60 CPU-h in all;
  the ≤ 6 threads of ruling B stay. This is a compute-only deviation from the stamped 6 CPU-h
  (**Dev-a**) and is listed wherever results of this model are reported.
- **V-3** One family at a time, P first, fold shards under ruling B / E14 unchanged (one thread per shard,
  at most 6 shards alive, a shard writes predictions and its own CPU time and no metric, one
  merge-and-score step per family, the bit-for-bit shard test green before the first real shard). The merge
  refuses to score when the CPU time summed over that family's shards is above 30 CPU-h; M3 then stops and
  reports, reads no partial table and changes no setting. Family G follows the P merge, as a shard set of
  its own or unsharded at one thread, under the same check. One counted configuration (slot #7); a second
  run of a family is refused.
- **V-4** Runner edit: `CPU_BUDGET_H["xgb_v4mr"]` 6.0 → 30.0, checked per family run; no other entry of
  that table changes; a further runner change that per-family shard sets need is made in the same edit,
  before the first `xgb_v4mr` shard, at a moment at which the runner's code-state rule invalidates no shard
  set in flight; afterwards the full M3 test set is run green and the runner's sha256 before and after is
  recorded here.
- **V-5** From the first `xgb_v4mr` shard until its family-G table is merged, `v4mr.py`, `niche.py`,
  `baselines.py` and the fit path of the runner do not change; if the runner file must change in between for
  another model, the bit-for-bit test of `xgb_v4mr` is run green again and both hashes are logged.
- **V-6** Capacity stays 12 threads in total and 6 per job (it is not raised). The load and the time are
  logged at the launch and at the end of each shard set. Parallel CPU jobs are a recorded compute-only
  deviation (**Dev-b**).
- **V-7** The D1 gate is held on **family P** by: every comparator of amendment §6 whose status is not
  "only if D3b / D2 / D4 runs" — the tuned genus-CLR grids of the second §6 row included —, `xgb_clr_stack`
  (unconditional by ruling B5), `xgb_v4clr_mo`, `xgb_nichez_context`, `xgb_v4mr`, ablation A0 and the
  calibration diagnostic. **Family-G tables do not hold the gate.** M3's message "gate prerequisites
  complete" lists for family P each model with its registry row ids and names the §6 rows it counted.
- **V-8** With that message goes an arithmetic projection (no new timing run) of the CPU time of
  `xgb_v4mr_stack` on `lodo`, from the timing record and the measured CPU time of the finished `xgb_v4mr`
  and `xgb_clr_stack` runs. Neither slot-#8 arm starts before the lead's gate evaluation is logged; D3b not
  before the lead rules on its figure.
- **V-9** No `xgb_v4mr` on any other scheme before a lead ruling on its figure there.
- **V-10** `xgb_v4clr_3000` family G runs after the gate prerequisites, as a shard set of its own; the
  runner change for it is made with no shard set in flight and its bit-for-bit case is green before use.

No gate outcome is stated by M3 here or anywhere else; the lead evaluates the gate once from registry rows.

**2. Comparators that finished overnight** (both lanes ran to the end of their job files; all rc 0):

- `rf_v4clr` family P (comparator, 6 threads, seeds 0–2): rc 0 at 02:49:44 UTC
  (`logs/archv2_rf_v4clr_P.log`), 4.95 CPU-h. Mean pooled R² 0.2805 (log_tn 0.106, log_tp 0.261, log_doc
  0.279, log_no3 0.110, log_sc 0.755, log_turb 0.106, do 0.402, log_chla 0.225). Registry rows
  `5330a1998cee` .. `11dbb071a6dc` (48). Table sha256
  `3424e773a14919b07459370b8b11701313e1911cd3358b37ff8ca99a463a90ad`.
- `xgb_v4clr_3000` family P (comparator of the §6 row "V4-XGB cap 3,000", six one-thread fold shards under
  ruling B, PIDs 2492300–2492305): shards rc 0 at 05:16:43 UTC, merge rc 0 at 05:17:30 UTC
  (`logs/archv2_merge_xgb_v4clr_3000_P.log`), 16.62 CPU-h summed over the 19 shards (this model has no entry
  in `CPU_BUDGET_H`, so the merge applied no cap; the figure is recorded for the record). Mean pooled R²
  0.3188 (log_tn 0.109, log_tp 0.305, log_doc 0.329, log_no3 0.257, log_sc 0.814, log_turb 0.130, do 0.378,
  log_chla 0.229). Registry rows `df945072caaf` .. `7966e9b6f161` (48). Table sha256
  `3852a3d681c573d1fbf242d31383b305257103abb97debcce8134ebd3bc7e89b`.
- Lane a ended 05:17:30 UTC (load 29.27), lane b ended 02:49:44 UTC (load 57.06). M3 holds no GPU and, until
  the launches of item 4, no CPU job. Load at the time of this entry: 40 (1 min). Disk: 170 GB free
  on `/`.

**3. The V-4 runner edit** (made with no shard set in flight: both lanes had ended, so the runner's
code-state rule — all shards of a set and their merge under one code state — invalidated nothing).

- `scripts/train/microbe_arch_v2_cv.py` sha256 **before**
  `d85d90a7d32d89a7a5dbbb4e9d0648cada355d452703ce1ce7742dea647e8cbd`, **after**
  `94ac5fa22d503ba43893341d951b5fb78c95c643c098977b3d3cc85ca8a98d66`.
- Changed, in one edit: (i) `CPU_BUDGET_H["xgb_v4mr"]` 6.0 → 30.0 with the ruling named in the comment; no
  other entry of the table changes (a test pins the whole table). (ii) The per-family shard sets V-3 needs:
  a new helper `shard_set(name, families)` makes the shard folder of a model of `SHARD_COMPARATORS` carry
  its family set (`_shards/<model>__P`, `__G`, `__PG`), and `shard_dir` takes the families and is called
  with `a.families` at its four call sites. The inversion-based models keep ONE shard set that serves both
  families, so there a fold is still fitted once (ruling of E14), and their folder name is unchanged. The
  budget is thereby checked per family run: each family has its own shard set and its own merge, which sums
  that family's shards only. Nothing in the fit path, in the seeds, in the thread counts or in the scoring
  changes, and no model that already has a development number can change a prediction by it.
- Effect on what is already on disk: the finished `xgb_v4clr_3000` family-P shard set keeps its old folder
  `cv_archv2/lodo/_shards/xgb_v4clr_3000`. It is merged and scored, its table exists, and `refuse_existing`
  refuses a second family-P run; the folder is simply no longer addressed. Its family-G set (V-10) will be
  `_shards/xgb_v4clr_3000__G`.
- `tests/test_microbe_archv2.py` (not a frozen file; sha256 before
  `231619e092dec852ec7f5f21a11462f4fd0001058aff33c75bfe9787c53ef61c`, after `ea7f6818cf742247bfd78bef69f68751ea73108c37863ed1e0935ae80de5b75c`): the bit-for-bit shard
  test of ruling B is now parametrised over the family set each comparator really runs with — `rf_clr_tuned`
  G, `xgb_clr_tuned` G, `xgb_clr_stack` P+G, **`xgb_v4mr` P and G separately**, `xgb_v4clr_3000` P and G
  separately — so the per-family case of `xgb_v4mr` is covered before its first real shard; it also checks
  that the other family is a shard set of its own that is not blocked and writes no metric, and that the
  budget table is exactly the one above.
- First test run of the changed shard tests (18:14 UTC): 7 passed, 2 failed — the new "other family is not
  blocked" check had been written for every sharded comparator, and the tuned genus-CLR grids of the
  existing track run on family G only (the runner refuses family P for them, as it should). The check is
  now limited to the models that run on both families; nothing in the runner changed for it. Both
  `xgb_v4mr` cases (P and G) and both `xgb_v4clr_3000` cases were green in that run already.
- Full M3 test set after that fix (18:14–18:31 UTC,
  `tests/{test_microbe_archv2,test_microbe_archv2_compare,test_microbe_niche,test_microbe,test_microbe_testlook,test_microbe_wq,test_neon}.py`):
  one failure left, and it was a second place that pinned the old budget figure —
  `test_runner_cpu_budget_stops_before_any_metric` asserts `CPU_BUDGET_H["xgb_v4mr"] == 6.0`. That assertion
  now reads 30.0 with the ruling named beside it (test file sha256 after this second fix
  `6bf49139c527b1a2e9160b7532bf5d30818e9b9461f07a33b0924d44d9cbce92`); every other test of the set passed,
  the tests of `scripts/eval/microbe_arch_v2_compare.py` included. The full set is being re-run as a whole
  for the green record; the first `xgb_v4mr` shard does not start before it is green, and that run, the
  `xgb_v4mr` shard launch and its PIDs are the next entry.

**4. Lane b's job file and the relaunch.** At 01:58:06 UTC (before the ruling was carried out) lane b's
job file was truncated in place to its first two lines, so that lane b ended after `rf_v4clr` P instead of
starting jobs that would have held a shard set in flight across the runner edit. The removed jobs —
`xgb_gfo` P, `xgb_v4clr_mo` P, the two reference calibration jobs and the `xgb_clr_stack` P+G shard set —
were not run; the previous content is kept at `logs/archv2_lane_b.jobs.before_truncate_0157Z`. They are
relaunched now as lane c (`logs/archv2_lane_c.jobs`: `xgb_gfo` P, `xgb_v4clr_mo` P,
`--calibration niche_s_a0 wa_tol --families P G`, `--calibration xgb_v4clr --families P`), one job at a
time at 6 threads. Lane c was started at 18:27:18 UTC (load 40.45, lane PID 2736187, first job `xgb_gfo` P)
— that is before the green re-run of the full test set, which M3 states plainly here: these are unsharded
comparator runs that the edit cannot reach (the edit changes the `xgb_v4mr` budget figure and the shard
folder name of the models of `SHARD_COMPARATORS`, and nothing else), their bit-for-bit shard tests are not
involved, and the only test that was red at that moment was the stale 6.0 assertion described above.
No `xgb_v4mr` shard is started before the full set is green. `xgb_clr_stack` (P+G, one shard set, ruling
B5) follows after the `xgb_v4mr` family-P shard set, to keep the capacity of V-6 (12 threads in total).

**5. What is still owed to the lead.** One line with the PIDs when the first `xgb_v4mr` family-P shard is on
real rows; later the single message "gate prerequisites complete" for family P with the registry row ids per
model and the §6 rows counted, together with the arithmetic `xgb_v4mr_stack` projection of V-8. Family P
still needs `xgb_gfo` (running), `xgb_v4clr_mo`, `xgb_clr_stack` and `xgb_v4mr`; every other comparator that
V-7 names is finished (the §6 rows 1–5 under `cv/lodo/` from the existing track, rows 6–8, 10 and 11 in this
round). No gate outcome is stated.
### E28 — full M3 test set green; `xgb_v4mr` family-P shard set started 18:50:55 UTC (logged 2026-11-11 18:51 UTC)

- **Full M3 test set, after the second fix of E27, 18:33–18:50:24 UTC, rc 0, 185 tests, no failure**
  (`tests/test_microbe_archv2.py`, `test_microbe_archv2_compare.py`, `test_microbe_niche.py`,
  `test_microbe.py`, `test_microbe_testlook.py`, `test_microbe_wq.py`, `test_neon.py`; output in the
  scratch file `pytest_full_m3_green.txt`). File hashes at that moment: runner
  `94ac5fa22d503ba43893341d951b5fb78c95c643c098977b3d3cc85ca8a98d66` (before the V-4 edit
  `d85d90a7d32d89a7a5dbbb4e9d0648cada355d452703ce1ce7742dea647e8cbd`), test file
  `6bf49139c527b1a2e9160b7532bf5d30818e9b9461f07a33b0924d44d9cbce92`. The bit-for-bit shard test of ruling B
  is green for `xgb_v4mr` family P and family G separately, which is the case V-3 asks for.
- **V-5 freeze begins with the first shard.** From now until the `xgb_v4mr` family-G table is merged,
  `sentinel/models/microbe/v4mr.py` (sha256
  `3638553f6ef813880286066999498120144ff6bf8b643a9c5fbcbae519777891`), `niche.py`
  (`32d01a52205836827f3ba645c115ac284fca25596c1c66bed31084e650a0f07c`), `baselines.py`
  (`8279d7d0b7fe8588662fa7036512c6c8233bf0230a573b75f919f5ec1820da04`) and the fit path of the runner do not
  change. If the runner must change in between for another model, the `xgb_v4mr` bit-for-bit test is run
  green again and both hashes are logged.
- **`xgb_v4mr` family P, fold shards (slot #7, lead ruling V-1 … V-6; counted configuration).** Launched
  18:50:55 UTC, 1-minute load 50.41 at the launch. Six one-thread processes, PIDs 2774309 (folds 0, 6, 12,
  18), 2774310 (1, 7, 13), 2774311 (2, 8, 14), 2774312 (3, 9, 15), 2774313 (4, 10, 16), 2774314 (5, 11, 17);
  shard set `cv_archv2/lodo/_shards/xgb_v4mr__P`; seeds 0–2, 19 folds, the 8 primary targets, the §3.1
  features and caps and the `xgb_v4clr` hyperparameters, nothing reduced. The merge-and-score step follows
  automatically in lane d and refuses to score if the CPU time summed over these shards is above 30 CPU-h;
  the projection of E23 is 21.2 CPU-h for the family. The PIDs and the start were sent to the lead in one
  line at 18:51 UTC. The load and the time at the end of the shard set will be logged with the result.
- **Parallel CPU work (Dev-b, V-6: 12 threads in total, 6 per job).** Lane c: `xgb_gfo` family P since
  18:27:18 UTC at 6 threads (fold 1 of 19 done at 18:37 UTC, 0.91 CPU-h per fold), then `xgb_v4clr_mo` P and
  the two reference calibration jobs. Lane d: the six one-thread `xgb_v4mr` shards. No GPU is held by M3.
- No gate outcome is stated. No slot-#8 arm is started. No git commit.
