# M3 SENTINEL-Microbe, architecture round v2 — lead ruling on the `xgb_v4mr` CPU figure and on the D1-gate prerequisites

Written Tue 10 Nov 2026 20:56 EST by the lead. **This is not a stamp.** It releases no key of the stamp guard
(`sentinel/models/microbe/stamp.py`) and its name is not a deviations-file name. It changes no model, feature set,
hyperparameter, seed, fold, row, label, metric, comparator list or gate threshold. It changes two resource figures
and fixes one reading, and it says so openly as deviations (section 3).

## 0. Files read (sha256 at the time of writing)

| File | sha256 | Mode |
|---|---|---|
| `docs/preregistration/microbe_arch_v2_amendment.md` | `022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a` | 0444 |
| `docs/preregistration/microbe_arch_v2_lead_rulings.md` | `e148a09def685f6fa730650e24ba384389daa1fe44d29c07151568da028ecd95` | 0444 |
| `docs/preregistration/microbe_arch_v2_deviations_01.md` | `a3fdc1ed74468123c96e744b88a25958762487e2f5106178b50d6f9a96245a58` | 0444 |
| `docs/preregistration/microbe_arch_v2_dev01_lead_rulings.md` | `25b00774eba005c2bd73abb6fd838f886d3cbd531ec09925db95ff71acbcb86c` | 0444 |
| `sentinel_runs/microbe/cv_archv2/xgb_v4mr_train_only_timing.json` (timing record) | `3bd3851ab28a7c257947398d06c0c45402e782d7b5f83f320f54d06835f79a7a` | — |
| `scripts/train/microbe_arch_v2_cv.py` (runner; will change under V-4) | `d85d90a7d32d89a7a5dbbb4e9d0648cada355d452703ce1ce7742dea647e8cbd` | — |

The M3 log `docs/preregistration/microbe_arch_v2_log.md` is append-only and its hash changes with every entry; it
is cited by entry number (E14, E20, E22, E23, E24).

## 1. Facts

- **Stamped text.** Amendment §3.3: "CPU: ≤ 6 threads; ≤ 6 CPU-h per track for D3a and ≤ 12 CPU-h per track for
  D3b. If D3b exceeds this, it runs on `lodo` only." Amendment §12, row 7: "D3 CPU budgets (§3.3) | D3b on `lodo`
  only". The stamped text names no consequence for D3a going over its figure.
- **Role of `xgb_v4mr`.** Lead rulings file: "`xgb_nichez_context` and `xgb_v4mr` enter the maximum that defines
  the strongest simple baseline for the gate"; B6 lists `xgb_v4mr` among the runs that must be finished on `lodo`
  before the gate is evaluated; B3 puts it in the comparator set of D1 and D2 ("each if run"). It is also arm D3a
  (slot #7).
- **Hold.** Lead message of 2026-11-11T00:24:55.761Z, item C (Annex A; log E20): if the train-only projection is
  above 6 CPU-h per track, `xgb_v4mr` is held until the lead rules, and M3 invents no fallback.
- **Timing record** (log E23; written by M3, not re-run by the lead): one thread, fold `domain=D01`, 1,166
  training events, seed 0, 1-minute load 49.84 at the start and 50.17 at the end. One call with 1 target 209.66
  CPU-s; one call with 3 targets 531.88 CPU-s; hence 161.11 CPU-s per target and 48.55 CPU-s per call for the
  features. Projection for `lodo`, 19 folds, 3 seeds, 8 targets per family: 21.18 CPU-h for family P, 21.18 CPU-h
  for family G, 41.58 CPU-h for both in one run. The lead recomputed the three projections from the two call
  times; they agree. M3 logged that the run read no held-out row and computed no score.
- **No number exists.** At the time of writing the registry holds no row of a model whose name contains
  `xgb_v4mr`.
- **Twin.** The D3a twin `xgb_v4clr_3000` (amendment §3.1: "XGB[V4-CLR, cap 3,000] without the aggregated levels")
  has no stamped CPU figure; M3's estimate for it is about 14 CPU-h on family P (log E23; an estimate, not a
  measurement).
- **Machine.** 48 logical CPUs (2 sockets × 12 cores × 2 threads). The load was above 48 during the timing run and
  is above 48 now, so CPU time per fit is at the level the record shows.

## 2. Rulings

- **V-1 — `xgb_v4mr` runs, exactly as amendment §3.1 specifies.** Features, caps, the `xgb_v4clr` hyperparameters,
  seeds 0–2, the 19 `lodo` folds, both families and all targets stay as stamped. Nothing is reduced to save time.
  Reasons: (a) the stamped text names no consequence for D3a above its figure (§12 row 7 names one for D3b only);
  (b) `xgb_v4mr` is part of the maximum in gate condition 2 and of the comparator set of ruling B3, so dropping or
  shrinking it would remove or weaken a comparator that D1 and D2 must beat; (c) the decision is taken from a
  train-only timing, before any `xgb_v4mr` number exists.
- **V-2 — new CPU ceiling (deviation from a stamped resource figure).** For scheme `lodo` the figure "≤ 6 CPU-h
  per track for D3a" is replaced by: **at most 30 CPU-h for each target family (P; G), 60 CPU-h in all.** The
  ceiling is fixed now from the projection (21.18 CPU-h per family, with a margin of about 40%) and is not raised
  again without a new lead file. "≤ 6 threads" stays. The D3b figure and its stated consequence are untouched
  (V-8).
- **V-3 — how it is run.** One family at a time, P first. Fold shards as lead ruling B (Annex A) and log E14
  allow, with those conditions unchanged: one thread per shard, at most 6 shards alive, shards write predictions
  and CPU time only, one merge-and-score step per family, the bit-for-bit shard test for `xgb_v4mr` green before
  the first real shard. The merge refuses to score when the CPU time summed over the family's shards is above 30
  CPU-h; then M3 stops and reports, reads no partial table and changes no setting. The family-G table is run after
  the P merge, either as a shard set of its own or unsharded at one thread, under the same 30 CPU-h check.
  `xgb_v4mr` stays one counted configuration on one track (slot #7); running it per family does not make a second
  one, and a second run of a family is refused.
- **V-4 — runner edit.** `CPU_BUDGET_H["xgb_v4mr"]` in `scripts/train/microbe_arch_v2_cv.py` goes from 6.0 to 30.0
  and is checked per family run. No other entry of that table changes. If per-family shard sets need a further
  runner change, it is made in the same edit, before the first `xgb_v4mr` shard. The edit is made at a moment at
  which the runner's own code-state rule does not invalidate a shard set in flight; that rule is never bypassed.
  After the edit the full M3 test set is run and must be green, and the log records the runner's sha256 before and
  after.
- **V-5 — code state of the arm.** From the first `xgb_v4mr` shard until its family-G table is merged,
  `sentinel/models/microbe/v4mr.py`, `niche.py`, `baselines.py` and the fit path of the runner do not change. If
  the runner file must change in between for another model, the bit-for-bit test of `xgb_v4mr` is run green again
  and both hashes are logged. A change that can alter an `xgb_v4mr` prediction makes a new counted configuration
  (amendment §12 row 16).
- **V-6 — capacity (the lead message of 2026-11-11T01:08:31.905Z put on disk; Annex B).** M3 may run CPU jobs in
  parallel with at most 12 threads in total and at most 6 threads per job. This is not raised here: the machine is
  oversubscribed and other workstreams need the same cores. `xgb_v4mr` takes its place inside the 12 threads; M3
  orders its lanes. The load and the time are logged at the launch and at the end of each shard set.
- **V-7 — which runs hold the D1 gate (reading of B5 and B6).** The stamped rulings file wins over the lead's
  order message (Annex B) where they differ. On **family P** the gate waits for: every comparator of amendment §6
  whose status is not "only if D3b / D2 / D4 runs" — the tuned genus-CLR grids of the second §6 row included —,
  `xgb_clr_stack` (unconditional by B5), `xgb_v4clr_mo`, `xgb_nichez_context`, `xgb_v4mr`, ablation A0 and the
  calibration diagnostic. Item (4) of the order message listed "tuned, stack, xgb_v4clr_mo" after the gate; for
  family P that was wrong and is withdrawn. **Family-G tables do not hold the gate:** B6 says the gate uses the 8
  primary targets only, and both gate conditions are computed per primary target, so no family-G number can enter
  either. M3's message "gate prerequisites complete" lists, for family P, each model with its registry row ids and
  names the §6 rows it counted.
- **V-8 — D3b is not ruled here.** With the "gate prerequisites complete" message M3 sends a projection of the CPU
  time of `xgb_v4mr_stack` on `lodo`, by arithmetic from the timing record and from the measured CPU time of the
  finished `xgb_v4mr` and `xgb_clr_stack` runs; no new timing run. Neither slot-#8 arm starts before the gate
  evaluation is logged, and D3b does not start before the lead has ruled on its figure.
- **V-9 — other schemes.** `xgb_v4mr` is not run on any scheme other than `lodo` before the lead has ruled on its
  CPU figure there.
- **V-10 — `xgb_v4clr_3000`, family G (M3's question).** After the gate prerequisites, as a shard set of its own.
  The runner change for it is made with no shard set in flight, with the bit-for-bit test for that case green
  before use, and subject to V-5.

## 3. Deviations this file records (to be listed wherever M3 results are reported, and in the post-results review)

- **Dev-a.** D3a CPU figure on `lodo`: 6 CPU-h per track (amendment §3.3) → 30 CPU-h per family, 60 in all (V-2).
  Compute only; decided from train-only timing before any `xgb_v4mr` number; the model is unchanged.
- **Dev-b.** Parallel CPU jobs: amendment §12 row 21 ("one heavy CPU job at a time"), the same sentence in the
  lead rulings file ("Resources: at most 6 CPU threads and one heavy CPU job at a time") and rule 5 of
  `LEAD_DECISIONS_20261110.md` ("keep CPU jobs sequential per workstream") → up to 12 threads in total across M3's
  CPU jobs, at most 6 per job (V-6; in force since the message of Annex B). Compute only; settings, seeds and
  thread count per model unchanged; the bit-for-bit shard tests cover the sharded models.
- **Reading, not a deviation.** V-7 applies B5 and B6 as stamped and limits the wait to family P for the reason
  given there.

## 4. How this file is read

It ends the hold of item C of Annex A. It wins over Annex B where V-7 differs from it, and over the sentence of
log E22 that gives `xgb_v4mr` one 6 CPU-h budget on `lodo`. Ruling A of log E14 (the `niche_s` budget) is
unchanged. The stamped amendment, the lead rulings file, deviations file 01 and its rulings hold in everything
else. If this file and a message of the lead differ, this file wins.

## Annex A — lead message to M3 of 2026-11-11T00:24:55.761Z (2,142 characters), verbatim

```text
Lead rulings on your questions A–C (record them as E20 in your log; E15–E19 accepted as reported).

A. YES — start the 6 one-thread D1 shards now, regardless of the load figure. The 40 limit is a hygiene figure of your launcher, not a stamped rule; rule 5 of the round only says ≤ 6 threads per job and CPU jobs sequential per workstream, and 6 one-thread shards are one 6-thread job. Conditions: (1) log the 1-min load and the time at launch and at the end; (2) the D1 shard set is your only CPU job while it runs (TabPFN on GPUs continues); (3) CPU time is the sum over shards, measured as before; if any stamped CPU cap is reached, stop and report — no restart, no setting change, without a ruling. Send me the PIDs and start time as soon as they are live (I owe the user a notice when a new architecture is training).

B. YES — the fold-shard pattern may be used for rf_clr_tuned, xgb_clr_tuned, xgb_clr_stack, xgb_v4mr and xgb_v4clr_3000, under the E14 conditions without exception: same code path per fold; every seed and every inner tuning split a function of the fold only, never of the shard or of the order; shards write predictions and CPU time only and compute no score; one merge-and-score step; the synthetic bit-for-bit equality test (sharded = unsharded tables) passes for EACH of these models before its real run; ≤ 6 threads per model in total; one model at a time; CPU time = sum over shards and counts against that model's stamped budget; a model stays a counted model (a re-run is refused). The "≤ 2 h" figure of rule 5 is the GPU-lock limit; a shard that runs longer on CPU only is not a breach, but keep shards short enough to be restartable only by the stamped rerun rule (infrastructure failure, logged).

C. Agreed as you propose: train-only timing of xgb_v4mr first (it may run after the D1 shards, not beside them), send me the projection; if > 6 CPU-h per track, hold xgb_v4mr until I rule. No fallback is invented by M3.

QMP confound result noted: rule not fired (0.210 / 0.388), first output stands (D7); batch share reported with every +QMP result.

Unchanged: nothing not released is touched; no git commits.
```

## Annex B — lead message to M3 of 2026-11-11T01:08:31.905Z (1,849 characters), verbatim

```text
Lead: D1 row ids received (E21); no gate outcome is stated by anyone yet — I evaluate the gate once, from registry rows, when everything ruling B6 lists is finished on lodo.

Capacity ruling: rule 5 limits threads per job (at most 6), not the number of jobs. You may run CPU jobs in parallel, with a total of at most 12 threads across your CPU jobs at any moment (for example two 6-thread jobs, or several 1-thread ruling-B jobs up to 12). The machine has 48 cores and load was about 62 at 20:06 EST, so no more than that. The xgb_v4mr one-thread TIMING run is the exception: run it with no other CPU job of yours alive, and state the load average in its record. Nothing else changes: same settings, same seeds, ruling-B models at 1 thread, bit-for-bit tests green before the shard change is used on real rows.

Order, so the gate is not held up by work it does not need: (1) ruling-B shard change + five tests; (2) xgb_v4mr timing -> projection to me; (3) everything B6 names as a gate prerequisite — niche_s_a0 (ablation A0), the calibration diagnostic on D1's development predictions, xgb_v4mr, and every unconditional section-6 comparator on lodo (WA-tol included); (4) then niche_s_tax and the remaining comparators and tables (family G comparators, tuned, stack, xgb_v4clr_mo). When (3) is complete send me one message with the registry row ids per model and family and the statement "gate prerequisites complete"; I log the evaluation under microbe/d1_gate_evaluation and tell you slot #8 (D2 draft or D3b). Do not start either slot-#8 arm before that.

The N_eff-at-grid-minimum diagnostic stays in the log as you wrote it; no grid change (that would be a new configuration after a development number). The three TabPFN GPU jobs continue; idle GPUs may be used for approved GPU work through the lock helper, never GPU 2, at most 2 h per job.
```
