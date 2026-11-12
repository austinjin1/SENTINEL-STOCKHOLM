# Handoff — M7 SENTINEL-Lite (updated 2026-11-11 13:12 EST; architecture round v2 is STAMPED and running)

## STAMPED — 2026-11-11 (read this first)

- **Stamp row `5879088974eb`**, workstream `M7-lite`, experiment `lite/prereg_snapshot`, written by the lead,
  2026-11-11T18:02:38+00:00. `python -m sentinel.models.litev2.stamp` → `"stamped": true`.
  **Ruling N-03: every v2 job log cites that row id; a run citing another `lite/prereg_snapshot` row id is void.**
  Each job log under `$SENTINEL_RUNS/lite/logs_arch_v2/` opens with it.
- **The binding text is `docs/preregistration/lite_arch_v2_lead_rulings.md`** (sha256 `95d5bdcb…d9c29`, 0444, 929
  lines): R-01…R-09, N-01…N-06, §5 (the five open points of deviations entry 11), §6 (every verifier item in the
  form that binds), §7 (D-0…D-5 and the §B wording), §8 (the pinned artefacts), §9 (deviation 10 corrected), §10
  (the order of work), §11 (how it is read). **It wins over the amendment, the addendum and the deviations file.**
  Work from it, not from messages.
- **D-0: selection rule S is the 3-seed ensemble mean** for every selectable configuration of the round (re-runs of
  configs 1–6, both twins, slots 7–8); a configuration with fewer than 3 finished seeds is **not selectable**. The
  carrier of any label stays the INT8 seed-0 pipeline (DS-09). D-1 (time-held-out stratum), D-2 (the two real-key
  tests stay unskipped), D-3 (E3a_turb needs 30 positives / 10 water bodies in **both** views), D-4 (the ledger and
  the per-job budgets as built, hard 45 GPU-h), D-5 (the A4 ablation keeps its own hash-checked anchor) are all as
  M7 proposed. §B binds word for word: carrier `scope="image"`, `scope="graph"` exported and descriptive, "mixed
  precision" wording, parity thresholds fixed and never loosened.
- Eight files are 0444 and must not be touched: the amendment, the addendum, the deviations file, the rulings file,
  `lite_split_national2b.json`, `lite_protocol.md`, `lite_national_protocol.md`, `labels_national2b.parquet`, the P2
  index (and the P2 lock, the C2 csv + meta, both code-readings files, the code-hash json). **A change to any of
  them voids the stamp until the lead writes a new row** — re-hashed and verified 13:04 EST, all match §8.
- **No code of the 39 files in `litev2_code_sha256_20261110.json` may change** (the stamp row pins that record).
  A5 (the rerun-view path of `litev2_comparators.py`) is due **before the national freeze**, not now, and the lead
  must be told the new hash when it is written (R-04 is the rule it serves).
- Hardware (lead, 13:09 EST): **GPU 0 is M7's only card.** GPUs 1 and 5 carry this project's M5 Bioassess-X jobs
  (the lead allocates between workstreams — do not take them, do not ask M5 to yield); GPUs 2, 3, 4, 6, 7, 8, 9
  belong to another user. Check `nvidia-smi` immediately before each launch, lock helper for every job,
  ≤ 2 h, ≤ 6 CPU threads. Remaining GPU work ≈ 30 GPU-h → about 30 h of wall clock on one card.

### Launch state (2026-11-11 13:12 EST)

| Job | Where | State |
|---|---|---|
| `litev2_tabpfn_memtest` (§10.1) | GPU 0, 13:05:22–13:06:27 | **done**: cap **10,000** fits (secchi, 9,371 rows, peak 5,883 MB alloc / 9,126 MB reserved, tabpfn 2.2.1); registry row `ad650fe62de2`, 0.018 GPU-h |
| `rerun_crop256_kd03_national2b` (config 3) | GPU 0, PID 2668410, from 13:06:46 | running, budget 2.0 h, Test-S KD mask active |
| `litev2_comparators --group tabular --device cpu` | CPU, 13:07:29–13:19:07 | **done**: S1–S9, P1, P2a/b registered on BOTH blocks; anchor/S6 artefact sha256 `63ef5fe56107fad85cfdb6285934e388c2d3a95d1d396c4f57a581b5aca52921` (10,651 rows, experiment `lite/v2_anchor_artefact`), run dir `litev2_comparators_tabular_20261111-130735` |
| re-run chain (twin of 3 → configs 1, 2, 4, 5, 6 → twin of 6) | GPU 0, one at a time | queued, driver `scratchpad/m7/rerun_chain.sh`, status file `$SENTINEL_RUNS/lite/logs_arch_v2/rerun_chain_status.txt` |
| DenseNet chain (12 fits + 4 evals, P3a/P3b x 1e-4/3e-4 x seeds 0–2) | second card only (`SENTINEL_GPUS=1,5`), one at a time | queued and waiting for M5 to free a card; driver `scratchpad/m7/densenet_chain.sh`, status `…/densenet_chain_status.txt` |

**State at 14:38 EST.** Done: the memory test (cap 10,000, row `ad650fe62de2`); the tabular comparator group
(S1–S9, P1, P2a/b on both blocks; anchor/S6 sha256 `63ef5fe5…a52921`); the re-run of configuration 3
(`natstudent_mnv4s_raw_crop256_kd03_national2b_20261111-130652`, 0.665 GPU-h, rows `61757e154160`, `a1f7401a7eb8`,
`d25a70813546`, `ff61285dfd19`) and of its no-KD twin (`…_nokd_national2b_20261111-134700`); the six warm-start
checkpoint hashes (`litev2_steps.py register-init`, row **`db88878cfd44`**, experiment `lite/v2_init_checkpoints`;
`launch.init_checkpoint` resolves all six); the first DenseNet fit (1.167 GPU-h, inside the 2.0 h budget, so 12 runs
project to 14.0 h against the 16.5 h estimate). The KD mask is verified from the re-run's own log:
`n_chips 10424, n_kd_masked 308, n_chips_with_teacher_before_mask 2593, n_masked_with_teacher 304,
n_chips_with_teacher 2289` — 308 is the registered count of row `a7349f26339f`.

**Queued and self-driving:** the re-run chain on GPU 0 (six left, through about 18:20 EST), the DenseNet chain on
GPU 5 (eleven fits plus four evals, to about 03:40 EST), and `scratchpad/m7/phase2_chain.sh` (driver PID 2839992),
which waits for the re-run chain and then runs, on GPU 0, `--arm embed` → `--group tabpfn` → `--group stacked`
(CPU, embeddings dir `$SENTINEL_RUNS/lite/litev2_embeddings/natstudent_mnv4s_raw_crop256_kd03_national2b_20261111-130652`)
→ `litev2_steps.py e3a-count`. Status files: `rerun_chain_status.txt`, `densenet_chain_status.txt`,
`phase2_status.txt` under `$SENTINEL_RUNS/lite/logs_arch_v2/`. **No candidate fit is queued anywhere** — that waits
for the gate to be empty and for an explicit check. Ledger 2.43 h finished of 45; round projection about 29 h.

**Gate state at 13:25 EST** (`H.missing_comparators`): `national_val` and `val_p2` still need **S10, S11 and the four
DenseNet comparators**; everything else is registered. Re-run timing: 40 epochs per seed, 18–20 s per epoch, about
0.7 h per 3-seed re-run, so the eight re-runs end about 18:40 EST. The Test-S KD mask is confirmed attached on the
re-run path (`lite_train.py:64,85` look the loader up on the module at run time, after `activate()` patches it); its
printed line is block-buffered and appears when a job exits.

**When the re-run chain ends, the next session does, in this order:** `litev2_steps.py register-init` (needs the
config-3 re-run and its twin) → `litev2_train.py --arm embed` (GPU) → `litev2_comparators.py --group tabpfn` (GPU,
0.75 h per block) and `--group stacked --embeddings $SENTINEL_RUNS/lite/litev2_embeddings/<config-3 re-run dir>`
(CPU) → DenseNet evals (in the chain) → `litev2_steps.py e3a-count` → candidates only once the gate is empty for
both blocks → selection once (3-seed ensemble mean, D-0) → R-04 refits if a configuration 1–6 wins.

Then, in §10 order: register the re-run checkpoint hashes (`litev2_steps.py register-init`) → `litev2_train.py
--arm embed` → `litev2_comparators.py --group stacked --embeddings …` and `--group tabpfn` → DenseNet P3a/P3b
(12 fits, 4 evals) → `litev2_steps.py e3a-count` → slots 7–8 and their twins (warm start, no fallback) → the N-02
sensitivity row → selection once → the R-04 refits if a configuration 1–6 wins. **No candidate fit before every H3
comparator row is registered for both blocks** (`H.require_gate`, enforced in code).

## (below: the pre-stamp record of 2026-11-10)


## Lead order 2026-11-10 20:20 EST — done except A5 (read this first; it overrides the launch order further down)

The independent review of the stamp package came back "stamp with lead rulings". The lead writes
`docs/preregistration/lite_arch_v2_lead_rulings.md`; the lead's 20:20 rulings are INTENDED and bind only once they are
in that file. **After the report of item C (sent, see below) no code changes until the stamp row exists** — the lead
re-hashes, runs the tests and stamps. Still nothing fitted, no validation row loaded.

Do not touch (20:20 list): the amendment, the label addendum, the deviations file, both code-readings files,
`lite_split_national2b.json`, `labels_national2b.parquet`, the P2 index, `lite_p2_lock.json`, the C2 csv + meta,
`FREEZE_lite.json`, `tests/test_lite.py`. A new reading or deviation is a NEW dated file.

What was written for the order (all outside every frozen hash list):
- **A1 Test-S KD mask on the re-run path** — `sentinel/models/litev2/launch.py` `activate_rerun_kd_mask()` wraps the
  frozen `lite.distill.load_kd_tensors` in the re-run process (`scripts/train/national2b_rerun.py` `activate()`, target
  `train` only): masked chips get `t_w = 0`, `t_mu = t_var = NaN`. Checked on training-role keys: 308 masked training
  chips in both views (rerun view 2,699 paired chips, v2 view 2,740). Third and last exception to "unchanged
  settings".
- **A2 warm start** — `launch.init_checkpoint(kd, seed)` is the only way a v2 job gets a leader checkpoint: exactly
  one finished `natstudent_mnv4s_raw_crop256_kd03_national2b_<time>` (KD) or `…crop256_nokd_national2b_<time>` (twins)
  directory, registered sha256 (`launch.register_init_checkpoints`, experiment `lite/v2_init_checkpoints`), no
  fallback; anything else is `LaunchError`. `--init-run` is optional and only checked. S11 embeddings use the same
  checkpoints (`litev2_train.py --arm embed`; `--group stacked` refuses another embeddings directory).
- **A3 Q5 tests** — `sentinel/models/litev2/checks.py`: `assert_refit_clean` (no val / val_n / val_p2 / test-role chip
  or water body in the refit table; cross-track: 2 km and HydroLAKES id), `assert_one_grouping` (no `train_p2` row
  within 2 km of, or sharing an id / HydroLAKES id with, a national-2b training row). Real keys (key columns only):
  0 chips, 0 water bodies; national train vs P2 held-out min 2.88 km; `train_p2` vs national train min 3.30 km.
  `test_time` / `test_time_n` are later dates of training water bodies by design: chips are disjoint, the water-body
  overlap is counted and reported (1,761 training rows in 221 water bodies), not asserted (Q5: separate stratum).
  `sentinel/models/litev2/anchor_store.py`: ONE anchor artefact `$SENTINEL_RUNS/lite/litev2_anchors/anchor_model.pkl`
  (+ `anchors_train.parquet` out-of-fold), fitted once on the training rows and registered (experiment
  `lite/v2_anchor_artefact`, `extra.anchor_sha256`) before a validation row is loaded; `eval_anchors(block)` writes
  `anchors_<block>.parquet` once; S6 = that table (`s6_predictions`); every reader checks the hashes; never rebuilt.
  Ablation A4 keeps its own reduced-row anchor in its run directory, now hash-checked (`a4_anchor_sha256.json`).
- **A4 stricter order** — `harness.require_gate(name)`: a candidate (anything not in `H.COMPARATORS`) is not FITTED
  and not evaluated before every comparator validation row of both blocks is registered. Called in `main`,
  `stage_fit` and `stage_eval` of `scripts/train/litev2_train.py`, before a row, run directory or ledger line.
- **A6 E3a_turb** — `checks.e3a_turb_count()` (stamp-guarded, one-shot, registered under
  `lite/e3a_turb_count_national2b`): positives and water bodies of `val_n` by the frozen `sentinel.events.lite.labels`,
  in both label views; < 30 positives or < 10 water bodies with a positive in either view → descriptive only.
- **A7 GPU-hours** — `launch.gpu_job(name, block, budget_h)`: append-only ledger
  `$SENTINEL_RUNS/lite/litev2_gpu_ledger.jsonl` (flock), ceiling 45 GPU-h, job cap 2 h, blocks = the addendum's compute
  table (`v2_networks, densenet, reruns, slot_twins, tabpfn, s11_embed`); an open job counts at its full budget, a
  finished one at its measured hours; a job refuses to start if logged total + own budget > 45, refuses GPU 2, and is
  stopped (SIGALRM) at its budget. It also refuses a job that was not started through the lock helper (`CUDA_VISIBLE_DEVICES` must be one GPU index) — the frozen runner prefers GPU 2 when it picks a GPU itself. Budgets: re-run 1.5 h; v2 network / twin 0.5 h per seed; DenseNet 2.0 h per seed
  (one seed per job); eval 0.25 h; embed 0.5 h; TabPFN memory test 0.5 h, S10 0.75 h per block
  (`--gpu-budget-h` on every script).
- One-shot steps: `scripts/train/litev2_steps.py row-checks | ledger | register-init | e3a-count`.
- **A5 (rerun-view path in `litev2_comparators.py`) is NOT written** — "do it last, may follow later"; needed before
  the national freeze only; write it after the stamp row exists (it then needs the lead's re-hash).
- Tests, run 20:41–20:53 EST: 120 passed, 0 failed in the 11 files `tests/test_litev2.py` (11), `_anchor_export` (6), `_anchor_store` (5), `_checks` (6), `_export` (9), `_harness` (7), `_launch` (14), `_national_closure` (2), `_nets` (28), `_stamp` (19), `_tabular` (13); the known failure `tests/test_lite.py::test_scoring_hash_covers_scorer_import_closure` still fails (record of deviation 10; not touched). New: `tests/test_litev2_launch.py` (A1, A2, A4, A7, steps), `tests/test_litev2_checks.py`
  (A3 i, ii, A6; two tests read key columns of the real tables, no label column of a validation or test role),
  `tests/test_litev2_anchor_store.py` (A3 iii).
- Code hashes for the stamp: `$SENTINEL_RUNS/lite/counts_arch_v2/litev2_code_sha256_20261110.json` (0444).

Intended rulings B (bind when in the rulings file): INT8 carrier = `scope="image"` (INT8 image branch; FP32 context,
pooling, heads, anchor sum), fixed before any trained model; `scope="graph"` also exported, metrics descriptive only;
every write-up says "mixed precision", never "fully INT8". Export parity = the three conditions of the export
readings, thresholds fixed now, never loosened after a failure; in addition report on real validation rows the count
> 1e-4 and the max of the float64-vs-float32 anchor difference (reported, not a pass condition). ANCHOR-Set stays
as written (307,341 parameters). 12 DenseNet runs are inside the ceiling.

Selection rule S of configs 1–6 used the 3-seed ensemble (mean μ): `scripts/eval/lite_select.py:50–52` reads
`summary.json["scores"]`, written by `scripts/train/lite_train.py:112–126`; registry experiment `lite_val_national2`,
models `lite_student_<cfg>_national2` (notes "3-seed ensemble (mean mu) …"), e.g. config 3 rows `5ec18d2e1078`,
`2008b92920f9`, `22f37bfdefbf`, `3d6035d7bc8b`. The selection itself has no registry row; it is the file
`$SENTINEL_RUNS/lite/selection_national2.json`.

**Launch order after the stamp (replaces (a)–(c) below).** Every GPU job:
`SENTINEL_GPUS=0,1,3,4,5,6,7,8,9 python -m sentinel.core.gpu --name <job> --min-free-mb 10000 -- python -B …`
(never GPU 2, ≤ 2 h, ≤ 6 CPU threads; CPU load is above the core count — prefer GPU work). Report to the lead at once
(job name, GPU, PID, start time) when the first job starts.
1. GPU: `scripts/train/litev2_tabpfn_memtest.py --register`; re-runs `scripts/train/national2b_rerun.py train --config
   configs/lite/<cfg>_national2.yaml --track national` — config 3 (`mnv4s_raw_crop256_kd03`) and its twin
   (`…crop256_nokd`) first, then `crop128_kd03`, `crop192_kd03`, `crop128_kd03_ssl`, `crop256_kd03_ssl`,
   `crop256_kd03_ctx`, `crop256_nokd_ctx`; 12 DenseNet fits (`litev2_train.py --arm P3a_densenet|P3b_densenet_parity
   --backbone-lr 1e-4|3e-4 --seeds <s> --stage fit`). CPU (one job): `litev2_comparators.py --group tabular` (builds
   and registers the anchor artefact), then `national2b_rerun.py baselines --track national`.
2. When the config-3 re-run and its twin have finished: `litev2_steps.py register-init`; then
   `litev2_train.py --arm embed` (GPU) and `litev2_comparators.py --group stacked --embeddings
   $SENTINEL_RUNS/lite/litev2_embeddings/<re-run dir>` (CPU); `--group tabpfn` (GPU); DenseNet `--stage eval` × 4;
   `litev2_steps.py e3a-count` once.
3. Only when `H.missing_comparators` is empty for both blocks: candidates `--arm anchor_lite` (slot 7), `--arm A1`
   (slot 8), both also `--no-kd`, `anchor_set`, `A3`, `A4`, `A5` (`--stage fit`, then `--stage eval`). Selection rule S
   once on the corrected validation rows; as-built and corrected tables both reported. Slots 7–8 cut-off
   2026-11-16 23:59 EST by registry timestamp.

## Architecture round v2 — state at 2026-11-10 19:49 EST (older; see the section above first)

**Nothing of v2 is fitted. No validation number exists on national-2b, P2, any v2 arm, comparator or re-run.**
The guard (`sentinel.models.litev2.stamp.stamp_status()`) says "not stamped" until the lead writes
`docs/preregistration/lite_arch_v2_lead_rulings.md` and the registry row (workstream `M7-lite`, experiment
`lite/prereg_snapshot`, `extra.written_by = "lead"`, `extra.sha256` of the four files, all four 0444).

Governing orders: lead order 2026-11-10 15:05 EST (confirmed 18:45), lead ruling 18:45 EST, lead messages 19:38 and
19:42 EST, M2 stamp rule (R-1 / C-1 of `docs/preregistration/orbit_arch_v2_lead_rulings.md`): until M2S/test is
sealed nobody scores a Sentinel-2-input model on a row of an M2 test-role water body dated ≤ 2023-12-31.
- 19:38: **no fit of any kind on real rows before the stamp, TabPFN memory test included**; synthetic-tensor tests are
  fine. A new deviation entry goes in a NEW file, never appended to `lite_arch_v2_deviations.md`.
- GPUs: lock helper for every job, ≤ 2 h per job, **never GPU 2** (another user's). The cap file
  `$SENTINEL_RUNS/.gpu_locks/ALLOWED` lists GPU 2, so every launch sets `SENTINEL_GPUS=0,1,3,4,5,6,7,8,9`.
  CPU is tight: one CPU-heavy job at a time, ≤ 6 threads.

Stamp package (sent to the lead 19:03 EST):
- `docs/preregistration/lite_arch_v2_amendment.md` v0.3, sha256 `2b369018…8a56` — FROZEN, never edit.
- `docs/preregistration/lite_arch_v2_label_addendum.md` (counts only, items i–x, power restated, C2 rule) — sha256
  `bcbd86de12ff116100f216f5c7c3c5de2e8f1de4d7bb567fd2badf5e12d29049`. Do not edit; a change is a new dated file.
- `docs/preregistration/lite_arch_v2_deviations.md` (entries 1–11) — sha256
  `23e6fd8705fe4bfc102df74160e46477c8c6b31b3ee38ff504370ed3fc00642c`. The lead stamps this byte state: do not edit
  or append.
- Pin `docs/preregistration/lite_split_national2b.json`, sha256 `03f0114a…5663`.
- Registry rows: `6bebe3649244` label version, `3ea898947cbd` P2 index, `e89036650629` F7, `a7349f26339f` mask,
  `68dfce1e2a72` stratum + reserved counts, `2e952b94162f` S9 list, `e7ebd23c8ede` DenseNet121 file.
- Count files (0444): `$SENTINEL_RUNS/lite/counts_arch_v2/{labels2b_counts,frozen_rederive_stopcheck,stamp_counts,addendum_counts}.json`;
  C2 keys csv `c2_teacher_scored_m2_test_lake_keys.csv` (sha `4786aa12…c699`, pinned in the M2 stamp — never edit,
  regenerate or move).

Sent to the lead 19:48 EST:
- Code-readings notes `docs/preregistration/lite_arch_v2_code_readings_20261110.md` (0444), sha256
  `002d5d0822dab92d725467f59966b106b866c1c958855b801fef330aaf22a4d4`. Section 7 = the four readings that touch a
  stamped number or sentence (ANCHOR-Set 307,341 parameters vs "about 0.15 M"; 12 DenseNet runs ≈ 14 GPU-h; the gate
  covers candidate validation, not candidate fits; A1 keeps the Δ² penalty and ANCHOR-Set has no radiometric
  augmentation). The lead decides whether any becomes a deviation entry (new file).
- Deviation 10, national list fixed on the lead's 19:42 order: `sentinel/models/lite/national_testlook.py`
  (sha256 `16546bb9…6d4f`) — `NATIONAL_CODE` now also hashes `sentinel/data/matchups.py`,
  `sentinel/models/orbit/wqp_surface.py`, `sentinel/models/litev2/__init__.py`. New test
  `tests/test_litev2_national_closure.py` (sha256 `7b3d619e…c415`): closure of the national scorer ⊆
  `national_frozen_files()`. `guard.SHARED_CODE`, `FREEZE_lite.json` and the failing 4-state test
  (`tests/test_lite.py::test_scoring_hash_covers_scorer_import_closure`) are NOT touched — the failure is the record
  of deviation 10.
- Launch order (below) and one question: may candidate fits run beside the comparators (validation still gated)?
  Default until answered: comparators and re-runs take every GPU first.

Sent to the lead 20:07 EST:
- Export-readings notes `docs/preregistration/lite_arch_v2_code_readings_export_20261110.md` (0444), sha256
  `895dcefe314937bc21077e2c34d78cf2252fcee0a143b1ff4a4fb8957eb5454f` (readings X1–X12). Two rulings asked:
  (f) INT8 scope of the carrier file — `scope="image"` (normalisation + encoder + decoder INT8, context MLPs, pooling,
  heads and the final anchor + residual sum FP32; my default) or `scope="graph"` (everything INT8; literal reading).
  **No trained v2 model is quantised until the lead rules.** (g) meaning of the §7 "passing export-parity test"
  (proposal: ONNX FP32 vs training code < 1e-4 log10 on every validation row; anchor file vs LightGBM on float32
  features < 1e-4; INT8 output finite; INT8-vs-FP32 metric difference reported, not a pass condition).
- Synthetic / random-weight numbers only: anchor file 174,000 numbers, 1.04 MB; ANCHOR-Lite 1,401,296 exported
  parameters, 1.08 GMACs/512², INT8 file 1.99 MB (image) / 1.81 MB (graph); pipeline ≈ 1.58 M parameters, ≈ 3.03 MB.

Open points waiting for the lead's rulings file (deviation 11): (a) F4 KD mask in the "unchanged" re-runs (default:
unchanged); (b) warm-start of slots 7–8 from corrected-label re-runs (the lead expects yes, seed-matched — wait for
the binding text); (c) compute ceiling 45 GPU-h (estimate ≈ 29); (d) N3 event counts recounted after the stamp;
(e) DenseNet121 licence finding.

Code of the round (`sentinel/models/litev2/`):
- labels and guards: `wqp_rule.py`, `labels2b.py` (`role_frame(view='rerun'|'v2')`, `activate_rerun`), `p2.py` +
  `p2_index.py`, `tests_mask.py` (Test-S KD mask), `stamp.py` (guard). Builders in `scripts/data/litev2_*.py`.
- `rows.py` — `train_rows()` (10,651 rows, 3,222 water bodies), `eval_rows(block)` (stamp-guarded), `chips`,
  `scene_table`.
- `tabular.py` — anchor (= S6), S1–S11, P1, P2a/b, TabPFN keep rule.
- `harness.py` — feature store, teacher means and KD tensors (F4-masked, training roles only), block scoring,
  registry rows, comparators-first gate (25 comparators).
- `nets.py` — `AnchorLite` (+ switches for A1/A3/A4/A5), `WaterSet`, `DenseNetRegressor`.
- `fit.py` — arm configs (`ARMS`), tensors, augmentation, loss (β-NLL + Tobit + NTRU offset + KD + Δ²), training,
  prediction, ensemble.
- Scripts: `scripts/train/national2b_rerun.py` (re-runs; `national2_run.py` untouched),
  `scripts/train/litev2_comparators.py` (`--group tabular|tabpfn|stacked`),
  `scripts/train/litev2_tabpfn_memtest.py`, `scripts/train/litev2_train.py` (`--arm … --stage fit|eval`, `--no-kd`,
  `--backbone-lr`, `--init-run`, `--arm embed`). All refuse before the stamp.
- Tests (synthetic only): `test_litev2.py` + `test_litev2_stamp.py` (30), `test_litev2_tabular.py` (13),
  `test_litev2_harness.py` (7), `test_litev2_nets.py` (28), `test_litev2_national_closure.py` (2).
- Built without a fit: `$SENTINEL_RUNS/lite/cache/v2train_50c611c100e4.npy` (training chips),
  `$SENTINEL_RUNS/lite/litev2_features/train_<key>.parquet`. A national-validation chip cache (`v2val`, ≈ 0.8 GB) is
  built only after the stamp.
- Export (new, outside every frozen hash list, synthetic tests only): `anchor_export.py` (compact float32 anchor
  file, `export_anchor` / `load_anchor` / `predict_anchor`; `tests/test_litev2_anchor_export.py`, 6) and `export.py`
  (multi-input ONNX wrapper of the ANCHOR-Lite family, `export_onnx`, `quantise_int8(scope=)`, `onnx_predict`,
  `count_macs`, `count_params`, `latency`; `tests/test_litev2_export.py`, 9). Frozen `lite/export.py` is imported
  only.
- Not written yet (needed only at freeze, after trained models exist): the export script that runs these on the
  seed-0 model of the selected configuration (pipeline latency incl. feature code and trees, INT8-vs-FP32 report,
  budget table, registry row), and an export wrapper for ANCHOR-Set (needed only if it is the P2 selection).

**After the stamp (not before).** Report to the lead at once (job name, GPU, PID, start time) when the first job
starts. Every GPU job: `SENTINEL_GPUS=0,1,3,4,5,6,7,8,9 python -m sentinel.core.gpu --name <job> --min-free-mb 10000 -- …`.
- (a) At once: `scripts/train/litev2_tabpfn_memtest.py --register` (GPU). On CPU:
  `scripts/train/litev2_comparators.py --group tabular` (S1–S9, P1, P2a/b; writes the anchors), then
  `scripts/train/national2b_rerun.py baselines --track national`.
- (b) GPU queue in priority order:
  1. `national2b_rerun.py train --config configs/lite/mnv4s_raw_crop256_kd03_national2.yaml --track national`
     (config 3), then `…crop256_nokd_national2.yaml` (≈ 35–40 min each);
  2. the other six: `crop128_kd03`, `crop192_kd03`, `crop128_kd03_ssl`, `crop256_kd03_ssl`, `crop256_kd03_ctx`,
     `crop256_nokd_ctx`;
  3. 12 DenseNet fits, one seed per job: `litev2_train.py --arm P3a_densenet|P3b_densenet_parity --backbone-lr
     1e-4|3e-4 --seeds <s> --stage fit`;
  4. `litev2_comparators.py --group tabpfn`; `litev2_train.py --arm embed --init-run <config-3 re-run>`, then
     `litev2_comparators.py --group stacked --embeddings <dir>` (CPU);
  5. DenseNet `--stage eval` × 4.
- (c) Only after the config-3 re-runs exist and the anchors are written: `--arm anchor_lite` (slot 7), `--arm A1`
  (slot 8), both with `--no-kd` (from the no-KD re-run), `anchor_set`, `A3`, `A4`, `A5` — `--stage fit` first;
  `--stage eval` only when the gate passes. Then selection rule S once on the corrected validation rows; as-built and
  corrected tables both reported. Slots 7–8 cut-off: 2026-11-16 23:59 EST by registry timestamp.


## Status
- **4-state single look done:** 2026-11-09 22:05–22:30 UTC under `FREEZE_lite.json` (sha `a066b57a…`; scoring code
  `a7a8be82…`).
  - Outputs: `$SENTINEL_RUNS/lite/test_look/` (sealed predictions, `test_results.json`, SCORED.json).
  - Registry: `lite_test*`.
- **Result: H-G6 FALSE, label "negative".**
  - Secchi ensemble 54.4 vs GBM-RGBN 44.2: Δ +10.2 [+4.4, +14.2]; ties band-ratio.
  - H2 did not replicate. Budgets met. Glint fragile.
- **Write-ups:** `docs/preregistration/lite_test_results.md` (new), RESULTS.md (dated entry), model card "Test
  results". The post-results review was PASS WITH CONCERNS; P1–P7 were applied on 2026-11-09 (exact reviewer text). E4a is reported as descriptive only. The `"confirmatory": true` role label in `events.E4a` of the JSON
  is stale and unread.
- **Never** edit `lite_protocol.md`, FREEZE files, the scorer or any hashed file. Post-test text goes in
  `*_test_results.md`.
- **Next:**
  1. the lead commissions the post-results review;
  2. after it passes, start the SOTA plan steps (below): counts-only checks first;
  3. the national-track protocol file with T-X strata, before the national freeze (decision (b): hold the national
     freeze for levers 1–3). Include **≥ 10 chips per target in each slice** for any interpretation (post-results
     review).
- Nothing is running. M7 holds no GPU and no CPU job.

## Pre-test review (2026-11-09): FAIL → fixed (REVIEW_REPORT_PRE_TRAINING.md, "M7 Lite pre-test review")
- **C1:** single look in `sentinel/models/lite/testlook.py`. It provides a registry refusal, O_EXCL 0444
  `TEST_LOOK.json` plus a claim row, sealed predictions (0444 + SHA256SUMS), `SCORED.json` and
  `--rescore-from-saved`.
- **C2:** `guard.check_access` accepts only `FREEZE_lite.json` with status FROZEN, the authorisation and an open
  claim. Tested by `test_candidate_never_unlocks_test`.
- **C3:** data pins (`testlook.compute_data_pins`): NAIP shards + index, S2 shards + index, scene stats, train
  cache, labels, the two manifests. The train cache was verified byte-identical to a fresh store decode. The
  `naip_matchups` manifest was rewritten 2026-11-08, with the old hashes kept.
- **C4:** `evaluate.compare_one_sided` (correct-tail p) and `holm_with_ci` (Holm ∧ CI excludes 0). The composite
  H1 / H1-deployable / H2, the seed rules, H-G6 and the label are computed in code. Families run on Test-A only;
  Test-B is descriptive. Pre-declared in protocol §8a (v1.4).
- **C5:** one pass predicts every model and baseline, including the INT8/FP32 ONNX deployable (hash-checked), the
  robustness grid and the teacher. All per-row predictions are saved and sealed.
- **W1:** no-KD re-run `mnv4s_raw_crop128_rerun`. **W6:** RGB-only ablation `mnv4s_raw_crop128_kd03_rgb`. Both are
  queued as one GPU job (`logs/gpu_w1w6.sh`, lock helper on GPUs 0/1, waiting for a slot). If the GPU stays busy,
  run them on CPU (`--device cpu`, about 2 h each).
- **W2–W5, W7:** done (p tail; B and device from the freeze; `pool.json`, teacher manifest and export report hashed;
  code hash = import closure, tested, with national files excluded; package versions; every-baseline paired
  comparisons).
- **W6:** done, except the RGB ablation still training.
- **W8:** `scripts/eval/lite_teacher_overlap_wb.py`. Test-A: 0 water bodies ≤ 2 km, 85/420 at 2–5 km, 7/31 flowing
  within 5 km. Secondary sensitivity added (`teacher_gt5km_wb`).
- **W9:** Test-A ∪ Test-N dropped (§12).
- **W10:** 16 configurations everywhere (`selection.json` regenerated); TRAINING_LOG entry for `221015`; wording
  fixed.
- **W1 result:** the re-run reproduces the old no-KD run exactly, so H2 on val is unchanged. **W6 result:** RGB-only
  gives 45.4 / 73.6 / 100.1 vs RGBN 46.6 / 72.6 / 107.1. NIR adds nothing measurable on val.
- **National track:** the KD student is selected (`natstudent_mnv4s_raw_crop128_kd03_nat_20261109-092920`, S 0.987;
  `selection_national.json`). Its freeze is separate and later; it also needs a national `testlook` look.
- **Event track (protocol §13, v1.5):** `sentinel/events/lite.py` and `scripts/eval/events_lite.py` (val stage),
  plus event scoring inside `lite_score_test.py`. It is **waiting for `sentinel/eval/events.py` v1** (the lead
  messages when it is released; verify its sha256 first). Then:
  1. run `events_lite.py`;
  2. adapt the calibrator serialisation (`Calibrator.to_dict` / `from_dict`) and the event registry rows to the
     real API;
  3. rebuild the candidate;
  4. run the full dry run.
- **Candidate builder:** `scripts/eval/lite_freeze_candidate.py` (status CANDIDATE, `scoring_code_sha256` PENDING).
- **Next:**
  1. when W1/W6 finish, re-run the builder;
  2. run the full `lite_score_test.py --dry-run-val --candidate docs/preregistration/FREEZE_lite_candidate.json`
     (CPU);
  3. report to the lead for the delta re-review.

  Do not write `FREEZE_lite.json` and do not score test.

## Queue history (2026-11-08/09; all finished; superseded by "Status" above)
GPU budget is GPUs 0/1 only (shared, ≤ 2 h per job, test/freeze-critical jobs first), so M7 runs on CPU
(`--device cpu` added to `lite_train.py`, `lite_robustness.py`, `lite_teacher_litesafe.py`, `lite_teacher_targets.py`).
- `$SENTINEL_RUNS/lite/logs/queue_cpu1.sh` (bash PID 915279): the robustness check on the selected model is done; the
  national-track student `mnv4s_raw_crop128 --track national` (≈ 3.5 h) is running. When it finishes it writes
  `logs/queue_cpu1.done`.
- `$SENTINEL_RUNS/lite/logs/queue_cpu2.sh` (bash PID 918948) waits for `queue_cpu1.done`, then runs:
  1. robustness of the median-pool variant;
  2. the corrected ONNX/INT8 export of all 3 seeds of the selected run (the new rows supersede the earlier export
     rows);
  3. the national Lite-safe teacher (5 seeds, CPU);
  4. national teacher targets;
  5. the national KD student `mnv4s_raw_crop128_kd03_nat`.

  It finishes with `logs/queue_cpu2.done`. Results are reported as `OK`/`FAIL` lines in `logs/queue_cpu{1,2}.out`.
  If the host reboots, re-run the missing steps from these scripts.

## State
- **Protocol:** `docs/preregistration/lite_protocol.md` v1.3. Changes: the W8 result and the Lite-safe teacher (§5);
  deviation D1 (turbidity/TSS scored on all water bodies, Test-A lake counts 58/62) in §3 and §11; the national track
  in §12.
- **Splits:**
  - original: `lite_split.json`, sha256 `46ed6e1f…7475`, unchanged;
  - national: `lite_split_national.json`, sha256 `8e3b1ee00c8a5d2b6c9f9b291c15ae0b3d329abaa98c1d22887f2ec874cf7d19`,
    pinned in `guard.FROZEN_NATIONAL_SPLIT_SHA256`. It was frozen before any model was selected on it, with a power
    check in §12.
- **Data:**
  - `naip_matchups_national` (`processed/naip_matchups/national/`, datasheet
    `docs/datasheets/naip_matchups_national.md`);
  - `lite_labels_national`.
  - The canonical 4-state files and manifest were restored byte-identical; the snapshot is in
    `processed/naip_matchups/snapshot_4state_20261024/`.
  - The chip stores are supersets with unchanged ids. As a result, the old `naip_matchups` manifest's index hashes
    no longer match. **Tell the data workstream.**
- **New code:**
  - `distill.py`: W8 overlap check (`lite_sites`, `orbit_seen_points`, `flag_overlap`, `overlap_report`,
    `assert_teacher_disjoint`, `litesafe_drop_mask`);
  - `national.py`;
  - `losses.pool_maps_median`;
  - `train.py`: `train_targets`, KD with crop, CPU path;
  - `scripts/train/lite_teacher_litesafe.py`, `lite_split_national.py`;
  - `scripts/eval/lite_select.py`, `lite_repool.py`;
  - `lite_compare.py`: `--subset`, national track;
  - `lite_export.py`: crop-consistent ONNX check, per-seed registry model names.
- **Tests:** `tests/test_lite.py`, 23 passing.
- **Lite-safe teacher (original track):** `$SENTINEL_RUNS/lite/teacher_litesafe/manifest.json` (5 seeds, sha256).
  Teacher targets are in `teacher_targets_litesafe/` (symlink).

## Results (val only; see `docs/model_cards/lite.md`)
- **Configurations tried, original track: 14.** That is the 8 from October, plus kd03, kd1, ep20, lrbb2e4, secchi-only
  and medpool.
  - 4 runs crashed on a code bug (`TARGETS` shadowing in `train.py`, now fixed) and were re-run. They are logged as
    failed in TRAINING_LOG.
  - The partial October run `…221015` was deleted.
- **Selected** (pre-registered S rule, `$SENTINEL_RUNS/lite/selection.json`): `student_mnv4s_raw_crop128_kd03_20261108-152950`,
  S = 0.868. Val MdSA: Secchi 46.6, turbidity 72.6, TSS 107.1.
- **H-G6 on val: not met.**
  - Secchi vs GBM-RGBN: +5.4 [+0.5, +9.6] (student worse).
  - Secchi vs band-ratio: −14.4 [−20.4, −9.1].
  - Turbidity/TSS: all CIs include 0.
- **H2 (KD) on val:** Secchi −2.6 [−5.2, −0.4], p1 = 0.01. Turbidity −4.8 and TSS −8.0, with CIs including 0.
- **Robustness:** the glint soft threshold fails (Secchi +124% at 25% glint). Haze and gain pass.
- **Export (per seed):** 1.32 M params; 1.13 GMACs/512²; INT8 1.65 MB; INT8 latency 32–41 ms at 1 thread (desktop
  proxy). The 3-seed ensemble exceeds the GMAC and INT8-size budgets.

## What to do next (2026-11-08 list; items 1–3 are done, see Status)
1. When the queues finish:
   - run `lite_select.py --baselines $SENTINEL_RUNS/lite/natbaselines_20261108-142125 --runs <natstudent_*> --out
     selection_national.json`;
   - run `lite_compare.py --student <natstudent> --baselines <natbaselines>`, with `--subset all`, `val` and `val_n`;
   - run `lite_compare.py --student <natstudent_kd03_nat> --baselines <natbaselines> --other <natstudent_crop128>`
     (H2, national).
   - Also: national-trained students on the frozen 4-state val (`--subset val`) against both the original and the
     national GBM.
2. Fill the `[PENDING]` items in the model card: INT8 Δ, medpool robustness, national track.
3. Freeze candidate: see the model card. **Decision needed from the lead:** deploy one seed (meets the budget) vs the
   3-seed ensemble (the protocol §4 headline, but 3× compute).
4. Ideas not yet tried: glint augmentation or a glint mask (aimed at the robustness failure); Secchi-only + KD; the
   national KD teacher's effect.

## Open issues
- **W8:** resolved through the Lite-safe teacher. The stock M2 teacher must never be used for Lite KD or for the
  red-edge gap. Note for M2: 545 Lite train sites lie in Orbit test-role areas. This matters only if Lite outputs
  ever feed back into M2 or Fuse.
- Low turbidity/TSS power on Test-A is the reason for the national track.
- Scene-mode export lacks scene-FiLM support (not selected, so not a blocker).

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.

## 2026-11-09 (lead): M7 Lite frozen; the single test look started
- **Review:** E-patch check PASS (REVIEW_REPORT_PRE_TRAINING.md "M7 Lite — E-patch check (2026-11-09)").
- **Freeze:** `docs/preregistration/FREEZE_lite.json` (0444) is the candidate plus status FROZEN, `scoring_code_sha256` a7a8be82… (re-checked as current), authorised_by/at, candidate_sha256 and `recovery_note_F1`. `testlook.load_frozen()` is OK.
- **Archive:** `sentinel_runs/freeze_archives/lite_freeze_20261109T170530.tar.gz` (82 files); registry row `lite/prereg_snapshot`.
- **Test look:** the lead started `lite_score_test.py --allow-test` (sentinel2, CPU). Log: `sentinel_runs/lite/logs/lite_test_look_20261109.log`.
- **Do NOT edit any hashed file** until the look is done and reviewed. That includes `sentinel/eval/{events,hazards,ew_scoring,early_warning}.py`. National-track work must leave all of them alone.

## SOTA plan (2026-11-09, per docs/plans/sota_campaign.md; nothing new runs until the lead has reviewed it)
Context:
- The 4-state test look is running or done under the lead. No hashed file is touched until it has been reviewed.
- The national-track freeze stays on its current path.
- All new work uses new files and new run dirs.

### 1. Diagnosis: why M7 is not SOTA
Facts (val):
- GBM-RGBN beats every student on Secchi by 5–8 MdSA points, on 4-state and national val alike.
- GBM + lat/lon/DOY is better still (35.4). A location/season prior carries a lot of the Secchi signal; the drone
  student, by design, has no location input.
- More data (national train ×1.7) helped GBM and the student about equally.
- KD gives about −2.5 points on Secchi. RGB-only ≈ RGBN.
- Turbidity and TSS have little skill for any model on the 4-state val (power-limited). On national val there is
  real turbidity skill vs the median.

What the CNN has that the GBM lacks, and whether it is used:
- **Spatial context and texture.** The CNN sees a 38–128 m crop. The GBM sees centre 9×9 / 31×31 medians,
  dark-NIR water percentiles and ratios. Crop 128 beat the full 256-px chip, so wider context has *not* helped so
  far. Likely reasons:
  - the GBM's medians are already a near-sufficient statistic for an optically deep, homogeneous water body;
  - at 2.8 k chips, the extra pixels add variance faster than signal.
- **Shoreline, bottom and depth cues.** Visible bottom (clear, shallow water), littoral vegetation and lake size or
  shape are plausibly informative for Secchi. They need more context than 128 px at 0.6 m (77 m). This is
  untested: no multi-scale or coarse-context input has been tried.
- **NIR.** NAIP NIR over water is near the 8-bit floor (vendor stretch). It carries little beyond red, which
  matches RGB-only ≈ RGBN.
- **Radiometry.** The GBM uses per-DOQQ scene-normalised features. The student's scene-normalised variant was the
  worst. The CNN has to learn invariance to per-state-year colour balance from few chips. This is a data-size
  problem the GBM sidesteps with hand-made normalisation.
- **Bottom line.** With today's inputs, the CNN has *no information the GBM lacks* except context it does not yet
  exploit. More capacity alone will not help (plan rule 5). The honest levers are therefore:
  1. more labelled data (both models get it);
  2. unlabelled-data pretraining (the only lever the GBM cannot use);
  3. context the GBM's per-pixel features cannot encode;
  4. tasks where per-pixel medians should fail by construction.

### 2. Levers, ranked by expected gain per GPU-hour
1. **More labelled data (CPU/network, ~0 GPU; comparator parity: GBM refit on the same rows).**
   - (a) **Temporal window for Secchi.** Lake Secchi is slow-varying, so widen the NAIP–sample window from ±1 d to
     ±7 d, for Secchi only, as a pre-declared new track.
     - Expect several times more Secchi matchups (count-only check first).
     - Turbidity and TSS stay at ±1 d (fast-varying).
   - (b) **LAGOS-US LIMNO Secchi** (EDI, CC-BY). It holds many lakes outside WQP, especially citizen monitoring.
     Also NARS lakes (NLA, local). Both have a datasheet and licence. They are matched to existing NAIP DOQQs.
   - (c) **GLORIA as a radiometric prior:** the existing GLORIA spectral pretraining. It did not help before
     (S 0.986), so it is low priority.
   - **Disk:** chips about 100 KB each, so 20 k new chips ≈ 2 GB. That is within the 10 GB cap; the size is
     reported first.
2. **Self-supervised pretraining on unlabelled NAIP water chips** (GPU, the main capacity lever the GBM cannot use).
   - Corpus: chips at HydroLAKES / JRC water locations across all states and years 2010–2023, excluding every test
     water body of every Lite track (the coordinates-only check, as in W8). Target 30 k chips ≈ 3 GB.
   - Method: masked-image / contrastive pretraining of the same MobileNetV4-S encoder, with
     radiometric-invariance augmentation (per-state colour jitter). This addresses the radiometry diagnosis.
   - Then fine-tune exactly as now.
   - Budget: about 6–10 GPU-h in ≤ 2 h chunks on GPUs 0/1.
3. **Context / multi-scale input** (cheap GPU or CPU). Feed a second, downsampled wide-context view (e.g. the full
   256 px plus a 4× coarser neighbourhood from the DOQQ, or 512 px at native resolution, pooled) alongside the
   centre crop. This tests the shoreline, bottom and lake-size hypothesis. 2–3 configurations.
4. **Harder, drone-relevant tracks** where per-pixel medians should fail by construction (see §3).
5. **Stronger teacher:** low priority. The Lite-safe teacher is weaker than stock, and KD gains are about
   2.5 points. A better M2 teacher, retrained Lite-safe, would help marginally. Only after 1–3.

Not proposed:
- larger backbones (rule 5);
- adding location inputs to the student (it would no longer be a drone model; GBM + context already exists as a
  comparator);
- re-tuning on spent blocks.

### 3. Track definitions and unseen test blocks
All tracks are pre-registered with justification **before** any val result on them. Originals are never replaced.
Each track gets its own testlook look with the v1.7 hardening.
- **T-N (national, existing):** Test-N / test_time_n. Frozen soon and never scored. It receives levers 1–3 only if
  its freeze has not happened yet; otherwise the improved model goes to T-F.
- **T-F (future NAIP cycle, unseen):**
  - Block: NAIP acquisitions newer than anything in our catalog (the catalog holds 2010–2023 only).
  - **Availability check first:** query Planetary Computer / USDA for NAIP 2024–2025 coverage. Only `naip:year`
    ≤ 2023 is cached.
  - WQP samples after 2024-01-31 matched to them. This also yields new samples for existing states.
  - Power check: label counts by target and water body, computed before any model sees them.
- **T-S (unseen states, out-of-region transfer):**
  - Justification: a drone deployed in a new state meets new vendor colour balance and limnology.
  - Block: states with no Lite chip at all. Today every state with WQP data is in the national build. So T-S needs
    new label sources (LAGOS/NLA lakes in states with sparse WQP) or a leave-states-out split defined only on new
    data.
  - Defined after the data audit and before any result.
- **T-X (a-priori difficulty strata within T-F / T-N, §15.5 item 5):**
  - (i) small water bodies (< 10 ha or not in HydroLAKES): farm and stormwater ponds, the drone use case, where
    the GBM's centre medians mix in shoreline;
  - (ii) turbidity / TSS above the training P90 (plume conditions);
  - (iii) Secchi below the training P10 (eutrophic).
  - Strata are fixed from train distributions only.
- **Glint/haze robustness as a track: NOT eligible as new.** Val glint results already exist for the student and
  the robust variant (rule 2). It is reported as a robustness comparison, including GBM under the same
  perturbations, scoped and descriptive.

Power rule: a track is run only if its unseen block has ≥ 30 positives / ≥ 10 water bodies per inferential target
(event plan §1.2), computed on counts only. Otherwise it is descriptive.

### 4. Caps and ETA
- **Caps:**
  - ≤ 8 student configurations per new track, ≤ 3 seeds each;
  - ≤ 3 context configurations;
  - ≤ 2 SSL pretraining runs;
  - baselines refit on identical rows (no tuning beyond the fixed hyper-parameters);
  - every configuration counted in TRAINING_LOG / registry.
- **Compute:** GPUs 0/1, ≤ 2 h per job; SSL about 6–10 GPU-h total; students on CPU (about 2–4 h each) or GPU.
- **Disk:** ≤ 10 GB new; reported before any build larger than that.
- **ETA:**
  1. data audit + counts-only power checks (T-F availability, ±7 d Secchi, LAGOS/NLA): 1 day;
  2. data builds: 1–2 days;
  3. SSL pretraining: 2 days (GPU-limited);
  4. context + fine-tuning configurations: 1–2 days;
  5. freezes + looks: 1 day per track.

  **Total ≈ 7–9 working days**, contingent on T-F data availability. Without NAIP 2024+, T-F reduces to new
  labels on 2022–2023 imagery outside every spent block.
- **Honest expectation:** Secchi SOTA against GBM-RGBN is uncertain. The most plausible SOTA claims are scoped:
  - turbidity on the national/future track (val already favours the student vs band-ratio);
  - the small-water-body stratum.

  Each is labelled by track.

### Lead approval and decisions (2026-11-09 evening)
- **Plan approved.** Order:
  1. nothing starts until the 4-state test look has finished and passed its post-results review;
  2. no hashed file is touched before then.
- **National-freeze decision: (b), HOLD the national freeze** until levers 1–3 finish.
  - Estimate: about 6–7 working days (1 audit/counts, 1–2 builds, 2 SSL, 1–2 context and fine-tuning), within the
    lead's 9-day limit.
  - Reason: Test-N / test_time_n are the only unseen blocks with power (T-F is unavailable, see below).
  - If levers 1–3 overrun 9 working days, the national track is frozen as-is with the best model available then.
    Later levers then need new label sources (T-S).
- **Consequences for the national split** (to be pre-registered before any new-data result):
  - lever 1 adds chips, so the national split becomes version `national-2`, using the frozen national-1 rule:
    1. every national-1 chip keeps its role;
    2. new chips in a national water body that holds frozen or national-1 evaluation chips → `excluded`;
    3. new water bodies → national-1 block roles where the block exists, otherwise the salted random rule;
    4. 2-km buffers, as in national-1.
  - Its sha is frozen before any model sees the new rows.
  - The ±7 d Secchi window applies to **both** the student and every baseline (refit on identical rows), with
    counts pre-declared first.
  - SSL excludes every val, test and buffer water body of the 4-state and national splits by coordinates. The
    exclusion manifest is recorded.
- **T-X strata** (water bodies < 10 ha; turbidity/TSS > train P90; Secchi < train P10, from national train only) go
  into the national-track protocol **before** the national freeze, so they are scored in its single look.
  - They will be written into a separate national protocol file (`docs/preregistration/lite_national_protocol.md`),
    which is not hashed by the 4-state freeze, once the 4-state look has been reviewed.
  - The 4-state protocol §12 is not edited.
- **T-F availability check (done 2026-11-09):** the Planetary Computer `naip` collection temporal extent is
  2010-01-01 to 2023-12-31. A STAC search returns 0 items for `naip:year` 2024 and 2025 (2022: ny, pr…; 2023: mt…
  present).
  - **T-F (future NAIP cycle) is NOT available from our source.**
  - NAIP 2024+ exists only through USDA/USGS channels that need an account or requester-pays billing. That is a
    user action, and no credentials will be handled.
  - Fallback: T-S (new label sources: LAGOS-US LIMNO, NLA) on existing 2010–2023 imagery, at water bodies outside
    every split.
- **Data-size surprises:** none so far. No new build has run. The ±7 d Secchi counts and the LAGOS/NLA audit
  (counts only) are the first step after the 4-state review, and their sizes are reported before any build.

### Disk rule (2026-11-09; docs/plans/sota_campaign.md "Disk rule")
- Before ANY build (±7 d matchups, LAGOS/NLA, SSL corpus), run `df -h /home/bcheng`.
- Stop and report if free space would drop below 60 GB. The M7 cap is 10 GB of new data.
- At 2026-11-09 ~23:30 UTC: 100 GB free (98%). M7 run root 3.2 GB after deleting the post-look `eval_test*` chip
  caches (0.5 GB; the predictions are sealed and the look reviewed).
- Kept:
  - `train_52e0873ef590.npy` (data pin of the 4-state freeze);
  - the national train/val caches (in use).

### SOTA step 1 status (2026-11-09 ~23:45 UTC)
- National-track additions are in `docs/preregistration/lite_national_protocol.md` v0.1 (N1 strata, N2 n-rule, N3
  E3a_turb endpoint, N4 national-2 rules incl. train-only for new ±7 d chips, N5 counts).
- Counts-only check: `scripts/eval/national_lite_counts.py` → `$SENTINEL_RUNS/lite/counts_national2/counts.json`.
  - The ±1 d run reproduces the existing build exactly.
  - Secchi at ±7 d gives 38,061 samples, 30,647 pairs and 22,601 new chips (≈ 3.2 GB plus about 0.3 GB of S2).
- **Waiting for the lead's go** before the ±7 d build. Planned as a NEW script `scripts/data/build_naip_matchups_window.py`
  writing to `processed/naip_matchups/national_pm7/`; check `df` first.
- Pending:
  - LAGOS-US LIMNO package id and size;
  - NLA NAIP-date counts (STAC queries).

### SOTA levers 1–3: implementation (2026-11-10, lead go for ±7 d and SSL)
The 4-state scoring-code hash is verified unchanged after every addition. All new code avoids the frozen globs:
- **Lever 1, ±7 d data.**
  - `scripts/data/build_naip_matchups_window.py` (generated from the data builder, which is unchanged): Secchi ±7 d,
    other targets ±1 d, S2 ±1 d. Output `processed/naip_matchups/national_pm7/`, manifest
    `naip_matchups_national_pm7`, chips appended to the existing stores.
  - Index snapshots (for the id check) are in `processed/naip_matchups/snapshot_national_20261108/`.
  - `df` before: 108 GB free.
  - Running since 2026-11-09 17:47 EST, PID 1590613; log in `national_pm7/build.log`.
- **National-2 split:** `sentinel/models/lite/national2.py`. New rows are train-only; evaluation blocks are
  identical to national-1. The file is outside the hash (name contains "national").
  - `scripts/train/national2_run.py {train,baselines,compare,targets,select}` runs the unchanged hashed `lite_*` entry
    points on national-2 (`national2.activate()`) and adds the `ssl_init` encoder hook. Registry experiment:
    `lite_val_national2`.
- **Lever 2, SSL.**
  - `scripts/data/build_naip_ssl_corpus.py`: separate store `processed/naip_chips/ssl`, manifest `naip_ssl_corpus`.
    It records the exclusion manifest of every evaluation/buffer water body of both splits (2 km + same HydroLAKES
    lake), and has a `--dry-run` mode.
  - `sentinel/models/litessl/simsiam.py` + `scripts/train/ssl_lite_pretrain.py`: SimSiam on the MobileNetV4-S
    encoder with colour-balance augmentation. GPU 0/1 via the lock helper, ≤ 2 h chunks, resumable.
- **Lever 3, context.** `model.py` and `train.py` are hashed, so context is varied with the existing `crop` key only
  (192 px and the full 256 px). With much more data the earlier "crop128 > 256" result may not hold.
- **National-2 configurations (cap 8, 3 seeds each, CPU or GPU 0/1):**
  1. `kd03` on national-2 data (lever 1);
  2. + `ssl_init` (lever 2);
  3. crop 192 (+ ssl if 2 helps);
  4. crop 256 (lever 3).

  Baselines are refitted on national-2 train (parity). Teacher targets for the new train chips use the national
  Lite-safe teacher (evaluation blocks unchanged, so it stays disjoint). Selection uses the protocol §5 S on
  val ∪ val_n vs national-2 GBM. All configurations are counted.

### Progress 2026-11-10 ~03:30 UTC
- **±7 d build done.**
  - 51,381 matched samples (Secchi 38,061, as pre-declared); 22,575 new NAIP chips; NAIP store 4.3 GB, S2 store
    388 MB (+3.1 GB in total).
  - Existing chip ids unchanged. `df` 108 → 95 GB free (shared disk).
  - Datasheet `docs/datasheets/naip_matchups_national_pm7.md`.
- **National-2 split frozen** (sha `976b5be7…`, N6 in `lite_national_protocol.md`).
  - train 10,535 chips (Secchi labels 9,371 vs 3,667); evaluation blocks identical to national-1 (checked).
  - Configs `configs/lite/*_national2.yaml` (outside the hash; 4-state hash re-verified unchanged).
- **Running:**
  - `logs/queue_nat2_cpu.sh` (PID 1700802): national-2 baselines (parity) → teacher targets → SSL corpus dry run →
    `df` check → SSL corpus build (25 k chips, about 3.5 GB);
  - `logs/queue_nat2_gpu.sh` (PID 1700941): waits for the targets, then crop128/crop192/crop256 KD students on
    national-2. One lock-helper job per config (GPUs 0/1, about 30 min each).
- **Disk projection:** + about 2.8 GB national-2 train chip cache + about 3.5 GB SSL corpus.
  - Total new M7 data would be about 9.4 GB, close to the 10 GB cap.
  - The national-1 train cache (`train_c32790f7dc29.npy`, 1.27 GB) will be deleted once the national-2 runs are
    done.

### National look machinery (2026-11-10, drafted; review pending; nothing claimed)
- `sentinel/models/lite/national_testlook.py`: generated from the frozen `testlook.py` with all v1.7 hardening.
  - Look `lite_national/test`; registry prefix `lite_national_test`; directory `lite/national_test_look`; freeze
    `docs/preregistration/FREEZE_lite_national.json`.
  - National-2 data pins.
  - `national_scoring_code_sha256()` covers the 4-state hashed files plus the national, SSL and event files.
  - `load_protected()` is the only loader of test_n / test_time_n, and only under an open claim.
- `scripts/eval/national_lite_score_test.py`: generated from the frozen scorer. Test-N is confirmatory and
  test_time_n descriptive. Plus `national_extras`: N1 strata, N2 per-target n flags, N3 E3a_turb endpoint, E4a and
  E3a_tss descriptive.
- `scripts/eval/events_lite_national.py`: the E3a_turb calibrators (val ∪ val_n), run on the selected national
  model.
- **Still to do for the national candidate:**
  1. SSL configs (4 and 5 of 8) → final national selection;
  2. a no-KD comparator of the selected config (H2; not a candidate);
  3. export of the selected seed 0 (INT8);
  4. `events_lite_national.py`;
  5. the national candidate builder;
  6. a dry run on val ∪ val_n;
  7. pre-test review.
- Configurations tried on national-2: 3 so far (crop128, crop192, crop256 KD), plus 2 queued (SSL crop128, SSL
  crop256). National-1 (earlier): 2.

### 2026-11-10 midday
- **GPU rule update:** GPUs 0–9 are now allowed, via the lock helper with `--min-free-mb`, ≤ 2 h jobs, never on
  another user's GPU.
  - The waiting SSL + crop256 job had held a helper process started under the old 0/1 cap for about 5 h without a
    GPU. It was stopped by PID (it had not started training) and relaunched.
  - Running in parallel: config 5 SSL + crop256 (GPU 2), config 6 context parity (GPU 5), and the no-KD crop256 H2
    comparator (GPU 7; not a candidate). Status files are `logs/<config>.status`.
- **Lever 2 (SSL) result, config 4: negative.**
  - SimSiam ran 40 epochs on 24,943 chips (43 min, GPU). The loss plateaued at −0.95, suggesting partial collapse.
  - The SSL-initialised crop128 student gave 53.6 / 103.0 / 103.3, worse than ImageNet init (50.7 / 94.8 / 98.8).
- **Lead §15.1 ruling:** GBM + context is an applicable baseline.
  - Config 6 gives the student lat/lon/DOY through scene-FiLM (`context_film` hook in `national2_run.py`).
  - Limitation: the frozen ExportWrapper has no conditioning input, so an INT8 deployable of config 6 needs a
    two-input national export.
- **Numeric chl-a and prediction intervals:** pre-declared in `lite_national_protocol.md` N7 and implemented in
  `national_extras` (secondary; chl-a never enters selection).

### 2026-11-10 afternoon — architecture round v2 (written 14:48 EST)
- **Dispatch:** amendment first, then recovered labels, P2, ANCHOR-Lite and re-run published comparators. Before
  the stamp: code, tests, data builds (protected rows: counts only), synthetic or train-only smoke tests. **No
  validation number on a new arm, comparator or P2 before the stamp.** No git commits. Never read `keys.env`.
- **Amendment:** `docs/preregistration/lite_arch_v2_amendment.md`, DRAFT v0.3, sha256 `2b369018…8a56`, 559 lines,
  **frozen for review — do not edit.** Reviews and verifier: `docs/preregistration/reviews/arch_v2_20261110/
  M7_review_{design,integrity}.json`, `M7_verify.json` (recommendation: revise then stamp; v0.3 itself need not be
  edited; a counts-only label addendum is required). The lead sends the rulings. The author's separate list is
  `docs/preregistration/lite_arch_v2_change_list_draft.md` (not a ruling).
- **Lead rulings so far (LEAD_DECISIONS_20261110.md):** Q12 — the P2 floor stays, P2 is descriptive this round,
  one hardened look, no label; the national track under frozen §8a is the label route. Holds: the national-2b
  label build is not finalised (no split hash, no registry row); the national-2 freeze candidate does not go to
  pre-test review. TabPFN v2 comparator required (only through `sentinel.models.chemtox.tabpfn_model.load_tabpfn`;
  nothing installed; no fit before the stamp).
- **Code layout:** all round-v2 code is in `sentinel/models/litev2/` (outside every frozen hash), tests in
  `tests/test_litev2.py` (11 tests pass):
  - `p2.py`, `p2_index.py` — P2 lock and index. Lock `docs/preregistration/lite_p2_lock.json` sha `f3d4c63e…52be`
    (1,963 sites; registry row `03323db99ad9`), written once, never rebuilt, salt never changed. Index 513 rows,
    sha `68688fac…ff44` (row `e6a469d28fe1`); 515 chips. **test_p2 label values are never read beyond a
    present/absent flag.** The index is still depth-blind (see label rule); not rebuilt until the lead rules.
  - `wqp_rule.py` — M7's WQP label reader. Imports M2's `sentinel/models/orbit/wqp_surface.py` (sha `0e3ccc1c…d328`),
    holds no rule of its own. `apply_rule` (whole `national` / `national_pm7` table → kept, dropped keys + reason,
    record), `check_built` (keys only), `surface_samples` (for the P2 index). Nothing is rebuilt with it yet.
  - `tests_mask.py` — M2 Test-S flag (conflict C2): `flags`, `counts`, `mask_teacher`, `assert_masked`.
- **Reserved-loader guard (lead, 14:23 EST):** `p2.py` and `p2_index.py` now use `sentinel.data.reserved`
  (`exclude_reserved` / `assert_no_reserved`); sha `173f57b2…ff8c` and `53e44637…c0aa`. Reserved new sonde sites
  (729) in the lock, the index and the chips: 0 by id in every role. `tests/test_reserved_loaders.py` passes in
  full at 14:50 EST. By coordinates, 3 lock sites lie within 100 m of a reserved site (test_p2 2, val_p2 1; no
  index row, no chip): reported to the lead, no ruling yet.
- **Report on the lead's 14:25 EST items 2–5 sent 14:51 EST.** Waiting on the lead: rulings / addendum
  instruction; WQP rule for M7 (new label version and pin, 4-state chips inside national-2, configs 1–6);
  P2 index rewiring and rebuild; C2 handling of the 103 scored rows and of the national look's teacher step;
  TabPFN row cap (6,000 or 10,000); train_p2 block rule; deviations file; DenseNet121 download and licence rows.
- **P2 log event (accepted by the lead; goes into the deviations file):** log path
  `/home/bcheng/STOCKHOLM/sentinel_runs/lite/logs/p2_plan.log`; it held 34 train_p2 and 12 val_p2 label values
  (no test_p2 value); deleted 2026-11-10 18:08:03 UTC (13:08 EST); no model or comparator code of this amendment
  existed then (only `litev2/{__init__,p2,p2_index}.py` and `scripts/data/build_lite_p2.py`). **No log is deleted
  again.**
- **Shared WQP label defect (counts only; `sentinel_runs/lite/counts_arch_v2/`):** the `national` and
  `national_pm7` matchup tables were built with `load_wqp('processed')`: QC-reject and unmapped sidecar rows were
  read as results and depth was not read. `wqp_sample_counts.json` (sample level, build × target × role),
  `wqp_defect_counts.json` (chip-label level, P2 included). `national_pm7`, 51,381 rows: 1,609 would be dropped
  under the project rule; 1,971 keep the sample but lose member results. The 4-state table came from `raw/wqp`
  with depth read; its keys are not keys of M2's side table. Do NOT edit `sentinel/data/matchups.py`.
- **M2 Test-S (keys only; `tests_overlap_item4.json`, `supp_counts_item345.json`):** national-2 train chips on
  Test-S 777 (308 with a teacher pairing), val 114 (41), val_n 18 (9). Ruled: the 308 pairings carry no KD term
  and no teacher-derived feature in any v2 arm or comparator; the 777 chips stay in training; no M7 output is an
  input to anything scored on Test-S. The 2026-11-09 4-state look scored the S2-input teacher on 103 rows
  (65 test + 38 test_time; 102 distinct site × S2-date) in M2 Test-S lakes: registry rows `b41eab495b6c`,
  `bc70b284498d`, `b9c6bb9ffafe`, `d89d96326ac8`, `d544efe3f909`, `b9f03a45c672`. The national look script
  repeats that teacher step (`scripts/eval/national_lite_score_test.py` l.178–189): waits for the lead's ruling.
- **Teacher provenance:** targets `sentinel_runs/lite/teacher_targets_litesafe_national2/teacher_targets.parquet`
  (→ `teacher_targets_national_20261109-234957/`); checkpoints `sentinel_runs/lite/teacher_litesafe_national/
  s{0..4}/model.pt` (manifest 2026-11-09T08:09:44; M2 base recipe retrained by M7 without rows overlapping Lite
  val/val_n/test/test_time/test_n/test_time_n sites: same site, ≤ 2 km, same lake). Its 7,903 training points
  (`seen_points.parquet`; no date stored) are all in orbit-v3 train-role geography: 0 on a test-role lake.
- **Test-S flag table (keys only):** `scripts/data/litev2_build_tests_mask.py` →
  `sentinel_runs/lite/tests_mask_national2/{flags.parquet,meta.json}`, flags sha `ba7fcc60…3c0e`; `kd_masked`:
  train 308, val 41, val_n 9. Not registered yet (the count is registered with the stamp).
- **Next (code and tests only until the stamp):** stamp guard with a refusal test; deviations file; national-2b
  builder on `wqp_rule` (hold); P2 index rewiring (one call + `depth_m`); anchors with out-of-fold folds;
  S6–S11 incl. TabPFN; P1, CyFi-style, DenseNet121 comparators; ANCHOR-Lite / ANCHOR-Set; Test-S mask wiring;
  train_p2 block rule. After the stamp: comparators first, then ANCHOR-Lite, GPUs via the lock helper, ≤ 2 h per
  job; report the first job's name, GPU and PID to the lead.
