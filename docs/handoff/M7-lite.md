# Handoff — M7 SENTINEL-Lite (updated 2026-11-09 ~15:00; candidate ready for delta re-review)

## Status
- Nothing is running. M7 holds no GPU and no CPU job.
- `FREEZE_lite.json` does NOT exist. Test has never been loaded or scored. Registry M7 rows are val only.
- **Candidate:** `docs/preregistration/FREEZE_lite_candidate.json`, built by `scripts/eval/lite_freeze_candidate.py`.
  - Status CANDIDATE; `scoring_code_sha256` is PENDING. The value at candidate time is `3c424b891c29b7aa…`
    (77 files); it was checked to still be current.
  - The guard refuses the candidate: only `FREEZE_lite.json` with status FROZEN, `authorised_by`/`authorised_at`
    and an open claim unlocks test.
  - The lead writes the freeze. **Do not edit any hashed file** (lite code, configs, `scripts/{train,eval}/lite_*`,
    the protocol, SHARED_CODE including `sentinel/eval/events.py` and `sentinel/events/*`). Doing so invalidates
    the candidate hash.
- **Final dry run** (val, full path including the event track):
  `$SENTINEL_RUNS/lite/smoke/dryrun_test_scoring_20261109-135812`.
  - H1 / H1-deployable: only Secchi vs band-ratio is rejected.
  - H2: Secchi only, so the composite fails.
  - H-G6 false; label "negative" (GBM-RGBN better on Secchi).
  - E4a: derived and direct tie.
  - Nothing "not run".
- **Single test look**, after the lead's authorisation: `python -B scripts/eval/lite_score_test.py --allow-test`
  (CPU, about 1.5 h). Crash recovery after sealing: `--rescore-from-saved`. Write-ups go in separate files.

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
