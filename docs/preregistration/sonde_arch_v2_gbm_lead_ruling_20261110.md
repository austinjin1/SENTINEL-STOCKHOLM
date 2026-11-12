# SENTINEL-Sonde (M1), architecture round v2 — lead ruling on the GBM settings

Written by the lead on Tue 10 Nov 2026 20:40 EST. Only the lead writes this file. Nothing is appended to it; a
later ruling goes in a new dated file. It is the “lead's GBM ruling” of ruling S of
`sonde_arch_v2_dev_lead_rulings.md` (“dev rulings”). It is not a stamp and it releases no driver. When it was
written, no step-1 number and no validation number of this round existed, and the benchmark it rules on computed
no score.

## 0. Files this ruling rests on (sha256 re-computed by the lead when this file was generated)

| File | sha256 |
|---|---|
| `docs/preregistration/sonde_arch_v2_amendment.md` (stamped, registry row 2921bb1ca000) | `6c7b169d60ad74aa941965c67641d9a9e72faea804db6ac5b5b7b77d6bd60226` |
| `docs/preregistration/sonde_arch_v2_dev_lead_rulings.md` | `9bd515416928d31194cb7bcc0c46e69c1202932764a6f3ada5dbd8cb58e547fe` |
| `docs/preregistration/sonde_arch_v2_gbm_timing_rule.md` (“timing rule”) | `9d627602f3868c4b8547bea128862c23effd9a56d7019c47459e24ef253ea658` |
| `$RUNS/sonde/v2_smoke/gbm_timing_loaded.json` (“the record”) | `21807d51bb9c2c9193887755437733ba7bd8fa48455527f8338604c8948af4ec` |
| `$RUNS/sonde/v2_smoke/gbm_timing_loaded.log` | `ade5a57e6fc4cea721f630b0420cfecd188a435895218c1635e9d92d4e3ec000` |
| `$RUNS/sonde/v2_smoke/gbm_timing_loaded_crash1.log` | `6ec70cf28fa3f791f83c37f8bf83deb90e2bd39d32060ee7f203498503b9143f` |
| `scripts/train/sonde_v2_gbm_timing.py` (after the one-line fix) | `193f0439f742eecf50c984028f6a72188168b74d116f0144dcaf591eb8814f50` |

`$RUNS` = `/home/bcheng/STOCKHOLM/sentinel_runs`. The record names the timing rule by the sha256 above.

## 1. What was measured

- **The run.** One complete run of the benchmark of timing rule §5: start 2026-11-11T01:08:43Z (20:08:43 EST), end
  2026-11-11T01:20:19Z; 1-minute load average 50.87 at the start and 62.43 at the end, so it is a record under dev
  rulings Q (load above 36). Train-role sites and the train block only, PF-GBM features (106 columns),
  `temp_c_mean`, h = 1, 50 rounds per booster, 6 CPU threads, LightGBM 4.7.0, max_bin 255; GPU fits with
  `device_type=gpu`, `gpu_use_dp=False` on GPU index 0 through the lock helper. No score, no forecast file, no
  registry row.
- **Seconds per boosting round** (boosters q05, q50, q95, L1, L2). CPU, full rows (1,597,042): 0.580, 0.793,
  0.645, 0.624, 0.600. CPU, 500,000 rows: 0.219, 0.317, 0.208, 0.299, 0.203. GPU, full rows: 0.225, 0.208, 0.184,
  0.196, 0.222. GPU, 500,000 rows: 0.101, 0.195, 0.347, 0.431, 0.161.
- **Time of a unit at full rows by timing rule §3** (largest variable, 2000 rounds, no credit for early stopping),
  re-computed by the lead from the record: Q19 7.10 h on CPU and 2.17 h on the GPU; Q 2.62 h and 0.80 h; LS 0.68 h
  and 0.23 h. The Q19 unit is above 2 h on both devices, so the record does not give the one conclusion ruling Q
  allows.
- **Equality of the two devices.** Synthetic check of timing rule §6: passed, largest relative difference of the
  pinball loss 0.0037. Training loss after 50 rounds on the fixed sample, GPU against CPU: largest relative
  difference 8.1e-09 at full rows and 0.0044 at 500,000 rows.
- **The crashed first run.** A first run (its log runs from 19:46:04 to 19:53:10 EST) timed the five CPU full-rows
  boosters at 1.102, 1.199, 0.828, 0.961, 0.803 s per round and then stopped with `KeyError: '500k'` in the
  projection code, before any record file was written. Its CPU figures are 1.28 to 1.90 times those of the
  complete run, more than the 20 % of the lead's re-run order (Annex A). On the slower figures the Q19 unit on CPU
  is about 11.0 h. The lead rules on the slower set, as that order says: on either set the Q19 unit at full rows
  on CPU is above 2 h, and no ruling below reduces a GBM setting.

## 2. Rulings

- **G-1 — status of the record.** The record is kept and is `not_used_for_cap`, as ruling Q says. No cap below
  full rows, no support for proposal E(5) and no choice of device rests on its wall times.
- **G-2 — fit rows: full rows.** Every GBM configuration (Q19, Q, LS) of both GBM arms (GBM-NWP and PF-GBM) is
  fitted on full rows, in step 1 and in CV: every fit row that passes the row masks of timing rule §2 is used and
  no driver applies a row cap to fit rows (`max_fit_rows` stays `None` for all three; a unit test asserts that the
  fit rows of a unit are exactly the masked rows). The early-stopping rows are as written in timing rule §2 (same
  masks on the early-stopping sites, at most 400,000 rows by the same seeded sub-sampling), identical in step 1
  and CV and for both arms. Full rows is the top of the cap ladder and the strongest setting of the reference; it
  needs no timing in its support.
- **G-3 — the 2-hour limit is met by resuming, not by a cap.** The stamped amendment §5 limits a *process* to 2 h
  “with checkpoint/resume”. A GBM unit may therefore be fitted over more than one process. The GBM drivers resume
  at booster level: each booster is fitted whole inside one process and saved to its own file as soon as it is
  finished, with its stopping round and wall time; a booster that has a file is never fitted again; a process
  starts a new booster only if its elapsed wall time plus 1.5 times the longest booster time it has seen so far
  (40 minutes while it has seen none) is at most 115 minutes, and otherwise ends; the same command continues the
  unit. The record of a unit lists every process that worked on it. An unfinished booster of a stopped process is
  fitted again from its start with the same seed. A unit test covers stop-and-resume and shows that a resumed unit
  uses the same rows, seeds and settings as an uninterrupted one. A booster that is identical to a booster of
  another configuration of the same arm, variable, horizon and fold in fit rows, early-stopping rows, objective,
  quantile level, seed and settings may be fitted once and used by both, if a test shows the identity; this
  changes no model. If a single booster ever takes more than 2 h, the engineer stops and reports (at the slowest
  figure seen so far, 1.199 s per round, 2000 rounds take 40 minutes).
- **G-4 — no repeat of the benchmark.** With G-2 and G-3 the cap ladder has no use: §4 of the timing rule is not
  applied, and the repeat at a load of 36 or less that ruling Q and timing rule §7 call for is not run, because
  its only purpose was to pick a cap by §4. The benchmark is not started a third time.
- **G-5 — recipe.** The frozen `GBMConfig` values of timing rule §1 stay for every GBM configuration of both arms
  in step 1 and in CV: learning rate 0.05, at most 2000 rounds, early stopping after 100 rounds without
  improvement on the early-stopping rows, num_leaves 63, min_data_in_leaf 200, feature_fraction 0.8,
  bagging_fraction 0.8, lambda_l2 1.0, max_bin 255. Proposal E(5) (learning rate 0.10, 1000 rounds) is not
  adopted.
- **G-6 — device: GPU for every GBM fit.** Every GBM booster of both arms, in step 1 and in CV, is fitted with
  LightGBM `device_type=gpu`, `gpu_use_dp=False`, max_bin 255 (the settings of the record), one GPU per process
  through the lock helper, at most 6 CPU threads, never on a GPU that holds another user's process. The grounds
  are not the wall times of the record: (i) the lead's order of preference, written before any timing existed (dev
  rulings §2, “frozen recipe at full rows on GPU > frozen recipe with the largest passing cap > your E(5)”); (ii)
  the stamped amendment §5, under which M1's CPU jobs are sequential while GPU jobs may run in parallel; (iii) the
  equality checks of §1 above, which passed. GBM-NWP and PF-GBM always use the same device and device settings.
  Every fit writes its device, GPU index and device settings to its parity record. If a GPU fit fails for a
  technical reason the engineer stops and reports; no fit falls back to the CPU without a new lead ruling. A
  finished booster is the model of record and is not refitted to compare devices.
- **G-7 — total GBM compute.** The upper bounds of the record at full rows on the GPU (2000 rounds for every
  booster, measured under load, no credit for early stopping) are 12.1 GPU-h for step 1 (Q19) and, per CV fold and
  arm, 81.9 GPU-h for Q19, 30.2 for Q and 11.0 for LS; with 5 folds and 2 arms that is about 819 GPU-h for Q19
  alone. The real cost depends on the stopping rounds, which nobody knows before the first fit. Rule: (a) every
  GBM process writes its GPU wall time to the registry, and the running total of GBM GPU-hours is kept per stage
  (step 1, CV); (b) step 1 runs under G-2 to G-6 and records the stopping round of every booster; (c) before the
  first CV GBM process the engineer sends the lead the CV projection re-computed from those stopping rounds
  (boosters × rounds × seconds per round, per configuration and arm); (d) if that projection, or later the running
  CV total plus the projected remainder, is above 300 GPU-h for the declared GBM configurations of both arms
  together, no further CV GBM process starts until the lead has ruled. (e) This file fixes no fallback, and none
  is foreseen: more GPUs in parallel and the order of the work are the means to meet the time. A reduction of a
  GBM setting after the stamp would change the reference after step-1 numbers exist. If the lead ever rules one,
  it is at least: a deviation entry in a new file, an independent review and a lead stamp row before any CV GBM
  fit under it; the same for both arms, every GBM configuration and all folds; decided from compute figures only;
  reported as a change of the reference made after step 1; step 1 repeated under it and both step-1 results
  reported; and no booster fitted under the old setting used beside one fitted under the new. This ruling does not
  change the declared configuration lists; which additional configurations (Q, LS) are declared is fixed in the
  consolidated file before the stamp (dev rulings B: Q19 is mandatory, Q and LS are additional).
- **G-8 — to be disclosed in the consolidated deviations file.** (i) Timing rule §5 names GPU 7, 8 or 9; the run
  used GPU 0, as the lead's message of 2026-11-11T00:37:58Z item 5 allowed (Annex B). (ii) The crashed first run:
  log path and sha256, its five CPU figures, the one-line fix with the script's sha256 before and after, the load
  at both starts. (iii) That §4 and §7 of the timing rule and the repeat clause of ruling Q were not applied, with
  the reason of G-4. (iv) The booster-level resume of G-3. (v) G-6 and G-7 in full.
- **G-9 — what follows.** The engineer enters this ruling in `sentinel/models/sonde_v2/declared.py`
  (`GBM_SETTINGS`: `ruled` true, `ruling` = this file's path and sha256, `device` gpu with the device settings of
  G-6, `max_fit_rows` `None` for Q19, Q and LS; the cap ladder is removed or left unused, and a test asserts that
  no driver reads it), writes the resume code and its tests, and then follows ruling S unchanged: final green test
  run → consolidated file with the code list → independent delta review → stamp → step 1. The delta review is
  asked to check G-2 to G-7 against the code. If the engineer sees a technical obstacle in G-3 or G-6, it reports
  to the lead before `declared.py` is changed; nothing else in this file is open.

## 3. How this file is read

- This file wins over the following, and only so far as they would apply a row cap to GBM fit rows or call for a
  repeat of the benchmark: §4 and §7 of `sonde_arch_v2_gbm_timing_rule.md`; in the dev rulings, E(2) (“the first
  cap whose fit is within 2 h is used”), E(4), the clarification of E(1) on the cap, the last sentence of the
  message in §3, and the sentence of ruling Q “… the benchmark is repeated at a load of 36 or less; only that
  second record enters §4 of the timing rule”. Everything else in the dev rulings and in the timing rule holds,
  including the row masks and the early-stopping row rule of timing rule §2.
- Where this file and the stamped amendment (`sonde_arch_v2_amendment.md`) or `sonde_arch_v2_lead_rulings.md`
  differ, the stamped files hold until the consolidated deviations file is stamped (dev rulings §5).
- The two annexes reproduce lead messages word for word from the session record, with the record's time. A time
  written inside a message is the lead's own label and may be a few minutes off; the record's time counts.

## Annex A — lead message to the M1 engineer, session record time 2026-11-11T00:57:18.621Z (19:57:18 EST), 1,308 characters

> Lead ruling on the crashed GBM timing run (M1): yes, one complete re-run is allowed.
> 1. The crash was a script bug before any record file was written, so ruling Q's "once" is not yet used. Fix the one line (`sonde_v2_gbm_timing.py:112`), then run the complete loaded benchmark exactly once more: same command, CPU + GPU 0 through the lock helper, `--out gbm_timing_loaded.json`. Start it as soon as `sonde_v2_counts.py step1` ends; do not wait for a lower load (the benchmark is meant to be measured under load, and waiting to pick a load would be a choice made after seeing numbers).
> 2. Disclosure, in the consolidated deviations file: the crashed log (path + full sha256, read-only), its five CPU full-rows figures, the one-line fix (file sha before/after), and the load at both starts. The record of the complete run is the record of the benchmark. If the CPU full-rows s/round of the two runs differ by more than 20 %, report both and I rule on the slower one.
> 3. No third start without a new lead ruling. If the second run fails for any reason, stop and report.
> 4. Timing only: no validation, test or protected row is read; train block only, as before.
> Then send me the numbers (CPU full rows, CPU 500k, GPU full rows, GPU 500k, s/round, projections) and I write the GBM ruling in a new dated lead file.

## Annex B — lead message to the M1 engineer, session record time 2026-11-11T00:37:58.747Z (19:37:58 EST), 2,142 characters

> Lead to M1 (19:38 EST): rulings A–P are now on disk and the benchmark may run under load for one conclusion only.
> 1. File: docs/preregistration/sonde_arch_v2_dev_lead_rulings.md, 0444, sha256 9bd515416928d31194cb7bcc0c46e69c1202932764a6f3ada5dbd8cb58e547fe (153 lines). §1 = A–P verbatim, §2 = the 14:54 EST clarifications and verifier items 1–7 verbatim, §3 = the 18:25 EST CPU rule verbatim, §4 = new rulings Q, R, S, §5 = how it is read (verifier conditions = every condition_for_rulings and missed item of M1p2_verify.json at sha 9ba443d9…fb14a). Cite it by path + letter (e.g. "dev rulings G(b)"). Read §4 and §5 in full before you use them.
> 2. Ruling Q (your point 1): do not wait. Run the benchmark once now at any load, with the load at start and end in the record. That record may be used ONLY to conclude "full rows" (PF-GBM unit at full rows, §3 of the timing rule, ≤ 2 h on CPU-6 or on GPU). It may not be used to pick a cap below full rows, to prefer a device because the other looked slower, or to support E(5). If it does not give "full rows", mark it not_used_for_cap and repeat at load ≤ 36 (after the GEFS extraction ends); only the second record enters §4 of the timing rule. Send me the numbers either way; I rule and you enter it in declared.py.
> 3. Ruling R (your point 3): the stamp row will carry extra.written_by = "lead", extra.sha256 = {path: sha} including docs/preregistration/sonde_arch_v2_deviations_v2.md, and extra.code_sha256 = the value you send (I re-compute it first). Send me the exact recipe for code_sha256 (file list, order, how the digest is formed) with the file.
> 4. Ruling S (your point 4): order confirmed — benchmark → my GBM ruling → declared.py → final green tests → consolidated file with code list → delta review → stamp → HF phase.
> 5. From the user (18:50 EST): idle GPUs may be used to speed up approved work, inside the plan. GPU 0 is the only idle one now; never GPU 2 (another user's job). The LightGBM GPU timing may use any idle GPU through the lock helper, not only 7–9 (those now carry M5 jobs) — record the index.
> No step-1 or validation number before the stamp. Keep messages short.
