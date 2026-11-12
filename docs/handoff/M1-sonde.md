# M1 SENTINEL-Sonde + EW-v2 — handoff note (updated 2026-11-11 ~00:20 UTC)

## UPDATE 2026-11-11 ~00:20 UTC (11-10 19:20 EST) — GO received; rulings A–P implemented; tests; benchmark pending (READ FIRST; supersedes the 19:50 section's HOLD)

**State.** The code HOLD is lifted (lead GO with rulings A–P + 7 verifier items, `reviews/arch_v2_20261110/M1p2_verify.json`,
`M1p2_review_code.json`). Nothing is stamped for the v2 code yet: NO step-1 / CV / validation number exists and none may be
produced before the lead stamps `docs/preregistration/sonde_arch_v2_deviations_v2.md` (NOT WRITTEN YET).
Lead's order: (1) code fixes + green tests — DONE except the final re-run after the last test append; (2) GBM timing
benchmark → numbers to the lead → lead rules GBM settings; (3) A8 + reference-window counts on real data; (4) write the ONE
consolidated deviations file (rule-A form, code sha256 list) → send path + sha256 → no edits after; (5) lead delta review
+ stamp; (6) step 1 (`--phase declare`, `hf` on a GPU, `gbm`, `eval`) — **send job name, GPU index, PID, start time the
moment the HF phase is really running** (the user waits for exactly that notice).

**Rulings in one line each (binding; full text only in the lead's messages — do not cite them as on-disk files).**
A one file `sonde_arch_v2_deviations_v2.md` supersedes parts 1+2 · B Q19 mandatory GBM config + step-1 comparator ·
C gage height restored · D every reference on the networks' train rows, ES rows = 15 % in-fold basin groups (seed
20261110 + fold); frozen `sonde/gbm.py` not edited · E full-rows config + time→cap rule written before timing
(`docs/preregistration/sonde_arch_v2_gbm_timing_rule.md`, 0444, sha 9d627602…a658) · F reference window 730 d from first
finite value ≥ 2007-10-01, ≥ 365 obs, issue ≥ window end + 42 d for every model/use · G(b) GEFS c00 only, one product,
hard errors · H DO %sat = transform of SONDE-HF quantiles; references write the 8 scored variables · I bias map recorded
fully · J R-NH rows labelled for its target; stage-2 optimiser recorded, identical · K C4 = CMAL as stamped · L declared
config lists hashed, drivers refuse others · M GBM-NWP degree-day features · N error model WY2008–2019 · O code sha256
list in the file, then frozen · P D1/D3/D4/D5/U6/U9 one line each.
Lead 18:26/18:30 EST: benchmark may run beside the extraction only if 1-min load ≤ 36 at start (record load start/end); a
timing under load may not justify a smaller GBM setting (repeat after the extraction ends if it would). Messages short.

**Code (all under the future freeze: `sentinel/models/sonde_v2/*.py`, `scripts/train/sonde_v2_*.py`, `tests/test_sonde_v2*.py`).**
- `declared.py` — every arm's configuration list, seeds, step-1 settings, stage-2 settings, `GBM_SETTINGS`
  (`ruled: False` → `gbm_config` raises; enter the lead's ruling here BEFORE the freeze: `ruled`, `ruling`,
  `max_fit_rows`, `device`). `declaration()` / `declaration_sha256()`.
- `parity.py` — stamp guard (`require_stamp`: lead-written registry row `M1-sonde` / `sonde/prereg_snapshot` with
  `extra.sha256[<each docs/preregistration/sonde_arch_v2_deviations_v2*.md>]`, `extra.written_by == "lead"`,
  `extra.code_sha256 == parity.code_snapshot_sha256()`; files read-only; code list between the
  `<!-- CODE-SHA256-BEGIN/END -->` markers = `parity.code_list_text()`), `declare` (DECLARATION.json 0444 + registry +
  parity log), `require_declared`, `record`, `check`.
- `products.py` — `build(data, use=…)`: GEFS c00 only, required months, coverage ≥ 0.90, reforecast-only bias map
  (valid ≤ 2021-09-30), error model on valid days ≤ 2019-09-30, perfect-prog sample + hashes.
- `prep.py`, `data_v2.py`, `train_v2.py`, `nets.py`, `gbm_nwp.py`, `chronos_cov.py`, `physics_features.py`, `nwp.py`.
- Drivers `scripts/train/sonde_v2_step1.py`, `sonde_v2_cv.py` (phases declare / folds / run / parity / combine),
  timing `sonde_v2_gbm_timing.py`. Both drivers have `DEVICE = "cuda"` (tests set "cpu"). Every CV run writes
  `OOF_DONE` as JSON with `no_held_rows`; `combine` accepts a fold without a variable's file only when that fold
  recorded no held-out val rows for it.
- **Bug found by the benchmark, fixed:** `physics_features.k600_raymond` did not broadcast site slope (S,1) against
  discharge (S,T) → every PF-GBM fit would have crashed. Unit test added.
- **Tests:** `tests/test_sonde_v2.py`, 45 passed at 19:12 EST; two end-to-end rehearsal tests added after that (step 1:
  declare→hf→gbm→eval; CV: folds, SONDE-HF 2 stages ×5 folds, PF-GBM LS ×5, GBM-NWP Q19, R-NH, Chronos stub, parity,
  combine) on synthetic data, CPU. Run: `OMP_NUM_THREADS=6 …/envs/sentinel2/bin/python -B -m pytest
  tests/test_sonde_v2.py -x -q -p no:cacheprovider` (no `py_compile`: it writes `.pyc`).
- Patch scripts (history of every edit): scratchpad `m1/p_*.py`.

**GBM timing benchmark.** First attempt 18:54 EST crashed on the k600 bug (no json). NOT rerun yet: 1-min load 42–57 >
36. Relaunch: `SENTINEL_GPUS=7,8,9 OMP_NUM_THREADS=6 nohup …/python -B scripts/train/sonde_v2_gbm_timing.py >
sentinel_runs/sonde/v2_smoke/gbm_timing.log 2>&1 &` (refuses above load 36; writes `v2_smoke/gbm_timing.json`). Labelled fit
rows at h=1 (train role, 1,610 sites): temp 1,597,042; DO mean 407,758; DO min 446,132; SC 722,454; pH 258,836; turb mean
129,743; turb median 132,941; NO3 9,086.

**Data jobs.** GEFS point extraction PID 2244958 (`sentinel-grib` env; reforecast 200710–201912 complete = 147 months;
operational 202009–202309 in progress) → `sentinel_data/processed/gefs_points/{reforecast,operational}/YYYYMM.parquet`.
Open-Meteo pull PID 2118394 keeps running; no model reads it (G(b)). Live trial ON HOLD.

**To tell the lead with the deviations file.** Stamp row must carry `written_by: "lead"` and `code_sha256` (row
2921bb1ca000 has neither); R-NH cost (8 targets × 5 configs × 5 folds × 2 stages); CV scorer and M1-H = later NEW files
under their own dated deviation (`sonde_arch_v2_deviations_v2_<name>.md`, which the guard also hashes).


## UPDATE 2026-11-10 ~19:50 UTC — deviations parts 1+2 REVISE; code HOLD (READ FIRST; supersedes the 18:30 gate)

**State.** Both reviews are back: REVISE. No step-1 or CV number until ONE consolidated file is stamped.
- Reviews (verbatim): `docs/preregistration/reviews/arch_v2_20261110/M1dev_review.json` (part 1; blocking D2, U1, U3,
  U8) and `M1p2_review_integrity.json` (part 2; blocking P2-01..P2-04, should-fix P2-05..P2-12).
- **HOLD:** no change to `sentinel/models/sonde_v2/`, `scripts/train/sonde_v2_*.py`, `tests/test_sonde_v2*.py` until
  the lead says the part-2 code lens has returned.
- **Never edit** `sonde_arch_v2_deviations.md` or `_part2.md` again (reviewed record). No appends either: the A8 and
  A12 notes go into the new file.
- **New file to write (after the code fixes):** `docs/preregistration/sonde_arch_v2_deviations_v2.md`. It supersedes
  parts 1 and 2 and is the only deviations file the lead stamps.

**Lead rulings A–P (binding), short form:**
- A: one file; every entry has stamped text, code file:line at the frozen state, reason, effect, counts; facts checked
  on disk the same day; no reference to a ruling that is not on disk.
- B: Q19 is mandatory for GBM-NWP and PF-GBM and is the step-1 comparator; Q and LS are extra configurations.
- C: gage height (`gage_ft_mean`, 1,258 sites indexed) is restored as auxiliary input and output. My part-2 sentence
  that it does not exist was false.
- D: every reference is fitted on the same train-block rows as the networks (no `fit_end` 2019-09-30); early-stopping
  and calibration rows come from the 15 % in-fold basin groups (seed 20261110 + fold). `sentinel/models/sonde/gbm.py`
  is NOT edited.
- E: GBM cap rule written as exact text BEFORE the timing run; full rows first, then named caps; first cap whose unit
  fit (one variable, one horizon, all levels, 6 threads) is within 2 h; `gbm_timing.json`; same rule in step 1.
- F: reference window = first 730 days, ≥ 365 observed; a row is used only if issue ≥ window end + 42 d (every model,
  every use); a window not fully inside the loaded dates gives NaN climatology; unit tests; counts in the file.
- G: each NWP slot standardised from its own train-block rows; a missing or low-coverage declared product is a hard
  error; unit test of a non-zero GFS mask on val batches.
- H: file and drivers say exactly which arm writes which targets; DO % saturation is a deterministic transform.
- I: bias map recorded completely; A8 counts as a dated addendum after the full reforecast.
- J: R-NH batches and early-stopping rows drawn from rows with a label for its own target; stage-2 optimiser recorded.
- K: C4 (CMAL head for SONDE-HF) implemented with a unit test.
- L: configuration lists hashed in the parity log and registry before the first run; drivers refuse an undeclared
  list; R-NH gets 5 configurations.
- M: forecast degree-days as stamped; no member mean/spread (R3) recorded as a deviation.
- N: error model fitted to 2019-09-30 on the full reforecast; step-1 eval asserts the same noise hash.
- O: after fixes and green tests, sha256 list of the sonde_v2 code in the file, then no more edits.
- P: D1, D3, D4, D5, U6, U9 each get one line.

**Reported to the lead at ~19:50 UTC (waiting for the go on code edits):**
- GEFS: 48 of 144 reforecast months and 1 of 61 operational months on disk at 19:47 UTC; the `--pilot` process continues by
  itself (pilot passed, 7 registry rows `sonde_arch_v2/gefs_pilot` at 18:44 UTC); about 1.7 min per reforecast month;
  WY2008–2019 complete about 22:35 UTC; operational through 2021-09 about 23:40 UTC; everything about 04:00 UTC 11-11;
  projected kept size 0.49 GB.
- GFS table: not buildable before about 11-28 at day-cap 4800 (103 of 1,906 sites complete). Step 1 does not use GFS.
  For CV the lead rules: wait, or GEFS-only for every arm (deviation).
- GBM arithmetic (estimates from the frozen GBM-w log, not measured): Q19 at full rows is about 57–86 h per fold and
  arm; proposal sent (lr 0.10, ceiling 1000 rounds, patience 50 for all GBM configurations; cap by the 2 h unit rule;
  lists {Q19, LS}).
- Points raised: F needs "first observation inside the WY2008+ record" (1,697 of 4,101 site-variables start before the
  cube start); M1-H code is not built yet, so rule O needs a way to add M1-H files later.
- Metadata-only count script: scratchpad `m1/meta_counts.py`.

**Running processes (all M1):** GEFS extraction PID 2244958; Test-A rehearsal PID 2134038 (fine-tuned Chronos seeds
running); Open-Meteo pull PID 2118394 (sleeping on the day cap until about 15:45 UTC 11-11).

## UPDATE 2026-11-10 ~18:30 UTC — review in progress; extraction resumed (READ FIRST)

**Files under review. Do NOT edit or append** until the lead says the review is back; any append changes the hash
under review.
- `sonde_arch_v2_deviations.md` (part 1, sha `031ddddc…`);
- `sonde_arch_v2_deviations_part2.md` (v0.2, sha `b17f3ec6…`, 229 lines; items 1–15).

**Notes waiting to be appended** later, as append-only dated addendum lines at the end of part 2, when the lead says so:
- **A8:** bias-map fallback counts per `BiasMap.source`, from train-block weather, once the full reforecast is
  extracted.
- **A12, measured lower bound:** GBM-NWP Q, temp_c_mean, fold 0, h = 3, 1.5 M-row cap, 7 boosters, 6 threads, ES on
  300k inner rows: more than 41.5 min of fitting for ONE horizon, unfinished (run killed at 44.9 min on the lead's
  instruction). Next:
  - after the extraction, a fixed-round benchmark (50 rounds per booster at the capped row count);
  - from timing arithmetic only, a declared-budget proposal for GBM-NWP and PF-GBM (same rule for both; ≤ 2 h per job,
    ≤ 6 threads; say what it costs vs early-stopping rounds);
  - the lead stamps it as an addendum before any CV run.

**Gate for step 1:**
1. the lead's stamp row (one rulings file `sonde_arch_v2_deviations_lead_rulings.md` + `sonde/prereg_snapshot`) for
   part 1 + part 2;
2. the full WY2008–2019 reforecast extracted.

Then run `scripts/train/sonde_v2_step1.py --phase hf` (GPU) and `--phase gbm` (CPU, alone), then `--phase eval`.
**Report job name, GPU and PID to the lead immediately.**

**Running processes (all M1):**
- GEFS extraction: PID 2244958, env `sentinel-grib`, resumed 18:28 UTC; log scratchpad `gefs_pilot.out`.
- Test-A rehearsal: PID 2134038; log scratchpad `m1live/m1_rehearsal.out`.
- Open-Meteo pull: PID 2118394, day-cap 4800.

The parity log now also hashes the perturbed perfect-prog input (`input_sha256.pp`) per run.

## UPDATE 2026-11-10 ~18:15 UTC — deviations part 2 written; parity fixes; GBM timing

- **`docs/preregistration/sonde_arch_v2_deviations_part2.md`** (sha `408cf516…`) records items 1–14 per the lead's
  ruling. Part 1 (`031ddddc…`) is under independent review; **do not edit either** without telling the lead. The lead
  stamps both together.
- **Step 1 starts only after:**
  - that stamp row exists (`sonde/prereg_snapshot`);
  - the full WY2008–2019 reforecast is extracted;
  - the A8 (bias-map fallback counts) and A12 (timing + declared GBM caps) addenda are sent.
- **Code fixed before step 1:**
  - R-NH gets the same deterministic forecast-branch inputs and the same head form as SONDE-HF.
  - GBMs get target-day clear-sky SW and day length.
  - Slope (`nhd_slope` from StreamCat) is a static for all models; the PF-GBM K600 is built.
  - Derived trained heads: DO amplitude, DO % saturation, T 7DADMax (20 targets).
  - 2-year reference-window climatology for all v2 models (M1-N §3).
  - Parity log `sentinel_runs/sonde/v2_cv/PARITY_LOG.jsonl`.
  - Q19 config.
  - Step-1 GBM cap 1.5 M.
- **GBM timing** (`scripts/train/sonde_v2_gbm_timing.py`; result `sentinel_runs/sonde/v2_smoke/gbm_timing.json`)
  runs Q vs Q19. The GEFS extraction auto-resumes when it exits (chained background job, `--pilot`; resume-safe).
- **M1-H:** `sentinel/models/sonde_v2/hourly.py` has the targets (hourly IV, hypoxia hours) and the naive references
  (persistence, diel, climatology), unit-tested. Still to build: the hourly tensors for SONDE-HF-hourly, GBM-hourly,
  LSTM-hourly, R-NH-hourly and Chronos-hourly, and the M1-H CV driver.

## UPDATE 2026-11-10 ~17:45 UTC — architecture round v2: amendment STAMPED; build in progress (read first)

**Stamp.**
- `docs/preregistration/sonde_arch_v2_amendment.md` (sha `6c7b169d…`; registry `2921bb1ca000`, 17:23 UTC) is stamped
  as written. Lead rulings: `sonde_arch_v2_lead_rulings.md` (sha `7cbf3987…`). Both files are 0444; never edit them.
- Rulings in brief:
  - R1: E2-B = {GBM-NWP, PF-GBM, R-NH, Chronos-2-cov}.
  - R2: no upstream inputs.
  - R3: GEFS c00, pilot first, separate env.
  - R4: M1-H in the same look, as a separate family.
  - R5: EW-H1h descriptive.
  - L2: the M1-H best reference is the worst case over all comparators.
  - L3: CV fits on in-fold sites × train block only.
  - L4: deviations go in a new file and need a stamp before any affected number.
  - L5: a public commit is required before any freeze.
- **Deviations draft:** `sonde_arch_v2_deviations.md` (D1–D5, sha `031ddddc…`) is UNDER INDEPENDENT REVIEW. **Do not
  edit it.** No step-1 or validation number on any arm D1–D5 touch until it is stamped. Fourteen further code-vs-text
  differences were reported to the lead by message (17:50 UTC); the lead decides which go into the file.
- **Q1 ruling: WAIT.** Step 1 uses the error model fitted on the full WY2008–2019 reforecast, never partial years.

**GEFS (R3).**
- Script `scripts/data/extract_gefs.py`, run in the separate conda env `sentinel-grib` (eccodes 2.50). The temporary
  pip-target install was removed; `sentinel2` is untouched.
- Outputs: `processed/gefs_points/{reforecast,operational}/YYYYMM.parquet` (split sites, issues ≤ 2023-09-30) and
  `processed/gefs_points_newsites/…` (0440). Manifest and package list: `processed/gefs_points/MANIFEST_extract.json`.
- **Pilot:** the correctness check passed (300 values, max |Δ| 0.0 against eccodes' nearest-point read). About 132 MB
  and about 4 s per init with 6 threads. About 2 MB kept per month, so the projected total is about 0.3–0.4 GB.
  Reforecast months 2014-10..2015-01 are done.
- **Paused 17:40 UTC** for the GBM timing job (one M1 CPU job at a time). Restart with
  `…/envs/sentinel-grib/bin/python -B scripts/data/extract_gefs.py --pilot --threads 6`. It resumes by month, reruns
  the operational-month check, logs the R3 registry rows (`sonde_arch_v2/gefs_pilot`) and then continues
  automatically unless a stop condition holds.
- Full reforecast ETA is about 5 h after the restart.

**Code** (frozen candidate files are not edited; v2 subclasses or imports them).
- **Modules:** `sentinel/models/sonde_v2/`:
  - `nwp` (alignment, bias map, error model, GBM trajectory features);
  - `physics_features`;
  - `gbm_nwp` (Q / Q19 / LS; per-product features);
  - `nets` (masked-mean NWP, monotone and CMAL heads, R-NH, SONDE-HF with ordering constraints);
  - `ensemble` (one E2/E2-B rule + cross-fit);
  - `chronos_cov`;
  - `data_v2` (daily via frozen `SondeTensors`, per-lead targets, NWP slots, hourly cube);
  - `train_v2` (resumable, ≤ 2 h);
  - `prep` (aux targets, all site types, train-only blocks);
  - `products`.
- **Scripts:**
  - `scripts/train/sonde_v2_step1.py` (phases hf / gbm / eval);
  - `scripts/train/sonde_v2_cv.py` (phases folds / run / combine; arms sonde_hf C1–C5, rnh R1–R2, gbm_nwp Q/Q19/LS,
    pf_gbm Q/Q19/LS, chronos_cov ZS);
  - `scripts/train/sonde_v2_smoke.py`;
  - `scripts/train/sonde_v2_gbm_timing.py`.
- **Tests:** `tests/test_sonde_v2.py` (16 pass).
- **Smokes** (train-role × train block, no score): SONDE-HF on GPU 4 (2.5 GB VRAM, 0.15 s/step, hourly cube 411 sites;
  a NaN bug in loss masking was found and fixed); R-NH on GPU 3 (2.6 GB, 0.12 s/step).

**Next, in order, after the deviations stamp and the full reforecast:**
1. `sonde_v2_step1.py --phase hf` (GPU) and `--phase gbm` (CPU), then `--phase eval` (decision + registry).
   **Report job / GPU / PID to the lead the moment the first real job starts.**
2. `sonde_v2_cv.py --phase folds` (once), then references (R-NH, GBM-NWP, PF-GBM, Chronos-2-cov), SONDE-HF CV and M1-H.

M1-H (hourly targets) code is not written yet.

**Test-A rehearsal** (plan `9e59bb9d`): 6 of 12 jobs at Δ = 0, including all 4 weather jobs under the ERA5 mask.
FT-Chronos s0–s2 (resumable), zs_uni, zs_multi and TimesFM remain.

## UPDATE 2026-11-10 ~17:05 UTC — GPU rule update (0–9) applied; ≤ 2 h jobs; final plan; full rehearsal running

- **GPUs 0–9.**
  - The frozen `runner.py` hard-codes `M1_GPUS = (0,)`; it is 0444 and hashed by FREEZE_ew/hydromet, so it is not
    edited. The new `sentinel/models/sonde/gpu_budget.widen()` widens it per process to the cap file
    (`.gpu_locks/ALLOWED` = 0–9) and sets `MIN_FREE_MB` = 10000, so only idle GPUs are taken (a 2080 Ti has 11019
    MiB) and never one another user occupies.
  - It is called in `sonde_predict.py` (`--min-free-mb`), `sonde_chronos_zeroshot.py`, `sonde_lstm.py`,
    `sonde_chronos_ft.py` and `sonde_virtual.py`. There is no hard-coded `SENTINEL_GPUS` in M1 code.
- **≤ 2 h jobs: violation found and fixed.**
  - FT-Chronos val prediction takes about 2 h 44 m per seed in one process. The earlier rehearsal FT jobs (s0–s2, on
    GPU 0) exceeded the 2 h cap; this is disclosed here and in `sonde_m1_protocol.md` §4.
  - **Fix:** `ChronosForecaster.predict_multi_resumable` with `sonde_predict.py --max-minutes 100`. The process stops
    only after a batch flush, saves the state and exits with code 75; the orchestrator resumes it in a new process.
    The batches are identical, so the forecasts are identical.
  - All multivariate Chronos jobs in the plan use it (FT s0–s2, zs_multi).
  - The orchestrator now runs CPU jobs alone, then up to 3 GPU jobs at once (2 CPU threads each, within the 6-thread
    rule).
- **Final plan:** `sonde_m1_testA_plan.json` sha **`9e59bb9d…`**, code hash **`50cf8541…`** (adds `gpu_budget.py`,
  `chronos.py` and `gpu.py` to the closure). Superseded plans are in scratchpad `m1live/`: `2cf2ddff`, `ed43ab6a`
  (its pi-report CIs were then added) and `37e552ba`.
- **Rehearsal restarted 16:57 UTC.** It re-runs **all 12 jobs** with the final commands, and each identity entry
  records `plan_sha256`. Earlier partial results are kept in `rehearsal_val/identity_v1_firstpass.json` and
  `identity_v2_partial_plan2cf2ddff.json`. ETA about 21:30–22:00 UTC (3 FT seeds in parallel take about 2.8 h).

## UPDATE 2026-11-10 ~17:15 UTC — lead rulings 1–4 executed

1. **Disk:** approved up to 15 GB for the look. `phase_look` now refuses to claim if free space is below 60 + 15 GB.
   Free space now: 86 GB.
2. **Weather drift:** the lead ruled the fix correct. The disclosure is in `sonde_m1_protocol.md` §4 (forcings rebuilt
   2026-10-27; the look reproduces the pre-rebuild inputs). The pass rule stays Δ = 0 on all 12 jobs. If
   chronos2_zs_uni still differs, report it before going further.
3. **Open-Meteo budget:** the pull was restarted with `--day-cap 4800`, 60 % of the shared 8,000/day ledger cap.
   - The cap applies to the rolling total of all tags, so M1 yields whenever M6 draws on the budget.
   - **Faster sources checked:**
     - Open-Meteo has no bulk discount: weight = (variables/10) × (days/14) per location, and multi-location calls
       are multiplied by the number of locations.
     - The Historical Forecast API stitches the earliest hours of each run, so it gives lead ≈ 0 and not lead days
       1–7. It is not equivalent.
     - No other NWP is on disk (only M1's 50-site pilot).
     - **NOAA GFS on AWS Open Data** (`noaa-gfs-bdp-pds`, anonymous, `.idx` byte ranges; 00z, f006–f168,
       6-hourly TMAX/TMIN):
       - 1° resolution: about 120 KB per step, ≈ 3.2 GB transferred over 944 runs and about 53k range requests; it
         could be done in about 1 day including build and an agreement check against the Open-Meteo pilot.
       - 0.25° resolution: ≈ 36 GB, which is too large.
       - Downsides: it is a different product from Open-Meteo's (1° vs about 0.25°), the new sites at look time would
         have to use the same source, eccodes/cfgrib is not in `sentinel2` (it needs a separate env), and archive
         paths change at the GFS v16 upgrade (about 2021-03-22). That upgrade is a model change inside the
         bias-correction window for **both** sources; bias correction should use only post-upgrade data.
     - Cheaper option inside Open-Meteo: request only the lead days the models use. Dropping leads 4 and 6 saves
       about 29 %, but only if the features use point-horizon forecasts and not the full 1–7-day trajectory. That is
       not decided.
   - **Recommendation:** stay with Open-Meteo at the 60 % cap (one product for the val, train and new sites; no new
     dependencies). Switch to the AWS 1° route only if the lead wants the faster ETA more than product consistency.
   - **Progress:** 399/5,718 chunks (279/918 val-site chunks; 120 come from the pilot overlap).
   - **Revised M1-N ETA (60 % cap):**
     - val-site NWP about 11-13, when val-only cycles can start;
     - full train-site NWP for the basin-group CV about 11-27;
     - **M1-N freeze candidate about 12-04 to 12-08.**

     With the AWS 1° route: NWP complete about 11-12, candidate about 11-26.
4. **Per-site coverage + bootstrap CIs** were added to `sonde_pi_report.py`:
   - `numeric_site_metrics_<block>.csv`;
   - medians over sites (≥ 30 rows) of coverage, MAE, RMSE and MAE skill;
   - basin-group cluster bootstrap 95 % CIs (B = 2,000) on the pooled MAE, RMSE, coverage and skills, computed on
     per-group sums with draws identical to the frozen `bootstrap_ci` (unit test).
   - The options are `--ci-for`, `--min-site-n` and `--boot-seed`. `m1n_protocol.md` §4a states it.
   - The look computes CIs for every model (no `--ci-for` in the look command); per-row tables are written for the
     candidate seeds only. Tests: `tests/test_sonde_pi_report.py` (4 pass).

## UPDATE 2026-11-10 ~16:00 UTC — hold ruling executed; numeric-forecast requirement; rehearsal findings; M1-N NWP pull

**Live trial: ON HOLD (user decision).** Nothing was frozen, run live, installed in cron, self-tested or submitted.
The hold is in `ew_v2_live_deviations.md` and the amendment status. Issuing hash `9ee1a2ac…` and scorer `ef598860…`
are unchanged and still freezable. A later start needs a dated amendment of the trial window.

**Disk ruling: no deletions.** An inventory of `sentinel_runs/sonde` against the registry, every preregistration and
freeze JSON, RESULTS/TRAINING_LOG/review reports and the run manifests found **0 GB of unreferenced superseded runs**.
- The only candidate, `full_pred_chronos_ft_multi_val` (0.79 GB), is an input of `full_events_score_val`
  (`score_manifest.json`), which registry rows cite. It is kept.
- Nothing was deleted. The look's ~13 GB of sealed test forecasts (+ ~1.2 GB per-row numeric tables) therefore needs
  a **lead disk decision** over M1's 10 GB new-data cap. Free space now: 87 GB on /home (floor 60 GB).
- M1 new data so far: new-site block 633 MB; NWP pull ≈ 1 GB expected.

**Numeric forecasts (user requirement relayed 2026-11-10).**
- **Script:** `scripts/eval/sonde_pi_report.py`. It reads saved forecasts and the frozen common rows only (no fitting,
  no evaluate.py edit). For every model, reference, variable and h it reports:
  - the median in original units;
  - 50/80/90 % prediction intervals (q25–q75, q10–q90, q05–q95) with coverage and mean width;
  - MAE, RMSE, bias;
  - MAE/RMSE skill vs persistence and vs climatology.

  It also writes per-row numeric forecasts (site, issue date, h, obs, q05…q95). Quantiles are sorted with the frozen
  evaluator's crossing guard.
- **Wiring:** the script is a step of the Test-A look (after scoring, before `SCORED.json`) and is in the look's
  import-closure hash. Tests: `tests/test_sonde_pi_report.py`.
- **Protocols:** `sonde_m1_protocol.md` §4 (look step 5); `m1n_protocol.md` v0.5 §4a + deviation row (secondary and
  descriptive; EW-v3-N probabilities additional).
- **Val check** (candidate ens3w s0, val block, common rows; `rehearsal_val/numeric_metrics_val_ens3w_s0.csv`):

  | var | h | MAE | RMSE | PI50 / PI80 / PI90 cov | MAE skill vs pers. | vs clim. |
  |---|---|---|---|---|---|---|
  | temp °C | 1 / 7 | 0.50 / 1.19 | 0.73 / 1.65 | .55/.85/.94 ; .49/.79/.89 | 0.10 / 0.22 | 0.72 / 0.34 |
  | DO-mean mg/L | 1 / 7 | 0.24 / 0.56 | 0.40 / 0.84 | .53/.84/.93 ; .49/.79/.89 | 0.09 / 0.17 | 0.70 / 0.30 |
  | SC µS/cm | 1 / 7 | 170 / 427 | 779 / 1683 | .55/.85/.93 ; .50/.81/.90 | 0.06 / 0.13 | 0.78 / 0.46 |
  | pH | 1 / 7 | 0.052 / 0.117 | 0.090 / 0.181 | .58/.79/.89 ; .50/.79/.89 | −0.06 / 0.09 | 0.73 / 0.40 |
  | turbidity-mean FNU | 1 / 7 | 7.5 / 12.9 | 43.8 / 61.5 | .51/.83/.92 ; .49/.79/.89 | 0.16 / 0.26 | 0.58 / 0.27 |
  | nitrate mg/L | 1 / 7 | 0.17 / 0.52 | 0.38 / 1.02 | .55/.85/.94 ; .49/.80/.89 | 0.05 / 0.09 | 0.85 / 0.54 |

  The intervals are close to nominal. The val caveat W6 applies: FT-Chronos and LSTM early-stop on val.
- **TODO before the M1-N freeze:** M1-N §4a also promises site-median coverage and HUC4-cluster bootstrap CIs on MAE/RMSE skill for the frozen model. The script currently gives pooled values only, and this must be added (and run on val/CV) before `FREEZE_m1n`. The Test-A look reports pooled values, as `sonde_m1_protocol.md` §4 states.

**Rehearsal findings (item 3): two problems found and fixed before any freeze.**
- **First-pass identity** (`rehearsal_val/identity_v1_firstpass.json`):
  - exact reproduction (Δ = 0) for gbm, lstm_s0 and chronos2_ft_multi s0/s1/s2;
  - **mismatch** for gbm_weather (max |Δ| 1.47), lstm_weather s0–s2 (up to 6.96) and chronos2_zs_uni (21.4).
- **Cause 1: forcings rebuild.** `forcings_daily` was rebuilt on 2026-10-27 (ERA5 added for sites without gridMET).
  The weather models were trained and val-scored on 2026-10-24 with 1,826 of 1,906 sites forced; now 1,905 are.
  - Diagnostic re-prediction of gbm_weather on val: every changed row is at the 79 ERA5-forced sites, and every other
    site has Δ = 0.
  - **Fix:** `sonde_predict.py --weather-exclude-sources era5_seamless` (a process-local wrap of the frozen loader,
    recorded in the manifest) reproduces the training-time inputs. gbm_weather re-predicted with it: **Δ = 0 in all 8
    files**.
  - The plan passes this flag to all 4 weather jobs. Rule and evidence: `sonde_m1_protocol.md` §4.
  - Lead option: instead refresh the val evidence under the current forcings. I did not choose that, because it would
    change the candidate's val numbers.
  - **Cross-workstream note:** any other model trained before 2026-10-27 with `use_weather` has the same drift.
- **Cause 2: comparison bug.** chronos2_zs_uni was compared with the zs_multi file in the same directory (prefix-less
  glob). It is now an exact file-name match plus per-row stats (rows changed, sites changed).
- **Plan regenerated** with the final code: `sonde_m1_testA_plan.json` sha `2cf2ddff…`, code hash `f40fdcb3…`
  (40 files, now including `sonde_pi_report.py` and the new `sonde_predict.py`). The stale `37e552ba…` plan was moved
  to scratchpad `m1live/sonde_m1_testA_plan_v1_stale.json`. Registry prereg_snapshot `aa728159c875`.
- **Rehearsal restarted** on that plan (16:19 UTC). It re-runs gbm_weather, lstm_weather s0–s2 and chronos2_zs_uni,
  then runs chronos2_zs_multi and timesfm25_zs. Wall time is dominated by waits for GPU 0/1, whose locks are held by
  M4/M5 jobs. The earlier 3–5 h "jobs" were mostly lock waits.
- **Status at 16:45 UTC:**
  - gbm_weather in the rehearsal: **Δ = 0**.
  - lstm_weather_s0 is waiting for a GPU lock. It then runs s1, s2, chronos2_zs_uni, chronos2_zs_multi and
    timesfm25_zs.
  - Results go to `sentinel_runs/sonde/m1_testlook/rehearsal_val/identity.json` (per-file max |Δ| plus row stats) and
    the scratchpad log `m1live/m1_rehearsal.out`. The run is resume-safe.
  - **Pass criterion for the pre-test review:** Δ = 0 for all 12 jobs. If chronos2_zs_uni still differs after the
    compare fix, the look must not proceed until that is explained.

**M1-N archived GFS pull: STARTED** (approved). Command:
`download_nwp_archive.py download --sites-file raw/openmeteo_nwp/sites_input_m1n_trainval.csv --models gfs_global
--variables temperature_2m --end 2023-09-30 --tag m1n_trainval`.
- **Scope:** 1,906 split train/val sites (1,600 train + 306 val; test-role sites excluded; reserved-site assertion
  passed). GFS 2-m temperature, lead days 1–7, 2021-03 → 2023-09-30 (the bias-correction window plus the val block).
  No Test-A period; new sites only at look time.
- **Code change:** `--variables` and `--tag` options were added to the (unfrozen) downloader.
- **ETA correction:** the plan's 2–3 days was wrong. The budget-ledgered plan is 5,718 chunks = 87,771 weighted calls,
  about **11 days** at 8,000 a day. Val-role sites are pulled first (about 1.8 days), so NWP-based validation on the
  val sites can start then. The train-role sites, needed for the basin-group CV, finish around 2026-11-21.
- **Log:** scratchpad `m1n_nwp_pull.out` and `raw/openmeteo_nwp/download_nwp_archive.log`.

## UPDATE 2026-11-10 ~02:30 UTC — S-1 fixed; new-site data locked; M1 Test-A look prepared

**Live.**
- **Issuing side:** hash check PASSED. The freeze waits for the user's OTS approval and the self-test.
- **S-1** (`score_live.py` only; issuing hash still `9ee1a2ac…`):
  - zero-row issue files are skipped;
  - days are classified from hash-log rows;
  - every issue file is test-loaded before the claim.
- **Rehearsals (scorer `ef598860…`):** all three looks sealed with 2 zero-row days counted as issued: the fixture
  2026-11-12 (empty USGS fetch) and the real late-data case 2026-11-11.
- **§10:** now states that the future-dated dry row is not an on-time example, and that dry mode accepts future dates
  while live refuses them.

**New-site block.**
- **Pull:** 729 of 729 sites, 0 failed. The earlier "17 failed" was a mistaken count of progress lines.
- **Manifest:** content pin `e615a10c…` over 1,829 files, 633 MB.
- **Locked:** files 000, directories 0300; a read raises PermissionError. Scorer-only unlock.
- **Registry snapshot:** `36848320a4c1`.

**Item 3: original M1 Test-A look (report-only).**
- **Plan:** `docs/preregistration/sonde_m1_testA_plan.json` (`37e552ba…`; 12 predict jobs, 6 ensembles,
  9 references, 3 strict controls, 3 candidate seeds). Checkpoints cross-checked against the candidate.
  - **The plan's recorded code hash is stale:** the script was edited afterwards (streaming rehearsal). Regenerate the
    plan when the script is final (delete the plan, then `--phase plan`).
- **Orchestrator:** `scripts/eval/sonde_m1_testlook.py`, hardened (claim with pid/host, sealed predictions, abort,
  rescore refused, full import-closure hash, FREEZE_sonde-only unlock).
- **Disclosure:** the EW-pass DO-label exposure is in `sonde_m1_protocol.md` §4.
- **Streaming val rehearsal RUNNING:** scratchpad `m1live/m1_rehearsal.out`. Identity report:
  `sentinel_runs/sonde/m1_testlook/rehearsal_val/identity.json`; resume-safe.
- **Disk:** the real look needs about 13 GB of test forecasts, which needs the lead's OK over the 10 GB cap or a
  clean-up of superseded M1 runs.
- **Then:** pre-test review, and the lead writes `FREEZE_sonde.json` with `testlook_plan_sha256` and
  `testlook_code_sha256`.

**M1-N next steps (not started; for the lead).**
1. Archived NWP (GFS `temperature_2m` lead days 1–7, from 2021-03) for split train/val sites, and for new sites at
   look time only, via `download_nwp_archive.py`. It is budget-ledgered (about 8k calls a day), so about 2–3 days. **[Superseded 2026-11-10: about 11 days; val-role sites first, about 1.8 days; see the top update.]**
2. Bias-correction maps fitted on the 2021-03..09 train-block overlap.
3. Basin-group CV over train/val roles (selection power).
4. Perfect-prog training and NWP-based validation for every comparator.
5. Up to 16 configurations per variable group; GPU ≤ 2 h per job.

ETA to an M1-N freeze candidate: about 7–10 working days after the NWP pull. **[2026-11-10: val-site NWP is ready about 11-12, so val-only cycles can start then; the CV over train sites needs the full pull, about 11-21; the freeze candidate is realistically about 11-28 to 12-02.]**

## UPDATE 2026-11-10 ~03:30 UTC — third round (lead rulings + delta review C-E1/E2, C-F1/F2)

**Live (amendment v0.4).**
- **C-E1:** the reserved-helper import was reverted from `live_data.py`. The issuing compute code is again identical to
  `27496e20…`, and the test `test_live_config_sites_never_reserved` replaces the import.
- **C-E2:**
  - a deleted-proof day is "issued but not eligible";
  - four day classes are reported in every look;
  - the forecast-window off-by-one ([D, D+L−1]) is fixed;
  - the > 3 % co-primary bound is added.
- **Recommended items, all done:** restore from the public copy (`ots_verify.check_refs`); dormant `push_evidence.py`
  (enabled only via the freeze record); environment read-only after the freeze (§11.6b); first-run qualifier table in
  §12.
- **Final issuing code hash: `9ee1a2ac…`** (52 files; adds `push_evidence.py` and the `run_daily.sh` call).
  §10 re-runs on this hash: done.

**Shared-tier guard (lead ruling; applied).**
- Mode 000 on the 27 + 21 raw IV new-site directories (lock records in those trees).
- `sentinel/data/reserved.py` (pinned helper + pre-claim audit), added to the non-frozen loaders: M1 events
  feasibility, onset counts, virtual; Sim `sim/data`, `causal/diel`; Downstream `killalert`; Fuse `tier_w`, `wp1`,
  `wp1_power`; Platform `data_access`.
  - Each file was verified against every freeze and pin record.
  - `sonde_score.py` (hashed by the M1 candidate) was not edited.
  - The M3 and NEON loaders read NEON tiers only.
- Tests: `tests/test_reserved_loaders.py`, including the repo-wide static guard.
- Owner notes appended to `sim.md`, `downstream.md`, `fuse.md`, `M3-microbe.md`, `M6-hydromet.md`.
- **Platform has no handoff file:** the lead should tell its owner that `data_access` now hides reserved sites.
- Pre-claim audit run on 2026-11-10: 0 hits.

**EW-v3 / M1-N (protocols v0.4, §2c).**
- **Tiers v2:** hard 85 (adds the 12 v1 outcome-level sites incl. 07288048), legacy 46; primary 644, S1b 598, S1 538.
- **Power:** primary H1 ≈ 46–50 (×0.5), so more likely than not underpowered. If the IV rule is not confirmed:
  ≈ 17–18, so certainly underpowered.
- **IV-rule development phase DONE: no variant met both criteria** (best R1/R3: 0.976 within 0.1 mg/L, onset Jaccard
  0.72; needed ≥ 0.99 and ≥ 0.90).
  - No confirmation was run; the confirmation half is untouched.
  - **Option (i) applied automatically: EW-v3-N H1 is NON-CONFIRMATORY.** EW-v3-N is descriptive as a whole.
  - **M1-N primary = DV-based sites** (DO 51 sites / 24 HUC4; SC 155 / 54); IV-only sites form a secondary cell.
- **Re-snapshot:** registry `ew_v3_m1n/prereg_snapshot` `1aac08c1ad24` (23 files incl. both split and audit files).
- **Live hash check (C-E1/C-E2):**
  - issuing code `9ee1a2ac…`; compute path identical to `27496e20…` (`live_data.py` identical to the v0.3 snapshot);
  - scorer `0fef84d5…`; environment `e950cae0…`; config `7693424f…`; amendment v0.4 `bd5e6897…`.
  - §10 re-run on `9ee1a2ac…`: replay identical (3.9e-6); dry cycle row 8 issued. 55 tests pass.
  - Exact live diff: `m1live/live_delta_v03_to_v04.diff`.

## UPDATE 2026-11-10 ~01:30 UTC — second review round (live delta FAIL + EW-v3/M1-N FAIL) addressed; ready for delta reviews

**Live trial (amendment v0.3).**
- **C-D1, OTS-only third-party eligibility:** `ots_verify.py`, outside the issuing closure.
  - Each proof is upgraded from the calendars, then the Bitcoin merkle root and block time are checked by two
    explorers.
  - Each day's proof files are hash-committed into the next hash-log row (`evidence_refs`), so deleting a proof is
    detected (`EvidenceDestroyed`).
  - The scorer refuses the look if `self_test()` fails.
  - The GitHub backend is dropped.
  - **Finding:** the reviewer's "fetch the proof from the calendars by row hash" is not possible in the OTS protocol.
    Upgrades are keyed by the commitment inside each proof (derived from the calendar's reply and a nonce), hence the
    hash-commitment design.
- **C-D2, dedicated environment:** `sentinel2-live` (clone of `sentinel2` + opentimestamps-client 0.8.0; about 8 GB of
  disk). Lockfiles are in `configs/live/`, and `run_daily.sh` uses this environment. The freeze draft sets
  `timestamp_backends: ["ots"]` from the start. **`ots_selftest.py` is prepared and NOT run (needs the owner's
  approval).**
- **W-D1:** qualifier map re-derived from legacy agreement (`derive_qualifier_map.py`): FORCEINTERPOLATION kept;
  EQUIP/MAINT masked. Table-driven test added.
- **W-D3:** separate label-pin records `FREEZE_ew_live_labels_<look>.json`; the label builder needs the scorer freeze;
  builder and verifier are in the scorer hash.
- **W-D5/6/7:** multi-series sensitivity analysis; count_only checked against its seal; missing days vs provisional
  onsets.
- **§10 re-run on the final code:** code `27496e20…`, environment `e950cae0…`.
  - Replay identical in 9/9 cells (≤ 3.9e-6).
  - Daily cycle 11-10 issued (row 6).
  - Forced failure 11-01 → failed row 7.
  - Rehearsal of all 3 looks sealed.
  - 33 + 2 tests pass.
- **Diffs:** cumulative `m1live/live_delta_review_v03_cumulative.diff` (base = pre-review copy); snapshot of this
  state in `m1live/snap_v03/`.

**EW-v3 / M1-N (protocols v0.3, §2b).**
- **v2 independence:** verified (`sonde_newsites_v1_audit.json`). Only `build_mc_benchmark_v2` reads v1, and it names 0
  new sites. No v2 checkpoint, run, threshold or selection uses v1.
- **Legacy tier = 57 sites** (v1 value-level use): kept in the primary; excluded in S1b, which has equal prominence.
- **Hard tier = 73:** adds 1 chloride-eligible new site whose derived H2 series exists in the shared tier. 90 October
  regressions at new sites are never used, and H2 is re-fitted inside the look.
- **Populations** (`sonde_newsites_tiers.json`): primary 656, S1b 599, S1 548, S2 729.
- **Projected H1 onsets (×0.5):** primary ≈ 50–54; S1b ≈ 37–40. Underpowered is about as likely as not.
- **C-N2:** archived issued GFS forecasts in every validation step; perfect-prog for training only; bias correction on
  train-block overlap.
- **C-N3:** `newsite_guard` now enforces (claim read from disk, this pid/host, not sealed/aborted) and has tests;
  `lock_newsites.py` (run after the manifest).
- **Shared-tier chmod:** a PROPOSAL only (2c below), awaiting the lead.
- **W-N2:** unified reference window. **W-N4:** spatial-only primary + S-T sensitivity; grab-sample and Orbit/Lite
  overlaps disclosed. **W-N5:** DST IV gaps disclosed; IV-rule validation running.
- **Registry snapshot:** `ew_v3_m1n/prereg_snapshot` `e24f41042cef`.

## UPDATE 2026-11-09 late evening — lead decisions executed (read first)

**1. New-site pull (approved): RUNNING.**
- Process: PID 1601264, detached; log in scratchpad `m1live/newsites_iv.out`.
- Frozen site list (metadata only): `docs/preregistration/sonde_newsites_list.json`, sha `6200a77c…`.
  - 729 sites: 424 stream / 153 lake / 133 estuary / 19 wetland.
  - By variable: DO 393, SC 661, NO3 36.
  - Excluded: split sites, 101 sites named in other workstreams' pre-registrations, other/unknown types.
- Scripts:
  - list: `scripts/data/build_newsite_list.py`;
  - pull: `scripts/data/download_newsites.py` (legacy DV done: 499 sites; legacy IV running);
  - progress: one `iv_done/<site>.json` per finished site (≈ 115 sites/h → done ≈ 11-10 early morning, well before 11-16).
- Storage: protected dir `processed/sonde_newsites/{dv,hourly,daily_iv}`. Raw IV is deleted per site (disk ≈ 60 MB so far).
- **When IV is done, run `python -B scripts/data/download_newsites.py manifest`.** It writes the content pin plus the
  integrity table (counts and coverage only). Then re-run `iv` once to retry failed sites.
- Access guard: `sentinel/data/newsite_guard.py`. It allows score-only reads with a lead freeze plus an open claim.
  Split loaders cannot reach these sites.
- **Rule: NO onsets, exceedances, value statistics or plots on new sites. Integrity counts only.**
- Note for the lead: about 470 of the new sites also have DV files in the shared `processed/sonde_daily` (built before
  this decision). M1 loaders cannot load them because they are not in the split. Other workstreams should be told the
  list is reserved.

**2. Pre-registrations written (for review, before any model sees new sites).**
- `docs/preregistration/ew_v3_protocol.md` (EW-v3-N).
- `docs/preregistration/m1n_protocol.md` (M1-N forecasting / virtual sensing).
- Registry `ew_v3_m1n/prereg_snapshot` (id 3abd1e26a883) holds the hashes of both protocols, the site list, the builder,
  the downloader and the guard.
- Design B is a site-specific 2-year reference with issues on the remainder. It was chosen from metadata power only (DO:
  243 sites / ≈ 914 issue site-years / 66 HUC4) over design A (85 sites / ≈ 306).
- Archived NWP is given to every comparator by perfect-prog: observed weather in training, issued GFS temperature
  (from 2021-03) at test.

**2b. New-site exposure audit (after M6's report).**
- Record: `docs/preregistration/sonde_newsites_overlap.json`, sha `9c72139d…`, built by `scripts/data/newsite_overlap.py`
  (metadata only).
- **Exact ID overlap with track N: 0.** The builder had already excluded the 21 named track-N stations.
- **113 sites are in the DEV smoke split.** The builder skipped all `sonde_split*` files; this oversight is disclosed.
- **3 sites were ever loaded by a run.**
- **Co-located track-N stations:** 4 within 100 m, 11 within 500 m, 24 within 1 km.
- **Powell DO-derived labels:** 48 sites.
- **Tiers:** hard tier = 72 sites; full exposure set = 180.
- **Pre-declared populations:**
  - **primary = minus the hard tier (657 sites)**: DO 208 sites, ≈ 676 issue site-years, H1 ≈ 50–54;
  - S1 = minus the full set (549 sites): H1 ≈ 33–36, likely underpowered;
  - S2 = all 729, descriptive.
- Both protocols are now v0.2 (§2a). Registry `ew_v3_m1n/prereg_snapshot` id 54e6dd711dda supersedes the earlier rows.
- **Lead decision (2026-11-09): primary = minus the hard tier (657 sites); S1 sensitivity; S2 descriptive.**
  - Rationale: exposure = actual data access or derived labels. Nominal DEV-split membership with no recorded load is
    not exposure.
  - Recorded in §2a of both protocols. Registry snapshot 8c6abf77e676.
  - Builder fixed: only `sonde_split.json` and the list's own files are skipped; test `tests/test_newsite_list.py`.
  - The frozen list is unchanged, and the 113-site oversight stays disclosed.

**2c. SHARED-TIER GUARD: PROPOSAL FOR THE LEAD (nothing chmod-ed; awaiting approval; 2026-11-10).**

*Copies of new-site files outside `processed/sonde_newsites`:*

| Location | New-site copies |
|---|---|
| `processed/sonde_daily_refresh_20261108/sites/` | 422 files |
| `processed/sonde_daily_20261022/sites/` | 422 files |
| `processed/sonde_hourly_refresh_20261108/sites/` | 27 files |
| `processed/sonde_hourly_20261024/sites/` | 27 files |
| `raw/usgs_iv/` | 27 site directories |
| `raw/usgs_iv_refresh_20261108/` | 21 site directories |

*Readers of those tiers* (65 files, classified):
- **Frozen, content-hashing the whole tier:**
  - `FREEZE_ew` (`sonde_events.verify_snapshots` globs `sonde_daily` / `sonde_hourly`; needed for the `test_ab` pass,
    ≥ 2027-06-30);
  - `FREEZE_hydromet` (pins `sonde_daily`; `hydromet/data.py` is index-driven).

  Mode 000 on any file there would break those frozen scorers.
- **Running now:** M6 `hydromet_baselines` / `hydromet_lstm` (track R) and Sim `sim_causal_step1`. No fd is open on
  these tiers at the moment, but they read `sonde_daily` per site by index.
- **Non-frozen, index- or glob-driven** (need the helper):
  - M1: `scripts/live/*` (config = split sites only), `sonde_onset_counts`, `sonde_virtual`, `sonde_score`,
    `events_feasibility_sonde` / `_external`;
  - Sim: `sim/data.py`, `sim_causal_real.py`, `sim_e1/e3`, `analysis/causal/{ladder,diel}`;
  - Downstream: `killalert.py`;
  - Fuse: `fuse_tiers.py`, `fuse_tier_w*`, `fuse_wp1_power`;
  - Platform: `data_access.py`, API and views;
  - Others: `neon_aquatic.py`, `microbe_wq.py`;
  - data builders: `build_chloride_derived`, `build_sonde_*`, `build_hydromet_wqp_national`.

*Proposed plan:*
1. **Mode 000 only where no frozen or running reader globs the directory.** That is the 27 + 21 new-site directories in
   `raw/usgs_iv{,_refresh_20261108}` (no FREEZE pins raw IV; only non-frozen builders glob it, and none is running),
   plus `processed/sonde_newsites` after its manifest (`lock_newsites.py --lock`).
   **No chmod in `sonde_daily` / `sonde_hourly`**: frozen readers glob and content-hash them.
2. **Hash-pinned exclusion helper.** `sentinel.data.newsite_guard.exclude_reserved` / `assert_not_newsite` goes into
   every NON-frozen loader listed above. The lead asks each owner (Sim, Downstream, Fuse, Platform, M3, NEON); M1 does
   its own. Each loader gets a test that feeds it a list containing a reserved site and expects a refusal or a drop.
   Frozen readers are left untouched: they are split-gated (EW) or pinned (HydroMet).
3. **Pre-claim audit** (in both protocols, §2b). Before any EW-v3 / M1-N claim, every registry row and every
   `data_info` / run site list written after 2026-11-09 is scanned for reserved IDs; any hit refuses the claim.
4. **Restore around frozen scorers.** If the lead later wants mode 000 on the `sonde_daily` / `hourly` copies as well,
   permissions are restored only around each frozen scorer that hashes the tier (EW `test_ab`, HydroMet looks), each
   restore logged as a dated step.

**3. Original M1 Test-A forecasting look (report-only): NOT started.** The candidate
(`sonde_m1_freeze_candidate.json`) exists. Still needed: a hardened testlook wrapper around `sonde_predict.py` /
`sonde_score.py`, and the EW-pass DO-label exposure disclosure. Next after the live delta review.

**4. EA allocation (approved)** is recorded in `docs/plans/ea_england_allocation.md`: ≤ 20 % of non-M6-test groups,
pre-2023 data, seeded draw after M6's split freeze.

**5. Live trial: review FAILED → all of C-L1..C-L3 and W-L1..W-L13 addressed. Delta review pending.**
- Amendment v0.2: `ew_v2_live_amendment.md`; post-freeze deviations file `ew_v2_live_deviations.md`.
- Config `7693424f…`: span to 2028-09-30, with 4 looks defined.
- New scripts:
  - `stamp.py`: OpenTimestamps / GitHub, hash-only, **disabled until the owner approves**;
  - `log_status.py`: status row for every scheduled day;
  - `build_label_snapshot.py`: one frozen label path + identity check;
  - `score_live.py`: hardened scorer, outside the issuing closure.
- Rewritten: `verify_hashlog.py` (no git dates; missing / failed days, orphans, timestamp evidence) and `run_daily.sh`
  (EXIT trap; `CRON_TZ=UTC`).
- Changed:
  - `live_common.py`: span; issuing closure; status rows; environment lock; static pins; placeholder refusal;
  - `live_data.py`: series-map fallback; unknown qualifiers masked; REVISED kept; missing pin refused;
  - `issue.py`: atomic day; pins; environment lock.
- Tests: `tests/test_live.py` passes 13/13, including the extended span and issue_rows ⊇ event_rows.
- Delta diff for the reviewer: scratchpad `m1live/live_delta_review.diff`. The pre-fix copies are in `m1live/prefix/`.
- **Re-runs on the final code** (code `3ca4adfd…`, config `7693424f…`, environment `b25fdf5e…`, scorer `67be86fb…`;
  details in amendment §10):
  - val replay: rows identical in 9/9 cells, max |Δp| 3.9e-6;
  - full dry daily cycle for 11-09: issued, nothing submitted, verify clean;
  - forced failure on 11-02 produced a failed row;
  - scorer rehearsal of prov_interim / count_only / final_wy2027: claim → seal; rerun refused; abort path exercised;
  - label-path identity check, which found the REVISED qualifier masking bug (fixed).
- **Remaining before the freeze:**
  - the lead's delta review;
  - the owner's sign-off on the freeze, push and cron;
  - the owner's approval of daily hash-only OTS submissions;
  - the frozen all-roles series map.

  Target: freeze-ready by about 11-20.
- **Next for M1 (in order):**
  1. When the IV pull ends: `download_newsites.py manifest`, then an `iv` rerun for failed sites (integrity counts
     only).
  2. Validate the IV-derived daily rule vs DV on train-role split sites (ew_v3 §2).
  3. Hardened testlook wrapper + freeze candidate for the original M1 Test-A forecasting look (report-only).
  4. After the EW-v3 / M1-N review: the NWP pull for training and new sites, and val cycles under the caps.

## CURRENT STATUS (2026-11-09 evening) — read this first
- **EW-v2 Test-A is spent** (single pass 2026-11-09, all hazards underpowered, no claim). Test-A rows are never used for training or tuning. Test-B (WY2026) at **test-role** sites is reserved for the `test_ab` pooled H1 pass (≥ 2027-06-30) and the event-layer tracks.
- **Nothing is running.** No FREEZE file was written by M1. No test block was loaded. The live dry run used train/val-role sites only.
- **Live-trial amendment (due before 2026-12-01): READY FOR THE LEAD'S FREEZE.**
  - Amendment: `docs/preregistration/ew_v2_live_amendment.md`. It states that Test-A results were known, re-tunes no thresholds, and lists every post-Test-A decision with its train/val basis.
  - Config (0444): `configs/live/ew_v2_live.json`, sha256 `a6467a31…`. Sites: H1 379, H4 43, H2 25 (all roles).
  - Pipeline: `scripts/live/`:
    - `fetch_usgs.py` (OGC `daily`, 20 sites/request) and `fetch_gridmet.py`;
    - `issue.py` with modes `live` / `dryrun` / `replay`;
    - `verify_hashlog.py`, `build_series_map.py`, `power_projection.py`, `freeze_draft.py`, `run_daily.sh`;
    - `live_common.py` (hash chain, import-closure code hash, guards) and `live_data.py`.
  - Dry-run evidence (amendment §10):
    - val replay reproduces calibration rows exactly; max |Δp| ≤ 4e-6;
    - OGC fetch vs snapshot: 0.16 % of values differ (provisional edits), after the frozen series map;
    - gridMET live values differ by gridMET's own revisions;
    - end-to-end dry issue, hash chain and guard refusals all OK.
  - Outputs: `sentinel_runs/live/ew_v2/dryrun/`.
  - **Lead steps (amendment §11):**
    1. `python -B scripts/live/build_series_map.py --roles train,val,test --out frozen`
    2. `python -B scripts/live/freeze_draft.py` (prints the draft; writes nothing)
    3. Write `FREEZE_ew_live.json`.
    4. Commit and push before 12-01 (include an empty `results/live/ew_v2_live_hashlog.jsonl`).
    5. Owner cron `0 12 * * * …/scripts/live/run_daily.sh live`.
    6. Push the hash log at least weekly.
  - **Still to write:** `scripts/live/score_live.py`, the hardened single-look scorer (amendment §9). It must be frozen before any WY2027 Approved label is retrieved, at the latest before the 2027-10-15 provisional interim.
  - **Key a-priori finding** (`power_projection.py`, registry `ew_v2_live/power`, train/val-role sites only):
    - H2 onsets fall mostly in Jul–Sep; only 8 % fall in Dec–Mar, so the "winter H2 trial" premise was wrong. H2's primary window is now the whole trial.
    - Projected WY2027 onsets (upper bound): H1 49–53, H4 5–7, H2 4–9. **The live trial is expected to be underpowered.** A count-only extension rule to WY2028 is pre-specified.
- **EA England:** M1 allocation proposal appended to `docs/plans/ea_england_allocation.md` (pending M6 and the lead; no overlap with M6's reservations). M1 has downloaded no EA data and trains on none until M6's split is frozen.
- **Disk:** about 100 GB free at 18:00. M1's new data so far is about 0.2 GB (`sentinel_runs/live`). Cap 10 GB.

## SOTA PLAN (lead directive `docs/plans/sota_campaign.md`; for lead review BEFORE any large download or compute)

### A. Diagnosis
**EW-v2 low event counts.** EW-v2 used the 2,678-site stream split. That is not why Test-A was underpowered. Metadata (USGS OGC `time-series-metadata`, 2026-11-09, no values) shows:
- **DO coverage.**
  - USGS has 1,430 sites with any DO record (daily or IV) in WY2008–26, totalling about 12,450 site-years.
  - The EW split already holds 861 of them, about **75 % of national DO site-years**.
  - The rest of the "5,500 daily sites" are temperature-only, groundwater, short-record or non-stream sites.
- **The binding constraints:**
  1. **Test cell size.** The test role is 20 % of basin groups × 2 water years (Test-A: 236 listed DO sites, ≈ 150 with rows, 22 basin groups).
  2. **Rarity.** H1 onsets are rare under the strict onset rule (3 clean days ≥ 4 mg/L, 7-day refractory): 0.15–0.16 onsets per observed site-year (train/val, `ew_v2_live/power`). The observed 43 matched the a-priori 44.
  3. **Nitrate and chloride.** Few sites are eligible for H4/H2: 186 USGS daily nitrate sites nationally; 52 SC→Cl-eligible sites.
- **What is still unused (clean = in no `docs/preregistration/*.json` of any workstream; groundwater and springs excluded):**

  | Variable | Sites | Site-years | Of which IV-only (not in our DV tier) | Active WY2027 |
  |---|---|---|---|---|
  | DO | 435 | ≈ 2,095 | 234 sites, 1,193 site-years | 115 |
  | NO3 | 44 | 118 | — | 12 |
  | SC | 815 | ≈ 4,512 | 389 sites, 1,908 site-years | 211 |

  - In total 428 sites need an IV pull.
  - Excluded: 88 DO sites that already appear in other workstreams' files (RiverPulse 39, Fuse tier C 37, HydroMet track N 21, D2n 19).

**EW model.** Val skill over climatology is small: the H1 stack's val-OOF BSS is +0.021 [−0.010, +0.041]. The DO hazards have no future-weather information. Probabilistic skill was not established even in the one adequately sized Test-A cell.

**M1 forecasting is not SOTA.** On the unseen val cell, ens3w vs 9 single-model references:
- clears the bar at most horizons for SC (4–5/5) and turbidity mean (4–5/5);
- is weaker for nitrate (2–3/5) and DO mean (2/5);
- fails for DO min (0/5) and temperature (0/5);
- has even fewer wins against the baseline-ensemble controls.

Causes:
1. The val-role cell has only 8–13 basin groups (nitrate 3), so CIs are wide and the selection signal is weak.
2. At 1–7 d, persistence/damped-anomaly plus GBM-with-weather already capture the information available from lags. Chronos-FT and the LSTM add no new information, because they get no future weather, upstream context or larger corpus.
3. The primary is very strict: CI > 0 at every h ∈ {1, 2, 3, 5, 7} for ≥ 3 of 5 variables against the best of 9 references.

Per sota_campaign rule 5 (diagnose before adding capacity), the levers below add information, not parameters.

### B. Levers ranked by expected gain per GPU-hour
1. **New-site block (data; 0 GPU-h).** National USGS sites never in any split (table above) become a large unseen test population for EW-v3 and M1. This is the single biggest power lever.
   - Pull: IV (and DV where it exists) for about 428 sites without a daily tier.
   - Route: the **legacy IV service before 2026-11-16** (about 2,100 site-year requests, a few hours; existing `download_usgs_iv.py`). After that date, the OGC `continuous` endpoint: about 15k requests, ≈ 15 h at 1,000 requests/h.
   - Size: ≈ 2–3 GB raw; aggregate to daily + hourly (≈ 0.3–0.5 GB) and delete the raw. Within the 10 GB cap.
   - **Needs lead approval by about 11-12 to use the legacy service.**
   - The pull writes into a protected directory, with no summary statistics printed (test data).
2. **Basin-group cross-validation for selection (method; 0–5 GPU-h).** Replace the 8–13-group val-role cell with 5-fold basin-group CV over train+val-role sites × val block. This uses existing data only, as more power for *selection*. Final models are refit; the test stays unseen.
3. **Archived NWP forecasts as known-future covariates (data; ≈ 10–20 GPU-h).** Source: `raw/openmeteo_nwp`, which is already on disk. Covers 3–14 d forecasting and the DO hazards (heat and storms drive DO crashes). Every comparator gets them too (GBM, LSTM, Chronos-2 covariates). This is where sequence or foundation models with future covariates can plausibly beat persistence and GBM. Lag features cannot.
4. **Upstream graph context (data; ≈ 5 GPU-h).** Recent values at flow-connected upstream sensors (NLDI/NHDPlus edges already built), given to all models.
5. **Sonde pretraining corpus (method; ≈ 30 GPU-h in ≤ 2 h chunks).** Hourly tier of all split train-role sites, plus EA rule-3 data (after M6's freeze, if the allocation is agreed), plus NEON train sites (check with Fuse). Never new-site-block or test-role data. Used for Chronos-2 fine-tuning with group attention across co-located sensors.
6. **Harder tracks.** Each is pre-registered with its practical justification **before** any result on it:
   - X3 extremes and X1/X2 storm/heat strata (already registered, M1-X);
   - X4 21/30-d horizons (needs NWP and seasonal context);
   - **N-b unsensed-variable nowcasting at new sites** (nitrate/turbidity/DO from co-located T/SC/Q + catchment; published nitrate KGE ≈ 0.18).

### C. Track definitions, unseen test blocks and power (train/val counts only; no test value read)
**EW-v3-N (primary confirmatory EW track).**
- **Test population:** the clean new-site block: stream, lake/reservoir and estuary types (types resolved from OGC monitoring-locations metadata). Approved values, WY2008–2025. Basin groups built for the new sites like the split.
  - Primary cell: new sites whose basin group contains no train/val-role split site.
  - Secondary cell: all new sites.
  - Lakes and estuaries are separate pre-declared cells.
- **Hazards:** the same frozen EW family {H1, H2, H4}, the same power rule (≥ 50 evaluable onsets, ≥ 2 basin groups) and Holm m = 3. No tier substitution: the Test-A-informed < 4 mg/L tier is not promoted.
- **Power (train/val rate 0.148–0.161 per observed site-year):**

  | Hazard | Basis | Projected onsets | Expectation |
  |---|---|---|---|
  | H1 | 2,095 DO site-years, ×0.5 discount for gaps, non-stream types and Approved-only | ≈ 155–340 | **well powered** |
  | H4 | 118 site-years | ≈ 9–22 | underpowered unless EA nitrate is added |
  | H2 | new eligible sites need WQP pairs ≤ WY2021; count still to be computed | likely < 50 | underpowered |

- **Models:** the v3 val cycle uses levers 2–4, on the EW split's train/val roles only.
- **Guard:** the hardened single look (§D).

**EW-v3-T (secondary, temporal).** Train/val-role split sites × WY2026, Approved after 2027-06-30. Projected H1 ≈ 40 onsets, so likely underpowered. It does not overlap test-role × Test-B, which is reserved.

**EW-v3-GB (secondary, new country).** EA England M1-test groups, pre-2023 data, if the allocation proposal is accepted. Power is computed from rule-3 data rates once M6 publishes the dataset.

**Live trial (Test-C).** As in the amendment; expected underpowered in WY2027.

**M1-N (forecasting, primary for the SOTA attempt).**
- **Test:** the new-site block, unseen × WY2008–25, Approved.
- **Cells:** unseen-station headline. Variables:
  - DO: 435 sites;
  - SC: 815 sites;
  - temperature and turbidity: counts still to be computed from metadata;
  - nitrate: 44 sites, descriptive only.
- **Primary:** the same structure as `sonde_m1_protocol.md` §1 (CRPS skill vs the best of the references, basin-group bootstrap, Holm over variables, 3 seeds). Each variable needs ≥ 10 basin groups, otherwise it is descriptive.
- **Tracks inside M1-N:**
  - M1-N-X4: 21/30-d horizons with NWP;
  - M1-N-Nb: virtual sensing at unseen sites.
- **The original M1 Test-A track stays as registered** (`sonde_m1_freeze_candidate.json`). Its single look is a lead decision. Note that test-role × Test-A DO values were already scored by the EW pass, so for DO that cell is no longer blind in spirit. Recommendation: score it once as the original track and report it, but put no new-track claim on it.

**Other countries (public, to verify before use).**
- UK EA: shared `ea_wq_sonde` via M6 only.
- Environment Canada / provincial real-time networks (e.g. Newfoundland & Labrador real-time WQ).
- Australian state sondes (Queensland, NSW WaterNSW, Victoria WMIS: EC/temperature/DO/turbidity).
- NEON aquatic sondes (34 sites; coordinate with Fuse, whose Tier D test sites are D03/D07/D13/D14).

Each needs a licence check, an API check, a datasheet and an allocation note before any download. Recommended order: EA (shared), NEON, then one Australian state. Canada last.

### D. Guard for every new look (sota_campaign rule 7)
- Registry refusal if any row of the look exists.
- O_EXCL marker plus a `look_claimed` registry row holding pid and host.
- Artifacts and data pins verified before the claim.
- `BaseException` → `ABORTED.json` plus a registry row; access closed.
- Predictions sealed (`SHA256SUMS`, 0444); a sealed row; rescore refused after sealing.
- Full import-closure code hash (as `scripts/live/live_common.code_sha256`).
- Only a lead-written FREEZE unlocks the test. Write-ups go to `*_test_results.md`.

### E. Caps and ETA
**Caps (all configurations counted and logged):**
- EW-v3: ≤ 12 val/CV configurations.
- M1-N: ≤ 16 configurations per variable group, ≤ 20 GPU runs per cycle, each ≤ 2 h on GPU 0/1 via the sonde runner's own lock (never wrapped in `sentinel.core.gpu`, which deadlocks).
- CPU ≤ 6 threads.
- New data ≤ 10 GB.

**ETA (agent-days, after lead approval):**

| Work | Effort | Compute |
|---|---|---|
| New-site typing, basin groups, clean-site audit | 1 d | — |
| IV pull + tier build | 1–2 d wall clock | background, CPU |
| EW-v3 protocol + split + power | 1 d | — |
| EW-v3 val cycle (levers 2–4) | 3–4 d | CPU, ≤ 5 GPU-h |
| EW-v3 freeze candidate + rehearsal + review | 2 d | — |
| M1-N protocol + CV redesign + NWP covariates | 4–5 d | ≈ 20 GPU-h |
| Pretraining | 4–6 d | ≈ 30 GPU-h |
| M1-N freeze candidate + rehearsal | 2 d | — |
| `score_live.py` | 1–2 d | any time before 2027-10 |

- **EW-v3 single look:** ≈ 10–12 days, i.e. around 2026-11-21…25, if approved by 11-11.
- **M1-N single look:** ≈ 3 weeks, i.e. around early December 2026.
- **Live trial:** a confirmatory result cannot exist before mid-2028 (or 2029 with the extension).

**Item 3 of the task (forecasting freeze candidate on an unseen block now): not reachable this session.**
- The bar is not met on val.
- The only unseen block large enough (new sites) does not exist locally yet.
- No forecasting freeze candidate was moved forward, and no test block was touched.

### F. Decisions needed from the lead
1. Freeze the live amendment (steps above) before 12-01.
2. Approve the new-site IV pull via the legacy service before 11-16 (≈ 2–3 GB transient, ≤ 0.5 GB kept).
3. Approve the EW-v3-N / M1-N track definitions (C) before any result on them.
4. Decide on the original M1 Test-A forecasting look.
5. EA allocation (Option A three-way draw, or M1 draws from M6's non-test groups after M6's freeze).

---
## Earlier notes (2026-11-08/09, kept for provenance)

Nothing is running. No FREEZE_*.json exists; no test / test_b data or test-role sites were ever loaded.
Runs root: `/home/bcheng/STOCKHOLM/sentinel_runs/sonde/<run>/`. Driver scripts/logs: scratchpad `m1/` (`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/m1/`).
Val split tag `val/sonde_split_fd4bb14fe730`. GPU: sonde scripts lock GPU 0 THEMSELVES (`runner.setup_gpu`, M1_GPUS=(0,)) — do NOT wrap them in `python -m sentinel.core.gpu` (deadlock: on 11-08 the wrapper held GPU 0 and the inner lock waited forever). GPUs 0/1 are shared now: jobs ≤ 2 h.

## Reconciled (all pre-pause runs had finished)
LSTM-weather s0–2, Chronos-FT+weather s0, FT inner-val s0, FT LoRA s0, TimesFM-2.5 scoring, ens3 vs control3 scoring, N-b + catchment: all had outputs. Unscored members were scored on 11-08 (`cyc3_score_members_ens3w`, registered). New: FT inner-val s1 (`cyc2_chronos_ft_innerval_s1`, trained+val-predicted, NOT yet scored); s2 not run (GPU policy).

## EW-v2 (time-critical: freeze/OSF by ~11-15, live 12-01)
- Code (val only): seed-bagged event GBM (`n_bags: 5`, JSON artifacts), monotone GBM variant, logistic stack (`StackLogistic`, fitted on val like Platt; OOF by WY), `weather_vars` (weather only for H2/H4), guard accepts freeze records listing their own `scoring_code_files` (`guard.EW_SCORING_CODE`, incl. events/common yaml). Tests added (all M1 tests pass).
- Finding: single-booster val BSS changed by ~0.01 with thread count alone → bagging; weather neutral for DO hazards once bagged.
- **Freeze candidate:** `docs/preregistration/ew_v2_freeze_candidate.json` (run `ew_v2_cand_bag5_wxrunoff`; thresholds sha 3c243c1b…; scoring code sha 72e93858…; 90 artifact hashes). Selected: H1 stack, H2 stack, H4 stack. Dry run (`ew_v2_cand_dryrun_val_insample`) reproduces calibration probabilities exactly. Proposed record name `FREEZE_ew.json`; score command in the candidate file. ANY edit to a hashed file changes the code hash → re-run `scripts/eval/sonde_ew_freeze_candidate.py --run <run> --configs-tried 8`.
- Val OOF (L=3, basin-group CI): H1 stack BSS +0.022 [−0.015, 0.043]; only persistence has CI > 0 (+0.010 [0.002, 0.021]); POD real 0.71 vs placebo 0.32 at 1 alarm-day/30. H2 (18 val pos) / H4 (12) underpowered.
- §9 filled (`ew_onset_counts_trainval`, registry `ew_v2_onset_counts`): projected Test-A H1 ≈ 44 onsets (< 50) → at risk of "underpowered"; lead decision pending on a pooled Test-A ∪ Test-B secondary.
- EW val metrics now registered (`ew_v2_val_oof`, `scripts/eval/sonde_events_register.py`).

## M1 forecasting
- ens3w (FT + LSTM-weather + ZS-multi) ≈ ens3; LSTM-weather is a stronger *reference* but not a better member. Unseen val cell vs 9 single-model refs (incl. LSTM-weather, TimesFM): SC 4–5/5 h, turb-mean 4–5, nitrate 2–3, DO mean 2, DO min 0, temp 0 → bar NOT met; vs baseline-ensemble controls even fewer. Val-role cell has only 8–13 basin groups (nitrate 3): power-limited.
- Freeze candidate (not frozen): `docs/preregistration/sonde_m1_freeze_candidate.json`; protocol `docs/preregistration/sonde_m1_protocol.md` (incl. harder track M1-X, val power check done, no model scored on strata).
- N-b + catchment: bar not met (no variable with gain CI > 0).

## Next steps (val only)
1. Lead review of EW candidate → FREEZE_ew.json → OSF; then Test-A EW scoring only when authorised.
2. Score FT inner-val s1 (+ s2 when GPU allows, ≤ 2 h chunks: split train / predict).
3. M1 unseen-cell levers: DO-min-specific member weighting pre-specified per variable; per-site scale normalisation (lever d); M1-X strata scoring (`sonde_score` stratified, after freeze of definitions).
4. Hourly-tier H1 (15-min ≥ 2 h rule) as a separate power-rescue track — needs protocol amendment before any result.
5. USGS: re-pull WY2024–26 DV/IV approval status before legacy WaterServices slow-down (11-16) or via OGC API afterwards.

## 2026-11-08 evening — EW-v2 pre-test gate FAIL → fixes (re-review pending)
- Lead decisions written into `ew_v2_protocol.md` v0.2 (C1 test-role primary + Holm {H1,H2,H4} + ≥50 evaluable onsets; C2 frozen invocation; C3 single-pass marker; C4 controls; C5 live trial split off; W1–W12; pooled Test-A∪B secondary with cut-off 2027-06-30; disclosures).
- Code: `sentinel/eval/ew_controls.py` (bss p-value, Holm, cluster placebo test, random alarms, circular shift, label permutation, Molchan, reliability, REV, DM); `sonde_events.py` score phase rewritten (role cells, primary_endpoint.json, controls.csv, negative-control sites, covariate shuffle, truncation audit, frozen-invocation lock, marker, pooled pass, `--rehearsal-freeze` val rehearsal); guard: `FREEZE_FILE_EW`, hashes `ew_controls.py`, `evaluate.py`, `splits.py`. Tests `tests/test_ew_controls.py` (5) pass + M1 suites.
- Registered: 5 offline stack variants (`ew_v2_stack_explore`), first calibrate run.
- Rehearsal on val (`ew_v2_rehearsal_val`, scratch record, val-role treated unseen): see report.
- USGS re-pull (WY2024–26, to 2026-11-08): DV 649+140 batches, 0 errors → `raw/usgs_dv/batches{,_median}_refresh_20261108`; daily rebuilt to `processed/sonde_daily_refresh_20261108` (pre-2023-10-01 rows verbatim from `sonde_daily`; identity check `compare_vs_sonde_daily.json`: values identical, 2 non-split sites differ only in empty `_prov` columns). IV re-pull running (`raw/usgs_iv_refresh_20261108`, PID chain 1012502, ~4 h) → then `build_sonde_hourly.py --out-name sonde_hourly_refresh_20261108 --refresh-raw usgs_iv_refresh_20261108 --keep-before-from sonde_hourly`. Promotion of the refreshed snapshots to the loader paths needs the lead (other workstreams read `sonde_daily`).
- OSF: `scripts/eval/sonde_ew_osf_package.py` (tarball + SHA-256 manifest), `docs/preregistration/ew_v2_osf_preregistration.md`.
- Next: hourly refresh build → promote snapshots (lead) → regenerate candidate with snapshot pins → OSF package → re-review → FREEZE_ew.json (lead). Live amendment (scripts/live/, provisional-input variant) before 12-01.
- Rehearsal (`ew_v2_rehearsal_val2`, code hash c75f3230…, val-role treated unseen): frozen-invocation lock, marker, verify_thresholds, unseen-site climatology refit, role cells, Holm, all controls ran end to end; truncation audit max feature diff 0.0 for every hazard. A guard for zero-onset cells was added afterwards → final rehearsal re-runs automatically on the final code (`ew_v2_rehearsal_val_final`).
- Background chain (scratchpad m1/`final_chain.sh`, PID 1089683; replaced the two earlier chains, killed by PID): IV_DONE → hourly refresh build → hourly identity check → `scripts/data/promote_sonde_snapshots.py` (lead-approved: old dirs → `sonde_daily_20261022`/`sonde_hourly_20261024`, loader paths become symlinks to the refreshed dirs via `mv -T`, new manifests with `supersedes`, datasheet + RESULTS.md notes; refuses unless both identity checks pass) → candidate regen (daily+hourly pins) → OSF tarball → final rehearsal. Status lines in `after_iv_chain.done` (IV, HOURLY_BUILD, HOURLY_COMPARE, PROMOTE, CANDIDATE, OSF, REHEARSAL_FINAL). Daily identity check: 5,500 sites, 0 differing, identical_until true.
- Live-trial variant calibrated (val only): `ew_live_variant_cal_nosubdaily` (H1/H4, no sub-daily features): H1 stack val-OOF BSS 0.0213 [−0.010, 0.041] vs 0.0222 with sub-daily → usable on provisional inputs. Live amendment + scripts/live/ pipeline still to build.

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.

## 2026-11-09 00:55 — EW-v2 re-review ("post final chain") C1–C4 + W1–W6 fixed → delta re-review pending
Nothing running. No `FREEZE_ew.json`, no real `.ew_scored_*` marker, no test block scored, nothing uploaded to OSF. Protocol now v0.3 (dated §12 entry). Backups of pre-fix files: scratchpad `m1/*.pre_delta.bak`.
- **C1 H2 labels:** `scripts/data/build_chloride_derived.py:36` (`compare`), `:74` (`--out-name/--sonde-daily/--compare-with`; manifest now `hash_files=True`) → rebuilt `processed/chloride_derived_refresh_20261108` from the promoted daily tier; identity check `compare_vs_chloride_derived.json`: 1,172 regressions identical (52 eligible), 52 series identical ≤ 2023-09-30 (max |Δ| 0, NaN pattern identical); 45,095 pairs vs 45,099 (only post-2021 "later" pairs differ, unused). Promoted by `scripts/data/promote_chloride_derived.py` (new; symlink, old → `chloride_derived_20261025`, datasheet note; record `processed/chloride_derived_promotion_20261108.json`). Score-time guard `scripts/train/sonde_events.py:232` `check_chloride_consistency` (cl_mgl_mean == clip(a+b·SC_cube), NaN pattern identical; rehearsal: 36 sites, max |Δ| 0.0026 = float32). `chloride_derived` + `forcings_daily` in the strict data check (`:660-662`, `:688`). Candidate refuses a chloride build not made from the loader's sonde_daily (`scripts/eval/sonde_ew_freeze_candidate.py:94`). `regressions.parquet` + identity check + chloride manifest/builder in the OSF tarball (`scripts/eval/sonde_ew_osf_package.py:59`).
- **C2 evaluable onsets:** one definition `sentinel/eval/ew_scoring.py:334` `evaluable_onsets` (+ `:364` `count_evaluable_onsets`), used by `placebo_comparison` (`:414`), `ew_controls._site_arrays` (`ew_controls.py:142`) and the gate (`sonde_events.py:896`); `primary_endpoint.json` records `pod_events_evaluable` and `gate_equals_pod_set`. Rehearsal: gate = POD = random = circshift for every hazard (H1-t4 now 4/4/4, was 4 vs 3).
- **C3 power rule:** `ew_controls.py:72` `is_underpowered` (< 50 evaluable onsets OR < 2 basin groups with onsets), `:84` `primary_decision` (underpowered/no result → Holm input p = 1, `p_one_sided_descriptive` kept; m = 3); used at `sonde_events.py:941`, `MIN_ONSET_BASIN_GROUPS` `:438`. Rehearsal: H2 p 0.0055 → Holm input 1, not rejected.
- **C4 single pass:** fixed marker `.ew_scored_<pass>.json` + legacy-name regex (`sonde_events.py:488`, `:510`), parsed registry check (`:491`), `log_pass_rows` (`:542`): `pass_claimed` row right after the marker (`:779`), primary + control rows at the end (`:951`; rehearsal writes to `<run>/registry_rehearsal.jsonl`, 210 rows).
- **W1** content pins `snapshot_content_sha256` (`:675`, files the loader reads), verified strictly, missing pin = failure. **W2** pooled pass reads a dated `FREEZE_ew_test_ab_data.json` (`pooled_data_pins` `:721`; protocol §9 rule). **W3** H2 projection ≈ 2 (upper ≈ 7) in §9. **W4** basin groups next to every p, CI note. **W5** `--split` + split-file sha refused before the marker (`:476`). **W6** OSF text updated (72 run records, 9th live-variant calibrate, hourly shares, H2 projection, pairs disclosure).
- **Tests:** `tests/test_ew_controls.py` 12 (7 new: re-written freeze refused, legacy marker, registry fallback, `--split`, evaluable onsets at block start / after last issue + L / no usable rows / censored, Holm p = 1, content pin, chloride consistency); with hazards/chloride/sonde_daily/sonde_models: 72 passed.
- **New shas:** `scoring_code_sha256` c110bcffb795459d401c0e9c2db993ca261cc897033346762e6488dc6ee9261e; thresholds 3c243c1b… (unchanged; selection unchanged: H1/H2/H4 stack); candidate file 5fe357362fdf…; protocol 0d5c8f2e1b2c…; OSF tarball `sentinel_runs/sonde/osf_ew_v2/ew_v2_osf.tar.gz` 02ad146b8b309d3b0e5bc25d46125d51ada748c8fccd15ad1dac45b12de12295 (126 files). Content pins: daily 65d86584…, hourly 6d876460…, forcings d843ab0e…, chloride c80234ab… (chloride manifest `chloride_derived_refresh_20261108` 8e13c787…).
- **Rehearsal** `ew_v2_rehearsal_val_delta` (val-role as unseen, scratch record `m1/freeze_rehearsal_delta_NOT_A_FREEZE.json`, chain `m1/delta_chain.sh`): all four snapshots match; all controls ran; truncation diff 0.0 everywhere; all three primary hazards underpowered on val (H1 3 onsets / 2 groups) → no claim, as expected.
- Next: delta re-review → lead writes `FREEZE_ew.json` → OSF. Not done here: static inputs (`processed/sites/sites.parquet`, `catchment/streamcat_sites.parquet`) are not content-pinned; the cube check covers cl_mgl_mean only (SC max isn't in the cube; cl_mgl_max is covered by the content pin + rebuild); the candidate's val POD/placebo evidence still comes from the v0.2 event set (descriptive only, disclosed in §12).

## 2026-11-09 — lead: EW-v2 FROZEN
- `docs/preregistration/FREEZE_ew.json` written (read-only) from the candidate's `freeze_record_draft` + `static_inputs` (sites.parquet `0e280071…`, streamcat_sites.parquet `b6886581…`), `osf_package` (`02ad146b…`, 126 files), `score_command`. Verified: load_freeze (code hash `c110bcff…`), all four strict data pins, OSF tarball sha, no pass claimed.
- Archive `sentinel_runs/freeze_archives/ew_v2_freeze_*.tar.gz` + `.sha256`; registry row `ew_v2/prereg_snapshot`.
- **Test-A is NOT to be scored until the OSF registration exists** (protocol header). Before the pass the lead re-verifies static_inputs and runs the registered `score_command` exactly (incl. `--allow-test`).

## 2026-11-09 (lead): OSF dropped; EW-v2 Test-A pass started
- The user decided not to use OSF. The public record is GitHub commit `cef72b6` (branch `v2-rebuild`); see `docs/preregistration/ew_v2_post_freeze_addendum.md` A1. Never write "pre-registered on OSF".
- Before the pass the lead re-verified: `load_freeze` code hash ok, all 4 snapshot pins match, both static_inputs shas match, and no `.ew_scored_*` marker existed.
- The registered `score_command` was started exactly as frozen, CPU only (6 threads). Log: `sentinel_runs/sonde/logs/ew_v2_testA_score_20261109.log`.
- Do NOT re-run. Post-results review next.

## 2026-11-10/11 — architecture round v2: GBM timing benchmark, lead GBM ruling, booster-level resume
Nothing of round v2 has run on real rows: no step-1 number, no CV number, no validation number. The drivers refuse
every phase until the lead stamps the consolidated deviations file.
- **Counts (CPU, train block only).** `sonde_v2_counts.py step1` → `sentinel_runs/sonde/v2_smoke/counts_step1.json`
  (2026-11-11T00:58Z). It pins the code shas of the files it was computed with, so it must be re-run now that the
  GBM code is final; `counts cv` still to run (the GEFS extraction PID 2244958 has ended; operational 61 months,
  reforecast 147 on disk).
- **GBM timing benchmark** (`sonde_v2_gbm_timing.py`, timing only, no score, no forecast, no registry row).
  First start crashed with `KeyError: '500k'` in its projection step (log `v2_smoke/gbm_timing_loaded_crash1.log`,
  0444, sha `6ec70cf2…143f`; five CPU full-row figures 1.102/1.199/0.828/0.961/0.803 s per round). One-line fix
  (script sha before `72e40f3a…cfd01`, after `193f0439f742eecf50c984028f6a72188168b74d116f0144dcaf591eb8814f50` —
  **the script is not to be edited again**, the lead cites that sha). Second start complete: record
  `v2_smoke/gbm_timing_loaded.json` (0444, sha `21807d51…af4ec`) + `.log` (sha `ade5a57e…c000`); load 50.87 → 62.43,
  GPU index 0, LightGBM 4.7.0, PF-GBM 106 columns, `temp_c_mean` h=1. Q19 at full rows is above 2 h on both devices,
  so by dev ruling Q the record is `not_used_for_cap`.
- **Lead GBM ruling on disk** (0444, never edit): `docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md`,
  sha `3028fc1bd0509152e7428ad8a0ccbe27dece086027da9713bc9dae20bbf32c5b`, registry row `6cad67b32813`. Cite as
  "GBM ruling G-n". G-1 record `not_used_for_cap`; G-2 full fit rows for Q19/Q/LS of both arms; G-3 the 2 h limit is
  met by booster-level resume, not by a cap; G-4 no third benchmark (timing rule §4/§7 and the repeat clause of
  ruling Q are not applied); G-5 the frozen `GBMConfig` recipe, E(5) not adopted; G-6 every GBM booster on GPU
  (`device_type=gpu`, `gpu_use_dp=False`, max_bin 255, one GPU per process through the lock helper, ≤ 6 threads,
  never another user's GPU, no CPU fallback); G-7 GPU-hours per process in the registry, the CV projection before
  the first CV GBM process and a 300 GPU-h gate; G-8 the disclosures; G-9 what to implement.
- **Code (G-9, written 2026-11-11).** New `sentinel/models/sonde_v2/gbm_resume.py` (booster store with one file +
  one record per booster, keys, resume clock, GPU placement check, GPU-hour registry rows, CV projection and gate).
  `declared.py`: `GBM_SETTINGS` is now the ruling (`ruled` true, ruling file + sha, full recipe, device gpu,
  `max_fit_rows` None ×3, `resume`, `cv_gpu_hours_limit` 300); the cap ladder stays only because the timing script
  reads it and a test asserts no driver does. `gbm_nwp.py` fits through the store (no cap, plain `predict`, Q re-uses
  Q19's seven identical boosters). `sonde_v2_step1.py --phase gbm` and `sonde_v2_cv.py --phase run` for GBM arms are
  resumable GPU processes that exit 75; new `sonde_v2_cv.py --phase gbm_budget` writes the CV projection.
- **Order from here (dev ruling S + G-9):** green tests → consolidated deviations file with the code list →
  independent delta review (the lead commissions it, asked to check G-2…G-7) → lead stamp → step 1. No fit of any
  arm on real rows before the stamp. The generator of the consolidated file is scratchpad `m1/gen_dev_v2.py`
  (`--final --gbm-ruling docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md`).
- **Live trial stays ON HOLD.** GPUs: another user holds 2, 3, 4, 6, 7, 8, 9 (13:10 EST 11-11); only 0, 1, 5 free —
  check `nvidia-smi` immediately before any launch.
- **State 2026-11-11 14:00 EST (ready for the delta review and the stamp).** Tests green (54 passed,
  `tests/test_sonde_v2.py`, 6 threads). Counts re-run on the final code: `counts_step1.json` (18:44Z),
  `counts_cv.json` (18:58Z). Consolidated file written read-only:
  `docs/preregistration/sonde_arch_v2_deviations_v2.md`, sha256
  `3ee342192e3d38c982eb3c95c8ecac38025cd0f86ed587e823e2e81feb9bdb56`; `code_sha256`
  `f4de13a0d71965c07946638c2319a2db2365ef89510cf1f2a3126ff04b015526` (21 files, §6 of that file). No file of the
  code list is edited from here on. The stamp row (lead, ruling R) is what the drivers wait for.
