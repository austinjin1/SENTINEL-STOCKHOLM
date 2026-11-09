# M6 SENTINEL-HydroMet: handoff (updated 2026-11-09 02:00)

## Registered (all self-checks PASS; RESULTS.md lines of 2026-11-08/09)
- `M6/val_v2` (cycles 1–4, 1,623 entries), `M6/val_v3` (cycle 5 upstream-basin forcings, 2,103 entries),
  `M6S/val_v2` (track S strata, 132), `M6N/val_v1` (track N val-role stations, 197), `M6N/inregion_val_v1` (197).
- No original-track endpoint won ("competitive"). Freeze candidate (protocol §8): `ealstm_ws` (mean min(Δgbm, Δphys)
  −0.004); blend `ealstm_gbmblend` −0.0003 reported separately. Re-evaluate after cycle 6 (`ealstm_ws_full`, val_v4).
- Protocol DRAFT v0.11: §9 rows v0.10 (upstream forcings + parity comparators), v0.11 (ws refit), §10 track R
  (folds sha256 142341a9…, amended for the GPU cut: `ealstm_full` refits optional), §11 track S, §12 track N
  (roles sha256 ffb41471…).

## Data built (manifests + datasheets)
- `processed/hydromet_upstream_forcing` (sonde sites; local NHDPlusV2 delineation + 74 NLDI polygons). `_trackn`
  files for track-N stations are being built by `q_cpu4.sh` (cache-only NLDI).
- `processed/hydromet_wqp_national` (track N; 12,771 unsensed WQP stream stations).
- Interim: `interim/hydromet_wqp_national/catchments` (21 VPU simplified catchments, ~1.5 GB, shared),
  `interim/hydromet_upstream_forcing/catchment_cells.npz`.

## GPU rule (2026-11-08 16:00): SENTINEL may use GPUs 0/1 only, shared, ≤ 2 h/job, critical jobs first.
## Running (scratchpad `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/`)
- `q_gpu.sh` (log `q_gpu2.log`): `ealstm_ws_full_v1_s{0,1,2}` → 15 track-R early-stopped seeds → track-R
  `ealstm_full` refits. One job at a time via the lock helper, starts only when a 0/1 lock is free, 10-min gaps.
- `q_cpu4.sh` (log `q_cpu4.log`): track-N station upstream forcings → track-N predictions of the `_ws` models →
  track-R baselines per fold (`trackr_<fold>_base`, `trackr_<fold>_gbmbag_s<k>`).
- Finished runs carry `predictions/.done`; queues skip them on restart.
- Helpers: `m6_table.py <score_dir>`, `m6_selfcheck.py <score_dir> <experiment>`.

## Pre-test review fixes (REVIEW_REPORT_PRE_TRAINING.md, "M6 HydroMet — pre-test review", 2026-11-09) — status 05:40
- **C1 single test look — DONE.** `sentinel/models/hydromet/testlook.py`: `claim` :121 (registry refusal + O_EXCL
  0444 marker per look + claim-time registry row), `require_open_claim` :151 (wired into `guard.check_access`,
  `guard.py:111-113`: protected data only with an open claim of this freeze), `seal_predictions` :160 (0444 + SHA256SUMS,
  written once), `check_sealed` :179 (crash re-score only), `mark_scored` :196. Driver `scripts/eval/hydromet_test_look.py`
  (order: freeze/code hash → `verify_data_pins` :164 → `claim` :171 → predict → seal → score → SCORED).
  `hydromet_score.py:36-40`, `hydromet_predict.py:33-37`, `hydromet_track_n.py:339-343` refuse protected data.
  Tests `tests/test_hydromet_testlook.py` (second look, registry rows, rewritten freeze, sealed/modified predictions,
  scored marker, CLI refusals, guard needs claim); `tests/test_hydromet.py` updated (claim sandbox). All pass.
- **C2 family/label locked — DONE.** Models, Holm size, comparators only from `freeze["looks"][look]`
  (`testlook.look_block` :65-75 checks n_hypotheses = models × endpoints); `pipeline.score` :219-223 refuses protected
  scoring without the frozen family; the look checks the family size (`hydromet_test_look.py` score()). Label model =
  `ealstm_ws_full`, confirmatory family = 28 (all 7 models' artifacts frozen): protocol §8 :234-239.
- **C3 tracks R/N runnable — DONE (code); data/rehearsal IN PROGRESS.** Track R: fold-unique keys
  `trackr_<fold>/<name>` (`pipeline.py:70`, `:121`), fold files accepted on the test look (`pipeline.py:127`,
  `runner.py:141`), separate `FREEZE_hydromet_trackR.json` block builder `hydromet_freeze_candidate.py trackr` :189;
  protocol §10 :338-352. Track N: comparators/models from the freeze block, untestable → p = 1
  (`hydromet_track_n.py:265`, family size check :272), data pins `hydromet_wqp_national` + `*_trackn` in the record.
  Rehearsals (`hydromet_test_look.py --rehearse-on-val`, :123): **M6/test PASS** (28 hypotheses; Δ and p identical to
  `M6/val_v4`, max |Δ| 2e-7); M6N/test waits for the track-N station basin forcings (CPU queue), then `M6N/val_v2`;
  M6R/test rehearsal waits for the R1 fold baselines.
- **W1 — DONE (code); regenerate after the track-N data.** `hydromet_freeze_candidate.py verify` checks code + protocol
  hashes and look blocks (:175-185); data pins are in the freeze schema and checked before every claim
  (`testlook.verify_data_pins` :87).
- **W2 — DONE.** Test-cell power and "competitive is largely a power outcome": protocol §8 :240-245.
- **W3 — DONE.** Published methods (Rahmani et al. 2021, Zhi et al. 2021, Jia et al. 2021) and "not SOTA" wording: §8 :246-252.
- **W4 — DONE.** Track-R contradictions fixed; fixed comparators justified; R1 log disclosure; secondary/descriptive,
  same test cells, no added test power: §10 :338-352.
- **W5 — DONE** (§12 cache-only NLDI disclosure; "six models" removed). **W6 — DONE** (RESULTS.md M6 cycle-6 line lists
  all variants and val looks). **O3 — DONE** (test_b own later freeze, §8 :261).
- Remaining before re-review: track-N station forcings → M6N rehearsal + `M6N/val_v2`; R1 baselines → M6R rehearsal
  (partial); final `hydromet_freeze_candidate.py write` + `verify`.

## Next steps
1. After `ealstm_ws_full_v1_s2`: score `M6/val_v4` (val_v3 command + `$H/ealstm_ws_full_v1_s{0,1,2}/predictions`),
   self-check, RESULTS line, render; apply the §8 rule; update the candidate.
2. Track N with `_ws` models: `hydromet_track_n.py score --predictions $H/trackn_val_v1/{full,compact,ws,wsphys}_inputs
   --comparators $H/score_val_v3/comparators.json --register --experiment M6N/val_v2` (and in-region).
3. Track R when all folds finish: `hydromet_score.py --track-r cv --primary --register --experiment M6R/cv_v1` over
   all `trackr_*` prediction dirs (drop `trackr_*_full_*` unless all 18 exist, protocol §10 amendment); then strata
   on track R (`hydromet_strata.py --track-r cv`).
4. Report the freeze candidate to the lead; do not create FREEZE or score test.
