# M6 SENTINEL-HydroMet: handoff (updated 2026-11-09 12:40)

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
- `q_gpu.sh` (PID 1176254, log `q_gpu2.log`): `ealstm_ws_full_v1_s{0,1,2}` (done) → track-R early-stopped seeds
  (R1, R2 done; R3 running) → track-R `ealstm_full` refits. One job at a time via the lock helper, starts only when a 0/1 lock is free, 10-min gaps.
- `q_cpu4.sh` (log `q_cpu4.log`): track-N station upstream forcings (done) → track-N `_ws` predictions (done) →
  track-R baselines per fold (`trackr_<fold>_base`, `trackr_<fold>_gbmbag_s<k>`; R1 base + gbmbag s0 done).
- Finished runs carry `predictions/.done`; queues skip them on restart.
- Helpers: `m6_table.py <score_dir>`, `m6_selfcheck.py <score_dir> <experiment>`.

## Pre-test review fixes (REVIEW_REPORT_PRE_TRAINING.md, "M6 HydroMet — pre-test review", 2026-11-09) — status 12:30 — ALL DONE, ready for delta re-review
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
- **C3 tracks R/N runnable — DONE.** Track R: fold-unique keys
  `trackr_<fold>/<name>` (`pipeline.py:70`, `:121`), fold files accepted on the test look (`pipeline.py:127`,
  `runner.py:141`), separate `FREEZE_hydromet_trackR.json` block builder `hydromet_freeze_candidate.py trackr` :189;
  protocol §10 :338-352. Track N: comparators/models from the freeze block, untestable → p = 1
  (`hydromet_track_n.py:265`, family size check :272), data pins `hydromet_wqp_national` + `*_trackn` in the record.
  Rehearsals on the final code (`hydromet_test_look.py --rehearse-on-val`): **M6/test PASS** (28 hypotheses; Δ and p
  equal `M6/val_v4`, max |Δ| 2e-7), **M6N/test PASS** (4 hypotheses; equal to `M6N/val_v2`, which was registered from the
  same predictions), **M6R/test PASS (partial)** (11 fold artifacts from R1–R3 merged; `hydromet_trackR_rehearsal.json`,
  rehearsal only). The track-R rehearsal exposed a degenerate one-group bootstrap (p = 0): fixed by
  `evaluate.MIN_TEST_GROUPS = 3` (`evaluate.py:261`; track N `hydromet_track_n.py` same rule) and protocol §7.
- **W1 — DONE.** Candidate regenerated 12:0x (scoring_code_sha256 `f519bb35…`, protocol `3a13670f…`, 25 artifacts,
  looks M6/test 28 / M6N/test 4, 7 data pins incl. track N); `verify` PASS. `hydromet_freeze_candidate.py verify` checks code + protocol
  hashes and look blocks (:175-185); data pins are in the freeze schema and checked before every claim
  (`testlook.verify_data_pins` :87).
- **W2 — DONE.** Test-cell power and "competitive is largely a power outcome": protocol §8 :240-245.
- **W3 — DONE.** Published methods (Rahmani et al. 2021, Zhi et al. 2021, Jia et al. 2021) and "not SOTA" wording: §8 :246-252.
- **W4 — DONE.** Track-R contradictions fixed; fixed comparators justified; R1 log disclosure; secondary/descriptive,
  same test cells, no added test power: §10 :338-352.
- **W5 — DONE** (§12 cache-only NLDI disclosure; "six models" removed). **W6 — DONE** (RESULTS.md M6 cycle-6 line lists
  all variants and val looks). **O3 — DONE** (test_b own later freeze, §8 :261).
- Track R's own freeze (`FREEZE_hydromet_trackR.json`) is built later by `hydromet_freeze_candidate.py trackr` once all
  folds finish; it must carry the same scoring_code_sha256 as the primary freeze (do not edit M6 code after the freeze).

## Delta re-review fixes (D1–D5, 2026-11-09 14:20) — DONE, sent for line-level check
- Diff vs the pre-edit snapshot: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/pre_delta2/delta2.diff`.
- D1 `guard.py:27-31, 66-67, 76-80` (only FREEZE_hydromet{,_trackR,_test_b}.json in docs/preregistration, status FROZEN +
  authorised_by; NOT_A_FREEZE / partial_rehearsal_only refused); `hydromet_test_look.py:176-177` (one freeze name per look).
- D2 `testlook.py:152-212` (claim scope = look block/roles/dataset; `abort`, `resume_aborted` ≤ 2, logged);
  `guard.py:104, 124` (dataset passed); `hydromet_track_n.py:117` (dataset="trackn"); driver abort/resume `:188-203`.
- D3 `testlook.py:248-255` (files not in SHA256SUMS refused).
- D4 `hydromet_freeze_candidate.py:199-255` (TRACK_R_MODELS fixed = ealstm, ealstm_gbmblend; pins copied from the primary
  freeze; per-file frozen-hash check); protocol §10 :331-333, :349-353. GPU queue restarted without refits (`q_gpu3.sh`).
- D5 protocol :182-183, :230, :435-438; sensitivity re-score `hydromet_test_look.py:98-109` (`M6N/test_sens427`).
- Candidate regenerated: scoring_code_sha256 `5bd7d329…`, protocol `55ab7182…`, 46 frozen files, 25 artifacts; verify PASS;
  rehearsals M6/test (= val_v4) and M6N/test (= M6N/val_v2, + sensitivity) PASS; 77 tests pass.
- events.py v1.0 released (sha 4279d242…): implement `sentinel/events/hydromet.py`, `scripts/eval/events_hydromet.py` once
  the track-R CV cell is complete (use `coldwater_eligible`, `onset_metrics`).

## Event track (B output; user requirement 2026-11-09)
- Pre-registered in `docs/preregistration/events_hydromet_protocol.md` (NOT in the hashed `hydromet_protocol.md`):
  E1a, E1b, E7-coldwater on track R; derived route primary (Gaussian-head z → Platt on the track-R CV cell), direct
  LightGBM head as challenger, comparators threshold-on-GBM / physics + regional DOY climatology; `M6E/test` on
  Test-A = E1b + E7-cold only (E1a labels already scored by EW-v2 → Test-B later). Code goes ONLY in unhashed paths:
  `sentinel/events/hydromet.py`, `scripts/eval/events_hydromet.py`, `configs/events/hydromet.yaml`, using the shared
  `sentinel/eval/events.py` (being built by the lead's team). Approved by the lead 2026-11-09 (E1a stays OFF Test-A).
  Not implemented yet: wait for the lead's events.py v1 announcement AND the track-R CV cell; never touch hashed M6 files.

## Next steps
1. Done: `M6/val_v4`, candidate `ealstm_ws_full`, pre-test fixes, regenerated candidate (verify PASS). Waiting for the
   lead's delta re-review → lead writes FREEZE_hydromet.json → `hydromet_test_look.py --look M6/test`, then `M6N/test`.
2. Done: `M6N/val_v2` (registered from the M6N rehearsal predictions). Optional: in-region look of the `_ws` models
   from `$H/trackn_val_v1/ws_inputs` (descriptive).
3. Track R when all folds finish: `hydromet_score.py --track-r cv --primary --comparators docs/preregistration/hydromet_track_r_comparators.json --models ealstm,ealstm_gbmblend[,ealstm_full] --register --experiment M6R/cv_v1` over
   all `trackr_*` prediction dirs (drop `trackr_*_full_*` unless all 18 exist, protocol §10 amendment); then strata
   on track R (`hydromet_strata.py --track-r cv`).
4. Then `hydromet_freeze_candidate.py trackr` → lead writes FREEZE_hydromet_trackR.json (same code hash) → `--look M6R/test` (secondary).
