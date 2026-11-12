# M6 SENTINEL-HydroMet: handoff (updated 2026-11-09 18:00)

## Ownership (lead, 2026-11-10)
- **M6 agent (this handoff's original author):** track R (queues `q_gpu4.sh` / `q_cpu4.sh`, `M6R/cv_v1` score, track-R
  freeze candidate via `hydromet_freeze_candidate.py trackr`) and the event track (`sentinel/events/hydromet.py`,
  `scripts/eval/events_hydromet.py`, `configs/events/hydromet.yaml`, `events_hydromet_protocol.md`). Does NOT implement levers.
- **M6 SOTA-lever engineer:** levers 1–4 and the EA England build; owns new files in `sentinel/hydromet_sota/`,
  `scripts/{train,eval}/sota_hydromet_*.py`, `configs/hydromet_sota/`, `processed/ea_wq_sonde`, and its own protocol files.
  `sentinel/hydromet_sota/reserved.py` (+ its test) was created by the M6 agent and is handed over to the lever engineer.
- **Shared, nobody edits:** the 46 frozen files (code hash `5bd7d329…`), `FREEZE_hydromet.json`, sealed `test_looks/`.
- GPU use by both: lock helper, `SENTINEL_GPU_MIN_FREE_MB=10000`, ≤ 2 h per job; CPU: one heavy job per agent, ≤ 6 threads;
  disk: check `df` before any build, stop below 60 GB free.

## SOTA levers (lever engineer; updated 2026-11-11 12:59 EST)
- **Pre-registration:** `docs/preregistration/hydromet_sota_protocol.md` v1.0 + §10 rows A1 (lead rulings: lever budget
  ≤ 8 EA-LSTM configs = C1a, C1b, C2a, C2b, C12 gated, lever 4 ≤ 3 gated; lever 3 retired; A* replaced by the arch-v2 budget),
  A2 (data-hygiene rules 1–5; rule 3 drops 2 USGS sites), A3 (lever 4 = one configuration C4a, defined before any lever
  result was scored — the C2a seed-0 epoch lines 0–10 were visible then, see the §10 visibility record; gated), the lead's confirmation of the decision metric, and the time-correction row. Decision metric = primary inner
  MSE (4 endpoints, 256 reference inner sites); the 7-target inner MSE is reported next to it and decides nothing; "moves" =
  3-seed mean ≤ reference 0.11786 − 0.006.
- **Status of lever models (Amendment A4, 14:57 EST; lead: final for this round):** SECONDARY in the joint US look — no lever
  model is a claim candidate and none can become one later; every lever configuration with 3 seeds at the joint freeze is
  co-scored there (same code, same rows) with its GBM bag and physics members; §8 Test-B/N2 of the lever protocol is superseded.
  Protocol sha256 9848790a…3bd1.
- **Do not share** lever inner numbers, logs or run paths with the arch-v2 engineer; counts and row ids go through the lead.
- **Snapshot rule (lead):** after every append to the protocol, pins, `hydromet_ea_protocol.md` or `configs/hydromet_sota/C*.yaml`
  a registry row `M6-hydromet` / `hydromet_sota/prereg_snapshot` (extra.sha256) is logged BEFORE the next lever score. Rows:
  `3deeb3b6bd27` (lead) → `09d19d653747` → `b5cb10e14e80` → `71bdf660366f` (A4) → **`b60e89141d7d`** (current, 15:10 EST; EA
  protocol v1.0). `sota_hydromet_score.py`
  (`check_snapshot`) refuses to score when the latest row differs from the files on disk and stores the row id with each result.
  Every written time comes from `date` run immediately before writing.
- **Scoring hold (lead ruling 15:00 EST):** `sentinel_runs/hydromet/sota/HOLD_VAL_SCORING` (0444) stays until the lead lifts it
  in writing — not before the arch-v2 validation go/no-go (G-D1a/b); lever and arch-v2 validation cells are then scored by the
  same code on the same rows in one step. While it exists `sota_hydromet_score.py val` and `sota_hydromet_trackn.py
  predict|score` refuse, and **nobody opens the val-cell prediction files that the training scripts write at the end of a fit
  or joins them to labels** (lead, 15:12 EST: the write step stays, the files are not deleted). Inner-split rows continue
  (`inner --register`).
- **Code (unhashed; signatures kept stable for the arch-v2 engineer):** `sentinel/hydromet_sota/{sites,data,lstm,gbm,physics,runner,ea,ea_geo,ea_data,ea_split}.py`;
  `scripts/data/sota_hydromet_{build_lever_data,ea_build,ea_forcing}.py`; `scripts/train/sota_hydromet_{lstm,gbm,physics,make_c12}.py`;
  `scripts/eval/sota_hydromet_{score,trackn}.py`; `configs/hydromet_sota/C*.yaml`; tests `tests/test_hydromet_sota_{lstm,parity,ea}.py`
  (trainer bit-identical to the frozen EA-LSTM with options off; exact resume) — all pass, as does `tests/test_reserved_loaders.py`.
- **Lever data** (`processed/hydromet_sota`, manifest `hydromet_sota`, pins `docs/preregistration/hydromet_sota_lever_pins.json`):
  445 new USGS fit sites (41 temp/DO, 404 discharge-only) in 32 fit groups; 3,545 WQP train-region stations (256,628 temp /
  230,992 DO station-days, train block only). Excluded: 416 M1-reserved, 2 on reserved COMIDs, test/val groups, 20 inner groups,
  new regions, flow-connected. Upstream forcings of the new sites: `processed/hydromet_upstream_forcing/upstream_forcing_sota.parquet`
  — complete and final (0444; sha256 925a21ec…984f; 3,445,923 rows = 447 site keys × 7,709 days; companions `coverage_sota.parquet`
  bd9df7b0…1e5e, `basins_sota.parquet` b93446b9…1f2e; accepted by the lead 15:00 EST). Consumers select by fit-site key and
  must hard-fail on the two `reserved_comid` keys; values after 2026-09-09 are provisional.
- **Runs** (`$SENTINEL_RUNS/hydromet/sota/<run>`; lanes `scratchpad/sota/sota_lane.sh`, PID files + logs in `sentinel_runs/hydromet/sota/logs/`):
  - `sota_C2a_s0` DONE, scored 13:27:20 EST after snapshot `09d19d653747`: registry `cc4ae2da6f60` primary inner MSE **0.117455**
    (reference seeds 0.11836 / 0.12396 / 0.11127, mean 0.11786, SD 0.0064 → no visible gain), `9772f206ad93` 7-target inner MSE
    0.216031 (reference seeds 0.2041 / 0.2243 / 0.2079). No tuning beyond the pre-declared list (lead).
  - `sota_C1b_s0` DONE 14:58:42 EST (best epoch 13 of 20; GPU 3 freed): registry `f375533e2d72` primary inner MSE **0.122183**,
    `d951e2dac901` 7-target inner MSE 0.207734 → no gain on the decision metric.
  - `sota_C1a_s0` DONE 15:15:35 EST (best epoch 19 of 26): registry `b85b9a401f4a` primary inner MSE **0.115168**,
    `685ac145fb61` 7-target inner MSE 0.197730 (snapshot `b60e89141d7d`) → lever-1 screening winner (C1b stops); 0.0027 below
    the reference mean, inside one reference SD; "moves" is decided on the 3-seed mean only.
  - **Full inner table (18:25 EST; registered 18:24 EST under snapshot `b60e89141d7d`; all 8 fits rc 0):** C1a seeds 0.115168 /
    0.124459 / 0.111152 (`b85b9a401f4a` / `a5975cb744af` / `2019d43c7e85`), mean **0.116926**, Δ −0.0009 → not moved; C2a seeds
    0.117455 / 0.123083 / 0.114725 (`cc4ae2da6f60` / `936d5b12bf08` / `03d0c884acd3`), mean **0.118421**, Δ +0.0006 → not
    moved; C2b s0 0.118925 (`1211205f1812`; lever-2 winner is C2a). 7-target rows: `685ac145fb61`, `41e9930ea9a5`,
    `60bdc73063af`, `9772f206ad93`, `3c31693beac5`, `ed46b322170d`, `d951e2dac901`, `26435719c9a8`.
    **Decision by §6: no lever candidate; C12 and C4a are NOT run; no further EA-LSTM fit (4 of 8 configurations used).**
    Reported to the lead 18:25 EST with the open question whether the parity queue below keeps running.
  - **Lead ruling (message labelled 18:45 EST; received before 18:33 EST by `date`):** result accepted (no lever candidate; C12 / C4a not run; no further EA-LSTM fit). The parity
    queue stays (A4 co-scoring of C1a / C2a as secondary arms) and may run up to 3 lever CPU jobs at once (≤ 6 threads each,
    a job starts only at 1-minute load ≤ 40; GPU 2 is never used).
  - **Parity fits COMPLETE (2026-11-11 12:59 EST; every job rc 0, no traceback; nothing is left running).** Lever 1: CPU lane
    (orchestrator PID 2281976 → lane 2312961) finished `phys:C1a` at 2026-11-10 22:29:25 EST, "CPU LANE DONE", and the
    orchestrator's own final `score inner` returned rc 0 at 22:29:36 ("DONE (C1a)"). Lever 2: the three
    `scratchpad/sota/sota_par.sh` chains (one per seed directory, started 18:35 EST after the v2 lever-2 orchestrator PID
    2281975 and its lane PID 2333315 were stopped by PID with no job started) reached CHAIN DONE at 23:59:49 (`C2a_s1`),
    2026-11-11 00:43:02 (`C2a_s0`, incl. `physgpu:C2a`) and 01:31:13 EST (`C2a_s2`). Logs `orch_L1_cpu.log`,
    `par_C2a_s{0,1,2}.log`, job output `cpu_<kind>_<cfg>_<seed>_<target>.out`.
  - **Members on disk for the A4 joint look** (val-cell prediction files exist but are NOT opened while the hold stands):
    `sota_C1a_s{0,1,2}` and `sota_C2a_s{0,1,2}` EA-LSTM; `sota_gbm_C1a_s{0,1,2}` and `sota_gbm_C2a_s{0,1,2}` with
    4 bag models + 4 val prediction files each (C2a also `qhat.npy` / `qstack.pkl`, inner log-Q RMSE 1.422 / 1.433 / 1.430);
    `sota_phys_C1a` (2 members) and `sota_phys_C2a` (4 members, incl. `air2stream_q_regional_ws`; 1,216 donors of 1,227
    candidates, calibration NSE median 0.963), both with `predictions/.done`. Parity fit metrics are in TRAINING_LOG.md
    (29 `SOTA/sota_{gbm,phys}_*` entries) through `runlog`; no parity registry row exists — those are written by the
    scoring step, which is on hold. C1b and C2b have one seed each and are therefore not co-scored under A4.
  - **Nothing further is scheduled.** The levers need no more compute; the joint look needs only the lead's go, the hold
    lifted, and one scoring step over the members above.
  - Orchestrators **v2** `scratchpad/sota/orchestrate_lever_v2.sh` (PIDs 2281976 lever 1, 2281975 lever 2; logs `orch_L1.log`,
    `orch_L2.log`), started 14:59:50 EST in place of v1 (PIDs 2198069 / 2198070 killed): INNER SPLIT ONLY. They check the hold
    file themselves and stop (rc 97) if it is missing or writable; rc 96 if `score inner` refuses (e.g. the snapshot row does
    not match the files on disk — log the snapshot row right after any protocol/pins/config edit, then restart the
    orchestrator); rc 95 without a valid winner. Flow: screening winner (`sota_hydromet_score.py inner`; no winner before both
    seed-0 runs of a lever exist) → seeds 1–2 (two GPU lanes) → GBM parity (`--stage qstack` first for lever 2) + physics
    parity on ONE sequential CPU lane (`sota_cpu.sh`, flock `logs/cpu_lane.lock`) → final inner table. No val or track-N step.
    Results are registered by hand with `sota_hydromet_score.py inner --register` (idempotent). C12 and lever 4 are gated,
    not automated.

## EA build: what is ready and where (lever engineer; for the arch-v2 engineer — read-only consumers; updated 2026-11-10 15:12 EST)
- **Reservation:** nothing dated ≥ 2023-10-01 is requested or stored (`sentinel/hydromet_sota/ea.py::RESERVED_FROM`; every parsed
  row checked). Test-role groups become fully reserved when `docs/preregistration/hydromet_ea_split.json` is frozen (not yet).
- **On hold by the lead (rulings §2.4):** T-E is stamped by reference only. **No England reading is QC-summarised, aggregated,
  fitted or scored by anyone** until the lead stamps the T-E entry and runs the split writer once himself. Order afterwards:
  M6 split freeze → M1's draw (`docs/preregistration/sonde_ea_split.json`, exists even if empty) → first M6 England fit.
- **Submitted for the stamp (15:11 EST):** `docs/preregistration/hydromet_ea_protocol.md` **v1.0**, sha256 23e88d79…39eb
  (snapshot row `b60e89141d7d`). Split rule (S = 0.50, ruled) in ONE module, `sentinel/hydromet_sota/ea_split.py`: canonical
  core (`CORE_KEYS`, incl. salt / share_test / share_val / pins), `core_sha256`, `verify` / `load` (re-draws the roles; never
  loosened), status for all 1,964 stations (eligible → role; ineligible → role of the flow-connected eligible station(s),
  else of its L7 unit's group, else "reserved"; tie-break test > val > train, 0 stations affected — lead to confirm), L7 units
  per group, `min_groups` (G ≥ 15 and G_eff ≥ 8), M1 file contract (`verify_m1`, `seal_m1`, `load_pair`). Writer:
  `sota_hydromet_ea_build.py split --i-have-the-lead-ruling` (LEAD ONLY, once; accepts only S = 0.50; writes 0444);
  `… preview` prints the same body and writes nothing (expected core sha256 1395957e…0246).
  Metadata result: 1,100 eligible, 55 groups (77 L7 units), 28 test / 10 val / 17 train groups, 583 / 136 / 381 eligible
  stations; statuses test 1,088 / val 232 / train 640 / reserved 4 (1,092 closed to M1 and others for all years). Test cell by
  metadata: 141 temperature / 123 DO stations in 21 groups, G_eff 11.15 / 12.07 → all 6 endpoints pass the minimum; the
  zero-shot go/no-go cell (train+val-role × EA-val) has G 20 / 18 but G_eff 6.53 / 6.18 (below 8) and the fine-tuned val cell
  G 8 / 7, G_eff ≈ 3.1 — reported to the lead, ruling pending. Open with the lead: tie-break, M1 file contract, the §3
  reach-distance rule (still PROPOSED), how gate G-D1b reads a cell below the minimum.
  The `daily` QC step refuses until the split file exists.
- **Ready:**
  - catalogue `processed/ea_wq_sonde/{stations,measures}.parquet` (1,964 stations with temperature, 1,912 with DO; 5,819 measures;
    raw JSON `raw/ea_hydrology/catalogue/`);
  - raw monthly cross-station readings `raw/ea_hydrology/readings/{temperature,dissolved-oxygen}/YYYY-MM.csv.gz`, 2012-01 … 2023-09,
    141 files per property (download finished 14:09 EST; 230 MB; not parsed for any statistic);
  - `stations_geo.parquet` (HydroRIVERS v1.0 snap, HydroBASINS L6/L7, flow pairs, draft groups) and `stations_strata.parquet`
    (near-discharge flag from the UWWTR Art. 15 discharge points, CC BY, `raw/ea_uwwtd/`; 734 / 1,964);
  - RiverATLAS subsets (built 14:30 EST by the `atlas` step; the geodatabase is extracted to interim for the run and removed):
    `processed/ea_wq_sonde/riveratlas_gb.parquet` (18,143 GB reaches × 296 columns) and
    `processed/hydromet_sota/riveratlas_us_sites.parquet` (10,168 US train/val-role locations: FINAL train/val sites, lever fit
    pool, track-N train/val stations; 10,066 with a reach within 5 km, median 154 m; column `snap_m`).
- **Loaders for consumers:** `sentinel/hydromet_sota/ea_data.py` — `COMMON_STATIC`; `load_us_common(roles, blocks, *, log, max_snap_m=1000.0)`
  is usable NOW (US only; the default keeps sites with a RiverATLAS reach within 1 km: 1,625 of 1,712 train-role and 279 of 300
  val-role sites — proposed rule in `hydromet_ea_protocol.md` §3, the lead has not ruled; `max_snap_m=None` keeps all);
  `load_ea` (refuses EA-test, test-role and reserved stations; requires BOTH `hydromet_ea_split.json` and `sonde_ea_split.json`
  to exist and verify; drops M1-test groups for all years) also needs `station_daily.parquet` and the forcings.
  `load_split()` / `load_splits()` open the split only through `ea_split`.
- **After the stamp and the lead's split-writer run:** `daily` → `processed/ea_wq_sonde/daily_stats.parquet` (per measure-day: n, mean/min/max, coverage, valid flag),
  `station_daily.parquet` (valid days, columns `temp_c_mean/max/min`, `do_mgl_mean/min/max`, `do_pct_mean`), `qc_measures.parquet`,
  `CONTENT_PINS.json`, manifest `ea_wq_sonde` (OGL v3). QC: quality Suspect/Missing dropped (EA WQ rows are almost all "Unchecked"),
  range, spike (5-reading median), flat line ≥ 12 h, day valid at ≥ 80 % expected readings and ≥ 20 h span. The realised
  G / G_eff check is made inside the look by `ea_split.min_groups`.
- **ERA5-Land pull** (`scripts/data/sota_hydromet_ea_forcing.py pull`, PID 2216139, PID file `sentinel_runs/hydromet/sota/logs/ea_forcing_pull.pid`,
  log `raw/openmeteo_ea/pull.log`): 450 cells / 2,416 requests / 43,011 weighted calls (16,342 for the EA-test window, first);
  own cap 3,200 per 24 h (40 % of the 8,000 ledger cap): 3,199 used by 13:10 EST, the process sleeps until the window reopens
  (≈ 13–14 days in total); then `combine` → `processed/ea_wq_sonde/forcings_era5land.parquet`.

## TEST RESULTS (single looks run by the lead, 2026-11-09) — written up
- `M6/test` (Holm 28, 0 won) and `M6N/test` (Holm 4, 0 won) + `M6N/test_sens427`; sealed outputs in
  `$SENTINEL_RUNS/hydromet/test_looks/{M6_test,M6N_test}/`. Write-up `docs/preregistration/hydromet_test_results.md`;
  RESULTS.md entry 2026-11-09; model card v1.0.0. **Label for `ealstm_ws_full`: competitive** (level with GBM, CI > 0
  vs physics on all 4 endpoints); track N: §15.1 "negative" on temperature for `ealstm_ws_full` (+0.161 °C RMSE, every seed worse; the registry
  `holm_reject: true` is the wrong direction — never read it without the sign), level on DO; blend DO −0.035 mg/L vs
  `forcing_gbm` (not Holm-significant, every-seed margin fails in sens427, not lower than `forcing_gbm_bag`).
  Post-results review: PASS WITH CONCERNS; R1–R4 and W1–W8 applied 2026-11-09 (RESULTS, test results, card, handoff).
- Both looks are spent: no re-scoring; `M6/test_b` needs its own later freeze with a later sonde pin.
- Do NOT edit the 46 frozen files (track R must share the code hash `5bd7d329…`).

## SOTA plan (docs/plans/sota_campaign.md; written 2026-11-09 before any new run — awaiting lead review)
Track R and the event track stay on their current path; nothing below changes them or the 46 frozen files.

### 1. Diagnosis — why the GBM ties (or beats) the EA-LSTM
- **No information gap.** Both see the same inputs: gridMET site-cell (+ basin-mean) forcings, the same 188 StreamCat
  statics, calendar. The GBM's hand-built causal features (`features.py`: trailing means/sums over 1–365 d, API, a
  degree-day snow bucket, the 30-d anomaly, air DO saturation) already encode the slow memory a sequence model would
  learn (air→water lag of days–weeks, snowmelt, seasonal storage). The EA-LSTM's only structural edge, learned
  temporal filters, is largely pre-computed for the GBM.
- **The binding constraint is the number of independent sites, not model capacity.** 1,456 fit sites in ~100 basin
  groups; the EA-LSTM memorises sites (train MSE 0.06 vs inner 0.21; static-input noise and fewer statics changed
  inner loss, protocol v0.4/v0.9). The upstream-forcing lever helped the LSTM on val (+0.012–0.019 KGE on temperature) but did **not** replicate on test (`ealstm_ws` − `ealstm` −0.001 / +0.001 / −0.003 / −0.015, all CIs include 0). Treat val-cycle gains of this size as selection noise when ranking levers.
- **Temperature is near an input-limited ceiling.** Median KGE ≈ 0.90 for every strong model on val and test; the
  remaining error is site-specific (groundwater, shading, dams/reservoir releases, snowmelt timing) that the inputs
  do not observe. More capacity cannot recover it; new information (more sites, discharge/hydrologic state, network
  position of dams/upstream heat) can.
- **Unsensed reaches (track N):** the LSTM transfers worse than the GBM to off-network grab-sample stations
  (temperature +0.16 °C RMSE on test). Likely causes: distribution shift in statics (small streams, `elev_m` =
  `elevcat`), and LSTM site memorisation through statics; GBM trees extrapolate flat.
- **DO / low-DO events:** DO is temperature-saturation plus unobserved metabolism/organic load; on val no evaluated model (cycle 1–4 EA-LSTMs and the GBMs; `ealstm_ws_full` was never evaluated) detects DO-min < 5 mg/L (CSI ≤ 0.033, `M6S/val_v2`). No current input carries metabolism or low-flow state.

### 2. Levers, ranked by expected gain per GPU-hour (parity: every new input/data also goes to the GBM and physics)
1. **More labelled sites (data first; CPU-heavy, GPU-light).** (a) All USGS DV temperature/DO stream sites with
   ≥ 365 days (adds sites below the split's 730-day rule; only train-role basin groups or new groups assigned by the
   frozen split's rule, never test-role groups); (b) national WQP grab samples as sparse weak labels from train-region
   stations, **train block (WY2008–21) only** (scale: ≈ 4,600 / 4,200 train-region stations had temp / DO samples in
   WY2022–23 alone; that block stays validation); masked loss for the LSTM, row weights for the GBM. Expected: largest gain for the LSTM, because site memorisation is its failure mode; also tests
   whether the track-N gap is a coverage problem. ≈ 6–10 GPU-h.
2. **Discharge as an auxiliary TARGET (never an input).** USGS daily discharge exists in `sonde_daily` for 3,297 sites
   (`discharge_cfs_mean`). Multi-task EA-LSTM (temp, DO + log-discharge head) learns hydrologic state (baseflow vs
   storm, low-flow) from labels the GBM cannot use directly; at prediction no discharge is read. Parity: the GBM gets
   the same information as a stacked feature from its own discharge GBM. Targets the low-flow DO / heat mechanism.
   ≈ 4–6 GPU-h.
3. **Pretraining on a larger hydrology corpus** (Caravan: ~6,800 basins, ERA5-Land forcings + streamflow; CAMELS-US
   with Daymet). Pretrain the EA-LSTM on discharge, fine-tune on temp/DO. Cost: forcing harmonisation (ERA5-Land vs
   gridMET) and a ~2–5 GB download (size estimate to the lead first). GBM parity: the same pretrained-state features.
   ≈ 8–12 GPU-h; uncertain gain.
4. **Upstream river-network context from forcings only** (down-weighted after the post-results review: basin-mean
   forcings gave +0.01–0.02 on val but ≈ 0 on test, so forcing-only network context is a weak lever; keep only a
   cheap variant that adds new information — routed upstream temperature with travel time and dam/reservoir
   position — and drop it first if levers 1–2 do not move inner-group loss). Beyond the basin-mean forcings already used: a graph over
   NHDPlus V2 (local delineation is built) that routes predicted upstream reach temperature with travel time
   (`pathtimema`), plus dam/reservoir position upstream (StreamCat dams are watershed totals only). Small expected
   gain (the basin-mean lever gave +0.01–0.02). ≈ 6–8 GPU-h.
5. **Physics-constrained multi-task temp+DO.** Already tried (`ealstm_physdo`: no gain on val or test). Lowest
   priority; only as part of lever 2.

### 3. Harder tracks (pre-registered before any result; practical justification first) and unseen test blocks
`M6/test` and `M6N/test` are spent (no training, tuning or second claim on them).
- **S-1 Extremes / heatwaves (justification: thermal-stress management needs the warm tail where all models are
  1.2–2.7 °C too cold).** Endpoint: RMSE and bias on days above the train-pooled P90 and in forcing-defined heatwave
  windows (track S definitions, frozen thresholds). Test block: Test-B (WY2026 test-role sites, Approved after
  2027-06-30). Power: ≈ 40–50 sites / 8–10 groups on val → the test-role cell is expected to have ≈ 3× that; a
  check is done before any fit.
- **S-2 Low-DO detection (justification: hypoxia is the harmful DO event; on val CSI ≤ 0.033 for every evaluated model).** Event E1a/E1b route
  via the event track; claim cell = Test-B together with M1 (E1a is spent on Test-A).
- **R Leave-region-out** (already pre-registered, secondary; its test cells equal the primary's, so it cannot carry
  a new claim).
- **N2 Unsensed reaches, future block:** WQP stations × WY2026 (unseen; new stations entering after 2025 included).
- **New geography (needs a data check, size to the lead first):** open continuous water-temperature networks outside
  the US (e.g. Swiss FOEN/BAFU, German state networks, UK EA continuous sites) with ERA5-Land forcings — a genuinely
  unseen block now, but with a forcing-source shift.
- Every new track: own protocol file (not the frozen one), frozen split sha, power check, single-look guard with the
  hardening of `sota_campaign.md` rule 7, write-up in `*_test_results.md`.

### 4. Caps and ETA
- Caps: ≤ 3 EA-LSTM configurations per lever (seed 0 on inner groups, then 3 seeds of the selected one), the same 3
  for the GBM; levers 1–2 first, 3–4 only if 1–2 move the inner-group loss; total ≤ 12 EA-LSTM configurations, all
  logged in `configs_tried.jsonl`. GPU jobs ≤ 2 h (chunked by epoch with checkpoint resume), GPUs 0/1 via the lock
  helper; CPU ≤ 6 threads.
- ETA: data builds 2–3 days (levers 1–2), training/val 4–6 days on shared GPUs, freeze of a new candidate within ≈ 2
  weeks. **The earliest unseen claim block is Test-B (Approved WY2026 values, after 2027-06-30)** unless a new-geography
  block is approved and built; a SOTA claim for M6 cannot be made before then.
- Honest expectation: temperature at sensor sites is close to an input ceiling; the most plausible SOTA scopes are
  unsensed reaches (more data) and low-flow/heat extremes (discharge-aware multi-task), not the primary benchmark.

## New-geography track: EA England (approved by the lead 2026-11-09; nothing downloaded yet)
- Primary block: EA Hydrology API water-quality sondes (OGL v3; ~1,960 stations, temp + DO, 15-min, 2012→present).
  Optional temperature-only secondary: Switzerland FOEN (~85 sites) + Bavaria GKD (153, CC BY 4.0). ≤ 20 GB total.
- Allocation note (written before any download): `docs/plans/ea_england_allocation.md` — all EA values ≥ 2023-10-01
  reserved now; all years at M6-EA test-role groups reserved once the split is frozen.
- Build order (after track R + event track): shared dataset `processed/ea_wq_sonde` (manifest, datasheet, OGL,
  content pins; daily stats from qualified values) → local basins (HydroBASINS / CAMELS-GB) → statics matched to the
  US StreamCat schema (same features for GBM/physics) → ERA5-Land forcings (Open-Meteo, as `forcings_daily` ERA5 path)
  → count independent groups + power check → pre-register `docs/preregistration/hydromet_ea_protocol.md` (zero-shot
  from US models AND fine-tuned on an EA train block with comparators fine-tuned identically; endpoints = standard +
  unsensed-reach + low-flow/heat extremes; impacted-site stratum "near discharge vs not" if derivable a priori) →
  freeze `hydromet_ea_split.json` → only then any fit/score.
- Research notes on all candidates (licences, counts, sizes): the data-check summary in the lead message of
  2026-11-09 (England / Switzerland / Bavaria / Ireland / Norway / France / Austria / TempER).

## Disk rule (lead, 2026-11-09): host at 98 %. Before ANY download/build (EA, Caravan, ERA5) check `df -h /home/bcheng`;
stop and report if free space would fall below 60 GB; M6 cap 20 GB total; delete intermediates when done.
- 2026-11-09 cleanup (≈ 8 GB freed): `interim/hydromet_upstream_forcing/trackn/agg`, `interim/hydromet_wqp_national/forc`
  (both already combined into pinned processed parquet files), the unregistered `test_look_rehearsal/{M6_test,M6R_test}`.
  Kept: sealed `test_looks/`, `trackn_val_v1` and `test_look_rehearsal/M6N_test` (registry artifacts of M6N/val_v1,
  inregion and val_v2), the shared simplified catchments (needed for lever 1 site delineation).

## Reserved sites: M6 must NEVER train, tune or select on them (lead notice 2026-11-09)
- `docs/preregistration/sonde_newsites_list.json` (sha `6200a77c…`): 729 USGS sites = M1 EW-v3 / M1-N unseen test block;
  ~470 have DV files in `processed/sonde_daily`. The SOTA lever-1 site builder MUST call
  `sentinel.hydromet_sota.reserved.exclude_reserved` (logs the count) and `assert_no_reserved` on the final
  training/tuning set; `tests/test_hydromet_sota_reserved.py` (pin, exclusion, failure on a listed site). Note: the
  list's overlap detail shows some listed sites also appear as track-N WQP stations (evaluation only, already scored).
- EA England: all values ≥ 2023-10-01 are reserved for M6's own test block (`docs/plans/ea_england_allocation.md`).
- **Code location rule for all SOTA work:** new code goes ONLY in unhashed paths (`sentinel/hydromet_sota/`,
  `scripts/{train,eval}/sota_hydromet_*.py`, `configs/hydromet_sota/`), never `sentinel/models/hydromet/*`,
  `configs/hydromet/*`, `scripts/*/hydromet_*.py` — the frozen code hash must stay `5bd7d329…` for the track-R freeze.

## Queue state (2026-11-10 12:00 EST) — GPU rule update: GPUs 0–9 via the cap file
- GPU queue `q_gpu4.sh` (log `q_gpu4.log`): any GPU 0–9 through the lock helper, `SENTINEL_GPU_MIN_FREE_MB=10000` so a
  GPU partly used by another user is never taken; jobs ≈ 1 h (< 2 h). Remaining track-R GPU work: `trackr_R6_pacific_ealstm_s2`
  only (all other 17 early-stopped seeds done; refits dropped per D4).
- CPU queue `q_cpu4.sh` (log `q_cpu4.log`): track-R baselines; R1–R3 base + bag done, R4 base running (since 08:27), then
  R4 bag ×3, R5, R6 (≈ 1.2 days; the bottleneck).
- When both finish: `M6R/cv_v1` score (see Next steps 3), then `hydromet_freeze_candidate.py trackr` (pins copied from
  FREEZE_hydromet.json), then the event track (`events_hydromet.py`, `events.py` v1.0 sha 4279d242…).
- SOTA levers may now run in parallel on free GPUs (lead, 2026-11-10); code only in unhashed paths
  (`sentinel/hydromet_sota/`, `scripts/*/sota_hydromet_*.py`); lever-1 builder must use `hydromet_sota.reserved`.

## Registered (all self-checks PASS; RESULTS.md lines of 2026-11-08/09)
- `M6/val_v2` (cycles 1–4, 1,623 entries), `M6/val_v3` (cycle 5 upstream-basin forcings, 2,103 entries),
  `M6S/val_v2` (track S strata, 132), `M6N/val_v1` (track N val-role stations, 197), `M6N/inregion_val_v1` (197).
- No original-track endpoint won ("competitive"). Frozen and tested: `ealstm_ws_full` (cycle 6, `M6/val_v4`); see TEST RESULTS above.
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
1. Done: FREEZE_hydromet.json written 2026-11-09 21:13Z; `M6/test` and `M6N/test` scored once.
2. Done: `M6N/val_v2` (registered from the M6N rehearsal predictions). Optional: in-region look of the `_ws` models
   from `$H/trackn_val_v1/ws_inputs` (descriptive).
3. Track R when all folds finish: `hydromet_score.py --track-r cv --primary --comparators docs/preregistration/hydromet_track_r_comparators.json --models ealstm,ealstm_gbmblend[,ealstm_full] --register --experiment M6R/cv_v1` over
   all `trackr_*` prediction dirs (drop `trackr_*_full_*` unless all 18 exist, protocol §10 amendment); then strata
   on track R (`hydromet_strata.py --track-r cv`).
4. Then `hydromet_freeze_candidate.py trackr` → lead writes FREEZE_hydromet_trackR.json (same code hash) → `--look M6R/test` (secondary).

## 2026-11-09 (lead): M6 frozen; primary and track-N looks started
- **Review:** the line-level check PASSED (REVIEW_REPORT_PRE_TRAINING.md "M6 HydroMet — line-level check (2026-11-09)").
- **Freeze:** `docs/preregistration/FREEZE_hydromet.json` (0444) is the candidate minus NOT_A_FREEZE and no_test_data_audit, plus status FROZEN, authorised_by, frozen_at and candidate_sha256. `load_freeze(check_code=True)` OK (code 5bd7d329…); `verify` PASS.
- **Archive:** `sentinel_runs/freeze_archives/hydromet_freeze_20261109T161338.tar.gz` (48 files, sha 138dae5e…); registry row `hydromet/prereg_snapshot`.
- **Looks:** the lead started `hydromet_test_look.py --look M6/test`, then `--look M6N/test`, on CPU. Logs are in `sentinel_runs/hydromet/logs/test_look_*_20261109.log`.
- **Do NOT edit any of the 46 frozen files until the track-R look.** The open minor items L1–L4 stay unfixed, since fixing them would change the code hash.


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
- **Information for M6; no M6 file edited.** FREEZE_hydromet-hashed files were left untouched. Your `hydromet_sota/reserved.py` already protects your own reserved sites. The 27 + 21 raw IV new-site directories are mode 000: any non-frozen builder that globs `raw/usgs_iv` will fail loudly on them (intended). Please use `sentinel.data.reserved` in any new non-frozen loader of the sonde tiers.
