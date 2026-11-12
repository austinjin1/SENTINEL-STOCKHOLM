# SENTINEL-HydroMet (M6) — SOTA levers 1–4: pre-registration

**Status:** v1.0, written 2026-11-10 12:08 EST **before any lever model was fitted or scored**. Only data counts
(locations, station-days, metadata spans) were computed before this file. Lead approval of levers 1–5: 2026-11-09
(caps: ≤ 3 configurations per lever, ≤ 12 EA-LSTM configurations in total; GBM and physics comparators always get the
same new data and inputs).

- This file is NOT part of the frozen M6 code hash. The 46 files of `FREEZE_hydromet.json` (`hydromet_protocol.md`
  included) are never edited; track R keeps code hash `5bd7d329…`.
- All lever code is in unhashed paths: `sentinel/hydromet_sota/`, `scripts/{data,train,eval}/sota_hydromet_*.py`,
  `configs/hydromet_sota/`.
- The EA England new-geography track (lever 5) gets its own protocol (`hydromet_ea_protocol.md`), split file and
  power check before any fit.

**Authors:** Bryan Cheng (AI-assisted). **Implements:** `docs/plans/sota_campaign.md`, RESEARCH_PLAN §15.1 / §15.5,
handoff `docs/handoff/M6-hydromet.md` ("SOTA plan").

## 1. Why each lever (practical justification, fixed before any result)
Diagnosis (handoff "SOTA plan" §1): the GBM ties the EA-LSTM because both see the same information; the binding
constraint is the number of independent training locations (the EA-LSTM memorises its 1,456 fit sites: train MSE ≈ 0.06
vs inner ≈ 0.21); temperature is near an input-limited ceiling; low-flow / hydrologic state is not observed by any input.

| Lever | Practical justification | Expected mechanism |
|---|---|---|
| 1 More labelled locations | Managers need estimates at the ~10⁵ unsensed reaches; WQP grab samples exist at thousands of such reaches but no model uses them for training | more locations per basin group → less site memorisation (the EA-LSTM's failure mode); tests whether the track-N gap is a coverage problem |
| 2 Discharge as an auxiliary target | Low flow drives summer heat and hypoxia; discharge labels exist at many gauges but discharge can never be an input at an unsensed reach | multi-task supervision teaches the hidden state baseflow vs storm flow; nothing is read at prediction |
| 3 Pretraining on a hydrology corpus | Only if ≤ 5 GB with manifest + licence (size check reported to the lead first) | more basins for the hydrologic state |
| 4 Routed upstream temperature + dam position (cheap variant) | Reservoir releases and upstream heat are unobserved by the site-cell/basin-mean forcings | network context; dropped first if levers 1–2 do not move the held-out-group loss |

## 2. What is held fixed
- **Split and blocks:** FINAL split `fd4bb14f…`; train block WY2008–2021 is the only fitting block; the val block
  (WY2022–2023) is used only for descriptive scoring. FINAL test-role sites, track-N test-role stations, the test
  blocks (Test-A spent; test_b WY2026) and the 729 M1-reserved sites are never loaded by any lever training code.
- **Inner (held-out-group) set:** the 256 inner sites in 20 train-role basin groups of the reference runs (recomputed by
  `sentinel.hydromet_sota.sites.final_frame`, checked equal to `ealstm_ws_v1_s0/data_info.json`). Lever data at inner
  groups are **excluded** (neither fit nor inner), so the early-stopping and selection criterion is identical across
  all levers and the reference. Inner windows and the inner loss use the 7 M6 targets only (never weak labels or
  auxiliary targets).
- **Reference recipe** (`ealstm_ws`, protocol v0.10): EA-LSTM, site-cell + upstream-basin gridMET (22 forcing
  channels) + DOY harmonics, 188 StreamCat/site statics, hidden 128, dropout 0.4, static-input noise 0.5, lr 5e-4,
  weight decay 1e-4, 200 steps/epoch, early stopping on inner point MSE (patience 6, ≤ 60 epochs), seeds 0–2.
  Reference runs: `ealstm_ws_v1_s{0,1,2}`; reference comparators `forcing_gbm_ws_bag_s{0,1,2}`, `air2stream_regional_ws`,
  `dosat_ratio[air2stream_regional]_ws`.
- **Trainer:** `sentinel.hydromet_sota.lstm.SotaEALSTM` subclasses the frozen `HydroMetEALSTM` (same network, decode,
  loss and prediction). With every lever option off it reproduces the frozen trainer exactly (same RNG stream; unit
  test `tests/test_hydromet_sota_lstm.py` compares histories and weights on CPU), so the existing reference runs need
  no re-run. Additions: per-epoch resume checkpoints (jobs ≤ 2 h), host-memory batch gathering (identical values),
  the weak-label stream (lever 1) and auxiliary targets (lever 2).

## 3. Lever data (built 2026-11-10, `processed/hydromet_sota`, `scripts/data/sota_hydromet_build_lever_data.py`)
Rules per location, first match wins (`sentinel/hydromet_sota/sites.py`): M1-reserved (list or any reserved number
inside the key) → already in the FINAL split → HUC4 not owned by a FINAL group (`new_region`; track N gives such
regions val/test by hash) → owning group val/test-role → owning group inner → flow-connected (NHDPlus V2 downstream
main path ≤ 500 km, either direction) to a FINAL val/test site or a track-N val/test/excluded station → **fit**.

| Pool | Candidates | fit | Excluded (rule: n) |
|---|---|---|---|
| USGS DV sites not in the split (eligible stream site, gridMET, StreamCat, ≥ 365 train-block days of temperature/DO or discharge; 416 M1-reserved removed first) | 819 | **447** in 32 fit groups: 42 with temperature/DO (lever 1a), 430 with discharge (lever 2; 405 discharge-only) | test group 148, val group 105, inner group 69, new region 30, flow-connected 20 |
| WQP grab-sample stations, track-N train role (`ffb41471…`), ≥ 4 accepted sample days in the train block (track-N filters: stream, accepted, uncensored, depth ≤ 2 m, no time-series composites) | 4,824 | **3,545** in 32 fit groups; 256,628 temperature and 230,992 DO (mg/L) station-days (3,517 / 3,288 stations) | new region 618, inner group 358 (6 via a flow-connected inner group), flow-connected 303 |

- Lever 1a adds only 42 dense sites (most new USGS temperature/DO sites start after WY2021); it is therefore folded
  into every lever-1 configuration instead of being a configuration of its own.
- Inputs for added locations are built with the M6 methods: WQP stations use the track-N inputs (station-cell gridMET,
  `upstream_forcing_trackn`, StreamCat of the COMID, `elev_m` = `elevcat`); new USGS sites use `forcings_daily`,
  StreamCat and upstream forcings built by the same builder (`upstream_forcing_sota.parquet`, same QC rule).
- Weak labels: the station-date mean of grab samples is a label for `temp_c_mean` / `do_mgl_mean` (as in track N);
  `temp_c_7dadm` and `do_mgl_min` cannot be observed from grab samples.
- Discharge target (lever 2): `log(discharge_cfs_mean + 0.01)` from `processed/sonde_daily`, train block only, at fit
  sites (FINAL fit sites with discharge + the 447 new sites). Never an input.

## 4. Pre-declared configurations (EA-LSTM; counted in `configs_tried.jsonl`, model prefix `sota_`)
Each configuration is first trained with seed 0 (screening); within a lever the configuration with the lowest
**primary inner MSE** (mean standardised point MSE over `temp_c_mean`, `temp_c_7dadm`, `do_mgl_mean`, `do_mgl_min`
on the 256 inner sites, at the early-stopping epoch) gets seeds 1 and 2.

| ID | Lever | Data / option | Counts toward 12 |
|---|---|---|---|
| C1a | 1 | +42 DV sites + WQP weak labels, weak loss share s = 0.25 | 1 |
| C1b | 1 | +42 DV sites + WQP weak labels, s = 0.5 | 2 |
| C2a | 2 | original data + log-discharge head at fit sites with discharge, λ = 0.5 | 3 |
| C2b | 2 | C2a + the 405 discharge-only sites, λ = 0.5 | 4 |
| C12 | 1+2 | best lever-1 data + best lever-2 option (only if both levers "move", §6) | 5 |
| C3x | 3 | pretraining (only if lead-approved after the size check; ≤ 3, declared in a dated amendment before any run) | ≤ 8 |
| C4a | 4 | routed upstream temperature + dam position (only if §6 gate passes; ≤ 3, declared before any run) | ≤ 11 |
| A* | arch | the top-ranked, critic-approved design of `docs/plans/architecture_research/M6_architecture_plan.md`, declared in a dated amendment before any run, within the remaining cap | ≤ 12 |

- **Weak-label stream (lever 1).** Each batch of 256 windows holds round(s·256) windows ending in the train block at
  WQP stations (station drawn ∝ √n_days) and the rest at dense fit sites (frozen `target_share` sampling). Per target,
  loss = (1 − s)·[mean over dense cells] + s·[mean over weak cells] (a part without cells of that target drops out);
  then the mean over targets as in the frozen loss. GBM parity: the weak rows of a target get row weights so that their
  total weight is the same share s of that target's training weight.
- **Auxiliary discharge (lever 2).** One extra Gaussian output for log-discharge, standardised on fit sites × train
  block; loss = frozen 7-target loss + λ·NLL(log Q). Discharge-only sites enter the dense stream with their discharge
  share. The discharge output is never scored and never an input.
- No hyperparameter of the reference recipe is tuned again; `s` and `λ` are the only lever settings.

## 5. Comparator parity (every configuration that gets 3 seeds)
- **GBM:** `forcing_gbm_ws` features (76 causal forcing features on site-cell + basin-mean channels, 188 statics,
  calendar), the same 3-point grid (`max_rows_per_site` ∈ {1000, 3000, all}) selected per target on the same inner
  rows, refit on fit + inner, bagged over seeds 0–2 (`sota_gbm_<cfg>_bag`). Primary targets only (per-target models;
  the secondary targets do not enter any comparison). Lever 1: the same added locations and weak rows (share-s weights).
  Lever 2: a discharge GBM on the same discharge labels, cross-fitted by basin group (5 folds) on the fit sites, gives a
  stacked input `f_qhat_log` (+ its 7- and 30-day trailing means); inner and val sites get the full-fit prediction.
- **Physics:** `air2stream_regional_ws` (donors: every fit + inner site with ≥ 365 train-block daily temperatures and
  calibration NSE ≥ 0.5, so added DV sites can become donors; grab samples never qualify) and
  `dosat_ratio[air2stream_regional]_ws` (monthly DO/DO_sat ratio over the same rows; weak rows weighted with share s via a
  weighted median). Lever 2 adds `air2stream_q_regional_ws`: the air2stream update divided by θ^a4 with θ = Q̂/mean(Q̂)
  from the discharge GBM (Toffolon & Piccolroaz 2015 flow-scaling term), a4 ∈ [0, 2].

## 6. Decision rules (inner groups only; no val result enters any decision)
- **Screening:** within a lever, lowest seed-0 primary inner MSE.
- **"Moves the held-out-group loss":** the selected configuration's 3-seed mean primary inner MSE is at least 0.006
  below the reference's (`ealstm_ws_v1_s{0,1,2}`: 0.1184 / 0.1240 / 0.1113, mean 0.1179; seed SD 0.0064). The
  threshold (≈ 1 seed SD) was fixed from these train-role inner numbers before any lever run.
- **Gate for levers 3–4 (handoff):** they run only if lever 1 or lever 2 moves the loss; lever 4 is dropped first.
- **Lever candidate:** the 3-seed configuration with the lowest 3-seed mean primary inner MSE, provided it moves the
  loss; otherwise no lever candidate exists and the M6 label stays "competitive".
- Every configuration and every comparator is reported (wins and losses), with the number of configurations tried.

## 7. Validation reporting (descriptive only; never a claim, never a selection input)
- Cell: FINAL val-role sites × val block (42-day embargo), ≥ 60 approved days, cells finite for every compared model.
- Per endpoint: median per-site KGE (+ RMSE); paired Δ vs (a) the lever's own GBM bag and physics members (parity),
  (b) the reference EA-LSTM `ealstm_ws` and its frozen comparators; basin-group paired bootstrap (B = 2,000, seed 0),
  95 % CI; every-seed signs. Track-N val cell (val-role WQP stations × val block): median per-station RMSE, same
  bootstrap. Registry: `M6SOTA/inner_v1`, `M6SOTA/val_v1`, `M6SOTA/trackn_val_v1`.
- The val cell has been used for six M6 selection cycles; its numbers are reported as context and given no weight in
  any decision (handoff: val gains of +0.01–0.02 did not replicate on test).

## 8. Unseen claim blocks and power (before any lever result)
- **Test-B** (WY2026, FINAL test-role sites; Approved data expected after 2027-06-30): the lever candidate, its parity
  GBM bag and physics members enter the single `M6/test_b` look (one freeze, `FREEZE_hydromet_test_b.json`, written by
  the lead with a later `sonde_daily` pin); Holm family = the frozen 7 models × 4 endpoints + the lever candidate × 4.
  Power from sonde-index metadata only (record spans, no values): 317 / 315 / 155 / 164 sites in 33 / 31 / 19 / 19
  basin groups (temp mean / 7-DADMax / DO mean / DO min) vs Test-A 34 / 33 / 22 / 22. Scaling the Test-A minimal
  detectable Δ (0.014 / 0.022 / 0.07 / 0.06 at α = 0.05, protocol §8) by √(G_A/G_B) and an assumed 1.2× inflation for one
  water year per site: ≈ 0.017 / 0.027 / 0.09 / 0.08 (α = 0.05), larger under Holm. Val-sized gaps (~0.01) are not
  detectable; a lever must widen the gap to be claimable. Stated with every result.
- **N2** (track-N test-role stations × WY2026, incl. stations first sampled after 2025): power check after the WQP
  WY2026 pull, before the freeze.
- **EA England** (lever 5): separate protocol.
- **Single-look hardening** (`sota_campaign.md` rule 7; testlook.py plus the M7/D4 lessons): sealed registry claim row,
  rescore refused after scoring, protected-data access closed at sealing, every artifact verified before the claim,
  BaseException aborts the claim, claim tied to pid/host, import-closure code hash of every module the look imports.
  Only a lead-written FREEZE unlocks a test block; no lever code path can load a test block or a test-role site.

## 9. Caps, compute and ETA
- ≤ 3 configurations per lever, ≤ 12 EA-LSTM configurations in total (table §4); GBM/physics get the same data per
  configuration that is seeded. GPU jobs ≤ 2 h (resume per epoch), any GPU 0–9 through `sentinel/core/gpu.py` with
  ≥ 10,000 MiB physically free; CPU ≤ 6 threads per job; M6 new-data cap 20 GB (lever data ≈ 0.6 GB).
- ETA: lever 1–2 screening 1–2 days once GPUs free up; 3-seed + parity 1–2 days; lever candidate (if any) within
  ≈ 1 week. No claim before Test-B data exist (after 2027-06-30) or the EA England block is frozen.

## 10. Deviations log
| Date | Change | Reason |
|---|---|---|
| 2026-11-10 | v1.0 | initial pre-registration |
| 2026-11-10 12:33 EST | **Amendment A1 (lead rulings, `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` "M6 HydroMet: rulings", 12:26 EST), recorded before any lever configuration was scored.** (i) Lever 3 (pretraining) is deferred and its 3 slots are retired (Caravan 24.8–29.0 GB exceeds the data cap; CAMELS-US forcings do not match gridMET); nothing is downloaded for it. (ii) The lever budget is **≤ 8 EA-LSTM configurations**: C1a, C1b, C2a, C2b, C12 (gated) and lever 4 ≤ 3 (gated). (iii) Row A* of §4 is replaced by the separate architecture-v2 budget (≤ 12, owned by the M6 architecture-v2 engineer, `hydromet_arch_v2_amendment.md`); the M6 total becomes ≤ 20, disclosed. (iv) Lever models are secondary arms; they can be a track's claim model only through the claim-model rule written in the architecture-v2 amendment before any result. (v) The lead approved the 4-primary-target selection criterion for arch-v2 arms; the lever decision metric stays as pre-registered in §4/§6 (primary inner MSE), and the 7-target inner MSE (the early-stopping criterion) is reported next to it; the lead is asked to confirm. (vi) Display name of M6: SENTINEL-Hydro (prose only). State at A1: C2a seed 0 in training since 12:20 EST (its log prints per-epoch inner losses; no lever result is registered or scored); no other lever run started. | Lead rulings 2026-11-10 |
| 2026-11-10 12:33 EST | **Amendment A2 (data-hygiene rules 1–5 of the architecture plan §6, mandatory per the lead's rulings), before any lever-1 / C2b fit.** Rule 3 added to `sites.py` as rule 1b `reserved_comid`: a location whose NHDPlus V2 COMID holds an M1-reserved site (and, for WQP stations, a FINAL-split site) is excluded. Effect: 2 USGS sites dropped → **445** new USGS fit sites (41 with temperature/DO, 429 with discharge, 404 discharge-only; 32 fit groups); WQP pool unchanged (3,545 stations). Rule 1: both `sentinel.hydromet_sota.reserved` and the shared `sentinel.data.reserved` assert on the fit pool and on every training set. Rule 4: code and outputs pinned in `docs/preregistration/hydromet_sota_lever_pins.json` (`lever_sites.parquet` `62c4bf53…`, `wqp_train_obs.parquet` `346fb06a…`); `load_lever_train` refuses unpinned data. Rule 5: protected rows are counted only (Test-B power from index metadata). The §3 table counts (447 / 42 / 430 / 405) are superseded by these. | Lead rulings 2026-11-10 |
| 2026-11-10 12:55 EST | **Amendment A3 — lever 4 defined before any lever-1/2 result** (no lever result row exists; C2a seed 0 still training; the gate of §6 is unchanged and lever 4 runs only if it passes). **C4a** (the only lever-4 configuration; C4b/C4c are not used): the data and options of the lever candidate (C12 if run, else the lever winner that moved the loss) plus (i) one dynamic channel `rt_tmean`: the upstream-basin mean air temperature shifted by τ days, τ = area-weighted median over the site's upstream NHDPlus V2 flowlines of `pathtimema(flowline) − pathtimema(site)` (VAA mean-annual travel time; τ ≥ 0, so the channel stays causal), and (ii) four statics of upstream impoundment from HydroLAKES pour points (on disk, `processed/hydrolakes`) with `Vol_total` ≥ 1 million m³, snapped to an NHDPlus V2 catchment: river distance to the nearest one upstream (km, VAA `pathlength` difference; 1,000 km if none), its travel time (days), log10 of its volume, and the number within 100 km upstream. Parity: the GBM gets `rt_tmean` with its 7- and 30-day trailing means and the four statics; the physics family adds the four statics to the air2stream donor-attribute vector. One configuration (counts toward the ≤ 8 lever budget). | Handoff SOTA plan lever 4 ("cheap variant"), pre-declared to avoid a post-hoc design |
| 2026-11-10 13:01 EST | Lead confirmation of A1 (v): the lever decision metric stays the primary inner MSE of §4/§6; the 7-target inner MSE is reported next to it and decides nothing. No lever result existed at this confirmation. | Lead reply 2026-11-10 |
| 2026-11-10 13:26 EST | **Correction of written times** (lead request; before-state = registry snapshot row `3deeb3b6bd27`, protocol sha256 `78159a59…8adf`). Corrected in place: (1) header "written 2026-11-10 13:00 EST" → **12:08 EST** (actual first write 12:08:16 EST); (2) A1 row "12:35 EST" → **12:33 EST** (A1 and A2 were appended together at 12:33:31 EST; the 12:35 was an estimate); (3) lead-confirmation row "13:10 EST" → **13:01 EST** (appended 13:01:23 EST); (4) `hydromet_ea_protocol.md` §4 PROPOSED block "13:10 EST" → **13:01 EST**. From now on every written time is taken from `date` immediately before writing. **Visibility record for A3 (written 12:55:19 EST):** C2a seed 0's log had printed epochs 0–10 (12:25:33–12:52:38 EST): 7-target inner MSE 0.2238–0.2431, primary inner MSE 0.1168–0.1265; I had read these lines. No other lever number existed. Also in this edit set: `hydromet_ea_protocol.md` §4 states that ineligible stations get no group or role and are never used, and names the reproducible split writer (`sota_hydromet_ea_build.py split --share-test S --i-have-the-lead-ruling`). | Lead request 2026-11-10 |
| 2026-11-10 14:57 EST | **Amendment A4 — joint US look (lead rulings `docs/preregistration/hydromet_arch_v2_lead_rulings.md`, sha256 `b144974a96c2140c933ccbe1e8d3247d9a1ffa0ad6663ee958237e33eb4179b0`, §2.1 Q7 and §2.2 "Co-scored project models"; lead stamp row `b37a0319964c`, 2026-11-10T19:52:24Z), recorded before any lever validation number.** (i) §8 "Test-B" and "N2" are superseded: there is no separate lever entry in an `M6/test_b` look. The unseen US blocks are scored once, in the joint US look (tracks T-B and N2) defined by `hydromet_arch_v2_amendment.md` and the rulings file. The words "Holm family = the frozen 7 models × 4 endpoints + the lever candidate × 4" are struck: the v1 models are analysed in the joint look exactly as frozen in `hydromet_protocol.md` (same models, Holm family, comparators, every-seed rule) and are reported as secondary with no new label. (ii) **Choice (Q7): no lever model is a claim candidate.** No lever model is or will be named as a claim candidate in the registry; every lever model is secondary in the joint look and can earn no label. This choice is final for this round. (iii) Co-scoring: every lever configuration that has 3 seeds at the joint freeze (the screening winner of lever 1 and of lever 2, and C12 / C4a if their §6 gate passes), with its parity GBM bag and physics members, is scored in the joint look by the same code and rows as every other model, every seed; none is withheld and the list follows from §6 only, never from a validation number. Their inner predictions are supplied to the arch-v2 engineer on request. Tables show them side by side with the v1 and arch-v2 models; no label is issued on an endpoint where the arch-v2 claim model's point estimate is not better than that of every v1 frozen model and every lever model scored in the same look. (iv) §6 "Lever candidate": still determined and reported by the rule as written; "otherwise … the M6 label stays 'competitive'" now reads "a lever candidate earns no label; an M6 label can come only from the arch-v2 claim families of the joint look". The §6 gate for C12 / C4a is unchanged. (v) §7 (descriptive validation reporting) is **on hold**: no lever val-cell or track-N val-cell number is computed until the lead lifts the hold in writing (lead rule 14:25 EST "scoring: inner split only"). The hold is enforced by the file `<runs>/hydromet/sota/HOLD_VAL_SCORING` (written 14:27:26 EST), which `sota_hydromet_score.py val` and `sota_hydromet_trackn.py predict/score` refuse to pass. No lever validation number exists. (vi) Upstream-forcing table of the lever track (rulings L7): `processed/hydromet_upstream_forcing/upstream_forcing_sota.parquet`, sha256 `925a21ecf6e3678eaaf824179e09e453851030673418aefa1285ddeea026984f`, 144,857,839 bytes, written 13:46:21 EST, set read-only (0444) at this amendment; 3,445,923 rows = 447 site keys × 7,709 days (2005-10-01 … 2026-11-08), no missing value, no duplicate (site, date); 447 basins (439 local, 8 NLDI polygon, 0 point fallbacks). It is complete and final: it is not rebuilt or extended. 445 keys are the USGS fit sites of A2; 2 keys (`USGS-02298123`, `USGS-262528080202700`) are the rule-1b `reserved_comid` sites dropped by A2 after the basin list was drawn — forcing rows only, never read by any loader (every reader selects fit-site keys). The `status` column is the builder's flag (the last 60 days of the 2020, 2022 and 2026 gridMET files, as in the v1 table); values after 2026-09-09 are gridMET provisional / early values and stay as downloaded. Companion files: `coverage_sota.parquet` `bd9df7b0…9a1e5e`, `basins_sota.parquet` `b93446b9…ac1f2e`. **Visibility at this amendment:** 2 lever result rows in the registry (`M6SOTA/inner_v1`: C2a seed 0, primary inner MSE 0.117455, 7-target 0.216031). I had read the training-log lines of C1a seed 0 (epochs 0–19: 7-target inner MSE 0.1977–0.2241, primary 0.1131–0.1206) and of C1b seed 0 (epochs 0–18: 0.2077–0.2345, primary 0.1134–0.1251); both still training, neither scored. No lever validation number and no arch-v2 inner row (`M6V2…`: 0 rows) exists. The choice in (ii) removes a claim path and adds none. | Lead rulings 2026-11-10 (arch-v2 stamp) |
