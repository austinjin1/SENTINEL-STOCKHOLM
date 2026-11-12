# Handoff — Downstream workstream (Goal 4: D1, D2, D3, D4, D5, D6, D8 + national tracks D1n, D2n) — 2026-11-08 (late)

## CURRENT STATE (read first; supersedes the dated sections below)
- **Test blocks SPENT (single looks 2026-11-08 20:22–22:12; never re-use them for a claim, §15.5 item 1):**
  D1 Chicago D1b 2024–25, D1n national beaches 2021–25, D5 NRSA 2023-24, D6 origins 2018 + 2021, D2 WA 2022–26.
  FREEZE files + prereg archives + `test_look/` markers exist; outcomes are in
  `docs/preregistration/downstream_<app>_test_results.md` and RESULTS.md. **Frozen protocols are read-only (0444) and
  must never be edited** (2026-11-09: post-freeze appends were moved out and the protocols restored byte-for-byte
  from the prereg archives; all FREEZE hashes verify). Labels (post-results review): D1n beats its re-implemented VB incumbent and the strongest simple
  baseline on its track, **NOT SOTA** (no Brooks-style RF compared); D1, D5, D6, D2 competitive.
  **H-G4 "beats incumbent": 1 of 4 (D1 via D1n, caveated) → not met.**
- **Not yet tested (validation only):** D2n national microcystin, D3 (underpowered), D8 (negative), D4 TapRisk
  (in progress). Any new D1n/D2 track must pre-declare Brooks-style RF + GBM comparators and the 2026-season test
  block before any fit.
- The dated sections below are history; numbers in the "State per application" table are validation-era.

Interpreter: always
`/home/bcheng/miniconda3/envs/sentinel2/bin/python` (base `python` has no lightgbm).
Single FIFO CPU slot `common.cpu_slot` (locks in `$SENTINEL_RUNS/downstream/locks/`). No GPU used.

## Reconciled 2026-11-08 (all pre-pause jobs had finished; results now recorded in the protocols)
D6 it2 (Kanankege re-impl), D1 cycles 1–2, D5 it2, D8 it1, D3 v2 — see each protocol's iteration log.

## Running / queued: nothing (all 2026-11-08 runs finished and are recorded in the protocols).
Runs this session: D2 `toxinwatch/val_emb7_it3`; D1 `safeswim/it3_d1b`; D6 `musselguard/phase1_it2b_kanankege_grid`; D5 `biointegrity/it3_ens`
(no refit); D1n `d1n_beaches/phase1`; D2n `d2n_microcystin/phase1`.

## State per application (val)
| App | Status | Freeze candidate |
|---|---|---|
| D1 SafeSwim (Chicago) | cycle 3: logistic `compact_each+q` 0.743 beats official City model +0.129 [0.046,0.209] AND feature-beach GM +0.043 [0.014,0.080] (week; beach clusters also > 0) → §4a met on val | L2 logistic, `compact_each+q` |
| D1n national beaches | NEW track (split sha aa601b50…). Logistic beats VB-MLR incumbent on known +0.039 [0.018,0.063] and new beaches +0.016 [0.002,0.040] (Holm) and the history-rate baseline; GBM known +0.050, new n.s. | L2 logistic (phase 1) |
| D2 ToxinWatch (WA) | it3 (WA + D2n national training rows; incumbents also refit pooled): GBM beats strongest incumbent (v2_pooled) known +0.009 [0.001,0.045], new +0.060 [0.006,0.163] (Holm over 2 sets) and the strongest simple baselines | GBM it3 |
| D2n national microcystin | NEW track (split sha 0d9eb684…). GBM ties INLA-class v2 (known −0.007 [−0.020,0.000]; new +0.001 [−0.019,0.023]); beats v1 and simple baselines → competitive | — |
| D3 KillAlert | v2 (IA+DE): wx_clogit beats season/location/year placebo (Holm) but not temp-anomaly clogit (+0.046 [−0.063,0.148]); 2nd label search found no adequate licensed source → **underpowered** | wx_clogit (if frozen) |
| D5 BioIntegrity | it3_ens (tuned GBM+RF mean): beats incumbent +0.014 [0.005,0.023] and every RF row (Holm) on new locations | `ens_tuned_gbm_rf` |
| D6 MusselGuard | pressure GBM front AUPRC 0.093 vs Kanankege re-impl 0.022: +0.071 [0.025,0.128], Holm (it2b: 23-config incumbent grid, interior optimum, same result) | `pressure_gbm` |
| D8 RiverPulse | it1 (asinh): GPP negative (Maavara-style RF best, all CIs span 0); ER: GBM < RF | none (negative) |

## Next steps
1. **H-G4 (stated before the looks, 2026-11-08):** from the five looks, "beats its incumbent" can count only
   D1 (+D1n national track), D2 and D5 — at most 3 of the 4 needed — so H-G4 cannot be met without D4 (started) or
   D7. D6 counts toward "implemented" only (D6 protocol §10/§12). On val: D1, D2, D5 (and D6) beat their incumbent. Freeze-candidate
   records (NOT freeze files): `docs/preregistration/FREEZE_CANDIDATE_downstream_{safeswim,d1n_beaches,biointegrity,
   musselguard,toxinwatch}.json`. The lead runs a pre-test review, then authorises test looks. **Do not edit
   those apps' protocols/modules/scripts or wqp_track.py before the review — it changes their scoring hashes.**
   Test paths written 2026-11-08 (`scripts/train/downstream_<app>_test.py`, `--allow-test` refuses without
   FREEZE + claims the single pass; `--dry-run-on-val` reproduces each app's validation numbers exactly —
   done for all five). Refusal tests: `tests/test_downstream_testpaths.py`. Records regenerated after the code
   was final; any further edit to a hashed file invalidates them. D2 caveat: claim holds under the
   pre-registered Holm (2 sets per comparator family) but NOT under an all-four Holm (reported).
2. Lead go-ahead → write FREEZE files (D1, D5, D6 first; D2 when chosen) and score each test once.
3. D8: site-regime + daily-anomaly decomposition, or report negative. D3: freeze as underpowered or stop.
4. Post-results review (R10) before numbers leave the repo.

## Requests to the lead (shared files)
- Done 2026-11-08 (approved): wqp_track.py + wqp_harmonize.py in APP_SCORING_FILES for d1n/d2n.
- Done (approved): national-row files in APP_SCORING_FILES["toxinwatch"]; testlook.py added to
  SHARED_SCORING_FILES (single-pass guard is part of every app's scoring hash).
- Never read/print credential files (lead instruction 2026-11-08).

## Pre-test gate fixes (2026-11-08, after REVIEW_REPORT_PRE_TRAINING.md "Downstream — pre-test gate")
All criticals applied; records regenerated (scoring sha: D1 4ff7af0b, D1n 95beea1c, D5 65dd002e, D6 776addaf,
D2 e49b7d03); dry runs re-run (all claim_met on val; D2 label "not robust to all-comparison correction" on val).
Shared: `testlook.py` rewritten (seed rule, data pins `docs/preregistration/downstream_<app>_data_pins.json`
verified before the claim, save-predictions-before-registry + `--rescore-from-saved`, AUPRC label fix,
`prereg_snapshot` for the lead). New runs: `d1n_beaches/alpha_wide` (α = 100 re-selected, interior),
`musselguard/it2_scrule` (SC rule at prevalence). **D6 no longer counts toward H-G4 "beats incumbent"**
(original §10 text) → on val, H-G4 "beats incumbent" = D1, D2, D5 (3 of 4 needed).
Awaiting the lead's delta review; do not edit hashed files.

## D4 TapRisk (started 2026-11-08 18:30; validation only; new files only)
- Protocol `docs/preregistration/downstream_taprisk.md` (v1.0 + §9 counts/power + reproduction deviation);
  novelty check `docs/novelty_checks/downstream_taprisk.md` (Pennino 2020 full text PMC8204728; Scanlon 2022).
- Split frozen: `downstream_taprisk_split.json` sha `a4a1fc7b…5b3298` (24/13/19 states). Power: D4a GW nitrate
  val_known 177 pos, val_new 41 (underpowered); D4b SW DBP 434 / 108.
- Code: `sentinel/models/downstream/{taprisk,taprisk_models}.py`, `scripts/train/downstream_taprisk.py`,
  `scripts/data/build_taprisk_links.py`.
- **Running:** NLDI snap of 35,803 service-area points (PID 1033208; ~1.9 s/point → ~19 h; surface systems first;
  checkpoint `processed/taprisk_catchment/system_links_partial.parquet`, cache `raw/taprisk_links/nldi_cache`, log
  `raw/taprisk_links/links_stdout.log`). If the host reboots, re-run `--stage links` (cached requests are free).
- App registration is self-contained in `sentinel/models/downstream/taprisk_registry.py` (do NOT edit common.py until the five pending looks are complete; then add taprisk to common.APPS + APP_SCORING_FILES before any D4 freeze).
- **Next (in order):** `build_taprisk_links.py --stage streamcat` → `--stage manifest` → write
  `docs/datasheets/taprisk_catchment.md` → `downstream_taprisk.py --reproduce` (tolerance ±0.05 vs GW sens 75.8 %
  / spec 81.5 %) → `--phase1 --task d4a_gw_nitrate` and `--task d4b_sw_dbp` → protocol iteration log.

## Test looks done (2026-11-08 20:22–22:12; one look each, no rescore; 576 registry test rows)
FREEZE files `docs/preregistration/FREEZE_downstream_{safeswim,d1n_beaches,biointegrity,musselguard,toxinwatch}.json`
(read-only), prereg archives `$SENTINEL_RUNS/downstream/<app>/prereg_archive/*.tar.gz` (sha in FREEZE + snapshot
rows), outputs `$SENTINEL_RUNS/downstream/<app>/test_look/` (TEST_LOOK / PREDICTIONS_SAVED / SCORED markers).
Labels: **D1n SOTA (national-beach track, vs re-implemented VB-MLR)**; D1 Chicago competitive; D5 competitive
(beats Hill-2017 RF both variants, Holm; not the feature-matched RFs); D6 competitive (2018 yes, 2021 no); D2
competitive. `downstream_<app>_test_results.md` + RESULTS.md 2026-11-08 entry + model cards `docs/model_cards/downstream_*.md`.
**H-G4 "beats incumbent": 1 app (D1 via D1n) — not met; D5 counting to be ruled by the lead.** Post-results
review (R10) pending. These apps' test blocks are spent (§15.5 item 1): any improved model needs a new unseen
test period (D1: 2026 qPCR reserve vs GM only; D1n/D2: 2026 season; D6: T = 2022 origin later).
**Superseded (2026-11-09 lead rule: never edit a file hashed by a FREEZE record):** common.py, wqp_track.py, testlook.py
and the five apps' modules/scripts/pins are FREEZE-hashed, so D4 registration stays in `taprisk_registry.py`
(import-only use of shared code); a D4 freeze hashes `taprisk_registry.scoring_files()`.

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.

## D4 TapRisk status (2026-11-09 evening)
Links + StreamCat done (`taprisk_catchment` manifest + datasheet). Reproduction: sens 0.813 / spec 0.815 vs published
0.758 / 0.815 (sens +0.005 outside tolerance, favourable; diagnosed in protocol §10). Phase 1 + it1_ens (validation):
the GBM beats the plain Pennino-form RF clearly (D4a both sets; D4b new states) but not the pre-registered
history-augmented RF on all sets → **competitive**; 34 configurations. Decision pending with the lead: (a) spend
the D4 test block on the phase-1 GBM (expected "competitive"), or (b) further pre-registered iterations, or (c) D7.

## 2026-11-09 late: D8 closed, D4 candidate ready
- D8: it2 decomposition not kept → final negative report (protocol §6b; RESULTS.md). No test look.
- D4: it2 done (40 configs); rule selected `stack_rf_gbm` (no qualifier; expected "competitive"). Test path
  `scripts/train/downstream_taprisk_test.py` (dry run reproduces val exactly), pins
  `docs/preregistration/downstream_taprisk_data_pins.json`, candidate record
  `docs/preregistration/FREEZE_CANDIDATE_downstream_taprisk.json` (incl. `registry_scoring_code_sha256` for D4 files
  outside common.scoring_files — verified by the test script before the claim). Awaiting independent pre-test review
  → lead FREEZE → single look. Write-ups only in `downstream_taprisk_test_results.md`.
- Next after the D4 look: D7 TroutRefuge.

## D7 TroutRefuge (started 2026-11-09; validation only; new files only)
Protocol `docs/preregistration/downstream_troutrefuge.md` (written before code; config cap 40; hardened guard spec §6;
E7 overlay §5; write-ups → `downstream_troutrefuge_test_results.md`). Split frozen `587caeca…`; power OK (val new
locations 201 / 806). Pending lead decision: MAST/MSST/MWST (project-excluded StreamCat metrics) as a comparator-only
thermal incumbent — default fallback is an observed air–water thermal rule. Next: phase-1 models, D7-only hardened guard
module `troutrefuge_guard.py`, E7 overlay feasibility (sonde sites ≤ 5 km of NRSA fish sites).
D4 candidate under independent pre-test review — do not touch D4 hashed files.
- 2026-11-09: MAST approved as comparator score only → `scripts/data/fetch_mast_d7.py` → `processed/d7_mast` (manifest +
  `docs/datasheets/d7_mast.md`, D7-only restricted use); `troutrefuge.assert_no_modelled_temperature` + tests; both thermal
  incumbents (MAST, air–water) in the claim family (Holm over 5). Add `processed/d7_mast` to the D7 data pins at candidate time.

- **Phase 1 + 1b done 2026-11-10 (validation only; 36 of 40 configurations; runs `troutrefuge/phase1`, `phase1b`).**
  Primary challenger (inner CV) = LightGBM. Val new locations (n 806, 201 pos): LightGBM 0.9515; RF occupancy incumbent
  0.9441 (Δ **+0.0074 [−0.0027, +0.0188]**, p 0.16 — tie; paired MDE 0.050); feature-matched RF 0.9475 (Δ +0.004, n.s.);
  MSST rule 0.9219 (Δ +0.030 [+0.012, +0.047]); MAST rule 0.883 (Δ +0.066); air–water rule 0.850 (Δ +0.102);
  location kNN 0.901 (Δ +0.051). Holm over six: thermal rules + simple rejected; **RF incumbent and parity RF not** →
  **"competitive" on validation; D7 does not currently count as "beats incumbent" for H-G4.**
- **Comparator correction (protocol §8, 2026-11-10):** MAST = mean *annual* stream temperature (not "August");
  the faithful thermal incumbent is MSST (mean summer). `msst_rule` added (comparator score only, same `d7_mast`
  restrictions); claim family is now six. Needs the lead's confirmation (approval named MAST only).
- **Numeric track (Jul–Aug field temperature, val new locations):** quantile LightGBM RMSE 2.96 °C, R² 0.69; MSST 3.23 /
  0.62 (paired −0.38 °C [−0.54, −0.22]); air–water 4.00 / 0.43; 90 % PI coverage 0.797 uncalibrated → 0.935 with
  pre-registered CQR (width 8.7 → 11.9 °C; slightly conservative).
- **Open:** (1) decision on the RF tie — freeze as "competitive", or pre-register a harder track (saturation directive);
  (2) `troutrefuge_guard.py` (to be derived from the D4 look module once that passes review); (3) E7 overlay feasibility.

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
- **Edited by M1:** `sentinel/models/downstream/killalert.py`. `iowa_gauges` (index-driven) now drops reserved sites and `_gauge_anoms` refuses them. **D3 KillAlert gauge candidates can change if a reserved site was among them; re-check your candidate list before any freeze.** The Platform loader `sentinel/platform/data_access.py` now hides and refuses reserved sites (daily index, hourly files, observations, site_record).

## 2026-11-10: D4 pre-test review (FAIL) corrections — awaiting delta review
- Audit of the five completed looks: exactly one look each (`docs/preregistration/downstream_test_look_audit_2026-11-09.md`).
- D4 fixes in D4-owned files only (shared testlook/common/wqp_track untouched): hardened look `taprisk_look.py`;
  test script rewritten (pre-claim checks, registry claim/seal/abort, hard-coded n_estimators, state-cluster rule, numeric
  track); C5 dated formal enforcement; C6 censored fit labels (`taprisk.build(scored_origin=)`); W3 lat/lon parity;
  closure files + AST closure test; pins regenerated (24 files, folder check); it2c re-run → rule again `stack_rf_gbm`
  (competitive expected); numeric track (UCMR4 HAA5/TOC/bromide, quantile GBM 90 % PI) validated (`numeric_val_v2`).
- Tests: `tests/test_downstream_taprisk_look.py` (P1, P4, P5, P6/P6d, P7c, P8b, wrong process, unlisted file, e2e score),
  `tests/test_downstream_taprisk_closure.py`.
- After the delta review: lead writes FREEZE (status FROZEN, authorised_by/at, candidate sha, `paths` copied from the
  candidate), prereg snapshot + archive, then one `--allow-test` in `sentinel2` with default paths.
- Then: D7 phase 1 (MAST comparator approved; numeric temperature track only if a measured temperature label fits).

## 2026-11-10 (18:30 UTC): D4 delta-review fixes done — candidate regenerated, awaiting the lead's FREEZE
- Delta review 🟡 (E1–E4, W-a–W-e) — all fixed or recorded, in D4-owned files only. 5 of the 30 hashed files changed
  vs the reviewed 11:49 candidate (`taprisk_look.py`, `downstream_taprisk_test.py`, `taprisk_numeric.py`, D4 pins,
  D4 protocol); `taprisk_models.py`, the split and all shared files unchanged. Diffs vs the reviewer's copy:
  scratchpad `ds/d4_delta_diffs/`.
- **E2:** environment pin = sha of `pip freeze --exclude-editable` (`taprisk_look.env_pin()` = `19a9cedf…`); it no
  longer moves with git HEAD.
- Candidate `FREEZE_CANDIDATE_downstream_taprisk.json` regenerated last (13:27:27 local), sha `6c07115a…`; closure hash
  `21920777…`, common hash `8a9c72ef…`, split `a4a1fc7b…`, pins `a14685a0…`, protocol `5c55bfaa…`, test script `5538cf79…`.
- **Dry run on validation is NOT byte-identical between runs** (threaded sklearn forest prediction; ≤ 2.7e-15 in the
  8 forest-derived columns, 27 other columns bit-identical, all AUROC values bit-identical; protocol last-but-one
  bullet + candidate `dry_run_reproducibility`). Lead decides: accept, or make forest prediction single-threaded
  (hashed change → delta review + two dry runs + new candidate). Do not change it without that decision.
- Tests: 83 pass across the 7 downstream guard/D4/D7 test files (20 D4 tests).
- **Caveat for the five completed apps:** their pins hold the HEAD-dependent `pip_freeze_sha256` (`c3ba836b…`); the
  shared `TL.pip_freeze_sha()` is now `75aa6588…` (only the editable-install commit id differs), so
  `TL.verify_data` refuses → a `--rescore-from-saved` on those five is refused until the lead decides how to handle
  it (the pins files are FREEZE-hashed; not editable).
- Do NOT commit, write a FREEZE file, or run `--allow-test`: the lead does the first two and authorises the third.

## 2026-11-10 (20:25 UTC): D4 single-threaded forest prediction — new candidate `2bff8c92…`; D7 transfer-track design note
Supersedes the candidate hashes and the "NOT byte-identical" bullet of the 18:30 UTC section above.
- **Lead ruling 14:25 EST:** forest *prediction* single-threaded; fitting unchanged. Done in
  `taprisk_models.BalancedRF.predict` (+4 lines) and `taprisk_numeric.fit_predict` (+1 line): `set_params(n_jobs=1)`
  before predicting. Fitting was shown not to be a source of non-identity (library diagnostic on random arrays,
  scratchpad `diag/rf_fit_determinism.py`).
- **Two dry runs on validation are byte-identical:** predictions sha `804e7a4b…764f`, results.json `c4add9e7…3c8e`,
  logs identical. Against the last threaded run: all 176 AUROC leaves bit-identical (also every AUPRC, paired
  difference, CI, p-value, Holm decision, claim flag); 9 of 1,580 leaves differ by ≤ 4.4e-16 (8 `pennino_rf_reg`
  RMSE/R² values or bounds, 1 `pennino_rf` Brier bound). Comparison: scratchpad `ds/d4_st_compare.json`.
- **Candidate regenerated last (15:20:01 local), sha `2bff8c924159…8a1d`**; closure hash `f63ed306…`, common hash
  `1623ccbd…`, split `a4a1fc7b…` (unchanged), pins `55bb6ddf…` (timestamp only; 24 file hashes unchanged), protocol
  `ae7133b4…` (+21 lines, append-only), env pin `19a9cedf…` and test script `5538cf79…` (unchanged).
  4 of 30 hashed files changed against candidate `6c07115a…` (diffs: scratchpad `ds/d4_delta_diffs_b/`); 6 of 30
  against the reviewed 11:49 candidate (`ds/d4_delta_diffs_cumulative_vs_1149/`).
- Tests: 85 pass (7 downstream guard/D4/D7 files; 22 D4 tests, two new for single-threaded prediction).
- Still: do NOT commit, write a FREEZE file, or run `--allow-test`. The lead runs the delta review, writes the FREEZE
  and authorises the one look. If a hashed file is touched again, regenerate pins, then the candidate, last.
- **D7:** MSST approved by the lead as an added comparator (score only, `d7_mast` restrictions); MAST stays; family of
  six, Holm; recorded in `downstream_troutrefuge.md` §8. **No D7 freeze candidate yet.** Counts-only design note for a
  pre-registered ecoregion-transfer track: `docs/preregistration/downstream_troutrefuge_ecoregion_transfer_design.md`
  (9 `ag_eco9` blocks; fit outside the block on earlier cycles, score the later cycle inside it; primary = pooled AUROC
  of out-of-ecoregion predictions with a within-ecoregion guard; HUC4 bootstrap; same six comparators; Bonferroni over
  the two tracks recommended). It is a proposal: nothing in the transfer track is run until the lead approves it
  (four decisions listed in its §11). Both tracks are then frozen together and looked at in one event.
