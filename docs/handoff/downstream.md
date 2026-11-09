# Handoff — Downstream workstream (Goal 4: D1, D2, D3, D4, D5, D6, D8 + national tracks D1n, D2n) — 2026-11-08 (late)

## CURRENT STATE (read first; supersedes the dated sections below)
- **Test blocks SPENT (single looks 2026-11-08 20:22–22:12; never re-use them for a claim, §15.5 item 1):**
  D1 Chicago D1b 2024–25, D1n national beaches 2021–25, D5 NRSA 2023-24, D6 origins 2018 + 2021, D2 WA 2022–26.
  FREEZE files + prereg archives + `test_look/` markers exist; protocol result sections and RESULTS.md hold the
  outcomes. Labels (post-results review): D1n beats its re-implemented VB incumbent and the strongest simple
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
competitive. Protocol result sections + RESULTS.md 2026-11-08 entry + model cards `docs/model_cards/downstream_*.md`.
**H-G4 "beats incumbent": 1 app (D1 via D1n) — not met; D5 counting to be ruled by the lead.** Post-results
review (R10) pending. These apps' test blocks are spent (§15.5 item 1): any improved model needs a new unseen
test period (D1: 2026 qPCR reserve vs GM only; D1n/D2: 2026 season; D6: T = 2022 origin later).
Pending lead-approved change (deferred until the post-results review): add taprisk to common.APPS and
taprisk_models.py to APP_SCORING_FILES["taprisk"] before any D4 freeze.

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.
