# Handoff — M2 SENTINEL-Orbit (updated 2026-11-09 15:30)

## 2026-11-09 session summary (read first)
- **WQP 2015–2023 build had died** (09 Nov 10:37: a read timeout made it exit with code 3 so a restart loop would
  resume it, but it was launched without a loop). Relaunched by `logs/wqp_hist_loop.sh` (loop PID **1419331**,
  python 1419333; resume-safe; touches `wqp_hist.done`). At 12:54 it had 35,052 scenes left at about 1,100/h, so
  it should finish about **2026-11-10 21:00**. chain6 (PID 912737) is still waiting on it.
- **chain7** (`logs/chain7_freeze_v5.sh`, PID **1500059**, CPU only) waits for `chain6.done`. Then it:
  - selects the final configuration by the pre-registered rule and writes `final_selection_dev_v5.json`;
  - runs the dev_v5 event track (main + track U);
  - runs the val dry run of the single-look scorer;
  - overwrites `docs/preregistration/orbit_freeze_candidate.json` (dev_v5, with track U and data pins);
  - touches `chain7.done`.
  It never writes FREEZE_orbit.json.
- **Event track (B output) added**: protocol v0.4 §12, adapter `sentinel/events/orbit.py`, dev script
  `scripts/eval/events_orbit.py`, config `configs/events/orbit.yaml`. These files are outside the orbit globs and are
  added deliberately to the code hash (`guard.EVENT_CODE`). The adapter pins `sentinel/eval/events.py` v1.0
  (sha256 `4279d242…`). Labels are computed from linear values.
- **Single-look enforcement**: new `sentinel/models/orbit/testlook.py`, rewritten `scripts/eval/orbit_test.py`, and
  `guard.py` changes (protocol §9b):
  - only FREEZE_orbit.json, with status FROZEN and the lead's `authorised_by`/`authorised`/`candidate_sha256`, unlocks
    test;
  - registry refusal;
  - O_EXCL claim marker + `look_claimed` row;
  - protected rows only while a claim is open;
  - sealed predictions with SHA256SUMS;
  - SCORED marker;
  - content-hash `data_pins`;
  - `--rescore-from-saved`.
  Tests: `tests/test_orbit_testlook.py` (7) and `tests/test_orbit.py` pass.
- **Dry run on dev_v4**: `test_looks_dryrun/satwq_v4_noaug/M2_test`, registry `orbit_dryrun_dev_v4`. It reproduces
  orbit_compare (chl-a 48.26 vs 48.25) and the event metrics.
- **Interim candidate** `orbit_freeze_candidate.json` (dev_v4, no track U) was written for review of its structure.
  chain7 replaces it.
- **Lead step at freeze** (see `orbit_prepare_freeze.py` docstring): copy the candidate to FREEZE_orbit.json, set
  `"status": "FROZEN"`, and add `authorised_by`, `authorised` and `candidate_sha256`. Change no other key.

### Event-track validation (dev_v4, `orbit_events_val_dev_v4`, development only)
E4a on valxval_lake: 309 positives, 90 lakes. Water-body cross-fitted calibration.

| Score | AUPRC | BSS vs climatology |
|---|---|---|
| derived | 0.857 [0.806, 0.916] | 0.604 [0.547, 0.689] |
| direct | 0.848 | 0.590 |
| MDN_B P | 0.837 | 0.566 |
| GBM-direct | 0.797 | 0.496 |
| S2 CI analogue | 0.648 | 0.309 |
| climatology | 0.269 | — |

- **ΔBrier, derived − comparator**: CI < 0 against every comparator, including direct (−0.0030 [−0.0047, −0.0011])
  and MDN_B (−0.0081 [−0.0151, −0.0023]).
- **ΔAUPRC vs MDN_B**: +0.021 [−0.008, +0.059]. The CI includes 0.
- **Tiers**:
  - at 12 µg/L, results are similar;
  - at ≥ 50 µg/L (75 positives), derived ties direct and GBM-direct.
- **E3**: 0 val positives, so descriptive only.

### Open risk for H1 (dev_v4 val)
The ensemble beats MDN_B by 4.4 MdSA points [−13.1, +0.2]. Individual seeds fail §15.1(iii) against MDN_B (52.6):

| Seed | MdSA |
|---|---|
| s0 | 53.6 |
| s1 | 56.5 |
| s2 | 49.2 |

dev_v5 (5 seeds, WQP 2015–23 rows) is the next check.


## Reconciliation after the 2026-11-04/06 reboots
- **National WQP S2 rebuild for 2024–2025** finished on 2026-10-25 at 11:11 (manifest `s2_wqp_matchups`).
  - Coverage: 68,445 candidates and 37,231 chips.
  - Same-day QC-pass samples: chl-a 14,195; Secchi 14,326; turbidity 9,348; TSS 5,499.
- **chain4** finished: `chain4.done`, registry experiment `orbit_val_dev_v3_fit-recent`.
- Nothing else was missing.
- The **AquaCorr chip extraction** finished at 120,000 requests, with 117,889 chips ok (manifest
  `s2_chips_aquacorr`, 2026-10-25 13:39).

## Running jobs (started 2026-11-08)
| PID | What | Output / flag |
|---|---|---|
| 637422 (DIED 11-09 10:37; relaunched as loop 1419331) | National WQP × S2 build for **2015–2023**: `build_s2_wqp_matchups.py --start 2015-06-23 --end 2023-12-31 --window 0 --max-chips 150000 --read-workers 4 --scene-workers 2 --out-name s2_wqp_matchups_hist` (niced) | Log `$SENTINEL_RUNS/orbit/logs/wqp_hist_2015_2023.log`; tables `processed/s2_wqp_matchups_hist/`, chips appended to `processed/s2_chips/wqp` (≤ ~5 GB); flag `logs/wqp_hist.done`. Estimate 1.5–2 days (3,114 planning dates; the 2024–25 build took 10.7 h for 2 years) |
| 638293 (finished 14:43) | **chain5** (`logs/chain5_dev_v4.sh`, GPU 1): runs `baselines_dev_v4`, then `satwq_v4_noaug_s0..2`, then compare plain and +gbm (registered, `orbit_val_dev_v4`), then the power analysis | `logs/chain5.done` |
| 912737 | **chain6** (`logs/chain6_dev_v5.sh`, GPU 1). It waits for `wqp_hist.done` and `chain5.done`, then runs: the WQP 2026-01..09 Test-B matchup build (`--out-name s2_wqp_matchups_2026`, flag `wqp_2026.done`); build dev_v5; track counts; `baselines_dev_v5`; `satwq_v5_{noaug,base}_s0..4`; compare plain and +gbm for both; power (headline and track U); track-U baselines and SatWQ refits | `logs/chain6.done` |

If the hist build dies, chain6 waits forever. In that case kill PID 912737, fix the build and relaunch both.

## dev_v4 validation results (development, not claims; registry `orbit_val_dev_v4`, `orbit_power_dev_v4`)
- `satwq_v4_noaug` (3 seeds), chl-a on valxval_lake (1,157 rows, 89 blocks). Differences are MdSA points;
  negative means SatWQ-Net is better; CIs are role-block bootstrap 95%.

  | Comparator | Difference | 95% CI |
  |---|---:|---|
  | MDN_B (val-selected) | −4.4 | [−13.2, 0.2] |
  | GBM imagery + context (val-selected) | −8.0 | [−17.3, 0.7] |
  | location kNN | −78 | CI excludes 0 |

- SatWQ-Net MdSA is 48.3%, vs 52.6% for MDN_B and 56.3% for GBM.
- On valxany_lake (3,546 rows, 170 blocks), both CIs exclude 0: −7.5 vs MDN_B and −9.6 vs GBM.
- Secchi on valxval_lake: −4.7 [−8.6, −0.7] vs MDN_B and −5.1 [−12.7, 0.9] vs GBM.
- The +gbm blend is weaker on dev_v4: −1.2 vs MDN_B and −4.8 vs GBM.
- **Power (W4).** The headline test cell holds 669 chl-a rows in 87 blocks. The expected test SE is about
  3.5 for MDN_B and 5.0 for GBM, so the MDE at 80% power is about 10–14 MdSA points.
  - Power at the val gains: 0.22 vs MDN_B and 0.35 vs GBM for chl-a. H1 is **under-powered**.
  - Secchi vs MDN_B: power 0.91.
  - Track U: chl-a MDE about 12–15 points.

## Changes this session (all before any test scoring)
- `build_s2_wqp_matchups.py`: new `--out-name` option. The 2015–2023 tables go next to the 2024–25 table,
  so the Fuse consumers of `s2_wqp_matchups` are not disturbed.
- `data.py`:
  - `WQP_MATCHUP_SETS` reads both WQP sets.
  - `protected_cell_counts` writes counts only for protected cells (split report).
  - **Turbidity unit rule** (`apply_turbidity_unit_policy`): the target = NTU family + AquaCorr
    unspecified; FNU and mixed-unit values go to `turbidity_fnu`, which is never fitted; `turbidity_all_units`
    keeps the pooled value. A test was added.
- `orbit_compare.py` writes `val_row_preds.parquet`.
- New `scripts/train/orbit_power.py` (power analysis, W4).
- Harder tracks (RESEARCH_PLAN §15.5 amendment of 2026-11-08):
  - `sentinel/models/orbit/track_u.py`, the `track_u` fit scheme and `scripts/train/orbit_tracks.py`;
  - frozen rule file `docs/preregistration/orbit_track_u.json`, sha256
    `953948b7f0251057728adfc4d9a75065e1d0324c9ec3536eb6dae99ae957b2cf`;
  - counts in `orbit_tracks_counts_dev_v4.json`.
- `orbit_test.py`:
  - `--version` for the dry run;
  - blend support (FREEZE `blend_with`);
  - track-E strata;
  - `--track U`;
  - turbidity strata by unit class.
- New `scripts/eval/orbit_prepare_freeze.py`. It writes `FREEZE_orbit_CANDIDATE.json`, never
  `FREEZE_orbit.json`.
- Protocol is now **v0.3** (§2.1 unit rule, §2.3 dev_v4/dev_v5, §5a power, §11 tracks E/U/L). Model card:
  turbidity unit note.
- `baselines.py`: the GBM thread count comes from `ORBIT_N_THREADS`.

## Next steps (after chain6)
1. Read the `compare_satwq_v5_*` paired tables. Select the final configuration by the pre-registered rule:
   lowest val MdSA (chl-a, valxval_lake) among the configurations scored on val (§6). Report the
   configurations-tried count from `configs_tried.jsonl`.
2. Run the dry run, `orbit_test.py --dry-run-dev --version dev_v5 --satwq … --baselines baselines_dev_v5
   --comparators-json … [--blend-with gbm_imagery_context]`, and check that it reproduces the compare numbers.
3. (Done 2026-11-09) Event track: the bloom cut, calibrators and FPR cuts are frozen in the events artifact.
4. (Automated in chain7) Write the candidate with `orbit_prepare_freeze.py`, including the track-U artefacts. Then request the
   lead's review and authorisation. **Do not score test.**
5. If a retrained teacher is wanted for M7, write a **new versioned manifest** (`orbit_teacher_v2.json`) and
   tell the lead. `orbit_teacher.json` stays unchanged.
6. Track L (Landsat 8/9) is proposed only: a chip extraction of about 1–3 GB.

## Lead decisions (2026-11-08)
- chl-a stays the primary target and Secchi stays secondary; the endpoint is not switched.
- The Test-B cell and the pooled analysis were declared before any test look (protocol §9a).
  `orbit_test.py` scores the headline, `testB_2026` and `pooled_2024_2026` cells.
- The dev_v3 `fitted_baselines.pkl` files were deleted with approval (about 800 MB freed); the predictions
  are kept.

- **GPU budget is now GPUs 0 and 1, shared by everyone** (2026-11-08 16:00). `PL.setup_gpu` now goes
  through `sentinel.core.gpu.acquire_gpu`: it prefers GPU 1, honours the cap file, and lets the helper set
  `CUDA_VISIBLE_DEVICES`.
  - Every chain6 GPU step is its own process and stays within 2 h: one seed is about 15–45 min. The baseline
    runs (MDN grid + GBM) now run on **CPU** (6 threads, about 2–3 h each), so no GPU is held for them. The lock is released between steps.
  - Compare, power and track counts run on CPU.
  - Before chain6 reaches its GPU steps (about 2 days from now), check whether test/freeze-critical jobs
    from other workstreams are waiting. If they are, pause chain6 by PID (`kill -STOP`, later `kill -CONT`).

## Open issues
- `fitted_baselines.pkl` is about 400 MB per baseline run (dev_v4, dev_v5, track U).
- `orbit_test.py` was rewritten with single-look enforcement (2026-11-09). An independent review is still recommended
  before the freeze.
- AquaCorr turbidity, TSS and Secchi units are unstated (documented).

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.
