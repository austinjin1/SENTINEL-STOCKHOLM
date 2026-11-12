# Handoff — SENTINEL-Fuse (Goal 2), updated 2026-11-08

## State (2026-11-08)
- **Tier D v2 built** (2026-11-08): 105,117 rows, 34 sites, now with `hydro` (EOS single-HOR anomaly + log Q),
  `macroinv` (830 events), `periphyton` (718 events). 19,398 rows have all 8 non-hub modalities. v1 table backup:
  scratchpad `fuse/daily_v1.parquet` (delete when no longer needed). Datasheet updated.
- **Protocol v1.1** (`docs/preregistration/fuse_protocol.md`): decision rules (C2), hard strata §8.1 (lead directive),
  P3 feasibility §8.2 (all 8 modalities at all 8 test sites), saturation note §8.3 (Tier W-P1 track proposed, not frozen),
  anchored comparators §6, stats §10 (site / 20-km-block clusters, sqrt(G/(G−1)) factor, LODO), deviations §11.
- **Review section appended** to REVIEW_REPORT_PRE_TRAINING.md ("Fuse"). Tests: `tests/test_fuse.py` 26/26.
- Registry: 24 mislabelled smoke entries superseded (split `smoke`).
- Splits verified by `scripts/data/make_fuse_splits.py check` (both byte-identical).

## Stage-0 results so far (dev CV, v1 table, run `stage0_d_p1_v2`)
Persistence beat every level GBM at h7 for DO (0.480 vs best 0.561) and nitrate (0.165 vs 0.377); concat ≈ best single for
turbidity/chl-a. → persistence-anchored `@a` comparators + `--anchored` Fuse option added.

## Stage-0 v3 (Tier D v2, anchored comparators; run `stage0_d_p1_v3`, registry `fuse_stage0_d` / `val_cv_lodgo5`)
Dev-CV CRPS (model space): DO single:sonde@a 0.461, concat@a 0.464, persistence 0.480; turbidity concat@a 0.452 vs
sonde@a 0.462; chl-a concat@a 0.462 vs 0.469; nitrate concat@a 0.184 vs nitrate@a 0.196. Paired skill concat@a vs best
single: DO −0.006 [−0.024, 0.014], turb +0.023 [0.001, 0.050], chl +0.016 [0.003, 0.031], nitrate +0.064 [0.037, 0.100];
vs persistence +0.04…+0.13. **Bar for Fuse = concat@a / single:<own>@a.** New modalities alone (single:hydro/macroinv/
periphyton) ≈ hub for these sensor targets (expected; P3 will tell whether they add inside the fusion).

## Jobs launched 2026-11-08 (check outputs, relaunch only what is missing)
- chain `scratchpad/fuse/chain_stage0_v3.sh`: `fuse_stage0.py --tier d --targets p1 --anchored` → run `stage0_d_p1_v3`,
  then `--targets lab --anchored` → `stage0_d_lab_v3`; flag `scratchpad/fuse/stage0_v3.done`.
- chain `scratchpad/fuse/chain_c.sh` (waits for that flag): Tier C rebuild (W3/W4) → split check → Stage-0 Tier C
  `gt8total` block20 with and without phyto (`stage0_c_gt8total_v1`, `stage0_c_gt8total_nophyto_v1`); flag `chain_c.done`.
- **GPU policy now: only GPUs 0/1, shared, ≤ 2 h per job, freeze-critical jobs first; prefer CPU.** The GPU-4 chain
  (`chain_gpu4_rest.sh`, TabPFN + seeds 1–4) was stopped by PID before TabPFN started. Seed 0 `fuse_d_v2a_s0` finished on
  GPU 4 (~25 min; early stopping at epoch 1–2 of 30 in every fold — noisy inner validation, train/val divergence).
- **Config #1 result (`fuse_d_v2a_s0`, dev CV, 1 seed; `scratchpad/fuse/compare_s0.log`, registry `fuse_compare_d`):**
  skill vs best single / vs GBM concat@a: DO −0.039 / −0.033, turbidity −0.015 / −0.039, nitrate −0.091 / −0.166, chl-a
  −0.069 / −0.087, TN −0.064 / −0.072, TP −0.126 / −0.129, DOC −0.014 / −0.017 (CIs exclude 0 for nitrate, chl-a, TP and
  turbidity-vs-concat) → **negative**. Beats stack, UME, hub; vs persistence +0.05…+0.09 except nitrate (−0.059) and TN.
  P3 preview (dev, inference LOMO, ΔCRPS/CRPS_hub pooled over 7 targets): sonde +0.251, chem +0.179, nitrate +0.170 (CIs > 0);
  hydro, temp, micro, macroinv ≈ 0; periphyton −0.004 (CI < 0) → 3 of 8.
- **Config #2 running on CPU** (PID 913187, 6 threads): `fuse_train.py --device cpu --seeds 0 --lomo --anchored --d 64
  --latents 16 --dropout 0.3 --wd 0.01 --lr 3e-4 --epochs 20 --patience 8 --run-name fuse_d_v2b_s0` (~146 s/epoch).
  Then compare: `fuse_compare.py --fuse-run fuse_d_v2b_s0 --stage0-run stage0_d_p1_v3,stage0_d_lab_v3 --targets
  y__do_mgl__h7,y__turb_fnu__h7,y__no3n_mgl__h7,y__chl_ugl__h7,ylab__tn_mgl,ylab__tp_mgl,ylab__doc_mgl`.
- `scratchpad/fuse/chain_wp1.sh` (PID 915082) waits for config #2, then builds the W-P1 table
  (`build_fuse_tiers.py wp1` → `processed/fuse_tier_wp1`; smoke on 6 sites passed); write its datasheet afterwards.
- Configurations tried (Fuse deep model, dev CV): 2 (v2a, v2b). TabPFN still to run (GPU 0/1 ≤ 2 h when free of
  freeze-critical jobs; `fuse_tabpfn.py --device cuda --max-rows 6000`, chunk by target if needed).
- Tier C rebuilt (73,160 rows; split check OK). Stage-0 Tier C dev CV (20-km folds), y_gt8_total: concat AUPRC 0.428
  [0.366, 0.484] (AUROC 0.925) vs best single (history) 0.397, prevalence 0.039; phyto on/off makes no difference (0.429).
- Deep model command used by the chain:
  `SENTINEL_GPUS=4 python -m sentinel.core.gpu --name fuse_train --min-free-mb 3000 -- python -B scripts/train/fuse_train.py --seeds 0 --lomo --anchored --run-name fuse_d_v2a`
  then `python -B scripts/eval/fuse_compare.py --fuse-run fuse_d_v2a --stage0-run stage0_d_p1_v3`
  (and `--stage0-run stage0_d_lab_v3 --targets ylab__tn_mgl,ylab__tp_mgl,ylab__doc_mgl`).

## Next configurations (declared in protocol §7.1 before any code/fit)
- **#3 Stage-2 pretraining** on W-P1 dev-subset rows (supervised DO/SC/turb/NO3 h7/h14 + masked-modality reconstruction +
  hub-contrastive InfoNCE τ 0.1), then fine-tune on Tier D (shared core/cross-attn/tokenizer groups/queries initialised;
  chem/micro/macroinv/periphyton fresh). Needs: W-P1 table (queued), a pretraining script (`scripts/train/fuse_pretrain.py`)
  and `--init-from` in `fuse_train.py` (match parameters by name + shape). GPU 0/1 ≤ 2 h chunks or CPU.
- **#4 residual on anchored GBM concat**: inner-cross-fitted `concat@a` μ as the per-row base (replaces the anchor in
  `FuseModel.forward`); LOMO removes m from both parts (refit concat@a without m per fold).

## W-P1 harder track (approved by lead; protocol §12)
Split frozen `docs/preregistration/fuse_tier_wp1_split.json` sha `b9e0d5c4…` (M1 roles; dev 2,114 sites / 78 basin groups,
test 564 / 42). Power (availability only, `results/fuse/wp1_power_check.json`): Test-W-A DO 182 sites, SC 237, turbidity 64,
nitrate 33; chl-a 4 → not evaluated. Next: write `sentinel/data/fuse_tier_wp1.py` + datasheet + manifest (reuse Tier D
feature code and Tier W lab history), then Stage-0 on its dev folds.

## Open
TabPFN GPU run (queued; script `scripts/train/fuse_tabpfn.py` uses the shared scikit-learn shim of `chemtox/tabpfn_model.py`), tuning-parity random search,
P4 per-pattern Stage-0 (`--pattern`), events Stage-0, M3 fold-aware OOF (request below), Holm p-values + `fuse_freeze.json` in
a final scoring script, Tier W-P1 harder track (protocol section + split sha + power check before any selection),
docs/model_cards/fuse.md.

## Requests to other workstreams
- M3 Microbe: fold-aware OOF with the Fuse dev fold map, excluding Fuse test domains D03/D07/D13/D14 (protocol §5).
- Orbit/data: national `s2_wqp_matchups` and an S2 extraction at NEON sites.

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.


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
- **Edited by M1:** `sentinel/data/fuse_tier_w.py` (sonde features never read for reserved sites), `sentinel/data/fuse_tier_wp1.py` (reserved sites dropped from the site list) and `scripts/eval/fuse_wp1_power.py` (reserved sites skipped). **Tier W / WP1 rebuilds can differ if a reserved site was present; regenerate and re-check before any Fuse freeze.** NEON tiers (`neon_sonde_*`, NEON site codes) cannot contain reserved USGS sites and were not edited.
