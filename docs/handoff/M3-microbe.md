# Handoff — M3 SENTINEL-Microbe (updated 2026-11-08 ~20:00 EST, resumed session)

## Running / queued (CPU, one job at a time, ≤ 6 threads)
Queue script `…/scratchpad/m3/queue4.sh` (PID 1073539; status file `queue4.status` next to it). The session scratchpad
is not durable: if it is gone, re-launch the remaining `microbe_cv.py` / `microbe_controls.py` commands listed below.
1. `cv_rel21_tables` (PID 1053775): `--schemes release_002_to_001`, models context_rf global_mean context_ridge
   ridge_clr ridge_v4clr xgb_clr xgb_v4clr rf_clr (cross-release transfer = alternative primary route).
2. `cv_lodo_ctx`: `xgb_v4clr_context xgb_clr_context --schemes lodo` (cycle 4, §10.4).
3. `cv_lodo_tuned`: `xgb_clr_tuned rf_clr_tuned --schemes lodo` (W6 tuning parity, ~10 h).
4. `cv_rel12_tables`: the same models as in item 1, run on `release_001_to_002`.
5. `ctl_perm_xgbv4`: `microbe_controls.py --permutation --perm-models xgb_v4clr --n-perm 3`.
6. `ctl_perm_rf`: `--perm-models rf_clr --n-perm 20` (~14 h).
Logs `$SENTINEL_RUNS/microbe/logs/<name>.log`, outputs `$SENTINEL_RUNS/microbe/cv/<scheme>/`.

GPU: the user directive restricts SENTINEL to GPUs 0 and 1 only. Jobs must be ≤ 2 h, and test-critical jobs from
other workstreams go first. Deferred: `seqset_id` (and optionally `seqset_dnabert`) on `lodo_x001`, about 20 min each:
`python -m sentinel.core.gpu --name microbe_x001_seqset_id --min-free-mb 3500 -- python -B scripts/train/microbe_cv.py --models seqset_id --schemes lodo_x001 --device cuda --iteration cycle2`.

## Where things stand (dev CV only; holdout untouched)
- Protocol log: `docs/preregistration/microbe_protocol.md` §10.2 (phase-1 completion, fusion control, cycle 2),
  §10.3 (cycle 3: V4 units), §10.4 (cycle 4: +context). Every candidate was declared before it ran.
- Mean pooled dev-LODO R² over the 8 targets:

  | Model | R² |
  |---|---|
  | V4-XGB | **0.314** |
  | V4-Ridge | 0.291 |
  | TabPFN | 0.291 |
  | XGB[genus-CLR] | 0.289 |
  | Ridge[genus-CLR] | 0.284 |
  | SeqClust-DNABERT | 0.256 |
  | RF[genus-CLR] | 0.252 |
  | SeqSet-kmer | 0.239 |

  Fusions:

  | Model | R² |
  |---|---|
  | Ens[V4-Ridge + V4-XGB] | **0.325** |
  | Ens[Ridge + XGB] (baseline) | 0.307 |
  | Ens[SeqSet-kmer + Ridge] | 0.304 |
- Enlarged track `lodo_x001` (adds .001 2017–18 events to training; same test rows; split sha256 `a6141698…`):
  - Genus tables gain slightly. Whole-amplicon DNA models lose: SeqClust 0.256 → 0.205, SeqSet-kmer 0.239 → 0.201.
  - Diagnosis: frozen DNABERT-S and k-mer embeddings separate the V3V4 and V4V5 regions. Cross-release read-mass
    overlap is 0.04 for DNABERT-S and 0.00 for k-mers, against 0.27 for genus.
  - Fix: in-silico V4 window (`dna.v4_window`). Cross-release overlap rises to 0.337.
- H-M3 dev analogue (paired site- and domain-unit bootstrap, Holm over 8 targets; preview B = 1000, not logged yet):
  - V4-XGB passes for SC only. DOC vs tier-0 just misses (+0.12, p = 0.054 / 0.062).
  - The bar of ≥ 2 targets is **not met yet**.
- Power: the minimum detectable ΔR² (80% power, Holm worst case) is 0.10–0.20 per target on dev LODO
  (`microbe_split_lodo_x001_report.json`). The 214-event 2024 holdout has less power.
- `tier0_best` (§10 amendment) is implemented in `microbe_compare.py` (`--refs tier0_best … --pass-refs …`).
- Data: the dry rebuild of `microbe_wq` against the 2026-10-25 upstream refresh is identical, so the frozen split
  stays valid. The upstream chemistry per-bottle parent-ID issue and its workaround are documented in
  `docs/datasheets/microbe_wq.md`.

## Next steps
1. When `cv_lodo_ctx` and `cv_rel21_tables` finish, run the logged comparison:
   `microbe_compare.py --schemes lodo lodo_x001 release_002_to_001 --refs tier0_best rf_clr xgb_clr ridge_clr tabpfn_clr ens_ridge_clr+xgb_clr rf_clr_context xgb_clr_context [xgb_clr_tuned rf_clr_tuned] --pass-refs …`.
   Logging requires the final ref set, so run it once.
2. Apply §6 selection over the full DNA-encoded candidate set: `microbe_select.py --candidates …`.
3. If a candidate passes, prepare the freeze candidate (configs/hashes, `scripts/eval/microbe_final.py` refit on
   dev + optional .001 rows) and ask the lead for authorisation. Never score the 2024 holdout without it.
4. Otherwise, run the declared next steps in §10.3: V4-window k-mer tables for SeqSet, and EMP V4 unlabeled pretraining
   (size estimate to the lead first).

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.
