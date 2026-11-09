# M4 ChemTox — round-4 review request (prospective pipeline), 2026-11-08

**State:** no prospective row read, fitted or scored. Data sha256 `3b2fc439…dc54d9` unchanged.
`--i-have-approval` has **never been used for a non-dry run**; it was used for three `--dry-run`
pre-flights on 2026-10-24 (22:28, 22:50, 23:00; TRAINING_LOG lines 4667/5044/5267) against scratch
selections, which fit on training rows only, read no target value and wrote no registry row (W4).
The test suite also passes the flag on a target-stripped copy. Lead decision: neither the reviewer nor M4 runs it before round 4 passes; the lead then authorises M4 explicitly to run the pre-flight and the single pass.

## Changes since round 3 (FAIL on N5, N6)

Fixed after round 3, before the pause (2026-10-24/25), not yet reviewed:
1. **N5** — ensembles carry frozen `oof_sources` keyed by frozen name; file names match the CV
   runner; a missing component or out-of-fold file is a hard error (freeze and scorer).
2. **N6** — `__published_reference__` = best masked CV of the four published runs, written inside the
   hashed block (`rf_published_mwlogp`, 0.9909); published runs live in their own `_cfg-pub_*`
   directories, so every search keeps exactly 32 configurations.
3. Per-seed ensembles (§4.2 `all_seeds_better` computable for ensembles), `extra_weight` applied in
   the scorer, dry run returns 8 if a pre-flight fit fails. `__allow_unmasked__` is now recorded in
   the frozen block (round-3 🟡).

New on 2026-11-08 (`scripts/eval/chemtox_cycle_select.py`, sha256[:16] `f5f70abc16821d19`; the
round-3 version is backed up in the M4 scratchpad):
4. **Tuning parity (lead decision).** Each training-data variant is now its own frozen entry
   (`lightgbm` / `lightgbm_aug`, `gpboost` / `gpboost_multitaxa`). Before this, the freeze took the
   best over variants, which gave those families 64 configurations against the comparators' 32.
   As a result, summary, `oof`, frozen and registry entries are now keyed by the frozen name, and
   `oof_source()` finds the file through the entry's `family`. **Please re-check the N5 path with a
   renamed member** (`gpboost_multitaxa` is in both ensembles).
5. **`--explicit FAMILY=TAG[,TAG]`**, so `chemprop` can enter selection from its masked
   `_dm_cfg-*` re-runs. Without it, chemprop (a candidate the plan names in §3) would have been
   left out of the freeze without any error. Two related fixes: `is_masked_dir()` now accepts
   `_dm_cfg-` (the N4 guard used to reject these directories), and `variant_extra()` replaces a
   parser that would have produced `extra="none_dm_cfg-…"`.
6. Comparator registry rows are now logged with `is_baseline=True`.
7. `chemeleon_tabpfn` is being re-run with `--accelerator gpu` on GPU 4. Both failed attempts are
   archived as `.txt` files next to `RERUN_NOTE.txt` (CPU limit on 10-25; CUDA out-of-memory on
   11-08, from a non-SENTINEL process holding GPU 4).
8. Tests: 28/28 green in `tests/test_chemtox_prospective.py` (new: variant/masking parsing).

## Masked-CV evidence for the freeze (CV only)

**FINALIZED 2026-11-08 14:20 EST.** Frozen-configuration hash (`--expect-config-hash`)
**`630b43e905d9b852`**; `selection.json` sha256
`bf2ed61c971fc1926db4d668c73c16c37752fc27a665e7208f19e2a356237328`; stage 2 appended to
`chemtox_prospective_plan.md` (from line ~400); 14 registry entries (`adore_chemtox_cycle2`,
2026-11-08). Ensemble members by the fixed rule: `catboost` 0.9480, `chemeleon_gpboost` 0.9504,
`chemprop` 0.9527, `chemeleon_lightgbm` 0.9593, `gpboost_multitaxa` 0.9669 (bar: re-tuned
`xgboost` 0.9733). `chemeleon_tabpfn` completed: 32/32 screened (config 0 lost to an external CUDA
OOM and re-screened alone, 1.3485, outside the top 5, so no extra finalist), best 5-fold 1.0427 —
frozen as a candidate, not an ensemble member.
`ensemble_mean` 0.9164, paired out-of-fold ΔRMSE vs xgboost −0.062 [−0.096, −0.031]. This is
optimistic, because the members were chosen on the same CV. No single model's CI excludes 0.

## Observations for the reviewer

* `chemeleon_gpboost`: best screen 0.9194 (folds 0–2, `num_boost_round` ≤ 300) against 0.9504 for
  the best 5-fold finalist. The screen ranks this family poorly. Per the lead, this is **not**
  re-screened: no family gets more than 32 configurations.
* `gpboost` (no extra rows): 2 of 5 finalists failed (GPBoost CG solver returned NaN). The family is
  frozen from 3 completed finalists. It is weak anyway (1.0158).
* Every family's winner uses `chemical=mw_logp` (§4.3).
* Holm family: 11 candidates (9 single + 2 ensembles), for each of the two reference comparisons.
* The §6 package hash still goes stale as other workstreams edit `sentinel/` (open 🟡). The
  authoritative value is the one each registry entry records.

## Please verify

The selection script diff against the backup; `nnls_weights()` / `build_training_frame()` in
`scripts/eval/chemtox_score_prospective.py` against the new frozen keys and `extra` values
(`ecotox`, `adore_taxa_all`); the stage-2 block against `selection.json` and its sha256; that every
`oof_sources` path is a masked 5-fold file.

**Deferred, not in this freeze:** species embeddings (separate pre-registered stage, lead decision).
Saturation (§15.5 items 5–7): **not saturated** (CV R² ≈ 0.68, RMSE 0.92–0.97 against a replicate
floor of 0.45). The prospective set is already the harder track. Any further track is proposed
after round 4.
