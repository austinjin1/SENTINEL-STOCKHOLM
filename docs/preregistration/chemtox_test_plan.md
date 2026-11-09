# Pre-registration — SENTINEL-ChemTox (M4) test scoring

**Status: DRAFT — awaiting the coordinator's approval. No SENTINEL model has been scored on any
ADORE test row.** The only test numbers that exist in the registry are the reproduction of the
published baseline (experiment `adore_published_baseline`, `is_baseline=True`), which the
coordinator accepted.

This document freezes, *before* the test rows are read: which configurations will be scored, on
which rows, with which metrics and comparisons, and what counts as success. It is hashed; the
hashes are listed in §6 and recomputed by the scoring script, which refuses to run if they differ.

## 1. Data and splits (unchanged from phase 1)

* Benchmark: ADORE (Schür et al. 2023, DOI 10.25678/0008C9, CC BY 4.0), converted release at
  `$SENTINEL_DATA/raw/adore/release`, manifests `adore_raw` + `adore`.
* Primary endpoint: challenge **`t-F2F`**, split column **`split_occurrence`**, target
  `result_conc1_mean_mol_log` (log10 mol/L), official **test** rows (n = 4133, 317 chemicals,
  92 taxa; train–test chemical overlap **0**, asserted in code).
* Secondary endpoints, scored in the same single pass: `t-F2F` × `split_scaffold-murcko`,
  `t-C2C` × `split_occurrence`, `a-CA2F-diff` × `split_occurrence` (the cross-taxa challenge,
  for which no published regression baseline exists).
* `a-CA2F-same` is **excluded** from test claims: its train and test partitions share chemicals by
  construction (991 under `split_occurrence`), so it cannot support a chemical-generalisation claim.

## 2. Frozen configurations (one per model)

Selected by mean RMSE over ADORE's official CV folds 0–4 only; the search budget is identical for
every family (32 configurations each; the published baseline's grid was 32). The table is written
by `scripts/eval/chemtox_select.py` into `$SENTINEL_RUNS/chemtox/select/<challenge>_<split>/selection.json`
and copied here verbatim with its hash:

| Model | Frozen configuration | CV RMSE (folds 0–4) | n configurations |
|---|---|---|---|
| Chemprop v2 D-MPNN | no fingerprint + published 37; `message_hidden` 200, `depth` 3, `ffn_hidden` 300, `ffn_layers` 2, `dropout` 0.0, `max_lr` 1e-3, `batch_size` 128, `max_epochs` 30, `species_dim` 16, `use_conditioning` False, `multitask` False, `n_seeds` 3, `init` none | **0.9575** | 32 |
| Reproduced published baseline — RF + MACCS | MACCS + published 37; 300 trees, `max_depth` 20, `max_features` 0.3 | 0.9605 | 8 |
| Reproduced published baseline — XGBoost + MACCS | MACCS + published 37; 800 rounds, `eta` 0.03, `max_depth` 6, `min_child_weight` 1, `subsample` 0.8 | 0.9929 | 32 |
| Frozen ChemBERTa-2-MTR + XGBoost | no fingerprint + published 37; CLS pooling, 1500 rounds, `eta` 0.02, `max_depth` 6, `subsample` 0.8 | 0.9905 | 32 |
| TabPFN v2 | Morgan + published 37 + RDKit descriptors; `n_estimators` 8, `max_train_rows` 9000, `max_features` 100, row subsampling | 1.0305 | 32 |
| **Ensemble — mean** of the five above | unweighted mean of the five components' predictions | 0.9406 | components' budgets |
| **Ensemble — NNLS stack** of the five above | non-negative least squares; weights refitted on the training portion's out-of-fold predictions only | **0.9314** | components' budgets |

Search protocol behind those numbers: 32 sampled configurations per family (fixed seed, deduplicated);
chemprop and TabPFN were screened on folds 0–2 with documented caps (chemprop ≤ 12 epochs, 1 seed,
batch ≥ 128; TabPFN ≤ 3000 training rows, ≤ 2 ensemble members) and the **top five were re-run with
the configuration as sampled on all five folds** — selection used only those full 5-fold scores.
The embeddings family ran all 32 on five folds directly. Everything is in
`$SENTINEL_RUNS/chemtox/search/<model>_t-F2F_split_occurrence/`.

Rules that hold whatever the numbers turn out to be:

1. Each model is scored on test **exactly once**, with the configuration frozen here. A changed
   configuration requires a new pre-registration document and is reported as such.
2. Ensembles are named as ensembles in every table, with each component's own test score beside
   them. The stack's weights are refitted on the full training partition's out-of-fold predictions
   (never on test rows) and are recorded before scoring.
3. Seeds: 3 seeds per model (0, 1, 2); the reported value is the mean, with the seed spread given.
   A "SOTA" claim additionally requires the gain to hold in all 3 seeds (§15.1 of the plan).
4. Auxiliary data (other ADORE challenges, ECOTOX) may only enter a frozen configuration if every
   chemical in **any** test partition of the evaluated challenge was removed by CAS **and**
   InChIKey, asserted by `sentinel.models.chemtox.aux.assert_no_test_chemicals` and by
   `tests/test_chemtox_models.py`. The per-fold variant also removes the validation fold's
   chemicals.

## 2b. What the improvement cycle showed on CV (evidence behind the frozen choices)

* **The D-MPNN no longer trails the trees.** Tuned chemprop 0.9575 vs RF 0.9605 vs XGBoost 0.9929;
  paired on identical out-of-fold rows the chemprop−RF RMSE difference is **−0.0017, 95% CI
  [−0.046, +0.043], p = 0.93** (chemical clusters) — indistinguishable, not a win.
* **Where the error lives (both models):** regression to the mean dominates. In the most toxic
  quintile RMSE is 1.28 (chemprop) / 1.36 (RF) with bias **+0.75 / +0.95** (toxicity
  under-predicted); in the least toxic quintile bias is −0.47 / −0.63. ~98% of MSE is scatter, not
  overall bias (overall bias +0.13 / +0.15). Error is flat in how often a chemical was tested
  (1–2 tests: 0.92 / 0.93; >25 tests: 0.96 / 0.96), so it is not a rare-chemical problem.
  Duration slices differ by < 0.02 RMSE; saltwater rows are worse for both (1.03 / 1.01).
* **Variant marginals over the 32 screened chemprop configurations** (mean / best CV RMSE):
  plain init 0.971 / 0.944, ECOTOX-pretrained encoder 0.982 / 0.960, CheMeleon 1.092 / 0.964;
  multitask across taxa 1.085 / 0.974 vs single-task 0.981 / 0.944; depth 3–4 beat depth 5
  (0.976–0.982 vs 1.080). The winning configuration has the taxonomy embedding **on**
  (`species_dim` 16) and FiLM conditioning **off**, and averages 3 seeds.
* **CheMeleon initialisation did not help** here (its single-configuration run scored 1.0145 vs
  0.9754 for the same architecture trained from scratch), and neither did multitask across taxa —
  partly because the exclusion rule leaves it almost no data (next bullet).
* **Auxiliary data after the exclusion rule:** ECOTOX contributes 17,658 rows / 1,043 chemicals
  (33,722 rows dropped as test chemicals of `t-F2F`); the other ADORE challenges contribute only
  **842 rows**, because the `t-C2C`/`t-A2A` chemical space lies almost entirely inside `t-F2F`'s
  held-out set. Both are matched on CAS **and** InChIKey over the union of all split columns, and
  the per-fold filter additionally removes the validation fold's chemicals.
* **The ensemble is where the gain is:** mean 0.9406, NNLS stack **0.9314**, against the best
  single component 0.9575. It is reported as an ensemble with every component's own score.

## 3. Metrics (fixed)

Primary: micro-RMSE. Reported alongside, for every model: MAE, R², macro RMSE_M / RMSE_C / RMSE_T
(Gasser et al. Appendix B.2), % within 1 log unit, median absolute fold error. Every table also
carries the global-mean / species-mean / species+duration-mean baselines, the replicate noise floor
(0.383), the published 0.927 / 0.904 / 0.938 / 0.845, and the split's chemical-overlap count.

## 4. Statistics (fixed)

* Uncertainty: cluster bootstrap with **chemical** as the resampling unit, 2000 resamples,
  percentile intervals.
* Comparisons: paired bootstrap on identical rows against (a) the reproduced XGBoost + MACCS
  baseline and (b) the best trivial baseline, chemical clusters, two-sided.
* Multiplicity: Holm across the per-model primary comparisons.
* A claim of improvement requires the paired CI to exclude 0 after Holm correction.

## 5. Decision rules (fixed before scoring)

* **Pass (M4 primary, plan §15.2 H-M4):** test micro-RMSE on `t-F2F`/`split_occurrence` below
  0.927 with the chemical-bootstrap CI excluding the published value.
* **Competitive:** CI overlaps the published value or the baseline — reported as "competitive",
  not as SOTA.
* **Negative:** worse than the reproduced baseline — reported as a negative result, with the
  tuning budget, the trivial baselines and the noise floor, per plan §R8 and report 04 §E.2.
* No reinterpretation of the primary endpoint after seeing test numbers; secondary endpoints are
  labelled secondary wherever they appear.

## 6. Hashes

Computed by `scripts/eval/chemtox_select.py --freeze` over the frozen configuration dictionaries
(sorted-key JSON, sha256, first 16 hex characters) and over the scoring script itself:

| Item | sha256[:16] |
|---|---|
| frozen configurations (`selection.json` → `frozen`) | `e7a2e32376ebc586` |
| `scripts/eval/chemtox_score_test.py` (at freezing) | `26471c3ef2667ab1` |
| `scripts/eval/chemtox_score_test.py` (amended, see below) | `8125e3238ca3d7d2` |
| `scripts/eval/chemtox_select.py` | `4eaf6bd0481a5426` |
| `sentinel` package (`registry.package_hash()`) | `b5fbf5c45ac4dec2` |

Frozen at 2026-10-24 from
`$SENTINEL_RUNS/chemtox/select/t-F2F_split_occurrence/selection.json`. The scoring script
recomputes the first hash and **exits** if it differs from the value passed in `--expect-hash`, so
a configuration edited after this document was written cannot be scored silently. The script also
refuses to read test rows without `--i-have-approval`, and refuses to score a model that already
has a test entry unless `--allow-rescore` is passed (which records `supersedes`).

### 6a. Amendment, 2026-10-24, before any test row was read

The scoring script was amended in two ways after the hashes above were written and **before** the
test pass ran: (i) it now scores the two ensembles (the stack's weights are fitted by
non-negative least squares on the components' *training-portion* out-of-fold predictions, so no
test row contributes to a weight), and (ii) it compares every model against **both** reproduced
baselines (XGBoost + MACCS and RF + MACCS), with Holm applied across models for the comparison
against XGBoost. **The frozen configurations and their hash are unchanged**
(`e7a2e32376ebc586`) — the amendment adds reporting, it does not change what is scored. The
amended script's hash is recorded above; the pre-amendment hash is kept for the audit trail.

## 6b. Deviations log

**2026-10-24 16:49–16:59 EDT — first test pass aborted on a loader bug; re-run.**

* *What happened.* The scoring script loaded a single dataset with the MACCS fingerprint column
  and reused it for every frozen configuration. The chemprop configuration (no fingerprint)
  scored normally; the TabPFN configuration then raised `KeyError: 'chem_Morgan_fp'`, because its
  frozen feature spec needs the Morgan bit column, which had not been loaded. The run aborted.
* *Fix.* `scripts/eval/chemtox_score_test.py` now builds **one dataset per distinct frozen feature
  spec** (`dataset_for(spec)`, caching by the spec's fields) and each model is fitted and scored on
  the dataset matching its own spec; the MACCS dataset remains the reference frame for rows,
  targets and fold labels. No model, hyper-parameter, split, metric or decision rule changed.
* *What had already been computed, stated plainly.* The aborted attempt **did** fit and score the
  chemprop configuration on the test rows (3 seeds) and its test RMSE was printed to the run log
  and read by the engineer before the crash. It is **not** the case that nothing was computed or
  seen. What did *not* happen: no registry entry was written (`adore_chemtox_test` was empty, which
  was verified), no other model or ensemble was scored, no prediction file was kept, and no
  configuration, metric or decision rule was altered in response to that number — the re-run uses
  byte-identical frozen configurations.
* *Integrity consequence.* Because one model's test score was observed before the completed pass,
  the chemprop result must be read as "scored once, with the first attempt's identical
  configuration observed after an aborted run", not as a blind single scoring. The number is the
  same either way (same configuration, same seeds, same rows), but the distinction is recorded here
  rather than hidden. The failed run is also in `TRAINING_LOG.md` with its traceback, as
  `TrainingRun` logs crashes automatically (plan §R8).
* *Hashes.* Frozen-configuration hash **unchanged** (`e7a2e32376ebc586`). The scoring script's hash
  changed with the fix: `8125e3238ca3d7d2` (aborted attempt) → `5d835ee992f53323` (re-run), and the
  `sentinel` package hash is `9078b0d4694b51f9` at the re-run.

**2026-10-24 17:23–17:52 EDT — two secondary endpoints aborted twice; primary endpoint untouched.**

* *First abort (17:22).* The scoring script defaulted the frozen-configuration file to the
  challenge/split being scored, so the secondary endpoints looked for a `selection.json` that does
  not exist (selection ran on the primary endpoint only). Nothing was computed. Fixed by passing
  `--selection` explicitly; the frozen configurations are the primary endpoint's, as the plan
  intends.
* *Second abort (17:47, 17:51).* `t-C2C` and `a-CA2F-diff` carry missing Add-my-Pet values
  (`tax_lh_licm` in 5,196 of 6,630 `t-C2C` rows; `tax_lh_amd`, `tax_lh_licm`, `tax_ps_*` in
  `a-CA2F-diff`), which `t-F2F` does not, so the standardised features contained NaN and TabPFN's
  feature selection refused them. Fixed in `PublishedFeatures.transform`: a missing continuous
  value now takes the **training** mean (0 after standardisation), which is a no-op wherever
  nothing is missing. Verified afterwards: the primary endpoint's matrices are unchanged
  (21,981 × 178, all finite), so **the already-registered primary results are not affected**.
* *What had been computed before each abort.* In both aborted attempts the chemprop configuration
  completed and its test RMSE was printed and read (`t-C2C` 1.1098, `a-CA2F-diff` 1.2567). No
  registry entry was written for either split (verified: only `t-F2F/split_occurrence/test` and
  `t-F2F/split_scaffold-murcko/test` were present), and no configuration or decision rule changed
  in response.
* *Hashes.* Frozen-configuration hash unchanged (`e7a2e32376ebc586`); the `sentinel` package hash
  changed with the imputation fix and is recorded in each registry entry's `code` block.
* *Reporting consequence.* The `t-C2C` and `a-CA2F-diff` numbers are reported as **secondary**
  endpoints scored after two aborted attempts of the same frozen configurations, and the stack
  ensemble is **not** reported for any secondary endpoint, because its weights were fitted on the
  primary endpoint's out-of-fold predictions only (the mean ensemble is reported instead).

## 7. What happens after scoring

Results go to the registry (`is_baseline=False`, split `<challenge>/<split>/test`), the model card
is updated from the registry, and the post-results review gate (`REVIEW_REPORT_POST_RESULTS.md`)
runs before any number leaves the repository.
