# Model card — SENTINEL-ChemTox (M4)

**Status: phase 1 (implemented, published baseline reproduced, SENTINEL models cross-validated
only).** No SENTINEL model has been scored on the ADORE test rows. Every number below is read
from `results/registry.jsonl`; this card contains no literal that code did not write there
(RESEARCH_PLAN §3 R1/R2). Registry IDs are listed with each result in `RESULTS` below.

## Task

Predict acute aquatic toxicity — log10 LC50/EC50 in mol/L (`result_conc1_mean_mol_log`) — from
chemical structure plus the experimental conditions and the test species, on the **ADORE**
benchmark (Schür et al. 2023, DOI 10.25678/0008C9, CC BY 4.0). Downstream (not yet run): per-taxon
predictions → species sensitivity distribution → HC5 → risk quotient for recorded spill incidents
(`sentinel/models/chemtox/severity.py`, RESEARCH_PLAN §5.4, docs/research/04 §D.4).

## Protocol (frozen before training)

| Item | Value |
|---|---|
| Challenges | `t-F2F` (primary), `t-C2C`, `a-CA2F-same`, `a-CA2F-diff` |
| Splits | **only** ADORE's own columns: primary `split_occurrence`, secondary `split_scaffold-murcko` |
| Chemical overlap train↔test | **0** for both (asserted in code and in `tests/test_chemtox_models.py`); `split_totallyrandom` shares chemicals and is used only as the leakage demonstration |
| Model selection | mean RMSE over ADORE's official CV folds 0–4 inside the training partition |
| Test rows | scored **once**, and in phase 1 only for the reproduced published baseline |
| Metrics | micro RMSE/MAE/R², macro RMSE_M (chemical×taxon), RMSE_C (chemical), RMSE_T (taxon) exactly as Gasser et al. Appendix B.2, % within 1 log unit, median absolute fold error |
| CIs | cluster bootstrap, **resampling unit = chemical** (`test_cas`), 2000 resamples |
| Reference points printed with every table | global/species/species+duration means (`adore_trivial_baselines`), replicate noise floor 0.383 log units, the published 0.927/0.904/0.938/0.845 |

## Featurisation

* **Published non-structural block (37 columns).** Duration as an ordinal scaled to [0, 1]; media
  type, exposure type, concentration type one-hot; molecular weight, water solubility, melting
  point and logP standardised on training rows; Add-my-Pet ecology (climate, ecozone, food) as
  many-hot vectors; life span, ultimate body length and three DEB pseudo-data variables
  standardised. Phylogenetic distances excluded (the paper used them only in its GP variant).
  One-hot with the first level dropped reproduces the paper's stated width of 37 exactly; keeping
  all levels gives 40 and adding the binary migration flag gives 38. The paper's text mentions
  migratory behaviour but no combination that includes it reaches 37 — the discrepancy is recorded
  in `features.py` and both variants are selectable.
* **Fingerprints.** ADORE's MACCS/PubChem/Morgan/ToxPrint bit strings, filtered to "informative"
  bits by **standard deviation over the unique training chemicals > 0.1**. That rule reproduces
  all eight published informative-bit counts (occurrence: 141/471/380/172; totally-random:
  143/469/385/174); a per-row standard deviation does not. MACCS + published block therefore gives
  **178** columns for the occurrence split — the paper's Table 1 value.
* **RDKit path** (`rdkit_features`) for molecules outside ADORE: MACCS, Morgan and a fixed list of
  23 descriptors, deterministic and unit-tested, for the spill-severity route.
* Every transformer is fit on training rows only — inside CV, on the four in-fold training folds.

## Models

| Model | Implementation | Phase-1 state |
|---|---|---|
| Published baseline: XGBoost / random forest + fingerprint + published block | `xgb.py`, `scripts/train/chemtox_xgb.py` | **reproduced; test scored once** |
| TabPFN v2 (fingerprint + descriptors) | `tabpfn_model.py`, `tabpfn==2.2.1`, Prior Labs Licence 1.1 (Apache-2.0 + attribution) | CV only, one configuration |
| Chemprop v2 D-MPNN + species/taxonomy embedding + duration/endpoint-conditioned (FiLM) head, published block as extra FFN inputs | `chemprop_model.py`, `chemprop==2.3.1` | CV only, one configuration |
| The same with **CheMeleon** foundation initialisation (Zenodo 15460715; cite DOI 10.48550/arXiv.2506.15792) | `chemprop_model.py` | CV only, one configuration |
| Frozen ChemBERTa-2 (or MoLFormer-XL) embeddings + XGBoost | `embeddings.py` | CV only, one configuration |

TabPFN v2 is pinned deliberately: 2.5 and later are non-commercial (docs/research/04 §E.2). It was
installed with `--no-deps` because its `scikit-learn<1.7` / `huggingface-hub<1` pins would
downgrade the shared environment; `tabpfn_model.py` re-creates the single private helper it needs
(`sklearn.utils.validation._is_pandas_df`) in-process and disables telemetry. Chemprop was
installed with `rdkit==2025.9.3` held fixed so other workstreams' RDKit is untouched.

## Improvement cycle (CV only, 2026-10-22/24)

Each SENTINEL family received the published baseline's budget of **32 configurations** (fixed-seed
random search; chemprop and TabPFN screened on folds 0–2 with documented caps, top five re-run as
sampled on all five folds). Best CV RMSE: chemprop **0.9575**, RF 0.9605, ChemBERTa-2-MTR + XGBoost
0.9905, XGBoost 0.9929, TabPFN 1.0305; **ensemble mean 0.9406**, **NNLS stack 0.9314**. Extra
ECOTOX pretraining data (17,658 rows / 1,043 chemicals) and cross-taxa multitask data (842 rows)
pass the exclusion rule in `aux.py`: every chemical in **any** test partition of the evaluated
challenge removed by CAS *and* InChIKey, plus the fold's validation chemicals per fold.
Details, variant marginals and the error anatomy are in `docs/preregistration/chemtox_test_plan.md`
§2b; the frozen configurations and hashes are in §2 and §6 of the same document.

## Improvement cycle 2 and the prospective test set (2026-10-24)

**Why there is a second test set.** The single pre-registered ADORE test pass is spent
(`docs/preregistration/chemtox_test_plan.md`): chemprop 0.873 [0.736, 0.986], NNLS stack 0.874,
XGBoost + MACCS 0.917, RF + MACCS 0.957, with the published 0.927 inside every interval →
**"competitive, not SOTA"**, and that stays the ADORE result forever. RESEARCH_PLAN §15.5
(amendment of 2026-10-24) lifts the iteration cap but forbids re-claiming on a spent test block, so
an improved model is scored on a **new, pre-registered, previously unseen** test set instead.

**The prospective test set** (`docs/preregistration/chemtox_prospective_plan.md`,
`docs/datasheets/chemtox_prospective.md`, manifest `chemtox_prospective`, sha256
`3b2fc4398db940a2…`): ECOTOX records (release 2026-09-15) curated with **ADORE's own rules**,
restricted to chemicals absent from ADORE by CAS *and* InChIKey (so InChIKey overlap is 0 by
construction) and to species ADORE already covers. Primary endpoint: **fish, novel chemicals inside
ADORE's chemical domain — 1,912 rows / 301 chemicals / 64 species / 310 publications**, target SD
1.730, replicate noise floor measured inside the set **0.452** log units. Because an unseen CAS can
still be a salt or a stereoisomer of an ADORE chemical, the file also carries a salt-, charge- and
stereochemistry-insensitive `parent_novel` flag, and the **strict subset (1,362 rows / 236
chemicals)** is co-reported with the primary endpoint by pre-registration. Secondary tracks:
cross-taxa crustaceans (230 rows / 112 chemicals), the all-carbon variants, an out-of-domain
inorganic/metal-salt extension (8,774 rows) and a temporal track (publications after ADORE's 2021
cut-off) that is **underpowered at 29 chemicals and labelled as such**.

**What cycle 2 changed** (all selection on ADORE's CV folds 0–4 only, equal 32-configuration budget
per family, the published baselines re-tuned with the same budget):

| Change | Where | Literature it follows |
|---|---|---|
| LightGBM and CatBoost added to the tree families | `gbdt.py` | the standard GBDT alternatives to the published XGBoost/RF |
| **GPBoost**: boosted trees + **grouped random effects on the taxonomy** (species ⊂ genus ⊂ group) | `gpboost_model.py` | Zubrod et al. 2024 *Environ. Int.* 186:108607; Yuan et al. 2026 *ETC* 45(9) — both select GPBoost for cross-species toxicity; shrinkage up the hierarchy is the structurally correct answer to our cross-taxa collapse |
| **Frozen CheMeleon fingerprints** as features (not fine-tuning) | `chemeleon_fp.py` | Ben Hicham et al. arXiv:2604.16123 — a tabular model on frozen CheMeleon wins 86% of 58 tasks while *fine-tuned* CheMeleon wins 41%, which is why our phase-1 fine-tune made things worse |
| Full 216-descriptor RDKit block and a chemical block limited to the properties RDKit can compute | `features.py`, `data.py` | the prospective set has no experimental water solubility or melting point for new chemicals |
| **k-NN read-across**, distance-gated, tuned with the same budget | `readacross.py` | the regulatory comparator any structure-based claim must beat; it was missing from the phase-1 tables |
| ECOTOX **augmentation**: ADORE-curated records ADORE does not contain, every prospective chemical *and parent structure* removed and asserted gone | `prospective.augmentation_rows` | more chemical coverage is the direct lever on novel-chemical generalisation |
| **Deployment masking** in CV: water solubility and melting point are blanked on validation rows | `cycle.py` | they are missing for every prospective row, so an unmasked CV comparison would select a feature the deployed model cannot have (found by the pre-training review, §4.3 of the plan) |

**CV results of cycle 2 (ADORE folds 0–4, masked; registry `chemtox/adore_chemtox_cycle2`,
frozen block `630b43e905d9b852`).** Best 5-fold configuration per 32-configuration search (CV only):
catboost 0.948, chemeleon_gpboost 0.950, chemprop 0.953 (phase-1 winner, 32 + 1 chemical-block
variant), chemeleon_lightgbm 0.959, gpboost_multitaxa 0.967, **re-tuned XGBoost 0.973 (strongest
comparator)**, lightgbm_aug 0.980, lightgbm 0.987, rf_published_mwlogp 0.991, rf 0.993,
xgboost_published_mwlogp 0.995, gpboost 1.016, chemeleon_tabpfn 1.043, read-across 1.155;
published configurations with ADORE's experimental water solubility / melting point imputed:
RF 1.256, XGBoost 1.460. ENSEMBLE mean of the five candidates below 0.973: 0.916; NNLS stack 0.921
(members chosen on the same CV, so optimistic). Every search winner uses the RDKit-computable
`mw_logp` chemical block.

**Prospective test results (single pre-registered pass, 2026-11-09; registry
`chemtox/chemtox_prospective_test`; report `prospective_results_primary.json` sha256
`709803d052dfa9f0…`).** RMSE in log10(mol/L), chemical-cluster bootstrap 95% CI, mean of 3 seeds.

| Track | Rows / chemicals | Best candidate | Published reference (`rf_published_mwlogp`) | Best baseline | Δ vs reference | Δ vs best baseline | Label |
|---|---|---|---|---|---|---|---|
| **`fish_novel` (PRIMARY)** | 1,912 / 301 | ENSEMBLE NNLS stack 1.208 [1.003, 1.470] | 1.237 | re-tuned XGBoost 1.218 | −0.029 [−0.095, +0.039] | −0.010 [−0.070, +0.052] | **competitive, not SOTA** (no Holm rejection) |
| `fish_novel_strict` (co-primary) | 1,362 / 236 | ENSEMBLE NNLS stack 1.205 [0.953, 1.533] | 1.264 | re-tuned XGBoost 1.217 | −0.059 [−0.125, −0.007] (p 0.016; Holm not rejected) | −0.012 [−0.073, +0.043] | competitive |
| `crust_novel` (secondary) | 230 / 112 | gpboost_multitaxa 1.306 [1.082, 1.538] | 1.619 | re-tuned RF 1.577 | −0.313 [−0.540, −0.123] (Holm rejected **within the track**, all seeds) | −0.271 [−0.479, −0.090] (Holm rejected **within the track**, all seeds) | **training-data effect, not a method improvement — not claimable** |
| `crust_novel_strict` (secondary) | 154 / 74 | gpboost_multitaxa 1.336 [1.037, 1.632] | 1.406 | re-tuned RF 1.387 | −0.070 [−0.278, +0.088] | −0.051 [−0.271, +0.113] | sign kept, not significant |

* The published method as published (water solubility and melting point imputed for new
  chemicals): XGBoost 1.528, RF 1.432, i.e. 0.29 / 0.20 log units worse than the same learner on
  `mw_logp`, consistent with the masked-CV finding.
* Every model's prospective error (≥ 1.2) is far above its CV error (0.92–0.99) and the in-set
  replicate floor (0.452). Trivial baselines: 1.83–1.84 on the primary track.
* **The `crust_novel` result is a training-data asymmetry and carries no claim** (post-results
  review C-M4P-1). `gpboost_multitaxa` is the only one of the 21 scored models whose frozen
  configuration carries `extra=adore_taxa_all`, so it is the only model trained on ADORE's
  crustacean and algae rows; all seven comparators, the published reference and the trivial
  baselines were fit on fish rows only, and **0 of the 9 crustacean species appear in their
  training data** (which is why all three trivial baselines collapse to the global mean, 1.854).
  In a post-hoc reviewer diagnostic the frozen `rf_published_mwlogp` configuration refit on the
  same 33,448 rows scores **1.271** and the re-tuned `rf` **1.273**, against 1.306 for the
  candidate: given the same data the published method matches or beats it. The pre-registered
  decision fields are computed correctly; the −0.313 / −0.271 measures the value of ADORE's
  crustacean rows, not of GPBoost. Consistently, the same model is significantly **worse** than the
  re-tuned XGBoost on the primary fish track (+0.101 [+0.024, +0.166]). A cross-taxa claim would
  need pre-registered multitaxa comparators (`rf_multitaxa`, `xgboost_multitaxa`) on a new test
  block.
* **Holm scope.** Holm is applied per track across the 11 candidates, as pre-registered; it is
  **not** corrected across the 9 tracks. The `crust_novel` p-values (0.001 / 0.004) would not
  survive a correction across tracks (Bonferroni over 7 × 11 needs p ≤ 0.00065).
* **Caveats.** (1) On `crust_novel_strict` the gain keeps its sign but its CI includes 0
  (Δ −0.070 [−0.278, +0.087]). (2) Seed 0 of
  `gpboost_multitaxa` was **not reproducible**: a first pass attempt crashed during fitting (plan
  deviation D1; the target column was loaded with the frame but never used or output) and the scored
  re-run's seed-0 predictions differ from the crashed attempt's (mean |Δ| 0.22, max 1.30 log units),
  while seeds 1–2 and 10 other models are bit-identical and the three random forests agree to
  1e-15. **Both ensembles contain this model and inherit the irreproducibility**, with an effect of
  ≤ 0.002 RMSE on the primary track (stack 1.2079 → 1.2065 with the crashed seed 0). The scored
  seed 0 is the worst of the three on `crust_novel` (1.379 against 1.279/1.311), so the reported
  number is the conservative one. Diagnosed without re-scoring (plan D1): the fit is bit-identical
  within a process and at CV-fold size, but intermittently differs **across processes** at the
  pass's data size; the GPU path, a first-fit effect, OpenMP threads and the thread count are ruled
  out, and the root cause is recorded as **unexplained** (leading candidate: the unseeded
  `gpb.GPModel` random-effects solver). Thread count separately changes the result, so this family
  is not reproducible across thread budgets.
* **The §2 `_trainpart` sensitivity entry does not exist and will not be retried** (lead decision):
  that pass failed with a GPBoost conjugate-gradient NaN/Inf on the training partition at `gpboost`
  seed 2 after 11 of 16 models (plan deviation D2), produced no registry row and no report, and
  re-rolling a non-deterministic failure would be a forking path. The plan's "what the extra rows
  did" analysis is therefore **missing and reported as missing**. Reported instead as a **stability
  finding**: `gpboost` cannot be fitted on the 21,981-row training partition although the same
  configuration fits the 26,114-row pooled set, and 2 of its 5 cycle-2 CV finalists failed the same
  way. The primary and secondary results stand without it. (3) 51% of the primary rows (117
  chemicals) are chemicals ADORE's own pipeline saw and dropped, mostly for missing chemical
  features (plan §1.3c); the tree models and the stack score **1.11–1.21** on those rows and
  **1.24–1.46** on the rest, while chemprop reverses (1.52 against 1.20) and is in fact the best of
  all models on the ADORE-unseen rows (1.202, against 1.243 for the stack and 1.287 for the
  reference) — the sub-population closest to "chemicals tested after ADORE", and the opposite of
  the headline ordering (descriptive, pre-registered breakdown). (4) Pre-registered
  parent-structure-clustered CIs (plan §4.5) cover the two fish tracks and change no conclusion;
  the reviewer computed `crust_novel` post hoc ([−0.614, −0.066] vs reference, [−0.541, −0.045] vs
  best baseline), also excluding 0. 11 candidates were Holm-corrected; 11 × 32-configuration searches plus
  chemprop (32 + 1) and four published configurations were tried in cycle 2.

**Pre-training review.** The new pipeline went through the review gate twice
(`REVIEW_REPORT_PRE_TRAINING.md`, section "M4 SENTINEL-ChemTox v2"): round 1 returned **FAIL** with
three blocking findings — a crash path in the scoring script that would have printed scores and then
died, the biased feature-spec comparison now fixed by deployment masking, and the salt/stereoisomer
novelty issue now measured and co-reported — all three fixed before any prospective row was scored,
and re-reviewed in round 2.

## Results

Read them from the registry rather than from this file:

```
python -B scripts/eval/chemtox_report.py --challenge t-F2F --split split_occurrence
```

* `experiment = adore_published_baseline`, `is_baseline = True` — the reproduction (CV and the
  single test scoring), with `extra.published`, `extra.deltas_vs_published` and
  `extra.n_configurations` on every entry.
* `experiment = adore_chemtox_cv`, `is_baseline = False` — the SENTINEL models, **CV only**.
* `experiment = adore_trivial_baselines` — global/species/species+duration means and the replicate
  noise floor (another engineer's run).

Reproduction criterion (RESEARCH_PLAN §15.2 H-M4): micro RMSE within ±0.03 of 0.927 for
XGBoost + MACCS on `t-F2F`/`split_occurrence`/test. The run that produced the reproduced numbers
also reports RMSE_M, RMSE_C and RMSE_T; note that the four published numbers come from **three
different model × representation combinations** (micro RMSE and RMSE_T from XGBoost + MACCS,
RMSE_M from RF + MACCS, RMSE_C from RF + ToxPrint), so the comparison is made model by model.

## Notes that affect how the numbers must be read

* **`a-CA2F-same` shares chemicals between train and test by construction** (991 chemicals under
  `split_occurrence`): that challenge trains on crustaceans + algae and tests on fish *for the same
  chemicals*, so the extrapolation is taxonomic, not chemical. The leakage guard reports the
  overlap there and raises everywhere else (`SHARED_CHEMICALS_BY_DESIGN` in `data.py`).
* **Two split-label conventions exist in the registry.** The trivial-baseline run logs
  `t-F2F/occurrence/cv`, the model runs log the column name `t-F2F/split_occurrence/cv`.
  `scripts/eval/chemtox_report.py` normalises both so one table holds everything; worth
  standardising before the paper tables are built.
* **TabPFN inference is chunked** (`predict_chunk`, default 2000 rows) and the per-chemical
  training subsample is 6000 rows: a single large batch segfaulted inside `libc10_cuda` on these
  11 GB Turing cards.
* **Chemprop caches molecular graphs** (`dset.cache = True`); without it the GPU sat at ~5%
  utilisation while the CPU re-featurised every epoch.

## Known limitations

* The published hyper-parameter grid (their ESI Table 2) is not retrievable — RSC is behind a bot
  check and the renkulab GitLab that hosts their code does not respond. Our grid covers the same
  six hyper-parameters; the number of configurations tried is logged with every entry.
* The macro RMSE_M formula divides by `N_C · N_T` in the paper; unobserved cells make that
  undefined, so the mean is taken over observed (chemical, taxon) cells and the cell count is
  recorded (`rmse_macro_cells`).
* `t-A2A` has 85 test rows; any algae-only statement will have very wide intervals
  (docs/research/04 §A.2).
* TRIDENT must not be run over ADORE test rows (its training overlaps ADORE); the comparison route
  is its released per-SMILES held-out errors, paired on shared chemicals — not implemented yet.
* `severity.py` is an interface with unit tests only: no severity number has been computed from
  real incidents, and the validation available for it is weak (NRC reports are initial and
  unvalidated, response-action fields are reporting-biased).

## Reproduce

```
# published baseline (CV + single test scoring)
python -B scripts/train/chemtox_xgb.py --model xgboost --fingerprint MACCS --score-test
python -B scripts/train/chemtox_xgb.py --model rf      --fingerprint MACCS    --score-test
python -B scripts/train/chemtox_xgb.py --model rf      --fingerprint ToxPrint --score-test

# SENTINEL models, CV on the official folds only (one GPU, preferring GPU 2)
python -B -m sentinel.core.gpu --name chemtox_tabpfn --prefer 2 -- \
    python -B scripts/train/chemtox_cv.py --model tabpfn --accelerator gpu
python -B -m sentinel.core.gpu --name chemtox_chemprop --prefer 2 -- \
    python -B scripts/train/chemtox_cv.py --model chemprop --accelerator gpu
python -B -m sentinel.core.gpu --name chemtox_chemeleon --prefer 2 -- \
    python -B scripts/train/chemtox_cv.py --model chemprop_chemeleon --accelerator gpu
python -B -m sentinel.core.gpu --name chemtox_embeddings --prefer 2 -- \
    python -B scripts/train/chemtox_cv.py --model embeddings --accelerator gpu

python -B -m pytest tests/test_chemtox_models.py -q
```

Data: `$SENTINEL_DATA/raw/adore/release` (converted release) and `$SENTINEL_DATA/processed/adore`
(official split assignments), manifests `adore_raw` and `adore`. Run artifacts:
`$SENTINEL_RUNS/chemtox/{published_baseline,cv}/<run>/` (split report, CV table of every
configuration, test predictions, registry IDs).

## Licences and citation

ADORE CC BY 4.0 (Schür et al., *Sci Data* 2023). Published baseline: Gasser et al.,
*Environ. Sci.: Adv.* 3:1124–1138 (2024), DOI 10.1039/D4VA00072B. Chemprop MIT; CheMeleon weights
from Zenodo record 15460715 (cite DOI 10.48550/arXiv.2506.15792); TabPFN v2 weights under the
Prior Labs Licence 1.1; ChemBERTa-2 and MoLFormer-XL as released on Hugging Face.
