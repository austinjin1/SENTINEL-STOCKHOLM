# Pre-registration — SENTINEL-ChemTox (M4) **prospective** test set

**Status: stage 1 frozen 2026-10-24 (data + rules). Stage 2 (frozen model configurations) is
appended below before any prospective row is scored. No model of any kind has been fitted to, or
scored on, these rows at the time stage 1 was written** — the builder
(`scripts/data/chemtox_build_prospective.py`) fits nothing and prints no target value.

## 0. Why a new test set exists

The single pre-registered ADORE test pass is **spent** (`docs/preregistration/chemtox_test_plan.md`;
chemprop 0.873 [0.736, 0.986], NNLS stack 0.874, XGBoost + MACCS 0.917, RF + MACCS 0.957, published
0.927 inside every CI → "competitive, not SOTA"). Those numbers stay reported as they are, forever.
RESEARCH_PLAN §15.5 (amendment of 2026-10-24) lifts the iteration cap but forbids re-claiming on a
spent test block: an improved model needs a **new, previously unseen, pre-registered** test set.
This document is that pre-registration.

## 1. What the prospective set is

ECOTOX (release **2026-09-15**, `$SENTINEL_DATA/processed/ecotox/ecotox_aquatic.parquet`, manifest
`ecotox`) records of acute lethality in fish, crustaceans and algae, curated with **ADORE's own
rules**, restricted to **chemicals ADORE never contains** (and, separately, to publications after
ADORE's data cut-off). ADORE's core is ECOTOX, so this is the same measurement process, the same
units and the same aggregation — only the chemicals (or the publication dates) are new.

### 1.1 ADORE's curation rules, recovered from the release

Recovered column by column from `raw/adore/release/processed/ecotox_mortality_filtered.parquet`
(33,448 rows) and the challenge files on 2026-10-24, and implemented in
`sentinel/models/chemtox/prospective.py`:

| Rule | ADORE's values | Applied to ECOTOX via |
|---|---|---|
| effect | `MOR` only | `effect_norm == MOR` |
| endpoint | `LC50`, `EC50` | `endpoint_norm ∈ {LC50, EC50}` |
| observation duration | exactly {24, 48, 72, 96} h | `duration_h ∈ {24,48,72,96}` |
| media type | `FW`, `SW` | `media_type_norm ∈ {FW, SW}` |
| concentration type | `A`, `F`, `T` (ECOTOX's `NC`/`D` never occur in ADORE) | `conc1_type_norm ∈ {A,F,T}` |
| exposure type | `S`, `F`, `R`, `NR` — all four occur, i.e. no filter | no filter |
| test location | `LAB`, `NR`, `FIELDA` — i.e. no filter | no filter |
| censoring | never `<` or `>`; `~` and min–max averages are kept | `censor_kind ∈ {"", approximate}` |
| target | `result_conc1_mean_mol_log` = log10(LC50/EC50 in mol/L) | `log10_conc_mol_L` |
| taxa | fish / crustaceans / algae, 203 species | `taxon_group ∈ {fish, crustaceans, algae}` |
| structure | a usable structure per chemical | `chem_smiles` present and RDKit-parseable |
| rows | one row per ECOTOX `result_id` (no replicate aggregation) | same |

Curation funnel on the 2026-09-15 release (from `report.json`): 471,097 aquatic rows → 153,346
(`MOR`) → 120,230 (endpoint) → 100,706 (duration) → 81,064 (taxa) → 80,800 (media) → 77,523
(concentration type) → 71,301 (not censored) → 60,834 (molar target) → **60,590 rows / 3,020
chemicals** with a structure. Of those, 27,781 rows are ADORE rows (matched by `result_id`).

Two documented deviations from ADORE, both unavoidable and both reported here rather than hidden:

1. **Mean-only concentrations.** ADORE reconstructed 529 of 26,114 t-F2F values (2.0%) as the
   average of a reported min and max (`result_conc1_mean_op == min_max_average`). The prospective
   set uses only rows with a reported mean. This removes rows; it does not change any row's value.
2. **Water solubility and melting point** (`chem_ws`, `chem_mp`) are experimental per-chemical
   values ADORE compiled from PubChem and are unavailable for chemicals outside ADORE. They are
   left missing and take the **training mean** inside `PublishedFeatures.transform` (the imputation
   that already exists, see `chemtox_test_plan.md` §6b). §4.3 pre-registers the CV ablation that
   quantifies what this costs and the feature spec actually used.

Everything else in the published 37-column block is reproducible for new chemicals: molecular
weight = RDKit `Descriptors.MolWt` (matches ADORE's `chem_mw` to 0.011 g/mol on average over 300
chemicals), log Kow = RDKit `Crippen.MolLogP` (matches `chem_rdkit_clogp` to 2e-15),
**MACCS** = RDKit `MACCSkeys.GenMACCSKeys` on the canonical SMILES (**bit-identical** for 300/300
chemicals checked) and **Morgan** = RDKit Morgan generator, radius 2, 2048 bits (200/200
bit-identical). ADORE's PubChem and ToxPrint fingerprints cannot be computed locally, so
prospective scoring is restricted to specs using MACCS, Morgan or no fingerprint. Species-level
Add-my-Pet taxonomy is copied from ADORE per species (asserted species-constant), which is why the
tracks are restricted to species ADORE already contains.

### 1.2 Exclusion logic (the part that makes this a valid test set)

A row enters a *novel-chemical* track only if

* its ECOTOX `result_id` is **absent from ADORE** (all 33,448 curated rows plus every challenge
  file — 158,330 rows inspected in total); **and**
* its chemical is absent from ADORE by **CAS** *and* by **InChIKey** (2,408 CAS / 2,407 InChIKeys),
  so **InChIKey-level overlap with ADORE is 0 by construction**; **and**
* its species is one of ADORE's 203 species (so the taxonomic block is defined).

`sentinel.models.chemtox.prospective.assert_disjoint_from_adore` raises on any overlap and is run
for every track inside the builder. The *temporal* track deliberately allows ADORE chemicals (its
novelty is the publication date) and is labelled accordingly everywhere.

**ADORE's cut-off.** ADORE's latest ECOTOX reference publication year is **2021**. Unit matters, so
both counts are given: over ADORE's **distinct** reference numbers, 2,432 of 2,444 match our ECOTOX
references table, their maximum publication year is 2021 and the 99th percentile 2019; weighting by
rows instead gives 2,395 matched references and a 99th percentile of 2016. Either way no ADORE row
comes from a publication after 2021, which is what the temporal track needs, so it is
`publication_year > 2021`.

### 1.3 Tracks, frozen with their sizes

| Track | Definition | Rows | Chemicals | Species | References |
|---|---|---|---|---|---|
| **`fish_novel` ∩ `in_adore_domain`** — **PRIMARY** | fish, novel chemical, `structure_class ∈ {organic, organic_multicomponent}` | **1,912** | **301** | 64 | 310 |
| `fish_novel` (all carbon-containing) | adds organometallic (367 rows) and carbon-bearing inorganics (34) | 2,313 | 363 | 71 | 364 |
| `crust_novel` ∩ `in_adore_domain` | crustaceans, novel chemical, ADORE classes | 230 | 112 | 9 | 98 |
| `crust_novel` (all carbon-containing) | cross-taxa track | 295 | 136 | 10 | 114 |
| `algae_novel` | algae, novel chemical | 3 | 2 | 2 | 2 |
| `noncarbon` | novel chemicals without carbon (metal salts, inorganics) | 8,774 | 225 | 114 | 1,136 |
| `temporal` | published after 2021, ADORE species, carbon-containing | 115 | 29 | 13 | 44 |

**Why the primary track is the `in_adore_domain` subset.** ADORE's 2,408 chemicals are 2,191
`organic`, 207 `organic_multicomponent` and 10 `inorganic`, with **no** organometallics
(`processed/chemicals/chem_ids.parquet`). Scoring a model on organometallics or metal salts is a
different, out-of-domain question; it is kept as a labelled extension track rather than mixed into
the primary endpoint. The flag is stored in the frozen file as `in_adore_domain`, so the primary
rows are fixed by the file, not by a later decision.

### 1.3a How novel the "novel" chemicals really are (added 2026-10-24 after the review gate)

A chemical absent from ADORE by CAS **and** InChIKey can still be a salt, a single stereoisomer or
a different counter-ion of a chemical ADORE contains — a different substance in a registry, the same
structure for a structure-based model. Measured with a salt-, charge- and stereochemistry-insensitive
structural key (largest fragment, charges neutralised, stereochemistry removed, InChIKey skeleton
block; `prospective.parent_key`), **65 of the primary track's 301 chemicals (550 of 1,912 rows) share
a parent structure with an ADORE chemical** — aniline hydrochloride against aniline, the four
difenoconazole stereoisomers against the racemate, zinc/copper dithiocarbamates against the
sodium salt. 5.0% of the 301 chemicals have a Morgan/Tanimoto similarity of 1.0 to a training
chemical.

The frozen file therefore carries a stored boolean **`parent_novel`** and a per-row
**`nn_tanimoto_adore_train`**, and this document pre-registers a **strict subset** as a mandatory
co-reported endpoint:

| Endpoint | Rows | Chemicals | Species | References | per-chemical NN Tanimoto (median, % < 0.4, % = 1.0) | target SD | replicate floor |
|---|---|---|---|---|---|---|---|
| **Primary** — `fish_novel` ∩ `in_adore_domain` | 1,912 | 301 | 64 | 310 | 0.444 · 40.2% · 5.0% | 1.730 | 0.452 |
| **Strict (co-primary, track `fish_novel_strict`)** — the same ∩ `parent_novel` | 1,362 | 236 | 54 | 238 | 0.391 · 50.8% · 0.8% | 1.764 | 0.421 |
| `crust_novel_strict` = `crust_novel` ∩ `in_adore_domain` ∩ `parent_novel` | 154 | 74 | 8 | 75 | — | 1.747 | 0.673 |

**Decision rule for the strict subset (frozen):** a "SOTA"/"improved" claim on the primary track is
reported **only together with** the strict subset's result, and only stands if the point estimate of
the improvement keeps its sign on the strict subset. If the two disagree, the weaker is the headline.
The strict subset is not a second chance: both are computed in the same single pass.

**Power.** Primary track: 301 chemicals (the resampling unit), 1,912 rows, target SD 1.730 log
units, 64 species, 310 independent publications; strict subset 236 chemicals / 1,362 rows. The
**replicate noise floor measured inside this set** (pooled within (chemical, species, duration) SD
over 361 groups / 1,180 rows) is **0.452** log units — higher than ADORE's 0.383, i.e. these rows are
intrinsically noisier. The temporal track (29 chemicals, 59 of its 115 rows parent-novel) is
**underpowered and is reported as underpowered**: no claim will rest on it.

**Domain of applicability (recorded before freezing).** Nearest-neighbour Morgan/Tanimoto similarity
to the 1,588 chemicals of ADORE's `t-F2F` **training partition**, per *chemical*: `fish_novel` (363
chemicals) median 0.425 (p10 0.218, p90 0.800), 43% below 0.4; the 301 primary chemicals median
0.444 (p10 0.222, p90 0.800), 40.2% below 0.4, 5.0% at 1.0; the 236 strict chemicals median 0.391,
50.8% below 0.4, 0.8% at 1.0; `crust_novel` median 0.476; `noncarbon` median 0.200 (99% below 0.4);
`temporal` median 0.571. Per *row* the primary track's median is 0.500 and 1.6% of rows sit at 1.0.

### 1.3b Composition and residual risks of the primary track (measured, not assumed)

* **Composition.** 958 of the 1,912 primary rows (50%) are `organic_multicomponent` against 8.6% of
  ADORE's chemicals, and 596 rows (82 chemicals) are metal-bearing — copper/zinc/manganese
  dithiocarbamates, copper 8-quinolinolate, Paris green. They are in-domain by the stated rule
  (ADORE itself contains 96 metal-bearing multicomponent chemicals), but the distribution is
  shifted, so §4.1 reports the per-`structure_class`, per-`has_metal`, per-duration and
  per-`parent_novel` breakdown of every model's error.
* **Row re-issue.** 839 of ADORE's 33,448 `result_id`s (2.5%) are absent from the 2026-09-15 ECOTOX
  release, so the `result_id` check alone could not catch an ADORE row re-issued under a new id.
  The novel tracks are protected anyway by the chemical-level exclusion and the temporal track by
  the publication year; the residual effect is limited to possible duplicate *training* rows among
  the ECOTOX augmentation, which is harmless.
* **Tracks are not disjoint.** 63 rows belong to two tracks (a post-2021 publication about a novel
  chemical is both `fish_novel` and `temporal`); each (row, track) pair occurs once and the count is
  in `report.json` (`rows_in_more_than_one_track`). No row is counted twice inside one track.

### 1.4 The frozen file

| Item | Value |
|---|---|
| Path | `$SENTINEL_DATA/processed/chemtox_prospective/prospective.parquet` |
| Rows (all tracks) | 11,500 |
| **sha256** | **`3b2fc4398db940a26c5273f815fdfe872a08b620e1f1b36bd4a12bb38dcc54d9`** |
| Report | `$SENTINEL_DATA/processed/chemtox_prospective/report.json` |
| Manifest | `manifests/chemtox_prospective.json` |
| Environment | `/home/bcheng/miniconda3/envs/sentinel2/bin/python` (pandas 2.3.3, pyarrow 26.0.0, rdkit 2025.09.3); the build is byte-reproducible in that environment (verified twice) |
| Builder | `scripts/data/chemtox_build_prospective.py` (sha256 `80a617aecd89610a46c72ca0622fef54a9431de80e500879fab7403cf7d0b5c8`) |
| Module | `sentinel/models/chemtox/prospective.py` (sha256 `74765aea3243aa1ee7bc8ad37c13967f3edfbe670b0a4ba43d214e3bc85fac8f`) |

The scoring script recomputes the file's sha256 and **exits** if it differs from the value passed
in `--expect-data-hash`, and it refuses to read prospective rows without `--i-have-approval`
(the same seal as the ADORE pass) and refuses to re-score a model that already has a prospective
entry unless `--allow-rescore` is passed, which records `supersedes`.

## 2. Training data for the prospective evaluation (frozen)

Every model — SENTINEL models **and** the reproduced baselines — is trained on exactly the same
rows, so the comparison is a model comparison:

* **Primary training set:** all ADORE `t-F2F` rows (26,114; both partitions of
  `split_occurrence` pooled). This is legitimate because every prospective chemical is absent from
  ADORE by CAS and InChIKey — the ADORE test rows carry no information about them — and it is what
  a deployed model would be trained on. The ADORE test partition is *training data* here; no claim
  is made on it.
* **Permitted auxiliary data (an ablation, not the default):** ECOTOX rows curated by the same
  rules whose chemicals are absent from the prospective tracks by CAS **and** InChIKey, asserted by
  `prospective.assert_prospective_chemicals_absent` before a fit. Inside CV, the validation fold's
  chemicals are additionally removed (`aux.drop_chemicals`), exactly as in phase 1.
* **Multi-taxa training rows (frozen rule, stated exactly).** A configuration may add **all** rows
  of ADORE's other challenges — `t-C2C` (6,630) and `t-A2A` (704) — because every prospective
  chemical is absent from ADORE entirely, so those rows cannot contain one (asserted by
  `prospective.assert_prospective_chemicals_absent`, which keeps all 7,334). This is a *weaker*
  filter than phase 1 used, and deliberately so: phase 1 had to remove every chemical of any
  `t-F2F` test partition and was left with 842 rows, while here the ADORE test rows are training
  data and no claim is made on them again. Inside CV the validation fold's own chemicals are still
  removed per fold. A model trained this way is named `*_multitaxa` everywhere, and this is the
  pre-registered lever on the cross-taxa weakness the ADORE pass exposed (`a-CA2F-diff` 1.257).
* **Naming of training-data variants (frozen):** `*_aug` = ECOTOX augmentation rows,
  `*_multitaxa` = ADORE other-taxa rows, `*_aug_multitaxa` = both, `*_trainpart` = the sensitivity
  fit on the training partition only, `*_published` = the reproduced published configuration,
  `*_mwlogp` = the published configuration with the chemical block reduced to MW + logP.
* **Sensitivity entry:** the selected model and the reproduced published baseline are also scored
  once trained on the official training partition only (21,981 rows), to show what the extra rows
  did. Those entries carry the model-name suffix `_trainpart`, so they are separate registry rows
  and never overwrite or supersede the primary entries.

Hyper-parameters are selected **only** on ADORE's official CV folds 0–4 inside the training
partition of `t-F2F`/`split_occurrence`. No prospective row ever enters selection, fitting of a
transformer, or an ensemble weight.

## 3. Models to be scored (names frozen here; their configurations in stage 2)

The names below are the registry `model` values and the keys of the frozen block, so a name in this
document always resolves to code (`scripts/eval/chemtox_score_prospective.py`).

**Comparators** (never candidates for a claim):

1. `xgboost_published` — the **published method** (Gasser et al. 2024): XGBoost on MACCS + the
   published 37-column block, the configuration frozen for the ADORE pass, refit on §2's rows.
   This is the comparator plan §15.1 (i) requires.
2. `rf_published` — the published method's random forest, likewise.
2b. `xgboost_published_mwlogp`, `rf_published_mwlogp` — the same published configurations with the
   chemical block reduced to MW + logP. ADORE's experimental water solubility and melting point do
   not exist for a new chemical, so the published spec has to impute them; scoring both variants and
   comparing against the better of the two keeps the published method at full strength
   (RESEARCH_PLAN §15.5 guard 4). **Which variant is the §15.1 (i) reference is decided by masked CV
   over all four published runs — both learners × both chemical blocks — and written into the
   **hashed** frozen block as `__published_reference__` by `chemtox_cycle_select.py --freeze`.** The
   scoring script reads it from there and **exits** if it is absent; it never picks one itself and
   never sees a prospective number before the choice is fixed. The masked CV scores of all four are
   recorded next to it (`__published_reference_cv__`).
3. `xgboost`, `rf` — the same two families **re-tuned** in improvement cycle 2 with the same
   32-configuration budget as every other family (so the trees are not handicapped by being left at
   their published spec). **Every** variant of `xgboost`, `rf` and `readacross` — including ones
   trained with extra rows (`*_aug`, `*_multitaxa`) — is a comparator, so a strong variant of a
   baseline can never be presented as the candidate that beat the baseline.
4. `readacross` — distance-gated k-nearest-neighbour read-across (Tanimoto on Morgan bits, same
   species / same duration where available), tuned with the same budget: the classical QSAR
   comparator that any structure-based claim must beat.
5. `global_mean`, `species_mean`, `species_duration_mean` — trivial baselines fitted on §2's rows.
6. `best_baseline` — an alias, resolved **per track** to whichever of 1–5 has the lowest RMSE on
   that track. Choosing it after seeing the scores is deliberate: it makes the claim harder, never
   easier, and it is what "the strongest simple baseline" means in plan §15.1 (i).

**Candidates:**

7. `chemprop` — the chemprop v2 D-MPNN configuration frozen for the ADORE pass, **re-run under
   masked CV in two explicit chemical-block variants (§4.3) and frozen as the better of the two**
   (`mw_logp`, masked CV 0.9527, against 0.9564 for the phase-1 spec), refit on §2's rows (its
   ADORE test number stays what it is; this is a new fit for a new test set). Its configuration
   count is reported as "32 + 1 chemical-block variant": the 32-configuration phase-1 search plus
   the §4.3 block choice.
8. The cycle-2 search winners of the remaining families (`lightgbm`, `gpboost`, `catboost`,
   `chemeleon_lightgbm`, `chemeleon_gpboost`, `chemeleon_tabpfn`, … — whichever exist in stage 2),
   plus `ensemble_mean` and `ensemble_stack_nnls` over them, with every component's own score
   reported next to the ensemble's.

Comparator or candidate is decided by an explicit `role` field written into the frozen block by
`chemtox_cycle_select.py --freeze`, not by pattern-matching a name, and the scoring script reads
that field (round-2 review finding N1). Ensembles list their components by the **frozen** names
they are scored under, every component must be present in the pass, and a missing component or a
missing out-of-fold matrix is a hard error rather than a silently smaller ensemble (N2).

The v1 NNLS stack is **not** scored: it would need all five v1 components refitted (including the
two that were weakest on CV), and every additional candidate costs power in the Holm correction.
The v1 comparison is carried by `chemprop`, the v1 winner. This deviation from the stage-1 draft is
recorded here rather than silently dropped.

## 4. Metrics, statistics and decision rules (frozen)

### 4.1 Metrics
Primary: **micro-RMSE** on the primary track. Reported beside it for every model: MAE, R², bias,
macro RMSE_M / RMSE_C / RMSE_T (Gasser et al. Appendix B.2), % within 1 log unit, median absolute
fold error, the within-set replicate floor (0.452), and breakdowns by `structure_class`,
`has_metal`, duration and `parent_novel` (cells with ≥ 10 rows).

### 4.2 Statistics
* Uncertainty: cluster bootstrap with **chemical** as the resampling unit, 2,000 resamples,
  percentile intervals (identical to the ADORE pass).
* Comparisons: **paired** bootstrap on identical rows (`sentinel.eval.bootstrap.paired_bootstrap`,
  the function the ADORE pass used), chemical clusters, two-sided, against `xgboost_published`,
  `rf_published`, `xgboost`, `rf`, `species_duration_mean`, `readacross` and `best_baseline`.
* Multiplicity: Holm across the **candidate** models (comparators are not in the family, so they
  cannot inflate or deflate the correction) for two reference comparisons: against
  `published_reference` (the best of the **four** published runs — both learners × both chemical
  blocks — on masked CV, as §3 2b and §4.3 define and the code implements) and against
  `best_baseline`. The bootstrap p-value is two-sided, so a Holm rejection counts only when the
  candidate's RMSE difference is **negative** (it is the better model).
* Seeds: 3 seeds (0, 1, 2) for every stochastic model; the reported value is the mean of the seeds'
  predictions, with every seed's own RMSE listed. Plan §15.1 (iii) is computed explicitly: each
  comparison records the per-seed RMSE difference and a boolean `all_seeds_better`. For a
  deterministic model the rule does not apply and is recorded as `null` with the reason — never as
  a pass. An **ensemble is built per seed** (seed *s* of the ensemble combines seed *s* of each
  stochastic component, a deterministic component contributing its single prediction), so the rule
  is computable for ensembles too and is applied to them.

### 4.3 The feature-spec question, settled on CV before freezing — with deployment masking
Every family's search samples both chemical blocks: `four` (ADORE's MW, water solubility, melting
point, logP) and `mw_logp` (only the two properties RDKit can compute for any molecule). Water
solubility and melting point are **missing for every prospective row**, where they take the training
mean.

A CV comparison that let the validation rows keep their real solubility and melting point would
therefore measure a feature the deployed model will not have, and would favour the `four` spec
exactly because it degrades at test time. **Fixed (2026-10-24, after the review gate):** during CV
those two columns are blanked on the **validation** rows of every fold
(`cycle.DEPLOYMENT_MISSING_COLUMNS`, `cycle.mask_deployment_missing`; training rows keep them, as a
deployed model's training data would). Run directories of masked searches carry the suffix `_dm`
and `chemtox_cycle_select.py` reads **only** those, so an unmasked run cannot enter selection.
Searches run before the fix were discarded, not merged.

The spec with the better masked-CV RMSE is what each family's frozen configuration uses, and the
same rule applies to the baselines and to `chemprop` — each of those is re-run under masked CV as a
single explicit configuration (`chemtox_cycle.py --config-json`), so no unmasked out-of-fold
prediction enters selection, a stack weight or an ensemble (round-2 review finding N4).

**First evidence, and it is substantial** (masked CV, ADORE folds 0–4, one explicit configuration
each, run directories `*_dm_cfg-pub_*`):

| Reproduced published configuration | chemical block | masked CV RMSE |
|---|---|---|
| XGBoost + MACCS + published block | `four` (incl. water solubility, melting point) | **1.4600** |
| XGBoost + MACCS + published block | `mw_logp` | **0.9945** |
| Random forest + MACCS + published block | `four` | **1.2561** |
| Random forest + MACCS + published block | `mw_logp` | **0.9909** |

In other words, *the published method cannot be applied to a chemical whose experimental water
solubility and melting point are unknown without losing about half a log unit of accuracy* — which
is exactly the situation on the prospective set, and a result that does not appear in the
literature. The best of the four, **`rf_published_mwlogp` (0.9909)**, is therefore the
`published_reference` fixed in the frozen block. All four numbers are reported; none is a
prospective number.

### 4.4 Decision rules
* **"SOTA / improved" on the prospective primary track** requires **all** of:
  (a) the paired CI of the RMSE difference against `published_reference` (the frozen choice above)
  to exclude 0 after Holm across the candidates; (b) the same against `best_baseline` (which includes the re-tuned trees,
  the read-across and the trivial baselines); (c) `all_seeds_better` true for both comparisons
  wherever the rule applies (plan §15.1 iii); and (d) the improvement keeping its sign on the
  **strict parent-novel endpoint** — scored as its own track `fish_novel_strict` (1,362 rows / 236
  chemicals) with its own cluster-bootstrap CI and its own paired comparisons, not as a breakdown
  cell. `crust_novel_strict` (154 rows / 74 chemicals) is reported the same way for the cross-taxa
  track.

  **Operationally**, in `prospective_results.json`: (a) is
  `tracks.fish_novel.holm_rejected_vs_published_reference.<model>` true; (b) is
  `tracks.fish_novel.holm_rejected_vs_best_baseline.<model>` true; (c) is
  `tracks.fish_novel.models.<model>.vs_published_reference.all_seeds_better` true (or `null` =
  not applicable for a deterministic model) and the same for `vs_best_baseline`; (d) is
  `tracks.fish_novel_strict.models.<model>.vs_published_reference.diff < 0` and the same for
  `vs_best_baseline`. A reader checks exactly those fields.
* **"Competitive"** if the CIs overlap; **"negative"** if worse. Reported as such, with the number
  of configurations tried in the improvement cycle and the selection effects stated (plan §15.5
  guard 2).
* The ADORE result stays reported unchanged; no prospective number is presented as an ADORE number,
  and the two test sets are never pooled.
* Secondary tracks are labelled secondary everywhere. The `temporal` track is labelled
  **underpowered** (29 chemicals) wherever it appears, and `algae_novel` (3 rows / 2 chemicals) is
  labelled **descriptive only, no inference**: its bootstrap p-values are degenerate on 2 clusters,
  so no Holm rejection on that track is reportable as a result.
* No reinterpretation of the primary endpoint after seeing prospective numbers.

## 5. What happens after scoring

Registry (`experiment = chemtox_prospective_test`, split `prospective/<track>/test`), model card
updated from the registry, a dated line in `RESULTS.md`, `scripts/eval/render_results.py` re-run,
and the post-results review gate before any number leaves the repository.

## 6. Hashes (stage 1, re-frozen 2026-10-24 after the review gate)

The pre-training review (`REVIEW_REPORT_PRE_TRAINING.md`, section "M4 SENTINEL-ChemTox v2") found
three blocking problems **before any model had been scored**: a crash path in the scoring script,
the biased feature-spec comparison fixed in §4.3, and the structural-novelty issue now measured in
§1.3a. The data file was rebuilt to carry the `parent_novel`, `parent_key` and
`nn_tanimoto_adore_train` columns, so its hash changed; the curation, the exclusion logic and every
track's size are unchanged (verified against the previous `report.json`). Nothing had been scored on
either version.

| Item | sha256[:16] |
|---|---|
| prospective data file (`3b2fc4398db940a2…`) | `3b2fc4398db940a2` |
| `sentinel/models/chemtox/prospective.py` | `74765aea3243aa1e` |
| `scripts/data/chemtox_build_prospective.py` | `80a617aecd89610a` |
| `scripts/eval/chemtox_cycle_select.py` (the freeze) | `ad55dddd40c8d66e` |
| `scripts/train/chemtox_cycle.py` (the CV runner) | `a1dc897cfeb85dcc` |
| `sentinel/models/chemtox/cycle.py` (deployment masking) | `920d58c258e9cc30` |
| `scripts/eval/chemtox_score_prospective.py` (the sealed scorer) | `5515402fca2f9d5e` |
| `sentinel` package (`registry.package_hash()`; informational — other workstreams edit the shared package continuously, and the authoritative value is the one each registry entry records at scoring time) | `7de259975a2e52f2` |

The previous stage-1 hashes, for the audit trail: data `ea078629fb084e6f`, module
`29d6c5b3ba252b38`, builder `80a617aecd89610a`, package `4ec4e551437950fc`.

Stage 2 (frozen configurations, their CV scores, their hash, and the scoring script's hash at the
moment of scoring) is appended below before the prospective rows are read.

## Stage 2 — frozen model configurations (appended before any prospective row is read)

Frozen by `scripts/eval/chemtox_cycle_select.py --freeze` on ADORE `t-F2F`/`split_occurrence` masked CV folds 0–4 only. **No prospective row had been read, fitted or scored.**

* selection file: `$SENTINEL_RUNS/chemtox/cycle2_select/t-F2F_split_occurrence/selection.json`, sha256 `bf2ed61c971fc1926db4d668c73c16c37752fc27a665e7208f19e2a356237328`
* **frozen-configuration hash** (`--expect-config-hash`): **`630b43e905d9b852`**
* prospective data sha256 (unchanged, stage 1): `3b2fc4398db940a26c5273f815fdfe872a08b620e1f1b36bd4a12bb38dcc54d9`
* code hashes at freeze (sha256[:16]): `sentinel_package` ca39f9ea257fcb1a, `chemtox_score_prospective.py` a0cb0385226e9aaf, `chemtox_cycle.py` 258b00023693e4da, `chemtox_cycle_select.py` f5f70abc16821d19, `prospective.py` 74765aea3243aa1e
* `__published_reference__` = **`rf_published_mwlogp`** (best masked CV of the four published runs: rf_published_mwlogp 0.9909, xgboost_published_mwlogp 0.9945, rf_published 1.2561, xgboost_published 1.4600)
* `__allow_unmasked__` = False

**Tuning parity (lead decision 2026-11-08).** Every family and every training-data variant of it is one 32-configuration fixed-seed search (screen on folds 0–2 with documented caps, top 5 re-run as sampled on all five folds; selection only on full 5-fold scores). Variants are frozen as separate pre-registered names (`lightgbm` / `lightgbm_aug`, `gpboost` / `gpboost_multitaxa`) rather than taking the best over variants, so no entry is the best of more than 32 configurations. `chemprop` is the phase-1 winner (32-configuration phase-1 search) re-run under masked CV in two explicit variants (frozen spec and `mw_logp` block); the better masked one is frozen.

| Name | role | family | extra rows | features | parameters | masked CV RMSE | configs |
|---|---|---|---|---|---|---|---|
| `catboost` | candidate | catboost | none | Morgan, chem=mw_logp, rdkit-full | bootstrap_type=Bernoulli, depth=6, eta=0.05, iterations=1500, l2_leaf_reg=10.0, rsm=1.0, subsample=1.0 | 0.9480 | 32 |
| `chemeleon_gpboost` | candidate | chemeleon_gpboost | none | MACCS, chem=mw_logp, rdkit, chemeleon(pca=0) | feature_fraction=0.6, group_columns=['tax_gs'], lambda_l2=1.0, learning_rate=0.02, min_data_in_leaf=50, num_boost_round=800, num_leaves=127 | 0.9504 | 32 |
| `chemprop` | candidate | chemprop | none | no-fp, chem=mw_logp | batch_size=128, depth=3, dropout=0.0, ffn_hidden=300, ffn_layers=2, init=none, max_epochs=30, max_lr=0.001, message_hidden=200, multitask=False, n_seeds=3, species_dim=16, use_conditioning=False | 0.9527 | 32 |
| `chemeleon_lightgbm` | candidate | chemeleon_lightgbm | none | no-fp, chem=mw_logp, chemeleon(pca=0) | bagging_fraction=1.0, bagging_freq=1, eta=0.01, feature_fraction=0.6, lambda_l2=0.0, min_child_samples=5, n_estimators=3000, num_leaves=127 | 0.9593 | 32 |
| `gpboost_multitaxa` | candidate | gpboost | adore_taxa_all | Morgan, chem=mw_logp, rdkit-full | feature_fraction=0.6, group_columns=['tax_gs', 'tax_genus', 'tax_group'], lambda_l2=5.0, learning_rate=0.05, min_data_in_leaf=50, num_boost_round=800, num_leaves=31 | 0.9669 | 32 |
| `lightgbm_aug` | candidate | lightgbm | ecotox | MACCS, chem=mw_logp, rdkit-full | bagging_fraction=1.0, bagging_freq=1, eta=0.05, feature_fraction=0.3, lambda_l2=0.0, min_child_samples=20, n_estimators=500, num_leaves=31 | 0.9797 | 32 |
| `lightgbm` | candidate | lightgbm | none | MACCS, chem=mw_logp | bagging_fraction=1.0, bagging_freq=1, eta=0.01, feature_fraction=0.3, lambda_l2=0.0, min_child_samples=5, n_estimators=500, num_leaves=255 | 0.9865 | 32 |
| `gpboost` | candidate | gpboost | none | Morgan, chem=mw_logp, rdkit-full | feature_fraction=0.6, group_columns=['tax_gs', 'tax_genus', 'tax_group'], lambda_l2=5.0, learning_rate=0.05, min_data_in_leaf=50, num_boost_round=800, num_leaves=31 | 1.0158 | 32 |
| `chemeleon_tabpfn` | candidate | chemeleon_tabpfn | none | no-fp, chem=mw_logp, rdkit, chemeleon(pca=64) | max_features=300, max_train_rows=9000, n_estimators=2, predict_chunk=2000, subsample_by=row | 1.0427 | 32 |
| `xgboost` | comparator | xgboost | none | Morgan, chem=mw_logp, rdkit-full | colsample_bytree=0.8, eta=0.03, max_depth=6, min_child_weight=5, n_estimators=800, reg_lambda=5.0, subsample=1.0 | 0.9733 | 32 |
| `rf_published_mwlogp` | comparator | rf | none | MACCS, chem=mw_logp | max_depth=20, max_features=0.3, n_estimators=300 | 0.9909 | 8 |
| `rf` | comparator | rf | none | Morgan, chem=mw_logp, rdkit-full | max_depth=40, max_features=sqrt, min_samples_leaf=1, min_samples_split=2, n_estimators=600 | 0.9928 | 32 |
| `xgboost_published_mwlogp` | comparator | xgboost | none | MACCS, chem=mw_logp | eta=0.03, max_depth=6, min_child_weight=1, n_estimators=800, subsample=0.8 | 0.9945 | 32 |
| `readacross` | comparator | readacross | none | MACCS, chem=mw_logp | duration_penalty=0.4, fingerprint=Morgan, k=20, match_species=False, min_neighbours=3, min_similarity=0.0, power=6.0 | 1.1546 | 32 |
| `rf_published` | comparator | rf | none | MACCS, chem=four | max_depth=20, max_features=0.3, n_estimators=300 | 1.2561 | 8 |
| `xgboost_published` | comparator | xgboost | none | MACCS, chem=four | eta=0.03, max_depth=6, min_child_weight=1, n_estimators=800, subsample=0.8 | 1.4600 | 32 |

**Ensemble membership (rule fixed in advance):** every candidate whose masked CV RMSE is below the strongest re-tuned comparator's (`xgboost` 0.9733).

* `ensemble_mean` — unweighted mean; components `catboost`, `chemeleon_gpboost`, `chemeleon_lightgbm`, `chemprop`, `gpboost_multitaxa`; masked CV RMSE 0.9164 (members chosen on the same CV, so this is optimistic). Out-of-fold sources: `catboost` ← `catboost_t-F2F_split_occurrence_extra-none_dm/oof_catboost_000.npy`; `chemeleon_gpboost` ← `chemeleon_gpboost_t-F2F_split_occurrence_extra-none_dm/oof_chemeleon_gpboost_000.npy`; `chemeleon_lightgbm` ← `chemeleon_lightgbm_t-F2F_split_occurrence_extra-none_dm/oof_chemeleon_lightgbm_006.npy`; `chemprop` ← `chemprop_t-F2F_split_occurrence_extra-none_dm_cfg-v1_chemprop_mwlogp/oof_chemprop_903.npy`; `gpboost_multitaxa` ← `gpboost_t-F2F_split_occurrence_extra-adore_taxa_all_dm/oof_gpboost_000.npy`
* `ensemble_stack_nnls` — non-negative least squares on training-portion out-of-fold predictions; components `catboost`, `chemeleon_gpboost`, `chemeleon_lightgbm`, `chemprop`, `gpboost_multitaxa`; masked CV RMSE 0.9212 (members chosen on the same CV, so this is optimistic). Out-of-fold sources: `catboost` ← `catboost_t-F2F_split_occurrence_extra-none_dm/oof_catboost_000.npy`; `chemeleon_gpboost` ← `chemeleon_gpboost_t-F2F_split_occurrence_extra-none_dm/oof_chemeleon_gpboost_000.npy`; `chemeleon_lightgbm` ← `chemeleon_lightgbm_t-F2F_split_occurrence_extra-none_dm/oof_chemeleon_lightgbm_006.npy`; `chemprop` ← `chemprop_t-F2F_split_occurrence_extra-none_dm_cfg-v1_chemprop_mwlogp/oof_chemprop_903.npy`; `gpboost_multitaxa` ← `gpboost_t-F2F_split_occurrence_extra-adore_taxa_all_dm/oof_gpboost_000.npy`

**§4.3 chemical-block decision:** `lightgbm_aug` mw_logp, `lightgbm` mw_logp, `catboost` mw_logp, `gpboost_multitaxa` mw_logp, `gpboost` mw_logp, `chemeleon_lightgbm` mw_logp, `chemeleon_gpboost` mw_logp, `chemeleon_tabpfn` mw_logp, `chemprop` mw_logp. The per-block masked-CV means of every search are in `selection.json` (`families.<name>.cv_rmse_by_chemical_block`).

**Holm family size:** 11 candidates (9 single models + 2 ensembles), for each of the two reference comparisons (§4.2). Comparators: `xgboost`, `rf`, `readacross`, `xgboost_published`, `xgboost_published_mwlogp`, `rf_published`, `rf_published_mwlogp`, plus the trivial baselines and `best_baseline`.

Paired out-of-fold CV comparison against the re-tuned `xgboost` (CV evidence only, chemical cluster bootstrap, 1,000 resamples; **not** a test claim):

| Name | ΔRMSE vs xgboost | 95% CI |
|---|---|---|
| `ensemble_mean` | -0.0619 | [-0.0960, -0.0311] |
| `ensemble_stack_nnls` | -0.0572 | [-0.0929, -0.0270] |
| `chemeleon_gpboost` | -0.0271 | [-0.0783, +0.0228] |
| `catboost` | -0.0268 | [-0.0563, +0.0023] |
| `chemprop` | -0.0236 | [-0.0862, +0.0333] |
| `chemeleon_lightgbm` | -0.0173 | [-0.0547, +0.0243] |
| `gpboost_multitaxa` | -0.0059 | [-0.0434, +0.0279] |
| `lightgbm_aug` | +0.0053 | [-0.0577, +0.0760] |
| `lightgbm` | +0.0092 | [-0.0440, +0.0580] |
| `rf` | +0.0163 | [-0.0197, +0.0511] |
| `gpboost` | +0.0451 | [+0.0152, +0.0763] |
| `chemeleon_tabpfn` | +0.0648 | [+0.0146, +0.1178] |
| `readacross` | +0.1769 | [+0.0989, +0.2511] |

## Stage 2 addendum — pre-pass items W1–W3 (round-4 gate), written 2026-11-08 before any score

Required by `REVIEW_REPORT_PRE_TRAINING.md`, "M4 ChemTox — round-4 pre-test gate (2026-11-08)".
Written **before** the pre-flight and the single pass; no prospective target value had been read
by any code at the time of writing (the side file below was built from identifier and flag columns
only, and from ADORE's own release). No model configuration, frozen hash or metric definition is
changed by this addendum: `630b43e905d9b852` stands.

### §1.3c — W1: half of the primary track is chemicals ADORE's own pipeline saw and dropped

§0/§1's wording "curated with ADORE's own rules … only the chemicals (or the publication dates)
are new" is **corrected here**: the chemicals are new *to the ADORE benchmark*, and for about half
of the rows they are new **because ADORE could not featurise them**, not because they were measured
or published later. Measured from ADORE's release (`processed/ecotox_mortality_processed.parquet`,
the table before ADORE's chemical-level filter) and the prospective file's `result_id`s:

| Track | Rows | Chemicals | Rows also in ADORE's pre-filter table | Chemicals | of those, `organic_multicomponent` rows |
|---|---|---|---|---|---|
| **`fish_novel` (PRIMARY)** | 1,912 | 301 | **977 (51%)** | **117** | 637 |
| `fish_novel_strict` (co-primary) | 1,362 | 236 | 553 (41%) | 79 | 214 |
| `crust_novel` | 230 | 112 | 96 | 50 | 49 |
| `crust_novel_strict` | 154 | 74 | 62 | 36 | 15 |
| `fish_novel_all` | 2,313 | 363 | 1,228 | 153 | 637 |
| `crust_novel_all` | 295 | 136 | 142 | 62 | 49 |
| `temporal` | 115 | 29 | **0** | 0 | 0 |
| `noncarbon` | 8,774 | 225 | 6,883 | 167 | 0 |
| `algae_novel` | 3 | 2 | 1 | 1 | 0 |

These rows had already passed ADORE's row-level rules (effect, endpoint, duration, media,
concentration type) and were dropped when ADORE filtered **chemicals** (3,295 → 2,408). Of the 136
primary chemicals that appear in ADORE's property table, 47 have no experimental water solubility or
melting point and 71 of the remaining 89 are flagged `chem_mol2vec_allowed=False`. So the dominant
exclusion mechanism is *missing chemical features*, which is also the most likely explanation of the
composition shift already recorded in §1.3b (50% `organic_multicomponent` against ADORE's 8.6%).

This is **not** leakage: no file of ADORE's pre-filter pipeline is training data for any model
scored here, the exclusion is a property of ADORE rather than of any SENTINEL model, and every
model is scored on exactly the same rows. It is a selection mechanism that the reader must know
about, because it makes the primary track harder *and* less representative of "chemicals tested
after ADORE" than §0's original wording suggests.

**Frozen, hashed side file** (computed before the pass, from the ADORE release and the prospective
file's identifier/flag columns only):
`docs/preregistration/chemtox_adore_prefilter_seen.json`, sha256
`357571f9271df93d586fff87ad5abde1127d89717d0b09c470b3487ad969d105`. It carries, per track, the
sorted `result_id` list of the affected rows with its own sha256, the affected CAS list, the counts
above, and the sha256 of both source files. Its `adore_processed_sha256` pins the ADORE table the
breakdown was computed from.

**Pre-registered reporting rule.** `adore_prefilter_seen` (true/false) is reported as an additional
breakdown of every model's error on the primary and strict tracks, beside the §4.1 breakdowns, with
cells of ≥ 10 rows. It is computed **after** the pass from the saved `pred_<model>.npy` files and
the frozen `result_id` list above; it is descriptive and **no decision rule depends on it**. A
large difference between the two cells is reported as a limitation of the track's
representativeness, never as a reinterpretation of the primary endpoint (§4.4).

### §4.5 — W2: parent-structure-clustered sensitivity CI

The primary analysis resamples **chemicals** (`test_cas`), as §4.2 pre-registers, and that does not
change. But 49 of the primary track's 301 CAS fall into **18 multi-CAS parent-structure groups**
(salts, counter-ions and stereoisomers of one another; strict track 16 CAS in 7 groups; `crust_novel`
30 CAS in 11 groups), so resampling them independently is mildly anti-conservative.

Pre-registered sensitivity analysis, computed after the pass from the saved predictions, with the
same function (`sentinel.eval.bootstrap.paired_bootstrap`), the same 2,000 resamples and the same
seed 0, but with the cluster variable set to the frozen `parent_key` column: the paired RMSE
difference of every candidate against `published_reference` and against `best_baseline` (the
per-track identities the primary pass recorded), on `fish_novel` and `fish_novel_strict`, with Holm
applied across the same candidate family.

**Decision rule (frozen here):** the §4.4 rules stay exactly as written on the CAS-clustered CIs. A
claim that satisfies §4.4 is additionally reported as **"qualified: not robust to
parent-structure clustering"** whenever the parent-clustered CI of the same comparison includes 0,
and that qualification appears in the same table and the same sentence as the claim. The
parent-clustered CI can never create a claim that §4.4 does not already grant.

### W3 — runbook: protect the primary report before the sensitivity pass

`chemtox_score_prospective.py` writes `prospective_results.json` to a fixed path, so the
`--train-partition-only` sensitivity pass would overwrite the primary pass's report, which is where
§4.4's decision fields live. Fixed by runbook, not by a code change (the scorer's hash stays
`a0cb0385226e9aaf`): **immediately** after the primary pass and **before** any further invocation of
the scorer,

1. copy `prospective_results.json` → `prospective_results_primary.json` and `score.log` →
   `score_primary.log` in `$SENTINEL_RUNS/chemtox/prospective_test/`;
2. record both sha256 values in `RESULTS.md` and the handoff note;
3. `chmod 0444` both copies.

The `_trainpart` entries remain separate registry rows and never overwrite the primary ones.

## Deviation D1 — first single-pass attempt crashed during fitting (recorded 2026-11-08 before the re-run, review item O7)

* **When / where.** Pass started 2026-11-08 15:53:32 EST on GPU 1 (lock helper, cap 0/1) with
  `--i-have-approval --expect-data-hash 3b2fc439…dc54d9 --expect-config-hash 630b43e905d9b852
  --accelerator gpu --threads 2`; W6 checks passed at start (scorer `a0cb0385226e9aaf`, ChemTox-scoped
  `8c5cd6b523a95fed`).
* **What failed.** At 19:58:50 the scorer process died with **rc 139 (segmentation fault)**, a native
  crash with no Python traceback, during the fit of **`chemeleon_tabpfn` seed 1** (seed 0 had
  fitted at 19:58:34). 14 of the 16 single models (all 3 seeds each) had been fitted; not yet fitted:
  `chemeleon_tabpfn` seeds 1–2 and `chemprop`; trivial baselines and ensembles are built after the fits.
* **What did not happen.** The scorer *uses* the prospective target column only after every fit, so
  **no prospective target value was used or output**; **no `prospective_results.json` was written**;
  the registry had **0** `chemtox_prospective_test` rows before and after. Precisely (post-results
  review observation O1): `pd.read_parquet(data_path)` loads **all** columns, so the target column
  was present in memory from the start and the frame was passed to every `builder.transform` and
  `model.predict`; no chemtox model's `predict` reads it, and nothing derived from it was computed,
  logged or written before the crash. The pass is therefore unspent, which is the case review item
  O7 anticipated ("crash during fitting, before any target is read and before any registry write, so
  a re-run is allowed"). For any future sealed pass the target columns should be dropped before
  fitting, so the statement is literally true as well as substantively true.
* **Archive.** The 14 partial `pred_*.npy` files, the crashed `score.log` and the wrapper log are kept
  read-only in `$SENTINEL_RUNS/chemtox/prospective_test/crashed_pass_20261108_1958/` with
  `SHA256SUMS` (sha256 of that file: `0f2af5f61a5161a6c0acc97dc424770ba816966756d982ef033bab4309bc607c`).
* **Re-run (lead-authorised).** Exactly as frozen: same command, same hashes, same frozen block, W6
  checks repeated, no code or configuration change. The only change is operational: the lock
  helper's `--min-free-mb` raised from 8000 to 10000, so the pass starts only on a nearly empty GPU.
* **Determinism check (pre-registered here).** After the re-run, each of the 14 models' predictions
  are compared with the archived ones; the result (identical or not, max absolute difference) is
  reported. A difference is reported as a reproducibility limitation; it does not change which run
  is scored (the re-run is the single scored pass).

### D1 outcome (recorded 2026-11-09, after the scored re-run)

The re-run (GPU 1, 20:29:56 → 01:02:09, W6 hashes re-confirmed at start) completed with rc 0; W3 was
applied at once (`prospective_results_primary.json` sha256
`709803d052dfa9f0c117f3d64af82b55e82616a1bf8dde88460ced6bb686eb0b`, `score_primary.log`
`2e96d089467c95f1c26e483302e6da97ad03e896e7ebad5206e874d0f97686a7`, both 0444). Determinism check
against the 14 archived models: 10 bit-identical (listed below); the three random forests (`rf`, `rf_published`, `rf_published_mwlogp`) differ by ≤ 4.4e-15
(floating-point summation order); **`gpboost_multitaxa` seed 0 differs on every row** (mean |Δ|
0.22, max 1.30 log units; seed-mean |Δ| 0.073, max 0.43), seeds 1–2 bit-identical. Bit-identical:
`xgboost`, `xgboost_published`, `xgboost_published_mwlogp`, `readacross`, `lightgbm`,
`lightgbm_aug`, `catboost`, `gpboost`, `chemeleon_lightgbm`, `chemeleon_gpboost`. Reported as a
reproducibility limitation of `gpboost_multitaxa`; the scored pass is the re-run, as fixed above.

**Cause, diagnosed 2026-11-09 without re-scoring** (frozen configuration, CPU only, ADORE CV
fold 0 and the pass's own 33,448 training rows, predictions taken on ADORE/ECOTOX *training* rows;
scripts in the M4 scratchpad `gpbdiag/`):

| Test | Result |
|---|---|
| Two fits of seed 0 in **one** process, 2 boosting threads (the pass's setting), fold 0 | **bit-identical** |
| Same, in a **second** process, fold 0 | bit-identical to the first (4 fits agree) |
| Seed 0 at **6** boosting threads, fold 0 | differs from 2 threads (max 0.5); two 6-thread fits also differ from each other |
| OpenMP threads 6 with boosting threads 2, fold 0 | identical to 2/2 → OpenMP is not the lever |
| Two fits on the **pass's full 33,448 training rows**, 2 threads, one process | **bit-identical** |
| The same fit in a **second** process | **differs, max 0.605** ← reproduces the pass's signature |
| Two such fits in separate processes with `PYTHONHASHSEED=0` | bit-identical (2 of 2) |

So: the fit is deterministic **within** a process and at the smaller fold-0 size, but intermittently
**non-deterministic across processes at the pass's data size**, which is exactly how the two passes
differed. Ruled out: the GPU path (GPBoost is CPU-only), a first-fit-in-process effect, OpenMP
threads, and the thread count itself (both passes ran at `--threads 2`). Thread count *does* change
the result and ≥ 6 threads looks non-deterministic, which is a separate reproducibility caveat for
any future run at a different thread budget.

**The root cause is not established.** Fixing `PYTHONHASHSEED` removed the cross-process difference
in both trials, which is consistent with an iteration order over hashed strings (the taxonomy group
labels are strings) feeding the fit, but with one observed differing pair this is **suggestive, not
proven** — and `GPBOOST_DEFAULTS` passes `seed` only to the boosting stage, while `gpb.GPModel`
(the random-effects component, whose variance parameters are fitted by an iterative solver) is
constructed with no seed, which remains the leading candidate. Recorded as **unexplained**, with
these facts, rather than as a diagnosis. Any future use of this family should set `PYTHONHASHSEED`,
fix the thread budget, and record both.
**Both ensembles contain this model and inherit the irreproducibility** (effect ≤ 0.002 RMSE on the
primary track: stack 1.2079 → 1.2065 with the crashed seed 0). On `crust_novel` the scored seed 0
is the worst of the three (1.379 against 1.279 / 1.311), so the reported number is conservative.

## Deviation D2 — the `_trainpart` sensitivity pass failed; sensitivity entry not produced (2026-11-09)

The §2 sensitivity entry (every frozen model refit on the official training partition only, 21,981
ADORE rows) was run once on 2026-11-09 01:59:06 EST on GPU 1, lead-approved as a one-off exception
to the 2-hour GPU rule, with the identical frozen block and hashes (W6 re-confirmed at start:
scorer `a0cb0385226e9aaf`, ChemTox-scoped `8c5cd6b523a95fed`).

**It failed at 04:06:08 with rc 1** — `GPBoostError: There was Nan or Inf value generated in the
Conjugate Gradient Method!` while fitting **`gpboost` seed 2** (11 of 16 single models done).
This is the same GPBoost CG failure that killed 2 of 5 `gpboost` finalists during cycle-2 CV, now
on the smaller training partition; review item O7 anticipated it as a crash during fitting, before
any target is read and before any registry write.

* **No sensitivity result exists.** 0 registry rows with a `_trainpart` model name; no
  `prospective_results.json` of this pass (the scorer writes it only after all tracks are scored);
  the exception propagated from `fit_predict` before the target column was used.
* **The primary pass is untouched.** `prospective_results_primary.json` still has sha256
  `709803d052dfa9f0…` and all 18 primary `pred_*.npy` verify against
  `pred_primary_SHA256SUMS.txt` (18/18 OK); the crashed-attempt archive verifies 16/16.
* **Archive.** The 11 partial `pred_*_trainpart.npy`, the log through the failure and the wrapper log
  are read-only in `$SENTINEL_RUNS/chemtox/prospective_test/failed_trainpart_20261109_0406/` with
  `SHA256SUMS` (sha256 `793f01190cbcbb21`).
* **Status: NOT OBTAINED, and it will not be retried (lead decision, 2026-11-09).** The
  pre-registered §2 `_trainpart` sensitivity entry does not exist: the pass failed with a GPBoost
  conjugate-gradient NaN/Inf on the training partition at `gpboost` seed 2, after 11 of 16 single
  models; the 11 partial predictions are archived and **unscored**. It is deliberately **not**
  re-run: re-rolling a non-deterministic fit failure until it happens to succeed is a forking path,
  and a skip-on-failure variant would change the frozen scorer. So the plan's sensitivity analysis
  of "what the extra training rows did" is simply missing, and is reported as missing.
* **Reported instead, as a result in its own right: `gpboost` is unstable on the smaller training
  partition.** The same conjugate-gradient failure killed 2 of its 5 cycle-2 CV finalists, and it
  now prevents the model from being fitted at all on the 21,981-row partition, while the same
  configuration fits on the 26,114-row pooled set. Together with the `gpboost_multitaxa` seed-0
  cross-process irreproducibility (D1), this is a **stability finding about the GPBoost family**,
  not a missing number to be filled in later.
* The primary and secondary results are **unaffected**: they were produced by the completed pass,
  whose seals were re-verified after this failure (18/18 primary predictions, 16/16 crashed archive,
  `prospective_results_primary.json` sha256 unchanged).
