# Pre-registration — SENTINEL-ChemTox (M4) **unseen-taxa** test blocks

**Status: stage 1 re-frozen 2026-11-09 (fourth revision, `prereg_snapshot_v6`: while running the
§5.1(3) determinism check that gates GPBoost's admissibility, the check helper itself was found
broken — `SeededGPBoost.determinism_check` predicted with the *training* matrix while passing the
validation frame's group data, which GPBoost rejects — so it now takes a matched
`X_pred`/`predict_rows` pair with a length assertion. The helper is used **only** for the
admissibility check and never in a fit that produces a score; no rule, feature, fold, number or
datum changed. Third revision, `prereg_snapshot_v5`: the two cosmetic
leftovers named in the confirmation note — the garbled comparator-count sentence and the
`ICERegressionBaseline` docstring — corrected before any fit; no rule, feature, fold or number
changed). Second revision: the cross-taxon channel amendment of
§1.4 was added on the lead's decision, before any number existed), originally re-frozen 2026-11-09 after the pre-registration review returned FAIL
(`REVIEW_REPORT_PRE_TRAINING.md`, "M4 ChemTox unseen-taxa — pre-registration review"). The data
freeze was found clean and is unchanged; this revision closes the six critical design findings
C-M4U-1…6 and the warnings W-M4U-1…7. Nothing has been fitted, selected or scored on these blocks,
and no model exists for this track.** A delta review of this revision is required before stage 2. The builder
(`scripts/data/chemtox_build_unseen_taxa.py`) fits nothing. Stage 2 (frozen model configurations)
is appended below before any test row is scored, and only after an independent pre-test review.

This is a **new, harder track** under RESEARCH_PLAN §15.5 items 5–7. It **does not replace or
re-label** the closed prospective result (`docs/preregistration/chemtox_prospective_plan.md`;
**competitive, not SOTA**, with the `crust_novel` result reported as a data-coverage effect). That
result stays reported exactly as it is, and every claim here names this track.

## 0. Why this track, and why these blocks

The prospective pass answered "new chemicals, same taxa" and the answer was *competitive*. The
weakness it exposed is taxonomic: the one Holm-rejected result came from the only model that had
seen crustacean rows, which measured training coverage rather than method quality. The harder
question, and the one a deployed hazard model actually faces, is a taxon the model has **never**
seen. ADORE covers fish, crustaceans and algae only.

ECOTOX, curated with ADORE's own acute-lethality rules, also contains **insects**, **molluscs** and
**amphibians** — whole taxonomic classes absent from ADORE and from every SENTINEL ChemTox model
fitted so far. Each is a separate test block.

## 1. Hold-out design (both parts decided before any fit)

### 1.1 Taxon hold-out at **class** level — the primary hold-out, deliberately conservative

Every row whose ECOTOX `class` is one of **`Insecta`, `Gastropoda`, `Bivalvia`, `Polyplacophora`,
`Cephalopoda`, `Amphibia`** is removed from the training pool of **every** model — candidates,
comparators, trivial baselines and any auxiliary/augmentation source — and from everything used for
selection. `sentinel.models.chemtox.unseen_taxa.assert_no_test_classes` raises on any violation and
is called on the pool inside the builder and must be called on every training frame of the pass.

**Rationale for class-level rather than species-level hold-out (the lead's instruction, and the
conservative choice).** Holding out species *inside* a class would leave congeners of the test
species in training — a model could then interpolate within a well-sampled genus and the block
would measure within-class generalisation, not cross-taxon extrapolation. Holding out the whole
class removes that path: nothing of the test lineage is available at any taxonomic level. It makes
the task strictly harder and the result strictly less flattering, which is the point.

Consequence, stated in advance: the taxonomy feature block (ADORE's Add-my-Pet traits, keyed per
species) is **undefined** for every test species here. **Decision: the trait block is dropped for
all models on this track**, so every model — candidate and comparator alike — is evaluated with
structure, chemistry and exposure features only.

**How the drop is implemented (C-M4U-1).** By
`sentinel.models.chemtox.unseen_taxa.TaxonBlindBuilder`, a wrapper around the existing
`FeatureBuilder` that deletes every column whose name begins with one of
`unseen_taxa.TRAIT_PREFIXES` (`tax_eco_*`, `tax_lh_*`, `tax_ps_*`, `tax_pdm*`, `tax_gs`,
`tax_group`, `tax_name`, `tax_all`, `tax_class`, `tax_order`, `tax_family`, `tax_genus`,
`tax_species`, `species_number`, `latin_name`) from the matrix **and** from `names_`. Every model
on this track is fitted and scored through it, and
`unseen_taxa.assert_no_trait_columns(builder.names_)` is called for every model in the pass;
`tests/test_chemtox_unseen_taxa.py` asserts that no trait column survives. The wrapper exists
because `FeatureSpec.published` is all-or-nothing (traits in, or traits *and* the exposure block
out) and `features.py` may not be edited while another workstream's frozen pass hashes it.

**Why the drop is not optional, i.e. what the naive default would have done.** ADORE's traits do
not exist for a species outside ADORE, so left in place every test row would take the *training
mean* for each continuous trait and an **all-zero** multi-hot — one identical, fish-derived
surrogate trait vector for an entire test block, fed to models that split on real trait values.
That is a systematic train→test shift unique to the test rows, not a neutral imputation, and it
would have been invisible in the metrics. Dropping the block weakens every model equally and keeps
the comparison a model comparison; sourcing traits for ~600 new species is a licence/QC question
and a different protocol.

### 1.2 Chemical blocking — a **reported stratum**, not an exclusion

A test chemical may legitimately appear in the training pool, measured on fish. That is realistic
and is not leakage of the test rows. So chemical novelty is **reported** rather than enforced:

* `chem_novel` — the chemical is absent from the training pool by CAS **and** InChIKey;
* `parent_novel` — additionally novel at parent-structure level (salt-, charge- and
  stereochemistry-insensitive, the `prospective.parent_key` function used in the closed track).

Both are stored in the frozen file, so the strata are fixed by the data. Row identity is enforced:
`assert_no_test_rows` guarantees no test `result_id` is ever a training row.

### 1.3 Curation

Identical to the rules recovered from ADORE in the prospective plan §1.1: effect `MOR`; endpoint
`LC50`/`EC50`; duration exactly {24, 48, 72, 96} h; media `FW`/`SW`; concentration type
`A`/`F`/`T`; uncensored (`~` kept); molar target `log10(mol/L)`; an RDKit-parseable structure; one
row per ECOTOX `result_id`. 471,097 aquatic rows → **76,319** curated rows.

**Feature spec and schema adapter, frozen here (W-M4U-5).** The frozen files carry neither
`chem_ws` nor `chem_mp` (ADORE's experimental water solubility and melting point), so the pipeline
default `FeatureSpec.chemical = "four"` would raise. This track pre-registers
**`chemical = "mw_logp"`** for every model, as the closed prospective track also concluded from
masked CV. The ECOTOX→ADORE column adapter is
`sentinel.models.chemtox.unseen_taxa.to_model_schema` (named and hashed in stage 2); it maps
`duration_h`, `media_type_norm`, `conc1_type_norm`, `exposure_type_norm`, `chem_structure_class`
and the target to the names the feature blocks expect, and it adds no taxonomy column.

The **primary endpoint of each block** is its `in_adore_domain` subset (`structure_class ∈
{organic, organic_multicomponent}`), as in the closed track: organometallics and metal salts are a
different, out-of-domain question and are reported as a labelled secondary stratum.

## 1.4 Amendment (2026-11-09, before any fit) — two pre-declared cross-taxon channels

§9 of the first revision stated that no model on this track had any route by which taxon
information could reach it, while the surrogate/ICE comparators did, so the track was close to
unwinnable by construction. On the lead's decision, and **before any number of this track
existed**, two channels are added. Both are declared here in full, with their fairness rule, their
leakage properties and their implementation, so nothing about them can be chosen after seeing a
score.

### 1.4.1 Channel A — chemical-matched surrogate information as features

`sentinel.models.chemtox.unseen_taxa.SurrogateFeatures` adds **seven** per-row columns, all
computed **only from training-pool rows**:

| Feature | Meaning |
|---|---|
| `surr_fish_geomean` | the chemical's mean log10 value over the pool's **fish** rows |
| `surr_fish_most_sensitive` | the same, minimum (the conservative regulatory reduction) |
| `surr_pool_geomean` | the same over **all** pool rows (fish + crustaceans + algae) |
| `surr_ice_prediction` | the ICE regression's prediction, `a + b · surrogate`, with `a`,`b` fitted on the pool's own cross-group pairs |
| `surr_available` | 1 if the chemical has any pool record, else 0 |
| `surr_n_records` | number of pool records for the chemical |
| `surr_n_pool_taxa` | number of distinct pool species measuring the chemical |

This hands the models exactly the quantity the surrogate/ICE comparators use, so a learner can
weigh it against structure instead of being denied the channel.

**The one frozen definition of "surrogate available" (delta review D-3).** A row is
surrogate-available **iff its chemical has any record in the training pool, matched by CAS or,
failing that, by `parent_key`** — the rule implemented in `unseen_taxa.surrogate_available` and
used, without exception, by `surr_available`, by the surrogate/ICE baselines' fallback, by
`surrogate_coverage` and by the claim-gating subset of §6(c). The first revision of this plan
quoted a *fish-only* coverage instead, which did not match the implementation; that discrepancy is
removed. Measured on the frozen blocks under the single rule (with `parent_key` now derived for the
pool, see §4):

| Block | Primary rows | Surrogate-available | Fraction | Chemicals | (fish-only, reported for transparency, gates nothing) |
|---|---|---|---|---|---|
| insects | 7,165 | 6,776 | **94.6 %** | 405 | 90.8 % |
| molluscs | 2,172 | 2,021 | **93.0 %** | 297 | 89.6 % |
| amphibians | 2,084 | 1,826 | **87.6 %** | 342 | 82.2 % |

A chemical with no pool record gets `surr_available = 0`, zero records and the pool mean; that
subset is reported separately, as §5.1(4) requires.

**Leakage properties (tested).** `fit` reads only the training frame and asserts the class hold-out
on it (`SurrogateFeatures.fit` → `assert_no_test_classes`), so it cannot be fitted on a frame that
contains test rows; `transform` is a pure lookup keyed on `test_cas`/`parent_key`. The test suite
proves that (a) overwriting every test target leaves all seven features bit-identical, (b) fitting
on pool + test rows raises, (c) an unmatched chemical gets `surr_available = 0`, zero records and
the pool mean, and (d) a hand-computed fish geomean/minimum equals the feature.

**Residual-learning candidate variant.** `unseen_taxa.ResidualSurrogateModel` predicts
`surrogate/ICE value + learned correction`: the wrapped learner is fitted on `y − base` and the base
is added back at prediction time. Pre-registered variants: `*_resid_ice` (base = ICE prediction)
and `*_resid_surrogate` (base = fish geomean). The base is computed by `SurrogateFeatures`, i.e.
from pool rows only.

### 1.4.2 Channel B — taxonomy **strictly above** the held-out class

`unseen_taxa.HigherRankFeatures` one-hot-encodes three ranks above class level —
`rank_phylum`, `rank_subphylum` (subphylum/superclass) and `rank_clade`. The mapping
`HIGHER_RANKS` is a **pure function of the ECOTOX `class` name**, so no feature *name* can spell a
class, order, family, genus or species, and the frozen files are unchanged (they carry no phylum
column and none was added).

**That alone is not sufficient, and the first wording of this section overstated it (delta review
D-2).** Four of the declared levels are identically zero in the pool and non-zero only on test
rows — `rank_phylum=Mollusca`, `rank_subphylum=Hexapoda`, `rank_subphylum=Conchifera`,
`rank_subphylum=Aculifera` — so *within this dataset* each is a one-to-one proxy for a held-out
class. No target information leaks (a column that is constant in training cannot be fitted), but
such a column is perfectly collinear with "is a test row", and for a neural model a weight on an
always-zero input keeps its random initialisation and then multiplies 1 on test rows only: an
arbitrary, seed-dependent, test-only offset that validation cannot reveal. That is the same defect
§1.1 rejects for the all-zero trait multi-hot.

**Frozen rule:** `HigherRankFeatures.fit` keeps only the levels that are **non-constant in the
fitting frame**, identically in every validation fold and at the final fit; a value whose level was
dropped contributes zeros across that rank. On the full pool this retains
**9 of 13** columns — `rank_phylum ∈ {Arthropoda, Chordata, Other}`,
`rank_subphylum ∈ {Crustacea, Vertebrata, Other}`, `rank_clade ∈ {Protostomia, Deuterostomia,
Other}` — and drops exactly the four proxies above. A test asserts that every retained level is
non-constant in the training frame.

**What each block therefore actually gets — corrected:**

| Block | Retained Channel-B signal relative to the pool | Honest reading |
|---|---|---|
| amphibians (`Amphibia`) | `rank_phylum=Chordata`, `rank_subphylum=Vertebrata`, `rank_clade=Deuterostomia` — **bit-identical to the fish vector** | Channel B gives amphibians a *relabelling as "fish-like"*, not new information. If this helps, it is because fish are the right surrogate lineage |
| insects (`Insecta`) | `rank_phylum=Arthropoda` and `rank_clade=Protostomia`, shared with the crustacean pool; `Hexapoda` is **dropped** | insects are represented as "an arthropod, not a crustacean-in-particular" — i.e. the crustacean vector minus `Crustacea` |
| molluscs (`Gastropoda`, `Bivalvia`, …) | `rank_clade=Protostomia` only; `Mollusca`, `Conchifera`, `Aculifera` all **dropped** | molluscs get **almost nothing** beyond "protostome", so Channel B should help them least — a pre-registered expectation, not a post-hoc explanation |

**Guard.** `unseen_taxa.assert_no_class_or_lower(names)` raises if a feature name carries a
class-or-lower token (the three `rank_*` names are the only exemption), and `ChannelBuilder.fit`
calls it with `assert_no_trait_columns` for **every** model. Both are **name** checks, and a name
check is defeated by a rename — which is why the constant-level rule above, and the
content invariant of §1.4.1 (overwriting every test target leaves every channel feature
bit-identical), carry the actual guarantee. `surr_n_pool_taxa` is so named because the earlier
`surr_n_species` tripped the name guard although the feature is a pool-side count that says nothing
about the test row's taxonomy; the rename is cosmetic and is recorded here for the audit trail.

### 1.4.3 Fairness — the channels are given to the comparators too (the C-M4P-1 lesson)

**Every channel given to a SENTINEL candidate is given to the mandatory comparators**, and those
channel-matched variants are themselves strongest-baseline comparators. Added to §5.1 as mandatory:

* `rf_chan`, `xgboost_chan`, `rf_multitaxa_chan`, `xgboost_multitaxa_chan` — the ADORE trees with
  the same Channel A + Channel B features;
* `rf_published_chan`, `xgboost_published_chan` — the published configurations likewise;
* `gpboost_published_chan` — the published cross-species method likewise.

**Claim rule, tightened accordingly:** a SOTA/first-in-class claim additionally requires the
candidate to beat the **channel-matched comparators** under §15.1, not only the channel-free ones
and the surrogate/ICE baselines. A candidate that wins only because it alone received a channel is,
by this rule, not a winner — which is exactly the defect C-M4P-1 recorded on the closed track.

### 1.4.4 Selection and configuration count

The channels are a **feature-set option inside the existing budget**, not extra budget: each
family's 32-configuration search samples `channels ∈ {none, surrogate, higher_rank, both}` as a
feature dimension. Selection stays on §4's leave-one-class-out rotation, and inside each fold the
surrogate block is re-fitted **on that fold's training classes only**, so a fold's surrogate
coverage is what test-time coverage will be (a fold that holds out Actinopterygii has no fish
surrogate, which is the honest analogue of a test block having no mollusc surrogate).

**Configuration inventory (one table; delta review W-D-3).** Channels are a feature dimension
inside each family's existing 32-configuration budget, not extra budget.

| Group | Entries | Searched (32 each) | Fixed |
|---|---|---|---|
| Candidates: `chemprop` (single-task), `lightgbm`, `catboost`, `chemeleon_lightgbm`, `gpboost` (SeededGPBoost), `*_resid_ice`, `*_resid_surrogate` | 7 | 224 | — |
| ADORE comparators re-tuned: `rf`, `xgboost`, `rf_multitaxa`, `xgboost_multitaxa` | 4 | 128 | — |
| Channel-matched comparators (§1.4.3): `rf_chan`, `xgboost_chan`, `rf_multitaxa_chan`, `xgboost_multitaxa_chan`, `rf_published_chan`, `xgboost_published_chan`, `gpboost_published_chan` | 7 | 224 | — |
| `readacross` | 1 | 32 | — |
| Published, fixed configurations: `rf_published`, `xgboost_published`, `rf_published_mwlogp`, `xgboost_published_mwlogp`, `gpboost_published` | 5 | — | 5 |
| Surrogate/ICE baselines: `fish_surrogate_geomean`, `fish_surrogate_most_sensitive`, `fish_surrogate_geomean_both`, `ice_regression` | 4 | — | 4 |
| Trivial: `global_mean`, `species_mean`, `species_duration_mean` | 3 | — | 3 |
| **Total** | **31 model entries** | **608** | **12** |

Ensembles add no configurations of their own (they combine selected members). Of the 31 model
entries, **21 are mandatory comparators** = 4 re-tuned ADORE trees + 7 channel-matched + 5 published
fixed-configuration + `readacross` + 4 surrogate/ICE baselines; the remaining 10 are 7 candidates
and 3 trivial baselines. The numbers reported with any claim are **608 searched + 12 fixed
configurations across 31 model entries**; the registry `prereg_snapshot_v4`/`v5` carries the same three figures so the count cannot drift.

## 2. The frozen blocks and their power (computed before any model exists)

| Block (classes) | Rows (all) | **Primary rows** | Chemicals | Species | Refs | Target SD | Replicate floor | `chem_novel` rows / chem | `parent_novel` rows / chem |
|---|---|---|---|---|---|---|---|---|---|
| **insects** (`Insecta`) | 8,619 | **7,165** | 571 | 332 | 590 | 1.733 | **0.649** (894 groups) | 477 / 190 | 389 / 166 |
| **molluscs** (`Gastropoda`, `Bivalvia`, `Polyplacophora`, `Cephalopoda`) | 4,430 | **2,172** | 363 | 163 | 288 | 1.361 | **0.437** (313 groups) | 168 / 74 | 151 / 66 |
| **amphibians** (`Amphibia`) | 2,677 | **2,084** | 462 | 107 | 253 | 1.711 | **0.402** (366 groups) | 307 / 134 | 258 / 120 |

**Training pool:** 60,575 rows, 3,018 chemicals, 1,080 species, groups {fish, crustaceans, algae},
with every held-out class removed and verified absent.

**Power.** The resampling unit is the **chemical**. Each block's primary endpoint has more
chemicals than the closed track's primary endpoint (301): insects 571, amphibians 462, molluscs
363. Insects and amphibians also exceed it in rows.

**Minimum detectable effect, computed rather than asserted (W-M4U-2).** `unseen_taxa.mde` gives
two synthetic predictors errors of the magnitude the closed track produced (1.2 log units) and
resamples each block's chemical clusters; the half-width of the paired-difference interval is the
smallest RMSE difference the block can resolve:

| Block | Clusters | MDE (correlated errors, ρ = 0.9) | MDE (independent errors, ρ = 0) | Replicate floor | **Floor for a claim** |
|---|---|---|---|---|---|
| insects | 571 | 0.011 | 0.026 | **0.649** | **0.649** |
| molluscs | 363 | 0.019 | 0.049 | 0.437 | **0.437** |
| amphibians | 462 | 0.021 | 0.053 | 0.402 | **0.402** |

Two different noise scales must not be conflated (delta review W-D-7). The **MDE** is the
precision of a *paired difference* and is what gates a claim: **no claim may rest on an RMSE
difference smaller than its block's MDE**, and a difference below it is reported as "inside the
measurement noise" whatever its CI says. The **replicate floor** bounds the *absolute* RMSE any
model can attain on that block; it is reported as context for every absolute error (and is why an
insect RMSE near 0.65 would already be near-irreducible), but it is **not** a threshold a
difference must clear.

**Study structure (W-M4U-1).** 359 `reference_number` values occur in both the pool and the test
blocks, so **21 % of insect, 29 % of mollusc and 16 % of amphibian** primary rows come from a
publication that also contributed pool rows, and each block is reference-concentrated (largest
single reference: insects 271 rows, molluscs 394, amphibians 92; top-10 references 25-33 %). This
is not row leakage — identity, species and lineage are disjoint — but same-lab/same-protocol
structure is an optimism channel that a chemical-clustered bootstrap does not absorb. Therefore:

* a **reference-clustered CI is co-primary** with the chemical-clustered CI for every primary
  comparison (`unseen_taxa.shared_reference_stats` records the structure; the CI uses
  `reference_number` as the bootstrap cluster);
* a **"no shared reference" stratum** is reported per block (rows whose publication contributed no
  pool row);
* a claim whose sign or significance differs between the two clusterings is reported as
  **qualified**, exactly as the closed track's parent-structure sensitivity was.

**Pre-declared power limits, so no claim rests on an underpowered stratum.** The `chem_novel` and
`parent_novel` strata are **reported but underpowered** (66-190 chemicals) and labelled as such;
molluscs' `chem_novel` stratum additionally has a much narrower target spread (SD 0.738 against
1.361 overall), so a difference there is not comparable to the primary endpoint.

**Block composition caveats (W-M4U-7).** "Molluscs" is four classes with a 96 % / 4 % split:
the primary endpoint is Gastropoda 1,414 rows + Bivalvia 757 + Cephalopoda 1, and only 2,172 of
4,430 mollusc rows (49 %) are in-domain, against 83 % for insects and 78 % for amphibians. Any
mollusc statement is worded **Gastropoda + Bivalvia**, and a per-class macro breakdown is reported.
The three blocks are **not** independent samples of "unseen-taxa difficulty" — insects are the
arthropod neighbours of the crustacean pool, amphibians the vertebrate neighbours of the fish pool —
so RMSE is never averaged across blocks and every claim names its block.

## 3. Frozen artefacts (stage 1)

| Item | Value |
|---|---|
| Test blocks | `$SENTINEL_DATA/processed/chemtox_unseen_taxa/test_blocks.parquet`, 15,726 rows |
| **sha256** | **`649dc7d11587ea2bc09fac4672ecdf554a3eb359f13e783efd44fbccce3e9463`** |
| Training pool | `.../training_pool.parquet`, sha256 `7e915fbe9a274afe67ef4fdab05fc23b7d88b3c89659d1e06fcdd2d334b5f956` |
| Report | `.../report.json` · Manifest | `manifests/chemtox_unseen_taxa.json` |
| Source | `processed/ecotox/ecotox_aquatic.parquet` (release 2026-09-15), sha256 `beec1f81769aecad…` |
| Module | `sentinel/models/chemtox/unseen_taxa.py` sha256 `1b2b0617f5617280…` |
| Builder | `scripts/data/chemtox_build_unseen_taxa.py` sha256 `7f3e4effeb6cc8b3…` |
| Environment | `/home/bcheng/miniconda3/envs/sentinel2/bin/python` (pandas 2.3.3, pyarrow 26.0.0, rdkit 2025.09.3) |

No file of this track was written after the blocks were frozen, and a registry `prereg_snapshot`
row records the freeze time and these hashes as ordering evidence.

## 4. Selection — validation only, never a test block

Hyper-parameters are selected **exclusively** inside the training pool, which contains no row of
any held-out class.

**Leave-one-class-out rotation (W-M4U-4).** The stage-1 draft used a single fish ↔ crustacean pair,
which is a thin basis for 32 configurations × ~8 families — and, since algae are only **100 of
60,575 pool rows (0.17 %)** by `taxon_group` (97 of them carry an algal ECOTOX `class`; the
remainder have a missing class and an ADORE group, see §1.2), "crustaceans + algae" is in practice
crustaceans.
`unseen_taxa.validation_splits` instead rotates over the four pool classes with ≥ 1,000 rows,
mirroring the class-level test hold-out.

**`parent_key` is derived for the pool before any fold is built (delta review D-1).** The frozen
`training_pool.parquet` carries no `parent_key` column — only the test blocks do — so the
parent-structure blocking this section credited was a **silent no-op** in the first revision, and
the surrogate block's parent-level fallback never fired. `unseen_taxa.add_parent_key` now derives
it from the frozen file's own canonical SMILES (no parquet is rebuilt), `to_model_schema` applies
it, and `validation_splits` **raises** if it is absent rather than skipping the blocking.
`require_parent_key` enforces that. With parent-structure blocking actually in force the folds are
smaller than the numbers the first revision quoted:

| Validation fold (held-out class) | Train rows (CAS + InChIKey + `parent_key` blocking) | Validation rows | first-revision train rows (CAS/InChIKey only) |
|---|---|---|---|
| Actinopterygii (ray-finned fish) | **676** | 42,533 | 838 |
| Malacostraca (amphipods, decapods…) | **8,378** | 9,386 | 9,724 |
| Branchiopoda (daphnids) | **11,523** | 7,085 | 12,750 |
| Maxillopoda (copepods) | **24,505** | 1,191 | 26,806 |

Parent-key blocking removes a further 162-2,301 training rows per fold, i.e. it was doing nothing
before and now bites. A test asserts that the parent-key step removes rows that CAS/InChIKey
blocking alone would have kept.

**Chemical blocking inside each fold (W-M4U-3).** The held-out class's chemicals are removed from
that fold's training rows by CAS, InChIKey **and** `parent_key`, so a salt, counter-ion or
stereoisomer cannot make a selection fold easier than the test blocks it mirrors. Every fold is
therefore simultaneously cross-class and cross-chemical.

**Selection score (frozen).** The **mean RMSE over the four folds**, with the **worse half**
reported beside it; ties are broken by the worst fold (the conservative direction). The
Actinopterygii fold trains on 838 rows and is the hardest; it is kept because it is the only fold
that validates fish-out-of-sample, and its weight is equal to the others' by construction of the
mean. **All four per-fold scores and the spread of the top five configurations are reported for
every family**, so selection noise is visible rather than hidden behind one number.

* Equal budget: **32 sampled configurations per family**, fixed seed, as in cycle 2; expensive
  families may be screened on two folds and the top five re-run on all four, with only full scores
  used for selection.
* No test block is loaded, read or touched during selection. The intended choke point is
  `unseen_taxa_guard.load_test_block`, which requires an authorisation marker that does not exist
  during selection (§7); honestly stated, that is a **convention plus a 0444 file**, not a
  technical impossibility — a direct `read_parquet` is not prevented by the interpreter. What the
  mechanism does guarantee is that an authorised look is recorded, single, and detectable after the
  fact (registry refusal, `O_EXCL` marker, write-once predictions, sealed outputs).

## 5. Models (names frozen here; configurations in stage 2)

### 5.1 Mandatory comparators — all trained on exactly the same rows as every candidate

1. `rf_published`, `xgboost_published` and their `mw_logp` variants — the published ADORE reference
   (Gasser et al. 2024). The §15.1(i) reference among them is the variant with the best validation
   score, fixed in the hashed block before the pass. **Note, stated in advance:** ADORE is a
   *within-taxa* benchmark, so this is the published reference for the *feature set*, not "the
   strongest published method for cross-taxon prediction" — that is item 3.
2. `rf`, `xgboost` — re-tuned with the same 32-configuration budget.
3. **`gpboost_published` — mandatory published comparator (C-M4U-3).** GPBoost (boosted trees +
   grouped random effects) is the published method for cross-chemical **and cross-species**
   sensitivity prediction (Zubrod et al., *Environ. Int.* 186:108607, 2024) and is benchmarked as
   the best 2D-structure-based cross-species model (Yuan et al., *ET&C* 45(9), 2026). It is a
   comparator here, not an optional candidate: without it this track could not support a §15.1
   SOTA claim at all.
   * **Reproducibility precondition, enforced in new code.**
     `sentinel.models.chemtox.unseen_taxa.SeededGPBoost` seeds `gpb.GPModel(seed=…)` (the in-tree
     wrapper seeds only the boosting stage), pins `num_parallel_threads` to the thread budget and
     uses `matrix_inversion_method="cholesky"` instead of the conjugate-gradient solver that
     produced the NaN/Inf failures. Stage 2 records a determinism check: two in-process fits
     (`SeededGPBoost.determinism_check`) **and** two fits in separate processes with
     `PYTHONHASHSEED=0`, both required to be bit-identical.
   * **If it still fails to fit (frozen rule).** The failure is **reported as a failure** — the
     model, the error, the data it failed on — and the comparison against it is recorded as "not
     available: the published cross-species method could not be fitted", never silently dropped and
     never replaced by a weaker comparator. A candidate cannot earn a SOTA label against a
     comparator that failed: in that case the strongest *available* published comparator sets the
     bar and the claim is downgraded to "competitive" with the failure stated beside it. This is
     the D2 lesson of the closed track, pre-registered rather than improvised.
4. **Chemical-matched surrogate baselines — mandatory (C-M4U-2).** Because the hold-out is at class
   level, the pool shares no species, genus, family or order with any test block, so
   `species_mean`, `species_duration_mean` and `readacross`'s species/group levels all collapse to
   a constant. The comparator that *does* transfer is the regulatory one: carry the same chemical's
   measured value across from a surrogate taxon.
   * `fish_surrogate_geomean` and `fish_surrogate_most_sensitive`
     (`unseen_taxa.FishSurrogateBaseline`): the chemical's pool value, reduced by geometric mean
     (= arithmetic mean in log10 space) or by the minimum (the most-sensitive, conservative
     regulatory choice). A `_both` variant additionally uses crustacean rows as surrogates.
   * `ice_regression` (`unseen_taxa.ICERegressionBaseline`): **interspecies correlation estimation**
     in EPA Web-ICE's own form, `log10(y_target) = a + b · log10(y_surrogate)`, with slope and
     intercept re-fitted on the pool's own pairs: the surrogate is the chemical's **fish** value and
     the response is **every non-fish pool row** (crustaceans *and* algae) — **17,081 usable pairs**
     on the full pool, giving `a = −0.650, b = 0.982`. (The first revision said "fish → crustacean,
     16,998 pairs", which was the crustacean-only count and did not match the code; corrected per
     delta review W-D-4, and `SurrogateFeatures` uses the identical fit.) **Pre-registered
     small-sample fallback:** below `ICERegressionBaseline.MIN_PAIRS = 30` usable pairs the
     regression is not fitted and `a = 0, b = 1` is kept, i.e. the ICE prediction degenerates to the
     surrogate value itself. Measured per fold on the frozen pool: Actinopterygii **0** usable pairs (the fold keeps only 6 fish
     rows, since Chondrichthyes/Cephalaspidomorphi and class-NA fish survive the hold-out), then
     701, 1,263 and 4,688 pairs for the Malacostraca, Branchiopoda and Maxillopoda folds — so the
     identity fallback fires in exactly one fold, and it is declared rather than silent.
     Relatedly, if a fitting frame contains **no** surrogate-group row at all, the surrogate is
     recorded **unavailable** (`groups_present_ = False`) and every `surr_fish_*` value falls back
     to the frame mean: it is never silently re-fitted on non-surrogate rows, which would both
     change the estimator and make the ICE regression circular.
     The published Web-ICE species-pair models are not distributable here, which is stated as a
     deviation: this is the method, re-fitted, not the published coefficients.
   * **Applicability, measured before any fit and pre-registered**, under the single rule of
     §1.4.1 (any pool record, by CAS or `parent_key`): **94.6 % of insect (405 chemicals), 93.0 %
     of mollusc (297) and 87.6 % of amphibian (342)** primary rows. **Rule for rows without a surrogate (frozen):** they receive the pool
     global mean, the fallback is counted (`coverage_`), and **every surrogate comparison is
     reported twice** — on the full primary endpoint, and on the surrogate-available subset, which
     is how ICE results are normally read. A candidate that beats a surrogate baseline only on the
     non-applicable rows has not beaten it.
5. `readacross` — distance-gated k-NN read-across. **Pre-registered behaviour on this track
   (C-M4U-4):** no test row can match a species or taxon group, so every row falls to the
   `"global"` neighbour pool and, when the similarity gate bites, to the global mean; on this track
   it is therefore a plain chemical-similarity kNN over fish/crustacean rows and is described as
   such. The `widening_` counters (species / group / global / fallback_mean) are reported with the
   results.
6. `global_mean`, `species_mean`, `species_duration_mean` — fitted on the pool. For an unseen class
   `species_mean` and `species_duration_mean` **necessarily equal** `global_mean`; this is reported
   as the structural reason, and **no claim may rest on beating them**.
7. **Channel-matched comparators — mandatory (§1.4.3).** `rf_chan`, `xgboost_chan`,
   `rf_multitaxa_chan`, `xgboost_multitaxa_chan`, `rf_published_chan`, `xgboost_published_chan`
   and `gpboost_published_chan`: the same learners with the same Channel A + Channel B features a
   candidate may use. These are the comparators that make the channels a fair test rather than a
   gift to the candidate, and a claim must beat them.
8. `best_baseline` — resolved per block to the lowest-RMSE comparator of 1-7 after the scores are
   seen. It is conservative for the candidate but its paired CI is **not nominal**, so it is
   labelled descriptive (W-M4U-6), and the pre-registered claim rule requires the surrogate and
   GPBoost comparisons in their own right, not only `best_baseline`.

### 5.2 Candidates

* `chemprop` — first candidate (it was the best model on the ADORE-unseen rows of the closed
  track). **Pre-registered as single-task / taxon-agnostic (C-M4U-4):** `ChempropTox._task_index`
  maps an unknown `tax_group` to output 0, i.e. a multitask configuration would silently predict
  every insect, mollusc and amphibian row with its **fish** head. `multitask=True` is therefore
  **inadmissible** on this track, enforced by `unseen_taxa.assert_taxon_agnostic` and covered by a
  test.
* `lightgbm`, `catboost`, `chemeleon_lightgbm` — the tree/CheMeleon families carried over.
* `gpboost` (re-tuned, via `SeededGPBoost`) — a candidate *variant* of the published comparator,
  admissible only under the same determinism check.
* Ensembles (`ensemble_mean`, `ensemble_stack_nnls`) over whichever candidates beat the strongest
  comparator on validation, with every component's own score reported beside them.

Comparator or candidate is decided by an explicit `role` field in the frozen block, never by
pattern-matching a name. **Every** model — comparator, candidate and baseline — is fitted through
`TaxonBlindBuilder` (§1.1), and `assert_no_trait_columns` is called for each.

## 6. Metrics, statistics and decision rules

* Primary metric per block: **micro-RMSE** on its primary (in-domain) endpoint, in log10(mol/L).
  Reported beside it: MAE, R², bias, macro RMSE by chemical and by species, % within 1 log unit,
  the block's replicate floor and MDE, and breakdowns by `chem_novel`, `parent_novel`,
  `structure_class`, duration, taxonomic order and (molluscs) class.
* Uncertainty: cluster bootstrap, 2,000 resamples, percentile intervals, **with two co-primary
  clusterings** — by **chemical** and by **`reference_number`** (§2) — plus a
  **parent-structure-clustered** sensitivity CI for every primary comparison.
* Comparisons: paired bootstrap on identical rows against every comparator, two-sided.
* Multiplicity: Holm across the candidates **within a block**, for four references
  (`published_reference`, `gpboost_published`, `best_baseline` and the best **channel-matched**
  comparator) and for the surrogate family
  (`fish_surrogate_*`, `ice_regression`). **Each block is its own family and every claim names its
  block.** A claim presented as holding "across unseen taxa" additionally requires Holm across the
  block × reference pairs. (This is the W-M4P-3 lesson of the closed track, written in beforehand.)
* Seeds 0, 1, 2 for every stochastic model; a claim requires the gain in **all** seeds (§15.1 iii),
  per block. A model whose seeds disagree is reported as unstable, not as a winner. For GPBoost the
  determinism check of §5.1(3) must also pass.
* **A "SOTA / first-in-class" claim on a block requires all of:**
  (a) the paired CI against `published_reference` excludes 0 after Holm;
  (b) the same against **`gpboost_published`** — the published cross-species method — or, if it
      failed to fit, the downgrade rule of §5.1(3) applies and no SOTA claim is available;
  (c) the same against the **best surrogate/ICE baseline**, on the **surrogate-available subset**
      as well as on the full endpoint;
  (d) the same against `best_baseline`;
  (d2) the same against the **channel-matched comparators** of §1.4.3 — a candidate that wins only
      because it alone received a cross-taxon channel does not qualify;
  (e) `all_seeds_better`;
  (f) the point estimate keeps its sign on the block's `parent_novel` stratum, on the "no shared
      reference" stratum, and under the reference-clustered CI;
  (g) the difference exceeds the block's **MDE** (§2) — the precision of a *paired difference*.
  **The replicate floor is NOT a difference threshold** and is not used as one: it bounds the
  *absolute* RMSE any model can attain, so requiring a candidate to beat a published comparator by
  more than 0.649 log units on insects would have made a claim unreachable for a reason unrelated to
  evidence (delta review W-D-7, over-correcting W-M4U-2). Each block's absolute RMSE **is** reported
  **relative to its replicate floor**, descriptively, as the honest statement of how much of the
  remaining error is irreducible.
  A **minimal practically-relevant difference** may be added as a further gate only if it is
  justified **independently of the observed data** (e.g. from regulatory assessment-factor
  practice) and written into this plan before scoring; none is declared now.
  Any claim failing only (f) or (g) is reported as **qualified**, with the failing condition named.
* "Competitive" if CIs overlap; "negative" if worse. The number of configurations tried and the
  selection effects are stated. **Honest negative results are the expected outcome** (§9) and are
  reported as results.
* The closed prospective and ADORE results are reported unchanged beside this track; no numbers are
  pooled across tracks or across blocks, and no result here is presented as a prospective number.
* "First-in-class" wording additionally requires the dated novelty search
  `docs/novelty_checks/chemtox_unseen_taxa.md` to have been **performed and verified**, not merely
  filed (it currently marks Web-ICE metadata as *to verify*).

## 7. One-time scoring guard (C-M4U-6)

Prose is not a guard, so the mechanism the repository already uses for test looks
(`sentinel.models.downstream.testlook`) is mirrored in
**`sentinel.models.chemtox.unseen_taxa_guard`**:

| Step | Function | Effect |
|---|---|---|
| Pins | `make_pins()` → `docs/preregistration/chemtox_unseen_taxa_pins.json` | sha256 of both parquet files, this plan, `unseen_taxa.py` and the guard itself |
| Verify | `verify_freeze()` | refuses if any pinned file, **or this plan**, has changed since the freeze |
| Claim | `authorise(block, code_sha=…)` | refuses if the registry already holds a row for that block; writes `AUTHORISED.json` with `O_CREAT\|O_EXCL`, mode 0444 |
| Read | **`load_test_block(block)`** | the **only** way stage-2 code may read test rows; requires a live claim of the current freeze |
| Predict | `save_predictions(block, pred)` | write-once (a second call raises, so a re-fit is impossible), before any registry write |
| Close | `mark_scored(block, summary)` | writes `SCORED.json` and seals the whole block directory 0444 with `SHA256SUMS` |

`registry_rows(block)` counts existing rows under experiment `chemtox_unseen_taxa_test` whose
split string contains `/<block>/`, so a second look is refused by the registry as well as by the
marker; the split naming is therefore part of the protocol (`unseen_taxa/<block>/test`).
`verify_freeze` also checks the pinned sha256 of `unseen_taxa.py` and `unseen_taxa_guard.py`
(delta review W-D-1), so the code defining the features and the folds cannot change after the
freeze undetected. `mark_scored` does not itself re-check authorisation; the scoring script's sha
is pinned in stage 2 instead. The frozen parquet files are themselves **mode 0444 with a `SHA256SUMS`
sidecar**, and `tests/test_chemtox_unseen_taxa.py` **pins all five sha256 values and fails (never
skips) if the data root exists but a hash differs** — a silent rebuild cannot leave the suite green.

## 8. Sequence

1. **Done:** protocol (this file), frozen blocks + sha256 + seal, power/MDE check, the new modules
   (`unseen_taxa.py`, `unseen_taxa_guard.py`), the novelty check, the tests, and a registry
   `prereg_snapshot` row.
2. **Next: an independent delta review** of this revision (the six criticals and the warnings),
   before any fit.
3. Then selection on §4's four validation folds only, one change at a time, tuning parity.
4. Then stage 2 (frozen configurations, feature list, determinism-check results, hashes) appended
   here, a dry-run pre-flight, and a **single** scored pass per block after the lead authorises it
   explicitly, through §7's guard.
5. Then a post-results review before any number leaves the repository.

Shared-host rules apply throughout: GPU work only through `sentinel.core.gpu` under the GPU 0/1
cap, ≤ 2 h per GPU job, other workstreams' test-critical jobs first, CPU ≤ 6 threads.

## 9. Design position after the §1.4 amendment

The first revision recorded that no model had a cross-taxon channel while the surrogate/ICE
comparators did, so the expected outcome was a quantified negative. **§1.4 changes that
deliberately and symmetrically:** candidates may now use chemical-matched surrogate information and
taxonomy above the held-out class, and so may every mandatory comparator, including the published
ADORE trees and the published GPBoost cross-species method.

What this does and does not change:

* **It makes the track winnable legitimately.** A candidate can now carry taxon-relevant signal to
  an unseen class, through the same inputs a regulator would use.
* **It raises the bar at the same time.** The surrogate/ICE baselines were already strong because
  they had the only cross-taxon channel; now the channel-matched trees and GPBoost have it too, and
  §6(c), (d) and (d2) require a candidate to beat all of them. A SOTA label is therefore *harder*
  to earn than in the first revision, not easier — it now means "better than the published methods
  given the same information", which is the only version of the claim worth making.
* **The honest expectations, recorded in advance.** The surrogate features should dominate
  structure-only features wherever a surrogate exists (82-91 % of rows), so the interesting
  measurements are (i) how much a learner adds *on top of* ICE, which the residual variants isolate
  directly, and (ii) what happens on the surrogate-unavailable rows, where chemistry is all there
  is. Channel B should help amphibians most (phylum + subphylum shared with fish), insects next
  (phylum shared with crustaceans) and molluscs least (clade only, §1.4.2) — if the observed
  ordering differs, that is a finding about the channel, not a licence to re-interpret the
  endpoint.
* **What is still out of scope**: species embeddings from taxonomy + genetic sequence (Yang et al.
  2026), which remain deferred to their own pre-registration, and anything at class rank or below,
  which §1.4.2's guard forbids.

A negative or merely "competitive" result remains a perfectly good outcome of this track and will
be reported as one.

---

## Amendment A3 (2026-11-09, dated; before any affected fit) — comparator definitions

Reviewed in `REVIEW_REPORT_PRE_TRAINING.md`, "M4 unseen-taxa — amendment A3 review (2026-11-09)":
A3.1 approve with changes, A3.2 approve with changes for `gpboost*` and **reject as drafted** for
`readacross`. All five blockers (C-A3-1…5) and text items T1–T5 are closed below. **No rule, fold,
feature or datum changes**; no test block has been read. Entries already fitted under
`prereg_snapshot_v6` (`xgboost`, `rf`) keep their rows and configurations unchanged and need no
re-run — A3 touches neither.

### A3.0 The v6 grouping caveat, moved from a registry note into this protocol

Recorded with `prereg_snapshot_v6` and now binding protocol text (it changes what a mandatory
comparator *is*, so it belongs here and not only in a registry `notes` field):

> The trait block is dropped on this track, so the pool has no `tax_gs`, and the only grouping
> variable Channel B admits is **`rank_phylum` (3 levels)**. The published method groups
> species/genus/group, so **`gpboost_published` here is an ADAPTED estimator** and must be
> labelled as such in every comparison.

**Why `rank_phylum` is admissible as a grouping key although the same information was rejected as a
one-hot feature** (delta review D-2): GPBoost maps a level unseen in training to a sentinel and
predicts the random effect's grand mean (`gpboost_model.UNSEEN`, `predict_var=False`), so `Mollusca`
on the mollusc block contributes a **shrinkage fallback**, not a seed-dependent weight on an
always-zero input. A one-hot column that is constant-0 in training and non-zero only on test rows
has no such fallback — that asymmetry is the whole argument.

Levels on the frozen pool: Chordata 42,608 / Arthropoda 17,805 / Other 162 (algae).

**Protocol invariant (T4).** No adapter on this track may create `tax_gs`, `tax_genus`,
`tax_family` or `tax_species`. The column's *absence* is what makes a below-class channel
impossible: `chemprop`'s `ConditionEncoder` keys a **species embedding** on `tax_gs`,
`readacross` matches species on it, and `gpboost_model.GROUP_COLUMNS_DEFAULT` groups on it.
Enforced by `assert_no_forbidden_columns` and
`tests/test_chemtox_unseen_select.py::test_adapter_never_creates_below_class_columns`.

**`tax_group` (T2).** `tax_group` is **not** above class rank: for insects it is one-to-one with
`Insecta` and for amphibians with `Amphibia` — the same proxy structure D-2 rejected for
`rank_subphylum=Hexapoda`. It is admissible only because (i) it is a `TRAIT_PREFIXES` member, so
`TaxonBlindBuilder` deletes it from every feature matrix; (ii) it is never a GPBoost grouping
column; and (iii) its test-block values match no training level. Its only use is read-across's
widening level (§5.1.5), which this track pins off (A3.2b).

**GPBoost adaptation is symmetric (T5).** The `gpboost` **candidate** is adapted in exactly the
same way as `gpboost_published` / `gpboost_published_chan` — all three group on `rank_phylum` only.
The candidate is therefore **not advantaged** by the adaptation, and the model card carries the
ADAPTED label for all three.

### A3.1 The four `*multitaxa*` entries are **aliases**

Measured on the frozen pool before deciding: the plain `rf` / `xgboost` entries already train on
**all pool classes**, not fish-only.

| Fold (held-out class) | Train rows | Distinct classes | Taxon groups |
|---|---|---|---|
| Actinopterygii | 676 | 10 | crustaceans 653 / algae 17 / fish 6 |
| Malacostraca | 8,378 | 10 | fish 7,200 / crustaceans 1,145 / algae 33 |
| Branchiopoda | 11,523 | 10 | fish 10,042 / crustaceans 1,455 / algae 26 |
| Maxillopoda | 24,505 | 11 | fish 19,156 / crustaceans 5,299 / algae 50 |

`validation_splits` removes only the held-out class and its chemicals; nothing restricts training to
fish. On the closed prospective track `multitaxa` meant *extra ADORE rows from the other
challenges*; those rows are already in this frozen pool, so **"multitaxa" has no distinct meaning
here**. The alternative reading is moreover **forbidden** by this plan: §5.1 requires every
mandatory comparator to be "trained on exactly the same rows as every candidate", and §3 pins the
pool by sha256, so a distinct multitaxa training set would break both.

**Resolution.** `rf_multitaxa` → alias of `rf`; `xgboost_multitaxa` → alias of `xgboost`;
`rf_multitaxa_chan` → alias of `rf_chan`; `xgboost_multitaxa_chan` → alias of `xgboost_chan`.
Identical rows, identical configurations, identical numbers; **never fitted separately and never
evaluated as independent comparators**. Every comparator table prints them as `= rf` / `= xgboost` /
`= rf_chan` / `= xgboost_chan` (T3), so the comparator-count change is auditable from the output.

**Claim condition (d2) is unchanged (T1).** It remains *the candidate against each channel-matched
comparator* (`rf_chan`, `xgboost_chan`, `rf_published_chan`, `xgboost_published_chan`,
`gpboost_published_chan`). The `_chan` vs non-`_chan` difference is **not** the claim condition; it
is reported beside it as the measured value of the channels.

### A3.2 Grouping and the two estimators that need it

**(a) GPBoost.** `gpboost` and `gpboost_published_chan` are fitted with
`group_columns=("rank_phylum",)`. `group_columns` is **removed from the sampled space** on this
track (C-A3-3), so the recorded configuration cannot misreport the grouping — `cycle.GPBOOST_SPACE`
samples it over `tax_gs` / `tax_genus` / `tax_family`, and recording a sampled value for a fit that
used `rank_phylum` would corrupt the audit trail in the worst direction. After every GPBoost fit the
runner asserts `tuple(model.inner_.group_columns_) == ("rank_phylum",)`.

**(b) Read-across — the draft was wrong (C-A3-1, C-A3-2).** A3 first claimed read-across
"degenerates" to the global kNN. It did not: `ReadAcrossKNN.fit` reads `rows["tax_gs"]`
**unconditionally** (`readacross.py:105`), before and regardless of `match_species`, so the entry
raised `KeyError('tax_gs')` and had never executed. `readacross.py` is shared with the closed
prospective track and is **not edited**. Instead a track-local adapter (`TrackReadAcross`):

* supplies a **constant** placeholder species key, so the column exists and carries nothing; and
* pins **`match_species=False`**, because every *validation* fold's held-out class shares its taxon
  group with training rows (fish for Actinopterygii; crustaceans, 1,145–5,299 rows, for the other
  three), whereas a *test* block's `taxon_group` matches no pool row at all. Leaving the group level
  live would make selection easier than test — inverting §1.4.4's own honest-analogue principle and
  contradicting §5.1.5.

Measured on the Actinopterygii fold: `widening_ = {species: 0, group: 0, global: N,
fallback_mean: k}`, asserted per fold at run time and in
`tests/test_chemtox_unseen_select.py::test_track_readacross_degenerates_to_global_knn`. On this
track read-across is therefore the plain chemical-similarity kNN §5.1.5 describes, and is reported
as such with its `widening_` counters.

### A3.3 Published comparators use the published configuration (C-A3-4)

§1.4.3 describes `rf_published_chan` / `xgboost_published_chan` / `gpboost_published_chan` as "the
published configurations likewise" — a **fixed** configuration — while the §1.4.4 inventory counted
them as 32-configuration searches. A3 settles it in favour of §1.4.3 and §5.1.1:

* the `*_published_chan` entries take the **published fixed** hyper-parameters (the same ones the
  closed track froze for `rf_published` / `xgboost_published`), with the **channel setting as the
  only varied dimension** → **3 configurations each**;
* the re-tuned `rf_chan` / `xgboost_chan` keep 32 configurations each, drawn with a **per-entry
  seed** so they cannot coincide with the published pair.

This closes the defect the review measured: one fixed search seed had made `rf_chan` ≡
`rf_published_chan`, `xgboost_chan` ≡ `xgboost_published_chan` and — worst — the **`gpboost`
candidate ≡ the `gpboost_published_chan` comparator** (32/32 identical), so claim conditions (b)
and (d2) would have compared the candidate with itself. No comparator now shares a draw with any
candidate or with another comparator (tested). `lightgbm`, `lightgbm_resid_ice` and
`lightgbm_resid_surrogate` deliberately do share one draw: all three are *candidates* with
different estimators, so holding hyper-parameters fixed makes the residual base a controlled
ablation.

### A3.4 Channel-matched comparators never select "no channel" (C-A3-5)

§1.4.4's `channels ∈ {none, surrogate, higher_rank, both}` is written for the **candidates**. For a
`_chan` comparator the empty set is **inadmissible**: 11 of its 32 draws were channel-free, and a
selected channel-free "channel-matched comparator" *is* the plain comparator, making (d2) vacuous —
the C-M4P-1 failure mode §1.4.3 exists to prevent. The `_chan` entries therefore sample over the
three non-empty settings only. Candidates keep all four.

### A3.5 The published GPBoost configuration (lead ruling 2026-11-09; **branch (1) applies**)

§5.1.3 names `gpboost_published` as a fixed published configuration but states none, so the lead
declared a rule, before any fit or result on this entry: (1) use the published source's values
verbatim if it states them; (2) otherwise the installed library's defaults, labelled as a weaker
stand-in. **Evidence gathered before any validation result on this entry existed, and recorded
here:**

| Source checked | Outcome |
|---|---|
| On-disk repository | Only the citation (`gpboost_model.py:11-13`, novelty check). **No hyper-parameters.** |
| Paper full text — Zubrod, Galic, Vaugeois & Dreier, *Environ. Int.* **186**:108607 (2024), DOI 10.1016/j.envint.2024.108607, PMID 38593686 | **Not accessible.** Publisher returns HTTP 403; Europe PMC records it as **not open access** ("subscription required"), no PMCID. Methods text unread. |
| Released repository README (`github.com/syngenta/bio-qsar`) | States package versions only (R 4.1.3, gpboost **1.2.1**) and defers to the publication. **No hyper-parameters.** |
| **Released fitted models** in the authors' own repository | **State the complete set verbatim** in their LightGBM/GPBoost `parameters:` echo block. **Branch (1) applies.** |

**Values, verbatim from the authors' released trait-free models** — `_without_DEB` is the matching
form for this track, which drops the trait block (§1.1):

| Parameter | `Fish_without_DEB.json` | `Invertebrate_without_DEB.json` |
|---|---|---|
| `objective` | regression | regression |
| `num_iterations` | 1000 (highest `Tree=999`) | 1000 (highest `Tree=999`) |
| `learning_rate` | 0.01 | 0.01 |
| `num_leaves` | **64** | **32** |
| `max_depth` | −1 | −1 |
| `min_data_in_leaf` | 10 | 10 |
| `lambda_l2` | **1** | **10** |
| `feature_fraction` / `bagging_fraction` | 1 / 1 | 1 / 1 |

Provenance: `Fish_without_DEB.json` sha256 `1b17db4c5f18abbd…` (7,171,431 bytes),
`Invertebrate_without_DEB.json` sha256 `770ecd0d1ed31360…` (3,369,787 bytes), retrieved
2026-11-09 from `raw.githubusercontent.com/syngenta/bio-qsar/main/`.

**Both sets are kept**, because the authors released taxon-specific models differing in exactly two
values and this track predicts both invertebrate (insects, molluscs) and vertebrate (amphibians)
blocks. Per §5.1.1's existing rule the reference is the variant with the best validation score,
fixed in the hashed block before the pass. So `gpboost_published_chan` is **2 published forms × 3
channel settings = 6 fixed configurations, no search**, and no hyper-parameter is chosen by us.

**Claim condition (b) is evaluable again**, against an estimator described exactly as:
*"published configuration, verbatim from the authors' released trait-free models; ADAPTED grouping
(`rank_phylum` only, A3.0); gpboost 1.8.0 against the authors' 1.2.1."* Two adaptations are
disclosed with every (b) comparison — the grouping (the authors grouped species/genus/group, which
this track forbids) and the library version. The `gpboost` **candidate** carries the identical
grouping adaptation (T5), so it gains no advantage.

**A note recorded because it would have biased in our favour.** Had branch (2) applied, the
library defaults of gpboost 1.8.0 (read from a fitted booster's own parameter echo:
`num_iterations` 100, `learning_rate` 0.1, `num_leaves` 31, `min_data_in_leaf` 20, `lambda_l2` 0,
`feature_fraction`/`bagging_fraction` 1) combined with the authors' `learning_rate` 0.01 would have
produced a **severely undertrained** published comparator (0.01 × 100 rounds against the authors'
0.01 × 1,000). A weak published comparator inflates our own candidate, which is the worst direction
for a SOTA claim. Branch (1) removes the issue; the figure is recorded so the hazard is on record.

### A3.6 Configuration inventory — one table, replacing §1.4.4's

| Group | Entries | Searched | Fixed |
|---|---|---|---|
| Candidates: `chemprop`, `lightgbm`, `catboost`, `chemeleon_lightgbm`, `gpboost`, `lightgbm_resid_ice`, `lightgbm_resid_surrogate` | 7 | 224 | — |
| ADORE comparators re-tuned: `rf`, `xgboost` | 2 | 64 | — |
| Channel-matched, re-tuned: `rf_chan`, `xgboost_chan` | 2 | 64 | — |
| Channel-matched, published fixed config × 3 channel settings: `rf_published_chan`, `xgboost_published_chan` | 2 | 6 | — |
| Channel-matched, published GPBoost: 2 verbatim published forms × 3 channel settings (A3.5) | 1 | 6 | — |
| `readacross` | 1 | 32 | — |
| Published, fixed configurations: `rf_published`, `xgboost_published`, `rf_published_mwlogp`, `xgboost_published_mwlogp`, `gpboost_published` | 5 | — | 5 |
| Surrogate/ICE baselines: `fish_surrogate_geomean`, `fish_surrogate_most_sensitive`, `fish_surrogate_geomean_both`, `ice_regression` | 4 | — | 4 |
| Trivial: `global_mean`, `species_mean`, `species_duration_mean` | 3 | — | 3 |
| **Total** | **27** | **396** | **12** |

Replaces "608 searched + 12 fixed across 31 model entries". The four aliases are counted once, in
their plain entries. **Mandatory comparators: 17** = 2 re-tuned ADORE trees + 5 channel-matched
(all runnable) + 5 published fixed + `readacross` + 4 surrogate/ICE. Of the 27 entries the
remaining 10 are 7 candidates + 3 trivial baselines. The numbers reported with any claim are the
**distinct** configuration counts above, and `prereg_snapshot_v7` carries them so they cannot drift.

---

## Amendment A4 (2026-11-09, dated) — documentation only, after the A3 delta check

**No rule, fold, feature, configuration or datum changed.** This closes the two conditions and the
documentation items of `REVIEW_REPORT_PRE_TRAINING.md`, "M4 unseen-taxa — A3 delta check
(2026-11-09)". Appended rather than edited in place, so every earlier hash stays reproducible.
At the time of writing, `gpboost_published_chan`, `gpboost` and `readacross` had **not** been
fitted for selection.

### A4.1 §1.4.4's inventory table is SUPERSEDED (delta review D1)

**§1.4.4's table and the paragraph after it (around lines 250–256) are superseded by A3.6 and must
not be read as current.** Specifically superseded:

| §1.4.4 says (stale) | Current, per A3.6 |
|---|---|
| "**Total** 31 model entries / 608 searched / 12 fixed" | **27 entries / 396 searched / 12 fixed** |
| "**21 are mandatory comparators**" | **17 mandatory comparators** (2 + 5 + 5 + 1 + 4) |
| "the registry `prereg_snapshot_v4`/`v5` carries the same three figures **so the count cannot drift**" | **False as written.** The authoritative counts are in **`prereg_snapshot_v7`** (396 / 12 / 27 / 17 / 5), and A3.6 is the governing table |

A reader meets §1.4.4 some 550 lines before A3.6, so this notice exists to stop the stale table
being taken as current. The four `*multitaxa*` entries counted there are **aliases** (A3.1), never
fitted and never independent comparators.

### A4.2 The published GPBoost comparator loses **three** things, not two (delta review A3.5-1)

A3.5 called the `_without_DEB` models "trait-free". That is correct as the *choice between the two
released forms* — neither file's feature list contains a DEB / Add-my-Pet parameter, so it is the
form that matches §1.1's trait drop — but it **overstates the match**, and the correct wording is
**"DEB-trait-free (the authors' species indicators are also absent here)"**, never "trait-free".

Both released files still carry **species one-hot fixed effects** — **10** in `Fish_without_DEB`
(e.g. `Oncorhynchus_mykiss`, `Pimephales_promelas`, `Daphnia_magna`) and **13** in
`Invertebrate_without_DEB` (e.g. `Gammarus_pulex`, `Chironomus_riparius`) — on top of the
species/genus/group random effects. §1.1 forbids any species-rank input on this track, so they are
absent from our fit. The published comparator therefore differs from the authors' own model in
**three** ways, **each of which makes it weaker here**, i.e. **each biases in our candidate's
favour**:

1. **grouping** reduced to `rank_phylum` (3 levels) from species/genus/group (A3.0);
2. **library version** gpboost 1.8.0 against the authors' 1.2.1;
3. **the authors' species one-hot fixed effects are absent** (10 / 13).

**All three must be stated with every claim-condition-(b) comparison** and in the model card.
Enforced in code as `PUBLISHED_GPBOOST_ADAPTATIONS`, written into `summary_index.json`.

**Selection breadth, reported beside (b).** The published GPBoost comparator is selected as the best
of **6** configurations (2 released forms × 3 channel settings) while `rf_published_chan` and
`xgboost_published_chan` are each the best of **3**. The direction is conservative — a wider choice
makes the published bar *harder* to beat — but the asymmetry is reported, not left implicit
(`PUBLISHED_SELECTION_BREADTH`).

Taken together with A3.5's closing note, the honest summary is: **our published GPBoost comparator
is a weakened form of the authors' model in three respects, and no (b) result may be presented
without them.**

### A4.3 The alias rows are emitted (delta review D3)

A3.1 promised that "every comparator table prints them as `= rf` / `= xgboost`" while no code wrote
them. `alias_rows()` now emits one row per alias (`rf_multitaxa = rf`, `xgboost_multitaxa =
xgboost`, `rf_multitaxa_chan = rf_chan`, `xgboost_multitaxa_chan = xgboost_chan`), each marked
`fitted: false`, into `summary_index.json`, and the stage-2 scorer prints them in the comparator
table. The alias map is also carried in `prereg_snapshot_v7`/`v8`.

### A4.4 Provenance durability and a citation correction

* **Durable provenance (A3.5-3).** The two released model files were retrieved into a session
  scratchpad that will not survive. The URL + sha256 + byte counts in A3.5 make retrieval
  reproducible (the reviewer reproduced both hashes independently), and a manifest entry
  `chemtox_published_gpboost` records source URLs, sha256, byte counts and retrieval date. **Only
  the nine hyper-parameter values are used**; neither file's weights, features or predictions enter
  any SENTINEL model.
* **Citation correction (D4).** The correct article title is **"Bio-QSARs 2.0: Unlocking a new level
  of predictive power for machine learning-based ecotoxicity predictions by exploiting chemical and
  biological information"** (Crossref, Europe PMC and the repository README agree). A3.5's citation
  — journal, volume, article number, DOI, PMID, authors — was already correct. **Known defect, not
  fixed here:** `sentinel/models/chemtox/gpboost_model.py:11` carries an incorrect short title
  ("…Unlocking a generic modeling approach for species sensitivity predictions"). That file is left
  untouched because it is covered by an existing freeze; the error is recorded here so the wrong
  title cannot reach the paper or the model card. Any later edit belongs to whoever owns that
  freeze.

### A4.5 The unlogged smoke fits (A3.5-2)

Two statements of execution in the handoff had no artefact behind them. They were **single unlogged
smoke fits on one validation fold**, used only to prove the code path runs, never for selection, and
they wrote **no registry row** (`--log-registry` now refuses any run that is not the full budget over
all four folds). Their logs are kept under
`$SENTINEL_RUNS/chemtox/unseen_taxa/stage2_select/smoke/`. The protocol's "nothing fitted for
selection" statements are exact: the registry holds no `gpboost*` or `readacross` selection row.

---

## Amendment A5 (2026-11-09, dated) — validation folds must mirror how the test block is built

**A correctness fix, not a tuning change.** Found by the gate-condition-2 smoke of
`gpboost_published_chan` plus an unlogged `rf` diagnostic, **before any logged channel result
existed**: at the time of discovery the only logged selection rows were `rf` and `xgboost`, both
with `channels` pinned off, and the registry held **0** rows for every channel-using entry. **No
test block was read at any point; the three blocks remain unspent.**

### A5.0 The defect

Channel A (the surrogate block) was **0.0 % available on every validation fold**, against
**94.6 % / 93.1 % / 87.6 %** on the test blocks (§2, D-3):

| Validation fold | Surrogate available on validation rows |
|---|---|
| Actinopterygii | **0 / 42,533 (0.0 %)** |
| Malacostraca | **0 / 9,386 (0.0 %)** |
| Branchiopoda | **0 / 7,085 (0.0 %)** |
| Maxillopoda | **0 / 1,191 (0.0 %)** |

**Mechanism — two pre-registered rules interacting, not a coding typo.** §4/W-M4U-3 removes the
held-out class's chemicals from the fold's training side by CAS, InChIKey **and** `parent_key`,
while §1.4.4 re-fits the surrogate block on that fold's training rows only. Together, **no
validation chemical can have any pool record**, so the surrogate is never available. §1.4.4's own
rationale — *"a fold's surrogate coverage is what test-time coverage will be"* — was therefore
**false as implemented** (0 % against ~90 %).

**Consequences, measured.** On validation rows all 7 surrogate features were **constant** (pool
mean −4.452, ICE of the pool mean −4.632, flag 0, counts 0): correlation with the target was exactly
**0.000** on validation against **0.92–0.96** on training. Compounding it, **self-inclusion target
leakage** on the training side: for all **736/736** training rows whose chemical had exactly one
pool record, `surr_pool_geomean` equalled that row's **own target** to machine precision.

**Not model-specific — it hit candidates identically.** Unlogged `rf` diagnostic, Malacostraca fold,
identical parameters, only the channel varying (`stage2_select/smoke/surrogate_channel_diagnostic.log`):

| channels | TRAIN RMSE | VAL RMSE |
|---|---|---|
| `()` | 0.3362 | 1.6433 |
| `('surrogate',)` | 0.3023 | **2.3210** |
| `('higher_rank',)` | 0.3192 | 1.5902 |
| `('surrogate','higher_rank')` | 0.2805 | 2.2827 |

Training error *improved* with the surrogate while validation degraded by **+0.68** — the signature
of a feature that exists only in training. Selection would therefore have rejected every Channel A
configuration for an artefact, and §9's recorded expectation that surrogate features should dominate
could never have been tested. Worst affected: `lightgbm_resid_ice` and `lightgbm_resid_surrogate`,
whose base was constant on validation, so in CV they reduced to plain `lightgbm` on a shifted target
while behaving completely differently at test time.

**Ruled out:** NaN handling (**zero** NaN in all 7 columns, train and validation, so GPBoost-vs-tree
NaN semantics cannot be the cause); unit or scale mismatch (the four value features are log10(mol/L)
on the target's scale — training mean −4.452 exactly equals the target mean −4.452; the two count
features are counts); held-out-class leakage (held-out-class rows in training = **0** for all four
folds, `assert_no_test_classes` passes in `fit`, and the zero availability itself proves no held-out
chemical contributed). **This was a selection-validity defect, never a test leak.**

### A5.1 Validation folds hold out the **class only**; W-M4U-3's chemical blocking is REMOVED

W-M4U-3's stated rationale, quoted from §4: *"The held-out class's chemicals are removed from that
fold's training rows by CAS, InChIKey **and** `parent_key`, so a salt, counter-ion or stereoisomer
cannot make a selection fold **easier than the test blocks it mirrors**."*

Measured against how the **test** block is actually built, §1.2 says the opposite: *"A test chemical
**may legitimately appear in the training pool**, measured on fish. That is realistic and **is not
leakage** of the test rows. So chemical novelty is **reported** rather than enforced."* The
`chem_novel` / `parent_novel` strata exist precisely because most test chemicals are **not** novel
(e.g. insects: 477 of 7,165 primary rows are `chem_novel`, i.e. **93 %** are not).

So W-M4U-3 forbade in selection exactly what test construction allows. It is **inconsistent with
the test and is removed** from the validation rotation. Chemical novelty remains a **reported
stratum** on both sides. The held-out class is still removed in full, and `assert_no_test_classes`
still guarantees no insect, mollusc or amphibian row is anywhere in the pool.

**New folds, and the availability they produce** (`validation_splits_classonly`, new code — the
pinned `unseen_taxa.py` is **not** edited):

| Fold (held-out class) | Train rows | Validation rows | Surrogate available | Held-out-class rows in train |
|---|---|---|---|---|
| Actinopterygii | 18,042 | 42,533 | **88.0 %** | 0 |
| Malacostraca | 51,189 | 9,386 | **98.4 %** | 0 |
| Branchiopoda | 53,490 | 7,085 | **95.0 %** | 0 |
| Maxillopoda | 59,384 | 1,191 | **98.5 %** | 0 |

Against the test blocks' 94.6 % / 93.1 % / 87.6 % — now comparable, which is the point. Note the
folds are **not** thereby "easier than they should be": the correct standard is fidelity to test
construction, not maximal difficulty.

### A5.2 Channel A is computed **out-of-class**, removing self-inclusion

A training row of class *C* now receives surrogate values computed from pool records of **other
classes only** — never its own record and never its own class (`OutOfClassSurrogateFeatures`,
`TrackChannelBuilder`, `ResidualOutOfClassModel`; all new code). This mirrors the test block exactly,
where the surrogate can only come from non-test classes. A row whose class is absent from the
fitting frame — every validation and every test row — is served by the full fit, which is likewise
"all classes present in training".

**Verified by perturbation, which is immune to coincidental equality:** shifting **every** training
target by +100 leaves all 7 surrogate features **bit-identical**, so no row's own target can reach
its own features (`test_a5_no_self_inclusion_in_the_surrogate_features`). Self-inclusion on the
Malacostraca fold falls from **825/825** single-record rows to **6/825**, and those six are data
coincidences (an equal value measured in another class), not self-inclusion — which is why the test
asserts the *mechanism* and not an equality count. Correlation with the target becomes **0.677
(train) / 0.776 (validation)**: validation is now *higher* than training, i.e. the over-reliance is
gone and the feature is genuinely informative where it is used.

**Effect on the pathology** (same unlogged `rf` diagnostic, post-fix, new folds;
`surrogate_channel_diagnostic_postA5.log`):

| channels | TRAIN RMSE | VAL RMSE |
|---|---|---|
| `()` | 0.6474 | 1.3412 |
| `('surrogate',)` | 0.5520 | **1.2286** |
| `('higher_rank',)` | 0.5617 | 1.2502 |
| `('surrogate','higher_rank')` | 0.5503 | **1.2183** |

The surrogate channel now **helps** (1.2286 against 1.3412), reversing a +0.68 penalty — which is
§9's recorded expectation, now actually testable.

### A5.3 All entries re-run on the new folds; no number crosses the change

The fold definitions change for **every** entry, so **all** entries re-run, `rf` and `xgboost`
included, and **the bar is recomputed**. Numbers from the old folds are **not** comparable with
numbers from the new folds and are never mixed, pooled or presented together. The two completed
entries are **superseded in the registry with this amendment as the reason**:

* `rf` — old-fold mean RMSE 1.5629 (worst 1.7397);
* `xgboost` — old-fold mean RMSE 1.5918 (worst 1.7318).

Both were obtained under plan `a3451d5e…`, with `channels` pinned off, on folds that applied
W-M4U-3. They are retained as superseded history, not as the bar. Absolute RMSE is expected to
**improve** on the new folds for a reason unrelated to method quality — the training sets are 2–7×
larger once chemical blocking is dropped (e.g. Malacostraca 8,378 → 51,189 rows) — which is another
reason old and new numbers must never be compared.

### A5.4 What did not change

No test block was read; no claim marker, no `SCORED.json`, no FREEZE file; `registry_rows()` is 0
for insects, molluscs and amphibians. The frozen data files are **unchanged**
(`649dc7d1…` / `7e915fbe…`). The pinned `unseen_taxa.py` and `unseen_taxa_guard.py` are
**unchanged** — every fix is in the unpinned runner and its test file. Channel B
(`HigherRankFeatures`) is reused unchanged: it is a pure function of the class name and carries no
target, so self-inclusion cannot arise there. The model set, the configuration inventory
(396 searched + 12 fixed across 27 entries), the claim conditions, the Holm family, the seeds, the
MDE and the replicate floors are all untouched.

---

## Amendment A6 (2026-11-09, dated) — corrections to A5, after the A5 delta check

**Documentation and verification only. No rule, fold, feature, configuration or datum changed**,
and A5's two fixes (A5.1 class-only folds, A5.2 out-of-class Channel A) stand as approved. A6
corrects statements A5 made that were wrong or unsupported. Appended; nothing above is edited, so
A5's original text remains readable beside its correction.

### A6.1 A5.2's verification claim was FALSE — the test was vacuous (review A5.2-a)

A5.2 stated: *"Verified by perturbation… shifting **every** training target by +100 leaves all 7
surrogate features **bit-identical**, so no row's own target can reach its own features."* **That
test proved nothing.** It shifted `result_conc1_mean_mol_log`, but every surrogate aggregate reads
`unseen_taxa.TARGET` = **`log10_conc_mol_L`**. `to_model_schema` keeps both columns, so the test
perturbed a column no surrogate code reads; it **could not fail**, and it would have passed against
the defective pinned code. Reproduced: perturbing only the unread column leaves the **pinned**
`SurrogateFeatures` bit-identical.

**The property A5.2 claims is nevertheless true** — the reviewer verified it independently with a
test that has power, and so does the replacement. The new test perturbs **one** training row's
target in **both** columns (so the column choice cannot hide anything) and checks two sides:

| Implementation | Perturbed row's own features | Another class's row sharing the chemical | Verdict |
|---|---|---|---|
| pinned `SurrogateFeatures` (pre-A5) | **changed** | changed | **FAIL — detects self-inclusion** |
| `OutOfClassSurrogateFeatures` (A5.2) | **unchanged** | **changed** | **PASS** |

The "changed" witness proves the check has power. A second test,
`test_a5_self_inclusion_check_has_power_against_the_pinned_path`, **requires the check to fail on
the pinned implementation**, so if the check ever loses its power the suite fails. A third documents
the vacuity of the old formulation.

### A6.2 The self-equality figures, each with its full frame (review A5.2-b)

A5.0 said 736 and A5.2 said "825/825 → 6/825" for what read as one quantity; both cannot be right.
**"825/825" was never measured** — it was an error of mine, conflating a new-fold denominator with
an old-code numerator. The frame of a figure includes **the definition of "single-record"**, which is
why two careful measurements disagree:

| Frame | Measured by me — single-record = CAS occurs once in the training frame | Measured by the reviewer |
|---|---|---|
| old folds, old (pinned) code | **736 / 736** | **736 / 736** |
| old folds, new (out-of-class) code | 6 / 736 | 15 / 934 |
| new folds, new code | 6 / 825 | 21 / 2,549 |

**736/736 is reproduced by both of us**, and also under a second definition I tried. I **could not
reproduce 15/934 or 21/2,549** under either definition I tested (CAS-once-in-frame;
once-within-own-class), so they are attributed to the reviewer rather than restated as measured
here. Every frame agrees the residual rate is small (≈0.2–0.8 %), and those residual equalities are
coincidences — an equal value measured in **another** class — not self-inclusion. **No claim rests on
these counts.** A5.2's argument rests on the **mechanism**, which A6.1's two-sided test now actually
verifies.

### A6.3 Train/serve availability asymmetry (review A5.2-c) and per-fold reporting (A5.1-a)

Out-of-class computation makes Channel A **less** often available on training rows than on the rows
it serves:

| Fold | Train (out-of-class) | Validation (served) | Test blocks, for reference |
|---|---|---|---|
| Actinopterygii | 88.1 % | 88.0 % | insects 94.6 % |
| Malacostraca | 82.2 % | 98.4 % | molluscs 93.1 % |
| Branchiopoda | 87.2 % | 95.0 % | amphibians 87.6 % |
| Maxillopoda | 90.3 % | 98.5 % | |

(The reviewer reports Branchiopoda train 87.1 %; mine is 87.2 % — agreement to within 0.1 point.)
This is the standard leave-one-group-out encoding trade-off. Its direction is **conservative**: a
feature more often missing in training is under-weighted, which cannot manufacture a win. It is
nonetheless a real train/serve difference and is disclosed here.

Two folds (98.4 %, 98.5 %) sit **above** the whole test range and Actinopterygii (88.0 %) at its
bottom, so selection slightly over-weights the surrogate-rich regime relative to the amphibian
block. **Reporting rule (A5.1-a):** every fold's score is reported beside that fold's validation
availability; the runner records `surrogate_available_val` per fold. No rule changes.

### A6.4 A5.3's provenance sentence was wrong (review A5.3-b)

A5.3 said `rf` and `xgboost` "were obtained under plan `a3451d5e…`" (v8). **Wrong.** Read from the
registry rows themselves: both original fits record **`plan_sha256 = 48553f19…` (v6)**. The
correction rows I wrote for `xgboost` under W-A3-2 record **`40fdb796…` (v7)**, because they were
written after the v7 re-freeze — they re-stamp a v6 fit. (The delta review says the re-stamp carried
the v8 hash; the rows say v7.) The A5 supersede rows I wrote recorded `plan_sha256_at_run =
a3451d5e…`, which is likewise wrong and is corrected here: **both were fitted under v6.**

### A6.5 The discovery record, completed (review A5.3-d)

A5 credited discovery to "the gate-condition-2 smoke plus an unlogged `rf` diagnostic". The record
was incomplete. An **aborted `lightgbm` selection run** also ran on the **old** folds: started
19:21:28 under plan `a3451d5e…`, killed at 19:26:06 when the queue was paused, after three complete
configurations and part of a fourth (train rows 676 / 8,378 / 11,523 / 24,505). Its Malacostraca
RMSEs **2.1975** (cfg00) and **2.2660** (cfg02) are the Channel-A artefact itself, against 1.7448 for
cfg01. It produced **no artefact and no registry row**.

What I can state honestly about causation: the diagnostic was written in response to the lead's
correctness-check request, which followed the smoke. I did not read the `lightgbm` log before writing
the diagnostic, so I do not claim the run triggered it — but it **independently exhibits the same
artefact**, on a different learner, and so corroborates the defect. It confirms rather than
contradicts that nothing about Channel A was selected, reported or logged under the defective folds:
the registry has never held a row for any channel-using entry.

### A6.6 Artefacts archived before the re-run (review A5.3-c)

The retired rows cite `stage2_select/xgboost.json` and `rf.json`, which the A5 re-run writes again.
**Before any re-run**, every old-fold artefact was **moved** — byte-identical, sha256 verified before
and after — into sealed 0444 directories:

* `stage2_select/old_folds_a5/` — `xgboost.json`, `xgboost.log`, `rf.json`, `rf.log`,
  `summary_index.json`, `queue.log`, `phase2_queue.log`, `BAR_DONE.flag`, plus `SHA256SUMS`;
* `stage2_select/smoke/aborted_old_folds/` — the aborted `lightgbm.log` and its `resume_queue.log`,
  plus `SHA256SUMS`.

The output directory held no old-fold artefact when the re-run began. Each retired row has an
appended `artefact_relocated` annotation pointing at the archived copy.

### A6.7 Registry integrity — twelve dangling rows, not two (review A5.3-a)

The review found two rows whose `supersedes` named a non-existent id. A new read-only audit,
`scripts/eval/registry_audit.py`, run over **the whole registry**, found **twelve** — all M4's, none
in any other workstream:

* **2 guessed ids** (`75fcc789…`, `6d85035a…`) from my first A5 supersede attempt;
* **10 version names in an id field** — my own `prereg_snapshot_v7`, `v8` and `v9` rows put
  `"prereg_snapshot_v6"` (etc.) into `supersedes`, which must hold a row **id**. The established
  convention (v1–v6) is that snapshots carry **no** `supersedes` and accumulate as ordered history.

This is the **third** instance of one error class — writing into `supersedes` a value I had assumed
rather than read. Remedies:

* the two guessed-id rows are **tombstoned** (their intended job was already done with verified ids);
* the ten snapshot rows are **not** tombstoned: their content is correct, and retiring them would
  hide v7–v9 while v1–v6 stay visible. Each is instead **acknowledged** by a reasoned row
  (`dangling_supersedes_acknowledged`) that retires nothing;
* the audit fails on any **unacknowledged** dangling row, so it cannot be satisfied silently;
  after the corrections it reports **OK**;
* **write-time refusal:** every M4 registry write carrying `supersedes` now goes through
  `scripts/eval/registry_checked.log_result_checked`, which refuses a value that names no existing
  row. Verified to refuse both a guessed id and a version name, writing nothing. It is scoped to M4
  because `sentinel/core/registry.py` is shared by all seven workstreams; adding the refusal there is
  a cross-workstream decision for the lead;
* `prereg_snapshot_v10` restores the convention: **no `supersedes`**, succession recorded as
  `extra.succeeds`.

---

## Amendment A7 (2026-11-09, dated) — chemprop on this track (lead ruling, option (a))

**Before any chemprop fit for selection.** No rule, fold, feature or datum changes, and **no
sampled configuration changes**. The registry holds no chemprop selection row.

### A7.1 Why chemprop could not run

A one-configuration GPU timing run crashed with `KeyError('tax_gs')`. `ChempropToxModel`'s
`ConditionEncoder.fit` reads `rows["tax_gs"]` **unconditionally** (`chemprop_model.py:71`),
whatever `species_dim` is, and builds one learned embedding per species. This track never creates
`tax_gs` (A3.0 invariant), so chemprop failed loudly — the invariant doing what it was added for:
with real species keys chemprop would learn embeddings for the pool's 1,080 species during
training, a below-class channel §1.1 forbids. Forcing `species_dim = 0` would not have helped,
because `fit` reads the column regardless.

### A7.2 The fix — a constant placeholder, as for read-across (C-A3-1)

`TrackChemprop` (new code; `chemprop_model.py` is shared with the closed track and is **not
edited**) supplies a **constant** placeholder species key on copies of the frames handed to
chemprop — never on the pool, where the A3.0 invariant still holds and is tested. Consequences,
verified structurally and asserted in tests:

* `n_species == 2` (the placeholder plus the out-of-vocabulary slot);
* training, validation and test rows **share one species index**.

**Disclosure.** `species_dim` stays **sampled over [0, 8, 16, 32]**, so the sampled space is
unchanged. On this track every `species_dim > 0` draw is therefore an **information-free learned
bias** — a single learned vector added for every row — **not a species channel**. Those draws are
counted in chemprop's 32 configurations as sampled; no selection advantage can come from them,
since a constant offset is already representable by the network.

chemprop remains **single-task** (C-M4U-4: `multitask = False` on every draw, enforced by
`assert_taxon_agnostic`). It runs on GPU 0/1 only, through `sentinel.core.gpu`, in shards under the
2 h cap.

### A7.3 Execution notes (no rule changes)

* **GPU cap.** `sentinel.core.gpu` falls back to a cap of [0, 1, 2, 3, 4] if
  `.gpu_locks/ALLOWED` is absent, and `--prefer` falls back to the next allowed GPU. The lead has
  made the cap file read-only at `0,1`; every M4 GPU job also sets `SENTINEL_GPUS=0,1`. The core
  `gpu.py` is inside other freezes' code hashes and is not edited.
* **Registry guard.** `registry.py` is likewise hashed by several freezes; M4's write-time guard
  and audit (A6.7) stay track-local, as ruled.
* **Sharding (approved).** Each entry runs as ≥ 2 shards of its **same** sampled list (global
  indices kept), each under the 2 h cap, followed by an integrity-checked merge that alone writes
  the registry.
* **One plan hash per entry.** The A5 re-run had started under v10 and was **stopped five minutes
  in** — one `rf` configuration, before any shard file or registry row — so that this amendment and
  its re-pin land **before** any entry runs. The partial log is archived in
  `stage2_select/smoke/aborted_v10_pause/`. Every entry is then run in full under v11, so no entry
  spans two plan hashes.
