# M4 ChemTox — architecture round v2 amendment: tracks T1 and T6

**New file, 2026-11-10.** It amends nothing in place. Every frozen artefact keeps its bytes: this
file edits no file under `docs/preregistration/` and nothing pinned by the T2 unseen-taxa freeze
(`chemtox_unseen_taxa_plan.md`, `chemtox_unseen_taxa_pins.json`, `unseen_taxa.py`,
`unseen_taxa_guard.py`, `test_blocks.parquet`, `training_pool.parquet`) or by the M5 BioAssess
freeze (`chemtox/{__init__,evaluate,features,tabpfn_model}.py`, `gpboost_model.py`). If anything
here has to change after the stamp, it changes by a dated addendum appended to this file, never by a
rewrite.

**Status: UNSTAMPED DRAFT. No fit, no validation number and no test look exists on T1 or T6.** The
lead stamps this file (sha256 + a `chemtox/prereg_snapshot` registry row + a commit) and that stamp
is the gate: until it exists, nothing on these two tracks may be fitted.

**Governing inputs.** `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` §"M4 ChemTox:
update 2026-11-10 12:25 EST" (which governs; where the plan differs, the decisions win),
`docs/plans/architecture_research/M4_architecture_plan.md` (§8.3 items 1–8, §8.2 caps, §4.0
references, §4.1 ablations, §5 targets, §7.1–7.2 tracks), `docs/plans/sota_campaign.md` rules 1–8,
RESEARCH_PLAN §15.1 and §15.5.

**What is NOT in scope.** T2 (unseen taxa) runs to completion exactly as pre-registered and is
reported whatever the outcome, including an honest negative; this amendment adds no entry, no
configuration, no architecture and no input to it. T3 (post-freeze temporal) stays deferred to the
next EPA release. T4 (mixtures) stays exploratory with no budget. T5 (cross-source) is **blocked on
a licence** (§10) and is recorded as blocked, not skipped. No spent block — the ADORE t-F2F
occurrence test or the prospective set — is re-scored, re-trained on or re-claimed.

---

## 1. The two tracks, and why they were chosen (sota_campaign rule 2: justification before any result)

Both justifications are recorded here, before any validation number on either track exists. Neither
track was chosen because a result favoured it; the only numbers that existed when they were chosen
are T2's in-flight selection scores and the counts in §3, and T2's are *unfavourable* (a mandatory
comparator leads it).

**T1 — unseen chemical clusters (new PRIMARY).** A hazard-screening model's real failure mode is a
chemical nobody has tested: a new pesticide, a new pharmaceutical metabolite, a spill of something
unusual. This is also the only regime in which M4 has ever seen a learned representation beat the
trees (chemprop 1.202 vs reference 1.287 on the ADORE-unseen-chemical rows of the spent prospective
block). The practical question it answers is the one a regulator actually asks of a new substance.

**T6 — exposure-duration extrapolation (SECONDARY; the genuinely harder task).** Standard tests run
24–96 h; the questions that matter after a spill are "what happens over a week?" and "what does a
one-hour pulse do?". Regulatory extrapolation across duration is done today with crude rules. This
is also the one axis where the comparator class is **structurally**, not rhetorically, handicapped:
beyond its last split a tree predicts a constant, and a monotone constraint fixes the direction of
that extrapolation but not its shape. The handicap is stated in advance, so the claim is falsifiable.

**T6 has two variants, both pre-registered now.** T6a: unseen duration, seen chemistry. T6b: unseen
duration **and** held-out chemical cluster. T6b is the smaller, stricter block; its MDE is computed
in §6 and it is reported as a full block, not promoted or demoted after the fact.

---

## 2. §8.3 item 1 — the T1 cluster file, the cutoff, and their hashes

**Blocking unit.** ECFP4 (Morgan radius 2, 2048 bits, RDKit `rdFingerprintGenerator`) Butina
clustering at Tanimoto **distance** cutoff **0.4**, over one canonical SMILES per `test_cas` (the
most frequent SMILES string recorded for that CAS in the frame). Whole clusters are held out.

**Why not Bemis–Murcko, and why 0.4.** Murcko scaffolds fail twice on this endpoint: 46.2 % of
usable acute rows are acyclic (metals and simple organics have no ring scaffold at all), and the
single benzene scaffold covers 24,443 rows / 1,052 chemicals — 20 % of the endpoint in one
"family". The cutoff is chosen on a purely structural criterion, declared before any fit: **the
largest cluster must hold under 10 % of the universe's rows.** At 0.4 the measured largest cluster
holds **5.34 %** (4,577 of 85,746 rows) across **2,603 clusters**, 2,067 of them single-chemical.
Acyclic chemicals and the benzene family are handled by the same rule, and the cluster is a
defensible resampling unit for the bootstrap.

**Frozen artefacts (0444, hashes recorded here before any model sees the split):**

| Artefact | Path | sha256 |
|---|---|---|
| Cluster assignment (3,895 chemicals: `test_cas`, `chem_smiles`, `t1_cluster`) | `docs/preregistration/chemtox_t1_clusters_v1.csv` | `a6e3363480c76c4a98d6377cfa44239c853168ee1b0be475c3fed1f20609fc4a` |
| Row partition, **label-free** (`result_id`, `test_cas`, `t1_cluster`, `duration_h`, `block`; 85,746 rows) | `$SENTINEL_DATA/processed/chemtox_arch_v2/chemtox_t1t6_split_v1.parquet` | `005f79578589225b520b84be92dac205dce507aff4d8e554a6251ca6a38f66cf` |
| Builder (deterministic; reproduces both files) | `$SENTINEL_RUNS/chemtox/arch_v2/build_t1t6_split.py` | `6fb9ed52b2f3c5cb4e761f1424b0d2b01c7239a86b66888788b421fd4a7c02a3` |
| Counts report | `$SENTINEL_RUNS/chemtox/arch_v2/t1t6_counts.json` | `1c443fed0f9751ed29e44d1684c5fb98b999a2b015ab133cfb4df6f85b7c1f2d` |
| MDE report (§6) | `$SENTINEL_RUNS/chemtox/arch_v2/t1t6_mde.json` | `c38a147dba3caf3b3f5516d9f0710b70e3d56a081d84bb9e86e253812d3a7e0f` |
| MDE script (calls the project helper) | `$SENTINEL_RUNS/chemtox/arch_v2/compute_t1t6_mde.py` | `2c171c0a6f69c5e6023493bde3c698d317c30610a9669932ff15f7a77dbec24a` |
| H4 join/QC report (§9) | `$SENTINEL_RUNS/chemtox/arch_v2/h4_bcf_join_qc.json` | `3ceead2ff05b3ccbee51cb77dbb8133b275725359a1f27a88c1300beb86857cd` |
| H4 QC script | `$SENTINEL_RUNS/chemtox/arch_v2/h4_bcf_join_qc.py` | `0659b5f742202b6a392128a9362c2c3e786b104520469fc72e81733268c0724b` |

The split file deliberately carries **no target column**: the frozen split is structure only, and
labels are read at score time through the guard of §8. A revision of either file is written as
`_v2` with this version preserved, as T2 preserved `chemtox_unseen_taxa_pins.v6/v8/v9.json`.

**Split seed.** `20261110`, declared here. Clusters are permuted with
`numpy.random.default_rng(20261110)` and accumulated until 15 % of in-window rows are reached →
**490 held-out clusters**.

---

## 3. §8.3 item 2 — the exclusion rules, and the measured partition

Three subtractions, each an **assertion** in the builder (not a comment), applied in this order:

1. **Usable acute endpoint.** `endpoint_norm ∈ {LC50, EC50}`, `censored == False`,
   `log10_conc_mol_L` present, `chem_smiles` present → **122,946 rows / 4,592 CAS / 2,550 species**
   (120,740 with `duration_h`). This reproduces the architecture plan's A.2 figures exactly.
2. **Class-level exclusion of T2's reserved blocks.** Rows whose ECOTOX `class` is in
   `{Insecta, Gastropoda, Bivalvia, Polyplacophora, Cephalopoda, Amphibia}` are removed outright:
   **−24,219 rows → 98,727 rows / 4,156 CAS / 1,586 species** (again exactly the plan's A.8
   figures). This is a **strict superset** of T2's reserved `result_id`s, so T2's three frozen
   blocks are kept out of T1/T6 **without reading `test_blocks.parquet`** — no frozen artefact is
   opened, at build time or ever.
3. **`result_id` subtraction of the spent prospective rows.** All 11,437 distinct `result_id`s in
   `processed/chemtox_prospective/prospective.parquet` are removed: **−11,437 rows → 87,290**. Then
   rows with no `duration_h` are dropped (both tracks' partition is defined on duration) and 18 rows
   whose SMILES RDKit cannot parse (2 chemicals) are dropped **explicitly and counted**, never
   silently → **universe 85,746 rows / 3,895 CAS / 1,572 species / 6,656 references**.

**The partition is one four-way, row-disjoint split of that universe**, so that no row is ever
scored twice and no block's look spends another block's rows (rule 1). In-window means
`24 ≤ duration_h ≤ 96`.

| Block | Rule | Rows | Chemicals | Clusters | Species | References | SD(target) | Replicate floor |
|---|---|---|---|---|---|---|---|---|
| `train_pool` | cluster retained **and** in-window | **59,729** | 2,960 | 1,979 | 1,362 | 5,145 | 1.786 | 0.504 (9,565 groups) |
| **`T1_test`** (primary) | cluster held out **and** in-window | **10,548** | 681 | 468 | 581 | 1,699 | 1.741 | 0.495 (1,747 groups) |
| **`T6a_test`** | cluster retained **and** out-of-window | **13,059** | 1,175 | 847 | 628 | 1,634 | 1.681 | 0.408 (2,088 groups) |
| **`T6b_test`** | cluster held out **and** out-of-window | **2,410** | 278 | 203 | 233 | 498 | 1.768 | 0.458 (337 groups) |

`T1_test` is **15.01 %** of the in-window rows, i.e. the plan's recommended 15 % block. Out-of-window
rows are **18.04 %** of the universe (5,486 below 24 h, 9,983 above 96 h; 322 distinct durations out
of window, 416 in the universe; maximum **26,298 h**).

**Shared-reference concentration, measured, because ECOTOX is publication-concentrated:** the share
of a block's rows whose `reference_number` also appears in the training pool is **74.4 % (T1_test)**,
53.8 % (T6a), 30.9 % (T6b). T1's 74 % is far above T2's 16–29 %, so the **reference-clustered CI is
co-primary on all three blocks** and a **`no shared reference` stratum** is reported for each
(§6). This is a disclosure, not a mitigation: a high shared-reference share means the primary CI
alone would overstate independence.

**One consequence stated plainly.** Because the pool is in-window only, T6's *selection* folds can
only extrapolate inside 24–96 h (§7), while T6's *test* blocks reach 274× beyond 96 h. Selection
therefore cannot verify long-range behaviour; only the single sealed look can, and the claim rests
on the look, not on selection. The alternative — putting part of the long tail into the pool — was
rejected because it would shrink the very regime the track exists to test.

---

## 4. §8.3 item 3 — reference models, comparators, and the parity statement

### 4.1 The parity statement (binding)

**Every input, every auxiliary target and every extra training row a candidate receives is given to
every mandatory comparator, on identical rows.** It is checked mechanically, not promised: the
runner logs, per fold and per entry, the training row count, the sorted feature-column names and
their hash, and the set of auxiliary targets used; a comparator whose row set or input set differs
from the candidate's by anything other than the estimator itself **fails the run**. This is the
C-M4P-1 lesson — the `crust_novel` result carries no claim because one model alone had extra rows —
and T2's claim condition (d2) in its general form. Ensembling parity too: if a candidate is an
ensemble, the strongest simple baseline is ensembled the same way with the same member count.

### 4.2 Mandatory references, built FIRST, never counted as candidates

| Ref | Model | Why mandatory | Status |
|---|---|---|---|
| **R1** | **Monotone-duration GBM** — LightGBM/XGBoost with `monotone_constraints` on **continuous** `duration_h` plus a log-duration spline basis | Kills the "a tree cannot be monotone in exposure duration" argument. Verified to fit in `sentinel2` (lightgbm 4.7.0, xgboost 3.2.0). On T6 it is the candidate's **direct rival** | to build (phase P1, CPU) |
| **R2** | **Interval-censored AFT GBM** — `xgboost` `objective=survival:aft` with `label_lower_bound`/`label_upper_bound` on the censored acute rows | Kills the "squared-error GBMs must drop censored rows, so the Tobit likelihood is neural-only" argument. Verified to fit | to build (P1, CPU) |
| **R3** | **Multitask-by-stacking GBM** — one GBM over acute + chronic rows with an endpoint indicator and per-endpoint offsets | Parity for the "more targets" lever (rule 4). If a GBM given the chronic rows closes the gap, no network is needed | to build (P1, CPU) |
| **R4** | **Every comparator gets the concatenated embedding block.** The frozen foundation-model embedding is an *input*, so parity requires it for R1–R3, `rf`, `xgboost` and the published configurations | The same one-time cache serves all of them | with the cache (P2) |
| **R5** | **TRIDENT head-to-head via its released per-SMILES 10×10-CV errors**, never by re-running TRIDENT | TRIDENT's training data overlaps our rows, so a re-run measures memorisation. Licence **cleared** (§10): CC BY 4.0, `03_Model_error_per_SMILES_from_10x10_CV.xlsx`, 4,825,857 B | to download (P1) |

R1–R3 are interesting results in their own right: if a monotone, AFT, multitask GBM closes most of
the T6 gap, that is a publishable negative that costs no GPU time, and §12 records it as the single
most likely outcome.

### 4.3 Published comparators and simple baselines (fixed configurations, no search)

* **ADORE published RF / XGBoost** (Gasser et al. 2024, DOI 10.1039/d4va00072b) — verbatim published
  configurations, the *simple-baseline* bar's published form.
* **Re-tuned `rf` and `xgboost` with the full input set** — the **strongest simple baseline** for
  claim condition (b). They are searched entries, not fixed, so the simple baseline is as strong as
  our own tuning can make it.
* **GPBoost (Zubrod et al. 2024, DOI 10.1016/j.envint.2024.108607)** — hyper-parameters verbatim
  from the authors' released `_without_DEB` models (both the fish form, `num_leaves` 64 /
  `lambda_l2` 1, and the invertebrate form, 32 / 10; `num_iterations` 1000, `learning_rate` 0.01,
  `max_depth` −1, `min_data_in_leaf` 10, fractions 1/1), manifest `chemtox_published_gpboost`.
  **On T1/T6 the taxonomy is in-distribution, so the authors' species/genus/group grouping is
  admissible here** and GPBoost is **not** the ADAPTED estimator it has to be on T2. The library
  difference (gpboost 1.8.0 vs the authors' 1.2.1) remains and is disclosed with every comparison.
* **MMoE (Wu et al. 2026, DOI 10.1021/acs.jcim.6c02487)** — re-implemented over our endpoint
  families as the **strongest published multi-endpoint architecture**, a *comparator and ensemble
  member*, never a novelty claim.
* **TRIDENT (Gustavsson et al. 2024)** via R5 — the strongest published method for this exact task.
* **`readacross`** — chemical-similarity kNN, with its `widening_` counters reported.
* **Trivial baselines** — global mean, per-species mean, per-chemical mean.
* **T6-only crude-rule baselines** — **Haber's rule** (`C × t` constant, the regulatory rule the
  track is motivated against) and **nearest-observed-duration carry-forward** (what a tree actually
  does past its last split). Both are the honest "crude extrapolation" floor T6 must beat.

**Which comparator is "strongest published" is decided by validation on the pool, inside the frozen
block, before the look** — never after seeing the test numbers. Condition (a) is read against that
one; every other published comparator is still reported.

---

## 5. §8.2 — configuration caps, with every configuration counted

Caps are as the lead's decisions set them (plan §8.2): **T1 384 searched, T6 128 searched, plus
fixed baselines.** No entry exceeds its track's per-entry cap, and **every configuration is counted,
including ones a candidate discards and ones that fail.** A failed fit is reported as a failure, not
dropped (T2's GPBoost-NaN precedent).

### 5.1 T1 — 12 searched entries × 32 configurations = **384 searched**

| # | Entry | Role | Notes |
|---|---|---|---|
| 1 | `ted` | candidate | ChemTox-TED (§6 of the plan's §4.1): D-MPNN + descriptors + MACCS + embedding → composition layer. GPU |
| 2 | `timehead` | candidate | frozen tabular model + learned monotone rate head. Cheap; run before `ted` |
| 3 | `rx_lightgbm` | candidate | published block **+ concatenated** embedding |
| 4 | `rx_catboost` | candidate | as above, CatBoost |
| 5 | `rx_chemprop` | candidate | `chemprop` D-MPNN single-task + embedding. GPU |
| 6 | `mmoe` | **comparator** + ensemble member | published L3 architecture |
| 7 | `r1_monotone_gbm` | **comparator** (R1) | monotone in continuous duration |
| 8 | `r2_aft_gbm` | **comparator** (R2) | interval-censored AFT |
| 9 | `r3_multitask_stack_gbm` | **comparator** (R3) | acute + chronic, endpoint indicator |
| 10 | `xgboost_emb` | **comparator** — strongest simple baseline | R4 parity arm |
| 11 | `rf_emb` | **comparator** | R4 parity arm |
| 12 | `lightgbm_noemb` | **information control** | identical to #3 with the embedding block removed; isolates what the embedding is worth |

### 5.2 T1 — fixed configurations: **11 across 10 entries**

`rf_published_adore` 1 · `xgboost_published_adore` 1 · `gpboost_published` **2** (fish form,
invertebrate form) · `readacross` 1 · `trivial_global_mean` 1 · `trivial_per_species_mean` 1 ·
`trivial_per_chemical_mean` 1 · `r5_trident_released_errors` 1 (**no fit** — a read of the released
file over the intersecting SMILES) · `ensemble_mean` 1 · `ensemble_decorrelated` 1.

**T1 total: 395 configurations across 22 entries.**

### 5.3 T6 — 8 searched entries × 16 configurations = **128 searched**

`ted` · `timehead` · `rx_lightgbm_dur` (candidates) · `r1_monotone_gbm` · `r2_aft_gbm` ·
`r3_multitask_stack_gbm` · `xgboost_emb_dur` (strongest simple baseline) · `mmoe` (comparators).
16 rather than 32 because T6's point is a **shape** claim, not a hyper-parameter search.

### 5.4 T6 — fixed configurations: **9 across 8 entries**

`rf_published_adore` 1 · `xgboost_published_adore` 1 · `gpboost_published` 2 · `readacross` 1 ·
`haber_rule` 1 · `nearest_duration_carry_forward` 1 · `trivial_global_mean` 1 ·
`r5_trident_released_errors` 1.

**T6 total: 137 configurations across 16 entries.**

### 5.5 Ablations (§7 below): **6 per track where `ted` is fitted = 12 fixed configurations**

Ablations are fitted on the **selected** `ted` configuration only, not searched.

### 5.6 Declared grand total

**384 + 11 + 128 + 9 + 12 = 544 configurations**, across 38 entries (22 on T1, 16 on T6; `ted`,
`timehead`, `mmoe`, R1–R3 and the published/trivial entries appear on both tracks and are counted
once per track because each track is a separate fit on a different training pool). The registry
carries these five figures on every row of both tracks so they cannot drift, and the final write-up
reconciles the realised count against them, including failures.

### 5.7 Compute budget and the pre-declared fallback

GPUs **0–9** through `python -m sentinel.core.gpu --name <job> --min-free-mb <n> --`, never a GPU
another user's process occupies, **≤ 2 h per job with checkpoints**, one configuration × one fold
per job (the `--fold-merge` pattern proven on T2's chemprop queue). CPU ≤ 6 threads, one heavy CPU
job at a time. **Budget 22–30 GPU-h for `ted`, ≤ 1 GPU-h for `timehead`, 8–12 GPU-h for `mmoe`,
1–2 GPU-h for the embedding cache.** **Pre-declared fallback:** if the timed smoke exceeds 30
min/job, `ted` drops to **8 configurations** per track and that is recorded, not renegotiated. VRAM
budget ≤ 4 GB of 11 GB so a second workstream can share the card, measured with
`torch.cuda.max_memory_allocated()` in the smoke. fp16 AMP with the composition layer **force-cast
to fp32 inside an autocast-disabled block** (fp16 `expm1` near zero loses all precision), with a
unit test asserting parity against an fp64 reference. **Disk:** `df` before every build or chunk,
hard stop below **60 GB free on `/`**, M4's new artefacts inside the **10 GB** cap (current
arch-v2 artefacts: 1.0 MB).

---

## 6. §8.3 item 5 — MDE per track and per stratum, by the project's own helper

Computed by **`sentinel.models.chemtox.unseen_taxa.mde`** — the project's own pinned helper, the
same function that produced T2's 0.011 / 0.019 / 0.021 — at `rmse_level = 1.2` (the error magnitude
this project actually produces), 2,000 cluster-bootstrap resamples, seed 0, α = 0.05. The resampling
unit is the block's own: **`t1_cluster`** for T1, **(`t1_cluster` × duration band)** for T6, so
85,746 rows from 11k (chemical, species) pairs are never treated as independent. Duration bands,
declared here: **< 24 · 24–96 · 96–168 · 168–720 · 720–2160 · > 2160 h**.

**From counts and metadata only.** The helper's statistic is the paired RMSE difference between two
*synthetic* predictors, so it is provably invariant to the target values. The script asserts this by
running every configuration twice, once with the real target and once with the target replaced by
zeros, and requiring the two to agree to 1e-12. **All twelve configurations below passed that
assertion**, so no label of any future test row informed any number in this table.

| Block | Role | Resampling unit | Stratum | Rows | Clusters | MDE (ρ=0.9) | MDE (ρ=0) |
|---|---|---|---|---|---|---|---|
| **T1_test** | primary | `t1_cluster` | all | 10,548 | 468 | **0.0097** | 0.0211 |
| T1_test | primary | `t1_cluster` | no shared reference | 2,697 | 130 | 0.0167 | 0.0505 |
| T1_test | **co-primary** | `reference_number` | all | 10,548 | 1,699 | **0.0094** | 0.0213 |
| T1_test | co-primary | `reference_number` | no shared reference | 2,697 | 572 | 0.0191 | 0.0471 |
| **T6a_test** | primary | cluster × band | all | 13,059 | 1,463 | **0.0085** | 0.0204 |
| T6a_test | primary | cluster × band | no shared reference | 6,031 | 900 | 0.0134 | 0.0303 |
| T6a_test | co-primary | `reference_number` | all | 13,059 | 1,634 | 0.0084 | 0.0216 |
| T6a_test | co-primary | `reference_number` | no shared reference | 6,031 | 854 | 0.0131 | 0.0288 |
| **T6b_test** | primary | cluster × band | all | 2,410 | 372 | **0.0197** | 0.0460 |
| T6b_test | primary | cluster × band | no shared reference | 1,666 | 269 | 0.0241 | 0.0529 |
| T6b_test | co-primary | `reference_number` | all | 2,410 | 498 | 0.0213 | 0.0460 |
| T6b_test | co-primary | `reference_number` | no shared reference | 1,666 | 340 | 0.0232 | 0.0562 |

**The binding threshold for a claim is the primary MDE at ρ=0.9 on that block and stratum**; the
ρ=0 column is the pessimistic bound and is reported beside it. Every block is better powered than
T2's best (0.011), and T1 and T6a are better powered than any block M4 has ever scored.

**The replicate floor is reported as an absolute RMSE bound, never as a difference threshold.** This
is the W-D-7 lesson, applied: pooled within-group SD over (CAS, species, duration, endpoint) groups
with ≥ 2 records is **0.504** on the training pool and **0.495 / 0.408 / 0.458** on T1 / T6a / T6b.
A model's absolute RMSE cannot honestly go below it; a *difference* of 0.02 between two models that
both sit near 1.2 is still meaningful, and conflating the two would be a category error. **No
practically-relevant difference threshold is declared**, because none can be justified independently
of the data.

---

## 7. §8.3 item 4 — the selection rule and the ablation set

### 7.1 Selection (validation only; the test blocks are never read during selection)

**T1 — 4 folds, leave-one-cluster-group-out.** The pool's 1,979 clusters are partitioned into 4
groups of near-equal row count by greedy largest-first assignment under seed `20261110`
(**14,334 / 15,945 / 14,325 / 15,125 rows; 439 / 432 / 566 / 542 clusters**). Each fold trains on
three groups and validates on the fourth. This mirrors how `T1_test` is built — whole clusters held
out, in-window durations — which is the A5 lesson from T2: **a selection fold must mirror the test
block's construction, or the chosen configuration is chosen on an artefact.**

**T6 — 3 duration-band hold-out folds inside the pool**, which mirror T6a (seen chemistry, unseen
duration): S1 train {24–72 h} validate {> 72–96 h} (22,971+ rows); S2 train {48–96 h} validate
{24–< 36 h} (13,890+); S3 train {24–48 h} validate {> 48–96 h} — a 2× extrapolation step. **For
T6b**, one combined fold holds out a cluster group **and** the top band at once. The disclosure of
§3 stands: the selection range is at most 2×, the test range up to 274×, so selection chooses a
configuration and the sealed look tests the shape claim.

**Score and winner.** Selection score = **mean RMSE over the folds**, with the worse half, the worst
fold, the per-fold scores and the top-5 spread all retained and reported. One winner per entry (the
lowest mean; ties broken by worst fold) is carried into the freeze. The frozen winner is refit with
**3 seeds**. Fold scores are validation-only and are never compared with the ADORE-CV or T2 numbers:
different pools, different blocking, so absolute RMSE is not commensurable across them.

**Ensembles.** `ensemble_decorrelated` selects members greedily to maximise
`mean pairwise (1 − corr(residuals))` subject to each member being within a declared margin of
**0.05 RMSE** of the best single model, with the member set **fixed before the test look**.
`ensemble_mean` is the unweighted mean of the frozen candidate winners. The strongest simple
baseline is ensembled identically with the same member count (§4.1).

### 7.2 The ablation set (plan §4.1) — pre-declared, because a win without it cannot be attributed

| Ablation | What it isolates |
|---|---|
| **A1** | D-MPNN trunk with free-form duration conditioning (TRIDENT-shaped), **no composition layer** — is the mechanism doing anything? |
| **A2** | Composition layer with `B = 0` (pure anchored law, no learned residual) — the physics-only floor |
| **A3** | Duration and endpoint fed to the trunk as well — **leak test**: was the restriction load-bearing? |
| **A4** | Acute-only training (no chronic, no censored rows) — the multitask contribution |
| **A5** | **+ BCF head** — admissible: the join/QC report has landed (§9) |
| **A6** | **+ Q10 temperature term** with a *free* prior (no literature constant is assumed; a prior centred on log 2 would be an unverified constant, so it is not used) |

**Attribution rule, binding:** if **A1** scores within the block's MDE of the full model, the win
belongs to the trunk and the data, **the mechanistic framing is dropped from the claim and the
paper**, and that is reported in the same table as the headline number. A claim for `ted` requires
A1 and A2 to be present; A3 must not improve on the full model (if it does, the restriction was not
load-bearing and that is reported).

---

## 8. §8.3 item 6 — the four scorer-side guard gaps, and the single-look discipline

`unseen_taxa_guard` already provides registry refusal, an `O_EXCL` claim marker with pid + host,
`code_sha256` on the claim, `load_test_block` as the only reader, write-once `save_predictions`,
`mark_scored`, `seal_dir` (0444 + `SHA256SUMS`) and `verify_freeze`. Each new track's scorer closes
the four gaps that are scorer-side, not guard-side:

1. **A lead-written `FREEZE` file is the only unlock for the test look** — never a file this
   workstream can write. The scorer refuses to proceed without it and verifies its sha256 against
   the value the lead records at the stamp.
2. **`BaseException` aborts the pass** before any claim, prediction or registry row is written
   (`except BaseException: cleanup; raise`), so a `KeyboardInterrupt` or an OOM cannot leave a
   half-written claim.
3. **A full import-closure code hash**, computed by the scorer over every module reachable from its
   own imports, and recorded as `code_sha` on the claim. This replaces the stale-package-hash
   problem that is still an open 🟡 on the prospective track.
4. **All artifacts verified before the claim** (every prediction file re-hashed against its
   `SHA256SUMS`), and **access closed at sealing** (0444 on the directory's contents).

**One look per block, ever.** `T1_test`, `T6a_test` and `T6b_test` are three separate blocks, each
scored exactly once, each with its own FREEZE file and its own claim marker. A rescore is refused
after scoring. Write-ups go to `chemtox_t1_test_results.md` and `chemtox_t6_test_results.md`, never
into a frozen file. **Reserved from today**, and never used for training or tuning by any
workstream: the rows of all three blocks, as listed by `block` in
`chemtox_t1t6_split_v1.parquet`.

---

## 9. §8.3 item 7 — the H4 (BCF) decision: the report landed, so A5 is IN

The lead's condition was "H4 enters only as ablation A5 and only if its join/QC report lands first;
otherwise dropped and recorded". **The report has landed** (`h4_bcf_join_qc.json`, sha256
`3ceead2f…`), so **A5 is admitted** — as an ablation of `ted` only, never as a track, a target of a
claim, or an input to any comparator-asymmetric arm.

**What the report measured, and the defect it found.** The architecture plan declined to assert a
usable row count; three things had to be measured.

* **The BCF value is in `bcf1_mean` / `bcf1_unit`, not `conc1_mean`.** On these 27,725 rows
  `conc1_*` is the *exposure water concentration* (`ug/L` 16,372, `mg/L` 3,110, …). Reading
  `conc1_mean` — the obvious thing to do, and what a first pass here did — would have silently
  trained a bioconcentration head on the exposure concentration. Recorded because it would not have
  raised an error.
* **Unit / basis reconciliation.** Admissible units are only those that are exactly dimensionless-
  equivalent: `L/kg` (20,577), `ml/g` (732) and blank/`--` → **21,494 rows**. Everything else is
  excluded and counted, including `RA` (4,498, a relative-activity code, not a BCF), `L/g`, `ml/mg`,
  `kg/L` (inverted), `mg/L`, `NR`/`NA`. The convertible ones (`L/g`, `ml/mg`) would add ~0.9k rows
  but each needs its own factor decision, so they are **excluded** rather than guessed.
  **BCFD/BAFD is dry weight** and dry- and wet-weight factors differ by roughly 4–5×, so the basis
  is carried as a categorical feature and **never pooled silently**: 14,951 wet / 2,467 dry.
* **CAS join and plausibility trim.** Digit-stripped CAS → the SMILES already resolved for the
  aquatic frame: 25,315 of 27,725 rows (790 of 990 distinct CAS). A CAS absent there is not
  resolvable without a new structure lookup, and none is done. Trim: `log10 BCF ∈ [−2, 7]`,
  removing 42 rows.

**Usable H4: 17,418 rows / 672 chemicals / 479 species** (16,590 uncensored; BCF 14,388 / BCFD 2,467
/ BAF 563; SD of log10 BCF 1.408). **588 of the 672 chemicals are in the T1/T6 universe and 456 are
in the training pool**, so the head has something to share parameters with. The circulating
"21,352 usable" figure is **not** reproducible as a usable count: 21,309 is the raw `L/kg` + `ml/g`
row count *before* the value, structure and plausibility filters.

---

## 10. New targets, data and licences

### 10.1 Targets (plan §5), all counted from on-disk frames

| ID | Target | Rows | Role here |
|---|---|---|---|
| **H1** | log10 LC50/EC50 (mol/L) | 122,946 usable (85,746 in the T1/T6 universe after §3's subtractions) | the primary target, unchanged |
| **H2** | the same target, **interval-censored** | 11,719 usable (13,426 censored in total) | added as auxiliary signal via R2 and `ted`'s Tobit term. The circulating "~21.5k" figure is wrong by ~1.8× |
| **H3** | log10 chronic effect concentration (NOEC/LOEC/EC10/MATC/NOEL/LOEL) | 225,081 | added as an auxiliary task (R3, `ted`, `mmoe`) |
| **H4** | log10 BCF/BCFD/BAF | **17,418 usable** (§9) | ablation A5 only |
| **H5** | log10 **acute-to-chronic ratio**, with an interval | identifiable on 10,334 (chemical, species) pairs | a **reported output, not a §15.1 label**, unless a pre-registered comparator exists — and none does, so **no SOTA claim may rest on it** (lead decision) |
| **H6** | per-row predictive sigma / calibrated interval | same rows as H1 | reported (coverage + NLL); table stakes against L1/L3, not a novelty |
| **H7** | within-one-bin severity class | same rows as H1 | a reported metric, L5's finding that a deep model can win here while only tying on regression |

Auxiliary rows (H2, H3, H4) obey §4.1 parity: whatever auxiliary target `ted` trains on, R3 and
`mmoe` train on the same rows. The §3 subtractions apply to auxiliary rows too — no chronic or BCF
row whose `result_id` is in the prospective frame, and no row from a T2-reserved class, enters any
training set.

### 10.2 Licences (both verified today; entered in `docs/licenses.md`)

* **TRIDENT Zenodo record 10260621 — CLEARED.** `cc-by-4.0` (record "Rights": Creative Commons
  Attribution 4.0 International), access `open`, v2, created 2023-12-05, 8 named creators. Both
  files the plan needs exist: `03_Model_error_per_SMILES_from_10x10_CV.xlsx` (4,825,857 B) and
  `02_Preprocessed_complete_data.xlsx` (30,897,579 B) — **35.7 MB downloaded, against M4's 10 GB
  cap**; the 2.8 GB weights and 3.2 GB source archives are **not** downloaded. **R5 is therefore
  admissible.**
* **EnviroTox — NOT CLEARED.** The site has no licence or terms-of-use page; the About page reads
  "Copyright © 2019 Health and Environmental Sciences Institute (HESI). All rights reserved." and
  disclaims all warranties; reuse permission is not stated and no download file could be located.
  **T5 (cross-source transfer) is blocked and is recorded as blocked**, not quietly dropped. It
  would need written permission from `ecoTTC@hesiglobal.org` first. EFSA OpenFoodTox and EPA
  Web-ICE v4.0 remain unverified and are not used.
* **REACH / ECHA** — untouched (standing lead ruling: only if `docs/licenses.md` clears it).
* **The embedding model is gated on its own licence.** MolFormer-XL (L9, Ross et al. 2022) is the
  intended source, with ChemBERTa-2 as the alternative; **neither is used until its weight licence
  is verified and entered in `docs/licenses.md`.** Pre-declared fallback if neither clears: the RX
  arm uses the in-tree Chemeleon embedding (already used by `chemeleon_lightgbm`) plus an expanded
  RDKit descriptor block, and that substitution is reported as a limitation, not hidden.

---

## 11. §8.3 item 8 — dated novelty check (2026-11-10), including L2, L3 and L4

Prior art checked against what this amendment proposes to claim. Bibliographic metadata for L1–L11
was retrieved on 2026-11-10 (Crossref / web search; see the architecture plan §2). **Numbers inside
L2, L3, L4 and L6 were NOT verified — only their metadata — and no number from them is relied on
here.**

| Work | What it already does | What it pre-empts | What is left |
|---|---|---|---|
| **L1 TRIDENT** (Gustavsson et al. 2024, *Sci. Adv.* 10(10):eadk6669) | SMILES transformer + DNN head **conditioned on log duration** and endpoint one-hots, trained jointly on EC50 and EC10 | duration *conditioning*, joint multi-endpoint training, released weights and errors | it **conditions** on duration; it does not **compose** duration through a differentiable law, and it does not evaluate duration as a hold-out |
| **L2 Liang et al.** (*ES&T*, 2026-11-02, DOI 10.1021/acs.est.6c12447) — published 8 days ago | links fragmented ECOTOX records into model-ready datasets; multitask with endpoint-specific heads over shared chemical/biological/experimental context | **the "more ECOTOX data + multitask" framing of our own levers L1/L3** | it is a comparator-class method, not a duration-extrapolation or cluster-hold-out evaluation. Its reported pooled aquatic R² ≈ 0.75 and its finding that multitask gains shrink with structural separation are **[UNVERIFIED]** and are not used |
| **L3 Wu et al. MMoE** (*JCIM* 66(20):13302, DOI 10.1021/acs.jcim.6c02487) | multi-gate MoE over 6 endpoints, **EC50 ≥ EC10 ordering constraint**, heteroscedastic uncertainty | **the "obvious next architecture": ordered multi-endpoint heads with uncertainty.** This is why `mmoe` is a comparator and ensemble member here, never a novelty claim | our ordering is derived from **one latent plus a non-negative ACR that is itself a reported output with an interval**, rather than imposed in output space |
| **L4 MarineTox** (Zhu et al. 2026, *ES&T* 60(21):14813) | multitask transfer from data-rich freshwater to 31 low-resource saltwater tasks | multitask transfer as the fix for data-poor taxa | a taxon-transfer claim, which is T2's (negative) business, not T1/T6's |
| **L5 3DMol-Tox** (Yuan et al., bioRxiv 2025.11.24.690199) | leakage-controlled benchmark; a 3D deep model **only matches GPBoost** on regression, better on within-one-bin | that a fancier encoder alone will win | its own recommendation — structure-aware splitting — is what T1 implements; it does not evaluate duration |
| **L10 GUTS** (Jager et al. 2011) | the standard TKTD survival framework from which time-dependent LC50 follows | **the mechanism itself: it is NOT novel as mechanism** | making it differentiable and training it jointly across heterogeneous endpoint families, and evaluating it as a duration hold-out |

**The honest claim this amendment permits, stated before any fit:** *"mechanism-composed time and
endpoint structure, evaluated on a pre-registered duration-extrapolation block and a pre-registered
novel-chemistry block, against published duration-aware and multitask comparators at input parity."*
Not "a new architecture beats SOTA". A SOTA label, if it comes, is scoped to its block and never
generalised, and RMSE is never averaged across blocks (rule 3).

**Also recorded:** the wrong article title for L6 in `gpboost_model.py:11` is a known defect, that
file is covered by an existing freeze and is **not** edited; the correct title is used here and in
every write-up so the wrong one cannot reach the paper.

---

## 12. Claim conditions, gates and kill rules

### 12.1 A §15.1 label on T1, T6a or T6b requires ALL of

| | Condition |
|---|---|
| (a) | paired CI (cluster bootstrap, 2,000 resamples, unit = the block's resampling unit) **excluding 0** against the **strongest published method** on that block, chosen on pool validation inside the frozen block before the look |
| (b) | the same against the **strongest simple baseline** |
| (c) | **Holm correction** across the candidate family within that block |
| (d) | **≥ 3 seeds**, with `all_seeds_better` reported |
| (e) | **one look** at the sealed block, unlocked only by the lead-written `FREEZE` file (§8) |
| (f) | the point difference **exceeds the block's primary MDE at ρ=0.9** (§6) for that block and stratum |
| (g) | **input parity**: the candidate beats **every** comparator that received the same inputs, auxiliary targets and rows (§4.1). A candidate that wins only because it alone received an input does not qualify |
| (h) | the **reference-clustered co-primary CI also excludes 0**, and the `no shared reference` stratum is reported |
| (i) | for `ted`: **A1 and A2 present**, and A1 **not** within the MDE of the full model (§7.2) |

### 12.2 Kill and fallback rules, all pre-declared

1. **Smoke over 30 min/job → `ted` drops to 8 configurations** per track. Recorded, not renegotiated.
2. **A selection winner that is a comparator is an honest negative**, reported with the same
   prominence as a win (rule 3), exactly as T2 is being reported.
3. **R5 fails → recorded, not worked around.** If the TRIDENT per-SMILES file does not contain
   usable per-SMILES errors, R5 is dropped, condition (a) is read against the strongest remaining
   published comparator (`mmoe` or `gpboost_published`), and the inability to compare with TRIDENT
   is disclosed beside every claim. **TRIDENT is never re-run on our rows** — that measures
   memorisation.
4. **The embedding licence fails → the Chemeleon fallback of §10.2**, reported as a limitation.
5. **A fit failure is reported as a failure**, with its configuration counted, never silently
   dropped (the T2 GPBoost-NaN precedent).
6. **Disk below 60 GB free on `/` → stop and report.** Checked before every build, chunk and shard.
7. **A `BaseException` aborts the pass** with no claim and no prediction written (§8).
8. **If T6b is underpowered in practice** — i.e. the realised candidate–comparator difference is
   below its 0.0197 MDE — it is reported as a measured null for that block. It is **not** demoted to
   "a stratum of T6a" after the fact; its status is fixed here, as a full block with its own look.
9. **Nothing here may be used to lower a bar.** If a result is within the MDE, the answer is "not
   resolvable on this block", not a weaker claim.

### 12.3 Honest outlook, recorded before any fit

The single **most likely** outcome (~40 %) is that **the references win**: R1–R3 close most of the
gap, the networks tie, and M4's honest result is "a monotone, interval-censored, multitask GBM on
all of ECOTOX is as good as anything we can build". That is a good result, it is cheap to reach, and
it is reported with the same prominence as a win. Subjective P(§15.1 label): **T1 ~25 %, T6a ~30 %,
T6b ~15 %**; P(at least one label within ~4 weeks) **35–40 %**. The second-most-likely failure is
that **the mechanism is not what works** (A1 ties the full model), in which case the claim becomes a
representation-plus-data claim — which L1 already made in 2024 — and the mechanistic framing is
dropped.

---

## 13. What may happen before the stamp, and what may not

**Permitted now** (lead's rule 2 for this round): build the P1 reference models R1–R3 on non-frozen
data; write code and unit tests; verify licences and write the `docs/licenses.md` entries (**done**,
§10.2); run the H4 join/QC report (**done**, §9); build the split and compute MDEs from counts and
metadata (**done**, §§2–6); smoke tests on synthetic or train-only data.

**Forbidden until the lead stamps this file:** any fit on T1 or T6, any validation number on either
track, any embedding run that touches T1's split, and any read of `T1_test`, `T6a_test` or
`T6b_test` beyond the row counts in §3.

**State at the time of writing:** no T1/T6 fit has run; no T1/T6 registry row exists; no `FREEZE`
file, claim marker or `SCORED.json` exists for any new block; T2's three frozen blocks remain
unspent (`registry_rows()` 0/0/0) and untouched by anything in this amendment.
