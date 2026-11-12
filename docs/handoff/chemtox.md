# Handoff — M4 SENTINEL-ChemTox v2 (updated 2026-11-09; prospective phase CLOSED)

**The prospective set has been scored once** (2026-11-09; 189 `chemtox_prospective_test` rows —
see the status section below). The ADORE test result stands unchanged ("competitive, not SOTA").
Prospective set frozen, sha256
`3b2fc4398db940a26c5273f815fdfe872a08b620e1f1b36bd4a12bb38dcc54d9` (details: plan §1).

## Running

All PIDs from the 2026-10-25 note are dead (host reboots 4 and 6 Nov); every job they ran finished.
No M4 job is running. `chemeleon_tabpfn` re-ran on GPU 4 (best 5-fold 1.0427) and the freeze is
**FINALIZED** (2026-11-08 14:20): frozen hash **`630b43e905d9b852`**, `selection.json` sha256
`bf2ed61c…237328`, stage 2 appended to the plan. Queue scripts: `$SENTINEL_RUNS/chemtox/m4_queue/`.

Round-4 review request: `docs/handoff/chemtox_round4_note.md`.

## Prospective pass status (2026-11-09) — PHASE CLOSED

**Final status: competitive, not SOTA.** No candidate beats either reference after Holm correction
on the primary track or the strict co-primary. The one Holm-rejected result (`crust_novel`) is a
**data-coverage effect, not a method improvement, and carries no claim**. The `_trainpart`
sensitivity entry was not obtained and will not be retried (D2). Nothing further is pending on this
block; the next step is the follow-on proposal at the end of this note.

**Scored once** (re-run after a pre-target crash, plan deviation D1). Registry
`chemtox_prospective_test` 189 rows (9 tracks × 21 models) + `chemtox_prospective_test_posthoc` 44
(W2). Protected report `$SENTINEL_RUNS/chemtox/prospective_test/prospective_results_primary.json`
sha256 `709803d052dfa9f0c117f3d64af82b55e82616a1bf8dde88460ced6bb686eb0b` (0444), `score_primary.log`
`2e96d089467c95f1…`; W1/W2 in `prospective_posthoc.json` (`scripts/eval/chemtox_prospective_posthoc.py`).
Primary: **competitive, not SOTA** (stack 1.208 vs reference 1.237 / XGBoost 1.218, no Holm rejection).
Secondary `crust_novel`: `gpboost_multitaxa` is Holm-rejected vs both references **within that
track only**, but this is a **training-data effect, not a method improvement, and carries no
claim** — it is the only scored model trained on ADORE's crustacean rows, and the published RF
refit on the same rows scores 1.271 vs 1.306 (post-results review C-M4P-1; post hoc). Full
caveats in the model card. Crashed attempt archived in `crashed_pass_20261108_1958/`.

**`_trainpart` sensitivity pass: RAN AND FAILED** (2026-11-09 01:59 → 04:06, GPU 1, rc 1 —
GPBoost conjugate-gradient NaN fitting `gpboost` seed 2; plan deviation D2). 0 `_trainpart` registry
rows, no report; 11 partial predictions archived read-only in
`prospective_test/failed_trainpart_20261109_0406/` (`SHA256SUMS` sha256 `793f01190cbcbb21`).
Primary seals re-verified after it: `pred_primary_SHA256SUMS.txt` 18/18 OK, crashed archive 16/16
OK, `prospective_results_primary.json` sha256 unchanged. **Outstanding decision for the lead:**
re-run as frozen (gpboost may fail again) or report the sensitivity entry for the 15 fittable models
and label `gpboost` "not fittable on the training partition". Nothing about the primary claim
changes either way.

Post-results review 🟡: C-M4P-1 relabel and W-M4P-1..6 done in the write-up; D1 seed-0 cause
recorded as **unexplained** after a CPU-only diagnosis.

## Masked CV (ADORE folds 0-4, best 5-fold finalist per family; `$SENTINEL_RUNS/chemtox/cycle2/*_dm*`)

| Family | role | RMSE |
|---|---|---|
| catboost | candidate | 0.9480 |
| chemeleon_gpboost | candidate | 0.9504 |
| chemprop v1 (`mw_logp` / frozen), single configs | candidate | 0.9527 / 0.9564 |
| chemeleon_lightgbm | candidate | 0.9593 |
| gpboost_multitaxa | candidate | 0.9669 |
| **xgboost re-tuned — strongest comparator (the bar)** | comparator | **0.9733** |
| lightgbm_aug (ECOTOX) / lightgbm | candidate | 0.9797 / 0.9865 |
| rf_published_mwlogp (published reference) / xgboost_published_mwlogp | comparator | 0.9909 / 0.9945 |
| rf re-tuned | comparator | 0.9928 |
| gpboost | candidate | 1.0158 (negative) |
| readacross | comparator | 1.1546 |
| embeddings v1 / tabpfn v1 | not frozen (not named in plan §3) | 1.1579 / 1.2397 |
| rf_published / xgboost_published (`four`, ws+mp imputed) | comparator | 1.2561 / 1.4600 |
| chemeleon_tabpfn | candidate | 1.0427 (config 0 lost to an external CUDA OOM, re-screened alone: 1.3485, rank 26/32, finalists unchanged) |

Supersedes the 10-25 conclusion: four candidate families now beat the 0.9733 bar on CV. These are
CV numbers only. Every family's winner uses `chemical=mw_logp`.

## Lead decisions (2026-11-08)

* Re-tuned XGBoost stays a comparator and is the strongest-baseline bar; the published
  configurations stay as additional reference comparators — report both.
* Tuning parity: no family gets more than 32 configurations (no `chemeleon_gpboost` re-screen; its
  screen 0.9194 vs 5-fold finalist 0.9504 gap is an observation for the review note).
* Species embeddings (Yang et al. 2026) **deferred** to a separately pre-registered follow-on stage
  after this freeze.
* Never use `--i-have-approval` until the lead explicitly authorises it.

## Next

1. Done: freeze + registry + stage 2 (hash above).
2. Round-4 review (lead runs it) must pass; then the lead explicitly authorises M4 to run the pre-flight / single scoring pass (plan §6, and
   the 10-25 runbook steps 3-5: dry-run pre-flight, the pass on GPU 4, `_trainpart` sensitivity,
   model card, RESULTS.md, render_results, post-results review).

Open 🟡 from round 3: `--allow-unmasked` leaves no trace in the frozen block (unused); §6 package
hash goes stale as other workstreams edit `sentinel/`. TRIDENT deliberately not run (training data
overlaps the prospective period).

## 2026-11-08 22:03 — lead note
M5's frozen test pass hashes `sentinel/models/chemtox/{__init__,evaluate,features}.py` and `chemtox/tabpfn_model.py`.
Do not edit those files until the M5 pass finishes (see docs/handoff/bioassess.md); put changes in new files instead.

## Follow-on M4 track (unseen taxa) — STAGE 1 RE-FROZEN 2026-11-09. Awaiting DELTA review

The first pre-registration review returned **FAIL** (`REVIEW_REPORT_PRE_TRAINING.md`, "M4 ChemTox
unseen-taxa — pre-registration review"): the data freeze was clean, the **design** had six blocking
problems. All six are closed and the plan is re-frozen. **No fitting has happened and none will
before the delta review passes.**

| Review item | What was wrong | Fix |
|---|---|---|
| **C-M4U-1** trait drop not implementable; the silent default would have given every test row one identical fish-derived trait vector | `FeatureSpec.published` is all-or-nothing and `features.py` is M5-hashed | New `unseen_taxa.TaxonBlindBuilder` wraps `FeatureBuilder` and deletes every `TRAIT_PREFIXES` column from the matrix **and** `names_`; `to_model_schema` supplies constant placeholders only so the in-tree block can fit; `assert_no_trait_columns` + a test prove none survives (8 dropped, 160 kept) |
| **C-M4U-2** no comparator could transfer to an unseen class | species/genus/family/order all disjoint, so every taxon-aware baseline collapses to a constant | New mandatory `fish_surrogate_geomean` / `_most_sensitive` and `ice_regression` (Web-ICE form, re-fitted, 16,998 pool pairs). Applicability measured and pre-registered: **90.8 % / 89.6 % / 82.2 %** of insect/mollusc/amphibian primary rows; no-surrogate rows take the pool mean and every surrogate comparison is reported **twice** (full endpoint + surrogate-available subset) |
| **C-M4U-3** GPBoost — the strongest published cross-species method — was optional, so no SOTA claim was reachable | Zubrod 2024 / Yuan 2026 | `gpboost_published` is now a **mandatory comparator**; new `unseen_taxa.SeededGPBoost` seeds `GPModel(seed=…)`, pins `num_parallel_threads` and switches to `matrix_inversion_method="cholesky"` (the D1/D2 defects); a determinism check (in-process **and** cross-process with `PYTHONHASHSEED=0`) is required in stage 2; a fit failure is **reported as a failure** with a pre-declared downgrade rule. Dated `docs/novelty_checks/chemtox_unseen_taxa.md` added |
| **C-M4U-4** chemprop would silently route unseen taxa to its **fish** head | `_task_index` maps unknown `tax_group` to 0 | chemprop pre-registered **single-task**; `assert_taxon_agnostic` makes `multitask=True` inadmissible, with a test; `readacross`'s 100 % fallback to the global pool is now stated and its `widening_` counters reported |
| **C-M4U-5** the hold-out guards were silent no-ops on the frames the pipeline builds, and NA-permissive | ADORE-schema/aux frames carry no `class`/`result_id` | Both guards now **raise** when the keying column is missing, accept `tax_group` for ADORE-schema frames, and treat `class`=NA as held out unless `taxon_group` is an ADORE group; 6 negative cases tested |
| **C-M4U-6** no one-time scoring guard; frozen files unprotected; the test could skip | `testlook.py` pattern unused | New `sentinel/models/chemtox/unseen_taxa_guard.py`: pins, `verify_freeze` (plan included), `authorise` (registry refusal + `O_EXCL` marker), **`load_test_block` as the only reader**, write-once `save_predictions`, `mark_scored` + `seal_dir`. Both parquet files and `report.json` are now **0444 with `SHA256SUMS`**; the test pins all hashes and **fails, not skips** |

Warnings also closed: computed **MDE** per block (insects 0.011 / molluscs 0.019 / amphibians 0.021
at ρ=0.9; the binding floor is the replicate noise 0.649 / 0.437 / 0.402, and no claim may rest
below the larger of the two); **reference-clustered CI is now co-primary** (21 % / 29 % / 16 % of
primary rows share a publication with the pool) plus a "no shared reference" stratum;
`parent_key` added to the validation-split de-duplication; selection moved from one fish↔crustacean
pair to a **leave-one-class-out rotation over the four pool classes ≥ 1,000 rows** (algae are 0.17 %
of the pool), with all four per-fold scores and the top-5 spread reported; `chemical="mw_logp"`
pre-registered with a named schema adapter (the `"four"` default would have crashed on `chem_ws`);
mollusc wording restricted to Gastropoda + Bivalvia; `best_baseline` labelled descriptive.

**Re-frozen artefacts.** Plan sha256 `4a047b844bb2c1ef…`, module `729864a16c213d1e…`, guard
`f8b4037a8a40db61…`, pins `chemtox_unseen_taxa_pins.json`, data **unchanged**
(`649dc7d11587ea2b…` / `7e915fbe9a274afe…`), registry `prereg_snapshot_v2` (7 rows). Tests: 7 new
+ 29 existing green. Untouched: `chemtox/{__init__,evaluate,features,tabpfn_model}.py`.

**§9 of the plan now states the design caveat before the fact:** with no trait block, no shared
lineage and no species embedding, no model has a cross-taxon channel while the surrogate/ICE
comparators do — so the expected honest outcome is a quantified negative, and a SOTA claim is close
to unreachable unless a legitimate cross-taxon channel is added in a stage-2 amendment.

### Stage-1 amendment (2026-11-09, lead decision, before any numbers): two cross-taxon channels

The track was close to unwinnable by construction (no model had a cross-taxon channel; the
surrogate/ICE comparators did). Both channels are now pre-declared in protocol §1.4, implemented in
`unseen_taxa.py`, and **given to the mandatory comparators as well**:

* **Channel A — surrogate inputs** (`SurrogateFeatures`, 7 columns): fish and pool geomean, the
  most-sensitive value, the ICE prediction, an availability flag, record and pool-taxa counts —
  computed **only from training-pool rows**. Plus residual-learning candidates
  (`ResidualSurrogateModel`: ICE/surrogate base + learned correction).
* **Channel B — ranks above the held-out class** (`HigherRankFeatures`, 13 one-hot columns):
  phylum / subphylum / clade, from a mapping that is a **pure function of the class name**, so
  nothing at class rank or below can enter. Bridges: amphibians share phylum + subphylum with the
  fish pool, insects share phylum with the crustacean pool, molluscs share clade only — a
  pre-registered expectation that molluscs gain least.
* **Fairness (C-M4P-1 lesson):** `rf_chan`, `xgboost_chan`, `rf_multitaxa_chan`,
  `xgboost_multitaxa_chan`, `rf_published_chan`, `xgboost_published_chan` and
  `gpboost_published_chan` are mandatory comparators, and claim condition (d2) requires a candidate
  to beat them. A candidate that wins only because it alone got a channel does not qualify. The
  SOTA bar is therefore **harder** than before the amendment.
* **Selection unchanged:** leave-one-class-out rotation, with the surrogate block re-fitted on each
  fold's training classes only. Channels are a feature dimension inside the 32-configuration
  budget: **608 searched configurations + 5 fixed baselines**, counted in the plan.
* **Leakage tests (4 new):** overwriting every test target leaves all 7 surrogate features
  bit-identical; fitting the block on pool+test raises; unmatched chemicals get
  `surr_available=0`/0 records/pool mean; a hand-computed fish geomean matches; no held-out class,
  genus, family or order can appear as a Channel-B name or value; `assert_no_class_or_lower` +
  `assert_no_trait_columns` run for every model in `ChannelBuilder.fit`.

**Re-frozen shas (data unchanged):** plan `c4891d7266a8a114…`, module `f161c71f9b48d03b…`, guard
`f8b4037a8a40db61…`, pins `6937c82b8a7d1662…`, novelty check `1c8f222e70b4f473…`, tests
`174d8295cedf9599…`, `test_blocks.parquet` `649dc7d11587ea2b…`, `training_pool.parquet`
`7e915fbe9a274afe…`. Registry `prereg_snapshot_v3` (7 rows, supersedes v2). 40/40 tests green.

### Delta review (🟡 PASS WITH CONCERNS) — D-1/D-2/D-3 + warnings closed 2026-11-09

| Item | Fix |
|---|---|
| **D-1** parent-key fold blocking was a silent no-op (`training_pool.parquet` has no `parent_key`) | `add_parent_key` derives it from the frozen file's own canonical SMILES (no rebuild); `validation_splits` now **raises** via `require_parent_key` instead of skipping. Folds shrink: train rows 676 / 8,378 / 11,523 / 24,505 (was 838 / 9,724 / 12,750 / 26,806); plan table corrected; a test proves the parent step removes rows CAS/InChIKey blocking would have kept |
| **D-2** four Channel-B levels were zero in the pool and non-zero only on test rows (one-to-one class proxies) | `HigherRankFeatures.fit` keeps only levels **non-constant in the fitting frame**, per fold and at final fit: **9 of 13 kept**, `Mollusca`/`Hexapoda`/`Conchifera`/`Aculifera` dropped. Plan §1.4.2 corrected and now states what each block actually gets — amphibians get the fish vector verbatim (a relabelling), insects "arthropod not crustacean", molluscs **only** `Protostomia`. Test asserts every retained level varies in training |
| **D-3** two definitions of "surrogate-available" | One rule: **any pool record, by CAS or `parent_key`** (`surrogate_available`), used by the feature, the baselines, the coverage report and the §6(c) claim gate. Corrected numbers: **94.6 % / 93.1 % / 87.6 %** (insects/molluscs/amphibians); fish-only figures retained for transparency only |
| W-D-1 | `verify_freeze` now checks the pinned `unseen_taxa.py` and `unseen_taxa_guard.py` sha256 |
| W-D-2 | guards accept `tax_group` and ADORE's `crusta` vocabulary (tested) |
| W-D-3 | one configuration inventory table: **608 searched + 12 fixed across 31 model entries** (registry carries the same three figures) |
| W-D-4 | ICE is fish → **all non-fish** pool rows, **17,081** pairs, `a=−0.650, b=0.982`; `MIN_PAIRS=30` identity fallback registered, with per-fold pair counts (0 / 701 / 1,263 / 4,688) |
| W-D-6 | guard wording softened: the choke point is a convention plus a 0444 file, not a technical impossibility |
| W-D-7 | **claim rule (g) replaced**: gated on the **MDE** only; the replicate floor is reported descriptively as an absolute-RMSE bound, never as a difference threshold; a practically-relevant difference may be added only if justified independently of the data (none declared) |

Two further defects found by my own new tests while fixing the above: `FishSurrogateBaseline` silently
re-fitted on non-fish rows when the surrogate groups were absent (would have changed the estimator
in a fold and made ICE circular) — removed, now records `groups_present_=False` and falls back to
the frame mean; and the fallback level was taken from the subset rather than the full frame.

**Re-frozen shas (data unchanged):** plan `36fe088f665d0484…`, module `33b0c7211051c475…`, guard
`e3721450bb97727c…`, pins `fff1ba54ac4c4ac7…`, novelty check `1c8f222e70b4f473…`, tests
`f70ea16607147915…`, `test_blocks.parquet` `649dc7d11587ea2b…`, `training_pool.parquet`
`7e915fbe9a274afe…`. Registry `prereg_snapshot_v4` (9 rows, supersedes v3). `verify_freeze()` passes
on all four pins. **46/46 tests green** (17 unseen-taxa + 29 prospective).

**Next: the reviewer's quick confirmation, then stage 2 (selection) may start.** No fit has run.

### 2026-11-09 (resumed) — stage 2 STARTED; state corrections

**Correction to the note above: the confirmation review has PASSED** (`REVIEW_REPORT_PRE_TRAINING.md`,
"M4 unseen-taxa — confirmation (2026-11-09)": gate reads *"Stage 2 (selection) may start"*), and the
live freeze is **`prereg_snapshot_v6`, not v4**. Two further revisions landed after the delta review
(v5: two cosmetic leftovers; v6: `SeededGPBoost.determinism_check` predicted with the *training*
matrix while passing the validation frame's group data). The v4 hashes quoted earlier in this file
are superseded. **Live pins, verified against disk today:**

| Item | sha256 (16) |
|---|---|
| plan | `48553f19ac3674b5` |
| `unseen_taxa.py` | `cf1493c60f694741` |
| `unseen_taxa_guard.py` | `e3721450bb97727c` |
| `test_blocks.parquet` | `649dc7d11587ea2b` (unchanged) |
| `training_pool.parquet` | `7e915fbe9a274afe` (unchanged) |

`verify_freeze()` passes on all pins. **No test block has been read or scored:**
`registry_rows()` = 0 for insects, molluscs and amphibians; no block directory, no claim, no FREEZE
file.

**Second correction: stage 2 had not in fact started.** No run directory, no registry fit row and no
selection runner existed for this track — the "608 + 12 configs" figure was the *planned* inventory,
not work in progress. Stage 2 therefore began today.

**New stage-2 runner: `scripts/eval/chemtox_unseen_select.py`** (new file; edits nothing pinned by
this freeze or by the M5 BioAssess freeze). It implements §4 + §1.4.4: the leave-one-class-out
rotation from `unseen_taxa.validation_splits`, every model fitted through `ChannelBuilder`
(`TaxonBlindBuilder` + the two channels), `channels` sampled as a feature dimension **inside** the
32-configuration budget, `chemical="mw_logp"` pinned, and the selection score as the mean RMSE over
the four folds with the worse half, the worst fold and the top-5 spread all retained. It calls
`verify_freeze()` and refuses to run if any block already holds test registry rows, and it reads
**only** `training_pool.parquet` — never `test_blocks.parquet`.

**Verified against the plan by execution before any budget was spent:**

* fold train rows **676 / 8,378 / 11,523 / 24,505** — exactly the plan's D-1 table, so parent-key
  blocking is genuinely in force;
* channel widths: surrogate **+7**, higher_rank **+9** (the "9 of 13 kept"), both **+16** — matching
  the confirmation review's `n_channel_features` 16 = 7 + 9;
* `assert_no_trait_columns` and `assert_no_class_or_lower` run for every model on every fold.

**Running now:** the two plain comparators that set the bar (`xgboost`, `rf`), 32 configurations each
over all four folds, CPU 6 threads, one entry per process under the 2 h cap
(`$SENTINEL_RUNS/chemtox/unseen_taxa/stage2_select/`). First numbers are **validation only** and are
cross-class extrapolation, so they are much worse than the ADORE CV numbers, as the design predicted
(e.g. `xgboost` cfg00: Maxillopoda 1.769, Actinopterygii 1.480).

*Two dev registry rows* (`chemtox_unseen_taxa_select`, `xgboost`, `n_configurations: 1`,
`folds: [Maxillopoda]`) come from a 1-configuration harness check and are **not** a selection
result; the real rows carry `n_configurations: 32` and all four folds.

**Blocked, needs a lead ruling before T2 can freeze:** `rf_multitaxa`, `xgboost_multitaxa`,
`rf_multitaxa_chan`, `xgboost_multitaxa_chan` have **no defined training-row rule on this track** —
see §6 of the SOTA plan at the end of this file. The runner refuses these four entries rather than
guessing. 4 of 21 mandatory comparators and claim condition (d2) depend on the answer.

### Original proposal (2026-11-09; superseded by the frozen stage 1 above)

Per RESEARCH_PLAN §15.5 items 5–7. The prospective phase is closed; its result stands as
**competitive, not SOTA**. This proposes the *next* unseen block and what to try on it. **No model
has been fitted or selected for it, and no split exists yet.**

**1. The honest data position.** A post-cut-off temporal block from the **current** ECOTOX release
(2026-09-15) is **too small to be a primary endpoint**: under ADORE's curation rules it holds 359
post-2021 rows, 229 of them fish / 58 chemicals, and 115 of those are already spent in the scored
`temporal` track. So a genuinely new temporal block **requires the next ECOTOX release**, which does
not exist locally yet (EPA publishes roughly twice a year; expect spring 2027). Honest conclusion: a
*temporal* follow-on cannot start now.

**2. What can start now: an unseen-taxa block (recommended primary).** ADORE covers fish,
crustaceans and algae only. Under the same rules ECOTOX also holds **insects 8,693 rows / 667
chemicals / 405 species, molluscs 4,805 / 470 / 298, amphibians 2,756 / 531 / 126** — taxa no
SENTINEL ChemTox model has ever been trained or scored on, and the strictest extrapolation in the
plan's sense (new taxon *and*, with the same chemical-exclusion logic, new chemicals). Power at
~470–670 chemicals per group beats the current primary track (301). It needs one data decision
first: ADORE's Add-my-Pet taxonomy block is undefined for these species, so either the block is
dropped for all models (clean, weakens every model equally) or trait values are sourced — a
licence/QC question for the lead. A chronic-endpoint block (NOEC/LOEC, 206,957 rows / 2,798
chemicals) is the larger alternative but changes the endpoint definition, so it is a separate
pre-registration, not this one.

**3. Candidate improvements, each suggested by a result above (one change at a time, tuning
parity, 32 configurations each).**
* **Graph models on unseen chemistry.** `chemprop` was the **best model on the ADORE-unseen rows**
  (1.202 vs stack 1.243, reference 1.287) while ranking last-but-four overall — the clearest signal
  in the pass that a graph model generalises where the tree models do not. First candidate.
* **Seeded `gpb.GPModel` + fixed `PYTHONHASHSEED` and thread budget**, to remove the D1
  irreproducibility and the D2 training-partition failure before GPBoost is used for any claim.
* **Pre-registered multitaxa comparators `rf_multitaxa` and `xgboost_multitaxa`** — mandatory, and
  the direct lesson of C-M4P-1: a cross-taxa claim is meaningless unless the comparators get the
  same rows.
* Deferred species embeddings (taxonomy + cytochrome-*b*) fit naturally here, since the unseen-taxa
  block is exactly what they are for; still needs the sequence-source decision.

**4. Comparators (frozen before any fit).** `rf_published` / `xgboost_published` and their
`mw_logp` variants, the re-tuned `rf` / `xgboost`, the new `*_multitaxa` pair, `readacross`, and
the three trivial baselines — all trained on identical rows.

**5. Cost under the GPU 0/1 cap.** Data build + split + power check: ~1 day, CPU, < 2 GB disk (the
ECOTOX release is already on disk; no new download). Searches: 4 families × 32 configurations,
~25 GPU-hours for chemprop and ~20 CPU-hours for the rest, all splittable into ≤ 2 h chunks. One
sealed pass: ~4 h, which again needs a one-off exception. **ETA ~8–10 working days** to a frozen,
reviewed block plus one scored pass, assuming the taxonomy decision lands early and GPU 0/1
contention stays as it is now.

**6. What I am not proposing.** No re-score of any frozen block, no second pass on the prospective
set, and no claim from the `crust_novel` result.

---

# SOTA plan — M4 ChemTox (2026-11-09, per `docs/plans/sota_campaign.md`)

**Written before any new compute beyond the in-flight unseen-taxa track. Awaiting lead approval.**
Nothing in this plan re-scores a spent block: the ADORE test and the prospective set stay closed and
their results stay labelled "competitive, not SOTA".

## 1. Diagnosis — why the stack only ties the reference

The prospective primary was stack **1.208** vs reference **1.237**, Δ **−0.029 [−0.095, 0.039]**.
On masked CV the same ensemble had looked like a real gain (paired OOF ΔRMSE vs xgboost
**−0.062 [−0.096, −0.031]**, CI excluding 0). The gain shrank by half and lost significance when it
met new chemicals. Four causes, in order of how much they explain:

1. **Every candidate shares the baseline's representation, so the stack has almost no inductive
   bias the GBM lacks.** All five frozen ensemble members (`catboost`, `chemeleon_gpboost`,
   `chemprop`, `chemeleon_lightgbm`, `gpboost_multitaxa`) and the bar (`xgboost`, 0.9733) run on the
   same published 37-column block + MACCS keys, and **every family's winner used
   `chemical=mw_logp`**. MACCS is a 166-bit fixed substructure key: on a scaffold it has no key for,
   all five members are wrong in the *same direction*. Ensembling then buys variance reduction
   (which CV rewards, because CV folds share chemical space) and almost no bias reduction (which the
   prospective block demands). That is precisely the −0.062 → −0.029 collapse.
   Per `sota_campaign.md` rule 5, the honest answer to "does the learned model have information or
   inductive bias the GBM lacks?" is currently **no** — so more capacity on these features will not
   help, and did not.
2. **The one model that *did* have a different inductive bias was starved.** `chemprop` (D-MPNN,
   learned graph representation) was the **best model on the ADORE-unseen rows** — 1.202 vs stack
   1.243 and reference 1.287 — while ranking last-but-four overall. The single clearest signal in
   the whole pass is that a graph representation generalises to new chemistry where the trees do
   not, and it got one 32-configuration budget on fingerprint-shaped features.
   `embeddings v1` (frozen chemical-LM + GBM) scored 1.1579 on CV and was dropped — but it
   *replaced* the published block instead of being concatenated with it, so that result does not
   show embeddings fail; it shows that discarding the exposure/duration covariates fails.
3. **A data ceiling we chose.** ADORE's curation filters leave **26,114 rows / 1,905 chemicals**
   (`t-F2F`). The ECOTOX release already on disk holds **471,097 aquatic rows / 6,941 chemicals**,
   of which **158,156 are LC50 + EC50** — the *same endpoint definition*, ~6× the rows and ~3.6× the
   chemicals. The only augmentation tried was `lightgbm_aug`, which did help (0.9797 vs 0.9865) and
   was never pursued at scale.
4. **The spent block could not have resolved a small win anyway.** 301 primary chemicals gave a CI
   half-width of ~0.067 around a Δ of −0.029. Even a real 0.03 effect was undetectable. Any future
   track must have its MDE computed *before* the fit (as the unseen-taxa track now does).

**Not saturated:** CV R² ≈ 0.68, RMSE 0.92–0.97 against a replicate floor of ~0.45. The headroom is
real; it is in representation and data, not in model size.

## 2. Ranked levers (expected gain per GPU-hour, highest first)

Parity rule for all of them: **every comparator gets the identical inputs and rows** — the
C-M4P-1 lesson, and the reason the `crust_novel` result carries no claim.

| # | Lever | Why it should work | Cost | Notes |
|---|---|---|---|---|
| **L1** | **More acute data, beyond ADORE's filters** | 26k → up to ~158k rows, 1.9k → 6.9k chemicals, same endpoint. Attacks cause 3 and widens scaffold coverage, which is what cause 1 needs | ~3 d CPU, no GPU | Already on disk. EnviroTox (~91k curated aquatic records, CC-BY) and EFSA OpenFoodTox aquatic endpoints add regulatory chemicals. **REACH/ECHA registered-substance ecotox is licence-gated** — dissemination data is viewable but bulk reuse is restricted, so it enters only if `docs/licenses.md` clears it, else not at all. Read-across stays a comparator *and* becomes a feature (the Channel-A pattern already built) |
| **L2** | **Pretrained molecular foundation-model embeddings, concatenated with the published block** | Gives the GBM a representation it provably lacks (cause 1), and is the cheapest fix for cause 2. MolFormer-XL / ChemBERTa-2 / Uni-Mol embedded **once** and cached, then CPU GBM + chemprop init | **~1–2 GPU-h total** (≤2 h chunks, GPU 0/1), then CPU | Highest gain per GPU-hour by a wide margin. Must be concatenated, never substituted — that was `embeddings v1`'s error. CPU-friendly at inference as required |
| **L3** | **Multitask across endpoints and species** | 271,894 NOEC/LOEC rows + 158k acute as auxiliary tasks with task embeddings; shared chemistry across endpoints is exactly the bias a per-endpoint GBM cannot express | ~10–15 GPU-h | **Not admissible on the unseen-taxa track** (`multitask=True` routes unknown taxa to the fish head, C-M4U-4). Belongs to the scaffold track |
| **L4** | **Taxonomy-aware transfer** | Species embeddings (Yang et al. 2026, taxonomy + cytochrome-*b*) on top of the Channel A/B machinery already written and reviewed | ~5 GPU-h + a sequence-source decision | Already deferred to its own pre-registration by lead decision; this is where it belongs |
| **L5** | **Ensemble *diversity* instead of ensemble size** | Select members to decorrelate representations (graph / embedding / fingerprint / descriptor) rather than 5 near-duplicates; cause 1 made the current stack redundant | ~0 extra | Do only after L1/L2 exist, else there is nothing to decorrelate |

## 3. Harder pre-registered tracks, each with an UNSEEN test block

Each is pre-registered with its MDE and power **before any fit**, a declared configuration cap, and
the §7-style single-look guard. Track choice is justified here, before any validation number on it
exists (`sota_campaign.md` rule 2).

| Track | Unseen test block | Why this track | Power | Status |
|---|---|---|---|---|
| **T1 — unseen chemical scaffolds** (*recommended new primary*) | all records whose Bemis–Murcko scaffold family never appears in training | Targets the diagnosis head-on: it is the regime where `chemprop` already beat the stack and the reference, and where MACCS-GBM is structurally weakest. A hazard model's real failure mode is a new scaffold | Thousands of chemicals available in full ECOTOX → far better powered than the spent 301 | **proposed** |
| **T2 — unseen taxa** (insects / molluscs / amphibians) | 3 frozen blocks, 15,726 rows | already frozen, delta-reviewed, gate passed | 571 / 363 / 462 chemicals; MDE 0.011 / 0.019 / 0.021 | **in flight** (§4) |
| **T5 — cross-source transfer** | records present in EnviroTox / OpenFoodTox but absent from ECOTOX | "a new source" is an explicit allowed unseen block (rule 1); tests whether gains survive a different curation pipeline, which is the strongest evidence of generalisation | to be computed at build | **proposed, second** |
| **T4 — mixtures / formulations** | formulation records, held out entirely | genuinely first-in-class territory; components → mixture toxicity | unknown until audited | **exploratory only**, after a data audit; no claim promised |
| **T3 — post-freeze publication date** | records published after the freeze | the cleanest temporal block | — | **deferred to ~spring 2027**: the current release holds only 359 post-2021 curated rows, 115 already spent. Not startable now, and not to be faked |

T2's pool is **frozen and pinned**, so L1/L2 cannot be applied to it. T2 therefore runs to
completion as the honest result it was pre-registered to produce; L1/L2 land in T1.

## 4. Single-look hardening for every new track

`unseen_taxa_guard` already provides: registry refusal, `O_EXCL` claim marker with **pid + host**,
`code_sha256` recorded on the claim, `load_test_block` as the only reader, write-once
`save_predictions`, `mark_scored` + `seal_dir` (0444 + `SHA256SUMS`), and `verify_freeze` over the
plan, both parquet files, `unseen_taxa.py` and the guard itself. **Four gaps each new track's
scorer must close** (they are scorer-side, not guard-side):

1. a **lead-written FREEZE file** as the only unlock for the test look — never a candidate-written
   file (the guard currently gates on pins alone; `downstream/testlook.py` has the FREEZE pattern to
   mirror);
2. **`BaseException` aborts** the pass without writing a claim or a prediction;
3. a **full import-closure code hash**, computed by the scorer and passed as `code_sha`;
4. **all artifacts verified before the claim**, and access closed at sealing.

Write-ups go to `*_test_results.md`. No FREEZE file has been written and no test block has been
scored.

## 5. Caps and ETA

| Item | Configuration cap | ETA (working days) |
|---|---|---|
| T2 finish (selection → freeze candidate → review) | **608 searched + 12 fixed, 31 entries** (unchanged, in the plan and registry) | **6–8**, blocked on the `multitaxa` ruling (§6) |
| L1 data expansion + QC + licence register | — | 3 |
| L2 embedding cache (GPU 0/1, ≤2 h chunks) | — | 1 |
| T1 pre-registration + data build + power/MDE | **384 searched (12 entries × 32) + fixed baselines**, declared before any fit | 3 |
| T1 selection | within the cap above | 4–5 |
| T1 freeze + pre-test review + one sealed pass | — | 2–3 |
| **Total to a SOTA attempt on T1** | | **~15–18 working days** (≈3 working weeks) |

GPU budget: **~25–35 GPU-h** total, all in ≤2 h chunks on GPUs 0/1 through `sentinel.core.gpu`.
CPU ≤6 threads per job, one CPU-heavy job at a time. Disk is at 98 % (100 GB free): the embedding
cache and expanded frames are budgeted at **≤10 GB**, with intermediates deleted.

## 6. Decision needed from the lead (blocks T2's last 4 comparator entries)

**`rf_multitaxa`, `xgboost_multitaxa`, `rf_multitaxa_chan`, `xgboost_multitaxa_chan` have no
defined training-row rule on the unseen-taxa track.** On the closed prospective track `multitaxa`
meant *extra ADORE rows from the other challenges* (`cycle.adore_other_taxa_rows`). The unseen-taxa
pool **already contains every ADORE taxon** (fish + crustaceans + algae, 60,575 rows), so the
entry's distinguishing feature does not exist here, and the plan names the four entries (§1.4.3,
inventory table) without defining them.

I have **not** guessed a definition: inventing one now would fix a mandatory comparator's training
rows after the freeze, which is the exact C-M4P-1 failure mode, and claim condition (d2) depends on
the `_chan` pair. `scripts/eval/chemtox_unseen_select.py` **refuses these four entries** until the
lead rules. Options: (a) define `multitaxa` as "all pool classes" and the plain entries as
"fish-only pool rows", making the pair a real contrast — this is my recommendation, and it needs a
dated plan amendment before any fit; (b) drop the four entries as not-applicable-on-this-track,
dropping the inventory to 480 searched + 12 fixed across 27 entries. Either way the amendment is
dated and pre-fit.

---

# Amendment A3 (draft, 2026-11-09) — awaiting the lead's reviewer

**Not yet applied to the pinned plan.** Appending it would change `plan_sha256` and so the live
freeze; the three affected entries are gated in code instead, so nothing is blocked by waiting.
On approval: append to the protocol, regenerate pins via `make_pins()`, log `prereg_snapshot_v7`,
re-verify `verify_freeze()`.

## Stage-1 amendment A3 (2026-11-09, before any affected fit) — two comparator-definition items

Neither item changes a rule, a fold, a feature, a datum or any number. Nothing covered here has
been fitted: at the time of writing only `xgboost` (and `rf`, queued behind it) have run, and
neither takes a grouping variable.

### A3.1 `multitaxa` entries are **aliases**, not new models (lead ruling, branch 1)

**Measured first, as instructed.** The plain `rf` / `xgboost` entries train on **all pool classes**,
not fish-only. Per fold, their training rows are:

| Fold (held-out class) | Train rows | Distinct classes | Taxon groups |
|---|---|---|---|
| Actinopterygii | 676 | 10 | crustaceans 653 / algae 17 / fish 6 |
| Malacostraca | 8,378 | 10 | fish 7,200 / crustaceans 1,145 / algae 33 |
| Branchiopoda | 11,523 | 10 | fish 10,042 / crustaceans 1,455 / algae 26 |
| Maxillopoda | 24,505 | 11 | fish 19,156 / crustaceans 5,299 / algae 50 |

`unseen_taxa.validation_splits` removes only the held-out class and its chemicals; no step
restricts training to fish. (The Actinopterygii fold keeps just 6 fish rows, exactly as §5.1.3
records.) The frozen pool is itself all of fish + crustaceans + algae.

On the closed prospective track `multitaxa` meant *extra ADORE rows from the other challenges*
(`cycle.adore_other_taxa_rows`). Here those rows are already in the pool, so **"multitaxa" has no
distinct meaning on this track.**

**Resolution.** `rf_multitaxa`, `xgboost_multitaxa`, `rf_multitaxa_chan` and
`xgboost_multitaxa_chan` are **aliased to their plain counterparts** — identical training rows,
identical configurations, identical numbers. No comparator's rows change. They are reported as
aliases, not as four additional independent comparators.

* **Claim condition (d2)** is evaluated on the **plain `_chan` vs non-`_chan` pair**
  (`rf_chan`, `xgboost_chan`, `rf_published_chan`, `xgboost_published_chan`,
  `gpboost_published_chan` against `rf`, `xgboost`), which is what (d2) was for: a candidate must
  not win merely because it alone received a cross-taxon channel. (d2) **remains evaluable.**
* **Configuration inventory becomes `480 searched + 12 fixed across 27 model entries`**, replacing
  608 + 12 across 31. The four aliases are counted once, in their plain entries. **Mandatory
  comparators: 17** (2 re-tuned ADORE trees + 5 channel-matched + 5 published fixed + `readacross`
  + 4 surrogate/ICE), replacing 21.
* Every claim reports the corrected counts; the registry snapshot carries them so they cannot drift.

### A3.2 The grouping variable for `gpboost*` and `readacross` (a defect I introduced, now fixed)

**What happened.** `gpboost_model.GROUP_COLUMNS_DEFAULT` is `("tax_gs", "tax_genus", "tax_group")`
and `readacross` hard-codes `SPECIES_COL = "tax_gs"`, but the frozen pool has none of them, so both
raised `KeyError: 'tax_gs'`. My first adapter fixed that by **deriving `tax_gs` from `latin_name`
and `tax_genus` from `genus`**. That was wrong and is removed.

The freeze-time caveat recorded with `prereg_snapshot_v6` says: *"the trait block is dropped on this
track, so the pool has no `tax_gs` and the only grouping variable Channel B admits is `rank_phylum`
(3 levels); the published method groups species/genus/group, so `gpboost_published` here is an
ADAPTED estimator and must be labelled as such in every comparison."* Real species and genus keys
would reintroduce taxonomy at **species and genus rank** — below class rank, which §1.1 and §1.4.2
forbid — and would let a random effect absorb species-level structure **during training**, changing
the estimator, even though an unseen test species collapses to the population mean at predict time.
`assert_no_class_or_lower` would not have caught it, because it inspects feature names and these are
grouping keys.

**Resolution, as pre-registered in the v6 caveat.** The adapter now supplies only ranks **strictly
above** the held-out class (`rank_phylum` / `rank_subphylum` / `rank_clade`, from
`unseen_taxa.higher_ranks`, a pure function of the class name), plus the pool-level `tax_group`
(fish / crustaceans / algae — a frozen pool column, above class rank for any held-out class, and
what §5.1.5 means by read-across's "group" widening level). `tax_gs` and `tax_genus` are **not
created**, so anything needing them fails loudly instead of silently regaining taxonomy.

* `gpboost` and `gpboost_published_chan` are fitted with `group_columns=("rank_phylum",)`, verified
  to be 3 levels on the pool. **`gpboost_published` is therefore an ADAPTED estimator** — the
  published method groups species/genus/group — and is labelled as such in **every** comparison and
  in the model card. Claim condition (b) is read against this adapted estimator, and the adaptation
  is disclosed beside the result.
* **`readacross`:** with no species column, every row falls to the `"group"`/`"global"` neighbour
  pool, i.e. the plain chemical-similarity kNN over fish/crustacean rows that §5.1.5 already
  describes as this track's behaviour. The `widening_` counters are reported as §5.1.5 requires.
* Both readings change what a **mandatory comparator is**, so `scripts/eval/chemtox_unseen_select.py`
  **refuses `gpboost`, `gpboost_published_chan` and `readacross`** until this amendment is reviewed
  (`--allow-grouping-dependent` unlocks them afterwards). The nine grouping-independent entries
  continue meanwhile.

### Re-freeze

Plan amended, pins regenerated, registry `prereg_snapshot_v7` logged with the corrected inventory
(480 / 12 / 27) and both items above. Data **unchanged**
(`649dc7d1…` / `7e915fbe…`). No test block read; no claim marker; no FREEZE file.

## Lead rulings received 2026-11-09 — disposition

| Ruling | Disposition |
|---|---|
| multitaxa: check plain entries' rows first | **Done, measured** — all pool classes (10-11 per fold, all three groups). Branch 1 applies: A3.1 aliases the four entries; (d2) stays evaluable on the plain `_chan` vs non-`_chan` pair |
| amendment reviewed before fitting | A3 drafted above; the four aliases need no fit, and `gpboost`/`gpboost_published_chan`/`readacross` are **refused by the runner** until review |
| SOTA plan approved; L2 first, then L1 | Accepted. **Not started** — T1's scaffold split must be pre-registered and frozen before any embedding run touches it, per the ruling |
| T1 = new primary track; pre-register + freeze split before any fit | Next deliverable after A3 clears |
| REACH/ECHA only if `docs/licenses.md` clears | Noted; untouched |
| L5 fine within caps; L3 stays off T2 | Noted; `chemprop` (GPU 0/1) and ensemble diversity are T1-side |
| full hardened scorer on every new track | The four scorer-side gaps (lead-written FREEZE only, BaseException abort, import-closure hash, all-artifacts-verified) are listed in the SOTA plan §4 and will be built into T1's scorer |
| disk: check `df`, stop below 60 GB, M4 cap 10 GB | **108 GB free at 17:50**; the phase-2 queue now checks `df` before each entry and breaks below the 60 GB floor. No download or build started |

---

## A3 review disposition (2026-11-09) — 5 blockers + 9 text/warning items closed in code

Review: `REVIEW_REPORT_PRE_TRAINING.md`, "M4 unseen-taxa — amendment A3 review (2026-11-09)".
A3.1 approve with changes; A3.2 approve with changes for `gpboost*`, **reject as drafted** for
`readacross`; re-freeze procedure approved. **61/61 M4 tests green** (29 prospective + 17
unseen-taxa + 15 new `tests/test_chemtox_unseen_select.py`). No pinned file was edited.

| Item | Fix, verified by execution |
|---|---|
| **C-A3-1** `readacross` crashed rather than degenerating (`readacross.py:105` reads `rows["tax_gs"]` unconditionally) | New track-local `TrackReadAcross` wrapper (shared `readacross.py` **not** edited): constant placeholder species key + `match_species` pinned off. Measured on the Actinopterygii fold: `widening_ = {species: 0, group: 0, global: 200, fallback_mean: 75}` — §5.1.5's pre-registered global kNN. Asserted per fold at run time **and** in a test |
| **C-A3-2** `tax_group` would revive the "group" widening level inside selection folds | `match_species=False` pinned for the track, so selection and test both run the plain kNN. Test proves a config that sampled `match_species=True` still runs with it off |
| **C-A3-3** recorded `gpboost` config misreported the grouping (all 4 sampled values were forbidden keys) | `group_columns` **removed from the sampled space** (`GPBOOST_SPACE_TRACK`), so the audit trail can only ever show `rank_phylum`; post-fit assertion `tuple(model.inner_.group_columns_) == ("rank_phylum",)`; `<entry>.json` and the registry now record **effective** params, never the sampled dict |
| **C-A3-4** `rf_chan`≡`rf_published_chan`, `xgboost_chan`≡`xgboost_published_chan`, candidate `gpboost`≡comparator `gpboost_published_chan` (32/32) | Settled §1.4.3 vs §1.4.4 in favour of §1.4.3: the `*_published_chan` entries use the **published fixed** config with channels as the only varied dimension → **3 configs each**; `rf_chan`/`xgboost_chan` keep 32 with **per-entry seeds**. Verified: **no comparator shares a draw** with any candidate or comparator |
| **C-A3-5** 11 of 32 `_chan` draws were channel-free, making (d2) vacuous | `_chan` entries sample the **3 non-empty** settings only (`CHANNEL_CHOICES_MATCHED`); candidates keep all 4. Verified 0 empty draws for every `_chan` entry, 11 retained for `lightgbm` |
| **T1** (d2) misstated | Restated: (d2) is **candidate vs each channel-matched comparator**; the `_chan` vs non-`_chan` difference is a diagnostic reported beside it |
| **T2** `tax_group` called "above class rank" | Corrected: it is a one-to-one proxy for `Insecta`/`Amphibia`; admissible only because `TaxonBlindBuilder` deletes it from every matrix, it is never a grouping column, and its test values match no training level |
| **T3** aliases invisible in output | `ALIASES` added; printed as `= rf` / `= xgboost` / `= rf_chan` / `= xgboost_chan` in comparator tables |
| **T4** "never create `tax_gs`" was only a resolved defect | Now a protocol invariant with `assert_no_forbidden_columns` + 2 tests, covering `tax_gs`/`tax_genus`/`tax_family`/`tax_species` (chemprop's `ConditionEncoder` keys a **species embedding** on `tax_gs`) |
| **T5** adaptation disclosed asymmetrically | The `gpboost` **candidate** is adapted identically (`rank_phylum` only), so it gains no advantage; ADAPTED label applies to all three |
| **W-A3-1** `assert_taxon_agnostic` never called | Called in `build_model` for every entry; `multitask=False` forced in every chemprop draw |
| **W-A3-2** smoke row in the append-only registry | `--log-registry` now **refuses** unless `len(folds)==4 and budget==32`. The two dev rows are superseded once the real `xgboost` run lands |
| **W-A3-3** `chemprop` unsamplable (`KeyError` — its space is in `spaces.py`) | Wired to `spaces.sample("chemprop")`; 32 draws, all single-task |
| **W-A3-4** unused `evaluate as E` import | Removed |
| Count reconciliation | One set of numbers: **390 runnable searched (393 with A3.5 unlocked) + 12 fixed across 27 entries**, 17 mandatory comparators, asserted in a test |

### New blocker found while fixing C-A3-4: `gpboost_published_chan` is BLOCKED

§5.1.3 names `gpboost_published` as a **fixed published configuration** (Zubrod et al. 2024) but
**never states one**, and no `PUBLISHED_FALLBACK` entry exists for it anywhere in the repo. Choosing
hyper-parameters now would define the published cross-species comparator — the reference for claim
condition **(b)** — after the freeze. The runner refuses the entry. **Consequence, stated plainly:
until the lead declares the configuration, claim condition (b) is not evaluable and no SOTA claim on
any block may rest on it.** Four distinct channel-matched comparators remain for (d2).

### Still to do (in order)

1. **Re-freeze, pending the in-flight `xgboost`/`rf` runs** — review addition (iii): re-freeze only
   when nothing is running, so no run spans two plan hashes. Then: copy pins to
   `chemtox_unseen_taxa_pins.v6.json` (addition i), append the A3 section (drafted, with the v6
   caveat moved into the plan text), `make_pins()`, log `prereg_snapshot_v7` stating that `xgboost`
   and `rf` were fitted under v6 and need no re-run (addition ii).
2. **Send the diff to the lead** for the short delta check — **before** unlocking anything.
3. Only then may `readacross` and `gpboost` be unlocked (`--allow-grouping-dependent`).
   `gpboost_published_chan` stays blocked regardless, pending its configuration.
4. Resume the 9 grouping-independent entries (`lightgbm`, `catboost`, `chemeleon_lightgbm`,
   `lightgbm_resid_ice`, `lightgbm_resid_surrogate`, `rf_chan`, `xgboost_chan`,
   `rf_published_chan`, `xgboost_published_chan`); `chemprop` separately on GPU 0/1.

### Sixth blocker resolved — published GPBoost config: **branch (1) applies** (lead ruling 2026-11-09)

Evidence gathered **before any validation result on this entry existed** (the entry has never been
fitted for selection):

| Source checked | Outcome |
|---|---|
| On-disk repo | citation only (`gpboost_model.py:11-13`); no hyper-parameters |
| Paper full text (*Environ. Int.* **186**:108607, DOI 10.1016/j.envint.2024.108607, PMID 38593686) | **not accessible** — publisher HTTP 403; Europe PMC records **not open access**, no PMCID |
| Released README (`github.com/syngenta/bio-qsar`) | package versions only (R 4.1.3, gpboost **1.2.1**); defers to the paper |
| **Released fitted models** (same repo) | **state the complete set verbatim** in their `parameters:` echo block → **branch (1)** |

Verbatim from the authors' **trait-free** models (`_without_DEB` — the matching form for this track,
which drops the trait block): `objective` regression, `num_iterations` **1000** (highest `Tree=999`),
`learning_rate` **0.01**, `max_depth` −1, `min_data_in_leaf` **10**, `feature_fraction` /
`bagging_fraction` 1 / 1; `num_leaves` **64** (fish) vs **32** (invertebrate); `lambda_l2` **1**
(fish) vs **10** (invertebrate). Provenance: `Fish_without_DEB.json` sha256 `1b17db4c5f18abbd…`
(7,171,431 B), `Invertebrate_without_DEB.json` sha256 `770ecd0d1ed31360…` (3,369,787 B), retrieved
2026-11-09.

Both forms are kept (the authors released taxon-specific models and this track predicts both
invertebrate and vertebrate blocks); per §5.1.1's existing rule the reference is the best-validation
variant, fixed in the hashed block before the pass. So `gpboost_published_chan` =
**2 forms × 3 channel settings = 6 fixed configurations, no search** — **no hyper-parameter chosen
by us**. Verified: both forms fit, grouping `('rank_phylum',)`, finite predictions.

**Claim condition (b) is evaluable again**, against an estimator labelled *"published configuration,
verbatim from the authors' released trait-free models; ADAPTED grouping (`rank_phylum` only);
gpboost 1.8.0 vs the authors' 1.2.1."* Both adaptations are disclosed with every (b) comparison, and
the `gpboost` **candidate** carries the identical grouping adaptation, so it gains no advantage.

**Recorded because it would have biased in our favour:** under branch (2) the gpboost 1.8.0 defaults
(`num_iterations` 100, `learning_rate` 0.1, `num_leaves` 31, `min_data_in_leaf` 20, `lambda_l2` 0,
fractions 1 — read from a fitted booster's own echo) combined with the authors' 0.01 learning rate
would have given a **severely undertrained** published comparator (0.01 × 100 vs the authors'
0.01 × 1,000). A weak published comparator inflates our candidate. Branch (1) removes the issue.

Inventory becomes **396 searched + 12 fixed across 27 entries**, 17 mandatory comparators, **5
distinct channel-matched comparators** (all runnable). **17/17** tests in
`tests/test_chemtox_unseen_select.py`, including verbatim-value assertions.

### Re-freeze DONE — `prereg_snapshot_v7` (2026-11-09 19:01). Awaiting the delta check

Performed in the approved order, with **nothing running** (review addition iii: `xgboost` finished
18:22:07, `rf` 18:59:52, both rc=0, so no run spans two plan hashes).

| Step | Result |
|---|---|
| (i) v6 pins preserved | `docs/preregistration/chemtox_unseen_taxa_pins.v6.json` sha256 `84e3f84c327efdf0…` |
| A3 appended to the plan | **append-only**: 1 hunk, **0 lines removed**, 214 added. Plan sha256 `48553f19ac3674b5…` → **`40fdb79621f0a2dd…`** |
| `make_pins()` | plan `40fdb796…`, module `cf1493c6…` (unchanged), guard `e3721450…` (unchanged), data **unchanged** (`649dc7d1…` / `7e915fbe…`). Pins re-sealed 0444 |
| `verify_freeze()` | **PASS** |
| (ii) `prereg_snapshot_v7` | 4 rows (`fa9b4eb24624`, `78441fdece49`, `05e99382a137`, `543b863d7ca3`), `supersedes: prereg_snapshot_v6`, carrying 396 / 12 / 27 / 17 and the published-GPBoost provenance. Records **`fitted_under_v6: [xgboost, rf]`, `need_rerun_after_a3: []`** |
| W-A3-2 smoke rows superseded | `9f275e63e4a7` → `6c49b7b94c2f`, `bc90a6ee00c8` → `6a23615fb292`, now the full 32×4 numbers |
| Tests | **34/34** green (17 unseen-taxa + 17 new select) |
| Test blocks | still unspent: `registry_rows()` 0/0/0, no claim marker, no FREEZE file |

**Note for the reviewer:** the plan is append-only, so §1.4.4's original "608 + 12 across 31" table
is still present and A3.6 supersedes it. A3.6 is the governing inventory (**396 searched + 12 fixed
across 27 entries**, 17 mandatory comparators, 5 distinct channel-matched).

### First selection results — the comparator bar (validation only, 32 configs × 4 folds, 0 failures)

These are cross-class extrapolation inside the training pool. No test block was read.

| Entry | best mean RMSE | worst fold | worse half | top-5 spread | channels |
|---|---|---|---|---|---|
| `rf` | **1.5629** | 1.7397 | 1.6809 | 0.0234 | none (pinned) |
| `xgboost` | 1.5918 | 1.7318 | 1.7001 | 0.0246 | none (pinned) |

Per fold — `rf` 1.4536 / 1.6221 / 1.7397 / 1.4362 and `xgboost` 1.5856 / 1.6684 / 1.7318 / 1.3815
(Actinopterygii / Malacostraca / Branchiopoda / Maxillopoda). Both far worse than the ADORE CV
numbers (0.97–0.99) and worse than the closed prospective track (~1.2), which is what §9 predicted
for a class-level hold-out: **`rf` at 1.5629 is the bar the candidates must beat.** Branchiopoda is
the hardest fold for both; the 676-row Actinopterygii fold is *not* the hardest, which is worth
noting against §4's expectation that it would be.

### Next (unchanged order)

1. **Delta review of the A3 diff** (`$SCRATCH/a3_plan.diff`) — before anything is unlocked.
2. Then `--allow-grouping-dependent` may unlock `readacross`, `gpboost` and
   `gpboost_published_chan`.
3. Resume the 9 grouping-independent entries, then `chemprop` on GPU 0/1.

---

## A3 delta check PASSED → unlocked; A4 documentation re-pin (`prereg_snapshot_v8`, 2026-11-09)

Review: `REVIEW_REPORT_PRE_TRAINING.md`, "M4 unseen-taxa — A3 delta check (2026-11-09)". All five
blockers, five text items and four warnings confirmed fixed **by the reviewer's own execution** (34
tests re-run, every count re-derived, both model-file sha256 re-downloaded and matched, and both v6
runs re-drawn under the post-A3 runner: **32/32 identical configurations**, so `need_rerun_after_a3:
[]` is correct). Both gate conditions and D1–D4 are now closed.

### A4 — documentation only, append-only (plan `40fdb796…` → **`a3451d5e82283877…`**)

`verify_freeze()` PASSES. Data, module and guard **unchanged**. `prereg_snapshot_v8` logged
(3 rows, supersedes v7); v7's counts and published values still govern. **No rule, fold, feature,
configuration or datum changed.**

| Item | Closed by |
|---|---|
| **A3.5-1** (gate condition 1) — "trait-free" overstates the match | **A4.2.** Both released files still carry **species one-hot fixed effects (10 fish / 13 invertebrate)**, which §1.1 forbids here. So the published comparator loses **three** things, not two: grouping → `rank_phylum`, library 1.8.0 vs 1.2.1, **and the authors' species indicators**. **Each makes it weaker here, i.e. each biases in our candidate's favour.** Wording corrected to **"DEB-trait-free (the authors' species indicators are also absent here)"**. In code as `PUBLISHED_GPBOOST_ADAPTATIONS`, written into `summary_index.json`, and in the model card |
| 6-vs-3 selection breadth | **A4.2** + `PUBLISHED_SELECTION_BREADTH`: published GPBoost is the best of **6** (2 forms × 3 channels), `rf_published_chan`/`xgboost_published_chan` the best of **3**. Conservative direction (a wider choice makes the published bar harder to beat) but now stated |
| **D1** stale §1.4.4 inventory | **A4.1** marks the table *and* its false "the count cannot drift" sentence superseded, with a side-by-side stale/current table. Appended, not edited in place, so earlier hashes stay reproducible |
| **D3** aliases promised but never emitted | **A4.3** — `alias_rows()` emits one row per alias into `summary_index.json`, each `fitted: false`, rendering `rf_multitaxa = rf` etc. |
| **D4** wrong article title | **A4.4** — corrected in the model card and recorded in the plan. **`gpboost_model.py:11` left untouched** (covered by an existing freeze, per the lead); the defect is recorded so the wrong title cannot reach the paper |
| **A3.5-3** provenance durability | Manifest **`chemtox_published_gpboost`** under `$SENTINEL_DATA/manifests/` — URLs, sha256, byte counts, retrieval date, the nine values, and the species-indicator counts. Only the hyper-parameters are used; no weights, features or predictions enter any SENTINEL model |
| **A3.5-2** unevidenced execution claims | **A4.5** — they were single **unlogged** one-fold smoke fits, never used for selection, writing no registry row. Logs now under `stage2_select/smoke/` |
| **D2** duplicate live registry pair | Closed, but **my first attempt made it worse** — see below |

**D2, with the mistake disclosed.** My first fix appended duplicate `rmse`/`rmse_worst_fold` rows
superseding the originals, which under append-only semantics (`current_only` drops any id named in a
`supersedes`) left **two** live pairs rather than one: 4 live − 2 superseded + 2 new = 4. The working
fix is a **retirement marker** carrying its own metric name (`d2_duplicate_retired`), which retires a
row without adding another live `rmse` row. Verified: `registry.lookup(...)` now returns exactly one
row per metric — `6c49b7b94c2f` (rmse 1.5918) and `6a23615fb292` (rmse_worst_fold 1.7318). No number
changed.

### Gate condition 2 — `gpboost_published_chan` one-fold smoke (unlogged, NOT for selection)

Those published values had never executed in the runner (the v6 determinism fit used different
ones). Ran on the Malacostraca fold, `--smoke` (no registry row, no `TrainingRun`); log kept at
`stage2_select/smoke/gpboost_published_chan_smoke.log`. **All 6 configurations fit, 0 failures**,
grouping `('rank_phylum',)`. Confirmed afterwards: **0** `gpboost*` and **0** `readacross` selection
rows in the registry.

Per-config RMSE on that one fold — fish_form 2.3307 / **1.6295** / 2.3013 and invertebrate_form
2.3174 / **1.6242** / 2.2974 for channels surrogate / higher_rank / both. **Observation for
selection, not a result:** the **surrogate channel appears to hurt GPBoost badly here** (~2.30 vs
~1.62 with `higher_rank` alone), which is the opposite of §9's expectation that surrogate features
should dominate. One fold, smoke only — but worth watching when the real four-fold numbers land.
For scale, `rf` scored 1.6221 on that same fold against the published GPBoost's best 1.6242.

### Resumed (running now)

Queue `m4_resume.sh`, one entry per process under the 2 h cap, 6 threads, `df` checked before each
entry with a hard stop below 60 GB free (105 GB now): **9 grouping-independent entries, then the 3
unlocked** (`readacross`, `gpboost_published_chan`, `gpboost`). `chemprop` follows separately on
GPU 0/1 via `sentinel.core.gpu`. Flag `RESUME_DONE.flag`.

**Bar to beat (validation, 4-fold mean): `rf` 1.5629**, `xgboost` 1.5918.

---

## 🔴 GENUINE BUG, reported to the lead 2026-11-09 — Channel A is 0% available in selection

**Correctness check only, as instructed. No fix applied, no configuration changed.** Queue **paused**
by PID (11 of the 12 queued entries use channels, and the lead's standing instruction is that every
affected entry re-runs, so continuing would burn shared-host CPU on numbers that must be redone).
Only `lightgbm` was in flight; it wrote no registry row. Nothing is running.

### The defect

| Channel A (surrogate) availability | Value |
|---|---|
| Validation fold Actinopterygii | **0 / 42,533 (0.0 %)** |
| Validation fold Malacostraca | **0 / 9,386 (0.0 %)** |
| Validation fold Branchiopoda | **0 / 7,085 (0.0 %)** |
| Validation fold Maxillopoda | **0 / 1,191 (0.0 %)** |
| **Test blocks** (plan D-3) | **94.6 % / 93.1 % / 87.6 %** |
| Counterfactual: class-only hold-out, no chemical blocking | 87.8 / 98.4 / 95.0 / 98.5 % |

**Mechanism — two pre-registered rules interacting, not a typo.** §4/W-M4U-3 removes the held-out
class's chemicals from the fold's training side by CAS, InChIKey **and** `parent_key`; §1.4.4
re-fits the surrogate block on that fold's training rows only. Together **no validation chemical can
have any pool record**, so the surrogate is never available. §1.4.4's stated rationale — "a fold's
surrogate coverage is what test-time coverage will be" — is therefore **false as implemented**
(0 % vs ~90 %).

**On validation rows all 7 surrogate features are constant** (pool mean −4.452, ICE of the pool mean
−4.632, flag 0, counts 0): correlation with `y` is exactly **0.000** on validation against
**0.92–0.96** on training. Compounding it, **self-inclusion target leakage** on the training side:
for all **736/736** training rows whose chemical has exactly one pool record,
`surr_pool_geomean` **equals that row's own target** to machine precision.

### Not GPBoost-specific — it hits candidates identically

Unlogged `rf` diagnostic, fold Malacostraca, identical parameters, only the channel varying
(`stage2_select/smoke/surrogate_channel_diagnostic.log`):

| channels | TRAIN RMSE | VAL RMSE |
|---|---|---|
| `()` | 0.3362 | 1.6433 |
| `('surrogate',)` | 0.3023 | **2.3210** |
| `('higher_rank',)` | 0.3192 | **1.5902** |
| `('surrogate','higher_rank')` | 0.2805 | 2.2827 |

Training error *improves* with the surrogate while validation degrades by **+0.68** — the signature
of a train-only feature. Worst affected: `lightgbm_resid_ice` and `lightgbm_resid_surrogate`, built
directly on a base that is constant in validation, so in CV they are plain `lightgbm` on a shifted
target while behaving completely differently at test time.

### The lead's other two hypotheses are ruled out

* **NaN handling:** **zero NaN** in all 7 columns on both train and validation, so GPBoost-vs-tree
  NaN semantics cannot be the cause.
* **Units/scale:** the four value features are log10(mol/L) on the target's scale (train mean
  −4.452 exactly equals the target mean −4.452); `surr_n_records` / `surr_n_pool_taxa` are counts
  (0–114 / 0–22), a legitimate different scale.
* **Held-out-class leakage:** held-out-class rows in train = **0** for all four folds,
  `assert_no_test_classes` passes in `fit`, and the zero availability itself proves no held-out
  chemical contributed.

### This is a selection-validity defect, NOT a test leak

The frozen blocks are untouched — `registry_rows()` **0/0/0**, no `AUTHORISED.json`, no
`SCORED.json`, no FREEZE file. The hold-out's integrity is intact. What is compromised is the
ability to **select** a Channel A configuration honestly: selection rejects every surrogate
configuration for an artefact, and §9's expectation that surrogate features should dominate can
never be tested, although at test time the surrogate is available for ~90 % of rows.

### Scope if a fix is authorised

Re-run every channel-using entry: all 7 candidates + all 5 channel-matched comparators. **`rf`,
`xgboost`** (complete, channels pinned off) and **`readacross`** (channels=none) are unaffected and
need no re-run. **No fix has been designed** — the options differ in what they cost the protocol and
choosing between them is a pre-registration decision, so it is left to the lead and the reviewer.

Live selection rows at pause: `xgboost` (full 32×4) and `rf` (full 32×4) only.

---

## A5 drafted + re-pinned `prereg_snapshot_v9` (2026-11-09). Diff sent; **no re-run started**

Lead's fix direction implemented. Diff for the reviewer:
`$SENTINEL_RUNS/chemtox/unseen_taxa/a5_plan.diff` (154 lines, **1 hunk, 0 removed** — append-only).
Plan `a3451d5e…` → **`9a6fab9d9fe41520…`**; `verify_freeze()` PASSES; **pinned `unseen_taxa.py`
and `unseen_taxa_guard.py` unchanged**, frozen data unchanged; v8 pins preserved as
`chemtox_unseen_taxa_pins.v8.json`. **23/23 tests green.** Nothing re-run yet.

### Step 1 — W-M4U-3's rationale, quoted, and the verdict

§4: *"The held-out class's chemicals are removed from that fold's training rows by CAS, InChIKey
**and** `parent_key`, so a salt, counter-ion or stereoisomer cannot make a selection fold **easier
than the test blocks it mirrors**."*

§1.2, on how the **test** block is built: *"A test chemical **may legitimately appear in the training
pool**, measured on fish. That is realistic and **is not leakage** of the test rows. So chemical
novelty is **reported** rather than enforced."* The `chem_novel` strata confirm it quantitatively —
only 477 of 7,165 insect primary rows are `chem_novel`, i.e. **93 % are not**.

**Verdict: it guards against something test construction explicitly allows, so it goes** (the
lead's first branch). No need to stop.

### Step 1 result — new folds and availability

| Fold | Train rows (was) | Val rows | Surrogate available (was) |
|---|---|---|---|
| Actinopterygii | 18,042 (676) | 42,533 | **88.0 %** (0.0 %) |
| Malacostraca | 51,189 (8,378) | 9,386 | **98.4 %** (0.0 %) |
| Branchiopoda | 53,490 (11,523) | 7,085 | **95.0 %** (0.0 %) |
| Maxillopoda | 59,384 (24,505) | 1,191 | **98.5 %** (0.0 %) |

Test blocks: 94.6 / 93.1 / 87.6 % — now comparable, which was the point. Held-out-class rows in
train: **0** for all four.

### Step 2 — Channel A computed out-of-class

`OutOfClassSurrogateFeatures`, `TrackChannelBuilder`, `ResidualOutOfClassModel` (all **new** code).
A training row of class *C* draws only on pool records of **other classes**; validation and test
rows, whose class is absent from the fitting frame, get the full fit — the same thing test sees.

**Verified by perturbation, immune to coincidental equality:** shifting **every** training target by
+100 leaves all 7 features **bit-identical**. Self-inclusion falls **825/825 → 6/825** on the
Malacostraca fold, and those 6 are data coincidences (an equal value in another class), which is
precisely why the test asserts the mechanism rather than an equality count. Correlation with the
target: **0.677 train / 0.776 validation** — validation now *higher*, so the over-reliance is gone.

**The pathology is reversed** (unlogged `rf` diagnostic, identical params, new folds):

| channels | VAL RMSE before A5 | VAL RMSE after A5 |
|---|---|---|
| `()` | 1.6433 | 1.3412 |
| `('surrogate',)` | **2.3210** | **1.2286** |
| `('higher_rank',)` | 1.5902 | 1.2502 |
| `('surrogate','higher_rank')` | 2.2827 | **1.2183** |

The surrogate channel now **helps** — §9's recorded expectation, finally testable.

### Step 3 — every entry re-runs; the bar is void

Fold definitions changed for all entries, so **`rf` and `xgboost` re-run too and the bar is
recomputed**. **There is now no live `rmse` row for this experiment**: all four old-fold value rows
are retired with A5 as the reason (`rf` 1.5629/1.7397 via `c9109a215a46`/`ff0c6e5ba482`; `xgboost`
1.5918/1.7318 via `6c49b7b94c2f`/`6a23615fb292`). Old and new numbers are never mixed or compared —
absolute RMSE will improve for a reason unrelated to method quality, since training sets grow 2–7×
once chemical blocking is dropped (Malacostraca 8,378 → 51,189).

**Second registry-handling error of mine, disclosed:** my first A5 supersede used two **guessed**
`rf` row ids (`3d50c03d8b64`, `b9a2fbc2b511`) that do not exist, so it retired nothing for `rf`. I
verified the real ids and superseded them properly; the correction is recorded in the replacement
rows' notes. Both the D2 mistake and this one were the same root cause — acting on an assumed
registry id instead of reading it first.

### Still true

No test block read; `registry_rows()` **0/0/0**; no claim marker, no `SCORED.json`, no FREEZE file.
Discovery order is on record: the defect came from the gate-condition-2 smoke plus an unlogged
diagnostic, **before any logged channel result existed**.

---

## A5 delta check passed → pre-run steps done → A6 (`prereg_snapshot_v10`) → RE-RUN LAUNCHED (2026-11-09)

Review: "M4 unseen-taxa — A5 delta check (2026-11-09)". A5.1 and the A5.2 code **approved**. Plan
`9a6fab9d…` → **`b33bac5db628fa02…`** (A6, append-only, 1 hunk, 0 removed); `verify_freeze()`
PASSES; pinned module/guard and frozen data unchanged; v9 pins preserved. **32/32** select tests +
17 unseen-taxa tests green.

### 1. Archive first (destructive ordering — done BEFORE any re-run)
Moved, **byte-identical (sha256 verified before and after)**, into sealed 0444 dirs:
`stage2_select/old_folds_a5/` (xgboost/rf `.json`+`.log`, summary_index, queue logs, flag) and
`stage2_select/smoke/aborted_old_folds/` (the aborted `lightgbm.log` + its queue log), each with
`SHA256SUMS`. The output dir held **no** old-fold artefact at launch. Each retired row got an
appended `artefact_relocated` pointer to its archived copy.

### 2. My A5 perturbation test was VACUOUS — replaced, with proven power
It shifted `result_conc1_mean_mol_log`; the surrogate reads `unseen_taxa.TARGET` =
**`log10_conc_mol_L`**. It perturbed an unread column, **could not fail**, and would have passed the
defective code. **I reported it to the lead as proof; it proved nothing.** Replacement perturbs one
row in **both** columns and is two-sided: **pinned `SurrogateFeatures` → FAIL** (own row changes =
self-inclusion detected); **`OutOfClassSurrogateFeatures` → PASS** (own unchanged, witness changed).
A permanent test requires it to keep failing the pinned path, so it cannot silently lose power.

### 3. Figures with frames — I could NOT reproduce two of the three I was asked to write in
| Frame | Me (single-record = CAS once in frame) | Reviewer |
|---|---|---|
| old folds, old code | **736/736** | **736/736** — agreed |
| old folds, new code | 6/736 | 15/934 |
| new folds, new code | 6/825 | 21/2,549 |

I tried two definitions and reproduced neither 15/934 nor 21/2,549, so they are **attributed to the
reviewer, not restated as measured**. "825/825" was never measured — my error. No claim rests on
these counts; the argument is the mechanism, which §2 now verifies.

### 4. Train/serve availability asymmetry — disclosed
Train (out-of-class) **88.1 / 82.2 / 87.2 / 90.3 %** vs validation **88.0 / 98.4 / 95.0 / 98.5 %**
(reviewer: Branchiopoda 87.1 — within 0.1 pt). Conservative direction. Per-fold availability is now
recorded beside every fold score and in each registry row.

### 5. Registry — TWELVE dangling rows, not two
New read-only `scripts/eval/registry_audit.py`, over the whole registry, found **12**, all M4's:
2 guessed ids + **10 of my own v7/v8/v9 snapshot rows that put a version NAME into `supersedes`**.
Third instance of one error class. Guessed-id rows **tombstoned**; snapshot rows **acknowledged, not
tombstoned** (content correct; retiring them would hide v7–v9 while v1–v6 stay visible). Audit now
**OK**; it fails on any *unacknowledged* dangling row. New `registry_checked.log_result_checked`
refuses any `supersedes` naming no existing row (verified on a guessed id and a version name). Core
`registry.py` deliberately **not** edited — shared by all seven workstreams, so a core guard is the
lead's call. v10 restores the convention: no `supersedes`, `extra.succeeds`.
**A5.3 attribution fixed:** rf/xgboost fitted under **v6 `48553f19…`**; the xgboost re-stamp rows
carry **v7 `40fdb796…`** (the review said v8 — the rows say v7).

### Runner fixes found while preparing the re-run
* **2 h cap:** one config takes ~155 s across the new folds (2–7× more training rows), so a 32-config
  entry averages ~83 min and a heavy one can breach the cap and die with no output. Added
  `--shard/--shards/--merge`: the **same** sampled list split across processes, global indices kept.
  The merge **refuses** missing, duplicated or drifted configurations and partial-fold shards, and
  alone writes the registry. 7 new tests, all on synthetic shards (no fitting).
* **Registry gate moved up front** — it used to fire only *after* all compute.
* **`summary_index.json` was overwritten** per invocation, so one-entry-per-process runs kept only
  the last entry; it now accumulates.
* Registry rows now carry per-fold availability and, for `gpboost_published_chan`, the three
  disclosed adaptations and the 6-vs-3 selection breadth.
* `chemprop` wired to `--accelerator gpu` (recorded params unchanged, so the merge stays clean).

### GPU safety, checked before any GPU use
`sentinel.core.gpu` defaults to a cap of **[0,1,2,3,4]** if `.gpu_locks/ALLOWED` is absent, and
`--prefer` falls back to the next allowed GPU. The lead's cap file exists and reads **`0,1`**, so
the helper can only lock GPU 0 or 1; `SENTINEL_GPUS=0,1` is also set. Peter's GPUs 2–9 untouched.

### Running now
`m4_a5_rerun.sh` (pid at launch 1667029): 14 CPU entries, `rf` and `xgboost` first, each as ≥2
shards → integrity-checked merge → registry; one CPU job at a time, 6 threads, `df` checked before
every shard (hard stop < 60 GB; 96 GB at launch). Flag `A5_RERUN_DONE.flag`. `chemprop` follows on
GPU 0/1 once its per-config time is measured. Test blocks unspent: 0/0/0, no claim, no FREEZE.

---

## Lead rulings applied → A7 (`prereg_snapshot_v11`, 2026-11-09) → re-run RESTARTED under v11

| Ruling | Done |
|---|---|
| **1. chemprop, option (a)** | `TrackChemprop` (new code; `chemprop_model.py` **unedited**) puts a **constant** placeholder species key on copies of the frames handed to chemprop, never on the pool. `species_dim` stays sampled over [0, 8, 16, 32] (sampled space unchanged). **3 new tests:** `n_species == 2` and train/validation share **one** index; the pool never gets the placeholder; the entry is built through the wrapper with its space intact. **Disclosed in A7.2:** every `species_dim > 0` draw is an **information-free learned bias**, not a species channel |
| **2. GPU cap** | Lead made `.gpu_locks/ALLOWED` read-only at `0,1` (verified); every M4 GPU job also sets `SENTINEL_GPUS=0,1`. Core `gpu.py` untouched |
| **3. Registry guard** | Stays track-local (core `registry.py` is in other freezes' hashes) |
| **4. Figures** | Attribution of the two irreproducible counts to the reviewer kept |
| Sharding/merge | Approved; in use |

**Plan `b33bac5d…` → `ce6cf8b02266328f…`** (A7, append-only, 1 hunk, 0 removed); v10 pins preserved;
`verify_freeze()` PASSES; pinned module/guard and data unchanged. **35/35** select tests green.
Audit **OK**.

**One plan hash per entry.** Implementing the ruling needed a re-pin, and the A5 re-run had already
started under v10. Rather than let `rf`'s two shards straddle v10 and v11 (the review-(iii) hazard),
I **stopped it five minutes in** — one `rf` configuration, before any shard file or registry row was
written — applied A7, re-pinned, and **restarted every entry from scratch under v11** (20:16:56).
Cost ≈ 5 min of CPU. Partial log archived in `stage2_select/smoke/aborted_v10_pause/`.

**Running:** CPU queue (14 entries, `rf`/`xgboost` first, ≥2 shards each → verified merge →
registry). chemprop GPU smoke in progress to size its shards; chemprop then runs on GPU 0/1 in
parallel (GPU-heavy, 2 CPU threads — not a second CPU-heavy job). Test blocks unspent: 0/0/0.

### chemprop: two near-misses of mine tonight, both caught before any damage (2026-11-09 ~22:30)

**1. A wrong time estimate would have wasted the whole chemprop GPU budget.** I estimated ~10 min
per configuration by reading early `Trainer.fit … max_epochs reached` log lines as fold completions.
The smoke actually took **5,472 s (91 min)** for one configuration — and it happened to draw a
**3-seed** configuration, one of the 8 slowest of 32. The queue I had armed put 4 configurations
(~6 h) into each 2 h `timeout`, so **every shard would have been killed before writing its file**.
It was gated on the smoke's rc=0, which passed; I stopped it **before any shard launched**, so no
GPU time was lost.

Sized properly from all 32 sampled draws (time ∝ `max_epochs` × `n_seeds`, calibrated on the
smoke): per **fold** median 10 / max ~30 min; per **configuration** 20–122 min; **total ≈ 26
GPU-h** — matching the plan's original ~25 GPU-h estimate. Two configurations sit at ~122 min, at
the cap, and the estimate is calibrated on the smallest hidden size, so configuration-level sharding
is unsafe. **Fix: one configuration × one fold per job** (128 jobs, ≤ ~30 min each ≈ 4× headroom),
reusing `run_entry` so the fit and fold RMSE are computed identically, with a `--fold-merge` that
refuses missing, duplicated or drifted jobs.

**2. A dead-code gate, caught by its own test.** The new "refuse `--log-registry` for a fold job"
check was nested inside a block guarded by `config_index is None`, so it could never fire. Its test
was therefore **not refused and ran a real chemprop fit on CPU** (51k rows, load 35) until my test
timeout killed it — twice (exit 143). Verified afterwards: **no stray fold file** in the output dir,
**0 chemprop registry rows**, no leftover process. Fixed (the fold-job refusal now runs first); the
gate tests now write to `tmp_path` and assert nothing was written, so a broken gate can never spill
into the real output directory.

**Refactor safety across a running queue.** `xgboost` shard 0 ran *before* I moved the per-config
aggregation into `aggregate_folds`, shard 1 *after*. A test proves `aggregate_folds` is
**bit-identical** to the old arithmetic over 500 randomised trials including NaN folds, so the
`xgboost` merge combines identically-computed shards. **14/14** synthetic tests pass. Fold jobs now
go through `TrainingRun` / `TRAINING_LOG.md` (the first version bypassed it).

**Running:** CPU queue (`rf` merged; `xgboost` shard 1 running) and the chemprop fold-job queue on
GPU 0/1 (128 jobs, lock helper, cap file `0,1`, `SENTINEL_GPUS=0,1`).

---

# Architecture round v2 (2026-11-10) — reconciliation, and the T1/T6 amendment awaiting the stamp

Governing decisions: `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` §"M4 ChemTox:
update 2026-11-10 12:25 EST". Two supersessions now in force: **GPUs 0–9** are allowed (cap file
`$SENTINEL_RUNS/.gpu_locks/ALLOWED` reads `0,1,2,3,4,5,6,7,8,9`, verified) and the **disk floor is
60 GB free on `/`** (179–183 GB free today).

## 1. T2 reconciliation (the pre-registered track; untouched, running to completion)

`verify_freeze()` PASSES on the v11 pins (plan `ce6cf8b02266328f…`, module `cf1493c60f694741…`,
guard `e3721450bb97727c…`, data unchanged). **Test blocks still unspent: `registry_rows()` 0/0/0, no
`FREEZE`, no `AUTHORISED.json`, no `SCORED.json`.**

**10 of the 15 fitted entries are complete and in the registry** (live `rmse` rows, 32 configs × 4
A5 folds each, validation only):

| Entry | role | best mean RMSE |
|---|---|---|
| `xgboost_chan` | comparator | **1.0977** |
| `lightgbm_resid_surrogate` | candidate | 1.1016 |
| `lightgbm` | candidate | 1.1053 |
| `lightgbm_resid_ice` | candidate | 1.1055 |
| `catboost` | candidate | 1.1107 |
| `rf_chan` | comparator | 1.1111 |
| `xgboost_published_chan` | comparator (published, fixed) | 1.1130 |
| `rf_published_chan` | comparator (published, fixed) | 1.1520 |
| `rf` | comparator (channels off) | 1.2146 |
| `xgboost` | comparator (channels off) | 1.2209 |

A **channel-matched comparator still leads**, so claim conditions (d)/(d2) are failing at selection,
as protocol §9 pre-registered. Missing: `chemeleon_lightgbm`, `readacross`, `gpboost_published_chan`,
`gpboost`, `chemprop`.

**Processes, as found and as left.**

* **PID 1734640 — chemprop fold queue: alive, healthy.** 86 of 128 fold JSONs (configs 00–19
  complete, 20–21 partial), ~42 jobs ≈ 7 h left at the measured ~10 min/job. Left running
  untouched. Note for later: it exports `SENTINEL_GPUS=0,1` from before the GPU widening, so it can
  only lock GPU 0 or 1 while 4 and 9 sit idle; not restarted, because restarting it mid-job risks a
  duplicate write for no change to the result.
* **PIDs 2151255/2151259/2151260 — already exited** before this session looked. They were the
  `lightgbm_resid_surrogate` shards; both finished rc=0 and the entry merged and logged at 10:05.
* **PIDs 2143384/2143385 — `chemeleon_lightgbm` shard 1 of 4: STOPPED by me (by PID), with its queue
  parent 1671783.** Reason: **`chemeleon_lightgbm:4` cannot finish a shard inside the 2 h cap.**
  Shard 0/4 had 8 configurations × 4 folds, reached only 5 of them in 2 h and died **rc=124 writing
  no file**; shards 1–3 would have done the same, the merge would then have refused, and the entry
  would have been lost after ~8 CPU-hours. Logs preserved as
  `chemeleon_lightgbm.shard{0,1}of4.ABORTED_*.log`.

**Re-queued as 16 shards** (`$SCRATCH/m4_a5_rerun2.sh`, queue PID 2161301, log
`stage2_select/a5_rerun_queue2.log`): `chemeleon_lightgbm:16`, then `readacross:8`,
`gpboost_published_chan:2`, `gpboost:8`. **Only the shard count changed.** Sharding is
`global_index % shards == shard` over the same freshly re-sampled list and `--merge` refuses unless
every global index appears exactly once with the re-sampled parameters, so the configurations
fitted are bit-identical to a 4-shard or single-process run — **no entry, configuration, fold,
feature or rule is added or changed, and the pre-registration is untouched.** The new queue also
skips the merge and says so when a shard file is missing, instead of merging a hole.

## 2. New: the T1/T6 amendment — **UNSTAMPED, this is the gate**

`docs/preregistration/chemtox_arch_v2_amendment.md`, sha256
**`b9326d49aca711f0ceeec1362c21883030a94f83dd6971a2db15cba1babc9a9e`** (565 lines, new file; no
frozen file was edited). It covers plan §8.3 items 1–8 for **T1** (new primary, unseen ECFP4-Butina
chemical clusters) and **T6** (secondary, duration extrapolation, T6a/T6b), plus the R1–R5
references with the parity statement, the §8.2 caps with every configuration counted, the selection
rule, the A1–A6 ablations and the MDEs.

**No fit, no validation number and no test look exists on T1 or T6, and none may before the stamp.**

Frozen (0444) artefacts, hashes in the amendment §2: `docs/preregistration/chemtox_t1_clusters_v1.csv`
(`a6e33634…`), `$SENTINEL_DATA/processed/chemtox_arch_v2/chemtox_t1t6_split_v1.parquet` (`005f7957…`,
**label-free**), and under `$SENTINEL_RUNS/chemtox/arch_v2/`: `t1t6_counts.json`, `t1t6_mde.json`,
`h4_bcf_join_qc.json` and the three scripts, with `SHA256SUMS`.

**The row-disjoint four-way partition** (85,746 rows, after excluding T2's six reserved classes
outright — a strict superset of its reserved `result_id`s, so no frozen file is read — and
subtracting all 11,437 spent prospective `result_id`s):

| Block | Rows | Chemicals | Clusters | MDE ρ=0.9 | Replicate floor |
|---|---|---|---|---|---|
| `train_pool` (clusters kept, 24–96 h) | 59,729 | 2,960 | 1,979 | — | 0.504 |
| **`T1_test`** (clusters held out, 24–96 h) | 10,548 | 681 | 468 | **0.0097** | 0.495 |
| **`T6a_test`** (clusters kept, outside 24–96 h) | 13,059 | 1,175 | 847 | **0.0085** | 0.408 |
| **`T6b_test`** (clusters held out, outside 24–96 h) | 2,410 | 278 | 203 | **0.0197** | 0.458 |

All three blocks are better powered than T2's best (0.011). MDEs come from the **project's own**
`unseen_taxa.mde` and were verified **target-blind**: every configuration was run twice, once with
the real target and once with the target replaced by zeros, and required to agree to 1e-12 (all 12
passed), so no label of a future test row informed the power calculation. Replicate floors are
reported as absolute RMSE bounds, never as difference thresholds (the W-D-7 lesson).

**Reserved from 2026-11-10, never for training or tuning by any workstream:** the rows of
`T1_test`, `T6a_test` and `T6b_test` as listed by `block` in `chemtox_t1t6_split_v1.parquet`.

## 3. Pre-stamp work completed (lead's rule 2 for this round)

* **Licences verified and entered in `docs/licenses.md`.** **TRIDENT Zenodo 10260621: CLEARED** —
  `cc-by-4.0`, access open, v2 created 2023-12-05; both needed files exist
  (`03_Model_error_per_SMILES_from_10x10_CV.xlsx` 4,825,857 B;
  `02_Preprocessed_complete_data.xlsx` 30,897,579 B → 35.7 MB, the 2.8 GB weights and 3.2 GB source
  archives not taken). **R5 is admissible.** **EnviroTox: NOT CLEARED** — no licence or terms page
  exists, "All rights reserved", no download file locatable; **T5 is blocked and recorded as
  blocked**, not skipped.
* **H4 (BCF) join/QC report LANDED, so ablation A5 is admitted.** **17,418 usable rows / 672
  chemicals / 479 species** (14,951 wet / 2,467 dry basis, kept separate; 588 of the 672 chemicals
  are in the T1/T6 universe, 456 in the training pool). **Defect found and recorded: the BCF value
  lives in `bcf1_mean`/`bcf1_unit`, not `conc1_mean`** — on these rows `conc1_*` is the exposure
  water concentration, so reading it would have silently trained a bioconcentration head on the
  exposure concentration without raising an error. The circulating "21,352 usable" is not a usable
  count: 21,309 is the raw `L/kg`+`ml/g` row count before the value, structure and plausibility
  filters.
* **Not started: the P1 reference models R1–R3.** They are CPU fits and the one heavy CPU slot is
  held by T2's pre-registered selection queue, which has priority (lead's rule 4: nothing already
  queued is delayed). They start when that queue drains.

## 4. Next, in order

1. **Lead stamps the amendment** (sha256 above + a `chemtox/prereg_snapshot` row + a commit). Until
   then nothing on T1/T6 may be fitted.
2. T2's queue drains: `chemeleon_lightgbm` (16 shards), `readacross`, `gpboost_published_chan`,
   `gpboost`, then chemprop's `--fold-merge`. Then T2's selection summary and freeze candidate.
3. P1 references R1–R3 on the T1/T6 training pool (CPU, after the T2 queue).
4. Download the two cleared TRIDENT files (35.7 MB) and build R5's per-SMILES intersection.
5. Verify the embedding-model weight licence, then the embedding cache (1–2 GPU-h, ≤ 2 h chunks).

---

# Arch-v2 amendment v2 (2026-11-10 14:50) — v1 NOT stamped; v2 written, awaiting the delta review

**V1 (`b9326d49…`) was reviewed and the stamp WITHHELD.** Three reviewers plus a second-stage
verifier confirmed **five blocking defects** needing the file changed, not a ruling. Reviews:
`docs/preregistration/reviews/arch_v2_20261110/M4_verify.json` (27 verdicts, 4 points, 6 missed) and
the three first-stage files beside it; the verifier governs where they differ. V1 is preserved
byte-for-byte at `docs/preregistration/chemtox_arch_v2_amendment_v1_unstamped.md` (sha verified
unchanged). **Nothing in v1 may be quoted as pre-registered.**

## v2

`docs/preregistration/chemtox_arch_v2_amendment.md`, sha256
**`fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b`** (1,150 lines). §0 maps every
verdict id to the section that answers it. **UNSTAMPED. No T1/T6 fit, no validation number, no
look.** `arch_v2_guard.verify_freeze()` correctly reports "not stamped" (the candidate pins sit at
`chemtox_arch_v2_pins.candidate.json`, not at the path the guard reads).

### The five blocking fixes

1. **SPENT-ADORE (L1).** `arch_v2.scored_adore_partitions()` derives from `results/registry.jsonl`
   **by code** every ADORE `challenge x split` partition with a scored test row, in any experiment —
   **a trivial-baseline scoring spends a block** (the conservative branch). **8 partitions / 28,265
   `result_id`s**, of which **26,808 rows sat in the v1 universe: 4,276 of v1's own `T1_test`
   (40.5 %) and 22,532 of its pool.** Every verifier figure reproduced exactly. They now leave the
   universe **entirely** (no block, no pool, no auxiliary/pretraining/ablation row), frozen as
   `chemtox_arch_v2_spent_adore_ids_v2.json`.
2. **MDE-ROW-INDEPENDENT (L5).** New module `sentinel/models/chemtox/arch_v2_power.py` (frozen T2
   code untouched) with a within-unit error component. Gate rose from ~0.010 to **0.096 / 0.103 /
   0.103**. Two conservative strictenings beyond the ruling: the **exact unequal-unit-size** formula
   (one unit holds 9.8–14.4 % of each block) and 8-realisation averaging in the bootstrap check.
3. **PROTECTED-LABEL-STATS (L6).** The split file has **no target column**; the power helper takes
   unit-size vectors; all per-block SD and replicate-floor columns struck (only the pool's 0.504
   survives, as absolute context). v1's two label-bearing reports kept bytes-unchanged as
   `*_v1_WITHDRAWN_label_stats.json`. The v1 pre-stamp read is disclosed in §15.1 with the list of
   design choices that predate it.
4. **AUX-LEAK (L7).** `arch_v2.auxiliary_rows()` applies seven counted exclusions and asserts the
   result. **H3 chronic 225,081 → 121,678 (T1) / 45,661 (T6)**; H2 11,719 → 5,388 / 4,455; H4 17,418
   → 10,050 / 2,984. On T6 the max labelled duration in training is now **96.0 h** (76,017
   out-of-window chronic rows removed), which is what makes §1.2's "structurally handicapped
   comparator" argument true rather than false.
5. **BUILDER-PROVENANCE (L8).** `arch_v2.build_split()` emits all three pinned artefacts at their
   pinned paths with the column trim inside the function. The v1 eight-column labelled intermediate
   (`…/scratchpad/t1t6/chemtox_t1t6_partition.parquet`, 1,609,554 B, sha `9ef801b8…`) is **deleted**,
   recorded first. No pinned artefact lives in a temporary directory.

### The v2 partition (30 % of units; L2 strict hold-out unit)

Hold-out unit = **connected component over (`t1_cluster`, `parent_key` family)**, so no cluster and
no parent family spans the hold-out — asserted in code and in two tests. 2,153 clusters → **2,029
units** (124 merged by a shared parent key).

Universe **58,824** rows / 3,210 CAS (122,946 → −24,219 T2 classes → −11,437 prospective → −26,808
spent ADORE → −1,640 duration floor/missing → −18 unparseable).

| Block | Rows | Units | **Gate (icc 0.5)** | icc 1.0 | Largest unit |
|---|---|---|---|---|---|
| `train_pool` | 32,106 | 1,293 | — | — | 13.5 % |
| `T1_test` | 11,363 | 546 | **0.0959** | 0.1911 | 9.8 % |
| `T6a_test` | 11,249 | 694 | **0.1027** | 0.2047 | 14.4 % |
| `T6b_test` | 4,106 | 289 | **0.1033** | 0.2044 | 13.5 % |

### The consequence the lead must weigh (§15.3)

**At ~0.10 the blocks cannot resolve any differential this project has ever measured between
parity-matched models** (prospective headline 0.029; largest T2 spread 0.070). All three blocks are
therefore pre-registered **DESCRIPTIVE by default** per L4; a §15.1 label needs a realised difference
above the gate. The driver is **unit-size concentration**, not the subtraction. Four options are put
to the lead in §15.4; my recommendation is to accept the descriptive framing and report honestly.

### Pinned v2 artefacts (0444; `SHA256SUMS_v2` in `$SENTINEL_RUNS/chemtox/arch_v2/`)

`chemtox_t1_clusters_v2.csv` `a906aac1…` · `chemtox_arch_v2_spent_adore_ids_v2.json` `3fa94b68…` ·
`$SENTINEL_DATA/processed/chemtox_arch_v2/chemtox_t1t6_split_v2.parquet` `ab71aace…` (label-free,
9 cols) · `t1t6_mde_v2.json` `f67d0376…` · `h4_bcf_join_qc_v2.json` `7dbdd501…` ·
`adore_partition_inventory_v2.json` `d2c643b6…`. **v1's split and cluster CSV are kept untouched as
the record and are SUPERSEDED — `chemtox_t1t6_split_v1.parquet` must not be registered as the
reservation (its blocks contain spent rows).**

### New code (all new files; nothing frozen edited)

`sentinel/models/chemtox/arch_v2.py` (builder + AUX-LEAK loader) · `arch_v2_power.py` (L5) ·
`arch_v2_guard.py` (L10: own pins, own experiment key `chemtox_arch_v2_test`, lead-written FREEZE as
the only unlock, BaseException abort, import-closure code hash, artefact verification + sealing,
protected-labels artefact built only at authorisation, `load_training_pool` as the pool reader and
`load_test_block` as the only sealed reader) · `scripts/eval/chemtox_arch_v2_power_report.py` ·
`scripts/eval/chemtox_arch_v2_h4_report.py`. **Tests: 18/18 power + 42/42 builder/guard; 148/148
across the whole M4 suite.** `unseen_taxa_guard.verify_freeze()` still passes; T2 blocks 0/0/0.

### T2 meanwhile (untouched, running)

chemprop **100 of 128** fold jobs. CPU queue on `chemeleon_lightgbm` shard 4/16 (slow config 20);
shards 0–3 rc=0 in ~18 min each, so the 4→16 re-shard fixed the 2 h-cap failure. Then `readacross`,
`gpboost_published_chan`, `gpboost`, then chemprop's `--fold-merge`.

### Next

1. **Lead delta-reviews v2 and rules on §15.4** — the 30 % share, the unscored ADORE partitions
   (4,632 further rows if they also go), the `gpboost_model.py` citation, and the
   descriptive-vs-claim framing. Then the stamp: `make_pins()` → `chemtox_arch_v2_pins.json` +
   `chemtox/prereg_snapshot` row + commit, and the §8.5 reservation entry in `sota_campaign.md`.
2. Guard review (L10) before any look.
3. P1 references R1–R3 on pool rows via `load_training_pool()`, after the T2 CPU queue drains.

---

# Arch-v2 STAMPED (partial) 2026-11-11T00:20:53Z; guard rebuilt for B1 (2026-11-10 19:35 EST)

Stamp row `0b9c47f967e9`. **Governing file: `docs/preregistration/chemtox_arch_v2_lead_rulings.md`**
(sha256 `553be413bef252e5900023dc1dc983dd8f228a5825194ca70505cf6649c7f060`, 343 lines, 0444) — **it
wins over the amendment wherever they differ, and the two must always be read and cited together.**
The amendment (`fb52cd4b…293b`) and `ecotox_aquatic.parquet` (`beec1f81…fc9e`) are 0444; both
verified on disk at their stamped hashes.

**What the stamp does NOT do:** it releases no data access. The lead did not write
`chemtox_arch_v2_pins.json`, so `verify_freeze()` keeps raising and neither `load_training_pool()`
nor `authorise()` can run. **All three blocks are DESCRIPTIVE ONLY (F1): no §15.1 label is available
on T1_test, T6a_test or T6b_test.** A powered claim needs a new track under lead point (f).
Hold-out share 30 % stands on grounds (i) and (iii); **ground (ii) is struck (B2) — no write-up may
cite a "10 % of a block" criterion as met.** The 97 unscored ADORE partitions stay in the universe
and **no ADORE partition is scored before all three looks** (point (b)).

## Item 1 (critical path) — DONE: guard meets B1 (i)–(v) and C-1

`sentinel/models/chemtox/arch_v2_guard.py`, **new sha256
`9bd7d60a7267ff219fdda0ea855973552dcfb85d649f208c9867d8cd0ecc066e`** (638 lines; was
`6e1ab9c1…f224fe` at the stamp).

| B1 | Implemented as |
|---|---|
| (i) | `freeze_sha()` checks sha256(FREEZE) against **both** lead records and refuses on any mismatch: the 0444 `docs/preregistration/chemtox_arch_v2_freeze_shas.json` (path is the guard constant `FREEZE_RECORD`; refused if absent, writable or malformed) **and** a registry row of `chemtox/prereg_snapshot` with `extra.written_by == "lead"`, `extra.block`, `extra.freeze_sha256`. Disagreement, duplication or absence of either refuses the look |
| (ii) | `require_authorised()` re-runs `verify_freeze()`; `load_test_block()` refuses once `SCORED.json` **or** `ABORTED.json` exists |
| (iii) | `_Pass.finish()` runs `verify_artifacts()` then `mark_scored()`; `mark_scored()` independently refuses unless a verification passed, so the order cannot be bypassed. `verify_artifacts()` now also rejects files absent from `SHA256SUMS` |
| (iv) | `ABORTED.json` carries `_written_inventory()`: every file with size and sha256, which markers exist, `look_spent` and the reason — no constant |
| (v) | `make_pins()` covers `ecotox_aquatic.parquet` (C-5) and the labels artefact; `verify_freeze()` **refuses pins that omit either** |

Also hardened while there: `import_closure_sha()` now anchors on `_CODE_ROOT` (this file's own
location) rather than the patchable `REPO`, so a claim's `code_sha` cannot be blanked by redirecting
path constants. An abort after `AUTHORISED.json` spends the block's look; `authorise()` and
`scored_pass()` refuse afterwards.

**Tests (synthetic / tmp paths only; no universe row read):**
`tests/test_chemtox_arch_v2_guard.py` `f314f53c1cbf3b1d96cfb43c214089c529acca09ac543731145f6eec8c2a7d76`
(542 lines, **47 passed**) · `tests/test_chemtox_arch_v2.py`
`17065211bf508e79ace19303675caa9b877820208b770db4324c9420f69f240b` (**27 passed, 7 skipped**) ·
`tests/test_chemtox_arch_v2_power.py` `b9af9b24cf4829e7a629f5883f3d470868a95170fde79e898ffedcab2481f5b4`
(**18 passed**). Total **92 passed, 7 skipped**.

**B1 re-pin constraint holds** — unchanged before and after: `arch_v2.py` `39ecc9cc…fe980`,
`arch_v2_power.py` `08419030…d4f7a`, amendment `fb52cd4b…293b`, split `ab71aace…23a3`, cluster table
`a906aac1…d2821f5`, spent-id list `3fa94b68…bd1d8`. Regenerated candidate pins:
`docs/preregistration/chemtox_arch_v2_pins.candidate.json`
`3eeac280f5013e4ef5911841043c3e847ef8d9af9813d47f81965d27ed61d51b`. **Awaiting the lead's
independent guard review; the lead writes the pins file (C-2 step 3), which opens S2.**

## Two items needing a lead ruling

1. **N2 vs `auxiliary_rows()`.** `arch_v2.auxiliary_rows()` reads `ecotox_aquatic.parquet`
   **directly**, which N2 makes a protocol breach, and `arch_v2.py` is byte-frozen by B1 — so the
   AUX-LEAK loader cannot be run post-stamp without breaching. Its 7 tests are gated behind
   `M4_ALLOW_DIRECT_ECOTOX_READ=1` and are **skipped** (they passed pre-stamp). `scripts/eval/chemtox_arch_v2_h4_report.py`
   has the same issue. Needs an addendum under C-7 routing them through the guard, or an exemption.
2. **The review index** `docs/preregistration/reviews/arch_v2_20261110/README.md` records the M4
   amendment hash as `4c4e9ba9…` — an intermediate draft, neither the reviewed v1 (`b9326d49…`) nor
   the stamped v2 (`fb52cd4b…`).

## T2 (unchanged, under its own plan and pins)

**chemprop COMPLETE**: 128/128 fold jobs, fold-merged 19:22, best mean RMSE **1.1679**, registry row
written. **11 of 15 fitted entries now live**; `xgboost_chan` **1.0977** (a mandatory
channel-matched comparator) still leads every candidate, so the pre-registered negative holds.

`chemeleon_lightgbm` **shard 4/16 hit rc=124** (2 h cap, 13:18→15:18) and wrote no file, so its
merge would refuse on a missing index. Repair queued (PID 2438617, waits for the running queue to
exit, then one config per process): configs 4 and 20 re-run as **shards 4 and 20 of 32**, which
`merge_shards()` accepts because it validates by **index coverage, folds and seed**, not by a
consistent shard count, and re-checks every configuration against the re-sampled list. No setting,
entry, configuration or fold changes. Then `readacross`, `gpboost_published_chan`, `gpboost`.

**GPU spreading is moot:** the only remaining T2 work is CPU (chemprop is done), and of GPUs 0–9
only GPU 0 is idle — 2, 3, 4, 5, 6, 7, 8, 9 all carry another user's process. Nothing was started
on a GPU.

## Next

1. Lead's guard review → lead writes `chemtox_arch_v2_pins.json` → S2 opens `train_pool` work.
2. Item 2 (no pool read): new modules + synthetic tests for every arm and comparator; config files
   before any pool fit (`rf_emb_dur` per F10, the closed F3 baseline list, B = 10,000 and the F6 seed
   pairing in the scorer, the F8 outcome wording in the report template); licence checks for the
   §5.6 embedding precedence.

## Addendum 01 (ruling N2-A) done — 2026-11-10 19:45 EST

`docs/preregistration/chemtox_arch_v2_addendum_01.md` sha256
**`9a195e643c420c39b7fd4451053fd3f482e738441d3ca626e85659cd342c9f2e`** (new, dated, proposed —
takes effect only when the lead reviews it with the guard and writes the pins file). It proposes
**one** change: N2's sanctioned readers of `ecotox_aquatic.parquet` for T1/T6 work become four —
`load_training_pool()`, `build_protected_labels()` and the new `load_auxiliary_rows()` and
`load_h4_rows()`. No bar, split, block, comparator, statistic, cap or outcome wording changes; all
three blocks stay descriptive only.

Guard **`c9620f372a471446721fca805b700483bda534ab7e99912fb6ce26d73f3268e9`** (was `9bd7d60a…c066e`).
Both new readers refuse unless `verify_freeze()` passes, fix every path to the pinned constants
(`_no_caller_paths` rejects a caller-supplied `ecotox`/`prospective`/`registry`/`splits`/
`cluster_table`/`split`/`results`/`tests`), and after the frozen `auxiliary_rows()` has run
re-check **independently** that no returned `result_id` is a sealed one, no returned
`(test_cas, latin_name, duration_h)` cell is a sealed cell, and no returned `test_cas` maps to a
held-out hold-out unit — hard error with counts only in the message. The exclusion report (counts
only, 0444) goes to `$RUNS/chemtox/arch_v2/aux_exclusions/`.

New pure module `sentinel/models/chemtox/arch_v2_h4.py`
**`a21e517ad47c8f7c975c094913f0d57b3af66c7d73b5e44a5a1364982e768865`** holds the BCF join/QC rules
unchanged and performs **no I/O** (a test asserts it). `scripts/eval/chemtox_arch_v2_h4_report.py`
**`a1267729f39d09c60c1eeaeb51973722dc8977a0fbafda22264df7211795092b`** now calls `load_h4_rows()`
and contains no `read_parquet` (also asserted by a test); it writes `h4_bcf_join_qc_v3.json`.

Tests **107 passed, 7 skipped**: guard `tests/test_chemtox_arch_v2_guard.py`
`61e26e7bf9231293b6b6507281f4a22ebcea770463fc91570597e8b6556ceff9` **62 passed** (47 for B1 + 15 for
addendum 01), `tests/test_chemtox_arch_v2.py` **27 passed, 7 skipped** (the gated AUX-LEAK tests stay
skipped on real data), `tests/test_chemtox_arch_v2_power.py` **18 passed**. Synthetic frames in tmp
dirs only — no universe row read.

Candidate pins `docs/preregistration/chemtox_arch_v2_pins.candidate.json`
**`3accb83704aabb62eecce078fddcdbbedcf05d4ee9f010760178c41a9add72d6`**, now 9 pinned repo files
(adds the addendum and `arch_v2_h4.py`) and 3 data paths (split, ecotox, labels = `null`);
reproduces exactly from `make_pins()`.

**Rulings §1.1 unchanged**: `arch_v2.py` `39ecc9cc…fe980`, `arch_v2_power.py` `08419030…d4f7a`,
amendment `fb52cd4b…293b`, cluster table `a906aac1…821f5`, spent-ids `3fa94b68…d1d8`, split
`ab71aace…23a3`, ecotox `beec1f81…fc9e`. Nothing ran on real data: no `aux_exclusions/`, no
`h4_bcf_join_qc_v3.json`, no pins, no freeze record, no labels artefact, `registry_rows()` 0/0/0,
and all three readers verified to refuse.

T2 repair recorded in `$RUNS/chemtox/unseen_taxa/stage2_select/RUN_LOG.md` with the rc=124 cause.

## Item 2 — arms, scorer, configuration, licences (2026-11-10 20:05 EST)

Nothing on real rows; the guard stays closed. None of the eight files frozen for the lead's review
was touched (hashes re-verified), and `arch_v2.py` / `arch_v2_power.py` are at their §1.1 hashes.

**New files.**

| File | sha256 | Pin? |
|---|---|---|
| `configs/chemtox/arch_v2.yaml` | `d59caab6f68abe6491cffe7bd144107f3db0a1ac75e143cdcdb35b044626a3ef` | **yes** |
| `sentinel/models/chemtox/arch_v2_scorer.py` | `2205246a114cc92085f48c5492cbd4354e7bb28c7dc2f1d9fa0c47b05e2be059` | **yes** |
| `sentinel/models/chemtox/arch_v2_arms.py` | `264e3bba9aa8c869a6edf76eb9172518f4625d01173a39336f3a46baca103f3e` | **yes** |
| `docs/preregistration/chemtox_arch_v2_report_template.md` | `f75dec7ebf94f7e183ed032a62e299d47041ffcb599496d6949d6f3ef87fd3b1` | **yes** |
| `tests/test_chemtox_arch_v2_scorer.py` | `e373c4787f4f4926186c44ffe583b21020daf6b2575a19c24fe7b907928c3769` | living |
| `tests/test_chemtox_arch_v2_arms.py` | `1b89b9a09c40894d9365e4bb126f1abdcf29999e3d83d683abf13af1ba227a80` | living |

The four "yes" files do **not** need another guard edit: `verify_freeze()` iterates over whatever
the pins file's `files` object contains, so the lead can add them to the pins JSON directly and they
are enforced from then on. `make_pins()` would need a guard edit to emit them by default, which is
deliberately **not** done while the guard is under review.

**The configuration** fixes, before any pool fit: the entry inventory and caps (384 / 128 searched,
26 fixed, 12 ablations, **550** across 44 entries — re-derived by `validate_caps()`, not trusted);
**F10** `rf_emb_dur` = `rf_emb` + `duration_h` and nothing else, as a **fixed** T6 entry so the 128
cap is untouched; the **closed F3 list** (17 entries, validated to contain no candidate and no
published-form arm); **F6** B = 10,000, seeds 0/1/2, seed-mean paired difference, index pairing,
`p = 2·min(share≤0, share≥0)` with `(count+1)/(B+1)`; **F5** both clusterings; **F7** the analytic
MDE governs and the bootstrap is never a gate; **F2** condition (a) by pool validation; **F8** the
five outcome wordings; per-entry GPU ceilings and the 95 GPU-h total.

**`arch_v2_scorer.py`** implements F1/F5/F6/F7/F8 over arrays — no I/O — so the statistical contract
is testable with the guard closed. `outcome_label()` cannot return a SOTA label and refuses any
block outside the three descriptive-only ones. Note for the look: the paired bootstrap is O(B x n),
so one leg on an 11k-row block at B = 10,000 takes a minute or two — budgeted, not a problem.

**`arch_v2_arms.py`** holds the registry (`validate_caps`, `condition_b_list`, `duration_variant`,
`feature_fingerprint`, `assert_parity`) and the comparators that need no new research code:
`GlobalMean`, `GroupMean` (per species / per chemical / per species x band), `HaberRule`,
`NearestDurationCarryForward`, plus **R1** monotone-duration LightGBM, **R2** XGBoost
`survival:aft`, **R3** multitask-by-stacking. `build_arm()` raises `ArmNotImplemented` for every
declared-but-unwritten arm (`ted`, `timehead`, `mmoe`, `rx_*`, the embedding arms, the ensembles,
the published arms, `readacross`), so **nothing can be reported as having run when its code does not
exist**; a test asserts no declared entry is unaccounted for.

Two tests are worth the lead's eye because they verify the §1.4 corrections the whole round rests
on: R1 is **proved monotone** on synthetic data where the true relationship is not, and R2 **fits
interval-censored targets** with half the rows right-censored.

**Tests: 35 scorer + 53 arms, both green**; full five-suite count in the next report.

**Licences (§5.6 precedence, resolved and recorded in the config before any fit; 3 rows added to
`docs/licenses.md`).** **MolFormer-XL `ibm-research/MoLFormer-XL-both-10pct`: CLEARED and CHOSEN** —
Apache-2.0, ungated, 374,640,429 B, 46.8 M params. **ChemBERTa-2 `DeepChem/ChemBERTa-77M-MLM`: NOT
CLEARED** — no licence tag and no model card; DeepChem's MIT licence covers the code, not these
weights. **CheMeleon (Zenodo 15460715): MIT, open** — the declared fallback, already in-tree, not
used because MolFormer-XL cleared first.

**Still to build:** the learned arms `ted`, `timehead`, `mmoe`, `rx_chemprop`, the embedding block
and the two ensemble builders, each in its own new module with synthetic tests.

## Item 2, second wave — the learned arms (2026-11-10 20:25 EST)

Nothing on real rows; the guard stays closed. The eight review-frozen files and the four to-pin
files are byte-unchanged, and `arch_v2.py` / `arch_v2_power.py` are at their §1.1 hashes.

**New modules** (all new files; `arch_v2_arms.py` deliberately NOT edited, so `build_arm` still
raises `ArmNotImplemented` and the new `arch_v2_build.build()` is the runner's entry point):

| Module | sha256 | Pin? |
|---|---|---|
| `arch_v2_timelaw.py` | `327bfcb2ccb0a533d6917a22c389f1c1d3b78213230eea9745334ab755267b47` | **yes** |
| `arch_v2_embed.py` | `595a8b02358cc1e7dfc5a0d8634e4c76ac726a5e8dfb5e415d2797a662b2e84c` | **yes** |
| `arch_v2_ted.py` | `4da8a6dda24a450691e13b5e23edf207cec97dc94e4d71b07b4898244ff69ecf` | **yes** |
| `arch_v2_timehead.py` | `19f7480aa56b832f9de3904d995d02d926a1633123de9723d2a67b4527bba3fd` | **yes** |
| `arch_v2_mmoe.py` | `348921c7dbcd09ff73944226353dcb8de998e411c664e72a20d5d28625e00dd8` | **yes** |
| `arch_v2_rx.py` | `886a39ca69e78863ed9d7d8846b9fc30a4dcf6cab6bddab238eea28e7e29f134` | **yes** |
| `arch_v2_ensemble.py` | `a489609e21ab99750d62e3ebf47f0472c60e3856d6c4547ed48262982ff638cc` | **yes** |
| `arch_v2_build.py` | `729aacf13feb7de241f1be7a1131af7a897e88af35fb90b0ad7d019d8518c4a8` | **yes** |

New tests (living): `test_chemtox_arch_v2_timelaw.py` `557d6f9c…`, `_embed.py` `151348fd…`,
`_ted.py` `e924e5b3…`, `_timehead.py` `14e9c4fd…`, `_mmoe_rx.py` `534c42b0…`,
`_ensemble.py` `c441d29e…`.

**A real bug the tests caught.** The first version of the saturation term used
`log1p(-exp(-kt))`; the amendment specifies `-log10(-expm1(-kt))`. The difference is not cosmetic:
at `kt ~ 1e-4` the `log1p` form has already lost most of the mantissa, and at `kt ~ 1e-9` fp32
rounds `exp(-kt)` to exactly 1.0 and the term becomes `-inf`. Fixed to the `expm1` pairing, with a
test that compares against an fp64 reference **and** asserts the naive form is measurably worse.

**Properties now enforced in arithmetic, not prose** (`arch_v2_timelaw.py`, shared by TED and
TimeHead): monotone non-increasing in duration over a 200-draw random sweep out to 26,298 h;
`EC10 <= EC50` for every parameter value and every t, via `k_chr = k·(1+softplus(v))` and the fact
that the saturation term is decreasing in k; `log10 ACR >= 0`; `NOEC <= MATC <= LOEC` by cumulative
softplus; the composition layer forced to fp32 inside an autocast-disabled block.

**TED** carries A1–A6 as constructor flags with `ablation_name()`, and
`assert_no_time_in_trunk()` fails the run if a duration- or endpoint-derived column reaches the
trunk — except under A3, which requires one to be present or it would measure nothing. A6's Q10 is
**learned with a free prior**; a test asserts it is not initialised at the literature value.
**TimeHead** refuses a gradient-carrying asymptote, because the tabular model is frozen by design.
**MMoE** imposes L3's ordering in output space, so the contrast with TED's one-latent derivation is
visible in code. **RX** makes `multitask=True` inadmissible and forces the constant placeholder
species key, disclosing `species_dim > 0` as an information-free learned bias. **Ensembles** refuse
to proceed without the matched baseline control and refuse any ensemble claim on T6.

`arch_v2_build.coverage()`: **30 declared entries, 0 unaccounted** — 12 learned, 9 delegated to the
pinned arms module, 9 named in `NOT_BUILDABLE` with a reason.

**Tests: 25 timelaw + 20 embed + 23 ted + 10 timehead + 27 mmoe/rx + 25 ensemble, all green.**

**One question for the lead:** `torch.cuda.is_bf16_supported()` returns **True** on these GPUs,
contradicting the architecture plan's note that it returns False. The amendment pre-registered
fp16 AMP with the composition layer in fp32. Since the composition layer is forced to fp32 either
way, the autocast dtype for the trunk changes nothing that is measured — but bf16 would be
numerically safer. Flagging rather than acting.

---

# Guard review → `fix_then_rereview`; G-1…G-12 implemented (2026-11-10 21:1x EST)

The independent guard review returned **`fix_then_rereview`**, so the lead did **not** write the
pins file: `verify_freeze()` still raises and no pool, auxiliary or sealed row may be read. Governing
file: `docs/preregistration/chemtox_arch_v2_lead_guard_rulings_01.md` (sha256 `63b7184c…fa1a9a`),
which wins over addendum 01; the stamped amendment and stamped rulings win over it. Review:
`docs/preregistration/reviews/arch_v2_20261110/M4v2_guard_review.{md,json}` — F1–F14, probes
P1–P7b, 9 mutations killed and **6 survived** (M7, M8, M9, M10, M12, M14).

## Why the pins were withheld, and what each fix is

* **F1 (blocking) — the gate did not require the lead.** `verify_freeze()` passed on a two-key JSON
  any workstream could write, or on a byte-copy of the candidate. It now requires **all** of: the
  pins file carrying no write bit; a `chemtox/prereg_snapshot` row with
  `extra.written_by == "lead"` whose `extra.pins_sha256` equals the sha256 of the pins on disk;
  every `PINNED_REPO_FILES` key and every pinned data path present with a 64-hex value; no unknown
  entry and no null but the labels path. **Every reader writes the pins sha into its report.**
* **F2 (blocking) — a test would have read real tables the moment the pins existed.** The real-path
  tests now assert the pins are absent and **skip before any reader call**; `aux_sandbox` redirects
  `G.H4_RESULTS` / `G.H4_TESTS`, and the base `sandbox` now provides every pinned data path.
* **F3 (blocking) — H4 applied four of the seven exclusions and had no test.** All seven now apply,
  in the stamped order, each with its own counter: the **class join** and **both spent-id cuts** are
  new. 24 new H4 tests cover the happy path, each of the seven, the T6 window, the re-check and the
  report. The pre-stamp counts **10,050 / 2,984 are marked SUPERSEDED**; H4 is re-measured once
  through the guard after the pins.
* **F4 — no route to an H2/H3 label.** The guard now joins the target itself for exactly the ids it
  returns, after the re-checks, and raises if any lacks one. **H5** has a reader of its own,
  `load_h5_pool_pairs()`, from `load_training_pool()` rows **only**; it takes no arguments, and the
  honest consequence is recorded: the pool is acute, so **there is no supervised ACR target** and
  none is fabricated.
* **F5 — a second look through a different runs root.** `make_pins()` records the resolved runs
  root and `verify_freeze()` refuses a different one; `authorise()` writes a claim row
  (`chemtox_arch_v2_look_claim`, **not** a `prereg_snapshot` row) **before** any sealed read, so a
  later `authorise()` refuses from any root while the same look's later steps proceed. The
  copied-FREEZE escape of probe P6 is now a test.
* **F6 — the labels artefact was trusted by path.** Write-once, hash-recorded in a marker and in
  `AUTHORISED.json`, verified at every read, and `load_test_block()` asserts the artefact's target
  equals the pinned frame's for every returned id. Probes P7 and P7b are tests; a legitimate second
  block reuses the recorded artefact.
* **F7 — unpinned inputs.** `prospective.parquet`, the ADORE split assignments and the two raw
  tables are now pinned and written into every report, and the reader **refuses if the
  registry-derived spent-ADORE set differs from the pinned list**. `PINNED_REPO_FILES` is 27 files.
* **F8 — reports were overwritten.** Exclusive creation with a counter under a fixed runs-root
  directory, 0444, carrying the pins sha, time, pid and every input hash. The H4 script takes
  **no output path**.
* **F9 — the re-check was not fail-closed.** It now covers all seven for H2/H3/H4, **takes no
  context from the caller** (it rebuilds every set from the pinned files, so one broken derivation
  cannot blind both), normalises CAS keys so `' 400'`, `'400.0'`, `'0400'`, `'40-0'` all resolve,
  and **raises** on chemistry that maps to no hold-out unit instead of counting it.
* **F10 — the reader list.** Five sanctioned readers, named in the guard docstring and in the
  revised addendum. The env switch `M4_ALLOW_DIRECT_ECOTOX_READ` is **removed**: the 7 items skip
  always, with the reason.
* **G-12** — `chemtox_arch_v2_addendum_01_rev1.md` replaces addendum 01 in full and opens with a
  14-row list of every change. **The pre-rewrite v2 H4 script (sha `5653546f…`) could NOT be
  recovered byte for byte** — no copy with that hash exists in the repo, the runs tree, the scratch
  directory or any git object — so it is **not** restored and **not** rewritten from memory, and the
  addendum says so. Addendum 01 itself is untouched at `9a195e64…9f2e`.

## Two defects found while fixing, both real

1. **A column collision in the pool loader.** The split and `ecotox_aquatic.parquet` both carry
   `test_cas` and `duration_h`, so the plain merge produced `test_cas_x` / `test_cas_y` and broke
   every caller that asked for `test_cas` by name. `_attach_split()` now adds only the split-only
   columns and asserts the row count is unchanged.
2. **The code hash could be blanked.** `import_closure_sha()` anchored on the patchable `REPO`;
   it now anchors on `_CODE_ROOT`, resolved from the module's own location.

## State

No pins file, no FREEZE, no freeze-sha record, no claim row, no labels artefact; `verify_freeze()`
raises; `registry_rows()` 0/0/0 for all three blocks. Nothing has been run on real data. The §1.1
hashes are unchanged, and `arch_v2.py` / `arch_v2_power.py` are byte-identical.

T2 continues unchanged under its own plan; the repair queue (PID 2438617) is still waiting behind
the CPU queue. Observation for the record only, per the lead's ruling: `is_bf16_supported()` returns
True on these GPUs, but fp16 AMP with the composition layer in fp32 stays as stamped.
