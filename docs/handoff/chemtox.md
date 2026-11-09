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
