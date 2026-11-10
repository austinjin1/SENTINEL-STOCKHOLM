# M4 SENTINEL-ChemTox: architecture research and SOTA plan

**Author:** architecture-research agent (read-only pass), 2026-11-10.
**Scope:** aquatic acute toxicity, log10 LC50/EC50 (mol/L), chemical x species x test conditions.
**Status of this file:** a proposal for the lead. It pre-registers nothing. No code was edited, no
model fitted, no dataset downloaded, no process touched, no frozen file read.

## Verification rules used in this document

Three labels appear throughout and they mean exactly what they say:

* **[MEASURED HERE]** — I computed the number in this session from files already on disk. The
  appendix gives the frame, the filter and the result so any of it can be re-derived.
* **[VERIFIED HERE]** — the citation's title, authors, journal and DOI were retrieved in this
  session (Crossref record or a web search result, URL given). Retrieving metadata is *not* reading
  the paper: where I rely on a result from inside a paywalled paper, that result is marked
  separately.
* **[UNVERIFIED]** — carried over from the workflow's literature pass, from `docs/research/04`, or
  from an earlier session's note, and **not** re-checked today. Nothing labelled this way may enter
  a claim, a model card or the paper until it is verified.

My web-search budget was exhausted after one query (session-wide limit, shared with six other
workstreams), so most citation checks went through the Crossref REST API by direct fetch. That
verifies bibliographic metadata, not the numbers inside the articles. Every in-paper number below is
labelled accordingly.

---

## 0. Decision summary

**The short version: M4's problem is not that its architectures are too weak. It is that every
architecture it has tried sees the same 26,114 rows, one endpoint, and a duration axis compressed to
four levels — while 122,946 usable acute rows, 225,081 chronic rows and a 444-value duration axis
sit unused on disk.** The live unseen-taxa selection says this out loud: the gap between model
families is ~0.07 RMSE, and the gap between *information conditions* on one fixed model family is
~0.11 RMSE. Information is worth about 1.5x architecture on the track that is running right now.

Six decisions I would ask the lead to take, in this order:

1. **Finish T2 and expect a negative.** As of 2026-11-10 11:37, with 10 of 27 entries complete, the
   best validation score on the unseen-taxa rotation belongs to a **mandatory comparator**
   (`xgboost_chan`, 1.0977), ahead of every finished candidate and ahead of chemprop's best
   complete configuration (1.1679, 19/32 configurations scored). Claim conditions (d) and (d2) are
   currently failing at the selection stage. [MEASURED HERE] T2 should run to completion and be
   reported as the honest negative it was pre-registered to produce. Do not add an architecture to
   it: the pins make that an amendment, and the A7 restart already showed what that costs.
2. **Build the two parity references before any new architecture.** A monotone-constrained GBM on
   continuous duration, and an interval-censored AFT GBM. Both are one-line capabilities of the
   installed libraries (LightGBM 4.7.0 `monotone_constraints`, XGBoost 3.2.0 `survival:aft`, both
   executed here [MEASURED HERE]). Two of the mechanistic design's six claimed GBM-impossible
   properties are therefore **not** GBM-impossible, and a claim made without these comparators would
   not survive review.
3. **Make T1 the new primary track, but block on ECFP4 clusters, not Bemis-Murcko scaffolds.**
   46.2% of usable acute rows have no Murcko ring scaffold at all, and the single benzene scaffold
   covers 24,443 rows / 1,052 chemicals — 20% of the endpoint in one "family". [MEASURED HERE]
   An ECFP4-Butina clustering at cutoff 0.4 gives 2,738 clusters whose largest holds 7.2% of rows,
   and a held-out 10-20% of those clusters gives a cluster-bootstrap MDE of 0.006-0.008 at rho=0.9 —
   better-powered than any T2 block. [MEASURED HERE]
4. **Add a genuinely harder track that the published competition does not already own: duration
   extrapolation (T6).** 15.5% of usable acute rows lie outside the 24-96 h window the current
   feature code can even represent, out to 26,298 h, and 11,191 (chemical, species) pairs carry two
   or more durations. [MEASURED HERE] This is the one regime where a piecewise-constant tree is
   structurally beaten rather than rhetorically beaten, and where a mechanistic time law earns its
   keep. It is also cheap.
5. **Add targets, not parameters.** Chronic endpoints (225,081 usable rows), the 11,719 censored
   acute rows the pipeline currently discards, and the acute-to-chronic ratio identifiable on 10,334
   (chemical, species) pairs. [MEASURED HERE] Note the correction: the circulating figure of
   "~21.5k censored acute rows" is **wrong by ~1.8x** — the true usable count is 11,719.
6. **Rank the architectures by what they add over a same-information GBM, and accept that the
   highest-expected-gain item has no novelty at all.** Concatenated foundation-model embeddings
   (~1-2 GPU-h, one-time cache) is the best expected-gain-per-GPU-hour item on the board and is a
   pure engineering fix. The novel architectures are ranked below it on expected gain and above it
   on what they would contribute to a paper.

**Honest bottom line on the user's framing.** "These architectures may not be optimal" is right, but
not in the way it sounds: the problem is not that the models are too simple, it is that the *task*
as currently posed is one a GBM is nearly optimal for — a tabular regression on ~160 fixed
descriptor columns with a 4-level duration code. Make the task harder along the axes where the data
is rich and the GBM is structurally weak (duration, endpoint ordering, censoring, novel chemistry),
and a learned architecture has somewhere to win. Add capacity without changing the task and it will
not.

---

## 1. Diagnosis: why simple ML wins or ties

### 1.1 What the scored blocks actually say

| Block | Status | Candidate | Strongest published | Strongest simple baseline | Verdict |
|---|---|---|---|---|---|
| ADORE t-F2F occurrence test (2026-10-24) | spent | chemprop 0.873 | published 0.927 inside every CI | XGB+MACCS 0.917 | competitive |
| Prospective `fish_novel` (2026-11-09, 1,912 rows / 301 chemicals) | spent | NNLS stack **1.208** [1.003, 1.470] | `rf_published_mwlogp` **1.237** (d -0.029 [-0.095, +0.039]) | re-tuned XGBoost **1.218** (d -0.010) | competitive, no Holm rejection |
| T2 unseen taxa (insects / molluscs / amphibians) | **frozen, unspent** | selection in flight | — | — | see 1.2 |

Source: `RESULTS.md` lines 29-30, 88; `docs/handoff/chemtox.md`. The masked-CV ensemble gain
(-0.062 [-0.096, -0.031], CI excluding 0) shrank to -0.029 and lost significance when it met new
chemicals. That halving is the whole diagnosis in one number: CV folds share chemical space, so
ensembling five models built on one representation buys variance reduction that CV rewards and bias
reduction that a novel-chemical block demands and does not get.

### 1.2 The decisive new evidence: information beats architecture, measured on the live track

T2's stage-2 selection has 10 of 27 entries complete (validation only; no test block read —
`registry_rows()` 0/0/0, no `AUTHORISED.json`, no FREEZE file). Best mean RMSE over the four
leave-one-class-out folds, read out of
`$SENTINEL_RUNS/chemtox/unseen_taxa/stage2_select/summary_index.json` at 2026-11-10 11:37
[MEASURED HERE]:

| Entry | Role | Best mean RMSE | Channels chosen |
|---|---|---|---|
| `xgboost_chan` | **comparator** | **1.0977** | higher_rank |
| `lightgbm_resid_surrogate` | candidate | 1.1016 | surrogate + higher_rank |
| `lightgbm` | candidate | 1.1053 | higher_rank |
| `lightgbm_resid_ice` | candidate | 1.1055 | surrogate + higher_rank |
| `catboost` | candidate | 1.1107 | higher_rank |
| `rf_chan` | comparator | 1.1111 | surrogate + higher_rank |
| `xgboost_published_chan` | comparator (published, fixed) | 1.1130 | higher_rank |
| `rf_published_chan` | comparator (published, fixed) | 1.1520 | surrogate + higher_rank |
| `rf` | comparator (channels pinned off) | 1.2146 | none |
| `xgboost` | comparator (channels pinned off) | 1.2209 | none |
| `chemprop` | candidate | **1.1679** best of 19/32 configurations with all four folds scored | (in flight) |

Two readings, both uncomfortable and both useful:

* **A channel-matched comparator is leading.** Claim condition (d2) requires a candidate to beat
  *each* channel-matched comparator. Right now none does. If the remaining entries
  (`chemeleon_lightgbm`, `gpboost`, `readacross`, `gpboost_published_chan`, the ensembles) do not
  change this, T2's honest outcome is "no candidate beats the published methods given the same
  information" — which is exactly the outcome protocol §9 pre-registered as likely.
* **Chemprop's ADORE-unseen advantage does not transfer to a class-level hold-out.** On the spent
  prospective block chemprop was the best model on unseen chemicals (1.202 vs stack 1.243,
  reference 1.287). On an unseen *class* its best complete configuration is 0.07 behind a
  channel-matched XGBoost. A graph representation decorrelates errors across *chemistry*; it carries
  no information across *taxonomy*, and T2 forbids every input that would.

Now the comparison that matters for architecture choice. From the project's own A5 diagnostic (one
`rf` configuration, identical parameters, only the channel varying), validation RMSE moved
1.3412 (no channel) -> 1.2286 (surrogate) -> 1.2502 (higher_rank) -> 1.2183 (both): **-0.123 from
information alone, on a fixed, simple model.** Across the whole completed model zoo, from
`rf_published_chan` to `xgboost_chan`, the spread is 0.054, and from the best tree to chemprop it is
0.070. **Information is worth roughly twice architecture here.** Any plan that spends GPU-hours on
architecture before spending CPU-hours on information is mis-ranked.

### 1.3 There is a lot of headroom; it is just not where capacity lives

[MEASURED HERE] On the acute endpoint with the three T2 classes excluded (the T1 universe):

* 15,398 exact-replicate groups (same CAS, species, duration, endpoint; >= 2 records) covering
  65,664 rows give a pooled within-group SD — the replicate floor — of **0.511** log units.
* The overall target SD is **1.768**.
* Current models score 1.10-1.22 on cross-class hold-outs and 1.21-1.24 on a cross-chemical block.

So the irreducible noise is ~0.51 and the models sit at ~1.1-1.2. Between a third and a half of the
achievable error reduction is still on the table. The task is nowhere near saturated. But the
remaining error is *bias* on chemistry and taxa the model has never seen, and bias is reduced by
better representations and more diverse data, not by more trees.

### 1.4 Why the GBM keeps up — and two claims about GBMs that are false

The GBM keeps up for a reason that is structural, not accidental: the task as posed *is* a tabular
regression. ~160 fixed columns (37 published descriptors + 167 MACCS bits, after `drop_first`), one
target, one endpoint, and a duration scalar linearly scaled over four levels
(`features.py` `DURATION_LEVELS=(24,48,72,96)`). On that problem a tuned GBM is close to the
Bayes-optimal *estimator class*, and a neural network's advantages — weight sharing, composition,
smoothness, transfer — have almost nothing to attach to.

Two arguments that appear in the current design discussion are, however, **wrong**, and I checked
them in the live environment rather than arguing about them [MEASURED HERE]:

| Claim in circulation | Reality in `sentinel2` (lightgbm 4.7.0, xgboost 3.2.0) |
|---|---|
| "A tree cannot guarantee monotonicity in exposure duration." | **False.** `LGBMRegressor(monotone_constraints=[...])` fits and enforces it; `XGBRegressor(monotone_constraints=(...))` likewise. Both executed successfully here. A monotone-in-duration GBM on continuous `duration_h` is a legitimate, cheap comparator and *must* be one. |
| "Squared-error GBMs must drop censored rows, so the Tobit likelihood is a neural-only advantage." | **False.** `xgboost.train({'objective': 'survival:aft', ...})` with `label_lower_bound` / `label_upper_bound` fits interval-censored targets; executed successfully here. The censored-data comparator is XGBoost AFT, not "the GBM cannot". |

What a tree genuinely cannot do, and what therefore remains as real inductive-bias headroom:

1. **Share parameters across endpoints.** A multitask GBM grows separate trees per target; a label
   on one endpoint can share *features* with another endpoint's label, never a *parameter*. This is
   the one lever the published competition has already proven works (TRIDENT; Wu et al.; MarineTox).
2. **Extrapolate a functional form past the last split.** Beyond its largest duration split a tree
   predicts a constant. A monotone constraint fixes the *direction* of that extrapolation, not its
   *shape*. On 15.5% of rows (durations outside 24-96 h, out to 26,298 h) the shape matters.
3. **Learn a representation of the molecule.** MACCS is a fixed 166-bit substructure key: on a
   scaffold it has no key for, every member of the current stack is wrong in the same direction.
4. **Emit a calibrated joint prediction** (e.g. an acute-to-chronic ratio with an interval) from one
   shared latent. A GBM can be clipped post hoc; it cannot report a coherent derived quantity.

Those four — not "more capacity" — are the whole design space worth spending GPU-hours on.

---
## 2. Literature: what the state of the art is, with verified citations

### 2.1 Verified in this session (metadata retrieved today; URL given)

| # | Work | Metadata verified today | What it is | Why it matters to M4 |
|---|---|---|---|---|
| L1 | **TRIDENT** — Gustavsson M, Kall S, Svedberg P, Inda-Diaz JS, Molander S, Coria J, Backhaus T, Kristiansson E. "Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms." *Science Advances* 10(10):eadk6669, 2024-03-06. | [VERIFIED HERE] web search; PubMed 38446886, PMC10917336, Chalmers record 542229 | Pretrained SMILES transformer (ChemBERTa-family) encoder + DNN head conditioned on **log duration** and endpoint/effect one-hots, trained jointly on EC50 and EC10 for fish, invertebrates and algae. Code + weights + web service released. | **The reference model for this exact task**, and the published demonstration that joint multi-endpoint training improves accuracy. Its duration conditioning is the thing any mechanistic time law must beat. |
| L2 | **Context-aware multitask from fragmented ECOTOX** — Liang Y, Cui B, Li F, Zhou D. *Environmental Science & Technology*, published 2026-11-02. DOI 10.1021/acs.est.6c12447 | [VERIFIED HERE] `api.crossref.org/works/10.1021/acs.est.6c12447` — title, 4 authors, journal, date all confirmed | Links EPA ECOTOX records into model-ready aquatic + soil datasets; multitask model with endpoint-specific heads over shared chemical / biological / experimental "context" representations. | Published **eight days ago** and it is lever L1+L3 of M4's own SOTA plan. It must now be in the comparator set and in the novelty check. Its reported pooled aquatic R2 ~0.75 and its finding that multitask gains shrink as structural separation grows are **[UNVERIFIED]** (paywalled; I verified only the metadata). |
| L3 | **Unified multitask MMoE** — Wu X, Hu B, Chen D, Yang J, He X. "Unified Multitask Model for Aquatic Toxicity Prediction across Organism Groups and Effect End Points: A Comparison of Molecular Representations." *J. Chem. Inf. Model.* 66(20):13302-13312, 2026. DOI 10.1021/acs.jcim.6c02487 | [VERIFIED HERE] Crossref — title, 5 authors, volume/issue/pages confirmed | Multi-gate Mixture-of-Experts over 6 fish/crustacean/algae EC50+EC10 tasks from a fused representation, with an EC50 >= EC10 ordering constraint and heteroscedastic uncertainty. | **The "obvious next architecture" for M4 is already published.** Any multi-endpoint ordered-head design must cite it as prior art and differentiate. Its claim to beat TRIDENT on all 6 tasks is **[UNVERIFIED]** (paywalled). |
| L4 | **MarineTox Predictor** — Zhu Y, Zhang M, Han P, Zhu S, Chen J, Li X. *Environ. Sci. Technol.* 60(21):14813-14825, 2026. DOI 10.1021/acs.est.5c15496 | [VERIFIED HERE] Crossref — title, 6 authors, volume/issue/pages confirmed | Multitask deep learning sharing substructure features from data-rich freshwater endpoints to 31 low-resource saltwater tasks over 26 marine organisms. | The strongest published evidence that **multitask transfer is the right fix for data-poor taxa** — i.e. for T2's problem. Its reported +140% accuracy gains are **[UNVERIFIED]**. |
| L5 | **3DMol-Tox / tox-learn** — Yuan R, Shaw J, Tang H, Ye Y. "Cross-chemical and cross-species toxicity prediction: benchmarking and a novel 3D-structure-based deep learning model." bioRxiv 2025.11.24.690199, posted 2025-11-26. DOI 10.1101/2025.11.24.690199 | [VERIFIED HERE] Crossref, **including the abstract**: 50,603 records / 5,889 compounds / 2,285 species from 7 sources; `tox-learn` released; 3DMol-Tox "matches the best 2D model (GPBoost) on regression accuracy" and is more consistent on within-one-bin classification; structure-aware splitting recommended because leakage is "a common cause of inflated reported performance" | A modern, leakage-controlled benchmark for M4's exact task. | **The single most important negative result for architecture choice:** a 3D geometric deep model only *matches* GPBoost on regression. Pure representation-architecture headroom in this space is thin. It also independently validates M4's split hygiene concern. |
| L6 | **Bio-QSARs 2.0** — Zubrod JP, Galic N, Vaugeois M, Dreier DA. "Bio-QSARs 2.0: Unlocking a new level of predictive power for machine learning-based ecotoxicity predictions by exploiting chemical and biological information." *Environment International* 186:108607, 2024. DOI 10.1016/j.envint.2024.108607 | [VERIFIED HERE] Crossref — title (the previously-recorded wrong title is corrected here), 4 authors, volume + article number confirmed | GPBoost: boosted trees + grouped random effects over chemical and biological descriptors, for cross-chemical and cross-species prediction. | M4's **mandatory published cross-species comparator**, with hyper-parameters taken verbatim from the authors' released models. Note the plan's standing disclosure: on T2 it is an **ADAPTED** estimator (grouping `rank_phylum` only, gpboost 1.8.0 vs 1.2.1, the authors' species one-hots absent) and all three adaptations weaken it, i.e. bias in our favour. |
| L7 | **ADORE** — Gasser L, Schur C, Perez-Cruz F, Schirmer K, Baity-Jesi M. "Machine learning-based prediction of fish acute mortality: implementation, interpretation, and regulatory relevance." *Environ. Sci.: Adv.* 3(8):1124-1138, 2024. DOI 10.1039/d4va00072b | [VERIFIED HERE] Crossref — title, 5 authors, volume/issue/pages confirmed | The benchmark and the RF/XGBoost reference M4 reproduces. | The **simple-baseline bar**. Explicitly a within-taxa fish benchmark, so it is not "the strongest published method" for cross-taxa or cross-scaffold work. |
| L8 | **Chemprop** — Heid E, Greenman KP, Chung Y, Li S-C, Graff DE, Vermeire FH, Wu H, Green WH, McGill CJ. "Chemprop: A Machine Learning Package for Chemical Property Prediction." *J. Chem. Inf. Model.* 64(1):9-17, 2024. DOI 10.1021/acs.jcim.3c01250 | [VERIFIED HERE] Crossref — title, 9 authors, volume/issue/pages confirmed | The D-MPNN package M4 already uses. | Correct citation for the candidate that is currently M4's only non-tabular representation. |
| L9 | **MolFormer** — Ross J, Belgodere B, Chenthamarakshan V, Padhi I, Mroueh Y, Das P. "Large-scale chemical language representations capture molecular structure and properties." *Nature Machine Intelligence* 4(12):1256-1264, 2022. DOI 10.1038/s42256-022-00580-7 | [VERIFIED HERE] Crossref — title, 6 authors, volume/issue/pages confirmed | A SMILES transformer pretrained on ~1.1B molecules (count **[UNVERIFIED]**), used as a frozen embedding. | The concrete published option for lever L2 (concatenated foundation-model embeddings), the highest expected-gain-per-GPU-hour item in this plan. |
| L10 | **GUTS** — Jager T, Albert C, Preuss TG, Ashauer R. "General Unified Threshold Model of Survival - a Toxicokinetic-Toxicodynamic Framework for Ecotoxicology." *Environ. Sci. Technol.* 45(7):2529-2540, 2011. DOI 10.1021/es103092a | [VERIFIED HERE] Crossref — title, 4 authors, volume/issue/pages confirmed | The standard TKTD survival framework: internal damage dynamics plus a threshold, from which time-dependent LC50 follows. | The correct citation for any mechanistic time-law head. It also means a mechanistic head is **not novel as mechanism** — the novelty would be in making it differentiable and training it jointly on heterogeneous endpoint families. |
| L11 | **EnviroTox** — Connors KA, Beasley A, Barron MG, Belanger SE, Bonnell M, Brill JL, de Zwart D, Kienzler A, Krailler J, Otter R, Phillips JL, Embry MR. "Creation of a Curated Aquatic Toxicology Database: EnviroTox." *Environ. Toxicol. Chem.* 38(5):1062-1073, 2019. DOI 10.1002/etc.4382 | [VERIFIED HERE] Crossref — title, 12 authors, volume/issue/pages confirmed | A curated aquatic toxicity database built to a published quality rubric. | The candidate **cross-source** test block for T5 (a different curation pipeline is the strongest generalisation evidence available). Licence terms and row counts **[UNVERIFIED]** — the landing page was not fetched today. |

### 2.2 Carried over but NOT verified today

Listed so they are not silently relied upon. Each must be verified before it enters a claim, a
model card or the paper.

* Forastiere et al., *Ecotoxicol. Environ. Saf.* 314:120057 (2026) — continuous genetic-similarity
  species embedding. **[UNVERIFIED]** Closest prior art for any taxon encoder.
* Yang et al., *ES&T* 60(3):2556-2565 (2026) — species embeddings from taxonomy + cytochrome-*b*.
  **[UNVERIFIED]** Already deferred by lead decision to its own pre-registration.
* Yuan et al., *Environ. Toxicol. Chem.* 45(9) (2026) — the journal version of L5. **[UNVERIFIED]**
  (the bioRxiv preprint L5 *is* verified).
* EPA Web-ICE v4.0 model/database downloads and their licence status. **[UNVERIFIED]** This matters:
  T2's `ice_regression` is a locally re-fitted Web-ICE-*form* surrogate (17,081 pool pairs,
  a = -0.650, b = 0.982), not the published coefficients. Downloading the published ICE models would
  upgrade a mandatory comparator from "adapted" to "the actual published method".
* TRIDENT's Zenodo deposit (DOI 10.5281/zenodo.10260621), its CC BY 4.0 licence, its ~147k rows and
  especially `03_Model_error_per_SMILES_from_10x10_CV.xlsx`. **[UNVERIFIED]** — I did not fetch the
  record today. If the per-SMILES error file is what it is described to be, it is the **only clean
  route to a TRIDENT head-to-head**, because running TRIDENT on our rows measures memorisation.
* Human-HTTK parameter-QSAR work (Dawson et al. 2021; Pradeep et al. 2020), the Lin & Chou 2022
  review, critical-body-residue literature (McCarty & Mackay), log BCF ~ log Kow regressions
  (Veith; Arnot & Gobas), Haber's rule, allometric elimination scaling. **All [UNVERIFIED].** No
  number from any of them is used anywhere in this plan — every anchor in the designs below is
  learned from our own training data for exactly this reason.
* "Tree-based models still outperform deep learning on tabular data" (Grinsztajn et al. 2022).
  **[UNVERIFIED]** — I could not verify it today and it is the best single citation for §1.4's
  structural argument. Worth one search when the budget refills.

### 2.3 What the literature implies for M4, in three sentences

1. **The multitask lever is proven and is being actively mined by three 2026 groups** (L2, L3, L4).
   It will improve M4's numbers, and it will not make M4 novel. Treat it as a mandatory comparator
   *and* an ensemble member, never as the novelty claim.
2. **The representation-architecture lever is nearly tapped out** (L5: a 3D geometric deep model
   only matches GPBoost). Do not spend GPU-hours on a fancier encoder expecting a win.
3. **The open space is the evaluation and the composition**: leakage-controlled harder tracks (L5's
   own recommendation), exposure-duration generalisation (nobody in L1-L5 evaluates it as a
   hold-out), and coherent derived quantities with intervals. That is where M4 can be first rather
   than fourth.

---

## 3. Is the current architecture optimal? Honest answer

**For the task as currently posed: yes, very nearly — and that is the problem.**

The stack is five members (`catboost`, `chemeleon_gpboost`, `chemprop`, `chemeleon_lightgbm`,
`gpboost_multitaxa`) over the same ~160-column block, every family's winner using
`chemical=mw_logp`. On masked CV four families beat the 0.9733 XGBoost bar. On a novel-chemical
block the ensemble's edge halves and loses significance. On an unseen class the best thing in the
zoo so far is a channel-matched comparator. That is the behaviour of an estimator class that has
already extracted what the representation contains.

Where the current architecture is genuinely sub-optimal, ranked by how much it costs:

| # | Defect | Evidence | Cost |
|---|---|---|---|
| 1 | **One representation across all members.** MACCS + 37 descriptors for everything. | CV gain -0.062 -> prospective -0.029; `docs/handoff/chemtox.md` §1 cause 1 | Large. It is why ensembling buys variance and not bias. |
| 2 | **Duration is a 4-level linear code.** `features.py` `DURATION_LEVELS=(24,48,72,96)`. | 15.5% of usable acute rows fall outside that window, out to 26,298 h; 444 distinct durations exist [MEASURED HERE] | Large and entirely self-inflicted. The model cannot represent the axis along which the data is richest after chemistry. |
| 3 | **One endpoint.** 225,081 usable chronic rows are unused; 11,719 censored acute rows are dropped (`aux.py:211`, `prospective.py:231`, `unseen_taxa.py:88`). [MEASURED HERE] | Large, and it is the lever the published competition is using. |
| 4 | **Embeddings were substituted, not concatenated.** `embeddings v1` scored 1.1579 on CV and was dropped — but it replaced the published block, so the result shows that discarding exposure covariates fails, not that embeddings fail. | `docs/handoff/chemtox.md` §1 cause 2 | Moderate; cheap to fix. |
| 5 | **Ensemble members chosen by score, not by error decorrelation.** | Five near-duplicate representations | Moderate; free to fix once (1) and (4) give it something to decorrelate. |
| 6 | **No uncertainty output.** Both L1 and L3 report uncertainty; M4 reports point predictions. | — | Moderate for the paper, small for RMSE. |

What is **not** wrong: the tabular models themselves, the tuning parity discipline, the split
hygiene, or the guard machinery. Those are the strongest parts of this workstream and nothing below
proposes touching them.

---
## 4. Ranked architecture proposals

### Scoring rubric (stated so the ranking can be argued with)

`score = 5.0 x P(this contributes to a RESEARCH_PLAN §15.1 label within ~4 weeks) + 3.0 x novelty-for-the-paper + 2.0 x (cheapness x robustness)`, each factor in [0, 1], total out of 10.
Novelty is scored against §2, i.e. against what L1-L11 already publish — not against what M4 has
previously built.

All five proposals live in **new modules only**. Nothing under `docs/preregistration/` is edited;
`sentinel/models/chemtox/{__init__,evaluate,features,tabpfn_model,gpboost_model}.py` are read-only
(M5's frozen pass hashes them, several are 0444), and T2's pinned `unseen_taxa.py` /
`unseen_taxa_guard.py` / plan are untouched. Every proposal below targets **T1 or T6**, never T2 —
T2 is frozen and must finish as pre-registered.

### 4.0 Mandatory references, built FIRST and never counted as candidates

These are not architecture proposals. They are the comparators without which any proposal below is
unreviewable, and two of them exist because §1.4 showed the GBM-impossibility arguments were wrong.

| Ref | What | Cost | Why it is mandatory |
|---|---|---|---|
| **R1** | **Monotone-duration GBM**: LightGBM / XGBoost with `monotone_constraints` on continuous `duration_h`, plus a log-duration spline basis. | CPU, hours | Removes the "trees cannot be monotone in time" argument. Verified to fit in `sentinel2` [MEASURED HERE]. |
| **R2** | **Interval-censored AFT GBM**: `xgboost` `survival:aft` with `label_lower_bound` / `label_upper_bound` on the 11,719 censored acute rows. | CPU, hours | Removes the "GBMs must drop censored rows" argument. Verified to fit [MEASURED HERE]. |
| **R3** | **Multitask-by-stacking GBM**: one GBM over acute + chronic rows with an endpoint indicator and per-endpoint offsets. | CPU, hours | Comparator parity for the "more targets" lever (`sota_campaign.md` rule 4). If a GBM given the chronic rows closes the gap, no network is needed. |
| **R4** | **Every comparator gets the concatenated embedding.** The frozen foundation-model block is an *input*, so parity requires it for R1-R3, `rf`, `xgboost`, and the published configurations. | the same one-time cache | The C-M4P-1 lesson: a candidate that wins only because it alone received a channel does not qualify. |
| **R5** | **TRIDENT head-to-head via its released per-SMILES CV errors**, never by re-running it. | CPU, <1 day | TRIDENT's training data overlaps our rows, so a re-run measures memorisation. This is the only admissible route, and it closes the standing "TRIDENT deliberately not run" gap in the novelty checks. Depends on verifying the Zenodo deposit **[UNVERIFIED]**. |

R1-R3 are also interesting results in their own right: if a monotone AFT multitask GBM closes most
of the gap on T6, that is a publishable negative and it costs no GPU time.

### 4.1 Rank 1 — **ChemTox-TED**: time-and-endpoint-decomposed multitask network. Score **7.0/10**

*(merges the mechanistic-head proposal with L1's duration conditioning and L3's ordering
constraint, then deletes everything that rested on an unverifiable constant or an unmeasured join)*

**One line.** A graph+descriptor encoder that does not predict log LC50 directly, but emits the
parameters of a monotone saturating time law and a non-negative acute-to-chronic ratio, composes
them to get the prediction at the row's actual duration and endpoint, and trains jointly on acute +
chronic + censored rows.

**Architecture.**

* *Chemical encoder*: own-code D-MPNN (hidden 256, 3 message-passing steps, mean readout) so
  gradients reach the composition layer — not the `chemprop` package wrapper, which stays an
  untouched comparator. Concatenated with the 37 published descriptors + 167 MACCS bits +
  `chem_rdkit_clogp` + `chem_mw` **and** the frozen foundation-model embedding of §4.2, through a
  small MLP, to `z_c` (256-d).
* *Condition encoder*: embeddings for `water_type`, `media_type_norm`, `exposure_type_norm`,
  `test_location_norm`, `organism_lifestage_norm`; z-scored `media_ph`, `media_temperature`,
  `media_hardness`, `dissolved_oxygen`, `media_salinity`, each with a missingness indicator.
* *Organism encoder*: on T1/T6 the full lineage is in-distribution, so a nested-rank gated embedding
  sum is admissible (see §4.5 for why this is gated and why it cannot go near T2).
* **The composition layer (the only part that is a design claim):**
  * `log10 k = (a0 + a1 logKow + a2 log10 MW) + B_k tanh(u_k)`, with `a0..a3` **initialised by
    least squares on the training duration series only** and `B_k` a declared bound in decades.
  * `log10 LC50(t) = log10 LC50_inf - log10(1 - exp(-k t))`, implemented as
    `-log10(-expm1(-kt))` with `kt` clamped, so `d/dt <= 0` holds analytically and a unit test
    asserts it over a random parameter sweep and a duration grid from 1 h to 26,298 h.
  * `log10 EC10 = log10 EC50_inf - softplus(u_acr) - log10(1 - exp(-k_chr t))`, so **EC10 <= EC50
    holds by construction** and `log10 ACR = softplus(u_acr)` is itself a reported output with an
    interval. Within the chronic family a cumulative-softplus chain enforces NOEC <= MATC <= LOEC.
  * **Duration and endpoint identity never enter the trunk.** They act only through this layer.
    That is the restriction a GBM cannot be given, and ablation A3 below adds them back as a leak
    test.
* *Heads/losses*: heteroscedastic Gaussian NLL for acute and chronic; a censored-Gaussian (Tobit)
  term for the censored acute rows; weights pre-declared and swept over two settings.

**What was deleted from the original proposal, and why.** These are not optional trims; each was
either resting on a number I could not verify or on a number that is wrong.

| Deleted / demoted | Reason |
|---|---|
| **The BCF-supervised bioconcentration latent, as a core component** | The usable row count cannot be confirmed from here. Raw `results.parquet` holds 27,725 rows whose endpoint code starts BCF/BCFD/BAF (BCF 20,422, BCFD 4,314, BAF 1,629, plus 1,360 qualified codes) [MEASURED HERE], but the "21,352 usable after CAS normalisation" figure depends on a digit-stripping join, a unit/basis reconciliation (BCFD is dry-weight) and a trim of an implausible log10 range, none of which I re-derived. **Demoted to ablation A5**, admissible only after a measured join-and-QC report. |
| **The Q10 temperature term with a prior centred on log 2** | The prior's centre is an unverified literature constant. A learned Q10 with a free prior is a confound, not a physical constraint. **Demoted to ablation A6.** |
| **The critical-body-residue variance penalty over "baseline-narcotic" rows** | Requires a narcosis mode-of-action flag that is not in the frame, and the band it encodes is **[UNVERIFIED]**. **Dropped.** |
| **The claim that monotonicity in t is a neural-only property** | False; see §1.4. R1 is now a mandatory comparator. |
| **The claim that censored rows are a neural-only advantage** | False; see §1.4. R2 is now a mandatory comparator. |
| **"~21,527 censored acute rows"** | Wrong. The true count is **13,426 censored acute rows, 11,719 of them usable** (SMILES present, molar concentration present) [MEASURED HERE]. The design's row budget must be restated at 11,719. |

**Inductive bias it has that a same-information GBM does not** (after R1-R3 exist, this list is
short and that is the point):

1. **A shape, not just a direction, in time.** R1 can be forced monotone; it still predicts
   piecewise-constant values and, beyond its last split, a constant. TED predicts at a duration
   never observed for that chemical because the *shape* is shared and only the rate is learned.
   This is the single testable advantage and T6 (§7.2) is the track built to test it.
2. **Parameter sharing across endpoint families.** A chronic row's gradient moves `u_acr` and `k`,
   which then multiply into every acute prediction for that chemical. R3 shares features only.
3. **A coherent derived quantity with an interval.** `log10 ACR` is identifiable on 10,334
   (chemical, species) pairs that carry both acute and chronic data [MEASURED HERE] and is
   predictable with a CI for pairs that carry only acute data. No GBM can emit this.
4. **Controlled extrapolation.** Every parameter is (anchored law) + bounded residual, so outside
   the data support the residual saturates onto a 3-parameter law rather than onto the nearest
   leaf's mean. `B = 0` is a pure mechanistic QSAR and `B` large is an unconstrained net, so the
   ablation sweep measures exactly how much the mechanism is worth.

**Novelty, honestly.** The mechanism is GUTS-standard (L10, verified) and is therefore **not new as
mechanism**. L1 conditions on log duration; L3 constrains EC50 >= EC10 in output space with
heteroscedastic uncertainty. What is not in L1-L5, as far as I can tell from §2: *composing*
duration through a differentiable law instead of conditioning on it, deriving the ordering from one
latent plus a non-negative ACR that is itself reported with an interval, and evaluating any of this
as a **duration hold-out**. The defensible contribution is therefore "mechanism-composed time and
endpoint structure, evaluated on a pre-registered duration-extrapolation block", which is a
narrower and more honest claim than "new architecture beats SOTA".

**Cost.** GPU-hours: I decline to repeat the original 37 GPU-h estimate as a plan figure, because
this workstream's last per-configuration estimate was wrong by 9x (10 min estimated vs 91 min
measured for chemprop). Pre-declared budget instead: **16 configurations x 3 selection folds = 48
jobs**, sized from the *slowest* of the 16 sampled draws after a timed smoke, with a hard
pre-declared fallback to 8 configurations if the smoke exceeds 30 min/job. At 25-35 min/job that is
**20-28 GPU-h**, plus ~2 GPU-h for the 3-seed refit: **budget 22-30 GPU-h, cap 32.** VRAM: measure
with `torch.cuda.max_memory_allocated()` in the smoke; budget **<= 4 GB** of the 11 GB so a second
workstream can share the card. All jobs are one configuration x one fold, <= 30 min, reusing the
`--fold-merge` pattern already proven for chemprop on T2. fp16 AMP with the **composition layer
force-cast to fp32 inside an autocast-disabled block** (fp16 `expm1` near zero loses all precision —
this is the single most important implementation detail and a unit test must assert parity against
an fp64 reference).

**Fair test plan.** Primary track T6 (duration extrapolation), secondary T1 (unseen chemical
clusters). Comparators at full parity: R1, R2, R3, R4-adjusted `rf` / `xgboost`, the published ADORE
configurations, `gpboost_published` (with its three disclosed adaptations), `chemprop` single-task,
and R5's TRIDENT per-SMILES errors on the intersecting SMILES. Holm within the block, 3 seeds,
`all_seeds_better` reported, one look at the sealed block, MDE computed before any fit, and the four
scorer-side gaps of `sota_campaign.md` §4 built in (lead-written FREEZE as the only unlock,
`BaseException` abort, full import-closure code hash, all artifacts verified before the claim).
**Required ablation set, pre-declared, because without it a win cannot be attributed:**

| Ablation | What it isolates |
|---|---|
| A1 | D-MPNN trunk, free-form duration conditioning (TRIDENT-shaped), no composition layer — *is the mechanism doing anything?* |
| A2 | Composition layer with `B = 0` (pure anchored law, no residual) — *the physics-only floor* |
| A3 | Duration and endpoint fed to the trunk as well (leak test) — *was the restriction load-bearing?* |
| A4 | Acute-only training (no chronic, no censored) — *the multitask contribution* |
| A5 | + BCF head (only if the join/QC report lands) |
| A6 | + Q10 term |

### 4.2 Rank 2 — **ChemTox-RX**: concatenated foundation-model representation + decorrelation-selected ensemble. Score **6.5/10**

**One line.** Embed every unique structure once with a pretrained chemical language model, *append*
the embedding to the published block (never substitute it), give the same block to every comparator,
and then select ensemble members to minimise error correlation rather than to minimise individual
score.

**Why it is ranked second despite having no novelty whatsoever.** It is the highest
expected-gain-per-GPU-hour item on the board by a wide margin: ~7,000 unique structures embedded
once is **1-2 GPU-h total** in <= 2 h chunks, cached to disk, and everything after that is CPU. It
directly attacks §3 defects 1 and 4, and it is the precondition for defect 5 — there is nothing to
decorrelate until a second representation exists. L9 (MolFormer, verified) is the concrete option;
ChemBERTa-2 and Uni-Mol are alternatives whose metadata is **[UNVERIFIED]** today.

**Required changes from the critique.**

* **Concatenate, never substitute.** `embeddings v1` scored 1.1579 and was dropped because it
  *replaced* the exposure/duration covariates. That result says nothing about embeddings.
* **Parity is not optional** (R4): the embedding block goes to `rf`, `xgboost`, the published
  configurations and R1-R3 on identical rows, or the result is another `crust_novel`.
* **Decorrelation must be a pre-declared objective**, e.g. choose members greedily to maximise
  `mean pairwise (1 - corr(residuals))` subject to each member being within a declared margin of
  the best single model, with the member set fixed before the test look.
* **Disk**: the embedding cache plus expanded frames must stay inside M4's 10 GB cap, with `df -h`
  before every build and a hard stop below 60 GB free. Free space moved during this session from
  **86 GB (98% full) to 184 GB (95% full)** as another user freed space [MEASURED HERE] — which is
  exactly why the rule is "check `df` before each build" and not "we have room".

**Cost.** 1-2 GPU-h (cache) + ~20-30 CPU-hours. VRAM ~2-4 GB for inference in batches. **Score
justification:** high P(contributes to a label), zero novelty, very cheap and very robust.

### 4.3 Rank 3 — **ChemTox-TimeHead**: a learned monotone time law bolted onto a frozen tabular model. Score **6.0/10**

**One line.** Keep the existing GBM exactly as it is, have it predict the *incipient* (long-duration
asymptotic) log LC50 on the subset where long-duration data exists, and learn a small per-row rate
head (2-3 parameters, from descriptors + species + conditions) that maps that asymptote to any
requested duration through the same monotone saturating law as §4.1.

**Why this exists.** It is the minimal experiment that separates *mechanism* from *representation*.
If TimeHead captures most of TED's T6 advantage, then the expensive graph trunk is not what is
working and the honest paper says so. If it does not, TED's trunk has earned its GPU-hours. Either
way the project learns something for ~1 GPU-h.

**Cost.** The rate head is a <50k-parameter MLP: **<= 1 GPU-h total**, or CPU-only. VRAM <1 GB.
**Fair test:** same T6 block, same comparators (R1 is the direct rival), same Holm/seeds/single-look
discipline. **Score justification:** moderate P(label) on its own, low-moderate novelty (it is a
two-stage TK fit, which is the published human-HTTK pattern **[UNVERIFIED]**), but extremely cheap
and it de-risks Rank 1.

### 4.4 Rank 4 — **ChemTox-MMoE**: multi-endpoint mixture-of-experts with ordering constraint. Score **4.5/10**

**One line.** L3's published design (shared fused representation, per-task gates, EC50 >= EC10
ordering, heteroscedastic uncertainty), re-implemented over our endpoint families.

**Why it is here at all.** Because it is the *strongest published method* for multi-endpoint aquatic
toxicity and claim condition (a) requires beating one. Build it as a **comparator and an ensemble
member**, not as a novelty claim. It is prior art (L3, verified metadata), it will probably improve
pooled numbers, and presenting it as new would not survive a novelty check.

**Cost.** ~8-12 GPU-h at a 16-configuration cap (no mechanistic layer, so cheaper per step than
TED). VRAM ~3 GB. **Fair test:** as §4.1's shared plan. **Score justification:** moderate P(label
contribution) as the comparator that makes a claim meaningful, near-zero novelty, moderate cost.

### 4.5 Rank 5 — **Taxon-path encoder with nested rank shrinkage. Score 3.5/10, and gated**

**One line.** Per-rank embedding tables (kingdom -> species) summed with learned per-rank gates, so
an unseen species contributes only its genus/family/order vectors.

**Why it is ranked last and gated.** On T2 it is **structurally inadmissible**: the protocol forbids
every input at class rank or below, `TaxonBlindBuilder` deletes those columns, and
`assert_no_forbidden_columns` would fail the run. On T1/T6 the species are in-distribution, so a
taxon encoder is admissible but is then competing against a one-hot that trees already handle well.
Its nearest prior art (Forastiere et al.; Yang et al.) is **[UNVERIFIED]**. Build it only as a
component inside TED's organism encoder, with an ablation that turns it off, and do not make a
separate claim from it. Species embeddings from genetic sequence stay deferred to their own
pre-registration, per the standing lead decision.

### 4.6 Dropped, with reasons

| Dropped | Why |
|---|---|
| **3D / geometric deep models** (equivariant nets, conformer ensembles) | L5 (verified abstract): a 3D-structure deep model only **matches** GPBoost on regression. The headroom is not here, and the compute is large. |
| **A bigger ensemble of the current five members** | §1.2: the spread across the whole completed model zoo is 0.054-0.070 while one information channel is worth 0.123. Adding a sixth near-duplicate is the thing that already failed. |
| **Species-genetic embeddings** | Deferred by lead decision to a separate pre-registration; prior art **[UNVERIFIED]**; inadmissible on T2 anyway. |
| **Mixtures / formulations** | Genuinely open territory, but no data audit exists. Exploratory only, no claim promised, and not in this plan's budget. |
| **Any amendment to T2 to add an architecture** | The pins make it an amendment; review addition (iii) forbids a run straddling two plan hashes; the A7 restart already cost a full re-run. T2 finishes as pre-registered. |
| **Re-running TRIDENT on our rows** | Measures memorisation. R5 (released per-SMILES errors) is the only admissible route. |

### 4.7 The fair-test plan all five share

1. **Comparator parity is checked mechanically, not promised.** Every input or extra row a candidate
   receives is given to every mandatory comparator on identical rows, asserted in code, with the
   row-count and column-name diff logged per fold.
2. **Tuning parity**: no family exceeds the track's declared per-entry configuration cap, and all
   configurations are counted, including those a candidate discards.
3. **Claim gates** (RESEARCH_PLAN §15.1): paired CI excluding 0 against **both** the strongest
   published method and the strongest simple baseline; Holm within the block; >= 3 seeds with
   `all_seeds_better` reported; one look at an unseen block frozen beforehand; the difference must
   exceed the block's pre-computed MDE; the claim names its block and RMSE is never averaged across
   blocks.
4. **Attribution**: no claim is made for an architecture without its ablation row. A win with A1
   (mechanism off) within the MDE of the full model is reported as "the mechanism is not what is
   working".
5. **Single-look hardening**: the guard machinery of `unseen_taxa_guard` plus the four scorer-side
   gaps. Write-ups go to `*_test_results.md`, never into a frozen file.
6. **Negative results are reported with the same prominence as positive ones**, per
   `sota_campaign.md` rule 3.

---
## 5. New numeric predictive targets (multi-task)

All counts below are **[MEASURED HERE]** from
`$SENTINEL_DATA/processed/ecotox/ecotox_aquatic.parquet` (471,097 rows) and
`$SENTINEL_DATA/raw/ecotox/release_2026-09-15/results.parquet`, with the filter stated in each row.
"Usable" means: uncensored unless stated, molar concentration present (`log10_conc_mol_L`), and
`chem_smiles` present.

| ID | Target | Label source | Rows | Chemicals | Species | Status in M4 today |
|---|---|---|---|---|---|---|
| **H1** | log10 LC50/EC50 (mol/L) — **primary, unchanged** | `endpoint_norm in {LC50, EC50}` | **122,946** (120,740 with `duration_h`) | 4,592 | 2,550 | ADORE's filters keep 26,114 of these. 4.7x the rows, 2.4x the chemicals and 18x the species are unused. |
| **H2** | the same target, **interval-censored** | same, `censored = True` | **11,719** usable (13,426 censored in total) | — | — | **Discarded** by all three pipelines (`aux.py:211`, `prospective.py:231`, `unseen_taxa.py:88`). Note: the "~21.5k" figure in circulation is wrong by ~1.8x. |
| **H3** | log10 chronic effect concentration | `endpoint_norm in {NOEC, LOEC, EC10, MATC, NOEL, LOEL}` | **225,081** | 3,298 | 2,018 | Unused. This is the largest single block of labelled signal on disk. |
| **H4** | log10 BCF / BCFD / BAF (bioconcentration) | raw `results.parquet`, endpoint code prefix BCF/BCFD/BAF | **27,725 raw** (BCF 20,422 / BCFD 4,314 / BAF 1,629 / 1,360 qualified codes) | — | — | Unused, and **not yet usable**: needs a digit-stripping CAS join, a dry-weight-vs-wet-weight basis reconciliation and a plausibility trim, none of which I re-derived. Gated behind a measured join/QC report. |
| **H5** | log10 **acute-to-chronic ratio**, derived and reported with an interval | identifiable on (chemical, species) pairs carrying both an acute and a chronic record | **10,334 pairs** | — | — | Does not exist anywhere in M4. A regulatory quantity used to derive chronic PNECs when chronic data are absent. The most interesting *new* numeric output available. |
| **H6** | per-row predictive **sigma** / calibrated interval | no new labels; scored by interval coverage and NLL on the same rows as H1 | — | — | — | M4 reports point predictions only. L1 and L3 both report uncertainty, so this is table stakes for comparison, not a novelty. |
| **H7** | **within-one-bin severity class** (a reported metric, not a new label) | binned H1 | same as H1 | — | — | L5's verified abstract reports that its 3D model is *consistently better on within-one-bin classification* even while only matching GPBoost on regression. Reporting this metric costs nothing and may be where a real difference shows. |

**Which of these actually change the answer.** H3 is the big one, and it is the lever three 2026
papers are already pulling (L2, L3, L4) — so it will help and will not make us novel. H2 is nearly
free and is pure added signal (11,719 rows, and censoring is informative, not missing). H5 is the
one genuinely new *reported quantity*, and it is cheap because it falls out of H1+H3 under an
ordering constraint. H4 is the most interesting mechanistically and the least certain practically.

**Mandatory subtractions before any of this is built**, enforced by an assertion rather than a
comment: the three reserved T2 unseen-taxa blocks (15,726 rows) and the spent prospective rows must
be removed by `result_id`. The cleanest way to do this without touching a frozen file at all is the
class-level exclusion of §7.1 — it is a strict superset of the reserved ids and needs no read of
`test_blocks.parquet`.

---

## 6. More data: verified sources, licence and size

| Source | On disk? | Size | Licence | Verified | Value |
|---|---|---|---|---|---|
| **EPA ECOTOX full aquatic release 2026-09-15** — `$SENTINEL_DATA/raw/ecotox/release_2026-09-15/` (55 parquet tables, 133 MB; `results.parquet` 52.8 MB) and `processed/ecotox/ecotox_aquatic.parquet` (471,097 rows, 19.9 MB) | **yes** | ~153 MB total, **zero new download, zero against the 10 GB M4 cap** | US federal data; no explicit licence statement on the EPA page. Treat as public domain; "CC0" is **[UNVERIFIED]**. Already cleared and in use for the frozen T2 track. | file presence, sizes and all row counts **[MEASURED HERE]** | **The main unused lever.** Chemical, species, duration, endpoint and (for a substantial minority of freshwater acute rows) water chemistry are on the same row, so no cross-dataset join is needed. |
| **TRIDENT data repository** — Zenodo DOI 10.5281/zenodo.10260621; code `github.com/StyrbjornKall/TRIDENT` | no | ~36 MB for the two files that matter (`02_Preprocessed_complete_data.xlsx`, `03_Model_error_per_SMILES_from_10x10_CV.xlsx`); **not** the ~2.8 GB weights | stated CC BY 4.0 — **[UNVERIFIED]** today | the *paper* is [VERIFIED HERE]; the **deposit, its licence, its ~147k rows and the per-SMILES error file are [UNVERIFIED]** | If the per-SMILES error file is as described, it is the only admissible route to a TRIDENT head-to-head (R5). Verify the record, enter it in `docs/licenses.md`, then download. |
| **EnviroTox** — `envirotoxdatabase.org` | no | ~91k curated aquatic records claimed; actual file size **[UNVERIFIED]** | **[UNVERIFIED]** | the *data paper* is [VERIFIED HERE] (Connors et al. 2019, DOI 10.1002/etc.4382); the download terms are not | The candidate **cross-source** block for T5. A different curation pipeline is the strongest generalisation evidence available. Licence first. |
| **EPA Web-ICE v4.0 downloads** — `www3.epa.gov/webice/iceDownloads.html` | no | "a few tens of MB" — **[UNVERIFIED]** | US EPA, no explicit terms — **[UNVERIFIED]** | **[UNVERIFIED]** — not fetched today | Would upgrade T2's `ice_regression` from a locally re-fitted Web-ICE-*form* surrogate (17,081 pool pairs, a = -0.650, b = 0.982) to the actual published coefficients. **Note: this cannot be applied to T2** — its pool is pinned. It is a T1/T6 and future-track improvement. |
| **EFSA OpenFoodTox** | no | **[UNVERIFIED]** | **[UNVERIFIED]** | **[UNVERIFIED]** | Regulatory chemicals; second candidate for T5. |
| **REACH / ECHA registered-substance ecotox** | no | — | **licence-gated**: dissemination data viewable, bulk reuse restricted | — | **Enters only if `docs/licenses.md` clears it, else not at all** (standing lead ruling). The TRIDENT deposit is the clean redistributed route to REACH-derived values. |

**Measured QC caveats that must be reported, not assumed:**

* **46.2% of usable acute rows have no Murcko ring scaffold** (metals, simple organics and other
  acyclic chemicals) [MEASURED HERE]. Any scaffold-based split needs a separate rule for them. §7.1
  solves this with ECFP4 clustering.
* **The benzene scaffold alone covers 24,443 rows / 1,052 chemicals** — 20% of the usable acute
  endpoint in one "family" [MEASURED HERE]. A naive Bemis-Murcko hold-out is dominated by it.
* 22 of 4,592 SMILES fail to parse in RDKit [MEASURED HERE] — small, but they must be counted and
  excluded explicitly rather than silently.
* Molar-conversion disagreement with the EPA standardised pull on ~2.2% of compared rows (mostly
  ion-basis) — carried over from an earlier session, **[UNVERIFIED]** by me.
* Disk: **86 GB free (98% full) early in this session, 184 GB (95% full) at the end of it** — a
  shared filesystem whose free space is not under our control [MEASURED HERE]. The 60 GB floor and the
  10 GB M4 cap bind. Everything in §4 fits with room to spare, because the dominant new artefact is
  an embedding cache for ~7,000 structures.

---

## 7. Harder task tracks: practical justification, unseen block, power

Per `sota_campaign.md` rule 2, each track's practical justification is recorded **here, before any
validation result on it exists**. Choosing a track after seeing a win is forbidden, and these
justifications are deliberately written in terms of what a hazard model is *for*, not in terms of
what we expect to win.

### 7.1 T1 — unseen chemical clusters (recommended new primary)

**Practical justification.** A hazard-screening model's real failure mode is a chemical nobody has
tested: a new pesticide, a new pharmaceutical metabolite, a spill of something unusual. This is also
the one regime where M4 has already seen a learned representation beat the trees (chemprop 1.202 vs
reference 1.287 on ADORE-unseen chemicals). It is the track the diagnosis points at.

**Blocking unit — and why not Bemis-Murcko.** Murcko scaffolds fail twice on this data (§6): 46.2%
of rows are acyclic, and one scaffold holds 20% of rows. Instead, cluster the 4,154 parseable
chemicals in the T1 universe by **ECFP4 (Morgan radius 2, 2048 bits) Butina clustering at Tanimoto
distance cutoff 0.4**, which gives **2,738 clusters whose largest holds 7.2% of rows**
[MEASURED HERE]. Hold out whole clusters. This handles acyclic chemicals and the benzene family by
the same rule and gives a defensible resampling unit.

**Unseen test block (candidate sizes, measured).** Universe: usable acute rows with the three T2
classes (`Insecta`, `Gastropoda`, `Bivalvia`, `Polyplacophora`, `Cephalopoda`, `Amphibia`) excluded
outright — a strict superset of T2's reserved `result_id`s, so T2 stays untouched and no frozen file
is read: **98,727 rows / 4,156 chemicals / 1,586 species** [MEASURED HERE].

| Held-out fraction of rows | Clusters in block | Rows | Chemicals | Cluster-bootstrap MDE, rho=0.9 | MDE, rho=0.99 |
|---|---|---|---|---|---|
| 10% | 154 | 9,940 | 221 | **0.008** | 0.005 |
| 15% | 355 | 14,854 | 500 | **0.006** | 0.002 |
| 20% | 382 | 22,302 | 572 | **0.008** | 0.003 |

**Power.** Computed the same way T2's MDE was (two synthetic predictors at the error magnitude this
project actually produces, ~1.2 log units, correlated at rho, 2,000 cluster-bootstrap resamples,
half-width of the paired-difference interval) [MEASURED HERE]. Every option is **better powered than
any T2 block** (0.011 / 0.019 / 0.021). **Recommendation: the 15% block** — 355 clusters, 14,854
rows, 500 chemicals, MDE 0.006. The replicate floor on this universe is **0.511** (15,398 groups,
65,664 rows; target SD 1.768) [MEASURED HERE] and is reported as an absolute-RMSE bound, never as a
difference threshold (the W-D-7 lesson).

**What must additionally be pre-registered**: the reference-clustered co-primary CI (ECOTOX is
reference-concentrated and T2 measured 16-29% shared-publication rows), a `no shared reference`
stratum, the cluster assignment file with its hash, and the cutoff 0.4 itself — chosen here on
structural grounds (largest cluster < 10% of rows) and frozen before any fit.

### 7.2 T6 — exposure-duration extrapolation (recommended new secondary; this is the genuinely harder task)

**Practical justification.** Standard tests run 24-96 h, but the questions that matter after a spill
are "what happens over a week?" and "what does a 1-hour pulse do?". Regulatory extrapolation across
duration is done today with crude rules; a model that generalises across duration is directly useful
and is a different *scientific* claim from "lower RMSE on the same rows". It is also the one axis
where a tabular model is structurally, not rhetorically, beaten: beyond its last split a tree
predicts a constant.

**Unseen test block.** Train on rows with `duration_h` in [24, 96]; test on rows outside it.
**15.5% of the 120,740 usable acute rows with a duration lie outside that window** (so ~18,700 rows)
and the maximum observed duration is **26,298 h**; 444 distinct durations exist, and **11,191
(chemical, species) pairs carry two or more distinct durations, covering 85,196 rows**
[MEASURED HERE].

**Two variants, and the honest difference between them:**

* **T6a (easier, better powered): unseen durations, seen chemistry.** The chemical may appear in
  training at a different duration. This measures duration generalisation in isolation and is the
  cleanest test of the mechanism.
* **T6b (harder): unseen durations AND unseen chemical clusters.** The intersection of T1's held-out
  clusters and T6's held-out durations. Much smaller; power must be computed before it is adopted,
  and if it is underpowered it is reported as a stratum of T6a, never as a separate claim.

**Power.** Must be computed before any fit, clustering on (chemical cluster x duration band) to
avoid treating 85,196 rows from 11,191 pairs as independent. I did not compute T6's MDE here because
the block definition (which duration bands, long-only or both tails) is a lead decision and the MDE
depends on it. **This is a pre-registration deliverable, not an assumption.**

**Why this track is where a novel architecture has its best shot.** The comparator set (R1 monotone
GBM, R2 AFT GBM, L1's duration-conditioned transformer via R5) is strong and honest, and the
candidate's advantage is a *shape* claim that can be stated in advance, tested, and falsified.

### 7.3 T5 — cross-source transfer (third priority)

**Practical justification.** `sota_campaign.md` rule 1 names "a new source" as an allowed unseen
block, and surviving a different curation pipeline is the strongest generalisation evidence
available — stronger than a within-source split, because curation choices are a real confound.

**Unseen block.** Records in EnviroTox and/or OpenFoodTox whose (CAS, species, endpoint, duration)
tuple is absent from our ECOTOX frame. **Size unknown until an overlap audit is run; licence
unverified** (§6). Do not promise a claim from this track before the audit.

### 7.4 Tracks that stay where they are

| Track | Decision |
|---|---|
| **T2 unseen taxa** | Finish as pre-registered. Expect a negative (§1.2). Do not amend it to add an architecture. |
| **T3 post-freeze temporal** | Not startable: the current release holds 359 post-2021 curated rows, 115 already spent. Needs the next EPA release (~spring 2027). Not to be faked. |
| **T4 mixtures / formulations** | Exploratory after a data audit. No claim promised, no budget here. |

---
## 8. Sequencing within the shared 2-GPU budget, and what must be pre-registered first

### 8.1 What the GPUs are doing right now

GPUs 0 and 1 only (RTX 2080 Ti, 11 GB, no bf16 — `torch.cuda.is_bf16_supported()` returns False
[MEASURED HERE]), acquired through `sentinel.core.gpu` with the lead's read-only
`.gpu_locks/ALLOWED` at `0,1` and `SENTINEL_GPUS=0,1` on every job. At the time of writing
[MEASURED HERE]:

* **T2 chemprop**: 77 of 128 fold-jobs written, i.e. **~51 jobs left**. At the measured per-fold
  median of ~10 min and max ~30 min, that is roughly **8-15 GPU-h remaining**, which on a card
  shared with six other workstreams is **2-4 calendar days**.
* **T2 CPU queue**: `chemeleon_lightgbm` shard 0 of 4 is running; `gpboost`, `readacross`,
  `gpboost_published_chan` and the ensembles follow. One CPU-heavy job at a time, 6 threads.

**Therefore: no new GPU work should be armed before the chemprop queue drains.** Everything in
weeks 1-2 below is deliberately CPU-only or a <= 2 GPU-h one-off, so it does not compete with a
track that is already pre-registered and in flight.

### 8.2 Sequence

| Phase | Work | Compute | Gate before starting |
|---|---|---|---|
| **P0 (now, no compute)** | Lead reviews this plan. Decide: T1 as new primary, T6 as new secondary, and whether H4 (BCF) is pursued at all. | none | — |
| **P1 (days 1-3, CPU)** | Build R1/R2/R3 (monotone GBM, AFT GBM, multitask-by-stacking GBM) on the **full** acute + chronic + censored frame. Verify the licence status of the TRIDENT Zenodo deposit and EnviroTox; write `docs/licenses.md` entries. Run the H4 join/QC report. | CPU <= 6 threads, 1 job at a time | none — these are reference models on non-frozen data |
| **P2 (days 2-4, CPU + 1-2 GPU-h)** | §4.2 embedding cache for ~7,000 unique structures, in <= 2 h chunks, **after** chemprop drains or in a gap. `df -h` before each chunk, hard stop below 60 GB. | **1-2 GPU-h**, then CPU | T1's split must be frozen first (standing lead ruling: no embedding run touches T1 before its split is pre-registered) |
| **P3 (days 3-6, CPU)** | **T1 pre-registration**: ECFP4-Butina cluster file + hash, the 15% block, MDE recomputed by the project's own `mde` helper, reference-clustered co-primary CI, candidate names, configuration cap, comparator list including R1-R5, the four scorer-side guard gaps. Then freeze. | CPU | P1's licence + QC answers |
| **P4 (days 4-7, CPU)** | **T6 pre-registration**: duration bands, T6a/T6b definitions, (cluster x duration) resampling unit, MDE computed, same guard hardening. | CPU | P3 frozen (T6 reuses T1's cluster file) |
| **P5 (days 6-12, GPU)** | §4.3 TimeHead (<= 1 GPU-h) first, then §4.1 TED selection: 16 configurations x 3 folds = 48 jobs of one config x one fold, sized from the **slowest** sampled draw after a timed smoke. Pre-declared fallback to 8 configurations if the smoke exceeds 30 min/job. | **22-30 GPU-h, cap 32** | chemprop drained; T1 and T6 frozen; smoke timed and VRAM measured |
| **P6 (days 10-14, GPU)** | §4.4 MMoE as comparator + ensemble member. | **8-12 GPU-h** | P5 under way |
| **P7 (days 13-18)** | Freeze candidate, pre-test review, **one** sealed scoring pass per track, model card, `RESULTS.md`, post-results review. | ~4 h, needs the usual one-off exception for a pass longer than 2 h | review passes |

**Totals: ~32-45 GPU-h and ~3-4 calendar weeks**, against M4's existing ~25-35 GPU-h SOTA-plan
budget. Configuration caps, declared now: **T1 — 12 entries x 32 = 384 searched + fixed baselines**;
**T6 — 8 entries x 16 = 128 searched + fixed baselines** (smaller because T6's point is a shape
claim, not a hyper-parameter search). Every configuration counted, including discarded ones.

### 8.3 What must be pre-registered before any fit, in order

1. **The T1 cluster file and its hash**, plus the Butina cutoff (0.4) and the justification for it
   (largest cluster < 10% of rows), frozen before any model sees the split.
2. **The class-level exclusion rule** that keeps T1/T6 away from T2's reserved blocks without
   reading `test_blocks.parquet`, plus the `result_id` subtraction of the spent prospective rows,
   as an assertion in the data builder.
3. **The comparator list, with R1-R5 named**, and the parity statement that every input a candidate
   receives goes to every mandatory comparator on identical rows.
4. **The ablation set of §4.1** (A1-A6), because a win without A1 and A2 cannot be attributed.
5. **MDE per track and per stratum**, computed by the project's own helper before any fit, with the
   replicate floor reported separately as an absolute bound.
6. **The four scorer-side gaps** (`sota_campaign.md` §4): a lead-written FREEZE file as the only
   unlock, `BaseException` aborts before any claim or prediction is written, a full import-closure
   code hash, and all artifacts verified before the claim.
7. **The H4 decision**: BCF enters as ablation A5 only if the join/QC report lands first. If it does
   not, A5 is dropped and that is recorded, not quietly skipped.
8. **A dated novelty check** that includes L2, L3 and L4 — the three 2026 multitask papers — since
   they partially pre-empt the "more data + multitask" framing and two of them postdate the existing
   novelty check.

---

## 9. Honest outlook: probability of a RESEARCH_PLAN §15.1 label, per track

A §15.1 label requires a paired CI excluding 0 against **both** the strongest published method and
the strongest simple baseline, Holm-corrected, over >= 3 seeds, on a single look at an unseen block,
with the difference exceeding the block's MDE. These are my subjective probabilities, with the
reasoning that drives each one. They are deliberately lower than the enthusiasm in the design
documents.

| Track | What would be claimed | P(label) | Reasoning |
|---|---|---|---|
| **T2 unseen taxa** (in flight) | "A chemistry-only model predicts acute lethality for an unseen taxonomic class better than the published interspecies-extrapolation methods given the same information" | **10%** | A channel-matched comparator currently leads selection (§1.2) and claim condition (d2) requires beating all five of them. The remaining entries are mostly comparators and ensembles. Protocol §9 pre-registered a negative as the likely honest outcome and the data is agreeing with it. |
| **T1 unseen chemical clusters** | "SOTA on unseen-chemical-cluster acute toxicity" | **25%** | Best-powered track (MDE 0.006) and the regime where a graph model has already won once. But: the strongest published comparator is TRIDENT, reachable only through R5's released errors; L5's verified finding is that architecture headroom here is thin; and L2's reported finding (**[UNVERIFIED]**) is that multitask gains *shrink* as structural separation grows — which is precisely this track. The honest expectation is "competitive, maybe a small win on the reference, hard against the best simple baseline given parity". |
| **T6a duration extrapolation** | "SOTA on exposure-duration extrapolation for acute aquatic toxicity" | **30%** | The best shot, because the comparator class is structurally handicapped in a way that can be stated in advance (trees predict a constant past the last split) and because ~18,700 held-out rows at 444 distinct durations is real data, not a contrived split. Discounted because R1 (monotone GBM) is a much stronger comparator than the design documents assumed, and because L1 already conditions on duration, so the published bar is a duration-aware transformer, not a naive QSAR. |
| **T6b duration x unseen cluster** | the same, stricter | **15%** | Smaller block, power unknown until computed; likely reported as a stratum. |
| **T5 cross-source** | "gains survive a different curation pipeline" | **15%** | Cannot be estimated properly before the overlap audit and the licence check. Listed for completeness. |
| **H5 calibrated acute-to-chronic ratio** | *first-in-class reported quantity*, not a SOTA label | **40%** that it is publishable as a novel reported output; **<10%** that it yields a §15.1 label | There is no established comparator for a predicted ACR with an interval, which is exactly why it cannot be a SOTA claim. It is a good paper contribution and a bad SOTA claim, and it should be presented as the former. |
| **M4 overall, within ~4 weeks** | at least one §15.1 label on at least one track | **35-40%** | Union of T1 and T6a, discounted for shared failure modes (the same representation, the same data, the same comparator parity) and for the shared host. |

### Three ways this plan fails, stated in advance

1. **The references win.** R1-R3 close most of the gap, the networks tie, and M4's honest result is
   "a monotone multitask AFT GBM on all of ECOTOX is as good as anything we can build". This is a
   perfectly good outcome, it is cheap to reach, and it should be reported with the same prominence
   as a win. It is also, in my estimate, **the single most likely outcome** (~40%).
2. **The mechanism is not what works.** TED beats the comparators but ablation A1 (mechanism off,
   duration conditioned free-form) is within the MDE of the full model. Then the win belongs to the
   trunk and the extra data, the mechanistic framing must be dropped from the paper, and the claim
   becomes a representation-plus-data claim — which L1 already made in 2024.
3. **The schedule loses to the shared host.** 32-45 GPU-h on two cards shared by seven workstreams,
   with every job under 2 h, is 3-4 calendar weeks only if contention stays as it is now. The
   mitigation is already in the sequence: weeks 1-2 are CPU-only, the one GPU-expensive item has a
   pre-declared fallback to half the configurations, and TimeHead (<= 1 GPU-h) is deliberately
   ordered before TED so that a cheap answer about the mechanism arrives before the expensive one.

### One thing I would push back on

The framing "if simple ML is doing best, the task needs to be harder" is right, but there is a
failure mode worth naming: a task can be made harder in a way that makes *everything* worse and
nothing significant, which is what a class-level hold-out did to T2 (every model 1.10-1.22 against
an ADORE-CV 0.95). Harder must mean **structured differently**, not merely **further away**. T6 is
harder in the useful sense — the held-out regime has a known functional structure that a candidate
can exploit and a comparator provably cannot. T2 was harder in the other sense: it removed the
information every model would have needed, and the result is a well-measured negative. Both are
worth having, but only one of them was ever winnable, and the difference was visible before the
compute was spent.

---

## Appendix A — provenance of every [MEASURED HERE] number

Read-only computations run in this session. Environment: `/home/bcheng/miniconda3/envs/sentinel2`
(python 3.11, pandas, rdkit, lightgbm 4.7.0, xgboost 3.2.0, torch 2.7.1+cu118, CUDA available,
bf16 unsupported). Each script ran with `OMP_NUM_THREADS <= 4` and `python -I`.

**A.1 Frames read (read-only, no frozen file touched).**
`/home/bcheng/STOCKHOLM/sentinel_data/processed/ecotox/ecotox_aquatic.parquet` (471,097 rows, 108
columns) and `/home/bcheng/STOCKHOLM/sentinel_data/raw/ecotox/release_2026-09-15/results.parquet`
(52.8 MB, `endpoint` column only).
`docs/preregistration/chemtox_unseen_taxa_pins.json` and `test_blocks.parquet` were **not** read.

**A.2 Endpoint counts.** `endpoint_norm` value counts: NOEC 156,210; LC50 118,349; LOEC 115,684;
EC50 39,807; NOEL 12,824; LOEL 7,982; EC10 4,567; MATC 3,816. LC50 + EC50 = **158,156**, matching
the handoff's figure. Usable acute (uncensored, `log10_conc_mol_L` present, `chem_smiles` present) =
**122,946** rows / 4,592 CAS / 2,550 `latin_name` / 9,045 `reference_number`; with `duration_h`
present, **120,740**.

**A.3 Duration.** 444 distinct `duration_h` values among usable acute rows; **15.50%** outside
[24, 96]; maximum 26,298 h; **11,191** (CAS, species) pairs with >= 2 distinct durations, covering
**85,196** rows.

**A.4 Censoring.** Acute rows with `censored` true: **13,426**; with SMILES **12,035**; with SMILES
and molar concentration **11,719**. (The circulating "21,527" / "~21.5k" figure is not reproducible
on this frame for the acute endpoint.)

**A.5 Chronic and ACR.** Usable chronic (`NOEC/LOEC/EC10/MATC/NOEL/LOEL`): **225,081** rows / 3,298
CAS / 2,018 species. (CAS, species) pairs present in both the usable acute and usable chronic sets:
**10,334**.

**A.6 Bioconcentration.** Raw `results.parquet` endpoint codes matching `^(BCF|BCFD|BAF|BAFD)`:
**27,725** rows — BCF 20,422, BCFD 4,314, BAF 1,629, `BAF/` 650, `BCF/` 627, `BCFD/` 83.

**A.7 Scaffolds.** Murcko scaffolds (RDKit `MurckoScaffoldSmiles`) over the 4,592 usable-acute CAS:
**46.2%** of rows acyclic, 22 rows invalid SMILES, 914 distinct cyclic scaffolds over 3,033
chemicals, 662 single-chemical families; largest family `c1ccccc1` = **24,443 rows / 1,052
chemicals**.

**A.8 T1 universe and clustering.** Usable acute rows excluding classes
{Insecta, Gastropoda, Bivalvia, Polyplacophora, Cephalopoda, Amphibia}: **98,727** rows / 4,156 CAS
/ 1,586 species. Morgan radius 2 / 2048-bit fingerprints for 4,154 parseable structures, Butina at
Tanimoto distance 0.4: **2,738 clusters**, 2,159 of them single-chemical, largest cluster 7,105 rows
= **7.2%** of the universe. Cluster table saved to
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/m4_t1_clusters.csv`
and the row-to-cluster map to
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/m4_t1_rows.parquet`
(scratchpad only; nothing written under the data root or the repo).

**A.9 T1 power.** Cluster-bootstrap (2,000 resamples, clusters resampled with replacement) of the
paired RMSE difference between two synthetic predictors with error SD 1.2 correlated at rho; block
built by shuffling clusters with a fixed seed and accumulating to the target row fraction. Results
in §7.1's table. Method mirrors `unseen_taxa.mde`; it is a reimplementation, **not** a call into the
pinned module.

**A.10 Replicate floor.** Pooled within-group SD of `log10_conc_mol_L` over groups keyed by
(CAS, species, `duration_h`, `endpoint_norm`) with >= 2 records, on the T1 universe: **0.511** over
15,398 groups covering 65,664 rows. Overall target SD **1.768**.

**A.11 Library capability checks** (the §1.4 corrections). In `sentinel2`:
`LGBMRegressor(monotone_constraints=[1,0])` fits; `XGBRegressor(monotone_constraints=(1,0,0))` fits;
`xgboost.train({'objective':'survival:aft','aft_loss_distribution':'normal', ...})` with
`label_lower_bound` / `label_upper_bound` set on the DMatrix fits. All three returned without error.

**A.12 T2 live selection state.** `summary_index.json` at 2026-11-10 11:37: 10 entries, best mean
RMSE per entry as in §1.2's table. Chemprop: 77 fold-job JSONs present; 19 configurations have all
four folds scored; best 4-fold mean **1.1679** (cfg09), worst **1.2953** (cfg04). `plan_sha256`
recorded in the summary matches the live pin. No test-block artefact was read, created or modified.

**A.13 Disk.** `df -h /home/bcheng` at the start of the session: 3.5 T total, 3.2 T used, **86 GB
available, 98% full**; at the end of the session: 3.1 T used, **184 GB available, 95% full**. The
change is another user's, not ours, and it is the reason the disk rule is a per-build check.

---

## Appendix B — what I did not do

* No training, no tuning, no fit of any kind on any project data (the only fits in this session are
  the three 5-tree throwaway library capability checks of A.11, on random numbers, in `/tmp`).
* No download, no `pip install`, no edit to any repo file other than this one, no `git`, no process
  signalled, no `keys.env` read.
* No frozen artefact read: not `test_blocks.parquet`, not the pins, not `AUTHORISED.json`.
* No citation asserted without either a retrieval in this session or an explicit **[UNVERIFIED]**
  label. In particular I did **not** verify the numbers inside L2, L3, L4 or L6 — only their
  bibliographic metadata — and I did not verify the Zenodo deposit, EnviroTox's licence, Web-ICE's
  downloads, or any of the toxicokinetic literature the mechanistic design is named after.
