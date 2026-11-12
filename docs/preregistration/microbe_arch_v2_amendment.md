# M3 SENTINEL-Microbe — architecture round v2 amendment: arms D1–D4 and tracks H1–H4

**New file, drafted 2026-11-10 (12:45 EST).** It amends nothing in place. No frozen file is edited:
`microbe_protocol.md`, `microbe_ea_protocol.md`, `events_microbe_protocol.md`, the two split files and
`sentinel/eval/events.py` keep their bytes. The plan (§8.1 item 1) proposed a separate
`microbe_geo_protocol.md` plus a `microbe_protocol.md` §10.9 entry; lead rule 1 replaces both with this one file.
After the stamp this file is immutable. Dated execution entries go to a separate append-only file,
`docs/preregistration/microbe_arch_v2_log.md` (created at its first entry).

**Status: UNSTAMPED DRAFT.** No fit and no development or validation number exists for D1, D2, D3 or D4, or for
any GEO, GAS or EA target. The lead stamps this file (sha256 + a `microbe/prereg_snapshot` registry row + a
commit). Until that stamp exists, only the work listed in §13 may happen.

**Governing inputs (sha256 at drafting).**

| File | sha256 |
|---|---|
| `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` (rules 1–6; "M3 Microbe" incl. rulings 4a/4b; governs where the plan differs) | `eedd50b07b0cfb7bd8b49f3f9fa822cfb4e1f4516dfac9973beebace6cb7866b` |
| `docs/plans/architecture_research/M3_architecture_plan.md` (§4 D1–D4, §5, §7, §8.1 items 1–3 and 5–6) | `66c89834518ef9c9afd8c0e74fb7565cfc0145bc079f5cecce80aed9f0355e17` |
| `docs/preregistration/microbe_protocol.md` (unchanged) | `7580d2c8570085547add54f034b2d952ea55ab68cad8732c7d9f1ec5767e2ec4` |
| `docs/preregistration/microbe_ea_protocol.md` (unchanged) | `85ee36671775cc0ad0c90c77a75e80884e1ed4992e3d6f2ac49652f7f3f167e6` |
| `docs/preregistration/events_microbe_protocol.md` (unchanged) | `b9b2bf21d5cc59293dc44a3de03f4bfa24348000b47ed1769b3ace23accd5d85` |
| `docs/preregistration/microbe_split.json` (frozen split, unchanged) | `aaeb05305afd8df9a1cc677f974d443b5417b1acdbd95ef1160368dd120626f8` |
| `docs/preregistration/microbe_split_lodo_x001.json` (unchanged) | `a61416982db82cd10323e2e565d00fda544fdb9c6bd339194f02486556e58caa` |

**What had been seen when this was written (complete list).**
- Development-LODO results on the 8 primary targets for every existing model, including cycles 5–6 (§1.1).
- For the new targets: **counts and detection-limit metadata only** (Appendix A). No GEO value, on any row, was
  printed, plotted or fitted. Holdout and reserve rows: counts only.
- qPCR table: row counts, join coverage, number of labs and batches. No copy-number value was inspected.
- EA: ENA run metadata only (titles, dates, read counts); the Zenodo record metadata of Thorpe et al.; the two
  modelling scripts of the Warren repository (code only). **No EA chemistry file has been downloaded**, neither
  the Warren CSV nor `water_chem.csv`.
- NEON dissolved gases (DP1.20097.001): the product catalogue entry only. No data file.
- The 2024 holdout, the pre-2017 reserve and Test-B have never been scored, and nothing here changes that.

**What is not changed.** The H-M3 primary endpoint and its 16-hypothesis Holm family (protocol §1, §10 C1); the
frozen split and the `lodo_x001` track; Test-B (§10.5); the event track; the registered EA analysis of total
oxidised nitrogen (TON); the permutation nulls; the rule that baselines are never weakened.

---

## 1. Context and the ledger of configurations

### 1.1 Where the existing track stands (development LODO, mean pooled R² over the 8 primary targets; registry)

| Model | `lodo` | `lodo_x001` |
|---|---|---|
| Ens[V4-Ridge + V4-XGB] | 0.325 | 0.325 |
| V4-XGB + context | 0.317 | — |
| V4-XGB | 0.314 | 0.305 |
| V4-Ridge | 0.291 | 0.301 |
| XGB[genus-CLR], untuned / tuned (W6) | 0.289 / 0.282 | — |
| SeqSet[`seqset_kmer_v4`] (cycle 5, #1 / #2) | 0.269 | 0.235 |
| SeqSet[`seqset_kmer_v4_150`] (cycle 6 control, #3) | 0.246 | — |
| SeqSet[`seqset_kmer_v4_150_emp`] (cycle 6 EMP-pretrained, #4) | 0.206 | — |

- **Cycle 5 outcome (negative).** The V4-token SeqSet stays below the V4 tables on both tracks, and the
  `lodo_x001` rows did not help it.
- **Cycle 6 outcome (negative).** #4 (0.206) is below its control #3 (0.246). By the rule fixed in protocol
  §10.8, **#5 is not run.** EMP pretraining is reported as a negative result. MiCoDa is not fetched.
- **W6.** `xgb_clr_tuned` is finished (0.282, below the untuned 0.289). `rf_clr_tuned` is still running. The
  logged paired comparisons are run once it ends, as already planned. Nothing in this amendment delays it.
- These numbers are selection-biased (about 25 SENTINEL and about 30 baseline configurations so far). They are
  the reason for the new arms and were known when the arms were designed. They are not evidence for any arm.

### 1.2 Slots on the existing tracks (protocol §10.8 cap)

| Slot | Configuration | Tracks | Status |
|---|---|---|---|
| #1, #2 | `seqset_kmer_v4` | `lodo`, `lodo_x001` | run |
| #3 | `seqset_kmer_v4_150` | `lodo` | run |
| #4 | `seqset_kmer_v4_150_emp` | `lodo` | run |
| #5 | `seqset_kmer_v4_150_emp` on `lodo_x001` | — | **not run (rule); retired, not reassigned** |
| **#6** | **NicheInvert-S** (D1, §2) | `lodo` + `lodo_x001` | new |
| **#7** | **V4-MR** `xgb_v4mr` (D3a, §3.1) | `lodo` + `lodo_x001` | new |
| **#8** | **NicheGeo-Phys** (D2, §4) **only if the D1 gate (§2.6) passes; otherwise V4-MR-Stack** (D3b, §3.2) | `lodo` + `lodo_x001` | new, conditional |

- The "V4-window DNA-LM" reserve slot of §10.8 is **dropped**. No DNA language model is built in this round.
- **Counting (needs the lead's explicit confirmation, OQ1).** §10.8 numbered configuration × track (#1 and #2
  are one model on two tracks). The lead dispatch asks for each of #6–#8 on both tracks. That gives 4 runs done
  + up to 6 new = **up to 10 configuration × track runs, against the "≤ 8" of §10.8.** The stamp of this file is
  taken as approval of 10. If the lead prefers the strict count of 8, the pre-declared fallback is: #6 on both
  tracks, #7 and #8 on `lodo` only.
- **Counted ablations (never selectable, never a claim; reported):**
  - A1: NicheInvert with a taxonomy prior instead of the sequence prior (`lodo` only);
  - A2, A3: NicheGeo-Phys without the physics decoder, and without the context prior (`lodo` only; only if D2
    runs).
- The plan's third D1 item, "WA-tol with inverse deshrinking", is a published method. It is run and reported as
  a **comparator** (§6), not as a SENTINEL configuration.
- Every run above is logged with `TrainingRun` and `log_result`, numbered in its run notes, and counted in the
  "configurations tried" total reported with any claim.

### 1.3 Rules common to all arms
- Identical folds, rows and labels for candidates and comparators. Stochastic models: mean of seeds 0–2
  (protocol §5). Deterministic models are stated as such.
- One change at a time. An arm's implementation files are hashed at its first development run. Any later change
  to a fitted quantity is a new counted configuration, declared in the log file before it runs.
- Development LODO only. No holdout, reserve, Test-B or EA label is read outside the sealed looks.
- Wording stays predictive. No arm supports a causal or mechanistic claim about any taxon (Juggins 2013).
- No novelty or first-in-class wording for any arm until a dated systematic search is in
  `docs/novelty_checks/microbe.md`. That search is owed before the freeze candidate.

---

## 2. Arm D1 — NicheInvert-S (slot #6)

**Practical reason.** The tables treat a sample as a feature vector and match it to similar training samples.
An agency that samples an unmonitored region needs a method that still works when the community is partly new.
Inverse calibration uses every sequence unit, including rare ones, gives units without history a niche from
their DNA, and returns an uncertainty with each estimate. This is the transfer-function idea of palaeolimnology
(weighted averaging), made multivariate and extended to exact sequence units.

### 2.1 Model (all constants fixed here)
- **Master vector.** D = 17 dimensions: the 8 primary targets (protocol §3), the GEO-8 targets (§7.1) and field
  water temperature (`field_temp_c`, °C, an unscored auxiliary). Each dimension is standardised with the mean
  and SD of the training-fold rows that have it. One fit serves both the primary family and the GEO family.
  **Deviation from the plan (D = 14), declared here (OQ2):** the plan's 14 had no DO % saturation, NH4 or
  UV254, so a second fit would have been needed for GEO.
- **Units and vocabulary.** The unit is the harmonised 230-bp V4 window (sha1), as in cycle 3. The fold
  vocabulary is every unit present in ≥ 2 training events. Reads outside the vocabulary are dropped from the
  likelihood, and their share of held-out read mass is reported.
- **Niche of unit k on dimension d**, from training events that have d, weighted by relative abundance p_jk:
  - optimum μ̂_kd = Σ_j p_jk y_jd / Σ_j p_jk;
  - tolerance t̂²_kd = Σ_j p_jk (y_jd − μ̂_kd)² / Σ_j p_jk, divided by (1 − 1/N2_k) when N2_k > 1;
  - N2_k = (Σ_j p_jk)² / Σ_j p²_jk (Hill's effective number of occurrences).
- **Sequence prior (the claim configuration).** x_k = the first 64 principal components of the standardised
  5-mer profile (1,024-d) of the unit's window. The PCA is fitted on the fold vocabulary and uses no label. A
  ridge map s_d(x) from x_k to μ̂_kd is fitted over training units with weights N2_k (alpha from
  {0.1, 1, 10, 100, 1000} by generalised cross-validation).
  - Shrunk optimum: μ_kd = λ_k μ̂_kd + (1 − λ_k) s_d(x_k), with λ_k = N2_k / (N2_k + κ) and **κ = 5**.
  - Shrunk tolerance: t²_kd = λ_k t̂²_kd + (1 − λ_k) t̄²_d, where t̄²_d is the N2-weighted median of t̂²_kd over
    units with N2 ≥ 5. Floor: t_kd ≥ 0.25 (standardised units). Precision w_kd = 1 / t²_kd.
- **Response.** log λ_k(z) = a_k − ½ Σ_d w_kd (z_d − μ_kd)². The intercept a_k is set in closed form so that the
  unit's mean modelled abundance over training events equals its mean observed relative abundance. Missing
  training dimensions are filled with the context-prior mean for this step only.
- **Context prior.** m(ctx) is a multi-output ridge on the tier-0 context features (day-of-year harmonics,
  latitude, longitude, elevation, aquatic type), with alpha by site-grouped inner cross-validation from the
  baseline grid. Σ is the covariance of its training residuals (pairwise-complete), shrunk to its diagonal with
  intensity 0.2, with an eigenvalue floor of 10⁻³.
- **Inference for one event.** Maximise L(z) = (N_eff / N_j) Σ_k n_jk log π_k(z) + log N(z; m(ctx), Σ), where
  π(z) is the softmax of log λ(z) over the whole fold vocabulary (so the absence of a unit is evidence) and N_j
  is the event's in-vocabulary read count. Start at z = m(ctx). At most 6 damped Newton steps; a step is kept
  only if L increases (up to 5 halvings).
- **Effective sample size.** N_eff ∈ {10, 30, 100, 300, 1000}, chosen inside each training fold by 3-fold
  domain-grouped inner cross-validation. Criterion: mean inner R² over the 16 scored dimensions.
- **Inverse deshrinking (cross-fitted).** Per dimension, a linear map from the inferred z_d to the label is
  fitted on the inner out-of-fold inferences at the chosen N_eff. Slope clipped to [0.5, 3]. It is applied to
  the outer predictions.
- **Output.** The deshrunk posterior mode, back on the target scale, plus a Laplace SD from the inverse
  negative Hessian. Coverage of the 90% interval is reported descriptively. No claim rests on it.
- **Determinism.** No stochastic step and therefore no seed average. If a randomised solver is used for the
  PCA, its seed is fixed to 0 and that is stated. This is disclosed against the "seeds 0–2" rule.
- **Inputs not used.** Community descriptors (reads, richness, primer) are not inputs. No chemistry or sensor
  value, site id, domain id or year is an input.
- **Class.** NicheInvert-S uses season and location through its prior, so it is a "+context" model in the sense
  of protocol §10.4. It must also beat the stronger +context genus baseline, and any claim is worded "community
  adds skill beyond season + location".
- **On EA** (no coordinates): a variant with a day-of-year-only prior, fitted on NEON. Units absent from NEON
  get the sequence-prior optimum, the pooled tolerance and the median intercept of NEON units with N2 < 5. This
  open-vocabulary case is where the sequence prior matters. Inside NEON it cannot matter much: unseen units are
  about 0.4% of held-out read mass (plan C9).
- **Implementation.** `sentinel/models/microbe/niche.py` with tests in `tests/test_microbe_niche.py`.

### 2.2 Tracks, unseen test blocks, power
- Development: `lodo` and `lodo_x001`. On `lodo_x001` the .001 2017–18 events are extra training events; they
  share units through the V4 window.
- Unseen blocks: the 2024 holdout (H-M3 family and GEO family in one sealed look); the pre-2017 cross-release
  reserve; Test-B; EA (§9).
- Power on the existing endpoint is unchanged and already on file
  (`microbe_split_lodo_x001_report.json`): development paired site-cluster SE of 0.03–0.06 per target, so a
  minimum detectable ΔR² of 0.105 (TP) to 0.203 (TN) at α = 0.05/8 and power 0.8.

### 2.3 Ablation A1 (counted, descriptive)
The taxonomy prior: shrink toward the abundance-weighted mean optimum of the unit's RDP genus (family if the
genus is unassigned) instead of s_d(x_k). `lodo` only. It is never selectable, because taxonomy labels are not
available for EA windows without a separate classifier run (plan C5).

### 2.4 Attribution control (not a comparator for the gate, never selectable)
`xgb_nichez_context`: XGB on the cross-fitted NicheInvert-S posterior modes and Laplace SDs plus the context
features. If this control beats NicheInvert-S with a paired CI excluding 0, a win is worded "niche-summary
features with a GBM" and no architecture label is given to the inversion itself.

### 2.5 Kill rules for D1
- Any unit test in §13 failing → no development run.
- Newton non-convergence (no increase of L at the first step) on more than 5% of training events in the
  train-only smoke test → fix before the first development run. After the first run, a fix is a new counted
  configuration.
- CPU budget: ≤ 6 threads, ≤ 6 CPU-h per track. If exceeded, the vocabulary is restricted to units in ≥ 3
  training events (declared in the log before the rerun; counts as the same configuration only if no
  development number from the aborted run was read).

### 2.6 The pre-declared gate from D1 to D2
Evaluated once, on `lodo`, from registry values. **Both** conditions must hold:
1. mean development-LODO R² of NicheInvert-S over the 8 primary targets ≥ **0.294** (V4-XGB 0.314 − 0.02);
2. on at least **2 of the 8 primary targets**, the point ΔR² of NicheInvert-S against the strongest simple
   baseline is ≥ **+0.03**. The strongest simple baseline is, per target, the maximum development-LODO R² over
   every model of the parity list (§6) that is not derived from NicheInvert.

If the gate fails, D2 is not run, the stop is reported, and slot #8 is V4-MR-Stack. The plan words the gate on
"the 8 targets"; the GEO targets do not enter it (OQ10).

**Honest expectation (from the plan, recorded before any fit).** P(condition 1) ≈ 0.45. P(D1 beats the
strongest simple baseline by ≥ 0.05 on ≥ 2 targets) ≈ 0.15.

---

## 3. Arm D3 — V4-MR (slot #7) and V4-MR-Stack (slot #8 if the gate fails)

**Practical reason.** `xgb_v4clr` pools about 48% of held-out read mass into one "other" feature. Raising the
unit cap and adding coarser taxonomic levels recovers that evidence at no GPU cost. It is a representation
change, not a new model class, and it is reported as such.

### 3.1 D3a `xgb_v4mr`
- **Features (all fitted on training rows only):**
  - CLR of V4 units: ≥ 5% training prevalence, the 3,000 most prevalent, the rest pooled (pseudocount 0.5);
  - CLR of genus, family and order counts (three `GenusCLR` blocks, each ≥ 5% prevalence, cap 500, rest
    pooled). The labels are the RDP labels that NEON publishes for the sequences;
  - the community descriptors, as for every tabular community model.
- **Learner.** XGB with the `xgb_v4clr` hyperparameters, unchanged (600 trees, learning rate 0.03, depth 4,
  subsample 0.8, colsample 0.3, min child weight 3). Seeds 0–2. No context features.
- **Twins (comparators, §6):** `xgb_gfo` = XGB[genus + family + order CLR] without V4 units, and
  `xgb_v4clr_3000` = XGB[V4-CLR, cap 3,000] without the aggregated levels. If `xgb_v4clr_3000` matches
  `xgb_v4mr`, the gain is reported as a cap effect.
- **Disclosure.** V4-MR uses taxonomy labels, so it is not purely DNA-encoded. Its eligibility under the
  selection rule is OQ3 (§11.1).
- **Expectation (plan).** P(`xgb_v4mr` ≥ V4-XGB + 0.02 on the mean) ≈ 0.35.

### 3.2 D3b `xgb_v4mr_stack` (only if the D1 gate fails)
- D3a's features plus **cross-fitted auxiliary predictions.** Inside each outer training fold, 5-fold
  domain-grouped inner cross-validation gives out-of-fold predictions of the 17 master dimensions
  (`xgb_v4mr`, seed 0) for the training rows. For held-out rows the auxiliary models are refitted on all outer
  training rows. The model for target t receives the predictions of the 16 other dimensions.
- No held-out label enters any feature. A unit test asserts this.
- **Twin (comparator):** `xgb_clr_stack` = XGB[genus-CLR] with the same construction on genus-CLR features.
  This gives the baseline the same auxiliary labels (lead rule 3).
- **Expectation (plan).** P(D3b beats its twin with a CI excluding 0 on ≥ 1 target) ≈ 0.10.

### 3.3 Kill rules for D3
- CPU: ≤ 6 threads; ≤ 6 CPU-h per track for D3a and ≤ 12 CPU-h per track for D3b. If D3b exceeds this, it
  runs on `lodo` only.
- D3b is not run at all if the D1 gate passes.

---

## 4. Arm D2 — NicheGeo-Phys (slot #8, only if the D1 gate passes)

**Practical reason.** Measured analytes constrain one another through exact chemistry (pool sums, the carbonate
system, conductivity from ions, oxygen saturation). A model that carries those identities can use the label of
one analyte as evidence about another and stays physically consistent outside the training range.

- **Design (plan D2, with critique items C1–C10 applied).** NicheInvert trained end-to-end through an unrolled
  inverse layer, followed by a fixed physics decoder.
  - Latent: the 17 master dimensions plus soluble reactive phosphorus (18-d). SRP is a latent only: 368 of
    1,182 usable development rows are censored, so its head is descriptive.
  - Exact identities only: TN ≥ dissolved inorganic N and TP ≥ SRP through pool sums; SC from ions (McCleskey
    et al. 2012); DO from DOsat(T, pressure(elevation)) (Benson & Krause 1984); pH from DIC and alkalinity
    (Plummer & Busenberg 1982), in float64.
  - At most 3 learned scalars with tight priors. Seeds 0–2.
- **Remaining hyperparameters** (optimiser, epochs, the 3 scalars and their priors) are fixed in a dated log
  entry, visible to the lead, **before** the first D2 development run. They may use nothing but D1's results.
- **Budget and stop rule.** ≤ 12 GPU-h in total, jobs ≤ 1 h 55 min through the lock helper, peak VRAM ≤ 4 GB,
  a checkpoint after every fold-seed. **Stop after 4 GPU-h** if the development mean R² (8 primary targets) is
  below 0.294 **and** no GEO target beats its best parity comparator by point estimate.
- **Mandatory twin:** XGB-chained-physics. XGB predicts the latents and the identical decoder derives SC, pH,
  DO, TN and TP. A GBM can use the same physics after the fact, so a D2 claim must beat this twin.
- **Ablations A2, A3** as in §1.2.
- **Expectation (plan).** P(D2 runs) ≈ 0.35 to 0.45. Given that it runs, P(≥ 0.03 over D1 on ≥ 2 targets) ≈ 0.3.

---

## 5. Arm D4 — qPCR input (input amendment to protocol §4)

**Practical reason.** Relative abundances cannot express biomass. Total 16S copies carry load information that
is plausibly related to trophic state (TP, chlorophyll, turbidity), and agencies that sequence can run the same
qPCR assay cheaply.

- **Amendment to the input list.** Microbial qPCR abundance (NEON DP1.20278.001, `mga_swGroupAbundances`,
  target gene 16S, "bacteria and archaea") is a **community-side input. It is never chemistry.** It joins by
  `dnaSampleID`. Non-provisional records only.
- **Features ("QMP block", fixed):**
  - log10 `meanCopyNumber` and log10 `nucleicAcidConcentration`, as published (mean over the event's samples);
  - log10 filtered sample volume (`amc_fieldGenetic.geneticFilteredSampleVolume`, joined by `geneticSampleID`);
  - a load index = the sum of the first two minus the third. This is log copies per mL up to the elution
    volume, which is assumed constant and is stated as an assumption;
  - one missing-value flag per feature. Trees receive NaN; ridge receives the training median.
- **Never inputs:** laboratory name, batch id, protocol version, processing date.
- **Where it applies.** On the **GEO track only** (plan §8.1 item 6), as the "+QMP" twin of a model (§7.5, G10).
  Primary-target scores of +QMP models are descriptive and never enter the §6 selection.
- **Parity.** Every comparator on the GEO track gets a +QMP twin with the same block. A control
  `xgb_context_qmp` (context + QMP, no community) shows what the block alone explains.
- **Coverage (label-free, Appendix A.3).** A 16S qPCR record exists for 1,159 of 1,215 development events
  (95.4%), 214 of 214 holdout events, 510 of 517 `.001` 2017–18 events and 172 of 190 reserve events. The
  filtered volume is missing for 25 holdout events (pooled or metagenome extractions of July–August 2024, the
  same 25 events that are time-matched to chemistry).
- **Kill rules.**
  - A 16S qPCR record on fewer than 90% of development events or of holdout events → D4 is not run.
  - Confound check, label-free, before any +QMP fit: the share of variance of each QMP feature explained by
    laboratory and by batch (2 laboratories, 263 batches). It is reported with every +QMP result. If laboratory
    alone explains more than 50% of the variance of the load index, +QMP results are descriptive only.
- **Expectation (plan).** P(≥ +0.03 on TP or chlorophyll for all models) ≈ 0.25. D4 cannot by itself separate
  SENTINEL from the baselines.

---

## 6. The parity list (comparators; lead rule 3)

Every comparator runs on the same folds, rows, labels and seeds rule as the candidate. "Family" says where it
is required: P = the 8 primary targets, G = GEO-8.

| Comparator | Definition | Family | Status |
|---|---|---|---|
| `tier0_best` | per target the best of GlobalMean, Ridge[DOY+location], RF[DOY+location] | P, G | P run; G to run |
| RF[genus-CLR], XGB[genus-CLR] | untuned defaults **and** the W6 tuned grids; the stronger is used | P, G | P: RF-tuned running; G to run |
| Ridge[genus-CLR], TabPFNv2[genus-CLR], Ens[Ridge + XGB] (genus) | as protocol §6 | P, G | P run; G to run |
| RF/XGB[genus-CLR + context] | the +context genus baselines of §10.4 | P, G | P run; G to run |
| V4-XGB, V4-XGB + context, V4-Ridge, Ens[V4-Ridge + V4-XGB] | cycles 3–4 | P, G | P run; G to run |
| **RF[V4-CLR]** | baseline RF defaults on the `xgb_v4clr` features | P, G | never run; required |
| **TabPFNv2[V4-CLR]** | tabpfn 2.2.1, 300 most prevalent units, 8 estimators (GPU, lock helper) | P, G | never run; required |
| **XGB[V4-CLR, 3,000]** `xgb_v4clr_3000` | D3a twin | P, G | to build |
| **XGB[genus + family + order]** `xgb_gfo` | D3a twin | P, G | to build |
| **WA-tol** | classical weighted averaging with tolerance down-weighting and inverse deshrinking (ter Braak & van Dam 1989; Birks et al. 1990), same units as D1, square-root relative abundance | P, G | to build |
| **WA-PLS** | 1–5 components chosen by inner domain-grouped CV (ter Braak & Juggins 1993), same units and transform | P, G | to build |
| Multi-output XGB `xgb_v4clr_mo` | one multi-output-tree model per family on the `xgb_v4clr` features; missing training labels set to the training mean (a stated limitation) | P, G | to build |
| `xgb_clr_stack` | D3b twin | P, G | only if D3b runs |
| XGB-chained-physics | D2 twin | P, G | only if D2 runs |
| +QMP twins, `xgb_context_qmp` | §5 | G | only if D4 runs |

- **Published method re-run here.** For this task the published methods are tree ensembles on taxon tables
  (Hermans et al. 2024; Warren et al. 2026) and the weighted-averaging transfer functions. The tuned
  RF/XGB[genus-CLR] and WA-tol / WA-PLS are their re-runs. TabPFN v2 stays the foundation-model comparator
  (lead decision; no TabPFN-2.5).
- **Baselines are never weakened.** Where a tuned and an untuned version exist, the stronger on development
  LODO is the comparator, per target.
- **"Strongest simple baseline"** for any SOTA or architecture label is, per target, the best development-LODO
  model of this list that is not the candidate itself. It is fixed at the freeze.
- **Hard requirement.** A family whose required comparators are not all finished at the freeze is not scored
  inferentially in the sealed look (§7.7, §12).

---

## 7. Track H1 — NEON GEO-8 (new target family, same sealed look as H-M3)

**Practical reason (fixed before any result).** Agencies need ionic, carbonate, oxygen and reduced-nitrogen
state in places they do not monitor: chloride for road-salt salinisation, sulphate for mining and acid
deposition, calcium and DIC for hardness and buffering, pH for acidification, DO saturation for hypoxia and
metabolism, ammonium for toxicity, UV254 for the aromaticity of dissolved organic matter. The list was chosen
on these grounds from the analytes NEON measures at the same visit.

**Honest note recorded now.** Cl, SO4, Ca and DIC are mostly between-site geology. Community models should
beat tier-0 easily on them, as they do for SC, but ridge or XGB on CLR tables will probably do as well. The
targets where a model-versus-GBM gap is most plausible are the microbially mediated ones: NH4 and DO saturation.

### 7.1 Targets, transforms, floors

| Target | Source column (`chem_event_wide`) | Transform | Floor |
|---|---|---|---|
| `ph` | `lab_ph` | none | — |
| `log_cl` | `cl_mgl` | log10 mg/L | 0.003 |
| `log_so4` | `so4_mgl` | log10 mg/L | 0.006 |
| `log_ca` | `ca_mgl` | log10 mg/L | 0.001 |
| `log_dic` | `dic_mgl` | log10 mg/L | 0.025 |
| `do_sat` | `field_do_sat_pct` (same-visit field meter) | none (%) | — |
| `log_nh4` | `nh4_n_mgl` | log10 mg N/L | 0.002 |
| `log_uv254` | `uv_absorbance_254_nm_au` | log10 absorbance units | 0.001 |

- **Floor rule.** The floor is the smallest non-zero detection limit among the non-provisional laboratory
  records of that analyte (Appendix A.2). It was read from record metadata, not from sample values.
- **Censoring** follows protocol §2.2 unchanged: the published value when present, otherwise the record's
  detection limit, then the floor; the flag `<target>_censored` is carried.
- **NH4 rule (fixed in the plan before any count).** NH4 is an inferential member only if at most 30% of its
  usable development rows are censored. Counts only: 247 of 1,182 = **20.9%**. NH4 is therefore in the family.
  (Holdout: 41 of 214 = 19.2%. The `.001` 2017–18 training rows: 220 of 493 = 44.6%; they are used with the
  same substitution rule and this is disclosed.)
- `do_sat` is a same-visit field reading. The primary target `do` is the sonde daily mean in mg/L. They are
  related but not the same measurement.
- **Descriptive heads (reported, no inference, no selection):** field SC and field DO in mg/L (alternate label
  source for `log_sc` and `do`). Mg, Na, K, Si, TDN, TDP and SRP are reported only if D2 runs.

### 7.2 Join, rows, split
- **Join.** The builder's own visit key: `sentinel.data.microbe_wq.collapse_visits` on the event's
  `chem_parent` (`parentSampleID`), with the registered ±24 h fallback already encoded in `chem_match`. No new
  key is written. A unit test requires the build to reproduce the counts of Appendix A.1 exactly.
- **Split.** Unchanged. Development = the 1,215 events of `microbe_split.json`; holdout = the 214 events from
  2024-01-01; `lodo_x001` adds the 517 `.001` 2017–18 events to training only.
- **Tables.** `scripts/data/build_microbe_geo.py` writes `processed/microbe_wq_geo/dev_targets.parquet`
  (development and `.001` 2017–18 rows) and `protected_labels.parquet` (holdout and reserve rows). The protected
  file is read only by the scorer under the open claim. The development loader refuses protected ids
  (unit-tested). The build prints counts only.

### 7.3 Counts and power (counts only; Appendix A.1)
- **Development.** 1,117 to 1,182 usable rows per target, all on 33 sites and 19 domains. This is the same
  cluster structure and nearly the same n as the primary grab targets, whose paired site-cluster SEs are
  0.03–0.06. Expected minimum detectable ΔR² at α = 0.05/8 and power 0.8: **about 0.10–0.20.**
- **Holdout.** 210 to 214 usable rows per target, 33 sites, 19 domains. The SE scales between 1× (if between-site
  variance dominates) and √(1,215/214) = 2.38× (if rows dominate). Minimum detectable ΔR²: **about 0.10–0.49.**
  The holdout alone is weak for small gains. The pooled 2024 + 2025 analysis (about twice the events) is the
  realistic power source and is secondary (§7.6).
- **Before the freeze** a paired-SE power file, `docs/preregistration/microbe_geo_power.json`, is written from
  the comparators' development out-of-fold predictions (method of `microbe_split_lodo_x001_report.json`). A
  target whose development minimum detectable ΔR² exceeds 0.30 is labelled "underpowered" in advance. It stays
  in the family, so the Holm denominator does not move.
- 25 of the holdout rows per target are time-matched (the July–August 2024 pooled extractions). The registered
  sensitivity analysis without time-matched rows applies.

### 7.4 Hypothesis H-GEO and the families
- **H-GEO.** On the 2024 holdout (leave-one-domain-out × time-out, protocol §8), the pooled R² of the frozen
  GEO model M\*_GEO exceeds **both** `tier0_best` and the stronger of RF/XGB[genus-CLR] (tuned or untuned), with
  the paired bootstrap 95% CI (B = 2,000) excluding 0 for **both** the site-cluster and the domain-cluster
  unit, on **≥ 2 of the 8 GEO targets**, Holm-corrected across the 8 targets. A target counts only if every
  comparison passes.
  - If M\*_GEO is a +context model, it must also beat the stronger +context genus baseline (§10.4 rule).
  - If M\*_GEO is a +QMP model, the comparators are the +QMP twins of those baselines.
- **Families.** H-M3 keeps its 16 hypotheses. H-GEO is a **separate** family of 8. Each family is tested at
  α = 0.05, so the error rate across both families can reach 0.10. Each claim is scoped to its own family.
- **SOTA or architecture label on GEO.** Additionally, on each claimed target, M\*_GEO must beat the frozen
  strongest simple baseline (§6) with the CI excluding 0 for both units.
- The reserve route is **descriptive** for GEO: only 62 of the 190 reserve events have a chemistry parent.
- All 8 targets are reported, win or lose, including targets "explained by season and location".

### 7.5 Candidates and cap (≤ 10 configuration × track entries scored on GEO targets)

| # | Model | Track |
|---|---|---|
| G1, G2 | NicheInvert-S (the same fits as slot #6; the GEO dimensions are part of its master vector) | `lodo`, `lodo_x001` |
| G3, G4 | `xgb_v4mr` | `lodo`, `lodo_x001` |
| G5, G6 | the slot-#8 model (NicheGeo-Phys or `xgb_v4mr_stack`) | `lodo`, `lodo_x001` |
| G7 | `xgb_v4clr` | `lodo` |
| G8 | `xgb_v4clr_context` | `lodo` |
| G9 | Ens[V4-Ridge + V4-XGB] | `lodo` |
| G10 | the +QMP twin (§5) of the G1–G9 entry with the highest GEO mean | that entry's track |

- Only G1–G10 are selectable for GEO. There is no spare. If the lead rules for the strict count in OQ1, G4 and
  G6 are dropped.
- G7–G9 have two roles. For H-GEO they are candidates, as they are for H-M3. For an architecture label on
  D1, D2 or D3 they are comparators (§6).
- V4-Ridge alone, RF[V4-CLR], TabPFNv2[V4-CLR] and `xgb_v4clr_3000` are comparators only on GEO. If one of
  them is the best development model on a target, that is reported, and no SOTA label is possible on that
  target without beating it.
- Comparators are fitted on the candidate's own track.

### 7.6 Secondary analyses
Test-B (2025 events) and the pooled 2024 + 2025 analysis of protocol §10.5 are extended to the GEO family with
the same rules (a separate secondary Holm family of 8; the fits saved at primary scoring are reused).

### 7.7 The sealed look
- GEO is scored in the **same** look `M3/test` as H-M3. Primary and GEO predictions are sealed together.
- `scripts/eval/microbe_final.py` and the guard tests are extended **before** the freeze candidate is sent.
  The import-closure hash then covers the GEO scoring code, and `FREEZE_microbe.json` pins both GEO tables.
- **Kill rules.**
  - This amendment is not stamped before the freeze → GEO is scored on Test-B only.
  - Any comparator that §6 requires for family G is unfinished at the freeze → GEO is not scored inferentially
    in the sealed look and moves to Test-B.
  - The build does not reproduce the Appendix A.1 counts → stop and report before any fit.

---

## 8. Track H2 — NEON GAS-3 (dissolved CO2, CH4, N2O)

**Practical reason.** Streams and lakes emit greenhouse gases, and methane and nitrous oxide are produced and
consumed by microbes. These are the targets where the community is closest to the process.

- **Source.** NEON DP1.20097.001 "Dissolved gases in surface water" (CC BY 4.0; active; 34 sites; RELEASE-2026;
  2014-10 onward). Nothing is downloaded before the stamp. The product size is unverified because the file
  listing needs a token.
- **Download rules.** Only through the project downloader that loads the token itself (M3 never reads the key
  file; OQ9 asks who runs it). Size check first; **cap 1 GB**; `df` before and after; the 60 GB floor applies.
- **Targets.** `log_co2`, `log_ch4`, `log_n2o`: log10 dissolved concentration. The concentration is computed
  from the headspace-equilibration records with a unit-tested reimplementation of NEON's published neonDissGas
  calculation. Field names and units are verified at implementation and recorded in the log.
- **Floors.** Rule fixed now: the smallest non-zero method detection limit in the product's laboratory records,
  per gas. The three numbers are written in a dated log entry before any fit.
- **Join.** The visit key first (`parentSampleID`), then the registered ±24 h fallback. A counts-only report is
  written before any fit.
- **Inferential rule (fixed now).** A gas is inferential only if it has ≥ 600 development rows on ≥ 25 sites
  and ≥ 15 domains, ≥ 120 holdout rows, and ≤ 30% censored development rows. If fewer than 2 gases qualify, the
  whole family is descriptive.
- **Family.** A separate Holm family of up to 3 hypotheses with the H-GEO form of claim.
- **Candidates (cap ≤ 4):** `xgb_v4clr`, `xgb_v4clr_context`, `xgb_v4mr`, and one spare declared in the log
  before it runs. Comparators: the §6 list for these targets.
- **Which look.** GAS-3 joins the sealed look `M3/test` only if (a) this file is stamped before the download,
  (b) the counts, the power file and all comparators are finished before the freeze candidate, and (c) none of
  that delays the freeze candidate. Otherwise GAS-3 is **Test-B-only**, declared as such in the log.
- **Power.** Unverified until the join. Counts only, before any fit.
- **Priority 3.** It is the first item dropped if the schedule slips.

---

## 9. Track H3 — EA multi-determinand, NEON → England (amends M3E; the registered TON analysis is unchanged)

**Practical reason.** As registered in M3E §1: a model trained on one national network and applied to another.
One determinand cannot show whether the transfer is general. Eight more can.

### 9.1 Label source and handling
- **File.** `water_chem.csv` in Thorpe et al., Zenodo record 18668846 (DOI 10.5281/zenodo.18668846; concept DOI
  10.5281/zenodo.17515762), **CC BY 4.0**. 309,159 bytes, md5 `39cf8a63480a8b1e872e329cc7839a6e`. Values are
  3-month means before sampling, taken from the EA Water Quality Archive.
- **Pin.** Size and md5 now (record metadata, evidence file in Appendix B). The **sha256 is computed at
  download** and written to `FREEZE_microbe_ea.json` (OQ7). A size or md5 mismatch stops the track.
- **Timing.** The file is downloaded only after `microbe_ea_ids.json` is frozen (M3E §2.5, §5).
- **Scorer-only.** It goes to its own empty directory and is read with `python -I` only by the scorer under
  the open claim. No modelling code can read it. Join by sample ID only, never by row order (the EA corrected
  a row-order error in a related file on 2025-10-24).
- **Header-only ID check.** A range request for the first 2,048 bytes; everything after the first newline is
  discarded before anything is shown or stored. This gives column names only. Default: run after the stamp
  (OQ7). How many ENA titles link is counted later by the scorer's count-only mode, which reads the ID column
  alone, after the ids freeze.
- **Never downloaded:** `correlations.csv`, `envfit.csv` and `titan_all.csv` from the same record. They are
  analyses of ASVs against the chemistry and would leak label information. `modules.csv`, `core.csv`,
  `diversity.csv` and `functions_ASVs.csv` are not needed.
- **Label-free files** that may be downloaded after the stamp, md5-pinned, for the H4 comparator only:
  `ASV_counts_16S_processed.csv` (75,726,804 bytes, md5 `6eb8218c37345e44d59eef7b08370542`) and
  `ASV_tax_16S_processed.csv` (2,527,388 bytes, md5 `18830ab74b0d75869c27e29ee63b9fa3`). SENTINEL's own input
  stays the windows of M3E §2.3.

### 9.2 Determinands and heads

| EA determinand (3-month mean) | NEON-trained head | Head's family |
|---|---|---|
| total phosphorus | `log_tp` | primary |
| dissolved organic carbon | `log_doc` | primary |
| conductivity at 25 °C | `log_sc` | primary |
| dissolved oxygen, mg/L | `do` | primary |
| pH | `ph` | GEO |
| chloride | `log_cl` | GEO |
| dissolved oxygen, % saturation | `do_sat` | GEO |
| ammoniacal nitrogen as N | `log_nh4` | GEO |

- These 8 form one **added Holm family**. TON → `log_no3` stays the registered analysis from the Warren file,
  with its own registered Holm rule. TON from `water_chem.csv` is a descriptive cross-check.
- **Descriptive only:** nitrate-N (the registered sensitivity), nitrite, reactive orthophosphate, silica,
  alkalinity, un-ionised ammonia, water temperature.
- EA has no elevation, so DO saturation is scored directly against the EA value (plan C10).
- **Disclosures carried from M3E:** labels are 3-month means, not same-visit; the matrix is benthic biofilm, not
  water column.

### 9.3 Models, comparators, metric, claim
- **Primary model per head:** the NEON-frozen model of that head's family (M\* or M\*_GEO) in its EA form:
  descriptors off (M3E §3), context reduced to day of year, and a +QMP model replaced by its non-QMP parent
  (EA has no qPCR).
- **EA model list (the registered cap of ≤ 6, fixed in the EA freeze):** M\*; M\*_GEO; NicheInvert-S with the
  open vocabulary (§2.1); `xgb_v4mr` (EA taxonomy by the exact-window mapping of M3E §3); the 150-bp view of
  V4-XGB; one spare. A slot that duplicates another stays empty. All but the primary model are secondary.
- **Comparators, all trained on NEON only:**
  - the **Warren et al. recipe** as the published method: RF and XGB on genus-level relative abundance.
    Hyperparameters: the paper's Table 1 values if they can be transcribed with a page reference; otherwise
    the published search space with a reduced budget (60 draws, 5 site-grouped folds, seed 42), disclosed as a
    deviation (OQ8). The repository has no licence, so its code is not copied; the recipe is reimplemented;
  - RF/XGB[genus-CLR], the stronger of tuned and untuned as frozen for NEON;
  - tier-0: GlobalMean and Ridge[DOY].
- **Metric.** Spearman ρ is primary (M3E §4). Pooled R² and group-centred R² are secondary.
- **Claim per determinand.** Δρ > 0 against **every** comparator, with the letter-group cluster CI excluding
  0, Holm over the 8 added determinands. A label is scoped to the track and the determinand, for example "SOTA
  on NEON → England rank skill for chloride".

### 9.4 Which rows are scored (default design B; OQ5)
Ruling 4a creates an EA-dev / EA-test partition. M3E registered all 22 groups in one look. Default proposed
here, declared before any EA label exists locally:
1. **Look `M3E/test`:** the frozen NEON-trained models predict **every** EA run and all predictions are
   sealed. Only the **EA-dev block (15 groups, 1,094 runs)** is scored. This is the primary H3 inference, for
   the registered TON analysis and for the 8 added determinands.
2. After that look, the scorer writes an EA-dev-only label extract. Those rows become H4 development data.
   EA-test labels stay scorer-only.
3. **Look `M3E/h4_test`:** the EA-test block (7 groups, 549 runs) is scored once, for H4 and for a
   pre-declared **replication** of H3 from the predictions sealed in step 1. The pooled 22-group H3 estimate
   is then reported as the originally registered figure. Neither can upgrade or rescue an H3 claim.

Alternatives for the lead: (A) the plan's text, one look on EA-test only for H3 and H4 (H3 then has 7
clusters); (C) H3 on all 22 groups as registered, then H4 on EA-test (the EA-test labels would be scored
twice, by different frozen models).

### 9.5 Power (no labels needed)
- EA-dev: about 1,070 linked runs in 15 groups. SE(ρ) ≈ 0.029 at ρ = 0.2 under independence, and 0.05–0.07 with
  a design effect of 3–5. For a paired Δρ the SE is plausibly 0.02–0.07, so the minimum detectable Δρ at Holm-8
  and power 0.8 is **about 0.07–0.25** (an estimate).
- The realised n, the group sizes and a design-effect bound are written label-free into `microbe_ea_ids.json`
  before scoring, as M3E §4 requires.

### 9.6 Kill rules
- Fewer than 90% of the grouped ENA runs link to an ID column of `water_chem.csv` → the added family is not
  run. The registered TON analysis proceeds unchanged.
- A determinand with < 500 linked labelled samples in the scored block, or < 10 letter groups with ≥ 10
  samples, is descriptive and leaves the Holm family. This is decided from counts and fixed in the EA freeze.
- No frozen NEON model for a head's family (for example GEO not frozen) → that head is descriptive.

---

## 10. Track H4 — EA-internal, area-blocked (ruling 4a; separate and conditional)

**Practical reason.** A national network must extrapolate to areas it has not sampled. The published benchmark
(Warren et al.) used a random 70/30 split. Re-running that recipe on the same data under area blocking is the
most direct comparison with the strongest published method.

### 10.1 Conditions (all must hold before any H4 fit)
1. This amendment is stamped and the lead has ruled on OQ6 (the M5-X key exclusion).
2. `microbe_ea_ids.json` holds the partition of §10.2 and was frozen **before** any EA label was downloaded.
3. The `M3E/test` look is finished, so the H3 models were frozen and their predictions sealed before any
   EA-dev label was released. **EA-dev rows never train an H3 model.**
4. The counts-based rules of §10.6 pass.

### 10.2 The partition (label-free; drawn 2026-11-10 from ENA run metadata only)
- **Input.** The ENA sample report of PRJEB90117: 1,649 runs, all titles unique. 1,643 titles match
  `^[A-Z]\d+_\d{4}$` and belong to 22 letter groups (A–U and X). Six titles are in neither partition:
  `Blank17`–`Blank21` and `D6300`.
- **Rule.** Groups sorted alphabetically; `numpy.random.default_rng(20261110).permutation(22)`; the first 7 are
  EA-test. The draw is accepted if EA-test holds 25–40% of the grouped runs, otherwise the seed is increased by
  1. The first draw was accepted.
- **Result.**
  - **EA-test = {A, F, I, M, N, R, X}: 549 runs (33.4%)**; by title year 198 / 192 / 159 (2021 / 2022 / 2023).
  - **EA-dev = {B, C, D, E, G, H, J, K, L, O, P, Q, S, T, U}: 1,094 runs**; by title year 486 / 372 / 236.
- **Script and output.** `scripts/data/microbe_ea_partition.py`; output and input in Appendix B. The partition
  is copied into `microbe_ea_ids.json`, together with the ids that pass read processing, before any label
  download.
- The letter group is presumed to be an EA area code. That is undocumented and is an assumption (M3E §2.5).

### 10.3 Task, development protocol, candidates
- **Task.** Predict TON and the 8 determinands of §9.2 for areas absent from training: fit on EA-dev, test on
  EA-test. Transforms and floors are those of the matching NEON heads.
- **Development.** Leave-one-letter-group-out cross-validation on EA-dev (15 folds). EA-test ids are refused by
  the development loader.
- **Metric.** Pooled R² is primary here (the level matters within one network). Spearman ρ is secondary.
- **SENTINEL candidates (cap ≤ 6):**
  1. `xgb_v4clr` fitted on EA-dev;
  2. NicheInvert-S fitted on EA-dev (day-of-year prior; master vector = the 9 determinands and water
     temperature);
  3. NicheInvert-S with NEON-informed niches (units shared with NEON are shrunk toward their NEON optimum);
  4. `xgb_v4clr` plus the frozen NEON models' predictions as features;
  5–6. spares, each declared in the log before it runs.
- **Comparators, all under the same blocking:**
  - the Warren recipe (RF and XGB) on the authors' own SILVA genus table and on our exact-window genus table;
  - RF, XGB and Ridge on genus-CLR; TabPFNv2[genus-CLR]; RF[V4-CLR]; WA-PLS on the V4 units;
  - tier-0: GlobalMean and Ridge[DOY].
  The published random-split figure is quoted for context only. It is not comparable.
- **Selection.** The candidate with the highest mean leave-one-group-out R² over the qualifying determinands;
  ties within 0.005 go to the simpler model (§11.1).

### 10.4 Test and inference
- **Test.** Fit on all of EA-dev, predict EA-test once, under look `M3E/h4_test` with its own lead-written
  freeze `FREEZE_microbe_ea_h4.json`.
- **Per determinand.** Paired ΔR² against every comparator, row bootstrap (B = 2,000), Holm over the qualifying
  determinands.
- **Across areas.** One pre-declared composite: the per-group mean difference in squared error, scaled by each
  determinand's EA-dev variance and averaged over the qualifying determinands, tested with an exact sign-flip
  test over the 7 EA-test groups at α = 0.05.
- **Claim.** "Better than <comparator> in unsampled areas of England" requires the row-level result **and**
  the composite test. "Consistent across areas" additionally requires a positive difference in ≥ 6 of the 7
  groups. Per-group results are always shown.

### 10.5 Power and its limits (counts only)
- EA-test group sizes: A 118, F 93, I 66, M 93, N 101, R 64, X 14.
- **With 7 clusters, the smallest two-sided p of a group-level sign-flip test is 2/2⁷ = 0.0156.** No effect
  size can pass a Holm correction over 8 or 9 determinands at the group level. That is why the group-level test
  is a single composite and the per-determinand inference is row-level.
- **The row bootstrap is optimistic.** Site codes are anonymised per year, so one site can recur in up to three
  years under different ids, and those rows cannot be clustered. This limit is stated with every H4 result.
- A paired-SE power file from the EA-dev cross-validation, `docs/preregistration/microbe_ea_h4_power.json`, is
  written before the H4 freeze. A determinand is confirmatory only if its minimum detectable ΔR² is ≤ 0.15.
  Otherwise it is descriptive.

### 10.6 Kill rules and firewalls
- A determinand with < 300 linked labelled EA-test samples, or < 5 EA-test groups with ≥ 20 samples, is
  descriptive. Fewer than 2 qualifying determinands → H4 is not run.
- **M5-X exclusion (ruling 4a; OQ6).** Any EA chemistry row that is a frozen M5-X test key is excluded from H4
  training. The biofilm sample ids are anonymised and carry no site or date key, so such rows probably cannot
  be identified. **H4 does not start until the lead rules how the exclusion is met.**
- **Firewall.** EA labels, EA-derived tables, H4-trained models and their predictions never leave M3 (M3E §6,
  extended to H4).
- H4 is conditional and comes after the NEON freeze candidate. It never delays the NEON look or `M3E/test`.

---

## 11. Selection rules and freeze contents

### 11.1 Selection
- **Primary family (unchanged rule, protocol §6).** Among the candidates on their tracks, the highest mean
  pooled development-LODO R² over the 8 primary targets; ties within 0.005 go to the simpler model. The rule is
  applied once, to the full candidate set, at freeze time. The candidate set gains slots #6–#8.
- **Eligibility (OQ3).** Default proposed: `xgb_v4mr` and `xgb_v4mr_stack` are eligible although they use
  taxonomy labels, and NicheInvert-S and NicheGeo-Phys are eligible as +context models under the §10.4 bar.
  Each carries its extra comparators (§2, §3, §4) and its claim wording.
- **Simplicity order for ties (fixed now):** ridge tables < tree tables < `xgb_v4mr` < `xgb_v4mr_stack` <
  NicheInvert-S < neural models (ordered by parameter count).
- **GEO family (OQ4).** M\*_GEO is selected separately by the same rule over the 8 GEO targets among G1–G10.
  M\* and M\*_GEO may differ. Both are frozen in `FREEZE_microbe.json`.
- **GAS-3:** the same rule over the qualifying gases among its ≤ 4 candidates.
- No selection uses the holdout, the reserve, Test-B or any EA label.

### 11.2 The freeze candidate (to the lead, no later than 2026-11-30; OQ12)
It contains: M\* and M\*_GEO with their tracks; the frozen strongest simple baseline per target and family;
the parity list as run; the count of configurations tried per family; the sha256 of this amendment and of its
log; the GEO data pins and power files; the import-closure hash including the GEO scorer; the outcome of the
D1 gate; and every arm not run, with the reason. Arms unfinished on that date are reported as not run. The
freeze candidate is not held back for them.

---

## 12. Every gate and kill rule in one table

| # | Rule | Consequence |
|---|---|---|
| 1 | Stamp missing (sha256 + `microbe/prereg_snapshot` + commit) | no development number on D1–D4 or on any GEO, GAS or EA target |
| 2 | Cycle 6: #4 below #3 (observed) | #5 not run; slot retired |
| 3 | D1 unit tests or train-only smoke test fail (§2.5) | no D1 development run |
| 4 | D1 gate (§2.6): mean ≥ 0.294 **and** ≥ 2 primary targets at ≥ +0.03 over the strongest simple baseline, on `lodo` | pass → slot #8 = D2; fail → D2 not run, slot #8 = D3b |
| 5 | D2 stop: after 4 GPU-h, mean < 0.294 and no GEO target ahead of its best comparator | D2 stopped and reported; D3b is **not** started in its place |
| 6 | D2 budget: 12 GPU-h, jobs ≤ 1 h 55 min, lock helper, never a GPU with another user's process | hard cap |
| 7 | D3 CPU budgets (§3.3) | D3b on `lodo` only |
| 8 | D4: qPCR record on < 90% of development or holdout events; laboratory explains > 50% of the load index | D4 not run; +QMP descriptive only |
| 9 | NH4: > 30% censored development rows (observed 20.9%) | would have been descriptive; it is inferential |
| 10 | GEO: amendment not stamped before the freeze, a required comparator unfinished, or the counts not reproduced (§7.7) | GEO on Test-B only; or stop |
| 11 | GAS-3: fails §8 inferential counts, or is not ready before the freeze candidate | descriptive; or Test-B only |
| 12 | GAS-3 download above 1 GB, or disk below the 60 GB floor | no download |
| 13 | H3: ID linkage < 90%; determinand counts (§9.6); md5 or size mismatch | added family not run; determinand descriptive; stop |
| 14 | H4: conditions of §10.1; counts of §10.6; lead ruling on OQ6 missing | H4 not run |
| 15 | H4 determinand with minimum detectable ΔR² > 0.15 | descriptive |
| 16 | Any implementation change after an arm's first development run | new counted configuration, declared first |
| 17 | Attribution control beats NicheInvert-S (§2.4) | no architecture label for the inversion |
| 18 | No dated novelty search before the freeze candidate | no novelty or first-in-class wording |
| 19 | Caps: 10 configuration × track runs on the existing tracks (OQ1); ≤ 10 GEO entries; ≤ 4 GAS; ≤ 6 EA models; ≤ 6 H4 candidates | nothing beyond the cap is run |
| 20 | Freeze candidate date 2026-11-30 | unfinished arms reported as not run |
| 21 | Resources: ≤ 6 CPU threads, one heavy CPU job at a time, 60 GB disk floor, GPU jobs ≤ 2 h | hard limits |

Rule 5 is stricter than the plan: a stopped D2 has used the slot, so D3b cannot replace it afterwards.

---

## 13. What may happen before the stamp, and what may not

**Allowed now (lead rule 2):**
- code and unit tests for D1, D3a, D3b, D4 and the parity baselines (RF[V4-CLR], TabPFNv2[V4-CLR],
  `xgb_v4clr_3000`, `xgb_gfo`, WA-tol, WA-PLS, multi-output XGB);
- the GEO target build and the QMP feature join, printing counts only;
- synthetic-data tests (in `tests/` only) and train-only smoke tests limited to shapes, finiteness and
  convergence on training rows. **No held-out or inner-fold skill number is computed or read.**
- the label-free EA read stream (approved in M3E, already queued behind W6);
- the counts scripts and the partition draw of Appendix B (done; label-free);
- everything already queued on the existing tracks: W6, the logged comparisons, the event development run.

**Unit tests owed before any development run:** the WA optimum and tolerance against a hand-worked example;
the inversion recovers planted gradients on a synthetic community; predictions do not change when held-out
labels change; missing dimensions are handled; the likelihood never decreases across accepted Newton steps;
determinism; no held-out label in any stacking feature; the GEO build reproduces Appendix A.1; the development
loader refuses protected rows; the partition script reproduces Appendix B.

**Not allowed before the stamp:** any development or validation number for D1, D2, D3 or D4 or on any GEO, GAS
or EA target; the header check of `water_chem.csv`; any download of `water_chem.csv`, of the Warren CSV or of
DP1.20097.001.

**Never, in this round:** a score on the 2024 holdout, the reserve, Test-B or EA labels outside the sealed
looks; a run of `microbe_final.py` without the lead-written freeze; an edit to a frozen file; a git commit by M3.

---

## 14. Open questions for the lead (each with the default this draft assumes)

| # | Question | Default in this draft |
|---|---|---|
| OQ1 | §10.8 says ≤ 8 configuration × track runs; #6–#8 on both tracks gives 10. Approve 10? | 10. Fallback: #7 and #8 on `lodo` only (and G4, G6 dropped) |
| OQ2 | D1 master vector with 17 dimensions instead of the plan's 14 | 17, so one fit serves both families |
| OQ3 | Are `xgb_v4mr` (taxonomy labels) and NicheInvert-S (+context) eligible for the §6 selection? | yes, with the stricter comparator sets and the stated wording |
| OQ4 | A separately selected M\*_GEO, or one model for both families? | separate |
| OQ5 | H3 / H4 look design (§9.4) and the H4 inference rule (§10.4) | design B; row-level Holm plus one group-level composite |
| OQ6 | How is the M5-X test-key exclusion met when EA biofilm ids are anonymised? | H4 waits. Options: a key list from M5-X if the header shows site and date keys; otherwise the firewall of §10.6 with the lead's acceptance; or H4 is dropped |
| OQ7 | Thorpe file: sha256 at download rather than now; header-only check after the stamp | as stated in §9.1 |
| OQ8 | Warren comparator: Table 1 values need transcribing from the paper; otherwise a reduced search (60 draws) | Table 1 if obtainable, else the reduced search, disclosed |
| OQ9 | Who runs the token-requiring DP1.20097.001 download? | the lead or the project downloader; M3 never reads the key file |
| OQ10 | D1 gate on the 8 primary targets only | yes, as the plan words it |
| OQ11 | Execution entries in the new log file, not in `microbe_protocol.md` §10 | new log file |
| OQ12 | Freeze-candidate cut-off 2026-11-30 with whatever is finished | yes |

---

## Appendix A. Counts (no label value was inspected)

Produced by `scripts/data/microbe_geo_counts.py`; stored in `microbe_arch_v2_evidence/geo_counts.json`.
"Usable" = a value, or a censored flag that the builder substitutes. Every target covers 33 sites and 19
domains in the development, holdout and `.001` 2017–18 blocks.

### A.1 GEO-8 and auxiliaries

| Column | Dev usable (values) | Dev censored | Holdout usable (values) | Holdout censored | Holdout time-matched | `.001` 2017–18 usable (censored) | Reserve usable |
|---|---|---|---|---|---|---|---|
| `lab_ph` | 1,165 (1,165) | 0 | 211 (211) | 0 | 25 | 472 (0) | 57 |
| `cl_mgl` | 1,178 (1,171) | 20 | 212 (212) | 0 | 25 | 503 (23) | 62 |
| `so4_mgl` | 1,178 (1,176) | 23 | 212 (212) | 0 | 25 | 503 (10) | 62 |
| `ca_mgl` | 1,117 (1,105) | 15 | 212 (212) | 0 | 25 | 503 (0) | 62 |
| `dic_mgl` | 1,171 (1,171) | 0 | 211 (211) | 0 | 24 | 493 (0) | 62 |
| `field_do_sat_pct` | 1,174 (1,174) | 0 | 210 (210) | 0 | 25 | 493 (0) | 49 |
| `nh4_n_mgl` | 1,182 (1,141) | 247 (20.9%) | 214 (208) | 41 (19.2%) | 25 | 493 (220) | 62 |
| `uv_absorbance_254_nm_au` | 1,182 (1,175) | 7 | 212 (210) | 2 | 25 | 506 (4) | 62 |
| `field_temp_c` (auxiliary) | 1,194 | 0 | 214 | 0 | 25 | 505 | 60 |
| `ortho_p_mgl` (D2 latent) | 1,182 (1,064) | 368 | 214 (210) | 58 | 25 | 494 (208) | 62 |
| `field_spcond_uscm` (descriptive) | 1,191 | 0 | 213 | 0 | 25 | 503 | 59 |
| `field_do_mgl` (descriptive) | 1,172 | 0 | 210 | 0 | 25 | 494 | 49 |

- Events: development 1,215; holdout 214; `.001` 2017–18 517 (514 with a chemistry parent); reserve 190 (62
  with a chemistry parent, on 22 sites and 15 domains).
- All 7 non-NH4 GEO values present together: development 1,018; holdout 202; `.001` 2017–18 423; reserve 46.
- Development rows are parent-matched except at most 1 per target.

### A.2 Detection-limit metadata (non-provisional laboratory records; smallest non-zero / median / largest)

| Analyte | Smallest non-zero | Median | Largest |
|---|---|---|---|
| Cl (mg/L) | 0.003 | 0.010 | 0.060 |
| SO4 (mg/L) | 0.006 | 0.010 | 0.090 |
| Ca (mg/L) | 0.001 | 0.001 | 0.022 |
| DIC (mg/L) | 0.025 | 0.025 | 0.200 |
| NH4-N (mg/L) | 0.002 | 0.004 | 0.050 |
| UV254 (absorbance units) | 0.001 | 0.000 | 0.001 |

### A.3 qPCR input coverage (16S, non-provisional, joined by `dnaSampleID`)

| Block | Events | With a 16S qPCR record | With record and filtered volume |
|---|---|---|---|
| development | 1,215 | 1,159 (95.4%) | 1,159 |
| holdout | 214 | 214 (100%) | 189 |
| `.001` 2017–18 | 517 | 510 (98.6%) | 483 |
| reserve | 190 | 172 (90.5%) | 139 |

2,363 non-provisional 16S records, 2 laboratories, 263 batches.

### A.4 EA letter groups (ENA titles; runs per group, 2021 / 2022 / 2023)

A 118 (40/38/40), B 135 (45/49/41), C 88 (31/29/28), D 70 (36/18/16), E 82 (38/30/14), F 93 (38/33/22),
G 71 (32/28/11), H 51 (24/19/8), I 66 (31/27/8), J 91 (33/35/23), K 91 (50/34/7), L 119 (47/35/37),
M 93 (16/33/44), N 101 (26/30/45), O 75 (16/31/28), P 62 (33/29/0), Q 50 (26/24/0), R 64 (33/31/0),
S 32 (21/11/0), T 42 (33/0/9), U 35 (21/0/14), X 14 (14/0/0). Total 1,643 grouped runs.

## Appendix B. Evidence files and scripts (sha256)

| File | sha256 |
|---|---|
| `scripts/data/microbe_geo_counts.py` | `692744722be6f0317d826d96bd445287a278ee842e2295a0c9164a731092f987` |
| `docs/preregistration/microbe_arch_v2_evidence/geo_counts.json` | `6ceacf683369e80736ddc09a5b8121d63244e7d9ec957dedd149e71c690991aa` |
| `scripts/data/microbe_ea_partition.py` | `61910b9c174f5f87f3707cc85c3a0de942647c8d2bae32ddf8325d3f2a1a7bdc` |
| `docs/preregistration/microbe_arch_v2_evidence/ena_PRJEB90117_samples.tsv` (ENA sample report, 1,649 runs) | `f8b9ee8f89aeffcfb9aa86f9be098fec1b917c77e166d64147fd7fc79c2e850d` |
| `docs/preregistration/microbe_arch_v2_evidence/ea_partition.json` (numpy 2.2.6, project environment) | `0947347cc28aa1a648abd9b74b67562f9ef4720fe08cdd6acec1e43483fc71d7` |
| `docs/preregistration/microbe_arch_v2_evidence/zenodo_18668846.json` (Zenodo record metadata, no data file) | `81acef9d1e67ddd5fcaf45d16666d32d2cd539a492708ca6ed5356784936d537` |

Warren et al. repository `Micro-Biology/Warren_et_al_-_microbial_indicators`, commit
`8931206a955975afd10c941cbf8a4590d9fe1c61`: only the two modelling scripts were read (sha256
`98dbd577eb415174843edbebe0a7e71cec156feb68618ea3afada1ad9c0dd472` and
`daa6cad753c99a43edb2b5c64df277bd2263b00b4350d066651d4239cdd4d406`). They are not copied into this repository,
because the repository has no licence. Its label CSV was not fetched.
