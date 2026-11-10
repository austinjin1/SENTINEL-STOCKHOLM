# M5 SENTINEL-BioAssess — architecture research and SOTA plan

**Date:** 2026-11-10. **Status:** research plan for lead review. It is read-only, so no model was trained, no data were downloaded and no frozen file was touched. **Scope:** M5 (benthic community → stream/lake chemistry) and its follow-on tracks.

**Inputs.** This plan builds on these files:
- `docs/handoff/bioassess.md`
- `docs/preregistration/bioassess_test_results.md`
- `docs/preregistration/bioassess_x_protocol.md`
- `docs/novelty_checks/bioassess_2026-11-09.md`
- `docs/research/04_ecotox_bioassessment.md` (§A.4, §B.3, §E.3)
- `docs/plans/sota_campaign.md`
- `RESEARCH_PLAN.md` §15.1 / §15.5
- the M5-X development outputs `$SENTINEL_RUNS/bioassess/x_track/dev/{selection_x_v1.json, dev_scores.csv, logs/}`
- the domain design "GeoNiche", supplied by the workflow

**Missing critic text.** The critic text attached to the designs was truncated in the input this step received. To recover it, I re-ran the critic's read-only diagnostic scripts on the existing M5-X development predictions (§1.3). The required changes in §4 are derived from those numbers and from the repository evidence.

**Searches.** This step's web-search budget was exhausted before it started. Every citation in §2 was therefore checked this session by a direct DOI (Crossref), arXiv or landing-page fetch, or is marked UNVERIFIED. **A dedicated novelty search for the proposed architectures is still owed** before any "novel" or "first-in-class" wording leaves the repository.

---

## 0. Decision summary (read this first)

1. **Simple ML ties because of information and sample size, not architecture.** On identical community inputs, TabPFN v2, LightGBM and RF extract about as much as BioSet. Two measurements show this:
   - **Calibration slope.** The strongest ensembles' within-region calibration slope is 0.97–1.41. An oracle linear "de-shrinkage" would add at most **+0.00 to +0.03** within-region R² (`r2_anomaly`).
   - **Training-set size.** With about 180k training rows (M5-X England), BioSet is already the **best single model** (dev `r2_anomaly` 0.313 vs 0.298). Architecture changes on the same inputs are worth about ±0.02. This is below the minimum detectable effect (MDE) of every test block we have.
2. **The current architecture is not optimal, but fixing it alone will not earn SOTA.** Its deficits are concrete:
   - 511k parameters fitted on about 3k rows in the US;
   - one assemblage only;
   - independent MSE heads with no uncertainty;
   - no regional context;
   - a noisy ±30-day grab-sample target.

   §3 sets out what an "optimal" model needs. It needs **new information**: a second assemblage, time-integrated targets, more labelled rows and regional context.
3. **Recommended path:**
   - **(a)** Leave M5-X alone. Its freeze candidate (12/12 configurations) awaits pre-test review. Development predicts SOTA on 2/12 targets, X-U orthophosphate and nitrate.
   - **(b)** Open one new pre-registered block, **France (Naïades/Hub'eau)**. It is an unseen country, the largest open EU archive pairing biology with chemistry, and it carries both macroinvertebrates and diatoms plus pesticide and metal chemistry.
   - **(c)** Build **BioSet-MA**, a multi-assemblage, multi-target, heteroscedastic set model with an episodic region-context module and a physics-coherence head, together with a **TabPFN hybrid**. Cap: ≤ 8 configurations, about 30 GPU-h in total including comparators.
4. **Fate of GeoNiche:**
   - **Not funded this cycle:** the full design (amortized classical calibration, importance-sampling refinement, empirical-Bayes region inversion). Its main mechanisms target error components that the measured diagnostics show to be small (de-shrinkage) or that the pre-registered primary metric removes (region-mean bias). Classical inversion is also the route that failed worst here: bio.infer dev `r2_anomaly` −1.08 / −1.14, and GJAM fidelity failed.
   - **Carried into BioSet-MA:** its physics operators (as a coherence head), its pool-aware absence handling and its region-adaptation idea (as an episodic context module).
5. **Harder tracks.** France F-A (leave-region-out × time-out), F-Z (zero-shot new country) and F-T (time-integrated, regulation-style annual statistics) are all scored in **one** look with one Holm family. F-E (a-priori extremes/exceedance stratum) is secondary. Pesticide/metal toxic pressure (F-TOX) is exploratory only. Each is justified on practical grounds below, before any French data exist on disk.
6. **Honest odds** (§9):
   - P(at least one new SOTA target on the France block) ≈ 0.35–0.45.
   - P(BioSet-MA *alone* is SOTA on any target) ≈ 0.10.
   - P(SOTA on at least 3 targets) ≈ 0.15.

---

## 1. Diagnosis: why simple ML wins or ties

### 1.1 Where M5 stands (verified against the files)

| Block | Best SENTINEL | Strongest comparator | Result |
|---|---|---|---|
| **M5 Test A** (NRSA 2018-19, LOEO × cycle-out, n = 1,917; single look done) | Ensemble C = mean(BioSet+geo, RF[c+g], TabPFNv2). Pooled R² TP 0.405, TN 0.514, conductivity 0.568, chloride 0.490, turbidity 0.441, pH 0.462 | Per-target baseline ensembles (mostly RF + TabPFN), pooled R² 0.400–0.556 | **SOTA as ensemble C** on chloride +0.021 [+0.013, +0.028], pH +0.024 [+0.015, +0.032] and conductivity +0.013 [+0.006, +0.019]. TN, TP and turbidity are competitive. **BioSet alone is significantly worse** on TN −0.059, conductivity −0.038, chloride −0.031 and pH −0.036 |
| M5 Test B (same regions, cycle-out) | C | `RF[geography]` is the best of 21 models on TN (0.680 vs 0.619), conductivity (0.740 vs 0.712) and chloride (0.706 vs 0.642) | The community adds nothing once the region has been sampled |
| **M5-X X-L** (NLA-2017 pseudo-test, 415 visits, 9 folds; dev, seed 0) | `ens_c_form` 0.395 (mean `r2_anomaly`) | Baseline ensemble 0.377; TabPFN 0.367; LightGBM 0.310; RF 0.289; **BioSet `_same` 0.296**; WA-PLS2 0.132; kNN-MAT 0.095; bio.infer −1.14 | Competitive on 6/6, with Δ −0.004 to +0.022 and every CI including 0 |
| **M5-X X-U** (EA 2016-20 pseudo-test, 4,788 samples, 19 folds; dev, seed 0) | `ens_top2` 0.347 | Baseline ensemble 0.325; RF[c+ctx] 0.298; UK-index RF+geo 0.295; LightGBM 0.287; **BioSet `_same` 0.313**; TabPFN 0.277 (≤ 6k rows); bio.infer −1.08 | **SOTA-expected on orthophosphate** +0.019 [+0.004, +0.034] **and nitrate** +0.033 [+0.014, +0.056]. Competitive on 4. For chloride and NH4 the per-target comparator is the UK-index regression |

Configuration counts: 13 SENTINEL configurations on M5, 12 of a cap of 12 on M5-X. **No SENTINEL configuration is left on track X.**

### 1.2 Seven reasons, with evidence

1. **The model has no information the GBM lacks** (campaign rule 5). BioSet sees the same taxa, abundances and geography as the Hellinger-feature baselines. Three proposed advantages were tested on development data and came out neutral or negative:
   - hierarchical lineage embeddings, resolution dropout and DeepSets (M5 dev);
   - UK masked-token pretraining (X-L 0.251 vs 0.262; X-U 0.274 vs 0.288);
   - the region-anomaly loss (+0.007, within noise).
2. **Small n in the US.** There are about 2,860–3,110 fit rows per LOEO fold for 511,238 parameters; the best epoch has a median of 13, so the network overfits in about 10 epochs. In this regime prior-fitted tabular models (TabPFN; Hollmann et al. 2025) and tree ensembles (Grinsztajn et al. 2022: tree models remain SOTA on medium-sized tabular data of about 10k samples) are expected to win.
3. **With large n the gap closes.** On X-U (about 175–200k training rows) BioSet `_same` is the best *single* model (0.313 vs 0.298). Data size, not architecture, was the binding constraint. One caveat: TabPFN v2 was capped at ≤ 6k rows there. Newer TabPFN releases lift that cap (TabPFN-2.5: 50k rows / 2k features), so the comparator bar is likely to rise on any new track.
4. **Shrinkage is not the bottleneck** (computed this session, read-only, on existing M5-X dev predictions, seed 0, using the critic's script `m5crit/geoniche_ceiling.py`):

   | Model (block) | Within-region slope (obs on pred) | Predicted SD / observed SD | Oracle global de-shrink gain in `r2_anomaly` | Oracle per-region stretch gain |
   |---|---|---|---|---|
   | `ens_c_form` (X-L) | 0.97–1.38 | 0.46–0.67 | 0.000–0.030 | 0.011–0.072 |
   | `ens_top2` (X-U) | 1.10–1.41 | 0.33–0.62 | 0.003–0.028 | 0.036–0.063 |
   | TabPFN (X-L) | 0.74–1.09 | 0.55–0.83 | 0.002–0.048 | 0.019–0.106 |
   | BioSet `_same` (X-L) | 0.68–0.90 | 0.61–0.87 | 0.004–0.076 | 0.034–0.108 |
   | bio.infer (X-L) | 0.23–0.39 | 1.20–2.15 | — (r2a −0.32 to −2.34) | — |

   The predictions of the strong models are roughly calibrated conditional means. Their low SD ratio reflects low R², not a correctable bias. The classical inversion (bio.infer) shows the opposite failure, over-dispersion with slopes of 0.2–0.4. That is the known variance cost of classical calibration under a misspecified forward model (ter Braak 1995).
5. **Region-level bias is the only large model-side error component, and the primary M5-X metric removes it.**
   - For `ens_c_form` (X-L), region-mean bias is 9–25 % of pooled SSE. For `ens_top2` (X-U) it is 6–23 %.
   - An oracle that knew each test region's mean would add **+0.043 to +0.091** pooled R² (X-L) and **+0.026 to +0.107** (X-U).
   - `r2_anomaly` subtracts region means, so a region-adaptation module can only help pooled R². This is the critic's key point against GeoNiche, and it shapes the metric choice in §7.
6. **Label noise from the target window.** Benthic communities integrate months of exposure, but the targets are ±30-day medians of grab samples. On UK pool rows dated ≤ 2020 (non-reserved sites; 207,579 samples at 21,466 sites; computed this session) the ±30-day median correlates with the prior-365-day mean as follows:
   - DO saturation **0.58**;
   - BOD5 **0.61**;
   - NH4 **0.74**;
   - pH **0.76**;
   - orthophosphate 0.915, nitrate 0.905, chloride 0.914, conductivity 0.957.

   For the organic-pollution variables, which BMWP/WHPT were built to indicate, the current target is a noisy proxy of what the fauna respond to.
7. **Information ceiling of macroinvertebrates for some analytes.** Best within-region skill is TP 0.40, turbidity 0.31 and NH4 0.24. Test B shows that location carries more than the community where the region is known. Diatoms, which are not used yet, are the classical nutrient and pH indicator group. Comparisons across organism groups exist: Hering et al. 2006; Johnson & Hering 2009; Marzin et al. 2012. Only their titles were verified this session; their abstracts were not retrieved, so no effect sizes are quoted.

### 1.3 Diagnosis in one sentence
On the current inputs every learner sits near the same information ceiling. What remains for SENTINEL is new information (diatoms, time-integrated targets, more labelled rows from a new country), structured outputs (multi-target, distributional) and regional context. Extra capacity, or classical inversion, on the same inputs is not the route.

---

## 2. Literature SOTA summary (citations verified this session unless marked)

Legend:
- **[DOI✓]**: Crossref record fetched this session.
- **[arXiv✓]**: arXiv page fetched this session.
- **[page✓]**: landing page fetched this session.
- **[wf]**: verified by web search earlier in this workflow session (literature step), not re-fetched here.
- **UNVERIFIED**: not checked this session.

### 2.1 Community → environment calibration (the published comparators)

| Method / paper | What it does | What matters for M5 | Verification |
|---|---|---|---|
| **WA-PLS.** ter Braak & Juggins 1993, *Hydrobiologia* 269-270:485-502 | Unimodal inverse calibration with PLS components | The canonical published comparator. Weak under spatial blocking here: M5-X dev `r2_anomaly` 0.13 (X-L) and 0.17 (X-U) | doi:10.1007/BF00028046 [DOI✓] |
| **Inverse vs classical calibration.** ter Braak 1995, *Chemom. Intell. Lab. Syst.* 28(1):165-180 | Compares kNN, PLS and WA-PLS (inverse) with classical inversion | Theory behind bio.infer, GJAM and GeoNiche. Classical inversion trades bias for variance, and our bio.infer numbers show the variance side | doi:10.1016/0169-7439(95)80048-E [DOI✓] |
| **bio.infer.** Yuan 2007, *J. Stat. Softw.* 22(3) | Per-taxon logistic niches, then maximum-likelihood inversion | Compared (package run itself, fidelity passed): dev `r2_anomaly` −1.14 (X-L), −1.08 (X-U) | doi:10.18637/jss.v022.i03 [DOI✓] |
| **GJAM.** Clark et al. 2017, *Ecol. Monogr.* 87(1):34-56 | Joint attribute model with inverse prediction | Not compared: v6 fidelity failed in both blocks (r 0.73–0.92 < 0.90) | doi:10.1002/ecm.1241 [DOI✓] |
| **sjSDM.** Pichler & Hartig 2021, *Methods Ecol. Evol.* 12(11):2159-2173 | Scalable GPU joint species distribution model | The only practical forward model for a neural classical calibration. Not recommended this cycle (§4, P6) | doi:10.1111/2041-210X.13687 [DOI✓] |
| **HMSC.** Ovaskainen et al. 2017, *Ecol. Lett.* 20:561-576 | Bayesian latent-factor JSDM | Forward model only | doi:10.1111/ele.12757 [wf] |
| **Classical inverse prediction with a multivariate GP.** Tipton et al. 2019, *Ann. Appl. Stat.* 13(4):2363-2388 | Dirichlet-multinomial plus multivariate-GP classical inversion | Closest statistical ancestor of GeoNiche. The workflow's design step reports that the modern analogue technique (MAT) gave the best *point* predictions there; that detail was not re-checked here | doi:10.1214/19-AOAS1281 [DOI✓] |
| **Tree ensembles for transfer functions.** Salonen et al. 2016, *The Holocene* 26(7):1040-1048 | RF, boosted trees and rotation forest vs WA/WA-PLS on chironomid and diatom sets | Tree ensembles win on large, noisy sets, matching M5 | doi:10.1177/0959683616632881 [DOI✓] |
| **MEMLM.** Sun, Holden & Birks 2024, *Clim. Past* 20(10):2373-2398 | Stacked RF + extra trees + LightGBM | Abstract (fetched): ML beats WA in cross-validation on larger sets, but "sometimes failed badly", possibly through extrapolation, and "cross-validation is not a sufficient measure". This supports single looks on unseen blocks and an a-priori extremes stratum | doi:10.5194/cp-20-2373-2024 [DOI✓] |
| **First ML bioindicator → chemistry study.** Džeroski, Demšar & Grbović 2000, *Appl. Intell.* 13(1):7-17 | Bioindicators → river chemistry | Prior art: the task is not new | doi:10.1023/A:1008323212047 [DOI✓] |
| **Multi-target tree ensembles.** Kocev et al. 2013, *Pattern Recognit.* 46(3):817-833 | Ensembles of predictive clustering trees for structured outputs | **A required comparator for any multi-target SENTINEL claim** (multi-output RF) | doi:10.1016/j.patcog.2012.09.023 [DOI✓] |
| **SPEAR(pesticides).** Liess & von der Ohe 2005, *Environ. Toxicol. Chem.* 24(4):954-965 | Trait-based invertebrate indicator of pesticide exposure | The published comparator for the exploratory F-TOX target | doi:10.1897/03-652.1 [DOI✓] |

### 2.2 Evaluation hazards

| Paper | Point | Verification |
|---|---|---|
| Telford & Birks 2005, *Quat. Sci. Rev.* 24(20-21):2173-2179 | Spatial autocorrelation inflates transfer-function skill under random CV | doi:10.1016/j.quascirev.2005.05.001 [DOI✓] |
| Telford & Birks 2009, *Quat. Sci. Rev.* 28(13-14):1309-1316 | Evaluation in spatially structured environments (h-block) | doi:10.1016/j.quascirev.2008.12.020 [DOI✓] |
| Juggins 2013, *Quat. Sci. Rev.* 64:20-32 | "New paradigm or sick science?": secondary gradients and confounding | doi:10.1016/j.quascirev.2012.12.014 [DOI✓] |

### 2.3 ML building blocks relevant to the proposals

| Paper | Use here | Verification |
|---|---|---|
| Lee et al. 2019, Set Transformer (ICML) | The BioSet and BioSet-MA encoder | arXiv:1810.00825 [arXiv✓] |
| Hollmann et al. 2025, *Nature* 637:319-326 (TabPFN) | The strongest single comparator on X-L; the P2 backbone | doi:10.1038/s41586-024-08328-6 [DOI✓] |
| Grinsztajn et al. 2025, TabPFN-2.5 | Abstract: "up to 50,000 data points and 2,000 features" | arXiv:2511.08667 [arXiv✓] |
| PriorLabs TabPFN repository README | Lists checkpoints v2, v2.5, v2.6, v3, v3.5. **Weights for 2.5 and later are under non-commercial licences**, unlocked by accepting terms (a token in headless use). Code is Apache-2.0. The README claims 3.5 accepts up to 1M rows and 20k features; that claim is UNVERIFIED beyond the README | https://github.com/PriorLabs/TabPFN [page✓] |
| Grinsztajn, Oyallon & Varoquaux 2022 | Trees beat deep learning on medium tabular data (about 10k samples) | arXiv:2207.08815 [arXiv✓] |
| Saerens, Latinne & Decaestecker 2002, *Neural Comput.* 14(1):21-41 | EM prior re-estimation under label shift: the basis of the transductive region adjustment | doi:10.1162/089976602753284446 [DOI✓] |
| Lipton, Wang & Smola 2018 (ICML), BBSE | Label-shift estimation with black-box predictors, the comparator-side version of region adaptation | arXiv:1802.03916 [arXiv✓] |
| Garnelo et al. 2018, Neural Processes | Episodic context-set conditioning: the basis of the P3 region-context module | arXiv:1807.01622 [arXiv✓] |
| McCleskey et al. 2012, *Geochim. Cosmochim. Acta* 77:369-382 | Conductivity computed from ions, for the P4 coherence head | doi:10.1016/j.gca.2011.10.031 [DOI✓] |

### 2.4 Multi-assemblage evidence (titles verified; abstracts not retrieved, so no numbers quoted)
- Hering et al. 2006, *Freshw. Biol.* 51(9):1757-1785. Diatoms, macrophytes, macroinvertebrates and fish compared across European streams. doi:10.1111/j.1365-2427.2006.01610.x [DOI✓]
- Johnson & Hering 2009, *J. Appl. Ecol.* 46(1):175-186. Responses of taxonomic groups to resource and habitat gradients. doi:10.1111/j.1365-2664.2008.01583.x [DOI✓]
- Marzin et al. 2012, *Ecol. Indic.* 23:56-65. Macrophytes, macroinvertebrates, diatoms and fish in French running waters. doi:10.1016/j.ecolind.2012.03.010 [DOI✓]
- Potapova & Charles 2007, *Ecol. Indic.* 7(1):48-70. Diatom metrics for eutrophication in US rivers. doi:10.1016/j.ecolind.2005.10.001 [DOI✓]

**UNVERIFIED (do not cite as-is):**
- Wohlwend et al. 2026 (*Ecography*), on DNN-JSDM transferability;
- the I2M2 index paper (Mondy et al. 2012);
- the UK WFD standard statistics (annual mean reactive P; 90th-percentile ammonia and BOD; 10th-percentile DO);
- the French physico-chemical classification rules (arrêté of 25 January 2010);
- the EU nitrate 50 mg/L threshold source;
- the TabPFN-3/3.5 papers.

**No published benchmark** matches NRSA LOEO, NLA, EA leave-area-out or a French leave-region-out design (novelty check 2026-11-09). Comparisons therefore remain re-runs on identical rows.

---

## 3. Is the current architecture optimal? (honest answer)

**No. But on the current inputs no architecture is likely to clear the bar by itself.**

| Deficit of BioSet / XBioSet | Evidence | Fix | Expected Δ vs strongest comparator, with parity |
|---|---|---|---|
| Over-parameterised for the US (511k parameters, about 3k rows) | Best epoch median 13; worse than TabPFN on X-L | Smaller trunk (about 250k), training on same-continent rows, ensembling | −0.01 to +0.01 |
| One assemblage | Diatoms are unused, although 15,878 UK invertebrate samples have a same-site diatom sample within ±30 days (computed) | Multi-assemblage tokens with modality dropout (P1) | **+0.00 to +0.05 on nutrients and pH** (comparators get diatom features too, so the *margin* is uncertain) |
| Independent MSE heads; no uncertainty | — | Heteroscedastic Student-t heads; multi-target sharing over about 15 analytes and 2 windows | +0.00 to +0.02 on sparse targets (UK chloride 37k of 77k, TP 10.8k); better event calibration |
| No regional context | Region-mean bias is 6–30 % of pooled SSE; the oracle ceiling is +0.03 to +0.11 pooled R² | Episodic region-context module (P3) | Pooled R² +0.01 to +0.03 realistic; `r2_anomaly` about 0 |
| Noisy ±30-day targets | r(±30 d, 365 d mean) is 0.58–0.76 for DO, BOD, NH4 and pH | Time-integrated targets (F-T) | Raises *everyone's* R². It does not favour SENTINEL unless multi-window multi-task sharing helps |
| Taxonomy bridging by names only | UK↔US zero-shot R² ≤ 0 (M5 T1/T2) | Lineage-sum plus indicator-value token features; pool-aware absences | Matters only for new-country tracks (F-Z) |
| Comparator TabPFN capped at 6k rows / 300 taxa | X-U TabPFN 0.277 vs RF 0.298 | **Not a SENTINEL fix: a comparator obligation.** New tracks must include current TabPFN (2.5 or later) where the licence is accepted by a human | Likely *lowers* SENTINEL's margin |

**Risk for M5-X wording (no protocol change proposed).** The frozen M5-X comparator is TabPFN 2.2.1 (v2), and TabPFN-2.5 has been public since November 2025. A reviewer may argue that the strongest published tabular method was not compared. This should be disclosed next to any M5-X SOTA label. Any new track must decide its TabPFN version policy *before* development (§8).

---

## 4. Ranked architecture proposals

### 4.0 Critic-required changes (applied to every proposal)

The critic text was truncated, so these were re-derived from the critic's diagnostics (§1.2) and from repository evidence.

| # | Requirement | How it is applied |
|---|---|---|
| C1 | The mechanism must target an error component the metric scores. Region-bias fixes are invisible to `r2_anomaly` | France tracks pre-register **pooled R² and `r2_anomaly` as co-primary**, with Holm over both (§7). The justification is fixed now: new-country deployment needs absolute levels, and M5's original endpoint was pooled LOEO R². Module P3 is judged on pooled R² only |
| C2 | Classical inversion failed here (bio.infer −1.08 / −1.14; GJAM fidelity failed) | The forward-JSDM inversion and importance-sampling refinement of GeoNiche are removed from the SOTA path (P6) |
| C3 | GeoNiche's 15-dimensional ion latent is not identifiable outside NRSA/NLA. The UK matched table has **no Ca/Mg/Na/K**, sulphate only 4,099 of 77,112 samples and hardness 19,334 | Physics is used only as a coherence head on analytes that are actually predicted (P4), never as a latent state that must be inferred |
| C4 | Compute: GeoNiche asked for 40 GPU-h | Measured from M5-X logs: one BioSet configuration × 19 X-U folds takes about **85–90 min** (about 4.5 min per fold-fit at about 180k rows); X-L `_same` × 9 folds takes 7 min. The full plan is held to about 30 GPU-h |
| C5 | Parity for transductive inputs | Every comparator gets the same unlabelled test-region community inputs as **region-descriptor features**, plus an extra comparator: the strongest baseline ensemble with a dev-fitted **region-shift correction** (BBSE / Saerens-style; §2.3) |
| C6 | The French data are UNVERIFIED (pairing rate, parameter availability) | Gate everything on an availability/pairing count before the split is frozen (§8, step 2) |
| C7 | Spent blocks | NRSA 2018-19 and 2023-24, NLA 2022 and UK 2021-26 `pool_fit` plus `reserved_test` are never used for training or new claims, **including new targets on the same rows** |
| C8 | Strongest comparator | Current TabPFN where licensed; multi-output RF (Kocev 2013) for multi-target; French index regressions as the practitioner baseline |

### 4.1 Ranking

Score = expected value for the campaign (probability of SOTA × practical value ÷ cost), on a 0–10 scale.

| Rank | Proposal | Score | GPU-h | Peak VRAM | Role |
|---|---|---|---|---|---|
| 1 | **P1 BioSet-MA**: multi-assemblage, multi-target, heteroscedastic set model | **6.5** | 17 | ~5 GB | Main SENTINEL single model and ensemble member on F |
| 2 | **P2 BioPFN**: learned set embedding plus retrieval context into TabPFN | **5.0** | 6 (+ comparator TabPFN 8–12, shared) | 8–10 GB | Most likely best *single* model; depends on the TabPFN licence |
| 3 | **P3 RegioSet**: episodic region-context module (GeoNiche's empirical Bayes, reduced to a cheap form) | **4.5** | 3 (an ablation inside P1) | +1 GB | Pooled-R² lever with a measured ceiling |
| 4 | **P4 Coherence head** (GeoNiche-lite physics) | **3.5** | 1–2 (or CPU post-hoc) | — | Consistent multi-analyte outputs; small accuracy gain |
| 5 | **P5 ΔBio-Siamese**: change detection between revisits | **2.5** | 3 | ~3 GB | Exploratory secondary track |
| 6 | **P6 GeoNiche (full)**: amortized classical calibration | **1.5** | 40 (design estimate) | 3.5 GB | **Not funded this cycle.** Optional ≤ 4 GPU-h simulation-fidelity probe only |

### 4.2 P1 — BioSet-MA (BioSet v3), the recommended main model

**One line.** A set transformer over 1–3 assemblages (invertebrates, diatoms; macrophytes optional in France). It is trained jointly on invertebrate-only, diatom-only and paired samples through modality dropout, and predicts about 15 analytes in 2 time windows with Student-t predictive distributions.

**Architecture** (about 250–350k parameters; fp16 autocast plus GradScaler, because the 2080 Ti has no bf16):

1. **Tokens.**
   - Each token = flat OTU embedding (d = 96) + lineage-sum embedding (genus→phylum, through `taxonomy_harmonised` extended with the SANDRE lineage) + **modality embedding** + the 4 existing abundance features + 1 **pool-prevalence** feature (the taxon's prevalence among training-pool samples within 200 km).
   - Optional **indicator-value channel:** the taxon's published sensitivity score, such as the WHPT taxon score or diatom TDI sensitivity, with a mask bit when it is missing. This is never chemistry.
   - Caps: ≤ 150 invertebrate tokens and ≤ 100 diatom tokens (most abundant kept).
2. **Per-modality encoder.** 2 pre-LN SAB layers, 4 heads, weights shared across modalities. Pooling is PMA(1) ⊕ abundance-weighted mean, giving one vector per modality.
3. **Fusion.** A 1-layer transformer over [modality vectors, context token, optional region token]. Modality dropout p = 0.3, always keeping at least one modality. The context token carries day of year (sin/cos), source/method, and lat/lon/elevation plus stream size and slope where available for *all* rows.
4. **Trunk.** 2 × 256 GELU layers, dropout 0.1.
5. **Heads.** Per target, a Student-t (ν = 4) mean and log-scale with a per-source scale offset. Targets: §5 primary + secondary, in windows {±30-day median, prior-365-day mean} where labels exist. The loss is masked, so a sample contributes whatever analytes it has.
6. **Optional modules (ablations).** P3 region token, P4 coherence penalty.

**Training.**
- AdamW, lr 1e-3, weight decay 1e-2, batch 256, token dropout 0.1.
- Inner validation: 15 % held out by 50-km blocks. Early stopping with patience 10 on inner-validation NLL.
- Seeds 0, 1 and 2.
- Same-continent training rows only (UK + France) for the France block. This is fixed a priori from the M5-X lesson that `_same` beat `multi`, not chosen on French results.

**Inductive bias a GBM lacks (honest):**
- (i) **Joint use of incomplete multi-modal, multi-target data.** It trains on 77k invertebrate-only, 16.5k diatom-only and 15.9k paired UK samples at once, sharing a representation across about 30 sparse target columns. A per-target GBM cannot share across targets; multi-output RF can, but only on complete target vectors.
- (ii) **Distributional outputs** give event probabilities without a second model.
- (iii) **Permutation-invariant attention over present taxa only,** which avoids reading out-of-range absences as evidence.

All the *inputs* can be given to baselines, so the margin comes only from (i) to (iii). With large n, BioSet was already the best single model on X-U.

**Fair test plan.**
- Identical folds and rows.
- Comparators get the same diatom Hellinger features (taxa in ≥ 10 training samples), the same context and the same pool-masked absences (NaN instead of 0 for out-of-pool taxa).
- Required comparators:
  - WA-PLS2, kNN-MAT, bio.infer (run itself, through its own OTU pipeline);
  - RF, LightGBM and multi-output RF;
  - TabPFN v2 2.2.1, plus TabPFN-2.5 if a human accepts its licence;
  - French index regressions (I2M2 and its metrics, IBD; ± geography);
  - region mean, RF[geography], nearest reference site;
  - baseline ensembles.
- Per-target comparator = the best comparator on F development data.
- Claims: paired location-cluster bootstrap with B = 10,000, Holm, 3 seeds, single look.
- **Kill rule (pre-registered).** If the P1 single model is more than 0.02 below the best single comparator on at least 4 of 6 primary targets on F development data, it stays only as an ensemble member.

**Budget.**

| Item | GPU-h |
|---|---|
| Smoke run | 1 |
| Development: 5 configurations × about 7 folds × 1 seed × about 0.15 h | 5.5 |
| 2 extra seeds for the top configuration | 2 |
| Test pass: 3 seeds × 7 folds | 3.5 |
| Margin for France-scale token counts (about 2× X-U per fit) | 5 |
| **Total** | **about 17** |

Peak VRAM is about 5 GB at batch 256 with 250 tokens, so it fits beside another job on an 11 GB card if the lock helper allows.

### 4.3 P2 — BioPFN (hybrid with a tabular foundation model)

**One line.** TabPFN fed with [top-K Hellinger taxa per modality (K ≤ 400 invertebrate + 300 diatom, within the version's feature limit), a 32-dimensional BioSet-MA sample embedding trained on the fold's training rows only, context], with an in-context training set built from **retrieval**: the 50 % of context rows nearest in embedding space plus a 50 % stratified random sample, up to N = 10k rows (v2) or ≤ 20k (2.5; VRAM on 11 GB UNVERIFIED).

**Why it might win.** TabPFN is the strongest single comparator on X-L. A learned community embedding and retrieval give it information from far more than its context cap. The novelty is modest: it is an engineering hybrid.

**Fair test plan.**
- Parity comparator: the same TabPFN version, same K and same context size, with **PCA(32) of the Hellinger features** in place of the learned embedding and random (non-retrieval) context.
- Second comparator: the same plus kNN-retrieved context, which isolates the embedding's contribution.

**Budget.** About 6 GPU-h for SENTINEL fits (development plus test). The TabPFN comparators are budgeted separately at about 8–12 GPU-h. VRAM is 8–10 GB, so P2 runs alone on a card.

**Blocker.** TabPFN-2.5 and later weights need licence acceptance and a token. Agents must not handle credentials, so the lead or user does this, or the track documents "v2.5 not compared, reason: licence".

### 4.4 P3 — RegioSet: episodic region context (GeoNiche's empirical-Bayes idea, simplified)

**Mechanism.**
- A region token is computed by attention pooling over the embeddings of ≤ 256 *unlabelled* samples from the same region or hydro-ecoregion (inputs only; the sample itself is excluded).
- It is trained episodically on **pseudo-regions** (random spatial clusters of training rows, 50–150 km), giving thousands of episodes instead of 9–19 real regions (Neural-Process style; Garnelo et al. 2018).
- At test time the token uses the unseen region's own unlabelled community inputs. Labels are never used.

**Ceiling (measured).** The oracle region-mean gain is +0.043–0.091 pooled R² (X-L) and +0.026–0.107 (X-U). Recovering 20–40 % of it would give **+0.01 to +0.03 pooled R²** and about 0 on `r2_anomaly`.

**Parity.** Comparators get PCA-20 of the region's mean Hellinger profile as features, computed the same way and also over pseudo-regions. There is also the region-shift-corrected baseline ensemble (C5). If P3 cannot beat that corrected ensemble on development data, it is dropped.

**Budget.** About 3 GPU-h as one ablation configuration inside P1.

### 4.5 P4 — Coherence head (what survives from GeoNiche's physics)

- **Soft constraints** on predicted means where both sides are predicted:
  - TN ≥ NO3-N + NH4-N;
  - TP ≥ PO4-P;
  - log conductivity consistent with the predicted ions (US rows only, using the McCleskey 2012 λ-sum; the design's check gave r = 0.998 on NRSA/NLA development data);
  - a monotone pH–alkalinity/ANC link.
- Penalty weight λ = 0.1, fixed a priori. A post-hoc projection variant needs no GPU.
- **Value:** no physically impossible outputs, and coherent multi-analyte reports for practitioners. Expected accuracy gain is below 0.01, so it is not a SOTA lever. It is one configuration (1–2 GPU-h) or CPU-only.

### 4.6 P5 — ΔBio-Siamese change detector (exploratory)

- **Target:** the change in the prior-365-day mean (NH4, BOD5, DO, pH, orthophosphate) between two visits at least 1 year apart at the same station.
- **Inputs:** both communities, through a shared encoder; the output is the change.
- **Practical motivation:** "did this river get better or worse?", which is how agencies assess trends.
- **Feasibility (computed).** On UK sites with at least 3 samples, the within-site share of variance in annual means is:
  - nitrate 0.04, conductivity 0.02, chloride 0.06, orthophosphate 0.10;
  - NH4 0.20, pH 0.17, DO 0.28, BOD5 0.29.

  Only the organic-pollution variables carry usable within-site signal.
- **Comparators:** GBM on [H₁, H₂ − H₁]; ΔWHPT-ASPT and ΔI2M2 regressions.
- **Status:** secondary or descriptive only. About 3 GPU-h.

### 4.7 P6 — GeoNiche full: not funded this cycle (reasons)

1. Its two headline mechanisms are de-shrinkage through classical calibration and region empirical-Bayes adaptation. On the strongest ensembles the measured de-shrinkage ceiling is ≤ 0.03 `r2_anomaly`. The region effect is removed by the M5-X primary metric, and P3 captures it at a fraction of the cost.
2. Classical inversion failed here: bio.infer −1.08 / −1.14, and GJAM fidelity failed. It amplifies noise (slopes 0.2–0.4) whenever the forward model is misspecified, and a 2,500-OTU Gaussian-logit JSDM is misspecified.
3. The latent is not identifiable without major ions, which the UK and possibly France lack.
4. It needs 40 GPU-h and has large implementation risk: an implicit-differentiation pH solve, IWAE, importance-sampling ESS collapse and tempering.

**Kept:** the physics operators (P4), pool-aware absences (P1), the region-adaptation idea (P3) and the simulation-recovery fidelity gate. The gate may be run as a ≤ 4 GPU-h research probe if the lead wants an interpretability result. It confers no SOTA path.

---

## 5. New numeric predictive targets (multi-task)

All targets are numeric, with predictive distributions (mean plus 5/50/95 % quantiles) where the model supports them. Counts marked "computed" were computed this session on allowed rows. France counts are UNVERIFIED until the availability check.

| Target | Label source | Count of labelled samples | Why | Track |
|---|---|---|---|---|
| **France primary (6):** pH, log conductivity, log chloride, log NO3-N, log NH4-N, log PO4-P | Naïades physico-chemistry, same SANDRE station, ±30 d median | UNVERIFIED; ≤ 9,703 test operations 2023-25, about 50k I2M2-era operations in total (workflow data-step counts) | Comparable to X-U; TP or suspended solids replace any target with too little coverage, under a pre-written rule | F-A, F-Z |
| **Organic pollution:** DO saturation, BOD5, nitrite | UK EA matched (on disk); Naïades | UK ±30 d: DO 73,709, BOD5 45,632, nitrite 69,275 (all years); ≤ 2020 allowed: DO 48,265, BOD5 34,054 (computed) | The classic BMWP/WHPT target domain, and currently unused | F-A secondary; UK development |
| **Time-integrated (regulation-style):** prior-365-day mean of pH, orthophosphate, NH4, NO3, BOD5 and DO, plus P90 for NH4/BOD5 and P10 for DO (recomputed from `wq_observations`) | `chemistry_matched_long.parquet` window `prior365d` (on disk); Naïades | UK ≤ 2020 allowed rows with ≥ 6 observations: pH 42,171, orthophosphate 40,912, NH4 42,216, NO3 36,303, BOD5 31,298, DO 41,503, chloride 23,752, conductivity 19,194 (computed) | Fauna integrate exposure over months. r(±30 d, 365 d) is only 0.58–0.76 for DO, BOD, NH4 and pH. WFD classifications use annual statistics (exact UK/FR rules UNVERIFIED; fix them from official sources in the protocol) | **F-T** |
| **Diatom-informed nutrients:** orthophosphate, TP, NO3, pH, alkalinity | UK diatom ±30 d matched chemistry (on disk) | 16,554 diatom samples (pH 16,039; orthophosphate 15,705; NO3 15,672; alkalinity 13,492; TP 4,658). 15,878 invertebrate samples have a same-site diatom sample within ±30 d (computed) | A second assemblage adds new information | F-MA stratum; UK development |
| **Major ions, ANC, DOC, SO4, TSS, silica** | NRSA/NLA chemistry (on disk) | NRSA chemistry has 8,831 visits, with ions ≥ 99.8 % complete on development cycles | Multi-task supervision; P4 coherence | US development only (no unseen US block before NRSA 2028-29) |
| **Fine sediment:** `PCT_SAFN`, `LRBS_USE` | NRSA PHAB (on disk, 8,801 visits) | Development cycles 2008-14 only | Sediment is a major invertebrate stressor; the UK PSI index exists for it | US development only; no unseen block now |
| **Exceedance events:** Cl > 230 mg/L, TP > 0.1 mg/L, pH outside [6.5, 9], turbidity > 50 NTU (as in M5-X v8), plus NO3 and PO4 thresholds | Derived from the targets above | Power check required (≥ 30 positives and ≥ 10 clusters) | Practitioner-relevant; MEMLM warns about extremes | **F-E** (secondary) |
| **Toxic pressure:** insecticide sum of toxic units; Cu and Zn | Naïades pesticide and metal analyses, plus M4 ChemTox / ECOTOX toxicity values | UNVERIFIED: censoring and detection limits dominate | Pesticides are rarely monitored; SPEAR is the published comparator; links to M4 | **F-TOX** (exploratory, descriptive unless powered) |

---

## 6. More data: sources, licence, size

| Source | Status | Licence | Size / counts | Use | Notes |
|---|---|---|---|---|---|
| **UK EA diatoms + extra analytes + prior-365-day windows** | **On disk, unused** | OGL v3.0 (existing manifest) | 28,148 diatom samples; 16,554 with ±30 d chemistry; 7.65M WQ observations | UK development for P1 (≤ 2020 pool rows only) | Zero download; start here |
| **France Naïades / Hub'eau Hydrobiologie** (invertebrates, diatoms, macrophytes, fish) | Not on disk. API page fetched this session [page✓] | Hub'eau terms: reuse "régie par la licence ouverte Etalab", with attribution [page✓]; the Naïades footer says etalab-2.0. Record the licence per file | Workflow data-step counts (2026-11-10): 21,857 stations; 6.92M invertebrate taxon records; 59,922 I2M2 operations (9,703 in 2023-25); 83,552 IBD operations. The API caps queries at 20k records deep, so use the bulk export | Training and the **unseen F block** | Validated year n−1 data are published by autumn of year n [page✓], so 2025 chemistry should be available now |
| **France Naïades / Hub'eau physico-chemistry** (including pesticides and metals) | Not on disk. API page fetched [page✓] | Etalab (as above) | National volume UNVERIFIED. Download only biology stations × {±30 d, prior 365 d} × about 25 parameters plus the pesticide/metal panel. Estimated 0.2–0.8 GB as parquet (UNVERIFIED) | Labels | Request a **≤ 3 GB** M5-F cap; run `df -h` first (86 GB free today) |
| **Sweden SLU MVM** (lakes and watercourses: chemistry and biology) | Landing page fetched [page✓] | **Not stated on the page:** UNVERIFIED | 1,377,628 samples; 21.9M values; 51,508 stations (page counts, 2026-11-10). Benthic fauna not confirmed on that page | A possible second new-country block later | Licence and benthic-fauna check before any plan |
| **Canada CABIN** | Open-data record fetched [page✓] | Open Government Licence – Canada; non-federal studies need permission | 33 CSVs (study, habitat and benthic data per drainage area); updated monthly; record modified 2026-01-15 | Zero-shot block, **only if** chemistry is present | The record says water chemistry is **not** in the listed files, so it is probably unusable. A one-file habitat CSV probe would settle it |
| StreamCat at NRSA sites | Pending COMID/NLDI lookups. `processed/catchment/streamcat_sites.parquet` currently covers 6,516 NEON/NWIS sites, not NRSA | As in docs/research/04 | — | US development only | Low priority: no unseen US block before NRSA 2028-29 |
| TabPFN-2.5 (or later) weights | README fetched [page✓] | **Non-commercial**; licence acceptance plus token | — | Comparator (and P2 backbone) | Needs a human licence decision |
| Ireland EPA, SEPA (Scotland) | UNVERIFIED | UNVERIFIED | UNVERIFIED | — | Not plannable without access checks |

**Reservation needed.** M6 listed France for hydromet (sensors and discharge, a different dataset). The lead should record an **M5-F reservation** of French biology and physico-chemistry test rows before download. The EA sonde allocation (`ea_england_allocation.md`) covers the Hydrology API, not the WQ-archive discrete samples M5 uses, so there is no conflict.

---

## 7. Harder tracks: practical justification, unseen block, power

All F tracks are written into **one** protocol and **one** frozen split, and scored in **one** single look. The confirmatory Holm family is F-A + F-Z + F-T primary targets × {pooled R², `r2_anomaly`}. F-E, F-MA and F-TOX are secondary. The justifications below are written now, before any French row is on disk.

| Track | Practical justification (a priori) | Unseen test block | Training | Power (to be recomputed from availability masks) |
|---|---|---|---|---|
| **F-A: new national network, leave-region-out × time-out** | Agencies want chemistry where only biology is sampled. France uses a different taxonomic system (SANDRE) and protocol (I2M2), so it tests generalisation to a new network | France 2023-25 operations with paired chemistry. Folds: about 8 aggregated hydro-ecoregion groups, or the 6 metropolitan water-agency basins if HER is unavailable; fixed in the protocol | UK (allowed pool rows) + France ≤ 2021, outside the fold region; 1-km buffer; 2022 buffer year | Scaled from X-U (MDE 0.015–0.024 at 2,149 clusters): about 0.013–0.020 at about 3,000 clusters, about 0.018–0.029 at about 1,500 (assumes similar per-cluster variance) |
| **F-Z: zero-shot new country** | Many countries have biomonitoring but sparse chemistry | The same France 2023-25 rows (same look; in the Holm family) | UK + US labelled rows only. French *unlabelled* communities allowed for P3 region context and the matching comparator descriptors | Same n. Expect low R² for every model, so a SOTA claim is possible only if one model transfers at all |
| **F-T: time-integrated regulatory statistics** | WFD classification uses annual statistics, and fauna integrate months of exposure | The same rows; targets are prior-365-day mean/P90/P10 (needs at least 6 observations) | As F-A | n is reduced by the ≥ 6-observation filter (UK ≤ 2020: 77–92 % of ±30 d rows retain it, computed) |
| F-E: extremes/exceedance stratum (secondary) | Exceedances drive management. MEMLM warns that ML fails when extrapolating | Rows above the training-pool P90 per target; event set from §5 | As F-A | Power flag F/U by the M5-X v8 rule (≥ 30 positives, ≥ 10 clusters) |
| F-MA: multi-assemblage stratum (secondary) | Agencies co-sample invertebrates and diatoms | F-A rows that have both assemblages | As F-A | UNVERIFIED (co-sampling rate in France) |
| F-TOX: toxic pressure (exploratory) | Pesticide monitoring is sparse; SPEAR exists | F-A rows with a pesticide panel | As F-A | Likely underpowered; descriptive unless ≥ 1,000 non-censored rows |
| UK-P: prospective England (later) | The truest unseen block (future samples) | EA invertebrate samples dated after the 2026-10-20 snapshot at `pool_fit` sites | — | About 3.5k samples per year; not usable until about mid-2027 |

**Never used** (spent or reserved): NRSA 2018-19 and 2023-24; NLA 2022; UK 2021-26 `pool_fit` and `reserved_test` invertebrate samples, including for new targets or diatom tracks at those sites and dates. Other workstreams' frozen rows go through `xtrack.frozen_test_keys()` or its France analogue.

---

## 8. Sequencing within the shared 2-GPU budget, caps and pre-registration

**Budget assumption.** About 7 GPU-h per day for M5 (a 2/7 share of GPUs 0 and 1). Every job ≤ 105 min through the lock helper; ≤ 6 CPU threads.

| Step | What | Compute | Gate before the next step |
|---|---|---|---|
| 0 | **M5-X:** pre-test review → freeze → single look (lead). Nothing in this plan touches it | Lead's GPU slot | Independent of this plan |
| 1 | **Pre-register M5-F protocol v0** before any French download | 0 | Lead approval |
| 2 | France **availability and pairing count**: download biology plus the chemistry subset, then masks only (no values summarised beyond `notna`) | CPU; ≤ 3 GB | Fix targets, folds and test rows; **freeze the split (sha256)**; power check |
| 3 | In parallel with step 2: **UK development of P1** on ≤ 2020 pool rows (invertebrate + diatom, multi-target, Student-t). Counted toward the F cap | about 3 GPU-h | P1 smoke tests pass; VRAM and time measured |
| 4 | F development baselines: RF, LightGBM, multi-output RF, WA-PLS2, kNN-MAT, bio.infer, French index regressions, region mean, RF[geography] | CPU, about 1–2 days | Development table |
| 5 | F development TabPFN v2 (+ 2.5 if licensed) | 8–12 GPU-h | — |
| 6 | F development SENTINEL: P1 base, P1+MA, P1+P3, P1+P4, P2, then 2 ensembles (C-form-F, top-2) = **8 configurations (cap 8)**; ablations use 1 seed, the top configuration 3 seeds | about 12 GPU-h | Selection by the pre-registered rule; kill rules (§4.2, §4.4) |
| 7 | Freeze candidate → independent pre-test review → lead freeze → **single look** with the xtestlook guard | about 6 GPU-h | — |

**Total:**
- **About 30 GPU-h:** SENTINEL about 21 (P1 17 + P2 about 4 on F) and TabPFN comparators 8–12.
- **About 5–7 working days** wall-clock at the assumed share, plus data access time.

**What must be pre-registered first (protocol v0, before download):**
1. The practical justification for F-A, F-Z, F-T, F-E, F-MA and F-TOX (§7 text), dated.
2. Block definitions:
   - the test period (2023-25), buffer year 2022, development pseudo-test 2019-21 with training ≤ 2017;
   - the fold rule (aggregated HER, or water-agency basins as fallback);
   - the 1-km buffer; same-continent training for F-A.
3. The target-selection rule from availability masks only, including replacement rules and the time-window definitions (±30 d median; prior-365-day mean/P90/P10 with ≥ 6 observations).
4. **Co-primary metrics: pooled R² and `r2_anomaly`.** Holm over the full confirmatory family; B = 10,000; ≥ 3 seeds; the decision table from the M5-X pattern (SOTA / competitive / inconclusive by realised MDE).
5. The comparator list and **TabPFN version policy:** v2 2.2.1 always; 2.5 if a human accepts the licence before step 5, otherwise "not compared, with reason". No version change after development starts.
6. The configuration cap (8) and the kill rules.
7. The data cap (≤ 3 GB) and the lead's M5-F France reservation.
8. Exceedance thresholds, taken from official sources and recorded with their URLs (currently UNVERIFIED).

**Do not:**
- add configurations to M5-X;
- define new targets on spent rows;
- choose between pooled R² and `r2_anomaly` after seeing French results;
- swap in a newer TabPFN after development;
- read French test-period chemistry values before the freeze (masks only).

---

## 9. Honest outlook

Probabilities are judgement calls informed by M5 and M5-X dev→test behaviour. They are not computed.

| Track / claim | P(SOTA on ≥ 1 target) | P(SOTA on ≥ 3) | Main risk |
|---|---|---|---|
| M5-X single look (already frozen-candidate; not part of this plan) | about 0.45 (development predicts 2/12) | about 0.10 | Development→test shrinkage; the TabPFN-version critique |
| F-A with ensemble (P1 + RF + TabPFN) | **about 0.35–0.45** | about 0.15 | TabPFN-2.5 raising the bar; pairing rate; MDE about 0.02 vs expected Δ +0.01–0.03 |
| F-A, P1 single model | about 0.10 | < 0.05 | GBM parity on the same inputs |
| F-Z zero-shot | about 0.20 | < 0.05 | Every model near 0 (UK↔US was ≤ 0) |
| F-T time-integrated | about 0.30 (correlated with F-A) | about 0.10 | Higher R² for everyone; a margin is not guaranteed |
| F-E extremes (secondary) | about 0.15 | — | Underpowered strata |
| F-TOX (exploratory) | about 0.05–0.10 | — | Censoring; power |
| P5 change detection | about 0.05 | — | Within-site signal of only 2–29 % of variance |
| GeoNiche full | < 0.05 this cycle | — | Diagnostics (§1.2), compute, identifiability |

**Bottom line.**
- The likeliest new SOTA labels are again **ensemble-scoped** and modest (+0.01 to +0.03 R²), now on a fresh, larger and more practically meaningful block with more targets.
- The most valuable outputs are probably scientific, whatever the labels:
  - (i) whether a second assemblage and time-integrated targets raise the achievable skill;
  - (ii) whether the community signal transfers to a third national network;
  - (iii) calibrated exceedance probabilities.

  All three are publishable whether or not the SOTA bar is met, provided every track is reported.

---

### Appendix A: diagnostics run for this plan (read-only; no fitting)

- **Calibration and region bias.** `.../scratchpad/m5crit/geoniche_ceiling.py` and `geoniche_between.py`, written by the critic step and re-run here. They read `$SENTINEL_RUNS/bioassess/x_track/dev/preds/*__s0.parquet` and pool-row truth (NLA 2017 / EA ≤ 2020 pseudo-test rows only).
- **UK target-window statistics.** Inline Python over `processed/uk_ea_ecology/{samples_invertebrate,chemistry_matched_long,sites}.parquet`, restricted to samples dated ≤ 2020 at `pool_fit`/`pool_val` sites (`uk.uk_site_roles`). No `reserved_test` and no 2021-26 rows were read.
- **Diatom co-sampling count.** Same-site invertebrate and diatom samples within ±30 days, across all years (sample metadata only; no chemistry values).
- **Fit timings.** `x_track/dev/logs/gpu_chain*.log` ("done …" timestamps).
