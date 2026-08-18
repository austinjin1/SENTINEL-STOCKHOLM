# SENTINEL's Transfer Gap: In-Distribution vs Out-of-Distribution, and How Transferable Representations Shrink It

*Cross-cutting analysis. All numbers are real, leakage-free, and copied verbatim from the per-model result JSONs under `results/`. Companion machine-readable file: `results/transfer_gap_analysis.json`. 2026-07-03.*

## The one finding

Every SENTINEL model looks strong under **in-distribution (ID)** evaluation — random split, within-study CV, with-geography, within-region — and drops sharply **out-of-distribution (OOD)** — geographic holdout, leave-one-study-out, no-geography, cross-region. This gap is the project's central scientific result. The fixes that reduce it all share one property: they are **representations built to be invariant to the deployment shift axis** (coarse taxonomy, per-sample rank-norm, the physically-correct red-edge band, non-redundant modality fusion).

## The transfer-gap table

| Model | Metric | ID setting → score | OOD setting → raw score | Transferable-representation fix → OOD score | Gap recovered |
|---|---|---|---|---|---|
| **MicroBiomeNet** | macro-F1 | random split **0.831** | geographic holdout (raw ASVs) **0.405** | multi-rank taxonomy CLR **0.676** | **+0.272 (~64% of gap)** |
| **ToxiGene** | AUROC | within-study CV **0.858** | LOSO cross-study (global-z genes) **0.557** | per-sample rank-norm **0.642** [CI 0.564–0.735] | **+0.085 (~28% of gap)** |
| **SpeciesHealthIndex** | macro-AUROC | with geo range-prior **0.944** | environment-only, no geo **0.853** | (biotic community context — no lift) **0.853** | gap = **0.091** (honest, range prior is the missing input) |
| **HydroViT** | chl-a R² | (station-holdout mean R² 0.441) | 4-band RGB+NIR **0.178** | + red-edge B05/NDCI/CIre **0.320** | **+0.143 OOD lift** [CI 0.100–0.188] |
| **Microcystin fusion** | AUROC | within-region 3-way **0.823** | cross-region (leave-region-out) **0.485** | + domain descriptors **0.527** | +0.042 (~12%); genotype key **DATA-BLOCKED** |
| **AquaSSM Track-A** | held-out FPR @matched op-pt | per-site baseline **9.7%** | non-stationary clean periods | rolling-CUSUM change-point **6.9%** | validated episodes 8/12 → **10/12** |

**Mean fraction of the ID→OOD collapse recovered by the feature fixes** (the three benchmarks with a like-for-like gap AND a feature fix — MicroBiomeNet, ToxiGene, Microcystin): **~35%**. The fixes are real, measured, statistically-supported wins — none closes the gap entirely, which is itself the honest message.

## (a) How much do the fixes reduce the gap? — quantified

- **MicroBiomeNet — taxonomy.** Exact 90-bp EMP ASVs are region-specific fingerprints; the same environment on different continents is dominated by different ASVs of the *same higher taxa*. Aggregating to phylum→genus CLR features cuts the geographic gap from 0.426 (0.831→0.405) to 0.155 (0.831→0.676), recovering **~64%**. A same-split raw-OTU GBM control lands at 0.571 — so the +0.272 gain is the **feature representation**, not the classifier family.
- **ToxiGene — rank normalization.** LOSO fails because top-variance genes under global z-scoring are dominated by cross-study scanner/array batch geometry, not exposure biology. Per-sample rank/quantile normalization equalizes each array's marginal distribution and lifts LOSO AUROC 0.557→0.642 (95% study-bootstrap CI [0.564, 0.735] clears both chance and the 0.55 floor); **8/12** held-out studies improve, stable across the regularization sweep. It is per-sample and label-free, hence trivially leakage-free. ComBat and real KEGG-pathway scores gave *smaller* gains — a measured, non-obvious finding.
- **HydroViT — red-edge.** The shipped 4-band (RGB+NIR) representation is blind to the ~705 nm chlorophyll red-edge. Adding B05 + NDCI/CIre lifts station-holdout chl-a R² 0.178→0.320 (Δ +0.143, CI [0.100, 0.188], P(improve)=1.0). mc>1.6 AUROC also +0.026 (CI excludes 0); mc>0.3 is null. This is the physically-correct band doing physically-expected work under a genuine OOD (station) split.
- **Microcystin — fusion, partial.** Within-region, 3-way fusion (optics + pigment + drivers) genuinely wins: **0.823 > 0.776 (2-way) > 0.684 (best single)**, driver gain CI excludes zero, negative control collapses it. But cross-**region** it falls to chance (0.485), and domain descriptors only nudge it to 0.527. The literature's transfer key — toxic-strain genotype (mcyD) — is **not co-located** with microcystin measurements on disk (0 of 227 fusion sites have a cyanobacteria genotype record). Honest DATA-BLOCKED null, not a failure to fabricate a win.

## (b) The honest reporting + deployment framework

Grounded in the domain-generalization / covariate-shift literature:

1. **Headline the OOD number.** Random/per-sample splits overstate ecological AUC by up to ~0.16 and MAE by 70–80% under spatial autocorrelation (spatial-blocking SDM literature). MicroBiomeNet's 0.831→0.405 is exactly this inflation. Report the worst realistic-shift number as the headline; list the random-split number only for honest comparison.
2. **Name the shift axis and enforce zero leakage on it.** Geography (k-means regions), study (LOSO), region (leave-region-out), station (GroupKFold), or time — state which, verify zero leakage, and report ID *and* OOD together, never OOD-only-when-flattering.
3. **Prefer representations invariant to that axis.** Coarse taxonomy over exact ASVs; per-sample rank-norm over global z-score; physical spectral indices over raw bands; modality fusion only where modalities are genuinely non-redundant (microcystin yes; chl-a/DO no — those are single-modality-dominated by physics).
4. **Per-domain calibration + conformal uncertainty at deployment.** Calibrate per new region/study/station with a small labeled set, and emit distribution-free prediction sets via conformal prediction. Because a new domain violates exchangeability, use **weighted conformal prediction** (Tibshirani et al. 2019) with importance weights estimated from *unlabeled* target features, keeping sets valid under covariate shift.
5. **Report worst-group, not just the mean.** Aggregation hides OOD failure: per-class `freshwater_sediment` F1 = 0.0 and per-study sub-chance AUROC vanish inside a macro average. Report worst-region / worst-study alongside the pooled mean.

## Honest verdict

The transfer gap is real and universal across SENTINEL. The transferable-representation fixes are genuine, leakage-free wins that recover a meaningful (~35% mean) fraction of the ID→OOD collapse — but **none fully closes it**, and two settings (Species range prior; Microcystin cross-region genotype) are honest gaps/nulls where the fix is unavailable or data-blocked. SENTINEL's contribution is not any single model beating a baseline; it is the demonstration, on real environmental data, that **(i)** in-distribution scores systematically overstate deployable skill, and **(ii)** representations chosen to be invariant to the deployment shift axis measurably shrink that overstatement.

### Sources (web)

- Koh et al. 2021, *WILDS: A Benchmark of in-the-Wild Distribution Shifts* — [arXiv:2012.07421](https://arxiv.org/pdf/2012.07421)
- Spatial-blocking cross-validation for SDMs — [arXiv:2502.03480](https://arxiv.org/abs/2502.03480); random splits overstate ecological AUC by up to 0.16
- Tibshirani et al. 2019, *Conformal Prediction Under Covariate Shift* — [NeurIPS](https://dl.acm.org/doi/10.5555/3454287.3454514)
- Angelopoulos & Bates 2021, *A Gentle Introduction to Conformal Prediction* — [arXiv:2107.07511](https://arxiv.org/pdf/2107.07511)
