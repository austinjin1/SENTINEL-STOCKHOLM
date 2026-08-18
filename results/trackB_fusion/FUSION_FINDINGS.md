# SENTINEL Multimodal Fusion — Deep-Research Findings

*Investigation into why fusion did not beat single encoders, and what is genuinely true. All numbers are real, leakage-free, station-grouped CV with site-clustered bootstrap CIs unless noted. July 2026.*

## Question
Multimodal fusion (satellite + in-situ sensor) tied or lost to the best single encoder on chl-a and DO. Is the fusion broken, or is there nothing to fuse?

## Method
Systematically separated "the code can't fuse" from "there's nothing to fuse": (1) audited the fusion→anomaly connection for bugs; (2) tested every on-disk label for complementarity; (3) tested better representations (CNN, end-to-end); (4) built the genuinely bimodal microcystin task (downloaded ~4k Sentinel-2 tiles + WQP drivers); (5) stress-tested robustness (temporal, cross-region, negative control); (6) tested whether any feature transform recovers cross-region transfer.

## Findings

**1. chl-a / DO / turbidity are single-modality-dominated (redundancy, not a bug).**
GBM-on-concat (a strong interaction learner) also fails to beat the best single: chl-a is optical (satellite dominant), DO/turbidity are chemistry (sensor/persistence dominant, satellite-blind). Literature-consistent (DO is non-optically-active). Adding a satellite CNN did not help (learned embedding dilutes, like the WaterDroneNet `cls_feat`).

**2. Real bugs in the *shipped* PerceiverIOFusion (now fixed for correctness).**
Dead confidence gate (constant 0.71), sample-0 registry (batched fusion broadcast row 0), linear-only projection, and a **circular anomaly label** (`train_fusion.py:402`, `|z-scored input|>3`). All fixed / retired. The Track-B experiments used a stateless path that already avoided these, so they were not the cause of the chl-a/DO ties.

**3. Fusion genuinely wins on microcystin — a truly bimodal task.**
The cyanotoxin is *optically invisible*, so satellite optics, in-situ pigment, and physicochemical drivers carry non-redundant information. Under station-grouped CV (`07`,`08`):

| Threshold | satellite | in-situ pigment | drivers | 2-way (sat+pig) | **3-way (+drivers)** |
|---|---|---|---|---|---|
| mc > 1.6 (WHO drinking) | 0.65 | 0.68 | 0.63 | 0.776 | **0.823** |
| mc > 0.3 (detection) | 0.66 | 0.72 | 0.57 | 0.786 | **0.800** |

Each modality adds: 3-way > 2-way > best single, CIs exclude zero (driver gain +0.047, site-bootstrap CI [0.028, 0.068], P=1.0), and the negative control (shuffle drivers → collapses to 2-way) confirms it is real signal, not overfit capacity.

**4. But the microcystin win is region-specific — a fundamental, not fixable, limit.**
Under leave-one-HUC-region-out holdout, *all* models (single and fusion) collapse to chance (~0.49). **No feature transform** — region-conditioning (lat/lon/month), within-site-relative, or global-rank — recovers it (`10`; all 0.42–0.53). The literature explains why: microcystin depends on the **toxic-strain fraction (mcyD genotype)**, which varies dramatically between watersheds (12–100% toxic in one lake vs 0.01–6% in another) and predicts toxin better than any pigment. That variable is not in optics/pigment/drivers, so the learned relationship cannot transfer.

## Conclusion
Fusion is not broken. On optical/chemistry labels the modalities are redundant (one dominates), so fusion correctly ties the best single. On microcystin the modalities are genuinely complementary and fusion wins — *within a calibrated region*. Cross-region deployment is blocked by an unobserved biological variable (toxic-strain genotype).

**The transfer path is SENTINEL's own microbial/metagenomic modality** (mcyD gene abundance), which directly measures the toxic-strain fraction. A satellite + pigment + driver + **metagenomic** fusion is the science-grounded route to a transferable cyanotoxin early-warning system — gated on acquiring metagenomic samples co-located with cyanotoxin measurements.

## Artifacts
`scripts/experiments/trackB_fusion/{07_microcystin_fusion,08_driver_arm,09_driver_robustness,10_region_transfer}.py`; `results/trackB_fusion/{mc_fusion,mc_driver_fusion,mc_driver_robustness,mc_region_transfer}.json`; fixed `sentinel/models/fusion/{model,projections,confidence_gating}.py`, retired label in `scripts/training/train_fusion.py`.
