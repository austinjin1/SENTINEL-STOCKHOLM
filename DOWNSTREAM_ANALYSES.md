# Downstream Analyses & Inference Plan

**Status**: Pending — run after all model training results are in and acceptable.
**Date**: 2026-04-07

---

## Priority 1: Case Study Inference (HIGH IMPACT)

Run full 5-modal fusion on 10 historical contamination events to measure detection lead time vs official discovery.

- **Script**: `sentinel/evaluation/case_study.py`
- **Events**: Gold King Mine, Flint Water Crisis, Toledo HAB, Animas River, Thames sewage, + 5 more
- **Metrics**: Detection lead time (hours), peak anomaly score, source attribution accuracy
- **Run**: `python -m sentinel.evaluation.case_study --output-dir results/case_studies`

## Priority 2: 31-Condition Modality Ablation (HIGH IMPACT)

Evaluate all 2^5 - 1 = 31 non-empty modality subsets to quantify each modality's marginal contribution.

- **Script**: `sentinel/evaluation/ablation.py`
- **Output**: `results/ablation/ablation_results.json`, `marginal_gains.csv`, `statistical_tests.json`
- **Run**: `python -m sentinel.evaluation.ablation --output-dir results/ablation`

## Priority 3: Causal Discovery (HIGH IMPACT — novel contribution)

PCMCI-based cross-modal causal discovery. Discovers pathways like:
agricultural runoff -> nutrient loading -> algal bloom -> DO depletion.

- **Scripts**: `sentinel/models/theory/causal_discovery.py` + `sentinel/evaluation/causal_chains.py`
- **Validates against**: 14 known environmental causal chains from literature

## Priority 4: Conformal Prediction Intervals (MEDIUM)

Calibrated uncertainty intervals with finite-sample coverage guarantees.

- **Script**: `sentinel/models/theory/conformal.py`
- **Contribution**: Demonstrate P(true state in prediction set) >= 1-alpha

## Priority 5: Optimal Sensor Placement (MEDIUM)

Information-theoretic optimization of monitoring station deployment.

- **Script**: `sentinel/models/theory/sensor_placement.py`
- **Method**: Submodular greedy with (1-1/e) approximation guarantee

## Priority 6: NEON Anomaly Scan (MEDIUM)

Run AquaSSM on 351M-row NEON aquatic dataset across 34 sites.

- **Data**: NEON parquet shards (already downloaded, 45.9 GB)
- **Output**: Per-site anomaly timeline, flagged unreported events

## Priority 7: Global Water Health Map (LOWER)

Run HydroViT on all available S2 tiles for water quality parameter maps.

- **Script**: `sentinel/evaluation/global_hotspots.py` (partially implemented)

## Priority 8: Publication Figures (AFTER all analyses)

Generate 10 Nature-style figures from all results.

- **Script**: `sentinel/evaluation/figures.py`
- **Run**: `python -m sentinel.evaluation.figures --results-dir results/ --output-dir figures/`
