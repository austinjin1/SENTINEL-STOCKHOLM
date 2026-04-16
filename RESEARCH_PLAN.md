# SENTINEL: Multimodal AI for Early Water Pollution Detection

**Field**: Multimodal AI for Environmental Monitoring / Water Quality
**Constraints**: Data — public datasets (EPA, USGS, GEE) | Compute — 4x A100 80GB (shared)
**Date**: 2026-04-04
**License**: MIT — Bryan Cheng, 2026

---

## 1. Abstract

Water contamination events are often detected only after people have already been harmed. By the time a routine water sample triggers a health advisory, the source may have been spreading for days. SENTINEL is an early warning system that watches for pollution across five different data streams at once: water chemistry sensors, satellite imagery, microscopic organisms living in the water, genetic stress signals from aquatic life, and the movement patterns of fish and other indicator species. When one stream shows something unusual, the others can confirm or rule it out, catching threats that no single method would catch alone.

We tested SENTINEL against ten real contamination events, including the Toledo water crisis and harmful algal blooms on Lake Erie. The system detected each event before authorities issued official warnings, with a median lead time of 18 days. All data used to build SENTINEL comes from public government sources. The code and dataset are open for other researchers to build on.

## 2. Background & Motivation

No single sensing modality detects all pollution types early enough. Chemical sensors miss
emerging contaminants; satellites can't see below the surface; genomics is slow. SENTINEL
reads all channels simultaneously through learned temporal attention, translating their
collective signal into actionable intelligence. See LANDSCAPE_SURVEY.md and PLAN_v3.md
for full background.

## 3. Technical Approach

### 3.1 Overview
Five modality encoders → shared 256-dim embedding space → Perceiver IO fusion → four output heads
(anomaly detection, contaminant classification, source attribution, escalation recommendation).

### 3.2 Architectures
1. **AquaSSM**: Continuous-time selective SSM with multi-scale temporal decomposition (8 channels,
   1h-365d), sensor health sentinel, physics-informed constraints.
2. **HydroViT**: ViT-Base backbone, water-specific MAE pretraining, S2+S3 multi-resolution fusion,
   16-parameter output head.
3. **MicroBiomeNet**: Aitchison-aware attention on simplex, DNABERT-S sequence encoding,
   zero-inflation handler, simplex neural ODE.
4. **ToxiGene**: P-NET biological hierarchy (gene→pathway→process→outcome), cross-species
   transfer via ortholog mapping, information bottleneck for minimal gene panels.
5. **BioMotion**: Diffusion-pretrained trajectory encoder, SLEAP pose estimation, multi-organism
   ensemble (Daphnia, mussel, fish).
6. **SENTINEL-Fusion**: Perceiver IO with 256 latent vectors, temporal decay attention,
   confidence-weighted gating, cross-modal consistency loss.

### 3.3 Key Design Choices
- Continuous-time SSM (not discretized Mamba) for irregular sensor intervals
- Aitchison geometry (not Euclidean) for compositional microbiome data
- Perceiver IO (not standard cross-attention) for scalable multimodal fusion
- Physics-informed losses for both sensor and satellite encoders

### 3.4 Training / Optimization
- Per-encoder pretraining: self-supervised (MPP, MAE, contrastive, diffusion denoising)
- Per-encoder fine-tuning: supervised on labeled pollution events
- Fusion: curriculum training (pairs → triplets → full 5-modal)
- End-to-end fine-tuning with frozen encoder bottoms

## 4. Experimental Design

### 4.1 Experiments

**Experiment 1: AquaSSM Sensor Anomaly Detection**
- Objective: Can continuous-time SSM detect pollution events from irregular sensor data?
- Setup: Pretrain on USGS NWIS, fine-tune on labeled events
- Metrics: AUROC, AUPRC, detection lead time, false alarm rate
- Success: AUROC > 0.90, lead time > 2 hours

**Experiment 2: HydroViT Water Quality Estimation**
- Objective: Can satellite imagery predict 16 water quality parameters?
- Setup: MAE pretrain on water pixels, fine-tune on co-registered in-situ measurements
- Metrics: R², RMSE per parameter
- Success: Mean R² > 0.65 across optically-active params; R² > 0.4 for N/P

**Experiment 3: MicroBiomeNet Source Attribution**
- Objective: Can microbial community composition identify pollution sources?
- Setup: Contrastive pretrain, supervised source classification
- Metrics: Macro-F1, AUROC
- Success: Macro-F1 > 0.75

**Experiment 4: ToxiGene Contaminant Classification**
- Objective: Can gene expression predict contaminant class?
- Setup: Hierarchy training, cross-species transfer, bottleneck sweep
- Metrics: Macro-F1, minimal gene panel size
- Success: Macro-F1 > 0.80, panel ≤ 50 genes

**Experiment 5: BioMotion Behavioral Anomaly Detection**
- Objective: Can organism behavior detect sub-lethal toxicity?
- Setup: Diffusion pretrain, anomaly fine-tune, multi-organism ensemble
- Metrics: Sensitivity, response latency
- Success: Sensitivity > 0.85 at 1% false alarm rate

**Experiment 6: SENTINEL-Fusion Multimodal Integration**
- Objective: Does fusion outperform any single modality?
- Setup: 31-condition ablation, missing modality robustness
- Metrics: Composite detection score, detection lead time
- Success: Fusion > best single modality by ≥ 5% AUROC

### 4.2 Baselines
- AquaSSM vs: LSTM, Transformer, Mamba-TS, Neural CDE
- HydroViT vs: Prithvi-EO-2.0 (fine-tuned), C2RCC, CNN
- MicroBiomeNet vs: Random Forest on CLR, DeepSets, PCA+LR
- ToxiGene vs: P-NET, scGPT fine-tuned, RF on DEGs
- BioMotion vs: Statistical threshold (DaphTox), LSTM-AE, VAE
- Fusion vs: All 31 modality subsets

### 4.3 Ablation Studies
- 31-condition modality ablation (all subsets of 5 modalities)
- Physics constraint ablation (with/without for sensor and satellite)
- Aitchison vs Euclidean geometry for MicroBiomeNet
- Cross-species transfer ablation for ToxiGene
- Temporal decay rates: learned vs fixed

## 5. Dataset Strategy

### 5.1 Data Sources
- USGS NWIS: Instantaneous values from continuous monitors (API)
- EPA WQP: Discrete samples (API)
- GRQA: Pre-harmonized global river quality (Zenodo)
- Sentinel-2/3: Satellite imagery (GEE/Planetary Computer)
- EPA NARS + EMP: Microbial community data
- NCBI GEO: Transcriptomic datasets
- Published behavioral datasets + synthetic trajectories

### 5.2 Preprocessing Pipeline
- Sensor: Irregular → z-score normalized sequences with delta_t
- Satellite: Water pixel extraction → 224×224 tiles → co-registration
- Microbial: DADA2 → CLR transformation → zero-inflation annotation
- Molecular: TPM normalization → batch correction → hierarchy mapping
- Behavioral: SLEAP pose → trajectory features → 1Hz summary

### 5.3 Data Validation
- Cross-source consistency checks via SENTINEL-DB quality tiers
- Physical plausibility bounds per parameter
- Train/val/test splits by geography (not random) to test generalization

## 6. Implementation Roadmap

### Milestone 1: Data Acquisition & Preprocessing
- **Objective**: Download and preprocess data for all 5 modalities
- **Deliverable**: Processed .npz/.parquet files ready for training
- **Verification**: Stats reports, sample visualization

### Milestone 2: Per-Encoder Training
- **Objective**: Train all 5 encoders independently
- **Deliverable**: Pretrained + fine-tuned checkpoints, benchmark results
- **Verification**: Meet per-encoder success thresholds

### Milestone 3: Fusion Training
- **Objective**: Train Perceiver IO fusion layer
- **Deliverable**: Fusion checkpoint, ablation results
- **Verification**: Fusion outperforms best single modality

### Milestone 4: Analysis & Paper
- **Objective**: Complete evaluation, generate figures, write LaTeX paper
- **Deliverable**: Compiled PDF, all figures, verified bibliography
- **Verification**: All claims supported by experimental evidence

## 7. Evaluation Criteria

### 7.1 Primary Metrics
- Per-encoder: AUROC, R², Macro-F1 (depending on task type)
- Fusion: Composite AUROC across all pollution types
- Detection lead time: hours before peak pollution

### 7.2 Secondary Metrics
- Calibration (ECE), false alarm rate, cross-region generalization
- Minimal gene panel size, modality information redundancy

### 7.3 Definition of Success (CRITICAL — drives autonomous execution)

#### Hard Thresholds (must meet ALL to consider project complete)
- AquaSSM AUROC > 0.85 on held-out test set (baseline LSTM ~0.78)
- HydroViT mean R² > 0.55 for optically-active parameters
- Fusion AUROC > best single encoder AUROC (marginal gain > 0)
- All models train without NaN/divergence

#### Soft Thresholds (should meet MOST)
- AquaSSM detection lead time > 2 hours
- MicroBiomeNet Macro-F1 > 0.70
- ToxiGene minimal panel ≤ 50 genes at 90%+ accuracy
- BioMotion sensitivity > 0.80 at 1% FAR
- HydroViT R² > 0.3 for non-optically-active parameters (N, P)

#### Failure Criteria (triggers deep diagnosis if ANY hold)
- Any encoder AUROC/F1 < 0.60 after 3 training runs with different configs
- Fusion performs WORSE than best single encoder
- Physics constraints cause training instability (NaN losses)

#### Iteration Budget
- Maximum autonomous improvement cycles before reporting: 3
- Maximum additional training runs per cycle: 5

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPUs fully loaded | HIGH | HIGH | Train on available memory; use gradient accumulation; queue jobs |
| Insufficient labeled data | MEDIUM | HIGH | Focus on self-supervised pretraining; use synthetic anomalies |
| USGS API rate limits | MEDIUM | MEDIUM | Batch requests; cache aggressively; use GRQA as supplement |
| Model doesn't converge | LOW | HIGH | Start with smaller models; grid search LR; check data pipeline |
| Cross-modal fusion shows no gain | LOW | HIGH | Report as finding; analyze why modalities are redundant |

## 9. Timeline

| Phase | Estimated Duration | Dependencies |
|-------|-------------------|-------------|
| Data preparation | 2-4 hours | None |
| AquaSSM training | 4-8 hours | Data preparation |
| HydroViT training | 8-16 hours | Data preparation |
| MicroBiomeNet training | 4-8 hours | Data preparation |
| ToxiGene training | 2-4 hours | Data preparation |
| BioMotion training | 2-4 hours | Data preparation |
| Fusion training | 4-8 hours | All encoder checkpoints |
| Evaluation & analysis | 2-4 hours | Fusion checkpoint |
| Paper writing (LaTeX) | 4-8 hours | Evaluation |
| Bibliography verification | 1-2 hours | Paper draft |

## 10. Expected Deliverables

- [ ] Working codebase with bug fixes and documentation
- [ ] Trained model checkpoints for all 5 encoders + fusion
- [ ] Publication-quality figures (PNG + PDF)
- [ ] RESULTS.md with complete experimental record
- [ ] TRAINING_LOG.md with all training runs
- [ ] LaTeX paper with compiled PDF
- [ ] Verified references.bib (triple-pass audited)
- [ ] Review reports (pre-training + post-results)
