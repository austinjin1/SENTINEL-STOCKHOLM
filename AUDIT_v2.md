# SENTINEL v2 Full Codebase Alignment Audit

**Date**: 2026-04-03  
**Total codebase**: 59 Python files, ~27,000 lines + 3 TS/TSX files (262 lines)

---

## Executive Summary

The **data pipeline** is now well-aligned with v2 (~95% complete). The **model architectures, training, evaluation, and dashboard** are still v1 and need significant rewrites. Here's the full breakdown:

| Module | v2 Status | Gap Severity |
|--------|-----------|-------------|
| **Data: SENTINEL-DB** | ✅ Complete | — |
| **Data: Satellite** | ✅ Complete (S2+S3, HydroViT labels) | — |
| **Data: Sensor** | ✅ Complete (irregular-time + international) | — |
| **Data: Microbial** | ✅ Complete (DNABERT-S, simplex, zero-inflation) | — |
| **Data: Molecular** | ✅ Complete (AOP-Wiki, orthologs, hierarchy) | — |
| **Data: ECOTOX** | ✅ Complete (unchanged, still needed) | — |
| **Data: Behavioral** | ✅ Complete (SLEAP, diffusion trajectories) | — |
| **Data: Alignment** | ✅ Complete (H3, HydroLAKES) | — |
| **Data: Case Studies** | ✅ Complete (10 events, collector) | — |
| **Model: Sensor Encoder** | ❌ v1 TCN | **CRITICAL** — need AquaSSM (continuous-time SSM) |
| **Model: Satellite Encoder** | ❌ v1 ViT-S/SSL4EO-S12 | **CRITICAL** — need HydroViT (water MAE, S2+S3, 16-param) |
| **Model: Microbial Encoder** | ❌ v1 basic Transformer + VAE | **CRITICAL** — need MicroBiomeNet (Aitchison, DNABERT-S, simplex ODE) |
| **Model: Molecular Encoder** | ❌ v1 Chem2Path MLP | **CRITICAL** — need ToxiGene (P-NET hierarchy, cross-species) |
| **Model: BioMotion** | ❌ MISSING entirely | **CRITICAL** — new 5th modality encoder |
| **Model: Digital Biosentinel** | ✅ v1 is valid | Minor — still used in v2 |
| **Model: Fusion** | ❌ v1 cross-modal attention + GRU | **CRITICAL** — need Perceiver IO (256 latent array) |
| **Model: Escalation** | ⚠️ v1 (4 modalities) | **MODERATE** — update to 5 modalities + BioMotion |
| **Model: Theory** | ❌ MISSING entirely | **HIGH** — 5 theoretical contributions |
| **Training: Satellite** | ❌ v1 (trains ViT-S) | **CRITICAL** — need HydroViT MAE training |
| **Training: Sensor** | ❌ MISSING | **CRITICAL** — need AquaSSM training |
| **Training: Microbial** | ❌ MISSING | **CRITICAL** — need MicroBiomeNet training |
| **Training: Molecular** | ❌ MISSING | **CRITICAL** — need ToxiGene training |
| **Training: BioMotion** | ❌ MISSING | **CRITICAL** — need diffusion pretraining |
| **Training: Biosentinel** | ❌ MISSING | MODERATE — straightforward |
| **Training: Fusion** | ❌ MISSING | **CRITICAL** — Perceiver IO staged training |
| **Training: Escalation** | ❌ MISSING (only concept) | MODERATE — update for 5 modalities |
| **Training: Base Trainer** | ✅ Complete | — |
| **Evaluation: Case Study** | ⚠️ v1 (4 modalities) | MODERATE — update for 5 modalities |
| **Evaluation: Metrics** | ⚠️ v1 (11 conditions) | MODERATE — expand to 31 ablation conditions |
| **Evaluation: Ablation** | ❌ MISSING | HIGH — 31-condition study (2^5 - 1 subsets) |
| **Evaluation: Figures** | ❌ MISSING | HIGH — 10 publication figures |
| **Evaluation: Info Analysis** | ❌ MISSING | HIGH — cross-modal MI estimation |
| **Evaluation: Global Hotspots** | ❌ MISSING | HIGH — HydroViT on full S2/S3 archive |
| **Evaluation: Causal Chains** | ❌ MISSING | HIGH — heterogeneous causal discovery |
| **Platform: Citizen QC** | ❌ MISSING | MODERATE — 3-stage QC pipeline |
| **Platform: Photo Analysis** | ❌ MISSING | MODERATE — ground-level HydroViT |
| **Platform: API** | ❌ MISSING | MODERATE — FastAPI research API |
| **Dashboard** | ⚠️ Scaffold only (262 lines) | **HIGH** — needs full component build |
| **Config** | ⚠️ v1 architecture params | MODERATE — update for v2 architectures |

---

## Detailed Gap Analysis

### 1. MODEL ARCHITECTURES (CRITICAL — largest gap)

#### 1.1 Sensor Encoder: TCN → AquaSSM
**Current**: `sensor_encoder/tcn.py` — standard TCN with 3 layers [64,128,256], causal convolutions, fixed 15-min intervals, 672 timesteps.

**v2 Needs**:
- Continuous-time selective state space model (Mamba + Neural CDE hybrid)
- Learned step size function: Δt = fθ(gap, x_prev) — adapts to arbitrary inter-observation intervals
- Multi-scale temporal decomposition: bank of parallel SSM channels (log-spaced 1h to 1yr)
- Physics-informed state constraints (DO-temp coupling, pH-alkalinity, conductivity-TDS)
- Sensor health sentinel auxiliary head (dedicated, not just anomaly classification)
- Input: irregular timestamps + Δt features (from new data pipeline), NOT fixed 672-step windows

**Files to rewrite**: `tcn.py` → `aqua_ssm.py`, update `mpp.py`, `anomaly.py`, `model.py`
**New files**: `physics_constraints.py`

#### 1.2 Satellite Encoder: ViT-S → HydroViT
**Current**: `satellite_encoder/backbone.py` — ViT-S/16, 22M params, 10-band S2 input, SSL4EO-S12 weights. `segmentation.py` — UPerNet with 8 anomaly classes. `temporal.py` — 2-layer transformer for temporal change detection.

**v2 Needs**:
- ViT-Base (86M params, not Small 22M) — or at minimum ViT-Base option
- Water-specific MAE pretraining with spectral physics consistency loss (predicted patches must satisfy water-leaving radiance constraints)
- Multi-resolution S2 (10m) + S3 OLCI (300m, 21 bands) fusion via resolution-aware cross-attention
- 16-parameter regression output head (chl-a through PAI), not just anomaly segmentation
- Temporal attention stack with cloud-weighted cross-attention (5-10 images)
- Input: water-pixel patches extracted via JRC mask (from new data pipeline)

**Files to rewrite**: `backbone.py` → `hydrovit_backbone.py`, `segmentation.py` → `parameter_head.py`, `temporal.py` → `temporal_stack.py`
**New files**: `multi_resolution.py`, `physics_loss.py`

#### 1.3 Microbial Encoder: Transformer+VAE → MicroBiomeNet
**Current**: `source_attribution.py` — standard transformer on CLR vectors. `vae.py` — standard VAE for community health scoring.

**v2 Needs**:
- Aitchison-aware attention (similarity in CLR-space, not Euclidean)
- DNABERT-S sequence encoding per OTU (DNA sequence → phylogenetic embedding)
- Abundance-weighted soft attention pooling
- Zero-inflation gating (structural vs sampling zeros)
- Neural ODE on Aitchison simplex for temporal trajectories (replaces VAE)
- Aitchison batch normalization

**Files to rewrite**: ALL — `source_attribution.py` → `aitchison_attention.py`, `vae.py` → `simplex_ode.py`
**New files**: `sequence_encoder.py`, `abundance_pooling.py`, `zero_inflation.py`, `model.py`

#### 1.4 Molecular Encoder: Chem2Path → ToxiGene
**Current**: `chem2path.py` — MLP backbone (512→256→128) with 7 per-pathway sigmoid heads. `bottleneck.py` — L1-gated gene selection.

**v2 Needs**:
- P-NET biological hierarchy network: Gene → Pathway → Biological Process → Adverse Outcome
- Sparse, biology-constrained connections from Reactome + AOP-Wiki (NOT fully connected MLP)
- Cross-species transfer via ortholog mapping (zebrafish ↔ Daphnia ↔ fathead minnow)
- Architecture mirrors the AOP framework — interpretable by design
- Keep information bottleneck concept but adapt to hierarchy

**Files to rewrite**: `chem2path.py` → `hierarchy_network.py`
**New files**: `cross_species.py`, update `model.py`

#### 1.5 BioMotion Encoder: MISSING
**v2 Needs** (entirely new):
- SLEAP-based pose estimation pipeline (12 Daphnia, 8 mussel, 22 fish keypoints)
- Diffusion-based trajectory pretraining (denoising score = anomaly signal)
- Multi-organism ensemble with shared anomaly reasoning layer
- Behavioral feature encoding (velocity, acceleration, turning angle, phototaxis)
- Projection to shared 256-dim embedding space

**New directory**: `sentinel/models/biomotion/`
**New files**: `pose_encoder.py`, `trajectory_encoder.py`, `multi_organism.py`, `model.py`

#### 1.6 Fusion Layer: Attention+GRU → Perceiver IO
**Current**: Cross-modal attention (8 heads) + GRU state (256-dim). Handles 4 modalities. Single 256-dim state vector. Temporal decay per-modality (learned tau).

**v2 Needs**:
- Perceiver IO architecture: fixed array of 256 learned latent vectors (NOT single 256-dim vector)
- Cross-attention: modality tokens → latent array → back (encode + decode)
- 5 modalities (add "behavioral")
- Confidence-weighted gating from calibrated encoder scores
- Cross-modal consistency loss for self-supervised alignment
- 4 output heads (add "recommended action" / escalation tier)
- Per-modality-PAIR decay rates (not per-modality)

**Files to rewrite**: ALL — `attention.py`, `embedding_registry.py`, `model.py`, `state.py`, `projections.py`, `heads.py`
**New files**: `latent_array.py`, `perceiver_attention.py`, `consistency_loss.py`, `confidence_gating.py`

#### 1.7 Escalation Controller: Update for 5 modalities
**Current**: 4 tiers (sensor → +satellite → +microbial → +molecular). 4 modalities. STATE_DIM=266.

**v2 Needs**:
- 5 modalities (add behavioral)
- Update tier definitions:
  - Tier 0: Sensor + BioMotion (continuous, fast response — behavioral detects in minutes)
  - Tier 1: + Satellite (triggered spatial analysis)
  - Tier 2: + Microbial (source attribution)
  - Tier 3: + Molecular + Digital Biosentinel (full characterization)
- STATE_DIM increase to accommodate 5 anomaly scores + 5 modality data

**Files to update**: `environment.py`, `policy.py` (larger state), `model.py`

### 2. THEORETICAL CONTRIBUTIONS (MISSING — `sentinel/models/theory/`)

5 novel theoretical contributions from the amended plan:

| # | Contribution | File Needed |
|---|-------------|-------------|
| 1 | HEMA — physics-informed contrastive alignment across modalities | `hema.py` |
| 2 | Heterogeneous temporal causal discovery across modality types | `causal_discovery.py` |
| 3 | Information-theoretic multi-modal sensor placement optimization | `sensor_placement.py` |
| 4 | Aitchison-aware neural networks (universal approximation + batch norm) | `aitchison_nn.py` |
| 5 | Multimodal conformal anomaly detection with coverage guarantees | `conformal.py` |

### 3. TRAINING PIPELINES (MOSTLY MISSING)

**Exists**: `trainer.py` (base class — good), `train_satellite.py` (v1 — trains ViT-S)

**Missing** (6 scripts):
- `train_sensor.py` — AquaSSM pretraining (continuous-time MPP) + fine-tuning
- `train_microbial.py` — MicroBiomeNet (Aitchison contrastive + simplex ODE)
- `train_molecular.py` — ToxiGene (hierarchy + cross-species transfer)
- `train_biomotion.py` — BioMotion (diffusion pretraining + fine-tuning)
- `train_biosentinel.py` — Digital Biosentinel (straightforward, model exists)
- `train_fusion.py` — Perceiver IO staged training (pairs → triplets → full 5-modal)
- `train_escalation.py` — PPO with curriculum on 5-modality environment

**Rewrite**: `train_satellite.py` — retrain for HydroViT (MAE + physics loss)

### 4. EVALUATION (MOSTLY MISSING)

**Exists**: `case_study.py` (v1, 4 modalities), `metrics.py` (v1, 11 ablation conditions)

**Missing** (5 scripts):
- `ablation.py` — 31-condition study (all 2^5 - 1 modality subsets)
- `figures.py` — 10 publication-quality figures (architecture diagram, observability matrix, timelines, ablation bar chart, indicator species heatmap, biomarker curve, decay values, UMAP, dashboard screenshot, cascade behavior)
- `missing_modality.py` — robustness to random modality dropout
- `information_analysis.py` — cross-modal mutual information estimation
- `global_hotspots.py` — HydroViT inference on full S2/S3 archive
- `causal_chains.py` — temporal causal chain discovery across modalities

**Update**: `case_study.py` and `metrics.py` for 5 modalities

### 5. PLATFORM (MISSING — `sentinel/platform/`)

- `citizen_qc.py` — 3-stage automated QC (plausibility → spatial consistency → temporal consistency)
- `photo_analysis.py` — ground-level photo water assessment (HydroViT branch for smartphone images)
- `test_kit.py` — test kit data input + validation
- `api.py` — FastAPI research API (model inference + data access endpoints)

### 6. DASHBOARD (SCAFFOLD ONLY — 262 lines of TS)

**Current**: `App.tsx` (121 lines, basic layout), `types.ts` (141 lines, type definitions), `main.tsx` (entry point). 3 demo JSON files. No components.

**v2 Needs** (~5,000-8,000 lines):
- `GlobalMapView.tsx` — Leaflet map with alert-colored monitoring sites + plume overlays
- `SiteDetailPanel.tsx` — multi-tab panel (sensor, satellite, microbial, behavioral, pathway, biosentinel)
- `AlertReport.tsx` — structured evidence report with PDF export
- `CaseStudyReplay.tsx` — timeline scrubber for live demo
- `charts/` — SensorChart, RadarChart, TaxaBarChart, ImpactGauge, TimelineChart, AnomalyHeatmap
- Citizen science UI (photo upload, test kit form, community dashboard)
- Water Health Score (0-100) display
- Gamification (levels, badges)

### 7. CONFIGURATION

**Current** `configs/default.yaml` references v1 architectures:
- `satellite.backbone: "vit_small_patch16_224"` → should offer `"hydrovit_base"`
- `sensor.tcn_channels: [64, 128, 256]` → should be `aquassm` config
- `microbial.source_attribution.embed_dim: 256` → should be `microbiomenet` config
- `fusion.state_dim: 256` → should be `perceiver.n_latents: 256`
- No behavioral modality config
- No theoretical contribution config

---

## Priority-Ordered Work Plan

### Priority 1: Model Architecture Rewrites (CRITICAL)
1. **AquaSSM** (sensor encoder) — continuous-time SSM
2. **HydroViT** (satellite encoder) — water foundation model  
3. **MicroBiomeNet** (microbial encoder) — Aitchison-aware
4. **ToxiGene** (molecular encoder) — P-NET hierarchy
5. **BioMotion** (behavioral encoder) — entirely new
6. **Perceiver IO fusion** — replace attention+GRU
7. **Escalation update** — 5 modalities

### Priority 2: Training Pipelines (CRITICAL)
8. All 8 training scripts (7 new + 1 rewrite)

### Priority 3: Theoretical Contributions (HIGH)
9. 5 theory modules

### Priority 4: Evaluation Expansion (HIGH)
10. 6 new evaluation scripts + 2 updates

### Priority 5: Platform + Dashboard (MODERATE)
11. 4 platform modules
12. Dashboard components (~15 files)

### Priority 6: Config Update (LOW)
13. Update default.yaml for v2 architectures

---

## Summary Scorecard

| Category | Files Exist | Files Needed | v2-Aligned | Gap |
|----------|-------------|-------------|------------|-----|
| Data Pipeline | 22 | 22 | 22 (100%) | 0 |
| Model Architectures | 26 | ~40 | 6 (15%) | ~34 |
| Training | 2 | 9 | 1 (11%) | 8 |
| Evaluation | 2 | 8 | 0 (0%) | 8 |
| Theory | 0 | 5 | 0 (0%) | 5 |
| Platform | 0 | 4 | 0 (0%) | 4 |
| Dashboard | 3 | ~18 | 0 (0%) | ~18 |
| Config/Utils | 3 | 3 | 2 (67%) | 1 |
| **TOTAL** | **58** | **~109** | **31 (28%)** | **~78** |

**Bottom line**: Data pipeline is done. ~78 more files needed across models, training, evaluation, theory, platform, and dashboard to reach full v2 alignment.
