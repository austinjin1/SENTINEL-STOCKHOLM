# SENTINEL v2 Implementation Checklist

## Legend
- [x] = Complete (from v1 build)
- [~] = Partially done (needs rewrite for v2)
- [ ] = Not started

---

## Phase 0: Infrastructure & SENTINEL-DB

### 0.1 Environment & Scaffolding
- [x] Project directory structure
- [x] pyproject.toml
- [x] environment.yml
- [x] .gitignore
- [x] Utils (config.py, logging.py)
- [ ] Update environment.yml for new deps (mamba-ssm, dnabert, sleap, h3-py, conformal-prediction libs)
- [ ] Update configs/default.yaml for v2 architectures

### 0.2 SENTINEL-DB Harmonization Pipeline (NEW)
- [ ] `sentinel/data/sentinel_db/ontology.py` — Unified parameter ontology (10,000+ names → 500 canonical)
- [ ] `sentinel/data/sentinel_db/spatial.py` — H3 hexagonal spatial hashing + deduplication
- [ ] `sentinel/data/sentinel_db/quality.py` — Quality tier assignment (Q1-Q4)
- [ ] `sentinel/data/sentinel_db/linking.py` — Cross-modality linking engine (co-registration)
- [ ] `sentinel/data/sentinel_db/ingest.py` — Multi-source ingest (EPA, USGS, EU Waterbase, GEMS/Water, citizen science)
- [ ] `sentinel/data/sentinel_db/schema.py` — Database schema / data models

### 0.3 Data Acquisition (partially done from v1)
- [x] Satellite: Sentinel-2 L2A via GEE + Planetary Computer
- [~] Satellite: Add Sentinel-3 OLCI download (21 bands, 300m, daily) — NEW for HydroViT
- [x] Satellite: Landsat 8/9 TIRS thermal
- [x] Satellite: Spectral index computation + tile preprocessing
- [x] Sensor: USGS NWIS download + preprocessing
- [ ] Sensor: Expand to EPA Water Quality Portal (https://www.waterqualitydata.us/)
- [ ] Sensor: Add EU Waterbase + GEMS/Water ingest
- [x] Microbial: EPA NARS + EMP + NCBI SRA download
- [x] Microbial: Preprocessing (DADA2/CLR pipeline)
- [x] Molecular: GEO + CTD download + preprocessing
- [ ] Molecular: Add AOP-Wiki hierarchy download for ToxiGene
- [ ] Molecular: Ortholog mapping tables (zebrafish↔Daphnia↔fathead minnow)
- [x] ECOTOX: Download + preprocessing
- [ ] Behavioral: Acquire Daphnia/mussel/fish behavioral video datasets
- [ ] Behavioral: DaphBASE + published ethology datasets
- [ ] Citizen science: FreshWater Watch integration
- [ ] Geographic: HydroLAKES + JRC Global Surface Water datasets
- [~] Geographic alignment (v1 has basic HUC-8; need H3 + global coverage)
- [~] Case studies (v1 has 10 events; expand with richer multi-modal data)

---

## Phase 1: Modality Encoders (Parallel Development)

### 1.1 AquaSSM — Sensor Encoder (REWRITE)
- [~] `sentinel/models/sensor_encoder/` — v1 has TCN, need SSM
- [ ] `aqua_ssm.py` — Continuous-time selective state space core
  - [ ] Learned step size function: Δt = fθ(t_k - t_{k-1}, x_{k-1})
  - [ ] Multi-scale temporal decomposition (parallel SSM bank, log-spaced 1hr→1yr)
  - [ ] Gated mixing layer across temporal scales
- [ ] `physics_constraints.py` — Physics-informed state constraints
  - [ ] DO solubility vs temperature coupling
  - [ ] pH-alkalinity carbonate chemistry
  - [ ] Conductivity-TDS relationship
- [ ] `sensor_health.py` — Sensor health sentinel auxiliary head
- [ ] `mpp.py` — Masked parameter prediction (adapt from v1 to SSM backbone)
- [ ] `model.py` — Complete AquaSSM encoder

### 1.2 HydroViT — Satellite Encoder (REWRITE)
- [~] `sentinel/models/satellite_encoder/` — v1 has ViT-S, need foundation model
- [ ] `hydrovit_backbone.py` — ViT-Base (86M params) water-specific backbone
  - [ ] 13-band input (S2 10 bands + S3 3 key bands)
  - [ ] Masked patch prediction (ViT-MAE style)
  - [ ] Spectral physics consistency loss (water-leaving radiance constraints)
- [ ] `multi_resolution.py` — S2 (10m) + S3 OLCI (300m) resolution-aware cross-attention
- [ ] `temporal_stack.py` — Temporal attention stack (5-10 images, cloud-weighted)
- [ ] `parameter_head.py` — 16-parameter output head
  - [ ] Chlorophyll-a, turbidity, Secchi depth, CDOM, TSS, TN, TP, DO, NH3, NO3, pH, temp, phycocyanin, oil probability, aCDOM, PAI
- [ ] `model.py` — Complete HydroViT encoder

### 1.3 MicroBiomeNet — Microbial Encoder (REWRITE)
- [~] `sentinel/models/microbial_encoder/` — v1 has basic transformer, need Aitchison-aware
- [ ] `aitchison_attention.py` — Attention in Aitchison geometry (CLR-space similarity)
- [ ] `sequence_encoder.py` — DNABERT-S integration for OTU sequence encoding
- [ ] `abundance_pooling.py` — Abundance-weighted soft attention pooling
- [ ] `zero_inflation.py` — Gating for structural vs sampling zeros
- [ ] `simplex_ode.py` — Neural ODE on Aitchison simplex for temporal trajectories
- [ ] `model.py` — Complete MicroBiomeNet encoder

### 1.4 ToxiGene — Molecular Encoder (REWRITE)
- [~] `sentinel/models/molecular_encoder/` — v1 has Chem2Path, need P-NET hierarchy
- [ ] `hierarchy_network.py` — P-NET-inspired biological hierarchy
  - [ ] Gene → Pathway → Biological Process → Adverse Outcome layers
  - [ ] Sparse, biology-constrained connections from Reactome + AOP-Wiki
- [ ] `cross_species.py` — Cross-species transfer via ortholog mapping
- [ ] `bottleneck.py` — Information bottleneck for minimal biomarker panel (adapt from v1)
- [ ] `model.py` — Complete ToxiGene encoder

### 1.5 BioMotion — Behavioral Encoder (NEW)
- [ ] `sentinel/models/biomotion/` — Create directory + __init__.py
- [ ] `pose_estimation.py` — SLEAP integration
  - [ ] 12 keypoints per Daphnia
  - [ ] 8 keypoints per mussel (valve positions)
  - [ ] 22 keypoints per fish
- [ ] `trajectory_encoder.py` — Diffusion-based trajectory pretraining
  - [ ] Denoising score as anomaly signal
  - [ ] Pose → behavioral features (velocity, acceleration, turning angle, phototaxis)
  - [ ] 30Hz → 1Hz summary statistics
- [ ] `multi_organism.py` — Multi-organism ensemble
  - [ ] Organism-specific trajectory encoders
  - [ ] Shared anomaly reasoning layer
- [ ] `model.py` — Complete BioMotion encoder

### 1.6 Digital Biosentinel (KEEP from v1)
- [x] Chemical encoder, species encoder, dose-response model
- [x] Calibration (temperature scaling, ECE, reliability diagrams)
- [x] Dataset + data loading
- [x] Full model with sentinel species panel

---

## Phase 2: Fusion & Escalation

### 2.1 SENTINEL-Fusion — Perceiver IO (REWRITE)
- [~] `sentinel/models/fusion/` — v1 has cross-modal attention + GRU
- [ ] `latent_array.py` — Fixed-size latent array (N=256 learned vectors)
- [ ] `perceiver_cross_attention.py` — Perceiver IO cross-attention
  - [ ] Modality tokens → latent tokens (encode)
  - [ ] Latent tokens → modality tokens (decode)
- [ ] `temporal_decay.py` — Per-modality-pair learned decay rates (adapt from v1)
- [ ] `confidence_gating.py` — Calibrated confidence-weighted gating
- [ ] `consistency_loss.py` — Cross-modal consistency loss (self-supervised alignment)
- [ ] `heads.py` — 4 output heads (anomaly, contaminant class, source attribution, escalation)
- [ ] `model.py` — Complete Perceiver IO fusion layer

### 2.2 Cascade Escalation (UPDATE from v1)
- [x] RL environment, policy, curriculum, decision tree extraction
- [ ] Update to 5 modalities (add BioMotion as Tier 0.5 — fastest response)
- [ ] Update tier definitions:
  - Tier 0: Sensor + BioMotion (continuous, fast)
  - Tier 1: + Satellite (triggered spatial analysis)
  - Tier 2: + Microbial (source attribution)
  - Tier 3: + Molecular + Digital Biosentinel (full characterization)

---

## Phase 3: Theoretical Contributions (NEW)

### 3.1 HEMA — Cross-Modal Transfer Learning
- [ ] `sentinel/models/theory/hema.py`
  - [ ] Physics-informed contrastive learning (positive pairs from physical co-occurrence)
  - [ ] Transfer error bound as function of cross-modal mutual information
  - [ ] Ben-David domain adaptation extension to heterogeneous modalities

### 3.2 Heterogeneous Temporal Causal Discovery
- [ ] `sentinel/models/theory/causal_discovery.py`
  - [ ] Cross-space conditional independence test
  - [ ] Extension of PCMCI to heterogeneous modalities
  - [ ] FDR control proof implementation

### 3.3 Sensor Placement Optimization
- [ ] `sentinel/models/theory/sensor_placement.py`
  - [ ] Submodular optimization over cross-modal conditional MI
  - [ ] GNN surrogate for gradient-based optimization
  - [ ] (1 - 1/e) approximation guarantee

### 3.4 Aitchison Neural Networks
- [ ] `sentinel/models/theory/aitchison_nn.py`
  - [ ] Universal approximation theorem implementation
  - [ ] Aitchison-space batch normalization
  - [ ] Compositional coherence proofs

### 3.5 Conformal Anomaly Detection
- [ ] `sentinel/models/theory/conformal.py`
  - [ ] Geometry-aware non-conformity scores
  - [ ] Distribution-free coverage guarantees
  - [ ] Change-point detection for regime partitioning

---

## Phase 4: Training Pipelines

### 4.1 Encoder Training
- [x] Base trainer class (trainer.py) — keep from v1
- [~] train_satellite.py — rewrite for HydroViT (MAE pretraining + physics loss)
- [ ] train_sensor.py — rewrite for AquaSSM (continuous-time SSM training)
- [ ] train_microbial.py — rewrite for MicroBiomeNet (Aitchison-aware + DNABERT-S)
- [ ] train_molecular.py — rewrite for ToxiGene (hierarchy + cross-species)
- [ ] train_biomotion.py — NEW (diffusion pretraining + fine-tuning)
- [x] train_biosentinel.py concept — keep from v1 (done in trainer)

### 4.2 Fusion Training
- [ ] train_fusion.py — rewrite for Perceiver IO staged training
  - [ ] Stage 1: Modality pairs (sensor+satellite, sensor+microbial, etc.)
  - [ ] Stage 2: Triplets
  - [ ] Stage 3: Full 5-modality system
  - [ ] Cross-modal consistency loss integration

### 4.3 Escalation Training
- [x] train_escalation.py concept — update for 5 modalities

---

## Phase 5: Evaluation & Analysis

### 5.1 Core Evaluation (partially done)
- [x] case_study.py — historical event replay (adapt for 5 modalities)
- [x] metrics.py — aggregate metrics
- [ ] Update for 31-condition ablation (all 2^5 - 1 modality subsets)

### 5.2 Novel Analyses (NEW)
- [ ] `sentinel/evaluation/ablation.py` — 31-condition ablation study
- [ ] `sentinel/evaluation/missing_modality.py` — Robustness to random modality dropout
- [ ] `sentinel/evaluation/information_analysis.py` — Cross-modal mutual information (redundancy vs complementarity)
- [ ] `sentinel/evaluation/modality_ranking.py` — Per-contaminant modality ranking
- [ ] `sentinel/evaluation/global_hotspots.py` — Global pollution hotspot mapping (HydroViT on full S2/S3 archive)
- [ ] `sentinel/evaluation/causal_chains.py` — Cross-modal causal chain discovery
- [ ] `sentinel/evaluation/figures.py` — Updated publication figures

---

## Phase 6: Public Platform (NEW)

### 6.1 Core Platform
- [~] Dashboard (v1 has React scaffold) — major expansion needed
- [ ] `sentinel/platform/citizen_qc.py` — 3-stage automated citizen data QC
  - [ ] Stage 1: Physical plausibility checks
  - [ ] Stage 2: Spatial consistency (vs satellite + nearest sensors)
  - [ ] Stage 3: Temporal consistency (per-contributor bias correction)
- [ ] `sentinel/platform/photo_analysis.py` — Ground-level photo water assessment
- [ ] `sentinel/platform/test_kit.py` — Test kit data input + validation
- [ ] `sentinel/platform/api.py` — Research API (FastAPI)

### 6.2 Dashboard Expansion
- [~] Global map view — update with citizen reports layer
- [~] Site detail panel — add BioMotion behavioral panel
- [ ] Community dashboard per water body (trends, alerts, reports)
- [ ] Photo upload + analysis UI
- [ ] Test kit input form
- [ ] Gamification (levels, badges, leaderboards)
- [ ] Water Health Score display (0-100)
- [ ] Push notification system

---

## Phase 7: Paper & Presentation

### 7.1 Publication Decomposition (6-8 papers)
- [ ] Paper 1: SENTINEL system overview (Nature Water target)
- [ ] Paper 2: AquaSSM (time series ML venue)
- [ ] Paper 3: HydroViT (remote sensing venue)
- [ ] Paper 4: MicroBiomeNet (bioinformatics venue)
- [ ] Paper 5: ToxiGene (computational toxicology venue)
- [ ] Paper 6: BioMotion (ecology/behavior venue)
- [ ] Paper 7: SENTINEL-Fusion + theoretical contributions (NeurIPS workshop)
- [ ] Paper 8: SENTINEL-DB dataset paper (scientific data venue)

### 7.2 SJWP Submission
- [ ] 10-minute presentation with narrative arc
- [ ] Live dashboard demo (case study replay)
- [ ] Judge Q&A preparation
- [ ] Final submission package

---

## Summary Statistics

| Category | v1 Done | v2 Rewrite | v2 New | Total v2 |
|----------|---------|------------|--------|----------|
| Data pipeline | 10 files | 2 rewrites | 8 new | 18 files |
| Sensor encoder | 5 files | 5 rewrites | 0 | 5 files |
| Satellite encoder | 4 files | 4 rewrites | 1 new | 5 files |
| Microbial encoder | 3 files | 3 rewrites | 3 new | 6 files |
| Molecular encoder | 3 files | 3 rewrites | 1 new | 4 files |
| BioMotion | 0 | 0 | 5 new | 5 files |
| Digital Biosentinel | 6 files | 0 | 0 | 6 files (keep) |
| Fusion | 7 files | 7 rewrites | 1 new | 8 files |
| Escalation | 5 files | 1 update | 0 | 5 files (mostly keep) |
| Theory | 0 | 0 | 5 new | 5 files |
| Training | 2 files | 1 rewrite | 5 new | 8 files |
| Evaluation | 2 files | 2 updates | 5 new | 9 files |
| Platform | 0 | 0 | 4 new | 4 files |
| Dashboard | 8 files | 3 updates | 5 new | 16 files |
| **Total** | **~55 files** | **~31 rewrites** | **~43 new** | **~104 files** |

**Estimated total code**: ~35,000-45,000 lines of Python + ~5,000-8,000 lines of TypeScript
