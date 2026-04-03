# Agent Dispatch Instructions for SENTINEL v2

## Overview
The v2 codebase requires ~104 files across 14 categories. The work is organized into
7 parallel agent tracks that can run simultaneously, plus 2 sequential tracks that
depend on encoder completion.

---

## Track 1: SENTINEL-DB & Data Pipeline (Agent 1)
**Priority: HIGHEST — everything depends on this**

### Files to create/modify:
1. `sentinel/data/sentinel_db/ontology.py` — Unified parameter ontology
   - Build mapping from 10,000+ parameter names (EPA/USGS/EU naming conventions) to ~500 canonical parameters
   - Exact matching + fuzzy string matching + fallback rules
   - Export as JSON lookup table
   
2. `sentinel/data/sentinel_db/spatial.py` — H3 spatial hashing
   - Use h3-py library for hexagonal grid indexing
   - Cross-source deduplication within H3 cells + temporal windows
   - Satellite pixel co-registration to in-situ stations

3. `sentinel/data/sentinel_db/quality.py` — Quality tier system (Q1-Q4)
   - Q1: Certified lab measurements with full QA/QC
   - Q2: Automated sensors with calibration records
   - Q3: Citizen science (default entry tier)
   - Q4: Derived/estimated values
   - Promotion rules (Q3→Q2 via validation against satellite + sensor baselines)

4. `sentinel/data/sentinel_db/linking.py` — Cross-modality linking
   - For each in-situ measurement: extract satellite pixel, find co-located microbiome samples, cross-reference ECOTOX
   - Spatial tolerance: configurable (default 500m for satellite, 10km for microbial)
   - Temporal tolerance: configurable (default ±3h for satellite, ±30d for microbial)

5. `sentinel/data/sentinel_db/ingest.py` — Multi-source data ingest
   - EPA Water Quality Portal (https://www.waterqualitydata.us/)
   - EU Waterbase
   - GEMS/Water (GEMStat)
   - Citizen science (FreshWater Watch)
   - Standardize all to SENTINEL-DB schema

6. `sentinel/data/sentinel_db/schema.py` — Data models (Pydantic)
   - WaterQualityRecord, SatelliteObservation, MicrobialSample, TranscriptomicSample, BehavioralRecording
   - LinkedRecord (multi-modal at same location-time)

7. Update `sentinel/data/satellite/download.py` — Add Sentinel-3 OLCI support
8. Update `sentinel/data/molecular/download.py` — Add AOP-Wiki hierarchy download
9. Add `sentinel/data/behavioral/download.py` — DaphBASE + ethology datasets

---

## Track 2: AquaSSM — Sensor Encoder (Agent 2)

### Files to create:
1. `sentinel/models/sensor_encoder/aqua_ssm.py` — Core SSM
   - Continuous-time selective state space (extend Mamba to continuous time)
   - Learned step size: Δt = fθ(gap, x_prev) where gap = t_k - t_{k-1}
   - When gap is small (15 min) → near-linear dynamics
   - When gap is large (6 hours) → complex state transitions
   - Multi-scale bank: parallel SSM channels with characteristic timescales
     initialized at log-spaced intervals (1h, 4h, 12h, 2d, 7d, 30d, 90d, 365d)
   - Gated mixing layer to combine scales

2. `sentinel/models/sensor_encoder/physics_constraints.py`
   - DO_max(T) = 14.62 - 0.3898*T + 0.006969*T² - 5.897e-5*T³ (saturation curve)
   - pH-alkalinity coupling via carbonate equilibrium
   - Conductivity ≈ k * TDS relationship
   - Implemented as soft penalty losses added during training

3. `sentinel/models/sensor_encoder/sensor_health.py`
   - Auxiliary classification head on SSM hidden state
   - Classes: normal, drift, fouling, failure, calibration_needed
   - When flagged, down-weight that sensor's anomaly contribution
   - Training: simulate failures on clean data (drift = linear trend, fouling = noise increase, failure = constant output)

4. `sentinel/models/sensor_encoder/mpp.py` — Adapt masked parameter prediction to SSM backbone
5. `sentinel/models/sensor_encoder/model.py` — Complete AquaSSM with all components

---

## Track 3: HydroViT — Satellite Encoder (Agent 3)

### Files to create:
1. `sentinel/models/satellite_encoder/hydrovit_backbone.py`
   - ViT-Base (86M params): 12 layers, 12 heads, embed_dim 768
   - Input: multi-band image patches (S2: 10 bands at 10m, S3: 21 bands at 300m)
   - MAE pretraining: mask 75% of patches, reconstruct
   - Spectral physics consistency loss: predicted water-leaving radiance must satisfy
     semi-analytical constraints (e.g., Rrs(B4)/Rrs(B3) within physical bounds for water)
   - JRC Global Surface Water mask to extract water-only pixels

2. `sentinel/models/satellite_encoder/multi_resolution.py`
   - Resolution-aware cross-attention between S2 (10m) and S3 (300m)
   - S3 provides: daily temporal coverage, 21 spectral bands (water-specific wavelengths)
   - S2 provides: spatial detail, 10 bands
   - Cross-attention: S3 tokens as queries, S2 tokens as keys/values (and vice versa)

3. `sentinel/models/satellite_encoder/temporal_stack.py`
   - Process 5-10 images at same location over time
   - Temporal cross-attention weighted by cloud-free confidence
   - Separate persistent features (bathymetry) from transient (pollution plumes)

4. `sentinel/models/satellite_encoder/parameter_head.py`
   - 16-parameter regression head
   - Parameters: chl-a, turbidity, Secchi, CDOM, TSS, TN, TP, DO, NH3, NO3, pH, temp, phycocyanin, oil_prob, aCDOM, PAI
   - PAI = multivariate deviation from site-specific seasonal baseline

5. `sentinel/models/satellite_encoder/model.py` — Complete HydroViT

---

## Track 4: MicroBiomeNet — Microbial Encoder (Agent 4)

### Files to create:
1. `sentinel/models/microbial_encoder/aitchison_attention.py`
   - Attention computed in Aitchison geometry (CLR coordinates)
   - Key: taxon-taxon similarity in compositional space, NOT Euclidean
   - Aitchison inner product: <clr(x), clr(y)> for query-key similarity
   - Aitchison batch normalization

2. `sentinel/models/microbial_encoder/sequence_encoder.py`
   - DNABERT-S integration for 16S rRNA sequence encoding
   - Each OTU → DNA sequence → DNABERT-S embedding → phylogenetic-aware representation
   - Pre-trained DNABERT-S weights from HuggingFace

3. `sentinel/models/microbial_encoder/abundance_pooling.py`
   - Soft attention pooling: weight sequence embeddings by CLR-transformed abundances
   - Captures both "who is there" (taxonomy) and "how much" (abundance)
   - Output: single sample embedding

4. `sentinel/models/microbial_encoder/zero_inflation.py`
   - Gating mechanism: P(structural zero | co-occurring taxa) vs P(sampling zero)
   - Structural zeros → hard mask (taxon truly absent)
   - Sampling zeros → impute with learned prior conditioned on neighbors

5. `sentinel/models/microbial_encoder/simplex_ode.py`
   - Neural ODE on Aitchison tangent space
   - Model community dynamics as continuous trajectory on simplex
   - Detect anomalous trajectory deviations signaling pollution onset
   - Replaces the VAE from v1

6. `sentinel/models/microbial_encoder/model.py` — Complete MicroBiomeNet

---

## Track 5: ToxiGene + BioMotion (Agent 5)

### ToxiGene files:
1. `sentinel/models/molecular_encoder/hierarchy_network.py`
   - P-NET architecture: Gene → Pathway → Biological Process → Adverse Outcome
   - Connections are sparse, constrained by Reactome + AOP-Wiki
   - Interpretable by design: activation patterns = known toxicological mechanisms

2. `sentinel/models/molecular_encoder/cross_species.py`
   - Ortholog mapping: zebrafish ↔ Daphnia magna ↔ fathead minnow ↔ rainbow trout
   - Shared gene embedding space aligned by orthology
   - Transfer learning from data-rich (zebrafish: thousands of GEO datasets) to data-poor species

3. `sentinel/models/molecular_encoder/bottleneck.py` — Adapt from v1
4. `sentinel/models/molecular_encoder/model.py` — Complete ToxiGene

### BioMotion files:
5. `sentinel/models/biomotion/__init__.py`
6. `sentinel/models/biomotion/pose_estimation.py` — SLEAP integration
7. `sentinel/models/biomotion/trajectory_encoder.py` — Diffusion pretraining
8. `sentinel/models/biomotion/multi_organism.py` — Ensemble + shared reasoning
9. `sentinel/models/biomotion/model.py` — Complete BioMotion

---

## Track 6: Fusion (Perceiver IO) + Escalation Update (Agent 6)

### Fusion files:
1. `sentinel/models/fusion/latent_array.py` — 256 learned latent vectors
2. `sentinel/models/fusion/perceiver_attention.py` — Perceiver IO cross-attention
3. `sentinel/models/fusion/temporal_decay.py` — Per-modality-pair decay (adapt from v1)
4. `sentinel/models/fusion/confidence_gating.py` — Calibrated confidence weighting
5. `sentinel/models/fusion/consistency_loss.py` — Cross-modal self-supervised alignment
6. `sentinel/models/fusion/heads.py` — 4 output heads
7. `sentinel/models/fusion/model.py` — Complete Perceiver IO fusion

### Escalation updates:
8. Update `sentinel/models/escalation/environment.py` — 5 modalities, new tier defs
9. Update `sentinel/models/escalation/policy.py` — Larger state space

---

## Track 7: Theory + Evaluation + Platform (Agent 7)

### Theory files:
1. `sentinel/models/theory/__init__.py`
2. `sentinel/models/theory/hema.py` — HEMA contrastive alignment
3. `sentinel/models/theory/causal_discovery.py` — Heterogeneous causal discovery
4. `sentinel/models/theory/sensor_placement.py` — Submodular optimization
5. `sentinel/models/theory/aitchison_nn.py` — Aitchison universal approximation
6. `sentinel/models/theory/conformal.py` — Conformal anomaly detection

### Evaluation files:
7. `sentinel/evaluation/ablation.py` — 31-condition study
8. `sentinel/evaluation/missing_modality.py` — Robustness analysis
9. `sentinel/evaluation/information_analysis.py` — MI estimation
10. `sentinel/evaluation/global_hotspots.py` — Global mapping
11. `sentinel/evaluation/causal_chains.py` — Causal chain discovery

### Platform files:
12. `sentinel/platform/__init__.py`
13. `sentinel/platform/citizen_qc.py` — 3-stage QC
14. `sentinel/platform/photo_analysis.py` — Photo-based assessment
15. `sentinel/platform/api.py` — FastAPI research API

---

## Sequential Track: Training Pipelines (after encoders done)
1. `sentinel/training/train_sensor.py` — AquaSSM training
2. `sentinel/training/train_satellite.py` — HydroViT training (rewrite)
3. `sentinel/training/train_microbial.py` — MicroBiomeNet training
4. `sentinel/training/train_molecular.py` — ToxiGene training
5. `sentinel/training/train_biomotion.py` — BioMotion training
6. `sentinel/training/train_fusion.py` — Perceiver IO staged training
7. `sentinel/training/train_escalation.py` — Updated RL training

## Sequential Track: Dashboard Expansion (after platform done)
1. Update all existing dashboard components for 5 modalities
2. Add citizen science UI components
3. Add BioMotion behavioral visualization panel
4. Add Water Health Score display
5. Add community dashboard features
