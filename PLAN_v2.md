# SENTINEL v2 — Amended Research Plan Alignment

## Summary of Changes (Original PDF → Amended DOCX)

The amended plan is a **major scope expansion**. Key differences:

| Aspect | Original (PDF) | Amended (DOCX) |
|--------|----------------|-----------------|
| **Name** | Scalable Environmental Network for Temporal Intelligence and Ecological Learning | Synergistic Environmental Tracking through Integrated Learning |
| **Modalities** | 4 (satellite, sensor, microbial, molecular) | **5** (adds BioMotion — behavioral video) |
| **Dataset** | Ad-hoc per-source downloads | **SENTINEL-DB**: 750M+ records, unified ontology, H3 spatial hashing, quality tiers, modality linking |
| **Sensor Encoder** | TCN (Temporal Convolutional Network) | **AquaSSM**: Continuous-time State Space Model (Mamba-based), multi-scale temporal decomposition, physics-informed constraints |
| **Satellite Encoder** | ViT-S/16 + SSL4EO-S12 + UPerNet | **HydroViT**: Water-specific foundation model, ViT-MAE pretraining on water pixels, S2+S3 multi-resolution fusion, 16-parameter output head, spectral physics consistency loss |
| **Microbial Encoder** | Simple transformer + VAE | **MicroBiomeNet**: Aitchison-aware attention on simplex, DNABERT-S sequence encoding, zero-inflation handler, neural ODE for simplex trajectories |
| **Molecular Encoder** | Chem2Path MLP + info bottleneck | **ToxiGene**: P-NET biological hierarchy network (gene→pathway→process→AOP), cross-species transfer via ortholog mapping, AOP-constrained architecture |
| **5th Modality** | None | **BioMotion**: Diffusion-pretrained trajectory encoder, SLEAP pose estimation (Daphnia/mussel/fish), multi-organism ensemble |
| **Fusion** | Cross-modal attention + GRU (256-dim state) | **Perceiver IO** with temporal memory (256 latent vectors), confidence-weighted gating, cross-modal consistency loss |
| **Digital Biosentinel** | Computational dose-response model (ECOTOX) | Kept as sub-component but **BioMotion replaces it as 5th encoder** |
| **Theory** | None explicit | **5 novel theoretical contributions**: HEMA, heterogeneous causal discovery, sensor placement, Aitchison NNs, conformal anomaly detection |
| **Platform** | React dashboard (research demo) | **Public-facing web app**: citizen science, photo-based analysis, test kit input, gamification, automated QC pipeline |
| **Evaluation** | 11 ablation conditions | **31 ablation conditions** (all 2^5 - 1 modality subsets) + missing modality robustness + cross-modal information analysis |
| **Compute** | ~100 A100-hours | **~1,400 A100-hours** |
| **Target venues** | SJWP only | SJWP + NeurIPS 2026 Workshop + Nature Water + ES&T |

---

## Gap Analysis: What We Have vs What We Need

### ✅ KEEP AS-IS (still valid under amended plan)
- `sentinel/data/ecotox/` — ECOTOX download + preprocessing (Digital Biosentinel still exists)
- `sentinel/models/digital_biosentinel/` — dose-response model (retained, feeds into fusion)
- `sentinel/models/escalation/` — RL cascade controller (same concept, adapt to 5 modalities)
- `sentinel/utils/config.py` and `sentinel/utils/logging.py` — universal utilities
- `configs/default.yaml` — needs updating but structure is fine
- Project scaffolding (pyproject.toml, environment.yml, .gitignore)

### 🔄 REWRITE (architecture fundamentally changed)
1. **`sentinel/models/sensor_encoder/`** — Replace TCN with AquaSSM
   - Continuous-time selective state space (Mamba + Neural CDE)
   - Multi-scale temporal decomposition (bank of SSM channels, log-spaced 1hr→1yr)
   - Physics-informed state constraints (DO-temp, pH-alkalinity, conductivity-TDS)
   - Sensor health sentinel auxiliary head
   - Masked parameter prediction pretraining (same concept, new backbone)

2. **`sentinel/models/satellite_encoder/`** — Replace ViT-S with HydroViT
   - Water-specific ViT-MAE pretraining with spectral physics consistency loss
   - Multi-resolution S2 (10m) + S3 OLCI (300m, 21 bands) fusion
   - Temporal attention stack (5-10 images, cloud-weighted)
   - 16-parameter output head (not just anomaly classes)

3. **`sentinel/models/microbial_encoder/`** — Replace with MicroBiomeNet
   - Aitchison-aware attention (CLR-space similarity, not Euclidean)
   - DNABERT-S sequence encoding per OTU
   - Abundance-weighted soft attention pooling
   - Zero-inflation gating (structural vs sampling zeros)
   - Neural ODE on simplex for temporal trajectories (replaces VAE)

4. **`sentinel/models/molecular_encoder/`** — Replace Chem2Path with ToxiGene
   - P-NET biological hierarchy (gene→pathway→process→AOP)
   - Sparse, biology-constrained connections from Reactome/AOP-Wiki
   - Cross-species transfer via ortholog mapping (zebrafish→Daphnia→fathead minnow)
   - Information bottleneck layer (keep this concept)

5. **`sentinel/models/fusion/`** — Replace attention+GRU with Perceiver IO
   - Fixed latent array of 256 vectors (not single 256-dim state)
   - Perceiver IO cross-attention: modality tokens → latent tokens → back
   - Temporal decay attention with learned per-modality-pair decay rates
   - Confidence-weighted gating from calibrated encoder scores
   - Cross-modal consistency loss for self-supervised alignment

### 🆕 NEW (doesn't exist yet)
1. **`sentinel/data/sentinel_db/`** — SENTINEL-DB harmonization pipeline
   - Unified ontology layer (10,000+ parameter names → 500 canonical)
   - H3 hexagonal spatial hashing for deduplication
   - Quality tier assignment (Q1-Q4)
   - Modality linking engine (cross-source co-registration)
   - EU Waterbase + GEMS/Water + citizen science ingest

2. **`sentinel/models/biomotion/`** — BioMotion encoder (5th modality)
   - SLEAP pose estimation pipeline (12 keypoints Daphnia, 8 mussel, 22 fish)
   - Diffusion-based trajectory pretraining (denoise corrupted movement)
   - Multi-organism ensemble with shared anomaly reasoning layer
   - Behavioral feature extraction (velocity, acceleration, turning angle, phototaxis)

3. **`sentinel/models/theory/`** — Theoretical contributions
   - HEMA: physics-informed contrastive alignment across modalities
   - Heterogeneous causal discovery (cross-space conditional independence)
   - Sensor placement optimization (submodular + GNN surrogate)
   - Aitchison universal approximation + batch normalization
   - Conformal anomaly detection with distribution-free guarantees

4. **`sentinel/platform/`** — Public-facing platform
   - Citizen science data ingestion + 3-stage QC pipeline
   - Photo-based water analysis (ground-level HydroViT branch)
   - Test kit data input + validation
   - Community dashboards per water body
   - Gamification (levels, badges, leaderboards)
   - Research API

5. **Training scripts** for all new architectures
6. **Expanded evaluation** (31 ablation conditions, causal chain discovery, global hotspot mapping)

---

## Updated Implementation Plan

### Phase 0: Infrastructure & SENTINEL-DB (Weeks 1-3)
Build the data foundation first — everything else depends on it.

### Phase 1: Individual Encoders (Weeks 2-7, parallel)
All 5 encoders can be developed simultaneously:
- AquaSSM (sensor)
- HydroViT (satellite)
- MicroBiomeNet (microbial)
- ToxiGene (molecular)
- BioMotion (behavioral)
Plus Digital Biosentinel (kept from v1)

### Phase 2: Fusion & Escalation (Weeks 6-9)
- Perceiver IO fusion layer
- RL cascade controller (updated for 5 modalities)
- Cross-modal consistency loss training

### Phase 3: Theory & Analysis (Weeks 7-11)
- Implement theoretical contributions
- Run 31-condition ablation
- Cross-modal information analysis
- Global hotspot mapping

### Phase 4: Platform & Presentation (Weeks 9-14)
- Public-facing web app
- Citizen science pipeline
- Paper writing (6-8 decomposed publications)
- SJWP presentation

---

## Compute Budget (Updated)

| Task | A100-Hours |
|------|-----------|
| AquaSSM pretraining | ~100 |
| HydroViT pretraining (water-MAE) | ~200 |
| HydroViT fine-tuning | ~50 |
| MicroBiomeNet pretraining | ~30 |
| ToxiGene training | ~20 |
| BioMotion pretraining | ~50 |
| Digital Biosentinel | ~10 |
| Fusion (staged) | ~150 |
| Escalation RL | ~10 |
| Ablation (31 conditions) | ~200 |
| Global hotspot inference | ~100 |
| Buffer/experiments | ~480 |
| **Total** | **~1,400** |
