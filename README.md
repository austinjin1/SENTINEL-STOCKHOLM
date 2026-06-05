# SENTINEL

**Scalable Environmental Network for Temporal Intelligence and Ecological Learning**

SENTINEL is a multimodal deep learning system for real-time freshwater ecosystem monitoring and contamination early warning. It fuses five heterogeneous sensing modalities -- physicochemical sensors, satellite imagery, microbial community profiles, molecular toxicogenomics, and organism behavioral assays -- through a Perceiver IO cross-attention architecture, detecting pollution events weeks to months before they become critical.

> **Stockholm Junior Water Prize 2026** -- Austin Jin & Bryan Cheng

---

## Overview

Freshwater ecosystems face accelerating threats from agricultural runoff, industrial discharge, harmful algal blooms, and climate change. Traditional monitoring relies on sparse grab-sampling and single-modality sensors, missing early signals that span multiple biological and chemical dimensions.

SENTINEL addresses this by learning joint representations across all five sensing modalities. In validation on real USGS data, the system detected 31 contamination events across 6 USGS historical, 6 NEON, and 19 research-validated cases with a mean lead time of **32 days** (66.4 days for the 6 original USGS events), zero false positives on clean reference sites, and graceful degradation to any 2 of 5 modalities.

**SENTINEL-Lite** (HydroDenseNet) extends the system to low-cost drone-based monitoring, predicting water quality parameters directly from dual-camera (RGB + Raspberry Pi NoIR Camera Module V2) aerial imagery without requiring expensive in-situ sensors.

---

## Results

### Model Performance

| Model | Task | Key Metric | Training Data |
|-------|------|-----------|---------------|
| **AquaSSM** | Sensor anomaly detection | AUROC 0.939, RMSE 0.83 | 127K real USGS sequences, 303 stations |
| **HydroViT** | Satellite WQ prediction | R^2 0.893 (water temp) | 5,464 Sentinel-2 / in-situ pairs |
| **MicroBiomeNet** | Microbial source attribution | F1 0.899 | 20,288 EMP samples, spatial holdout |
| **ToxiGene** | Molecular toxicity classification | F1 0.492 (n=9 real GEO holdout) | GEO transcriptomics, cross-species |
| **BioMotion** | Behavioral anomaly detection | AUROC 0.807 | 29,421 EPA ECOTOX assays |
| **Perceiver IO Fusion** | Multimodal contamination detection | **AUROC 0.992** (ablation) / **0.939** (holdout) | 31-condition ablation |
| **Stream Network GNN** | Contamination propagation | AUROC 1.000, F1 0.991 | 561 NHDPlus sites, 338 edges |
| **Species Health Index** | Keystone species forecasting | R^2 0.9996, occ_acc 99.9% | 6 species, 5,462 real BioData sites |
| **Disease Forecast** | Pathogen risk prediction | AUROC 0.988, Acc 93.1% | 4 pathogens, 499K real USGS samples |
| **Digital Twin** | Ecosystem forecasting | 1d R²=0.688 (7d+ R²<0) | 10 state vars, 6 horizons; useful at 1-day only |
| **SENTINEL-Lite** | Imagery-only WQ screening | Temp R^2=0.776, DO R^2=0.463 | 57K train, 399 stations, dual-camera drone |

### Real-World Case Studies (8/10 USGS Events Detected; 31 Total Events)

Pre-registered detections on real USGS NWIS sensor data (no synthetic data):

| Event | Lead Time | USGS Records |
|-------|-----------|-------------|
| Lake Erie HAB 2023 | 59.3 days | 7,199 |
| Gulf of Mexico Dead Zone 2023 | 87.2 days | 3,486 |
| Chesapeake Bay Hypoxia 2018 | 89.8 days | 34,831 |
| Klamath River HAB 2021 | 59.2 days | 7,200 |
| Jordan Lake HAB, NC | 44.3 days | 5,755 |
| Mississippi River Salinity 2023 | 58.6 days | 4,168 |
| Iowa Nitrate Crisis 2015 | 59.3 days | 7,196 |
| Dan River Coal Ash 2014 | 13.3 days | 3,456 |

Two events (Neuse River Hypoxia 2022, Toledo Water Crisis 2014) had insufficient sensor data coverage for detection. An additional 6 NEON and 19 research-validated events bring the total to 31 detected events with a mean lead time of 32 days.

### SENTINEL-Lite (HydroDenseNet)

SENTINEL-Lite predicts water quality parameters directly from 4-band (RGB+NIR) satellite/drone imagery at 224x224 resolution, without any in-situ sensor input. The drone payload uses a dual-camera setup: a Raspberry Pi Camera Module 3 Wide for visible-spectrum imaging and a **Raspberry Pi NoIR Camera Module V2** (8MP, 1080P30) for near-infrared capture. The NIR band is critical for water quality inference -- chlorophyll absorption, turbidity scattering, and surface reflectance patterns are most discriminative in the NIR range.

| Target | R^2 | MAE | Pearson r |
|--------|-----|-----|-----------|
| Temperature | 0.776 | 2.56 C | 0.882 |
| Dissolved Oxygen | 0.463 | 1.25 mg/L | 0.757 |
| Specific Conductance | 0.442 | 1956 uS/cm | 0.675 |
| Turbidity | 0.181 | 10.96 NTU | 0.435 |

Architecture: DenseNet121 backbone with SpectralStem + CBAM attention + multi-scale FPN + per-target expert MLPs. 8.4M parameters. 57K train / 11K test from 399 USGS stations with spatial holdout (test stations geographically unseen during training).

#### Drone-to-Analysis Pipeline

SENTINEL-Lite is designed for field deployment on a multispectral drone rig. The full pipeline:

```
Drone (RPi + dual camera)
  → ROS2 publisher (/sentinel/drone/image_raw)
  → WiFi/USB tether to ground station
  → Local image cache (NPZ files with GPS + timestamp)
  → HydroDenseNet inference (GPU or CPU)
  → Anomaly scoring (EPA/WHO thresholds)
  → If alert: LoRa RF trigger → nearest fixed SENTINEL station
  → Full multimodal confirmation
```

See `sentinel/platform/` for the full pipeline implementation and `sentinel/models/waterdronenet/` for model details.

### Prospective Validation

Pre-registered predictions at 18 USGS sites with hash-verified timestamps:
- **Registration hash**: e59732...65d5
- **Sites monitored**: 18 across 8 states
- **Prediction runs**: 10 (May 26--28, 2026)
- **False alerts**: 0
- **Notable signal**: Chattahoochee River at Atlanta showed a transient anomaly peak (0.173) that self-resolved -- correctly classified as non-alert
- **Monitoring window**: 120 days from May 26, 2026

---

## Architecture

### Modality-Specific Encoders

**AquaSSM** (Sensor Encoder) -- A continuous-time state space model (Mamba-based) for irregularly-sampled multivariate sensor streams. Pre-trained with masked parameter prediction on 6 water quality parameters (DO, pH, specific conductance, temperature, turbidity, ORP) at 15-minute resolution. Multi-scale temporal kernels (1 hour to 1 year) capture both rapid transients and seasonal patterns.

**HydroViT** (Satellite Encoder) -- A water-specific vision transformer built on ViT-S/16 with CNN-ViT hybrid architecture and masked autoencoder pre-training on Sentinel-2 L2A tiles (10 spectral bands). Multi-resolution cross-attention fuses 10m and 20m bands. Predicts 9 water quality parameters from satellite imagery.

**MicroBiomeNet** (Microbial Encoder) -- An Aitchison-geometry-aware transformer for compositional microbiome data. CLR-transformed attention with Aitchison batch normalization handles the simplex constraint. Integrates DNABERT-S sequence encoder, zero-inflation gate, simplex neural ODE for temporal dynamics, and abundance-weighted pooling. Performs 8-class aquatic source attribution.

**ToxiGene** (Molecular Encoder) -- A biologically-constrained hierarchy network (gene -> pathway -> process -> outcome) for multi-label toxicity classification from RNA-seq expression profiles. Sparse Reactome-constrained layers enforce known biology. Cross-species encoder with ortholog alignment enables transfer across zebrafish, Daphnia, and fathead minnow. Information bottleneck identifies minimal gene panels (30--50 genes) achieving 90%+ accuracy.

**BioMotion** (Behavioral Encoder) -- A diffusion-pretrained trajectory encoder for multi-organism behavioral anomaly detection. Per-species keypoint configurations (Daphnia: 12, mussel: 8, fish: 22). Phase 1: diffusion denoising pre-training learns normal baselines. Phase 2: fine-tuning detects LOEC/EC50-level behavioral impairment.

### Perceiver IO Fusion

The fusion module integrates asynchronous, irregularly-arriving modality embeddings into a unified waterway state representation:

1. **Projection Bank** -- Maps each modality's native dimension to a shared 256-d embedding space
2. **Temporal Decay** -- Learned per-modality-pair exponential decay weights stale embeddings (sensor: ~2h, behavioral: ~5min, satellite: ~5 days, microbial: ~7 days, molecular: ~3 days)
3. **Confidence Gate** -- Calibrated per-modality gating suppresses unreliable inputs
4. **Perceiver Cross-Attention** -- 256 learned latents x 256-d, updated recurrently via 8-head cross-attention with 4 self-attention layers
5. **Output** -- Fused 256-d state vector with per-modality attention weights for interpretability

### Cascade Escalation Controller

A PPO-trained reinforcement learning policy that optimizes the cost-accuracy tradeoff of which modalities to activate:

| Tier | Modalities | Cost |
|------|-----------|------|
| 0 (always-on) | Sensor + Behavioral | Low |
| 1 | + Satellite | Medium |
| 2 | + Microbial | Medium-High |
| 3 | + Molecular (full pipeline) | High |

Trained with curriculum learning over 500K timesteps. Includes `extract_decision_tree` to distill the neural policy into a human-readable monitoring protocol for resource-constrained field deployment.

### Downstream Models

- **Stream Network GNN**: Graph attention network over real NHDPlus river topology (561 sites, 338 edges) for upstream-downstream contamination propagation
- **Species Health Index**: Forecasts condition of 6 keystone bioindicator species (R²=0.9996, 5,462 BioData sites)
- **Disease Forecast**: Predicts risk for 4 waterborne pathogens (AUROC=0.988, 93.1% accuracy on 499K real USGS samples)
- **Digital Twin Engine**: Neural-ODE hybrid ecosystem simulator for multi-horizon forecasting. Useful at 1-day horizon (R²=0.688); longer horizons degrade (7d+ R²<0) due to ODE trajectory divergence

### Screening & Deployment

- **SENTINEL-Lite (HydroDenseNet)**: Imagery-only water quality prediction from dual-camera drone (RGB + Raspberry Pi NoIR Camera Module V2 8MP 1080P30) -- low-cost screening without fixed sensors
- **Contrastive Pretraining**: CLIP-style InfoNCE cross-modal alignment

```
Sensor          Satellite       Microbial       Molecular       Behavioral
(AquaSSM)       (HydroViT)      (MicroBiomeNet) (ToxiGene)      (BioMotion)
   |               |                |               |               |
   +-------+-------+--------+-------+-------+-------+---------------+
           |                |               |
           v                v               v
   +------------------------------------------------+
   |           Perceiver IO Fusion Layer             |
   |   Confidence-weighted gating + cross-attention  |
   |        Learned latent array (256 x 256)         |
   +----------------+-----------------+-------------+
                    |                 |
          +---------v------+  +------v----------+
          |    Anomaly     |  |    Source        |
          |   Detection    |  |  Attribution     |
          +----------------+  +-----------------+
                    |
          +---------v--------------+
          |  Cascade Escalation    |        +------------------+
          |  Controller (PPO/RL)  |        | SENTINEL-Lite    |
          +------------------------+        | (HydroDenseNet)  |
                                            +------------------+
                    Stream Network GNN
                    Digital Twin Engine
                    Species Health Index
                    Disease Forecast
```

---

## Data Infrastructure: SENTINEL-DB

SENTINEL-DB harmonizes **390M+ environmental records** (~85 GB) from 13+ data sources spanning 105 countries and 94,000+ monitoring sites:

| Source | Records | Type |
|--------|---------|------|
| NEON Aquatic | 351.7M | Continuous high-frequency sonde data (34 sites) |
| EPA WQP | 18.27M | Discrete water quality samples |
| GRQA v1.3 | 17.99M | Harmonized global river quality |
| WQP (cyanotoxins, nutrients) | 755K | Water Quality Portal HAB-related |
| USGS BioData | 701K | Invertebrate + fish + WQP biological records |
| EPA ECOTOX | 1.23M | Ecotoxicology dose-response endpoints |
| Canada WQP | 787K | Discrete water quality samples |
| NOAA HABs (ERDDAP chl-a) | 146K | VIIRS + MODIS chlorophyll-a |
| USGS NWIS | 364K sequences | Real-time sensor time series |
| NHDPlusV2 | 561 sites, 338 edges | Stream network topology |
| Sentinel-2 | 2,986 tiles | Multispectral satellite imagery |
| EMP 16S rRNA | 20,288 | Microbiome OTU tables |
| NCBI GEO | 4 datasets | Aquatic transcriptomics |
| GBIF Freshwater | 2,355 | Bioindicator species occurrences |

---

## Evaluation Framework

SENTINEL includes 20+ experiments spanning:

| Category | Experiments |
|----------|------------|
| **Core detection** | Multimodal case studies (historical events), baseline comparisons, EPA violation correlation |
| **Ablation** | Full 31-condition (2^5 - 1) modality subset analysis with statistical significance testing |
| **Robustness** | Missing modality degradation, cross-site generalization, label noise sensitivity |
| **Uncertainty** | MC dropout calibration, conformal prediction (94% coverage), bootstrap confidence intervals |
| **Interpretability** | Parameter attribution, causal chain discovery (PCMCI+), cross-modal alignment (CKA) |
| **Downstream** | False positive rate (0.000 on NEON reference sites), temporal persistence, pollution fingerprinting |
| **Operational** | Cascade escalation, seasonal patterns, risk index ranking, early warning ROC |
| **Predictability audit** | Honest assessment of what is/isn't learnable from each modality combination |

### Key Findings

1. **Multimodal fusion outperforms any single modality** -- AUROC 0.992 (ablation) / 0.939 (holdout) vs. 0.943 (sensor-only), p = 0.002
2. **Modalities contribute unique information** -- Near-zero mutual information between sensor and behavioral channels (MINE estimate: I = 0.01 nats)
3. **Robust to missing modalities** -- AUROC > 0.90 with only 2 of 5 modalities via confidence-weighted gating
4. **Zero false positives on clean sites** -- FPR = 0.000 across 10 NEON reference sites; 31.3x signal-to-noise ratio
5. **Biological hierarchy enables interpretability** -- ToxiGene's gene-pathway-process-outcome mapping provides causal chains
6. **DO, pH, turbidity require multimodal sensing** -- Predictability audit confirms these are unrecoverable from temperature + conductivity alone, validating the multimodal architecture

---

## Platform

SENTINEL includes a deployable platform layer:

- **REST API** (`sentinel/platform/api.py`) -- FastAPI serving real-time assessment, anomaly alerts, time-series queries, and model inference
- **Streamlit Dashboard** (`sentinel/dashboard/app.py`) -- Real-time monitoring visualization
- **Citizen Science QC** -- Three-stage quality control (physical plausibility, spatial consistency, temporal consistency) for community observations
- **Photo Analysis** -- Water quality estimation from smartphone photos via HydroViT backbone
- **Docker deployment** with CI/CD pipeline (`.github/workflows/ci.yml`)
- **Prospective validation** with hash-verified pre-registration

---

## Project Structure

```
sentinel/                        # Core Python package
  data/                          # Data acquisition & preprocessing
    satellite/                   # Sentinel-2 download & tiling
    sensor/                      # USGS NWIS sensor time series
    microbial/                   # 16S rRNA community data
    molecular/                   # Toxicogenomics expression data
    ecotox/                      # EPA ECOTOX dose-response data
    behavioral/                  # Daphnia/fish trajectory data
    sentinel_db/                 # Unified database (schema, ontology, spatial indexing)
    alignment/                   # Geographic co-location linking
    case_studies/                # Historical contamination event data
    splits.py                    # Spatial/temporal holdout splitting
  models/                        # Neural network architectures
    sensor_encoder/              # AquaSSM
    satellite_encoder/           # HydroViT
    microbial_encoder/           # MicroBiomeNet
    molecular_encoder/           # ToxiGene
    biomotion/                   # BioMotion
    fusion/                      # Perceiver IO + MoME fusion
    escalation/                  # PPO cascade controller
    graph/                       # Stream Network GNN
    waterdronenet/               # SENTINEL-Lite (HydroDenseNet)
    twin/                        # Digital Twin Engine
    biology/                     # Species Health, Disease Forecast, ARG Surveillance
    digital_biosentinel/         # Dose-response prediction
    theory/                      # Conformal prediction, causal discovery
  training/                      # Training loops
  evaluation/                    # 20-experiment evaluation suite
  platform/                      # REST API, citizen science QC, photo analysis
  dashboard/                     # Streamlit monitoring dashboard
  utils/                         # Configuration, logging

scripts/                         # Standalone scripts
  train_*.py                     # Training scripts for each model
  benchmark_*.py                 # SOTA comparison benchmarks
  exp*.py                        # Numbered + named experiments
  download_*.py                  # Data acquisition from public sources
  prospective_validation.py      # Live pre-registered predictions
  run_all.py                     # Full training orchestrator

results/                         # Reproducible experiment outputs (JSON/CSV)
  benchmarks/                    # Per-model holdout metrics
  prospective/                   # Pre-registered predictions + evaluations

configs/default.yaml             # All hyperparameters and data paths
```

---

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate sentinel

# Install package
pip install -e .
```

### Data Acquisition

All training data is freely available from public sources. No proprietary or restricted data is used.

| Modality | Source | Access |
|----------|--------|--------|
| Sensor | USGS NWIS (~3,000 stations) | `dataretrieval` Python package |
| Satellite | Sentinel-2 L2A (10 bands, 10m) | Microsoft Planetary Computer STAC API |
| Microbial | Earth Microbiome Project | Qiita platform |
| Molecular | NCBI GEO (transcriptomics) | GEOparse |
| Ecotoxicology | EPA ECOTOX (~1M records) | EPA bulk download |
| Water Quality | GRQA, EPA WQP, NEON, Canada WQP | Various public APIs |
| Stream Network | NHDPlusV2 | USGS |
| Biological | USGS BioData, GBIF | Public APIs |
| Climate | NEON, NOAA | Public APIs |

```bash
# Download all data sources
python scripts/data_acquisition/download_all.py
```

### Training

```bash
# Train individual encoders
python scripts/train_aquassm.py --gpu 0
python scripts/train_hydrovit.py --gpu 1
python scripts/train_microbiomenet.py --gpu 2
python scripts/train_toxigene.py --gpu 3
python scripts/train_biomotion.py --gpu 0

# Train Perceiver IO fusion
python scripts/train_fusion.py --gpu 0

# Train downstream models
python scripts/train_stream_gnn.py --gpu 0
python scripts/train_waterdronenet.py --gpu 0
python scripts/train_species_health.py --gpu 1
python scripts/train_disease_forecast.py --gpu 2
python scripts/train_twin.py --gpu 3

# Or run everything with the orchestrator
python scripts/run_all.py
```

### Evaluation

```bash
# Run full experiment suite
python scripts/exp1_case_studies.py
python scripts/exp2_baseline_comparison.py
# ... (20+ experiments, see scripts/exp*.py)

# Prospective validation
python scripts/prospective_validation.py
```

---

## Falsifiable Claims

1. First multimodal AI system for freshwater ecosystem monitoring validated on real USGS data
2. Mean 32-day early warning across 31 contamination events (66.4-day mean for 6 original USGS case studies)
3. First pre-registered prospective water quality prediction system with hash-verified timestamps
4. 390M+ record freshwater database spanning 13+ sources
5. Graceful degradation: AUROC > 0.90 with any 2 of 5 modalities
6. Zero false positives on 10 NEON clean reference sites
7. Stream network GNN: AUROC 1.000 on real NHDPlus topology (561 sites)
8. SENTINEL-Lite: Temp R²=0.776, DO R²=0.463 from dual-camera (RGB + NoIR) aerial imagery alone (no sensor input), 399 USGS stations spatial holdout

---

## License

MIT

## Authors

Austin Jin and Bryan Cheng
