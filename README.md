# SENTINEL

**Scalable Environmental Network for Temporal Intelligence and Ecological Learning**

SENTINEL is a multimodal deep learning framework for real-time water quality monitoring and contamination detection. It fuses five heterogeneous sensing modalities — physicochemical sensors, satellite imagery, microbial community profiles, molecular toxicogenomics, and organism behavioral assays — through a Perceiver IO cross-attention architecture, detecting pollution events earlier and more reliably than any single data source alone. 

> **Stockholm Junior Water Prize 2026 Submission** — Austin Jin & Bryan Cheng

---

## Results

| Encoder | Task | Performance | Training Data | Threshold |
|---|---|---|---|---|
| **AquaSSM** | Sensor anomaly detection | AUROC = 0.920 | 20,000 USGS NWIS sequences, 1,115 stations | > 0.85 |
| **HydroViT** | Satellite water quality regression | R² = 0.749 (water temp) | 4,202 Sentinel-2 / in-situ pairs | > 0.55 |
| **MicroBiomeNet** | Microbial source attribution | F1 = 0.913 (8-class) | 20,288 EMP 16S rRNA samples | > 0.70 |
| **ToxiGene** | Contaminant classification | F1 = 0.894 (8-class) | 4 GEO datasets + 268K ECOTOX records | > 0.80 |
| **BioMotion** | Behavioral anomaly detection | AUROC = 1.000 | 17,074 EPA ECOTOX Daphnia assays | > 0.80 |
| **Perceiver IO Fusion** | Multimodal detection | **AUROC = 0.992** | 31-condition ablation, 10 events | > best single |

The full 5-modality fusion (AUROC = 0.992) significantly outperforms the best single modality (sensor-only, AUROC = 0.943; *p* = 0.002, paired permutation test). The system maintains AUROC > 0.90 with as few as 2 modalities available, degrading gracefully across 100 random-drop trials.

---

## Architecture

### Modality-Specific Encoders

**AquaSSM** (Sensor Encoder) — A continuous-time state space model for irregularly-sampled multivariate sensor streams. Pre-trained with masked parameter prediction (MPP) on 6 water quality parameters (DO, pH, specific conductance, temperature, turbidity, ORP) at 15-minute resolution. Multi-scale temporal kernels (1 hour to 1 year) capture both rapid transients and seasonal patterns. Physics-constraint loss enforces thermodynamic consistency.

**HydroViT** (Satellite Encoder) — A water-specific vision foundation model built on ViT-S/16 with masked autoencoder (MAE) pre-training on 2,986 Sentinel-2 L2A tiles (10 spectral bands). Multi-resolution cross-attention fuses 10m and 20m bands. A temporal attention stack integrates revisit sequences. The water quality regression head predicts 9 parameters with positive skill, including water temperature (R² = 0.749), TSS (R² = 0.160), and nitrate (R² = 0.155). Spectral physics loss enforces known band-ratio relationships.

**MicroBiomeNet** (Microbial Encoder) — An Aitchison-geometry-aware transformer for compositional microbiome data. CLR-transformed attention with Aitchison batch normalization handles the simplex constraint of relative abundance data. Integrates a DNABERT-S sequence encoder, zero-inflation gate for sparse OTU tables, simplex neural ODE for temporal dynamics, and abundance-weighted pooling. Performs 8-class aquatic source attribution (freshwater natural/impacted, saline, sediments, soil runoff, animal fecal, plant-associated).

**ToxiGene** (Molecular Encoder) — A biologically-constrained hierarchy network (P-NET architecture: gene → pathway → process → outcome) for multi-label toxicity classification directly from RNA-seq expression profiles. Sparse Reactome-constrained linear layers enforce known biology. Cross-species encoder with ortholog alignment enables transfer across zebrafish, Daphnia, and fathead minnow. Information bottleneck identifies minimal gene panels (30-50 genes) achieving 90%+ of full-panel accuracy. To our knowledge, ToxiGene is the first supervised method for this task.

**BioMotion** (Behavioral Encoder) — A diffusion-pretrained trajectory encoder for multi-organism behavioral anomaly detection. Pose encoder with sinusoidal timestamps and per-species keypoint configurations (Daphnia: 12, mussel: 8, fish: 22). Phase 1: diffusion denoising pre-training learns normal concentration-response baselines. Phase 2: fine-tuning detects LOEC/EC50-level behavioral impairment. Cross-organism attention enables ensemble inference across species.

### Perceiver IO Fusion

The `PerceiverIOFusion` module integrates asynchronous, irregularly-arriving modality embeddings into a unified waterway state representation:

1. **Projection Bank** — Maps each modality's native dimension to a shared 256-d embedding space
2. **Embedding Registry** — Maintains the latest embedding and timestamp per modality, handling asynchronous updates
3. **Temporal Decay** — Learned per-modality-pair exponential decay weights stale embeddings (sensor: ~2h half-life, behavioral: ~5min, satellite: ~5 days, microbial: ~7 days, molecular: ~3 days)
4. **Confidence Gate** — Calibrated per-modality gating suppresses unreliable inputs
5. **Perceiver Cross-Attention** — A learned latent array (256 latents x 256-d) serves as a compressed waterway state, updated recurrently across observation events via 8-head cross-attention with 4 self-attention layers
6. **Output** — Fused state vector (256-d), updated latent state, and per-modality attention weights for interpretability

### Cascade Escalation Controller

A PPO-trained (Stable Baselines 3) reinforcement learning policy that optimizes the cost-accuracy tradeoff of which modalities to activate:

| Tier | Modalities | Cost |
|------|-----------|------|
| 0 (always-on) | Sensor + Behavioral | Low |
| 1 | + Satellite | Medium |
| 2 | + Microbial | Medium-High |
| 3 | + Molecular (full pipeline) | High |

State representation: 256-d fused state + 5 modality flags + 4 tier one-hots + 2 scalars (267-d total). Trained with curriculum learning (easy → mixed → hard events) over 500K timesteps. Includes `extract_decision_tree` to distill the neural policy into a human-readable monitoring protocol for resource-constrained field deployment.

```
Sensor          Satellite       Microbial       Molecular       Behavioral
(AquaSSM)       (HydroViT)      (MicroBiomeNet) (ToxiGene)      (BioMotion)
   │               │                │               │               │
   └───────┬───────┴────────┬───────┴───────┬───────┴───────────────┘
           │                │               │
           ▼                ▼               ▼
   ┌────────────────────────────────────────────────┐
   │           Perceiver IO Fusion Layer             │
   │   Confidence-weighted gating + cross-attention  │
   │        Learned latent array (256 x 256)         │
   └──────────────┬─────────────────┬───────────────┘
                  │                 │
        ┌─────────▼──────┐  ┌──────▼──────────┐
        │    Anomaly     │  │    Source        │
        │   Detection    │  │  Attribution     │
        └────────────────┘  └─────────────────┘
                  │
        ┌─────────▼──────────────┐
        │  Cascade Escalation    │
        │  Controller (PPO/RL)   │
        └────────────────────────┘
```

---

## SENTINEL-DB

SENTINEL-DB harmonizes **390M+ environmental records** (~85 GB) from 13 data sources spanning 105 countries and 94,000+ monitoring sites into a unified schema:

| Source | Records | Type |
|--------|---------|------|
| NEON Aquatic | 351.7M | Continuous high-frequency sonde data (34 sites, 24 months) |
| EPA WQP | 18.27M | Discrete water quality samples |
| GRQA v1.3 | 17.99M | Harmonized global river quality |
| EPA ECOTOX | 1.23M | Ecotoxicology dose-response endpoints |
| Canada WQP | 787K | Discrete water quality samples |
| USGS NWIS | 364K sequences | Real-time sensor time series |
| Sentinel-2 | 2,986 tiles | Multispectral satellite imagery |
| EPA NARS | 2,111 | National aquatic resource surveys |
| WHO/World Bank | 18K | WASH indicators |
| NCBI GEO | 4 datasets | Aquatic transcriptomics |
| EMP 16S rRNA | 20,288 | Microbiome OTU tables |
| GBIF Freshwater | 2,355 | Bioindicator species occurrences |
| Behavioral Assay | 5,000 trajectories | Daphnia motion data |

**Key design features:**
- **Unified parameter ontology** — Maps 10,000+ raw parameter names across EPA WQP, USGS NWIS, EU Waterbase, GEMStat, and citizen science to ~500 canonical parameters with standardized units. Includes a unit conversion table and fuzzy-match fallback.
- **H3 hexagonal spatial indexing** (resolution 8) — Enables cross-source spatial queries and satellite co-registration within configurable tolerance (default: 500m spatial, 3h temporal).
- **Quality tiers** — Q1 (ISO-certified lab), Q2 (calibrated in-situ sensor), Q3 (citizen science), Q4 (derived/modelled). Quality-aware weighting propagates through training and inference.
- **Pydantic v2 schema** — Type-safe records with canonical parameter name, value, unit, UTC timestamp, lat/lon, H3 hex index, source ID, and quality tier.

---

## Evaluation Framework

SENTINEL includes a comprehensive evaluation suite spanning 20 experiments:

| Category | Experiments |
|----------|------------|
| **Core detection** | Multimodal case studies (10 historical events), baseline comparisons, EPA violation correlation |
| **Ablation** | Full 31-condition (2^5 - 1) modality subset analysis with statistical significance testing |
| **Robustness** | Missing modality degradation (100 random-drop trials), cross-site generalization, label noise sensitivity |
| **Uncertainty** | MC dropout calibration, conformal prediction with distribution-free coverage guarantees, bootstrap CIs |
| **Interpretability** | Parameter attribution, causal chain discovery, cross-modal alignment (CKA), attention visualization |
| **Downstream** | False positive rate on clean reference sites, temporal persistence, pollution fingerprinting, discovery scan |
| **Operational** | Cascade escalation analysis, seasonal patterns, risk index ranking, early warning ROC, sensor placement optimization |

All experiment results are stored as reproducible JSON/CSV outputs in `results/`.

---

## Platform

SENTINEL includes a deployable platform layer (`sentinel/platform/`):

- **REST API** (`api.py`) — FastAPI application serving real-time water quality assessment, anomaly alerts, time-series queries, and model inference endpoints
- **Citizen Science QC** (`citizen_qc.py`) — Three-stage quality control pipeline (physical plausibility → spatial consistency → temporal consistency) for community-contributed water quality observations
- **Photo Analysis** (`photo_analysis.py`) — Estimates water quality from smartphone photos via HydroViT/ResNet backbone, cross-referenced against satellite-derived values
- **Test Kit Validation** (`test_kit.py`) — Calibrates home water quality test kits against reference measurements, applies per-kit bias correction, and ingests validated results into SENTINEL-DB

---

## Project Structure

```
sentinel/                        # Core Python package
├── data/                        # Data acquisition & preprocessing
│   ├── satellite/               # Sentinel-2 download & tiling
│   ├── sensor/                  # USGS NWIS sensor time series
│   ├── microbial/               # 16S rRNA community data
│   ├── molecular/               # Toxicogenomics expression data
│   ├── ecotox/                  # EPA ECOTOX dose-response data
│   ├── behavioral/              # Daphnia trajectory data
│   ├── sentinel_db/             # Unified database (schema, ontology, spatial indexing)
│   ├── alignment/               # Geographic co-location linking
│   └── case_studies/            # Historical contamination event data
├── models/                      # Neural network architectures
│   ├── sensor_encoder/          # AquaSSM — continuous-time SSM
│   ├── satellite_encoder/       # HydroViT — MAE + ViT-S/16
│   ├── microbial_encoder/       # MicroBiomeNet — Aitchison transformer
│   ├── molecular_encoder/       # ToxiGene — P-NET biological hierarchy
│   ├── biomotion/               # BioMotion — diffusion trajectory encoder
│   ├── digital_biosentinel/     # Dose-response prediction (~1M ECOTOX records)
│   ├── fusion/                  # Perceiver IO cross-modal fusion
│   ├── escalation/              # PPO cascade controller
│   └── theory/                  # Conformal prediction, causal discovery, Aitchison NN
├── training/                    # Training loops for each encoder + fusion + escalation
├── evaluation/                  # 20-experiment evaluation suite
├── platform/                    # REST API, citizen science QC, photo analysis
└── utils/                       # Configuration, logging
scripts/                         # Standalone scripts
├── data acquisition             # Download from USGS, EPA, GRQA, GEO, EMP, etc.
├── training                     # Per-encoder and fusion training scripts
├── benchmarking                 # SOTA comparisons for each encoder
└── experiments                  # exp1-exp20 + named experiments
results/                         # Reproducible experiment outputs (JSON/CSV)
configs/                         # YAML configuration (hyperparameters, data, evaluation)
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

| Modality | Source | Access Method |
|----------|--------|---------------|
| Sensor | USGS NWIS (~3,000 stations) | `dataretrieval` Python package |
| Satellite | Sentinel-2 L2A (10 bands, 10m) | Microsoft Planetary Computer STAC API |
| Microbial | Earth Microbiome Project | Qiita platform |
| Molecular | NCBI GEO (transcriptomics) | GEOparse |
| Ecotoxicology | EPA ECOTOX (~1M records) | EPA bulk download |
| Water Quality | GRQA, EPA WQP, NEON, Canada WQP | Various public APIs |

```bash
# Download all data sources
python scripts/data_acquisition/download_all.py
```

### Training

Training follows a staged pipeline: (1) self-supervised pre-training per encoder, (2) supervised fine-tuning per encoder, (3) fusion training, (4) escalation controller training.

```bash
# Stage 1-2: Train individual encoders
python -m sentinel.training.train_sensor --config configs/default.yaml
python -m sentinel.training.train_satellite --config configs/default.yaml
python -m sentinel.training.train_microbial --config configs/default.yaml
python -m sentinel.training.train_molecular --config configs/default.yaml
python -m sentinel.training.train_biomotion --config configs/default.yaml
python -m sentinel.training.train_biosentinel --config configs/default.yaml

# Stage 3: Train Perceiver IO fusion
python -m sentinel.training.train_fusion --config configs/default.yaml

# Stage 4: Train cascade escalation controller
python -m sentinel.training.train_escalation --config configs/default.yaml
```

### Evaluation

```bash
# Run case studies on historical contamination events
python -m sentinel.evaluation.case_study --config configs/default.yaml

# Run 31-condition modality ablation
python -m sentinel.evaluation.ablation --config configs/default.yaml
```

---

## Key Findings

1. **Multimodal fusion outperforms any single modality** — AUROC 0.992 vs. 0.943 (sensor-only), detecting all 10 historical contamination events
2. **Modalities contribute unique information** — Near-zero mutual information between sensor and behavioral channels (MINE estimate: *I* = 0.01 nats), confirming independent sensing
3. **Robust to missing modalities** — AUROC > 0.90 with only 2 of 5 modalities; graceful degradation via confidence-weighted gating
4. **Biological hierarchy enables interpretability** — ToxiGene's gene → pathway → process → outcome mapping provides causal chains; information bottleneck identifies field-deployable 30-50 gene panels
5. **Zero false positives on clean sites** — FPR = 0.000 across 10 NEON reference sites; 31.3x signal-to-noise ratio between contaminated and clean temporal windows

---

## Configuration

All hyperparameters, data paths, model architectures, training schedules, evaluation settings, and case study definitions are specified in `configs/default.yaml`. Key configurable sections:

- **Data** — Sensor parameters, satellite bands, microbial features, molecular pathways, behavioral keypoints, SENTINEL-DB spatial/temporal tolerances
- **Models** — Architecture choices, embedding dimensions, number of layers/heads, diffusion steps, fusion latent array size
- **Training** — Per-encoder pre-training and fine-tuning schedules (learning rates, batch sizes, epochs, optimizers, schedulers), fusion two-stage training, escalation PPO hyperparameters with curriculum phases
- **Evaluation** — 31 ablation conditions, 15 named ablation configurations, 8 evaluation metrics, 3 case study definitions (Lake Erie HAB, East Palestine derailment, Chesapeake Bay blooms)

---

## License

MIT

## Authors

Austin Jin and Bryan Cheng
