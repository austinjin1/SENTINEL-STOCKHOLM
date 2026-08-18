# SENTINEL

**Scalable Environmental Network for Temporal Intelligence and Ecological Learning**

SENTINEL is a multimodal deep learning system for freshwater ecosystem monitoring and contamination early warning. It fuses five heterogeneous sensing modalities -- physicochemical sensors, satellite imagery, microbial community profiles, molecular toxicogenomics, and organism behavioral assays -- through a Perceiver IO cross-attention architecture, with a calibrated sensor-stream anomaly detector for contamination early warning.

> **Stockholm Junior Water Prize 2026** -- Austin Jin & Bryan Cheng

---

## Overview

Freshwater ecosystems face accelerating threats from agricultural runoff, industrial discharge, harmful algal blooms, and climate change. Traditional monitoring relies on sparse grab-sampling and single-modality sensors, missing early signals that span multiple biological and chemical dimensions.

SENTINEL addresses this by learning joint representations across the five sensing modalities. On **real NEON continuous sensor data**, a label-free, per-site-calibrated AquaSSM detector (self-supervised NEON fine-tuned) holds a **4.6% held-out false-positive rate** on clean reference sites and gives a **median 29-day early warning** on the dissolved-oxygen hypoxia episodes that meet that specificity bar (**5 of 12** real NEON episodes; it over-fires on non-stationary sites, which we report). The five encoders are individually benchmarked on real held-out data; multimodal fusion is reported honestly — including the cases where it does **not** beat the best single encoder.

**SENTINEL-Lite** (HydroDenseNet) is an imagery-only screening model that predicts water quality parameters from 4-band (RGB+NIR) imagery without in-situ sensors. It is trained and evaluated on real **Sentinel-2 satellite** tiles; a low-cost dual-camera drone payload (RGB + RPi NoIR) is described as *proposed deployment hardware*, not a data source used in this work.

---

## Results

### Model Performance

All metrics below are on real, held-out data. Items marked *simulation* or
*circular* are reported honestly as such and are **not** validation of real-world
skill (see [Honest Scope](#honest-scope-what-is-and-isnt-validated)).

| Model | Task | Key Metric (real, held-out) | Data |
|-------|------|-----------|---------------|
| **AquaSSM** | Sensor reconstruction + label-free anomaly detection | Recon RMSE 0.83; lead-time detector **median 29-day** early warning at **4.6% held-out FPR** on clean controls (NEON, **5/12** episodes validated; over-fires on non-stationary sites; self-supervised NEON fine-tuned) | 127K real USGS sequences; NEON sonde |
| **HydroViT** | Satellite WQ prediction | rebuilt on **real** Sentinel-2 × in-situ pairs, station holdout: **water-temp R² 0.729** (DO 0.58, SpCond 0.59); **chl-a R² 0.32** with red-edge bands B05–B07 (up from 0.18 on 4-band, CI excludes 0) — replaces the retracted fabricated 0.893 | 12k real Sentinel-2 tiles + red-edge |
| **MicroBiomeNet** | Microbial source attribution | true **geographic** holdout macro-F1 **0.676** using transferable **family/genus taxonomic features** (up from 0.405 on raw OTUs — higher taxa generalize cross-region). The old "0.831" was a per-sample random split (leakage) | full EMP 16S rRNA release |
| **ToxiGene** | Molecular toxicity classification | valid real **zebrafish** GEO (12 series): within-study AUROC 0.86; **cross-study LOSO 0.64** via per-sample rank-normalization (batch correction), up from 0.55 raw — a real, if modest, cross-study transfer; replaces the invalid 0.492 | NCBI GEO, real metadata labels |
| **BioMotion** | ECOTOX effect classification | AUROC **0.85** (effect-group-free GBM; 0.87 with) / **0.84** novel-chemical, beating the 3M-param diffusion encoder (0.807; 0.42 standalone). Signal is mostly study-design (concentration/duration spread) and survives dropping the endpoint-category features. | 29,421 EPA ECOTOX Daphnia assays |
| **Perceiver IO Fusion** | Multimodal contamination detection | On microcystin (optically-invisible cyanotoxin), fusion beats every single modality under station-grouped CV — 3-way (satellite optics + in-situ pigment + physicochemical drivers temp/P) **0.82** vs 0.68 best single, *each modality adds* (CI-significant; negative control confirms real signal). **But it is region-specific**: under a true cross-region (leave-one-HUC-out) holdout, all models — single and fusion — collapse to chance (~0.49). So the modalities *are* complementary, but the pigment→toxin relationship doesn't transfer to unseen watersheds (deployment needs per-region calibration). On chl-a/DO, fusion only ties (single-modality-dominated). | Sentinel-2 + in-situ + WQP drivers |
| **Stream Network GNN** | Contamination propagation | *Simulation only* — trained on synthetic propagation labels; not validated on real events | Real NHDPlus topology (561 sites, 338 edges) |
| **Species Health Index** | Keystone species occupancy | full network **retrained leakage-free** (the "AUROC 1.000" was circular — input held the per-species counts): honest macro-AUROC **0.949** (mostly a geographic range prior; env-only floor ~0.85). Widespread mussel 0.88 is the genuine bioindicator signal | 4 species (2 always-absent dropped), real BioData |
| **Disease Forecast** | Pathogen risk index | rule-based WHO/CDC/EPA index; labels are closed-form functions of the sensor inputs (*circular* — not a skill metric) | 4 pathogens, real USGS samples |
| **Digital Twin** | Ecosystem forecasting | 1d R²=0.688 (persistence wins beyond 1d — twin overall R²=0.07 vs persistence 0.92); uncertainty head now **conformal-calibrated to 90% coverage** (was 14%) | 10 state vars, 6 horizons |
| **SENTINEL-Lite** | Imagery-only WQ screening | Temp R^2=0.776, DO R^2=0.463 (Sentinel-2 imagery; drone payload is *proposed* hardware) | 57K train, 399 stations, spatial holdout |

### Honest Scope: what is and isn't validated

**Genuinely validated on real, held-out data:** AquaSSM (recon RMSE 0.83 + the
label-free NEON lead-time detector, 5/12 episodes); BioMotion (ECOTOX GBM 0.85);
the satellite chl-a spectral branch (0.716); **HydroViT water-temp R² 0.729**
(rebuilt on real Sentinel-2 × in-situ pairs, station holdout); **Species Health
macro-AUROC 0.949** (retrained leakage-free). Plus the negative results (digital
twin < persistence, predictability audit R²<0, ARG R²≈0).

**Real but region-limited:** the **multimodal fusion win on microcystin** — 3-way
(optics + pigment + drivers) beats every single modality under station-grouped CV
(0.82 vs 0.68, each modality adds), demonstrating genuine complementarity — but it
**collapses to chance under a cross-region holdout**, so it is a within-region
result, not a transferable capability. On chl-a/DO, fusion only *ties* the best single.

**Real but weak / heavily caveated:** MicroBiomeNet (true geographic-holdout
macro-F1 **0.405** on the full EMP release — up from 0.304, but cross-region
transfer is genuinely hard; the old 0.831 was random-split leakage); ToxiGene
(real zebrafish GEO — within-study AUROC 0.86 but **cross-study only ~0.55**, so
exposure signatures don't transfer); Species-Health skill is mostly a geographic
range prior (env-only floor ~0.85); BioMotion's label partly reflects study design.

**Reported as simulation / proof-of-concept (not validated detection):** the
Stream-Network GNN (synthetic propagation labels), conformal coverage (real 0.9367,
guarantee not met), lagged-correlation "causal chains" (0/375 validated), the
disease-forecast risk index (circular labels), and the SENTINEL-Lite drone
(hardware concept; imagery is real Sentinel-2).

**At genuine ceiling (honest negatives, reported as findings):** the Digital Twin
loses to persistence beyond 1 day (uncertainty head conformal-calibrated to 90%
coverage so at least the error bars are honest); cross-study molecular-toxicity
transfer is near chance.

**Retracted (computed on synthetic data, removed):** the "0.992 fusion beats
single modality" claim, the sensor↔behavioral mutual-information figure, the
contrastive Recall@1, and the "31 events / 32-day / zero false positives"
case-study summary.

### Real-World Case Studies — Calibrated Lead-Time Detector (NEON)

The sensor-stream early-warning result is the **label-free AquaSSM
reconstruction-residual detector**, per-site calibrated to a target 5% sustained
false-positive rate and evaluated on **real NEON continuous water-quality data**
(dissolved-oxygen hypoxia episodes with a verified DO < 4 mg/L onset). Specificity
is measured on held-out clean periods; lead time is credited **only** on episodes
that meet the FPR bar — a trigger-happy detector's "early" alarm is not a real
early warning.

Tested on **12 NEON DO-hypoxia episodes** (doubled from 6 by adding the unused
NEON sites; the site list is pre-committed and **all** episodes are reported,
including failures).

| Episode (NEON site) | Onset | Held-out FPR | Lead | Specificity-validated |
|---------------------|-------|--------------|------|------------------------|
| KING | 2024-07-31 | 0.0% | 36.8 d | yes |
| MAYF | 2025-04-17 | 4.7% | 31.8 d | yes |
| BIGC | 2025-08-01 | 4.8% | 29.2 d | yes |
| PRIN | 2024-10-25 | 4.6% | 22.5 d | yes |
| COMO | 2026-01-30 | 4.9% | 4.0 d | yes |
| ARIK / MCDI / CRAM / GUIL / BARC / LIRO / LECO | — | 4.7–73.8% | — | no — over-fire or not detected early |

- **Specificity:** pooled held-out sustained FPR on the 6 truly-clean control
  sites = **4.6%** (95% CI 4.3–4.9%); but the detector **over-fires on
  non-stationary sites** (7/12 episodes + 2 noisy controls exceed the 5% bar) —
  reported, not hidden.
- **Early warning:** median **29.2 days** on the **5/12** specificity-validated
  episodes (KING, MAYF, BIGC, PRIN, COMO; IQR 22.5–31.8). Self-supervised NEON
  fine-tuning of the reconstruction backbone (masked-parameter prediction on
  clean windows, **no labels**) is what made the detector usable on NEON.
- This is the AquaSSM **sensor** encoder only, on real NEON data — not a
  multimodal or forecast model. The earlier "31 events / 32-day / zero false
  positives" figure was produced on a synthetic simulation harness and is retracted.

Reproduce: `scripts/experiments/trackA_leadtime/{01_extract_episodes,05_finetune_neon,02_detect,03_report}.py`.

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
- **Sustained alarms at the deployment threshold**: 0 (note: the calibrated detector's honestly-measured held-out FPR is 4.6%, not zero — see Case Studies)
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

- **Stream Network GNN**: Graph attention network over real NHDPlus river topology (561 sites, 338 edges) for upstream-downstream contamination propagation. *Trained and evaluated on synthetic propagation labels — simulation only, not validated on real events.*
- **Species Health Index**: Forecasts condition/occupancy of 6 keystone bioindicator species (5,462 real BioData sites). *Under a leakage-free per-site holdout the honest macro-AUROC is 0.949 (the earlier pooled R²=0.9996 was a circular-label artifact and is withdrawn); this is largely a geographic range prior, with an environmental-feature floor near 0.85.*
- **Disease Forecast**: A rule-based WHO/CDC/EPA pathogen **risk index**. *The pathogen "labels" are closed-form functions of the sensor inputs (circular), so the high AUROC is not a measure of predictive skill.*
- **Digital Twin Engine**: Neural-ODE hybrid ecosystem simulator for multi-horizon forecasting. Useful at 1-day horizon (R²=0.688); longer horizons degrade (7d+ R²<0) due to ODE trajectory divergence

### Screening & Deployment

- **SENTINEL-Lite (HydroDenseNet)**: Imagery-only water quality prediction from 4-band (RGB+NIR) imagery -- low-cost screening without fixed sensors. Trained/evaluated on real **Sentinel-2** tiles; the dual-camera drone payload (RGB + RPi NoIR) is *proposed deployment hardware*.
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

SENTINEL-DB harmonizes **~390M raw environmental records** (~85 GB) from 14 public data sources spanning 105 countries and 94,000+ monitoring sites. The 390M figure is a **raw, pre-QC** count dominated by NEON high-frequency sonde data (351.7M raw → ~52M after NEON quality-flag filtering); usable post-QC volume is far smaller:

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
| **Uncertainty** | MC dropout calibration, conformal prediction (real sensor coverage 0.9367 — the 95% guarantee is **not** met), bootstrap confidence intervals |
| **Interpretability** | Parameter attribution, lagged partial-correlation screening (0/375 chains validated; *not* PCMCI+), cross-modal alignment (CKA) |
| **Downstream** | False-positive rate (**4.6%** held-out on clean NEON reference sites), temporal persistence, pollution fingerprinting |
| **Operational** | Cascade escalation, seasonal patterns, risk index ranking, early warning ROC |
| **Predictability audit** | Honest assessment of what is/isn't learnable from each modality combination |

### Key Findings

1. **Multimodal fusion genuinely combines complementary modalities — but only within a calibrated region.** On microcystin (an optically-invisible cyanotoxin), a 3-modality fusion (satellite optics + in-situ pigment + physicochemical drivers) beats the best single modality under station-grouped CV — AUROC **0.82, +0.08** over the best single (a literature-index satellite arm at ~0.71, above pigment 0.684); the CI-clean load-bearing step is the 2-way→3-way physicochemical-driver increment (+0.047, CI excludes zero, negative-control confirmed). The modalities are non-redundant because the toxin is optically invisible (phycocyanin↔microcystin 0.73 vs chl-a↔microcystin 0.08) and its production is driven by temperature/nutrients. **However**, under a true cross-region (leave-one-HUC-out) holdout, *all* models collapse to chance (~0.49) — and **no feature transform** (region-conditioning, within-site-relative, rank) recovers it. The missing variable is the **toxin-producing strain fraction (mcyD genotype)**, which pigment/optics/drivers cannot see and which differs by watershed (literature: mcyD abundance predicts microcystin better than any pigment). The genuine transfer path is therefore SENTINEL's own **microbial/metagenomic modality**, gated on co-locating metagenomics with cyanotoxin sites. On **chl-a**/**DO**/turbidity, fusion only *ties* the best single (single-modality-dominated, a literature-consistent redundancy). The earlier "0.992 fusion outperforms single modality" was synthetic and is **retracted**.
2. **Calibrated specificity on real clean sites** -- held-out sustained false-positive rate **4.6%** (95% CI 4.3–4.9%) on clean NEON reference sites. Not zero, but honestly measured at a fixed operating point.
3. **Real early warning where the signal supports it** -- label-free median **29-day** lead on **5/12** specificity-validated DO-hypoxia episodes (NEON); the other 7 over-fire on clean water or aren't detected early. Self-supervised NEON fine-tuning of the reconstruction backbone (no labels) is what made the detector usable on NEON.
4. **Honest negatives are reported, not hidden** -- digital twin useful only at the 1-day horizon (a naive baseline wins beyond it); predictability audit R²<0 for DO/pH/turbidity; ARG surveillance R²≈−0.008; foundation/MoME fusion AUROC 0.37–0.42.
5. **Biological hierarchy as interpretability, not discovery** -- ToxiGene's gene→pathway→process→outcome mapping is a design feature for interpretability, not a validated causal-discovery result.

---

## Platform

SENTINEL includes a deployable platform layer:

- **REST API** (`sentinel/platform/api.py`) -- FastAPI serving real-time assessment, anomaly alerts, time-series queries, and model inference
- **Streamlit Dashboard** (`sentinel/dashboard/app.py`) -- Real-time monitoring visualization
- **Citizen Science QC** -- Three-stage quality control (physical plausibility, spatial consistency, temporal consistency) for community observations
- **Photo Analysis** -- Water quality estimation from smartphone photos via HydroViT backbone
- **Docker deployment** (`Dockerfile`, `docker-compose.yml`)
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
    fusion/                      # Perceiver IO fusion
    escalation/                  # PPO cascade controller
    graph/                       # Stream Network GNN
    waterdronenet/               # SENTINEL-Lite (HydroDenseNet)
    twin/                        # Digital Twin Engine
    biology/                     # Species Health, Disease Forecast
    theory/                      # Conformal prediction, causal discovery
  training/                      # Training loops
  evaluation/                    # 20-experiment evaluation suite
  platform/                      # REST API, citizen science QC, photo analysis
  dashboard/                     # Streamlit monitoring dashboard
  utils/                         # Configuration, logging

scripts/                         # Standalone scripts, grouped by purpose
  data_acquisition/              # download_*, ingest, co-registration (see DATABASE.md)
  preprocessing/                 # process_/prepare_/expand_ → training-ready data
  training/                      # train_*.py — one per model
  benchmarks/                    # SOTA comparison benchmarks
  experiments/                   # exp*.py — the 20 paper analyses + case studies
  evaluation/                    # held-out / conformal / causal evaluation
  deployment/                    # prospective validation + SENTINEL-Lite/Mini tests
  figures/                       # figure generation
  pipeline/                      # run_all.py orchestrator, results compilation

results/                         # Reproducible experiment outputs (JSON/CSV)
  benchmarks/                    # Per-model holdout metrics
  prospective/                   # Pre-registered predictions + evaluations

configs/default.yaml             # All hyperparameters and data paths
DATABASE.md                      # How to download/build every data source
```

---

## Setup

```bash
# Create the environment and install the package
conda env create -f environment.yml
conda activate physiformer
pip install -e .
```

> All commands below assume `PYTHONNOUSERSITE=1` so the conda PyTorch is used
> (a stray `~/.local` torch can otherwise shadow it). Prefix commands with it or
> `export PYTHONNOUSERSITE=1` once per shell.

### 1. Get the data

All training data is public. **See [DATABASE.md](DATABASE.md)** for the full
per-source download + preprocessing guide. Quick start:

```bash
export PYTHONNOUSERSITE=1
python scripts/data_acquisition/download_all.py     # core sources
# then per-modality preprocessing, e.g.
python scripts/preprocessing/process_usgs_to_training.py
```

### 2. Train the models

```bash
export PYTHONNOUSERSITE=1

# Five modality encoders
python scripts/training/train_aquassm.py
python scripts/training/train_hydrovit.py
python scripts/training/train_microbiomenet.py
python scripts/training/train_toxigene.py
python scripts/training/train_biomotion.py

# Fusion + downstream / deployment models
python scripts/training/train_fusion.py
python scripts/training/train_stream_gnn.py
python scripts/training/train_twin.py
python scripts/training/train_species_health.py
python scripts/training/train_disease_forecast.py
python scripts/training/train_hydrodensenet.py     # SENTINEL-Lite

# Or run the whole pipeline in dependency order
python scripts/pipeline/run_all.py                 # add --dry-run to preview
```

### 3. Reproduce the experiments

The 20 numbered analyses from the paper live in `scripts/experiments/`:

```bash
export PYTHONNOUSERSITE=1
python scripts/experiments/exp1_case_studies_real.py     # historical events
python scripts/experiments/exp9_bootstrap_ci.py          # bootstrap 95% CIs
python scripts/experiments/exp12_multimodal_integration.py  # 31-condition ablation
# ... see scripts/experiments/ for exp1–exp20 and named case studies

# SOTA benchmarks, held-out evaluation, prospective validation
python scripts/benchmarks/benchmark_aquassm.py
python scripts/evaluation/conformal_real_eval.py
python scripts/deployment/prospective_validation.py
```

## Pretrained models

The trained checkpoints for every model in the paper are published on the
Hugging Face Hub: **[huggingface.co/bryan7264/SENTINEL](https://huggingface.co/bryan7264/SENTINEL)**
(`aquassm.pt`, `hydrovit.pt`, `microbiomenet.pt`, `toxigene.pt`, `biomotion.pt`,
`sentinel_fusion.pt`, `stream_gnn.pt`, `digital_twin.pt`, `hydrodensenet.pt`,
`species_health.pt`, `waterborne_disease.pt`). Architectures are defined under
`sentinel/models/`.

---

## Falsifiable Claims

1. A multimodal architecture for freshwater monitoring with five encoders each benchmarked on real, held-out public data.
2. Label-free median **29-day** early warning on **5/12** specificity-validated DO-hypoxia episodes (NEON), at **4.6%** held-out false-positive rate on clean controls (over-fires on non-stationary sites — reported).
3. Multimodal fusion is reported honestly and **does not** beat the best single encoder on real co-located data (chl-a held-out AUROC 0.65 vs sensor-only 0.64).
4. ~390M **raw** records (pre-QC; ~52M post-NEON-QC) from 14 public sources.
5. SENTINEL-Lite: Temp R²=0.776, DO R²=0.463 from 4-band **Sentinel-2** imagery alone (no sensor input), 399 stations spatial holdout. The drone payload is *proposed* hardware.
6. Honest negatives, reported in full: digital-twin R²<0 beyond 1 day; predictability audit R²<0 for DO/pH/turbidity; ARG surveillance R²≈−0.008.

**Reported as simulation / proof-of-concept (not validated detection):** the Stream-Network GNN (synthetic propagation labels on real NHDPlus topology), the conformal pipeline (real coverage 0.9367, 95% guarantee not met), the lagged-correlation causal screening (0/375 validated), the disease-forecast risk index (circular labels), and the SENTINEL-Lite drone hardware concept (Sentinel-2 proxy).

---

## License

MIT

## Authors

Austin Jin and Bryan Cheng
