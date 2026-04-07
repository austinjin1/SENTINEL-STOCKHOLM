SENTINEL
Synergistic Environmental Tracking through Integrated Learning


Comprehensive Research Plan
A Multimodal AI Platform for Early Water Pollution Detection
April 2026
Target Venues: Stockholm Junior Water Prize 2026
Secondary: NeurIPS 2026 Workshop, Nature Water, ES&T

Table of Contents
1. Executive Summary & Thesis
2. SENTINEL-DB: The Largest Multimodal Water Quality Dataset
3. Architecture I: AquaSSM — State Space Model for Sensor Time Series
4. Architecture II: HydroViT — Foundation Model for Satellite Water Quality
5. Architecture III: MicroBiomeNet — Compositional Deep Learning for Aquatic Metagenomics
6. Architecture IV: ToxiGene — Pathway-Informed Transcriptomic Toxicity Classifier
7. Architecture V: BioMotion — Behavioral Anomaly Detection from Video
8. SENTINEL-Fusion: Cross-Modal Temporal Attention with Cascading Alerts
9. Novel Theoretical Contributions
10. Evaluation Framework & Benchmarks
11. The SENTINEL Platform: Public-Facing Application
12. Expected Analyses, Findings & Insights
13. Timeline & Compute Budget
14. Risk Mitigation
15. References & Key Datasets

1. Executive Summary & Thesis

SENTINEL is the first multimodal AI framework that unifies five fundamentally different environmental sensing modalities — physicochemical sensor time series, satellite remote sensing, aquatic metagenomics, transcriptomic stress biomarkers, and organismal behavioral signals — into a single coherent system for early water pollution detection. The project makes contributions at four levels:

Data. SENTINEL-DB aggregates and harmonizes 750M+ discrete water quality records from EPA, USGS, EU Waterbase, GEMS/Water, and citizen science platforms with petabyte-scale satellite imagery, 50,000+ aquatic transcriptomic datasets, and aquatic microbiome sequences — the largest multimodal water quality dataset ever assembled.
Architecture. Five novel, modality-specific neural architectures — each state-of-the-art on its respective data type — feed into SENTINEL-Fusion, a Perceiver IO–derived cross-modal temporal attention framework that handles asynchronous, irregularly-sampled, heterogeneous data streams. Each architecture introduces specific technical novelties: continuous-time state space dynamics for sensor data, compositional geometry-aware attention for metagenomics, pathway-constrained sparse networks for transcriptomics, and diffusion-pretrained trajectory encoders for behavioral data.
Theory. Five novel theoretical contributions: (1) cross-modal environmental transfer learning with physics-informed contrastive alignment, (2) heterogeneous temporal causal discovery across modality types, (3) information-theoretic joint optimization of multi-modal sensor placement, (4) Aitchison-aware neural networks for compositional microbiome data, and (5) multimodal conformal anomaly detection with distribution-free coverage guarantees.
Platform. A public-facing web application where anyone can input water observations (photos, test kit readings, GPS-tagged reports) and receive instant AI-powered pollution risk assessments, source attribution, and recommended actions. Citizen-submitted data flows back into SENTINEL-DB through an automated quality control pipeline, creating a continuously improving system.

Core thesis: No single sensing modality can detect all pollution types early enough. Biology runs a massively redundant detection system across molecular, organismal, community, chemical, and visual channels. SENTINEL is the first AI framework that reads all these channels simultaneously, fuses them through learned temporal attention, and translates their collective signal into actionable intelligence for both scientists and the public.

2. SENTINEL-DB: The Largest Multimodal Water Quality Dataset

2.1 Data Sources & Scale
SENTINEL-DB unifies data across five modality classes from 14+ primary sources. No existing dataset spans more than two of these modalities. The harmonization itself is a novel contribution.

Source
Records
Modality
Coverage
Access
EPA Water Quality Portal
430M+
In-situ chem/phys
USA (2M+ sites)
REST API
USGS NWIS
185M+
In-situ chem/phys
USA (393K sites)
REST API
EU Waterbase
60M+
In-situ chem/phys
39 EU countries
DISCODATA
GEMS/Water (GEMStat)
50M+
In-situ chem/phys
91 countries
Zenodo CC BY
GRQA (harmonized)
17M+
River quality (43 params)
Global rivers
Zenodo
Copernicus Marine + S2/S3
Petabytes
Satellite WQ products
Global daily
API + GEE
MODIS Ocean Color
Petabytes
Satellite Chl-a, TSM
Global daily 250m
GEE
HydroLAKES / LakeATLAS
1.4M lakes
Lake characteristics
Global
GEE
Earth Microbiome Project
~27K samples
16S/18S/ITS/shotgun
Global environments
Qiita
NCBI GEO/SRA (aquatic)
~50K datasets
Transcriptomics/metagen.
Global
E-utilities
EPA ECOTOX
1.1M+ tests
Ecotox endpoints
14K species, 13K chems
Bulk download
FreshWater Watch
50K+ datasets
Citizen nutrients/turbidity
Global
Web platform
WHO/UNICEF JMP
5K+ national sets
WASH indicators
200+ countries
washdata.org
GLEON Lake Observatories
Millions
High-freq sensor
Global lake network
Varies


2.2 Novel Harmonization Pipeline
The harmonization challenge is non-trivial: these sources use different parameter naming conventions (over 300 unique names for dissolved oxygen alone across EPA/USGS/EU systems), different units, different detection limits, different quality flags, and fundamentally different data models. SENTINEL-DB introduces:

Unified Ontology Layer. A parameter mapping ontology that resolves the 10,000+ unique parameter names across all sources to a canonical set of ~500 standardized parameters, using a combination of exact matching, fuzzy string matching, and an LLM-assisted disambiguation step for ambiguous cases. The ontology is released as an open resource.
Spatio-Temporal Alignment Engine. Cross-source deduplication using spatial hashing (H3 hexagonal grid) and temporal windowing. When multiple sources report the same parameter at the same location/time, records are merged with source-weighted quality scores. Satellite-derived values are co-registered to in-situ station locations with uncertainty quantification.
Modality Linking. The key novelty: linking records across modality types. For every in-situ measurement, we extract the corresponding satellite pixel values (within temporal and spatial tolerance), identify co-located microbiome samples from SRA/EMP, and cross-reference EPA ECOTOX records for species known to inhabit that water body. This creates the first dataset where a single location-time coordinate can yield sensor readings + satellite spectra + microbial community composition + known ecotoxicological thresholds.
Quality Tiers. Every record receives a quality tier (Q1–Q4) based on provenance, measurement method, QA/QC flags, and cross-source consistency. Citizen science data enters at Q3 by default and can be promoted to Q2 through automated validation against satellite and sensor baselines.

2.3 Dataset Statistics (Target)
Total discrete in-situ records: >750 million. Unique monitoring locations: >3 million globally. Satellite coverage: continuous daily global at 300m (S3 OLCI), 5-day at 10m (S2 MSI). Linked multimodal records (2+ modalities at same location-time): target >10 million. Temporal span: 1906–2026 (GEMS/Water earliest records through present). This would make SENTINEL-DB the largest water quality dataset by over an order of magnitude versus the next largest harmonized collection (GRQA at 17M records).

3. Architecture I: AquaSSM
Continuous-Time State Space Model for Water Sensor Anomaly Detection
3.1 Problem
Water quality sensor networks produce multivariate time series (dissolved oxygen, pH, conductivity, temperature, turbidity, ORP) at irregular intervals with frequent gaps from sensor failures, fouling, and maintenance. Existing approaches either discretize time (losing information) or use attention-based models that scale quadratically with sequence length (infeasible for continuous monitoring). Mamba-based models offer linear scaling but assume regular sampling.
3.2 Novel Architecture
AquaSSM combines the linear-time efficiency of structured state space models with the irregular-time handling of neural controlled differential equations. The key innovations:

Continuous-Time Selective State Space. We extend the Mamba selective scan mechanism to continuous time. Instead of discretizing the state transition matrix A with a fixed step size, AquaSSM parameterizes the step size as a learned function of the actual time gap between observations: Δt = fθ(t_k - t_{k-1}, x_{k-1}). This means the model naturally adapts its dynamics to arbitrary inter-observation intervals without interpolation or binning. When sensors report every 15 minutes, the effective Δt is small and dynamics are near-linear; when a 6-hour gap occurs, Δt is large and the model traverses more complex state transitions.
Multi-Scale Temporal Decomposition. Water quality signals contain patterns at vastly different timescales: diurnal cycles (hours), storm events (days), seasonal trends (months), and long-term degradation (years). AquaSSM uses a bank of parallel SSM channels with different characteristic timescales (learned, initialized at log-spaced intervals from 1 hour to 1 year). A gated mixing layer combines these channels, allowing the model to simultaneously capture fast anomalies and slow trends.
Sensor Health Sentinel. A dedicated auxiliary head that predicts whether each sensor is functioning correctly, trained on known maintenance records and flagged data from USGS. When the health sentinel flags a sensor as potentially drifting, that sensor's contribution to the anomaly score is down-weighted. This addresses the critical real-world problem of false alarms from sensor malfunctions.
Physics-Informed State Constraints. Soft constraints encoding known physical relationships: dissolved oxygen solubility decreases with temperature, pH and alkalinity are coupled through carbonate chemistry, conductivity relates to total dissolved solids. These constraints act as inductive biases during training, improving data efficiency and reducing physically impossible predictions.
3.3 Training
Pretraining: Self-supervised on 6M+ time series from 3,724 USGS continuous monitoring stations (following HydroGEM's approach but with the continuous-time SSM backbone instead of TCN-Transformer). Pretext task: masked parameter prediction — randomly mask 15–40% of parameters at each timestep and predict them from the remaining parameters and temporal context. This forces the model to learn inter-parameter correlations.
Fine-tuning: Supervised anomaly detection on curated pollution event labels from USGS event reports, EPA violation records, and state environmental agency incident databases. We construct a labeled dataset of ~50,000 confirmed pollution events matched to sensor time series, with event type annotations (organic pollution, chemical spill, thermal discharge, agricultural runoff, sewage overflow).
Evaluation: Benchmarked against Mamba-TS, Neural CDE, Chronos-2 (fine-tuned), HydroGEM, LSTM, Transformer. Metrics: AUROC and AUPRC for anomaly detection, F1 for event classification, mean detection lead time (how far in advance of peak pollution the model flags the anomaly), false alarm rate per station-year. Cross-basin transfer evaluation: train on Eastern US, test on Western US and European stations.

4. Architecture II: HydroViT
Vision Foundation Model for Satellite-Derived Water Quality Estimation
4.1 Problem
Satellite remote sensing can estimate water quality parameters over vast spatial extents, but existing approaches are limited to optically-active parameters (chlorophyll-a, turbidity, CDOM) through physics-based spectral algorithms. Recent work by Zhi et al. (Nature Water, 2024) showed that deep learning can predict non-optically-active parameters (nitrate, phosphate) by exploiting learned correlations — but no foundation model has been built specifically for water quality remote sensing.
4.2 Novel Architecture
Water-Specific Pretraining Objective. Unlike general Earth observation foundation models (Prithvi-EO-2.0, Clay) that pretrain on all land cover types, HydroViT pretrains exclusively on water pixels extracted from Sentinel-2 and Sentinel-3. The pretraining objective combines masked patch prediction (standard ViT-MAE) with a novel spectral physics consistency loss: predicted patches must satisfy known water-leaving radiance constraints from semi-analytical ocean color models. This physics-informed pretraining learns water-specific spectral representations rather than generic land-cover features.
Temporal Attention Stack. Water quality changes over days to weeks. HydroViT processes temporal stacks of 5–10 images at the same location and learns to separate persistent features (bathymetry, land use) from transient signals (pollution plumes, algal blooms, sediment events). A temporal cross-attention layer attends across the time dimension, weighted by cloud-free confidence scores (automatically masking cloudy frames).
Multi-Resolution Fusion. Sentinel-2 (10m, 5-day) and Sentinel-3 OLCI (300m, daily) are fused through a resolution-aware cross-attention module. S3 provides daily context and broad spectral coverage (21 bands including water-specific wavelengths); S2 provides spatial detail. The model learns to use S3 for temporal interpolation and S2 for spatial sharpening.
16-Parameter Output Head. Jointly predicts 16 water quality parameters from a single image: chlorophyll-a, turbidity, Secchi depth, CDOM, total suspended solids, total nitrogen, total phosphorus, dissolved oxygen (estimated), ammonia, nitrate, pH (estimated), water temperature, phycocyanin (cyanobacteria proxy), oil film probability, colored dissolved organic matter absorption coefficient, and a novel Pollution Anomaly Index (PAI) representing the multivariate deviation from site-specific seasonal baselines.
4.3 Training
Pretraining data: ~500M water pixel patches extracted from Sentinel-2 L2A and Sentinel-3 OLCI global archives (2015–2026) via Google Earth Engine. Water mask from JRC Global Surface Water dataset. Pretraining compute: ~200 A100-hours for the base model (ViT-Base, 86M parameters).
Fine-tuning data: In-situ measurements from SENTINEL-DB co-registered with satellite overpasses within ±3 hours and 500m. Target: 2M+ paired satellite-in situ observations globally for the 16 output parameters. This co-registration dataset alone is a novel resource.
Evaluation: Benchmarked against Prithvi-EO-2.0 (fine-tuned), Clay Foundation Model (fine-tuned), standard physics-based algorithms (C2RCC, ACOLITE), and CNN baselines. Metrics: R², RMSE, and bias for each of 16 parameters, plus spatial resolution of detectable pollution plumes. A key novel evaluation: can the model detect non-optically-active parameters (N, P) in regions with no in-situ training data? This tests whether learned spectral-chemical correlations generalize geographically.

5. Architecture III: MicroBiomeNet
Compositional Geometry-Aware Deep Learning for Aquatic Metagenomics
5.1 Problem
Microbial community composition in water shifts predictably in response to different pollution types. However, microbiome data is compositional (relative abundances that sum to 1, lying on a simplex rather than in Euclidean space), zero-inflated (most taxa are absent from most samples), and high-dimensional (thousands of OTUs). Standard neural networks ignore the compositional constraint, leading to predictions that are statistically invalid. Existing compositional-aware methods (Gordon-Rodriguez et al., NeurIPS 2022) target the human gut microbiome; no environmental water microbiome application exists.
5.2 Novel Architecture
Aitchison-Aware Attention. MicroBiomeNet operates natively on the Aitchison simplex. All internal representations use centered log-ratio (CLR) coordinates. The attention mechanism computes similarity in Aitchison geometry rather than Euclidean space, meaning the model respects the fundamental mathematical structure of compositional data. This is a first-of-kind architecture.
Abundance-Aware Sequence Encoding. Each OTU is first encoded by a DNA sequence language model (DNABERT-S) to capture phylogenetic relationships. These sequence embeddings are then weighted by their CLR-transformed abundances through a soft attention pooling layer, producing a single sample embedding that captures both "who is there" (taxonomy) and "how much" (abundance). This extends the 2025 Abundance-Aware Set Transformer with explicit compositional geometry.
Zero-Inflation Handler. A gating mechanism learns to distinguish structural zeros (taxon truly absent from the environment) from sampling zeros (taxon present but below detection limit). Structural zeros are treated as hard constraints; sampling zeros are imputed with a learned prior conditioned on co-occurring taxa. This addresses the most severe data quality issue in environmental metagenomics.
Temporal Community Trajectory Module. For sites with longitudinal sampling, MicroBiomeNet tracks community composition over time as a trajectory on the simplex. A neural ODE defined on the simplex (using the Aitchison tangent space) models community dynamics, enabling prediction of future community states and detection of anomalous trajectory deviations that signal pollution onset.
5.3 Training
Pretraining: Self-supervised contrastive learning on ~100K aquatic microbiome samples from SRA/EMP/MGnify. Positive pairs: same site at adjacent timepoints; negative pairs: different sites. Pretext task forces the model to learn which community features are site-specific versus transient.
Fine-tuning: Supervised classification on samples with paired water chemistry from SENTINEL-DB. Labels: pollution source type (agricultural runoff, sewage, industrial, mining, pristine reference). Additional regression head for predicting co-measured chemical parameters from microbial composition alone.
Key evaluation: Can microbial community composition predict pollution type and severity as well as, or better than, direct chemical measurement? If so, this validates metagenomics as a cheap, comprehensive alternative to expensive chemical testing — a finding with massive practical implications.

6. Architecture IV: ToxiGene
Pathway-Informed Transcriptomic Toxicity Classifier
6.1 Problem
When organisms are exposed to toxicants, specific gene pathways activate before any visible harm occurs. These molecular responses are highly specific to contaminant classes: CYP1A/AHR pathway activation indicates PAH exposure, metallothionein induction indicates heavy metals, vitellogenin in males indicates endocrine disruptors, etc. The question is: given a gene expression profile from a sentinel organism, can we predict the contaminant class and mechanism of toxicity?
6.2 Novel Architecture
Biological Hierarchy Network. Inspired by P-NET but adapted for ecotoxicology. The network architecture mirrors the gene → pathway → biological process → adverse outcome hierarchy from Reactome and the Adverse Outcome Pathway (AOP) framework. Connections between layers are sparse and constrained to known biological relationships. This means the model is interpretable by design: activation patterns directly correspond to known toxicological mechanisms.
Cross-Species Transfer Module. Aquatic toxicogenomics data exists across multiple model organisms (zebrafish, fathead minnow, Daphnia magna, medaka, rainbow trout) with different gene sets. ToxiGene uses ortholog mapping to align gene embeddings across species, enabling transfer learning from data-rich species (zebrafish: thousands of GEO datasets) to data-poor species.
Minimal Biomarker Panel Discovery. An information bottleneck layer between the gene input and the pathway layer learns the minimal gene set maximally diagnostic for contaminant classification. If we can show that 20–50 genes predict contaminant type as well as the full transcriptome (~20,000 genes), this panel becomes directly deployable as a cheap qPCR field kit — a translational contribution SJWP judges would value highly.
6.3 Training Data
Training data: ~5,000 transcriptomic datasets from GEO covering zebrafish, Daphnia, and fathead minnow exposed to characterized contaminants. Labels derived from EPA ECOTOX cross-referenced with GEO metadata. AOP-Wiki provides the pathway hierarchy. Validation: the zebrafish AOP-Anchored Transcriptome Analysis catalogue (78 substances, 1,099 genes, 82.9% validated accuracy) serves as an external benchmark.

7. Architecture V: BioMotion
Diffusion-Pretrained Behavioral Anomaly Detection for Aquatic Biosentinels
7.1 Problem
Aquatic organisms change behavior in response to sub-lethal toxicity long before they die: Daphnia magna alter swimming velocity and phototaxis within minutes; mussels change valve-gaping frequency; fish modify ventilation rate and schooling patterns. These behavioral signatures are the fastest-responding biological signal, but no general-purpose deep learning model exists for aquatic behavioral anomaly detection. Commercial systems (DaphTox, MFB) use simple statistical thresholds.
7.2 Novel Architecture
Diffusion-Based Trajectory Pretraining. Following the 2024 diffusion pre-training approach for fish behavior recognition, BioMotion pretrains a trajectory encoder by learning to denoise corrupted movement trajectories. This captures the latent dynamics of normal behavior without labeled data. The denoising score itself becomes an anomaly signal: trajectories that are hard to denoise (high reconstruction error) are behaviorally abnormal.
Multi-Organism Ensemble. Different organisms detect different contaminant classes: Daphnia are exquisitely sensitive to organophosphates; mussels respond to heavy metals; fish detect neurotoxins. BioMotion maintains organism-specific trajectory encoders but fuses them through a shared anomaly reasoning layer, mimicking the multi-species biomonitor concept (MFB system) but with deep learning replacing statistical thresholds.
Pose Estimation Pipeline. Integrated SLEAP-based pose estimation extracting 12 keypoints per Daphnia, 8 per mussel (valve positions), 22 per fish. Pose sequences are encoded into behavioral feature time series (velocity, acceleration, turning angle, inter-individual distance, phototactic index) at 30Hz, then downsampled to 1Hz summary statistics for the anomaly detection module.
7.3 Training
Pretraining: self-supervised on ~16,000 normal behavior trajectories from published aquatic ethology datasets and DaphBASE. Fine-tuning: exposure-response trajectories from EPA ECOTOX behavioral endpoints, supplemented with published video datasets of contaminant-exposed organisms. Evaluation: detection sensitivity (minimum detectable toxicant concentration), response latency (time from exposure to alert), and false alarm rate compared to the DaphTox commercial system threshold.

8. SENTINEL-Fusion: Cross-Modal Temporal Attention

8.1 The Core Challenge
The five modality encoders produce embeddings at radically different timescales and sampling rates: satellite imagery every 1–5 days, sensor time series every 15 minutes, behavioral data continuously at 1Hz, microbial community data weekly, and transcriptomic data sporadically. Standard multimodal fusion (early, late, or intermediate) assumes synchronized inputs. SENTINEL-Fusion must handle asynchronous, irregularly-sampled, heterogeneous streams where some modalities may be entirely missing for extended periods.
8.2 Architecture: Perceiver IO with Temporal Memory
Modality-Agnostic Latent Array. A fixed-size array of N learned latent vectors (N=256) serves as a compressed "state of the waterway" representation. When any modality encoder produces a new embedding, it updates the latent array through cross-attention (modality tokens attend to latent tokens, then latent tokens attend back). This Perceiver IO–derived design scales linearly in both input size and number of modalities.
Temporal Decay Attention. Each latent vector carries a timestamp of its last update. Attention weights between a new observation and existing latent vectors are modulated by an exponential temporal decay: recent information is weighted more heavily, but stale information from a missing modality is not discarded — it simply decays toward the learned prior. The decay rates are learned per modality pair, capturing which modality combinations have fast versus slow information decay.
Confidence-Weighted Gating. Each modality encoder outputs both an embedding and a calibrated confidence score (trained with temperature scaling). The fusion layer gates each modality's contribution by its confidence, automatically down-weighting unreliable inputs (cloudy satellite images, drifting sensors, low-coverage metagenomic samples) without explicit rules.
Cascading Alert Heads. The fused latent array feeds four output heads: (1) anomaly score (binary: is something wrong?), (2) contaminant class prediction (multi-label: what type of pollution?), (3) source attribution (where is it coming from?), and (4) recommended action (escalation tier: increase monitoring frequency, deploy sampling, issue public alert). The escalation logic is itself a learned policy trained with reinforcement learning on historical event sequences.
8.3 Cross-Modal Training
The fusion layer is trained end-to-end with all modality encoders (partially frozen after their individual pretraining). Training data: SENTINEL-DB records where 2+ modalities are available at the same location-time. We use a curriculum: first train on modality pairs (sensor+satellite, sensor+microbial, etc.), then triplets, then the full system. A novel cross-modal consistency loss encourages modality encoders to produce embeddings that are similar when they observe the same underlying water quality state, providing a self-supervised alignment signal.
8.4 Key Novelty Claim
No existing multimodal fusion architecture handles the specific combination of challenges present in environmental monitoring: (a) asynchronous sampling across modalities, (b) modalities from fundamentally different mathematical spaces (Euclidean time series, image patches, compositional simplex, graph-structured pathways, trajectory sequences), (c) missing modalities as the norm rather than exception, and (d) cascade decision-making over what additional data to collect. SENTINEL-Fusion is designed from the ground up for this setting.

9. Novel Theoretical Contributions

9.1 Cross-Modal Environmental Transfer Learning
We propose a Hierarchical Environmental Modality Alignment (HEMA) framework that learns shared representations across satellite spectra, physicochemical time series, and metagenomic compositions. The core insight: these modalities observe the same underlying environmental state through different lenses. HEMA uses physics-informed contrastive learning where positive pairs are defined not by data augmentation but by physical co-occurrence (satellite pixel and sensor reading from the same location-time should produce similar embeddings). The theoretical contribution is a bound on transfer error between modalities as a function of their mutual information with the latent environmental state, extending the Ben-David domain adaptation theory to heterogeneous modality types.
9.2 Heterogeneous Temporal Causal Discovery
We extend the PCMCI causal discovery algorithm (Runge et al., 2019) to heterogeneous multimodal data. Current causal discovery methods operate on homogeneous multivariate time series. We formalize cross-modal Granger causality where the cause variable lives in one mathematical space (e.g., spectral indices in R^n) and the effect variable lives in another (e.g., community composition on the simplex). The theoretical contribution is a conditional independence test that operates across geometric spaces, with provable false discovery rate control under stated regularity conditions. The practical application: discovering causal chains like "satellite-detected turbidity increase → dissolved oxygen drop → microbial community shift toward anaerobic taxa" with formal statistical guarantees.
9.3 Information-Theoretic Multi-Modal Sensor Placement
Given a fixed budget for monitoring infrastructure, how should one jointly allocate physicochemical sensor stations, satellite ground-truth sites, and metagenomic sampling points to maximize total information about water quality across a watershed? We formulate this as a submodular optimization problem over conditional mutual information across modality types. The theoretical contribution is a proof that the greedy algorithm achieves a (1 - 1/e) approximation guarantee even with cross-modal information terms, and a differentiable GNN surrogate that enables gradient-based optimization for large-scale watersheds.
9.4 Compositional Neural Networks for Environmental Microbiomes
We formalize the first neural network architectures that are provably compositionally coherent: predictions are invariant to the total count (sequencing depth) and equivariant under sub-compositional operations. The theoretical contribution is a universal approximation theorem for functions on the Aitchison simplex using CLR-coordinate networks with Aitchison-space batch normalization. We prove that standard neural networks applied to raw relative abundances are not even consistent estimators of compositional functionals, establishing a rigorous theoretical motivation for the MicroBiomeNet architecture.
9.5 Multimodal Conformal Anomaly Detection
We develop the first conformal prediction framework for multimodal anomaly detection with heterogeneous data types. The framework provides distribution-free coverage guarantees: if we claim an anomaly alert covers the true pollution state with 95% confidence, it does so regardless of the true data distribution, even under temporal non-stationarity. The key technical challenge is defining non-conformity scores that are meaningful across Euclidean, compositional, and graph-structured spaces. We introduce geometry-aware non-conformity scores with physics-informed constraints and prove finite-sample coverage guarantees under the assumption that the data is exchangeable within detected regimes (using change-point detection to partition the stream into approximately stationary segments).

10. Evaluation Framework & Benchmarks

10.1 Per-Modality Benchmarks

Model
Task
Baselines
Primary Metrics
AquaSSM
Sensor anomaly detection
Mamba-TS, Neural CDE, Chronos-2, HydroGEM, LSTM, Transformer
AUROC, AUPRC, Detection Lead Time, False Alarm Rate
HydroViT
Satellite WQ parameter retrieval
Prithvi-EO-2.0, Clay, C2RCC, ACOLITE, CNN
R², RMSE per parameter, Spatial Resolution
MicroBiomeNet
Pollution source classification from 16S
Random Forest on CLR, DeepSets, Abundance-Aware SetTF, PCA+LR
Macro-F1, AUROC, Brier Score
ToxiGene
Contaminant class from transcriptomics
P-NET, scGPT, Linear probe, RF on DEGs
Macro-F1, Biomarker panel size, Cross-species transfer acc.
BioMotion
Behavioral anomaly detection
DaphTox threshold, LSTM, Transformer, VAE
Sensitivity, Min detectable conc., Response latency


10.2 Fusion Evaluation
The central evaluation question: does multimodal fusion outperform any single modality? We design three evaluation protocols:

Ablation study. Systematically evaluate all 31 possible subsets of 5 modalities (5 singles, 10 pairs, 10 triples, 5 quadruples, 1 full) on a held-out test set of labeled pollution events. This quantifies the marginal information gain of each additional modality.
Missing modality robustness. Evaluate performance when modalities are randomly dropped at test time (simulating real-world sensor failures, cloudy satellite passes, unavailable metagenomic data). Plot performance degradation curves as a function of number of available modalities.
Detection lead time comparison. For pollution events where all modalities are available, compare how far in advance each individual modality versus the fused system detects the event. The hypothesis: fusion detects earlier than any single modality because different modalities have different response latencies (behavioral: minutes; sensor: hours; microbial: days; satellite: depends on overpass timing).
10.3 Novel Analyses
Cross-modal information redundancy analysis. Using mutual information estimation, quantify how much unique versus redundant information each modality contributes. This answers: are we measuring the same thing five ways, or are these genuinely complementary signals? Publication-worthy finding regardless of outcome.
Contaminant-specific modality ranking. For each contaminant class (heavy metals, pesticides, nutrients, pharmaceuticals, hydrocarbons, microplastics proxy), rank modalities by detection power. This produces a practical guide: for a given concern, which monitoring investments yield the most information?
Global pollution hotspot mapping. Apply the trained HydroViT model to the full Sentinel-2/3 archive to produce a global map of water bodies showing anomalous water quality trends (2015–2026). This analysis alone could be a high-impact publication.
Causal chain discovery. Apply the heterogeneous causal discovery algorithm to identify temporal causal relationships across modalities. Expected findings: satellite-detectable spectral changes causally precede sensor-measured chemistry changes, which causally precede microbial community shifts. If confirmed, this establishes a temporal hierarchy of environmental response that informs optimal monitoring strategies.

11. The SENTINEL Platform: Public-Facing Application

11.1 Core User Flows

Flow 1: Instant Water Assessment. Any user opens the web app, drops a pin on the map (or uses GPS), and SENTINEL returns a real-time water quality risk assessment for that location. The assessment fuses the most recent satellite-derived estimates (HydroViT), nearest sensor station data (AquaSSM), and any available community reports. Output: an overall Water Health Score (0–100), parameter-level estimates with confidence intervals, trend indicators (improving/stable/declining), and comparison to EPA/WHO standards. No test kit required — this works for any water body on Earth with satellite coverage.
Flow 2: Photo-Based Water Analysis. User uploads a photo of a water body (taken with any smartphone). A fine-tuned vision model (branch of HydroViT adapted for ground-level imagery) estimates visible parameters: turbidity, surface algal coverage, color anomalies, oil sheen probability, foam presence. Combined with GPS metadata, the photo analysis is cross-referenced with satellite and sensor data for that location to produce a fused assessment. Photos are stored (with permission) and contribute to SENTINEL-DB.
Flow 3: Test Kit Data Input. Users with home water test kits can input their readings (pH, nitrate, phosphate, dissolved oxygen, etc.) through a simple form. SENTINEL validates the reading against satellite and sensor baselines for that location (flagging likely measurement errors), incorporates it into the assessment, and stores the observation in SENTINEL-DB at quality tier Q3. Over time, validated citizen observations can be promoted to Q2.
Flow 4: Community Dashboard. Each water body (lake, river segment, bay) gets a public dashboard showing historical trends, recent alerts, community reports, satellite imagery time-lapses, and model predictions. Users can "watch" water bodies and receive push notifications when the anomaly score exceeds a threshold.
Flow 5: Research API. Full programmatic access to SENTINEL-DB, trained model weights, and inference endpoints. Researchers can submit data, run models, and access embeddings for downstream analysis. All models are open-weight; the dataset is open-access under CC BY 4.0.
11.2 Automated Citizen Data Quality Control
The single largest barrier to citizen science adoption in environmental monitoring is data quality. SENTINEL introduces a three-stage automated QC pipeline:

Stage 1: Physical Plausibility. Is the reported value within the physically possible range for that parameter at that location and season? (e.g., pH of 15 is rejected; dissolved oxygen of 20 mg/L in summer is flagged for review.)
Stage 2: Spatial Consistency. Does the value agree with the satellite-derived estimate and nearest sensor stations within a learned tolerance? Large deviations are flagged, not rejected — they might indicate real local variation or a calibration issue with the test kit.
Stage 3: Temporal Consistency. For repeat contributors, does the observation follow expected temporal patterns? A user who consistently reports pH values 0.5 units higher than the satellite baseline may have a systematic kit bias, which can be learned and corrected.
11.3 Gamification & Engagement
Drawing from iNaturalist's proven model: contributor levels (Water Watcher → Stream Sentinel → Watershed Guardian → Basin Expert), badges for consistency (monthly streaks), accuracy (validated observations), and coverage (new locations). Leaderboards per watershed. Annual reports showing each contributor's impact on data coverage. School challenges with classroom-level competitions. All engagement features designed to maximize data quality and spatial coverage, not just volume.

12. Expected Analyses, Findings & Insights

12.1 Primary Research Findings (Expected)

Finding 1: Multimodal fusion detects pollution events 2–5x earlier than any single modality. The behavioral modality fires first (minutes), followed by sensor anomalies (hours), satellite detection (days), and microbial community shift (days–weeks). Fusion leverages the fast response of behavioral signals with the diagnostic specificity of genomic signals.
Finding 2: Microbial community composition predicts pollution type comparably to chemical testing. A trained MicroBiomeNet achieves >85% accuracy on source attribution (agricultural vs. sewage vs. industrial vs. mining), suggesting that metagenomics could complement or partially replace expensive analytical chemistry for pollution characterization.
Finding 3: A panel of 20–50 genes is sufficient for contaminant class prediction. ToxiGene's information bottleneck identifies a minimal biomarker panel that performs within 5% of the full transcriptome. This panel is directly translatable to a qPCR-based field diagnostic.
Finding 4: Non-optically-active parameters are satellite-predictable through learned correlations. HydroViT achieves R² > 0.6 for nitrate and phosphate estimation from satellite imagery in regions with sufficient training data, validating and extending Zhi et al.'s (2024) finding with a foundation model approach.
Finding 5: Global pollution hotspot mapping reveals previously unmonitored degradation. Applying HydroViT to the full Sentinel-2/3 archive identifies water bodies with significant quality decline (2015–2026) that have no in-situ monitoring stations — the "blind spots" of existing monitoring networks.
12.2 Theoretical Insights

Insight 1: Modalities are more complementary than redundant. Cross-modal mutual information analysis reveals that satellite, sensor, and metagenomic modalities share less than 30% of their total information about water quality state, justifying the multimodal approach.
Insight 2: Optimal monitoring networks are heterogeneous. The sensor placement theorem shows that mixed networks (some chemical sensors, some satellite ground-truth sites, some metagenomic sampling points) achieve higher information gain per dollar than homogeneous networks of any single type.
Insight 3: Compositional-aware models significantly outperform Euclidean models for microbiome data. MicroBiomeNet's Aitchison-aware architecture outperforms standard neural networks by 8–15% on classification tasks, validating the theoretical argument for compositional coherence.
12.3 Publication Decomposition
The SENTINEL project naturally decomposes into 6–8 standalone publications:

Paper
Venue Target
Core Contribution
SENTINEL-DB: A Multimodal Water Quality Benchmark
NeurIPS Datasets & Benchmarks / Nature Scientific Data
Largest harmonized water quality dataset; ontology; benchmark tasks
AquaSSM: Continuous-Time State Space Models for Water Monitoring
ICML / NeurIPS
Novel architecture; sensor health sentinel; physics constraints
HydroViT: Foundation Model for Satellite Water Quality
ICLR / Remote Sensing of Environment
Water-specific pretraining; 16-parameter prediction; non-optical params
MicroBiomeNet: Compositional Deep Learning for Aquatic Metagenomics
NeurIPS / Nature Methods
Aitchison-aware attention; simplex neural ODEs; zero-inflation handling
SENTINEL-Fusion: Asynchronous Multimodal Environmental Monitoring
ICML / Nature Water
Perceiver IO fusion; temporal decay attention; cascade alerts
Theoretical Foundations of Multimodal Environmental AI
JMLR / Annals of Statistics
Five theoretical contributions; proofs; guarantees
SENTINEL Platform: Democratizing Water Quality Intelligence
SJWP / CHI / Nature Sustainability
Public platform; citizen science QC; gamification; global hotspot map


13. Timeline & Compute Budget

13.1 Phase Timeline

Phase
Duration
Deliverables
Compute (A100-hrs)
Phase 0: Data Collection & Harmonization
Weeks 1–3
SENTINEL-DB v1.0: harmonized in-situ + satellite co-registration
50 (data processing)
Phase 1: AquaSSM Development
Weeks 2–5
Pretrained + fine-tuned sensor anomaly model; benchmark results
200
Phase 2: HydroViT Development
Weeks 3–7
Pretrained satellite foundation model; 16-parameter evaluation
400
Phase 3: MicroBiomeNet Development
Weeks 4–7
Compositional microbiome classifier; source attribution results
150
Phase 4: ToxiGene Development
Weeks 5–8
Pathway-informed toxicity classifier; biomarker panel discovery
100
Phase 5: BioMotion Development
Weeks 5–8
Behavioral anomaly detector; benchmarked against DaphTox
100
Phase 6: SENTINEL-Fusion
Weeks 7–10
Full multimodal fusion; ablation studies; cascade alert system
300
Phase 7: Theoretical Results
Weeks 8–11
Proofs; empirical validation of theoretical bounds
50
Phase 8: Platform Development
Weeks 9–12
Web app; API; citizen data pipeline; global hotspot map
50
Phase 9: Paper Writing & SJWP Submission
Weeks 11–14
SJWP paper (20 pages); supplementary; code release
~0


Total estimated compute: ~1,400 A100-hours (~350 A100-GPU-hours across 4 GPUs = ~15 days of continuous training). Well within the capacity of 4x A100 GPUs over a 14-week period.
13.2 Critical Path
The critical path runs through Phase 0 (data) → Phase 2 (HydroViT, most compute-intensive) → Phase 6 (Fusion). AquaSSM and HydroViT can be developed in parallel. MicroBiomeNet, ToxiGene, and BioMotion are fully parallelizable. The platform can be built concurrently with model development using mock data, then connected to real models in Phase 8.

14. Risk Mitigation

Risk
Likelihood
Impact
Mitigation
Insufficient paired multimodal data
Medium
High
Focus fusion training on sensor+satellite pairs (most abundant); genomic modalities validated independently then connected via location-time linkage
HydroViT pretraining too compute-heavy
Medium
Medium
Fall back to fine-tuning Prithvi-EO-2.0 or Clay on water pixels; less novel but still effective
MicroBiomeNet compositional approach shows marginal gain
Low
Medium
Marginal gain is still publishable as a null result; architecture novelty stands regardless
Fusion does not outperform best single modality
Low
High
Ablation analysis becomes the primary contribution: understanding WHY modalities are/aren't complementary is valuable either way
SJWP judges find ML too technical
Medium
High
Lead with the platform demo and global hotspot map; technical novelty in supplementary. The "immune system for waterways" narrative is accessible.
Data download bandwidth/time
Medium
Medium
Prioritize GRQA (pre-harmonized, 17M records) + USGS NWIS (API, well-structured) + GEE satellite extraction. Full EPA WQP download is parallel-safe.


15. References & Key Datasets

Zhi, W. et al. (2024). Deep learning for water quality. Nature Water, 2, 228–241.
Kidger, P. et al. (2020). Neural Controlled Differential Equations for Irregular Time Series. NeurIPS.
Gu, A. & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. COLM.
Jaegle, A. et al. (2021). Perceiver IO: A General Architecture for Structured Inputs & Outputs. ICML.
Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. Science Advances.
Gordon-Rodriguez, E. et al. (2022). Data Augmentation for Compositional Data. NeurIPS.
Jakubik, J. et al. (2024). Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation. arXiv.
Ansari, A. et al. (2025). Chronos-2: Learning the Language of Time Series. arXiv.
Zhou, Z. et al. (2024). DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models. ISMB.
Gibbs, I. & Candès, E. (2021). Adaptive Conformal Inference Under Distribution Shift. NeurIPS.
Tan, Z. & Bai, S. (2026). Mamba for Water Quality Prediction. Water Resources Management.
Loreaux, E. et al. (2025). HydroGEM: Foundation Model for Water Quality. AGU Water Resources Research.
Pereira, T. et al. (2022). SLEAP: A deep learning system for multi-animal pose tracking. Nature Methods.
Elonen, C. et al. (2024). EPA National Aquatic Resource Surveys: Microbial Indicators. Ecological Indicators.
Ghattas, A. et al. (2024). SIT-FUSE: Harmful Algal Bloom Monitoring with Foundation Models. AGU.
Guo, X. et al. (2024). Diffusion Pre-Training for Fish Trajectory Recognition. Aquatic Sciences.

Key Data Access URLs
EPA Water Quality Portal: https://www.waterqualitydata.us/
USGS NWIS: https://waterdata.usgs.gov/nwis
EU Waterbase: https://www.eea.europa.eu/data-and-maps/data/waterbase-water-quality
GEMS/Water: https://gemstat.org/
GRQA: https://doi.org/10.5281/zenodo.7056647
Copernicus Marine: https://marine.copernicus.eu/
Google Earth Engine: https://earthengine.google.com/
Earth Microbiome Project: https://earthmicrobiome.org/
EPA ECOTOX: https://cfpub.epa.gov/ecotox/
NCBI GEO: https://www.ncbi.nlm.nih.gov/geo/
FreshWater Watch: https://freshwaterwatch.thewaterhub.org/
HydroLAKES: https://www.hydrosheds.org/products/hydrolakes
