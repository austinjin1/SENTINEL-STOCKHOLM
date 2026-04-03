# SENTINEL Implementation Checklist

## Phase 0: Environment Setup
- [x] Create project directory structure
- [x] Write pyproject.toml
- [x] Write environment.yml (conda)
- [x] Write .gitignore
- [x] Write default.yaml config
- [x] Write README.md
- [x] Create all __init__.py files

## Phase 1: Data Acquisition Pipeline (Weeks 1-3)
- [ ] Satellite: Google Earth Engine Sentinel-2 L2A downloader
- [ ] Satellite: Microsoft Planetary Computer STAC alternative
- [ ] Satellite: Landsat 8/9 TIRS thermal download
- [ ] Satellite: Spectral index computation (NDCI, FAI, NDTI, MNDWI, Oil Index)
- [ ] Satellite: Tile preprocessing (512→224, band stacking, rolling buffer)
- [ ] Sensor: USGS NWIS station discovery (~3,000 qualifying stations)
- [ ] Sensor: Bulk sensor data download (batched by state)
- [ ] Sensor: Preprocessing (z-score normalization, gap filling, quality filtering)
- [ ] Microbial: EPA NARS data download
- [ ] Microbial: Earth Microbiome Project download (Qiita/Redbiom)
- [ ] Microbial: NCBI SRA targeted study search
- [ ] Microbial: Bioinformatics pipeline (DADA2 → UCHIME → SILVA 138 → CLR)
- [ ] Molecular: GEO toxicogenomics download (GEOparse)
- [ ] Molecular: CTD bulk download
- [ ] Molecular: Harmonization (TPM/RMA, ComBat batch correction)
- [ ] Molecular: Stress-response gene panel curation (~200 genes, 7 pathways)
- [ ] ECOTOX: EPA ECOTOX bulk download
- [ ] ECOTOX: Endpoint standardization and filtering
- [ ] Geographic alignment: HUC-8 watershed linking via NLDI
- [ ] Geographic alignment: Cross-modality site matching
- [ ] Case studies: 10 historical event data collection
- [ ] Case studies: EPA emergency response cross-referencing

## Phase 2: Data Processing (Weeks 2-5)
- [ ] Sentinel-2 tile extraction and spectral index computation
- [ ] USGS sensor data quality filtering and normalization
- [ ] 16S bioinformatics pipeline (standardized ASV tables)
- [ ] CLR transformation on all microbial abundance data
- [ ] GEO transcriptomics harmonization (ComBat batch correction)
- [ ] Molecular pathway activation labeling
- [ ] ECOTOX endpoint standardization
- [ ] Geographic alignment index (all sources linked by HUC-8)
- [ ] Retrospective anomaly labels (USGS × EPA event records)
- [ ] Case study data packages (30 days before → 60 days after each event)

## Phase 3: Individual Encoder Training (Weeks 4-8)

### Satellite Encoder
- [ ] Load SSL4EO-S12 ViT-S pretrained weights
- [ ] Fine-tune UPerNet segmentation on Tick Tick Bloom + MARIDA
- [ ] Train temporal change detection module
- [ ] Validate segmentation accuracy

### Sensor Encoder
- [ ] Self-supervised pretraining via Masked Parameter Prediction (MPP)
- [ ] Supervised anomaly fine-tuning with event labels
- [ ] Sensor health classifier (drift/fouling/failure detection)
- [ ] Validate anomaly detection on held-out stations

### Microbial Encoder
- [ ] Train source attribution transformer on EPA NARS
- [ ] Train community trajectory VAE on reference condition sites
- [ ] Extract attention weights → indicator species rankings
- [ ] Validate against established indicator literature

### Molecular Encoder
- [ ] Train Chem2Path multi-task predictor on GEO data
- [ ] Run information bottleneck sweep → minimal biomarker panel
- [ ] Validate against CTD chemical-gene interaction annotations

### Digital Biosentinel
- [ ] Train multi-output dose-response predictor on ECOTOX
- [ ] Calibrate with temperature scaling
- [ ] Produce calibration plots (reliability diagrams)
- [ ] Integration test with sensor encoder outputs

### Baselines
- [ ] Run single-modality baselines on all validation events

## Phase 4: Fusion & Escalation (Weeks 7-10)
- [ ] Identify co-located sites with ≥3 modalities (~500-1,000 sites)
- [ ] Stage 1: Freeze encoders, train fusion layer + output heads
- [ ] Stage 2: End-to-end fine-tuning (unfreeze top encoder layers)
- [ ] Train RL escalation policy via PPO on replayed historical events
- [ ] Extract interpretable decision tree from trained RL policy
- [ ] Run full ablation study (11 conditions)

## Phase 5: Validation & Analysis (Weeks 9-12)
- [ ] Run SENTINEL on all case study events in simulated real-time
- [ ] Measure detection lead time, source attribution accuracy, false positive rate
- [ ] Generate case study timeline figures (hero figures)
- [ ] Validate indicator species against ecotoxicology literature
- [ ] Validate biomarker panel (gene count vs. accuracy curve)
- [ ] Compile all aggregate metrics (detection AUC, mean lead time, etc.)
- [ ] Validate Digital Biosentinel predictions against documented impacts
- [ ] Produce all 10 key figures

## Phase 6: Dashboard & Presentation (Weeks 10-14)
- [ ] Build React dashboard: Global map view
- [ ] Build site detail panel (sensor, satellite, microbial, pathway, biosentinel)
- [ ] Build alert report generator
- [ ] Pre-compute results for 3 primary case studies (static JSON)
- [ ] Build case study replay component (timeline scrubber)
- [ ] Write paper following outline in Section 9.1
- [ ] Prepare 10-minute presentation (narrative arc in Section 10.1)
- [ ] Prepare answers for 10 anticipated judge questions
- [ ] Practice presentation with live dashboard demo

## Phase 7: Submission
- [ ] Final paper edit and formatting
- [ ] Code repository documentation
- [ ] Submit to SJWP
