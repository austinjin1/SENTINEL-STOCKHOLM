# SENTINEL: Scalable Environmental Network for Temporal Intelligence and Ecological Learning

**Stockholm Junior Water Prize Submission**

SENTINEL is a computational framework for planetary-scale water quality intelligence. It fuses four data modalities — satellite remote sensing, physicochemical sensor networks, microbial community profiles, and molecular stress biomarker data — through a novel asynchronous cross-modal temporal attention architecture.

## Key Innovations

1. **Early Anomaly Detection** — Quantified temporal lead over single-modality baselines via multimodal fusion
2. **Automated Source Attribution** — Contaminant class identification from multimodal evidence
3. **Digital Biosentinel** — Computational model trained on ~1M ecotoxicology records that predicts organism responses without requiring live organism deployment
4. **Cascade Escalation** — RL-based policy that optimizes detection sensitivity against computational cost

## Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Satellite   │  │   Sensor    │  │  Microbial   │  │  Molecular  │
│  Encoder     │  │   Encoder   │  │  Encoder     │  │  Encoder    │
│ (ViT-S/16)  │  │   (TCN)     │  │(Transformer) │  │ (Chem2Path) │
└──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘
       │                 │                │                  │
       └────────┬────────┴────────┬───────┴──────────────────┘
                │                 │
    ┌───────────▼─────────────────▼──────────┐
    │  Cross-Modal Temporal Attention Fusion  │
    │  (Learned decay, async updates, GRU)   │
    └───────────┬─────────────────┬──────────┘
                │                 │
    ┌───────────▼───┐   ┌────────▼────────┐   ┌──────────────┐
    │   Anomaly     │   │    Source       │   │   Digital    │
    │  Detection    │   │  Attribution   │   │ Biosentinel  │
    └───────────────┘   └────────────────┘   └──────────────┘
                │
    ┌───────────▼───────────┐
    │  Cascade Escalation   │
    │  Controller (RL/PPO)  │
    └───────────────────────┘
```

## Data Sources (All Freely Available)

| Modality | Source | Access |
|----------|--------|--------|
| Satellite | Sentinel-2 L2A, Landsat 8/9 TIRS | Google Earth Engine / Planetary Computer |
| Sensor | USGS NWIS (~3,000 stations) | `dataretrieval` Python package |
| Microbial | EPA NARS, Earth Microbiome Project | EPA website, Qiita |
| Molecular | GEO, CTD, ArrayExpress | NCBI, CTDbase |
| Ecotoxicology | EPA ECOTOX (~1M records) | EPA bulk download |

## Project Structure

```
sentinel/
├── data/                    # Data acquisition & preprocessing
│   ├── satellite/           # Sentinel-2, Landsat download & tiling
│   ├── sensor/              # USGS NWIS sensor time series
│   ├── microbial/           # 16S rRNA community data
│   ├── molecular/           # Toxicogenomics expression data
│   ├── ecotox/              # EPA ECOTOX dose-response data
│   ├── alignment/           # Geographic co-location linking
│   └── case_studies/        # Historical event data collection
├── models/                  # Neural network architectures
│   ├── satellite_encoder/   # ViT-S + UPerNet + temporal
│   ├── sensor_encoder/      # TCN + MPP + anomaly scoring
│   ├── microbial_encoder/   # Source attribution + VAE
│   ├── molecular_encoder/   # Chem2Path + info bottleneck
│   ├── digital_biosentinel/ # Dose-response prediction
│   ├── fusion/              # Cross-modal temporal attention
│   └── escalation/          # RL cascade controller
├── training/                # Training scripts (staged)
├── evaluation/              # Metrics, ablation, figures
└── utils/                   # Config, logging, common utilities
dashboard/                   # React + TypeScript interactive demo
configs/                     # YAML configuration files
scripts/                     # Standalone data acquisition scripts
```

## Quick Start

```bash
# Create environment
conda env create -f environment.yml
conda activate sentinel

# Install package
pip install -e .

# Download data (requires API keys for GEE and USGS)
python scripts/data_acquisition/download_all.py

# Train encoders (Phase 3)
python -m sentinel.training.train_satellite --config configs/default.yaml
python -m sentinel.training.train_sensor --config configs/default.yaml
python -m sentinel.training.train_microbial --config configs/default.yaml
python -m sentinel.training.train_molecular --config configs/default.yaml
python -m sentinel.training.train_biosentinel --config configs/default.yaml

# Train fusion (Phase 4)
python -m sentinel.training.train_fusion --config configs/default.yaml

# Train escalation controller
python -m sentinel.training.train_escalation --config configs/default.yaml

# Run evaluation
python -m sentinel.evaluation.case_study --config configs/default.yaml
python -m sentinel.evaluation.ablation --config configs/default.yaml
python -m sentinel.evaluation.figures --config configs/default.yaml

# Launch dashboard
cd dashboard && npm install && npm start
```

## Validation

SENTINEL is validated on 10-15 documented historical contamination events spanning diverse contaminant classes and watershed types, demonstrating:
- Earlier detection via multimodal fusion
- More accurate source identification
- Lower false positive rates than any individual modality

## License

MIT

## Citation

If you use SENTINEL in your research, please cite our Stockholm Junior Water Prize submission.
