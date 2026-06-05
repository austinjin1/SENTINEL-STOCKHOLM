# WaterDroneNet (HydroDenseNet) — SENTINEL-Lite Model

Image-only water quality prediction from drone multispectral imagery. This is the ML model component of the SENTINEL Mini system.

## What It Does

Takes a 4-channel image (RGB + NIR, 224x224) and predicts water quality parameters with calibrated uncertainty:

| Target | Test R² | Test MAE | Unit |
|--------|---------|----------|------|
| Temperature | 0.776 | 2.56 | °C |
| Dissolved Oxygen | 0.463 | 1.25 | mg/L |
| Specific Conductance | 0.442 | 1955.6 | µS/cm |
| Turbidity | 0.181 | 10.96 | NTU |

Trained on 57K real Sentinel-2 tiles paired with USGS NWIS measurements from 399 stations. Evaluated on spatially held-out stations (never-seen geography).

## Architecture

```
4ch image (224x224)
    │
    ├── SpectralStem (learnable band interactions)
    │       │
    │   DenseNet121 body
    │       │
    │   Multi-Scale FPN (all 4 DenseBlock outputs)
    │       │
    │   CBAM attention (channel + spatial) at each scale
    │       │
    │   Multi-pool aggregation (avg + max + std)
    │
    ├── Band Ratio Encoder (NDWI, NIR/Red, etc. → 48-dim)
    │
    └── Per-Target Expert MLPs → (μ, σ) per target
            │
        Trust Router (flags low-confidence predictions)
```

- **Parameters**: 8.4M
- **Input**: 4 bands — Blue (B02), Green (B03), Red (B04), NIR (B08)
- **Output**: Per-target Gaussian (mean + sigma) with trust flags

## Camera Payload

Designed for a dual-camera drone rig:
- **RGB**: Raspberry Pi Camera Module 3 Wide
- **NIR**: Raspberry Pi NoIR Camera Module V2 (8MP, 1080p30)

The two cameras capture simultaneous frames. RGB provides B02/B03/B04; the NoIR camera (with a blue filter removed) captures NIR equivalent to Sentinel-2 B08.

## Quick Inference

```python
import torch
from sentinel.models.waterdronenet.waterdronenet import HydroDenseNet

model = HydroDenseNet(targets=["DO", "Turb", "Temp", "SpCond"])
ckpt = torch.load("checkpoints/waterdronenet/waterdronenet_v4_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# img: [B, 4, 224, 224] tensor, float32, values in [0, 1]
with torch.no_grad():
    output = model(img)
    # output.predictions: dict of target -> (mu, sigma)
    # output.trust_flags: dict of target -> bool
```

## Files

| File | Description |
|------|-------------|
| `waterdronenet.py` | HydroDenseNet model definition |
| `ph_strip_reader.py` | pH test strip color reader (separate accessory) |
| `__init__.py` | Module exports |

## Checkpoints

All in `checkpoints/waterdronenet/`:
- `waterdronenet_v4_best.pt` — current best (HydroDenseNet architecture)
- Earlier versions (v1-v3) use the older ViT-S architecture

## Training

```bash
python scripts/train_waterdronenet_v4.py --epochs 100 --gpu 0
```

Data: `data/sentinel2_tiles/` (individual .npz files per tile, lazy-loaded).
