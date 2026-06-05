# SENTINEL Mini — Drone-to-Analysis Pipeline

End-to-end pipeline from drone image capture to water quality prediction and station activation.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DRONE (Raspberry Pi)                     │
│                                                              │
│  RGB Camera ──┐                                              │
│               ├── capture_sync() ──► 4ch image (RGB+NIR)     │
│  NoIR Camera ─┘                          │                   │
│                                          │                   │
│              ROS2 publisher ◄────────────┘                   │
│              /sentinel/drone/image_raw                        │
│                     │                                        │
└─────────────────────┼────────────────────────────────────────┘
                      │  WiFi / USB tether
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               GROUND STATION (Windows PC)                    │
│                                                              │
│  ROS2 subscriber                                             │
│  /sentinel/drone/image_raw                                   │
│         │                                                    │
│         ▼                                                    │
│  Local Image Cache                                           │
│  └── images/                                                 │
│      ├── 2026-06-03T14_30_00_lat41.50_lon-81.70.npz         │
│      ├── 2026-06-03T14_30_15_lat41.50_lon-81.71.npz         │
│      └── ...                                                 │
│         │                                                    │
│         ▼                                                    │
│  HydroDenseNet Inference                                     │
│  ├── Load waterdronenet_v4_best.pt                           │
│  ├── Predict: DO, Turb, Temp, SpCond (μ ± σ)                │
│  └── Trust router filters low-confidence                     │
│         │                                                    │
│         ▼                                                    │
│  Anomaly Scoring                                             │
│  ├── Compare predictions vs EPA/WHO thresholds               │
│  ├── Assign alert level (NOMINAL → CRITICAL)                 │
│  └── Log to results/                                         │
│         │                                                    │
│         ▼ (if alert level ≥ WARNING)                         │
│  Trigger System                                              │
│  ├── Find nearest K SENTINEL stations (within 15km LoRa)     │
│  ├── Send RF activation command                              │
│  └── Await confirmation from full multimodal SENTINEL        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Drone Image Capture (ROS2 — your friend's part)

The drone runs ROS2 on a Raspberry Pi 4/5 with the dual-camera payload:

- **RGB**: Raspberry Pi Camera Module 3 Wide
- **NIR**: Raspberry Pi NoIR Camera Module V2 (8MP, 1080p30)

The ROS2 node should:
1. Capture synchronized RGB + NIR frames
2. Stack into a 4-channel array: `[Blue, Green, Red, NIR]` — matching Sentinel-2 band order (B02, B03, B04, B08)
3. Resize to 224x224
4. Normalize to [0, 1] float32
5. Publish on `/sentinel/drone/image_raw` as a custom message (4ch image + GPS + timestamp)

**ROS2 message format** (suggested):

```
# SentinelImage.msg
std_msgs/Header header
float64 latitude
float64 longitude
float64 altitude_m
sensor_msgs/Image rgb_image    # 3ch, 224x224
sensor_msgs/Image nir_image    # 1ch, 224x224
string drone_id
string flight_id
```

### 2. Local Image Cache (Ground Station)

The ROS2 subscriber on the Windows machine receives images and writes them to a local cache:

```
sentinel_mini_cache/
├── images/
│   ├── {timestamp}_{lat}_{lon}.npz     # 4ch array + metadata
│   └── ...
├── predictions/
│   ├── {timestamp}_{lat}_{lon}.json    # model output + alert level
│   └── ...
└── alerts/
    └── {timestamp}_alert.json          # triggered alerts only
```

Each `.npz` file contains:
```python
{
    "image": np.ndarray,      # shape (4, 224, 224), float32, [0, 1]
    "latitude": float,
    "longitude": float,
    "altitude_m": float,
    "timestamp": str,         # ISO 8601
    "drone_id": str,
    "flight_id": str,
}
```

### 3. HydroDenseNet Inference

The model runs on the ground station GPU (or CPU). See `sentinel/models/waterdronenet/README.md` for model details.

**Input**: 4-channel image tensor `[1, 4, 224, 224]`
**Output**: Per-target predictions with uncertainty (μ, σ) and trust flags

### 4. Anomaly Scoring & Trigger

Uses thresholds defined in `sentinel/platform/sentinel_mini_trigger.py`:

| Parameter | Warning | Critical | Direction |
|-----------|---------|----------|-----------|
| DO | < 4.0 mg/L | < 2.0 mg/L | below |
| pH | < 6.5 or > 9.0 | < 5.0 or > 10.0 | both |
| Turbidity | > 50 NTU | > 200 NTU | above |
| Temperature | > 30°C | > 35°C | above |
| SpCond | > 2500 µS/cm | > 10000 µS/cm | above |

Alert levels: `NOMINAL` → `WATCH` → `WARNING` → `ALERT` → `CRITICAL`

If alert level ≥ WARNING, the trigger system:
1. Finds nearest K fixed SENTINEL stations within LoRa range (15km)
2. Sends RF activation command with priority and requested parameters
3. Activated station switches from SLEEP/STANDBY → ACTIVE/CONFIRM mode
4. Station runs full multimodal analysis and sends confirmation back

## Existing Code

| File | What It Does |
|------|-------------|
| `sentinel/models/waterdronenet/waterdronenet.py` | HydroDenseNet model (inference-ready) |
| `sentinel/platform/sentinel_mini_trigger.py` | Anomaly scoring, station selection, trigger commands |
| `scripts/test_sentinel_mini_pipeline.py` | End-to-end pipeline test (simulated) |
| `checkpoints/waterdronenet/waterdronenet_v4_best.pt` | Trained model weights |

## What Needs to Be Built (Your Friend's Part)

1. **ROS2 capture node** — synchronized dual-camera capture on RPi, publishes 4ch images
2. **ROS2 subscriber** — receives images on Windows, saves to local cache
3. **Inference runner** — watches cache directory, runs HydroDenseNet on new images, writes predictions
4. **Dashboard/viewer** — (optional) live map showing drone path + predictions + alerts

## Model Performance

Trained on real Sentinel-2 imagery paired with USGS measurements. Spatial holdout (geographically unseen test stations):

| Target | R² | MAE | Pearson r |
|--------|-----|-----|-----------|
| Temperature | 0.776 | 2.56°C | 0.882 |
| Dissolved Oxygen | 0.463 | 1.25 mg/L | 0.757 |
| Specific Conductance | 0.442 | 1955.6 µS/cm | 0.675 |
| Turbidity | 0.181 | 10.96 NTU | 0.435 |

**Important**: The model was trained on satellite imagery (10m resolution). Drone imagery at lower altitude will have different spatial characteristics. In-field calibration with known water quality measurements is recommended before operational use.

## Running Inference Standalone

Without ROS2, you can test the model on any 4-channel image:

```python
import numpy as np
import torch
from sentinel.models.waterdronenet.waterdronenet import HydroDenseNet

# Load model
model = HydroDenseNet(targets=["DO", "Turb", "Temp", "SpCond"])
ckpt = torch.load("checkpoints/waterdronenet/waterdronenet_v4_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load a cached image
data = np.load("sentinel_mini_cache/images/example.npz")
img = torch.from_numpy(data["image"]).unsqueeze(0)  # [1, 4, 224, 224]

with torch.no_grad():
    output = model(img)

for target in ["DO", "Turb", "Temp", "SpCond"]:
    mu, sigma = output.predictions[target]
    trust = output.trust_flags[target]
    print(f"{target}: {mu.item():.2f} ± {sigma.item():.2f}  (trusted: {trust.item()})")
```
