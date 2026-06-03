"""pH Strip Reader — extract pH from camera images of colorimetric test strips.

Operational flow:
  1. Drone dips pH strip into water body
  2. Drone camera photographs strip against reference background
  3. This module extracts dominant strip color → maps to pH value
  4. Result fed into SentinelMiniTriggerSystem alongside WaterDroneNet predictions

Supports two modes:
  - Calibration-based: HSV hue lookup against known strip color curves (no training)
  - Learned: Tiny CNN trained on synthetic/real strip images (more robust to lighting)

Standard universal indicator pH color progression:
  pH 1-2: Red         (H ≈ 0°)
  pH 3:   Orange-Red  (H ≈ 15°)
  pH 4:   Orange      (H ≈ 25°)
  pH 5:   Yellow      (H ≈ 45°)
  pH 6:   Yellow-Green(H ≈ 65°)
  pH 7:   Green       (H ≈ 85°)
  pH 8:   Blue-Green  (H ≈ 110°)
  pH 9:   Blue        (H ≈ 140°)
  pH 10:  Indigo      (H ≈ 170°)
  pH 11+: Purple      (H ≈ 200°+)

For freshwater monitoring (pH 5-9), the green-blue range gives ~10°/pH unit
resolution — easily distinguishable even with cheap cameras.

MIT License — Bryan Cheng, 2026
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# Known universal indicator pH → HSV hue mapping (OpenCV scale: 0-180)
# Based on standard colorimetric chemistry
PH_COLOR_TABLE = np.array([
    # [pH, hue_cv, saturation, value]  — hue in OpenCV 0-180 scale
    [1.0,    0,  255, 200],   # deep red
    [2.0,    5,  255, 210],   # red
    [3.0,   10,  240, 220],   # orange-red
    [4.0,   15,  230, 230],   # orange
    [5.0,   25,  220, 240],   # yellow-orange
    [6.0,   35,  200, 230],   # yellow-green
    [7.0,   50,  180, 200],   # green
    [8.0,   70,  190, 180],   # blue-green
    [9.0,   95,  200, 170],   # blue
    [10.0, 115,  210, 160],   # deep blue
    [11.0, 130,  190, 150],   # indigo
    [12.0, 145,  170, 140],   # violet
    [13.0, 155,  150, 130],   # purple
    [14.0, 160,  130, 120],   # dark purple
], dtype=np.float32)

PH_VALUES = PH_COLOR_TABLE[:, 0]
HUE_VALUES = PH_COLOR_TABLE[:, 1]


def hue_to_ph(hue_cv: float) -> float:
    """Convert OpenCV hue (0-180) to pH using interpolation.

    Args:
        hue_cv: Hue value in OpenCV scale (0-180).

    Returns:
        Estimated pH value.
    """
    return float(np.interp(hue_cv, HUE_VALUES, PH_VALUES))


def ph_to_hue(ph: float) -> float:
    """Convert pH to expected OpenCV hue."""
    return float(np.interp(ph, PH_VALUES, HUE_VALUES))


def ph_to_rgb(ph: float) -> tuple:
    """Convert pH to approximate RGB color for visualization."""
    hue_cv = ph_to_hue(ph)
    # OpenCV HSV to RGB approximation
    h = hue_cv / 180.0 * 360.0  # convert to 0-360
    s = 0.8
    v = 0.85

    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


class PHStripReader:
    """Read pH from a camera image of a colorimetric pH strip.

    Works by extracting the dominant color in the strip region,
    converting to HSV, and mapping hue to pH via calibration curve.
    """

    def __init__(self, strip_brand: str = "universal"):
        """
        Args:
            strip_brand: Which calibration curve to use.
                "universal" — standard universal indicator (pH 1-14)
        """
        self.brand = strip_brand
        self.ph_values = PH_VALUES.copy()
        self.hue_values = HUE_VALUES.copy()

    def read_from_rgb(self, image: np.ndarray,
                      roi: Optional[tuple] = None) -> dict:
        """Extract pH from an RGB image of a pH strip.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), uint8.
            roi: Optional (x1, y1, x2, y2) bounding box for strip region.
                 If None, uses center 50% of image.

        Returns:
            dict with keys:
                ph: Estimated pH value
                confidence: 0-1 confidence based on saturation
                hue: Extracted hue value
                dominant_rgb: Dominant color as (R, G, B)
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            crop = image[y1:y2, x1:x2]
        else:
            h, w = image.shape[:2]
            crop = image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

        # Convert RGB to HSV manually (no OpenCV dependency)
        rgb_float = crop.astype(np.float32) / 255.0
        r, g, b = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue computation
        hue = np.zeros_like(delta)
        mask_r = (cmax == r) & (delta > 0.01)
        mask_g = (cmax == g) & (delta > 0.01)
        mask_b = (cmax == b) & (delta > 0.01)

        hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
        hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

        # Saturation
        sat = np.where(cmax > 0.01, delta / cmax, 0)

        # Filter for saturated pixels (the strip, not the background)
        sat_mask = sat > 0.15
        if sat_mask.sum() < 10:
            # Very low saturation — strip might be grey/neutral
            sat_mask = np.ones_like(sat, dtype=bool)

        # Median hue of saturated pixels (robust to noise)
        hue_filtered = hue[sat_mask]
        sat_filtered = sat[sat_mask]

        if len(hue_filtered) == 0:
            return {"ph": 7.0, "confidence": 0.0, "hue": 0.0,
                    "dominant_rgb": (128, 128, 128)}

        # Handle hue wraparound (red can be near 0° or 360°)
        # Use circular mean
        hue_rad = np.radians(hue_filtered)
        mean_sin = np.mean(np.sin(hue_rad))
        mean_cos = np.mean(np.cos(hue_rad))
        mean_hue_deg = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

        # Convert to OpenCV scale (0-180)
        hue_cv = mean_hue_deg / 2.0

        # Map to pH
        ph = hue_to_ph(hue_cv)

        # Confidence based on saturation (higher saturation = clearer reading)
        mean_sat = float(np.mean(sat_filtered))
        confidence = min(1.0, mean_sat / 0.5)  # saturate at 50%+ saturation

        # Dominant RGB
        dominant_r = int(np.median(crop[sat_mask, 0]))
        dominant_g = int(np.median(crop[sat_mask, 1]))
        dominant_b = int(np.median(crop[sat_mask, 2]))

        return {
            "ph": round(float(ph), 2),
            "confidence": round(float(confidence), 3),
            "hue_cv": round(float(hue_cv), 1),
            "hue_deg": round(float(mean_hue_deg), 1),
            "dominant_rgb": (dominant_r, dominant_g, dominant_b),
            "saturation": round(float(mean_sat), 3),
            "n_pixels": int(sat_mask.sum()),
        }


class PHStripCNN(nn.Module):
    """Tiny CNN for pH strip reading — more robust to lighting than HSV lookup.

    Input: 64x64 RGB crop of the pH strip
    Output: pH value (scalar regression)

    Only 12K parameters — runs on any device.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),  # 64→32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32→16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # mean + log_sigma
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 64, 64) RGB strip images, normalized to [0, 1]
        Returns:
            dict with "ph" (B,) and "uncertainty" (B,)
        """
        feat = self.features(x).flatten(1)
        out = self.head(feat)
        ph = out[:, 0] * 7.0 + 7.0  # scale to pH 0-14 range
        uncertainty = torch.exp(out[:, 1]).clamp(0.01, 3.0)
        return {"ph": ph, "uncertainty": uncertainty}


def generate_synthetic_strip(ph: float, size: int = 64,
                              noise_std: float = 15.0,
                              lighting_factor: float = 1.0) -> np.ndarray:
    """Generate a synthetic pH strip image for training.

    Args:
        ph: Target pH value (1-14)
        size: Image size (square)
        noise_std: Pixel noise standard deviation
        lighting_factor: Brightness multiplier (0.5-1.5 for augmentation)

    Returns:
        RGB image as (size, size, 3) uint8 array
    """
    rgb = ph_to_rgb(ph)
    img = np.full((size, size, 3), rgb, dtype=np.float32)

    # Add realistic noise
    img += np.random.randn(size, size, 3) * noise_std

    # Lighting variation
    img *= lighting_factor

    # Color cast (simulates different lighting conditions)
    cast = np.random.randn(3) * 10
    img += cast

    # Slight gradient (uneven lighting)
    grad = np.linspace(-10, 10, size).reshape(-1, 1, 1)
    img += grad

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_training_set(n_samples: int = 5000,
                           size: int = 64) -> tuple:
    """Generate synthetic pH strip dataset.

    Returns:
        images: (N, 3, size, size) float32 tensor, normalized [0,1]
        labels: (N,) float32 tensor of pH values
    """
    images = []
    labels = []

    for _ in range(n_samples):
        ph = np.random.uniform(1.0, 14.0)
        noise = np.random.uniform(5, 30)
        lighting = np.random.uniform(0.5, 1.5)

        img = generate_synthetic_strip(ph, size, noise, lighting)

        # To CHW float
        img_t = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        images.append(img_t)
        labels.append(ph)

    return (torch.tensor(np.array(images)),
            torch.tensor(np.array(labels, dtype=np.float32)))


def train_strip_reader(n_epochs: int = 50, n_train: int = 10000,
                        n_val: int = 1000, lr: float = 1e-3,
                        device: str = "cpu") -> PHStripCNN:
    """Train the tiny CNN strip reader on synthetic data.

    Returns trained model. Takes ~10 seconds on CPU.
    """
    model = PHStripCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_imgs, train_labels = generate_training_set(n_train)
    val_imgs, val_labels = generate_training_set(n_val)

    train_imgs = train_imgs.to(device)
    train_labels = train_labels.to(device)
    val_imgs = val_imgs.to(device)
    val_labels = val_labels.to(device)

    best_val_mae = float("inf")
    best_state = None

    batch_size = 256

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)

        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            out = model(train_imgs[idx])

            # Gaussian NLL
            diff = out["ph"] - train_labels[idx]
            loss = (diff ** 2 / (2 * out["uncertainty"] ** 2)
                    + torch.log(out["uncertainty"])).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_imgs)
            val_mae = (val_out["ph"] - val_labels).abs().mean().item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: "
                  f"loss={epoch_loss / n_batches:.4f}, val_mae={val_mae:.3f}")

    model.load_state_dict(best_state)
    print(f"  Best val MAE: {best_val_mae:.3f} pH units")

    return model
