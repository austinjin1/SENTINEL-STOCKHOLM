"""Track B shared core — encoders, date-aligned sensor windows, batched fusion.

Honest co-located fusion: a real Sentinel-2 tile (WaterDroneNet) and a real
date-aligned in-situ sensor window (AquaSSM) at the SAME USGS station/time,
classifying a NON-CIRCULAR label (held-out chlorophyll-a, or a future DO
exceedance). The fusion here is a batched, stateless re-wiring of SENTINEL's
own PerceiverIO submodules (the shipped PerceiverIOFusion keeps only sample-0
in a stateful registry, so it cannot train on batches).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel.models.waterdronenet.waterdronenet import WaterDroneNet           # noqa: E402
from sentinel.models.sensor_encoder.model import SensorEncoder                  # noqa: E402
from sentinel.models.fusion.projections import NATIVE_DIMS, ProjectionBank      # noqa: E402
from sentinel.models.fusion.latent_array import LatentArray                     # noqa: E402
from sentinel.models.fusion.perceiver_attention import PerceiverCrossAttention  # noqa: E402
from sentinel.models.fusion.embedding_registry import MODALITY_IDS, NUM_MODALITIES  # noqa: E402

SAT_CKPT = ROOT / "checkpoints" / "waterdronenet" / "waterdronenet_best.pt"
SEN_CKPT = ROOT / "checkpoints" / "sensor" / "aquassm_real_pretrained.pt"
RAW_SENSOR = ROOT / "data" / "raw" / "sensor" / "full"

WIN = 512
# AquaSSM channel order: [pH, DO, turbidity, conductivity, temperature, ORP].
# Raw-parquet column -> channel index.
COL2CH = {"pH": 0, "DO": 1, "Turb": 2, "SpCond": 3, "Temp": 4}
# Global robust per-channel (center, scale); filled from drone_wq_pairs targets.
SENSOR_STATS = None


def _load_flat(model, ckpt, name):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # ignore benign no_data_tokens buffers when judging load health (the intended
    # heuristic); gate on the CRITICAL missing keys, not the raw list.
    crit = [m for m in missing if not m.endswith("no_data_tokens")]
    if len(crit) > 0.5 * len(list(model.state_dict())):
        raise RuntimeError(f"{name}: too many missing keys ({len(crit)} critical of {len(missing)})")
    return model


def load_satellite_encoder(device="cuda"):
    m = WaterDroneNet()
    _load_flat(m, SAT_CKPT, "WaterDroneNet")
    return m.eval().to(device)


def load_sensor_encoder(device="cuda"):
    m = SensorEncoder(num_params=6, output_dim=256)
    _load_flat(m, SEN_CKPT, "SensorEncoder")
    return m.eval().to(device)


def set_sensor_stats_from_targets(targets: np.ndarray):
    """targets columns = [DO, pH, Turb, Temp, SpCond]; map to channel stats."""
    global SENSOR_STATS
    tmap = {"DO": 0, "pH": 1, "Turb": 2, "Temp": 3, "SpCond": 4}
    stats = np.zeros((6, 2), "float64")
    stats[:, 1] = 1.0
    for col, ch in COL2CH.items():
        v = targets[:, tmap[col]]
        v = v[np.isfinite(v)]
        if v.size > 16:
            med = np.median(v)
            iqr = np.subtract(*np.percentile(v, [75, 25]))
            stats[ch] = [med, (iqr / 1.349) if iqr > 1e-6 else (np.std(v) or 1.0)]
    SENSOR_STATS = stats
    return stats


_PARQUET_CACHE: dict[str, pd.DataFrame] = {}


def _site_df(site: str):
    if site not in _PARQUET_CACHE:
        f = RAW_SENSOR / f"{site}.parquet"
        _PARQUET_CACHE[site] = pd.read_parquet(f) if f.exists() else None
    return _PARQUET_CACHE[site]


def build_sensor_window(site: str, date: str):
    """Most-recent <=WIN observations up to `date` -> ([WIN,6] z, delta_ts[WIN]).

    Returns (x, dt, n_real) or None when too little history exists.
    """
    df = _site_df(site)
    if df is None or len(df) == 0:
        return None
    ts = pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    sub = df[df.index <= ts]
    if len(sub) < 32:
        return None
    sub = sub.iloc[-WIN:]
    n = len(sub)
    x = np.zeros((WIN, 6), "float32")          # left-pad with zeros (= baseline mean post z)
    dt = np.full(WIN, 900.0, "float32")
    off = WIN - n
    idx_s = (sub.index.asi8 // 1_000_000_000).astype("float64")
    d = np.diff(idx_s, prepend=idx_s[0] - 900.0)
    d = np.clip(d, 60.0, 30 * 86400.0)
    dt[off:] = d.astype("float32")
    for col, ch in COL2CH.items():
        if col in sub.columns:
            v = pd.to_numeric(sub[col], errors="coerce").to_numpy("float64")
            c, s = SENSOR_STATS[ch]
            z = (v - c) / s
            z = np.where(np.isfinite(z), z, 0.0)
            x[off:, ch] = z.astype("float32")
    return x, dt, n


@torch.no_grad()
def embed_satellite(model, images: np.ndarray, device="cuda", batch=64):
    out = []
    for i in range(0, len(images), batch):
        xb = torch.from_numpy(images[i:i + batch]).float().to(device)
        out.append(model(xb)["cls_feat"].cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def embed_sensor(model, X: np.ndarray, DT: np.ndarray, device="cuda", batch=64):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        dt = torch.from_numpy(DT[i:i + batch]).to(device)
        r = model(xb, delta_ts=dt, compute_anomaly=False)
        out.append(r["embedding"].cpu().numpy())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# Batched, stateless fusion classifier (same submodules as PerceiverIOFusion)
# ---------------------------------------------------------------------------

class FusionClassifier(nn.Module):
    """ProjectionBank -> PerceiverIO cross-attention -> BCE head.

    Calling with only one modality present (the other left as a learned
    no-data token) gives the sensor-only / satellite-only ablations within the
    identical architecture.
    """

    def __init__(self, num_latents: int = 64, num_heads: int = 8,
                 num_process_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.bank = ProjectionBank(NATIVE_DIMS, 256)
        self.latent_array = LatentArray(num_latents=num_latents, latent_dim=256)
        self.perceiver = PerceiverCrossAttention(256, num_heads, num_process_layers, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def fuse(self, sat=None, sensor=None):
        ref = sat if sat is not None else sensor
        B = ref.shape[0]
        device = ref.device
        modemb = {mid: None for mid in MODALITY_IDS}
        if sat is not None:
            modemb["satellite"] = self.bank("satellite", sat)
        if sensor is not None:
            modemb["sensor"] = self.bank("sensor", sensor)
        latents = self.latent_array.get_latents(B).to(device)
        tbias = torch.zeros(B, NUM_MODALITIES, device=device)
        _, fused, _ = self.perceiver(latents, modemb, tbias)
        return fused

    def forward(self, sat=None, sensor=None):
        return self.head(self.fuse(sat, sensor)).squeeze(-1)


N_SPECTRAL = 32


def spectral_feats(img: np.ndarray) -> list:
    """Chl-sensitive spectral panel from a tile [4,H,W] = B02,B03,B04,B08."""
    b, g, r, nir = img[0], img[1], img[2], img[3]
    eps = 1e-6
    f = []
    for ch in (b, g, r, nir):
        f += [float(ch.mean()), float(ch.std()),
              float(np.median(ch)), float(np.percentile(ch, 90))]
    bm, gm, rm, nm = b.mean(), g.mean(), r.mean(), nir.mean()
    f += [gm / (bm + eps), gm / (rm + eps), rm / (gm + eps), nm / (rm + eps),
          nm / (gm + eps), bm / (gm + eps), (gm - rm) / (gm + rm + eps),
          (bm - gm) / (bm + gm + eps)]
    for idx in ((nir - r) / (nir + r + eps), (g - nir) / (g + nir + eps),
                g / (b + eps), (g - r) / (g + r + eps)):
        f += [float(idx.mean()), float(idx.std())]
    return f


class _Adapter(nn.Module):
    """Nonlinear per-modality adapter (provides the nonlinearity GBM exploited)."""

    def __init__(self, in_dim: int, out: int = 256, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(p),
            nn.LayerNorm(256), nn.Linear(256, out), nn.LayerNorm(out))

    def forward(self, x):
        return self.net(x)


class FusionClassifierV2(nn.Module):
    """Perceiver fusion with NONLINEAR adapters + spectral-augmented satellite."""

    def __init__(self, sat_dim: int, sen_dim: int = 256, num_latents: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.sat_ad = _Adapter(sat_dim, 256, dropout)
        self.sen_ad = _Adapter(sen_dim, 256, dropout)
        self.latent_array = LatentArray(num_latents=num_latents, latent_dim=256)
        self.perceiver = PerceiverCrossAttention(256, 8, 2, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def fuse(self, sat=None, sensor=None):
        ref = sat if sat is not None else sensor
        B, device = ref.shape[0], ref.device
        modemb = {mid: None for mid in MODALITY_IDS}
        if sat is not None:
            modemb["satellite"] = self.sat_ad(sat)
        if sensor is not None:
            modemb["sensor"] = self.sen_ad(sensor)
        lat = self.latent_array.get_latents(B).to(device)
        tbias = torch.zeros(B, NUM_MODALITIES, device=device)
        _, fused, _ = self.perceiver(lat, modemb, tbias)
        return fused

    def forward(self, sat=None, sensor=None):
        return self.head(self.fuse(sat, sensor)).squeeze(-1)


class ConcatFusion(nn.Module):
    """Simple early-fusion MLP control (does the Perceiver earn its complexity?)."""

    def __init__(self, sat_dim: int, sen_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.sat_ad = _Adapter(sat_dim, 128, dropout)
        self.sen_ad = _Adapter(sen_dim, 128, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, sat, sensor):
        return self.head(torch.cat([self.sat_ad(sat), self.sen_ad(sensor)], -1)).squeeze(-1)


class LateEnsembleHead(nn.Module):
    """Two independent unimodal MLP heads; probabilities averaged at eval.

    The honest control for 'does the Perceiver beat trivial averaging'.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.sat = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 128), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(128, 1))
        self.sen = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, sat, sensor):
        return self.sat(sat).squeeze(-1), self.sen(sensor).squeeze(-1)
