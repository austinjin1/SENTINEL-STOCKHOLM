"""Track A shared core — label-free reconstruction-residual detector.

The genuinely-real, label-free signal: feed a per-site-standardized sensor
window through the MPP-pretrained AquaSSM backbone, mask each parameter and
reconstruct it, and read the per-parameter reconstruction error. A hypoxia
event makes DO decouple from its usual multivariate/temporal relationship, so
its reconstruction error rises. No anomaly labels are used anywhere.

This module only loads the model and produces raw per-window reconstruction
errors. Calibration (per-site error normalization + FPR threshold), the
sustained-exceedance rule, and lead-time logic live in 02/03 so the detector
itself stays assumption-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel.models.sensor_encoder.model import SensorEncoder  # noqa: E402

# Model channel order (mpp.PARAMETER_NAMES). Real NEON params occupy 0..3;
# temperature(4) and ORP(5) are zero-padded (no NEON column for them here).
CHANNELS = ["pH", "DO", "turbidity", "conductivity", "temperature", "ORP"]
REAL_PARAMS = ["pH", "DO", "turbidity", "conductivity"]
REAL_IDX = [0, 1, 2, 3]
DO_IDX = 1
WIN = 128
DELTA_T = 900.0          # 15-min step (seconds)


def load_detector(ckpt: str | Path, device: str = "cuda"):
    """Load the MPP-pretrained AquaSSM into a SensorEncoder (no anomaly head)."""
    model = SensorEncoder(num_params=6, output_dim=256)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ssm_loaded = all(not k.startswith("ssm.") for k in missing)
    rec_loaded = all("reconstruction_head" not in k for k in missing)
    if not (ssm_loaded and rec_loaded):
        raise RuntimeError(
            f"Backbone/reconstruction head did NOT fully load from {ckpt}; "
            f"missing ssm/rec keys present in: {[m for m in missing if m.startswith('ssm.') or 'reconstruction_head' in m][:5]}")
    model.eval().to(device)
    return model


def robust_stats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Median center + IQR-derived scale over QC-clean finite values."""
    v = values[mask]
    v = v[np.isfinite(v)]
    if v.size < 16:
        return 0.0, 1.0
    med = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-6:
        scale = iqr / 1.349
    else:
        s = float(np.std(v))
        scale = s if s > 1e-6 else 1.0
    return med, scale


def zscore_channel(values: np.ndarray, mask: np.ndarray,
                   center: float, scale: float) -> np.ndarray:
    """Z-score a param vs frozen clean-baseline stats; gap-fill missing bins.

    Preserves absolute level: a DO baseline far below the clean median yields a
    strongly negative z, which is exactly the precursor we want to detect.
    Vectorized forward-then-back fill; remaining NaN -> 0 (baseline mean).
    """
    z = (values - center) / scale
    z = np.where(mask & np.isfinite(z), z, np.nan)
    z = _ffill(z)
    z = _ffill(z[::-1])[::-1]
    return np.where(np.isfinite(z), z, 0.0).astype("float32")


def _ffill(a: np.ndarray) -> np.ndarray:
    """O(N) vectorized forward-fill of NaNs."""
    valid = ~np.isnan(a)
    idx = np.where(valid, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[~valid.cumsum().astype(bool)] = np.nan   # leading run before first valid
    return out


def build_windows(series: dict, in_stats: dict, lo: int, hi: int, stride: int):
    """Build z-scored [N,128,6] windows over bins [lo,hi).

    Returns X (float32 [N,128,6]), end_ts (int64 [N]) = timestamp of the last
    bin in each window, do_cov (float [N]) = fraction of DO bins QC-observed.
    """
    ts = series["ts"]
    chans = {}
    for p in REAL_PARAMS:
        chans[p] = zscore_channel(series[p], series[f"mask_{p}"],
                                  in_stats[p][0], in_stats[p][1])
    do_mask = series["mask_DO"]

    xs, ets, cov = [], [], []
    hi = min(hi, ts.size)
    for end in range(lo + WIN, hi + 1, stride):
        sl = slice(end - WIN, end)
        w = np.zeros((WIN, 6), dtype="float32")
        for j, p in zip(REAL_IDX, REAL_PARAMS):
            w[:, j] = chans[p][sl]
        xs.append(w)
        ets.append(ts[end - 1])
        cov.append(float(do_mask[sl].mean()))
    if not xs:
        return (np.zeros((0, WIN, 6), "float32"),
                np.zeros((0,), "int64"), np.zeros((0,), "float32"))
    return np.stack(xs), np.asarray(ets, "int64"), np.asarray(cov, "float32")


@torch.no_grad()
def recon_errors(model, X: np.ndarray, device: str = "cuda",
                 batch: int = 64) -> np.ndarray:
    """Per-window per-param mean reconstruction error -> [N,6]."""
    out = []
    dt = torch.full((min(batch, len(X)) or 1, WIN), DELTA_T, device=device)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        d = dt[:xb.shape[0]]
        res = model.anomaly_detector.compute_reconstruction_errors(xb, delta_ts=d)
        out.append(res["mean_errors"].cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 6), "float32")
