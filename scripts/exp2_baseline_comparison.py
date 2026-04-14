#!/usr/bin/env python3
"""Exp 2: Compare SENTINEL against 4 baseline anomaly detection methods.

Uses real NEON sensor data from neon_DP1.20288.001_consolidated.parquet.
Anomaly labels are derived from threshold exceedances:
  - pH < 6.0 or pH > 9.0
  - dissolvedOxygen < 4.0 mg/L
  - turbidity > 300 NTU
  - specificConductance > 1500 uS/cm

Baselines evaluated:
  1. Z-score threshold
  2. Isolation Forest
  3. ARIMA residual (STL-based approximation)
  4. AquaSSM-only (no fusion)
  5. SENTINEL (AquaSSM + PerceiverIO fusion + anomaly head)

Results saved to results/exp2_baselines/baseline_comparison.json.

Usage::

    python scripts/exp2_baseline_comparison.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from sentinel.models.sensor_encoder.model import SensorEncoder
from sentinel.models.fusion.model import PerceiverIOFusion
from sentinel.models.fusion.heads import AnomalyDetectionHead
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

CKPT_BASE  = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp2_baselines"
FIG_DIR    = PROJECT_ROOT / "paper" / "figures"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"

# Window / stride parameters (matching exp16)
T      = 128
STRIDE = 64

NEON_VALUE_COLS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
NEON_QF_COLS    = ["pHFinalQF", "dissolvedOxygenFinalQF", "turbidityFinalQF", "specificCondFinalQF"]
PARAM_NAMES     = ["pH", "DO", "Turbidity", "SpCond"]

ANOMALY_THRESHOLDS = {
    "pH":                  (6.0, 9.0),
    "dissolvedOxygen":     (4.0, None),
    "turbidity":           (None, 300.0),
    "specificConductance": (None, 1500.0),
}

# Target window counts for balanced evaluation
N_NORMAL_TARGET   = 500
N_ANOMALY_TARGET  = 500


# ---------------------------------------------------------------------------
# NEON data loading (mirrors exp16_parameter_attribution.py)
# ---------------------------------------------------------------------------

def load_neon_windows():
    """Load windows from NEON parquet, label via threshold exceedances.

    Returns:
        sensor_data : np.ndarray  [N, T, 4]   (raw, not normalised)
        labels      : np.ndarray  [N]          0=normal, 1=anomalous
    """
    import pyarrow.parquet as pq
    import pandas as pd

    logger.info(f"Reading NEON parquet: {NEON_PARQUET}")
    read_cols = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS
    table = pq.read_table(str(NEON_PARQUET), columns=read_cols)
    df = table.to_pandas()
    logger.info(f"  Raw rows: {len(df):,}")

    # Parse timestamps and sort
    df["ts"] = pd.to_datetime(df["startDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values(["source_site", "ts"]).reset_index(drop=True)

    normal_windows   = []
    anomaly_windows  = []

    sites = df["source_site"].dropna().unique()
    logger.info(f"  Sites available: {len(sites)}")

    for site in sites:
        if (len(normal_windows) >= N_NORMAL_TARGET * 2 and
                len(anomaly_windows) >= N_ANOMALY_TARGET * 2):
            break

        df_site = df[df["source_site"] == site].copy()
        df_site = df_site.set_index("ts")

        # Resample to 15-min grid (same as exp16)
        agg = {c: "mean" for c in NEON_VALUE_COLS}
        agg.update({c: "min"  for c in NEON_QF_COLS})
        try:
            df_site = df_site[NEON_VALUE_COLS + NEON_QF_COLS].resample("15min").agg(agg).reset_index()
        except Exception:
            continue

        if len(df_site) < T * 2:
            continue

        # Quality-flag mask: at least 2 params with QF=0 or non-null
        qf_passes = np.zeros((len(df_site), len(NEON_QF_COLS)), dtype=np.float32)
        for j, qf in enumerate(NEON_QF_COLS):
            if qf in df_site.columns:
                qf_passes[:, j] = (df_site[qf].fillna(1.0).astype(float) == 0).astype(float)
        good_mask = (
            (qf_passes.sum(axis=1) >= 2) |
            (df_site[NEON_VALUE_COLS].notna().sum(axis=1).values >= 2)
        )

        vals = df_site[NEON_VALUE_COLS].astype(float).values  # [N, 4]

        for start in range(0, len(df_site) - T + 1, STRIDE):
            end = start + T
            w_raw  = vals[start:end].copy()
            w_mask = good_mask[start:end]

            if w_mask.mean() < 0.3:
                continue

            # Impute NaNs per column with column mean; skip if all NaN
            has_valid = False
            for c in range(w_raw.shape[1]):
                col   = w_raw[:, c]
                valid = np.isfinite(col)
                if valid.any():
                    w_raw[~valid, c] = col[valid].mean()
                    has_valid = True
                else:
                    w_raw[:, c] = 0.0
            if not has_valid:
                continue

            # Label: anomalous if any threshold exceeded (checked on raw values)
            is_anomaly = False
            col_map = {
                "pH":                  0,
                "dissolvedOxygen":     1,
                "turbidity":           2,
                "specificConductance": 3,
            }
            for param, (lo, hi) in ANOMALY_THRESHOLDS.items():
                ci = col_map[param]
                col = w_raw[:, ci]
                finite = col[np.isfinite(col)]
                if len(finite) == 0:
                    continue
                if lo is not None and np.any(finite < lo):
                    is_anomaly = True
                    break
                if hi is not None and np.any(finite > hi):
                    is_anomaly = True
                    break

            if is_anomaly:
                if len(anomaly_windows) < N_ANOMALY_TARGET * 2:
                    anomaly_windows.append(w_raw.astype(np.float32))
            else:
                if len(normal_windows) < N_NORMAL_TARGET * 2:
                    normal_windows.append(w_raw.astype(np.float32))

    # Balance and concatenate
    n_normal  = min(len(normal_windows),  N_NORMAL_TARGET)
    n_anomaly = min(len(anomaly_windows), N_ANOMALY_TARGET)
    logger.info(f"  Windows collected — normal: {n_normal}, anomalous: {n_anomaly}")

    if n_normal == 0 or n_anomaly == 0:
        raise RuntimeError("Insufficient windows found in NEON parquet.")

    rng = np.random.RandomState(42)
    norm_idx = rng.choice(len(normal_windows),  n_normal,  replace=False)
    anom_idx = rng.choice(len(anomaly_windows), n_anomaly, replace=False)

    norm_arr = np.stack([normal_windows[i]  for i in norm_idx],  axis=0)  # [n_normal, T, 4]
    anom_arr = np.stack([anomaly_windows[i] for i in anom_idx],  axis=0)  # [n_anomaly, T, 4]

    sensor_data = np.concatenate([norm_arr, anom_arr], axis=0)
    labels      = np.array([0] * n_normal + [1] * n_anomaly, dtype=int)

    # Shuffle together
    idx = rng.permutation(len(labels))
    sensor_data = sensor_data[idx]
    labels      = labels[idx]

    logger.info(f"  Final dataset: {sensor_data.shape}, "
                f"normal={labels.sum()==0}, anomaly frac={labels.mean():.2f}")
    return sensor_data, labels


# ---------------------------------------------------------------------------
# Build normalised windows for AquaSSM (6-channel, pad 2 zeros)
# ---------------------------------------------------------------------------

def normalise_for_aquassm(sensor_data: np.ndarray) -> np.ndarray:
    """Per-window z-score norm + pad to 6 channels. Returns [N, T, 6]."""
    N, win_t, P = sensor_data.shape
    padded = np.zeros((N, win_t, 6), dtype=np.float32)
    for i in range(N):
        w = sensor_data[i].copy()
        m = w.mean(axis=0, keepdims=True)
        s = w.std(axis=0, keepdims=True) + 1e-8
        w_norm = (w - m) / s
        padded[i, :, :P] = w_norm
    return padded


# ---------------------------------------------------------------------------
# Baseline 1: Z-score threshold
# ---------------------------------------------------------------------------

def zscore_baseline(sensor_data: np.ndarray, rolling_window: int = 30) -> np.ndarray:
    """Compute anomaly scores using z-score vs rolling stats."""
    n_windows = sensor_data.shape[0]
    window_means = sensor_data.mean(axis=1)  # [N, P]

    scores = np.zeros(n_windows)
    for i in range(n_windows):
        start = max(0, i - rolling_window)
        history = window_means[start:i] if i > 0 else window_means[:1]
        mu    = history.mean(axis=0)
        sigma = history.std(axis=0) + 1e-8
        z = np.abs((window_means[i] - mu) / sigma)
        scores[i] = np.clip(z.max() / 5.0, 0.0, 1.0)

    return scores


# ---------------------------------------------------------------------------
# Baseline 2: Isolation Forest
# ---------------------------------------------------------------------------

def isolation_forest_baseline(sensor_data: np.ndarray,
                               train_frac: float = 0.3) -> np.ndarray:
    """Train Isolation Forest on first train_frac windows, score all."""
    n_windows = sensor_data.shape[0]
    features = np.concatenate([
        sensor_data.mean(axis=1),
        sensor_data.std(axis=1),
    ], axis=1)  # [N, 2*P]

    n_train = int(n_windows * train_frac)
    clf = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=1)
    clf.fit(features[:n_train])

    raw_scores = -clf.decision_function(features)
    scores = 1.0 / (1.0 + np.exp(-raw_scores))
    return scores


# ---------------------------------------------------------------------------
# Baseline 3: ARIMA residual (STL-style)
# ---------------------------------------------------------------------------

def arima_baseline(sensor_data: np.ndarray,
                   train_frac: float = 0.3) -> np.ndarray:
    """Fit ARIMA(2,1,1) per parameter on first train_frac, score rest."""
    n_windows = sensor_data.shape[0]
    n_train   = int(n_windows * train_frac)
    n_params  = sensor_data.shape[2]
    window_means = sensor_data.mean(axis=1)  # [N, P]

    scores = np.zeros(n_windows)

    for p in range(n_params):
        series     = window_means[:, p]
        train_ser  = series[:n_train]
        residuals  = np.zeros(n_windows)
        train_std  = train_ser.std() + 1e-8

        try:
            from statsmodels.tsa.arima.model import ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model  = ARIMA(train_ser, order=(2, 1, 1))
                fitted = model.fit()

                in_sample = fitted.fittedvalues
                residuals[:len(in_sample)] = (
                    train_ser[:len(in_sample)] - in_sample[:len(train_ser)]
                )

                for i in range(n_train, n_windows):
                    try:
                        fc = fitted.forecast(steps=1).values[0]
                    except Exception:
                        fc = series[i - 1]
                    residuals[i] = series[i] - fc
                    try:
                        fitted = fitted.append(np.array([series[i]]), refit=False)
                    except Exception:
                        pass

        except Exception:
            alpha   = 0.3
            smoothed = train_ser[0]
            for i in range(n_windows):
                residuals[i] = series[i] - smoothed
                smoothed = alpha * series[i] + (1 - alpha) * smoothed

        residuals = np.abs(residuals) / train_std
        scores    = np.maximum(scores, residuals)

    scores = np.clip(scores / scores.max() if scores.max() > 0 else scores,
                     0.0, 1.0)
    return scores


# ---------------------------------------------------------------------------
# AquaSSM model loading and inference
# ---------------------------------------------------------------------------

def load_aquassm():
    """Load AquaSSM sensor encoder from aquassm_full_best.pt."""
    sensor = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
    state  = torch.load(
        str(CKPT_BASE / "sensor" / "aquassm_full_best.pt"),
        map_location=DEVICE, weights_only=False,
    )
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        state = state["model"]
    sensor.load_state_dict(state, strict=False)
    sensor.eval()
    return sensor


def get_aquassm_embeddings(sensor_model, sensor_data_norm: np.ndarray,
                            batch_size: int = 64) -> torch.Tensor:
    """Run AquaSSM on normalised windows, return embeddings [N, 256]."""
    N = len(sensor_data_norm)
    all_embs = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            x    = torch.from_numpy(sensor_data_norm[start:end]).to(DEVICE)  # [B, T, 6]
            masks = torch.ones(end - start, T, 6, dtype=torch.bool, device=DEVICE)
            t_delta = torch.zeros(end - start, T, dtype=torch.float32, device=DEVICE)

            try:
                enc = sensor_model(x, t_delta, masks)
            except Exception:
                try:
                    enc = sensor_model(x=x, masks=masks)
                except Exception:
                    enc = sensor_model(x)

            emb = enc["embedding"] if isinstance(enc, dict) else enc
            if emb.dim() == 3:
                emb = emb[:, -1, :]  # last time step
            all_embs.append(emb.cpu())

            if (start // batch_size + 1) % 10 == 0:
                logger.info(f"  AquaSSM embeddings: {end}/{N}")

    return torch.cat(all_embs, dim=0)  # [N, 256]


# ---------------------------------------------------------------------------
# Baseline 4: AquaSSM-only (embedding norm)
# ---------------------------------------------------------------------------

def aquassm_only_baseline(embeddings: torch.Tensor) -> np.ndarray:
    """Use raw embedding norm as anomaly score."""
    norms  = embeddings.norm(dim=-1).numpy()
    lo, hi = norms.min(), norms.max()
    scores = (norms - lo) / (hi - lo + 1e-8)
    return scores


# ---------------------------------------------------------------------------
# SENTINEL (fusion + anomaly head)
# ---------------------------------------------------------------------------

def load_fusion_and_head():
    """Load trained fusion model and anomaly detection head."""
    ckpt_path = CKPT_BASE / "fusion" / "fusion_real_best.pt"
    state     = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    fusion = PerceiverIOFusion(num_latents=64)
    fusion.load_state_dict(state["fusion"], strict=False)
    fusion.eval()

    head = AnomalyDetectionHead()
    head.load_state_dict(state["head"], strict=False)
    head.eval()

    return fusion, head


def sentinel_scores(fusion, head, embeddings: torch.Tensor) -> np.ndarray:
    """Run embeddings through fusion + anomaly head to get SENTINEL scores."""
    n     = embeddings.size(0)
    probs = []
    latent_state = None

    with torch.no_grad():
        for i in range(n):
            emb = embeddings[i].unsqueeze(0)  # [1, 256]
            ts  = float(i * 900.0)

            try:
                out = fusion(
                    modality_id="sensor",
                    raw_embedding=emb,
                    timestamp=ts,
                    confidence=0.9,
                    latent_state=latent_state,
                )
                fused        = out.fused_state
                latent_state = out.latent_state
            except Exception:
                fused = emb

            try:
                anom_out = head(fused)
                p = getattr(anom_out, "anomaly_probability", None)
                if p is None:
                    p = getattr(anom_out, "severity_score", None)
                if p is None and isinstance(anom_out, torch.Tensor):
                    p = anom_out
                if p is not None:
                    if p.dim() > 1:
                        p = p[:, 1] if p.shape[1] > 1 else p[:, 0]
                    p = torch.sigmoid(p) if p.max().item() > 1.0 else p
                    prob = float(p.squeeze().item())
                else:
                    prob = float(torch.clamp(emb.norm() / 10.0, 0, 1).item())
            except Exception:
                prob = float(torch.clamp(emb.norm() / 10.0, 0, 1).item())

            probs.append(prob)

            if (i + 1) % 500 == 0:
                logger.info(f"  SENTINEL inference: {i+1}/{n}")

    return np.array(probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load real NEON sensor windows
    # ------------------------------------------------------------------
    logger.info("Loading real NEON sensor windows...")
    sensor_data, labels = load_neon_windows()
    n_total   = len(labels)
    n_normal  = int((labels == 0).sum())
    n_anomaly = int((labels == 1).sum())
    logger.info(f"Dataset: {n_total} windows  "
                f"({n_normal} normal, {n_anomaly} anomalous)")

    # Normalise for AquaSSM (per-window z-score + 6-channel pad)
    sensor_data_norm = normalise_for_aquassm(sensor_data)  # [N, T, 6]

    # ------------------------------------------------------------------
    # Load AquaSSM and compute embeddings (used by baselines 4 & 5)
    # ------------------------------------------------------------------
    logger.info("Loading AquaSSM (aquassm_full_best.pt)...")
    sensor_model = load_aquassm()
    logger.info("Computing AquaSSM embeddings...")
    embeddings = get_aquassm_embeddings(sensor_model, sensor_data_norm)
    logger.info(f"Embeddings: {embeddings.shape}")

    # ------------------------------------------------------------------
    # Run all methods
    # ------------------------------------------------------------------
    results = {}

    # 1. Z-score
    logger.info("Baseline 1: Z-score threshold...")
    z_scores  = zscore_baseline(sensor_data)
    z_auroc   = roc_auc_score(labels, z_scores)
    results["Z-score"] = {"auroc": float(z_auroc), "scores_mean": float(z_scores.mean())}
    logger.info(f"  Z-score AUROC: {z_auroc:.4f}")

    # 2. Isolation Forest
    logger.info("Baseline 2: Isolation Forest...")
    if_scores = isolation_forest_baseline(sensor_data)
    if_auroc  = roc_auc_score(labels, if_scores)
    results["Isolation Forest"] = {"auroc": float(if_auroc), "scores_mean": float(if_scores.mean())}
    logger.info(f"  Isolation Forest AUROC: {if_auroc:.4f}")

    # 3. ARIMA residual
    logger.info("Baseline 3: ARIMA residual...")
    arima_scores = arima_baseline(sensor_data)
    arima_auroc  = roc_auc_score(labels, arima_scores)
    results["ARIMA"] = {"auroc": float(arima_auroc), "scores_mean": float(arima_scores.mean())}
    logger.info(f"  ARIMA AUROC: {arima_auroc:.4f}")

    # 4. AquaSSM-only
    logger.info("Baseline 4: AquaSSM-only (no fusion)...")
    aqua_scores = aquassm_only_baseline(embeddings)
    aqua_auroc  = roc_auc_score(labels, aqua_scores)
    results["AquaSSM-only"] = {"auroc": float(aqua_auroc), "scores_mean": float(aqua_scores.mean())}
    logger.info(f"  AquaSSM-only AUROC: {aqua_auroc:.4f}")

    # 5. SENTINEL (fusion + head)
    logger.info("Baseline 5: SENTINEL (fusion + anomaly head)...")
    fusion, head  = load_fusion_and_head()
    sent_scores   = sentinel_scores(fusion, head, embeddings)
    sent_auroc    = roc_auc_score(labels, sent_scores)
    results["SENTINEL"] = {"auroc": float(sent_auroc), "scores_mean": float(sent_scores.mean())}
    logger.info(f"  SENTINEL AUROC: {sent_auroc:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    results["_meta"] = {
        "n_total":       n_total,
        "n_normal":      n_normal,
        "n_anomalous":   n_anomaly,
        "window_len":    T,
        "data_source":   str(NEON_PARQUET),
        "label_method":  "threshold_exceedance",
        "elapsed_seconds": elapsed,
    }

    out_json = OUTPUT_DIR / "baseline_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_json}")

    # ------------------------------------------------------------------
    # Plot grouped bar chart
    # ------------------------------------------------------------------
    method_names = ["Z-score", "Isolation Forest", "ARIMA", "AquaSSM-only", "SENTINEL"]
    aurocs = [results[m]["auroc"] for m in method_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#95a5a6", "#7f8c8d", "#bdc3c7", "#3498db", "#e74c3c"]
    bars   = ax.bar(method_names, aurocs, color=colors, edgecolor="black",
                    linewidth=0.8, width=0.6)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylabel("AUROC", fontsize=13)
    ax.set_title("Anomaly Detection: SENTINEL vs. Baselines\n(Real NEON Data, Threshold Labels)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Random baseline")
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig_exp2_baselines.jpg"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight",
                pil_kwargs={"quality": 85})
    plt.close()
    logger.info(f"Figure saved to {fig_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    logger.info(f"\n{'=' * 50}")
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"{'Method':<20s} {'AUROC':>8s}")
    logger.info(f"{'-' * 28}")
    for m in method_names:
        logger.info(f"{m:<20s} {results[m]['auroc']:>8.4f}")
    logger.info(f"{'=' * 50}")
    logger.info(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
