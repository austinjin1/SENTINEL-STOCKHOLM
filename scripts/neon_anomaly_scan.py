#!/usr/bin/env python3
"""NEON Aquatic Anomaly Scan — AquaSSM on 62.7M rows across 34 sites.

Reads NEON DP1.20288.001 chemical sonde WQ parquet, builds sliding 128-step
windows per site, runs the trained AquaSSM sensor encoder + fusion anomaly
head, and outputs a per-site anomaly timeline.

Usage:
    python scripts/neon_anomaly_scan.py

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"
CKPT_BASE    = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "neon_anomaly_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 128        # sequence length (matches AquaSSM training)
STRIDE = 64    # 50% overlap between windows
BATCH  = 64    # inference batch size

# 5 NEON WQ params available + 1 zero-padded (ORP missing in NEON)
# Order matches AquaSSM training: pH, DO, turbidity, conductivity, temp, ORP
NEON_VALUE_COLS = [
    "pH",
    "dissolvedOxygen",
    "turbidity",
    "specificConductance",
    # temp not in DP1.20288 → pad with 0
    # ORP not in NEON → pad with 0
]
NEON_QF_COLS = [
    "pHFinalQF",
    "dissolvedOxygenFinalQF",
    "turbidityFinalQF",
    "specificCondFinalQF",
]
# EPA anomaly thresholds (same as AquaSSM training labels)
ANOMALY_THRESHOLDS = {
    "pH":                  (6.0, 9.5),    # (low, high)
    "dissolvedOxygen":     (4.0, None),   # below 4 mg/L
    "turbidity":           (None, 300.0), # above 300 NTU
    "specificConductance": (None, 1500.0),# above 1500 µS/cm
}

# source_site is the actual NEON 4-char site code (siteID is often null)
READ_COLS = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    from sentinel.models.sensor_encoder.model import SensorEncoder
    from sentinel.models.fusion.model import PerceiverIOFusion
    from sentinel.models.fusion.heads import AnomalyDetectionHead

    sensor = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
    state = torch.load(
        str(CKPT_BASE / "sensor" / "aquassm_real_best.pt"),
        map_location=DEVICE, weights_only=False,
    )
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        state = state["model"]
    sensor.load_state_dict(state, strict=False)
    sensor.eval()
    logger.info("AquaSSM loaded")

    fusion_state = torch.load(
        str(CKPT_BASE / "fusion" / "fusion_real_best.pt"),
        map_location=DEVICE, weights_only=False,
    )
    fusion = PerceiverIOFusion(num_latents=64).to(DEVICE)
    head   = AnomalyDetectionHead().to(DEVICE)
    fusion.load_state_dict(fusion_state["fusion"], strict=False)
    head.load_state_dict(fusion_state["head"], strict=False)
    fusion.eval(); head.eval()
    logger.info("Fusion + AnomalyHead loaded")

    return sensor, fusion, head


# ---------------------------------------------------------------------------
# Site processing
# ---------------------------------------------------------------------------

def build_windows(df_site: pd.DataFrame):
    """Convert site time series to sliding-window tensors.

    Downsamples to 15-minute resolution before building windows to reduce
    the ~3M rows/site to ~35K rows, making window building tractable.
    """
    # Parse timestamps
    df_site = df_site.copy()
    df_site["ts"] = pd.to_datetime(df_site["startDateTime"], utc=True, errors="coerce")
    df_site = df_site.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # Downsample to 15-minute resolution (NEON records every 1 min)
    df_site = df_site.set_index("ts")
    agg = {}
    for col in NEON_VALUE_COLS:
        agg[col] = "mean"
    for col in NEON_QF_COLS:
        agg[col] = "min"   # min QF: 0 if ANY record passed
    df_site = df_site[NEON_VALUE_COLS + NEON_QF_COLS].resample("15min").agg(agg)
    df_site = df_site.reset_index()
    df_site.rename(columns={"ts": "startDateTime"}, inplace=True)

    # Mask: at least 2 parameters have valid data
    qf_passes = np.zeros((len(df_site), len(NEON_QF_COLS)), dtype=np.float32)
    for j, qf in enumerate(NEON_QF_COLS):
        if qf in df_site.columns:
            qf_passes[:, j] = (df_site[qf].fillna(1.0).astype(float) == 0).astype(float)
    good_mask = (qf_passes.sum(axis=1) >= 2) | (
        df_site[NEON_VALUE_COLS].notna().sum(axis=1).values >= 2
    )

    # Build value matrix (N, 4) — pad to 6 params
    vals_4 = df_site[NEON_VALUE_COLS].astype(float).values  # (N, 4)
    # Append two zero columns for temp and ORP
    zeros = np.zeros((len(df_site), 2), dtype=np.float32)
    vals = np.concatenate([vals_4, zeros], axis=1).astype(np.float32)  # (N, 6)

    # Timestamps as seconds since start
    ts = df_site["startDateTime"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
    ts_sec = (ts - ts.iloc[0]).dt.total_seconds().values.astype(np.float32)
    ts_sec = np.nan_to_num(ts_sec, nan=0.0)

    # Anomaly ground truth from EPA thresholds
    is_anomaly = np.zeros(len(df_site), dtype=bool)
    for col, (low, high) in ANOMALY_THRESHOLDS.items():
        v = df_site[col].astype(float)
        if low is not None:
            is_anomaly |= (v < low).values
        if high is not None:
            is_anomaly |= (v > high).values

    # Build windows
    windows, window_ts, window_labels = [], [], []
    N = len(df_site)
    for start in range(0, N - T + 1, STRIDE):
        end = start + T
        w_vals = vals[start:end]  # (T, 6)
        w_ts   = ts_sec[start:end]
        w_mask = good_mask[start:end]
        w_label = is_anomaly[start:end].any()

        # Skip if more than 70% invalid rows
        if w_mask.mean() < 0.3:
            continue

        # Replace NaNs/infs with column mean for valid rows; 0 if all NaN
        for c in range(w_vals.shape[1]):
            col = w_vals[:, c]
            valid = np.isfinite(col)
            if valid.any():
                w_vals[~valid, c] = col[valid].mean()
            else:
                w_vals[:, c] = 0.0

        # Normalize per parameter (z-score within window, keep float32)
        m = w_vals.mean(axis=0, keepdims=True).astype(np.float32)
        s = w_vals.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
        w_vals_norm = ((w_vals - m) / s).astype(np.float32)

        dt = np.diff(w_ts, prepend=0.0).clip(0, 3600)

        windows.append((w_vals_norm, w_ts, dt))
        window_ts.append(float(w_ts[0]))
        window_labels.append(bool(w_label))

    return windows, window_ts, window_labels


@torch.no_grad()
def run_site(site_id: str, df_site: pd.DataFrame, sensor, fusion, head) -> dict:
    windows, win_ts, win_labels = build_windows(df_site)
    if not windows:
        return {"site": site_id, "n_windows": 0, "status": "insufficient_data"}

    scores, times = [], []
    for b_start in range(0, len(windows), BATCH):
        batch = windows[b_start:b_start + BATCH]
        B = len(batch)

        vals_arr = np.stack([w[0] for w in batch], axis=0)      # (B, T, 6)
        ts_arr   = np.stack([w[1] for w in batch], axis=0)      # (B, T)
        dt_arr   = np.stack([w[2] for w in batch], axis=0)      # (B, T)

        values    = torch.from_numpy(vals_arr.astype(np.float32)).to(DEVICE)
        timestamps= torch.from_numpy(ts_arr.astype(np.float32)).to(DEVICE)
        delta_ts  = torch.from_numpy(dt_arr.astype(np.float32)).to(DEVICE)
        # masks must be [B, T, num_params] for per-parameter validity
        masks     = torch.ones(B, T, values.shape[2], dtype=torch.bool, device=DEVICE)

        try:
            enc = sensor(x=values, timestamps=timestamps,
                         delta_ts=delta_ts, masks=masks)
            emb = enc["embedding"]  # (B, 256)
        except Exception:
            try:
                enc = sensor(x=values, delta_ts=delta_ts)
                emb = enc["embedding"]
            except Exception as e:
                logger.warning(f"  Sensor forward failed: {e}")
                continue

        # Fuse sensor embedding — try PerceiverIO API first, fall back to raw emb
        try:
            fout = fusion(modality_id="sensor", raw_embedding=emb,
                          timestamp=timestamps[:, 0], confidence=0.9)
            fused = getattr(fout, "fused_state", emb)
        except Exception:
            try:
                fout = fusion(sensor_embedding=emb)
                fused = getattr(fout, "fused_state", emb)
            except Exception:
                fused = emb  # fallback: use raw sensor embedding

        try:
            hout = head(fused)
            # AnomalyOutput is a dataclass, not a dict — use getattr
            prob = getattr(hout, "anomaly_probability", None)
            if prob is None:
                prob = getattr(hout, "severity_score", None)
            if prob is None:
                logits = getattr(hout, "logits", None)
                if logits is not None:
                    prob = logits[:, 1] if logits.dim() > 1 else logits
            if prob is not None:
                if prob.dim() > 1:
                    prob = prob[:, 1]  # take anomaly class
                prob = torch.sigmoid(prob) if prob.max() > 1 else prob
                scores.extend(prob.cpu().tolist())
            else:
                scores.extend([0.0] * B)
        except Exception as e:
            logger.warning(f"  Head forward failed: {e}")
            scores.extend([0.0] * B)

        times.extend(win_ts[b_start:b_start + B])

    if not scores:
        return {"site": site_id, "n_windows": len(windows), "status": "inference_failed"}

    scores_arr = np.array(scores)
    labels_arr = np.array(win_labels[:len(scores)])

    # Find top anomalous windows
    top_idx = np.argsort(scores_arr)[::-1][:20]
    top_events = [
        {"window_start_sec": float(times[i]), "score": float(scores_arr[i]),
         "labeled_anomaly": bool(labels_arr[i])}
        for i in top_idx
    ]

    return {
        "site": site_id,
        "n_rows": len(df_site),
        "n_windows": len(scores),
        "status": "success",
        "mean_score": float(scores_arr.mean()),
        "max_score": float(scores_arr.max()),
        "p95_score": float(np.percentile(scores_arr, 95)),
        "n_label_anomaly": int(labels_arr.sum()),
        "label_anomaly_rate": float(labels_arr.mean()),
        "top_events": top_events,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"NEON parquet: {NEON_PARQUET}")
    t0 = time.time()

    # Load models
    sensor, fusion, head = load_models()

    # Read parquet — only needed columns
    logger.info("Reading NEON parquet columns...")
    t_read = time.time()
    pf = pq.ParquetFile(str(NEON_PARQUET))
    table = pf.read(columns=READ_COLS)
    df = table.to_pandas()
    logger.info(f"Loaded {len(df):,} rows in {time.time()-t_read:.1f}s")

    sites = sorted(df["source_site"].dropna().unique())
    logger.info(f"Sites: {len(sites)} — {sites}")

    # Process each site
    all_results = {}
    for i, site in enumerate(sites):
        df_site = df[df["source_site"] == site].copy()
        logger.info(f"[{i+1}/{len(sites)}] {site}: {len(df_site):,} rows")
        t_site = time.time()
        result = run_site(site, df_site, sensor, fusion, head)
        result["elapsed_s"] = round(time.time() - t_site, 1)
        all_results[site] = result
        logger.info(
            f"  → {result.get('n_windows', 0)} windows, "
            f"max_score={result.get('max_score', 0):.4f}, "
            f"mean={result.get('mean_score', 0):.4f}  [{result['elapsed_s']}s]"
        )

    # Sort sites by max anomaly score
    ranked = sorted(
        [(s, r) for s, r in all_results.items() if r.get("status") == "success"],
        key=lambda x: x[1]["max_score"], reverse=True
    )

    # Summary
    summary = {
        "n_sites_processed": len(all_results),
        "n_sites_success": len(ranked),
        "total_windows": sum(r.get("n_windows", 0) for r in all_results.values()),
        "elapsed_s": round(time.time() - t0, 1),
        "top_sites_by_anomaly_score": [
            {"site": s, "max_score": r["max_score"], "mean_score": r["mean_score"],
             "n_label_anomaly": r.get("n_label_anomaly", 0)}
            for s, r in ranked[:10]
        ],
        "per_site": all_results,
    }

    out_path = OUTPUT_DIR / "neon_scan_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary table
    logger.info("\n=== TOP ANOMALOUS NEON SITES ===")
    logger.info(f"{'Site':<12} {'MaxScore':>9} {'MeanScore':>10} {'LabelRate':>10} {'Windows':>8}")
    logger.info("-" * 55)
    for s, r in ranked[:15]:
        logger.info(
            f"{s:<12} {r['max_score']:>9.4f} {r['mean_score']:>10.4f} "
            f"{r.get('label_anomaly_rate', 0):>10.1%} {r['n_windows']:>8}"
        )
    logger.info(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
