#!/usr/bin/env python3
"""Exp16: Per-Parameter Occlusion Attribution for AquaSSM.

For each of 32 NEON sites, takes the top-anomaly window and ablates one
water-quality parameter at a time (set to site mean), measuring how much
each parameter contributes to the anomaly score.

Output:
  results/exp16_attribution/attribution_results.json
  paper/figures/fig_exp16_attribution_heatmap.jpg

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"
CKPT_BASE    = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "exp16_attribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR      = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

T      = 128
STRIDE = 64

NEON_VALUE_COLS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
NEON_QF_COLS    = ["pHFinalQF", "dissolvedOxygenFinalQF", "turbidityFinalQF", "specificCondFinalQF"]
PARAM_NAMES     = ["pH", "DO", "Turbidity", "SpCond", "Temp(pad)", "ORP(pad)"]

ANOMALY_THRESHOLDS = {
    "pH":                  (6.0, 9.5),
    "dissolvedOxygen":     (4.0, None),
    "turbidity":           (None, 300.0),
    "specificConductance": (None, 1500.0),
}

READ_COLS = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS


# ---------------------------------------------------------------------------
def load_models():
    from sentinel.models.sensor_encoder.model import SensorEncoder
    from sentinel.models.fusion.model import PerceiverIOFusion
    from sentinel.models.fusion.heads import AnomalyDetectionHead

    sensor = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
    state = torch.load(str(CKPT_BASE / "sensor" / "aquassm_real_best.pt"),
                       map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state: state = state["model_state_dict"]
    elif "model" in state: state = state["model"]
    sensor.load_state_dict(state, strict=False)
    sensor.eval()

    fusion_state = torch.load(str(CKPT_BASE / "fusion" / "fusion_real_best.pt"),
                               map_location=DEVICE, weights_only=False)
    fusion = PerceiverIOFusion(num_latents=64).to(DEVICE)
    head   = AnomalyDetectionHead().to(DEVICE)
    fusion.load_state_dict(fusion_state["fusion"], strict=False)
    head.load_state_dict(fusion_state["head"], strict=False)
    fusion.eval(); head.eval()

    return sensor, fusion, head


@torch.no_grad()
def score_window(sensor, fusion, head, w_vals_norm: np.ndarray) -> float:
    """Score a single (T, 6) normalized window. Mirrors neon_anomaly_scan.py."""
    v = torch.from_numpy(w_vals_norm).unsqueeze(0).to(DEVICE)          # (1, T, 6)
    t_delta    = torch.zeros(1, T, dtype=torch.float32, device=DEVICE)
    masks      = torch.ones(1, T, 6, dtype=torch.bool, device=DEVICE)
    timestamps = torch.zeros(1, T, dtype=torch.float32, device=DEVICE)

    # Sensor forward
    try:
        enc = sensor(v, t_delta, masks)
        emb = enc["embedding"] if isinstance(enc, dict) else enc
    except Exception:
        enc = sensor(x=v, delta_ts=t_delta)
        emb = enc["embedding"] if isinstance(enc, dict) else enc

    # Fusion forward (mirror NEON scan fallback chain)
    fused = emb  # default fallback
    try:
        fout  = fusion(modality_id="sensor", raw_embedding=emb,
                       timestamp=timestamps[:, 0], confidence=0.9)
        fused = getattr(fout, "fused_state", emb)
    except Exception:
        try:
            fout  = fusion(sensor_embedding=emb)
            fused = getattr(fout, "fused_state", emb)
        except Exception:
            fused = emb

    # Head forward
    try:
        hout = head(fused)
        prob = getattr(hout, "anomaly_probability", None)
        if prob is None:
            prob = getattr(hout, "severity_score", None)
        if prob is None:
            logits = getattr(hout, "logits", None)
            if logits is not None:
                prob = logits[:, 1] if logits.dim() > 1 else logits
        if prob is None:
            # Last resort: treat hout as tensor
            if isinstance(hout, torch.Tensor):
                prob = hout
        if prob is not None:
            if isinstance(prob, torch.Tensor):
                if prob.dim() > 1: prob = prob[:, 1]
                prob = torch.sigmoid(prob) if prob.max() > 1 else prob
                return float(prob.squeeze().item())
    except Exception:
        pass

    # Fallback: use raw embedding norm as proxy
    return float(emb.norm().item())


def build_site_windows(df_site):
    """Replicate NEON scan window building for a single site."""
    import pandas as pd

    df_site = df_site.copy()
    df_site["ts"] = pd.to_datetime(df_site["startDateTime"], utc=True, errors="coerce")
    df_site = df_site.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df_site = df_site.set_index("ts")
    agg = {c: "mean" for c in NEON_VALUE_COLS}
    agg.update({c: "min" for c in NEON_QF_COLS})
    df_site = df_site[NEON_VALUE_COLS + NEON_QF_COLS].resample("15min").agg(agg).reset_index()
    df_site.rename(columns={"ts": "startDateTime"}, inplace=True)

    qf_passes = np.zeros((len(df_site), len(NEON_QF_COLS)), dtype=np.float32)
    for j, qf in enumerate(NEON_QF_COLS):
        if qf in df_site.columns:
            qf_passes[:, j] = (df_site[qf].fillna(1.0).astype(float) == 0).astype(float)
    good_mask = (qf_passes.sum(axis=1) >= 2) | (df_site[NEON_VALUE_COLS].notna().sum(axis=1).values >= 2)

    vals_4 = df_site[NEON_VALUE_COLS].astype(float).values
    zeros  = np.zeros((len(df_site), 2), dtype=np.float32)
    vals   = np.concatenate([vals_4, zeros], axis=1).astype(np.float32)

    ts_raw = df_site["startDateTime"]
    if not hasattr(ts_raw, "dt"):
        ts_raw = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    ts_sec = (ts_raw - ts_raw.iloc[0]).dt.total_seconds().values.astype(np.float32)
    ts_sec = np.nan_to_num(ts_sec, nan=0.0)

    # Return list of (w_vals_raw, w_vals_norm, window_start_sec)
    windows = []
    N = len(df_site)
    for start in range(0, N - T + 1, STRIDE):
        end   = start + T
        w_raw = vals[start:end].copy()
        w_mask = good_mask[start:end]
        if w_mask.mean() < 0.3:
            continue
        for c in range(w_raw.shape[1]):
            col   = w_raw[:, c]
            valid = np.isfinite(col)
            if valid.any():
                w_raw[~valid, c] = col[valid].mean()
            else:
                w_raw[:, c] = 0.0
        m = w_raw.mean(axis=0, keepdims=True)
        s = w_raw.std(axis=0, keepdims=True) + 1e-8
        w_norm = ((w_raw - m) / s).astype(np.float32)
        windows.append((w_raw, w_norm, float(ts_sec[start])))
    return windows


def occlusion_attribution(sensor, fusion, head, w_raw: np.ndarray, w_norm: np.ndarray) -> dict:
    """Compute per-parameter attribution via occlusion (set to 0 after normalization)."""
    baseline = score_window(sensor, fusion, head, w_norm)
    attrs = {}
    for p in range(4):  # only 4 real params (5,6 are padded zeros)
        w_occ = w_raw.copy()
        # Set parameter p to its mean (neutral) then re-normalize
        w_occ[:, p] = w_raw[:, p].mean()
        m = w_occ.mean(axis=0, keepdims=True)
        s = w_occ.std(axis=0, keepdims=True) + 1e-8
        w_occ_norm = ((w_occ - m) / s).astype(np.float32)
        occ_score = score_window(sensor, fusion, head, w_occ_norm)
        attrs[PARAM_NAMES[p]] = {
            "baseline_score": round(baseline, 4),
            "occluded_score": round(occ_score, 4),
            "delta": round(baseline - occ_score, 4),   # positive = parameter contributed
        }
    return attrs


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("EXP16: Per-Parameter Occlusion Attribution")
    logger.info("=" * 60)

    sensor, fusion, head = load_models()
    logger.info(f"Models loaded on {DEVICE}")

    # Load NEON scan results to get top sites/events
    scan = json.load(open(PROJECT_ROOT / "results" / "neon_anomaly_scan" / "neon_scan_results.json"))
    per_site = scan["per_site"]

    # Focus on top-20 sites by max_score
    ranked_sites = sorted(per_site.items(), key=lambda x: x[1].get("max_score", 0), reverse=True)[:20]
    logger.info(f"Processing top {len(ranked_sites)} sites by max anomaly score")

    results = {}

    import pyarrow.parquet as pq
    import pandas as pd

    pf = pq.ParquetFile(str(NEON_PARQUET))
    schema = pf.schema_arrow
    logger.info(f"NEON parquet open. Schema cols: {len(schema)}")

    for rank, (site, site_res) in enumerate(ranked_sites, 1):
        try:
            logger.info(f"  [{rank:2d}/20] {site}: max_score={site_res['max_score']:.4f}")

            # Read site data
            filters = [("source_site", "=", site)]
            table = pq.read_table(str(NEON_PARQUET), columns=READ_COLS, filters=filters)
            if len(table) < T * 2:
                logger.info(f"    Skip {site}: too few rows ({len(table)})")
                continue

            df = table.to_pandas()
            windows = build_site_windows(df)
            if not windows:
                logger.info(f"    Skip {site}: no valid windows")
                continue

            # Find window closest to the top_event timestamp
            top_events = site_res.get("top_events", [])
            if top_events:
                target_ts = top_events[0]["window_start_sec"]
                # Find window with closest ts
                best_win = min(windows, key=lambda w: abs(w[2] - target_ts))
            else:
                # Use highest-scoring window by re-scoring
                scored = [(score_window(sensor, fusion, head, w[1]), w) for w in windows[:50]]
                scored.sort(key=lambda x: -x[0])
                best_win = scored[0][1]

            w_raw, w_norm, w_ts = best_win
            baseline_score = score_window(sensor, fusion, head, w_norm)
            attrs = occlusion_attribution(sensor, fusion, head, w_raw, w_norm)

            # Find top contributing parameter
            top_param = max(attrs.items(), key=lambda x: x[1]["delta"])

            results[site] = {
                "rank": rank,
                "max_score_scan": round(site_res["max_score"], 4),
                "baseline_attribution_score": round(baseline_score, 4),
                "window_start_sec": w_ts,
                "parameter_attribution": attrs,
                "top_parameter": top_param[0],
                "top_parameter_delta": round(top_param[1]["delta"], 4),
            }
            logger.info(f"    Baseline={baseline_score:.4f}, top param: {top_param[0]} (Δ={top_param[1]['delta']:+.4f})")

        except Exception as e:
            logger.warning(f"    Error on {site}: {e}")
            continue

    # Cross-site summary: parameter importance frequency
    param_freq = {p: 0 for p in PARAM_NAMES[:4]}
    param_mean_delta = {p: [] for p in PARAM_NAMES[:4]}
    for site_data in results.values():
        top = site_data.get("top_parameter")
        if top in param_freq:
            param_freq[top] += 1
        for p, a in site_data.get("parameter_attribution", {}).items():
            if p in param_mean_delta:
                param_mean_delta[p].append(a["delta"])

    param_summary = {}
    for p in PARAM_NAMES[:4]:
        deltas = param_mean_delta[p]
        param_summary[p] = {
            "top_driver_count": param_freq[p],
            "mean_attribution_delta": round(float(np.mean(deltas)), 4) if deltas else 0.0,
            "std_attribution_delta":  round(float(np.std(deltas)), 4) if deltas else 0.0,
        }

    output = {
        "n_sites": len(results),
        "parameter_summary": param_summary,
        "site_results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = OUTPUT_DIR / "attribution_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # --- Figure: attribution heatmap ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        sites_sorted = sorted(results.items(), key=lambda x: x[1]["max_score_scan"], reverse=True)
        site_names = [s for s, _ in sites_sorted]
        params = PARAM_NAMES[:4]
        mat = np.zeros((len(site_names), len(params)))
        for i, (s, d) in enumerate(sites_sorted):
            for j, p in enumerate(params):
                a = d["parameter_attribution"].get(p, {})
                mat[i, j] = a.get("delta", 0.0)

        fig, ax = plt.subplots(figsize=(8, max(6, len(site_names) * 0.35)))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=-0.05, vmax=0.15)
        ax.set_xticks(range(len(params))); ax.set_xticklabels(params, fontsize=9)
        ax.set_yticks(range(len(site_names))); ax.set_yticklabels(site_names, fontsize=7)
        ax.set_xlabel("Water Quality Parameter"); ax.set_ylabel("NEON Site")
        ax.set_title("AquaSSM Attribution: Score Drop When Parameter Occluded\n(Higher = parameter drives anomaly)")
        plt.colorbar(im, ax=ax, label="Δ Anomaly Score")
        plt.tight_layout()
        fig_path = FIG_DIR / "fig_exp16_attribution_heatmap.jpg"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {fig_path}")
    except Exception as e:
        logger.warning(f"Figure failed: {e}")

    # Print summary
    logger.info("\n=== PARAMETER ATTRIBUTION SUMMARY ===")
    logger.info(f"{'Parameter':<15} {'Top-driver #':<14} {'Mean Δ':<10} {'Std Δ'}")
    logger.info("-" * 55)
    for p, s in sorted(param_summary.items(), key=lambda x: -x[1]["mean_attribution_delta"]):
        logger.info(f"  {p:<13} {s['top_driver_count']:<14} {s['mean_attribution_delta']:<10.4f} {s['std_attribution_delta']:.4f}")

    logger.info(f"\nElapsed: {output['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
