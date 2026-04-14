#!/usr/bin/env python3
"""Experiment E: Pollution Type Fingerprint.

Groups NEON windows by pollution proxy thresholds and computes mean anomaly
score distribution per group:
  - HAB proxy:           pH > 9
  - Hypoxia:             DO < 4 mg/L
  - High conductance:    SpCond > 1500 µS/cm
  - Temperature anomaly: temp > 30°C  (NEON has no water temp, use sensor depth
                         as fallback — we report N/A for temp group if absent)

For each group: mean anomaly score, quantiles, top contributing sites.

Output: results/exp_pollution_fingerprint/fingerprint_results.json

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"
SCAN_RESULTS = PROJECT_ROOT / "results" / "neon_anomaly_scan" / "neon_scan_results.json"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "exp_pollution_fingerprint"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BASE    = PROJECT_ROOT / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T      = 128
STRIDE = 64

NEON_VALUE_COLS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
NEON_QF_COLS    = ["pHFinalQF", "dissolvedOxygenFinalQF", "turbidityFinalQF", "specificCondFinalQF"]
READ_COLS = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS

# Pollution type definitions (applied to raw window mean values)
# Index in NEON_VALUE_COLS: pH=0, DO=1, turbidity=2, SpCond=3
POLLUTION_TYPES = {
    "HAB":              {"channel": 0, "op": ">",  "value": 9.0,    "label": "pH > 9 (HAB proxy)"},
    "Hypoxia":          {"channel": 1, "op": "<",  "value": 4.0,    "label": "DO < 4 mg/L"},
    "HighConductance":  {"channel": 3, "op": ">",  "value": 1500.0, "label": "SpCond > 1500 µS/cm"},
    # Temperature: NEON parquet lacks water temp column; we flag this as N/A
}


def load_models():
    from sentinel.models.sensor_encoder.model import SensorEncoder
    from sentinel.models.fusion.model import PerceiverIOFusion
    from sentinel.models.fusion.heads import AnomalyDetectionHead

    sensor = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
    state  = torch.load(str(CKPT_BASE / "sensor" / "aquassm_full_best.pt"),
                        map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state: state = state["model_state_dict"]
    elif "model" in state:          state = state["model"]
    sensor.load_state_dict(state, strict=False)
    sensor.eval()

    fusion_state = torch.load(str(CKPT_BASE / "fusion" / "fusion_real_best.pt"),
                               map_location=DEVICE, weights_only=False)
    fusion = PerceiverIOFusion(num_latents=64).to(DEVICE)
    head   = AnomalyDetectionHead().to(DEVICE)
    fusion.load_state_dict(fusion_state["fusion"], strict=False)
    head.load_state_dict(fusion_state["head"],   strict=False)
    fusion.eval(); head.eval()
    return sensor, fusion, head


@torch.no_grad()
def score_window(sensor, fusion, head, w_norm: np.ndarray) -> float:
    v          = torch.from_numpy(w_norm).unsqueeze(0).to(DEVICE)
    t_delta    = torch.zeros(1, T, dtype=torch.float32, device=DEVICE)
    masks      = torch.ones(1, T, 6, dtype=torch.bool,  device=DEVICE)
    timestamps = torch.zeros(1, T, dtype=torch.float32, device=DEVICE)

    try:
        enc = sensor(v, t_delta, masks)
        emb = enc["embedding"] if isinstance(enc, dict) else enc
    except Exception:
        enc = sensor(x=v, delta_ts=t_delta)
        emb = enc["embedding"] if isinstance(enc, dict) else enc

    fused = emb
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

    try:
        hout = head(fused)
        prob = getattr(hout, "anomaly_probability", None)
        if prob is None: prob = getattr(hout, "severity_score", None)
        if prob is None:
            logits = getattr(hout, "logits", None)
            if logits is not None:
                prob = logits[:, 1] if logits.dim() > 1 else logits
        if prob is None and isinstance(hout, torch.Tensor):
            prob = hout
        if prob is not None:
            if isinstance(prob, torch.Tensor):
                if prob.dim() > 1: prob = prob[:, 1]
                prob = torch.sigmoid(prob) if prob.max() > 1 else prob
                return float(prob.squeeze().item())
    except Exception:
        pass
    return float(emb.norm().item())


BATCH_SIZE = 32


@torch.no_grad()
def score_windows_batch(sensor, fusion, head, windows: list) -> list[float]:
    """Batch-score a list of (T,6) normalized windows."""
    all_scores = []
    for i in range(0, len(windows), BATCH_SIZE):
        batch = windows[i:i + BATCH_SIZE]
        B = len(batch)
        v          = torch.from_numpy(np.stack(batch)).to(DEVICE)
        t_delta    = torch.zeros(B, T, dtype=torch.float32, device=DEVICE)
        masks      = torch.ones(B, T, 6, dtype=torch.bool,  device=DEVICE)
        timestamps = torch.zeros(B, T, dtype=torch.float32, device=DEVICE)
        try:
            enc = sensor(v, t_delta, masks)
            emb = enc["embedding"] if isinstance(enc, dict) else enc
        except Exception:
            enc = sensor(x=v, delta_ts=t_delta)
            emb = enc["embedding"] if isinstance(enc, dict) else enc
        fused = emb
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
        try:
            hout = head(fused)
            prob = getattr(hout, "anomaly_probability", None)
            if prob is None: prob = getattr(hout, "severity_score", None)
            if prob is None:
                logits = getattr(hout, "logits", None)
                if logits is not None:
                    prob = logits[:, 1] if logits.dim() > 1 else logits
            if prob is None and isinstance(hout, torch.Tensor):
                prob = hout
            if prob is not None:
                if isinstance(prob, torch.Tensor):
                    if prob.dim() > 1: prob = prob[:, 1]
                    prob = torch.sigmoid(prob) if prob.max() > 1 else prob
                    all_scores.extend(prob.cpu().tolist())
                    continue
        except Exception:
            pass
        norms = emb.norm(dim=-1) if emb.dim() > 1 else emb.norm()
        all_scores.extend(norms.cpu().tolist())
    return all_scores


def build_windows_with_raw(df):
    """Returns list of (w_raw_mean, w_norm) where w_raw_mean is mean over T for each channel."""
    import pandas as pd
    df = df.copy()
    df["ts"] = pd.to_datetime(df["startDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True).set_index("ts")
    agg = {c: "mean" for c in NEON_VALUE_COLS}
    agg.update({c: "min" for c in NEON_QF_COLS})
    df = df[NEON_VALUE_COLS + NEON_QF_COLS].resample("15min").agg(agg).reset_index()
    df.rename(columns={"ts": "startDateTime"}, inplace=True)

    qf_passes = np.zeros((len(df), len(NEON_QF_COLS)), dtype=np.float32)
    for j, qf in enumerate(NEON_QF_COLS):
        if qf in df.columns:
            qf_passes[:, j] = (df[qf].fillna(1.0).astype(float) == 0).astype(float)
    good_mask = (qf_passes.sum(axis=1) >= 2) | (df[NEON_VALUE_COLS].notna().sum(axis=1).values >= 2)

    vals_4 = df[NEON_VALUE_COLS].astype(float).values
    zeros  = np.zeros((len(df), 2), dtype=np.float32)
    vals   = np.concatenate([vals_4, zeros], axis=1).astype(np.float32)

    windows = []
    N = len(df)
    for start in range(0, N - T + 1, STRIDE):
        end   = start + T
        w_raw = vals[start:end].copy()
        if good_mask[start:end].mean() < 0.3:
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
        w_norm    = ((w_raw - m) / s).astype(np.float32)
        w_raw_mean = w_raw[:, :4].mean(axis=0)  # mean of 4 real channels over window
        windows.append((w_raw_mean, w_norm))
    return windows


def classify_window(w_raw_mean: np.ndarray) -> list[str]:
    """Return list of pollution type labels this window falls into."""
    labels = []
    for ptype, cfg in POLLUTION_TYPES.items():
        val = w_raw_mean[cfg["channel"]]
        if np.isnan(val):
            continue
        if cfg["op"] == ">" and val > cfg["value"]:
            labels.append(ptype)
        elif cfg["op"] == "<" and val < cfg["value"]:
            labels.append(ptype)
    return labels


def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment E: Pollution Type Fingerprint")
    print("=" * 60)

    with open(SCAN_RESULTS) as f:
        scan = json.load(f)
    per_site = scan["per_site"]
    all_sites = list(per_site.keys())

    sensor, fusion, head = load_models()
    print(f"Models loaded on {DEVICE}")
    print(f"Processing {len(all_sites)} NEON sites...")

    # Batch-read all sites at once - much faster than per-site reads
    print("  Batch-reading NEON parquet for all sites...")
    t_read = time.time()
    table  = pq.read_table(str(NEON_PARQUET), columns=READ_COLS)
    df_all = table.to_pandas()
    print(f"  Read {len(df_all)} rows in {time.time()-t_read:.1f}s")

    # Group windows: pollution_type -> list of (score, site)
    groups: dict[str, list] = {ptype: [] for ptype in POLLUTION_TYPES}
    groups["Normal"] = []  # windows not in any pollution group
    site_contributions: dict[str, dict[str, int]] = {}

    for site in all_sites:
        t_site = time.time()
        try:
            df_site = df_all[df_all["source_site"] == site].copy()
            if len(df_site) < T * 2:
                continue
            windows = build_windows_with_raw(df_site)
            if not windows:
                continue

            site_contributions[site] = {ptype: 0 for ptype in list(POLLUTION_TYPES.keys()) + ["Normal"]}

            w_raw_means = [w[0] for w in windows]
            w_norms     = [w[1] for w in windows]
            scores      = score_windows_batch(sensor, fusion, head, w_norms)

            for score, w_raw_mean in zip(scores, w_raw_means):
                labels = classify_window(w_raw_mean)
                if labels:
                    for lbl in labels:
                        groups[lbl].append((score, site))
                        site_contributions[site][lbl] = site_contributions[site].get(lbl, 0) + 1
                else:
                    groups["Normal"].append((score, site))
                    site_contributions[site]["Normal"] += 1

            counts = {k: len(v) for k, v in groups.items()}
            print(f"  {site}: {len(windows)} windows  groups_so_far={counts}"
                  f"  ({time.time()-t_site:.1f}s)")
        except Exception as e:
            print(f"  {site}: ERROR {e}")
            continue

    # Compute statistics per group
    group_stats = {}
    for ptype, entries in groups.items():
        if not entries:
            group_stats[ptype] = {"n_windows": 0, "note": "no windows in this category"}
            continue
        scores = [e[0] for e in entries]
        arr    = np.array(scores)
        # Top contributing sites
        from collections import Counter
        site_counts = Counter(e[1] for e in entries)
        top_sites   = site_counts.most_common(5)

        group_stats[ptype] = {
            "n_windows":  len(scores),
            "mean_score": round(float(arr.mean()), 4),
            "median":     round(float(np.median(arr)), 4),
            "p25":        round(float(np.percentile(arr, 25)), 4),
            "p75":        round(float(np.percentile(arr, 75)), 4),
            "p90":        round(float(np.percentile(arr, 90)), 4),
            "p95":        round(float(np.percentile(arr, 95)), 4),
            "max_score":  round(float(arr.max()), 4),
            "std_score":  round(float(arr.std()), 4),
            "top_sites":  [{"site": s, "n": n} for s, n in top_sites],
        }
        label = POLLUTION_TYPES.get(ptype, {}).get("label", ptype)
        print(f"\n  {ptype} ({label}):")
        print(f"    n_windows={len(scores)}  mean={arr.mean():.4f}"
              f"  median={np.median(arr):.4f}"
              f"  p90={np.percentile(arr, 90):.4f}"
              f"  max={arr.max():.4f}")
        print(f"    Top sites: {top_sites[:3]}")

    # Temperature anomaly: report as N/A (no water temp column in NEON)
    group_stats["TempAnomaly"] = {
        "n_windows": 0,
        "note": "Water temperature column absent from NEON DP1.20288.001; "
                "temperature anomaly group cannot be computed from this dataset.",
        "threshold_definition": "waterTemp > 30°C",
    }

    # Site-level summary
    site_summary = {}
    for site, contrib in site_contributions.items():
        dominant = max(contrib, key=contrib.get) if contrib else "Normal"
        site_summary[site] = {
            "window_type_counts": contrib,
            "dominant_type":      dominant,
        }

    output = {
        "experiment":          "E: Pollution Type Fingerprint",
        "pollution_definitions": {
            ptype: cfg["label"] for ptype, cfg in POLLUTION_TYPES.items()
        },
        "group_statistics":    group_stats,
        "site_contributions":  site_summary,
        "elapsed_s":           round(time.time() - t0, 1),
    }
    out_path = OUTPUT_DIR / "fingerprint_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
