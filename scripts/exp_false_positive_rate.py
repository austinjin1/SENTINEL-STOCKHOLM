#!/usr/bin/env python3
"""Experiment B: False Positive Rate at Non-Event Sites.

Uses NEON scan results to identify non-event sites (label_anomaly_rate ~0 or
low mean score) and computes FPR at threshold=0.9 vs. the 6 case study events.

Because the NEON scan only stores top-20 events per site (not all window scores),
we re-run AquaSSM inference on 10 low-anomaly sites using the NEON parquet,
collecting all window scores, then compute fraction > 0.9.

Output: results/exp_false_positive/false_positive_results.json

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
CASE_DIR     = PROJECT_ROOT / "results" / "case_studies_real"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "exp_false_positive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BASE    = PROJECT_ROOT / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T      = 128
STRIDE = 64
FPR_THRESHOLD = 0.9

NEON_VALUE_COLS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
NEON_QF_COLS    = ["pHFinalQF", "dissolvedOxygenFinalQF", "turbidityFinalQF", "specificCondFinalQF"]
READ_COLS = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS

# Sites with no documented pollution events:
# Choose 10 sites from NEON scan that have label_anomaly_rate == 0.0
# (NEON scan uses threshold-based labeling, so 0.0 means no windows flagged
# by the heuristic anomaly labels in the raw data)
NON_EVENT_TARGET = 10


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


BATCH_SIZE = 32


@torch.no_grad()
def score_windows_batch(sensor, fusion, head, windows: list) -> list[float]:
    """Score a list of (T,6) normalized windows using batched inference."""
    all_scores = []
    for i in range(0, len(windows), BATCH_SIZE):
        batch = windows[i:i + BATCH_SIZE]
        B = len(batch)
        v          = torch.from_numpy(np.stack(batch)).to(DEVICE)   # (B, T, 6)
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
        # Fallback: embedding norm for this batch
        norms = emb.norm(dim=-1) if emb.dim() > 1 else emb.norm()
        all_scores.extend(norms.cpu().tolist())
    return all_scores


@torch.no_grad()
def score_window(sensor, fusion, head, w_norm: np.ndarray) -> float:
    """Single-window convenience wrapper."""
    return score_windows_batch(sensor, fusion, head, [w_norm])[0]


def build_windows(df):
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
        w_norm = ((w_raw - m) / s).astype(np.float32)
        windows.append(w_norm)
    return windows


def score_site_windows(sensor, fusion, head, df_site) -> list[float]:
    windows = build_windows(df_site)
    if not windows:
        return []
    return score_windows_batch(sensor, fusion, head, windows)


def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment B: False Positive Rate at Non-Event Sites")
    print("=" * 60)

    # Identify non-event sites from NEON scan
    with open(SCAN_RESULTS) as f:
        scan = json.load(f)
    per_site = scan["per_site"]

    # Select 10 sites with lowest label_anomaly_rate (prefer 0.0)
    ranked = sorted(per_site.items(),
                    key=lambda x: (x[1]["label_anomaly_rate"], x[1]["mean_score"]))
    # Take top 10 cleanest sites that have enough windows
    non_event_sites = [
        site for site, v in ranked
        if v["label_anomaly_rate"] == 0.0 and v["n_windows"] >= 20
    ][:NON_EVENT_TARGET]

    # Pad with low-rate sites if needed
    if len(non_event_sites) < NON_EVENT_TARGET:
        extras = [site for site, v in ranked
                  if site not in non_event_sites and v["n_windows"] >= 20]
        non_event_sites += extras[:NON_EVENT_TARGET - len(non_event_sites)]

    print(f"Non-event sites selected ({len(non_event_sites)}): {non_event_sites}")

    sensor, fusion, head = load_models()
    print(f"Models loaded on {DEVICE}")

    # Batch-read all non-event sites in one parquet call for efficiency
    print("  Batch-reading NEON parquet for non-event sites...")
    t_read = time.time()
    table = pq.read_table(str(NEON_PARQUET), columns=READ_COLS,
                          filters=[("source_site", "in", non_event_sites)])
    df_all = table.to_pandas()
    print(f"  Read {len(df_all)} rows in {time.time()-t_read:.1f}s")

    # Score non-event sites
    non_event_results = {}
    all_fpr = []
    for site in non_event_sites:
        t_site = time.time()
        df_site = df_all[df_all["source_site"] == site].copy()
        if len(df_site) < T * 2:
            print(f"  {site}: too few rows ({len(df_site)})")
            continue
        scores = score_site_windows(sensor, fusion, head, df_site)
        if not scores:
            print(f"  {site}: no valid windows")
            continue
        n_above = sum(1 for s in scores if s >= FPR_THRESHOLD)
        fpr = n_above / len(scores)
        all_fpr.append(fpr)
        print(f"  {site}: {len(scores)} windows, {n_above} above {FPR_THRESHOLD}"
              f" -> FPR={fpr:.4f}  ({time.time()-t_site:.1f}s)")
        non_event_results[site] = {
            "n_windows":       len(scores),
            "n_above_thresh":  n_above,
            "fpr":             round(fpr, 6),
            "mean_score":      round(float(np.mean(scores)), 4),
            "max_score":       round(float(np.max(scores)), 4),
            "p95_score":       round(float(np.percentile(scores, 95)), 4),
        }

    mean_fpr      = float(np.mean(all_fpr)) if all_fpr else 0.0
    specificity   = 1.0 - mean_fpr

    # Case study sites: fraction of ALL windows above threshold
    case_event_files = {
        "lake_erie_hab_2023":        "HAB",
        "jordan_lake_hab_nc":        "HAB",
        "klamath_river_hab_2021":    "HAB",
        "gulf_dead_zone_2023":       "hypoxia",
        "chesapeake_hypoxia_2018":   "hypoxia",
        "mississippi_salinity_2023": "salinity",
    }
    case_results = {}
    for eid, etype in case_event_files.items():
        path = CASE_DIR / f"{eid}_scores.json"
        with open(path) as f:
            d = json.load(f)
        all_scores = [s["anomaly_probability"] for s in d["scores"]]
        n_above    = sum(1 for s in all_scores if s >= FPR_THRESHOLD)
        rate       = n_above / len(all_scores)
        case_results[eid] = {
            "event_type":    etype,
            "n_windows":     len(all_scores),
            "n_above_thresh": n_above,
            "high_score_rate": round(rate, 4),
            "mean_score":    round(float(np.mean(all_scores)), 4),
            "max_score":     round(float(np.max(all_scores)), 4),
        }
        print(f"  [case] {eid}: {n_above}/{len(all_scores)} above {FPR_THRESHOLD}"
              f" -> rate={rate:.4f}")

    mean_case_rate = float(np.mean([v["high_score_rate"] for v in case_results.values()]))

    print(f"\n  Mean FPR (non-event sites):   {mean_fpr:.6f}")
    print(f"  Specificity at t=0.9:          {specificity:.6f}")
    print(f"  Mean high-score rate (events): {mean_case_rate:.4f}")

    output = {
        "experiment":       "B: False Positive Rate at Non-Event Sites",
        "threshold":        FPR_THRESHOLD,
        "non_event_sites":  non_event_results,
        "mean_fpr":         round(mean_fpr, 6),
        "specificity":      round(specificity, 6),
        "case_study_sites": case_results,
        "mean_case_high_score_rate": round(mean_case_rate, 4),
        "elapsed_s":        round(time.time() - t0, 1),
    }
    out_path = OUTPUT_DIR / "false_positive_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
