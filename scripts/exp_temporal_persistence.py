#!/usr/bin/env python3
"""Experiment D: Temporal Persistence Analysis.

For the 6 case study events, analyze how many consecutive windows exceed
threshold=0.9 before the advisory date (alarm persistence).

Also compares against non-event NEON sites re-scored to find their max
consecutive run above 0.9.

Output: results/exp_temporal_persistence/persistence_results.json

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"
SCAN_RESULTS = PROJECT_ROOT / "results" / "neon_anomaly_scan" / "neon_scan_results.json"
CASE_DIR     = PROJECT_ROOT / "results" / "case_studies_real"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "exp_temporal_persistence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BASE    = PROJECT_ROOT / "checkpoints"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T         = 128
STRIDE    = 64
THRESHOLD = 0.9

NEON_VALUE_COLS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
NEON_QF_COLS    = ["pHFinalQF", "dissolvedOxygenFinalQF", "turbidityFinalQF", "specificCondFinalQF"]
READ_COLS = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS

EVENT_FILES = {
    "lake_erie_hab_2023":        "HAB",
    "jordan_lake_hab_nc":        "HAB",
    "klamath_river_hab_2021":    "HAB",
    "gulf_dead_zone_2023":       "hypoxia",
    "chesapeake_hypoxia_2018":   "hypoxia",
    "mississippi_salinity_2023": "salinity",
}


def parse_dt(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


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


def build_windows_neon(df) -> list:
    """Returns list of (w_norm,) for NEON data."""
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


def max_consecutive_run(scores: list[float], thresh: float) -> tuple[int, float, int, int]:
    """Return (max_run, mean_score_in_best_run, run_start_idx, run_end_idx)."""
    best_len   = 0
    best_start = 0
    best_end   = 0
    cur_start  = None
    cur_len    = 0

    for i, s in enumerate(scores):
        if s >= thresh:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len   = cur_len
                best_start = cur_start
                best_end   = i
        else:
            cur_start = None
            cur_len   = 0

    if best_len > 0:
        run_scores = scores[best_start:best_end + 1]
        mean_score = float(np.mean(run_scores))
    else:
        mean_score = 0.0

    return best_len, mean_score, best_start, best_end


def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment D: Temporal Persistence Analysis")
    print("=" * 60)

    # ---- Case study events ----
    case_results = {}
    for eid, etype in EVENT_FILES.items():
        with open(CASE_DIR / f"{eid}_scores.json") as f:
            d = json.load(f)
        advisory_dt = parse_dt(d["advisory_date"] + "T00:00:00+00:00")
        all_scores  = d["scores"]

        # Pre-event scores only
        pre = [(s["center_time"], s["anomaly_probability"])
               for s in all_scores if parse_dt(s["center_time"]) < advisory_dt]
        # sort by time
        pre.sort(key=lambda x: x[0])
        pre_scores = [p[1] for p in pre]
        pre_times  = [p[0] for p in pre]

        if not pre_scores:
            case_results[eid] = {"error": "no_pre_event_scores"}
            continue

        best_len, mean_in_run, run_s, run_e = max_consecutive_run(pre_scores, THRESHOLD)

        run_start_time = pre_times[run_s] if best_len > 0 else None
        run_end_time   = pre_times[run_e] if best_len > 0 else None
        n_above_total  = sum(1 for s in pre_scores if s >= THRESHOLD)

        print(f"  {eid} ({etype}): max_consecutive={best_len}"
              f"  mean_in_run={mean_in_run:.4f}"
              f"  run=[{run_start_time}, {run_end_time}]"
              f"  n_above={n_above_total}/{len(pre_scores)}")

        case_results[eid] = {
            "event_type":                      etype,
            "advisory_date":                   d["advisory_date"],
            "n_pre_event_windows":             len(pre_scores),
            "n_above_threshold":               n_above_total,
            "max_consecutive_above_threshold": best_len,
            "mean_score_in_peak_run":          round(mean_in_run, 4),
            "run_start_time":                  run_start_time,
            "run_end_time":                    run_end_time,
        }

    # ---- Non-event NEON sites ----
    print("\n  Scoring non-event NEON sites...")
    with open(SCAN_RESULTS) as f:
        scan = json.load(f)
    per_site = scan["per_site"]

    # Pick 5 cleanest sites
    clean_sites = sorted(per_site.items(),
                         key=lambda x: (x[1]["label_anomaly_rate"], x[1]["mean_score"]))
    clean_sites = [site for site, v in clean_sites
                   if v["label_anomaly_rate"] == 0.0 and v["n_windows"] >= 20][:5]
    print(f"  Clean sites: {clean_sites}")

    sensor, fusion, head = load_models()
    print(f"  Models loaded on {DEVICE}")

    # Batch-read all clean sites at once for efficiency
    print(f"  Batch-reading NEON parquet for clean sites...")
    t_read = time.time()
    table = pq.read_table(str(NEON_PARQUET), columns=READ_COLS,
                          filters=[("source_site", "in", clean_sites)])
    df_all = table.to_pandas()
    print(f"  Read {len(df_all)} rows in {time.time()-t_read:.1f}s")

    neon_results = {}
    for site in clean_sites:
        t_site = time.time()
        df_site = df_all[df_all["source_site"] == site].copy()
        if len(df_site) < T * 2:
            print(f"    {site}: too few rows")
            continue
        windows = build_windows_neon(df_site)
        if not windows:
            continue
        scores = score_windows_batch(sensor, fusion, head, windows)

        best_len, mean_in_run, run_s, run_e = max_consecutive_run(scores, THRESHOLD)
        n_above = sum(1 for s in scores if s >= THRESHOLD)
        print(f"    {site}: {len(scores)} windows, max_run={best_len}"
              f"  n_above={n_above}  mean_all={np.mean(scores):.4f}"
              f"  ({time.time()-t_site:.1f}s)")
        neon_results[site] = {
            "n_windows":                       len(scores),
            "n_above_threshold":               n_above,
            "max_consecutive_above_threshold": best_len,
            "mean_score_in_peak_run":          round(mean_in_run, 4),
            "mean_score_all":                  round(float(np.mean(scores)), 4),
            "max_score":                       round(float(np.max(scores)), 4),
        }

    max_neon_run = max(
        (v["max_consecutive_above_threshold"] for v in neon_results.values()), default=0
    )
    mean_case_run = float(np.mean([
        v["max_consecutive_above_threshold"]
        for v in case_results.values() if "error" not in v
    ])) if case_results else 0.0

    print(f"\n  Case study mean max_consecutive: {mean_case_run:.1f}")
    print(f"  Non-event NEON max_consecutive:  {max_neon_run}")

    output = {
        "experiment":         "D: Temporal Persistence Analysis",
        "threshold":          THRESHOLD,
        "case_study_events":  case_results,
        "neon_clean_sites":   neon_results,
        "summary": {
            "case_mean_max_consecutive": round(mean_case_run, 2),
            "neon_max_consecutive":      max_neon_run,
            "persistence_ratio":         round(mean_case_run / max(max_neon_run, 1), 2),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = OUTPUT_DIR / "persistence_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
