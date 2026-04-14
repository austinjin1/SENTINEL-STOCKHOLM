#!/usr/bin/env python3
"""
exp_discovery_scan.py — SENTINEL Discovery Scan on Current Data

Scans recent (2025-2026) USGS NWIS data across 18 major water bodies to
identify potential current water quality crises using AquaSSM.

Per-site reporting:
  - max/mean anomaly probability
  - Number of windows > 0.9 (high alert)
  - Time periods with sustained elevated scores
  - Most recent 30-day max score
  - Alert flag if max_score > 0.9 or recent_30d_max > 0.8

Author: Bryan Cheng, SENTINEL project, 2026-04-14
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "results" / "discovery_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_CKPT = PROJECT_ROOT / "checkpoints" / "sensor" / "aquassm_full_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# USGS parameter codes
# ─────────────────────────────────────────────────────────────────────────────
PARAM_CODES = ["00300", "00400", "00095", "00010", "63680"]
PARAM_NAMES = ["DO", "pH", "SpCond", "Temp", "Turb"]

# Scan date range: Jan 2025 – April 2026
SCAN_START = "2025-01-01"
SCAN_END   = "2026-04-14"

ALERT_MAX_THRESHOLD     = 0.9
ALERT_RECENT_THRESHOLD  = 0.8
HIGH_ALERT_THRESHOLD    = 0.9
SUSTAINED_WINDOW_N      = 3    # consecutive windows above 0.5 = sustained

# ─────────────────────────────────────────────────────────────────────────────
# Discovery sites
# ─────────────────────────────────────────────────────────────────────────────
DISCOVERY_SITES = [
    # Florida
    {"name": "Lake Okeechobee",             "site": "02276900", "lat": 26.97, "lon": -80.82},
    {"name": "St. Johns River FL",          "site": "02231000", "lat": 30.19, "lon": -81.78},
    # Midwest
    {"name": "Raccoon River IA (nitrate)",  "site": "05482000", "lat": 41.60, "lon": -93.61},
    {"name": "Des Moines River",            "site": "05481000", "lat": 41.59, "lon": -93.61},
    {"name": "Illinois River",              "site": "05586100", "lat": 38.97, "lon": -90.44},
    {"name": "Maumee River OH",             "site": "04193500", "lat": 41.44, "lon": -84.08},
    # Gulf Coast
    {"name": "Mississippi River at Baton Rouge", "site": "07374000", "lat": 30.44, "lon": -91.19},
    {"name": "Atchafalaya River",           "site": "07381490", "lat": 29.69, "lon": -91.25},
    # Southeast
    {"name": "Neuse River NC",              "site": "02089500", "lat": 35.10, "lon": -77.05},
    {"name": "Cape Fear River NC",          "site": "02101726", "lat": 35.78, "lon": -79.06},
    # Great Lakes
    {"name": "Sandusky River OH (Lake Erie tributary)", "site": "04199500", "lat": 41.32, "lon": -83.12},
    {"name": "Maumee River (Toledo area)",  "site": "04197170", "lat": 41.57, "lon": -83.65},
    # West
    {"name": "Klamath River CA",            "site": "11530500", "lat": 41.55, "lon": -122.30},
    {"name": "Sacramento River CA",         "site": "11447650", "lat": 38.26, "lon": -121.97},
    # Mid-Atlantic
    {"name": "Patuxent River MD",           "site": "01589485", "lat": 39.09, "lon": -76.74},
    {"name": "Potomac River DC",            "site": "01646500", "lat": 39.00, "lon": -77.15},
    # Pacific Northwest
    {"name": "Willamette River OR",         "site": "14211720", "lat": 45.52, "lon": -122.67},
    # Texas
    {"name": "San Antonio River TX",        "site": "08178800", "lat": 29.50, "lon": -98.43},
]


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyHead (matches aquassm_full_best.pt)
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    from sentinel.models.sensor_encoder.model import SensorEncoder

    ckpt = torch.load(str(SENSOR_CKPT), map_location="cpu", weights_only=False)
    model = SensorEncoder(num_params=6, output_dim=256)
    model_state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"  SensorEncoder: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval().to(DEVICE)

    head = AnomalyHead()
    head_state = ckpt.get("head", None)
    if head_state is not None:
        head.load_state_dict(head_state, strict=False)
    else:
        print("  WARNING: no head state in checkpoint — using random head")
    head.eval().to(DEVICE)

    val_auroc = ckpt.get("val_auroc", "N/A")
    print(f"  Checkpoint epoch={ckpt.get('epoch','?')}, val_auroc={val_auroc}")
    return model, head


# ─────────────────────────────────────────────────────────────────────────────
# USGS data fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_usgs_iv(site_no: str, start: str, end: str):
    import dataretrieval.nwis as nwis
    try:
        df, _ = nwis.get_iv(sites=site_no, parameterCd=PARAM_CODES, start=start, end=end)
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        print(f"    dataretrieval failed: {e}")

    try:
        import urllib.request
        import pandas as pd
        codes_str = ",".join(PARAM_CODES)
        url = (f"https://nwis.waterservices.usgs.gov/nwis/iv/"
               f"?sites={site_no}&parameterCd={codes_str}"
               f"&startDT={start}&endDT={end}&format=rdb")
        with urllib.request.urlopen(url, timeout=45) as resp:
            text = resp.read().decode("utf-8")
        lines = [l for l in text.splitlines() if not l.startswith("#")]
        if len(lines) < 3:
            return None
        col_names = lines[0].split("\t")
        data_lines = lines[2:]
        rows = [row.split("\t") for row in data_lines if row.strip()]
        df = pd.DataFrame(rows, columns=col_names)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df = df.dropna(subset=["datetime"]).set_index("datetime")
        return df if len(df) > 0 else None
    except Exception as e2:
        print(f"    Raw URL fallback failed: {e2}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_for_aquassm(df, seq_len: int = 128, stride: int = 64):
    import pandas as pd

    param_data = {}
    for code, name in zip(PARAM_CODES, PARAM_NAMES):
        col = None
        for c in df.columns:
            if c == code or (c.startswith(code) and not c.endswith("_cd")):
                col = c
                break
        if col is not None and col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
        else:
            vals = np.full(len(df), np.nan)
        param_data[name] = vals

    n_steps = len(df)
    if n_steps < seq_len:
        return []

    values_6 = np.column_stack([
        param_data["pH"],
        param_data["DO"],
        param_data["Turb"],
        param_data["SpCond"],
        param_data["Temp"],
        np.zeros(n_steps),
    ])
    masks = ~np.isnan(values_6)
    values_6 = np.nan_to_num(values_6, nan=0.0)

    try:
        idx = df.index.get_level_values(-1) if hasattr(df.index, "levels") else df.index
        timestamps = idx.astype(np.int64) / 1e9
    except Exception:
        timestamps = np.arange(n_steps, dtype=np.float64) * 900.0

    means = np.array([7.5, 8.0, 15.0, 500.0, 18.0, 200.0])
    stds  = np.array([1.0, 3.0, 20.0, 300.0,  8.0, 100.0])
    values_6 = (values_6 - means) / (stds + 1e-8)

    delta_ts = np.diff(timestamps, prepend=timestamps[0] - 900.0)
    delta_ts = np.clip(delta_ts, 1.0, 86400.0)

    windows = []
    for start_idx in range(0, n_steps - seq_len + 1, stride):
        end_idx = start_idx + seq_len
        w_ts   = timestamps[start_idx:end_idx]
        w_vals = values_6[start_idx:end_idx]
        w_dt   = delta_ts[start_idx:end_idx]
        w_mask = masks[start_idx:end_idx]

        if w_mask[:, :5].mean() < 0.30:
            continue

        center_ts = float(w_ts[seq_len // 2])
        windows.append({
            "x":        torch.tensor(w_vals, dtype=torch.float32).unsqueeze(0),
            "delta_ts": torch.tensor(w_dt,   dtype=torch.float32).unsqueeze(0),
            "masks":    torch.tensor(w_mask,  dtype=torch.float32).unsqueeze(0),
            "center_time": datetime.utcfromtimestamp(center_ts).isoformat(),
            "center_ts": center_ts,
        })

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# AquaSSM inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, head, windows):
    results = []
    with torch.no_grad():
        for w in windows:
            x   = w["x"].to(DEVICE)
            dt  = w["delta_ts"].to(DEVICE)
            msk = w["masks"].to(DEVICE)
            try:
                out = model(x=x, delta_ts=dt, masks=msk, compute_anomaly=False)
                embedding = out["embedding"]
            except Exception:
                try:
                    B, T, P = x.shape
                    ts_dummy = torch.zeros(B, T, device=DEVICE)
                    out = model(x, ts_dummy, dt, msk, compute_anomaly=False)
                    embedding = out["embedding"]
                except Exception as e2:
                    continue

            if isinstance(embedding, dict):
                embedding = embedding.get("embedding", next(iter(embedding.values())))
            if embedding.dim() == 3:
                embedding = embedding.mean(dim=1)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            if embedding.shape[-1] != 256:
                if embedding.shape[-1] > 256:
                    embedding = embedding[:, :256]
                else:
                    pad = torch.zeros(1, 256 - embedding.shape[-1], device=DEVICE)
                    embedding = torch.cat([embedding, pad], dim=-1)

            logit = head(embedding)
            prob  = torch.sigmoid(logit).item()

            results.append({
                "center_time": w["center_time"],
                "center_ts":   w["center_ts"],
                "anomaly_probability": float(prob),
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Site analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_site(site_info: dict, scores: List[dict]) -> Dict[str, Any]:
    """Compute per-site statistics from AquaSSM scores."""
    if not scores:
        return {
            "name": site_info["name"],
            "site": site_info["site"],
            "lat": site_info["lat"],
            "lon": site_info["lon"],
            "status": "no_scores",
            "max_score": None,
            "mean_score": None,
            "n_windows_above_09": 0,
            "recent_30d_max": None,
            "alert": False,
            "sustained_periods": [],
        }

    probs = [s["anomaly_probability"] for s in scores]
    max_score  = float(max(probs))
    mean_score = float(np.mean(probs))

    n_high = sum(1 for p in probs if p > HIGH_ALERT_THRESHOLD)

    # Recent 30-day max
    cutoff_ts = (datetime.utcnow() - timedelta(days=30)).timestamp()
    recent_probs = [s["anomaly_probability"] for s in scores if s["center_ts"] >= cutoff_ts]
    recent_30d_max = float(max(recent_probs)) if recent_probs else None

    # Sustained elevated periods: find runs of >= SUSTAINED_WINDOW_N windows with prob > 0.5
    SUSTAINED_THRESHOLD = 0.5
    sustained_periods = []
    run_start = None
    run_count = 0
    run_max   = 0.0
    for s in scores:
        p = s["anomaly_probability"]
        if p > SUSTAINED_THRESHOLD:
            if run_start is None:
                run_start = s["center_time"]
                run_count = 1
                run_max = p
            else:
                run_count += 1
                run_max = max(run_max, p)
        else:
            if run_start is not None and run_count >= SUSTAINED_WINDOW_N:
                sustained_periods.append({
                    "start": run_start,
                    "end": scores[scores.index(s) - 1]["center_time"] if scores.index(s) > 0 else run_start,
                    "n_windows": run_count,
                    "max_prob": round(run_max, 4),
                })
            run_start = None
            run_count = 0
            run_max   = 0.0
    # Close open run
    if run_start is not None and run_count >= SUSTAINED_WINDOW_N:
        sustained_periods.append({
            "start": run_start,
            "end": scores[-1]["center_time"],
            "n_windows": run_count,
            "max_prob": round(run_max, 4),
        })

    alert = (max_score > ALERT_MAX_THRESHOLD or
             (recent_30d_max is not None and recent_30d_max > ALERT_RECENT_THRESHOLD))

    return {
        "name": site_info["name"],
        "site": site_info["site"],
        "lat": site_info["lat"],
        "lon": site_info["lon"],
        "status": "ok",
        "n_windows": len(scores),
        "max_score":  round(max_score, 4),
        "mean_score": round(mean_score, 4),
        "n_windows_above_09": n_high,
        "recent_30d_max": round(recent_30d_max, 4) if recent_30d_max is not None else None,
        "alert": alert,
        "sustained_periods": sustained_periods,
        # Top 20 highest-scoring windows for reference
        "top_scores": sorted(
            [{"center_time": s["center_time"],
              "anomaly_probability": round(s["anomaly_probability"], 4)}
             for s in scores if s["anomaly_probability"] > 0.5],
            key=lambda x: x["anomaly_probability"], reverse=True
        )[:20],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SENTINEL Discovery Scan — Current Water Quality Crisis Detection")
    print(f"  Checkpoint : {SENSOR_CKPT}")
    print(f"  Device     : {DEVICE}")
    print(f"  Date range : {SCAN_START} → {SCAN_END}")
    print(f"  Sites      : {len(DISCOVERY_SITES)}")
    print("=" * 70)

    print("\nLoading AquaSSM model...")
    model, head = load_models()

    site_results = []
    n_with_data  = 0
    n_alerts     = 0

    for site_info in DISCOVERY_SITES:
        name     = site_info["name"]
        site_no  = site_info["site"]
        print(f"\n{'─'*60}")
        print(f"Site: {name} ({site_no})")

        df = None
        try:
            print(f"  Fetching USGS IV {SCAN_START} → {SCAN_END}...")
            df = fetch_usgs_iv(site_no, SCAN_START, SCAN_END)
        except Exception as e:
            print(f"  Fetch exception: {e}")

        if df is None or len(df) == 0:
            print(f"  No data returned")
            site_results.append({
                "name": name,
                "site": site_no,
                "lat": site_info["lat"],
                "lon": site_info["lon"],
                "status": "no_data",
                "max_score": None,
                "mean_score": None,
                "n_windows_above_09": 0,
                "recent_30d_max": None,
                "alert": False,
                "sustained_periods": [],
            })
            time.sleep(0.3)
            continue

        n_with_data += 1
        print(f"  Records: {len(df)}")

        windows = preprocess_for_aquassm(df, seq_len=128, stride=64)
        print(f"  Windows: {len(windows)}")

        if not windows:
            site_results.append({
                "name": name,
                "site": site_no,
                "lat": site_info["lat"],
                "lon": site_info["lon"],
                "status": "insufficient_data",
                "n_records": len(df),
                "max_score": None,
                "mean_score": None,
                "n_windows_above_09": 0,
                "recent_30d_max": None,
                "alert": False,
                "sustained_periods": [],
            })
            time.sleep(0.3)
            continue

        scores = run_inference(model, head, windows)
        print(f"  Scores: {len(scores)}")

        analysis = analyse_site(site_info, scores)

        max_s = analysis.get("max_score")
        r30   = analysis.get("recent_30d_max")
        is_alert = analysis.get("alert", False)

        print(f"  max_score={max_s}  mean_score={analysis.get('mean_score')}  "
              f"n>0.9={analysis['n_windows_above_09']}  "
              f"recent_30d_max={r30}  alert={is_alert}")

        if is_alert:
            n_alerts += 1
            print(f"  *** ALERT ***")

        site_results.append(analysis)
        time.sleep(0.3)

    # Sort by max_score descending
    def sort_key(r):
        ms = r.get("max_score")
        return ms if ms is not None else -1.0

    site_results_sorted = sorted(site_results, key=sort_key, reverse=True)

    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "checkpoint": str(SENSOR_CKPT),
        "scan_start": SCAN_START,
        "scan_end":   SCAN_END,
        "n_sites_attempted": len(DISCOVERY_SITES),
        "n_sites_with_data": n_with_data,
        "n_alerts": n_alerts,
        "alert_criteria": {
            "max_score_threshold": ALERT_MAX_THRESHOLD,
            "recent_30d_max_threshold": ALERT_RECENT_THRESHOLD,
        },
        "sites_ranked": site_results_sorted,
    }

    out_path = OUTPUT_DIR / "discovery_scan_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("=== SENTINEL Discovery Scan Results ===")
    print("=" * 70)
    print(f"Sites attempted : {len(DISCOVERY_SITES)}")
    print(f"Sites with data : {n_with_data}")
    print(f"Sites on ALERT  : {n_alerts}")
    print()
    print("Top 10 sites by max anomaly score:")
    print(f"  {'Rank':<5} {'Name':<40} {'max':>6} {'mean':>6} {'n>0.9':>6} {'r30d':>6} {'alert':>6}")
    print(f"  {'─'*75}")
    for i, r in enumerate(site_results_sorted[:10]):
        ms  = f"{r['max_score']:.4f}" if r.get("max_score") is not None else "N/A"
        mn  = f"{r['mean_score']:.4f}" if r.get("mean_score") is not None else "N/A"
        n09 = str(r.get("n_windows_above_09", 0))
        r30 = f"{r['recent_30d_max']:.4f}" if r.get("recent_30d_max") is not None else "N/A"
        alrt = "ALERT" if r.get("alert") else "-"
        print(f"  {i+1:<5} {r['name']:<40} {ms:>6} {mn:>6} {n09:>6} {r30:>6} {alrt:>6}")

    print()
    alert_sites = [r["name"] for r in site_results_sorted if r.get("alert")]
    if alert_sites:
        print("ALERT sites:")
        for s in alert_sites:
            print(f"  - {s}")
    else:
        print("No sites triggered ALERT criteria.")
    print("=" * 70)


if __name__ == "__main__":
    main()
