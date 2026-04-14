#!/usr/bin/env python3
"""
exp1_case_studies_real.py — SENTINEL Real Case Study Experiment

Fetches real USGS NWIS continuous sensor data for 10 documented water-quality
events, runs AquaSSM inference (backbone + binary anomaly head from
checkpoints/sensor/aquassm_full_best.pt), and computes per-event detection
lead times vs. advisory dates.

This replaces exp1_case_studies_v3.py, which used hard-coded lead times.

Author: Bryan Cheng, SENTINEL project, 2026-04-14
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "results" / "case_studies_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = PROJECT_ROOT / "checkpoints" / "sensor" / "aquassm_full_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# USGS parameter codes (same as exp1_usgs_anomaly_detection.py)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_CODES = ["00300", "00400", "00095", "00010", "63680"]
PARAM_NAMES = ["DO", "pH", "SpCond", "Temp", "Turb"]

# ─────────────────────────────────────────────────────────────────────────────
# Events — 10 documented water-quality incidents with nearby USGS stations
# ─────────────────────────────────────────────────────────────────────────────
EVENTS = [
    # HAB events — detectable via DO depletion, pH spikes, phycocyanin surrogates
    {
        "event_id": "lake_erie_hab_2023",
        "name": "Lake Erie HAB 2023",
        "advisory_date": "2023-07-15",
        "lat": 41.50, "lon": -82.90,
        "usgs_site": "04199500",   # Sandusky River at Fremont OH
        "pre_event_days": 60,
        "post_event_days": 14,
    },
    {
        "event_id": "gulf_dead_zone_2023",
        "name": "Gulf Dead Zone 2023",
        "advisory_date": "2023-07-01",
        "lat": 29.50, "lon": -90.50,
        "usgs_site": "07374000",   # Mississippi at Baton Rouge
        "pre_event_days": 90,
        "post_event_days": 30,
    },
    {
        "event_id": "chesapeake_hypoxia_2018",
        "name": "Chesapeake Bay Hypoxia 2018",
        "advisory_date": "2018-07-20",
        "lat": 39.20, "lon": -76.50,
        "usgs_site": "01578310",   # Susquehanna at Conowingo
        "pre_event_days": 90,
        "post_event_days": 30,
    },
    {
        "event_id": "iowa_nitrate_2015",
        "name": "Iowa/Des Moines Nitrate Crisis 2015",
        "advisory_date": "2015-05-01",
        "lat": 41.60, "lon": -93.61,
        "usgs_site": "05482000",   # Raccoon River at Van Meter
        "pre_event_days": 60,
        "post_event_days": 14,
    },
    {
        "event_id": "neuse_river_hypoxia_2022",
        "name": "Neuse River Hypoxia 2022",
        "advisory_date": "2022-08-15",
        "lat": 35.10, "lon": -77.05,
        "usgs_site": "02089500",   # Neuse River at Kinston
        "pre_event_days": 60,
        "post_event_days": 30,
    },
    {
        "event_id": "klamath_river_hab_2021",
        "name": "Klamath River HAB 2021",
        "advisory_date": "2021-08-01",
        "lat": 41.55, "lon": -122.30,
        "usgs_site": "11530500",   # Klamath River near Seiad Valley
        "pre_event_days": 60,
        "post_event_days": 14,
    },
    {
        "event_id": "jordan_lake_hab_nc",
        "name": "Jordan Lake HAB NC",
        "advisory_date": "2022-07-15",
        "lat": 35.78, "lon": -79.06,
        "usgs_site": "02097517",   # New Hope Creek at Blands
        "pre_event_days": 45,
        "post_event_days": 14,
    },
    {
        "event_id": "mississippi_salinity_2023",
        "name": "Mississippi River Salinity Intrusion 2023",
        "advisory_date": "2023-10-01",
        "lat": 29.95, "lon": -90.06,
        "usgs_site": "07374000",   # Mississippi at Baton Rouge
        "pre_event_days": 60,
        "post_event_days": 30,
    },
    {
        "event_id": "dan_river_coal_ash_2014",
        "name": "Dan River Coal Ash Spill 2014",
        "advisory_date": "2014-02-02",
        "lat": 36.48, "lon": -79.50,
        "usgs_site": "02075500",   # Dan River at Paces
        "pre_event_days": 14,
        "post_event_days": 21,
    },
    {
        "event_id": "toledo_water_crisis_2014",
        "name": "Toledo Water Crisis 2014",
        "advisory_date": "2014-08-02",
        "lat": 41.66, "lon": -83.55,
        "usgs_site": "04197170",   # Maumee River at Waterville
        "pre_event_days": 14,
        "post_event_days": 7,
    },
]

# Detection thresholds to try
THRESHOLDS = [0.08, 0.10, 0.12]
PRIMARY_THRESHOLD = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyHead (mirrors train_aquassm_full.py)
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyHead(nn.Module):
    """Binary anomaly head: 256 → 128 → 64 → 1 (logit)."""

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
    """Load SensorEncoder backbone + AnomalyHead from aquassm_full_best.pt."""
    from sentinel.models.sensor_encoder.model import SensorEncoder

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)

    model = SensorEncoder(num_params=6, output_dim=256)
    model_state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"  SensorEncoder: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    model.to(DEVICE)

    head = AnomalyHead()
    head_state = ckpt.get("head", None)
    if head_state is not None:
        missing_h, unexpected_h = head.load_state_dict(head_state, strict=False)
        print(f"  AnomalyHead: {len(missing_h)} missing, {len(unexpected_h)} unexpected keys")
    else:
        print("  WARNING: no head state found in checkpoint — using random head")
    head.eval()
    head.to(DEVICE)

    print(f"  Checkpoint epoch={ckpt.get('epoch', '?')}, val_auroc={ckpt.get('val_auroc', '?'):.4f}")
    return model, head


# ─────────────────────────────────────────────────────────────────────────────
# USGS data download
# ─────────────────────────────────────────────────────────────────────────────
def fetch_usgs_iv(site_no: str, start: str, end: str):
    """Download USGS NWIS instantaneous-values data.

    Tries dataretrieval.nwis first; falls back to a raw URL query if that
    fails or returns empty.
    """
    import dataretrieval.nwis as nwis

    try:
        df, _ = nwis.get_iv(
            sites=site_no,
            parameterCd=PARAM_CODES,
            start=start,
            end=end,
        )
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        print(f"    dataretrieval get_iv failed: {e}")

    # Fallback: raw NWIS REST API
    try:
        import urllib.request
        codes_str = ",".join(PARAM_CODES)
        url = (
            f"https://nwis.waterservices.usgs.gov/nwis/iv/"
            f"?sites={site_no}&parameterCd={codes_str}"
            f"&startDT={start}&endDT={end}&format=rdb"
        )
        import io
        import pandas as pd
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")

        # Parse RDB format (skip # comment lines, then two header rows)
        lines = [l for l in text.splitlines() if not l.startswith("#")]
        if len(lines) < 3:
            return None
        # lines[0] = col names, lines[1] = data types
        col_names = lines[0].split("\t")
        data_lines = lines[2:]  # skip type row
        rows = [row.split("\t") for row in data_lines if row.strip()]
        df = pd.DataFrame(rows, columns=col_names)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df = df.dropna(subset=["datetime"]).set_index("datetime")
        return df if len(df) > 0 else None
    except Exception as e2:
        print(f"    Raw URL fallback also failed: {e2}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — identical logic to exp1_usgs_anomaly_detection.py
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_for_aquassm(df, seq_len: int = 128, stride: int = 64):
    """Convert NWIS dataframe to sliding-window tensors for AquaSSM.

    Model expects x of shape [B, T, P=6]: pH, DO, Turb, SpCond, Temp, ORP(zeros).
    """
    import pandas as pd

    # Extract per-parameter columns
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

    # Model order: pH, DO, Turb, SpCond, Temp, ORP(zeros)
    values_6 = np.column_stack([
        param_data["pH"],
        param_data["DO"],
        param_data["Turb"],
        param_data["SpCond"],
        param_data["Temp"],
        np.zeros(n_steps),          # ORP — not available
    ])

    # Validity mask: False where raw data was NaN (before zero-filling)
    masks = ~np.isnan(values_6)
    # NOTE: ORP (column 5) was set to zeros above, not NaN — its mask stays True
    # so the SSM sees "valid zeros" rather than zeroing out every timestep via the
    # min-across-params gating in AquaSSM.forward.  This matches how the model
    # was trained (ORP zero-padded as a valid zero input).
    values_6 = np.nan_to_num(values_6, nan=0.0)

    # Timestamps as unix seconds
    try:
        idx = df.index.get_level_values(-1) if hasattr(df.index, "levels") else df.index
        timestamps = idx.astype(np.int64) / 1e9
    except Exception:
        timestamps = np.arange(n_steps, dtype=np.float64) * 900.0

    # Z-score normalise with rough global WQ norms
    # Order: pH, DO, Turb, SpCond, Temp, ORP
    means = np.array([7.5, 8.0, 15.0, 500.0, 18.0, 200.0])
    stds  = np.array([1.0, 3.0, 20.0, 300.0,  8.0, 100.0])
    values_6 = (values_6 - means) / (stds + 1e-8)

    # Delta-t in seconds
    delta_ts = np.diff(timestamps, prepend=timestamps[0] - 900.0)
    delta_ts = np.clip(delta_ts, 1.0, 86400.0)

    windows = []
    for start_idx in range(0, n_steps - seq_len + 1, stride):
        end_idx = start_idx + seq_len
        w_ts   = timestamps[start_idx:end_idx]
        w_vals = values_6[start_idx:end_idx]
        w_dt   = delta_ts[start_idx:end_idx]
        w_mask = masks[start_idx:end_idx]

        # Require at least 30% valid across the 5 real params
        if w_mask[:, :5].mean() < 0.30:
            continue

        center_ts = float(w_ts[seq_len // 2])
        windows.append({
            # Model forward expects x=[B,T,P], delta_ts=[B,T], masks=[B,T,P]
            "x":       torch.tensor(w_vals, dtype=torch.float32).unsqueeze(0),
            "delta_ts": torch.tensor(w_dt, dtype=torch.float32).unsqueeze(0),
            "masks":   torch.tensor(w_mask, dtype=torch.float32).unsqueeze(0),
            "center_time": datetime.utcfromtimestamp(center_ts).isoformat(),
            "center_ts": center_ts,
        })

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, head, windows):
    """Run SensorEncoder + AnomalyHead on each window.

    SensorEncoder.forward(x, timestamps, delta_ts, masks, compute_anomaly=False)
    returns dict with 'embedding' [B, 256].  The AnomalyHead converts that to
    a scalar logit, then sigmoid → anomaly_probability.
    """
    results = []

    with torch.no_grad():
        for w in windows:
            x       = w["x"].to(DEVICE)
            dt      = w["delta_ts"].to(DEVICE)
            msk     = w["masks"].to(DEVICE)

            try:
                out = model(x=x, delta_ts=dt, masks=msk, compute_anomaly=False)
                embedding = out["embedding"]  # [1, 256]
            except Exception as e:
                # Fallback: try with timestamps arg or positional
                try:
                    B, T, P = x.shape
                    ts_dummy = torch.zeros(B, T, device=DEVICE)
                    out = model(x, ts_dummy, dt, msk, compute_anomaly=False)
                    embedding = out["embedding"]
                except Exception as e2:
                    print(f"      Inference error: {e2}")
                    continue

            # Ensure [1, 256]
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

            logit = head(embedding)                    # [1] or scalar
            prob  = torch.sigmoid(logit).item()

            results.append({
                "center_time": w["center_time"],
                "center_ts":   w["center_ts"],
                "anomaly_probability": float(prob),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-event analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_event(event: dict, scores: list[dict], threshold: float) -> dict:
    """Compute detection time and lead time for a single event."""
    advisory_dt = datetime.strptime(event["advisory_date"], "%Y-%m-%d")
    advisory_ts = advisory_dt.timestamp()

    # Scores before advisory
    pre_scores = [s for s in scores if s["center_ts"] < advisory_ts]
    all_probs  = [s["anomaly_probability"] for s in scores]
    pre_probs  = [s["anomaly_probability"] for s in pre_scores]

    max_prob = float(max(all_probs)) if all_probs else 0.0

    # Find FIRST pre-advisory window exceeding threshold
    first_detection = None
    for s in pre_scores:  # already time-ordered
        if s["anomaly_probability"] > threshold:
            first_detection = s
            break

    if first_detection is not None:
        lead_secs = advisory_ts - first_detection["center_ts"]
        lead_hours = lead_secs / 3600.0
        return {
            "status": "detected",
            "first_detection_time": first_detection["center_time"],
            "lead_time_hours": round(lead_hours, 2),
            "lead_time_days":  round(lead_hours / 24.0, 2),
            "max_anomaly_prob": round(max_prob, 4),
            "n_pre_event_windows": len(pre_scores),
            "anomaly_scores_pre_event": [
                round(p, 4) for p in pre_probs[-20:]
            ],
        }
    else:
        return {
            "status": "not_detected",
            "first_detection_time": None,
            "lead_time_hours": None,
            "lead_time_days": None,
            "max_anomaly_prob": round(max_prob, 4),
            "n_pre_event_windows": len(pre_scores),
            "anomaly_scores_pre_event": [
                round(p, 4) for p in pre_probs[-20:]
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SENTINEL Real Case Study Experiment")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nLoading models...")
    model, head = load_models()

    event_results = []
    n_data = 0
    n_detected = 0

    for event in EVENTS:
        print(f"\n{'─' * 60}")
        print(f"Event: {event['name']} ({event['advisory_date']})")
        print(f"USGS site: {event['usgs_site']}")

        advisory_dt = datetime.strptime(event["advisory_date"], "%Y-%m-%d")
        start_dt    = advisory_dt - timedelta(days=event["pre_event_days"])
        end_dt      = advisory_dt + timedelta(days=event["post_event_days"])
        start_str   = start_dt.strftime("%Y-%m-%d")
        end_str     = end_dt.strftime("%Y-%m-%d")

        print(f"  Fetching {start_str} → {end_str}...")

        # Primary site
        df = None
        site_used = event["usgs_site"]
        try:
            df = fetch_usgs_iv(event["usgs_site"], start_str, end_str)
        except Exception as e:
            print(f"  Fetch exception: {e}")

        # If primary failed, try nearby sites via get_info
        if df is None or len(df) == 0:
            print(f"  Primary site returned no data — searching nearby stations...")
            df, site_used = try_nearby_stations(
                event["lat"], event["lon"], start_str, end_str, radius_km=50
            )

        if df is None or len(df) == 0:
            print(f"  No data for {event['name']} — skipping")
            event_results.append({
                "event_id": event["event_id"],
                "name": event["name"],
                "advisory_date": event["advisory_date"],
                "usgs_site": site_used,
                "status": "no_data",
                "n_windows": 0,
                "max_anomaly_prob": None,
                "first_detection_time": None,
                "lead_time_hours": None,
                "lead_time_days": None,
                "anomaly_scores_pre_event": [],
            })
            time.sleep(0.5)
            continue

        n_data += 1
        print(f"  Records: {len(df)}  site_used={site_used}")

        # Preprocess
        windows = preprocess_for_aquassm(df, seq_len=128, stride=64)
        print(f"  Windows: {len(windows)}")

        if len(windows) == 0:
            event_results.append({
                "event_id": event["event_id"],
                "name": event["name"],
                "advisory_date": event["advisory_date"],
                "usgs_site": site_used,
                "status": "insufficient_data",
                "n_windows": 0,
                "n_records": len(df),
                "max_anomaly_prob": None,
                "first_detection_time": None,
                "lead_time_hours": None,
                "lead_time_days": None,
                "anomaly_scores_pre_event": [],
            })
            time.sleep(0.5)
            continue

        # Inference
        print(f"  Running AquaSSM inference on {len(windows)} windows...")
        scores = run_inference(model, head, windows)
        print(f"  Scores computed: {len(scores)}")

        if len(scores) == 0:
            event_results.append({
                "event_id": event["event_id"],
                "name": event["name"],
                "advisory_date": event["advisory_date"],
                "usgs_site": site_used,
                "status": "inference_failed",
                "n_windows": len(windows),
                "max_anomaly_prob": None,
                "first_detection_time": None,
                "lead_time_hours": None,
                "lead_time_days": None,
                "anomaly_scores_pre_event": [],
            })
            time.sleep(0.5)
            continue

        # Analyse at primary threshold
        analysis = analyse_event(event, scores, threshold=PRIMARY_THRESHOLD)

        row = {
            "event_id": event["event_id"],
            "name": event["name"],
            "advisory_date": event["advisory_date"],
            "usgs_site": site_used,
            "n_records": len(df),
            "n_windows": len(scores),
            **analysis,
        }

        # Also compute detections at all thresholds for reference
        thresh_results = {}
        for t in THRESHOLDS:
            a = analyse_event(event, scores, threshold=t)
            thresh_results[f"thresh_{str(t).replace('.', '_')}"] = {
                "status": a["status"],
                "lead_time_hours": a["lead_time_hours"],
            }
        row["threshold_sweep"] = thresh_results

        if analysis["status"] == "detected":
            n_detected += 1
            print(
                f"  DETECTED: lead={analysis['lead_time_hours']:.1f}h "
                f"({analysis['lead_time_days']:.1f} days), "
                f"max_prob={analysis['max_anomaly_prob']:.4f}"
            )
        else:
            print(
                f"  NOT DETECTED (threshold={PRIMARY_THRESHOLD}), "
                f"max_prob={analysis['max_anomaly_prob']:.4f}"
            )

        event_results.append(row)

        # Save individual scores file
        scores_path = OUTPUT_DIR / f"{event['event_id']}_scores.json"
        with open(scores_path, "w") as f:
            json.dump({
                "event_id": event["event_id"],
                "advisory_date": event["advisory_date"],
                "usgs_site": site_used,
                "scores": scores,
            }, f, indent=2, default=str)

        time.sleep(0.5)   # rate-limit USGS API

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────
    detected_leads = [
        r["lead_time_hours"]
        for r in event_results
        if r.get("status") == "detected" and r["lead_time_hours"] is not None
    ]

    stats = {
        "mean_lead_time_hours":   round(float(np.mean(detected_leads)), 2) if detected_leads else None,
        "median_lead_time_hours": round(float(np.median(detected_leads)), 2) if detected_leads else None,
        "min_lead_time_hours":    round(float(np.min(detected_leads)), 2) if detected_leads else None,
        "max_lead_time_hours":    round(float(np.max(detected_leads)), 2) if detected_leads else None,
        "detection_rate":         round(n_detected / len(EVENTS), 4) if EVENTS else 0.0,
        "n_events_with_data":     n_data,
        "n_events_detected":      n_detected,
        "n_events_attempted":     len(EVENTS),
    }

    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "checkpoint": str(CKPT_PATH),
        "n_events_attempted": len(EVENTS),
        "n_events_with_data": n_data,
        "n_events_detected": n_detected,
        "detection_threshold": PRIMARY_THRESHOLD,
        "events": event_results,
        "statistics": stats,
    }

    out_path = OUTPUT_DIR / "case_studies_real.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary printout
    # ─────────────────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("=== SENTINEL Real Case Study Results ===")
    print("=" * 60)
    print(f"Events attempted:                  {len(EVENTS)}")
    print(f"Events with real USGS data:        {n_data}")
    print(f"Events detected (threshold={PRIMARY_THRESHOLD}): {n_detected}")
    rate_pct = 100.0 * n_detected / n_data if n_data > 0 else 0.0
    print(f"Detection rate (of events w/data): {rate_pct:.1f}%")
    if detected_leads:
        print(f"Mean lead time:  {stats['mean_lead_time_hours']:.1f} h  "
              f"({stats['mean_lead_time_hours']/24:.1f} days)")
        print(f"Median lead time:{stats['median_lead_time_hours']:.1f} h  "
              f"({stats['median_lead_time_hours']/24:.1f} days)")
    print()
    print("Event results:")
    for r in event_results:
        eid    = r["event_id"]
        status = r.get("status", "?")
        mp     = r.get("max_anomaly_prob")
        mp_str = f"{mp:.4f}" if mp is not None else "N/A"
        lt     = r.get("lead_time_hours")
        if status == "detected" and lt is not None:
            print(f"  {eid:40s}: DETECTED,     lead={lt:.1f}h ({lt/24:.1f}d), max_prob={mp_str}")
        elif status == "not_detected":
            print(f"  {eid:40s}: NOT_DETECTED, max_prob={mp_str}")
        else:
            print(f"  {eid:40s}: {status.upper()}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: search nearby USGS stations when primary site has no data
# ─────────────────────────────────────────────────────────────────────────────
def try_nearby_stations(lat: float, lon: float,
                        start: str, end: str,
                        radius_km: float = 50.0):
    """Try to find any nearby USGS site with IV data via bounding-box search."""
    import math
    import dataretrieval.nwis as nwis

    # Convert radius to rough degree offset
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))

    bbox_str = (
        f"{lon - deg_lon:.4f},{lat - deg_lat:.4f},"
        f"{lon + deg_lon:.4f},{lat + deg_lat:.4f}"
    )

    try:
        site_info, _ = nwis.get_info(
            bBox=bbox_str,
            parameterCd=PARAM_CODES,
            siteType="ST",
            hasDataTypeCd="iv",
        )
    except Exception as e:
        print(f"    get_info bBox search failed: {e}")
        return None, "unknown"

    if site_info is None or len(site_info) == 0:
        print("    No nearby stations found in bounding box")
        return None, "unknown"

    # Sort by distance
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    candidates = []
    for _, row in site_info.iterrows():
        try:
            s_lat = float(row.get("dec_lat_va", 0))
            s_lon = float(row.get("dec_long_va", 0))
            if s_lat == 0 or s_lon == 0:
                continue
            dist = haversine_km(lat, lon, s_lat, s_lon)
            candidates.append((dist, str(row["site_no"]).strip()))
        except Exception:
            continue

    candidates.sort()
    print(f"    Found {len(candidates)} candidate sites within {radius_km} km")

    for dist, site_no in candidates[:5]:
        print(f"    Trying site {site_no} ({dist:.1f} km)...")
        df = fetch_usgs_iv(site_no, start, end)
        time.sleep(0.5)
        if df is not None and len(df) > 0:
            print(f"    Got {len(df)} records from {site_no}")
            return df, site_no

    return None, "none_found"


if __name__ == "__main__":
    main()
