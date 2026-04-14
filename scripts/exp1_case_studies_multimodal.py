#!/usr/bin/env python3
"""
exp1_case_studies_multimodal.py — SENTINEL Multimodal Case Study Experiment

Re-runs the 6 confirmed case study events using ALL available modalities:
  - AquaSSM (sensor): USGS NWIS data, sliding windows T=128
  - HydroViT (satellite): Microsoft Planetary Computer Sentinel-2 L2A tiles
  - Fusion: PerceiverIOFusion combining sensor + satellite

No microbial/molecular/behavioral data for field historical events — expected.

Author: Bryan Cheng, SENTINEL project, 2026-04-14
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "results" / "case_studies_multimodal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_CKPT   = PROJECT_ROOT / "checkpoints" / "sensor" / "aquassm_full_best.pt"
SAT_CKPT      = PROJECT_ROOT / "checkpoints" / "satellite" / "hydrovit_wq_v9.pt"
FUSION_CKPT   = PROJECT_ROOT / "checkpoints" / "fusion" / "fusion_real_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# USGS parameters
# ─────────────────────────────────────────────────────────────────────────────
PARAM_CODES = ["00300", "00400", "00095", "00010", "63680"]
PARAM_NAMES = ["DO", "pH", "SpCond", "Temp", "Turb"]

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PATCH_SIZE = 224
MAX_CLOUD = 40

# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────
EVENTS = [
    {"event_id": "lake_erie_hab_2023",       "name": "Lake Erie HAB 2023",
     "advisory_date": "2023-07-15", "lat": 41.50, "lon": -82.90, "usgs_site": "04199500"},
    {"event_id": "gulf_dead_zone_2023",      "name": "Gulf Dead Zone 2023",
     "advisory_date": "2023-07-01", "lat": 29.50, "lon": -90.50, "usgs_site": "07374000"},
    {"event_id": "chesapeake_hypoxia_2018",  "name": "Chesapeake Bay Hypoxia 2018",
     "advisory_date": "2018-07-20", "lat": 39.20, "lon": -76.50, "usgs_site": "01589485"},
    {"event_id": "klamath_river_hab_2021",   "name": "Klamath River HAB 2021",
     "advisory_date": "2021-08-01", "lat": 41.55, "lon": -122.30, "usgs_site": "11530500"},
    {"event_id": "jordan_lake_hab_nc",       "name": "Jordan Lake HAB NC",
     "advisory_date": "2022-07-15", "lat": 35.78, "lon": -79.06, "usgs_site": "02097517"},
    {"event_id": "mississippi_salinity_2023","name": "Mississippi Salinity Intrusion 2023",
     "advisory_date": "2023-10-01", "lat": 29.95, "lon": -90.06, "usgs_site": "07374000"},
]

DETECTION_THRESHOLD = 0.10
FUSION_WINDOW_H     = 12.0   # ±12h matching window for sensor/satellite


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyHead (mirrors aquassm_full_best.pt)
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
def load_sensor_models():
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
        print("  AnomalyHead (sensor): loaded from checkpoint")
    else:
        print("  WARNING: no head state in sensor checkpoint — random head")
    head.eval().to(DEVICE)

    print(f"  Sensor ckpt: epoch={ckpt.get('epoch','?')}, val_auroc={ckpt.get('val_auroc','N/A')}")
    return model, head


def load_satellite_model():
    from sentinel.models.satellite_encoder.model import SatelliteEncoder
    sat_model = SatelliteEncoder(in_chans=13, shared_embed_dim=256)
    ckpt = torch.load(str(SAT_CKPT), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = sat_model.load_state_dict(state, strict=False)
    print(f"  SatelliteEncoder: {len(missing)} missing, {len(unexpected)} unexpected keys")
    sat_model.eval().to(DEVICE)
    return sat_model


def load_fusion_models():
    from sentinel.models.fusion.model import PerceiverIOFusion
    from sentinel.models.fusion.heads import AnomalyDetectionHead

    fusion_ckpt = torch.load(str(FUSION_CKPT), map_location="cpu", weights_only=False)

    fusion = PerceiverIOFusion(num_latents=64)
    fusion.load_state_dict(fusion_ckpt["fusion"], strict=False)
    fusion.eval().to(DEVICE)

    head = AnomalyDetectionHead()
    head.load_state_dict(fusion_ckpt["head"], strict=False)
    head.eval().to(DEVICE)

    print("  Fusion (PerceiverIOFusion + AnomalyDetectionHead): loaded")
    return fusion, head


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
        import io
        import pandas as pd
        codes_str = ",".join(PARAM_CODES)
        url = (f"https://nwis.waterservices.usgs.gov/nwis/iv/"
               f"?sites={site_no}&parameterCd={codes_str}"
               f"&startDT={start}&endDT={end}&format=rdb")
        with urllib.request.urlopen(url, timeout=30) as resp:
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
            "x":       torch.tensor(w_vals, dtype=torch.float32).unsqueeze(0),
            "delta_ts": torch.tensor(w_dt, dtype=torch.float32).unsqueeze(0),
            "masks":   torch.tensor(w_mask, dtype=torch.float32).unsqueeze(0),
            "center_time": datetime.utcfromtimestamp(center_ts).isoformat(),
            "center_ts": center_ts,
        })

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Sensor inference
# ─────────────────────────────────────────────────────────────────────────────
def run_sensor_inference(model, head, windows):
    results = []
    with torch.no_grad():
        for w in windows:
            x  = w["x"].to(DEVICE)
            dt = w["delta_ts"].to(DEVICE)
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
                    print(f"      Sensor inference error: {e2}")
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
                "embedding": embedding.cpu(),
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Satellite: Planetary Computer query
# ─────────────────────────────────────────────────────────────────────────────
def query_s2_tiles(lat: float, lon: float, start_date: str, end_date: str):
    """Return list of (tile_date_str, tensor [13,224,224]) for S2 tiles found."""
    tiles = []
    try:
        import pystac_client
        import planetary_computer
        import rasterio

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        bbox = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        )
        items = list(search.items())
        print(f"    S2: found {len(items)} tiles")

        if not items:
            return tiles

        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        # Take up to 6 tiles spread across the date range
        step = max(1, len(items) // 6)
        selected = items[::step][:6]

        for item in selected:
            try:
                tile_date = item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"
                signed_item = planetary_computer.sign(item)
                band_arrays = []
                ok = True
                for band_name in S2_BANDS:
                    if band_name not in signed_item.assets:
                        band_arrays.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
                        continue
                    href = signed_item.assets[band_name].href
                    with rasterio.open(href) as src:
                        from rasterio.transform import rowcol
                        row, col = rowcol(src.transform, lon, lat)
                        half = PATCH_SIZE // 2
                        row_start = max(0, int(row) - half)
                        col_start = max(0, int(col) - half)
                        row_end = row_start + PATCH_SIZE
                        col_end = col_start + PATCH_SIZE
                        if row_end > src.height:
                            row_start = max(0, src.height - PATCH_SIZE)
                            row_end = row_start + PATCH_SIZE
                        if col_end > src.width:
                            col_start = max(0, src.width - PATCH_SIZE)
                            col_end = col_start + PATCH_SIZE
                        window = rasterio.windows.Window(
                            col_off=col_start, row_off=row_start,
                            width=PATCH_SIZE, height=PATCH_SIZE,
                        )
                        data = src.read(1, window=window).astype(np.float32)
                        if data.shape != (PATCH_SIZE, PATCH_SIZE):
                            padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                            padded[:data.shape[0], :data.shape[1]] = data
                            data = padded
                        data = data / 10000.0
                        band_arrays.append(data)

                if len(band_arrays) == len(S2_BANDS):
                    bands_10 = np.stack(band_arrays, axis=0)
                    zeros_3  = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    full = np.concatenate([bands_10, zeros_3], axis=0)
                    tiles.append((tile_date, torch.from_numpy(full)))
                    print(f"      Loaded tile {tile_date} (cloud={item.properties.get('eo:cloud_cover','?')}%)")
            except Exception as e:
                print(f"      Failed to load tile: {e}")
                continue

    except ImportError as e:
        print(f"    Planetary Computer libs not available: {e}")
    except Exception as e:
        print(f"    S2 query failed: {e}")
        traceback.print_exc()

    return tiles


def run_satellite_inference(sat_model, tiles):
    """Run HydroViT on list of (date_str, tensor) tiles.

    Returns list of dicts with:
      - tile_date, center_ts
      - embedding [1,256]     — shared embed for standalone scoring
      - fusion_embedding [1,384] — cls_token used by fusion projection bank
    """
    results = []
    with torch.no_grad():
        for (tile_date, tensor) in tiles:
            try:
                x = tensor.unsqueeze(0).to(DEVICE)
                out = sat_model(x)

                # 256-dim embedding for standalone anomaly scoring
                embedding = out["embedding"]
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)

                # 384-dim cls_token — matches fusion ProjectionBank native_dim for satellite
                fusion_emb = out.get("cls_token", None)
                if fusion_emb is None:
                    # Fallback: zero-pad embedding to 384
                    fusion_emb = torch.zeros(1, 384, device=DEVICE)
                    fusion_emb[:, :embedding.shape[-1]] = embedding
                if fusion_emb.dim() == 1:
                    fusion_emb = fusion_emb.unsqueeze(0)

                # Parse date to unix timestamp
                dt = datetime.strptime(tile_date, "%Y-%m-%d")
                ts = dt.timestamp()
                results.append({
                    "tile_date": tile_date,
                    "center_ts": ts,
                    "embedding": embedding.cpu(),
                    "fusion_embedding": fusion_emb.cpu(),
                })
            except Exception as e:
                print(f"      HydroViT forward error for {tile_date}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fusion inference
# ─────────────────────────────────────────────────────────────────────────────
def run_fusion_inference(fusion, fusion_head, sensor_scores, sat_scores, window_h=12.0):
    """For each sensor window, find a satellite observation within ±window_h hours.

    If found: run both through fusion → AnomalyDetectionHead.
    Returns list of dicts with center_time, center_ts, fusion_anomaly_prob.
    """
    fusion_results = []
    window_s = window_h * 3600.0

    with torch.no_grad():
        for s in sensor_scores:
            s_ts = s["center_ts"]
            s_emb = s["embedding"].to(DEVICE)   # [1, 256]

            # Find nearest satellite obs
            best_sat = None
            best_diff = float("inf")
            for sat in sat_scores:
                diff = abs(sat["center_ts"] - s_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_sat = sat

            if best_sat is None or best_diff > window_s:
                continue   # no matching satellite within ±12h

            # Use 384-dim cls_token for fusion (matches projection bank native_dim)
            sat_emb = best_sat.get("fusion_embedding", best_sat["embedding"]).to(DEVICE)

            try:
                # Reset fusion registry for each fusion call
                fusion.registry.reset()
                latent = None

                out1 = fusion(
                    modality_id="sensor",
                    raw_embedding=s_emb,
                    timestamp=s_ts,
                    confidence=1.0,
                    latent_state=latent,
                )
                latent = out1.latent_state

                out2 = fusion(
                    modality_id="satellite",
                    raw_embedding=sat_emb,
                    timestamp=best_sat["center_ts"],
                    confidence=1.0,
                    latent_state=latent,
                )

                anomaly_out = fusion_head(out2.fused_state)
                prob = anomaly_out.anomaly_probability.item()

                fusion_results.append({
                    "center_time": s["center_time"],
                    "center_ts":   s_ts,
                    "fusion_anomaly_prob": float(prob),
                    "matched_sat_date": best_sat["tile_date"],
                    "time_diff_h": round(best_diff / 3600.0, 2),
                })
            except Exception as e:
                print(f"      Fusion error at {s['center_time']}: {e}")
                continue

    return fusion_results


# ─────────────────────────────────────────────────────────────────────────────
# Lead time analysis
# ─────────────────────────────────────────────────────────────────────────────
def compute_lead_time(scores_list, advisory_ts, key="anomaly_probability", threshold=0.10):
    """Return first detection time and lead time in hours."""
    pre_scores = [s for s in scores_list if s.get("center_ts", 0) < advisory_ts]
    if not pre_scores:
        return None, None

    for s in pre_scores:
        if s.get(key, 0) > threshold:
            lead_h = (advisory_ts - s["center_ts"]) / 3600.0
            return s.get("center_time", s.get("tile_date", "?")), round(lead_h, 2)

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SENTINEL Multimodal Case Study Experiment")
    print(f"  Sensor ckpt   : {SENSOR_CKPT}")
    print(f"  Satellite ckpt: {SAT_CKPT}")
    print(f"  Fusion ckpt   : {FUSION_CKPT}")
    print(f"  Device        : {DEVICE}")
    print("=" * 70)

    print("\nLoading models...")
    sensor_model, sensor_head = load_sensor_models()
    sat_model   = load_satellite_model()
    fusion, fusion_head = load_fusion_models()

    event_results = []

    for event in EVENTS:
        eid  = event["event_id"]
        name = event["name"]
        print(f"\n{'─'*70}")
        print(f"Event: {name} ({event['advisory_date']})")
        print(f"  USGS site: {event['usgs_site']}, lat={event['lat']}, lon={event['lon']}")

        advisory_dt = datetime.strptime(event["advisory_date"], "%Y-%m-%d")
        advisory_ts = advisory_dt.timestamp()

        pre_days   = 90
        post_days  = 14
        start_dt   = advisory_dt - timedelta(days=pre_days)
        end_dt     = advisory_dt + timedelta(days=post_days)
        start_str  = start_dt.strftime("%Y-%m-%d")
        end_str    = end_dt.strftime("%Y-%m-%d")
        # Satellite: advisory_date - 90 days to advisory_date + 7 days
        sat_end_str = (advisory_dt + timedelta(days=7)).strftime("%Y-%m-%d")

        result_row: Dict[str, Any] = {
            "event_id": eid,
            "name": name,
            "advisory_date": event["advisory_date"],
            "usgs_site": event["usgs_site"],
            "lat": event["lat"],
            "lon": event["lon"],
        }

        # ── Sensor modality ──────────────────────────────────────────────────
        print(f"\n  [Sensor] Fetching {start_str} → {end_str}...")
        sensor_scores = []
        try:
            df = fetch_usgs_iv(event["usgs_site"], start_str, end_str)
            if df is not None and len(df) > 0:
                print(f"    Records: {len(df)}")
                windows = preprocess_for_aquassm(df, seq_len=128, stride=64)
                print(f"    Windows: {len(windows)}")
                if windows:
                    raw_scores = run_sensor_inference(sensor_model, sensor_head, windows)
                    # Detach embeddings for JSON but keep in sensor_scores for fusion
                    sensor_scores = raw_scores
                    sensor_probs_only = [
                        {"center_time": s["center_time"],
                         "center_ts": s["center_ts"],
                         "anomaly_probability": s["anomaly_probability"]}
                        for s in raw_scores
                    ]
                    print(f"    Sensor scores: {len(sensor_scores)}, "
                          f"max={max(s['anomaly_probability'] for s in sensor_scores):.4f}")

                    _, sensor_lead_h = compute_lead_time(
                        sensor_scores, advisory_ts, "anomaly_probability", DETECTION_THRESHOLD
                    )
                    result_row["sensor_lead_time_h"] = sensor_lead_h
                    result_row["sensor_max_prob"] = round(
                        max(s["anomaly_probability"] for s in sensor_scores), 4)
                    result_row["sensor_scores_sample"] = [
                        {"center_time": s["center_time"],
                         "anomaly_probability": round(s["anomaly_probability"], 4)}
                        for s in sensor_scores[-30:]
                    ]
                    result_row["sensor_status"] = "ok"
                else:
                    print("    Insufficient data for windows")
                    result_row["sensor_status"] = "insufficient_data"
            else:
                print("    No USGS data returned")
                result_row["sensor_status"] = "no_data"
        except Exception as e:
            print(f"    Sensor error: {e}")
            result_row["sensor_status"] = f"error: {e}"

        time.sleep(0.5)

        # ── Satellite modality ───────────────────────────────────────────────
        print(f"\n  [Satellite] Querying PC for {start_str} → {sat_end_str}...")
        sat_scores_for_fusion = []
        result_row["n_s2_tiles_found"] = 0  # default; updated below if tiles found
        try:
            tiles = query_s2_tiles(event["lat"], event["lon"], start_str, sat_end_str)
            result_row["n_s2_tiles_found"] = len(tiles)

            if tiles:
                sat_scores_for_fusion = run_satellite_inference(sat_model, tiles)
                print(f"    Satellite scores computed: {len(sat_scores_for_fusion)}")

                # For satellite lead time, compute per-tile anomaly via fusion head only
                # (satellite has no standalone anomaly head, use fusion head with sat emb only)
                # Use 384-dim fusion_embedding (cls_token) for fusion projection bank
                sat_anomaly_probs = []
                with torch.no_grad():
                    for sat_entry in sat_scores_for_fusion:
                        # Use fusion_embedding (384-dim cls_token) for fusion
                        emb = sat_entry.get("fusion_embedding", sat_entry["embedding"]).to(DEVICE)
                        fusion.registry.reset()
                        out = fusion(
                            modality_id="satellite",
                            raw_embedding=emb,
                            timestamp=sat_entry["center_ts"],
                            confidence=1.0,
                            latent_state=None,
                        )
                        aout = fusion_head(out.fused_state)
                        prob = aout.anomaly_probability.item()
                        sat_anomaly_probs.append({
                            "center_time": sat_entry["tile_date"],
                            "center_ts": sat_entry["center_ts"],
                            "anomaly_probability": float(prob),
                        })

                if sat_anomaly_probs:
                    _, sat_lead_h = compute_lead_time(
                        sat_anomaly_probs, advisory_ts, "anomaly_probability", DETECTION_THRESHOLD
                    )
                    result_row["satellite_lead_time_h"] = sat_lead_h
                    result_row["satellite_max_prob"] = round(
                        max(s["anomaly_probability"] for s in sat_anomaly_probs), 4)
                    result_row["satellite_scores"] = [
                        {"date": s["center_time"],
                         "anomaly_probability": round(s["anomaly_probability"], 4)}
                        for s in sat_anomaly_probs
                    ]
                    result_row["s2_status"] = "ok"
                else:
                    result_row["s2_status"] = "inference_failed"
            else:
                print("    No S2 tiles found")
                result_row["s2_status"] = "no_data"
                result_row["n_s2_tiles_found"] = 0
        except Exception as e:
            print(f"    Satellite error: {e}")
            result_row["s2_status"] = f"error: {e}"
            # Don't reset n_s2_tiles_found if it was already set above

        # ── Fusion modality ──────────────────────────────────────────────────
        print(f"\n  [Fusion] Combining sensor + satellite...")
        if sensor_scores and sat_scores_for_fusion:
            try:
                fusion_scores = run_fusion_inference(
                    fusion, fusion_head, sensor_scores, sat_scores_for_fusion,
                    window_h=FUSION_WINDOW_H,
                )
                print(f"    Fusion windows: {len(fusion_scores)}")

                if fusion_scores:
                    _, fusion_lead_h = compute_lead_time(
                        fusion_scores, advisory_ts, "fusion_anomaly_prob", DETECTION_THRESHOLD
                    )
                    result_row["fusion_lead_time_h"] = fusion_lead_h
                    result_row["fusion_max_prob"] = round(
                        max(s["fusion_anomaly_prob"] for s in fusion_scores), 4)
                    result_row["fusion_scores_sample"] = [
                        {"center_time": s["center_time"],
                         "fusion_anomaly_prob": round(s["fusion_anomaly_prob"], 4),
                         "matched_sat_date": s.get("matched_sat_date", "?"),
                         "time_diff_h": s.get("time_diff_h", None)}
                        for s in fusion_scores[-20:]
                    ]
                    result_row["fusion_status"] = "ok"
                else:
                    print("    No fusion windows computed (no matching sensor/satellite windows)")
                    result_row["fusion_status"] = "no_matching_windows"
                    result_row["fusion_lead_time_h"] = None
            except Exception as e:
                print(f"    Fusion error: {e}")
                traceback.print_exc()
                result_row["fusion_status"] = f"error: {e}"
                result_row["fusion_lead_time_h"] = None
        else:
            reason = "no_sensor_data" if not sensor_scores else "no_satellite_data"
            print(f"    Skipping fusion: {reason}")
            result_row["fusion_status"] = f"skipped_{reason}"
            result_row["fusion_lead_time_h"] = None

        # Summary print
        print(f"\n  Summary for {name}:")
        print(f"    Sensor lead time   : {result_row.get('sensor_lead_time_h', 'N/A')} h")
        print(f"    Satellite lead time: {result_row.get('satellite_lead_time_h', 'N/A')} h")
        print(f"    Fusion lead time   : {result_row.get('fusion_lead_time_h', 'N/A')} h")
        print(f"    n_s2_tiles_found   : {result_row.get('n_s2_tiles_found', 0)}")

        event_results.append(result_row)
        time.sleep(0.5)

    # ── Aggregate statistics ─────────────────────────────────────────────────
    sensor_leads = [r["sensor_lead_time_h"] for r in event_results
                    if r.get("sensor_lead_time_h") is not None]
    sat_leads = [r.get("satellite_lead_time_h") for r in event_results
                 if r.get("satellite_lead_time_h") is not None]
    fusion_leads = [r.get("fusion_lead_time_h") for r in event_results
                    if r.get("fusion_lead_time_h") is not None]

    statistics = {
        "sensor_mean_lead_h":    round(float(np.mean(sensor_leads)), 2) if sensor_leads else None,
        "sensor_median_lead_h":  round(float(np.median(sensor_leads)), 2) if sensor_leads else None,
        "sensor_n_detected":     len(sensor_leads),
        "satellite_mean_lead_h": round(float(np.mean(sat_leads)), 2) if sat_leads else None,
        "satellite_n_detected":  len(sat_leads),
        "fusion_mean_lead_h":    round(float(np.mean(fusion_leads)), 2) if fusion_leads else None,
        "fusion_n_detected":     len(fusion_leads),
        "n_events":              len(EVENTS),
        "detection_threshold":   DETECTION_THRESHOLD,
    }

    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "sensor_checkpoint":    str(SENSOR_CKPT),
        "satellite_checkpoint": str(SAT_CKPT),
        "fusion_checkpoint":    str(FUSION_CKPT),
        "n_events": len(EVENTS),
        "statistics": statistics,
        "events": event_results,
    }

    out_path = OUTPUT_DIR / "case_studies_multimodal.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("=== SENTINEL Multimodal Case Study Results ===")
    print("=" * 70)
    for r in event_results:
        eid  = r["event_id"]
        s_lt = r.get("sensor_lead_time_h")
        sat_lt = r.get("satellite_lead_time_h")
        f_lt = r.get("fusion_lead_time_h")
        n_tiles = r.get("n_s2_tiles_found", 0)
        print(f"  {eid:40s}")
        print(f"      sensor_lead={s_lt} h  sat_lead={sat_lt} h  "
              f"fusion_lead={f_lt} h  n_tiles={n_tiles}")
    print()
    print(f"Statistics: {json.dumps(statistics, indent=2)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
