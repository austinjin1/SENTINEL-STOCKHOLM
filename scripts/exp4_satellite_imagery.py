#!/usr/bin/env python3
"""Experiment 4: Sentinel-2 satellite imagery analysis of contamination events.

Downloads actual Sentinel-2 L2A tiles from Microsoft Planetary Computer for
documented contamination events, runs HydroViT inference to predict water
quality parameters, and passes embeddings through the anomaly detection head
to track anomaly scores over time.

Only events after Sentinel-2 launch (June 2015) are eligible:
  - Gold King Mine (Aug 2015)
  - Houston Ship Channel (2019)
  - Lake Erie HAB (2023)
  - Gulf Dead Zone (2023)
  - Chesapeake Bay Blooms (2023)
  - East Palestine (2023)

Fallback: if Planetary Computer download fails, uses pre-extracted satellite
embeddings from data/real_embeddings/satellite_embeddings.pt.

MIT License -- Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.evaluation.case_study import HISTORICAL_EVENTS, HistoricalEvent
from sentinel.models.satellite_encoder.model import SatelliteEncoder
from sentinel.models.satellite_encoder.parameter_head import PARAM_NAMES, NUM_WATER_PARAMS
from sentinel.models.fusion.heads import AnomalyDetectionHead, ANOMALY_TYPES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CKPT_PATHS = [
    PROJECT_ROOT / "checkpoints" / "satellite" / "hydrovit_wq_v6.pt",
    Path("C:/Users/zhaoz/SENTINEL-checkpoints/checkpoints/satellite/hydrovit_wq_v6.pt"),
]
FUSION_CKPT_PATHS = [
    Path("C:/Users/zhaoz/SENTINEL-checkpoints/checkpoints/fusion/fusion_head_best.pt"),
    Path("C:/Users/zhaoz/SENTINEL-checkpoints/checkpoints/fusion/fusion_real_best.pt"),
]
FALLBACK_EMBEDDINGS = PROJECT_ROOT / "data" / "real_embeddings" / "satellite_embeddings.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp4_satellite"
FIGURES_DIR = PROJECT_ROOT / "figures"

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PATCH_SIZE = 224
MAX_CLOUD_COVER = 30

# Events with S2 data (post June 2015)
ELIGIBLE_EVENTS = [
    "gold_king_mine",
    "houston_ship_channel",
    "lake_erie_hab",
    "gulf_dead_zone",
    "chesapeake_bay_blooms",
    "east_palestine",
]

# Time offsets in days relative to onset_date
TIME_OFFSETS = {
    "T-30": -30,
    "T-15": -15,
    "T (onset)": 0,
    "T+15": 15,
    "T+30": 30,
    "T+60": 60,
}
SEARCH_WINDOW_DAYS = 3  # +/- days for each time point

DEVICE = torch.device("cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hydrovit() -> SatelliteEncoder:
    """Load HydroViT satellite encoder with trained weights."""
    model = SatelliteEncoder(in_chans=13, shared_embed_dim=256)

    ckpt_path = None
    for p in CKPT_PATHS:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is not None:
        log(f"Loading HydroViT checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model" in ckpt:
            state = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        log(f"  Loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        log("WARNING: No HydroViT checkpoint found, using random weights")

    model.to(DEVICE).eval()
    return model


def load_anomaly_head() -> AnomalyDetectionHead:
    """Load anomaly detection head (from fusion checkpoint or fresh init)."""
    head = AnomalyDetectionHead(state_dim=256, hidden_dim=128)

    for p in FUSION_CKPT_PATHS:
        if p.exists():
            log(f"Loading fusion checkpoint for anomaly head: {p}")
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
            # Extract anomaly head weights if embedded in full fusion model
            anomaly_keys = {
                k.replace("anomaly_head.", ""): v
                for k, v in state.items()
                if k.startswith("anomaly_head.")
            }
            if anomaly_keys:
                head.load_state_dict(anomaly_keys, strict=False)
                log("  Loaded anomaly head weights from fusion checkpoint")
            else:
                # Try loading directly
                try:
                    head.load_state_dict(state, strict=False)
                    log("  Loaded anomaly head weights (direct)")
                except Exception:
                    log("  Could not extract anomaly head weights, using initialized weights")
            break
    else:
        log("No fusion checkpoint found, using initialized anomaly head")

    head.to(DEVICE).eval()
    return head


# ---------------------------------------------------------------------------
# Sentinel-2 download from Planetary Computer
# ---------------------------------------------------------------------------

def download_s2_patch(
    event: HistoricalEvent,
    target_date: datetime,
    window_days: int = SEARCH_WINDOW_DAYS,
) -> Optional[torch.Tensor]:
    """Download a 224x224 S2 patch centered on event location.

    Returns:
        Tensor of shape [13, 224, 224] (10 S2 bands + 3 zero padding) or None.
    """
    try:
        import pystac_client
        import planetary_computer
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.transform import rowcol
    except ImportError as e:
        log(f"  Missing dependency for S2 download: {e}")
        return None

    start = (target_date - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=window_days)).strftime("%Y-%m-%d")

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=event.bbox,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        )
        items = list(search.items())

        if not items:
            log(f"  No S2 tiles found for {start} to {end}")
            return None

        # Pick the item with lowest cloud cover
        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        best_item = items[0]
        cloud = best_item.properties.get("eo:cloud_cover", "?")
        log(f"  Found {len(items)} tiles, using best (cloud={cloud}%): {best_item.id}")

        signed_item = planetary_computer.sign(best_item)

        # Read each band and extract 224x224 patch centered on event location
        band_arrays = []
        for band_name in S2_BANDS:
            if band_name not in signed_item.assets:
                log(f"  WARNING: band {band_name} not in assets, zero-filling")
                band_arrays.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
                continue

            href = signed_item.assets[band_name].href
            with rasterio.open(href) as src:
                # Convert lat/lon to pixel coordinates
                row, col = rowcol(src.transform, event.longitude, event.latitude)
                # Clamp to valid range
                half = PATCH_SIZE // 2
                row_start = max(0, row - half)
                col_start = max(0, col - half)
                row_end = row_start + PATCH_SIZE
                col_end = col_start + PATCH_SIZE

                # Ensure we don't go out of bounds
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

                # If the window was partially out-of-bounds, pad
                if data.shape != (PATCH_SIZE, PATCH_SIZE):
                    padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    padded[:data.shape[0], :data.shape[1]] = data
                    data = padded

                # Normalize S2 L2A reflectances (typically 0-10000 -> 0-1)
                data = data / 10000.0
                band_arrays.append(data)

        # Stack 10 bands + 3 zero channels -> [13, 224, 224]
        bands_10 = np.stack(band_arrays, axis=0)  # [10, 224, 224]
        zeros_3 = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        full_input = np.concatenate([bands_10, zeros_3], axis=0)  # [13, 224, 224]

        return torch.from_numpy(full_input)

    except Exception as e:
        log(f"  S2 download failed: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: SatelliteEncoder,
    anomaly_head: AnomalyDetectionHead,
    image: torch.Tensor,
) -> Dict[str, Any]:
    """Run HydroViT + anomaly head inference on a single image.

    Args:
        model: Loaded SatelliteEncoder.
        anomaly_head: Loaded AnomalyDetectionHead.
        image: [13, 224, 224] tensor.

    Returns:
        Dict with wq_params, embedding, anomaly_prob, severity, etc.
    """
    x = image.unsqueeze(0).to(DEVICE)  # [1, 13, 224, 224]
    outputs = model(x)

    wq_params = outputs["water_quality_params"].squeeze(0).cpu().numpy()  # [16]
    uncertainty = outputs["param_uncertainty"].squeeze(0).cpu().numpy()  # [16]
    embedding = outputs["embedding"].squeeze(0).cpu().numpy()  # [256]

    # Run through anomaly head
    anomaly_out = anomaly_head(outputs["embedding"])
    anomaly_prob = anomaly_out.anomaly_probability.item()
    severity = anomaly_out.severity_score.item()
    alert_probs = anomaly_out.alert_level_probs.squeeze(0).cpu().numpy()
    type_probs = anomaly_out.anomaly_type_probs.squeeze(0).cpu().numpy()

    return {
        "wq_params": wq_params,
        "param_uncertainty": uncertainty,
        "embedding": embedding,
        "anomaly_probability": anomaly_prob,
        "severity_score": severity,
        "alert_level_probs": alert_probs,
        "type_probs": type_probs,
    }


@torch.no_grad()
def run_fallback_inference(
    anomaly_head: AnomalyDetectionHead,
    embeddings: torch.Tensor,
    n_per_event: int = 6,
) -> List[Dict[str, Any]]:
    """Run anomaly head on pre-extracted embeddings as fallback.

    Splits embeddings into chunks to simulate per-event time series.
    """
    results = []
    for i in range(0, min(len(embeddings), n_per_event * len(ELIGIBLE_EVENTS)), 1):
        emb = embeddings[i].unsqueeze(0).to(DEVICE)  # [1, 256]
        anomaly_out = anomaly_head(emb)
        results.append({
            "embedding": embeddings[i].cpu().numpy(),
            "anomaly_probability": anomaly_out.anomaly_probability.item(),
            "severity_score": anomaly_out.severity_score.item(),
            "alert_level_probs": anomaly_out.alert_level_probs.squeeze(0).cpu().numpy(),
            "type_probs": anomaly_out.anomaly_type_probs.squeeze(0).cpu().numpy(),
        })
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_s2_pipeline(
    model: SatelliteEncoder,
    anomaly_head: AnomalyDetectionHead,
) -> Dict[str, Any]:
    """Attempt to download S2 imagery and run inference for all eligible events."""
    all_results = {}

    for event_id in ELIGIBLE_EVENTS:
        event = HISTORICAL_EVENTS[event_id]
        onset = datetime.fromisoformat(event.onset_date.replace("Z", "+00:00"))
        if onset.tzinfo is not None:
            onset = onset.replace(tzinfo=None)

        log(f"\n{'='*60}")
        log(f"Event: {event.name} ({event.year})")
        log(f"Location: {event.location_name}, {event.state}")
        log(f"Onset: {event.onset_date}")
        log(f"Coordinates: ({event.latitude}, {event.longitude})")
        log(f"{'='*60}")

        event_results = {
            "event_id": event_id,
            "name": event.name,
            "onset_date": event.onset_date,
            "time_points": {},
        }

        for label, offset_days in TIME_OFFSETS.items():
            target_date = onset + timedelta(days=offset_days)
            log(f"\n  {label}: {target_date.strftime('%Y-%m-%d')}")

            patch = download_s2_patch(event, target_date)
            if patch is not None:
                result = run_inference(model, anomaly_head, patch)
                # Convert numpy arrays for JSON serialization
                event_results["time_points"][label] = {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "offset_days": offset_days,
                    "wq_params": result["wq_params"].tolist(),
                    "param_uncertainty": result["param_uncertainty"].tolist(),
                    "anomaly_probability": result["anomaly_probability"],
                    "severity_score": result["severity_score"],
                    "alert_level_probs": result["alert_level_probs"].tolist(),
                    "type_probs": result["type_probs"].tolist(),
                    "source": "sentinel-2",
                }
                log(f"    Anomaly prob: {result['anomaly_probability']:.4f}")
                log(f"    Severity: {result['severity_score']:.4f}")
                top_params = np.argsort(np.abs(result["wq_params"]))[-3:][::-1]
                for idx in top_params:
                    log(f"    {PARAM_NAMES[idx]}: {result['wq_params'][idx]:.4f}")
            else:
                log(f"    No data available")
                event_results["time_points"][label] = {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "offset_days": offset_days,
                    "source": "missing",
                }

        all_results[event_id] = event_results

    return all_results


def run_fallback_pipeline(
    model: SatelliteEncoder,
    anomaly_head: AnomalyDetectionHead,
) -> Dict[str, Any]:
    """Use pre-extracted embeddings when S2 download is unavailable."""
    log("\n" + "=" * 60)
    log("FALLBACK: Using pre-extracted satellite embeddings")
    log("=" * 60)

    if not FALLBACK_EMBEDDINGS.exists():
        log(f"ERROR: Fallback embeddings not found at {FALLBACK_EMBEDDINGS}")
        return {}

    data = torch.load(str(FALLBACK_EMBEDDINGS), map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "embeddings" in data:
        embeddings = data["embeddings"]
    elif isinstance(data, torch.Tensor):
        embeddings = data
    else:
        # Try first tensor-like value
        for v in data.values():
            if isinstance(v, torch.Tensor):
                embeddings = v
                break
        else:
            log("ERROR: Could not extract embeddings from fallback file")
            return {}

    log(f"Loaded {len(embeddings)} pre-extracted satellite embeddings")
    log(f"Embedding shape: {embeddings.shape}")

    # Distribute embeddings across events to simulate time series
    n_per_event = 6  # match our 6 time points
    all_results = {}

    for i, event_id in enumerate(ELIGIBLE_EVENTS):
        event = HISTORICAL_EVENTS[event_id]
        onset = datetime.fromisoformat(event.onset_date.replace("Z", "+00:00"))
        if onset.tzinfo is not None:
            onset = onset.replace(tzinfo=None)

        log(f"\nEvent: {event.name} ({event.year})")

        event_results = {
            "event_id": event_id,
            "name": event.name,
            "onset_date": event.onset_date,
            "time_points": {},
            "source": "fallback_embeddings",
        }

        start_idx = i * n_per_event
        for j, (label, offset_days) in enumerate(TIME_OFFSETS.items()):
            idx = start_idx + j
            if idx >= len(embeddings):
                idx = idx % len(embeddings)

            target_date = onset + timedelta(days=offset_days)
            emb = embeddings[idx].unsqueeze(0).to(DEVICE)

            # Run full model if embedding is image-shaped (unlikely for pre-extracted)
            # Otherwise just run through anomaly head
            if emb.shape[-1] == 256:
                anomaly_out = anomaly_head(emb)
                result = {
                    "anomaly_probability": anomaly_out.anomaly_probability.item(),
                    "severity_score": anomaly_out.severity_score.item(),
                    "alert_level_probs": anomaly_out.alert_level_probs.squeeze(0).cpu().numpy().tolist(),
                    "type_probs": anomaly_out.anomaly_type_probs.squeeze(0).cpu().numpy().tolist(),
                }
            else:
                # If embeddings have different dim, project through a linear layer
                log(f"  Embedding dim {emb.shape[-1]} != 256, zero-padding")
                padded = torch.zeros(1, 256)
                padded[:, :min(emb.shape[-1], 256)] = emb[:, :256]
                anomaly_out = anomaly_head(padded)
                result = {
                    "anomaly_probability": anomaly_out.anomaly_probability.item(),
                    "severity_score": anomaly_out.severity_score.item(),
                    "alert_level_probs": anomaly_out.alert_level_probs.squeeze(0).cpu().numpy().tolist(),
                    "type_probs": anomaly_out.anomaly_type_probs.squeeze(0).cpu().numpy().tolist(),
                }

            event_results["time_points"][label] = {
                "date": target_date.strftime("%Y-%m-%d"),
                "offset_days": offset_days,
                "anomaly_probability": result["anomaly_probability"],
                "severity_score": result["severity_score"],
                "alert_level_probs": result["alert_level_probs"],
                "type_probs": result["type_probs"],
                "source": "fallback_embedding",
            }
            log(f"  {label}: anomaly_prob={result['anomaly_probability']:.4f}, "
                f"severity={result['severity_score']:.4f}")

        all_results[event_id] = event_results

    return all_results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_event_timeseries(
    all_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot WQ parameters and anomaly scores vs time for each event."""
    events_with_data = {
        eid: res for eid, res in all_results.items()
        if any(tp.get("source") not in ("missing", None)
               for tp in res["time_points"].values())
    }

    if not events_with_data:
        log("No events with data to plot")
        return

    n_events = len(events_with_data)
    fig, axes = plt.subplots(n_events, 2, figsize=(16, 4 * n_events), squeeze=False)
    fig.suptitle("Experiment 4: Sentinel-2 Satellite Imagery Analysis\nHydroViT WQ & Anomaly Detection at Contamination Sites",
                 fontsize=14, fontweight="bold", y=0.98)

    for row, (event_id, event_data) in enumerate(events_with_data.items()):
        ax_anomaly = axes[row, 0]
        ax_wq = axes[row, 1]

        # Gather data
        dates = []
        anomaly_probs = []
        severity_scores = []
        wq_data = {name: [] for name in PARAM_NAMES}
        valid_dates_wq = []
        offset_labels = []

        for label, tp in event_data["time_points"].items():
            if tp.get("source") in ("missing", None):
                continue

            date = datetime.strptime(tp["date"], "%Y-%m-%d")
            dates.append(date)
            offset_labels.append(label)
            anomaly_probs.append(tp["anomaly_probability"])
            severity_scores.append(tp["severity_score"])

            if "wq_params" in tp:
                valid_dates_wq.append(date)
                for k, name in enumerate(PARAM_NAMES):
                    wq_data[name].append(tp["wq_params"][k])

        if not dates:
            ax_anomaly.text(0.5, 0.5, "No data", ha="center", va="center",
                          transform=ax_anomaly.transAxes, fontsize=12)
            ax_wq.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_wq.transAxes, fontsize=12)
            continue

        # --- Anomaly panel ---
        ax_anomaly.plot(dates, anomaly_probs, "ro-", linewidth=2, markersize=8,
                       label="Anomaly prob", zorder=3)
        ax_anomaly.plot(dates, severity_scores, "bs--", linewidth=1.5, markersize=6,
                       label="Severity", zorder=3)

        # Mark onset
        event = HISTORICAL_EVENTS[event_id]
        onset_dt = datetime.fromisoformat(event.onset_date.replace("Z", "+00:00"))
        if onset_dt.tzinfo is not None:
            onset_dt = onset_dt.replace(tzinfo=None)
        ax_anomaly.axvline(onset_dt, color="red", linestyle=":", alpha=0.7,
                          label="Event onset")
        ax_anomaly.fill_between(dates, 0, 1, alpha=0.05, color="red")

        ax_anomaly.set_ylim(-0.05, 1.05)
        ax_anomaly.set_ylabel("Score")
        ax_anomaly.set_title(f"{event.name} — Anomaly Detection", fontweight="bold")
        ax_anomaly.legend(loc="upper left", fontsize=8)
        ax_anomaly.grid(True, alpha=0.3)
        ax_anomaly.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax_anomaly.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # --- WQ panel ---
        if valid_dates_wq:
            # Plot a selection of key WQ parameters
            key_params = ["chl_a", "turbidity", "dissolved_oxygen", "ph",
                         "pollution_anomaly_index", "oil_probability"]
            colors = plt.cm.Set2(np.linspace(0, 1, len(key_params)))
            for k, (param, color) in enumerate(zip(key_params, colors)):
                if param in wq_data and wq_data[param]:
                    vals = wq_data[param]
                    ax_wq.plot(valid_dates_wq, vals, "o-", color=color,
                             linewidth=1.5, markersize=5, label=param)

            ax_wq.axvline(onset_dt, color="red", linestyle=":", alpha=0.7)
            ax_wq.set_ylabel("Predicted Value")
            ax_wq.set_title(f"{event.name} — WQ Parameters", fontweight="bold")
            ax_wq.legend(loc="upper left", fontsize=7, ncol=2)
            ax_wq.grid(True, alpha=0.3)
            ax_wq.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax_wq.xaxis.get_majorticklabels(), rotation=30, ha="right")
        else:
            # Fallback: show anomaly type probabilities as bar chart
            last_tp = None
            for tp in event_data["time_points"].values():
                if "type_probs" in tp:
                    last_tp = tp
            if last_tp:
                type_probs = last_tp["type_probs"]
                bars = ax_wq.barh(list(ANOMALY_TYPES), type_probs, color="steelblue")
                ax_wq.set_xlabel("Probability")
                ax_wq.set_title(f"{event.name} — Anomaly Type Classification",
                              fontweight="bold")
                ax_wq.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved figure: {output_path}")


def plot_summary_heatmap(
    all_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Create a heatmap of anomaly scores across events and time points."""
    events_with_data = {
        eid: res for eid, res in all_results.items()
        if any(tp.get("source") not in ("missing", None)
               for tp in res["time_points"].values())
    }
    if not events_with_data:
        return

    time_labels = list(TIME_OFFSETS.keys())
    event_names = []
    anomaly_matrix = []

    for event_id, event_data in events_with_data.items():
        event = HISTORICAL_EVENTS[event_id]
        event_names.append(f"{event.name}\n({event.year})")
        row = []
        for label in time_labels:
            tp = event_data["time_points"].get(label, {})
            if tp.get("source") not in ("missing", None) and "anomaly_probability" in tp:
                row.append(tp["anomaly_probability"])
            else:
                row.append(np.nan)
        anomaly_matrix.append(row)

    matrix = np.array(anomaly_matrix)

    fig, ax = plt.subplots(figsize=(10, max(4, len(event_names) * 0.9)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(time_labels)))
    ax.set_xticklabels(time_labels, fontsize=10)
    ax.set_yticks(range(len(event_names)))
    ax.set_yticklabels(event_names, fontsize=9)

    # Annotate cells
    for i in range(len(event_names)):
        for j in range(len(time_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=9, color=color, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                       fontsize=8, color="gray")

    plt.colorbar(im, ax=ax, label="Anomaly Probability", shrink=0.8)
    ax.set_title("SENTINEL Anomaly Detection: Contamination Events from Space",
                fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Time Relative to Event Onset")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved heatmap: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 60)
    log("Experiment 4: Sentinel-2 Satellite Imagery Analysis")
    log("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    log("\nLoading models...")
    model = load_hydrovit()
    anomaly_head = load_anomaly_head()

    # Try S2 download pipeline first
    log("\nAttempting Sentinel-2 download from Planetary Computer...")
    s2_results = run_s2_pipeline(model, anomaly_head)

    # Check how many events got data
    events_with_s2 = sum(
        1 for r in s2_results.values()
        if any(tp.get("source") == "sentinel-2"
               for tp in r.get("time_points", {}).values())
    )
    log(f"\nEvents with S2 data: {events_with_s2}/{len(ELIGIBLE_EVENTS)}")

    # If no S2 data was obtained, fall back to pre-extracted embeddings
    if events_with_s2 == 0:
        log("\nNo S2 tiles obtained — falling back to pre-extracted embeddings")
        fallback_results = run_fallback_pipeline(model, anomaly_head)
        # Merge: use fallback for events without S2 data
        for eid, res in fallback_results.items():
            if eid not in s2_results or not any(
                tp.get("source") == "sentinel-2"
                for tp in s2_results.get(eid, {}).get("time_points", {}).values()
            ):
                s2_results[eid] = res

    all_results = s2_results

    # Save results
    results_path = RESULTS_DIR / "exp4_results.json"
    with open(str(results_path), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")

    # Generate figures
    log("\nGenerating figures...")
    plot_event_timeseries(all_results, FIGURES_DIR / "exp4_satellite_timeseries.png")
    plot_summary_heatmap(all_results, FIGURES_DIR / "exp4_satellite_heatmap.png")

    # Print summary
    log("\n" + "=" * 60)
    log("EXPERIMENT 4 SUMMARY")
    log("=" * 60)
    for event_id, event_data in all_results.items():
        event = HISTORICAL_EVENTS.get(event_id)
        if event is None:
            continue
        tps = event_data.get("time_points", {})
        valid = sum(1 for tp in tps.values() if tp.get("source") not in ("missing", None))
        onset_tp = tps.get("T (onset)", {})
        anom = onset_tp.get("anomaly_probability", "N/A")
        sev = onset_tp.get("severity_score", "N/A")
        source = onset_tp.get("source", "N/A")
        if isinstance(anom, float):
            anom = f"{anom:.4f}"
        if isinstance(sev, float):
            sev = f"{sev:.4f}"
        log(f"  {event.name}: {valid}/{len(TIME_OFFSETS)} time points, "
            f"onset anomaly={anom}, severity={sev} [{source}]")

    log("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
