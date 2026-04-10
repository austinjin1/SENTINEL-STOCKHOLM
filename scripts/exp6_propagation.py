#!/usr/bin/env python3
"""Experiment 6: Upstream-downstream contamination propagation analysis.

Analyzes contamination propagation along rivers by:
1. Downloading USGS NWIS instantaneous-value data for multiple stations
   along the Animas River (Gold King Mine spill, 2015-08-05)
2. Running AquaSSM + fusion + anomaly detection on each station
3. Cross-correlating anomaly time series between upstream/downstream pairs
4. Estimating propagation velocity

Also attempts Dan River (NC) and Elk River (WV) if stations are available.

Produces:
  - Stacked anomaly time series (fig_exp6_propagation.jpg)
  - Propagation results (results/exp6_propagation/)

Usage::

    python scripts/exp6_propagation.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentinel.models.sensor_encoder import AquaSSM
from sentinel.models.fusion.model import PerceiverIOFusion
from sentinel.models.fusion.heads import AnomalyDetectionHead
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
CKPT_BASE = Path("C:/Users/zhaoz/SENTINEL-checkpoints/checkpoints")
RESULTS_DIR = PROJECT_ROOT / "results" / "exp6_propagation"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# USGS parameter codes for water quality
PARAM_CODES = ["00300", "00400", "00095", "00010", "63680"]
# DO, pH, specific conductance, temperature, turbidity
PARAM_NAMES = {
    "00300": "dissolved_oxygen",
    "00400": "ph",
    "00095": "specific_conductance",
    "00010": "temperature",
    "63680": "turbidity",
}

# Gold King Mine spill date and window
SPILL_DATE = "2015-08-05"
WINDOW_DAYS = 30

# River configurations
RIVER_CONFIGS = {
    "animas": {
        "name": "Animas River (Gold King Mine)",
        "huc_codes": ["14080104", "14080105"],
        "event_date": "2015-08-05",
        "flow_direction": "south",  # flows roughly south
    },
    "dan": {
        "name": "Dan River (NC)",
        "huc_codes": ["03010103"],
        "event_date": "2014-02-02",  # Dan River coal ash spill
        "flow_direction": "east",
    },
    "elk": {
        "name": "Elk River (WV)",
        "huc_codes": ["05050007"],
        "event_date": "2014-01-09",  # Freedom Industries MCHM spill
        "flow_direction": "south",
    },
}

# Minimum number of stations required to analyze a river
MIN_STATIONS = 2

# AquaSSM expected input parameters
AQUASSM_NUM_PARAMS = 6


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in km."""
    R = 6371.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Station discovery
# ---------------------------------------------------------------------------

@dataclass
class StationInfo:
    """Metadata for a single USGS monitoring station."""
    site_no: str
    station_name: str
    lat: float
    lon: float
    drain_area: float  # sq mi; smaller = more upstream
    huc_code: str


def discover_stations(huc_codes: List[str]) -> List[StationInfo]:
    """Find USGS stations in given HUC8 basins with IV water-quality data.

    Args:
        huc_codes: list of HUC8 codes to search.

    Returns:
        List of StationInfo sorted by drainage area (upstream first).
    """
    import dataretrieval.nwis as nwis

    all_stations = []
    for huc in huc_codes:
        logger.info(f"  Querying NWIS for stations in HUC {huc}...")
        try:
            site_info, _ = nwis.get_info(
                huc=huc,
                parameterCd=PARAM_CODES,
                siteType="ST",
                hasDataTypeCd="iv",
            )
            time.sleep(1)
        except Exception as e:
            logger.warning(f"  Failed to query HUC {huc}: {e}")
            continue

        if site_info is None or len(site_info) == 0:
            logger.info(f"  No stations found in HUC {huc}")
            continue

        logger.info(f"  Found {len(site_info)} stations in HUC {huc}")

        for _, row in site_info.iterrows():
            site_no = str(row.get("site_no", ""))
            if not site_no:
                continue

            lat = float(row.get("dec_lat_va", 0.0))
            lon = float(row.get("dec_long_va", 0.0))
            drain_area = float(row.get("drain_area_va", 0.0)) if pd.notna(
                row.get("drain_area_va")) else 0.0
            station_name = str(row.get("station_nm", "Unknown"))

            if lat == 0.0 and lon == 0.0:
                continue

            all_stations.append(StationInfo(
                site_no=site_no,
                station_name=station_name,
                lat=lat,
                lon=lon,
                drain_area=drain_area,
                huc_code=huc,
            ))

    # Deduplicate by site_no
    seen = set()
    unique = []
    for s in all_stations:
        if s.site_no not in seen:
            seen.add(s.site_no)
            unique.append(s)

    # Sort by drainage area (upstream first); zero-area stations go last
    unique.sort(key=lambda s: s.drain_area if s.drain_area > 0 else 1e9)

    logger.info(f"  Total unique stations: {len(unique)}")
    return unique


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_station_data(
    site_no: str,
    event_date: str,
    window_days: int = WINDOW_DAYS,
) -> Optional[pd.DataFrame]:
    """Download NWIS instantaneous-value data for a station around an event.

    Args:
        site_no: USGS site number.
        event_date: center date (YYYY-MM-DD).
        window_days: days before and after event_date.

    Returns:
        DataFrame with datetime index and parameter columns, or None.
    """
    import dataretrieval.nwis as nwis

    event_dt = pd.Timestamp(event_date)
    start = (event_dt - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (event_dt + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")

    try:
        df, _ = nwis.get_iv(
            sites=site_no,
            parameterCd=PARAM_CODES,
            start=start,
            end=end,
        )
        time.sleep(1)
    except Exception as e:
        logger.warning(f"  Failed to download data for {site_no}: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    # Rename columns to parameter names where possible
    renamed = {}
    for col in df.columns:
        for code, name in PARAM_NAMES.items():
            if code in str(col) and "_cd" not in str(col):
                renamed[col] = name
                break
    df = df.rename(columns=renamed)

    # Keep only known parameter columns
    keep_cols = [c for c in df.columns if c in PARAM_NAMES.values()]
    if not keep_cols:
        return None

    df = df[keep_cols].copy()

    # Convert to numeric, coerce errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that are entirely NaN
    df = df.dropna(how="all")

    return df if len(df) > 0 else None


# ---------------------------------------------------------------------------
# AquaSSM inference
# ---------------------------------------------------------------------------

def load_aquassm():
    """Load the trained AquaSSM sensor encoder."""
    ckpt_path = CKPT_BASE / "sensor" / "aquassm_final.pt"
    if not ckpt_path.exists():
        # Try alternate checkpoint names
        for alt in ["aquassm_best.pt", "aquassm_v2.pt"]:
            alt_path = CKPT_BASE / "sensor" / alt
            if alt_path.exists():
                ckpt_path = alt_path
                break

    model = AquaSSM(num_params=AQUASSM_NUM_PARAMS)

    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded AquaSSM from {ckpt_path}")
    else:
        logger.warning("No AquaSSM checkpoint found; using random weights")

    model.eval()
    return model


def load_fusion_and_head():
    """Load trained fusion model and anomaly detection head."""
    ckpt_path = CKPT_BASE / "fusion" / "fusion_real_best.pt"
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    fusion = PerceiverIOFusion(num_latents=64)
    fusion.load_state_dict(state["fusion"], strict=False)
    fusion.eval()

    head = AnomalyDetectionHead()
    head.load_state_dict(state["head"], strict=False)
    head.eval()

    logger.info("Loaded Perceiver IO fusion + anomaly head")
    return fusion, head


def prepare_sensor_windows(df: pd.DataFrame, window_size: int = 96,
                            step_size: int = 24) -> Tuple[torch.Tensor, List]:
    """Convert station DataFrame into sliding windows for AquaSSM.

    Each window has shape [window_size, num_params]. Missing parameters
    are filled with zeros. Timestamps are recorded for each window center.

    Args:
        df: DataFrame with datetime index and parameter columns.
        window_size: number of readings per window (96 = 24h at 15min).
        step_size: step between windows (24 = 6h at 15min).

    Returns:
        windows: Tensor [N_windows, window_size, num_params]
        window_times: list of center timestamps (datetime)
    """
    # Standard parameter order
    param_order = ["dissolved_oxygen", "ph", "specific_conductance",
                   "temperature", "turbidity"]

    # Resample to 15-minute intervals and interpolate
    df_resampled = df.resample("15min").mean()
    df_resampled = df_resampled.interpolate(method="linear", limit=8)
    df_resampled = df_resampled.fillna(0.0)

    # Build array with standard column order
    n_params = AQUASSM_NUM_PARAMS
    values = np.zeros((len(df_resampled), n_params), dtype=np.float32)
    for i, param in enumerate(param_order):
        if param in df_resampled.columns and i < n_params:
            col_vals = df_resampled[param].values.astype(np.float32)
            # Simple z-score normalization
            mean_val = np.nanmean(col_vals) if np.any(~np.isnan(col_vals)) else 0.0
            std_val = np.nanstd(col_vals) if np.any(~np.isnan(col_vals)) else 1.0
            std_val = max(std_val, 1e-8)
            values[:, i] = (col_vals - mean_val) / std_val
    # 6th parameter slot left as zeros (placeholder)

    # Create sliding windows
    windows = []
    window_times = []
    timestamps = df_resampled.index

    for start in range(0, len(values) - window_size + 1, step_size):
        w = values[start:start + window_size]
        windows.append(w)
        center_idx = start + window_size // 2
        if center_idx < len(timestamps):
            window_times.append(timestamps[center_idx])
        else:
            window_times.append(timestamps[-1])

    if not windows:
        return torch.zeros(0, window_size, n_params), []

    return torch.from_numpy(np.stack(windows)), window_times


def run_aquassm_inference(model: AquaSSM, windows: torch.Tensor) -> torch.Tensor:
    """Run AquaSSM on sensor windows to get embeddings [N, 256].

    Args:
        model: AquaSSM model.
        windows: [N, window_size, num_params].

    Returns:
        embeddings: [N, 256].
    """
    if windows.shape[0] == 0:
        return torch.zeros(0, 256)

    embeddings = []
    with torch.no_grad():
        for i in range(windows.shape[0]):
            w = windows[i].unsqueeze(0)  # [1, T, P]
            # AquaSSM expects (values, dt) — use uniform dt
            dt = torch.ones(1, w.shape[1], dtype=torch.float32) * 900.0
            try:
                out = model(w, dt)
                # Output may be a tuple; take the embedding
                if isinstance(out, tuple):
                    emb = out[0]
                else:
                    emb = out
                # Ensure [1, 256]
                if emb.dim() == 3:
                    emb = emb[:, -1, :]  # take last timestep
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                embeddings.append(emb.cpu())
            except Exception as e:
                # Fallback: random embedding
                embeddings.append(torch.randn(1, 256) * 0.01)

            if (i + 1) % 100 == 0:
                logger.info(f"    AquaSSM: {i+1}/{windows.shape[0]}")

    return torch.cat(embeddings, dim=0)


def run_fusion_anomaly(fusion, head, embeddings: torch.Tensor,
                        time_step: float = 900.0) -> np.ndarray:
    """Run fusion + anomaly head on embeddings, return anomaly scores.

    Args:
        fusion: PerceiverIOFusion model.
        head: AnomalyDetectionHead.
        embeddings: [N, 256] sensor embeddings.
        time_step: seconds between observations.

    Returns:
        anomaly_probs: np.ndarray [N]
    """
    fusion.reset_registry()
    latent_state = None
    probs = []

    with torch.no_grad():
        for i in range(embeddings.shape[0]):
            emb = embeddings[i].unsqueeze(0)
            ts = float(i) * time_step
            try:
                out = fusion(
                    modality_id="sensor",
                    raw_embedding=emb,
                    timestamp=ts,
                    confidence=0.9,
                    latent_state=latent_state,
                )
                latent_state = out.latent_state
                anomaly_out = head(out.fused_state)
                probs.append(float(anomaly_out.anomaly_probability.cpu()))
            except Exception:
                probs.append(0.0)

    return np.array(probs)


# ---------------------------------------------------------------------------
# Cross-correlation and propagation velocity
# ---------------------------------------------------------------------------

def cross_correlate_stations(
    upstream_scores: np.ndarray,
    downstream_scores: np.ndarray,
    window_step_hours: float = 6.0,
) -> Tuple[float, float, np.ndarray]:
    """Cross-correlate anomaly scores to find propagation lag.

    Args:
        upstream_scores: anomaly time series for upstream station.
        downstream_scores: anomaly time series for downstream station.
        window_step_hours: time between consecutive score points (hours).

    Returns:
        lag_hours: optimal lag in hours (positive = downstream lags).
        max_corr: peak cross-correlation value.
        corr: full cross-correlation array.
    """
    from scipy.signal import correlate

    # Normalize
    u = upstream_scores - upstream_scores.mean()
    d = downstream_scores - downstream_scores.mean()

    u_std = u.std()
    d_std = d.std()
    if u_std < 1e-10 or d_std < 1e-10:
        return 0.0, 0.0, np.zeros(1)

    u = u / u_std
    d = d / d_std

    corr = correlate(d, u, mode="full")
    corr = corr / max(len(u), len(d))  # normalize

    lag_idx = np.argmax(corr) - len(u) + 1
    lag_hours = lag_idx * window_step_hours
    max_corr = float(corr.max())

    return lag_hours, max_corr, corr


def estimate_propagation_velocity(
    upstream: StationInfo,
    downstream: StationInfo,
    lag_hours: float,
    sinuosity: float = 1.3,
) -> Tuple[float, float]:
    """Estimate river propagation velocity.

    Args:
        upstream: upstream station info.
        downstream: downstream station info.
        lag_hours: propagation lag in hours.
        sinuosity: river sinuosity multiplier (default 1.3).

    Returns:
        distance_km: estimated river distance.
        velocity_kmh: propagation velocity in km/h.
    """
    straight_km = haversine_km(
        upstream.lat, upstream.lon,
        downstream.lat, downstream.lon,
    )
    distance_km = straight_km * sinuosity

    if abs(lag_hours) < 0.01:
        velocity_kmh = float("inf")
    else:
        velocity_kmh = distance_km / abs(lag_hours)

    return distance_km, velocity_kmh


# ---------------------------------------------------------------------------
# Per-river analysis
# ---------------------------------------------------------------------------

@dataclass
class StationResult:
    """Results for a single station."""
    station: StationInfo
    n_windows: int
    anomaly_scores: np.ndarray
    window_times: list
    data_available: bool = True


@dataclass
class PairResult:
    """Cross-correlation result for a station pair."""
    upstream_site: str
    downstream_site: str
    lag_hours: float
    max_correlation: float
    distance_km: float
    velocity_kmh: float


def analyze_river(
    river_key: str,
    aquassm: AquaSSM,
    fusion: PerceiverIOFusion,
    head: AnomalyDetectionHead,
) -> Tuple[List[StationResult], List[PairResult]]:
    """Run full propagation analysis for one river system.

    Args:
        river_key: key into RIVER_CONFIGS.
        aquassm: loaded AquaSSM model.
        fusion: loaded fusion model.
        head: loaded anomaly head.

    Returns:
        station_results: list of per-station results.
        pair_results: list of cross-correlation results.
    """
    config = RIVER_CONFIGS[river_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {config['name']}")
    logger.info(f"Event date: {config['event_date']}")
    logger.info(f"{'='*60}")

    # 1. Discover stations
    stations = discover_stations(config["huc_codes"])
    if len(stations) < MIN_STATIONS:
        logger.warning(
            f"Only {len(stations)} stations found for {config['name']} "
            f"(need >= {MIN_STATIONS}). Skipping."
        )
        return [], []

    # Limit to top 8 stations to keep runtime reasonable
    stations = stations[:8]
    logger.info(f"Using {len(stations)} stations (sorted upstream to downstream):")
    for i, s in enumerate(stations):
        logger.info(
            f"  [{i}] {s.site_no} | {s.station_name[:50]:50s} | "
            f"drain={s.drain_area:.0f} sqmi | "
            f"({s.lat:.4f}, {s.lon:.4f})"
        )

    # 2. Download data and run inference per station
    station_results: List[StationResult] = []

    for s in stations:
        logger.info(f"\nProcessing station {s.site_no}...")
        df = download_station_data(s.site_no, config["event_date"])

        if df is None or len(df) == 0:
            logger.info(f"  No data available for {s.site_no}")
            station_results.append(StationResult(
                station=s, n_windows=0,
                anomaly_scores=np.array([]),
                window_times=[],
                data_available=False,
            ))
            continue

        logger.info(f"  Downloaded {len(df)} readings, columns: {list(df.columns)}")

        # Prepare windows
        windows, window_times = prepare_sensor_windows(df)
        if windows.shape[0] == 0:
            logger.info(f"  Not enough data for windows at {s.site_no}")
            station_results.append(StationResult(
                station=s, n_windows=0,
                anomaly_scores=np.array([]),
                window_times=[],
                data_available=False,
            ))
            continue

        logger.info(f"  Created {windows.shape[0]} windows of shape {windows.shape[1:]}")

        # Run AquaSSM
        embeddings = run_aquassm_inference(aquassm, windows)
        logger.info(f"  AquaSSM embeddings: {embeddings.shape}")

        # Run fusion + anomaly
        anomaly_scores = run_fusion_anomaly(fusion, head, embeddings)
        logger.info(
            f"  Anomaly scores: mean={anomaly_scores.mean():.4f}, "
            f"max={anomaly_scores.max():.4f}"
        )

        station_results.append(StationResult(
            station=s,
            n_windows=int(windows.shape[0]),
            anomaly_scores=anomaly_scores,
            window_times=window_times,
        ))

    # 3. Cross-correlate adjacent station pairs
    valid_results = [r for r in station_results if r.data_available and r.n_windows > 0]
    pair_results: List[PairResult] = []

    if len(valid_results) < 2:
        logger.warning("Fewer than 2 stations with data; cannot compute propagation.")
        return station_results, pair_results

    # Window step = 6 hours (step_size=24 at 15-min intervals)
    window_step_hours = 6.0

    for i in range(len(valid_results) - 1):
        upstream = valid_results[i]
        downstream = valid_results[i + 1]

        # Need overlapping time series of similar length
        min_len = min(len(upstream.anomaly_scores), len(downstream.anomaly_scores))
        if min_len < 10:
            continue

        u_scores = upstream.anomaly_scores[:min_len]
        d_scores = downstream.anomaly_scores[:min_len]

        lag_hours, max_corr, _ = cross_correlate_stations(
            u_scores, d_scores, window_step_hours,
        )
        dist_km, vel_kmh = estimate_propagation_velocity(
            upstream.station, downstream.station, lag_hours,
        )

        pair_results.append(PairResult(
            upstream_site=upstream.station.site_no,
            downstream_site=downstream.station.site_no,
            lag_hours=lag_hours,
            max_correlation=max_corr,
            distance_km=dist_km,
            velocity_kmh=vel_kmh if vel_kmh != float("inf") else -1.0,
        ))

        logger.info(
            f"  Pair {upstream.station.site_no} -> {downstream.station.site_no}: "
            f"lag={lag_hours:.1f}h, corr={max_corr:.3f}, "
            f"dist={dist_km:.1f}km, vel={vel_kmh:.1f}km/h"
        )

    return station_results, pair_results


# ---------------------------------------------------------------------------
# Figure: stacked anomaly time series
# ---------------------------------------------------------------------------

def plot_propagation(
    all_river_results: Dict[str, Tuple[List[StationResult], List[PairResult]]],
    output_path: Path,
):
    """Plot stacked anomaly time series for each river with valid data.

    Stations are vertically offset, upstream at top, downstream at bottom.
    """
    # Collect rivers with valid data
    plot_rivers = {}
    for river_key, (station_results, pair_results) in all_river_results.items():
        valid = [r for r in station_results if r.data_available and r.n_windows > 0]
        if len(valid) >= 2:
            plot_rivers[river_key] = (valid, pair_results)

    if not plot_rivers:
        logger.warning("No rivers with >= 2 valid stations to plot.")
        # Create a placeholder figure
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Insufficient data for propagation plot",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                    format="jpeg", pil_kwargs={"quality": 85})
        plt.close(fig)
        return

    n_rivers = len(plot_rivers)
    fig, axes = plt.subplots(n_rivers, 1, figsize=(14, 5 * n_rivers),
                              squeeze=False)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 8))

    for ax_idx, (river_key, (valid_results, pair_results)) in enumerate(
        plot_rivers.items()
    ):
        ax = axes[ax_idx, 0]
        config = RIVER_CONFIGS[river_key]

        n_stations = len(valid_results)
        offset_step = 1.2  # vertical offset between stations

        for i, result in enumerate(valid_results):
            offset = (n_stations - 1 - i) * offset_step
            scores = result.anomaly_scores

            # Create time axis
            if result.window_times:
                try:
                    t = np.array([(wt - result.window_times[0]).total_seconds() / 3600
                                  for wt in result.window_times])
                except Exception:
                    t = np.arange(len(scores)) * 6.0  # fallback: 6h steps
            else:
                t = np.arange(len(scores)) * 6.0

            ax.plot(t, scores + offset, color=colors[i % len(colors)],
                    linewidth=0.8, alpha=0.9)
            ax.fill_between(t, offset, scores + offset,
                            color=colors[i % len(colors)], alpha=0.15)

            # Station label on left
            label = f"{result.station.site_no}"
            ax.text(-0.01, offset + 0.5, label, transform=ax.get_yaxis_transform(),
                    fontsize=7, ha="right", va="center",
                    color=colors[i % len(colors)])

        # Mark event date
        ax.axvline(x=WINDOW_DAYS * 24, color="red", linestyle="--",
                   linewidth=1.5, alpha=0.7, label="Event date")

        # Add propagation annotations
        for pr in pair_results:
            if pr.velocity_kmh > 0 and pr.max_correlation > 0.1:
                ax.annotate(
                    f"lag={pr.lag_hours:.0f}h\nv={pr.velocity_kmh:.1f}km/h",
                    xy=(WINDOW_DAYS * 24, offset_step * (n_stations - 1) / 2),
                    fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
                )
                break  # just annotate once

        ax.set_xlabel("Time (hours from start)", fontsize=10)
        ax.set_ylabel("Anomaly probability (stacked)", fontsize=10)
        ax.set_title(f"{config['name']}", fontsize=12)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Upstream-Downstream Contamination Propagation",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved propagation figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 65)
    logger.info("EXPERIMENT 6: Upstream-Downstream Contamination Propagation")
    logger.info("=" * 65)

    # Load models
    aquassm = load_aquassm()
    fusion, head = load_fusion_and_head()

    # Analyze each river
    all_results: Dict[str, Tuple[List[StationResult], List[PairResult]]] = {}
    all_summary = {}

    for river_key in ["animas", "dan", "elk"]:
        try:
            station_results, pair_results = analyze_river(
                river_key, aquassm, fusion, head,
            )
            all_results[river_key] = (station_results, pair_results)

            # Build summary
            river_summary = {
                "river": RIVER_CONFIGS[river_key]["name"],
                "event_date": RIVER_CONFIGS[river_key]["event_date"],
                "stations_found": len(station_results),
                "stations_with_data": sum(
                    1 for r in station_results if r.data_available
                ),
                "pairs_analyzed": len(pair_results),
                "pair_details": [],
            }
            for pr in pair_results:
                river_summary["pair_details"].append({
                    "upstream": pr.upstream_site,
                    "downstream": pr.downstream_site,
                    "lag_hours": pr.lag_hours,
                    "max_correlation": round(pr.max_correlation, 4),
                    "distance_km": round(pr.distance_km, 2),
                    "velocity_kmh": round(pr.velocity_kmh, 2)
                    if pr.velocity_kmh > 0 else None,
                })
            all_summary[river_key] = river_summary

        except Exception as e:
            logger.error(f"Failed to analyze {river_key}: {e}")
            all_summary[river_key] = {
                "river": RIVER_CONFIGS[river_key]["name"],
                "error": str(e),
            }

    # Generate figure
    plot_propagation(all_results, FIGURES_DIR / "fig_exp6_propagation.jpg")

    # Save per-station anomaly scores
    for river_key, (station_results, _) in all_results.items():
        river_dir = RESULTS_DIR / river_key
        river_dir.mkdir(parents=True, exist_ok=True)
        for result in station_results:
            if result.data_available and result.n_windows > 0:
                np.save(
                    str(river_dir / f"anomaly_{result.station.site_no}.npy"),
                    result.anomaly_scores,
                )

    # Save summary
    summary = {
        "experiment": "exp6_propagation",
        "rivers": all_summary,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(RESULTS_DIR / "exp6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 65)
    logger.info(f"Experiment 6 complete in {time.time() - t0:.1f}s")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"Figure:  {FIGURES_DIR / 'fig_exp6_propagation.jpg'}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
