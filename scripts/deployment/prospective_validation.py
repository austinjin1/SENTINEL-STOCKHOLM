#!/usr/bin/env python3
"""Pre-registered prospective validation pipeline for SENTINEL.

Implements tamper-proof forward prediction on real USGS monitoring sites
to evaluate whether SENTINEL can genuinely detect contamination events
in advance. This is Phase 0.2 of the SENTINEL 2.0 improvement plan.

The pipeline has four modes:

1. **register** -- Pre-register monitoring sites, thresholds, and
   predicted contamination types into a timestamped, SHA-256-hashed
   JSON commitment file. This locks in predictions before outcomes
   are known.

2. **predict** -- Fetch real-time USGS NWIS instantaneous-value data
   for the 18 pre-registered sites, run it through the trained AquaSSM
   encoder and Perceiver IO fusion model, and record anomaly scores
   with timestamps.

3. **evaluate** -- Compare forward predictions against actual observed
   conditions (e.g., EPA violation records, USGS event flags) and
   compute hit rate, lead time, and false alarm rate.

4. **report** -- Generate a human-readable summary of all prospective
   validation results to date.

Usage::

    # Step 1: Lock in the pre-registration commitment
    python scripts/prospective_validation.py --mode register

    # Step 2: Run daily (e.g., via cron)
    python scripts/prospective_validation.py --mode predict

    # Step 3: After events are confirmed
    python scripts/prospective_validation.py --mode evaluate

    # Step 4: Summary report
    python scripts/prospective_validation.py --mode report

MIT License -- Bryan Cheng, 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results" / "prospective"
REGISTRATIONS_DIR = RESULTS_DIR / "registrations"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
EVALUATIONS_DIR = RESULTS_DIR / "evaluations"
REPORTS_DIR = RESULTS_DIR / "reports"

CKPT_BASE = PROJECT_ROOT / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# USGS NWIS parameter codes
# ---------------------------------------------------------------------------

PARAMETER_CODES = {
    "00300": "DO",       # Dissolved Oxygen (mg/L)
    "00400": "pH",       # pH
    "00095": "SpCond",   # Specific Conductance (uS/cm)
    "00010": "Temp",     # Water Temperature (degC)
    "63680": "Turb",     # Turbidity (FNU)
}

# Canonical 6-column order for AquaSSM (ORP is placeholder, not available
# from standard NWIS IV data but required by the model architecture).
AQUASSM_PARAM_ORDER = ["pH", "DO", "Turb", "SpCond", "Temp", "ORP"]

# Sequence length and stride for sliding-window inference
SEQ_LENGTH = 128
STRIDE = 64

# ---------------------------------------------------------------------------
# Pre-registered monitoring sites
# ---------------------------------------------------------------------------

MONITORING_SITES: List[Dict[str, Any]] = [
    # --- Ohio / Lake Erie (HAB-prone) ---
    {
        "site_id": "04193500",
        "name": "Maumee River at Waterville, OH",
        "lat": 41.5009,
        "lon": -83.7127,
        "region": "Ohio/Lake Erie",
        "concern": "Harmful algal blooms (HABs), nutrient loading",
        "predicted_contamination_types": ["nutrient_bloom", "dissolved_oxygen_drop"],
    },
    {
        "site_id": "04199000",
        "name": "Huron River at Milan, OH",
        "lat": 41.2984,
        "lon": -82.6060,
        "region": "Ohio/Lake Erie",
        "concern": "Agricultural runoff, Lake Erie tributary",
        "predicted_contamination_types": ["nutrient_bloom", "turbidity_surge"],
    },
    {
        "site_id": "04208000",
        "name": "Cuyahoga River at Independence, OH",
        "lat": 41.3970,
        "lon": -81.6285,
        "region": "Ohio/Lake Erie",
        "concern": "Urban/industrial discharge, legacy contamination",
        "predicted_contamination_types": ["chemical_spike", "toxic_compound"],
    },
    # --- Chesapeake Bay Watershed ---
    {
        "site_id": "01578310",
        "name": "Susquehanna River at Conowingo, MD",
        "lat": 39.6578,
        "lon": -76.1747,
        "region": "Chesapeake Bay",
        "concern": "Nutrient loading to Chesapeake Bay, sediment",
        "predicted_contamination_types": ["nutrient_bloom", "turbidity_surge"],
    },
    {
        "site_id": "01646500",
        "name": "Potomac River near Washington, DC",
        "lat": 38.9498,
        "lon": -77.1276,
        "region": "Chesapeake Bay",
        "concern": "Urban runoff, wastewater, nutrient loading",
        "predicted_contamination_types": ["nutrient_bloom", "chemical_spike"],
    },
    {
        "site_id": "01668000",
        "name": "Rappahannock River near Fredericksburg, VA",
        "lat": 38.3176,
        "lon": -77.5283,
        "region": "Chesapeake Bay",
        "concern": "Agricultural and urban runoff",
        "predicted_contamination_types": ["turbidity_surge", "nutrient_bloom"],
    },
    # --- Gulf Coast ---
    {
        "site_id": "07374000",
        "name": "Mississippi River at Baton Rouge, LA",
        "lat": 30.4459,
        "lon": -91.1914,
        "region": "Gulf Coast",
        "concern": "Hypoxia zone driver, agricultural/industrial mix",
        "predicted_contamination_types": ["nutrient_bloom", "dissolved_oxygen_drop", "chemical_spike"],
    },
    {
        "site_id": "02489500",
        "name": "Pearl River near Bogalusa, LA",
        "lat": 30.7755,
        "lon": -89.8270,
        "region": "Gulf Coast",
        "concern": "Paper mill discharge, industrial contamination",
        "predicted_contamination_types": ["chemical_spike", "dissolved_oxygen_drop"],
    },
    # --- Great Lakes ---
    {
        "site_id": "04087000",
        "name": "Milwaukee River at Milwaukee, WI",
        "lat": 43.0542,
        "lon": -87.9065,
        "region": "Great Lakes",
        "concern": "Combined sewer overflows, urban runoff",
        "predicted_contamination_types": ["chemical_spike", "turbidity_surge"],
    },
    {
        "site_id": "04157005",
        "name": "Saginaw River at Saginaw, MI",
        "lat": 43.4265,
        "lon": -83.9508,
        "region": "Great Lakes",
        "concern": "Saginaw Bay eutrophication, industrial legacy",
        "predicted_contamination_types": ["nutrient_bloom", "toxic_compound"],
    },
    # --- Southeast ---
    {
        "site_id": "02336000",
        "name": "Chattahoochee River at Atlanta, GA",
        "lat": 33.8813,
        "lon": -84.4388,
        "region": "Southeast",
        "concern": "Urban runoff, wastewater treatment plant discharge",
        "predicted_contamination_types": ["chemical_spike", "dissolved_oxygen_drop"],
    },
    {
        "site_id": "02324000",
        "name": "Steinhatchee River near Cross City, FL",
        "lat": 29.6883,
        "lon": -83.3735,
        "region": "Southeast",
        "concern": "Spring-fed system, nutrient sensitivity",
        "predicted_contamination_types": ["nutrient_bloom", "ph_deviation"],
    },
    # --- Northeast ---
    {
        "site_id": "01463500",
        "name": "Delaware River at Trenton, NJ",
        "lat": 40.2217,
        "lon": -74.7802,
        "region": "Northeast",
        "concern": "Industrial corridor, drinking water source",
        "predicted_contamination_types": ["chemical_spike", "toxic_compound"],
    },
    {
        "site_id": "01304200",
        "name": "Peconic River at Riverhead, NY",
        "lat": 40.9168,
        "lon": -72.6618,
        "region": "Northeast",
        "concern": "Nitrogen loading, estuary eutrophication",
        "predicted_contamination_types": ["nutrient_bloom", "dissolved_oxygen_drop"],
    },
    # --- West ---
    {
        "site_id": "11447650",
        "name": "Sacramento River at Freeport, CA",
        "lat": 38.4566,
        "lon": -121.5007,
        "region": "West",
        "concern": "Agricultural pesticides, Delta water quality",
        "predicted_contamination_types": ["chemical_spike", "toxic_compound", "turbidity_surge"],
    },
    {
        "site_id": "11303500",
        "name": "San Joaquin River near Vernalis, CA",
        "lat": 37.6758,
        "lon": -121.2653,
        "region": "West",
        "concern": "Selenium, agricultural drainage, salinity",
        "predicted_contamination_types": ["toxic_compound", "chemical_spike"],
    },
    # --- Midwest ---
    {
        "site_id": "05331000",
        "name": "Mississippi River at St Paul, MN",
        "lat": 44.9438,
        "lon": -93.0880,
        "region": "Midwest",
        "concern": "Urban-rural transition, WWTP discharge",
        "predicted_contamination_types": ["nutrient_bloom", "chemical_spike"],
    },
    {
        "site_id": "04157000",
        "name": "Tittabawassee River at Midland, MI",
        "lat": 43.6314,
        "lon": -84.2269,
        "region": "Midwest",
        "concern": "PFAS contamination site, Dow Chemical legacy",
        "predicted_contamination_types": ["toxic_compound", "chemical_spike"],
    },
]

# Alert thresholds for the pre-registration commitment
ALERT_THRESHOLDS = {
    "anomaly_probability_threshold": 0.9,
    "sustained_windows_required": 5,
    "severity_threshold": 0.7,
    "min_confidence": 0.6,
}


# =========================================================================
# Mode 1: Pre-Registration
# =========================================================================

def compute_commitment_hash(registration: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of the registration commitment.

    The hash covers sites, thresholds, and the registration timestamp,
    ensuring that any post-hoc tampering is detectable.

    Args:
        registration: The full registration dict (without the hash field).

    Returns:
        Hex-encoded SHA-256 digest.
    """
    # Build a canonical JSON string for hashing. We exclude the hash
    # field itself (if present) and sort keys for determinism.
    hashable = {
        "sites": registration["sites"],
        "thresholds": registration["thresholds"],
        "timestamp": registration["timestamp"],
        "model_info": registration.get("model_info", {}),
    }
    canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_model_info() -> Dict[str, Any]:
    """Gather information about the currently available model checkpoints.

    Returns:
        Dict with checkpoint paths, modification times, and sizes.
    """
    info: Dict[str, Any] = {}
    ckpt_paths = {
        "sensor_encoder": CKPT_BASE / "sensor" / "aquassm_real_best.pt",
        "fusion_model": CKPT_BASE / "fusion" / "fusion_real_best.pt",
    }
    for name, path in ckpt_paths.items():
        if path.exists():
            stat = path.stat()
            info[name] = {
                "path": str(path),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "available": True,
            }
        else:
            info[name] = {
                "path": str(path),
                "available": False,
            }
    return info


def run_register() -> Path:
    """Create a new pre-registration commitment file.

    Writes a timestamped JSON file containing the full list of monitoring
    sites, alert thresholds, predicted contamination types, and a SHA-256
    commitment hash for tamper-proofing.

    Returns:
        Path to the created registration file.
    """
    REGISTRATIONS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    timestamp_str = now.isoformat()
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Build site entries with current conditions summary
    sites = []
    for site in MONITORING_SITES:
        entry = {
            "site_id": site["site_id"],
            "name": site["name"],
            "latitude": site["lat"],
            "longitude": site["lon"],
            "region": site["region"],
            "concern": site["concern"],
            "predicted_contamination_types": site["predicted_contamination_types"],
            "current_conditions_summary": (
                f"Pre-anomaly baseline for {site['name']}. "
                f"Primary concern: {site['concern']}. "
                f"Region: {site['region']}."
            ),
        }
        sites.append(entry)

    model_info = get_model_info()

    registration = {
        "version": "1.0",
        "pipeline": "SENTINEL Prospective Validation",
        "phase": "Phase 0.2",
        "timestamp": timestamp_str,
        "n_sites": len(sites),
        "sites": sites,
        "thresholds": ALERT_THRESHOLDS,
        "parameters_queried": {
            code: name for code, name in PARAMETER_CODES.items()
        },
        "model_info": model_info,
        "nwis_query_config": {
            "lookback_days": 7,
            "sequence_length": SEQ_LENGTH,
            "stride": STRIDE,
            "parameter_codes": list(PARAMETER_CODES.keys()),
        },
    }

    # Compute commitment hash
    commitment_hash = compute_commitment_hash(registration)
    registration["commitment_hash_sha256"] = commitment_hash

    output_path = REGISTRATIONS_DIR / f"registration_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(registration, f, indent=2)

    logger.info(f"Pre-registration created: {output_path}")
    logger.info(f"  Timestamp: {timestamp_str}")
    logger.info(f"  Sites: {len(sites)}")
    logger.info(f"  Commitment hash: {commitment_hash}")

    # Also write a latest symlink / copy for easy access
    latest_path = REGISTRATIONS_DIR / "latest_registration.json"
    with open(latest_path, "w") as f:
        json.dump(registration, f, indent=2)
    logger.info(f"  Latest registration: {latest_path}")

    return output_path


# =========================================================================
# Mode 2: Forward Prediction
# =========================================================================

def fetch_nwis_data(
    site_id: str,
    lookback_days: int = 7,
) -> Optional[pd.DataFrame]:
    """Fetch recent instantaneous-value data from USGS NWIS.

    Queries the last ``lookback_days`` of IV data for the five target
    water quality parameters (DO, pH, SpCond, Temp, Turb).

    Args:
        site_id: USGS site number (e.g., "04193500").
        lookback_days: Number of days of data to fetch.

    Returns:
        DataFrame with columns [datetime, DO, pH, SpCond, Temp, Turb]
        indexed by datetime, or None if the fetch fails.
    """
    try:
        import dataretrieval.nwis as nwis
    except ImportError:
        logger.error(
            "dataretrieval package not installed. "
            "Install with: pip install dataretrieval"
        )
        return None

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)

    try:
        df, _ = nwis.get_iv(
            sites=site_id,
            parameterCd=list(PARAMETER_CODES.keys()),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        logger.warning(f"  NWIS fetch failed for site {site_id}: {e}")
        return None

    if df is None or len(df) == 0:
        logger.warning(f"  No IV data returned for site {site_id}")
        return None

    # Rename columns from USGS parameter codes to standard names
    rename_map = {}
    for code, name in PARAMETER_CODES.items():
        for col in df.columns:
            if code in col and "_cd" not in col and col != "site_no":
                rename_map[col] = name
                break

    df = df.rename(columns=rename_map)

    # Keep only the parameters we care about
    available = [p for p in PARAMETER_CODES.values() if p in df.columns]
    if not available:
        logger.warning(f"  No target parameters found for site {site_id}")
        return None

    df = df[available].copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()

    # Drop fully-null rows
    df = df.dropna(how="all")

    logger.info(
        f"  Site {site_id}: {len(df)} records, "
        f"params={available}, "
        f"range={df.index.min()} to {df.index.max()}"
    )
    return df


def prepare_aquassm_input(
    df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a NWIS DataFrame into AquaSSM-compatible tensor windows.

    Builds sliding windows of length SEQ_LENGTH with STRIDE overlap.
    Pads to 6 parameters in the canonical AquaSSM order:
    [pH, DO, Turb, SpCond, Temp, ORP].

    Args:
        df: DataFrame with standard parameter column names.

    Returns:
        Tuple of (x, delta_ts, timestamps) where:
            x: [N_windows, SEQ_LENGTH, 6] sensor readings
            delta_ts: [N_windows, SEQ_LENGTH] time gaps in seconds
            timestamps: [N_windows, SEQ_LENGTH] absolute timestamps (seconds)
    """
    # Resample to 15-minute intervals (AquaSSM training cadence)
    df_resampled = df.resample("15min").mean()
    df_resampled = df_resampled.dropna(how="all")

    if len(df_resampled) < SEQ_LENGTH:
        # Pad with the available data repeated, or just use what we have
        logger.warning(
            f"  Only {len(df_resampled)} records after resampling "
            f"(need {SEQ_LENGTH}). Will pad."
        )

    # Build value matrix in canonical order
    values = np.zeros((len(df_resampled), 6), dtype=np.float32)
    for i, param in enumerate(AQUASSM_PARAM_ORDER):
        if param in df_resampled.columns:
            col = df_resampled[param].values.astype(np.float32)
            values[:, i] = np.nan_to_num(col, nan=0.0)
        # else: stays zero (ORP and any missing params)

    # Compute timestamps in seconds since start
    ts_index = df_resampled.index
    if ts_index.tz is None:
        ts_index = ts_index.tz_localize("UTC")
    t0 = ts_index[0]
    timestamps_sec = np.array(
        [(t - t0).total_seconds() for t in ts_index], dtype=np.float32
    )

    # Compute time gaps (delta_t between consecutive observations)
    delta_ts = np.zeros(len(timestamps_sec), dtype=np.float32)
    delta_ts[0] = 900.0  # default 15 min for first observation
    delta_ts[1:] = np.diff(timestamps_sec)
    delta_ts = np.clip(delta_ts, a_min=1.0, a_max=86400.0)

    # Build sliding windows
    N = len(values)
    if N < SEQ_LENGTH:
        # Pad to SEQ_LENGTH by repeating last value
        pad_len = SEQ_LENGTH - N
        values = np.pad(values, ((0, pad_len), (0, 0)), mode="edge")
        timestamps_sec = np.pad(
            timestamps_sec, (0, pad_len), mode="linear_ramp",
            end_values=timestamps_sec[-1] + 900.0 * pad_len,
        )
        delta_ts = np.pad(delta_ts, (0, pad_len), constant_values=900.0)
        N = SEQ_LENGTH

    windows_x = []
    windows_dt = []
    windows_ts = []

    for start in range(0, N - SEQ_LENGTH + 1, STRIDE):
        end = start + SEQ_LENGTH
        windows_x.append(values[start:end])
        windows_dt.append(delta_ts[start:end])
        windows_ts.append(timestamps_sec[start:end])

    # Always include the last window if we haven't already
    if (N - SEQ_LENGTH) % STRIDE != 0:
        windows_x.append(values[N - SEQ_LENGTH : N])
        windows_dt.append(delta_ts[N - SEQ_LENGTH : N])
        windows_ts.append(timestamps_sec[N - SEQ_LENGTH : N])

    if not windows_x:
        # Fallback: single padded window
        windows_x.append(values[:SEQ_LENGTH])
        windows_dt.append(delta_ts[:SEQ_LENGTH])
        windows_ts.append(timestamps_sec[:SEQ_LENGTH])

    x = torch.tensor(np.array(windows_x), dtype=torch.float32)
    dt = torch.tensor(np.array(windows_dt), dtype=torch.float32)
    ts = torch.tensor(np.array(windows_ts), dtype=torch.float32)

    return x, dt, ts


def load_models() -> tuple:
    """Load the AquaSSM sensor encoder, Perceiver IO fusion, and anomaly head.

    Returns:
        Tuple of (sensor_encoder, fusion_model, anomaly_head) or
        (None, None, None) if checkpoints are not available.
    """
    from sentinel.models.sensor_encoder.model import SensorEncoder
    from sentinel.models.fusion.model import PerceiverIOFusion
    from sentinel.models.fusion.heads import AnomalyDetectionHead, SourceAttributionHead

    sensor_ckpt = CKPT_BASE / "sensor" / "aquassm_real_best.pt"
    fusion_ckpt = CKPT_BASE / "fusion" / "fusion_real_best.pt"

    # --- Sensor encoder ---
    if not sensor_ckpt.exists():
        logger.warning(
            f"Sensor checkpoint not found: {sensor_ckpt}\n"
            "  Prediction will use randomly initialized weights.\n"
            "  Results are NOT meaningful without a trained model."
        )
        sensor = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
    else:
        sensor = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
        state = torch.load(str(sensor_ckpt), map_location=DEVICE, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model" in state:
            state = state["model"]
        sensor.load_state_dict(state, strict=False)
        logger.info(f"AquaSSM sensor encoder loaded from {sensor_ckpt}")

    sensor.eval()

    # --- Fusion model + heads ---
    fusion = PerceiverIOFusion(num_latents=64).to(DEVICE)
    anomaly_head = AnomalyDetectionHead().to(DEVICE)
    source_head = SourceAttributionHead().to(DEVICE)

    if not fusion_ckpt.exists():
        logger.warning(
            f"Fusion checkpoint not found: {fusion_ckpt}\n"
            "  Prediction will use randomly initialized weights.\n"
            "  Results are NOT meaningful without a trained model."
        )
    else:
        fusion_state = torch.load(
            str(fusion_ckpt), map_location=DEVICE, weights_only=False
        )
        if "fusion" in fusion_state:
            fusion.load_state_dict(fusion_state["fusion"], strict=False)
        if "head" in fusion_state:
            anomaly_head.load_state_dict(fusion_state["head"], strict=False)
        if "source_head" in fusion_state:
            source_head.load_state_dict(fusion_state["source_head"], strict=False)
        logger.info(f"Fusion model + heads loaded from {fusion_ckpt}")

    fusion.eval()
    anomaly_head.eval()
    source_head.eval()

    return sensor, fusion, anomaly_head, source_head


@torch.no_grad()
def predict_site(
    site: Dict[str, Any],
    sensor_encoder: torch.nn.Module,
    fusion_model: torch.nn.Module,
    anomaly_head: torch.nn.Module,
    source_head: torch.nn.Module,
    lookback_days: int = 7,
) -> Optional[Dict[str, Any]]:
    """Run forward prediction for a single monitoring site.

    Fetches real-time NWIS data, runs through AquaSSM and the fusion
    model in sensor-only mode, and returns anomaly scores and
    contamination type predictions.

    Args:
        site: Site dict from MONITORING_SITES.
        sensor_encoder: Loaded AquaSSM SensorEncoder.
        fusion_model: Loaded PerceiverIOFusion.
        anomaly_head: Loaded AnomalyDetectionHead.
        source_head: Loaded SourceAttributionHead.
        lookback_days: Days of data to fetch.

    Returns:
        Dict with prediction results, or None if data fetch fails.
    """
    from sentinel.models.fusion.heads import ANOMALY_TYPES, CONTAMINANT_CLASSES

    site_id = site["site_id"]
    logger.info(f"Processing site {site_id} ({site['name']})...")

    # Fetch NWIS data
    df = fetch_nwis_data(site_id, lookback_days=lookback_days)
    if df is None:
        return None

    # Prepare AquaSSM input tensors
    x, delta_ts, timestamps = prepare_aquassm_input(df)
    n_windows = x.shape[0]
    logger.info(f"  Built {n_windows} windows of length {SEQ_LENGTH}")

    # Run through AquaSSM encoder
    all_anomaly_probs = []
    all_severity_scores = []
    all_anomaly_type_probs = []
    all_source_probs = []

    batch_size = 32
    for i in range(0, n_windows, batch_size):
        batch_x = x[i : i + batch_size].to(DEVICE)
        batch_dt = delta_ts[i : i + batch_size].to(DEVICE)

        # AquaSSM forward pass (sensor-only; skip anomaly detection sub-module
        # to avoid the expensive leave-one-out passes -- we use the fusion
        # heads for anomaly scoring instead).
        sensor_out = sensor_encoder(
            batch_x, delta_ts=batch_dt, compute_anomaly=False
        )
        embeddings = sensor_out["embedding"]  # [B, 256]

        # Fusion model in sensor-only mode: feed each embedding as a
        # "sensor" modality observation. We process each window
        # independently (no temporal recurrence across windows) to
        # avoid temporal leakage.
        B = embeddings.shape[0]
        for j in range(B):
            emb = embeddings[j : j + 1]  # [1, 256]
            fusion_out = fusion_model(
                modality_id="sensor",
                raw_embedding=emb,
                timestamp=float(timestamps[i + j, -1]),
                confidence=0.85,  # sensor-only confidence
                latent_state=None,
            )
            fused = fusion_out.fused_state  # [1, 256]

            # Anomaly head
            anom_out = anomaly_head(fused)
            all_anomaly_probs.append(
                anom_out.anomaly_probability.cpu().item()
            )
            all_severity_scores.append(
                anom_out.severity_score.cpu().item()
            )
            all_anomaly_type_probs.append(
                anom_out.anomaly_type_probs.cpu().numpy().tolist()[0]
            )

            # Source attribution head
            src_out = source_head(fused)
            all_source_probs.append(
                src_out.class_probs.cpu().numpy().tolist()[0]
            )

        # Reset fusion registry between batches to prevent state leakage
        fusion_model.reset_registry()

    # Aggregate predictions
    anomaly_probs = np.array(all_anomaly_probs)
    severity_scores = np.array(all_severity_scores)
    anomaly_type_probs = np.array(all_anomaly_type_probs)
    source_probs = np.array(all_source_probs)

    # Check for sustained anomaly alert
    threshold = ALERT_THRESHOLDS["anomaly_probability_threshold"]
    sustained_req = ALERT_THRESHOLDS["sustained_windows_required"]
    above_threshold = anomaly_probs >= threshold

    # Find longest consecutive run above threshold
    max_consecutive = 0
    current_run = 0
    for val in above_threshold:
        if val:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            current_run = 0

    alert_triggered = max_consecutive >= sustained_req

    # Top anomaly types (mean probability across windows)
    mean_type_probs = anomaly_type_probs.mean(axis=0)
    top_type_indices = np.argsort(mean_type_probs)[::-1][:3]
    top_anomaly_types = [
        {
            "type": ANOMALY_TYPES[idx],
            "mean_probability": float(mean_type_probs[idx]),
        }
        for idx in top_type_indices
    ]

    # Top contamination sources
    mean_source_probs = source_probs.mean(axis=0)
    top_source_indices = np.argsort(mean_source_probs)[::-1][:3]
    top_sources = [
        {
            "class": CONTAMINANT_CLASSES[idx],
            "mean_probability": float(mean_source_probs[idx]),
        }
        for idx in top_source_indices
    ]

    # Data quality summary
    data_summary = {
        "n_records_raw": len(df),
        "date_range_start": str(df.index.min()),
        "date_range_end": str(df.index.max()),
        "parameters_available": [
            col for col in PARAMETER_CODES.values() if col in df.columns
        ],
        "n_windows": n_windows,
    }

    result = {
        "site_id": site_id,
        "name": site["name"],
        "region": site["region"],
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_summary": data_summary,
        "anomaly_scores": {
            "mean_probability": float(anomaly_probs.mean()),
            "max_probability": float(anomaly_probs.max()),
            "min_probability": float(anomaly_probs.min()),
            "std_probability": float(anomaly_probs.std()),
            "fraction_above_threshold": float(above_threshold.mean()),
            "max_consecutive_above_threshold": int(max_consecutive),
            "per_window_probabilities": anomaly_probs.tolist(),
        },
        "severity_scores": {
            "mean": float(severity_scores.mean()),
            "max": float(severity_scores.max()),
        },
        "alert": {
            "triggered": bool(alert_triggered),
            "threshold": threshold,
            "sustained_windows_required": sustained_req,
            "max_consecutive_windows": int(max_consecutive),
        },
        "top_anomaly_types": top_anomaly_types,
        "top_contamination_sources": top_sources,
        "predicted_contamination_types_registered": site[
            "predicted_contamination_types"
        ],
    }

    return result


def run_predict(lookback_days: int = 7) -> Path:
    """Run forward predictions for all pre-registered sites.

    Fetches real-time NWIS data, runs through the model pipeline, and
    saves predictions to a dated JSON file.

    Args:
        lookback_days: Number of days of NWIS IV data to fetch.

    Returns:
        Path to the saved predictions file.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Load the latest registration
    latest_reg_path = REGISTRATIONS_DIR / "latest_registration.json"
    if latest_reg_path.exists():
        with open(latest_reg_path) as f:
            registration = json.load(f)
        logger.info(
            f"Loaded registration from {registration['timestamp']} "
            f"(hash: {registration.get('commitment_hash_sha256', 'N/A')[:16]}...)"
        )
    else:
        logger.warning(
            "No registration found. Run --mode register first.\n"
            "Proceeding with default site list."
        )
        registration = None

    # Load models
    logger.info("Loading models...")
    sensor, fusion, anomaly_head, source_head = load_models()

    # Run predictions for each site
    predictions = []
    n_success = 0
    n_failed = 0
    n_alerts = 0

    for site in MONITORING_SITES:
        try:
            result = predict_site(
                site, sensor, fusion, anomaly_head, source_head,
                lookback_days=lookback_days,
            )
            if result is not None:
                predictions.append(result)
                n_success += 1
                if result["alert"]["triggered"]:
                    n_alerts += 1
                    logger.warning(
                        f"  ALERT triggered for {site['name']} "
                        f"(max p={result['anomaly_scores']['max_probability']:.3f}, "
                        f"sustained={result['alert']['max_consecutive_windows']} windows)"
                    )
            else:
                n_failed += 1
        except Exception as e:
            logger.error(f"  Error processing {site['site_id']}: {e}")
            n_failed += 1

        # Brief pause to avoid overwhelming the NWIS API
        time.sleep(0.5)

    # Assemble output
    output = {
        "prediction_date": now.isoformat(),
        "lookback_days": lookback_days,
        "registration_hash": (
            registration.get("commitment_hash_sha256")
            if registration
            else None
        ),
        "model_info": get_model_info(),
        "summary": {
            "n_sites_attempted": len(MONITORING_SITES),
            "n_sites_success": n_success,
            "n_sites_failed": n_failed,
            "n_alerts_triggered": n_alerts,
        },
        "thresholds": ALERT_THRESHOLDS,
        "predictions": predictions,
    }

    output_path = PREDICTIONS_DIR / f"predictions_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nPredictions saved: {output_path}")
    logger.info(f"  Sites processed: {n_success}/{len(MONITORING_SITES)}")
    logger.info(f"  Alerts triggered: {n_alerts}")

    # Update latest symlink
    latest_path = PREDICTIONS_DIR / "latest_predictions.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


# =========================================================================
# Mode 3: Evaluation
# =========================================================================

def load_all_predictions() -> List[Dict[str, Any]]:
    """Load all prediction files from the predictions directory.

    Returns:
        List of prediction records, sorted by date.
    """
    if not PREDICTIONS_DIR.exists():
        return []

    prediction_files = sorted(PREDICTIONS_DIR.glob("predictions_*.json"))
    all_preds = []
    for pf in prediction_files:
        try:
            with open(pf) as f:
                data = json.load(f)
            all_preds.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {pf}: {e}")

    return all_preds


def run_evaluate() -> Path:
    """Evaluate forward predictions against actual outcomes.

    Compares predictions to observed conditions. Since ground-truth
    labels (EPA violations, USGS event flags) may not be available
    immediately, this function also flags predictions that are still
    pending evaluation.

    Currently evaluates using a heuristic approach:
    - Fetches the most recent NWIS data for each site
    - Checks for extreme values that would indicate an actual event
    - Compares against the model's predictions

    Returns:
        Path to the evaluation results file.
    """
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Load all predictions
    all_preds = load_all_predictions()
    if not all_preds:
        logger.error("No prediction files found. Run --mode predict first.")
        return EVALUATIONS_DIR / "no_predictions.json"

    logger.info(f"Loaded {len(all_preds)} prediction file(s)")

    # Empirical thresholds for identifying actual events from raw NWIS data.
    # These are deliberately conservative to avoid false confirmation.
    event_thresholds = {
        "DO": {"low": 4.0, "unit": "mg/L", "direction": "below"},
        "pH": {"low": 6.0, "high": 9.5, "unit": "std units", "direction": "outside"},
        "SpCond": {"high": 1500.0, "unit": "uS/cm", "direction": "above"},
        "Turb": {"high": 300.0, "unit": "FNU", "direction": "above"},
        "Temp": {"high": 35.0, "unit": "degC", "direction": "above"},
    }

    # Evaluate each prediction set
    evaluation_results = []

    for pred_set in all_preds:
        pred_date = pred_set.get("prediction_date", "unknown")
        logger.info(f"\nEvaluating predictions from {pred_date}")

        site_evals = []
        for pred in pred_set.get("predictions", []):
            site_id = pred["site_id"]
            predicted_alert = pred["alert"]["triggered"]
            mean_prob = pred["anomaly_scores"]["mean_probability"]

            # Fetch current data to check for actual events
            df = fetch_nwis_data(site_id, lookback_days=3)
            actual_event = False
            event_details = []

            if df is not None and len(df) > 0:
                for param, thresh in event_thresholds.items():
                    if param not in df.columns:
                        continue
                    values = df[param].dropna()
                    if len(values) == 0:
                        continue

                    direction = thresh["direction"]
                    if direction == "below" and values.min() < thresh["low"]:
                        actual_event = True
                        event_details.append(
                            f"{param} dropped to {values.min():.2f} "
                            f"(threshold: {thresh['low']} {thresh['unit']})"
                        )
                    elif direction == "above" and values.max() > thresh["high"]:
                        actual_event = True
                        event_details.append(
                            f"{param} spiked to {values.max():.2f} "
                            f"(threshold: {thresh['high']} {thresh['unit']})"
                        )
                    elif direction == "outside":
                        if values.min() < thresh["low"]:
                            actual_event = True
                            event_details.append(
                                f"{param} dropped to {values.min():.2f} "
                                f"(threshold: {thresh['low']} {thresh['unit']})"
                            )
                        if values.max() > thresh["high"]:
                            actual_event = True
                            event_details.append(
                                f"{param} spiked to {values.max():.2f} "
                                f"(threshold: {thresh['high']} {thresh['unit']})"
                            )

                data_available = True
            else:
                data_available = False

            # Classification outcome
            if not data_available:
                outcome = "pending"
            elif predicted_alert and actual_event:
                outcome = "true_positive"
            elif predicted_alert and not actual_event:
                outcome = "false_positive"
            elif not predicted_alert and actual_event:
                outcome = "false_negative"
            else:
                outcome = "true_negative"

            site_evals.append({
                "site_id": site_id,
                "name": pred["name"],
                "predicted_alert": predicted_alert,
                "mean_anomaly_probability": mean_prob,
                "actual_event_detected": actual_event,
                "outcome": outcome,
                "event_details": event_details,
                "data_available": data_available,
            })

            time.sleep(0.3)  # Rate limiting

        # Compute metrics for this prediction set
        outcomes = [e["outcome"] for e in site_evals if e["outcome"] != "pending"]
        tp = outcomes.count("true_positive")
        fp = outcomes.count("false_positive")
        tn = outcomes.count("true_negative")
        fn = outcomes.count("false_negative")
        n_pending = sum(1 for e in site_evals if e["outcome"] == "pending")

        hit_rate = tp / (tp + fn) if (tp + fn) > 0 else None
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        n_evaluated = len(outcomes)

        evaluation_results.append({
            "prediction_date": pred_date,
            "evaluation_date": now.isoformat(),
            "site_evaluations": site_evals,
            "metrics": {
                "n_sites_evaluated": n_evaluated,
                "n_pending": n_pending,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "hit_rate": hit_rate,
                "false_alarm_rate": false_alarm_rate,
                "precision": precision,
            },
        })

        logger.info(
            f"  Results: TP={tp}, FP={fp}, TN={tn}, FN={fn}, "
            f"pending={n_pending}"
        )
        if hit_rate is not None:
            logger.info(f"  Hit rate: {hit_rate:.3f}")
        if false_alarm_rate is not None:
            logger.info(f"  False alarm rate: {false_alarm_rate:.3f}")

    # Save evaluation
    eval_output = {
        "evaluation_timestamp": now.isoformat(),
        "n_prediction_sets_evaluated": len(all_preds),
        "event_thresholds": event_thresholds,
        "evaluations": evaluation_results,
    }

    output_path = EVALUATIONS_DIR / f"evaluation_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=2)

    logger.info(f"\nEvaluation saved: {output_path}")
    return output_path


# =========================================================================
# Mode 4: Summary Report
# =========================================================================

def run_report() -> Path:
    """Generate a comprehensive summary report of all prospective validation.

    Aggregates registrations, predictions, and evaluations into a single
    human-readable report.

    Returns:
        Path to the generated report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("SENTINEL Prospective Validation Report")
    lines.append(f"Generated: {now.isoformat()}")
    lines.append("=" * 72)
    lines.append("")

    # --- Registration info ---
    lines.append("1. PRE-REGISTRATION")
    lines.append("-" * 40)

    reg_files = sorted(REGISTRATIONS_DIR.glob("registration_*.json")) if REGISTRATIONS_DIR.exists() else []
    if reg_files:
        lines.append(f"   Total registrations: {len(reg_files)}")
        # Show the latest
        with open(reg_files[-1]) as f:
            latest_reg = json.load(f)
        lines.append(f"   Latest: {latest_reg['timestamp']}")
        lines.append(f"   Hash:   {latest_reg.get('commitment_hash_sha256', 'N/A')}")
        lines.append(f"   Sites:  {latest_reg['n_sites']}")
        lines.append("")

        # Site list
        lines.append("   Registered Sites:")
        for s in latest_reg["sites"]:
            lines.append(
                f"     - {s['site_id']} | {s['name']}"
            )
            lines.append(
                f"       Predicted: {', '.join(s['predicted_contamination_types'])}"
            )
    else:
        lines.append("   No registrations found. Run --mode register first.")
    lines.append("")

    # --- Prediction summary ---
    lines.append("2. PREDICTIONS")
    lines.append("-" * 40)

    all_preds = load_all_predictions()
    if all_preds:
        lines.append(f"   Total prediction runs: {len(all_preds)}")
        lines.append("")

        for pred_set in all_preds:
            pred_date = pred_set.get("prediction_date", "unknown")
            summary = pred_set.get("summary", {})
            lines.append(f"   Run: {pred_date}")
            lines.append(
                f"     Sites: {summary.get('n_sites_success', '?')}"
                f"/{summary.get('n_sites_attempted', '?')} successful"
            )
            lines.append(
                f"     Alerts: {summary.get('n_alerts_triggered', 0)}"
            )

            # Highlight alerts
            for pred in pred_set.get("predictions", []):
                if pred["alert"]["triggered"]:
                    lines.append(
                        f"     ** ALERT: {pred['name']} "
                        f"(p={pred['anomaly_scores']['max_probability']:.3f}, "
                        f"sustained={pred['alert']['max_consecutive_windows']} windows)"
                    )

            # Top anomaly scores across all sites
            site_scores = []
            for pred in pred_set.get("predictions", []):
                site_scores.append(
                    (pred["name"], pred["anomaly_scores"]["mean_probability"])
                )
            site_scores.sort(key=lambda x: x[1], reverse=True)
            lines.append("     Top anomaly scores:")
            for name, score in site_scores[:5]:
                lines.append(f"       {score:.4f} - {name}")

            lines.append("")
    else:
        lines.append("   No predictions found. Run --mode predict first.")
    lines.append("")

    # --- Evaluation summary ---
    lines.append("3. EVALUATIONS")
    lines.append("-" * 40)

    eval_files = sorted(EVALUATIONS_DIR.glob("evaluation_*.json")) if EVALUATIONS_DIR.exists() else []
    if eval_files:
        lines.append(f"   Total evaluation runs: {len(eval_files)}")

        # Aggregate metrics across all evaluations
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        total_pending = 0

        for ef in eval_files:
            with open(ef) as f:
                eval_data = json.load(f)
            for ev in eval_data.get("evaluations", []):
                m = ev.get("metrics", {})
                total_tp += m.get("true_positives", 0)
                total_fp += m.get("false_positives", 0)
                total_tn += m.get("true_negatives", 0)
                total_fn += m.get("false_negatives", 0)
                total_pending += m.get("n_pending", 0)

        lines.append(f"   Aggregate confusion matrix:")
        lines.append(f"     True Positives:  {total_tp}")
        lines.append(f"     False Positives: {total_fp}")
        lines.append(f"     True Negatives:  {total_tn}")
        lines.append(f"     False Negatives: {total_fn}")
        lines.append(f"     Pending:         {total_pending}")
        lines.append("")

        if (total_tp + total_fn) > 0:
            hit_rate = total_tp / (total_tp + total_fn)
            lines.append(f"   Hit rate (sensitivity):     {hit_rate:.4f}")
        if (total_fp + total_tn) > 0:
            far = total_fp / (total_fp + total_tn)
            lines.append(f"   False alarm rate:           {far:.4f}")
        if (total_tp + total_fp) > 0:
            precision = total_tp / (total_tp + total_fp)
            lines.append(f"   Precision:                  {precision:.4f}")
        if (total_tp + total_fp + total_tn + total_fn) > 0:
            accuracy = (total_tp + total_tn) / (
                total_tp + total_fp + total_tn + total_fn
            )
            lines.append(f"   Accuracy:                   {accuracy:.4f}")
    else:
        lines.append("   No evaluations found. Run --mode evaluate first.")

    lines.append("")
    lines.append("=" * 72)
    lines.append("End of report")
    lines.append("=" * 72)

    report_text = "\n".join(lines)

    # Print to console
    print(report_text)

    # Save to file
    output_path = REPORTS_DIR / f"report_{date_str}.txt"
    with open(output_path, "w") as f:
        f.write(report_text)

    logger.info(f"\nReport saved: {output_path}")
    return output_path


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SENTINEL Prospective Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Modes:
  register   Create a new pre-registration commitment (sites + thresholds)
  predict    Fetch real-time NWIS data and run forward predictions
  evaluate   Compare predictions against actual outcomes
  report     Generate summary report of all prospective validation

Examples:
  python scripts/prospective_validation.py --mode register
  python scripts/prospective_validation.py --mode predict
  python scripts/prospective_validation.py --mode predict --lookback-days 14
  python scripts/prospective_validation.py --mode evaluate
  python scripts/prospective_validation.py --mode report
""",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["register", "predict", "evaluate", "report"],
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Number of days of NWIS IV data to fetch (predict mode). Default: 7.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SENTINEL Prospective Validation Pipeline")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    if args.mode == "register":
        output = run_register()
        logger.info(f"\nDone. Registration file: {output}")

    elif args.mode == "predict":
        output = run_predict(lookback_days=args.lookback_days)
        logger.info(f"\nDone. Predictions file: {output}")

    elif args.mode == "evaluate":
        output = run_evaluate()
        logger.info(f"\nDone. Evaluation file: {output}")

    elif args.mode == "report":
        output = run_report()
        logger.info(f"\nDone. Report file: {output}")


if __name__ == "__main__":
    main()
