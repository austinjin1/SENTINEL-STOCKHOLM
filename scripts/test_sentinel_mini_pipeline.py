#!/usr/bin/env python3
"""End-to-end SENTINEL Mini → Full SENTINEL trigger pipeline test.

Simulates the complete operational flow:
  1. Drone flies over water body, captures multispectral imagery
  2. WaterDroneNet predicts water quality from imagery
  3. AnomalyScorer evaluates predictions → alert level
  4. If alert >= WARNING: StationSelector picks nearest USGS stations
  5. RFController sends trigger command
  6. Full SENTINEL station provides real sensor measurements
  7. Confirmation engine compares drone prediction vs ground truth
  8. Report: confirmed, partial, or refuted

Uses real Sentinel-2 imagery (cached from crisis test) and real USGS
sensor data from nearby stations as ground truth.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_PATH = PROJECT / "checkpoints" / "waterdronenet" / "waterdronenet_best.pt"
CRISIS_TILES = PROJECT / "data" / "processed" / "satellite" / "crisis_tiles"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"
RESULTS_DIR = PROJECT / "results"

TARGET_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]
PARAM_RANGES = {"DO": (0, 20), "pH": (4, 10), "Turb": (0, 1000),
                "Temp": (-5, 45), "SpCond": (0, 50000)}

# Map WaterDroneNet targets to USGS sensor column names
WDN_TO_SENSOR = {
    "DO": "DO",
    "pH": "pH",
    "Turb": "Turb",
    "Temp": "Temp",
    "SpCond": "SpCond",
}

# Case study events with nearby USGS stations
EVENTS = [
    {
        "event_id": "lake_erie_hab_2023",
        "name": "Lake Erie HAB 2023",
        "advisory_date": "2023-07-15",
        "lat": 41.50, "lon": -82.90,
        "type": "HAB",
    },
    {
        "event_id": "chesapeake_hypoxia_2018",
        "name": "Chesapeake Bay Hypoxia 2018",
        "advisory_date": "2018-07-20",
        "lat": 39.20, "lon": -76.50,
        "type": "hypoxia",
    },
    {
        "event_id": "klamath_river_hab_2021",
        "name": "Klamath River HAB 2021",
        "advisory_date": "2021-08-01",
        "lat": 41.55, "lon": -122.30,
        "type": "HAB",
    },
    {
        "event_id": "jordan_lake_hab_nc",
        "name": "Jordan Lake HAB NC",
        "advisory_date": "2022-07-15",
        "lat": 35.78, "lon": -79.06,
        "type": "HAB",
    },
    {
        "event_id": "mississippi_salinity_2023",
        "name": "Mississippi River Salinity 2023",
        "advisory_date": "2023-10-01",
        "lat": 29.95, "lon": -90.06,
        "type": "salinity",
    },
    {
        "event_id": "gulf_dead_zone_2023",
        "name": "Gulf Dead Zone 2023",
        "advisory_date": "2023-07-01",
        "lat": 29.50, "lon": -90.50,
        "type": "hypoxia",
    },
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_station_network():
    """Load all USGS stations with GPS coords and sensor data."""
    stations = {}
    sensor_ids = {f.stem for f in SENSOR_DIR.glob("*.parquet")}

    for f in sorted(SITE_INFO_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            sid = d["site_no"]
            if sid not in sensor_ids:
                continue
            stations[sid] = {
                "lat": float(d["lat"]),
                "lon": float(d["lon"]),
                "name": d.get("station_nm", ""),
            }
        except Exception:
            continue

    return stations


def find_nearest_stations(lat, lon, stations, k=5, max_km=50):
    """Find K nearest stations within max_km."""
    dists = []
    for sid, info in stations.items():
        d = haversine_km(lat, lon, info["lat"], info["lon"])
        if d <= max_km:
            dists.append((sid, d, info))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def get_station_measurements(station_id, date_str, window_days=3):
    """Get real USGS measurements for a station near a date."""
    fpath = SENSOR_DIR / f"{station_id}.parquet"
    if not fpath.exists():
        return None

    try:
        df = pd.read_parquet(fpath)
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        df.index = pd.to_datetime(df.index, utc=True)

        dt = pd.Timestamp(date_str, tz="UTC")
        start = dt - pd.Timedelta(days=window_days)
        end = dt + pd.Timedelta(days=window_days)

        window = df.loc[start:end]
        if len(window) == 0:
            return None

        daily = window.resample("1D").mean()

        measurements = {}
        for wdn_col, sensor_col in WDN_TO_SENSOR.items():
            if sensor_col in daily.columns:
                vals = daily[sensor_col].dropna()
                if len(vals) > 0:
                    val = float(vals.mean())
                    lo, hi = PARAM_RANGES.get(wdn_col, (float("-inf"), float("inf")))
                    if lo <= val <= hi:
                        measurements[wdn_col] = val

        return measurements if measurements else None
    except Exception:
        return None


def load_model(device):
    """Load WaterDroneNet."""
    from sentinel.models.waterdronenet import WaterDroneNet

    model = WaterDroneNet()
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    target_mean = ckpt.get("target_mean", np.zeros(5))
    target_std = ckpt.get("target_std", np.ones(5))

    return model, target_mean, target_std


@torch.no_grad()
def predict_from_image(model, image, target_mean, target_std, device):
    """Run WaterDroneNet on a single image."""
    img_t = torch.from_numpy(image).unsqueeze(0).to(device)
    with autocast("cuda"):
        out = model(img_t)
    mu = out["mu"].cpu().float().squeeze(0).numpy()
    sigma = out["sigma"].cpu().float().squeeze(0).numpy()
    mu_real = mu * target_std + target_mean
    sigma_real = sigma * np.abs(target_std)
    return {
        "predictions": {col: float(mu_real[i]) for i, col in enumerate(TARGET_COLS)},
        "uncertainties": {col: float(sigma_real[i]) for i, col in enumerate(TARGET_COLS)},
    }


def main():
    from sentinel.platform.sentinel_mini_trigger import (
        AnomalyScorer, StationSelector, RFController, SentinelMiniTriggerSystem,
        FixedStation, StationMode, ConfirmationResult,
    )

    log("=" * 65)
    log("SENTINEL Mini → Full SENTINEL End-to-End Pipeline Test")
    log("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Load WaterDroneNet
    model, target_mean, target_std = load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"WaterDroneNet loaded ({n_params:,} params)")

    # Load station network
    all_stations = load_station_network()
    log(f"USGS station network: {len(all_stations)} stations")

    # Build FixedStation objects for trigger system
    fixed_stations = [
        FixedStation(
            station_id=sid,
            lat=info["lat"],
            lon=info["lon"],
            name=info["name"],
            mode=StationMode.STANDBY,
            capabilities=["sensor", "satellite"],
        )
        for sid, info in all_stations.items()
    ]

    # Create trigger system with 50km range (extended for testing)
    trigger_system = SentinelMiniTriggerSystem(
        stations=fixed_stations,
        k_stations=3,
        max_range_m=50_000,  # 50 km for testing (LoRa extended range)
        log_dir=PROJECT / "logs" / "pipeline_test_triggers",
    )

    log(f"Trigger system ready: {len(fixed_stations)} stations, 50km range")

    results = {"pipeline_tests": [], "summary": {}}
    total_tests = 0
    confirmations = {"confirmed": 0, "partial": 0, "refuted": 0, "no_data": 0, "no_trigger": 0}

    for event in EVENTS:
        log(f"\n{'━' * 60}")
        log(f"EVENT: {event['name']}")
        log(f"Location: ({event['lat']}, {event['lon']})")
        log(f"Advisory: {event['advisory_date']}")

        # Find cached tiles for this event
        event_tiles = sorted(CRISIS_TILES.glob(f"{event['event_id']}_*.npz"))
        if not event_tiles:
            log(f"  No cached imagery — skipping")
            continue

        # Find nearest USGS stations
        nearby = find_nearest_stations(
            event["lat"], event["lon"], all_stations, k=5, max_km=50
        )
        if not nearby:
            log(f"  No USGS stations within 50km — skipping")
            continue

        log(f"  Cached imagery: {len(event_tiles)} tiles")
        log(f"  Nearest stations:")
        for sid, dist, info in nearby:
            log(f"    {sid} ({info['name'][:40]}): {dist:.1f} km")

        # Test each image through the full pipeline
        event_result = {
            "event_id": event["event_id"],
            "name": event["name"],
            "type": event["type"],
            "nearby_stations": [(sid, round(d, 1)) for sid, d, _ in nearby],
            "pipeline_runs": [],
        }

        for tile_path in event_tiles:
            # Parse date from filename
            date_str = tile_path.stem.split("_")[-1]
            if len(date_str) != 10:
                # Handle event IDs with underscores
                parts = tile_path.stem.replace(event["event_id"] + "_", "")
                date_str = parts

            total_tests += 1

            # 1. Load imagery
            img = np.load(tile_path)["image"]

            # 2. WaterDroneNet prediction
            pred = predict_from_image(model, img, target_mean, target_std, device)

            # 3. Process through trigger system
            trigger_result = trigger_system.process_drone_detection(
                predictions=pred["predictions"],
                uncertainties=pred["uncertainties"],
                lat=event["lat"],
                lon=event["lon"],
                altitude_m=50.0,
                drone_id="MINI-TEST",
                flight_id=f"test_{event['event_id']}",
            )

            alert_level = trigger_result["alert_level"]
            stations_triggered = trigger_result["stations_triggered"]

            # 4. Get ground truth from triggered (or nearest) stations
            confirm_stations = stations_triggered if stations_triggered else [nearby[0][0]]
            ground_truth = {}
            for sid in confirm_stations[:3]:
                measurements = get_station_measurements(sid, date_str)
                if measurements:
                    ground_truth[sid] = measurements

            # 5. Compare drone predictions vs ground truth
            run_status = "no_data"
            confirmed_params = []
            refuted_params = []

            if ground_truth:
                for sid, measured in ground_truth.items():
                    for param, actual_val in measured.items():
                        predicted_val = pred["predictions"].get(param)
                        uncertainty = pred["uncertainties"].get(param, 999)
                        if predicted_val is None:
                            continue

                        error = abs(predicted_val - actual_val)
                        # Confirmed if prediction within 2 sigma of actual
                        if error <= 2 * uncertainty:
                            confirmed_params.append(param)
                        else:
                            refuted_params.append(param)

                if len(confirmed_params) > 0 and len(confirmed_params) >= len(refuted_params):
                    run_status = "confirmed"
                elif len(confirmed_params) > 0:
                    run_status = "partial"
                elif len(refuted_params) > 0:
                    run_status = "refuted"

                # 6. Process confirmation through trigger system
                if stations_triggered and ground_truth:
                    for sid in stations_triggered:
                        if sid in ground_truth:
                            confirmation = ConfirmationResult(
                                station_id=sid,
                                command_id=trigger_result["commands_sent"][0]["command_id"] if trigger_result["commands_sent"] else "test",
                                detection_confirmed=(run_status in ("confirmed", "partial")),
                                confidence=len(confirmed_params) / max(len(confirmed_params) + len(refuted_params), 1),
                                measured_values=ground_truth[sid],
                                anomaly_confirmed={
                                    p: (p in confirmed_params)
                                    for p in ground_truth[sid].keys()
                                },
                            )
                            conf_result = trigger_system.process_confirmation(confirmation)
                            log(f"    Confirmation: {conf_result['status']} "
                                f"({conf_result['confidence']:.0%} confidence) "
                                f"→ {conf_result['recommendation']}")

            if not stations_triggered:
                run_status = "no_trigger"
                confirmations["no_trigger"] += 1
            elif not ground_truth:
                confirmations["no_data"] += 1
            else:
                confirmations[run_status] += 1

            # Log result
            flag = {
                "confirmed": "+", "partial": "~", "refuted": "x",
                "no_data": "?", "no_trigger": "-"
            }.get(run_status, "?")

            gt_str = ""
            if ground_truth:
                first_gt = list(ground_truth.values())[0]
                gt_parts = [f"{k}={v:.1f}" for k, v in list(first_gt.items())[:4]]
                gt_str = f" | GT: {', '.join(gt_parts)}"

            log(f"  [{flag}] {date_str}: alert={alert_level:8s} | "
                f"pred: DO={pred['predictions']['DO']:.1f} "
                f"Temp={pred['predictions']['Temp']:.1f} "
                f"pH={pred['predictions']['pH']:.1f}"
                f"{gt_str} | "
                f"triggered={len(stations_triggered)} stations | {run_status}")

            event_result["pipeline_runs"].append({
                "date": date_str,
                "alert_level": alert_level,
                "stations_triggered": stations_triggered,
                "predictions": pred["predictions"],
                "ground_truth": ground_truth,
                "status": run_status,
                "confirmed_params": confirmed_params,
                "refuted_params": refuted_params,
            })

        results["pipeline_tests"].append(event_result)

    # Summary
    log(f"\n{'━' * 60}")
    log("PIPELINE TEST SUMMARY")
    log(f"{'━' * 60}")
    log(f"Total drone scans: {total_tests}")
    log(f"Triggers fired: {total_tests - confirmations['no_trigger']}")
    log(f"Confirmations: {confirmations}")
    log(f"  Confirmed (pred within 2σ of actual): {confirmations['confirmed']}")
    log(f"  Partial (some params match): {confirmations['partial']}")
    log(f"  Refuted (pred outside 2σ): {confirmations['refuted']}")
    log(f"  No sensor data available: {confirmations['no_data']}")
    log(f"  No trigger (nominal alert): {confirmations['no_trigger']}")

    total_with_gt = confirmations["confirmed"] + confirmations["partial"] + confirmations["refuted"]
    if total_with_gt > 0:
        accuracy = (confirmations["confirmed"] + confirmations["partial"]) / total_with_gt
        log(f"\nConfirmation rate (where ground truth available): {accuracy:.0%}")

    # System status
    status = trigger_system.get_status()
    log(f"\nTrigger system status:")
    log(f"  Total triggers sent: {status['total_triggers']}")
    log(f"  Total detections logged: {status['total_detections']}")

    results["summary"] = {
        "total_tests": total_tests,
        "confirmations": confirmations,
        "trigger_system_status": status,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / "sentinel_mini_pipeline_test.json"

    def to_ser(obj):
        if isinstance(obj, dict):
            return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_ser(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(to_ser(results), f, indent=2)

    log(f"\nResults saved to {out_path}")
    log("DONE")


if __name__ == "__main__":
    main()
