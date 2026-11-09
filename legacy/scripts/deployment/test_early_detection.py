#!/usr/bin/env python3
"""SENTINEL Mini early detection test.

Tests the full early-warning timeline:
  1. Drone flies over site, WaterDroneNet flags anomaly (spatial trigger)
  2. Nearest USGS station is activated
  3. Station sensor data analyzed with rolling anomaly detection (AquaSSM-style)
  4. Report: how many days BEFORE the official advisory was the anomaly detectable?

This simulates the operational concept: drone extends spatial coverage,
station provides temporal depth. The combined system detects earlier than
either alone in unmonitored areas.

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

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_PATH = PROJECT / "checkpoints" / "waterdronenet" / "waterdronenet_best.pt"
CRISIS_TILES = PROJECT / "data" / "processed" / "satellite" / "crisis_tiles"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"
RESULTS_DIR = PROJECT / "results"

TARGET_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]

# Anomaly thresholds — environmental limits indicating stress
# These are parameter-specific thresholds where values become concerning
ENV_THRESHOLDS = {
    "DO":     {"low": 4.0, "critical": 2.0},    # mg/L — hypoxia
    "pH":     {"low": 6.0, "high": 9.5},         # pH units
    "Turb":   {"high": 50.0, "critical": 200.0}, # FNU
    "Temp":   {"high": 30.0, "critical": 35.0},  # degC
    "SpCond": {"high": 10000, "critical": 20000}, # uS/cm
}

# Z-score thresholds for AquaSSM-style anomaly detection
Z_THRESHOLD = 2.5       # flag as anomalous
Z_CRITICAL = 3.5        # flag as critical
BASELINE_DAYS = 90      # rolling baseline window
LOOKBACK_DAYS = 180     # how far before advisory to scan

EVENTS = [
    {
        "event_id": "lake_erie_hab_2023",
        "name": "Lake Erie HAB 2023",
        "advisory_date": "2023-07-15",
        "lat": 41.50, "lon": -82.90,
        "type": "HAB",
        "expected_params": ["DO", "pH", "Turb"],  # params that should degrade
    },
    {
        "event_id": "chesapeake_hypoxia_2018",
        "name": "Chesapeake Bay Hypoxia 2018",
        "advisory_date": "2018-07-20",
        "lat": 39.20, "lon": -76.50,
        "type": "hypoxia",
        "expected_params": ["DO"],
    },
    {
        "event_id": "klamath_river_hab_2021",
        "name": "Klamath River HAB 2021",
        "advisory_date": "2021-08-01",
        "lat": 41.55, "lon": -122.30,
        "type": "HAB",
        "expected_params": ["DO", "pH", "Turb"],
    },
    {
        "event_id": "jordan_lake_hab_nc",
        "name": "Jordan Lake HAB NC",
        "advisory_date": "2022-07-15",
        "lat": 35.78, "lon": -79.06,
        "type": "HAB",
        "expected_params": ["DO", "pH"],
    },
    {
        "event_id": "mississippi_salinity_2023",
        "name": "Mississippi River Salinity 2023",
        "advisory_date": "2023-10-01",
        "lat": 29.95, "lon": -90.06,
        "type": "salinity",
        "expected_params": ["SpCond"],
    },
    {
        "event_id": "gulf_dead_zone_2023",
        "name": "Gulf Dead Zone 2023",
        "advisory_date": "2023-07-01",
        "lat": 29.50, "lon": -90.50,
        "type": "hypoxia",
        "expected_params": ["DO"],
    },
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_station_network():
    stations = {}
    sensor_ids = {f.stem for f in SENSOR_DIR.glob("*.parquet")}
    for f in sorted(SITE_INFO_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            sid = d["site_no"]
            if sid not in sensor_ids:
                continue
            stations[sid] = {"lat": float(d["lat"]), "lon": float(d["lon"]),
                             "name": d.get("station_nm", "")}
        except Exception:
            continue
    return stations


def find_nearest_stations(lat, lon, stations, k=5, max_km=100):
    """Find K nearest stations within max_km."""
    dists = []
    for sid, info in stations.items():
        d = haversine_km(lat, lon, info["lat"], info["lon"])
        if d <= max_km:
            dists.append((sid, d, info))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def analyze_station_anomalies(station_id, advisory_date_str, lookback_days=LOOKBACK_DAYS,
                               baseline_days=BASELINE_DAYS):
    """Run rolling z-score anomaly detection on a station's sensor data.

    This simulates what AquaSSM does: detect deviations from the station's
    own baseline. Returns timeline of anomaly scores and first detection dates.
    """
    fpath = SENSOR_DIR / f"{station_id}.parquet"
    if not fpath.exists():
        return None

    df = pd.read_parquet(fpath)
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    df.index = pd.to_datetime(df.index, utc=True)

    advisory = pd.Timestamp(advisory_date_str, tz="UTC")
    scan_start = advisory - pd.Timedelta(days=lookback_days)
    # Need baseline_days before scan_start for initial baseline
    data_start = scan_start - pd.Timedelta(days=baseline_days)
    scan_end = advisory + pd.Timedelta(days=30)  # also check after advisory

    window = df.loc[data_start:scan_end]
    if len(window) < 100:
        return None

    # Resample to daily means
    daily = window.resample("1D").mean()

    results = {
        "station_id": station_id,
        "data_range": f"{daily.index.min().date()} to {daily.index.max().date()}",
        "n_days": len(daily),
        "params": {},
        "first_anomaly_date": None,
        "first_anomaly_param": None,
        "first_anomaly_lead_days": None,
        "timeline": [],
    }

    first_anomaly = None

    for param in ["DO", "pH", "Turb", "Temp", "SpCond"]:
        if param not in daily.columns:
            continue

        series = daily[param].dropna()
        if len(series) < baseline_days // 2:
            continue

        # Compute rolling baseline statistics
        rolling_mean = series.rolling(f"{baseline_days}D", min_periods=30).mean()
        rolling_std = series.rolling(f"{baseline_days}D", min_periods=30).std()

        # Z-scores relative to rolling baseline
        z_scores = (series - rolling_mean) / rolling_std.clip(lower=0.01)

        # Find anomalies in the scan window (before advisory)
        scan_mask = (z_scores.index >= scan_start) & (z_scores.index <= scan_end)
        scan_z = z_scores[scan_mask]

        # Also check absolute thresholds
        scan_vals = series[scan_mask]

        param_result = {
            "n_values": int(len(scan_z)),
            "z_mean": float(scan_z.mean()) if len(scan_z) > 0 else None,
            "z_max": float(scan_z.abs().max()) if len(scan_z) > 0 else None,
            "val_mean": float(scan_vals.mean()) if len(scan_vals) > 0 else None,
            "val_min": float(scan_vals.min()) if len(scan_vals) > 0 else None,
            "val_max": float(scan_vals.max()) if len(scan_vals) > 0 else None,
            "anomaly_dates": [],
            "first_anomaly": None,
            "lead_days": None,
        }

        # Find first z-score anomaly
        for dt, z in scan_z.items():
            is_anomalous = False

            # Z-score based detection
            if abs(z) > Z_THRESHOLD:
                is_anomalous = True

            # Absolute threshold detection (DO dropping, Turb rising, etc.)
            if param in ENV_THRESHOLDS and dt in scan_vals.index:
                val = scan_vals[dt]
                th = ENV_THRESHOLDS[param]
                if "low" in th and val < th["low"]:
                    is_anomalous = True
                if "high" in th and val > th["high"]:
                    is_anomalous = True

            if is_anomalous and dt < advisory:
                date_str = str(dt.date())
                param_result["anomaly_dates"].append(date_str)
                if param_result["first_anomaly"] is None:
                    param_result["first_anomaly"] = date_str
                    lead = (advisory - dt).days
                    param_result["lead_days"] = lead

                    if first_anomaly is None or dt < first_anomaly:
                        first_anomaly = dt
                        results["first_anomaly_date"] = date_str
                        results["first_anomaly_param"] = param
                        results["first_anomaly_lead_days"] = lead

        results["params"][param] = param_result

    # Build daily timeline around advisory (30 days before to 15 days after)
    timeline_start = advisory - pd.Timedelta(days=60)
    timeline_end = advisory + pd.Timedelta(days=15)

    for day_offset in range(-60, 16):
        dt = advisory + pd.Timedelta(days=day_offset)
        if dt not in daily.index:
            continue

        day_data = {"date": str(dt.date()), "days_before_advisory": -day_offset}
        for param in ["DO", "pH", "Turb", "Temp", "SpCond"]:
            if param in daily.columns and not pd.isna(daily.loc[dt, param]):
                day_data[param] = round(float(daily.loc[dt, param]), 2)
        results["timeline"].append(day_data)

    return results


def get_drone_trigger_dates(event_id):
    """Get drone trigger dates from previous pipeline test results."""
    results_path = RESULTS_DIR / "sentinel_mini_pipeline_test.json"
    if not results_path.exists():
        return []

    with open(results_path) as f:
        data = json.load(f)

    for test in data.get("pipeline_tests", []):
        if test["event_id"] == event_id:
            triggers = []
            for run in test.get("pipeline_runs", []):
                if run["alert_level"] not in ("nominal",):
                    triggers.append({
                        "date": run["date"],
                        "alert_level": run["alert_level"],
                        "predictions": run.get("predictions", {}),
                        "status": run.get("status", "unknown"),
                    })
            return triggers

    return []


def main():
    log("=" * 70)
    log("SENTINEL MINI EARLY DETECTION TIMELINE TEST")
    log("=" * 70)
    log("Tests: How early does the combined drone + station system detect")
    log("anomalies before official advisories?")
    log("")

    all_stations = load_station_network()
    log(f"Station network: {len(all_stations)} USGS stations")

    results = {"events": [], "summary": {}}
    detection_count = 0
    total_events = 0

    for event in EVENTS:
        log(f"\n{'━' * 70}")
        log(f"EVENT: {event['name']}")
        log(f"Advisory date: {event['advisory_date']}")
        log(f"Expected degradation: {', '.join(event['expected_params'])}")

        total_events += 1

        # Find nearby stations (expanded to 100km for better coverage)
        nearby = find_nearest_stations(event["lat"], event["lon"], all_stations,
                                        k=5, max_km=100)
        if not nearby:
            log(f"  No USGS stations within 100km")
            results["events"].append({
                "event_id": event["event_id"],
                "name": event["name"],
                "status": "no_stations",
            })
            continue

        log(f"  Nearby stations ({len(nearby)}):")
        for sid, dist, info in nearby:
            log(f"    {sid} ({info['name'][:45]}): {dist:.1f} km")

        # Get drone trigger dates from previous test
        drone_triggers = get_drone_trigger_dates(event["event_id"])
        drone_lead_days = None
        first_drone_date = None

        if drone_triggers:
            first_drone_date = drone_triggers[0]["date"]
            advisory_dt = datetime.strptime(event["advisory_date"], "%Y-%m-%d")
            drone_dt = datetime.strptime(first_drone_date, "%Y-%m-%d")
            drone_lead_days = (advisory_dt - drone_dt).days
            log(f"\n  DRONE TRIGGER:")
            log(f"    First trigger: {first_drone_date} "
                f"({drone_lead_days} days before advisory)")
            log(f"    Alert level: {drone_triggers[0]['alert_level']}")
            log(f"    Total triggers: {len(drone_triggers)} of available tiles")
        else:
            log(f"\n  DRONE TRIGGER: None (no triggers in previous test)")

        # Analyze each station's sensor data for early anomalies
        log(f"\n  STATION ANOMALY ANALYSIS (rolling {BASELINE_DAYS}-day z-score):")

        event_result = {
            "event_id": event["event_id"],
            "name": event["name"],
            "advisory_date": event["advisory_date"],
            "type": event["type"],
            "drone_triggers": drone_triggers,
            "drone_first_trigger": first_drone_date,
            "drone_lead_days": drone_lead_days,
            "station_analyses": [],
            "best_station_lead_days": None,
            "best_station_param": None,
            "combined_lead_days": None,
        }

        best_station_lead = None

        for sid, dist, info in nearby:
            analysis = analyze_station_anomalies(sid, event["advisory_date"])
            if analysis is None:
                log(f"    {sid}: insufficient data")
                continue

            event_result["station_analyses"].append(analysis)

            # Report findings
            log(f"\n    Station {sid} ({info['name'][:35]}, {dist:.0f}km):")

            for param in event["expected_params"]:
                if param in analysis["params"]:
                    p = analysis["params"][param]
                    if p["first_anomaly"]:
                        log(f"      {param}: ANOMALY detected {p['lead_days']}d before advisory "
                            f"(first: {p['first_anomaly']}, |z|_max={p['z_max']:.1f})")
                        log(f"        Range: {p['val_min']:.1f} - {p['val_max']:.1f} "
                            f"(mean={p['val_mean']:.1f})")
                    else:
                        if p["n_values"] > 0:
                            log(f"      {param}: No anomaly (|z|_max={p['z_max']:.1f}, "
                                f"range={p['val_min']:.1f}-{p['val_max']:.1f})")
                        else:
                            log(f"      {param}: No data")

            # Other params with anomalies
            for param, p in analysis["params"].items():
                if param not in event["expected_params"] and p["first_anomaly"]:
                    log(f"      {param}: unexpected anomaly {p['lead_days']}d before "
                        f"(z_max={p['z_max']:.1f})")

            if analysis["first_anomaly_lead_days"] is not None:
                if best_station_lead is None or analysis["first_anomaly_lead_days"] > best_station_lead:
                    best_station_lead = analysis["first_anomaly_lead_days"]
                    event_result["best_station_lead_days"] = best_station_lead
                    event_result["best_station_param"] = analysis["first_anomaly_param"]

        # Combined early detection: earliest of drone trigger or station anomaly
        combined_lead = None
        combined_source = None

        if drone_lead_days is not None and best_station_lead is not None:
            if drone_lead_days >= best_station_lead:
                combined_lead = drone_lead_days
                combined_source = "drone"
            else:
                combined_lead = best_station_lead
                combined_source = "station"
        elif drone_lead_days is not None:
            combined_lead = drone_lead_days
            combined_source = "drone"
        elif best_station_lead is not None:
            combined_lead = best_station_lead
            combined_source = "station"

        event_result["combined_lead_days"] = combined_lead
        event_result["combined_source"] = combined_source

        # Summary for this event
        log(f"\n  ┌────────────────────────────────────────────")
        log(f"  │ EARLY DETECTION SUMMARY: {event['name']}")
        log(f"  │")
        if drone_lead_days is not None:
            log(f"  │ Drone trigger:    {drone_lead_days:3d} days before advisory")
        else:
            log(f"  │ Drone trigger:    NONE")
        if best_station_lead is not None:
            log(f"  │ Station anomaly:  {best_station_lead:3d} days before advisory "
                f"({event_result['best_station_param']})")
        else:
            log(f"  │ Station anomaly:  NONE (no data or no anomaly)")
        if combined_lead is not None:
            log(f"  │ Combined lead:    {combined_lead:3d} days ({combined_source} first)")
            detection_count += 1
        else:
            log(f"  │ Combined lead:    NOT DETECTED")
        log(f"  └────────────────────────────────────────────")

        results["events"].append(event_result)

    # Overall summary
    log(f"\n{'━' * 70}")
    log("OVERALL EARLY DETECTION RESULTS")
    log(f"{'━' * 70}")

    lead_times = []
    drone_leads = []
    station_leads = []

    log(f"\n{'Event':<35} {'Drone':>8} {'Station':>8} {'Combined':>8} {'Source':>8}")
    log(f"{'─' * 35} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for ev in results["events"]:
        name = ev["name"][:35]
        dl = f"{ev.get('drone_lead_days', '—')}d" if ev.get("drone_lead_days") is not None else "—"
        sl = f"{ev.get('best_station_lead_days', '—')}d" if ev.get("best_station_lead_days") is not None else "—"
        cl = f"{ev.get('combined_lead_days', '—')}d" if ev.get("combined_lead_days") is not None else "—"
        src = ev.get("combined_source", "—") or "—"

        log(f"{name:<35} {dl:>8} {sl:>8} {cl:>8} {src:>8}")

        if ev.get("combined_lead_days") is not None:
            lead_times.append(ev["combined_lead_days"])
        if ev.get("drone_lead_days") is not None:
            drone_leads.append(ev["drone_lead_days"])
        if ev.get("best_station_lead_days") is not None:
            station_leads.append(ev["best_station_lead_days"])

    log(f"\nDetected: {detection_count}/{total_events} events")
    if lead_times:
        log(f"Mean combined lead time: {np.mean(lead_times):.1f} days")
        log(f"Max combined lead time:  {max(lead_times)} days")
        log(f"Min combined lead time:  {min(lead_times)} days")
    if drone_leads:
        log(f"Mean drone lead time:    {np.mean(drone_leads):.1f} days")
    if station_leads:
        log(f"Mean station lead time:  {np.mean(station_leads):.1f} days")

    # Save results
    out_path = RESULTS_DIR / "early_detection_test.json"

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
