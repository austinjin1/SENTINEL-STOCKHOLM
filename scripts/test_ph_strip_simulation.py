#!/usr/bin/env python3
"""Simulate adding a pH strip sensor to the SENTINEL Mini drone.

Reruns the pipeline test but replaces WaterDroneNet's pH prediction
(R²=0.124) with a direct pH strip reading. Uses nearby station pH
as ground truth for what the strip would read. Adds ±0.3 noise to
simulate strip accuracy.

Compares detection rates: imagery-only vs imagery + pH strip.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from datetime import datetime
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

# pH strip specs (typical colorimetric strip)
PH_STRIP_NOISE_STD = 0.3   # ±0.3 pH units accuracy
PH_STRIP_COST = 0.15       # dollars per strip

# Anomaly thresholds — matching AnomalyScorer
# But with tighter pH thresholds since we now have real measurements
THRESHOLDS_IMAGERY_ONLY = {
    "DO": {"low": 4.0},
    "pH": {"low": 6.0, "high": 9.5},
    "Turb": {"high": 50.0},
    "SpCond": {"high": 10000},
}

# With pH strip, we can use tighter thresholds AND z-score detection
THRESHOLDS_WITH_STRIP = {
    "DO": {"low": 4.0},
    "pH": {"low": 6.5, "high": 9.0},  # tighter — strip gives real measurement
    "Turb": {"high": 50.0},
    "SpCond": {"high": 10000},
}

EVENTS = [
    {"event_id": "lake_erie_hab_2023", "name": "Lake Erie HAB 2023",
     "advisory_date": "2023-07-15", "lat": 41.50, "lon": -82.90, "type": "HAB"},
    {"event_id": "chesapeake_hypoxia_2018", "name": "Chesapeake Bay Hypoxia 2018",
     "advisory_date": "2018-07-20", "lat": 39.20, "lon": -76.50, "type": "hypoxia"},
    {"event_id": "klamath_river_hab_2021", "name": "Klamath River HAB 2021",
     "advisory_date": "2021-08-01", "lat": 41.55, "lon": -122.30, "type": "HAB"},
    {"event_id": "jordan_lake_hab_nc", "name": "Jordan Lake HAB NC",
     "advisory_date": "2022-07-15", "lat": 35.78, "lon": -79.06, "type": "HAB"},
    {"event_id": "mississippi_salinity_2023", "name": "Mississippi River Salinity 2023",
     "advisory_date": "2023-10-01", "lat": 29.95, "lon": -90.06, "type": "salinity"},
    {"event_id": "gulf_dead_zone_2023", "name": "Gulf Dead Zone 2023",
     "advisory_date": "2023-07-01", "lat": 29.50, "lon": -90.50, "type": "hypoxia"},
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
    dists = []
    for sid, info in stations.items():
        d = haversine_km(lat, lon, info["lat"], info["lon"])
        if d <= max_km:
            dists.append((sid, d, info))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def get_station_ph(station_id, date_str, window_days=3):
    """Get real pH from a station near a date (simulates what strip would read)."""
    fpath = SENSOR_DIR / f"{station_id}.parquet"
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index, utc=True)
        dt = pd.Timestamp(date_str, tz="UTC")
        start = dt - pd.Timedelta(days=window_days)
        end = dt + pd.Timedelta(days=window_days)
        window = df.loc[start:end]
        if "pH" not in window.columns:
            return None
        vals = window["pH"].dropna()
        if len(vals) == 0:
            return None
        return float(vals.mean())
    except Exception:
        return None


def get_station_baseline_ph(station_id, date_str, baseline_days=90):
    """Get rolling baseline pH stats for z-score computation."""
    fpath = SENSOR_DIR / f"{station_id}.parquet"
    if not fpath.exists():
        return None, None
    try:
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index, utc=True)
        dt = pd.Timestamp(date_str, tz="UTC")
        start = dt - pd.Timedelta(days=baseline_days)
        end = dt - pd.Timedelta(days=7)  # exclude recent week
        window = df.loc[start:end]
        if "pH" not in window.columns:
            return None, None
        vals = window["pH"].dropna()
        if len(vals) < 100:
            return None, None
        return float(vals.mean()), float(vals.std())
    except Exception:
        return None, None


def score_anomaly(predictions, thresholds):
    """Simple anomaly scoring against thresholds."""
    flags = []
    for param, pred in predictions.items():
        if param not in thresholds:
            continue
        th = thresholds[param]
        if "low" in th and pred < th["low"]:
            flags.append((param, "low", pred, th["low"]))
        if "high" in th and pred > th["high"]:
            flags.append((param, "high", pred, th["high"]))
    return flags


def load_model(device):
    from sentinel.models.waterdronenet import WaterDroneNet
    model = WaterDroneNet()
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    target_mean = ckpt.get("target_mean", np.zeros(5))
    target_std = ckpt.get("target_std", np.ones(5))
    return model, target_mean, target_std


@torch.no_grad()
def predict_from_image(model, image, target_mean, target_std, device):
    img_t = torch.from_numpy(image).unsqueeze(0).to(device)
    with autocast("cuda"):
        out = model(img_t)
    mu = out["mu"].cpu().float().squeeze(0).numpy()
    mu_real = mu * target_std + target_mean
    return {col: float(mu_real[i]) for i, col in enumerate(TARGET_COLS)}


def main():
    log("=" * 70)
    log("pH STRIP SENSOR SIMULATION")
    log("=" * 70)
    log("What if the drone also carries a $0.15 pH strip?")
    log(f"Strip accuracy: ±{PH_STRIP_NOISE_STD} pH units")
    log("")

    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_mean, target_std = load_model(device)
    log(f"WaterDroneNet loaded on {device}")

    all_stations = load_station_network()
    log(f"Station network: {len(all_stations)} stations")

    results = {"events": []}

    # Counters
    total_scans = 0
    imagery_triggers = 0
    strip_triggers = 0
    strip_extra_triggers = 0  # triggers that ONLY the strip version caught

    for event in EVENTS:
        log(f"\n{'━' * 70}")
        log(f"EVENT: {event['name']}")
        log(f"Advisory: {event['advisory_date']}")

        event_tiles = sorted(CRISIS_TILES.glob(f"{event['event_id']}_*.npz"))
        if not event_tiles:
            log(f"  No cached imagery — skipping")
            continue

        nearby = find_nearest_stations(event["lat"], event["lon"], all_stations,
                                        k=3, max_km=100)
        if not nearby:
            log(f"  No stations within 100km — skipping")
            continue

        advisory_dt = datetime.strptime(event["advisory_date"], "%Y-%m-%d")

        event_result = {
            "event_id": event["event_id"],
            "name": event["name"],
            "scans": [],
            "imagery_only": {"triggers": 0, "first_trigger_date": None, "lead_days": None},
            "with_ph_strip": {"triggers": 0, "first_trigger_date": None, "lead_days": None},
        }

        log(f"  {len(event_tiles)} tiles, nearest station: {nearby[0][0]} ({nearby[0][1]:.0f}km)")
        log(f"")
        log(f"  {'Date':<12} {'Imagery pH':>10} {'Strip pH':>10} {'Actual pH':>10} "
            f"{'Img Alert':>10} {'Strip Alert':>10} {'Lead':>6}")
        log(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 6}")

        for tile_path in event_tiles:
            date_str = tile_path.stem.split("_")[-1]
            if len(date_str) != 10:
                parts = tile_path.stem.replace(event["event_id"] + "_", "")
                date_str = parts

            total_scans += 1

            # 1. WaterDroneNet prediction (imagery only)
            img = np.load(tile_path)["image"]
            preds_imagery = predict_from_image(model, img, target_mean, target_std, device)

            # 2. Get actual pH from nearest station (what strip would read)
            actual_ph = None
            for sid, dist, info in nearby:
                actual_ph = get_station_ph(sid, date_str)
                if actual_ph is not None:
                    break

            # 3. Simulate pH strip reading (actual + noise)
            strip_ph = None
            if actual_ph is not None:
                strip_ph = actual_ph + np.random.normal(0, PH_STRIP_NOISE_STD)
                strip_ph = np.clip(strip_ph, 0, 14)

            # 4. Score: imagery only
            img_flags = score_anomaly(preds_imagery, THRESHOLDS_IMAGERY_ONLY)
            img_alert = len(img_flags) > 0

            # 5. Score: imagery + pH strip
            preds_with_strip = dict(preds_imagery)
            if strip_ph is not None:
                preds_with_strip["pH"] = strip_ph  # replace imagery pH with strip

            strip_flags = score_anomaly(preds_with_strip, THRESHOLDS_WITH_STRIP)

            # Also check pH z-score against baseline
            ph_zscore = None
            ph_z_flag = False
            if strip_ph is not None:
                for sid, dist, info in nearby:
                    base_mean, base_std = get_station_baseline_ph(sid, date_str)
                    if base_mean is not None:
                        ph_zscore = (strip_ph - base_mean) / max(base_std, 0.05)
                        if abs(ph_zscore) > 2.0:
                            ph_z_flag = True
                            strip_flags.append(("pH", "z-score", strip_ph, ph_zscore))
                        break

            strip_alert = len(strip_flags) > 0

            if img_alert:
                imagery_triggers += 1
            if strip_alert:
                strip_triggers += 1
            if strip_alert and not img_alert:
                strip_extra_triggers += 1

            # Track first triggers
            try:
                tile_dt = datetime.strptime(date_str, "%Y-%m-%d")
                lead = (advisory_dt - tile_dt).days
            except ValueError:
                lead = None

            if img_alert:
                event_result["imagery_only"]["triggers"] += 1
                if event_result["imagery_only"]["first_trigger_date"] is None:
                    event_result["imagery_only"]["first_trigger_date"] = date_str
                    event_result["imagery_only"]["lead_days"] = lead

            if strip_alert:
                event_result["with_ph_strip"]["triggers"] += 1
                if event_result["with_ph_strip"]["first_trigger_date"] is None:
                    event_result["with_ph_strip"]["first_trigger_date"] = date_str
                    event_result["with_ph_strip"]["lead_days"] = lead

            # Display
            img_ph_str = f"{preds_imagery['pH']:.1f}"
            strip_str = f"{strip_ph:.1f}" if strip_ph is not None else "—"
            actual_str = f"{actual_ph:.1f}" if actual_ph is not None else "—"
            img_alert_str = "TRIGGER" if img_alert else "—"
            strip_alert_str = "TRIGGER" if strip_alert else "—"
            lead_str = f"{lead}d" if lead is not None else "—"

            # Highlight extra catches
            if strip_alert and not img_alert:
                strip_alert_str = "NEW ✓"

            if img_alert:
                img_flag_params = [f[0] for f in img_flags]
                img_alert_str = f"{'|'.join(img_flag_params)}"

            if strip_alert:
                strip_flag_params = []
                for f in strip_flags:
                    if f[1] == "z-score":
                        strip_flag_params.append(f"pH(z={f[3]:.1f})")
                    else:
                        strip_flag_params.append(f[0])
                strip_alert_str = f"{'|'.join(strip_flag_params)}"
                if not img_alert:
                    strip_alert_str = f"NEW: {strip_alert_str}"

            log(f"  {date_str:<12} {img_ph_str:>10} {strip_str:>10} {actual_str:>10} "
                f"{img_alert_str:>10} {strip_alert_str:>20} {lead_str:>6}")

            event_result["scans"].append({
                "date": date_str,
                "imagery_ph": round(preds_imagery["pH"], 2),
                "strip_ph": round(strip_ph, 2) if strip_ph else None,
                "actual_ph": round(actual_ph, 2) if actual_ph else None,
                "ph_zscore": round(ph_zscore, 2) if ph_zscore is not None else None,
                "imagery_alert": img_alert,
                "strip_alert": strip_alert,
                "lead_days": lead,
            })

        # Event summary
        log(f"")
        il = event_result["imagery_only"]
        sl = event_result["with_ph_strip"]
        il_lead = f", first {il['lead_days']}d early" if il['lead_days'] else ", none"
        sl_lead = f", first {sl['lead_days']}d early" if sl['lead_days'] else ", none"
        log(f"  Imagery only:    {il['triggers']}/{len(event_tiles)} triggers{il_lead}")
        log(f"  With pH strip:   {sl['triggers']}/{len(event_tiles)} triggers{sl_lead}")

        if sl["triggers"] > il["triggers"]:
            extra = sl["triggers"] - il["triggers"]
            log(f"  pH strip gained: +{extra} additional triggers")

        results["events"].append(event_result)

    # Overall summary
    log(f"\n{'━' * 70}")
    log("OVERALL COMPARISON")
    log(f"{'━' * 70}")
    log(f"")
    log(f"Total drone scans: {total_scans}")
    log(f"")
    log(f"{'Mode':<25} {'Triggers':>10} {'Rate':>8}")
    log(f"{'─' * 25} {'─' * 10} {'─' * 8}")
    log(f"{'Imagery only':<25} {imagery_triggers:>10} {imagery_triggers/max(total_scans,1):.0%}")
    log(f"{'Imagery + pH strip':<25} {strip_triggers:>10} {strip_triggers/max(total_scans,1):.0%}")
    log(f"{'NEW catches (strip only)':<25} {strip_extra_triggers:>10}")
    log(f"")

    log(f"{'Event':<35} {'Img Lead':>10} {'Strip Lead':>10} {'Gain':>8}")
    log(f"{'─' * 35} {'─' * 10} {'─' * 10} {'─' * 8}")

    for ev in results["events"]:
        name = ev["name"][:35]
        il = ev["imagery_only"]["lead_days"]
        sl = ev["with_ph_strip"]["lead_days"]
        il_str = f"{il}d" if il is not None else "—"
        sl_str = f"{sl}d" if sl is not None else "—"

        gain = ""
        if sl is not None and il is not None:
            diff = sl - il
            gain = f"+{diff}d" if diff > 0 else f"{diff}d"
        elif sl is not None and il is None:
            gain = "NEW"

        log(f"{name:<35} {il_str:>10} {sl_str:>10} {gain:>8}")

    log(f"\nCost per flight: +${PH_STRIP_COST:.2f} (pH strip)")
    log(f"Accuracy: ±{PH_STRIP_NOISE_STD} pH units")
    log(f"")

    # Save
    out_path = RESULTS_DIR / "ph_strip_simulation.json"

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
    log(f"Results saved to {out_path}")
    log("DONE")


if __name__ == "__main__":
    main()
