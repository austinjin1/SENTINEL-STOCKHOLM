#!/usr/bin/env python3
"""Test WaterDroneNet (SENTINEL Mini) on historical water quality crises.

Downloads Sentinel-2 imagery for each case study site across a time series
before and during the documented event. Runs WaterDroneNet inference and
uses the AnomalyScorer to determine if the drone model would have detected
the crisis from imagery alone.

This tests the full SENTINEL Mini pipeline:
  1. S2 imagery download → WaterDroneNet inference → AnomalyScorer → Alert

Output: results/waterdronenet_crisis_test.json

MIT License — Bryan Cheng, 2026
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_PATH = PROJECT / "checkpoints" / "waterdronenet" / "waterdronenet_best.pt"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TILE_CACHE = PROJECT / "data" / "processed" / "satellite" / "crisis_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

# Target info
TARGET_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]
BANDS = ["B02", "B03", "B04", "B08"]
PATCH_SIZE = 224
MAX_CLOUD = 40  # slightly more lenient for case study dates

# ──────────────────────────────────────────────────────────────────────
# Case study events (same as exp1_case_studies_real.py)
# ──────────────────────────────────────────────────────────────────────
EVENTS = [
    {
        "event_id": "lake_erie_hab_2023",
        "name": "Lake Erie HAB 2023",
        "advisory_date": "2023-07-15",
        "lat": 41.50, "lon": -82.90,
        "usgs_site": "04199500",
        "pre_event_days": 60,
        "type": "HAB",
    },
    {
        "event_id": "gulf_dead_zone_2023",
        "name": "Gulf Dead Zone 2023",
        "advisory_date": "2023-07-01",
        "lat": 29.50, "lon": -90.50,
        "usgs_site": "07374000",
        "pre_event_days": 90,
        "type": "hypoxia",
    },
    {
        "event_id": "chesapeake_hypoxia_2018",
        "name": "Chesapeake Bay Hypoxia 2018",
        "advisory_date": "2018-07-20",
        "lat": 39.20, "lon": -76.50,
        "usgs_site": "01578310",
        "pre_event_days": 90,
        "type": "hypoxia",
    },
    {
        "event_id": "klamath_river_hab_2021",
        "name": "Klamath River HAB 2021",
        "advisory_date": "2021-08-01",
        "lat": 41.55, "lon": -122.30,
        "usgs_site": "11530500",
        "pre_event_days": 60,
        "type": "HAB",
    },
    {
        "event_id": "jordan_lake_hab_nc",
        "name": "Jordan Lake HAB NC",
        "advisory_date": "2022-07-15",
        "lat": 35.78, "lon": -79.06,
        "usgs_site": "02097517",
        "pre_event_days": 45,
        "type": "HAB",
    },
    {
        "event_id": "mississippi_salinity_2023",
        "name": "Mississippi River Salinity 2023",
        "advisory_date": "2023-10-01",
        "lat": 29.95, "lon": -90.06,
        "usgs_site": "07374000",
        "pre_event_days": 60,
        "type": "salinity",
    },
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_tile(lat, lon, date_str, event_id):
    """Download single 4-band S2 tile. Returns image array or None."""
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer

    cache_path = TILE_CACHE / f"{event_id}_{date_str}.npz"
    if cache_path.exists():
        try:
            return np.load(cache_path)["image"]
        except Exception:
            cache_path.unlink(missing_ok=True)

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        bbox = [lon - 0.015, lat - 0.015, lon + 0.015, lat + 0.015]
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        t_start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
        t_end = (dt + timedelta(days=5)).strftime("%Y-%m-%d")

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{t_start}/{t_end}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
            max_items=5,
        )
        items = list(search.items())
        if not items:
            return None

        target_ts = dt.timestamp()
        items.sort(key=lambda it: abs(
            datetime.fromisoformat(
                it.properties["datetime"].replace("Z", "+00:00")
            ).timestamp() - target_ts
        ))
        item = items[0]
        signed = planetary_computer.sign(item)

        bands_data = []
        for band_name in BANDS:
            asset = signed.assets.get(band_name)
            if asset is None:
                return None
            with rasterio.open(asset.href) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                py_idx, px_idx = src.index(x, y)
                half = PATCH_SIZE // 2
                if not (half <= px_idx < src.width - half and
                        half <= py_idx < src.height - half):
                    return None
                window = rasterio.windows.Window(
                    px_idx - half, py_idx - half, PATCH_SIZE, PATCH_SIZE)
                data = src.read(1, window=window).astype(np.float32) / 10000.0
                if data.shape != (PATCH_SIZE, PATCH_SIZE):
                    return None
                bands_data.append(data)

        if len(bands_data) != len(BANDS):
            return None

        image = np.stack(bands_data, axis=0)
        if np.isnan(image).any() or image.max() < 0.001:
            return None

        np.savez_compressed(cache_path, image=image)
        return image
    except Exception as e:
        log(f"    Download error for {event_id} {date_str}: {e}")
        return None


def load_model(device):
    """Load WaterDroneNet from checkpoint."""
    from sentinel.models.waterdronenet import WaterDroneNet

    model = WaterDroneNet()
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    target_mean = ckpt.get("target_mean", np.zeros(5))
    target_std = ckpt.get("target_std", np.ones(5))

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Loaded WaterDroneNet ({n_params:,} params) from epoch {ckpt['epoch']}")
    log(f"Target norm: mean={target_mean}, std={target_std}")

    return model, target_mean, target_std


@torch.no_grad()
def predict(model, image, target_mean, target_std, device):
    """Run WaterDroneNet on a single image, return denormalized predictions."""
    img_t = torch.from_numpy(image).unsqueeze(0).to(device)

    with autocast("cuda"):
        out = model(img_t)

    mu = out["mu"].cpu().float().squeeze(0).numpy()
    sigma = out["sigma"].cpu().float().squeeze(0).numpy()
    trust = torch.sigmoid(out["trust_logit"]).cpu().float().item()

    # Denormalize
    mu_real = mu * target_std + target_mean
    sigma_real = sigma * np.abs(target_std)

    return {
        "predictions": {col: float(mu_real[i]) for i, col in enumerate(TARGET_COLS)},
        "uncertainties": {col: float(sigma_real[i]) for i, col in enumerate(TARGET_COLS)},
        "trust": trust,
    }


def score_anomaly(predictions, uncertainties):
    """Score predictions using the SENTINEL Mini trigger system's AnomalyScorer."""
    from sentinel.platform.sentinel_mini_trigger import AnomalyScorer, AlertLevel

    scorer = AnomalyScorer()
    alert_level, scores = scorer.score(predictions, uncertainties)
    return alert_level.value, scores


def main():
    log("=" * 65)
    log("WaterDroneNet Crisis Prediction Test")
    log("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Load model
    model, target_mean, target_std = load_model(device)

    results = {"events": [], "summary": {}}
    detected = 0
    total = 0

    for event in EVENTS:
        log(f"\n{'─' * 55}")
        log(f"Event: {event['name']}")
        log(f"Advisory date: {event['advisory_date']}")
        log(f"Location: ({event['lat']}, {event['lon']})")
        log(f"Type: {event['type']}")

        advisory_dt = datetime.strptime(event["advisory_date"], "%Y-%m-%d")
        pre_days = event["pre_event_days"]

        # Generate sample dates: every 5 days from pre_event to advisory + 10
        sample_dates = []
        for d in range(-pre_days, 15, 5):
            dt = advisory_dt + timedelta(days=d)
            sample_dates.append(dt.strftime("%Y-%m-%d"))

        # Download imagery for each date
        log(f"  Downloading {len(sample_dates)} S2 tiles...")
        time_series = []
        for date_str in sample_dates:
            img = download_tile(event["lat"], event["lon"], date_str, event["event_id"])
            if img is not None:
                pred = predict(model, img, target_mean, target_std, device)
                alert_level, anomaly_scores = score_anomaly(
                    pred["predictions"], pred["uncertainties"]
                )

                days_before = (advisory_dt - datetime.strptime(date_str, "%Y-%m-%d")).days

                entry = {
                    "date": date_str,
                    "days_before_advisory": days_before,
                    "predictions": pred["predictions"],
                    "uncertainties": pred["uncertainties"],
                    "trust": pred["trust"],
                    "alert_level": alert_level,
                    "anomaly_scores": anomaly_scores,
                }
                time_series.append(entry)

                flag = "*" if alert_level not in ("nominal", "watch") else " "
                log(f"  {flag} {date_str} (T-{days_before:3d}d): "
                    f"alert={alert_level:8s} | "
                    f"DO={pred['predictions']['DO']:.1f} "
                    f"pH={pred['predictions']['pH']:.1f} "
                    f"Turb={pred['predictions']['Turb']:.1f} "
                    f"Temp={pred['predictions']['Temp']:.1f}")
            else:
                log(f"    {date_str}: no imagery available")

        # Determine if event was detected
        # Detection = any alert >= WARNING before advisory date
        pre_alerts = [
            e for e in time_series
            if e["days_before_advisory"] > 0
            and e["alert_level"] in ("warning", "alert", "critical")
        ]

        event_detected = len(pre_alerts) > 0
        total += 1

        if event_detected:
            detected += 1
            first_alert = max(pre_alerts, key=lambda x: x["days_before_advisory"])
            lead_time = first_alert["days_before_advisory"]
            log(f"  ✓ DETECTED — lead time: {lead_time} days")
        else:
            lead_time = None
            # Check if we got any imagery at all
            if not time_series:
                log(f"  ✗ NO IMAGERY AVAILABLE")
            else:
                # Report max anomaly score seen
                max_score = max(
                    max(e["anomaly_scores"].values()) if e["anomaly_scores"] else 0
                    for e in time_series
                )
                log(f"  ✗ NOT DETECTED (max anomaly score: {max_score:.2f})")

        event_result = {
            "event_id": event["event_id"],
            "name": event["name"],
            "type": event["type"],
            "advisory_date": event["advisory_date"],
            "detected": event_detected,
            "lead_time_days": lead_time,
            "n_images": len(time_series),
            "n_pre_alerts": len(pre_alerts),
            "time_series": time_series,
        }
        results["events"].append(event_result)

    # Summary
    lead_times = [e["lead_time_days"] for e in results["events"]
                  if e["lead_time_days"] is not None]
    results["summary"] = {
        "total_events": total,
        "detected": detected,
        "detection_rate": detected / max(total, 1),
        "mean_lead_time_days": float(np.mean(lead_times)) if lead_times else None,
        "events_with_imagery": sum(1 for e in results["events"] if e["n_images"] > 0),
        "model_checkpoint": str(CKPT_PATH.name),
        "timestamp": datetime.now().isoformat(),
    }

    log(f"\n{'=' * 55}")
    log(f"SUMMARY")
    log(f"{'=' * 55}")
    log(f"Events tested: {total}")
    log(f"Detected: {detected}/{total} ({100*detected/max(total,1):.0f}%)")
    if lead_times:
        log(f"Mean lead time: {np.mean(lead_times):.1f} days")
    log(f"Events with imagery: {results['summary']['events_with_imagery']}/{total}")

    # Save results
    out_path = RESULTS_DIR / "waterdronenet_crisis_test.json"

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
