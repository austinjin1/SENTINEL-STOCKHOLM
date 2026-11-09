"""SENTINEL Mini → Full SENTINEL Trigger System.

Implements the drone-to-station activation pipeline:

1. SENTINEL Mini (WaterDroneNet) runs on drone with multispectral camera
2. When anomaly detected (WQ prediction exceeds thresholds), trigger fires
3. RF controller activates nearest fixed coastal SENTINEL station(s)
4. Full multimodal SENTINEL confirms/refutes drone finding

The trigger system supports:
- Anomaly scoring from WaterDroneNet predictions + uncertainty
- Station proximity matching (nearest K stations within range)
- RF activation commands (LoRa protocol for coastal arrays)
- Confirmation feedback loop (station data validates drone alert)
- Power-saving duty cycling (stations sleep until triggered)

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum RF range for LoRa trigger (meters)
MAX_RF_RANGE_M: float = 15_000.0  # 15 km typical LoRa range

#: Default number of stations to activate per trigger
DEFAULT_K_STATIONS: int = 3

#: Anomaly threshold multipliers (sigma units above/below expected)
ANOMALY_THRESHOLDS = {
    "DO": {"low": 4.0, "critical": 2.0, "unit": "mg/L", "direction": "below"},
    "pH": {"low": 6.5, "high": 9.0, "critical_low": 5.0, "critical_high": 10.0, "unit": "pH"},
    "Turb": {"high": 50.0, "critical": 200.0, "unit": "NTU", "direction": "above"},
    "Temp": {"high": 30.0, "critical": 35.0, "unit": "°C", "direction": "above"},
    "SpCond": {"high": 2500.0, "critical": 10000.0, "unit": "μS/cm", "direction": "above"},
}

#: Monitoring parameters for full station activation
FULL_STATION_PARAMS = [
    "dissolved_oxygen", "ph", "turbidity", "temperature",
    "conductivity", "chlorophyll_a", "nitrate", "phosphate",
    "total_suspended_solids", "specific_conductance",
]


class AlertLevel(Enum):
    """Alert severity from drone detection."""
    NOMINAL = "nominal"
    WATCH = "watch"          # Elevated but within bounds
    WARNING = "warning"      # Exceeds threshold, low confidence
    ALERT = "alert"          # Exceeds threshold, high confidence
    CRITICAL = "critical"    # Severe exceedance, immediate action


class StationMode(Enum):
    """Operating mode of a fixed SENTINEL station."""
    SLEEP = "sleep"          # Low-power, periodic heartbeat only
    STANDBY = "standby"      # Listening for triggers
    ACTIVE = "active"        # Full sensor suite running
    CONFIRM = "confirm"      # Targeted confirmation sampling
    CONTINUOUS = "continuous" # 24/7 monitoring (high-priority sites)


@dataclass
class DroneDetection:
    """A single anomaly detection from SENTINEL Mini."""
    timestamp: str
    lat: float
    lon: float
    altitude_m: float
    predictions: Dict[str, float]    # param -> predicted value
    uncertainties: Dict[str, float]  # param -> sigma
    alert_level: AlertLevel
    anomaly_scores: Dict[str, float] # param -> z-score or threshold ratio
    image_hash: str                  # SHA-256 of source imagery
    drone_id: str = "MINI-001"
    flight_id: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "lat": self.lat,
            "lon": self.lon,
            "altitude_m": self.altitude_m,
            "predictions": self.predictions,
            "uncertainties": self.uncertainties,
            "alert_level": self.alert_level.value,
            "anomaly_scores": self.anomaly_scores,
            "image_hash": self.image_hash,
            "drone_id": self.drone_id,
            "flight_id": self.flight_id,
        }


@dataclass
class FixedStation:
    """A fixed coastal SENTINEL station in the network."""
    station_id: str
    lat: float
    lon: float
    name: str = ""
    mode: StationMode = StationMode.STANDBY
    rf_channel: int = 1
    capabilities: List[str] = field(default_factory=lambda: [
        "sensor", "satellite", "microbial"
    ])
    last_heartbeat: Optional[str] = None
    battery_pct: float = 100.0


@dataclass
class TriggerCommand:
    """RF command sent to activate a fixed station."""
    command_id: str
    target_station_id: str
    trigger_source: str  # drone_id
    detection: DroneDetection
    requested_mode: StationMode
    requested_params: List[str]
    priority: int  # 1=highest
    duration_minutes: int = 60  # how long to stay active
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.command_id:
            self.command_id = hashlib.sha256(
                f"{self.target_station_id}:{self.timestamp}".encode()
            ).hexdigest()[:12]


@dataclass
class ConfirmationResult:
    """Result from a fixed station confirming/refuting a drone detection."""
    station_id: str
    command_id: str
    detection_confirmed: bool
    confidence: float  # 0-1
    measured_values: Dict[str, float]
    anomaly_confirmed: Dict[str, bool]  # per-param confirmation
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class AnomalyScorer:
    """Scores WaterDroneNet predictions against anomaly thresholds.

    Takes raw predictions + uncertainties from the drone model and
    determines if any parameter exceeds environmental thresholds,
    accounting for prediction uncertainty.
    """

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or ANOMALY_THRESHOLDS

    def score(
        self,
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> Tuple[AlertLevel, Dict[str, float]]:
        """Score predictions against thresholds.

        Returns (alert_level, per_param_scores).
        Score > 1.0 means threshold exceeded.
        """
        scores = {}
        max_severity = 0  # 0=nominal, 1=watch, 2=warning, 3=alert, 4=critical

        for param, value in predictions.items():
            if param not in self.thresholds:
                continue

            thresh = self.thresholds[param]
            sigma = uncertainties.get(param, float("inf"))
            score = 0.0

            direction = thresh.get("direction", "above")

            if direction == "below":
                # Anomalous when LOW (e.g., dissolved oxygen)
                low = thresh.get("low", float("inf"))
                critical = thresh.get("critical", 0.0)
                if value < critical:
                    score = 2.0 + (critical - value) / max(sigma, 0.01)
                elif value < low:
                    score = 1.0 + (low - value) / max(sigma, 0.01)
                else:
                    score = max(0, (low - value) / max(sigma, 0.01))

            elif direction == "above":
                # Anomalous when HIGH (e.g., turbidity)
                high = thresh.get("high", float("inf"))
                critical = thresh.get("critical", float("inf"))
                if value > critical:
                    score = 2.0 + (value - critical) / max(sigma, 0.01)
                elif value > high:
                    score = 1.0 + (value - high) / max(sigma, 0.01)
                else:
                    score = max(0, (value - high) / max(sigma, 0.01))

            else:
                # Both directions (e.g., pH)
                low = thresh.get("low", 0.0)
                high = thresh.get("high", 14.0)
                crit_low = thresh.get("critical_low", 0.0)
                crit_high = thresh.get("critical_high", 14.0)
                if value < crit_low or value > crit_high:
                    score = 2.0
                elif value < low:
                    score = 1.0 + (low - value) / max(sigma, 0.01)
                elif value > high:
                    score = 1.0 + (value - high) / max(sigma, 0.01)

            scores[param] = score

            # Map score to severity
            if score >= 2.0:
                max_severity = max(max_severity, 4)
            elif score >= 1.5:
                max_severity = max(max_severity, 3)
            elif score >= 1.0:
                # High uncertainty → warning, low uncertainty → alert
                if sigma < abs(value) * 0.2:
                    max_severity = max(max_severity, 3)
                else:
                    max_severity = max(max_severity, 2)
            elif score >= 0.5:
                max_severity = max(max_severity, 1)

        severity_map = {
            0: AlertLevel.NOMINAL,
            1: AlertLevel.WATCH,
            2: AlertLevel.WARNING,
            3: AlertLevel.ALERT,
            4: AlertLevel.CRITICAL,
        }
        return severity_map[max_severity], scores


class StationSelector:
    """Selects which fixed stations to activate based on drone detection.

    Uses geographic proximity + station capabilities + battery status
    to choose the optimal K stations for confirmation.
    """

    def __init__(self, stations: List[FixedStation]):
        self.stations = {s.station_id: s for s in stations}

    def _haversine_m(self, lat1: float, lon1: float,
                     lat2: float, lon2: float) -> float:
        """Haversine distance in meters."""
        R = 6_371_000.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

    def select(
        self,
        detection: DroneDetection,
        k: int = DEFAULT_K_STATIONS,
        max_range_m: float = MAX_RF_RANGE_M,
    ) -> List[Tuple[FixedStation, float]]:
        """Select K nearest stations within RF range.

        Returns list of (station, distance_m) tuples, sorted by distance.
        Filters out stations with low battery or already in continuous mode.
        """
        candidates = []
        for station in self.stations.values():
            if station.mode == StationMode.CONTINUOUS:
                continue  # already active, no need to trigger
            if station.battery_pct < 10.0:
                continue  # insufficient battery

            dist = self._haversine_m(
                detection.lat, detection.lon, station.lat, station.lon
            )
            if dist <= max_range_m:
                candidates.append((station, dist))

        candidates.sort(key=lambda x: x[1])
        return candidates[:k]


class RFController:
    """Simulated RF controller for LoRa-based station activation.

    In production, this interfaces with a LoRa transceiver module
    (e.g., SX1276) to send activation commands over ISM bands.
    For development, it logs commands and simulates responses.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/rf_triggers")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.command_log: List[TriggerCommand] = []

    def send_trigger(self, command: TriggerCommand) -> bool:
        """Send activation command to a fixed station.

        Returns True if acknowledgment received (simulated).
        In production: sends LoRa packet, waits for ACK with timeout.
        """
        self.command_log.append(command)

        # Log to file
        log_entry = {
            "command_id": command.command_id,
            "station": command.target_station_id,
            "source": command.trigger_source,
            "mode": command.requested_mode.value,
            "params": command.requested_params,
            "priority": command.priority,
            "duration_min": command.duration_minutes,
            "timestamp": command.timestamp,
            "detection_alert": command.detection.alert_level.value,
            "detection_lat": command.detection.lat,
            "detection_lon": command.detection.lon,
        }

        log_file = self.log_dir / f"trigger_{command.command_id}.json"
        log_file.write_text(json.dumps(log_entry, indent=2))

        logger.info(
            "RF trigger sent: station=%s mode=%s priority=%d cmd=%s",
            command.target_station_id,
            command.requested_mode.value,
            command.priority,
            command.command_id,
        )
        return True  # simulated ACK

    def get_history(self, limit: int = 50) -> List[dict]:
        """Return recent trigger history."""
        return [
            {
                "command_id": c.command_id,
                "station": c.target_station_id,
                "alert": c.detection.alert_level.value,
                "timestamp": c.timestamp,
            }
            for c in self.command_log[-limit:]
        ]


class SentinelMiniTriggerSystem:
    """End-to-end SENTINEL Mini → Full SENTINEL trigger pipeline.

    Orchestrates:
    1. Anomaly scoring from drone WaterDroneNet predictions
    2. Station selection (nearest K within RF range)
    3. RF trigger command generation and transmission
    4. Confirmation result processing
    5. Alert escalation or de-escalation

    Usage
    -----
    >>> system = SentinelMiniTriggerSystem(stations=station_list)
    >>> result = system.process_drone_detection(
    ...     predictions={"DO": 3.2, "Turb": 85.0, ...},
    ...     uncertainties={"DO": 0.8, "Turb": 12.0, ...},
    ...     lat=41.5, lon=-81.7, altitude_m=50.0,
    ...     drone_id="MINI-001",
    ... )
    >>> print(result["alert_level"])  # "alert"
    >>> print(result["stations_triggered"])  # ["USGS-04199500", ...]
    """

    def __init__(
        self,
        stations: List[FixedStation],
        rf_controller: Optional[RFController] = None,
        k_stations: int = DEFAULT_K_STATIONS,
        max_range_m: float = MAX_RF_RANGE_M,
        log_dir: Optional[Path] = None,
    ):
        self.scorer = AnomalyScorer()
        self.selector = StationSelector(stations)
        self.rf = rf_controller or RFController(log_dir=log_dir)
        self.k_stations = k_stations
        self.max_range_m = max_range_m
        self.detection_log: List[dict] = []

    def process_drone_detection(
        self,
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
        lat: float,
        lon: float,
        altitude_m: float = 50.0,
        drone_id: str = "MINI-001",
        flight_id: str = "",
        image_hash: str = "",
    ) -> dict:
        """Process a drone detection and trigger stations if needed.

        Parameters
        ----------
        predictions : dict
            WaterDroneNet predictions {param: value}.
        uncertainties : dict
            WaterDroneNet uncertainties {param: sigma}.
        lat, lon : float
            GPS coordinates of observation.
        altitude_m : float
            Drone altitude in meters.
        drone_id : str
            Identifier for the drone unit.

        Returns
        -------
        dict with keys:
            alert_level, anomaly_scores, stations_triggered,
            commands_sent, detection_id
        """
        # 1. Score anomalies
        alert_level, anomaly_scores = self.scorer.score(predictions, uncertainties)

        # 2. Create detection record
        if not image_hash:
            image_hash = hashlib.sha256(
                json.dumps(predictions, sort_keys=True).encode()
            ).hexdigest()

        detection = DroneDetection(
            timestamp=datetime.now(timezone.utc).isoformat(),
            lat=lat,
            lon=lon,
            altitude_m=altitude_m,
            predictions=predictions,
            uncertainties=uncertainties,
            alert_level=alert_level,
            anomaly_scores=anomaly_scores,
            image_hash=image_hash,
            drone_id=drone_id,
            flight_id=flight_id,
        )

        result = {
            "alert_level": alert_level.value,
            "anomaly_scores": anomaly_scores,
            "stations_triggered": [],
            "commands_sent": [],
            "detection": detection.to_dict(),
        }

        # 3. Only trigger stations for WARNING or higher
        if alert_level.value in ("nominal", "watch"):
            logger.info("Detection %s: %s — no trigger needed",
                        alert_level.value, anomaly_scores)
            self.detection_log.append(result)
            return result

        # 4. Select nearest stations
        selected = self.selector.select(
            detection, k=self.k_stations, max_range_m=self.max_range_m
        )

        if not selected:
            logger.warning("No stations in range (%.0f m) for detection at (%.4f, %.4f)",
                           self.max_range_m, lat, lon)
            self.detection_log.append(result)
            return result

        # 5. Determine activation mode based on alert level
        if alert_level == AlertLevel.CRITICAL:
            mode = StationMode.ACTIVE
            duration = 120  # 2 hours
            priority = 1
        elif alert_level == AlertLevel.ALERT:
            mode = StationMode.CONFIRM
            duration = 60  # 1 hour
            priority = 2
        else:  # WARNING
            mode = StationMode.CONFIRM
            duration = 30  # 30 min
            priority = 3

        # 6. Determine which params to monitor based on anomaly scores
        triggered_params = [
            p for p, s in anomaly_scores.items() if s >= 0.5
        ]
        if not triggered_params:
            triggered_params = list(predictions.keys())

        # 7. Send RF triggers
        for station, dist_m in selected:
            cmd = TriggerCommand(
                command_id="",
                target_station_id=station.station_id,
                trigger_source=drone_id,
                detection=detection,
                requested_mode=mode,
                requested_params=triggered_params,
                priority=priority,
                duration_minutes=duration,
            )

            success = self.rf.send_trigger(cmd)
            if success:
                station.mode = mode
                result["stations_triggered"].append(station.station_id)
                result["commands_sent"].append({
                    "command_id": cmd.command_id,
                    "station": station.station_id,
                    "distance_m": round(dist_m, 1),
                    "mode": mode.value,
                })

        logger.info(
            "Trigger complete: %s level, %d stations activated for %d min",
            alert_level.value, len(result["stations_triggered"]), duration,
        )

        self.detection_log.append(result)
        return result

    def process_confirmation(
        self,
        confirmation: ConfirmationResult,
    ) -> dict:
        """Process a confirmation result from a fixed station.

        Compares station measurements against the drone detection
        that triggered it. Updates alert status accordingly.

        Returns
        -------
        dict with confirmed/refuted status and recommendation.
        """
        n_confirmed = sum(1 for v in confirmation.anomaly_confirmed.values() if v)
        n_total = len(confirmation.anomaly_confirmed)

        if n_total == 0:
            status = "inconclusive"
            recommendation = "Continue monitoring"
        elif n_confirmed / n_total >= 0.5:
            status = "confirmed"
            recommendation = "Escalate to full SENTINEL analysis"
        elif n_confirmed > 0:
            status = "partial"
            recommendation = "Extend monitoring window, increase sampling"
        else:
            status = "refuted"
            recommendation = "Return to standby, log false positive for model improvement"

        result = {
            "status": status,
            "confirmed_params": [
                p for p, v in confirmation.anomaly_confirmed.items() if v
            ],
            "refuted_params": [
                p for p, v in confirmation.anomaly_confirmed.items() if not v
            ],
            "confidence": confirmation.confidence,
            "recommendation": recommendation,
            "station_id": confirmation.station_id,
            "command_id": confirmation.command_id,
            "measured_values": confirmation.measured_values,
        }

        logger.info(
            "Confirmation from %s: %s (%.0f%% confidence)",
            confirmation.station_id, status, confirmation.confidence * 100,
        )

        return result

    def get_status(self) -> dict:
        """Get current system status."""
        station_modes = {}
        for sid, station in self.selector.stations.items():
            mode = station.mode.value
            station_modes[mode] = station_modes.get(mode, 0) + 1

        return {
            "total_stations": len(self.selector.stations),
            "station_modes": station_modes,
            "total_detections": len(self.detection_log),
            "total_triggers": len(self.rf.command_log),
            "recent_alerts": [
                d for d in self.detection_log[-10:]
                if d["alert_level"] not in ("nominal", "watch")
            ],
        }


def create_station_network_from_usgs(
    site_info_dir: Path,
    sensor_dir: Path,
) -> List[FixedStation]:
    """Create a FixedStation network from USGS site info and sensor data.

    Parameters
    ----------
    site_info_dir : Path
        Directory with USGS site info JSON files.
    sensor_dir : Path
        Directory with sensor parquet files.

    Returns
    -------
    List of FixedStation objects.
    """
    stations = []
    sensor_ids = {f.stem for f in sensor_dir.glob("*.parquet")}

    for f in sorted(site_info_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            sid = d["site_no"]
            if sid not in sensor_ids:
                continue
            stations.append(FixedStation(
                station_id=sid,
                lat=float(d["lat"]),
                lon=float(d["lon"]),
                name=d.get("station_nm", ""),
                mode=StationMode.STANDBY,
                capabilities=["sensor", "satellite"],
            ))
        except Exception:
            continue

    logger.info("Created station network: %d stations", len(stations))
    return stations
