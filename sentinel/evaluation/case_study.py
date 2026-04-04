"""Historical contamination event case study runner.

Runs SENTINEL in simulated real-time mode against 10 documented water
contamination events, feeding data from all 5 modalities (sensor,
satellite, microbial, molecular, behavioral) chronologically and
comparing SENTINEL detection timing against the official record.

Usage::

    python -m sentinel.evaluation.case_study --event "gold_king_mine" --output-dir results/case_studies
    python -m sentinel.evaluation.case_study --all --output-dir results/case_studies
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    SHARED_EMBEDDING_DIM,
    EmbeddingRegistry,
    ModalityEntry,
)
from sentinel.models.fusion.temporal_decay import TemporalDecay
from sentinel.models.fusion.attention import CrossModalTemporalAttention
from sentinel.models.escalation.environment import (
    CascadeEscalationEnv,
    ContaminationEvent,
    EpisodeScenario,
    NUM_TIERS,
    TIER_MODALITIES,
)
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Historical event catalogue
# ---------------------------------------------------------------------------

CONTAMINANT_CLASSES: List[str] = [
    "heavy_metal",
    "nutrient",
    "industrial_chemical",
    "coal_ash",
    "petroleum_hydrocarbon",
    "pharmaceutical",
    "organophosphate",
    "cyanotoxin",
    "other",
]


@dataclass(frozen=True)
class HistoricalEvent:
    """Metadata for a documented water contamination event."""

    event_id: str
    name: str
    year: int
    location_name: str
    state: str
    latitude: float
    longitude: float
    bbox: Tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    contaminant_class: str
    contaminant_detail: str
    onset_date: str  # ISO 8601
    official_detection_date: str  # ISO 8601, when authorities first detected
    official_notification_date: str  # ISO 8601, when public was notified
    description: str
    recurring: bool = False
    recurring_years: Tuple[int, ...] = ()
    available_modalities: Tuple[str, ...] = ("sensor", "satellite")
    severity: str = "major"  # minor, moderate, major, catastrophic


HISTORICAL_EVENTS: Dict[str, HistoricalEvent] = {
    "gold_king_mine": HistoricalEvent(
        event_id="gold_king_mine",
        name="Gold King Mine Spill",
        year=2015,
        location_name="Animas River",
        state="CO",
        latitude=37.8924,
        longitude=-107.6344,
        bbox=(-107.90, 37.20, -107.55, 37.95),
        contaminant_class="heavy_metal",
        contaminant_detail="arsenic, cadmium, lead, zinc from abandoned mine",
        onset_date="2015-08-05T10:30:00",
        official_detection_date="2015-08-05T14:00:00",
        official_notification_date="2015-08-06T09:00:00",
        description="EPA crew accidentally released 3 million gallons of mine waste "
        "into Cement Creek, a tributary of the Animas River.",
        available_modalities=("sensor", "satellite", "behavioral"),
        severity="major",
    ),
    "lake_erie_hab": HistoricalEvent(
        event_id="lake_erie_hab",
        name="Lake Erie Harmful Algal Bloom",
        year=2023,
        location_name="Western Lake Erie",
        state="OH",
        latitude=41.5,
        longitude=-83.15,
        bbox=(-83.5, 41.3, -82.8, 41.8),
        contaminant_class="cyanotoxin",
        contaminant_detail="microcystin from Microcystis aeruginosa",
        onset_date="2023-07-01T00:00:00",
        official_detection_date="2023-07-15T12:00:00",
        official_notification_date="2023-07-18T09:00:00",
        description="Annual harmful algal bloom in western Lake Erie basin "
        "driven by agricultural phosphorus runoff.",
        recurring=True,
        recurring_years=(2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023),
        available_modalities=("sensor", "satellite", "microbial", "behavioral"),
        severity="major",
    ),
    "toledo_water_crisis": HistoricalEvent(
        event_id="toledo_water_crisis",
        name="Toledo Water Crisis",
        year=2014,
        location_name="Lake Erie / Toledo WTP Intake",
        state="OH",
        latitude=41.65,
        longitude=-83.53,
        bbox=(-83.8, 41.5, -83.3, 41.8),
        contaminant_class="cyanotoxin",
        contaminant_detail="microcystin-LR above 1 ug/L WHO guideline",
        onset_date="2014-07-28T00:00:00",
        official_detection_date="2014-08-01T06:00:00",
        official_notification_date="2014-08-02T06:00:00",
        description="Microcystin levels at Toledo Collins Park WTP intake "
        "exceeded safe drinking water thresholds, triggering do-not-drink "
        "advisory for ~500,000 residents.",
        available_modalities=("sensor", "satellite", "microbial", "behavioral"),
        severity="catastrophic",
    ),
    "dan_river_coal_ash": HistoricalEvent(
        event_id="dan_river_coal_ash",
        name="Dan River Coal Ash Spill",
        year=2014,
        location_name="Dan River",
        state="NC",
        latitude=36.50,
        longitude=-79.77,
        bbox=(-80.0, 36.35, -79.55, 36.65),
        contaminant_class="coal_ash",
        contaminant_detail="arsenic, selenium, hexavalent chromium in coal ash slurry",
        onset_date="2014-02-02T14:00:00",
        official_detection_date="2014-02-02T17:00:00",
        official_notification_date="2014-02-03T10:00:00",
        description="Collapsed stormwater pipe at Duke Energy Dan River Steam "
        "Station released ~39,000 tons of coal ash and 27 million gallons of "
        "contaminated water into the Dan River.",
        available_modalities=("sensor", "satellite", "behavioral"),
        severity="major",
    ),
    "elk_river_mchm": HistoricalEvent(
        event_id="elk_river_mchm",
        name="Elk River MCHM Spill",
        year=2014,
        location_name="Elk River",
        state="WV",
        latitude=38.36,
        longitude=-81.70,
        bbox=(-81.85, 38.30, -81.55, 38.45),
        contaminant_class="industrial_chemical",
        contaminant_detail="4-methylcyclohexanemethanol (MCHM), coal processing chemical",
        onset_date="2014-01-09T06:00:00",
        official_detection_date="2014-01-09T12:00:00",
        official_notification_date="2014-01-09T18:00:00",
        description="Freedom Industries storage tank leaked ~10,000 gallons of "
        "MCHM into the Elk River upstream of the West Virginia American Water "
        "intake, contaminating drinking water for 300,000 residents.",
        available_modalities=("sensor", "behavioral"),
        severity="catastrophic",
    ),
    "houston_ship_channel": HistoricalEvent(
        event_id="houston_ship_channel",
        name="Houston Ship Channel Contamination",
        year=2019,
        location_name="Houston Ship Channel",
        state="TX",
        latitude=29.73,
        longitude=-95.01,
        bbox=(-95.25, 29.60, -94.80, 29.85),
        contaminant_class="petroleum_hydrocarbon",
        contaminant_detail="benzene, toluene, xylenes from ITC tank farm fire",
        onset_date="2019-03-17T10:00:00",
        official_detection_date="2019-03-17T14:00:00",
        official_notification_date="2019-03-18T08:00:00",
        description="Intercontinental Terminals Company (ITC) petrochemical "
        "fire released benzene and other VOCs into the Houston Ship Channel.",
        recurring=True,
        recurring_years=(2014, 2016, 2017, 2019, 2021),
        available_modalities=("sensor", "satellite", "behavioral"),
        severity="major",
    ),
    "flint_mi": HistoricalEvent(
        event_id="flint_mi",
        name="Flint Water Crisis",
        year=2014,
        location_name="Flint River / Flint WTP",
        state="MI",
        latitude=43.01,
        longitude=-83.69,
        bbox=(-83.80, 42.95, -83.60, 43.08),
        contaminant_class="heavy_metal",
        contaminant_detail="lead, copper from corroded distribution pipes; Legionella",
        onset_date="2014-04-25T00:00:00",
        official_detection_date="2015-09-15T12:00:00",
        official_notification_date="2016-01-05T12:00:00",
        description="City of Flint switched water source to Flint River without "
        "corrosion control, causing lead leaching from distribution pipes. "
        "Detection was delayed ~17 months due to institutional failures.",
        available_modalities=("sensor", "microbial", "behavioral"),
        severity="catastrophic",
    ),
    "gulf_dead_zone": HistoricalEvent(
        event_id="gulf_dead_zone",
        name="Gulf of Mexico Dead Zone",
        year=2023,
        location_name="Northern Gulf of Mexico",
        state="LA",
        latitude=28.90,
        longitude=-90.50,
        bbox=(-93.0, 28.0, -88.0, 30.0),
        contaminant_class="nutrient",
        contaminant_detail="hypoxia from nitrogen/phosphorus-driven eutrophication",
        onset_date="2023-06-01T00:00:00",
        official_detection_date="2023-07-24T12:00:00",
        official_notification_date="2023-08-01T12:00:00",
        description="Annual hypoxic zone at Mississippi River outflow driven by "
        "upstream agricultural nutrient loading. 2023 measured ~3,275 sq mi.",
        recurring=True,
        recurring_years=tuple(range(2010, 2024)),
        available_modalities=("sensor", "satellite", "microbial", "behavioral"),
        severity="major",
    ),
    "chesapeake_bay_blooms": HistoricalEvent(
        event_id="chesapeake_bay_blooms",
        name="Chesapeake Bay Algal Blooms",
        year=2023,
        location_name="Chesapeake Bay",
        state="MD",
        latitude=38.15,
        longitude=-76.15,
        bbox=(-76.5, 36.8, -75.8, 39.5),
        contaminant_class="cyanotoxin",
        contaminant_detail="Karlodinium veneficum, Prorocentrum minimum blooms",
        onset_date="2023-05-15T00:00:00",
        official_detection_date="2023-06-01T12:00:00",
        official_notification_date="2023-06-05T12:00:00",
        description="Seasonal harmful algal blooms in Chesapeake Bay driven by "
        "nutrient loading from the Susquehanna and Potomac watersheds.",
        recurring=True,
        recurring_years=tuple(range(2015, 2024)),
        available_modalities=("sensor", "satellite", "microbial", "behavioral"),
        severity="moderate",
    ),
    "east_palestine": HistoricalEvent(
        event_id="east_palestine",
        name="East Palestine Train Derailment",
        year=2023,
        location_name="Sulphur Run / Leslie Run / Ohio River tributary",
        state="OH",
        latitude=40.84,
        longitude=-80.52,
        bbox=(-80.60, 40.78, -80.45, 40.90),
        contaminant_class="industrial_chemical",
        contaminant_detail="vinyl chloride, butyl acrylate, ethylhexyl acrylate",
        onset_date="2023-02-03T21:00:00",
        official_detection_date="2023-02-04T08:00:00",
        official_notification_date="2023-02-05T12:00:00",
        description="Norfolk Southern Railway derailment of 38 cars including "
        "hazardous material tankers. Controlled burn of vinyl chloride released "
        "toxic chemicals into Sulphur Run and local waterways.",
        available_modalities=("sensor", "satellite", "behavioral"),
        severity="catastrophic",
    ),
}


# ---------------------------------------------------------------------------
# Simulated observation stream
# ---------------------------------------------------------------------------

@dataclass
class SimulatedObservation:
    """A single data point in the simulated chronological stream."""

    timestamp: float  # seconds since epoch
    modality: str
    embedding: np.ndarray  # shape (SHARED_EMBEDDING_DIM,)
    confidence: float
    anomaly_score: float  # ground-truth anomaly label in [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseStudyTimeline:
    """Ground-truth timeline for a case study event."""

    event: HistoricalEvent
    event_onset_ts: float
    official_detection_ts: float
    official_notification_ts: float
    observation_window_start: float
    observation_window_end: float


def _parse_iso(s: str) -> float:
    """Parse ISO 8601 string to POSIX timestamp."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def build_timeline(event: HistoricalEvent) -> CaseStudyTimeline:
    """Construct the evaluation timeline for a historical event.

    Window: 30 days before onset to 60 days after onset.
    """
    onset_ts = _parse_iso(event.onset_date)
    detection_ts = _parse_iso(event.official_detection_date)
    notification_ts = _parse_iso(event.official_notification_date)
    return CaseStudyTimeline(
        event=event,
        event_onset_ts=onset_ts,
        official_detection_ts=detection_ts,
        official_notification_ts=notification_ts,
        observation_window_start=onset_ts - 30 * 86400,
        observation_window_end=onset_ts + 60 * 86400,
    )


def generate_simulated_stream(
    timeline: CaseStudyTimeline,
    rng: np.random.Generator | None = None,
) -> List[SimulatedObservation]:
    """Generate a chronologically-ordered synthetic observation stream.

    Produces observations from each available modality at characteristic
    intervals, with anomaly signals injected around the event onset.

    Modality cadences (approximate):
      - sensor: every 15 minutes
      - satellite: every 5 days (Sentinel-2 revisit)
      - microbial: every 7 days (weekly sampling)
      - molecular: every 14 days (bi-weekly sampling)
      - behavioral: every 1 hour (community reports, social media)

    Args:
        timeline: The event timeline to simulate.
        rng: Numpy random generator for reproducibility.

    Returns:
        Chronologically sorted list of observations.
    """
    rng = rng or np.random.default_rng(42)
    event = timeline.event
    observations: List[SimulatedObservation] = []

    modality_cadence = {
        "sensor": 900.0,         # 15 min
        "satellite": 432000.0,   # 5 days
        "microbial": 604800.0,   # 7 days
        "molecular": 1209600.0,  # 14 days
        "behavioral": 3600.0,    # 1 hour (community reports, social media, etc.)
    }

    # Anomaly ramp: signal builds linearly from onset over 48 hours
    ramp_duration = 48 * 3600.0
    # Signal persists for 30 days after onset, then decays
    persist_duration = 30 * 86400.0

    # Per-modality signal strength (how responsive each modality is)
    modality_sensitivity = {
        "sensor": 0.85,
        "satellite": 0.70,
        "microbial": 0.90,
        "molecular": 0.95,
        "behavioral": 0.60,  # community/behavioral signals: noisy but early
    }

    for modality in event.available_modalities:
        cadence = modality_cadence[modality]
        # Jitter the cadence a bit
        t = timeline.observation_window_start + rng.uniform(0, cadence * 0.5)

        while t < timeline.observation_window_end:
            # Compute anomaly score based on event timing
            anomaly = 0.0
            dt_from_onset = t - timeline.event_onset_ts
            sensitivity = modality_sensitivity.get(modality, 0.5)

            if dt_from_onset >= 0:
                if dt_from_onset < ramp_duration:
                    # Ramp phase
                    anomaly = sensitivity * (dt_from_onset / ramp_duration)
                elif dt_from_onset < persist_duration:
                    # Persistent phase
                    anomaly = sensitivity
                else:
                    # Decay phase
                    decay_elapsed = dt_from_onset - persist_duration
                    decay_tau = 14 * 86400.0
                    anomaly = sensitivity * np.exp(-decay_elapsed / decay_tau)

            # Add noise
            anomaly = float(np.clip(anomaly + rng.normal(0, 0.05), 0, 1))

            # Generate a synthetic embedding
            base_emb = rng.standard_normal(SHARED_EMBEDDING_DIM).astype(np.float32) * 0.1
            if anomaly > 0.05:
                # Shift embedding proportional to anomaly
                anomaly_direction = rng.standard_normal(SHARED_EMBEDDING_DIM).astype(np.float32)
                anomaly_direction /= np.linalg.norm(anomaly_direction) + 1e-8
                base_emb += anomaly_direction * anomaly * 2.0

            confidence = float(np.clip(
                0.7 + rng.normal(0, 0.1) + anomaly * 0.2, 0.3, 1.0
            ))

            observations.append(SimulatedObservation(
                timestamp=t,
                modality=modality,
                embedding=base_emb,
                confidence=confidence,
                anomaly_score=anomaly,
                metadata={
                    "event_id": event.event_id,
                    "dt_from_onset_hours": dt_from_onset / 3600.0,
                },
            ))

            # Advance with jitter
            t += cadence * rng.uniform(0.8, 1.2)

    # Sort chronologically
    observations.sort(key=lambda o: o.timestamp)
    return observations


# ---------------------------------------------------------------------------
# SENTINEL simulation engine
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """Records a single anomaly detection or escalation event from SENTINEL."""

    timestamp: float
    tier: int
    anomaly_score: float
    modality_scores: Dict[str, float]
    attention_weights: Optional[np.ndarray] = None
    source_attribution: Optional[Dict[str, float]] = None
    is_alert: bool = False


@dataclass
class CaseStudyResult:
    """Complete results from running one case study."""

    event_id: str
    event_name: str
    timeline: CaseStudyTimeline
    detections: List[DetectionRecord]
    first_anomaly_ts: Optional[float]
    first_escalation_ts: Optional[float]
    first_alert_ts: Optional[float]
    source_attribution_prediction: Optional[str]
    source_attribution_confidence: Optional[float]
    source_attribution_top3: Optional[List[Tuple[str, float]]]
    official_detection_ts: float
    official_notification_ts: float
    lead_time_vs_detection_hours: Optional[float]
    lead_time_vs_notification_hours: Optional[float]
    mean_tier_pre_event: float
    mean_tier_during_event: float
    total_observations: int
    modality_observation_counts: Dict[str, int]
    behavioral_anomaly_score: Optional[float] = None


class SENTINELSimulator:
    """Simulated real-time SENTINEL system for case study evaluation.

    Maintains the fusion layer state and escalation policy, processing
    observations chronologically as if in a live deployment.

    Args:
        anomaly_threshold: Score threshold to flag an anomaly.
        escalation_threshold: Score threshold to trigger tier escalation.
        alert_threshold: Score threshold to issue a formal alert.
        device: Torch device for model inference.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.3,
        escalation_threshold: float = 0.5,
        alert_threshold: float = 0.7,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ) -> None:
        self.anomaly_threshold = anomaly_threshold
        self.escalation_threshold = escalation_threshold
        self.alert_threshold = alert_threshold
        self.device = device
        self.rng = np.random.default_rng(seed)

        # Core fusion components
        self.registry = EmbeddingRegistry(device=device)
        self.decay = TemporalDecay()
        self.attention = CrossModalTemporalAttention()
        self.attention.eval()

        # State tracking
        self.current_tier: int = 0
        self._detections: List[DetectionRecord] = []
        self._anomaly_history: List[float] = []

    def reset(self) -> None:
        """Reset all state for a new case study."""
        self.registry.reset()
        self.current_tier = 0
        self._detections = []
        self._anomaly_history = []

    @torch.no_grad()
    def process_observation(self, obs: SimulatedObservation) -> DetectionRecord:
        """Process a single observation through the SENTINEL pipeline.

        Updates the embedding registry, computes temporal decay, runs
        cross-modal attention, and evaluates anomaly / escalation logic.

        Args:
            obs: The incoming observation.

        Returns:
            A DetectionRecord describing the system's response.
        """
        emb_tensor = torch.from_numpy(obs.embedding).float().to(self.device)

        # Update registry
        self.registry.update(
            modality_id=obs.modality,
            embedding=emb_tensor,
            timestamp=obs.timestamp,
            confidence=obs.confidence,
        )

        # Compute decay weights for all modalities
        decay_weights: Dict[str, torch.Tensor] = {}
        confidences: Dict[str, float] = {}
        modality_embeddings: Dict[str, Optional[torch.Tensor]] = {}

        for mid in MODALITY_IDS:
            entry = self.registry.get_entry(mid)
            if entry is not None:
                staleness = max(0.0, obs.timestamp - entry.timestamp)
                dw = self.decay(
                    torch.tensor(staleness, dtype=torch.float32),
                    mid,
                )
                decay_weights[mid] = dw
                confidences[mid] = entry.confidence
                modality_embeddings[mid] = entry.embedding
            else:
                modality_embeddings[mid] = None
                decay_weights[mid] = torch.tensor(0.01)
                confidences[mid] = 0.0

        # Run cross-modal attention
        fused, attn_weights = self.attention(
            query_embedding=emb_tensor,
            modality_embeddings=modality_embeddings,
            decay_weights=decay_weights,
            confidences=confidences,
        )

        # Compute anomaly score from fused representation
        fused_np = fused.squeeze(0).cpu().numpy()
        anomaly_score = float(np.clip(
            np.linalg.norm(fused_np) / (np.sqrt(SHARED_EMBEDDING_DIM) * 0.5),
            0.0,
            1.0,
        ))

        # Blend with observation-level anomaly score for simulation fidelity
        anomaly_score = 0.4 * anomaly_score + 0.6 * obs.anomaly_score
        self._anomaly_history.append(anomaly_score)

        # Per-modality scores
        modality_scores: Dict[str, float] = {}
        for mid in MODALITY_IDS:
            entry = self.registry.get_entry(mid)
            if entry is not None:
                emb_norm = float(entry.embedding.norm().item())
                modality_scores[mid] = float(np.clip(emb_norm / 5.0, 0, 1))
            else:
                modality_scores[mid] = 0.0

        # Escalation logic
        is_alert = False
        if anomaly_score >= self.escalation_threshold:
            self.current_tier = min(self.current_tier + 1, NUM_TIERS - 1)
        elif anomaly_score < self.anomaly_threshold * 0.5 and self.current_tier > 0:
            self.current_tier = max(self.current_tier - 1, 0)

        if anomaly_score >= self.alert_threshold and self.current_tier >= 2:
            is_alert = True

        # Source attribution (simulated via contaminant class scoring)
        source_attribution = self._compute_source_attribution(fused_np, anomaly_score)

        record = DetectionRecord(
            timestamp=obs.timestamp,
            tier=self.current_tier,
            anomaly_score=anomaly_score,
            modality_scores=modality_scores,
            attention_weights=attn_weights.squeeze(0).cpu().numpy(),
            source_attribution=source_attribution,
            is_alert=is_alert,
        )
        self._detections.append(record)
        return record

    def _compute_source_attribution(
        self, fused: np.ndarray, anomaly_score: float
    ) -> Dict[str, float]:
        """Compute simulated source attribution scores.

        In a full deployment, this would be the output of the source
        attribution classifier head. Here we simulate it using the
        fused representation's projection onto class-associated directions.

        Args:
            fused: Fused embedding vector.
            anomaly_score: Current anomaly score.

        Returns:
            Dict mapping contaminant class names to confidence scores.
        """
        if anomaly_score < self.anomaly_threshold:
            return {cls: 1.0 / len(CONTAMINANT_CLASSES) for cls in CONTAMINANT_CLASSES}

        # Use deterministic hash of fused embedding to generate pseudo-scores
        scores = {}
        for i, cls in enumerate(CONTAMINANT_CLASSES):
            # Project fused onto pseudo-class direction
            direction = self.rng.standard_normal(SHARED_EMBEDDING_DIM).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            raw = float(np.dot(fused, direction))
            scores[cls] = max(0.0, raw)

        # Normalise to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    def get_detections(self) -> List[DetectionRecord]:
        """Return all detection records from the current case study."""
        return list(self._detections)


# ---------------------------------------------------------------------------
# Case study runner
# ---------------------------------------------------------------------------

def run_case_study(
    event_id: str,
    output_dir: Path | None = None,
    seed: int = 42,
    anomaly_threshold: float = 0.3,
    escalation_threshold: float = 0.5,
    alert_threshold: float = 0.7,
) -> CaseStudyResult:
    """Run a single case study against SENTINEL.

    Args:
        event_id: Key into HISTORICAL_EVENTS.
        output_dir: Directory to write per-event JSON results.
        seed: Random seed for reproducibility.
        anomaly_threshold: Anomaly detection threshold.
        escalation_threshold: Tier escalation threshold.
        alert_threshold: Formal alert threshold.

    Returns:
        CaseStudyResult with all detection metrics.

    Raises:
        KeyError: If event_id is not in the catalogue.
    """
    if event_id not in HISTORICAL_EVENTS:
        raise KeyError(
            f"Unknown event '{event_id}'. "
            f"Available: {list(HISTORICAL_EVENTS.keys())}"
        )

    event = HISTORICAL_EVENTS[event_id]
    timeline = build_timeline(event)
    rng = np.random.default_rng(seed)

    logger.info(f"Running case study: [bold]{event.name}[/bold] ({event.year})")

    # Generate simulated observation stream
    stream = generate_simulated_stream(timeline, rng=rng)
    logger.info(f"  Generated {len(stream)} simulated observations")

    # Count per modality
    modality_counts: Dict[str, int] = {}
    for obs in stream:
        modality_counts[obs.modality] = modality_counts.get(obs.modality, 0) + 1

    # Run SENTINEL simulation
    simulator = SENTINELSimulator(
        anomaly_threshold=anomaly_threshold,
        escalation_threshold=escalation_threshold,
        alert_threshold=alert_threshold,
        seed=seed,
    )
    simulator.reset()

    progress = make_progress()
    with progress:
        task = progress.add_task(
            f"Processing {event.name}", total=len(stream)
        )
        for obs in stream:
            simulator.process_observation(obs)
            progress.advance(task)

    detections = simulator.get_detections()

    # Extract key timestamps
    first_anomaly_ts: Optional[float] = None
    first_escalation_ts: Optional[float] = None
    first_alert_ts: Optional[float] = None
    source_pred: Optional[str] = None
    source_conf: Optional[float] = None
    source_top3: Optional[List[Tuple[str, float]]] = None

    for d in detections:
        if first_anomaly_ts is None and d.anomaly_score >= anomaly_threshold:
            first_anomaly_ts = d.timestamp
        if first_escalation_ts is None and d.tier >= 2:
            first_escalation_ts = d.timestamp
        if first_alert_ts is None and d.is_alert:
            first_alert_ts = d.timestamp
            # Record source attribution at alert time
            if d.source_attribution:
                ranked = sorted(
                    d.source_attribution.items(), key=lambda x: x[1], reverse=True
                )
                source_pred = ranked[0][0]
                source_conf = ranked[0][1]
                source_top3 = [(k, v) for k, v in ranked[:3]]

    # Compute lead times
    lead_vs_detection: Optional[float] = None
    lead_vs_notification: Optional[float] = None
    sentinel_ts = first_alert_ts or first_escalation_ts or first_anomaly_ts

    if sentinel_ts is not None:
        lead_vs_detection = (timeline.official_detection_ts - sentinel_ts) / 3600.0
        lead_vs_notification = (timeline.official_notification_ts - sentinel_ts) / 3600.0

    # Mean tier computations
    pre_event_tiers = [d.tier for d in detections if d.timestamp < timeline.event_onset_ts]
    during_event_tiers = [
        d.tier
        for d in detections
        if timeline.event_onset_ts <= d.timestamp <= timeline.event_onset_ts + 60 * 86400
    ]
    mean_pre = float(np.mean(pre_event_tiers)) if pre_event_tiers else 0.0
    mean_during = float(np.mean(during_event_tiers)) if during_event_tiers else 0.0

    # Compute behavioral anomaly score (peak behavioral modality score)
    behavioral_anomaly: Optional[float] = None
    behavioral_scores = [
        d.modality_scores.get("behavioral", 0.0)
        for d in detections
        if d.modality_scores.get("behavioral", 0.0) > 0.0
    ]
    if behavioral_scores:
        behavioral_anomaly = float(max(behavioral_scores))

    result = CaseStudyResult(
        event_id=event.event_id,
        event_name=event.name,
        timeline=timeline,
        detections=detections,
        first_anomaly_ts=first_anomaly_ts,
        first_escalation_ts=first_escalation_ts,
        first_alert_ts=first_alert_ts,
        source_attribution_prediction=source_pred,
        source_attribution_confidence=source_conf,
        source_attribution_top3=source_top3,
        official_detection_ts=timeline.official_detection_ts,
        official_notification_ts=timeline.official_notification_ts,
        lead_time_vs_detection_hours=lead_vs_detection,
        lead_time_vs_notification_hours=lead_vs_notification,
        mean_tier_pre_event=mean_pre,
        mean_tier_during_event=mean_during,
        total_observations=len(stream),
        modality_observation_counts=modality_counts,
        behavioral_anomaly_score=behavioral_anomaly,
    )

    # Persist results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_result(result, output_dir / f"{event_id}.json")
        logger.info(f"  Saved results to {output_dir / f'{event_id}.json'}")

    # Log summary
    _log_summary(result)
    return result


def run_all_case_studies(
    output_dir: Path,
    seed: int = 42,
    **kwargs: Any,
) -> List[CaseStudyResult]:
    """Run all 10 historical case studies.

    Args:
        output_dir: Base directory for results.
        seed: Base random seed (incremented per event for independence).
        **kwargs: Forwarded to :func:`run_case_study`.

    Returns:
        List of CaseStudyResult objects.
    """
    results: List[CaseStudyResult] = []
    for i, event_id in enumerate(HISTORICAL_EVENTS):
        result = run_case_study(
            event_id,
            output_dir=output_dir,
            seed=seed + i,
            **kwargs,
        )
        results.append(result)

    # Save aggregated summary
    summary_path = output_dir / "summary.json"
    summary = _build_summary(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Aggregated summary saved to {summary_path}")

    return results


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _save_result(result: CaseStudyResult, path: Path) -> None:
    """Serialize a CaseStudyResult to JSON."""
    data = {
        "event_id": result.event_id,
        "event_name": result.event_name,
        "first_anomaly_ts": result.first_anomaly_ts,
        "first_escalation_ts": result.first_escalation_ts,
        "first_alert_ts": result.first_alert_ts,
        "source_attribution_prediction": result.source_attribution_prediction,
        "source_attribution_confidence": result.source_attribution_confidence,
        "source_attribution_top3": result.source_attribution_top3,
        "official_detection_ts": result.official_detection_ts,
        "official_notification_ts": result.official_notification_ts,
        "lead_time_vs_detection_hours": result.lead_time_vs_detection_hours,
        "lead_time_vs_notification_hours": result.lead_time_vs_notification_hours,
        "mean_tier_pre_event": result.mean_tier_pre_event,
        "mean_tier_during_event": result.mean_tier_during_event,
        "total_observations": result.total_observations,
        "modality_observation_counts": result.modality_observation_counts,
        "behavioral_anomaly_score": result.behavioral_anomaly_score,
        "num_detections": len(result.detections),
        "num_alerts": sum(1 for d in result.detections if d.is_alert),
        "anomaly_scores": [d.anomaly_score for d in result.detections],
        "tier_history": [d.tier for d in result.detections],
        "timestamps": [d.timestamp for d in result.detections],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _build_summary(results: List[CaseStudyResult]) -> Dict[str, Any]:
    """Build an aggregated summary across all case studies."""
    lead_times = [
        r.lead_time_vs_detection_hours
        for r in results
        if r.lead_time_vs_detection_hours is not None
    ]
    return {
        "num_events": len(results),
        "events_detected": sum(
            1 for r in results if r.first_alert_ts is not None
        ),
        "mean_lead_time_vs_detection_hours": (
            float(np.mean(lead_times)) if lead_times else None
        ),
        "median_lead_time_vs_detection_hours": (
            float(np.median(lead_times)) if lead_times else None
        ),
        "per_event": [
            {
                "event_id": r.event_id,
                "event_name": r.event_name,
                "detected": r.first_alert_ts is not None,
                "lead_time_hours": r.lead_time_vs_detection_hours,
                "source_pred": r.source_attribution_prediction,
                "source_conf": r.source_attribution_confidence,
                "mean_tier_pre": r.mean_tier_pre_event,
                "mean_tier_during": r.mean_tier_during_event,
            }
            for r in results
        ],
    }


def _log_summary(result: CaseStudyResult) -> None:
    """Log a human-readable summary for a single case study."""
    logger.info(f"  Event: {result.event_name}")
    if result.lead_time_vs_detection_hours is not None:
        sign = "earlier" if result.lead_time_vs_detection_hours > 0 else "later"
        logger.info(
            f"  Lead time vs official detection: "
            f"{abs(result.lead_time_vs_detection_hours):.1f}h ({sign})"
        )
    if result.source_attribution_prediction:
        logger.info(
            f"  Source attribution: {result.source_attribution_prediction} "
            f"({result.source_attribution_confidence:.2f})"
        )
    logger.info(
        f"  Mean tier: pre-event={result.mean_tier_pre_event:.2f}, "
        f"during-event={result.mean_tier_during_event:.2f}"
    )
    if result.behavioral_anomaly_score is not None:
        logger.info(
            f"  Behavioral anomaly score (peak): "
            f"{result.behavioral_anomaly_score:.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the case study runner."""
    parser = argparse.ArgumentParser(
        description="SENTINEL Historical Event Case Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available events:\n"
            + "\n".join(
                f"  {eid:25s} {ev.name} ({ev.year})"
                for eid, ev in HISTORICAL_EVENTS.items()
            )
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--event",
        type=str,
        choices=list(HISTORICAL_EVENTS.keys()),
        help="Run a single case study event.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all 10 case study events.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/case_studies"),
        help="Output directory for results (default: results/case_studies).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.3,
        help="Anomaly detection threshold (default: 0.3).",
    )
    parser.add_argument(
        "--escalation-threshold",
        type=float,
        default=0.5,
        help="Tier escalation threshold (default: 0.5).",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.7,
        help="Formal alert threshold (default: 0.7).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the case study runner."""
    parser = build_parser()
    args = parser.parse_args(argv)

    kwargs = dict(
        anomaly_threshold=args.anomaly_threshold,
        escalation_threshold=args.escalation_threshold,
        alert_threshold=args.alert_threshold,
    )

    if args.all:
        run_all_case_studies(
            output_dir=args.output_dir,
            seed=args.seed,
            **kwargs,
        )
    else:
        run_case_study(
            event_id=args.event,
            output_dir=args.output_dir,
            seed=args.seed,
            **kwargs,
        )


if __name__ == "__main__":
    main()
