"""
SENTINEL-DB unified data models.

Pydantic v2 models for the canonical water quality database, covering
in-situ measurements, satellite observations, microbial community data,
transcriptomic samples, and behavioral recordings.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Quality tier enumeration
# ---------------------------------------------------------------------------


class QualityTier(str, Enum):
    """Data quality tier classification.

    Q1 — Certified laboratory analysis (ISO 17025, GLP).
    Q2 — Calibrated in-situ sensors with QA/QC.
    Q3 — Citizen science or unvalidated observations.
    Q4 — Derived, estimated, or modelled values.
    """

    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


# ---------------------------------------------------------------------------
# Canonical water quality record
# ---------------------------------------------------------------------------


class WaterQualityRecord(BaseModel):
    """A single harmonized water quality measurement."""

    canonical_param: str = Field(
        ..., description="Canonical parameter name from the SENTINEL ontology."
    )
    value: float = Field(..., description="Measured value in canonical units.")
    unit: str = Field(..., description="Canonical unit for the parameter.")
    timestamp: datetime = Field(..., description="Measurement timestamp (UTC).")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    h3_index: str = Field(..., description="H3 hexagonal cell index (resolution 8).")
    source: str = Field(
        ..., description="Data source identifier (e.g. 'EPA_WQP', 'EU_Waterbase')."
    )
    quality_tier: QualityTier = Field(
        default=QualityTier.Q3, description="Quality tier assignment."
    )
    raw_param_name: str = Field(
        default="", description="Original parameter name before harmonization."
    )
    raw_unit: str = Field(
        default="", description="Original unit before conversion."
    )
    site_id: str = Field(
        default="", description="Original site/station identifier."
    )

    model_config = {"frozen": False, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Satellite observation
# ---------------------------------------------------------------------------


class SatelliteObservation(BaseModel):
    """A co-registered satellite observation (Sentinel-2/3)."""

    tile_id: str = Field(..., description="Satellite tile identifier.")
    timestamp: datetime
    bands: dict[str, float] = Field(
        default_factory=dict,
        description="Band reflectance values keyed by band name.",
    )
    cloud_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Cloud cover percentage."
    )
    platform: str = Field(
        ..., description="Satellite platform (e.g. 'S2', 'S3')."
    )
    h3_index: str = Field(..., description="H3 cell index for pixel center.")
    resolution_m: float = Field(
        ..., gt=0.0, description="Spatial resolution in meters."
    )

    model_config = {"frozen": False, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Microbial community sample
# ---------------------------------------------------------------------------


class MicrobialSample(BaseModel):
    """An eDNA / 16S amplicon sequencing sample."""

    sample_id: str
    timestamp: datetime
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    h3_index: str
    asv_counts: dict[str, float] = Field(
        default_factory=dict,
        description="ASV/OTU abundance counts keyed by taxon identifier.",
    )
    clr_vector: list[float] = Field(
        default_factory=list,
        description="Centered log-ratio transformed abundance vector.",
    )
    source: str = Field(default="", description="Data source identifier.")
    quality_tier: QualityTier = Field(default=QualityTier.Q2)

    model_config = {"frozen": False, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Transcriptomic sample
# ---------------------------------------------------------------------------


class TranscriptomicSample(BaseModel):
    """A transcriptomic (RNA-seq or microarray) exposure sample."""

    sample_id: str
    species: str = Field(..., description="Organism species name.")
    chemical_exposure: str = Field(
        ..., description="Chemical name or CAS number."
    )
    concentration_mg_l: float = Field(
        ..., ge=0.0, description="Exposure concentration in mg/L."
    )
    platform: str = Field(
        ..., description="Sequencing platform ('rnaseq' or 'microarray')."
    )
    gene_expression: dict[str, float] = Field(
        default_factory=dict,
        description="Gene expression values keyed by gene symbol.",
    )

    model_config = {"frozen": False, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Behavioral recording
# ---------------------------------------------------------------------------


class BehavioralRecording(BaseModel):
    """A behavioral toxicology recording from DAM or video tracking."""

    recording_id: str
    species: str
    timestamp: datetime
    duration_s: float = Field(..., gt=0.0, description="Recording duration in seconds.")
    organism_count: int = Field(..., ge=1, description="Number of organisms tracked.")
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Extracted behavioral features (e.g. velocity, turning_rate).",
    )
    anomaly_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Behavioral anomaly score [0, 1]."
    )

    model_config = {"frozen": False, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Linked multimodal record
# ---------------------------------------------------------------------------


class LinkedMultimodalRecord(BaseModel):
    """A spatiotemporally linked record combining all available modalities."""

    location_h3: str = Field(
        ..., description="H3 cell index for the spatial anchor."
    )
    timestamp: datetime = Field(
        ..., description="Primary timestamp for temporal alignment."
    )
    water_quality: list[WaterQualityRecord] = Field(default_factory=list)
    satellite: Optional[SatelliteObservation] = Field(default=None)
    microbial: Optional[MicrobialSample] = Field(default=None)
    transcriptomic: Optional[TranscriptomicSample] = Field(default=None)
    behavioral: Optional[BehavioralRecording] = Field(default=None)

    @computed_field  # type: ignore[misc]
    @property
    def n_modalities(self) -> int:
        """Count of non-null modalities present in this record."""
        count = 0
        if self.water_quality:
            count += 1
        if self.satellite is not None:
            count += 1
        if self.microbial is not None:
            count += 1
        if self.transcriptomic is not None:
            count += 1
        if self.behavioral is not None:
            count += 1
        return count

    model_config = {"frozen": False, "extra": "forbid"}
