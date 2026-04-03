"""
SENTINEL-DB: Unified multimodal water quality database layer.

Provides harmonized data models, parameter ontology mapping, H3 spatial
indexing, quality tier assignment, cross-modality linking, and multi-source
data ingest for the SENTINEL framework.
"""

from __future__ import annotations

# --- Schema / data models ---
from sentinel.data.sentinel_db.schema import (
    BehavioralRecording,
    LinkedMultimodalRecord,
    MicrobialSample,
    QualityTier,
    SatelliteObservation,
    TranscriptomicSample,
    WaterQualityRecord,
)

# --- Ontology ---
from sentinel.data.sentinel_db.ontology import (
    CanonicalParameter,
    build_ontology,
    harmonize_unit,
    load_ontology,
    resolve_parameter,
    save_ontology,
)

# --- Spatial ---
from sentinel.data.sentinel_db.spatial import (
    coregister_satellite_to_station,
    deduplicate_records,
    find_nearby_records,
    h3_to_latlon,
    latlon_to_h3,
)

# --- Quality ---
from sentinel.data.sentinel_db.quality import (
    assign_quality_tier,
    check_spatial_consistency,
    check_temporal_consistency,
    promote_tier,
    validate_physical_plausibility,
)

# --- Linking ---
from sentinel.data.sentinel_db.linking import (
    build_linked_record,
    find_ecotox_matches,
    find_microbial_match,
    find_satellite_match,
    link_all,
)

# --- Ingest ---
from sentinel.data.sentinel_db.ingest import (
    ingest_epa_wqp,
    ingest_eu_waterbase,
    ingest_freshwater_watch,
    ingest_gemstat,
    ingest_grqa,
)

__all__ = [
    # Schema
    "QualityTier",
    "WaterQualityRecord",
    "SatelliteObservation",
    "MicrobialSample",
    "TranscriptomicSample",
    "BehavioralRecording",
    "LinkedMultimodalRecord",
    # Ontology
    "CanonicalParameter",
    "resolve_parameter",
    "harmonize_unit",
    "build_ontology",
    "save_ontology",
    "load_ontology",
    # Spatial
    "latlon_to_h3",
    "h3_to_latlon",
    "find_nearby_records",
    "deduplicate_records",
    "coregister_satellite_to_station",
    # Quality
    "assign_quality_tier",
    "validate_physical_plausibility",
    "check_spatial_consistency",
    "check_temporal_consistency",
    "promote_tier",
    # Linking
    "find_satellite_match",
    "find_microbial_match",
    "find_ecotox_matches",
    "build_linked_record",
    "link_all",
    # Ingest
    "ingest_epa_wqp",
    "ingest_eu_waterbase",
    "ingest_gemstat",
    "ingest_grqa",
    "ingest_freshwater_watch",
]
