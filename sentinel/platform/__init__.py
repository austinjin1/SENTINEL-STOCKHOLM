"""
SENTINEL platform modules.

Public-facing components for citizen science data ingestion, photo-based
water analysis, test kit validation, environmental justice overlay,
citizen eDNA kit processing, and the research API.

Modules
-------
citizen_qc
    Three-stage automated QC pipeline for citizen-contributed data.
photo_analysis
    Photo-based water body assessment using vision models.
test_kit
    Home water test kit input validation and bias calibration.
ej_overlay
    Environmental Justice dashboard overlay linking SENTINEL alerts
    to EPA EJScreen demographics.
edna_kit
    Citizen science eDNA kit ingestion, validation, and integration.
api
    FastAPI research API for programmatic platform access.
"""

from sentinel.platform.citizen_qc import (
    CitizenQCPipeline,
    CitizenRecord,
    QCReport,
    check_plausibility,
    check_spatial_consistency,
    check_temporal_consistency,
)
from sentinel.platform.edna_kit import (
    SpeciesDetection,
    eDNACommunityAnalyzer,
    eDNAIngestion,
    eDNAKitResult,
    eDNAValidationReport,
    eDNAValidator,
    edna_to_sentinel_embedding,
    submit_to_platform,
)
from sentinel.platform.ej_overlay import (
    CommunityGap,
    EJAnnotatedAlert,
    EJOverlayEngine,
    EJScreenData,
    EJScreenFetcher,
    EquityReport,
    MonitoringSite,
    UndermonitoredCommunitiesFinder,
)
from sentinel.platform.photo_analysis import (
    PhotoAnalysisResult,
    PhotoDataset,
    PhotoWaterAnalyzer,
    cross_reference_with_remote_sensing,
)
from sentinel.platform.test_kit import (
    TEST_KIT_PARAMETERS,
    TestKitReading,
    TestKitSubmission,
    ValidationResult,
    calibrate_kit_readings,
    validate_test_kit_reading,
)

__all__ = [
    # citizen_qc
    "CitizenQCPipeline",
    "CitizenRecord",
    "QCReport",
    "check_plausibility",
    "check_spatial_consistency",
    "check_temporal_consistency",
    # edna_kit
    "SpeciesDetection",
    "eDNACommunityAnalyzer",
    "eDNAIngestion",
    "eDNAKitResult",
    "eDNAValidationReport",
    "eDNAValidator",
    "edna_to_sentinel_embedding",
    "submit_to_platform",
    # ej_overlay
    "CommunityGap",
    "EJAnnotatedAlert",
    "EJOverlayEngine",
    "EJScreenData",
    "EJScreenFetcher",
    "EquityReport",
    "MonitoringSite",
    "UndermonitoredCommunitiesFinder",
    # photo_analysis
    "PhotoAnalysisResult",
    "PhotoDataset",
    "PhotoWaterAnalyzer",
    "cross_reference_with_remote_sensing",
    # test_kit
    "TEST_KIT_PARAMETERS",
    "TestKitReading",
    "TestKitSubmission",
    "ValidationResult",
    "calibrate_kit_readings",
    "validate_test_kit_reading",
]
