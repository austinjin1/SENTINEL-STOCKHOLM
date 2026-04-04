"""
SENTINEL platform modules.

Public-facing components for citizen science data ingestion, photo-based
water analysis, test kit validation, and the research API.

Modules
-------
citizen_qc
    Three-stage automated QC pipeline for citizen-contributed data.
photo_analysis
    Photo-based water body assessment using vision models.
test_kit
    Home water test kit input validation and bias calibration.
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
