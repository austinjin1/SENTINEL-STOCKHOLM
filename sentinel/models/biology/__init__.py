"""Biological assessment models for SENTINEL.

Modules
-------
species_health
    Sentinel Species Health Index -- hierarchical occupancy + abundance
    model predicting daily health indices for keystone freshwater species.

disease_forecast
    Disease outbreak forecasting heads (Phase 3.3).  Cyanotoxin
    concentration, Vibrio risk, Naegleria fowleri habitat, and
    schistosomiasis snail-host habitat forecasters.

arg_surveillance
    Antibiotic resistance gene (ARG) surveillance from 16S community
    composition (Phase 3.4).  Predicts ARG abundance, environmental
    burden score, and temporal resistance trends.

occupancy
    Species Occupancy + eDNA Community Forecasting (Phase 3.5).
    Multi-species occupancy shift prediction and generative 16S
    community composition from environmental embeddings.

field_dose_response
    Lab-to-Field Dose-Response Extrapolation (Phase 3.6).  Bridges
    268K ECOTOX lab records to field observations with species
    sensitivity distributions.

inverse_aop
    Inverse AOP-Wiki Pathway Activation (Phase 3.7).  Predicts
    adverse outcome pathway activation from environmental data
    without molecular biomarkers.

metatranscriptomic
    Environmental Metatranscriptomic Pathogen Surveillance (Phase 3.8).
    Known and novel pathogen detection from environmental RNA-seq.

bioremediation
    Bioremediation Recommender (Phase 4).  Prescribes remediation
    strategies based on microbial community and pollutant class,
    with degrader detection, amendment selection, and outcome
    prediction.
"""

from .disease_forecast import (
    AlertLevel,
    ConformalCalibrator,
    CyanotoxinForecaster,
    CyanotoxinOutput,
    DiseaseForecaster,
    DiseaseRiskSummary,
    IntegratedDiseaseRisk,
    NaegleriaForecaster,
    NaegleriaOutput,
    SchistosomiasisForecaster,
    SchistosomiasisOutput,
    VibrioOutput,
    VibrioRiskForecaster,
    encode_temporal_context,
)

from .arg_surveillance import (
    ARGPredictor,
    ARGPredictionOutput,
    ARGSurveillanceLoss,
    ARGSurveillancePipeline,
    ARGBurdenIndex,
    BurdenAlertLevel,
    CommunityToARGMapper,
    TemporalARGAlert,
    TemporalARGTracker,
    WHOPriority,
)

from .occupancy import (
    OccupancyShiftModel,
    OccupancyShiftOutput,
    eDNACommunityPredictor,
    eDNACommunityOutput,
    TaxonomicPriorModule,
)

from .field_dose_response import (
    FieldDoseResponseModel,
    FieldDoseResponseOutput,
    DoseResponseCurveParams,
    SSDOutput,
    SpeciesSensitivityDistribution,
)

from .inverse_aop import (
    InverseAOPPredictor,
    AOPPredictionOutput,
    AOPCategory,
    AOPAlertLevel,
)

from .metatranscriptomic import (
    MetatranscriptomicSurveillance,
    NovelPathogenDetector,
    EnvironmentalSurveillancePipeline,
    PathogenSignatureDB,
    EnvironmentalRNAEncoder,
    PathogenDetectionOutput,
    SurveillanceOutput,
    PathogenAlertLevel,
)

from .bioremediation import (
    AMENDMENT_CATALOG,
    BioremediationRecommender as BiologyBioremediationRecommender,
    CONTAMINANT_CLASSES,
    DEGRADER_DATABASE,
    RemediationRecommendation,
)

__all__ = [
    # Base
    "DiseaseForecaster",
    # Forecasters
    "CyanotoxinForecaster",
    "VibrioRiskForecaster",
    "NaegleriaForecaster",
    "SchistosomiasisForecaster",
    # Integrated
    "IntegratedDiseaseRisk",
    # Outputs
    "CyanotoxinOutput",
    "VibrioOutput",
    "NaegleriaOutput",
    "SchistosomiasisOutput",
    "DiseaseRiskSummary",
    "AlertLevel",
    # Utilities
    "ConformalCalibrator",
    "encode_temporal_context",
    # ARG surveillance (Phase 3.4)
    "ARGPredictor",
    "ARGPredictionOutput",
    "ARGSurveillanceLoss",
    "ARGSurveillancePipeline",
    "ARGBurdenIndex",
    "BurdenAlertLevel",
    "CommunityToARGMapper",
    "TemporalARGAlert",
    "TemporalARGTracker",
    "WHOPriority",
    # Occupancy + eDNA (Phase 3.5)
    "OccupancyShiftModel",
    "OccupancyShiftOutput",
    "eDNACommunityPredictor",
    "eDNACommunityOutput",
    "TaxonomicPriorModule",
    # Field dose-response (Phase 3.6)
    "FieldDoseResponseModel",
    "FieldDoseResponseOutput",
    "DoseResponseCurveParams",
    "SSDOutput",
    "SpeciesSensitivityDistribution",
    # Inverse AOP (Phase 3.7)
    "InverseAOPPredictor",
    "AOPPredictionOutput",
    "AOPCategory",
    "AOPAlertLevel",
    # Metatranscriptomic surveillance (Phase 3.8)
    "MetatranscriptomicSurveillance",
    "NovelPathogenDetector",
    "EnvironmentalSurveillancePipeline",
    "PathogenSignatureDB",
    "EnvironmentalRNAEncoder",
    "PathogenDetectionOutput",
    "SurveillanceOutput",
    "PathogenAlertLevel",
    # Bioremediation (Phase 4)
    "AMENDMENT_CATALOG",
    "BiologyBioremediationRecommender",
    "CONTAMINANT_CLASSES",
    "DEGRADER_DATABASE",
    "RemediationRecommendation",
]
