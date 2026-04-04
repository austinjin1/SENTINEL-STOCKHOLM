"""
SENTINEL Research API.

FastAPI application providing programmatic access to the SENTINEL water
quality monitoring platform.  Endpoints cover real-time assessment,
photo-based analysis, test kit submission, anomaly alerts, time series,
model inference, and case studies.
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the SENTINEL FastAPI application."""
    app = FastAPI(
        title="SENTINEL Water Quality API",
        description=(
            "Programmatic access to the SENTINEL multimodal water quality "
            "monitoring platform.  Fuses satellite, sensor, and citizen "
            "science data for real-time assessment."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # CORS -- allow dashboard and third-party clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Authentication placeholder
# ---------------------------------------------------------------------------

_VALID_API_KEYS: set[str] = set()  # populated from config / DB at startup


async def _verify_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> str:
    """Validate the API key from the request header.

    When ``_VALID_API_KEYS`` is empty (development mode), all requests
    are allowed.
    """
    if _VALID_API_KEYS and x_api_key not in _VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ---------------------------------------------------------------------------
# Rate limiting placeholder
# ---------------------------------------------------------------------------

_RATE_LIMIT_WINDOW_S: int = 60
_RATE_LIMIT_MAX_REQUESTS: int = 120
_request_counts: dict[str, list[float]] = {}


def _check_rate_limit(api_key: str) -> None:
    """Simple in-memory sliding-window rate limiter.

    For production, replace with Redis-backed or middleware-based limiter.
    """
    now = time.time()
    window = _request_counts.setdefault(api_key, [])
    # Prune old entries
    cutoff = now - _RATE_LIMIT_WINDOW_S
    _request_counts[api_key] = [t for t in window if t > cutoff]
    window = _request_counts[api_key]

    if len(window) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {_RATE_LIMIT_MAX_REQUESTS} requests "
                f"per {_RATE_LIMIT_WINDOW_S}s"
            ),
        )
    window.append(now)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QualityTierEnum(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class ParameterReading(BaseModel):
    """A single parameter reading within an assessment."""

    parameter: str
    value: float
    unit: str
    quality_tier: QualityTierEnum = QualityTierEnum.Q3
    source: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AssessmentResponse(BaseModel):
    """Real-time water quality assessment for a location."""

    latitude: float
    longitude: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parameters: list[ParameterReading] = Field(default_factory=list)
    overall_quality_index: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Composite water quality index (0=worst, 100=best).",
    )
    data_sources: list[str] = Field(default_factory=list)


class PhotoUploadResponse(BaseModel):
    """Response from photo analysis endpoint."""

    turbidity_class: str
    algal_coverage: float
    color_anomaly: bool
    oil_sheen: float
    foam_presence: float
    water_color_index: int
    confidence: float
    model_version: str


class TestKitReadingRequest(BaseModel):
    """A single test kit reading in a submission."""

    parameter: str
    value: float
    unit: str = ""
    kit_brand: str = ""
    test_method: str = ""


class TestKitSubmissionRequest(BaseModel):
    """Batch test kit submission request."""

    readings: list[TestKitReadingRequest]
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    contributor_id: str


class TestKitSubmissionResponse(BaseModel):
    """Response from test kit submission."""

    accepted_count: int
    rejected_count: int
    records: list[ParameterReading] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


class SiteInfo(BaseModel):
    """Summary information about a monitoring site."""

    site_id: str
    name: str = ""
    latitude: float
    longitude: float
    parameters: list[str] = Field(default_factory=list)
    quality_tier: QualityTierEnum = QualityTierEnum.Q3
    last_observation: Optional[datetime] = None


class AlertInfo(BaseModel):
    """An active anomaly alert."""

    alert_id: str
    site_id: str
    parameter: str
    severity: str = "warning"
    message: str
    triggered_at: datetime
    latitude: float
    longitude: float


class TimeseriesPoint(BaseModel):
    """A single point in a time series."""

    timestamp: datetime
    value: float
    quality_tier: QualityTierEnum = QualityTierEnum.Q3


class TimeseriesResponse(BaseModel):
    """Time series data for a site and parameter."""

    site_id: str
    parameter: str
    unit: str
    points: list[TimeseriesPoint] = Field(default_factory=list)


class PredictRequest(BaseModel):
    """Model inference request with raw modality data."""

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    timestamp: Optional[datetime] = None
    satellite_bands: Optional[dict[str, float]] = None
    sensor_readings: Optional[dict[str, float]] = None
    citizen_readings: Optional[dict[str, float]] = None
    microbial_clr: Optional[list[float]] = None


class PredictResponse(BaseModel):
    """Model inference response."""

    predictions: dict[str, float] = Field(default_factory=dict)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    modalities_used: list[str] = Field(default_factory=list)


class CaseStudySummary(BaseModel):
    """Summary of a case study event."""

    event_id: str
    title: str
    description: str = ""
    location: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)


class CaseStudyDetail(CaseStudySummary):
    """Full case study data package."""

    parameters: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    site_ids: list[str] = Field(default_factory=list)
    summary_statistics: dict[str, Any] = Field(default_factory=dict)
    download_url: str = ""


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    """Register all API endpoints on *app*."""

    # -- Health check -------------------------------------------------------

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """API health check."""
        return HealthResponse()

    # -- Assessment ---------------------------------------------------------

    @app.get(
        "/api/v1/assessment/{lat}/{lon}",
        response_model=AssessmentResponse,
        tags=["assessment"],
    )
    async def get_assessment(
        lat: float,
        lon: float,
        api_key: str = Depends(_verify_api_key),
    ) -> AssessmentResponse:
        """Real-time water quality assessment for a location.

        Fuses nearest satellite, sensor, and community data sources.
        """
        _check_rate_limit(api_key)

        if not (-90.0 <= lat <= 90.0):
            raise HTTPException(status_code=400, detail="Latitude must be in [-90, 90]")
        if not (-180.0 <= lon <= 180.0):
            raise HTTPException(status_code=400, detail="Longitude must be in [-180, 180]")

        # Placeholder: in production, query SENTINEL-DB and run fusion
        return AssessmentResponse(
            latitude=lat,
            longitude=lon,
            parameters=[],
            overall_quality_index=0.0,
            data_sources=[],
        )

    # -- Photo upload -------------------------------------------------------

    @app.post(
        "/api/v1/photo",
        response_model=PhotoUploadResponse,
        tags=["citizen"],
    )
    async def upload_photo(
        file: UploadFile,
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        api_key: str = Depends(_verify_api_key),
    ) -> PhotoUploadResponse:
        """Upload a photo of a water body for visual analysis."""
        _check_rate_limit(api_key)

        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Uploaded file must be an image"
            )

        # Read image bytes and run analysis
        try:
            from PIL import Image as PILImage
            import io

            contents = await file.read()
            image = PILImage.open(io.BytesIO(contents)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not decode image: {exc}"
            )

        from sentinel.platform.photo_analysis import PhotoWaterAnalyzer

        analyzer = PhotoWaterAnalyzer()
        result = analyzer.analyze_photo(image, lat, lon)

        return PhotoUploadResponse(
            turbidity_class=result.turbidity_class.value,
            algal_coverage=result.algal_coverage,
            color_anomaly=result.color_anomaly,
            oil_sheen=result.oil_sheen,
            foam_presence=result.foam_presence,
            water_color_index=result.water_color_index,
            confidence=result.turbidity_confidence,
            model_version=result.model_version,
        )

    # -- Test kit submission ------------------------------------------------

    @app.post(
        "/api/v1/testkit",
        response_model=TestKitSubmissionResponse,
        tags=["citizen"],
    )
    async def submit_test_kit(
        body: TestKitSubmissionRequest,
        api_key: str = Depends(_verify_api_key),
    ) -> TestKitSubmissionResponse:
        """Submit home water test kit readings."""
        _check_rate_limit(api_key)

        from sentinel.platform.test_kit import (
            TestKitReading,
            TestKitSubmission,
        )

        readings = [
            TestKitReading(
                parameter=r.parameter,
                value=r.value,
                unit=r.unit,
                kit_brand=r.kit_brand,
                test_method=r.test_method,
            )
            for r in body.readings
        ]

        submitter = TestKitSubmission()
        wqrs = submitter.submit(
            readings,
            body.latitude,
            body.longitude,
            body.contributor_id,
        )

        records = [
            ParameterReading(
                parameter=r.canonical_param,
                value=r.value,
                unit=r.unit,
                quality_tier=QualityTierEnum(r.quality_tier.value),
                source=r.source,
            )
            for r in wqrs
        ]

        return TestKitSubmissionResponse(
            accepted_count=len(wqrs),
            rejected_count=len(body.readings) - len(wqrs),
            records=records,
        )

    # -- Site endpoints -----------------------------------------------------

    @app.get(
        "/api/v1/site/{site_id}",
        response_model=SiteInfo,
        tags=["sites"],
    )
    async def get_site(
        site_id: str,
        api_key: str = Depends(_verify_api_key),
    ) -> SiteInfo:
        """Get information for a specific monitoring site."""
        _check_rate_limit(api_key)

        # Placeholder: query SENTINEL-DB
        raise HTTPException(status_code=404, detail=f"Site {site_id!r} not found")

    @app.get(
        "/api/v1/sites",
        response_model=list[SiteInfo],
        tags=["sites"],
    )
    async def list_sites(
        min_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
        max_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
        min_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
        max_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
        parameter: Optional[str] = Query(None, description="Filter by parameter"),
        quality_tier: Optional[QualityTierEnum] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        api_key: str = Depends(_verify_api_key),
    ) -> list[SiteInfo]:
        """List monitoring sites with optional filters."""
        _check_rate_limit(api_key)

        # Placeholder: query SENTINEL-DB with filters
        return []

    # -- Alerts -------------------------------------------------------------

    @app.get(
        "/api/v1/alerts",
        response_model=list[AlertInfo],
        tags=["alerts"],
    )
    async def get_alerts(
        min_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
        max_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
        min_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
        max_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
        severity: Optional[str] = Query(None),
        api_key: str = Depends(_verify_api_key),
    ) -> list[AlertInfo]:
        """Get active anomaly alerts, optionally filtered by bounding box."""
        _check_rate_limit(api_key)

        # Placeholder: query alert store
        return []

    # -- Timeseries ---------------------------------------------------------

    @app.get(
        "/api/v1/timeseries/{site_id}",
        response_model=TimeseriesResponse,
        tags=["timeseries"],
    )
    async def get_timeseries(
        site_id: str,
        parameter: str = Query(..., description="Parameter to retrieve"),
        start: Optional[datetime] = Query(None),
        end: Optional[datetime] = Query(None),
        api_key: str = Depends(_verify_api_key),
    ) -> TimeseriesResponse:
        """Get sensor time series for a site and parameter."""
        _check_rate_limit(api_key)

        # Placeholder: query SENTINEL-DB
        return TimeseriesResponse(
            site_id=site_id,
            parameter=parameter,
            unit="",
            points=[],
        )

    # -- Model inference ----------------------------------------------------

    @app.post(
        "/api/v1/predict",
        response_model=PredictResponse,
        tags=["model"],
    )
    async def predict(
        body: PredictRequest,
        api_key: str = Depends(_verify_api_key),
    ) -> PredictResponse:
        """Run multimodal fusion model inference.

        Submit raw modality data and receive fused predictions.
        """
        _check_rate_limit(api_key)

        modalities_used: list[str] = []
        if body.satellite_bands:
            modalities_used.append("satellite")
        if body.sensor_readings:
            modalities_used.append("sensor")
        if body.citizen_readings:
            modalities_used.append("citizen")
        if body.microbial_clr:
            modalities_used.append("microbial")

        if not modalities_used:
            raise HTTPException(
                status_code=400,
                detail="At least one modality must be provided",
            )

        # Placeholder: run SENTINEL fusion model
        return PredictResponse(
            predictions={},
            anomaly_score=0.0,
            confidence=0.0,
            modalities_used=modalities_used,
        )

    # -- Case studies -------------------------------------------------------

    @app.get(
        "/api/v1/casestudies",
        response_model=list[CaseStudySummary],
        tags=["casestudies"],
    )
    async def list_case_studies(
        tag: Optional[str] = Query(None, description="Filter by tag"),
        limit: int = Query(50, ge=1, le=200),
        api_key: str = Depends(_verify_api_key),
    ) -> list[CaseStudySummary]:
        """List available case study events."""
        _check_rate_limit(api_key)

        # Placeholder: query case study store
        return []

    @app.get(
        "/api/v1/casestudies/{event_id}",
        response_model=CaseStudyDetail,
        tags=["casestudies"],
    )
    async def get_case_study(
        event_id: str,
        api_key: str = Depends(_verify_api_key),
    ) -> CaseStudyDetail:
        """Get full data package for a case study event."""
        _check_rate_limit(api_key)

        # Placeholder: query case study store
        raise HTTPException(
            status_code=404, detail=f"Case study {event_id!r} not found"
        )


# ---------------------------------------------------------------------------
# Module-level app instance (for ``uvicorn sentinel.platform.api:app``)
# ---------------------------------------------------------------------------

app = create_app()
