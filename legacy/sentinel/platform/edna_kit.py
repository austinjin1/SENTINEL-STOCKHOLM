"""
SENTINEL citizen science eDNA kit results module.

Handles ingestion, validation, and integration of citizen-contributed
environmental DNA (eDNA) kit results into the SENTINEL platform.
Supports multiple commercial eDNA kit formats (Jonah Ventures,
ZymoBIOMICS, generic OTU tables) and performs a three-stage validation
pipeline before embedding results into the SENTINEL shared latent space.

Three-Stage Validation Pipeline
-------------------------------
1. **Format validation** -- required fields, GPS bounds, date sanity.
2. **Biological plausibility** -- species range checks, contamination
   marker detection (human DNA, E. coli controls).
3. **Community-level QC** -- diversity index checks, comparison against
   EPA NARS reference communities, Shannon diversity bounds.

This is Phase 5 of the SENTINEL platform expansion.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6_371.0

# GPS bounds for continental US + common sampling regions
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

# Common lab contaminants to flag in eDNA results
LAB_CONTAMINANTS = {
    "homo sapiens": "human DNA contamination",
    "homo sapiens sapiens": "human DNA contamination",
    "escherichia coli": "possible E. coli control spike-in",
    "e. coli": "possible E. coli control spike-in",
    "lambda phage": "sequencing control",
    "phix174": "Illumina sequencing control",
    "saccharomyces cerevisiae": "possible yeast contamination",
}

# Shannon diversity bounds for freshwater eDNA communities
SHANNON_MIN_REASONABLE = 0.5
SHANNON_MAX_REASONABLE = 6.0

# USGS NAS invasive species commonly detected via eDNA in freshwater
INVASIVE_SPECIES_LIST = {
    "dreissena polymorpha": "Zebra mussel",
    "dreissena rostriformis bugensis": "Quagga mussel",
    "neogobius melanostomus": "Round goby",
    "cyprinus carpio": "Common carp",
    "hypophthalmichthys molitrix": "Silver carp",
    "hypophthalmichthys nobilis": "Bighead carp",
    "mylopharyngodon piceus": "Black carp",
    "ctenopharyngodon idella": "Grass carp",
    "petromyzon marinus": "Sea lamprey",
    "orconectes rusticus": "Rusty crayfish",
    "bythotrephes longimanus": "Spiny water flea",
    "cercopagis pengoi": "Fishhook water flea",
    "didymosphenia geminata": "Didymo / rock snot",
    "myriophyllum spicatum": "Eurasian watermilfoil",
    "potamopyrgus antipodarum": "New Zealand mudsnail",
    "pylodictis olivaris": "Flathead catfish (invasive range)",
    "channa argus": "Northern snakehead",
}

# EPA NARS ecoregion reference community diversity ranges
# Format: ecoregion -> (shannon_min, shannon_max, expected_richness_min, expected_richness_max)
ECOREGION_REFERENCES: Dict[str, Tuple[float, float, int, int]] = {
    "northern_appalachians": (1.8, 4.5, 20, 150),
    "southern_appalachians": (2.0, 4.8, 25, 180),
    "coastal_plains": (1.5, 4.2, 15, 120),
    "upper_midwest": (1.8, 4.5, 20, 160),
    "temperate_plains": (1.5, 4.0, 15, 130),
    "southern_plains": (1.2, 3.8, 10, 100),
    "northern_plains": (1.5, 4.0, 15, 120),
    "western_mountains": (1.8, 4.5, 18, 140),
    "xeric_west": (1.0, 3.5, 10, 80),
    "default": (1.0, 5.0, 10, 200),
}

# Shared embedding dimension matching SENTINEL fusion layer
SENTINEL_EMBEDDING_DIM = 256


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SpeciesDetection:
    """A single species detection from eDNA metabarcoding."""

    taxon_name: str
    read_count: int
    confidence: float = 0.0
    ncbi_taxon_id: Optional[int] = None
    taxonomic_rank: str = "species"
    sequence_id: str = ""


@dataclass
class eDNAKitResult:
    """Complete result from a citizen science eDNA sampling kit.

    Attributes
    ----------
    kit_id:
        Unique identifier for this kit / sample submission.
    collector_name:
        Name of the citizen scientist who collected the sample.
    collection_date:
        Date and time when the water sample was collected.
    latitude, longitude:
        GPS coordinates of the sampling location.
    water_body_name:
        Name of the sampled water body (lake, river, stream, etc.).
    sample_volume_ml:
        Volume of water filtered, in millilitres.
    filter_type:
        Filter membrane type used for eDNA capture.
    species_detections:
        List of detected taxa with read counts and confidence scores.
    raw_otu_table:
        Optional raw OTU/ASV abundance table as a numpy array,
        shape ``(n_samples, n_taxa)``.
    metadata:
        Arbitrary additional metadata from the kit provider.
    """

    kit_id: str
    collector_name: str
    collection_date: datetime
    latitude: float
    longitude: float
    water_body_name: str
    sample_volume_ml: float = 1000.0
    filter_type: str = "0.45um_MCE"
    species_detections: List[SpeciesDetection] = field(default_factory=list)
    raw_otu_table: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation flags
# ---------------------------------------------------------------------------


@dataclass
class ValidationFlag:
    """A single validation warning or error."""

    stage: int
    severity: str  # "error", "warning", "info"
    message: str
    field_name: str = ""


@dataclass
class eDNAValidationReport:
    """Full validation report for an eDNA kit result."""

    kit_id: str
    is_valid: bool = True
    flags: List[ValidationFlag] = field(default_factory=list)
    stage_passed: Dict[int, bool] = field(
        default_factory=lambda: {1: False, 2: False, 3: False}
    )
    contamination_warnings: List[str] = field(default_factory=list)
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    quality_tier: str = "Q3"

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "warning")


# ---------------------------------------------------------------------------
# Stage 1-3 Validator
# ---------------------------------------------------------------------------


class eDNAValidator:
    """Three-stage validation pipeline for citizen eDNA kit results.

    Follows the same pattern as :class:`CitizenQCPipeline` in
    ``citizen_qc.py`` but tailored for metabarcoding / eDNA data.

    Stages
    ------
    1. Format validation -- required fields, GPS bounds, date sanity.
    2. Biological plausibility -- species range checks against known
       distributions, contamination marker detection for common lab
       contaminants (human DNA, E. coli controls).
    3. Community-level QC -- diversity index checks, comparison against
       reference communities for the ecoregion, flag if Shannon
       diversity is unreasonably low or high.
    """

    def __init__(
        self,
        *,
        known_distributions: Optional[Dict[str, List[str]]] = None,
        ecoregion: str = "default",
    ) -> None:
        self.known_distributions = known_distributions or {}
        self.ecoregion = ecoregion

    def validate(self, result: eDNAKitResult) -> eDNAValidationReport:
        """Run full three-stage validation on an eDNA kit result.

        Parameters
        ----------
        result:
            The eDNA kit result to validate.

        Returns
        -------
        :class:`eDNAValidationReport` with per-stage results and flags.
        """
        report = eDNAValidationReport(kit_id=result.kit_id)

        # Stage 1: Format validation
        self._validate_format(result, report)
        if report.error_count > 0:
            report.is_valid = False
            logger.info(
                f"eDNA kit {result.kit_id} failed Stage 1 format validation "
                f"with {report.error_count} errors"
            )
            return report
        report.stage_passed[1] = True

        # Stage 2: Biological plausibility
        self._validate_biology(result, report)
        if report.error_count > 0:
            report.is_valid = False
            logger.info(
                f"eDNA kit {result.kit_id} failed Stage 2 biological "
                f"plausibility with {report.error_count} errors"
            )
            return report
        report.stage_passed[2] = True

        # Stage 3: Community-level QC
        self._validate_community(result, report)
        report.stage_passed[3] = report.error_count == 0
        report.is_valid = report.error_count == 0

        # Assign quality tier
        if report.error_count == 0 and report.warning_count == 0:
            report.quality_tier = "Q1"
        elif report.error_count == 0:
            report.quality_tier = "Q2"
        else:
            report.quality_tier = "Q3"

        logger.info(
            f"eDNA kit {result.kit_id} validation complete: "
            f"valid={report.is_valid}, tier={report.quality_tier}, "
            f"errors={report.error_count}, warnings={report.warning_count}"
        )
        return report

    # -- Stage 1: Format validation -----------------------------------------

    def _validate_format(
        self, result: eDNAKitResult, report: eDNAValidationReport
    ) -> None:
        """Check required fields, GPS bounds, and date sanity."""
        # Required string fields
        for field_name in ("kit_id", "collector_name", "water_body_name"):
            value = getattr(result, field_name, "")
            if not value or not value.strip():
                report.flags.append(
                    ValidationFlag(
                        stage=1,
                        severity="error",
                        message=f"Required field '{field_name}' is empty.",
                        field_name=field_name,
                    )
                )

        # GPS bounds
        if not (LAT_MIN <= result.latitude <= LAT_MAX):
            report.flags.append(
                ValidationFlag(
                    stage=1,
                    severity="error",
                    message=(
                        f"Latitude {result.latitude} out of range "
                        f"[{LAT_MIN}, {LAT_MAX}]."
                    ),
                    field_name="latitude",
                )
            )

        if not (LON_MIN <= result.longitude <= LON_MAX):
            report.flags.append(
                ValidationFlag(
                    stage=1,
                    severity="error",
                    message=(
                        f"Longitude {result.longitude} out of range "
                        f"[{LON_MIN}, {LON_MAX}]."
                    ),
                    field_name="longitude",
                )
            )

        # Date sanity: not in the future, not before eDNA methods existed
        now = datetime.utcnow()
        if result.collection_date > now:
            report.flags.append(
                ValidationFlag(
                    stage=1,
                    severity="error",
                    message="Collection date is in the future.",
                    field_name="collection_date",
                )
            )

        earliest = datetime(2010, 1, 1)
        if result.collection_date < earliest:
            report.flags.append(
                ValidationFlag(
                    stage=1,
                    severity="warning",
                    message=(
                        f"Collection date {result.collection_date.isoformat()} "
                        "is before 2010, when citizen eDNA kits became available."
                    ),
                    field_name="collection_date",
                )
            )

        # Sample volume
        if result.sample_volume_ml <= 0:
            report.flags.append(
                ValidationFlag(
                    stage=1,
                    severity="error",
                    message="Sample volume must be positive.",
                    field_name="sample_volume_ml",
                )
            )

        # Must have at least one detection
        if not result.species_detections:
            report.flags.append(
                ValidationFlag(
                    stage=1,
                    severity="error",
                    message="No species detections found in kit result.",
                    field_name="species_detections",
                )
            )

    # -- Stage 2: Biological plausibility -----------------------------------

    def _validate_biology(
        self, result: eDNAKitResult, report: eDNAValidationReport
    ) -> None:
        """Check species ranges and detect lab contamination markers."""
        contaminant_reads = 0
        total_reads = sum(d.read_count for d in result.species_detections)

        for detection in result.species_detections:
            taxon_lower = detection.taxon_name.strip().lower()

            # Check for lab contaminants
            if taxon_lower in LAB_CONTAMINANTS:
                reason = LAB_CONTAMINANTS[taxon_lower]
                proportion = (
                    detection.read_count / total_reads
                    if total_reads > 0
                    else 0.0
                )
                contaminant_reads += detection.read_count

                if proportion > 0.10:
                    report.flags.append(
                        ValidationFlag(
                            stage=2,
                            severity="warning",
                            message=(
                                f"High contamination: {detection.taxon_name} "
                                f"({reason}) comprises {proportion:.1%} of "
                                f"total reads."
                            ),
                            field_name="species_detections",
                        )
                    )
                    report.contamination_warnings.append(
                        f"{detection.taxon_name}: {reason} "
                        f"({proportion:.1%} of reads)"
                    )
                else:
                    report.flags.append(
                        ValidationFlag(
                            stage=2,
                            severity="info",
                            message=(
                                f"Low-level contaminant detected: "
                                f"{detection.taxon_name} ({reason}), "
                                f"{proportion:.1%} of reads."
                            ),
                            field_name="species_detections",
                        )
                    )

            # Check species range if distributions are available
            if (
                self.known_distributions
                and taxon_lower in self.known_distributions
            ):
                known_regions = self.known_distributions[taxon_lower]
                # Simple region check -- in production this would be a
                # spatial lookup against known occurrence polygons
                if known_regions and self.ecoregion not in known_regions:
                    report.flags.append(
                        ValidationFlag(
                            stage=2,
                            severity="warning",
                            message=(
                                f"Species '{detection.taxon_name}' not "
                                f"expected in ecoregion '{self.ecoregion}'. "
                                f"Known in: {', '.join(known_regions)}."
                            ),
                            field_name="species_detections",
                        )
                    )

            # Flag negative or zero read counts
            if detection.read_count < 0:
                report.flags.append(
                    ValidationFlag(
                        stage=2,
                        severity="error",
                        message=(
                            f"Negative read count ({detection.read_count}) "
                            f"for {detection.taxon_name}."
                        ),
                        field_name="species_detections",
                    )
                )

        # Flag if contaminant reads dominate the sample
        if total_reads > 0 and contaminant_reads / total_reads > 0.50:
            report.flags.append(
                ValidationFlag(
                    stage=2,
                    severity="error",
                    message=(
                        f"Contaminant reads ({contaminant_reads}) exceed 50% "
                        f"of total reads ({total_reads}). Sample likely "
                        "compromised."
                    ),
                    field_name="species_detections",
                )
            )

    # -- Stage 3: Community-level QC ----------------------------------------

    def _validate_community(
        self, result: eDNAKitResult, report: eDNAValidationReport
    ) -> None:
        """Check diversity indices and compare against reference communities."""
        if not result.species_detections:
            return

        # Build abundance array from detections
        abundances = np.array(
            [d.read_count for d in result.species_detections], dtype=np.float64
        )
        abundances = abundances[abundances > 0]

        if len(abundances) == 0:
            report.flags.append(
                ValidationFlag(
                    stage=3,
                    severity="error",
                    message="No positive read counts in species detections.",
                    field_name="species_detections",
                )
            )
            return

        # Compute diversity indices
        analyzer = eDNACommunityAnalyzer()
        total = abundances.sum()
        proportions = abundances / total

        shannon = -np.sum(proportions * np.log(proportions))
        simpson = 1.0 - np.sum(proportions ** 2)
        richness = len(abundances)

        report.diversity_metrics = {
            "shannon": float(shannon),
            "simpson": float(simpson),
            "richness": int(richness),
        }

        # Check Shannon diversity bounds
        if shannon < SHANNON_MIN_REASONABLE:
            report.flags.append(
                ValidationFlag(
                    stage=3,
                    severity="warning",
                    message=(
                        f"Shannon diversity ({shannon:.2f}) is unreasonably "
                        f"low (< {SHANNON_MIN_REASONABLE}). Sample may be "
                        "dominated by a single taxon or have low quality."
                    ),
                    field_name="species_detections",
                )
            )

        if shannon > SHANNON_MAX_REASONABLE:
            report.flags.append(
                ValidationFlag(
                    stage=3,
                    severity="warning",
                    message=(
                        f"Shannon diversity ({shannon:.2f}) is unreasonably "
                        f"high (> {SHANNON_MAX_REASONABLE}). Possible "
                        "sequencing artefacts or over-splitting of OTUs."
                    ),
                    field_name="species_detections",
                )
            )

        # Compare against ecoregion reference
        ref = ECOREGION_REFERENCES.get(
            self.ecoregion, ECOREGION_REFERENCES["default"]
        )
        ref_shannon_min, ref_shannon_max, ref_rich_min, ref_rich_max = ref

        if shannon < ref_shannon_min or shannon > ref_shannon_max:
            report.flags.append(
                ValidationFlag(
                    stage=3,
                    severity="warning",
                    message=(
                        f"Shannon diversity ({shannon:.2f}) outside expected "
                        f"range [{ref_shannon_min}, {ref_shannon_max}] for "
                        f"ecoregion '{self.ecoregion}'."
                    ),
                    field_name="species_detections",
                )
            )

        if richness < ref_rich_min:
            report.flags.append(
                ValidationFlag(
                    stage=3,
                    severity="warning",
                    message=(
                        f"Species richness ({richness}) below expected "
                        f"minimum ({ref_rich_min}) for ecoregion "
                        f"'{self.ecoregion}'."
                    ),
                    field_name="species_detections",
                )
            )
        elif richness > ref_rich_max:
            report.flags.append(
                ValidationFlag(
                    stage=3,
                    severity="info",
                    message=(
                        f"Species richness ({richness}) above expected "
                        f"maximum ({ref_rich_max}) for ecoregion "
                        f"'{self.ecoregion}'. May indicate a highly diverse "
                        "site or OTU over-splitting."
                    ),
                    field_name="species_detections",
                )
            )


# ---------------------------------------------------------------------------
# Ingestion parsers
# ---------------------------------------------------------------------------


class eDNAIngestion:
    """Parse eDNA kit results from multiple commercial formats.

    Supports:
    - Jonah Ventures CSV format
    - ZymoBIOMICS format
    - Generic OTU table CSV
    """

    @staticmethod
    def ingest_jonah_ventures(filepath: str | Path) -> eDNAKitResult:
        """Parse a Jonah Ventures eDNA results CSV.

        Jonah Ventures format has columns:
        ``Sample_ID, Taxon, Reads, Confidence, Rank, ...``
        with metadata in a header block.

        Parameters
        ----------
        filepath:
            Path to the Jonah Ventures CSV file.

        Returns
        -------
        :class:`eDNAKitResult` with parsed detections.
        """
        filepath = Path(filepath)
        metadata: Dict[str, Any] = {"provider": "jonah_ventures"}
        detections: List[SpeciesDetection] = []

        kit_id = ""
        collector_name = ""
        collection_date = datetime.utcnow()
        latitude = 0.0
        longitude = 0.0
        water_body = ""

        with open(filepath, newline="", encoding="utf-8") as fh:
            # Try to parse metadata header lines (lines starting with #)
            lines = fh.readlines()

        header_end = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                if ":" in stripped:
                    key, _, val = stripped[1:].partition(":")
                    key = key.strip().lower()
                    val = val.strip()
                    metadata[key] = val
                    if key in ("sample_id", "kit_id"):
                        kit_id = val
                    elif key in ("collector", "collector_name"):
                        collector_name = val
                    elif key in ("date", "collection_date"):
                        try:
                            collection_date = datetime.fromisoformat(val)
                        except ValueError:
                            pass
                    elif key in ("latitude", "lat"):
                        latitude = float(val)
                    elif key in ("longitude", "lon", "lng"):
                        longitude = float(val)
                    elif key in ("water_body", "site_name"):
                        water_body = val
                header_end = i + 1
            else:
                break

        # Parse CSV body
        csv_text = "".join(lines[header_end:])
        import io

        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            # Normalise column names
            norm_row = {k.strip().lower(): v.strip() for k, v in row.items()}

            taxon = norm_row.get("taxon", norm_row.get("species", ""))
            reads_str = norm_row.get("reads", norm_row.get("read_count", "0"))
            conf_str = norm_row.get("confidence", norm_row.get("conf", "0"))
            rank = norm_row.get("rank", norm_row.get("taxonomic_rank", "species"))

            if not taxon:
                continue

            try:
                reads = int(float(reads_str))
                conf = float(conf_str)
            except (ValueError, TypeError):
                reads = 0
                conf = 0.0

            detections.append(
                SpeciesDetection(
                    taxon_name=taxon,
                    read_count=reads,
                    confidence=conf,
                    taxonomic_rank=rank,
                )
            )

        if not kit_id:
            kit_id = filepath.stem

        return eDNAKitResult(
            kit_id=kit_id,
            collector_name=collector_name,
            collection_date=collection_date,
            latitude=latitude,
            longitude=longitude,
            water_body_name=water_body,
            species_detections=detections,
            metadata=metadata,
        )

    @staticmethod
    def ingest_zymobiomics(filepath: str | Path) -> eDNAKitResult:
        """Parse a ZymoBIOMICS eDNA results file.

        ZymoBIOMICS format uses tab-separated values with columns:
        ``#OTU ID, taxonomy, <sample_columns>``

        Parameters
        ----------
        filepath:
            Path to the ZymoBIOMICS TSV file.

        Returns
        -------
        :class:`eDNAKitResult` with parsed detections.
        """
        filepath = Path(filepath)
        metadata: Dict[str, Any] = {"provider": "zymobiomics"}
        detections: List[SpeciesDetection] = []
        otu_rows: List[List[float]] = []

        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = None
            for row in reader:
                if not row or row[0].startswith("##"):
                    continue
                if header is None:
                    header = [h.strip().lower() for h in row]
                    continue

                if len(row) < 2:
                    continue

                otu_id = row[0].strip()
                # Last column is typically taxonomy
                taxonomy = row[-1].strip() if len(row) > 2 else row[1].strip()
                # Extract species from taxonomy string
                # Typical format: k__Bacteria;p__...;g__Genus;s__Species
                species_name = taxonomy
                for part in taxonomy.split(";"):
                    part = part.strip()
                    if part.startswith("s__"):
                        species_name = part[3:].replace("_", " ")
                        break

                # Sum read counts from all sample columns
                sample_counts = []
                for val in row[1:-1]:
                    try:
                        sample_counts.append(float(val.strip()))
                    except (ValueError, TypeError):
                        pass

                total_reads = int(sum(sample_counts)) if sample_counts else 0
                if sample_counts:
                    otu_rows.append(sample_counts)

                detections.append(
                    SpeciesDetection(
                        taxon_name=species_name,
                        read_count=total_reads,
                        confidence=0.0,
                        sequence_id=otu_id,
                    )
                )

        raw_otu = np.array(otu_rows, dtype=np.float64) if otu_rows else None

        return eDNAKitResult(
            kit_id=filepath.stem,
            collector_name="",
            collection_date=datetime.utcnow(),
            latitude=0.0,
            longitude=0.0,
            water_body_name="",
            species_detections=detections,
            raw_otu_table=raw_otu,
            metadata=metadata,
        )

    @staticmethod
    def ingest_generic_otu(filepath: str | Path) -> eDNAKitResult:
        """Parse a generic OTU abundance table from CSV.

        Expected format: first column is taxon name, remaining columns
        are sample read counts.

        Parameters
        ----------
        filepath:
            Path to the generic OTU CSV file.

        Returns
        -------
        :class:`eDNAKitResult` with parsed detections.
        """
        filepath = Path(filepath)
        detections: List[SpeciesDetection] = []
        otu_rows: List[List[float]] = []

        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return eDNAKitResult(
                    kit_id=filepath.stem,
                    collector_name="",
                    collection_date=datetime.utcnow(),
                    latitude=0.0,
                    longitude=0.0,
                    water_body_name="",
                    metadata={"provider": "generic_otu"},
                )

            for row in reader:
                if not row:
                    continue
                taxon = row[0].strip()
                counts = []
                for val in row[1:]:
                    try:
                        counts.append(float(val.strip()))
                    except (ValueError, TypeError):
                        counts.append(0.0)

                total_reads = int(sum(counts))
                otu_rows.append(counts)

                detections.append(
                    SpeciesDetection(
                        taxon_name=taxon,
                        read_count=total_reads,
                        confidence=0.0,
                    )
                )

        raw_otu = np.array(otu_rows, dtype=np.float64) if otu_rows else None

        return eDNAKitResult(
            kit_id=filepath.stem,
            collector_name="",
            collection_date=datetime.utcnow(),
            latitude=0.0,
            longitude=0.0,
            water_body_name="",
            species_detections=detections,
            raw_otu_table=raw_otu,
            metadata={"provider": "generic_otu"},
        )

    @staticmethod
    def normalize_taxonomy(
        detections: List[SpeciesDetection],
    ) -> List[SpeciesDetection]:
        """Map species detections to NCBI taxonomy IDs.

        This performs name normalisation (case folding, synonym resolution)
        and attempts to resolve NCBI taxonomy IDs.  In production this
        would call the NCBI Taxonomy E-utilities API; here we perform
        local name cleaning.

        Parameters
        ----------
        detections:
            List of :class:`SpeciesDetection` to normalise.

        Returns
        -------
        Updated list with cleaned ``taxon_name`` and ``ncbi_taxon_id``
        fields where resolvable.
        """
        # Common synonym mappings for freshwater taxa
        synonyms: Dict[str, str] = {
            "e. coli": "Escherichia coli",
            "d. polymorpha": "Dreissena polymorpha",
            "c. carpio": "Cyprinus carpio",
            "p. marinus": "Petromyzon marinus",
        }

        normalised: List[SpeciesDetection] = []
        for det in detections:
            name = det.taxon_name.strip()
            name_lower = name.lower()

            # Resolve synonyms
            if name_lower in synonyms:
                name = synonyms[name_lower]

            # Capitalise genus (first word), lowercase epithet
            parts = name.split()
            if len(parts) >= 2:
                name = parts[0].capitalize() + " " + " ".join(
                    p.lower() for p in parts[1:]
                )
            elif len(parts) == 1:
                name = parts[0].capitalize()

            normalised.append(
                SpeciesDetection(
                    taxon_name=name,
                    read_count=det.read_count,
                    confidence=det.confidence,
                    ncbi_taxon_id=det.ncbi_taxon_id,
                    taxonomic_rank=det.taxonomic_rank,
                    sequence_id=det.sequence_id,
                )
            )

        return normalised

    @staticmethod
    def compute_relative_abundance(
        otu_table: np.ndarray,
    ) -> np.ndarray:
        """Compute centred log-ratio (CLR) transform of an OTU table.

        The CLR transform handles compositionality in sequencing data
        by log-transforming relative abundances and centring them.

        Parameters
        ----------
        otu_table:
            Raw count table, shape ``(n_taxa, n_samples)`` or
            ``(n_taxa,)`` for a single sample.

        Returns
        -------
        CLR-transformed abundance table of the same shape.
        """
        table = otu_table.astype(np.float64)

        # Add pseudocount to avoid log(0)
        table = table + 1.0

        if table.ndim == 1:
            log_table = np.log(table)
            geometric_mean = np.mean(log_table)
            return log_table - geometric_mean
        else:
            log_table = np.log(table)
            geometric_means = np.mean(log_table, axis=0, keepdims=True)
            return log_table - geometric_means


# ---------------------------------------------------------------------------
# Community analysis
# ---------------------------------------------------------------------------


class eDNACommunityAnalyzer:
    """Analyse eDNA community composition and ecological health.

    Provides diversity indices, reference community comparisons,
    invasive species detection, and human-readable health reports.
    """

    @staticmethod
    def compute_diversity_indices(
        otu_table: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard alpha diversity indices.

        Parameters
        ----------
        otu_table:
            Abundance vector (1-D) or matrix with taxa as rows.

        Returns
        -------
        Dictionary with ``"shannon"``, ``"simpson"``, and ``"chao1"``
        indices.
        """
        if otu_table.ndim > 1:
            abundances = otu_table.sum(axis=1).astype(np.float64)
        else:
            abundances = otu_table.astype(np.float64)

        abundances = abundances[abundances > 0]

        if len(abundances) == 0:
            return {"shannon": 0.0, "simpson": 0.0, "chao1": 0.0}

        total = abundances.sum()
        proportions = abundances / total

        # Shannon diversity: H' = -sum(p_i * ln(p_i))
        shannon = float(-np.sum(proportions * np.log(proportions)))

        # Simpson diversity: 1 - sum(p_i^2)
        simpson = float(1.0 - np.sum(proportions ** 2))

        # Chao1 richness estimator
        s_obs = len(abundances)
        singletons = int(np.sum(abundances == 1))
        doubletons = int(np.sum(abundances == 2))
        if doubletons > 0:
            chao1 = float(
                s_obs + (singletons ** 2) / (2.0 * doubletons)
            )
        elif singletons > 0:
            chao1 = float(
                s_obs + singletons * (singletons - 1) / 2.0
            )
        else:
            chao1 = float(s_obs)

        return {
            "shannon": round(shannon, 4),
            "simpson": round(simpson, 4),
            "chao1": round(chao1, 2),
        }

    @staticmethod
    def compare_to_reference(
        otu_table: np.ndarray,
        ecoregion: str = "default",
    ) -> Dict[str, Any]:
        """Compare community composition against EPA NARS references.

        Parameters
        ----------
        otu_table:
            Abundance vector (1-D) or table with taxa as rows.
        ecoregion:
            EPA NARS ecoregion identifier.

        Returns
        -------
        Dictionary with comparison metrics and status flags.
        """
        if otu_table.ndim > 1:
            abundances = otu_table.sum(axis=1).astype(np.float64)
        else:
            abundances = otu_table.astype(np.float64)

        abundances = abundances[abundances > 0]

        if len(abundances) == 0:
            return {
                "status": "insufficient_data",
                "message": "No positive abundances to compare.",
            }

        total = abundances.sum()
        proportions = abundances / total
        shannon = float(-np.sum(proportions * np.log(proportions)))
        richness = len(abundances)

        ref = ECOREGION_REFERENCES.get(
            ecoregion, ECOREGION_REFERENCES["default"]
        )
        ref_sh_min, ref_sh_max, ref_r_min, ref_r_max = ref

        status = "within_reference"
        messages: List[str] = []

        if shannon < ref_sh_min:
            status = "below_reference"
            messages.append(
                f"Shannon diversity ({shannon:.2f}) below reference "
                f"minimum ({ref_sh_min}) for {ecoregion}."
            )
        elif shannon > ref_sh_max:
            status = "above_reference"
            messages.append(
                f"Shannon diversity ({shannon:.2f}) above reference "
                f"maximum ({ref_sh_max}) for {ecoregion}."
            )

        if richness < ref_r_min:
            messages.append(
                f"Richness ({richness}) below reference minimum "
                f"({ref_r_min})."
            )
        elif richness > ref_r_max:
            messages.append(
                f"Richness ({richness}) above reference maximum "
                f"({ref_r_max})."
            )

        # Deviation score: how far from the reference centre
        ref_sh_mid = (ref_sh_min + ref_sh_max) / 2.0
        ref_sh_range = (ref_sh_max - ref_sh_min) / 2.0
        deviation = (
            abs(shannon - ref_sh_mid) / ref_sh_range
            if ref_sh_range > 0
            else 0.0
        )

        return {
            "status": status,
            "ecoregion": ecoregion,
            "shannon": round(shannon, 4),
            "richness": richness,
            "reference_shannon_range": (ref_sh_min, ref_sh_max),
            "reference_richness_range": (ref_r_min, ref_r_max),
            "deviation_score": round(min(deviation, 2.0), 4),
            "messages": messages,
        }

    @staticmethod
    def detect_invasive_species(
        detections: List[SpeciesDetection],
    ) -> List[Dict[str, Any]]:
        """Check species detections against the USGS NAS invasive list.

        Parameters
        ----------
        detections:
            List of :class:`SpeciesDetection` from an eDNA kit.

        Returns
        -------
        List of dicts for each detected invasive species, including
        common name, read count, and confidence.
        """
        invasives_found: List[Dict[str, Any]] = []

        for det in detections:
            taxon_lower = det.taxon_name.strip().lower()
            if taxon_lower in INVASIVE_SPECIES_LIST:
                invasives_found.append(
                    {
                        "scientific_name": det.taxon_name,
                        "common_name": INVASIVE_SPECIES_LIST[taxon_lower],
                        "read_count": det.read_count,
                        "confidence": det.confidence,
                        "taxonomic_rank": det.taxonomic_rank,
                    }
                )

        if invasives_found:
            logger.info(
                f"Detected {len(invasives_found)} invasive species: "
                + ", ".join(
                    f"{inv['common_name']} ({inv['scientific_name']})"
                    for inv in invasives_found
                )
            )

        return invasives_found

    @staticmethod
    def generate_health_report(result: eDNAKitResult) -> str:
        """Generate a human-readable ecosystem health report.

        Produces a text summary suitable for citizen scientists,
        covering species richness, diversity, invasive species, and
        overall ecosystem health assessment.

        Parameters
        ----------
        result:
            A validated :class:`eDNAKitResult`.

        Returns
        -------
        Multi-line text report string.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  SENTINEL eDNA Ecosystem Health Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Kit ID:        {result.kit_id}")
        lines.append(f"  Collector:     {result.collector_name}")
        lines.append(f"  Date:          {result.collection_date.strftime('%Y-%m-%d')}")
        lines.append(f"  Water Body:    {result.water_body_name}")
        lines.append(
            f"  Location:      ({result.latitude:.4f}, {result.longitude:.4f})"
        )
        lines.append(f"  Sample Volume: {result.sample_volume_ml} mL")
        lines.append(f"  Filter Type:   {result.filter_type}")
        lines.append("")

        # Species summary
        n_species = len(result.species_detections)
        total_reads = sum(d.read_count for d in result.species_detections)
        lines.append(f"  Species Detected:  {n_species}")
        lines.append(f"  Total Reads:       {total_reads:,}")
        lines.append("")

        # Diversity indices
        if result.species_detections:
            abundances = np.array(
                [d.read_count for d in result.species_detections],
                dtype=np.float64,
            )
            abundances = abundances[abundances > 0]

            if len(abundances) > 0:
                analyzer = eDNACommunityAnalyzer()
                indices = analyzer.compute_diversity_indices(abundances)
                lines.append("  Diversity Indices:")
                lines.append(f"    Shannon (H'):   {indices['shannon']:.3f}")
                lines.append(f"    Simpson (1-D):  {indices['simpson']:.3f}")
                lines.append(f"    Chao1 Est.:     {indices['chao1']:.1f}")
                lines.append("")

                # Interpret Shannon diversity
                shannon = indices["shannon"]
                if shannon < 1.0:
                    health = "Poor -- very low diversity, ecosystem may be stressed"
                elif shannon < 2.0:
                    health = "Fair -- below average diversity"
                elif shannon < 3.0:
                    health = "Good -- moderate diversity, typical of healthy systems"
                elif shannon < 4.0:
                    health = "Very Good -- high diversity, strong ecosystem health"
                else:
                    health = "Excellent -- exceptionally high diversity"
                lines.append(f"  Health Assessment: {health}")
                lines.append("")

        # Invasive species
        analyzer = eDNACommunityAnalyzer()
        invasives = analyzer.detect_invasive_species(result.species_detections)
        if invasives:
            lines.append("  *** INVASIVE SPECIES DETECTED ***")
            for inv in invasives:
                lines.append(
                    f"    - {inv['common_name']} ({inv['scientific_name']}): "
                    f"{inv['read_count']:,} reads"
                )
            lines.append("")
            lines.append(
                "  Please report invasive species detections to your local"
            )
            lines.append(
                "  wildlife agency and the USGS Nonindigenous Aquatic Species"
            )
            lines.append("  database (https://nas.er.usgs.gov/).")
            lines.append("")
        else:
            lines.append("  No invasive species detected.")
            lines.append("")

        # Top species
        sorted_dets = sorted(
            result.species_detections,
            key=lambda d: d.read_count,
            reverse=True,
        )
        top_n = min(10, len(sorted_dets))
        if top_n > 0:
            lines.append(f"  Top {top_n} Species by Read Count:")
            for i, det in enumerate(sorted_dets[:top_n], 1):
                pct = (
                    det.read_count / total_reads * 100
                    if total_reads > 0
                    else 0.0
                )
                lines.append(
                    f"    {i:2d}. {det.taxon_name:<35s} "
                    f"{det.read_count:>8,} reads ({pct:.1f}%)"
                )
            lines.append("")

        lines.append(
            "  Thank you for contributing to freshwater ecosystem monitoring!"
        )
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integration functions
# ---------------------------------------------------------------------------


def edna_to_sentinel_embedding(
    result: eDNAKitResult,
    model: Optional[Any] = None,
) -> np.ndarray:
    """Convert eDNA results to the SENTINEL shared embedding space.

    Uses a MicroBiomeNet encoder (if provided) to project the community
    composition into the SENTINEL 256-dimensional embedding space.
    Falls back to a deterministic feature-based embedding when no model
    is available.

    Parameters
    ----------
    result:
        Validated :class:`eDNAKitResult`.
    model:
        Optional MicroBiomeNet encoder (``nn.Module`` with a ``forward``
        method accepting an abundance tensor).  If *None*, a
        handcrafted feature embedding is computed.

    Returns
    -------
    numpy array of shape ``(256,)`` in the SENTINEL embedding space.
    """
    abundances = np.array(
        [d.read_count for d in result.species_detections], dtype=np.float64
    )

    if len(abundances) == 0:
        return np.zeros(SENTINEL_EMBEDDING_DIM, dtype=np.float32)

    # If a model is provided, use it for encoding
    if model is not None:
        try:
            import torch
            from torch.amp import autocast

            # CLR-transform the abundances
            clr = eDNAIngestion.compute_relative_abundance(abundances)
            # Pad or truncate to model's expected input size
            if hasattr(model, "input_dim"):
                input_dim = model.input_dim
            else:
                input_dim = len(clr)

            padded = np.zeros(input_dim, dtype=np.float32)
            n = min(len(clr), input_dim)
            padded[:n] = clr[:n]

            tensor = torch.from_numpy(padded).unsqueeze(0)
            if next(model.parameters(), None) is not None:
                device = next(model.parameters()).device
                tensor = tensor.to(device)

            model.eval()
            with torch.no_grad():
                with autocast("cuda"):
                    embedding = model(tensor)

            return embedding.squeeze(0).cpu().numpy()

        except Exception as exc:
            logger.warning(
                f"MicroBiomeNet encoding failed, falling back to "
                f"feature embedding: {exc}"
            )

    # Fallback: handcrafted feature embedding
    embedding = np.zeros(SENTINEL_EMBEDDING_DIM, dtype=np.float32)

    # Diversity features (indices 0-15)
    pos_abundances = abundances[abundances > 0]
    if len(pos_abundances) > 0:
        total = pos_abundances.sum()
        props = pos_abundances / total
        shannon = float(-np.sum(props * np.log(props)))
        simpson = float(1.0 - np.sum(props ** 2))
        richness = len(pos_abundances)
        evenness = shannon / np.log(richness) if richness > 1 else 0.0

        embedding[0] = shannon / 5.0  # normalise to ~[0,1]
        embedding[1] = simpson
        embedding[2] = min(richness / 200.0, 1.0)
        embedding[3] = float(evenness)
        embedding[4] = float(np.log1p(total)) / 20.0
    else:
        return embedding

    # CLR-transformed abundance features (indices 16-143)
    clr = eDNAIngestion.compute_relative_abundance(pos_abundances)
    n_clr = min(len(clr), 128)
    embedding[16 : 16 + n_clr] = clr[:n_clr].astype(np.float32)

    # Spatial features (indices 144-147)
    embedding[144] = result.latitude / 90.0
    embedding[145] = result.longitude / 180.0
    embedding[146] = math.sin(math.radians(result.latitude))
    embedding[147] = math.cos(math.radians(result.longitude))

    # Temporal features (indices 148-151)
    doy = result.collection_date.timetuple().tm_yday
    embedding[148] = math.sin(2 * math.pi * doy / 365.0)
    embedding[149] = math.cos(2 * math.pi * doy / 365.0)
    embedding[150] = result.collection_date.year / 2030.0
    embedding[151] = result.sample_volume_ml / 5000.0

    # Invasive species indicator features (indices 152-159)
    analyzer = eDNACommunityAnalyzer()
    invasives = analyzer.detect_invasive_species(result.species_detections)
    embedding[152] = min(len(invasives) / 5.0, 1.0)
    for i, inv in enumerate(invasives[:7]):
        embedding[153 + i] = inv["read_count"] / max(total, 1.0)

    # L2 normalise the full embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding


def submit_to_platform(
    result: eDNAKitResult,
    api_url: str = "https://api.sentinel-water.org/v1/edna",
) -> Dict[str, Any]:
    """Submit validated eDNA results to the SENTINEL platform API.

    Validates the result, generates an embedding, and posts to the
    SENTINEL REST API.

    Parameters
    ----------
    result:
        A validated :class:`eDNAKitResult`.
    api_url:
        SENTINEL platform API endpoint URL.

    Returns
    -------
    Dictionary with submission status and response metadata.
    """
    # Validate first
    validator = eDNAValidator()
    report = validator.validate(result)

    if not report.is_valid:
        return {
            "status": "rejected",
            "kit_id": result.kit_id,
            "reason": "Validation failed",
            "errors": [f.message for f in report.flags if f.severity == "error"],
        }

    # Generate embedding
    embedding = edna_to_sentinel_embedding(result)

    # Build submission payload
    payload = {
        "kit_id": result.kit_id,
        "collector_name": result.collector_name,
        "collection_date": result.collection_date.isoformat(),
        "latitude": result.latitude,
        "longitude": result.longitude,
        "water_body_name": result.water_body_name,
        "sample_volume_ml": result.sample_volume_ml,
        "filter_type": result.filter_type,
        "quality_tier": report.quality_tier,
        "species_count": len(result.species_detections),
        "total_reads": sum(d.read_count for d in result.species_detections),
        "diversity_metrics": report.diversity_metrics,
        "embedding": embedding.tolist(),
        "detections": [
            {
                "taxon_name": d.taxon_name,
                "read_count": d.read_count,
                "confidence": d.confidence,
            }
            for d in result.species_detections
        ],
    }

    # Attempt HTTP submission
    try:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
            return {
                "status": "submitted",
                "kit_id": result.kit_id,
                "quality_tier": report.quality_tier,
                "response": response_data,
            }
    except Exception as exc:
        logger.warning(
            f"API submission failed for kit {result.kit_id}: {exc}. "
            "Result validated and embedding computed locally."
        )
        return {
            "status": "offline",
            "kit_id": result.kit_id,
            "quality_tier": report.quality_tier,
            "embedding_shape": list(embedding.shape),
            "message": (
                "Validated locally but could not reach SENTINEL API. "
                "Result can be submitted later."
            ),
        }


# ---------------------------------------------------------------------------
# Demo / standalone test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("SENTINEL eDNA Kit Module -- Demo\n")

    # Build a synthetic eDNA kit result
    synthetic_detections = [
        SpeciesDetection("Oncorhynchus mykiss", 4520, 0.95, taxonomic_rank="species"),
        SpeciesDetection("Salmo trutta", 3100, 0.92, taxonomic_rank="species"),
        SpeciesDetection("Cottus bairdii", 2800, 0.88, taxonomic_rank="species"),
        SpeciesDetection("Rhinichthys atratulus", 1950, 0.85, taxonomic_rank="species"),
        SpeciesDetection("Campostoma anomalum", 1700, 0.82, taxonomic_rank="species"),
        SpeciesDetection("Ephemeroptera sp.", 3200, 0.75, taxonomic_rank="order"),
        SpeciesDetection("Plecoptera sp.", 2100, 0.72, taxonomic_rank="order"),
        SpeciesDetection("Trichoptera sp.", 1800, 0.70, taxonomic_rank="order"),
        SpeciesDetection("Dreissena polymorpha", 850, 0.88, taxonomic_rank="species"),
        SpeciesDetection("Cladophora glomerata", 1500, 0.65, taxonomic_rank="species"),
        SpeciesDetection("Anabaena circinalis", 600, 0.60, taxonomic_rank="species"),
        SpeciesDetection("Daphnia pulex", 950, 0.78, taxonomic_rank="species"),
        SpeciesDetection("Homo sapiens", 120, 0.99, taxonomic_rank="species"),
    ]

    demo_result = eDNAKitResult(
        kit_id="EDNA-2026-0042",
        collector_name="Jane Citizen",
        collection_date=datetime(2026, 5, 15, 10, 30),
        latitude=42.3601,
        longitude=-71.0589,
        water_body_name="Charles River",
        sample_volume_ml=1000.0,
        filter_type="0.45um_MCE",
        species_detections=synthetic_detections,
        metadata={"kit_provider": "demo", "primer_set": "MiFish-U/E"},
    )

    # Stage 1-3 Validation
    print("--- Validation ---")
    validator = eDNAValidator(ecoregion="northern_appalachians")
    report = validator.validate(demo_result)
    print(f"Kit ID:       {report.kit_id}")
    print(f"Valid:         {report.is_valid}")
    print(f"Quality Tier:  {report.quality_tier}")
    print(f"Stages Passed: {report.stage_passed}")
    print(f"Errors:        {report.error_count}")
    print(f"Warnings:      {report.warning_count}")
    if report.contamination_warnings:
        print(f"Contamination: {report.contamination_warnings}")
    if report.diversity_metrics:
        print(f"Diversity:     {report.diversity_metrics}")
    for flag in report.flags:
        print(f"  [{flag.severity.upper()}] Stage {flag.stage}: {flag.message}")
    print()

    # Community analysis
    print("--- Community Analysis ---")
    analyzer = eDNACommunityAnalyzer()
    abundances = np.array(
        [d.read_count for d in demo_result.species_detections], dtype=np.float64
    )
    indices = analyzer.compute_diversity_indices(abundances)
    print(f"Diversity indices: {indices}")

    ref_comparison = analyzer.compare_to_reference(
        abundances, ecoregion="northern_appalachians"
    )
    print(f"Reference comparison: {ref_comparison['status']}")
    for msg in ref_comparison.get("messages", []):
        print(f"  {msg}")

    invasives = analyzer.detect_invasive_species(demo_result.species_detections)
    if invasives:
        print(f"Invasive species found: {len(invasives)}")
        for inv in invasives:
            print(f"  - {inv['common_name']} ({inv['scientific_name']})")
    print()

    # Taxonomy normalisation
    print("--- Taxonomy Normalisation ---")
    normalised = eDNAIngestion.normalize_taxonomy(demo_result.species_detections)
    for orig, norm in zip(demo_result.species_detections[:3], normalised[:3]):
        print(f"  {orig.taxon_name} -> {norm.taxon_name}")
    print()

    # Embedding
    print("--- SENTINEL Embedding ---")
    embedding = edna_to_sentinel_embedding(demo_result)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm:  {np.linalg.norm(embedding):.4f}")
    print(f"First 10 dims:   {embedding[:10]}")
    print()

    # Health report
    print("--- Health Report ---")
    health_report = analyzer.generate_health_report(demo_result)
    print(health_report)
    print()

    # Platform submission (will fail gracefully without network)
    print("--- Platform Submission ---")
    submission = submit_to_platform(demo_result)
    print(f"Submission status: {submission['status']}")
    print(f"Quality tier:      {submission.get('quality_tier', 'N/A')}")
    if submission.get("message"):
        print(f"Message:           {submission['message']}")
