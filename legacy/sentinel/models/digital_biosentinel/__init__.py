"""Digital Biosentinel — computational replacement for wet-lab biosentinel organisms.

Trained on EPA ECOTOX (~1M records, ~12,000 chemicals, ~13,000 species),
the Digital Biosentinel predicts ecological impact of detected water
contaminants across a panel of standard EPA sentinel species.
"""

from .calibration import (
    TemperatureScaler,
    expected_calibration_error,
    plot_reliability_diagram,
)
from .chemical_encoder import (
    CHEMICAL_CLASS_TO_IDX,
    CHEMICAL_CLASSES,
    ChemicalEncoder,
)
from .dataset import (
    ECOTOXDataset,
    build_balanced_sampler,
    build_vocabularies,
    compute_descriptor_stats,
    ecotox_collate_fn,
    split_by_chemical,
)
from .dose_response import DoseResponseModel, DoseResponseOutput
from .model import (
    DigitalBiosentinel,
    DigitalBiosentinelOutput,
    SENTINEL_SPECIES_PANEL,
    SentinelSpecies,
    SpeciesImpactPrediction,
)
from .species_encoder import (
    TAXONOMIC_RANKS,
    TROPHIC_LEVEL_TO_IDX,
    TROPHIC_LEVELS,
    SpeciesEncoder,
)

__all__ = [
    # Core model
    "DigitalBiosentinel",
    "DigitalBiosentinelOutput",
    "SpeciesImpactPrediction",
    "SentinelSpecies",
    "SENTINEL_SPECIES_PANEL",
    # Encoders
    "ChemicalEncoder",
    "SpeciesEncoder",
    # Dose-response
    "DoseResponseModel",
    "DoseResponseOutput",
    # Calibration
    "TemperatureScaler",
    "expected_calibration_error",
    "plot_reliability_diagram",
    # Dataset
    "ECOTOXDataset",
    "ecotox_collate_fn",
    "build_balanced_sampler",
    "build_vocabularies",
    "compute_descriptor_stats",
    "split_by_chemical",
    # Constants
    "CHEMICAL_CLASSES",
    "CHEMICAL_CLASS_TO_IDX",
    "TAXONOMIC_RANKS",
    "TROPHIC_LEVELS",
    "TROPHIC_LEVEL_TO_IDX",
]
