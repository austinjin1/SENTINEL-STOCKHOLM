"""Full Digital Biosentinel model.

Combines the chemical encoder, species encoder, dose-response model,
and temperature-scaling calibration into a single end-to-end module.

At inference time the model accepts detected water chemistry (parameter
values from the SENTINEL sensor encoder) and predicts ecological impact
for a panel of key sentinel species used by the EPA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .calibration import TemperatureScaler
from .chemical_encoder import (
    CHEMICAL_CLASS_TO_IDX,
    CHEMICAL_CLASSES,
    NUM_MOLECULAR_DESCRIPTORS,
    ChemicalEncoder,
)
from .dose_response import DoseResponseModel, DoseResponseOutput
from .species_encoder import (
    TAXONOMIC_RANKS,
    TROPHIC_LEVEL_TO_IDX,
    TROPHIC_LEVELS,
    SpeciesEncoder,
)


# ---------------------------------------------------------------------------
# Key sentinel species panel (EPA standard test organisms)
# ---------------------------------------------------------------------------

@dataclass
class SentinelSpecies:
    """Metadata for a sentinel species in the monitoring panel."""

    common_name: str
    latin_name: str
    taxonomy: Dict[str, str]
    trophic_level: str
    primary_endpoint: str
    sensitivity_note: str


SENTINEL_SPECIES_PANEL: List[SentinelSpecies] = [
    SentinelSpecies(
        common_name="Water flea",
        latin_name="Daphnia magna",
        taxonomy={
            "phylum": "Arthropoda",
            "class": "Branchiopoda",
            "order": "Cladocera",
            "family": "Daphniidae",
            "genus": "Daphnia",
            "species": "magna",
        },
        trophic_level="primary_consumer",
        primary_endpoint="mortality",
        sensitivity_note="Sensitive to organophosphates and metals",
    ),
    SentinelSpecies(
        common_name="Fathead minnow",
        latin_name="Pimephales promelas",
        taxonomy={
            "phylum": "Chordata",
            "class": "Actinopterygii",
            "order": "Cypriniformes",
            "family": "Cyprinidae",
            "genus": "Pimephales",
            "species": "promelas",
        },
        trophic_level="secondary_consumer",
        primary_endpoint="mortality",
        sensitivity_note="Standard EPA acute/chronic test fish",
    ),
    SentinelSpecies(
        common_name="Rainbow trout",
        latin_name="Oncorhynchus mykiss",
        taxonomy={
            "phylum": "Chordata",
            "class": "Actinopterygii",
            "order": "Salmoniformes",
            "family": "Salmonidae",
            "genus": "Oncorhynchus",
            "species": "mykiss",
        },
        trophic_level="tertiary_consumer",
        primary_endpoint="mortality",
        sensitivity_note="Coldwater salmonid indicator, sensitive to low DO and metals",
    ),
    SentinelSpecies(
        common_name="Water flea (small)",
        latin_name="Ceriodaphnia dubia",
        taxonomy={
            "phylum": "Arthropoda",
            "class": "Branchiopoda",
            "order": "Cladocera",
            "family": "Daphniidae",
            "genus": "Ceriodaphnia",
            "species": "dubia",
        },
        trophic_level="primary_consumer",
        primary_endpoint="reproduction_effect",
        sensitivity_note="Reproductive endpoint, 7-day chronic test standard",
    ),
    SentinelSpecies(
        common_name="Scud",
        latin_name="Hyalella azteca",
        taxonomy={
            "phylum": "Arthropoda",
            "class": "Malacostraca",
            "order": "Amphipoda",
            "family": "Hyalellidae",
            "genus": "Hyalella",
            "species": "azteca",
        },
        trophic_level="primary_consumer",
        primary_endpoint="mortality",
        sensitivity_note="Sediment toxicity indicator, sensitive to pyrethroids",
    ),
]


# ---------------------------------------------------------------------------
# Mechanism inference helper
# ---------------------------------------------------------------------------

# Mapping from dominant chemical class to plausible toxicity mechanism
_CLASS_MECHANISM_MAP: Dict[str, str] = {
    "organophosphate": "acetylcholinesterase inhibition",
    "organochlorine": "GABA receptor disruption / chloride channel blockade",
    "carbamate": "reversible acetylcholinesterase inhibition",
    "pyrethroid": "sodium channel modulation (Type I/II)",
    "neonicotinoid": "nicotinic acetylcholine receptor agonism",
    "triazine": "photosystem II inhibition (primary producers)",
    "phenol": "membrane disruption / narcosis",
    "phthalate": "endocrine disruption (anti-androgenic)",
    "polycyclic_aromatic_hydrocarbon": "DNA adduct formation / oxidative stress",
    "heavy_metal": "enzyme inhibition / oxidative stress / ion mimicry",
    "surfactant": "gill membrane damage / surfactant narcosis",
    "pharmaceutical": "target-specific pharmacological activity",
    "per_polyfluoroalkyl": "lipid metabolism disruption / immunotoxicity",
    "nitroaromatic": "methemoglobinemia / oxidative stress",
    "chlorinated_solvent": "central nervous system depression / narcosis",
    "petroleum_hydrocarbon": "baseline narcosis / membrane disruption",
    "inorganic_acid": "pH shock / direct tissue damage",
    "inorganic_base": "pH shock / caustic tissue damage",
    "nutrient": "eutrophication / dissolved oxygen depletion",
    "other": "non-specific / unknown mechanism",
}


def _infer_mechanism(chemical_class: str) -> str:
    """Return the most likely toxicity mechanism for a chemical class."""
    return _CLASS_MECHANISM_MAP.get(chemical_class, "unknown mechanism")


# ---------------------------------------------------------------------------
# Species-level impact prediction result
# ---------------------------------------------------------------------------

@dataclass
class SpeciesImpactPrediction:
    """Prediction result for a single sentinel species."""

    species_name: str
    common_name: str
    mortality_probability: float
    growth_inhibition_pct: float
    reproduction_effect: float
    behavioral_change_probability: float
    uncertainty: float
    confidence_pct: float
    exposure_hours: float
    primary_endpoint: str
    mechanism: str
    risk_level: str  # "low", "moderate", "high", "critical"

    def summary(self) -> str:
        """Human-readable summary string."""
        return (
            f"At detected concentrations, SENTINEL predicts "
            f"{self.mortality_probability * 100:.1f}% mortality risk for "
            f"{self.species_name} within {self.exposure_hours:.0f} hours, "
            f"primarily through {self.mechanism}, with "
            f"{self.confidence_pct:.0f}% confidence."
        )


@dataclass
class DigitalBiosentinelOutput:
    """Full output from the Digital Biosentinel model."""

    impact_predictions: Dict[str, SpeciesImpactPrediction]
    uncertainty: Dict[str, float]
    dominant_mechanism: str
    overall_risk_level: str
    chemical_classes_detected: List[str]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DigitalBiosentinel(nn.Module):
    """End-to-end Digital Biosentinel model.

    Combines chemical encoding, species encoding, multi-endpoint dose-
    response prediction, and post-hoc calibration into a single module
    for ecological impact assessment.

    Parameters
    ----------
    num_chemicals : int
        Chemical vocabulary size (ECOTOX catalogue).
    species_vocab_sizes : Dict[str, int]
        Per-rank vocabulary sizes for the species encoder.
    chemical_dim : int
        Chemical embedding dimensionality.
    species_dim : int
        Species embedding dimensionality.
    backbone_dims : Tuple[int, ...]
        Dose-response backbone hidden sizes.
    dropout : float
        Dropout probability.
    mc_samples : int
        Monte Carlo dropout sample count.
    default_exposure_hours : float
        Default acute exposure duration (hours) if not specified.
    """

    def __init__(
        self,
        num_chemicals: int = 12000,
        species_vocab_sizes: Optional[Dict[str, int]] = None,
        chemical_dim: int = 128,
        species_dim: int = 64,
        backbone_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
        mc_samples: int = 20,
        default_exposure_hours: float = 48.0,
    ) -> None:
        super().__init__()
        self.default_exposure_hours = default_exposure_hours
        self.mc_samples = mc_samples

        if species_vocab_sizes is None:
            species_vocab_sizes = {
                "phylum": 40,
                "class": 120,
                "order": 500,
                "family": 2000,
                "genus": 6000,
                "species": 13000,
            }

        # --- Sub-models --------------------------------------------------------
        self.chemical_encoder = ChemicalEncoder(
            num_chemicals=num_chemicals,
            embedding_dim=chemical_dim,
            dropout=dropout * 0.33,  # lighter dropout for encoder
        )

        self.species_encoder = SpeciesEncoder(
            vocab_sizes=species_vocab_sizes,
            embedding_dim=species_dim,
            dropout=dropout * 0.33,
        )

        self.dose_response = DoseResponseModel(
            chemical_dim=chemical_dim,
            species_dim=species_dim,
            backbone_dims=backbone_dims,
            dropout=dropout,
            mc_samples=mc_samples,
        )

        # --- Calibration (fitted post-training) --------------------------------
        self.calibrator_mortality = TemperatureScaler()
        self.calibrator_behavioral = TemperatureScaler()

        # --- Vocabulary look-ups (populated during data loading) ----------------
        self._chemical_to_idx: Dict[str, int] = {}
        self._species_taxon_to_idx: Dict[str, Dict[str, int]] = {
            rank: {} for rank in TAXONOMIC_RANKS
        }

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def register_chemical_vocab(self, cas_to_idx: Dict[str, int]) -> None:
        """Register CAS-number-to-index mapping (built from ECOTOX data)."""
        self._chemical_to_idx = cas_to_idx

    def register_species_vocab(
        self, taxon_to_idx: Dict[str, Dict[str, int]]
    ) -> None:
        """Register per-rank taxonomy-to-index mapping."""
        self._species_taxon_to_idx = taxon_to_idx

    def _resolve_chemical_idx(self, cas: str) -> int:
        """Map CAS number to vocabulary index; 0 for unknown."""
        return self._chemical_to_idx.get(cas, 0)

    def _resolve_taxonomy_indices(
        self, taxonomy: Dict[str, str], device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Map taxonomy dict to per-rank index tensors (batch size 1)."""
        result: Dict[str, torch.Tensor] = {}
        for rank in TAXONOMIC_RANKS:
            name = taxonomy.get(rank, "")
            idx = self._species_taxon_to_idx.get(rank, {}).get(name, 0)
            result[rank] = torch.tensor([idx], dtype=torch.long, device=device)
        return result

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        chemical_idx: torch.Tensor,
        class_idx: torch.Tensor,
        descriptors: torch.Tensor,
        taxonomy_indices: Dict[str, torch.Tensor],
        trophic_idx: torch.Tensor,
        log_concentration: torch.Tensor,
        log_exposure_hours: torch.Tensor,
        trophic_level_numeric: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> Tuple[DoseResponseOutput, torch.Tensor, torch.Tensor]:
        """Full forward pass for training or batched inference.

        Parameters
        ----------
        chemical_idx : Tensor[B]
        class_idx : Tensor[B]
        descriptors : Tensor[B, num_descriptors]
        taxonomy_indices : Dict[str, Tensor[B]]
        trophic_idx : Tensor[B]
        log_concentration : Tensor[B]
        log_exposure_hours : Tensor[B]
        trophic_level_numeric : Tensor[B]
        use_mc_dropout : bool

        Returns
        -------
        dose_output : DoseResponseOutput
        chemical_embed : Tensor[B, chemical_dim]
        species_embed : Tensor[B, species_dim]
        """
        chemical_embed, is_unknown = self.chemical_encoder(
            chemical_idx, class_idx, descriptors,
        )

        species_embed = self.species_encoder(
            taxonomy_indices, trophic_idx,
        )

        dose_output = self.dose_response(
            chemical_embed=chemical_embed,
            log_concentration=log_concentration,
            species_embed=species_embed,
            log_exposure_hours=log_exposure_hours,
            trophic_level=trophic_level_numeric,
            use_mc_dropout=use_mc_dropout,
        )

        return dose_output, chemical_embed, species_embed

    # ------------------------------------------------------------------
    # Inference entry-point: predict ecological impact
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_ecological_impact(
        self,
        detected_chemicals: List[Dict[str, Any]],
        exposure_hours: Optional[float] = None,
        species_panel: Optional[List[SentinelSpecies]] = None,
        use_mc_dropout: bool = True,
    ) -> DigitalBiosentinelOutput:
        """Predict ecological impact for the sentinel species panel.

        This is the main inference entry-point.  Given a list of detected
        chemicals (from SENTINEL sensor/source-attribution output), it
        returns per-species impact predictions with confidence estimates.

        Parameters
        ----------
        detected_chemicals : list of dict
            Each dict must contain:
              - ``"cas"`` : str — CAS registry number
              - ``"concentration_mg_l"`` : float — detected concentration
              - ``"chemical_class"`` : str — from CHEMICAL_CLASSES
              - ``"descriptors"`` : list[float] — molecular descriptors
                (length ``NUM_MOLECULAR_DESCRIPTORS``)
        exposure_hours : float, optional
            Acute exposure duration.  Defaults to ``default_exposure_hours``.
        species_panel : list of SentinelSpecies, optional
            Override the default EPA panel.
        use_mc_dropout : bool
            Whether to use MC dropout for uncertainty estimation.

        Returns
        -------
        DigitalBiosentinelOutput
        """
        self.eval()

        if exposure_hours is None:
            exposure_hours = self.default_exposure_hours
        if species_panel is None:
            species_panel = SENTINEL_SPECIES_PANEL

        device = next(self.parameters()).device

        # We take the *worst-case* chemical for each species (most toxic).
        # For each chemical × species pair, run the model and keep the max.

        impact_predictions: Dict[str, SpeciesImpactPrediction] = {}
        uncertainty_dict: Dict[str, float] = {}
        chemical_classes_seen: List[str] = []

        for sp in species_panel:
            best_mortality = 0.0
            best_output: Optional[DoseResponseOutput] = None
            best_chem_class = "other"

            for chem in detected_chemicals:
                cas = chem["cas"]
                conc = chem["concentration_mg_l"]
                chem_class = chem.get("chemical_class", "other")
                desc = chem.get("descriptors", [0.0] * NUM_MOLECULAR_DESCRIPTORS)

                if chem_class not in chemical_classes_seen:
                    chemical_classes_seen.append(chem_class)

                # Prepare tensors (batch size 1)
                chem_idx_t = torch.tensor(
                    [self._resolve_chemical_idx(cas)],
                    dtype=torch.long, device=device,
                )
                class_idx_t = torch.tensor(
                    [CHEMICAL_CLASS_TO_IDX.get(chem_class, len(CHEMICAL_CLASSES) - 1)],
                    dtype=torch.long, device=device,
                )
                desc_t = torch.tensor(
                    [desc], dtype=torch.float32, device=device,
                )
                tax_idx = self._resolve_taxonomy_indices(sp.taxonomy, device)
                trophic_idx_t = torch.tensor(
                    [TROPHIC_LEVEL_TO_IDX.get(sp.trophic_level, 1)],
                    dtype=torch.long, device=device,
                )

                log_conc = torch.tensor(
                    [math.log10(max(conc, 1e-12))],
                    dtype=torch.float32, device=device,
                )
                log_hours = torch.tensor(
                    [math.log10(max(exposure_hours, 0.1))],
                    dtype=torch.float32, device=device,
                )
                trophic_numeric = torch.tensor(
                    [float(TROPHIC_LEVEL_TO_IDX.get(sp.trophic_level, 1))],
                    dtype=torch.float32, device=device,
                )

                dose_out, _, _ = self.forward(
                    chemical_idx=chem_idx_t,
                    class_idx=class_idx_t,
                    descriptors=desc_t,
                    taxonomy_indices=tax_idx,
                    trophic_idx=trophic_idx_t,
                    log_concentration=log_conc,
                    log_exposure_hours=log_hours,
                    trophic_level_numeric=trophic_numeric,
                    use_mc_dropout=use_mc_dropout,
                )

                mort_val = dose_out.mortality.item()
                if mort_val > best_mortality:
                    best_mortality = mort_val
                    best_output = dose_out
                    best_chem_class = chem_class

            if best_output is None:
                continue

            # Compute confidence from uncertainty
            unc_val = best_output.uncertainty.item()
            confidence = max(0.0, min(100.0, (1.0 - unc_val) * 100.0))

            # Risk level classification
            mort_p = best_output.mortality.item()
            if mort_p >= 0.75:
                risk = "critical"
            elif mort_p >= 0.50:
                risk = "high"
            elif mort_p >= 0.25:
                risk = "moderate"
            else:
                risk = "low"

            prediction = SpeciesImpactPrediction(
                species_name=sp.latin_name,
                common_name=sp.common_name,
                mortality_probability=mort_p,
                growth_inhibition_pct=best_output.growth_inhibition.item(),
                reproduction_effect=best_output.reproduction_effect.item(),
                behavioral_change_probability=best_output.behavioral_change.item(),
                uncertainty=unc_val,
                confidence_pct=confidence,
                exposure_hours=exposure_hours,
                primary_endpoint=sp.primary_endpoint,
                mechanism=_infer_mechanism(best_chem_class),
                risk_level=risk,
            )

            impact_predictions[sp.latin_name] = prediction
            uncertainty_dict[sp.latin_name] = unc_val

        # Determine dominant mechanism (from highest-risk species)
        if impact_predictions:
            worst_species = max(
                impact_predictions.values(),
                key=lambda p: p.mortality_probability,
            )
            dominant_mechanism = worst_species.mechanism
            overall_risk = worst_species.risk_level
        else:
            dominant_mechanism = "none detected"
            overall_risk = "low"

        return DigitalBiosentinelOutput(
            impact_predictions=impact_predictions,
            uncertainty=uncertainty_dict,
            dominant_mechanism=dominant_mechanism,
            overall_risk_level=overall_risk,
            chemical_classes_detected=chemical_classes_seen,
        )

    # ------------------------------------------------------------------
    # Calibration workflow
    # ------------------------------------------------------------------

    def calibrate_on_validation(
        self,
        val_logits_mortality: torch.Tensor,
        val_labels_mortality: torch.Tensor,
        val_logits_behavioral: Optional[torch.Tensor] = None,
        val_labels_behavioral: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Fit temperature scalers on validation data.

        Should be called **after** training is complete.  Gathers
        pre-sigmoid logits from the validation set and fits T.

        Parameters
        ----------
        val_logits_mortality : Tensor[N]
            Pre-sigmoid mortality logits.
        val_labels_mortality : Tensor[N]
            Binary mortality labels.
        val_logits_behavioral : Tensor[N], optional
        val_labels_behavioral : Tensor[N], optional
        verbose : bool

        Returns
        -------
        dict
            ``{"mortality_temperature": T1, "behavioral_temperature": T2}``.
        """
        result: Dict[str, float] = {}

        t_mort = self.calibrator_mortality.calibrate(
            val_logits_mortality, val_labels_mortality, verbose=verbose,
        )
        result["mortality_temperature"] = t_mort

        if val_logits_behavioral is not None and val_labels_behavioral is not None:
            t_behav = self.calibrator_behavioral.calibrate(
                val_logits_behavioral, val_labels_behavioral, verbose=verbose,
            )
            result["behavioral_temperature"] = t_behav

        return result
