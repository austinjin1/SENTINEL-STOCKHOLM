"""PyTorch Dataset for EPA ECOTOX data.

Handles loading, preprocessing, class balancing (oversampling rare
endpoint types), train/val/test splitting by chemical identity for
proper generalisation evaluation, and data augmentation (concentration
jittering).

Expected ECOTOX data format: a processed Parquet or CSV file with columns:
    cas_number, chemical_name, chemical_class, concentration_mg_l,
    species_latin_name, phylum, class_, order, family, genus, species_epithet,
    trophic_level, exposure_hours,
    effect_mortality (0/1), effect_growth_inhibition (float),
    effect_reproduction (float), effect_behavioral (0/1),
    molecular_descriptors (JSON array or 8 separate columns mw, logkow, …)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from .chemical_encoder import CHEMICAL_CLASS_TO_IDX, CHEMICAL_CLASSES, NUM_MOLECULAR_DESCRIPTORS
from .species_encoder import TAXONOMIC_RANKS, TROPHIC_LEVEL_TO_IDX

logger = logging.getLogger(__name__)

# Descriptor column names (if stored as separate columns)
DESCRIPTOR_COLUMNS = [
    "molecular_weight",
    "log_kow",
    "water_solubility_mg_l",
    "vapour_pressure_pa",
    "henrys_law_constant",
    "log_bcf",
    "polar_surface_area",
    "num_rotatable_bonds",
]


class ECOTOXDataset(Dataset):
    """PyTorch Dataset wrapping processed EPA ECOTOX records.

    Parameters
    ----------
    records : list of dict
        Pre-processed ECOTOX records.  Each dict should contain the keys
        described in the module docstring.
    chemical_vocab : Dict[str, int]
        Mapping from CAS number to chemical vocabulary index.
    species_vocab : Dict[str, Dict[str, int]]
        Per-rank mapping from taxon name to vocabulary index.
    augment : bool
        Whether to apply concentration jittering (training only).
    jitter_std : float
        Standard deviation of log10-concentration Gaussian noise.
    descriptor_means : Optional[np.ndarray]
        Descriptor standardisation means (length ``NUM_MOLECULAR_DESCRIPTORS``).
    descriptor_stds : Optional[np.ndarray]
        Descriptor standardisation stds.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        chemical_vocab: Dict[str, int],
        species_vocab: Dict[str, Dict[str, int]],
        augment: bool = False,
        jitter_std: float = 0.1,
        descriptor_means: Optional[np.ndarray] = None,
        descriptor_stds: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.records = records
        self.chemical_vocab = chemical_vocab
        self.species_vocab = species_vocab
        self.augment = augment
        self.jitter_std = jitter_std

        self.descriptor_means = (
            descriptor_means
            if descriptor_means is not None
            else np.zeros(NUM_MOLECULAR_DESCRIPTORS, dtype=np.float32)
        )
        self.descriptor_stds = (
            descriptor_stds
            if descriptor_stds is not None
            else np.ones(NUM_MOLECULAR_DESCRIPTORS, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        # --- Chemical features -------------------------------------------------
        cas = rec.get("cas_number", "")
        chemical_idx = self.chemical_vocab.get(cas, 0)
        chem_class = rec.get("chemical_class", "other")
        class_idx = CHEMICAL_CLASS_TO_IDX.get(chem_class, len(CHEMICAL_CLASSES) - 1)

        # Molecular descriptors
        descriptors = self._extract_descriptors(rec)

        # Concentration (log10-scaled, with optional jitter)
        conc = max(rec.get("concentration_mg_l", 1e-6), 1e-12)
        log_conc = math.log10(conc)
        if self.augment:
            log_conc += np.random.normal(0.0, self.jitter_std)

        # --- Species features --------------------------------------------------
        taxonomy_indices: Dict[str, int] = {}
        for rank in TAXONOMIC_RANKS:
            # ECOTOX uses "class_" to avoid Python keyword collision
            col = "class_" if rank == "class" else rank
            if rank == "species":
                col = "species_epithet"
            name = rec.get(col, "")
            taxonomy_indices[rank] = self.species_vocab.get(rank, {}).get(name, 0)

        trophic_str = rec.get("trophic_level", "primary_consumer")
        trophic_idx = TROPHIC_LEVEL_TO_IDX.get(trophic_str, 1)
        trophic_numeric = float(trophic_idx)

        # Exposure duration (log10 hours)
        hours = max(rec.get("exposure_hours", 48.0), 0.1)
        log_hours = math.log10(hours)

        # --- Targets -----------------------------------------------------------
        targets: Dict[str, float] = {}
        if "effect_mortality" in rec:
            targets["mortality"] = float(rec["effect_mortality"])
        if "effect_growth_inhibition" in rec:
            targets["growth_inhibition"] = float(rec["effect_growth_inhibition"])
        if "effect_reproduction" in rec:
            targets["reproduction_effect"] = float(rec["effect_reproduction"])
        if "effect_behavioral" in rec:
            targets["behavioral_change"] = float(rec["effect_behavioral"])

        return {
            "chemical_idx": chemical_idx,
            "class_idx": class_idx,
            "descriptors": descriptors,
            "taxonomy_indices": taxonomy_indices,
            "trophic_idx": trophic_idx,
            "log_concentration": log_conc,
            "log_exposure_hours": log_hours,
            "trophic_level_numeric": trophic_numeric,
            "targets": targets,
        }

    def _extract_descriptors(self, rec: Dict[str, Any]) -> np.ndarray:
        """Extract and standardise molecular descriptors from a record."""
        if "molecular_descriptors" in rec:
            raw = rec["molecular_descriptors"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            desc = np.array(raw, dtype=np.float32)
        else:
            desc = np.array(
                [rec.get(col, 0.0) for col in DESCRIPTOR_COLUMNS],
                dtype=np.float32,
            )

        # Pad or truncate to expected length
        if len(desc) < NUM_MOLECULAR_DESCRIPTORS:
            desc = np.pad(desc, (0, NUM_MOLECULAR_DESCRIPTORS - len(desc)))
        elif len(desc) > NUM_MOLECULAR_DESCRIPTORS:
            desc = desc[:NUM_MOLECULAR_DESCRIPTORS]

        # Replace NaN with 0
        desc = np.nan_to_num(desc, nan=0.0)

        # Standardise
        safe_std = np.where(self.descriptor_stds > 1e-8, self.descriptor_stds, 1.0)
        desc = (desc - self.descriptor_means) / safe_std

        return desc


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def ecotox_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that handles the nested taxonomy_indices dict."""
    B = len(batch)

    chemical_idx = torch.tensor([b["chemical_idx"] for b in batch], dtype=torch.long)
    class_idx = torch.tensor([b["class_idx"] for b in batch], dtype=torch.long)
    descriptors = torch.tensor(
        np.stack([b["descriptors"] for b in batch]), dtype=torch.float32,
    )

    taxonomy_indices: Dict[str, torch.Tensor] = {}
    for rank in TAXONOMIC_RANKS:
        taxonomy_indices[rank] = torch.tensor(
            [b["taxonomy_indices"][rank] for b in batch], dtype=torch.long,
        )

    trophic_idx = torch.tensor([b["trophic_idx"] for b in batch], dtype=torch.long)
    log_concentration = torch.tensor(
        [b["log_concentration"] for b in batch], dtype=torch.float32,
    )
    log_exposure_hours = torch.tensor(
        [b["log_exposure_hours"] for b in batch], dtype=torch.float32,
    )
    trophic_level_numeric = torch.tensor(
        [b["trophic_level_numeric"] for b in batch], dtype=torch.float32,
    )

    # Targets: gather only keys present in at least one record
    all_target_keys = set()
    for b in batch:
        all_target_keys.update(b["targets"].keys())

    targets: Dict[str, torch.Tensor] = {}
    for key in all_target_keys:
        vals = []
        for b in batch:
            vals.append(b["targets"].get(key, float("nan")))
        targets[key] = torch.tensor(vals, dtype=torch.float32)

    return {
        "chemical_idx": chemical_idx,
        "class_idx": class_idx,
        "descriptors": descriptors,
        "taxonomy_indices": taxonomy_indices,
        "trophic_idx": trophic_idx,
        "log_concentration": log_concentration,
        "log_exposure_hours": log_exposure_hours,
        "trophic_level_numeric": trophic_level_numeric,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Class-balanced sampler
# ---------------------------------------------------------------------------

def build_balanced_sampler(
    dataset: ECOTOXDataset,
    endpoint: str = "mortality",
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples rare effect types.

    The sampler assigns higher weight to records with rare positive effects
    (e.g., behavioral change is much rarer than mortality in ECOTOX).

    Parameters
    ----------
    dataset : ECOTOXDataset
    endpoint : str
        Which binary endpoint to balance on.  Supported:
        ``"mortality"``, ``"behavioral_change"``, or ``"multi"``
        (balances across all four endpoint categories).

    Returns
    -------
    WeightedRandomSampler
    """
    N = len(dataset)
    weights = np.ones(N, dtype=np.float64)

    if endpoint == "multi":
        # Assign each record to a category based on which endpoints are positive
        category_counts: Dict[str, int] = {}
        categories: List[str] = []
        for i in range(N):
            rec = dataset.records[i]
            cat_parts = []
            if rec.get("effect_mortality", 0) > 0.5:
                cat_parts.append("M")
            if rec.get("effect_behavioral", 0) > 0.5:
                cat_parts.append("B")
            if rec.get("effect_growth_inhibition", 0) > 10.0:
                cat_parts.append("G")
            if abs(rec.get("effect_reproduction", 0)) > 10.0:
                cat_parts.append("R")
            cat = "+".join(cat_parts) if cat_parts else "none"
            categories.append(cat)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Inverse frequency weighting
        for i, cat in enumerate(categories):
            weights[i] = N / max(category_counts[cat], 1)
    else:
        # Simple binary balancing
        target_key = {
            "mortality": "effect_mortality",
            "behavioral_change": "effect_behavioral",
        }.get(endpoint, f"effect_{endpoint}")

        pos_count = sum(
            1 for r in dataset.records if r.get(target_key, 0) > 0.5
        )
        neg_count = N - pos_count

        if pos_count > 0 and neg_count > 0:
            w_pos = N / (2.0 * pos_count)
            w_neg = N / (2.0 * neg_count)
            for i in range(N):
                if dataset.records[i].get(target_key, 0) > 0.5:
                    weights[i] = w_pos
                else:
                    weights[i] = w_neg

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=N,
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Train / val / test split by chemical identity
# ---------------------------------------------------------------------------

def split_by_chemical(
    records: List[Dict[str, Any]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records so that each chemical appears in only one split.

    This tests the model's ability to generalise to unseen chemicals,
    which is the realistic deployment scenario.

    Parameters
    ----------
    records : list of dict
    train_frac, val_frac, test_frac : float
        Must sum to 1.0 (approximately).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_records, val_records, test_records : tuple of lists
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
    )

    rng = np.random.RandomState(seed)

    # Group records by CAS number
    cas_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        cas = rec.get("cas_number", "unknown")
        cas_to_records.setdefault(cas, []).append(rec)

    # Shuffle chemicals
    chemicals = list(cas_to_records.keys())
    rng.shuffle(chemicals)

    # Allocate chemicals to splits to approximate target fractions
    total = len(records)
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    train_target = int(total * train_frac)
    val_target = int(total * val_frac)

    for cas in chemicals:
        recs = cas_to_records[cas]
        if len(train_records) < train_target:
            train_records.extend(recs)
        elif len(val_records) < val_target:
            val_records.extend(recs)
        else:
            test_records.extend(recs)

    logger.info(
        f"Split: {len(train_records)} train, {len(val_records)} val, "
        f"{len(test_records)} test "
        f"({len(cas_to_records)} unique chemicals)"
    )

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocabularies(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """Build chemical and species vocabularies from the training records.

    Parameters
    ----------
    records : list of dict

    Returns
    -------
    chemical_vocab : Dict[str, int]
        CAS → index (1-based; 0 is reserved for unknown).
    species_vocab : Dict[str, Dict[str, int]]
        Per-rank name → index (1-based; 0 for unknown).
    """
    # Chemical vocabulary
    cas_set: set = set()
    for rec in records:
        cas = rec.get("cas_number", "")
        if cas:
            cas_set.add(cas)
    chemical_vocab = {cas: idx + 1 for idx, cas in enumerate(sorted(cas_set))}

    # Species vocabulary (per rank)
    species_vocab: Dict[str, Dict[str, int]] = {}
    for rank in TAXONOMIC_RANKS:
        col = "class_" if rank == "class" else rank
        if rank == "species":
            col = "species_epithet"
        names: set = set()
        for rec in records:
            name = rec.get(col, "")
            if name:
                names.add(name)
        species_vocab[rank] = {
            name: idx + 1 for idx, name in enumerate(sorted(names))
        }

    logger.info(
        f"Built vocabularies: {len(chemical_vocab)} chemicals, "
        + ", ".join(
            f"{len(species_vocab[r])} {r}" for r in TAXONOMIC_RANKS
        )
    )

    return chemical_vocab, species_vocab


def compute_descriptor_stats(
    records: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of molecular descriptors for standardisation.

    Parameters
    ----------
    records : list of dict

    Returns
    -------
    means, stds : np.ndarray  (each of shape [NUM_MOLECULAR_DESCRIPTORS])
    """
    all_desc = []
    for rec in records:
        if "molecular_descriptors" in rec:
            raw = rec["molecular_descriptors"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            desc = list(raw)
        else:
            desc = [rec.get(col, 0.0) for col in DESCRIPTOR_COLUMNS]

        # Pad / truncate
        desc = desc[:NUM_MOLECULAR_DESCRIPTORS]
        while len(desc) < NUM_MOLECULAR_DESCRIPTORS:
            desc.append(0.0)

        all_desc.append(desc)

    arr = np.array(all_desc, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)

    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)

    return means, stds
