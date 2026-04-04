"""
ToxiGene molecular encoder training pipeline for SENTINEL.

Four-phase training:
  Phase 1: Data preparation — load hierarchy graph (sparse adjacency from
           build_hierarchy_graph output), expression data, pathway labels.
  Phase 2: Hierarchy training — train gene expression -> pathway -> outcome
           prediction with multi-task BCE loss, split by chemical.
  Phase 3: Cross-species transfer — train on zebrafish, evaluate transfer
           to data-poor species (Daphnia) via ortholog alignment.
  Phase 4: Information bottleneck sweep — sweep L1 lambda to find minimal
           biomarker gene panel preserving classification accuracy.

Usage:
    python -m sentinel.training.train_molecular --phase 1 --data-dir data/molecular
    python -m sentinel.training.train_molecular --phase 2 --data-dir data/molecular/prepared
    python -m sentinel.training.train_molecular --phase 3 --data-dir data/molecular/prepared
    python -m sentinel.training.train_molecular --phase 4 --data-dir data/molecular/prepared --checkpoint outputs/molecular/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from sentinel.models.molecular_encoder.model import (
    NATIVE_DIM,
    SHARED_EMBED_DIM,
    MolecularEncoder,
)
from sentinel.models.molecular_encoder.cross_species import CrossSpeciesEncoder
from sentinel.models.molecular_encoder.bottleneck import HierarchyBottleneck
from sentinel.training.trainer import (
    BaseTrainer,
    EarlyStopping,
    MetricTracker,
    TrainerConfig,
    build_scheduler,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ToxiGeneDataset(Dataset):
    """Gene expression dataset for ToxiGene hierarchy training.

    Loads pre-computed outputs from prepare_toxigene_inputs:
      - Expression matrix .npy (n_genes, n_samples) or (n_samples, n_genes)
      - Hierarchy adjacency .npz files
      - Pathway labels, outcome labels
      - Chemical metadata for split-by-chemical

    Args:
        data_dir: Directory containing prepared ToxiGene inputs.
        split_indices: Optional subset indices.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.data_dir = Path(data_dir)

        # Load expression matrix
        expr_path = self.data_dir / "expression_matrix.npy"
        self.expression = np.load(expr_path).astype(np.float32)

        # Load metadata
        meta_path = self.data_dir / "expression_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.gene_names: List[str] = self.metadata.get("gene_names", [])
        self.sample_names: List[str] = self.metadata.get("sample_names", [])

        # Determine orientation: (n_genes, n_samples) -> transpose to (n_samples, n_genes)
        shape = self.metadata.get("shape", list(self.expression.shape))
        if shape[0] == len(self.gene_names) and shape[1] == len(self.sample_names):
            # Stored as (n_genes, n_samples), need (n_samples, n_genes)
            self.expression = self.expression.T

        # Load outcome labels
        outcome_path = self.data_dir / "outcome_labels.npy"
        if outcome_path.exists():
            self.outcome_labels = np.load(outcome_path).astype(np.float32)
        else:
            # Try JSON
            label_path = self.data_dir / "outcome_labels.json"
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    label_data = json.load(f)
                self.outcome_labels = np.array(
                    label_data.get("labels", []), dtype=np.float32
                )
            else:
                logger.warning("No outcome labels found; using zeros")
                self.outcome_labels = np.zeros(
                    (self.expression.shape[0], 7), dtype=np.float32
                )

        # Load pathway labels (optional)
        pw_path = self.data_dir / "pathway_labels.npy"
        self.pathway_labels: Optional[np.ndarray] = None
        if pw_path.exists():
            self.pathway_labels = np.load(pw_path).astype(np.float32)

        # Chemical IDs for split-by-chemical
        chem_path = self.data_dir / "chemical_ids.json"
        if chem_path.exists():
            with open(chem_path, "r", encoding="utf-8") as f:
                chem_data = json.load(f)
            self.chemical_ids = np.array(chem_data.get("chemical_ids", []))
        else:
            self.chemical_ids = np.arange(self.expression.shape[0])

        # Species labels (for cross-species training)
        species_path = self.data_dir / "species_labels.json"
        if species_path.exists():
            with open(species_path, "r", encoding="utf-8") as f:
                species_data = json.load(f)
            self.species_labels = np.array(species_data.get("species", []))
        else:
            self.species_labels = np.array(
                ["unknown"] * self.expression.shape[0]
            )

        # Subset
        if split_indices is not None:
            self.expression = self.expression[split_indices]
            self.outcome_labels = self.outcome_labels[split_indices]
            if self.pathway_labels is not None:
                self.pathway_labels = self.pathway_labels[split_indices]
            self.chemical_ids = self.chemical_ids[split_indices]
            self.species_labels = self.species_labels[split_indices]

        logger.info(
            f"ToxiGeneDataset: {self.expression.shape[0]} samples, "
            f"{self.expression.shape[1]} genes, "
            f"{self.outcome_labels.shape[1]} outcomes"
        )

    def __len__(self) -> int:
        return self.expression.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {
            "gene_expression": torch.from_numpy(self.expression[idx]),
            "outcome_labels": torch.from_numpy(self.outcome_labels[idx]),
        }

        if self.pathway_labels is not None:
            result["pathway_labels"] = torch.from_numpy(self.pathway_labels[idx])

        return result


class CrossSpeciesDataset(Dataset):
    """Multi-species gene expression dataset for cross-species transfer.

    Wraps a ToxiGeneDataset and provides species-specific expression data
    with species identifiers for the CrossSpeciesEncoder.

    Args:
        data_dir: Directory containing prepared ToxiGene inputs.
        species_filter: Only include samples from this species (or None for all).
        split_indices: Optional subset indices.
    """

    def __init__(
        self,
        data_dir: str | Path,
        species_filter: Optional[str] = None,
        split_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.base_dataset = ToxiGeneDataset(data_dir, split_indices=split_indices)

        # Filter by species if requested
        if species_filter is not None:
            mask = self.base_dataset.species_labels == species_filter
            filtered_indices = np.where(mask)[0]
            if len(filtered_indices) == 0:
                logger.warning(
                    f"No samples for species '{species_filter}'. "
                    f"Available: {np.unique(self.base_dataset.species_labels)}"
                )
            self.base_dataset = ToxiGeneDataset(
                data_dir, split_indices=filtered_indices
            )

        self.species_labels = self.base_dataset.species_labels

        # Load per-species gene name mappings
        mapping_path = self.data_dir / "species_gene_mappings.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.species_gene_mappings = json.load(f)
        else:
            self.species_gene_mappings = {}

        logger.info(
            f"CrossSpeciesDataset: {len(self.base_dataset)} samples"
            + (f" (species={species_filter})" if species_filter else "")
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base_dataset[idx]
        item["species"] = self.species_labels[idx]
        return item


# ---------------------------------------------------------------------------
# Adjacency loading utilities
# ---------------------------------------------------------------------------

def load_hierarchy_adjacency(
    data_dir: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Load sparse adjacency matrices from .npz files.

    Returns:
        pathway_adj: Gene-to-pathway adjacency [n_pathways, n_genes].
        process_adj: Pathway-to-process adjacency [n_processes, n_pathways].
        outcome_adj: Process-to-outcome adjacency [n_outcomes, n_processes].
        gene_names: List of gene names.
    """
    import scipy.sparse

    data_dir = Path(data_dir)

    # Load metadata for gene names
    meta_path = data_dir / "expression_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    gene_names = metadata.get("gene_names", [])

    # Load hierarchy metadata
    hier_meta_path = data_dir / "hierarchy_metadata.json"
    if hier_meta_path.exists():
        with open(hier_meta_path, "r", encoding="utf-8") as f:
            hier_meta = json.load(f)
    else:
        hier_meta = {}

    # Find and load adjacency .npz files
    npz_files = sorted(data_dir.glob("hierarchy_layer*.npz"))

    if len(npz_files) >= 3:
        pathway_adj_sp = scipy.sparse.load_npz(npz_files[0])
        process_adj_sp = scipy.sparse.load_npz(npz_files[1])
        outcome_adj_sp = scipy.sparse.load_npz(npz_files[2])
    else:
        # Construct default adjacency if files missing
        n_genes = len(gene_names) or 200
        n_pathways = hier_meta.get("n_pathways", 7)
        n_processes = hier_meta.get("n_processes", 5)
        n_outcomes = hier_meta.get("n_outcomes", 4)

        logger.warning(
            f"Fewer than 3 hierarchy .npz files found ({len(npz_files)}). "
            f"Constructing identity-like adjacency matrices."
        )
        pathway_adj_sp = scipy.sparse.eye(n_pathways, n_genes, format="csr")
        process_adj_sp = scipy.sparse.eye(n_processes, n_pathways, format="csr")
        outcome_adj_sp = scipy.sparse.eye(n_outcomes, n_processes, format="csr")

    pathway_adj = torch.from_numpy(pathway_adj_sp.toarray()).float()
    process_adj = torch.from_numpy(process_adj_sp.toarray()).float()
    outcome_adj = torch.from_numpy(outcome_adj_sp.toarray()).float()

    logger.info(
        f"Hierarchy adjacency: genes={pathway_adj.shape[1]}, "
        f"pathways={pathway_adj.shape[0]}, "
        f"processes={process_adj.shape[0]}, "
        f"outcomes={outcome_adj.shape[0]}"
    )

    return pathway_adj, process_adj, outcome_adj, gene_names


def split_by_chemical(
    dataset: ToxiGeneDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataset by chemical, not by sample.

    Ensures all samples from the same chemical stay together, preventing
    chemical identity leakage between train and validation.

    Returns:
        (train_indices, val_indices)
    """
    rng = np.random.RandomState(seed)
    unique_chems = np.unique(dataset.chemical_ids)
    rng.shuffle(unique_chems)

    n_val_chems = max(1, int(len(unique_chems) * val_fraction))
    val_chems = set(unique_chems[:n_val_chems].tolist())

    val_mask = np.array([cid in val_chems for cid in dataset.chemical_ids])
    train_indices = np.where(~val_mask)[0]
    val_indices = np.where(val_mask)[0]

    logger.info(
        f"Split by chemical: {len(unique_chems)} chemicals -> "
        f"train={len(train_indices)} samples ({len(unique_chems) - n_val_chems} chems), "
        f"val={len(val_indices)} samples ({n_val_chems} chems)"
    )

    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Phase 2: Hierarchy Trainer
# ---------------------------------------------------------------------------

@dataclass
class HierarchyConfig(TrainerConfig):
    """Config for hierarchy training (Phase 2)."""

    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    patience: int = 15
    wandb_run_name: str = "molecular-hierarchy"

    # Data
    data_dir: str = "data/molecular/prepared"
    val_fraction: float = 0.2

    # Model
    num_chem_classes: int = 50
    lambda_l1: float = 0.01
    dropout: float = 0.2

    # Loss weights
    outcome_weight: float = 1.0
    pathway_weight: float = 0.5
    bottleneck_weight: float = 1.0


class HierarchyTrainer(BaseTrainer):
    """Phase 2: ToxiGene hierarchy training.

    Trains gene expression -> pathway -> outcome prediction with
    multi-task BCE loss. Data is split by chemical (not sample).
    """

    def __init__(
        self,
        config: HierarchyConfig,
        pathway_adj: torch.Tensor,
        process_adj: torch.Tensor,
        outcome_adj: torch.Tensor,
        gene_names: List[str],
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(config)
        self.hier_config = config
        self.pathway_adj = pathway_adj
        self.process_adj = process_adj
        self.outcome_adj = outcome_adj
        self.gene_names = gene_names
        self.train_indices = train_indices
        self.val_indices = val_indices

    def build_model(self) -> nn.Module:
        model = MolecularEncoder(
            gene_names=self.gene_names,
            pathway_adj=self.pathway_adj,
            process_adj=self.process_adj,
            outcome_adj=self.outcome_adj,
            num_chem_classes=self.hier_config.num_chem_classes,
            lambda_l1=self.hier_config.lambda_l1,
            dropout=self.hier_config.dropout,
        )
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        if self.train_indices is not None and self.val_indices is not None:
            train_ds = ToxiGeneDataset(
                self.hier_config.data_dir, split_indices=self.train_indices
            )
            val_ds = ToxiGeneDataset(
                self.hier_config.data_dir, split_indices=self.val_indices
            )
        else:
            full_ds = ToxiGeneDataset(self.hier_config.data_dir)
            train_idx, val_idx = split_by_chemical(
                full_ds, val_fraction=self.hier_config.val_fraction
            )
            train_ds = ToxiGeneDataset(
                self.hier_config.data_dir, split_indices=train_idx
            )
            val_ds = ToxiGeneDataset(
                self.hier_config.data_dir, split_indices=val_idx
            )

        logger.info(f"Hierarchy training: train={len(train_ds)}, val={len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        gene_expr = batch["gene_expression"]
        outcome_targets = batch["outcome_labels"]
        pathway_targets = batch.get("pathway_labels", None)

        # Forward
        outputs = self.model(gene_expression=gene_expr)

        # Compute multi-task loss
        losses = self.model.compute_loss(
            outputs=outputs,
            outcome_targets=outcome_targets,
            pathway_targets=pathway_targets,
            outcome_weight=self.hier_config.outcome_weight,
            pathway_weight=self.hier_config.pathway_weight,
            bottleneck_weight=self.hier_config.bottleneck_weight,
        )

        losses["total"].backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        result = {k: v.item() if isinstance(v, torch.Tensor) else float(v)
                  for k, v in losses.items()}
        result["loss"] = result["total"]
        return result

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        total_outcome_loss = 0.0
        correct_outcomes = 0
        total_outcomes = 0
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            gene_expr = batch["gene_expression"]
            outcome_targets = batch["outcome_labels"]
            pathway_targets = batch.get("pathway_labels", None)

            outputs = self.model(gene_expression=gene_expr)
            losses = self.model.compute_loss(
                outputs=outputs,
                outcome_targets=outcome_targets,
                pathway_targets=pathway_targets,
                outcome_weight=self.hier_config.outcome_weight,
                pathway_weight=self.hier_config.pathway_weight,
                bottleneck_weight=self.hier_config.bottleneck_weight,
            )

            total_loss += losses["total"].item()
            total_outcome_loss += losses["outcome"].item()

            # Per-outcome binary accuracy (threshold at 0.5)
            preds = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float()
            correct_outcomes += (preds == outcome_targets).sum().item()
            total_outcomes += outcome_targets.numel()
            n_batches += 1

        self.model.train()
        n = max(n_batches, 1)

        return {
            "loss": total_loss / n,
            "outcome_loss": total_outcome_loss / n,
            "outcome_accuracy": correct_outcomes / max(total_outcomes, 1),
            "num_selected_genes": float(self.model.bottleneck.num_selected),
        }


# ---------------------------------------------------------------------------
# Phase 3: Cross-species transfer
# ---------------------------------------------------------------------------

@dataclass
class CrossSpeciesConfig(TrainerConfig):
    """Config for cross-species transfer training (Phase 3)."""

    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    patience: int = 15
    wandb_run_name: str = "molecular-cross-species"

    # Data
    data_dir: str = "data/molecular/prepared"
    source_species: str = "danio_rerio"  # zebrafish (data-rich)
    target_species: str = "daphnia_magna"  # Daphnia (data-poor)
    val_fraction: float = 0.2

    # Model
    checkpoint_path: str = ""  # Phase 2 checkpoint
    dropout: float = 0.2


def run_phase3(
    data_dir: str,
    output_dir: str,
    checkpoint_path: str = "",
    source_species: str = "danio_rerio",
    target_species: str = "daphnia_magna",
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 42,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """Phase 3: Cross-species transfer training.

    Train on data-rich species (zebrafish), evaluate transfer to data-poor
    species (Daphnia) using ortholog alignment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # Load hierarchy
    pathway_adj, process_adj, outcome_adj, gene_names = load_hierarchy_adjacency(
        data_dir
    )

    # Load ortholog mappings
    data_dir_path = Path(data_dir)
    ortholog_path = data_dir_path / "ortholog_mappings.json"
    if ortholog_path.exists():
        with open(ortholog_path, "r", encoding="utf-8") as f:
            ortholog_data = json.load(f)
    else:
        logger.warning(f"Ortholog mappings not found at {ortholog_path}")
        ortholog_data = {}

    ortholog_gene_names = ortholog_data.get("ortholog_gene_names", gene_names)
    n_orthologs = len(ortholog_gene_names)

    # Build ortholog-space adjacency if needed
    # If ortholog space differs from gene space, adjust pathway_adj
    if n_orthologs != len(gene_names):
        logger.info(
            f"Ortholog space ({n_orthologs}) differs from gene space "
            f"({len(gene_names)}). Adjusting adjacency."
        )
        # Truncate or pad pathway_adj columns to match ortholog space
        pw_adj_np = pathway_adj.numpy()
        if pw_adj_np.shape[1] > n_orthologs:
            pw_adj_np = pw_adj_np[:, :n_orthologs]
        elif pw_adj_np.shape[1] < n_orthologs:
            pw_adj_np = np.pad(
                pw_adj_np, ((0, 0), (0, n_orthologs - pw_adj_np.shape[1]))
            )
        pathway_adj = torch.from_numpy(pw_adj_np).float()

    # Build CrossSpeciesEncoder
    cross_encoder = CrossSpeciesEncoder(
        ortholog_gene_names=ortholog_gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
        dropout=0.2,
    )

    # Register species with their ortholog mappings
    for species_name, species_info in ortholog_data.get("species", {}).items():
        n_species_genes = species_info.get("n_genes", n_orthologs)
        mapping = np.array(
            species_info.get("mapping_matrix", np.eye(n_orthologs, n_species_genes).tolist()),
            dtype=np.float32,
        )
        mapping_tensor = torch.from_numpy(mapping)
        cross_encoder.register_species(species_name, n_species_genes, mapping_tensor)
        logger.info(f"Registered species: {species_name} ({n_species_genes} genes)")

    # Ensure source and target are registered
    for sp in [source_species, target_species]:
        if sp not in cross_encoder.registered_species:
            logger.warning(
                f"Species '{sp}' not in ortholog mappings. "
                f"Registering with identity mapping."
            )
            cross_encoder.register_species(
                sp, n_orthologs, torch.eye(n_orthologs)
            )

    cross_encoder.to(dev)

    # Load Phase 2 checkpoint into the hierarchy network if available
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_state = state.get("model_state_dict", state)
            # Extract hierarchy weights from MolecularEncoder checkpoint
            hierarchy_state = {}
            for k, v in model_state.items():
                if k.startswith("hierarchy."):
                    hierarchy_state[k[len("hierarchy."):]] = v
            if hierarchy_state:
                cross_encoder.hierarchy.load_state_dict(
                    hierarchy_state, strict=False
                )
                logger.info(f"Loaded hierarchy weights from Phase 2 checkpoint")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")

    # Load datasets
    source_ds = CrossSpeciesDataset(data_dir, species_filter=source_species)
    target_ds = CrossSpeciesDataset(data_dir, species_filter=target_species)

    logger.info(f"Source species ({source_species}): {len(source_ds)} samples")
    logger.info(f"Target species ({target_species}): {len(target_ds)} samples")

    # Split source into train/val
    n_source = len(source_ds)
    n_val = max(1, int(n_source * 0.2))
    indices = list(range(n_source))
    random.seed(seed)
    random.shuffle(indices)

    source_train = Subset(source_ds, indices[n_val:])
    source_val = Subset(source_ds, indices[:n_val])

    source_train_loader = DataLoader(
        source_train, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    source_val_loader = DataLoader(
        source_val, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    target_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        cross_encoder.parameters(), lr=lr, weight_decay=0.01
    )
    total_steps = len(source_train_loader) * epochs
    scheduler = build_scheduler("cosine", optimizer, total_steps)

    early_stop = EarlyStopping(patience=15, mode="min")
    best_val_loss = float("inf")
    metrics_history: Dict[str, List[float]] = defaultdict(list)

    # Training loop
    for epoch in range(epochs):
        cross_encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in source_train_loader:
            gene_expr = batch["gene_expression"].to(dev)
            outcome_targets = batch["outcome_labels"].to(dev)
            species_list = batch["species"]

            # Forward through cross-species encoder
            # Process per-species in the batch
            all_features = []
            all_pathway_act = []
            all_outcome_logits = []

            for i in range(gene_expr.shape[0]):
                sp = species_list[i] if isinstance(species_list, list) else species_list
                features_i, pw_i, outcome_i = cross_encoder(
                    gene_expr[i:i+1], species=sp
                )
                all_features.append(features_i)
                all_pathway_act.append(pw_i)
                all_outcome_logits.append(outcome_i)

            outcome_logits = torch.cat(all_outcome_logits, dim=0)

            # Multi-task BCE loss
            outcome_loss = F.binary_cross_entropy_with_logits(
                outcome_logits, outcome_targets, reduction="mean"
            )

            optimizer.zero_grad(set_to_none=True)
            outcome_loss.backward()
            nn.utils.clip_grad_norm_(cross_encoder.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += outcome_loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        metrics_history["train_loss"].append(avg_train_loss)

        # Validate on source species
        source_val_loss, source_val_acc = _evaluate_cross_species(
            cross_encoder, source_val_loader, dev, source_species
        )
        metrics_history["source_val_loss"].append(source_val_loss)
        metrics_history["source_val_acc"].append(source_val_acc)

        # Evaluate on target species (zero-shot transfer)
        target_loss, target_acc = _evaluate_cross_species(
            cross_encoder, target_loader, dev, target_species
        )
        metrics_history["target_loss"].append(target_loss)
        metrics_history["target_acc"].append(target_acc)

        logger.info(
            f"Epoch {epoch:3d} | train_loss={avg_train_loss:.4f} | "
            f"source_val: loss={source_val_loss:.4f} acc={source_val_acc:.4f} | "
            f"target: loss={target_loss:.4f} acc={target_acc:.4f}"
        )

        # Checkpoint best
        if source_val_loss < best_val_loss:
            best_val_loss = source_val_loss
            ckpt_dir = output_path / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": cross_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_dir / "best_model.pt",
            )

        if early_stop.step(source_val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final summary
    summary = {
        "source_species": source_species,
        "target_species": target_species,
        "source_samples": len(source_ds),
        "target_samples": len(target_ds),
        "best_source_val_loss": best_val_loss,
        "final_target_accuracy": metrics_history["target_acc"][-1],
        "final_source_accuracy": metrics_history["source_val_acc"][-1],
        "total_epochs": len(metrics_history["train_loss"]),
    }

    summary_path = output_path / "cross_species_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nCross-species transfer results:")
    logger.info(f"  Source ({source_species}) accuracy: {summary['final_source_accuracy']:.4f}")
    logger.info(f"  Target ({target_species}) accuracy: {summary['final_target_accuracy']:.4f}")
    logger.info(f"  Summary saved to {summary_path}")

    return summary


@torch.no_grad()
def _evaluate_cross_species(
    model: CrossSpeciesEncoder,
    dataloader: DataLoader,
    device: torch.device,
    species: str,
) -> Tuple[float, float]:
    """Evaluate cross-species model on a single-species dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for batch in dataloader:
        gene_expr = batch["gene_expression"].to(device)
        outcome_targets = batch["outcome_labels"].to(device)

        # Batch forward (all same species)
        features, pw_act, outcome_logits = model(gene_expr, species=species)

        loss = F.binary_cross_entropy_with_logits(
            outcome_logits, outcome_targets, reduction="mean"
        )
        total_loss += loss.item()

        preds = (torch.sigmoid(outcome_logits) > 0.5).float()
        correct += (preds == outcome_targets).sum().item()
        total += outcome_targets.numel()
        n_batches += 1

    model.train()
    return (
        total_loss / max(n_batches, 1),
        correct / max(total, 1),
    )


# ---------------------------------------------------------------------------
# Phase 4: Information Bottleneck Sweep
# ---------------------------------------------------------------------------

def run_phase4(
    data_dir: str,
    output_dir: str,
    checkpoint_path: str = "",
    n_lambda_steps: int = 20,
    lambda_max: float = 0.1,
    epochs_per_step: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 42,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """Phase 4: L1 lambda sweep for minimal biomarker panel discovery.

    Sweeps L1 lambda from 0 to lambda_max in n_lambda_steps. At each lambda,
    trains the bottleneck and records (gene_count, accuracy). Finds the
    elbow point for the minimal biomarker panel.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # Load hierarchy
    pathway_adj, process_adj, outcome_adj, gene_names = load_hierarchy_adjacency(
        data_dir
    )

    # Load dataset and split by chemical
    full_ds = ToxiGeneDataset(data_dir)
    train_idx, val_idx = split_by_chemical(full_ds, seed=seed)

    # Lambda sweep
    lambdas = np.linspace(0.0, lambda_max, n_lambda_steps)
    sweep_results: List[Dict[str, Any]] = []

    for step_idx, lam in enumerate(lambdas):
        logger.info(f"\n{'='*60}")
        logger.info(f"BOTTLENECK SWEEP: step {step_idx + 1}/{n_lambda_steps}, lambda={lam:.5f}")
        logger.info(f"{'='*60}")

        # Build model with this lambda
        model = MolecularEncoder(
            gene_names=gene_names,
            pathway_adj=pathway_adj.clone(),
            process_adj=process_adj.clone(),
            outcome_adj=outcome_adj.clone(),
            lambda_l1=float(lam),
        )

        # Load base checkpoint if available
        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model_state_dict", state)
                # Load all except bottleneck (we want fresh gates for each lambda)
                filtered_state = {
                    k: v for k, v in model_state.items()
                    if not k.startswith("bottleneck.")
                }
                model.load_state_dict(filtered_state, strict=False)

        model.to(dev)

        # Build data loaders
        train_ds = ToxiGeneDataset(data_dir, split_indices=train_idx)
        val_ds = ToxiGeneDataset(data_dir, split_indices=val_idx)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False,
        )

        # Optimizer — only train bottleneck gates and hierarchy
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        early_stop = EarlyStopping(patience=10, mode="min")
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_n_genes = len(gene_names)

        for epoch in range(epochs_per_step):
            model.train()
            epoch_loss = 0.0
            n_train_batches = 0

            for batch in train_loader:
                gene_expr = batch["gene_expression"].to(dev)
                outcome_targets = batch["outcome_labels"].to(dev)
                pathway_targets = batch.get("pathway_labels")
                if pathway_targets is not None:
                    pathway_targets = pathway_targets.to(dev)

                outputs = model(gene_expression=gene_expr)
                losses = model.compute_loss(
                    outputs=outputs,
                    outcome_targets=outcome_targets,
                    pathway_targets=pathway_targets,
                    bottleneck_weight=1.0,
                )

                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += losses["total"].item()
                n_train_batches += 1

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    gene_expr = batch["gene_expression"].to(dev)
                    outcome_targets = batch["outcome_labels"].to(dev)

                    outputs = model(gene_expression=gene_expr)
                    losses = model.compute_loss(
                        outputs=outputs,
                        outcome_targets=outcome_targets,
                        bottleneck_weight=1.0,
                    )

                    val_loss += losses["total"].item()
                    preds = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float()
                    val_correct += (preds == outcome_targets).sum().item()
                    val_total += outcome_targets.numel()
                    n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)
            val_acc = val_correct / max(val_total, 1)
            n_genes = model.bottleneck.num_selected

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_n_genes = n_genes

            if early_stop.step(avg_val_loss):
                break

        step_result = {
            "lambda_l1": float(lam),
            "gene_count": best_n_genes,
            "accuracy": best_val_acc,
            "val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "selected_genes": model.bottleneck.get_selected_genes(),
        }
        sweep_results.append(step_result)

        logger.info(
            f"  lambda={lam:.5f} -> {best_n_genes} genes, "
            f"accuracy={best_val_acc:.4f}, val_loss={best_val_loss:.4f}"
        )

    # Find elbow point
    gene_counts = np.array([r["gene_count"] for r in sweep_results])
    accuracies = np.array([r["accuracy"] for r in sweep_results])

    elbow_idx = _find_elbow(gene_counts, accuracies)
    elbow_result = sweep_results[elbow_idx]

    logger.info(f"\nElbow point found at step {elbow_idx}:")
    logger.info(f"  lambda={elbow_result['lambda_l1']:.5f}")
    logger.info(f"  gene_count={elbow_result['gene_count']}")
    logger.info(f"  accuracy={elbow_result['accuracy']:.4f}")
    logger.info(f"  selected genes: {elbow_result['selected_genes'][:20]}...")

    # Save results
    # Remove non-serializable gene lists for the main sweep data
    sweep_data = []
    for r in sweep_results:
        sweep_data.append({
            "lambda_l1": r["lambda_l1"],
            "gene_count": r["gene_count"],
            "accuracy": r["accuracy"],
            "val_loss": r["val_loss"],
            "epochs_trained": r["epochs_trained"],
        })

    summary = {
        "sweep_results": sweep_data,
        "elbow_index": elbow_idx,
        "elbow_lambda": elbow_result["lambda_l1"],
        "elbow_gene_count": elbow_result["gene_count"],
        "elbow_accuracy": elbow_result["accuracy"],
        "minimal_biomarker_panel": elbow_result["selected_genes"],
        "n_lambda_steps": n_lambda_steps,
        "lambda_range": [0.0, lambda_max],
    }

    summary_path = output_path / "bottleneck_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Bottleneck sweep saved to {summary_path}")

    return summary


def _find_elbow(gene_counts: np.ndarray, accuracies: np.ndarray) -> int:
    """Find elbow point in the gene_count vs accuracy curve.

    Uses the maximum distance from the line connecting the first and last
    points of the (gene_count, accuracy) curve.

    Returns:
        Index of the elbow point.
    """
    if len(gene_counts) < 3:
        return 0

    # Normalize to [0,1] for distance computation
    gc_norm = (gene_counts - gene_counts.min()) / max(gene_counts.max() - gene_counts.min(), 1e-10)
    ac_norm = (accuracies - accuracies.min()) / max(accuracies.max() - accuracies.min(), 1e-10)

    # Line from first to last point
    p1 = np.array([gc_norm[0], ac_norm[0]])
    p2 = np.array([gc_norm[-1], ac_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return len(gene_counts) // 2

    line_unit = line_vec / line_len

    # Distance of each point from the line
    distances = np.zeros(len(gene_counts))
    for i in range(len(gene_counts)):
        point = np.array([gc_norm[i], ac_norm[i]])
        vec = point - p1
        proj = np.dot(vec, line_unit)
        proj_point = p1 + proj * line_unit
        distances[i] = np.linalg.norm(point - proj_point)

    return int(np.argmax(distances))


# ---------------------------------------------------------------------------
# Phase 1: Data preparation
# ---------------------------------------------------------------------------

def run_phase1(
    data_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
) -> Dict[str, str]:
    """Phase 1: Prepare ToxiGene inputs from raw data.

    Expects data_dir to contain:
      - Expression data (parquet or CSV)
      - Hierarchy data (Reactome + AOP-Wiki directories)
      - Ortholog mappings JSON

    Produces: expression matrix, hierarchy adjacency .npz, labels.
    """
    data_dir_path = Path(data_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Find input directories
    expression_dir = data_dir_path / "expression"
    if not expression_dir.exists():
        expression_dir = data_dir_path

    hierarchy_dir = data_dir_path / "hierarchy"
    if not hierarchy_dir.exists():
        hierarchy_dir = data_dir_path

    ortholog_path = data_dir_path / "ortholog_mappings.json"

    logger.info(f"Expression dir: {expression_dir}")
    logger.info(f"Hierarchy dir: {hierarchy_dir}")
    logger.info(f"Ortholog path: {ortholog_path}")

    from sentinel.data.molecular.preprocessing import prepare_toxigene_inputs

    result = prepare_toxigene_inputs(
        expression_dir=str(expression_dir),
        hierarchy_dir=str(hierarchy_dir),
        ortholog_path=str(ortholog_path),
        output_dir=str(output_dir_path),
    )

    logger.info("Phase 1 complete. Outputs:")
    for key, path in result.items():
        logger.info(f"  {key}: {path}")

    return result


# ---------------------------------------------------------------------------
# Phase 2: Hierarchy training
# ---------------------------------------------------------------------------

def run_phase2(
    data_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 42,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """Phase 2: Train ToxiGene hierarchy network."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load hierarchy adjacency
    pathway_adj, process_adj, outcome_adj, gene_names = load_hierarchy_adjacency(
        data_dir
    )

    # Split by chemical
    full_ds = ToxiGeneDataset(data_dir)
    train_idx, val_idx = split_by_chemical(full_ds, seed=seed)

    config = HierarchyConfig(
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        output_dir=str(output_path),
        device=device,
        seed=seed,
        use_wandb=use_wandb,
        data_dir=data_dir,
    )

    trainer = HierarchyTrainer(
        config,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
        gene_names=gene_names,
        train_indices=train_idx,
        val_indices=val_idx,
    )
    trainer.setup()
    result = trainer.train()

    logger.info(
        f"Phase 2 complete. Best val loss: {result['best_val_metric']:.4f} "
        f"at epoch {result['best_epoch']}"
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SENTINEL ToxiGene molecular encoder training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Training phase (1=data prep, 2=hierarchy, "
        "3=cross-species, 4=bottleneck sweep)",
    )
    parser.add_argument("--data-dir", type=str, default="data/molecular",
                        help="Input data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/molecular",
                        help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-wandb", action="store_true")

    # Phase 3 specific
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to model checkpoint (Phase 3/4)")
    parser.add_argument("--source-species", type=str, default="danio_rerio",
                        help="Source (data-rich) species for transfer (Phase 3)")
    parser.add_argument("--target-species", type=str, default="daphnia_magna",
                        help="Target (data-poor) species for transfer (Phase 3)")

    # Phase 4 specific
    parser.add_argument("--n-lambda-steps", type=int, default=20,
                        help="Number of L1 lambda steps to sweep (Phase 4)")
    parser.add_argument("--lambda-max", type=float, default=0.1,
                        help="Maximum L1 lambda value (Phase 4)")
    parser.add_argument("--epochs-per-step", type=int, default=50,
                        help="Epochs per lambda step (Phase 4)")

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.phase == 1:
        run_phase1(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
        )

    elif args.phase == 2:
        run_phase2(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )

    elif args.phase == 3:
        run_phase3(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            source_species=args.source_species,
            target_species=args.target_species,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )

    elif args.phase == 4:
        run_phase4(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            n_lambda_steps=args.n_lambda_steps,
            lambda_max=args.lambda_max,
            epochs_per_step=args.epochs_per_step,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )


if __name__ == "__main__":
    main()
