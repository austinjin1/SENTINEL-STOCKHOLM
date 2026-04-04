"""
Digital Biosentinel training pipeline for SENTINEL.

Trains the dose-response model on ECOTOX data with:
  - Class-balanced oversampling of rare effect types
  - Multi-endpoint loss (mortality, growth, reproduction, behavioral)
  - Post-training temperature-scaling calibration
  - ECE reporting and reliability diagram generation

Usage:
    python -m sentinel.training.train_biosentinel --data-dir data/ecotox/processed
    python -m sentinel.training.train_biosentinel --data-dir data/ecotox/processed --calibrate-only --checkpoint outputs/biosentinel/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sentinel.models.digital_biosentinel.calibration import (
    TemperatureScaler,
    expected_calibration_error,
)
from sentinel.models.digital_biosentinel.dataset import (
    ECOTOXDataset,
    build_balanced_sampler,
    build_vocabularies,
    compute_descriptor_stats,
    ecotox_collate_fn,
    split_by_chemical,
)
from sentinel.models.digital_biosentinel.dose_response import DoseResponseOutput
from sentinel.models.digital_biosentinel.model import DigitalBiosentinel
from sentinel.models.digital_biosentinel.species_encoder import TAXONOMIC_RANKS
from sentinel.training.trainer import (
    BaseTrainer,
    EarlyStopping,
    TrainerConfig,
    build_scheduler,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BiosentinelTrainConfig(TrainerConfig):
    """Configuration for Digital Biosentinel training."""

    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 100
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    early_stopping: bool = True
    patience: int = 15
    wandb_run_name: str = "biosentinel-dose-response"

    # Data
    data_dir: str = "data/ecotox/processed"
    data_format: str = "parquet"  # "parquet" or "csv"
    oversample_rare: bool = True
    oversample_endpoint: str = "multi"

    # Loss weights
    mortality_weight: float = 1.0
    growth_weight: float = 0.5
    reproduction_weight: float = 0.5
    behavioral_weight: float = 2.0  # upweight rare behavioral effects

    # Model
    chemical_embed_dim: int = 128
    species_embed_dim: int = 64
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.3
    mc_dropout_passes: int = 20

    # Calibration
    calibrate_after_training: bool = True
    calibration_n_bins: int = 15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ecotox_records(
    data_dir: Path,
    data_format: str = "parquet",
) -> List[Dict[str, Any]]:
    """Load processed ECOTOX records from disk."""
    if data_format == "parquet":
        try:
            import pandas as pd
            files = sorted(data_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {data_dir}")
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            records = df.to_dict("records")
        except ImportError:
            raise ImportError("pandas and pyarrow required for parquet format")
    elif data_format == "csv":
        import pandas as pd
        files = sorted(data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No .csv files in {data_dir}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        records = df.to_dict("records")
    elif data_format == "json":
        files = sorted(data_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No .json files in {data_dir}")
        records = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
    else:
        raise ValueError(f"Unknown data format: {data_format}")

    logger.info(f"Loaded {len(records)} ECOTOX records from {data_dir}")
    return records


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BiosentinelTrainer(BaseTrainer):
    """Trainer for the Digital Biosentinel dose-response model."""

    def __init__(self, config: BiosentinelTrainConfig) -> None:
        super().__init__(config)
        self.train_config = config
        self._chemical_vocab: Dict[str, int] = {}
        self._species_vocab: Dict[str, Dict[str, int]] = {}
        self._descriptor_means: Optional[np.ndarray] = None
        self._descriptor_stds: Optional[np.ndarray] = None
        self._train_sampler = None

    def build_model(self) -> nn.Module:
        # Build vocabulary sizes from species vocab
        species_vocab_sizes = {}
        for rank in TAXONOMIC_RANKS:
            species_vocab_sizes[rank] = len(self._species_vocab.get(rank, {})) + 1

        model = DigitalBiosentinel(
            num_chemicals=len(self._chemical_vocab) + 1,
            species_vocab_sizes=species_vocab_sizes,
            chemical_dim=self.train_config.chemical_embed_dim,
            species_dim=self.train_config.species_embed_dim,
            backbone_dims=self.train_config.hidden_dims,
            dropout=self.train_config.dropout,
            mc_samples=self.train_config.mc_dropout_passes,
        )
        model.register_chemical_vocab(self._chemical_vocab)
        model.register_species_vocab(self._species_vocab)
        return model

    def build_datasets(self) -> Tuple[Any, Any]:
        data_dir = Path(self.train_config.data_dir)
        records = load_ecotox_records(data_dir, self.train_config.data_format)

        # Split by chemical identity for proper generalization
        train_recs, val_recs, _test_recs = split_by_chemical(
            records, seed=self.config.seed,
        )

        # Build vocabularies from training set only
        self._chemical_vocab, self._species_vocab = build_vocabularies(train_recs)
        self._descriptor_means, self._descriptor_stds = compute_descriptor_stats(
            train_recs
        )

        train_ds = ECOTOXDataset(
            records=train_recs,
            chemical_vocab=self._chemical_vocab,
            species_vocab=self._species_vocab,
            augment=True,
            descriptor_means=self._descriptor_means,
            descriptor_stds=self._descriptor_stds,
        )
        val_ds = ECOTOXDataset(
            records=val_recs,
            chemical_vocab=self._chemical_vocab,
            species_vocab=self._species_vocab,
            augment=False,
            descriptor_means=self._descriptor_means,
            descriptor_stds=self._descriptor_stds,
        )

        # Build balanced sampler for oversampling rare endpoints
        if self.train_config.oversample_rare:
            self._train_sampler = build_balanced_sampler(
                train_ds, endpoint=self.train_config.oversample_endpoint,
            )

        return train_ds, val_ds

    def setup(self) -> None:
        """Override setup to build datasets first (need vocabs for model)."""
        train_ds, val_ds = self.build_datasets()

        self.model = self.build_model()
        self.model.to(self.device)

        # Build dataloaders with custom collate and optional balanced sampler
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            sampler=self._train_sampler,
            shuffle=self._train_sampler is None,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=ecotox_collate_fn,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=ecotox_collate_fn,
            drop_last=False,
        )

        self.optimizer = self.build_optimizer()
        total_steps = len(self.train_loader) * self.config.epochs
        warmup = self.config.warmup_steps or (
            self.config.warmup_epochs * len(self.train_loader)
        )
        self.scheduler = build_scheduler(
            self.config.scheduler, self.optimizer, total_steps, warmup,
        )

    def _compute_loss(
        self,
        dose_output: DoseResponseOutput,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-endpoint loss."""
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=self.device)

        cfg = self.train_config

        # Mortality: binary cross-entropy
        if "mortality" in targets:
            mask = ~torch.isnan(targets["mortality"])
            if mask.any():
                mort_loss = F.binary_cross_entropy(
                    dose_output.mortality[mask].clamp(1e-7, 1.0 - 1e-7),
                    targets["mortality"][mask],
                )
                losses["mortality_loss"] = mort_loss
                total = total + cfg.mortality_weight * mort_loss

        # Growth inhibition: MSE
        if "growth_inhibition" in targets:
            mask = ~torch.isnan(targets["growth_inhibition"])
            if mask.any():
                growth_loss = F.mse_loss(
                    dose_output.growth_inhibition[mask],
                    targets["growth_inhibition"][mask],
                )
                losses["growth_loss"] = growth_loss
                total = total + cfg.growth_weight * growth_loss

        # Reproduction effect: MSE
        if "reproduction_effect" in targets:
            mask = ~torch.isnan(targets["reproduction_effect"])
            if mask.any():
                repro_loss = F.mse_loss(
                    dose_output.reproduction_effect[mask],
                    targets["reproduction_effect"][mask],
                )
                losses["reproduction_loss"] = repro_loss
                total = total + cfg.reproduction_weight * repro_loss

        # Behavioral change: binary cross-entropy
        if "behavioral_change" in targets:
            mask = ~torch.isnan(targets["behavioral_change"])
            if mask.any():
                behav_loss = F.binary_cross_entropy(
                    dose_output.behavioral_change[mask].clamp(1e-7, 1.0 - 1e-7),
                    targets["behavioral_change"][mask],
                )
                losses["behavioral_loss"] = behav_loss
                total = total + cfg.behavioral_weight * behav_loss

        losses["loss"] = total
        return losses

    def train_step(self, batch: Any) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        dose_output, _chem_emb, _sp_emb = self.model(
            chemical_idx=batch["chemical_idx"],
            class_idx=batch["class_idx"],
            descriptors=batch["descriptors"],
            taxonomy_indices=batch["taxonomy_indices"],
            trophic_idx=batch["trophic_idx"],
            log_concentration=batch["log_concentration"],
            log_exposure_hours=batch["log_exposure_hours"],
            trophic_level_numeric=batch["trophic_level_numeric"],
        )

        losses = self._compute_loss(dose_output, batch["targets"])
        loss = losses["loss"]

        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {k: v.detach() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        all_losses: Dict[str, List[float]] = {}
        all_mort_logits: List[torch.Tensor] = []
        all_mort_labels: List[torch.Tensor] = []
        all_behav_logits: List[torch.Tensor] = []
        all_behav_labels: List[torch.Tensor] = []

        for batch in dataloader:
            batch = self._to_device(batch)

            dose_output, _, _ = self.model(
                chemical_idx=batch["chemical_idx"],
                class_idx=batch["class_idx"],
                descriptors=batch["descriptors"],
                taxonomy_indices=batch["taxonomy_indices"],
                trophic_idx=batch["trophic_idx"],
                log_concentration=batch["log_concentration"],
                log_exposure_hours=batch["log_exposure_hours"],
                trophic_level_numeric=batch["trophic_level_numeric"],
            )

            losses = self._compute_loss(dose_output, batch["targets"])
            for k, v in losses.items():
                all_losses.setdefault(k, []).append(v.item())

            # Collect logits for calibration
            if "mortality" in batch["targets"]:
                mask = ~torch.isnan(batch["targets"]["mortality"])
                if mask.any():
                    all_mort_logits.append(dose_output.mortality[mask].cpu())
                    all_mort_labels.append(batch["targets"]["mortality"][mask].cpu())

            if "behavioral_change" in batch["targets"]:
                mask = ~torch.isnan(batch["targets"]["behavioral_change"])
                if mask.any():
                    all_behav_logits.append(dose_output.behavioral_change[mask].cpu())
                    all_behav_labels.append(batch["targets"]["behavioral_change"][mask].cpu())

        self.model.train()

        metrics = {k: sum(v) / len(v) for k, v in all_losses.items() if v}

        # Store calibration data for post-training use
        if all_mort_logits:
            self._val_mort_probs = torch.cat(all_mort_logits)
            self._val_mort_labels = torch.cat(all_mort_labels)
        if all_behav_logits:
            self._val_behav_probs = torch.cat(all_behav_logits)
            self._val_behav_labels = torch.cat(all_behav_labels)

        return metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def run_calibration(
    model: DigitalBiosentinel,
    val_mort_probs: torch.Tensor,
    val_mort_labels: torch.Tensor,
    val_behav_probs: Optional[torch.Tensor] = None,
    val_behav_labels: Optional[torch.Tensor] = None,
    n_bins: int = 15,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Run temperature scaling calibration and report ECE.

    Returns dict with temperature values and ECE scores.
    """
    results: Dict[str, float] = {}

    # Convert probabilities to logits for temperature scaling
    mort_logits = torch.log(val_mort_probs.clamp(1e-7, 1.0 - 1e-7) /
                            (1.0 - val_mort_probs.clamp(1e-7, 1.0 - 1e-7)))

    # Pre-calibration ECE
    pre_ece, pre_accs, pre_confs, pre_counts = expected_calibration_error(
        val_mort_probs, val_mort_labels, n_bins=n_bins,
    )
    results["mortality_pre_calibration_ece"] = pre_ece
    logger.info(f"Mortality pre-calibration ECE: {pre_ece:.4f}")

    # Fit temperature
    temp_mort = model.calibrator_mortality.calibrate(
        mort_logits, val_mort_labels, verbose=True,
    )
    results["mortality_temperature"] = temp_mort
    logger.info(f"Mortality temperature: {temp_mort:.4f}")

    # Post-calibration ECE
    calibrated_probs = model.calibrator_mortality.predict_calibrated(mort_logits)
    post_ece, post_accs, post_confs, post_counts = expected_calibration_error(
        calibrated_probs, val_mort_labels, n_bins=n_bins,
    )
    results["mortality_post_calibration_ece"] = post_ece
    logger.info(f"Mortality post-calibration ECE: {post_ece:.4f}")

    # Behavioral calibration
    if val_behav_probs is not None and val_behav_labels is not None:
        behav_logits = torch.log(val_behav_probs.clamp(1e-7, 1.0 - 1e-7) /
                                 (1.0 - val_behav_probs.clamp(1e-7, 1.0 - 1e-7)))

        pre_ece_b, _, _, _ = expected_calibration_error(
            val_behav_probs, val_behav_labels, n_bins=n_bins,
        )
        results["behavioral_pre_calibration_ece"] = pre_ece_b

        temp_behav = model.calibrator_behavioral.calibrate(
            behav_logits, val_behav_labels, verbose=True,
        )
        results["behavioral_temperature"] = temp_behav

        cal_probs_b = model.calibrator_behavioral.predict_calibrated(behav_logits)
        post_ece_b, post_accs_b, post_confs_b, post_counts_b = expected_calibration_error(
            cal_probs_b, val_behav_labels, n_bins=n_bins,
        )
        results["behavioral_post_calibration_ece"] = post_ece_b
        logger.info(f"Behavioral post-calibration ECE: {post_ece_b:.4f}")

    # Save calibration plots
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _save_reliability_diagram(
            pre_accs, pre_confs, pre_counts,
            post_accs, post_confs, post_counts,
            title="Mortality Calibration",
            path=output_dir / "calibration_mortality.png",
        )
        logger.info(f"Saved calibration plot: {output_dir / 'calibration_mortality.png'}")

        if val_behav_probs is not None:
            _save_reliability_diagram(
                [], [], [],
                post_accs_b, post_confs_b, post_counts_b,
                title="Behavioral Calibration",
                path=output_dir / "calibration_behavioral.png",
            )

    return results


def _save_reliability_diagram(
    pre_accs: List[float],
    pre_confs: List[float],
    pre_counts: List[int],
    post_accs: List[float],
    post_confs: List[float],
    post_counts: List[int],
    title: str,
    path: Path,
) -> None:
    """Save a reliability diagram comparing pre- and post-calibration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, accs, confs, counts, label in [
            (axes[0], pre_accs, pre_confs, pre_counts, "Pre-calibration"),
            (axes[1], post_accs, post_confs, post_counts, "Post-calibration"),
        ]:
            if not accs:
                ax.set_title(f"{label} (no data)")
                continue

            n_bins = len(accs)
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.bar(bin_centers, accs, width=1.0 / n_bins, alpha=0.7,
                   edgecolor="black", label="Accuracy")
            ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"{title} — {label}")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(str(path), dpi=150)
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available; skipping reliability diagram")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL Digital Biosentinel training pipeline",
    )
    parser.add_argument("--data-dir", type=str, default="data/ecotox/processed")
    parser.add_argument("--data-format", type=str, default="parquet",
                        choices=["parquet", "csv", "json"])
    parser.add_argument("--output-dir", type=str, default="outputs/biosentinel")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Resume from checkpoint or calibrate-only checkpoint")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Skip training, just run calibration on a trained model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sentinel")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = BiosentinelTrainConfig(
        data_dir=args.data_dir,
        data_format=args.data_format,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        num_workers=args.num_workers,
        fp16=args.fp16,
        oversample_rare=not args.no_oversample,
    )
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr

    trainer = BiosentinelTrainer(config)

    if args.calibrate_only:
        # Load trained model, build datasets for vocab, run calibration only
        logger.info("Calibrate-only mode: loading model and running calibration")
        trainer.setup()
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        # Run validation to collect logits
        val_metrics = trainer.validate(trainer.val_loader)
        logger.info(f"Validation metrics: {val_metrics}")

        # Run calibration
        cal_results = run_calibration(
            model=trainer.model,
            val_mort_probs=getattr(trainer, "_val_mort_probs", torch.tensor([])),
            val_mort_labels=getattr(trainer, "_val_mort_labels", torch.tensor([])),
            val_behav_probs=getattr(trainer, "_val_behav_probs", None),
            val_behav_labels=getattr(trainer, "_val_behav_labels", None),
            n_bins=config.calibration_n_bins,
            output_dir=Path(args.output_dir) / "calibration",
        )
        cal_path = Path(args.output_dir) / "calibration" / "calibration_results.json"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal_results, f, indent=2)
        logger.info(f"Calibration results saved to {cal_path}")
        return

    # Full training pipeline
    logger.info("Starting Digital Biosentinel training")
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)

    summary = trainer.train()
    logger.info(f"Training complete: {json.dumps(summary, indent=2, default=str)}")

    # Post-training calibration
    if config.calibrate_after_training:
        logger.info("Running post-training calibration...")
        val_metrics = trainer.validate(trainer.val_loader)

        cal_results = run_calibration(
            model=trainer.model.module if hasattr(trainer.model, "module") else trainer.model,
            val_mort_probs=getattr(trainer, "_val_mort_probs", torch.tensor([])),
            val_mort_labels=getattr(trainer, "_val_mort_labels", torch.tensor([])),
            val_behav_probs=getattr(trainer, "_val_behav_probs", None),
            val_behav_labels=getattr(trainer, "_val_behav_labels", None),
            n_bins=config.calibration_n_bins,
            output_dir=Path(args.output_dir) / "calibration",
        )

        # Save calibration results
        cal_path = Path(args.output_dir) / "calibration" / "calibration_results.json"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal_results, f, indent=2)
        logger.info(f"Calibration results saved to {cal_path}")

        # Save final model with calibration state
        final_path = Path(args.output_dir) / "checkpoints" / "final_calibrated.pt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        torch.save({
            "model_state_dict": model_to_save.state_dict(),
            "chemical_vocab": trainer._chemical_vocab,
            "species_vocab": trainer._species_vocab,
            "descriptor_means": trainer._descriptor_means,
            "descriptor_stds": trainer._descriptor_stds,
            "calibration": cal_results,
        }, str(final_path))
        logger.info(f"Final calibrated model saved to {final_path}")


if __name__ == "__main__":
    main()
