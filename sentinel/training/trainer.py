"""
Base trainer class for SENTINEL training pipelines.

Provides common training utilities: checkpointing, early stopping, learning
rate scheduling, logging (wandb + console via rich), gradient clipping, metric
tracking, best model selection, and optional distributed training (DDP).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Base configuration for all trainers."""

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    batch_size: int = 32
    epochs: int = 50
    warmup_steps: int = 0
    warmup_epochs: int = 0
    scheduler: str = "cosine"  # cosine | linear | constant

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    output_dir: str = "outputs"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # Logging
    use_wandb: bool = True
    wandb_project: str = "sentinel"
    wandb_run_name: str = ""
    log_every_n_steps: int = 50

    # Distributed
    distributed: bool = False
    local_rank: int = -1

    # Hardware
    device: str = "auto"
    fp16: bool = False
    num_workers: int = 4
    pin_memory: bool = True

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Tracks training and validation metrics across epochs."""

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = {}
        self._step_buffer: Dict[str, List[float]] = {}

    def update(self, name: str, value: float) -> None:
        """Add a single value to the step buffer (for averaging)."""
        self._step_buffer.setdefault(name, []).append(value)

    def flush_epoch(self, epoch: int) -> Dict[str, float]:
        """Average buffered step values and record as epoch metrics."""
        epoch_metrics: Dict[str, float] = {"epoch": float(epoch)}
        for name, values in self._step_buffer.items():
            avg = sum(values) / len(values) if values else 0.0
            self._history.setdefault(name, []).append(avg)
            epoch_metrics[name] = avg
        self._step_buffer.clear()
        return epoch_metrics

    def record_epoch(self, name: str, value: float) -> None:
        """Record a single epoch-level metric directly."""
        self._history.setdefault(name, []).append(value)

    def get_best(self, name: str, mode: str = "min") -> tuple[float, int]:
        """Return (best_value, best_epoch) for a metric."""
        values = self._history.get(name, [])
        if not values:
            return (float("inf") if mode == "min" else float("-inf"), -1)
        fn = min if mode == "min" else max
        best_val = fn(values)
        best_epoch = values.index(best_val)
        return best_val, best_epoch

    def get_last(self, name: str) -> float:
        """Return the most recent value for a metric."""
        values = self._history.get(name, [])
        return values[-1] if values else float("nan")

    def to_dict(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor a metric and signal when training should stop."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (
            (value < self.best - self.min_delta) if self.mode == "min"
            else (value > self.best + self.min_delta)
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# LR Scheduler builders
# ---------------------------------------------------------------------------

def build_cosine_schedule(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine decay with optional linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def build_linear_schedule(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
) -> LambdaLR:
    """Linear decay with optional warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
) -> Optional[LambdaLR]:
    """Factory for LR schedulers."""
    if name == "cosine":
        return build_cosine_schedule(optimizer, total_steps, warmup_steps)
    elif name == "linear":
        return build_linear_schedule(optimizer, total_steps, warmup_steps)
    elif name == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------

class BaseTrainer(ABC):
    """Abstract base trainer with checkpointing, logging, and DDP support.

    Subclasses must implement:
        - build_model()
        - build_datasets()
        - train_step(batch) -> dict[str, float]
        - validate(dataloader) -> dict[str, float]
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.metrics = MetricTracker()
        self.global_step = 0
        self.current_epoch = 0

        # Resolve device
        if config.device == "auto":
            if config.distributed and config.local_rank >= 0:
                self.device = torch.device(f"cuda:{config.local_rank}")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # Distributed setup
        self.is_main_process = True
        if config.distributed:
            self._init_distributed()

        # Reproducibility
        self._seed_everything(config.seed)

        # Output directory
        self.output_dir = Path(config.output_dir)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        self._wandb_run = None
        if config.use_wandb and self.is_main_process:
            self._init_wandb()

        # Placeholders set by subclass
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LambdaLR] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.scaler: Optional[torch.amp.GradScaler] = None

        if config.fp16 and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the model."""
        ...

    @abstractmethod
    def build_datasets(self) -> tuple[Any, Any]:
        """Return (train_dataset, val_dataset)."""
        ...

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Run one training step. Return dict of loss values."""
        ...

    @abstractmethod
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run full validation. Return dict of metric values."""
        ...

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Build model, datasets, optimizer, scheduler. Call before train()."""
        self.model = self.build_model()
        self.model.to(self.device)

        if self.config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
            )

        train_ds, val_ds = self.build_datasets()
        self.train_loader = self._build_dataloader(train_ds, shuffle=True)
        self.val_loader = self._build_dataloader(val_ds, shuffle=False)

        self.optimizer = self.build_optimizer()
        total_steps = len(self.train_loader) * self.config.epochs
        warmup = self.config.warmup_steps or (
            self.config.warmup_epochs * len(self.train_loader)
        )
        self.scheduler = build_scheduler(
            self.config.scheduler, self.optimizer, total_steps, warmup
        )

    def build_optimizer(self) -> Optimizer:
        """Build AdamW optimizer with weight decay. Override to customize."""
        assert self.model is not None
        # Separate weight decay for norm/bias params
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(param_groups, lr=self.config.lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Main training loop. Returns final metrics."""
        assert self.model is not None and self.train_loader is not None
        assert self.optimizer is not None

        early_stop = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="min",
        ) if self.config.early_stopping else None

        best_val_metric = float("inf")
        best_epoch = 0

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            if self.config.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = self._train_epoch()
            epoch_metrics = self.metrics.flush_epoch(epoch)

            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate(self.val_loader)
                for k, v in val_metrics.items():
                    self.metrics.record_epoch(f"val_{k}", v)
                    epoch_metrics[f"val_{k}"] = v

            # Logging
            if self.is_main_process:
                self._log_epoch(epoch, epoch_metrics)

                # Checkpointing
                val_loss = val_metrics.get("loss", float("inf"))
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    best_epoch = epoch
                    self._save_checkpoint(epoch, is_best=True)
                elif (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch)

                # Early stopping
                if early_stop is not None and early_stop.step(val_loss):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={self.config.patience})"
                    )
                    break

        # Final summary
        summary = {
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "total_epochs": self.current_epoch + 1,
            "total_steps": self.global_step,
            "metrics": self.metrics.to_dict(),
        }

        if self.is_main_process:
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=_json_default)
            logger.info(f"Training complete. Summary saved to {summary_path}")

            if self._wandb_run is not None:
                self._wandb_run.finish()

        return summary

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        assert self.model is not None and self.train_loader is not None
        self.model.train()

        epoch_losses: Dict[str, List[float]] = {}
        progress = make_progress() if self.is_main_process else None

        if progress:
            progress.start()
            task = progress.add_task(
                f"Epoch {self.current_epoch}", total=len(self.train_loader)
            )

        for batch in self.train_loader:
            batch = self._to_device(batch)

            # Forward + backward with optional AMP
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    losses = self.train_step(batch)
                loss = losses.get("loss", sum(losses.values()))
                self.scaler.scale(torch.tensor(loss, device=self.device) if not isinstance(loss, torch.Tensor) else loss).backward()
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.train_step(batch)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            # Track metrics
            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                self.metrics.update(k, val)
                epoch_losses.setdefault(k, []).append(val)

            # Step-level logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_step(losses)

            if progress:
                loss_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v.item():.4f}"
                    for k, v in losses.items()
                )
                progress.update(task, advance=1, description=f"Epoch {self.current_epoch} [{loss_str}]")

        if progress:
            progress.stop()

        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        """Save model, optimizer, and scheduler state."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        model_to_save = (
            self.model.module if isinstance(self.model, DDP) else self.model
        )

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = ckpt_dir / "best_model.pt"
            shutil.copy2(path, best_path)
            logger.info(f"New best model at epoch {epoch}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints(ckpt_dir)
        return path

    def _cleanup_checkpoints(self, ckpt_dir: Path) -> None:
        """Keep only the last N checkpoints plus best_model.pt."""
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        while len(ckpts) > self.config.keep_last_n_checkpoints:
            old = ckpts.pop(0)
            old.unlink()

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return the epoch to resume from."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(state["model_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state.get("global_step", 0)
        epoch = state.get("epoch", 0)
        logger.info(f"Resumed from checkpoint at epoch {epoch}")
        return epoch

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        """Initialize PyTorch distributed process group."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        self.config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main_process = self.config.local_rank == 0
        torch.cuda.set_device(self.config.local_rank)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or None,
                config=self.config.__dict__,
                reinit=True,
            )
        except ImportError:
            logger.warning("wandb not installed; logging to console only")
        except Exception as exc:
            logger.warning(f"wandb init failed: {exc}; logging to console only")

    def _log_step(self, metrics: Dict[str, Any]) -> None:
        """Log step-level metrics."""
        if self._wandb_run is not None:
            import wandb
            flat = {}
            for k, v in metrics.items():
                flat[f"train/{k}"] = v.item() if isinstance(v, torch.Tensor) else v
            flat["train/lr"] = self.optimizer.param_groups[0]["lr"]
            wandb.log(flat, step=self.global_step)

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch-level metrics to console and wandb."""
        parts = [f"Epoch {epoch:3d}"]
        for k, v in sorted(metrics.items()):
            if k == "epoch":
                continue
            parts.append(f"{k}={v:.5f}")
        logger.info(" | ".join(parts))

        if self._wandb_run is not None:
            import wandb
            wandb.log(
                {f"epoch/{k}": v for k, v in metrics.items()},
                step=self.global_step,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_dataloader(
        self,
        dataset: Any,
        shuffle: bool = True,
    ) -> DataLoader:
        """Build DataLoader with optional distributed sampler."""
        sampler = None
        if self.config.distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def _to_device(self, batch: Any) -> Any:
        """Recursively move batch tensors to the training device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        return batch

    def _seed_everything(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def freeze_model(self, model: nn.Module) -> None:
        """Freeze all parameters in a model."""
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze_model(self, model: nn.Module) -> None:
        """Unfreeze all parameters in a model."""
        for p in model.parameters():
            p.requires_grad = True

    def unfreeze_top_layers(self, model: nn.Module, n_layers: int = 4) -> None:
        """Unfreeze the last N transformer layers of a model.

        Assumes transformer blocks are accessible via model.blocks or similar.
        """
        # Freeze everything first
        self.freeze_model(model)
        # Find block-like submodules and unfreeze top N
        blocks = None
        for attr in ("blocks", "layers", "encoder.layer", "transformer.layers"):
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                blocks = list(obj)
                break
            except AttributeError:
                continue

        if blocks is not None:
            for block in blocks[-n_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            logger.info(f"Unfroze top {n_layers} blocks ({len(blocks)} total)")
        else:
            # Fallback: unfreeze last N% of parameters by name
            params = list(model.named_parameters())
            cutoff = max(1, len(params) - len(params) // (n_layers or 1))
            for name, p in params[cutoff:]:
                p.requires_grad = True
            logger.info(f"Unfroze last {len(params) - cutoff} params by position")


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} not JSON serializable")
