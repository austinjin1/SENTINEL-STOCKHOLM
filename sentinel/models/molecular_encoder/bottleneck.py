"""Information Bottleneck for minimal biomarker panel discovery.

Uses a gated selection layer with L1 penalty to identify the smallest
subset of genes from full transcriptome data that achieves near-full
classification accuracy for pathway prediction. The "elbow" point
where accuracy plateaus with minimal genes (target: 15-25 genes at
95%+ accuracy) defines the optimal biomarker panel.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chem2path import NUM_PATHWAYS


class GatedSelectionLayer(nn.Module):
    """Differentiable gene selection via learned sigmoid gates.

    Each gene has a learnable score; sigmoid(score) determines how much
    of that gene's expression value passes through. L1 penalty on the
    gate values drives sparsity.

    Args:
        input_dim: Number of genes (full transcriptome).
        temperature: Temperature for sigmoid sharpening during inference.
    """

    def __init__(
        self,
        input_dim: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.temperature = temperature

        # Learnable gate scores, initialized near zero (all gates ~0.5)
        self.gate_scores = nn.Parameter(torch.zeros(input_dim))

    @property
    def gates(self) -> torch.Tensor:
        """Current gate values (soft selection)."""
        return torch.sigmoid(self.gate_scores / self.temperature)

    @property
    def num_selected(self) -> int:
        """Number of genes currently selected (gate > 0.5)."""
        return int((self.gates > 0.5).sum().item())

    def get_selected_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get binary mask of selected genes.

        Args:
            threshold: Gate value threshold for selection.

        Returns:
            Boolean mask [input_dim].
        """
        return self.gates > threshold

    def get_selected_indices(self, threshold: float = 0.5) -> torch.Tensor:
        """Get indices of selected genes.

        Args:
            threshold: Gate value threshold.

        Returns:
            Indices of selected genes [num_selected].
        """
        return torch.where(self.gates > threshold)[0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply gated selection.

        Args:
            x: Gene expression values [B, input_dim].

        Returns:
            selected: Gated expression values [B, input_dim].
            gates: Current gate values [input_dim].
        """
        gates = self.gates  # [input_dim]
        selected = x * gates.unsqueeze(0)  # [B, input_dim]
        return selected, gates

    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 regularization on gate values to encourage sparsity."""
        return self.gates.sum()


class InformationBottleneck(nn.Module):
    """Information bottleneck model for biomarker panel discovery.

    Full pipeline: gene expression -> gated selection -> classifier.
    The gated selection layer is regularized with L1 to find the minimal
    gene set achieving target classification accuracy.

    Args:
        input_dim: Number of genes (full transcriptome dimension).
        hidden_dim: Classifier hidden dimension.
        num_pathways: Number of output pathway classes.
        lambda_l1: L1 penalty weight (higher = more sparsity).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 20000,
        hidden_dim: int = 256,
        num_pathways: int = NUM_PATHWAYS,
        lambda_l1: float = 0.01,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_pathways = num_pathways
        self.lambda_l1 = lambda_l1

        # Gated gene selection
        self.selection = GatedSelectionLayer(input_dim)

        # Pathway classifier on selected genes
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pathways),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through gated selection and classifier.

        Args:
            x: Full gene expression vector [B, input_dim].

        Returns:
            Dict with:
                'logits': Pathway classification logits [B, num_pathways].
                'selected': Gated expression values [B, input_dim].
                'gates': Gate values [input_dim].
                'num_selected': Number of selected genes (scalar).
        """
        selected, gates = self.selection(x)
        logits = self.classifier(selected)

        return {
            "logits": logits,
            "selected": selected,
            "gates": gates,
            "num_selected": torch.tensor(
                self.selection.num_selected,
                dtype=torch.float32,
                device=x.device,
            ),
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss with L1 sparsity penalty.

        Args:
            outputs: Forward pass outputs.
            targets: Pathway classification targets [B, num_pathways] (multi-label).

        Returns:
            Dict with 'total', 'classification', 'l1_penalty', and 'num_selected'.
        """
        # Multi-label classification loss
        cls_loss = F.binary_cross_entropy_with_logits(
            outputs["logits"], targets.float(), reduction="mean"
        )

        # L1 sparsity penalty
        l1 = self.selection.l1_penalty()

        total = cls_loss + self.lambda_l1 * l1

        return {
            "total": total,
            "classification": cls_loss,
            "l1_penalty": l1,
            "num_selected": outputs["num_selected"],
        }


def sweep_lambda(
    model_factory,
    train_loader,
    val_loader,
    lambda_values: list[float],
    num_epochs: int = 50,
    device: torch.device = torch.device("cpu"),
) -> list[dict[str, float]]:
    """Sweep L1 penalty lambda to find the optimal sparsity-accuracy tradeoff.

    Trains a separate model for each lambda value and records:
    - Number of selected genes (gate > 0.5)
    - Classification accuracy on validation set

    The "elbow" point is where accuracy plateaus with minimal genes,
    targeting 15-25 genes at 95%+ of full-transcriptome accuracy.

    Args:
        model_factory: Callable that returns a new InformationBottleneck model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        lambda_values: List of L1 penalty values to sweep.
        num_epochs: Training epochs per lambda.
        device: Training device.

    Returns:
        List of dicts with 'lambda', 'num_genes', 'accuracy' per sweep point.
    """
    results = []

    for lam in lambda_values:
        model = model_factory(lambda_l1=lam).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                losses = model.compute_loss(outputs, batch_y)
                losses["total"].backward()
                optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = (torch.sigmoid(outputs["logits"]) > 0.5).float()
                correct += (preds == batch_y).all(dim=1).sum().item()
                total += batch_y.shape[0]

        accuracy = correct / max(total, 1)
        num_genes = model.selection.num_selected

        results.append({
            "lambda": lam,
            "num_genes": num_genes,
            "accuracy": accuracy,
        })

    return results


def find_elbow_point(
    sweep_results: list[dict[str, float]],
    target_accuracy_fraction: float = 0.95,
) -> dict[str, float]:
    """Find the elbow point: minimal genes achieving target accuracy.

    Args:
        sweep_results: Output from sweep_lambda().
        target_accuracy_fraction: Fraction of best accuracy to target (0.95 = 95%).

    Returns:
        The sweep result dict at the elbow point.
    """
    if not sweep_results:
        raise ValueError("Empty sweep results")

    # Find maximum accuracy across all lambda values
    best_accuracy = max(r["accuracy"] for r in sweep_results)
    target = best_accuracy * target_accuracy_fraction

    # Filter to results meeting target accuracy
    valid = [r for r in sweep_results if r["accuracy"] >= target]
    if not valid:
        # Fall back to best accuracy result
        return min(sweep_results, key=lambda r: -r["accuracy"])

    # Among valid results, find the one with fewest genes
    return min(valid, key=lambda r: r["num_genes"])
