"""Hierarchy-aware information bottleneck for minimal gene panel discovery.

Applies gated selection at the gene input layer of the biological hierarchy
network. L1 penalty on gates drives sparsity, enabling discovery of the
minimal gene set (target: 20-50 genes) that preserves 95%+ of full
transcriptome classification accuracy.

Adapted from the original information bottleneck to work with the P-NET
hierarchy architecture rather than a standalone classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HierarchyBottleneck(nn.Module):
    """Gated gene selection layer for the hierarchy network.

    Each gene has a learnable score; sigmoid(score) acts as a soft gate
    controlling how much of that gene's expression passes into the
    hierarchy network. L1 regularization on gate values encourages the
    network to rely on as few genes as possible.

    Args:
        gene_names: List of gene names matching the hierarchy input.
        lambda_l1: L1 penalty weight (higher = more sparsity, fewer genes).
        temperature: Sigmoid temperature for gate sharpening. Lower values
            make gates more binary during inference.
    """

    def __init__(
        self,
        gene_names: list[str],
        lambda_l1: float = 0.01,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.lambda_l1 = lambda_l1
        self.temperature = temperature

        # Learnable gate scores, initialized near zero (gates start at ~0.5)
        self.gate_scores = nn.Parameter(torch.zeros(self.n_genes))

    @property
    def gates(self) -> torch.Tensor:
        """Current soft gate values in [0, 1]."""
        return torch.sigmoid(self.gate_scores / self.temperature)

    @property
    def num_selected(self) -> int:
        """Number of genes currently passing the 0.5 threshold."""
        return int((self.gates > 0.5).sum().item())

    def get_selected_genes(self, threshold: float = 0.5) -> list[str]:
        """Return gene names passing the gate threshold.

        Args:
            threshold: Gate value threshold for selection.

        Returns:
            List of selected gene name strings.
        """
        mask = self.gates > threshold
        indices = torch.where(mask)[0]
        return [self.gene_names[i] for i in indices.tolist()]

    def get_selected_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get boolean mask of selected genes.

        Args:
            threshold: Gate value threshold.

        Returns:
            Boolean tensor [n_genes].
        """
        return self.gates > threshold

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """Apply gated selection to gene expression.

        Args:
            gene_expression: Raw expression values [B, n_genes].

        Returns:
            Gated expression [B, n_genes] with unselected genes suppressed.
        """
        gates = self.gates  # [n_genes]
        return gene_expression * gates.unsqueeze(0)  # [B, n_genes]

    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 regularization loss on gate values."""
        return self.gates.sum()

    def compute_loss(self) -> torch.Tensor:
        """Compute weighted L1 penalty (convenience method).

        Returns:
            Scalar L1 loss multiplied by lambda_l1.
        """
        return self.lambda_l1 * self.l1_penalty()


def sweep_lambda(
    model_factory,
    train_loader,
    val_loader,
    lambda_values: list[float],
    num_epochs: int = 50,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    """Sweep L1 penalty lambda to find optimal sparsity-accuracy tradeoff.

    Trains a separate model for each lambda and records the number of
    selected genes vs. validation accuracy. The elbow point where accuracy
    plateaus with minimal genes (target: 20-50 genes at 95%+ accuracy)
    defines the optimal biomarker panel.

    Args:
        model_factory: Callable(lambda_l1=float) returning a full
            ToxiGeneEncoder model ready for training.
        train_loader: Training DataLoader yielding (inputs_dict, targets).
        val_loader: Validation DataLoader.
        lambda_values: L1 penalty values to sweep.
        num_epochs: Training epochs per lambda value.
        device: Training device. Defaults to CPU.

    Returns:
        List of dicts with 'lambda', 'num_genes', 'accuracy' per sweep point.
    """
    if device is None:
        device = torch.device("cpu")

    results = []

    for lam in lambda_values:
        model = model_factory(lambda_l1=lam).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Training
        model.train()
        for _epoch in range(num_epochs):
            for batch_inputs, batch_targets in train_loader:
                # Move inputs to device
                batch_inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_inputs.items()
                }
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                outputs = model(**batch_inputs)
                loss = model.compute_loss(outputs, batch_targets)
                loss["total"].backward()
                optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_inputs.items()
                }
                batch_targets = batch_targets.to(device)

                outputs = model(**batch_inputs)
                preds = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float()
                correct += (preds == batch_targets).all(dim=1).sum().item()
                total += batch_targets.shape[0]

        accuracy = correct / max(total, 1)
        num_genes = model.bottleneck.num_selected

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
        target_accuracy_fraction: Fraction of best accuracy to target
            (0.95 = 95% of full-transcriptome accuracy).

    Returns:
        The sweep result dict at the elbow point.
    """
    if not sweep_results:
        raise ValueError("Empty sweep results")

    best_accuracy = max(r["accuracy"] for r in sweep_results)
    target = best_accuracy * target_accuracy_fraction

    valid = [r for r in sweep_results if r["accuracy"] >= target]
    if not valid:
        return min(sweep_results, key=lambda r: -r["accuracy"])

    return min(valid, key=lambda r: r["num_genes"])
