"""Information-Theoretic Sensor Placement.

Optimal allocation of environmental monitoring infrastructure across modality
types using submodular optimization with information-theoretic objectives.

The core idea: given a fixed budget and a set of candidate sensor locations
(each with a modality type and cost), select the subset that maximizes the
conditional mutual information I(Y; X_S) where Y is the environmental state
we wish to monitor and X_S is the set of selected sensors.

Because mutual information is submodular for Gaussian processes (Krause et
al., 2008), the greedy algorithm achieves a (1 - 1/e) approximation to the
optimal solution.  For large-scale watersheds where even greedy evaluation
is expensive, a :class:`GNNSurrogate` learns to predict marginal information
gain, enabling gradient-based optimization.

References
----------
[1] Krause, A. et al. (2008). "Near-Optimal Sensor Placements in Gaussian
    Processes: Theory, Efficient Algorithms and Empirical Studies." JMLR.
[2] Nemhauser, G. et al. (1978). "An analysis of approximations for
    maximizing submodular set functions." Mathematical Programming.
[3] Wei, K. et al. (2015). "Submodularity in Data Subset Selection and
    Active Learning." ICML 2015.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    """A selected sensor placement."""

    location_id: int
    modality: str
    cost: float
    marginal_gain: float
    cumulative_gain: float


@dataclass
class CandidateSensor:
    """A candidate sensor that could be placed."""

    location_id: int
    modality: str
    cost: float = 1.0
    latitude: float = 0.0
    longitude: float = 0.0


# ---------------------------------------------------------------------------
# Submodular objective
# ---------------------------------------------------------------------------


class SubmodularObjective(nn.Module):
    """Information-theoretic submodular objective for sensor placement.

    Models the joint distribution of environmental measurements as a
    multivariate Gaussian (GP posterior), under which the mutual
    information objective I(Y; X_S) is submodular.

    The conditional mutual information is computed as:

    .. math::

        f(S) = I(Y; X_S) = \\frac{1}{2} \\log \\det(I + \\sigma^{-2} K_S)

    where :math:`K_S` is the kernel matrix for the selected sensor
    locations and :math:`\\sigma^2` is the noise variance.

    The greedy algorithm selects sensors one at a time, each time choosing
    the sensor with the highest marginal gain:

    .. math::

        s^* = \\arg\\max_{s \\in V \\setminus S} f(S \\cup \\{s\\}) - f(S)

    This achieves a :math:`(1 - 1/e) \\approx 0.632` approximation to OPT
    for monotone submodular functions under cardinality constraints, and
    extends to knapsack constraints (cost-weighted greedy).

    Args:
        n_candidates: Number of candidate sensor locations.
        feature_dim: Dimensionality of location features.
        kernel_type: Kernel for GP covariance (``"rbf"`` or ``"matern"``).
        noise_variance: Observation noise variance :math:`\\sigma^2`.
        length_scale: Kernel length scale.
    """

    def __init__(
        self,
        n_candidates: int,
        feature_dim: int,
        kernel_type: str = "rbf",
        noise_variance: float = 0.1,
        length_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_candidates = n_candidates
        self.feature_dim = feature_dim
        self.kernel_type = kernel_type
        self.noise_variance = noise_variance

        self.log_length_scale = nn.Parameter(torch.tensor(math.log(length_scale)))
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(noise_variance)))

    @property
    def length_scale(self) -> torch.Tensor:
        return self.log_length_scale.exp()

    @property
    def noise_var(self) -> torch.Tensor:
        return self.log_noise_var.exp()

    def _compute_kernel(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute kernel matrix.

        Args:
            x: ``(N, D)`` features.
            y: ``(M, D)`` features. If ``None``, computes ``K(x, x)``.

        Returns:
            Kernel matrix ``(N, M)`` or ``(N, N)``.
        """
        if y is None:
            y = x
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        ls2 = self.length_scale ** 2

        if self.kernel_type == "rbf":
            return torch.exp(-0.5 * dist_sq / ls2)
        elif self.kernel_type == "matern":
            # Matern 5/2
            dist = torch.cdist(x, y, p=2)
            r = math.sqrt(5.0) * dist / self.length_scale
            return (1.0 + r + r.pow(2) / 3.0) * torch.exp(-r)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def evaluate(
        self,
        features: torch.Tensor,
        selected_indices: List[int],
    ) -> torch.Tensor:
        """Evaluate the submodular objective f(S).

        .. math::

            f(S) = \\frac{1}{2} \\log \\det(I + \\sigma^{-2} K_S)

        Args:
            features: ``(N, D)`` features for all candidate locations.
            selected_indices: Indices of selected sensors.

        Returns:
            Objective value (scalar tensor).
        """
        if len(selected_indices) == 0:
            return torch.tensor(0.0, device=features.device)

        idx = torch.tensor(selected_indices, device=features.device)
        x_s = features[idx]
        K_s = self._compute_kernel(x_s)

        # f(S) = 0.5 * log det(I + sigma^{-2} K_S)
        n = K_s.size(0)
        eye = torch.eye(n, device=K_s.device)
        M = eye + K_s / self.noise_var
        # Use slogdet for numerical stability
        sign, logdet = torch.linalg.slogdet(M)
        return 0.5 * logdet

    def marginal_gain(
        self,
        features: torch.Tensor,
        selected_indices: List[int],
        candidate_idx: int,
    ) -> torch.Tensor:
        """Compute marginal gain of adding a single sensor.

        .. math::

            \\Delta(s | S) = f(S \\cup \\{s\\}) - f(S)

        Args:
            features: ``(N, D)`` features for all candidate locations.
            selected_indices: Currently selected indices.
            candidate_idx: Index of the candidate to evaluate.

        Returns:
            Marginal gain (scalar tensor).
        """
        f_current = self.evaluate(features, selected_indices)
        f_augmented = self.evaluate(
            features, selected_indices + [candidate_idx]
        )
        return f_augmented - f_current

    def forward(
        self, features: torch.Tensor, selected_mask: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate objective from a differentiable selection mask.

        For use with the GNN surrogate during training.

        Args:
            features: ``(N, D)`` candidate features.
            selected_mask: ``(N,)`` soft selection mask in ``[0, 1]``.

        Returns:
            Approximate objective value (scalar).
        """
        # Soft kernel weighting: K_soft = diag(mask) @ K @ diag(mask)
        K = self._compute_kernel(features)
        mask = selected_mask.unsqueeze(0) * selected_mask.unsqueeze(1)
        K_soft = K * mask
        n = K_soft.size(0)
        M = torch.eye(n, device=K.device) + K_soft / self.noise_var
        sign, logdet = torch.linalg.slogdet(M)
        return 0.5 * logdet


# ---------------------------------------------------------------------------
# GNN surrogate for scalable placement
# ---------------------------------------------------------------------------


class _GraphConvLayer(nn.Module):
    """Simple graph convolution: aggregate neighbor features + self."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features ``(N, D_in)``.
            adj: Adjacency matrix ``(N, N)``, may be weighted.

        Returns:
            Updated features ``(N, D_out)``.
        """
        # Degree-normalize adjacency
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg

        neigh_agg = adj_norm @ x  # (N, D_in)
        return self.self_linear(x) + self.neigh_linear(neigh_agg)


class GNNSurrogate(nn.Module):
    """GNN that predicts marginal information gain for candidate placements.

    Takes a watershed graph with existing sensor locations encoded as node
    features and predicts, for each candidate location, the marginal
    information gain from placing a sensor there.

    This avoids the O(N^3) cost of evaluating the exact submodular
    objective for each candidate and enables gradient-based optimization
    for large-scale watersheds.

    Args:
        node_feature_dim: Dimensionality of input node features (includes
            location, existing sensor indicators, environmental covariates).
        hidden_dim: Hidden layer size in the GNN.
        n_layers: Number of graph convolution layers.
        n_modalities: Number of modality types (output per modality).
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_modalities: int = 5,
    ) -> None:
        super().__init__()
        self.n_modalities = n_modalities

        layers: List[nn.Module] = []
        in_dim = node_feature_dim
        for i in range(n_layers):
            out_dim = hidden_dim
            layers.append(_GraphConvLayer(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        self.gnn_layers = nn.ModuleList(layers)

        # Prediction head: marginal gain per modality for each node
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_modalities),
            nn.Softplus(),  # gains are non-negative
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Predict marginal information gain for each node and modality.

        Args:
            node_features: ``(N, D)`` node features.
            adjacency: ``(N, N)`` adjacency matrix.

        Returns:
            Predicted marginal gains ``(N, n_modalities)``.
        """
        x = node_features
        for layer in self.gnn_layers:
            if isinstance(layer, _GraphConvLayer):
                x = layer(x, adjacency)
            else:
                x = layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Greedy optimization
# ---------------------------------------------------------------------------


def optimize_placement(
    features: torch.Tensor,
    adjacency: Optional[torch.Tensor],
    candidates: List[CandidateSensor],
    budget: float,
    modality_costs: Dict[str, float],
    objective: Optional[SubmodularObjective] = None,
    surrogate: Optional[GNNSurrogate] = None,
    use_surrogate: bool = False,
) -> List[Placement]:
    """Select sensors via cost-weighted greedy submodular maximization.

    Uses the classic greedy algorithm which, for monotone submodular
    functions under a knapsack constraint, achieves a
    :math:`(1 - 1/e)` approximation to the optimal solution.

    If ``use_surrogate=True`` and a :class:`GNNSurrogate` is provided,
    marginal gains are predicted by the GNN rather than computed exactly,
    enabling scalability to large candidate sets.

    Args:
        features: ``(N, D)`` candidate location features.
        adjacency: ``(N, N)`` adjacency matrix (required if using surrogate).
        candidates: List of candidate sensors with costs and modalities.
        budget: Total budget for sensor placement.
        modality_costs: Default cost per modality type.
        objective: Exact submodular objective (used if not using surrogate).
        surrogate: GNN surrogate for fast gain prediction.
        use_surrogate: Whether to use the GNN surrogate.

    Returns:
        Ordered list of :class:`Placement` objects.
    """
    if objective is None and not use_surrogate:
        objective = SubmodularObjective(
            n_candidates=len(candidates),
            feature_dim=features.size(1),
        )

    selected_indices: List[int] = []
    placements: List[Placement] = []
    remaining_budget = budget
    available = set(range(len(candidates)))
    cumulative_gain = 0.0

    # Precompute surrogate gains if using GNN
    surrogate_gains: Optional[torch.Tensor] = None
    if use_surrogate and surrogate is not None and adjacency is not None:
        with torch.no_grad():
            surrogate_gains = surrogate(features, adjacency)  # (N, M)

    while available and remaining_budget > 0:
        best_idx = -1
        best_gain = -float("inf")
        best_cost = 0.0

        for idx in available:
            c = candidates[idx]
            cost = c.cost if c.cost > 0 else modality_costs.get(c.modality, 1.0)
            if cost > remaining_budget:
                continue

            if use_surrogate and surrogate_gains is not None:
                # Look up modality index (simplified: use hash)
                modality_idx = hash(c.modality) % surrogate_gains.size(1)
                gain = surrogate_gains[idx, modality_idx].item()
            else:
                assert objective is not None
                gain = objective.marginal_gain(
                    features, selected_indices, idx
                ).item()

            # Cost-effectiveness: gain per unit cost
            efficiency = gain / max(cost, 1e-8)
            if efficiency > best_gain:
                best_gain = efficiency
                best_idx = idx
                best_cost = cost

        if best_idx < 0:
            break  # No affordable candidate improves objective

        # Select the best candidate
        c = candidates[best_idx]
        actual_cost = c.cost if c.cost > 0 else modality_costs.get(c.modality, 1.0)

        if use_surrogate and surrogate_gains is not None:
            modality_idx = hash(c.modality) % surrogate_gains.size(1)
            marginal = surrogate_gains[best_idx, modality_idx].item()
        else:
            assert objective is not None
            marginal = objective.marginal_gain(
                features, selected_indices, best_idx
            ).item()

        cumulative_gain += marginal
        selected_indices.append(best_idx)
        available.discard(best_idx)
        remaining_budget -= actual_cost

        placements.append(
            Placement(
                location_id=c.location_id,
                modality=c.modality,
                cost=actual_cost,
                marginal_gain=marginal,
                cumulative_gain=cumulative_gain,
            )
        )
        logger.info(
            f"Placed sensor #{len(placements)}: location={c.location_id}, "
            f"modality={c.modality}, gain={marginal:.4f}, "
            f"budget_remaining={remaining_budget:.2f}"
        )

    logger.info(
        f"Placement complete: {len(placements)} sensors, "
        f"total gain={cumulative_gain:.4f}"
    )
    return placements
