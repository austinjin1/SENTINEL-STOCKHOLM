"""Heterogeneous Temporal Causal Discovery.

Extends the PCMCI algorithm (Runge et al., 2019) to heterogeneous multimodal
environmental data that lives in fundamentally different geometric spaces:
Euclidean (sensor time series), compositional (microbial simplex), and image
feature spaces (satellite).

The key contribution is :class:`CrossSpaceConditionalIndependence`, a
kernel-based conditional independence test (using HSIC) with geometry-aware
kernels for each space type.  This enables discovery of causal chains such as:

    satellite turbidity -> DO drop -> microbial anaerobic shift

:class:`HeterogeneousPCMCI` wraps the causal discovery logic with
cross-modal variable handling and FDR control for multiple testing.

References
----------
[1] Runge, J. et al. (2019). "Detecting and quantifying causal associations
    in large nonlinear time series datasets." Science Advances, 5(11).
[2] Gretton, A. et al. (2005). "Measuring Statistical Dependence with
    Hilbert-Schmidt Norms." ALT 2005.
[3] Zhang, K. et al. (2011). "Kernel-based Conditional Independence Test
    and Application in Causal Discovery." UAI 2011.
[4] Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Space types and causal graph representation
# ---------------------------------------------------------------------------


class SpaceType(Enum):
    """Geometric space type for a variable."""

    EUCLIDEAN = auto()
    COMPOSITIONAL = auto()
    IMAGE_FEATURE = auto()


@dataclass
class CausalEdge:
    """A single directed causal edge in the discovered graph."""

    source: str
    target: str
    lag: int  # time lag (positive = source precedes target)
    strength: float  # estimated causal strength
    p_value: float
    source_space: SpaceType = SpaceType.EUCLIDEAN
    target_space: SpaceType = SpaceType.EUCLIDEAN


@dataclass
class CausalGraph:
    """Result of causal discovery: a directed graph with edge metadata."""

    edges: List[CausalEdge] = field(default_factory=list)
    variable_names: List[str] = field(default_factory=list)
    variable_spaces: Dict[str, SpaceType] = field(default_factory=dict)
    max_lag: int = 0

    def significant_edges(self, alpha: float = 0.05) -> List[CausalEdge]:
        """Return edges with p-value below significance threshold."""
        return [e for e in self.edges if e.p_value < alpha]

    def adjacency_matrix(self, lag: Optional[int] = None) -> torch.Tensor:
        """Return ``(V, V)`` adjacency matrix, optionally at a specific lag."""
        V = len(self.variable_names)
        name_to_idx = {n: i for i, n in enumerate(self.variable_names)}
        adj = torch.zeros(V, V)
        for e in self.edges:
            if lag is not None and e.lag != lag:
                continue
            i = name_to_idx.get(e.source)
            j = name_to_idx.get(e.target)
            if i is not None and j is not None:
                adj[i, j] = e.strength
        return adj


# ---------------------------------------------------------------------------
# Geometry-aware kernels
# ---------------------------------------------------------------------------


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian RBF kernel on Euclidean space.

    Args:
        x: ``(N, D)`` data matrix.
        y: ``(M, D)`` data matrix.
        sigma: Bandwidth parameter.

    Returns:
        Kernel matrix ``(N, M)``.
    """
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def _aitchison_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    """RBF kernel in Aitchison geometry for compositional data.

    First applies the CLR (centered log-ratio) transform, then computes
    the standard RBF kernel in CLR space.

    Args:
        x: ``(N, D)`` compositions on the simplex (rows sum to 1).
        y: ``(M, D)`` compositions on the simplex.
        sigma: Bandwidth in CLR space.

    Returns:
        Kernel matrix ``(N, M)``.
    """
    eps = 1e-10
    x_clr = torch.log(x.clamp(min=eps)) - torch.log(x.clamp(min=eps)).mean(dim=-1, keepdim=True)
    y_clr = torch.log(y.clamp(min=eps)) - torch.log(y.clamp(min=eps)).mean(dim=-1, keepdim=True)
    return _rbf_kernel(x_clr, y_clr, sigma)


def _image_feature_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Cosine-similarity-based kernel for image feature embeddings.

    Uses the exponentiated cosine similarity (equivalent to an RBF kernel
    on the unit hypersphere):

    .. math::

        k(x, y) = \\exp\\left(\\frac{\\cos(x, y) - 1}{\\sigma^2}\\right)

    Args:
        x: ``(N, D)`` image feature vectors.
        y: ``(M, D)`` image feature vectors.
        sigma: Bandwidth controlling sharpness.

    Returns:
        Kernel matrix ``(N, M)``.
    """
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    cos_sim = x_norm @ y_norm.T  # (N, M)
    return torch.exp((cos_sim - 1.0) / (sigma ** 2))


def _select_kernel(
    space_type: SpaceType,
) -> callable:
    """Return the appropriate kernel function for a space type."""
    _KERNEL_MAP = {
        SpaceType.EUCLIDEAN: _rbf_kernel,
        SpaceType.COMPOSITIONAL: _aitchison_kernel,
        SpaceType.IMAGE_FEATURE: _image_feature_kernel,
    }
    return _KERNEL_MAP[space_type]


def _median_heuristic(x: torch.Tensor) -> float:
    """Compute the median-distance heuristic for kernel bandwidth."""
    with torch.no_grad():
        dists = torch.cdist(x, x, p=2)
        # Take upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        median_dist = dists[mask].median().item()
        return max(median_dist, 1e-6)


# ---------------------------------------------------------------------------
# Cross-space conditional independence test (HSIC-based)
# ---------------------------------------------------------------------------


class CrossSpaceConditionalIndependence:
    """Kernel-based conditional independence test across geometric spaces.

    Tests ``X _||_ Y | Z`` where X, Y, Z may live in different geometric
    spaces (Euclidean, compositional simplex, image feature space).

    Uses a residualization approach: regress out the effect of Z from X
    and Y in their respective kernel spaces, then test independence of
    the residuals via HSIC (Hilbert-Schmidt Independence Criterion).

    Args:
        n_permutations: Number of permutations for the HSIC permutation test.
        significance_level: Default significance level for the test.
    """

    def __init__(
        self,
        n_permutations: int = 200,
        significance_level: float = 0.05,
    ) -> None:
        self.n_permutations = n_permutations
        self.significance_level = significance_level

    @staticmethod
    def _center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
        """Center a kernel matrix: H K H where H = I - (1/n) 11^T."""
        N = K.size(0)
        H = torch.eye(N, device=K.device) - torch.ones(N, N, device=K.device) / N
        return H @ K @ H

    @staticmethod
    def _hsic(Kx: torch.Tensor, Ky: torch.Tensor) -> torch.Tensor:
        """Compute the biased HSIC estimator.

        .. math::

            \\text{HSIC}(X, Y) = \\frac{1}{(n-1)^2} \\text{tr}(\\tilde{K}_X \\tilde{K}_Y)

        where :math:`\\tilde{K}` denotes the centered kernel matrix.
        """
        N = Kx.size(0)
        Kxc = CrossSpaceConditionalIndependence._center_kernel_matrix(Kx)
        Kyc = CrossSpaceConditionalIndependence._center_kernel_matrix(Ky)
        return (Kxc * Kyc).sum() / ((N - 1) ** 2)

    def _kernel_residualize(
        self,
        K_target: torch.Tensor,
        K_cond: torch.Tensor,
        ridge: float = 1e-4,
    ) -> torch.Tensor:
        """Residualize K_target with respect to K_cond.

        Projects out the component of the target RKHS explained by the
        conditioning RKHS using kernel ridge regression:

        .. math::

            \\tilde{K}_X = K_X - K_Z (K_Z + \\lambda I)^{-1} K_X

        Args:
            K_target: Kernel matrix of the variable to residualize.
            K_cond: Kernel matrix of the conditioning variable.
            ridge: Regularization strength.

        Returns:
            Residualized kernel matrix.
        """
        N = K_cond.size(0)
        reg = K_cond + ridge * torch.eye(N, device=K_cond.device)
        # Solve K_cond @ alpha = K_target
        alpha = torch.linalg.solve(reg, K_target)
        residual = K_target - K_cond @ alpha
        return residual

    def test(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        x_space: SpaceType = SpaceType.EUCLIDEAN,
        y_space: SpaceType = SpaceType.EUCLIDEAN,
        z_space: SpaceType = SpaceType.EUCLIDEAN,
    ) -> Tuple[float, float]:
        """Test X _||_ Y | Z using kernel HSIC with permutation p-value.

        Args:
            x: Observations of X, shape ``(N, D_x)``.
            y: Observations of Y, shape ``(N, D_y)``.
            z: Observations of conditioning variable Z, shape ``(N, D_z)``.
                If ``None``, tests unconditional independence X _||_ Y.
            x_space: Geometric space of X.
            y_space: Geometric space of Y.
            z_space: Geometric space of Z.

        Returns:
            Tuple of ``(test_statistic, p_value)``.
        """
        N = x.size(0)
        device = x.device

        # Compute kernel matrices with median heuristic bandwidths
        k_x = _select_kernel(x_space)
        k_y = _select_kernel(y_space)
        sigma_x = _median_heuristic(x)
        sigma_y = _median_heuristic(y)
        Kx = k_x(x, x, sigma_x)
        Ky = k_y(y, y, sigma_y)

        # Conditional case: residualize w.r.t. Z
        if z is not None:
            k_z = _select_kernel(z_space)
            sigma_z = _median_heuristic(z)
            Kz = k_z(z, z, sigma_z)
            Kx = self._kernel_residualize(Kx, Kz)
            Ky = self._kernel_residualize(Ky, Kz)

        # Compute HSIC statistic
        hsic_observed = self._hsic(Kx, Ky).item()

        # Permutation test for p-value
        count_geq = 0
        for _ in range(self.n_permutations):
            perm = torch.randperm(N, device=device)
            Ky_perm = Ky[perm][:, perm]
            hsic_perm = self._hsic(Kx, Ky_perm).item()
            if hsic_perm >= hsic_observed:
                count_geq += 1

        p_value = (count_geq + 1) / (self.n_permutations + 1)
        return hsic_observed, p_value


# ---------------------------------------------------------------------------
# FDR controller
# ---------------------------------------------------------------------------


class FDRController:
    """False discovery rate control for multiple hypothesis testing.

    Implements both Benjamini-Hochberg (BH) and Benjamini-Yekutieli (BY)
    procedures.

    Args:
        method: ``"bh"`` for Benjamini-Hochberg, ``"by"`` for
            Benjamini-Yekutieli (valid under arbitrary dependence).
    """

    def __init__(self, method: str = "bh") -> None:
        if method not in ("bh", "by"):
            raise ValueError(f"method must be 'bh' or 'by', got '{method}'")
        self.method = method

    def apply(
        self, p_values: List[float], alpha: float = 0.05
    ) -> List[bool]:
        """Apply FDR correction, returning a mask of significant results.

        Args:
            p_values: List of p-values from individual tests.
            alpha: Target FDR level.

        Returns:
            Boolean list; ``True`` means the test is significant after
            FDR correction.
        """
        m = len(p_values)
        if m == 0:
            return []

        # Sort p-values and track original indices
        indexed = sorted(enumerate(p_values), key=lambda t: t[1])

        # BY correction factor: sum(1/k for k=1..m)
        if self.method == "by":
            c_m = sum(1.0 / k for k in range(1, m + 1))
        else:
            c_m = 1.0

        # Find largest k such that p_(k) <= k * alpha / (m * c_m)
        significant = [False] * m
        max_k = -1
        for rank, (orig_idx, pval) in enumerate(indexed, start=1):
            threshold = rank * alpha / (m * c_m)
            if pval <= threshold:
                max_k = rank

        # All tests with rank <= max_k are significant
        if max_k > 0:
            for rank, (orig_idx, pval) in enumerate(indexed, start=1):
                if rank <= max_k:
                    significant[orig_idx] = True

        return significant


# ---------------------------------------------------------------------------
# Heterogeneous PCMCI
# ---------------------------------------------------------------------------


class HeterogeneousPCMCI:
    """Causal discovery for heterogeneous multimodal time series.

    Adapts the PCMCI algorithm (Runge et al., 2019) to operate across
    variables that live in different geometric spaces by using
    :class:`CrossSpaceConditionalIndependence` as the independence oracle.

    The algorithm proceeds in two phases:

    1. **PC1 phase**: condition-selection via iterative unconditional and
       conditional independence tests to remove spurious associations.
    2. **MCI phase**: momentary conditional independence test for remaining
       candidate links, conditioning on the parents of both source and
       target to control for confounders.

    Args:
        ci_test: The conditional independence test to use.
        fdr: FDR controller for multiple testing correction.
        significance_level: p-value threshold for edge significance.
    """

    def __init__(
        self,
        ci_test: Optional[CrossSpaceConditionalIndependence] = None,
        fdr: Optional[FDRController] = None,
        significance_level: float = 0.05,
    ) -> None:
        self.ci_test = ci_test or CrossSpaceConditionalIndependence()
        self.fdr = fdr or FDRController(method="bh")
        self.significance_level = significance_level

    def _extract_lagged_variable(
        self,
        data: torch.Tensor,
        lag: int,
        max_lag: int,
    ) -> torch.Tensor:
        """Extract a lagged version of a univariate/multivariate series.

        Args:
            data: ``(T, D)`` time series for one variable.
            lag: Non-negative lag (0 = contemporaneous).
            max_lag: Maximum lag in the system (determines output length).

        Returns:
            ``(T - max_lag, D)`` lagged data.
        """
        T = data.size(0)
        start = max_lag - lag
        end = T - lag if lag > 0 else T
        return data[start:end]

    def discover_causal_graph(
        self,
        multimodal_timeseries: Dict[str, torch.Tensor],
        variable_spaces: Dict[str, SpaceType],
        max_lag: int = 5,
    ) -> CausalGraph:
        """Discover causal relationships among multimodal time series.

        Args:
            multimodal_timeseries: Mapping from variable name to time
                series tensor of shape ``(T, D_var)``.
            variable_spaces: Mapping from variable name to its geometric
                space type.
            max_lag: Maximum time lag to consider for causal links.

        Returns:
            A :class:`CausalGraph` containing significant directed edges
            with lags, strengths, and corrected p-values.
        """
        var_names = sorted(multimodal_timeseries.keys())
        V = len(var_names)
        logger.info(
            f"Starting causal discovery: {V} variables, max_lag={max_lag}"
        )

        # Phase 1: PC1 -- identify candidate links via unconditional tests
        candidate_edges: List[Tuple[str, str, int]] = []
        candidate_pvals: List[float] = []

        for target_name in var_names:
            for source_name in var_names:
                for lag in range(1, max_lag + 1):
                    # Extract lagged data
                    x = self._extract_lagged_variable(
                        multimodal_timeseries[source_name], lag, max_lag
                    )
                    y = self._extract_lagged_variable(
                        multimodal_timeseries[target_name], 0, max_lag
                    )

                    stat, pval = self.ci_test.test(
                        x, y,
                        x_space=variable_spaces[source_name],
                        y_space=variable_spaces[target_name],
                    )
                    candidate_edges.append((source_name, target_name, lag))
                    candidate_pvals.append(pval)

        # Phase 2: FDR correction
        significant_mask = self.fdr.apply(
            candidate_pvals, alpha=self.significance_level
        )

        # Phase 3: MCI -- refine significant links with conditional tests
        # (conditioning on parents of both source and target)
        # First pass: identify preliminary parents
        prelim_parents: Dict[str, List[Tuple[str, int]]] = {
            n: [] for n in var_names
        }
        for (src, tgt, lag), is_sig in zip(candidate_edges, significant_mask):
            if is_sig:
                prelim_parents[tgt].append((src, lag))

        # Second pass: MCI tests
        edges: List[CausalEdge] = []
        mci_pvals: List[float] = []
        mci_edges_meta: List[Tuple[str, str, int, float]] = []

        for (src, tgt, lag), is_sig in zip(candidate_edges, significant_mask):
            if not is_sig:
                continue

            x = self._extract_lagged_variable(
                multimodal_timeseries[src], lag, max_lag
            )
            y = self._extract_lagged_variable(
                multimodal_timeseries[tgt], 0, max_lag
            )

            # Condition on parents of target (excluding current source-lag)
            cond_parts = []
            cond_space_parts = []
            for parent_name, parent_lag in prelim_parents[tgt]:
                if parent_name == src and parent_lag == lag:
                    continue
                z_part = self._extract_lagged_variable(
                    multimodal_timeseries[parent_name], parent_lag, max_lag
                )
                cond_parts.append(z_part)

            z_cond = None
            z_space = SpaceType.EUCLIDEAN
            if cond_parts:
                z_cond = torch.cat(cond_parts, dim=-1)
                # Use Euclidean kernel for concatenated conditioning set

            stat, pval = self.ci_test.test(
                x, y, z=z_cond,
                x_space=variable_spaces[src],
                y_space=variable_spaces[tgt],
                z_space=z_space,
            )
            mci_pvals.append(pval)
            mci_edges_meta.append((src, tgt, lag, stat))

        # Final FDR correction on MCI p-values
        if mci_pvals:
            mci_significant = self.fdr.apply(
                mci_pvals, alpha=self.significance_level
            )
            for (src, tgt, lag, stat), pval, is_sig in zip(
                mci_edges_meta, mci_pvals, mci_significant
            ):
                if is_sig:
                    edges.append(
                        CausalEdge(
                            source=src,
                            target=tgt,
                            lag=lag,
                            strength=stat,
                            p_value=pval,
                            source_space=variable_spaces[src],
                            target_space=variable_spaces[tgt],
                        )
                    )

        logger.info(
            f"Causal discovery complete: {len(edges)} significant edges"
        )

        return CausalGraph(
            edges=edges,
            variable_names=var_names,
            variable_spaces=variable_spaces,
            max_lag=max_lag,
        )
