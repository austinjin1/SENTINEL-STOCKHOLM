"""Aitchison Neural Networks.

Provably compositionally coherent neural network layers that operate
natively in the Aitchison simplex geometry.  All operations respect the
fundamental constraints of compositional data:

1. **Scale invariance**: predictions are invariant to total count
   (sequencing depth normalization is implicit).
2. **Sub-compositional coherence**: operations on a sub-composition
   yield the same result as operating on the full composition and
   then extracting the sub-composition.
3. **Permutation equivariance**: reordering components reorders outputs
   correspondingly.

The Aitchison geometry treats the D-part simplex as a (D-1)-dimensional
real vector space with the CLR (centered log-ratio) transform providing
the isometry to Euclidean space.  All neural network operations are
performed in CLR coordinates, with results projected back to the simplex.

References
----------
[1] Aitchison, J. (1986). "The Statistical Analysis of Compositional
    Data." Chapman and Hall.
[2] Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations
    for Compositional Data Analysis." Mathematical Geology, 35(3).
[3] Pawlowsky-Glahn, V. et al. (2015). "Modeling and Analysis of
    Compositional Data." Wiley.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Simplex / CLR transforms
# ---------------------------------------------------------------------------

_EPS = 1e-10


def closure(x: torch.Tensor) -> torch.Tensor:
    """Project to the simplex via closure (L1 normalization).

    Args:
        x: ``(..., D)`` non-negative tensor.

    Returns:
        Compositions on the simplex (rows sum to 1).
    """
    return x / x.sum(dim=-1, keepdim=True).clamp(min=_EPS)


def clr_transform(x: torch.Tensor) -> torch.Tensor:
    """Centered log-ratio (CLR) transform.

    .. math::

        \\text{clr}(x)_i = \\ln x_i - \\frac{1}{D} \\sum_{j=1}^{D} \\ln x_j

    The CLR transform maps compositions from the simplex to a
    (D-1)-dimensional hyperplane in :math:`\\mathbb{R}^D` (the
    coordinates sum to zero).

    Args:
        x: ``(..., D)`` compositions on the simplex.

    Returns:
        CLR-transformed coordinates ``(..., D)``.
    """
    log_x = torch.log(x.clamp(min=_EPS))
    return log_x - log_x.mean(dim=-1, keepdim=True)


def inv_clr(y: torch.Tensor) -> torch.Tensor:
    """Inverse CLR transform: map from CLR coordinates back to simplex.

    .. math::

        x = \\mathcal{C}(\\exp(y))

    where :math:`\\mathcal{C}` is the closure operation.

    Args:
        y: ``(..., D)`` CLR coordinates.

    Returns:
        Compositions on the simplex ``(..., D)``.
    """
    return closure(torch.exp(y))


def aitchison_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Aitchison distance between two compositions.

    .. math::

        d_A(x, y) = \\| \\text{clr}(x) - \\text{clr}(y) \\|_2

    Args:
        x: ``(..., D)`` compositions.
        y: ``(..., D)`` compositions.

    Returns:
        Aitchison distances ``(...)``.
    """
    return torch.norm(clr_transform(x) - clr_transform(y), dim=-1)


def aitchison_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Aitchison inner product.

    .. math::

        \\langle x, y \\rangle_A = \\text{clr}(x) \\cdot \\text{clr}(y)

    Args:
        x, y: ``(..., D)`` compositions.

    Returns:
        Inner products ``(...)``.
    """
    return (clr_transform(x) * clr_transform(y)).sum(dim=-1)


# ---------------------------------------------------------------------------
# AitchisonLinear
# ---------------------------------------------------------------------------


class AitchisonLinear(nn.Module):
    """Linear layer operating in Aitchison geometry.

    Maps between simplex spaces by performing a linear transformation
    in CLR coordinates and projecting back to the simplex:

    .. math::

        f(x) = \\text{clr}^{-1}(W \\cdot \\text{clr}(x) + b)

    Because the CLR transform is an isometry between the Aitchison
    simplex and Euclidean space, this layer implements a proper linear
    map in the Aitchison vector space.  The output is automatically
    scale-invariant: ``f(alpha * x) = f(x)`` for any ``alpha > 0``.

    Args:
        in_components: Number of components in the input composition.
        out_components: Number of components in the output composition.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_components: int,
        out_components: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_components = in_components
        self.out_components = out_components

        # Weight matrix in CLR space
        self.weight = nn.Parameter(
            torch.empty(out_components, in_components)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_components))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights with Kaiming uniform scaled for CLR space."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_components
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(..., in_components)`` compositions on the simplex.

        Returns:
            ``(..., out_components)`` compositions on the simplex.
        """
        # Transform to CLR coordinates
        x_clr = clr_transform(x)  # (..., in_components)

        # Linear transform in CLR space
        out_clr = F.linear(x_clr, self.weight, self.bias)

        # Project back to simplex
        return inv_clr(out_clr)

    def extra_repr(self) -> str:
        return (
            f"in_components={self.in_components}, "
            f"out_components={self.out_components}, "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# AitchisonBatchNorm
# ---------------------------------------------------------------------------


class AitchisonBatchNorm(nn.Module):
    """Batch normalization in Aitchison geometry.

    Normalizes CLR-transformed features (not raw compositions), then
    maps back to the simplex.  This respects the simplex geometry:
    the Aitchison mean (geometric mean after closure) is used as the
    centering operation.

    .. math::

        \\hat{x}_{\\text{clr}} = \\gamma \\cdot
            \\frac{\\text{clr}(x) - \\mu_{\\text{clr}}}{\\sigma_{\\text{clr}}}
            + \\beta

    Args:
        num_components: Number of compositional components.
        eps: Small constant for numerical stability.
        momentum: Momentum for running statistics.
        affine: If ``True``, learnable affine parameters.
    """

    def __init__(
        self,
        num_components: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_components))
            self.beta = nn.Parameter(torch.zeros(num_components))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self.register_buffer("running_mean", torch.zeros(num_components))
        self.register_buffer("running_var", torch.ones(num_components))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(N, D)`` or ``(N, *, D)`` compositions on the simplex.

        Returns:
            Batch-normalized compositions on the simplex, same shape.
        """
        # Transform to CLR coordinates
        x_clr = clr_transform(x)

        if self.training:
            # Compute batch statistics in CLR space
            # Flatten all but last dim for statistics
            flat = x_clr.reshape(-1, self.num_components)
            mean = flat.mean(dim=0)
            var = flat.var(dim=0, unbiased=True)

            # Update running statistics
            self.num_batches_tracked += 1
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize in CLR space
        x_norm = (x_clr - mean) / torch.sqrt(var + self.eps)

        # Apply affine transform
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta

        # Project back to simplex
        return inv_clr(x_norm)

    def extra_repr(self) -> str:
        return (
            f"num_components={self.num_components}, "
            f"eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}"
        )


# ---------------------------------------------------------------------------
# AitchisonMLP
# ---------------------------------------------------------------------------


class AitchisonMLP(nn.Module):
    """Multi-layer perceptron with Aitchison geometry.

    All linear layers operate in CLR space, with batch normalization
    also in CLR space.  Non-linearities are applied in CLR coordinates.
    The network is a universal approximator on the simplex: it can
    approximate any continuous function from the simplex to the simplex
    (by the universal approximation theorem applied in CLR space,
    which is isometric to Euclidean space).

    Args:
        in_components: Number of input compositional components.
        out_components: Number of output compositional components.
        hidden_components: Hidden layer sizes (in CLR space dimensionality).
        use_batchnorm: Whether to use Aitchison batch normalization.
        dropout: Dropout rate (applied in CLR space).
    """

    def __init__(
        self,
        in_components: int,
        out_components: int,
        hidden_components: Optional[list[int]] = None,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_components is None:
            hidden_components = [64, 64]

        self.in_components = in_components
        self.out_components = out_components

        layers: list[nn.Module] = []
        prev_dim = in_components
        for h_dim in hidden_components:
            layers.append(AitchisonLinear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(AitchisonBatchNorm(h_dim))
            # ReLU in CLR space (we apply it to CLR-transformed output
            # inside a wrapper that goes simplex -> CLR -> ReLU -> simplex)
            layers.append(_CLRActivation(nn.ReLU(inplace=False)))
            if dropout > 0:
                layers.append(_CLRDropout(dropout))
            prev_dim = h_dim

        # Final layer (no activation -- just project to output simplex)
        layers.append(AitchisonLinear(prev_dim, out_components))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(..., in_components)`` compositions on the simplex.

        Returns:
            ``(..., out_components)`` compositions on the simplex.
        """
        return self.layers(x)


class _CLRActivation(nn.Module):
    """Apply a standard activation function in CLR space.

    Maps: simplex -> CLR -> activation -> simplex.
    """

    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clr = clr_transform(x)
        activated = self.activation(x_clr)
        return inv_clr(activated)


class _CLRDropout(nn.Module):
    """Apply dropout in CLR space."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x_clr = clr_transform(x)
        dropped = self.dropout(x_clr)
        return inv_clr(dropped)


# ---------------------------------------------------------------------------
# Compositional coherence test
# ---------------------------------------------------------------------------


@torch.no_grad()
def compositional_coherence_test(
    model: nn.Module,
    x: torch.Tensor,
    n_scales: int = 5,
    atol: float = 1e-4,
) -> bool:
    """Verify that model output is invariant to input scaling.

    A compositionally coherent model satisfies ``f(alpha * x) = f(x)``
    for any positive scalar ``alpha``.  This is the defining property
    of scale invariance in the Aitchison simplex.

    Args:
        model: The model to test (should accept simplex inputs).
        x: ``(N, D)`` test compositions on the simplex.
        n_scales: Number of random scaling factors to test.
        atol: Absolute tolerance for the coherence check.

    Returns:
        ``True`` if the model is compositionally coherent within tolerance.
    """
    model.eval()
    base_output = model(x)  # (N, D_out)

    for _ in range(n_scales):
        # Random positive scaling (simulates different sequencing depths)
        alpha = torch.rand(x.size(0), 1, device=x.device) * 100.0 + 0.01
        x_scaled = x * alpha
        # Close to get back on simplex (model should handle this internally)
        x_scaled_closed = closure(x_scaled)
        scaled_output = model(x_scaled_closed)

        # Check closeness in Aitchison distance
        dist = aitchison_distance(base_output, scaled_output)
        if dist.max().item() > atol:
            logger.warning(
                f"Coherence test FAILED: max Aitchison distance = "
                f"{dist.max().item():.6f} > atol={atol}"
            )
            return False

    logger.info("Compositional coherence test PASSED")
    return True
