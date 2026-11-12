"""The monotone saturating time law and the endpoint-ordering head shared by TED and TimeHead.

New module. It is the *design claim* of the M4 arch-v2 candidates, so every property the
pre-registration asserts is enforced in the arithmetic here and tested in
``tests/test_chemtox_arch_v2_timelaw.py`` rather than argued for in prose. No I/O, no project row:
everything below is a ``torch.nn.Module`` or a pure function over tensors.

Governing text in precedence order: ``chemtox_arch_v2_lead_rulings.md`` (sha256 ``553be413…``),
``chemtox_arch_v2_addendum_01.md``, the stamped amendment (§4.1 of the architecture plan as carried
into amendment §7.2).

The law
-------
With an incipient (long-duration asymptotic) value ``y_inf`` and a per-row rate ``k > 0``::

    log10 LC50(t) = y_inf - log10(1 - exp(-k t))

implemented as ``y_inf - LOG10E * log1p(-exp(-k t))`` through :func:`saturation_term`, which uses
``expm1``/``log1p`` so that small ``k t`` does not lose precision. Three properties follow
analytically and are asserted by tests:

* **monotone non-increasing in t** — ``1 - exp(-k t)`` rises with ``t``, so its negative log falls;
* **y_inf is the asymptote** — the term tends to 0 as ``k t`` grows;
* **a shape, not a step** — beyond the last observed duration the prediction keeps following the
  law instead of flattening onto a leaf mean, which is the one testable advantage over R1.

The rate
--------
``log10 k = (a0 + a1 logKow + a2 log10 MW) + B * tanh(u)``, where ``a0..a2`` are **initialised by
least squares on the training duration series only** (:func:`anchor_least_squares`) and ``B`` is a
declared bound in decades. ``B = 0`` is a pure anchored QSAR — that is ablation **A2** — and large
``B`` is an unconstrained net, so the ablation sweep measures what the mechanism is worth.

The endpoint ordering
---------------------
``log10 EC10 = y_inf - softplus(u_acr) - log10(1 - exp(-k_chr t))`` with
``k_chr = k * (1 + softplus(v))``. Because ``-log10(1 - exp(-k t))`` is **decreasing in k**, forcing
``k_chr >= k`` makes the chronic saturation term no larger than the acute one, and ``softplus > 0``
then makes ``EC10 <= EC50`` hold **for every parameter value and every t**, not merely in
expectation. ``log10 ACR = softplus(u_acr)`` is therefore non-negative by construction and is itself
a reported output. Within the chronic family :func:`chronic_chain` enforces
``NOEC <= MATC <= LOEC`` by a cumulative softplus.

Numerics
--------
``expm1`` near zero is destroyed by fp16, so :func:`saturation_term` force-casts to fp32 inside an
``autocast``-disabled block (:func:`fp32_block`) and a test asserts parity against an fp64
reference. ``k t`` is clamped to :data:`KT_MIN`/:data:`KT_MAX` so neither tail can produce ``inf``
or ``nan`` — with the clamp recorded, because it bounds how far the law may be extrapolated before
it saturates numerically.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

__all__ = ["LOG10E", "KT_MIN", "KT_MAX", "DEFAULT_B_DECADES", "fp32_block", "saturation_term",
           "anchor_least_squares", "RateHead", "AcuteHead", "AcuteChronicHead", "chronic_chain",
           "TimeLawError"]

#: ``log10(x) = LOG10E * ln(x)``
LOG10E = 1.0 / math.log(10.0)

#: ``k t`` is clamped to this range. The lower bound keeps ``-log10(-expm1(-kt))`` finite at the
#: short-duration tail; the upper bound is where the saturation term is already < 1e-9 decades, so
#: clamping it changes no prediction at single precision.
KT_MIN, KT_MAX = 1e-9, 40.0

#: The declared bound, in decades, on the learned residual of ``log10 k``.
DEFAULT_B_DECADES = 1.5


class TimeLawError(ValueError):
    """A pre-registered property of the time law was violated by the caller."""


class fp32_block:
    """Disable autocast and work in fp32: ``expm1``/``log1p`` near zero need the mantissa.

    This is the single most important implementation detail of the composition layer, so it is a
    named context manager rather than a comment.
    """

    def __enter__(self):
        self._ctx = torch.amp.autocast("cuda", enabled=False)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def saturation_term(k: Tensor, t: Tensor) -> Tensor:
    """``-log10(1 - exp(-k t))``, non-negative and non-increasing in ``t``.

    Computed in fp32 (or better) with ``log1p``/``expm1``; ``k t`` is clamped to
    ``[KT_MIN, KT_MAX]``.
    """
    if torch.any(k <= 0):
        raise TimeLawError("the rate k must be strictly positive")
    if torch.any(t <= 0):
        raise TimeLawError("the duration t must be strictly positive")
    with fp32_block():
        kk = k.to(torch.float32) if k.dtype not in (torch.float32, torch.float64) else k
        tt = t.to(torch.float32) if t.dtype not in (torch.float32, torch.float64) else t
        kt = torch.clamp(kk * tt, min=KT_MIN, max=KT_MAX)
        # 1 - exp(-kt) MUST be formed as -expm1(-kt), never as 1 - exp(-kt) or log1p(-exp(-kt)):
        # at kt ~ 1e-4 the latter has already lost most of the mantissa, and at kt ~ 1e-9 fp32
        # rounds exp(-kt) to exactly 1.0 and the term becomes -inf. expm1 is exact in that tail.
        # The two tests test_saturation_term_matches_an_fp64_reference and
        # ..._is_finite_at_both_extremes exist because the first version of this line got it wrong.
        one_minus = torch.clamp(-torch.expm1(-kt), min=torch.finfo(torch.float32).tiny)
        return -LOG10E * torch.log(one_minus)


def anchor_least_squares(log_kow: np.ndarray, log10_mw: np.ndarray,
                         log10_k: np.ndarray) -> tuple[float, float, float]:
    """Least-squares anchor ``a0, a1, a2`` for ``log10 k ~ a0 + a1 logKow + a2 log10 MW``.

    Fitted on the **training** duration series only, which is what makes the anchor a prior rather
    than a leak. Returns zeros for the slopes when the system is degenerate (a single distinct
    chemical), so a tiny fold cannot produce a wild anchor.
    """
    x1 = np.asarray(log_kow, dtype=float).ravel()
    x2 = np.asarray(log10_mw, dtype=float).ravel()
    y = np.asarray(log10_k, dtype=float).ravel()
    if not (x1.size == x2.size == y.size) or y.size == 0:
        raise TimeLawError("anchor_least_squares needs three equal-length non-empty arrays")
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x1[ok]).size < 2:
        return (float(np.mean(y[ok])) if ok.any() else 0.0, 0.0, 0.0)
    A = np.column_stack([np.ones(ok.sum()), x1[ok], x2[ok]])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


class RateHead(nn.Module):
    """``log10 k = (a0 + a1 logKow + a2 log10 MW) + B * tanh(u(z))``.

    ``B = 0`` is ablation **A2**: the pure anchored law with no learned residual.
    """

    def __init__(self, latent_dim: int, *, b_decades: float = DEFAULT_B_DECADES,
                 anchor: tuple[float, float, float] = (0.0, 0.0, 0.0), hidden: int = 32) -> None:
        super().__init__()
        if b_decades < 0:
            raise TimeLawError("b_decades must be non-negative")
        self.b_decades = float(b_decades)
        self.register_buffer("anchor", torch.tensor(anchor, dtype=torch.float32))
        self.u = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def set_anchor(self, a0: float, a1: float, a2: float) -> None:
        with torch.no_grad():
            self.anchor.copy_(torch.tensor([a0, a1, a2], dtype=self.anchor.dtype))

    def forward(self, z: Tensor, log_kow: Tensor, log10_mw: Tensor) -> Tensor:
        a0, a1, a2 = self.anchor[0], self.anchor[1], self.anchor[2]
        base = a0 + a1 * log_kow.reshape(-1) + a2 * log10_mw.reshape(-1)
        if self.b_decades == 0.0:
            return base
        return base + self.b_decades * torch.tanh(self.u(z).reshape(-1))

    def rate(self, z: Tensor, log_kow: Tensor, log10_mw: Tensor) -> Tensor:
        """``k`` itself, strictly positive."""
        with fp32_block():
            return torch.pow(torch.tensor(10.0, device=z.device, dtype=torch.float32),
                             self.forward(z, log_kow, log10_mw).to(torch.float32))


class AcuteHead(nn.Module):
    """``log10 LC50(t) = y_inf(z) - log10(1 - exp(-k t))``: the acute composition layer."""

    def __init__(self, latent_dim: int, *, b_decades: float = DEFAULT_B_DECADES,
                 anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.y_inf = nn.Linear(latent_dim, 1)
        self.rate_head = RateHead(latent_dim, b_decades=b_decades, anchor=anchor)

    def forward(self, z: Tensor, t: Tensor, log_kow: Tensor, log10_mw: Tensor) -> Tensor:
        k = self.rate_head.rate(z, log_kow, log10_mw)
        with fp32_block():
            return self.y_inf(z).reshape(-1).to(torch.float32) + saturation_term(k, t.reshape(-1))


class AcuteChronicHead(nn.Module):
    """Acute and chronic from ONE latent, with ``EC10 <= EC50`` true by construction.

    ``k_chr = k * (1 + softplus(v))`` so ``k_chr >= k``; since the saturation term is decreasing in
    ``k``, the chronic term cannot exceed the acute one, and the extra ``-softplus(u_acr)`` then
    makes the ordering strict. ``log10 ACR = softplus(u_acr) >= 0`` is a reported output.
    """

    def __init__(self, latent_dim: int, *, b_decades: float = DEFAULT_B_DECADES,
                 anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.acute = AcuteHead(latent_dim, b_decades=b_decades, anchor=anchor)
        self.u_acr = nn.Linear(latent_dim, 1)
        self.v_chr = nn.Linear(latent_dim, 1)

    def log10_acr(self, z: Tensor) -> Tensor:
        """The acute-to-chronic ratio in decades, non-negative by construction (target H5)."""
        with fp32_block():
            return nn.functional.softplus(self.u_acr(z).reshape(-1).to(torch.float32))

    def forward(self, z: Tensor, t: Tensor, log_kow: Tensor, log10_mw: Tensor) -> dict:
        tt = t.reshape(-1)
        k = self.acute.rate_head.rate(z, log_kow, log10_mw)
        with fp32_block():
            y_inf = self.acute.y_inf(z).reshape(-1).to(torch.float32)
            acute = y_inf + saturation_term(k, tt)
            k_chr = k * (1.0 + nn.functional.softplus(self.v_chr(z).reshape(-1).to(torch.float32)))
            acr = nn.functional.softplus(self.u_acr(z).reshape(-1).to(torch.float32))
            chronic = y_inf - acr + saturation_term(k_chr, tt)
        return {"acute": acute, "chronic": chronic, "log10_acr": acr, "k": k, "k_chronic": k_chr}


def chronic_chain(base: Tensor, deltas: Tensor) -> dict:
    """``NOEC <= MATC <= LOEC`` by a cumulative softplus on two non-negative increments.

    ``base`` is ``log10 NOEC``; ``deltas`` has two columns whose softplus gives the gaps.
    """
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        raise TimeLawError("chronic_chain needs a (n, 2) delta tensor")
    with fp32_block():
        b = base.reshape(-1).to(torch.float32)
        d = nn.functional.softplus(deltas.to(torch.float32))
        noec = b
        matc = noec + d[:, 0]
        loec = matc + d[:, 1]
    return {"noec": noec, "matc": matc, "loec": loec}
