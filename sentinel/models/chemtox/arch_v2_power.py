"""Power / MDE for the M4 architecture-round-v2 tracks T1 and T6 (lead ruling L5).

Why this module exists
----------------------
``unseen_taxa.mde`` draws the two synthetic predictors' errors **independently per row**. Under a
cluster bootstrap that makes the sampling noise of the paired RMSE difference scale with the number
of *rows*, not the number of resampling *units*, so the minimum detectable effect it reports is too
small — on the v1 T1/T6 blocks by a factor of 2.5-5x (verifier finding ``MDE-ROW-INDEPENDENT``).
Errors of two competing toxicity models are in fact strongly shared within a chemical cluster: the
same unrepresented scaffold is mispredicted the same way for every row that uses it.

This module adds the missing within-unit error component, parameterised by ``icc`` — the
intra-class correlation of the two models' *errors* inside a resampling unit:

===========  ============================================================================
``icc = 0``  the v1 behaviour: errors independent across rows within a unit (optimistic)
``icc = 0.5``  half the error variance is a unit-level offset (the realistic middle)
``icc = 1``  the error is constant inside a unit, so the effective sample size is the
             number of units (conservative, and the bound a cluster-level claim needs)
===========  ============================================================================

``sentinel/models/chemtox/unseen_taxa.py`` and ``unseen_taxa_guard.py`` are frozen T2 code and are
**not edited**; this is a new module and T2's figures are untouched.

Label-free by construction
--------------------------
Every function here takes **counts only** — a vector of resampling-unit sizes — and never a frame,
a target or a prediction. The statistic is a paired RMSE difference between two synthetic
predictors, and for predictor ``p = y + e`` the residual ``p - y`` is ``e``, so the observed target
cancels exactly and cannot enter. This is what lets the v2 power table be computed before any
``FREEZE`` file exists (lead ruling L6).

The model
---------
For row ``i`` in unit ``u``, with total error SD ``s``::

    a_ui = sqrt(icc) * g_u      + sqrt(1 - icc) * z_ui          (model A, unit var 1)
    b_ui = sqrt(icc) * g'_u     + sqrt(1 - icc) * z'_ui          (model B, unit var 1)
    g'_u = rho * g_u  + sqrt(1 - rho^2) * h_u
    z'_ui= rho * z_ui + sqrt(1 - rho^2) * w_ui
    eA = s * a_ui,  eB = s * b_ui

so each row's two errors are marginally N(0, s^2) and correlated at ``rho`` whatever ``icc`` is:
``icc`` moves error *structure*, never error *magnitude*. That separation is what makes the icc
sweep an honest sensitivity analysis rather than a rescaling.

Closed form
-----------
With ``D = eA^2 - eB^2`` and the delta-method step ``RMSE_A - RMSE_B ~= (MSE_A - MSE_B) / (2 s)``,
and using ``Var(D_i) = 4 s^4 (1 - rho^2)`` and ``Cov(D_i, D_j) = 4 s^4 icc^2 (1 - rho^2)`` for two
rows in the same unit, the variance is exact for **unequal** unit sizes ``n_u``::

    SD(d) = s * sqrt(1 - rho^2) * sqrt( sum_u n_u * (1 + icc^2 * (n_u - 1)) ) / N
    MDE   = z_(1-alpha/2) * SD(d)

For equal sizes this is ``s sqrt(1-rho^2) sqrt((1 + icc^2 (nbar-1)) / N)``, which collapses to the
row-level ``z s sqrt(1-rho^2)/sqrt(N)`` at ``icc = 0`` (what ``unseen_taxa.mde`` estimates) and to
the unit-level ``z s sqrt(1-rho^2)/sqrt(U)`` at ``icc = 1``. Using the exact unequal-size sum
rather than the mean unit size matters here and always in the conservative direction: one T1/T6
hold-out unit holds about 10 % of the universe's rows, and the equal-size approximation understates
the MDE of a block containing it.

Note the design effect carries ``icc**2``, not ``icc``: the statistic is built from *squared*
errors, and the within-unit correlation of squared errors is the square of the within-unit
correlation of the errors (``Cov(a_i^2, a_j^2) = 2 icc^2`` for standard normals correlated at
``icc``).

``mde_bootstrap`` is kept as an independent check, but it conditions on **one** realisation of the
error vectors, which at high ``icc`` and unequal unit sizes makes a single call depend strongly on
which unit-level draws landed on the large units. It therefore averages over ``n_sim`` independent
error realisations; the pre-registered figure the amendment quotes is the analytic one, which is
deterministic. ``tests/test_chemtox_arch_v2_power.py`` asserts both limits and checks the formula
against Monte Carlo at intermediate ``icc``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = ["mde_closed_form", "mde_bootstrap", "mde_table", "design_effect",
           "effective_units", "effective_sample_size", "z_two_sided", "DEFAULT_RMSE_LEVEL",
           "DEFAULT_RHO", "ICC_GRID"]

#: The error magnitude this project actually produces on a hold-out block (log10 mol/L). Taken
#: from the spent prospective pass (1.21-1.24) and T2's selection range (1.10-1.22), and the same
#: value T2's protocol used, so T1/T6 power is stated on the same assumption.
DEFAULT_RMSE_LEVEL = 1.2

#: Competing models' errors are strongly correlated; 0.9 is T2's pre-registered value. rho = 0
#: is the pessimistic bound and is reported beside it.
DEFAULT_RHO = 0.9

#: The icc values the amendment tabulates. 0.0 is reported only to show what v1 assumed.
ICC_GRID = (0.0, 0.5, 1.0)


def z_two_sided(alpha: float = 0.05) -> float:
    """Two-sided normal quantile. 1.959963985... at alpha = 0.05."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    # Acklam/Moro-free: bisection on erf is exact enough and keeps scipy out of the import closure.
    lo, hi = 0.0, 40.0
    target = 1.0 - alpha / 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _sizes(unit_sizes: Iterable[int]) -> np.ndarray:
    s = np.asarray(list(unit_sizes), dtype=np.int64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("unit_sizes must be a non-empty 1-D sequence of row counts")
    if (s <= 0).any():
        raise ValueError("every resampling unit must hold at least one row")
    return s


def design_effect(unit_sizes: Iterable[int], icc: float) -> float:
    """``1 + icc**2 * (nbar - 1)`` on the mean unit size. 1.0 at icc = 0."""
    s = _sizes(unit_sizes)
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must be in [0, 1], got {icc}")
    nbar = float(s.sum()) / float(s.size)
    return 1.0 + (icc ** 2) * (nbar - 1.0)


def effective_units(unit_sizes: Iterable[int], icc: float) -> float:
    """Effective sample size ``N / design_effect``: N rows at icc = 0, U units at icc = 1."""
    s = _sizes(unit_sizes)
    return float(s.sum()) / design_effect(s, icc)


def mde_closed_form(unit_sizes: Iterable[int], *, rmse_level: float = DEFAULT_RMSE_LEVEL,
                    rho: float = DEFAULT_RHO, icc: float = 0.0, alpha: float = 0.05) -> float:
    """Closed-form MDE (half-width of the paired-difference interval). Exact for unequal units.

    Counts only: ``unit_sizes`` is a vector of row counts per resampling unit. This is the figure
    the amendment pre-registers, because it is deterministic.
    """
    s = _sizes(unit_sizes)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must be in [0, 1], got {icc}")
    n = float(s.sum())
    f = s.astype(float)
    var_sum = float((f * (1.0 + (icc ** 2) * (f - 1.0))).sum())
    return (z_two_sided(alpha) * rmse_level * math.sqrt(1.0 - rho * rho)
            * math.sqrt(var_sum) / n)


def effective_sample_size(unit_sizes: Iterable[int], icc: float) -> float:
    """``N^2 / sum_u n_u (1 + icc^2 (n_u - 1))``: N rows at icc = 0, U units at icc = 1 (equal n)."""
    s = _sizes(unit_sizes).astype(float)
    n = float(s.sum())
    return n * n / float((s * (1.0 + (icc ** 2) * (s - 1.0))).sum())


def mde_bootstrap(unit_sizes: Iterable[int], *, rmse_level: float = DEFAULT_RMSE_LEVEL,
                  rho: float = DEFAULT_RHO, icc: float = 0.0, n_boot: int = 2000,
                  seed: int = 0, alpha: float = 0.05, n_sim: int = 8) -> dict:
    """Cluster-bootstrap MDE, averaged over ``n_sim`` error realisations. Counts only.

    Units are resampled with replacement ``n_boot`` times per realisation; the MDE is the mean
    over realisations of the half-width of the ``1 - alpha`` percentile interval of
    ``RMSE_A - RMSE_B``. Averaging matters at high ``icc`` with unequal units, where a single
    realisation's answer depends on which unit-level draws landed on the large units.
    """
    s = _sizes(unit_sizes)
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must be in [0, 1], got {icc}")
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    if n_sim < 1:
        raise ValueError("n_sim must be at least 1")
    rng = np.random.default_rng(seed)
    n_units, n_rows = int(s.size), int(s.sum())
    unit_of = np.repeat(np.arange(n_units), s)
    idx = np.concatenate(([0], np.cumsum(s)))
    wu, wr = math.sqrt(icc), math.sqrt(max(0.0, 1.0 - icc))
    c = math.sqrt(1.0 - rho * rho)

    halves = np.empty(n_sim)
    for k in range(n_sim):
        g = rng.normal(size=n_units)
        h = rng.normal(size=n_units)
        z = rng.normal(size=n_rows)
        w = rng.normal(size=n_rows)
        eA = rmse_level * (wu * g[unit_of] + wr * z)
        eB = rmse_level * (wu * (rho * g + c * h)[unit_of] + wr * (rho * z + c * w))
        sumA = np.add.reduceat(eA * eA, idx[:-1])
        sumB = np.add.reduceat(eB * eB, idx[:-1])
        draws = rng.integers(0, n_units, size=(n_boot, n_units))
        tot = s[draws].sum(axis=1).astype(float)
        d = (np.sqrt(sumA[draws].sum(axis=1) / tot)
             - np.sqrt(sumB[draws].sum(axis=1) / tot))
        lo, hi = np.quantile(d, [alpha / 2.0, 1.0 - alpha / 2.0])
        halves[k] = (hi - lo) / 2.0
    return {"n_rows": n_rows, "n_units": n_units,
            "mean_unit_size": round(n_rows / n_units, 4),
            "max_unit_size": int(s.max()),
            "max_unit_row_share": round(float(s.max()) / n_rows, 4),
            "icc": icc, "rho": rho, "rmse_level": rmse_level, "n_boot": n_boot, "seed": seed,
            "n_sim": n_sim,
            "design_effect_mean_size": round(design_effect(s, icc), 4),
            "effective_sample_size": round(effective_sample_size(s, icc), 1),
            "mde_bootstrap": float(halves.mean()),
            "mde_bootstrap_sd_over_sims": float(halves.std(ddof=1)) if n_sim > 1 else 0.0,
            "mde_closed_form": mde_closed_form(s, rmse_level=rmse_level, rho=rho, icc=icc,
                                               alpha=alpha),
            "label_free": True}


def mde_table(unit_sizes: Iterable[int], *, rmse_level: float = DEFAULT_RMSE_LEVEL,
              rhos: Sequence[float] = (DEFAULT_RHO, 0.0),
              iccs: Sequence[float] = ICC_GRID, n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05, n_sim: int = 8) -> list[dict]:
    """``mde_bootstrap`` over an icc x rho grid, for one block. Counts only."""
    s = _sizes(unit_sizes)
    return [mde_bootstrap(s, rmse_level=rmse_level, rho=r, icc=i, n_boot=n_boot, seed=seed,
                          alpha=alpha, n_sim=n_sim)
            for i in iccs for r in rhos]
