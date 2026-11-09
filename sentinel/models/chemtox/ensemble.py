# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Ensembles of ChemTox models, scored on out-of-fold predictions only.

Two forms, both named as ensembles wherever they are reported (never as "the model"):

* :func:`average_ensemble` — the unweighted mean of the components' out-of-fold predictions;
* :func:`stacked_ensemble` — non-negative least-squares weights over the components, fitted
  **leave-one-fold-out**: the weights applied to fold *k* are fitted on folds != *k*, so no fold
  contributes to its own weights. This is the only leak-free way to score a stack from the same
  CV predictions the components were selected on; the weights per held-out fold are returned so
  the stack can be refitted on all folds when (and only when) the frozen configuration is scored
  on test.

Components must come from runs over the *same* challenge/split, aligned row-by-row on the
challenge file's row order, with NaN where a row was not part of any validation fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import evaluate as E


@dataclass
class EnsembleResult:
    """Out-of-fold ensemble predictions plus the per-component scores behind them."""

    name: str
    components: tuple[str, ...]
    pred: np.ndarray
    mask: np.ndarray
    metrics: dict = field(default_factory=dict)
    component_metrics: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "components": list(self.components), "metrics": self.metrics,
                "component_metrics": self.component_metrics, "weights": self.weights,
                "n_scored": int(self.mask.sum())}


def _stack(oof: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = sorted(oof)
    P = np.column_stack([np.asarray(oof[n], dtype=float) for n in names])
    return names, P


def common_mask(oof: dict[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    """Rows where the target and every component prediction are finite."""
    m = np.isfinite(np.asarray(y, dtype=float))
    for v in oof.values():
        m &= np.isfinite(np.asarray(v, dtype=float))
    return m


def _score(y, p, chem, taxon) -> dict:
    return E.report(y, p, chem=chem, taxon=taxon)


def average_ensemble(oof: dict[str, np.ndarray], y, *, chem, taxon, name: str = "ensemble_mean"):
    """Unweighted mean of component out-of-fold predictions."""
    names, P = _stack(oof)
    m = common_mask(oof, y)
    pred = np.full(len(np.asarray(y)), np.nan)
    pred[m] = P[m].mean(axis=1)
    y = np.asarray(y, dtype=float)
    return EnsembleResult(
        name=name, components=tuple(names), pred=pred, mask=m,
        metrics=_score(y[m], pred[m], np.asarray(chem)[m], np.asarray(taxon)[m]),
        component_metrics={n: _score(y[m], np.asarray(oof[n], dtype=float)[m],
                                     np.asarray(chem)[m], np.asarray(taxon)[m]) for n in names},
        weights={n: 1.0 / len(names) for n in names},
    )


def _nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls

    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(A.shape[1], 1.0 / A.shape[1])


def stacked_ensemble(
    oof: dict[str, np.ndarray],
    y,
    *,
    chem,
    taxon,
    folds: np.ndarray,
    name: str = "ensemble_stack_nnls",
) -> EnsembleResult:
    """Leave-one-fold-out non-negative stack of component out-of-fold predictions.

    ``folds`` is the per-row CV fold label (NaN/-1 for rows outside the folds).
    """
    names, P = _stack(oof)
    y = np.asarray(y, dtype=float)
    folds = np.asarray(folds)
    m = common_mask(oof, y) & np.isfinite(pd_numeric(folds))
    pred = np.full(len(y), np.nan)
    weights: dict[str, dict] = {}
    for k in sorted(set(pd_numeric(folds)[m].astype(int))):
        held = m & (pd_numeric(folds) == k)
        fit = m & (pd_numeric(folds) != k)
        w = _nnls(P[fit], y[fit])
        pred[held] = P[held] @ w
        weights[str(k)] = {n: float(wi) for n, wi in zip(names, w)}
    scored = m & np.isfinite(pred)
    return EnsembleResult(
        name=name, components=tuple(names), pred=pred, mask=scored,
        metrics=_score(y[scored], pred[scored], np.asarray(chem)[scored], np.asarray(taxon)[scored]),
        component_metrics={n: _score(y[scored], np.asarray(oof[n], dtype=float)[scored],
                                     np.asarray(chem)[scored], np.asarray(taxon)[scored])
                           for n in names},
        weights=weights,
    )


def pd_numeric(a) -> np.ndarray:
    """Fold labels as float (strings and None become NaN)."""
    import pandas as pd

    return pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
