# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""bio.infer-style maximum-likelihood inference of an environmental variable from taxon occurrences.

Re-implementation (not the R package) of the US EPA taxon–environment inference of Yuan (2007,
*Freshwater Biology*; R package ``bio.infer``): for every taxon with >= ``min_occ`` occurrences in
the training samples, a logistic regression of presence/absence on the environmental variable with
linear and quadratic terms (the taxon–environment relationship); for a new sample, the inferred
value is the grid value that maximises the likelihood of its observed presences *and* absences over
the modelled taxa (uniform prior over the training range). Inputs are only taxon occurrences (no
geography, no descriptors). Used post hoc (2026-11-09) as a published-method comparator for M5.

Settings are fixed a priori (``min_occ=30``, 200-point grid, near-unpenalised logistic fits with an
L2 term ``C=100`` on the standardised scale only to keep separable taxa finite); nothing is tuned.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class BioInferML:
    def __init__(self, min_occ: int = 30, n_grid: int = 200, C: float = 100.0):
        self.min_occ, self.n_grid, self.C = int(min_occ), int(n_grid), float(C)

    def fit(self, P: np.ndarray, y: np.ndarray) -> "BioInferML":
        """``P``: (n, T) presence/absence of candidate taxa in training samples; ``y``: (n,) variable."""
        P = np.asarray(P, bool)
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        P, y = P[ok], y[ok]
        self.mu_, self.sd_ = float(y.mean()), float(y.std() or 1.0)
        z = (y - self.mu_) / self.sd_
        X = np.column_stack([z, z ** 2])
        occ = P.sum(0)
        keep = np.flatnonzero((occ >= self.min_occ) & (occ <= len(y) - self.min_occ))
        coef = np.zeros((len(keep), 3))
        for j, k in enumerate(keep):
            m = LogisticRegression(C=self.C, max_iter=500).fit(X, P[:, k])
            coef[j] = [m.intercept_[0], m.coef_[0, 0], m.coef_[0, 1]]
        self.taxa_idx_, self.coef_ = keep, coef
        self.grid_ = np.linspace(z.min(), z.max(), self.n_grid)
        return self

    def log_likelihood(self, P: np.ndarray) -> np.ndarray:
        """(n, n_grid) log-likelihood of each sample's presences/absences at each grid value."""
        Pk = np.asarray(P, bool)[:, self.taxa_idx_].astype(float)
        g = self.grid_
        eta = self.coef_[:, [0]] + self.coef_[:, [1]] * g[None, :] + self.coef_[:, [2]] * g[None, :] ** 2   # (T, G)
        log_p = -np.logaddexp(0.0, -eta)          # log sigmoid
        log_q = -np.logaddexp(0.0, eta)           # log (1 - sigmoid)
        return Pk @ log_p + (1.0 - Pk) @ log_q

    def predict(self, P: np.ndarray) -> np.ndarray:
        if len(self.taxa_idx_) == 0:
            return np.full(len(P), self.mu_)
        z = self.grid_[np.argmax(self.log_likelihood(P), axis=1)]
        return self.mu_ + self.sd_ * z
