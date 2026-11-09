# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GPBoost: tree boosting on the chemistry **plus** mixed-effects random intercepts for the taxon.

Why (cycle 2, 2026-10-24). Our phase-1 models treat the test species through fixed features (an
embedding for chemprop, Add-my-Pet traits for the trees) and collapse when asked to extrapolate
across taxa (ADORE ``a-CA2F-diff`` test RMSE 1.257 for chemprop, the worst of all models). Two
independent 2024–2026 studies pick GPBoost — gradient boosting for the fixed effects combined with
a Gaussian-process / grouped random-effects model for the structured part — for exactly this task:

* Zubrod, Galic, Vaugeois & Dreier, "Bio-QSARs 2.0: Unlocking a generic modeling approach for
  species sensitivity predictions", *Environment International* 186:108607 (2024) — GPBoost for
  fish and aquatic invertebrates, cross-chemical *and* cross-species prediction.
* Yuan, Shaw, Tang & Ye, *Environmental Toxicology and Chemistry* 45(9):2466–2482 (2026) —
  benchmark of cross-chemical / cross-species toxicity prediction over 50,603 records and 2,285
  species: GPBoost is the best 2D-structure-based model, and how species information is
  *represented* matters more than which learner is used.

The mechanism that matters here: a nested random effect (taxon group ⊃ family ⊃ genus ⊃ species)
is estimated with shrinkage toward the parent mean, so a species seen only a few times — or a
taxon the model has never seen at all — falls back up the hierarchy instead of extrapolating a
fixed feature vector. Gasser et al.'s negative result (a *fixed* phylogenetic-distance GP kernel
did not help) is consistent with this: it is the jointly estimated grouped effect, not a fixed
kernel, that carries the information.

``group_columns`` chooses the hierarchy actually used (default species + genus + taxon group).
Groups unseen in training are mapped to a sentinel level, for which GPBoost predicts the grand
mean of the random effect (i.e. exactly the shrinkage fallback), with ``predict_var=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GROUP_COLUMNS_DEFAULT = ("tax_gs", "tax_genus", "tax_group")
UNSEEN = "<unseen>"

GPBOOST_DEFAULTS = {"objective": "regression_l2", "verbose": -1, "num_threads": 8}


@dataclass
class GPBoostTox:
    """GPBoost regressor: boosted trees on X, grouped random effects on the taxonomy.

    ``params``: GPBoost/LightGBM parameters (``learning_rate``, ``num_leaves``,
    ``min_data_in_leaf``, ``lambda_l2``, ``feature_fraction``, ``bagging_fraction``) plus
    ``num_boost_round`` and ``group_columns`` (tuple of row columns forming the hierarchy).
    """

    params: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        self.model_ = None
        self.gp_ = None
        self.group_columns_ = tuple(self.params.get("group_columns", GROUP_COLUMNS_DEFAULT))
        self.levels_: dict[str, set] = {}

    # ------------------------------------------------------------------ helpers
    def _group_frame(self, rows: pd.DataFrame, *, fit: bool) -> np.ndarray:
        cols = []
        for c in self.group_columns_:
            v = rows[c].astype("string").fillna("<NA>")
            if fit:
                self.levels_[c] = set(v.unique().tolist())
            else:
                v = v.where(v.isin(self.levels_.get(c, set())), UNSEEN)
            cols.append(v.to_numpy())
        return np.column_stack(cols)

    # ------------------------------------------------------------------ protocol
    def fit(self, X, y, train_rows: pd.DataFrame | None = None, sample_weight=None,
            **_ctx) -> GPBoostTox:
        import gpboost as gpb

        if train_rows is None:
            raise ValueError("GPBoostTox needs train_rows (the taxonomy columns live there)")
        p = dict(GPBOOST_DEFAULTS)
        p.update({k: v for k, v in self.params.items()
                  if k not in ("num_boost_round", "group_columns")})
        p["seed"] = self.seed
        rounds = int(self.params.get("num_boost_round", 500))
        groups = self._group_frame(train_rows, fit=True)
        self.gp_ = gpb.GPModel(group_data=groups, likelihood="gaussian")
        dset = gpb.Dataset(np.asarray(X, dtype=np.float32), label=np.asarray(y, dtype=float),
                           weight=None if sample_weight is None
                           else np.asarray(sample_weight, dtype=float))
        self.model_ = gpb.train(params=p, train_set=dset, gp_model=self.gp_,
                                num_boost_round=rounds)
        return self

    def predict(self, X, rows: pd.DataFrame | None = None, **_ctx) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("GPBoostTox.predict before fit")
        if rows is None:
            raise ValueError("GPBoostTox needs rows (the taxonomy columns live there)")
        groups = self._group_frame(rows, fit=False)
        pred = self.model_.predict(data=np.asarray(X, dtype=np.float32), group_data_pred=groups,
                                   predict_var=False, pred_latent=False)
        out = pred["response_mean"] if isinstance(pred, dict) else pred
        return np.asarray(out, dtype=float)


def make_gpboost(seed: int = 0):
    return lambda params: GPBoostTox(params=params, seed=seed)
