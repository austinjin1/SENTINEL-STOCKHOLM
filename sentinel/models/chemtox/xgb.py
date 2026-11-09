# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""XGBoost / random-forest reproduction of the published ADORE baseline.

Gasser et al. (2024) report, for ``t-F2F``, molar LC50, ``split_occurrence``, on the official test
rows: best micro RMSE **0.927** (XGBoost + MACCS), best chemical-and-taxon macro RMSE_M **0.904**
(random forest + MACCS), best taxon macro RMSE_T **0.938** (XGBoost + MACCS) and best chemical
macro RMSE_C **0.845** (random forest + ToxPrint). Three different model/representation
combinations produce those four numbers, so reproducing all four means running RF as well as
XGBoost — which is why :class:`RandomForestTox` lives here too.

They optimise ``n_estimators``, ``eta``, ``gamma``, ``max_depth``, ``min_child_weight`` and
``subsample`` by grid search with 5-fold CV on the training data (their ESI Table 2 is not
retrievable, so :data:`XGB_GRID` is our own grid over the same six hyper-parameters, and the number
of configurations tried is recorded in the registry entry).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: grid over the six hyper-parameters the published baseline tuned
XGB_GRID = {
    "n_estimators": (300, 800),
    "eta": (0.03, 0.1),
    "max_depth": (6, 10),
    "min_child_weight": (1, 5),
    "gamma": (0.0,),
    "subsample": (0.8, 1.0),
}
#: a small grid for smoke tests
XGB_GRID_SMOKE = {"n_estimators": (20,), "eta": (0.3,), "max_depth": (3,)}

RF_GRID = {
    "n_estimators": (300,),
    "max_depth": (None, 20),
    "min_samples_split": (2, 5),
    "max_features": (0.3, "sqrt"),
}

XGB_DEFAULTS = {
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_jobs": 8,
    "random_state": 0,
}


@dataclass
class XGBoostTox:
    """XGBoost regressor on a ChemTox design matrix (CPU histogram method)."""

    params: dict
    seed: int = 0

    def __post_init__(self):
        self.model_ = None

    def fit(self, X, y, sample_weight=None, **_ctx) -> XGBoostTox:
        import xgboost as xgb

        p = dict(XGB_DEFAULTS)
        p.update(self.params)
        p["random_state"] = self.seed
        if "eta" in p:  # the paper's name for the learning rate
            p["learning_rate"] = p.pop("eta")
        self.model_ = xgb.XGBRegressor(**p)
        self.model_.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=float),
                        sample_weight=None if sample_weight is None
                        else np.asarray(sample_weight, dtype=float))
        return self

    def predict(self, X, **_ctx) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("XGBoostTox.predict before fit")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float32)), dtype=float)


@dataclass
class RandomForestTox:
    """scikit-learn random forest, the published baseline's other tree model."""

    params: dict
    seed: int = 0

    def __post_init__(self):
        self.model_ = None

    def fit(self, X, y, sample_weight=None, **_ctx) -> RandomForestTox:
        from sklearn.ensemble import RandomForestRegressor

        p = {"n_estimators": 300, "n_jobs": 8}
        p.update(self.params)
        p["random_state"] = self.seed
        self.model_ = RandomForestRegressor(**p)
        self.model_.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=float),
                        sample_weight=None if sample_weight is None
                        else np.asarray(sample_weight, dtype=float))
        return self

    def predict(self, X, **_ctx) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("RandomForestTox.predict before fit")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float32)), dtype=float)


def make_xgb(seed: int = 0):
    return lambda params: XGBoostTox(params=params, seed=seed)


def make_rf(seed: int = 0):
    return lambda params: RandomForestTox(params=params, seed=seed)
