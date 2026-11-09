# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""LightGBM and CatBoost regressors for ChemTox (improvement cycle 2).

The published ADORE baseline used XGBoost and a random forest. LightGBM and CatBoost are the two
other standard gradient-boosting implementations and routinely differ by a few percent on sparse
binary fingerprint matrices; CatBoost's ordered boosting and symmetric trees in particular behave
differently on many correlated bits. They are given the **same search budget** as the baselines
(`scripts/train/chemtox_cycle.py --budget 32`), so no family is advantaged by tuning.

Both wrappers follow the ChemTox model protocol: ``fit(X, y, sample_weight=None, **context)`` and
``predict(X, **context)``. ``sample_weight`` is used when ECOTOX augmentation rows are added to a
fold's training rows with a weight below 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LGBM_DEFAULTS = {"objective": "regression", "n_jobs": 8, "verbose": -1, "force_row_wise": True}
CATBOOST_DEFAULTS = {"loss_function": "RMSE", "thread_count": 8, "verbose": False,
                     "allow_writing_files": False}


@dataclass
class LightGBMTox:
    """LightGBM on a ChemTox design matrix."""

    params: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        self.model_ = None

    def fit(self, X, y, sample_weight=None, **_ctx) -> LightGBMTox:
        import lightgbm as lgb

        p = dict(LGBM_DEFAULTS)
        p.update(self.params)
        p["random_state"] = self.seed
        if "eta" in p:
            p["learning_rate"] = p.pop("eta")
        self.model_ = lgb.LGBMRegressor(**p)
        self.model_.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=float),
                        sample_weight=None if sample_weight is None
                        else np.asarray(sample_weight, dtype=float))
        return self

    def predict(self, X, **_ctx) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("LightGBMTox.predict before fit")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float32)), dtype=float)


@dataclass
class CatBoostTox:
    """CatBoost on a ChemTox design matrix (dense float32; all features numeric)."""

    params: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        self.model_ = None

    def fit(self, X, y, sample_weight=None, **_ctx) -> CatBoostTox:
        from catboost import CatBoostRegressor

        p = dict(CATBOOST_DEFAULTS)
        p.update(self.params)
        p["random_seed"] = self.seed
        if "eta" in p:
            p["learning_rate"] = p.pop("eta")
        self.model_ = CatBoostRegressor(**p)
        self.model_.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=float),
                        sample_weight=None if sample_weight is None
                        else np.asarray(sample_weight, dtype=float))
        return self

    def predict(self, X, **_ctx) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("CatBoostTox.predict before fit")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float32)), dtype=float)


def make_lgbm(seed: int = 0):
    return lambda params: LightGBMTox(params=params, seed=seed)


def make_catboost(seed: int = 0):
    return lambda params: CatBoostTox(params=params, seed=seed)
