# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""TabPFN v2 on ADORE fingerprints + descriptors.

**Version and licence.** Pinned to ``tabpfn==2.2.1`` — the last 2.x release, whose weights are
under the **Prior Labs Licence 1.1** (Apache-2.0 plus an attribution clause), i.e. usable here as
a research baseline. TabPFN 2.5 and later are non-commercial and may require a token
(docs/research/04 §E.2), so they are deliberately not used.

**Environment note.** TabPFN 2.x declares ``scikit-learn<1.7`` and imports the private helper
``sklearn.utils.validation._is_pandas_df``, which newer scikit-learn no longer exposes. Rather
than downgrade scikit-learn for the whole shared environment, :func:`_patch_sklearn_compat`
re-creates that one private helper before TabPFN is imported. It is the only compatibility shim,
it is applied in-process only, and it is a no-op on scikit-learn versions that still provide it.
Telemetry is switched off (``TABPFN_DISABLE_TELEMETRY=1``) before import.

**Shape limits.** TabPFN v2 is designed for <= 10k training rows and a few hundred features, so
the ADORE matrices are reduced: fingerprint bits are kept by training-fold mutual information with
the target, and the training rows are subsampled per chemical (``max_train_rows``, default 6000 —
whole chemicals are kept or dropped, never split). Both reductions are fit on training rows only.
Inference runs in chunks (``predict_chunk``) because a single large batch segfaulted inside
``libc10_cuda`` on the 11 GB Turing cards of this fleet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

MAX_TRAIN_ROWS_DEFAULT = 6000
PREDICT_CHUNK_DEFAULT = 2000
MAX_FEATURES_DEFAULT = 300
TABPFN_PIN = "tabpfn==2.2.1"
TABPFN_LICENSE = "Prior Labs Licence 1.1 (Apache-2.0 + attribution)"


def _patch_sklearn_compat() -> None:
    """Provide ``sklearn.utils.validation._is_pandas_df`` for TabPFN 2.x on new scikit-learn."""
    import sklearn.utils.validation as V

    if not hasattr(V, "_is_pandas_df"):
        import sys

        def _is_pandas_df(X):  # pragma: no cover - trivial
            pd = sys.modules.get("pandas")
            return pd is not None and isinstance(X, pd.DataFrame)

        V._is_pandas_df = _is_pandas_df


# applied at import time as well, so that *any* later `import tabpfn` (e.g. a bare
# `importorskip("tabpfn")` in a test) also sees the helper
_patch_sklearn_compat()


def load_tabpfn():
    """Import TabPFN with telemetry disabled and the compatibility shim applied."""
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    _patch_sklearn_compat()
    import tabpfn
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor, tabpfn.__version__


@dataclass
class TabPFNTox:
    """TabPFN v2 regressor with training-row and feature budgets.

    ``params``: ``n_estimators`` (TabPFN ensemble members), ``device`` (``cpu`` or ``cuda``),
    ``max_train_rows``, ``max_features``, ``subsample_by`` (``chemical`` keeps whole chemicals when
    subsampling, ``row`` samples rows).
    """

    params: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        self.model_ = None
        self.keep_: np.ndarray | None = None
        self.version_: str | None = None
        self.predict_chunk_ = PREDICT_CHUNK_DEFAULT

    # ------------------------------------------------------------------ feature / row budgets
    def _select_features(self, X: np.ndarray, y: np.ndarray, max_features: int) -> np.ndarray:
        if X.shape[1] <= max_features:
            return np.ones(X.shape[1], dtype=bool)
        from sklearn.feature_selection import mutual_info_regression

        rng = np.random.default_rng(self.seed)
        sub = rng.choice(len(X), size=min(len(X), 4000), replace=False)
        mi = mutual_info_regression(X[sub], y[sub], random_state=self.seed)
        keep = np.zeros(X.shape[1], dtype=bool)
        keep[np.argsort(mi)[::-1][:max_features]] = True
        return keep

    def _subsample(self, X: np.ndarray, y: np.ndarray, groups, max_rows: int, by: str):
        if len(X) <= max_rows:
            return X, y
        rng = np.random.default_rng(self.seed)
        if by == "chemical" and groups is not None:
            g = np.asarray(groups, dtype=object).astype(str)
            uniq = np.unique(g)
            rng.shuffle(uniq)
            take = np.zeros(len(X), dtype=bool)
            for c in uniq:
                m = g == c
                if take.sum() + m.sum() > max_rows:
                    continue
                take |= m
            if take.sum() < max_rows // 2:  # pathological: fall back to row sampling
                idx = rng.choice(len(X), size=max_rows, replace=False)
                return X[idx], y[idx]
            return X[take], y[take]
        idx = rng.choice(len(X), size=max_rows, replace=False)
        return X[idx], y[idx]

    # ------------------------------------------------------------------ sklearn-style API
    def fit(self, X, y, *, train_rows=None, **_ctx) -> TabPFNTox:
        TabPFNRegressor, version = load_tabpfn()
        self.version_ = version
        p = dict(self.params)
        max_features = int(p.pop("max_features", MAX_FEATURES_DEFAULT))
        max_rows = int(p.pop("max_train_rows", MAX_TRAIN_ROWS_DEFAULT))
        by = str(p.pop("subsample_by", "chemical"))
        self.predict_chunk_ = int(p.pop("predict_chunk", PREDICT_CHUNK_DEFAULT))
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        self.keep_ = self._select_features(X, y, max_features)
        groups = None if train_rows is None else train_rows["test_cas"].astype("string").to_numpy()
        Xs, ys = self._subsample(X[:, self.keep_], y, groups, max_rows, by)
        p.setdefault("device", "cpu")
        p.setdefault("n_estimators", 4)
        p["random_state"] = self.seed
        self.model_ = TabPFNRegressor(**p)
        self.model_.fit(Xs, ys)
        self.n_train_ = len(Xs)
        return self

    def predict(self, X, **_ctx) -> np.ndarray:
        """Chunked inference: one big batch segfaulted the CUDA allocator on these 11 GB cards."""
        if self.model_ is None:
            raise RuntimeError("TabPFNTox.predict before fit")
        X = np.asarray(X, dtype=np.float32)[:, self.keep_]
        chunk = int(getattr(self, "predict_chunk_", PREDICT_CHUNK_DEFAULT))
        out = [np.asarray(self.model_.predict(X[i:i + chunk]), dtype=float)
               for i in range(0, len(X), chunk)]
        return np.concatenate(out) if out else np.empty(0)


def make_tabpfn(seed: int = 0):
    return lambda params: TabPFNTox(params=params, seed=seed)
