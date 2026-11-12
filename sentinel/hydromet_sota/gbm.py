# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GBM parity comparators for the M6 SOTA levers (``docs/preregistration/hydromet_sota_protocol.md`` §5).

* :class:`WeightedForcingGBM` — the frozen ``forcing_gbm`` (same columns, 3-point ``max_rows_per_site`` grid selected per
  target on the same uncapped inner rows, refit on fit + inner with the selected iteration count) trained on the lever
  data. Weak-label rows (lever 1) get one common row weight so that their total weight is the share ``s`` of the target's
  training weight (the EA-LSTM's weak loss share). Inner rows are dense inner sites only, unweighted.
* :func:`qhat_features` — lever 2 parity: a discharge GBM on the same discharge labels (fit sites × train block),
  cross-fitted by basin group (5 folds) on the fit sites; inner / weak / val locations get the full-fit prediction. Its
  output enters the WQ GBM as dynamic inputs ``f_qhat_log``, ``f_qhat_log_m7``, ``f_qhat_log_m30`` (a model output from
  forcings and statics; no discharge is read at prediction).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from ..models.hydromet.baselines import LGBMConfig, TabularGBM, check_fit_sites, sample_rows
from ..models.hydromet.data import HydroMetData, to_model
from ..models.hydromet.features import FeatureSet, trailing

WEAK_TARGETS = ("temp_c_mean", "do_mgl_mean")
QHAT_NAMES = ("f_qhat_log", "f_qhat_log_m7", "f_qhat_log_m30")


def _lgbm(X, y, w, Xv, yv, cfg: LGBMConfig, seed: int, n_estimators: int | None = None):
    import lightgbm as lgb

    params = dict(num_leaves=cfg.num_leaves, learning_rate=cfg.learning_rate,
                  n_estimators=n_estimators or cfg.n_estimators, min_child_samples=cfg.min_child_samples,
                  subsample=cfg.subsample, subsample_freq=cfg.subsample_freq, colsample_bytree=cfg.colsample_bytree,
                  reg_lambda=cfg.reg_lambda, n_jobs=cfg.n_jobs, random_state=seed, verbose=-1)
    m = lgb.LGBMRegressor(**params)
    if Xv is not None and len(yv) >= 50 and n_estimators is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)],
                  callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
        best = int(m.best_iteration_ or cfg.n_estimators)
        rmse = float(np.sqrt(np.mean((m.predict(Xv, num_iteration=best) - yv) ** 2)))
        return m, best, rmse
    if n_estimators is None:
        n_estimators = min(cfg.n_estimators, cfg.no_inner_n_estimators)
        m = lgb.LGBMRegressor(**{**params, "n_estimators": n_estimators})
    m.fit(X, y, sample_weight=w)
    return m, int(n_estimators), float("nan")


def weak_weight(n_dense: int, n_weak: int, share: float) -> float:
    """Common weak-row weight w with n_weak·w / (n_dense + n_weak·w) = share."""
    if share <= 0 or n_weak == 0:
        return 0.0
    return share / (1.0 - share) * n_dense / n_weak


class WeightedForcingGBM(TabularGBM):
    def fit(self, data: HydroMetData, fs: FeatureSet, fit_idx, inner_idx, *, weak_idx=None, weak_share: float = 0.0,
            log=print, min_sites: int = 3):
        weak_idx = np.asarray([] if weak_idx is None else weak_idx, int)
        check_fit_sites(data, fit_idx, inner_idx, weak_idx)
        for v in self.targets:
            vs, vt = sample_rows(data, v, inner_idx, max_per_site=None)
            Xv = fs.design_matrix(vs, vt, self.columns) if len(vs) else None
            yv = to_model(v, data.y[vs, vt, data.vi(v)]) if len(vs) else None
            use_weak = weak_share > 0 and v in WEAK_TARGETS and len(weak_idx)
            if use_weak:
                ws_, wt_ = sample_rows(data, v, weak_idx, max_per_site=None)
                Xw = fs.design_matrix(ws_, wt_, self.columns)
                yw = to_model(v, data.y[ws_, wt_, data.vi(v)])
            best, tried = None, []
            for k, gcfg in enumerate(self.grid):
                si, ti = sample_rows(data, v, fit_idx, max_per_site=gcfg.max_rows_per_site, seed=self.seed)
                if np.unique(si).size < min_sites:
                    break
                X = fs.design_matrix(si, ti, self.columns)
                y = to_model(v, data.y[si, ti, data.vi(v)])
                w = np.ones(len(y))
                isw = np.zeros(len(y), bool)
                if use_weak:
                    ww = weak_weight(len(y), len(yw), weak_share)
                    X, y, w = np.concatenate([X, Xw]), np.concatenate([y, yw]), np.r_[w, np.full(len(yw), ww)]
                    isw = np.r_[isw, np.ones(len(yw), bool)]
                m, it, rmse = _lgbm(X, y, w, Xv, yv, gcfg, self.seed)
                rec = {"grid_index": k, "max_rows_per_site": gcfg.max_rows_per_site, "n_train": int(len(y)),
                       "n_weak": int(len(yw)) if use_weak else 0, "best_iteration": it, "inner_rmse_model_space": rmse}
                tried.append(rec)
                log(f"[{self.name}] {v} grid {k}: {rec}")
                if best is None or (np.isfinite(rmse) and rmse < best[0]):
                    best = (rmse, k, m, it, X, y, isw)
                del X, y, w
            if best is None:
                self.info[v] = {"skipped": True}
                continue
            _, k, m, it, X, y, isw = best
            gcfg = self.grid[k]
            if gcfg.refit_full and Xv is not None and len(yv):
                n_d = int((~isw).sum()) + len(yv)
                w2 = np.r_[np.where(isw, weak_weight(n_d, int(isw.sum()), weak_share), 1.0), np.ones(len(yv))]
                m, _, _ = _lgbm(np.concatenate([X, Xv]), np.concatenate([y, yv]), w2, None, None, gcfg, self.seed,
                                n_estimators=max(it, 10))
            self.models[v] = m
            self.info[v] = {"selected_grid_index": k, "grid": tried, "n_inner_rows": int(len(vs)),
                            "weak_share": weak_share if use_weak else 0.0, "refit_full": bool(gcfg.refit_full)}
        return self


def _q_rows(aux_q, site_idx, tmask, max_per_site, seed):
    m = np.isfinite(aux_q[site_idx]) & tmask[None, :]
    si, ti = np.nonzero(m)
    if max_per_site and len(si):
        rs = np.random.default_rng(seed)
        o = rs.permutation(len(si))
        si, ti = si[o], ti[o]
        import pandas as pd

        keep = pd.Series(si).groupby(si).cumcount().to_numpy() < max_per_site
        si, ti = si[keep], ti[keep]
    return np.asarray(site_idx)[si], ti


def _predict_full(model, fs: FeatureSet, columns, site_idx, chunk: int = 60) -> np.ndarray:
    T = fs.dyn.shape[1]
    out = np.full((len(site_idx), T), np.nan, dtype="float32")
    for c in range(0, len(site_idx), chunk):
        sl = np.asarray(site_idx[c:c + chunk])
        si = np.repeat(sl, T)
        ti = np.tile(np.arange(T), len(sl))
        out[c:c + len(sl)] = model.predict(fs.design_matrix(si, ti, columns)).reshape(len(sl), T)
    return out


class QStack:
    """Discharge GBM (log Q) for the lever-2 stacked inputs: grouped cross-fitting on fit sites, full fit otherwise."""

    def __init__(self, columns, cfg: LGBMConfig, seed: int = 0, n_folds: int = 5, max_rows_per_site: int = 1000):
        self.columns, self.cfg, self.seed, self.n_folds, self.max_rows = list(columns), cfg, seed, n_folds, max_rows_per_site
        self.full = None
        self.info: dict = {}

    def fit_oof(self, data: HydroMetData, fs: FeatureSet, aux_q: np.ndarray, fit_idx, inner_idx, log=print) -> np.ndarray:
        """Fits the full model (early-stopped on inner sites' discharge) and returns (S, T) log-Q predictions: out-of-fold
        for fit sites (5 folds by basin group), full-model for every other location."""
        tmask = data.block_mask("train")
        fit_idx, inner_idx = np.asarray(fit_idx, int), np.asarray(inner_idx, int)
        vs, vt = _q_rows(aux_q, inner_idx, tmask, None, self.seed)
        Xv = fs.design_matrix(vs, vt, self.columns) if len(vs) else None
        yv = aux_q[vs, vt].astype("float64") if len(vs) else None
        si, ti = _q_rows(aux_q, fit_idx, tmask, self.max_rows, self.seed)
        X, y = fs.design_matrix(si, ti, self.columns), aux_q[si, ti].astype("float64")
        cfg = dataclasses.replace(self.cfg, refit_full=False)
        self.full, it, rmse = _lgbm(X, y, None, Xv, yv, cfg, self.seed)
        self.info = {"n_rows": int(len(y)), "n_sites": int(np.unique(si).size), "best_iteration": it,
                     "inner_rmse_logq": rmse, "n_inner_rows": int(0 if yv is None else len(yv))}
        log(f"[qstack] full discharge GBM: {self.info}")
        qhat = np.full(data.y.shape[:2], np.nan, dtype="float32")
        others = np.setdiff1d(np.arange(len(data.sites)), fit_idx)
        if len(others):
            qhat[others] = _predict_full(self.full, fs, self.columns, others)
        groups = np.unique(data.basin_group[fit_idx])
        rs = np.random.default_rng(self.seed)
        fold_of = {g: k % self.n_folds for k, g in enumerate(groups[rs.permutation(len(groups))])}
        fold = np.array([fold_of[g] for g in data.basin_group[fit_idx]])
        self.info["fold_sizes"] = np.bincount(fold, minlength=self.n_folds).tolist()
        for k in range(self.n_folds):
            tr, te = fit_idx[fold != k], fit_idx[fold == k]
            m_ = np.isin(si, tr)
            mk, _, _ = _lgbm(X[m_], y[m_], None, None, None, cfg, self.seed, n_estimators=max(it, 10))
            qhat[te] = _predict_full(mk, fs, self.columns, te)
            log(f"[qstack] fold {k}: {len(te)} sites predicted out of fold")
        return qhat

    def predict(self, fs: FeatureSet, site_idx) -> np.ndarray:
        return _predict_full(self.full, fs, self.columns, np.asarray(site_idx, int))


def append_qhat(fs: FeatureSet, qhat: np.ndarray) -> FeatureSet:
    """Append the stacked discharge inputs (causal: same-day value and trailing 7/30-day means of the prediction)."""
    q = qhat.astype("float64")
    add = np.stack([q, trailing(q, 7, "mean"), trailing(q, 30, "mean")], axis=-1).astype("float32")
    return FeatureSet(np.concatenate([fs.dyn, add], axis=-1), list(fs.dyn_names) + list(QHAT_NAMES), fs.static, fs.cal,
                      fs.lag)
