# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Global LightGBM baseline for SENTINEL-Sonde (RESEARCH_PLAN §5.1 baseline "GBM").

One model per (variable, horizon), shared by all sites ("global"). Target: the climatological
anomaly in model space, ``a = z(d+h) − clim(d+h)``; features: :class:`.features.FeatureBuilder`
(issue-date-causal lags/rolling statistics of the target, other variables' anomalies, log-discharge
level/change, target-day DOY and climatology, staleness/missingness, static site attributes).

Rows and slices (all inside the TRAIN block, train-role sites only):

* **fit rows**: issue dates d with d + h <= ``fit_end`` (default 2019-09-30, WY2008–2019);
* **residual-quantile rows**: issue dates d >= ``cal_start`` + ``cal_embargo_days`` (default
  2019-10-01 + 42 d) with d + h <= 2021-09-30 (WY2020–2021). Never used for fitting; their
  standardised residuals r = (a − â)/s, s = clim std at the target day, give 19 empirical
  quantiles per horizon. Forecast quantiles: clim(d+h) + â + s·Q_r(τ) (monotone by construction).
* **early stopping** (number of boosting rounds only) on the VALIDATION block (train-role and
  val-role sites, common scorable rows), consistent with the LSTM; never on test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.sonde_splits import TIME_BLOCKS
from ..baselines.sonde_baselines import QUANTILES
from .common import HORIZONS, Rows, SondeData
from .features import FeatureBuilder


@dataclass
class GBMConfig:
    objective: str = "l1"
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    max_rounds: int = 2000
    early_stopping_rounds: int = 100
    num_threads: int = 16
    fit_end: str = "2019-09-30"
    cal_start: str = "2019-10-01"
    cal_embargo_days: int = 42
    row_stride: int = 1
    max_fit_rows: int | None = None
    es_max_rows: int = 400_000
    scale_floor_frac: float = 0.05
    seed: int = 0


def fit_cal_masks(dates: pd.DatetimeIndex, ti: np.ndarray, h: int, cfg: GBMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over training rows (issue index ``ti``): fit vs residual-quantile slice."""
    d = dates[ti]
    tgt = d + pd.Timedelta(days=h)
    fit = np.asarray(tgt <= pd.Timestamp(cfg.fit_end))
    cal = np.asarray((d >= pd.Timestamp(cfg.cal_start) + pd.Timedelta(days=cfg.cal_embargo_days))
                     & (tgt <= pd.Timestamp(TIME_BLOCKS["train"][1])))
    return fit, cal


class SondeGBM:
    def __init__(self, cfg: GBMConfig | None = None, horizons=HORIZONS):
        self.cfg = cfg or GBMConfig()
        self.horizons = tuple(horizons)
        self.models: dict = {}       # (var, h) -> lgb.Booster
        self.res_q: dict = {}        # (var, h) -> (19,) standardised residual quantiles
        self.scale_floor: dict = {}  # var -> float
        self.diag: dict = {}
        self._fb = None

    def _builder(self, data: SondeData) -> FeatureBuilder:
        if self._fb is None or self._fb.cube is not data.cube:
            self._fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
        return self._fb

    def _params(self) -> dict:
        c = self.cfg
        return {"objective": c.objective, "learning_rate": c.learning_rate, "num_leaves": c.num_leaves,
                "min_data_in_leaf": c.min_data_in_leaf, "feature_fraction": c.feature_fraction,
                "bagging_fraction": c.bagging_fraction, "bagging_freq": c.bagging_freq,
                "lambda_l2": c.lambda_l2, "num_threads": c.num_threads, "seed": c.seed,
                "deterministic": True, "force_row_wise": True, "verbose": -1}

    @staticmethod
    def _target(fb: FeatureBuilder, v: str, si, ti, h):
        z = fb.z(v)
        T = z.shape[1]
        tgt = ti.astype(np.int64) + h
        y = np.full(si.size, np.nan)
        ok = tgt < T
        y[ok] = z[si[ok], tgt[ok]]
        cm_t, cs_t = fb.clim_at(v, ti, si, shift=h)
        return y - cm_t, cs_t

    def training_rows(self, data: SondeData, var: str, h: int, train_sites: np.ndarray):
        """(si, ti) of train-role rows in the TRAIN block with an issue observation and a target."""
        b = data.baselines(var)
        t_idx = data.issue_idx("train", h)
        m = b.scorable(t_idx, h)[train_sites]
        si_loc, k = np.nonzero(m)
        si, ti = train_sites[si_loc], t_idx[k]
        if self.cfg.row_stride > 1:
            keep = (ti % self.cfg.row_stride) == 0
            si, ti = si[keep], ti[keep]
        return si.astype(np.int64), ti.astype(np.int64)

    def fit_var(self, data: SondeData, var: str, *, train_sites: np.ndarray, es_rows: Rows | None,
                log=print) -> dict:
        import lightgbm as lgb

        fb = self._builder(data)
        blocks = {**fb.target_block(var), **fb.context_block(exclude=var), **fb.weather_block(data.weather)}
        j = data.cube.var_index(var)
        cs_train = data.clim_std[train_sites, :, j]
        self.scale_floor[var] = float(self.cfg.scale_floor_frac * np.nanmedian(cs_train)) \
            if np.isfinite(cs_train).any() else 1e-3
        out = {}
        rs = np.random.default_rng(self.cfg.seed)
        for h in self.horizons:
            si, ti = self.training_rows(data, var, h, train_sites)
            fit, cal = fit_cal_masks(data.cube.dates, ti, h, self.cfg)
            if fit.sum() < 1000 or cal.sum() < 200:
                log(f"[gbm] {var} h={h}: too few rows (fit={int(fit.sum())}, cal={int(cal.sum())}); skipped")
                out[h] = {"skipped": True, "n_fit": int(fit.sum()), "n_cal": int(cal.sum())}
                continue
            fi = np.flatnonzero(fit)
            if self.cfg.max_fit_rows and fi.size > self.cfg.max_fit_rows:
                fi = np.sort(rs.choice(fi, self.cfg.max_fit_rows, replace=False))
            X_fit = fb.table(var, si[fi], ti[fi], h, blocks=blocks)
            y_fit, _ = self._target(fb, var, si[fi], ti[fi], h)
            ds_fit = lgb.Dataset(X_fit, y_fit, free_raw_data=True)
            valid_sets, callbacks = [], [lgb.log_evaluation(0)]
            n_es = 0
            if es_rows is not None:
                em = np.flatnonzero(es_rows.score_mask[h])
                if em.size > self.cfg.es_max_rows:
                    em = np.sort(rs.choice(em, self.cfg.es_max_rows, replace=False))
                n_es = int(em.size)
                if em.size >= 100:
                    esi, eti = es_rows.si[em].astype(np.int64), es_rows.ti[em].astype(np.int64)
                    X_es = fb.table(var, esi, eti, h, blocks=blocks)
                    y_es, _ = self._target(fb, var, esi, eti, h)
                    valid_sets = [lgb.Dataset(X_es, y_es, reference=ds_fit)]
                    callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False))
            booster = lgb.train(self._params(), ds_fit, num_boost_round=self.cfg.max_rounds,
                                valid_sets=valid_sets or None, callbacks=callbacks)
            # residual quantiles on the held-out WY2020–2021 slice (never fitted)
            ci = np.flatnonzero(cal)
            X_cal = fb.table(var, si[ci], ti[ci], h, blocks=blocks)
            y_cal, s_cal = self._target(fb, var, si[ci], ti[ci], h)
            pred = booster.predict(X_cal, num_iteration=booster.best_iteration or None)
            r = (y_cal - pred) / np.maximum(s_cal, self.scale_floor[var])
            r = r[np.isfinite(r)]
            self.res_q[(var, h)] = np.quantile(r, QUANTILES)
            self.models[(var, h)] = booster
            out[h] = {"n_fit": int(fi.size), "n_cal": int(r.size), "n_es": n_es,
                      "best_iteration": int(booster.best_iteration or booster.current_iteration()),
                      "fit_sites": int(np.unique(si[fi]).size), "cal_sites": int(np.unique(si[ci]).size),
                      "fit_issue_max": str(data.cube.dates[ti[fi]].max().date()),
                      "cal_issue_min": str(data.cube.dates[ti[ci]].min().date())}
            log(f"[gbm] {var} h={h}: fit={out[h]['n_fit']} cal={out[h]['n_cal']} es={n_es} "
                f"iters={out[h]['best_iteration']}")
        self.diag[var] = out
        return out

    def predict_rows(self, data: SondeData, var: str, rows: Rows) -> np.ndarray:
        fb = self._builder(data)
        blocks = {**fb.target_block(var), **fb.context_block(exclude=var), **fb.weather_block(data.weather)}
        q = np.full((len(rows), len(self.horizons), len(QUANTILES)), np.nan, dtype=np.float32)
        si, ti = rows.si.astype(np.int64), rows.ti.astype(np.int64)
        for k, h in enumerate(self.horizons):
            if (var, h) not in self.models:
                continue
            X = fb.table(var, si, ti, h, blocks=blocks)
            booster = self.models[(var, h)]
            a_hat = booster.predict(X, num_iteration=booster.best_iteration or None)
            cm_t, cs_t = fb.clim_at(var, ti, si, shift=h)
            s = np.maximum(cs_t, self.scale_floor[var])
            q[:, k] = (cm_t + a_hat)[:, None] + s[:, None] * self.res_q[(var, h)][None, :]
        return q

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {"config": asdict(self.cfg), "horizons": list(self.horizons),
                "scale_floor": self.scale_floor, "diag": self.diag,
                "res_q": {f"{v}|{h}": q.tolist() for (v, h), q in self.res_q.items()}}
        for (v, h), m in self.models.items():
            m.save_model(str(out / f"gbm_{v}_h{h}.txt"), num_iteration=m.best_iteration or None)
        (out / "gbm_meta.json").write_text(json.dumps(meta, indent=1, default=str))

    @classmethod
    def load(cls, out_dir: str | Path) -> SondeGBM:
        import lightgbm as lgb

        out = Path(out_dir)
        meta = json.loads((out / "gbm_meta.json").read_text())
        self = cls(GBMConfig(**meta["config"]), meta["horizons"])
        self.scale_floor = meta["scale_floor"]
        self.diag = meta["diag"]
        for key, q in meta["res_q"].items():
            v, h = key.split("|")
            self.res_q[(v, int(h))] = np.asarray(q)
            self.models[(v, int(h))] = lgb.Booster(model_file=str(out / f"gbm_{v}_h{h}.txt"))
        return self
