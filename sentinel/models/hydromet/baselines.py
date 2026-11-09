# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Baseline ladder for SENTINEL-HydroMet (RESEARCH_PLAN §5.6, §14.3). No baseline uses any in-situ
sensor input: they see only forcings (causal, :mod:`.features`), static attributes and calendar.
All are fitted on train-role sites × the train block; "inner" train-role basin groups (never val-
or test-role sites) are used for early stopping / window selection.

* ``doy_location_gbm`` — DOY + location climatology learned across train sites: LightGBM on lat, lon,
  elevation and DOY harmonics (no weather). All targets.
* ``air2water_linear`` / ``air2water_mohseni`` — pooled regression of water temperature on the
  trailing w-day mean air temperature (w selected on inner sites); linear, and the Mohseni et al.
  (1998) logistic ``Tw = μ + (α − μ)/(1 + exp(γ(β − Ta_w)))``. temp_c_mean (+ 7-DADMax via
  :func:`dadm_from_mean`).
* ``air2stream_regional`` — air2stream-type model (:func:`.physics.air2stream_simulate`, no discharge
  term) calibrated per train site, transferred to an unseen site by **donor output averaging**: the k
  nearest train sites in standardised attribute space are each run with the target site's own
  forcing and averaged (inverse-distance weights). temp_c_mean (+ 7-DADMax).
* ``dosat_ratio[<temp source>]`` — DO = DO_sat(T̂_w, P(elev)) (Benson–Krause) × the pooled
  train-site median ratio DO/DO_sat by calendar month, with T̂_w from a temperature baseline.
  do_mgl_mean and do_mgl_min.
* ``forcing_gbm`` — LightGBM on causal forcing windows + all StreamCat/site attributes + calendar.
  All targets.

Predictions are point estimates in ORIGINAL units: ``predict(data, fs, site_idx, t_idx)`` returns
{target: (len(site_idx), len(t_idx)) float32}.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGETS, HydroMetData, from_model, to_model
from .features import FeatureSet, trailing
from .physics import A2S_BOUNDS, A2S_PARAMS, air2stream_simulate, do_sat_mgl, pressure_atm_from_elevation

DOY_LOCATION_STATIC = ["lat", "lon", "elev_m"]
DONOR_ATTRS = ["lat", "lon", "elev_m", "wsareasqkm", "bfiws", "tmean9120ws", "precip9120ws", "elevws",
               "wtdepws", "pctimp2019ws", "pctdecid2019ws", "pctconif2019ws", "damnrmstorws", "nhd_slope"]
LOG_DONOR_ATTRS = {"wsareasqkm", "damnrmstorws", "nhd_slope"}


# ----------------------------------------------------------------------------- helpers

def check_fit_sites(data: HydroMetData, *idx_sets) -> None:
    for idx in idx_sets:
        if idx is not None and len(idx) and not np.isin(data.roles[np.asarray(idx, int)], ["train"]).all():
            raise ValueError("baselines may only be fitted on train-role sites")


def sample_rows(data: HydroMetData, v: str, site_idx, block: str = "train", *, max_per_site: int | None = None,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Observed (site, day) rows of target ``v`` in ``block`` at ``site_idx`` (≤ max_per_site per site)."""
    m = data.target_mask(v, block, site_idx)
    si, ti = np.nonzero(m)
    if max_per_site and len(si):
        rs = np.random.default_rng(seed)
        order = rs.permutation(len(si))
        si, ti = si[order], ti[order]
        rank = pd.Series(si).groupby(si).cumcount().to_numpy()
        keep = rank < max_per_site
        si, ti = si[keep], ti[keep]
        o = np.lexsort((ti, si))
        si, ti = si[o], ti[o]
    return si, ti


def dadm_offset_by_month(data: HydroMetData, fit_idx) -> np.ndarray:
    """Pooled median of (7-DADMax − trailing 7-day mean of daily-mean temperature) per month, fitted
    on fit sites × train block (observed values only; targets, not inputs)."""
    tm = data.vi("temp_c_mean")
    td = data.vi("temp_c_7dadm")
    train = data.block_mask("train")
    tmean = data.y[fit_idx, :, tm].astype("float64")
    tmean[:, ~train] = np.nan            # no target value from outside the train block enters the window
    m7 = trailing(tmean, 7, "mean", min_frac=1.0)
    diff = data.y[fit_idx, :, td] - m7
    diff[:, ~train] = np.nan
    month = data.dates.month.to_numpy()
    out = np.zeros(13)
    for mo in range(1, 13):
        x = diff[:, month == mo]
        x = x[np.isfinite(x)]
        out[mo] = np.median(x) if x.size else np.nan
    out[0] = np.nanmedian(out[1:])
    return np.where(np.isfinite(out), out, out[0])


def dadm_from_mean(tw_hat_full: np.ndarray, offset_by_month: np.ndarray, months: np.ndarray) -> np.ndarray:
    """7-DADMax estimate = trailing 7-day mean of the predicted daily-mean temperature + monthly offset."""
    return trailing(tw_hat_full.astype("float64"), 7, "mean", min_frac=1.0) + offset_by_month[months][None, :]


@dataclass
class LGBMConfig:
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 3000
    min_child_samples: int = 200
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 100
    n_jobs: int = 8
    max_rows_per_site: int | None = 1000   # None = all train days of a site
    no_inner_n_estimators: int = 500       # trees when a target has < 50 inner rows (no early stopping)
    refit_full: bool = True


def _fit_lgbm(X, y, Xv, yv, cfg: LGBMConfig, seed: int, fixed_size: bool = False):
    """Early-stopped on (Xv, yv) when >= 50 inner rows; ``fixed_size`` = exactly cfg.n_estimators (refits);
    otherwise (no usable inner rows) cfg.no_inner_n_estimators trees."""
    import lightgbm as lgb

    params = dict(num_leaves=cfg.num_leaves, learning_rate=cfg.learning_rate, n_estimators=cfg.n_estimators,
                  min_child_samples=cfg.min_child_samples, subsample=cfg.subsample, subsample_freq=cfg.subsample_freq,
                  colsample_bytree=cfg.colsample_bytree, reg_lambda=cfg.reg_lambda, n_jobs=cfg.n_jobs,
                  random_state=seed, verbose=-1)
    m = lgb.LGBMRegressor(**params)
    best = cfg.n_estimators
    if Xv is not None and len(yv) >= 50:
        with warnings.catch_warnings():   # LightGBM 4.7 deprecation notice for eval_set (behaviour unchanged)
            warnings.simplefilter("ignore", category=Warning)
            m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
        best = int(m.best_iteration_ or cfg.n_estimators)
        inner_rmse = float(np.sqrt(np.mean((m.predict(Xv, num_iteration=best) - yv) ** 2)))
    else:   # no usable inner rows (e.g. a target almost absent from the inner groups): fixed, capped size
        best = cfg.n_estimators if fixed_size else min(cfg.n_estimators, cfg.no_inner_n_estimators)
        inner_rmse = float("nan")
    if cfg.refit_full and Xv is not None and len(yv):
        m = lgb.LGBMRegressor(**{**params, "n_estimators": max(best, 10)})
        m.fit(np.concatenate([X, Xv]), np.concatenate([y, yv]))
    elif Xv is None or len(yv) < 50:
        m = lgb.LGBMRegressor(**{**params, "n_estimators": best})
        m.fit(X, y)
    return m, {"best_iteration": best, "inner_rmse_model_space": inner_rmse, "n_train": int(len(y)),
               "n_inner": int(0 if yv is None else len(yv))}


class BaselineBase:
    name = "base"
    targets: tuple = ()

    def __init__(self):
        self.info: dict = {}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ----------------------------------------------------------------------------- GBMs

class TabularGBM(BaselineBase):
    """LightGBM per target on a fixed list of input columns (model-space targets).

    Tuning (review W5): ``grid`` is a list of LGBMConfig candidates; for each target every candidate is
    early-stopped on the SAME inner-site rows (all inner train-role rows, uncapped) and the one with the
    lowest inner RMSE is kept (then refitted on fit + inner sites with its best iteration if
    ``refit_full``). Val-role sites are never used. ``max_rows_per_site = None`` = all train days."""

    def __init__(self, name: str, columns: list[str], targets=tuple(TARGETS), cfg: LGBMConfig | None = None,
                 seed: int = 0, grid: list | None = None):
        super().__init__()
        self.name, self.columns, self.targets = name, list(columns), tuple(targets)
        self.cfg = cfg or LGBMConfig()
        self.grid = list(grid) if grid else [self.cfg]
        self.seed = seed
        self.models: dict = {}

    def fit(self, data: HydroMetData, fs: FeatureSet, fit_idx, inner_idx, log=print, min_sites: int = 3):
        import dataclasses

        check_fit_sites(data, fit_idx, inner_idx)
        for v in self.targets:
            vs, vt = sample_rows(data, v, inner_idx, max_per_site=None)
            Xv = fs.design_matrix(vs, vt, self.columns) if len(vs) else None
            yv = to_model(v, data.y[vs, vt, data.vi(v)]) if len(vs) else None
            best, tried = None, []
            for k, gcfg in enumerate(self.grid):
                si, ti = sample_rows(data, v, fit_idx, max_per_site=gcfg.max_rows_per_site, seed=self.seed)
                n_sites = int(np.unique(si).size)
                if n_sites < min_sites:
                    break
                X = fs.design_matrix(si, ti, self.columns)
                y = to_model(v, data.y[si, ti, data.vi(v)])
                m, inf = _fit_lgbm(X, y, Xv, yv, dataclasses.replace(gcfg, refit_full=False), self.seed)
                tried.append({"grid_index": k, **{f: getattr(gcfg, f) for f in ("num_leaves", "max_rows_per_site",
                                                                                "learning_rate", "min_child_samples")},
                              **inf})
                log(f"[{self.name}] {v} grid {k}: {inf['n_train']} rows / {n_sites} sites, best_iter "
                    f"{inf['best_iteration']}, inner RMSE(model space) {inf['inner_rmse_model_space']:.4f}")
                score = inf["inner_rmse_model_space"]
                if best is None or (np.isfinite(score) and score < best[0]):
                    best = (score, k, m, inf, X, y)
                del X, y
            if best is None:
                log(f"[{self.name}] {v}: fewer than {min_sites} fit sites with data; skipped")
                self.info[v] = {"skipped": True}
                continue
            _, k, m, inf, X, y = best
            gcfg = self.grid[k]
            if gcfg.refit_full and Xv is not None and len(yv):
                m, _ = _fit_lgbm(np.concatenate([X, Xv]), np.concatenate([y, yv]), None, None,
                                 dataclasses.replace(gcfg, n_estimators=max(inf["best_iteration"], 10),
                                                     refit_full=False), self.seed, fixed_size=True)
            self.models[v] = m
            self.info[v] = {"selected_grid_index": k, "grid": tried, "n_inner_rows": int(len(vs)),
                            "n_inner_sites": int(np.unique(vs).size), "refit_full": bool(gcfg.refit_full)}
        return self

    def predict(self, data: HydroMetData, fs: FeatureSet, site_idx, t_idx) -> dict:
        site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
        si = np.repeat(site_idx, len(t_idx))
        ti = np.tile(t_idx, len(site_idx))
        X = fs.design_matrix(si, ti, self.columns)
        out = {}
        for v, m in self.models.items():
            out[v] = from_model(v, m.predict(X)).reshape(len(site_idx), len(t_idx)).astype("float32")
        return out


def doy_location_gbm(cfg: LGBMConfig | None = None, seed: int = 0) -> TabularGBM:
    from .data import CALENDAR_COLS

    return TabularGBM("doy_location_gbm", DOY_LOCATION_STATIC + list(CALENDAR_COLS), cfg=cfg, seed=seed)


def forcing_gbm(fs: FeatureSet, cfg: LGBMConfig | None = None, seed: int = 0, grid=None) -> TabularGBM:
    return TabularGBM("forcing_gbm", fs.columns(), cfg=cfg, seed=seed, grid=grid)


# ----------------------------------------------------------------------------- air -> water

class AirWaterRegression(BaselineBase):
    """Pooled air→water temperature regression (linear or Mohseni logistic) on the trailing w-day
    mean air temperature; w chosen by inner-site RMSE."""

    targets = ("temp_c_mean", "temp_c_7dadm")

    def __init__(self, kind: str = "linear", windows=(1, 3, 7, 14, 30)):
        super().__init__()
        if kind not in ("linear", "mohseni"):
            raise ValueError(kind)
        self.kind, self.windows = kind, tuple(windows)
        self.name = f"air2water_{kind}"
        self.w = None
        self.params = None
        self.dadm_offset = None

    @staticmethod
    def _mohseni(x, mu, alpha, gamma, beta):
        return mu + (alpha - mu) / (1.0 + np.exp(np.clip(gamma * (beta - x), -50, 50)))

    def _fit_one(self, x, y):
        if self.kind == "linear":
            A = np.c_[np.ones_like(x), x]
            return np.linalg.lstsq(A, y, rcond=None)[0]
        from scipy.optimize import curve_fit

        p0 = (0.0, max(np.percentile(y, 99), 5.0), 0.2, np.median(x))
        p, _ = curve_fit(self._mohseni, x, y, p0=p0, maxfev=20000,
                         bounds=([-5, 0, 0.01, -20], [10, 45, 2.0, 40]))
        return p

    def _apply(self, x, p):
        if self.kind == "linear":
            return np.maximum(p[0] + p[1] * x, 0.0)
        return np.maximum(self._mohseni(x, *p), 0.0)

    def fit(self, data: HydroMetData, fs: FeatureSet, fit_idx, inner_idx, log=print):
        check_fit_sites(data, fit_idx, inner_idx)
        j = data.vi("temp_c_mean")
        si, ti = sample_rows(data, "temp_c_mean", fit_idx, max_per_site=2000)
        vs, vt = sample_rows(data, "temp_c_mean", inner_idx, max_per_site=2000)
        best = None
        for w in self.windows:
            k = fs.dyn_names.index(f"f_tmean_c_m{w}")
            x, y = fs.dyn[si, ti, k].astype("float64"), data.y[si, ti, j].astype("float64")
            ok = np.isfinite(x) & np.isfinite(y)
            p = self._fit_one(x[ok], y[ok])
            if len(vs):
                xv, yv = fs.dyn[vs, vt, k].astype("float64"), data.y[vs, vt, j].astype("float64")
                okv = np.isfinite(xv) & np.isfinite(yv)
                score = float(np.sqrt(np.mean((self._apply(xv[okv], p) - yv[okv]) ** 2)))
            else:
                score = float(np.sqrt(np.mean((self._apply(x[ok], p) - y[ok]) ** 2)))
            self.info[f"inner_rmse_w{w}"] = score
            if best is None or score < best[0]:
                best = (score, w, p)
        _, self.w, self.params = best
        self.dadm_offset = dadm_offset_by_month(data, np.r_[fit_idx, inner_idx])   # same cell as air2stream
        self.info.update({"window": self.w, "params": np.asarray(self.params).tolist(), "n_rows": int(len(si))})
        log(f"[{self.name}] window {self.w} d, params {np.round(self.params, 3).tolist()}, "
            f"inner RMSE {best[0]:.3f} °C")
        return self

    def predict(self, data: HydroMetData, fs: FeatureSet, site_idx, t_idx) -> dict:
        site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
        k = fs.dyn_names.index(f"f_tmean_c_m{self.w}")
        full = self._apply(fs.dyn[site_idx, :, k].astype("float64"), self.params)
        dadm = dadm_from_mean(full, self.dadm_offset, data.dates.month.to_numpy())
        return {"temp_c_mean": full[:, t_idx].astype("float32"), "temp_c_7dadm": dadm[:, t_idx].astype("float32")}

    def predict_full(self, data, fs, site_idx) -> np.ndarray:
        k = fs.dyn_names.index(f"f_tmean_c_m{self.w}")
        return self._apply(fs.dyn[np.asarray(site_idx, int), :, k].astype("float64"), self.params)


# ----------------------------------------------------------------------------- air2stream

def _air_temp(fs: FeatureSet, site_idx) -> np.ndarray:
    """Same-day (lag-shifted) daily-mean air temperature from the feature set (respects lag)."""
    return fs.dyn[np.asarray(site_idx, int), :, fs.dyn_names.index("f_tmean_c_m1")].astype("float64")


class Air2StreamRegional(BaselineBase):
    name = "air2stream_regional"
    targets = ("temp_c_mean", "temp_c_7dadm")

    def __init__(self, k_donors: int = 10, iters: int = 300, lr: float = 0.05, min_obs: int = 365,
                 min_calib_nse: float = 0.5, device: str = "cpu", threads: int = 2):
        super().__init__()
        self.threads = int(threads)
        self.k, self.iters, self.lr, self.min_obs, self.min_nse, self.device = (k_donors, iters, lr, min_obs,
                                                                                 min_calib_nse, device)
        self.donor_sites: list = []
        self.donor_params: dict = {}
        self.donor_x = None
        self.attr_mean = self.attr_std = None
        self.dadm_offset = None

    def _attr_matrix(self, data: HydroMetData, idx) -> np.ndarray:
        df = data.static.iloc[np.asarray(idx, int)].reindex(columns=DONOR_ATTRS).astype("float64").copy()
        for c in LOG_DONOR_ATTRS:
            df[c] = np.log10(df[c].clip(lower=0) + 1e-3)
        return df.to_numpy()

    def calibrate(self, data: HydroMetData, fs: FeatureSet, site_idx, log=print) -> dict:
        """Per-site least-squares calibration (Adam on bounded parameters, all sites in parallel)
        on the train block. Returns params {name: (S,)} and calibration NSE / n_obs."""
        import torch

        site_idx = np.asarray(site_idx, int)
        ta = _air_temp(fs, site_idx)
        train_end = int(np.flatnonzero(data.block_mask("train"))[-1]) + 1
        ta = ta[:, :train_end]
        mask = data.target_mask("temp_c_mean", "train", None)[site_idx, :train_end]
        yobs = np.where(mask, data.y[site_idx, :train_end, data.vi("temp_c_mean")], 0.0)
        from .physics import _ffill

        ta = _ffill(ta)
        dev = torch.device(self.device)
        T_ta = torch.tensor(ta, dtype=torch.float32, device=dev)
        T_y = torch.tensor(yobs, dtype=torch.float32, device=dev)
        T_m = torch.tensor(mask, dtype=torch.float32, device=dev)
        init = {"a1": 0.3, "a2": 0.1, "a3": 0.1, "a5": 0.2, "a6": 0.6}
        raw = {}
        for kname in A2S_PARAMS:
            lo, hi = A2S_BOUNDS[kname]
            p = (init[kname] - lo) / (hi - lo)
            raw[kname] = torch.full((len(site_idx),), float(np.log(p / (1 - p))), device=dev, requires_grad=True)
        opt = torch.optim.Adam(list(raw.values()), lr=self.lr)
        doy = data.dates.dayofyear.to_numpy()[:train_end]
        n_obs = T_m.sum(dim=1).clamp(min=1.0)
        # the recursion is ~5,000 tiny sequential ops per pass: intra-op threading only adds overhead
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            self._adam_loop(torch, raw, opt, T_ta, T_y, T_m, doy, n_obs, log)
        finally:
            torch.set_num_threads(prev_threads)
        return self._finish_calibration(torch, raw, ta, doy, mask, yobs, site_idx)

    def _adam_loop(self, torch, raw, opt, T_ta, T_y, T_m, doy, n_obs, log):
        for it in range(self.iters):
            opt.zero_grad()
            params = {k: A2S_BOUNDS[k][0] + (A2S_BOUNDS[k][1] - A2S_BOUNDS[k][0]) * torch.sigmoid(raw[k])
                      for k in A2S_PARAMS}
            sim = air2stream_simulate(T_ta, doy, params)
            loss_site = (((sim - T_y) ** 2) * T_m).sum(dim=1) / n_obs
            loss = loss_site.mean()
            loss.backward()
            opt.step()
            if it % 100 == 0 or it == self.iters - 1:
                log(f"[{self.name}] calibration iter {it}: mean site MSE {float(loss):.3f}")

    def _finish_calibration(self, torch, raw, ta, doy, mask, yobs, site_idx):
        with torch.no_grad():
            params = {k: (A2S_BOUNDS[k][0] + (A2S_BOUNDS[k][1] - A2S_BOUNDS[k][0]) * torch.sigmoid(raw[k]))
                      .cpu().numpy().astype("float64") for k in A2S_PARAMS}
        sim = air2stream_simulate(ta, doy, params)
        nse = np.full(len(site_idx), np.nan)
        for i in range(len(site_idx)):
            m = mask[i]
            if m.sum() >= 30:
                yy = yobs[i, m]
                nse[i] = 1 - np.sum((sim[i, m] - yy) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-9)
        return {"params": params, "nse": nse, "n_obs": mask.sum(axis=1)}

    def fit(self, data: HydroMetData, fs: FeatureSet, fit_idx, inner_idx, log=print):
        check_fit_sites(data, fit_idx, inner_idx)
        cand = np.asarray([i for i in np.r_[fit_idx, inner_idx]
                           if data.target_mask("temp_c_mean", "train", [i]).sum() >= self.min_obs], int)
        cal = self.calibrate(data, fs, cand, log=log)
        good = np.isfinite(cal["nse"]) & (cal["nse"] >= self.min_nse)
        self.donor_sites = [data.sites[i] for i in cand[good]]
        self.donor_params = {k: v[good] for k, v in cal["params"].items()}
        X = self._attr_matrix(data, cand[good])
        with warnings.catch_warnings():   # an attribute missing at every donor -> NaN -> neutral 0 below
            warnings.simplefilter("ignore", RuntimeWarning)
            self.attr_mean = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
        self.attr_std = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        self.donor_x = np.nan_to_num((X - self.attr_mean) / self.attr_std, nan=0.0)
        self.dadm_offset = dadm_offset_by_month(data, np.r_[fit_idx, inner_idx])
        self.info = {"n_candidates": int(len(cand)), "n_donors": int(good.sum()),
                     "calib_nse_median": float(np.nanmedian(cal["nse"])) if len(cand) else None,
                     "k_donors": self.k}
        log(f"[{self.name}] {good.sum()}/{len(cand)} donor sites (calibration NSE ≥ {self.min_nse}); "
            f"median calibration NSE {self.info['calib_nse_median']}")
        return self

    def predict_full(self, data: HydroMetData, fs: FeatureSet, site_idx) -> np.ndarray:
        site_idx = np.asarray(site_idx, int)
        if not len(self.donor_sites):
            raise RuntimeError("no donor sites")
        Xq = np.nan_to_num((self._attr_matrix(data, site_idx) - self.attr_mean) / self.attr_std, nan=0.0)
        d = np.sqrt(((Xq[:, None, :] - self.donor_x[None, :, :]) ** 2).sum(-1))
        # a target site that is itself a donor (only when predicting train sites) is excluded
        own = np.array([[s == t for t in self.donor_sites] for s in (data.sites[i] for i in site_idx)],
                       dtype=bool).reshape(len(site_idx), len(self.donor_sites))
        d[own] = np.inf
        k = min(self.k, self.donor_x.shape[0])
        nn = np.argsort(d, axis=1)[:, :k]
        w = 1.0 / (np.take_along_axis(d, nn, axis=1) + 0.1)
        w = w / w.sum(axis=1, keepdims=True)
        ta = _air_temp(fs, site_idx)
        rep_ta = np.repeat(ta, k, axis=0)
        params = {p: self.donor_params[p][nn.ravel()] for p in A2S_PARAMS}
        sim = air2stream_simulate(rep_ta, data.dates.dayofyear.to_numpy(), params).reshape(len(site_idx), k, -1)
        return (sim * w[:, :, None]).sum(axis=1)

    def predict(self, data: HydroMetData, fs: FeatureSet, site_idx, t_idx) -> dict:
        t_idx = np.asarray(t_idx, int)
        full = self.predict_full(data, fs, site_idx)
        dadm = dadm_from_mean(full, self.dadm_offset, data.dates.month.to_numpy())
        return {"temp_c_mean": full[:, t_idx].astype("float32"), "temp_c_7dadm": dadm[:, t_idx].astype("float32")}


# ----------------------------------------------------------------------------- DO saturation

class DOSatRatio(BaselineBase):
    """DO = DO_sat(T̂_w, P(elev)) × pooled median ratio by month (fit sites × train block)."""

    targets = ("do_mgl_mean", "do_mgl_min")

    def __init__(self, temp_model: BaselineBase, temp_label: str):
        super().__init__()
        self.temp_model = temp_model
        self.name = f"dosat_ratio[{temp_label}]"
        self.ratio: dict = {}

    def fit(self, data: HydroMetData, fs: FeatureSet, fit_idx, inner_idx, log=print):
        check_fit_sites(data, fit_idx, inner_idx)
        idx = np.r_[fit_idx, inner_idx].astype(int)
        p = pressure_atm_from_elevation(data.static["elev_m"].to_numpy())[idx][:, None]
        tw = data.y[idx, :, data.vi("temp_c_mean")].astype("float64")
        sat = np.where(np.isfinite(tw), do_sat_mgl(np.nan_to_num(tw), p), np.nan)
        month = data.dates.month.to_numpy()
        for v in self.targets:
            r = data.y[idx, :, data.vi(v)] / sat
            r[:, ~data.block_mask("train")] = np.nan
            out = np.full(13, np.nan)
            for mo in range(1, 13):
                x = r[:, month == mo]
                x = x[np.isfinite(x) & (x > 0) & (x < 3)]
                out[mo] = np.median(x) if x.size >= 30 else np.nan
            out[0] = np.nanmedian(out[1:]) if np.isfinite(out[1:]).any() else 1.0
            self.ratio[v] = np.where(np.isfinite(out), out, out[0])
            self.info[v] = {"ratio_by_month": np.round(self.ratio[v][1:], 4).tolist()}
            log(f"[{self.name}] {v}: monthly DO/DO_sat ratio {np.round(self.ratio[v][1:], 3).tolist()}")
        return self

    def predict(self, data: HydroMetData, fs: FeatureSet, site_idx, t_idx) -> dict:
        site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
        tw = self.temp_model.predict(data, fs, site_idx, t_idx)["temp_c_mean"].astype("float64")
        p = pressure_atm_from_elevation(data.static["elev_m"].to_numpy())[site_idx][:, None]
        sat = do_sat_mgl(np.maximum(tw, 0.0), p)
        mo = data.dates.month.to_numpy()[t_idx]
        return {v: (sat * self.ratio[v][mo][None, :]).astype("float32") for v in self.targets}


@dataclass
class BaselineSuite:
    """All fitted baselines of one run, keyed by name."""

    models: dict = field(default_factory=dict)

    def add(self, m: BaselineBase):
        self.models[m.name] = m
        return m


def fit_suite(cfg: dict, data: HydroMetData, fs: FeatureSet, fit_idx, inner_idx, log=print) -> BaselineSuite:
    """Build and fit the baselines named in ``cfg['models']`` (dependencies are fitted first)."""
    import dataclasses

    names = list(cfg.get("models", []))
    lcfg = LGBMConfig(**(cfg.get("lgbm") or {}))
    seed = int(cfg.get("seed", 0))
    suite = BaselineSuite()
    # e.g. "_nogauge" for the static-input ablation (protocol §7); "_bag_s{seed}" for the seed-bagged GBM
    suffix = str(cfg.get("name_suffix", "")).format(seed=seed)
    built: dict = {}

    def need(name):
        if name in built:
            return built[name]
        if name == "doy_location_gbm":
            m = doy_location_gbm(lcfg, seed)
        elif name == "forcing_gbm":
            grid = [dataclasses.replace(lcfg, **g) for g in (cfg.get("forcing_gbm_grid") or [{}])]
            m = forcing_gbm(fs, lcfg, seed, grid=grid)
        elif name in ("air2water_linear", "air2water_mohseni"):
            m = AirWaterRegression(name.split("_")[1], tuple(cfg.get("air2water_windows", (1, 3, 7, 14, 30))))
        elif name == "air2stream_regional":
            m = Air2StreamRegional(**(cfg.get("air2stream") or {}))
        elif name.startswith("dosat_ratio[") and name.endswith("]"):
            src = name[len("dosat_ratio["):-1]
            m = DOSatRatio(need(src), src)
        else:
            raise ValueError(f"unknown baseline {name!r}")
        m.name = m.name + suffix
        log(f"[baselines] fitting {m.name}")
        m.fit(data, fs, fit_idx, inner_idx, log=log)
        built[name] = m
        return suite.add(m)

    for n in names:
        need(n)
    return suite
