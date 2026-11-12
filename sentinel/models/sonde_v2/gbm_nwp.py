# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GBM-NWP and PF-GBM references (``sonde_arch_v2_amendment.md`` §2b; lead rulings B, D, E, M).

Subclass of the frozen-candidate ``SondeGBM`` (whose file is not edited): the same lag/roll/context/weather/static
feature table, plus per-horizon NWP trajectory features of the ONE declared product (:func:`nwp.gbm_nwp_blocks`)
and, for PF-GBM, uncalibrated physics features (:mod:`physics_features`). Uncertainty configurations:

* ``quantile`` with ``quantile_levels=19`` (config Q19, the stamped variant and the step-1 comparator): LightGBM
  quantile objectives at all 19 forecast levels, sorted;
* ``quantile`` with ``quantile_levels=7`` (config Q): 7 levels (5, 10, 25, 50, 75, 90, 95 %), sorted, then linearly
  interpolated / tail-extrapolated (in level-z space) to the 19 forecast levels;
* ``location_scale`` (config LS): an L1 location model plus an input-dependent scale model (L2 on the log of the
  |standardised residual| of the FIT rows), with the 19 empirical quantiles of the re-standardised residuals of the
  EARLY-STOPPING rows.

Rows (identical rule in step 1 and CV, for both GBM arms):
* fit rows = the frozen ``training_rows`` of the fit sites (issue days of the TRAIN block with target day
  issue + h ≤ 2021-09-30, a recent observation and an observed target) ∩ the reference-window issue rule
  (``data.issue_ok``) ∩ a finite anomaly target. No ``fit_end`` and no calibration slice inside the train block
  (``fit_cal_masks`` of the frozen file is not used).
* FULL ROWS (lead GBM ruling of 2026-11-10, G-2): every fit row that passes those masks is used; no cap is applied
  to fit rows (``max_fit_rows`` must be ``None``; :meth:`SondeGBMNWP.fit_var_nwp` refuses anything else).
* early-stopping rows = the same rule on the early-stopping sites (the inner 15 % of in-fold basin groups), after
  the finite-target mask, a seeded site-stratified sample of at most ``es_max_rows`` (:func:`stratified_cap`).
A (variable, horizon) with < ``MIN_FIT_ROWS`` fit rows or < ``MIN_ES_ROWS`` early-stopping rows is skipped (NaN
forecast). The NWP input is ONE (S, T, L, F) bias-corrected tensor per use; a list is a hard error.

Boosters (G-3, G-6): every booster is fitted whole, saved to its own file with its stopping round and wall time and
read back from that file (:class:`gbm_resume.UnitStore`); the file is the model that forecasts. A booster with a
record is not fitted again. The boosters of one arm, seed, fold, variable and horizon lie in one directory and are
named by what they are (``q05`` … ``q95``, ``ls_loc``, ``ls_scale``), so configuration Q uses the seven boosters of
Q19 whose key (rows, targets, objective, level, seed, settings) is identical instead of fitting them again.
Device ``gpu`` = LightGBM ``device_type=gpu``, ``gpu_use_dp=False``, the declared ``max_bin``; it needs a
:class:`gbm_resume.GBMProcess` that holds the GPU lock.
"""

from __future__ import annotations

import tempfile
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from sentinel.models.baselines.sonde_baselines import QUANTILES
from sentinel.models.sonde.common import Rows, SondeData
from sentinel.models.sonde.features import FeatureBuilder
from sentinel.models.sonde.gbm import GBMConfig, SondeGBM

from . import nwp as NWP
from . import physics_features as PF
from .prep import NOT_MODELLED

QLEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
MIN_FIT_ROWS = 1000
MIN_ES_ROWS = 200


@dataclass
class GBMNWPConfig(GBMConfig):
    quantile_mode: str = "quantile"        # "quantile" | "location_scale"
    quantile_levels: int = 19              # 19 (fitted directly; config Q19) | 7 (interpolated to 19; config Q)
    physics: bool = False                  # PF-GBM
    num_threads: int = 6
    dd_base_c: float = 0.0
    ew_decay: float = 0.7
    max_bin: int = 255                     # LightGBM default, written out so CPU and GPU fits use the same value
    device: str = "cpu"                    # "gpu" in every declared run (G-6); "cpu" only in unit tests
    gpu_device_id: int = 0                 # OpenCL index inside CUDA_VISIBLE_DEVICES (= the one locked GPU)


def qname(a) -> str:
    """File name of the quantile booster of level ``a`` (0.05 → ``q05``)."""
    return f"q{int(round(100 * float(a))):02d}"


def interp_quantiles(qk: np.ndarray, levels=QLEVELS, out_levels=QUANTILES) -> np.ndarray:
    """(N, K) sorted quantiles at ``levels`` → (N, 19): linear in normal-score space, linear tail extrapolation."""
    z_in = norm.ppf(np.asarray(levels))
    z_out = norm.ppf(np.asarray(out_levels))
    qk = np.sort(qk, axis=1)
    out = np.empty((qk.shape[0], len(z_out)))
    for j, z in enumerate(z_out):
        i = int(np.clip(np.searchsorted(z_in, z) - 1, 0, len(z_in) - 2))
        w = (z - z_in[i]) / (z_in[i + 1] - z_in[i])
        out[:, j] = qk[:, i] + w * (qk[:, i + 1] - qk[:, i])
    return np.sort(out, axis=1)


def stratified_cap(si: np.ndarray, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    """Indices (sorted) of a site-stratified subsample of exactly ``cap`` rows: every site keeps a share
    proportional to its row count (largest-remainder rounding), rows drawn at random within the site."""
    n = int(si.size)
    if cap is None or n <= cap:
        return np.arange(n)
    _, inv, cnt = np.unique(si, return_inverse=True, return_counts=True)
    exact = cap * cnt / n
    quota = np.floor(exact).astype(np.int64)
    rem = int(cap - quota.sum())
    if rem > 0:
        quota[np.argsort(-(exact - quota), kind="stable")[:rem]] += 1
    order = np.lexsort((rng.random(n), inv))                  # by site, random within site
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n) - np.repeat(start, cnt)
    return np.flatnonzero(rank < quota[inv])


def _one_product(nwp_x) -> np.ndarray:
    if not isinstance(nwp_x, np.ndarray) or nwp_x.ndim != 4:
        raise ValueError("exactly one declared NWP product tensor (S, T, L, F) is allowed; got "
                         f"{type(nwp_x).__name__}")
    return nwp_x


class SondeGBMNWP(SondeGBM):
    def __init__(self, cfg: GBMNWPConfig | None = None, horizons=None):
        cfg = cfg or GBMNWPConfig()
        super().__init__(cfg, horizons=horizons or (1, 2, 3, 5, 7, 10, 14))
        self.qmodels: dict = {}     # (var, h) -> list of boosters (quantile mode)
        self.smodels: dict = {}     # (var, h) -> scale booster (location_scale)
        self.rows_info: dict = {}   # (var, h) -> row counts and fit window

    # ------------------------------------------------------------------ features
    def _builder(self, data: SondeData) -> FeatureBuilder:
        """Frozen feature builder over every cube variable except the series no model reads (``NOT_MODELLED``)."""
        if self._fb is None or self._fb.cube is not data.cube:
            self._fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static,
                                      variables=[v for v in data.cube.variables if v not in NOT_MODELLED])
        return self._fb

    def _params(self) -> dict:
        p = {**super()._params(), "max_bin": int(self.cfg.max_bin)}
        if self.cfg.device == "gpu":
            p.pop("force_row_wise", None)
            p.update({"device_type": "gpu", "gpu_device_id": int(self.cfg.gpu_device_id), "gpu_use_dp": False})
        elif self.cfg.device != "cpu":
            raise ValueError(self.cfg.device)
        return p

    def _nwp_blocks(self, nwp_x, h: int) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return NWP.gbm_nwp_blocks(_one_product(nwp_x), h, base_c=self.cfg.dd_base_c, decay=self.cfg.ew_decay)

    def _phys_blocks(self, data: SondeData, nwp_x, h: int) -> dict:
        nwp_x = _one_product(nwp_x)
        fb = self._builder(data)
        out = {}
        S, T = nwp_x.shape[:2]
        L = min(h, nwp_x.shape[2])
        tair = nwp_x[:, :, :L, NWP.F_IDX["tmean_c"]].reshape(S * T, L)
        if "temp_c_mean" in data.cube.variables:
            tw_last = fb.last("temp_c_mean")[0].reshape(-1)          # model space == °C for temperature
            tw = PF.air2stream_forecast(tw_last, tair)[:, -1].reshape(S, T)
            out["pf_tw_target"] = tw
            elev = data.static["elev_m"].to_numpy(float)
            out["pf_dosat_target"] = PF.dosat_forecast(tw, elev)
        q_last = None
        if "discharge_cfs_mean" in data.cube.variables:
            from sentinel.models.sonde.data import from_model_space
            q_last = from_model_space("discharge_cfs_mean", fb.last("discharge_cfs_mean")[0])
            slope = data.static["slope"].to_numpy(float) if "slope" in data.static else np.full(S, np.nan)
            out["pf_k600"] = PF.k600_raymond(q_last, slope[:, None])
        if q_last is not None and "spcond_uscm_mean" in data.cube.variables:
            from sentinel.models.sonde.data import from_model_space
            sc_last = from_model_space("spcond_uscm_mean", fb.last("spcond_uscm_mean")[0])
            p = nwp_x[:, :, :L, NWP.F_IDX["precip_mm"]]
            p_sum = np.where(np.isfinite(p).any(axis=2), np.nansum(p, axis=2), np.nan)   # NaN, not 0, if missing
            area = np.exp(data.static["log_drain_area_km2"].to_numpy(float))[:, None]
            out["pf_sc_mix"] = PF.sc_dilution(sc_last, q_last, p_sum, area, h)
        return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}

    def _det_blocks(self, data: SondeData, h: int) -> dict:
        """Deterministic target-day inputs given to every comparator (deviations file, entry on inputs beyond the stamped description): clear-sky shortwave and
        day length of day t + h (DOY sin/cos of the target day is already in the frozen feature table)."""
        from .data_v2 import clear_sky_sw, day_length
        lat = data.static["lat"].to_numpy(float)
        lat = np.where(np.isfinite(lat), lat, 40.0)
        doy = (data.cube.dates.dayofyear.to_numpy() - 1 + h) % 365
        D = np.broadcast_to(doy[None, :], (lat.size, doy.size))
        return {"det_clearsky_sw": clear_sky_sw(lat, D).astype(np.float32),
                "det_day_length": day_length(lat, D).astype(np.float32)}

    def blocks_h(self, data: SondeData, var: str, nwp_x, h: int) -> dict:
        fb = self._builder(data)
        b = {**fb.target_block(var), **fb.context_block(exclude=var), **fb.weather_block(data.weather),
             **self._nwp_blocks(nwp_x, h), **self._det_blocks(data, h)}
        if self.cfg.physics:
            b.update(self._phys_blocks(data, nwp_x, h))
        return b

    # ------------------------------------------------------------------ rows
    def labelled_rows(self, data, var: str, h: int, sites: np.ndarray, *, cap: int | None, salt: int):
        """(si, ti, y, s, n_total): the frozen training rows of ``sites`` ∩ reference-window issue rule ∩ finite
        target, then the seeded site-stratified cap. ``n_total`` = rows before the cap."""
        fb = self._builder(data)
        si, ti = self.training_rows(data, var, h, np.asarray(sites))
        keep = data.issue_ok(var, si, ti)
        si, ti = si[keep], ti[keep]
        y, s = self._target(fb, var, si, ti, h)
        ok = np.isfinite(y) & np.isfinite(s)
        si, ti, y, s = si[ok], ti[ok], y[ok], s[ok]
        n_total = int(si.size)
        rng = np.random.default_rng([int(self.cfg.seed), zlib.crc32(var.encode()), int(h), int(salt)])
        pick = stratified_cap(si, cap, rng)
        return si[pick], ti[pick], y[pick], s[pick], n_total

    # ------------------------------------------------------------------ fit
    def fit_var_nwp(self, data, var: str, *, train_sites: np.ndarray, es_sites: np.ndarray, nwp_fit: np.ndarray,
                    nwp_es: np.ndarray | None = None, store_root=None, process=None, config: str = "adhoc",
                    log=print) -> dict:
        """Fit (or read from their files) every booster of ``var``. ``store_root``: booster directory of this arm,
        seed and fold (a temporary one if None); ``process``: the :class:`gbm_resume.GBMProcess` of the driver;
        ``config``: declared configuration name (unit record). Raises ``gbm_resume.ProcessBudgetReached`` when
        the process may not start the next booster; calling again continues."""
        import lightgbm as lgb

        from . import gbm_resume as GR

        if self.cfg.max_fit_rows is not None:
            raise ValueError("GBM ruling G-2: fit rows are never capped (max_fit_rows must be None)")
        if process is None:
            if self.cfg.device != "cpu":
                raise ValueError("a GPU fit needs the GBMProcess of the driver (GPU lock, G-6)")
            process = GR.GBMProcess("adhoc", "adhoc", self.cfg, log=log)
        tmp = None
        if store_root is None:
            tmp = tempfile.TemporaryDirectory(prefix="sonde_v2_gbm_")
            store_root = tmp.name
        fb = self._builder(data)
        j = data.cube.var_index(var)
        cs_train = data.clim_std[train_sites, :, j]
        self.scale_floor[var] = float(self.cfg.scale_floor_frac * np.nanmedian(cs_train)) \
            if np.isfinite(cs_train).any() else 1e-3
        nwp_es = nwp_fit if nwp_es is None else nwp_es
        dates = data.cube.dates
        out = {}
        lock = GR.unit_lock(Path(store_root) / var)
        try:
            for h in self.horizons:
                out[h] = self._fit_unit(lgb, GR, data, fb, var, h, train_sites, es_sites, nwp_fit, nwp_es, dates,
                                        Path(store_root) / var / f"h{h}", process, config)
                log(f"[gbm_nwp] {var} h={h}: {out[h]}")
        finally:
            lock.close()
            if tmp is not None:
                tmp.cleanup()
        self.diag[var] = out
        return out

    def _fit_unit(self, lgb, GR, data, fb, var, h, train_sites, es_sites, nwp_fit, nwp_es, dates, directory,
                  process, config) -> dict:
        si, ti, y_fit, s_fit, n_total = self.labelled_rows(data, var, h, train_sites, cap=None, salt=0)
        if int(si.size) != n_total:
            raise RuntimeError("GBM ruling G-2: the fit rows are not the full masked rows")
        esi, eti, y_es, s_es, n_es_total = self.labelled_rows(data, var, h, es_sites,
                                                              cap=self.cfg.es_max_rows, salt=1)
        info = {"n_fit_total": n_total, "n_fit": int(si.size), "max_fit_rows": None,
                "n_es_total": n_es_total, "n_es": int(esi.size), "es_max_rows": self.cfg.es_max_rows,
                "fit_rows_sha256": GR.array_sha(si, ti), "es_rows_sha256": GR.array_sha(esi, eti)}
        if si.size:
            info.update({"fit_issue_min": str(dates[ti.min()].date()), "fit_issue_max": str(dates[ti.max()].date()),
                         "fit_target_max": str((dates[ti.max()] + np.timedelta64(h, "D")).date()),
                         "fit_sites": int(np.unique(si).size), "es_sites": int(np.unique(esi).size)})
        self.rows_info[(var, h)] = info
        unit = {"variable": var, "h": int(h), "seed": int(self.cfg.seed), "quantile_mode": self.cfg.quantile_mode,
                "quantile_levels": int(self.cfg.quantile_levels), "physics": bool(self.cfg.physics), **info}
        if si.size < MIN_FIT_ROWS or esi.size < MIN_ES_ROWS:
            GR.UnitStore(directory, config=config, process=process, info={**unit, "skipped": True}).close("skipped")
            return {"skipped": True, **info}
        X_fit = fb.table(var, si, ti, h, blocks=self.blocks_h(data, var, nwp_fit, h))
        X_es = fb.table(var, esi, eti, h, blocks=self.blocks_h(data, var, nwp_es, h))
        self.feature_columns = cols = list(X_fit.columns)
        base = self._params()
        xf, xe = GR.frame_sha(X_fit), GR.frame_sha(X_es)
        unit.update({"skipped": False, "n_columns": len(cols), "columns_sha256": GR._sha(cols), "x_fit_sha256": xf,
                     "x_es_sha256": xe, "settings": {k: base[k] for k in sorted(base)},
                     "max_rounds": int(self.cfg.max_rounds),
                     "early_stopping_rounds": int(self.cfg.early_stopping_rounds)})
        store = GR.UnitStore(directory, config=config, process=process, info=unit)
        metas = {}

        def booster(name, params, y, yes):
            key, parts = GR.booster_key(params, max_rounds=self.cfg.max_rounds,
                                        early_stopping_rounds=self.cfg.early_stopping_rounds, x_fit=xf,
                                        y_fit=GR.array_sha(y), x_es=xe, y_es=GR.array_sha(yes), columns=cols)

            def fit(counter):
                ds = lgb.Dataset(X_fit, y, free_raw_data=False)
                vs = [lgb.Dataset(X_es, yes, reference=ds)]
                return lgb.train(params, ds, num_boost_round=self.cfg.max_rounds, valid_sets=vs,
                                 callbacks=[lgb.log_evaluation(0), counter,
                                            lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)])

            b, meta = store.get(name, key, parts, fit, check=X_es)
            metas[name] = GR.booster_brief(meta)
            return b

        try:
            if self.cfg.quantile_mode == "quantile":
                levels = QUANTILES if self.cfg.quantile_levels == 19 else QLEVELS
                boosters = [booster(qname(a), {**base, "objective": "quantile", "alpha": float(a)}, y_fit, y_es)
                            for a in levels]
                self.qmodels[(var, h)] = boosters
                pc = np.stack([b.predict(X_es) for b in boosters], 1)
                q19 = np.sort(pc, axis=1) if self.cfg.quantile_levels == 19 else interp_quantiles(pc)
                res = {**info, "es_cov90": float(np.mean((y_es >= q19[:, 0]) & (y_es <= q19[:, -1])))}
            elif self.cfg.quantile_mode == "location_scale":
                loc = booster("ls_loc", {**base, "objective": "l1"}, y_fit, y_es)
                r_fit = (y_fit - loc.predict(X_fit)) / np.maximum(s_fit, self.scale_floor[var])
                r_es0 = (y_es - loc.predict(X_es)) / np.maximum(s_es, self.scale_floor[var])
                sc = booster("ls_scale", {**base, "objective": "l2"}, np.log(np.abs(r_fit) + 1e-3),
                             np.log(np.abs(r_es0) + 1e-3))
                self.models[(var, h)] = loc
                self.smodels[(var, h)] = sc
                r = r_es0 / np.exp(sc.predict(X_es))
                r = r[np.isfinite(r)]
                self.res_q[(var, h)] = np.quantile(r, QUANTILES)        # residual quantiles: early-stopping rows
                res = {**info, "n_residual_rows": int(r.size)}
            else:
                raise ValueError(self.cfg.quantile_mode)
        except BaseException as e:
            store.close(f"error:{type(e).__name__}")       # no effect if the store has closed itself (budget, …)
            raise
        store.close("complete")
        return {**res, "rounds": [m["best_iteration"] for m in metas.values()], "boosters": metas}

    def predict_rows_nwp(self, data: SondeData, var: str, rows: Rows, nwp_eval: np.ndarray) -> np.ndarray:
        """Forecast quantiles. Every booster is the one read from its file (best iteration only): plain predict."""
        fb = self._builder(data)
        q = np.full((len(rows), len(self.horizons), len(QUANTILES)), np.nan, dtype=np.float32)
        si, ti = rows.si.astype(np.int64), rows.ti.astype(np.int64)
        for k, h in enumerate(self.horizons):
            if (var, h) not in self.qmodels and (var, h) not in self.smodels:
                continue
            X = fb.table(var, si, ti, h, blocks=self.blocks_h(data, var, nwp_eval, h))
            cm_t, cs_t = fb.clim_at(var, ti, si, shift=h)
            if (var, h) in self.qmodels:
                pk = np.stack([b.predict(X) for b in self.qmodels[(var, h)]], 1)
                qq = np.sort(pk, axis=1) if pk.shape[1] == 19 else interp_quantiles(pk)
                q[:, k] = cm_t[:, None] + qq
            else:
                loc, sc = self.models[(var, h)], self.smodels[(var, h)]
                a_hat = loc.predict(X)
                s = np.maximum(cs_t, self.scale_floor[var]) * np.exp(sc.predict(X))
                q[:, k] = (cm_t + a_hat)[:, None] + s[:, None] * self.res_q[(var, h)][None, :]
        return q
