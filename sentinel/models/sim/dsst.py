# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train-cool / test-hot differential split-sample evaluation (protocol §5.4).

All comparators are re-fitted on the COOL water years only (``fit_days``), so no statistic sees a hot
year: :class:`NaiveFit` re-implements M1's persistence and climatology with an arbitrary fit mask
(equivalence with :class:`SondeBaselines` for the train-block mask is unit-tested), physics_only uses
:class:`PhysicsOnly(fit_t=…)` and the Sim error correction :class:`SimEC(fit_days=…)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.stats import norm

from ..baselines.sonde_baselines import QUANTILES, last_value
from . import data as D
from .baselines import MIN_FIT, PhysicsOnly
from .hindcast import SIM_MODELS, SimEC, crpss_ci, row_crps


def water_year(dates: pd.DatetimeIndex) -> np.ndarray:
    return (dates + pd.DateOffset(months=3)).year.to_numpy()


def fit_climatology(z: np.ndarray, dates: pd.DatetimeIndex, fit_days: np.ndarray, window: int = 15,
                    min_years: int = 3):
    """M1 climatology algorithm (circular ±window DOY smoothing, min-years fallback) on ``fit_days``."""
    S, T = z.shape
    doy = dates.dayofyear.to_numpy() - 1
    yidx = dates.year.to_numpy() - dates.year.min()
    n_years = int(yidx.max()) + 1
    k = np.ones(2 * window + 1)

    def circ(a):
        return np.convolve(np.concatenate([a[-window:], a, a[:window]]), k, "valid")

    mean = np.full((S, 366), np.nan)
    std = np.full((S, 366), np.nan)
    zf = np.where(fit_days[None, :], z, np.nan)
    for i in range(S):
        zi = zf[i]
        ok = np.isfinite(zi)
        if ok.sum() < 365:
            continue
        s_sum = circ(np.bincount(doy[ok], weights=zi[ok], minlength=366))
        s_sq = circ(np.bincount(doy[ok], weights=zi[ok] ** 2, minlength=366))
        s_cnt = circ(np.bincount(doy[ok], minlength=366).astype(float))
        pres = np.zeros((366, n_years), dtype=bool)
        pres[doy[ok], yidx[ok]] = True
        support = maximum_filter1d(pres.astype(np.uint8), size=2 * window + 1, axis=0, mode="wrap").sum(axis=1)
        m = s_sum / np.maximum(s_cnt, 1)
        var = np.maximum(s_sq / np.maximum(s_cnt, 1) - m ** 2, 1e-8)
        fb = support < min_years
        m[fb] = np.mean(zi[ok])
        var[fb] = np.var(zi[ok]) + 1e-8
        mean[i], std[i] = m, np.sqrt(var)
    return mean, std


class NaiveFit:
    """Persistence + climatology fitted on ``fit_days`` (model-space z (S, T))."""

    def __init__(self, z: np.ndarray, dates: pd.DatetimeIndex, fit_days: np.ndarray, horizons):
        self.z, self.dates = z, dates
        self.doy = dates.dayofyear.to_numpy() - 1
        self.cm, self.cs = fit_climatology(z, dates, fit_days)
        self.last, _ = last_value(z)
        S, T = z.shape
        self.chg_q = {}
        for h in horizons:
            pair = fit_days[: T - h] & fit_days[h:]
            chg = np.where(pair[None, :] & np.isfinite(self.last[:, : T - h]) & np.isfinite(z[:, h:]),
                           z[:, h:] - self.last[:, : T - h], np.nan)
            q = np.full((S, len(QUANTILES)), np.nan)
            for i in range(S):
                c = chg[i][np.isfinite(chg[i])]
                if c.size >= MIN_FIT:
                    q[i] = np.quantile(c, QUANTILES)
            self.chg_q[h] = q

    def forecast(self, model: str, t_idx: np.ndarray, h: int) -> np.ndarray:
        d_tgt = self.doy[np.minimum(t_idx + h, len(self.doy) - 1)]
        if model == "climatology":
            return self.cm[:, d_tgt][..., None] + self.cs[:, d_tgt][..., None] * norm.ppf(QUANTILES)[None, None, :]
        if model == "persistence":
            return self.last[:, t_idx][..., None] + self.chg_q[h][:, None, :]
        raise ValueError(model)

    def scorable(self, t_idx, h):
        tgt = t_idx + h
        return np.isfinite(self.last[:, t_idx]) & np.isfinite(self.z[:, tgt]) & np.isfinite(self.cm[:, self.doy[t_idx]])


def dsst_rows(d: D.SimData, sim_daily: dict, eval_sites: np.ndarray, fit_years, eval_years, *,
              horizons=(3, 7), variables=("temp_c_mean", "do_mgl_mean")) -> dict:
    wy = water_year(d.dates)
    in_train = D.block_mask(d.dates, "train")
    fit_days = in_train & np.isin(wy, list(fit_years))
    phys = PhysicsOnly(d, fit_t=np.flatnonzero(fit_days))
    out = {}
    for v in variables:
        j = D.TARGETS.index(v)
        z = D.to_model(v, d.y[:, :, j])
        nf = NaiveFit(z, d.dates, fit_days, horizons)
        ec = SimEC(z, np.asarray(sim_daily[v], float), fit_days, d.idx("cal") if fit_years is not None else [],
                   horizons)
        for h in horizons:
            ev = in_train & np.isin(wy, list(eval_years))
            T = len(d.dates)
            t_idx = np.flatnonzero(ev[: T - h] & ev[h:])        # issue and target in an evaluation year
            ok = nf.scorable(t_idx, h)[eval_sites]
            fc = {m: nf.forecast(m, t_idx, h)[eval_sites] for m in ("persistence", "climatology")}
            fc["physics_only"] = phys.forecast(v, t_idx, h)[eval_sites]
            for m in SIM_MODELS:
                fc[m] = ec.forecast(m, t_idx, h)[eval_sites]
            for q in fc.values():
                ok &= np.isfinite(q).all(axis=-1)
            si, ti = np.nonzero(ok)
            if not si.size:
                continue
            y = z[eval_sites][si, t_idx[ti] + h]
            groups = d.basin_group[eval_sites][si]
            out[(v, h)] = {"y": y, "groups": groups, "n_sites": len(np.unique(si)),
                           "q": {m: np.sort(q[si, ti], axis=-1) for m, q in fc.items()}}
    return out


def score_dsst(rows: dict, n_boot: int = 2000) -> pd.DataFrame:
    recs = []
    for (v, h), r in rows.items():
        crps = {m: row_crps(r["y"], q) for m, q in r["q"].items()}
        for m in SIM_MODELS:
            rec = {"var": v, "h": h, "model": m, "n_rows": len(r["y"]), "n_sites": r["n_sites"],
                   "crps": float(crps[m].mean())}
            for ref in ("persistence", "climatology", "physics_only"):
                ci = crpss_ci(crps[m], crps[ref], r["groups"], n_boot=n_boot)
                rec.update({f"crpss_vs_{ref}": ci["point"], f"crpss_vs_{ref}_lo": ci["ci_low"],
                            f"crpss_vs_{ref}_hi": ci["ci_high"]})
            recs.append(rec)
        for ref in ("persistence", "climatology", "physics_only"):
            recs.append({"var": v, "h": h, "model": ref, "n_rows": len(r["y"]), "n_sites": r["n_sites"],
                         "crps": float(crps[ref].mean())})
    return pd.DataFrame(recs)
