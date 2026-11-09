# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Issue-date hindcast benchmark for SENTINEL-Sim (protocol §4).

At issue day d (valid per ``SondeSplit.time_mask`` for the block, 42-day embargo) and horizon h the
Sim variants predict the target-variable value at d + h (model space):

* ``sim_raw``       — the simulation itself (driven by observed weather + discharge; no WQ input);
* ``sim_ec_site``   — sim(d+h) + φ_h,site · r_last(d), where r = obs − sim and r_last is the latest
                       residual ≤ 3 days old; φ and residual quantiles fitted on the SAME site's
                       train-block data (same information set as the persistence/climatology baselines);
* ``sim_ec_pooled`` — the same, but φ and quantiles pooled over cal sites (no site history at all).

Comparators: persistence, climatology, damped_anomaly (M1 code) and physics_only (:mod:`.baselines`).
Common rows: :meth:`SondeBaselines.scorable` ∧ every model finite. Score: per-row CRPS from 19
quantiles (2 × mean pinball loss), pooled; CRPSS = 1 − ΣCRPS_model / ΣCRPS_ref with a basin-group
cluster bootstrap (B = 2000) of the paired ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...eval import metrics as EM
from ..baselines.sonde_baselines import QUANTILES, last_value
from . import data as D
from .baselines import MIN_FIT, PhysicsOnly, naive_baselines

HORIZONS = (1, 3, 7, 14)
PRIMARY_H = (3, 7)
PRIMARY_VARS = ("temp_c_mean", "do_mgl_mean")
REFS = ("persistence", "climatology", "damped_anomaly", "physics_only")
SIM_MODELS = ("sim_raw", "sim_ec_site", "sim_ec_pooled")


def row_crps(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row CRPS ≈ 2 · mean_τ pinball_τ for y (n,), q (n, 19)."""
    diff = y[:, None] - q
    qs = QUANTILES[None, :]
    return 2.0 * np.mean(np.maximum(qs * diff, (qs - 1) * diff), axis=1)


class SimEC:
    """Error-corrected simulation for one variable (model space arrays (S, T))."""

    def __init__(self, z_obs: np.ndarray, sim: np.ndarray, fit_days: np.ndarray, cal_rows: np.ndarray,
                 horizons=HORIZONS):
        """``fit_days`` (T,) bool: days whose (issue, target) pairs may be used for fitting φ / quantiles
        (both the issue day and the target day must be fit days)."""
        self.z, self.sim = z_obs, sim
        self.r = z_obs - sim
        self.r_last, _ = last_value(self.r)
        self.h = tuple(horizons)
        fit_days = np.asarray(fit_days, bool)
        T = z_obs.shape[1]
        S = z_obs.shape[0]
        self.phi_site, self.q_site, self.phi_pool, self.q_pool, self.q_raw_site, self.q_raw_pool = {}, {}, {}, {}, {}, {}
        for h in self.h:
            pair = fit_days[: T - h] & fit_days[h:]
            a0 = np.where(pair[None, :], self.r_last[:, : T - h], np.nan)
            a1 = np.where(pair[None, :], self.r[:, h:], np.nan)
            ok = np.isfinite(a0) & np.isfinite(a1)
            num = np.nansum(np.where(ok, a0 * a1, 0), axis=1)
            den = np.nansum(np.where(ok, a0 * a0, 0), axis=1)
            phi_s = np.clip(np.where(den > 0, num / np.maximum(den, 1e-12), np.nan), 0.0, 1.0)
            c = cal_rows
            phi_p = float(np.clip(num[c].sum() / max(den[c].sum(), 1e-12), 0.0, 1.0)) if len(c) else 0.0
            enough = ok.sum(axis=1) >= MIN_FIT
            phi_s = np.where(enough, phi_s, phi_p)
            res_s = np.where(ok, a1 - phi_s[:, None] * a0, np.nan)
            res_p = np.where(ok, a1 - phi_p * a0, np.nan)
            raw = np.where(np.isfinite(a1), a1, np.nan)
            qs = np.full((S, len(QUANTILES)), np.nan)
            qr = np.full((S, len(QUANTILES)), np.nan)
            for i in np.flatnonzero(enough):
                qs[i] = np.nanquantile(res_s[i], QUANTILES)
            rr = raw[:, :]
            for i in range(S):
                ri = rr[i][np.isfinite(rr[i])]
                if ri.size >= MIN_FIT:
                    qr[i] = np.quantile(ri, QUANTILES)
            pool_res = res_p[c][np.isfinite(res_p[c])] if len(c) else np.array([])
            pool_raw = raw[c][np.isfinite(raw[c])] if len(c) else np.array([])
            qp = np.quantile(pool_res, QUANTILES) if pool_res.size else np.full(len(QUANTILES), np.nan)
            qrp = np.quantile(pool_raw, QUANTILES) if pool_raw.size else np.full(len(QUANTILES), np.nan)
            self.phi_site[h], self.phi_pool[h] = phi_s, phi_p
            self.q_site[h] = np.where(np.isfinite(qs), qs, qp[None, :])
            self.q_pool[h] = qp
            self.q_raw_site[h] = np.where(np.isfinite(qr), qr, qrp[None, :])
            self.q_raw_pool[h] = qrp

    def forecast(self, model: str, t_idx: np.ndarray, h: int) -> np.ndarray:
        tgt = np.minimum(t_idx + h, self.sim.shape[1] - 1)
        s = self.sim[:, tgt]
        if model == "sim_raw":
            return s[..., None] + self.q_raw_site[h][:, None, :]
        if model == "sim_ec_site":
            return (s + self.phi_site[h][:, None] * self.r_last[:, t_idx])[..., None] + self.q_site[h][:, None, :]
        if model == "sim_ec_pooled":
            return (s + self.phi_pool[h] * self.r_last[:, t_idx])[..., None] + self.q_pool[h][None, None, :]
        raise ValueError(model)


def hindcast_rows(d: D.SimData, sim_daily: dict, eval_sites: np.ndarray, block: str, split,
                  horizons=HORIZONS, variables=tuple(D.TARGETS)) -> dict:
    """Return {(var, h): dict(rows=DataFrame(site, basin_group, issue_date), y, q={model: (n,19)})}.

    ``sim_daily[var]`` is (S, T) aligned with ``d.sites``/``d.dates`` (model space for SC/turbidity).
    Fitting uses train-block days of ALL sites (eval sites' own history; cal sites for pooled terms).
    """
    nb = naive_baselines(d, horizons)
    phys = PhysicsOnly(d)
    fit_days = D.block_mask(d.dates, "train")
    cal_rows = d.idx("cal")
    out = {}
    for v in variables:
        j = D.TARGETS.index(v)
        z = D.to_model(v, d.y[:, :, j])
        ec = SimEC(z, np.asarray(sim_daily[v], float), fit_days, cal_rows, horizons)
        for h in horizons:
            t_idx = np.flatnonzero(split.time_mask(d.dates, block, horizon_days=h))
            if not t_idx.size:
                continue
            ok = nb[v].scorable(t_idx, h)[eval_sites]
            fc = {m: nb[v].forecast(m, t_idx, h)[eval_sites] for m in ("persistence", "climatology",
                                                                        "damped_anomaly")}
            fc["physics_only"] = phys.forecast(v, t_idx, h)[eval_sites]
            for m in SIM_MODELS:
                fc[m] = ec.forecast(m, t_idx, h)[eval_sites]
            for m, q in fc.items():
                ok &= np.isfinite(q).all(axis=-1)
            si, ti = np.nonzero(ok)
            if not si.size:
                continue
            y = nb[v].observed(t_idx, h)[eval_sites][si, ti]
            rows = pd.DataFrame({"site": np.asarray(d.sites)[eval_sites][si],
                                 "basin_group": d.basin_group[eval_sites][si],
                                 "issue_date": d.dates[t_idx[ti]]})
            out[(v, h)] = {"rows": rows, "y": y, "q": {m: np.sort(q[si, ti], axis=-1) for m, q in fc.items()},
                           "phi_pool": ec.phi_pool[h]}
    return out


def crpss_ci(crps_m: np.ndarray, crps_ref: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Point and percentile CI of 1 − ΣCRPS_m/ΣCRPS_ref resampling basin groups (paired)."""
    g, inv = np.unique(groups, return_inverse=True)
    sm = np.bincount(inv, weights=crps_m, minlength=len(g))
    sr = np.bincount(inv, weights=crps_ref, minlength=len(g))
    point = 1.0 - sm.sum() / sr.sum()
    rs = np.random.default_rng(seed)
    pick = rs.integers(0, len(g), (n_boot, len(g)))
    bs = 1.0 - sm[pick].sum(axis=1) / sr[pick].sum(axis=1)
    return {"point": float(point), "ci_low": float(np.quantile(bs, 0.025)), "ci_high": float(np.quantile(bs, 0.975)),
            "p_le0": float(np.mean(bs <= 0)), "n_groups": int(len(g)), "n_boot": n_boot}


def score(hc: dict, *, n_boot: int = 2000) -> pd.DataFrame:
    recs = []
    for (v, h), r in hc.items():
        y = r["y"]
        groups = r["rows"]["basin_group"].to_numpy()
        crps = {m: row_crps(y, q) for m, q in r["q"].items()}
        med = {m: q[:, len(QUANTILES) // 2] for m, q in r["q"].items()}
        for m in crps:
            base = {"var": v, "h": h, "model": m, "n_rows": len(y), "n_sites": r["rows"]["site"].nunique(),
                    "n_groups": len(np.unique(groups)), "crps": float(crps[m].mean()),
                    "rmse_median": EM.rmse(y, med[m]), "nse_pooled": EM.nse(y, med[m])}
            for ref in REFS:
                if ref == m:
                    continue
                if m in SIM_MODELS:
                    ci = crpss_ci(crps[m], crps[ref], groups, n_boot=n_boot)
                    base.update({f"crpss_vs_{ref}": ci["point"], f"crpss_vs_{ref}_lo": ci["ci_low"],
                                 f"crpss_vs_{ref}_hi": ci["ci_high"]})
            recs.append(base)
    return pd.DataFrame(recs)
