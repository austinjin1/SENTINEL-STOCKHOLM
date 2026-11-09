# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""E1 — diel-resolved warming sensitivity of DO: solubility vs metabolism (protocol §9 E1).

Per site (train/cal/val-role Sim-eligible sites, WY2008–2023, JJA days), anomalies against the site's
DOY climatology (±15 d, fitted on WY2008–2023) of Y ∈ {DO_mean, DO_min} are regressed on the water-
temperature anomaly with log-discharge (observed only) and shortwave anomalies as controls; OLS with
Newey–West (lag 7) standard errors. The solubility expectation β_sol = s̄ · dDO_sat/dT at the site's mean
JJA water temperature (s̄ = median Y / DO_sat(T_mean)); β_bio = β − β_sol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ...models.sim import data as D
from ...models.sim import physics
from ...models.sim.dsst import fit_climatology, water_year

MIN_SUMMERS, MIN_SUMMER_DAYS = 3, 60
WY_RANGE = (2008, 2023)


def _anoms(x: np.ndarray, dates, fit) -> np.ndarray:
    cm, _ = fit_climatology(x[None, :], dates, fit)
    return x - cm[0, dates.dayofyear.to_numpy() - 1]


def site_frame(d: D.SimData, i: int, *, exclude_provisional: bool = False) -> pd.DataFrame | None:
    wy = water_year(d.dates)
    period = (wy >= WY_RANGE[0]) & (wy <= WY_RANGE[1])
    jT, jM, jN = (D.TARGETS.index(v) for v in ("temp_c_mean", "do_mgl_mean", "do_mgl_min"))
    tw, dom, don = (d.y[i, :, j].astype(float).copy() for j in (jT, jM, jN))
    if exclude_provisional:
        for arr, j in ((tw, jT), (dom, jM), (don, jN)):
            arr[d.prov[i, :, j]] = np.nan
    jja = np.isin(d.dates.month, (6, 7, 8)) & period
    ok = jja & np.isfinite(tw) & np.isfinite(dom) & np.isfinite(don)
    yrs = pd.Series(wy[ok]).value_counts()
    if (yrs >= MIN_SUMMER_DAYS).sum() < MIN_SUMMERS:
        return None
    q = np.where(d.q_missing[i], np.nan, d.q_m3s[i].astype(float))
    lq = np.log(np.maximum(q, 0) + D.Q_FLOOR)
    sw = d.forcing[i, :, D.FORCING_RAW.index("srad_wm2")].astype(float)
    ta = 0.5 * (d.forcing[i, :, 0] + d.forcing[i, :, 1]).astype(float)
    fit = period.copy()
    df = pd.DataFrame({"tw": tw, "do_mean": dom, "do_min": don, "lq": lq, "sw": sw, "ta": ta,
                       "a_tw": _anoms(tw, d.dates, fit), "a_mean": _anoms(dom, d.dates, fit),
                       "a_min": _anoms(don, d.dates, fit), "a_lq": _anoms(lq, d.dates, fit),
                       "a_sw": _anoms(sw, d.dates, fit), "a_ta": _anoms(ta, d.dates, fit), "wy": wy},
                      index=d.dates)
    return df[jja]


def _ols_hac(y, X, t_days=None, maxlag: int = 7):
    """OLS slope of the first regressor with Newey–West (Bartlett, ``maxlag`` days) standard errors in
    CALENDAR time (review W5, 2026-11-08): observation pairs are weighted by their actual separation in
    days, so the gaps between summers (and missing days within a summer) are not treated as adjacent."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 60:
        return None
    y, X = y[ok], X[ok]
    t = np.arange(len(y)) if t_days is None else np.asarray(t_days)[ok]
    Xc = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ y
    u = Xc * (y - Xc @ beta)[:, None]
    S = u.T @ u
    o = np.argsort(t, kind="stable")
    t, u = t[o], u[o]
    for k in range(1, maxlag + 1):
        # pairs exactly k days apart
        j = np.searchsorted(t, t + k)
        hit = (j < len(t)) & (t[np.minimum(j, len(t) - 1)] == t + k)
        if hit.any():
            G = u[hit].T @ u[j[hit]]
            S += (1 - k / (maxlag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(beta[1]), float(np.sqrt(max(V[1, 1], 0.0))), int(len(y))


def site_slopes(df: pd.DataFrame, pressure: float) -> dict | None:
    X = df[["a_tw", "a_lq", "a_sw"]].to_numpy()
    if not np.isfinite(df["a_lq"]).any():
        X = df[["a_tw", "a_sw"]].to_numpy()
    out = {}
    tmean = float(np.nanmean(df["tw"]))
    sat = physics.do_sat_mgl(df["tw"].to_numpy(), pressure)
    dsat = float(physics.do_sat_mgl(tmean + 0.5, pressure) - physics.do_sat_mgl(tmean - 0.5, pressure))
    tdays = ((df.index - pd.Timestamp("2000-01-01")) // pd.Timedelta(days=1)).to_numpy()
    for k, col in (("mean", "do_mean"), ("min", "do_min")):
        r = _ols_hac(df[f"a_{k}"].to_numpy(), X, tdays)
        if r is None:
            return None
        s_bar = float(np.nanmedian(df[col].to_numpy() / sat))
        out[f"beta_{k}"], out[f"se_{k}"], out["n_days"] = r
        out[f"s_bar_{k}"] = s_bar
        out[f"beta_sol_{k}"] = s_bar * dsat
        out[f"beta_bio_{k}"] = r[0] - s_bar * dsat
    ratio = df["do_mean"].to_numpy() / sat
    ratio = ratio[np.isfinite(ratio)]
    out.update({"t_mean_jja": tmean, "dDOsat_dT": dsat,
                "psat_iqr": float(np.quantile(ratio, 0.75) - np.quantile(ratio, 0.25)),
                "n_summers": int(df[df[["tw", "do_mean", "do_min"]].notna().all(axis=1)].groupby("wy").size()
                                 .ge(MIN_SUMMER_DAYS).sum())})
    return out


def heatwave_effect(df: pd.DataFrame, s: dict) -> float:
    """Mean over heatwave days (air-T anomaly ≥ site JJA p90 for ≥ 3 consecutive days) of the DO_min
    anomaly minus its solubility-only expectation β_sol,min · T_w anomaly."""
    a = df["a_ta"].to_numpy()
    thr = np.nanquantile(a, 0.9)
    hot = a >= thr
    run = np.zeros(len(a), int)
    for t in range(len(a)):
        run[t] = run[t - 1] + 1 if hot[t] and t > 0 and (df.index[t] - df.index[t - 1]).days == 1 else int(hot[t])
    # mark every day belonging to a run of length ≥ 3
    hw = np.zeros(len(a), bool)
    t = len(a) - 1
    while t >= 0:
        if run[t] >= 3:
            hw[t - run[t] + 1: t + 1] = True
            t -= run[t]
        else:
            t -= 1
    resid = df["a_min"].to_numpy() - s["beta_sol_min"] * df["a_tw"].to_numpy()
    v = resid[hw & np.isfinite(resid)]
    return float(v.mean()) if v.size >= 5 else np.nan


def cluster_median_ci(x: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0, stat=np.median):
    ok = np.isfinite(x)
    x, groups = x[ok], groups[ok]
    g, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(g))]
    rs = np.random.default_rng(seed)
    bs = np.array([stat(x[np.concatenate([members[k] for k in rs.integers(0, len(g), len(g))])]) for _ in range(n_boot)])
    return {"point": float(stat(x)), "ci_low": float(np.quantile(bs, 0.025)), "ci_high": float(np.quantile(bs, 0.975)),
            "n_sites": int(x.size), "n_groups": int(len(g))}


def run_e1(d: D.SimData, *, exclude_provisional: bool = False) -> pd.DataFrame:
    rows = []
    for i in range(len(d.sites)):
        if d.role[i] not in ("train", "cal", "val"):
            continue
        df = site_frame(d, i, exclude_provisional=exclude_provisional)
        if df is None:
            continue
        s = site_slopes(df, float(d.pressure[i]))
        if s is None:
            continue
        s.update({"site": d.sites[i], "role": d.role[i], "basin_group": d.basin_group[i], "lat": d.lat[i],
                  "lon": d.lon[i], "heatwave_resid_min": heatwave_effect(df, s)})
        # split-half (odd vs even water years)
        for part, sel in (("odd", df["wy"] % 2 == 1), ("even", df["wy"] % 2 == 0)):
            sub = df[sel]
            ss = site_slopes(sub, float(d.pressure[i])) if len(sub) >= 60 else None
            s[f"beta_min_{part}"] = ss["beta_min"] if ss else np.nan
            s[f"beta_mean_{part}"] = ss["beta_mean"] if ss else np.nan
        rows.append(s)
    return pd.DataFrame(rows)


def summarise(tab: pd.DataFrame, tab_noprov: pd.DataFrame | None = None, powell_er: pd.Series | None = None) -> dict:
    g = tab["basin_group"].to_numpy()
    diff = (tab["beta_min"] - tab["beta_mean"]).to_numpy()
    out = {"n_sites": int(len(tab)), "n_groups": int(tab["basin_group"].nunique())}
    out["H_E1a_median_beta_min_minus_mean"] = cluster_median_ci(diff, g)
    out["H_E1b_median_beta_bio_min"] = cluster_median_ci(tab["beta_bio_min"].to_numpy(), g)
    out["median_beta_bio_mean"] = cluster_median_ci(tab["beta_bio_mean"].to_numpy(), g)
    out["median_beta_min"] = cluster_median_ci(tab["beta_min"].to_numpy(), g)
    out["median_beta_mean"] = cluster_median_ci(tab["beta_mean"].to_numpy(), g)
    out["share_min_faster_than_solubility"] = cluster_median_ci(
        (tab["beta_min"] < tab["beta_sol_min"]).astype(float).to_numpy(), g, stat=np.mean)
    out["H_E1a_pass"] = out["H_E1a_median_beta_min_minus_mean"]["ci_high"] < 0
    out["H_E1b_pass"] = out["H_E1b_median_beta_bio_min"]["ci_high"] < 0
    # (i) physics check at stable-%sat sites
    st = tab[tab["psat_iqr"] <= 0.10]
    lo = st["beta_mean"] - 1.96 * st["se_mean"] - st["beta_sol_mean"]
    hi = st["beta_mean"] + 1.96 * st["se_mean"] - st["beta_sol_mean"]
    share = float(((lo <= 0) & (hi >= 0)).mean()) if len(st) else np.nan
    out["V1_stable_psat"] = {"n_sites": int(len(st)), "share_ci_contains_solubility": share,
                             "pass": bool(len(st) >= 5 and share >= 0.70)}
    # (ii) split half
    m = tab[["beta_min_odd", "beta_min_even"]].dropna()
    rho = float(spearmanr(m["beta_min_odd"], m["beta_min_even"]).statistic) if len(m) >= 10 else np.nan
    out["V2_split_half_spearman_beta_min"] = {"rho": rho, "n_sites": int(len(m)), "pass": bool(rho >= 0.5)}
    # (iii) Powell ER agreement
    if powell_er is not None:
        j = tab.set_index("site").join(powell_er.rename("powell_er_mean"), how="inner").dropna(subset=["powell_er_mean"])
        if len(j) >= 10:
            rs = np.random.default_rng(0)
            r0 = float(spearmanr(j["beta_bio_min"], j["powell_er_mean"]).statistic)
            bs = []
            for _ in range(2000):
                k = rs.integers(0, len(j), len(j))
                bs.append(spearmanr(j["beta_bio_min"].to_numpy()[k], j["powell_er_mean"].to_numpy()[k]).statistic)
            out["V3_powell_er_spearman"] = {"rho": r0, "ci_low": float(np.nanquantile(bs, 0.025)),
                                            "ci_high": float(np.nanquantile(bs, 0.975)), "n_sites": int(len(j)),
                                            "pass": bool(np.nanquantile(bs, 0.025) > 0)}
        else:
            out["V3_powell_er_spearman"] = {"n_sites": int(len(j)), "pass": False, "note": "fewer than 10 overlap sites"}
    hw = cluster_median_ci(tab["heatwave_resid_min"].to_numpy(), g)
    out["V4_heatwave_resid_min"] = {**hw, "pass": bool(hw["ci_high"] < 0)}
    if tab_noprov is not None and len(tab_noprov):
        d2 = float(np.nanmedian(tab_noprov["beta_min"] - tab_noprov["beta_mean"]))
        d1 = out["H_E1a_median_beta_min_minus_mean"]["point"]
        rel = abs(d2 - d1) / max(abs(d1), 1e-9)
        out["V5_provisional_exclusion"] = {"median_diff_noprov": d2, "rel_change": rel, "pass": bool(rel < 0.25)}
    out["finding"] = bool((out["H_E1a_pass"] or out["H_E1b_pass"]) and out["V1_stable_psat"]["pass"]
                          and out["V2_split_half_spearman_beta_min"]["pass"])
    return out
