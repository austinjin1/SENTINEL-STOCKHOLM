# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Step-1 simulation-recovery generator (protocol §7 step 1). **SYNTHETIC DATA — declared (R9).**

Exogenous drivers are REAL gridMET series of real sites (realistic autocorrelation and covariance);
every river variable is generated from known structural equations, with the dissolved-oxygen part
produced by the SENTINEL-Sim physics module (exact hourly integration of the one-station metabolism model):

    T_w[t]   = 0.75·T_w[t−1] + 0.25·(2 + 0.85·T_air[t]) + 0.04·(SW[t] − 200)/10 + ε_T        (°C)
    S[t]     = 0.85·S[t−1] + P[t];  Q[t] = Q_base·(0.2 + 0.8·S[t]/mean S)·exp(ε_Q)               (m³/s)
    log turb = 1.0 + 0.8·(log Q − log Q_base) + AR(1) ε                                                  (log FNU)
    log SC   = 6.0 − 0.3·(log Q − log Q_base) + AR(1) ε                                                  (log µS/cm)
    GPP_d    = 1 + 7·SW/300 · exp(−0.3·max(log Q − log Q_base, 0)) · (1 + ε)                             (g O2 m⁻² d⁻¹)
    ER_d     = −(2 + 0.3·GPP_d)·1.07^(T_w − 20)·(1 + ε)
    K600_d   = 3 + 4·(Q/Q_base)^0.5;  depth = 0.409·Q^0.294
    DO       = physics.integrate_do(GPP, ER, K600, diel T_w, light, depth) + sensor noise
    D_max, D_min = daily max / min DO − DO_sat(T_w mean)

Ground truth = DIRECT parent links whose 1-SD intervention on the parent (other variables frozen)
changes the child by ≥ 0.1 SD (:func:`true_links`). D_max and D_min share unobserved GPP/ER noise, so the
(dmax, dmin) pair is latent-confounded and excluded from scoring (reported separately). Missingness: 20% of
days at random plus three 30-day blocks per river variable. Analyses use the same anomaly pipeline as
real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ...models.sim import physics

VARS = ["tair", "sw", "precip", "logq", "tw", "dmax", "dmin", "logsc", "logturb"]
EXOG = ("tair", "sw", "precip")
EFFECT_SD = 0.1


def _ar1(n, phi, sd, rs):
    e = rs.normal(0, sd, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def generate(tair, sw, precip, dates: pd.DatetimeIndex, lat: float, *, seed: int = 0, intervene: dict | None = None,
             freeze: dict | None = None) -> pd.DataFrame:
    """One synthetic site. ``intervene`` {var: shift} adds a shift to a variable right after it is
    generated (descendants respond); ``freeze`` {var: series} overrides a variable with given values
    (used to hold the other parents fixed for DIRECT effects). Random draws depend only on ``seed``."""
    iv = intervene or {}
    fz = freeze or {}
    rs = np.random.default_rng(seed)
    n = len(dates)
    tair = np.asarray(tair, float) + iv.get("tair", 0.0)
    sw = np.asarray(sw, float) + iv.get("sw", 0.0)
    p_raw = np.maximum(np.asarray(precip, float), 0)
    precip_log = np.log1p(p_raw) + iv.get("precip", 0.0)
    p = np.expm1(np.maximum(precip_log, 0))
    eT, eQ = rs.normal(0, 0.3, n), rs.normal(0, 0.08, n)
    e_tu, e_sc = _ar1(n, 0.6, 0.15, rs), _ar1(n, 0.8, 0.03, rs)
    e_g, e_r = rs.normal(0, 0.1, n), rs.normal(0, 0.1, n)
    sens = rs.normal(0, 0.05, (n, 24))
    tw = np.empty(n)
    tw[0] = max(2 + 0.85 * tair[0], 0)
    swm = 200.0                           # fixed reference (W/m²)
    for t in range(1, n):
        tw[t] = 0.75 * tw[t - 1] + 0.25 * (2 + 0.85 * tair[t]) + 0.04 * (sw[t] - swm) / 10 + eT[t]
    tw = np.maximum(tw, 0.0) + iv.get("tw", 0.0)
    tw = np.asarray(fz["tw"], float) if "tw" in fz else tw
    S = np.empty(n)
    S[0] = p.mean() / 0.15
    for t in range(1, n):
        S[t] = 0.85 * S[t - 1] + p[t]
    q_base = 5.0
    q = q_base * (0.2 + 0.8 * S / S.mean()) * np.exp(eQ)
    logq = np.log(q) + iv.get("logq", 0.0)
    logq = np.asarray(fz["logq"], float) if "logq" in fz else logq
    q = np.exp(logq)
    lq = logq - np.log(q_base)            # fixed reference (constant shifts must propagate)
    logturb = 1.0 + 0.8 * lq + e_tu + iv.get("logturb", 0.0)
    logsc = 6.0 - 0.3 * lq + e_sc + iv.get("logsc", 0.0)
    gpp = np.maximum(1 + 7 * sw / 300 * np.exp(-0.3 * np.maximum(lq, 0)) * (1 + e_g), 0.0)
    er = -(2 + 0.3 * gpp) * 1.07 ** (tw - 20) * (1 + e_r)
    k600 = 3 + 4 * (q / q_base) ** 0.5
    z = physics.depth_m(q)
    tt = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)[None]  # noqa: E731
    th = physics.diel_temperature(tt(tw), tt(np.full(n, 1.5))).clamp(min=0)
    lw = torch.as_tensor(physics.hourly_light_weights(np.array([lat]), dates.dayofyear.to_numpy()), dtype=torch.float64)
    do_h, _ = physics.integrate_do(tt(gpp), tt(er), tt(k600), th, lw, tt(z), 1.0)
    do_h = do_h[0].numpy().clip(min=0) + sens
    sat = physics.do_sat_mgl(tw, 1.0)
    dmax = do_h.max(axis=1) - sat + iv.get("dmax", 0.0)
    dmin = do_h.min(axis=1) - sat + iv.get("dmin", 0.0)
    return pd.DataFrame({"tair": tair, "sw": sw, "precip": precip_log, "logq": logq, "tw": tw, "dmax": dmax,
                         "dmin": dmin, "logsc": logsc, "logturb": logturb}, index=dates)


def add_gaps(df: pd.DataFrame, *, seed: int = 0, frac: float = 0.2, blocks: int = 3, block_len: int = 30):
    rs = np.random.default_rng(seed + 7)
    out = df.copy()
    n = len(df)
    for v in VARS:
        if v in EXOG:
            continue
        m = rs.random(n) < frac
        for _ in range(blocks):
            s = rs.integers(0, n - block_len)
            m[s: s + block_len] = True
        out.loc[m, v] = np.nan
    return out


def true_links(tair, sw, precip, dates, lat, *, seed: int = 0) -> dict:
    """{(parent, child): DIRECT standardized effect} for every ordered pair with a river child: the parent is
    shifted by 1 SD (whole series) while every other variable except the child is frozen at its baseline
    values; effect = mean |Δchild| / SD(child). Pairs among exogenous weather variables are not scored."""
    base = generate(tair, sw, precip, dates, lat, seed=seed)
    sd = base.std()
    eff = {}
    for c in VARS:
        if c in EXOG:
            continue
        for p in VARS:
            if p == c:
                continue
            fz = {v: base[v].to_numpy() for v in ("tw", "logq") if v not in (p, c)}
            sh = generate(tair, sw, precip, dates, lat, seed=seed, intervene={p: float(sd[p])}, freeze=fz)
            eff[(p, c)] = float(np.nanmean(np.abs(sh[c] - base[c])) / sd[c]) if sd[c] > 0 else 0.0
    return eff


def truth_set(eff: dict, thr: float = EFFECT_SD) -> set:
    return {k for k, v in eff.items() if v >= thr}
