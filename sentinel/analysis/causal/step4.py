# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Causal ladder step 4 (protocol §7): out-of-sample predictive gain and natural-experiment agreement.

(a) Predictive gain of a link X → Y at a val-role site: ridge model of the Y anomaly at day t on Y lags
    1–7 + lags 0–7 of the other replicated parents of Y, with vs without X lags 0–7 (lag 0 = same-day
    observed values, for weather AND river parents — a conditional hindcast, not a forecast).
    Limitation (stated in protocol §7): step 3 uses these val-role sites (WY2008–2023) for replication,
    and step 4 tests on the same sites' val block. Fitted on the site's train block,
    evaluated on the val block. Gain = 1 − MSE_with / MSE_without; site-bootstrap CI of the mean gain.
(b) Event agreement: storm days (cause anomaly ≥ the site's train-block p95, for logq / precip causes) or
    heatwave days (≥ 3 consecutive days of air-T anomaly ≥ p90, for tair / tw causes); effect = mean
    Y anomaly on event days minus on matched non-event days (same site, |ΔDOY| ≤ 15, other water years,
    cause anomaly below its median); sign must equal the link sign with a site-bootstrap CI excluding 0.
    Shortwave-driven links have no pre-registered natural experiment ("n/a" → cannot reach rung 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ...models.sim import data as D
from ...models.sim.dsst import water_year
from .prep import WEATHER, site_matrix

LAGS = 7


def _design(m: pd.DataFrame, y: str, parents: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cols = [m[y].shift(k).rename(f"{y}_l{k}") for k in range(1, LAGS + 1)]
    for p in parents:
        lags = range(0, LAGS + 1) if p in WEATHER or p != y else range(1, LAGS + 1)
        cols += [m[p].shift(k).rename(f"{p}_l{k}") for k in lags]
    X = pd.concat(cols, axis=1)
    return X.to_numpy(), m[y].to_numpy()


def predictive_gain_site(m: pd.DataFrame, cause: str, effect: str, other_parents: list[str], train_days, val_days):
    if cause not in m or effect not in m:
        return np.nan
    others = [p for p in other_parents if p in m and p not in (cause, effect)]
    out = []
    for parents in (others, others + [cause]):
        X, y = _design(m, effect, parents)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        tr, va = ok & train_days, ok & val_days
        if tr.sum() < 200 or va.sum() < 60:
            return np.nan
        r = Ridge(alpha=1.0).fit(X[tr], y[tr])
        out.append(np.mean((y[va] - r.predict(X[va])) ** 2))
    return float(1 - out[1] / out[0])


def event_effect_site(m: pd.DataFrame, cause: str, effect: str, train_days) -> float:
    if cause not in m or effect not in m or cause == "sw":
        return np.nan
    c = m[cause].to_numpy()
    yv = m[effect].to_numpy()
    if cause in ("tair", "tw"):
        a = m["tair"].to_numpy()
        thr = np.nanquantile(a[train_days], 0.9)
        hot = np.nan_to_num(a >= thr).astype(bool)
        ev = np.zeros(len(a), bool)
        run = 0
        for t in range(len(a)):
            run = run + 1 if hot[t] else 0
            if run >= 3:
                ev[t - run + 1: t + 1] = True
        base_ok = np.nan_to_num(a < np.nanmedian(a[train_days]), nan=0).astype(bool)
    else:
        thr = np.nanquantile(c[train_days], 0.95)
        ev = np.nan_to_num(c >= thr).astype(bool)
        base_ok = np.nan_to_num(c < np.nanmedian(c[train_days]), nan=0).astype(bool)
    doy = m.index.dayofyear.to_numpy()
    wy = water_year(m.index)
    diffs = []
    for t in np.flatnonzero(ev & np.isfinite(yv)):
        match = base_ok & np.isfinite(yv) & (wy != wy[t]) & (np.minimum(np.abs(doy - doy[t]), 366 - np.abs(doy - doy[t])) <= 15)
        if match.sum() >= 5:
            diffs.append(yv[t] - yv[match].mean())
    return float(np.mean(diffs)) if len(diffs) >= 5 else np.nan


def evaluate_links(d: D.SimData, links: pd.DataFrame, parents_of: dict, val_sites, *, n_boot: int = 2000,
                   seed: int = 0) -> pd.DataFrame:
    """``links``: rows with cause, effect, modal_sign (step-3 passers). ``parents_of[effect]``: replicated
    parents. Evaluated at val-role sites (train block fit, val block test / events over WY2008–2023)."""
    wy = water_year(d.dates)
    days = (wy >= 2008) & (wy <= 2023)
    rows = []
    mats = {}
    for i in val_sites:
        mats[i] = site_matrix(d, i, days)
    for _, L in links.iterrows():
        gains, effs = [], []
        for i, m in mats.items():
            if m is None:
                continue
            idx = m.index
            trd = np.asarray(D.block_mask(idx, "train"))
            vad = np.asarray(D.block_mask(idx, "val"))
            gains.append(predictive_gain_site(m, L["cause"], L["effect"], parents_of.get(L["effect"], []), trd, vad))
            effs.append(event_effect_site(m, L["cause"], L["effect"], trd))
        rs = np.random.default_rng(seed)
        rec = {"cause": L["cause"], "effect": L["effect"], "modal_sign": L["modal_sign"]}
        for name, vals in (("gain", gains), ("event", effs)):
            v = np.array([x for x in vals if np.isfinite(x)])
            if v.size >= 5:
                bs = [rs.choice(v, v.size).mean() for _ in range(n_boot)]
                rec.update({f"{name}_mean": float(v.mean()), f"{name}_lo": float(np.quantile(bs, 0.025)),
                            f"{name}_hi": float(np.quantile(bs, 0.975)), f"{name}_n": int(v.size)})
            else:
                rec.update({f"{name}_mean": np.nan, f"{name}_n": int(v.size)})
        gain_ok = rec.get("gain_lo", -1) > 0
        s = L["modal_sign"]
        ev_ok = (s > 0 and rec.get("event_lo", -1) > 0) or (s < 0 and rec.get("event_hi", 1) < 0)
        rec["step4_pass"] = bool(gain_ok and ev_ok)
        rows.append(rec)
    return pd.DataFrame(rows)
