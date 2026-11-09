# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""E3 — groundwater buffering classes (Hare et al. 2021 metrics) and DO / thermal resilience (protocol §9 E3).

Per water year with ≥ 300 days of both daily-mean air (gridMET) and water temperature, fit
y = α·sin(ωt) + β·cos(ωt) + C (ω = 2π/365.25 d⁻¹); amplitude A = √(α² + β²); phase lag Δφ (days) =
water peak time − air peak time. Median over years. A_r = A_water / A_air. Classes (Hare 2021, verified in
docs/novelty_checks/sim_E3_2026-10-25.md): Δφ > 40 d → dam-influenced (excluded); Δφ ≥ 10 d → shallow
groundwater; else A_r < 0.65 → deep groundwater; 0.65 ≤ A_r ≤ 1.1 → atmospheric; A_r > 1.1 removed.

Hare data rules applied (review W4, 2026-11-08): water-temperature days < 1 °C excluded from the fit; a
water year is used only if its valid-observation record (before the ice exclusion) has no gap ≥ 30 days;
≥ 2 water years required; site Δφ < −10 d → dropped ("negative_lag"), −10 ≤ Δφ < 0 → set to 0. Not
replicated: Hare's "within 25 km downstream of a major dam" rule (no dam point data locally; the declared
sensitivity uses StreamCat NID dam storage instead) and the manual check of A_r < 0.4 sites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...models.sim import data as D
from ...models.sim.checks import _weekly_slope
from ...models.sim.dsst import water_year

OMEGA = 2 * np.pi / 365.25


def _sine(t, y):
    X = np.column_stack([np.sin(OMEGA * t), np.cos(OMEGA * t), np.ones_like(t)])
    (a, b, c), *_ = np.linalg.lstsq(X, y, rcond=None)
    amp = np.hypot(a, b)
    # y = amp·sin(ωt + φ) with φ = atan2(b, a); peak at ωt + φ = π/2
    phi = np.arctan2(b, a)
    t_peak = ((np.pi / 2 - phi) / OMEGA) % 365.25
    return amp, t_peak


def site_metrics(d: D.SimData, i: int, wy_range=(2008, 2023)) -> dict | None:
    wy = water_year(d.dates)
    tw = d.y[i, :, D.TARGETS.index("temp_c_mean")].astype(float)
    ta = 0.5 * (d.forcing[i, :, 0] + d.forcing[i, :, 1]).astype(float)
    t = (d.dates - d.dates[0]).days.to_numpy().astype(float)
    ars, lags = [], []
    for y in range(wy_range[0], wy_range[1] + 1):
        rec = (wy == y) & np.isfinite(tw)
        if rec.sum() < 300:
            continue
        tt = t[rec]
        gaps = np.diff(np.concatenate([[t[wy == y].min() - 1], tt, [t[wy == y].max() + 1]]))
        if gaps.max() - 1 >= 30:
            continue
        m = rec & np.isfinite(ta) & (tw >= 1.0)
        if m.sum() < 200:
            continue
        aw, pw = _sine(t[m], tw[m])
        aa, pa = _sine(t[m], ta[m])
        lag = (pw - pa + 182.625) % 365.25 - 182.625
        ars.append(aw / aa)
        lags.append(lag)
    if len(ars) < 2:
        return None
    ar, lag = float(np.median(ars)), float(np.median(lags))
    if lag < -10:
        cls = "negative_lag"
    elif lag < 0:
        lag = 0.0
    if lag < -10:
        pass
    elif lag > 40:
        cls = "dam"
    elif lag >= 10:
        cls = "shallow_gw"
    elif ar < 0.65:
        cls = "deep_gw"
    elif ar <= 1.1:
        cls = "atmospheric"
    else:
        cls = "removed"
    tr = (wy >= wy_range[0]) & (wy <= wy_range[1])
    return {"site": d.sites[i], "amp_ratio": ar, "phase_lag_d": lag, "n_years": len(ars), "hare_class": cls,
            "thermal_sensitivity": _weekly_slope(tw[tr], ta[tr], d.dates[tr]), "basin_group": d.basin_group[i]}


def classify_all(d: D.SimData) -> pd.DataFrame:
    rows = [site_metrics(d, i) for i in range(len(d.sites)) if d.role[i] in ("train", "cal", "val")]
    return pd.DataFrame([r for r in rows if r])


def class_difference(tab: pd.DataFrame, col: str, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """median(col | class a) − median(col | class b), basin-group cluster bootstrap (two-sided)."""
    x = tab[tab["hare_class"].isin([a, b]) & tab[col].notna()]
    g, inv = np.unique(x["basin_group"], return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(g))]
    vals, cls = x[col].to_numpy(), x["hare_class"].to_numpy()

    def stat(idx):
        va, vb = vals[idx][cls[idx] == a], vals[idx][cls[idx] == b]
        return np.median(va) - np.median(vb) if va.size and vb.size else np.nan

    rs = np.random.default_rng(seed)
    bs = np.array([stat(np.concatenate([members[k] for k in rs.integers(0, len(g), len(g))])) for _ in range(n_boot)])
    return {"diff": float(stat(np.arange(len(x)))), "ci_low": float(np.nanquantile(bs, 0.025)),
            "ci_high": float(np.nanquantile(bs, 0.975)), "n_a": int((cls == a).sum()), "n_b": int((cls == b).sum()),
            "n_groups": int(len(g))}
