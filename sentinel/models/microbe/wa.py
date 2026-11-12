# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Weighted-averaging transfer functions — comparators of the parity list (amendment microbe_arch_v2 §6).

``wa_tol``   classical weighted averaging with tolerance down-weighting and inverse deshrinking
             (ter Braak & van Dam 1989; Birks et al. 1990).
``wa_pls``   weighted-averaging partial least squares, 1–5 components chosen by inner domain-grouped CV
             (ter Braak & Juggins 1993).

Both use the units of NicheInvert-S (harmonised V4 windows present in >= 2 training events of the fold) and the
square root of the in-vocabulary relative abundance. One model per target, fitted on the training rows that have
the target. Deterministic (no seed).

Implementation readings (fixed before any development run, repeated in ``microbe_arch_v2_log.md``):

W1  Tolerance² = abundance-weighted variance around the optimum, divided by (1 - 1/N2) when N2 > 1. A unit with
    N2 < 2 or a non-finite tolerance takes the N2-weighted mean tolerance² of the other units. Floor: tolerance
    >= 0.1 x the training SD of the target.
W2  Inverse deshrinking: unweighted OLS of the training values on the initial estimates (apparent, as in the
    classical method).
W3  WA-PLS follows the published algorithm with row weights R_i / sum(R) (R_i = row total of the transformed
    abundances): weighted centring; species scores; site scores; orthogonalisation against earlier components;
    weighted standardisation; weighted regression. Each component is kept as species coefficients, so that a
    new event's score is the abundance-weighted mean of the coefficients of its units.
W4  Component count: ``GroupKFold(3)`` over the training domains, pooled squared error, ties to fewer components.
    Each inner fit is from scratch on the inner training rows (same vocabulary).
W5  A unit absent from the rows of a fit has no coefficient and is left out of that fit's weighted means. An
    event without any fitted unit takes the training mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

NAMES = ("wa_tol", "wa_pls")
LABELS = {"wa_tol": "WA-tol[V4 units, sqrt]", "wa_pls": "WA-PLS[V4 units, sqrt]"}
DEFAULTS = {"min_events": 2, "max_components": 5, "inner_folds": 3, "tol_min_n2": 2.0, "tol_floor_frac": 0.1,
            "min_rows": 10}


def sqrt_abundance(C, vocab: np.ndarray) -> sparse.csr_matrix:
    """Square root of the in-vocabulary relative abundance, [events x len(vocab)]."""
    Y = sparse.csr_matrix(C, dtype=np.float64)[:, vocab].tocsr()
    n = np.asarray(Y.sum(1)).ravel()
    Y = sparse.diags(np.divide(1.0, n, out=np.zeros_like(n), where=n > 0)) @ Y
    Y.data = np.sqrt(Y.data)
    return Y.tocsr()


def _weighted_mean(Y: sparse.csr_matrix, coef: np.ndarray, default: float) -> np.ndarray:
    """Row-wise mean of ``coef`` weighted by ``Y`` (``default`` for an empty row)."""
    num, den = Y @ coef, np.asarray(Y.sum(1)).ravel()
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), default)


# ------------------------------------------------------------------------------------------------ WA (tol)

def wa_fit(Y: sparse.csr_matrix, x: np.ndarray, *, tol: bool = True, params: dict | None = None) -> dict:
    """Weighted averaging with inverse deshrinking; ``tol`` turns on tolerance down-weighting (reading W1)."""
    p = {**DEFAULTS, **(params or {})}
    x = np.asarray(x, float)
    YT = Y.T.tocsr()
    c = np.asarray(YT.sum(1)).ravel()
    cols = np.flatnonzero(c > 0)
    YT, c = YT[cols], c[cols]
    u = (YT @ x) / c
    w = np.ones(len(cols))
    if tol:
        var = np.maximum((YT @ (x * x)) / c - u * u, 0.0)
        Y2 = YT.copy()
        Y2.data = Y2.data ** 2
        n2 = c * c / np.asarray(Y2.sum(1)).ravel()
        corr = n2 > 1.0 + 1e-6
        var[corr] = var[corr] / (1.0 - 1.0 / n2[corr])
        good = (n2 >= p["tol_min_n2"]) & np.isfinite(var)
        pooled = float(np.average(var[good], weights=n2[good])) if good.any() else float(x.var())
        var = np.where(good, var, pooled)
        w = 1.0 / np.maximum(var, (p["tol_floor_frac"] * float(x.std())) ** 2 + 1e-300)
    fit = {"cols": cols, "u": u, "w": w, "mean": float(x.mean()), "b0": 0.0, "b1": 1.0}
    x0 = _wa_initial(fit, Y)
    xm = float(x0.mean())
    vx = float(((x0 - xm) ** 2).sum())
    if vx > 0:
        fit["b1"] = float(((x0 - xm) * (x - fit["mean"])).sum() / vx)
        fit["b0"] = fit["mean"] - fit["b1"] * xm
    return fit


def _wa_initial(fit: dict, Y: sparse.csr_matrix) -> np.ndarray:
    Yw = Y[:, fit["cols"]] @ sparse.diags(fit["w"])
    return _weighted_mean(Yw.tocsr(), fit["u"], np.nan)


def wa_predict(fit: dict, Y: sparse.csr_matrix) -> np.ndarray:
    x0 = _wa_initial(fit, Y)
    return np.where(np.isfinite(x0), fit["b0"] + fit["b1"] * x0, fit["mean"])


# -------------------------------------------------------------------------------------------------- WA-PLS

def wapls_fit(Y: sparse.csr_matrix, x: np.ndarray, n_comp: int = 5) -> dict:
    """WA-PLS (reading W3). ``beta[m - 1]`` holds the species coefficients of the m-component model; the
    prediction for an event is ``xbar`` + the abundance-weighted mean of ``beta[m - 1]`` over its units."""
    x = np.asarray(x, float)
    c_all = np.asarray(Y.sum(0)).ravel()
    cols = np.flatnonzero(c_all > 0)
    Yk = Y[:, cols].tocsr()
    R, Ck = np.asarray(Yk.sum(1)).ravel(), c_all[cols]
    rows = R > 0
    w = np.where(rows, R, 0.0) / R[rows].sum()
    xbar = float((w * x).sum())
    Rs = np.where(rows, R, 1.0)
    r = np.where(rows, x - xbar, 0.0)
    comps, scores, betas = [], [], []
    beta = np.zeros(len(cols))
    for _ in range(int(n_comp)):
        b = (Yk.T @ r) / Ck                                    # species scores
        t = np.where(rows, (Yk @ b) / Rs, 0.0)                 # site scores
        for bl, tl in zip(comps, scores):                      # orthogonal to the earlier components
            g = float((w * t * tl).sum())
            t, b = t - g * tl, b - g * bl
        mean = float((w * t).sum())
        sd = float(np.sqrt((w * (t - mean) ** 2).sum()))
        if not np.isfinite(sd) or sd < 1e-10:
            break
        t, b = np.where(rows, (t - mean) / sd, 0.0), (b - mean) / sd
        a = float((w * (x - xbar) * t).sum())
        comps.append(b), scores.append(t)
        beta = beta + a * b
        betas.append(beta.copy())
        r = np.where(rows, r - a * t, 0.0)
    return {"cols": cols, "xbar": xbar, "beta": betas}


def wapls_predict(fit: dict, Y: sparse.csr_matrix, m: int) -> np.ndarray:
    if not fit["beta"]:
        return np.full(Y.shape[0], fit["xbar"])
    beta = fit["beta"][min(int(m), len(fit["beta"])) - 1]
    return fit["xbar"] + _weighted_mean(Y[:, fit["cols"]].tocsr(), beta, 0.0)


def wapls_select(Y: sparse.csr_matrix, x: np.ndarray, groups: np.ndarray, params: dict | None = None) -> tuple[int, list]:
    """Number of components by inner group-wise CV (reading W4). Returns (m, pooled squared error per m)."""
    from sklearn.model_selection import GroupKFold

    p = {**DEFAULTS, **(params or {})}
    M = int(p["max_components"])
    k = min(int(p["inner_folds"]), len(np.unique(groups)))
    if k < 2:
        return 1, []
    sse = np.zeros(M)
    for a, b in GroupKFold(k).split(np.zeros(len(x)), groups=groups):
        f = wapls_fit(Y[a], x[a], M)
        for m in range(1, M + 1):
            sse[m - 1] += float(((wapls_predict(f, Y[b], m) - x[b]) ** 2).sum())
    return int(np.argmin(sse)) + 1, sse.tolist()


# -------------------------------------------------------------------------------------------- design level

LAST: dict = {}


def fit_predict(name: str, design, train_ids, test_ids, *, targets, seed: int = 0, n_jobs: int = 6,
                params: dict | None = None, device: str = "cpu") -> pd.DataFrame:
    """Same contract as ``baselines.fit_predict``."""
    from threadpoolctl import threadpool_limits

    from sentinel.models.microbe.niche import unit_table

    if name not in NAMES:
        raise ValueError(f"unknown weighted-averaging model {name}")
    p = {**DEFAULTS, **(params or {})}
    S = design.samples
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    out = pd.DataFrame(np.nan, index=pd.Index(test_ids, name="sample_id"), columns=list(targets))
    ut = unit_table(design)
    Ctr, Cte = ut.counts[ut.rows(train_ids)], ut.counts[ut.rows(test_ids)]
    vocab = np.flatnonzero(np.asarray((Ctr > 0).sum(0)).ravel() >= p["min_events"])
    dom = S.loc[train_ids, "domain_id"].astype(str).to_numpy()
    diag = {}
    with threadpool_limits(limits=int(n_jobs)):
        Ytr, Yte = sqrt_abundance(Ctr, vocab), sqrt_abundance(Cte, vocab)
        for t in targets:
            x = S.loc[train_ids, t].to_numpy(float)
            ok = np.isfinite(x)
            if ok.sum() < p["min_rows"]:
                continue
            if name == "wa_tol":
                out[t] = wa_predict(wa_fit(Ytr[ok], x[ok], tol=True, params=p), Yte)
            else:
                m, sse = wapls_select(Ytr[ok], x[ok], dom[ok], p)
                out[t] = wapls_predict(wapls_fit(Ytr[ok], x[ok], p["max_components"]), Yte, m)
                diag[t] = {"n_components": m}
    LAST.clear()
    LAST.update(name=name, n_units=int(len(vocab)), targets=diag)
    return out
