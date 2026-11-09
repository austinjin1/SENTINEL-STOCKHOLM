# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Stage-0 cross-validated comparators on the development folds (validation only; no test rows).

``run_fold_baselines`` fits every comparator configuration for one development fold and returns
long-format predictions for the fold's evaluation rows. Test-role rows are removed by the caller
right after loading (``drop_protected``) and never reach this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines as bl

PERSISTENCE_D = {"y__do_mgl__h7": "sonde__do_mgl__lastvalid", "y__turb_fnu__h7": "sonde__turb_fnu__lastvalid",
                 "y__no3n_mgl__h7": "nitrate__no3n_mgl__lastvalid", "y__chl_ugl__h7": "sonde__chl_ugl__lastvalid",
                 "y__spcond_uscm__h7": "sonde__spcond_uscm__lastvalid", "y__ph__h7": "sonde__ph__lastvalid",
                 "y__fdom_qsu__h7": "sonde__fdom_qsu__lastvalid", "y__temp_c__h7": "temp__temp_c__lastvalid",
                 "y__do_mgl_min__h7": "sonde__do_mgl__lastmin"}
for _h in (1, 3, 14):
    for _k, _v in list(PERSISTENCE_D.items()):
        if _k.endswith("__h7"):
            PERSISTENCE_D[_k.replace("__h7", f"__h{_h}")] = _v
# lab targets: the previous chemistry event (strictly before d, <= 35 d) of the same analyte, same model space
for _a in ("tn_mgl", "tp_mgl", "doc_mgl", "no3_no2_n_mgl", "nh4_n_mgl", "ortho_p_mgl", "tss_mgl", "cl_mgl",
           "so4_mgl", "si_mgl", "ca_mgl"):
    PERSISTENCE_D[f"ylab__{_a}"] = f"chem__{_a}"


def anchor_owner(col: str | None) -> str | None:
    """Modality that owns an anchor column (``sonde__do_mgl__lastvalid`` → ``sonde``)."""
    return None if col is None else col.split("__")[0]


def drop_protected(df: pd.DataFrame, site_role: dict, site_col: str = "site", keep=("dev",)) -> pd.DataFrame:
    """Remove every row whose site is not in a development role (test / buffer never pass)."""
    r = df[site_col].map(site_role)
    out = df[r.isin(keep).to_numpy()].reset_index(drop=True)
    assert not out[site_col].map(site_role).isin(["test", "test_spatial", "test_region", "buffer"]).any()
    return out


def feature_sets(X: pd.DataFrame, mods: list[str], hub: list[str]) -> dict[str, list[str]]:
    hubc = [c for c in X.columns if c.split("__")[0] in hub]
    sets = {"hub": hubc}
    for m in mods:
        mc = [c for c in X.columns if c.split("__")[0] == m]
        if mc:
            sets[f"single:{m}"] = mc + hubc
    sets["concat"] = [c for c in X.columns if c.split("__")[0] in set(mods) | set(hub)]
    return sets


def run_fold_baselines(X: pd.DataFrame, pres: pd.DataFrame, y: np.ndarray, kind: str, groups: np.ndarray,
                       tr: np.ndarray, ev: np.ndarray, inner_fold: np.ndarray, mods: list[str], hub: list[str], *,
                       persistence_col: str | None = None, seed: int = 0, threads: int = 6, do_stack: bool = True,
                       do_pattern: bool = False, pattern_min_rows: int = 300, anchored: bool = False,
                       log=print) -> pd.DataFrame:
    """Return long predictions for rows ``ev``: columns row, config, mu, sigma (gauss) or p (binary).

    ``inner_fold`` gives each training row its development fold id (for the stacking inner OOF).
    With ``anchored`` (Gaussian targets with a persistence column), every configuration whose
    features include the anchor's modality is also fit in the persistence-anchored parametrisation
    (``<config>@a``, :func:`baselines.gbm_gauss_anchored`); the level fits are kept.
    """
    sets = feature_sets(X, mods, hub)
    out = []
    ytr = y[tr]
    gtr = groups[tr]
    Xtr, Xev = X.iloc[tr], X.iloc[ev]
    preds = {}
    for name, cols in sets.items():
        if kind == "gauss":
            mu, sg, info = bl.gbm_gauss(Xtr[cols], ytr, gtr, Xev[cols], seed=seed, threads=threads)
            preds[name] = (mu, sg)
        else:
            p = bl.gbm_binary(Xtr[cols], ytr, gtr, Xev[cols], seed=seed, threads=threads)
            preds[name] = (p, None)
        log(f"    {name}: done")
    owner = anchor_owner(persistence_col) if (anchored and kind == "gauss" and persistence_col in X.columns) else None
    if owner is not None:
        a_all = X[persistence_col].to_numpy(float)
        for name in [n for n in sets if n == "concat" or n == f"single:{owner}"]:
            cols = sets[name]
            mu, sg, _ = bl.gbm_gauss_anchored(Xtr[cols], ytr, gtr, Xev[cols], a_all[tr], a_all[ev], seed=seed,
                                              threads=threads)
            preds[f"{name}@a"] = (mu, sg)
            log(f"    {name}@a: done")
    # UME: mixture of single-modality predictions for modalities present in the row (hub if none)
    singles = [m for m in mods if f"single:{m}" in preds]
    W = pres.iloc[ev][singles].to_numpy(bool) if singles else np.zeros((len(ev), 0), bool)
    if singles:
        A = np.column_stack([preds[f"single:{m}"][0] for m in singles])
        if kind == "gauss":
            S = np.column_stack([preds[f"single:{m}"][1] for m in singles])
            none = ~W.any(1)
            mu, sg = bl.mixture_moments(A, S, W)
            mu[none], sg[none] = preds["hub"][0][none], preds["hub"][1][none]
            preds["ume"] = (mu, sg)
        else:
            w = W.astype(float)
            p = (A * w).sum(1) / np.maximum(w.sum(1), 1)
            p[~W.any(1)] = preds["hub"][0][~W.any(1)]
            preds["ume"] = (p, None)
    # Stacking (Super Learner style): inner OOF single-modality means on training rows
    if do_stack and singles:
        inner_ids = np.unique(inner_fold)
        Ztr = np.full((len(tr), len(singles) + 1), np.nan)
        cfgs = ["hub"] + [f"single:{m}" for m in singles]
        for j in inner_ids:
            va = np.flatnonzero(inner_fold == j)
            it = np.flatnonzero(inner_fold != j)
            for ci, c in enumerate(cfgs):
                cols = sets[c]
                if kind == "gauss":
                    Ztr[va, ci] = bl.gbm_mean(Xtr[cols].iloc[it], ytr[it], gtr[it], Xtr[cols].iloc[va], seed=seed,
                                              threads=threads)
                else:
                    Ztr[va, ci] = bl.gbm_binary(Xtr[cols].iloc[it], ytr[it], gtr[it], Xtr[cols].iloc[va], seed=seed,
                                                threads=threads)
        Zev = np.column_stack([preds[c][0] for c in cfgs])
        P = pres.iloc[tr][singles].to_numpy(float)
        Pev = pres.iloc[ev][singles].to_numpy(float)
        # absent modality → its single model equals the hub model in information: mask to NaN
        Ztr[:, 1:][P == 0] = np.nan
        Zev = Zev.astype(float)
        Zev[:, 1:][Pev == 0] = np.nan
        ztr = pd.DataFrame(np.column_stack([Ztr, P]))
        zev = pd.DataFrame(np.column_stack([Zev, Pev]))
        if kind == "gauss":
            mu, sg, _ = bl.gbm_gauss(ztr, ytr, gtr, zev, seed=seed, threads=threads)
            preds["stack"] = (mu, sg)
        else:
            preds["stack"] = (bl.gbm_binary(ztr, ytr, gtr, zev, seed=seed, threads=threads), None)
        log("    stack: done")
    # Persistence
    if persistence_col is not None and kind == "gauss" and persistence_col in X.columns:
        last_tr = X[persistence_col].to_numpy(float)[tr]
        ok = np.isfinite(last_tr) & np.isfinite(ytr)
        rmse = float(np.sqrt(np.mean((ytr[ok] - last_tr[ok]) ** 2))) if ok.any() else np.nan
        last_ev = X[persistence_col].to_numpy(float)[ev]
        preds["persistence"] = (last_ev, np.full(len(ev), rmse))
    # Per-pattern models (P4)
    if do_pattern:
        pk_tr = bl.pattern_key(pres.iloc[tr][mods])
        pk_ev = bl.pattern_key(pres.iloc[ev][mods])
        mu_p = np.full(len(ev), np.nan)
        sg_p = np.full(len(ev), np.nan)
        hubc = sets["hub"]
        for pat in pk_ev.unique():
            ms = [] if pat == "hub" else pat.split("+")
            sup = pres.iloc[tr][ms].all(axis=1).to_numpy() if ms else np.ones(len(tr), bool)
            if sup.sum() < pattern_min_rows:
                continue
            cols = [c for c in X.columns if c.split("__")[0] in set(ms)] + hubc
            evm = (pk_ev == pat).to_numpy()
            idx = np.flatnonzero(sup)
            if kind == "gauss":
                mu, sg, _ = bl.gbm_gauss(Xtr[cols].iloc[idx], ytr[idx], gtr[idx], Xev[cols].iloc[evm], seed=seed,
                                         threads=threads)
                mu_p[evm], sg_p[evm] = mu, sg
            else:
                mu_p[evm] = bl.gbm_binary(Xtr[cols].iloc[idx], ytr[idx], gtr[idx], Xev[cols].iloc[evm], seed=seed,
                                          threads=threads)
        preds["pattern"] = (mu_p, sg_p if kind == "gauss" else None)
        log("    pattern: done")
    for name, (a, b) in preds.items():
        d = pd.DataFrame({"row": ev, "config": name})
        if kind == "gauss":
            d["mu"], d["sigma"] = a, b
        else:
            d["p"] = a
        out.append(d)
    return pd.concat(out, ignore_index=True)
