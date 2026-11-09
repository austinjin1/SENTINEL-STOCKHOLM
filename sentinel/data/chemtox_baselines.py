# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Trivial baselines and the replicate noise floor for ADORE (docs/research/04 §A.2).

Baselines (fit on the official training partition only, never on test rows):
  * ``global_mean``            mean training log10 LC50;
  * ``species_mean``           per-species (``tax_gs``) training mean, falling back to the
                               global mean for species unseen in training;
  * ``species_duration_mean``  per (species, ``result_obs_duration_mean``) training mean, falling
                               back to the species mean, then to the global mean.
These are *baselines*, not SENTINEL results; they bound what a model must beat.

Noise floor: experiments with identical (CAS, species, duration, concentration type, media type,
exposure type) are treated as replicates; the pooled within-key SD
``sqrt(sum_g sum_i (y_gi - ybar_g)^2 / sum_g (n_g - 1))`` over keys with >= 2 results is the
repeatability floor of the label itself. Missing key values form their own group (as in the
research computation, which grouped on raw CSV strings).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from sentinel.eval import metrics as M

SPECIES_COL = "tax_gs"
DURATION_COL = "result_obs_duration_mean"
REPLICATE_KEYS = (
    "test_cas",
    "tax_gs",
    "result_obs_duration_mean",
    "result_conc1_type",
    "test_media_type",
    "test_exposure_type",
)
BASELINES = ("global_mean", "species_mean", "species_duration_mean")


def _key(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    """Hashable group key per row; NaN/None become the literal '<NA>' so they group together."""
    parts = [df[c].astype("string").fillna("<NA>") for c in cols]
    k = parts[0]
    for p in parts[1:]:
        k = k + "\x1f" + p
    return k


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_col: str,
    *,
    species_col: str = SPECIES_COL,
    duration_col: str = DURATION_COL,
) -> dict:
    """Predictions of the three trivial baselines for ``test`` (fit on ``train``) + fallback counts."""
    ytr = pd.to_numeric(train[y_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ytr)
    tr = train.loc[ok]
    ytr = ytr[ok]
    gm = float(np.mean(ytr))
    sp_tr = tr[species_col].astype("string").fillna("<NA>").to_numpy()
    spd_tr = _key(tr, [species_col, duration_col]).to_numpy()
    sp_mean = pd.Series(ytr).groupby(sp_tr).mean()
    spd_mean = pd.Series(ytr).groupby(spd_tr).mean()
    sp_te = test[species_col].astype("string").fillna("<NA>")
    spd_te = _key(test, [species_col, duration_col])
    p_sp = sp_te.map(sp_mean).astype("float64")
    p_spd = spd_te.map(spd_mean).astype("float64")
    n_sp_fallback = int(p_sp.isna().sum())
    n_spd_fallback_to_sp = int((p_spd.isna() & p_sp.notna()).sum())
    n_spd_fallback_to_global = int((p_spd.isna() & p_sp.isna()).sum())
    p_sp = p_sp.fillna(gm)
    p_spd = p_spd.fillna(sp_te.map(sp_mean).astype("float64")).fillna(gm)
    return {
        "preds": {
            "global_mean": np.full(len(test), gm),
            "species_mean": p_sp.to_numpy(),
            "species_duration_mean": p_spd.to_numpy(),
        },
        "global_mean_value": gm,
        "n_train": int(ok.sum()),
        "fallbacks": {
            "species_mean_to_global": n_sp_fallback,
            "species_duration_to_species": n_spd_fallback_to_sp,
            "species_duration_to_global": n_spd_fallback_to_global,
        },
    }


def score(y, p) -> dict:
    return {"rmse": M.rmse(y, p), "mae": M.mae(y, p), "r2": M.r2(y, p)}


def evaluate_split(
    df: pd.DataFrame,
    split: str,
    y_col: str,
    *,
    species_col: str = SPECIES_COL,
    duration_col: str = DURATION_COL,
) -> dict:
    """Test-set scores (fit on all official training rows) and 5-fold CV scores (official folds)."""
    a = df[split].astype("string")
    is_test = (a == "test").fillna(False).to_numpy()
    is_train = (a.notna() & (a != "test")).fillna(False).to_numpy()
    train, test = df.loc[is_train], df.loc[is_test]
    y_te = pd.to_numeric(test[y_col], errors="coerce").to_numpy(dtype=float)
    fit = fit_predict(train, test, y_col, species_col=species_col, duration_col=duration_col)
    out = {
        "split": split,
        "n_train": fit["n_train"],
        "n_test": int(np.isfinite(y_te).sum()),
        "test_sd": float(np.nanstd(y_te)),
        "fallbacks_test": fit["fallbacks"],
        "test": {b: score(y_te, fit["preds"][b]) for b in BASELINES},
        "cv": {},
    }
    is_fold = a.str.fullmatch(r"\d+").fillna(False).to_numpy()
    folds = sorted(int(v) for v in a[is_fold].unique())
    per_fold = {b: [] for b in BASELINES}
    oof = {b: np.full(len(df), np.nan) for b in BASELINES}
    n_val = []
    for k in folds:
        val_m = is_fold & (a == str(k)).fillna(False).to_numpy()
        trn_m = is_fold & ~val_m
        f = fit_predict(
            df.loc[trn_m], df.loc[val_m], y_col, species_col=species_col, duration_col=duration_col
        )
        y_val = pd.to_numeric(df.loc[val_m, y_col], errors="coerce").to_numpy(dtype=float)
        n_val.append(int(np.isfinite(y_val).sum()))
        for b in BASELINES:
            per_fold[b].append(score(y_val, f["preds"][b]))
            oof[b][val_m] = f["preds"][b]
    y_all = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    for b in BASELINES:
        if not per_fold[b]:
            continue
        agg = {}
        for m in ("rmse", "mae", "r2"):
            vals = [d[m] for d in per_fold[b]]
            agg[m] = float(np.mean(vals))
            agg[m + "_fold_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            agg[m + "_per_fold"] = [float(v) for v in vals]
        pooled = score(y_all[is_fold], oof[b][is_fold])
        agg.update({f"pooled_{m}": v for m, v in pooled.items()})
        out["cv"][b] = agg
    out["cv_folds"] = folds
    out["cv_n"] = int(np.isfinite(y_all[is_fold]).sum())
    out["cv_n_val_per_fold"] = n_val
    return out


def replicate_noise_floor(df: pd.DataFrame, y_col: str, keys: Iterable[str] = REPLICATE_KEYS) -> dict:
    """Pooled within-experiment-key SD (ddof=1 pooling) and per-key SD summaries (ddof=0)."""
    y = pd.to_numeric(df[y_col], errors="coerce")
    ok = y.notna().to_numpy()
    k = _key(df.loc[ok], list(keys)).to_numpy()
    s = pd.Series(y[ok].to_numpy(dtype=float))
    g = s.groupby(k)
    n = g.size()
    multi = n[n >= 2].index
    dev2 = (s - g.transform("mean")) ** 2
    in_multi = pd.Series(k).isin(multi).to_numpy()
    ss = float(dev2[in_multi].sum())
    dof = int((n[multi] - 1).sum())
    per_key_sd = g.std(ddof=0)[multi]
    return {
        "pooled_sd": float(np.sqrt(ss / dof)) if dof > 0 else float("nan"),
        "dof": dof,
        "n_keys": int(len(n)),
        "n_keys_with_replicates": int(len(multi)),
        "mean_per_key_sd": float(per_key_sd.mean()),
        "median_per_key_sd": float(per_key_sd.median()),
        "keys": list(keys),
    }


def within_chemical_sd(
    df: pd.DataFrame, y_col: str, *, chem_col: str = "test_cas", min_tests: int = 5
) -> dict:
    """SD of the label across species/conditions within a chemical (chemicals with >= min_tests)."""
    y = pd.to_numeric(df[y_col], errors="coerce")
    sub = pd.DataFrame({"c": df[chem_col].astype("string"), "y": y}).dropna()
    g = sub.groupby("c")["y"]
    n = g.size()
    sd = g.std(ddof=0)[n >= min_tests]
    return {
        "n_chemicals": int(len(sd)),
        "mean_sd": float(sd.mean()),
        "median_sd": float(sd.median()),
        "min_tests": min_tests,
    }
