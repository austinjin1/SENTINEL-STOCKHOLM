# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""UK EA <-> NRSA transfer in a shared family-level representation (protocol §7.4).

Used by the development diagnostic (``scripts/eval/bioassess_uk_transfer_dev.py``: UK pool ->
NRSA 2008-14, NRSA 2008-14 -> UK pool validation blocks) and, after freeze, by the single test pass
(``scripts/eval/bioassess_final.py``: UK pool -> NRSA 2018-19, NRSA 2008-14 -> reserved UK blocks).

Representation: harmonised **family** log10-abundance class (1-5, 0 = absent); NRSA 300-counts are
converted with the same class rule as EA data; coarser identifications keep their own name.
Families: those present in >= ``min_n`` samples of the *training* side. Shared targets have identical
units on both sides (UK conductivity determinand at 25 °C; UK turbidity NTU determinand 6396 only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHARED = {  # NRSA target -> UK target
    "ph_lab": "uk_ph",
    "log10_cond_us_cm": "uk_log10_cond25_us_cm",
    "log10_chloride_mg_l": "uk_log10_chloride_mg_l",
    "log10_tp_ug_l": "uk_log10_tp_ug_l",
    "log10_turbidity_ntu": "uk_log10_turb_ntu",
}
RF = {"n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.3}


def _family(t: pd.DataFrame) -> pd.Series:
    return t["family"].where(t["family"].notna(), t["name"])


def families_of(tokens: pd.DataFrame, id_col: str, ids, min_n: int = 50) -> list[str]:
    t = tokens[tokens[id_col].isin(set(ids))]
    n = pd.DataFrame({"id": t[id_col].to_numpy(), "fam": _family(t).to_numpy()}).drop_duplicates()
    c = n["fam"].value_counts()
    return sorted(c.index[c >= min_n])


def family_classes(tokens: pd.DataFrame, id_col: str, ids, families) -> np.ndarray:
    """[len(ids), len(families)] log10-abundance class matrix (0 = absent)."""
    t = tokens[tokens[id_col].isin(set(ids))]
    df = pd.DataFrame({"id": t[id_col].to_numpy(), "fam": _family(t).to_numpy(),
                       "count": t["count"].to_numpy(float)})
    if "log_class" in t:                       # UK: published classes (max over merged records)
        df["cls"] = t["log_class"].to_numpy(float)
        g = df.groupby(["id", "fam"])["cls"].max()
    else:                                      # NRSA: class of the summed family count
        g = df.groupby(["id", "fam"])["count"].sum()
        g = np.clip(np.floor(np.log10(np.maximum(g, 1.0))) + 1.0, 1, 5)
    m = g.unstack(fill_value=0.0)
    return m.reindex(index=pd.Index(ids), columns=list(families)).fillna(0.0).to_numpy(np.float32)


def fit_predict(X_tr, y_tr, X_te, *, seed: int = 0, n_jobs: int = 6) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    ok = np.isfinite(y_tr)
    return RandomForestRegressor(**RF, n_jobs=n_jobs, random_state=seed).fit(X_tr[ok], y_tr[ok]).predict(X_te)


def uk_to_us(uk, us, uk_train_ids, us_eval_ids, *, max_rows: int = 40000, seed: int = 0, n_jobs: int = 6) -> pd.DataFrame:
    """RF on UK rows -> predictions for NRSA rows (columns = NRSA target names)."""
    rs = np.random.default_rng(seed)
    fams = families_of(uk.tokens, "analysis_id", uk_train_ids)
    X_us = family_classes(us.tokens, "visit_id", us_eval_ids, fams)
    out = pd.DataFrame(index=pd.Index(us_eval_ids))
    for t_us, t_uk in SHARED.items():
        y = uk.Y.loc[pd.Index(uk_train_ids), t_uk]
        ids = y.index[y.notna()]
        if len(ids) > max_rows:
            ids = pd.Index(rs.choice(ids.to_numpy(), max_rows, replace=False))
        X_uk = family_classes(uk.tokens, "analysis_id", ids, fams)
        out[t_us] = fit_predict(X_uk, y.loc[ids].to_numpy(float), X_us, seed=seed, n_jobs=n_jobs)
    return out


def us_to_uk(us, uk, us_train_ids, uk_eval_ids, *, seed: int = 0, n_jobs: int = 6) -> pd.DataFrame:
    """RF on NRSA rows -> predictions for UK rows (columns = NRSA target names)."""
    fams = families_of(us.tokens, "visit_id", us_train_ids)
    X_us = family_classes(us.tokens, "visit_id", us_train_ids, fams)
    X_uk = family_classes(uk.tokens, "analysis_id", uk_eval_ids, fams)
    out = pd.DataFrame(index=pd.Index(uk_eval_ids))
    for t_us in SHARED:
        out[t_us] = fit_predict(X_us, us.Y.loc[pd.Index(us_train_ids), t_us].to_numpy(float), X_uk, seed=seed,
                                n_jobs=n_jobs)
    return out
