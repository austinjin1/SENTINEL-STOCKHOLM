# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M3 SENTINEL-Microbe event-track adapter (docs/preregistration/events_microbe_protocol.md).

Chronic state classification at the sample date, built on the shared, frozen ``sentinel.eval.events``
(v1.0, never edited). Everything here is computed from development-LODO out-of-fold (OOF) predictions,
so held-out rows never inform their own probability:

* :func:`derived_probability` (route i, primary): Gaussian(mu = OOF prediction, sigma = residual SD of the
  OTHER domains) in the model's target space -> ``events.exceedance_probability`` -> Platt map fitted on the
  OTHER domains' rows (``events.recalibrate``), applied to domain g. ``frozen_maps`` fits sigma and Platt
  once on all development rows (for the test).
* :func:`direct_probability` (route ii, challenger): binary head on M*'s inputs, trained per LODO fold
  on training rows only, then cross-fitted Platt like route i.
* :func:`climatology_probability`: L2-logistic on season + location (C by site-grouped inner CV), per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.eval import events as EV

# event code -> (M3 target column, raw value column, space of the target)
EVENT_TARGETS = {
    "E5b": ("log_tp", "raw_tp_mgl", "log10"),
    "E4a": ("log_chla", "raw_chla_ugl", "log10"),
    "E2c": ("log_sc", "raw_sc_uscm", "log10"),
    "E4a_t12": ("log_chla", "raw_chla_ugl", "log10"),
    "E1a_grab": ("do", "raw_do_mgl", "linear"),
    "E1_do5": ("do", "raw_do_mgl", "linear"),
    "E3a_turb": ("log_turb", "raw_turb_fnu", "log10"),
}
# Local descriptive definition (not in the shared table): DO daily mean < 5 mg/L (plan §2.1 M3 row "DO < 5").
LOCAL_EVENTS = {"E1_do5": EV.EventDef("E1_do5", "low DO (sonde daily mean < 5 mg/L; M3 descriptive)", "chronic",
                                      ("do_mgl",), "mg/L", "below", 5.0, "snapshot",
                                      anchor="EPA (1986) warmwater 5 mg/L (descriptive tier, M3 only)")}
INFERENTIAL = ("E5b", "E4a")
PLATT_MAX_SLOPE = 20.0   # events protocol §4.1 amendment: |a| > 20 or non-finite -> raw probability (all methods)
BORDERLINE = ("E2c",)
DESCRIPTIVE = ("E4a_t12", "E1a_grab", "E1_do5", "E3a_turb")


def event_def(code: str) -> EV.EventDef:
    return LOCAL_EVENTS.get(code) or EV.get_event(code)


def labels(samples: pd.DataFrame, code: str) -> np.ndarray:
    """0/1/NaN state from the raw (untransformed) value, via the shared ``events.event_state``."""
    _, raw, _ = EVENT_TARGETS[code]
    return EV.event_state(samples[raw].astype(float).to_numpy(), event_def(code))


def _platt_crossfit(p_raw: np.ndarray, y: np.ndarray, dom: np.ndarray) -> np.ndarray:
    out = np.full(p_raw.shape, np.nan)
    for g in np.unique(dom):
        tr = (dom != g) & np.isfinite(p_raw) & np.isfinite(y)
        te = dom == g
        try:
            cal = EV.recalibrate(p_raw[tr], y[tr], method="platt")
        except ValueError:          # one class outside g: leave uncalibrated rows NaN (reported)
            continue
        a_ = cal.params.get("a", np.nan)
        if not (np.isfinite(a_) and abs(a_) <= PLATT_MAX_SLOPE and np.isfinite(cal.params.get("b", np.nan))):
            out[te] = p_raw[te]     # degenerate map (near-constant predictor): keep the raw probability
            continue
        with np.errstate(over="ignore"):
            out[te] = cal.transform(p_raw[te])
    return out


def derived_probability(samples: pd.DataFrame, oof: pd.DataFrame, code: str) -> dict:
    """Route (i). Returns raw and recalibrated probabilities aligned to ``oof.index`` + per-domain sigma."""
    target, _, space = EVENT_TARGETS[code]
    S = samples.loc[oof.index]
    mu = oof[target].to_numpy(float)
    yt = S[target].to_numpy(float)
    dom = S["domain_id"].astype(str).to_numpy()
    res = yt - mu
    ok = np.isfinite(res)
    sig = np.full(mu.shape, np.nan)
    sig_by_dom = {}
    for g in np.unique(dom):
        s = float(np.std(res[ok & (dom != g)], ddof=1))
        sig_by_dom[g] = s
        sig[dom == g] = s
    p_raw = EV.exceedance_probability(EV.Gaussian(mu, sig, space=space), event_def(code))
    p_raw = np.where(np.isfinite(mu), p_raw, np.nan)
    y = labels(S, code)
    return {"p_raw": p_raw, "p": _platt_crossfit(p_raw, y, dom), "y": y, "sigma_by_domain": sig_by_dom}


def frozen_maps(samples: pd.DataFrame, oof: pd.DataFrame, code: str) -> dict:
    """Sigma and Platt map fitted once on ALL development OOF rows (the M3 validation set), frozen for the
    test pass (protocol §4.1)."""
    target, _, space = EVENT_TARGETS[code]
    S = samples.loc[oof.index]
    mu = oof[target].to_numpy(float)
    res = S[target].to_numpy(float) - mu
    sigma = float(np.std(res[np.isfinite(res)], ddof=1))
    p_raw = EV.exceedance_probability(EV.Gaussian(mu, np.full(mu.shape, sigma), space=space), event_def(code))
    cal = EV.recalibrate(p_raw, labels(S, code), method="platt")
    return {"code": code, "target": target, "space": space, "sigma": sigma, "platt": cal.params,
            "n_fit": cal.n_fit, "n_pos_fit": cal.n_pos_fit, "fit_sha256": cal.fit_sha256}


def _fold_features(name: str, design, tr, te, params=None):
    """Exactly the inputs of the M3 table model ``name`` (mirrors baselines.fit_predict)."""
    from sentinel.models.microbe import baselines as B
    from sentinel.models.microbe import features as F

    p = {**B.DEFAULTS, **(params or {})}
    S = design.samples
    clr_p = dict(p["clr"])
    toks = design.tokens
    if name in B.V4MODELS:
        toks, clr_p["rank"] = B.v4_tokens(design), "v4"
    clr = F.GenusCLR(**clr_p).fit(toks, tr)
    Xtr, Xte = clr.transform(toks, tr, "clr"), clr.transform(toks, te, "clr")
    if name in B.CONTEXT_TABLES:
        Ctr, fill = F.context_features(S, tr)
        Cte, _ = F.context_features(S, te, fill)
        Xtr, Xte = np.column_stack([Xtr, Ctr]), np.column_stack([Xte, Cte])
    en = p.get("descriptors", True)
    Dtr = F.sample_descriptors(S, tr, primer_onehot=True, enabled=en)
    Dte = F.sample_descriptors(S, te, primer_onehot=True, enabled=en)
    return np.column_stack([Xtr, Dtr]), np.column_stack([Xte, Dte])


def direct_probability(design, folds, code: str, model: str, *, seeds=(0, 1, 2), n_jobs: int = 6,
                       min_train_pos: int = 5) -> dict:
    """Route (ii): binary head on M*'s inputs (XGB with M*'s hyperparameters for XGB models, else L2-logistic
    with C by site-grouped inner CV), per LODO fold on training rows; mean of seeds; cross-fitted Platt."""
    from sentinel.models.microbe import baselines as B

    S = design.samples
    y_all = pd.Series(labels(S, code), index=S.index)
    tested = np.concatenate([f.test for f in folds])
    p = pd.Series(np.nan, index=pd.Index(tested))
    skipped = []
    for f in folds:
        ytr = y_all.loc[f.train].to_numpy()
        ok = np.isfinite(ytr)
        if ytr[ok].sum() < min_train_pos or ytr[ok].sum() == ok.sum():
            skipped.append(f.name)
            continue
        Xtr, Xte = _fold_features(model, design, f.train, f.test)
        if model.startswith("xgb"):
            import xgboost as xgb

            prm = {k: v for k, v in B.DEFAULTS["xgb"].items()}
            preds = [xgb.XGBClassifier(**prm, objective="binary:logistic", random_state=s, n_jobs=n_jobs,
                                       tree_method="hist").fit(Xtr[ok], ytr[ok].astype(int)).predict_proba(Xte)[:, 1]
                     for s in seeds]
            p.loc[f.test] = np.mean(preds, 0)
        else:
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.model_selection import GroupKFold
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            grp = S.loc[f.train, "site_id"].astype(str).to_numpy()[ok]
            cv = list(GroupKFold(min(5, len(np.unique(grp)))).split(Xtr[ok], ytr[ok], grp))
            m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1.0], cv=cv,
                                                                     max_iter=2000, scoring="neg_log_loss"))
            p.loc[f.test] = m.fit(Xtr[ok], ytr[ok].astype(int)).predict_proba(Xte)[:, 1]
    p_raw = p.to_numpy(float)
    dom = S.loc[p.index, "domain_id"].astype(str).to_numpy()
    y = y_all.loc[p.index].to_numpy()
    return {"index": p.index, "p_raw": p_raw, "p": _platt_crossfit(np.clip(p_raw, 1e-6, 1 - 1e-6), y, dom),
            "y": y, "skipped_folds": skipped}


def climatology_probability(design, folds, code: str) -> dict:
    """Season + location event climatology (L2-logistic on the tier-0 context features, C by site-grouped
    inner CV) and the constant training-fold prevalence, per LODO fold."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from sentinel.models.microbe import features as F

    S = design.samples
    y_all = pd.Series(labels(S, code), index=S.index)
    tested = np.concatenate([f.test for f in folds])
    clim = pd.Series(np.nan, index=pd.Index(tested))
    prev = pd.Series(np.nan, index=pd.Index(tested))
    for f in folds:
        ytr = y_all.loc[f.train].to_numpy()
        ok = np.isfinite(ytr)
        prev.loc[f.test] = float(ytr[ok].mean())
        if ytr[ok].sum() < 2 or ytr[ok].sum() == ok.sum():
            clim.loc[f.test] = float(ytr[ok].mean())
            continue
        Ctr, fill = F.context_features(S, f.train)
        Cte, _ = F.context_features(S, f.test, fill)
        grp = S.loc[f.train, "site_id"].astype(str).to_numpy()[ok]
        cv = list(GroupKFold(min(5, len(np.unique(grp)))).split(Ctr[ok], ytr[ok], grp))
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=cv, max_iter=2000,
                                                                 scoring="neg_log_loss"))
        clim.loc[f.test] = m.fit(Ctr[ok], ytr[ok].astype(int)).predict_proba(Cte)[:, 1]
    return {"index": clim.index, "climatology": clim.to_numpy(float), "prevalence": prev.to_numpy(float)}
