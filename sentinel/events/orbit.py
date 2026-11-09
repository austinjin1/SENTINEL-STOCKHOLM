# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit (M2) event-layer adapter (orbit_protocol.md §12; docs/plans/event_layer_plan.md §2–§4).

(B) output = "is this lake in an algal bloom in this S2 scene" (state, not onset).

- **E4a** chl-a > 20 µg/L (tiers 12 and 50), in-lake rows.
  - Route (i) primary ``derived``: exceedance of the frozen SatWQ-Net equal-weight seed-ensemble mixture.
  - Route (ii) challenger ``direct``: L2 logistic head on the standardised fused embedding (``model.fuse`` output,
    read by a forward hook — the network is not modified) of each frozen seed, fitted on fit-cell labels, C chosen
    on inner log-loss, P averaged over seeds.
  - Comparators: ``index_opt`` / GBM imagery+context (Gaussian P with the inner residual SD), MDN_B (mixture
    exceedance), ``gbm_direct`` (LightGBM binary), spectral indices CI_S2 / MCI / NDCI (univariate logistic), and
    ``climatology`` (LightGBM binary on DOY + location) + constant fit prevalence.
  - Every score gets one Platt recalibration fitted on the val cell (``events.recalibrate``), frozen in the artifact.
- **E3** TSS > 80 mg/L, turbidity > 50 NTU: derived route only, descriptive.

Metrics, exceedance and recalibration come from ``sentinel.eval.events`` (imported, never copied). This module
reads no data; callers pass frames. It is outside the orbit freeze globs and added deliberately to the orbit
scoring-code hash (``sentinel.models.orbit.guard.EVENT_CODE``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# event codes of sentinel.eval.events (thresholds, directions and anchors come from there, never from here)
E4A = "E4a"                         # chl-a > 20 ug/L (= EW-v2 H5b)
E4A_TIERS = ("E4a_t12", "E4a_t50")  # > 12 (WHO 2021 AL1), >= 50 ug/L
E4A_CODES = (E4A,) + E4A_TIERS
E3_DEFS = (("E3a_tss", "tss"), ("E3a_turb", "turbidity"))    # TSS > 80 mg/L, turbidity > 50 NTU (snapshot)
HEAD_CS = (0.01, 0.1, 1.0, 10.0)
GBM_CLS_GRID = [dict(objective="binary", num_leaves=nl, min_child_samples=mc, learning_rate=lr, feature_fraction=0.8,
                     bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)
                for nl, mc, lr in itertools.product((15, 63), (20, 60), (0.03, 0.1))]
GBM_CLS_SEEDS = (0, 1, 2, 3, 4)
SPECTRAL = ("CI_S2", "MCI", "NDCI")
# primary score columns (raw P before recalibration) produced by :func:`event_scores`
ROUTES = ("derived", "direct")
COMPARATORS = ("index_opt_P", "mdn_b_P", "gbm_P", "gbm_direct", "CI_S2", "MCI", "NDCI", "climatology")


# released, immutable shared module (lead notice 2026-11-09): sentinel/eval/events.py v1.0
EVENTS_V1_SHA256 = "4279d2427d777405ce1cb140cab995567a8086fee5fd4776950e4de1065954ae"


def EV():
    """The shared event module (``sentinel/eval/events.py``), refused unless it is the released v1.0 file."""
    import hashlib
    from pathlib import Path

    from ..eval import events

    sha = hashlib.sha256(Path(events.__file__).read_bytes()).hexdigest()
    if sha != EVENTS_V1_SHA256 or events.__version__ != "1.0":
        raise RuntimeError(f"sentinel/eval/events.py sha256 {sha[:12]} / version {events.__version__} is not the "
                           "released v1.0 file")
    return events


def labels(y_linear, code: str) -> np.ndarray:
    """Event state {1, 0, NaN} via ``events.event_state`` on LINEAR in-situ values (never back-transformed log10:
    10**log10(20) > 20 would turn an observed 20.0 ug/L into a positive)."""
    return np.asarray(EV().event_state(np.asarray(y_linear, float), code), float)


# ------------------------------------------------------------------------------------- spectral indices
def spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Water-median (3x3) reflectance indices. CI_S2 = -SS(665; 560, 705): the Wynne et al. (2008) line-height
    form on S2 bands (labelled analogue). MCI = SS(705; 665, 740) (Gower et al. 2005). NDCI (Mishra & Mishra 2012)."""
    b3, b4, b5, b6 = (df[f"rw3_{b}"].to_numpy(float) for b in ("B03", "B04", "B05", "B06"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = -(b4 - b3 - (b5 - b3) * (665.0 - 560.0) / (705.0 - 560.0))
        mci = b5 - b4 - (b6 - b4) * (705.0 - 665.0) / (740.0 - 665.0)
        ndci = (b5 - b4) / (b5 + b4)
    return pd.DataFrame({"CI_S2": ci, "MCI": mci, "NDCI": ndci}, index=df.index)


# ------------------------------------------------------------------------------------- SatWQ inference
def satwq_mixture_embedding(ckpts, df, pix, device="cpu"):
    """Equal-weight seed-ensemble mixture (log10; w/mu/sg [N, T, K*S]) and per-seed fused embeddings [S, N, d].
    The embedding is the input to the mixture head (``model.fuse`` output), captured by a forward hook."""
    import torch

    from ..models.orbit import pipeline as PL
    from ..models.orbit.satwq import SatWQNet, predict_mixture

    ws, mus, sgs, embs = [], [], [], []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        m = SatWQNet(**ck["arch"]).to(device)
        m.load_state_dict(ck["state_dict"])
        nm = PL.Norm.from_dict(ck["norm"])
        ctx_off = bool(ck["config"]["finetune"]["aug"].get("ctx_off", False))
        cap = []
        h = m.fuse.register_forward_hook(lambda mod, inp, out: cap.append(out.detach().cpu().numpy()))
        try:
            w, mu, sg = predict_mixture(m, PL.make_set(df, pix, nm), device, nm.y_mean, nm.y_std, ctx_off=ctx_off)
        finally:
            h.remove()
        ws.append(w / len(ckpts))
        mus.append(mu)
        sgs.append(sg)
        embs.append(np.concatenate(cap).astype(np.float32))
    return (np.concatenate(ws, -1), np.concatenate(mus, -1), np.concatenate(sgs, -1)), np.stack(embs)


def mixture_p(w, mu, sg, code: str) -> np.ndarray:
    """P(event) of a log10 Gaussian mixture [N, K] via ``events.exceedance_probability`` (NaN rows stay NaN)."""
    ev = EV()
    mu = np.asarray(mu, float)
    ok = np.all(np.isfinite(mu), 1)
    out = np.full(len(mu), np.nan)
    if ok.any():
        out[ok] = ev.exceedance_probability(ev.GaussianMixture(np.asarray(w, float)[ok], mu[ok],
                                                               np.asarray(sg, float)[ok], space="log10"), code)
    return out


def gaussian_p(mu_log10, sd_log10: float, code: str) -> np.ndarray:
    """P(event) of Gaussian(point prediction, residual SD) in log10 via ``events.exceedance_probability``."""
    ev = EV()
    mu = np.asarray(mu_log10, float)
    return np.asarray(ev.exceedance_probability(ev.Gaussian(mu, np.full_like(mu, float(sd_log10)), space="log10"),
                                                code), float)


# ------------------------------------------------------------------------------------- fitting (fit cell only)
def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_direct_head(emb_fit, y_fit, emb_inner, y_inner, Cs=HEAD_CS):
    """Per-seed StandardScaler + L2 logistic regression; one C for all seeds by mean inner log-loss."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    mf, mi = np.isfinite(y_fit), np.isfinite(y_inner)
    scores, fits = {}, {}
    for C in Cs:
        heads, ll = [], []
        for s in range(emb_fit.shape[0]):
            sc = StandardScaler().fit(emb_fit[s][mf])
            lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(emb_fit[s][mf]), y_fit[mf].astype(int))
            heads.append((sc, lr))
            ll.append(_logloss(y_inner[mi], lr.predict_proba(sc.transform(emb_inner[s][mi]))[:, 1]))
        scores[C], fits[C] = float(np.mean(ll)), heads
    cb = min(scores, key=scores.get)
    return {"C": cb, "inner_logloss": scores, "heads": fits[cb]}


def predict_direct_head(head, emb) -> np.ndarray:
    return np.mean([lr.predict_proba(sc.transform(emb[s]))[:, 1] for s, (sc, lr) in enumerate(head["heads"])], 0)


def fit_gbm_classifier(X_fit, y_fit, X_inner, y_inner, *, grid=GBM_CLS_GRID, seeds=GBM_CLS_SEEDS, n_threads=6,
                       max_rounds=3000):
    """LightGBM binary; grid by inner log-loss (early stopping on inner); inner-best config refitted with 5 seeds."""
    import lightgbm as lgb

    mf, mi = np.isfinite(y_fit), np.isfinite(y_inner)
    dtr = lgb.Dataset(X_fit[mf], y_fit[mf], free_raw_data=False)
    scores, iters = [], []
    for params in grid:
        dva = lgb.Dataset(X_inner[mi], y_inner[mi], reference=dtr)
        bst = lgb.train(dict(params, verbose=-1, num_threads=n_threads, seed=0), dtr, max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        scores.append(_logloss(y_inner[mi], bst.predict(X_inner[mi], num_iteration=bst.best_iteration)))
        iters.append(max(1, int(bst.best_iteration)))
    jb = int(np.argmin(scores))
    ens = [lgb.train(dict(grid[jb], verbose=-1, num_threads=n_threads, seed=s), dtr, iters[jb]) for s in seeds]
    return {"models": ens, "cols": list(X_fit.columns), "config": grid[jb], "config_index": jb,
            "inner_logloss_grid": scores, "n_rounds": iters[jb]}


def predict_gbm_classifier(m, X) -> np.ndarray:
    return np.mean([b.predict(X[m["cols"]]) for b in m["models"]], 0)


def fit_univariate_logistic(x_fit, y_fit):
    from sklearn.linear_model import LogisticRegression

    ok = np.isfinite(x_fit) & np.isfinite(y_fit)
    mu, sd = float(np.mean(x_fit[ok])), float(np.std(x_fit[ok]) or 1.0)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(((x_fit[ok] - mu) / sd)[:, None], y_fit[ok].astype(int))
    return {"mu": mu, "sd": sd, "lr": lr}


def predict_univariate_logistic(m, x) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    out[ok] = m["lr"].predict_proba(((x[ok] - m["mu"]) / m["sd"])[:, None])[:, 1]
    return out


def residual_sd(pred_log10, y_log10) -> float:
    ok = np.isfinite(pred_log10) & np.isfinite(y_log10)
    return float(np.std(np.asarray(y_log10)[ok] - np.asarray(pred_log10)[ok], ddof=1))


# ------------------------------------------------------------------------------------- scores
def event_scores(art: dict, *, mix, emb, chl_preds: dict, Xgbm: pd.DataFrame, Xclim: pd.DataFrame,
                 spec: pd.DataFrame, mdn_mix, code: str) -> pd.DataFrame:
    """Raw (uncalibrated) P(event ``code``) per route/comparator for one frame.

    mix = (w, mu, sg) of the SatWQ ensemble for chl-a [N, K]; emb = per-seed embeddings [S, N, d];
    chl_preds = {"index_opt": log10, "gbm": log10} point predictions; mdn_mix = (w, mu, sg) of MDN_B for chl-a;
    Xgbm = imagery+context features; Xclim = DOY + location features; spec = :func:`spectral_indices`."""
    a = art["by_event"][code]
    out = {"derived": mixture_p(*mix, code),
           "direct": predict_direct_head(a["direct_head"], emb),
           "index_opt_P": gaussian_p(chl_preds["index_opt"], art["resid_sd"]["index_opt"], code),
           "gbm_P": gaussian_p(chl_preds["gbm"], art["resid_sd"]["gbm"], code),
           "mdn_b_P": mixture_p(*mdn_mix, code),
           "gbm_direct": predict_gbm_classifier(a["gbm_direct"], Xgbm),
           "climatology": predict_gbm_classifier(a["climatology"], Xclim)}
    for s in SPECTRAL:
        out[s] = predict_univariate_logistic(a["spectral"][s], spec[s].to_numpy(float))
    return pd.DataFrame(out)


def calibrate_all(art: dict, scores: pd.DataFrame, code: str) -> pd.DataFrame:
    """Apply the frozen val Platt calibrators (``events.Calibrator``) of the artifact, one per score column."""
    cals = art["by_event"][code]["calibrators"]
    return pd.DataFrame({c: cals[c](scores[c].to_numpy(float)) for c in scores.columns}, index=scores.index)


MDN_B7 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")


def chl_point_inputs(fitted, comparators, df, Ximg, Xctx):
    """index_opt and the val-selected GBM imagery+context chl-a point predictions (log10); MDN_B mixture."""
    gname = comparators["chla"]["gbm_imagery_context"]["model"]
    base, _, v = gname.partition("#v")
    X = pd.concat([Ximg, Xctx], axis=1)
    gbm = fitted[base](df, X, **({"variant": int(v)} if v else {}))["chla"]
    idx = fitted["index_opt"](df, Ximg)["chla"]
    mname = comparators["chla"]["mdn_b"]["model"]
    from ..models.orbit.data import BANDS

    bands = BANDS if mname.startswith("mdn_b12") else MDN_B7
    Xl = np.stack([df[f"rw3_{b}"].to_numpy(float) for b in bands], 1)
    mdn = mdn_mixture(fitted[(mname, "chla")], Xl)
    return {"index_opt": idx, "gbm": gbm}, mdn, {"gbm": gname, "mdn_b": mname}


def mdn_mixture(fitted_mdn, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MDN_B per-init mixtures [E, N, K] -> one equal-weight mixture [N, E*K] (as ``MDNRegressor.quantiles``)."""
    logw, mu, sg = fitted_mdn.mixture(X)
    E, N, K = mu.shape
    w = np.exp(logw).transpose(1, 0, 2).reshape(N, E * K) / E
    return w, mu.transpose(1, 0, 2).reshape(N, E * K), sg.transpose(1, 0, 2).reshape(N, E * K)
