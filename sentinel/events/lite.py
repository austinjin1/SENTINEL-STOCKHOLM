# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite event-layer adapter (docs/plans/event_layer_plan.md §2–§4; lite_protocol.md §13).

(B) output = "is this water body in a bloom / plume in this image" (state, not onset).

- **E4a**, algal bloom: chl-a > 20 µg/L, lakes/reservoirs.
  - Route (i), primary: P from each seed's Gaussian(μ, σ) in log10, via ``sentinel.eval.events``, averaged over
    seeds.
  - Route (ii), challenger: L2 logistic head (C = 1, fixed) on the frozen student's water-pooled decoder features,
    fitted on train chips, averaged over seeds.
  - Comparators: band-ratio and GBM-RGBN chl-a regressions turned into P with their own train cross-validated
    residual SD, and location climatology (positive rate of the 10 nearest train sites, add-one smoothed towards
    the train prevalence).
  - Every route is recalibrated with one Platt fit on val (``events.recalibrate``). Val metrics use water-body
    cross-fitted calibration.
- **E3a_tss / E3a_turb**, sediment plume (snapshot codes of events v1.0): TSS > 80 mg/L, turbidity > 50 NTU. Route (i) only, descriptive on the
  original track.

This module computes no metric itself: metrics, recalibration and exceedance come from ``sentinel.eval.events``
(imported, never copied). It reads no test data; callers pass frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

E4A = {"code": "E4a", "target": "chla", "threshold": 20.0, "population": "lake", "tiers": ("E4a_t12", "E4a_t50")}
E3 = ({"code": "E3a_tss", "target": "tss", "threshold": 80.0, "population": "all"},
      {"code": "E3a_turb", "target": "turbidity", "threshold": 50.0, "population": "all"})
EVENTS_V1_SHA256 = "4279d2427d777405ce1cb140cab995567a8086fee5fd4776950e4de1065954ae"   # released, immutable
HEAD_C = 1.0
CLIM_K = 10


def EV():
    """The shared event module (released v1.0, immutable; its sha256 is checked here and pinned in the freeze)."""
    import hashlib
    from pathlib import Path

    from ..eval import events

    sha = hashlib.sha256(Path(events.__file__).read_bytes()).hexdigest()
    if sha != EVENTS_V1_SHA256:
        raise RuntimeError(f"sentinel/eval/events.py sha256 {sha[:12]} is not the released v1.0 {EVENTS_V1_SHA256[:12]}")
    return events


def labels(y_log10, code: str = "E4a") -> np.ndarray:
    """Event state (1/0, NaN where missing) via ``events.event_state`` on linear values. Back-transformed values are
    rounded to 6 decimals so that a label of exactly the threshold stays at the threshold (release note)."""
    y = np.asarray(y_log10, float)
    x = np.where(np.isfinite(y), np.round(10.0 ** y, 6), np.nan)
    return np.asarray(EV().event_state(x, code), float)


def derived_p(mu_log10, sd_log10, code: str = "E4a") -> np.ndarray:
    """Route (i): P(event) from Gaussian(μ, σ) in log10 space, via ``events.exceedance_probability``."""
    ev = EV()
    return np.asarray(ev.exceedance_probability(ev.Gaussian(np.asarray(mu_log10, float), np.asarray(sd_log10, float),
                                                            space="log10"), code), float)


def derived_p_ensemble(mus, sds, code: str = "E4a") -> np.ndarray:
    """Seed-averaged exceedance probability (equal-weight mixture of the seed Gaussians)."""
    return np.mean([derived_p(m, s, code) for m, s in zip(mus, sds)], 0)


# ------------------------------------------------------------------ route (ii): head on frozen student features
def pooled_embedding(model, inp, cfg, batch: int = 64) -> np.ndarray:
    """(N, dec_ch) water-pooled decoder features (input of the final 1×1 conv) over the same centre window and crop
    as ``train.predict``. Inference only (weights frozen)."""
    import torch

    from ..models.lite.train import centre_crop, normalise_input, scene_cond

    feats = {}
    hook = model.head[0].register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    out = []
    model.eval()
    try:
        with torch.no_grad():
            for s in range(0, inp.n, batch):
                dn = inp.dn[s:s + batch].float()
                w = inp.w[s:s + batch]
                dn, w = centre_crop(dn, w, cfg.get("crop", 0))
                x, x01 = normalise_input(dn, inp.sp[s:s + batch], model, cfg["input_mode"], cfg["rgb_only"])
                model(x, x01 if model.needs_x01 else None, scene_cond(inp.sp[s:s + batch]))
                f = feats["f"].float()
                out.append((f * w.unsqueeze(1)).sum((2, 3)).cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(out)


def fit_head(X: np.ndarray, y: np.ndarray) -> dict:
    """L2 logistic regression (C = 1, lbfgs), standardised features; train chips only. JSON-serialisable."""
    from sklearn.linear_model import LogisticRegression

    ok = np.isfinite(y)
    mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-6
    clf = LogisticRegression(C=HEAD_C, max_iter=2000).fit((X[ok] - mu) / sd, y[ok].astype(int))
    return {"mean": mu.tolist(), "std": sd.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "n_train": int(ok.sum()), "n_pos": int(y[ok].sum()), "C": HEAD_C}


def head_p(X: np.ndarray, head: dict) -> np.ndarray:
    z = ((X - np.asarray(head["mean"])) / np.asarray(head["std"])) @ np.asarray(head["coef"]) + head["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ comparators
def climatology_p(tr: pd.DataFrame, y_tr: np.ndarray, ev_lat, ev_lon, k: int = CLIM_K) -> np.ndarray:
    """Positive rate among the k nearest train *sites* (site rate = mean over its chips), add-one smoothed towards the
    train prevalence. Train-only information."""
    from ..eval.splits import nearest_distance_km  # noqa: F401  (same geometry conventions)
    from sklearn.neighbors import BallTree

    d = pd.DataFrame({"site_id": tr["site_id"].to_numpy(), "lat": tr["lat"].to_numpy(float),
                      "lon": tr["lon"].to_numpy(float), "y": y_tr})
    d = d[np.isfinite(d["y"])]
    s = d.groupby("site_id").agg(lat=("lat", "mean"), lon=("lon", "mean"), y=("y", "mean"))
    p0 = float(d["y"].mean())
    tree = BallTree(np.radians(s[["lat", "lon"]].to_numpy()), metric="haversine")
    _, idx = tree.query(np.radians(np.c_[np.asarray(ev_lat, float), np.asarray(ev_lon, float)]), k=min(k, len(s)))
    rates = s["y"].to_numpy()[idx]
    return (rates.sum(1) + p0) / (idx.shape[1] + 1.0)


def cv_residual_sd(fit_predict, X_tr: pd.DataFrame, y_tr: np.ndarray, groups, n_folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold residual SD (log10) of a baseline regression on train, water-body-grouped folds."""
    from sklearn.model_selection import GroupKFold

    ok = np.isfinite(y_tr)
    Xo, yo, go = X_tr[ok].reset_index(drop=True), y_tr[ok], np.asarray(groups)[ok]
    res = np.full(len(yo), np.nan)
    for a, b in GroupKFold(n_splits=n_folds).split(Xo, yo, go):
        res[b] = yo[b] - fit_predict(Xo.iloc[a], yo[a], Xo.iloc[b])
    return float(np.nanstd(res, ddof=1))


# ------------------------------------------------------------------ calibration (val) and metrics (shared module)
def crossfit_calibrated(p_val: np.ndarray, y_val: np.ndarray, groups, n_folds: int = 5) -> np.ndarray:
    """Val probabilities recalibrated out-of-fold by water body (honest val metrics); uses events.recalibrate."""
    from sklearn.model_selection import GroupKFold

    ev = EV()
    out = np.full(len(p_val), np.nan)
    ok = np.isfinite(y_val) & np.isfinite(p_val)
    idx = np.flatnonzero(ok)
    g = np.asarray(groups)[ok]
    for a, b in GroupKFold(n_splits=n_folds).split(idx, groups=g):
        cal = ev.recalibrate(p_val[idx[a]], y_val[idx[a]], method="platt")
        out[idx[b]] = cal.transform(p_val[idx[b]])
    return out


def fit_calibrator(p_val: np.ndarray, y_val: np.ndarray):
    ok = np.isfinite(y_val) & np.isfinite(p_val)
    return EV().recalibrate(p_val[ok], y_val[ok], method="platt")


def calibrator_to_dict(cal) -> dict:
    from dataclasses import asdict

    d = asdict(cal)
    d["params"] = {k: (np.asarray(v).tolist() if not np.isscalar(v) else float(v)) for k, v in d["params"].items()}
    return d


def calibrator_from_dict(d: dict):
    ev = EV()
    params = {k: (np.asarray(v, float) if isinstance(v, list) else float(v)) for k, v in d["params"].items()}
    return ev.Calibrator(d["method"], params, int(d["n_fit"]), int(d["n_pos_fit"]), d["fit_sha256"])
