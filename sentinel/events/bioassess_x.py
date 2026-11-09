# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X secondary event track (protocol ``bioassess_x_protocol.md`` amendment v8; ``docs/plans/event_layer_plan.md``).

Event definitions, exceedance probabilities, recalibration and metrics come from the shared
``sentinel.eval.events`` (imported, never copied). This adapter only maps M5-X targets (log10 / identity scales) to
the events' units, builds the routes (derived from the M5-X predictive distribution; direct LightGBM challenger for
powered events) and the comparators (threshold-on-baseline; prevalence; climatology), all fitted on training /
development rows only. It reads no data itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (block, event) -> (M5-X target column, scale of the target, factor to the event's unit)
#   scale "log10": value_in_event_unit = 10 ** y * factor ; "identity": value = y
TRACK = {
    ("X-L", "E2a"): ("log10_chloride_mg_l", "log10", 1.0),        # mg/L
    ("X-L", "E5b"): ("log10_tp_ug_l", "log10", 1e-3),             # µg/L -> mg/L
    ("X-L", "E6"): ("ph_lab", "identity", 1.0),
    ("X-L", "E3"): ("log10_turbidity_ntu", "log10", 1.0),         # NTU
    ("X-U", "E2a"): ("uk_log10_chloride_mg_l", "log10", 1.0),
    ("X-U", "E6"): ("uk_ph", "identity", 1.0),
}
# shared snapshot codes (events.py v1): lab chloride, TP, lab pH range, turbidity > 50 NTU (snapshot variant of E3a)
EVENT_CODE = {"E2a": "E2a_lab", "E5b": "E5b", "E6": "E6_snap", "E3": "E3a_turb"}
# dev power rule (v8.2 as corrected by v8a: the rule governs; X-U E6 35 pos / 25 clusters is powered)
POWERED = {("X-L", "E5b"), ("X-L", "E6"), ("X-U", "E6")}            # direct head only here
DESCRIPTIVE_ONLY = {("X-L", "E2a"), ("X-L", "E3"), ("X-U", "E2a")}
COMPARATORS = ("rf_geography", "tabpfn_community")                   # threshold-on-baseline (derived route)
MIN_POS, MIN_CLUSTERS = 30, 10


EVENTS_V1_SHA256 = "4279d2427d777405ce1cb140cab995567a8086fee5fd4776950e4de1065954ae"   # released v1.0 (0444)


def events_module():
    """The shared module (plan §1.3), released v1.0 only (sha256-pinned); ImportError if absent."""
    import hashlib

    from sentinel.eval import events as EV  # noqa: PLC0415

    if hashlib.sha256(open(EV.__file__, "rb").read()).hexdigest() != EVENTS_V1_SHA256:
        raise ImportError("sentinel/eval/events.py is not the released v1.0 (sha256 mismatch)")
    return EV


def event_def(code: str):
    return events_module().get_event(EVENT_CODE[code])


def to_event_units(y, block: str, code: str) -> np.ndarray:
    _, scale, fac = TRACK[(block, code)]
    y = np.asarray(y, float)
    # back-transform rounded to 6 decimals (10**log10(230) can exceed 230 in floating point; lead caveat, events v1.0)
    return np.round(10.0 ** y * fac, 6) if scale == "log10" else y


def labels(y, block: str, code: str) -> np.ndarray:
    """Event state {0, 1, NaN} of the measured target, by ``events.event_state`` in the event's units."""
    return np.asarray(events_module().event_state(to_event_units(y, block, code), event_def(code)), float)


def derived_probability(mu, sd, block: str, code: str) -> np.ndarray:
    """P(event) from a Gaussian predictive distribution on the model's target scale (log10 or identity), via
    ``events.exceedance_probability`` (log-space aware)."""
    EV = events_module()
    _, scale, fac = TRACK[(block, code)]
    mu = np.asarray(mu, float)
    if scale == "log10":
        mu = mu + np.log10(fac)                    # target unit -> event unit in log10 space (e.g. µg/L -> mg/L)
    sigma = np.broadcast_to(np.asarray(sd, float), mu.shape).astype(float)
    pred = EV.Gaussian(mu=mu, sigma=sigma, space="log10" if scale == "log10" else "linear")
    return np.asarray(EV.exceedance_probability(pred, event_def(code)), float)


def residual_sd(y, p, fold, leave_out=None) -> float:
    """Out-of-fold residual SD on development rows (optionally excluding one development fold: cross-fitting)."""
    y, p, fold = np.asarray(y, float), np.asarray(p, float), np.asarray(fold)
    ok = np.isfinite(y) & np.isfinite(p)
    if leave_out is not None:
        ok &= fold != leave_out
    return float(np.std(y[ok] - p[ok], ddof=1))


def crossfit_calibrated(p_raw, y_evt, fold) -> np.ndarray:
    """Development numbers: Platt (``events.recalibrate``) fitted on the other development folds, applied to fold g."""
    EV = events_module()
    p_raw, y_evt, fold = np.asarray(p_raw, float), np.asarray(y_evt, float), np.asarray(fold)
    out = np.full(len(p_raw), np.nan)
    for g in np.unique(fold):
        tr = (fold != g) & np.isfinite(y_evt) & np.isfinite(p_raw)
        te = (fold == g) & np.isfinite(p_raw)
        if tr.sum() < 10 or len(np.unique(y_evt[tr])) < 2:
            out[te] = p_raw[te]
            continue
        cal = EV.recalibrate(p_raw[tr], y_evt[tr])
        out[te] = cal(p_raw[te]) if callable(cal) else cal.transform(p_raw[te])
    return out


def frozen_calibrated(p_dev_raw, y_dev, p_new_raw) -> np.ndarray:
    """Test look: Platt fitted on ALL development out-of-fold probabilities (frozen data), applied to new rows."""
    EV = events_module()
    ok = np.isfinite(p_dev_raw) & np.isfinite(y_dev)
    cal = EV.recalibrate(np.asarray(p_dev_raw, float)[ok], np.asarray(y_dev, float)[ok])
    return np.asarray(cal(p_new_raw) if callable(cal) else cal.transform(p_new_raw), float)


def month_of(doy_sin, doy_cos) -> np.ndarray:
    doy = (np.arctan2(np.asarray(doy_sin, float), np.asarray(doy_cos, float)) % (2 * np.pi)) * 365.25 / (2 * np.pi)
    return np.clip((doy // 30.44).astype(int) + 1, 1, 12)


def prevalence_and_climatology(y_train, month_train, month_test) -> tuple[np.ndarray, np.ndarray]:
    """Fold-training prevalence, and prevalence by sampling month (add-one smoothing; fallback prevalence)."""
    y_train = np.asarray(y_train, float)
    ok = np.isfinite(y_train)
    prev = float(y_train[ok].mean()) if ok.any() else np.nan
    mt = np.asarray(month_train)[ok]
    yt = y_train[ok]
    clim = {}
    for m in range(1, 13):
        sel = mt == m
        clim[m] = (yt[sel].sum() + prev) / (sel.sum() + 1.0) if sel.any() else prev
    return np.full(len(month_test), prev), np.array([clim.get(int(m), prev) for m in month_test])


def direct_classifier(X_train, y_train, X_test, seed: int, n_jobs: int = 5) -> np.ndarray:
    """Challenger head: LightGBM binary on the community features (M5 fixed GBM hyperparameters)."""
    import lightgbm as lgb

    from sentinel.models.bioassess.xdev import GBM_PARAMS

    ok = np.isfinite(y_train)
    m = lgb.LGBMClassifier(**GBM_PARAMS, random_state=seed, n_jobs=n_jobs)
    m.fit(X_train[ok], y_train[ok].astype(int))
    return m.predict_proba(X_test)[:, 1]


def power_flag(y_evt, clusters) -> dict:
    y = np.asarray(y_evt, float)
    pos = y == 1
    n_pos, n_cl = int(pos.sum()), int(pd.Series(np.asarray(clusters)[pos]).nunique())
    try:
        word = events_module().power_flag(n_pos, n_cl, MIN_POS, MIN_CLUSTERS)
    except ImportError:
        word = "feasible" if (n_pos >= MIN_POS and n_cl >= MIN_CLUSTERS) else "underpowered"
    flag = "F" if str(word).lower() in ("f", "feasible") else "U"
    return {"n": int(np.isfinite(y).sum()), "n_pos": n_pos, "n_clusters_pos": n_cl, "flag": flag, "flag_word": word}


def metric_value(m: dict, key: str) -> tuple[float, float | None, float | None]:
    """(point, ci_low, ci_high) of a ``state_metrics`` entry ({point, ci_low, ci_high} dict or a float)."""
    v = m.get(key)
    if isinstance(v, dict):
        return float(v.get("point", np.nan)), v.get("ci_low"), v.get("ci_high")
    return (float(v) if v is not None else np.nan), None, None


def metrics(y_evt, p, clusters, p_ref=None, n_boot: int = 2000) -> dict:
    return events_module().state_metrics(np.asarray(y_evt, float), np.asarray(p, float), groups=np.asarray(clusters),
                                         p_ref=p_ref, n_boot=n_boot)


def compare(y_evt, p_a, p_b, clusters) -> dict:
    return events_module().compare_routes(np.asarray(y_evt, float), np.asarray(p_a, float), np.asarray(p_b, float),
                                          groups=np.asarray(clusters))


# --------------------------------------------------------------------------- fold-level pieces (dev and test)
def train_event_labels(D, block: str, code: str, ids) -> np.ndarray:
    """Event labels of training rows (same-continent rows of the block; the M5-X loader guard already applied)."""
    from sentinel.models.bioassess import xdev as XD

    t = TRACK[(block, code)][0]
    y = XD.labels(D, block, t, ids, "same")
    out = np.full(len(y), np.nan)
    ok = np.isfinite(y)
    out[ok] = labels(y[ok], block, code)
    return out


def fold_prevalence_climatology(D, block: str, code: str, fo) -> tuple[np.ndarray, np.ndarray]:
    s_tr, s_te = D.samples.loc[fo.train], D.samples.loc[fo.test]
    y = train_event_labels(D, block, code, fo.train)
    return prevalence_and_climatology(y, month_of(s_tr["doy_sin"], s_tr["doy_cos"]),
                                      month_of(s_te["doy_sin"], s_te["doy_cos"]))


def fold_direct(D, TM, block: str, code: str, fo, seed: int, n_jobs: int = 5) -> np.ndarray:
    """Direct challenger for one fold: features fitted on the fold's labelled training rows only."""
    from sentinel.models.bioassess import xdev as XD

    y = train_event_labels(D, block, code, fo.train)
    lab = fo.train[np.isfinite(y)]
    yl = y[np.isfinite(y)]
    fz = XD.XFeaturizer(block).fit(TM, D, lab)
    return direct_classifier(fz.transform(TM, D, lab), yl, fz.transform(TM, D, fo.test), seed, n_jobs)
