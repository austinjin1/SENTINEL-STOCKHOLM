# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 event heads for SENTINEL-Sonde: P(onset within L days), L ∈ {1, 3, 7}, for the daily-tier
hazards of :mod:`sentinel.eval.hazards` (H1 hypoxia + tiers on DO daily minimum, H4 nitrate,
H5b in-situ chlorophyll).

Labels: ``hazards.daily_onsets`` + ``hazards.issue_labels`` per site on observed data; an issue
row (site, d) is used when the label is valid (d not an event day, d observed, outcome known),
d is an issue date of the block with d + L inside the block, and the hazard variable is
forecastable at d (common rows of :mod:`.common`, i.e. finite climatology) — the SAME rows for
every method.

Methods
  (a) ``quantile``     exceedance probability from any model's 19 predictive quantiles at the
                       target days h ≤ L (piecewise-linear CDF with linear tails), combined over
                       h by max (default) or independence;
  (b) ``event_gbm``    LightGBM binary classifier on the causal issue-date features;
      ``lstm_head``    logistic head on the frozen LSTM encoder embedding;
  baselines
      ``climatology``  site × DOY harmonic logistic (K harmonics), partial pooling: pooled fit on
                       train-role sites + ridge-penalised per-site deviations fitted on the site's
                       own TRAINING-years rows (also for val-role sites, whose training-period
                       history is legitimately available);
      ``persistence``  P(onset | current value bin), bins of the margin to the threshold, pooled
                       over train-role sites × train block (Laplace-smoothed);
      ``threshold``    current-value rule: score = −margin (alarm when within δ of the
                       criterion; δ is set by the false-alarm-budget calibration on val);
      ``logistic``     L2 logistic regression on lag features.
All fitting uses train-role sites × TRAIN block (climatology: each site's training years);
early stopping / calibration only on the validation block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...eval.hazards import HAZARDS, daily_onsets, issue_labels
from ..baselines.sonde_baselines import QUANTILES
from .common import SondeData
from .data import from_model_space, to_model_space

EVENT_HAZARDS = ("H1_hypoxia", "H1_hypoxia_t4", "H1_hypoxia_t2", "H2_chloride", "H2_chloride_acute",
                 "H4_nitrate", "H5b_chl_bloom")
CHLORIDE_HAZARD_VARS = ("cl_mgl_4dmean", "cl_mgl_max")
LEADS = (1, 3, 7)


# ----------------------------------------------------------------------------- labels & rows

@dataclass
class HazardLabels:
    hazard: str
    event: np.ndarray                 # (S, T) bool
    onset: np.ndarray                 # (S, T) bool
    y: dict = field(default_factory=dict)      # L -> (S, T) int8
    valid: dict = field(default_factory=dict)  # L -> (S, T) bool


def hazard_labels(data: SondeData, hazard: str, leads=LEADS, approved_only: bool = True) -> HazardLabels:
    """Event/onset/label arrays; with ``approved_only`` provisional values are treated as missing
    (protocol §6: Approved values only)."""
    hz = HAZARDS[hazard]
    j = data.cube.var_index(hz.variable)
    x = data.cube.y[:, :, j].astype(float)
    if approved_only:
        x = np.where(data.cube.prov[:, :, j], np.nan, x)
    S, T = x.shape
    ev = np.zeros((S, T), bool)
    on = np.zeros((S, T), bool)
    lab = HazardLabels(hazard, ev, on)
    for L in leads:
        lab.y[L] = np.zeros((S, T), np.int8)
        lab.valid[L] = np.zeros((S, T), bool)
    for s in range(S):
        if not np.isfinite(x[s]).any():
            continue
        e, o = daily_onsets(x[s], hz)
        ev[s], on[s] = e, o
        for L in leads:
            y, v = issue_labels(x[s], e, o, L, refractory_days=hz.refractory_days)
            lab.y[L][s], lab.valid[L][s] = y, v
    return lab


@dataclass
class EventRows:
    hazard: str
    lead: int
    block: str
    si: np.ndarray
    ti: np.ndarray
    y: np.ndarray
    site_no: np.ndarray
    issue_date: np.ndarray
    role: np.ndarray

    def __len__(self) -> int:
        return int(self.si.size)


def event_rows(data: SondeData, labels: HazardLabels, lead: int, block: str, roles=("train", "val")) -> EventRows:
    hz = HAZARDS[labels.hazard]
    b = data.baselines(hz.variable)
    t_idx = data.issue_idx(block, lead)
    s_idx = data.site_idx(roles)
    forecastable = np.isfinite(b.last[np.ix_(s_idx, t_idx)]) & np.isfinite(b.cm[s_idx][:, b.doy[t_idx]])
    m = labels.valid[lead][np.ix_(s_idx, t_idx)] & forecastable
    a, k = np.nonzero(m)
    si, ti = s_idx[a].astype(np.int64), t_idx[k].astype(np.int64)
    return EventRows(labels.hazard, lead, block, si, ti, labels.y[lead][si, ti].astype(np.int8),
                     np.asarray(data.sites)[si], data.cube.dates.to_numpy()[ti], data.roles[si])


# ----------------------------------------------------------------------------- (a) quantiles

def exceedance_prob(q: np.ndarray, threshold: float, direction: str, levels=QUANTILES) -> np.ndarray:
    """P(X < thr) ('below') or P(X > thr) ('above') from quantiles q (n, K) of one variable (any
    monotone space; ``threshold`` in the same space). Piecewise-linear CDF through (q_k, τ_k);
    tails extend the CDF linearly with the slope of the outermost interval, clipped to [0, 1]."""
    q = np.sort(np.asarray(q, float), axis=-1)
    tau = np.asarray(levels, float)
    n, K = q.shape
    x = np.full(n, float(threshold))
    F = np.empty(n)
    eps = 1e-9
    lo_slope = (tau[1] - tau[0]) / np.maximum(q[:, 1] - q[:, 0], eps)
    hi_slope = (tau[-1] - tau[-2]) / np.maximum(q[:, -1] - q[:, -2], eps)
    below = x <= q[:, 0]
    above = x >= q[:, -1]
    F[below] = np.clip(tau[0] - lo_slope[below] * (q[below, 0] - x[below]), 0.0, tau[0])
    F[above] = np.clip(tau[-1] + hi_slope[above] * (x[above] - q[above, -1]), tau[-1], 1.0)
    mid = ~(below | above)
    if mid.any():
        qm = q[mid]
        k = np.clip((qm < x[mid, None]).sum(axis=1) - 1, 0, K - 2)
        r = np.arange(qm.shape[0])
        q0, q1 = qm[r, k], qm[r, k + 1]
        w = np.where(q1 - q0 > eps, (x[mid] - q0) / np.maximum(q1 - q0, eps), 0.5)
        F[mid] = tau[k] + w * (tau[k + 1] - tau[k])
    return F if direction == "below" else 1.0 - F


def onset_prob_from_quantiles(q: np.ndarray, horizons, lead: int, hazard: str, *, combine: str = "max") -> np.ndarray:
    """q (n, H, 19) model-space quantiles of the hazard variable -> P(event day in (d, d+lead])."""
    from .evaluate import clip_plausible

    hz = HAZARDS[hazard]
    thr = float(to_model_space(hz.variable, np.array([hz.threshold]))[0])
    q = clip_plausible(hz.variable, q)
    # horizons available <= lead (e.g. 1,2,3,5,7 for L = 7: days 4 and 6 are not forecast; the
    # same for every quantile model, absorbed by Platt recalibration)
    ks = [k for k, h in enumerate(horizons) if h <= lead]
    p = np.stack([exceedance_prob(q[:, k], thr, hz.direction) for k in ks], axis=1)
    if combine == "max":
        return p.max(axis=1)
    if combine == "indep":
        return 1.0 - np.prod(1.0 - p, axis=1)
    raise ValueError(combine)


# ----------------------------------------------------------------------------- features

def margin_model_space(var: str, hazard: str, z: np.ndarray) -> np.ndarray:
    """Distance to the threshold on the SAFE side (positive = no event) in model space."""
    hz = HAZARDS[hazard]
    thr = float(to_model_space(var, np.array([hz.threshold]))[0])
    return (z - thr) if hz.direction == "below" else (thr - z)


LAG_FEATURES = ("margin_last", "margin_min7", "margin_max7", "anom_lag1", "anom_lag2", "anom_mean3",
                "anom_mean7", "slope7", "std7", "age", "frac_obs7", "doy_sin", "doy_cos", "logq", "dlogq1",
                "dlogq7", "anom0_temp_c_mean")


def event_features(fb, data: SondeData, rows: EventRows, blocks: dict | None = None) -> pd.DataFrame:
    hz = HAZARDS[rows.hazard]
    thr = float(to_model_space(hz.variable, np.array([hz.threshold]))[0])
    X = fb.table(hz.variable, rows.si, rows.ti, rows.lead, threshold_model=thr, blocks=blocks)
    if hz.direction == "above":  # orient margins so that positive = safe side
        for c in ("margin_last", "margin_min7", "margin_max7"):
            X[c] = -X[c]
        X["margin_min7"], X["margin_max7"] = X["margin_max7"].copy(), X["margin_min7"].copy()
    return X


# ----------------------------------------------------------------------------- (b) classifiers

@dataclass
class EventGBMConfig:
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    max_rounds: int = 1500
    early_stopping_rounds: int = 100
    num_threads: int = 16
    seed: int = 0
    # optional monotone constraints {feature: +1/-1}; margins are oriented so that larger = safer,
    # hence -1 (risk can only fall as the margin to the criterion grows). Absent features are ignored.
    monotone: dict | None = None
    # seed bagging: average of n_bags boosters with seeds seed, seed+1, ... (variance reduction; val
    # showed run-to-run BSS differences of ~0.01 from thread count / seed alone, 2026-11-08)
    n_bags: int = 1


class EventGBM:
    def __init__(self, cfg: EventGBMConfig | None = None):
        self.cfg = cfg or EventGBMConfig()
        self.booster = None
        self.info = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_es: pd.DataFrame | None = None, y_es: np.ndarray | None = None):
        import lightgbm as lgb

        c = self.cfg
        params = {"objective": "binary", "learning_rate": c.learning_rate, "num_leaves": c.num_leaves,
                  "min_data_in_leaf": c.min_data_in_leaf, "feature_fraction": c.feature_fraction,
                  "bagging_fraction": c.bagging_fraction, "bagging_freq": c.bagging_freq, "lambda_l2": c.lambda_l2,
                  "num_threads": c.num_threads, "seed": c.seed, "deterministic": True, "force_row_wise": True,
                  "verbose": -1}
        if c.monotone:
            mono = [int(c.monotone.get(col, 0)) for col in X.columns]
            if any(mono):
                params.update({"monotone_constraints": mono, "monotone_constraints_method": "intermediate"})
        self.boosters = []
        for k in range(max(1, int(c.n_bags))):
            ds = lgb.Dataset(X, y.astype(float))
            valid, cb = None, [lgb.log_evaluation(0)]
            if X_es is not None and y_es is not None and y_es.sum() > 0:
                valid = [lgb.Dataset(X_es, y_es.astype(float), reference=ds)]
                cb.append(lgb.early_stopping(c.early_stopping_rounds, verbose=False))
            self.boosters.append(lgb.train({**params, "seed": c.seed + k}, ds, num_boost_round=c.max_rounds,
                                           valid_sets=valid, callbacks=cb))
        self.booster = self.boosters[0]
        self.info = {"n": int(len(y)), "n_pos": int(y.sum()), "n_bags": len(self.boosters),
                     "best_iteration": [int(b.best_iteration or b.current_iteration()) for b in self.boosters]}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        bs = getattr(self, "boosters", None) or [self.booster]
        return np.mean([b.predict(X, num_iteration=b.best_iteration or None) for b in bs], axis=0)

    def to_json(self) -> str:
        """All bagged boosters (each truncated at its best iteration) in one JSON artifact."""
        import json

        bs = getattr(self, "boosters", None) or [self.booster]
        return json.dumps({"boosters": [b.model_to_string(num_iteration=b.best_iteration or None) for b in bs]})

    @classmethod
    def from_file(cls, path) -> "EventGBM":
        import json

        import lightgbm as lgb

        m, path = cls(), str(path)
        if path.endswith(".json"):
            m.boosters = [lgb.Booster(model_str=t) for t in json.loads(open(path).read())["boosters"]]
        else:
            m.boosters = [lgb.Booster(model_file=path)]
        m.booster = m.boosters[0]
        return m


class LagLogistic:
    """L2 logistic regression on standardised lag features (NaN -> 0 + missing indicator)."""

    def __init__(self, C: float = 1.0, features=LAG_FEATURES):
        self.C, self.features = C, list(features)

    def _mat(self, X: pd.DataFrame) -> np.ndarray:
        A = X.reindex(columns=self.features).to_numpy(dtype=float)
        miss = ~np.isfinite(A)
        Z = np.where(miss, 0.0, (np.where(miss, 0.0, A) - self.mu) / self.sd)
        return np.concatenate([Z, miss[:, self.miss_cols].astype(float)], axis=1)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression

        # parity: optional sub-daily (sd_*) and weather features join the lag features when present
        extra = [c for c in X.columns if c.startswith("sd_") or c.startswith(("precip_", "tmax_", "tmin_", "srad_",
                                                                               "wind_", "vpd_"))]
        self.features = list(dict.fromkeys(list(self.features) + extra))
        A = X.reindex(columns=self.features).to_numpy(dtype=float)
        self.mu = np.nanmean(A, axis=0)
        self.sd = np.nanstd(A, axis=0) + 1e-6
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        self.miss_cols = np.flatnonzero((~np.isfinite(A)).any(axis=0))
        self.lr = LogisticRegression(C=self.C, max_iter=1000)
        self.lr.fit(self._mat(X), y.astype(int))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self._mat(X))[:, 1]


class EmbeddingLogistic:
    """Logistic head on frozen encoder embeddings (the LSTM 'event head')."""

    def __init__(self, C: float = 0.1):
        self.C = C

    def fit(self, E: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """``sample_weight`` = 1/keep-rate for subsampled negatives keeps probabilities unbiased."""
        from sklearn.linear_model import LogisticRegression

        self.mu, self.sd = E.mean(axis=0), E.std(axis=0) + 1e-6
        self.lr = LogisticRegression(C=self.C, max_iter=2000)
        self.lr.fit((E - self.mu) / self.sd, y.astype(int), sample_weight=sample_weight)
        return self

    def predict(self, E: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba((E - self.mu) / self.sd)[:, 1]


class StackLogistic:
    """Stacked combination of method probabilities: L2 logistic regression on standardised
    logit(p_m) of the member methods' RAW (un-recalibrated) probabilities. Fitted on the calibration
    (val) block only, like Platt scaling; its output is already a calibrated probability."""

    def __init__(self, members, C: float = 1.0):
        self.members, self.C = list(members), float(C)

    def _mat(self, raw: dict) -> np.ndarray:
        return np.column_stack([_logit(raw[m]) for m in self.members])

    def fit(self, raw: dict, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression

        A = self._mat(raw)
        self.mu, self.sd = A.mean(axis=0), A.std(axis=0) + 1e-9
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit((A - self.mu) / self.sd, np.asarray(y, int))
        return self

    def predict(self, raw: dict) -> np.ndarray:
        return self.lr.predict_proba((self._mat(raw) - self.mu) / self.sd)[:, 1]

    def to_dict(self) -> dict:
        return {"members": self.members, "C": self.C, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.lr.coef_.tolist(), "intercept": self.lr.intercept_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StackLogistic":
        m = cls(d["members"], d["C"])
        m.mu, m.sd = np.asarray(d["mu"], float), np.asarray(d["sd"], float)
        m.lr = _lr_from(d["coef"], d["intercept"])
        return m


# ----------------------------------------------------------------------------- baselines

def _harmonics(doy: np.ndarray, K: int) -> np.ndarray:
    ang = 2 * np.pi * (np.asarray(doy, float) + 0.5) / 365.25
    cols = [np.ones_like(ang)]
    for k in range(1, K + 1):
        cols += [np.sin(k * ang), np.cos(k * ang)]
    return np.stack(cols, axis=1)


def _ridge_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray, lam: float, iters: int = 50) -> np.ndarray:
    """Newton/IRLS for logit(p) = offset + X b with penalty lam·||b||² (b starts at 0)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(offset + X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        g = X.T @ (p - y) + 2 * lam * b
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 2 * lam * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        b -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


class HarmonicClimatology:
    """Site × DOY onset-probability climatology (harmonic logistic, partial pooling)."""

    def __init__(self, K: int = 2, ridge: float = 5.0):
        self.K, self.ridge = K, ridge
        self.beta_pool = None
        self.delta: dict = {}

    def fit(self, doy_pool, y_pool, site_rows: dict):
        """doy_pool/y_pool: train-role sites × train block rows; site_rows: site -> (doy, y) from
        that site's OWN training-years rows (any non-test role)."""
        Xp = _harmonics(doy_pool, self.K)
        y_pool = np.asarray(y_pool, float)
        prior = np.clip(y_pool.mean(), 1e-6, 1 - 1e-6)
        off = np.full(y_pool.size, np.log(prior / (1 - prior)))
        b = _ridge_logistic(Xp, y_pool, off, lam=1e-3)
        b[0] += off[0]
        self.beta_pool = b
        for s, (d, y) in site_rows.items():
            if len(y) < 30:
                continue
            Xs = _harmonics(d, self.K)
            self.delta[s] = _ridge_logistic(Xs, np.asarray(y, float), Xs @ self.beta_pool, lam=self.ridge)
        return self

    def predict(self, site: np.ndarray, doy: np.ndarray) -> np.ndarray:
        X = _harmonics(doy, self.K)
        eta = X @ self.beta_pool
        site = np.asarray(site)
        for s in np.unique(site):
            if s in self.delta:
                m = site == s
                eta[m] += X[m] @ self.delta[s]
        return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


class StatePersistence:
    """P(onset within L | bin of the current margin to the threshold), pooled, Laplace-smoothed."""

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def fit(self, margin: np.ndarray, y: np.ndarray):
        m = np.asarray(margin, float)
        ok = np.isfinite(m)
        m, y = m[ok], np.asarray(y, float)[ok]
        qs = np.concatenate([[0.0, 0.002, 0.005, 0.01, 0.02], np.linspace(0.05, 1.0, self.n_bins)])
        self.edges = np.unique(np.quantile(m, qs))
        b = np.clip(np.searchsorted(self.edges, m, side="right") - 1, 0, len(self.edges) - 2)
        cnt = np.bincount(b, minlength=len(self.edges) - 1).astype(float)
        pos = np.bincount(b, weights=y, minlength=len(self.edges) - 1)
        prior = y.mean() if y.size else 0.0
        self.freq = (pos + prior) / (cnt + 1.0)
        self.prior = prior
        return self

    def predict(self, margin: np.ndarray) -> np.ndarray:
        m = np.asarray(margin, float)
        b = np.clip(np.searchsorted(self.edges, m, side="right") - 1, 0, len(self.edges) - 2)
        return np.where(np.isfinite(m), self.freq[b], self.prior)


def doy_of(data: SondeData, ti: np.ndarray, lead: int = 0) -> np.ndarray:
    return (data.cube.dates[ti] + pd.to_timedelta(lead, "D")).dayofyear.to_numpy() - 1


# ----------------------------------------------------------------------------- recalibration

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration logit(p') = a + b·logit(p) (weakly regularised)."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, int)
    if y.min() == y.max():
        return 0.0, 1.0
    lr = LogisticRegression(C=100.0, max_iter=1000).fit(_logit(p)[:, None], y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def platt_apply(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    a, b = ab
    return 1 / (1 + np.exp(-(a + b * _logit(p))))


def crossfit_platt(p: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold Platt recalibration (e.g. folds = water years of the calibration block), so a
    recalibrated probability is never scored on the rows it was fitted on."""
    out = np.full(len(p), np.nan)
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        ab = platt_fit(p[tr], y[tr]) if tr.any() and np.asarray(y)[tr].sum() > 0 else (0.0, 1.0)
        out[te] = platt_apply(p[te], ab)
    return out


def water_year(dates) -> np.ndarray:
    d = pd.to_datetime(np.asarray(dates))
    return (d.year + (d.month >= 10)).to_numpy()


# ----------------------------------------------------------------------------- serialisation
# The EW 'calibrate' phase fits everything and writes these dicts; the 'score' phase only loads them.

def climatology_to_dict(c: HarmonicClimatology) -> dict:
    return {"K": c.K, "ridge": c.ridge, "beta_pool": c.beta_pool.tolist(),
            "delta": {s: d.tolist() for s, d in c.delta.items()}}


def climatology_from_dict(d: dict) -> HarmonicClimatology:
    c = HarmonicClimatology(d["K"], d["ridge"])
    c.beta_pool = np.asarray(d["beta_pool"])
    c.delta = {s: np.asarray(v) for s, v in d["delta"].items()}
    return c


def fit_site_deviation(c: HarmonicClimatology, site: str, doy: np.ndarray, y: np.ndarray) -> None:
    """Per-site deviation from the site's OWN training-years rows (pooled part frozen). Used in the
    score phase for sites not seen at calibration (e.g. test-role sites): it depends only on
    training-year labels, never on the scored block."""
    if len(y) >= 30:
        Xs = _harmonics(doy, c.K)
        c.delta[site] = _ridge_logistic(Xs, np.asarray(y, float), Xs @ c.beta_pool, lam=c.ridge)


def persistence_to_dict(p: StatePersistence) -> dict:
    return {"n_bins": p.n_bins, "edges": p.edges.tolist(), "freq": p.freq.tolist(), "prior": float(p.prior)}


def persistence_from_dict(d: dict) -> StatePersistence:
    p = StatePersistence(d["n_bins"])
    p.edges, p.freq, p.prior = np.asarray(d["edges"]), np.asarray(d["freq"]), d["prior"]
    return p


def logistic_to_dict(m: LagLogistic) -> dict:
    return {"C": m.C, "features": m.features, "mu": m.mu.tolist(), "sd": m.sd.tolist(),
            "miss_cols": m.miss_cols.tolist(), "coef": m.lr.coef_.tolist(), "intercept": m.lr.intercept_.tolist()}


def _lr_from(coef, intercept):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.coef_, lr.intercept_ = np.asarray(coef), np.asarray(intercept)
    lr.classes_ = np.array([0, 1])
    return lr


def logistic_from_dict(d: dict) -> LagLogistic:
    m = LagLogistic(d["C"], d["features"])
    m.mu, m.sd, m.miss_cols = np.asarray(d["mu"]), np.asarray(d["sd"]), np.asarray(d["miss_cols"], dtype=int)
    m.lr = _lr_from(d["coef"], d["intercept"])
    return m


def embedding_to_dict(m: EmbeddingLogistic) -> dict:
    return {"C": m.C, "mu": m.mu.tolist(), "sd": m.sd.tolist(), "coef": m.lr.coef_.tolist(),
            "intercept": m.lr.intercept_.tolist()}


def embedding_from_dict(d: dict) -> EmbeddingLogistic:
    m = EmbeddingLogistic(d["C"])
    m.mu, m.sd = np.asarray(d["mu"]), np.asarray(d["sd"])
    m.lr = _lr_from(d["coef"], d["intercept"])
    return m


# ----------------------------------------------------------------------------- placebo support

def placebo_candidates(data: SondeData, labels: HazardLabels, approved_only: bool = True) -> np.ndarray:
    """(S, T) days on which a real onset COULD have started: observed, not an event day, the
    ``pre_days`` before all observed and on the safe side of threshold ± margin, and no event day
    within the refractory window before. Placebo pseudo-onsets are drawn only from these days."""
    hz = HAZARDS[labels.hazard]
    j = data.cube.var_index(hz.variable)
    x = data.cube.y[:, :, j].astype(float)
    if approved_only:
        x = np.where(data.cube.prov[:, :, j], np.nan, x)
    safe = (x >= hz.threshold + hz.pre_margin) if hz.direction == "below" else (x <= hz.threshold - hz.pre_margin)
    safe &= np.isfinite(x)
    S, T = x.shape
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(safe, axis=1)], axis=1)
    ce = np.concatenate([np.zeros((S, 1)), np.cumsum(labels.event, axis=1)], axis=1)
    t = np.arange(T)
    lo_p = np.maximum(t - hz.pre_days, 0)
    lo_r = np.maximum(t - hz.refractory_days, 0)
    pre_ok = (cs[:, t] - cs[:, lo_p]) == hz.pre_days
    no_recent = (ce[:, t] - ce[:, lo_r]) == 0
    return np.isfinite(x) & ~labels.event & pre_ok & no_recent & (t >= hz.pre_days)


def event_runs(site: str, dates: pd.DatetimeIndex, event: np.ndarray):
    """Every run of consecutive event days (onset or not) as early_warning.Event objects, used to
    exclude ALL event periods (± horizon + exclusion) from placebo sampling."""
    from ...eval.early_warning import Event

    a = np.asarray(event, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], a, [0]]))
    out = []
    for s0, e0 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
        out.append(Event(site, pd.Timestamp(dates[s0]), pd.Timestamp(dates[e0]), pd.Timestamp(dates[s0]),
                         np.nan, 24.0 * (e0 - s0 + 1), False))
    return out



def sc_quantiles_to_chloride(q_sc_model: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Map SC quantile forecasts (model space, log(SC+1)) through each row's fixed site regression
    Cl = a + b·SC (b > 0, monotone, so quantiles map exactly). Daily-Cl exceedance at h is used as
    the proxy for the 4-day-mean / daily-max event (out-of-fold Platt recalibration absorbs it)."""
    sc = from_model_space("spcond_uscm_mean", q_sc_model)
    return np.sort(a[:, None, None] + b[:, None, None] * sc, axis=-1)
