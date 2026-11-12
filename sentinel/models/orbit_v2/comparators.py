# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Parity comparators of the architecture round v2 (orbit_arch_v2_amendment.md §2.5, §3.5; lead rulings R-10 to
R-14b). Building blocks only: nothing here reads a file, and no function is given a protected row.

* Folds: 5 folds grouped by role block within the fit cell (§3.5), fixed by the block ids alone.
* Features: ring + context summaries (§3.5) and per-context-date slots (R-13), from the dev_v5m arrays.
* GBM family: 16-configuration grid x 3 feature sets x {single target, multi-target by stacking}; each member is
  a 3-seed bag, the validation-selected member a 9-seed bag (R-14).
* MDN_B family: {7-band, 12-band, 12-band + context median + ring} x {5k, 10k, 20k iterations} x {linear, log
  input} x {single target, joint head}; 10 initialisations; two read-outs (R-14). :class:`JointMDN` is the joint
  head with the interval likelihood for left-censored values and with missing targets left out of the product.
* Location kNN: the v1 formula for any target.
* Trophic chain (R-11): OLS of the log target on out-of-fold predicted log chl-a and log Secchi.
* MDN_B + GBM average (R-10): equal weights in log10 space.
* Censoring variants of the GBM and OLS references (R-14b): DL / sqrt(2) substitution, or quantified rows only.
* Member selection: lowest validation MdSA on the target's selection cell; exact ties go to the earlier member
  of the declared order.
"""

from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np
import pandas as pd

from ..orbit import baselines as BL
from ..orbit.data import BANDS
from ..orbit.evaluate import mdsa_log10
from . import chipfeat as CF
from . import context as CX

N_FOLDS = 5                                 # §3.5
GBM_GRID = BL.GBM_GRID                      # "the existing 16-configuration grid"
GBM_MEMBER_SEEDS = (0, 1, 2)                # each grid member: a 3-seed bag
GBM_SELECTED_SEEDS = tuple(range(9))        # the validation-selected member: a 9-seed bag (3 x 3)
GBM_FEATURE_SETS = ("v1", "v1+ring+context", "v1+ring+context_dates")       # the third is R-13
GBM_MODES = ("single", "stacked")
GBM_MAX_ROUNDS, GBM_PATIENCE = 3000, 100    # as baselines.fit_gbm
MDN_N_INIT = 10
MDN_INPUTS = ("7band", "12band", "12band+context+ring")
MDN_ITERS = (5000, 10000, 20000)
MDN_XLOG = (False, True)
MDN_HEADS = ("single", "joint")
MDN_READOUTS = ("published", "candidate")   # median over inits of the top component; median of the averaged mixture
MDN_BANDS7 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")    # 443-783 nm, the Pahlevan et al. MSI input
KNN_KS = (4, 8, 16, 32)
CENSORING = ("dl_sqrt2", "quantified")      # R-14b
N_Q = 8                                     # chipfeat.date_quality
assert len(GBM_GRID) == 16


class ComparatorError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ folds

def role_block_folds(blocks, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold of each row, [N] int in 0..n_folds-1. Blocks are ordered by the sha1 of their id and dealt in turn,
    so the folds depend on the set of block ids only (no label, no row count, no random state)."""
    b = pd.Series(np.asarray(blocks)).astype(str)
    uniq = sorted(b.unique(), key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    if len(uniq) < n_folds:
        raise ComparatorError(f"{len(uniq)} role blocks for {n_folds} folds")
    return b.map({u: i % n_folds for i, u in enumerate(uniq)}).to_numpy(int)


def out_of_fold(fit_predict, folds: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """[N] out-of-fold predictions. ``fit_predict(train_index, test_index)`` returns predictions for the test
    index from a model that saw the train index only. ``rows``: boolean mask of rows that may be used for
    fitting (rows outside it are predicted but never fitted on)."""
    folds = np.asarray(folds)
    use = np.ones(len(folds), bool) if rows is None else np.asarray(rows, bool)
    out = np.full(len(folds), np.nan)
    for f in np.unique(folds):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero((folds != f) & use)
        out[te] = np.asarray(fit_predict(tr, te), float)
    return out


# --------------------------------------------------------------------------------------------- features

def slot_arrays(cidx: np.ndarray, cdt: np.ndarray, carrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rw3 [N, K, 12], q [N, K, N_Q], dt [N, K]) of each row's context slots in their stored order; NaN where a
    slot is empty."""
    cidx = np.asarray(cidx)
    if cidx.ndim != 2 or cidx.shape[1] != CX.K:
        raise ComparatorError(f"context index of shape {cidx.shape}; K = {CX.K}")
    has = cidx >= 0
    safe = np.where(has, cidx, 0)
    m = len(np.asarray(carrays["rw3"]))
    if m == 0:
        if has.any():
            raise ComparatorError("context index points into an empty context cache")
        n = len(cidx)
        return (np.full((n, CX.K, len(BANDS)), np.nan), np.full((n, CX.K, N_Q), np.nan), np.full((n, CX.K), np.nan))
    rw3 = np.where(has[..., None], np.asarray(carrays["rw3"], float)[safe], np.nan)
    q = np.where(has[..., None], np.asarray(carrays["q"], float)[safe], np.nan)
    dt = np.where(has, np.asarray(cdt, float), np.nan)
    return rw3, q, dt


def context_band_summary(rw3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(median [N, 12], IQR [N, 12]) over the context chips of a row; NaN for a row without context. A row with
    one context chip has IQR 0."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows
        med = np.nanmedian(rw3, axis=1)
        q75, q25 = np.nanpercentile(rw3, [75, 25], axis=1)
    return med, q75 - q25


def ring_context_features(feats: dict, n_pix: np.ndarray, carrays: dict) -> pd.DataFrame:
    """§3.5 "ring + context summaries": the land-ring vector, the glint features, the median and minimum d_p of
    the centre water pixels, and the per-band median and IQR over the K context chips' 3 x 3 water medians. No
    count feature."""
    ring, glint, dp = (np.asarray(feats[k], float) for k in ("ring", "glint", "dp_m"))
    n_pix = np.asarray(n_pix).astype(int)
    valid = np.arange(dp.shape[1])[None, :] < n_pix[:, None]
    dpv = np.where(valid, dp, np.nan)
    X = {f"ring_{j}": ring[:, j] for j in range(ring.shape[1])}
    X.update({f"glint_{n}": glint[:, j] for j, n in enumerate(CF.GLINT_NAMES)})
    any_pix = valid.any(1)
    X["dp_median"] = np.where(any_pix, np.nanmedian(np.where(any_pix[:, None], dpv, 0.0), axis=1), np.nan)
    X["dp_min"] = np.where(any_pix, np.nanmin(np.where(any_pix[:, None], dpv, 0.0), axis=1), np.nan)
    med, iqr = context_band_summary(slot_arrays(feats["cidx"], feats["cdt"], carrays)[0])
    for j, b in enumerate(BANDS):
        X[f"ctx_med_{b}"], X[f"ctx_iqr_{b}"] = med[:, j], iqr[:, j]
    return pd.DataFrame(X)


def context_date_features(feats: dict, carrays: dict) -> pd.DataFrame:
    """R-13: for each of the K = 6 context slots in the stored order, the 12 band 3 x 3 water medians, q_d and
    dt; NaN for an empty slot."""
    rw3, q, dt = slot_arrays(feats["cidx"], feats["cdt"], carrays)
    X = {}
    for s in range(CX.K):
        for j, b in enumerate(BANDS):
            X[f"s{s}_rw3_{b}"] = rw3[:, s, j]
        for j in range(q.shape[2]):
            X[f"s{s}_q{j}"] = q[:, s, j]
        X[f"s{s}_dt"] = dt[:, s]
    return pd.DataFrame(X)


def gbm_feature_set(name: str, v1: pd.DataFrame, feats: dict, n_pix, carrays: dict) -> pd.DataFrame:
    if name not in GBM_FEATURE_SETS:
        raise ComparatorError(f"unknown GBM feature set {name!r}")
    v1 = v1.reset_index(drop=True)
    if name == "v1":
        return v1
    rc = ring_context_features(feats, n_pix, carrays)
    if name == "v1+ring+context":
        return pd.concat([v1, rc], axis=1)
    keep = [c for c in rc.columns if not c.startswith("ctx_")]
    return pd.concat([v1, rc[keep], context_date_features(feats, carrays)], axis=1)


def stack_features(X: pd.DataFrame, others: dict, target: str) -> pd.DataFrame:
    """Multi-target by stacking: predictions of every OTHER target are added as features (out-of-fold for the
    rows the stacked model is fitted on)."""
    add = {f"stack_{t}": np.asarray(p, float) for t, p in sorted(others.items()) if t != target}
    if not add or any(len(p) != len(X) for p in add.values()):
        raise ComparatorError("stacking needs predictions of the other targets for every row")
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(add)], axis=1)


def mdn_inputs(name: str, rw3_row: np.ndarray, feats: dict, carrays: dict, fill: dict | None = None):
    """Input matrix of an MDN_B member and the fill values used. ``rw3_row`` [N, 12]: the row's own 3 x 3 water
    medians. Every missing input (a band without a water pixel in the 3 x 3 window, the context medians of a row
    without context) is replaced by the column median of the rows given at fit time, so that a member predicts
    every row (``fill`` None computes the medians: pass fit rows first, then reuse the returned ``fill`` for
    every other row)."""
    rw3_row = np.asarray(rw3_row, float)
    if name not in MDN_INPUTS:
        raise ComparatorError(f"unknown MDN_B input {name!r}")
    if name == "7band":
        X = rw3_row[:, [BANDS.index(b) for b in MDN_BANDS7]]
    elif name == "12band":
        X = rw3_row
    else:
        med, _ = context_band_summary(slot_arrays(feats["cidx"], feats["cdt"], carrays)[0])
        X = np.concatenate([rw3_row, med, np.asarray(feats["ring"], float)], axis=1)
    if fill is None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = {"median": np.nan_to_num(np.nanmedian(X, axis=0)).tolist()}
    f = np.asarray(fill["median"], float)
    if f.shape != (X.shape[1],):
        raise ComparatorError(f"fill values of another input ({f.shape[0]} columns for {X.shape[1]})")
    return np.where(np.isfinite(X), X, f[None, :]), fill


# -------------------------------------------------------------------------------------------- censoring

def censoring_variant(name: str, y_log: np.ndarray, cens: np.ndarray, dl_log: np.ndarray) -> np.ndarray:
    """R-14b fit labels [N] (NaN = row not used). ``y_log``: log10 value of quantified rows; ``cens``: True for
    a left-censored row; ``dl_log``: log10 detection limit. "dl_sqrt2": censored rows enter at DL / sqrt(2);
    "quantified": censored rows are left out."""
    y_log, cens, dl_log = np.asarray(y_log, float), np.asarray(cens, bool), np.asarray(dl_log, float)
    if name == "quantified":
        return np.where(cens, np.nan, y_log)
    if name == "dl_sqrt2":
        return np.where(cens, dl_log - 0.5 * math.log10(2.0), y_log)
    raise ComparatorError(f"unknown censoring variant {name!r}")


# -------------------------------------------------------------------------------------------------- kNN

def fit_knn(lat, lon, y_log):
    """Location kNN of v1 (baselines.fit_location_knn) for one target: point-level mean of log10 values,
    inverse-distance weights 1 / (d_km + 0.1) on the sphere. Returns ``predict(lat, lon, k)``."""
    from sklearn.neighbors import BallTree

    y_log = np.asarray(y_log, float)
    m = np.isfinite(y_log)
    pts = pd.DataFrame({"lat": np.asarray(lat, float)[m], "lon": np.asarray(lon, float)[m], "y": y_log[m]})
    pts = pts.groupby(["lat", "lon"], as_index=False).y.mean()
    if not len(pts):
        raise ComparatorError("kNN: no fit value")
    tree = BallTree(np.radians(pts[["lat", "lon"]].to_numpy()), metric="haversine")
    yv = pts.y.to_numpy()

    def predict(lat, lon, k):
        d, i = tree.query(np.radians(np.stack([np.asarray(lat, float), np.asarray(lon, float)], 1)), k=min(k, len(yv)))
        w = 1.0 / (d * 6371.0 + 0.1)
        return (w * yv[i]).sum(1) / w.sum(1)
    return predict


# -------------------------------------------------------------------------------------------------- GBM

def gbm_members() -> list[tuple[str, str, int]]:
    """(feature set, mode, grid index) of every GBM family member, in the declared order."""
    return [(fs, mode, j) for fs in GBM_FEATURE_SETS for mode in GBM_MODES for j in range(len(GBM_GRID))]


def member_name(fs: str, mode: str, j: int, censoring: str | None = None) -> str:
    return f"gbm[{fs}|{mode}|g{j:02d}" + (f"|{censoring}]" if censoring else "]")


def fit_gbm_bag(X_fit, y_fit, X_inner, y_inner, params: dict, seeds=GBM_MEMBER_SEEDS, *, n_threads: int = BL.N_THREADS,
                max_rounds: int = GBM_MAX_ROUNDS, patience: int = GBM_PATIENCE):
    """A seed bag of one grid configuration. Every seed is stopped early on the inner rows (as v1's grid fit).
    Returns ``predict(X)`` = the mean over the bag in log10 space; ``predict.n_rounds`` lists the rounds."""
    import lightgbm as lgb

    mf, mi = np.isfinite(y_fit), np.isfinite(y_inner)
    if mf.sum() < 50 or mi.sum() < 10:                     # as baselines.fit_gbm
        raise ComparatorError(f"GBM: n_fit={int(mf.sum())} n_inner={int(mi.sum())}")
    Xf, Xi = np.asarray(X_fit, float)[mf], np.asarray(X_inner, float)[mi]
    bag = []
    for s in seeds:
        dtr = lgb.Dataset(Xf, np.asarray(y_fit, float)[mf], free_raw_data=False)
        dva = lgb.Dataset(Xi, np.asarray(y_inner, float)[mi], reference=dtr)
        p = dict(params, verbose=-1, num_threads=n_threads, seed=int(s))
        bag.append(lgb.train(p, dtr, max_rounds, valid_sets=[dva],
                             callbacks=[lgb.early_stopping(patience, verbose=False)]))

    def predict(X):
        X = np.asarray(X, float)
        return np.mean([b.predict(X, num_iteration=b.best_iteration) for b in bag], 0)
    predict.n_rounds = [int(b.best_iteration) for b in bag]
    predict.seeds = [int(s) for s in seeds]
    return predict


# ---------------------------------------------------------------------------- average, chain, selection

def average_log10(mdn_log: np.ndarray, gbm_log: np.ndarray) -> np.ndarray:
    """R-10: equal-weight average, in log10 space, of the validation-selected MDN_B and GBM members."""
    a, b = np.asarray(mdn_log, float), np.asarray(gbm_log, float)
    if a.shape != b.shape:
        raise ComparatorError("average: the two members predict different rows")
    return 0.5 * (a + b)


def ols_fit(z_chla, z_secchi, y_log) -> np.ndarray:
    """Trophic chain (§2.5 item 5): least squares of the log target on (1, log chl-a, log Secchi). Rows with a
    non-finite entry are left out. Returns the 3 coefficients."""
    A = np.stack([np.ones(len(y_log)), np.asarray(z_chla, float), np.asarray(z_secchi, float)], 1)
    y = np.asarray(y_log, float)
    m = np.isfinite(y) & np.all(np.isfinite(A), 1)
    if m.sum() < 10:
        raise ComparatorError(f"trophic chain: {int(m.sum())} usable fit rows")
    return np.linalg.lstsq(A[m], y[m], rcond=None)[0]


def ols_predict(coef, z_chla, z_secchi) -> np.ndarray:
    return coef[0] + coef[1] * np.asarray(z_chla, float) + coef[2] * np.asarray(z_secchi, float)


def select_member(scores: dict, order=None) -> str:
    """Member with the lowest validation MdSA on the target's selection cell. Members without a finite score
    cannot be selected; an exact tie goes to the earlier member of ``order`` (default: the dict order)."""
    order = list(scores) if order is None else list(order)
    if set(order) != set(scores):
        raise ComparatorError("selection order and scored members differ")
    ok = [m for m in order if scores[m] is not None and np.isfinite(scores[m])]
    if not ok:
        raise ComparatorError("no family member has a validation score")
    return min(ok, key=lambda m: (float(scores[m]), order.index(m)))


def val_mdsa(pred_log, y_log, cell_mask) -> float:
    """Validation MdSA of a member on a selection cell: quantified rows of the cell only; NaN if any of them has
    no prediction (a member must predict every row of the cell to be scored)."""
    m = np.asarray(cell_mask, bool) & np.isfinite(np.asarray(y_log, float))
    p = np.asarray(pred_log, float)[m]
    if not m.any() or not np.all(np.isfinite(p)):
        return float("nan")
    return mdsa_log10(p - np.asarray(y_log, float)[m])


# ------------------------------------------------------------------------------------------------ MDN_B

def mdn_members() -> list[tuple[str, int, bool, str]]:
    """(input, iterations, log input, head) of every trained MDN_B member; each gives two family members, one per
    read-out of ``MDN_READOUTS``."""
    return list(itertools.product(MDN_INPUTS, MDN_ITERS, MDN_XLOG, MDN_HEADS))


def mdn_name(inp: str, n_iter: int, x_log: bool, head: str, readout: str) -> str:
    return f"mdn_b[{inp}|{n_iter // 1000}k|{'log' if x_log else 'lin'}|{head}|{readout}]"


def joint_nll(logw, mu, sigma, y, state):
    """Negative log-likelihood per row of a mixture of K diagonal Gaussians over T targets.

    ``logw`` [..., K]; ``mu``, ``sigma`` [..., K, T]; ``y`` [..., T]; ``state`` [..., T]: 0 = no value (the
    target is left out of the product), 1 = quantified (Gaussian density at y), 2 = left-censored (the interval
    likelihood P(Y <= y), y being the detection limit). With T = 1 and state 1 this is ``mdn.mixture_nll``."""
    import torch

    z = (y[..., None, :] - mu) / sigma
    dens = -0.5 * z ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    cdf = torch.special.log_ndtr(z)
    st = state[..., None, :]
    lp = torch.where(st == 1, dens, torch.where(st == 2, cdf, torch.zeros_like(dens))).sum(-1)
    return -torch.logsumexp(logw + lp, -1)


class JointMDN:
    """MDN_B with a joint multi-target head: v1's network body (5 x 100 ReLU, 10 initialisations trained side by
    side, Adam 1e-3, L2 1e-3, batch 128, RobustScaler inputs, targets MinMax-scaled to (-1, 1) on quantified fit
    values) with K = 5 components, each a diagonal Gaussian over the T targets. T = 1 is the single-target
    member. Rows enter the fit if every input is finite and at least one target has a value or a limit."""

    def __init__(self, n_targets: int, n_ens: int = MDN_N_INIT, n_iter: int = 10000, batch: int = 128, lr: float = 1e-3,
                 l2: float = 1e-3, seed: int = 0, device: str = "cpu", x_log: bool = False, k: int = 5):
        self.t, self.n_ens, self.n_iter, self.batch, self.lr, self.l2 = n_targets, n_ens, n_iter, batch, lr, l2
        self.seed, self.device, self.x_log, self.k = seed, device, x_log, k

    def _x(self, X):
        from ..orbit.mdn import X_FLOOR

        X = np.asarray(X, float)
        return np.log10(np.clip(X, X_FLOOR, None)) if self.x_log else X

    def _net(self, n_in):
        import torch.nn as nn

        from ..orbit.mdn import _EnsLinear

        layers, d = [], n_in
        for _ in range(5):
            layers.append(_EnsLinear(self.n_ens, d, 100))
            d = 100
        return nn.ModuleList(layers), _EnsLinear(self.n_ens, d, self.k * (1 + 2 * self.t))

    def _forward(self, x):
        import torch

        h = x
        for layer in self.hidden:
            h = torch.relu(layer(h))
        o = self.out(h)
        k, t = self.k, self.t
        logw = torch.log_softmax(o[..., :k], -1)
        mu = o[..., k:k + k * t].reshape(*o.shape[:-1], k, t)
        sigma = torch.nn.functional.softplus(o[..., k + k * t:].reshape(*o.shape[:-1], k, t)) + 1e-3
        return logw, mu, sigma

    def fit(self, X, y_log, state):
        """``y_log`` [N, T]: log10 value (state 1) or log10 detection limit (state 2); ``state`` [N, T]."""
        import torch
        from sklearn.preprocessing import RobustScaler

        y_log, state = np.asarray(y_log, float).reshape(len(X), self.t), np.asarray(state).reshape(len(X), self.t)
        state = np.where(np.isfinite(y_log), state, 0).astype(np.int64)
        m = np.all(np.isfinite(np.asarray(X, float)), 1) & (state > 0).any(1)
        X, y, st = self._x(np.asarray(X, float)[m]), y_log[m], state[m]
        if not (st == 1).any(0).all():
            raise ComparatorError("MDN_B: a target has no quantified fit value")
        self.xs = RobustScaler().fit(X)
        q = np.where(st == 1, y, np.nan)
        self.ylo, self.yhi = np.nanmin(q, 0), np.nanmax(q, 0)
        xs = torch.tensor(self.xs.transform(X), dtype=torch.float32, device=self.device)
        ys = torch.tensor(np.nan_to_num(2 * (y - self.ylo) / (self.yhi - self.ylo + 1e-12) - 1), dtype=torch.float32,
                          device=self.device)
        sts = torch.tensor(st, device=self.device)
        torch.manual_seed(self.seed)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.hidden, self.out = self._net(X.shape[1])
        self.hidden.to(self.device), self.out.to(self.device)
        params = list(self.hidden.parameters()) + list(self.out.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=2 * self.l2)       # as v1 (Keras L2 = 2 * l2 decay)
        n = xs.shape[0]
        for _ in range(self.n_iter):
            idx = torch.randint(0, n, (self.n_ens, self.batch), generator=g).to(self.device)
            logw, mu, sg = self._forward(xs[idx])
            loss = joint_nll(logw, mu, sg, ys[idx], sts[idx]).mean(1).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.n_fit = int(n)
        return self

    def mixture(self, X):
        """Per-initialisation mixtures in log10 units: logw [E, N, K], mu and sigma [E, N, K, T]; NaN rows where
        an input is not finite."""
        import torch

        X = np.asarray(X, float)
        ok = np.all(np.isfinite(X), 1)
        xs = np.zeros((len(X), X.shape[1]))
        if ok.any():
            xs[ok] = self.xs.transform(self._x(X[ok]))
        with torch.no_grad():
            t = torch.tensor(xs, dtype=torch.float32, device=self.device)[None].expand(self.n_ens, -1, -1)
            logw, mu, sg = (a.cpu().numpy().astype(float) for a in self._forward(t.contiguous()))
        scale = (self.yhi - self.ylo) / 2
        mu = (mu + 1) * scale + self.ylo
        sg = sg * scale
        mu[:, ~ok] = np.nan
        return logw, mu, sg

    def predict(self, X, readout: str = "published") -> np.ndarray:
        """[N, T] log10 point predictions. "published": median over initialisations of the mean of the component
        with the largest mixing weight. "candidate": median of the mixture averaged over initialisations."""
        logw, mu, sg = self.mixture(X)
        if readout == "published":
            k = np.argmax(logw, -1)
            est = np.take_along_axis(mu, k[..., None, None], 2)[:, :, 0, :]             # [E, N, T]
            return np.median(est, 0)
        if readout != "candidate":
            raise ComparatorError(f"unknown read-out {readout!r}")
        from ..orbit.satwq import mixture_quantiles

        E, N, K, T = mu.shape
        w = np.exp(logw).transpose(1, 0, 2).reshape(N, E * K) / E
        out = np.full((N, T), np.nan)
        ok = np.all(np.isfinite(mu.reshape(E, N, K * T)), axis=(0, 2))
        for t in range(T):
            m_, s_ = (a[..., t].transpose(1, 0, 2).reshape(N, E * K) for a in (mu, sg))
            if ok.any():
                out[ok, t] = np.asarray(mixture_quantiles(w[ok], m_[ok], s_[ok], (0.5,))).reshape(-1)
        return out
