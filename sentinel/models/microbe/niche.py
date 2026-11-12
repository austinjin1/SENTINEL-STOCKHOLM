# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""NicheInvert-S: inverse niche calibration on exact V4 sequence units (amendment microbe_arch_v2 §2, arm D1).

Every sequence unit k gets a Gaussian niche on a D-dimensional standardised chemistry vector z (optimum ``mu_kd``,
tolerance ``t_kd``, intercept ``a_k``). A held-out event is inverted: z maximises the multinomial likelihood of its
reads under ``softmax_k(a_k - 0.5 * sum_d (z_d - mu_kd)^2 / t_kd^2)`` plus a Gaussian season/location prior. The
method is predictive only; no niche parameter supports a claim about a taxon.

All constants are those of amendment §2.1 (``DEFAULTS``). **Implementation readings** — points the amendment leaves
open, fixed here before any development run and repeated in ``docs/preregistration/microbe_arch_v2_log.md``:

R1  Relative abundance ``p_jk`` is the read share among the event's in-vocabulary reads (the likelihood's N_j).
R2  ``N2`` is computed per dimension over the training events that have that dimension (identical for every
    dimension without missing labels). A unit with no labelled training event on a dimension has lambda = 0: its
    optimum is the prior and its tolerance the pooled one. The (1 - 1/N2) correction applies when N2 > 1 + 1e-6.
R3  PCA: eigendecomposition of the 1,024 x 1,024 covariance of the column-standardised 5-mer profiles of the fold
    vocabulary (each unit one row, unweighted; float64; sign of each component fixed so that its largest absolute
    loading is positive). No randomised solver, so no seed. Scores are not whitened.
R4  Sequence-prior ridge: weighted least squares with weights N2_kd, weighted centring (unpenalised intercept),
    penalty alpha on the 64 coefficients, alpha by classical GCV  n * RSS_w / (n - df)^2,  df = trace of the hat
    matrix including the intercept, n = number of units with a labelled event.
R5  Pooled tolerance: N2-weighted median over units with N2 >= 5 of the corrected tolerance; 1.0 when no unit
    qualifies (synthetic tests only).
R6  Intercept: a_k = log sum_j p_jk - log sum_j exp(q0_k(z_j)) over training events with in-vocabulary reads,
    q0_k(z) = -0.5 * sum_d w_kd (z_d - mu_kd)^2, i.e. the unnormalised response matches the mean abundance.
R7  Context prior: one ridge per dimension (``baselines._ridge_grouped``: standardised inputs, alpha by
    leave-sites-out inner CV, <= 5 folds). Sigma: pairwise-complete covariance (>= 3 pairs, else 0) of the in-sample
    training residuals, 0.8 * S + 0.2 * diag(S), eigenvalues floored at 1e-3.
R8  Newton: the negative Hessian is  N_eff * [diag(P w - E_pi w) + Cov_pi(g)] + Sigma^-1  with
    g_k = w_k * (mu_k - z). When it is not positive definite the Gauss–Newton form  N_eff * Cov_pi(g) + Sigma^-1
    is used for that event (also for the Laplace SD). Cov_pi(g) is summed over units with pi_k > 1e-9 / K
    (excluded mass < 1e-9); L and its gradient are exact. A step is kept only if L strictly increases; trial
    step lengths 1, 1/2, ..., 1/32 (5 halvings). An event whose step is rejected stops.
R9  N_eff: inner folds are ``GroupKFold(3)`` over the training domains. Each inner model is refitted from scratch
    (vocabulary, standardisation, PCA, priors). The criterion is the mean over the scored dimensions of the R²
    of the *deshrunk* pooled inner out-of-fold inferences (the model's output); ties go to the smaller N_eff.
R10 Deshrinking: per dimension an OLS line (label on inner out-of-fold inference, target scale), slope clipped to
    [0.5, 3] and the intercept recomputed after clipping. The Laplace SD is multiplied by the slope.
R11 A dimension with fewer than 10 finite training labels is not fitted (prediction NaN), as for the baselines.
R12 Open vocabulary (EA variant only): the softmax of an event runs over the fold vocabulary plus that event's
    own unseen units. Inside NEON development runs unseen reads are dropped and their share is reported.
R13 Ablation A1 (taxonomy prior): the prior optimum of a unit is sum(S1) / sum(S0) over the vocabulary units of
    its RDP genus (majority label by reads; the family when the genus is unassigned; the vocabulary-wide value
    when both are).

Readings added after the stamp (lead rulings file §2.2 A1 and §2.1 B8; proposed in
``docs/preregistration/microbe_arch_v2_deviations_01.md``, which must be stamped before the affected run):

R14 ``niche_s_qmp`` (the +QMP twin, G10 only): the 4 QMP features (training-median filled) and their 4 missing
    flags are appended to the tier-0 context columns of the context-prior ridge. Nothing else changes.
R15 Ablation A0 ``niche_s_a0`` (no context prior, kappa = 0, no deshrinking clip):
    (a) flat prior on z: the context term is dropped, the search starts at the training mean (z = 0), and
        Sigma^-1 is replaced by the numerical ridge ``flat_prior_ridge`` * I (1e-6), which only keeps the Newton
        system solvable on dead or unidentified directions;
    (b) kappa = 0: lambda = 1 for every unit with a labelled event (optimum = its weighted average, tolerance =
        its own corrected tolerance, floored as in D1). A unit with N2 <= 1 has no tolerance estimate and takes
        the pooled tolerance; a unit with no labelled event keeps reading R2;
    (c) the deshrinking line is fitted as in R10 but its slope is not clipped;
    (d) with a flat prior the posterior mode does not depend on N_eff, so the grid is the single value 100 (it
        only scales the reported Laplace SD).

No stochastic step: results do not depend on ``seed``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from sentinel.data.microbe_geo import GEO_TARGETS
from sentinel.models.microbe import baselines as B
from sentinel.models.microbe import features as F
from sentinel.models.microbe.design import TARGET_NAMES
from sentinel.models.microbe.dna import V4_LEN, kmer_profile, v4_window

AUXILIARY = ("field_temp_c",)
MASTER = tuple(TARGET_NAMES) + tuple(GEO_TARGETS) + AUXILIARY       # D = 17
SCORED = tuple(TARGET_NAMES) + tuple(GEO_TARGETS)                   # the 16 scored dimensions

DEFAULTS = {
    "min_events": 2,                                   # vocabulary: units present in >= 2 training events
    "kmer": 5, "n_pc": 64,
    "prior_alphas": (0.1, 1.0, 10.0, 100.0, 1000.0),
    "kappa": 5.0,
    "tol_pool_min_n2": 5.0,
    "tol_floor": 0.25,                                 # standardised units
    "context_alphas": tuple(B.DEFAULTS["ridge_alphas"]),
    "context_cols": None,                              # None = all tier-0 columns; (0, 1) = DOY harmonics only
    "context_prior": True,                             # False = flat prior on z (ablation A0, reading R15)
    "flat_prior_ridge": 1e-6,
    "qmp": False,                                      # True = QMP block in the context prior (reading R14)
    "cov_shrink": 0.2, "eig_floor": 1e-3,
    "n_eff_grid": (10.0, 30.0, 100.0, 300.0, 1000.0),
    "n_eff_default": 100.0,                            # only when no inner CV is possible (< 2 training domains)
    "inner_folds": 3,
    "newton_steps": 6, "halvings": 5,
    "slope_clip": (0.5, 3.0),                          # None = no clip (ablation A0)
    "prior": "sequence",                               # "taxonomy" = ablation A1
    "min_rows": 10,
    "hess_tol": 1e-9, "chunk": 128,
}
VARIANTS = {
    "niche_s": {},                                     # the claim configuration
    "niche_s_tax": {"prior": "taxonomy"},              # ablation A1 (lodo only, never selectable)
    "niche_s_doy": {"context_cols": (0, 1)},           # EA variant: day-of-year-only prior
    "niche_s_a0": {"context_prior": False, "kappa": 0.0, "slope_clip": None,       # ablation A0 (lead ruling A1;
                   "n_eff_grid": (100.0,)},                                         # lodo only, never selectable)
    "niche_s_qmp": {"qmp": True},                      # +QMP twin (GEO entry G10 only)
}
NEVER_SELECTABLE = ("niche_s_tax", "niche_s_a0", "xgb_nichez_context")
CONTROL = "xgb_nichez_context"                         # attribution control (§2.4), never selectable
NAMES = tuple(VARIANTS) + (CONTROL,)
LABELS = {"niche_s": "NicheInvert-S", "niche_s_tax": "NicheInvert-S[taxonomy prior]",
          "niche_s_doy": "NicheInvert-S[DOY prior]",
          "niche_s_a0": "NicheInvert-S[A0: flat prior, kappa=0, no clip]", "niche_s_qmp": "NicheInvert-S+QMP",
          CONTROL: "XGB[NicheInvert-z + SD + DOY+location]"}


# ----------------------------------------------------------------------------------------------- unit table

@dataclass
class UnitTable:
    """Label-free event x unit read counts of a design (units = harmonised 230-bp V4 windows)."""

    events: pd.Index                 # sample ids, in ``design.samples`` order
    units: np.ndarray                # unit keys ('v4:' + sha1(window)[:16]), sorted
    counts: sparse.csr_matrix        # [events x units] reads
    windows: np.ndarray              # the window sequence of every unit
    genus: np.ndarray                # majority RDP genus by reads ('' = unassigned); ablation A1 only
    family: np.ndarray
    n_reads_all: np.ndarray          # all reads of the event, with or without a V4 window
    _profile: np.ndarray | None = field(default=None, repr=False)

    def rows(self, ids) -> np.ndarray:
        r = self.events.get_indexer(pd.Index(np.asarray(ids)))
        if (r < 0).any():
            raise KeyError(f"{int((r < 0).sum())} events are not in the unit table")
        return r

    def profile(self, k: int = 5) -> np.ndarray:
        """[units x 4**k] float32 k-mer frequencies of the windows (label-free, computed once)."""
        if self._profile is None or self._profile.shape[1] != 4 ** k:
            self._profile = kmer_profile(list(self.windows), k)
        return self._profile


def _majority(unit: np.ndarray, label: pd.Series, count: np.ndarray, n_units: int) -> np.ndarray:
    lab = label.fillna("").astype(str).replace({F.UNASSIGNED: ""}).to_numpy()
    g = pd.DataFrame({"u": unit, "l": lab, "c": count}).groupby(["u", "l"], sort=True)["c"].sum().reset_index()
    g = g.sort_values(["u", "c", "l"], ascending=[True, False, True], kind="stable").drop_duplicates("u")
    out = np.full(n_units, "", object)
    out[g["u"].to_numpy()] = g["l"].to_numpy()
    return out


def build_unit_table(design, L: int = V4_LEN) -> UnitTable:
    """Mirrors ``dna.add_v4_units`` (primer of the first event of each sequence; key = sha1 of the window)."""
    tok, S = design.tokens, design.samples
    events = pd.Index(S.index)
    prim = S["primer_set"].astype(str).reindex(tok["sample_id"]).to_numpy()
    sp = pd.DataFrame({"seq_id": tok["seq_id"].to_numpy(), "primer": prim}).drop_duplicates("seq_id")
    seq = design.sequences.drop_duplicates("seq_id").set_index("seq_id")["sequence"].reindex(sp["seq_id"]).to_numpy()
    win = [v4_window(s_, p_, L) for s_, p_ in zip(seq, sp["primer"].to_numpy())]
    key = np.array([("v4:" + hashlib.sha1(w.encode()).hexdigest()[:16]) if w else "" for w in win], object)
    has = key != ""
    units, first, inv = np.unique(key[has].astype(str), return_index=True, return_inverse=True)
    windows = np.array([w for w in win if w], object)[first]
    seq_unit = pd.Series(-1, index=sp["seq_id"].to_numpy(), dtype=np.int64)
    seq_unit.iloc[np.flatnonzero(has)] = inv
    u = seq_unit.reindex(tok["seq_id"]).to_numpy()
    r = events.get_indexer(tok["sample_id"])
    c = tok["count"].to_numpy(float)
    in_design = r >= 0
    n_all = np.bincount(r[in_design], c[in_design], minlength=len(events))
    ok = in_design & (u >= 0)
    counts = sparse.csr_matrix((c[ok], (r[ok], u[ok])), shape=(len(events), len(units)))
    counts.sum_duplicates()
    tax = {}
    for rank in ("genus", "family"):
        tax[rank] = (_majority(u[ok], tok[rank][ok], c[ok], len(units)) if rank in tok
                     else np.full(len(units), "", object))
    return UnitTable(events, units, counts, windows, tax["genus"], tax["family"], n_all)


_UT_CACHE: dict = {}


def unit_table(design) -> UnitTable:
    """Cached per token table (a design copy with other labels shares it: the table is label-free)."""
    key = (id(design.tokens), len(design.tokens), id(design.sequences), tuple(design.samples.index[:3]),
           len(design.samples))
    if key not in _UT_CACHE:
        _UT_CACHE.clear()
        _UT_CACHE[key] = build_unit_table(design)
    return _UT_CACHE[key]


# ------------------------------------------------------------------------------------------------ numerics

def niche_stats(P: sparse.csr_matrix, Z: np.ndarray) -> dict:
    """Weighted-averaging optimum, tolerance and N2 of every unit (columns of ``P``) on every dimension.

    ``P`` [events x units]: relative abundances; ``Z`` [events x D]: values, NaN = the event lacks the dimension.
    Returns ``mu`` (NaN where the unit has no labelled event), ``var`` (N2-corrected tolerance²), ``n2``, ``s0``
    (sum of p over labelled events) and ``s1`` (sum of p * z)."""
    M = np.isfinite(Z).astype(float)
    Z0 = np.where(M > 0, Z, 0.0)
    PT = P.T.tocsr()
    s0, s1, s2 = PT @ M, PT @ Z0, PT @ (Z0 * Z0)
    P2 = PT.copy()
    P2.data = P2.data ** 2
    q = P2 @ M
    lab = s0 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(lab, s1 / s0, np.nan)
        var = np.where(lab, np.maximum(s2 / s0 - mu * mu, 0.0), 0.0)
        n2 = np.where(lab, s0 * s0 / q, 0.0)
    corr = n2 > 1.0 + 1e-6
    var[corr] = var[corr] / (1.0 - 1.0 / n2[corr])
    return {"mu": mu, "var": var, "n2": n2, "s0": s0, "s1": s1}


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    """Smallest value whose cumulative weight reaches half of the total (lower weighted median)."""
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


@dataclass
class SeqBasis:
    mean: np.ndarray
    scale: np.ndarray
    comp: np.ndarray                 # [4**k x n_pc]
    explained: np.ndarray

    def transform(self, prof: np.ndarray) -> np.ndarray:
        out = np.empty((prof.shape[0], self.comp.shape[1]))
        for s in range(0, prof.shape[0], 16384):
            out[s:s + 16384] = ((prof[s:s + 16384].astype(np.float64) - self.mean) / self.scale) @ self.comp
        return out


def fit_seq_basis(prof: np.ndarray, n_pc: int) -> SeqBasis:
    """PCA of the column-standardised profiles (reading R3). Deterministic; uses no label."""
    n, d = prof.shape
    mean = prof.mean(0, dtype=np.float64)
    scale = prof.std(0, dtype=np.float64)
    scale[scale <= 0] = 1.0
    G = np.zeros((d, d))
    for s in range(0, n, 16384):
        X = (prof[s:s + 16384].astype(np.float64) - mean) / scale
        G += X.T @ X
    G /= max(n - 1, 1)
    ev, V = np.linalg.eigh(G)
    order = np.argsort(-ev, kind="stable")
    ev, V = ev[order], V[:, order]
    keep = min(int(n_pc), max(n - 1, 1), int((ev > 1e-10 * max(ev[0], 1e-300)).sum()))
    V = V[:, :max(keep, 1)]
    big = np.abs(V).argmax(0)
    V = V * np.sign(V[big, np.arange(V.shape[1])])
    return SeqBasis(mean, scale, V, ev[:V.shape[1]])


def ridge_gcv(X: np.ndarray, y: np.ndarray, w: np.ndarray, alphas) -> dict:
    """Weighted ridge with an unpenalised intercept; alpha by generalised cross-validation (reading R4)."""
    n = X.shape[0]
    sw = w.sum()
    xm, ym = (w[:, None] * X).sum(0) / sw, float((w * y).sum() / sw)
    rw = np.sqrt(w)
    Xc, yc = (X - xm) * rw[:, None], (y - ym) * rw
    e, V = np.linalg.eigh(Xc.T @ Xc)
    e = np.maximum(e, 0.0)
    vb, yy = V.T @ (Xc.T @ yc), float(yc @ yc)
    best, gcvs = None, []
    for a in alphas:
        c = vb / (e + a)
        rss = max(yy - 2.0 * float(c @ vb) + float((c * c * e).sum()), 0.0)
        df = float((e / (e + a)).sum()) + 1.0
        g = n * rss / (n - df) ** 2 if n - df > 1e-9 else np.inf
        gcvs.append(g)
        if best is None or g < best[0]:
            best = (g, float(a), c)
    coef = V @ best[2]
    return {"coef": coef, "intercept": ym - float(xm @ coef), "alpha": best[1], "gcv": gcvs}


def taxonomy_prior(s0: np.ndarray, s1: np.ndarray, genus: np.ndarray, family: np.ndarray) -> np.ndarray:
    """Ablation A1 (reading R13): [units x D] prior optimum from the unit's genus, else family, else all units."""
    def level(lab):
        codes, inv = np.unique(lab.astype(str), return_inverse=True)
        a0, a1 = np.zeros((len(codes), s0.shape[1])), np.zeros((len(codes), s0.shape[1]))
        np.add.at(a0, inv, s0)
        np.add.at(a1, inv, s1)
        with np.errstate(divide="ignore", invalid="ignore"):
            opt = np.where(a0 > 0, a1 / a0, np.nan)
        opt[codes == ""] = np.nan
        return opt[inv]

    out = level(genus)
    fam = level(family)
    out = np.where(np.isfinite(out), out, fam)
    with np.errstate(divide="ignore", invalid="ignore"):
        glob = np.where(s0.sum(0) > 0, s1.sum(0) / s0.sum(0), 0.0)
    return np.where(np.isfinite(out), out, glob[None, :])


def _row_normalise(C: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    C = sparse.csr_matrix(C, dtype=np.float64)
    n = np.asarray(C.sum(1)).ravel()
    inv = np.divide(1.0, n, out=np.zeros_like(n), where=n > 0)
    return sparse.diags(inv) @ C, n


# --------------------------------------------------------------------------------------------------- model

@dataclass
class NicheModel:
    vocab: np.ndarray                # column indices of the unit table
    mu: np.ndarray                   # [K x D] shrunk optima (standardised units)
    w: np.ndarray                    # [K x D] precisions 1 / t²
    a: np.ndarray                    # [K] intercepts
    y_mean: np.ndarray
    y_sd: np.ndarray
    alive: np.ndarray                # [D] dimensions with >= min_rows training labels
    ctx_cols: np.ndarray
    ctx_fs: list                     # per-dimension predictor of the standardised value (None = 0)
    sigma: np.ndarray
    sigma_inv: np.ndarray
    basis: SeqBasis | None
    prior_coef: np.ndarray | None    # [n_pc x D]
    prior_icpt: np.ndarray | None    # [D]
    t_pool: np.ndarray               # [D] pooled tolerance²
    a_rare: float
    diag: dict

    def prior_mean(self, ctx: np.ndarray) -> np.ndarray:
        X = np.asarray(ctx, float)[:, self.ctx_cols]
        return np.column_stack([np.zeros(len(X)) if f is None else f(X) for f in self.ctx_fs])

    def to_target(self, z: np.ndarray) -> np.ndarray:
        y = z * self.y_sd + self.y_mean
        y[:, ~self.alive] = np.nan
        return y

    def open_units(self, prof: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parameters of units absent from the vocabulary (reading R12): sequence-prior optimum, pooled
        tolerance, median intercept of the rare vocabulary units."""
        if self.basis is None:
            raise ValueError("the open vocabulary needs the sequence prior")
        mu = self.basis.transform(prof) @ self.prior_coef + self.prior_icpt
        w = np.broadcast_to(1.0 / np.maximum(self.t_pool, p["tol_floor"] ** 2), mu.shape).copy()
        return mu, w, np.full(len(mu), self.a_rare)


def _standardise(Y: np.ndarray, min_rows: int):
    n = np.isfinite(Y).sum(0)
    alive = n >= min_rows
    mean, sd = np.zeros(Y.shape[1]), np.ones(Y.shape[1])
    for d in np.flatnonzero(alive):
        v = Y[np.isfinite(Y[:, d]), d]
        s = float(v.std())
        if s > 0:
            mean[d], sd[d] = float(v.mean()), s
        else:
            alive[d] = False
    Z = (Y - mean) / sd
    Z[:, ~alive] = np.nan
    return Z, mean, sd, alive


def _residual_cov(R: np.ndarray, shrink: float, floor: float) -> tuple[np.ndarray, np.ndarray]:
    S = pd.DataFrame(R).cov(min_periods=3).to_numpy()
    d = np.diag(S).copy()
    d[~np.isfinite(d)] = 1.0
    S = np.where(np.isfinite(S), S, 0.0)
    np.fill_diagonal(S, d)
    S = (1.0 - shrink) * S + shrink * np.diag(d)
    ev, V = np.linalg.eigh((S + S.T) / 2.0)
    ev = np.maximum(ev, floor)
    return (V * ev) @ V.T, (V / ev) @ V.T


def fit_niche(C_tr, Y_tr: np.ndarray, ctx_tr: np.ndarray, groups_tr: np.ndarray, prof: np.ndarray | None,
              params: dict | None = None, taxa: tuple | None = None) -> NicheModel:
    """Fit the niche model on training events only.

    ``C_tr`` [n x U] read counts over all units of the table; ``Y_tr`` [n x D] labels on the target scale (NaN =
    missing); ``ctx_tr`` [n x F] tier-0 context; ``groups_tr`` site ids (context-ridge inner CV); ``prof`` [U x
    4**k] k-mer profiles; ``taxa`` = (genus, family) labels per unit for the taxonomy prior."""
    p = {**DEFAULTS, **(params or {})}
    C_tr = sparse.csr_matrix(C_tr, dtype=np.float64)
    Y_tr = np.asarray(Y_tr, float)
    n, D = Y_tr.shape
    prevalence = np.asarray((C_tr > 0).sum(0)).ravel()
    vocab = np.flatnonzero(prevalence >= p["min_events"])
    if vocab.size == 0:
        raise ValueError("no unit is present in enough training events")
    P, N = _row_normalise(C_tr[:, vocab])
    has = N > 0
    Z, y_mean, y_sd, alive = _standardise(Y_tr, p["min_rows"])
    st = niche_stats(P, Z)
    mu_hat, var_hat, n2, s0 = st["mu"], st["var"], st["n2"], st["s0"]
    lab = s0 > 0

    # ---- prior optimum s_d(x_k)
    basis = coef = icpt = None
    alphas = [None] * D
    if p["prior"] == "sequence":
        if prof is None:
            raise ValueError("the sequence prior needs k-mer profiles")
        basis = fit_seq_basis(prof[vocab], p["n_pc"])
        X = basis.transform(prof[vocab])
        coef, icpt = np.zeros((X.shape[1], D)), np.zeros(D)
        for d in np.flatnonzero(alive):
            k = np.flatnonzero(lab[:, d])
            if k.size >= 3:
                r = ridge_gcv(X[k], mu_hat[k, d], n2[k, d], p["prior_alphas"])
                coef[:, d], icpt[d], alphas[d] = r["coef"], r["intercept"], r["alpha"]
            elif k.size:
                icpt[d] = float(np.average(mu_hat[k, d], weights=n2[k, d]))
        prior = X @ coef + icpt
    elif p["prior"] == "taxonomy":
        if taxa is None:
            raise ValueError("the taxonomy prior needs the unit labels")
        prior = taxonomy_prior(s0, st["s1"], np.asarray(taxa[0])[vocab], np.asarray(taxa[1])[vocab])
    else:
        raise ValueError(p["prior"])

    # ---- shrinkage (optimum toward the prior, tolerance toward the pooled value), floor
    kappa = float(p["kappa"])
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(n2 + kappa > 0, n2 / (n2 + kappa), 0.0)
    mu = np.where(lab, lam * np.where(lab, mu_hat, 0.0) + (1.0 - lam) * prior, prior)
    t_pool = np.ones(D)
    for d in np.flatnonzero(alive):
        k = n2[:, d] >= p["tol_pool_min_n2"]
        if k.any():
            t_pool[d] = weighted_median(var_hat[k, d], n2[k, d])
    if kappa == 0.0:                                   # reading R15b: no tolerance estimate at N2 <= 1
        var_hat = np.where(lab & (n2 <= 1.0 + 1e-6), t_pool[None, :], var_hat)
    var = np.maximum(lam * var_hat + (1.0 - lam) * t_pool, p["tol_floor"] ** 2)
    mu[:, ~alive], var[:, ~alive] = 0.0, 1.0          # a dead dimension cancels in the softmax
    w = 1.0 / var

    # ---- context prior m(ctx) and residual covariance
    cols = np.arange(ctx_tr.shape[1]) if p["context_cols"] is None else np.asarray(p["context_cols"], int)
    Xc = np.asarray(ctx_tr, float)[:, cols]
    fs, ctx_alpha = [None] * D, [None] * D
    if p["context_prior"]:
        for d in np.flatnonzero(alive):
            ok = np.isfinite(Z[:, d])
            fs[d], ctx_alpha[d] = B._ridge_grouped(Xc[ok], Z[ok, d], np.asarray(groups_tr)[ok],
                                                   list(p["context_alphas"]))
        m = np.column_stack([np.zeros(n) if f is None else f(Xc) for f in fs])
        sigma, sigma_inv = _residual_cov(Z - m, p["cov_shrink"], p["eig_floor"])
    else:                                              # reading R15a: flat prior, start at the training mean
        m = np.zeros((n, D))
        sigma, sigma_inv = np.eye(D), float(p["flat_prior_ridge"]) * np.eye(D)

    # ---- intercept (reading R6); missing dimensions take the context-prior mean for this step only
    Zf = np.where(np.isfinite(Z), Z, m)[has]
    Zf2 = Zf * Zf
    WM, c0 = w * mu, -0.5 * (w * mu * mu).sum(1)
    lse = np.empty(len(vocab))
    for s in range(0, len(vocab), 8192):
        k = slice(s, s + 8192)
        Q = c0[k][None, :] + Zf @ WM[k].T - 0.5 * (Zf2 @ w[k].T)
        mx = Q.max(0)
        lse[k] = mx + np.log(np.exp(Q - mx).sum(0))
    psum = np.asarray(P.sum(0)).ravel()
    a = np.log(psum) - lse
    P2 = P.copy()
    P2.data = P2.data ** 2
    n2_all = psum ** 2 / np.asarray(P2.sum(0)).ravel()
    rare = n2_all < p["tol_pool_min_n2"]
    a_rare = float(np.median(a[rare])) if rare.any() else float(np.median(a))

    diag = {"n_train": int(n), "n_train_with_reads": int(has.sum()), "n_units": int(len(vocab)),
            "n_units_table": int(C_tr.shape[1]), "prior": p["prior"], "prior_alpha": alphas,
            "context_prior": bool(p["context_prior"]), "kappa": kappa,
            "context_alpha": ctx_alpha, "n_pc": None if basis is None else int(basis.comp.shape[1]),
            "t_pool": t_pool.tolist(), "share_units_rare": float(rare.mean()),
            "in_vocab_read_share_train": float(N.sum() / max(C_tr.sum(), 1.0))}
    return NicheModel(vocab, mu, w, a, y_mean, y_sd, alive, cols, fs, sigma, sigma_inv, basis, coef, icpt, t_pool,
                      a_rare, diag)


# ----------------------------------------------------------------------------------------------- inference

def moment_features(WM: np.ndarray, W: np.ndarray) -> np.ndarray:
    """[K x (D(D+1) + D*D + 2D + 1)] products of the unit parameters. With pi a row of unit probabilities,
    ``pi @ moment_features`` holds the pi-weighted sums of WM_d WM_e (d <= e), WM_d W_e (all d, e), W_d W_e
    (d <= e), WM_d, W_d and 1, from which :func:`cov_from_moments` builds the covariance term of the Hessian."""
    K, D = WM.shape
    i, j = np.triu_indices(D)
    return np.hstack([WM[:, i] * WM[:, j], (WM[:, :, None] * W[:, None, :]).reshape(K, D * D), W[:, i] * W[:, j],
                      WM, W, np.ones((K, 1))])


def cov_from_moments(R: np.ndarray, z: np.ndarray, gbar: np.ndarray) -> np.ndarray:
    """[B x D x D] covariance term sum_k pi_k g_k g_k' with g_k = WM_k - z * W_k - gbar, from ``R = pi @
    moment_features`` [B x F]. The same quantity as :func:`cov_rowwise` (the definition; kept for the unit test),
    computed with one matrix product per chunk of events instead of one K x D pass per event."""
    B, D = z.shape
    i, j = np.triu_indices(D)
    T = D * (D + 1) // 2

    def sym(v):
        M = np.empty((B, D, D))
        M[:, i, j] = v
        M[:, j, i] = v
        return M

    M1, M2, M3 = sym(R[:, :T]), R[:, T:T + D * D].reshape(B, D, D), sym(R[:, T + D * D:2 * T + D * D])
    o = 2 * T + D * D
    a, c, s = R[:, o:o + D], R[:, o + D:o + 2 * D], R[:, o + 2 * D]
    X = M2 * z[:, None, :]                                           # sum pi WM_d W_e z_e
    Suu = M1 - X - X.transpose(0, 2, 1) + M3 * z[:, :, None] * z[:, None, :]
    ub = a - z * c                                                   # sum pi (WM - z W)
    Y = ub[:, :, None] * gbar[:, None, :]
    cov = Suu - Y - Y.transpose(0, 2, 1) + s[:, None, None] * gbar[:, :, None] * gbar[:, None, :]
    return 0.5 * (cov + cov.transpose(0, 2, 1))


def cov_rowwise(Pi: np.ndarray, WM: np.ndarray, W: np.ndarray, z: np.ndarray, gbar: np.ndarray,
                thr: float) -> np.ndarray:
    """Definition of the covariance term (the implementation before 2026-11-10): units with pi <= thr are left
    out. Reference of the unit test of :func:`cov_from_moments`."""
    out = np.empty((len(z), z.shape[1], z.shape[1]))
    for b in range(len(z)):
        idx = np.flatnonzero(Pi[b] > thr)
        G = (WM[idx] - z[b] * W[idx] - gbar[b]) * np.sqrt(Pi[b, idx])[:, None]
        out[b] = G.T @ G
    return out


def infer(model: NicheModel, C, ctx: np.ndarray, n_eff: float, *, params: dict | None = None,
          extra: tuple | None = None) -> dict:
    """Posterior mode and Laplace SD (standardised units) of every event of ``C`` [n x U].

    ``extra`` = (counts [n x E], profiles [E x 4**k]) turns on the open vocabulary (reading R12).
    Returns ``mode``, ``sd`` [n x D], ``first_ok`` (the first Newton step increased L), ``has_reads``, ``n_steps``,
    ``trace`` [n x (steps + 1)] (L after each iteration), ``in_vocab_share`` and ``exact_hessian``."""
    p = {**DEFAULTS, **(params or {})}
    C = sparse.csr_matrix(C, dtype=np.float64)
    n, K = C.shape[0], len(model.vocab)
    D = model.mu.shape[1]
    Cv = C[:, model.vocab]
    Wt, MUt, at = model.w, model.mu, model.a
    present = None
    if extra is not None:
        Ce = sparse.csr_matrix(extra[0], dtype=np.float64)
        mu_e, w_e, a_e = model.open_units(extra[1], p)
        Cv = sparse.hstack([Cv, Ce], format="csr")
        Wt, MUt, at = np.vstack([Wt, w_e]), np.vstack([MUt, mu_e]), np.concatenate([at, a_e])
        present = (Ce > 0).tocsr()
    Kt = len(at)
    WMt = Wt * MUt
    ct = at - 0.5 * (Wt * MUt * MUt).sum(1)
    P, N = _row_normalise(Cv)
    total = np.asarray(C.sum(1)).ravel() + (0.0 if extra is None else np.asarray(extra[0].sum(1)).ravel())
    has = N > 0
    PW, PWM, Pc = P @ Wt, P @ WMt, P @ ct
    m = model.prior_mean(ctx)
    Si = model.sigma_inv
    steps, halv = int(p["newton_steps"]), int(p["halvings"])
    thr = p["hess_tol"] / Kt

    z = m.copy()
    sd = np.tile(np.sqrt(np.diag(model.sigma)), (n, 1))
    first_ok = np.zeros(n, bool)
    n_steps = np.zeros(n, int)
    exact = np.zeros(n, bool)
    trace = np.full((n, steps + 1), np.nan)

    def evaluate(zc, rows):
        """L and pi at ``zc`` for the events ``rows``."""
        Q = ct[None, :] + zc @ WMt.T - 0.5 * ((zc * zc) @ Wt.T)
        if present is not None:
            Q[:, K:][~present[rows].toarray()] = -np.inf
        mx = Q.max(1, keepdims=True)
        np.exp(Q - mx, out=Q)
        tot = Q.sum(1, keepdims=True)
        Q /= tot
        ll = Pc[rows] + (zc * PWM[rows]).sum(1) - 0.5 * (zc * zc * PW[rows]).sum(1) - (mx[:, 0] + np.log(tot[:, 0]))
        dz = zc - m[rows]
        return n_eff * ll - 0.5 * np.einsum("bi,ij,bj->b", dz, Si, dz), Q

    FM = moment_features(WMt, Wt)

    def grad_hess(zc, rows, Pi):
        Epw, Epwm = Pi @ Wt, Pi @ WMt
        gbar = Epwm - Epw * zc
        g = n_eff * ((PWM[rows] - PW[rows] * zc) - gbar) - (zc - m[rows]) @ Si
        Hs, ex = np.empty((len(rows), D, D)), np.zeros(len(rows), bool)
        covs = cov_from_moments(np.where(Pi > thr, Pi, 0.0) @ FM, zc, gbar)
        for b in range(len(rows)):
            cov = covs[b]
            H = n_eff * (np.diag(PW[rows[b]] - Epw[b]) + cov) + Si
            try:
                np.linalg.cholesky(H)
                ex[b] = True
            except np.linalg.LinAlgError:
                H = n_eff * cov + Si
            Hs[b] = H
        return g, Hs, ex

    rows_all = np.flatnonzero(has)
    for s in range(0, len(rows_all), int(p["chunk"])):
        ch = rows_all[s:s + int(p["chunk"])]
        zc = z[ch].copy()
        L, Pi = evaluate(zc, ch)
        act = np.ones(len(ch), bool)
        trace[ch, 0] = L
        for it in range(steps):
            ia = np.flatnonzero(act)
            if ia.size == 0:
                trace[ch, it + 1:] = L[:, None]
                break
            g, Hs, _ = grad_hess(zc[ia], ch[ia], Pi[ia])
            delta = np.linalg.solve(Hs, g[:, :, None])[:, :, 0]
            acc = np.zeros(ia.size, bool)
            step = 1.0
            for _ in range(halv + 1):
                sel = np.flatnonzero(~acc)
                zt = zc[ia[sel]] + step * delta[sel]
                Lt, Pt = evaluate(zt, ch[ia[sel]])
                ok = Lt > L[ia[sel]]
                good = ia[sel[ok]]
                zc[good], L[good], Pi[good] = zt[ok], Lt[ok], Pt[ok]
                acc[sel[ok]] = True
                if acc.all():
                    break
                step *= 0.5
            if it == 0:
                first_ok[ch[ia]] = acc
            n_steps[ch[ia[acc]]] += 1
            act[ia[~acc]] = False
            trace[ch, it + 1] = L
        _, Hs, ex = grad_hess(zc, ch, Pi)
        z[ch], exact[ch] = zc, ex
        sd[ch] = np.sqrt(np.maximum(np.diagonal(np.linalg.inv(Hs), axis1=1, axis2=2), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, N / total, np.nan)
    return {"mode": z, "sd": sd, "first_ok": first_ok, "has_reads": has, "n_steps": n_steps, "trace": trace,
            "in_vocab_share": share, "exact_hessian": exact}


# --------------------------------------------------------------------------------------------- fold level

def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 3:
        return float("nan")
    sst = float(((y[ok] - y[ok].mean()) ** 2).sum())
    return float(1.0 - ((y[ok] - yhat[ok]) ** 2).sum() / sst) if sst > 0 else float("nan")


def deshrink_fit(pred: np.ndarray, y: np.ndarray, clip=(0.5, 3.0)) -> tuple[float, float]:
    """(slope, intercept) of the inverse-deshrinking line y ~ pred (reading R10). ``clip=None``: no clip (A0)."""
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 3:
        return 1.0, 0.0
    xm, ym = float(pred[ok].mean()), float(y[ok].mean())
    vx = float(((pred[ok] - xm) ** 2).sum())
    b = float(((pred[ok] - xm) * (y[ok] - ym)).sum() / vx) if vx > 0 else 1.0
    if clip is not None:
        b = float(np.clip(b, clip[0], clip[1]))
    return b, ym - b * xm


@dataclass
class FoldResult:
    pred: np.ndarray                 # [n_test x D] deshrunk posterior mode, target scale
    sd: np.ndarray                   # [n_test x D] Laplace SD x slope, target scale
    raw: np.ndarray                  # [n_test x D] posterior mode before deshrinking, target scale
    raw_sd: np.ndarray
    train_oof: np.ndarray | None     # [n_train x D] inner out-of-fold modes at the chosen N_eff (target scale)
    train_oof_sd: np.ndarray | None
    n_eff: float
    slope: np.ndarray
    intercept: np.ndarray
    diag: dict


def fit_fold(C_tr, C_te, Y_tr: np.ndarray, ctx_tr: np.ndarray, ctx_te: np.ndarray, site_tr, domain_tr,
             prof: np.ndarray | None, *, params: dict | None = None, taxa: tuple | None = None,
             scored: np.ndarray | None = None, extra_te: tuple | None = None) -> FoldResult:
    """One outer fold: N_eff by inner domain-grouped CV, cross-fitted deshrinking, inversion of the test events.

    No argument carries a held-out label. ``scored``: boolean mask of the dimensions entering the N_eff criterion
    (default all)."""
    p = {**DEFAULTS, **(params or {})}
    C_tr = sparse.csr_matrix(C_tr, dtype=np.float64)
    Y_tr = np.asarray(Y_tr, float)
    n, D = Y_tr.shape
    site_tr, domain_tr = np.asarray(site_tr).astype(str), np.asarray(domain_tr).astype(str)
    scored = np.ones(D, bool) if scored is None else np.asarray(scored, bool)
    grid = [float(x) for x in p["n_eff_grid"]]
    k = min(int(p["inner_folds"]), len(np.unique(domain_tr)))
    slope, icpt = np.ones(D), np.zeros(D)
    oof = oof_sd = None
    diag = {"inner_folds": int(k), "inner": {}}
    n_eff = float(p["n_eff_default"])
    if k >= 2:
        from sklearn.model_selection import GroupKFold

        oofs = {ne: np.full((n, D), np.nan) for ne in grid}
        sds = {ne: np.full((n, D), np.nan) for ne in grid}
        fail = {ne: 0 for ne in grid}
        n_inf = 0
        for a, b in GroupKFold(k).split(np.zeros(n), groups=domain_tr):
            mi = fit_niche(C_tr[a], Y_tr[a], ctx_tr[a], site_tr[a], prof, p, taxa)
            n_inf += len(b)
            for ne in grid:
                r = infer(mi, C_tr[b], ctx_tr[b], ne, params=p)
                oofs[ne][b] = mi.to_target(r["mode"])
                sds[ne][b] = r["sd"] * mi.y_sd
                fail[ne] += int((r["has_reads"] & ~r["first_ok"]).sum())
        best = None
        for ne in grid:
            lines = [deshrink_fit(oofs[ne][:, d], Y_tr[:, d], p["slope_clip"]) for d in range(D)]
            r2s = [r2(Y_tr[:, d], lines[d][1] + lines[d][0] * oofs[ne][:, d]) if scored[d] else np.nan
                   for d in range(D)]
            crit = float(np.nanmean(r2s)) if np.isfinite(r2s).any() else -np.inf
            diag["inner"][str(ne)] = {"criterion": crit, "first_step_fail_share": fail[ne] / max(n_inf, 1)}
            if best is None or crit > best[0]:
                best = (crit, ne, lines)
        n_eff = best[1]
        slope, icpt = np.array([l[0] for l in best[2]]), np.array([l[1] for l in best[2]])
        oof, oof_sd = oofs[n_eff], sds[n_eff]
    model = fit_niche(C_tr, Y_tr, ctx_tr, site_tr, prof, p, taxa)
    r = infer(model, C_te, ctx_te, n_eff, params=p, extra=extra_te)
    raw, raw_sd = model.to_target(r["mode"]), r["sd"] * model.y_sd
    raw_sd[:, ~model.alive] = np.nan
    wr = r["has_reads"]
    diag.update(model.diag)
    diag.update({"n_eff": n_eff, "slope": slope.tolist(), "intercept": icpt.tolist(), "n_test": int(C_te.shape[0]),
                 "n_test_without_reads": int((~wr).sum()),
                 "test_first_step_fail_share": float((wr & ~r["first_ok"]).sum() / max(wr.sum(), 1)),
                 "test_mean_newton_steps": float(r["n_steps"][wr].mean()) if wr.any() else 0.0,
                 "test_exact_hessian_share": float(r["exact_hessian"][wr].mean()) if wr.any() else 0.0,
                 "test_in_vocab_read_share": float(np.nanmean(r["in_vocab_share"])) if len(wr) else float("nan")})
    return FoldResult(icpt + slope * raw, slope * raw_sd, raw, raw_sd, oof, oof_sd, n_eff, slope, icpt, diag)


# -------------------------------------------------------------------------------------------- design level

_FOLD_CACHE: dict = {}               # keyed by variant + parameters + rows + training labels + context: the
FOLD_CACHE_SIZE = 64                 # control ``xgb_nichez_context`` and both target families reuse D1's folds
LAST: dict = {}                      # the FoldResult of the most recent design-level call (diagnostics, SDs)


def _fold(name: str, design, train_ids, test_ids, params: dict | None) -> tuple[FoldResult, np.ndarray, np.ndarray]:
    S = design.samples
    missing = [t for t in MASTER if t not in S.columns]
    if missing:
        raise KeyError(f"design.samples lacks {missing}: call archv2.attach_master(design) first")
    variant = "niche_s" if name == CONTROL else name
    p = {**DEFAULTS, **VARIANTS[variant], **(params or {})}
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    Y_tr = S.loc[train_ids, list(MASTER)].to_numpy(float)
    ctx_tr, fill = F.context_features(S, train_ids)
    ctx_te, _ = F.context_features(S, test_ids, fill)
    if p["qmp"]:                                       # reading R14: training-median fill + missing flags
        from sentinel.models.microbe.v4mr import _qmp_block

        (qtr, gtr), (qte, gte) = _qmp_block(train_ids, p.get("qmp_table")), _qmp_block(test_ids, p.get("qmp_table"))
        with np.errstate(all="ignore"):
            med = np.where(np.isfinite(qtr).any(0), np.nanmedian(np.where(np.isfinite(qtr), qtr, np.nan), 0), 0.0)
        ctx_tr = np.column_stack([ctx_tr, np.where(np.isfinite(qtr), qtr, med), gtr])
        ctx_te = np.column_stack([ctx_te, np.where(np.isfinite(qte), qte, med), gte])
    h = hashlib.sha1()
    for part in (variant, repr((id(design.tokens), len(design.tokens))),
                 repr(sorted((k, repr(v)) for k, v in p.items() if k != "qmp_table")),
                 "|".join(map(str, train_ids)),
                 "|".join(map(str, test_ids))):
        h.update(part.encode())
    h.update(np.ascontiguousarray(Y_tr).tobytes())
    h.update(np.ascontiguousarray(ctx_tr).tobytes())
    h.update(np.ascontiguousarray(ctx_te).tobytes())
    key = h.hexdigest()
    if key not in _FOLD_CACHE:
        ut = unit_table(design)
        tr, te = ut.rows(train_ids), ut.rows(test_ids)
        prof = ut.profile(int(p["kmer"])) if p["prior"] == "sequence" else None
        res = fit_fold(ut.counts[tr], ut.counts[te], Y_tr, ctx_tr, ctx_te, S.loc[train_ids, "site_id"],
                       S.loc[train_ids, "domain_id"], prof, params=p, taxa=(ut.genus, ut.family),
                       scored=np.array([t in SCORED for t in MASTER]))
        while len(_FOLD_CACHE) >= FOLD_CACHE_SIZE:     # a FoldResult is small (no model, no count matrix)
            _FOLD_CACHE.pop(next(iter(_FOLD_CACHE)))
        _FOLD_CACHE[key] = res
    return _FOLD_CACHE[key], ctx_tr, ctx_te


def fit_predict(name: str, design, train_ids, test_ids, *, targets, seed: int = 0, n_jobs: int = 6,
                params: dict | None = None, device: str = "cpu") -> pd.DataFrame:
    """Same contract as ``baselines.fit_predict``. The inversion always fits the 17 master dimensions (one fit
    serves both target families; the fold is cached, so a second call for the other family costs nothing)."""
    from threadpoolctl import threadpool_limits

    targets = list(targets)
    bad = [t for t in targets if t not in MASTER]
    if bad:
        raise ValueError(f"{name}: {bad} are not master dimensions")
    if name not in NAMES:
        raise ValueError(f"unknown niche model {name}")
    cols = [MASTER.index(t) for t in targets]
    test_ids = np.asarray(test_ids)
    out = pd.DataFrame(np.nan, index=pd.Index(test_ids, name="sample_id"), columns=targets)
    with threadpool_limits(limits=int(n_jobs)):
        res, ctx_tr, ctx_te = _fold(name, design, train_ids, test_ids, params)
    LAST.clear()
    LAST.update(name=name, result=res, test_ids=test_ids)
    if name != CONTROL:
        out[targets] = res.pred[:, cols]
        return out
    if res.train_oof is None:
        raise ValueError(f"{CONTROL} needs inner out-of-fold inferences (>= 2 training domains)")
    import xgboost as xgb

    Xtr = np.column_stack([res.train_oof, res.train_oof_sd, ctx_tr]).astype(np.float32)
    Xte = np.column_stack([res.raw, res.raw_sd, ctx_te]).astype(np.float32)
    Ytr = design.samples.loc[np.asarray(train_ids), targets].astype(float)
    for t in targets:
        y = Ytr[t].to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        m = xgb.XGBRegressor(**B.DEFAULTS["xgb"], random_state=seed, n_jobs=n_jobs, tree_method="hist")
        out[t] = m.fit(Xtr[ok], y[ok]).predict(Xte)
    return out
