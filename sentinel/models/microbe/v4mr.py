# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Block-feature tabular models of architecture round v2 (amendment microbe_arch_v2 §3, §5, §6).

* arm D3a ``xgb_v4mr``: CLR of V4 units (cap 3,000) + CLR of genus, family and order counts (cap 500 each) +
  community descriptors, XGB with the ``xgb_v4clr`` hyperparameters;
* its twins ``xgb_gfo`` and ``xgb_v4clr_3000``;
* parity comparators ``rf_v4clr``, ``tabpfn_v4clr`` and the multi-output ``xgb_v4clr_mo``;
* arm D3b ``xgb_v4mr_stack`` and its twin ``xgb_clr_stack`` (cross-fitted predictions of the other master
  dimensions as extra features);
* arm D4: the ``<model>_qmp`` twins and the control ``xgb_context_qmp``.

Every transform is fitted on the fold's training rows. A CLR block here is the sparse equivalent of
``features.GenusCLR`` (same prevalence filter, same tie order, same pooled "other" column, same pseudocount);
the equality is unit-tested, and the block re-implementations of baseline tables (``REIMPL``) exist only to
carry the QMP block.

Implementation readings (fixed before any development run, repeated in ``microbe_arch_v2_log.md``):

V1  ``xgb_v4clr_mo`` (lead ruling B5; proposed in ``microbe_arch_v2_deviations_01.md``): ONE
    ``multi_output_tree`` model on all 17 master dimensions, whatever target family is asked for (the fit is cached
    for the fold, so families P and G read the same model). Targets are standardised with the training mean and SD
    of their labelled rows. A missing label is MASKED, not filled: the squared-error objective is supplied as a
    custom function whose gradient and Hessian are zero where the label is missing (``base_score`` 0 = the
    training mean after standardisation). A row with no label on any dimension is dropped; a dimension with fewer
    than 10 labelled training rows or zero variance is left out of the model (prediction NaN).
V2  Stacking: ``GroupKFold(5)`` over the training domains; auxiliary model = the base model with seed 0 fitted
    on the 17 master dimensions; the model for target t receives the 16 other dimensions.
V3  QMP block: 4 features + 4 missing flags. Trees and TabPFN receive NaN, ridge the training median.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from sentinel.models.microbe import baselines as B
from sentinel.models.microbe import features as F

V4MR_BLOCKS = (("v4", 3000), ("genus", 500), ("family", 500), ("order", 500))
GFO_BLOCKS = (("genus", 500), ("family", 500), ("order", 500))


def _spec(blocks=(), learner="xgb", context=False, descriptors=True, stack=None, qmp=False) -> dict:
    return {"blocks": tuple(blocks), "learner": learner, "context": context, "descriptors": descriptors,
            "stack": stack, "qmp": qmp}


SPECS = {
    "xgb_v4mr": _spec(V4MR_BLOCKS),
    "xgb_gfo": _spec(GFO_BLOCKS),
    "xgb_v4clr_3000": _spec((("v4", 3000),)),
    "rf_v4clr": _spec((("v4", 500),), "rf"),
    "tabpfn_v4clr": _spec((("v4", 300),), "tabpfn"),
    "xgb_v4clr_mo": _spec((("v4", 500),), "xgb_mo"),
    "xgb_v4mr_stack": _spec(V4MR_BLOCKS, stack="xgb_v4mr"),
    "xgb_clr_stack": _spec((("genus", 500),), stack="xgb_clr"),
}
# Block re-implementations of tables of ``baselines.py`` (features unit-tested equal). Not run under these names.
REIMPL = {
    "xgb_clr": _spec((("genus", 500),)),
    "rf_clr": _spec((("genus", 500),), "rf"),
    "ridge_clr": _spec((("genus", 500),), "ridge"),
    "tabpfn_clr": _spec((("genus", 300),), "tabpfn"),
    "rf_clr_tuned": _spec((("genus", 500),), "rf_clr_tuned"),
    "xgb_clr_tuned": _spec((("genus", 500),), "xgb_clr_tuned"),
    "xgb_v4clr": _spec((("v4", 500),)),
    "ridge_v4clr": _spec((("v4", 500),), "ridge"),
    "rf_clr_context": _spec((("genus", 500),), "rf", context=True),
    "xgb_clr_context": _spec((("genus", 500),), context=True),
    "xgb_v4clr_context": _spec((("v4", 500),), context=True),
    "context_ridge": _spec((), "ridge", context=True, descriptors=False),
    "context_rf": _spec((), "rf", context=True, descriptors=False),
}
QMP_TWINS = {f"{n}_qmp": {**s, "qmp": True} for n, s in {**SPECS, **REIMPL}.items()}
QMP_TWINS["xgb_context_qmp"] = _spec((), "xgb", context=True, descriptors=False, qmp=True)
ALL = {**SPECS, **QMP_TWINS}
NAMES = tuple(ALL)
STOCHASTIC = {n for n, s in ALL.items() if s["learner"] != "ridge"}
NEEDS_GPU = {n for n, s in ALL.items() if s["learner"] == "tabpfn"}
LABELS = {"xgb_v4mr": "V4-MR-XGB[V4-3000 + genus + family + order]", "xgb_gfo": "XGB[genus + family + order]",
          "xgb_v4clr_3000": "V4-XGB[cap 3,000]", "rf_v4clr": "RF[V4-CLR]", "tabpfn_v4clr": "TabPFNv2[V4-CLR]",
          "xgb_v4clr_mo": "V4-XGB[multi-output]", "xgb_v4mr_stack": "V4-MR-Stack", "xgb_clr_stack": "XGB[genus-CLR]-Stack",
          "xgb_context_qmp": "XGB[DOY+location+QMP]"}


# ----------------------------------------------------------------------------------------- sparse CLR blocks

@dataclass
class RankCounts:
    labels: np.ndarray               # sorted, without the unassigned label
    counts: sparse.csr_matrix        # [events x labels], events in ``design.samples`` order
    total: np.ndarray                # all reads of the event (assigned or not)


_RC_CACHE: dict = {}


def rank_counts(design, rank: str) -> RankCounts:
    key = (id(design.tokens), len(design.tokens), len(design.samples), tuple(design.samples.index[:3]))
    if _RC_CACHE.get("key") != key:
        _RC_CACHE.clear()
        _RC_CACHE["key"] = key
    if rank not in _RC_CACHE:
        if rank == "v4":
            from sentinel.models.microbe.niche import unit_table

            ut = unit_table(design)
            _RC_CACHE[rank] = RankCounts(ut.units, ut.counts, ut.n_reads_all)
        else:
            tok, events = design.tokens, pd.Index(design.samples.index)
            g = tok[rank].fillna(F.UNASSIGNED).astype(str).replace({"": F.UNASSIGNED}).to_numpy()
            r = events.get_indexer(tok["sample_id"])
            c = tok["count"].to_numpy(float)
            ok = r >= 0
            total = np.bincount(r[ok], c[ok], minlength=len(events))
            ok &= g != F.UNASSIGNED
            labels, inv = np.unique(g[ok], return_inverse=True)
            m = sparse.csr_matrix((c[ok], (r[ok], inv)), shape=(len(events), len(labels)))
            m.sum_duplicates()
            _RC_CACHE[rank] = RankCounts(labels, m, total)
    return _RC_CACHE[rank]


def block_fit(rc: RankCounts, tr_rows: np.ndarray, cap: int, min_prev: float) -> np.ndarray:
    """Columns kept: prevalence >= ``min_prev`` in the training rows, the ``cap`` most prevalent, ties in label
    order (as ``GenusCLR.fit``)."""
    prev = np.asarray((rc.counts[tr_rows] > 0).sum(0)).ravel() / len(tr_rows)
    cand = np.flatnonzero(prev >= min_prev)
    keep = cand[np.argsort(-prev[cand], kind="stable")][:cap]
    if not keep.size:
        raise ValueError("no label passes the training-fold prevalence filter")
    return keep


def block_transform(rc: RankCounts, keep: np.ndarray, rows: np.ndarray, pseudo: float) -> np.ndarray:
    kept = rc.counts[rows][:, keep].toarray()
    other = rc.total[rows] - kept.sum(1)
    return F.clr(np.column_stack([kept, other]), pseudo).astype(np.float32)


def _rows(design, ids) -> np.ndarray:
    r = pd.Index(design.samples.index).get_indexer(pd.Index(np.asarray(ids)))
    if (r < 0).any():
        raise KeyError("ids outside the design")
    return r


def _qmp_block(ids, table: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    from sentinel.data import microbe_geo as G

    if table is None:
        q = G.load_qmp(ids)
    else:
        q = table.reindex(pd.Index(np.asarray(ids)).astype(str))
    feats = q[list(G.QMP_FEATURES)].to_numpy(float)
    return feats, np.isnan(feats).astype(np.float32)


_X_CACHE: dict = {}


def features(name: str, design, train_ids, test_ids, params: dict | None = None) -> dict:
    """{'Xtr', 'Xte', 'qmp_cols', 'names'} for a model name (without stacking features). Cached for one fold."""
    spec = {**ALL, **REIMPL}[name]
    p = {**B.DEFAULTS, **(params or {})}
    clr_p = p["clr"]
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    h = hashlib.sha1(repr((spec["blocks"], spec["context"], spec["descriptors"], spec["qmp"], sorted(clr_p.items()),
                           p.get("descriptors", True), id(design.tokens), id(design.samples), len(design.samples),
                           id(p.get("qmp_table")) if spec["qmp"] else None)).encode())
    h.update("|".join(map(str, train_ids)).encode())
    h.update(b"//")
    h.update("|".join(map(str, test_ids)).encode())
    key = h.hexdigest()
    if key in _X_CACHE:
        return _X_CACHE[key][0]
    S = design.samples
    tr, te = _rows(design, train_ids), _rows(design, test_ids)
    Xtr, Xte, names = [], [], []
    for rank, cap in spec["blocks"]:
        rc = rank_counts(design, rank)
        keep = block_fit(rc, tr, int(cap), float(clr_p["min_prev"]))
        Xtr.append(block_transform(rc, keep, tr, float(clr_p["pseudo"])))
        Xte.append(block_transform(rc, keep, te, float(clr_p["pseudo"])))
        names += [f"{rank}:{l}" for l in rc.labels[keep]] + [f"{rank}:{F.OTHER}"]
    if spec["context"]:
        ctr, fill = F.context_features(S, train_ids)
        cte, _ = F.context_features(S, test_ids, fill)
        Xtr.append(ctr), Xte.append(cte)
        names += list(F.CONTEXT_COLS)
    if spec["descriptors"]:
        on = p.get("descriptors", True)
        dtr = F.sample_descriptors(S, train_ids, primer_onehot=True, enabled=on)
        dte = F.sample_descriptors(S, test_ids, primer_onehot=True, enabled=on)
        Xtr.append(dtr), Xte.append(dte)
        names += [f"desc{i}" for i in range(dtr.shape[1])]
    n0 = sum(x.shape[1] for x in Xtr)
    qmp_cols = np.zeros(0, int)
    if spec["qmp"]:
        from sentinel.data.microbe_geo import QMP_FEATURES, QMP_FLAGS

        (ftr, gtr), (fte, gte) = _qmp_block(train_ids, p.get("qmp_table")), _qmp_block(test_ids, p.get("qmp_table"))
        Xtr += [ftr.astype(np.float32), gtr]
        Xte += [fte.astype(np.float32), gte]
        qmp_cols = np.arange(n0, n0 + ftr.shape[1])
        names += list(QMP_FEATURES) + list(QMP_FLAGS)
    out = {"Xtr": np.column_stack(Xtr).astype(np.float32), "Xte": np.column_stack(Xte).astype(np.float32),
           "qmp_cols": qmp_cols, "names": names}
    _X_CACHE.clear()                                         # the entry keeps the keyed objects alive (ids stay unique)
    _X_CACHE[key] = (out, (design.tokens, design.samples, p.get("qmp_table")))
    return out


# ------------------------------------------------------------------------------------------------- stacking

_STACK_CACHE: dict = {}


def _base_fit_predict(base: str, design, train_ids, test_ids, *, targets, n_jobs, params) -> pd.DataFrame:
    if base in SPECS:
        return fit_predict(base, design, train_ids, test_ids, targets=targets, seed=0, n_jobs=n_jobs, params=params)
    return B.fit_predict(base, design, train_ids, test_ids, targets=targets, seed=0, n_jobs=n_jobs, params=params)


def stack_features(base: str, design, train_ids, test_ids, *, n_jobs: int = 6, params: dict | None = None,
                   inner_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fitted auxiliary predictions of the 17 master dimensions (reading V2).

    Training rows: out-of-fold predictions of domain-grouped inner folds of the training rows. Held-out rows:
    predictions of the auxiliary models refitted on all training rows. No label of a held-out row is read, and
    no training row's prediction comes from a model that saw that row's labels."""
    from sklearn.model_selection import GroupKFold

    from sentinel.models.microbe.niche import MASTER

    S = design.samples
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    h = hashlib.sha1(("|".join(map(str, train_ids)) + "//" + "|".join(map(str, test_ids)) + base
                      + repr(sorted((k, repr(v)) for k, v in (params or {}).items() if k != "qmp_table"))).encode())
    h.update(np.ascontiguousarray(S.loc[train_ids, list(MASTER)].to_numpy(float)).tobytes())
    key = h.hexdigest()
    if key in _STACK_CACHE:
        return _STACK_CACHE[key]
    dom = S.loc[train_ids, "domain_id"].astype(str).to_numpy()
    k = min(int(inner_folds), len(np.unique(dom)))
    if k < 2:
        raise ValueError("stacking needs at least two training domains")
    A_tr = pd.DataFrame(np.nan, index=pd.Index(train_ids, name="sample_id"), columns=list(MASTER))
    for a, b in GroupKFold(k).split(np.zeros(len(train_ids)), groups=dom):
        A_tr.iloc[b] = _base_fit_predict(base, design, train_ids[a], train_ids[b], targets=MASTER, n_jobs=n_jobs,
                                         params=params).to_numpy()
    A_te = _base_fit_predict(base, design, train_ids, test_ids, targets=MASTER, n_jobs=n_jobs, params=params)
    _STACK_CACHE.clear()
    _STACK_CACHE[key] = (A_tr, A_te[list(MASTER)])
    return _STACK_CACHE[key]


# ------------------------------------------------------------------------------- masked multi-output model

_MO_CACHE: dict = {}


def masked_squared_error(mask: np.ndarray):
    """Squared-error objective that ignores the entries where ``mask`` is False (gradient = Hessian = 0)."""
    M = np.asarray(mask, bool)

    def objective(y_true, y_pred):
        pred = np.asarray(y_pred, float).reshape(M.shape)
        y = np.asarray(y_true, float).reshape(M.shape)
        return np.where(M, pred - y, 0.0), M.astype(float)

    return objective


def masked_mo_fit_predict(Xtr: np.ndarray, Y: np.ndarray, Xte: np.ndarray, *, seed: int, n_jobs: int,
                          xgb_params: dict, min_rows: int = 10) -> np.ndarray:
    """[n_test x Y.shape[1]] predictions of one masked multi-output model; NaN for a dimension left out."""
    import xgboost as xgb

    Y = np.asarray(Y, float)
    out = np.full((len(Xte), Y.shape[1]), np.nan)
    with np.errstate(all="ignore"):
        cols = [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).sum() >= min_rows and np.nanstd(Y[:, j]) > 0]
    if not cols:
        return out
    mean, sd = np.nanmean(Y[:, cols], 0), np.nanstd(Y[:, cols], 0)
    Z = (Y[:, cols] - mean) / sd
    M = np.isfinite(Z)
    rows = M.any(1)
    Z, M = np.where(M, Z, 0.0)[rows], M[rows]                # XGBoost refuses NaN labels; the mask carries them
    m = xgb.XGBRegressor(**xgb_params, random_state=seed, n_jobs=n_jobs, tree_method="hist",
                         multi_strategy="multi_output_tree", objective=masked_squared_error(M), base_score=0.0)
    pred = np.asarray(m.fit(Xtr[rows], Z).predict(Xte)).reshape(len(Xte), len(cols))
    out[:, cols] = pred * sd + mean
    return out


def _masked_mo(design, train_ids, test_ids, Xtr, Xte, seed, n_jobs, xgb_params) -> pd.DataFrame:
    from sentinel.models.microbe.niche import MASTER

    S = design.samples
    lacking = [c for c in MASTER if c not in S.columns]
    if lacking:
        raise KeyError(f"design.samples lacks {lacking}: call archv2.attach_master(design) first")
    Y = S.loc[train_ids, list(MASTER)].to_numpy(float)
    h = hashlib.sha1(repr((int(seed), sorted(xgb_params.items()), Xtr.shape, Xte.shape)).encode())
    for arr in (Xtr, Xte, Y):
        h.update(np.ascontiguousarray(arr).tobytes())
    key = h.hexdigest()
    if key not in _MO_CACHE:
        pred = masked_mo_fit_predict(Xtr, Y, Xte, seed=seed, n_jobs=n_jobs, xgb_params=xgb_params)
        _MO_CACHE.clear()
        _MO_CACHE[key] = pd.DataFrame(pred, index=pd.Index(test_ids, name="sample_id"), columns=list(MASTER))
    return _MO_CACHE[key]


# -------------------------------------------------------------------------------------------- design level

def fit_predict(name: str, design, train_ids, test_ids, *, targets, seed: int = 0, n_jobs: int = 6,
                params: dict | None = None, device: str = "cpu", _spec_table: dict | None = None) -> pd.DataFrame:
    """Same contract as ``baselines.fit_predict``."""
    table = _spec_table or ALL
    if name not in table:
        raise ValueError(f"unknown block model {name}")
    spec = table[name]
    p = {**B.DEFAULTS, **(params or {})}
    S = design.samples
    targets = list(targets)
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    out = pd.DataFrame(np.nan, index=pd.Index(test_ids, name="sample_id"), columns=targets)
    fx = features(name, design, train_ids, test_ids, params)
    Xtr, Xte = fx["Xtr"], fx["Xte"]
    Ytr = S.loc[train_ids, targets].astype(float)
    groups = S.loc[train_ids, "site_id"].astype(str).to_numpy()
    learner = spec["learner"]
    A_tr = A_te = None
    if spec["stack"]:
        A_tr, A_te = stack_features(spec["stack"], design, train_ids, test_ids, n_jobs=n_jobs,
                                    params={k: v for k, v in (params or {}).items()})
    if learner == "ridge" and fx["qmp_cols"].size:           # reading V3: ridge gets the training median
        Xtr, Xte = Xtr.copy(), Xte.copy()
        for c in fx["qmp_cols"]:
            med = float(np.nanmedian(Xtr[:, c])) if np.isfinite(Xtr[:, c]).any() else 0.0
            Xtr[np.isnan(Xtr[:, c]), c] = med
            Xte[np.isnan(Xte[:, c]), c] = med

    if learner == "xgb_mo":                                  # reading V1: one masked model on the 17 dimensions
        pred = _masked_mo(design, train_ids, test_ids, Xtr, Xte, seed, n_jobs, p["xgb"])
        missing = [t for t in targets if t not in pred.columns]
        if missing:
            raise ValueError(f"{name}: {missing} are not master dimensions")
        out[targets] = pred[targets].to_numpy()
        return out

    from sentinel.models.microbe.niche import MASTER

    for t in targets:
        y = Ytr[t].to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        Xa, Xb = Xtr, Xte
        if A_tr is not None:
            others = [c for c in MASTER if c != t]
            Xa = np.column_stack([Xtr, A_tr[others].to_numpy(np.float32)])
            Xb = np.column_stack([Xte, A_te[others].to_numpy(np.float32)])
        if learner == "xgb":
            import xgboost as xgb

            m = xgb.XGBRegressor(**p["xgb"], random_state=seed, n_jobs=n_jobs, tree_method="hist")
            out[t] = m.fit(Xa[ok], y[ok]).predict(Xb)
        elif learner == "rf":
            out[t] = B._rf(seed, n_jobs, p["rf"]).fit(Xa[ok], y[ok]).predict(Xb)
        elif learner == "ridge":
            f, _ = B._ridge_grouped(Xa[ok], y[ok], groups[ok], p["ridge_alphas"])
            out[t] = f(Xb)
        elif learner in B.TUNE_GRIDS:
            out[t] = B._tuned_fit_predict(learner, Xa[ok], y[ok], groups[ok], Xb, seed, n_jobs, p)
        elif learner == "tabpfn":
            from sentinel.models.chemtox.tabpfn_model import load_tabpfn

            TabPFNRegressor, _ = load_tabpfn()
            idx = np.flatnonzero(ok)
            mx = int(p["tabpfn"]["max_train_rows"])
            if idx.size > mx:
                idx = np.random.default_rng(seed).choice(idx, mx, replace=False)
            m = TabPFNRegressor(n_estimators=int(p["tabpfn"]["n_estimators"]), random_state=seed, device=device,
                                ignore_pretraining_limits=True)
            m.fit(Xa[idx], y[idx])
            out[t] = np.concatenate([m.predict(Xb[s:s + 2000]) for s in range(0, len(Xb), 2000)])
        else:
            raise ValueError(learner)
    return out
