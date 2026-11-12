# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""TabPFN hybrid (M5-F; protocol §9.2): TabPFN v2 on [top-K Hellinger taxa per modality, BioSet-MA embedding,
context] with a retrieved context per group of query rows.

* Features: for each modality the K taxa (``mixed_otu``) most prevalent among the fold's training samples (at
  least 10 samples), as Hellinger values sqrt(count / sample total); NaN where the row has no sample of the
  modality. Then the low-dimensional BioSet-MA sample embedding of the same fold and the context columns.
* Query rows are grouped by k-means in the standardised embedding space, at most ``group_rows`` rows per group.
* The context of a group, per target: half the labelled training rows nearest to the group centroid, half a
  seeded random sample of the rest stratified by source × region, within the TabPFN row budget of the comparator.
* The regressor call is the comparator's own (``xdev._fit_learner('tabpfn', ...)``: installed TabPFN v2, same
  parameters), so the hybrid differs from TabPFN[c+ctx] only by its features and its context selection.

Fitted statistics (taxon lists, embedding scaler) come from training rows only; callers pass ids through
``ftrack.assert_fit_ids`` first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.models.bioassess import xdev as XD
from sentinel.models.bioassess.bioset_ma import CONTEXT_COLS, CONTEXT_COLS_NO_COORDS, KEY_COL, MODALITIES

HYBRID = {"k_taxa": {"invertebrate": 200, "diatom": 100}, "min_prevalence": 10, "group_rows": 500,
          "budget": XD.TABPFN_MAX_ROWS, "near_frac": 0.5,
          "context_cols": CONTEXT_COLS, "context_cols_fz": CONTEXT_COLS_NO_COORDS}
RETRIEVALS = ("hybrid", "knn", "random")      # "knn" and "random" are the controls of protocol §8 item 8


def _taxon_col(tokens: pd.DataFrame) -> str:
    return "mixed_otu" if "mixed_otu" in tokens else "name"


@dataclass
class HellingerFeaturizer:
    """Top-K taxa per modality by training prevalence -> Hellinger columns (NaN where the modality is absent)."""

    k_taxa: dict = field(default_factory=lambda: dict(HYBRID["k_taxa"]))
    min_prevalence: int = HYBRID["min_prevalence"]
    modalities: tuple = MODALITIES
    taxa: dict = field(default_factory=dict)

    def fit(self, design, train_ids) -> "HellingerFeaturizer":
        S = design.samples.loc[pd.Index(train_ids)]
        for m in self.modalities:
            keys = set(S[KEY_COL[m]].dropna().astype(str))
            t = design.tokens[m]
            t = t[t["key"].isin(keys)]
            col = _taxon_col(t)
            prev = t.dropna(subset=[col]).drop_duplicates(["key", col]).groupby(col).size()
            prev = prev[prev >= self.min_prevalence]
            order = sorted(prev.index, key=lambda x: (-int(prev[x]), str(x)))      # prevalence, then name
            self.taxa[m] = [str(x) for x in order[: int(self.k_taxa[m])]]
        return self

    @property
    def columns(self) -> list[str]:
        return [f"{m[:3]}|{x}" for m in self.modalities for x in self.taxa[m]]

    def transform(self, design, ids) -> np.ndarray:
        S = design.samples.loc[pd.Index(ids)]
        out = []
        for m in self.modalities:
            names = self.taxa[m]
            X = np.full((len(S), len(names)), np.nan, np.float32)
            k = S[KEY_COL[m]]
            has = k.notna().to_numpy()
            if has.any() and names:
                uniq, inv = np.unique(k[has].astype(str).to_numpy(), return_inverse=True)
                t = design.tokens[m]
                t = t[t["key"].isin(set(uniq))]
                col = _taxon_col(t)
                tot = t.groupby("key")["count"].sum()
                row = pd.Series(np.arange(len(uniq)), index=uniq)
                cix = pd.Series(np.arange(len(names)), index=names)
                tt = t[t[col].astype(str).isin(cix.index)]
                H = np.zeros((len(uniq), len(names)), np.float64)
                np.add.at(H, (row.reindex(tt["key"].astype(str)).to_numpy(int),
                              cix.reindex(tt[col].astype(str)).to_numpy(int)), tt["count"].to_numpy(float))
                denom = tot.reindex(uniq).to_numpy(float)
                ok = np.isfinite(denom) & (denom > 0)
                H[ok] = np.sqrt(H[ok] / denom[ok, None])
                H[~ok] = np.nan                                     # key without any token row: modality absent
                X[has] = H[np.asarray(inv).reshape(-1)].astype(np.float32)
            out.append(X)
        return np.concatenate(out, 1) if out else np.zeros((len(S), 0), np.float32)


def assemble(H: np.ndarray, E: np.ndarray, ctx: np.ndarray | None = None) -> np.ndarray:
    """Feature matrix [Hellinger taxa, embedding, context] (float32; NaN allowed)."""
    parts = [np.asarray(H, np.float32), np.asarray(E, np.float32)]
    if ctx is not None:
        parts.append(np.asarray(ctx, np.float32))
    return np.concatenate(parts, 1)


def group_queries(E: np.ndarray, max_rows: int = 500, seed: int = 0) -> np.ndarray:
    """Group labels 0..G-1 of the query rows: k-means with k = ceil(n / max_rows), then any group above
    ``max_rows`` is split in two until none is. Every row belongs to exactly one group."""
    from sklearn.cluster import KMeans

    E = np.asarray(E, np.float64)
    n = len(E)
    lab = np.zeros(n, np.int64)
    if n <= max_rows:
        return lab
    k = int(math.ceil(n / max_rows))
    if np.ptp(E, axis=0).max() > 0:
        lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(E).labels_.astype(np.int64)
    nxt = int(lab.max()) + 1
    work = [g for g in np.unique(lab) if (lab == g).sum() > max_rows]
    while work:
        g = work.pop()
        idx = np.flatnonzero(lab == g)
        if np.ptp(E[idx], axis=0).max() > 0:
            two = KMeans(n_clusters=2, n_init=2, random_state=seed).fit(E[idx]).labels_
        else:
            two = np.zeros(len(idx), int)
        if two.min() == two.max():                                  # identical points: split by order
            two = (np.arange(len(idx)) >= len(idx) // 2).astype(int)
        lab[idx[two == 1]] = nxt
        for h in (g, nxt):
            if (lab == h).sum() > max_rows:
                work.append(h)
        nxt += 1
    _, lab = np.unique(lab, return_inverse=True)
    return np.asarray(lab, np.int64).reshape(-1)


def retrieve_context(Etr: np.ndarray, labelled: np.ndarray, centroid: np.ndarray, strata: np.ndarray, budget: int,
                     seed: int, how: str = "hybrid", near_frac: float = 0.5) -> np.ndarray:
    """Sorted indices of the training rows used as context: all labelled rows if they fit in ``budget``; else
    ``near_frac`` of the budget nearest to ``centroid`` and the rest a stratified random sample of the others."""
    if how not in RETRIEVALS:
        raise ValueError(how)
    cand = np.flatnonzero(np.asarray(labelled, bool))
    if len(cand) <= budget:
        return cand
    n_near = {"hybrid": int(round(near_frac * budget)), "knn": budget, "random": 0}[how]
    near = np.zeros(0, np.int64)
    if n_near:
        d = ((np.asarray(Etr, np.float64)[cand] - np.asarray(centroid, np.float64)) ** 2).sum(1)
        near = cand[np.argsort(d, kind="stable")[:n_near]]
    rest = np.setdiff1d(cand, near)
    n_rand = budget - len(near)
    pick = rest[XD.stratified_subsample(np.asarray(strata)[rest], n_rand, seed)] if n_rand > 0 else rest[:0]
    return np.sort(np.concatenate([near, pick]))


def tabpfn_regressor(device: str = "cuda", n_jobs: int = 6):
    """The comparator's TabPFN v2 call (same code path and parameters as TabPFN[c+ctx])."""
    def run(Xa, ya, Xb, seed):
        return XD._fit_learner("tabpfn", Xa, ya, Xb, seed, n_jobs, device=device)
    return run


def hybrid_predict(Xtr: np.ndarray, Etr: np.ndarray, ytr: np.ndarray, strata: np.ndarray, Xq: np.ndarray,
                   Eq: np.ndarray, *, budget: int = HYBRID["budget"], group_rows: int = HYBRID["group_rows"],
                   seed: int = 0, regressor=None, retrieval: str = "hybrid", groups: np.ndarray | None = None,
                   return_info: bool = False):
    """Predictions of one target for the query rows.

    ``Xtr`` / ``Xq``: assembled features; ``Etr`` / ``Eq``: embeddings used for grouping and retrieval (they are
    standardised with training-row statistics here); ``ytr``: training labels (NaN = unlabelled, never in a
    context); ``strata``: source × region labels of the training rows."""
    Xtr, Xq = np.asarray(Xtr, np.float32), np.asarray(Xq, np.float32)
    ytr = np.asarray(ytr, float)
    mu, sd = np.asarray(Etr, np.float64).mean(0), np.asarray(Etr, np.float64).std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Zt, Zq = (np.asarray(Etr, np.float64) - mu) / sd, (np.asarray(Eq, np.float64) - mu) / sd
    lab = group_queries(Zq, group_rows, seed) if groups is None else np.asarray(groups, np.int64)
    regressor = regressor or tabpfn_regressor()
    labelled = np.isfinite(ytr)
    pred = np.full(len(Xq), np.nan)
    info = []
    if not labelled.any():
        return (pred, info) if return_info else pred
    for g in np.unique(lab):
        q = np.flatnonzero(lab == g)
        ctx = retrieve_context(Zt, labelled, Zq[q].mean(0), strata, budget, seed + int(g), retrieval)
        pred[q] = np.asarray(regressor(Xtr[ctx], ytr[ctx], Xq[q], seed), float)
        info.append({"group": int(g), "n_query": int(len(q)), "n_context": int(len(ctx))})
    return (pred, info) if return_info else pred


__all__ = ["HYBRID", "RETRIEVALS", "HellingerFeaturizer", "assemble", "group_queries", "retrieve_context",
           "tabpfn_regressor", "hybrid_predict"]
