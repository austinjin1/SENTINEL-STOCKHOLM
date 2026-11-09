# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""k-nearest-neighbour **read-across**: the classical comparator any QSAR claim must beat.

Regulatory read-across predicts a chemical's toxicity from structurally similar chemicals tested on
the same species. It is the oldest method in this field, it needs no training, and it is *not* a
trivial baseline — on aquatic toxicity it is often within a few percent of tuned tree models. The
ADORE phase-1 tables did not contain it; the prospective pre-registration requires it
(``docs/preregistration/chemtox_prospective_plan.md`` §3.6), so it lives here.

Prediction for a query row:

1. candidate neighbours are the training rows that match the query on **species** (``tax_gs``);
   if fewer than ``min_neighbours`` match, the pool widens to the same **taxon group**, then to all
   training rows (the widening level is recorded);
2. within the pool, similarity is Tanimoto on the query's fingerprint bits (Morgan by default,
   MACCS optionally), optionally multiplied by a duration-agreement factor
   (``duration_penalty``: 1 for the same observation duration, ``duration_penalty`` otherwise);
3. the prediction is the similarity-weighted mean of the ``k`` most similar neighbours' targets,
   with weights ``sim ** power``; neighbours below ``min_similarity`` are dropped, and if nothing
   survives the pool's (or the global) mean is used.

The model reads the raw rows through the ChemTox model protocol's ``train_rows``/``rows`` context,
so it does not depend on the design matrix at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FP_COLUMNS = {"Morgan": "chem_Morgan_fp", "MACCS": "chem_MACCS_fp"}
DURATION_COL = "result_obs_duration_mean"
SPECIES_COL = "tax_gs"
GROUP_COL = "tax_group"


def bits_matrix(series: pd.Series) -> np.ndarray:
    """Bit-string column → uint8 matrix (rows with a missing string become all-zero)."""
    s = series.astype("string").fillna("")
    width = int(s.str.len().max() or 0)
    out = np.zeros((len(s), width), dtype=np.uint8)
    for i, v in enumerate(s.to_numpy()):
        if not v or len(v) != width:
            continue
        out[i] = np.frombuffer(v.encode("ascii"), dtype=np.uint8) - ord("0")
    return out


def tanimoto(query: np.ndarray, ref: np.ndarray, ref_popcount: np.ndarray | None = None
             ) -> np.ndarray:
    """Tanimoto similarity of one bit vector against a (n, width) reference matrix.

    ``ref`` should already be float32 and ``ref_popcount`` its row sums; both are precomputed once
    per fit, because recasting the whole training pool on every query dominated the runtime.
    """
    q = query.astype(np.float32, copy=False)
    r = ref if ref.dtype == np.float32 else ref.astype(np.float32)
    pc = r.sum(axis=1) if ref_popcount is None else ref_popcount
    inter = r @ q
    denom = pc + q.sum() - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom > 0, inter / denom, 0.0)
    return sim


@dataclass
class ReadAcrossKNN:
    """Similarity-weighted k-NN read-across over training rows (no fitted parameters)."""

    params: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        self.k = int(self.params.get("k", 5))
        self.power = float(self.params.get("power", 3.0))
        self.min_similarity = float(self.params.get("min_similarity", 0.3))
        self.min_neighbours = int(self.params.get("min_neighbours", 3))
        self.duration_penalty = float(self.params.get("duration_penalty", 1.0))
        self.fingerprint = str(self.params.get("fingerprint", "Morgan"))
        self.match_species = bool(self.params.get("match_species", True))
        self.rows_ = None
        self.bits_ = None
        self.popcount_ = None
        self.y_ = None
        self.widening_ = {"species": 0, "group": 0, "global": 0, "fallback_mean": 0}

    # -------------------------------------------------------------------- protocol
    def fit(self, X, y, train_rows=None, sample_weight=None, **_ctx) -> ReadAcrossKNN:
        """Store the training pool. ``sample_weight`` is **not** supported and raises.

        Read-across is an unweighted similarity average by definition; silently ignoring weights
        would make a weighted-augmentation configuration look like something it is not.
        """
        if train_rows is None:
            raise ValueError("ReadAcrossKNN needs train_rows (the raw training frame)")
        if sample_weight is not None and not np.allclose(np.asarray(sample_weight, dtype=float), 1.0):
            raise ValueError("ReadAcrossKNN does not support sample_weight (see the fit docstring)")
        self.rows_ = train_rows.reset_index(drop=True)
        self.y_ = np.asarray(y, dtype=float)
        self.bits_ = bits_matrix(self.rows_[FP_COLUMNS[self.fingerprint]]).astype(np.float32)
        self.popcount_ = self.bits_.sum(axis=1)
        self.species_ = self.rows_[SPECIES_COL].astype("string").fillna("<NA>").to_numpy()
        self.group_ = self.rows_[GROUP_COL].astype("string").fillna("<NA>").to_numpy()
        self.duration_ = pd.to_numeric(self.rows_[DURATION_COL], errors="coerce").to_numpy()
        self.global_mean_ = float(np.nanmean(self.y_))
        return self

    def predict(self, X, rows=None, **_ctx) -> np.ndarray:
        if rows is None:
            raise ValueError("ReadAcrossKNN needs rows (the raw frame being predicted)")
        rows = rows.reset_index(drop=True)
        qbits = bits_matrix(rows[FP_COLUMNS[self.fingerprint]])
        qspecies = rows[SPECIES_COL].astype("string").fillna("<NA>").to_numpy()
        qgroup = rows[GROUP_COL].astype("string").fillna("<NA>").to_numpy()
        qdur = pd.to_numeric(rows[DURATION_COL], errors="coerce").to_numpy()
        out = np.empty(len(rows), dtype=float)
        for i in range(len(rows)):
            pool = np.ones(len(self.rows_), dtype=bool)
            level = "global"
            if self.match_species:
                m = self.species_ == qspecies[i]
                if m.sum() >= self.min_neighbours:
                    pool, level = m, "species"
                else:
                    m = self.group_ == qgroup[i]
                    if m.sum() >= self.min_neighbours:
                        pool, level = m, "group"
            self.widening_[level] += 1
            idx = np.flatnonzero(pool)
            sim = tanimoto(qbits[i], self.bits_[idx], self.popcount_[idx])
            if self.duration_penalty < 1.0:
                same = self.duration_[idx] == qdur[i]
                sim = sim * np.where(same, 1.0, self.duration_penalty)
            keep = sim >= self.min_similarity
            if not keep.any():
                out[i] = float(np.nanmean(self.y_[idx])) if idx.size else self.global_mean_
                self.widening_["fallback_mean"] += 1
                continue
            s = sim[keep]
            yy = self.y_[idx[keep]]
            if s.size > self.k:
                top = np.argpartition(-s, self.k - 1)[: self.k]
                s, yy = s[top], yy[top]
            w = np.power(s, self.power)
            out[i] = float(np.sum(w * yy) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(yy))
        return out


def make_readacross(seed: int = 0):
    return lambda params: ReadAcrossKNN(params=params, seed=seed)
