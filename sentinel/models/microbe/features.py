# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Training-fold tabular features for SENTINEL-Microbe baselines (protocol §4).

Every transform here is *fit on training rows only* and then applied to held-out rows:

* :class:`GenusCLR` — genus count matrix -> prevalence filter (genera present in >= ``min_prev``
  of training samples, at most ``max_features`` most prevalent) + one pooled "other" column ->
  centred log-ratio with pseudocount (default 0.5). Presence/absence variant for the P/A baseline.
* :func:`context_features` — day-of-year harmonics, latitude, longitude, elevation, aquatic site
  type (stream / river / lake one-hot). These are the tier-0 "season + location" inputs; they are
  never mixed into the community-only models.

The token table (long format) has columns ``sample_id``, ``seq_id``, ``count`` and taxonomy
ranks (``genus`` etc.); the sample table is indexed by ``sample_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UNASSIGNED = "__unassigned__"
OTHER = "__other__"
SITE_TYPES = ("stream", "river", "lake")
CONTEXT_COLS = ("doy_sin", "doy_cos", "latitude", "longitude", "elevation",
                "type_stream", "type_river", "type_lake")


_GC_CACHE: dict = {}


def _all_genus_counts(tokens: pd.DataFrame, rank: str) -> pd.DataFrame:
    """Per-sample genus counts for every sample in ``tokens`` (row-wise, label-free -> computing it
    once for all rows and subsetting is identical to computing it per subset). Cached per table."""
    key = (id(tokens), len(tokens), rank)
    if key not in _GC_CACHE:
        g = tokens[rank].fillna(UNASSIGNED).astype(str).replace({"": UNASSIGNED})
        m = tokens.assign(_g=g).groupby(["sample_id", "_g"])["count"].sum().unstack("_g", fill_value=0)
        _GC_CACHE.clear()
        _GC_CACHE[key] = m.astype(float)
    return _GC_CACHE[key]


def genus_counts(tokens: pd.DataFrame, sample_ids, rank: str = "genus") -> pd.DataFrame:
    """[samples x genera] read counts (unassigned genus -> ``__unassigned__``); genera absent from the
    requested samples are dropped, so the result equals a per-subset computation."""
    m = _all_genus_counts(tokens, rank).reindex(pd.Index(sample_ids)).fillna(0.0)
    return m.loc[:, m.sum(0) > 0]


def clr(counts: np.ndarray, pseudo: float = 0.5) -> np.ndarray:
    x = np.log(np.asarray(counts, float) + pseudo)
    return x - x.mean(1, keepdims=True)


@dataclass
class GenusCLR:
    min_prev: float = 0.05
    max_features: int = 500
    pseudo: float = 0.5
    rank: str = "genus"
    keep_: list = field(default_factory=list)

    def fit(self, tokens: pd.DataFrame, train_ids) -> GenusCLR:
        C = genus_counts(tokens, train_ids, self.rank)
        C = C.drop(columns=[UNASSIGNED], errors="ignore")
        prev = (C > 0).mean(0)
        prev = prev[prev >= self.min_prev]
        self.keep_ = list(prev.sort_values(ascending=False, kind="stable").index[: self.max_features])
        if not self.keep_:
            raise ValueError("no genus passes the training-fold prevalence filter")
        return self

    def counts(self, tokens: pd.DataFrame, ids) -> np.ndarray:
        C = genus_counts(tokens, ids, self.rank)
        kept = C.reindex(columns=self.keep_, fill_value=0.0)
        other = C.sum(1) - kept.sum(1)
        return np.column_stack([kept.to_numpy(float), other.to_numpy(float)])

    def transform(self, tokens: pd.DataFrame, ids, kind: str = "clr") -> np.ndarray:
        X = self.counts(tokens, ids)
        if kind == "clr":
            return clr(X, self.pseudo).astype(np.float32)
        if kind == "pa":
            return (X[:, :-1] > 0).astype(np.float32)
        if kind == "relab":
            return (X / np.maximum(X.sum(1, keepdims=True), 1e-12)).astype(np.float32)
        raise ValueError(kind)

    @property
    def names(self) -> list[str]:
        return list(self.keep_) + [OTHER]


def context_features(samples: pd.DataFrame, ids, fill: dict | None = None) -> tuple[np.ndarray, dict]:
    """Tier-0 context matrix; missing elevation filled with the *training-fold* median (``fill``).

    Returns (X, fill) — call with ``fill=None`` on training rows, then pass the returned ``fill`` for
    held-out rows.
    """
    s = samples.loc[pd.Index(ids)]
    doy = pd.to_datetime(s["collect_date"]).dt.dayofyear.to_numpy(float)
    ang = 2 * np.pi * (doy - 1) / 365.25
    num = s[["latitude", "longitude", "elevation"]].astype(float)
    if fill is None:
        fill = {c: float(num[c].median()) for c in num.columns}
    num = num.fillna(fill)
    typ = s["aquatic_type"].astype(str).str.lower()
    X = np.column_stack([np.sin(ang), np.cos(ang), num.to_numpy(float)]
                        + [(typ == t).to_numpy(float) for t in SITE_TYPES])
    return X.astype(np.float32), fill


def sample_descriptors(samples: pd.DataFrame, ids, primer_onehot: bool = True, enabled: bool = True) -> np.ndarray:
    """Community-side sample descriptors: log10 reads, log10 richness (+ primer-set one-hot).

    ``enabled=False`` (cross-release scheme, protocol §6) returns no columns: sequencing depth and
    richness differ ~10x between the two NEON pipelines, so they would encode the release."""
    if not enabled:
        return np.zeros((len(ids), 0), np.float32)
    s = samples.loc[pd.Index(ids)]
    cols = [np.log10(np.maximum(s["n_reads"].to_numpy(float), 1.0)),
            np.log10(np.maximum(s["n_seqs"].to_numpy(float), 1.0))]
    if primer_onehot:
        for p in ("V4", "V3V4", "V4V5"):
            cols.append((s["primer_set"].astype(str) == p).to_numpy(float))
    return np.column_stack(cols).astype(np.float32)
