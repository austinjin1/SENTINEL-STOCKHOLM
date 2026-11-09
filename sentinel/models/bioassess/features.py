# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tabular community features fit on the training fold only.

``CommunityFeaturizer.fit(design, train_ids)`` chooses the harmonised taxa (``mixed_otu``) present in
>= ``min_samples`` *training* samples; ``transform(design, ids)`` returns Hellinger-transformed
relative abundances (relative to each sample's full count, so dropped rare taxa do not renormalise
the rest) and, optionally, the three sample descriptors. Chemistry never enters (inputs come only
from ``design.tokens`` and the descriptor columns of ``design.samples``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DESCRIPTORS = ("log10_total_count", "log10_n_taxa", "boatable")
GEOGRAPHY = ("lat", "lon", "elevation_m")
SEASON = ("doy_sin", "doy_cos")          # sampling day-of-year (metadata)
CONTEXT = GEOGRAPHY + SEASON             # the RESEARCH_PLAN §14.3 trivial context (location + DOY)


def abundance_matrix(tokens: pd.DataFrame, ids, key: str = "mixed_otu") -> pd.DataFrame:
    t = tokens[tokens["visit_id"].isin(pd.Index(ids))]
    m = t.pivot_table(index="visit_id", columns=key, values="count", aggfunc="sum", fill_value=0.0)
    return m.reindex(pd.Index(ids, name="visit_id")).fillna(0.0)


@dataclass
class CommunityFeaturizer:
    min_samples: int = 10
    key: str = "mixed_otu"
    descriptors: bool = True
    geography: bool = False
    season: bool = False              # add sampling day-of-year (sin/cos)
    max_taxa: int | None = None       # keep the most prevalent (training) taxa only
    taxa_: list = field(default_factory=list)
    geo_fill_: dict = field(default_factory=dict)

    def fit(self, design, train_ids) -> "CommunityFeaturizer":
        A = abundance_matrix(design.tokens, train_ids, self.key)
        prev = (A > 0).sum(0)
        prev = prev[prev >= self.min_samples].sort_values(ascending=False, kind="stable")
        if self.max_taxa:
            prev = prev.iloc[: self.max_taxa]
        self.taxa_ = sorted(prev.index)
        if self.geography:
            s = design.samples.loc[pd.Index(train_ids)]
            self.geo_fill_ = {c: float(s[c].median()) for c in GEOGRAPHY}
        return self

    def feature_names(self) -> list[str]:
        names = [f"taxon::{t}" for t in self.taxa_]
        if self.descriptors:
            names += list(DESCRIPTORS)
        if self.geography:
            names += list(GEOGRAPHY)
        if self.season:
            names += list(SEASON)
        return names

    def transform(self, design, ids) -> np.ndarray:
        ids = pd.Index(ids)
        A = abundance_matrix(design.tokens, ids, self.key)
        tot = A.sum(1).replace(0, np.nan)
        H = np.sqrt(A.reindex(columns=self.taxa_, fill_value=0.0).div(tot, axis=0).fillna(0.0)).to_numpy(float)
        parts = [H]
        s = design.samples.loc[ids]
        if self.descriptors:
            parts.append(s[list(DESCRIPTORS)].to_numpy(float))
        if self.geography:
            parts.append(s[list(GEOGRAPHY)].fillna(self.geo_fill_).to_numpy(float))
        if self.season:
            parts.append(s[list(SEASON)].to_numpy(float))
        return np.hstack(parts)


def geography_matrix(design, train_ids, ids, season: bool = False) -> np.ndarray:
    s_tr = design.samples.loc[pd.Index(train_ids)]
    fill = {c: float(s_tr[c].median()) for c in GEOGRAPHY}
    X = design.samples.loc[pd.Index(ids), list(GEOGRAPHY)].fillna(fill).to_numpy(float)
    if season:
        X = np.hstack([X, design.samples.loc[pd.Index(ids), list(SEASON)].to_numpy(float)])
    return X


class Standardiser:
    """Column mean/std fit on training rows (NaN-aware); std 0 -> 1."""

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.std_ = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.std_

    def inverse(self, Z):
        return np.asarray(Z, float) * self.std_ + self.mean_
