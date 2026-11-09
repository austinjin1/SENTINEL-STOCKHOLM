# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Leakage-safe data splits.

v1 failures this module exists to prevent: a "strict spatial holdout" that was a random
station split, calibration and evaluation on the same windows, and pretraining stations
reappearing in "zero-shot" tests. Every split returned here can be checked with
:func:`assert_disjoint` and summarised with :func:`split_report`, and every experiment
must save that report next to its results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, float)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_distance_km(lat_a, lon_a, lat_b, lon_b, chunk: int = 2048) -> np.ndarray:
    """For each point in A, the great-circle distance to the nearest point in B."""
    lat_a, lon_a = np.asarray(lat_a, float), np.asarray(lon_a, float)
    lat_b, lon_b = np.asarray(lat_b, float), np.asarray(lon_b, float)
    out = np.empty(lat_a.size)
    for s in range(0, lat_a.size, chunk):
        d = haversine_km(lat_a[s:s + chunk, None], lon_a[s:s + chunk, None], lat_b[None], lon_b[None])
        out[s:s + chunk] = d.min(axis=1)
    return out


def spatial_blocks(lat, lon, block_km: float) -> np.ndarray:
    """Assign points to ~square blocks of side ``block_km`` (equirectangular, per-row cos(lat))."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    row = np.floor((lat + 90.0) * (np.pi * EARTH_RADIUS_KM / 180.0) / block_km).astype(int)
    row_lat = np.radians((row + 0.5) * block_km / (np.pi * EARTH_RADIUS_KM / 180.0) - 90.0)
    km_per_deg_lon = np.pi * EARTH_RADIUS_KM / 180.0 * np.maximum(np.cos(row_lat), 1e-3)
    col = np.floor((lon + 180.0) * km_per_deg_lon / block_km).astype(int)
    return np.array([f"{r}_{c}" for r, c in zip(row, col)])


@dataclass
class SiteSplit:
    """Assignment of sites (or lakes, studies, chemicals …) to train/val/test."""

    train: list
    val: list
    test: list
    method: str
    params: dict = field(default_factory=dict)

    def role(self, site) -> str | None:
        if site in self._lookup:
            return self._lookup[site]
        return None

    def __post_init__(self):
        assert_disjoint(train=self.train, val=self.val, test=self.test)
        self._lookup = {s: "train" for s in self.train}
        self._lookup.update({s: "val" for s in self.val})
        self._lookup.update({s: "test" for s in self.test})

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params, "train": list(self.train),
                "val": list(self.val), "test": list(self.test)}


def group_split(groups: Sequence, *, test_frac: float, val_frac: float, seed: int,
                weights: dict | None = None) -> SiteSplit:
    """Random split over *groups* (whole groups go to one side). Optionally balance by weight
    (e.g. number of samples per group) instead of by group count."""
    uniq = pd.unique(pd.Series(list(groups)))
    rs = np.random.default_rng(seed)
    order = rs.permutation(len(uniq))
    uniq = uniq[order]
    w = np.array([weights.get(g, 1.0) if weights else 1.0 for g in uniq], float)
    cum = np.cumsum(w) / w.sum()
    test = list(uniq[cum <= test_frac]) or [uniq[0]]
    rest = [g for g in uniq if g not in set(test)]
    w_rest = np.array([weights.get(g, 1.0) if weights else 1.0 for g in rest], float)
    cum_rest = np.cumsum(w_rest) / max(w.sum(), 1e-12)
    val = [g for g, c in zip(rest, cum_rest) if c <= val_frac] or [rest[0]]
    train = [g for g in rest if g not in set(val)]
    return SiteSplit(train, val, test, "group_random",
                     {"test_frac": test_frac, "val_frac": val_frac, "seed": seed,
                      "weighted": weights is not None})


def spatial_block_split(sites: Sequence, lat, lon, *, block_km: float, test_frac: float,
                        val_frac: float, seed: int, buffer_km: float = 0.0,
                        site_weights: dict | None = None) -> SiteSplit:
    """Hold out whole spatial blocks; drop train/val sites within ``buffer_km`` of any test site.

    Dropped buffer sites are recorded in ``params['buffer_dropped']`` (they are excluded,
    not silently moved to another role).
    """
    sites = list(sites)
    blocks = spatial_blocks(lat, lon, block_km)
    block_of = dict(zip(sites, blocks))
    bw = None
    if site_weights:
        bw = {}
        for s, b in block_of.items():
            bw[b] = bw.get(b, 0.0) + site_weights.get(s, 1.0)
    bsplit = group_split(blocks, test_frac=test_frac, val_frac=val_frac, seed=seed, weights=bw)
    role = {b: r for r, bl in (("train", bsplit.train), ("val", bsplit.val), ("test", bsplit.test))
            for b in bl}
    train = [s for s in sites if role[block_of[s]] == "train"]
    val = [s for s in sites if role[block_of[s]] == "val"]
    test = [s for s in sites if role[block_of[s]] == "test"]
    dropped: list = []
    if buffer_km > 0 and test:
        coords = dict(zip(sites, zip(np.asarray(lat, float), np.asarray(lon, float))))
        tlat = np.array([coords[s][0] for s in test])
        tlon = np.array([coords[s][1] for s in test])

        def keep(group):
            if not group:
                return group, []
            glat = np.array([coords[s][0] for s in group])
            glon = np.array([coords[s][1] for s in group])
            d = nearest_distance_km(glat, glon, tlat, tlon)
            return [s for s, di in zip(group, d) if di > buffer_km], \
                   [s for s, di in zip(group, d) if di <= buffer_km]

        train, d1 = keep(train)
        val, d2 = keep(val)
        dropped = d1 + d2
    return SiteSplit(train, val, test, "spatial_block",
                     {"block_km": block_km, "test_frac": test_frac, "val_frac": val_frac,
                      "seed": seed, "buffer_km": buffer_km, "buffer_dropped": dropped})


def spatial_kfold(sites: Sequence, lat, lon, *, n_folds: int, block_km: float, seed: int) -> dict:
    """Map each site to a fold id by assigning whole spatial blocks to folds (balanced by sites)."""
    sites = list(sites)
    blocks = spatial_blocks(lat, lon, block_km)
    counts = pd.Series(blocks).value_counts()
    rs = np.random.default_rng(seed)
    # Largest blocks first, each to the currently smallest fold (random tie-break).
    order = counts.sample(frac=1.0, random_state=int(rs.integers(1 << 31))).sort_values(
        ascending=False, kind="stable")
    load = np.zeros(n_folds)
    fold_of_block = {}
    for b, c in order.items():
        f = int(rs.choice(np.flatnonzero(load == load.min())))
        fold_of_block[b] = f
        load[f] += c
    return {s: fold_of_block[b] for s, b in zip(sites, blocks)}


@dataclass
class TimeSplit:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    embargo: pd.Timedelta

    def masks(self, times) -> dict[str, np.ndarray]:
        """Train: t < train_end - embargo; val: train_end <= t < val_end - embargo; test: t >= val_end."""
        t = pd.to_datetime(pd.Series(times)).to_numpy()
        tr = t < (self.train_end - self.embargo).to_datetime64()
        va = (t >= self.train_end.to_datetime64()) & (t < (self.val_end - self.embargo).to_datetime64())
        te = t >= self.val_end.to_datetime64()
        return {"train": tr, "val": va, "test": te}

    def to_dict(self) -> dict:
        return {"train_end": str(self.train_end), "val_end": str(self.val_end),
                "embargo": str(self.embargo)}


def time_split(train_end: str, val_end: str, embargo: str = "0D") -> TimeSplit:
    ts = TimeSplit(pd.Timestamp(train_end), pd.Timestamp(val_end), pd.Timedelta(embargo))
    if not ts.train_end < ts.val_end:
        raise ValueError("train_end must precede val_end")
    return ts


def assert_disjoint(**named_sets: Iterable) -> None:
    """Raise AssertionError if any two of the named collections share an element."""
    items = {k: set(v) for k, v in named_sets.items()}
    keys = list(items)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            both = items[a] & items[b]
            if both:
                sample = sorted(map(str, both))[:5]
                raise AssertionError(f"LEAK: {len(both)} items in both {a} and {b}, e.g. {sample}")


def split_report(split: SiteSplit, lat_of: dict | None = None, lon_of: dict | None = None) -> dict:
    """Counts per role and, if coordinates are given, nearest train->test distances."""
    rep = {"method": split.method, "params": {k: v for k, v in split.params.items()
                                              if k != "buffer_dropped"},
           "n_train": len(split.train), "n_val": len(split.val), "n_test": len(split.test),
           "n_buffer_dropped": len(split.params.get("buffer_dropped", []))}
    if lat_of and lon_of and split.train and split.test:
        d = nearest_distance_km([lat_of[s] for s in split.test], [lon_of[s] for s in split.test],
                                [lat_of[s] for s in split.train], [lon_of[s] for s in split.train])
        rep.update({"test_to_train_km_min": float(d.min()),
                    "test_to_train_km_median": float(np.median(d))})
    return rep
