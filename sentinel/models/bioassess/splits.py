# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Cross-validation schemes over *location clusters* (protocol §6).

Every fold is a :class:`sentinel.eval.splits.SiteSplit` whose members are ``location_id`` values
(1-km clusters over all NRSA cycles), so within-cycle revisits (same ``site_id``) and cross-cycle
re-sampled locations (different ``site_id``, same place) can never straddle train and test.
:func:`check_folds` verifies this on locations, sites and visits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.eval import splits as ES

SCHEME_SPLIT_NAME = {"loeo": "nrsa0814/loeo_cv", "block50": "nrsa0814/block50_cv",
                     "random5": "nrsa0814/random5_cv"}


def _locations(samples: pd.DataFrame) -> pd.DataFrame:
    """One row per location cluster: centroid and its (unique) ecoregion."""
    g = samples.groupby("location_id")
    loc = g.agg(lat=("lat", "mean"), lon=("lon", "mean"), ag_eco9=("ag_eco9", "first"),
                n_eco=("ag_eco9", "nunique"))
    if (loc["n_eco"] > 1).any():
        raise AssertionError(f"{int((loc['n_eco'] > 1).sum())} location clusters span >1 ecoregion")
    return loc.sort_index()


def loeo_folds(samples: pd.DataFrame, group_col: str = "ag_eco9") -> list[ES.SiteSplit]:
    loc = _locations(samples)
    out = []
    for g in sorted(loc[group_col].unique()):
        test = sorted(loc.index[loc[group_col] == g])
        train = sorted(loc.index[loc[group_col] != g])
        out.append(ES.SiteSplit(train=train, val=[], test=test, method="leave_one_group_out",
                                params={"group_column": group_col, "held_out": g, "unit": "location_id"}))
    return out


def block_folds(samples: pd.DataFrame, block_km: float = 50.0, n_folds: int = 10, seed: int = 0) -> list[ES.SiteSplit]:
    loc = _locations(samples)
    fold_of = ES.spatial_kfold(list(loc.index), loc["lat"].to_numpy(), loc["lon"].to_numpy(),
                               n_folds=n_folds, block_km=block_km, seed=seed)
    f = pd.Series(fold_of)
    out = []
    for k in range(n_folds):
        test = sorted(f.index[f == k])
        train = sorted(f.index[f != k])
        out.append(ES.SiteSplit(train=train, val=[], test=test, method="spatial_block_kfold",
                                params={"block_km": block_km, "n_folds": n_folds, "fold": k, "seed": seed,
                                        "unit": "location_id"}))
    return out


def random_folds(samples: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> list[ES.SiteSplit]:
    uniq = np.array(sorted(samples["location_id"].unique()))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    parts = np.array_split(uniq[perm], n_folds)
    out = []
    for k, p in enumerate(parts):
        test = sorted(p.tolist())
        train = sorted(set(uniq) - set(test))
        out.append(ES.SiteSplit(train=train, val=[], test=test, method="random_location_kfold",
                                params={"n_folds": n_folds, "fold": k, "seed": seed, "unit": "location_id"}))
    return out


def make_folds(samples: pd.DataFrame, scheme: str) -> list[ES.SiteSplit]:
    if scheme == "loeo":
        return loeo_folds(samples)
    if scheme == "block50":
        return block_folds(samples, 50.0, 10, 0)
    if scheme == "random5":
        return random_folds(samples, 5, 0)
    raise ValueError(scheme)


def inner_validation(samples: pd.DataFrame, train_locations, frac: float = 0.15, block_km: float = 50.0,
                     seed: int = 0) -> tuple[list, list]:
    """Split a training fold's locations into (fit, early-stopping validation) by whole 50-km blocks."""
    loc = _locations(samples[samples["location_id"].isin(set(train_locations))])
    sp = ES.spatial_block_split(list(loc.index), loc["lat"].to_numpy(), loc["lon"].to_numpy(),
                                block_km=block_km, test_frac=frac, val_frac=0.0, seed=seed)
    # group_split always puts >= 1 group in val; fold it back into fit (val_frac = 0 requested)
    fit = sorted(set(sp.train) | set(sp.val))
    return fit, sorted(sp.test)


def masks(samples: pd.DataFrame, fold: ES.SiteSplit) -> tuple[np.ndarray, np.ndarray]:
    loc = samples["location_id"].to_numpy()
    return np.isin(loc, fold.train), np.isin(loc, fold.test)


def check_folds(folds: list[ES.SiteSplit], samples: pd.DataFrame) -> None:
    """Raise on any leak: tests partition the rows; no location/site/visit in train and test."""
    tests = {f"fold{i}": f.test for i, f in enumerate(folds)}
    ES.assert_disjoint(**tests)
    if set().union(*map(set, tests.values())) != set(samples["location_id"]):
        raise AssertionError("test folds do not cover every location exactly once")
    for f in folds:
        tr, te = masks(samples, f)
        if (tr & te).any():
            raise AssertionError("row in both train and test")
        ES.assert_disjoint(train_location=f.train, test_location=f.test)
        ES.assert_disjoint(train_site=samples.loc[tr, "site_id"], test_site=samples.loc[te, "site_id"])
        ES.assert_disjoint(train_visit=samples.index[tr], test_visit=samples.index[te])


def fold_reports(folds: list[ES.SiteSplit], samples: pd.DataFrame) -> list[dict]:
    loc = _locations(samples)
    lat_of, lon_of = loc["lat"].to_dict(), loc["lon"].to_dict()
    out = []
    for f in folds:
        tr, te = masks(samples, f)
        rep = ES.split_report(f, lat_of, lon_of)
        rep.update({"n_train_rows": int(tr.sum()), "n_test_rows": int(te.sum())})
        out.append({"report": rep, "split": f.to_dict()})
    return out
