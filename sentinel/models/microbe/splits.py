# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Cross-validation schemes for SENTINEL-Microbe (protocol §6) and the frozen-split record.

All schemes hold out *whole* groups: a NEON domain (primary), a site, or a domain x primer set.
Sites never straddle a training and a held-out fold (checked by :func:`check_fold`). The
development / holdout partition is defined in :mod:`sentinel.models.microbe.design`; the schemes
below only ever see development rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.eval.splits import assert_disjoint


@dataclass
class Fold:
    name: str
    train: np.ndarray          # sample ids
    test: np.ndarray           # sample ids

    def to_dict(self) -> dict:
        return {"name": self.name, "train": sorted(map(str, self.train)), "test": sorted(map(str, self.test))}


def check_fold(samples: pd.DataFrame, fold: Fold, group_col: str = "site_id") -> None:
    assert_disjoint(train=fold.train, test=fold.test)
    assert_disjoint(train_sites=samples.loc[fold.train, group_col], test_sites=samples.loc[fold.test, group_col])


def lodo(samples: pd.DataFrame, min_test: int = 1) -> list[Fold]:
    """Leave-one-NEON-domain-out (one fold per domain present in ``samples``)."""
    folds = []
    for g in sorted(samples["domain_id"].astype(str).unique()):
        te = samples.index[samples["domain_id"].astype(str) == g].to_numpy()
        tr = samples.index[samples["domain_id"].astype(str) != g].to_numpy()
        if len(te) >= min_test:
            folds.append(Fold(f"domain={g}", tr, te))
    return folds


def loso(samples: pd.DataFrame) -> list[Fold]:
    """Leave-one-site-out."""
    folds = []
    for s in sorted(samples["site_id"].astype(str).unique()):
        m = samples["site_id"].astype(str) == s
        folds.append(Fold(f"site={s}", samples.index[~m].to_numpy(), samples.index[m].to_numpy()))
    return folds


def primer_out(samples: pd.DataFrame, train_primer: str, test_primer: str, min_test: int = 5) -> list[Fold]:
    """Leave-one-domain-out x primer transfer: train on ``train_primer`` rows outside domain g,
    test on ``test_primer`` rows inside g. Both a primer *and* a domain shift (and, because NEON's
    primer change is confounded with time, a time shift)."""
    folds = []
    p = samples["primer_set"].astype(str)
    dom = samples["domain_id"].astype(str)
    for g in sorted(dom.unique()):
        te = samples.index[(dom == g) & (p == test_primer)].to_numpy()
        tr = samples.index[(dom != g) & (p == train_primer)].to_numpy()
        if len(te) >= min_test and len(tr):
            folds.append(Fold(f"domain={g}|{train_primer}->{test_primer}", tr, te))
    return folds


def release_out(samples: pd.DataFrame, train_src: str = "DP1.20141.002", test_src: str = "DP1.20141.001",
                min_test: int = 5) -> list[Fold]:
    """Leave-one-domain-out x release transfer: train on ``train_src`` events outside domain g, test on
    ``test_src`` events (older bioinformatics pipeline: different ASV/OTU definitions) inside g."""
    folds = []
    src = samples["source_product"].astype(str)
    dom = samples["domain_id"].astype(str)
    for g in sorted(dom.unique()):
        te = samples.index[(dom == g) & (src == test_src)].to_numpy()
        tr = samples.index[(dom != g) & (src == train_src)].to_numpy()
        if len(te) >= min_test and len(tr):
            folds.append(Fold(f"domain={g}|{train_src}->{test_src}", tr, te))
    return folds


def lodo_enlarged(samples: pd.DataFrame, test_src: str = "DP1.20141.002", min_test: int = 1) -> list[Fold]:
    """Enlarged-training LODO (protocol §10.2, cycle 2): test = ``test_src`` (.002) development events
    of domain g — exactly the primary LODO test rows — and train = every development event outside g
    from BOTH releases (the .001 events of 2017-01 -> 2018-10; the pre-2017 cross-release reserve is
    removed by ``load_design``). A training-set enlargement, not a new evaluation set."""
    folds = []
    src = samples["source_product"].astype(str)
    dom = samples["domain_id"].astype(str)
    for g in sorted(dom.unique()):
        te = samples.index[(dom == g) & (src == test_src)].to_numpy()
        tr = samples.index[dom != g].to_numpy()
        if len(te) >= min_test and len(tr):
            folds.append(Fold(f"domain={g}|all->{test_src}", tr, te))
    return folds


def random_kfold(samples: pd.DataFrame, k: int = 5, seed: int = 0) -> list[Fold]:
    """Row-random k-fold — sites straddle folds. Leakage-gap context and site-identity probe ONLY."""
    rs = np.random.default_rng(seed)
    f = rs.permutation(np.arange(len(samples)) % k)
    return [Fold(f"random{k}={i}", samples.index[f != i].to_numpy(), samples.index[f == i].to_numpy())
            for i in range(k)]


SCHEMES = {
    "lodo": lambda s: lodo(s),
    "loso": lambda s: loso(s),
    "primer_v4v5_to_v3v4": lambda s: primer_out(s, "V4V5", "V3V4"),
    "primer_v3v4_to_v4v5": lambda s: primer_out(s, "V3V4", "V4V5"),
    "random5": lambda s: random_kfold(s, 5, 0),
    "release_002_to_001": lambda s: release_out(s),
    "release_001_to_002": lambda s: release_out(s, "DP1.20141.001", "DP1.20141.002"),
    "lodo_x001": lambda s: lodo_enlarged(s),
}
GROUPED = {"lodo", "loso", "primer_v4v5_to_v3v4", "primer_v3v4_to_v4v5", "release_002_to_001",
           "release_001_to_002", "lodo_x001"}


def make_folds(samples: pd.DataFrame, scheme: str) -> list[Fold]:
    folds = SCHEMES[scheme](samples)
    if scheme in GROUPED:
        for f in folds:
            check_fold(samples, f)
    return folds


def inner_validation(samples: pd.DataFrame, train_ids, frac: float = 0.15, seed: int = 0,
                     min_sites: int = 2, unit: str = "site") -> tuple[np.ndarray, np.ndarray]:
    """Split a fold's *training* rows into (fit, early-stopping validation) by whole groups.

    ``unit='domain'`` (domain hold-out schemes) holds out whole training domains so early stopping
    mimics the outer shift (review O5); ``unit='site'`` (site hold-out) whole sites."""
    col = {"site": "site_id", "domain": "domain_id"}[unit]
    tr = samples.loc[pd.Index(train_ids)]
    groups = np.array(sorted(tr[col].astype(str).unique()))
    rs = np.random.default_rng(seed)
    n_val = max(min_sites, int(round(frac * len(groups))))
    n_val = min(n_val, max(len(groups) - 1, 0))
    val_groups = set(rs.choice(groups, n_val, replace=False)) if n_val else set()
    is_val = tr[col].astype(str).isin(val_groups).to_numpy()
    return tr.index[~is_val].to_numpy(), tr.index[is_val].to_numpy()


def split_record(samples_dev: pd.DataFrame, samples_holdout: pd.DataFrame, holdout_rule: dict,
                 schemes=("lodo", "loso")) -> dict:
    """Serializable record of the development/holdout partition and every development fold."""
    rec = {
        "holdout_rule": holdout_rule,
        "development_ids": sorted(map(str, samples_dev.index)),
        "holdout_ids": sorted(map(str, samples_holdout.index)),
        "n_development": len(samples_dev),
        "n_holdout": len(samples_holdout),
        "schemes": {s: [f.to_dict() for f in make_folds(samples_dev, s)] for s in schemes},
    }
    assert_disjoint(development=rec["development_ids"], holdout=rec["holdout_ids"])
    return rec


def write_frozen_split(rec: dict, path: str | Path) -> str:
    """Write the split JSON (sorted keys) and return its sha256."""
    blob = json.dumps(rec, sort_keys=True, indent=1).encode()
    Path(path).write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()
