# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Official-fold cross-validation, grid search and registry logging for SENTINEL-ChemTox.

Model selection uses **only** ADORE's own CV folds 0-4 inside the training partition
(``docs/research/04`` §E.1: "model selection only on ADORE's CV folds 0-4; the test rows are
touched exactly once, and the number of configurations tried is reported"). Every configuration
evaluated is counted and returned, so the count can be reported and logged.

A "model" here is anything with ``fit(X, y, **context)`` and ``predict(X)``; ``make_model(params)``
builds one. ``context`` carries the per-fold frames (``train_rows``, ``valid_rows``) so models that
need more than the design matrix (species embeddings, SMILES) can take it.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core.registry import log_result

from . import evaluate as E
from .data import ChemToxDataset, FeatureSpec

WORKSTREAM = "chemtox"
TARGET_NAMES = {"molar": "log10_LC50_EC50_mol_per_L", "mass": "log10_LC50_EC50_mg_per_L"}


@dataclass
class CVResult:
    """Per-fold and aggregate out-of-fold scores of one configuration."""

    model: str
    spec: dict
    params: dict
    fold_metrics: list[dict] = field(default_factory=list)
    mean: dict = field(default_factory=dict)
    sd: dict = field(default_factory=dict)
    pooled: dict = field(default_factory=dict)
    oof: np.ndarray | None = None
    oof_mask: np.ndarray | None = None
    seconds: float = 0.0

    @property
    def rmse(self) -> float:
        return float(self.mean.get("rmse", np.nan))

    def to_dict(self) -> dict:
        out = {
            "model": self.model, "spec": self.spec, "params": self.params,
            "fold_metrics": self.fold_metrics, "mean": self.mean, "sd": self.sd,
            "pooled": self.pooled, "seconds": round(self.seconds, 2),
        }
        return out


def run_cv(
    ds: ChemToxDataset,
    spec: FeatureSpec,
    make_model: Callable[[dict], object],
    params: dict | None = None,
    *,
    model_name: str = "model",
    folds: Sequence[int] | None = None,
) -> CVResult:
    """Fit/score one configuration on the official folds; features re-fit inside every fold."""
    params = dict(params or {})
    y, chem, taxon = ds.y, ds.chem, ds.taxon
    oof = np.full(len(ds.df), np.nan)
    mask = np.zeros(len(ds.df), dtype=bool)
    fold_metrics: list[dict] = []
    t0 = time.time()
    for k, trn, val in ds.cv_splits():
        if folds is not None and k not in folds:
            continue
        Xtr, Xva, builder = ds.matrices(spec, trn, val)
        model = make_model(params)
        model.fit(Xtr, y[trn], train_rows=ds.rows(trn), valid_rows=ds.rows(val),
                  feature_names=builder.names_, builder=builder)
        p = np.asarray(model.predict(Xva, rows=ds.rows(val)), dtype=float)
        oof[val] = p
        mask[val] = True
        m = E.report(y[val], p, chem=chem[val], taxon=taxon[val])
        m.update({"fold": k, "n_features": builder.n_features})
        fold_metrics.append(m)
    keys = [k for k in E.METRIC_NAMES if all(k in m for m in fold_metrics)]
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    sd = {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else float("nan")
          for k in keys}
    pooled = E.report(y[mask], oof[mask], chem=chem[mask], taxon=taxon[mask])
    return CVResult(model=model_name, spec=spec.to_dict(), params=params, fold_metrics=fold_metrics,
                    mean=mean, sd=sd, pooled=pooled, oof=oof, oof_mask=mask, seconds=time.time() - t0)


def expand_grid(grid: dict[str, Iterable]) -> list[dict]:
    """Cartesian product of a hyper-parameter grid, in a deterministic order."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[list(grid[k]) for k in keys])]


@dataclass
class SearchResult:
    """Everything tried, and the winner by mean CV RMSE over the official folds."""

    best: CVResult
    all_results: list[CVResult]
    n_configurations: int

    def table(self) -> pd.DataFrame:
        rows = []
        for r in self.all_results:
            row = {"model": r.model, "features": r.spec, "seconds": round(r.seconds, 1)}
            row.update({f"cv_{k}": v for k, v in r.mean.items()})
            row.update(r.params)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def grid_search(
    ds: ChemToxDataset,
    specs: Sequence[FeatureSpec],
    make_model: Callable[[dict], object],
    grid: dict[str, Iterable],
    *,
    model_name: str = "model",
    log=None,
    select_metric: str = "rmse",
) -> SearchResult:
    """CV every (feature spec x hyper-parameter) combination; select by mean CV ``select_metric``."""
    results: list[CVResult] = []
    combos = [(s, p) for s in specs for p in expand_grid(grid)]
    for i, (spec, params) in enumerate(combos, start=1):
        r = run_cv(ds, spec, make_model, params, model_name=model_name)
        results.append(r)
        if log:
            log.info("[%d/%d] %s %s %s -> CV %s %.4f (%.1fs)", i, len(combos), model_name,
                     spec.name(), params, select_metric, r.mean.get(select_metric, float("nan")),
                     r.seconds)
    best = min(results, key=lambda r: r.mean.get(select_metric, np.inf))
    return SearchResult(best=best, all_results=results, n_configurations=len(results))


# ----------------------------------------------------------------------------- registry logging


def split_label(ds: ChemToxDataset, kind: str) -> str:
    """``'<challenge>/<split>/<cv|test>'`` — the registry's split string for this workstream."""
    return f"{ds.challenge}/{ds.split}/{kind}"


def log_metrics(
    ds: ChemToxDataset,
    *,
    experiment: str,
    model: str,
    kind: str,
    metrics: dict,
    config: dict,
    n: int | None = None,
    cis: dict | None = None,
    is_baseline: bool = False,
    notes: str = "",
    extra: dict | None = None,
    data_manifests: Sequence[str] = ("adore_raw", "adore"),
    artifacts: Sequence[str] = (),
    only: Sequence[str] | None = None,
) -> list[dict]:
    """Write one registry entry per metric (``sentinel.core.registry`` is the single write path)."""
    ids = []
    names = list(only) if only else [k for k in metrics if k not in ("n", "n_chemicals", "n_taxa",
                                                                     "rmse_macro_cells")]
    for name in names:
        value = metrics.get(name)
        if value is None or not np.isfinite(value):
            continue
        ci = (cis or {}).get(name, {})
        entry = log_result(
            workstream=WORKSTREAM,
            experiment=experiment,
            model=model,
            metric=name,
            value=float(value),
            split=split_label(ds, kind),
            n=int(n if n is not None else metrics.get("n", 0)) or None,
            ci_low=ci.get("ci_low"),
            ci_high=ci.get("ci_high"),
            ci_method=ci.get("ci_method"),
            target=TARGET_NAMES[ds.target],
            is_baseline=is_baseline,
            config=config,
            data_manifests=list(data_manifests),
            artifacts=list(artifacts),
            notes=notes,
            extra=dict(extra or {}),
        )
        ids.append({"metric": name, "id": entry["id"], "value": float(value)})
    return ids


def write_json(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    return p


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
