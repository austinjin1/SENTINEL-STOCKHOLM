# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Metrics for SENTINEL-ChemTox, in the same definitions as the published ADORE baseline.

Micro metrics come from :mod:`sentinel.eval.metrics` (RMSE, MAE, R2). The macro variants are
Gasser et al. (2024) Appendix B.2, which weight chemicals and taxa instead of data points:

* ``rmse_macro`` (their RMSE_M) — equal weight per observed (chemical, taxon) cell;
* ``rmse_chemical`` (RMSE_C)    — equal weight per chemical;
* ``rmse_taxon`` (RMSE_T)       — equal weight per taxon.

Each is sqrt(mean over groups of the within-group mean squared error). The published formulae
divide the (chemical, taxon) sum by ``N_C * N_T``; since the overwhelming majority of cells are
unobserved (and the formula's inner ``1/n_ct`` is undefined for them), the mean here is taken over
**observed** cells, which is the only computable reading. ``rmse_macro_cells`` records how many
cells that was.

Two further metrics are reported because the regulatory and the TRIDENT literature use them:

* ``within_1_log`` — fraction of predictions within one log10 unit (ECHA / Karamertzanis yardstick);
* ``median_abs_fold_error`` — median of 10**|error|, comparable to TRIDENT's fold errors.

Confidence intervals use :func:`sentinel.eval.bootstrap.bootstrap_ci` with the chemical as the
resampling unit (``groups = chemical id``), never the row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.eval import metrics as M
from sentinel.eval.bootstrap import bootstrap_ci

#: metrics logged for every ChemTox configuration
METRIC_NAMES = (
    "rmse", "mae", "r2", "rmse_macro", "rmse_chemical", "rmse_taxon",
    "within_1_log", "median_abs_fold_error",
)


def _finite(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m], m


def grouped_rmse(y, p, groups) -> float:
    """sqrt(mean over groups of the within-group mean squared error)."""
    y, p, m = _finite(y, p)
    g = np.asarray(groups, dtype=object)[m]
    if y.size == 0:
        return float("nan")
    se = pd.Series((p - y) ** 2)
    per_group = se.groupby(pd.Series(g).astype(str)).mean()
    return float(np.sqrt(per_group.mean()))


def rmse_macro(y, p, chem, taxon) -> float:
    """RMSE_M — equal weight per observed (chemical, taxon) cell."""
    chem = pd.Series(np.asarray(chem, dtype=object)).astype(str)
    taxon = pd.Series(np.asarray(taxon, dtype=object)).astype(str)
    return grouped_rmse(y, p, (chem + "\x1f" + taxon).to_numpy())


def within_1_log(y, p) -> float:
    y, p, _ = _finite(y, p)
    return float(np.mean(np.abs(p - y) <= 1.0)) if y.size else float("nan")


def median_abs_fold_error(y, p) -> float:
    """Median of 10**|log10 error| — the multiplicative error on the native concentration scale."""
    y, p, _ = _finite(y, p)
    return float(np.median(np.power(10.0, np.abs(p - y)))) if y.size else float("nan")


def n_cells(chem, taxon) -> int:
    c = pd.Series(np.asarray(chem, dtype=object)).astype(str)
    t = pd.Series(np.asarray(taxon, dtype=object)).astype(str)
    return int((c + "\x1f" + t).nunique())


def report(y, p, *, chem=None, taxon=None) -> dict[str, float]:
    """All ChemTox metrics for one set of predictions."""
    yy, _pp, _ = _finite(y, p)
    out = {
        "rmse": M.rmse(y, p),
        "mae": M.mae(y, p),
        "r2": M.r2(y, p),
        "bias": M.bias(y, p),
        "within_1_log": within_1_log(y, p),
        "median_abs_fold_error": median_abs_fold_error(y, p),
        "n": int(yy.size),
    }
    if chem is not None:
        out["rmse_chemical"] = grouped_rmse(y, p, chem)
        out["n_chemicals"] = int(pd.Series(np.asarray(chem, dtype=object)).astype(str).nunique())
    if taxon is not None:
        out["rmse_taxon"] = grouped_rmse(y, p, taxon)
        out["n_taxa"] = int(pd.Series(np.asarray(taxon, dtype=object)).astype(str).nunique())
    if chem is not None and taxon is not None:
        out["rmse_macro"] = rmse_macro(y, p, chem, taxon)
        out["rmse_macro_cells"] = n_cells(chem, taxon)
    return out


_METRIC_FUNCS = {
    "rmse": lambda y, p: M.rmse(y, p),
    "mae": lambda y, p: M.mae(y, p),
    "r2": lambda y, p: M.r2(y, p),
    "within_1_log": within_1_log,
    "median_abs_fold_error": median_abs_fold_error,
}


def bootstrap_report(
    y, p, *, chem, taxon=None, metrics=("rmse", "mae", "r2", "within_1_log", "median_abs_fold_error",
                                        "rmse_chemical", "rmse_taxon", "rmse_macro"),
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, dict]:
    """Chemical-cluster bootstrap CIs (resampling unit = chemical) for the requested metrics."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    chem_a = np.asarray(chem, dtype=object).astype(str)
    taxon_a = None if taxon is None else np.asarray(taxon, dtype=object).astype(str)
    out: dict[str, dict] = {}
    for name in metrics:
        if name in _METRIC_FUNCS:
            ci = bootstrap_ci(_METRIC_FUNCS[name], y, p, groups=chem_a, n_boot=n_boot, seed=seed)
        elif name == "rmse_chemical":
            ci = bootstrap_ci(lambda a, b, c: grouped_rmse(a, b, c), y, p, chem_a,
                              groups=chem_a, n_boot=n_boot, seed=seed)
        elif name == "rmse_taxon" and taxon_a is not None:
            ci = bootstrap_ci(lambda a, b, t: grouped_rmse(a, b, t), y, p, taxon_a,
                              groups=chem_a, n_boot=n_boot, seed=seed)
        elif name == "rmse_macro" and taxon_a is not None:
            ci = bootstrap_ci(lambda a, b, c, t: rmse_macro(a, b, c, t), y, p, chem_a, taxon_a,
                              groups=chem_a, n_boot=n_boot, seed=seed)
        else:
            continue
        ci["ci_method"] = f"cluster bootstrap by chemical, {n_boot} resamples, seed {seed}"
        out[name] = ci
    return out


def per_taxon_table(y, p, taxon, *, min_n: int = 1) -> pd.DataFrame:
    """Per-taxon RMSE/bias/count table (for diagnostics and the SSD module)."""
    y, p, m = _finite(y, p)
    t = pd.Series(np.asarray(taxon, dtype=object)[m]).astype(str)
    df = pd.DataFrame({"taxon": t.to_numpy(), "y": y, "p": p})
    g = df.groupby("taxon")
    out = pd.DataFrame({
        "n": g.size(),
        "rmse": g.apply(lambda d: float(np.sqrt(np.mean((d["p"] - d["y"]) ** 2))), include_groups=False),
        "bias": g.apply(lambda d: float(np.mean(d["p"] - d["y"])), include_groups=False),
    }).reset_index()
    return out[out["n"] >= min_n].sort_values("n", ascending=False).reset_index(drop=True)
