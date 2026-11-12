# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Scoring for SENTINEL-Lite (protocol §4): MdSA & co. in log10 space for Secchi/turbidity/TSS, trophic-class
metrics for chl-a; water-body cluster-bootstrap CIs and paired comparisons."""

from __future__ import annotations

import numpy as np

from ...eval import metrics as M
from ...eval.bootstrap import bootstrap_ci, paired_bootstrap
from .data import REG_TARGETS, trophic_class_log10

N_BOOT = 2000


def _lin(x):
    return np.power(10.0, np.asarray(x, float))


def mdsa_log(y_log, p_log) -> float:
    return M.mdsa(_lin(y_log), _lin(p_log))


def reg_report(y_log, p_log, sigma_log=None) -> dict:
    y, p = _lin(y_log), _lin(p_log)
    out = {"mdsa": M.mdsa(y, p), "sspb": M.sspb(y, p), "log_rmse": M.log_rmse(y, p), "log_r2": M.log_r2(y, p),
           "log_slope": M.log_slope(y, p), "n": int(np.isfinite(np.asarray(y_log, float)).sum())}
    if sigma_log is not None:
        z = 1.2815515655446004  # 80% central interval
        out["cov80"] = M.interval_coverage(np.asarray(y_log), np.asarray(p_log) - z * np.asarray(sigma_log),
                                           np.asarray(p_log) + z * np.asarray(sigma_log))
    return out


def balanced_accuracy(yc, pc) -> float:
    yc, pc = np.asarray(yc), np.asarray(pc)
    accs = [np.mean(pc[yc == k] == k) for k in np.unique(yc)]
    return float(np.mean(accs)) if accs else np.nan


def macro_f1(yc, pc) -> float:
    yc, pc = np.asarray(yc), np.asarray(pc)
    f = []
    for k in np.unique(yc):
        tp = np.sum((pc == k) & (yc == k))
        prec = tp / max(np.sum(pc == k), 1)
        rec = tp / max(np.sum(yc == k), 1)
        f.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f)) if f else np.nan


def qw_kappa(yc, pc, k: int = 4) -> float:
    yc, pc = np.asarray(yc, int), np.asarray(pc, int)
    O = np.zeros((k, k))
    for a, b in zip(yc, pc):
        O[a, b] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def class_report(y_log, p_log) -> dict:
    yc, pc = trophic_class_log10(y_log), trophic_class_log10(p_log)
    return {"bal_acc": balanced_accuracy(yc, pc), "macro_f1": macro_f1(yc, pc), "qw_kappa": qw_kappa(yc, pc),
            "accuracy": float(np.mean(yc == pc)), "n": int(len(yc))}


def _ba_log(y_log, p_log):
    return balanced_accuracy(trophic_class_log10(y_log), trophic_class_log10(p_log))


def primary_metric(target: str):
    """(metric fn on log arrays, name, higher_is_better)."""
    if target in REG_TARGETS:
        return mdsa_log, "mdsa", False
    return _ba_log, "bal_acc", True


def score(target: str, y_log, p_log, groups, *, sigma_log=None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    y_log, p_log = np.asarray(y_log, float), np.asarray(p_log, float)
    ok = np.isfinite(y_log) & np.isfinite(p_log)
    y_log, p_log, g = y_log[ok], p_log[ok], np.asarray(groups)[ok]
    s = None if sigma_log is None else np.asarray(sigma_log, float)[ok]
    rep = reg_report(y_log, p_log, s) if target in REG_TARGETS else class_report(y_log, p_log)
    fn, name, _ = primary_metric(target)
    ci = bootstrap_ci(fn, y_log, p_log, groups=g, n_boot=n_boot, seed=seed)
    rep.update({"primary": name, "value": ci["point"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                "n_groups": ci["n_groups"], "n": int(ok.sum())})
    return rep


def compare(target: str, y_log, p_a, p_b, groups, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired water-body bootstrap of primary(a) - primary(b) on identical rows."""
    y_log, p_a, p_b = (np.asarray(x, float) for x in (y_log, p_a, p_b))
    ok = np.isfinite(y_log) & np.isfinite(p_a) & np.isfinite(p_b)
    fn, name, hib = primary_metric(target)
    r = paired_bootstrap(fn, y_log[ok], p_a[ok], p_b[ok], groups=np.asarray(groups)[ok], n_boot=n_boot,
                         seed=seed, higher_is_better=hib)
    r["metric"] = name
    r["n"] = int(ok.sum())
    return r


# ---------------------------------------------------------------- primary populations (protocol §3, review W12)
PRIMARY_POPULATION = {"secchi": "lake", "turbidity": "all", "tss": "all", "chla": "lake"}
LAKE_CLASSES = ("lake_reservoir",)


def population_mask(site_class, pop: str) -> np.ndarray:
    sc = np.asarray(site_class).astype(str)
    return np.isin(sc, LAKE_CLASSES) if pop == "lake" else np.ones(len(sc), bool)


def primary_mask(target: str, site_class) -> np.ndarray:
    return population_mask(site_class, PRIMARY_POPULATION[target])


def score_primary(target: str, y_log, p_log, groups, site_class, *, sigma_log=None, n_boot: int = N_BOOT,
                  seed: int = 0) -> dict:
    """Score on the target's primary population; the other population and per-class values go in 'secondary'."""
    y_log, p_log, groups = np.asarray(y_log, float), np.asarray(p_log, float), np.asarray(groups)
    s = None if sigma_log is None else np.asarray(sigma_log, float)
    m = primary_mask(target, site_class)
    rep = score(target, y_log[m], p_log[m], groups[m], sigma_log=None if s is None else s[m], n_boot=n_boot, seed=seed)
    rep["population"] = PRIMARY_POPULATION[target]
    other = "all" if rep["population"] == "lake" else "lake"
    mo = population_mask(site_class, other)
    fn = primary_metric(target)[0]
    ok = np.isfinite(y_log) & np.isfinite(p_log)
    rep["secondary_population"] = {"population": other, "value": fn(y_log[mo & ok], p_log[mo & ok]) if (mo & ok).any()
                                   else float("nan"), "n": int((mo & ok).sum())}
    rep["by_class"] = {c: {"value": fn(y_log[(np.asarray(site_class) == c) & ok], p_log[(np.asarray(site_class) == c) & ok]),
                           "n": int(((np.asarray(site_class) == c) & ok).sum())}
                       for c in np.unique(np.asarray(site_class).astype(str)) if ((np.asarray(site_class) == c) & ok).sum() >= 10}
    return rep


# ---------------------------------------------------------------- one-sided tests (pre-test review W2, C4)
def compare_one_sided(target: str, y_log, p_a, p_b, groups, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired water-body bootstrap of primary(a) - primary(b) on identical rows. It uses the same resampling as
    ``sentinel.eval.bootstrap.paired_bootstrap`` (same rng and order, so the CI is identical), plus:

    - ``p_one_sided``: the bootstrap probability that a is **not** better, read directly from the correct tail. For
      MdSA (lower is better) this is P(d ≥ 0); for balanced accuracy it is P(d ≤ 0).
    - ``ci_excludes_0_better``: the 95% percentile CI lies entirely on the "a better" side.
    """
    from sentinel.eval.bootstrap import _group_index

    y_log, p_a, p_b = (np.asarray(x, float) for x in (y_log, p_a, p_b))
    ok = np.isfinite(y_log) & np.isfinite(p_a) & np.isfinite(p_b)
    fn, name, hib = primary_metric(target)
    y, pa, pb, g = y_log[ok], p_a[ok], p_b[ok], np.asarray(groups)[ok]
    idx_groups = _group_index(g, len(y))
    rs = np.random.default_rng(seed)
    point = float(fn(y, pa) - fn(y, pb))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rs.integers(0, len(idx_groups), len(idx_groups))
        idx = np.concatenate([idx_groups[k] for k in pick])
        diffs[b] = fn(y[idx], pa[idx]) - fn(y[idx], pb[idx])
    d = diffs[np.isfinite(diffs)]
    lo, hi = (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))) if d.size else (np.nan, np.nan)
    p1 = float(np.mean(d <= 0)) if hib else float(np.mean(d >= 0))
    excl = (lo > 0) if hib else (hi < 0)
    return {"diff": point, "ci_low": lo, "ci_high": hi, "p_one_sided": p1, "ci_excludes_0_better": bool(excl),
            "a_better_point": bool(point > 0 if hib else point < 0), "metric": name, "n": int(ok.sum()),
            "n_groups": len(idx_groups), "n_boot_valid": int(d.size)}


def holm_with_ci(results: dict, alpha: float = 0.05) -> dict:
    """Protocol §1 / RESEARCH_PLAN §15.1(ii): a comparison is rejected only if Holm (on the one-sided p) rejects AND
    the paired 95% CI excludes 0 in the student-better direction. Adds ``holm`` and ``reject`` in place."""
    from sentinel.eval.bootstrap import holm_bonferroni

    pv = {k: (float(r["p_one_sided"]) if np.isfinite(r.get("p_one_sided", np.nan)) else 1.0) for k, r in results.items()}
    holm = holm_bonferroni(pv, alpha=alpha)        # NaN p -> 1 (delta review observation)
    for k, r in results.items():
        r["holm"] = bool(holm[k])
        r["reject"] = bool(holm[k]) and bool(r["ci_excludes_0_better"])
    return results
