# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Scoring for SENTINEL-HydroMet estimates (no sensor input); docs/preregistration/hydromet_protocol.md.

* **Cells.** For target v and block b the scored (site, day) cells are observed, *approved*
  (non-provisional) values of v inside the block (42-day embargo after the start of non-train
  blocks) at the evaluated sites, with finite predictions from EVERY compared model; cells dropped
  because some model is undefined are counted. Sites with < ``min_days`` cells are not scored.
* **Per-site metrics** (``sentinel.eval.metrics``): n, RMSE, MAE, bias, NSE, KGE (original units); for
  log-modelled targets also NSE/RMSE on the model (log) scale.
* **Uncertainty.** Sites cluster in few basin groups (HUC4 merged across flow connections; the split's
  spatial unit), so the PRIMARY resampling unit is the **basin group** (all scored sites of a group
  are drawn together; ``sentinel.eval.bootstrap``). Site resampling is reported as a labelled
  sensitivity analysis (``*_site`` columns). ``n_groups`` is reported next to ``n_sites`` everywhere.
* **Pooled cross-site skill** over all scored cells (between-site level errors count), with
  group-bootstrap CIs computed from per-group sufficient statistics (exact, fast).
* **Primary comparison** (protocol §6–§7): seed-ensemble SENTINEL model vs the best comparator of
  each of two families (GBM; physics/process) — intersection–union: an endpoint is won only if it
  beats BOTH; Holm across the (model × endpoint) hypotheses on the IUT p-value; plus the every-seed rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...eval import metrics as M
from ...eval.bootstrap import bootstrap_ci, holm_bonferroni, paired_bootstrap
from .data import LOG_TARGETS, TARGETS, HydroMetData, to_model

HIGHER_IS_BETTER = {"nse": True, "kge": True, "nse_log": True, "rmse": False, "mae": False, "rmse_log": False}
SUMMARY_METRICS = ("nse", "kge", "rmse", "nse_log", "rmse_log")

PRIMARY_TARGETS = ("temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min")
SECONDARY_TARGETS = ("spcond_uscm_mean", "turb_fnu_mean", "no3no2_mgl_mean")
PRIMARY_METRIC = "kge"
# ealstm_gbmblend: protocol v0.7 (cycle 2); ealstm_full: protocol v0.8 (cycle 3, refit on all train-role sites)
SENTINEL_MODELS = ("ealstm", "ealstm_physdo", "ealstm_gbmblend", "ealstm_full", "ealstm_compact",   # +compact: v0.9
                   "ealstm_ws",   # +upstream-basin forcings: v0.10
                   "ealstm_ws_full")   # +ws refit on fit + inner sites: v0.11
# forcing_gbm_bag: 3-seed bag (protocol v0.6); forcing_gbm_compact: compact static set (v0.9);
# *_ws: upstream-basin forcings (v0.10; GBM appended like the EA-LSTM, physics driven by basin-mean air temperature)
_T_GBM = ("forcing_gbm", "doy_location_gbm", "forcing_gbm_bag", "forcing_gbm_compact", "forcing_gbm_ws_bag")
_T_PHYS = ("air2stream_regional", "air2water_linear", "air2water_mohseni", "air2stream_regional_ws",
           "air2water_linear_ws", "air2water_mohseni_ws")
_DO_GBM = _T_GBM
_DO_PHYS = ("dosat_ratio[air2stream_regional]", "dosat_ratio[doy_location_gbm]", "dosat_ratio[air2stream_regional]_ws")
COMPARATOR_FAMILIES = {   # protocol §6: the SENTINEL model must beat the best member of BOTH families
    "temp_c_mean": {"gbm": _T_GBM, "physics": _T_PHYS},
    "temp_c_7dadm": {"gbm": _T_GBM, "physics": _T_PHYS},
    "do_mgl_mean": {"gbm": _DO_GBM, "physics": _DO_PHYS},
    "do_mgl_min": {"gbm": _DO_GBM, "physics": _DO_PHYS},
}


def _nanmedian(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else np.nan


# ----------------------------------------------------------------------------- cells & per-site metrics

def common_mask(data: HydroMetData, v: str, block: str, site_idx, t_idx, preds: dict,
                approved_only: bool = True) -> tuple[np.ndarray, dict]:
    """(n_sites, n_t) mask of scored cells for target v, identical for every model in ``preds``
    ({model: {v: (n_sites, n_t)}}); returns the mask and per-model counts of dropped cells."""
    site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
    base = data.target_mask(v, block, site_idx, approved_only=approved_only)[site_idx][:, t_idx]
    drops, m = {}, base.copy()
    for name, p in preds.items():
        if v not in p:
            continue
        fin = np.isfinite(p[v])
        drops[name] = int((base & ~fin).sum())
        m &= fin
    return m, drops


def site_metrics(data: HydroMetData, preds: dict, site_idx, t_idx, block: str, *, min_days: int = 60,
                 approved_only: bool = True, targets=TARGETS) -> tuple[pd.DataFrame, dict]:
    """Per (model, target, site) metrics on the common cells (+ role label and basin group)."""
    site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
    rows, info = [], {}
    for v in targets:
        models = [k for k, p in preds.items() if v in p]
        if not models:
            continue
        mask, drops = common_mask(data, v, block, site_idx, t_idx, {k: preds[k] for k in models}, approved_only)
        obs = data.y[site_idx][:, t_idx, data.vi(v)]
        n_site = mask.sum(axis=1)
        scored = np.flatnonzero(n_site >= min_days)
        info[v] = {"models": models, "n_cells": int(mask.sum()), "n_sites_with_obs": int((n_site > 0).sum()),
                   "n_sites_scored": int(scored.size),
                   "n_groups_scored": int(np.unique(data.basin_group[site_idx[scored]]).size),
                   "n_cells_scored": int(n_site[scored].sum()), "dropped_nonfinite_by_model": drops,
                   "min_days": min_days}
        for k in models:
            P = preds[k][v]
            for i in scored:
                m = mask[i]
                y, p = obs[i, m].astype("float64"), P[i, m].astype("float64")
                gi = site_idx[i]
                rec = {"model": k, "var": v, "site_no": data.sites[gi], "role": data.role_label(gi),
                       "basin_group": str(data.basin_group[gi]), "n": int(m.sum()), "rmse": M.rmse(y, p),
                       "mae": M.mae(y, p), "bias": M.bias(y, p), "nse": M.nse(y, p), "kge": M.kge(y, p)}
                if v in LOG_TARGETS:
                    ly, lp = to_model(v, y), to_model(v, p)
                    rec["nse_log"], rec["rmse_log"] = M.nse(ly, lp), M.rmse(ly, lp)
                rows.append(rec)
    return pd.DataFrame(rows), info


def summarize(site_df: pd.DataFrame, *, refs=(), n_boot: int = 2000, seed: int = 0,
              metrics=SUMMARY_METRICS) -> pd.DataFrame:
    """Median across sites; 95% CI by resampling BASIN GROUPS (primary) and by sites (sensitivity,
    ``*_site``); paired per-site differences vs ``refs`` (median of per-site differences)."""
    out = []
    if site_df.empty:
        return pd.DataFrame()
    for (model, v, role), g in site_df.groupby(["model", "var", "role"]):
        for met in metrics:
            if met not in g or not np.isfinite(g[met]).any():
                continue
            vals = g[met].to_numpy(float)
            grp = g["basin_group"].to_numpy()
            ci = bootstrap_ci(_nanmedian, vals, groups=grp, n_boot=n_boot, seed=seed)
            cs = bootstrap_ci(_nanmedian, vals, n_boot=n_boot, seed=seed)
            rec = {"model": model, "var": v, "role": role, "metric": met, "n_sites": int(np.isfinite(vals).sum()),
                   "n_groups": int(np.unique(grp).size), "median": ci["point"], "ci_low": ci["ci_low"],
                   "ci_high": ci["ci_high"], "ci_unit": "basin_group", "ci_low_site": cs["ci_low"],
                   "ci_high_site": cs["ci_high"], "n_cells": int(g["n"].sum())}
            for r in refs:
                if r == model:
                    continue
                h = site_df[(site_df["model"] == r) & (site_df["var"] == v) & (site_df["role"] == role)]
                j = g.set_index("site_no")[[met, "basin_group"]].rename(columns={met: "a"}).join(
                    h.set_index("site_no")[met].rename("b"), how="inner").dropna()
                if not len(j):
                    continue
                d = (j["a"] - j["b"]).to_numpy()
                cd = bootstrap_ci(_nanmedian, d, groups=j["basin_group"].to_numpy(), n_boot=n_boot, seed=seed)
                better = d > 0 if HIGHER_IS_BETTER.get(met, True) else d < 0
                rec[f"diff_vs_{r}"] = cd["point"]
                rec[f"diff_vs_{r}_ci_low"] = cd["ci_low"]
                rec[f"diff_vs_{r}_ci_high"] = cd["ci_high"]
                rec[f"frac_better_vs_{r}"] = float(better.mean())
            out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- pooled cross-site skill

_STATS = ("n", "sy", "syy", "sp", "spp", "syp", "sse")


def _suff_stats(y, p, groups):
    """Per-group sufficient statistics (G, 7) for pooled NSE/KGE/RMSE."""
    ug, inv = np.unique(groups, return_inverse=True)
    S = np.zeros((ug.size, len(_STATS)))
    for k, x in enumerate((np.ones_like(y), y, y * y, p, p * p, y * p, (p - y) ** 2)):
        S[:, k] = np.bincount(inv, weights=x, minlength=ug.size)
    return S


def _metrics_from_stats(T):
    """T (..., 7) summed statistics -> dict of pooled nse, kge, rmse (population moments, as metrics.py)."""
    n, sy, syy, sp, spp, syp, sse = (T[..., k] for k in range(7))
    with np.errstate(invalid="ignore", divide="ignore"):
        my, mp = sy / n, sp / n
        vy, vp = syy / n - my ** 2, spp / n - mp ** 2
        cov = syp / n - my * mp
        nse = 1 - sse / (vy * n)
        r = cov / np.sqrt(np.maximum(vy, 0) * np.maximum(vp, 0))
        kge = 1 - np.sqrt((r - 1) ** 2 + (np.sqrt(np.maximum(vp, 0) / vy) - 1) ** 2 + (mp / my - 1) ** 2)
        rmse = np.sqrt(sse / n)
    return {"nse": nse, "kge": kge, "rmse": rmse}


def _pooled_boot(y, p, groups, n_boot, seed, alpha=0.05):
    S = _suff_stats(y, p, groups)
    point = _metrics_from_stats(S.sum(axis=0))
    rs = np.random.default_rng(seed)
    G = S.shape[0]
    counts = np.stack([np.bincount(rs.integers(0, G, G), minlength=G) for _ in range(n_boot)])
    boot = _metrics_from_stats(counts @ S)
    out = {}
    for k, v in point.items():
        b = boot[k][np.isfinite(boot[k])]
        out[k] = (float(v), float(np.quantile(b, alpha / 2)) if b.size else np.nan,
                  float(np.quantile(b, 1 - alpha / 2)) if b.size else np.nan)
    return out


def pooled_metrics(data: HydroMetData, preds: dict, site_idx, t_idx, block: str, *, min_days: int = 60,
                   n_boot: int = 2000, seed: int = 0, targets=TARGETS) -> pd.DataFrame:
    """Pooled cross-site NSE/KGE/RMSE (+ log-scale NSE for log targets) over all scored cells of all
    scored sites, on the same cells as :func:`site_metrics`. CIs: basin-group bootstrap (primary)
    and site bootstrap (``*_site``)."""
    site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
    out = []
    for v in targets:
        models = [k for k, p in preds.items() if v in p]
        if not models:
            continue
        mask, _ = common_mask(data, v, block, site_idx, t_idx, {k: preds[k] for k in models})
        mask &= (mask.sum(axis=1) >= min_days)[:, None]
        si, ti = np.nonzero(mask)
        if not si.size:
            continue
        obs = data.y[site_idx][:, t_idx, data.vi(v)][si, ti].astype("float64")
        grp = data.basin_group[site_idx[si]]
        for k in models:
            p = preds[k][v][si, ti].astype("float64")
            scales = [("", obs, p)] + ([("log_", to_model(v, obs), to_model(v, p))] if v in LOG_TARGETS else [])
            for pre, yy, pp in scales:
                g_res = _pooled_boot(yy, pp, grp, n_boot, seed)
                s_res = _pooled_boot(yy, pp, si, n_boot, seed)
                for met in (("nse",) if pre else ("nse", "kge", "rmse")):
                    out.append({"model": k, "var": v, "metric": f"pooled_{pre}{met}", "value": g_res[met][0],
                                "ci_low": g_res[met][1], "ci_high": g_res[met][2], "ci_unit": "basin_group",
                                "ci_low_site": s_res[met][1], "ci_high_site": s_res[met][2],
                                "n_sites": int(np.unique(si).size), "n_groups": int(np.unique(grp).size),
                                "n_cells": int(si.size)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- primary comparison

def select_comparators(site_df: pd.DataFrame, *, role: str = "val", metric: str = PRIMARY_METRIC,
                       families: dict = COMPARATOR_FAMILIES) -> dict:
    """Best member of each comparator family per primary endpoint by median per-site ``metric`` on
    val-role sites × val block ONLY; frozen into FREEZE_hydromet.json ``comparators``."""
    if role != "val" or (site_df["role"] == "test").any():
        raise ValueError("comparators are selected on val-role sites only, never on test-role sites")
    sd = site_df[site_df["role"] == role]
    chosen = {}
    for v, fams in families.items():
        for fam, cands in fams.items():
            med = {c: _nanmedian(sd[(sd["model"] == c) & (sd["var"] == v)][metric]) for c in cands}
            med = {c: m for c, m in med.items() if np.isfinite(m)}
            if med:
                chosen.setdefault(v, {})[fam] = max(med, key=med.get)
    return chosen


def _paired(site_df, a_name, b_name, v, metric, n_boot, seed, groups: bool):
    a = site_df[(site_df["model"] == a_name) & (site_df["var"] == v)].set_index("site_no")
    b = site_df[(site_df["model"] == b_name) & (site_df["var"] == v)].set_index("site_no")[metric]
    j = a[[metric, "basin_group"]].rename(columns={metric: "a"}).join(b.rename("b"), how="inner").dropna()
    if len(j) < 3:
        return None, j
    res = paired_bootstrap(lambda _y, p: _nanmedian(p), np.zeros(len(j)), j["a"].to_numpy(), j["b"].to_numpy(),
                           groups=j["basin_group"].to_numpy() if groups else None, n_boot=n_boot, seed=seed)
    return res, j


MIN_TEST_GROUPS = 3   # protocol §7 (2026-11-09): fewer basin groups -> hypothesis not testable, p = 1


def primary_comparison(site_df: pd.DataFrame, comparators: dict, *, models=SENTINEL_MODELS, seed_models=None,
                       endpoints=PRIMARY_TARGETS, metric: str = PRIMARY_METRIC, n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> pd.DataFrame:
    """Protocol §7. For each SENTINEL model m (seed ensemble) and endpoint v, against the selected
    comparator of each family f ∈ {gbm, physics}: Δ_f = median_site(KGE_m) − median_site(KGE_f) on the
    sites scored by both, paired bootstrap resampling BASIN GROUPS (primary; site resampling as
    sensitivity), 95% CI and two-sided p. Intersection–union: p_IUT = max_f p_f; Holm across all
    (model × endpoint) p_IUT. Every-seed rule: Δ_f > 0 (point) for every seed model in
    ``seed_models[m]`` and both families. ``won`` = Holm reject ∧ Δ_f > 0 ∧ CI_f excludes 0 for both
    families ∧ every-seed rule. Hypotheses that cannot be tested (e.g. a model not run) are kept in the
    Holm family with p = 1 (status "not available"), so the pre-registered family size is preserved."""
    seed_models = seed_models or {}
    rows = []
    for m in models:
        for v in endpoints:
            comps = comparators.get(v, {})
            if set(comps) != {"gbm", "physics"}:
                continue
            rec = {"model": m, "var": v, "metric": f"median_site_{metric}", "ci_unit": "basin_group"}
            ok = True
            for fam in ("gbm", "physics"):
                ref = comps[fam]
                res, j = _paired(site_df, m, ref, v, metric, n_boot, seed, groups=True)
                if res is None or j["basin_group"].nunique() < MIN_TEST_GROUPS:   # bootstrap needs >= 3 groups
                    ok = False
                    break
                sens, _ = _paired(site_df, m, ref, v, metric, n_boot, seed, groups=False)
                seed_d = []
                for sm in seed_models.get(m, []):
                    r2, j2 = _paired(site_df, sm, ref, v, metric, 10, seed, groups=False)
                    seed_d.append(_nanmedian(j2["a"]) - _nanmedian(j2["b"]) if len(j2) else np.nan)
                rec.update({f"{fam}_comparator": ref, f"{fam}_n_sites": int(len(j)),
                            f"{fam}_n_groups": int(j["basin_group"].nunique()),
                            f"{fam}_median_model": _nanmedian(j["a"]), f"{fam}_median_comparator": _nanmedian(j["b"]),
                            f"{fam}_diff": res["diff"], f"{fam}_ci_low": res["ci_low"], f"{fam}_ci_high": res["ci_high"],
                            f"{fam}_p": res["p_value"], f"{fam}_ci_low_site": sens["ci_low"],
                            f"{fam}_ci_high_site": sens["ci_high"], f"{fam}_p_site": sens["p_value"],
                            f"{fam}_seed_diffs": json.dumps([round(float(x), 6) for x in seed_d])})
                rec[f"{fam}_every_seed_positive"] = bool(seed_d) and all(np.isfinite(seed_d)) and min(seed_d) > 0
            if ok:
                rec["p_iut"] = max(rec["gbm_p"], rec["physics_p"])
                rec["n_seeds"] = len(seed_models.get(m, []))
                rec["status"] = "tested"
            else:   # hypothesis not testable (model not run / too few sites): stays in the Holm family, p = 1
                rec = {"model": m, "var": v, "metric": f"median_site_{metric}", "ci_unit": "basin_group",
                       "p_iut": 1.0, "status": "not available", "gbm_every_seed_positive": False,
                       "physics_every_seed_positive": False, "gbm_diff": np.nan, "physics_diff": np.nan,
                       "gbm_ci_low": np.nan, "physics_ci_low": np.nan, "n_seeds": len(seed_models.get(m, []))}
            rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        rej = holm_bonferroni({f"{r.model}|{r.var}": r.p_iut for r in df.itertuples()}, alpha=alpha)
        df["holm_reject"] = [rej[f"{r.model}|{r.var}"] for r in df.itertuples()]
        df["n_hypotheses"] = len(df)
        df["won"] = (df["holm_reject"] & (df["gbm_diff"] > 0) & (df["physics_diff"] > 0)
                     & (df["gbm_ci_low"] > 0) & (df["physics_ci_low"] > 0)
                     & df["gbm_every_seed_positive"] & df["physics_every_seed_positive"])
    return df


# Protocol §7 secondary (descriptive, no multiplicity control): (model, matching reference, analysis, targets)
_DO = ("do_mgl_mean", "do_mgl_min")
_TEMP = ("temp_c_mean", "temp_c_7dadm")
SECONDARY_PAIRS = (
    ("ealstm_physdo", "ealstm", "physics_do_head", _DO),
    ("ealstm_nogauge_s0", "ealstm_s0", "static_ablation_no_gauge_derived", TARGETS),
    ("forcing_gbm_nogauge", "forcing_gbm", "static_ablation_no_gauge_derived", TARGETS),
    ("air2stream_regional_nogauge", "air2stream_regional", "static_ablation_no_gauge_derived", _TEMP),
    ("dosat_ratio[air2stream_regional]_nogauge", "dosat_ratio[air2stream_regional]",
     "static_ablation_no_gauge_derived", _DO),
    ("ealstm_ws", "ealstm", "upstream_basin_forcing", TARGETS),   # v0.10
    ("forcing_gbm_ws_bag", "forcing_gbm_bag", "upstream_basin_forcing", TARGETS),
    ("air2stream_regional_ws", "air2stream_regional", "upstream_basin_forcing", _TEMP),
)


def secondary_comparisons(site_df: pd.DataFrame, *, pairs=SECONDARY_PAIRS, metric: str = PRIMARY_METRIC,
                          n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Protocol §7 secondary paired analyses (descriptive): Δ = median_site(metric of model) −
    median_site(metric of its matching reference) on the sites scored for both, paired bootstrap
    resampling BASIN GROUPS (site resampling as sensitivity). Pairs whose models are absent are skipped."""
    rows = []
    present = set(site_df["model"]) if len(site_df) else set()
    for a, b, analysis, targets in pairs:
        if a not in present or b not in present:
            continue
        for v in targets:
            res, j = _paired(site_df, a, b, v, metric, n_boot, seed, groups=True)
            if res is None:
                continue
            sens, _ = _paired(site_df, a, b, v, metric, n_boot, seed, groups=False)
            rows.append({"analysis": analysis, "model": a, "reference": b, "var": v, "metric": f"median_site_{metric}",
                         "median_model": _nanmedian(j["a"]), "median_reference": _nanmedian(j["b"]),
                         "diff": res["diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                         "p_value": res["p_value"], "ci_unit": "basin_group", "ci_low_site": sens["ci_low"],
                         "ci_high_site": sens["ci_high"], "n_sites": int(len(j)),
                         "n_groups": int(j["basin_group"].nunique())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- prediction files

def save_predictions(path, *, model: str, preds: dict, sites, dates, block: str, meta: dict | None = None,
                     dist: dict | None = None) -> Path:
    """npz with ``pred`` (V, n_sites, n_t) original units for ``targets`` (+ optional model-space
    ``mu``/``sigma`` for probabilistic models). ``model`` is the artifact name (``<model>_s<seed>``
    for seeded deep models)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tv = [v for v in TARGETS if v in preds]
    arr = np.stack([preds[v] for v in tv]).astype("float32")
    extra = {}
    if dist:
        extra = {f"{k}__{v}": np.asarray(x, "float32") for k, dd in dist.items() for v, x in dd.items()}
    np.savez_compressed(path, pred=arr, targets=np.array(tv), sites=np.asarray(sites, str),
                        dates=np.asarray(pd.DatetimeIndex(dates).values.astype("datetime64[D]")),
                        header=json.dumps({"model": model, "block": block, **(meta or {})}, default=str), **extra)
    return path


def load_predictions(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["header"] = json.loads(str(d["header"]))
    d["targets"] = [str(t) for t in d["targets"]]
    d["sites"] = [str(s) for s in d["sites"]]
    return d


def align_predictions(pred_file: dict, data: HydroMetData, site_idx, t_idx) -> dict:
    """Map a loaded prediction file onto (site_idx, t_idx) of ``data`` (missing -> NaN)."""
    site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
    spos = {s: i for i, s in enumerate(pred_file["sites"])}
    dpos = pd.DatetimeIndex(pred_file["dates"].astype("datetime64[ns]")).get_indexer(data.dates[t_idx])
    si = np.array([spos.get(data.sites[i], -1) for i in site_idx])
    out = {}
    for k, v in enumerate(pred_file["targets"]):
        P = np.full((len(site_idx), len(t_idx)), np.nan, dtype="float32")
        ok_s, ok_t = si >= 0, dpos >= 0
        sub = pred_file["pred"][k][si[ok_s]][:, dpos[ok_t]]
        P[np.ix_(np.flatnonzero(ok_s), np.flatnonzero(ok_t))] = sub
        out[v] = P
    return out
