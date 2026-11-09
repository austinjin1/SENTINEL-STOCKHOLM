# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Insight study E1 (protocol §9): diel-resolved warming sensitivity of DO (solubility vs metabolism).

    python -B scripts/eval/sim_e1.py [--register]

Train/cal/val-role Sim-eligible sites, WY2008–2023 (test-role sites untouched; their replication is
run only after the Sim freeze). Writes $SENTINEL_RUNS/sim/insights/e1_{sites.csv,summary.json}.
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")   # review W7: <= 6 threads

import numpy as np
import pandas as pd

from sentinel.analysis.insights import e1
from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

OUT = paths.RUNS_ROOT / "sim" / "insights"


def powell_er_means(d) -> pd.Series:
    from sentinel.models.sim.latent import load_powell

    lab = load_powell()
    lab = lab[pd.to_datetime(lab["date"]).dt.month.isin([6, 7, 8])]
    return lab.groupby("nwis_id")["er"].mean()


def explain(tab: pd.DataFrame, d) -> dict:
    """Ridge of β_bio,min on standardised static traits with basin-group 5-fold CV (R²_CV)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    from sentinel.models.sim import data as D

    pos = {s: i for i, s in enumerate(d.sites)}
    X = np.array([d.static_raw[pos[s]] for s in tab["site"]], float)
    y = tab["beta_bio_min"].to_numpy()
    ok = np.isfinite(y)
    X, y, g = X[ok], y[ok], tab["basin_group"].to_numpy()[ok]
    pred = np.full_like(y, np.nan)
    for trn, tst in GroupKFold(n_splits=5).split(X, y, g):
        fill = np.nanmean(X[trn], axis=0)             # impute with training-fold means only
        X_ = np.where(np.isfinite(X), X, fill)
        mu, sd = X_[trn].mean(0), X_[trn].std(0) + 1e-9
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((X_[trn] - mu) / sd, y[trn])
        pred[tst] = m.predict((X_[tst] - mu) / sd)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    full = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((X - mu) / sd, y)
    coef = sorted(zip(D.STATIC_NAMES, full.coef_), key=lambda t: -abs(t[1]))[:8]
    return {"r2_cv_group5": float(r2), "n_sites": int(len(y)), "top_standardised_coefs": coef}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    from sentinel.models.sim import runner

    paths.ensure(OUT)
    d = runner.load_data("validation")
    cfg = {"wy_range": list(e1.WY_RANGE), "min_summers": e1.MIN_SUMMERS, "min_summer_days": e1.MIN_SUMMER_DAYS,
           "hac_lags": 7, "n_boot": 2000}
    with TrainingRun("sim", "insight_E1", config=cfg, data=["sonde_daily", "forcings_daily", "labels_powell_metabolism"],
                     plan_ref="sim_protocol §9 E1", iteration="v1", artifacts_dir=str(OUT)) as run:
        tab = e1.run_e1(d)
        tab_np = e1.run_e1(d, exclude_provisional=True)
        tab.to_csv(OUT / "e1_sites.csv", index=False)
        summ = e1.summarise(tab, tab_np, powell_er_means(d))
        summ["explain_beta_bio_min"] = explain(tab, d)
        (OUT / "e1_summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print(json.dumps(summ, indent=1, default=str))
        run.metric("finding", float(summ["finding"]))
        if a.register:
            art = [str(OUT / "e1_summary.json"), str(OUT / "e1_sites.csv")]
            base = dict(workstream="sim", experiment="insight_E1", model="E1 per-site OLS (HAC) anomaly regression",
                        split="train_val_sites_wy2008_2023", config=cfg, artifacts=art,
                        data_manifests=["sonde_daily", "forcings_daily"])
            for key, metric in (("H_E1a_median_beta_min_minus_mean", "median_beta_min_minus_beta_mean_mgl_per_C"),
                                ("H_E1b_median_beta_bio_min", "median_beta_bio_min_mgl_per_C"),
                                ("median_beta_min", "median_beta_min_mgl_per_C"),
                                ("median_beta_mean", "median_beta_mean_mgl_per_C"),
                                ("share_min_faster_than_solubility", "share_sites_min_declines_faster_than_solubility"),
                                ("V4_heatwave_resid_min", "median_heatwave_DOmin_resid_vs_solubility_mgl")):
                v = summ[key]
                log_result(**base, metric=metric, value=v["point"], ci_low=v["ci_low"], ci_high=v["ci_high"],
                           ci_method="basin-group cluster bootstrap B=2000", n=v["n_sites"],
                           notes=f"finding={summ['finding']}")
            log_result(**base, metric="split_half_spearman_beta_min", value=summ["V2_split_half_spearman_beta_min"]["rho"],
                       n=summ["V2_split_half_spearman_beta_min"]["n_sites"])
            log_result(**base, metric="stable_psat_share_ci_contains_solubility",
                       value=summ["V1_stable_psat"]["share_ci_contains_solubility"], n=summ["V1_stable_psat"]["n_sites"])
            if "rho" in summ.get("V3_powell_er_spearman", {}):
                v = summ["V3_powell_er_spearman"]
                log_result(**base, metric="spearman_beta_bio_min_vs_powell_er", value=v["rho"], ci_low=v["ci_low"],
                           ci_high=v["ci_high"], ci_method="site bootstrap B=2000", n=v["n_sites"])
            log_result(**base, metric="r2_cv_traits_beta_bio_min", value=summ["explain_beta_bio_min"]["r2_cv_group5"],
                       n=summ["explain_beta_bio_min"]["n_sites"], notes="ridge, basin-group 5-fold CV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
