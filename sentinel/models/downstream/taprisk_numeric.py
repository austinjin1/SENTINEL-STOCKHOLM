# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 numeric track (protocol §11; user requirement 2026-11-10; secondary, descriptive, no claim):
log10 system-mean HAA5 (µg/L, UCMR4 2018–2020 distribution samples) with 90 % prediction intervals; raw-water TOC and
bromide (log10 µg/L or mg/L as reported) reported the same way. Features known by 31 Dec 2017 (decision 1 Jan 2018)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths
from . import common as C
from . import taprisk as TR

UCMR = paths.PROCESSED / "labels_ucmr"
DECISION_T = 2017
TARGETS = {"log10_haa5": ("ucmr4_haa.parquet", "HAA5"), "log10_toc": ("ucmr4_source_water.parquet", "total organic carbon"),
           "log10_bromide": ("ucmr4_source_water.parquet", "bromide")}
FLOOR = {"log10_haa5": 1.0, "log10_toc": 500.0, "log10_bromide": 10.0}   # µg/L, ~½ MRL (protocol §11 correction)
QGBM = {"num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.03, "n_estimators": 400}
QUANTILES = (0.05, 0.5, 0.95)


def targets(allowed_pwsid=None) -> pd.DataFrame:
    """W-b: rows of systems outside the frame (e.g. test states on a validation run) are dropped BEFORE any
    aggregation, so no out-of-scope value is ever read into a statistic."""
    out = []
    for name, (f, analyte) in TARGETS.items():
        d = pd.read_parquet(UCMR / f, columns=["pwsid", "analyte", "censored", "value", "mrl"])
        d = d[d["analyte"] == analyte]
        if allowed_pwsid is not None:
            d = d[d["pwsid"].isin(allowed_pwsid)]
        v = np.where(d["censored"].astype(bool), 0.5 * d["mrl"].astype(float), d["value"].astype(float))
        m = pd.Series(v, index=d.index).groupby(d["pwsid"]).mean()          # system mean of raw values
        out.append(np.log10(np.maximum(m, FLOOR[name])).rename(name))
    return pd.concat(out, axis=1)


def build(split: dict, *, include_test: bool = False, split_sha: str | None = None) -> pd.DataFrame:
    if include_test:
        from .taprisk_look import require_open

        require_open(split_sha)
    roles = split["assignments"]["state_role"]
    sy = TR.load_systems()
    sy["srole"] = sy["state_code"].astype(str).map(roles)
    keep = ["train", "val", "test"] if include_test else ["train", "val"]
    sy = sy[sy["srole"].isin(keep)].reset_index(drop=True)
    from .wqp_track import huc_of_points

    h = huc_of_points(sy["lat"].to_numpy(float), sy["lon"].to_numpy(float))
    sy["huc8"], sy["huc4"] = h["huc8"].to_numpy(), h["huc4"].to_numpy()
    f = pd.concat([sy[["pwsid", "state_code", "primary_source_code", "huc8", "huc4", "lat", "lon", "srole"]],
                   TR.system_features(sy)], axis=1)
    v = TR.load_violations()
    v = v[v["pwsid"].isin(set(f["pwsid"]))]
    f = f.merge(TR.history_at(v, DECISION_T, "dbp_mcl", "dbp"), left_on="pwsid", right_index=True, how="left")
    for c in ("hist_task_3y", "hist_task_10y", "hist_other_hb_3y", "hist_mr_group_3y"):
        f[c] = f[c].fillna(0.0)
    f = f.merge(TR.ambient_huc8(("doc",), DECISION_T), left_on="huc8", right_index=True, how="left")
    f = TR.join_streamcat(f)
    f = f.merge(targets(set(f["pwsid"])), left_on="pwsid", right_index=True, how="inner")
    f["role"] = f["srole"]
    if not include_test:
        C.assert_no_test(f)
    from .crossfit import phase1_no_upstream

    phase1_no_upstream(f.columns)
    return f


def feature_cols(df) -> list[str]:
    return TR.streamcat_columns() + TR.HIST_FEATURES + TR.SYS_FEATURES + ["amb_sw_median", "amb_gw_median", "lat", "lon"]


def fit_predict(df: pd.DataFrame, fit_roles, eval_role: str, target: str, seed: int = 0) -> pd.DataFrame:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor

    fr = df[df["role"].isin(fit_roles) & df[target].notna()].reset_index(drop=True)
    ev = df[(df["role"] == eval_role) & df[target].notna()].reset_index(drop=True)
    cols = feature_cols(df)
    y = fr[target].to_numpy(float)
    q = {}
    for a in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=seed, verbose=-1, n_jobs=C.N_THREADS, **QGBM)
        q[a] = m.fit(fr[cols], y).predict(ev[cols])
    lo, med, hi = np.minimum(q[0.05], q[0.5]), q[0.5], np.maximum(q[0.95], q[0.5])
    sc = TR.streamcat_columns()
    A = fr[sc].to_numpy(float)
    medv = np.nanmedian(A, axis=0)
    medv = np.where(np.isfinite(medv), medv, 0.0)
    rf = RandomForestRegressor(n_estimators=500, max_features="sqrt", n_jobs=C.N_THREADS, random_state=seed).fit(
        np.where(np.isnan(A), medv, A), y)
    B = ev[sc].to_numpy(float)
    st_med = fr.groupby("state_code")[target].median()
    hist_med = fr.groupby(fr["hist_task_10y"] > 0)[target].median()
    out = ev[["pwsid", "state_code", "huc8"]].copy()
    out["y"] = ev[target].to_numpy(float)
    out["qgbm_lo"], out["qgbm_med"], out["qgbm_hi"] = lo, med, hi
    rf.set_params(n_jobs=1)  # single-threaded prediction: threaded forests sum trees in arrival order (not bit-reproducible)
    out["pennino_rf_reg"] = rf.predict(np.where(np.isnan(B), medv, B))
    out["state_median"] = ev["state_code"].map(st_med).fillna(np.median(y)).to_numpy(float)
    out["history_median"] = (ev["hist_task_10y"] > 0).map(hist_med).fillna(np.median(y)).to_numpy(float)
    out["train_median"] = np.median(y)
    return out


def score(pred: pd.DataFrame) -> dict:
    from ...eval import bootstrap as B

    y, g = pred["y"].to_numpy(float), pred["huc8"].astype(str).to_numpy()
    res = {"n": int(len(pred)), "models": {}}
    for m in ("qgbm_med", "pennino_rf_reg", "state_median", "history_median", "train_median"):
        res["models"][m] = C.regression_scores(y, pred[m].to_numpy(float), g)
    cov = ((pred["qgbm_lo"] <= pred["y"]) & (pred["y"] <= pred["qgbm_hi"])).to_numpy(float)
    width = (pred["qgbm_hi"] - pred["qgbm_lo"]).to_numpy(float)
    ci = B.bootstrap_ci(lambda a, b: float(np.mean(a)), cov, cov, groups=g, n_boot=C.N_BOOT, seed=0)
    res["pi90"] = {"coverage": float(cov.mean()), "coverage_ci": [ci["ci_low"], ci["ci_high"]], "mean_width_log10": float(width.mean())}
    return res


def validate(split: dict, sha: str) -> dict:
    """Validation run: fit train states, score val states (protocol §11 numeric track; 2 configurations)."""
    from sentinel.core.runlog import TrainingRun

    df = build(split)
    out_dir = C.run_dir(TR.APP, "numeric_val_v2")
    for name, cfg in (("qgbm", {**QGBM, "quantiles": QUANTILES}), ("pennino_rf_reg", {"n_estimators": 500, "max_features": "sqrt"})):
        C.log_config_tried(TR.APP, f"numeric/{name}", cfg, status="numeric track (fixed config)")
    res = {"split_sha256": sha, "targets": {}}
    with TrainingRun(C.WORKSTREAM, f"{TR.APP}/numeric_val", config={"split_sha256": sha}, data=list(TR.DATA_MANIFESTS) + ["labels_ucmr"],
                     plan_ref=C.PLAN_REF, artifacts_dir=str(out_dir)):
        for t in TARGETS:
            p = fit_predict(df, ("train",), "val", t)
            r = score(p)
            res["targets"][t] = r
            p.to_parquet(out_dir / f"val_predictions_{t}.parquet", index=False)
            for m, sc in r["models"].items():
                for met in ("rmse", "r2"):
                    C.log_metric(app=TR.APP, experiment=f"numeric_v2/{t}/val_states", model=m, metric=met, value=sc[met],
                                 split="val_states_ucmr4_2018_2020", n=sc["n"], ci=dict(zip(("ci_low", "ci_high"), sc[f"{met}_ci"])),
                                 ci_method="cluster bootstrap over HUC8, B=2000", is_baseline=m != "qgbm_med",
                                 data_manifests=["labels_ucmr", "taprisk_catchment", "labels_sdwis"])
            C.log_metric(app=TR.APP, experiment=f"numeric_v2/{t}/val_states", model="qgbm_pi90", metric="coverage",
                         value=r["pi90"]["coverage"], split="val_states_ucmr4_2018_2020", n=r["n"],
                         ci=dict(zip(("ci_low", "ci_high"), r["pi90"]["coverage_ci"])), ci_method="cluster bootstrap over HUC8, B=2000",
                         data_manifests=["labels_ucmr"], extra={"mean_width_log10": r["pi90"]["mean_width_log10"]})
        C.write_json(out_dir / "numeric_results.json", res)
    return res
