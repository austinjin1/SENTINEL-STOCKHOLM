# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D7 TroutRefuge phase 1 (protocol §3, §4, phase-1 budget, numeric track). Validation only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import biointegrity as BI
from . import common as C
from . import troutrefuge as TT

GRIDS = {
    "rf": [{"max_features": mf, "min_samples_leaf": ml} for mf in ("sqrt", 0.2, 0.33, 0.5) for ml in (1, 5)],
    "gbm": [{"num_leaves": nl, "min_child_samples": mc, "colsample_bytree": cs}
            for nl in (7, 15) for mc in (20, 60) for cs in (0.5, 0.8)],
    "logistic": [{"C": c} for c in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0)],
}
SEEDS = (0, 1, 2)
QGBM = {"num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.03, "n_estimators": 400}


def _fit_predict(kind, Xtr, ytr, Xev, params, seed=0):
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, **params))
        return m.fit(Xtr.astype(float), ytr).predict_proba(Xev[Xtr.columns].astype(float))[:, 1]
    return BI.fit_predict(kind, Xtr, ytr, Xev, seed=seed, params=params)[1]


def inner_select(kind, X, tr, family, log_name):
    scores = []
    y = tr["y_cold"].to_numpy(float)
    for p in GRIDS[family]:
        pred = np.full(len(tr), np.nan)
        for eco in sorted(tr["ag_eco9"].dropna().unique()):
            te = (tr["ag_eco9"] == eco).to_numpy()
            if te.all() or not te.any():
                continue
            pred[te] = _fit_predict(kind, X[~te], y[~te], X[te], p)
        ok = np.isfinite(pred)
        scores.append(C.fast_auroc(y[ok], pred[ok]))
        C.log_config_tried(TT.APP, log_name, p, status=f"inner_loeo_auroc={scores[-1]:.4f}")
    k = int(np.nanargmax(scores))
    return GRIDS[family][k], scores


def airwater_rule(tr: pd.DataFrame, ev: pd.DataFrame, sc_tr: pd.DataFrame, sc_ev: pd.DataFrame):
    """OLS of Jul–Aug field water temperature on the watershed PRISM tmax normal (thermal incumbent 2)."""
    m = tr["date"].dt.month.isin([7, 8]).to_numpy() & tr["temp_field_degc"].notna().to_numpy() & sc_tr["tmax9120ws"].notna().to_numpy()
    x, t = sc_tr.loc[m, "tmax9120ws"].to_numpy(float), tr.loc[m, "temp_field_degc"].to_numpy(float)
    b1, b0 = np.polyfit(x, t, 1)
    C.log_config_tried(TT.APP, "airwater_rule", {"b0": float(b0), "b1": float(b1), "n": int(m.sum())}, status="OLS fit")
    return b0 + b1 * sc_ev["tmax9120ws"].to_numpy(float), (float(b0), float(b1), int(m.sum()))


def phase1(split: dict, sha: str) -> dict:
    from sentinel.core.runlog import TrainingRun

    u, blocks = TT.build_frame()
    X_ch = pd.concat([blocks["geo"], blocks["streamcat"], blocks["wx"]], axis=1)
    X_inc = blocks["streamcat"]
    X_geo = blocks["geo"]
    for X in (X_ch, X_inc, X_geo):
        TT.assert_no_modelled_temperature(X.columns)
    tr_m, va_m = (u["role"] == "train").to_numpy(), (u["role"] == "val").to_numpy()
    tr, va = u[tr_m].reset_index(drop=True), u[va_m].reset_index(drop=True)
    Xs = {k: (X[tr_m].reset_index(drop=True), X[va_m].reset_index(drop=True)) for k, X in
          (("ch", X_ch), ("inc", X_inc), ("geo", X_geo))}
    y = tr["y_cold"].to_numpy(float)
    out_dir = C.run_dir(TT.APP, "phase1")
    res = {"split_sha256": sha, "selection": {}, "sets": {}}
    with TrainingRun(C.WORKSTREAM, f"{TT.APP}/phase1", config={"split_sha256": sha}, data=list(TT.DATA_MANIFESTS) + ["d7_mast"],
                     plan_ref=C.PLAN_REF, artifacts_dir=str(out_dir)) as run:
        specs = {"rf_occupancy_incumbent": ("rf", "inc", "rf"), "feature_matched_rf": ("rf", "ch", "rf"),
                 "gbm_challenger": ("gbm", "ch", "gbm"), "logistic_challenger": ("logistic", "ch", "logistic")}
        sel = {}
        for name, (kind, xs, fam) in specs.items():
            p, sc = inner_select(kind, Xs[xs][0], tr, fam, name)
            sel[name] = p
            res["selection"][name] = {"params": p, "inner_auroc": sc, "best_inner_auroc": float(np.nanmax(sc))}
        primary = max(("gbm_challenger", "logistic_challenger"), key=lambda k: res["selection"][k]["best_inner_auroc"])
        res["primary_challenger"] = primary
        preds, per_seed = {}, {}
        for name, (kind, xs, fam) in specs.items():
            Xtr, Xva = Xs[xs]
            seeds = SEEDS if kind in ("gbm", "rf") else (0,)
            ps = [_fit_predict(kind, Xtr, y, Xva, sel[name], seed=s) for s in seeds]
            preds[name], per_seed[name] = np.mean(ps, axis=0), ps
        preds["logistic_geo"] = _fit_predict("logistic", Xs["geo"][0], y, Xs["geo"][1], {"C": 1.0})
        preds["ecoregion_rate"] = BI.ecoregion_rate(tr, va, y="y_cold")
        preds["location_knn"] = BI.knn_rate(tr, va, k=15, y="y_cold")
        preds["mast_rule"] = -va["mast_c"].to_numpy(float)
        aw, aw_fit = airwater_rule(tr, va, Xs["inc"][0], Xs["inc"][1])
        preds["airwater_rule"] = -aw
        res["airwater_fit"] = aw_fit
        fit_locs = set(tr["location_id"])
        sets = {"val_newloc": ~va["location_id"].isin(fit_locs).to_numpy(), "val_all": np.ones(len(va), bool)}
        for sname, m in sets.items():
            yy, g4 = va["y_cold"].to_numpy(float)[m], va["huc4"].astype(str).fillna("NA").to_numpy()[m]
            sc = {k: C.binary_scores(yy, v[m], g4, prob=k not in ("mast_rule", "airwater_rule")) for k, v in preds.items()}
            simple = max(("ecoregion_rate", "location_knn", "logistic_geo"), key=lambda k: sc[k]["auroc"])
            comps = ["rf_occupancy_incumbent", "mast_rule", "airwater_rule", "feature_matched_rf", simple]
            r = {"n": int(m.sum()), "n_pos": int(yy.sum()), "strongest_simple": simple, "scores": sc, "paired": {}}
            for ch in ("gbm_challenger", "logistic_challenger"):
                pv = {}
                for b in comps:
                    d = C.paired(C.fast_auroc, yy, preds[ch][m], preds[b][m], g4)
                    ok = np.isfinite(preds[b][m])
                    d["per_seed_diff"] = [C.fast_auroc(yy[ok], q[m][ok]) - C.fast_auroc(yy[ok], preds[b][m][ok]) for q in per_seed[ch]]
                    r["paired"][f"{ch}-{b}"] = d
                    pv[b] = d["p_value"]
                    C.log_metric(app=TT.APP, experiment=f"phase1/{sname}", model=ch, metric=f"auroc_diff_vs_{b}", value=d["diff"],
                                 split=f"{sname}_nrsa1819", n=int(ok.sum()), ci=d, ci_method="paired cluster bootstrap over HUC4, B=2000",
                                 data_manifests=list(TT.DATA_MANIFESTS), extra={"p_value": d["p_value"]})
                from sentinel.eval.bootstrap import holm_bonferroni

                h = holm_bonferroni(pv)
                for b in comps:
                    r["paired"][f"{ch}-{b}"]["holm_reject"] = bool(h[b])
            for k, v in sc.items():
                C.log_metric(app=TT.APP, experiment=f"phase1/{sname}", model=k, metric="auroc", value=v["auroc"], split=f"{sname}_nrsa1819",
                             n=v["n"], ci={"ci_low": v["auroc_ci"][0], "ci_high": v["auroc_ci"][1]}, ci_method="cluster bootstrap over HUC4, B=2000",
                             is_baseline=k not in ("gbm_challenger", "logistic_challenger"), data_manifests=list(TT.DATA_MANIFESTS))
            res["sets"][sname] = r
            run.metric(f"{sname}/{primary}/auroc", sc[primary]["auroc"])
        mast20 = (va["mast_c"] <= 20).astype(float).where(va["mast_c"].notna())
        res["mast_binary_rule"] = {"sens": float(((mast20 == 1) & (va["y_cold"] == 1)).sum() / max((va["y_cold"] == 1).sum(), 1)),
                                   "spec": float(((mast20 == 0) & (va["y_cold"] == 0)).sum() / max((va["y_cold"] == 0).sum(), 1))}
        pd.DataFrame({"visit_id": va["visit_id"], "y_cold": va["y_cold"], "huc4": va["huc4"], "newloc": sets["val_newloc"],
                      **{f"s_{k}": v for k, v in preds.items()}}).to_parquet(out_dir / "val_predictions.parquet", index=False)
        res["numeric"] = numeric_track(u, blocks, tr_m, va_m, sets["val_newloc"], aw_fit)
        C.write_json(out_dir / "phase1_results.json", res)
    return res


def numeric_track(u, blocks, tr_m, va_m, newloc_va, aw_fit) -> dict:
    import lightgbm as lgb

    from sentinel.eval import bootstrap as B

    X = pd.concat([blocks["geo"], blocks["streamcat"], blocks["wx"], blocks["wx7"]], axis=1)
    X["doy"] = u["date"].dt.dayofyear.to_numpy(float)
    TT.assert_no_modelled_temperature(X.columns)
    jja = u["date"].dt.month.isin([7, 8]).to_numpy() & u["temp_field_degc"].notna().to_numpy()
    tm, vm = tr_m & jja, va_m & jja
    y = u.loc[tm, "temp_field_degc"].to_numpy(float)
    q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=0, verbose=-1, n_jobs=C.N_THREADS, **QGBM)
         .fit(X[tm], y).predict(X[vm]) for a in (0.05, 0.5, 0.95)}
    C.log_config_tried(TT.APP, "numeric/qgbm_temp", {**QGBM, "quantiles": [0.05, 0.5, 0.95]}, status="numeric track (fixed)")
    va_idx = np.flatnonzero(va_m)
    newloc_full = np.zeros(len(u), bool)
    newloc_full[va_idx[newloc_va]] = True
    nl = newloc_full[vm]
    yv = u.loc[vm, "temp_field_degc"].to_numpy(float)
    g = u.loc[vm, "huc4"].astype(str).fillna("NA").to_numpy()
    eco_mean = u.loc[tm].groupby("ag_eco9")["temp_field_degc"].mean()
    comps = {"qgbm_med": q[0.5], "mast": u.loc[vm, "mast_c"].to_numpy(float),
             "airwater_rule": aw_fit[0] + aw_fit[1] * blocks["streamcat"].loc[vm, "tmax9120ws"].to_numpy(float),
             "ecoregion_mean": u.loc[vm, "ag_eco9"].map(eco_mean).to_numpy(float)}
    out = {}
    for sname, m in (("val_newloc_julaug", nl), ("val_all_julaug", np.ones(len(yv), bool))):
        r = {k: C.regression_scores(yv[m], v[m], g[m]) for k, v in comps.items()}
        lo, hi = np.minimum(q[0.05], q[0.5])[m], np.maximum(q[0.95], q[0.5])[m]
        cov = ((lo <= yv[m]) & (yv[m] <= hi)).astype(float)
        ci = B.bootstrap_ci(lambda a, b: float(np.mean(a)), cov, cov, groups=g[m], n_boot=C.N_BOOT, seed=0)
        out[sname] = {"n": int(m.sum()), "models": r, "pi90": {"coverage": float(cov.mean()), "coverage_ci": [ci["ci_low"], ci["ci_high"]],
                                                                "mean_width_c": float((hi - lo).mean())}}
        for k, v in r.items():
            for met in ("rmse", "r2"):
                C.log_metric(app=TT.APP, experiment=f"numeric/temp_field/{sname}", model=k, metric=met, value=v[met],
                             split=f"{sname}_nrsa1819", n=v["n"], ci=dict(zip(("ci_low", "ci_high"), v[f"{met}_ci"])),
                             ci_method="cluster bootstrap over HUC4, B=2000", is_baseline=k != "qgbm_med",
                             data_manifests=list(TT.DATA_MANIFESTS))
    return out


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def cqr_margin(y, lo, hi, level: float = 0.9) -> float:
    """Conformalised quantile regression margin (Romano et al. 2019): the ceil((n+1)*level)/n empirical quantile of
    E = max(lo - y, y - hi) over out-of-fold calibration rows."""
    e = np.maximum(np.asarray(lo, float) - np.asarray(y, float), np.asarray(y, float) - np.asarray(hi, float))
    e = np.sort(e[np.isfinite(e)])
    k = int(np.ceil((len(e) + 1) * level))
    return float(e[min(k, len(e)) - 1])


def phase1b(split: dict, sha: str) -> dict:
    """Protocol §8 amendment 2026-11-10: MSST thermal incumbent (comparator only), Holm over six, numeric MSST
    comparator and CQR-calibrated prediction intervals. No learned family is re-tuned: phase-1 selections and saved
    validation predictions are reused; the LightGBM challenger is refitted at its selected configuration only to get
    per-seed predictions (reproduction is checked against the saved mean)."""
    import json

    import lightgbm as lgb

    from sentinel.core.runlog import TrainingRun
    from sentinel.eval import bootstrap as B

    p1_dir = C.run_dir(TT.APP, "phase1")
    p1 = json.loads((p1_dir / "phase1_results.json").read_text())
    if p1["split_sha256"] != sha:
        raise RuntimeError("phase-1 results belong to another split")
    saved = pd.read_parquet(p1_dir / "val_predictions.parquet")
    u, blocks = TT.build_frame()
    X_ch = pd.concat([blocks["geo"], blocks["streamcat"], blocks["wx"]], axis=1)
    TT.assert_no_modelled_temperature(X_ch.columns)
    tr_m, va_m = (u["role"] == "train").to_numpy(), (u["role"] == "val").to_numpy()
    tr, va = u[tr_m].reset_index(drop=True), u[va_m].reset_index(drop=True)
    if list(va["visit_id"]) != list(saved["visit_id"]):
        raise RuntimeError("validation rows differ from the saved phase-1 predictions")
    out_dir = C.run_dir(TT.APP, "phase1b")
    res = {"split_sha256": sha, "phase1_results_sha256": C.sha256_text((p1_dir / "phase1_results.json").read_text()), "sets": {}}
    with TrainingRun(C.WORKSTREAM, f"{TT.APP}/phase1b", config={"split_sha256": sha}, data=list(TT.DATA_MANIFESTS) + ["d7_mast"],
                     plan_ref=C.PLAN_REF, artifacts_dir=str(out_dir)) as run:
        C.log_config_tried(TT.APP, "msst_rule", {"score": "-MSST (Hill 2013), year-matched", "binary": "MSST <= 20 C"},
                           status="comparator only (amendment 2026-11-10)")
        primary = p1["primary_challenger"]
        sel = p1["selection"]["gbm_challenger"]["params"]
        y = tr["y_cold"].to_numpy(float)
        Xtr, Xva = X_ch[tr_m].reset_index(drop=True), X_ch[va_m].reset_index(drop=True)
        seeds = [_fit_predict("gbm", Xtr, y, Xva, sel, seed=s) for s in SEEDS]
        res["gbm_repro_max_abs_diff"] = float(np.max(np.abs(np.mean(seeds, axis=0) - saved["s_gbm_challenger"].to_numpy(float))))
        per_seed = {"gbm_challenger": seeds, "logistic_challenger": [saved["s_logistic_challenger"].to_numpy(float)]}
        msst = -va["msst_c"].to_numpy(float)
        yv, g_all = saved["y_cold"].to_numpy(float), saved["huc4"].astype(str).fillna("NA").to_numpy()
        for sname, m in (("val_newloc", saved["newloc"].to_numpy(bool)), ("val_all", np.ones(len(saved), bool))):
            yy, g4 = yv[m], g_all[m]
            r = {"msst_rule": C.binary_scores(yy, msst[m], g4, prob=False), "paired": {}, "holm_six": {}}
            ok = np.isfinite(msst[m])
            for ch in ("gbm_challenger", "logistic_challenger"):
                s_ch = saved[f"s_{ch}"].to_numpy(float)[m]
                d = C.paired(C.fast_auroc, yy, s_ch, msst[m], g4)
                d["per_seed_diff"] = [C.fast_auroc(yy[ok], q[m][ok]) - C.fast_auroc(yy[ok], msst[m][ok]) for q in per_seed[ch]]
                r["paired"][f"{ch}-msst_rule"] = d
                pv = {k.split("-", 1)[1]: v["p_value"] for k, v in p1["sets"][sname]["paired"].items() if k.startswith(ch + "-")}
                pv["msst_rule"] = d["p_value"]
                h = B.holm_bonferroni(pv)
                r["holm_six"][ch] = {"p_values": pv, "reject": {k: bool(v) for k, v in h.items()}}
                C.log_metric(app=TT.APP, experiment=f"phase1b/{sname}", model=ch, metric="auroc_diff_vs_msst_rule", value=d["diff"],
                             split=f"{sname}_nrsa1819", n=int(ok.sum()), ci=d, ci_method="paired cluster bootstrap over HUC4, B=2000",
                             data_manifests=list(TT.DATA_MANIFESTS), extra={"p_value": d["p_value"]})
            v = r["msst_rule"]
            C.log_metric(app=TT.APP, experiment=f"phase1b/{sname}", model="msst_rule", metric="auroc", value=v["auroc"],
                         split=f"{sname}_nrsa1819", n=v["n"], ci={"ci_low": v["auroc_ci"][0], "ci_high": v["auroc_ci"][1]},
                         ci_method="cluster bootstrap over HUC4, B=2000", is_baseline=True, data_manifests=list(TT.DATA_MANIFESTS))
            res["sets"][sname] = r
        b20 = (va["msst_c"] <= 20).astype(float).where(va["msst_c"].notna())
        res["msst_binary_rule"] = {"n": int(b20.notna().sum()),
                                   "sens": float(((b20 == 1) & (va["y_cold"] == 1)).sum() / max(((va["y_cold"] == 1) & b20.notna()).sum(), 1)),
                                   "spec": float(((b20 == 0) & (va["y_cold"] == 0)).sum() / max(((va["y_cold"] == 0) & b20.notna()).sum(), 1))}
        pd.DataFrame({"visit_id": va["visit_id"], "s_msst_rule": msst,
                      **{f"s_gbm_seed{s}": q for s, q in zip(SEEDS, seeds)}}).to_parquet(out_dir / "val_predictions_1b.parquet", index=False)

        # ---- numeric track: MSST comparator + CQR-calibrated 90 % interval ----
        X = pd.concat([blocks["geo"], blocks["streamcat"], blocks["wx"], blocks["wx7"]], axis=1)
        X["doy"] = u["date"].dt.dayofyear.to_numpy(float)
        TT.assert_no_modelled_temperature(X.columns)
        jja = u["date"].dt.month.isin([7, 8]).to_numpy() & u["temp_field_degc"].notna().to_numpy()
        tm, vm = tr_m & jja, va_m & jja
        yt = u.loc[tm, "temp_field_degc"].to_numpy(float)

        def qfit(Xa, ya, Xb):
            return {a: lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=0, verbose=-1, n_jobs=C.N_THREADS, **QGBM)
                    .fit(Xa, ya).predict(Xb) for a in (0.05, 0.5, 0.95)}

        Xt, eco = X[tm].reset_index(drop=True), u.loc[tm, "ag_eco9"].reset_index(drop=True)
        lo_oof, hi_oof = np.full(len(yt), np.nan), np.full(len(yt), np.nan)
        for e in sorted(eco.dropna().unique()):
            te = (eco == e).to_numpy()
            if te.all() or not te.any():
                continue
            qq = qfit(Xt[~te], yt[~te], Xt[te])
            lo_oof[te], hi_oof[te] = np.minimum(qq[0.05], qq[0.5]), np.maximum(qq[0.95], qq[0.5])
        Q = cqr_margin(yt, lo_oof, hi_oof, 0.9)
        C.log_config_tried(TT.APP, "numeric/qgbm_temp_cqr", {**QGBM, "cqr_level": 0.9, "calibration": "inner LOEO OOF, training cycles",
                                                             "margin_c": Q, "n_cal": int(np.isfinite(lo_oof).sum())},
                           status="numeric track PI calibration (amendment 2026-11-10)")
        q = qfit(X[tm], yt, X[vm])
        va_idx = np.flatnonzero(va_m)
        newloc_full = np.zeros(len(u), bool)
        newloc_full[va_idx[saved["newloc"].to_numpy(bool)]] = True
        nl = newloc_full[vm]
        yn = u.loc[vm, "temp_field_degc"].to_numpy(float)
        g = u.loc[vm, "huc4"].astype(str).fillna("NA").to_numpy()
        ms = u.loc[vm, "msst_c"].to_numpy(float)
        num = {"cqr_margin_c": Q, "n_cal": int(np.isfinite(lo_oof).sum()),
               "oof_uncalibrated_coverage": float(np.nanmean(((lo_oof <= yt) & (yt <= hi_oof))[np.isfinite(lo_oof)]))}
        for sname, m in (("val_newloc_julaug", nl), ("val_all_julaug", np.ones(len(yn), bool))):
            r = {"qgbm_med": C.regression_scores(yn[m], q[0.5][m], g[m]), "msst": C.regression_scores(yn[m], ms[m], g[m])}
            d = C.paired(_rmse, yn[m], q[0.5][m], ms[m], g[m], higher_is_better=False)
            lo0, hi0 = np.minimum(q[0.05], q[0.5])[m], np.maximum(q[0.95], q[0.5])[m]
            pi = {}
            for tag, lo, hi in (("uncalibrated", lo0, hi0), ("cqr", lo0 - Q, hi0 + Q)):
                cov = ((lo <= yn[m]) & (yn[m] <= hi)).astype(float)
                ci = B.bootstrap_ci(lambda a, b: float(np.mean(a)), cov, cov, groups=g[m], n_boot=C.N_BOOT, seed=0)
                pi[tag] = {"coverage": float(cov.mean()), "coverage_ci": [ci["ci_low"], ci["ci_high"]], "mean_width_c": float((hi - lo).mean())}
                C.log_metric(app=TT.APP, experiment=f"numeric/temp_field/{sname}", model=f"qgbm_pi90_{tag}", metric="pi90_coverage",
                             value=pi[tag]["coverage"], split=f"{sname}_nrsa1819", n=int(m.sum()), ci=ci,
                             ci_method="cluster bootstrap over HUC4, B=2000", data_manifests=list(TT.DATA_MANIFESTS),
                             extra={"mean_width_c": pi[tag]["mean_width_c"]})
            num[sname] = {"n": int(m.sum()), "models": r, "rmse_diff_qgbm_minus_msst": d, "pi90": pi}
            for met in ("rmse", "r2"):
                v = r["msst"]
                C.log_metric(app=TT.APP, experiment=f"numeric/temp_field/{sname}", model="msst", metric=met, value=v[met],
                             split=f"{sname}_nrsa1819", n=v["n"], ci=dict(zip(("ci_low", "ci_high"), v[f"{met}_ci"])),
                             ci_method="cluster bootstrap over HUC4, B=2000", is_baseline=True, data_manifests=list(TT.DATA_MANIFESTS))
        res["numeric"] = num
        run.metric(f"val_newloc/{primary}/auroc_diff_vs_msst_rule", res["sets"]["val_newloc"]["paired"][f"{primary}-msst_rule"]["diff"])
        C.write_json(out_dir / "phase1b_results.json", res)
    return res
