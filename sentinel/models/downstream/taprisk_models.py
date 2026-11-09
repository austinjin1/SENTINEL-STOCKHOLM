# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk models: Pennino-design reproduction (protocol §5) and phase-1 validation (§6–§7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import taprisk as TR
from . import wqp_track as WT

N_FORESTS = 10
TOPK_GRID = [None, 50, 25, 10]
LOGIT_GRID = [0.01, 0.1, 1.0, 10.0]
GBM_GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (7, 15) for mc in (20, 100)]
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------- Pennino-form balanced RF
class BalancedRF:
    """Pennino et al. 2020 form: R randomForest defaults (500 trees, mtry = sqrt(p)) on a class-balanced
    under-sample (all positives + an equal random draw of negatives), repeated ``n`` times; score = mean
    probability. Missing values: training-median imputation (randomForest needs complete cases)."""

    def __init__(self, n=N_FORESTS, seed=0, cols=None):
        self.n, self.seed, self.cols = n, seed, cols

    def fit(self, X: pd.DataFrame, y):
        from sklearn.ensemble import RandomForestClassifier

        self.cols = list(self.cols or X.columns)
        A = X[self.cols].to_numpy(float)
        self.med_ = np.nanmedian(A, axis=0)
        self.med_ = np.where(np.isfinite(self.med_), self.med_, 0.0)
        A = np.where(np.isnan(A), self.med_, A)
        y = np.asarray(y, int)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for i in range(self.n):
            idx = np.concatenate([pos, rng.choice(neg, size=min(len(pos), len(neg)), replace=False)])
            m = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=C.N_THREADS, random_state=self.seed * 100 + i)
            self.models_.append(m.fit(A[idx], y[idx]))
        return self

    def importances(self) -> pd.Series:
        return pd.Series(np.mean([m.feature_importances_ for m in self.models_], axis=0), index=self.cols)

    def predict(self, X: pd.DataFrame):
        A = X[self.cols].to_numpy(float)
        A = np.where(np.isnan(A), self.med_, A)
        return np.mean([m.predict_proba(A)[:, 1] for m in self.models_], axis=0)


def sens_spec(y, p, thr=0.5) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    tp, fn = int(((p > thr) & (y == 1)).sum()), int(((p <= thr) & (y == 1)).sum())
    tn, fp = int(((p <= thr) & (y == 0)).sum()), int(((p > thr) & (y == 0)).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "pcc": (tp + tn) / max(len(y), 1)}


# --------------------------------------------------------------------------- reproduction (protocol §5)
def reproduce(split: dict, sha: str) -> dict:
    """GW catchments (snapped COMIDs) with >= 1 GW CWS, label any nitrate MCL violation overlapping 2013–2017,
    train + val states only (test states never used), balanced RF ×10, 10-fold CV on the pre-balanced data."""
    from sklearn.model_selection import StratifiedKFold

    roles = split["assignments"]["state_role"]
    sy = TR.load_systems()
    sy = sy[(sy["primary_source_code"] == "GW") & sy["state_code"].astype(str).map(roles).isin(["train", "val"])]
    v = TR.load_violations()
    lo, hi = pd.Timestamp("2013-01-01"), pd.Timestamp("2017-12-31")
    m = (v["label"] == "nitrate_mcl") & v["is_health_based"].astype(bool) & (v["compl_per_begin_date"] <= hi) & (v["end"] >= lo)
    viol = set(v.loc[m, "pwsid"])
    sy = sy.assign(y=sy["pwsid"].isin(viol).astype(float))
    alive = (sy["deact"].isna() | (sy["deact"] >= lo)) & (sy["first"].isna() | (sy["first"] <= hi))
    sy = TR.join_streamcat(sy[alive])
    sy = sy[sy["comid"].notna()]
    cols = TR.streamcat_columns()
    cat = sy.groupby("comid").agg(y=("y", "max"), **{c: (c, "first") for c in cols})
    y = cat["y"].to_numpy(int)
    rng = np.random.default_rng(0)
    res = {"n_catchments": int(len(cat)), "n_violating": int(y.sum()), "reps": []}
    for rep in range(N_FORESTS):
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        idx = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        Xb, yb = cat.iloc[idx][cols], y[idx]
        p = np.empty(len(idx))
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=rep).split(Xb, yb):
            p[te] = BalancedRF(n=1, seed=rep).fit(Xb.iloc[tr], yb[tr]).predict(Xb.iloc[te])
        res["reps"].append({**sens_spec(yb, p), "auroc": C.fast_auroc(yb, p)})
    agg = {k: float(np.mean([r[k] for r in res["reps"]])) for k in ("sensitivity", "specificity", "pcc", "auroc")}
    res["mean"] = agg
    res["published"] = {"sensitivity": 0.758, "specificity": 0.815}
    res["within_tolerance"] = bool(abs(agg["sensitivity"] - 0.758) <= 0.05 and abs(agg["specificity"] - 0.815) <= 0.05)
    for k in ("sensitivity", "specificity", "pcc", "auroc"):
        C.log_metric(app=TR.APP, experiment="incumbent_reproduction/gw_nitrate_2013_2017", model="pennino_rf", metric=k,
                     value=agg[k], split="repro_2013_2017_train_val_states_cv10_balanced", n=int(2 * y.sum()), is_baseline=True,
                     data_manifests=TR.DATA_MANIFESTS, extra={"published": res["published"].get(k), "split_sha256": sha,
                                                               "within_tolerance": res["within_tolerance"]})
    C.write_json(C.run_dir(TR.APP, "reproduction") / "results.json", res)
    res["summary"] = {"n_catchments": res["n_catchments"], "n_violating": res["n_violating"], **agg,
                      "within_tolerance": res["within_tolerance"]}
    return res


# --------------------------------------------------------------------------- phase 1
def _inner(tr):
    return (tr["origin"] == 2010).to_numpy(), (tr["origin"] == 2013).to_numpy()


def phase1(task: str, split: dict, sha: str) -> dict:
    from sentinel.core.runlog import TrainingRun

    df = TR.build(task, split)
    C.assert_no_test(df)
    sc_cols = TR.streamcat_columns()
    allf = sc_cols + TR.HIST_FEATURES + TR.SYS_FEATURES + TR.AMB_FEATURES + ["lat", "lon"]
    histf = sc_cols + TR.HIST_FEATURES + TR.SYS_FEATURES + TR.AMB_FEATURES
    tr = df[df["role"] == "train"].reset_index(drop=True)
    y = tr["y"].to_numpy(float)
    a, b = _inner(tr)
    sel, scores, sets = {}, {}, {}
    for es in ("val_known", "val_new"):
        sets[es] = df[df["evalset"] == es].reset_index(drop=True)

    def log(name, p, v):
        C.log_config_tried(TR.APP, f"{task}/{name}", p, status=f"inner_auroc={v:.4f}")

    # Pennino-form incumbents: top-k by importance from a balanced RF on inner-fit rows
    for name, cols in (("pennino_rf", sc_cols), ("pennino_rf_hist", histf)):
        imp = BalancedRF(seed=0).fit(tr.loc[a, cols], y[a]).importances().sort_values(ascending=False)
        logl = []
        for k in TOPK_GRID:
            use = list(imp.index) if k is None else list(imp.index[:k])
            v = C.fast_auroc(y[b], BalancedRF(seed=0, cols=use).fit(tr.loc[a], y[a]).predict(tr.loc[b]))
            logl.append({"topk": k, "inner_auroc": v})
            log(name, {"topk": k}, v)
        best = max(logl, key=lambda r: r["inner_auroc"])["topk"]
        imp_full = BalancedRF(seed=0, cols=cols).fit(tr, y).importances().sort_values(ascending=False)
        use = list(imp_full.index) if best is None else list(imp_full.index[:best])
        m = BalancedRF(seed=0, cols=use).fit(tr, y)
        sel[name] = {"topk": best, "log": logl, "features": use}
        scores[name] = {es: m.predict(s) for es, s in sets.items()}
    # logistic
    logl = [{"C": c, "inner_auroc": C.fast_auroc(y[b], WT.fit_logistic(tr.loc[a, allf], y[a], c)(tr.loc[b, allf]))} for c in LOGIT_GRID]
    for r in logl:
        log("logistic", {"C": r["C"]}, r["inner_auroc"])
    cbest = max(logl, key=lambda r: r["inner_auroc"])["C"]
    f = WT.fit_logistic(tr[allf], y, cbest)
    sel["logistic"] = {"C": cbest, "log": logl}
    scores["logistic"] = {es: f(s[allf]) for es, s in sets.items()}
    # GBM
    logl, iters = [], {}
    for p in GBM_GRID:
        mm = WT.fit_gbm(tr.loc[a, allf], y[a], p, 0, Xv=tr.loc[b, allf], yv=y[b])
        iters[str(p)] = max(int(mm.best_iteration_ or 200), 50)
        v = C.fast_auroc(y[b], mm.predict_proba(tr.loc[b, allf])[:, 1])
        logl.append({"params": p, "inner_auroc": v})
        log("gbm_challenger", p, v)
    pbest = max(logl, key=lambda r: r["inner_auroc"])["params"]
    gb = [WT.fit_gbm(tr[allf], y, pbest, s, n_estimators=iters[str(pbest)]) for s in SEEDS]
    sel["gbm_challenger"] = {"params": pbest, "n_estimators": iters[str(pbest)], "log": logl}
    scores["gbm_challenger"] = {es: np.mean([g.predict_proba(s[allf])[:, 1] for g in gb], axis=0) for es, s in sets.items()}
    per_seed = {es: [g.predict_proba(s[allf])[:, 1] for g in gb] for es, s in sets.items()}
    # simple baselines (fit on train rows)
    p0 = y.mean()
    st = tr.groupby("state_code")["y"].agg(["sum", "size"])
    st_rate = (st["sum"] + 50 * p0) / (st["size"] + 50)
    size_cols = ["log_pop", "log_conn", "owner_federal", "owner_state", "owner_local", "owner_private", "owner_public_private", "is_gu"]
    fsz = WT.fit_logistic(tr[size_cols], y, 1.0)
    from sklearn.neighbors import BallTree

    tree = BallTree(np.radians(tr[["lat", "lon"]].to_numpy(float)), metric="haversine")
    amb = "amb_gw_median" if task.endswith("nitrate") else "amb_sw_median"
    amb_med = np.nanmedian(tr[amb].to_numpy(float))
    for es, s in sets.items():
        _, idx = tree.query(np.radians(s[["lat", "lon"]].to_numpy(float)), k=25)
        scores.setdefault("location_knn25", {})[es] = y[idx].mean(axis=1)
        scores.setdefault("state_rate", {})[es] = s["state_code"].map(st_rate).fillna(p0).to_numpy(float)
        scores.setdefault("persistence", {})[es] = (s["hist_task_3y"] + 0.01 * s["hist_task_10y"]).to_numpy(float)
        scores.setdefault("ambient_rule", {})[es] = s[amb].fillna(amb_med).to_numpy(float)
        scores.setdefault("size_source_logit", {})[es] = fsz(s[size_cols])
    simple = ["location_knn25", "state_rate", "persistence", "ambient_rule", "size_source_logit"]
    res = {"task": task, "split_sha256": sha, "selection": sel, "sets": {}, "n_train": int(len(tr)), "n_train_pos": int(y.sum())}
    out_dir = C.run_dir(TR.APP, f"phase1_{task}")
    with TrainingRun(C.WORKSTREAM, f"{TR.APP}/phase1/{task}", config={"split_sha256": sha, "task": task},
                     data=list(TR.DATA_MANIFESTS), plan_ref=C.PLAN_REF, artifacts_dir=str(out_dir)) as run:
        for es, s in sets.items():
            yy, g = s["y"].to_numpy(float), s["huc8"].astype(str).to_numpy()
            r = {"n": int(len(s)), "n_pos": int(yy.sum()), "scores": {}, "paired": {}, "pennino_metrics": {}}
            for name, sc in scores.items():
                bs = C.binary_scores(yy, sc[es], g, prob=name not in ("persistence", "ambient_rule"))
                r["scores"][name] = bs
                for met in ("auroc", "auprc"):
                    if np.isfinite(bs.get(met, np.nan)):
                        C.log_metric(app=TR.APP, experiment=f"{task}/{es}", model=name, metric=met, value=bs[met],
                                     split=f"{es}_origin2016", n=bs["n"], ci={"ci_low": bs[f"{met}_ci"][0], "ci_high": bs[f"{met}_ci"][1]},
                                     ci_method="cluster bootstrap over HUC8, B=2000", is_baseline=name not in ("gbm_challenger", "logistic"),
                                     data_manifests=TR.DATA_MANIFESTS, extra={"n_pos": bs["n_pos"]})
            for name in ("pennino_rf", "pennino_rf_hist"):
                r["pennino_metrics"][name] = sens_spec(yy, scores[name][es])
            best_simple = max(simple, key=lambda k: r["scores"][k]["auroc"])
            r["strongest_simple_baseline"] = best_simple
            for ch in ("gbm_challenger", "logistic"):
                for bname in ["pennino_rf", "pennino_rf_hist"] + simple:
                    d = C.paired(C.fast_auroc, yy, scores[ch][es], scores[bname][es], g)
                    if ch == "gbm_challenger":
                        d["per_seed_diff"] = [C.fast_auroc(yy, q) - C.fast_auroc(yy, scores[bname][es]) for q in per_seed[es]]
                    r["paired"][f"{ch}-{bname}"] = d
                    C.log_metric(app=TR.APP, experiment=f"{task}/{es}", model=ch, metric=f"auroc_diff_vs_{bname}", value=d["diff"],
                                 split=f"{es}_origin2016", n=int(len(s)), ci=d, ci_method="paired cluster bootstrap over HUC8, B=2000",
                                 data_manifests=TR.DATA_MANIFESTS, extra={"p_value": d["p_value"]})
            res["sets"][es] = r
            run.metric(f"{es}/gbm_auroc", r["scores"]["gbm_challenger"]["auroc"])
            pred = s[["pwsid", "state_code", "huc8", "origin", "y"]].copy()
            for name, sc in scores.items():
                pred[f"s_{name}"] = sc[es]
            pred.to_parquet(out_dir / f"val_predictions_{es}.parquet", index=False)
        C.write_json(out_dir / "phase1_results.json", res)
    res["summary"] = {es: {"auroc": {k: round(v["auroc"], 4) for k, v in r["scores"].items()},
                           "strongest_simple": r["strongest_simple_baseline"],
                           "gbm_vs": {k.split("-", 1)[1]: [round(r["paired"][k][x], 4) for x in ("diff", "ci_low", "ci_high")]
                                      for k in r["paired"] if k.startswith("gbm")}} for es, r in res["sets"].items()}
    return res
