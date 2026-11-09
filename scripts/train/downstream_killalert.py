#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D3 KillAlert driver (docs/preregistration/downstream_killalert.md).

    python -B scripts/train/downstream_killalert.py --freeze-split   # once, before any model selection
    python -B scripts/train/downstream_killalert.py --build-only     # features + counts per role (no fitting)
    python -B scripts/train/downstream_killalert.py                  # phase 1: fit on train, score val only

Test-role events (2018–2026) are dropped right after role assignment (common.drop_protected); nothing in
this script can score them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.models.downstream import common as C  # noqa: E402
from sentinel.models.downstream import crossfit  # noqa: E402
from sentinel.models.downstream import killalert as K  # noqa: E402

C.limit_threads()


def freeze(events: pd.DataFrame, v2: bool = False) -> str:
    folds = K.spatial_folds(events)
    meta = {"version": "v2 Iowa+Delaware pooled" if v2 else "v1 Iowa",
            "label_vintage_de": "labels_delaware_fishkill retrieved 2026-10-25" if v2 else None,"blocks": K.BLOCKS, "unit": "Iowa DNR fish-kill event (event_id); referent days derived in code",
            "half_window_days": K.HALF_WINDOW, "exclusion_km": K.EXCL_KM, "exclusion_days": K.EXCL_DAYS,
            "spatial": "50-km block 5-fold over train+val events (sentinel.eval.splits.spatial_kfold seed 0)",
            "label_vintage": "labels_iowa_fishkill 20261022"}
    return C.freeze_split(K.APP, {"event_role": dict(zip(events["event_id"], events["role"])),
                                  "spatial_fold50": {k: int(v) for k, v in folds.items()}}, meta,
                          path=C.PREREG / K.SPLIT_V2_PATH_NAME if v2 else None)


ITER = "base"


def build(events: pd.DataFrame, all_events: pd.DataFrame, split: dict, *, lag_end_days: int = 1) -> pd.DataFrame:
    """Case-crossover frame (train+val only). All events (incl. date-suspect ones) are built; analyses
    subset by flags. ``all_events`` (every geocoded kill, any year) is used only to exclude referent days
    near recorded kills."""
    role = split["assignments"]["event_role"]
    events = events.assign(role=events["event_id"].map(role))
    events = C.drop_protected(events)                      # test-role events never enter phase 1
    last = pd.Timestamp("2026-10-19")
    cc = K.build_case_crossover(events, all_events, last_date=last)
    cc = K.add_weather(cc, lag_end_days=lag_end_days,
                       windows=K.LONG_WINDOWS if ITER in ("it1_longwin", "v2") else (1, 3, 7, 14, 30))
    cc = K.add_gauges(cc, lag_end_days=lag_end_days)
    C.assert_no_test(cc)
    crossfit.phase1_no_upstream(cc.columns)
    return cc


def report_counts(cc: pd.DataFrame) -> dict:
    out = {}
    for r, sub in cc.groupby("role"):
        cases = sub[sub["is_case"] == 1]
        prim = cases[~cases["date_suspect"]]
        out[r] = {"strata_all": int(len(cases)), "strata_all_primary": int(len(prim)),
                  "strata_env_wq_primary": int(prim["is_env_wq"].sum()),
                  "strata_env_or_natsusp_primary": int(prim["is_env_or_natsusp"].sum()),
                  "date_suspect_events": int(cases["date_suspect"].sum()), "late_entry_events": int(cases["late_entry"].sum()),
                  "est_location_events": int(cases["est_location"].sum()),
                  "referents_total": int((sub["is_case"] == 0).sum()),
                  "referents_per_stratum_median": float(cases["n_referents"].median()),
                  "cause_known": int(cases["cause_origin"].isin(["Natural", "Anthropogenic"]).sum()),
                  "cause_natural": int((cases["cause_origin"] == "Natural").sum()),
                  "wx_cell_km_median": float(sub["wx_cell_km"].median()),
                  "gauge_q_within_30km_frac": float(sub["g_q_site"].notna().mean()),
                  "missing_frac_max_wx": float(sub[K.WX_FEATURES].isna().mean().max()),
                  "missing_frac_gauge_raw": sub[K.GAUGE_FEATURES].isna().mean().round(3).to_dict()}
    return out


TASKS = {"env_wq": "is_env_wq", "all_cause": None, "env_or_natsusp": "is_env_or_natsusp"}
VAL_SPLIT = "val_2010_2017"
BASELINES = ("season_climatology", "location_only", "year_only_clogit", "temp_anom_clogit")
CHALLENGERS = ("wx_clogit", "wx_gbm")


def _season_score(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Month climatology: P(case | month) from train rows (constant within a matched set ± a few days)."""
    rate = train.groupby(train["date"].dt.month)["is_case"].mean()
    return rows["date"].dt.month.map(rate).fillna(train["is_case"].mean()).to_numpy(float)


def _location_score(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(max_iter=1000).fit(train[["lat", "lon"]], train["is_case"])
    return m.predict_proba(rows[["lat", "lon"]])[:, 1]


def _task_rows(cc: pd.DataFrame, col: str | None, *, include_suspect: bool = False) -> pd.DataFrame:
    sub = cc if include_suspect else cc[~cc["date_suspect"]]
    return sub if col is None else sub[sub[col].astype(bool)]


def _fit_score(tr: pd.DataFrame, te: pd.DataFrame, name: str, params: dict | None) -> np.ndarray:
    if name == "season_climatology":
        return _season_score(tr, te)
    if name == "location_only":
        return _location_score(tr, te)
    kind, cols, _ = K.MODEL_SPECS[name]
    sc, _ = K.predict_seeds(kind, params, K.design(tr, cols), tr["is_case"].to_numpy(), tr["stratum"].to_numpy(),
                            K.design(te, cols))
    return sc


def _log_conc(task, name, c, *, split, year_of, sha, config=None, experiment=None, notes=""):
    ci = K.bootstrap_mean(c)
    ci_y = K.bootstrap_mean(c, clusters=year_of)
    exp = experiment or task
    C.log_metric(app=K.APP, experiment=exp if ITER == "base" else f"{exp}@{ITER}", model=name, metric="matched_concordance", value=ci["point"],
                 split=split, n=ci["n_strata"], ci=ci, ci_method="bootstrap over matched sets (events), B=2000",
                 is_baseline=name in BASELINES, config=config, data_manifests=K.DATA_MANIFESTS, notes=notes,
                 extra={"year_cluster_ci": [ci_y["ci_low"], ci_y["ci_high"]], "split_sha256": sha})
    return ci, ci_y


def phase1(cc: pd.DataFrame, cc3: pd.DataFrame, split: dict, sha: str) -> dict:
    """Fit on train-block strata, score val-block strata; sensitivities and 50-km spatial CV."""
    from sentinel.core.runlog import TrainingRun
    from sentinel.eval.bootstrap import holm_bonferroni

    out_dir = C.run_dir(K.APP, "phase1" if ITER == "base" else f"phase1_{ITER}")
    year_of = cc.groupby("stratum")["event_date"].first().dt.year
    summary = {"split_sha256": sha, "tasks": {}}
    with TrainingRun(C.WORKSTREAM, f"killalert/phase1/{ITER}", config={"split_sha256": sha, "iteration": ITER},
                     iteration=ITER,
                                                     data=K.DATA_MANIFESTS, plan_ref=C.PLAN_REF,
                                                     artifacts_dir=str(out_dir)) as run:
        for task, col in TASKS.items():
            sub = _task_rows(cc, col)
            tr, va = sub[sub["role"] == "train"], sub[sub["role"] == "val"]
            C.assert_no_test(tr, va)
            res, conc, selected = {}, {}, {}
            for name, (kind, cols, grid) in K.MODEL_SPECS.items():
                best, log = K.inner_cv_select(kind, grid, K.design(tr, cols), tr["is_case"].to_numpy(),
                                              tr["stratum"].to_numpy(), event_year=tr["event_date"].dt.year.to_numpy())
                for r in log:
                    C.log_config_tried(K.APP, f"{task}/{name}@{ITER}", r["params"], status=f"inner_cv={r['inner_cv_concordance']:.4f}")
                selected[name] = best
                res[name] = {"selected": best, "inner_cv": log}
            scores_va = {}
            for name in BASELINES + CHALLENGERS:
                s = _fit_score(tr, va, name, selected.get(name))
                scores_va[name] = s
                c = K.stratum_concordance(va["is_case"], s, va["stratum"])
                conc[name] = c
                ci, ci_y = _log_conc(task, name, c, split=VAL_SPLIT, year_of=year_of, sha=sha, config=selected.get(name))
                pooled = C.binary_scores(va["is_case"], s, va["stratum"])
                res.setdefault(name, {}).update({"matched_concordance": ci, "year_cluster_ci": ci_y, "pooled": pooled})
                if np.isfinite(pooled.get("auroc", np.nan)):
                    C.log_metric(app=K.APP, experiment=task if ITER == "base" else f"{task}@{ITER}", model=name, metric="pooled_auroc", value=pooled["auroc"],
                                 split=VAL_SPLIT, n=pooled["n"],
                                 ci={"ci_low": pooled["auroc_ci"][0], "ci_high": pooled["auroc_ci"][1]},
                                 ci_method="cluster bootstrap over matched sets, B=2000", is_baseline=name in BASELINES,
                                 data_manifests=K.DATA_MANIFESTS, extra={"split_sha256": sha})
                if name == "wx_gbm":   # per-seed spread (R7)
                    kind, cols, _ = K.MODEL_SPECS[name]
                    _, per = K.predict_seeds(kind, selected[name], K.design(tr, cols), tr["is_case"].to_numpy(),
                                             tr["stratum"].to_numpy(), K.design(va, cols))
                    res[name]["per_seed_concordance"] = [float(K.stratum_concordance(va["is_case"], p, va["stratum"]).mean())
                                                         for p in per]
                run.metric(f"{task}/{name}/concordance", ci["point"])
            # primary challenger chosen on TRAIN inner CV (not on validation)
            prim = max(CHALLENGERS, key=lambda m: max(r["inner_cv_concordance"] for r in res[m]["inner_cv"]))
            res["primary_challenger"] = prim
            if "state" in va and va["state"].nunique() > 1:      # v2 per-state secondary (same pooled models)
                res["per_state"] = {}
                for st in sorted(va["state"].unique()):
                    m_st = (va["state"] == st).to_numpy()
                    res["per_state"][st] = {}
                    for name in BASELINES + CHALLENGERS:
                        s_ = scores_va[name]
                        c = K.stratum_concordance(va["is_case"].to_numpy()[m_st], s_[m_st], va["stratum"].to_numpy()[m_st])
                        ci, _ = _log_conc(task, name, c, split=VAL_SPLIT, year_of=year_of, sha=sha,
                                          config=selected.get(name), experiment=f"{task}/state_{st}",
                                          notes=f"per-state secondary ({st}); pooled model")
                        res["per_state"][st][name] = ci
            for chal in CHALLENGERS:
                pv, paired = {}, {}
                for b in BASELINES:
                    r = K.bootstrap_mean(conc[chal], other=conc[b])
                    paired[b], pv[b] = r, r["p_value"]
                holm = holm_bonferroni(pv)
                for b in BASELINES:
                    paired[b]["holm_reject_0.05"] = holm[b]
                res[chal]["paired_vs_baselines"] = paired
            # sensitivities (same selected hyperparameters; no retuning) — base iteration only
            sens = {}
            if ITER != "base":
                cc3 = cc3.iloc[:0]
            variants = {"lag_d3": _task_rows(cc3, col), "excl_est_location": sub[~sub["est_location"]],
                        "incl_date_suspect": _task_rows(cc, col, include_suspect=True),
                        "excl_late_entry": sub[~sub["late_entry"]]}
            for vname, vs in (variants.items() if ITER == "base" else ()):
                vtr, vva = vs[vs["role"] == "train"], vs[vs["role"] == "val"]
                sens[vname] = {}
                for name in ("year_only_clogit", "temp_anom_clogit") + CHALLENGERS:
                    c = K.stratum_concordance(vva["is_case"], _fit_score(vtr, vva, name, selected[name]), vva["stratum"])
                    ci, _ = _log_conc(task, name, c, split=VAL_SPLIT, year_of=year_of, sha=sha, config=selected[name],
                                      experiment=f"{task}/sens_{vname}", notes=f"sensitivity {vname}")
                    sens[vname][name] = ci
            res["sensitivity"] = sens
            # 50-km spatial CV over development (train+val) events, hyperparameters from train inner CV
            folds = split["assignments"]["spatial_fold50"]
            fo = sub["stratum"].map(folds)
            sp = {}
            for name in ("year_only_clogit", "temp_anom_clogit") + CHALLENGERS:
                vals = []
                for k in sorted(fo.dropna().unique()):
                    te_m = (fo == k).to_numpy()
                    tr_m = (fo != k).to_numpy() & fo.notna().to_numpy()
                    a, b = sub[tr_m], sub[te_m]
                    vals.append(K.stratum_concordance(b["is_case"], _fit_score(a, b, name, selected[name]), b["stratum"]))
                c = pd.concat(vals)
                ci, _ = _log_conc(task, name, c, split="dev_2000_2017_spatial50_cv", year_of=year_of, sha=sha,
                                  config=selected[name], experiment=f"{task}/spatial50cv",
                                  notes="50-km block 5-fold CV over train+val events (secondary)")
                sp[name] = ci
            res["spatial50_cv"] = sp
            summary["tasks"][task] = res
        C.write_json(out_dir / "phase1_results.json", summary)
    return summary


def phase1_cause(events: pd.DataFrame, split: dict, sha: str) -> dict:
    """Natural vs anthropogenic cause classifier: train-block events -> val-block events."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    from sentinel.eval.metrics import auroc

    role = split["assignments"]["event_role"]
    ev = C.drop_protected(events.assign(role=events["event_id"].map(role)))
    ev = ev[~ev["date_suspect"]]
    k = K.cause_frame(ev)
    cases = K.build_case_crossover(k, k.iloc[:0], last_date=pd.Timestamp("2026-10-19"), half_window=0)
    cases = K.add_gauges(K.add_weather(cases[cases["is_case"] == 1]))
    k = k.merge(cases[["event_id", "g_q_site"] + K.WX_FEATURES + K.GAUGE_FEATURES], on="event_id", how="left")
    sc, sc_cols = None, []
    # StreamCat enters only as its own logged iteration (it1_streamcat): the base cause run (2026-10-24 23:04)
    # ran before processed/downstream_catchment existed, so adding it silently would change "base"
    if ITER == "it1_streamcat" and (paths.MANIFESTS / "downstream_catchment.json").exists() and \
            (paths.PROCESSED / "downstream_catchment" / "streamcat_comid.parquet").exists():
        links = pd.read_parquet(paths.PROCESSED / "downstream_catchment" / "unit_links.parquet")
        links = links[links["unit_source"] == "iowa_fishkill"][["unit_id", "comid"]].rename(columns={"unit_id": "event_id"})
        k = k.merge(links, on="event_id", how="left")
        sc = pd.read_parquet(paths.PROCESSED / "downstream_catchment" / "streamcat_comid.parquet")
        sc_cols = [c for c in K.STREAMCAT_COLS if c in sc]
        sc = sc[["comid"] + sc_cols]
    f = K.cause_features(k, sc)
    full = K.CAUSE_BASE_FEATURES + K.WX_FEATURES + K.GAUGE_FEATURES + sc_cols
    f[K.GAUGE_FEATURES] = f[K.GAUGE_FEATURES].fillna(0.0)
    f["g_no_gauge"] = f["g_q_site"].isna().astype(float)
    full.append("g_no_gauge")
    tr, va = f[f["role"] == "train"].reset_index(drop=True), f[f["role"] == "val"].reset_index(drop=True)
    out = {"n_train": int(len(tr)), "n_val": int(len(va)), "val_natural": int(va["y_natural"].sum()),
           "streamcat_columns": sc_cols, "models": {}}
    years = tr["date"].dt.year.to_numpy()
    yfold = {y: i % 5 for i, y in enumerate(sorted(np.unique(years)))}
    fold = np.array([yfold[y] for y in years])

    def cv(fit_pred):
        vals = []
        for kk in range(5):
            a, b = fold != kk, fold == kk
            if tr.loc[b, "y_natural"].nunique() < 2:
                continue
            vals.append(auroc(tr.loc[b, "y_natural"], fit_pred(tr[a], tr[b])))
        return float(np.mean(vals))

    def logit_fp(C_):
        def fp(a, b):
            m = LogisticRegression(max_iter=3000, C=C_).fit(a[K.CAUSE_BASE_FEATURES], a["y_natural"])
            return m.predict_proba(b[K.CAUSE_BASE_FEATURES])[:, 1]
        return fp

    def gbm_fp(p):
        def fp(a, b):
            return np.mean([lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, n_jobs=C.N_THREADS, verbose=-1,
                                               random_state=sd, subsample=0.8, subsample_freq=1, **p)
                            .fit(a[full], a["y_natural"]).predict_proba(b[full])[:, 1] for sd in K.SEEDS], axis=0)
        return fp

    grids = {"season_waterbody_logit": [(c_, logit_fp(c_)) for c_ in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)],
             "gbm_full": [(p, gbm_fp(p)) for p in ({"num_leaves": nl, "min_child_samples": mc}
                                                   for nl in (3, 7, 15) for mc in (10, 30))]}
    preds = {"prior": np.full(len(va), tr["y_natural"].mean())}
    for name, grid in grids.items():
        log = []
        for prm, fp in grid:
            v = cv(fp)
            log.append({"params": prm, "inner_cv_auroc": v})
            C.log_config_tried(K.APP, f"cause/{name}@{ITER}", {"params": prm}, status=f"inner_cv_auroc={v:.4f}")
        best = max(log, key=lambda r: r["inner_cv_auroc"])
        fp = dict((str(p), f_) for p, f_ in grid)[str(best["params"])]
        preds[name] = fp(tr, va)
        out["models"][name] = {"selected": best["params"], "inner_cv": log}
    for name, p in preds.items():
        s_ = C.binary_scores(va["y_natural"], p, va["event_id"], prob=True)
        out["models"].setdefault(name, {}).update(s_)
        if np.isfinite(s_.get("auroc", np.nan)):
            C.log_metric(app=K.APP, experiment="cause_natural" if ITER == "base" else f"cause_natural@{ITER}", model=name, metric="auroc", value=s_["auroc"],
                         split=VAL_SPLIT, n=s_["n"], ci={"ci_low": s_["auroc_ci"][0], "ci_high": s_["auroc_ci"][1]},
                         ci_method="bootstrap over events, B=2000", is_baseline=name != "gbm_full",
                         data_manifests=K.DATA_MANIFESTS + (["downstream_catchment"] if sc_cols else []),
                         extra={"split_sha256": sha, "brier": s_.get("brier")})
    out["paired_gbm_vs_season_waterbody"] = C.paired(auroc, va["y_natural"], preds["gbm_full"],
                                                     preds["season_waterbody_logit"], va["event_id"])
    C.write_json(C.run_dir(K.APP, "phase1" if ITER == "base" else f"phase1_{ITER}") / "cause_results.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze-split", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--phase1", action="store_true", help="fit train / score val (only after the pre-training review)")
    ap.add_argument("--skip-cause", action="store_true")
    ap.add_argument("--iteration", default="base", choices=["base", "it1_longwin", "it1_streamcat", "v2"])
    ap.add_argument("--cause-only", action="store_true")
    a = ap.parse_args()
    global ITER
    ITER = a.iteration
    v2 = ITER == "v2"
    K.MODEL_SPECS = K.iteration_specs({"it1_streamcat": "base", "v2": "it1_longwin"}.get(ITER, ITER))
    events, all_events = K.load_events(states=("IA", "DE") if v2 else ("IA",))
    if a.freeze_split:
        sha = freeze(events, v2=v2)
        print("frozen split sha256:", sha)
        return 0
    if v2:
        split, sha = C.load_split(K.APP, expect_sha=K.SPLIT_SHA_V2, path=C.PREREG / K.SPLIT_V2_PATH_NAME)
    else:
        split, sha = C.load_split(K.APP, expect_sha=K.SPLIT_SHA)
    if a.cause_only:
        with C.cpu_slot("killalert_cause"):
            cause = phase1_cause(events, split, sha)
        print(json.dumps({m: {k: v.get(k) for k in ("auroc", "auroc_ci", "brier")} for m, v in cause["models"].items()},
                         indent=1, default=str), flush=True)
        return 0
    with C.cpu_slot("killalert_phase1" if a.phase1 else "killalert_build"):   # one heavy job per workstream
        cc = build(events, all_events, split)
        out_dir = C.run_dir(K.APP, "build")
        cc.to_parquet(out_dir / "case_crossover_trainval.parquet", index=False)
        counts = report_counts(cc)
        C.write_json(out_dir / "build_counts.json", {"split_sha256": sha, "counts": counts})
        print(json.dumps({"split_sha256": sha, "counts": counts}, indent=1, default=str), flush=True)
        if not a.phase1:
            return 0
        cc3 = build(events, all_events, split, lag_end_days=3) if ITER == "base" else cc.iloc[:0]
        res = phase1(cc, cc3, split, sha)
        brief = {t: {m: (round(v["matched_concordance"]["point"], 3), round(v["matched_concordance"]["ci_low"], 3),
                         round(v["matched_concordance"]["ci_high"], 3))
                     for m, v in r.items() if isinstance(v, dict) and "matched_concordance" in v}
                 for t, r in res["tasks"].items()}
        print(json.dumps(brief, indent=1), flush=True)
        if not a.skip_cause:
            cause = phase1_cause(events, split, sha)
            print(json.dumps({m: {k: v.get(k) for k in ("auroc", "auroc_ci", "brier")} for m, v in cause["models"].items()},
                             indent=1, default=str), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
