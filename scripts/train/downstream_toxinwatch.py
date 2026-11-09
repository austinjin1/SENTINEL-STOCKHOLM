#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D2 ToxinWatch driver (protocol: docs/preregistration/downstream_toxinwatch.md).

    python -B scripts/train/downstream_toxinwatch.py --build-only   # split freeze check + features + counts
    python -B scripts/train/downstream_toxinwatch.py                # fit on train, score validation only

Test-role rows (test lakes, years >= 2022) are dropped right after role assignment; nothing here can
score them (phase 1).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.eval import metrics as M  # noqa: E402
from sentinel.models.downstream import common as C  # noqa: E402
from sentinel.models.downstream import toxinwatch as T  # noqa: E402

SEEDS = (0, 1, 2)
EVAL_SETS = {  # primary: first two (review C3/W2); the rest exploratory
    "val_known_2019_2021": lambda d: (d["subset"] == "val_known") & d["lake_has_train_rows"],
    "val_new_lakes_2019_2021": lambda d: d["subset"] == "val_new",
    "val_known_2019_2021_no_train_rows_EXPL": lambda d: (d["subset"] == "val_known") & ~d["lake_has_train_rows"],
    "val_new_lakes_2007_2018_EXPL": lambda d: d["subset"] == "val_new_pre2019"}
PRIMARY_SETS = ("val_known_2019_2021", "val_new_lakes_2019_2021")
INCUMBENTS = {"base": ("st_glmm_incumbent",), "it1": ("st_glmm_incumbent",),
              "it2": ("st_glmm_incumbent", "st_glmm_incumbent_v2")}


def prepare(embargo_days: int = T.EMBARGO_DAYS, iteration: str = "base"):
    sd, sites = T.load_sitedays()
    lake_role = T.lake_role_assignment(sd)
    cluster = sites.set_index("site_id")["lake_cluster"].to_dict()
    split_file = C.PREREG / T.SPLIT_FILE
    sha = C.freeze_split(T.APP, {"lake_role": lake_role, "site_cluster": cluster}, T.split_meta(), path=split_file)
    if T.SPLIT_SHA is not None:
        C.load_split(T.APP, expect_sha=T.SPLIT_SHA, path=split_file)
    rows = T.assign_rows(sd, lake_role)
    # row counts only for the test role (review W3: never write test positive counts)
    counts_all = rows[rows["label_determined"]].groupby("subset").agg(
        n=("y", "size"), n_lakes=("lake_cluster", "nunique")).reset_index()
    df = C.drop_protected(rows)
    n_prot = df.attrs["n_dropped_protected"]
    df = df[df["label_determined"]].reset_index(drop=True)
    C.assert_no_test(df)
    train = df[df["role"] == "train"]
    df["lake_has_train_rows"] = df["lake_cluster"].isin(set(train["lake_cluster"]))
    p0 = float(train["y"].mean())
    hist = T.cluster_days(df)  # train+val history only (test rows already removed)
    X, cols = T.build_features(df, hist, p0=p0, embargo_days=embargo_days, iteration=iteration,
                               sitehist=T.site_days(df))
    return df, X, cols, sha, counts_all, n_prot, p0


def build_report(df, X, cols, sha, counts_all, n_prot) -> dict:
    return {"split_sha256": sha, "counts_by_subset_incl_test": counts_all.to_dict("records"),
            "n_rows_dropped_protected": int(n_prot), "n_rows_phase1": int(len(df)),
            "phase1_counts": df.groupby("subset").agg(n=("y", "size"), n_pos=("y", "sum"),
                                                      n_lakes=("lake_cluster", "nunique"),
                                                      scum=("scum_info_any", "sum")).reset_index().to_dict("records"),
            "n_features": len(cols), "features": cols,
            "missing_frac": {c: float(X[c].isna().mean()) for c in cols},
            "wx_cell_km": X["wx_cell_km"].describe().to_dict()}


def _logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select_configs(train, Xtr, cols, wx_cols, cfg):
    """Inner time split of TRAINING rows only (fit <= 2015, score 2016–2018); 4 configs per learned model."""
    inner = (train["date"].dt.year >= T.INNER_FROM_YEAR).to_numpy()
    tr_i, ev_i = train[~inner], train[inner]
    Xa, Xb = Xtr[~inner], Xtr[inner]
    out = {}
    lr_a = T.knn_lake_rate(tr_i, tr_i, exclude_self=True)
    lr_b = T.knn_lake_rate(tr_i, ev_i, exclude_self=True)
    for inc in INCUMBENTS[cfg.get("iteration", "base")]:
        sc = {}
        for c in T.GRIDS[inc]:
            m = T.StGlmm(wx_cols=wx_cols, **c).fit(tr_i, Xa, lr_a, tr_i["y"])
            sc[json.dumps(c)] = _logloss(ev_i["y"], m.predict_proba(ev_i, Xb, lr_b))
            C.log_config_tried(T.APP, inc, {**cfg, "hp": c, "inner_logloss": sc[json.dumps(c)]})
        out[inc] = (json.loads(min(sc, key=sc.get)), sc)
    sc = {}
    for c in T.GRIDS["logistic"]:
        m = T.fit_logistic(Xa, tr_i["y"], cols, **c)
        sc[json.dumps(c)] = _logloss(ev_i["y"], m.predict_proba(Xb[cols].to_numpy(float))[:, 1])
        C.log_config_tried(T.APP, "logistic", {**cfg, "hp": c, "inner_logloss": sc[json.dumps(c)]})
    out["logistic"] = (json.loads(min(sc, key=sc.get)), sc)
    sc = {}
    for c in T.GRIDS["gbm_challenger"]:
        m = T.fit_gbm(Xtr, train, cols, seed=0, params=c)  # early stopping on the same inner split
        sc[json.dumps(c)] = m.inner_logloss_
        C.log_config_tried(T.APP, "gbm_challenger", {**cfg, "hp": c, "inner_logloss": sc[json.dumps(c)]})
    out["gbm_challenger"] = (json.loads(min(sc, key=sc.get)), sc)
    return out


def score_all(df, X, cols, p0, out_dir: Path, cfg: dict, register: bool, run) -> dict:
    sfx = "" if cfg.get("iteration", "base") == "base" else "_" + cfg["iteration"]
    learned = {"logistic", "gbm_challenger"}  # the incumbent design is unchanged by iterations
    tr = df["role"] == "train"
    train, Xtr = df[tr], X[tr]
    ev_mask = df["role"] == "val"
    ev, Xev = df[ev_mask], X[ev_mask]
    wx_cols = [c for c in cols if c.startswith("wx_")]
    sel = select_configs(train, Xtr, cols, wx_cols, cfg)
    preds = T.baseline_scores(train, Xtr, ev, Xev)
    lr_tr = T.knn_lake_rate(train, train, exclude_self=True)
    lr_ev = T.knn_lake_rate(train, ev, exclude_self=True)
    for name in INCUMBENTS[cfg.get("iteration", "base")]:
        inc = T.StGlmm(wx_cols=wx_cols, **sel[name][0]).fit(train, Xtr, lr_tr, train["y"])
        preds[name] = inc.predict_proba(ev, Xev, lr_ev)
    preds["logistic"] = T.fit_logistic(Xtr, train["y"], cols, **sel["logistic"][0]).predict_proba(
        Xev[cols].to_numpy(float))[:, 1]
    g, seed_iters = [], {}
    for s in SEEDS:
        m = T.fit_gbm(Xtr, train, cols, seed=s, params=sel["gbm_challenger"][0])
        seed_iters[s] = m.best_inner_iteration_
        g.append(m.predict_proba(Xev[cols])[:, 1])
    preds["gbm_challenger"] = np.mean(g, axis=0)
    for s, p in zip(SEEDS, g):
        preds[f"gbm_challenger_s{s}"] = p
    pd.DataFrame({"row": ev.index, "site_id": ev["site_id"], "lake_cluster": ev["lake_cluster"], "date": ev["date"],
                  "subset": ev["subset"], "scum": ev["scum_info_any"], "y": ev["y"], **preds}).to_parquet(
        out_dir / "val_predictions.parquet", index=False)
    results = {"selection": {k: {"chosen": v[0], "inner_logloss": v[1]} for k, v in sel.items()},
               "gbm_inner_best_iter": seed_iters}
    probs = {"prevalence", "climatology_month", "location_knn", "lake_history_rate", "st_glmm_incumbent",
             "st_glmm_incumbent_v2", "logistic",
             "gbm_challenger"} | {f"gbm_challenger_s{s}" for s in SEEDS}
    simple = ("persistence", "location_knn", "climatology_month", "lake_history_rate")
    comparators = simple + tuple(INCUMBENTS[cfg.get("iteration", "base")]) + ("logistic",)
    for es, f in EVAL_SETS.items():
        m = f(ev).to_numpy()
        if m.sum() == 0:
            continue
        y, grp = ev["y"].to_numpy()[m], ev["lake_cluster"].to_numpy()[m]
        res = {name: C.binary_scores(y, p[m], grp, prob=name in probs) for name, p in preds.items()}
        scum = ev["scum_info_any"].to_numpy(bool)[m]
        for strat, sm in (("scum", scum), ("no_scum", ~scum)):
            res[f"gbm_challenger|{strat}"] = C.binary_scores(y[sm], preds["gbm_challenger"][m][sm], grp[sm])
        best_simple = max(simple, key=lambda b: np.nan_to_num(res[b]["auroc"], nan=-1))
        pairs = {}
        for chal in ("gbm_challenger", "logistic") + tuple(INCUMBENTS[cfg.get("iteration", "base")]):
            for b in comparators:
                if b != chal:
                    pairs[f"{chal}-{b}"] = C.paired(M.auroc, y, preds[chal][m], preds[b][m], grp)
        pairs_pr = {f"gbm_challenger-{b}": C.paired(M.auprc, y, preds["gbm_challenger"][m], preds[b][m], grp)
                    for b in comparators}
        results[es] = {"scores": res, "best_simple_baseline": best_simple, "paired_auroc": pairs,
                       "paired_auprc": pairs_pr, "primary": es in PRIMARY_SETS}
        if register:
            for name, sc in res.items():
                if "|" in name or re.search(r"_s\d+$", name):
                    continue
                for met in ("auroc", "auprc") + (("brier",) if "brier" in sc else ()):
                    if np.isfinite(sc.get(met, np.nan)):
                        C.log_metric(app=T.APP, experiment="mc_exceed8", model=f"toxinwatch_{name}" + (sfx if name in learned else ""), metric=met,
                                     value=sc[met], split=es, n=sc["n"],
                                     ci=dict(zip(("ci_low", "ci_high"), sc[f"{met}_ci"])),
                                     ci_method="cluster bootstrap over lake clusters, B=2000",
                                     target="microcystin>8ug/L", is_baseline=name != "gbm_challenger", config=cfg,
                                     data_manifests=T.DATA_MANIFESTS,
                                     artifacts=[str(out_dir / "val_predictions.parquet")],
                                     extra={"n_pos": sc["n_pos"], "prevalence": sc["prevalence"],
                                            "n_groups": sc["n_groups"], "split_sha256": cfg.get("split_sha256"),
                                            "seed_scores": {s: res[f"gbm_challenger_s{s}"].get(met) for s in SEEDS}
                                            if name == "gbm_challenger" else None})
            for key, pr in pairs.items():
                if not np.isfinite(pr["diff"]):
                    continue
                C.log_metric(app=T.APP, experiment="mc_exceed8", model=f"toxinwatch_{key.replace('-', '_minus_')}{sfx}",
                             metric="auroc_diff", value=pr["diff"], split=es, n=int(m.sum()), ci=pr,
                             ci_method="paired cluster bootstrap (lake), B=2000", target="microcystin>8ug/L",
                             config=cfg, data_manifests=T.DATA_MANIFESTS, extra={"p_value": pr["p_value"]})
        run.metric(f"{es}_gbm_auroc", res["gbm_challenger"]["auroc"])
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--embargo-days", type=int, default=T.EMBARGO_DAYS, help="7 primary; 3 = sensitivity")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--iteration", default="base", choices=["base", "it1", "it2"],
                    help="it1 = + sampling-site history and lake sampling-intensity features; it2 = it1 + "
                         "closer INLA-analogue incumbent v2 (protocol §8)")
    a = ap.parse_args()
    C.limit_threads()
    cfg = {"app": T.APP, "iteration": a.iteration, "embargo_days": a.embargo_days, "gbm": T.GBM_PARAMS, "seeds": SEEDS, "phase": 1}
    name = "build" if a.build_only else f"val_emb{a.embargo_days}" + ("" if a.iteration == "base" else f"_{a.iteration}")
    out_dir = C.run_dir(T.APP, name)
    with (contextlib.nullcontext() if a.build_only else C.cpu_slot(f"toxinwatch_{name}")):
        if a.build_only:
            df, X, cols, sha, counts_all, n_prot, p0 = prepare(a.embargo_days, a.iteration)
            rep = build_report(df, X, cols, sha, counts_all, n_prot)
            C.write_json(out_dir / "build_report.json", rep)
            print(json.dumps({k: rep[k] for k in ("split_sha256", "counts_by_subset_incl_test", "phase1_counts",
                                                  "n_features", "wx_cell_km")}, indent=1, default=str))
            print("max missing frac:", max(rep["missing_frac"].values()))
            return 0
        with TrainingRun(C.WORKSTREAM, f"{T.APP}/{name}", config=cfg, data=list(T.DATA_MANIFESTS),
                         plan_ref=C.PLAN_REF + " D2", iteration=a.iteration, artifacts_dir=str(out_dir)) as run:
            df, X, cols, sha, counts_all, n_prot, p0 = prepare(a.embargo_days, a.iteration)
            cfg["split_sha256"] = sha
            C.write_json(out_dir / "build_report.json", build_report(df, X, cols, sha, counts_all, n_prot))
            res = score_all(df, X, cols, p0, out_dir, cfg, register=not a.no_register and a.embargo_days == T.EMBARGO_DAYS, run=run)
            C.write_json(out_dir / "val_results.json", res)
            run.note(f"split {sha[:12]}; validation only; configs tried so far {C.n_configs_tried(T.APP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
