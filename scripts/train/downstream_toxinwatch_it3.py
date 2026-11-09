#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D2 ToxinWatch it3: WA training rows + D2n national training rows (protocol §8 it3; validation only).

    python -B scripts/train/downstream_toxinwatch_it3.py

Learners (logistic, GBM) and pooled incumbent variants are fit on WA train + national train rows; the it2
WA-only incumbents and the WA-only simple baselines are refit unchanged. Scored on the WA validation sets.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.eval import metrics as M  # noqa: E402
from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads()

from sentinel.models.downstream import d2n_microcystin as N  # noqa: E402
from sentinel.models.downstream import toxinwatch as T  # noqa: E402
from sentinel.models.downstream import wqp_track as WT  # noqa: E402

_spec = importlib.util.spec_from_file_location("tw_driver", ROOT / "scripts" / "train" / "downstream_toxinwatch.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

ITER = "it3"
NAT_MAP = {"tw_n_prior": "tw_prior_n", "tw_n_exc_prior": "tw_prior_n_exceed", "tw_days_since_last": "tw_days_since_last",
           "tw_last_exceed": "tw_last_exceed", "tw_last_log_value": "tw_last_log_value",
           "tw_prior_max_log_365": "tw_prior_max_log_365", "tw_exc_365": "tw_prior_exceed_365",
           "tw_n_365": "tw_prior_n_365", "tw_same_year_exc": "tw_prior_exceed_same_year", "tw_n_30": "tw_lake_n_30d",
           "lat": "lat", "lon": "lon", "doy_sin": "doy_sin", "doy_cos": "doy_cos"}


def national_rows(cols: list[str], p0_wa: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split, sha = C.load_split(N.APP, expect_sha=N.SPLIT_SHA)
    rows, X, _ = WT.build_frame(N.CFG, split)
    m = (rows["role"] == "train").to_numpy()
    rows, X = rows[m].reset_index(drop=True), X[m].reset_index(drop=True)
    C.assert_no_test(rows)
    Xn = pd.DataFrame(index=rows.index)
    for a, b in NAT_MAP.items():
        Xn[b] = X[a].to_numpy(float)
    Xn["tw_prior_rate_s"] = (Xn["tw_prior_n_exceed"] + 2 * p0_wa) / (Xn["tw_prior_n"] + 2)
    Xn["is_river"] = 0.0
    Xn["src_national"] = 1.0
    for c in cols:
        if c.startswith("wx_"):
            Xn[c] = X[c].to_numpy(float) if c in X else np.nan
    for c in cols:
        if c not in Xn:
            Xn[c] = np.nan          # WA-only site/intensity features
    Xn["wx_cell_km"] = X["wx_cell_km"].to_numpy(float)
    far = Xn["wx_cell_km"].to_numpy() > T.MAX_WX_KM
    Xn.loc[far, [c for c in Xn.columns if c.startswith("wx_") and c != "wx_cell_km"]] = np.nan
    lat_lon = X[["lat", "lon"]].to_numpy(float)
    df = pd.DataFrame({"lake_cluster": "nat:" + rows["cluster"].astype(str), "site_id": "nat:" + rows["cluster"].astype(str),
                       "date": rows["date"], "y": rows["y"].astype(float), "role": "train", "subset": "train_national",
                       "latitude": lat_lon[:, 0], "longitude": lat_lon[:, 1], "scum_info_any": False,
                       "label_determined": True, "src_national": 1.0})
    return df, Xn, sha


def main() -> int:
    df, X, cols, sha, counts_all, n_prot, p0 = D.prepare(T.EMBARGO_DAYS, "it1")
    out_dir = C.run_dir(T.APP, "val_emb7_it3")
    cfg = {"app": T.APP, "iteration": ITER, "embargo_days": T.EMBARGO_DAYS, "gbm": T.GBM_PARAMS, "seeds": D.SEEDS,
           "phase": 1, "split_sha256": sha}
    D.INCUMBENTS[ITER] = ("st_glmm_incumbent", "st_glmm_incumbent_v2")
    with C.cpu_slot("toxinwatch_val_emb7_it3"), TrainingRun(C.WORKSTREAM, f"{T.APP}/val_emb7_it3", config=cfg,
                                                           data=list(T.DATA_MANIFESTS) + list(N.DATA_MANIFESTS),
                                                           plan_ref=C.PLAN_REF + " D2", iteration=ITER,
                                                           artifacts_dir=str(out_dir)) as run:
        nat, Xn, nat_sha = national_rows(cols, p0)
        cfg["d2n_split_sha256"] = nat_sha
        df = df.assign(src_national=0.0)
        X = X.assign(src_national=0.0)
        lcols = cols + ["src_national"]
        tr = (df["role"] == "train").to_numpy()
        wa_train, Xwa = df[tr].reset_index(drop=True), X[tr].reset_index(drop=True)
        ev, Xev = df[df["role"] == "val"], X[df["role"] == "val"]
        pool = pd.concat([wa_train, nat], ignore_index=True)
        Xpool = pd.concat([Xwa.reset_index(drop=True), Xn.reindex(columns=X.columns)], ignore_index=True)
        Xpool.index = pool.index
        wx_cols = [c for c in cols if c.startswith("wx_")]
        # selection: WA-only incumbents (it2 design) and pooled incumbents + pooled learners
        sel_wa = D.select_configs(wa_train, Xwa, cols, wx_cols, {**cfg, "fit_rows": "wa"})
        sel_pool = D.select_configs(pool, Xpool, lcols, wx_cols, {**cfg, "fit_rows": "wa+national"})
        preds = T.baseline_scores(wa_train, Xwa, ev, Xev)          # simple baselines: WA rows only (= it2)
        for fit_rows, train, Xtr, sel, sfx in (("wa", wa_train, Xwa, sel_wa, ""), ("pool", pool, Xpool, sel_pool, "_pooled")):
            lr_tr = T.knn_lake_rate(train, train, exclude_self=True)
            lr_ev = T.knn_lake_rate(train, ev, exclude_self=True)
            for name in D.INCUMBENTS[ITER]:
                inc = T.StGlmm(wx_cols=wx_cols, **sel[name][0]).fit(train, Xtr, lr_tr, train["y"])
                preds[name + sfx] = inc.predict_proba(ev, Xev, lr_ev)
        preds["logistic"] = T.fit_logistic(Xpool, pool["y"], lcols, **sel_pool["logistic"][0]).predict_proba(
            Xev[lcols].to_numpy(float))[:, 1]
        g = []
        for s in D.SEEDS:
            m = T.fit_gbm(Xpool, pool, lcols, seed=s, params=sel_pool["gbm_challenger"][0])
            g.append(m.predict_proba(Xev[lcols])[:, 1])
        preds["gbm_challenger"] = np.mean(g, axis=0)
        for s, p in zip(D.SEEDS, g):
            preds[f"gbm_challenger_s{s}"] = p
        pd.DataFrame({"row": ev.index, "lake_cluster": ev["lake_cluster"], "date": ev["date"], "subset": ev["subset"],
                      "y": ev["y"], **preds}).to_parquet(out_dir / "val_predictions.parquet", index=False)
        incs = [n + s for n in D.INCUMBENTS[ITER] for s in ("", "_pooled")]
        simple = ("persistence", "location_knn", "climatology_month", "lake_history_rate")
        res = {"selection_wa": {k: v[0] for k, v in sel_wa.items()}, "selection_pool": {k: v[0] for k, v in sel_pool.items()},
               "n_national_train_rows": int(len(nat)), "n_national_train_pos": int(nat["y"].sum())}
        for es in D.PRIMARY_SETS:
            msk = D.EVAL_SETS[es](ev).to_numpy()
            y, grp = ev["y"].to_numpy()[msk], ev["lake_cluster"].to_numpy()[msk]
            sc = {k: C.binary_scores(y, p[msk], grp) for k, p in preds.items()}
            strongest_inc = max(incs, key=lambda k: sc[k]["auroc"])
            pairs = {f"{ch}-{b}": C.paired(M.auroc, y, preds[ch][msk], preds[b][msk], grp)
                     for ch in ("gbm_challenger", "logistic") for b in incs + list(simple)}
            res[es] = {"auroc": {k: v["auroc"] for k, v in sc.items()}, "auroc_ci": {k: v["auroc_ci"] for k, v in sc.items()},
                       "strongest_incumbent": strongest_inc, "paired": pairs,
                       "per_seed_auroc": [C.fast_auroc(y, p[msk]) for p in g]}
            for name, s_ in sc.items():
                if re.search(r"_s\d+$", name):
                    continue
                C.log_metric(app=T.APP, experiment="mc_exceed8@it3", model=f"toxinwatch_{name}", metric="auroc",
                             value=s_["auroc"], split=es, n=s_["n"], ci=dict(zip(("ci_low", "ci_high"), s_["auroc_ci"])),
                             ci_method="cluster bootstrap over lake clusters, B=2000", target="microcystin>8ug/L",
                             is_baseline=name not in ("gbm_challenger", "logistic"), config=cfg,
                             data_manifests=list(T.DATA_MANIFESTS) + ["wqp_discrete"], extra={"n_pos": s_["n_pos"]})
            for key, pr in pairs.items():
                C.log_metric(app=T.APP, experiment="mc_exceed8@it3", model=f"toxinwatch_{key.replace('-', '_minus_')}",
                             metric="auroc_diff", value=pr["diff"], split=es, n=int(msk.sum()), ci=pr,
                             ci_method="paired cluster bootstrap (lake), B=2000", target="microcystin>8ug/L",
                             config=cfg, data_manifests=list(T.DATA_MANIFESTS) + ["wqp_discrete"],
                             extra={"p_value": pr["p_value"]})
            run.metric(f"{es}_gbm_auroc", sc["gbm_challenger"]["auroc"])
        C.write_json(out_dir / "val_results.json", res)
    for es in D.PRIMARY_SETS:
        r = res[es]
        print(es, {k: round(v, 3) for k, v in r["auroc"].items() if not re.search(r"_s\d+$", k)}, "strongest", r["strongest_incumbent"])
        for k, pr in r["paired"].items():
            print("  ", k, round(pr["diff"], 3), [round(pr["ci_low"], 3), round(pr["ci_high"], 3)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
