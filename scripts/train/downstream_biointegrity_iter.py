#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D5 BioIntegrity improvement iterations (validation only; protocol §4 tuning parity, §9 iteration log).

    python -B scripts/train/downstream_biointegrity_iter.py --iteration it1_mmi_regression
    python -B scripts/train/downstream_biointegrity_iter.py --iteration it2_tuned

* ``it1_mmi_regression`` — ONE change: the learners are trained on the continuous benthic MMI score of all
  training first visits (Fair included) instead of the Good/Poor class; Poor score = −predicted MMI.
  Parity: the RF family gets the same change (``rf_reg``); the incumbent stays exactly as published.
* ``it2_tuned`` — ONE change on top of phase 1 (it1 was rejected): hyperparameters chosen by inner
  leave-one-ecoregion-out CV on the training cycles (criterion = pooled Poor-vs-Good AUROC), the SAME
  budget (8 configurations) for every learned family, incumbent RF included.

Fitting happens inside ``common.cpu_slot``; bootstrap scoring runs after the slot is released.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads()

from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.models.downstream import biointegrity as BI  # noqa: E402
from sentinel.models.downstream.crossfit import phase1_no_upstream  # noqa: E402

APP = BI.APP
DATA = ["nars_streams", "downstream_catchment", "forcings_daily", "gridmet_sites"]
CLIM_YEARS = range(2000, 2015)
INCUMBENT = "hill2017_rf_streamcat"

GRIDS = {  # 8 configurations per family (equal budget); first entry = phase-1 default
    "rf": [{}, {"max_features": 0.2}, {"max_features": 0.33}, {"min_samples_leaf": 3},
           {"min_samples_leaf": 5, "max_features": 0.2}, {"max_features": "log2"},
           {"min_samples_leaf": 2, "max_features": 0.1}, {"max_features": 0.5, "min_samples_leaf": 3}],
    "gbm": [{}, {"num_leaves": 7}, {"num_leaves": 31, "min_child_samples": 40}, {"learning_rate": 0.015, "n_estimators": 800},
            {"colsample_bytree": 0.25}, {"colsample_bytree": 0.8, "reg_lambda": 5.0},
            {"num_leaves": 7, "learning_rate": 0.015, "n_estimators": 1000}, {"min_child_samples": 60, "subsample": 0.6}],
}


def specs(iteration: str):
    """(name, kind, feature set, is_baseline, tune)"""
    base = [(INCUMBENT, "rf", "streamcat", True, False),
            ("rf_geo_streamcat", "rf", "geo_streamcat", True, False),
            ("gbm_geo_streamcat", "gbm", "geo_streamcat", False, False)]
    if iteration == "it1_mmi_regression":
        return base + [("rf_reg_geo_streamcat", "rf_reg", "geo_streamcat", True, False),
                       ("rf_reg_geo_streamcat_nbr", "rf_reg", "geo_streamcat_nbr", True, False),
                       ("gbm_reg_geo_streamcat", "gbm_reg", "geo_streamcat", False, False),
                       ("gbm_reg_geo_streamcat_nbr", "gbm_reg", "geo_streamcat_nbr", False, False)]
    if iteration == "it2_tuned":
        # it1 (MMI regression) was rejected on validation, so it2 tunes the classification families only
        return base + [("tuned_hill2017_rf_streamcat", "rf", "streamcat", True, True),
                       ("tuned_rf_geo_streamcat", "rf", "geo_streamcat", True, True),
                       ("tuned_gbm_geo_streamcat", "gbm", "geo_streamcat", False, True)]
    raise ValueError(iteration)


def target_for(kind: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if kind in BI.REGRESSION_KINDS:
        f = frame[frame["bent_mmi"].notna()]
        return f, f["bent_mmi"].to_numpy(float)
    f = frame[frame["y_poor"].notna()]
    return f, f["y_poor"].to_numpy(float)


def inner_select(kind, X, tr, family, seed=0):
    """Inner LOEO CV on training cycles; criterion = pooled Poor-vs-Good AUROC (same for every family)."""
    scores = []
    for params in GRIDS[family]:
        p = pd.Series(np.nan, index=tr.index)
        for eco in BI.ECO9:
            te = tr["ag_eco9"] == eco
            if not te.any():
                continue
            fit_rows, y = target_for(kind, tr[~te])
            _, pp = BI.fit_predict(kind, X.loc[fit_rows.index], y, X.loc[tr.index[te]], seed=seed, params=params)
            p.loc[tr.index[te]] = pp
        m = tr["y_poor"].notna()
        scores.append(C.fast_auroc(tr.loc[m, "y_poor"].to_numpy(), p[m].to_numpy()))
        C.log_config_tried(APP, f"inner/{kind}", {"params": params, "inner_auroc": scores[-1]})
    k = int(np.nanargmax(scores))
    return GRIDS[family][k], scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iteration", required=True, choices=["it1_mmi_regression", "it2_tuned"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    units = BI.load_units()
    split, sha = C.load_split(APP, expect_sha=BI.SPLIT_SHA)
    if (units["visit_id"].map(split["assignments"]["visit_role"]) != units["role"]).any():
        raise C.ProtectedDataError("units no longer match the frozen split")
    u = C.drop_protected(units)
    C.assert_no_test(u)
    blocks, info = BI.build_features(u, CLIM_YEARS)
    tr, va = u[u["role"] == "train"], u[u["role"] == "val"]
    trb = tr[tr["y_poor"].notna()]
    blocks["nbr"] = BI.neighbour_block(u, trb)
    for b in blocks.values():
        phase1_no_upstream(b.columns)
    Xall = {fs: BI.assemble(blocks, names) for fs, names in BI.FEATURE_SETS.items()}
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    rd = C.run_dir(APP, f"{a.iteration}_{stamp}")
    cfg = {"app": APP, "iteration": a.iteration, "split_sha256": sha, "specs": specs(a.iteration),
           "grids": GRIDS if a.iteration == "it2_tuned" else None, "seeds": a.seeds,
           "bootstrap_unit": "huc4 (huc8 secondary)", "excluded_streamcat": list(BI.EXCLUDED_STREAMCAT)}
    preds, chosen = {}, {}
    with TrainingRun("downstream", f"{APP}/{a.iteration}", config=cfg, data=DATA, plan_ref=C.PLAN_REF,
                     iteration=a.iteration, artifacts_dir=str(rd)) as run:
        with C.cpu_slot(f"{APP}_{a.iteration}"):
            for name, kind, fs, _, tune in specs(a.iteration):
                X = Xall[fs]
                params = {}
                if tune:
                    family = "rf" if kind.startswith("rf") else "gbm"
                    params, sc = inner_select(kind, X, tr, family)
                    chosen[name] = {"params": params, "inner_auroc": sc}
                fit_rows, y = target_for(kind, tr)
                ps = []
                for seed in range(a.seeds):
                    _, pp = BI.fit_predict(kind, X.loc[fit_rows.index], y, X.loc[va.index], seed=seed, params=params)
                    ps.append(pp)
                    C.log_config_tried(APP, name, {"kind": kind, "features": fs, "seed": seed, "params": params,
                                                   "iteration": a.iteration})
                preds[name] = np.mean(ps, axis=0)
        # ---- scoring (outside the CPU slot)
        yv = va["y_poor"].to_numpy()
        units_g = {"huc4": va["huc4"].astype(str).fillna("NA").to_numpy(),
                   "huc8": va["huc8"].astype(str).fillna("NA").to_numpy()}
        subsets = {"newloc": va["new_location"].to_numpy(bool), "all": np.ones(len(va), bool)}
        res = {"val": {}, "paired": {}, "chosen": chosen}
        comps = [INCUMBENT, "rf_geo_streamcat"] + [n for n, *_ in specs(a.iteration) if n.startswith(("rf_reg", "tuned_rf", "tuned_hill"))]
        for sub, m in subsets.items():
            for name, s in preds.items():
                sc = C.binary_scores(yv[m], s[m], units_g["huc4"][m])
                res["val"][f"{sub}/{name}"] = sc
                if not a.no_register:
                    for met in ("auroc", "auprc"):
                        C.log_metric(app=APP, experiment=f"iter/{a.iteration}", model=name, metric=met, value=sc[met],
                                     split=f"val_nrsa1819_{sub}", n=sc["n"],
                                     ci={"ci_low": sc[f"{met}_ci"][0], "ci_high": sc[f"{met}_ci"][1]},
                                     ci_method="cluster bootstrap over HUC4, B=2000",
                                     target="bent_mmi_cond Poor vs Good",
                                     is_baseline=name in (INCUMBENT, "rf_geo_streamcat") or "_rf" in name or name.startswith("rf"),
                                     config=cfg, data_manifests=DATA, artifacts=[str(rd)],
                                     extra={"n_pos": sc["n_pos"], "split_sha256": sha, "chosen": chosen.get(name)})
            for name in preds:
                for comp in comps:
                    if name == comp:
                        continue
                    pr = C.paired(C.fast_auroc, yv[m], preds[name][m], preds[comp][m], units_g["huc4"][m])
                    pr8 = C.paired(C.fast_auroc, yv[m], preds[name][m], preds[comp][m], units_g["huc8"][m])
                    pr["ci_huc8"] = [pr8["ci_low"], pr8["ci_high"]]
                    res["paired"][f"{sub}/{name}_minus_{comp}"] = pr
                    if not a.no_register and not name.startswith(("rf", "hill", "tuned_rf", "tuned_hill")):
                        C.log_metric(app=APP, experiment=f"iter/{a.iteration}", model=name,
                                     metric=f"auroc_diff_vs_{comp}", value=pr["diff"], split=f"val_nrsa1819_{sub}",
                                     ci=pr, ci_method="paired cluster bootstrap over HUC4, B=2000",
                                     target="bent_mmi_cond Poor vs Good", config=cfg, data_manifests=DATA,
                                     artifacts=[str(rd)], extra={"p_value": pr["p_value"], "ci_huc8": pr["ci_huc8"]})
        pd.DataFrame({"visit_id": va["visit_id"].to_numpy(), "y_poor": yv, **preds}).to_parquet(
            rd / "val_predictions.parquet", index=False)
        C.write_json(rd / "results.json", res)
        run.note(f"configs tried so far {C.n_configs_tried(APP)}")
    for k, v in res["val"].items():
        print(k, round(v["auroc"], 4), [round(x, 4) for x in v["auroc_ci"]])
    for k, v in res["paired"].items():
        print(k, round(v["diff"], 4), [round(v["ci_low"], 4), round(v["ci_high"], 4)], v["ci_huc8"], round(v["p_value"], 3))
    print(json.dumps(chosen, default=str))
    print("run dir:", rd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
