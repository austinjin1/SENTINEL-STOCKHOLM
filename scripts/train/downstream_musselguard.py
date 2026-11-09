#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D6 MusselGuard driver (docs/preregistration/downstream_musselguard.md).

    python -B scripts/train/downstream_musselguard.py --freeze-split
    python -B scripts/train/downstream_musselguard.py --build-only [--wqp]
    python -B scripts/train/downstream_musselguard.py            # phase 1 (after review): fit train origins, score val origin

Test origins (2018, 2021) are never built in phase 1: roles are checked with common.check_roles before
any origin frame is constructed, and rows are filtered with common.drop_protected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256  # noqa: E402
from sentinel.models.downstream import common as C  # noqa: E402
from sentinel.models.downstream import crossfit  # noqa: E402
from sentinel.models.downstream import musselguard as MG  # noqa: E402

C.limit_threads()


def freeze() -> str:
    uni = paths.PROCESSED / "wbd_huc12" / "huc12.parquet"
    meta = {"unit": "CONUS WBD HUC12 (processed/wbd_huc12)", "universe_sha256": file_sha256(uni),
            "label": f"first NAS record (group {MG.GROUP}, first_year_any) in (T, T+{MG.HORIZON}]",
            "horizon_years": MG.HORIZON, "front_km": MG.FRONT_KM,
            "spatial": "leave-one-HUC2-out (01-18) over train origins",
            "label_vintage": "labels_nas_dreissena retrieved 2026-10-22"}
    return C.freeze_split(MG.APP, {"origin_role": {str(k): v for k, v in MG.ORIGINS.items()},
                                   "spatial_fold_huc2": {h: i for i, h in enumerate(MG.CONUS_HUC2)}}, meta)


ITER = "base"


def build(split: dict, *, use_wqp: bool) -> pd.DataFrame:
    roles = split["assignments"]["origin_role"]
    origins = [int(t) for t, r in roles.items() if r in ("train", "val")]
    C.check_roles(MG.APP, {roles[str(t)] for t in origins})
    uni, first, src = MG.load_universe(), MG.load_first_detection(), MG.load_records()
    if ITER in ("it2_kanankege", "it2b_kanankege_grid"):
        uni = pd.concat([uni, MG.population_pressure(uni)], axis=1)
    frames = []
    for T in sorted(origins):
        wqp = MG.wqp_huc8_proxies(T) if use_wqp else None
        f = MG.build_origin(T, uni, first, src, wqp=wqp)
        f["role"] = roles[str(T)]
        frames.append(f)
    df = C.drop_protected(pd.concat(frames, ignore_index=True))
    C.assert_no_test(df)
    crossfit.phase1_no_upstream(df.columns)
    return df


def counts(df: pd.DataFrame) -> dict:
    out = {}
    for T, sub in df.groupby("origin"):
        out[str(T)] = {"role": sub["role"].iloc[0], "candidates": int(len(sub)), "positives": int(sub["y"].sum()),
                       "infested_at_T": int(sub["n_infested_at_T"].iloc[0]),
                       "front_candidates": int(sub["front"].sum()), "front_positives": int(sub.loc[sub["front"], "y"].sum()),
                       "mn_candidates": int(sub["mn"].sum()), "mn_positives": int(sub.loc[sub["mn"], "y"].sum()),
                       "positives_beyond_front": int(sub.loc[~sub["front"], "y"].sum()),
                       "missing_frac": sub[[c for c in MG.PRESSURE_FEATURES + MG.WQP_FEATURES if c in sub]].isna().mean().round(3).to_dict()}
    return out


VAL_SPLIT = "val_origin2015_2016_2018"
STRATA = {"front100km": "front", "conus": None, "mn": "mn"}


def _seed_avg(kind, params, Xtr, ytr, Xva):
    import numpy as np

    seeds = MG.SEEDS if kind == "gbm" else (0,)
    per = [MG.fit_model(kind, params, Xtr, ytr, seed=sd).predict_proba(Xva)[:, 1] for sd in seeds]
    return np.mean(per, axis=0), per


def phase1(df: pd.DataFrame, sha: str) -> dict:
    import numpy as np

    from sentinel.core.runlog import TrainingRun
    from sentinel.eval.bootstrap import holm_bonferroni
    from sentinel.eval.metrics import auprc, auroc

    tr, va = df[df["role"] == "train"].reset_index(drop=True), df[df["role"] == "val"].reset_index(drop=True)
    C.assert_no_test(tr, va)
    out_dir = C.run_dir(MG.APP, "phase1" if ITER == "base" else f"phase1_{ITER}")
    res = {"split_sha256": sha, "models": {}, "n_train": int(len(tr)), "n_val": int(len(va))}
    with TrainingRun(C.WORKSTREAM, f"musselguard/phase1/{ITER}", config={"split_sha256": sha, "iteration": ITER}, iteration=ITER,
                                                       data=MG.DATA_MANIFESTS, plan_ref=C.PLAN_REF,
                                                       artifacts_dir=str(out_dir)) as run:
        scores = {"distance_only": -va["dist_nearest_km"].to_numpy(float)}
        baselines = ["distance_only"]
        rules = [("gravity", [c for c in MG.RULE_FEATURES if c.startswith("gravity")]),
                 ("kernel_smoother", [c for c in MG.RULE_FEATURES if c.startswith("smooth")])]
        if ITER in ("it2_kanankege", "it2b_kanankege_grid"):
            rules.append(("kanankege_reimpl", MG.incumbent_features()))
        for rule, cols in rules:
            col, log = MG.select_rule(tr, cols)
            for r in log:
                C.log_config_tried(MG.APP, f"{rule}@{ITER}", r["params"], status=f"train_front_auprc={r['train_front_auprc']:.4f}")
            scores[rule] = va[col].to_numpy(float)
            res["models"][rule] = {"selected": col, "selection": log}
            baselines.append(rule)
        if "huc8_spcond_median" in va:
            scores["sc_rule"] = MG.sc_rule_score(va)
            baselines.append("sc_rule")
        per_seed = {}
        specs = dict(MG.MODEL_SPECS)
        if ITER in ("it2_kanankege", "it2b_kanankege_grid"):
            specs["pressure_pop_gbm"] = ("gbm", MG.PRESSURE_FEATURES + MG.RULE_FEATURES + MG.LOCATION_FEATURES
                                         + MG.POP_FEATURES, None)
        for name, (kind, cols, grid) in specs.items():
            grid = grid or MG.GBM_GRID
            Xtr, Xva = MG.design(tr, cols), MG.design(va, cols)
            best, log = MG.inner_cv_select(kind, grid, Xtr, tr["y"].to_numpy(), tr["huc2"].to_numpy(),
                                           tr["front"].to_numpy(), seeds=MG.SEEDS)
            for r in log:
                C.log_config_tried(MG.APP, f"{name}@{ITER}", r["params"], status=f"inner_cv_front_auprc={r['inner_cv_front_auprc']:.4f}")
            scores[name], per = _seed_avg(kind, best, Xtr, tr["y"].to_numpy(), Xva)
            per_seed[name] = [MG.front_auprc(va["y"], p, va["front"]) for p in per]
            res["models"][name] = {"selected": best, "inner_cv": log, "val_front_auprc_per_seed": per_seed[name]}
            if name not in ("pressure_gbm", "pressure_pop_gbm"):
                baselines.append(name)
        for name, sc in scores.items():
            res["models"].setdefault(name, {})
            for stratum, col in STRATA.items():
                mask = np.ones(len(va), bool) if col is None else va[col].to_numpy(bool)
                ok = np.isfinite(sc[mask])
                bs = C.binary_scores(va["y"].to_numpy()[mask][ok], sc[mask][ok], va["huc8"].to_numpy()[mask][ok])
                res["models"][name][stratum] = bs
                for met in ("auroc", "auprc"):
                    if np.isfinite(bs.get(met, np.nan)):
                        C.log_metric(app=MG.APP, experiment=f"rolling_origin/{stratum}" + ("" if ITER == "base" else f"@{ITER}"),
                                     model=name, metric=met,
                                     value=bs[met], split=VAL_SPLIT, n=bs["n"],
                                     ci={"ci_low": bs[f"{met}_ci"][0], "ci_high": bs[f"{met}_ci"][1]},
                                     ci_method="cluster bootstrap over HUC8, B=2000", is_baseline=name in baselines,
                                     config={"selected": res["models"][name].get("selected")},
                                     data_manifests=MG.DATA_MANIFESTS,
                                     extra={"split_sha256": sha, "prevalence": bs["prevalence"],
                                            "per_seed_front_auprc": per_seed.get(name)})
            run.metric(f"{name}/front_auprc", res["models"][name]["front100km"].get("auprc", float("nan")))
        front = va["front"].to_numpy(bool)
        for chal in [c for c in ("pressure_gbm", "pressure_pop_gbm") if c in scores]:
            pv, paired = {}, {}
            for base in baselines:
                for met, f in (("auprc", auprc), ("auroc", auroc)):
                    ok = front & np.isfinite(scores[base])
                    r = C.paired(f, va["y"].to_numpy()[ok], scores[chal][ok], scores[base][ok], va["huc8"].to_numpy()[ok])
                    paired[f"{met}_vs_{base}"] = r
                    if met == "auprc":
                        pv[base] = r["p_value"]
            holm = holm_bonferroni(pv)
            for base in baselines:
                paired[f"auprc_vs_{base}"]["holm_reject"] = holm[base]
            res[f"paired_{chal}_front"] = paired
        # spatial secondary: leave-one-HUC2-out over train origins (challenger + distance + gravity)
        loho = {"pressure_gbm": [], "distance_only": [], "gravity": []}
        kind, cols, _ = MG.MODEL_SPECS["pressure_gbm"]
        X = MG.design(tr, cols)
        pred = np.full(len(tr), np.nan)
        for h2 in sorted(tr["huc2"].unique()):
            te = (tr["huc2"] == h2).to_numpy()
            if tr.loc[te, "y"].sum() == 0:
                continue
            pred[te] = _seed_avg(kind, res["models"]["pressure_gbm"]["selected"], X[~te], tr.loc[~te, "y"].to_numpy(), X[te])[0]
        ok = np.isfinite(pred)
        loho_res = {"pressure_gbm": MG.front_auprc(tr["y"][ok], pred[ok], tr["front"][ok]),
                    "distance_only": MG.front_auprc(tr["y"][ok], -tr["dist_nearest_km"][ok], tr["front"][ok]),
                    "gravity": MG.front_auprc(tr["y"][ok], tr[res["models"]["gravity"]["selected"]][ok], tr["front"][ok])}
        res["leave_one_huc2_out_train_origins_front_auprc"] = loho_res
        for name, v in loho_res.items():
            if np.isfinite(v):
                C.log_metric(app=MG.APP, experiment="loho_huc2/front100km" + ("" if ITER == "base" else f"@{ITER}"), model=name, metric="auprc", value=v,
                             split="dev_train_origins_leave_one_huc2_out", n=int(ok.sum()), is_baseline=name != "pressure_gbm",
                             data_manifests=MG.DATA_MANIFESTS, notes="no CI (pooled CV predictions; secondary)",
                             extra={"split_sha256": sha})
        C.write_json(out_dir / "phase1_results.json", res)
        if ITER != "base":      # val predictions for later paired re-analysis (no test rows exist here)
            vp = va[["huc12", "origin", "y", "front", "mn", "huc8"]].copy()
            for name, sc in scores.items():
                vp[f"score_{name}"] = sc
            vp.to_parquet(out_dir / "val_predictions.parquet", index=False)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze-split", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--phase1", action="store_true", help="fit train origins / score val origin (after review only)")
    ap.add_argument("--wqp", action="store_true", help="add WQP SC/pH HUC8 proxies (needs the rebuilt wqp_discrete)")
    ap.add_argument("--iteration", default="base", choices=["base", "it2_kanankege", "it2b_kanankege_grid"])
    a = ap.parse_args()
    global ITER
    ITER = a.iteration
    MG.USE_EXT_GRID = ITER == "it2b_kanankege_grid"
    if a.freeze_split:
        print("frozen split sha256:", freeze())
        return 0
    split, sha = C.load_split(MG.APP, expect_sha=MG.SPLIT_SHA)
    with C.cpu_slot("musselguard_phase1" if a.phase1 else "musselguard_build"):   # one heavy job per workstream
        df = build(split, use_wqp=a.wqp)
        out_dir = C.run_dir(MG.APP, "build" if ITER == "base" else f"build_{ITER}")
        df.to_parquet(out_dir / "origins_trainval.parquet", index=False)
        rep = {"split_sha256": sha, "wqp": a.wqp, "counts": counts(df)}
        C.write_json(out_dir / "build_counts.json", rep)
        print(json.dumps(rep, indent=1, default=str), flush=True)
        if not a.phase1:
            return 0
        res = phase1(df, sha)
        print(json.dumps({m: {k: v.get("auprc") for k, v in r.items() if isinstance(v, dict) and "auprc" in v}
                          for m, r in res["models"].items()}, indent=1, default=str), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
