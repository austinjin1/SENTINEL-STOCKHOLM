#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D6 MusselGuard single test look (FREEZE_CANDIDATE_downstream_musselguard.json; protocol §6, §10, §11, §12).

    python -B scripts/train/downstream_musselguard_test.py --dry-run-on-val      # origin 2015 as "test", fit train origins
    python -B scripts/train/downstream_musselguard_test.py --allow-test          # ONLY after FREEZE_downstream_musselguard.json exists
    python -B scripts/train/downstream_musselguard_test.py --rescore-from-saved  # crash recovery only (no re-fit)

Frozen: ``pressure_gbm`` (7 leaves, min_child 300, lr 0.03, 400 trees, 3 seeds) refit on train + val origins
(2006–2015). Incumbent = Kanankege re-implementation at the it2 selection (a = 320 km, k = 32, nugget 0.1;
§11b written rule). Baselines: distance-only, gravity α = 4, kernel smoother h = 20 km, location-only GBM,
pressure logistic (C = 1), SC rule (HUC8 median WQP specific conductance ≤ T; Whittier-2008 analogue).
**Each test origin (2018, 2021) is scored separately**; primary endpoint front-stratum (≤ 100 km) AUPRC,
paired HUC8 bootstrap, Holm across 7 baselines × 2 origins, both origins required, plus the seed rule.
CONUS / MN strata and pooled origins are secondary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads()

from sentinel.models.downstream import crossfit  # noqa: E402
from sentinel.models.downstream import musselguard as MG  # noqa: E402
from sentinel.models.downstream import testlook as TL  # noqa: E402

INCUMBENT = (320, 32, 0.1)            # it2 selection -> column "kanankege_r320"
INCUMBENT_COL = MG.kanankege_col(*INCUMBENT)
RULES = {"distance_only": None, "gravity": "gravity_a4", "kernel_smoother": "smooth_h20", "kanankege_reimpl": INCUMBENT_COL}
LEARNED = {"location_only_gbm": {"num_leaves": 7, "min_child_samples": 50, "learning_rate": 0.03, "n_estimators": 400},
           "pressure_logit": {"C": 1.0},
           "pressure_gbm": {"num_leaves": 7, "min_child_samples": 300, "learning_rate": 0.03, "n_estimators": 400}}
BASELINES = ["distance_only", "gravity", "kernel_smoother", "kanankege_reimpl", "location_only_gbm", "pressure_logit", "sc_rule"]
MODELS = list(RULES) + list(LEARNED) + ["sc_rule"]


def frame(origins: list[int]) -> pd.DataFrame:
    MG.USE_EXT_GRID = True
    MG.KRIGING_GRID_EXT = [INCUMBENT]           # the frozen incumbent configuration only
    uni, first, src = MG.load_universe(), MG.load_first_detection(), MG.load_records()
    uni = pd.concat([uni, MG.population_pressure(uni)], axis=1)
    fr = []
    for T in origins:
        f = MG.build_origin(T, uni, first, src)
        f["role"] = MG.ORIGINS[T]
        prox = MG.wqp_huc8_proxies(T)[["huc8", "huc8_spcond_median"]]
        f = f.assign(huc8=f["huc8"].astype(str)).merge(prox, on="huc8", how="left")
        fr.append(f)
    df = pd.concat(fr, ignore_index=True)
    crossfit.phase1_no_upstream(df.columns)
    return df


def predict(mode: str, roles: dict, sha: str) -> pd.DataFrame:
    fit_roles, ev_role = (("train", "val"), "test") if mode == "test" else (("train",), "val")
    origins = sorted(t for t, r in roles.items() if r in fit_roles + (ev_role,))
    if mode != "test":
        C.check_roles(MG.APP, {roles[t] for t in origins})
    df = frame(origins)
    if mode != "test":
        C.assert_no_test(df)
    tr = df[df["role"].isin(fit_roles)].reset_index(drop=True)
    ev = df[df["role"] == ev_role].reset_index(drop=True)
    p = ev[["huc12", "origin", "y", "front", "mn", "huc8"]].copy()
    for name, col in RULES.items():
        p[f"s_{name}"] = -ev["dist_nearest_km"].to_numpy(float) if col is None else ev[col].to_numpy(float)
    p["s_sc_rule"] = MG.sc_rule_score(ev)
    for name, params in LEARNED.items():
        kind, cols, _ = MG.MODEL_SPECS[name]
        Xtr, Xev = MG.design(tr, cols), MG.design(ev, cols)
        seeds = MG.SEEDS if kind == "gbm" else (0,)
        ps = [MG.fit_model(kind, params, Xtr, tr["y"].to_numpy(), seed=s).predict_proba(Xev)[:, 1] for s in seeds]
        p[f"s_{name}"] = np.mean(ps, axis=0)
        for i, q in enumerate(ps):
            p[f"s_{name}_seed{i}"] = q
    return p


def score(pred: pd.DataFrame, mode: str, sha: str) -> dict:
    tag = "test_origin" if mode != "dryrun" else "dryrun_val_origin"
    res = {"mode": mode, "split_sha256": sha, "incumbent": INCUMBENT_COL, "learned": LEARNED,
           "origins_eval": sorted(int(t) for t in pred["origin"].unique()), "sets": {}}
    groups_def = [(f"{tag}_{int(T)}", pred["origin"].to_numpy() == T) for T in sorted(pred["origin"].unique())]
    if len(groups_def) > 1:
        groups_def.append((f"{tag}s_pooled", np.ones(len(pred), bool)))
    for oname, om in groups_def:
        for stratum, col in (("front100km", "front"), ("conus", None), ("mn", "mn")):
            m = om & (np.ones(len(pred), bool) if col is None else pred[col].to_numpy(bool))
            s = pred[m]
            name = f"{oname}_{stratum}"
            is_primary = stratum == "front100km" and not oname.endswith("pooled")
            per = {"pressure_gbm": [s[f"s_pressure_gbm_seed{i}"].to_numpy() for i in range(len(MG.SEEDS))]}
            r = TL.score_set(app=MG.APP, mode=mode, split_sha=sha, set_name=name, experiment="test_look",
                             y=s["y"].to_numpy(float), scores={k: s[f"s_{k}"].to_numpy() for k in MODELS},
                             groups={"huc8": s["huc8"].to_numpy()},
                             primary={"pressure_gbm": BASELINES} if is_primary else {},
                             secondary={} if is_primary else {"pressure_gbm": BASELINES},
                             metric=C.fast_auprc, prob_models=tuple(LEARNED), manifests=MG.DATA_MANIFESTS + ["wqp_discrete"],
                             per_seed=per)
            r["pressure_gbm_per_seed_auprc"] = [C.fast_auprc(s["y"].to_numpy(float), q) for q in per["pressure_gbm"]]
            r["sc_rule_coverage"] = float(np.isfinite(s["s_sc_rule"]).mean())
            res["sets"][name] = r
    prim = {k: v for k, v in res["sets"].items() if k.endswith("front100km") and "pooled" not in k}
    res["holm_primary"] = TL.holm_primary(prim, "huc8")
    res.update(TL.claim_met(res["holm_primary"]))
    res["both_origins_required"] = True
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    TL.add_mode_args(ap)
    a = ap.parse_args(argv)
    mode = TL.mode_from_args(a)
    split, sha = C.load_split(MG.APP, expect_sha=MG.SPLIT_SHA)
    roles = {int(t): r for t, r in split["assignments"]["origin_role"].items()}
    if roles != MG.ORIGINS:
        raise C.ProtectedDataError("origin roles differ from the frozen split")
    TL.verify_data(MG.APP)
    claim = None
    with C.cpu_slot(f"musselguard_{mode}"):
        if mode == "test":
            claim = TL.authorise(MG.APP, sha)
            pred = predict(mode, roles, sha)
            TL.save_predictions(MG.APP, sha, pred)
        elif mode == "rescore":
            pred = TL.authorise_rescore(MG.APP, sha)
        else:
            pred = predict(mode, roles, sha)
        res = score(pred, mode, sha)
        res["claim"] = claim
        rd = TL.look_dir(MG.APP) if mode != "dryrun" else C.run_dir(MG.APP, "testpath_dryrun")
        if mode == "dryrun":
            pred.to_parquet(rd / "predictions.parquet", index=False)
        C.write_json(rd / ("results.json" if mode != "rescore" else "results_rescore.json"), res)
        if mode != "dryrun":
            TL.mark_scored(MG.APP, {"claim_met": res["claim_met"]})
    for k, v in res["sets"].items():
        if k.endswith("front100km"):
            print(k, json.dumps({m: round(x["auprc"], 4) for m, x in v["scores"].items() if x.get("auprc") == x.get("auprc")}))
    print(json.dumps({k: (round(v["diff"], 4), v["holm_reject"], v["per_seed_diff"]) for k, v in res["holm_primary"].items()}),
          "claim_met:", res["claim_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
