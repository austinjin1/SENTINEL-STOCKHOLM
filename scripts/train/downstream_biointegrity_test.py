#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D5 BioIntegrity single test look (FREEZE_CANDIDATE_downstream_biointegrity.json; protocol §1, §5, §6, §10).

    python -B scripts/train/downstream_biointegrity_test.py --dry-run-on-val      # NRSA 2018-19 as "test"; fit 2008-09 + 2013-14
    python -B scripts/train/downstream_biointegrity_test.py --allow-test          # ONLY after FREEZE_downstream_biointegrity.json exists
    python -B scripts/train/downstream_biointegrity_test.py --rescore-from-saved  # crash recovery only (no re-fit)

Frozen challenger ``ens_tuned_gbm_rf`` = 0.5 × (tuned GBM + tuned RF on geo+StreamCat), it2 hyperparameters,
3 seeds per member (seed s of the ensemble = mean of the two members' seed-s predictions). Test model
re-fitted on NRSA 2008-09 + 2013-14 + 2018-19 (§6); test new location = location cluster with no first visit
in any fitting cycle (§6). **Primary family (§8 W8, it3 rule): all four RF rows** (Hill-2017 incumbent
published + tuned, feature-matched RF default + tuned), paired AUROC over HUC4 (HUC8 secondary), Holm over
four, plus the seed rule. Secondary (reported): accuracy at p = 0.5, all first visits, revisited locations
with persistence, the frozen val_test_new_location subset, logistic on geo, ecoregion / location-kNN rates.
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

from sentinel.models.downstream import biointegrity as BI  # noqa: E402
from sentinel.models.downstream import testlook as TL  # noqa: E402

CLIM_YEARS = range(2000, 2015)
DATA = ["nars_streams", "downstream_catchment", "forcings_daily", "gridmet_sites"]
SEEDS = (0, 1, 2)
MODELS = {  # name: (kind, feature set, params) — it2 choices (results.json "chosen")
    "hill2017_rf_streamcat": ("rf", "streamcat", {}),
    "tuned_hill2017_rf_streamcat": ("rf", "streamcat", {"max_features": 0.5, "min_samples_leaf": 3}),
    "rf_geo_streamcat": ("rf", "geo_streamcat", {}),
    "tuned_rf_geo_streamcat": ("rf", "geo_streamcat", {"min_samples_leaf": 2, "max_features": 0.1}),
    "tuned_gbm_geo_streamcat": ("gbm", "geo_streamcat", {"min_child_samples": 60, "subsample": 0.6}),
    "logistic_geo": ("logistic", "geo", {}),
}
RF4 = ["hill2017_rf_streamcat", "tuned_hill2017_rf_streamcat", "rf_geo_streamcat", "tuned_rf_geo_streamcat"]
PRIMARY = {"ens_tuned_gbm_rf": RF4}
SECONDARY = {"ens_tuned_gbm_rf": ["tuned_gbm_geo_streamcat", "logistic_geo", "ecoregion_rate", "location_knn"]}
ALL = list(MODELS) + ["ens_tuned_gbm_rf", "ecoregion_rate", "location_knn"]


def predict(mode: str, split: dict, sha: str) -> pd.DataFrame:
    units = BI.load_units()
    if (units["visit_id"].map(split["assignments"]["visit_role"]) != units["role"]).any():
        raise C.ProtectedDataError("units no longer match the frozen split")
    if mode == "test":
        u = units[units["role"].isin(["train", "val", "test"])].reset_index(drop=True)
        fit_roles, ev_role = ("train", "val"), "test"
    else:
        u = C.drop_protected(units).reset_index(drop=True)
        C.assert_no_test(u)
        fit_roles, ev_role = ("train",), "val"
    fit_all = u[u["role"].isin(fit_roles)]
    fit = fit_all[fit_all["y_poor"].notna()]
    fit_locs = set(fit_all["location_id"])
    ev = u[u["role"] == ev_role]                         # every unit of the evaluated cycle (Fair included)
    blocks, _ = BI.build_features(u, CLIM_YEARS)
    Xall = {fs: BI.assemble(blocks, BI.FEATURE_SETS[fs]) for fs in {v[1] for v in MODELS.values()}}
    p = ev[["visit_id", "y_poor", "y_ord", "huc4", "huc8", "location_id"]].copy()
    p["is_new"] = ~ev["location_id"].isin(fit_locs).to_numpy()
    p["is_frozen_new"] = ev["visit_id"].map(split["assignments"]["val_test_new_location"]).fillna(False).astype(bool).to_numpy()
    for name, (kind, fs, params) in MODELS.items():
        X = Xall[fs]
        seeds = SEEDS if kind in ("rf", "gbm") else (0,)
        ps = [BI.fit_predict(kind, X.loc[fit.index], fit["y_poor"].to_numpy(float), X.loc[ev.index], seed=s, params=params)[1]
              for s in seeds]
        p[f"s_{name}"] = np.mean(ps, axis=0)
        for i, q in enumerate(ps):
            p[f"s_{name}_seed{i}"] = q
    p["s_ens_tuned_gbm_rf"] = 0.5 * (p["s_tuned_gbm_geo_streamcat"] + p["s_tuned_rf_geo_streamcat"])
    for i in range(len(SEEDS)):
        p[f"s_ens_tuned_gbm_rf_seed{i}"] = 0.5 * (p[f"s_tuned_gbm_geo_streamcat_seed{i}"] + p[f"s_tuned_rf_geo_streamcat_seed{i}"])
    p["s_ecoregion_rate"] = BI.ecoregion_rate(fit, ev)
    p["s_location_knn"] = BI.knn_rate(fit, ev)
    p["s_persistence"] = BI.persistence(fit_all, ev)
    return p.reset_index(drop=True)


def score(pred: pd.DataFrame, mode: str, sha: str) -> dict:
    pref = "test_nrsa2324" if mode != "dryrun" else "dryrun_val_nrsa1819"
    lab = pred[pred["y_poor"].notna()]
    sets = {f"{pref}_newloc": lab["is_new"].to_numpy(bool),                 # primary
            f"{pref}_all_first_visits": np.ones(len(lab), bool),
            f"{pref}_frozen_newloc_flag": lab["is_frozen_new"].to_numpy(bool),
            f"{pref}_revisited_locations": ~lab["is_new"].to_numpy(bool)}
    res = {"mode": mode, "split_sha256": sha, "models": {k: {"kind": v[0], "features": v[1], "params": v[2]} for k, v in MODELS.items()},
           "n_units_predicted": int(len(pred)), "sets": {}}
    for name, m in sets.items():
        s = lab[m]
        y = s["y_poor"].to_numpy(float)
        models = ALL + (["persistence"] if name.endswith("revisited_locations") else [])
        sc = {k: s[f"s_{k}"].to_numpy() for k in models}
        primary = PRIMARY if name.endswith("_newloc") else {}
        secondary = SECONDARY if primary else {"ens_tuned_gbm_rf": RF4 + SECONDARY["ens_tuned_gbm_rf"]
                                               + (["persistence"] if "persistence" in sc else [])}
        per = {"ens_tuned_gbm_rf": [s[f"s_ens_tuned_gbm_rf_seed{i}"].to_numpy() for i in range(len(SEEDS))]}
        r = TL.score_set(app=BI.APP, mode=mode, split_sha=sha, set_name=name, experiment="test_look", y=y, scores=sc,
                         groups={"huc4": s["huc4"].astype(str).fillna("NA").to_numpy(), "huc8": s["huc8"].astype(str).fillna("NA").to_numpy()},
                         primary=primary, secondary=secondary, prob_models=tuple(MODELS) + ("ens_tuned_gbm_rf",),
                         manifests=DATA, per_seed=per)
        r["accuracy_at_0.5"] = {k: C.accuracy_at(y, v) for k, v in sc.items() if k not in ("persistence",)}
        r["per_seed_auroc"] = {k: [C.fast_auroc(y, s[f"s_{k}_seed{i}"].to_numpy()) for i in range(len(SEEDS))]
                               for k in ["ens_tuned_gbm_rf"] + [k for k, v in MODELS.items() if v[0] in ("rf", "gbm")]}
        res["sets"][name] = r
    res["holm_primary"] = TL.holm_primary({k: v for k, v in res["sets"].items() if k.endswith("_newloc")}, "huc4")
    res.update(TL.claim_met(res["holm_primary"]))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    TL.add_mode_args(ap)
    a = ap.parse_args(argv)
    mode = TL.mode_from_args(a)
    split, sha = C.load_split(BI.APP, expect_sha=BI.SPLIT_SHA)
    TL.verify_data(BI.APP)
    claim = None
    with C.cpu_slot(f"biointegrity_{mode}"):
        if mode == "test":
            claim = TL.authorise(BI.APP, sha)
            pred = predict(mode, split, sha)
            TL.save_predictions(BI.APP, sha, pred)
        elif mode == "rescore":
            pred = TL.authorise_rescore(BI.APP, sha)
        else:
            pred = predict(mode, split, sha)
        res = score(pred, mode, sha)
        res["claim"] = claim
        rd = TL.look_dir(BI.APP) if mode != "dryrun" else C.run_dir(BI.APP, "testpath_dryrun")
        if mode == "dryrun":
            pred.to_parquet(rd / "predictions.parquet", index=False)
        C.write_json(rd / ("results.json" if mode != "rescore" else "results_rescore.json"), res)
        if mode != "dryrun":
            TL.mark_scored(BI.APP, {"claim_met": res["claim_met"]})
    r = [v for k, v in res["sets"].items() if k.endswith("_newloc")][0]
    print(json.dumps({k: r["scores"][k]["auroc"] for k in ALL}, indent=1))
    print(json.dumps({k: (round(v["diff"], 4), v["holm_reject"], v["per_seed_diff"]) for k, v in res["holm_primary"].items()}),
          "claim_met:", res["claim_met"], "seed_rule_ok:", res["seed_rule_ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
