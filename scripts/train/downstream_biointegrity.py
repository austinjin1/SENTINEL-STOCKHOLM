#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D5 BioIntegrity driver (protocol: docs/preregistration/downstream_biointegrity.md).

    python -B scripts/train/downstream_biointegrity.py --freeze-split     # once, before any model fit
    python -B scripts/train/downstream_biointegrity.py --build-only       # features + counts (no fits)
    python -B scripts/train/downstream_biointegrity.py                     # dev CV + validation (2018-19)

Test-role visits (NRSA 2023-24) are dropped right after role assignment; nothing here scores them.
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
from sentinel.eval import metrics as M  # noqa: E402
from sentinel.models.downstream import biointegrity as BI  # noqa: E402
from sentinel.models.downstream.crossfit import phase1_no_upstream  # noqa: E402

APP = BI.APP
DATA = ["nars_streams", "downstream_catchment", "forcings_daily", "gridmet_sites"]
CLIM_YEARS = range(2000, 2015)   # antecedent-weather climatology: years before the validation cycle
INCUMBENT = "hill2017_rf_streamcat"


def assign_roles(units: pd.DataFrame, expect_sha: str | None) -> tuple[pd.DataFrame, str]:
    split, sha = C.load_split(APP, expect_sha=expect_sha)
    role = split["assignments"]["visit_role"]
    newloc = split["assignments"]["val_test_new_location"]
    u = units.copy()
    u["role_frozen"] = u["visit_id"].map(role)
    if (u["role_frozen"] != u["role"]).any() or u["role_frozen"].isna().any():
        raise C.ProtectedDataError("units no longer match the frozen split (rebuild changed the unit set)")
    u["new_location_frozen"] = u["visit_id"].map(newloc)
    return u.drop(columns=["role_frozen"]), sha


def models_spec():
    """(name, kind, feature set, is_baseline). Fixed a priori; every row is reported."""
    return [
        ("logistic_geo", "logistic", "geo", True),
        (INCUMBENT, "rf", "streamcat", True),
        ("rf_geo_streamcat", "rf", "geo_streamcat", True),
        ("gbm_geo_streamcat", "gbm", "geo_streamcat", False),
        ("gbm_geo_streamcat_wx", "gbm", "geo_streamcat_wx", False),
        ("rf_geo_streamcat_nbr", "rf", "geo_streamcat_nbr", True),
        ("gbm_geo_streamcat_nbr", "gbm", "geo_streamcat_nbr", False),
        ("gbm_geo_streamcat_wx_nbr", "gbm", "geo_streamcat_wx_nbr", False),
        ("oracle_gbm_chem", "gbm", "oracle_chem", False),
        ("oracle_gbm_chem_habitat", "gbm", "oracle_chem_habitat", False),
    ]


def score_block(name, y, s, groups, is_prob=True):
    out = C.binary_scores(y, s, groups, prob=is_prob)
    m = np.isfinite(y) & np.isfinite(s)
    out["accuracy"] = C.accuracy_at(y[m], s[m]) if is_prob else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze-split", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--split-sha", default=BI.SPLIT_SHA, help="expected split sha256 (protocol; enforced)")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    units = BI.load_units()
    if a.freeze_split:
        asg, meta = BI.split_assignments(units)
        sha = C.freeze_split(APP, asg, meta)
        rep = {"sha256": sha, "n_units": int(len(units)),
               "by_role": units.groupby("role")["y_poor"].agg(["size", "count", "sum"]).to_dict(),
               "by_role_class": pd.crosstab(units["role"], units[BI.TARGET]).to_dict(),
               "new_location_by_role": pd.crosstab(units["role"], units["new_location"]).to_dict()}
        C.write_json(C.PREREG / f"downstream_{APP}_split_report.json", rep)
        print(json.dumps(rep, indent=1, default=str))
        return 0

    units, sha = assign_roles(units, a.split_sha)
    u = C.drop_protected(units)          # NRSA 2023-24 (test) never enters model code
    C.assert_no_test(u)
    blocks, info = BI.build_features(u, CLIM_YEARS)
    for b in blocks.values():
        phase1_no_upstream(b.columns)
    tr, va = u[u["role"] == "train"], u[u["role"] == "val"]
    if a.build_only:
        print(json.dumps({"split_sha256": sha, "n_dropped_test": u.attrs.get("n_dropped_protected"),
                          "train": pd.crosstab(tr["survey"], tr[BI.TARGET]).to_dict(),
                          "val": pd.crosstab(va["new_location"], va[BI.TARGET]).to_dict(),
                          "feature_info": info,
                          "missing_frac": {k: float(v.isna().all(axis=1).mean()) for k, v in blocks.items()},
                          "n_cols": {k: int(v.shape[1]) for k, v in blocks.items()}}, indent=1, default=str))
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    rd = C.run_dir(APP, f"phase1_{stamp}")
    cfg = {"app": APP, "split_sha256": sha, "clim_years": [min(CLIM_YEARS), max(CLIM_YEARS)],
           "models": models_spec(), "seeds": a.seeds, "bootstrap_unit": "huc4 (huc8 secondary)",
           "excluded_streamcat": list(BI.EXCLUDED_STREAMCAT), "n_boot": C.N_BOOT}
    reg = not a.no_register
    results = {"split_sha256": sha, "feature_info": info, "dev": {}, "val": {}, "paired": {}}
    with C.cpu_slot(f"{APP}_phase1"), TrainingRun("downstream", f"{APP}/phase1", config=cfg, data=DATA,
                                                 plan_ref=C.PLAN_REF, artifacts_dir=str(rd)) as run:
        trb = tr[tr["y_poor"].notna()]
        blocks["nbr"] = BI.neighbour_block(u, trb)       # spatial context from training-cycle labels only
        Xall = {fs: BI.assemble(blocks, names) for fs, names in BI.FEATURE_SETS.items()}

        # ---- incumbent reproduction under its own (random/OOB) design: Hill 2017 used NRSA 2008-09
        t08 = trb[trb["survey"] == "nrsa0809"]
        rf = BI.make_model("rf", seed=0)
        X08 = Xall["streamcat"].loc[t08.index].astype(float)
        X08 = X08.fillna(X08.median()).fillna(0.0)
        rf.fit(X08, t08["y_poor"].to_numpy())
        oob = rf.oob_decision_function_[:, 1]
        y08 = t08["y_poor"].to_numpy()
        results["dev"]["hill2017_oob_nrsa0809"] = {"oob_accuracy": float(np.mean((oob >= 0.5) == (y08 == 1))),
                                                    "oob_auroc": float(M.auroc(y08, oob)), "n": int(len(t08)),
                                                    "published_accuracy": 0.75, "published_oob_auc_per_04": 0.84}
        run.metric("hill_oob_acc_0809", results["dev"]["hill2017_oob_nrsa0809"]["oob_accuracy"])

        # ---- development CV: leave-one-ecoregion-out on the training cycles (spatial transfer)
        for name, kind, fs, _ in models_spec():
            if name.startswith("oracle") or "_wx" in name or "_nbr" in name:
                continue
            p = np.full(len(trb), np.nan)
            for eco in BI.ECO9:
                te = (trb["ag_eco9"] == eco).to_numpy()
                if te.sum() == 0:
                    continue
                X = Xall[fs].loc[trb.index]
                _, pp = BI.fit_predict(kind, X[~te], trb["y_poor"].to_numpy()[~te], X[te], seed=0)
                p[te] = pp
            sc = score_block(name, trb["y_poor"].to_numpy(), p, trb["huc4"].astype(str).to_numpy())
            results["dev"][f"loeo_{name}"] = sc

        # ---- validation: train on 2008-09 + 2013-14, score 2018-19 first visits
        yv = va["y_poor"].to_numpy()
        gv = va["huc8"].astype(str).fillna("NA").to_numpy()
        subsets = {"all": np.ones(len(va), bool), "newloc": va["new_location"].to_numpy(bool),
                   "revisit": ~va["new_location"].to_numpy(bool)}
        preds = {"prevalence": np.full(len(va), trb["y_poor"].mean()),
                 "ecoregion_rate": BI.ecoregion_rate(trb, va),
                 "location_knn15": BI.knn_rate(trb, va, k=15)}
        pers = BI.persistence(tr, va)                       # ordinal class at the same location
        preds["persistence_prev_cycle"] = np.where(np.isnan(pers), np.nan, pers / 2.0)
        for name, kind, fs, _ in models_spec():
            ps = []
            for seed in range(a.seeds if kind != "logistic" else 1):
                _, pp = BI.fit_predict(kind, Xall[fs].loc[trb.index], trb["y_poor"].to_numpy(), Xall[fs].loc[va.index],
                                       seed=seed)
                ps.append(pp)
                C.log_config_tried(APP, name, {"kind": kind, "features": fs, "seed": seed})
            preds[name] = np.mean(ps, axis=0)
            results["val"].setdefault("seed_auroc", {})[name] = [
                float(M.auroc(yv[np.isfinite(yv)], q[np.isfinite(yv)])) for q in ps]
        pd.DataFrame({"visit_id": va["visit_id"].to_numpy(), "y_poor": yv, **preds}).to_parquet(
            rd / "val_predictions.parquet", index=False)

        base_names = {"prevalence", "ecoregion_rate", "location_knn15", "persistence_prev_cycle", "logistic_geo",
                      INCUMBENT, "rf_geo_streamcat", "rf_geo_streamcat_nbr"}
        # resampling units (review W6): HUC4 is the claim unit; HUC8 (pre-registered v1.0) reported too
        units = {"huc4": va["huc4"].astype(str).fillna("NA").to_numpy(), "huc8": gv}
        for sub, msk in subsets.items():
            for name, s in preds.items():
                if name == "persistence_prev_cycle" and sub != "revisit":
                    continue                       # defined on revisited locations only (review W1)
                sc = score_block(name, yv[msk], s[msk], units["huc4"][msk])
                sc8 = score_block(name, yv[msk], s[msk], units["huc8"][msk])
                sc["auroc_ci_huc8"] = sc8.get("auroc_ci")
                results["val"][f"{sub}/{name}"] = sc
                if reg and np.isfinite(sc.get("auroc", np.nan)):
                    for met in ("auroc", "auprc", "brier", "accuracy"):
                        if met in sc and np.isfinite(sc[met]):
                            C.log_metric(app=APP, experiment="phase1_val", model=name, metric=met, value=sc[met],
                                         split=f"val_nrsa1819_{sub}", n=sc["n"],
                                         ci=None if f"{met}_ci" not in sc else
                                         {"ci_low": sc[f"{met}_ci"][0], "ci_high": sc[f"{met}_ci"][1]},
                                         ci_method="cluster bootstrap over HUC4, B=2000" if f"{met}_ci" in sc else None,
                                         target="bent_mmi_cond Poor vs Good", is_baseline=name in base_names,
                                         config=cfg, data_manifests=DATA, artifacts=[str(rd)],
                                         extra={"n_pos": sc["n_pos"], "prevalence": sc["prevalence"],
                                                "split_sha256": sha, "auroc_ci_huc8": sc["auroc_ci_huc8"]})
            # paired comparisons vs the incumbent and vs the feature-matched RF (strongest simple baseline, W8)
            for comp in (INCUMBENT, "rf_geo_streamcat", "rf_geo_streamcat_nbr"):
                for name in preds:
                    if name == comp or name.startswith("prevalence") or name == "persistence_prev_cycle":
                        continue
                    pr = C.paired(C.fast_auroc, yv[msk], preds[name][msk], preds[comp][msk], units["huc4"][msk])
                    pr8 = C.paired(C.fast_auroc, yv[msk], preds[name][msk], preds[comp][msk], units["huc8"][msk])
                    pr["ci_huc8"] = [pr8["ci_low"], pr8["ci_high"]]
                    results["paired"][f"{sub}/{name}_minus_{comp}"] = pr
        if reg:
            claims = [("gbm_geo_streamcat", INCUMBENT), ("gbm_geo_streamcat", "rf_geo_streamcat"),
                      ("gbm_geo_streamcat_wx_nbr", INCUMBENT), ("gbm_geo_streamcat_wx_nbr", "rf_geo_streamcat_nbr"),
                      ("gbm_geo_streamcat_nbr", INCUMBENT), ("gbm_geo_streamcat_wx", INCUMBENT)]
            for key in ("newloc", "all"):
                for name, comp in claims:
                    pr = results["paired"][f"{key}/{name}_minus_{comp}"]
                    C.log_metric(app=APP, experiment="phase1_val", model=name, metric=f"auroc_diff_vs_{comp}",
                                 value=pr["diff"], split=f"val_nrsa1819_{key}", ci=pr,
                                 ci_method="paired cluster bootstrap over HUC4, B=2000", target="bent_mmi_cond Poor vs Good",
                                 config=cfg, data_manifests=DATA, artifacts=[str(rd)],
                                 extra={"p_value": pr["p_value"], "comparator": comp, "ci_huc8": pr["ci_huc8"]})
            h = results["dev"]["hill2017_oob_nrsa0809"]
            C.log_metric(app=APP, experiment="incumbent_reproduction", model=INCUMBENT, metric="oob_accuracy",
                         value=h["oob_accuracy"], split="dev_oob_nrsa0809", n=h["n"], is_baseline=True, config=cfg,
                         data_manifests=DATA, notes="Hill et al. 2017 design (random/OOB), published 75%",
                         extra={"oob_auroc": h["oob_auroc"]})
            for k, sc in results["dev"].items():
                if k.startswith("loeo_"):
                    C.log_metric(app=APP, experiment="dev_loeo", model=k[5:], metric="auroc", value=sc["auroc"],
                                 split="dev_loeo_nrsa0814", n=sc["n"],
                                 ci={"ci_low": sc["auroc_ci"][0], "ci_high": sc["auroc_ci"][1]},
                                 ci_method="cluster bootstrap over HUC4, B=2000", target="bent_mmi_cond Poor vs Good",
                                 is_baseline=k[5:] != "gbm_geo_streamcat", config=cfg, data_manifests=DATA)
        C.write_json(rd / "results.json", results)
        run.metric("val_newloc_auroc_gbm", results["val"]["newloc/gbm_geo_streamcat"]["auroc"])
        run.metric("val_newloc_auroc_incumbent", results["val"][f"newloc/{INCUMBENT}"]["auroc"])
        run.note(f"split {sha[:12]}; configs tried so far {C.n_configs_tried(APP)}")
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                          if kk in ("auroc", "auroc_ci", "accuracy", "n", "n_pos", "diff", "ci_low", "ci_high", "p_value",
                                    "oob_accuracy", "oob_auroc")}
                      for part in ("dev", "val", "paired") for k, v in results[part].items() if isinstance(v, dict)},
                     indent=1, default=str))
    print("run dir:", rd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
