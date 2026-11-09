#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D8 RiverPulse driver (protocol: docs/preregistration/downstream_riverpulse.md).

    python -B scripts/train/downstream_riverpulse.py --build-only          # split freeze check + features + counts
    python -B scripts/train/downstream_riverpulse.py [--confidence H]      # fit on train sites, score val sites

Test-role sites are dropped right after role assignment (phase 1 = validation only).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.eval import metrics as M  # noqa: E402
from sentinel.models.downstream import common as C  # noqa: E402
from sentinel.models.downstream import riverpulse as R  # noqa: E402

SEEDS = (0, 1, 2)
# iteration transforms (protocol §6b): fitted on f(y), predictions back-transformed to raw units
TRANSFORMS = {"base": (lambda y: y, lambda z: z), "it1": (np.arcsinh, np.sinh)}
LEARNED = ("light_season", "rf_segatto_incumbent", "rf_maavara_incumbent", "gbm_challenger")


def prepare(confidence=("H", "M")):
    sites = R.load_sites()
    role, prov = R.site_roles(sites)
    split_file = C.PREREG / R.SPLIT_FILE
    sha = C.freeze_split(R.APP, {"site_role": role}, R.split_meta(prov), path=split_file)
    if R.SPLIT_SHA is not None:
        C.load_split(R.APP, expect_sha=R.SPLIT_SHA, path=split_file)
    lab = R.load_labels(confidence=confidence)
    lab["role"] = lab["site_id"].map(role).fillna("test")
    counts_all = lab.groupby("role").agg(n_days=("gpp", "size"), n_sites=("site_id", "nunique")).reset_index()
    df = C.drop_protected(lab).reset_index(drop=True)
    n_prot = df.attrs["n_dropped_protected"]
    C.assert_no_test(df)
    X, cols = R.build_features(df, sites)
    huc4 = pd.Series(R._huc4_lookup(sites).to_numpy(), index=sites["site_id"])
    df["group"] = df["site_id"].map(huc4).fillna("nohuc:" + df["site_id"]).astype(str)  # bootstrap unit (W5)
    return df, X, cols, sites, sha, counts_all, n_prot


def build_report(df, X, cols, sites, sha, counts_all, n_prot) -> dict:
    sc_cols = [c for c in cols if c.startswith("sc_")]
    per_site_sc = X[sc_cols].notna().any(axis=1).groupby(df["site_id"]).any()
    src = R.streamcat_for_sites(sites).set_index("site_id")["sc_source"]
    used = sorted(df["site_id"].unique())
    return {"split_sha256": sha, "streamcat_source_counts_phase1_sites": src.reindex(used).fillna("none").value_counts().to_dict(),
            "n_basin_groups": df.groupby("role")["group"].nunique().to_dict(), "counts_by_role_incl_test": counts_all.to_dict("records"),
            "n_days_dropped_protected": int(n_prot),
            "phase1_counts": df.groupby("role").agg(n_days=("gpp", "size"), n_sites=("site_id", "nunique"),
                                                    gpp_mean=("gpp", "mean"), er_mean=("er", "mean")).reset_index().to_dict("records"),
            "n_features": len(cols), "n_streamcat_features": len(sc_cols),
            "sites_without_streamcat": sorted(per_site_sc.index[~per_site_sc]),
            "missing_frac_max": float(X[cols].isna().mean().max()),
            "missing_frac_weather": {c: float(X[c].isna().mean()) for c in cols if c.startswith("wx_")},
            "wx_cell_km": X["wx_cell_km"].describe().to_dict()}


def select(df, X, cols, cfg) -> dict:
    """Inner selection on TRAIN sites only: one shared inner hold-out of 20 % of train HUC4 groups; 4
    configurations per learned model; criterion = inner MSE of daily GPP (the primary target)."""
    tr = df[df["role"] == "train"]
    Xtr = X.loc[tr.index]
    grp = tr["group"].to_numpy()
    inner = np.isin(grp, list(R.inner_groups(grp)))
    fwd, inv = TRANSFORMS[cfg.get("iteration", "base")]
    y_raw = tr["gpp"].to_numpy(float)
    y = fwd(y_raw)
    out = {}
    sc = {}
    for c in R.GRIDS["light_season"]:
        m = R.fit_light_season(Xtr[~inner], y[~inner], **c)
        sc[json.dumps(c)] = float(np.mean((inv(m.predict(Xtr.loc[inner, list(R.LIGHT_COLS)].to_numpy(float))) - y_raw[inner]) ** 2))
    out["light_season"] = sc
    sc = {}
    for c in R.GRIDS["rf_segatto_incumbent"]:
        m = R.fit_rf(Xtr[~inner], y[~inner], cols, params={**c, "n_estimators": R.RF_SELECT_TREES})
        sc[json.dumps(c)] = float(np.mean((inv(m.predict(Xtr.loc[inner, cols].to_numpy(np.float32))) - y_raw[inner]) ** 2))
    out["rf_segatto_incumbent"] = sc
    sc = {}
    mcols = list(R.MAAVARA_FEATURES)
    for c in R.GRIDS["rf_maavara_incumbent"]:
        m = R.fit_rf(Xtr[~inner], y[~inner], mcols, params={**c, "n_estimators": R.RF_SELECT_TREES})
        sc[json.dumps(c)] = float(np.mean((inv(m.predict(Xtr.loc[inner, mcols].to_numpy(np.float32))) - y_raw[inner]) ** 2))
    out["rf_maavara_incumbent"] = sc
    sc = {}
    for c in R.GRIDS["gbm_challenger"]:
        m = R.fit_gbm(Xtr, y, cols, grp, seed=0, params=c)
        if cfg.get("iteration", "base") == "base":
            sc[json.dumps(c)] = m.inner_mse_
        else:  # raw-unit inner MSE (same criterion as the other models)
            sc[json.dumps(c)] = float(np.mean((inv(m.predict(Xtr.loc[inner, cols])) - y_raw[inner]) ** 2))
    out["gbm_challenger"] = sc
    for name, scores in out.items():
        for k, v in scores.items():
            C.log_config_tried(R.APP, name, {**cfg, "hp": json.loads(k), "inner_mse_gpp": v})
    return {name: (json.loads(min(v, key=v.get)), v) for name, v in out.items()}


def score(df, X, cols, sites, out_dir: Path, cfg: dict, register: bool, run, split_tag: str) -> dict:
    tr = (df["role"] == "train").to_numpy()
    va = (df["role"] == "val").to_numpy()
    train, ev = df[tr], df[va]
    sel = select(df, X, cols, cfg)
    results = {"selection": {k: {"chosen": v[0], "inner_mse_gpp": v[1]} for k, v in sel.items()}}
    pred_frame = {"site_id": ev["site_id"].to_numpy(), "date": ev["date"].to_numpy(), "group": ev["group"].to_numpy()}
    simple = ("doy_climatology", "location_knn", "light_season")
    fwd, inv = TRANSFORMS[cfg.get("iteration", "base")]
    sfx = "" if cfg.get("iteration", "base") == "base" else "_" + cfg["iteration"]
    for target in R.TARGETS:
        ytr = fwd(train[target].to_numpy(float))
        preds = {"doy_climatology": R.predict_doy(R.doy_climatology_fit(train, target), ev),
                 "location_knn": R.location_knn(train, sites, ev, target)}
        preds["light_season"] = inv(R.fit_light_season(X[tr], ytr, **sel["light_season"][0]).predict(
            X.loc[va, list(R.LIGHT_COLS)].to_numpy(float)))
        rf = R.fit_rf(X[tr], ytr, cols, params=sel["rf_segatto_incumbent"][0])
        preds["rf_segatto_incumbent"] = inv(rf.predict(X.loc[va, cols].to_numpy(np.float32)))
        mcols = list(R.MAAVARA_FEATURES)
        rfm = R.fit_rf(X[tr], ytr, mcols, params=sel["rf_maavara_incumbent"][0])
        preds["rf_maavara_incumbent"] = inv(rfm.predict(X.loc[va, mcols].to_numpy(np.float32)))
        g = [inv(R.fit_gbm(X[tr], ytr, cols, train["group"].to_numpy(), seed=s, params=sel["gbm_challenger"][0]).predict(
            X.loc[va, cols])) for s in SEEDS]
        preds["gbm_challenger"] = np.mean(g, axis=0)
        for s, p in zip(SEEDS, g):
            preds[f"gbm_challenger_s{s}"] = p
        y, grp = ev[target].to_numpy(float), ev["group"].to_numpy()
        res = {name: C.regression_scores(y, p, grp) for name, p in preds.items()}
        sy = {name: R.site_year_means(ev, p, target) for name, p in preds.items() if not name[-3:-1] == "_s"}
        hmap = ev.groupby("site_id")["group"].first()
        res_sy = {name: C.regression_scores(t["obs"], t["pred"], t["site_id"].map(hmap)) for name, t in sy.items()}
        best_simple = max(simple, key=lambda b: res[b]["r2"])
        pairs = {}
        for chal in ("gbm_challenger", "rf_segatto_incumbent", "rf_maavara_incumbent"):
            for b in simple + ("rf_segatto_incumbent", "rf_maavara_incumbent", "gbm_challenger"):
                if b != chal:
                    pairs[f"{chal}-{b}"] = C.paired(M.nse, y, preds[chal], preds[b], grp)
        results[target] = {"daily": res, "site_year_mean": res_sy, "paired_r2": pairs, "best_simple_baseline": best_simple}
        for name, p in preds.items():
            pred_frame[f"{target}_{name}"] = p
        pred_frame[target] = y
        if register:
            for level, rr in (("daily", res), ("site_year_mean", res_sy)):
                for name, sc in rr.items():
                    if name[-3:-1] == "_s":
                        continue
                    for met in ("r2", "rmse", "spearman"):
                        C.log_metric(app=R.APP, experiment=f"{target}_{level}",
                                     model=f"riverpulse_{name}" + (sfx if name in LEARNED else ""), metric=met,
                                     value=sc[met], split=split_tag, n=sc["n"],
                                     ci=dict(zip(("ci_low", "ci_high"), sc[f"{met}_ci"])),
                                     ci_method="cluster bootstrap over HUC4 basin groups, B=2000", target=target,
                                     is_baseline=name != "gbm_challenger", config=cfg, data_manifests=R.DATA_MANIFESTS,
                                     artifacts=[str(out_dir / "val_predictions.parquet")],
                                     extra={"n_groups": sc["n_groups"], "split_sha256": cfg.get("split_sha256"),
                                            "streamcat_source_counts": cfg.get("streamcat_source_counts"),
                                            "seed_scores": {s: res[f"gbm_challenger_s{s}"][met] for s in SEEDS}
                                            if (name == "gbm_challenger" and level == "daily") else None})
            for key, pr in pairs.items():
                C.log_metric(app=R.APP, experiment=f"{target}_daily", model=f"riverpulse_{key.replace('-', '_minus_')}{sfx}",
                             metric="r2_diff", value=pr["diff"], split=split_tag, n=len(y), ci=pr,
                             ci_method="paired cluster bootstrap (HUC4 group), B=2000", target=target, config=cfg,
                             data_manifests=R.DATA_MANIFESTS, extra={"p_value": pr["p_value"]})
        run.metric(f"{target}_gbm_r2", res["gbm_challenger"]["r2"])
    pd.DataFrame(pred_frame).to_parquet(out_dir / "val_predictions.parquet", index=False)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--confidence", default="HM", choices=["HM", "H"], help="HM primary; H = sensitivity")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--iteration", default="base", choices=["base", "it1"],
                    help="it1 = asinh target transform for every learned model (back-transformed; protocol §6b)")
    ap.add_argument("--allow-missing-catchment", action="store_true",
                    help="train without manifests/downstream_catchment.json (recorded in cfg; not for results)")
    a = ap.parse_args()
    C.limit_threads()
    conf = tuple(a.confidence)
    cfg = {"app": R.APP, "iteration": a.iteration, "confidence": conf, "gbm": R.GBM_PARAMS, "seeds": SEEDS, "phase": 1,
           "rf_grid": R.GRIDS["rf_segatto_incumbent"], "gbm_grid": R.GRIDS["gbm_challenger"],
           "allow_missing_catchment": bool(a.allow_missing_catchment), "catchment_ready": R.catchment_ready()}
    name = "build" if a.build_only else f"val_conf{a.confidence}" + ("" if a.iteration == "base" else f"_{a.iteration}")
    out_dir = C.run_dir(R.APP, name)
    with (contextlib.nullcontext() if a.build_only else C.cpu_slot(f"riverpulse_{name}")):
        if a.build_only:
            rep = build_report(*prepare(conf))
            C.write_json(out_dir / "build_report.json", rep)
            print(json.dumps(rep, indent=1, default=str))
            return 0
        if not R.catchment_ready() and not a.allow_missing_catchment:
            raise SystemExit("refusing to train: manifests/downstream_catchment.json missing (review C4); "
                             "pass --allow-missing-catchment only for diagnostics")
        with TrainingRun(C.WORKSTREAM, f"{R.APP}/{name}", config=cfg, data=list(R.DATA_MANIFESTS),
                         plan_ref=C.PLAN_REF + " D8", iteration=a.iteration, artifacts_dir=str(out_dir)) as run:
            df, X, cols, sites, sha, counts_all, n_prot = prepare(conf)
            cfg["split_sha256"] = sha
            rep = build_report(df, X, cols, sites, sha, counts_all, n_prot)
            cfg["streamcat_source_counts"] = rep["streamcat_source_counts_phase1_sites"]
            C.write_json(out_dir / "build_report.json", rep)
            tag = "val_sites_2007_2016" + ("" if a.confidence == "HM" else "_confH")
            res = score(df, X, cols, sites, out_dir, cfg, register=not a.no_register and not a.allow_missing_catchment, run=run, split_tag=tag)
            C.write_json(out_dir / "val_results.json", res)
            run.note(f"split {sha[:12]}; validation sites only; configs tried so far {C.n_configs_tried(R.APP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
