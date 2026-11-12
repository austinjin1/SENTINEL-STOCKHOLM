#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D8 RiverPulse it2: site-regime + daily-anomaly decomposition (protocol §6b it2). Validation only.

    python -B scripts/train/downstream_riverpulse_decomp.py

Challenger: sinh(R(site) + A(site, day)); R = RF on one row per training site (static features -> site mean of
asinh(target)); A = LightGBM (it1-selected hyperparameters) on asinh(target) − site mean. Compared on identical
rows with the it1 validation predictions of the incumbents / baselines (same split, same asinh treatment).
"""

from __future__ import annotations

import importlib.util
import json
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

from sentinel.models.downstream import riverpulse as R  # noqa: E402

_spec = importlib.util.spec_from_file_location("rp_driver", ROOT / "scripts" / "train" / "downstream_riverpulse.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
SEEDS = (0, 1, 2)
RF_PARAMS = {"n_estimators": 500, "min_samples_leaf": 3, "max_features": 0.33}


def main() -> int:
    from sklearn.ensemble import RandomForestRegressor

    df, X, cols, sites, sha, _, _ = D.prepare(("H", "M"))
    it1 = json.loads((C.run_dir(R.APP, "val_confHM_it1") / "val_results.json").read_text())
    gpar = it1["selection"]["gbm_challenger"]["chosen"]
    prev = pd.read_parquet(C.run_dir(R.APP, "val_confHM_it1") / "val_predictions.parquet")
    tr, va = (df["role"] == "train").to_numpy(), (df["role"] == "val").to_numpy()
    train, ev = df[tr], df[va]
    if not (np.array_equal(ev["site_id"].to_numpy(), prev["site_id"].to_numpy())
            and np.array_equal(pd.to_datetime(ev["date"]).to_numpy(), pd.to_datetime(prev["date"]).to_numpy())):
        raise RuntimeError("validation rows differ from the it1 predictions")
    static = [c for c in cols if c.startswith("sc_")] + [c for c in ("lat", "lon") if c in X.columns]
    out_dir = C.run_dir(R.APP, "val_confHM_it2_decomp")
    for name, cfg in (("regime_rf", RF_PARAMS), ("anomaly_gbm", gpar)):
        C.log_config_tried(R.APP, f"{name}@it2_decomp", cfg, status="fixed (no search)")
    results = {}
    with TrainingRun(C.WORKSTREAM, f"{R.APP}/it2_decomp", config={"split_sha256": sha, "rf": RF_PARAMS, "gbm": gpar},
                     data=list(R.DATA_MANIFESTS), plan_ref=C.PLAN_REF + " D8", artifacts_dir=str(out_dir)) as run:
        frame = {"site_id": ev["site_id"].to_numpy(), "date": ev["date"].to_numpy()}
        for target in R.TARGETS:
            z = np.arcsinh(train[target].to_numpy(float))
            site_mean = pd.Series(z, index=train.index).groupby(train["site_id"]).mean()
            S_tr = X.loc[train.index, static].groupby(train["site_id"]).first().loc[site_mean.index]
            reg = RandomForestRegressor(random_state=0, n_jobs=C.N_THREADS, **RF_PARAMS).fit(
                np.nan_to_num(S_tr.to_numpy(np.float32), nan=-9999.0),
                site_mean.to_numpy())
            S_ev = X.loc[ev.index, static]
            r_ev = reg.predict(np.nan_to_num(S_ev.to_numpy(np.float32), nan=-9999.0))
            anom = z - train["site_id"].map(site_mean).to_numpy()
            per = []
            for s in SEEDS:
                g = R.fit_gbm(X[tr], anom, cols, train["group"].to_numpy(), seed=s, params=gpar)
                per.append(np.sinh(r_ev + g.predict(X.loc[va, cols])))
            pred = np.mean(per, axis=0)
            y, grp = ev[target].to_numpy(float), ev["group"].to_numpy()
            comps = {k: prev[f"{target}_{k}"].to_numpy(float) for k in
                     ("doy_climatology", "location_knn", "light_season", "rf_segatto_incumbent", "rf_maavara_incumbent", "gbm_challenger")}
            sc = {"decomp": C.regression_scores(y, pred, grp), **{k: C.regression_scores(y, v, grp) for k, v in comps.items()}}
            best_simple = max(("doy_climatology", "location_knn", "light_season"), key=lambda k: sc[k]["r2"])
            strong_rf = max(("rf_segatto_incumbent", "rf_maavara_incumbent"), key=lambda k: sc[k]["r2"])
            pairs = {b: C.paired(M.nse, y, pred, comps[b], grp) for b in comps}
            for b in pairs:
                pairs[b]["per_seed_diff"] = [float(M.nse(y, q) - M.nse(y, comps[b])) for q in per]
            results[target] = {"r2": {k: v["r2"] for k, v in sc.items()}, "r2_ci": {k: v["r2_ci"] for k, v in sc.items()},
                               "best_simple": best_simple, "strongest_rf": strong_rf, "paired": pairs,
                               "per_seed_r2": [float(M.nse(y, q)) for q in per]}
            C.log_metric(app=R.APP, experiment=f"{target}_daily@it2_decomp", model="riverpulse_decomp", metric="r2",
                         value=sc["decomp"]["r2"], split="val_sites", n=sc["decomp"]["n"],
                         ci={"ci_low": sc["decomp"]["r2_ci"][0], "ci_high": sc["decomp"]["r2_ci"][1]},
                         ci_method="cluster bootstrap over HUC4 basin groups, B=2000", target=target,
                         data_manifests=R.DATA_MANIFESTS, extra={"split_sha256": sha})
            for b, pr in pairs.items():
                C.log_metric(app=R.APP, experiment=f"{target}_daily@it2_decomp", model=f"riverpulse_decomp_minus_{b}",
                             metric="r2_diff", value=pr["diff"], split="val_sites", n=len(y), ci=pr,
                             ci_method="paired cluster bootstrap (HUC4 group), B=2000", target=target,
                             data_manifests=R.DATA_MANIFESTS, extra={"p_value": pr["p_value"]})
            frame[f"{target}_decomp"] = pred
            run.metric(f"{target}_decomp_r2", sc["decomp"]["r2"])
        pd.DataFrame(frame).to_parquet(out_dir / "val_predictions.parquet", index=False)
        C.write_json(out_dir / "val_results.json", results)
    for t, r in results.items():
        print(t, {k: round(v, 3) for k, v in r["r2"].items()}, "strongest RF", r["strongest_rf"], "best simple", r["best_simple"])
        for b in (r["strongest_rf"], r["best_simple"]):
            p = r["paired"][b]
            print("   decomp -", b, round(p["diff"], 3), [round(p["ci_low"], 3), round(p["ci_high"], 3)], round(p["p_value"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
