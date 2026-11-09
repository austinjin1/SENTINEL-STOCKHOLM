#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D5 it3_ens: equal-weight GBM+RF probability average from the saved it2 validation predictions
(protocol §9; no refit, no weight tuning). Validation only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import biointegrity as BI  # noqa: E402
from sentinel.models.downstream import common as C  # noqa: E402

APP = BI.APP
IT2 = "it2_tuned_20261025-012347"
COMPS = ["hill2017_rf_streamcat", "tuned_hill2017_rf_streamcat", "rf_geo_streamcat", "tuned_rf_geo_streamcat",
         "tuned_gbm_geo_streamcat"]


def main() -> int:
    units = BI.load_units()
    split, sha = C.load_split(APP, expect_sha=BI.SPLIT_SHA)
    u = C.drop_protected(units)
    C.assert_no_test(u)
    va = u[u["role"] == "val"].set_index("visit_id")
    p = pd.read_parquet(C.run_dir(APP, IT2) / "val_predictions.parquet").set_index("visit_id")
    va = va.loc[p.index]
    p["ens_tuned_gbm_rf"] = 0.5 * (p["tuned_gbm_geo_streamcat"] + p["tuned_rf_geo_streamcat"])
    y = p["y_poor"].to_numpy(float)
    g4 = va["huc4"].astype(str).fillna("NA").to_numpy()
    g8 = va["huc8"].astype(str).fillna("NA").to_numpy()
    res = {}
    for sub, m in {"newloc": va["new_location"].to_numpy(bool), "all": np.ones(len(va), bool)}.items():
        sc = C.binary_scores(y[m], p["ens_tuned_gbm_rf"].to_numpy()[m], g4[m])
        res[f"{sub}/auroc"] = sc
        C.log_metric(app=APP, experiment="iter/it3_ens", model="ens_tuned_gbm_rf", metric="auroc", value=sc["auroc"],
                     split=f"val_nrsa1819_{sub}", n=sc["n"], ci={"ci_low": sc["auroc_ci"][0], "ci_high": sc["auroc_ci"][1]},
                     ci_method="cluster bootstrap over HUC4, B=2000", target="bent_mmi_cond Poor vs Good",
                     data_manifests=["nars_streams", "downstream_catchment"], extra={"split_sha256": sha, "from_run": IT2})
        for comp in COMPS:
            pr = C.paired(C.fast_auroc, y[m], p["ens_tuned_gbm_rf"].to_numpy()[m], p[comp].to_numpy()[m], g4[m])
            pr8 = C.paired(C.fast_auroc, y[m], p["ens_tuned_gbm_rf"].to_numpy()[m], p[comp].to_numpy()[m], g8[m])
            pr["ci_huc8"] = [pr8["ci_low"], pr8["ci_high"]]
            res[f"{sub}/minus_{comp}"] = pr
            C.log_metric(app=APP, experiment="iter/it3_ens", model="ens_tuned_gbm_rf", metric=f"auroc_diff_vs_{comp}",
                         value=pr["diff"], split=f"val_nrsa1819_{sub}", n=int(m.sum()), ci=pr,
                         ci_method="paired cluster bootstrap over HUC4, B=2000", target="bent_mmi_cond Poor vs Good",
                         data_manifests=["nars_streams", "downstream_catchment"],
                         extra={"p_value": pr["p_value"], "ci_huc8": pr["ci_huc8"], "from_run": IT2})
            print(sub, comp, round(pr["diff"], 4), [round(pr["ci_low"], 4), round(pr["ci_high"], 4)],
                  [round(x, 4) for x in pr["ci_huc8"]], round(pr["p_value"], 3))
        print(sub, "ens auroc", round(sc["auroc"], 4))
    C.log_config_tried(APP, "ens_tuned_gbm_rf", {"weights": [0.5, 0.5], "from_run": IT2, "iteration": "it3_ens"})
    C.write_json(C.run_dir(APP, "it3_ens") / "results.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
