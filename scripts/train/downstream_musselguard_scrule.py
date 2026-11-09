#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D6: the pre-registered SC-rule baseline (Whittier-2008 analogue, protocol §6) on validation origin 2015
(pre-test gate 2026-11-08, protocol §12). The rule needs no fit: HUC8 median WQP specific conductance over
[T − 9, T] → class 0..3. Paired against the saved pressure_gbm validation predictions (identical model in it2
and it2b). Validation only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402
from sentinel.models.downstream import musselguard as MG  # noqa: E402

VAL_SPLIT = "val_origin2015_2016_2018"


def main() -> int:
    vp = pd.read_parquet(C.run_dir(MG.APP, "phase1_it2b_kanankege_grid") / "val_predictions.parquet")
    C.assert_no_test(vp.assign(role="val"))
    if set(vp["origin"].unique()) != {2015}:
        raise C.ProtectedDataError("expected validation origin 2015 only")
    prox = MG.wqp_huc8_proxies(2015)
    vp["huc8"] = vp["huc8"].astype(str)
    vp = vp.merge(prox[["huc8", "huc8_spcond_median"]], on="huc8", how="left")
    vp["score_sc_rule"] = MG.sc_rule_score(vp)
    res = {}
    for stratum, col in (("front100km", "front"), ("conus", None), ("mn", "mn")):
        m = np.ones(len(vp), bool) if col is None else vp[col].to_numpy(bool)
        sub = vp[m]
        ok = np.isfinite(sub["score_sc_rule"].to_numpy())
        bs = C.binary_scores(sub["y"].to_numpy()[ok], sub["score_sc_rule"].to_numpy()[ok], sub["huc8"].to_numpy()[ok])
        d = C.paired(C.fast_auprc, sub["y"].to_numpy(), sub["score_pressure_gbm"].to_numpy(), sub["score_sc_rule"].to_numpy(),
                     sub["huc8"].to_numpy())
        res[stratum] = {"sc_rule": bs, "coverage": float(ok.mean()), "pressure_gbm_minus_sc_rule_auprc": d}
        exp = f"rolling_origin/{stratum}@it2_scrule"
        for met in ("auroc", "auprc"):
            if np.isfinite(bs.get(met, np.nan)):
                C.log_metric(app=MG.APP, experiment=exp, model="sc_rule", metric=met, value=bs[met], split=VAL_SPLIT,
                             n=bs["n"], ci={"ci_low": bs[f"{met}_ci"][0], "ci_high": bs[f"{met}_ci"][1]},
                             ci_method="cluster bootstrap over HUC8, B=2000", is_baseline=True,
                             data_manifests=MG.DATA_MANIFESTS + ["wqp_discrete"], extra={"coverage": res[stratum]["coverage"]})
        C.log_metric(app=MG.APP, experiment=exp, model="pressure_gbm", metric="auprc_diff_vs_sc_rule", value=d["diff"],
                     split=VAL_SPLIT, n=int(ok.sum()), ci=d, ci_method="paired cluster bootstrap over HUC8, B=2000 (SC-covered rows)",
                     data_manifests=MG.DATA_MANIFESTS + ["wqp_discrete"], extra={"p_value": d["p_value"]})
    C.log_config_tried(MG.APP, "sc_rule@it2_scrule", {"rule": "HUC8 median SC classes <100/100-200/200-300/>300"},
                       status="rule, no fit")
    C.write_json(C.run_dir(MG.APP, "it2_scrule") / "results.json", res)
    print(json.dumps({k: {"sc_auprc": v["sc_rule"].get("auprc"), "coverage": v["coverage"],
                          "diff": [v["pressure_gbm_minus_sc_rule_auprc"][x] for x in ("diff", "ci_low", "ci_high", "p_value")]}
                      for k, v in res.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
