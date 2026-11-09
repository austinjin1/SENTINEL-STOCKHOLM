#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D1n incumbent re-tuning (protocol §10, pre-test gate 2026-11-08): VB-MLR ridge α re-selected on the INNER
training-year split only (fit 2004–2012, score 2013–2015) over a widened grid; then validation is re-reported
for the re-selected incumbent and the unchanged claimed logistic (C = 1). Validation only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads()

from sentinel.models.downstream import d1n_beaches as A  # noqa: E402
from sentinel.models.downstream import wqp_track as WT  # noqa: E402

GRID = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
TAG = "@alpha_wide"


def main() -> int:
    split, sha = C.load_split(A.APP, expect_sha=A.SPLIT_SHA)
    with C.cpu_slot(f"{A.APP}_alpha_wide"):
        rows, X, ct = WT.build_frame(A.CFG, split)
        C.assert_no_test(rows)
        trm = (rows["role"] == "train").to_numpy()
        tr, Xtr = rows[trm].reset_index(drop=True), X[trm].reset_index(drop=True)
        a, b = WT._inner(tr, A.CFG)
        lv, y = tr["logv"].to_numpy(float), tr["y"].to_numpy(float)
        tra, trb = tr[a].reset_index(drop=True), tr[b].reset_index(drop=True)
        Xa, Xb = Xtr[a].reset_index(drop=True), Xtr[b].reset_index(drop=True)
        log = []
        for al in GRID:
            s = WT.VBMLR(al).fit(tra, Xa, lv[a]).predict(trb, Xb)
            v = C.fast_auroc(y[b], s)
            log.append({"alpha": al, "inner_auroc": v})
            C.log_config_tried(A.APP, f"vb_mlr_incumbent{TAG}", {"alpha": al}, status=f"inner_auroc={v:.4f}")
        best = max(log, key=lambda r: r["inner_auroc"])["alpha"]
        vb = WT.VBMLR(best).fit(tr, Xtr, lv)
        cols = WT.feature_cols(Xtr)
        logit = WT.fit_logistic(Xtr[cols], y, 1.0)
        res = {"grid": log, "selected_alpha": best, "sets": {}}
        for es in ("val_known", "val_new"):
            m = (rows["evalset"] == es).to_numpy()
            if es == "val_known":
                m &= rows["cluster"].isin(set(tr["cluster"])).to_numpy()
            ev, Xe = rows[m].reset_index(drop=True), X[m].reset_index(drop=True)
            yy, g = ev["y"].to_numpy(float), ev["huc8"].to_numpy()
            s_vb, s_lg = vb.predict(ev, Xe), logit(Xe[cols])
            sc_vb, sc_lg = C.binary_scores(yy, s_vb, g), C.binary_scores(yy, s_lg, g)
            d = C.paired(C.fast_auroc, yy, s_lg, s_vb, g)
            res["sets"][es] = {"vb_mlr_auroc": sc_vb["auroc"], "vb_mlr_ci": sc_vb["auroc_ci"], "logistic_auroc": sc_lg["auroc"],
                               "logistic_minus_vb": d}
            split_name = f"{es}_{A.CFG.val_years[0]}_{A.CFG.val_years[1]}"
            C.log_metric(app=A.APP, experiment=f"{es}{TAG}", model="vb_mlr_incumbent", metric="auroc", value=sc_vb["auroc"],
                         split=split_name, n=sc_vb["n"], ci={"ci_low": sc_vb["auroc_ci"][0], "ci_high": sc_vb["auroc_ci"][1]},
                         ci_method="cluster bootstrap over HUC8, B=2000", is_baseline=True, data_manifests=A.DATA_MANIFESTS,
                         extra={"alpha": best})
            C.log_metric(app=A.APP, experiment=f"{es}{TAG}", model="logistic", metric="auroc_diff_vs_vb_mlr_incumbent",
                         value=d["diff"], split=split_name, n=sc_vb["n"], ci=d, ci_method="paired cluster bootstrap over HUC8, B=2000",
                         data_manifests=A.DATA_MANIFESTS, extra={"p_value": d["p_value"], "alpha": best})
        C.write_json(C.run_dir(A.APP, "alpha_wide") / "results.json", res)
    print(json.dumps({"grid": log, "selected": best,
                      "sets": {k: {"vb": round(v["vb_mlr_auroc"], 4), "logit": round(v["logistic_auroc"], 4),
                                   "diff": [round(v["logistic_minus_vb"][x], 4) for x in ("diff", "ci_low", "ci_high", "p_value")]}
                               for k, v in res["sets"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
