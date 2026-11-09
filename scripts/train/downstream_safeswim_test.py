#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D1 SafeSwim (Chicago D1b qPCR Enterococci nowcast) single test look (FREEZE_CANDIDATE_downstream_safeswim.json).

    python -B scripts/train/downstream_safeswim_test.py --dry-run-on-val      # val 2022-23 in the test role; nothing registered
    python -B scripts/train/downstream_safeswim_test.py --allow-test          # ONLY after FREEZE_downstream_safeswim.json exists
    python -B scripts/train/downstream_safeswim_test.py --rescore-from-saved  # crash recovery only (no re-fit)

Frozen: L2 logistic (C = 1) on ``compact_each+q`` (26 columns, asserted), fitted on the D1b training years
(2015–2019, 2021). Test rows = 2024–2025 non-input beaches with an official City prediction; the 2026
reserve season is never loaded (protocol §11). Primary (§4a): vs official City model and vs feature-beach GM,
paired AUROC under ISO-week AND beach clusters; Holm with m = 4 (the two unscored D1a / D1b-forecast primary
comparisons count with p = 1), both cluster units must reject. Deterministic → seed rule trivially met.
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

from sentinel.models.downstream import safeswim as S  # noqa: E402
from sentinel.models.downstream import testlook as TL  # noqa: E402

FROZEN_FEATURES = ["doy", "latitude", "prev_log", "prev_gap_d", "wx_precip_sum_1d", "wx_precip_sum_3d", "wx_tmax_mean_3d",
                   "inp_sd_mean_log", "inp_sd_max_log", "inp_sd_n", "inpx_calumet", "inpx_63rdstreet", "inpx_rainbow",
                   "inpx_montrose", "inpx_southshore", "inpx_nearest_log", "inpx_nearest_km",
                   "drv_q_05536000_log", "drv_q_05536000_max3", "drv_q_05536000_rel", "drv_q_05536290_log",
                   "drv_q_05536290_max3", "drv_q_05536290_rel", "drv_q_05536890_log", "drv_q_05536890_max3",
                   "drv_q_05536890_rel"]
FROZEN = {"task": "d1b", "mode": "nowcast", "model": "logistic_l2", "config": S.MODEL_CONFIGS["logistic_l2"],
          "feature_set": "compact_each+q", "features": FROZEN_FEATURES, "fit_roles": ["train"], "rows": "official_rows",
          "test_years": [2024, 2025]}
PRIMARY = {"logistic_l2": ["official_city_model", "feature_beach_gm"]}
SECONDARY = {"logistic_l2": ["persistence", "climatology_beach_month", "location_only", "rain_rule_1d"],
             "gbm_hgb": ["official_city_model", "feature_beach_gm"]}
UNSCORED_PRIMARY = ("d1b_forecast-persistence", "d1a_forecast-persistence")
MODELS = ["official_city_model", "feature_beach_gm", "persistence", "climatology_beach_month", "location_only",
          "rain_rule_1d", "logistic_l2", "gbm_hgb"]


def predict(mode: str, split: dict, sha: str) -> pd.DataFrame:
    X = S.build("d1b", "nowcast", split, include_test=(mode == "test"), split_sha=sha)
    cols = S.feature_columns(X, "d1b", "nowcast", feature_set=FROZEN["feature_set"])
    if cols != FROZEN_FEATURES:
        raise C.ProtectedDataError(f"frozen feature set changed: {sorted(set(cols) ^ set(FROZEN_FEATURES))}")
    tr = X[X["role"].isin(FROZEN["fit_roles"])]
    ev = X[X["role"] == ("test" if mode == "test" else "val")].copy()
    if mode == "test":
        ev = ev[ev["date"].dt.year.isin(FROZEN["test_years"])]
        if ev.empty or not ev["date"].dt.year.isin(FROZEN["test_years"]).all():
            raise C.ProtectedDataError("test rows outside 2024-2025")
    sc = S.baseline_scores(tr, ev, "d1b", "nowcast")
    sc["logistic_l2"] = S.fit_predict("logistic_l2", tr[cols], tr["y"].to_numpy(), ev[cols])
    sc["gbm_hgb"] = S.fit_predict("gbm_hgb", tr[cols], tr["y"].to_numpy(), ev[cols])
    rm = np.isfinite(sc["official_city_model"])                 # official_rows
    p = ev.loc[rm, ["beach_key", "date", "y"]].copy()
    p["week"] = S.cluster_ids(ev)[rm]
    for k in MODELS:
        p[f"s_{k}"] = np.asarray(sc[k], float)[rm]
    return p.reset_index(drop=True)


def score(pred: pd.DataFrame, mode: str, sha: str) -> dict:
    name = "test_2024_2025_official_rows" if mode != "dryrun" else "dryrun_val_2022_2023_official_rows"
    r = TL.score_set(app=S.APP, mode=mode, split_sha=sha, set_name=name, experiment="d1b_nowcast/test_look",
                     y=pred["y"].to_numpy(float), scores={k: pred[f"s_{k}"].to_numpy() for k in MODELS},
                     groups={"week": pred["week"].to_numpy(), "beach": pred["beach_key"].to_numpy()},
                     primary=PRIMARY, secondary=SECONDARY,
                     prob_models=("logistic_l2", "gbm_hgb", "location_only", "climatology_beach_month"),
                     manifests=S.DATA_MANIFESTS)
    res = {"mode": mode, "frozen": FROZEN, "split_sha256": sha, "sets": {name: r},
           "years_scored": sorted(pred["date"].dt.year.unique().tolist())}
    res["holm_primary_week"] = TL.holm_primary(res["sets"], "week", unscored=UNSCORED_PRIMARY)
    res["holm_primary_beach"] = TL.holm_primary(res["sets"], "beach", unscored=UNSCORED_PRIMARY)
    cw = TL.claim_met(res["holm_primary_week"], deterministic=True)
    cb = TL.claim_met(res["holm_primary_beach"], deterministic=True)
    res["claim_week"], res["claim_beach"] = cw, cb
    res["claim_met"] = bool(cw["claim_met"] and cb["claim_met"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    TL.add_mode_args(ap)
    a = ap.parse_args(argv)
    mode = TL.mode_from_args(a)
    split, sha = S.load_frozen_split()
    if sha != S.SPLIT_SHA:
        raise C.ProtectedDataError("split sha mismatch")
    TL.verify_data(S.APP)
    claim = None
    with C.cpu_slot(f"safeswim_{mode}"):
        if mode == "test":
            claim = TL.authorise(S.APP, sha)
            pred = predict(mode, split, sha)
            TL.save_predictions(S.APP, sha, pred)
        elif mode == "rescore":
            pred = TL.authorise_rescore(S.APP, sha)
        else:
            pred = predict(mode, split, sha)
        res = score(pred, mode, sha)
        res["claim"] = claim
        rd = TL.look_dir(S.APP) if mode != "dryrun" else C.run_dir(S.APP, "testpath_dryrun")
        if mode == "dryrun":
            pred.to_parquet(rd / "predictions.parquet", index=False)
        C.write_json(rd / ("results.json" if mode != "rescore" else "results_rescore.json"), res)
        if mode != "dryrun":
            TL.mark_scored(S.APP, {"claim_met": res["claim_met"]})
    r = next(iter(res["sets"].values()))
    print(json.dumps({k: r["scores"][k]["auroc"] for k in ("logistic_l2", "official_city_model", "feature_beach_gm")}, indent=1))
    print({u: {k: (v["p_value"], v["holm_reject"]) for k, v in res[f"holm_primary_{u}"].items()} for u in ("week", "beach")},
          "claim_met:", res["claim_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
