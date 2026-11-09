#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D1n single test look (FREEZE_CANDIDATE_downstream_d1n_beaches.json; protocol §5, §10).

    python -B scripts/train/downstream_d1n_beaches_test.py --dry-run-on-val      # same pipeline, val as "test", nothing registered
    python -B scripts/train/downstream_d1n_beaches_test.py --allow-test          # ONLY after FREEZE_downstream_d1n_beaches.json exists
    python -B scripts/train/downstream_d1n_beaches_test.py --rescore-from-saved  # crash recovery only (no re-fit)

Order: pinned data verified → FREEZE/claim (testlook.authorise) → test rows built → predictions saved → scored.
Frozen: claimed model L2 logistic C = 1 (deterministic → seed rule trivially met); comparators VB-MLR
incumbent (ridge α = 100, re-selected on the widened inner grid) and the strongest simple baseline on
validation (history_rate); GBM (15 leaves, min_child 20, 503 trees, 3 seeds) reported. Test fit = train + val
rows for every model (§10); dry-run fit = train rows. Holm over the 4 primary comparisons (2 × 2 sets).
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

from sentinel.models.downstream import d1n_beaches as A  # noqa: E402
from sentinel.models.downstream import testlook as TL  # noqa: E402
from sentinel.models.downstream import wqp_track as WT  # noqa: E402

FROZEN = {"logistic_C": 1.0, "vb_alpha": 100.0,
          "gbm": {"num_leaves": 15, "min_child_samples": 20}, "gbm_n_estimators": 503, "seeds": (0, 1, 2),
          "strongest_simple_baseline": "history_rate", "claimed_model": "logistic"}
PRIMARY = {"logistic": ["vb_mlr_incumbent", "history_rate"]}
SECONDARY = {"gbm_challenger": ["vb_mlr_incumbent", "history_rate"], "logistic": ["climatology", "location_only", "persistence"]}
MODELS = ["climatology", "location_only", "history_rate", "persistence", "logistic", "vb_mlr_incumbent", "gbm_challenger"]


def predict(mode: str, split: dict, sha: str) -> pd.DataFrame:
    cfg = A.CFG
    rows, X, ct = WT.build_frame(cfg, split, include_test=(mode == "test"), split_sha=sha)
    fit_m = rows["role"].isin(["train", "val"] if mode == "test" else ["train"]).to_numpy()
    fr, Xf = rows[fit_m].reset_index(drop=True), X[fit_m].reset_index(drop=True)
    Xtrain_role = X[(rows["role"] == "train").to_numpy()].reset_index(drop=True)      # strata thresholds: training rows
    fit_clusters = set(fr["cluster"])
    if mode == "test":
        sets = {"test_known_2021_2025": (rows["evalset"] == "test_known") & rows["cluster"].isin(fit_clusters),
                "test_new_2021_2025": rows["evalset"] == "test_new"}
    else:
        sets = {"dryrun_val_known": (rows["evalset"] == "val_known") & rows["cluster"].isin(fit_clusters),
                "dryrun_val_new": rows["evalset"] == "val_new"}
    cols = WT.feature_cols(Xf)
    y_f = fr["y"].to_numpy(float)
    logit = WT.fit_logistic(Xf[cols], y_f, FROZEN["logistic_C"])
    vb = WT.VBMLR(FROZEN["vb_alpha"]).fit(fr, Xf, fr["logv"].to_numpy(float))
    gbms = [WT.fit_gbm(Xf[cols], y_f, FROZEN["gbm"], s, n_estimators=FROZEN["gbm_n_estimators"]) for s in FROZEN["seeds"]]
    out = []
    for name, m in sets.items():
        m = m.to_numpy()
        ev, Xe = rows[m].reset_index(drop=True), X[m].reset_index(drop=True)
        sc = WT.baseline_scores(fr, Xf, ev, Xe, ct, cfg)
        sc["logistic"] = logit(Xe[cols])
        sc["vb_mlr_incumbent"] = vb.predict(ev, Xe)
        per = [g.predict_proba(Xe[cols])[:, 1] for g in gbms]
        sc["gbm_challenger"] = np.mean(per, axis=0)
        p = ev[["cluster", "date", "y", "huc8"]].copy()
        p["set"] = name
        for k in MODELS:
            p[f"s_{k}"] = np.asarray(sc[k], float)
        for i, q in enumerate(per):
            p[f"s_gbm_challenger_seed{i}"] = q
        for sn, f in A.strata().items():
            p[f"stratum_{sn}"] = np.asarray(f(ev, Xe, Xtrain_role), bool)
        out.append(p)
    return pd.concat(out, ignore_index=True)


def score(pred: pd.DataFrame, mode: str, sha: str) -> dict:
    res = {"mode": mode, "frozen": FROZEN, "split_sha256": sha, "sets": {}}
    for name, p in pred.groupby("set", sort=True):
        y = p["y"].to_numpy(float)
        sc = {k: p[f"s_{k}"].to_numpy() for k in MODELS}
        per = [p[f"s_gbm_challenger_seed{i}"].to_numpy() for i in range(len(FROZEN["seeds"]))]
        r = TL.score_set(app=A.APP, mode=mode, split_sha=sha, set_name=name, experiment="test_look", y=y, scores=sc,
                         groups={"huc8": p["huc8"].to_numpy()}, primary=PRIMARY, secondary=SECONDARY,
                         prob_models=("logistic", "gbm_challenger", "climatology", "location_only", "history_rate"),
                         manifests=A.DATA_MANIFESTS)
        r["gbm_per_seed_auroc"] = [C.fast_auroc(y, q) for q in per]
        r["strata"] = {}
        for c in [c for c in p.columns if c.startswith("stratum_")]:
            m = p[c].to_numpy(bool)
            r["strata"][c[8:]] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                                  "auroc": {k: C.fast_auroc(y[m], v[m]) for k, v in sc.items()} if 0 < y[m].sum() < m.sum() else None}
        res["sets"][name] = r
    res["holm_primary"] = TL.holm_primary(res["sets"], "huc8")
    res.update(TL.claim_met(res["holm_primary"], deterministic=True))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    TL.add_mode_args(ap)
    a = ap.parse_args(argv)
    mode = TL.mode_from_args(a)
    split, sha = C.load_split(A.APP, expect_sha=A.SPLIT_SHA)
    TL.verify_data(A.APP)
    claim = None
    with C.cpu_slot(f"{A.APP}_{mode}"):
        if mode == "test":
            claim = TL.authorise(A.APP, sha)
            pred = predict(mode, split, sha)
            TL.save_predictions(A.APP, sha, pred)
        elif mode == "rescore":
            pred = TL.authorise_rescore(A.APP, sha)
        else:
            pred = predict(mode, split, sha)
        res = score(pred, mode, sha)
        res["claim"] = claim
        rd = TL.look_dir(A.APP) if mode != "dryrun" else C.run_dir(A.APP, "testpath_dryrun")
        if mode == "dryrun":
            pred.to_parquet(rd / "predictions.parquet", index=False)
        C.write_json(rd / ("results.json" if mode != "rescore" else "results_rescore.json"), res)
        if mode != "dryrun":
            TL.mark_scored(A.APP, {"claim_met": res["claim_met"]})
    print(json.dumps({k: {"logistic": v["scores"]["logistic"]["auroc"], "incumbent": v["scores"]["vb_mlr_incumbent"]["auroc"]}
                      for k, v in res["sets"].items()}, indent=1))
    print(json.dumps({k: (round(v["diff"], 4) if v["diff"] is not None else None, v["holm_reject"]) for k, v in res["holm_primary"].items()}),
          "claim_met:", res["claim_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
