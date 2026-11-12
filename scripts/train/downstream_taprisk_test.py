#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk single test look (protocol downstream_taprisk.md §10 test-look spec + §11 corrections).

    python -B scripts/train/downstream_taprisk_test.py --dry-run-on-val      # val origin 2016 as "test"; nothing registered
    python -B scripts/train/downstream_taprisk_test.py --allow-test          # ONLY with a lead-written FREEZE (status FROZEN)
    python -B scripts/train/downstream_taprisk_test.py --rescore-from-saved  # only from the registry-sealed predictions

Order (hardened, ``taprisk_look``): pre-claim checks (lead FREEZE, D4 import-closure hash, pinned paths, data +
environment pins, no unpinned files) → claim (marker + ``test_claim`` registry row) → test rows (only in the claiming
process) → predictions sealed (0444, SHA256SUMS, ``test_sealed`` row; access closed) → scoring rows → SCORED.
Any BaseException before sealing → ABORTED + ``test_aborted`` row. Frozen: challenger by the re-applied it2 rule,
comparators Pennino-form RF / history RF / history RF + rich (+ lat/lon) / strongest simple; D4a primary (Holm over 8 by
HUC8, and the ``test_new`` comparisons also Holm-rejected under a state-cluster bootstrap) + per-seed rule (only the GBM
seed varies); D4b and the numeric track (log10 HAA5, TOC, bromide with 90 % PIs) secondary. Fit rows: train + val,
labels censored at the scored origin (C6).
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

from sentinel.models.downstream import taprisk as TR  # noqa: E402
from sentinel.models.downstream import taprisk_it2 as T2  # noqa: E402
from sentinel.models.downstream import taprisk_look as LK  # noqa: E402
from sentinel.models.downstream import taprisk_numeric as TN  # noqa: E402
from sentinel.models.downstream import testlook as TL  # noqa: E402

TASKS = ("d4a_gw_nitrate", "d4b_sw_dbp")
CHALLENGER = "stack_rf_gbm"          # re-applied it2 rule (protocol §11; it2c results)
FROZEN_P1 = {  # phase-1 selections, hard-coded (C7): no artifact is read after the claim
    "d4a_gw_nitrate": {"selection": {"gbm_challenger": {"params": {"num_leaves": 15, "min_child_samples": 20}, "n_estimators": 88},
                                     "pennino_rf": {"topk": 50}, "pennino_rf_hist": {"topk": 50}}},
    "d4b_sw_dbp": {"selection": {"gbm_challenger": {"params": {"num_leaves": 15, "min_child_samples": 20}, "n_estimators": 75},
                                 "pennino_rf": {"topk": None}, "pennino_rf_hist": {"topk": None}}},
}
SIMPLE = {"d4a_gw_nitrate": {"known": "location_knn25", "new": "persistence"},
          "d4b_sw_dbp": {"known": "persistence", "new": "persistence"}}
COMPS = ["pennino_rf", "pennino_rf_hist", "pennino_rf_hist_rich"]
MODELS = ["stack_rf_gbm", "gbm_rich", "gbm_phase1", "pennino_rf", "pennino_rf_hist", "pennino_rf_hist_rich", "location_knn25",
          "persistence", "state_rate", "ambient_rule", "size_source_logit"]
SEED_CANDS = ("stack_rf_gbm", "gbm_rich", "gbm_phase1")


def predict(mode: str, split: dict, sha: str) -> pd.DataFrame:
    test = mode == "test"
    S = 2019 if test else 2016
    fit_roles = ("train", "val") if test else ("train",)
    out = []
    for task in TASKS:
        df = T2.build_rich(task, split, include_test=test, split_sha=sha, scored_origin=S)
        scorer, _ = T2.fit_all(task, df, fit_roles, FROZEN_P1[task])
        sets = {"test_known": "test_known", "test_new": "test_new"} if test else {"dryrun_val_known": "val_known", "dryrun_val_new": "val_new"}
        for name, es in sets.items():
            s = df[df["evalset"] == es].reset_index(drop=True)
            st, per = scorer(s)
            p = s[["pwsid", "state_code", "huc8", "origin", "y"]].copy()
            p["task"], p["set"] = task, name
            for k in MODELS:
                p[f"s_{k}"] = np.asarray(st[k], float)
            for k in SEED_CANDS:
                for i, q in enumerate(per[k]):
                    p[f"s_{k}_seed{i}"] = q
            out.append(p)
    # numeric track (secondary): fit on train (+ val in the look) states, score val (dry run) or test states
    ndf = TN.build(split, include_test=test, split_sha=sha)
    for t in TN.TARGETS:
        q = TN.fit_predict(ndf, fit_roles if not test else ("train", "val"), "test" if test else "val", t)
        q = q.rename(columns={"y": "y_num"})
        q["task"], q["set"] = f"numeric_{t}", ("test_states" if test else "dryrun_val_states")
        out.append(q)
    return pd.concat(out, ignore_index=True)


def score(pred: pd.DataFrame, mode: str, sha: str) -> dict:
    res = {"mode": mode, "split_sha256": sha, "challenger": CHALLENGER, "frozen_p1": FROZEN_P1, "sets": {}, "numeric": {}}
    cls = pred[~pred["task"].str.startswith("numeric_")]
    for (task, name), s in cls.groupby(["task", "set"], sort=True):
        kind = "known" if name.endswith("known") else "new"
        comp = COMPS + [SIMPLE[task][kind]]
        y = s["y"].to_numpy(float)
        sc = {k: s[f"s_{k}"].to_numpy() for k in MODELS}
        primary = {CHALLENGER: comp} if task == "d4a_gw_nitrate" else {}
        secondary = {k: comp for k in SEED_CANDS if k != CHALLENGER or not primary}
        per = {k: [s[f"s_{k}_seed{i}"].to_numpy() for i in range(3)] for k in SEED_CANDS}
        groups = {"huc8": s["huc8"].astype(str).to_numpy()}
        if task == "d4a_gw_nitrate" and kind == "new":
            groups["state"] = s["state_code"].astype(str).to_numpy()       # W1: stricter state-cluster unit
        set_name = f"{name}_{task}"
        r = TL.score_set(app=TR.APP, mode=mode, split_sha=sha, set_name=set_name, experiment="test_look", y=y, scores=sc,
                         groups=groups, primary=primary, secondary=secondary,
                         prob_models=tuple(m for m in MODELS if m not in ("persistence", "ambient_rule")),
                         manifests=TR.DATA_MANIFESTS, per_seed=per)
        from sentinel.models.downstream import taprisk_models as TM

        r["pennino_metrics"] = {k: TM.sens_spec(y, sc[k]) for k in COMPS}
        res["sets"][set_name] = r
    prim = {k: v for k, v in res["sets"].items() if k.endswith("d4a_gw_nitrate")}
    res["holm_primary"] = TL.holm_primary(prim, "huc8")
    new_only = {k: v for k, v in prim.items() if "new" in k}
    res["holm_test_new_state_cluster"] = TL.holm_primary(new_only, "state") if new_only else {}
    c1 = TL.claim_met(res["holm_primary"])
    c2 = TL.claim_met(res["holm_test_new_state_cluster"]) if new_only else {"claim_met": False}
    res.update({"holm_ok": c1["holm_ok"], "seed_rule_ok": c1["seed_rule_ok"], "state_cluster_ok": c2["claim_met"],
                "claim_met": bool(c1["claim_met"] and c2["claim_met"]),
                "seed_rule": "only the GBM seed varies (RF level-0 and meta use seed 0)"})
    num = pred[pred["task"].str.startswith("numeric_")]
    for task, q in num.groupby("task"):
        r = TN.score(q.drop(columns=["y"], errors="ignore").rename(columns={"y_num": "y"}))
        res["numeric"][task] = r
        if mode in ("test", "rescore"):
            for m, v in r["models"].items():
                for met in ("rmse", "r2"):
                    TL.log_test_metric(app=TR.APP, split_sha=sha, experiment="test_look", model=f"{task}/{m}", metric=met,
                                       value=v[met], split="test_states_numeric", n=v["n"],
                                       ci=dict(zip(("ci_low", "ci_high"), v[f"{met}_ci"])), ci_method="cluster bootstrap over HUC8, B=2000",
                                       data_manifests=["labels_ucmr"], extra={"secondary": True, "descriptive": True})
            TL.log_test_metric(app=TR.APP, split_sha=sha, experiment="test_look", model=f"{task}/qgbm_pi90", metric="coverage",
                               value=r["pi90"]["coverage"], split="test_states_numeric", n=r["n"],
                               ci=dict(zip(("ci_low", "ci_high"), r["pi90"]["coverage_ci"])), ci_method="cluster bootstrap over HUC8, B=2000",
                               data_manifests=["labels_ucmr"], extra={"mean_width_log10": r["pi90"]["mean_width_log10"]})
    res["label_if_not_met"] = "competitive"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    TL.add_mode_args(ap)
    a = ap.parse_args(argv)
    mode = TL.mode_from_args(a)
    split, sha = C.load_split(TR.APP, expect_sha=TR.SPLIT_SHA)
    if mode == "dryrun":
        TL.verify_data(TR.APP)
        LK.verify_pinned_folders()
    elif mode == "test":
        LK.pre_claim(sha)
    # rescore: LK.rescore_load re-verifies the whole frozen state (E4)
    claim = None
    with C.cpu_slot(f"taprisk_{mode}"):
        if mode == "test":
            try:                            # E3: the claim row is written inside the handler, not before it
                claim = LK.claim(sha)
                pred = predict(mode, split, sha)
                LK.seal(sha, pred)
            except BaseException as e:      # noqa: BLE001 — every failure before sealing closes the claim
                LK.abort(sha, repr(e))
                raise
        elif mode == "rescore":
            pred = LK.rescore_load(sha)
        else:
            pred = predict(mode, split, sha)
        res = score(pred, mode, sha)
        res["claim"] = claim
        if mode == "dryrun":
            rd = C.run_dir(TR.APP, "testpath_dryrun")
            pred.to_parquet(rd / "predictions.parquet", index=False)
            C.write_json(rd / "results.json", res)
        else:
            C.write_json(TL.look_dir(TR.APP) / ("results.json" if mode == "test" else "results_rescore.json"), res)
            LK.mark_scored({"claim_met": res["claim_met"]})
    print(json.dumps({k: {m: round(v["scores"][m]["auroc"], 4) for m in (CHALLENGER, "pennino_rf", "pennino_rf_hist",
                                                                      "pennino_rf_hist_rich")} for k, v in res["sets"].items()}, indent=1))
    print(json.dumps({k: {"r2": round(v["models"]["qgbm_med"]["r2"], 3), "pi90_cov": round(v["pi90"]["coverage"], 3)}
                      for k, v in res["numeric"].items()}))
    print(json.dumps({k: (round(v["diff"], 4), v["holm_reject"]) for k, v in res["holm_primary"].items()}), "claim_met:", res["claim_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
