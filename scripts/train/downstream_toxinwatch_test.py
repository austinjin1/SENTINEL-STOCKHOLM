#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D2 ToxinWatch single test look for the it3 configuration (FREEZE_CANDIDATE_downstream_toxinwatch.json).

    python -B scripts/train/downstream_toxinwatch_test.py --dry-run-on-val      # WA val 2019-21 as "test"; nothing registered
    python -B scripts/train/downstream_toxinwatch_test.py --allow-test          # ONLY after FREEZE_downstream_toxinwatch.json exists
    python -B scripts/train/downstream_toxinwatch_test.py --rescore-from-saved  # crash recovery only (no re-fit)

Frozen (protocol §8 it3): GBM challenger (num_leaves 7, min_child 60, inner early stopping, seeds 0–2) and
logistic (C 0.01) on WA train lakes 2007–2018 + the 12,357 D2n national train rows (row count pinned); four
INLA-class incumbents (C 1.0; WA-only and pooled × v1/v2); simple baselines on WA train rows. Test sets: WA
known lakes (train-role clusters with ≥ 1 training row) 2022–2026 and WA test lakes 2022–2026.
**Governing rule (protocol §5, lead decision 2026-11-08):** GBM vs the strongest incumbent on validation
(v2_pooled) and vs the strongest simple baseline on validation (known: location_knn; new: lake history
rate), lake-cluster paired bootstrap, **Holm over the two test sets separately per comparator family**, plus
the seed rule; the Holm across all four primary comparisons is reported as a sensitivity — if only it fails,
the claim is labelled "not robust to all-comparison correction". Scum-flag breakdown reported (§5).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads()

from sentinel.models.downstream import d2n_microcystin as N  # noqa: E402
from sentinel.models.downstream import testlook as TL  # noqa: E402
from sentinel.models.downstream import toxinwatch as T  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IT3 = _load("tw_it3", "scripts/train/downstream_toxinwatch_it3.py")
D = IT3.D
FROZEN = {"incumbent_C": 1.0, "logistic_C": 0.01, "gbm": {"num_leaves": 7, "min_child_samples": 60}, "seeds": (0, 1, 2),
          "strongest_incumbent": "st_glmm_incumbent_v2_pooled",
          "strongest_simple": {"known": "location_knn", "new": "lake_history_rate"}, "n_national_train_rows": 12357}
INCS = ["st_glmm_incumbent", "st_glmm_incumbent_v2", "st_glmm_incumbent_pooled", "st_glmm_incumbent_v2_pooled"]
SIMPLE = ["persistence", "lake_history_rate", "location_knn", "climatology_month", "prevalence"]
MODELS = INCS + SIMPLE + ["logistic", "gbm_challenger"]


def wa_frame(mode: str, sha: str):
    sd, sites = T.load_sitedays()
    lake_role = T.lake_role_assignment(sd)
    split, sha_file = C.load_split(T.APP, expect_sha=T.SPLIT_SHA, path=C.PREREG / T.SPLIT_FILE)
    if split["assignments"]["lake_role"] != {str(k): v for k, v in lake_role.items()} or sha_file != sha:
        raise C.ProtectedDataError("WA lake roles differ from the frozen v2 split")
    rows = T.assign_rows(sd, lake_role)
    df = rows if mode == "test" else C.drop_protected(rows)
    df = df[df["label_determined"]].reset_index(drop=True)
    if mode != "test":
        C.assert_no_test(df)
    df["lake_role"] = df["lake_cluster"].map(lake_role).fillna("test")
    train = df[df["role"] == "train"]
    df["lake_has_train_rows"] = df["lake_cluster"].isin(set(train["lake_cluster"]))
    p0 = float(train["y"].mean())
    X, cols = T.build_features(df, T.cluster_days(df), p0=p0, iteration="it1", sitehist=T.site_days(df))
    return df, X, cols, p0


def national_train_count() -> int:
    from sentinel.models.downstream import wqp_track as WT

    split, nsha = C.load_split(N.APP, expect_sha=N.SPLIT_SHA)
    rows, _, _ = WT.build_frame(N.CFG, split)
    return int((rows["role"] == "train").sum())


def predict(mode: str, sha: str) -> pd.DataFrame:
    df, X, cols, p0 = wa_frame(mode, sha)
    nat, Xn, nat_sha = IT3.national_rows(cols, p0)
    if nat_sha != N.SPLIT_SHA:
        raise C.ProtectedDataError("D2n split sha mismatch")
    TL.check_expected(T.APP, "n_national_train_rows", int(len(nat)))
    df, X = df.assign(src_national=0.0), X.assign(src_national=0.0)
    lcols = cols + ["src_national"]
    wx_cols = [c for c in cols if c.startswith("wx_")]
    tr = (df["role"] == "train").to_numpy()
    wa_train, Xwa = df[tr].reset_index(drop=True), X[tr].reset_index(drop=True)
    pool = pd.concat([wa_train, nat], ignore_index=True)
    Xpool = pd.concat([Xwa, Xn.reindex(columns=X.columns)], ignore_index=True)
    yr = df["date"].dt.year
    if mode == "test":
        sets = {"test_known_2022_2026": ((df["lake_role"] == "train") & df["lake_has_train_rows"] & (yr >= T.TEST_FROM_YEAR)),
                "test_new_lakes_2022_2026": ((df["lake_role"] == "test") & (yr >= T.TEST_FROM_YEAR))}
    else:
        sets = {"dryrun_val_known_2019_2021": D.EVAL_SETS["val_known_2019_2021"](df),
                "dryrun_val_new_lakes_2019_2021": D.EVAL_SETS["val_new_lakes_2019_2021"](df)}
    evm = np.any([m.to_numpy() for m in sets.values()], axis=0)
    ev, Xev = df[evm], X[evm]
    preds = T.baseline_scores(wa_train, Xwa, ev, Xev)
    for train, Xtr, sfx in ((wa_train, Xwa, ""), (pool, Xpool, "_pooled")):
        lr_tr = T.knn_lake_rate(train, train, exclude_self=True)
        lr_ev = T.knn_lake_rate(train, ev, exclude_self=True)
        for v in ("v1", "v2"):
            inc = T.StGlmm(wx_cols=wx_cols, C_reg=FROZEN["incumbent_C"], variant=v).fit(train, Xtr, lr_tr, train["y"])
            preds[("st_glmm_incumbent" if v == "v1" else "st_glmm_incumbent_v2") + sfx] = inc.predict_proba(ev, Xev, lr_ev)
    preds["logistic"] = T.fit_logistic(Xpool, pool["y"], lcols, C_reg=FROZEN["logistic_C"]).predict_proba(
        Xev[lcols].to_numpy(float))[:, 1]
    g = [T.fit_gbm(Xpool, pool, lcols, seed=s, params=FROZEN["gbm"]).predict_proba(Xev[lcols])[:, 1] for s in FROZEN["seeds"]]
    preds["gbm_challenger"] = np.mean(g, axis=0)
    p = pd.DataFrame({"lake_cluster": ev["lake_cluster"].to_numpy(), "site_id": ev["site_id"].to_numpy(),
                      "date": ev["date"].to_numpy(), "y": ev["y"].to_numpy(float),
                      "scum": ev["scum_info_any"].fillna(False).astype(bool).to_numpy()})
    p["set"] = None
    for name, m in sets.items():
        p.loc[m.to_numpy()[evm], "set"] = name
    for k in MODELS:
        p[f"s_{k}"] = np.asarray(preds[k], float)
    for i, q in enumerate(g):
        p[f"s_gbm_challenger_seed{i}"] = q
    p.attrs["n_national"] = int(len(nat))
    return p


def score(pred: pd.DataFrame, mode: str, sha: str) -> dict:
    res = {"mode": mode, "frozen": FROZEN, "split_sha256": sha, "d2n_split_sha256": N.SPLIT_SHA, "sets": {}}
    for name, s in pred.groupby("set", sort=True):
        kind = "known" if "known" in name else "new"
        y = s["y"].to_numpy(float)
        sc = {k: s[f"s_{k}"].to_numpy() for k in MODELS}
        prim = {"gbm_challenger": [FROZEN["strongest_incumbent"], FROZEN["strongest_simple"][kind]]}
        sec = {"gbm_challenger": [b for b in INCS + SIMPLE[:-1] + ["logistic"] if b not in prim["gbm_challenger"]],
               "logistic": [FROZEN["strongest_incumbent"], FROZEN["strongest_simple"][kind]]}
        per = {"gbm_challenger": [s[f"s_gbm_challenger_seed{i}"].to_numpy() for i in range(len(FROZEN["seeds"]))]}
        r = TL.score_set(app=T.APP, mode=mode, split_sha=sha, set_name=name, experiment="mc_exceed8@it3/test_look",
                         y=y, scores=sc, groups={"lake": s["lake_cluster"].to_numpy()}, primary=prim, secondary=sec,
                         prob_models=tuple(INCS) + ("logistic", "gbm_challenger", "location_knn", "climatology_month",
                                                    "lake_history_rate", "prevalence"),
                         manifests=list(T.DATA_MANIFESTS) + ["wqp_discrete", "wbd_huc12"], per_seed=per)
        r["gbm_per_seed_auroc"] = [C.fast_auroc(y, q) for q in per["gbm_challenger"]]
        r["scum_breakdown"] = {}
        for st, m in (("scum", s["scum"].to_numpy(bool)), ("no_scum", ~s["scum"].to_numpy(bool))):
            r["scum_breakdown"][st] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                                       "auroc": {k: (C.fast_auroc(y[m], v[m]) if 0 < y[m].sum() < m.sum() else None) for k, v in sc.items()}}
        res["sets"][name] = r
    fam = {}
    for f, is_inc in (("incumbent", True), ("simple", False)):
        fam[f] = TL.holm_primary({s: {"paired": {k: v for k, v in r["paired"].items() if v["kind"] == "primary"
                                                 and k.endswith(FROZEN["strongest_incumbent"]) == is_inc}}
                                  for s, r in res["sets"].items()}, "lake")
    res["holm_primary"] = fam
    res["claim_families"] = {f: TL.claim_met(h) for f, h in fam.items()}
    res["claim_met"] = bool(all(c["claim_met"] for c in res["claim_families"].values()))
    res["holm_all_four_sensitivity"] = TL.holm_primary(res["sets"], "lake")
    res["claim_met_all_four_holm"] = TL.claim_met(res["holm_all_four_sensitivity"])["claim_met"]
    res["label"] = ("beats incumbent and strongest simple baseline" if res["claim_met"] and res["claim_met_all_four_holm"]
                    else "beats both, not robust to all-comparison correction" if res["claim_met"] else "no claim (competitive/negative)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    TL.add_mode_args(ap)
    a = ap.parse_args(argv)
    mode = TL.mode_from_args(a)
    sha = T.SPLIT_SHA
    TL.verify_data(T.APP)
    n_nat = national_train_count()                 # pre-claim (no WA test row is read): pinned row count
    TL.check_expected(T.APP, "n_national_train_rows", n_nat)
    claim = None
    with C.cpu_slot(f"toxinwatch_{mode}"):
        if mode == "test":
            claim = TL.authorise(T.APP, sha)
            pred = predict(mode, sha)
            TL.save_predictions(T.APP, sha, pred)
        elif mode == "rescore":
            pred = TL.authorise_rescore(T.APP, sha)
        else:
            pred = predict(mode, sha)
        res = score(pred, mode, sha)
        res["claim"] = claim
        rd = TL.look_dir(T.APP) if mode != "dryrun" else C.run_dir(T.APP, "testpath_dryrun")
        if mode == "dryrun":
            pred.to_parquet(rd / "predictions.parquet", index=False)
        C.write_json(rd / ("results.json" if mode != "rescore" else "results_rescore.json"), res)
        if mode != "dryrun":
            TL.mark_scored(T.APP, {"claim_met": res["claim_met"], "label": res["label"]})
    print(json.dumps({s: {k: r["scores"][k]["auroc"] for k in ("gbm_challenger", FROZEN["strongest_incumbent"], "location_knn",
                                                              "lake_history_rate")} for s, r in res["sets"].items()}, indent=1))
    print(json.dumps({f: {k: (round(v["p_value"], 3), v["holm_reject"], v["per_seed_diff"]) for k, v in h.items()}
                      for f, h in res["holm_primary"].items()}), "claim_met:", res["claim_met"], "|", res["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
