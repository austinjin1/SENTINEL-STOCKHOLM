#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X development selection (protocol §6–§7, amendment v4.2/v4.9). No model is fitted here.

1. Comparator variants: for every pooled comparator, the variant (same-continent / multi-source) with the higher
   development ``r2_anomaly`` per (block, target).
2. Baseline ensembles (§6.7): RF[c+g]+TabPFN, RF[c+g]+LGBM+TabPFN, RF[c+context]+TabPFN — means of the members'
   chosen-variant predictions.
3. SENTINEL ensembles (v4.9 e/f): mean(best BioSet, RF[c+g], TabPFN); mean(best BioSet, the two strongest baselines
   by mean development r2_anomaly).
4. Per-target comparator = the strongest of all comparators by development r2_anomaly; selection of C-X = highest
   mean development r2_anomaly over the block's 6 targets among the SENTINEL configurations; 2 secondaries.
   Protocol v11 item 1: two comparators are added by rule from the selected C-X — ``cxr_nobioset`` (mean of its
   baseline members) and ``cxr_swap`` (those plus the best listed single baseline that is not a member) — and
   compete for the per-target comparator. They never enter the SENTINEL configurations or their member choice.
5. Paired development comparison (location-cluster bootstrap, B = 2,000) of each selected candidate vs the
   per-target comparator; registry rows (experiment ``bioassess_x_dev``), ``x_track/dev/selection_x_v1.json``.

    python -B scripts/eval/bioassess_x_select.py [--log]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from sentinel.core.registry import file_sha256, log_result
from sentinel.models.bioassess import evaluate as EV
from sentinel.models.bioassess import xdev as XD
from sentinel.models.bioassess import xtrack as X

PREDS = XD.DEV_DIR / "preds"
FOLDS = XD.DEV_DIR / "folds_dev_v1.json"
OUT = XD.DEV_DIR / "selection_x_v1.json"
BIOSET = ("bsx_flat_geo", "bsx_flat_geo_rloss", "bsx_flat_geo_ukpt", "bsx_flat_geo_same")
BASE_ENS = {"bens_rfg_tabpfn": ("rf_community_geography", "tabpfn_community"),
            "bens_rfg_gbm_tabpfn": ("rf_community_geography", "gbm_community", "tabpfn_community"),
            "bens_rfctx_tabpfn": ("rf_community_context", "tabpfn_community")}
SENTINEL_ENS = ("ens_c_form", "ens_top2")
RULE_COMPARATORS = ("cxr_nobioset", "cxr_swap")       # protocol v11 item 1
# SOTA-campaign classes (docs/plans/sota_campaign.md): published community->chemistry inference methods vs simple /
# practitioner baselines (GJAM enters only if its v6 fidelity passed: finish.sh ingests it only then)
PUBLISHED = ("wapls2", "knn_community", "bioinfer", "gjam")
SIMPLE = ("region_mean", "rf_geography", "nearest_reference", "idx_linear", "idx_linear_geo", "idx_rf", "idx_rf_geo")


def expectation(pr: dict) -> str:
    lo, hi = pr.get("ci_low"), pr.get("ci_high")
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return "not estimable"
    return "SOTA-expected" if lo > 0 else ("negative" if hi < 0 else "competitive")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_preds(block, ids) -> dict[str, dict[str, pd.DataFrame]]:
    """model -> variant -> predictions (index = ids)."""
    out: dict = {}
    for f in sorted((PREDS / block).glob("*__s0.parquet")):
        model, variant, _ = f.stem.split("__")
        out.setdefault(model, {})[variant] = pd.read_parquet(f).reindex(ids)
    return out


def r2a(y, p, region):
    return EV.anomaly_r2(y, p, region)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    S, frozen, D = XD.load_context()
    folds = XD.read_folds(FOLDS, S, frozen)
    for b in XD.BLOCKS:
        for fo in folds[b]:
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
    result = {"folds_sha256": file_sha256(FOLDS), "blocks": {}}
    for b in XD.BLOCKS:
        T = list(XD.TARGETS[b])
        ids = pd.Index(np.concatenate([fo.test for fo in folds[b]]))
        Y = pd.DataFrame({t: XD.labels(D, b, t, ids, "same") for t in T}, index=ids)
        region = D.samples.loc[ids, "ag_eco9"].astype(str).to_numpy()
        cluster = D.samples.loc[ids, "location_id"].astype(str).to_numpy()
        P = load_preds(b, ids)
        score = lambda p, t: r2a(Y[t].to_numpy(float), np.asarray(p, float), region)  # noqa: E731
        # 1. chosen variants -> one prediction frame per comparator
        comp, choice = {}, {}
        for m, vs in P.items():
            if m.startswith(("bsx_", "ens_", "cxr_")):       # cxr_: rule comparators of an earlier run (rebuilt below)
                continue
            frame, ch = pd.DataFrame(index=ids), {}
            for t in T:
                cands = {v: df[t] for v, df in vs.items() if t in df and np.isfinite(df[t].to_numpy(float)).any()}
                if not cands:
                    continue
                v = max(cands, key=lambda v: (score(cands[v], t), v == "same"))
                frame[t], ch[t] = cands[v], v
            comp[m], choice[m] = frame, ch
        # 2. baseline ensembles
        for e, mem in BASE_ENS.items():
            if all(m in comp for m in mem):
                comp[e] = sum(comp[m][T] for m in mem) / len(mem)
                choice[e] = {t: [choice[m].get(t) for m in mem] for t in T}
        mean_r2a = {m: float(np.nanmean([score(df[t], t) for t in T if t in df])) for m, df in comp.items()}
        # 3. SENTINEL configurations
        sent = {}
        for m in BIOSET:
            if m in P:
                (v, df), = P[m].items()
                sent[m] = df[T]
        if sent:
            best_bs = max(sent, key=lambda m: np.nanmean([score(sent[m][t], t) for t in T]))
            if all(m in comp for m in ("rf_community_geography", "tabpfn_community")):
                sent["ens_c_form"] = (sent[best_bs] + comp["rf_community_geography"][T] + comp["tabpfn_community"][T]) / 3
            base_rank = sorted((m for m in comp if not m.startswith("bens_") and m not in ("bioinfer", "gjam")),
                               key=lambda m: -mean_r2a[m])
            top2 = base_rank[:2]
            sent["ens_top2"] = (sent[best_bs] + comp[top2[0]][T] + comp[top2[1]][T]) / 3
        else:
            best_bs, top2 = None, []
        for e in SENTINEL_ENS:
            if e in sent:
                f = PREDS / b / f"{e}__chosen__s0.parquet"
                sent[e].assign(fold=P[best_bs][next(iter(P[best_bs]))]["fold"]).to_parquet(f)
        for e in BASE_ENS:
            if e in comp:
                comp[e].assign(fold=next(iter(P["rf_community_geography"].values()))["fold"]).to_parquet(
                    PREDS / b / f"{e}__chosen__s0.parquet")
        sent_mean = {m: float(np.nanmean([score(df[t], t) for t in T])) for m, df in sent.items()}
        ranked = sorted(sent_mean, key=lambda m: -sent_mean[m])
        # 3b. rule comparators (protocol v11 item 1), from the selected C-X; no fit, no choice among alternatives
        rule = {}
        if ranked:
            mem = {"ens_c_form": ["rf_community_geography", "tabpfn_community"], "ens_top2": list(top2)}.get(ranked[0])
            if mem is None:
                raise ValueError(f"{b}: C-X {ranked[0]} has no baseline member; protocol v11 item 1 cannot be applied")
            singles = sorted(m for m in XD.models_for(b) if m in comp and m not in mem)
            swap = max(singles, key=lambda m: mean_r2a[m])
            rule = {"cxr_nobioset": {"members": list(mem)}, "cxr_swap": {"members": [*mem, swap], "swap": swap,
                    "swap_dev_mean_r2_anomaly": mean_r2a[swap],
                    "singles_ranked": sorted(((m, mean_r2a[m]) for m in singles), key=lambda kv: -kv[1])}}
            for e in RULE_COMPARATORS:
                mm = rule[e]["members"]
                same = [n for n, bm in BASE_ENS.items() if n in comp and sorted(bm) == sorted(mm)]
                rule[e]["same_members_as"] = same[:1]
                # the same model as a listed baseline ensemble: its frame is reused bit for bit, so the tie for the
                # per-target comparator is exact and stays with the ensemble's own (earlier listed) name
                comp[e] = comp[same[0]][T].copy() if same else sum(comp[m][T] for m in mm) / len(mm)
                choice[e] = {t: [choice[m].get(t) for m in mm] for t in T}
                mean_r2a[e] = float(np.nanmean([score(comp[e][t], t) for t in T]))
                comp[e].assign(fold=next(iter(P["rf_community_geography"].values()))["fold"]).to_parquet(
                    PREDS / b / f"{e}__chosen__s0.parquet")
            log(b, "rule comparators (v11 item 1): members", mem, "| swap member", swap)
        # 4. per-target comparator
        per_target = {}
        for t in T:
            sc = {m: score(df[t], t) for m, df in comp.items() if t in df and np.isfinite(df[t].to_numpy(float)).any()}
            best = max(sc, key=sc.get)
            per_target[t] = {"model": best, "variant": choice.get(best, {}).get(t), "dev_r2_anomaly": sc[best],
                             "ranking": sorted(sc.items(), key=lambda kv: -kv[1])}
        # 5. paired development comparison
        paired = {}
        for m in ranked[:3]:
            paired[m] = {}
            for t in T:
                ref = comp[per_target[t]["model"]][t].to_numpy(float)
                pr = EV.paired_vs(Y[t].to_numpy(float), sent[m][t].to_numpy(float), ref, region, cluster,
                                  metric="r2_anomaly", n_boot=2000, seed=0)
                paired[m][t] = {k: pr[k] for k in pr if isinstance(pr[k], (int, float))}
        # SOTA expectation of C-X per target (dev, seed 0): vs strongest published method and strongest simple baseline
        sota = {}
        if ranked:
            cxm = ranked[0]
            for t in T:
                row = {}
                for cls, pool in (("published", PUBLISHED), ("simple", SIMPLE), ("per_target_comparator", None)):
                    cand = [per_target[t]["model"]] if pool is None else \
                        [m for m in pool if m in comp and t in comp[m] and np.isfinite(comp[m][t].to_numpy(float)).any()]
                    if not cand:
                        row[cls] = {"reference": None, "expectation": "no reference available"}
                        continue
                    ref = max(cand, key=lambda m: score(comp[m][t], t))
                    pr = EV.paired_vs(Y[t].to_numpy(float), sent[cxm][t].to_numpy(float), comp[ref][t].to_numpy(float),
                                      region, cluster, metric="r2_anomaly", n_boot=2000, seed=0)
                    row[cls] = {"reference": ref, "variant": choice.get(ref, {}).get(t), "ref_dev_r2_anomaly":
                                score(comp[ref][t], t), "delta": pr["diff"], "ci_low": pr["ci_low"],
                                "ci_high": pr["ci_high"], "expectation": expectation(pr)}
                pub, sim = row["published"].get("expectation"), row["simple"].get("expectation")
                row["overall"] = ("SOTA-expected" if pub == sim == "SOTA-expected" and
                                  row["per_target_comparator"].get("expectation") == "SOTA-expected" else
                                  "negative" if "negative" in (pub, sim) else "competitive")
                sota[t] = row
        blk = {"n_pseudo_test": int(len(ids)), "best_bioset": best_bs, "ens_top2_members": top2,
               "sota_expectation_cx": sota,
               "sentinel_mean_r2_anomaly": sent_mean, "selected": ranked[:1], "secondary": ranked[1:3],
               "comparator_mean_r2_anomaly": mean_r2a, "variant_choice": choice, "per_target_comparator": per_target,
               "paired_dev_vs_comparator": paired, "rule_comparators": rule,
               "n_sentinel_configurations": len(sent)}
        result["blocks"][b] = blk
        log(b, "selected", ranked[:3], {m: round(v, 4) for m, v in sent_mean.items()})
        for t in T:
            log(" ", t, per_target[t]["model"], per_target[t]["variant"], round(per_target[t]["dev_r2_anomaly"], 4))
        if a.log and ranked:
            for m in ranked[:3]:
                for t in T:
                    pr = paired[m][t]
                    log_result(workstream="bioassess", experiment="bioassess_x_dev", model=f"{m} - comparator",
                               metric="delta_r2_anomaly_vs_comparator", value=float(pr.get("diff", np.nan)),
                               split=f"{b}_dev", target=t, ci_low=pr.get("ci_low"), ci_high=pr.get("ci_high"),
                               ci_method="paired location-cluster bootstrap B=2000", n=int(len(ids)),
                               config={"candidate": m, "comparator": per_target[t]["model"],
                                       "comparator_variant": per_target[t]["variant"]},
                               notes="M5-X development selection (seed 0); informative only, not a test; "
                                     "protocol v11 re-run: rule comparators cxr_nobioset / cxr_swap added to the "
                                     "comparator list, no model fitted, C-X unchanged",
                               extra={"selected": m == ranked[0], "p": pr.get("p_value")})
    result["n_configurations"] = {b: result["blocks"][b]["n_sentinel_configurations"] for b in XD.BLOCKS}
    OUT.write_text(json.dumps(result, indent=1, default=float))
    log("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
