#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Aggregate the ChemTox search shards, pick one configuration per model, build the ensembles.

Reads every ``shard*_results.json`` written by ``scripts/train/chemtox_search.py``, selects the
best configuration per model by **mean RMSE over the official CV folds**, logs that configuration's
CV metrics to the registry (with the full search budget in ``extra.n_configurations``), and then
scores the mean and leave-one-fold-out NNLS stack of the per-model best out-of-fold predictions,
reporting the components' individual scores alongside — never a stack presented as a single model.

Also re-derives the baseline out-of-fold predictions (XGBoost / random forest on the published
feature set) so the ensemble has the strong simple components in it.

    python -B scripts/eval/chemtox_select.py --models chemprop tabpfn embeddings --log-registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import package_hash
from sentinel.data.sources import adore as A
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import ensemble as EN
from sentinel.models.chemtox import spaces
from sentinel.models.chemtox import xgb as XG
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_select")
EXPERIMENT = "adore_chemtox_search"
BASELINE_COMPONENTS = {
    "xgboost": ({"fingerprint": "MACCS"}, {"n_estimators": 800, "eta": 0.03, "max_depth": 6,
                                            "min_child_weight": 1, "subsample": 0.8}),
    "rf": ({"fingerprint": "MACCS"}, {"n_estimators": 300, "max_depth": 20, "max_features": 0.3}),
}


def search_dir(model: str, challenge: str, split: str) -> Path:
    return paths.RUNS_ROOT / "chemtox" / "search" / f"{model}_{challenge}_{split}"


def load_shards(model: str, challenge: str, split: str) -> tuple[list[dict], Path]:
    d = search_dir(model, challenge, split)
    records: list[dict] = []
    for f in sorted(d.glob("shard*_results.json")):
        payload = json.loads(f.read_text())
        records.extend(payload.get("records", []))
    return records, d


def full_cv_records(records: list[dict]) -> list[dict]:
    """Only configurations scored on all five official folds (screening runs are excluded).

    Screening (a fold subset with capped epochs/rows) narrows the field; selection uses the
    finalists' full 5-fold scores, so a model is never chosen on a cheaper proxy.
    """
    out, seen = [], set()
    for r in records:
        if "cv_mean" in r and not r.get("screen") and len(r.get("fold_metrics", [])) >= 5:
            if r["index"] in seen:  # a resumed run re-lists earlier records; count each config once
                continue
            seen.add(r["index"])
            out.append(r)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["chemprop", "tabpfn", "embeddings"])
    ap.add_argument("--challenge", default="t-F2F")
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar")
    ap.add_argument("--budget", type=int, default=spaces.DEFAULT_BUDGET)
    ap.add_argument("--log-registry", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--freeze", action="store_true",
                    help="write the frozen configuration block and its hashes (pre-registration)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seeds the frozen configurations will be scored with on test")
    return ap.parse_args(argv)


def frozen_block(summary: dict, ens: dict, *, seeds: list[int]) -> dict:
    """One frozen configuration per model, plus the ensemble definitions, for the pre-registration.

    Attached data frames are not part of a configuration's identity (they are rebuilt from the
    challenge files and the exclusion rule), so the parameter dictionaries carry their *labels*
    (``init``, ``multitask``) and the loader rebuilds the frames at scoring time.
    """
    frozen: dict[str, dict] = {}
    for model, s in summary.items():
        if "cv_mean" not in s:
            continue
        features = s.get("best_features") or BASELINE_COMPONENTS.get(model, ({}, {}))[0]
        params = s.get("best_params") or BASELINE_COMPONENTS.get(model, ({}, {}))[1]
        params = {k: v for k, v in params.items() if not str(v).startswith("<DataFrame")}
        frozen[model] = {"features": dict(features), "params": dict(params), "seeds": list(seeds),
                         "cv_rmse": s["cv_mean"]["rmse"],
                         "n_configurations": s.get("n_configurations")}
    for name, r in ens.items():
        frozen[name] = {"ensemble": True, "components": list(r["components"]),
                        "method": "unweighted mean" if "mean" in name
                        else "non-negative least squares refitted on training-portion "
                             "out-of-fold predictions",
                        "cv_rmse": r["metrics"]["rmse"], "seeds": list(seeds)}
    return frozen


def frozen_hash(frozen: dict) -> str:
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()[:16]


def script_hash(path: Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"


def baseline_oof(ds: ChemToxDataset, name: str) -> np.ndarray:
    """Out-of-fold predictions of a baseline component on the official folds."""
    features, params = BASELINE_COMPONENTS[name]
    spec = FeatureSpec(**features)
    maker = XG.make_xgb(0) if name == "xgboost" else XG.make_rf(0)
    r = CV.run_cv(ds, spec, maker, params, model_name=name)
    return r.oof, r.mean


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out) if args.out else paths.run_dir(
        "chemtox", f"select/{args.challenge}_{args.split}")

    ds = ChemToxDataset.load(args.challenge, args.split, target=args.target,
                             specs=FeatureSpec(fingerprint="MACCS"))
    ds.assert_no_chemical_leakage(log=log)
    y, chem, taxon = ds.y, ds.chem, ds.taxon
    folds = ds.assignment.where(ds.assignment.str.fullmatch(r"\d+").fillna(False))

    summary: dict[str, dict] = {}
    oof: dict[str, np.ndarray] = {}

    for model in args.models:
        records, d = load_shards(model, args.challenge, args.split)
        ok = full_cv_records(records)
        screened = list({r["index"]: r for r in records if r.get("screen") and "cv_mean" in r}.values())
        failed = [r for r in records if "error" in r]
        if not ok:
            log.warning("%s: no completed configurations in %s", model, d)
            continue
        best = min(ok, key=lambda r: r["cv_mean"]["rmse"])
        log.info("%s: %d screened + %d full 5-fold runs (%d failed); best config %d CV RMSE %.4f",
                 model, len(screened), len(ok), len(failed), best["index"], best["cv_mean"]["rmse"])
        p = d / f"oof_{model}_{best['index']:03d}.npy"
        if p.exists():
            oof[model] = np.load(p)
        summary[model] = {
            "n_configurations": args.budget,
            "n_screened": len(screened),
            "n_full_cv": len(ok),
            "n_completed": len(ok),
            "n_failed": len(failed),
            "best_index": best["index"],
            "best_features": best["features"],
            "best_params": best["params"],
            "cv_mean": best["cv_mean"],
            "cv_sd": best["cv_sd"],
            "pooled": best["pooled"],
            "all_cv_rmse": sorted(r["cv_mean"]["rmse"] for r in ok),
            "failures": [r.get("error") for r in failed],
        }

    for name in BASELINE_COMPONENTS:
        pred, mean = baseline_oof(ds, name)
        oof[name] = pred
        summary[name] = {"n_configurations": 32 if name == "xgboost" else 8,
                         "note": "published-baseline component re-run for the ensemble",
                         "cv_mean": mean}
        log.info("%s component CV RMSE %.4f", name, mean["rmse"])

    ens: dict[str, dict] = {}
    if len(oof) >= 2:
        avg = EN.average_ensemble(oof, y, chem=chem, taxon=taxon)
        stk = EN.stacked_ensemble(oof, y, chem=chem, taxon=taxon, folds=folds.to_numpy())
        for r in (avg, stk):
            ens[r.name] = r.to_dict()
            log.info("%s: CV RMSE %.4f over %d rows (components: %s)", r.name,
                     r.metrics["rmse"], int(r.mask.sum()),
                     {k: round(v["rmse"], 4) for k, v in r.component_metrics.items()})
        np.save(out_dir / "oof_ensemble_mean.npy", avg.pred)
        np.save(out_dir / "oof_ensemble_stack.npy", stk.pred)

    payload = {"challenge": args.challenge, "split": args.split, "target": args.target,
               "models": summary, "ensembles": ens}
    if args.freeze:
        payload["frozen"] = frozen_block(summary, ens, seeds=args.seeds)
        payload["frozen_hash"] = frozen_hash(payload["frozen"])
        payload["code_hashes"] = {
            "sentinel_package": package_hash(),
            "chemtox_score_test.py": script_hash(
                Path(__file__).resolve().parent / "chemtox_score_test.py"),
            "chemtox_select.py": script_hash(Path(__file__).resolve()),
        }
        log.info("frozen-configuration hash: %s", payload["frozen_hash"])
        for k, v in payload["code_hashes"].items():
            log.info("  %-26s %s", k, v)
    CV.write_json(out_dir / "selection.json", payload)
    log.info("wrote %s", out_dir / "selection.json")

    if args.log_registry:
        for model, s in summary.items():
            if "cv_mean" not in s or model in BASELINE_COMPONENTS:
                continue
            CV.log_metrics(ds, experiment=EXPERIMENT, model=model, kind="cv", metrics=s["cv_mean"],
                           config={"challenge": args.challenge, "split": args.split,
                                   "model": model, "features": s["best_features"],
                                   "params": s["best_params"],
                                   "selection": "best of the equal-budget random search, "
                                                "mean RMSE over official CV folds 0-4"},
                           n=int(s["pooled"]["n"]), is_baseline=False,
                           notes="improvement-cycle search winner; CV only, test rows untouched",
                           extra={"n_configurations": s["n_configurations"],
                                  "n_completed": s["n_completed"], "cv_sd": s["cv_sd"],
                                  "pooled_out_of_fold": s["pooled"],
                                  "best_index": s["best_index"]},
                           artifacts=[str(out_dir / "selection.json")])
        for name, r in ens.items():
            CV.log_metrics(ds, experiment=EXPERIMENT, model=name, kind="cv", metrics=r["metrics"],
                           config={"challenge": args.challenge, "split": args.split,
                                   "components": r["components"],
                                   "method": "unweighted mean" if "mean" in name
                                   else "leave-one-fold-out non-negative least squares"},
                           n=r["n_scored"], is_baseline=False,
                           notes=("ENSEMBLE of " + ", ".join(r["components"]) +
                                  "; component CV scores in extra.component_metrics"),
                           extra={"component_metrics": r["component_metrics"],
                                  "weights": r["weights"], "n_configurations": "see components"},
                           artifacts=[str(out_dir / "selection.json")])
        log.info("registry updated")

    table = pd.DataFrame({k: v.get("cv_mean", {}) for k, v in summary.items()}).T
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print("\n=== CV (official folds 0-4) ===")
        print(table)
        if ens:
            print("\n=== ensembles (CV) ===")
            print(pd.DataFrame({k: v["metrics"] for k, v in ens.items()}).T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
