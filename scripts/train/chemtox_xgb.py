#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Reproduce the published ADORE baseline (Gasser et al. 2024) with XGBoost / random forest.

Protocol (RESEARCH_PLAN §5.4, docs/research/04 §D.3):
  * data: one ADORE challenge file, official split column (default ``t-F2F`` / ``split_occurrence``);
  * features: informative fingerprint bits (SD over unique training chemicals > 0.1) + the 37
    experimental/chemical/taxonomic columns — 178 columns for MACCS/occurrence, matching the
    published Table 1 (141 + 37);
  * selection: grid search scored by mean RMSE over ADORE's own CV folds 0-4; every configuration
    is counted and written to the run's JSON;
  * test: the official test rows are scored **once**, for the reproduced published baseline only
    (``--score-test``), with chemical-cluster bootstrap CIs.

All metrics go to the registry through ``sentinel.core.registry.log_result`` with
``is_baseline=True`` (these are reproductions of a published baseline, not SENTINEL results).

    python -B scripts/train/chemtox_xgb.py --model xgboost --fingerprint MACCS --score-test
    python -B scripts/train/chemtox_xgb.py --model rf --fingerprint ToxPrint --score-test
    python -B scripts/train/chemtox_xgb.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.runlog import TrainingRun
from sentinel.data.sources import adore as A
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import evaluate as E
from sentinel.models.chemtox import xgb as XG
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_xgb")
EXPERIMENT = "adore_published_baseline"
PUBLISHED = {  # Gasser et al. 2024, t-F2F, molar LC50, split_occurrence, test rows
    "rmse": {"value": 0.927, "model": "XGBoost + MACCS"},
    "rmse_macro": {"value": 0.904, "model": "random forest + MACCS"},
    "rmse_taxon": {"value": 0.938, "model": "XGBoost + MACCS"},
    "rmse_chemical": {"value": 0.845, "model": "random forest + ToxPrint"},
}
TOLERANCE = 0.03  # RESEARCH_PLAN §15.2 H-M4: reproduce within +-0.03 of 0.927


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge", default="t-F2F", choices=A.CHALLENGES)
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar", choices=("molar", "mass"))
    ap.add_argument("--model", default="xgboost", choices=("xgboost", "rf"))
    ap.add_argument("--fingerprint", default="MACCS", choices=("MACCS", "PubChem", "Morgan", "ToxPrint"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-test", action="store_true",
                    help="after selection, fit on the whole training partition and score the "
                         "official test rows once (allowed for the published-baseline reproduction)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="tiny grid, one fold, no registry writes")
    ap.add_argument("--no-registry", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_name = f"{args.model}_{args.fingerprint}_{args.challenge}_{args.split}"
    out_dir = paths.run_dir("chemtox", f"published_baseline/{run_name}")
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    spec = FeatureSpec(fingerprint=args.fingerprint, published=True)
    ds = ChemToxDataset.load(args.challenge, args.split, target=args.target, specs=spec)
    ds.assert_no_chemical_leakage()
    report = ds.split_report()
    log.info("%s/%s: %d train / %d test rows, chemical overlap %d",
             args.challenge, args.split, report["n_train"], report["n_test"], report["chem_overlap"])
    CV.write_json(out_dir / "split_report.json", report)

    make = XG.make_xgb(args.seed) if args.model == "xgboost" else XG.make_rf(args.seed)
    if args.smoke:
        grid = XG.XGB_GRID_SMOKE if args.model == "xgboost" else {"n_estimators": (20,), "max_depth": (4,)}
    else:
        grid = XG.XGB_GRID if args.model == "xgboost" else XG.RF_GRID

    model_name = f"gasser2024_{args.model}_{args.fingerprint.lower()}"
    config = {
        "challenge": args.challenge, "split": args.split, "target": args.target,
        "features": spec.to_dict(), "model": args.model, "grid": {k: list(v) for k, v in grid.items()},
        "seed": args.seed, "selection": "mean RMSE over ADORE official CV folds 0-4",
    }
    with TrainingRun("chemtox", f"{EXPERIMENT}:{run_name}", config=config,
                     data=["adore_raw", "adore"], plan_ref="RESEARCH_PLAN §5.4 / research 04 §D.3",
                     artifacts_dir=str(out_dir)) as run:
        search = CV.grid_search(ds, [spec], make, grid, model_name=model_name, log=log)
        best = search.best
        log.info("best of %d configurations: %s -> CV RMSE %.4f (fold SD %.4f)",
                 search.n_configurations, best.params, best.rmse, best.sd.get("rmse", float("nan")))
        run.metric("cv_rmse", best.rmse)
        run.metric("n_configurations", search.n_configurations)
        n_features = best.fold_metrics[0]["n_features"]
        run.note(f"{n_features} feature columns ({args.fingerprint} informative bits + published 37)")
        search.table().to_csv(out_dir / "cv_configurations.csv", index=False)
        CV.write_json(out_dir / "cv_results.json", {
            "n_configurations": search.n_configurations,
            "best": best.to_dict(),
            "all": [r.to_dict() for r in search.all_results],
        })

        extra_cv = {
            "n_configurations": search.n_configurations,
            "fold_metrics": best.fold_metrics,
            "fold_sd": best.sd,
            "pooled_out_of_fold": best.pooled,
            "n_features": n_features,
            "split_report": {k: report[k] for k in ("n_train", "n_test", "chem_overlap",
                                                    "n_chem_train", "n_chem_test")},
        }
        if not (args.smoke or args.no_registry):
            ids = CV.log_metrics(ds, experiment=EXPERIMENT, model=model_name, kind="cv",
                                 metrics=best.mean, config=config | {"best_params": best.params},
                                 n=int(best.pooled["n"]), is_baseline=True, extra=extra_cv,
                                 notes="published-baseline reproduction, selection on official CV folds",
                                 artifacts=[str(out_dir / "cv_results.json")])
            CV.write_json(out_dir / "registry_ids_cv.json", {"cv": ids})

        if not args.score_test:
            log.info("selection only; test rows untouched")
            return 0

        # ---- single test scoring of the reproduced published baseline ----
        trn = np.flatnonzero(ds.train_mask())
        tst = np.flatnonzero(ds.test_mask())
        Xtr, Xte, builder = ds.matrices(spec, trn, tst)
        model = make(best.params)
        model.fit(Xtr, ds.y[trn], train_rows=ds.rows(trn), valid_rows=ds.rows(tst),
                  feature_names=builder.names_)
        pred = np.asarray(model.predict(Xte, rows=ds.rows(tst)), dtype=float)
        y, chem, taxon = ds.y[tst], ds.chem[tst], ds.taxon[tst]
        test_metrics = E.report(y, pred, chem=chem, taxon=taxon)
        cis = E.bootstrap_report(y, pred, chem=chem, taxon=taxon, n_boot=args.n_boot, seed=args.seed)
        log.info("TEST %s: %s", model_name,
                 {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})
        for k, pub in PUBLISHED.items():
            if k in test_metrics:
                d = test_metrics[k] - pub["value"]
                log.info("  %s: ours %.4f vs published %.3f (%s) -> delta %+.4f%s", k, test_metrics[k],
                         pub["value"], pub["model"], d,
                         "  WITHIN +-0.03" if abs(d) <= TOLERANCE else "  OUTSIDE +-0.03")
        run.metric("test_rmse", test_metrics["rmse"])
        np.save(out_dir / "test_predictions.npy", pred)
        CV.write_json(out_dir / "test_results.json", {
            "metrics": test_metrics, "cis": cis, "published": PUBLISHED,
            "best_params": best.params, "n_configurations": search.n_configurations,
            "n_features": builder.n_features,
            "per_taxon": E.per_taxon_table(y, pred, taxon, min_n=20).to_dict("records"),
        })
        if not (args.smoke or args.no_registry):
            ids = CV.log_metrics(
                ds, experiment=EXPERIMENT, model=model_name, kind="test", metrics=test_metrics,
                config=config | {"best_params": best.params}, n=int(test_metrics["n"]), cis=cis,
                is_baseline=True,
                notes=("reproduction of Gasser et al. 2024 (DOI 10.1039/D4VA00072B); published "
                       "RMSE 0.927 XGBoost+MACCS, RMSE_M 0.904 RF+MACCS, RMSE_T 0.938 XGBoost+MACCS, "
                       "RMSE_C 0.845 RF+ToxPrint"),
                extra={"published": PUBLISHED, "n_configurations": search.n_configurations,
                       "best_params": best.params, "n_features": builder.n_features,
                       "deltas_vs_published": {k: test_metrics[k] - v["value"]
                                               for k, v in PUBLISHED.items() if k in test_metrics}},
                artifacts=[str(out_dir / "test_results.json")],
            )
            CV.write_json(out_dir / "registry_ids_test.json", {"test": ids})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
