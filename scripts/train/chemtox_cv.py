#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Cross-validate one SENTINEL-ChemTox model on ADORE's official folds (no test scoring).

Phase 1 runs **one default configuration per model family** — deep models are not tuned yet
(RESEARCH_PLAN §15.5 keeps the iteration budget for later, and §R6 requires the configuration count
to be reported). Every configuration evaluated is counted, written to the run directory and logged
in the registry entry's ``extra.n_configurations``.

Models: ``xgboost``, ``rf``, ``tabpfn``, ``chemprop``, ``chemprop_chemeleon``, ``embeddings``.
Metrics go to the registry with split ``'<challenge>/<split>/cv'``; the official ``test`` rows are
never read by this script.

    python -B scripts/train/chemtox_cv.py --model tabpfn
    python -B -m sentinel.core.gpu --name chemtox_chemprop -- \
        python -B scripts/train/chemtox_cv.py --model chemprop --accelerator gpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.runlog import TrainingRun
from sentinel.data.sources import adore as A
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import xgb as XG
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_cv")
EXPERIMENT = "adore_chemtox_cv"

#: one default configuration per family (feature spec, hyper-parameters, extra columns needed)
MODELS = {
    "xgboost": {
        "spec": {"fingerprint": "MACCS"},
        "params": {"n_estimators": 800, "eta": 0.03, "max_depth": 6, "subsample": 0.8},
        "extra_columns": (),
    },
    "rf": {
        "spec": {"fingerprint": "MACCS"},
        "params": {"n_estimators": 300, "max_depth": 20, "max_features": 0.3},
        "extra_columns": (),
    },
    "tabpfn": {
        "spec": {"fingerprint": "MACCS", "rdkit_descriptors": True},
        "params": {"n_estimators": 4, "max_train_rows": 6000, "max_features": 300, "predict_chunk": 2000},
        "extra_columns": ("chem_rdkit_can_smiles",),
    },
    "chemprop": {
        "spec": {"fingerprint": None},
        "params": {"max_epochs": 20, "batch_size": 64, "message_hidden": 300, "depth": 3,
                   "ffn_hidden": 300, "ffn_layers": 2, "species_dim": 16, "dropout": 0.0},
        "extra_columns": ("chem_rdkit_can_smiles",),
    },
    "chemprop_chemeleon": {
        "spec": {"fingerprint": None},
        "params": {"max_epochs": 20, "batch_size": 64, "ffn_hidden": 300, "ffn_layers": 2,
                   "species_dim": 16, "dropout": 0.0, "from_foundation": "CheMeleon"},
        "extra_columns": ("chem_rdkit_can_smiles",),
    },
    "embeddings": {
        "spec": {"fingerprint": None},
        "params": {"encoder": "ChemBERTa-2", "pooling": "cls", "n_estimators": 800, "eta": 0.03,
                   "max_depth": 6, "subsample": 0.8},
        "extra_columns": ("chem_rdkit_can_smiles",),
    },
}


def make_model(name: str, seed: int, accelerator: str):
    if name == "xgboost":
        return XG.make_xgb(seed)
    if name == "rf":
        return XG.make_rf(seed)
    if name == "tabpfn":
        from sentinel.models.chemtox.tabpfn_model import make_tabpfn

        return make_tabpfn(seed)
    if name.startswith("chemprop"):
        from sentinel.models.chemtox.chemprop_model import make_chemprop

        return make_chemprop(seed)
    if name == "embeddings":
        from sentinel.models.chemtox.embeddings import make_embedding_gbm

        return make_embedding_gbm(seed)
    raise KeyError(name)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--challenge", default="t-F2F", choices=A.CHALLENGES)
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar", choices=("molar", "mass"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds 0-4")
    ap.add_argument("--accelerator", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--no-registry", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfgdef = MODELS[args.model]
    spec = FeatureSpec(**cfgdef["spec"])
    params = dict(cfgdef["params"])
    if args.model.startswith("chemprop"):
        params["accelerator"] = args.accelerator
        if args.max_epochs:
            params["max_epochs"] = args.max_epochs
    if args.model == "tabpfn":
        params["device"] = "cuda" if args.accelerator == "gpu" else "cpu"
    if args.model == "embeddings":
        params["device"] = "cuda" if args.accelerator == "gpu" else "cpu"

    run_name = f"{args.model}_{args.challenge}_{args.split}"
    out_dir = paths.run_dir("chemtox", f"cv/{run_name}")
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    ds = ChemToxDataset.load(args.challenge, args.split, target=args.target, specs=spec,
                             extra_columns=cfgdef["extra_columns"])
    ds.assert_no_chemical_leakage()
    report = ds.split_report()
    CV.write_json(out_dir / "split_report.json", report)
    log.info("%s/%s: %d train / %d test rows, chemical overlap %d; folds %s", args.challenge,
             args.split, report["n_train"], report["n_test"], report["chem_overlap"], report["folds"])

    config = {"challenge": args.challenge, "split": args.split, "target": args.target,
              "model": args.model, "features": spec.to_dict(), "params": params, "seed": args.seed,
              "folds": args.folds or report["folds"],
              "selection": "single default configuration (phase 1: no tuning)"}
    with TrainingRun("chemtox", f"{EXPERIMENT}:{run_name}", config=config,
                     data=["adore_raw", "adore"], plan_ref="RESEARCH_PLAN §5.4 / research 04 §D.3",
                     artifacts_dir=str(out_dir)) as run:
        r = CV.run_cv(ds, spec, make_model(args.model, args.seed, args.accelerator), params,
                      model_name=args.model, folds=args.folds)
        log.info("CV (%d folds) %s: %s", len(r.fold_metrics), args.model,
                 {k: round(v, 4) for k, v in r.mean.items()})
        log.info("pooled out-of-fold: %s",
                 {k: round(v, 4) for k, v in r.pooled.items() if isinstance(v, float)})
        for k, v in r.mean.items():
            run.metric(f"cv_{k}", v)
        run.metric("n_configurations", 1)
        CV.write_json(out_dir / "cv_results.json", r.to_dict() | {"n_configurations": 1})
        if not args.no_registry:
            ids = CV.log_metrics(
                ds, experiment=EXPERIMENT, model=args.model, kind="cv", metrics=r.mean,
                config=config, n=int(r.pooled["n"]), is_baseline=False,
                notes="phase-1 CV on the official folds; one configuration; test rows untouched",
                extra={"n_configurations": 1, "fold_metrics": r.fold_metrics, "fold_sd": r.sd,
                       "pooled_out_of_fold": r.pooled,
                       "n_features": r.fold_metrics[0]["n_features"]},
                artifacts=[str(out_dir / "cv_results.json")],
            )
            CV.write_json(out_dir / "registry_ids_cv.json", {"cv": ids})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
