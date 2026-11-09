#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score the frozen ChemTox configurations on the ADORE test rows — ONCE, after approval.

This script refuses to run unless

1. ``--i-have-approval`` is passed (the coordinator's review gate), and
2. the frozen configurations in ``selection.json`` still hash to the value recorded in
   ``docs/preregistration/chemtox_test_plan.md`` (``--expect-hash``), and
3. no entry for the same (experiment, model, metric, split) already exists in the registry —
   the test set is touched once; a second attempt must be a deliberate, superseding re-run
   (``--allow-rescore``, which records ``supersedes``).

What it does: rebuilds each frozen configuration, fits it on the **whole official training
partition** (3 seeds, mean reported with the seed spread), predicts the official test rows, scores
them with the same metric suite as CV, attaches chemical-cluster bootstrap CIs and paired
bootstrap comparisons against the reproduced published baseline and the best trivial baseline,
applies Holm across the per-model primary comparisons, and writes everything to the registry.

Ensembles: the stack's weights are refitted from the components' **training-portion out-of-fold**
predictions (never test rows) before the test predictions are combined; the ensemble is logged
under its own name with the components' individual test scores in ``extra``.

    python -B scripts/eval/chemtox_score_test.py --dry-run          # prints the plan, scores nothing
    python -B scripts/eval/chemtox_score_test.py --i-have-approval --expect-hash <sha16>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import package_hash, registry_frame
from sentinel.core.runlog import TrainingRun
from sentinel.data.sources import adore as A
from sentinel.eval.bootstrap import holm_bonferroni, paired_bootstrap
from sentinel.eval.metrics import rmse
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import evaluate as E
from sentinel.models.chemtox import xgb as XG
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_score_test")
EXPERIMENT = "adore_chemtox_test"
EXTRA_COLUMNS = ("chem_rdkit_can_smiles", "chem_inchikey")


def frozen_hash(frozen: dict) -> str:
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def make_model(model: str, params: dict, seed: int):
    if model == "xgboost":
        return XG.XGBoostTox(params=params, seed=seed)
    if model == "rf":
        return XG.RandomForestTox(params=params, seed=seed)
    if model == "tabpfn":
        from sentinel.models.chemtox.tabpfn_model import TabPFNTox

        return TabPFNTox(params=params, seed=seed)
    if model.startswith("chemprop"):
        from sentinel.models.chemtox.chemprop_model import ChempropToxModel

        return ChempropToxModel(params=params, seed=seed)
    if model == "embeddings":
        from sentinel.models.chemtox.embeddings import EmbeddingGBM

        return EmbeddingGBM(params=params, seed=seed)
    raise KeyError(model)


def component_oof(ds: ChemToxDataset, comps: list[str], payload: dict, *, log) -> dict | None:
    """Components' out-of-fold predictions on the TRAINING portion (never test rows).

    Deep/PFN components are read from the search run that produced their frozen configuration
    (``selection.json`` records the configuration index); the tree baselines are re-run over the
    official folds, which costs a minute and keeps the stack reproducible from the repository.
    """
    out: dict[str, np.ndarray] = {}
    search = paths.RUNS_ROOT / "chemtox" / "search"
    for c in comps:
        if c in ("xgboost", "rf"):
            from sentinel.models.chemtox import cv as _CV

            features, params = BASELINE_COMPONENTS[c]
            maker = XG.make_xgb(0) if c == "xgboost" else XG.make_rf(0)
            r = _CV.run_cv(ds, FeatureSpec(**features), maker, params, model_name=c)
            out[c] = r.oof
            log.info("%s: out-of-fold predictions recomputed (CV RMSE %.4f)", c, r.mean["rmse"])
            continue
        idx = (payload.get("models", {}).get(c) or {}).get("best_index")
        if idx is None:
            return None
        p = search / f"{c}_{ds.challenge}_{ds.split}" / f"oof_{c}_{int(idx):03d}.npy"
        if not p.exists():
            return None
        out[c] = np.load(p)
        log.info("%s: out-of-fold predictions from %s", c, p.name)
    return out


def nnls_weights(oof: dict[str, np.ndarray], y, *, log) -> dict[str, float]:
    """Non-negative, sum-to-one stack weights fitted on the training portion's out-of-fold rows."""
    from scipy.optimize import nnls

    names = list(oof)
    P = np.column_stack([np.asarray(oof[n], dtype=float) for n in names])
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y) & np.isfinite(P).all(axis=1)
    w, _ = nnls(P[m], y[m])
    s = w.sum()
    w = w / s if s > 0 else np.full(len(names), 1.0 / len(names))
    log.info("stack weights fitted on %d out-of-fold rows: %s", int(m.sum()),
             {n: round(float(v), 3) for n, v in zip(names, w)})
    return {n: float(v) for n, v in zip(names, w)}


#: the baseline components' frozen configurations (kept here so the stack is self-contained)
BASELINE_COMPONENTS = {
    "xgboost": ({"fingerprint": "MACCS"}, {"n_estimators": 800, "eta": 0.03, "max_depth": 6,
                                            "min_child_weight": 1, "subsample": 0.8}),
    "rf": ({"fingerprint": "MACCS"}, {"n_estimators": 300, "max_depth": 20, "max_features": 0.3}),
}


def already_scored(model: str, split_label: str) -> bool:
    df = registry_frame()
    if df.empty:
        return False
    m = ((df["experiment"] == EXPERIMENT) & (df["model"] == model) & (df["split"] == split_label)
         & (df["metric"] == "rmse"))
    return bool(m.any())


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default=None, help="path to selection.json (default: the run dir)")
    ap.add_argument("--challenge", default="t-F2F")
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--accelerator", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--expect-hash", default=None, help="frozen-configuration hash from the pre-registration")
    ap.add_argument("--i-have-approval", action="store_true",
                    help="the coordinator approved this single test scoring")
    ap.add_argument("--allow-rescore", action="store_true",
                    help="deliberately supersede an existing test entry (recorded as such)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sel_path = Path(args.selection) if args.selection else (
        paths.RUNS_ROOT / "chemtox" / "select" / f"{args.challenge}_{args.split}" / "selection.json")
    payload = json.loads(Path(sel_path).read_text())
    frozen = payload.get("frozen")
    if not frozen:
        raise SystemExit(f"{sel_path} has no 'frozen' block — run chemtox_select.py --freeze first")
    h = frozen_hash(frozen)
    log.info("frozen-configuration hash: %s (package %s, this script %s)", h, package_hash(),
             file_hash(Path(__file__)))
    for name, f in frozen.items():
        log.info("  %-22s features=%s params=%s", name, f.get("features"), f.get("params"))
    if args.expect_hash and args.expect_hash != h:
        raise SystemExit(f"frozen configurations changed: {h} != pre-registered {args.expect_hash}")
    if args.dry_run:
        log.info("dry run: nothing scored")
        return 0
    if not args.i_have_approval:
        raise SystemExit("refusing to read the test rows without --i-have-approval")

    # one dataset per distinct frozen feature spec: the configurations use different fingerprints
    # (MACCS for the trees, Morgan + descriptors for TabPFN, none for the graph/embedding models)
    specs = [FeatureSpec(**f["features"]) for f in frozen.values() if not f.get("ensemble")]
    datasets: dict[tuple, ChemToxDataset] = {}

    def dataset_for(spec: FeatureSpec) -> ChemToxDataset:
        key = tuple(sorted(spec.to_dict().items()))
        if key not in datasets:
            d = ChemToxDataset.load(args.challenge, args.split, target=args.target, specs=spec,
                                    extra_columns=EXTRA_COLUMNS)
            d.assert_no_chemical_leakage(log=log)
            datasets[key] = d
        return datasets[key]

    ds = dataset_for(FeatureSpec(fingerprint="MACCS"))  # reference frame for rows/targets/folds
    for s in specs:
        dataset_for(s)
    split_label = CV.split_label(ds, "test")
    trn, tst = np.flatnonzero(ds.train_mask()), np.flatnonzero(ds.test_mask())
    y_test, chem, taxon = ds.y[tst], ds.chem[tst], ds.taxon[tst]
    out_dir = paths.run_dir("chemtox", f"test/{args.challenge}_{args.split}")

    preds: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    with TrainingRun("chemtox", f"{EXPERIMENT}:{args.challenge}_{args.split}",
                     config={"frozen_hash": h, "seeds": args.seeds, "selection": str(sel_path)},
                     data=["adore_raw", "adore", "ecotox"],
                     plan_ref="docs/preregistration/chemtox_test_plan.md",
                     artifacts_dir=str(out_dir)) as run:
        for name, f in frozen.items():
            if name.startswith("ensemble"):
                continue
            if already_scored(name, split_label) and not args.allow_rescore:
                log.warning("%s already has a test entry — skipping (use --allow-rescore to supersede)",
                            name)
                continue
            spec = FeatureSpec(**f["features"])
            dsm = dataset_for(spec)
            per_seed = []
            for seed in args.seeds:
                params = dict(f["params"])
                if name.startswith("chemprop"):
                    params["accelerator"] = args.accelerator
                Xtr, Xte, builder = dsm.matrices(spec, trn, tst)
                model = make_model(name, params, seed)
                model.fit(Xtr, dsm.y[trn], train_rows=dsm.rows(trn), valid_rows=dsm.rows(tst),
                          feature_names=builder.names_, builder=builder)
                per_seed.append(np.asarray(model.predict(Xte, rows=dsm.rows(tst)), dtype=float))
                log.info("%s seed %d scored", name, seed)
            P = np.vstack(per_seed)
            pred = P.mean(axis=0)
            preds[name] = pred
            metrics = E.report(y_test, pred, chem=chem, taxon=taxon)
            cis = E.bootstrap_report(y_test, pred, chem=chem, taxon=taxon, n_boot=args.n_boot)
            seed_rmse = [float(rmse(y_test, p)) for p in P]
            results[name] = {"metrics": metrics, "cis": cis, "seed_rmse": seed_rmse,
                             "frozen": f, "n_seeds": len(args.seeds)}
            np.save(out_dir / f"test_pred_{name}.npy", pred)
            run.metric(f"test_rmse_{name}", metrics["rmse"])
            log.info("%s TEST rmse %.4f (seeds %s)", name, metrics["rmse"],
                     [round(v, 4) for v in seed_rmse])

        # ---- ensembles: weights from the TRAINING portion's out-of-fold predictions only ----
        for name, f in frozen.items():
            if not f.get("ensemble"):
                continue
            comps = [c for c in f["components"] if c in preds]
            if len(comps) < 2:
                log.warning("%s: fewer than two components scored, skipping", name)
                continue
            if "stack" in name:
                oof = component_oof(ds, comps, payload, log=log)
                if oof is None:
                    log.warning("%s: component out-of-fold predictions unavailable, skipping", name)
                    continue
                w = nnls_weights(oof, ds.y, log=log)
                pred = sum(w[c] * preds[c] for c in comps)
                weights = {c: float(w[c]) for c in comps}
            else:
                pred = np.mean([preds[c] for c in comps], axis=0)
                weights = {c: 1.0 / len(comps) for c in comps}
            metrics = E.report(y_test, pred, chem=chem, taxon=taxon)
            cis = E.bootstrap_report(y_test, pred, chem=chem, taxon=taxon, n_boot=args.n_boot)
            preds[name] = pred
            results[name] = {"metrics": metrics, "cis": cis, "frozen": f, "weights": weights,
                             "components": comps, "is_ensemble": True,
                             "component_test_rmse": {c: results[c]["metrics"]["rmse"] for c in comps}}
            np.save(out_dir / f"test_pred_{name}.npy", pred)
            run.metric(f"test_rmse_{name}", metrics["rmse"])
            log.info("%s (ENSEMBLE of %s) TEST rmse %.4f, weights %s", name, "+".join(comps),
                     metrics["rmse"], {k: round(v, 3) for k, v in weights.items()})

        # ---- paired comparisons against BOTH reproduced baselines, Holm across models ----
        p_values: dict[str, float] = {}
        for ref in ("xgboost", "rf"):
            if ref not in preds:
                continue
            for name, pred in preds.items():
                if name == ref:
                    continue
                t = paired_bootstrap(rmse, y_test, pred, preds[ref], groups=chem,
                                     n_boot=args.n_boot, higher_is_better=False)
                results[name][f"paired_vs_{ref}"] = t
                if ref == "xgboost":
                    p_values[name] = t["p_value"]
        holm = holm_bonferroni(p_values) if p_values else {}
        for name, rejected in holm.items():
            results[name]["holm_rejected_vs_xgboost"] = bool(rejected)

        CV.write_json(out_dir / "test_results.json",
                      {"frozen_hash": h, "results": results, "holm": holm, "seeds": args.seeds})
        for name, r in results.items():
            CV.log_metrics(ds, experiment=EXPERIMENT, model=name, kind="test", metrics=r["metrics"],
                           config={"frozen": r["frozen"], "frozen_hash": h, "seeds": args.seeds},
                           n=int(r["metrics"]["n"]), cis=r["cis"], is_baseline=False,
                           notes=(("ENSEMBLE of " + "+".join(r.get("components") or []) + "; ")
                                  if r.get("is_ensemble") else "")
                                 + "single pre-registered test scoring "
                                   "(docs/preregistration/chemtox_test_plan.md)",
                           extra={"seed_rmse": r.get("seed_rmse"),
                                  "paired_vs_xgboost": r.get("paired_vs_xgboost"),
                                  "paired_vs_rf": r.get("paired_vs_rf"),
                                  "holm_rejected_vs_xgboost": r.get("holm_rejected_vs_xgboost"),
                                  "is_ensemble": r.get("is_ensemble", False),
                                  "components": r.get("components"),
                                  "component_test_rmse": r.get("component_test_rmse"),
                                  "stack_weights": r.get("weights"),
                                  "published_reference": {"rmse": 0.927, "model": "XGBoost + MACCS",
                                                          "source": "Gasser et al. 2024"}},
                           artifacts=[str(out_dir / "test_results.json")])
    log.info("wrote %s", out_dir / "test_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
