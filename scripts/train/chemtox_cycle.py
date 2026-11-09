#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Improvement cycle 2 for ChemTox: equal-budget search on ADORE's CV folds only.

Every family — including the published baselines (XGBoost, random forest) — gets the same budget
(``--budget``, default 32, fixed-seed random search from
:mod:`sentinel.models.chemtox.cycle`), so the comparison is between equally tuned models
(RESEARCH_PLAN §15.5 guard 4). Scores come from ADORE's official folds 0–4 inside the training
partition of the challenge; extra training rows (ECOTOX records ADORE does not contain, and/or the
other ADORE challenges) are training data only and are filtered per fold. **Nothing here reads the
ADORE test rows or the prospective test set** — only its *chemical list*, to exclude it from
training.

    python -B scripts/train/chemtox_cycle.py --model lightgbm --budget 32
    python -B scripts/train/chemtox_cycle.py --model xgboost --extra ecotox --budget 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.runlog import TrainingRun
from sentinel.data.sources import adore as A
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import cycle as CYC
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_cycle")
EXPERIMENT = "adore_chemtox_cycle2"
EXTRA_COLUMNS = ("chem_rdkit_can_smiles", "chem_inchikey", "chem_Morgan_fp", "chem_MACCS_fp",
                 "tax_genus", "tax_family", "tax_order", "tax_class", "reference_number")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--challenge", default="t-F2F", choices=A.CHALLENGES)
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar", choices=("molar", "mass"))
    ap.add_argument("--budget", type=int, default=CYC.DEFAULT_BUDGET)
    ap.add_argument("--search-seed", type=int, default=20261024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--extra", default="none",
                    choices=("none", "ecotox", "adore_taxa", "adore_taxa_all", "both"),
                    help="extra training rows (never evaluation rows)")
    ap.add_argument("--extra-weight", type=float, default=1.0)
    ap.add_argument("--no-deployment-masking", action="store_true",
                    help="keep the validation rows' water solubility / melting point during CV "
                         "(the biased comparison; see cycle.DEPLOYMENT_MISSING_COLUMNS)")
    ap.add_argument("--extra-any-species", action="store_true",
                    help="allow ECOTOX rows whose species ADORE does not cover (taxonomy imputed)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--accelerator", default="cpu", choices=("cpu", "gpu"),
                    help="device for the families that use one (tabpfn, embeddings, chemprop)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--screen", action="store_true",
                    help="apply the documented screening caps (cycle.SCREEN_CAPS); screened runs "
                         "are never used for selection, only to pick the finalists")
    ap.add_argument("--config-json", default=None,
                    help="run exactly one configuration given as JSON "
                         "{\"features\": {...}, \"params\": {...}} instead of the search "
                         "(used to reproduce a phase-1 frozen configuration under the masked CV "
                         "protocol so its out-of-fold predictions are comparable)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args(argv)


def set_thread_budget(n: int) -> int:
    n = max(1, min(int(n), 8))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    from sentinel.models.chemtox import gbdt as G
    from sentinel.models.chemtox import xgb as XG

    XG.XGB_DEFAULTS["n_jobs"] = n
    G.LGBM_DEFAULTS["n_jobs"] = n
    G.CATBOOST_DEFAULTS["thread_count"] = n
    return n


def completed(out_dir: Path, *, screen: bool) -> tuple[set[int], list[dict]]:
    """Indices already finished in this directory **in the same mode** (screened or full)."""
    done, recs = set(), []
    for f in sorted(out_dir.glob("shard*_results.json")):
        for r in json.loads(f.read_text()).get("records", []):
            if "cv_mean" in r and bool(r.get("screen")) == bool(screen) and r["index"] not in done:
                done.add(int(r["index"]))
                recs.append(r)
    return done, recs


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    masking = not args.no_deployment_masking
    label = (f"{args.model}_{args.challenge}_{args.split}_extra-{args.extra}"
             f"{'_dm' if masking else '_nodm'}")
    if args.config_json:
        # an explicit single configuration is not part of the family's 32-configuration search,
        # so it must not land in the search directory and inflate the configuration count
        label = f"{label}_cfg-{args.tag or 'explicit'}"
    out_dir = paths.run_dir("chemtox", f"cycle2/{label}")
    fh = logging.FileHandler(out_dir / f"cycle_{args.tag or args.shard}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    threads = set_thread_budget(args.threads)

    if args.config_json:
        explicit = json.loads(Path(args.config_json).read_text()
                              if Path(args.config_json).exists() else args.config_json)
        configs = [{"features": explicit.get("features", {}), "params": explicit.get("params", {})}]
        mine = [(int(explicit.get("index", 900)), configs[0])]
        log.info("explicit configuration: %s", json.dumps(explicit)[:400])
    else:
        configs = CYC.sample(args.model, args.budget, seed=args.search_seed)
        mine = CYC.shard(configs, args.shard, args.shards)
    if args.indices is not None and not args.config_json:
        wanted = set(args.indices)
        mine = [(i, c) for i, c in enumerate(configs) if i in wanted]
    resumed: list[dict] = []
    if args.resume:
        done, resumed = completed(out_dir, screen=args.screen)
        mine = [(i, c) for i, c in mine if i not in done]
        log.info("resume: %d configurations left", len(mine))
    if args.limit:
        mine = mine[: args.limit]

    extra = CYC.load_extra_rows(args.extra, challenge=args.challenge, split=args.split,
                                require_adore_species=not args.extra_any_species)
    log.info("extra training rows: %s", json.dumps(extra.info, default=str)[:600])
    CV.write_json(out_dir / "extra_rows_info.json", extra.info)

    datasets: dict[tuple, ChemToxDataset] = {}
    tag = args.tag or (f"screen{args.shard}" if args.screen else f"shard{args.shard}")
    config = {"challenge": args.challenge, "split": args.split, "target": args.target,
              "model": args.model, "budget": args.budget, "search_seed": args.search_seed,
              "shard": [args.shard, args.shards], "seed": args.seed, "threads": threads,
              "extra": args.extra, "extra_weight": args.extra_weight,
              "deployment_masking": masking,
              "extra_any_species": bool(args.extra_any_species),
              "screen": bool(args.screen), "screen_caps": CYC.SCREEN_CAPS.get(args.model, {}),
              "accelerator": args.accelerator,
              "extra_rows": extra.info.get("n_rows", 0),
              "selection": "mean RMSE over ADORE official CV folds 0-4 (ADORE rows only)"}
    records: list[dict] = list(resumed)
    with TrainingRun("chemtox", f"{EXPERIMENT}:{label}_shard{args.shard}", config=config,
                     data=["adore_raw", "adore", "ecotox", "chemtox_prospective"],
                     plan_ref=("RESEARCH_PLAN §5.4 / §15.5 amendment (improvement cycle 2, CV only);"
                               " docs/preregistration/chemtox_prospective_plan.md"),
                     artifacts_dir=str(out_dir)) as run:
        for n, (idx, cfg) in enumerate(mine, start=1):
            spec = FeatureSpec(**cfg["features"])
            key = tuple(sorted(cfg["features"].items()))
            if key not in datasets:
                datasets[key] = ChemToxDataset.load(args.challenge, args.split, target=args.target,
                                                    specs=spec, extra_columns=EXTRA_COLUMNS)
                datasets[key].assert_no_chemical_leakage(log=log)
            ds = datasets[key]
            t0 = time.time()
            params = CYC.apply_screen_caps(args.model, cfg["params"]) if args.screen \
                else cfg["params"]
            params = CYC.attach_device(args.model, params, args.accelerator)
            try:
                r = CYC.run_cv_extra(ds, spec, CYC.make_model(args.model, args.seed),
                                     params, model_name=args.model, folds=args.folds,
                                     extra=extra, extra_weight=args.extra_weight,
                                     deployment_masking=masking)
            except Exception as exc:  # noqa: BLE001 — a bad draw must not kill the shard
                log.warning("config %d FAILED (%s: %s)", idx, type(exc).__name__, exc)
                records.append({"index": idx, "features": cfg["features"], "params": params,
                                "error": f"{type(exc).__name__}: {exc}",
                                "screen": bool(args.screen),
                                "seconds": round(time.time() - t0, 1)})
                CV.write_json(out_dir / f"shard{tag}_results.json",
                              {"config": config, "records": records})
                continue
            suffix = "_screen" if args.screen else ""
            np.save(out_dir / f"oof_{args.model}_{idx:03d}{suffix}.npy", r.oof)
            np.save(out_dir / f"oofmask_{args.model}_{idx:03d}{suffix}.npy", r.oof_mask)
            records.append({"index": idx, "features": cfg["features"], "params": r.params,
                            "cv_mean": r.mean, "cv_sd": r.sd, "pooled": r.pooled,
                            "fold_metrics": r.fold_metrics, "seconds": round(r.seconds, 1),
                            "screen": bool(args.screen), "folds": args.folds})
            log.info("[%d/%d] config %d CV RMSE %.4f (%.0fs) %s", n, len(mine), idx,
                     r.mean.get("rmse", float("nan")), r.seconds, spec.name())
            CV.write_json(out_dir / f"shard{tag}_results.json",
                          {"config": config, "records": records})
        ok = [r for r in records if "cv_mean" in r]
        if ok:
            best = min(ok, key=lambda r: r["cv_mean"]["rmse"])
            run.metric("best_cv_rmse", best["cv_mean"]["rmse"])
            run.metric("n_configurations_in_shard", len(ok))
            run.note(f"best: config {best['index']} CV RMSE {best['cv_mean']['rmse']:.4f} "
                     f"({FeatureSpec(**best['features']).name()}, extra={args.extra})")
    print(json.dumps({"model": args.model, "extra": args.extra, "n_done": len(records),
                      "best_cv_rmse": min((r["cv_mean"]["rmse"] for r in records
                                           if "cv_mean" in r), default=None)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
