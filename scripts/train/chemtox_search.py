#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Equal-budget random search for one ChemTox model, scored on ADORE's official CV folds only.

The published baseline was tuned over 32 grid points, so every SENTINEL model gets the same
budget here (``--budget``, default 32, drawn by :func:`sentinel.models.chemtox.spaces.sample` with
a fixed seed). Nothing in this script reads the ``test`` rows.

Each configuration's per-fold metrics and out-of-fold predictions are written to the run directory
(the predictions are what `scripts/eval/chemtox_select.py` needs to build and score ensembles).
Shards (``--shard k --shards n``) split the *same* sampled list round-robin so several processes
can share one GPU; the aggregate and the registry entry are written by the select script, which
reports the full budget as ``n_configurations``.

    python -B -m sentinel.core.gpu --name chemtox_search --prefer 2 -- \
        python -B scripts/train/chemtox_search.py --model chemprop --shard 0 --shards 3 \
               --accelerator gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# CPU budget (docs/CONTRIBUTING_V2.md): at most 8 threads per job. Thread pools read these when
# numpy/torch/xgboost load, so they are set before those imports.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.runlog import TrainingRun
from sentinel.data.sources import adore as A
from sentinel.models.chemtox import aux as AUX
from sentinel.models.chemtox import cv as CV
from sentinel.models.chemtox import spaces
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

log = logging.getLogger("chemtox_search")
EXPERIMENT = "adore_chemtox_search"
EXTRA_COLUMNS = ("chem_rdkit_can_smiles", "chem_inchikey")


def make_model(model: str, seed: int):
    if model == "chemprop":
        from sentinel.models.chemtox.chemprop_model import make_chemprop

        return make_chemprop(seed)
    if model == "tabpfn":
        from sentinel.models.chemtox.tabpfn_model import make_tabpfn

        return make_tabpfn(seed)
    if model == "embeddings":
        from sentinel.models.chemtox.embeddings import make_embedding_gbm

        return make_embedding_gbm(seed)
    raise KeyError(model)


#: screening caps — documented, applied only when ``--screen`` is passed. Screening scores every
#: configuration cheaply on a fold subset; the finalists are then re-run with the configuration as
#: sampled, on all five folds, and only those full 5-fold scores are used for selection.
SCREEN_CAPS = {
    "chemprop": {"max_epochs": 12, "n_seeds": 1},
    "tabpfn": {"max_train_rows": 3000, "n_estimators": 2},
    "embeddings": {},
}


#: floors applied while screening (larger batches = fewer optimiser steps per epoch; the shared
#: machine is CPU-bound in graph collation, so this is what makes 32 screens affordable)
SCREEN_FLOORS = {"chemprop": {"batch_size": 128}}


def apply_caps(model: str, p: dict) -> dict:
    out = dict(p)
    for k, cap in SCREEN_CAPS.get(model, {}).items():
        if k in out and isinstance(out[k], (int, float)) and out[k] > cap:
            out[k] = cap
    for k, floor in SCREEN_FLOORS.get(model, {}).items():
        if k in out and isinstance(out[k], (int, float)) and out[k] < floor:
            out[k] = floor
    return out


def resolve_params(model: str, params: dict, args, side_data: dict) -> dict:
    """Turn a sampled configuration into runnable parameters (attach device / auxiliary data)."""
    p = dict(params)
    if getattr(args, "screen", False):
        p = apply_caps(model, p)
    if model == "chemprop":
        p["accelerator"] = args.accelerator
        init = p.pop("init", "none")
        if init == "chemeleon":
            p["from_foundation"] = "CheMeleon"
        elif init == "ecotox":
            p["pretrain"] = "ecotox"
            p["ecotox_rows"] = side_data["ecotox"]
            p["pretrain_epochs"] = args.pretrain_epochs
        if p.get("multitask"):
            p["aux_rows"] = side_data["adore_aux"]
        p["init"] = init  # keep the label for the record
    elif model == "tabpfn" or model == "embeddings":
        p["device"] = "cuda" if args.accelerator == "gpu" else "cpu"
    return p


def loggable(params: dict) -> dict:
    """Parameters without the attached data frames (so they can go in JSON / the registry)."""
    out = {}
    for k, v in params.items():
        if hasattr(v, "shape") and hasattr(v, "columns"):
            out[k] = f"<DataFrame rows={len(v)}>"
        else:
            out[k] = v
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(spaces.SPACES))
    ap.add_argument("--challenge", default="t-F2F", choices=A.CHALLENGES)
    ap.add_argument("--split", default=A.PRIMARY_SPLIT)
    ap.add_argument("--target", default="molar", choices=("molar", "mass"))
    ap.add_argument("--budget", type=int, default=spaces.DEFAULT_BUDGET)
    ap.add_argument("--search-seed", type=int, default=20261022)
    ap.add_argument("--seed", type=int, default=0, help="model seed")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--accelerator", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many configurations")
    ap.add_argument("--screen", action="store_true",
                    help="apply the documented screening caps (SCREEN_CAPS) to the sampled values")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="run only these configuration indices (the finalist pass)")
    ap.add_argument("--tag", default=None, help="suffix for the shard result file (e.g. 'final')")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads for this job (torch, XGBoost, BLAS); policy cap is 8")
    ap.add_argument("--resume", action="store_true",
                    help="skip configurations already completed in this run directory (same mode)")
    return ap.parse_args(argv)


def set_thread_budget(n: int) -> int:
    """Apply the per-job CPU thread cap to torch and to the XGBoost/RF defaults."""
    n = max(1, min(int(n), 8))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    import torch

    torch.set_num_threads(n)
    from sentinel.models.chemtox import xgb as XG

    XG.XGB_DEFAULTS["n_jobs"] = n
    return n


def completed_indices(out_dir: Path, *, screen: bool) -> tuple[set[int], list[dict]]:
    """Indices (and records) already finished in this directory for the same screen/full mode."""
    done: set[int] = set()
    recs: list[dict] = []
    for f in sorted(out_dir.glob("shard*_results.json")):
        for r in json.loads(f.read_text()).get("records", []):
            if "cv_mean" in r and bool(r.get("screen")) == bool(screen) and r["index"] not in done:
                done.add(int(r["index"]))
                recs.append(r)
    return done, recs


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = paths.run_dir("chemtox", f"search/{args.model}_{args.challenge}_{args.split}")
    fh = logging.FileHandler(out_dir / f"search_{args.tag or args.shard}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    threads = set_thread_budget(args.threads)
    log.info("CPU thread budget for this job: %d", threads)

    configs = spaces.sample(args.model, args.budget, seed=args.search_seed)
    mine = spaces.shard(configs, args.shard, args.shards)
    if args.indices is not None:
        wanted = set(args.indices)
        mine = [(i, c) for i, c in enumerate(configs) if i in wanted]
    if args.limit:
        mine = mine[: args.limit]
    resumed: list[dict] = []
    if args.resume:
        done, resumed = completed_indices(out_dir, screen=args.screen)
        before = len(mine)
        mine = [(i, c) for i, c in mine if i not in done]
        log.info("resume: %d of %d configurations already done, %d left", before - len(mine),
                 before, len(mine))
    log.info("%s: %d of %d configurations in shard %d/%d", args.model, len(mine), len(configs),
             args.shard, args.shards)

    # one dataset per distinct feature spec (loading all 1,106 columns repeatedly is wasteful)
    datasets: dict[tuple, ChemToxDataset] = {}
    side_data: dict[str, object] = {}
    if args.model == "chemprop":
        if any(c["params"].get("multitask") for _, c in mine):
            aux_rows, info = AUX.adore_auxiliary(args.challenge, split=args.split, columns="all")
            side_data["adore_aux"] = aux_rows
            log.info("multitask auxiliary rows: %d (%s)", info["n_rows"], info["per_challenge"])
            CV.write_json(out_dir / "adore_auxiliary_info.json", info)
        if any(c["params"].get("init") == "ecotox" for _, c in mine):
            eco, einfo = AUX.ecotox_auxiliary(args.challenge)
            side_data["ecotox"] = eco
            log.info("ECOTOX pretraining rows: %d over %d chemicals (dropped %d test-chemical rows)",
                     einfo["n_final"], einfo["n_chemicals"], einfo["n_dropped"])
            CV.write_json(out_dir / "ecotox_auxiliary_info.json", einfo)

    tag = args.tag or (f"screen_shard{args.shard}" if args.screen else f"shard{args.shard}")
    config = {"challenge": args.challenge, "split": args.split, "target": args.target,
              "model": args.model, "budget": args.budget, "search_seed": args.search_seed,
              "shard": [args.shard, args.shards], "seed": args.seed,
              "screen": bool(args.screen), "screen_caps": SCREEN_CAPS.get(args.model, {}),
              "screen_floors": SCREEN_FLOORS.get(args.model, {}),
              "folds": args.folds, "indices": args.indices, "tag": tag, "threads": threads,
              "selection": "mean RMSE over ADORE official CV folds (subset while screening)"}
    records: list[dict] = list(resumed)
    with TrainingRun("chemtox", f"{EXPERIMENT}:{args.model}_shard{args.shard}", config=config,
                     data=["adore_raw", "adore", "ecotox"],
                     plan_ref="RESEARCH_PLAN §5.4 / research 04 §D.3 (improvement cycle, CV only)",
                     artifacts_dir=str(out_dir)) as run:
        for n, (idx, cfg) in enumerate(mine, start=1):
            spec = FeatureSpec(**cfg["features"])
            key = tuple(sorted(cfg["features"].items()))
            if key not in datasets:
                datasets[key] = ChemToxDataset.load(args.challenge, args.split, target=args.target,
                                                    specs=spec, extra_columns=EXTRA_COLUMNS)
                datasets[key].assert_no_chemical_leakage(log=log)
            ds = datasets[key]
            params = resolve_params(args.model, cfg["params"], args, side_data)
            t0 = time.time()
            try:
                r = CV.run_cv(ds, spec, make_model(args.model, args.seed), params,
                              model_name=args.model, folds=args.folds)
            except Exception as exc:  # noqa: BLE001 — a bad draw must not kill the shard
                log.warning("config %d FAILED (%s: %s)", idx, type(exc).__name__, exc)
                records.append({"index": idx, "features": cfg["features"],
                                "params": loggable(params), "error": f"{type(exc).__name__}: {exc}",
                                "screen": bool(args.screen), "folds": args.folds,
                                "seconds": round(time.time() - t0, 1)})
                continue
            suffix = "_screen" if args.screen else ""
            np.save(out_dir / f"oof_{args.model}_{idx:03d}{suffix}.npy", r.oof)
            np.save(out_dir / f"oofmask_{args.model}_{idx:03d}{suffix}.npy", r.oof_mask)
            rec = {"index": idx, "features": cfg["features"], "params": loggable(params),
                   "cv_mean": r.mean, "cv_sd": r.sd, "pooled": r.pooled,
                   "fold_metrics": r.fold_metrics, "seconds": round(r.seconds, 1),
                   "screen": bool(args.screen), "folds": args.folds}
            records.append(rec)
            log.info("[%d/%d] config %d CV RMSE %.4f (%.1fs) %s", n, len(mine), idx,
                     r.mean.get("rmse", float("nan")), r.seconds,
                     {k: v for k, v in loggable(params).items()
                      if k in ("init", "multitask", "n_seeds", "species_dim", "use_conditioning",
                               "encoder", "max_features", "max_train_rows")})
            CV.write_json(out_dir / f"shard{tag}_results.json",
                          {"config": config, "records": records})
        ok = [r for r in records if "cv_mean" in r]
        if ok:
            best = min(ok, key=lambda r: r["cv_mean"]["rmse"])
            run.metric("best_cv_rmse", best["cv_mean"]["rmse"])
            run.metric("n_configurations_in_shard", len(ok))
            run.note(f"best in shard: config {best['index']} CV RMSE {best['cv_mean']['rmse']:.4f}")
            log.info("shard best: config %d CV RMSE %.4f", best["index"], best["cv_mean"]["rmse"])
    print(json.dumps({"shard": args.shard, "n_done": len(records),
                      "best_cv_rmse": min((r["cv_mean"]["rmse"] for r in records if "cv_mean" in r),
                                          default=None)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
