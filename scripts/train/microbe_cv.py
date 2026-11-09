#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Phase-1 cross-validation of SENTINEL-Microbe (M3) models and baselines on NEON development events.

Protocol: docs/preregistration/microbe_protocol.md (drafted before any data or model existed).
Development rows only (``load_design`` refuses holdout events). For every (model, scheme): out-of-fold
predictions (mean of seeds for stochastic models) saved under ``$SENTINEL_RUNS/microbe/cv/<scheme>/``,
pooled / per-domain / anomaly R², RMSE, Spearman, MSE skill vs RF[DOY+location] (when its OOF file
exists), site-cluster bootstrap CIs. One ``TrainingRun`` per (model, scheme); one registry entry per
(target, metric) — workstream 'M3-microbe', experiment 'microbe_phase1_cv', split
'neon_dev/<scheme>_cv'.

    python -B scripts/train/microbe_cv.py --models context_rf rf_clr --schemes lodo
    python -m sentinel.core.gpu --name microbe_cv --prefer 4 -- \
        python -B scripts/train/microbe_cv.py --models seqset_dnabert --schemes lodo --device cuda
"""

from __future__ import annotations

import os

MAX_THREADS = 6   # shared-host rule 2026-10-24 22:30: one CPU-heavy job, <= 6 threads
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun

REPO = Path(__file__).resolve().parents[2]
WORKSTREAM = "M3-microbe"
EXPERIMENT = "microbe_phase1_cv"
CFG_SEQSET = REPO / "configs" / "microbe" / "seqset.yaml"
DATA_MANIFESTS = ["microbe_wq"]
log = logging.getLogger("microbe_cv")
V1_TRAIN_SCHEMES = {"lodo_x001"}   # enlarged-training LODO (protocol §10.2): .001 2017-18 rows in training only


def cv_dir(scheme: str, tag: str = "") -> Path:
    return paths.ensure(paths.RUNS_ROOT / "microbe" / f"cv{tag}" / scheme)


def model_label(name: str, cfg: dict | None = None) -> str:
    from sentinel.models.microbe.baselines import LABELS

    if name in LABELS:
        return LABELS[name]
    suffix = "+context" if cfg and cfg.get("context") else ""
    return f"SeqSet[{name}]{suffix}"


def score_and_log(design, oof: pd.DataFrame, *, name: str, label: str, scheme: str, targets, is_baseline: bool,
                  config: dict, n_boot: int, per_seed: dict | None, artifacts: list[str], run: TrainingRun,
                  ref: pd.DataFrame | None, extra_notes: str = "", experiment: str = EXPERIMENT,
                  manifests=None) -> dict:
    from sentinel.models.microbe import evaluate as E

    S = design.samples.loc[oof.index]
    region = S["domain_id"].astype(str).to_numpy()
    site = S["site_id"].astype(str).to_numpy()
    res = {}
    for t in targets:
        y = S[t].to_numpy(float)
        p = oof[t].to_numpy(float)
        if np.sum(np.isfinite(y) & np.isfinite(p)) < 10:
            continue
        sm = E.summary(y, p, region)
        ci = E.cis(y, p, region, site, n_boot=n_boot)
        seeds = {}
        if per_seed:
            seeds = {int(s): float(E.summary(y, ps[t].reindex(oof.index).to_numpy(float), region)["r2"])
                     for s, ps in per_seed.items()}
        skill = E.mse_skill(y, p, ref[t].reindex(oof.index).to_numpy(float)) if ref is not None else None
        res[t] = {**{k: v for k, v in sm.items()}, "ci": ci, "per_seed_r2": seeds, "mse_skill_vs_context": skill}
        common = dict(workstream=WORKSTREAM, experiment=experiment, model=label, split=f"neon_dev/{scheme}_cv",
                      target=t, is_baseline=is_baseline, config=config, data_manifests=manifests or DATA_MANIFESTS,
                      artifacts=artifacts, n=sm["n"])
        log_result(metric="r2", value=sm["r2"], ci_low=ci["r2"]["ci_low"], ci_high=ci["r2"]["ci_high"],
                   ci_method=f"percentile cluster bootstrap B={n_boot}, unit=site",
                   extra={"ci_domain": ci["r2"]["ci_domain"], "per_seed_r2": seeds,
                          "r2_by_domain": sm["r2_by_region"]}, notes=extra_notes, **common)
        log_result(metric="r2_anomaly", value=sm["r2_anomaly"], ci_low=ci["r2_anomaly"]["ci_low"],
                   ci_high=ci["r2_anomaly"]["ci_high"], ci_method=f"percentile cluster bootstrap B={n_boot}, unit=site",
                   extra={"ci_domain": ci["r2_anomaly"]["ci_domain"]}, notes=extra_notes, **common)
        for m in ("r2_region_median", "rmse", "spearman"):
            if np.isfinite(sm[m]):
                log_result(metric=m, value=sm[m], notes=extra_notes, **common)
        if skill is not None and np.isfinite(skill):
            log_result(metric="mse_skill_vs_context_rf", value=skill, notes=extra_notes, **common)
        run.metric(f"r2_{t}", sm["r2"])
    if res:
        run.metric("mean_r2", float(np.mean([r["r2"] for r in res.values()])))
    return res


def run_one(name: str, scheme: str, a, design, folds, targets) -> None:
    from sentinel.models.microbe import baselines as B
    from sentinel.models.microbe import cv as CV

    out = cv_dir(scheme, a.tag)
    is_base = name in B.BASELINES and name not in B.CANDIDATE_TABLES
    is_table = name in B.BASELINES
    cfg = {"model": name, "scheme": scheme}
    if is_table:
        seeds = list(a.seeds) if name in B.STOCHASTIC else [0]
        cfg.update({"seeds": seeds, "params": B.DEFAULTS, "descriptors": not scheme.startswith("release")})
    else:
        mcfg = CV.load_config(CFG_SEQSET, name.replace("+context", ""))
        if name.endswith("+context"):
            mcfg["context"] = True
        if scheme.startswith("release"):
            mcfg["descriptors"] = False
        mcfg["inner_val_unit"] = "site" if scheme == "loso" else "domain"
        cfg.update(mcfg)
        cfg["seeds"] = list(a.seeds)
    label = model_label(name.replace("+context", ""), cfg if not is_base else None)
    ref_path = out / "context_rf.parquet"
    with TrainingRun(WORKSTREAM, f"{EXPERIMENT}:{name}:{scheme}", config=cfg, data=DATA_MANIFESTS,
                     plan_ref="RESEARCH_PLAN §5.3; microbe_protocol §6", iteration=a.iteration,
                     artifacts_dir=str(out)) as run:
        per_seed = None
        if is_table:
            bparams = {"descriptors": False} if scheme.startswith("release") else {}
            if name in B.SEQEMB:
                bparams["seq_table"] = CV.load_seq_table(B.SEQEMB[name], design.tokens["seq_id"].unique())
                cfg["encoder"] = B.SEQEMB[name]
            preds = [CV.oof_baseline(name, design, folds, targets=targets, seed=s, n_jobs=MAX_THREADS,
                                     device=a.device, params=bparams) for s in cfg["seeds"]]
            per_seed = {s: p for s, p in zip(cfg["seeds"], preds)} if len(preds) > 1 else None
            oof = sum(p[list(targets)] for p in preds) / len(preds)
        else:
            table, seq_row = None, None
            if cfg.get("seq_mode", "frozen") == "frozen":
                table, seq_row = CV.load_seq_table(cfg["encoder"], design.tokens["seq_id"].unique())
            diag_path = out / f"{name}_diag.jsonl"
            diag_path.unlink(missing_ok=True)

            def fold_log(d):
                with open(diag_path, "a") as fh:
                    fh.write(json.dumps(d) + "\n")

            res_ = CV.oof_seqset(cfg, design, folds, targets=targets, seeds=tuple(cfg["seeds"]),
                                 device=a.device, table=table, seq_row=seq_row, fold_log=fold_log,
                                 return_embeddings=a.save_embeddings)
            oof_full, ps, diag = res_[:3]
            if a.save_embeddings:
                emb = pd.DataFrame.from_dict(res_[3], orient="index")
                emb.columns = [f"e{i}" for i in range(emb.shape[1])]
                emb.to_parquet(out / f"{name}_embeddings.parquet")
            tested = np.unique(np.concatenate([f.test for f in folds]))
            oof = oof_full.loc[tested]
            per_seed = {s: v.loc[tested] for s, v in ps.items()}
            run.metric("n_params", diag[0]["n_params"])
            run.note(f"median best epoch {np.median([d['best_epoch'] for d in diag]):.0f}")
        oof.to_parquet(out / f"{name}.parquet")
        if per_seed:
            for s, p in per_seed.items():
                p[list(targets)].to_parquet(out / f"{name}_seed{s}.parquet")
        ref = pd.read_parquet(ref_path) if ref_path.exists() and name != "context_rf" else None
        res = score_and_log(design, oof[list(targets)], name=name, label=label, scheme=scheme, targets=targets,
                            is_baseline=is_base, config=cfg, n_boot=a.n_boot, per_seed=per_seed,
                            artifacts=[str(out / f"{name}.parquet")], run=run, ref=ref,
                            extra_notes=a.notes)
        (out / f"{name}_scores.json").write_text(json.dumps(res, indent=1, default=float))
        log.info("%s %s: %s", name, scheme, {t: round(r["r2"], 3) for t, r in res.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--schemes", nargs="+", default=["lodo"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iteration", default="base")
    ap.add_argument("--tag", default="", help="suffix for the output dir (improvement cycles)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--save-embeddings", action="store_true", help="SeqSet: save OOF sample embeddings (probe)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import torch

    torch.set_num_threads(2 if a.device != "cpu" else MAX_THREADS)   # GPU jobs: 2 threads (shared-host rule)
    from sentinel.models.microbe import splits as SP
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    design = load_design()
    targets = list(TARGET_NAMES)
    for scheme in a.schemes:
        needs_v1 = scheme.startswith("release") or scheme in V1_TRAIN_SCHEMES
        if needs_v1 or getattr(design, "_v1", False):
            design = load_design(with_release_v1=needs_v1)
            design._v1 = needs_v1
        folds = SP.make_folds(design.samples, scheme)
        (cv_dir(scheme, a.tag) / "folds.json").write_text(json.dumps([f.to_dict() for f in folds]))
        for m in a.models:
            run_one(m, scheme, a, design, folds, targets)


if __name__ == "__main__":
    main()
