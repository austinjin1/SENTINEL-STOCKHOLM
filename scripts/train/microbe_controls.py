#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Mandatory controls for SENTINEL-Microbe (protocol §5): permutation null and site-identity probe.

* ``--permutation``: row-permute the target block (all targets together), refit under LODO, score
  pooled R² — RF[genus-CLR] with ``--n-perm`` permutations (seed 0 model), optionally a SeqSet
  candidate (``--seqset``) with fewer. Registry experiment 'microbe_phase1_controls', metric
  'r2_permutation_null_{mean,q95,max}' plus the real-vs-null p-value.
* ``--probe``: 5-fold stratified (row-random) site classification from genus CLR (logistic
  regression, RF) and — if OOF SeqSet embeddings exist — from them. Metric 'site_id_accuracy'.

    python -B scripts/train/microbe_controls.py --permutation --probe
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
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

WORKSTREAM = "M3-microbe"
EXPERIMENT = "microbe_phase1_controls"
REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger("microbe_controls")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutation", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--perm-models", nargs="+", default=["rf_clr"], help="table models for the null")
    ap.add_argument("--seqset", default=None, help="SeqSet candidate for the permutation null")
    ap.add_argument("--n-perm-seqset", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import torch

    torch.set_num_threads(2 if a.device != "cpu" else 6)
    from sentinel.eval import metrics as EM
    from sentinel.models.microbe import baselines as B
    from sentinel.models.microbe import cv as CV
    from sentinel.models.microbe import evaluate as E
    from sentinel.models.microbe import features as F
    from sentinel.models.microbe import splits as SP
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    design = load_design()
    S = design.samples
    targets = list(TARGET_NAMES)
    out = paths.ensure(paths.RUNS_ROOT / "microbe" / "controls")
    folds = SP.make_folds(S, "lodo")
    common = dict(workstream=WORKSTREAM, experiment=EXPERIMENT, data_manifests=["microbe_wq"])

    if a.permutation:
        real_dir = paths.RUNS_ROOT / "microbe" / f"cv{a.tag}" / "lodo"
        jobs = [(m, a.n_perm) for m in a.perm_models] + ([(a.seqset, a.n_perm_seqset)] if a.seqset else [])
        for name, n_perm in jobs:
            cfg = {"model": name, "n_perm": n_perm, "scheme": "lodo", "seeds": list(a.seeds),
                   "permutation": "within each fold's training rows, all targets jointly", "tag": a.tag}
            with TrainingRun(WORKSTREAM, f"{EXPERIMENT}:perm:{name}", config=cfg, data=["microbe_wq"],
                             plan_ref="microbe_protocol §5", artifacts_dir=str(out)) as run:
                null = {t: [] for t in targets}
                for k in range(n_perm):
                    # within-fold permutation of TRAINING targets only (review W1); same seed rule as the
                    # real score (mean of the seeded fits)
                    if name in B.BASELINES:
                        seeds = a.seeds if name in B.STOCHASTIC else [0]
                        bp = ({"seq_table": CV.load_seq_table(B.SEQEMB[name], design.tokens["seq_id"].unique())}
                              if name in B.SEQEMB else None)
                        preds = [CV.oof_baseline(name, design, folds, targets=targets, seed=s, n_jobs=6,
                                                 permute_seed=k + 1, params=bp) for s in seeds]
                        oof = sum(p[targets] for p in preds) / len(preds)
                    else:
                        mcfg = CV.load_config(REPO / "configs" / "microbe" / "seqset.yaml", name)
                        mcfg["inner_val_unit"] = "domain"
                        table, seq_row = (CV.load_seq_table(mcfg["encoder"], design.tokens["seq_id"].unique())
                                          if mcfg.get("seq_mode", "frozen") == "frozen" else (None, None))
                        oof, _, _ = CV.oof_seqset(mcfg, design, folds, targets=targets, seeds=tuple(a.seeds),
                                                  device=a.device, table=table, seq_row=seq_row, permute_seed=k + 1)
                        oof = oof.dropna(how="all")
                    for t in targets:
                        null[t].append(EM.r2(S.loc[oof.index, t].to_numpy(float), oof[t].to_numpy(float)))
                    log.info("perm %s %d: %s", name, k, {t: round(v[-1], 3) for t, v in null.items()})
                real_p = real_dir / f"{name}.parquet"
                real = pd.read_parquet(real_p) if real_p.exists() else None
                summ = {}
                for t in targets:
                    v = np.array(null[t], float)
                    r = (EM.r2(S.loc[real.index, t].to_numpy(float), real[t].to_numpy(float))
                         if real is not None else np.nan)
                    p = float((1 + np.sum(v >= r)) / (1 + len(v))) if np.isfinite(r) else None
                    summ[t] = {"null": v.tolist(), "real_r2": r, "p_value": p}
                    model_label = B.LABELS.get(name, f"SeqSet[{name}]")
                    for m, val in (("r2_permutation_null_mean", v.mean()), ("r2_permutation_null_q95", np.quantile(v, 0.95)),
                                   ("r2_permutation_null_max", v.max())):
                        log_result(model=model_label, metric=m, value=float(val), split="neon_dev/lodo_cv", target=t,
                                   n=len(v), config=cfg, is_baseline=name in B.BASELINES and name not in B.SEQEMB,
                                   extra={"null": v.tolist(), "real_r2": r, "p_value": p},
                                   notes="row permutation of all targets jointly; model refit per fold", **common)
                    run.metric(f"null_mean_{t}", float(v.mean()))
                (out / f"permutation_{name}.json").write_text(json.dumps(summ, indent=1, default=float))

    if a.probe:
        cfg = {"probe": "site_id", "cv": "stratified 5-fold over rows", "features": "genus CLR (all dev rows)"}
        with TrainingRun(WORKSTREAM, f"{EXPERIMENT}:site_probe", config=cfg, data=["microbe_wq"],
                         plan_ref="microbe_protocol §5", artifacts_dir=str(out)) as run:
            ids = S.index.to_numpy()
            # Feature filter fit on all development rows: the probe is a descriptive diagnostic of site
            # signal, not a model; it is never used for selection.
            X = F.GenusCLR().fit(design.tokens, ids).transform(design.tokens, ids)
            res = {}
            for mdl in ("logreg", "rf"):
                res[f"clr_{mdl}"] = E.site_identity_probe(X, S["site_id"].to_numpy(), model=mdl)
            res["domain_clr_logreg"] = E.site_identity_probe(X, S["domain_id"].to_numpy(), model="logreg")
            for emb_p in sorted((paths.RUNS_ROOT / "microbe" / f"cv{a.tag}" / "lodo").glob("*_embeddings.parquet")):
                emb = pd.read_parquet(emb_p).reindex(ids)
                ok = emb.notna().all(1).to_numpy()
                res[f"{emb_p.stem.replace('_embeddings', '')}_oof_embedding_logreg"] = E.site_identity_probe(
                    emb.to_numpy()[ok], S["site_id"].to_numpy()[ok])
            for k, r in res.items():
                log_result(model=f"probe[{k}]", metric="site_id_accuracy" if not k.startswith("domain") else
                           "domain_id_accuracy", value=r["accuracy"], split="neon_dev/random5_probe",
                           n=r["n"], config=cfg, extra=r, notes="chance = chance_uniform / chance_majority in extra",
                           **common)
                run.metric(f"acc_{k}", r["accuracy"])
            (out / "site_probe.json").write_text(json.dumps(res, indent=1))
            log.info("probe: %s", {k: round(v["accuracy"], 3) for k, v in res.items()})


if __name__ == "__main__":
    main()
