#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Late-fusion ensembles of existing out-of-fold predictions (protocol §10.1, cycle-1 item 3).

The ensemble's OOF prediction is the unweighted mean of its members' OOF predictions (each member
was itself trained only on the training rows of every fold), so no new fitting happens and no
held-out information enters. Scored and logged exactly like microbe_cv.py (experiment
'microbe_phase1_cv', model 'Ens[<a>+<b>]').

    python -B scripts/train/microbe_ensemble.py --members seqset_kmer ridge_clr --schemes lodo
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import microbe_cv as MCV

from sentinel.core.runlog import TrainingRun


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--schemes", nargs="+", default=["lodo"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--iteration", default="cycle1")
    a = ap.parse_args(argv)
    from sentinel.models.microbe.baselines import LABELS
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    targets = list(TARGET_NAMES)
    name = "ens_" + "+".join(a.members)
    label = "Ens[" + "+".join(LABELS.get(m, f"SeqSet[{m}]") for m in a.members) + "]"
    for scheme in a.schemes:
        design = load_design(with_release_v1=scheme.startswith("release"))
        out = MCV.cv_dir(scheme, a.tag)
        parts = [pd.read_parquet(out / f"{m}.parquet")[targets] for m in a.members]
        idx = parts[0].index
        for p in parts[1:]:
            if not p.index.sort_values().equals(idx.sort_values()):
                raise ValueError("members were scored on different rows")
        oof = sum(p.reindex(idx) for p in parts) / len(parts)
        cfg = {"model": name, "members": a.members, "scheme": scheme, "rule": "unweighted mean of member OOF"}
        with TrainingRun(MCV.WORKSTREAM, f"{MCV.EXPERIMENT}:{name}:{scheme}", config=cfg, data=MCV.DATA_MANIFESTS,
                         plan_ref="microbe_protocol §10.1", iteration=a.iteration, artifacts_dir=str(out)) as run:
            oof.to_parquet(out / f"{name}.parquet")
            ref_p = out / "context_rf.parquet"
            ref = pd.read_parquet(ref_p) if ref_p.exists() else None
            res = MCV.score_and_log(design, oof, name=name, label=label, scheme=scheme, targets=targets,
                                    is_baseline=False, config=cfg, n_boot=a.n_boot, per_seed=None,
                                    artifacts=[str(out / f"{name}.parquet")], run=run, ref=ref,
                                    extra_notes="late fusion of OOF predictions")
            (out / f"{name}_scores.json").write_text(json.dumps(res, indent=1, default=float))
            print(scheme, name, {t: round(r["r2"], 3) for t, r in res.items()})


if __name__ == "__main__":
    main()
