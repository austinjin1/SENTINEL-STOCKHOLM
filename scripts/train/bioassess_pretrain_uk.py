#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Pretrain the BioSet encoder on the UK EA invertebrate pool (masked taxa + community->chemistry).

Reserved UK test blocks are excluded (sentinel.models.bioassess.uk); NRSA contributes only taxon
*names* of the development cycles to the vocabulary (no NRSA labels are used). Writes
``<runs>/bioassess/uk_pretrain_<stamp>/encoder.pt`` and the pointer ``<runs>/bioassess/uk_pretrain_latest.json``
used by candidate ``bioset_st_ukpt`` (``pretrained: auto``). Wrapped in TrainingRun.

    python -B scripts/train/bioassess_pretrain_uk.py [--candidate bioset_st_ukpt] [--gpu-prefer 1]
"""

from __future__ import annotations

import os

# CPU budget (docs/CONTRIBUTING_V2.md): <= 6 threads per job, one CPU-heavy job per workstream. BLAS pools are pinned to 1 thread
# before numpy is imported; model-level parallelism (RF/LightGBM n_jobs, joblib, torch) is capped
# at MAX_THREADS below.
MAX_THREADS = 6   # CONTRIBUTING_V2 "Shared-host allocation" (2026-10-24 22:30)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["SENTINEL_GPUS"] = "4"   # M5 uses only the shared queue GPU 4 (CONTRIBUTING_V2)

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger("bioassess_pretrain_uk")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs/bioassess/bioset.yaml"))
    ap.add_argument("--candidate", default="bioset_st_ukpt")
    ap.add_argument("--gpu-prefer", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    import yaml

    torch.set_num_threads(2)

    device = "cpu"
    if not a.cpu:
        from sentinel.core.gpu import acquire_gpu

        g = acquire_gpu("bioassess_pretrain_uk", prefer=a.gpu_prefer)
        log.info("GPU %d", g)
        device = "cuda"
    from sentinel.core.runlog import TrainingRun
    from sentinel.models.bioassess.cv import load_bioset_config
    from sentinel.models.bioassess.data import load_nrsa
    from sentinel.models.bioassess.pretrain import pretrain_uk
    from sentinel.models.bioassess.uk import load_uk

    cfg = load_bioset_config(a.config, a.candidate)
    pcfg = dict(yaml.safe_load(Path(a.config).read_text())["pretrain_uk"])
    if a.epochs is not None:
        pcfg["epochs"] = a.epochs
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = paths.run_dir("bioassess", f"uk_pretrain_{stamp}")
    uk = load_uk()
    nrsa = load_nrsa()          # development cycles only (holdout guard)
    log.info("UK pool %s", uk.info)
    config = {"candidate": a.candidate, "encoder_cfg": cfg, "pretrain_cfg": pcfg, "seed": a.seed,
              "uk_info": uk.info, "nrsa_vocab_surveys": nrsa.info["surveys"], "device": device}
    with TrainingRun("bioassess", "uk_pretrain_bioset", config=config, data=["uk_ea_ecology", "taxonomy_harmonised",
                     "nars_streams"], plan_ref="RESEARCH_PLAN.md §5.5; docs/preregistration/bioassess_protocol.md §6",
                     artifacts_dir=str(run_dir)) as run:
        ck, hist = pretrain_uk(cfg, pcfg, uk, nrsa.tokens, device=device, seed=a.seed, log=log.info)
        best = hist[ck["best_epoch"]]
        run.metric("best_epoch", ck["best_epoch"])
        run.metric("val_loss", best["val_loss"])
        run.metric("val_family_acc", best["val_family_acc"])
        for k, v in best["val_chem_r2"].items():
            run.metric(f"val_r2/{k}", v)
        run.note("UK pool validation blocks only (reserved UK test untouched); not a reported result")
        out = run_dir / "encoder.pt"
        torch.save(ck, out)
        (run_dir / "history.json").write_text(json.dumps(hist, indent=1, default=float))
        (run_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
        pointer = {"path": str(out), "sha256": file_sha256(out), "created": stamp, "candidate": a.candidate,
                   "best_epoch": ck["best_epoch"]}
        (paths.RUNS_ROOT / "bioassess" / "uk_pretrain_latest.json").write_text(
            json.dumps(pointer, indent=1))
        log.info("saved %s", pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
