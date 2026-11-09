#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X BioSet configurations on the development folds (protocol §7, amendment v4.9) + per-fold UK pretraining.

Configurations (``sentinel/models/bioassess/xbioset.py::CONFIGS``; each counted once per block):
``bsx_flat_geo`` (multi-source, source + analysis-method tokens, geography), ``bsx_flat_geo_rloss`` (+ region
anomaly loss), ``bsx_flat_geo_ukpt`` (+ per-fold UK pretraining, encoder via ``xtrack.load_pretrained_encoder``),
``bsx_flat_geo_same`` (same-continent rows only). Seed 0 (amendment v4.3).

Every fold's fit/validation/pretraining ids pass ``xtrack.assert_fit_ids`` (in ``xbioset.fit_fold``; re-checked
here). Folds are saved one by one, so a ≤ 2 h GPU chunk (``--max-minutes``) stops cleanly and the next continues.

    python -m sentinel.core.gpu --name m5x_bioset_dev --min-free-mb 4000 -- \
        python -B scripts/train/bioassess_x_bioset.py --block X-U --config bsx_flat_geo --max-minutes 105
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from sentinel.core.registry import file_sha256
from sentinel.core.runlog import TrainingRun
from sentinel.models.bioassess import xbioset as XB
from sentinel.models.bioassess import xdev as XD
from sentinel.models.bioassess import xtrack as X

FOLDS = XD.DEV_DIR / "folds_dev_v1.json"
PARTS = XD.DEV_DIR / "bioset_parts"
PRETRAIN = XD.DEV_DIR / "pretrain"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(a) -> int:
    import torch

    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    S, frozen, D = XD.load_context()
    folds = XD.read_folds(FOLDS, S, frozen)[a.block]
    prep = XB.Prepared(D)
    out_dir = PARTS / a.block / a.config
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rc = {"model": a.config, "block": a.block, "cfg": XB.base_cfg(a.config), "spec": XB.CONFIGS[a.config], "seed": 0,
          "folds_sha256": file_sha256(FOLDS), "device": device}
    n_have = 0
    with TrainingRun("bioassess", f"bioassess_x_dev/{a.block}/{a.config}", config=rc,
                     data=["nars_streams", "nars_lakes", "uk_ea_ecology", "taxonomy_harmonised"],
                     plan_ref="bioassess_x_protocol.md §7, amendment v4.9", iteration="dev",
                     artifacts_dir=str(out_dir)) as tr_run:
        done = 0
        for k, fo in enumerate(folds):
            f = out_dir / f"fold{k:02d}.parquet"
            if f.exists():
                continue
            if (time.time() - t_start) / 60 > a.max_minutes:
                tr_run.note(f"chunk stopped before fold {k} (time budget)")
                break
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
            t0 = time.time()
            preds, diag = XB.fit_fold(prep, a.config, a.block, fo, S, frozen, seeds=(0,), device=device,
                                      pretrain_root=PRETRAIN, fold_index=k, log=log)
            P = preds[0]
            P["fold"] = fo.g
            P.to_parquet(f)
            (out_dir / f"fold{k:02d}_diag.json").write_text(json.dumps(diag, default=float))
            done += 1
            log(a.block, a.config, k, fo.g, f"{(time.time() - t0) / 60:.1f} min", f"best_epoch={diag[0]['best_epoch']}",
                f"n_fit={diag[0]['n_fit']}")
        tr_run.metric("folds_done_this_chunk", done)
        n_have = len(list(out_dir.glob("fold[0-9][0-9].parquet")))
        tr_run.metric("folds_total_done", n_have)
    if n_have == len(folds):
        P = pd.concat([pd.read_parquet(out_dir / f"fold{k:02d}.parquet") for k in range(len(folds))])
        variant = "same" if XB.CONFIGS[a.config].get("same_continent") else "multi"
        f = XD.DEV_DIR / "preds" / a.block / f"{a.config}__{variant}__s0.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        P.to_parquet(f)
        log("complete", f)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=XD.BLOCKS)
    ap.add_argument("--config", required=True, choices=sorted(XB.CONFIGS))
    ap.add_argument("--max-minutes", type=float, default=105)
    ap.add_argument("--cpu", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
