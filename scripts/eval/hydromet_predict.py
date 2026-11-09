#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Predict with SAVED SENTINEL-HydroMet artifacts (baseline pickles / EA-LSTM checkpoints, keyed
``<model>_s<seed>``) — no fitting. This is the only path to test-role sites or test/test_b blocks: it
needs --allow-test AND a valid docs/preregistration/FREEZE_hydromet.json, and every artifact's sha256
must equal the frozen ``weights_sha256[name]``. Logic: sentinel.models.hydromet.pipeline.

    python -B scripts/eval/hydromet_predict.py --run <run_dir> [<run_dir> ...] --block val --roles val
    python -B scripts/eval/hydromet_predict.py --run <dirs> --block test --roles test --allow-test --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sentinel.models.hydromet import runner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="run dirs (config.json + models/*.pkl or *_best.pt)")
    ap.add_argument("--block", default="val")
    ap.add_argument("--roles", default="val", help="comma list of site roles to predict")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--freeze-file", default=None)
    ap.add_argument("--gpu", action="store_true", help="lock one GPU (prefer 3) for EA-LSTM inference")
    ap.add_argument("--out", required=True, help="output directory for the prediction files")
    a = ap.parse_args()
    if a.allow_test or a.block in ("test", "test_b") or "test" in a.roles:
        raise SystemExit("protected data (test blocks / test-role sites) are scored ONLY through "
                         "scripts/eval/hydromet_test_look.py (single-pass guard; models, comparators and the Holm family "
                         "come from the freeze block)")
    runner.env_threads(8)
    cfgs = [json.loads((Path(r) / "config.json").read_text()) for r in a.run]
    lags = {runner.check_mode(c) for c in cfgs}
    abl = {c.get("static_ablation") for c in cfgs}
    if len(lags) != 1 or len(abl) != 1 or len({bool(c.get("_smoke")) for c in cfgs}) != 1:
        raise ValueError("runs differ in forcing lag / static ablation / smoke flag; predict them separately")
    cfg = cfgs[0]
    roles = tuple(r.strip() for r in a.roles.split(",") if r.strip())
    if a.gpu:
        runner.setup_gpu("hydromet_predict")
    from sentinel.models.hydromet import guard, pipeline

    # the same guarded preparation as training/scoring (FINAL split only, smoke = train-role sites, static
    # ablation, inner groups), but never in training mode
    args = argparse.Namespace(eval_blocks=a.block, split=cfg.get("split"), allow_test=a.allow_test,
                              freeze_file=a.freeze_file)
    cfg = dict(cfg)
    log_dir = Path(a.out)
    data, _, _ = runner.prepare(args, cfg, log_dir, training=False, roles=roles)
    access = data.access
    if cfg.get("_smoke") and (access["protected"] or set(roles) - {"train", "inner"}):
        raise guard.ProtectedDataError("smoke artifacts are only applied to train-role sites")
    artifacts = pipeline.discover_artifacts(a.run)
    si, ti = data.site_idx(roles), np.flatnonzero(data.block_mask(a.block))
    pipeline.predict_artifacts(data, artifacts, si, ti, a.block, a.out, lag=lags.pop(), access=access,
                               smoke=bool(cfg.get("_smoke")), device="cuda" if a.gpu else "cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
