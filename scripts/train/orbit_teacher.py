#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Export the SENTINEL-Orbit teacher for M7 (docs/model_cards/orbit.md): write the teacher manifest
docs/model_cards/orbit_teacher.json (checkpoint paths + sha256 + config label) and teacher predictions for a
chip store (default processed/s2_chips/naip_teacher) to $SENTINEL_RUNS/orbit/teacher/<store>_teacher_preds.parquet.
Imagery only: no in-situ labels are read. Each chip row carries its split role/block (orbit split v3) so M7 can
hold out the same lakes.

    python -B scripts/train/orbit_teacher.py --satwq satwq_base_s0 satwq_base_s1 satwq_base_s2 --label satwq_base
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit import split as SP  # noqa: E402
from sentinel.models.orbit.guard import artifact_sha256  # noqa: E402
from sentinel.models.orbit.teacher import TEACHER_MANIFEST, OrbitTeacher  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--satwq", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--store", default="naip_teacher")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cpu", action="store_true", help="CPU inference (2 threads), no GPU lock")
    a = ap.parse_args()
    ckpts = [str(paths.RUNS_ROOT / "orbit" / r / "model.pt") for r in a.satwq]
    man = {"label": a.label, "created": _dt.datetime.now().isoformat(timespec="seconds"), "checkpoints": ckpts,
           "sha256": [artifact_sha256(c) for c in ckpts], "split_sha256": SP.split_sha256(),
           "note": "SatWQ-Net seed ensemble trained on the fit cell (train role x <= 2021) of orbit dev data; "
                   "development teacher (validation only, test not scored)"}
    TEACHER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TEACHER_MANIFEST.write_text(json.dumps(man, indent=1))
    if not a.cpu:
        PL.setup_gpu(f"orbit_teacher_{a.label}")
    import torch

    torch.set_num_threads(2)
    t = OrbitTeacher.from_manifest(device="cpu" if a.cpu else None)
    store = paths.processed("s2_chips") / a.store
    res = t.predict_store(store, workers=a.workers)
    idx = pd.read_parquet(store / "index.parquet") if (store / "index.parquet").exists() else None
    if idx is not None:
        idx = idx[idx.status == "ok"].set_index("chip_id").loc[res.chip_id]
        sp = SP.assign(idx.lon.to_numpy(), idx.lat.to_numpy(),
                       pd.to_datetime(idx.sensing_time_utc, utc=True).dt.tz_localize(None).to_numpy())
        for c in ("waterbody", "in_lake", "block", "role", "tblock", "buffer_excluded"):
            res[f"orbit_{c}"] = sp[c].to_numpy()
    out = paths.ensure(paths.RUNS_ROOT / "orbit" / "teacher") / f"{a.store}_teacher_preds.parquet"
    res.to_parquet(out, index=False)
    man["predictions"] = {a.store: {"path": str(out), "sha256": artifact_sha256(out), "n_chips": int(len(res))}}
    TEACHER_MANIFEST.write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
