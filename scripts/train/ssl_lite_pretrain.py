#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SOTA lever 2: SimSiam pretraining of the Lite encoder on the unlabelled NAIP SSL corpus (manifest
``naip_ssl_corpus``). It runs on one GPU of the shared budget (0/1, via the lock helper), with jobs ≤ 2 h; it
resumes from ``state.pt`` until ``done``. Output: ``$SENTINEL_RUNS/lite_ssl/<run>/ssl_encoder.pt``, which holds
the encoder state, the corpus band stats and the config.

    python -m sentinel.core.gpu --name lite_ssl --min-free-mb 6000 -- \\
        python -B scripts/train/ssl_lite_pretrain.py --run simsiam_v1 --epochs 40 --max-minutes 105
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.data import chips as C  # noqa: E402
from sentinel.models.litessl.simsiam import train_simsiam  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-minutes", type=float, default=105.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    out = paths.ensure(paths.RUNS_ROOT / "lite_ssl" / (("smoke_" if a.smoke else "") + a.run))
    store = C.ChipStore(paths.processed("naip_chips") / "ssl")
    ids = store._ok.index.astype(str).tolist()
    if a.smoke:
        ids = ids[:512]
    chips = np.stack([store.load(i) for i in ids]).astype(np.uint8)
    if chips.shape[1] != 4:
        chips = np.moveaxis(chips, -1, 1)
    state = torch.load(out / "state.pt", map_location="cpu", weights_only=False) if (out / "state.pt").exists() else None
    cfg = {"method": "simsiam", "backbone": "mnv4s", "epochs": 2 if a.smoke else a.epochs, "batch": a.batch,
           "seed": a.seed, "n_chips": len(ids), "corpus_manifest": "naip_ssl_corpus", "smoke": a.smoke}
    with TrainingRun("M7-lite", f"{'SMOKE/' if a.smoke else ''}lite_ssl/{a.run}", config=cfg, data=["naip_ssl_corpus"],
                     plan_ref="docs/handoff/M7-lite.md SOTA plan lever 2", artifacts_dir=str(out)) as run:
        st = train_simsiam(chips, epochs=cfg["epochs"], batch=a.batch, seed=a.seed, max_minutes=a.max_minutes,
                           state=state, log=print)
        torch.save(st, out / "state.pt")
        run.metric("epochs_done", st["epoch"])
        run.metric("final_loss", st["history"][-1]["loss"])
        if st["done"]:
            torch.save({"encoder": st["encoder"], "x_mean": st["x_mean"], "x_std": st["x_std"], "config": cfg,
                        "history": st["history"]}, out / "ssl_encoder.pt")
            run.note("SSL complete; encoder saved")
    (out / "status.json").write_text(json.dumps({"epoch": st["epoch"], "done": st["done"]}))
    print(json.dumps({"epoch": st["epoch"], "done": st["done"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
