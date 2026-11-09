# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Pretrain the SENTINEL-Lite spectral branch on GLORIA (NAIP ADS100 simulated bands); CPU, 8 threads.

    python -B scripts/train/lite_gloria_pretrain.py   # -> $SENTINEL_RUNS/lite/gloria_pretrain_<stamp>/spectral.pt
"""

from __future__ import annotations

import numpy as np

from sentinel.models.lite import runner as R

R.cpu_threads(6)

import torch  # noqa: E402

from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import gloria as GL  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402


def main() -> int:
    split = G.load_split()
    tr = D.role_frame(split, ("train",), training=True)
    y = tr[[f"y_{t}" for t in D.TARGETS]].to_numpy(float)
    ym, ys = np.nanmean(y, 0), np.nanstd(y, 0)
    cfg = {"steps": 4000, "batch": 256, "lr": 3e-3, "scale": "4*pi*Rrs", "bands": "naip_ads100 top-hat",
           "split_sha256": split["_sha256"]}
    run_dir = R.run_directory("gloria_pretrain")
    with R.training_run("gloria_pretrain", cfg, run_dir) as run:
        br, info = GL.pretrain(ym, ys, steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
                               log=lambda s: (print(s), run.note(s)))
        torch.save({"spectral": br.state_dict(), "y_mean": ym.tolist(), "y_std": ys.tolist(), "info": info},
                   run_dir / "spectral.pt")
        R.write_json(run_dir / "info.json", info)
        print(run_dir / "spectral.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
