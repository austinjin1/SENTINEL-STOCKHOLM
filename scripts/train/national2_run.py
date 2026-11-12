#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Run an existing Lite entry point on the national-2 split (lite_national_protocol.md N4) without editing any
script hashed by the 4-state freeze: ``national2.activate()`` patches the national-track loaders, then the original
``main`` runs unchanged with ``--track national``.

    python -B scripts/train/national2_run.py train     --config configs/lite/<cfg>.yaml --track national --device cpu
    python -B scripts/train/national2_run.py baselines --track national
    python -B scripts/train/national2_run.py compare   --student <run> --baselines <run> --subset all
    python -B scripts/train/national2_run.py targets   --manifest litesafe_national --track national --device cpu
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {"train": "scripts/train/lite_train.py", "baselines": "scripts/train/lite_baselines.py",
           "compare": "scripts/eval/lite_compare.py", "targets": "scripts/train/lite_teacher_targets.py",
           "select": "scripts/eval/lite_select.py"}


def _ssl_init_hook() -> None:
    """Config key ``ssl_init`` (path to ssl_encoder.pt): after the student is built (ImageNet init), its encoder is
    replaced by the SSL-pretrained encoder (lever 2). Patched at run time; train.py is not edited."""
    import torch

    from sentinel.core import paths
    from sentinel.models.lite import train as T

    orig = T.build_model

    def build_model(cfg):
        m = orig(cfg)
        if cfg.get("ssl_init"):
            p = Path(cfg["ssl_init"])
            p = p if p.is_absolute() else paths.RUNS_ROOT / p
            sd = torch.load(p, map_location="cpu", weights_only=False)
            m.encoder.load_state_dict(sd["encoder"], strict=True)
        return m

    T.build_model = build_model


def _context_film_hook(argv) -> None:
    """Config key ``context_film: true`` (lead ruling 2026-11-10: information parity with GBM + context). In raw input
    mode the per-chip scene parameters are used ONLY by the scene-FiLM conditioning (``train.scene_cond`` = the 8
    numbers sp/255). For such configs, ``data.scene_norm_params`` is replaced by an encoding of the context
    (lat, lon, day of year) the GBM + context baseline uses: [lat/90, lon/180, sin DOY, cos DOY, (lat/90)², (lon/180)²,
    lat·lon/(90·180), 1]·255. The model must use ``film: true, film_source: scene``. Patched only in this process.
    Limitation: the frozen ExportWrapper has no conditioning input, so an ONNX deployable of such a model needs a
    separate two-input export."""
    if "--config" not in argv:
        return
    import yaml

    cfg = yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text()) or {}
    if not cfg.get("context_film"):
        return
    if not (cfg.get("film") and cfg.get("film_source") == "scene" and cfg.get("input_mode", "raw") == "raw"):
        raise SystemExit("context_film needs film: true, film_source: scene, input_mode: raw")
    import numpy as np

    from sentinel.models.lite import data as D

    def scene_norm_params(lab, stats=None):
        lat = lab["lat"].to_numpy(float) / 90.0
        lon = lab["lon"].to_numpy(float) / 180.0
        doy = 2 * np.pi * lab["doy"].to_numpy(float) / 365.25
        f = np.stack([lat, lon, np.sin(doy), np.cos(doy), lat ** 2, lon ** 2, lat * lon, np.ones_like(lat)], 1)
        return (255.0 * f).reshape(len(lab), 2, 4).transpose(0, 2, 1).astype(np.float32)

    D.scene_norm_params = scene_norm_params


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        raise SystemExit(f"usage: national2_run.py {{{','.join(TARGETS)}}} [args]")
    from sentinel.models.lite import national2

    national2.activate()
    _ssl_init_hook()
    _context_film_hook(sys.argv)
    spec = importlib.util.spec_from_file_location("lite_entry", ROOT / TARGETS[sys.argv[1]])
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [TARGETS[sys.argv[1]]] + sys.argv[2:]
    spec.loader.exec_module(mod)
    return mod.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
