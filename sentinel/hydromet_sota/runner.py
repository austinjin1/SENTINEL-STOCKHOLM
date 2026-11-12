# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Plumbing for the M6 SOTA drivers (``scripts/{train,eval}/sota_hydromet_*.py``): lever configs over the frozen base
recipe, run directories under ``$SENTINEL_RUNS/hydromet/sota/``, GPU locks (≥ 10,000 MiB physically free), the
TrainingRun record and the shared configuration counter."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..core import paths
from ..data.reserved import assert_no_reserved as shared_assert_no_reserved
from ..models.hydromet import runner as HR
from . import reserved as R

WORKSTREAM = "M6-hydromet"
PLAN_REF = "docs/preregistration/hydromet_sota_protocol.md (M6 SOTA levers)"
ROOT = paths.RUNS_ROOT / "hydromet" / "sota"
MIN_FREE_MB = 10000
EXIT_RESUME = 75


def load_lever_config(path) -> dict:
    """Lever YAML over the base config (``base_config``, merged with ``configs/hydromet/common.yaml``) and the inner
    selection (``hparams_from``): returns a flat config with ``ealstm`` = base ``ealstm`` ∪ selected grid point."""
    import yaml

    p = Path(path)
    lev = yaml.safe_load(p.read_text())
    base = HR.load_config(paths.REPO_ROOT / lev["base_config"])
    sel_p = paths.RUNS_ROOT / lev["hparams_from"]
    sel = json.loads(sel_p.read_text())
    cfg = {k: v for k, v in base.items() if k not in ("grid", "smoke", "seeds", "model", "iteration")}
    cfg["ealstm"] = {**base["ealstm"], **sel["selected"]}
    cfg.update({k: v for k, v in lev.items()})
    cfg["_config_file"] = str(p)
    cfg["_smoke"] = False
    cfg["hparams_selected"] = sel["selected"]
    if cfg.get("upstream_forcing") != "append":
        raise ValueError("lever configs use the ealstm_ws input set (upstream_forcing: append)")
    return cfg


def assert_training_sites_clean(sites, where: str) -> None:
    """Refuse an M1-reserved site in a lever training / tuning / selection set (shared helper and the M6 guard;
    same pinned 729-site list). Called by every lever training script on the loaded location set before fitting."""
    sites = [str(s) for s in sites]
    shared_assert_no_reserved(sites, where=where)
    R.assert_no_reserved([s.split(":", 1)[-1] for s in sites])


def run_dir(name: str) -> Path:
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def acquire_gpu(name: str) -> int:
    from ..core import gpu as G

    need = max(MIN_FREE_MB, int(os.environ.get("SENTINEL_GPU_MIN_FREE_MB", MIN_FREE_MB)))
    return G.acquire_gpu(name, min_free_mb=need)


def training_run(model: str, cfg: dict, rdir: Path, data=("sonde_daily", "forcings_daily", "streamcat",
                                                            "hydromet_upstream_forcing", "hydromet_sota")):
    from ..core.runlog import TrainingRun

    return TrainingRun(WORKSTREAM, f"SOTA/{model}", config=HR.public_config(cfg), data=list(data), plan_ref=PLAN_REF,
                       iteration=f"sota-lever{cfg.get('lever', '?')}", artifacts_dir=str(rdir))


def log_config_tried(model: str, cfg: dict, rdir: Path, status: str = "started") -> None:
    HR.log_config_tried(model, cfg, rdir, status=status)


def write_json(path, obj) -> None:
    HR.write_json(path, obj)
