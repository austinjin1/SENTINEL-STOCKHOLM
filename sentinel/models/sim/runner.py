# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing for Sim CLIs: cached data build, checkpoint I/O, simulation output files."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

from ...core import paths
from . import data as D
from .model import LATENTS, OUT_DAILY, SimModel, simulate

CACHE_DIR = paths.RUNS_ROOT / "sim" / "cache"


def _code_key() -> str:
    h = hashlib.sha256()
    root = paths.REPO_ROOT
    for rel in ("sentinel/models/sim/data.py", "sentinel/models/sim/physics.py", "sentinel/models/sim/guard.py",
                "sentinel/models/sonde/data.py", "sentinel/data/sonde_splits.py", "sentinel/models/hydromet/physics.py"):
        h.update((root / rel).read_bytes())
    for m in D.DATA_MANIFESTS:                      # processed-data versions
        f = paths.MANIFESTS / f"{m}.json"
        h.update(f.read_bytes() if f.exists() else b"-")
    cal = D.CAL_FILE
    h.update(cal.read_bytes() if cal.exists() else b"no-cal-file")
    return h.hexdigest()[:10]


def load_data(mode: str = "validation", *, allow_test: bool = False, rebuild: bool = False) -> D.SimData:
    """Build (or load the cached) full SimData for ``mode``; cache keyed by mode + data-code hash."""
    paths.ensure(CACHE_DIR)
    f = CACHE_DIR / f"simdata_{mode}_{_code_key()}.pkl"
    if f.exists() and not rebuild:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        # re-check the guard on every load (cache cannot bypass it)
        D.guard.authorise(mode, allow_test=allow_test)
        return d
    d = D.build(mode, allow_test=allow_test)
    with open(f, "wb") as fh:
        pickle.dump(d, fh, protocol=4)
    return d


def save_model(model: SimModel, cfg: dict, d: D.SimData, path: Path) -> None:
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "norm": d.norm,
                "provenance": d.provenance()}, path)


def load_model(path: Path, device="cpu", d: D.SimData | None = None) -> tuple[SimModel, dict]:
    """Load a checkpoint; if ``d`` is given, install the checkpoint's normalisation statistics in it
    (inputs must be normalised exactly as during that model's training)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if d is not None:
        d.norm = ck["norm"]
    cfg = ck["cfg"]
    m = SimModel(ck["norm"], n_dyn=len(D.DYN_NAMES), n_static=len(D.STATIC_NAMES), hidden=cfg["hidden"],
                 static_embed=cfg["static_embed"], dropout=cfg["dropout"], n_layers=cfg["n_layers"],
                 metab_struct=cfg.get("metab_struct", False), drop_dyn=cfg.get("drop_dyn", []))
    sd = dict(ck["state_dict"])
    for k, v in m.state_dict().items():      # v1 checkpoints predate the dyn_keep / sw buffers
        if k not in sd and k in ("dyn_keep", "sw_mean", "sw_std"):
            sd[k] = v
    m.load_state_dict(sd)
    return m.to(device).eval(), ck


def simulate_sites(model, d: D.SimData, idx, *, device="cpu", start: str = D.TIME_BLOCKS["train"][0]) -> dict:
    t0 = int(np.searchsorted(d.dates, np.datetime64(start)))
    return simulate(model, d, idx, t0, len(d.dates), device=device)


def save_sim(path: Path, sim: dict, d: D.SimData, idx, start: str = D.TIME_BLOCKS["train"][0]) -> None:
    t0 = int(np.searchsorted(d.dates, np.datetime64(start)))
    np.savez_compressed(path, sites=np.asarray([d.sites[i] for i in idx]), t0=t0,
                        dates=np.asarray(d.dates[t0:].values.astype("datetime64[D]")),
                        **{k: v.astype("float32") for k, v in sim.items()})


def load_sim(path: Path, d: D.SimData, keys=tuple(OUT_DAILY) + tuple(LATENTS)) -> dict:
    """Simulation file → dict var → (S_all, T_all) aligned with d.sites/d.dates (NaN elsewhere)."""
    z = np.load(path, allow_pickle=False)
    pos = {s: i for i, s in enumerate(d.sites)}
    rows = np.array([pos[s] for s in z["sites"]])
    t0 = int(z["t0"])
    out = {}
    for k in keys:
        if k not in z:
            continue
        a = np.full((len(d.sites), len(d.dates)), np.nan, dtype="float32")
        a[rows, t0:t0 + z[k].shape[1]] = z[k]
        out[k] = a
    return out


def ensemble_sims(files, d: D.SimData) -> dict:
    sims = [load_sim(f, d) for f in files]
    return {k: np.nanmean(np.stack([s[k] for s in sims]), axis=0) for k in sims[0]}
