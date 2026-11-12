# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Arms of architecture round v2 (orbit_arch_v2_amendment.md §3): configurations, target matrices, data sets and
the training of one network of a seed unit.

* **A0** (§3.3): the v1 network and recipe of the selected dev_v5 configuration, trained with EMA weights as seed
  units; four base targets; fitted on the dev_v5 rows of dev_v5m (added rows carry no base-target value).
* **A1** (§3.4): SatWQ-LakeSet-MT; c1 = with latitude/longitude, c2 = coordinate-free, c3 = no context set.

Both arms start from the same GLORIA-pretrained v1 network of the same network seed (§3.1: one initialisation
recipe); A1 transfers its pixel encoder and date pooling by name. Everything of v1 (batching, augmentation,
pretraining loop, normalisation, fit and inner cells) is imported, never copied.

No function of this module computes a validation number. The scripts that do call ``stamp.require_stamp`` first.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..orbit import data as D
from ..orbit import features as F1
from ..orbit import pipeline as PL
from ..orbit import satwq as V1
from . import lakeset as LS
from . import targets as T
from . import train as TR
from .chipfeat import GLINT_NAMES, RING_DIM

ARMS = {
    "a0": {"model": "satwq_v1", "use_set": False, "latlon": None, "new_targets": False},
    "a1_c1": {"model": "lakeset_mt", "use_set": True, "latlon": True, "new_targets": True},
    "a1_c2": {"model": "lakeset_mt", "use_set": True, "latlon": False, "new_targets": True},
    "a1_c3": {"model": "lakeset_mt", "use_set": False, "latlon": False, "new_targets": True},
}
# §3.4 item 8
LOSS_WEIGHTS = {"chla": 1.0, "secchi": 1.0, "tss": 1.0, "turbidity": 1.0, "tp": 0.5, "tn": 0.5, "doc": 0.5,
                "turbidity_fnu": 0.25, "ssc": 0.25, "acdom440": 0.25}
MIN_HEAD_FIT = 200          # a new-target head with fewer fit-cell values is dropped before any training
P_SET_DROP = 0.3
# constants an arm file must repeat exactly (a complete configuration on disk that cannot drift from the code)
FIXED = {"ema_decay": TR.EMA_DECAY, "amp": True, "p_set_drop": P_SET_DROP, "k_set": LS.K_SET,
         "p_context": LS.P_CONTEXT, "n_sab": 2, "min_head_fit": MIN_HEAD_FIT, "loss_weights": LOSS_WEIGHTS,
         "units": list(TR.UNITS), "inits": list(TR.INITS), "checkpoint_every": 5}
BASE_KEYS = ("config_file", "config_sha256", "set", "fit_scheme", "selection_registry_row")


class ArmError(RuntimeError):
    """An arm rule of the amendment would be broken."""


# ----------------------------------------------------------------------------------------- configuration
def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def _set(cfg: dict, kv: str) -> None:
    import yaml

    k, v = kv.split("=", 1)
    node = cfg
    ks = k.split(".")
    for kk in ks[:-1]:
        node = node.setdefault(kk, {})
    node[ks[-1]] = yaml.safe_load(v)


def resolve_config(arm_file: Path, *, smoke: bool = False, smoke_base: Path | None = None,
                   repo_root: Path | None = None) -> dict:
    """The complete configuration of one arm: the arm file (constants checked against the code) plus the v1
    configuration it inherits (§3.1, §3.3, §3.4: recipe of the dev_v5 configuration selected by protocol §6/§6a).

    A real run needs ``base.config_file``, its sha256, the fit scheme and the registry row of the selection in the
    arm file. A smoke test may name any v1 configuration with ``smoke_base``; it is marked as such."""
    import yaml

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    spec = yaml.safe_load(Path(arm_file).read_text())
    arm = spec.get("arm")
    if arm not in ARMS:
        raise ArmError(f"{arm_file}: unknown arm {arm!r}; arms are {sorted(ARMS)}")
    for k, v in FIXED.items():
        if spec.get(k) != v:
            raise ArmError(f"{arm_file}: {k} = {spec.get(k)!r} differs from the amendment value {v!r}")
    extra = set(spec) - set(FIXED) - {"arm", "base", "budget_minutes", "smoke"}
    if extra:
        raise ArmError(f"{arm_file}: unknown keys {sorted(extra)}")
    base = dict(spec.get("base") or {})
    if set(base) - set(BASE_KEYS):
        raise ArmError(f"{arm_file}: unknown base keys {sorted(set(base) - set(BASE_KEYS))}")
    if smoke_base is not None:
        if not smoke:
            raise ArmError("a base configuration given on the command line is for smoke tests only")
        base = {"config_file": str(smoke_base), "config_sha256": _sha(root / smoke_base), "set": [],
                "fit_scheme": "base", "selection_registry_row": None}
    else:
        missing = [k for k in ("config_file", "config_sha256", "fit_scheme", "selection_registry_row")
                   if not base.get(k)]
        if missing:
            raise ArmError(f"{arm_file}: base.{missing} not set. A0 and A1 inherit the dev_v5 configuration selected "
                           "by protocol §6/§6a; a real run starts only after that selection is registered "
                           "(a smoke test names a v1 configuration explicitly)")
    f = root / base["config_file"]
    if _sha(f) != base["config_sha256"]:
        raise ArmError(f"{f}: content differs from base.config_sha256")
    v1 = yaml.safe_load(f.read_text())
    v1_smoke = v1.pop("smoke", {}) or {}
    for kv in base.get("set") or []:
        _set(v1, kv)
    if smoke:
        v1 = _deep_update(v1, v1_smoke)
    if (v1.get("blend") or {}).get("enabled"):
        raise ArmError("the inherited configuration forms a blend; no blend is formed for a new arm (§3.1)")
    cfg = {"arm": arm, **{k: copy.deepcopy(v) for k, v in FIXED.items()}, "base": base, "v1": v1,
           "fit_scheme": base.get("fit_scheme") or "base", "budget_minutes": float(spec.get("budget_minutes", 105)),
           "smoke": bool(smoke), "arm_file": str(arm_file), "arm_file_sha256": _sha(arm_file)}
    if cfg["budget_minutes"] > 115:
        raise ArmError("one process stays below 2 h (§3.1): budget_minutes <= 115")
    return cfg


def _deep_update(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_update(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ----------------------------------------------------------------------------------------- targets
def target_matrix(df: pd.DataFrame, targets) -> tuple[np.ndarray, np.ndarray]:
    """(log10 value [N, T], censored [N, T]). A left-censored entry holds log10 of its detection limit (§2.4: it
    enters the loss through the mixture CDF). Base targets are never censored."""
    n = len(df)
    y = np.full((n, len(targets)), np.nan)
    cens = np.zeros((n, len(targets)), bool)
    for j, t in enumerate(targets):
        v = df[t].to_numpy(float)
        if t in T.SPECS:
            c = df[f"{t}_cens"].to_numpy(bool)
            dl = df[f"{t}_dl"].to_numpy(float)
            if (c & np.isfinite(v)).any():
                raise ArmError(f"{t}: a row is both quantified and censored")
            if (c & ~(dl > 0)).any():
                raise ArmError(f"{t}: a censored row has no positive detection limit")
            v = np.where(c, dl, v)
            cens[:, j] = c
        elif t not in D.TARGETS:
            raise ArmError(f"unknown target {t!r}")
        with np.errstate(divide="ignore", invalid="ignore"):
            y[:, j] = np.log10(v)
    y[~np.isfinite(y)] = np.nan
    return y, cens


def heads(df: pd.DataFrame, fit: np.ndarray, arm: str) -> tuple[list, dict]:
    """(targets of the arm's heads, {new target: fit-cell value count}). Base targets come first, in v1's order.
    A new-target head whose fit cell holds fewer than MIN_HEAD_FIT values (quantified or censored) is dropped
    (§3.4 item 8); the counts of every new target are returned."""
    if arm not in ARMS:
        raise ArmError(f"unknown arm {arm!r}")
    tg = list(D.TARGETS)
    counts = {}
    if ARMS[arm]["new_targets"]:
        for t in T.SPECS:
            n = int((np.isfinite(df[t].to_numpy(float)) | df[f"{t}_cens"].to_numpy(bool))[fit].sum())
            counts[t] = n
            if n >= MIN_HEAD_FIT:
                tg.append(t)
    return tg, counts


# ----------------------------------------------------------------------------------------- cells
def cells(df: pd.DataFrame, arm: str, fit_scheme: str, targets) -> dict:
    """Fit and inner masks. The cells are v1's (protocol §2.2). A0: dev_v5 rows only. A1: every fit-cell row with a
    value of one of its heads. The inner (early-stopping) rows are the dev_v5 rows of the inner cell for both
    arms, so that both stop on the same rows."""
    fit, inner = F1.fit_and_inner(df, fit_scheme)
    added = df["added"].to_numpy(bool)
    if ARMS[arm]["new_targets"]:
        y, _ = target_matrix(df, targets)
        fit = fit & np.isfinite(y).any(1)
    else:
        fit = fit & ~added
    return {"fit": fit, "inner": inner & ~added, "fit_cell_all": F1.fit_and_inner(df, fit_scheme)[0]}


# ----------------------------------------------------------------------------------------- inputs
def geo_matrix(df: pd.DataFrame, pix: dict, glint: np.ndarray | None) -> np.ndarray:
    """v1 geometry; with ``glint`` the three glint features of §3.4 item 5 are appended."""
    geo, _ = PL.geo_ctx(df, pix)
    if glint is None:
        return geo
    g = np.asarray(glint, float)
    if g.shape != (len(df), len(GLINT_NAMES)):
        raise ArmError(f"glint features have shape {g.shape}")
    return np.concatenate([geo, g], 1)


def ctx_matrix(df: pd.DataFrame, pix: dict, latlon: bool) -> np.ndarray:
    """DOY sin/cos; latitude/longitude (v1's unit vector) only with ``latlon`` (§3.4 item 6)."""
    _, ctx = PL.geo_ctx(df, pix)
    return (ctx if latlon else ctx[:, :2]).astype(np.float32)


def context_store(cmeta: pd.DataFrame, carrays: dict) -> V1.Batchable:
    """The context chips as a v1 Batchable (8 pixels per chip; no geometry, context or target)."""
    n = len(cmeta)
    return V1.Batchable(carrays["refl"], carrays["n_pix"], carrays["offsets"],
                        PL.lam_matrix(cmeta["platform"].to_numpy()), np.zeros((n, 1), np.float32),
                        np.zeros((n, 1), np.float32), np.zeros((n, 1), np.float32))


def make_set(arm: str, df: pd.DataFrame, pix: dict, feats: dict, ctx_cache, norm: PL.Norm, mask: np.ndarray,
             targets):
    """Batches of the rows in ``mask`` for one arm. A0: exactly v1's ``pipeline.make_set``."""
    spec = ARMS[arm]
    if spec["model"] == "satwq_v1":
        if list(targets) != list(D.TARGETS):
            raise ArmError("A0 has the four base targets only")
        return PL.make_set(df, pix, norm, mask)
    m = np.asarray(mask, bool)
    y, cens = target_matrix(df, targets)
    geo = geo_matrix(df, pix, feats["glint"])
    base = V1.Batchable(pix["refl"][m], pix["n_pix"][m], pix["offsets"][m], PL.lam_matrix(df.platform.to_numpy()[m]),
                        norm.geo(geo[m]), ctx_matrix(df, pix, spec["latlon"])[m], norm.y(y[m]))
    ring = np.nan_to_num(np.asarray(feats["ring"], np.float32)[m])
    if not spec["use_set"]:
        return LS.SetBatchable(base, feats["dp_m"][m], ring, cens=cens[m])
    cmeta, carr = ctx_cache
    return LS.SetBatchable(base, feats["dp_m"][m], ring, cstore=context_store(cmeta, carr), c_dp_m=carr["dp_m"],
                           c_q=np.nan_to_num(np.asarray(carr["q"], np.float32)), cidx=feats["cidx"][m],
                           cdt=feats["cdt"][m], cens=cens[m])


def fit_norm(arm: str, df: pd.DataFrame, pix: dict, feats: dict, fit: np.ndarray, targets) -> PL.Norm:
    """Fit-cell normalisation (v1's class). Censored values do not enter the mean and SD of a target."""
    spec = ARMS[arm]
    y, cens = target_matrix(df, targets)
    yq = np.where(cens, np.nan, y)
    n_q = np.isfinite(yq[fit]).sum(0)
    if (n_q < 2).any():
        bad = [t for t, n in zip(targets, n_q) if n < 2]
        raise ArmError(f"targets {bad} have fewer than 2 quantified fit values")
    geo = geo_matrix(df, pix, feats["glint"] if spec["model"] == "lakeset_mt" else None)
    return PL.Norm(geo[fit], yq[fit])


class SetOff:
    """A view of a SetBatchable in which every context set is empty (descriptive ablation at inference)."""

    def __init__(self, ds):
        self.ds, self.y = ds, ds.y

    def __len__(self):
        return len(self.ds)

    def batch(self, idx, device, *, train=False, rng=None, aug=None):
        return self.ds.batch(idx, device, train=train, rng=rng, aug={**(aug or {}), "set_off": True})


# ----------------------------------------------------------------------------------------- networks
def v1_arch(v1: dict) -> dict:
    return {"d": v1["d"], "n_layers": v1["n_layers"], "n_heads": v1["n_heads"], "k": v1["k"], "dropout": v1["dropout"]}


def build_v1(v1: dict) -> V1.SatWQNet:
    return V1.SatWQNet(n_geo=len(PL.GEO_COLS), n_ctx=len(PL.CTX_COLS), **v1_arch(v1))


def build_model(arm: str, v1: dict, n_targets: int):
    spec = ARMS[arm]
    if spec["model"] == "satwq_v1":
        return build_v1(v1)
    return LS.SatWQLakeSetMT(n_geo=len(PL.GEO_COLS) + len(GLINT_NAMES), n_ctx=5 if spec["latlon"] else 2,
                             n_targets=n_targets, n_ring=RING_DIM, use_set=spec["use_set"], n_sab=FIXED["n_sab"],
                             **v1_arch(v1))


def pretrain_key(v1: dict, norm_y4: tuple, seed: int, fit_scheme: str) -> str:
    """Identity of a pretrained v1 network: everything its weights depend on."""
    return config_hash({"arch": v1_arch(v1), "pretrain": v1.get("pretrain"), "ssl": v1.get("ssl") or {},
                        "y_mean": np.round(np.asarray(norm_y4[0], float), 10).tolist(),
                        "y_std": np.round(np.asarray(norm_y4[1], float), 10).tolist(), "seed": int(seed),
                        "fit_scheme": fit_scheme})[:20]


def pretrained_v1(v1: dict, norm4: PL.Norm, *, seed: int, device, fit_scheme: str, gloria, cache_dir: Path | None,
                  log=print) -> tuple[V1.SatWQNet, dict]:
    """The v1 network after v1's SSL (if the inherited configuration has it) and GLORIA stages, for one network
    seed. ``norm4``: the fit-cell normalisation of the four base targets (as v1 computes it). With ``cache_dir`` the
    result is stored and reused, so that A0 and A1 networks of the same seed start from the same weights."""
    from ...core.seed import seed_everything

    key = pretrain_key(v1, (norm4.y_mean, norm4.y_std), seed, fit_scheme)
    seed_everything(seed)                 # as scripts/train/orbit_satwq.py: seeded before the network is built
    model = build_v1(v1).to(device)
    f = None if cache_dir is None else Path(cache_dir) / f"v1_pretrained_{key}.pt"
    if f is not None and f.is_file():
        st = torch.load(f, map_location=device, weights_only=False)
        model.load_state_dict(st["state_dict"])
        log(f"  pretrained v1 network read from {f}")
        return model, {**st["info"], "reused": True}
    info = {"key": key, "seed": int(seed), "reused": False}
    sc = v1.get("ssl") or {}
    if sc.get("enabled"):
        from ..orbit import ssl as SSL

        variant = str(sc.get("pixels", "v1"))
        if fit_scheme == "track_u" and not variant.endswith("_trackU"):
            raise ArmError("track-U refits must use the track-U SSL pixel set")
        h = SSL.pretrain_ssl(model, epochs=int(sc["epochs"]), mask_frac=float(sc["mask_frac"]), lr=float(sc["lr"]),
                             batch=int(sc["batch"]), device=device, seed=seed, max_pixels=sc.get("max_pixels"),
                             variant=variant)
        info["ssl"] = {"n_pixels": h["n_pixels"], "pixels_sha256": h["pixels_sha256"],
                       "final_mse": h["history"][-1]["ssl_mse"]}
    pc = v1["pretrain"]
    if pc.get("enabled"):
        if gloria is None:
            raise ArmError("the inherited configuration pretrains on GLORIA; no GLORIA rows were given")
        gs = PL.gloria_set(gloria, norm4)
        t0 = time.time()
        h = V1.run_epochs(model, gs, inner_set=None, epochs=pc["epochs"], lr=pc["lr"], batch=pc["batch"],
                          device=device, seed=seed, aug=pc["aug"], y_mean=norm4.y_mean, y_std=norm4.y_std, log=log)
        info["pretrain"] = {"n_gloria": int(len(gloria)), "final_nll": h["history"][-1]["train_nll"],
                            "minutes": (time.time() - t0) / 60}
    if f is not None:
        TR._atomic_save({"state_dict": model.state_dict(), "info": info}, f)
    return model, info


def init_network(arm: str, v1: dict, pre: V1.SatWQNet, n_targets: int, seed: int, device):
    """The arm's network at the first fine-tuning step. A0: the pretrained v1 network. A1: a new network seeded
    with the network seed, with the pretrained pixel encoder and date pooling."""
    if ARMS[arm]["model"] == "satwq_v1":
        return pre, sorted(pre.state_dict())
    torch.manual_seed(seed)
    model = build_model(arm, v1, n_targets).to(device)
    done = model.load_v1_encoder(pre.state_dict())
    return model, done


def train_network(cfg: dict, df: pd.DataFrame, pix: dict, feats: dict, ctx_cache, *, unit: int, init: int, device,
                  gloria, run_dir: Path, pretrain_cache: Path | None, smoke_n_fit: int | None = None, log=print,
                  max_seconds: float | None = None) -> dict:
    """Train one network of a seed unit. Returns {'model', 'norm', 'targets', 'head_counts', 'cells', 'history',
    'complete', 'seed', 'pretrain'}; with ``complete`` False the time budget ran out after a checkpoint and the
    same call continues the run. Nothing here touches a row outside the fit and inner cells."""
    arm, v1, scheme = cfg["arm"], cfg["v1"], cfg["fit_scheme"]
    if unit not in TR.UNITS or init not in TR.INITS:
        raise ArmError(f"unit {unit} / init {init} outside the seed-unit design {TR.UNITS} x {TR.INITS}")
    seed = TR.network_seed(unit, init)
    t0 = time.time()
    fit_all = F1.fit_and_inner(df, scheme)[0]
    targets, head_counts = heads(df, fit_all, arm)
    cl = cells(df, arm, scheme, targets)
    fit = cl["fit"].copy()
    if smoke_n_fit:
        idx = np.flatnonzero(fit)
        fit = np.zeros_like(fit)
        fit[np.random.default_rng(0).choice(idx, min(len(idx), int(smoke_n_fit)), replace=False)] = True
    # Normalisation: the arm's whole fit cell (a smoke test trains on a subset of it but keeps these statistics).
    # The four base targets are standardised on the dev_v5 fit rows for both arms: v1's statistics, which the
    # GLORIA stage uses.
    norm4 = fit_norm("a0", df, pix, feats, cl["fit_cell_all"] & ~df["added"].to_numpy(bool), list(D.TARGETS))
    norm = fit_norm(arm, df, pix, feats, cl["fit"], targets)
    nb = len(D.TARGETS)
    if not (np.allclose(norm.y_mean[:nb], norm4.y_mean) and np.allclose(norm.y_std[:nb], norm4.y_std)):
        raise ArmError("base-target statistics of the arm's fit rows differ from those of the dev_v5 fit rows")
    tr = make_set(arm, df, pix, feats, ctx_cache, norm, fit, targets)
    inn = make_set(arm, df, pix, feats, ctx_cache, norm, cl["inner"], targets)
    pre, pinfo = pretrained_v1(v1, norm4, seed=seed, device=device, fit_scheme=scheme, gloria=gloria,
                               cache_dir=pretrain_cache, log=log)
    model, transferred = init_network(arm, v1, pre, len(targets), seed, device)
    fc = v1["finetune"]
    aug = dict(fc["aug"])
    if ARMS[arm]["model"] == "lakeset_mt":
        aug["p_set_drop"] = P_SET_DROP
        aug.pop("ctx_off", None)          # A1's context input is DOY (and lat/lon in c1): §3.4 item 6
    sel = [targets.index(t) for t in v1["select_targets"]]
    wts = None if ARMS[arm]["model"] == "satwq_v1" else np.array([LOSS_WEIGHTS[t] for t in targets], float)
    left = None if max_seconds is None else max(60.0, max_seconds - (time.time() - t0))
    h = TR.run_epochs_ema(model, tr, inner_set=inn, epochs=fc["epochs"], lr=fc["lr"], batch=fc["batch"], device=device,
                          seed=seed, aug=aug, y_mean=norm.y_mean, y_std=norm.y_std, select_idx=sel,
                          patience=fc.get("patience", 15), log=log, ema_decay=cfg["ema_decay"], loss_weights=wts,
                          ckpt=Path(run_dir) / "finetune.ckpt", ckpt_every=cfg["checkpoint_every"],
                          amp=bool(cfg["amp"]) and torch.device(device).type == "cuda", max_seconds=left)
    return {"model": model, "norm": norm, "targets": targets, "head_counts": head_counts,
            "cells": {**cl, "fit_used": fit}, "history": h, "complete": h["complete"], "seed": seed,
            "pretrain": pinfo, "n_transferred": len(transferred), "aug": aug, "sets": {"fit": tr, "inner": inn}}


def save_network(path: Path, cfg: dict, res: dict) -> None:
    arm = cfg["arm"]
    TR._atomic_save({"state_dict": res["model"].state_dict(), "norm": res["norm"].to_dict(), "config": cfg,
                     "arm": arm, "arch": {**v1_arch(cfg["v1"]), **ARMS[arm]}, "targets": list(res["targets"]),
                     "head_counts": res["head_counts"], "seed": res["seed"], "bands": list(D.BANDS),
                     "geo_cols": list(PL.GEO_COLS) + (list(GLINT_NAMES) if ARMS[arm]["model"] == "lakeset_mt" else []),
                     "best_epoch": res["history"]["best_epoch"]}, Path(path))
