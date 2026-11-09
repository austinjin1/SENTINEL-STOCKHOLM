# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chronos-2 fine-tuning for SENTINEL-Sonde through the official ``Chronos2Pipeline.fit()``
(chronos-forecasting 2.3.2; ``finetune_mode`` 'full' (default) or 'lora' (ablation; needs ``peft``,
pinned 0.21.2 — if it were missing, 'lora' raises instead of silently falling back to full).

Training inputs: one PreparedInput per train-role site ('multi': all co-located variables as
targets [+ climatology known-future covariates]) or per (site, variable) ('uni'), each truncated
to the TRAIN block (first..last observation <= 2021-09-30), so every randomly sliced training
target lies in WY2008–2021. Validation inputs (checkpoint selection inside fit(): eval loss every
``eval_steps``, best checkpoint restored): windows of context + 14 days whose 14-day target window
ends at cut dates in the VALIDATION block (issue date >= val start + 42-day embargo), for val-role
and train-role sites. Evaluation then reuses :class:`.chronos.ChronosForecaster` on the common rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ...data.sonde_splits import TIME_BLOCKS
from .chronos import ChronosConfig, clim_track
from .common import SondeData
from .data import to_model_space


@dataclass
class ChronosFTConfig:
    mode: str = "multi"
    context: int = 512
    future_clim: bool = True
    finetune_mode: str = "full"
    learning_rate: float = 1e-5
    num_steps: int = 2000
    batch_size: int = 128
    eval_steps: int = 200
    val_cut_every_days: int = 28
    val_max_sites: int | None = 300
    min_past: int = 60
    min_obs: int = 120
    fp16: bool = False
    weather_covariates: bool = False
    inner_val: bool = False          # select checkpoints on HELD-OUT TRAIN-role basin groups (not val sites)
    inner_val_frac: float = 0.15
    seed: int = 0
    logging_steps: int = 50

    def inference_config(self, batch_size: int = 1024) -> ChronosConfig:
        return ChronosConfig(mode=self.mode, context=self.context, future_clim=self.future_clim,
                             batch_size=batch_size, weather_covariates=self.weather_covariates)


def _site_series(data: SondeData, s: int, variables: list[str], t0: int, t1: int) -> np.ndarray:
    return np.stack([to_model_space(v, data.cube.y[s, t0:t1, data.cube.var_index(v)]) for v in variables]
                    ).astype(np.float32)


def _prepared(target: np.ndarray, cov: np.ndarray | None, pred_len: int, past: np.ndarray | None = None) -> dict:
    """rows = [targets | past-only covariates (lagged weather) | known-future covariates (climatology)]."""
    import torch

    k = target.shape[0]
    past = past if past is not None else np.zeros((0, target.shape[1]), np.float32)
    nfut = 0 if cov is None else cov.shape[0]
    parts = [target, past] + ([cov] if cov is not None else [])
    ctx = np.concatenate(parts, axis=0)
    return {"context": torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.float32)),
            "future_covariates": torch.full((ctx.shape[0], pred_len), float("nan")),
            "n_targets": k, "n_covariates": past.shape[0] + nfut, "n_future_covariates": nfut}


def _weather_rows(data: SondeData, s: int, t0: int, t1: int) -> np.ndarray | None:
    if data.weather is None or not np.isfinite(data.weather[s, :, 0]).any():
        return None
    w = data.weather[s, t0:t1].astype(np.float32).T.copy()                 # (W, L), lagged observed
    w[0] = np.log1p(np.maximum(w[0], 0))
    return w


def site_variables(data: SondeData, s: int) -> list[str]:
    return [v for v in data.cube.variables if np.isfinite(data.clim_mean[s, 0, data.cube.var_index(v)])]


def inner_split(data: SondeData, cfg: ChronosFTConfig) -> tuple[np.ndarray, np.ndarray]:
    """Train-role sites split by BASIN GROUP into (fit sites, inner-validation sites), seeded."""
    tr = data.site_idx(("train",))
    if not cfg.inner_val:
        return tr, np.array([], dtype=int)
    grp = np.array([str(data.split.basin_group.get(data.sites[s], data.sites[s])) for s in tr])
    ug = np.unique(grp)
    rs = np.random.default_rng(cfg.seed + 7919)
    held = set(rs.choice(ug, max(1, int(round(cfg.inner_val_frac * ug.size))), replace=False))
    m = np.array([g in held for g in grp])
    return tr[~m], tr[m]


def build_train_inputs(data: SondeData, cfg: ChronosFTConfig, pred_len: int = 14) -> tuple[list, dict]:
    end = int(np.searchsorted(data.cube.dates, pd.Timestamp(TIME_BLOCKS["train"][1]), side="right"))
    inputs, n_sites = [], 0
    fit_sites, _ = inner_split(data, cfg)
    for s in fit_sites:
        vars_s = site_variables(data, s)
        if not vars_s:
            continue
        z = _site_series(data, s, vars_s, 0, end)                     # TRAIN block only
        obs_any = np.isfinite(z).any(axis=0)
        if obs_any.sum() < cfg.min_obs:
            continue
        a, b = int(np.argmax(obs_any)), int(end - np.argmax(obs_any[::-1]))  # first..last obs
        groups = [list(range(len(vars_s)))] if cfg.mode == "multi" else [[j] for j in range(len(vars_s))]
        for g in groups:
            tgt = z[g, a:b]
            if np.isfinite(tgt).sum() < cfg.min_obs:
                continue
            cov = None
            if cfg.future_clim:
                ti = np.array([a], dtype=np.int64)
                si = np.array([s], dtype=np.int64)
                cov = np.concatenate([clim_track(data, vars_s[j], si, ti, 0, b - a) for j in g], axis=0)
            past = _weather_rows(data, s, a, b) if cfg.weather_covariates else None
            inputs.append(_prepared(tgt, cov, pred_len, past))
        n_sites += 1
    return inputs, {"n_inputs": len(inputs), "n_sites": n_sites,
                    "last_train_date": str(data.cube.dates[end - 1].date())}


def build_val_inputs(data: SondeData, cfg: ChronosFTConfig, pred_len: int = 14, seed: int = 0) -> tuple[list, dict]:
    """Windows (context + pred_len days) whose target window ends at cut dates in the val block —
    or, with ``inner_val``, at cut dates in WY2020–21 (issue >= 2019-10-01 + 42 d, inside the TRAIN
    block) for the HELD-OUT train-role basin groups only (no val data at all)."""
    C = cfg.context
    if cfg.inner_val:
        t_tr = data.issue_idx("train", pred_len)
        t_issue = t_tr[data.cube.dates[t_tr] >= pd.Timestamp("2019-10-01") + pd.Timedelta(days=42)]
        _, sites = inner_split(data, cfg)
    else:
        t_issue = data.issue_idx("val", pred_len)            # issue dates with targets inside val
        sites = data.site_idx(("train", "val"))
    cuts = t_issue[::cfg.val_cut_every_days] + pred_len      # index of the last target day
    if cfg.val_max_sites and sites.size > cfg.val_max_sites:
        rs = np.random.default_rng(seed)
        # keep every val-role site, subsample train-role sites
        vs = sites[data.roles[sites] == "val"]
        ts = sites[data.roles[sites] == "train"]
        n_t = max(cfg.val_max_sites - vs.size, 0)
        sites = np.sort(np.concatenate([vs, rs.choice(ts, min(n_t, ts.size), replace=False)]))
    inputs = []
    for s in sites:
        vars_s = site_variables(data, s)
        if not vars_s:
            continue
        for c in cuts:
            t0 = c + 1 - (C + pred_len)
            if t0 < 0:
                continue
            z = _site_series(data, s, vars_s, t0, c + 1)
            fut_obs = np.isfinite(z[:, -pred_len:]).any(axis=1)
            ctx_obs = np.isfinite(z[:, :-pred_len]).any(axis=1)
            groups = [list(range(len(vars_s)))] if cfg.mode == "multi" else [[j] for j in range(len(vars_s))]
            for g in groups:
                if not (fut_obs[g].any() and ctx_obs[g].any()):
                    continue
                cov = None
                if cfg.future_clim:
                    si, ti = np.array([s]), np.array([t0], dtype=np.int64)
                    cov = np.concatenate([clim_track(data, vars_s[j], si, ti, 0, C + pred_len) for j in g], axis=0)
                past = _weather_rows(data, s, t0, c + 1) if cfg.weather_covariates else None
                inputs.append(_prepared(z[g], cov, pred_len, past))
    return inputs, {"n_inputs": len(inputs), "n_sites": int(sites.size), "n_cuts": int(cuts.size),
                    "first_issue": str(data.cube.dates[cuts.min() - pred_len].date()) if cuts.size else None}


def finetune(pipeline, data: SondeData, cfg: ChronosFTConfig, output_dir, *, pred_len: int = 14, log=print):
    """Run the official fit(); returns (fine-tuned pipeline, info dict)."""
    if cfg.finetune_mode == "lora":
        from transformers.utils.import_utils import is_peft_available

        if not is_peft_available():
            raise RuntimeError("finetune_mode='lora' needs the `peft` package (not installed in sentinel2); "
                               "use finetune_mode='full' or install+pin peft first")
    tr, tr_info = build_train_inputs(data, cfg, pred_len)
    va, va_info = build_val_inputs(data, cfg, pred_len, seed=cfg.seed)
    log(f"[chronos-ft] train inputs {tr_info}; val inputs {va_info}")
    if not tr or not va:
        raise RuntimeError("empty fine-tuning train or validation inputs")
    extra = {"eval_steps": cfg.eval_steps, "save_steps": cfg.eval_steps, "logging_steps": cfg.logging_steps,
             "seed": cfg.seed, "data_seed": cfg.seed, "fp16": bool(cfg.fp16), "report_to": "none"}
    ft = pipeline.fit(inputs=tr, prediction_length=pred_len, validation_inputs=va,
                      finetune_mode=cfg.finetune_mode, context_length=cfg.context,
                      learning_rate=cfg.learning_rate, num_steps=cfg.num_steps, batch_size=cfg.batch_size,
                      output_dir=str(output_dir), min_past=cfg.min_past, remove_printer_callback=True,
                      **extra)
    return ft, {"train": tr_info, "val": va_info, "config": asdict(cfg)}
