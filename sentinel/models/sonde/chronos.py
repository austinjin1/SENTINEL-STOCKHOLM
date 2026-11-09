# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chronos-2 (amazon/chronos-2, 120M, Apache-2.0) forecasts for SENTINEL-Sonde.

Two input modes, both on the common rows of :mod:`.common` (identical to the baselines):

* ``uni``   — one univariate series per (site, variable, issue date): the variable's model-space
  history over the ``context`` days ending at the issue date (inclusive; NaN = missing, handled
  natively by Chronos-2's observation mask).
* ``multi`` — one group per (site, issue date): every co-located target variable with at least
  one observation in the context is a *target* of the group (group attention shares information
  across them, i.e. each acts as a past covariate for the others); optionally each variable's
  training-years climatology is added as a *known-future covariate* (deterministic given the
  date and the training years, so legitimately known at issue time).

Quantiles: Chronos-2 predicts 21 levels (0.01, 0.05, ..., 0.95, 0.99); our 19 levels
(0.05 … 0.95) are a subset and are taken directly. Horizon h is step h of a 14-step forecast.
Inference runs in fp32 (the 2080 Ti has no bf16; T5-style models are fp16-fragile).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ...core import paths
from ..baselines.sonde_baselines import QUANTILES
from .common import HORIZONS, Rows, SondeData
from .data import to_model_space

HF_CACHE = paths.RUNS_ROOT / "hf_cache"
MODEL_ID = "amazon/chronos-2"


@dataclass
class ChronosConfig:
    model_id: str = MODEL_ID
    mode: str = "uni"            # 'uni' | 'multi'
    context: int = 512
    batch_size: int = 1024       # series (targets + covariates) per forward pass
    future_clim: bool = True     # multi mode: climatology as known-future covariates
    weather_covariates: bool = False  # multi mode: lagged OBSERVED weather as PAST-ONLY covariates
    chunk_rows: int = 50_000
    local_files_only: bool = True


def load_pipeline(model_id_or_path: str = MODEL_ID, *, device: str = "cuda", local_files_only: bool = True):
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    import torch
    from chronos import Chronos2Pipeline

    kw = {"device_map": device, "dtype": torch.float32}
    if not os.path.isdir(str(model_id_or_path)):
        kw.update({"cache_dir": str(HF_CACHE / "hub"), "local_files_only": local_files_only})
    try:
        return Chronos2Pipeline.from_pretrained(model_id_or_path, **kw)
    except TypeError:  # older transformers name for dtype
        kw["torch_dtype"] = kw.pop("dtype")
        return Chronos2Pipeline.from_pretrained(model_id_or_path, **kw)


def padded_series(z: np.ndarray, context: int) -> np.ndarray:
    """Left-pad (S, T) with context-1 NaN columns so window [t-C+1, t] = zpad[:, t : t+C]."""
    return np.concatenate([np.full((z.shape[0], context - 1), np.nan, dtype=np.float32),
                           z.astype(np.float32)], axis=1)


def windows(zpad: np.ndarray, si: np.ndarray, ti: np.ndarray, context: int) -> np.ndarray:
    """(n, C) contexts ending at issue index ti (inclusive): strictly no value after ti."""
    ar = np.arange(context)
    return zpad[si[:, None], ti[:, None] + ar[None, :]]


def clim_track(data: SondeData, v: str, si: np.ndarray, ti: np.ndarray, start: int, length: int) -> np.ndarray:
    """Training-years climatology of v for days ti+start … ti+start+length-1 (calendar DOY)."""
    import pandas as pd

    j = data.cube.var_index(v)
    base = data.cube.dates[ti]
    offs = np.arange(start, start + length)
    doy = (base.to_numpy()[:, None] + offs[None, :].astype("timedelta64[D]"))
    doy = pd.DatetimeIndex(doy.ravel()).dayofyear.to_numpy().reshape(doy.shape) - 1
    return data.clim_mean[si[:, None], doy, j].astype(np.float32)


class ChronosForecaster:
    def __init__(self, pipeline, cfg: ChronosConfig | None = None, horizons=HORIZONS):
        self.pipe = pipeline
        self.cfg = cfg or ChronosConfig()
        self.horizons = tuple(horizons)
        self.Hmax = max(self.horizons)
        levels = [round(float(q), 4) for q in pipeline.quantiles]
        self.q_idx = [levels.index(round(float(q), 4)) for q in QUANTILES]
        self.h_idx = [h - 1 for h in self.horizons]

    def _run(self, inputs) -> list:
        import torch

        with torch.no_grad():
            return self.pipe.predict(inputs, prediction_length=self.Hmax, batch_size=self.cfg.batch_size,
                                     context_length=self.cfg.context)

    def _select(self, pred: np.ndarray) -> np.ndarray:
        """(..., 21, Hmax) -> (..., H, 19) in our horizon/quantile order, sorted."""
        p = pred[..., self.q_idx, :][..., self.h_idx]           # (..., 19, H)
        return np.sort(np.swapaxes(p, -1, -2), axis=-1)

    # ------------------------------------------------------------------ univariate
    def predict_uni(self, data: SondeData, var: str, rows: Rows) -> np.ndarray:
        import torch

        C = self.cfg.context
        z = to_model_space(var, data.cube.y[:, :, data.cube.var_index(var)])
        zpad = padded_series(z, C)
        out = np.full((len(rows), len(self.horizons), len(QUANTILES)), np.nan, dtype=np.float32)
        for a in range(0, len(rows), self.cfg.chunk_rows):
            si = rows.si[a:a + self.cfg.chunk_rows].astype(np.int64)
            ti = rows.ti[a:a + self.cfg.chunk_rows].astype(np.int64)
            ctx = windows(zpad, si, ti, C)
            preds = self._run(torch.from_numpy(ctx[:, None, :]))
            p = torch.stack([x[0] for x in preds]).numpy()          # (n, 21, Hmax)
            out[a:a + si.size] = self._select(p)
        return out

    # ------------------------------------------------------------------ multivariate (grouped)
    def prepared_inputs(self, data: SondeData, s: int, ti: np.ndarray, variables: list[str]):
        """PreparedInput dicts for one site at issue indices ``ti``: rows = [targets..., known-future
        climatology covariates...]. Returns (inputs, target_vars_per_input)."""
        import torch

        C, H = self.cfg.context, self.Hmax
        n = ti.size
        si = np.full(n, s, dtype=np.int64)
        ctx_t, names = [], []
        for v in variables:
            z = to_model_space(v, data.cube.y[s:s + 1, :, data.cube.var_index(v)])
            ctx_t.append(windows(padded_series(z, C), np.zeros(n, dtype=np.int64), ti, C))  # (n, C)
            names.append(v)
        ctx_t = np.stack(ctx_t, axis=1)                             # (n, k, C)
        has = np.isfinite(ctx_t).any(axis=-1)                       # (n, k)
        covs_p, covs_f = None, None
        if self.cfg.future_clim:
            covs_p = np.stack([clim_track(data, v, si, ti, -(C - 1), C) for v in names], axis=1)   # (n, k, C)
            covs_f = np.stack([clim_track(data, v, si, ti, 1, H) for v in names], axis=1)           # (n, k, H)
        wctx = None
        if self.cfg.weather_covariates and data.weather is not None and np.isfinite(data.weather[s, :, 0]).any():
            w = data.weather[s].astype(np.float32).copy()                     # (T, W), already lagged
            w[:, 0] = np.log1p(np.maximum(w[:, 0], 0))                         # precipitation
            wctx = np.stack([windows(padded_series(w[None, :, j], C), np.zeros(n, dtype=np.int64), ti, C)
                             for j in range(w.shape[1])], axis=1)              # (n, W, C) past only
        inputs, tv = [], []
        for i in range(n):
            keep = np.flatnonzero(has[i])
            if keep.size == 0:
                inputs.append(None)
                tv.append([])
                continue
            k = keep.size
            # rows: [targets | past-only weather | known-future climatology] (Chronos2Dataset order)
            past_only = wctx[i] if wctx is not None else np.zeros((0, C), np.float32)
            nw = past_only.shape[0]
            if covs_p is not None:
                context = np.concatenate([ctx_t[i, keep], past_only, covs_p[i, keep]], axis=0)
                future = np.concatenate([np.full((k + nw, H), np.nan, np.float32), covs_f[i, keep]], axis=0)
                nfut = k
            else:
                context = np.concatenate([ctx_t[i, keep], past_only], axis=0)
                future = np.full((k + nw, H), np.nan, np.float32)
                nfut = 0
            inputs.append({"context": torch.from_numpy(np.ascontiguousarray(context, dtype=np.float32)),
                           "future_covariates": torch.from_numpy(np.ascontiguousarray(future, dtype=np.float32)),
                           "n_targets": k, "n_covariates": nw + nfut, "n_future_covariates": nfut})
            tv.append([names[j] for j in keep])
        return inputs, tv

    def predict_multi(self, data: SondeData, rows_by_var: dict[str, Rows]) -> dict[str, np.ndarray]:
        out = {v: np.full((len(r), len(self.horizons), len(QUANTILES)), np.nan, dtype=np.float32)
               for v, r in rows_by_var.items()}
        # union of (site, issue) over variables
        pos = {}
        for v, r in rows_by_var.items():
            for k, (s, t) in enumerate(zip(r.si.tolist(), r.ti.tolist())):
                pos.setdefault((s, t), []).append((v, k))
        by_site: dict[int, list[int]] = {}
        for (s, t) in pos:
            by_site.setdefault(s, []).append(t)
        site_vars = {s: [v for v in data.cube.variables if np.isfinite(data.clim_mean[s, 0, data.cube.var_index(v)])]
                     for s in by_site}
        batch_inputs, batch_keys = [], []

        def flush():
            if not batch_inputs:
                return
            preds = self._run(batch_inputs)
            for (s, t, tvars), p in zip(batch_keys, preds):
                sel = self._select(p.numpy())                     # (k, H, 19)
                for j, v in enumerate(tvars):
                    for (vv, k) in pos[(s, t)]:
                        if vv == v:
                            out[v][k] = sel[j]
            batch_inputs.clear()
            batch_keys.clear()

        for s, ts in by_site.items():
            ts = np.array(sorted(ts), dtype=np.int64)
            inputs, tv = self.prepared_inputs(data, s, ts, site_vars[s])
            for t, inp, tvars in zip(ts.tolist(), inputs, tv):
                if inp is None:
                    continue
                batch_inputs.append(inp)
                batch_keys.append((s, t, tvars))
            if len(batch_inputs) >= self.cfg.chunk_rows // 4:
                flush()
        flush()
        return out

    def predict(self, data: SondeData, rows_by_var: dict[str, Rows]) -> dict[str, np.ndarray]:
        if self.cfg.mode == "uni":
            return {v: self.predict_uni(data, v, r) for v, r in rows_by_var.items()}
        if self.cfg.mode == "multi":
            return self.predict_multi(data, rows_by_var)
        raise ValueError(f"unknown Chronos mode {self.cfg.mode!r}")
