# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chronos-2-cov: zero-shot Chronos-2 with archived NWP as KNOWN-FUTURE covariates (``sonde_arch_v2_amendment.md``
§2b; reference; the gated Chronos-2-NWP fine-tune reuses it). Subclass of the frozen-candidate forecaster (its file is
not edited): every PreparedInput gets extra known-future rows appended after the existing ones, keeping the
Chronos2Dataset row order [targets | past-only covariates | known-future covariates]. For field f the row's context
part is the LAGGED observed weather (already published at each past day) and its future part is the NWP forecast for
leads 1..H (NaN beyond the product's horizon or where missing).
"""

from __future__ import annotations

import numpy as np

from sentinel.models.sonde.chronos import ChronosForecaster, padded_series, windows

from . import nwp as NWP
from .prep import NOT_MODELLED

COV_FIELDS = ("tmean_c", "precip_mm", "srad_wm2")
WEATHER_COL = {"tmean_c": None, "precip_mm": 0, "srad_wm2": 3}     # lagged gridMET columns (tmean = (tmax+tmin)/2)


class ChronosCovForecaster(ChronosForecaster):
    def __init__(self, pipeline, cfg=None, horizons=None, *, nwp_x: np.ndarray | None = None, fields=COV_FIELDS):
        super().__init__(pipeline, cfg, horizons) if horizons is not None else super().__init__(pipeline, cfg)
        if nwp_x is not None and (not isinstance(nwp_x, np.ndarray) or nwp_x.ndim != 4):
            raise ValueError("exactly one declared NWP product tensor (S, T, L, F) is allowed")
        self.nwp_x = nwp_x            # (S, T, L, F) bias-corrected issued forecasts of the ONE declared product
        self.fields = tuple(fields)

    def _past_weather(self, data, s: int, ti: np.ndarray, C: int) -> np.ndarray:
        w = data.weather[s].astype(np.float32) if data.weather is not None else None
        rows = []
        for f in self.fields:
            if w is None:
                rows.append(np.full((ti.size, C), np.nan, np.float32))
                continue
            col = WEATHER_COL[f]
            series = (w[:, 1] + w[:, 2]) / 2 if col is None else w[:, col]
            if f == "precip_mm":
                series = np.log1p(np.maximum(series, 0))
            rows.append(windows(padded_series(series[None, :], C), np.zeros(ti.size, dtype=np.int64), ti, C))
        return np.stack(rows, axis=1)                                            # (n, k, C)

    def _future_nwp(self, s: int, ti: np.ndarray, H: int) -> np.ndarray:
        out = np.full((ti.size, len(self.fields), H), np.nan, np.float32)
        if self.nwp_x is None:
            return out
        L = min(H, self.nwp_x.shape[2])
        for j, f in enumerate(self.fields):
            v = self.nwp_x[s, ti, :L, NWP.F_IDX[f]]
            if f == "precip_mm":
                v = np.log1p(np.maximum(v, 0))
            out[:, j, :L] = v
        return out

    def prepared_inputs(self, data, s: int, ti: np.ndarray, variables: list[str]):
        import torch

        variables = [v for v in variables if v not in NOT_MODELLED]      # e.g. observed DO % saturation
        inputs, tv = super().prepared_inputs(data, s, ti, variables)
        C, H = self.cfg.context, self.Hmax
        past = self._past_weather(data, s, ti, C)
        fut = self._future_nwp(s, ti, H)
        k = len(self.fields)
        for i, inp in enumerate(inputs):
            if inp is None:
                continue
            ctx = np.concatenate([inp["context"].numpy(), past[i]], axis=0)
            futc = np.concatenate([inp["future_covariates"].numpy(), fut[i]], axis=0)
            inp["context"] = torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.float32))
            inp["future_covariates"] = torch.from_numpy(np.ascontiguousarray(futc, dtype=np.float32))
            inp["n_covariates"] = int(inp["n_covariates"]) + k
            inp["n_future_covariates"] = int(inp["n_future_covariates"]) + k
        return inputs, tv
