# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""TimesFM-2.5 (google/timesfm-2.5-200m-pytorch, Apache-2.0; package timesfm 3.0.2) zero-shot
forecasts on the common SENTINEL-Sonde rows — same protocol as the Chronos-2 ``uni`` mode:
univariate model-space history over the ``context`` days ending at the issue date (inclusive; the
package fills internal NaNs by linear interpolation and strips leading NaNs), 14-step horizon.

Quantiles: TimesFM-2.5 returns the mean and deciles 0.1…0.9 (continuous quantile head). Our 19
levels are obtained by interpolation between adjacent deciles that is linear in the normal score
Φ⁻¹(τ), with linear-in-Φ⁻¹ extrapolation for 0.05 / 0.95 (a fixed, documented format conversion
applied identically to every row; exact for Gaussian forecasts). ``infer_is_positive`` is off (model space can be
negative).
"""

from __future__ import annotations

import os

import numpy as np

from ...core import paths
from ..baselines.sonde_baselines import QUANTILES
from .chronos import padded_series, windows
from .common import HORIZONS, Rows, SondeData
from .data import to_model_space

HF_CACHE = paths.RUNS_ROOT / "hf_cache"
MODEL_ID = "google/timesfm-2.5-200m-pytorch"
DECILES = np.round(np.arange(0.1, 0.91, 0.1), 2)


def deciles_to_19(dec: np.ndarray) -> np.ndarray:
    """(..., 9) deciles 0.1..0.9 -> (..., 19) quantiles 0.05..0.95: piecewise-linear in the
    normal-score Φ⁻¹(τ) between adjacent deciles, linear extrapolation of the outermost segment
    in Φ⁻¹(τ) for 0.05 / 0.95 (exact for Gaussian forecasts)."""
    from scipy.stats import norm

    dec = np.sort(np.asarray(dec, float), axis=-1)
    zd = norm.ppf(DECILES)
    out = np.empty(dec.shape[:-1] + (len(QUANTILES),))
    for k, q in enumerate(QUANTILES):
        z = norm.ppf(q)
        i = int(np.clip(np.searchsorted(zd, z) - 1, 0, len(zd) - 2))
        w = (z - zd[i]) / (zd[i + 1] - zd[i])
        out[..., k] = dec[..., i] + w * (dec[..., i + 1] - dec[..., i])
    return np.sort(out, axis=-1)


class TimesFMForecaster:
    def __init__(self, context: int = 512, batch_size: int = 256, horizons=HORIZONS, model_id: str = MODEL_ID,
                 chunk_rows: int = 50_000):
        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        import timesfm

        self.context, self.horizons, self.chunk = int(context), tuple(horizons), chunk_rows
        self.Hmax = max(self.horizons)
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, cache_dir=str(HF_CACHE / "hub"))
        self.model.compile(timesfm.ForecastConfig(
            max_context=self.context, max_horizon=self.Hmax, normalize_inputs=True, per_core_batch_size=batch_size,
            use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=False,
            fix_quantile_crossing=True))

    def predict_uni(self, data: SondeData, var: str, rows: Rows) -> np.ndarray:
        z = to_model_space(var, data.cube.y[:, :, data.cube.var_index(var)])
        zpad = padded_series(z, self.context)
        out = np.full((len(rows), len(self.horizons), len(QUANTILES)), np.nan, dtype=np.float32)
        h_idx = [h - 1 for h in self.horizons]
        for a in range(0, len(rows), self.chunk):
            si = rows.si[a:a + self.chunk].astype(np.int64)
            ti = rows.ti[a:a + self.chunk].astype(np.int64)
            ctx = windows(zpad, si, ti, self.context)
            _, qf = self.model.forecast(horizon=self.Hmax, inputs=[c for c in ctx])   # (n, H, 10): mean + deciles
            out[a:a + si.size] = deciles_to_19(qf[:, h_idx, 1:10])
        return out

    def predict(self, data: SondeData, rows_by_var: dict[str, Rows]) -> dict[str, np.ndarray]:
        return {v: self.predict_uni(data, v, r) for v, r in rows_by_var.items()}
