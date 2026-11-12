# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Inner selection criterion of the architecture-v2 round (amendment §7), computable for every model class.

``J = 0.8·J_sonde + 0.2·J_grab``. J_sonde = mean over the primary targets of the standardised point MSE on the inner
sonde sites (the lever protocol's ``inner_primary_mse`` scale); J_grab = mean over {T, DO} of the standardised point
MSE of the hour-resolved prediction at the inner grab set, each station's weight capped at 30 sample-days.
All inputs are in model space (:class:`~.data.Scales`); train block and train-role locations only."""

from __future__ import annotations

import numpy as np

from .data import DAILY, INST, PRIMARY

W_SONDE, W_GRAB, GRAB_CAP = 0.8, 0.2, 30


def j_sonde(pred_daily: np.ndarray, y_daily: np.ndarray, crit_factor: dict, variables=("T", "DO")) -> dict:
    """``pred_daily`` (n, T, ≥7) and ``y_daily`` (n, T, ≥7) in model space (NaN = no label / no prediction)."""
    by = {}
    for v, k in PRIMARY.items():
        if k.split("_")[0] not in variables:
            continue
        j = DAILY.index(k)
        ok = np.isfinite(y_daily[:, :, j]) & np.isfinite(pred_daily[:, :, j])
        by[v] = float(np.mean((pred_daily[:, :, j][ok] - y_daily[:, :, j][ok]) ** 2) * crit_factor[v]) if ok.any() else None
    vals = [x for x in by.values() if x is not None]
    return {"j_sonde": float(np.mean(vals)) if len(vals) == len(by) and vals else None, "by_target": by,
            "n_cells": {v: int((np.isfinite(y_daily[:, :, DAILY.index(k)]) & np.isfinite(pred_daily[:, :, DAILY.index(k)])).sum())
                        for v, k in PRIMARY.items() if k.split("_")[0] in variables}}


def j_grab(pred_inst: np.ndarray, y_inst: np.ndarray, variables=("T", "DO"), cap: int = GRAB_CAP) -> dict:
    """``pred_inst`` / ``y_inst`` (n, T, 2) in model space. Row weight = min(n_s, cap) / n_s for a station with n_s
    scored rows of the variable."""
    by = {}
    for j, v in enumerate(INST):
        if v not in variables:
            continue
        ok = np.isfinite(y_inst[:, :, j]) & np.isfinite(pred_inst[:, :, j])
        n = ok.sum(axis=1).astype(float)
        if not ok.any():
            by[v] = None
            continue
        w = np.where(n > 0, np.minimum(n, cap) / np.maximum(n, 1), 0.0)
        se = np.where(ok, (pred_inst[:, :, j] - y_inst[:, :, j]) ** 2, 0.0).sum(axis=1)
        by[v] = float((w * se).sum() / (w * n).sum())
    vals = [x for x in by.values() if x is not None]
    return {"j_grab": float(np.mean(vals)) if len(vals) == len(by) and vals else None, "by_variable": by}


def combine(js: dict, jg: dict | None) -> float | None:
    """J; J_sonde alone when there is no inner grab set (no-grab variants)."""
    a = js.get("j_sonde")
    if a is None:
        return None
    b = None if jg is None else jg.get("j_grab")
    return float(a) if b is None else float(W_SONDE * a + W_GRAB * b)
