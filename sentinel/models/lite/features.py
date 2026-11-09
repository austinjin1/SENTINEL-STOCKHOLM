# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chip statistics for the SENTINEL-Lite baselines (protocol §5) and the student's pooling window.

All features come from the chip's own pixels plus its DOQQ statistics (no labels, no location unless a
``context`` variant asks for it). The water mask is the dark-NIR rule NIR < G on raw DN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CENTRE_RADIUS_M = 15.0   # pooling half-width in metres (~30 m window), also used by the student
BAND_IDX = {"R": 0, "G": 1, "B": 2, "NIR": 3}


def centre_halfwidth_px(gsd_m) -> np.ndarray:
    return np.maximum(np.round(CENTRE_RADIUS_M / np.asarray(gsd_m, float)).astype(int), 2)


def water_mask(chips: np.ndarray) -> np.ndarray:
    """(N, H, W) bool: NIR < G and pixel inside the image (any band > 0)."""
    c = chips.astype(np.int16)
    return (c[:, 3] < c[:, 1]) & (c.max(axis=1) > 0)


def _window(a, cy, cx, h):
    return a[..., cy - h:cy + h + 1, cx - h:cx + h + 1]


def chip_features(chips: np.ndarray, gsd_m, scene_p50: np.ndarray) -> pd.DataFrame:
    """Per-chip RGBN statistics. ``scene_p50``: (N, 4) DOQQ band medians (R, G, B, NIR)."""
    n, _, H, W = chips.shape
    cy, cx = H // 2, W // 2
    hw = centre_halfwidth_px(gsd_m)
    wm = water_mask(chips)
    rows = []
    for i in range(n):
        x = chips[i].astype(np.float32)
        f: dict = {}
        c9 = _window(x, cy, cx, 4).reshape(4, -1)
        c31 = _window(x, cy, cx, 15).reshape(4, -1)
        win = _window(x, cy, cx, int(hw[i])).reshape(4, -1)
        wwin = _window(wm[i], cy, cx, int(hw[i])).reshape(-1)
        f["water_frac_win"] = float(wwin.mean())
        f["water_frac_chip"] = float(wm[i].mean())
        wpx = win[:, wwin] if wwin.sum() >= 5 else win
        for b, k in BAND_IDX.items():
            p50 = max(float(scene_p50[i, k]), 1.0)
            f[f"c9_{b}"] = float(np.median(c9[k]))
            f[f"c31_{b}"] = float(np.median(c31[k]))
            q25, q50, q75 = np.percentile(wpx[k], [25, 50, 75])
            f[f"w_{b}_q25"], f[f"w_{b}_q50"], f[f"w_{b}_q75"] = float(q25), float(q50), float(q75)
            f[f"c9s_{b}"] = (f[f"c9_{b}"] + 1.0) / p50          # +1 DN: NIR can be 0
            f[f"ws_{b}"] = (float(q50) + 1.0) / p50
            f[f"scene_p50_{b}"] = p50
        eps = 1.0
        for pre in ("c9", "w"):
            g = (lambda b: f[f"{pre}_{b}"]) if pre == "c9" else (lambda b: f[f"w_{b}_q50"])
            R, G, B, N = g("R") + eps, g("G") + eps, g("B") + eps, g("NIR") + eps
            f[f"{pre}_B_R"], f[f"{pre}_G_R"], f[f"{pre}_B_G"], f[f"{pre}_N_R"] = B / R, G / R, B / G, N / R
            f[f"{pre}_ndvi"] = (N - R) / (N + R)
        rows.append(f)
    out = pd.DataFrame(rows)
    out["gsd_m"] = np.asarray(gsd_m, float)
    return out


GBM_IMAGE_FEATURES = None  # all columns of chip_features (set lazily)
CONTEXT_FEATURES = ("lat", "lon", "doy")


def pooling_weights(chips: np.ndarray, gsd_m, stride: int = 4) -> np.ndarray:
    """(N, H/stride, W/stride) float32 weights for the student's chip-level pooling: centre window ×
    water fraction per output cell; falls back to the plain centre window when < 5% water."""
    n, _, H, W = chips.shape
    h, w = H // stride, W // stride
    wm = water_mask(chips).reshape(n, h, stride, w, stride).mean(axis=(2, 4)).astype(np.float32)
    hw = np.maximum(np.round(centre_halfwidth_px(gsd_m) / stride).astype(int), 1)
    out = np.zeros((n, h, w), np.float32)
    cy, cx = h // 2, w // 2
    for i in range(n):
        win = np.zeros((h, w), np.float32)
        win[cy - hw[i]:cy + hw[i], cx - hw[i]:cx + hw[i]] = 1.0
        ww = win * wm[i]
        out[i] = ww if ww.sum() >= 0.05 * win.sum() else win
        out[i] /= out[i].sum()
    return out
