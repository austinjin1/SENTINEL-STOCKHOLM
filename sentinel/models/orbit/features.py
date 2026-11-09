# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tabular features for SENTINEL-Orbit baselines (orbit_protocol.md §4.2) and the inner split."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .data import BANDS, TARGETS

INNER_FRAC = 0.15
INNER_SEED = 0


def inner_block(blocks, frac: float = INNER_FRAC, seed: int = INNER_SEED) -> np.ndarray:
    """True for role blocks whose hash (salt ``inner|<seed>``) is below ``frac``."""
    return np.array([int(hashlib.sha256(f"inner|{seed}|{b}".encode()).hexdigest()[:12], 16) / 16 ** 12 < frac
                     for b in pd.Series(blocks).astype(str)])


def inner_mask(df: pd.DataFrame, frac: float = INNER_FRAC, seed: int = INNER_SEED) -> np.ndarray:
    """Inner (early-stopping / tuning) water bodies: train-role rows in inner blocks (any year). Whole
    blocks (hence whole lakes) go to one side."""
    return (df["role"] == "train").to_numpy() & inner_block(df["block"], frac, seed)


def cell_mask(df: pd.DataFrame, cell: str) -> np.ndarray:
    return (df["cell"] == cell).to_numpy()


FIT_SCHEMES = ("base", "recent", "track_u")


def fit_and_inner(df: pd.DataFrame, scheme: str = "base") -> tuple[np.ndarray, np.ndarray]:
    """``base``: fit = train role x train years (<= 2021), minus inner blocks and buffer rows; inner = inner
    blocks x train years. ``recent`` (iteration cycle 2): train role x (train + val years, <= 2023) — val-role
    water bodies are never fitted, so the val-role selection cell stays unseen; inner = inner blocks x the
    same years."""
    if scheme not in FIT_SCHEMES:
        raise ValueError(f"unknown fit scheme {scheme!r}")
    cells = ["trainxtrain", "trainxval"] if scheme == "recent" else ["trainxtrain"]
    fit_cell = np.isin(df["cell"].to_numpy(), cells) & ~df["buffer_excluded"].to_numpy(bool)
    if scheme == "track_u":
        # harder track U (orbit_protocol.md §11.2): drop held-out HUC2 regions and a 25-km buffer from fit + inner
        from . import track_u as U

        fit_cell &= ~U.assign(df["anchor_lon"], df["anchor_lat"]).u_excluded_from_fit.to_numpy(bool)
    inner = inner_mask(df) & fit_cell
    return fit_cell & ~inner, inner


def log_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {t: np.log10(df[t].to_numpy(float)) for t in TARGETS}


def doy_harmonics(doy) -> tuple[np.ndarray, np.ndarray]:
    a = 2 * np.pi * (np.asarray(doy, float) - 1) / 365.25
    return np.sin(a), np.cos(a)


def unit_vector(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def _safe_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r[~np.isfinite(r)] = np.nan
    return r


def image_features(df: pd.DataFrame, pix: dict | None = None) -> pd.DataFrame:
    """Band statistics: 3x3 / 7x7 water medians, pixel-set percentiles (p10/p50/p90 of the cached water
    pixels), band ratios/indices, geometry, AOT, shore distance and water-pixel counts."""
    X = {}
    for b in BANDS:
        X[f"rw3_{b}"] = df[f"rw3_{b}"].to_numpy(float)
        r7 = df[f"rw7_{b}"].to_numpy(float)
        X[f"rw7_{b}"] = np.where(np.isfinite(r7), r7, X[f"rw3_{b}"])
    if pix is not None:
        refl = pix["refl"].astype(np.float32)
        q = np.nanpercentile(refl, [10, 50, 90], axis=1)  # [3, N, 12]
        for j, b in enumerate(BANDS):
            X[f"px10_{b}"], X[f"px50_{b}"], X[f"px90_{b}"] = q[0, :, j], q[1, :, j], q[2, :, j]
    g = {b: X[f"rw3_{b}"] for b in BANDS}
    X["r_B02_B03"] = _safe_ratio(g["B02"], g["B03"])
    X["r_B04_B03"] = _safe_ratio(g["B04"], g["B03"])
    X["r_B05_B04"] = _safe_ratio(g["B05"], g["B04"])
    X["r_B08_B04"] = _safe_ratio(g["B08"], g["B04"])
    X["r_B8A_B04"] = _safe_ratio(g["B8A"], g["B04"])
    X["r_B01B02max_B03"] = _safe_ratio(np.fmax(g["B01"], g["B02"]), g["B03"])
    X["ndci"] = _safe_ratio(g["B05"] - g["B04"], g["B05"] + g["B04"])
    X["ndwi"] = _safe_ratio(g["B03"] - g["B08"], g["B03"] + g["B08"])
    X["sun_zenith"] = df.sun_zenith.to_numpy(float)
    X["view_zenith"] = df.view_zenith.to_numpy(float)
    X["cos_raa"] = np.cos(np.radians(df.sun_azimuth.to_numpy(float) - df.view_azimuth.to_numpy(float)))
    X["aot_c3"] = df.aot_c3.to_numpy(float)
    X["log_dist_land"] = np.log1p(df.jrc_dist_land_m.to_numpy(float))
    X["n_water_win"] = (pix["n_water_win"] if pix is not None else np.full(len(df), np.nan)).astype(float)
    X["scl_c3_n_water"] = df.scl_c3_n_water.to_numpy(float)
    return pd.DataFrame(X, index=df.index)


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    s, c = doy_harmonics(df.doy)
    return pd.DataFrame({"lat": df.lat.to_numpy(float), "lon": df.lon.to_numpy(float), "doy_sin": s, "doy_cos": c},
                        index=df.index)
