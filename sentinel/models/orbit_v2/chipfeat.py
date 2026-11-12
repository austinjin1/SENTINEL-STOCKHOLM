# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Per-chip image features of arm A1 (orbit_arch_v2_amendment.md §3.4 items 1, 4, 5). Imagery and image metadata
only; no label is read here.

* :func:`dp_map` — d_p: distance (m) of every SCL-water pixel to the nearest non-water pixel, from a Euclidean
  distance transform of the SCL water mask over the full 64x64 chip, capped at the distance to the chip edge
  (the chip is padded with one ring of non-water, so an edge pixel has d_p = 10 m).
* :func:`ring_features` — land ring: asinh median L2A spectrum (12 bands) of SCL vegetation (4) / not-vegetated (5)
  / unclassified (7) pixels 30-300 m from the chip's water pixels, the land fraction of the chip, the land NDVI
  and a "ring missing" flag (< 5 ring pixels).
* :func:`glint_features` — cos 2w (sun-view angle), the specular glint angle (rad) and the Fresnel reflectance at
  w for n = 1.34.
* :func:`baseline_group` — Sen2Cor processing-baseline group {< 04.00, 04.xx, >= 05.00}.
"""

from __future__ import annotations

import numpy as np

from ...data import chips as C
from ...data.sources import stac as S
from ..orbit.data import BANDS, CENTER

PIXEL_M = 10.0
SCL_WATER = 6
SCL_LAND = (4, 5, 7)            # vegetation, not vegetated, unclassified
RING_MIN_M, RING_MAX_M = 30.0, 300.0
RING_MIN_PIXELS = 5
REFL_SCALE = 0.01               # as SatWQ-Net v1 (asinh(rho / 0.01))
N_WATER = 1.34
RING_DIM = len(BANDS) + 3       # 12 asinh medians, land fraction, land NDVI, missing flag
GLINT_NAMES = ("cos_2omega", "glint_angle_rad", "fresnel_omega")
BASELINE_GROUPS = ("<04.00", "04.xx", ">=05.00")


def _layers(layers):
    return {n: i for i, n in enumerate(layers)}


def water_mask(chip: np.ndarray, layers=C.S2_STORE_LAYERS) -> np.ndarray:
    return chip[_layers(layers)["SCL"]] == SCL_WATER


def dp_map(chip: np.ndarray, layers=C.S2_STORE_LAYERS) -> np.ndarray:
    """[H, W] float32: metres from each water pixel to the nearest non-water pixel (0 at non-water pixels)."""
    from scipy.ndimage import distance_transform_edt

    w = np.pad(water_mask(chip, layers), 1, constant_values=False)
    return (distance_transform_edt(w)[1:-1, 1:-1] * PIXEL_M).astype(np.float32)


def dp_at(dp: np.ndarray, offsets: np.ndarray, n_pix: int) -> np.ndarray:
    """d_p of the cached pixels: ``offsets`` [P, 2] = (dy, dx) from the chip centre (data.chip_pixels)."""
    out = np.zeros(len(offsets), np.float32)
    if n_pix:
        r = CENTER + offsets[:n_pix, 0].astype(int)
        c = CENTER + offsets[:n_pix, 1].astype(int)
        out[:n_pix] = dp[r, c]
    return out


def ring_features(chip: np.ndarray, layers=C.S2_STORE_LAYERS) -> np.ndarray:
    """[RING_DIM] float32. All zero except the missing flag (= 1) when the ring has < 5 pixels or no water pixel."""
    from scipy.ndimage import distance_transform_edt

    L = _layers(layers)
    scl = chip[L["SCL"]]
    water = scl == SCL_WATER
    dn = np.stack([chip[L[b]] for b in BANDS])
    land = np.isin(scl, SCL_LAND) & np.all(dn != S.NODATA_I16, axis=0)
    out = np.zeros(RING_DIM, np.float32)
    out[len(BANDS)] = float(land.mean())
    if not water.any():
        out[-1] = 1.0
        return out
    dist = distance_transform_edt(~water) * PIXEL_M          # metres to the nearest water pixel
    ring = land & (dist >= RING_MIN_M) & (dist <= RING_MAX_M)
    if int(ring.sum()) < RING_MIN_PIXELS:
        out[-1] = 1.0
        return out
    med = np.median(dn[:, ring].astype(np.float32) * 1e-4, axis=1)
    out[:len(BANDS)] = np.arcsinh(med / REFL_SCALE)
    b4, b8 = med[BANDS.index("B04")], med[BANDS.index("B08")]
    den = b8 + b4
    out[len(BANDS) + 1] = float(np.clip((b8 - b4) / den, -1.0, 1.0)) if abs(den) > 1e-6 else 0.0
    return out


def glint_features(sun_zenith, view_zenith, sun_azimuth, view_azimuth) -> np.ndarray:
    """[N, 3]: cos 2w, glint angle (rad), Fresnel reflectance at w (unpolarised, n = 1.34). Angles in degrees.

    Unit vectors from the surface point to the sun (s) and to the sensor (v): cos 2w = s.v. The specular direction
    of the sun on a flat surface is r = (-s_x, -s_y, s_z); the glint angle is the angle between r and v."""
    ts, tv = np.radians(np.asarray(sun_zenith, float)), np.radians(np.asarray(view_zenith, float))
    dphi = np.radians(np.asarray(sun_azimuth, float) - np.asarray(view_azimuth, float))
    cos_2w = np.clip(np.cos(ts) * np.cos(tv) + np.sin(ts) * np.sin(tv) * np.cos(dphi), -1.0, 1.0)
    cos_g = np.clip(np.cos(ts) * np.cos(tv) - np.sin(ts) * np.sin(tv) * np.cos(dphi), -1.0, 1.0)
    w = 0.5 * np.arccos(cos_2w)
    return np.stack([cos_2w, np.arccos(cos_g), fresnel(w)], -1)


def fresnel(omega, n: float = N_WATER) -> np.ndarray:
    """Unpolarised Fresnel reflectance of a flat air-water interface at incidence angle ``omega`` (rad)."""
    w = np.asarray(omega, float)
    t = np.arcsin(np.clip(np.sin(w) / n, -1.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (np.sin(w - t) / np.sin(w + t)) ** 2
        rp = (np.tan(w - t) / np.tan(w + t)) ** 2
        rho = 0.5 * (rs + rp)
    return np.where(w < 1e-6, ((n - 1) / (n + 1)) ** 2, rho)


def baseline_group(processing_baseline) -> np.ndarray:
    """Group index per chip: 0 = < 04.00, 1 = 04.xx, 2 = >= 05.00 (unparseable -> 2, the current baseline)."""
    out = np.full(len(processing_baseline), 2, np.int64)
    for i, s in enumerate(np.asarray(processing_baseline, dtype=object)):
        try:
            v = float(str(s).upper().lstrip("N"))
        except ValueError:
            continue
        if v >= 100:                  # 'N0500' style
            v /= 100.0
        out[i] = 0 if v < 4.0 else (1 if v < 5.0 else 2)
    return out


def date_quality(sun_zenith, view_zenith, sun_azimuth, view_azimuth, aot, processing_baseline) -> np.ndarray:
    """q_d of a context date (§3.4 item 3): (cos SZA, cos VZA, cos RAA, AOT, glint angle) + one-hot baseline group
    -> [N, 8] float32 (unstandardised; NaN AOT -> NaN, filled after standardisation)."""
    sz, vz = np.asarray(sun_zenith, float), np.asarray(view_zenith, float)
    raa = np.asarray(sun_azimuth, float) - np.asarray(view_azimuth, float)
    g = glint_features(sz, vz, sun_azimuth, view_azimuth)[:, 1]
    oh = np.eye(3, dtype=np.float32)[baseline_group(processing_baseline)]
    return np.concatenate([np.stack([np.cos(np.radians(sz)), np.cos(np.radians(vz)), np.cos(np.radians(raa)),
                                     np.asarray(aot, float), g], 1).astype(np.float32), oh], 1)


def chip_features(chip: np.ndarray, offsets: np.ndarray, n_pix: int, layers=C.S2_STORE_LAYERS):
    """(d_p of the cached pixels [P], ring vector [RING_DIM]) of one chip."""
    return dp_at(dp_map(chip, layers), offsets, n_pix), ring_features(chip, layers)
