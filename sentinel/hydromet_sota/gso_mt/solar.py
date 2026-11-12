# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Solar geometry for the M6 architecture-v2 input set (amendment §3) and the local solar hour of a sample (§2.2).

Everything is a deterministic function of latitude, longitude, elevation, day of year and the ``srad`` forcing:
no in-situ input. Formulas: FAO-56 (Allen et al. 1998) eq. 21–25 (extraterrestrial radiation, sunset hour angle,
declination, inverse relative distance), eq. 34 (daylight hours) and eq. 37 (clear-sky radiation
Rso = (0.75 + 2e-5 z) Ra). Every number comes from :mod:`sentinel.hydromet_sota.gso_mt.constants` (source per value;
this file holds no physical constant of its own)."""

from __future__ import annotations

import numpy as np

from . import constants as K

SOLAR_CHANNELS = ("photoperiod_h", "clearsky_wm2", "cloud_ratio", "noon_elev_deg")
GSC_WM2 = K.GSC_WM2                       # FAO-56 solar constant 0.0820 MJ m-2 min-1 = 1366.7 W m-2
MIX_HOURS = np.arange(8, 18)              # a sample without a time: uniform mixture over 08..17 h local solar


def local_solar_hour(utc_hour, lon_deg):
    """Mean local solar hour in [0, 24): (UTC hour + longitude / 15) mod 24. The equation of time (≤ 16 min) is
    ignored (amendment §2.2)."""
    return np.mod(np.asarray(utc_hour, dtype="float64") + np.asarray(lon_deg, dtype="float64") / K.DEG_PER_HOUR, 24.0)


def declination(doy):
    return K.DECLINATION_AMPLITUDE * np.sin(2.0 * np.pi * np.asarray(doy, dtype="float64") / K.DAYS_PER_YEAR
                                            - K.DECLINATION_PHASE)


def solar_channels(lat_deg, elev_m, doy, srad_wm2):
    """(S, T, 4) array of :data:`SOLAR_CHANNELS` from lat (S,), elevation (S,), day of year (T,), srad (S, T).

    * photoperiod (h) = 24/π · ω_s;
    * clear-sky radiation (daily mean W m⁻²) = (0.75 + 2e-5 z) · Ra, Ra = extraterrestrial daily mean;
    * cloud ratio = srad / clear-sky, clipped to [0, 1.2] (NaN where srad is missing);
    * noon solar elevation (deg) = 90 − |lat − declination|.
    A missing elevation is treated as 0 m (changes the clear-sky term by ≤ 2e-5 per metre)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype="float64"))[:, None]
    z = np.nan_to_num(np.asarray(elev_m, dtype="float64"), nan=0.0)[:, None]
    doy = np.asarray(doy, dtype="float64")[None, :]
    dec = declination(doy)
    dr = 1.0 + K.DR_AMPLITUDE * np.cos(2.0 * np.pi * doy / K.DAYS_PER_YEAR)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0))
    ra = (GSC_WM2 / np.pi) * dr * (ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws))
    rso = np.maximum((K.RSO_A + K.RSO_B * z) * ra, 1.0)
    photo = K.HOURS_PER_RADIAN * ws
    cloud = np.clip(np.asarray(srad_wm2, dtype="float64") / rso, 0.0, 1.2)
    elev = 90.0 - np.abs(np.rad2deg(lat) - np.rad2deg(dec))
    return np.stack([np.broadcast_to(photo, cloud.shape), rso * np.ones_like(cloud), cloud,
                     np.broadcast_to(elev, cloud.shape)], axis=-1).astype("float32")
