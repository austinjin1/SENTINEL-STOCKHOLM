# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""gridMET (Abatzoglou 2013; CONUS, 1/24 deg ~4 km, daily, 1979-present) point extraction.

Access method (verified 2026-10-22)
-----------------------------------
* **Bulk yearly netCDF** ``https://www.northwestknowledge.net/metdata/data/<var>_<year>.nc``
  (~45-60 MB per variable-year, ~20 MB/s). One file serves all sites, so ~270 requests cover
  10 variables x 27 years for every site. This is the method used here.
* THREDDS NCSS point service
  (``thredds.northwestknowledge.net:8080/thredds/ncss/grid/agg_met_<var>_1979_CurrentYear_CONUS.nc``)
  works but its CSV output returns the *packed* integers without applying ``scale_factor``
  (e.g. precipitation 215 for 21.5 mm) and a per-site, per-variable request reads ~160 chunks
  server-side. Not used; documented as a trap.

Conventions in the files: ``note5`` -- "Days correspond approximately to calendar days ending at
midnight, Mountain Standard Time (7 UTC the next calendar day)". Recent slices are provisional
(``last_permanent_slice`` / ``last_provisional_slice`` global attributes, 1-based), so each value
carries a ``gridmet_status`` of permanent / provisional / early.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BULK_URL = "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
NCSS_URL = (
    "http://thredds.northwestknowledge.net:8080/thredds/ncss/grid/agg_met_{var}_1979_CurrentYear_CONUS.nc"
)
USER_AGENT = "SENTINEL-v2 research (public-data download)"

# short name -> (netCDF variable, expected units in file, harmonised column, harmonised unit)
VARIABLES: dict[str, tuple[str, str, str, str]] = {
    "pr": ("precipitation_amount", "mm", "precip_mm", "mm/day"),
    "tmmx": ("air_temperature", "K", "tmax_c", "degC"),
    "tmmn": ("air_temperature", "K", "tmin_c", "degC"),
    "srad": ("surface_downwelling_shortwave_flux_in_air", "W m-2", "srad_wm2", "W/m2 (daily mean)"),
    "vs": ("wind_speed", "m/s", "wind_ms", "m/s (10 m, daily mean)"),
    "rmax": ("relative_humidity", "%", "rhmax_pct", "%"),
    "rmin": ("relative_humidity", "%", "rhmin_pct", "%"),
    "sph": ("specific_humidity", "kg/kg", "sph_kgkg", "kg/kg"),
    "vpd": ("mean_vapor_pressure_deficit", "kPa", "vpd_kpa", "kPa (daily mean)"),
    "pet": ("potential_evapotranspiration", "mm", "pet_mm", "mm/day (ASCE grass reference ET)"),
}
KELVIN = 273.15
EARTH_R_KM = 6371.0088


def file_url(var: str, year: int) -> str:
    if var not in VARIABLES:
        raise KeyError(var)
    return BULK_URL.format(var=var, year=int(year))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def nearest_valid_cells(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    valid: np.ndarray,
    max_radius_cells: int = 3,
) -> pd.DataFrame:
    """Nearest grid cell with data for each point; search a (2r+1)^2 window if the nearest is masked.

    ``grid_lat`` may be descending (gridMET is north->south). Points outside the grid extent by
    more than half a cell get ``i = j = -1``.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dlat = abs(float(grid_lat[1] - grid_lat[0]))
    dlon = abs(float(grid_lon[1] - grid_lon[0]))
    i0 = np.rint((lat - grid_lat[0]) / (grid_lat[1] - grid_lat[0])).astype(int)
    j0 = np.rint((lon - grid_lon[0]) / (grid_lon[1] - grid_lon[0])).astype(int)
    ny, nx = valid.shape
    inside = (
        (lat >= grid_lat.min() - dlat / 2)
        & (lat <= grid_lat.max() + dlat / 2)
        & (lon >= grid_lon.min() - dlon / 2)
        & (lon <= grid_lon.max() + dlon / 2)
    )
    out_i = np.full(lat.shape, -1)
    out_j = np.full(lat.shape, -1)
    dist = np.full(lat.shape, np.nan)
    for k in range(lat.size):
        if not inside[k]:
            continue
        best = None
        for r in range(0, max_radius_cells + 1):
            ii = np.arange(max(i0[k] - r, 0), min(i0[k] + r, ny - 1) + 1)
            jj = np.arange(max(j0[k] - r, 0), min(j0[k] + r, nx - 1) + 1)
            if ii.size == 0 or jj.size == 0:
                continue
            gi, gj = np.meshgrid(ii, jj, indexing="ij")
            ok = valid[gi, gj]
            if ok.any():
                d = haversine_km(lat[k], lon[k], grid_lat[gi[ok]], grid_lon[gj[ok]])
                m = int(np.argmin(d))
                best = (int(gi[ok][m]), int(gj[ok][m]), float(d[m]))
                break
        if best is not None:
            out_i[k], out_j[k], dist[k] = best
    return pd.DataFrame(
        {
            "i": out_i,
            "j": out_j,
            "cell_dist_km": dist,
            "cell_lat": np.where(out_i >= 0, grid_lat[np.clip(out_i, 0, ny - 1)], np.nan),
            "cell_lon": np.where(out_j >= 0, grid_lon[np.clip(out_j, 0, nx - 1)], np.nan),
        }
    )


def read_grid(nc_path: Path) -> dict:
    """Open a gridMET yearly file -> dict(lat, lon, dates, var_name, units, data[day,lat,lon] float32,
    status per day, attrs). Packing (scale_factor/add_offset/_FillValue) is applied here."""
    import netCDF4  # local import: optional dependency of the geo stack

    with netCDF4.Dataset(nc_path) as ds:
        var3 = [k for k, v in ds.variables.items() if v.ndim == 3]
        if len(var3) != 1:
            raise ValueError(f"expected one 3-D variable in {nc_path}, got {var3}")
        v = ds.variables[var3[0]]
        if tuple(v.dimensions) != ("day", "lat", "lon"):
            raise ValueError(f"unexpected dims {v.dimensions}")
        v.set_auto_maskandscale(True)
        data = np.ma.filled(v[:].astype("float32"), np.nan)
        day = ds.variables["day"]
        dates = pd.to_datetime(
            netCDF4.num2date(
                day[:], day.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True
            )
        ).normalize()
        attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
        n = len(dates)
        status = np.array(["permanent"] * n, dtype=object)
        lp = attrs.get("last_permanent_slice")
        lv = attrs.get("last_provisional_slice")
        if lp is not None:
            lp = int(lp)
            status[lp:] = "provisional"
            if lv is not None:
                status[int(lv) :] = "early"
        return {
            "lat": np.asarray(ds.variables["lat"][:], float),
            "lon": np.asarray(ds.variables["lon"][:], float),
            "dates": dates,
            "var_name": var3[0],
            "units": getattr(v, "units", ""),
            "data": data,
            "status": status,
            "attrs": attrs,
        }


def extract_cells(grid: dict, cells: pd.DataFrame) -> np.ndarray:
    """Values [n_days, n_cells] at (i, j) of ``cells`` (rows with i < 0 -> NaN)."""
    i = cells["i"].to_numpy()
    j = cells["j"].to_numpy()
    out = np.full((grid["data"].shape[0], len(cells)), np.nan, dtype="float32")
    ok = i >= 0
    out[:, ok] = grid["data"][:, i[ok], j[ok]]
    return out


def to_harmonised(var: str, values: np.ndarray) -> np.ndarray:
    """File units -> harmonised units (only temperature changes: K -> degC, exact offset)."""
    if var in ("tmmx", "tmmn"):
        return values - KELVIN
    return values


def download(url: str, dest: Path, *, retries: int = 6, timeout: float = 600, session=None) -> int:
    """Stream ``url`` to ``dest`` (atomic rename); returns bytes. Verifies Content-Length."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    last = None
    for k in range(retries):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with s.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("Retry-After", 60)))
                    continue
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length", -1))
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        n += len(chunk)
            if expected >= 0 and n != expected:
                raise IOError(f"short read {n}/{expected}")
            tmp.replace(dest)
            return n
        except FileNotFoundError:
            raise
        except (requests.RequestException, IOError) as e:
            last = repr(e)
            tmp.unlink(missing_ok=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"download failed: {url} :: {last}")


def ncss_scale_trap_example() -> dict:
    """Documented NCSS pitfall (no network): CSV values are packed integers; multiply by scale."""
    return {
        "raw": 215,
        "scale_factor": 0.1,
        "add_offset": 0.0,
        "physical_mm": 21.5,
        "note": "NCSS CSV for agg_met_pr_* on 2024-01-08 at (39.63,-90.61) returned 215",
    }


def deg_cell_area_km2(lat: float, res_deg: float = 1 / 24) -> float:
    """Approximate area of a res_deg x res_deg cell at latitude ``lat`` (for documentation)."""
    return (EARTH_R_KM * math.radians(res_deg)) ** 2 * math.cos(math.radians(lat))
