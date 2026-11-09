# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Hydrometeorological drivers for D1 SafeSwim (Chicago beaches): pure request builders and parsers.

Sources (verified 2026-10-25):

* **USGS NWIS daily discharge** (``parameterCd=00060``, ``statCd=00003`` daily mean, ft³/s), legacy
  WaterServices ``https://waterservices.usgs.gov/nwis/dv/`` (JSON). Gauges with Chicago-area records
  spanning 2006–2026: 05536000 NB Chicago R at Niles, 05536105 NB Chicago R at Albany Ave,
  05536290 Little Calumet R at South Holland, 05536890 Chicago Sanitary & Ship Canal near Lemont.
* **NOAA CO-OPS Great Lakes daily mean water level**, station 9087044 Calumet Harbor, IGLD85 metres,
  ``https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`` (``product=daily_mean``, one-year
  requests, LST).
* **NOAA NDBC historical standard meteorology** (hourly), buoys 45007 (southern Lake Michigan,
  offshore) and 45174 (Wilmette, nearshore; seasonal), ``https://www.ndbc.noaa.gov/data/historical/
  stdmet/<id>h<year>.txt.gz``. Missing codes (99, 99.0, 999, 999.0, 9999.0 per NDBC docs) → NaN.

Units are kept in column names; missing codes become NaN with the raw code counted in ``qualifier``.
"""

from __future__ import annotations

import gzip
import io

import numpy as np
import pandas as pd

USER_AGENT = "SENTINEL-v2 research (public-data download)"
USGS_DV = "https://waterservices.usgs.gov/nwis/dv/"
COOPS = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NDBC = "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"
GAUGES = {"05536000": "NB Chicago R at Niles", "05536105": "NB Chicago R at Albany Ave",
          "05536290": "Little Calumet R at South Holland", "05536890": "Chicago S&S Canal near Lemont"}
COOPS_STATION = "9087044"
BUOYS = {"45007": "southern Lake Michigan (offshore)", "45174": "Wilmette (nearshore)"}
NDBC_MISSING = {"WDIR": 999, "WSPD": 99.0, "GST": 99.0, "WVHT": 99.0, "DPD": 99.0, "APD": 99.0, "MWD": 999,
                "PRES": 9999.0, "ATMP": 999.0, "WTMP": 999.0, "DEWP": 999.0, "VIS": 99.0, "TIDE": 99.0}


def usgs_dv_request(site: str, start: str, end: str) -> tuple[str, dict]:
    return USGS_DV, {"format": "json", "sites": site, "parameterCd": "00060", "statCd": "00003",
                     "startDT": start, "endDT": end, "siteStatus": "all"}


def parse_usgs_dv(payload: dict, site: str) -> pd.DataFrame:
    """WaterML-JSON daily values → (site_no, date, discharge_cfs, qualifiers). Values ≤ −999999 → NaN."""
    rows = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        nodata = ts.get("variable", {}).get("noDataValue", -999999.0)
        for block in ts.get("values", []):
            for v in block.get("value", []):
                try:
                    x = float(v["value"])
                except (TypeError, ValueError):
                    x = np.nan
                if not np.isfinite(x) or x == nodata or x <= -999999:
                    x = np.nan
                rows.append({"site_no": site, "date": pd.Timestamp(v["dateTime"][:10]), "discharge_cfs": x,
                             "qualifiers": ",".join(v.get("qualifiers", []))})
    df = pd.DataFrame(rows, columns=["site_no", "date", "discharge_cfs", "qualifiers"])
    return df.drop_duplicates(["site_no", "date"]).sort_values("date").reset_index(drop=True)


def coops_request(year: int, station: str = COOPS_STATION) -> tuple[str, dict]:
    return COOPS, {"product": "daily_mean", "station": station, "begin_date": f"{year}0101",
                   "end_date": f"{year}1231", "datum": "IGLD", "units": "metric", "time_zone": "lst",
                   "format": "json", "application": "SENTINEL-v2"}


def parse_coops_daily(payload: dict) -> pd.DataFrame:
    data = payload.get("data") or []
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["date", "water_level_m_igld", "flags"])
    out = pd.DataFrame({"date": pd.to_datetime(df["t"].str[:10]),
                        "water_level_m_igld": pd.to_numeric(df["v"], errors="coerce"),
                        "flags": df.get("f", pd.Series("", index=df.index)).astype(str)})
    return out.sort_values("date").reset_index(drop=True)


NDBC_ALIASES = {"WD": "WDIR", "BAR": "PRES", "YYYY": "YY", "YY": "YY", "#YY": "YY"}


def parse_ndbc_stdmet(raw: bytes, station: str) -> pd.DataFrame:
    """NDBC stdmet (.txt or .txt.gz bytes) → hourly rows in UTC with NDBC missing codes set to NaN."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    lines = raw.decode("ascii", "replace").splitlines()
    header = lines[0].lstrip("#").split()
    body = [ln for ln in lines[1:] if not ln.startswith("#") and ln.strip()]
    header = [NDBC_ALIASES.get(h, h) for h in header]                 # pre-2007 files: WD, BAR, no 'mm'
    df = pd.read_csv(io.StringIO("\n".join(body)), sep=r"\s+", header=None, names=header, dtype=str)
    if len(df) and len(df.iloc[0, 0]) == 2:                              # two-digit years (pre-1999 files)
        df[df.columns[0]] = "19" + df[df.columns[0]]
    yr = df.columns[0]
    t = pd.to_datetime(df[yr] + "-" + df["MM"] + "-" + df["DD"] + " " + df["hh"] + ":" + df.get("mm", "00"),
                       format="%Y-%m-%d %H:%M", utc=True, errors="coerce")
    out = pd.DataFrame({"station": station, "time_utc": t})
    n_masked = 0
    for col, code in NDBC_MISSING.items():
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce")
            m = np.isclose(v, code)
            n_masked += int(m.sum())
            out[col] = v.mask(m)
        else:
            out[col] = np.nan
    out.attrs["n_masked"] = n_masked
    return out[out["time_utc"].notna()].reset_index(drop=True)


def ndbc_daily(hourly: pd.DataFrame, tz: str = "America/Chicago") -> pd.DataFrame:
    """Local-calendar-day summaries: mean/max wave height (m), mean wind speed (m/s), mean onshore
    wind component for a west-shore beach (m/s: +WSPD·sin(WDIR), WDIR = direction the wind comes from; positive = from the
    east, i.e. onshore at Chicago), mean water temperature (°C), n hourly obs."""
    h = hourly.copy()
    h["date"] = pd.to_datetime(h["time_utc"].dt.tz_convert(tz).dt.date)
    h["onshore"] = h["WSPD"] * np.sin(np.radians(h["WDIR"]))
    g = h.groupby(["station", "date"])
    out = g.agg(wvht_mean_m=("WVHT", "mean"), wvht_max_m=("WVHT", "max"), wspd_mean_ms=("WSPD", "mean"),
                onshore_wind_ms=("onshore", "mean"), wtmp_mean_c=("WTMP", "mean"), n_obs=("WSPD", "size"))
    return out.reset_index()
