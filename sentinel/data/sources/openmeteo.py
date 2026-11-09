# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Open-Meteo clients: ERA5 reanalysis history and archived NWP forecasts (Previous Runs API).

Endpoints (verified 2026-10-22, free tier, no key, CC BY 4.0, non-commercial):
* ``https://archive-api.open-meteo.com/v1/archive`` -- ERA5 / ERA5-Land (``models=era5_seamless``
  = ERA5-Land 0.1 deg where available, ERA5 0.25 deg otherwise). Used for sites outside gridMET.
* ``https://previous-runs-api.open-meteo.com/v1/forecast`` -- archived NWP at fixed lead offsets:
  ``<var>_previous_dayN`` = value predicted 24*N h before the valid hour (N = 1..7). Coverage
  found 2026-10-22: GFS (``gfs_global``) 2 m temperature from 2021-03, other GFS variables and
  most other models from early 2024. The Single Runs API keeps only recent runs.

Free-tier limits: 600 calls/min, 5,000/h, 10,000/day per IP. A request costs *fractional* calls
(pricing-page calculator, 2026-10-22)::

    n = n_variables * max(n_models, 1) / 10 ;  t = n_days / 14
    calls = max(1, max(n, t * n)) * n_locations

so 27 years of 9 daily ERA5 variables cost ~630 calls per location. :class:`BudgetLedger`
enforces conservative rolling caps across processes (file lock) before every request.

OPERATIONAL vs ORACLE weather: ERA5 is *observed* (reanalysis) weather -- an oracle when used
for future days. Only Previous-Runs NWP values whose ``max_init_utc`` (+ latency) precedes the
forecast issue time may feed "Operational" forecasts (docs/research/01 d.1).
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
USER_AGENT = "SENTINEL-v2 research (public-data download)"

ERA5_DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "dew_point_2m_mean",
    "et0_fao_evapotranspiration",
]
NWP_HOURLY_VARS = ["temperature_2m", "precipitation", "shortwave_radiation"]
MJ_PER_DAY_TO_WM2 = 1e6 / 86400.0  # MJ m-2 day-1 -> W m-2 (daily mean)
FREE_LIMITS = {"minute": 600, "hour": 5000, "day": 10000}


# --------------------------------------------------------------------------- cost model
def call_weight(n_vars: int, n_days: int, n_models: int = 1, n_locations: int = 1) -> float:
    """Fractional API-call cost of one request (Open-Meteo pricing calculator formula)."""
    n = n_vars * max(n_models, 1) / 10.0
    t = n_days / 14.0
    return max(1.0, max(n, t * n)) * n_locations


def n_days(start: str, end: str) -> int:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days + 1


def etc_timezone(utc_offset_h: int) -> str:
    """Fixed standard-time zone name accepted by Open-Meteo (POSIX sign: Etc/GMT+9 == UTC-9)."""
    o = int(utc_offset_h)
    if o == 0:
        return "GMT"
    return f"Etc/GMT{'+' if o < 0 else '-'}{abs(o)}"


# --------------------------------------------------------------------------- request builders
def era5_daily_request(
    lat: float,
    lon: float,
    start: str,
    end: str,
    utc_offset_h: int,
    variables=ERA5_DAILY_VARS,
    model: str = "era5_seamless",
) -> tuple[str, dict]:
    return ARCHIVE_URL, {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "daily": ",".join(variables),
        "models": model,
        "timezone": etc_timezone(utc_offset_h),
        "wind_speed_unit": "ms",
        "cell_selection": "land",
    }


def previous_runs_request(
    lat: float, lon: float, start: str, end: str, model: str, variables=NWP_HOURLY_VARS, leads=range(1, 8)
) -> tuple[str, dict]:
    hourly = [f"{v}_previous_day{n}" for v in variables for n in leads]
    return PREVIOUS_RUNS_URL, {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(hourly),
        "models": model,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }


def request_weight(url: str, params: dict) -> float:
    key = "daily" if "daily" in params else "hourly"
    nv = len(params[key].split(","))
    nm = len(params.get("models", "x").split(","))
    return call_weight(nv, n_days(params["start_date"], params["end_date"]), nm)


# --------------------------------------------------------------------------- parsers
def parse_daily(payload: dict) -> tuple[pd.DataFrame, dict]:
    d = payload.get("daily") or {}
    df = pd.DataFrame(d)
    if "time" in df:
        df["date"] = pd.to_datetime(df.pop("time")).dt.date
    meta = {
        k: payload.get(k) for k in ("latitude", "longitude", "elevation", "utc_offset_seconds", "timezone")
    }
    meta["units"] = payload.get("daily_units") or {}
    return df, meta


def parse_hourly(payload: dict) -> tuple[pd.DataFrame, dict]:
    h = payload.get("hourly") or {}
    df = pd.DataFrame(h)
    if "time" in df:
        df["time_utc"] = pd.to_datetime(df.pop("time")).dt.tz_localize("UTC")
    meta = {
        k: payload.get(k) for k in ("latitude", "longitude", "elevation", "utc_offset_seconds", "timezone")
    }
    meta["units"] = payload.get("hourly_units") or {}
    return df, meta


# --------------------------------------------------------------------------- physics helpers (FAO-56)
def sat_vp_kpa(t_c):
    """Saturation vapour pressure (kPa), FAO-56 eq. 11."""
    t = np.asarray(t_c, float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def pressure_kpa_from_elev(z_m):
    """Standard-atmosphere surface pressure (kPa), FAO-56 eq. 7."""
    z = np.asarray(z_m, float)
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26


def specific_humidity_from_dewpoint(td_c, p_kpa):
    """q (kg/kg) from dew point and pressure: q = 0.622 e / (p - 0.378 e)."""
    e = sat_vp_kpa(td_c)
    return 0.622 * e / (np.asarray(p_kpa, float) - 0.378 * e)


def vpd_fao56(tmax_c, tmin_c, td_c):
    """Daily mean VPD (kPa): (e°(Tmax)+e°(Tmin))/2 - e°(Tdew) (FAO-56 eqs. 12, 14); floored at 0."""
    es = (sat_vp_kpa(tmax_c) + sat_vp_kpa(tmin_c)) / 2.0
    return np.maximum(es - sat_vp_kpa(td_c), 0.0)


def era5_to_harmonised(df: pd.DataFrame, elevation_m: float) -> pd.DataFrame:
    """Open-Meteo daily ERA5 columns -> SENTINEL harmonised forcing columns (units in names)."""
    out = pd.DataFrame({"date": df["date"]})
    out["precip_mm"] = df["precipitation_sum"]
    out["tmax_c"] = df["temperature_2m_max"]
    out["tmin_c"] = df["temperature_2m_min"]
    out["srad_wm2"] = df["shortwave_radiation_sum"] * MJ_PER_DAY_TO_WM2
    out["wind_ms"] = df["wind_speed_10m_mean"]
    out["rhmax_pct"] = df["relative_humidity_2m_max"]
    out["rhmin_pct"] = df["relative_humidity_2m_min"]
    p = pressure_kpa_from_elev(elevation_m if elevation_m is not None else 0.0)
    out["sph_kgkg"] = specific_humidity_from_dewpoint(df["dew_point_2m_mean"], p)
    out["vpd_kpa"] = vpd_fao56(df["temperature_2m_max"], df["temperature_2m_min"], df["dew_point_2m_mean"])
    out["pet_mm"] = df["et0_fao_evapotranspiration"]
    return out


# --------------------------------------------------------------------------- NWP aggregation
def aggregate_previous_runs(
    hourly: pd.DataFrame,
    utc_offset_h: int,
    leads=range(1, 8),
    cycle_hours: int = 6,
    variables=NWP_HOURLY_VARS,
) -> pd.DataFrame:
    """Hourly ``<var>_previous_dayN`` -> daily aggregates per *local standard-time* day and lead.

    For lead N and local day L, every contributing hourly value was predicted >= 24*N h before its
    valid hour, i.e. by a run initialised no later than ``max_init_utc`` = floor_cycle(last valid
    hour - 24*N h). With local-midnight issue times that is before local midnight starting day
    L - N + 1; for a 00 UTC issue use only rows with ``max_init_utc + latency <= issue``.
    Days need all 24 hours of a variable to be aggregated (else NaN; ``n_hours_*`` kept).

    A day uses the hourly stamps 00:00..23:00 local for every variable. Open-Meteo stamps
    precipitation and shortwave radiation at the *end* of their preceding-hour interval, so their
    daily window is [23:00 of L-1, 23:00 of L) local. This is one hour earlier than the calendar day,
    and it keeps the ``max_init_utc`` bound strict. Shifting the window by +1 h would let a run
    initialised *at* the issue time contribute to the last hour.
    """
    h = hourly.copy()
    loc = h["time_utc"] + pd.Timedelta(hours=int(utc_offset_h))
    h["date_local"] = loc.dt.tz_localize(None).dt.normalize()
    rows = []
    for n in leads:
        cols = {v: f"{v}_previous_day{n}" for v in variables}
        if not all(c in h.columns for c in cols.values()):
            continue
        g = h.groupby("date_local")
        agg = pd.DataFrame({"date_local": list(g.groups.keys())})
        agg = agg.set_index("date_local")
        tcol, pcol, scol = (
            cols.get("temperature_2m"),
            cols.get("precipitation"),
            cols.get("shortwave_radiation"),
        )
        if tcol:
            agg["tmax_c"] = g[tcol].max()
            agg["tmin_c"] = g[tcol].min()
            agg["tmean_c"] = g[tcol].mean()
            agg["n_hours_t"] = g[tcol].count()
        if pcol:
            agg["precip_mm"] = g[pcol].sum(min_count=1)
            agg["n_hours_p"] = g[pcol].count()
        if scol:
            agg["srad_wm2"] = g[scol].mean()
            agg["n_hours_s"] = g[scol].count()
        for v, nh in (
            ("tmax_c", "n_hours_t"),
            ("tmin_c", "n_hours_t"),
            ("tmean_c", "n_hours_t"),
            ("precip_mm", "n_hours_p"),
            ("srad_wm2", "n_hours_s"),
        ):
            if v in agg and nh in agg:
                agg.loc[agg[nh] < 24, v] = np.nan
        agg = agg.reset_index()
        last_valid_utc = (
            pd.to_datetime(agg["date_local"]) + pd.Timedelta(hours=23) - pd.Timedelta(hours=int(utc_offset_h))
        ).dt.tz_localize("UTC")
        latest = last_valid_utc - pd.Timedelta(hours=24 * n)
        agg["max_init_utc"] = latest.dt.floor(f"{cycle_hours}h")
        agg["lead_day"] = n
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date_local"] = pd.to_datetime(out["date_local"]).dt.date
    return out


def usable_at(daily: pd.DataFrame, issue_utc, latency_h: float = 6.0) -> pd.DataFrame:
    """Rows of an aggregated NWP table that an operational forecast issued at ``issue_utc`` may use:
    ``max_init_utc + latency_h <= issue_utc`` (choose ``latency_h`` >= the model's publication delay)."""
    t = pd.Timestamp(issue_utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return daily[daily["max_init_utc"] + pd.Timedelta(hours=float(latency_h)) <= t]


# --------------------------------------------------------------------------- budget + HTTP
class DailyLimitReached(RuntimeError):
    pass


class BudgetLedger:
    """Cross-process rolling budget for Open-Meteo calls (JSON-lines ledger + fcntl lock).

    Caps are deliberately below the free-tier limits so that other SENTINEL jobs on the same IP
    keep headroom. ``reserve`` blocks until the weight fits the minute/hour window and raises
    :class:`DailyLimitReached` when the rolling 24 h cap would be exceeded.
    """

    def __init__(self, path: Path, day_cap: float = 8000, hour_cap: float = 4000, minute_cap: float = 400):
        self.path = Path(path)
        self.caps = {"day": day_cap, "hour": hour_cap, "minute": minute_cap}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[tuple[float, float]]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            try:
                rec = json.loads(ln)
                out.append((float(rec["t"]), float(rec["w"])))
            except (ValueError, KeyError):
                continue
        return out

    def used(self, now: float | None = None) -> dict:
        now = now or time.time()
        recs = self._read()
        return {
            "minute": sum(w for t, w in recs if now - t < 60),
            "hour": sum(w for t, w in recs if now - t < 3600),
            "day": sum(w for t, w in recs if now - t < 86400),
        }

    def reserve(self, weight: float, tag: str = "", max_wait_s: float = 3700) -> None:
        if weight > self.caps["minute"]:
            raise ValueError(f"single request weight {weight:.1f} exceeds minute cap; split it")
        t0 = time.time()
        while True:
            with open(self.path.with_suffix(".lock"), "a") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                    now = time.time()
                    u = self.used(now)
                    if u["day"] + weight > self.caps["day"]:
                        raise DailyLimitReached(
                            f"rolling 24h budget {u['day']:.0f}+{weight:.1f} > {self.caps['day']}"
                        )
                    if (
                        u["hour"] + weight <= self.caps["hour"]
                        and u["minute"] + weight <= self.caps["minute"]
                    ):
                        with open(self.path, "a") as fh:
                            fh.write(json.dumps({"t": now, "w": round(weight, 3), "tag": tag}) + "\n")
                        return
                finally:
                    fcntl.flock(lk, fcntl.LOCK_UN)
            if time.time() - t0 > max_wait_s:
                raise TimeoutError("budget wait exceeded")
            time.sleep(5)

    def seconds_until_day_room(self, weight: float) -> float:
        """Seconds until the rolling 24 h window has room for ``weight``."""
        now = time.time()
        recs = sorted((t, w) for t, w in self._read() if now - t < 86400)
        total = sum(w for _, w in recs)
        if total + weight <= self.caps["day"]:
            return 0.0
        for t, w in recs:
            total -= w
            if total + weight <= self.caps["day"]:
                return max(0.0, t + 86400 - now)
        return 0.0


class OpenMeteoClient:
    """Sequential client with gzip-JSON cache (resume-safe), budget ledger and 429 handling."""

    def __init__(self, cache_dir: Path, ledger: BudgetLedger, *, timeout: float = 180, retries: int = 6):
        self.cache_dir = Path(cache_dir)
        self.ledger = ledger
        self.timeout, self.retries = timeout, retries
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        self.stats = {"requests": 0, "cache_hits": 0, "weight": 0.0, "http_429": 0, "errors": 0}

    def _cp(self, url: str, params: dict) -> Path:
        key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json.gz"

    def cached(self, url: str, params: dict) -> dict | None:
        cp = self._cp(url, params)
        if cp.exists():
            with gzip.open(cp, "rt") as fh:
                return json.load(fh)
        return None

    def get(self, url: str, params: dict, tag: str = "") -> dict:
        rec = self.cached(url, params)
        if rec is not None:
            self.stats["cache_hits"] += 1
            return rec
        w = request_weight(url, params)
        last = None
        for k in range(self.retries):
            self.ledger.reserve(w, tag)
            self.stats["requests"] += 1
            self.stats["weight"] += w
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last = repr(e)
                time.sleep(15 * (k + 1))
                continue
            if r.status_code == 200:
                body = r.json()
                rec = {
                    "body": body,
                    "url": url,
                    "params": params,
                    "weight": w,
                    "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                }
                cp = self._cp(url, params)
                cp.parent.mkdir(parents=True, exist_ok=True)
                tmp = cp.with_suffix(".tmp")
                with gzip.open(tmp, "wt") as fh:
                    json.dump(rec, fh)
                tmp.replace(cp)
                return rec
            text = r.text[:300]
            if r.status_code == 429:
                self.stats["http_429"] += 1
                if "Daily" in text or "daily" in text:
                    raise DailyLimitReached(text)
                time.sleep(3700 if ("Hourly" in text or "hourly" in text) else 65)
                continue
            if r.status_code == 400:
                self.stats["errors"] += 1
                raise ValueError(f"HTTP 400 {text}")
            last = f"HTTP {r.status_code}: {text}"
            time.sleep(30 * (k + 1))
        self.stats["errors"] += 1
        raise RuntimeError(f"Open-Meteo GET failed: {url} {params} :: {last}")


def year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-year pieces (each <= 366 days -> ~26 calls at 10 vars)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    y = s.year
    while True:
        a = max(s, pd.Timestamp(f"{y}-01-01"))
        b = min(e, pd.Timestamp(f"{y}-12-31"))
        if a > e:
            break
        out.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
        y += 1
    return out


def cell_id(lat: float, lon: float, res: float = 0.1) -> str:
    """Stable id of the res-degree box containing a point (used to dedupe ERA5 requests)."""
    return f"{int(math.floor(lat / res + 0.5)):+05d}_{int(math.floor(lon / res + 0.5)):+05d}"
