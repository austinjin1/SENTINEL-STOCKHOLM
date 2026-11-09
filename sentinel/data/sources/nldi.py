# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""USGS Network-Linked Data Index (NLDI) client: site -> NHDPlus V2 COMID and main-stem navigation.

Service root moved to ``https://api.water.usgs.gov/nldi`` (the old labs.waterdata.usgs.gov
path returns 404). The same site can resolve to *different* COMIDs depending on the lookup
source (docs/research/07 c.1), so every link records the method used:

* ``nldi_nwissite``  -- ``/linked-data/nwissite/USGS-<no>``: NWIS surface-water sites that are
  reach-indexed (hydrolocation with reachcode + measure). Preferred for gauges.
* ``nldi_position``  -- ``/linked-data/comid/position?coords=POINT(lon lat)``: point-in-catchment
  of the site coordinates (no measure). Used for NEON and for USGS sites missing from nwissite.

Navigation (``UM`` upstream main stem, ``DM`` downstream main stem) returns features of a data
source (here ``nwissite``) within ``distance`` km. NLDI does not return along-channel distances;
those are computed from NHDPlus V2 VAA ``pathlength``/``lengthkm`` and the reach measure
(:func:`network_position_km`).

All functions are pure (build request / parse response); :class:`NLDIClient` adds polite HTTP
(<= 4 workers, retries, 429/Retry-After, on-disk gzip JSON cache so runs are resume-safe).
"""

from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE = "https://api.water.usgs.gov/nldi/linked-data"
USER_AGENT = "SENTINEL-v2 research (public-data download)"
NAV_MODES = ("UM", "DM", "UT", "DD")


# --------------------------------------------------------------------------- request builders
def feature_request(source: str, identifier: str) -> tuple[str, dict]:
    """Feature lookup, e.g. ``feature_request('nwissite', 'USGS-05586300')``."""
    return f"{BASE}/{source}/{identifier}", {"f": "json"}


def position_request(lon: float, lat: float) -> tuple[str, dict]:
    """Point-in-catchment COMID lookup (WKT uses lon lat order)."""
    return f"{BASE}/comid/position", {"f": "json", "coords": f"POINT({lon:.7f} {lat:.7f})"}


def navigation_request(
    source: str, identifier: str, mode: str, data_source: str = "nwissite", distance_km: float = 500
) -> tuple[str, dict]:
    """Features of ``data_source`` along ``mode`` navigation from a feature (or ``source='comid'``)."""
    if mode not in NAV_MODES:
        raise ValueError(f"mode must be one of {NAV_MODES}")
    return (
        f"{BASE}/{source}/{identifier}/navigation/{mode}/{data_source}",
        {"distance": f"{distance_km:g}", "f": "json"},
    )


# --------------------------------------------------------------------------- parsers
def _int_or_none(v: Any) -> int | None:
    try:
        return None if v is None or v == "" else int(v)
    except (TypeError, ValueError):
        return None


def parse_feature(payload: dict | None) -> dict:
    """Parse a feature / position FeatureCollection into a flat dict (COMID as string)."""
    out = {
        "comid": None,
        "reachcode": None,
        "measure": None,
        "mainstem": None,
        "name": None,
        "nldi_source": None,
        "nldi_type": None,
        "nldi_lon": None,
        "nldi_lat": None,
    }
    if not payload:
        return out
    feats = payload.get("features") or []
    if not feats:
        return out
    f = feats[0]
    p = f.get("properties") or {}
    comid = _int_or_none(p.get("comid"))
    if comid is None and p.get("source") == "comid":  # position lookup: identifier is the COMID
        comid = _int_or_none(p.get("identifier") or f.get("id"))
    geom = f.get("geometry") or {}
    lon = lat = None
    if geom.get("type") == "Point":
        lon, lat = geom["coordinates"][:2]
    meas = p.get("measure")
    out.update(
        {
            "comid": None if comid is None else str(comid),
            "reachcode": p.get("reachcode") or None,
            "measure": None if meas in (None, "") else float(meas),
            "mainstem": p.get("mainstem") or None,
            "name": p.get("name"),
            "nldi_source": p.get("source"),
            "nldi_type": p.get("type"),
            "nldi_lon": lon,
            "nldi_lat": lat,
        }
    )
    return out


def parse_navigation(payload: dict | None) -> pd.DataFrame:
    """Navigation FeatureCollection -> one row per returned feature."""
    cols = ["identifier", "comid", "reachcode", "measure", "name", "source"]
    if not payload:
        return pd.DataFrame(columns=cols)
    rows = []
    for f in payload.get("features") or []:
        p = f.get("properties") or {}
        comid = _int_or_none(p.get("comid") if "comid" in p else p.get("nhdplus_comid"))
        meas = p.get("measure")
        rows.append(
            {
                "identifier": str(p.get("identifier") if p.get("identifier") is not None else f.get("id")),
                "comid": None if comid is None else str(comid),
                "reachcode": p.get("reachcode") or None,
                "measure": None if meas in (None, "") else float(meas),
                "name": p.get("name"),
                "source": p.get("source"),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def network_position_km(pathlength_km: float, lengthkm: float, measure: float | None) -> tuple[float, str]:
    """Distance (km) from the site to the network terminal outlet along the main path.

    NHDPlus V2 ``pathlength`` is measured from the *downstream* end of the flowline; reach
    measures run 0 (downstream end) .. 100 (upstream end). Without a measure (position lookups)
    the flowline midpoint is used and flagged.
    """
    if measure is None or pd.isna(measure):
        return pathlength_km + 0.5 * lengthkm, "vaa_pathlength_midpoint"
    m = min(max(float(measure), 0.0), 100.0)
    return pathlength_km + lengthkm * m / 100.0, "vaa_pathlength_measure"


# --------------------------------------------------------------------------- HTTP client
def _cache_key(url: str, params: dict) -> str:
    s = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(s.encode()).hexdigest()


class NLDIClient:
    """Polite, cached NLDI GET client (thread-safe; one ``requests.Session`` per thread)."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        timeout: float = 120,
        retries: int = 6,
        backoff: float = 5.0,
        min_interval: float = 1.25,
        max_429_waits: int = 12,
        offline: bool = False,
    ):
        """``min_interval`` is a *global* spacing between requests across threads: the USGS API
        gateway in front of NLDI allows 3,600 requests/hour per IP (``x-ratelimit-limit``), shared
        with every other job on the machine; 1.25 s keeps us at <= 2,880/h."""
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.min_interval, self.max_429_waits = min_interval, max_429_waits
        self.offline = offline  # cache-only: uncached requests return status -2 (no network)
        self._local = threading.local()
        self.stats = {"requests": 0, "cache_hits": 0, "errors": 0, "http_429": 0, "ratelimit_remaining": None}
        self._lock = threading.Lock()
        self._pace_lock = threading.Lock()
        self._next_t = 0.0

    def _pace(self) -> None:
        with self._pace_lock:
            now = time.monotonic()
            wait = self._next_t - now
            self._next_t = max(now, self._next_t) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update(
                {"User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip"}
            )
            self._local.s = s
        return s

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / key[:2] / f"{key}.json.gz"

    def get(self, url: str, params: dict) -> dict:
        """Return ``{'status': int, 'body': json|None, 'retrieved_utc': str}``; 404s are cached."""
        key = _cache_key(url, params)
        cp = self._cache_path(key)
        if cp is not None and cp.exists():
            try:
                with gzip.open(cp, "rt") as fh:
                    rec = json.load(fh)
                with self._lock:
                    self.stats["cache_hits"] += 1
                return rec
            except (OSError, json.JSONDecodeError):
                cp.unlink(missing_ok=True)
        if self.offline:
            with self._lock:
                self.stats["not_cached"] = self.stats.get("not_cached", 0) + 1
            return {"status": -2, "body": None, "detail": "not cached (offline)", "url": url, "params": params}
        last = None
        k = n429 = 0
        while k < self.retries:
            try:
                self._pace()
                with self._lock:
                    self.stats["requests"] += 1
                r = self._session().get(url, params=params, timeout=self.timeout)
                self.stats["ratelimit_remaining"] = r.headers.get("x-ratelimit-remaining")
                if r.status_code == 200:
                    rec = {
                        "status": 200,
                        "body": r.json(),
                        "url": url,
                        "params": params,
                        "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    }
                    self._write(cp, rec)
                    return rec
                if r.status_code in (400, 404):
                    rec = {
                        "status": r.status_code,
                        "body": None,
                        "detail": r.text[:300],
                        "url": url,
                        "params": params,
                        "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    }
                    self._write(cp, rec)
                    return rec
                if r.status_code == 429:  # gateway quota: wait the full Retry-After, not a retry
                    with self._lock:
                        self.stats["http_429"] += 1
                    n429 += 1
                    if n429 > self.max_429_waits:
                        last = "HTTP 429 (gave up)"
                        break
                    wait = float(r.headers.get("Retry-After", 300))
                    with self._pace_lock:  # stop every thread until the window reopens
                        self._next_t = max(self._next_t, time.monotonic() + min(wait, 3600) + 5)
                    continue
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except (requests.RequestException, ValueError) as e:
                last = repr(e)
            time.sleep(self.backoff * (2**k) * (0.75 + 0.5 * random.random()))
            k += 1
        with self._lock:
            self.stats["errors"] += 1
        return {"status": -1, "body": None, "detail": str(last), "url": url, "params": params}

    @staticmethod
    def _write(cp: Path | None, rec: dict) -> None:
        if cp is None:
            return
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump(rec, fh)
        tmp.replace(cp)

    # convenience wrappers ----------------------------------------------------------------
    def feature(self, source: str, identifier: str) -> tuple[dict, int]:
        rec = self.get(*feature_request(source, identifier))
        return parse_feature(rec.get("body")), rec["status"]

    def position(self, lon: float, lat: float) -> tuple[dict, int]:
        rec = self.get(*position_request(lon, lat))
        return parse_feature(rec.get("body")), rec["status"]

    def navigate(
        self, source: str, identifier: str, mode: str, data_source: str = "nwissite", distance_km: float = 500
    ) -> tuple[pd.DataFrame, int]:
        rec = self.get(*navigation_request(source, identifier, mode, data_source, distance_km))
        return parse_navigation(rec.get("body")), rec["status"]
