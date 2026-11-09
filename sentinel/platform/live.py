# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Optional real-time values from the USGS Water Data OGC API (``api.waterdata.usgs.gov/ogcapi/v1``).

* Collections: ``latest-continuous`` (most recent value of every continuous series at a site)
  and ``continuous`` (recent window, ISO 8601 duration such as ``PT24H``).
* API key: read only from the ``API_USGS_PAT`` environment variable or
  ``~/.config/sentinel/keys.env``; sent as the ``X-Api-Key`` header (never in the URL), never
  logged, never returned. Without a key the API is usable at a low anonymous rate limit.
* Politeness: at most two concurrent upstream requests, a short in-memory cache, one retry on
  5xx, and on HTTP 429 the ``Retry-After`` interval is honoured for *all* later requests (they
  fail fast without contacting USGS until it has passed).
* Every value is labelled "provisional, live": it is fetched at request time, is not
  quality-assured by SENTINEL and is never used for any reported metric.
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from . import variables as V
from .config import USER_AGENT, USGS_OGC_BASE

log = logging.getLogger("sentinel.platform.live")

USGS_SITE_RE = re.compile(r"^\d{8,15}$")
MAX_HOURS = 168
DATA_STATUS = "provisional, live"
NOTICE = (
    "Real-time values fetched from the USGS Water Data OGC API at request time. Provisional data "
    "are subject to revision (https://waterdata.usgs.gov/provisional-data-statement/). They are not "
    "quality-assured by SENTINEL and are not used for any reported metric."
)
PROPERTIES = (
    "time_series_id",
    "monitoring_location_id",
    "parameter_code",
    "statistic_id",
    "time",
    "value",
    "unit_of_measure",
    "approval_status",
    "qualifier",
    "last_modified",
)


class LiveUnavailable(RuntimeError):
    def __init__(self, status_code: int, detail: str, retry_after: int | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after


def read_api_key(keys_file: str | os.PathLike | None = None) -> str | None:
    """API_USGS_PAT from the environment, else from keys.env. The value is never logged."""
    env = os.environ.get("API_USGS_PAT", "").strip()
    if env:
        return env
    path = Path(keys_file) if keys_file else Path.home() / ".config" / "sentinel" / "keys.env"
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if line.startswith("API_USGS_PAT="):
            val = line.split("=", 1)[1].strip().strip("'\"")
            return val or None
    return None


def _retry_after_seconds(value: str | None, default: int = 60) -> int:
    if not value:
        return default
    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(1, int((when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()))
    except (TypeError, ValueError):
        return default


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_features(doc: dict) -> list[dict]:
    """Tidy records from an OGC API FeatureCollection (values arrive as strings)."""
    out = []
    for f in doc.get("features") or []:
        p = f.get("properties") or {}
        code = p.get("parameter_code")
        name = V.ALL_PARAMS.get(code) if code else None
        q = p.get("qualifier")
        out.append(
            {
                "parameter_code": code,
                "variable": name,
                "label": V.LABELS.get(name) if name else None,
                "statistic_id": p.get("statistic_id"),
                "time_utc": p.get("time"),
                "value": _to_float(p.get("value")),
                "value_raw": p.get("value"),
                "unit": p.get("unit_of_measure"),
                "approval_status": p.get("approval_status"),
                "qualifier": q if q is None or isinstance(q, list) else [q],
                "time_series_id": p.get("time_series_id"),
                "last_modified": p.get("last_modified"),
            }
        )
    out.sort(key=lambda r: (r["parameter_code"] or "", r["statistic_id"] or "", r["time_utc"] or ""))
    return out


class LiveClient:
    def __init__(
        self,
        base_url: str = USGS_OGC_BASE,
        *,
        session: Any = None,
        keys_file: str | None = None,
        cache_ttl_s: float = 300.0,
        timeout_s: float = 20.0,
        max_concurrent: int = 2,
        max_pages: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self.keys_file = keys_file
        self.cache_ttl_s = cache_ttl_s
        self.timeout_s = timeout_s
        self.max_pages = max_pages
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._blocked_until = 0.0

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def blocked_for(self) -> int:
        return max(0, int(round(self._blocked_until - time.time())))

    def _request(self, url: str, params: dict | None, headers: dict) -> dict:
        wait = self.blocked_for()
        if wait > 0:
            raise LiveUnavailable(503, f"USGS rate limit reached; retry after {wait} s", wait)
        last = None
        for attempt in range(2):
            r, err = None, None
            with self._sem:
                try:
                    r = self.session.get(url, params=params, headers=headers, timeout=self.timeout_s)
                except Exception as e:  # noqa: BLE001 — network errors: never echo request details
                    err = e
            if err is not None:
                last = LiveUnavailable(504, f"USGS Water Data API unreachable ({err.__class__.__name__})")
                log.warning("USGS live request failed: %s", err.__class__.__name__)
                time.sleep(1.0 * (attempt + 1))
                continue
            status = getattr(r, "status_code", 0)
            log.info("USGS live request: status=%s", status)
            if status == 200:
                try:
                    return r.json()
                except ValueError as e:
                    raise LiveUnavailable(502, "USGS response was not valid JSON") from e
            if status == 429:
                wait = min(_retry_after_seconds((r.headers or {}).get("Retry-After")), 3600)
                with self._lock:
                    self._blocked_until = max(self._blocked_until, time.time() + wait)
                raise LiveUnavailable(503, f"USGS rate limit reached (HTTP 429); retry after {wait} s", wait)
            if status in (401, 403):
                raise LiveUnavailable(502, f"USGS rejected the request (HTTP {status}); check API_USGS_PAT")
            if status == 404:
                raise LiveUnavailable(404, "USGS has no such monitoring location or collection")
            if 500 <= status < 600:
                last = LiveUnavailable(502, f"USGS Water Data API error (HTTP {status})")
                time.sleep(2.0 * (attempt + 1))
                continue
            raise LiveUnavailable(502, f"unexpected USGS response (HTTP {status})")
        raise last or LiveUnavailable(502, "USGS request failed")

    def _collection(self, collection: str, params: dict) -> tuple[list[dict], dict]:
        key = (collection, tuple(sorted(params.items())))
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]["records"], {**hit[1]["meta"], "cache": "hit"}
        api_key = read_api_key(self.keys_file)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        url = f"{self.base_url}/collections/{collection}/items"
        q = {"f": "json", "skipGeometry": "true", "properties": ",".join(PROPERTIES), **params}
        records: list[dict] = []
        doc = self._request(url, q, headers)
        records.extend(parse_features(doc))
        pages = 1
        nxt = next((ln.get("href") for ln in doc.get("links") or [] if ln.get("rel") == "next"), None)
        while nxt and pages < self.max_pages:
            if not str(nxt).startswith(self.base_url):
                break  # never send the key to another host
            doc = self._request(nxt, None, headers)
            records.extend(parse_features(doc))
            pages += 1
            nxt = next((ln.get("href") for ln in doc.get("links") or [] if ln.get("rel") == "next"), None)
        retrieved = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        meta = {
            "source": "USGS Water Data OGC API",
            "collection": collection,
            "endpoint": url,
            "query": {k: v for k, v in q.items()},
            "retrieved_utc": retrieved,
            "upstream_timestamp": doc.get("timeStamp"),
            "authenticated": bool(api_key),
            "pages": pages,
            "truncated": bool(nxt),
            "cache": "miss",
        }
        with self._lock:
            self._cache[key] = (now + self.cache_ttl_s, {"records": records, "meta": meta})
        return records, meta

    def fetch(self, site_no: str, *, window_hours: int | None = None) -> dict:
        if not USGS_SITE_RE.match(site_no):
            raise LiveUnavailable(422, "live data are available for USGS site numbers (8-15 digits) only")
        loc = f"USGS-{site_no}"
        if window_hours is None:
            records, meta = self._collection(
                "latest-continuous", {"monitoring_location_id": loc, "limit": "500"}
            )
        else:
            if not (1 <= int(window_hours) <= MAX_HOURS):
                raise LiveUnavailable(422, f"window_hours must be between 1 and {MAX_HOURS}")
            records, meta = self._collection(
                "continuous",
                {"monitoring_location_id": loc, "time": f"PT{int(window_hours)}H", "limit": "10000"},
            )
        n_prov = sum(1 for r in records if (r.get("approval_status") or "").lower() != "approved")
        return {
            "site_id": site_no,
            "monitoring_location_id": loc,
            "data_status": DATA_STATUS,
            "notice": NOTICE,
            "n_values": len(records),
            "n_not_approved": n_prov,
            "values": records,
            "provenance": meta,
        }
