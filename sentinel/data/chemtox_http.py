# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Polite HTTP helpers shared by the ChemTox data sources (ADORE, ECOTOX, NRC, PubChem, CompTox).

Rules (docs/CONTRIBUTING_V2.md): generic User-Agent, retries with backoff, honour HTTP 429 /
``Retry-After``, a per-host minimum interval between requests, no credentials or personal data in
requests. Downloads stream to ``<dest>.part`` while hashing, and are renamed only after the size /
sha256 checks pass, so an interrupted run never leaves a truncated file under the final name.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "SENTINEL-v2 research (public-data download)"
log = logging.getLogger(__name__)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


class RateLimiter:
    """Thread-safe minimum interval between successive requests (e.g. 0.21 s for <= 5 req/s)."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval_s - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def _retry_after_seconds(resp: requests.Response, default: float) -> float:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return default
    try:
        return max(float(ra), 0.0)
    except ValueError:
        try:
            when = parsedate_to_datetime(ra)
            return max((when - _dt.datetime.now(when.tzinfo)).total_seconds(), 0.0)
        except (TypeError, ValueError):
            return default


def get(
    url: str,
    *,
    params: dict | None = None,
    session: requests.Session | None = None,
    retries: int = 6,
    backoff: float = 5.0,
    timeout: float = 120.0,
    ok_status: tuple[int, ...] = (200,),
    pass_status: tuple[int, ...] = (),
    rate_limiter: RateLimiter | None = None,
    method: str = "GET",
    data: Any = None,
    headers: dict | None = None,
) -> requests.Response:
    """Request with retries. Statuses in ``ok_status``/``pass_status`` are returned to the caller
    (``pass_status`` e.g. 404 = "no record" for lookup APIs); 429/5xx are retried with backoff."""
    s = session or make_session()
    last = None
    for k in range(retries):
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            r = s.request(method, url, params=params, data=data, headers=headers, timeout=timeout)
        except requests.RequestException as e:  # network hiccup -> retry
            last = repr(e)
            time.sleep(backoff * (k + 1))
            continue
        if r.status_code in ok_status or r.status_code in pass_status:
            return r
        last = f"HTTP {r.status_code}: {r.text[:200]!r}"
        if r.status_code in (429, 503):
            time.sleep(_retry_after_seconds(r, backoff * (k + 1)))
            continue
        if 500 <= r.status_code < 600 or r.status_code == 408:
            time.sleep(backoff * (k + 1))
            continue
        break  # other 4xx: do not hammer the server
    raise RuntimeError(f"{method} failed: {url} params={params} :: {last}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    dest: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    session: requests.Session | None = None,
    retries: int = 5,
    backoff: float = 15.0,
    timeout: float = 300.0,
    chunk: int = 1 << 20,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Stream ``url`` to ``dest`` with sha256 + size verification; return a provenance record.

    If ``dest`` exists and (when given) matches ``expected_sha256``/``expected_size`` it is kept
    and re-hashed instead of downloaded again (resume-safe re-runs).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dest.exists():
        digest = sha256_file(dest)
        size = dest.stat().st_size
        if (expected_sha256 is None or digest == expected_sha256) and (
            expected_size is None or size == expected_size
        ):
            return {"url": url, "path": str(dest), "bytes": size, "sha256": digest, "reused_existing": True}
        log.warning("existing %s fails verification; downloading again", dest)
    s = session or make_session()
    part = dest.with_name(dest.name + ".part")
    last = None
    for k in range(retries):
        try:
            with s.get(url, stream=True, timeout=timeout) as r:
                if r.status_code in (429, 503):
                    time.sleep(_retry_after_seconds(r, backoff * (k + 1)))
                    last = f"HTTP {r.status_code}"
                    continue
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}"
                    if r.status_code == 404:
                        raise FileNotFoundError(f"404 for {url}")
                    time.sleep(backoff * (k + 1))
                    continue
                h = hashlib.sha256()
                n = 0
                with open(part, "wb") as fh:
                    for block in r.iter_content(chunk_size=chunk):
                        if block:
                            fh.write(block)
                            h.update(block)
                            n += len(block)
                rec = {
                    "url": url,
                    "path": str(dest),
                    "bytes": n,
                    "sha256": h.hexdigest(),
                    "http_last_modified": r.headers.get("Last-Modified"),
                    "http_etag": r.headers.get("ETag"),
                    "http_content_length": r.headers.get("Content-Length"),
                    "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    "reused_existing": False,
                }
            cl = rec["http_content_length"]
            if cl is not None and int(cl) != n:
                last = f"truncated: got {n} of {cl} bytes"
                time.sleep(backoff * (k + 1))
                continue
            if expected_size is not None and n != expected_size:
                raise RuntimeError(f"size mismatch for {url}: {n} != {expected_size}")
            if expected_sha256 is not None and rec["sha256"] != expected_sha256:
                raise RuntimeError(f"sha256 mismatch for {url}: {rec['sha256']} != {expected_sha256}")
            part.replace(dest)
            return rec
        except requests.RequestException as e:
            last = repr(e)
            time.sleep(backoff * (k + 1))
    if part.exists():
        part.unlink()
    raise RuntimeError(f"download failed after {retries} tries: {url} :: {last}")
