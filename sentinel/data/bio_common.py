# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared helpers for the biological-community data sources (NARS, UK EA).

* polite HTTP (generic User-Agent, retries with backoff, honours 429/Retry-After)
* censored-value parsing (``"<0.03"`` -> value 0.03, censored=True, detection_limit=0.03)
* missing-code masking (sentinel numbers and non-numeric qualifier strings -> NaN + flag)
* string IDs with leading zeros (HUC8, COMID, site codes)
* zstd Parquet writer

Nothing here fabricates or imputes data: values that cannot be parsed become NaN and the
original token is preserved in a ``qualifier`` column.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

USER_AGENT = "SENTINEL-v2 research (public-data download)"

# Numeric sentinels that mean "missing" in US federal / NARS files.
MISSING_NUMBERS = (-999999.0, -99999.0, -9999.0, 999999.0)
# Non-numeric tokens that mean "no value" (kept in the qualifier column).
MISSING_TOKENS = {"", "NA", "N/A", "NAN", "NULL", "NONE", ".", "-", "--", "***", "ICE", "EQP",
                  "DIS", "SSN", "ND", "NR", "NS", "MISSING"}

_CENSOR_RE = re.compile(r"^\s*(?P<op><=|>=|<|>)\s*(?P<num>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*$")


# --------------------------------------------------------------------------- HTTP

def make_session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    return s


_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        _SESSION = make_session()
    return _SESSION


def http_request(method: str, url: str, *, params=None, data=None, headers=None, retries: int = 6,
                 backoff: float = 5.0, timeout: float = 300, stream: bool = False,
                 ok_status: Iterable[int] = (200,), sess=None):
    """Request with retries. Honours ``Retry-After`` on 429/503. Raises RuntimeError on failure."""
    import requests

    sess = sess or session()
    last = None
    for k in range(retries):
        try:
            r = sess.request(method, url, params=params, data=data, headers=headers, timeout=timeout,
                             stream=stream)
            if r.status_code in ok_status:
                return r
            last = f"HTTP {r.status_code}: {r.text[:200] if not stream else ''}"
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else backoff * (k + 1) * 2
                log.warning("%s %s -> %s; sleeping %.0fs", method, url, r.status_code, wait)
                time.sleep(min(wait, 600))
                continue
            if 400 <= r.status_code < 500 and r.status_code not in (408, 425):
                raise RuntimeError(f"{method} {url} {params} -> {last}")
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(backoff * (k + 1))
    raise RuntimeError(f"{method} failed after {retries} tries: {url} {params} :: {last}")


def download_file(url: str, dest: Path, *, expected_bytes: int | None = None, chunk: int = 1 << 20,
                  retries: int = 6) -> Path:
    """Stream ``url`` to ``dest`` (atomic rename). Skips if ``dest`` exists with the expected size."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(retries):
        r = http_request("GET", url, stream=True, retries=retries)
        try:
            with open(tmp, "wb") as fh:
                fh.writelines(r.iter_content(chunk))
        finally:
            r.close()
        size = tmp.stat().st_size
        if expected_bytes is None or size == expected_bytes:
            tmp.replace(dest)
            return dest
        log.warning("size mismatch for %s: %d != %d (try %d)", url, size, expected_bytes, k + 1)
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"could not download {url} with expected size {expected_bytes}")


def head_info(url: str) -> dict:
    r = http_request("HEAD", url, retries=4)
    h = r.headers
    return {"url": url, "content_length": int(h["Content-Length"]) if h.get("Content-Length") else None,
            "last_modified": h.get("Last-Modified"), "etag": h.get("ETag")}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ value parsing

def parse_censored(values) -> pd.DataFrame:
    """Parse a column of reported results.

    Returns columns ``value`` (float), ``censored`` (bool, True for "<x"/"<=x"), ``censor_op``
    ("<", ">", "" ...), ``detection_limit`` (float for "<x", else NaN) and ``qualifier`` (the
    original token when it was not a plain number, else ""). Non-numeric text (e.g. "No flow")
    gives value NaN with the text in ``qualifier``. Right-censored values (">x") keep ``value`` = x
    and ``censored`` False but ``censor_op`` ">" so they can be excluded if needed.
    """
    s = pd.Series(values, copy=False).astype("string").fillna("").str.strip()
    num = pd.to_numeric(s, errors="coerce")
    m = s.str.extract(_CENSOR_RE)
    op = m["op"].fillna("")
    cnum = pd.to_numeric(m["num"], errors="coerce")
    value = num.where(num.notna(), cnum)
    is_left = op.isin(["<", "<="])
    qualifier = s.where(num.isna(), "")
    out = pd.DataFrame({
        "value": value.astype("float64"),
        "censored": is_left.to_numpy(bool),
        "censor_op": op.to_numpy(object),
        "detection_limit": cnum.where(is_left).astype("float64"),
        "qualifier": qualifier.to_numpy(object),
    }, index=s.index)
    return out


def mask_missing(series: pd.Series, *, extra_numbers: Iterable[float] = ()) -> tuple[pd.Series, pd.Series]:
    """Numeric coercion with missing-code masking. Returns (values, flag) where flag holds the
    original token for anything that was masked (sentinel number or non-numeric string)."""
    raw = series.astype("string").fillna("").str.strip()
    val = pd.to_numeric(raw, errors="coerce")
    sentinels = set(MISSING_NUMBERS) | {float(x) for x in extra_numbers}
    is_sentinel = val.isin(list(sentinels))
    nonnum = val.isna() & (raw != "")
    flag = raw.where(is_sentinel | nonnum, "")
    val = val.mask(is_sentinel)
    return val.astype("float64"), flag.astype(object)


def id_str(series: pd.Series, width: int | None = None) -> pd.Series:
    """IDs as strings: strip float artefacts ('1234.0'), keep leading zeros, optional zero-pad."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    s = s.mask(s.isin(["", "nan", "NaN", "None", "<NA>"]))
    if width:
        s = s.where(~s.fillna("").str.fullmatch(r"\d+"), s.str.zfill(width))
    return s.astype(object).where(s.notna(), None)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").upper() for c in df.columns]
    return df


def read_csv_str(path_or_buf, **kw) -> pd.DataFrame:
    """Read a CSV with every column as string (parsing is done explicitly afterwards)."""
    kw.setdefault("dtype", str)
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [])
    kw.setdefault("low_memory", False)
    try:
        df = pd.read_csv(path_or_buf, encoding="utf-8-sig", **kw)
    except UnicodeDecodeError:
        if hasattr(path_or_buf, "seek"):
            path_or_buf.seek(0)
        df = pd.read_csv(path_or_buf, encoding="latin-1", **kw)
    return clean_columns(df)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path


def to_float(series: pd.Series) -> pd.Series:
    """Numeric coercion with NARS/US missing codes masked (no flag column)."""
    v, _ = mask_missing(series)
    return v


def first_present(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return n
    return None


def nan_if_empty(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.mask(s == "").astype(object).where(s.notna() & (s != ""), None)


def bool_from_yn(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[s.isin(["Y", "YES", "TRUE", "T", "1"])] = True
    out[s.isin(["N", "NO", "FALSE", "F", "0"])] = False
    return out


def safe_log10(x, floor: float) -> np.ndarray:
    """log10 with an explicit floor (documented clip; used only for modelling targets)."""
    x = np.asarray(x, float)
    return np.log10(np.clip(x, floor, None))
