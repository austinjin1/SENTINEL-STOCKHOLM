# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared helpers for the label / event-catalogue clients (``events_*.py``, ``downstream_*.py``).

* one HTTP session with the project User-Agent, retries with exponential backoff, and
  HTTP 429 / ``Retry-After`` handling (CONTRIBUTING_V2 politeness rules);
* pagers for Socrata SODA 2.x (data.cityofchicago.org, data.ny.gov), ArcGIS REST
  FeatureServer/MapServer layers and the CKAN datastore API (data.ca.gov);
* a dependency-free XLSX reader (zip + XML; ``openpyxl`` is not in the reference env) and a
  stdlib HTML table parser (``lxml``/``bs4`` are not in the reference env);
* small tidy-table utilities: string IDs with zero padding, calendar/coordinate validation
  that NaN-s impossible values but keeps a flag column, zstd Parquet writer, coverage stats.

All functions are pure apart from the network calls; parsing functions take text/bytes so
they can be unit-tested on recorded fixtures without network access.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import io
import json
import re
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

USER_AGENT = "SENTINEL-v2 research (public-data download)"
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}

_SESSION: requests.Session | None = None


class HTTPFailure(RuntimeError):
    """Raised when a request fails permanently (non-retryable status or retries exhausted)."""


def session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
        _SESSION = s
    return _SESSION


def _retry_after(resp: requests.Response, default: float) -> float:
    ra = resp.headers.get("Retry-After", "")
    if ra.strip().isdigit():
        return min(float(ra), 600.0)
    return default


def http_get(url: str, params: dict | None = None, *, retries: int = 6, backoff: float = 5.0,
             timeout: float = 180.0, stream: bool = False, allow_404: bool = False,
             headers: dict | None = None) -> requests.Response | None:
    """GET with retries. Returns the response (status 200/206), or None for 404 if allowed."""
    last = ""
    for attempt in range(retries):
        wait = min(backoff * (2 ** attempt), 300.0)
        try:
            r = session().get(url, params=params, timeout=timeout, stream=stream, headers=headers)
        except requests.RequestException as e:  # network hiccup → retry
            last = repr(e)
            time.sleep(wait)
            continue
        if r.status_code in (200, 206):
            return r
        if r.status_code == 404 and allow_404:
            return None
        if r.status_code in RETRYABLE:
            last = f"HTTP {r.status_code}"
            time.sleep(_retry_after(r, wait))
            continue
        raise HTTPFailure(f"HTTP {r.status_code} for {r.url}: {r.text[:300]!r}")
    raise HTTPFailure(f"GET failed after {retries} tries: {url} {params or ''} :: {last}")


def get_json(url: str, params: dict | None = None, **kw) -> Any:
    r = http_get(url, params, **kw)
    if r is None:
        return None
    return r.json()


def get_text(url: str, params: dict | None = None, *, encoding: str | None = None, **kw) -> str:
    r = http_get(url, params, **kw)
    if r is None:
        return ""
    if encoding:
        r.encoding = encoding
    return r.text


def download_file(url: str, dest: str | Path, *, params: dict | None = None, overwrite: bool = False,
                  timeout: float = 900.0, retries: int = 5, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest`` (atomic via ``.part``). Skips if ``dest`` exists unless overwrite."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = ""
    for attempt in range(retries):
        try:
            r = http_get(url, params, timeout=timeout, stream=True, retries=3)
            with open(part, "wb") as fh:
                for block in r.iter_content(chunk_size=chunk):
                    if block:
                        fh.write(block)
            part.replace(dest)
            return dest
        except (requests.RequestException, HTTPFailure) as e:
            last = repr(e)
            time.sleep(min(10.0 * (attempt + 1), 120.0))
    raise HTTPFailure(f"download failed: {url} :: {last}")


def http_head(url: str, *, timeout: float = 60.0) -> dict:
    """Return selected headers (size, last-modified, etag) without downloading the body."""
    r = session().head(url, timeout=timeout, allow_redirects=True)
    return {"status": r.status_code, "url": r.url,
            "content_length": r.headers.get("Content-Length"),
            "last_modified": r.headers.get("Last-Modified"), "etag": r.headers.get("ETag")}


# ----------------------------------------------------------------------------- Socrata (SODA 2.x)

def socrata_metadata(domain: str, dataset: str) -> dict:
    """Dataset metadata (name, rowsUpdatedAt, license, attribution, columns) from /api/views."""
    return get_json(f"https://{domain}/api/views/{dataset}.json")


def socrata_metadata_summary(meta: dict) -> dict:
    """The provenance fields a datasheet needs, from :func:`socrata_metadata` output."""
    lic = meta.get("license") or {}
    def _ts(v):
        return (_dt.datetime.fromtimestamp(int(v), tz=_dt.timezone.utc).isoformat(timespec="seconds")
                if v else None)
    return {
        "id": meta.get("id"), "name": meta.get("name"),
        "attribution": meta.get("attribution"),
        "license": lic.get("name") or meta.get("licenseId"),
        "rows_updated_at": _ts(meta.get("rowsUpdatedAt")),
        "view_last_modified": _ts(meta.get("viewLastModified")),
        "columns": [c.get("fieldName") for c in meta.get("columns", [])],
    }


def socrata_csv(domain: str, dataset: str, *, select: str | None = None, where: str | None = None,
                order: str = ":id", page_size: int = 50000, sleep: float = 0.5,
                max_rows: int | None = None) -> pd.DataFrame:
    """All rows of a Socrata dataset as an all-string DataFrame (CSV endpoint, stable :id order)."""
    url = f"https://{domain}/resource/{dataset}.csv"
    frames, offset = [], 0
    while True:
        params = {"$limit": page_size, "$offset": offset, "$order": order}
        if select:
            params["$select"] = select
        if where:
            params["$where"] = where
        text = get_text(url, params)
        page = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        frames.append(page)
        offset += len(page)
        if len(page) < page_size or (max_rows is not None and offset >= max_rows):
            break
        time.sleep(sleep)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out if max_rows is None else out.iloc[:max_rows]


def socrata_count(domain: str, dataset: str, where: str | None = None) -> int:
    params = {"$select": "count(*) as n"}
    if where:
        params["$where"] = where
    rows = get_json(f"https://{domain}/resource/{dataset}.json", params)
    return int(rows[0]["n"]) if rows else 0


# ----------------------------------------------------------------------------- ArcGIS REST

def arcgis_layer_info(layer_url: str) -> dict:
    return get_json(layer_url.rstrip("/"), {"f": "json"})


def arcgis_count(layer_url: str, where: str = "1=1") -> int:
    d = get_json(layer_url.rstrip("/") + "/query", {"where": where, "returnCountOnly": "true", "f": "json"})
    return int(d.get("count", 0))


def arcgis_query(layer_url: str, *, where: str = "1=1", out_fields: str = "*",
                 return_geometry: bool = False, out_sr: int = 4326, page_size: int | None = None,
                 fmt: str = "json", order_by: str | None = None, sleep: float = 0.3,
                 extra: dict | None = None) -> list[dict]:
    """All features of a layer query, paged with resultOffset (falls back to objectId batches).

    ``fmt='json'`` returns Esri JSON features ({"attributes", "geometry"}); ``fmt='geojson'``
    returns GeoJSON features. Geometry precision/generalisation can be set via ``extra``
    (e.g. {"geometryPrecision": 6}).
    """
    url = layer_url.rstrip("/") + "/query"
    info = arcgis_layer_info(layer_url)
    max_rc = int(info.get("maxRecordCount") or 1000)
    page = min(page_size or max_rc, max_rc)
    oid = info.get("objectIdField") or next(
        (f["name"] for f in info.get("fields", []) if f.get("type") == "esriFieldTypeOID"), "OBJECTID")
    base = {"where": where, "outFields": out_fields, "returnGeometry": str(return_geometry).lower(),
            "outSR": out_sr, "f": fmt}
    if extra:
        base.update(extra)
    supports_paging = bool((info.get("advancedQueryCapabilities") or {}).get("supportsPagination", True))
    feats: list[dict] = []
    if supports_paging:
        offset = 0
        while True:
            p = dict(base, resultOffset=offset, resultRecordCount=page, orderByFields=order_by or oid)
            d = get_json(url, p)
            if "error" in d:
                raise HTTPFailure(f"ArcGIS error {d['error']} for {url}")
            batch = d.get("features", [])
            feats.extend(batch)
            more = d.get("exceededTransferLimit") or (d.get("properties") or {}).get("exceededTransferLimit")
            if not batch or not more:
                break
            offset += len(batch)
            time.sleep(sleep)
        return feats
    ids = get_json(url, {"where": where, "returnIdsOnly": "true", "f": "json"}).get("objectIds") or []
    for k in range(0, len(ids), page):
        p = dict(base, objectIds=",".join(str(i) for i in sorted(ids)[k:k + page]))
        p.pop("where")
        d = get_json(url, p)
        feats.extend(d.get("features", []))
        time.sleep(sleep)
    return feats


def esri_attributes_frame(features: Sequence[dict]) -> pd.DataFrame:
    """Esri JSON features → DataFrame of attributes (plus x/y for point geometry if present)."""
    rows = []
    for f in features:
        a = dict(f.get("attributes") or f.get("properties") or {})
        g = f.get("geometry") or {}
        if isinstance(g, dict) and "x" in g and "y" in g:
            a["_x"], a["_y"] = g["x"], g["y"]
        rows.append(a)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- CKAN (data.ca.gov)

def ckan_package(base: str, package_id: str) -> dict:
    d = get_json(f"{base.rstrip('/')}/api/3/action/package_show", {"id": package_id})
    if not d.get("success"):
        raise HTTPFailure(f"CKAN package_show failed for {package_id}")
    return d["result"]


def ckan_datastore_rows(base: str, resource_id: str, *, page_size: int = 10000,
                        sleep: float = 0.5) -> tuple[list[dict], list[dict]]:
    """All records of a CKAN datastore resource → (records, fields)."""
    url = f"{base.rstrip('/')}/api/3/action/datastore_search"
    recs: list[dict] = []
    fields: list[dict] = []
    offset, total = 0, None
    while True:
        d = get_json(url, {"resource_id": resource_id, "limit": page_size, "offset": offset})
        if not d.get("success"):
            raise HTTPFailure(f"CKAN datastore_search failed for {resource_id}")
        res = d["result"]
        if not fields:
            fields = res.get("fields", [])
        total = res.get("total", total)
        batch = res.get("records", [])
        recs.extend(batch)
        offset += len(batch)
        if not batch or (total is not None and offset >= int(total)):
            break
        time.sleep(sleep)
    return recs, fields


# ----------------------------------------------------------------------------- XLSX (stdlib)

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
_BUILTIN_DATE_FMTS = set(range(14, 23)) | set(range(27, 37)) | set(range(45, 48)) | set(range(50, 59))


def _xlsx_zip(src: str | Path | bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)


def _is_date_format(code: str) -> bool:
    code = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", code or "")  # drop literals, colours, escapes
    return bool(re.search(r"[dmyhs]", code, flags=re.I)) and not re.fullmatch(r"[#0.,%E+\-\s]*", code)


def _xlsx_meta(z: zipfile.ZipFile) -> tuple[dict[str, str], list[str], list[bool], bool]:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid_target = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", _NS)}
    sheets = {}
    for s in wb.findall("m:sheets/m:sheet", _NS):
        tgt = rid_target[s.get(f"{{{_NS['r']}}}id")].lstrip("/")
        sheets[s.get("name")] = tgt if tgt.startswith("xl/") else "xl/" + tgt
    pr = wb.find("m:workbookPr", _NS)
    date1904 = pr is not None and pr.get("date1904") in ("1", "true")
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in sst.findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    is_date: list[bool] = []
    if "xl/styles.xml" in z.namelist():
        st = ET.fromstring(z.read("xl/styles.xml"))
        custom = {int(n.get("numFmtId")): n.get("formatCode", "")
                  for n in st.findall("m:numFmts/m:numFmt", _NS)}
        for xf in st.findall("m:cellXfs/m:xf", _NS):
            fid = int(xf.get("numFmtId", "0"))
            is_date.append(fid in _BUILTIN_DATE_FMTS or (fid in custom and _is_date_format(custom[fid])))
    return sheets, shared, is_date, date1904


def xlsx_sheet_names(src: str | Path | bytes) -> list[str]:
    with _xlsx_zip(src) as z:
        return list(_xlsx_meta(z)[0])


def excel_serial_to_datetime(v: float, date1904: bool = False) -> _dt.datetime:
    base = _dt.datetime(1904, 1, 1) if date1904 else _dt.datetime(1899, 12, 30)
    return base + _dt.timedelta(days=float(v))


def _short_number(text: str) -> str:
    """Excel stores 17 significant digits ('1.1000000000000001'); return the shortest repr."""
    if re.fullmatch(r"-?\d+", text):
        return text
    try:
        v = float(text)
    except ValueError:
        return text
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(src: str | Path | bytes, sheet: str | int | None = None, *, header: int | None = 0,
              skip_blank_rows: bool = True) -> pd.DataFrame:
    """Read one worksheet into an all-string DataFrame.

    Date-formatted numeric cells become ISO strings (``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SS``);
    other numbers keep Excel's stored text (e.g. ``2015``, ``8.4``). ``header`` is the 0-based
    row used for column names (None → numbered columns).
    """
    with _xlsx_zip(src) as z:
        sheets, shared, is_date, date1904 = _xlsx_meta(z)
        names = list(sheets)
        name = names[sheet] if isinstance(sheet, int) else (sheet or names[0])
        root = ET.fromstring(z.read(sheets[name]))
    grid: list[list[str]] = []
    for row in root.iter(f"{{{_NS['m']}}}row"):
        cells: dict[int, str] = {}
        for k, c in enumerate(row.findall("m:c", _NS)):
            ref = c.get("r")
            ci = _col_index(ref) if ref else k
            t = c.get("t", "n")
            v = c.find("m:v", _NS)
            text = v.text if v is not None and v.text is not None else ""
            if t == "s" and text != "":
                text = shared[int(text)]
            elif t == "inlineStr":
                text = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            elif t == "b":
                text = "TRUE" if text == "1" else ("FALSE" if text == "0" else text)
            elif t == "n" and text != "":
                s = int(c.get("s", "0"))
                if s < len(is_date) and is_date[s]:
                    d = excel_serial_to_datetime(float(text), date1904)
                    d = d.replace(microsecond=0) + (_dt.timedelta(seconds=1) if d.microsecond >= 500000
                                                    else _dt.timedelta(0))
                    text = (d.date().isoformat() if d.time() == _dt.time(0)
                            else d.isoformat(timespec="seconds"))
                else:
                    text = _short_number(text)
            cells[ci] = text
        r_attr = row.get("r")
        r_idx = int(r_attr) - 1 if r_attr else len(grid)
        while len(grid) < r_idx:
            grid.append([])
        width = max(cells) + 1 if cells else 0
        grid.append([cells.get(i, "") for i in range(width)])
    if skip_blank_rows:
        grid = [r for r in grid if any(x.strip() for x in r)]
    if not grid:
        return pd.DataFrame()
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    if header is None:
        return pd.DataFrame(grid, dtype=str)
    cols = [c.strip() or f"col_{i}" for i, c in enumerate(grid[header])]
    seen: dict[str, int] = {}
    uniq = []
    for c in cols:  # de-duplicate header names
        seen[c] = seen.get(c, 0) + 1
        uniq.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return pd.DataFrame(grid[header + 1:], columns=uniq, dtype=str)


# ----------------------------------------------------------------------------- HTML tables (stdlib)

class _TableParser(HTMLParser):
    def __init__(self, expand_colspan: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.expand_colspan = expand_colspan
        self._span = 1
        self.tables: list[list[list[str]]] = []
        self.links: list[tuple[str, str]] = []
        self._stack: list[list[list[str]]] = []
        self._cell: list[str] | None = None
        self._href: str | None = None
        self._atext: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._stack.append([])
        elif tag == "tr" and self._stack:
            self._stack[-1].append([])
        elif tag in ("td", "th") and self._stack:
            if not self._stack[-1]:
                self._stack[-1].append([])
            self._cell = []
            span = dict(attrs).get("colspan") or "1"
            self._span = int(span) if str(span).strip().isdigit() and int(span) > 0 else 1
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._atext = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._stack and self._cell is not None:
            self._stack[-1][-1].append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            if self.expand_colspan and self._span > 1:
                self._stack[-1][-1].extend([""] * (self._span - 1))
            self._cell = None
            self._span = 1
        elif tag == "table" and self._stack:
            self.tables.append(self._stack.pop())
        elif tag == "a" and self._href is not None:
            self.links.append((self._href, re.sub(r"\s+", " ", "".join(self._atext)).strip()))
            self._href = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)
        if self._href is not None:
            self._atext.append(data)


def html_tables(text: str, *, expand_colspan: bool = False) -> list[list[list[str]]]:
    """All <table>s → list of rows → list of cell texts (whitespace-normalised).

    ``expand_colspan=True`` pads a ``colspan=n`` cell with n−1 empty cells so columns align
    (``rowspan`` is not expanded).
    """
    p = _TableParser(expand_colspan=expand_colspan)
    p.feed(text)
    p.close()
    return [[r for r in t if r] for t in p.tables]


def html_links(text: str) -> list[tuple[str, str]]:
    """All <a href> links as (href, text), entity-decoded."""
    p = _TableParser()
    p.feed(text)
    p.close()
    return [(_html.unescape(h), t) for h, t in p.links]


def table_to_frame(rows: list[list[str]], header: bool = True) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if not header:
        return pd.DataFrame(rows, dtype=str)
    return pd.DataFrame(rows[1:], columns=[c or f"col_{i}" for i, c in enumerate(rows[0])], dtype=str)


# ----------------------------------------------------------------------------- tidy-table utilities

def retrieval_date() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def str_id(values: pd.Series | Iterable, width: int | None = None) -> pd.Series:
    """IDs as strings: '7090002.0' → '7090002', optional zero padding, blanks → <NA>."""
    s = pd.Series(values, dtype="object")
    out = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
    out = out.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    out = out.where(out.notna() & (out != "") & (out.str.lower() != "nan"))
    if width:
        out = out.where(out.isna(), out.str.zfill(width))
    return out.astype("string")


def clean_dates(values: pd.Series, *, min_date: str = "1900-01-01", max_date: str | None = None,
                fmt: str | None = None, dayfirst: bool = False) -> tuple[pd.Series, pd.Series]:
    """Parse dates; unparseable or out-of-calendar-range → NaT, with a flag Series.

    Flags: '' (ok), 'missing', 'unparseable', 'out_of_range'. ``max_date`` defaults to tomorrow
    (UTC), so future dates (e.g. CA FHAB years 2555/2626) are rejected.
    """
    raw = pd.Series(values, dtype="object")
    txt = raw.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
    # format="mixed" parses element-wise: pandas 2 otherwise infers ONE format from the first
    # value and silently coerces every differently formatted value to NaT.
    kw = {"format": fmt or "mixed", "dayfirst": dayfirst, "errors": "coerce"}
    has_tz = txt.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any()
    parsed = pd.to_datetime(txt.where(txt != ""), utc=bool(has_tz), **kw)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    hi = pd.Timestamp(max_date) if max_date else pd.Timestamp(_dt.date.today() + _dt.timedelta(days=1))
    lo = pd.Timestamp(min_date)
    flag = pd.Series("", index=raw.index, dtype="object")
    flag[txt == ""] = "missing"
    flag[(txt != "") & parsed.isna()] = "unparseable"
    # years pandas cannot represent (e.g. CA FHAB '2555', Chicago '0023') are calendar errors
    yr = pd.to_numeric(txt.str.extract(r"(?<!\d)(\d{4})(?!\d)", expand=False), errors="coerce")
    flag[(flag == "unparseable") & ((yr < 1678) | (yr > 2261))] = "out_of_range"
    oor = parsed.notna() & ((parsed < lo) | (parsed > hi))
    flag[oor] = "out_of_range"
    return parsed.mask(oor), flag


def clean_latlon(lat: pd.Series | Iterable, lon: pd.Series | Iterable, *,
                 bbox: tuple[float, float, float, float] | None = None
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Numeric lat/lon; missing, (0,0), out-of-globe or outside bbox (W,S,E,N) → NaN + flag.

    Flags: '' (ok), 'missing', 'zero', 'invalid', 'outside_bbox', 'swapped_fixed' (lat/lon
    columns were swapped and are corrected when that puts the point inside ``bbox``).
    """
    index = lat.index if isinstance(lat, pd.Series) else None
    la = pd.Series(pd.to_numeric(pd.Series(list(lat), dtype="object"), errors="coerce").to_numpy(float),
                   index=index)
    lo = pd.Series(pd.to_numeric(pd.Series(list(lon), dtype="object"), errors="coerce").to_numpy(float),
                   index=la.index)
    flag = pd.Series("", index=la.index, dtype="object")
    miss = la.isna() | lo.isna()
    flag[miss] = "missing"
    zero = ~miss & (la == 0) & (lo == 0)
    flag[zero] = "zero"
    present = ~miss & ~zero
    inval = present & ((la.abs() > 90) | (lo.abs() > 180))
    if bbox is None:
        flag[inval] = "invalid"
    else:
        w, s, e, n = bbox
        inside = (lo >= w) & (lo <= e) & (la >= s) & (la <= n)
        # swapped columns (lat ≈ -93, lon ≈ 41) are fixed only when the swap lands inside bbox
        fix = present & ~inside & (la >= w) & (la <= e) & (lo >= s) & (lo <= n)
        la2, lo2 = la.copy(), lo.copy()
        la2[fix], lo2[fix] = lo[fix], la[fix]
        la, lo = la2, lo2
        flag[fix] = "swapped_fixed"
        flag[present & ~fix & inval] = "invalid"
        flag[present & ~fix & ~inval & ~inside] = "outside_bbox"
    bad = flag.isin(["missing", "zero", "invalid", "outside_bbox"])
    return la.mask(bad), lo.mask(bad), flag


def to_number(values: pd.Series | Iterable, *, missing_codes: Iterable[float] = (-999999, -99999, -9999,
                                                                                    999999)
              ) -> tuple[pd.Series, pd.Series]:
    """Numeric values with documented sentinel codes and non-numeric strings → NaN.

    Returns (value, qualifier) where qualifier keeps the original text of anything masked
    (e.g. 'ND', '<0.1', '-9999').
    """
    raw = pd.Series(values, dtype="object")
    txt = raw.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
    num = pd.to_numeric(txt.str.replace(",", "", regex=False), errors="coerce")
    sentinel = num.isin(list(missing_codes))
    qual = txt.where((num.isna() & (txt != "")) | sentinel, "")
    return num.mask(sentinel).astype(float), qual


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path


def coverage_stats(df: pd.DataFrame, *, date_col: str | None = None, site_col: str | None = None,
                   lat_col: str | None = None, by_year: bool = True) -> dict:
    """Row/site/time coverage summary for manifests and datasheets."""
    out: dict[str, Any] = {"n_rows": int(len(df))}
    if site_col and site_col in df:
        out["n_sites"] = int(df[site_col].nunique())
    if lat_col and lat_col in df:
        out["n_with_coords"] = int(df[lat_col].notna().sum())
    if date_col and date_col in df:
        d = pd.to_datetime(df[date_col], errors="coerce")
        if d.notna().any():
            out["date_min"] = str(d.min().date())
            out["date_max"] = str(d.max().date())
            if by_year:
                counts = d.dt.year.value_counts().sort_index()
                out["rows_by_year"] = {int(k): int(v) for k, v in counts.items()}
        out["n_missing_date"] = int(d.isna().sum())
    return out


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
