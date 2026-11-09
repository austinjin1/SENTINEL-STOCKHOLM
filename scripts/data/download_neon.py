#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download NEON aquatic data products (RELEASE-2026 + flagged PROVISIONAL months).

Layout under ``paths.raw('neon')``::

    <code>/<SITE>/<YYYY-MM>/<file>.csv        observational tables (small; kept)
    <code>/<SITE>/<YYYY-MM>/<file>.parquet    sensor streams (CSV parsed to zstd Parquet, CSV deleted)
    <code>/<SITE>/_site/<file>                sensor_positions etc. (latest per site)
    <code>/_shared/<file>                     variables / categoricalCodes / validation / readme / lab summaries
    <code>/_external/<file>.parquet           taxon tables fetched via downloadFileUrl (16S)
    _meta/products/<code>.json, _meta/sites/*.json, _meta/aop_availability.parquet
    _ledger/<code>.jsonl                      resume-safe ledger (one JSON line per file / site-month)

Credentials: the token is read from ``$NEON_TOKEN`` or ``~/.config/sentinel/keys.env`` and sent
only as the ``X-API-Token`` header to data.neonscience.org (the token session never follows
redirects). File downloads (pre-signed storage URLs) use a separate session WITHOUT the token.
Ledger/manifest URLs are stripped of query strings. Politeness: <= 2 concurrent requests in
total, exponential backoff, HTTP 429 / Retry-After honoured.

Examples::

    python -B scripts/data/download_neon.py --products p1            # Priority 1 products
    python -B scripts/data/download_neon.py --externals              # 16S taxon tables
    python -B scripts/data/download_neon.py --products p2 --aop --site-meta
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import io
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import requests

from sentinel.core import paths
from sentinel.data.sources import neon

pa.set_cpu_count(1)  # light job: 2 request threads, CSV->Parquet conversion single-threaded
pa.set_io_thread_count(2)
ROOT = paths.raw("neon")
LEDGER_DIR = paths.ensure(ROOT / "_ledger")
META_DIR = paths.ensure(ROOT / "_meta")


# --------------------------------------------------------------------------- HTTP
class SignedUrlExpired(RuntimeError):
    pass


class BadRequest(RuntimeError):
    """HTTP 400 from the API (not retried), e.g. 'An expanded package does not exist'."""


class NeonHttp:
    """Polite HTTP: one global semaphore (<= max_concurrent requests in flight), retries."""

    def __init__(self, token: str | None, max_concurrent: int = 2):
        self._token = token
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self._local = threading.local()
        self.n_requests = 0
        self._lock = threading.Lock()

    def _sessions(self) -> tuple[requests.Session, requests.Session]:
        if not hasattr(self._local, "api"):
            api = requests.Session()
            api.headers.update(neon.request_headers(self._token))
            api.max_redirects = 0
            files = requests.Session()
            files.headers.update(neon.request_headers(None))  # never carries the token
            self._local.api, self._local.files = api, files
        return self._local.api, self._local.files

    def _request(self, kind: str, url: str, params: dict | None, timeout: float,
                 retries: int = 8) -> requests.Response:
        api, files = self._sessions()
        sess = api if kind == "api" else files
        last = ""
        for k in range(retries):
            try:
                with self._sem:
                    with self._lock:
                        self.n_requests += 1
                    r = sess.get(url, params=params, timeout=timeout,
                                 allow_redirects=(kind != "api"))
            except requests.RequestException as e:
                last = neon.redact(repr(e), self._token)
                time.sleep(min(120, 5 * 2 ** k) * (0.5 + random.random()))
                continue
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = r.headers.get("Retry-After") or r.headers.get("X-RateLimit-Reset") or "30"
                try:
                    wait_s = max(1.0, float(wait))
                except ValueError:
                    wait_s = 30.0
                time.sleep(min(wait_s, 600) + random.random())
                continue
            if r.status_code in (301, 302, 303, 307, 308) and kind == "api":
                raise RuntimeError(f"unexpected redirect from API: {neon.strip_url(url)}")
            if r.status_code == 404:
                raise FileNotFoundError(neon.strip_url(url))
            if r.status_code in (400, 403) and kind == "file":
                raise SignedUrlExpired(f"HTTP {r.status_code} {neon.strip_url(url)}")
            if r.status_code == 400 and kind == "api":
                raise BadRequest(neon.redact(r.text[:300], self._token))
            if r.status_code in (401, 403) and kind == "api":
                raise PermissionError(f"HTTP {r.status_code} from NEON API (token missing/invalid?) "
                                      f"{neon.strip_url(url)}")
            last = f"HTTP {r.status_code}"
            time.sleep(min(300, 5 * 2 ** k) * (0.5 + random.random()))
        raise RuntimeError(f"GET failed after {retries} tries: {neon.strip_url(url)} :: {last}")

    def api_json(self, url: str, params: dict | None = None, timeout: float = 120) -> dict:
        return self._request("api", url, params, timeout).json()

    def file_bytes(self, url: str, timeout: float = 600) -> bytes:
        return self._request("file", url, None, timeout).content


# --------------------------------------------------------------------------- ledger
class Ledger:
    def __init__(self, code: str):
        self.path = LEDGER_DIR / f"{code}.jsonl"
        self._lock = threading.Lock()

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # torn last line after a kill
        return out

    def done_site_months(self) -> set[tuple[str, str, str]]:
        return {(e["site"], e["month"], e["release"]) for e in self.entries()
                if e.get("type") == "sitemonth" and e.get("status") in ("done", "empty")}

    def write(self, rec: dict) -> None:
        rec = {**rec, "t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        if "url" in rec:
            rec["url"] = neon.strip_url(rec["url"])
        line = json.dumps(rec, default=str)
        with self._lock:
            with self.path.open("a") as fh:
                fh.write(line + "\n")


# --------------------------------------------------------------------------- core
def product_meta(http: NeonHttp, code: str, refresh: bool = False) -> dict:
    p = paths.ensure(META_DIR / "products") / f"{code}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    url, params = neon.product_request(code)
    j = http.api_json(url, params)
    p.write_text(json.dumps(j))
    return j


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _store(spec: neon.ProductSpec, site: str, month: str, f: dict, data: bytes) -> Path:
    base = ROOT / spec.code
    role = f["role"]
    if role in ("shared", "lab"):
        dest = paths.ensure(base / "_shared") / neon.shared_name(f["name"])
        if not dest.exists():
            dest.write_bytes(data)
        return dest
    if role == "site_meta":
        dest = paths.ensure(base / site / "_site") / f["name"]
        dest.write_bytes(data)
        return dest
    d = paths.ensure(base / site / month)
    if spec.kind == "is" and role == "data" and not f["table"].startswith(
            ("ais_", "science_review", "csd_gaugeWater")):
        frame = neon.sensor_csv_to_frame(io.BytesIO(data))
        dest = d / (f["name"][:-4] + ".parquet")
        tmp = dest.with_suffix(".parquet.part")
        frame.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(dest)
        return dest
    dest = d / f["name"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def fetch_site_month(http: NeonHttp, spec: neon.ProductSpec, ledger: Ledger, site: str,
                     month: str, release: str, shared_seen: set, lock: threading.Lock) -> dict:
    url, params = neon.data_request(spec.code, site, month, release, spec.package)
    try:
        try:
            j = http.api_json(url, params)
        except BadRequest as e:
            if spec.package != "expanded" or "expanded package does not exist" not in str(e):
                raise
            # some site-months publish only the basic package
            spec = dataclasses.replace(spec, package="basic")
            url, params = neon.data_request(spec.code, site, month, release, "basic")
            j = http.api_json(url, params)
    except FileNotFoundError:
        ledger.write({"type": "sitemonth", "site": site, "month": month, "release": release,
                      "status": "empty", "n_files": 0})
        return {"files": 0, "bytes": 0}
    data = j.get("data") or {}
    got_release = data.get("release") or release
    files = neon.select_files(data.get("files") or [], spec)
    # one readme per product (documentation for the datasheet)
    rd = neon.readme_file(data.get("files") or [])
    readme_dest = ROOT / spec.code / "_shared" / "readme.txt"
    if rd is not None and not readme_dest.exists():
        with lock:
            if not readme_dest.exists():
                paths.ensure(readme_dest.parent)
                readme_dest.write_bytes(http.file_bytes(rd["url"]))
    n, nbytes = 0, 0
    for f in files:
        if f["role"] == "site_meta" and (ROOT / spec.code / site / "_site" / f["name"]).exists():
            continue
        key = neon.shared_name(f["name"]) if f["role"] in ("shared", "lab") else None
        if key is not None:
            with lock:
                if key in shared_seen:
                    continue
                shared_seen.add(key)
            if (ROOT / spec.code / "_shared" / key).exists():
                continue
        for attempt in range(3):
            try:
                b = http.file_bytes(f["url"])
            except SignedUrlExpired:
                # re-list to get fresh signatures
                j2 = http.api_json(url, params)
                fresh = {x["name"]: x for x in (j2.get("data") or {}).get("files") or []}
                if f["name"] not in fresh:
                    raise
                f = {**f, "url": fresh[f["name"]]["url"]}
                continue
            md5_ok = not f.get("md5") or _md5(b) == f["md5"]
            if not md5_ok:
                if attempt < 2:
                    continue
                if f["role"] not in ("shared", "lab", "site_meta"):
                    raise RuntimeError(f"md5 mismatch: {f['name']}")
                # NEON occasionally lists a stale md5 for metadata files: keep, but record it
            break
        dest = _store(spec, site, month, f, b)
        n += 1
        nbytes += len(b)
        ledger.write({"type": "file", "site": site, "month": month, "release": got_release,
                      "name": f["name"], "role": f["role"], "table": f["table"],
                      "bytes": len(b), "md5": f.get("md5"), "stored": str(dest.relative_to(ROOT)),
                      "stored_bytes": dest.stat().st_size, "url": f["url"],
                      **({} if md5_ok else {"md5_mismatch": _md5(b)})})
    ledger.write({"type": "sitemonth", "site": site, "month": month, "release": release,
                  "release_returned": got_release, "status": "done" if files else "empty",
                  "n_files": n, "bytes": nbytes, "package": spec.package})
    return {"files": n, "bytes": nbytes}


def download_product(http: NeonHttp, short: str, sites: list[str], workers: int,
                     max_months: int | None, refresh_meta: bool) -> dict:
    spec = neon.PRODUCTS[short]
    meta = product_meta(http, spec.code, refresh=refresh_meta)
    sm = neon.site_months(meta, sites)
    ledger = Ledger(spec.code)
    done = ledger.done_site_months()
    todo = [r for r in sm.itertuples(index=False) if (r.site, r.month, r.release) not in done]
    if max_months:
        todo = todo[:max_months]
    print(f"[{short} {spec.code}] site-months available={len(sm)} done={len(done)} "
          f"todo={len(todo)}", flush=True)
    shared_seen: set = set()
    lock = threading.Lock()
    tot = {"files": 0, "bytes": 0, "errors": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_site_month, http, spec, ledger, r.site, r.month, r.release,
                          shared_seen, lock): r for r in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                res = fu.result()
                tot["files"] += res["files"]
                tot["bytes"] += res["bytes"]
            except Exception as e:  # noqa: BLE001 — logged, retried on the next run
                tot["errors"] += 1
                msg = neon.redact(f"{type(e).__name__}: {e}", http._token)
                ledger.write({"type": "error", "site": r.site, "month": r.month,
                              "release": r.release, "error": msg[:500]})
                print(f"  ERROR {short} {r.site} {r.month}: {msg[:200]}", flush=True)
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {short}: {i}/{len(todo)} site-months, {tot['files']} files, "
                      f"{tot['bytes'] / 1e9:.2f} GB, {el / 60:.1f} min, errors={tot['errors']}",
                      flush=True)
    return tot


# --------------------------------------------------------------------------- externals
EXTERNAL_SOURCES = {
    # short: (metadata table regex, url column, name column)
    "microbe_tax": (r"mct_surfaceWaterSampleMetadata_16S", "downloadFileUrl", "downloadFileName"),
    "microbe_comp_v1": (r"mcc_swSeqVariantMetadata_16S", "downloadFileUrl", "downloadFileName"),
}


def external_urls(short: str) -> pd.DataFrame:
    spec = neon.PRODUCTS[short]
    pat, ucol, ncol = EXTERNAL_SOURCES[short]
    rows = []
    for f in sorted((ROOT / spec.code).glob(f"*/*/*.csv")):
        info = neon.parse_file_name(f.name)
        if info.get("role") != "data" or not re.fullmatch(pat, info["table"] or ""):
            continue
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        for u, n in zip(d.get(ucol, []), d.get(ncol, [])):
            if u:
                rows.append({"url": u, "name": n or u.rsplit("/", 1)[-1]})
    # reference sequence tables listed in lab summaries
    for f in (ROOT / spec.code / "_shared").glob("*taxonTableLabSummary*.csv"):
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        if "sequenceReferenceFileUrl" in d.columns:
            for u, g in zip(d["sequenceReferenceFileUrl"], d.get("targetGene", [""] * len(d))):
                if u and "16S" in g:
                    rows.append({"url": u, "name": u.rsplit("/", 1)[-1]})
    return pd.DataFrame(rows, columns=["url", "name"]).drop_duplicates("url")


def download_externals(http: NeonHttp, short: str, workers: int) -> dict:
    spec = neon.PRODUCTS[short]
    ledger = Ledger(spec.code)
    done = {e["url"] for e in ledger.entries() if e.get("type") == "external"}
    urls = external_urls(short)
    todo = urls[~urls["url"].map(neon.strip_url).isin(done)]
    outdir = paths.ensure(ROOT / spec.code / "_external")
    print(f"[{short} externals] unique={len(urls)} done={len(urls) - len(todo)} "
          f"todo={len(todo)}", flush=True)
    tot = {"files": 0, "bytes": 0, "errors": 0}

    def one(u: str, name: str) -> int:
        b = http.file_bytes(u)  # public storage URL; token-free session
        df = neon.read_csv_bytes(b)
        dest = outdir / (Path(name).stem + ".parquet")
        tmp = dest.with_suffix(".parquet.part")
        df.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(dest)
        ledger.write({"type": "external", "name": name, "bytes": len(b), "md5": _md5(b),
                      "rows": int(len(df)), "stored": str(dest.relative_to(ROOT)),
                      "stored_bytes": dest.stat().st_size, "url": u})
        return len(b)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, r.url, r.name): r for r in todo.itertuples(index=False)}
        for i, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                tot["bytes"] += fu.result()
                tot["files"] += 1
            except Exception as e:  # noqa: BLE001
                tot["errors"] += 1
                msg = neon.redact(f"{type(e).__name__}: {e}", http._token)
                ledger.write({"type": "external_error", "name": r.name, "url": r.url,
                              "error": msg[:500]})
                print(f"  ERROR external {r.name}: {msg[:200]}", flush=True)
            if i % 50 == 0 or i == len(todo):
                print(f"  {short} externals: {i}/{len(todo)} {tot['bytes'] / 1e9:.2f} GB "
                      f"errors={tot['errors']}", flush=True)
    return tot


# --------------------------------------------------------------------------- site + AOP meta
def download_site_meta(http: NeonHttp, sites: list[str]) -> None:
    out = paths.ensure(META_DIR / "sites")
    for s in sites:
        for kind, (url, params) in (("site", neon.site_request(s)),
                                    ("location", neon.location_request(s))):
            p = out / f"{s}_{kind}.json"
            if not p.exists():
                p.write_text(json.dumps(http.api_json(url, params)))
        # named locations of sensor stations / sampling points (children of the site)
        loc = json.loads((out / f"{s}_location.json").read_text())["data"]
        kids = [k for k in loc.get("locationChildren") or []
                if k.startswith(s + ".AOS.") or k.startswith(("CFGLOC", "STREAM", "BUOY"))]
        cp = out / f"{s}_children.json"
        have = json.loads(cp.read_text()) if cp.exists() else {}
        for k in kids:
            if k not in have:
                url, params = neon.location_request(k)
                try:
                    have[k] = http.api_json(url, params).get("data")
                except FileNotFoundError:
                    have[k] = None
        cp.write_text(json.dumps(have))
    print(f"site metadata: {len(sites)} sites -> {out}", flush=True)


def download_aop_availability(http: NeonHttp, sites: list[str]) -> Path:
    rows = []
    for code, name in neon.AOP_PRODUCTS.items():
        j = product_meta(http, code)
        sm = neon.site_months(j, sites)
        sm["product"] = code
        sm["product_name"] = name
        rows.append(sm)
    df = pd.concat(rows, ignore_index=True)
    p = META_DIR / "aop_availability.parquet"
    df.to_parquet(p, compression="zstd", index=False)
    print(f"AOP availability: {len(df)} site-months over {df['site'].nunique()} aquatic sites",
          flush=True)
    return p


# --------------------------------------------------------------------------- main
def _expand(products: list[str]) -> list[str]:
    out = []
    for p in products:
        if p in ("p1", "p2"):
            pr = int(p[1])
            out += [s for s in neon.PRIORITY_ORDER if neon.PRODUCTS[s].priority == pr]
        elif p == "all":
            out += list(neon.PRIORITY_ORDER)
        elif p in neon.PRODUCTS:
            out.append(p)
        else:
            raise SystemExit(f"unknown product {p!r}; choose from {list(neon.PRODUCTS)}")
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=[], help="short names, p1, p2 or all")
    ap.add_argument("--sites", nargs="*", default=list(neon.AQUATIC_SITES))
    ap.add_argument("--workers", type=int, default=2, help="<= 2 (politeness)")
    ap.add_argument("--max-months", type=int, default=None, help="testing: cap site-months")
    ap.add_argument("--externals", action="store_true", help="16S taxon tables via downloadFileUrl")
    ap.add_argument("--site-meta", action="store_true")
    ap.add_argument("--aop", action="store_true", help="record AOP availability only")
    ap.add_argument("--refresh-meta", action="store_true")
    a = ap.parse_args(argv)
    workers = max(1, min(2, a.workers))
    token = neon.read_token()
    if not token:
        print("NEON_TOKEN not found in environment or ~/.config/sentinel/keys.env", file=sys.stderr)
        return 2
    http = NeonHttp(token, max_concurrent=workers)
    summary = {}
    if a.site_meta:
        download_site_meta(http, a.sites)
    if a.aop:
        download_aop_availability(http, a.sites)
    for short in _expand(a.products):
        summary[short] = download_product(http, short, a.sites, workers, a.max_months,
                                          a.refresh_meta)
    if a.externals:
        for short in EXTERNAL_SOURCES:
            summary[f"{short}_externals"] = download_externals(http, short, workers)
    print(json.dumps({"summary": summary, "requests": http.n_requests}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
