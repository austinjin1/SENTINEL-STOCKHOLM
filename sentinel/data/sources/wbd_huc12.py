# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""USGS Watershed Boundary Dataset (WBD) HUC12 universe: codes, names, area, states, downstream HUC
(``tohuc``) and an approximate centroid.

Service: ``https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6`` (layer 6 = 12-digit HU,
ArcGIS 11.3, ``maxRecordCount`` 2000, no key; verified 2026-10-24: 103,068 features). Geometry is
requested strongly generalised (``maxAllowableOffset`` 0.02°, 3 decimals, WGS84) only to compute a
representative point; the centroid is therefore approximate (≈ 1–2 km) and is used for distance
features at the ≥ 10-km scale only. Used as the candidate universe for D6 MusselGuard
(docs/preregistration/downstream_musselguard.md).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LAYER = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6"
QUERY = f"{LAYER}/query"
USER_AGENT = "SENTINEL-v2 research (public-data download)"
FIELDS = "huc12,name,areasqkm,states,tohuc,hutype,humod"


def query_params(huc2: str, offset: int, page: int = 2000) -> dict:
    """One page of HUC12 features whose code starts with ``huc2``."""
    if not (len(huc2) == 2 and huc2.isdigit()):
        raise ValueError("huc2 must be a 2-digit string")
    return {"where": f"huc12 LIKE '{huc2}%'", "outFields": FIELDS, "returnGeometry": "true",
            "maxAllowableOffset": "0.02", "geometryPrecision": "3", "outSR": "4326",
            "orderByFields": "huc12", "resultOffset": str(offset), "resultRecordCount": str(page), "f": "json"}


def ring_centroid(rings: list) -> tuple[float, float]:
    """Area-weighted centroid (lon, lat) of the largest outer ring; bbox centre if degenerate."""
    best, best_a = None, -1.0
    for r in rings or []:
        a = np.asarray(r, float)
        if a.ndim != 2 or len(a) < 3:
            continue
        x, y = a[:, 0], a[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        area = 0.5 * cross.sum()
        if abs(area) > best_a:
            best_a, best = abs(area), (a, cross, area)
    if best is None:
        pts = np.concatenate([np.asarray(r, float).reshape(-1, 2) for r in rings]) if rings else np.empty((0, 2))
        if not len(pts):
            return float("nan"), float("nan")
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    a, cross, area = best
    if abs(area) < 1e-12:
        return float((a[:, 0].min() + a[:, 0].max()) / 2), float((a[:, 1].min() + a[:, 1].max()) / 2)
    cx = ((a[:-1, 0] + a[1:, 0]) * cross).sum() / (6 * area)
    cy = ((a[:-1, 1] + a[1:, 1]) * cross).sum() / (6 * area)
    return float(cx), float(cy)


def parse_page(payload: dict) -> pd.DataFrame:
    rows = []
    for f in payload.get("features") or []:
        at = f.get("attributes") or {}
        lon, lat = ring_centroid((f.get("geometry") or {}).get("rings"))
        h = at.get("huc12")
        rows.append({"huc12": None if h is None else str(h).strip(), "name": at.get("name"),
                     "areasqkm": at.get("areasqkm"), "states": at.get("states"),
                     "tohuc": None if at.get("tohuc") in (None, "") else str(at.get("tohuc")).strip(),
                     "hutype": at.get("hutype"), "humod": at.get("humod"), "lon": lon, "lat": lat})
    return pd.DataFrame(rows, columns=["huc12", "name", "areasqkm", "states", "tohuc", "hutype", "humod", "lon", "lat"])


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Validate codes (12 digits, strings), drop duplicates, derive HUC2/4/8, flag odd ``tohuc``."""
    d = df.copy()
    d["huc12"] = d["huc12"].astype("string")
    ok = d["huc12"].str.fullmatch(r"\d{12}").fillna(False)
    d = d[ok].drop_duplicates("huc12").reset_index(drop=True)
    for k in (2, 4, 8):
        d[f"huc{k}"] = d["huc12"].str[:k]
    t = d["tohuc"].astype("string")
    d["tohuc_kind"] = np.where(t.str.fullmatch(r"\d{12}").fillna(False), "huc12",
                               np.where(t.isna(), "missing", "special"))  # OCEAN / CANADA / CLOSED BASIN …
    return d


class WBDClient:
    def __init__(self, cache_dir: Path, *, min_interval: float = 2.0, retries: int = 6, timeout: float = 180):
        self.cache_dir = Path(cache_dir)
        self.min_interval, self.retries, self.timeout = min_interval, retries, timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        self.stats = {"requests": 0, "cache_hits": 0}

    def get(self, params: dict) -> dict:
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
        cp = self.cache_dir / f"{key}.json.gz"
        if cp.exists():
            self.stats["cache_hits"] += 1
            with gzip.open(cp, "rt") as fh:
                return json.load(fh)
        last = None
        for k in range(self.retries):
            self.stats["requests"] += 1
            try:
                r = self.s.get(QUERY, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    body = r.json()
                    if "error" in body:
                        last = str(body["error"])[:300]
                    else:
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        with gzip.open(cp, "wt") as fh:
                            json.dump(body, fh)
                        time.sleep(self.min_interval)
                        return body
                elif r.status_code == 429:
                    time.sleep(float(r.headers.get("Retry-After", 120)))
                    continue
                else:
                    last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = repr(e)
            time.sleep(15 * (k + 1))
        raise RuntimeError(f"WBD query failed: {params.get('where')} offset {params.get('resultOffset')}: {last}")
