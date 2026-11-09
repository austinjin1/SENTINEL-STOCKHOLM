# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EPA StreamCat (NHDPlus V2 catchment/watershed attributes) via the keyless StreamCat API.

Download location (checked 2026-10-22): the historical bulk HydroRegion/State CSV zips
(``gaftp.epa.gov/epadatacommons/ORD/NHDPlusLandscapeAttributes/StreamCat/HydroRegions``, formerly
``newftp.epa.gov``) return 404, and the EPA ORD data-commons S3 bucket denies access (403). The
EPA StreamCat page now routes bulk access through the web tool and the REST API
``https://api.epa.gov/StreamCat/streams`` (Swagger: usepa.github.io/StreamCatWebServices_Public),
which needs no key (header ``x-ratelimit-limit: 1000`` per hour). ``POST /metrics`` accepts long
COMID lists; ``/variable_info`` gives units/descriptions and ``/changelog`` the DB version.

Metric names are requested without the AOI suffix (``pctcrop2019``) plus ``aoi=cat,ws``; columns
come back suffixed (``pctcrop2019cat``, ``pctcrop2019ws``). Excluded on purpose: model-predicted
stream temperature (MAST/MSST/MWST -- overlaps SENTINEL targets) and predicted biological
condition (``prg_bmmi0809``, Hill et al. 2017 -- a label for downstream tasks; docs/research/07).
"""

from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.epa.gov/StreamCat/streams"
METRICS_URL = f"{API}/metrics"
VARINFO_URL = f"{API}/variable_info"
CHANGELOG_URL = f"{API}/changelog"
USER_AGENT = "SENTINEL-v2 research (public-data download)"

# (group, aoi, metric names) -- ~180 columns after AOI expansion. Years: latest available
# (NLCD/impervious 2019, census 2010, NNI 2017, point-source loads 2012).
CURATED: list[tuple[str, str, list[str]]] = [
    (
        "landcover",
        "cat,ws",
        [
            "pctow2019",
            "pcturbop2019",
            "pcturblo2019",
            "pcturbmd2019",
            "pcturbhi2019",
            "pctbl2019",
            "pctdecid2019",
            "pctconif2019",
            "pctmxfst2019",
            "pctshrb2019",
            "pctgrs2019",
            "pcthay2019",
            "pctcrop2019",
            "pctwdwet2019",
            "pcthbwet2019",
            "pctimp2019",
        ],
    ),
    (
        "riparian100m",
        "catrp100,wsrp100",
        ["pctimp2019", "pctcrop2019", "pctdecid2019", "pctconif2019", "pctmxfst2019"],
    ),
    (
        "agri_nutrients",
        "cat,ws",
        ["pctagdrainage", "fert", "manure", "cbnf", "nani", "nsurp", "pestic1997", "inorgnwetdep2008"],
    ),
    (
        "nutrient_inventory",
        "cat,ws",
        [
            "n_tin_2017",
            "p_tin_2017",
            "n_ff_2017",
            "p_ff_2017",
            "n_lw_2017",
            "p_lw_2017",
            "n_hw_2017",
            "p_hw_2017",
            "n_usgsww_2012",
            "p_usgsww_2012",
        ],
    ),
    (
        "infrastructure",
        "cat,ws",
        [
            "popden2010",
            "huden2010",
            "rddens",
            "rdcrs",
            "damdens",
            "damnidstor",
            "damnrmstor",
            "npdesdens",
            "superfunddens",
            "tridens",
            "wwtpalldens",
            "wwtpmajordens",
            "septic",
            "minedens",
            "coalminedens",
            "canaldens",
        ],
    ),
    (
        "lithology",
        "ws",
        [
            "pctcarbresid",
            "pctnoncarbresid",
            "pctalkintruvol",
            "pctsilicic",
            "pctextruvol",
            "pctcolluvsed",
            "pctglactilclay",
            "pctglactilloam",
            "pctglactilcrs",
            "pctglaclakecrs",
            "pctglaclakefine",
            "pcthydric",
            "pcteolcrs",
            "pcteolfine",
            "pctsallake",
            "pctalluvcoast",
            "pctcoastcrs",
            "pctwater",
        ],
    ),
    (
        "geochem",
        "ws",
        [
            "al2o3",
            "cao",
            "fe2o3",
            "k2o",
            "mgo",
            "na2o",
            "p2o5",
            "s",
            "sio2",
            "n",
            "compstrgth",
            "hydrlcond",
            "rockn",
        ],
    ),
    ("soils", "cat,ws", ["clay", "sand", "om", "perm", "rckdep", "wtdep", "kffact"]),
    (
        "climate_hydro_topo",
        "cat,ws",
        [
            "precip9120",
            "tmax9120",
            "tmean9120",
            "tmin9120",
            "bfi",
            "runoff",
            "precip_minus_evt",
            "wetindex",
            "elev",
        ],
    ),
    (
        "indices_channel",
        "other",
        ["iwi", "ici", "bankfulldepth", "bankfullwidth", "wettedwidth", "thalwegdepth"],
    ),
]
EXCLUDED = {"mast", "msst", "mwst", "prg_bmmi0809"}


_NNI = re.compile(r"^[np]_[a-z]+_\d{4}$")  # National Nutrient Inventory / USGS point-source loads


def wants_flags(names: list[str], aoi: str) -> bool:
    """Whether ``showareasqkm``/``showpctfull`` can be sent. The API fails with these flags for riparian
    (``*rp100``: HTTP 403 "a function referenced by the SQL statement ... does not exist"), ``other``
    and National-Nutrient-Inventory metrics (HTTP 555 ORA-00923), observed 2026-10-22."""
    return "rp100" not in aoi and aoi != "other" and not any(_NNI.match(n) for n in names)


def metrics_request(names: list[str], comids: list[str], aoi: str) -> tuple[str, dict]:
    """POST form for ``/metrics`` (flags added only where the API supports them, see :func:`wants_flags`)."""
    bad = [n for n in names if any(n.startswith(e) for e in EXCLUDED)]
    if bad:
        raise ValueError(f"excluded metrics requested: {bad}")
    data = {"name": ",".join(names), "aoi": aoi, "comid": ",".join(str(c) for c in comids)}
    if wants_flags(names, aoi):
        data.update({"showareasqkm": "true", "showpctfull": "true"})
    return METRICS_URL, data


def parse_metrics(payload: dict) -> pd.DataFrame:
    items = (payload or {}).get("items") or []
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=["comid"])
    df["comid"] = df["comid"].astype("int64").astype(str)
    return df


def expected_columns(names: list[str], aoi: str) -> list[str]:
    if aoi == "other":
        return list(names)
    return [f"{n}{a}" for n in names for a in aoi.split(",")]


def metric_stem(col: str) -> tuple[str, str]:
    """'pctcrop2019wsrp100' -> ('pctcrop2019', 'wsrp100')."""
    for suf in ("catrp100", "wsrp100", "cat", "ws"):
        if col.endswith(suf):
            return col[: -len(suf)], suf
    return col, "other"


def column_dictionary(columns: list[str], varinfo: pd.DataFrame) -> pd.DataFrame:
    """Join requested columns to StreamCat ``/variable_info`` (units, description, dataset)."""
    vi = varinfo.copy()
    vi["has_year"] = vi["METRIC_NAME"].str.contains("[Year]", regex=False)
    vi["prefix"] = (
        vi["METRIC_NAME"]
        .str.replace("[AOI]", "", regex=False)
        .str.replace("[Year]", "", regex=False)
        .str.lower()
    )
    no_year = vi[~vi["has_year"]].drop_duplicates("prefix").set_index("prefix")
    with_year = vi[vi["has_year"]].drop_duplicates("prefix").set_index("prefix")
    rows = []
    for c in columns:
        stem, aoi = metric_stem(c)
        base = stem.rstrip("0123456789")
        year = None
        if stem in no_year.index:
            r = no_year.loc[stem]
        elif base and base != stem and base in with_year.index:
            r, year = with_year.loc[base], stem[len(base) :]
        elif stem in with_year.index:
            r = with_year.loc[stem]
        else:
            r = None
        rows.append(
            {
                "column": c,
                "metric": stem,
                "aoi": aoi,
                "year": year,
                "description": None if r is None else r["METRIC_DESCRIPTION"],
                "units": None if r is None else r["METRIC_UNITS"],
                "category": None if r is None else r["INDICATOR_CATEGORY"],
                "dataset": None if r is None else r["FINAL_TABLE"],
                "source_name": None if r is None else r["SOURCE_NAME"],
                "source_url": None if r is None else r["SOURCE_URL"],
            }
        )
    return pd.DataFrame(rows)


class StreamCatClient:
    """Sequential POST client with gzip-JSON cache and 429/5xx handling (API limit 1,000/h)."""

    def __init__(self, cache_dir: Path, *, timeout: float = 600, retries: int = 6, min_interval: float = 4.0):
        self.cache_dir = Path(cache_dir)
        self.timeout, self.retries, self.min_interval = timeout, retries, min_interval
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        self.stats = {"requests": 0, "cache_hits": 0, "errors": 0, "ratelimit_remaining": None}

    def _cp(self, url: str, data: dict) -> Path:
        key = hashlib.sha1((url + json.dumps(data, sort_keys=True)).encode()).hexdigest()
        return self.cache_dir / f"{key}.json.gz"

    def post(self, url: str, data: dict) -> dict:
        cp = self._cp(url, data)
        if cp.exists():
            self.stats["cache_hits"] += 1
            with gzip.open(cp, "rt") as fh:
                return json.load(fh)
        last = None
        for k in range(self.retries):
            self.stats["requests"] += 1
            try:
                r = self.s.post(url, data=data, timeout=min(self.timeout, 300))
            except requests.RequestException as e:
                last = repr(e)
                time.sleep(20 * (k + 1))
                continue
            self.stats["ratelimit_remaining"] = r.headers.get("x-ratelimit-remaining")
            if r.status_code == 200:
                rec = {
                    "body": r.json(),
                    "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    "url": url,
                    "params": {
                        k2: (v if k2 != "comid" else f"{len(v.split(','))} comids") for k2, v in data.items()
                    },
                }
                cp.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(cp, "wt") as fh:
                    json.dump(rec, fh)
                time.sleep(self.min_interval)
                return rec
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 900)))
                continue
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in (400, 403, 555) and ("ORA-" in r.text or "SQL" in r.text):
                break  # deterministic SQL/parameter error: do not hammer the API
            time.sleep(30 * (k + 1))
        self.stats["errors"] += 1
        raise RuntimeError(f"StreamCat POST failed: {url} :: {last}")

    def get_text(self, url: str, retries: int = 6) -> str:
        """GET with retries (the API intermittently answers 502/503 'upstream connect error')."""
        last = None
        for k in range(retries):
            try:
                r = self.s.get(url, timeout=min(self.timeout, 180))
                if r.status_code == 200:
                    return r.text
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.RequestException as e:
                last = repr(e)
            time.sleep(15 * (k + 1))
        raise RuntimeError(f"StreamCat GET failed: {url} :: {last}")
