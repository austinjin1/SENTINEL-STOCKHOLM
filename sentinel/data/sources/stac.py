# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""STAC search + windowed COG chip reads for Sentinel-2 L2A, Landsat C2 L2, NAIP and JRC GSW.

Sources (all anonymous; verified live 2026-10-22, see docs/datasheets/s2_chips.md):
  * Sentinel-2 L2A primary: Element 84 Earth Search ``sentinel-2-c1-l2a`` (ESA Collection-1
    processing; ``datetime`` = granule SENSING_TIME; ``raster:bands`` carry scale 1e-4 and the BOA
    offset -0.1). Gaps found at test points: 2015-06..2018-01 and 2022-01..2022-11.
  * Sentinel-2 L2A fallback: Planetary Computer ``sentinel-2-l2a`` (mixed processing baselines
    02.12..05.11; no scale/offset metadata; ``datetime`` = datatake start, NOT the granule sensing
    time, so the exact time is read from the granule metadata XML; 2022-23 contain duplicate
    original + reprocessed items for the same acquisition).
  * Landsat C2 L2, NAIP and JRC Global Surface Water: Planetary Computer (SAS-signed hrefs).

Radiometry rules (the v1 defect, docs/research/10 §0):
  * S2 L2A digital numbers for processing baseline >= 04.00 carry BOA_ADD_OFFSET = -1000
    (reflectance = (DN - 1000) / 10000). Items whose offset was already removed upstream
    (Earth Search legacy ``earthsearch:boa_offset_applied: true``) must NOT be corrected again even
    though their ``raster:bands`` still advertise offset -0.1. ``s2_offset_decision`` encodes this
    and is unit-tested for both cases.
  * Chips store *harmonised* int16 DN: ``value = raw_DN + offset_dn`` (offset_dn in {0, -1000}),
    so reflectance = value * 1e-4 for every item regardless of baseline; nodata (raw DN 0) is -32768.

Footprints (the second v1 defect): every S2 band is resampled onto the same 10 m chip grid by
nearest-neighbour pixel replication on the nested MGRS grids (a 20 m pixel becomes exactly 2x2
10 m pixels, a 60 m pixel 6x6), so all layers cover the identical 640 m x 640 m footprint and the
original radiometric values are preserved (no interpolation across shorelines). The effective
resolution of the 20/60 m bands is unchanged by this.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# GDAL settings for anonymous HTTPS COG reads (set before rasterio/GDAL initialise; user env wins).
for _k, _v in {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",
    "CPL_VSIL_CURL_CACHE_SIZE": "268435456",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "67108864",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_LOW_SPEED_TIME": "60",     # abort transfers stalled below 1 B/s for 60 s
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1",
    "GDAL_HTTP_USERAGENT": "SENTINEL-v2 research (public-data download)",
    "AWS_NO_SIGN_REQUEST": "YES",
}.items():
    os.environ.setdefault(_k, _v)

USER_AGENT = "SENTINEL-v2 research (public-data download)"
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
ES = "earth-search"
MPC = "planetary-computer"
S2_ES_C1 = "sentinel-2-c1-l2a"
S2_ES_LEGACY = "sentinel-2-l2a"
S2_MPC = "sentinel-2-l2a"
LANDSAT_MPC = "landsat-c2-l2"
NAIP_MPC = "naip"
JRC_MPC = "jrc-gsw"

NODATA_I16 = -32768
JRC_NODATA = 255

# ------------------------------------------------------------------------------------ band tables


@dataclass(frozen=True)
class Layer:
    name: str            # canonical layer name stored in chips
    es_key: str          # Earth Search C1 asset key
    mpc_key: str         # Planetary Computer asset key
    res_m: int           # native resolution (S2 grid), metres
    center_nm: float     # nominal centre wavelength (S2A, MPC eo:bands), nm; NaN for non-spectral
    fwhm_nm: float
    kind: str            # 'reflectance' | 'aot' | 'scl'


S2_LAYERS: tuple[Layer, ...] = (
    Layer("B01", "coastal", "B01", 60, 443, 27, "reflectance"),
    Layer("B02", "blue", "B02", 10, 490, 98, "reflectance"),
    Layer("B03", "green", "B03", 10, 560, 45, "reflectance"),
    Layer("B04", "red", "B04", 10, 665, 38, "reflectance"),
    Layer("B05", "rededge1", "B05", 20, 704, 19, "reflectance"),
    Layer("B06", "rededge2", "B06", 20, 740, 18, "reflectance"),
    Layer("B07", "rededge3", "B07", 20, 783, 28, "reflectance"),
    Layer("B08", "nir", "B08", 10, 842, 145, "reflectance"),
    Layer("B8A", "nir08", "B8A", 20, 865, 33, "reflectance"),
    Layer("B09", "nir09", "B09", 60, 945, 26, "reflectance"),
    Layer("B11", "swir16", "B11", 20, 1610, 143, "reflectance"),
    Layer("B12", "swir22", "B12", 20, 2190, 242, "reflectance"),
    # AOT: ES C1 20 m, MPC 10 m; DN * 0.001 = aerosol optical thickness (550 nm); no BOA offset.
    Layer("AOT", "aot", "AOT", 20, float("nan"), float("nan"), "aot"),
    Layer("SCL", "scl", "SCL", 20, float("nan"), float("nan"), "scl"),
)
S2_LAYER_NAMES = tuple(l.name for l in S2_LAYERS)
S2_REFLECTANCE_LAYERS = tuple(l.name for l in S2_LAYERS if l.kind == "reflectance")
S2_SCALE = 1e-4
AOT_SCALE = 1e-3

SCL_CLASSES = {
    0: "no_data", 1: "saturated_defective", 2: "dark_or_topographic_shadow", 3: "cloud_shadow",
    4: "vegetation", 5: "not_vegetated", 6: "water", 7: "unclassified", 8: "cloud_medium",
    9: "cloud_high", 10: "thin_cirrus", 11: "snow_ice",
}
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)   # never usable for water-leaving signal

LANDSAT_LAYERS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "lwir11",
                  "qa_pixel", "qa_aerosol")
QA_PIXEL_BITS = {"fill": 0, "dilated_cloud": 1, "cirrus": 2, "cloud": 3, "cloud_shadow": 4,
                 "snow": 5, "clear": 6, "water": 7}
NAIP_BANDS = ("red", "green", "blue", "nir")

# ------------------------------------------------------------------------- small pure helpers


def parse_baseline(value: Any) -> float | None:
    """'05.09' -> 5.09; None/'' -> None. Values < 1 (e.g. Earth Search legacy '00.01', which marks
    non-ESA processing) are returned as-is so callers can treat them as unknown."""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def platform_code(platform: str | None) -> str:
    """'sentinel-2a' / 'Sentinel-2A' -> 'S2A'; 'landsat-8' -> 'L8'."""
    p = (platform or "").lower().replace("_", "-")
    m = re.search(r"sentinel-?2([abcd])", p)
    if m:
        return "S2" + m.group(1).upper()
    m = re.search(r"landsat-?(\d)", p)
    if m:
        return "L" + m.group(1)
    return platform or ""


def parse_time(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# ------------------------------------------------------------------------------- offset logic


@dataclass(frozen=True)
class OffsetDecision:
    offset_dn: int            # add to raw DN to obtain harmonised DN (0 or -1000)
    already_applied: bool     # upstream already removed the offset (do not correct again)
    source: str               # which evidence decided it
    baseline: str | None
    consistent: bool          # all available evidence agrees
    note: str = ""


def s2_offset_decision(props: dict, raster_bands: dict | None = None,
                       xml_offset_dn: int | None = None) -> OffsetDecision:
    """Decide the DN offset for one S2 L2A item.

    ``props``: STAC item properties. ``raster_bands``: the ``raster:bands[0]`` dict of a reflectance
    asset (may carry ``scale``/``offset``). ``xml_offset_dn``: BOA_ADD_OFFSET parsed from the
    product metadata (MTD_MSIL2A.xml) when available.

    Rules, in order:
      1. ``earthsearch:boa_offset_applied`` true -> offset 0 (already corrected; raster:bands lie).
      2. Expected offset from the processing baseline: -1000 if baseline >= 04.00 else 0.
         Baselines < 1 (Earth Search legacy '00.01') or missing are *unknown*.
      3. If raster:bands has an offset (Earth Search), offset_dn = round(offset / scale).
      4. Else if the product XML offset is known, use it.
      5. Else use the baseline rule (Planetary Computer: no offset metadata on assets).
    ``consistent`` is False when the pieces of evidence disagree (callers skip such items).
    """
    bl_str = props.get("s2:processing_baseline")
    bl = parse_baseline(bl_str)
    bl_known = bl is not None and bl >= 1.0
    expected = (-1000 if bl >= 4.0 else 0) if bl_known else None
    applied = props.get("earthsearch:boa_offset_applied")
    if applied is True or str(applied).lower() == "true":
        note = "offset already removed upstream (earthsearch:boa_offset_applied=true)"
        ok = xml_offset_dn is None or xml_offset_dn in (0, -1000)
        return OffsetDecision(0, True, "earthsearch:boa_offset_applied", bl_str, ok, note)
    meta_dn = None
    if raster_bands and raster_bands.get("offset") is not None:
        scale = raster_bands.get("scale") or S2_SCALE
        meta_dn = int(round(float(raster_bands["offset"]) / float(scale)))
    evidence = {k: v for k, v in (("raster:bands", meta_dn), ("product_xml", xml_offset_dn),
                                  ("baseline_rule", expected)) if v is not None}
    if not evidence:
        return OffsetDecision(0, False, "none", bl_str, False, "unknown processing baseline")
    if not bl_known and "product_xml" not in evidence:
        # e.g. Earth Search legacy baseline '00.01' (non-ESA processing): raster:bands alone is not
        # trusted, because the same metadata is wrong for offset-corrected legacy items.
        return OffsetDecision(int(next(iter(evidence.values()))), False, next(iter(evidence)), bl_str,
                              False, f"baseline {bl_str!r} unknown; product XML needed")
    for src in ("raster:bands", "product_xml", "baseline_rule"):
        if src in evidence:
            chosen, source = evidence[src], src
            break
    consistent = len(set(evidence.values())) == 1 and chosen in (0, -1000)
    note = "" if consistent else f"evidence disagrees: {evidence}"
    return OffsetDecision(int(chosen), False, source, bl_str, consistent, note)


def harmonise_dn(raw: np.ndarray, offset_dn: int) -> np.ndarray:
    """uint16 raw DN -> int16 harmonised DN (raw + offset_dn); raw 0 (nodata) -> -32768.
    Values that would exceed int16 (saturated, reflectance > 3.27) are clipped to 32767."""
    raw = np.asarray(raw)
    out = raw.astype(np.int32) + int(offset_dn)
    out = np.clip(out, -32767, 32767)
    out[raw == 0] = NODATA_I16
    return out.astype(np.int16)


def to_reflectance(harmonised: np.ndarray) -> np.ndarray:
    """Harmonised int16 DN -> float32 reflectance with NaN for nodata."""
    a = np.asarray(harmonised).astype(np.float32) * S2_SCALE
    a[np.asarray(harmonised) == NODATA_I16] = np.nan
    return a


def parse_boa_offsets(product_xml: bytes | str) -> dict[int, int]:
    """BOA_ADD_OFFSET per band_id from MTD_MSIL2A.xml (empty dict for baselines < 04.00)."""
    root = ET.fromstring(product_xml)
    out = {}
    for el in root.iter():
        if el.tag.split("}")[-1] == "BOA_ADD_OFFSET":
            out[int(el.attrib.get("band_id", -1))] = int(float(el.text))
    return out


def xml_offset_from_bands(offsets: dict[int, int]) -> int | None:
    """Single DN offset from per-band BOA_ADD_OFFSET values: {} (baseline < 04.00, no list) -> 0;
    all equal -> that value; differing per-band offsets -> None (not representable; flagged)."""
    if not offsets:
        return 0
    vals = set(offsets.values())
    return next(iter(vals)) if len(vals) == 1 else None


def parse_granule_xml(granule_xml: bytes | str) -> dict[str, Any]:
    """SENSING_TIME and mean sun / per-band mean viewing angles from MTD_TL.xml."""
    root = ET.fromstring(granule_xml)
    res: dict[str, Any] = {}
    view_z, view_a = [], []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "SENSING_TIME" and "sensing_time" not in res:
            res["sensing_time"] = el.text.strip()
        elif tag == "Mean_Sun_Angle":
            for ch in el:
                t = ch.tag.split("}")[-1]
                if t == "ZENITH_ANGLE":
                    res["sun_zenith"] = float(ch.text)
                elif t == "AZIMUTH_ANGLE":
                    res["sun_azimuth"] = float(ch.text)
        elif tag == "Mean_Viewing_Incidence_Angle":
            for ch in el:
                t = ch.tag.split("}")[-1]
                if t == "ZENITH_ANGLE":
                    view_z.append(float(ch.text))
                elif t == "AZIMUTH_ANGLE":
                    view_a.append(float(ch.text))
    if view_z:
        res["view_zenith"] = float(np.mean(view_z))
    if view_a:  # circular mean
        r = np.deg2rad(view_a)
        res["view_azimuth"] = float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)
    return res


# --------------------------------------------------------------------------- SCL / QA parsing


def centre_slice(size: int, k: int, centre: int | None = None) -> slice:
    """Slice of length k centred on pixel ``centre`` (default size // 2, the point's pixel)."""
    c = size // 2 if centre is None else centre
    return slice(c - k // 2, c - k // 2 + k)


def scl_fractions(scl: np.ndarray) -> dict[str, float]:
    """Fraction of pixels in each SCL class (keys ``SCL_CLASSES`` names; sums to 1)."""
    a = np.asarray(scl).astype(np.int64).ravel()
    n = a.size
    counts = np.bincount(np.clip(a, 0, 12), minlength=13)
    out = {name: float(counts[c]) / n if n else float("nan") for c, name in SCL_CLASSES.items()}
    out["other"] = float(counts[12]) / n if n else float("nan")
    return out


def scl_summary(scl: np.ndarray, prefix: str) -> dict[str, float]:
    """Class fractions plus water / valid counts for one SCL window, keys prefixed."""
    fr = scl_fractions(scl)
    a = np.asarray(scl)
    out = {f"{prefix}_{k}": v for k, v in fr.items()}
    out[f"{prefix}_n"] = int(a.size)
    out[f"{prefix}_n_water"] = int((a == 6).sum())
    out[f"{prefix}_n_valid"] = int((~np.isin(a, SCL_INVALID)).sum())
    return out


def qa_pixel_flags(qa: np.ndarray) -> dict[str, np.ndarray]:
    """Landsat C2 QA_PIXEL bit flags -> boolean arrays."""
    q = np.asarray(qa).astype(np.uint16)
    return {name: ((q >> bit) & 1).astype(bool) for name, bit in QA_PIXEL_BITS.items()}


def qa_pixel_fractions(qa: np.ndarray, prefix: str) -> dict[str, float]:
    f = qa_pixel_flags(qa)
    n = np.asarray(qa).size
    return {f"{prefix}_{k}": float(v.sum()) / n for k, v in f.items()}


# ------------------------------------------------------------------------ chip grid geometry


@dataclass(frozen=True)
class ChipGrid:
    """A size x size chip on a projected grid of resolution ``res`` with origin (x0, y0)."""
    crs: str
    x0: float          # grid origin (upper-left corner of pixel 0,0), CRS units
    y0: float
    res: float
    row0: int          # first chip row/col in the grid
    col0: int
    size: int
    point_row: int     # point's pixel inside the chip (== size // 2)
    point_col: int
    point_x: float
    point_y: float

    @property
    def transform(self):
        from affine import Affine
        return Affine(self.res, 0.0, self.x0 + self.col0 * self.res, 0.0, -self.res,
                      self.y0 - self.row0 * self.res)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.x0 + self.col0 * self.res
        top = self.y0 - self.row0 * self.res
        return left, top - self.size * self.res, left + self.size * self.res, top

    @property
    def subpixel(self) -> tuple[float, float]:
        """Point position inside its pixel (0..1 from the pixel's upper-left corner)."""
        fx = (self.point_x - self.x0) / self.res
        fy = (self.y0 - self.point_y) / self.res
        return fy - math.floor(fy), fx - math.floor(fx)


def chip_grid(x: float, y: float, x0: float, y0: float, res: float, size: int, crs: str) -> ChipGrid:
    """Chip whose pixel (size//2, size//2) contains the projected point (x, y)."""
    col = math.floor((x - x0) / res)
    row = math.floor((y0 - y) / res)
    half = size // 2
    return ChipGrid(crs, x0, y0, res, row - half, col - half, size, half, half, x, y)


def replicate_window(read: Callable[[int, int, int, int], np.ndarray], grid: ChipGrid,
                     factor: int) -> np.ndarray:
    """Read a band whose native pixels are ``factor`` x the chip resolution on a grid sharing the
    chip grid's origin, and return it on the chip grid by nearest-neighbour replication.

    ``read(row_off, col_off, height, width)`` must return the native-grid window (boundless, fill 0).
    """
    if factor == 1:
        return read(grid.row0, grid.col0, grid.size, grid.size)
    r0 = grid.row0 // factor
    c0 = grid.col0 // factor
    r1 = (grid.row0 + grid.size - 1) // factor
    c1 = (grid.col0 + grid.size - 1) // factor
    native = read(r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    up = np.repeat(np.repeat(native, factor, axis=0), factor, axis=1)
    ro = grid.row0 - r0 * factor
    co = grid.col0 - c0 * factor
    return up[ro:ro + grid.size, co:co + grid.size]


# --------------------------------------------------------------------------- STAC clients


_tls = threading.local()
_host_sem = {ES: threading.BoundedSemaphore(4), MPC: threading.BoundedSemaphore(4)}


def client(source: str):
    """Thread-local pystac_client.Client (MPC hrefs are signed lazily, not by the client)."""
    import pystac_client
    key = f"client_{source}"
    c = getattr(_tls, key, None)
    if c is None:
        url = EARTH_SEARCH_URL if source == ES else PLANETARY_COMPUTER_URL
        c = pystac_client.Client.open(url, headers={"User-Agent": USER_AGENT})
        setattr(_tls, key, c)
    return c


def stac_search(source: str, *, collections: Sequence[str], retries: int = 6, **kw) -> list[dict]:
    """``ItemSearch.items_as_dicts()`` with retries/backoff and <= 4 concurrent requests per host."""
    kw.setdefault("limit", 100)
    last: Exception | None = None
    for k in range(retries):
        try:
            with _host_sem[source]:
                return list(client(source).search(collections=list(collections), **kw).items_as_dicts())
        except Exception as e:  # noqa: BLE001 — APIError, HTTP/timeout errors: back off and retry
            last = e
            msg = str(e)
            wait = 60 if ("429" in msg or "Too Many" in msg) else 3 * (2 ** k)
            time.sleep(min(wait, 120))
            setattr(_tls, f"client_{source}", None)
    raise RuntimeError(f"STAC search failed ({source} {collections} {kw}): {last!r}")


def http_get(url: str, *, retries: int = 5, timeout: float = 120) -> bytes:
    import requests
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 30)))
                continue
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(3 * (2 ** k))
    raise RuntimeError(f"GET failed: {url[:120]} :: {last}")


SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
_sas_cache: dict[tuple[str, str], tuple[str, float]] = {}
_sas_lock = threading.Lock()


def _sas_token(account: str, container: str) -> str:
    """Anonymous Planetary Computer SAS token for a blob container, cached until 5 min before expiry.

    Replaces ``planetary_computer.sign``: its token request has no HTTP timeout and hung the
    extraction twice (2026-10-22); here every request has connect/read timeouts and retries."""
    key = (account, container)
    now = time.time()
    with _sas_lock:
        tok = _sas_cache.get(key)
        if tok and tok[1] - now > 300:
            return tok[0]
    import requests
    last = None
    for k in range(6):
        try:
            r = requests.get(f"{SAS_TOKEN_URL}/{account}/{container}", timeout=(10, 30),
                             headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                d = r.json()
                exp = parse_time(d.get("msft:expiry"))
                exp_ts = exp.timestamp() if exp else now + 1800
                with _sas_lock:
                    _sas_cache[key] = (d["token"], exp_ts)
                return d["token"]
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 * (2 ** k))
    raise RuntimeError(f"SAS token failed for {account}/{container}: {last}")


def sign(href: str) -> str:
    """Append an anonymous Planetary Computer SAS token to a blob-storage href (others unchanged)."""
    m = re.match(r"https://([^.]+)\.blob\.core\.windows\.net/([^/?]+)/", href)
    if not m:
        return href
    base = href.split("?", 1)[0]
    return f"{base}?{_sas_token(m.group(1), m.group(2))}"


# ----------------------------------------------------------------------------- S2 scenes


@dataclass
class S2Scene:
    id: str
    source: str
    collection: str
    platform: str
    tile: str
    datetime_item: str           # STAC item datetime as published
    sensing_time: str | None     # granule SENSING_TIME (exact overpass, UTC), if known
    baseline: str | None
    cloud_cover: float
    product_uri: str
    relative_orbit: int | None
    sun_zenith: float | None
    sun_azimuth: float | None
    view_zenith: float | None
    view_azimuth: float | None
    epsg: int | None
    geometry: dict
    assets: dict[str, str]                       # canonical layer -> href
    raster_bands: dict[str, dict]                # canonical layer -> raster:bands[0]
    props: dict = field(repr=False, default_factory=dict)
    metadata_hrefs: dict[str, str] = field(default_factory=dict)
    offset: OffsetDecision | None = None
    offset_xml_verified: bool = False

    @property
    def date(self) -> str:
        return (self.sensing_time or self.datetime_item)[:10]

    @property
    def acquisition_key(self) -> tuple[str, str, int | None]:
        """Same physical acquisition (platform, date, relative orbit) regardless of tile/processing."""
        return self.platform, self.date, self.relative_orbit


def _relative_orbit(props: dict) -> int | None:
    if props.get("sat:relative_orbit") is not None:
        return int(props["sat:relative_orbit"])
    m = re.search(r"_R(\d{3})_", props.get("s2:product_uri", "") or "")
    return int(m.group(1)) if m else None


def parse_s2_item(item: dict, source: str) -> S2Scene:
    """STAC item dict (Earth Search C1/legacy or Planetary Computer) -> S2Scene."""
    p = item["properties"]
    coll = item.get("collection", "")
    if source == ES:
        tile = (p.get("grid:code") or "").replace("MGRS-", "")
        if not tile:
            tile = f"{p.get('mgrs:utm_zone')}{p.get('mgrs:latitude_band')}{p.get('mgrs:grid_square')}"
        keys = {l.name: l.es_key for l in S2_LAYERS}
        sun_z = 90.0 - p["view:sun_elevation"] if p.get("view:sun_elevation") is not None else None
        sun_a = p.get("view:sun_azimuth")
        view_z, view_a = p.get("view:incidence_angle"), p.get("view:azimuth")
        sensing = p.get("datetime") if coll == S2_ES_C1 else None
        meta = {"product": item["assets"].get("product_metadata", {}).get("href"),
                "granule": item["assets"].get("granule_metadata", {}).get("href")}
    else:
        tile = p.get("s2:mgrs_tile", "")
        keys = {l.name: l.mpc_key for l in S2_LAYERS}
        sun_z, sun_a = p.get("s2:mean_solar_zenith"), p.get("s2:mean_solar_azimuth")
        view_z = view_a = None
        sensing = None  # MPC datetime is the datatake start; read MTD_TL.xml for the granule time
        meta = {"product": item["assets"].get("product-metadata", {}).get("href"),
                "granule": item["assets"].get("granule-metadata", {}).get("href")}
    assets, rbands = {}, {}
    for name, key in keys.items():
        a = item["assets"].get(key)
        if a is None:
            continue
        assets[name] = a["href"]
        rb = a.get("raster:bands")
        if rb:
            rbands[name] = rb[0]
    epsg = p.get("proj:epsg")
    if epsg is None and str(p.get("proj:code", "")).startswith("EPSG:"):
        epsg = int(str(p["proj:code"]).split(":")[1])
    sc = S2Scene(
        id=item["id"], source=source, collection=coll, platform=platform_code(p.get("platform")),
        tile=tile, datetime_item=p.get("datetime"), sensing_time=sensing,
        baseline=p.get("s2:processing_baseline"), cloud_cover=float(p.get("eo:cloud_cover", np.nan)),
        product_uri=p.get("s2:product_uri", ""), relative_orbit=_relative_orbit(p),
        sun_zenith=sun_z, sun_azimuth=sun_a, view_zenith=view_z, view_azimuth=view_a, epsg=epsg,
        geometry=item.get("geometry"), assets=assets, raster_bands=rbands, props=p,
        metadata_hrefs={k: v for k, v in meta.items() if v},
    )
    sc.offset = s2_offset_decision(p, rbands.get("B04"))
    return sc


def complete_scene_metadata(scene: S2Scene, *, verify_offset_xml: bool = False) -> S2Scene:
    """Fetch granule (and optionally product) XML for exact sensing time, view angles and the
    BOA_ADD_OFFSET evidence. Always done for Planetary Computer items (their STAC datetime is the
    datatake start and they carry no offset metadata)."""
    need_granule = scene.sensing_time is None or scene.view_zenith is None
    if need_granule and "granule" in scene.metadata_hrefs:
        g = parse_granule_xml(http_get(sign(scene.metadata_hrefs["granule"])))
        scene.sensing_time = scene.sensing_time or g.get("sensing_time")
        if scene.view_zenith is None:
            scene.view_zenith, scene.view_azimuth = g.get("view_zenith"), g.get("view_azimuth")
        if scene.sun_zenith is None:
            scene.sun_zenith, scene.sun_azimuth = g.get("sun_zenith"), g.get("sun_azimuth")
    if (verify_offset_xml or scene.source == MPC) and "product" in scene.metadata_hrefs:
        offs = parse_boa_offsets(http_get(sign(scene.metadata_hrefs["product"])))
        scene.offset = s2_offset_decision(scene.props, scene.raster_bands.get("B04"),
                                          xml_offset_from_bands(offs))
        scene.offset_xml_verified = True
    return scene


def search_s2(source: str, *, date: str | None = None, start: str | None = None, end: str | None = None,
              tiles: Iterable[str] | None = None, intersects: dict | None = None,
              bbox: Sequence[float] | None = None, legacy: bool = False) -> list[S2Scene]:
    """S2 L2A items for one UTC day (``date``) or [start, end] (inclusive dates), optionally limited
    to MGRS tiles. Earth Search uses ``sentinel-2-c1-l2a`` unless ``legacy``."""
    if date:
        start = end = date
    dtr = f"{start}T00:00:00Z/{end}T23:59:59Z"
    kw: dict[str, Any] = {"datetime": dtr}
    if intersects is not None:
        kw["intersects"] = intersects
    if bbox is not None:
        kw["bbox"] = list(bbox)
    if source == ES:
        coll = S2_ES_LEGACY if legacy else S2_ES_C1
        if tiles:
            kw["query"] = {"grid:code": {"in": [f"MGRS-{t}" for t in sorted(set(tiles))]}}
    else:
        coll = S2_MPC
        if tiles:
            kw["query"] = {"s2:mgrs_tile": {"in": sorted(set(tiles))}}
    items = stac_search(source, collections=[coll], **kw)
    return [parse_s2_item(it, source) for it in items]


def dedupe_mpc_reprocessing(scenes: list[S2Scene]) -> list[S2Scene]:
    """Planetary Computer holds original + reprocessed items for one acquisition and tile (e.g.
    2022: 04.00 and 05.10). Keep the highest processing baseline per (tile, acquisition)."""
    best: dict[tuple, S2Scene] = {}
    for s in scenes:
        k = (s.tile, s.acquisition_key)
        cur = best.get(k)
        if cur is None or (parse_baseline(s.baseline) or 0) > (parse_baseline(cur.baseline) or 0):
            best[k] = s
    return list(best.values())


def point_interior_distance(geometry: dict, lon: float, lat: float) -> float:
    """Signed distance (degrees, lon scaled by cos lat) from the point to the footprint boundary;
    positive inside."""
    from shapely.geometry import Point, shape
    g = shape(geometry)
    p = Point(lon, lat)
    d = g.boundary.distance(p)  # degrees; only used to rank footprints for the same point
    return d if g.contains(p) else -d


def rank_scenes_for_point(scenes: list[S2Scene], lon: float, lat: float) -> list[S2Scene]:
    """Scenes covering the point, best first: Earth Search C1 > Planetary Computer, higher
    baseline, point most interior to the data footprint, then id (deterministic)."""
    out = []
    for s in scenes:
        d = point_interior_distance(s.geometry, lon, lat)
        if d > 0:
            out.append((0 if s.source == ES else 1, -(parse_baseline(s.baseline) or 0), -d, s.id, s))
    out.sort(key=lambda t: t[:4])
    return [t[-1] for t in out]


# -------------------------------------------------------------------------- raster reading


def _open(href: str):
    import rasterio
    return rasterio.open(sign(href))


def with_retries(fn: Callable[[], Any], tries: int = 4, base: float = 2.0) -> Any:
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — network / signing hiccups
            last = e
            time.sleep(base * (2 ** k))
    raise last  # type: ignore[misc]


def read_window_fn(ds) -> Callable[[int, int, int, int], np.ndarray]:
    from rasterio.windows import Window

    def read(r, c, h, w):
        return ds.read(1, window=Window(c, r, w, h), boundless=True, fill_value=0)
    return read


def project_point(lon: float, lat: float, crs) -> tuple[float, float]:
    from pyproj import Transformer
    tr = _transformer("EPSG:4326", str(crs))
    return tr.transform(lon, lat)


_tr_cache: dict[tuple[str, str], Any] = {}
_tr_lock = threading.Lock()


def _transformer(src: str, dst: str):
    from pyproj import Transformer
    key = (src, dst)
    with _tr_lock:
        t = _tr_cache.get(key)
        if t is None:
            t = Transformer.from_crs(src, dst, always_xy=True)
            _tr_cache[key] = t
    return t


def s2_band_windows(href: str, layer: Layer, points: Sequence[tuple[float, float]], size: int = 64,
                    ref_res: float = 10.0) -> tuple[list[np.ndarray], list[ChipGrid], dict]:
    """Read one S2 band for several points of the same scene onto 10 m chip grids.

    Returns (arrays on the chip grids, chip grids, band info). The chip grid is anchored on the
    tile origin (shared by all S2 bands), so all layers of a chip align exactly.
    """
    def run():
        with _open(href) as ds:
            t = ds.transform
            x0, y0 = t.c, t.f
            res = t.a
            factor = int(round(res / ref_res))
            if abs(res - factor * ref_res) > 1e-6 or abs(t.e + res) > 1e-6 or t.b or t.d:
                raise ValueError(f"unexpected S2 grid for {layer.name}: {t}")
            reader = read_window_fn(ds)
            arrays, grids = [], []
            for lon, lat in points:
                x, y = project_point(lon, lat, ds.crs)
                g = chip_grid(x, y, x0, y0, ref_res, size, str(ds.crs))
                arrays.append(replicate_window(reader, g, factor))
                grids.append(g)
            info = {"crs": str(ds.crs), "x0": x0, "y0": y0, "native_res": res, "factor": factor,
                    "dtype": ds.dtypes[0], "block": ds.block_shapes[0]}
            return arrays, grids, info
    return with_retries(run)


def read_s2_chips(scene: S2Scene, points: Sequence[tuple[float, float]], size: int = 64,
                  layers: Sequence[Layer] = S2_LAYERS, max_workers: int = 1,
                  executor=None) -> tuple[list[np.ndarray], list[ChipGrid], dict]:
    """All layers for several points of one scene -> int16 chips (L, size, size) on the 10 m grid.

    Reflectance layers are harmonised (raw + offset_dn; nodata -32768); AOT is raw DN (scale 1e-3,
    nodata -32768); SCL is the class code (0 = no data). ``executor``: optional shared thread pool
    used to read layers in parallel (the caller bounds total concurrency).
    """
    if scene.offset is None or not scene.offset.consistent:
        raise ValueError(f"inconsistent/unknown BOA offset for {scene.id}: {scene.offset}")
    missing = [l.name for l in layers if l.name not in scene.assets]
    if missing:
        raise KeyError(f"{scene.id} lacks assets {missing}")

    def one(layer: Layer):
        return layer, s2_band_windows(scene.assets[layer.name], layer, points, size)

    if executor is not None:
        results = list(executor.map(one, layers))
    else:
        results = [one(l) for l in layers]
    grids_ref = None
    stack = [np.empty((len(layers), size, size), np.int16) for _ in points]
    info: dict[str, Any] = {}
    for li, (layer, (arrays, grids, binfo)) in enumerate(results):
        if grids_ref is None:
            grids_ref = grids
            info.update({"crs": binfo["crs"], "x0": binfo["x0"], "y0": binfo["y0"]})
        else:
            for g0, g1 in zip(grids_ref, grids):
                if (g0.row0, g0.col0, g0.crs, g0.x0, g0.y0) != (g1.row0, g1.col0, g1.crs, g1.x0, g1.y0):
                    raise ValueError(f"layer {layer.name} grid differs from reference in {scene.id}")
        for pi, a in enumerate(arrays):
            if layer.kind == "reflectance":
                stack[pi][li] = harmonise_dn(a, scene.offset.offset_dn)
            elif layer.kind == "aot":
                stack[pi][li] = harmonise_dn(a, 0)
            else:
                stack[pi][li] = np.asarray(a).astype(np.int16)
    return stack, grids_ref or [], info


# ----------------------------------------------------------------------------------- JRC GSW


def jrc_tile_key(lon: float, lat: float) -> tuple[int, int]:
    """JRC GSW 10-degree tiles are named by their upper-left corner, e.g. (-100, 50) = 100W_50N."""
    return int(math.floor(lon / 10.0) * 10), int(math.floor(lat / 10.0) * 10 + 10)


class JrcOccurrence:
    """Cached JRC Global Surface Water v1.3 occurrence (0-100 %, 1984-2020; 0.00025 deg) windows."""

    def __init__(self, half_px: int = 48):
        self.half_px = half_px
        self._items: dict[tuple[int, int], str | None] = {}
        self._lock = threading.Lock()
        self._local = threading.local()  # per-thread open datasets (10-degree tiles are reused)

    def _dataset(self, href: str):
        cache = self._local.__dict__.setdefault("ds", {})
        ds = cache.get(href)
        if ds is None:
            ds = _open(href)
            cache[href] = ds
        return ds

    def _href(self, lon: float, lat: float) -> str | None:
        key = jrc_tile_key(lon, lat)
        with self._lock:
            if key in self._items:
                return self._items[key]
        lon_c, lat_c = key[0] + 5.0, key[1] - 5.0
        items = stac_search(MPC, collections=[JRC_MPC],
                            intersects={"type": "Point", "coordinates": [lon_c, lat_c]})
        href = items[0]["assets"]["occurrence"]["href"] if items else None
        with self._lock:
            self._items[key] = href
        return href

    def window(self, lon: float, lat: float) -> tuple[np.ndarray, Any] | None:
        """(uint8 window centred on the point's pixel, its affine transform); 255 = no data."""
        href = self._href(lon, lat)
        if href is None:
            return None

        def run():
            from rasterio.windows import Window
            ds = self._dataset(href)
            r, c = ds.index(lon, lat)
            h = self.half_px
            w = Window(c - h, r - h, 2 * h + 1, 2 * h + 1)
            a = ds.read(1, window=w, boundless=True, fill_value=JRC_NODATA)
            return a, ds.window_transform(w)
        try:
            a, tr = with_retries(run)
        except Exception:
            self._local.__dict__.clear()  # drop possibly stale handles (expired SAS token)
            a, tr = with_retries(run)
        a = a.copy()
        a[a > 100] = JRC_NODATA
        return a, tr


def jrc_point_stats(occ: np.ndarray, transform, lon: float, lat: float,
                    water_thr: int = 50, radius_m: float = 60.0) -> dict[str, float]:
    """Occurrence at the point and distances (m) to the nearest water (occ >= thr) and non-water
    pixel centres inside the window; NaN when none is found inside the window."""
    a = np.asarray(occ)
    rows, cols = np.indices(a.shape)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    kx = 111_320.0 * math.cos(math.radians(lat))
    ky = 110_574.0
    d = np.hypot((xs - lon) * kx, (ys - lat) * ky)
    valid = a != JRC_NODATA
    water = valid & (a >= water_thr)
    land = valid & (a < water_thr)
    rc = np.unravel_index(np.argmin(d), d.shape)
    occ_pt = float(a[rc]) if valid[rc] else float("nan")
    within = d <= radius_m
    out = {
        "jrc_occ_point": occ_pt,
        "jrc_occ_max_60m": float(a[within & valid].max()) if (within & valid).any() else float("nan"),
        "jrc_dist_water_m": float(d[water].min()) if water.any() else float("nan"),
        "jrc_dist_land_m": float(d[land].min()) if land.any() else float("nan"),
        "jrc_window_radius_m": float(min(abs(transform.a) * kx, abs(transform.e) * ky) * a.shape[0] / 2),
    }
    out["jrc_water_ok"] = bool((occ_pt >= water_thr) or (out["jrc_dist_water_m"] <= radius_m))
    return out


def jrc_on_chip_grid(occ: np.ndarray, transform, grid: ChipGrid) -> np.ndarray:
    """Reproject a JRC occurrence window (EPSG:4326) onto a chip grid (nearest); 255 = no data."""
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject
    dst = np.full((grid.size, grid.size), JRC_NODATA, np.uint8)
    reproject(np.asarray(occ, np.uint8), dst, src_transform=transform, src_crs=CRS.from_epsg(4326),
              dst_transform=grid.transform, dst_crs=CRS.from_string(grid.crs),
              resampling=Resampling.nearest, src_nodata=JRC_NODATA, dst_nodata=JRC_NODATA)
    return dst


# ---------------------------------------------------------------------------------- Landsat


def search_landsat(*, start: str, end: str, intersects: dict | None = None,
                   bbox: Sequence[float] | None = None,
                   platforms: Sequence[str] = ("landsat-8", "landsat-9")) -> list[dict]:
    kw: dict[str, Any] = {"datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                          "query": {"platform": {"in": list(platforms)}}}
    if intersects is not None:
        kw["intersects"] = intersects
    if bbox is not None:
        kw["bbox"] = list(bbox)
    return stac_search(MPC, collections=[LANDSAT_MPC], **kw)


def landsat_scene_meta(item: dict) -> dict[str, Any]:
    p = item["properties"]
    return {
        "item_id": item["id"], "source": MPC, "collection": LANDSAT_MPC,
        "platform": platform_code(p.get("platform")), "datetime_utc": p.get("datetime"),
        "cloud_cover": p.get("eo:cloud_cover"), "wrs_path": p.get("landsat:wrs_path"),
        "wrs_row": p.get("landsat:wrs_row"), "collection_category": p.get("landsat:collection_category"),
        "sun_zenith": (90 - p["view:sun_elevation"]) if p.get("view:sun_elevation") is not None else None,
        "sun_azimuth": p.get("view:sun_azimuth"), "view_off_nadir": p.get("view:off_nadir"),
        "scale_offset": {k: (item["assets"][k].get("raster:bands") or [{}])[0]
                         for k in LANDSAT_LAYERS if k in item["assets"]},
    }


def read_landsat_chips(item: dict, points: Sequence[tuple[float, float]], size: int = 32,
                       executor=None) -> tuple[list[np.ndarray], list[ChipGrid]]:
    """uint16 chips (10, size, size) at 30 m: 7 SR bands + ST (lwir11) + QA_PIXEL + QA_AEROSOL (raw DN;
    scale/offset in ``landsat_scene_meta``). All C2 L2 30 m layers share one grid."""
    keys = [k for k in LANDSAT_LAYERS if k in item["assets"]]

    def one(key):
        def run():
            with _open(item["assets"][key]["href"]) as ds:
                t = ds.transform
                reader = read_window_fn(ds)
                outs, grids = [], []
                for lon, lat in points:
                    x, y = project_point(lon, lat, ds.crs)
                    g = chip_grid(x, y, t.c, t.f, t.a, size, str(ds.crs))
                    outs.append(reader(g.row0, g.col0, size, size))
                    grids.append(g)
                return outs, grids
        return key, with_retries(run)

    res = list(executor.map(one, keys)) if executor is not None else [one(k) for k in keys]
    chips = [np.zeros((len(LANDSAT_LAYERS), size, size), np.uint16) for _ in points]
    grids = res[0][1][1]
    for key, (outs, _g) in res:
        li = LANDSAT_LAYERS.index(key)
        for pi, a in enumerate(outs):
            chips[pi][li] = np.asarray(a).astype(np.uint16)
    return chips, grids


# -------------------------------------------------------------------------------------- NAIP


def search_naip(*, intersects: dict | None = None, bbox: Sequence[float] | None = None,
                start: str | None = None, end: str | None = None, state: str | None = None,
                year: int | None = None) -> list[dict]:
    kw: dict[str, Any] = {}
    if intersects is not None:
        kw["intersects"] = intersects
    if bbox is not None:
        kw["bbox"] = list(bbox)
    if start and end:
        kw["datetime"] = f"{start}T00:00:00Z/{end}T23:59:59Z"
    args = []
    if state:
        args.append({"op": "=", "args": [{"property": "naip:state"}, state.lower()]})
    if year:
        args.append({"op": "=", "args": [{"property": "naip:year"}, str(year)]})
    if args:
        kw["filter"] = {"op": "and", "args": args} if len(args) > 1 else args[0]
        kw["filter_lang"] = "cql2-json"
    return stac_search(MPC, collections=[NAIP_MPC], **kw)


def naip_acquisition_date(item: dict) -> str:
    """NAIP STAC ``datetime`` holds the acquisition date (time of day is not recorded)."""
    return item["properties"]["datetime"][:10]


def read_naip_chips(item: dict, points: Sequence[tuple[float, float]],
                    size: int = 256) -> tuple[list[np.ndarray], list[ChipGrid], dict]:
    """uint8 chips (4, size, size) at native GSD, band order R, G, B, NIR (digital numbers)."""
    def run():
        from rasterio.windows import Window
        with _open(item["assets"]["image"]["href"]) as ds:
            t = ds.transform
            outs, grids = [], []
            for lon, lat in points:
                x, y = project_point(lon, lat, ds.crs)
                g = chip_grid(x, y, t.c, t.f, t.a, size, str(ds.crs))
                a = ds.read(window=Window(g.col0, g.row0, size, size), boundless=True, fill_value=0)
                outs.append(a.astype(np.uint8))
                grids.append(g)
            info = {"crs": str(ds.crs), "gsd": t.a, "count": ds.count, "width": ds.width,
                    "height": ds.height}
            return outs, grids, info
    return with_retries(run)


def naip_scene_stats(item: dict, decimation: int = 16) -> dict[str, Any]:
    """Per-scene (DOQQ) radiometric statistics from a decimated read of the whole image: per-band
    percentiles/mean/std over non-zero pixels, and the same over 'dark-NIR' (NIR < green, water-like)
    pixels. NAIP is 8-bit, vendor colour-balanced and not reflectance; these statistics support
    per-scene normalisation, they do not calibrate anything."""
    def run():
        with _open(item["assets"]["image"]["href"]) as ds:
            h, w = max(1, ds.height // decimation), max(1, ds.width // decimation)
            a = ds.read(out_shape=(ds.count, h, w)).astype(np.float32)
            return a, ds.width, ds.height
    a, W, H = with_retries(run)
    valid = (a > 0).all(axis=0)
    darknir = valid & (a[3] < a[1])
    out: dict[str, Any] = {"naip_item_id": item["id"], "decimation": decimation, "width": W, "height": H,
                           "valid_frac": float(valid.mean()), "darknir_frac": float(darknir.mean())}
    qs = (1, 2, 5, 25, 50, 75, 95, 98, 99)
    for bi, b in enumerate(NAIP_BANDS):
        v = a[bi][valid]
        if v.size:
            out[f"{b}_mean"], out[f"{b}_std"] = float(v.mean()), float(v.std())
            for q, val in zip(qs, np.percentile(v, qs)):
                out[f"{b}_p{q:02d}"] = float(val)
        dv = a[bi][darknir]
        out[f"{b}_darknir_median"] = float(np.median(dv)) if dv.size else float("nan")
    return out
