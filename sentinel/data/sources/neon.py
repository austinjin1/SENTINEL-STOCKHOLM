# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""NEON Data Portal API client (aquatic products): build requests, select files, parse tables.

All functions here are pure (no network): build a request, pick files from an API response,
parse a downloaded CSV into a tidy, typed DataFrame. The polite HTTP loop, the resume-safe
ledger and the raw layout live in ``scripts/data/download_neon.py``.

Access (verified 2026-10-24): ``/products`` and ``/sites`` are open; ``/data`` needs an API
token since 2026-06-30, sent ONLY as the ``X-API-Token`` header (never as a query parameter, so
no URL we log or store can carry it). File URLs returned by ``/data`` are pre-signed Google
Cloud Storage links; their query string (signature) is stripped by :func:`strip_url` before
anything is written to disk. License: CC BY 4.0 (previously CC0).

Releases: every artifact records the release. Data in ``RELEASE-2026`` are pinned; newer
months exist only as ``PROVISIONAL`` (not citable, may change) and are always flagged
(``provisional`` column). OTU identifiers in DP1.20141.002 are release-specific
(``ay26_16s_otu.*`` in RELEASE-2026) and must never be matched across releases.

QC conventions: sensor values with ``*FinalQF == 1`` are masked to NaN (the flag is kept in
the raw Parquet); chemistry keeps ``belowDetectionQF`` as ``censored`` + ``detection_limit``
(no DL/2 substitution); units come from each product's ``variables`` file and stay in a
``unit`` column / units table.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

API = "https://data.neonscience.org/api/v0"
USER_AGENT = "SENTINEL-v2 research (public-data download)"
RELEASE = "RELEASE-2026"
PROVISIONAL = "PROVISIONAL"
LICENSE = "CC BY 4.0 (NEON data; since 2026-06-30, previously CC0)"
CITATION = ("NSF National Ecological Observatory Network (NEON). Data products as listed, "
            "RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; "
            "https://data.neonscience.org (accessed 2026-10-24).")

# 34 aquatic sites (24 wadeable streams, 3 non-wadeable rivers, 7 lakes; 19 domains).
AQUATIC_SITES = (
    "ARIK", "BARC", "BIGC", "BLDE", "BLUE", "BLWA", "CARI", "COMO", "CRAM", "CUPE", "FLNT",
    "GUIL", "HOPB", "KING", "LECO", "LEWI", "LIRO", "MART", "MAYF", "MCDI", "MCRA", "OKSR",
    "POSE", "PRIN", "PRLA", "PRPO", "REDB", "SUGG", "SYCA", "TECR", "TOMB", "TOOK", "WALK",
    "WLOU",
)


@dataclass(frozen=True)
class ProductSpec:
    """What to fetch for one NEON product.

    ``tables``: regexes matched against the table name of data files (empty = every table).
    ``kind``: ``os`` (observational, small CSVs kept) or ``is`` (instrumented sensor streams,
    converted to zstd Parquet and the CSV deleted). ``tmi``: for sensor products, the
    allowed temporal index codes (e.g. ``('030',)`` = 30-min files only).
    """

    short: str
    code: str
    name: str
    package: str = "basic"
    kind: str = "os"
    priority: int = 1
    tables: tuple[str, ...] = ()
    tmi: tuple[str, ...] = ()
    lab_files: bool = True
    note: str = ""


PRODUCTS: dict[str, ProductSpec] = {s.short: s for s in [
    # ---- Priority 1 (M3 Microbe + chemistry + sonde) ------------------------------------
    ProductSpec("microbe_tax", "DP1.20141.002", "Surface water microbe community taxonomy",
                package="expanded", tables=(r"mct_surfaceWaterSampleMetadata_(16S|ITS)",
                                             r"mct_taxonTableLabSummary"),
                note="taxon tables (per sequencing run) are external files: downloadFileUrl"),
    ProductSpec("microbe_comp_v1", "DP1.20141.001",
                "Surface water microbe community composition (superseded; 2014-2018)",
                package="expanded", tables=(r"mcc_swSeqVariantMetadata_(16S|ITS)",
                                             r"mcc_taxonTableLabSummary"),
                note="per-sample ASV tables are external files: downloadFileUrl"),
    ProductSpec("microbe_seq", "DP1.20282.001", "Surface water microbe marker gene sequences",
                package="expanded",
                note="metadata only (field, extraction, PCR primers, sequencing, raw-file ids); "
                     "FASTQ not downloaded"),
    ProductSpec("chem", "DP1.20093.001", "Chemical properties of surface water",
                package="expanded", note="expanded for lab detection limits + blanks"),
    ProductSpec("sonde", "DP1.20288.001", "Water quality (multisonde)", kind="is",
                tables=(r"waq_instantaneous", r"ais_maintenance")),
    ProductSpec("nitrate", "DP1.20033.001", "Nitrate in surface water (SUNA)", kind="is",
                tables=(r"NSW_15_minute", r"ais_maintenance", r"science_review_flags")),
    ProductSpec("metagenome", "DP1.20281.001", "Surface water microbe metagenome sequences",
                package="expanded",
                note="metadata + raw-file identifiers only; sequence data via NMDC later"),
    # ---- Priority 2 ------------------------------------------------------------------------
    ProductSpec("tsd", "DP1.20264.001", "Temperature at specific depth in surface water",
                kind="is", priority=2, tables=(r"TSD_30_min",), tmi=("030",)),
    ProductSpec("tsw", "DP1.20053.001", "Temperature (PRT) in surface water", kind="is",
                priority=2, tables=(r"TSW_30min",), tmi=("030",)),
    ProductSpec("eos", "DP1.20016.001", "Elevation of surface water", kind="is", priority=2,
                tables=(r"EOS_30_min",), tmi=("030",)),
    ProductSpec("discharge", "DP4.00130.001", "Continuous discharge", kind="is", priority=2,
                tables=(r"csd_continuousDischarge", r"csd_gaugeWaterColumnRegression")),
    ProductSpec("macroinv", "DP1.20120.001", "Macroinvertebrate collection", priority=2),
    ProductSpec("periphyton", "DP1.20163.001",
                "Periphyton, seston, and phytoplankton chemical properties", priority=2),
    ProductSpec("fish", "DP1.20107.001", "Fish electrofishing, gill netting, and fyke netting "
                "counts", priority=2),
    ProductSpec("zooplankton", "DP1.20219.001", "Zooplankton collection", priority=2),
    ProductSpec("microbe_qpcr", "DP1.20278.001", "Surface water microbe group abundances",
                priority=2, note="qPCR gene copies; biomass covariate for M3"),
]}

BY_CODE = {s.code: s for s in PRODUCTS.values()}
PRIORITY_ORDER = ("microbe_tax", "microbe_comp_v1", "microbe_seq", "chem", "metagenome",
                  "nitrate", "sonde", "tsw", "tsd", "eos", "discharge", "macroinv",
                  "periphyton", "zooplankton", "fish", "microbe_qpcr")

# AOP products: availability recorded only (no download) for the drone workstream.
AOP_PRODUCTS = {
    "DP1.30006.001": "Spectrometer orthorectified surface directional reflectance - flightline",
    "DP3.30006.001": "Spectrometer orthorectified surface directional reflectance - mosaic",
    "DP1.30010.001": "High-resolution orthorectified camera imagery",
    "DP3.30010.001": "High-resolution orthorectified camera imagery mosaic",
    "DP1.30003.001": "Discrete return LiDAR point cloud",
}


# --------------------------------------------------------------------------- credentials
def read_token(env: Mapping[str, str] | None = None, keys_path: Path | None = None) -> str | None:
    """NEON token from ``$NEON_TOKEN`` or ``~/.config/sentinel/keys.env`` (``KEY=value``).

    The value is returned to the caller only; it is never printed, logged or written.
    """
    env = os.environ if env is None else env
    tok = (env.get("NEON_TOKEN") or "").strip()
    if tok:
        return tok
    keys_path = keys_path or Path.home() / ".config" / "sentinel" / "keys.env"
    try:
        lines = keys_path.read_text().splitlines()
    except OSError:
        return None
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("export "):
            ln = ln[7:].strip()
        if ln.startswith("NEON_TOKEN="):
            v = ln.split("=", 1)[1].strip().strip('"').strip("'")
            return v or None
    return None


def request_headers(token: str | None) -> dict[str, str]:
    h = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    if token:
        h["X-API-Token"] = token
    return h


def redact(text: str, token: str | None) -> str:
    """Remove the token (and any signed-URL query) from a message before it is logged."""
    if token:
        text = text.replace(token, "<redacted>")
    return re.sub(r"(https?://[^\s?\"']+)\?[^\s\"']*", r"\1", text)


def strip_url(url: str) -> str:
    """Drop query string + fragment (GCS signatures, any credentials) from a URL."""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))


# --------------------------------------------------------------------------- requests
def product_request(code: str) -> tuple[str, dict]:
    return f"{API}/products/{code}", {}


def site_request(site: str | None = None) -> tuple[str, dict]:
    return (f"{API}/sites/{site}" if site else f"{API}/sites"), {}


def location_request(name: str) -> tuple[str, dict]:
    return f"{API}/locations/{name}", {}


def data_request(code: str, site: str, month: str, release: str = RELEASE,
                 package: str = "basic") -> tuple[str, dict]:
    """Files for one product/site/month. ``release='PROVISIONAL'`` for provisional months."""
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"month must be YYYY-MM, got {month!r}")
    params = {"package": package}
    if release and release != PROVISIONAL:  # 'PROVISIONAL' is not a valid release value;
        params["release"] = release          # omitting it returns the provisional data
    return f"{API}/data/{code}/{site}/{month}", params


def site_months(product_payload: dict, sites: Iterable[str] | None = None,
                releases: Iterable[str] = (RELEASE, PROVISIONAL)) -> pd.DataFrame:
    """(site, month, release) rows available for a product, from ``/products/<code>`` JSON."""
    data = product_payload.get("data", product_payload)
    keep = set(sites) if sites is not None else None
    rel_ok = set(releases)
    rows = []
    for sc in data.get("siteCodes") or []:
        s = sc["siteCode"]
        if keep is not None and s not in keep:
            continue
        for rel in sc.get("availableReleases") or []:
            if rel["release"] not in rel_ok:
                continue
            for m in rel.get("availableMonths") or []:
                rows.append({"site": s, "month": m, "release": rel["release"]})
    df = pd.DataFrame(rows, columns=["site", "month", "release"])
    # a month listed under both labels: the pinned release wins
    df["_r"] = (df["release"] != RELEASE).astype(int)
    df = df.sort_values(["site", "month", "_r"]).drop_duplicates(["site", "month"])
    return df.drop(columns="_r").reset_index(drop=True)


# --------------------------------------------------------------------------- file names
_DATA_RE = re.compile(
    r"^NEON\.(?P<domain>D\d{2})\.(?P<site>[A-Z]{4})\.(?P<dpl>DP\d)\.(?P<prod>\d{5})\.(?P<rev>\d{3})\."
    r"(?:(?P<hor>\d{3})\.(?P<ver>\d{3})\.(?P<tmi>\d{3})\.)?(?P<table>[A-Za-z0-9_]+)\."
    r"(?:(?P<month>\d{4}-\d{2})\.)?(?:(?P<package>basic|expanded)\.)?(?P<ts>\d{8}T\d{6}Z)\.csv$")
_LAB_RE = re.compile(r"^NEON\.(?P<lab>[^.]+)\.(?P<table>[A-Za-z0-9_]+)\.(?P<ts>\d{8}T\d{6}Z)\.csv$")
_META_TABLES = {"variables", "categoricalCodes", "validation", "readme", "EML"}


def parse_file_name(name: str) -> dict:
    """Classify a NEON file name -> dict(role, table, hor, ver, tmi, month, package, ...).

    roles: ``data`` (site-month table), ``site_meta`` (sensor_positions, site-level tables),
    ``shared`` (variables / categoricalCodes / validation: identical across sites),
    ``lab`` (lab-level summary files, ``NEON.<lab>.<table>``), ``doc`` (readme / EML),
    ``external`` (anything else, e.g. per-run taxon tables listed in the package).
    """
    m = _DATA_RE.match(name)
    if m:
        d = m.groupdict()
        t = d["table"]
        if t in _META_TABLES:
            role = "shared"
        elif d["month"] is None:
            role = "site_meta"
        else:
            role = "data"
        d["role"] = role
        return d
    m = _LAB_RE.match(name)
    if m and not re.match(r"^D\d{2}$", m.group("lab")):
        d = m.groupdict()
        d["role"] = "lab"
        return d
    if name.endswith((".xml", ".txt")):
        t = "EML" if name.endswith(".xml") else "readme"
        return {"role": "doc", "table": t}
    return {"role": "external", "table": None}


def shared_name(name: str) -> str:
    """Site-independent name for a shared file (drops the ``NEON.Dxx.SITE.`` prefix)."""
    return re.sub(r"^NEON\.D\d{2}\.[A-Z]{4}\.", "", name)


def select_files(files: list[dict], spec: ProductSpec) -> list[dict]:
    """Choose the files of one ``/data`` response to download.

    Keeps data tables of ``spec.package`` (tables shared between packages appear once),
    matching ``spec.tables``/``spec.tmi``; shared metadata (variables, categoricalCodes,
    validation), lab summaries and site metadata (sensor_positions); skips readme/EML (one
    readme per product is fetched separately) and in-package external files (we fetch
    taxon tables via their ``downloadFileUrl`` instead, deduplicated across site-months).
    Returns dicts with name, url (signed; never store it unstripped), size, md5, role, table.
    """
    seen: set[str] = set()
    out = []
    for f in files:
        name = f["name"]
        if name in seen:
            continue
        info = parse_file_name(name)
        role, table = info["role"], info.get("table")
        if role in ("doc", "external"):
            continue
        if role == "lab" and not spec.lab_files:
            continue
        if role == "data":
            pkg = info.get("package")
            if pkg and pkg != spec.package:
                continue
            if spec.tables and not any(re.fullmatch(p, table) for p in spec.tables):
                continue
            if spec.tmi and info.get("tmi") and info["tmi"] not in spec.tmi:
                continue
        seen.add(name)
        out.append({"name": name, "url": f["url"], "size": int(f.get("size") or 0),
                    "md5": f.get("md5"), "role": role, "table": table,
                    "hor": info.get("hor"), "ver": info.get("ver"), "tmi": info.get("tmi")})
    # when the expanded package was requested, basic copies of the same table are redundant
    return out


def readme_file(files: list[dict], package: str = "basic") -> dict | None:
    for f in files:
        if f["name"].endswith(".txt") and "readme" in f["name"]:
            return f
    return None


# --------------------------------------------------------------------------- parsing
def read_variables(path_or_buf) -> pd.DataFrame:
    """A product ``variables`` file: table, fieldName, description, dataType, units, ..."""
    v = pd.read_csv(path_or_buf, dtype=str, keep_default_na=False)
    v.columns = [c.strip() for c in v.columns]
    return v


_INT_TYPES = {"signed integer", "unsigned integer", "integer"}
_REAL_TYPES = {"real"}
_TIME_TYPES = {"dateTime"}


def read_table(path_or_buf, variables: pd.DataFrame | None = None, table: str | None = None
               ) -> pd.DataFrame:
    """Read one NEON CSV table with types from the variables file.

    dateTime -> UTC ``datetime64[ns, UTC]``; real -> float64; integers -> Int64; everything
    else (IDs, codes, barcodes, siteID, namedLocation) -> string, so leading zeros survive.
    Without ``variables`` all columns are read as strings except recognised time columns.
    """
    df = pd.read_csv(path_or_buf, dtype=str, keep_default_na=False, na_values=[""])
    types: dict[str, str] = {}
    if variables is not None and len(variables):
        v = variables if table is None else variables[variables["table"] == table]
        types = dict(zip(v["fieldName"], v["dataType"]))
    for c in df.columns:
        t = types.get(c)
        if t in _TIME_TYPES or (t is None and c in ("startDateTime", "endDateTime",
                                                      "collectDate", "endDate")):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="mixed")
        elif t in _REAL_TYPES:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t in _INT_TYPES:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        else:
            df[c] = df[c].astype("string")
    return df


def units_for(variables: pd.DataFrame, table: str) -> dict[str, str]:
    v = variables[variables["table"] == table]
    return {f: u for f, u in zip(v["fieldName"], v["units"]) if u and u != "NA"}


def sensor_csv_to_frame(path_or_buf) -> pd.DataFrame:
    """Instrumented-sensor CSV -> compact typed frame for raw Parquet storage (lossless for
    the published precision): times UTC, QF columns Int8, values float32."""
    df = pd.read_csv(path_or_buf, dtype=str, keep_default_na=False, na_values=[""])
    for c in df.columns:
        if c in ("startDateTime", "endDateTime", "endDate"):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="mixed")
        elif c.endswith("QF") or c.endswith("QFSciRvw") or c == "buoyNAFlag":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int8")
        elif c in ("siteID", "stationHorizontalID", "namedLocation", "curveID", "regressionID"):
            df[c] = df[c].astype("string")
        else:
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().sum() == df[c].notna().sum():
                # float32 keeps >= 7 significant digits: exact for NEON's published
                # precision below 1e4; larger magnitudes (discharge in L/s) stay float64.
                big = bool((num.abs() >= 1e4).any())
                df[c] = num.astype("float64" if big else "float32")
            else:
                df[c] = df[c].astype("string")
    return df


# --------------------------------------------------------------------------- sensors
@dataclass(frozen=True)
class SensorVar:
    out: str        # SENTINEL variable name (unit-suffixed)
    value: str      # NEON value column
    qf: str         # NEON final quality flag column
    unit: str       # NEON unit string (from variables file)
    alts: tuple[tuple[str, str], ...] = ()  # alternative (value, qf) column names


def resolve_vars(columns: Iterable[str], variables: Iterable[SensorVar]) -> list[SensorVar]:
    """Map each SensorVar onto the columns present (first matching alternative)."""
    cols = set(columns)
    out = []
    for v in variables:
        cands = ((v.value, v.qf), *v.alts)
        hit = next(((a, q) for a, q in cands if a in cols and q in cols), None)
        if hit is None:  # value without any known QF column: kept, nothing masked
            hit = next(((a, q) for a, q in cands if a in cols), None)
        if hit is not None:
            out.append(SensorVar(v.out, hit[0], hit[1], v.unit))
    return out


SONDE_VARS = (
    SensorVar("spcond_uscm", "specificConductance", "specificCondFinalQF",
              "microsiemensPerCentimeter"),
    SensorVar("do_mgl", "dissolvedOxygen", "dissolvedOxygenFinalQF", "milligramsPerLiter"),
    SensorVar("do_sat_sealevel_pct", "seaLevelDissolvedOxygenSat", "seaLevelDOSatFinalQF",
              "percent"),
    SensorVar("do_sat_local_pct", "localDissolvedOxygenSat", "localDOSatFinalQF", "percent"),
    SensorVar("ph", "pH", "pHFinalQF", "pH"),
    SensorVar("chl_ugl", "chlorophyll", "chlorophyllFinalQF", "microgramsPerLiter"),
    SensorVar("chl_rfu", "chlaRelativeFluorescence", "chlaRelFluoroFinalQF",
              "relativeFluorescenceUnits"),
    SensorVar("turb_fnu", "turbidity", "turbidityFinalQF", "formazinNephelometricUnit"),
    SensorVar("fdom_qsu", "fDOM", "fDOMFinalQF", "quinineSulfateUnit"),
    SensorVar("sensor_depth_m", "sensorDepth", "sensorDepthFinalQF", "meter"),
)
NITRATE_VARS = (SensorVar("no3_umoll", "surfWaterNitrateMean", "finalQF", "micromolesPerLiter"),)
TSW_VARS = (SensorVar("temp_c", "surfWaterTempMean", "finalQF", "celsius"),)
TSD_VARS = (SensorVar("temp_c", "tsdWaterTempMean", "finalQF", "celsius",
                      alts=(("tsdWaterTempMean", "tsdWaterTempFinalQF"),
                            ("surfWaterTempMean", "finalQF"))),
            SensorVar("thermistor_depth_m", "thermistorDepth", "tsdWaterTempFinalQF", "meter"))
EOS_VARS = (SensorVar("wse_m", "surfacewaterElevMean", "sWatElevFinalQF", "meter"),)
DISCHARGE_VARS = (
    SensorVar("q_ls", "maxpostDischarge", "dischargeFinalQF", "litersPerSecond"),
    SensorVar("stage_m", "equivalentStage", "dischargeFinalQF", "meter"),
)


def mask_qf(df: pd.DataFrame, variables: Iterable[SensorVar]) -> tuple[pd.DataFrame, dict]:
    """Set values to NaN where the final QF == 1 (fail) — or, if a science-review column
    ``<qf>SciRvw`` exists, where it == 2 (redacted). Returns (frame, counts per variable)."""
    df = df.copy()
    counts = {}
    for v in variables:
        if v.value not in df.columns:
            continue
        val = pd.to_numeric(df[v.value], errors="coerce").astype("float64")
        bad = pd.Series(False, index=df.index)
        if v.qf in df.columns:
            bad |= pd.to_numeric(df[v.qf], errors="coerce").fillna(0).astype(int).eq(1)
        sr = v.qf + "SciRvw"
        if sr in df.columns:
            bad |= pd.to_numeric(df[sr], errors="coerce").fillna(0).astype(int).eq(2)
        n_valid_before = int(val.notna().sum())
        val[bad] = np.nan
        counts[v.out] = {"n": n_valid_before, "masked_qf": int((bad & df[v.value].notna()).sum())}
        df[v.value] = val
    return df, counts


def aggregate_hourly(df: pd.DataFrame, variables: Iterable[SensorVar],
                     time_col: str = "startDateTime") -> pd.DataFrame:
    """QF-masked instantaneous frame -> hourly (UTC) ``<out>_mean/_min/_max/_n``.

    Hours with no valid value are absent. Values that end up NaN are not counted.
    """
    variables = [v for v in variables if v.value in df.columns]
    if df.empty or not variables:
        return pd.DataFrame()
    t = pd.to_datetime(df[time_col], utc=True)
    hour = t.dt.floor("h")
    cols = {}
    for v in variables:
        s = pd.to_numeric(df[v.value], errors="coerce").astype("float64")
        g = s.groupby(hour)
        n = g.count()
        cols[f"{v.out}_mean"] = g.mean()
        cols[f"{v.out}_min"] = g.min()
        cols[f"{v.out}_max"] = g.max()
        cols[f"{v.out}_n"] = n
    out = pd.DataFrame(cols)
    ncols = [c for c in out.columns if c.endswith("_n")]
    out = out[out[ncols].sum(axis=1) > 0]
    for c in out.columns:
        out[c] = out[c].astype("int32") if c.endswith("_n") else out[c].astype("float32")
    out.index.name = "time_utc"
    return out


def hourly_to_daily(hourly: pd.DataFrame, utc_offset_h: float, min_hours: int = 18
                    ) -> pd.DataFrame:
    """Hourly mean/min/max/n -> local-standard-time daily stats.

    ``date`` is the local standard-time calendar day (fixed UTC offset, no DST, matching the
    USGS DV convention). Daily mean = count-weighted mean of hourly means; min/max of hourly
    min/max; ``<var>_hours`` = hours with data; days with fewer than ``min_hours`` hours for
    a variable are NaN for that variable (kept with their hour count).
    """
    if hourly.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(hourly.index, utc=True)
    local = (idx.tz_convert(None) + pd.Timedelta(hours=float(utc_offset_h)))
    day = pd.Index(local.normalize(), name="date")
    bases = sorted({c[:-5] for c in hourly.columns if c.endswith("_mean")})
    cols = {}
    for b in bases:
        m = hourly[f"{b}_mean"].astype("float64").to_numpy()
        n = hourly[f"{b}_n"].astype("float64").to_numpy()
        w = np.where(np.isfinite(m), n, 0.0)
        frame = pd.DataFrame({"s": np.where(np.isfinite(m), m * n, 0.0), "w": w,
                              "mn": hourly[f"{b}_min"].to_numpy(),
                              "mx": hourly[f"{b}_max"].to_numpy(),
                              "h": (w > 0).astype(int)}, index=day)
        g = frame.groupby(level=0)
        s, ww, hrs = g["s"].sum(), g["w"].sum(), g["h"].sum()
        mean = (s / ww.replace(0, np.nan))
        ok = hrs >= min_hours
        cols[f"{b}_mean"] = mean.where(ok)
        cols[f"{b}_min"] = g["mn"].min().where(ok)
        cols[f"{b}_max"] = g["mx"].max().where(ok)
        cols[f"{b}_hours"] = hrs
    out = pd.DataFrame(cols)
    hcols = [c for c in out.columns if c.endswith("_hours")]
    out = out[out[hcols].sum(axis=1) > 0]
    for c in out.columns:
        out[c] = out[c].astype("int16") if c.endswith("_hours") else out[c].astype("float32")
    return out


# --------------------------------------------------------------------------- chemistry
CENSOR_CODES = {"BDL", "ND"}  # below method detection limit / not detected


def tidy_chem(ext: pd.DataFrame, lab_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """``swc_externalLabDataByAnalyte`` -> long tidy table with censoring + detection limits.

    ``value`` is the concentration NEON published (NaN when not reported, e.g. ``ND``);
    ``censored`` = belowDetectionQF in {BDL, ND}; ``detection_limit`` = lab method detection
    limit for that lab/analyte/date when the lab summary provides one. No substitution.
    """
    d = ext.copy()
    out = pd.DataFrame({
        "site": d["siteID"].astype("string"),
        "domain": d["domainID"].astype("string"),
        "named_location": d["namedLocation"].astype("string"),
        "sample_id": d["sampleID"].astype("string"),
        "sample_code": d.get("sampleCode", pd.Series(pd.NA, index=d.index)).astype("string"),
        "time_utc": pd.to_datetime(d["collectDate"], utc=True),
        "lab": d["laboratoryName"].astype("string"),
        "analyte": d["analyte"].astype("string").str.strip(),
        "value": pd.to_numeric(d["analyteConcentration"], errors="coerce").astype("float64"),
        "unit": d["analyteUnits"].astype("string"),
        "qualifier": d["belowDetectionQF"].astype("string"),
        "shipment_warm_qf": pd.to_numeric(d.get("shipmentWarmQF"), errors="coerce").astype("Int8"),
        "lab_data_qf": d.get("externalLabDataQF", pd.Series(pd.NA, index=d.index)).astype("string"),
        "sample_condition": d.get("sampleCondition", pd.Series(pd.NA, index=d.index)).astype("string"),
        "uid": d["uid"].astype("string"),
    })
    out["sample_fraction"] = out["sample_id"].str.extract(r"\.(FIL|RAW|PCN|POM(?:\.\d+)?)$",
                                                         expand=False).astype("string")
    out["censored"] = out["qualifier"].isin(list(CENSOR_CODES)).fillna(False).astype(bool)
    out["detection_limit"] = np.nan
    if lab_summary is not None and len(lab_summary):
        ls = lab_summary.copy()
        mdl_col = next((c for c in ("methodDetectionLimit", "mdl", "detectionLimit")
                        if c in ls.columns), None)
        if mdl_col and "analyte" in ls.columns:
            ls["analyte"] = ls["analyte"].astype(str).str.strip()
            ls["mdl"] = pd.to_numeric(ls[mdl_col], errors="coerce")
            lab_col = "laboratoryName" if "laboratoryName" in ls.columns else None
            start = pd.to_datetime(ls.get("labSpecificStartDate"), utc=True, errors="coerce")
            end = pd.to_datetime(ls.get("labSpecificEndDate"), utc=True, errors="coerce")
            ls["_s"], ls["_e"] = start, end
            unit_col = "analyteUnits" if "analyteUnits" in ls.columns else None
            ls = ls.drop_duplicates([c for c in ("uid",) if c in ls.columns] or None)
            for _, r in ls.dropna(subset=["mdl"]).iterrows():
                m = out["analyte"].eq(r["analyte"])
                if lab_col:
                    m &= out["lab"].eq(str(r[lab_col]))
                if unit_col and isinstance(r[unit_col], str) and r[unit_col]:
                    m &= out["unit"].eq(r[unit_col]) | out["unit"].isna()
                if pd.notna(r["_s"]):
                    m &= out["time_utc"] >= r["_s"]
                if pd.notna(r["_e"]):
                    m &= out["time_utc"] <= r["_e"] + pd.Timedelta(days=1)
                out.loc[m, "detection_limit"] = r["mdl"]
    # derived: value below the matched MDL although NEON did not flag it (early years)
    out["below_mdl"] = (out["value"] < out["detection_limit"]).fillna(False).astype(bool)
    out["date"] = out["time_utc"].dt.tz_convert(None).dt.normalize()
    return out


def fill_missing_units(long: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing ``unit`` only when the SAME lab reports that analyte in exactly one unit
    elsewhere (EcoCore_CSU omitted units for some analytes in 2020 and 2022-23); UV absorbance
    (never given a unit by EcoCore) -> 'absorbance units' (the unit ISWS reports). Every filled
    row has ``unit_imputed=True``; anything else stays NaN (unit is part of identity)."""
    out = long.copy()
    out["unit_imputed"] = False
    miss = out["unit"].isna()
    known = out[~miss].groupby(["lab", "analyte"])["unit"].agg(lambda u: sorted(set(u)))
    for (lab, analyte), idx in out[miss].groupby(["lab", "analyte"]).groups.items():
        units = known.get((lab, analyte))
        fill = None
        if units is not None and len(units) == 1:
            fill = units[0]
        elif str(analyte).startswith("UV Absorbance"):
            fill = "absorbance units"
        if fill is not None:
            out.loc[idx, "unit"] = fill
            out.loc[idx, "unit_imputed"] = True
    return out


# --------------------------------------------------------------------------- microbes
PRIMER_SETS = {
    # (forward, reverse) -> label (NEON_markerGenes_userGuide_vH.1, Table 1)
    ("GTGCCAGCMGCCGCGGTAA", "GGACTACNVGGGTWTCTAAT"): "16S_V4_515F-806R",
    ("CCTACGGGNBGCASCAG", "GACTACNVGGGTATCTAATCC"): "16S_V3V4_Pro341F-Pro805R",
    ("GTGYCAGCMGCCGCGGTAA", "CCGYCAATTYMTTTRAGTTT"): "16S_V4V5_515FY-926R",
}


ITS_PAIR = ("CTTGGTCATTTAGAGGAAGTAA", "GCTGCGTTCTTCATCGATGC")  # ITS1F / ITS2 (fungal)
ITS_IN_16S_ISSUE = ("ITS1F/ITS2 primers recorded in a 16S v4v5 PCR record (GMCF lab, "
                    "2019-10..2021-10); amplicon treated as V4V5 (targetSubfragment, 98% "
                    "Bacteria reads, same amplicon length as V4V5)")


def primer_record_issue(forward: str | None, reverse: str | None,
                        subfragment: str | None = None) -> str | None:
    f = (forward or "").strip().upper()
    r = (reverse or "").strip().upper()
    if (f, r) == ITS_PAIR and (subfragment or "").strip().lower() == "v4v5":
        return ITS_IN_16S_ISSUE
    return None


def primer_set(forward: str | None, reverse: str | None, subfragment: str | None = None) -> str:
    """Label a 16S primer pair; unknown pairs -> ``16S_other:<subfragment>`` (never guessed
    from the collection date). The known GMCF metadata error (ITS primers on a 16S v4v5
    record, see :func:`primer_record_issue`) is labelled V4V5."""
    f = (forward or "").strip().upper()
    r = (reverse or "").strip().upper()
    lab = PRIMER_SETS.get((f, r))
    if lab:
        return lab
    if primer_record_issue(f, r, subfragment):
        return "16S_V4V5_515FY-926R"
    if not f and not r:
        return "unknown"
    return f"16S_other:{(subfragment or '').strip() or 'na'}"


TAXON_RANKS = ("domain", "kingdom", "phylum", "class", "order", "family", "genus",
               "specificEpithet", "scientificName")


def tidy_taxon_table(df: pd.DataFrame, source_file: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run (DP1.20141.002 ``*_16S_map.csv``) or per-sample (DP1.20141.001) taxon table
    -> (counts, otus).

    counts: dnaSampleID, sequencerRunID, sequenceName, count (int64), source_file.
    otus:   sequenceName, sequence, ranks..., confidences (when present) — one row per name.
    """
    d = df.copy()
    for c in ("dnaSampleID", "sequenceName", "sequencerRunID", "taxonSequence"):
        if c not in d.columns:
            d[c] = pd.NA
    cnt = pd.to_numeric(d["individualCount"], errors="coerce")
    counts = pd.DataFrame({
        "dnaSampleID": d["dnaSampleID"].astype("string"),
        "sequencerRunID": d["sequencerRunID"].astype("string"),
        "sequenceName": d["sequenceName"].astype("string"),
        "count": cnt.astype("Int64"),
        "source_file": source_file,
    })
    counts = counts[counts["count"].notna() & (counts["count"] > 0)]
    keep = ["sequenceName", "taxonSequence", *[r for r in TAXON_RANKS if r in d.columns]]
    keep += [c for c in d.columns if c.endswith("Confidence")]
    if "completeTaxonomy" in d.columns:
        keep.append("completeTaxonomy")
    otus = d[keep].drop_duplicates("sequenceName").rename(columns={"taxonSequence": "sequence"})
    for c in otus.columns:
        if c.endswith("Confidence"):
            otus[c] = pd.to_numeric(otus[c], errors="coerce").astype("float32")
        else:
            otus[c] = otus[c].astype("string")
    return counts.reset_index(drop=True), otus.reset_index(drop=True)


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_values=[""])
