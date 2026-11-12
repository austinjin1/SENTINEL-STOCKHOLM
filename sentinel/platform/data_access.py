# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Read-only access to the local data products shown by the Console.

Sources (all under ``$SENTINEL_DATA``; any of them may be absent while other workstreams are
still building them, and every caller must handle that):

* ``processed/sites/sites.parquet``                 site table (preferred for coordinates)
* ``raw/usgs_dv/dv_series_catalog.parquet``         USGS series catalogue (fallback coordinates)
* ``raw/usgs_dv/iv_core_sites.parquet``             core 15-min sites (fallback coordinates)
* ``processed/sonde_daily/{index.parquet,sites/*}``  daily tier (sentinel/data/sonde_daily.py)
* ``processed/sonde_hourly/sites/*``               hourly tier (sentinel/data/sonde_hourly.py)

Nothing here writes to disk. Results are cached in memory and invalidated when a file's
modification time or size changes.
"""

from __future__ import annotations

from ..data.reserved import is_reserved

import functools
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from . import variables as V
from .config import Settings

SITE_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
TIERS = ("daily", "hourly")


class DataUnavailable(LookupError):
    """A local data product (or a site within it) does not exist (yet)."""


class BadRequest(ValueError):
    """Invalid parameters (unknown variable, malformed id or date)."""


# ------------------------------------------------------------------ locations


def daily_dir(s: Settings) -> Path:
    return s.processed / "sonde_daily"


def daily_sites_dir(s: Settings) -> Path:
    return daily_dir(s) / "sites"


def daily_index_path(s: Settings) -> Path:
    return daily_dir(s) / "index.parquet"


def hourly_sites_dir(s: Settings) -> Path:
    return s.processed / "sonde_hourly" / "sites"


def sites_table_path(s: Settings) -> Path:
    return s.processed / "sites" / "sites.parquet"


def dv_catalog_path(s: Settings) -> Path:
    return s.raw / "usgs_dv" / "dv_series_catalog.parquet"


def iv_core_path(s: Settings) -> Path:
    return s.raw / "usgs_dv" / "iv_core_sites.parquet"


def normalize_site_id(raw: str) -> str:
    sid = (raw or "").strip()
    if sid.upper().startswith("USGS-"):
        sid = sid[5:]
    if not SITE_ID_RE.match(sid):
        raise BadRequest(f"invalid site id {raw!r}")
    return sid


# ------------------------------------------------------------------ caching

_lock = threading.Lock()
_cache: dict[Any, tuple[tuple, Any]] = {}


def _stamp(p: Path) -> tuple:
    try:
        st = p.stat()
        return (str(p), st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return (str(p), None, None)


def _cached(key: Any, stamp: tuple, fn: Callable[[], Any]) -> Any:
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    value = fn()
    with _lock:
        _cache[key] = (stamp, value)
    return value


def _iso_date(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _iso_utc(x) -> str | None:
    if x is None or pd.isna(x):
        return None
    t = pd.Timestamp(x)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ site catalogue

_ID_COLS = ("site_id", "site_no", "monitoring_location_id", "id", "site")
_NAME_COLS = ("name", "station_nm", "site_name", "monitoring_location_name", "station_name")
_LAT_COLS = ("lat", "latitude", "dec_lat_va", "y")
_LON_COLS = ("lon", "longitude", "dec_long_va", "lng", "long", "x")
_TYPE_COLS = ("site_type", "site_tp_cd", "type", "site_type_code")
_HUC_COLS = ("huc", "huc_cd", "huc8", "huc12", "huc_12", "hydrologic_unit_code")
_STATE_COLS = ("state", "state_cd", "state_code", "state_name")
_AGENCY_COLS = ("agency", "agency_cd", "agency_code", "network")
_HUC12_COLS = ("huc12", "huc_12")
CATALOG_COLS = [
    "name",
    "lat",
    "lon",
    "site_type",
    "huc",
    "huc12",
    "state",
    "agency",
    "coord_source",
    "catalog_wq_begin",
    "catalog_wq_end",
    "catalog_source",
]


def _pick(df: pd.DataFrame, cands: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in lower:
            return lower[c]
    return None


def _clean_str(sr: pd.Series) -> pd.Series:
    out = sr.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "None", "<NA>"]))


def _standardise(df: pd.DataFrame, source: str) -> pd.DataFrame | None:
    """Map a site table with unknown column names onto the Console's catalogue columns."""
    cid, clat, clon = _pick(df, _ID_COLS), _pick(df, _LAT_COLS), _pick(df, _LON_COLS)
    if cid is None or clat is None or clon is None:
        return None
    out = pd.DataFrame({"site_id": _clean_str(df[cid]).str.replace(r"^USGS-", "", regex=True)})
    out["lat"] = pd.to_numeric(df[clat], errors="coerce").astype("float64")
    out["lon"] = pd.to_numeric(df[clon], errors="coerce").astype("float64")
    for col, cands in (
        ("name", _NAME_COLS),
        ("site_type", _TYPE_COLS),
        ("huc", _HUC_COLS),
        ("huc12", _HUC12_COLS),
        ("state", _STATE_COLS),
        ("agency", _AGENCY_COLS),
        ("coord_source", ("coord_source",)),
    ):
        c = _pick(df, cands)
        out[col] = _clean_str(df[c]) if c is not None else pd.Series(pd.NA, index=df.index, dtype="string")
    for col, cand in (("catalog_wq_begin", "wq_begin"), ("catalog_wq_end", "wq_end")):
        c = _pick(df, (cand,))
        out[col] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d") if c is not None else None
    bad = ~out["lat"].between(-90, 90) | ~out["lon"].between(-180, 180)
    out.loc[bad, ["lat", "lon"]] = np.nan
    out["catalog_source"] = source
    out = out.dropna(subset=["site_id"]).drop_duplicates("site_id", keep="first")
    return out.set_index("site_id")


# A processed site row that disagrees with the raw USGS catalogue by more than this many
# degrees (or by state) is treated as a conflict and the raw catalogue row is used instead.
COORD_CONFLICT_DEG = 0.02


def _read_standardised(s: Settings, p: Path, source: str, notes: list[str]) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        t = _standardise(pd.read_parquet(p), source)
    except Exception as e:  # noqa: BLE001 — a half-written file must not break the API
        notes.append(f"{s.rel(p)} unreadable ({type(e).__name__}); not used")
        return None
    if t is None:
        notes.append(f"{s.rel(p)} has no recognisable id/lat/lon columns; not used")
    return t


def _conflicts(table: pd.DataFrame, raw: pd.DataFrame) -> pd.Index:
    """Ids present in both whose coordinates disagree with the raw USGS catalogue.

    Coordinates decide when both tables have them (the raw catalogue's ``state`` is the state that
    was queried, which legitimately differs for border sites); state decides only otherwise.
    """
    common = table.index.intersection(raw.index)
    if common.empty:
        return common
    a, b = table.loc[common], raw.loc[common]
    dist = np.hypot(
        a["lat"].to_numpy(float) - b["lat"].to_numpy(float),
        a["lon"].to_numpy(float) - b["lon"].to_numpy(float),
    )
    have = np.isfinite(dist)
    sa, sb = a["state"].astype("string").str.upper(), b["state"].astype("string").str.upper()
    state_diff = (sa != sb).fillna(False).to_numpy(bool)
    bad = (have & (np.nan_to_num(dist, nan=0.0) > COORD_CONFLICT_DEG)) | (~have & state_diff)
    return common[bad]


def site_catalog(s: Settings) -> tuple[pd.DataFrame, dict]:
    """Site metadata (name, coordinates, type, HUC, state) indexed by site id.

    ``processed/sites/sites.parquet`` is preferred, but every site it shares with the raw USGS
    series catalogue is cross-checked; where the coordinates disagree, the raw USGS row is used
    and the site is flagged ``catalog_conflict``.
    """
    pst, pdv, piv = sites_table_path(s), dv_catalog_path(s), iv_core_path(s)
    stamp = (_stamp(pst), _stamp(pdv), _stamp(piv))

    def build():
        notes: list[str] = []
        used: list[str] = []
        table = _read_standardised(s, pst, "sites_table", notes)
        raw = _read_standardised(s, pdv, "usgs_dv_catalog", notes)
        core = _read_standardised(s, piv, "usgs_iv_core_sites", notes)
        n_conflict = 0
        frames = []
        if table is not None:
            table = table.copy()
            table["catalog_conflict"] = False
            if raw is not None:
                bad = _conflicts(table, raw)
                n_conflict = len(bad)
                if n_conflict:
                    n_common = len(table.index.intersection(raw.index))
                    fix = raw.loc[bad].copy()
                    fix["catalog_conflict"] = True
                    table = pd.concat([table.drop(index=bad), fix])
                    notes.append(
                        f"processed/sites/sites.parquet disagrees with the raw USGS series catalogue "
                        f"(coordinates, or state where coordinates are missing) for {n_conflict} of "
                        f"{n_common} shared sites; the raw USGS values are shown for those sites "
                        "(flag catalog_conflict)"
                    )
                    used.append("usgs_dv_catalog")
            frames.append(table)
            used.append("sites")
        elif not pst.exists():
            notes.append(
                "processed/sites/sites.parquet not found; coordinates and metadata come from the "
                "raw USGS series catalogue"
            )
        for t, name in ((raw, "usgs_dv_catalog"), (core, "usgs_iv_core_sites")):
            if t is None:
                continue
            have = pd.concat(frames).index if frames else pd.Index([])
            extra = t.index.difference(have)
            if len(extra) == 0:
                continue
            t = t.loc[extra].copy()
            t["catalog_conflict"] = False
            frames.append(t)
            used.append(name)
        if not frames:
            cat = pd.DataFrame(columns=CATALOG_COLS + ["catalog_conflict"]).rename_axis("site_id")
            notes.append(
                "no site catalogue found (processed/sites/sites.parquet, "
                "raw/usgs_dv/dv_series_catalog.parquet)"
            )
        else:
            cat = pd.concat(frames)
            cat = cat[~cat.index.duplicated(keep="first")]
        files = [p for p in (pst, pdv, piv) if p.exists()]
        return cat, {"datasets": _uniq(used), "files": files, "notes": notes, "n_conflict": n_conflict}

    return _cached(("catalog", str(s.data_root)), stamp, build)


# ------------------------------------------------------------------ daily tier availability


def daily_index(s: Settings) -> pd.DataFrame | None:
    p = daily_index_path(s)

    def build():
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["site_no"] = df["site_no"].astype(str)
        df = df[~df["site_no"].map(is_reserved)]          # reserved new-site block (sentinel.data.reserved)
        return df.drop_duplicates(["site_no", "var"], keep="last").reset_index(drop=True)

    return _cached(("daily_index", str(p)), _stamp(p), build)


# ------------------------------------------------------------------ hourly tier availability


def hourly_files(s: Settings) -> dict[str, Path]:
    d = hourly_sites_dir(s)
    if not d.is_dir():
        return {}
    out = {}
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".parquet") and SITE_ID_RE.match(e.name[:-8]) \
                    and not is_reserved(e.name[:-8]):     # reserved new-site block
                out[e.name[:-8]] = Path(e.path)
    return out


def _hourly_value_columns(names: Iterable[str]) -> list[str]:
    return [c for c in names if V.split_column(c, V.HOURLY_STATS)[1] is not None]


def hourly_info(path: Path) -> dict:
    """Columns and time range of one hourly table, from parquet metadata (cheap)."""

    def build():
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        names = [n for n in pf.schema_arrow.names]
        info = {
            "columns": _hourly_value_columns(names),
            "n_rows": pf.metadata.num_rows,
            "start": None,
            "end": None,
        }
        if "time_utc" in names:
            j = names.index("time_utc")
            lo = hi = None
            try:
                for i in range(pf.metadata.num_row_groups):
                    st = pf.metadata.row_group(i).column(j).statistics
                    if st is None or not st.has_min_max:
                        raise ValueError("no statistics")
                    lo = st.min if lo is None else min(lo, st.min)
                    hi = st.max if hi is None else max(hi, st.max)
            except (ValueError, TypeError):
                col = pq.read_table(path, columns=["time_utc"]).column(0).to_pandas()
                lo, hi = col.min(), col.max()
            info["start"], info["end"] = _iso_utc(lo), _iso_utc(hi)
        return info

    return _cached(("hourly_info", str(path)), _stamp(path), build)


# ------------------------------------------------------------------ sites table


def _uniq(seq: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _field(meta: dict, col: str) -> Any:
    v = meta.get(col)
    return None if v is None or pd.isna(v) else v


def sites_table(s: Settings) -> tuple[pd.DataFrame, dict]:
    """One row per known site: catalogue metadata + which variables have local data."""
    cat, cat_meta = site_catalog(s)
    idx = daily_index(s)
    hfiles = hourly_files(s)
    stamp = (
        _stamp(sites_table_path(s)),
        _stamp(dv_catalog_path(s)),
        _stamp(iv_core_path(s)),
        _stamp(daily_index_path(s)),
        tuple(sorted((k, _stamp(v)[1]) for k, v in hfiles.items())),
    )

    def build():
        rows: dict[str, dict] = {}
        if idx is not None and not idx.empty:
            for site, g in idx.groupby("site_no", sort=False):
                cols = [c for c in g["var"].tolist()]
                rows[site] = {
                    "daily_columns": cols,
                    "daily_variables": _uniq(V.split_column(c)[0] for c in cols),
                    "daily_start": _iso_date(g["first"].min()),
                    "daily_end": _iso_date(g["last"].max()),
                }
        for site, path in hfiles.items():
            try:
                info = hourly_info(path)
            except Exception:  # noqa: BLE001 — file being written; skip until readable
                continue
            r = rows.setdefault(site, {})
            r.update(
                {
                    "hourly_columns": info["columns"],
                    "hourly_variables": _uniq(V.split_column(c, V.HOURLY_STATS)[0] for c in info["columns"]),
                    "hourly_start": info["start"][:10] if info["start"] else None,
                    "hourly_end": info["end"][:10] if info["end"] else None,
                }
            )
        ids = _uniq([*cat.index.tolist(), *rows.keys()])
        cat_rec = cat.to_dict("index")
        recs = []
        for sid in ids:
            meta = cat_rec.get(sid) or {}
            r = rows.get(sid, {})
            dv, hv = r.get("daily_variables", []), r.get("hourly_variables", [])
            allv = _uniq([*dv, *hv])
            starts = [x for x in (r.get("daily_start"), r.get("hourly_start")) if x]
            ends = [x for x in (r.get("daily_end"), r.get("hourly_end")) if x]
            g = functools.partial(_field, meta)
            stype = g("site_type")
            recs.append(
                {
                    "id": sid,
                    "name": g("name"),
                    "lat": g("lat"),
                    "lon": g("lon"),
                    "site_type": stype,
                    "site_type_label": V.site_type_label(stype),
                    "huc": g("huc"),
                    "huc12": g("huc12"),
                    "state": g("state"),
                    "agency": g("agency"),
                    "coord_source": g("coord_source"),
                    "catalog_source": g("catalog_source"),
                    "catalog_conflict": bool(g("catalog_conflict") or False),
                    "catalog_wq_begin": g("catalog_wq_begin"),
                    "catalog_wq_end": g("catalog_wq_end"),
                    "daily_columns": r.get("daily_columns", []),
                    "daily_variables": dv,
                    "daily_start": r.get("daily_start"),
                    "daily_end": r.get("daily_end"),
                    "hourly_columns": r.get("hourly_columns", []),
                    "hourly_variables": hv,
                    "hourly_start": r.get("hourly_start"),
                    "hourly_end": r.get("hourly_end"),
                    "variables": allv,
                    "wq_variables": [v for v in allv if v in V.WQ_NAMES],
                    "record_start": min(starts) if starts else None,
                    "record_end": max(ends) if ends else None,
                    "has_data": bool(allv),
                }
            )
        df = pd.DataFrame(recs)
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    "id",
                    "name",
                    "lat",
                    "lon",
                    "site_type",
                    "site_type_label",
                    "huc",
                    "huc12",
                    "state",
                    "agency",
                    "coord_source",
                    "catalog_source",
                    "catalog_conflict",
                    "catalog_wq_begin",
                    "catalog_wq_end",
                    "daily_columns",
                    "daily_variables",
                    "daily_start",
                    "daily_end",
                    "hourly_columns",
                    "hourly_variables",
                    "hourly_start",
                    "hourly_end",
                    "variables",
                    "wq_variables",
                    "record_start",
                    "record_end",
                    "has_data",
                ]
            )
        df["n_wq_variables"] = df["wq_variables"].map(len) if len(df) else pd.Series(dtype=int)
        datasets = list(cat_meta["datasets"])
        notes = list(cat_meta["notes"])
        files = list(cat_meta["files"])
        if idx is not None:
            datasets.append("sonde_daily")
            files.append(daily_index_path(s))
        else:
            notes.append("daily tier index (processed/sonde_daily/index.parquet) not found")
        if hfiles:
            datasets.append("sonde_hourly")
        else:
            notes.append("hourly tier (processed/sonde_hourly/sites/) not found or empty")
        return df, {
            "datasets": datasets,
            "files": files,
            "notes": notes,
            "n_catalog_conflict": cat_meta.get("n_conflict", 0),
        }

    return _cached(("sites_table", str(s.data_root)), stamp, build)


def filter_sites(
    df: pd.DataFrame,
    *,
    var: str | None = None,
    tier: str = "any",
    huc: str | None = None,
    state: str | None = None,
    site_type: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    q: str | None = None,
    has_data: str = "true",
) -> pd.DataFrame:
    if df.empty:
        return df
    m = pd.Series(True, index=df.index)
    if has_data == "true":
        m &= df["has_data"]
    elif has_data == "false":
        m &= ~df["has_data"]
    if var:
        base = V.split_column(var)[0]
        if tier == "daily":
            m &= df["daily_columns"].map(lambda cs: var in cs) | df["daily_variables"].map(
                lambda vs: var in vs
            )
        elif tier == "hourly":
            m &= df["hourly_variables"].map(lambda vs: base in vs)
        else:
            m &= df["variables"].map(lambda vs: base in vs)
    elif tier == "daily":
        m &= df["daily_variables"].map(bool)
    elif tier == "hourly":
        m &= df["hourly_variables"].map(bool)
    if huc:
        m &= df["huc"].fillna("").astype(str).str.startswith(huc)
    if state:
        m &= df["state"].fillna("").astype(str).str.upper() == state.upper()
    if site_type:
        m &= df["site_type"].fillna("").astype(str).str.upper().str.startswith(site_type.upper())
    if bbox:
        lon0, lat0, lon1, lat1 = bbox
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lon = pd.to_numeric(df["lon"], errors="coerce")
        m &= lat.between(lat0, lat1) & lon.between(lon0, lon1)
    if q:
        m &= df["name"].fillna("").astype(str).str.contains(q, case=False, regex=False) | df["id"].astype(
            str
        ).str.contains(q, regex=False)
    return df[m]


def site_record(s: Settings, site_id: str) -> dict:
    if is_reserved(site_id):
        raise DataUnavailable(f"site {site_id} is reserved (held-out evaluation block)")
    df, meta = sites_table(s)
    hit = df[df["id"] == site_id] if not df.empty else df
    if hit.empty:
        raise DataUnavailable(f"site {site_id} is not in the local catalogue or data products")
    return {"site": hit.iloc[0].to_dict(), "meta": meta}


def daily_coverage(s: Settings, site_id: str) -> list[dict]:
    idx = daily_index(s)
    if idx is None:
        return []
    g = idx[idx["site_no"] == site_id]
    out = []
    for _, r in g.iterrows():
        d = V.describe(r["var"])
        d.update(
            {
                "n_days": int(r["n_days"]),
                "first": _iso_date(r["first"]),
                "last": _iso_date(r["last"]),
                "frac_provisional": None if pd.isna(r["frac_provisional"]) else float(r["frac_provisional"]),
            }
        )
        out.append(d)
    return out


# ------------------------------------------------------------------ observations


@dataclass
class Observations:
    site_id: str
    tier: str
    variables: list[dict]
    rows: pd.DataFrame
    total_rows: int
    files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _f32_clean(x: np.ndarray) -> np.ndarray:
    """float32 -> float64 at float32's own decimal precision (7.9f32 -> 7.9, not 7.9000001)."""
    x = np.asarray(x)
    if x.dtype == np.float32:
        return x.astype(str).astype(np.float64)
    return x.astype(np.float64)


def _parse_time(x: str | None, *, utc: bool) -> pd.Timestamp | None:
    if x is None or str(x).strip() == "":
        return None
    try:
        t = pd.Timestamp(str(x).strip())
    except (ValueError, TypeError) as e:
        raise BadRequest(f"unparseable date/time {x!r}") from e
    if utc:
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_localize(None) if t.tzinfo is not None else t


def _resolve_vars(requested: list[str] | None, available: list[str], stats: tuple[str, ...]) -> list[str]:
    if not requested:
        return list(available)
    out = []
    for v in requested:
        v = v.strip()
        if not v:
            continue
        if v in available:
            out.append(v)
            continue
        expanded = [c for c in available if V.split_column(c, stats)[0] == v]
        if not expanded:
            raise BadRequest(f"variable {v!r} has no data at this site in this tier; available: {available}")
        out.extend(expanded)
    return _uniq(out)


def observations(
    s: Settings,
    site_id: str,
    *,
    tier: str = "daily",
    variables: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Observations:
    if tier not in TIERS:
        raise BadRequest(f"tier must be one of {TIERS}")
    if is_reserved(site_id):                           # reserved new-site block: never served
        raise DataUnavailable(f"site {site_id} is reserved (held-out evaluation block)")
    if tier == "daily":
        path = daily_sites_dir(s) / f"{site_id}.parquet"
        if not path.exists():
            raise DataUnavailable(
                f"no daily table for site {site_id} ({s.rel(path)}); the daily tier may "
                "not include this site or may still be building"
            )
        import pyarrow.parquet as pq

        names = pq.ParquetFile(path).schema_arrow.names
        avail = [c for c in names if V.split_column(c)[1] is not None]
        cols = _resolve_vars(variables, avail, V.DAILY_STATS)
        read = (
            cols
            + [f"{c}_prov" for c in cols if f"{c}_prov" in names]
            + [f"{c}_ts" for c in cols if f"{c}_ts" in names]
        )
        df = pd.read_parquet(path, columns=read)
        idx = pd.DatetimeIndex(df.index)
        t0, t1 = _parse_time(start, utc=False), _parse_time(end, utc=False)
        keep = np.ones(len(df), dtype=bool)
        if t0 is not None:
            keep &= idx >= t0.normalize()
        if t1 is not None:
            keep &= idx <= t1
        df = df[keep]
        frames, meta = [], []
        for c in cols:
            vals = df[c].to_numpy()
            ok = np.isfinite(vals.astype("float64"))
            prov = df[f"{c}_prov"].to_numpy()[ok] if f"{c}_prov" in df else np.zeros(ok.sum(), bool)
            ts = df[f"{c}_ts"].astype(str).to_numpy()[ok] if f"{c}_ts" in df else np.full(ok.sum(), None)
            fr = pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(df.index[ok]).strftime("%Y-%m-%d"),
                    "variable": c,
                    "value": _f32_clean(vals[ok]),
                    "provisional": prov.astype(bool),
                    "ts_id": ts,
                }
            )
            frames.append(fr)
            d = V.describe(c)
            d.update({"n": int(ok.sum()), "n_provisional": int(prov.sum())})
            meta.append(d)
        rows = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["date", "variable", "value", "provisional", "ts_id"])
        )
        rows = rows.sort_values(["date", "variable"], kind="stable").reset_index(drop=True)
        return Observations(site_id, tier, meta, rows, len(rows), [path])

    path = hourly_sites_dir(s) / f"{site_id}.parquet"
    if not path.exists():
        raise DataUnavailable(
            f"no hourly table for site {site_id} ({s.rel(path)}); the hourly tier covers "
            "only sites whose 15-min download is complete"
        )
    info = hourly_info(path)
    avail = info["columns"]
    cols = _resolve_vars(variables, avail, V.HOURLY_STATS)
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    bases = _uniq(V.split_column(c, V.HOURLY_STATS)[0] for c in cols)
    extra = [f"{b}_prov" for b in bases if f"{b}_prov" in names] + [
        f"{b}_n" for b in bases if f"{b}_n" in names
    ]
    df = pd.read_parquet(path, columns=cols + extra)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    t0, t1 = _parse_time(start, utc=True), _parse_time(end, utc=True)
    keep = np.ones(len(df), dtype=bool)
    if t0 is not None:
        keep &= idx >= t0
    if t1 is not None:
        if end and len(str(end).strip()) <= 10:  # a bare date includes that whole day
            t1 = t1 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        keep &= idx <= t1
    df, idx = df[keep], idx[keep]
    frames, meta = [], []
    tstr = idx.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in cols:
        b = V.split_column(c, V.HOURLY_STATS)[0]
        vals = df[c].to_numpy()
        ok = np.isfinite(vals.astype("float64"))
        prov = df[f"{b}_prov"].to_numpy()[ok] if f"{b}_prov" in df else np.zeros(ok.sum(), bool)
        nraw = df[f"{b}_n"].to_numpy()[ok].astype(int) if f"{b}_n" in df else np.full(ok.sum(), -1)
        frames.append(
            pd.DataFrame(
                {
                    "time_utc": tstr[ok],
                    "variable": c,
                    "value": _f32_clean(vals[ok]),
                    "provisional": prov.astype(bool),
                    "n_raw": nraw,
                }
            )
        )
        d = V.describe_hourly(c)
        d.update({"n": int(ok.sum()), "n_provisional": int(prov.sum())})
        meta.append(d)
    rows = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["time_utc", "variable", "value", "provisional", "n_raw"])
    )
    rows = rows.sort_values(["time_utc", "variable"], kind="stable").reset_index(drop=True)
    return Observations(
        site_id,
        tier,
        meta,
        rows,
        len(rows),
        [path],
        notes=[
            "hourly values are UTC-hour aggregates (mean/min/max) of 15-min USGS values; "
            "n_raw = number of 15-min values in the hour; an hour is provisional if any "
            "of its 15-min values was provisional"
        ],
    )


def data_status(s: Settings) -> dict:
    """What local products exist (for /health and the Home page). Counts are computed live."""
    idx = daily_index(s)
    hfiles = hourly_files(s)
    return {
        "data_root": str(s.data_root),
        "sites_table": sites_table_path(s).exists(),
        "usgs_dv_catalog": dv_catalog_path(s).exists(),
        "daily_index": idx is not None,
        "daily_sites": int(idx["site_no"].nunique()) if idx is not None else 0,
        "daily_site_files": sum(1 for _ in daily_sites_dir(s).glob("*.parquet"))
        if daily_sites_dir(s).is_dir()
        else 0,
        "hourly_sites": len(hfiles),
    }
