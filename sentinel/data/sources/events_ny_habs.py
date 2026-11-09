# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""New York State DEC harmful algal blooms (HABs) → tidy secondary bloom labels.

Sources (verified 2026-10-22; see docs/datasheets/labels_ny_habs.md):
  * data.ny.gov ``qtg2-hjth`` "Harmful Algal Bloom Statewide Occurrence Summary: 2012-2018"
    — one row per waterbody × county × year with the MAXIMUM bloom status of the year
    (S suspicious, C confirmed, HT confirmed with high toxins), first/last date on the DEC
    notification list and weeks listed.
  * data.ny.gov ``95my-wijm`` "Harmful Algal Blooms by Waterbody Summary: Beginning 2019"
    — one row per waterbody-year, ``bloom_type`` = "confirmed" only, first/last NYHABS report
    date and number of reports.
  * NYHABS public ArcGIS views (DEC-owned, anonymous): per-report CONFIRMED blooms of the
    CURRENT SEASON only ("Archived" = older than 14 days, "Current" = last 14 days; status
    changes from the last 27 h are hidden). Earlier seasons' per-report records are not public;
    the downloader therefore keeps dated snapshots so repeated runs build an archive.

Status definitions differ by era: in 2012–2018 "C" required water-sampling confirmation, from
2019 "confirmed" may rest on visual observations / photos OR sampling. ``status_basis`` records
this. Rows are lake centroids / report points; non-listed waterbody-years are NOT negatives.
Licence: OPEN-NY Terms of Use (data.ny.gov; no attribution or share-alike requirement).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import events_common as C

DOMAIN = "data.ny.gov"
DATASET_2012_2018 = "qtg2-hjth"
DATASET_2019_ON = "95my-wijm"
DATASETS = {DATASET_2012_2018: "2012-2018", DATASET_2019_ON: "2019+"}
LICENSE = "OPEN-NY Terms of Use (data.ny.gov, 2013): free reuse for lawful purposes; no attribution required"

_ARC = "https://services6.arcgis.com/DZHaqZm9cxOD4CWM/arcgis/rest/services"
NYHABS_LAYERS = {
    # view name: (layer url, ArcGIS Online item id)
    "archived": (f"{_ARC}/Archived_HAB_Reports/FeatureServer/0", "41a2a043386444f688b369f9cff6dbac"),
    "archived_dil": (f"{_ARC}/Archived_HAB_Reports_DIL/FeatureServer/0", "19b4069e70364ba38b2b2b6656cc56d7"),
    "current": (f"{_ARC}/Current_HAB_Reports_7_3_19/FeatureServer/0", "302fac4b1ef941898d3a9ea9eb7a8d1d"),
    "current_dil": (f"{_ARC}/Current_HAB_Reports_DIL/FeatureServer/0", "9492e4e687654caa982e7ed5fe4b4a68"),
}
NYHABS_APP = "https://nysdec.maps.arcgis.com/apps/webappviewer/index.html?id=ae91142c812a4ab997ba739ed9723e6e"

NY_BBOX = (-80.0, 40.4, -71.7, 45.1)
LOCAL_TZ = "America/New_York"
MIN_DATE = "2012-01-01"

STATUS_ORDER = ["suspicious", "confirmed", "confirmed_high_toxin"]
STATUS_DTYPE = pd.CategoricalDtype(STATUS_ORDER, ordered=True)
_STATUS = {
    "s": "suspicious", "suspicious": "suspicious", "suspicious bloom": "suspicious",
    "c": "confirmed", "confirmed": "confirmed", "confirmed bloom": "confirmed",
    "ht": "confirmed_high_toxin", "confirmed_with_high_toxins": "confirmed_high_toxin",
    "confirmed with high toxins": "confirmed_high_toxin", "confirmed with high toxins bloom": "confirmed_high_toxin",
}
EXTENT_CLASS = {"small localized": "small_localized", "large localized": "large_localized",
                "widespread/lakewide": "widespread_lakewide", "widespread or lakewide": "widespread_lakewide",
                "open water": "open_water", "not reported": "not_reported"}
EXTENT_RANK = {"small_localized": 1, "large_localized": 2, "widespread_lakewide": 3}


# ----------------------------------------------------------------------------- helpers

def _txt(values) -> pd.Series:
    s = pd.Series(values, dtype="object")
    return s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())


def _na(values) -> pd.Series:
    s = _txt(values)
    return s.where(s != "").astype("string")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series("", index=df.index, dtype="object")


def read_socrata_csv(src: str | Path | bytes) -> pd.DataFrame:
    """Socrata CSV export (``$select=*,:id``) → all-string DataFrame."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    return pd.read_csv(src, dtype=str, keep_default_na=False)


def name_key(names: pd.Series | Iterable) -> pd.Series:
    """Waterbody name → matching key (lower-case, punctuation removed, 'st.' → 'st')."""
    t = _txt(names).str.lower().str.replace(r"[’'`]", "", regex=True)
    t = t.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return t.astype("string")


def bloom_status(codes: pd.Series | Iterable) -> pd.Series:
    """S/C/HT (2012–18), 'confirmed' (2019+), NYHABS HAB_STATUS → ordered status category."""
    t = _txt(codes).str.lower().str.replace(r"\s+", " ", regex=True)
    mapped = t.map(_STATUS)
    unknown = sorted(set(t[(t != "") & mapped.isna()]))
    if unknown:
        raise ValueError(f"unknown NY HAB status codes: {unknown}")
    return pd.Series(pd.Categorical(mapped, dtype=STATUS_DTYPE), index=t.index)


def status_rank(status: pd.Series) -> pd.Series:
    """suspicious 1 < confirmed 2 < confirmed_high_toxin 3."""
    return pd.Series(status).cat.codes.astype("Int8").replace(-1, pd.NA) + 1


def data_columns(df: pd.DataFrame) -> list[str]:
    """Published data columns (Socrata system/computed and geocode columns excluded)."""
    return [c for c in df.columns if not c.startswith(":") and c not in ("geocoded_column", "georeference")]


def record_id(dataset: str, df: pd.DataFrame, cols: Sequence[str] | None = None) -> tuple[pd.Series, pd.Series]:
    """Content hash of all published data fields (stable across Socrata re-uploads).

    Exact duplicate rows (3 in qtg2-hjth on 2026-10-22) get an occurrence suffix ``#2`` …;
    returns (record_id, is_exact_duplicate) where the flag is True for 2nd+ occurrences.
    """
    cols = list(cols) if cols is not None else data_columns(df)
    keys = df[cols].astype(str).agg("\x1f".join, axis=1)
    occ = keys.groupby(keys).cumcount()
    base = keys.map(lambda k: f"{dataset}:" + hashlib.sha1(f"{dataset}\x1f{k}".encode()).hexdigest()[:12])
    rid = base.where(occ == 0, base + "#" + (occ + 1).astype(str))
    return rid.astype("string"), (occ > 0)


def _dates(values, *, max_date: str | None, year: pd.Series | None = None):
    d, f = C.clean_dates(_txt(values), min_date=MIN_DATE, max_date=max_date)
    d = d.dt.normalize()
    if year is not None:
        mism = d.notna() & year.notna() & (d.dt.year != year.astype("float"))
        f = f.where(~mism, "year_mismatch")
    return d, f.astype("string")


def _coords(lat, lon):
    la, lo, fl = C.clean_latlon(_txt(lat).replace("", np.nan), _txt(lon).replace("", np.nan), bbox=NY_BBOX)
    return la, lo, fl.astype("string")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: v is None or isinstance(v, str)
                                                or (isinstance(v, float) and np.isnan(v))).all():
            df[c] = df[c].astype("string")
    return df


# ----------------------------------------------------------------------------- waterbody-year summaries

_COMMON = ["source", "dataset", "era", "record_id", "socrata_row_id", "waterbody_name", "waterbody_key",
           "is_exact_duplicate", "county", "n_counties", "dec_region", "year", "bloom_status_code",
           "bloom_status", "status_rank",
           "status_basis", "label_confirmed", "label_high_toxin", "first_report_date", "first_report_flag",
           "last_report_date", "last_report_flag", "date_basis", "n_reports", "n_weeks_listed",
           "n_weeks_with_updates", "latitude", "longitude", "coord_flag"]


def tidy_summary_2012_2018(raw: pd.DataFrame, *, max_date: str | None = None) -> pd.DataFrame:
    df = raw.copy()
    year = pd.to_numeric(_txt(_col(df, "year")), errors="coerce").astype("Int16")
    st = bloom_status(_col(df, "bloom_type"))
    first, ff = _dates(_col(df, "date_of_first_listing"), max_date=max_date, year=year)
    last, lf = _dates(_col(df, "date_of_last_listing"), max_date=max_date, year=year)
    lat, lon, cf = _coords(_col(df, "latitude"), _col(df, "longitude"))
    county = _na(_col(df, "county"))
    rid, dup = record_id(DATASET_2012_2018, df)
    out = pd.DataFrame({
        "source": "ny_dec_habs", "dataset": DATASET_2012_2018, "era": "2012-2018",
        "record_id": rid, "is_exact_duplicate": dup.to_numpy(),
        "socrata_row_id": _na(_col(df, ":id")),
        "waterbody_name": _na(_col(df, "waterbody_name")),
        "waterbody_key": name_key(_col(df, "waterbody_name")),
        "county": county,
        "n_counties": county.fillna("").str.count(",").add(1).where(county.notna()).astype("Int8"),
        "dec_region": C.str_id(_col(df, "dec_region")),
        "year": year,
        "bloom_status_code": _na(_col(df, "bloom_type")),
        "bloom_status": st,
        "status_rank": status_rank(st),
        # 2012-18 data dictionary: C = water sampling confirmed a cyanobacteria HAB; S = visual/photo
        "status_basis": pd.Series(np.where(st.astype(str) == "suspicious", "visual_or_photo",
                                           "water_sampling"), index=df.index),
        "label_confirmed": st.isin(["confirmed", "confirmed_high_toxin"]).astype("boolean"),
        "label_high_toxin": (st == "confirmed_high_toxin").astype("boolean"),
        "first_report_date": first, "first_report_flag": ff,
        "last_report_date": last, "last_report_flag": lf,
        "date_basis": "dec_notification_listing",
        "n_reports": pd.array([pd.NA] * len(df), dtype="Int32"),
        "n_weeks_listed": pd.to_numeric(_txt(_col(df, "number_of_weeks_on_dec_notification_list")),
                                        errors="coerce").astype("Int16"),
        "n_weeks_with_updates": pd.to_numeric(_txt(_col(df, "number_of_weeks_with_updates")),
                                              errors="coerce").astype("Int16"),
        "latitude": lat, "longitude": lon, "coord_flag": cf,
    }, index=df.index)
    return _finish(out[_COMMON])


def tidy_summary_2019(raw: pd.DataFrame, *, max_date: str | None = None) -> pd.DataFrame:
    df = raw.copy()
    year = pd.to_numeric(_txt(_col(df, "year")), errors="coerce").astype("Int16")
    st = bloom_status(_col(df, "bloom_type"))
    first, ff = _dates(_col(df, "date_first_report"), max_date=max_date, year=year)
    last, lf = _dates(_col(df, "date_last_report"), max_date=max_date, year=year)
    lat, lon, cf = _coords(_col(df, "latitude"), _col(df, "longitude"))
    county = _na(_col(df, "county"))
    rid, dup = record_id(DATASET_2019_ON, df)
    out = pd.DataFrame({
        "source": "ny_dec_habs", "dataset": DATASET_2019_ON, "era": "2019+",
        "record_id": rid, "is_exact_duplicate": dup.to_numpy(),
        "socrata_row_id": _na(_col(df, ":id")),
        "waterbody_name": _na(_col(df, "waterbody_name")),
        "waterbody_key": name_key(_col(df, "waterbody_name")),
        "county": county,
        "n_counties": county.fillna("").str.count(",").add(1).where(county.notna()).astype("Int8"),
        "dec_region": pd.array([pd.NA] * len(df), dtype="string"),
        "year": year,
        "bloom_status_code": _na(_col(df, "bloom_type")),
        "bloom_status": st,
        "status_rank": status_rank(st),
        # 2019+ data dictionary: confirmed = visual observations/photos OR water sampling
        "status_basis": "visual_photo_or_sampling",
        "label_confirmed": st.isin(["confirmed", "confirmed_high_toxin"]).astype("boolean"),
        # the 2019+ summary does not distinguish high-toxin blooms
        "label_high_toxin": pd.array([pd.NA] * len(df), dtype="boolean"),
        "first_report_date": first, "first_report_flag": ff,
        "last_report_date": last, "last_report_flag": lf,
        "date_basis": "nyhabs_report",
        "n_reports": pd.to_numeric(_txt(_col(df, "number_of_reports")), errors="coerce").astype("Int32"),
        "n_weeks_listed": pd.array([pd.NA] * len(df), dtype="Int16"),
        "n_weeks_with_updates": pd.array([pd.NA] * len(df), dtype="Int16"),
        "latitude": lat, "longitude": lon, "coord_flag": cf,
    }, index=df.index)
    return _finish(out[_COMMON])


def waterbody_years(raw_2012_2018: pd.DataFrame, raw_2019: pd.DataFrame, *,
                    max_date: str | None = None) -> pd.DataFrame:
    """Both eras at the published grain (2012–18: waterbody × county × year)."""
    a = tidy_summary_2012_2018(raw_2012_2018, max_date=max_date)
    b = tidy_summary_2019(raw_2019, max_date=max_date)
    out = pd.concat([a, b], ignore_index=True)
    out["bloom_status"] = out["bloom_status"].astype(STATUS_DTYPE)
    return out.sort_values(["year", "waterbody_key", "county"], na_position="last",
                           kind="mergesort").reset_index(drop=True)


def lake_years(wy: pd.DataFrame, *, coord_decimals: int = 3) -> pd.DataFrame:
    """Collapse published rows to one row per lake-year.

    Key = (era, waterbody_key, year, lat/lon rounded to ``coord_decimals`` ≈ 100 m). 2012–18
    multi-county lakes (e.g. Canandaigua: Ontario + Yates) are listed once per county, sometimes
    with different dates/status; the collapsed row takes the max status, earliest first date,
    latest last date, max weeks listed and joins the counties. Rows without coordinates keep
    their own key. Exact duplicate published rows are dropped first.
    """
    df = wy[~wy["is_exact_duplicate"].astype(bool)].copy()
    lat_r = df["latitude"].round(coord_decimals)
    lon_r = df["longitude"].round(coord_decimals)
    df["_k"] = (df["era"].astype(str) + "|" + df["waterbody_key"].astype(str) + "|" + df["year"].astype(str)
                + "|" + lat_r.astype(str) + "|" + lon_r.astype(str))
    nocoord = df["latitude"].isna()
    df.loc[nocoord, "_k"] = df.loc[nocoord, "_k"] + "|" + df.loc[nocoord, "record_id"].astype(str)

    def _join(s):
        vals = sorted({x.strip() for v in s.dropna() for x in str(v).split(",") if x.strip()})
        return ", ".join(vals) if vals else pd.NA

    g = df.groupby("_k", sort=False)
    out = g.agg(
        source=("source", "first"), dataset=("dataset", "first"), era=("era", "first"),
        waterbody_name=("waterbody_name", "first"), waterbody_key=("waterbody_key", "first"),
        year=("year", "first"), status_rank=("status_rank", "max"),
        status_basis=("status_basis", "first"),
        first_report_date=("first_report_date", "min"), last_report_date=("last_report_date", "max"),
        date_basis=("date_basis", "first"),
        n_weeks_listed=("n_weeks_listed", "max"), latitude=("latitude", "first"),
        longitude=("longitude", "first"), n_source_records=("record_id", "size"),
    )
    out["counties"] = g["county"].agg(_join)
    out["dec_regions"] = g["dec_region"].agg(_join)
    out["record_ids"] = g["record_id"].agg(lambda s: ";".join(sorted(s.astype(str))))
    nrep = g["n_reports"].agg(lambda s: s.sum() if s.notna().any() else pd.NA)
    out["n_reports"] = pd.array(nrep.tolist(), dtype="Int32")
    rank = out["status_rank"].astype("Int8")
    out["status_rank"] = rank
    out["bloom_status"] = pd.Categorical([STATUS_ORDER[int(r) - 1] if pd.notna(r) else None for r in rank],
                                         dtype=STATUS_DTYPE)
    out["label_confirmed"] = (rank >= 2).astype("boolean")
    hi = (rank == 3).astype("boolean")
    out["label_high_toxin"] = hi.where(out["era"] == "2012-2018", pd.NA).astype("boolean")
    out["lake_year_id"] = ["nyhab_ly:" + hashlib.sha1(k.encode()).hexdigest()[:12] for k in out.index]
    cols = ["source", "lake_year_id", "dataset", "era", "waterbody_name", "waterbody_key", "counties",
            "dec_regions", "year", "bloom_status", "status_rank", "status_basis", "label_confirmed",
            "label_high_toxin", "first_report_date", "last_report_date", "date_basis", "n_reports",
            "n_weeks_listed", "latitude", "longitude", "n_source_records", "record_ids"]
    out = _finish(out.reset_index(drop=True)[cols])
    return out.sort_values(["year", "waterbody_key"], kind="mergesort").reset_index(drop=True)


# ----------------------------------------------------------------------------- NYHABS per-report views

def nyhabs_frame(features: Sequence[dict], view: str) -> pd.DataFrame:
    """Esri JSON features of one NYHABS view → tidy per-report rows.

    ArcGIS date fields are epoch milliseconds in UTC (layer ``dateFieldsTimeReference`` = UTC,
    verified 2026-10-22); ``date`` is the local (America/New_York) calendar date of observation.
    """
    a = C.esri_attributes_frame(features)
    if a.empty:
        return pd.DataFrame()
    t_obs = pd.to_datetime(pd.to_numeric(_col(a, "date_time"), errors="coerce"), unit="ms", utc=True)
    t_st = pd.to_datetime(pd.to_numeric(_col(a, "STATUS_DATE"), errors="coerce"), unit="ms", utc=True)
    lat, lon, cf = _coords(_col(a, "_y"), _col(a, "_x"))
    ext = _txt(_col(a, "extent_bloom")).str.lower().map(EXTENT_CLASS)
    gid = _txt(_col(a, "globalid")).str.strip("{}").str.lower()
    out = pd.DataFrame({
        "source": "ny_dec_nyhabs",
        "globalid": gid.where(gid != "").astype("string"),
        "objectid": C.str_id(_col(a, "objectid")),
        "view": view,
        "report_time_utc": t_obs,
        "date": t_obs.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize(),
        "status_time_utc": t_st,
        "county": _na(_col(a, "county")),
        "data_provider": _na(_col(a, "data_provider")),
        "waterbody_name": _na(_col(a, "water_name")),
        "waterbody_key": name_key(_col(a, "water_name")),
        "lake_sections": _na(_col(a, "lake_sections")),
        "extent_bloom": _na(_col(a, "extent_bloom")),
        "extent_class": ext.astype("string"),
        "extent_rank": ext.map(EXTENT_RANK).astype("Int8"),
        "hab_status": _na(_col(a, "HAB_STATUS")),
        "bloom_status": bloom_status(_col(a, "HAB_STATUS")),
        "latitude": lat, "longitude": lon, "coord_flag": cf,
    })
    return _finish(out)


def merge_nyhabs_views(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge the four views on ``globalid``: lake_sections comes from the non-DIL views and
    status_time from the DIL views; ``views`` lists every view a report was seen in."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    allv = pd.concat(frames, ignore_index=True)
    allv = allv[allv["globalid"].notna()]
    views = allv.groupby("globalid")["view"].agg(lambda s: ";".join(sorted(set(s))))
    # first non-null per column within each globalid
    merged = allv.groupby("globalid", sort=False).first()
    merged["views"] = views
    merged["window"] = np.where(merged["views"].str.contains("current"), "current_14d", "archived")
    merged = merged.drop(columns=["view"]).reset_index()
    merged["bloom_status"] = merged["bloom_status"].astype(STATUS_DTYPE)
    return _finish(merged.sort_values(["report_time_utc", "globalid"], kind="mergesort").reset_index(drop=True))


def accumulate_snapshots(snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{snapshot_date: merged view frame} → one row per globalid across snapshots.

    Values come from the LATEST snapshot containing the report (statuses can be edited);
    ``first_seen`` / ``last_seen`` record the snapshot dates.
    """
    parts = []
    for snap, f in sorted(snapshots.items()):
        if f is None or f.empty:
            continue
        g = f.copy()
        g["snapshot_date"] = snap
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    allv = pd.concat(parts, ignore_index=True)
    seen = allv.groupby("globalid")["snapshot_date"].agg(["min", "max", "size"])
    latest = allv.sort_values("snapshot_date", kind="mergesort").drop_duplicates("globalid", keep="last")
    latest = latest.set_index("globalid")
    latest["first_seen"] = seen["min"]
    latest["last_seen"] = seen["max"]
    latest["n_snapshots"] = seen["size"].astype("Int16")
    latest = latest.drop(columns=["snapshot_date"]).reset_index()
    latest["bloom_status"] = latest["bloom_status"].astype(STATUS_DTYPE)
    return _finish(latest.sort_values(["report_time_utc", "globalid"], kind="mergesort").reset_index(drop=True))
