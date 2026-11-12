#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the shared ``processed/ea_wq_sonde`` dataset (EA England water-quality sondes; OGL v3) for the M6
new-geography track (``docs/plans/ea_england_allocation.md``; logic ``sentinel/hydromet_sota/ea.py``).

    python -B scripts/data/sota_hydromet_ea_build.py catalogue   # stations + measures (temperature, dissolved-oxygen)
    python -B scripts/data/sota_hydromet_ea_build.py readings    # monthly cross-station CSVs 2012-01 .. 2023-09 (gzip)
    python -B scripts/data/sota_hydromet_ea_build.py daily       # QC + daily statistics
    python -B scripts/data/sota_hydromet_ea_build.py manifest    # manifest, content pins
    python -B scripts/data/sota_hydromet_ea_build.py geo         # HydroRIVERS snapping, flow connections, basin groups
    python -B scripts/data/sota_hydromet_ea_build.py atlas       # RiverATLAS regional subsets (GB reaches; US sites)
    python -B scripts/data/sota_hydromet_ea_build.py strata      # near-discharge stratum (UWWTR Art. 15 table)
    python -B scripts/data/sota_hydromet_ea_build.py preview     # split + per-endpoint G / G_eff from metadata (prints; writes nothing)
    python -B scripts/data/sota_hydromet_ea_build.py split --i-have-the-lead-ruling   # LEAD ONLY, once

Reservation: nothing dated on or after 2023-10-01 is requested or stored (``max-date`` is exclusive and every parsed
row is checked). One request at a time, gzip, ≥ 2 s between requests (EA fair use). Raw monthly files are kept
compressed under ``raw/ea_hydrology`` for provenance (≈ 0.3 GB).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.hydromet_sota import ea as EA
from sentinel.hydromet_sota import ea_split as ES

RAW = paths.raw("ea_hydrology")
OUT = paths.processed("ea_wq_sonde")
UA = {"User-Agent": "SENTINEL-v2 research (public-data download; contact via project)", "Accept-Encoding": "gzip"}
log = logging.getLogger("ea_wq_sonde")


def _get(url, params=None, *, timeout=300, tries=5):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code} {r.text[:200]}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(20 * (k + 1))
    raise RuntimeError(f"{url} {params}: {last}")


def step_catalogue() -> None:
    cdir = RAW / "catalogue"
    cdir.mkdir(parents=True, exist_ok=True)
    for kind in ("stations", "measures"):
        for p in EA.PROPS:
            r = _get(f"{EA.BASE}/id/{kind}.json", {"observedProperty": p, "_limit": 10000})
            body = r.json()
            body["_retrieved_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
            (cdir / f"{kind}_{p}.json").write_text(json.dumps(body))
            log.info("%s %s: %d items", kind, p, len(body.get("items", [])))
            time.sleep(2)
    rows = []
    for p in EA.PROPS:
        for it in json.loads((cdir / f"stations_{p}.json").read_text())["items"]:
            st = it.get("status")
            st = st[0].get("label") if isinstance(st, list) and st else (st.get("label") if isinstance(st, dict) else st)
            rows.append({"station": it.get("notation"), "label": it.get("label"), "lat": it.get("lat"), "lon": it.get("long"),
                         "easting": it.get("easting"), "northing": it.get("northing"), "date_opened": it.get("dateOpened"),
                         "date_closed": it.get("dateClosed"), "status": st, "river_name": it.get("riverName"),
                         "catchment_area_km2": it.get("catchmentArea"), "nrfa_station": it.get("nrfaStationID"),
                         "property": p})
    st = pd.DataFrame(rows)
    st = st.groupby("station", as_index=False).agg({**{c: "first" for c in st.columns if c not in ("station", "property")},
                                                    "property": lambda s: ",".join(sorted(set(s)))})
    OUT.mkdir(parents=True, exist_ok=True)
    st.to_parquet(OUT / "stations.parquet", index=False)
    mrows = []
    for p in EA.PROPS:
        for it in json.loads((cdir / f"measures_{p}.json").read_text())["items"]:
            stn = it.get("station") or {}
            mrows.append({"measure": it.get("notation"), "station": (stn.get("@id") or "").rsplit("/", 1)[-1],
                          "property": p, "var": EA.measure_var(it.get("notation", "")), "unit": it.get("unitName"),
                          "period_s": it.get("period"), "value_type": it.get("valueType"),
                          "observation_type": (it.get("observationType") or {}).get("label")})
    pd.DataFrame(mrows).to_parquet(OUT / "measures.parquet", index=False)
    log.info("catalogue: %d stations, %d measures", len(st), len(mrows))


def step_readings() -> None:
    for p in EA.PROPS:
        d = RAW / "readings" / p
        d.mkdir(parents=True, exist_ok=True)
        for m0 in EA.month_starts():
            f = d / f"{m0:%Y-%m}.csv.gz"
            if f.exists():
                continue
            url, params = EA.readings_url(p, m0)
            t0 = time.time()
            r = _get(url, params)
            raw = r.content
            df = EA.parse_readings(raw)          # refuses any reserved-period row
            if len(df) >= params["_limit"] - 1:
                raise RuntimeError(f"{p} {m0:%Y-%m}: row limit reached; split the month")
            tmp = f.with_suffix(".tmp")
            with gzip.open(tmp, "wb") as fh:
                fh.write(raw)
            tmp.replace(f)
            log.info("%s %s: %d rows, %d measures, %.1f MB raw, %.0f s", p, f"{m0:%Y-%m}", len(df), df["measure"].nunique(),
                     len(raw) / 1e6, time.time() - t0)
            time.sleep(2)


def step_daily() -> None:
    meas = pd.read_parquet(OUT / "measures.parquet").set_index("measure")
    parts, qc = [], []
    for p in EA.PROPS:
        files = sorted((RAW / "readings" / p).glob("*.csv.gz"))
        df = pd.concat([EA.parse_readings(gzip.open(f).read()) for f in files], ignore_index=True)
        df["var"] = df["measure"].map(lambda m: EA.measure_var(m))
        df = df[df["var"].notna()]
        log.info("%s: %d readings in %d measures", p, len(df), df["measure"].nunique())
        for (m, var), g in df.groupby(["measure", "var"], sort=False):
            if meas.reindex([m])["period_s"].iloc[0] == 86400:    # daily qualified series of flow stations: separate
                continue
            kept, info = EA.clean_measure(g, var)
            dd = EA.daily_stats(kept, var)
            qc.append({"measure": m, "var": var, **info, "n_days": int(len(dd)),
                       "n_valid_days": int(dd["valid"].sum()) if len(dd) else 0})
            if len(dd):
                dd.insert(0, "var", var)
                dd.insert(0, "measure", m)
                parts.append(dd)
        del df
    daily = pd.concat(parts, ignore_index=True)
    daily["station"] = daily["measure"].map(meas["station"])
    if (daily["date"] >= EA.RESERVED_FROM).any():
        raise RuntimeError("reserved-period day in the daily table")
    daily.to_parquet(OUT / "daily_stats.parquet", index=False)
    pd.DataFrame(qc).to_parquet(OUT / "qc_measures.parquet", index=False)
    # wide station-day targets (valid days only), names as the M6 / M1 daily tiers
    v = daily[daily["valid"]]
    wide = []
    for var, cols in (("temp_c", {"mean": "temp_c_mean", "max": "temp_c_max", "min": "temp_c_min"}),
                      ("do_mgl", {"mean": "do_mgl_mean", "min": "do_mgl_min", "max": "do_mgl_max"}),
                      ("do_pct", {"mean": "do_pct_mean"})):
        x = v[v["var"] == var].groupby(["station", "date"])[list(cols)].mean().rename(columns=cols)
        wide.append(x)
    w = pd.concat(wide, axis=1).reset_index()
    w.to_parquet(OUT / "station_daily.parquet", index=False)
    log.info("daily: %d measure-days (%d valid); station-day table %d rows, %d stations", len(daily),
             int(daily["valid"].sum()), len(w), w["station"].nunique())


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def step_manifest() -> None:
    files = [OUT / n for n in ("stations.parquet", "measures.parquet", "daily_stats.parquet", "qc_measures.parquet",
                               "station_daily.parquet")]
    raw_files = sorted((RAW / "readings").rglob("*.csv.gz"))
    pins = {"processed": {f.name: _sha(f) for f in files},
            "raw_readings_sha256_of_list": hashlib.sha256("\n".join(f"{f.relative_to(RAW)} {_sha(f)}"
                                                                     for f in raw_files).encode()).hexdigest(),
            "n_raw_files": len(raw_files)}
    (OUT / "CONTENT_PINS.json").write_text(json.dumps(pins, indent=1))
    w = pd.read_parquet(OUT / "station_daily.parquet")
    stats = {"stations_with_valid_days": int(w["station"].nunique()), "station_days": int(len(w)),
             "temp_station_days": int(w["temp_c_mean"].notna().sum()), "do_station_days": int(w["do_mgl_mean"].notna().sum()),
             "date_min": str(w["date"].min().date()), "date_max": str(w["date"].max().date()),
             "reserved_from": str(EA.RESERVED_FROM.date()), "content_pins": pins}
    write_manifest("ea_wq_sonde", files=files + [OUT / "CONTENT_PINS.json"],
                   sources=[f"{EA.BASE}/id/stations.json", f"{EA.BASE}/id/measures.json",
                            f"{EA.BASE}/data/readings.csv (observedProperty=temperature|dissolved-oxygen, 2012-01-01..2023-09-30)"],
                   description="EA England water-quality sonde temperature / DO, QC'd daily statistics (2012-01-01 .. "
                               "2023-09-30; values from 2023-10-01 are the reserved M6 test block and are not stored).",
                   license=EA.LICENCE, synthetic=False, stats=stats)


HS = paths.RAW / "hydrosheds"
GB_BBOX = (-7.0, 49.5, 2.5, 56.5)
US_BBOX = (-125.0, 24.0, -66.5, 49.6)
SHP_DIR = paths.interim("ea_wq_sonde") / "shp"   # extracted shapefiles WITHOUT the ESRI .sbn/.sbx index (GDAL rejects it)
RIV_SHP = str(SHP_DIR / "HydroRIVERS_v10_eu.shp")
HYB = str(SHP_DIR / "hybas_eu_lev{lev:02d}_v1c.shp")


def _extract_shapefiles() -> None:
    import zipfile

    SHP_DIR.mkdir(parents=True, exist_ok=True)
    for zf, stems in ((HS / "HydroRIVERS_v10_eu_shp.zip", ("HydroRIVERS_v10_eu",)),
                      (HS / "hybas_eu_lev01-12_v1c.zip", ("hybas_eu_lev06_v1c", "hybas_eu_lev07_v1c"))):
        with zipfile.ZipFile(zf) as z:
            for n in z.namelist():
                stem, ext = Path(n).stem, Path(n).suffix.lower()
                if stem in stems and ext in (".shp", ".shx", ".dbf", ".prj") and not (SHP_DIR / Path(n).name).exists():
                    (SHP_DIR / Path(n).name).write_bytes(z.read(n))
ATLAS_ZIP = HS / "RiverATLAS_Data_v10.gdb.zip"
ATLAS_DIR = paths.interim("ea_wq_sonde") / "RiverATLAS_v10.gdb"   # temporary extraction (7 GB), removed by step_atlas


def _extract_atlas() -> str:
    """Extract the geodatabase (the 6.75 GB table is deflated inside the zip, so /vsizip random access is too slow)."""
    import shutil
    import zipfile

    free_gb = shutil.disk_usage(paths.DATA_ROOT).free / 1e9
    if free_gb < 60 + 8:
        raise SystemExit(f"only {free_gb:.0f} GB free; the extraction needs 7 GB above the 60 GB floor")
    with zipfile.ZipFile(ATLAS_ZIP) as z:
        for i in z.infolist():
            if i.filename.startswith("RiverATLAS_v10.gdb/") and not i.is_dir():
                dst = ATLAS_DIR / Path(i.filename).name
                if not (dst.exists() and dst.stat().st_size == i.file_size):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(i) as src, open(dst, "wb") as fh:
                        shutil.copyfileobj(src, fh, 1 << 24)
    return str(ATLAS_DIR)


def _station_points():
    import geopandas as gpd
    from pyproj import Transformer

    st = pd.read_parquet(OUT / "stations.parquet").set_index("station")
    lat, lon = st["lat"].astype(float), st["lon"].astype(float)
    miss = ~(np.isfinite(lat) & np.isfinite(lon)) & st["easting"].notna() & st["northing"].notna()
    if miss.any():   # British National Grid -> WGS84 where lat/long are absent
        tr = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        x, y = tr.transform(st.loc[miss, "easting"].astype(float).to_numpy(), st.loc[miss, "northing"].astype(float).to_numpy())
        lon[miss], lat[miss] = x, y
    ok = np.isfinite(lat) & np.isfinite(lon)
    return gpd.GeoDataFrame(st[ok], geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs="EPSG:4326")


def step_geo() -> None:
    import geopandas as gpd

    from sentinel.hydromet_sota import ea_geo as EG

    _extract_shapefiles()
    pts = _station_points()
    riv = gpd.read_file(RIV_SHP, bbox=GB_BBOX, engine="pyogrio")
    log.info("HydroRIVERS GB subset: %d reaches", len(riv))
    snap = EG.snap_points(pts, riv)
    attrs = riv.drop(columns="geometry").set_index("HYRIV_ID")
    nd = dict(zip(riv["HYRIV_ID"], riv["NEXT_DOWN"]))
    out = pts.drop(columns="geometry").join(snap)
    for c in ("NEXT_DOWN", "MAIN_RIV", "UPLAND_SKM", "CATCH_SKM", "ORD_STRA", "DIST_DN_KM", "HYBAS_L12"):
        out[c.lower()] = attrs[c].reindex(out["hyriv_id"]).to_numpy()
    for lev in (6, 7):
        hb = gpd.read_file(HYB.format(lev=lev), bbox=GB_BBOX, engine="pyogrio")[["HYBAS_ID", "geometry"]]
        j = gpd.sjoin(pts[["geometry"]], hb, how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        out[f"hybas_l{lev}"] = j["HYBAS_ID"].reindex(out.index).to_numpy()
    pairs = EG.flow_pairs(out["hyriv_id"], nd)
    out["group_l7"] = EG.basin_groups(out["hybas_l7"], pairs)
    out["group_l6"] = EG.basin_groups(out["hybas_l6"], pairs)
    out["n_flow_partners"] = pd.Series([a for a, _ in pairs]).value_counts().reindex(out.index).fillna(0).astype(int).to_numpy()
    out.reset_index().to_parquet(OUT / "stations_geo.parquet", index=False)
    log.info("geo: %d stations snapped (median %.0f m; > 1 km: %d); L7 groups %d, L6 groups %d; flow pairs %d",
             int(out["hyriv_id"].notna().sum()), out["snap_m"].median(), int((out["snap_m"] > 1000).sum()),
             out["group_l7"].nunique(), out["group_l6"].nunique(), len(pairs))


UWWTD = paths.RAW / "ea_uwwtd" / "UWWTR_Art15_24thOct2024.ods"


def step_strata() -> None:
    """Near-discharge stratum from public metadata only (UWWTR Article 15 discharge points, CC BY)."""
    import geopandas as gpd

    from sentinel.hydromet_sota import ea_geo as EG

    _extract_shapefiles()
    d = EG.read_ods_sheet(UWWTD, "T_DischargePoints")
    lat = pd.to_numeric(d["dcpLatitude"], errors="coerce")
    lon = pd.to_numeric(d["dcpLongitude"], errors="coerce")
    ok = lat.notna() & lon.notna()
    dcp = gpd.GeoDataFrame(d[ok].reset_index(drop=True), geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs="EPSG:4326")
    pts = _station_points()
    riv = gpd.read_file(RIV_SHP, bbox=GB_BBOX, engine="pyogrio")
    geo = pd.read_parquet(OUT / "stations_geo.parquet").set_index("station")
    res = EG.near_discharge(pts, dcp, riv, geo["hyriv_id"])
    res.reset_index().rename(columns={"index": "station"}).to_parquet(OUT / "stations_strata.parquet", index=False)
    log.info("near-discharge: %d of %d stations (%d discharge points with coordinates); source sha256 %s",
             int(res["near_discharge"].sum()), len(res), len(dcp), _sha(UWWTD))


def step_atlas() -> None:
    """RiverATLAS v1.0 (CC BY 4.0) regional subsets: GB reaches (attributes only) and the US M6 locations snapped to
    their nearest RiverATLAS reach (``processed/hydromet_sota/riveratlas_us_sites.parquet``)."""
    import geopandas as gpd
    import pyogrio

    ATLAS = _extract_atlas()
    gb = pyogrio.read_dataframe(ATLAS, bbox=GB_BBOX, read_geometry=False)
    gb.to_parquet(OUT / "riveratlas_gb.parquet", index=False)
    log.info("RiverATLAS GB: %d reaches x %d columns", len(gb), gb.shape[1])
    del gb
    from sentinel.models.hydromet import data as D

    sp, _ = D.load_split()
    st = D.site_table()
    keep = [s for s, r in sp.site_role.items() if r in ("train", "val")]
    us = st.reindex(keep).dropna(subset=["lat", "lon"])
    pts = gpd.GeoDataFrame({"site_key": us["site_key"].to_numpy()}, geometry=gpd.points_from_xy(us["lon"], us["lat"]),
                           crs="EPSG:4326")
    pool = pd.read_parquet(paths.PROCESSED / "hydromet_sota" / "lever_sites.parquet")
    pool = pool[pool["rule"] == "fit"]
    wst = pd.read_parquet(paths.PROCESSED / "hydromet_wqp_national" / "stations.parquet").set_index("station_key")
    tn = json.loads((paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_track_n_stations.json").read_text())["role_of"]
    wk = sorted(set(pool.loc[pool["source"] == "wqp_grab", "key"]) | {k for k, r in tn.items() if r in ("train", "val")})
    wk = [k for k in wk if k in wst.index]
    pts_w = gpd.GeoDataFrame({"site_key": wk}, geometry=gpd.points_from_xy(wst.loc[wk, "lon"], wst.loc[wk, "lat"]),
                             crs="EPSG:4326")
    newu = pool.loc[pool["source"] == "usgs_dv", "key"].tolist()
    st2 = D.site_table().set_index("site_key").reindex(newu).dropna(subset=["lat", "lon"])
    pts_n = gpd.GeoDataFrame({"site_key": st2.index.to_numpy()}, geometry=gpd.points_from_xy(st2["lon"], st2["lat"]),
                             crs="EPSG:4326")
    allp = pd.concat([pts, pts_n, pts_w], ignore_index=True).drop_duplicates("site_key").set_index("site_key")
    us_r = pyogrio.read_dataframe(ATLAS, bbox=US_BBOX)
    log.info("RiverATLAS CONUS subset: %d reaches", len(us_r))
    snap = gpd.sjoin_nearest(allp.to_crs("EPSG:5070"), us_r[["HYRIV_ID", "geometry"]].to_crs("EPSG:5070"), how="left",
                             max_distance=5000, distance_col="snap_m")
    snap = snap[~snap.index.duplicated(keep="first")]
    att = us_r.drop(columns="geometry").set_index("HYRIV_ID")
    res = att.reindex(snap["HYRIV_ID"]).reset_index()
    res.insert(0, "snap_m", snap["snap_m"].to_numpy())
    res.insert(0, "site_key", snap.index.to_numpy())
    (paths.PROCESSED / "hydromet_sota").mkdir(parents=True, exist_ok=True)
    res.to_parquet(paths.PROCESSED / "hydromet_sota" / "riveratlas_us_sites.parquet", index=False)
    log.info("RiverATLAS US locations: %d snapped (median %.0f m, > 1 km: %d)", int(res["HYRIV_ID"].notna().sum()),
             res["snap_m"].median(), int((res["snap_m"] > 1000).sum()))
    import shutil

    shutil.rmtree(ATLAS_DIR, ignore_errors=True)   # the zip in raw/ stays the pinned source
    log.info("removed the temporary extraction %s", ATLAS_DIR)


SPLIT_FILE = ES.SPLIT_FILE
SALT, VAL_SHARE, BIG_GROUP_FRAC, RULE_TEXT = ES.SALT, ES.SHARE_VAL, ES.BIG_GROUP_FRAC, ES.RULE_TEXT
STUDY_FROM = pd.Timestamp("2012-01-01")
VAL_FROM = pd.Timestamp("2021-11-12")        # EA-val 2021-10-01 + 42-day embargo
VAL_TO = pd.Timestamp("2023-09-30")
TEST_FROM = pd.Timestamp("2023-11-12")       # EA-test 2023-10-01 + 42-day embargo; catalogue dates only before the look
TEST_TO = pd.Timestamp("2026-09-30")
MAX_SNAP_M = 1000.0
FLOW_SNAP_M = 5000.0                         # nearest-reach search radius of `geo` (flow connections of ineligible stations)
MIN_DAYS = 60
ELIGIBILITY = {"max_snap_m": MAX_SNAP_M, "flow_snap_max_m": FLOW_SNAP_M, "min_days": MIN_DAYS, "hydrobasins_level": 7,
               "snap_crs": "EPSG:27700", "study_from": str(STUDY_FROM.date()), "test_from": str(TEST_FROM.date()),
               "test_to": str(TEST_TO.date())}


def metadata_eligibility() -> pd.DataFrame:
    """Per catalogue station, from catalogue dates and geometry only (no EA value)."""
    geo = pd.read_parquet(OUT / "stations_geo.parquet").set_index("station")
    st = pd.read_parquet(OUT / "stations.parquet").set_index("station")
    if set(st.index) != set(geo.index):
        raise SystemExit("stations.parquet and stations_geo.parquet hold different stations")
    st = st.reindex(geo.index)
    opened = pd.to_datetime(st["date_opened"], errors="coerce")
    closed = pd.to_datetime(st["date_closed"], errors="coerce").fillna(TEST_TO).clip(upper=TEST_TO)
    out = pd.DataFrame(index=geo.index)
    out["snap_m"] = geo["snap_m"]
    out["span_days"] = (closed - opened.clip(lower=STUDY_FROM)).dt.days
    out["test_days_meta"] = (closed - opened.clip(lower=TEST_FROM)).dt.days
    out["val_days_meta"] = (closed.clip(upper=VAL_TO) - opened.clip(lower=VAL_FROM)).dt.days
    out["eligible"] = (geo["snap_m"].fillna(np.inf) <= MAX_SNAP_M) & (out["span_days"] >= MIN_DAYS) & geo["hybas_l7"].notna()
    out["test_capable"] = out["eligible"] & (out["test_days_meta"] >= MIN_DAYS)
    out["val_capable"] = out["eligible"] & (out["val_days_meta"] >= MIN_DAYS)
    me = pd.read_parquet(OUT / "measures.parquet")
    out["has_temp_measure"] = out.index.isin(set(me.loc[me["var"] == "temp_c", "station"]))
    out["has_do_measure"] = out.index.isin(set(me.loc[me["var"] == "do_mgl", "station"]))
    sf = OUT / "stations_strata.parquet"
    if sf.exists():
        out["near_discharge"] = pd.read_parquet(sf).set_index("station")["near_discharge"].reindex(out.index).fillna(False)
    out["hybas_l7"], out["hyriv_id"] = geo["hybas_l7"], geo["hyriv_id"]
    return out


def _next_down() -> dict:
    import geopandas as gpd

    _extract_shapefiles()
    riv = gpd.read_file(RIV_SHP, bbox=GB_BBOX, engine="pyogrio", read_geometry=False, columns=["HYRIV_ID", "NEXT_DOWN"])
    return dict(zip(riv["HYRIV_ID"], riv["NEXT_DOWN"]))


def ea_groups(el: pd.DataFrame, nd: dict | None = None) -> pd.Series:
    from sentinel.hydromet_sota import ea_geo as EG

    nd = _next_down() if nd is None else nd
    e = el[el["eligible"]]
    return EG.basin_groups(e["hybas_l7"].astype("int64").astype(str), EG.flow_pairs(e["hyriv_id"], nd))


def flow_partners(el: pd.DataFrame, nd: dict) -> dict:
    """Ineligible station -> eligible stations it is flow-connected to (either direction; nearest reach within 5 km)."""
    from sentinel.hydromet_sota import ea_geo as EG

    elig = set(el.index[el["eligible"]])
    out = {}
    for a, b in EG.flow_pairs(el["hyriv_id"], nd):
        if (a in elig) != (b in elig):
            s, o = (b, a) if a in elig else (a, b)
            out.setdefault(s, set()).add(o)
    return {s: sorted(v) for s, v in out.items()}


def layer_pins() -> dict:
    """sha256 of every input of the split: the two HydroSHEDS archives, the shapefile members read from them (checked
    equal to the extracted copies), the UWWTR table and the four catalogue-metadata tables."""
    import zipfile

    pins = {}
    for zf, stems in ((HS / "HydroRIVERS_v10_eu_shp.zip", ("HydroRIVERS_v10_eu",)),
                      (HS / "hybas_eu_lev01-12_v1c.zip", ("hybas_eu_lev06_v1c", "hybas_eu_lev07_v1c"))):
        pins[f"raw/hydrosheds/{zf.name}"] = _sha(zf)
        with zipfile.ZipFile(zf) as z:
            for n in sorted(z.namelist()):
                if Path(n).stem in stems and Path(n).suffix.lower() in (".shp", ".shx", ".dbf", ".prj"):
                    h = hashlib.sha256(z.read(n)).hexdigest()
                    if _sha(SHP_DIR / Path(n).name) != h:
                        raise SystemExit(f"extracted {Path(n).name} differs from the member of {zf.name}")
                    pins[f"{zf.name}::{Path(n).name}"] = h
    pins[f"raw/ea_uwwtd/{UWWTD.name}"] = _sha(UWWTD)
    for f in ("stations.parquet", "measures.parquet", "stations_geo.parquet", "stations_strata.parquet"):
        pins[f"processed/ea_wq_sonde/{f}"] = _sha(OUT / f)
    return pins


def _cells(el: pd.DataFrame, grp: pd.Series, role: pd.Series) -> dict:
    """Metadata G / G_eff per endpoint of the three T-E cells (expected stations: eligible, open >= 60 days in the
    cell's scored window by catalogue dates, with the endpoint's measure in the catalogue). Counts only; no EA value."""
    e = el.loc[grp.index]
    cells = {"test (test-role groups x EA-test, 2023-11-12..2026-09-30)": (role == "test") & e["test_capable"],
             "zero_shot_go_no_go (train+val-role groups x EA-val, 2021-11-12..2023-09-30; before M1's draw)":
                 role.isin(["train", "val"]) & e["val_capable"],
             "fine_tuned_val (val-role groups x EA-val, 2021-11-12..2023-09-30; before M1's draw)":
                 (role == "val") & e["val_capable"]}
    out = {}
    for name, m in cells.items():
        out[name] = {}
        for ep, var in ES.ENDPOINTS.items():
            mm = m & e["has_temp_measure" if var == "temp_c" else "has_do_measure"]
            out[name][ep] = ES.min_groups(grp[mm].value_counts().to_dict())
    return out


def split_body(share_test: float = ES.SHARE_TEST) -> dict:
    el = metadata_eligibility()
    nd = _next_down()
    grp = ea_groups(el, nd)
    unit_of = {s: (str(int(u)) if pd.notna(u) else None) for s, u in el["hybas_l7"].items()}
    core = ES.build_core(all_stations=list(el.index), station_group=grp.to_dict(), unit_of=unit_of,
                         flow_partners=flow_partners(el, nd), eligibility=ELIGIBILITY, inputs_sha256=layer_pins(),
                         share_test=share_test)
    role_of = core["role_of_group"]
    r = grp.map(role_of)
    e = el.loc[grp.index]
    counts = {}
    for x in ES.ROLES:
        m = r == x
        tc = m & e["test_capable"]
        counts[x] = {"groups": int(grp[m].nunique()), "eligible_stations": int(m.sum()),
                     "test_capable_stations": int(tc.sum()), "test_capable_groups": int(grp[tc].nunique()),
                     "test_capable_with_do_measure": int((tc & e["has_do_measure"]).sum()),
                     "test_capable_with_temp_measure": int((tc & e["has_temp_measure"]).sum()),
                     "test_capable_near_discharge": int((tc & e.get("near_discharge", False)).sum())}
    st = pd.DataFrame.from_dict(core["stations"], orient="index")
    status = {f"{a}/{b}": int(n) for (a, b), n in st.groupby(["status", "basis"]).size().items()}
    body = ES.seal(
        core, track="M6 T-E EA England (docs/preregistration/hydromet_ea_protocol.md §4)",
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        reserved={"all_stations_from": "2023-10-01",
                  "closed_to_other_workstreams_all_years": "stations in closed_stations (status test or reserved)"},
        counts_metadata_only={"n_catalogue": int(len(el)), "n_eligible": int(el["eligible"].sum()),
                              "n_test_capable": int(el["test_capable"].sum()), "n_groups": int(grp.nunique()),
                              "largest_group": int(grp.value_counts().max()), "roles": counts,
                              "station_status_by_basis": status, "n_closed_stations": len(core["closed_stations"])},
        min_groups_metadata={"G_min": ES.G_MIN, "G_eff_min": ES.G_EFF_MIN, "cells": _cells(el, grp, r)},
        note="No EA value was read to build this file (catalogue dates, measure list and geometry only).")
    return ES.verify(body, ruled_parameters=False)


def step_preview(share_test: float) -> None:
    b = split_body(share_test)
    print(json.dumps({k: b[k] for k in ("share_test", "share_val", "salt", "eligibility", "counts_metadata_only",
                                         "min_groups_metadata", "inputs_sha256", "sha256")}, indent=1))


def step_split(share_test: float | None, ruled: bool) -> None:
    if not ruled:
        raise SystemExit("the EA split is written once, by the lead, after the T-E entry is stamped "
                         "(--i-have-the-lead-ruling)")
    if share_test is not None and float(share_test) != ES.SHARE_TEST:
        raise SystemExit(f"S = {ES.SHARE_TEST} is the ruling (hydromet_arch_v2_lead_rulings.md Q4); got {share_test}")
    body = ES.verify(split_body(ES.SHARE_TEST))          # ruled salt / S / val share / guard
    if SPLIT_FILE.exists():
        old = ES.load(SPLIT_FILE)                         # never loosened: a file that fails the check is refused
        if old["sha256"] != body["sha256"]:
            raise SystemExit(f"{SPLIT_FILE} exists with another sha256 — the EA split is frozen")
        log.info("split unchanged (%s)", body["sha256"])
        return
    SPLIT_FILE.write_text(json.dumps(body, indent=1))
    SPLIT_FILE.chmod(0o444)
    if ES.load(SPLIT_FILE)["sha256"] != body["sha256"]:
        raise SystemExit("the written split does not read back with the same sha256")
    log.info("EA split written: %s sha256 %s; counts %s", SPLIT_FILE, body["sha256"],
             json.dumps(body["counts_metadata_only"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["catalogue", "readings", "daily", "manifest", "geo", "atlas", "preview", "split", "strata"])
    ap.add_argument("--share-test", type=float, default=None, help="S of hydromet_ea_protocol.md §4 (ruled: 0.50, the default; another value is accepted by `preview` only)")
    ap.add_argument("--i-have-the-lead-ruling", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(RAW / "build.log")])
    {"catalogue": step_catalogue, "readings": step_readings,
     "daily": lambda: step_daily() if SPLIT_FILE.exists() else sys.exit("daily QC waits for the frozen EA split (lead hold)"), "manifest": step_manifest,
     "geo": step_geo, "atlas": step_atlas, "strata": step_strata,
     "preview": lambda: step_preview(a.share_test if a.share_test is not None else ES.SHARE_TEST),
     "split": lambda: step_split(a.share_test, a.i_have_the_lead_ruling)}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
