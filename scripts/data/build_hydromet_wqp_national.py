#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build ``processed/hydromet_wqp_national``: the M6 track-N ("national unsensed reaches") evaluation set —
WQP discrete stream water-temperature and DO samples at stations WITHOUT a continuous sensor, with the same
inputs M6 uses (gridMET site-cell forcings, StreamCat attributes of the NHDPlus V2 COMID). Only the val
(WY2022–2023) and test (WY2024–2025) blocks are kept; nothing here is ever a training row.

    python -B scripts/data/build_hydromet_wqp_national.py catchments   # NHDPlusV2 simplified catchments (EPA S3)
    python -B scripts/data/build_hydromet_wqp_national.py stations     # eligibility, COMID join, daily obs
    python -B scripts/data/build_hydromet_wqp_national.py streamcat    # StreamCat for the station COMIDs
    python -B scripts/data/build_hydromet_wqp_national.py forcing      # gridMET point forcings (yearly files)
    python -B scripts/data/build_hydromet_wqp_national.py manifest

Rules (protocol §12; fixed before any model output on these stations was computed):
* WQP ``site_class == stream``, CONUS; variables ``temp`` (deg C) and ``do`` (mg/L); ``status_class == accepted``;
  uncensored finite values; activity types other than time-series composites; depth NaN or <= 2 m.
* Daily value = mean of the samples of a station-date. Station x variable x block eligible with >= 8 sample days in
  the (embargoed) block.
* Unsensed: drop WQP ``USGS-<no>`` stations whose site has any USGS daily-value water-quality series, and every
  station whose COMID carries a USGS DV water-quality site or a FINAL sonde-split site.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import re
import shutil
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import gridmet as G  # noqa: E402
from sentinel.data.sources import streamcat as SC  # noqa: E402

NAME = "hydromet_wqp_national"
OUT = paths.processed(NAME)
INT = paths.interim(NAME)
WQP = paths.PROCESSED / "wqp_discrete"
S3 = "https://dmap-data-commons-ow.s3.amazonaws.com"
CATCH_PREFIX = "NHDPlusV21/Data/Extensions/SimplifiedCatchments/"
CONUS_VPU = re.compile(r"NHDPlusV21_[A-Z]{2}_(0[1-9]|1[0-8])[A-Z]*_NHDPlusCatchment2_\d+\.7z$")
BLOCKS = {"val": ("2021-11-12", "2023-09-30"), "test": ("2023-11-12", "2025-09-30")}   # 42-day embargo applied
KEEP_FROM = "2021-10-01"
MIN_DAYS = 8
MAX_DEPTH_M = 2.0
VARS = {"temp": ("temp_c_mean", "deg C"), "do": ("do_mgl_mean", "mg/L")}
NON_STATES = {"AK", "HI", "PR", "GU", "VI", "AS", "MP"}
UA = {"User-Agent": "SENTINEL-v2 research (public-data download)"}
log = logging.getLogger(NAME)


# ----------------------------------------------------------------------------- catchments

def step_catchments() -> None:
    import geopandas as gpd
    import py7zr

    r = requests.get(S3, params={"prefix": CATCH_PREFIX, "max-keys": 1000}, headers=UA, timeout=120)
    r.raise_for_status()
    keys = [k for k in re.findall(r"<Key>([^<]+)</Key>", r.text) if CONUS_VPU.search(k)]
    log.info("%d CONUS simplified-catchment archives", len(keys))
    cdir = INT / "catchments"
    cdir.mkdir(parents=True, exist_ok=True)
    for k in keys:
        vpu = re.search(r"NHDPlusV21_([A-Z]{2}_\w+?)_NHDPlusCatchment2", k).group(1)
        out = cdir / f"{vpu}.parquet"
        if out.exists():
            continue
        tmp = INT / "tmp7z"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        arc = tmp / Path(k).name
        G.download(f"{S3}/{k}", arc)
        with py7zr.SevenZipFile(arc, "r") as z:
            z.extractall(tmp)
        shp = [p for p in tmp.rglob("*.shp") if "atchment" in p.name]
        if len(shp) != 1:
            raise RuntimeError(f"{k}: expected one catchment shapefile, got {shp}")
        g = gpd.read_file(shp[0])
        col = next(c for c in g.columns if c.upper() == "FEATUREID")
        g = g[[col, "geometry"]].rename(columns={col: "comid"})
        g["comid"] = g["comid"].astype("int64").astype(str)
        g = g.to_crs("EPSG:4269") if g.crs is not None and g.crs.to_epsg() != 4269 else g
        g.to_parquet(out, compression="zstd")
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("catchments %s: %d", vpu, len(g))


# ----------------------------------------------------------------------------- stations + obs

def _sensor_sets() -> tuple[set, set]:
    """(USGS site numbers with any DV water-quality series, COMIDs of those sites and of the FINAL split sites)."""
    from sentinel.models.hydromet import data as D

    idx = pd.read_parquet(D.SONDE_INDEX, columns=["site_no"])
    dv = set(idx["site_no"].astype(str))
    sp, _ = D.load_split()
    split_sites = set(sp.site_role)
    st = pd.read_parquet(D.SITES_FILE)
    st = st[st["network"] == "USGS"].drop_duplicates("site_id")
    keys = set(st.loc[st["site_id"].astype(str).isin(dv | split_sites), "site_key"].astype(str))
    sc = pd.read_parquet(D.STREAMCAT_FILE, columns=["site_key", "comid"])
    nodes = pd.read_parquet(paths.PROCESSED / "site_network" / "nodes.parquet")
    ccol = "comid" if "comid" in nodes.columns else None
    comids = set(sc.loc[sc["site_key"].isin(keys), "comid"].dropna().astype(str))
    if ccol and "site_key" in nodes.columns:
        comids |= set(nodes.loc[nodes["site_key"].astype(str).isin(keys), ccol].dropna().astype("int64").astype(str))
    return dv, comids


def step_stations() -> None:
    import geopandas as gpd
    import pyarrow.dataset as ds

    obs = []
    for v, (target, unit) in VARS.items():
        d = ds.dataset(WQP / f"variable={v}" / "data.parquet")
        flt = (ds.field("site_class") == "stream") & (ds.field("activity_start_date") >= pd.Timestamp(KEEP_FROM).date())
        t = d.to_table(columns=["site_id", "state", "activity_start_date", "value", "unit", "censored", "depth_m",
                                "activity_type", "status_class"], filter=flt).to_pandas()
        n0 = len(t)
        t = t[(t["unit"] == unit) & (t["status_class"] == "accepted") & ~t["censored"].fillna(False).astype(bool)
              & np.isfinite(t["value"]) & ~t["activity_type"].fillna("").str.contains("Time Series|Quality Control")
              & (t["depth_m"].isna() | (t["depth_m"] <= MAX_DEPTH_M)) & ~t["state"].isin(NON_STATES)]
        t["date"] = pd.to_datetime(t["activity_start_date"])
        day = t.groupby(["site_id", "date"], as_index=False).agg(value=("value", "mean"), n_samples=("value", "size"))
        day["target"] = target
        obs.append(day)
        log.info("%s: %d rows -> %d kept rows -> %d station-days", v, n0, len(t), len(day))
    obs = pd.concat(obs, ignore_index=True)
    # block eligibility
    elig = []
    for b, (lo, hi) in BLOCKS.items():
        x = obs[(obs["date"] >= lo) & (obs["date"] <= hi)]
        n = x.groupby(["site_id", "target"]).size()
        e = n[n >= MIN_DAYS].reset_index(name="n_days")
        e["block"] = b
        elig.append(e)
    elig = pd.concat(elig, ignore_index=True)
    stations = pd.read_parquet(WQP / "_stations.parquet",
                               columns=["site_id", "organization", "provider", "state", "huc8", "lat_std", "lon_std",
                                        "datum_std", "site_type", "site_class"]).drop_duplicates("site_id")
    stations = stations[stations["site_id"].isin(set(elig["site_id"]))].rename(columns={"lat_std": "lat", "lon_std": "lon"})
    info = {"n_candidates": int(len(stations))}
    dv, sensor_comids = _sensor_sets()
    usgs_no = stations["site_id"].str.extract(r"^USGS-(\d+)$")[0]
    stations["excl_usgs_dv_site"] = usgs_no.isin(dv).to_numpy()
    # COMID by point-in-catchment (NHDPlusV2 simplified catchments)
    pts = gpd.GeoDataFrame(stations[["site_id"]], geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
                           crs="EPSG:4269")
    hits = []
    for f in sorted((INT / "catchments").glob("*.parquet")):
        c = gpd.read_parquet(f)
        j = gpd.sjoin(pts, c, how="inner", predicate="within")[["site_id", "comid"]]
        hits.append(j)
    hits = pd.concat(hits).drop_duplicates("site_id")
    stations = stations.merge(hits, on="site_id", how="left")
    stations["excl_no_comid"] = stations["comid"].isna()
    stations["excl_sensor_comid"] = stations["comid"].isin(sensor_comids)
    keep = ~(stations["excl_usgs_dv_site"] | stations["excl_no_comid"] | stations["excl_sensor_comid"])
    info.update({k: int(stations[k].sum()) for k in ("excl_usgs_dv_site", "excl_no_comid", "excl_sensor_comid")})
    stations = stations[keep].copy()
    stations["station_key"] = "WQP:" + stations["site_id"]
    info["n_stations"] = int(len(stations))
    stations.to_parquet(OUT / "stations.parquet", compression="zstd", index=False)
    o = obs[obs["site_id"].isin(set(stations["site_id"]))]
    o.to_parquet(OUT / "obs_daily.parquet", compression="zstd", index=False)
    e = elig[elig["site_id"].isin(set(stations["site_id"]))]
    e.to_parquet(OUT / "eligibility.parquet", compression="zstd", index=False)
    info["eligible_station_target_block"] = e.groupby(["target", "block"]).size().to_dict()
    pd.Series(info, dtype=object).to_json(INT / "stations_info.json", default_handler=str)
    log.info("stations: %s", info)


# ----------------------------------------------------------------------------- StreamCat

def step_streamcat() -> None:
    st = pd.read_parquet(OUT / "stations.parquet")
    comids = sorted(st["comid"].dropna().astype(str).unique(), key=int)
    client = SC.StreamCatClient(INT / "streamcat_cache")
    import json

    known_bad = set()
    scm = paths.MANIFESTS / "streamcat.json"
    if scm.exists():
        known_bad = {(f["metric"], f["aoi"]) for f in json.loads(scm.read_text()).get("stats", {}).get("failed_metrics", [])}
    wide = pd.DataFrame({"comid": comids}).set_index("comid")
    metric_cols = []
    for group, aoi, names in SC.CURATED:
        names = [n for n in names if (n, aoi) not in known_bad]
        if not names:
            continue
        frames = []
        for b in range(0, len(comids), 500):
            url, data = SC.metrics_request(names, comids[b:b + 500], aoi)
            frames.append(SC.parse_metrics(client.post(url, data)["body"]))
        df = pd.concat(frames, ignore_index=True).drop_duplicates("comid").set_index("comid")
        wide = wide.join(df[[c for c in df.columns if c not in wide.columns]], how="left")
        metric_cols += [c for c in SC.expected_columns(names, aoi) if c in wide.columns]
        log.info("streamcat %s/%s done (quota left %s)", group, aoi, client.stats.get("ratelimit_remaining"))
    wide = wide.reset_index()
    vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet",
                          columns=["comid", "slope", "streamorde", "totdasqkm", "lengthkm"])
    vaa["comid"] = vaa["comid"].astype("int64").astype(str)
    vaa = vaa.rename(columns={"slope": "nhd_slope", "streamorde": "nhd_streamorde", "totdasqkm": "nhd_totdasqkm",
                              "lengthkm": "nhd_lengthkm"})
    vaa.loc[vaa["nhd_slope"] < 0, "nhd_slope"] = np.nan
    wide = wide.merge(vaa, on="comid", how="left")
    metric_cols = list(dict.fromkeys(metric_cols))
    wide["streamcat_found"] = wide[metric_cols].notna().any(axis=1)
    wide = wide.drop(columns=[c for c in wide.columns if "pctfull" in c])
    wide.to_parquet(OUT / "streamcat_comid.parquet", compression="zstd", index=False)
    log.info("streamcat: %d comids, %d found", len(wide), int(wide["streamcat_found"].sum()))


# ----------------------------------------------------------------------------- forcing

def step_forcing() -> None:
    st = pd.read_parquet(OUT / "stations.parquet")
    years = list(range(2005, 2026))
    ncd = INT / "nc"
    ncd.mkdir(parents=True, exist_ok=True)
    cells_f = OUT / "station_cells.parquet"
    agg = INT / "forc"
    agg.mkdir(exist_ok=True)
    cells = pd.read_parquet(cells_f) if cells_f.exists() else None
    for year in years:
        for var in G.VARIABLES:
            out = agg / f"{var}_{year}.npz"
            if out.exists():
                continue
            p = ncd / f"{var}_{year}.nc"
            if not p.exists():
                G.download(G.file_url(var, year), p)
            grid = G.read_grid(p)
            if cells is None:
                valid = np.isfinite(grid["data"]).all(axis=0)
                cells = G.nearest_valid_cells(st["lat"].to_numpy(), st["lon"].to_numpy(), grid["lat"], grid["lon"], valid)
                cells.insert(0, "station_key", st["station_key"].to_numpy())
                cells.to_parquet(cells_f, compression="zstd", index=False)
            m = grid["dates"] >= pd.Timestamp("2005-10-01")
            vals = G.to_harmonised(var, G.extract_cells({"data": grid["data"][m]}, cells))   # (days, stations)
            np.savez(out, values=vals.T.astype("float32"), dates=grid["dates"][m].strftime("%Y-%m-%d").to_numpy().astype("U10"))
            p.unlink(missing_ok=True)
            log.info("forcing %s %d", var, year)
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = cells["station_key"].tolist()
    cols, dates = {}, None
    for var, (_, _, col, _) in G.VARIABLES.items():
        zz = [np.load(agg / f"{var}_{y}.npz") for y in years]
        d = pd.to_datetime(np.concatenate([z["dates"] for z in zz]))
        a = np.concatenate([z["values"] for z in zz], axis=1)
        if dates is None:
            dates = d
        if not (d == dates).all():
            raise RuntimeError(f"date mismatch {var}")
        cols[col] = a
    tmp = OUT / "forcings.parquet.part"
    w = None
    for c in range(0, len(keys), 500):
        ks = keys[c:c + 500]
        df = pd.DataFrame({"station_key": np.repeat(np.asarray(ks, dtype=object), len(dates)), "date": np.tile(dates.date, len(ks))})
        for col, a in cols.items():
            df[col] = a[c:c + 500].reshape(-1)
        t = pa.Table.from_pandas(df, preserve_index=False)
        w = w or pq.ParquetWriter(tmp, t.schema, compression="zstd")
        w.write_table(t, row_group_size=len(df))
    w.close()
    tmp.replace(OUT / "forcings.parquet")
    log.info("forcings: %d stations x %d days", len(keys), len(dates))


def step_manifest() -> None:
    import json

    st = pd.read_parquet(OUT / "stations.parquet")
    el = pd.read_parquet(OUT / "eligibility.parquet")
    sc = pd.read_parquet(OUT / "streamcat_comid.parquet", columns=["comid", "streamcat_found"])
    cells = pd.read_parquet(OUT / "station_cells.parquet")
    info = json.loads((INT / "stations_info.json").read_text())
    stats = {**info, "n_comids": int(len(sc)), "n_streamcat_found": int(sc["streamcat_found"].sum()),
             "n_no_gridmet_cell": int((cells["i"] < 0).sum()), "blocks": BLOCKS, "min_days": MIN_DAYS,
             "max_depth_m": MAX_DEPTH_M, "eligible": el.groupby(["target", "block"]).size().astype(int).to_dict(),
             "by_provider": st["provider"].value_counts().to_dict(),
             "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    stats["eligible"] = {f"{k[0]}|{k[1]}": v for k, v in stats["eligible"].items()}
    files = [OUT / f for f in ("stations.parquet", "obs_daily.parquet", "eligibility.parquet", "streamcat_comid.parquet",
                               "station_cells.parquet", "forcings.parquet")]
    write_manifest(NAME, files=files,
                   sources=["processed/wqp_discrete (manifest wqp_discrete)", f"{S3}/{CATCH_PREFIX}",
                            SC.METRICS_URL + " (POST)", G.BULK_URL],
                   description="M6 track N: WQP discrete stream temperature/DO at stations without a continuous sensor "
                               "(val/test blocks only) + site-cell gridMET forcings + StreamCat of the NHDPlusV2 COMID.",
                   license="public domain (WQP/USGS/EPA; gridMET CC0; NHDPlusV2 public domain)", synthetic=False,
                   stats=stats)
    log.info("manifest: %s", json.dumps(stats, default=str)[:1500])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["catchments", "stations", "streamcat", "forcing", "manifest", "all"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(INT / "build.log")])
    for s in ("catchments", "stations", "streamcat", "forcing", "manifest"):
        if a.step in (s, "all"):
            globals()[f"step_{s}"]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
