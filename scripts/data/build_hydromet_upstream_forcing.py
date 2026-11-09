#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build ``processed/hydromet_upstream_forcing``: upstream-basin-averaged daily gridMET forcings for the
M6-eligible sites (FINAL sonde split, all roles; inputs only, no targets are read).

    python -B scripts/data/build_hydromet_upstream_forcing.py basins_local  # NHDPlusV2 network + catchments (no NLDI)
    python -B scripts/data/build_hydromet_upstream_forcing.py aggregate   # gridMET yearly files -> basin means
    python -B scripts/data/build_hydromet_upstream_forcing.py combine     # parquet + coverage + manifest
    python -B scripts/data/build_hydromet_upstream_forcing.py all

gridMET yearly netCDFs are downloaded one at a time to the interim directory and deleted after use.
Logic: ``sentinel.data.hydromet_upstream``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import hydromet_upstream as U  # noqa: E402
from sentinel.data.sources import gridmet as G  # noqa: E402
from sentinel.data.sources import nldi as N  # noqa: E402

NAME = "hydromet_upstream_forcing"
OUT = paths.processed(NAME)
INT = paths.interim(NAME)
START = "2005-10-01"           # = sentinel.models.hydromet.data.DATA_START
YEARS = list(range(2005, _dt.date.today().year + 1))
VARS = list(G.VARIABLES)       # pr tmmx tmmn srad vs rmax rmin sph vpd pet
log = logging.getLogger(NAME)
SET = "sonde"        # --set: "sonde" (M6-eligible sonde sites) | "trackn" (M6 track-N WQP stations, protocol §12)
SFX = ""             # output-file suffix ("_trackn" for track N)
WORK = INT           # per-set work dir (weights, per-year aggregates); catchment cells and netCDFs are shared


def site_list() -> pd.DataFrame:
    if SET == "trackn":   # track-N stations: site_key = station_key
        nd = paths.PROCESSED / "hydromet_wqp_national"
        st = pd.read_parquet(nd / "stations.parquet")
        sc = pd.read_parquet(nd / "streamcat_comid.parquet", columns=["comid", "wsareasqkm"]).drop_duplicates("comid")
        df = pd.DataFrame({"site_no": st["site_id"], "site_key": st["station_key"], "lat": st["lat"], "lon": st["lon"],
                           "comid": st["comid"].astype(str), "comid_method": "nhdplusv2_simplified_catchment"})
        df["streamcat_wsareasqkm"] = sc.set_index("comid")["wsareasqkm"].reindex(df["comid"]).to_numpy()
        return df.sort_values("site_key").reset_index(drop=True)
    return _sonde_site_list()


def _site_cell_forcings(keys: list[str], dates: pd.DatetimeIndex) -> np.ndarray:
    from sentinel.models.hydromet import data as D

    if SET != "trackn":
        return D.load_forcings(keys, dates)
    import pyarrow.dataset as ds

    out = np.full((len(keys), len(dates), len(D.FORCING_VARS)), np.nan, dtype="float32")
    pos = {k: i for i, k in enumerate(keys)}
    dset = ds.dataset(paths.PROCESSED / "hydromet_wqp_national" / "forcings.parquet")
    for c in range(0, len(keys), 1000):
        tab = dset.to_table(columns=["station_key", "date", "tmax_c", "precip_mm"],
                            filter=ds.field("station_key").isin(keys[c:c + 1000]) & (ds.field("date") >= dates[0].date())
                            & (ds.field("date") <= dates[-1].date())).to_pandas()
        si = tab["station_key"].map(pos).to_numpy()
        ti = (pd.to_datetime(tab["date"]) - dates[0]).dt.days.to_numpy()
        for v in ("tmax_c", "precip_mm"):
            out[si, ti, D.FORCING_VARS.index(v)] = tab[v].to_numpy(dtype="float32")
    return out


def _sonde_site_list() -> pd.DataFrame:
    from sentinel.models.hydromet import data as D

    sp, _ = D.load_split()
    sites, info = D.eligible_sites(sp, ("train", "val", "test"))
    st = D.site_table().loc[sites]
    sc = pd.read_parquet(D.STREAMCAT_FILE, columns=["site_key", "comid", "comid_method", "wsareasqkm"])
    sc = sc.drop_duplicates("site_key").set_index("site_key")
    df = pd.DataFrame({"site_no": sites, "site_key": st["site_key"].to_numpy(),
                       "lat": st["lat"].to_numpy(), "lon": st["lon"].to_numpy()})
    df["comid"] = sc["comid"].reindex(df["site_key"]).astype(str).to_numpy()
    df["comid_method"] = sc["comid_method"].reindex(df["site_key"]).to_numpy()
    df["streamcat_wsareasqkm"] = sc["wsareasqkm"].reindex(df["site_key"]).to_numpy()
    log.info("sites: %s", info)
    return df.sort_values("site_key").reset_index(drop=True)


def step_basins() -> None:
    from shapely import wkb

    df = site_list()
    cli = N.NLDIClient(INT / "nldi_cache")
    rows = []
    for k, r in enumerate(df.itertuples(index=False)):
        rec = cli.get(U.basin_url(N.BASE, r.comid), U.BASIN_PARAMS)
        g = U.parse_basin(rec.get("body")) if rec["status"] == 200 else None
        rows.append({"site_key": r.site_key, "status": rec["status"], "retrieved_utc": rec.get("retrieved_utc"),
                     "geometry_wkb": wkb.dumps(g) if g is not None else None,
                     "basin_area_km2": U.geodesic_area_km2(g) if g is not None else np.nan})
        if k % 200 == 0:
            log.info("basins %d/%d  %s", k, len(df), cli.stats)
    b = df.merge(pd.DataFrame(rows), on="site_key")
    b.to_parquet(OUT / f"basins{SFX}.parquet", compression="zstd", index=False)
    log.info("basins written: %d ok of %d; NLDI stats %s", int(b["geometry_wkb"].notna().sum()), len(b), cli.stats)


def _nc_path(var: str, year: int) -> Path:
    return INT / "nc" / f"{var}_{year}.nc"


def _fetch(var: str, year: int) -> Path:
    p = _nc_path(var, year)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        n = G.download(G.file_url(var, year), p)
        log.info("downloaded %s %d (%.0f MB)", var, year, n / 1e6)
    return p


CATCH_DIR = paths.INTERIM / "hydromet_wqp_national" / "catchments"   # NHDPlusV2 simplified catchments (per VPU)


def _grid():
    grid = G.read_grid(_fetch("tmmx", 2020))
    valid = np.isfinite(grid["data"]).all(axis=0)
    return U.grid_geometry(grid["lat"], grid["lon"]), valid


def catchment_matrix(geo: dict, pos: dict, n_nodes: int):
    """CSR (VAA node × gridMET cell) sub-cell counts of every CONUS NHDPlusV2 simplified catchment (cached)."""
    import geopandas as gpd
    from scipy import sparse

    cf = INT / "catchment_cells.npz"
    if cf.exists():
        z = np.load(cf)
        return sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    parts = []
    files = sorted(CATCH_DIR.glob("*.parquet"))
    if len(files) != 21:
        raise RuntimeError(f"expected 21 CONUS VPU catchment files in {CATCH_DIR}, found {len(files)}")
    for f in files:
        g = gpd.read_parquet(f)
        g["idx"] = g["comid"].map(pos)
        n0 = len(g)
        g = g[g["idx"].notna()].copy()
        g["idx"] = g["idx"].astype("int64")
        parts.append(U.catchment_subcells(g, geo))
        log.info("catchment cells %s: %d of %d catchments on VAA flowlines", f.stem, len(g), n0)
    c = np.concatenate([p[0] for p in parts])
    cell = np.concatenate([p[1] for p in parts])
    n = np.concatenate([p[2] for p in parts])
    C = sparse.csr_matrix((n, (c, cell)), shape=(n_nodes, geo["ny"] * geo["nx"]))
    C.sum_duplicates()
    np.savez(cf, data=C.data, indices=C.indices, indptr=C.indptr, shape=np.array(C.shape))
    return C


def step_basins_local() -> None:
    """Upstream basins from the NHDPlus V2 network + simplified catchments (no NLDI request)."""
    df = site_list()
    vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet", columns=["comid", "hydroseq", "dnhydroseq"])
    vaa["comid"] = vaa["comid"].astype("int64").astype(str)
    g, pos = U.reverse_graph(vaa)
    geo, valid = _grid()
    C = catchment_matrix(geo, pos, len(vaa))
    ny, nx = geo["ny"], geo["nx"]
    coslat = np.repeat(np.cos(np.radians(geo["lat"])), nx)
    valid_flat = valid.ravel()
    sub_km2 = (111.32 * abs(geo["dlat"]) / U.SUPERSAMPLE) * (111.32 * geo["dlon"] / U.SUPERSAMPLE)
    rows, recs = [], []
    for r in df.itertuples(index=False):
        rec = {"site_key": r.site_key}
        if r.comid in pos:
            nodes = U.upstream_nodes(g, pos[r.comid])
            idx, w, info = U.basin_weights_local(nodes, C, coslat, valid_flat)
            rec["basin_area_km2"] = float(np.asarray(C[nodes].sum(axis=0)).ravel() @ coslat) * sub_km2
        else:
            idx, w, info = np.zeros(0, np.int64), np.zeros(0), {"n_cells": 0, "cover_frac": 0.0, "n_catchments": 0}
            rec["basin_area_km2"] = np.nan
        if w.sum() <= 0:   # no network/catchment: the site's own gridMET cell (= site-cell forcing)
            i = int(np.clip(round((r.lat - geo["lat"][0]) / geo["dlat"]), 0, ny - 1))
            j = int(np.clip(round((r.lon - geo["lon"][0]) / geo["dlon"]), 0, nx - 1))
            idx, w = np.array([i * nx + j], np.int64), np.array([1.0])
            info = {**info, "n_cells": 1, "fallback": True}
        rows.append((idx, w))
        recs.append({**rec, **info})
    b = df.merge(pd.DataFrame(recs), on="site_key")
    b["method"] = "local"
    # QC rule (fixed before use): the main-path graph misses basins upstream of divergences/diversions (e.g.
    # Santee–Cooper). Where the local area differs from StreamCat's watershed area by more than a factor of 2, the NLDI
    # basin polygon (which follows divergences) is used instead — a few dozen polite requests.
    from shapely import wkb  # noqa: F401

    ratio = b["basin_area_km2"] / b["streamcat_wsareasqkm"]
    bad = np.flatnonzero(~((ratio >= 0.5) & (ratio <= 2.0)).to_numpy())
    cli = N.NLDIClient(INT / "nldi_cache", offline=(SET == "trackn"))   # track N: cache only (shared quota)
    for k in bad:
        r = b.iloc[k]
        rec = cli.get(U.basin_url(N.BASE, r["comid"]), U.BASIN_PARAMS)
        gm = U.parse_basin(rec.get("body")) if rec["status"] == 200 else None
        if gm is None:
            b.loc[b.index[k], "method"] = "local (NLDI basin unavailable)"
            continue
        idx, w, info = U.basin_cell_weights(gm, geo, valid)
        rows[k] = (idx, w)
        b.loc[b.index[k], ["n_cells", "cover_frac", "fallback"]] = [info["n_cells"], info["cover_frac"], info["fallback"]]
        b.loc[b.index[k], "basin_area_km2"] = U.geodesic_area_km2(gm)
        b.loc[b.index[k], "method"] = "nldi_polygon"
    log.info("basins replaced by NLDI polygons: %d of %d flagged (NLDI stats %s)", int((b["method"] == "nldi_polygon").sum()),
             len(bad), cli.stats)
    W = U.weight_matrix(rows, ny * nx)
    np.savez(WORK / "weights.npz", data=W.data, indices=W.indices, indptr=W.indptr, shape=np.array(W.shape))
    # QC against NLDI basins already cached (offline: no new request)
    from shapely import wkb  # noqa: F401

    cli = N.NLDIClient(INT / "nldi_cache", offline=True)
    nl = []
    for r in b.itertuples(index=False):
        rec = cli.get(U.basin_url(N.BASE, r.comid), U.BASIN_PARAMS)
        gm = U.parse_basin(rec.get("body")) if rec["status"] == 200 else None
        nl.append(U.geodesic_area_km2(gm) if gm is not None else np.nan)
    b["nldi_basin_area_km2"] = nl
    b.to_parquet(OUT / f"basins{SFX}.parquet", compression="zstd", index=False)
    rat = b["basin_area_km2"] / b["nldi_basin_area_km2"]
    rs = b["basin_area_km2"] / b["streamcat_wsareasqkm"]
    log.info("area ratio vs StreamCat after the rule: %s; outside [0.5, 2]: %d", rs.quantile([.01, .1, .5, .9, .99]).round(3).tolist(),
             int((~((rs >= 0.5) & (rs <= 2))).sum()))
    log.info("local basins: %d sites, fallback %d, n_cells median %.0f; area ratio local/NLDI on %d cached: %s",
             len(b), int(b["fallback"].sum()), b["n_cells"].median(), int(rat.notna().sum()),
             rat.quantile([.01, .1, .5, .9, .99]).round(3).tolist())


def step_weights():
    from scipy import sparse

    z = np.load(WORK / "weights.npz", allow_pickle=False)
    b = pd.read_parquet(OUT / f"basins{SFX}.parquet")
    return sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])), b


def step_aggregate() -> None:
    W, b = step_weights()
    keep_last = None
    for year in YEARS:
        for var in VARS:
            out = WORK / "agg" / f"{var}_{year}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                p = _fetch(var, year)
            except FileNotFoundError:
                log.warning("missing %s %d", var, year)
                continue
            grid = G.read_grid(p)
            m = grid["dates"] >= pd.Timestamp(START)
            vals = np.empty((W.shape[0], int(m.sum())), np.float32)
            dd = np.flatnonzero(m)
            for c in range(0, len(dd), 61):
                sl = dd[c:c + 61]
                vals[:, c:c + len(sl)] = U.aggregate_grid(W, grid["data"][sl])
            vals = G.to_harmonised(var, vals)
            np.savez(out, values=vals, dates=grid["dates"][m].strftime("%Y-%m-%d").to_numpy().astype("U10"),
                     status=grid["status"][m].astype("U11"))
            if not (var == "tmmx" and year == 2020) or keep_last:
                p.unlink(missing_ok=True)
            log.info("%s %d: %d days in %.0f s", var, year, vals.shape[1], time.time() - t0)
    _nc_path("tmmx", 2020).unlink(missing_ok=True)


def step_combine() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    b = pd.read_parquet(OUT / f"basins{SFX}.parquet")
    keys = b["site_key"].tolist()
    arrays, status, dates_ref = {}, None, None
    for var in VARS:
        vals, dts, sts = [], [], []
        for year in YEARS:
            f = WORK / "agg" / f"{var}_{year}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            vals.append(z["values"])
            dts.append(z["dates"])
            sts.append(z["status"])
        d = pd.to_datetime(np.concatenate(dts))
        v = np.concatenate(vals, axis=1)
        s = np.concatenate(sts)
        col = G.VARIABLES[var][2]
        if dates_ref is None:
            dates_ref, status = d, pd.Series(s, index=d)
        n = min(len(dates_ref), len(d))   # variables can differ by the last 1-2 days of the current year
        if not (d[:n] == dates_ref[:n]).all():
            raise RuntimeError(f"date mismatch for {var}")
        dates_ref = dates_ref[:n]
        arrays = {k: a[:, :n] for k, a in arrays.items()}
        arrays[col] = v[:, :n]
        rank = {"permanent": 0, "provisional": 1, "early": 2}
        st_prev = status.iloc[:n].map(rank).to_numpy()
        status = pd.Series(np.maximum(st_prev, pd.Series(s[:n]).map(rank).to_numpy()), index=dates_ref).map(
            {v_: k_ for k_, v_ in rank.items()})
    out_f = OUT / f"upstream_forcing{SFX}.parquet"
    tmp = out_f.with_suffix(".parquet.part")
    writer = None
    for c in range(0, len(keys), 100):
        df = U.to_long(keys[c:c + 100], dates_ref, {k: a[c:c + 100] for k, a in arrays.items()})
        df["status"] = np.tile(status.to_numpy().astype(str), len(keys[c:c + 100]))
        t = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t, row_group_size=len(df))
    writer.close()
    tmp.replace(out_f)
    # QC against the site-cell forcings (processed/forcings_daily) on WY2008-2021 days
    from sentinel.models.hydromet import data as D

    qc_d = pd.date_range("2007-10-01", "2021-09-30", freq="D")
    pos = np.searchsorted(dates_ref, qc_d)
    site_f = _site_cell_forcings(keys, qc_d)
    up_t = arrays["tmax_c"][:, pos]
    up_p = arrays["precip_mm"][:, pos]
    sf_t = site_f[:, :, D.FORCING_VARS.index("tmax_c")]
    sf_p = site_f[:, :, D.FORCING_VARS.index("precip_mm")]
    b["missing_frac"] = np.isnan(arrays["tmax_c"]).mean(axis=1)
    b["tmax_mean_diff_c"] = np.nanmean(up_t - sf_t, axis=1)
    b["precip_mean_ratio"] = np.nanmean(up_p, axis=1) / np.nanmean(sf_p, axis=1)
    b["area_ratio_vs_streamcat"] = b["basin_area_km2"] / b["streamcat_wsareasqkm"]
    cov = b.drop(columns=[c for c in ("geometry_wkb",) if c in b.columns])
    cov.to_parquet(OUT / f"coverage{SFX}.parquet", compression="zstd", index=False)
    small = cov["n_cells"] <= 1
    stats = {
        "n_sites": len(keys), "n_basins_ok": int((~b["fallback"].astype(bool)).sum()), "basin_method": b["method"].value_counts().to_dict(),
        "n_fallback_point": int(cov["fallback"].sum()), "date_start": str(dates_ref[0].date()),
        "date_end": str(dates_ref[-1].date()), "n_rows": int(len(keys) * len(dates_ref)),
        "n_cells_quantiles": cov["n_cells"].quantile([0, .25, .5, .75, 1]).round(1).tolist(),
        "cover_frac_lt_0.95": int((cov["cover_frac"] < 0.95).sum()),
        "area_ratio_vs_streamcat_quantiles": cov["area_ratio_vs_streamcat"].quantile([.01, .1, .5, .9, .99]).round(3).tolist(),
        "tmax_diff_single_cell_abs_max_c": float(np.nanmax(np.abs(cov.loc[small, "tmax_mean_diff_c"]))) if small.any() else None,
        "tmax_diff_quantiles_c": cov["tmax_mean_diff_c"].quantile([.01, .5, .99]).round(3).tolist(),
        "status_counts": status.value_counts().to_dict(), "supersample": U.SUPERSAMPLE,
        "area_ratio_local_vs_nldi_quantiles": (b["basin_area_km2"] / b["nldi_basin_area_km2"]).quantile([.01, .1, .5, .9, .99]).round(3).tolist(),
        "n_nldi_qc": int(b["nldi_basin_area_km2"].notna().sum()), "years": YEARS, "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    write_manifest(NAME + SFX, files=[out_f, OUT / f"basins{SFX}.parquet", OUT / f"coverage{SFX}.parquet"],
                   sources=["https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/Extensions/SimplifiedCatchments/",
                            "raw/nhdplus_vaa (HydroShare 6092c8a62fac45be97a09bfd0b0bf726)", G.BULK_URL,
                            N.BASE + "/comid/{comid}/basin (QC subset only, cached)"],
                   description="Upstream-basin (NHDPlus V2 total drainage basin of the COMID; NLDI polygon where the "
                               "local area disagrees with StreamCat by > 2x) area-weighted daily gridMET forcings for "
                               + ("M6 track-N WQP stations" if SET == "trackn" else "M6-eligible sonde sites")
                               + "; observed weather only.",
                   license="public domain (gridMET CC0; USGS NLDI / NHDPlus V2 public domain)",
                   synthetic=False, stats=stats)
    log.info("combine done: %s", json.dumps(stats, default=str)[:1500])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["basins", "basins_local", "aggregate", "combine", "all"])
    ap.add_argument("--set", default="sonde", choices=["sonde", "trackn"])
    a = ap.parse_args()
    global SET, SFX, WORK
    SET, SFX = a.set, ("_trackn" if a.set == "trackn" else "")
    WORK = INT / "trackn" if a.set == "trackn" else INT
    WORK.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(INT / "build.log")])
    if a.step == "basins":   # NLDI polygons (superseded 2026-11-08 by basins_local: the shared NLDI quota was exhausted)
        step_basins()
    if a.step in ("basins_local", "all"):
        step_basins_local()
    if a.step in ("aggregate", "all"):
        step_aggregate()
    if a.step in ("combine", "all"):
        step_combine()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
