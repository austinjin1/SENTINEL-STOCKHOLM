# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Upstream-area-averaged daily gridMET forcings for SENTINEL-HydroMet (M6) sites.

For every M6-eligible site the total upstream drainage basin of its NHDPlus V2 COMID is taken from
USGS NLDI (``/linked-data/comid/{comid}/basin``, simplified polygon). The basin is rasterised onto the
gridMET 1/24° grid with fractional cell coverage (``SUPERSAMPLE`` × ``SUPERSAMPLE`` sub-cells per cell),
weighted by cell area (cos latitude) and restricted to valid (land) gridMET cells. The daily forcing of
the basin is the weighted mean of the cell values (a sparse matrix product per gridMET yearly file).
Basins smaller than one sub-cell fall back to the gridMET cell containing the polygon's representative
point. Inputs are observed weather only — no in-situ measurement of any kind (M6 rule).

Pure functions; the command-line build is ``scripts/data/build_hydromet_upstream_forcing.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SUPERSAMPLE = 5
BASIN_PARAMS = {"simplified": "true", "splitCatchment": "false"}


def basin_url(base: str, comid: str) -> str:
    return f"{base}/comid/{comid}/basin"


def parse_basin(payload: dict | None):
    """NLDI basin FeatureCollection -> shapely (Multi)Polygon, or None."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if not payload or not payload.get("features"):
        return None
    geoms = [shape(f["geometry"]) for f in payload["features"] if f.get("geometry")]
    geoms = [g if g.is_valid else g.buffer(0) for g in geoms]
    geoms = [g for g in geoms if not g.is_empty]
    if not geoms:
        return None
    return geoms[0] if len(geoms) == 1 else unary_union(geoms)


def geodesic_area_km2(geom) -> float:
    from pyproj import Geod

    return abs(Geod(ellps="WGS84").geometry_area_perimeter(geom)[0]) / 1e6


def grid_geometry(lat: np.ndarray, lon: np.ndarray) -> dict:
    """Regular-grid description from gridMET cell-centre vectors (lat descending)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dlat = float(np.median(np.diff(lat)))
    dlon = float(np.median(np.diff(lon)))
    if not (dlat < 0 < dlon):
        raise ValueError("expected descending latitude and ascending longitude")
    return {"ny": len(lat), "nx": len(lon), "lat0": float(lat[0]) - dlat / 2, "lon0": float(lon[0]) - dlon / 2,
            "dlat": dlat, "dlon": dlon, "lat": lat, "lon": lon}


def basin_cell_weights(geom, grid: dict, valid: np.ndarray, *, ss: int = SUPERSAMPLE) -> tuple[np.ndarray, np.ndarray, dict]:
    """Flat cell indices (i * nx + j) and normalised weights (sum 1) of the basin on ``grid``.

    Weight = covered fraction of the cell (from ss × ss sub-cell centres) × cos(cell latitude), over
    valid cells only. ``info``: n_cells, cover_frac (share of the basin's covered area that lies on
    valid cells), fallback flag."""
    from rasterio import features
    from rasterio.transform import Affine

    ny, nx = grid["ny"], grid["nx"]
    dlat, dlon = grid["dlat"], grid["dlon"]
    minx, miny, maxx, maxy = geom.bounds
    j0 = max(int(math.floor((minx - grid["lon0"]) / dlon)), 0)
    j1 = min(int(math.ceil((maxx - grid["lon0"]) / dlon)), nx)
    i0 = max(int(math.floor((maxy - grid["lat0"]) / dlat)), 0)
    i1 = min(int(math.ceil((miny - grid["lat0"]) / dlat)), ny)
    info = {"fallback": False}
    idx = np.zeros(0, np.int64)
    w = np.zeros(0)
    tot = 0.0
    if j1 > j0 and i1 > i0:
        h, wd = (i1 - i0) * ss, (j1 - j0) * ss
        tr = Affine(dlon / ss, 0, grid["lon0"] + j0 * dlon, 0, dlat / ss, grid["lat0"] + i0 * dlat)
        sub = features.rasterize([(geom, 1)], out_shape=(h, wd), transform=tr, fill=0, dtype="uint8",
                                 all_touched=False)
        frac = sub.reshape(i1 - i0, ss, j1 - j0, ss).sum(axis=(1, 3)) / float(ss * ss)
        ii, jj = np.nonzero(frac)
        coslat = np.cos(np.radians(grid["lat"][ii + i0]))
        area = frac[ii, jj] * coslat
        tot = float(area.sum())
        ok = valid[ii + i0, jj + j0]
        idx = ((ii + i0) * nx + (jj + j0))[ok].astype(np.int64)
        w = area[ok]
    if w.sum() <= 0:  # basin smaller than a sub-cell (or entirely on masked cells): representative point
        p = geom.representative_point()
        i = int(np.clip(round((p.y - grid["lat"][0]) / dlat), 0, ny - 1))
        j = int(np.clip(round((p.x - grid["lon"][0]) / dlon), 0, nx - 1))
        if not valid[i, j]:   # nearest valid cell in a 7 x 7 window
            win = [(a, b) for a in range(max(i - 3, 0), min(i + 4, ny)) for b in range(max(j - 3, 0), min(j + 4, nx))
                   if valid[a, b]]
            if not win:
                return np.zeros(0, np.int64), np.zeros(0), {"fallback": True, "n_cells": 0, "cover_frac": 0.0}
            i, j = min(win, key=lambda ab: (ab[0] - i) ** 2 + (ab[1] - j) ** 2)
        idx, w, tot = np.array([i * nx + j], np.int64), np.array([1.0]), 1.0
        info["fallback"] = True
    info["n_cells"] = int(len(idx))
    info["cover_frac"] = float(w.sum() / tot) if tot > 0 else 0.0
    return idx, w / w.sum(), info


def weight_matrix(rows: list[tuple[np.ndarray, np.ndarray]], n_cells: int):
    """CSR (n_sites, n_cells) from per-site (flat index, weight) pairs."""
    from scipy import sparse

    indptr = np.zeros(len(rows) + 1, np.int64)
    indptr[1:] = np.cumsum([len(r[0]) for r in rows])
    ind = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int64)
    val = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0)
    return sparse.csr_matrix((val, ind, indptr), shape=(len(rows), n_cells))


def aggregate_grid(W, data: np.ndarray) -> np.ndarray:
    """Basin means (n_sites, n_days) of a (n_days, ny, nx) field; NaN cells on a day are dropped
    and the remaining weights renormalised (a basin with no finite cell that day -> NaN)."""
    nd = data.shape[0]
    flat = data.reshape(nd, -1).T                     # (cells, days)
    fin = np.isfinite(flat)
    num = W @ np.where(fin, flat, 0.0).astype("float64")
    den = W @ fin.astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-9, num / den, np.nan)
    return out.astype("float32")


def to_long(site_keys: list[str], dates: pd.DatetimeIndex, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """(site_key, date, <var>...) long table from {var: (n_sites, n_days)}."""
    S, T = len(site_keys), len(dates)
    df = pd.DataFrame({"site_key": np.repeat(np.asarray(site_keys, dtype=object), T),
                       "date": np.tile(dates.date, S)})
    for v, a in arrays.items():
        if a.shape != (S, T):
            raise ValueError(f"{v}: shape {a.shape} != {(S, T)}")
        df[v] = a.reshape(-1)
    return df


# ----------------------------------------------------------------------------- local delineation (no NLDI calls)
# Upstream basin = all NHDPlus V2 catchments whose flowline drains (``dnhydroseq`` chain) to the site's COMID,
# including the site's own catchment; catchment geometry from the NHDPlusV2 simplified catchments (EPA S3).

def catchment_subcells(gdf, grid: dict, *, ss: int = SUPERSAMPLE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise all catchments of one VPU (``gdf`` with column ``idx`` = global catchment index) at ss × the
    gridMET resolution; returns (catchment idx, flat coarse cell, number of covered sub-cells). Catchments that cover
    no sub-cell centre get their representative-point cell with count 1."""
    from rasterio import features
    from rasterio.transform import Affine

    ny, nx = grid["ny"], grid["nx"]
    dlat, dlon = grid["dlat"], grid["dlon"]
    minx, miny, maxx, maxy = gdf.total_bounds
    j0 = max(int(math.floor((minx - grid["lon0"]) / dlon)), 0)
    j1 = min(int(math.ceil((maxx - grid["lon0"]) / dlon)), nx)
    i0 = max(int(math.floor((maxy - grid["lat0"]) / dlat)), 0)
    i1 = min(int(math.ceil((miny - grid["lat0"]) / dlat)), ny)
    h, w = (i1 - i0) * ss, (j1 - j0) * ss
    tr = Affine(dlon / ss, 0, grid["lon0"] + j0 * dlon, 0, dlat / ss, grid["lat0"] + i0 * dlat)
    lab = features.rasterize(zip(gdf.geometry, gdf["idx"].astype("int64") + 1), out_shape=(h, w), transform=tr,
                             fill=0, dtype="int32", all_touched=False)
    r, c = np.nonzero(lab)
    cid = lab[r, c].astype(np.int64) - 1
    cell = ((r // ss) + i0).astype(np.int64) * nx + ((c // ss) + j0)
    key = cid * (ny * nx) + cell
    uk, cnt = np.unique(key, return_counts=True)
    out_c, out_cell, out_n = uk // (ny * nx), uk % (ny * nx), cnt.astype(np.float64)
    missing = np.setdiff1d(gdf["idx"].to_numpy(np.int64), np.unique(out_c))
    if missing.size:
        sub = gdf.set_index("idx").loc[missing]
        p = sub.geometry.representative_point()
        ii = np.clip(np.rint((p.y.to_numpy() - grid["lat"][0]) / dlat), 0, ny - 1).astype(np.int64)
        jj = np.clip(np.rint((p.x.to_numpy() - grid["lon"][0]) / dlon), 0, nx - 1).astype(np.int64)
        out_c = np.r_[out_c, missing]
        out_cell = np.r_[out_cell, ii * nx + jj]
        out_n = np.r_[out_n, np.ones(missing.size)]
    return out_c, out_cell, out_n


def reverse_graph(vaa: pd.DataFrame):
    """CSR adjacency downstream -> upstream over VAA flowlines (index = VAA row order) and comid -> index map."""
    from scipy import sparse

    comid = vaa["comid"].astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(comid)}
    by_hs = pd.Series(np.arange(len(vaa)), index=vaa["hydroseq"].to_numpy())
    dn = by_hs.reindex(vaa["dnhydroseq"].to_numpy()).to_numpy()
    ok = np.isfinite(dn) & (dn != np.arange(len(vaa)))
    src = dn[ok].astype(np.int64)          # downstream node
    dst = np.flatnonzero(ok)               # its upstream neighbour
    g = sparse.csr_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(len(vaa), len(vaa)))
    return g, pos


def upstream_nodes(g, start: int) -> np.ndarray:
    from scipy.sparse.csgraph import breadth_first_order

    return breadth_first_order(g, start, directed=True, return_predecessors=False)


def basin_weights_local(nodes: np.ndarray, C, coslat_flat: np.ndarray, valid_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cell weights of the union of catchments ``nodes`` (rows of CSR ``C``: catchment × cell sub-cell counts)."""
    v = np.asarray(C[nodes].sum(axis=0)).ravel()
    idx = np.flatnonzero(v)
    area = v[idx] * coslat_flat[idx]
    tot = float(area.sum())
    ok = valid_flat[idx]
    idx, w = idx[ok], area[ok]
    info = {"n_cells": int(len(idx)), "cover_frac": float(w.sum() / tot) if tot > 0 else 0.0,
            "n_catchments": int(len(nodes)), "fallback": False}
    if w.sum() <= 0:
        return np.zeros(0, np.int64), np.zeros(0), {**info, "n_cells": 0}
    return idx.astype(np.int64), w / w.sum(), info
