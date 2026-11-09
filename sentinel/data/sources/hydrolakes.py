# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""HydroLAKES v1.0 lake polygons (Messager et al. 2016, Nat. Commun. 7:13603; doi:10.1038/ncomms13603).

Used by SENTINEL-Orbit (M2) to give every in-situ point a *water-body identity* so that whole lakes
(including very large ones such as Lake Erie) are held out together (RESEARCH_PLAN.md §5.2, §14.2).

* Source: https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip (CC BY 4.0).
* Coverage: global lakes >= 10 ha. We keep the North America and Europe bounding boxes.
* Assignment rule (:func:`assign_points`): a point belongs to the lake whose polygon contains it;
  otherwise to the nearest lake polygon within ``max_dist_m`` (shore stations, polygon generalisation);
  otherwise to none. Distances are computed in a local equal-distance projection per point batch
  (EPSG:5070 for North America, EPSG:3035 for Europe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

URL = "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"
DOI = "10.1038/ncomms13603"
LICENSE = "CC BY 4.0"
SHP_IN_ZIP = "HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"
KEEP_COLS = ("Hylak_id", "Lake_name", "Country", "Continent", "Lake_area", "Shore_len", "Depth_avg",
             "Vol_total", "Elevation", "Wshd_area", "Pour_long", "Pour_lat")
REGIONS = {  # lon_min, lat_min, lon_max, lat_max ; projected CRS for distance work
    "north_america": ((-170.0, 14.0, -50.0, 72.0), "EPSG:5070"),
    "europe": ((-25.0, 34.0, 45.0, 72.0), "EPSG:3035"),
}


def read_region(zip_path, region: str):
    """Read HydroLAKES polygons inside a region bounding box (geopandas GeoDataFrame, EPSG:4326)."""
    import pyogrio

    bbox, _ = REGIONS[region]
    gdf = pyogrio.read_dataframe(f"zip://{zip_path}!{SHP_IN_ZIP}", bbox=bbox, columns=list(KEEP_COLS))
    gdf["region"] = region
    gdf["Hylak_id"] = gdf["Hylak_id"].astype("int64")
    return gdf


def region_of(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    out = np.full(lon.shape, "", dtype=object)
    for name, ((x0, y0, x1, y1), _) in REGIONS.items():
        m = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1) & (out == "")
        out[m] = name
    return out


def assign_points(lon, lat, lakes, *, max_dist_m: float = 100.0) -> pd.DataFrame:
    """Lake assignment for points. Returns a DataFrame (one row per input point, same order) with
    ``hylak_id`` (Int64, <NA> if none), ``lake_dist_m`` (0 inside), ``lake_area_km2``, ``lake_name``,
    ``lake_pour_lon/lat``. ``lakes`` is a GeoDataFrame from :func:`read_region` (any regions)."""
    import geopandas as gpd

    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    n = lon.size
    out = pd.DataFrame({"hylak_id": pd.array([pd.NA] * n, dtype="Int64"), "lake_dist_m": np.nan})
    reg = region_of(lon, lat)
    for name, (_, crs) in REGIONS.items():
        sel = np.flatnonzero(reg == name)
        if not sel.size:
            continue
        lk = lakes[lakes["region"] == name][["Hylak_id", "geometry"]]
        if lk.empty:
            continue
        pts = gpd.GeoDataFrame({"i": sel}, geometry=gpd.points_from_xy(lon[sel], lat[sel]), crs="EPSG:4326")
        pts_p = pts.to_crs(crs)
        lk_p = lk.to_crs(crs)
        j = gpd.sjoin_nearest(pts_p, lk_p, how="left", max_distance=max_dist_m, distance_col="dist")
        # ties (equidistant / overlapping polygons): keep the smallest Hylak_id deterministically
        j = j.sort_values(["i", "dist", "Hylak_id"]).drop_duplicates("i")
        ok = j["Hylak_id"].notna()
        out.loc[j.loc[ok, "i"].to_numpy(), "hylak_id"] = j.loc[ok, "Hylak_id"].astype("int64").to_numpy()
        out.loc[j.loc[ok, "i"].to_numpy(), "lake_dist_m"] = j.loc[ok, "dist"].to_numpy()
    attrs = lakes.drop_duplicates("Hylak_id").set_index("Hylak_id")
    hid = out["hylak_id"]
    has = hid.notna().to_numpy()
    ids = hid[has].astype("int64").to_numpy()
    out["lake_area_km2"] = np.nan
    out["lake_name"] = None
    out["lake_pour_lon"] = np.nan
    out["lake_pour_lat"] = np.nan
    if has.any():
        a = attrs.loc[ids]
        out.loc[has, "lake_area_km2"] = a["Lake_area"].to_numpy()
        out.loc[has, "lake_name"] = a["Lake_name"].to_numpy()
        out.loc[has, "lake_pour_lon"] = a["Pour_long"].to_numpy()
        out.loc[has, "lake_pour_lat"] = a["Pour_lat"].to_numpy()
    return out
