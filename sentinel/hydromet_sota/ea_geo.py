# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Local river-network geography for the EA England track (HydroSHEDS v1: HydroRIVERS + HydroBASINS; Lehner & Grill
2013; RiverATLAS/HydroATLAS v1.0, Linke et al. 2019, CC BY 4.0).

* :func:`snap_points` — nearest HydroRIVERS reach (projected metres; British National Grid for England, an equal-area
  CRS elsewhere) with the snap distance;
* :func:`downstream_chain` — reach → reaches on its ``NEXT_DOWN`` chain (to the outlet);
* :func:`basin_groups` — the US split's rule transposed: a group is a HydroBASINS unit (level 7 by default) merged
  (union-find) with every other unit holding a station flow-connected to one of its stations (one station on the other's
  downstream chain). Upstream/downstream neighbours can never sit in different groups.
* :func:`upstream_reaches` — reach → all reaches draining to it (reverse ``NEXT_DOWN`` graph).

Pure functions; I/O lives in ``scripts/data/sota_hydromet_ea_build.py``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def snap_points(points, rivers, crs: str = "EPSG:27700", max_m: float = 5000.0) -> pd.DataFrame:
    """``points``: GeoDataFrame (any CRS) with index = station; ``rivers``: GeoDataFrame with ``HYRIV_ID``.
    Returns per station the nearest reach id and the distance (m); NaN id beyond ``max_m``."""
    import geopandas as gpd

    p = points.to_crs(crs)
    r = rivers[["HYRIV_ID", "geometry"]].to_crs(crs)
    j = gpd.sjoin_nearest(p, r, how="left", max_distance=max_m, distance_col="snap_m")
    j = j[~j.index.duplicated(keep="first")]
    return pd.DataFrame({"hyriv_id": j["HYRIV_ID"], "snap_m": j["snap_m"]}, index=p.index)


def downstream_chain(start_ids, next_down: dict, max_steps: int = 100000) -> dict:
    out = {}
    for s in set(start_ids):
        path, cur, seen = [], next_down.get(s, 0), {s}
        while cur and cur not in seen and len(path) < max_steps:   # NEXT_DOWN = 0 at the outlet
            path.append(cur)
            seen.add(cur)
            cur = next_down.get(cur, 0)
        out[s] = path
    return out


def upstream_reaches(reach_id, up_graph: dict) -> list:
    seen, stack = {reach_id}, [reach_id]
    while stack:
        c = stack.pop()
        for u in up_graph.get(c, ()):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return sorted(seen)


def reverse_graph(next_down: dict) -> dict:
    g = defaultdict(list)
    for a, b in next_down.items():
        if b:
            g[b].append(a)
    return g


def flow_pairs(station_reach: pd.Series, next_down: dict) -> list[tuple]:
    """Pairs (station a, station b) with b's reach on a's downstream chain (or the same reach)."""
    chains = downstream_chain(station_reach.dropna().unique(), next_down)
    by_reach = defaultdict(list)
    for s, r in station_reach.dropna().items():
        by_reach[r].append(s)
    pairs = []
    for s, r in station_reach.dropna().items():
        for o in by_reach[r]:
            if o != s:
                pairs.append((s, o))
        for c in chains.get(r, []):
            for o in by_reach.get(c, ()):
                pairs.append((s, o))
    return pairs


def basin_groups(unit: pd.Series, pairs) -> pd.Series:
    """station -> group id: basin units merged across station flow connections (US split rule transposed)."""
    uf = _UF()
    for u in unit.dropna().unique():
        uf.find(str(u))
    for a, b in pairs:
        ua, ub = unit.get(a), unit.get(b)
        if pd.notna(ua) and pd.notna(ub):
            uf.union(str(ua), str(ub))
    return unit.map(lambda u: uf.find(str(u)) if pd.notna(u) else np.nan).rename("group")


def read_ods_sheet(path, sheet: str) -> pd.DataFrame:
    """Minimal ODS reader (no odfpy): the named sheet of an .ods file as a DataFrame (first row = header)."""
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0", "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
    T = "{%s}" % ns["table"]
    rows = []
    with zipfile.ZipFile(path) as z, z.open("content.xml") as fh:
        in_sheet = False
        for ev, el in ET.iterparse(fh, events=("start", "end")):
            if ev == "start" and el.tag == T + "table":
                in_sheet = el.get(T + "name") == sheet
            elif ev == "end" and el.tag == T + "table-row" and in_sheet:
                cells = []
                for c in el.findall(T + "table-cell"):
                    rep = min(int(c.get(T + "number-columns-repeated", "1")), 200)
                    txt = "".join("".join(p.itertext()) for p in c.findall("{%s}p" % ns["text"]))
                    cells += [txt] * rep
                while cells and cells[-1] == "":
                    cells.pop()
                if cells:
                    rows.append(cells)
                el.clear()
            elif ev == "end" and el.tag == T + "table":
                if in_sheet:
                    break
    if not rows:
        return pd.DataFrame()
    hdr = rows[0]
    return pd.DataFrame([r + [""] * (len(hdr) - len(r)) for r in rows[1:]], columns=hdr + [f"_x{i}" for i in range(
        max(0, max(len(r) for r in rows[1:]) - len(hdr)))] if len(rows) > 1 else hdr)


def near_discharge(stations, dcp, rivers, snap_st: pd.Series, *, euclid_m: float = 1000.0, river_km: float = 5.0,
                   crs: str = "EPSG:27700") -> pd.DataFrame:
    """Per station: distance (m) to the nearest discharge point, and whether a discharge point snapped (≤ 1 km) to a
    reach upstream of the station's reach lies within ``river_km`` river distance."""
    import geopandas as gpd

    st = stations.to_crs(crs)
    dp = dcp.to_crs(crs)
    nn = gpd.sjoin_nearest(st[["geometry"]], dp[["geometry"]], how="left", distance_col="dcp_dist_m")
    nn = nn[~nn.index.duplicated(keep="first")]
    dsn = snap_points(dcp, rivers, crs=crs, max_m=1000.0)
    nd = dict(zip(rivers["HYRIV_ID"], rivers["NEXT_DOWN"]))
    dist_dn = dict(zip(rivers["HYRIV_ID"], rivers["DIST_DN_KM"]))
    chains = downstream_chain(dsn["hyriv_id"].dropna().unique(), nd)
    below = defaultdict(list)   # reach -> river distances (km) to discharge points upstream of it
    for r in dsn["hyriv_id"].dropna():
        d0 = dist_dn.get(r, np.nan)
        for c in [r] + chains.get(r, []):
            dd = d0 - dist_dn.get(c, np.nan)
            if np.isfinite(dd) and dd <= river_km:
                below[c].append(dd)
            elif np.isfinite(dd) and dd > river_km:
                break
    out = pd.DataFrame(index=st.index)
    out["dcp_dist_m"] = nn["dcp_dist_m"].reindex(st.index).to_numpy()
    out["dcp_upstream_within_km"] = [bool(below.get(r)) if pd.notna(r) else False for r in snap_st.reindex(st.index)]
    out["near_discharge"] = (out["dcp_dist_m"] <= euclid_m) | out["dcp_upstream_within_km"]
    return out
