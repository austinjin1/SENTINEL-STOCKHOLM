# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL site registry: one row per monitoring site with a stable ``site_key``.

Sources
-------
* USGS sites: ``raw('usgs_dv')/dv_series_catalog.parquet`` (NWIS series catalog of sites with
  daily water-quality series) + NWIS site service ``siteOutput=expanded`` (time zone, altitude
  datum, drainage area).
* NEON aquatic sites: ``https://data.neonscience.org/api/v0/sites`` (sites whose product list
  contains DP1.20288.001 "Water quality") + ``/locations/<site>`` for habitat, elevation and the
  sensor-station coordinates (wadeable streams: downstream station S2; lakes/rivers: buoy).
* HUC codes missing in NWIS (and for NEON) come from a USGS WBD point query (HUC12 layer).

Schema (``processed('sites')/sites.parquet``)
---------------------------------------------
site_key ('USGS-<site_no>' | 'NEON-<code>'), network, site_id, name, lat, lon, coord_datum,
coord_source, site_type (USGS ``site_tp_cd``; NEON streams/rivers -> 'ST', lakes -> 'LK'),
site_type_detail, state, huc2, huc4, huc8, huc12, huc_source, elev_m, elev_datum, elev_source,
tz_cd, utc_offset_h (standard time), drain_area_km2, in_conus, wq_parm_cds, n_wq_series,
wq_begin, wq_end, has_do, has_sc, has_temp ... All IDs are strings (leading zeros kept).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from sentinel.core import paths

USER_AGENT = "SENTINEL-v2 research (public-data download)"
NEON_API = "https://data.neonscience.org/api/v0"
NEON_WQ_PRODUCT = "DP1.20288.001"
WBD_HUC12_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query"
NWIS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
FT_TO_M = 0.3048  # international foot (exact)
SQMI_TO_KM2 = 2.589988110336  # US survey/international square mile -> km2

# Standard-time UTC offsets (hours) for NWIS tz_cd codes (daylight codes map to standard).
TZ_STD_OFFSET_H = {
    "UTC": 0,
    "GMT": 0,
    "EST": -5,
    "EDT": -5,
    "CST": -6,
    "CDT": -6,
    "MST": -7,
    "MDT": -7,
    "PST": -8,
    "PDT": -8,
    "AKST": -9,
    "AKDT": -9,
    "HST": -10,
    "HDT": -10,
    "AST": -4,
    "ADT": -4,
    "SST": -11,
    "GST": 10,
    "ChST": 10,
}
# NEON aquatic sites lie in single-zone parts of their states; standard offsets by state.
STATE_STD_OFFSET_H = {
    "MA": -5,
    "VA": -5,
    "FL": -5,
    "GA": -5,
    "PR": -4,
    "WI": -6,
    "KS": -6,
    "TN": -5,
    "AL": -6,
    "ND": -6,
    "CO": -7,
    "OK": -6,
    "TX": -6,
    "WY": -7,
    "AZ": -7,
    "UT": -7,
    "WA": -8,
    "OR": -8,
    "CA": -8,
    "AK": -9,
}
NEON_STATE_TZ = {-5: "EST", -6: "CST", -7: "MST", -8: "PST", -9: "AKST", -4: "AST"}

# Parameter codes used for the per-site WQ summary flags.
P_TEMP, P_DO, P_SC, P_PH, P_TURB, P_NO3 = (
    "00010",
    "00300",
    "00095",
    "00400",
    ("63680", "63675"),
    ("99133", "99137"),
)

# CONUS bounding box used only for routing forcings (gridMET vs Open-Meteo); exact coverage
# is decided by the gridMET land mask at extraction time.
CONUS_BBOX = (24.4, 49.5, -125.0, -66.8)  # lat_min, lat_max, lon_min, lon_max


# --------------------------------------------------------------------------- small helpers
def usgs_key(site_no: str) -> str:
    return f"USGS-{site_no}"


def neon_key(code: str) -> str:
    return f"NEON-{code}"


def split_huc(huc: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    """Zero-padded HUC string -> (huc2, huc4, huc8, huc12). Non-digit / short codes -> None."""
    if huc is None or (isinstance(huc, float) and np.isnan(huc)):
        return None, None, None, None
    h = str(huc).strip()
    if not h.isdigit() or len(h) < 2:
        return None, None, None, None
    if len(h) % 2 == 1:  # malformed odd-length code: do not guess
        return None, None, None, None
    return (
        h[:2],
        h[:4] if len(h) >= 4 else None,
        h[:8] if len(h) >= 8 else None,
        h[:12] if len(h) >= 12 else None,
    )


def in_conus(lat: float, lon: float) -> bool:
    a, b, c, d = CONUS_BBOX
    return bool(a <= lat <= b and c <= lon <= d)


def _cached_get_json(
    url: str,
    params: dict | None,
    cache_dir: Path | None,
    *,
    retries: int = 5,
    sleep: float = 0.3,
    timeout: float = 120,
) -> dict | None:
    """GET JSON with retries; responses cached as gzip JSON (resume-safe). 404 -> None."""
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    cp = (cache_dir / f"{key}.json.gz") if cache_dir else None
    if cp is not None and cp.exists():
        with gzip.open(cp, "rt") as fh:
            return json.load(fh)
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    last = None
    for k in range(retries):
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                body = r.json()
                if cp is not None:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(cp, "wt") as fh:
                        json.dump(body, fh)
                time.sleep(sleep)
                return body
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 60)))
                continue
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.RequestException, ValueError) as e:
            last = repr(e)
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"GET failed: {url} {params} :: {last}")


# --------------------------------------------------------------------------- USGS
def usgs_sites_from_catalog(cat: pd.DataFrame, wq_params: Iterable[str]) -> pd.DataFrame:
    """Collapse the NWIS DV series catalog (one row per series) to one row per site."""
    wq_params = set(wq_params)
    c = cat[(cat["data_type_cd"] == "dv") & cat["parm_cd"].isin(wq_params)].copy()
    c["begin"] = pd.to_datetime(c["begin_date"], errors="coerce")
    c["end"] = pd.to_datetime(c["end_date"], errors="coerce")
    # Every column is a Series indexed by site_no so that pandas aligns by label (an Index or
    # array here would be assigned positionally and silently scramble rows against the
    # groupby results, which are sorted by site_no).
    first = c.drop_duplicates("site_no").set_index("site_no").sort_index()
    g = c.groupby("site_no")
    out = pd.DataFrame(
        {
            "site_id": first.index.to_series().astype(str),
            "agency_cd": first["agency_cd"].astype(str),
            "name": first["station_nm"].astype(str),
            "site_tp_cd": first["site_tp_cd"].astype(str),
            "lat": pd.to_numeric(first["dec_lat_va"], errors="coerce"),
            "lon": pd.to_numeric(first["dec_long_va"], errors="coerce"),
            "coord_acy_cd": first["coord_acy_cd"].astype(str),
            "coord_datum": first["dec_coord_datum_cd"].astype(str),
            "alt_va_catalog": pd.to_numeric(first["alt_va"], errors="coerce"),
            "huc_cd": first["huc_cd"].astype(str),
            "state": first["state"].astype(str),
            "wq_parm_cds": g["parm_cd"].agg(lambda s: ",".join(sorted(set(s)))),
            "n_wq_series": g.size(),
            "wq_begin": g["begin"].min(),
            "wq_end": g["end"].max(),
        }
    ).reset_index(drop=True)
    # alignment guard: every row must still describe its own site (name/coords from the catalog)
    chk = first.loc[out["site_id"]]
    if not (
        (chk["station_nm"].astype(str).to_numpy() == out["name"].to_numpy()).all()
        and np.allclose(
            pd.to_numeric(chk["dec_lat_va"], errors="coerce").to_numpy(),
            out["lat"].to_numpy(),
            equal_nan=True,
        )
    ):
        raise AssertionError("site rows misaligned with catalog")
    # per-parameter overlap windows for the common flags (mean or any stat)
    for flag, pcs in {
        "has_temp": (P_TEMP,),
        "has_do": (P_DO,),
        "has_sc": (P_SC,),
        "has_ph": (P_PH,),
        "has_turb": P_TURB,
        "has_no3": P_NO3,
    }.items():
        sites_with = set(c.loc[c["parm_cd"].isin(pcs), "site_no"])
        out[flag] = out["site_id"].isin(sites_with)
    return out


def fetch_nwis_site_info(site_nos: list[str], cache_dir: Path | None, batch: int = 100) -> pd.DataFrame:
    """NWIS expanded site output (tz_cd, altitude + datum, drainage area) for many sites."""
    from sentinel.data.sources import usgs_legacy as U  # existing NWIS client (RDB parser)

    frames = []
    for b in range(0, len(site_nos), batch):
        chunk = site_nos[b : b + batch]
        key = hashlib.sha1(",".join(chunk).encode()).hexdigest()
        cp = cache_dir / f"nwis_site_{key}.parquet" if cache_dir else None
        if cp is not None and cp.exists():
            frames.append(pd.read_parquet(cp))
            continue
        params = {"format": "rdb", "sites": ",".join(chunk), "siteOutput": "expanded", "siteStatus": "all"}
        df = U.read_rdb(U._get(NWIS_SITE_URL, params))
        keep = [
            k
            for k in [
                "agency_cd",
                "site_no",
                "tz_cd",
                "local_time_fg",
                "alt_va",
                "alt_acy_va",
                "alt_datum_cd",
                "drain_area_va",
                "contrib_drain_area_va",
                "state_cd",
                "county_cd",
                "huc_cd",
                "dec_coord_datum_cd",
                "site_tp_cd",
            ]
            if k in df.columns
        ]
        df = df[keep].copy() if not df.empty else pd.DataFrame(columns=keep)
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cp, index=False)
        frames.append(df)
        time.sleep(0.5)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.drop_duplicates("site_no") if not out.empty else out


# --------------------------------------------------------------------------- NEON
def _neon_props(loc: dict) -> dict:
    return {
        p["locationPropertyName"].replace("Value for ", ""): p["locationPropertyValue"]
        for p in loc.get("locationProperties") or []
    }


def neon_aquatic_sites(cache_dir: Path | None) -> pd.DataFrame:
    """The NEON aquatic sites (product DP1.20288.001) with sensor-station coordinates."""
    sites = _cached_get_json(f"{NEON_API}/sites", None, cache_dir)["data"]
    rows = []
    for s in sites:
        prods = {p.get("dataProductCode") for p in s.get("dataProducts") or []}
        if NEON_WQ_PRODUCT not in prods:
            continue
        code = s["siteCode"]
        loc = _cached_get_json(f"{NEON_API}/locations/{code}", None, cache_dir)["data"]
        props = _neon_props(loc)
        habitat = props.get("HABITAT", "")
        lat, lon, elev = (
            loc["locationDecimalLatitude"],
            loc["locationDecimalLongitude"],
            loc["locationElevation"],
        )
        coord_source, sensor_name = "NEON site location", None
        # water body child (STREAMxxxx / LAKExxxx / RIVERxxxx) -> sensor station / buoy
        for child in loc.get("locationChildren") or []:
            if not child.startswith(("STREAM", "LAKE", "RIVER")):
                continue
            wb = _cached_get_json(f"{NEON_API}/locations/{child}", None, cache_dir)["data"]
            kids = wb.get("locationChildren") or []
            pref = ["S2LOC", "BUOY", "S1LOC"] if child.startswith("STREAM") else ["BUOY", "S2LOC", "S1LOC"]
            for p in pref:
                hit = [k for k in kids if k.startswith(p)]
                if hit:
                    st = _cached_get_json(f"{NEON_API}/locations/{hit[0]}", None, cache_dir)["data"]
                    lat, lon = st["locationDecimalLatitude"], st["locationDecimalLongitude"]
                    elev = st.get("locationElevation", elev)
                    coord_source = {
                        "S2LOC": "NEON downstream sensor station (S2)",
                        "BUOY": "NEON buoy",
                        "S1LOC": "NEON upstream sensor station (S1)",
                    }[p]
                    sensor_name = hit[0]
                    break
            break
        rows.append(
            {
                "site_id": code,
                "name": s.get("siteName") or s.get("siteDescription"),
                "lat": float(lat),
                "lon": float(lon),
                "coord_source": coord_source,
                "coord_datum": props.get("Geodetic datum", "WGS84"),
                "site_lat": s.get("siteLatitude"),
                "site_lon": s.get("siteLongitude"),
                "sensor_location": sensor_name,
                "habitat": habitat,
                "state": s.get("stateCode"),
                "domain": s.get("domainCode"),
                "elev_m": None if elev is None else float(elev),
                "neon_site_type": s.get("siteType"),
            }
        )
    return pd.DataFrame(rows)


def wbd_huc12(lon: float, lat: float, cache_dir: Path | None) -> str | None:
    """HUC12 polygon containing a point (USGS WBD MapServer layer 6)."""
    params = {
        "geometry": f"{lon:.6f},{lat:.6f}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "huc12",
        "returnGeometry": "false",
        "f": "json",
    }
    body = _cached_get_json(WBD_HUC12_URL, params, cache_dir)
    feats = (body or {}).get("features") or []
    return str(feats[0]["attributes"]["huc12"]) if feats else None


# --------------------------------------------------------------------------- registry
REGISTRY_COLUMNS = [
    "site_key",
    "network",
    "site_id",
    "name",
    "lat",
    "lon",
    "coord_datum",
    "coord_source",
    "site_type",
    "site_type_detail",
    "state",
    "huc2",
    "huc4",
    "huc8",
    "huc12",
    "huc_source",
    "elev_m",
    "elev_datum",
    "elev_source",
    "tz_cd",
    "utc_offset_h",
    "drain_area_km2",
    "in_conus",
    "wq_parm_cds",
    "n_wq_series",
    "wq_begin",
    "wq_end",
    "has_temp",
    "has_do",
    "has_sc",
    "has_ph",
    "has_turb",
    "has_no3",
]


def build_registry(
    usgs: pd.DataFrame, nwis_info: pd.DataFrame | None, neon: pd.DataFrame | None, wbd_lookup=None
) -> pd.DataFrame:
    """Combine USGS + NEON rows into the registry schema.

    ``wbd_lookup(lon, lat) -> huc12|None`` fills missing HUCs (None disables the lookup).
    """
    u = usgs.copy()
    info = (
        nwis_info.set_index("site_no")
        if nwis_info is not None and not nwis_info.empty
        else pd.DataFrame(index=pd.Index([], name="site_no"))
    )

    def col(name):
        return u["site_id"].map(info[name]) if name in info.columns else pd.Series(None, index=u.index)

    alt_ft = pd.to_numeric(col("alt_va"), errors="coerce")
    alt_ft = alt_ft.where(alt_ft.notna(), u["alt_va_catalog"])
    tz = col("tz_cd").replace("", None)
    da_sqmi = pd.to_numeric(col("drain_area_va"), errors="coerce")
    hucs = u["huc_cd"].where(u["huc_cd"].str.fullmatch(r"\d{8}|\d{10}|\d{12}"), None)
    reg_u = pd.DataFrame(
        {
            "site_key": u["site_id"].map(usgs_key),
            "network": "USGS",
            "site_id": u["site_id"],
            "name": u["name"],
            "lat": u["lat"],
            "lon": u["lon"],
            "coord_datum": u["coord_datum"],
            "coord_source": "NWIS dec_lat_va/dec_long_va (acy " + u["coord_acy_cd"].fillna("") + ")",
            "site_type": u["site_tp_cd"],
            "site_type_detail": u["site_tp_cd"],
            "state": u["state"],
            "huc_raw": hucs,
            "huc_source": np.where(hucs.notna(), "nwis_huc_cd", None),
            "elev_m": (alt_ft * FT_TO_M).round(2),
            "elev_datum": col("alt_datum_cd").replace("", None),
            "elev_source": np.where(alt_ft.notna(), "NWIS alt_va (ft->m)", None),
            "tz_cd": tz,
            "utc_offset_h": tz.map(TZ_STD_OFFSET_H),
            "drain_area_km2": (da_sqmi * SQMI_TO_KM2).round(3),
            "wq_parm_cds": u["wq_parm_cds"],
            "n_wq_series": u["n_wq_series"],
            "wq_begin": u["wq_begin"],
            "wq_end": u["wq_end"],
            **{f: u[f] for f in ["has_temp", "has_do", "has_sc", "has_ph", "has_turb", "has_no3"]},
        }
    )
    frames = [reg_u]
    if neon is not None and not neon.empty:
        n = neon.copy()
        hab = n["habitat"].fillna("")
        off = n["state"].map(STATE_STD_OFFSET_H)
        frames.append(
            pd.DataFrame(
                {
                    "site_key": n["site_id"].map(neon_key),
                    "network": "NEON",
                    "site_id": n["site_id"],
                    "name": n["name"],
                    "lat": n["lat"],
                    "lon": n["lon"],
                    "coord_datum": n["coord_datum"],
                    "coord_source": n["coord_source"],
                    "site_type": np.where(hab.str.contains("Lake"), "LK", "ST"),
                    "site_type_detail": hab,
                    "state": n["state"],
                    "huc_raw": None,
                    "huc_source": None,
                    "elev_m": n["elev_m"],
                    "elev_datum": "NEON locationElevation",
                    "elev_source": "NEON locations API",
                    "tz_cd": off.map(NEON_STATE_TZ),
                    "utc_offset_h": off,
                    "drain_area_km2": np.nan,
                    "wq_parm_cds": NEON_WQ_PRODUCT,
                    "n_wq_series": np.nan,
                    "wq_begin": pd.NaT,
                    "wq_end": pd.NaT,
                    **{f: False for f in ["has_temp", "has_do", "has_sc", "has_ph", "has_turb", "has_no3"]},
                }
            )
        )
    reg = pd.concat(frames, ignore_index=True)
    if wbd_lookup is not None:
        miss = reg["huc_raw"].isna() & reg["lat"].notna()
        for i in reg.index[miss]:
            h = wbd_lookup(float(reg.at[i, "lon"]), float(reg.at[i, "lat"]))
            if h:
                reg.at[i, "huc_raw"] = h
                reg.at[i, "huc_source"] = "wbd_point_huc12"
    parts = reg["huc_raw"].map(split_huc)
    reg["huc2"] = parts.map(lambda t: t[0])
    reg["huc4"] = parts.map(lambda t: t[1])
    reg["huc8"] = parts.map(lambda t: t[2])
    reg["huc12"] = parts.map(lambda t: t[3])
    reg["in_conus"] = [
        in_conus(a, b) if pd.notna(a) and pd.notna(b) else False for a, b in zip(reg["lat"], reg["lon"])
    ]
    reg = reg.drop(columns=["huc_raw"])
    for c in ["site_key", "site_id", "huc2", "huc4", "huc8", "huc12", "state", "tz_cd"]:
        reg[c] = reg[c].astype("string")
    reg["n_wq_series"] = pd.to_numeric(reg["n_wq_series"], errors="coerce").astype("Int64")
    if reg["site_key"].duplicated().any():
        raise ValueError("duplicate site_key in registry")
    return reg[REGISTRY_COLUMNS].sort_values("site_key").reset_index(drop=True)


def load_sites(path: Path | None = None) -> pd.DataFrame:
    """Read the built registry (``processed('sites')/sites.parquet``)."""
    return pd.read_parquet(path or (paths.PROCESSED / "sites" / "sites.parquet"))


# --------------------------------------------------------------------------- flow network
# Builders for processed('site_network'): site -> COMID nodes with NHDPlus V2 VAA attributes and
# directed flow-connection edges (src upstream -> dst downstream). Used by scripts/data/download_nldi.py.
VAA_NODE_COLS = [
    "hydroseq",
    "dnhydroseq",
    "levelpathi",
    "terminalpa",
    "pathlength",
    "lengthkm",
    "totdasqkm",
    "streamorde",
    "slope",
    "pathtimema",
    "ftype",
    "vpuid",
]


def attach_vaa(links: pd.DataFrame, vaa: pd.DataFrame) -> pd.DataFrame:
    """Add ``vaa_*`` reach attributes and ``pos_km_to_outlet`` (main-path distance to the outlet)."""
    from sentinel.data.sources.nldi import network_position_km

    v = vaa.drop_duplicates("comid").set_index("comid")
    n = links.copy()
    cid = pd.to_numeric(n["comid"], errors="coerce")
    has = cid.notna() & cid.isin(v.index)
    sel = v.reindex(cid[has].astype("int64"))
    for c in VAA_NODE_COLS:
        n["vaa_" + c] = pd.Series(sel[c].to_numpy(), index=n.index[has]).reindex(n.index)
    for c in ["vaa_hydroseq", "vaa_dnhydroseq", "vaa_levelpathi", "vaa_terminalpa"]:
        n[c] = pd.to_numeric(n[c], errors="coerce").astype("Int64")
    for c in [
        "vaa_pathlength",
        "vaa_lengthkm",
        "vaa_totdasqkm",
        "vaa_streamorde",
        "vaa_slope",
        "vaa_pathtimema",
    ]:
        n[c] = pd.to_numeric(n[c], errors="coerce")
    pos, meth = [], []
    for pl, ln, ms in zip(n["vaa_pathlength"], n["vaa_lengthkm"], n["measure"]):
        if pd.isna(pl):
            pos.append(np.nan)
            meth.append(None)
        else:
            p, m = network_position_km(float(pl), float(ln), None if pd.isna(ms) else float(ms))
            pos.append(p)
            meth.append(m)
    n["pos_km_to_outlet"] = pos
    n["pos_method"] = meth
    return n


def vaa_downstream_edges(
    nodes: pd.DataFrame, vaa: pd.DataFrame, distance_km: float, eligible: set | None = None
) -> pd.DataFrame:
    """Sites on each site's downstream main path (VAA ``dnhydroseq`` chain) within ``distance_km``.

    On the site's own flowline only sites further downstream (smaller position) count.
    """
    hs = vaa.drop_duplicates("hydroseq").set_index("hydroseq")
    dn, hs_comid, hs_pl = hs["dnhydroseq"].to_dict(), hs["comid"].to_dict(), hs["pathlength"].to_dict()
    n = nodes[nodes["vaa_hydroseq"].notna()]
    if eligible is not None:
        n = n[n["site_key"].isin(eligible)]
    by_comid: dict[int, list] = {}
    for key, c, p in zip(n["site_key"], n["comid"], n["pos_km_to_outlet"]):
        by_comid.setdefault(int(c), []).append((key, p))
    rows = []
    for key, h0, start in zip(n["site_key"], n["vaa_hydroseq"], n["pos_km_to_outlet"]):
        h, first, steps = int(h0), True, 0
        while h and h in hs_comid and steps < 50000:
            for other, opos in by_comid.get(int(hs_comid[h]), []):
                if other == key or (first and not opos < start):
                    continue
                d = start - opos
                if 0 <= d <= distance_km:
                    rows.append({"src": key, "dst": other, "distance_km": d})
            if start - hs_pl[h] > distance_km:
                break
            h = int(dn.get(h, 0) or 0)
            first = False
            steps += 1
    return pd.DataFrame(rows, columns=["src", "dst", "distance_km"])


EDGE_COLUMNS = [
    "src",
    "dst",
    "distance_km",
    "found_by",
    "src_comid",
    "dst_comid",
    "distance_method",
    "direction_check",
    "travel_time_days",
]


def combine_edges(
    nodes: pd.DataFrame, nav: pd.DataFrame | None, vaa_edges: pd.DataFrame | None, site_keys: set
) -> pd.DataFrame:
    """Merge NLDI navigation hits (UM: hit upstream of query; DM: hit downstream) and VAA edges.

    ``found_by`` lists every method that found the pair (e.g. ``nldi_DM+nldi_UM+vaa_dm``);
    ``distance_km`` = |position difference| when both sites lie on the same terminal network;
    ``direction_check`` = 'reversed' when the position contradicts the navigation direction;
    ``travel_time_days`` = difference of VAA ``pathtimema`` (mean-annual-flow travel time).
    """
    rows = []
    if nav is not None and len(nav):
        nv = nav[nav["identifier"].isin(site_keys) & (nav["identifier"] != nav["query_site_key"])]
        up = nv["mode"].eq("UM")
        rows.append(
            pd.DataFrame(
                {
                    "src": np.where(up, nv["identifier"], nv["query_site_key"]),
                    "dst": np.where(up, nv["query_site_key"], nv["identifier"]),
                    "found_by": "nldi_" + nv["mode"],
                }
            )
        )
    if vaa_edges is not None and len(vaa_edges):
        rows.append(vaa_edges.assign(found_by="vaa_dm")[["src", "dst", "found_by"]])
    if not rows:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    e = pd.concat(rows, ignore_index=True)
    g = e.groupby(["src", "dst"])["found_by"].agg(lambda s: "+".join(sorted(set(s)))).reset_index()
    ni = nodes.set_index("site_key")
    ps, pd_ = g["src"].map(ni["pos_km_to_outlet"]), g["dst"].map(ni["pos_km_to_outlet"])
    same = g["src"].map(ni["vaa_terminalpa"]).astype("Float64") == g["dst"].map(ni["vaa_terminalpa"]).astype(
        "Float64"
    )
    dist = (ps - pd_).where(same.fillna(False).astype(bool))
    g["distance_km"] = dist.abs().round(3)
    g["direction_check"] = np.where(dist.isna(), "unknown", np.where(dist >= -0.01, "ok", "reversed"))
    mth = ni["pos_method"]
    both_meas = (g["src"].map(mth) == "vaa_pathlength_measure") & (
        g["dst"].map(mth) == "vaa_pathlength_measure"
    )
    g["distance_method"] = np.where(
        dist.isna(), None, np.where(both_meas, "vaa_pathlength_measure", "vaa_pathlength_midpoint")
    )
    tt = g["src"].map(ni["vaa_pathtimema"]) - g["dst"].map(ni["vaa_pathtimema"])
    g["travel_time_days"] = tt.where(same.fillna(False).astype(bool)).round(3)
    g["src_comid"] = g["src"].map(ni["comid"])
    g["dst_comid"] = g["dst"].map(ni["comid"])
    return g[EDGE_COLUMNS].sort_values(["src", "dst"]).reset_index(drop=True)


def flow_components(site_keys: list[str], edges: pd.DataFrame) -> pd.Series:
    """Connected components of the undirected edge graph -> 'FG#####' group id per site."""
    parent = {k: k for k in site_keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, d in zip(edges["src"], edges["dst"]):
        if s in parent and d in parent:
            a, b = find(s), find(d)
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = {k: find(k) for k in site_keys}
    ids = {r: f"FG{i:05d}" for i, r in enumerate(sorted(set(roots.values())))}
    return pd.Series({k: ids[r] for k, r in roots.items()})
