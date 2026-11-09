# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""COMID links and StreamCat attributes for downstream-application label units.

The SENTINEL StreamCat table (`processed/catchment/streamcat_sites.parquet`) covers only registry sites
(USGS/NEON). Downstream labels live elsewhere: NRSA visits (D5 BioIntegrity), Iowa fish-kill points
(D3 KillAlert), Powell/StreamPULSE metabolism gauges (D8 RiverPulse). This module gathers those points,
keeps a published COMID where one exists (NRSA 2013-14/2018-19 site files, Powell `nhdplus_id`,
StreamPULSE), and otherwise snaps lat/lon to an NHDPlus V2 catchment with NLDI `comid/position`
(point-in-catchment; method recorded per row). Pure functions only; HTTP lives in
`sentinel.data.sources.nldi` / `.streamcat`, the driver is `scripts/data/build_downstream_links.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import paths

POINT_COLUMNS = ["unit_source", "unit_id", "lat", "lon", "comid_published", "comid_published_source"]


def _clean_comid(s: pd.Series) -> pd.Series:
    """COMIDs as strings without float artifacts; '', '0', 'nan' -> NA."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.where(~out.isin(["", "0", "nan", "<NA>", "None"]), pd.NA)


def nrsa_points(visits: pd.DataFrame) -> pd.DataFrame:
    """One row per NRSA site (survey-scoped site_id) with its published COMID if any."""
    v = visits.copy()
    v["unit_id"] = v["survey"].astype(str) + ":" + v["site_id"].astype(str)
    g = v.groupby("unit_id").agg(lat=("lat", "median"), lon=("lon", "median"),
                                 comid_published=("comid", "first"),
                                 comid_published_source=("comid_source", "first")).reset_index()
    g["unit_source"] = "nrsa"
    g["comid_published"] = _clean_comid(g["comid_published"])
    return g[POINT_COLUMNS]


def iowa_points(events: pd.DataFrame) -> pd.DataFrame:
    e = events[events["latitude"].notna() & events["longitude"].notna()]
    out = pd.DataFrame({"unit_source": "iowa_fishkill", "unit_id": e["event_id"].astype(str),
                        "lat": e["latitude"].astype(float), "lon": e["longitude"].astype(float),
                        "comid_published": pd.Series(pd.NA, index=e.index, dtype="string"),
                        "comid_published_source": pd.NA})
    return out[POINT_COLUMNS].reset_index(drop=True)


def powell_points(sites: pd.DataFrame) -> pd.DataFrame:
    s = sites[sites["modeled"].astype(bool)]
    # Bernhardt-2022 disagreement sites are re-snapped (datasheet: 12 sites differ)
    agree = s["comid_agree"].fillna(True).astype(bool) if "comid_agree" in s else True
    comid = _clean_comid(s["comid"]).where(agree, pd.NA)
    out = pd.DataFrame({"unit_source": "powell", "unit_id": s["site_id"].astype(str),
                        "lat": s["latitude"].astype(float), "lon": s["longitude"].astype(float),
                        "comid_published": comid, "comid_published_source": "powell_nhdplus_id"})
    return out[POINT_COLUMNS].reset_index(drop=True)


def streampulse_points(sites: pd.DataFrame, with_data: set | None = None) -> pd.DataFrame:
    """US StreamPULSE sites (optionally only those with daily metabolism rows)."""
    s = sites[sites["in_us"].fillna(False).astype(bool) & sites["latitude"].notna()]
    if with_data is not None:
        s = s[s["site_id"].astype(str).isin(with_data)]
    out = pd.DataFrame({"unit_source": "streampulse", "unit_id": s["site_id"].astype(str),
                        "lat": s["latitude"].astype(float), "lon": s["longitude"].astype(float),
                        "comid_published": _clean_comid(s["comid"]),
                        "comid_published_source": s["comid_source"].astype("string")})
    return out[POINT_COLUMNS].reset_index(drop=True)


def parse_position_snap(rec: dict) -> dict:
    """NLDI comid/position response -> {'comid_snap', 'snap_status'} (parser shared with site_network)."""
    from .sources.nldi import parse_feature

    st = rec.get("status")
    if st != 200 or not rec.get("body"):
        return {"comid_snap": None, "snap_status": f"http_{st}"}
    c = parse_feature(rec["body"]).get("comid")
    return {"comid_snap": c, "snap_status": "ok" if c else "no_feature"}


def resolve_comids(points: pd.DataFrame, snaps: pd.DataFrame | None) -> pd.DataFrame:
    """Final COMID per unit: published if present, else the NLDI position snap."""
    p = points.copy()
    if snaps is not None and len(snaps):
        p = p.merge(snaps[["unit_source", "unit_id", "comid_snap", "snap_status"]],
                    on=["unit_source", "unit_id"], how="left")
    else:
        p["comid_snap"], p["snap_status"] = pd.NA, pd.NA
    pub = p["comid_published"].notna()
    p["comid"] = p["comid_published"].where(pub, p["comid_snap"]).astype("string")
    p["comid_method"] = np.where(pub, "published:" + p["comid_published_source"].astype(str),
                                 np.where(p["comid_snap"].notna(), "nldi_position", "unlinked"))
    return p


def validate_published(points: pd.DataFrame, valid_comids: set) -> pd.DataFrame:
    """Published COMIDs that are not digit strings (spreadsheet sci-notation, e.g. '1.03E+11') or not in
    the NHDPlus V2 VAA are dropped (``comid_published_invalid`` keeps the raw value) so the unit is snapped."""
    p = points.copy()
    c = p["comid_published"].astype("string")
    bad = c.notna() & (~c.str.fullmatch(r"\d+").fillna(False) | ~c.isin(valid_comids))
    p["comid_published_invalid"] = c.where(bad, pd.NA)
    p.loc[bad, "comid_published"] = pd.NA
    return p


def load_points() -> pd.DataFrame:
    P = paths.PROCESSED
    frames = [nrsa_points(pd.read_parquet(P / "nars_streams" / "visits.parquet")),
              iowa_points(pd.read_parquet(P / "labels_iowa_fishkill" / "events.parquet")),
              powell_points(pd.read_parquet(P / "labels_powell_metabolism" / "sites.parquet")),
              streampulse_points(pd.read_parquet(P / "labels_streampulse" / "sites.parquet"),
                                 set(pd.read_parquet(P / "labels_streampulse" / "daily_metabolism.parquet",
                                                     columns=["site_id"])["site_id"].astype(str)))]
    pts = pd.concat(frames, ignore_index=True)
    if pts.duplicated(["unit_source", "unit_id"]).any():
        raise AssertionError("duplicate (unit_source, unit_id) in downstream points")
    return pts
