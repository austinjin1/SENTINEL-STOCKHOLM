#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk: COMID links + StreamCat attributes for treating community water systems.

    python -B scripts/data/build_taprisk_links.py --stage points      # service-area representative points (local)
    python -B scripts/data/build_taprisk_links.py --stage links       # NLDI comid/position snap (cached, paced; resumable)
    python -B scripts/data/build_taprisk_links.py --stage streamcat   # curated StreamCat metrics for the snapped COMIDs
    python -B scripts/data/build_taprisk_links.py --stage manifest

Units: CWS (SDWIS ``pws_type_code`` CWS) with an own source (``primary_source_code`` SW, GU or GW — purchased
water excluded: its source catchment is the seller's), active at some time 2010–2024, with an EPA SAB v3
service-area polygon (``labels_cws_boundaries``). Location = polygon representative point (inside the polygon).
SDWIS source-facility coordinates are not public, so — unlike Pennino et al. 2020, who used facility
locations — the NHDPlusV2 catchment is that of the service-area point (documented proxy). Surface systems
(SW, GU) are snapped first. No labels are read here.
Outputs: processed('taprisk_catchment')/{system_points.parquet, system_links.parquet, streamcat_comid.parquet,
streamcat_dictionary.csv}; manifest ``taprisk_catchment``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import downstream_links as DL  # noqa: E402
from sentinel.data.sources import nldi as NL  # noqa: E402
from sentinel.data.sources import streamcat as SC  # noqa: E402

log = logging.getLogger("taprisk_links")
OUT = paths.PROCESSED / "taprisk_catchment"
RAW = paths.RAW / "taprisk_links"
SOURCES = ("SW", "GU", "GW")


def stage_points() -> pd.DataFrame:
    import shapely

    s = pd.read_parquet(paths.PROCESSED / "labels_sdwis" / "sdwis_systems.parquet",
                        columns=["pwsid", "pws_type_code", "primary_source_code", "pws_deactivation_date",
                                 "first_reported_date", "state_code", "population_served"])
    s = s[(s["pws_type_code"] == "CWS") & s["primary_source_code"].isin(SOURCES)].copy()
    deact = pd.to_datetime(s["pws_deactivation_date"], errors="coerce")
    first = pd.to_datetime(s["first_reported_date"], errors="coerce")
    s = s[(deact.isna() | (deact >= "2010-01-01")) & (first.isna() | (first < "2025-01-01"))]
    b = pd.read_parquet(paths.PROCESSED / "labels_cws_boundaries" / "cws_boundaries.parquet",
                        columns=["pwsid", "geometry", "is_modeled", "area_km2", "pwsid_duplicated"])
    b = b[b["pwsid"].isin(set(s["pwsid"]))].drop_duplicates("pwsid")
    g = shapely.from_wkb(b["geometry"].to_numpy())
    ok = ~shapely.is_missing(g)
    g = shapely.make_valid(g)
    rp = shapely.point_on_surface(g)
    b = b.assign(lon=shapely.get_x(rp), lat=shapely.get_y(rp), geom_ok=ok).drop(columns=["geometry"])
    pts = s.merge(b, on="pwsid", how="inner")
    pts["surface"] = pts["primary_source_code"].isin(["SW", "GU"])
    pts = pts.sort_values(["surface", "pwsid"], ascending=[False, True]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(OUT / "system_points.parquet", index=False)
    log.info("points: %d systems (%s)", len(pts), pts["primary_source_code"].value_counts().to_dict())
    return pts


def stage_links(checkpoint: int = 1000) -> pd.DataFrame:
    pts = pd.read_parquet(OUT / "system_points.parquet")
    client = NL.NLDIClient(RAW / "nldi_cache", min_interval=1.25)
    rows = []
    for i, r in enumerate(pts.itertuples(index=False)):
        if not (np.isfinite(r.lon) and np.isfinite(r.lat)):
            rows.append({"pwsid": r.pwsid, "comid_snap": None, "snap_status": "no_point"})
            continue
        url, params = NL.position_request(float(r.lon), float(r.lat))
        rows.append({"pwsid": r.pwsid, **DL.parse_position_snap(client.get(url, params))})
        if (i + 1) % checkpoint == 0:
            pd.DataFrame(rows).to_parquet(OUT / "system_links_partial.parquet", index=False)
            log.info("snap %d/%d stats=%s", i + 1, len(pts), client.stats)
    links = pts[["pwsid", "primary_source_code", "state_code", "lat", "lon", "is_modeled"]].merge(
        pd.DataFrame(rows), on="pwsid", how="left").rename(columns={"comid_snap": "comid"})
    links.to_parquet(OUT / "system_links.parquet", index=False)
    (RAW / "nldi_stats.json").write_text(json.dumps(client.stats, indent=1))
    (OUT / "system_links_partial.parquet").unlink(missing_ok=True)
    log.info("links: %s", links["snap_status"].value_counts().to_dict())
    return links


def stage_streamcat(batch: int = 500) -> pd.DataFrame:
    links = pd.read_parquet(OUT / "system_links.parquet")
    comids = sorted(links["comid"].dropna().astype(str).unique(), key=int)
    client = SC.StreamCatClient(RAW / "streamcat_cache")
    varinfo = pd.read_csv(paths.RAW / "streamcat" / "variable_info.csv")
    known_bad = set()
    sc_stats = paths.MANIFESTS / "streamcat.json"
    if sc_stats.exists():
        known_bad = {(f["metric"], f["aoi"]) for f in json.loads(sc_stats.read_text()).get("stats", {}).get("failed_metrics", [])}
    wide = pd.DataFrame({"comid": comids}).set_index("comid")
    metric_cols = []
    for group, aoi, names in SC.CURATED:
        names = [n for n in names if (n, aoi) not in known_bad]
        if not names:
            continue
        frames = []
        for b in range(0, len(comids), batch):
            url, data = SC.metrics_request(names, comids[b:b + batch], aoi)
            frames.append(SC.parse_metrics(client.post(url, data)["body"]))
        df = pd.concat(frames, ignore_index=True).drop_duplicates("comid").set_index("comid")
        wide = wide.join(df[[c for c in df.columns if c not in wide.columns]], how="left")
        metric_cols += [c for c in SC.expected_columns(names, aoi) if c in wide.columns]
        log.info("group %s/%s done", group, aoi)
    wide = wide.reset_index()
    metric_cols = list(dict.fromkeys(metric_cols))
    wide["streamcat_found"] = wide[metric_cols].notna().any(axis=1)
    wide.drop(columns=[c for c in wide.columns if "pctfull" in c]).to_parquet(OUT / "streamcat_comid.parquet", index=False)
    area = [c for c in ["catareasqkm", "wsareasqkm"] if c in wide.columns]
    SC.column_dictionary(area + metric_cols, varinfo).to_csv(OUT / "streamcat_dictionary.csv", index=False)
    log.info("streamcat: %d comids, %d found, %d cols", len(wide), int(wide["streamcat_found"].sum()), len(metric_cols))
    return wide


def stage_manifest() -> None:
    links = pd.read_parquet(OUT / "system_links.parquet")
    files = [OUT / f for f in ("system_points.parquet", "system_links.parquet", "streamcat_comid.parquet", "streamcat_dictionary.csv")
             if (OUT / f).exists()]
    stats = {"n_systems": int(len(links)), "by_source": links["primary_source_code"].value_counts().to_dict(),
             "snap_status": links["snap_status"].value_counts().to_dict(), "n_unique_comids": int(links["comid"].nunique())}
    write_manifest("taprisk_catchment", files=files, sources=[NL.BASE + "/comid/position", SC.METRICS_URL + " (POST)"],
                   description="D4 TapRisk: NLDI COMID snap of treating CWS service-area points (EPA SAB v3) and curated "
                               "StreamCat attributes of those COMIDs",
                   license="US public domain (EPA SDWIS/SAB/StreamCat, USGS NLDI/NHDPlus V2)", synthetic=False, stats=stats)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["points", "links", "streamcat", "manifest"], required=True)
    a = ap.parse_args()
    {"points": stage_points, "links": stage_links, "streamcat": stage_streamcat, "manifest": stage_manifest}[a.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
