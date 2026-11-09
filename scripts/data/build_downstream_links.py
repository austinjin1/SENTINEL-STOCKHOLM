#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""COMID links + StreamCat attributes for downstream label units (NRSA sites, Iowa fish kills,
Powell / StreamPULSE metabolism gauges).

Stage 1 (`links`): NLDI comid/position snap for every unit without a published COMID (cached,
<= 1 request / 1.25 s; the USGS gateway quota of 3,600/h is shared with other jobs).
Stage 2 (`streamcat`): curated StreamCat metric groups (same list as `processed/catchment`) for the
union of COMIDs, sequential POSTs (API limit 1,000/h).

Outputs: processed('downstream_catchment')/{unit_links.parquet, streamcat_comid.parquet,
streamcat_dictionary.csv}; manifest `downstream_catchment`.

    python -B scripts/data/build_downstream_links.py --stage all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import downstream_links as DL  # noqa: E402
from sentinel.data.sources import nldi as NL  # noqa: E402
from sentinel.data.sources import streamcat as SC  # noqa: E402

log = logging.getLogger("downstream_links")


def stage_links(out_dir: Path, raw: Path) -> pd.DataFrame:
    pts = DL.load_points()
    vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet", columns=["comid"])
    pts = DL.validate_published(pts, set(vaa["comid"].astype("int64").astype(str)))
    log.info("invalid published COMIDs (re-snapped): %s",
             pts.loc[pts["comid_published_invalid"].notna()].groupby("unit_source").size().to_dict())
    need = pts[pts["comid_published"].isna()]
    log.info("%d units, %d with a published COMID, %d to snap", len(pts), len(pts) - len(need), len(need))
    client = NL.NLDIClient(raw / "nldi_cache", min_interval=1.25)
    rows = []
    for i, r in enumerate(need.itertuples(index=False)):
        url, params = NL.position_request(float(r.lon), float(r.lat))
        rec = client.get(url, params)
        rows.append({"unit_source": r.unit_source, "unit_id": r.unit_id, **DL.parse_position_snap(rec)})
        if i % 200 == 0:
            log.info("snap %d/%d stats=%s", i, len(need), client.stats)
    snaps = pd.DataFrame(rows, columns=["unit_source", "unit_id", "comid_snap", "snap_status"])
    links = DL.resolve_comids(pts, snaps)
    links.to_parquet(out_dir / "unit_links.parquet", index=False, compression="zstd")
    (raw / "nldi_stats.json").write_text(json.dumps(client.stats, indent=1))
    log.info("links: %s", links.groupby(["unit_source", "comid_method"]).size().to_dict())
    return links


def stage_streamcat(out_dir: Path, raw: Path, batch: int = 500) -> pd.DataFrame:
    links = pd.read_parquet(out_dir / "unit_links.parquet")
    comids = sorted(links["comid"].dropna().astype(str).unique(), key=int)
    log.info("%d unique COMIDs for StreamCat", len(comids))
    client = SC.StreamCatClient(raw / "streamcat_cache")
    vi_p = paths.RAW / "streamcat" / "variable_info.csv"
    varinfo = pd.read_csv(vi_p)
    known_bad = set()
    sc_stats = paths.MANIFESTS / "streamcat.json"
    if sc_stats.exists():
        st = json.loads(sc_stats.read_text()).get("stats", {})
        known_bad = {(f["metric"], f["aoi"]) for f in st.get("failed_metrics", [])}
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
        keep = [c for c in df.columns if c not in wide.columns]
        wide = wide.join(df[keep], how="left")
        metric_cols += [c for c in SC.expected_columns(names, aoi) if c in wide.columns]
        log.info("group %s/%s -> %d cols (quota left %s)", group, aoi, len(keep), client.stats["ratelimit_remaining"])
    wide = wide.reset_index()
    vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet",
                          columns=["comid", "slope", "streamorde", "totdasqkm", "lengthkm"])
    vaa["comid"] = vaa["comid"].astype("int64").astype(str)
    vaa = vaa.rename(columns={"slope": "nhd_slope", "streamorde": "nhd_streamorde",
                              "totdasqkm": "nhd_totdasqkm", "lengthkm": "nhd_lengthkm"})
    vaa.loc[vaa["nhd_slope"] < 0, "nhd_slope"] = float("nan")
    wide = wide.merge(vaa, on="comid", how="left")
    metric_cols = list(dict.fromkeys(metric_cols))
    wide["streamcat_found"] = wide[metric_cols].notna().any(axis=1)
    pct = [c for c in wide.columns if "pctfull" in c]
    wide.drop(columns=pct).to_parquet(out_dir / "streamcat_comid.parquet", index=False, compression="zstd")
    area = [c for c in ["catareasqkm", "wsareasqkm"] if c in wide.columns]
    SC.column_dictionary(area + metric_cols, varinfo).to_csv(out_dir / "streamcat_dictionary.csv", index=False)
    log.info("streamcat: %d comids, %d found, %d metric cols", len(wide), int(wide["streamcat_found"].sum()), len(metric_cols))
    return wide


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["links", "streamcat", "all"], default="all")
    ap.add_argument("--batch", type=int, default=500)
    a = ap.parse_args()
    raw = paths.raw("downstream_links")
    out_dir = paths.processed("downstream_catchment")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "build.log"), logging.StreamHandler()])
    if a.stage in ("links", "all"):
        stage_links(out_dir, raw)
    if a.stage in ("streamcat", "all"):
        stage_streamcat(out_dir, raw, a.batch)
    links = pd.read_parquet(out_dir / "unit_links.parquet")
    files = [out_dir / "unit_links.parquet"]
    stats = {"links": {f"{k[0]}|{k[1]}": int(v) for k, v in links.groupby(["unit_source", "comid_method"]).size().items()}}
    if (out_dir / "streamcat_comid.parquet").exists():
        sc = pd.read_parquet(out_dir / "streamcat_comid.parquet", columns=["comid", "streamcat_found"])
        files += [out_dir / "streamcat_comid.parquet", out_dir / "streamcat_dictionary.csv"]
        stats.update({"n_comids": int(len(sc)), "n_streamcat_found": int(sc["streamcat_found"].sum())})
    write_manifest("downstream_catchment", files=files,
                   sources=[NL.BASE + "/comid/position", SC.METRICS_URL + " (POST)"],
                   description="COMID links (published or NLDI position snap) and curated StreamCat attributes "
                               "for downstream label units (NRSA sites, Iowa fish kills, Powell/StreamPULSE gauges)",
                   license="US public domain (EPA StreamCat, USGS NLDI/NHDPlus V2)", synthetic=False, stats=stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
