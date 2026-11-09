#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download NY DEC harmful-algal-bloom summaries (data.ny.gov) + NYHABS season reports and build labels.

Raw (one folder per retrieval date = label vintage): $SENTINEL_DATA/raw/ny_habs/<YYYY-MM-DD>/
    qtg2-hjth.csv, 95my-wijm.csv (+ *_metadata.json), nyhabs_<view>.json (+ *_layer.json)
Processed: $SENTINEL_DATA/processed/labels_ny_habs/
    waterbody_years.parquet   published rows, both eras (2012–18 per county)
    lake_years.parquet        one row per lake-year (multi-county rows collapsed)
    nyhabs_reports.parquet    per-report confirmed blooms, accumulated over all raw vintages
Manifest:  $SENTINEL_DATA/manifests/labels_ny_habs.json

    python -B scripts/data/download_events_ny_habs.py                    # today's vintage + build
    python -B scripts/data/download_events_ny_habs.py --vintage 2026-10-22 --skip-download

NYHABS public views only hold the current season; re-run during each season (e.g. weekly) to
archive per-report records — every vintage folder is kept and merged by globalid.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.data.sources import events_common as C
from sentinel.data.sources import events_ny_habs as N

log = logging.getLogger("ny_habs")


def fetch(vdir: Path, with_nyhabs: bool) -> dict:
    info: dict = {"socrata": {}, "nyhabs": {}}
    for ds in N.DATASETS:
        meta = C.socrata_metadata(N.DOMAIN, ds)
        C.save_json(meta, vdir / f"{ds}_metadata.json")
        df = C.socrata_csv(N.DOMAIN, ds, select="*,:id")
        dest = vdir / f"{ds}.csv"
        df.to_csv(dest, index=False)
        n_api = C.socrata_count(N.DOMAIN, ds)
        if n_api != len(df):
            log.warning("%s: downloaded %d rows but API count is %d", ds, len(df), n_api)
        info["socrata"][ds] = {"rows": len(df), "api_count": n_api, **C.socrata_metadata_summary(meta)}
        log.info("%s: %d rows", ds, len(df))
    if with_nyhabs:
        for view, (url, item) in N.NYHABS_LAYERS.items():
            try:
                layer = C.arcgis_layer_info(url)
                C.save_json(layer, vdir / f"nyhabs_{view}_layer.json")
                feats = C.arcgis_query(url, return_geometry=True)
                C.save_json(feats, vdir / f"nyhabs_{view}.json")
                info["nyhabs"][view] = {"url": url, "item": item, "features": len(feats),
                                        "view_definition": layer.get("viewDefinitionQuery"),
                                        "last_edit": (layer.get("editingInfo") or {}).get("lastEditDate")}
                log.info("NYHABS %s: %d features", view, len(feats))
            except C.HTTPFailure as e:
                log.error("NYHABS %s failed: %s", view, e)
                info["nyhabs"][view] = {"url": url, "error": str(e)}
    C.save_json(info, vdir / "fetch_info.json")
    return info


def load_nyhabs_snapshots(root: Path) -> dict:
    snaps = {}
    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = []
        for view in N.NYHABS_LAYERS:
            f = vdir / f"nyhabs_{view}.json"
            if f.exists():
                frames.append(N.nyhabs_frame(json.loads(f.read_text()), view))
        merged = N.merge_nyhabs_views(frames)
        if not merged.empty:
            snaps[vdir.name] = merged
    return snaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vintage", default=None, help="raw folder date (default: today, UTC)")
    ap.add_argument("--skip-download", action="store_true", help="rebuild from an existing raw vintage")
    ap.add_argument("--no-nyhabs", action="store_true", help="skip the NYHABS ArcGIS season views")
    a = ap.parse_args()
    vintage = a.vintage or C.retrieval_date()
    root = paths.raw("ny_habs")
    vdir = paths.ensure(root / vintage)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(root / "download.log"), logging.StreamHandler()])
    log.info("vintage %s (skip_download=%s)", vintage, a.skip_download)
    if not a.skip_download:
        fetch(vdir, not a.no_nyhabs)
    info = json.loads((vdir / "fetch_info.json").read_text())

    max_date = (dt.date.fromisoformat(vintage) + dt.timedelta(days=1)).isoformat()
    raw12 = N.read_socrata_csv(vdir / f"{N.DATASET_2012_2018}.csv")
    raw19 = N.read_socrata_csv(vdir / f"{N.DATASET_2019_ON}.csv")
    wy = N.waterbody_years(raw12, raw19, max_date=max_date)
    ly = N.lake_years(wy)
    snaps = load_nyhabs_snapshots(root)
    reports = N.accumulate_snapshots(snaps)
    out = paths.processed("labels_ny_habs")
    files = [C.write_parquet(wy, out / "waterbody_years.parquet"),
             C.write_parquet(ly, out / "lake_years.parquet")]
    if not reports.empty:
        files.append(C.write_parquet(reports, out / "nyhabs_reports.parquet"))
    for f in files:
        log.info("processed %s", f.name)

    stats = {
        "vintage": vintage, "retrieval_date": vintage, "max_valid_date": max_date,
        "request": {"socrata": {ds: f"https://{N.DOMAIN}/resource/{ds}.csv?$select=*,:id&$order=:id"
                                for ds in N.DATASETS},
                    "nyhabs": {v: u for v, (u, _) in N.NYHABS_LAYERS.items()}},
        "socrata": info.get("socrata"), "nyhabs_fetch": info.get("nyhabs"),
        "waterbody_years": {**C.coverage_stats(wy, date_col="first_report_date", site_col="waterbody_key",
                                               lat_col="latitude"),
                            "by_era_status": {f"{e}|{s}": int(n) for (e, s), n in
                                              wy.groupby(["era", "bloom_status"], observed=True).size().items()},
                            "rows_by_year": {int(k): int(v) for k, v in wy["year"].value_counts().sort_index().items()}},
        "lake_years": {"n_rows": len(ly), "by_era": ly["era"].value_counts().to_dict(),
                       "collapsed_multi_record": int((ly["n_source_records"] > 1).sum()),
                       "n_lakes": int(ly["waterbody_key"].nunique())},
        "nyhabs_reports": ({**C.coverage_stats(reports, date_col="date", site_col="waterbody_key",
                                               lat_col="latitude"),
                            "snapshots": sorted(snaps),
                            "extent_class": reports["extent_class"].value_counts().to_dict(),
                            "status": reports["bloom_status"].value_counts().to_dict()}
                           if not reports.empty else {}),
    }
    raw_files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "download.log")
    sources = ([f"https://{N.DOMAIN}/d/{ds}" for ds in N.DATASETS]
               + [u for u, _ in N.NYHABS_LAYERS.values()] + [N.NYHABS_APP])
    write_manifest(
        "labels_ny_habs", files=raw_files + files, sources=sources,
        description=("NY DEC HAB waterbody-year summaries 2012–2018 (S/C/HT max status) and 2019+ "
                     "(confirmed only), collapsed lake-years, and NYHABS per-report confirmed blooms "
                     "for the season(s) snapshotted. Secondary, effort-driven positive-only labels."),
        license=N.LICENSE, synthetic=False, stats=stats)
    log.info("manifest written: waterbody_years=%d lake_years=%d nyhabs_reports=%d", len(wy), len(ly),
             len(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
