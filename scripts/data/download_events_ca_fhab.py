#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the California FHAB portal (data.ca.gov CKAN) and build tidy secondary HAB labels.

Raw (one folder per retrieval date = label vintage): $SENTINEL_DATA/raw/ca_fhab/<YYYY-MM-DD>/
    package_show.json, bloom_reports.csv, cases.csv, responses.csv, results.csv, docs/*.pdf
Processed: $SENTINEL_DATA/processed/labels_ca_fhab/{bloom_reports,advisories,cases,results}.parquet
Manifest:  $SENTINEL_DATA/manifests/labels_ca_fhab.json

    python -B scripts/data/download_events_ca_fhab.py                  # download today's vintage + build
    python -B scripts/data/download_events_ca_fhab.py --vintage 2026-10-22 --skip-download   # rebuild

The published CSV is the primary artefact; if its download fails the CKAN datastore API is paged
instead (``--use-datastore`` forces that). Row counts are cross-checked against the datastore total.
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
from sentinel.data.sources import events_ca_fhab as F
from sentinel.data.sources import events_common as C

log = logging.getLogger("ca_fhab")


def datastore_total(resource_id: str) -> int | None:
    try:
        d = C.get_json(F.DATASTORE_URL, {"resource_id": resource_id, "limit": 0})
        return int(d["result"]["total"]) if d.get("success") else None
    except Exception as e:  # noqa: BLE001 — informational only
        log.warning("datastore total failed for %s: %s", resource_id, e)
        return None


def fetch(vdir: Path, use_datastore: bool) -> dict:
    pkg = C.ckan_package(F.CKAN_BASE, F.PACKAGE_ID)
    C.save_json(pkg, vdir / "package_show.json")
    res = F.resolve_resources(pkg)
    info: dict = {"package_metadata_modified": pkg.get("metadata_modified"),
                  "package_license": f"{pkg.get('license_id')} / {pkg.get('license_title')}",
                  "resources": {}}
    for short in F.DATA_RESOURCES:
        r = res[short]
        dest = vdir / f"{short}.csv"
        rec = {"resource_id": r["id"], "name": r.get("name"), "url": r.get("url"),
               "last_modified": r.get("last_modified") or r.get("metadata_modified"),
               "datastore_total": datastore_total(r["id"]) if r.get("datastore_active") else None}
        method = None
        if not use_datastore:
            try:
                C.download_file(r["url"], dest, overwrite=True)
                method = "csv_download"
            except C.HTTPFailure as e:
                log.warning("CSV download failed for %s (%s); falling back to datastore", short, e)
        if method is None:
            records, fields = C.ckan_datastore_rows(F.CKAN_BASE, r["id"], page_size=5000)
            F.datastore_records_to_frame(records, fields).to_csv(dest, index=False)
            method = "datastore_search"
        n = len(F.read_raw_csv(dest))
        rec.update(method=method, rows=n, bytes=dest.stat().st_size)
        if rec["datastore_total"] is not None and rec["datastore_total"] != n:
            log.warning("%s: CSV has %d rows but datastore total is %d", short, n, rec["datastore_total"])
        log.info("%s: %d rows via %s", short, n, method)
        info["resources"][short] = rec
    for short in F.DOC_RESOURCES:
        r = res.get(short)
        if r is None:
            log.warning("doc resource %s not found", short)
            continue
        name = Path(r["url"].split("?")[0]).name or f"{short}.pdf"
        dest = vdir / "docs" / name
        C.download_file(r["url"], dest, overwrite=True)
        info["resources"][short] = {"resource_id": r["id"], "url": r["url"], "bytes": dest.stat().st_size}
    C.save_json(info, vdir / "fetch_info.json")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vintage", default=None, help="raw folder date (default: today, UTC)")
    ap.add_argument("--skip-download", action="store_true", help="rebuild from an existing raw vintage")
    ap.add_argument("--use-datastore", action="store_true", help="page the CKAN datastore instead of CSVs")
    a = ap.parse_args()
    vintage = a.vintage or C.retrieval_date()
    root = paths.raw("ca_fhab")
    vdir = paths.ensure(root / vintage)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(root / "download.log"), logging.StreamHandler()])
    log.info("vintage %s (skip_download=%s)", vintage, a.skip_download)
    if not a.skip_download:
        fetch(vdir, a.use_datastore)
    info = json.loads((vdir / "fetch_info.json").read_text())

    raw = {k: F.read_raw_csv(vdir / f"{k}.csv") for k in F.DATA_RESOURCES}
    max_date = (dt.date.fromisoformat(vintage) + dt.timedelta(days=1)).isoformat()
    tables = F.build_tables(raw, max_date=max_date)
    out = paths.processed("labels_ca_fhab")
    files = []
    for name, df in tables.items():
        files.append(C.write_parquet(df, out / f"{name}.parquet"))
        log.info("processed %s: %d rows", name, len(df))

    br, adv = tables["bloom_reports"], tables["advisories"]
    stats = {
        "vintage": vintage, "retrieval_date": vintage, "max_valid_date": max_date,
        "request": {"package_show": f"{F.CKAN_BASE}/api/3/action/package_show?id={F.PACKAGE_ID}",
                    "csv_or_datastore": {k: v.get("method") for k, v in info["resources"].items()
                                         if k in F.DATA_RESOURCES}},
        "package_metadata_modified": info.get("package_metadata_modified"),
        "package_license": info.get("package_license"),
        "raw_rows": {k: len(v) for k, v in raw.items()},
        "raw_sha1": {k: F.frame_sha1(v) for k, v in raw.items()},
        "datastore_totals": {k: v.get("datastore_total") for k, v in info["resources"].items()
                             if k in F.DATA_RESOURCES},
        "tables": {
            "bloom_reports": C.coverage_stats(br, date_col="date", site_col="waterbody_name", lat_col="latitude"),
            "advisories": C.coverage_stats(adv, date_col="date", site_col="advisory_id", lat_col="latitude"),
            "cases": C.coverage_stats(tables["cases"], date_col="case_start_date", site_col="case_id"),
            "results": C.coverage_stats(tables["results"], date_col="date", lat_col="latitude"),
        },
        "raw_bloom_report_rows_by_observation_year": {
            int(k): int(v) for k, v in F.parse_date(raw["bloom_reports"]["Observation_Date"], max_date=max_date)[0]
            .dt.year.value_counts().sort_index().items()},
        "end_dates": F.end_date_stats(raw["bloom_reports"], raw["responses"]),
        "bloom_report_flags": {"date_flag": br["date_flag"].value_counts().to_dict(),
                               "coord_flag": br["coord_flag"].value_counts().to_dict()},
        "bloom_report_advisory_class": br["advisory_class"].value_counts().to_dict(),
        "advisory_class": adv["advisory_class"].value_counts().to_dict(),
        "advisory_tier_ge_caution": int(adv["label_tier_ge_caution"].sum()),
        "advisory_observation_channel": adv["observation_channel"].value_counts().to_dict(),
        "results_with_value": int(tables["results"]["value"].notna().sum()),
        "results_placeholders": int(tables["results"]["is_placeholder"].sum()),
    }
    raw_files = sorted(p for p in vdir.rglob("*") if p.is_file())
    sources = [F.PACKAGE_URL, F.DATASTORE_URL] + [v["url"] for v in info["resources"].values() if v.get("url")]
    write_manifest(
        "labels_ca_fhab", files=raw_files + files, sources=sources,
        description=("California Water Boards FHAB portal (data.ca.gov): voluntary bloom reports, cases, "
                     "advisory responses (CCHAB Caution/Warning/Danger tiers) and sparse results; tidy "
                     "secondary HAB labels. Reports are 'suspected' until staff recommend an advisory."),
        license=F.LICENSE, synthetic=False, stats=stats)
    log.info("manifest written; bloom_reports=%d advisories=%d cases=%d results=%d",
             len(br), len(adv), len(tables["cases"]), len(tables["results"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
