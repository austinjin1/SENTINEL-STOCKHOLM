#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the ECHO SDWA bulk snapshot and build SDWIS drinking-water labels (TapRisk, D4).

Steps (each resume-safe):
  1. download SDWA_latest_downloads.zip (558 MB, quarterly) -> raw('sdwis')
  2. stream-scan SDWA_VIOLATIONS_ENFORCEMENT.csv (4.1 GB uncompressed; never extracted) for the
     nitrate / DBP / SWTR rule-and-contaminant superset -> interim('sdwis')
  3. build processed('labels_sdwis'): systems (contact fields dropped), source facilities,
     areas served, reference codes, violations (one row per PWSID+VIOLATION_ID with enforcement
     summary and label), PWS x quarter label panel for surface-water-sourced CWS
  4. optional Envirofacts cross-check for a few systems; manifest 'labels_sdwis'
  5. delete the zip (sha256 + Last-Modified + ETag kept in the manifest) unless --keep-zip

    python -B scripts/data/download_labels_sdwis.py
    python -B scripts/data/download_labels_sdwis.py --keep-zip --panel-start 2005Q1 --panel-end 2026Q2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.core.registry import file_sha256  # noqa: E402
from sentinel.data.sources import downstream_drinking_water as D  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("labels_sdwis")
ZIP_NAME = "SDWA_latest_downloads.zip"


def dictionary_text() -> str:
    """ECHO SDWA download summary page (data dictionary) as plain text, for provenance."""
    import html
    import re
    t = C.get_text(D.SDWA_DICTIONARY_URL)
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"\s+", " ", t)
    i = t.find("SDWIS Description The Safe")
    return t[i:] if i >= 0 else t


def envirofacts_crosscheck(viol: pd.DataFrame, pwsids: list[str]) -> list[dict]:
    """Compare distinct violation IDs per PWS (scanned rule superset) with Envirofacts SDWIS."""
    out = []
    for p in pwsids:
        try:
            rows = C.get_json(f"{D.ENVIROFACTS_VIOLATION_URL}/pwsid/equals/{p}/json", retries=3) or []
        except C.HTTPFailure as e:  # cross-check only; never fatal
            out.append({"pwsid": p, "error": str(e)[:200]})
            continue
        ef = {str(r.get("violation_id")) for r in rows
              if str(r.get("rule_code")) in D.SCAN_RULE_CODES
              or str(r.get("contaminant_code")) in D.SCAN_CONTAMINANT_CODES}
        mine = viol[viol["pwsid"] == p]
        ours = set(mine["violation_id"].astype(str))
        only = mine[~mine["violation_id"].astype(str).isin(ef)]
        yrs = only["compl_per_begin_date"].dt.year.dropna()
        out.append({"pwsid": p, "envirofacts_rows_all_rules": len(rows), "envirofacts_n": len(ef),
                    "echo_bulk_n": len(ours), "n_common": len(ef & ours), "only_envirofacts": len(ef - ours),
                    "only_echo": len(ours - ef),
                    "only_echo_status": only["violation_status"].value_counts().to_dict(),
                    "only_echo_years": [int(yrs.min()), int(yrs.max())] if len(yrs) else None})
        time.sleep(1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true", help="keep the 558 MB zip after parsing")
    ap.add_argument("--panel-start", default="2005Q1")
    ap.add_argument("--panel-end", default="2026Q2")
    ap.add_argument("--crosscheck", type=int, default=4, help="systems to cross-check (0 = skip)")
    ap.add_argument("--rebuild", action="store_true", help="re-run the scan even if interim exists")
    a = ap.parse_args()

    raw, interim, out = paths.raw("sdwis"), paths.interim("sdwis"), paths.processed("labels_sdwis")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    retrieved = C.retrieval_date()
    zpath = raw / ZIP_NAME
    head_path = raw / "http_head.json"
    if not zpath.exists():
        head = C.http_head(D.SDWA_ZIP_URL)
        C.save_json(dict(head, retrieved=retrieved), head_path)
        log.info("downloading %s (%s bytes, Last-Modified %s)", D.SDWA_ZIP_URL, head["content_length"],
                 head["last_modified"])
        C.download_file(D.SDWA_ZIP_URL, zpath)
    elif not head_path.exists():
        C.save_json(dict(C.http_head(D.SDWA_ZIP_URL), retrieved=retrieved), head_path)
    head = json.loads(head_path.read_text())
    zip_sha = file_sha256(zpath)
    members = D.zip_members(zpath)
    C.save_json({"zip": ZIP_NAME, "sha256": zip_sha, "bytes": zpath.stat().st_size, "members": members},
                raw / "zip_members.json")
    dict_path = raw / "sdwa_download_summary.txt"
    if not dict_path.exists():
        try:
            dict_path.write_text(dictionary_text())
        except C.HTTPFailure as e:
            log.warning("data dictionary page not saved: %s", e)

    scan_path = interim / "violations_enforcement_scan.parquet"
    scan_meta = interim / "violations_enforcement_scan.json"
    if a.rebuild or not scan_path.exists():
        t = time.time()
        scan = D.scan_violations(zpath)
        C.write_parquet(scan, scan_path)
        C.save_json({"n_scanned": scan.attrs["n_scanned"], "n_kept": len(scan), "zip_sha256": zip_sha,
                     "created": retrieved}, scan_meta)
        log.info("scan: %d rows read, %d kept (%.0fs)", scan.attrs["n_scanned"], len(scan), time.time() - t)
    else:
        scan = pd.read_parquet(scan_path)
        log.info("scan reused: %d rows", len(scan))
    meta = json.loads(scan_meta.read_text()) if scan_meta.exists() else {}
    if meta.get("zip_sha256") not in (None, zip_sha):
        raise SystemExit("interim scan was made from a different zip; re-run with --rebuild")
    n_scanned = meta.get("n_scanned")

    ref = D.read_sdwa_csv(zpath, "SDWA_REF_CODE_VALUES.csv")
    C.write_parquet(ref.rename(columns=str.lower), out / "sdwis_ref_codes.parquet")
    snapshot = sorted(set(scan["SUBMISSIONYEARQUARTER"]))
    systems = D.tidy_pws_systems(D.read_sdwa_csv(zpath, "SDWA_PUB_WATER_SYSTEMS.csv"))
    C.write_parquet(systems, out / "sdwis_systems.parquet")
    log.info("systems: %d", len(systems))
    fac = D.tidy_source_facilities(D.read_sdwa_csv(zpath, "SDWA_FACILITIES.csv"))
    C.write_parquet(fac, out / "sdwis_source_facilities.parquet")
    geo = D.tidy_geographic_areas(D.read_sdwa_csv(zpath, "SDWA_GEOGRAPHIC_AREAS.csv"))
    C.write_parquet(geo, out / "sdwis_areas_served.parquet")
    log.info("source facilities: %d; areas served: %d", len(fac), len(geo))

    t = time.time()
    viol = D.attach_system_attributes(D.tidy_violations(scan, ref, snapshot_date=retrieved), systems)
    C.write_parquet(viol, out / "sdwis_violations.parquet")
    log.info("violations: %d (%.0fs)", len(viol), time.time() - t)
    panel = D.pws_quarter_labels(viol, systems, start=a.panel_start, end=a.panel_end, snapshot_date=retrieved)
    C.write_parquet(panel, out / "sdwis_pws_quarter_labels.parquet")
    log.info("PWS-quarter panel: %d rows, %d systems", len(panel), panel["pwsid"].nunique())

    sw_cws = systems[(systems["pws_type_code"] == "CWS") & systems["is_surface_source"]]
    lab = viol[viol["label"] != ""]
    lab_sw = lab[lab["pwsid"].isin(set(sw_cws["pwsid"])) & lab["is_surface_source"]]
    stats = {
        "retrieved": retrieved, "url": D.SDWA_ZIP_URL, "zip_last_modified": head.get("last_modified"),
        "zip_etag": head.get("etag"), "zip_sha256": zip_sha, "zip_bytes": zpath.stat().st_size,
        "zip_kept": bool(a.keep_zip), "submission_year_quarter": snapshot,
        "violations_rows_scanned": n_scanned, "violations_rows_kept_superset": int(len(scan)),
        "scan_rule_codes": sorted(D.SCAN_RULE_CODES), "scan_contaminant_codes": sorted(D.SCAN_CONTAMINANT_CODES),
        "n_systems": int(len(systems)), "n_cws": int((systems["pws_type_code"] == "CWS").sum()),
        "n_cws_active": int(((systems["pws_type_code"] == "CWS") & (systems["pws_activity_code"] == "A")).sum()),
        "n_surface_cws": int(len(sw_cws)),
        "n_surface_cws_active": int((sw_cws["pws_activity_code"] == "A").sum()),
        "surface_cws_by_source": sw_cws["primary_source_code"].value_counts().to_dict(),
        "n_violations": int(len(viol)), "n_violations_health_based": int(viol["is_health_based"].sum()),
        "violations_by_label_all_systems": lab["label"].value_counts().to_dict(),
        "violations_by_label_surface_cws": lab_sw["label"].value_counts().to_dict(),
        "violation_compl_begin_min": str(viol["compl_per_begin_date"].min().date()),
        "violation_compl_begin_max": str(viol["compl_per_begin_date"].max().date()),
        "date_flags": {c: viol[c].value_counts().to_dict() for c in viol.columns if c.endswith("_date_flag")},
        "label_definitions": D.LABEL_DEFINITIONS,
        "panel_start": a.panel_start, "panel_end": a.panel_end, "panel_rows": int(len(panel)),
        "panel_systems": int(panel["pwsid"].nunique()),
        "panel_prevalence": {c: round(float(panel[c].mean()), 5) for c in panel.columns if c.startswith("hb_")},
        "panel_positive_quarters": {c: int(panel[c].sum()) for c in panel.columns if c.startswith("hb_")},
        "n_source_facilities": int(len(fac)), "n_areas_served": int(len(geo)),
    }
    if a.crosscheck:
        ids = (lab_sw.groupby("pwsid").size().sort_values(ascending=False).index[: a.crosscheck // 2].tolist()
               + sw_cws.sample(a.crosscheck - a.crosscheck // 2, random_state=0)["pwsid"].tolist())
        stats["envirofacts_crosscheck"] = envirofacts_crosscheck(viol, [str(i) for i in ids])
        log.info("cross-check: %s", stats["envirofacts_crosscheck"])
    C.save_json(stats, out / "build_stats.json")
    files = sorted(out.glob("*.parquet")) + [out / "build_stats.json", raw / "zip_members.json", head_path]
    if dict_path.exists():
        files.append(dict_path)
    write_manifest("labels_sdwis", files=files,
                   sources=[D.SDWA_ZIP_URL, D.SDWA_DICTIONARY_URL, D.ENVIROFACTS_VIOLATION_URL],
                   description="SDWIS (ECHO SDWA bulk snapshot) drinking-water systems, source facilities, areas "
                               "served, nitrate/DBP/SWTR violations (one row per PWSID+VIOLATION_ID, enforcement "
                               "summary) and a PWS x quarter health-based violation label panel for "
                               "surface-water-sourced community water systems. Contact fields dropped.",
                   license="US Government work (EPA ECHO/SDWIS), public domain", stats=stats)
    if not a.keep_zip:
        zpath.unlink()
        log.info("deleted %s (sha256 %s recorded)", zpath, zip_sha)
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
