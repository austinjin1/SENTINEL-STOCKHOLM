#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download and tidy StreamPULSE public metabolism estimates (bulk downloads, no login).

The StreamPULSE bulk page (https://data.streampulse.org/download_bulk) serves plain POST
forms; no account or token is needed for public data (embargoed sites are excluded by the
portal). Data policy: Open Data Commons Attribution License v1.0. The Bernhardt et al. 2022
figshare site table (MIT) is added for NHDPlusV2 COMIDs of StreamPULSE sites.

Raw:       $SENTINEL_DATA/raw/streampulse/        (3 zips, figshare site-info zip, policy snapshot)
Processed: $SENTINEL_DATA/processed/labels_streampulse/
           daily_metabolism.parquet  models.parquet  sites.parquet  bernhardt2022_site_info.parquet
Manifest:  $SENTINEL_DATA/manifests/labels_streampulse.json

    python -B scripts/data/download_labels_streampulse.py            # download (if absent) + parse
    python -B scripts/data/download_labels_streampulse.py --refresh  # re-download (portal updates daily)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import downstream_metabolism as M  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("streampulse")
NAME = "labels_streampulse"
ZIP_NAMES = {"daily_results": "all_daily_model_results.csv.zip",
             "model_summary": "all_model_summary_data.csv.zip",
             "basic_site": "all_basic_site_data.csv.zip"}
FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/19074299"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download the StreamPULSE zips")
    a = ap.parse_args()
    raw = paths.raw("streampulse")
    out = paths.processed(NAME)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    retrieved = C.retrieval_date()

    headers = {}
    for key, fname in ZIP_NAMES.items():
        url = M.SP_BASE + M.SP_ENDPOINTS[key]
        side = raw / (fname + ".headers.json")
        p, hdr = M.post_download(url, raw / fname, overwrite=a.refresh)
        if hdr.get("skipped_existing") and side.exists():   # headers recorded at download time
            hdr = {**json.loads(side.read_text()), "skipped_existing": True}
        elif not hdr.get("skipped_existing"):
            hdr["retrieved"] = retrieved
            C.save_json(hdr, side)
        headers[fname] = hdr
        log.info("%s -> %s (%d bytes) %s", url, p.name, p.stat().st_size, hdr)
    # Terms snapshot and the figshare record (licence provenance).
    policy = C.get_text(M.SP_POLICY_URL)
    (raw / "streampulse_policy.html").write_text(policy)
    fs_meta = C.get_json(FIGSHARE_ARTICLE)
    C.save_json(fs_meta, raw / "figshare_19074299.json")
    bern_zip = C.download_file(M.BERNHARDT_SITE_INFO_URL, raw / "lotic_site_info_full.csv.zip",
                               overwrite=a.refresh)

    # ---- parse
    def member(fname: str) -> bytes:
        return M.zip_member(raw / fname, ".csv")[1]

    sites = M.parse_sp_sites(M.read_csv(member(ZIP_NAMES["basic_site"])))
    bern = M.read_bernhardt_site_info(M.zip_member(bern_zip, ".csv")[1].decode("utf-8"))
    if bern["latitude"].isna().any() or bern["site_id"].duplicated().any():
        raise RuntimeError("Bernhardt 2022 site table repair failed (NaN lat or duplicate ids)")
    sites = sites.merge(bern[["site_id", "comid", "vpu", "stream_order", "ws_area_km2"]],
                        on="site_id", how="left", validate="one_to_one")
    sites["comid_source"] = sites["comid"].notna().map(
        {True: "Bernhardt et al. 2022 lotic_site_info_full (doi:10.6084/m9.figshare.19074299)", False: ""})
    models = M.parse_sp_models(M.read_csv(member(ZIP_NAMES["model_summary"])))
    daily = M.parse_sp_daily(M.read_csv(member(ZIP_NAMES["daily_results"])), sites)

    f_daily = C.write_parquet(daily, out / "daily_metabolism.parquet")
    f_models = C.write_parquet(models, out / "models.parquet")
    f_sites = C.write_parquet(sites, out / "sites.parquet")
    f_bern = C.write_parquet(bern, out / "bernhardt2022_site_info.parquet")

    cov = C.coverage_stats(daily, date_col="date", site_col="site_id")
    dsites = daily.drop_duplicates("site_id")
    stats = {
        "retrieval_date": retrieved,
        "request": {"endpoints": {k: M.SP_BASE + v for k, v in M.SP_ENDPOINTS.items()}, "method": "POST",
                    "form_data": {}, "figshare_file": M.BERNHARDT_SITE_INFO_URL},
        "response_headers": headers,
        "figshare_record": {"title": fs_meta.get("title"), "doi": fs_meta.get("doi"),
                            "license": (fs_meta.get("license") or {}).get("name"),
                            "modified": fs_meta.get("modified_date")},
        "n_daily_rows": len(daily), "n_daily_sites": int(daily["site_id"].nunique()),
        "date_min": cov.get("date_min"), "date_max": cov.get("date_max"),
        "rows_by_year": cov.get("rows_by_year"),
        "quality_counts": M.sp_quality_counts(daily),
        "daily_sites_by_data_source": dsites["data_source"].astype("string").fillna("<not in site table>")
        .value_counts().to_dict(),
        "daily_sites_not_in_site_table": sorted(dsites.loc[~dsites["in_site_table"], "site_id"].tolist()),
        "daily_sites_matched_case_insensitive": sorted(
            dsites.loc[dsites["site_match"].astype(str) == "case_insensitive", "site_id"].tolist()),
        "n_rows_dup_site_date_case_insensitive": int(daily["dup_site_date_ci"].sum()),
        "daily_sites_with_usgs_site_no": int(dsites["usgs_site_no"].notna().sum()),
        "daily_sites_with_comid": int(dsites["comid"].notna().sum()),
        "daily_sites_in_us": int(dsites["in_us"].fillna(False).astype(bool).sum()),
        "daily_sites_in_us_unknown": int(dsites["in_us"].isna().sum()),
        "n_site_table_rows": len(sites),
        "site_ids_with_duplicate_registrations": sorted(sites.loc[sites["n_registrations"] > 1, "site_id"]),
        "site_table_by_data_source": sites["data_source"].value_counts().to_dict(),
        "n_models": len(models), "n_models_current_best": int(models["current_best"].fillna(False).sum()),
        "dropped_personal_columns": list(M.SP_DROP_PERSONAL),
        "n_bernhardt_site_rows": len(bern),
    }
    log.info("daily rows %d, sites %d, %s..%s; %s", stats["n_daily_rows"], stats["n_daily_sites"],
             stats["date_min"], stats["date_max"], stats["quality_counts"])
    raw_files = [raw / f for f in ZIP_NAMES.values()] + [bern_zip, raw / "streampulse_policy.html",
                                                        raw / "figshare_19074299.json"]
    raw_files += [p for p in (raw / (f + ".headers.json") for f in ZIP_NAMES.values()) if p.exists()]
    write_manifest(
        NAME, files=[f_daily, f_models, f_sites, f_bern, *raw_files],
        sources=[M.SP_BASE + v for v in M.SP_ENDPOINTS.values()]
        + [M.SP_POLICY_URL, M.BERNHARDT_SITE_INFO_URL, FIGSHARE_ARTICLE],
        description=("StreamPULSE public daily metabolism model results (GPP, ER in g O2 m-2 d-1; K600 "
                     "in d-1; 95% CIs; valid_day flag), model summaries and site table (contact "
                     "columns dropped), plus Bernhardt et al. 2022 site table for COMIDs. "
                     "Model-derived labels (streamMetabolizer, from diel DO)."),
        license=f"{M.SP_LICENSE}; Bernhardt 2022 figshare site table: MIT",
        synthetic=False, stats=stats)
    log.info("manifest written: %s", NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
