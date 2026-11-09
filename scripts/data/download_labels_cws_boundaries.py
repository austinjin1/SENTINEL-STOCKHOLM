#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download EPA public-water-system service-area boundaries (SAB v3) and build CWS polygons (TapRisk, D4).

Official bulk file ("Download Map Data" on the EPA page): GitHub USEPA/ORD_SAB_Model
``Version_History/PWS_Boundaries_Latest.zip`` (570 MB, Git LFS; its ETag is the sha256 of the file).
Outputs in processed('labels_cws_boundaries'):
  cws_boundaries.parquet             GeoParquet, EPSG:4326, one row per CWS polygon (layer 'CWS')
  cws_unmapped.parquet               CWS that SAB v3 could not map (Missing_CWS_V_3_0.csv)
  cws_tract_crosswalk.parquet        PWS <-> 2020 census tract weights (area + building based)
  cws_block_group_crosswalk.parquet  PWS <-> 2020 block group weights
The zip and the extracted 686 MB GeoPackage are deleted afterwards (sha256 kept) unless --keep-raw.
``--check-service`` compares PWSIDs with the hosted ArcGIS FeatureServer (attributes only).

    python -B scripts/data/download_labels_cws_boundaries.py --check-service
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.core.registry import file_sha256  # noqa: E402
from sentinel.data.sources import downstream_drinking_water as D  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("labels_cws_boundaries")
ZIP_NAME = "PWS_Boundaries_Latest.zip"
MEMBERS = {"3_0/Service_Areas_V_3_0.gpkg": "Service_Areas_V_3_0.gpkg",
           "3_0/Missing_CWS_V_3_0.csv": "Missing_CWS_V_3_0.csv",
           "3_0/Census_Tables/Tracts_V_3_0.csv": "Tracts_V_3_0.csv",
           "3_0/Census_Tables/Block_Groups_V_3_0.csv": "Block_Groups_V_3_0.csv"}
DOC_MEMBERS = {"3_0/README.md": "SAB_v3_README.md", "3_0/Census_Tables/README.md": "SAB_v3_Census_Tables_README.md",
               "3_0/Documentation.pdf": "SAB_Documentation.pdf"}


def service_pwsids() -> tuple[int, set[str], dict]:
    info = C.arcgis_layer_info(D.SAB_FEATURESERVER_URL)
    feats = C.arcgis_query(D.SAB_FEATURESERVER_URL, out_fields="PWSID", return_geometry=False)
    ids = {str(f["attributes"]["PWSID"]).strip() for f in feats}
    edit = info.get("editingInfo") or {}
    return len(feats), ids, {k: pd.to_datetime(v, unit="ms", utc=True).isoformat() for k, v in edit.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-raw", action="store_true", help="keep the zip and extracted GeoPackage")
    ap.add_argument("--check-service", action="store_true", help="compare with the hosted FeatureServer")
    a = ap.parse_args()
    raw, tmp, out = paths.raw("cws_boundaries"), paths.interim("cws_boundaries"), paths.processed(
        "labels_cws_boundaries")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    retrieved = C.retrieval_date()
    head = C.http_head(D.SAB_ZIP_URL)
    zpath = C.download_file(D.SAB_ZIP_URL, raw / ZIP_NAME)
    sha = file_sha256(zpath)
    etag = (head.get("etag") or "").strip('"')
    log.info("zip %s sha256 %s (ETag %s, match=%s)", zpath, sha, etag, sha == etag)
    members = D.zip_members(zpath)
    C.save_json(dict(head, retrieved=retrieved, sha256=sha, sha256_matches_etag=sha == etag, members=members),
                raw / "zip_info.json")
    with zipfile.ZipFile(zpath) as z:
        for m, dest in {**MEMBERS, **DOC_MEMBERS}.items():
            target = (raw if m in DOC_MEMBERS else tmp) / dest
            if not target.exists():
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)

    import pyogrio

    gpkg = tmp / "Service_Areas_V_3_0.gpkg"
    layers = {str(n): int(pyogrio.read_info(gpkg, layer=str(n))["features"]) for n, _ in pyogrio.list_layers(gpkg)}
    src_crs = pyogrio.read_info(gpkg, layer=D.CWS_LAYER)["crs"]
    cws = D.read_cws_boundaries(gpkg)
    C.write_parquet(cws, out / "cws_boundaries.parquet")
    log.info("CWS polygons: %d (%d unique PWSID)", len(cws), cws["pwsid"].nunique())
    miss = D.tidy_missing_cws(pd.read_csv(tmp / "Missing_CWS_V_3_0.csv", dtype=str, keep_default_na=False,
                                          encoding="cp1252"))
    C.write_parquet(miss, out / "cws_unmapped.parquet")
    tr = D.tidy_census_crosswalk(pd.read_csv(tmp / "Tracts_V_3_0.csv", dtype=str, keep_default_na=False), "tract")
    C.write_parquet(tr, out / "cws_tract_crosswalk.parquet")
    bg = D.tidy_census_crosswalk(pd.read_csv(tmp / "Block_Groups_V_3_0.csv", dtype=str, keep_default_na=False),
                                 "block_group")
    C.write_parquet(bg, out / "cws_block_group_crosswalk.parquet")

    stats = {"retrieved": retrieved, "url": D.SAB_ZIP_URL, "zip_last_modified": head.get("last_modified"),
             "zip_etag_lfs_oid": etag, "zip_sha256": sha, "zip_bytes": zpath.stat().st_size,
             "gpkg_layers": layers, "source_crs": src_crs, "stored_crs": "EPSG:4326",
             "n_cws_polygons": int(len(cws)), "n_unique_pwsid": int(cws["pwsid"].nunique()),
             "n_duplicated_pwsid_rows": int(cws["pwsid_duplicated"].sum()),
             "boundary_source": cws["boundary_source"].value_counts(dropna=False).rename(index=str).to_dict(),
             "share_modeled": round(float(cws["is_modeled"].mean()), 4),
             "model_method": cws["model_method"].value_counts(dropna=False).rename(index=str).to_dict(),
             "n_invalid_geometry": int((~cws["geometry_valid"]).sum()),
             "n_states": int(cws["state"].nunique()),
             "population_served_total": int(cws["population_served"].sum()),
             "n_unmapped_cws": int(len(miss)), "n_tract_links": int(len(tr)), "n_block_group_links": int(len(bg)),
             "not_kept": "T_NTNC layer (non-community systems) and the census-block crosswalk (403 MB CSV)"}
    sys_path = paths.PROCESSED / "labels_sdwis" / "sdwis_systems.parquet"
    if sys_path.exists():  # coverage of SDWIS community water systems (cross-source check)
        s = pd.read_parquet(sys_path, columns=["pwsid", "pws_type_code", "pws_activity_code", "is_surface_source"])
        act = s[(s["pws_type_code"] == "CWS") & (s["pws_activity_code"] == "A")]
        ids = set(cws["pwsid"].astype(str))
        stats["sdwis_active_cws_with_boundary"] = round(float(act["pwsid"].isin(ids).mean()), 4)
        sw = act[act["is_surface_source"]]
        stats["sdwis_active_surface_cws_with_boundary"] = round(float(sw["pwsid"].isin(ids).mean()), 4)
        stats["sdwis_active_surface_cws"] = int(len(sw))
    if a.check_service:
        n, sids, edit = service_pwsids()
        gids = set(cws["pwsid"].astype(str))
        stats["featureserver"] = {"url": D.SAB_FEATURESERVER_URL, "n_features": n, "n_unique_pwsid": len(sids),
                                  "n_common_pwsid": len(sids & gids), "only_featureserver": len(sids - gids),
                                  "only_bulk_gpkg": len(gids - sids), "editing_info": edit}
        log.info("FeatureServer: %s", stats["featureserver"])
    C.save_json(stats, out / "build_stats.json")
    docs = [raw / d for d in DOC_MEMBERS.values() if (raw / d).exists()]
    write_manifest("labels_cws_boundaries",
                   files=sorted(out.glob("*.parquet")) + [out / "build_stats.json", raw / "zip_info.json"] + docs,
                   sources=[D.SAB_ZIP_URL, D.SAB_PAGE_URL, D.SAB_REPO_URL]
                   + ([D.SAB_FEATURESERVER_URL] if a.check_service else []),
                   description="EPA public water system service-area boundaries v3 (ORD SAB model), community "
                               "water systems layer, as GeoParquet (EPSG:4326) with boundary provenance (system "
                               "sourced vs modeled), plus unmapped-CWS list and census tract/block-group "
                               "crosswalks. Service areas are where water is delivered, NOT source-water intakes.",
                   license="US Government work (EPA); repository under MIT license", stats=stats)
    if not a.keep_raw:
        zpath.unlink()
        for dest in MEMBERS.values():
            (tmp / dest).unlink(missing_ok=True)
        log.info("deleted zip and extracted files (sha256 %s recorded)", sha)
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
