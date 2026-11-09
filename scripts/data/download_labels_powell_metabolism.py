#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download and tidy the Powell Center stream-metabolism release (Appling et al. 2018).

USGS data release doi:10.5066/F70864KX (ScienceBase parent 59bff507e4b091459a5e0982).
Only the label-relevant children are downloaded (site data, model configurations, model
diagnostics, daily metabolism estimates); the multi-GB timeseries / model-input / model-
output children are catalogued (sizes recorded in the manifest) but not downloaded.

Raw:       $SENTINEL_DATA/raw/powell_metabolism/        (metadata JSON, small files)
Processed: $SENTINEL_DATA/processed/labels_powell_metabolism/
           daily_metabolism.parquet  sites.parquet  models.parquet
Manifest:  $SENTINEL_DATA/manifests/labels_powell_metabolism.json

    python -B scripts/data/download_labels_powell_metabolism.py
    python -B scripts/data/download_labels_powell_metabolism.py --keep-raw   # keep the 101 MB zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.core.registry import file_sha256  # noqa: E402
from sentinel.data.sources import downstream_metabolism as M  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("powell_metabolism")
NAME = "labels_powell_metabolism"


def fetch_metadata(raw: Path) -> tuple[dict, dict]:
    """Parent + child item JSON (saved raw) → (items by key, summaries by key)."""
    meta_dir = paths.ensure(raw / "metadata")
    items, summ = {}, {}
    parent = M.sciencebase_item(M.POWELL_PARENT)
    C.save_json(parent, meta_dir / "item_parent.json")
    summ["parent"] = M.item_summary(parent)
    children = M.sciencebase_children(M.POWELL_PARENT)
    C.save_json(children, meta_dir / "children_listing.json")
    summ["children_listing"] = [{"id": c.get("id"), "title": c.get("title")} for c in children]
    for key, item_id in M.POWELL_ITEMS.items():
        try:
            item = M.sciencebase_item(item_id)
        except C.HTTPFailure as e:  # e.g. the 2026-03 restructure hid item 2
            item = {"errors": {"message": str(e)}}
        C.save_json(item, meta_dir / f"item_{key}.json")
        items[key] = item
        summ[key] = M.item_summary(item)
        log.info("item %s: %s", key, summ[key].get("title") or summ[key].get("error"))
    return items, summ


def download_needed(items: dict, raw: Path) -> dict[str, dict]:
    got = {}
    for key, names in M.POWELL_NEEDED.items():
        files = {f["name"]: f for f in M.item_files(items[key])}
        for name in names:
            if name not in files:
                raise RuntimeError(f"{name} not listed in ScienceBase item {key}")
            f = files[name]
            dest = raw / name
            if dest.exists() and f["bytes"] and dest.stat().st_size != int(f["bytes"]):
                log.warning("size mismatch for existing %s; re-downloading", name)
                dest.unlink()
            C.download_file(f["url"], dest, timeout=1800)
            if f["bytes"] and dest.stat().st_size != int(f["bytes"]):
                raise RuntimeError(f"{name}: got {dest.stat().st_size} bytes, ScienceBase lists {f['bytes']}")
            got[name] = {"path": dest, "url": f["url"], "bytes": dest.stat().st_size, "item": key}
            log.info("have %s (%d bytes)", name, dest.stat().st_size)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep daily_predictions.zip (101 MB) after parsing")
    a = ap.parse_args()
    raw = paths.raw("powell_metabolism")
    out = paths.processed(NAME)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    retrieved = C.retrieval_date()
    items, summ = fetch_metadata(raw)
    got = download_needed(items, raw)

    # ---- parse
    sites_raw = M.read_tsv(raw / "site_data.tsv")
    _, diag_b = M.zip_member(raw / "diagnostics.zip", ".tsv")
    _, cfg_b = M.zip_member(raw / "config.zip", ".tsv")
    models = M.parse_powell_models(M.read_tsv(diag_b), M.read_tsv(cfg_b))
    daily_name, daily_b = M.zip_member(raw / "daily_predictions.zip", ".tsv")
    daily_raw = M.read_tsv(daily_b)
    del daily_b
    log.info("daily_predictions: %d raw rows, %d columns", len(daily_raw), daily_raw.shape[1])
    daily = M.parse_powell_daily(daily_raw, models)
    del daily_raw
    sites = M.parse_powell_sites(sites_raw)
    agg = daily.groupby("site_id").agg(n_days=("date", "size"), date_min=("date", "min"),
                                       date_max=("date", "max"))
    nmod = models.groupby("site_id").agg(n_models=("resolution_min", "size"),
                                         site_min_confidence=("site_min_confidence", "first"),
                                         site_confidence=("site_confidence", "first"))
    sites = sites.merge(agg, left_on="site_id", right_index=True, how="left")
    sites = sites.merge(nmod, left_on="site_id", right_index=True, how="left")
    sites["n_days"] = sites["n_days"].fillna(0).astype("int64")
    sites["n_models"] = sites["n_models"].fillna(0).astype("int64")
    sites["modeled"] = sites["n_days"] > 0
    missing_sites = sorted(set(daily["site_id"]) - set(sites["site_id"]))
    if missing_sites:
        raise RuntimeError(f"daily sites missing from site_data: {missing_sites[:5]}")
    # Second, independent COMID assignment (Bernhardt et al. 2022 site table, MIT licence).
    bern_zip = C.download_file(M.BERNHARDT_SITE_INFO_URL, raw / "lotic_site_info_full.csv.zip")
    bern = M.read_bernhardt_site_info(M.zip_member(bern_zip, ".csv")[1].decode("utf-8"))
    sites = sites.merge(bern[["site_id", "comid"]].rename(columns={"comid": "comid_bernhardt2022"}),
                        on="site_id", how="left", validate="one_to_one")
    both = sites["comid"].notna() & sites["comid_bernhardt2022"].notna()
    sites["comid_agree"] = pd.Series(pd.NA, index=sites.index, dtype="boolean")
    sites.loc[both, "comid_agree"] = (sites.loc[both, "comid"] == sites.loc[both, "comid_bernhardt2022"])
    got[bern_zip.name] = {"path": bern_zip, "url": M.BERNHARDT_SITE_INFO_URL,
                          "bytes": bern_zip.stat().st_size, "item": "bernhardt2022_figshare"}

    f_daily = C.write_parquet(daily, out / "daily_metabolism.parquet")
    f_sites = C.write_parquet(sites, out / "sites.parquet")
    f_models = C.write_parquet(models, out / "models.parquet")

    # ---- stats
    q = M.powell_quality_counts(daily)
    cov = C.coverage_stats(daily, date_col="date", site_col="site_id")
    stats = {
        "retrieval_date": retrieved,
        "request": {"parent_item": M.POWELL_PARENT, "doi": M.POWELL_DOI, "files": sorted(got)},
        "daily_member": daily_name,
        "n_daily_rows": len(daily), "n_daily_sites": int(daily["site_id"].nunique()),
        "n_site_table_rows": len(sites), "n_sites_modeled": int(sites["modeled"].sum()),
        "n_models": len(models),
        "model_confidence_counts": models["model_confidence"].value_counts(dropna=False).to_dict(),
        "date_min": cov.get("date_min"), "date_max": cov.get("date_max"),
        "rows_by_year": cov.get("rows_by_year"),
        "n_rows_with_qualifier": int((daily["qualifier"] != "").sum()),
        "n_gpp_missing": int(daily["gpp"].isna().sum()), "n_er_missing": int(daily["er"].isna().sum()),
        "n_sites_with_comid": int(sites["comid"].notna().sum()),
        "n_modeled_sites_with_comid": int(sites.loc[sites["modeled"], "comid"].notna().sum()),
        "comid_cross_check_bernhardt2022": {
            "n_sites_in_both": int(both.sum()),
            "n_agree": int(sites["comid_agree"].fillna(False).sum()),
            "n_modeled_in_both": int((both & sites["modeled"]).sum()),
            "n_modeled_agree": int((sites["comid_agree"].fillna(False) & sites["modeled"]).sum())},
        "quality_counts": q,
        "research_check": {"expected_sites": 356, "expected_site_days": 490907,
                           "sites_match": int(daily["site_id"].nunique()) == 356,
                           "site_days_match": len(daily) == 490907},
        "sciencebase_items": summ,
        "skipped_children_bytes": {k: summ[k].get("total_bytes") for k in
                                   ("3_timeseries", "4_model_inputs", "6_model_outputs",
                                    "2a_site_coordinates", "2b_site_catchments")},
    }
    log.info("daily rows %d, sites %d, %s..%s", stats["n_daily_rows"], stats["n_daily_sites"],
             stats["date_min"], stats["date_max"])
    log.info("quality: %s", q)

    zip_path = raw / "daily_predictions.zip"
    deleted = []
    if not a.keep_raw and zip_path.exists():
        deleted.append({"name": zip_path.name, "bytes": zip_path.stat().st_size,
                        "sha256": file_sha256(zip_path), "url": got[zip_path.name]["url"]})
        zip_path.unlink()
        log.info("deleted %s after parsing (sha256 recorded in manifest)", zip_path.name)
    stats["deleted_raw_files"] = deleted

    meta_files = sorted((raw / "metadata").glob("*.json"))
    raw_files = [p for p in (raw / n for n in got) if p.exists()]
    sources = [M.SB_ITEM_URL.format(id=M.POWELL_PARENT) + "?format=json"] + [g["url"] for g in got.values()]
    write_manifest(
        NAME, files=[f_daily, f_sites, f_models, *raw_files, *meta_files], sources=sources,
        description=("Powell Center daily stream metabolism (GPP, ER in g O2 m-2 d-1; K600 in d-1; "
                     "95% credible intervals, Rhat, n_eff; daily predictors) for 356 US rivers, "
                     "Appling et al. 2018 (doi:10.1038/sdata.2018.292), USGS data release "
                     "doi:10.5066/F70864KX. Model-derived labels (streamMetabolizer, from diel DO)."),
        license=("US Government public domain (USGS data release; FGDC use constraints: none); "
                 "comid_bernhardt2022 column from figshare 19074299 (MIT)"),
        synthetic=False, stats=stats)
    log.info("manifest written: %s", NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
