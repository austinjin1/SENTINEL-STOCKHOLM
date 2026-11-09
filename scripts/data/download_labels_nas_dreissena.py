#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download USGS NAS dreissenid occurrence records and build MusselGuard label tables.

Species: 5 zebra mussel (Dreissena polymorpha), 95 quagga mussel (Dreissena bugensis),
3118 Dreissena sp. (unidentified dreissenid). Steps (resume-safe; ``--refresh`` re-downloads):

1. provenance snapshots (species records, API documentation, disclaimer, DB characterisation);
2. all occurrence pages per species (``limit``/``offset``), stored as gzipped JSON envelopes;
3. paging check: the same query with a second page size must return the identical key set;
4. temporal-accuracy tags: one query per ``yrAcc`` class (Actual / Estimated / Publication year);
5. tidy tables → processed('labels_nas_dreissena'); manifest 'labels_nas_dreissena'.

    python -B scripts/data/download_labels_nas_dreissena.py            # ~25 requests, ~2–5 min
    python -B scripts/data/download_labels_nas_dreissena.py --refresh  # new vintage
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import downstream_dreissena as D  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("nas_dreissena")

RESEARCH_COUNTS = {"5": 11034, "95": 1523}  # docs/research/07 §a.2 E7, measured 2026-10-22
COLLECTION_INFO_URL = "https://nas.er.usgs.gov/queries/CollectionInfo.aspx?SpeciesID={sid}"


def _dump_gz(obj, path: Path) -> Path:
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)
    tmp.replace(path)
    return path


def _load_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_species(sid: int, raw: Path, page_size: int, sleep: float, refresh: bool) -> tuple[list, dict]:
    dest = raw / f"occurrence_species_{sid}.json.gz"
    if dest.exists() and not refresh:
        blob = _load_gz(dest)
        log.info("species %d: reusing %s (retrieved %s)", sid, dest.name, blob["retrieved_utc"])
    else:
        t0 = time.time()
        pages = D.nas_fetch_pages(sid, page_size=page_size, sleep=sleep)
        blob = {"retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "url": D.NAS_SEARCH_URL, "params": {"species_ID": sid, "limit": page_size},
                "pages": pages}
        _dump_gz(blob, dest)
        log.info("species %d: %d pages in %.0f s → %s", sid, len(pages), time.time() - t0, dest.name)
    recs, report = D.records_from_pages(blob["pages"])
    report["retrieved_utc"] = blob["retrieved_utc"]
    return recs, report


def verify_paging(sid: int, keys: set[int], page_size: int, sleep: float) -> dict:
    recs, rep = D.records_from_pages(D.nas_fetch_pages(sid, page_size=page_size, sleep=sleep))
    other = {int(r["key"]) for r in recs}
    return {"page_size": page_size, "n_pages": rep["n_pages"], "n_keys": len(other),
            "n_duplicate_keys": rep["n_duplicate_keys"], "missing_vs_primary": len(keys - other),
            "extra_vs_primary": len(other - keys), "identical": other == keys}


def fetch_yracc(sid: int, raw: Path, page_size: int, sleep: float, refresh: bool) -> dict[str, list[int]]:
    dest = raw / f"yracc_species_{sid}.json"
    if dest.exists() and not refresh:
        return json.loads(dest.read_text())["classes"]
    classes = D.nas_year_accuracy_keys(sid, page_size=page_size, sleep=sleep)
    C.save_json({"retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "url": D.NAS_SEARCH_URL, "filter": "yrAcc", "classes": classes}, dest)
    return classes


def collection_info_count(sid: int) -> int | None:
    """'N results for' on the NAS website's collection-info page (independent cross-check)."""
    try:
        text = C.get_text(COLLECTION_INFO_URL.format(sid=sid), timeout=120, retries=3)
    except C.HTTPFailure as e:
        log.warning("collection info %d failed: %s", sid, e)
        return None
    m = re.search(r"([\d,]+)\s+results for", re.sub(r"<[^>]+>", " ", text))
    return int(m[1].replace(",", "")) if m else None


def snapshot_provenance(raw: Path, refresh: bool) -> list[Path]:
    files = []
    for sid in D.NAS_SPECIES:
        p = raw / f"species_{sid}.json"
        if refresh or not p.exists():
            C.save_json(C.get_json(f"{D.NAS_API}/species/{sid}"), p)
        files.append(p)
    for name, url in (("api_documentation.html", D.NAS_DOC_URL), ("disclaimer.html", D.NAS_DISCLAIMER_URL),
                      ("db_characterization.html", D.NAS_DBCHAR_URL)):
        p = raw / name
        if refresh or not p.exists():
            p.write_text(C.get_text(url, timeout=120))
        files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", type=int, nargs="+", default=list(D.NAS_SPECIES))
    ap.add_argument("--page-size", type=int, default=5000)
    ap.add_argument("--verify-page-size", type=int, default=2000,
                    help="second page size for the paging check (0 = skip)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--refresh", action="store_true", help="re-download everything (new vintage)")
    a = ap.parse_args()

    raw = paths.raw("nas_dreissena")
    out = paths.processed("labels_nas_dreissena")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    retrieved = C.retrieval_date()
    prov_files = snapshot_provenance(raw, a.refresh)

    all_recs, ya_map, stats = [], {}, {"retrieval_date": retrieved, "species": {}}
    for sid in a.species:
        recs, report = fetch_species(sid, raw, a.page_size, a.sleep, a.refresh)
        keys = {int(r["key"]) for r in recs}
        if a.verify_page_size:
            report["paging_check"] = verify_paging(sid, keys, a.verify_page_size, a.sleep)
            log.info("species %d paging check: %s", sid, report["paging_check"])
        classes = fetch_yracc(sid, raw, a.page_size, a.sleep, a.refresh)
        m, conflicts = D.year_accuracy_map(classes)
        ya_map.update(m)
        tagged = set(m)
        report["year_accuracy"] = {k: len(v) for k, v in classes.items()}
        report["year_accuracy_untagged"] = len(keys - tagged)
        report["year_accuracy_extra_keys"] = len(tagged - keys)
        report["year_accuracy_conflicts"] = len(conflicts)
        report["website_collection_info_count"] = collection_info_count(sid)
        report["research_count_2026_10_22"] = RESEARCH_COUNTS.get(str(sid))
        stats["species"][str(sid)] = report
        log.info("species %d: %d unique records; yrAcc %s; untagged %d", sid, len(recs),
                 report["year_accuracy"], report["year_accuracy_untagged"])
        all_recs.extend(recs)

    occ = D.parse_nas_records(all_recs, year_accuracy=ya_map)
    refs = D.nas_references(all_recs)
    first = D.first_detection_by_huc12(occ)
    f_occ = C.write_parquet(occ, out / "nas_dreissena_occurrences.parquet")
    f_ref = C.write_parquet(refs, out / "nas_dreissena_references.parquet")
    f_first = C.write_parquet(first, out / "nas_dreissena_first_detection_huc12.parquet")

    for sid, g in occ.groupby("species_id"):
        s = stats["species"][sid]
        us = g[g["country"] == "US"]
        s.update({
            "n_records": len(g), "year_min": int(g["year"].min()), "year_max": int(g["year"].max()),
            "status": {str(k): int(v) for k, v in g["status"].fillna("(blank)").value_counts().items()},
            "record_type": {str(k): int(v) for k, v in
                            g["record_type"].fillna("(blank)").value_counts().items()},
            "n_established": int(g["is_established"].sum()),
            "n_huc12": int(g["huc12"].nunique()), "n_us_states": int(us["state"].nunique()),
            "n_us_records": len(us), "country": {str(k): int(v) for k, v in
                                                     g["country"].fillna("(unmapped)").value_counts().items()},
            "unmapped_state_values": sorted(
                g.loc[g["country"].isna(), "state_raw"].dropna().unique().tolist()),
            "n_with_coords": int(g["latitude"].notna().sum()),
            "coord_flag": {str(k or "ok"): int(v) for k, v in g["coord_flag"].value_counts().items()},
            "date_precision": {str(k): int(v) for k, v in g["date_precision"].value_counts().items()},
        })
    stats["first_detection_rows"] = {str(k): int(v) for k, v in first["species_group"].value_counts().items()}
    stats["n_reference_links"] = len(refs)
    stats["citation"] = D.NAS_CITATION.format(year=retrieved[:4], accessed=_dt.date.fromisoformat(retrieved)
                                              .strftime("%m/%d/%Y"))
    stats["request"] = {"endpoint": D.NAS_SEARCH_URL, "species_ID": a.species, "limit": a.page_size,
                        "yrAcc": list(D.YR_ACC_CLASSES), "sleep_s": a.sleep}
    C.save_json(stats, out / "nas_dreissena_stats.json")

    raw_files = sorted(raw.glob("occurrence_species_*.json.gz")) + sorted(raw.glob("yracc_species_*.json"))
    write_manifest(
        "labels_nas_dreissena",
        files=[f_occ, f_ref, f_first, out / "nas_dreissena_stats.json", *raw_files, *prov_files],
        sources=[D.NAS_SEARCH_URL + f"?species_ID={sid}" for sid in a.species]
        + [f"{D.NAS_API}/species/{sid}" for sid in a.species] + [D.NAS_DOC_URL, D.NAS_DISCLAIMER_URL],
        description="USGS Nonindigenous Aquatic Species (NAS) occurrence records for zebra mussel (5), "
                    "quagga mussel (95) and Dreissena sp. (3118): presence-only records with current "
                    "population status, record type, coordinates + accuracy, WBD HUC8/10/12, temporal-"
                    "accuracy class (yrAcc); plus a per-HUC12 first-detection table for rolling-origin "
                    "tests.",
        license="US Government work (public domain); NAS data disclaimer applies (provisional, "
                "'as-is'); cite: " + stats["citation"],
        synthetic=False, stats=stats)
    log.info("done: %d occurrences, %d first-detection rows", len(occ), len(first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
