#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download and tidy the Iowa DNR Fish Kill Database (D3 KillAlert labels).

Steps (resume-safe; each step caches its raw responses under $SENTINEL_DATA/raw/iowa_fishkill/):
  1. API snapshot (all "Final" events) + CSV export + API spec + Codex/doc pages;
  2. HTML detail page per event (kill length, valuation, comments, species tables), 1 req/s;
  3. ADBNet 305(b) assessment-segment metadata for every impacted segment;
  4. optional NLDI point-in-catchment COMID per event point (--nldi);
  5. tidy Parquet tables in $SENTINEL_DATA/processed/labels_iowa_fishkill/ + manifest.

    python -B scripts/data/download_events_iowa_fishkill.py --nldi
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402
from sentinel.data.sources import events_iowa_fishkill as I  # noqa: E402

log = logging.getLogger("iowa_fishkill")
NAME = "labels_iowa_fishkill"


def _bundle(tmp_dir: Path, bundle: Path, suffix: str) -> Path:
    """Pack per-item cache files into one gzipped JSON-lines bundle {"id":..., "body":...}."""
    with gzip.open(bundle, "wt", encoding="utf-8") as out:
        for f in sorted(tmp_dir.glob(f"*{suffix}"), key=lambda p: int(p.name.split(".")[0])):
            opener = gzip.open if f.name.endswith(".gz") else open
            with opener(f, "rt", encoding="utf-8") as fh:
                out.write(json.dumps({"id": f.name.split(".")[0], "body": fh.read()}) + "\n")
    return bundle


def _read_bundle(bundle: Path) -> dict[str, str]:
    with gzip.open(bundle, "rt", encoding="utf-8") as fh:
        return {d["id"]: d["body"] for d in map(json.loads, fh)}


def step_snapshot(raw: Path, stamp: str, refresh: bool) -> tuple[Path, Path]:
    api = raw / "api" / f"events_strict_{stamp}.json"
    csv = raw / "api" / f"FishkillEvents_{stamp}.csv"
    if refresh or not api.exists():
        payload = I.fetch_events()
        C.save_json(payload, api)
        log.info("API: total=%s rows=%d", payload.get("total"), len(payload.get("rows", [])))
    C.download_file(I.CSV_EXPORT, csv, overwrite=refresh)
    spec = raw / "api" / "swagger_v1.json"
    C.download_file(I.SWAGGER, spec, overwrite=refresh)
    for name, url in I.DOC_PAGES.items():
        C.download_file(url, raw / "docs" / f"{name}.html", overwrite=refresh)
        time.sleep(0.5)
    return api, csv


def step_pages(raw: Path, stamp: str, ids: list[str], sleep: float) -> Path:
    bundle = raw / f"event_pages_{stamp}.jsonl.gz"
    if bundle.exists():
        return bundle
    tmp = raw / "pages_tmp"
    tmp.mkdir(exist_ok=True)
    todo = [i for i in ids if not (tmp / f"{i}.html.gz").exists()]
    log.info("pages: %d cached, %d to fetch", len(ids) - len(todo), len(todo))
    for k, eid in enumerate(todo, 1):
        try:
            page = I.fetch_event_page(eid)
        except C.HTTPFailure as e:
            log.error("page %s failed: %s", eid, e)
            continue
        with gzip.open(tmp / f"{eid}.html.gz", "wt", encoding="utf-8") as fh:
            fh.write(page)
        if k % 100 == 0:
            log.info("pages: %d/%d", k, len(todo))
        time.sleep(sleep)
    missing = [i for i in ids if not (tmp / f"{i}.html.gz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} event pages missing (re-run to resume): {missing[:10]}")
    _bundle(tmp, bundle, ".html.gz")
    for f in tmp.glob("*.html.gz"):
        f.unlink()
    tmp.rmdir()
    return bundle


def step_segments(raw: Path, stamp: str, seg_ids: list[str], sleep: float) -> Path:
    bundle = raw / f"adbnet_segments_{stamp}.jsonl.gz"
    if bundle.exists():
        return bundle
    tmp = raw / "segments_tmp"
    tmp.mkdir(exist_ok=True)
    todo = [s for s in seg_ids if not (tmp / f"{s}.json").exists()]
    log.info("segments: %d cached, %d to fetch", len(seg_ids) - len(todo), len(todo))
    for k, sid in enumerate(todo, 1):
        try:
            js = I.fetch_segment(sid)
        except C.HTTPFailure as e:
            log.error("segment %s failed: %s", sid, e)
            continue
        (tmp / f"{sid}.json").write_text(json.dumps(js))
        if k % 100 == 0:
            log.info("segments: %d/%d", k, len(todo))
        time.sleep(sleep)
    _bundle(tmp, bundle, ".json")
    for f in tmp.glob("*.json"):
        f.unlink()
    tmp.rmdir()
    return bundle


def step_nldi(raw: Path, stamp: str, ev: pd.DataFrame, sleep: float) -> Path:
    bundle = raw / f"nldi_position_{stamp}.jsonl.gz"
    if bundle.exists():
        return bundle
    tmp = raw / "nldi_tmp"
    tmp.mkdir(exist_ok=True)
    pts = ev.dropna(subset=["latitude", "longitude"])
    todo = [r for r in pts.itertuples() if not (tmp / f"{r.event_id}.json").exists()]
    log.info("nldi: %d cached, %d to fetch", len(pts) - len(todo), len(todo))
    for k, r in enumerate(todo, 1):
        try:
            js = I.fetch_nldi_comid(r.longitude, r.latitude)
        except C.HTTPFailure as e:
            log.error("nldi %s failed: %s", r.event_id, e)
            continue
        (tmp / f"{r.event_id}.json").write_text(json.dumps(js))
        if k % 50 == 0:
            log.info("nldi: %d/%d", k, len(todo))
        time.sleep(sleep)
    _bundle(tmp, bundle, ".json")
    for f in tmp.glob("*.json"):
        f.unlink()
    tmp.rmdir()
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", default=C.retrieval_date().replace("-", ""),
                    help="snapshot tag (YYYYMMDD); re-use an existing snapshot by passing its tag")
    ap.add_argument("--refresh", action="store_true", help="re-download the API snapshot for this stamp")
    ap.add_argument("--no-pages", action="store_true", help="skip HTML detail pages")
    ap.add_argument("--no-segments", action="store_true", help="skip ADBNet segment metadata")
    ap.add_argument("--nldi", action="store_true", help="add NLDI point-in-catchment COMIDs")
    ap.add_argument("--no-attains", action="store_true", help="skip the ATTAINS assessment-unit cross-check")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between page requests")
    a = ap.parse_args()

    raw = paths.raw("iowa_fishkill")
    out = paths.processed(NAME)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    log.info("start stamp=%s", a.stamp)

    api_path, csv_path = step_snapshot(raw, a.stamp, a.refresh)
    payload = json.loads(api_path.read_text())
    ev = I.parse_events(payload)
    seg_links = I.parse_event_segments(payload)
    csv = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if len(csv) != len(ev) or set(csv["id"]) != set(ev["event_id"]):
        log.warning("CSV export (%d rows) and API (%d rows) disagree", len(csv), len(ev))

    files = [api_path, csv_path, raw / "api" / "swagger_v1.json"]
    pages = comids = segs = None
    species = pd.DataFrame()
    if not a.no_pages:
        b = step_pages(raw, a.stamp, list(ev["event_id"]), a.sleep)
        files.append(b)
        pages = I.parse_event_pages(_read_bundle(b))
        species = I.species_counts(pages)
    if not a.no_segments:
        b = step_segments(raw, a.stamp, sorted(seg_links["segment_id"].unique(), key=int), a.sleep / 2)
        files.append(b)
        segs = I.parse_segments(json.loads(v) for v in _read_bundle(b).values())
        if not a.no_attains:
            att_path = raw / f"attains_au_{a.stamp}.json"
            if not att_path.exists():
                ids = sorted({i for i in segs["adb_code"].map(I.attains_au_id) if i})
                C.save_json(I.fetch_attains(ids), att_path)
                log.info("attains: %d AU ids queried", len(ids))
            files.append(att_path)
            segs = I.add_attains(segs, I.parse_attains(json.loads(att_path.read_text())))
    if a.nldi:
        b = step_nldi(raw, a.stamp, ev, max(a.sleep / 3, 0.2))
        files.append(b)
        rows = [{"event_id": k, "comid": I.parse_nldi_position(json.loads(v))}
                for k, v in _read_bundle(b).items()]
        comids = pd.DataFrame(rows)
        comids["event_id"] = C.str_id(comids["event_id"])
        comids["comid"] = C.str_id(comids["comid"])
        comids["comid_method"] = "nldi_comid_position_point_in_catchment_nhdplusv2"

    events = I.assemble_events(ev, pages, comids, species if pages is not None else None)
    outputs = {
        "events": C.write_parquet(events, out / "events.parquet"),
        "event_segments": C.write_parquet(seg_links, out / "event_segments.parquet"),
    }
    if segs is not None:
        outputs["segments"] = C.write_parquet(segs, out / "segments.parquet")
    if pages is not None:
        outputs["species_counts"] = C.write_parquet(species, out / "species_counts.parquet")

    stats = {
        "retrieval_stamp": a.stamp,
        "request": {"events": f"{I.API_EVENTS}?f=strict", "csv": I.CSV_EXPORT,
                    "pages": I.EVENT_PAGE, "segments": I.ADBNET_SEGMENT,
                    "nldi": I.NLDI_POSITION if a.nldi else None},
        "api_total": payload.get("total"),
        "csv_rows": int(len(csv)),
        **C.coverage_stats(events, date_col="date", lat_col="latitude"),
        "max_event_id": int(pd.to_numeric(events["event_id"]).max()),
        "cause_origin": events["cause_origin"].value_counts().to_dict(),
        "cause": events["cause"].value_counts().to_dict(),
        "waterbody_type": events["waterbody_type"].value_counts().to_dict(),
        "n_est_location": int(events["est_location"].sum()),
        "coord_flag": events["coord_flag"].value_counts().to_dict(),
        "n_est_fish_killed": int(events["est_fish_killed"].notna().sum()),
        "n_env_wq": int(events["is_env_wq"].sum()),
        "n_since_2000": int((events["year"] >= 2000).sum()),
    }
    if pages is not None:
        stats.update({"n_pages_parsed": int(len(pages)),
                      "n_page_date_mismatch": int(events["page_date_mismatch"].sum()),
                      "n_with_kill_length": int(events["kill_length_mi"].notna().sum()),
                      "n_with_comments": int(events["comments_text"].notna().sum()),
                      "n_events_with_species_table": int(species["event_id"].nunique()),
                      "species_table_check": events["species_table_check"].value_counts().to_dict(),
                      "n_species_rows": int((~species["is_total_row"]).sum()) if len(species) else 0})
    if segs is not None:
        stats.update({"n_segments": int(len(segs)), "n_segment_links": int(len(seg_links))})
        if "in_attains" in segs:
            stats.update({"n_segments_in_attains": int(segs["in_attains"].sum()),
                          "n_segments_attains_fishkill_cause": int(
                              segs["attains_cause_unknown_fish_kills"].isin(["Cause", "Y", "Yes"]).sum())})
    if comids is not None:
        stats.update({"n_comid": int(events["comid"].notna().sum())})
    write_manifest(NAME, files=list(outputs.values()) + files,
                   sources=[I.API_EVENTS, I.CSV_EXPORT, I.EVENT_PAGE, I.ADBNET_SEGMENT, I.SWAGGER]
                   + ([I.NLDI_POSITION] if a.nldi else [])
                   + ([] if a.no_attains or a.no_segments else list(I.ATTAINS_LAYERS.values())),
                   description="Iowa DNR Fish Kill Database: all released ('Final') fish-kill events "
                               "1981-present with cause/origin/magnitude, coordinates, detail-page fields "
                               "(kill length, valuation, comments, species tables), 305(b) segments.",
                   license="Iowa DNR public data; no explicit license; cite 'Iowa Dept. of Natural "
                           "Resources, "
                           "2013, Fish Kill Database, online at https://programs.iowadnr.gov/fishkill/, "
                           "accessed on <date>'; API termsOfService https://www.iowa.gov/policies",
                   stats=stats)
    log.info("done: %s", json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
