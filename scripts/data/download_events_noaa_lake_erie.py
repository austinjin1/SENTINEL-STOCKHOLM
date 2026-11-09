#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download every published vintage of the NOAA NCCOS western Lake Erie bloom Severity Index.

Raw files go to $SENTINEL_DATA/raw/noaa_lake_erie_hab/ with the server's Last-Modified date in
the file name, so a later in-place update by NOAA becomes a NEW file instead of silently
replacing the frozen label vintage. Bulletins that no longer resolve at the official URL are
fetched from the Internet Archive capture listed in the client (flagged ``retrieved_via``).

    python -B scripts/data/download_events_noaa_lake_erie.py
"""

from __future__ import annotations

import argparse
import email.utils
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
from sentinel.data.sources import events_common as C  # noqa: E402
from sentinel.data.sources import events_noaa_lake_erie as N  # noqa: E402

log = logging.getLogger("noaa_lake_erie")
NAME = "labels_noaa_lake_erie_hab"


def _lm_date(last_modified: str | None) -> str:
    if not last_modified:
        return "nodate"
    return email.utils.parsedate_to_datetime(last_modified).date().isoformat()


def fetch(url: str, dest_dir: Path, prefix: str = "", wayback: str | None = None) -> dict:
    """Download ``url`` (or its Internet Archive capture if the official URL is gone)."""
    head = C.http_head(url)
    via, src = "official", url
    if head["status"] != 200:
        if not wayback:
            return {"url": url, "status": head["status"], "path": None, "retrieved_via": None}
        src = N.WAYBACK.format(ts=wayback, url=url)
        via = f"web.archive.org capture {wayback} (official URL returned HTTP {head['status']})"
        head = {"status": 200, "last_modified": None, "etag": None, "content_length": None}
    name = Path(url).name
    tag = _lm_date(head.get("last_modified")) if via == "official" else f"wayback{wayback}"
    dest = dest_dir / f"{prefix}{tag}_{name}"
    C.download_file(src, dest)
    rec = {"url": url, "download_url": src, "status": 200, "path": str(dest), "retrieved_via": via,
           "last_modified": head.get("last_modified"), "etag": head.get("etag"),
           "bytes": dest.stat().st_size, "sha256": file_sha256(dest),
           "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    log.info("%s -> %s (%s)", url, dest.name, via)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    raw = paths.raw("noaa_lake_erie_hab")
    out = paths.processed(NAME)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])

    recs = []
    page = raw / "product_page" / f"lake_erie_forecast_page_{C.retrieval_date()}.html"
    C.download_file(N.PRODUCT_PAGE, page)
    for f in N.SEVERITY_FILES:
        r = fetch(f["url"], raw / "severity")
        r.update(label_vintage=f["label_vintage"], kind="severity_xlsx")
        recs.append(r)
        time.sleep(a.sleep)
    for b in N.BULLETINS:
        r = fetch(b["url"], raw / "bulletins", prefix=f"{b['season']}_{b['kind']}_", wayback=b.get("wayback"))
        r.update(season=b["season"], kind=b["kind"])
        recs.append(r)
        time.sleep(a.sleep)
    C.save_json(recs, raw / f"retrieval_{C.retrieval_date()}.json")
    missing = [r["url"] for r in recs if not r.get("path")]
    if missing:
        log.warning("not retrievable: %s", missing)

    # --- severity spreadsheets (one row per vintage × year)
    frames, metas = [], {}
    for r in recs:
        if r.get("kind") != "severity_xlsx" or not r.get("path"):
            continue
        df, meta = N.parse_severity_xlsx(r["path"], r["label_vintage"])
        df["vintage_last_modified"] = pd.to_datetime(_lm_date(r["last_modified"]), errors="coerce")
        df["file_url"] = r["url"]
        df["file_sha256"] = r["sha256"]
        frames.append(df)
        metas[r["label_vintage"]] = meta
    sev = N.severity_long(frames)

    # --- bulletin text → as-issued SI and seasonal forecasts
    texts_dir = raw / "bulletins_text"
    texts_dir.mkdir(exist_ok=True)
    issued, fcst = [], []
    for r in recs:
        if r.get("kind") not in ("final_assessment", "seasonal_forecast") or not r.get("path"):
            continue
        txt = N.pdf_to_text(r["path"])
        (texts_dir / (Path(r["path"]).stem + ".txt")).write_text(txt)
        if r["kind"] == "final_assessment":
            t = N.parse_final_assessment(txt, r["season"])
            t["doc_url"], t["retrieved_via"] = r["url"], r["retrieved_via"]
            issued.append(t)
        else:
            d = N.parse_seasonal_forecast(txt, r["season"])
            d.update(doc_url=r["url"], retrieved_via=r["retrieved_via"])
            fcst.append(d)
    as_issued = pd.concat(issued, ignore_index=True)
    as_issued["year"] = as_issued["year"].astype("Int64")
    as_issued["season"] = as_issued["season"].astype("Int64")
    forecasts = pd.DataFrame(fcst).sort_values("season").reset_index(drop=True)
    comp = N.vintage_comparison(sev, as_issued)

    outputs = [
        C.write_parquet(sev, out / "severity_index.parquet"),
        C.write_parquet(as_issued, out / "severity_as_issued.parquet"),
        C.write_parquet(forecasts, out / "seasonal_forecasts.parquet"),
        C.write_parquet(comp, out / "severity_vintage_comparison.parquet"),
    ]
    C.save_json(metas, out / "severity_metadata_sheets.json")
    outputs.append(out / "severity_metadata_sheets.json")

    stats = {
        "retrieval_date": C.retrieval_date(),
        "vintages": {v: {"years": f"{int(g.year.min())}-{int(g.year.max())}", "n_years": int(len(g)),
                         "last_modified": str(g["vintage_last_modified"].iloc[0].date())}
                     for v, g in sev.groupby("label_vintage")},
        "n_as_issued_primary": int((as_issued["statement_type"] == "primary").sum()),
        "as_issued_primary_seasons": sorted(int(x) for x in as_issued.loc[
            as_issued["statement_type"] == "primary", "season"].unique()),
        "n_forecasts_parsed": int(forecasts["parse_ok"].sum()),
        "n_si_floor_censored_current": int(sev.loc[sev.label_vintage == "curated_2000to2025",
                                                   "si_floor_censored"].sum()),
        "not_retrievable": missing,
        "retrieved_via_wayback": [r["url"] for r in recs if r.get("retrieved_via", "") and
                                  str(r["retrieved_via"]).startswith("web.archive")],
    }
    raw_files = [Path(r["path"]) for r in recs if r.get("path")]
    raw_files += [page, raw / f"retrieval_{C.retrieval_date()}.json"]
    write_manifest(NAME, files=outputs + raw_files,
                   sources=[N.PRODUCT_PAGE] + [r["url"] for r in recs],
                   description="NOAA NCCOS western Lake Erie cyanoHAB bloom Severity Index: current curated "
                               "2000-2025 file, first reprocessed 2000-2024 file, as-issued values from "
                               "season-end bulletins, and seasonal forecasts. Label vintages kept separate.",
                   license="US Government work (NOAA), public domain; attribution requested: 'The Lake Erie "
                           "Severity Index (SI) is derived from NOAA's National Centers for Coastal Ocean "
                           "Science (NCCOS) HAB Monitoring Program.'",
                   stats=stats)
    log.info("done: %s", json.dumps(stats, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
