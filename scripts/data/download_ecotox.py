#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the latest EPA ECOTOX quarterly ASCII release and convert every table to Parquet.

The newest ``ecotox_ascii_MM_DD_YYYY.zip`` linked from https://gaftp.epa.gov/ecotox/ is downloaded
(EPA publishes no checksum; size is checked against Content-Length and the sha256 is recorded),
each pipe-delimited table becomes $SENTINEL_DATA/raw/ecotox/release_<YYYY-MM-DD>/<table>.parquet
(all strings), and row/column counts are checked against the release's own release_notes.txt.
The 137 MB zip is deleted afterwards unless ``--keep-zip`` (manifest keeps URL + sha256).

    python -B scripts/data/download_ecotox.py [--release 2026-09-15] [--keep-zip]
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
from sentinel.data import chemtox_http as H  # noqa: E402
from sentinel.data.sources import ecotox as E  # noqa: E402

log = logging.getLogger("download_ecotox")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=None, help="YYYY-MM-DD; default = newest on the index page")
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args()
    out = paths.raw("ecotox")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()],
    )
    s = H.make_session()
    index_html = H.get(E.INDEX_URL, session=s).text
    releases = E.releases_from_index(index_html)
    log.info("releases on index: %s", releases)
    if a.release:
        match = [r for r in releases if r[0] == a.release]
        if not match:
            raise SystemExit(f"release {a.release} not on the index page")
        rel_date, fname = match[0]
    else:
        rel_date, fname = E.latest_release(index_html)
    url = E.INDEX_URL + fname
    head = H.get(url, session=s, method="HEAD")
    size = int(head.headers["Content-Length"]) if head.headers.get("Content-Length") else None
    rec = H.download(url, out / fname, expected_size=size, session=s)
    rec["http_last_modified"] = rec.get("http_last_modified") or head.headers.get("Last-Modified")
    log.info("%s: %d bytes sha256 %s", fname, rec["bytes"], rec["sha256"])
    tdir = out / f"release_{rel_date}"
    report = E.convert_release_zip(out / fname, tdir, log=log)
    mismatches = [t for t, m in report["tables"].items() if m.get("counts_match") is False]
    unchecked = [t for t, m in report["tables"].items() if not m.get("expected_found")]
    if unchecked:
        log.warning("tables missing from release notes (counts not checked): %s", unchecked)
    if mismatches:
        log.error("row/column counts differ from release notes for: %s", mismatches)
    report["download"] = rec
    report["release_date"] = rel_date
    (tdir / "conversion_report.json").write_text(json.dumps(report, indent=2))
    files = [Path(m["path"]) for m in report["tables"].values()] + [Path(o["path"]) for o in report["other"]]
    files.append(tdir / "conversion_report.json")
    write_manifest(
        "ecotox_raw",
        files=files,
        sources=[E.INDEX_URL, url],
        description=f"EPA ECOTOX Knowledgebase ASCII release {rel_date}: every pipe-delimited table "
        "converted to all-string Parquet; counts checked against release_notes.txt.",
        license=E.LICENSE,
        stats={
            "release": rel_date,
            "zip": fname,
            "zip_bytes": rec["bytes"],
            "zip_sha256": rec["sha256"],
            "http_last_modified": rec.get("http_last_modified"),
            "retrieved_utc": rec.get("retrieved_utc"),
            "tables_not_in_release_notes": unchecked,
            "tables": {
                t: {k: m[k] for k in ("rows", "columns", "bad_lines", "repaired_lines", "encoding")}
                | {"counts_match": m.get("counts_match")}
                for t, m in report["tables"].items()
            },
            "count_mismatches": mismatches,
            "zip_deleted_after_conversion": not a.keep_zip,
        },
    )
    if not a.keep_zip:
        (out / fname).unlink()
        log.info("deleted %s after conversion", fname)
    log.info("done")
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
