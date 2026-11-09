#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the USCG National Response Center FOIA incident workbooks (CY90.xlsx ... CY{current}.xlsx).

One workbook per calendar year (1990-present, ~10-18 MB each, 10 sheets) plus DataDictionary.xlsx
from https://nrc.uscg.mil/FOIAFiles/. Sequential, 1 s apart. A file is re-downloaded only when the
server's Last-Modified / Content-Length differ from the sidecar record of the previous download
(the current-year file is updated continuously). Raw workbooks are kept (each < 100 MB) under
$SENTINEL_DATA/raw/nrc/; manifest ``nrc_raw``.

    python -B scripts/data/download_nrc.py [--first 1990] [--last 2026]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chemtox_http as H  # noqa: E402
from sentinel.data.sources import nrc as N  # noqa: E402

log = logging.getLogger("download_nrc")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1990)
    ap.add_argument("--last", type=int, default=dt.date.today().year)
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    out = paths.raw("nrc")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()],
    )
    s = H.make_session()
    sidecar_path = out / "downloads.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    targets = [(N.DICTIONARY_URL, out / "DataDictionary.xlsx")]
    targets += [(N.year_url(y), out / N.year_filename(y)) for y in range(a.first, a.last + 1)]
    for url, dest in targets:
        head = H.get(url, session=s, method="HEAD", pass_status=(404,))
        if head.status_code == 404:
            log.warning("404 (not published): %s", url)
            continue
        lm, cl = head.headers.get("Last-Modified"), head.headers.get("Content-Length")
        prev = sidecar.get(dest.name)
        if (
            dest.exists()
            and prev
            and prev.get("http_last_modified") == lm
            and str(prev.get("bytes")) == str(cl)
            and dest.stat().st_size == prev.get("bytes")
        ):
            log.info("%s unchanged (Last-Modified %s); kept", dest.name, lm)
            continue
        rec = H.download(url, dest, session=s, skip_existing=False)
        rec["http_last_modified"] = rec.get("http_last_modified") or lm
        sidecar[dest.name] = rec
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        log.info("%s: %d bytes sha256 %s (Last-Modified %s)", dest.name, rec["bytes"], rec["sha256"][:12], lm)
        time.sleep(a.sleep)
    files = sorted(out.glob("*.xlsx")) + [sidecar_path]
    write_manifest(
        "nrc_raw",
        files=files,
        sources=[N.BASE_URL] + [u for u, _ in targets],
        description="USCG National Response Center FOIA incident workbooks, one per calendar "
        "year (initial, unvalidated reports), plus the NRC data dictionary.",
        license="US Government work (no license stated)",
        stats={
            "years": [a.first, a.last],
            "downloads": sidecar,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    )
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
