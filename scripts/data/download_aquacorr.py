#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download AquaCorr-Global v1.0 from Zenodo (record 21927415, CC-BY-4.0) into raw('aquacorr_global').

Files: matchups parquet (377 MB), metadata json, GLORIA validation parquet, Rrs zarr zip (89 MB).
The csv.gz export duplicates the parquet and is skipped. md5 checksums from the Zenodo API are
verified; downloads are resumable. Audit + tidy conversion: scripts/data/build_aquacorr.py.

    python -B scripts/data/download_aquacorr.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import aquacorr as A  # noqa: E402

log = logging.getLogger("aquacorr_dl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=list(A.WANTED_FILES))
    a = ap.parse_args()
    out = paths.raw("aquacorr_global")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()])
    rec = A.fetch_record()
    (out / "zenodo_record.json").write_text(json.dumps(rec, indent=1))
    files = {f["key"]: f for f in rec.get("files", [])}
    log.info("record %s: %d files; license=%s", rec.get("doi"), len(files),
             (rec.get("metadata", {}).get("license") or {}).get("id"))
    got = []
    for key in a.files:
        if key not in files:
            log.error("file %s not in Zenodo record (have: %s)", key, sorted(files))
            continue
        f = files[key]
        md5 = (f.get("checksum") or "").split(":", 1)[-1] or None
        url = f["links"]["self"]
        log.info("downloading %s (%.1f MB)", key, f["size"] / 1e6)
        dest = A.download_file(url, out / key, md5, log=log.info)
        got.append(dest)
        log.info("ok %s", dest)
    write_manifest(
        "aquacorr_global_raw", files=got + [out / "zenodo_record.json"],
        sources=[A.RECORD_API] + [files[k]["links"]["self"] for k in a.files if k in files],
        description="AquaCorr-Global v1.0 raw files from Zenodo (matchups parquet, metadata, GLORIA "
                    "validation subset, Rrs zarr archive). md5 verified against the Zenodo API.",
        license=A.LICENSE, synthetic=False,
        stats={"zenodo_record": A.ZENODO_RECORD, "doi": A.ZENODO_DOI,
               "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "files": {k: files[k]["size"] for k in a.files if k in files}})
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
