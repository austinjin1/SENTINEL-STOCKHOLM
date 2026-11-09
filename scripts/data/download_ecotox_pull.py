#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download EPA's pre-joined ECOTOX aquatic data pull and keep only its standardised columns (QC).

``https://gaftp.epa.gov/ecotox/data-pulls/aquatic_all.zip`` (one 617-column CSV) carries EPA's own
standardised concentration (``CONC1_MEAN_STD`` / ``CONC1_UNITS_STD``) and duration
(``TEST_DURATION_STD`` / ``DURATION_UNITS_STD``) per result. SENTINEL does not use these values as
data — they are an independent check of the unit standardisation in
``sentinel/data/chemtox_ecotox.py`` (see build_ecotox.py, ``summary.json: epa_crosscheck``). The CSV
is streamed out of the zip; only the subset Parquet is kept and the zip is deleted (> 100 MB).

    python -B scripts/data/download_ecotox_pull.py [--keep-zip]
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chemtox_http as H  # noqa: E402
from sentinel.data.sources import ecotox as E  # noqa: E402

log = logging.getLogger("download_ecotox_pull")
URL = E.INDEX_URL + "data-pulls/aquatic_all.zip"
KEEP = [
    "RESULT_NUMBER",
    "TEST_NUMBER",
    "TEST_CAS",
    "DTXSID",
    "ENDPOINT",
    "CONC1_TYPE",
    "CONC1_MEAN_OP",
    "CONC1_MEAN",
    "CONC1_UNIT",
    "CONC1_MEAN_STD_OP",
    "CONC1_MEAN_STD",
    "CONC1_UNITS_STD",
    "CONC1_TYPE_STD",
    "TEST_DURATION_STD_OP",
    "TEST_DURATION_STD",
    "DURATION_UNITS_STD",
    "DATA_VERSION",
    "EXPORT_DATE",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args()
    out = paths.ensure(paths.raw("ecotox") / "data_pull")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()],
    )
    s = H.make_session()
    head = H.get(URL, session=s, method="HEAD")
    size = int(head.headers["Content-Length"]) if head.headers.get("Content-Length") else None
    zpath = out / "aquatic_all.zip"
    rec = H.download(URL, zpath, expected_size=size, session=s)
    log.info(
        "aquatic_all.zip: %d bytes sha256 %s (Last-Modified %s)",
        rec["bytes"],
        rec["sha256"],
        head.headers.get("Last-Modified"),
    )
    with zipfile.ZipFile(zpath) as z:
        member = next(i for i in z.infolist() if i.filename.lower().endswith(".csv"))
        with z.open(member) as fh:
            rd = csv.reader(io.TextIOWrapper(fh, encoding="latin-1", newline=""))
            header = next(rd)
            idx = [header.index(c) for c in KEEP if c in header]
            names = [header[i] for i in idx]
            rows, bad = [], 0
            for row in rd:
                if len(row) != len(header):
                    bad += 1
                    continue
                rows.append([row[i] for i in idx])
    df = pd.DataFrame(rows, columns=names, dtype="string").replace("", pd.NA)
    version = (
        str(df["DATA_VERSION"].dropna().iloc[0])
        if "DATA_VERSION" in df.columns and df["DATA_VERSION"].notna().any()
        else "unknown"
    )
    dest = out / f"aquatic_all_{version}_std_subset.parquet"
    df.to_parquet(dest, index=False, compression="zstd")
    log.info("%s: %d rows (%d malformed rows skipped), %d columns kept", dest.name, len(df), bad, len(names))
    write_manifest(
        "ecotox_pull_subset",
        files=[dest],
        sources=[URL],
        description="EPA ECOTOX aquatic data pull: standardised concentration/duration columns only "
        "(QC reference for SENTINEL's unit standardisation; not used as data).",
        license=E.LICENSE,
        stats={
            "zip_bytes": rec["bytes"],
            "zip_sha256": rec["sha256"],
            "csv_member": member.filename,
            "csv_bytes": member.file_size,
            "rows": int(len(df)),
            "malformed_rows_skipped": bad,
            "data_version": version,
            "http_last_modified": head.headers.get("Last-Modified"),
            "columns": names,
            "zip_deleted": not a.keep_zip,
        },
    )
    if not a.keep_zip:
        zpath.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
