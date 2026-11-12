#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Header check of ``water_chem.csv`` in Zenodo record 18668846 (lead ruling L4, M3 architecture round v2).

What it does, and nothing more:

* ONE HTTP GET with ``Range: bytes=0-8191`` to the record's public file link (no credential, no cookie, no key
  file). A redirect of that request to another https address is followed by the HTTP library with the same Range
  header; a redirect to anything else is refused.
* The response body is read in pieces of 256 bytes. Reading stops at the first newline. Everything from that
  newline on is dropped at once and the connection is closed; no data row is kept, parsed, printed or stored.
  If no newline arrives within 8,192 bytes the script stops and stores nothing.
* The bytes before the first newline are split into column names with the csv module. The names are printed and
  written to the output file (JSON: record, file name, column names). Nothing else is written.
* It runs ONCE: it refuses to start if the output file exists, and it creates that file exclusively.

It imports the standard library only and reads no project file. Run it isolated, after the lead's written
approval of this file's sha256:

    python -I -B scripts/data/microbe_ea_header_check.py

Exit codes: 0 header stored; 2 refused (already run) or no usable header; 3 network or HTTP error.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RECORD = 18668846
FILE = "water_chem.csv"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{FILE}/content"
RANGE_END = 8191                     # inclusive: at most 8,192 bytes are requested
PIECE = 256
MAX_COLUMNS = 400
MAX_NAME_CHARS = 120
OUT = Path(__file__).resolve().parents[2] / "docs" / "preregistration" / "microbe_arch_v2_evidence" / \
    "ea_water_chem_header.json"


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise urllib.error.HTTPError(req.full_url, code, "redirect to a non-https address refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def first_line(resp) -> bytes | None:
    """The bytes before the first newline, or None if none arrives within the requested range."""
    kept = b""
    while len(kept) <= RANGE_END:
        piece = resp.read(PIECE)
        if not piece:
            return None
        cut = piece.find(b"\n")
        if cut >= 0:
            kept += piece[:cut]          # the rest of the piece is dropped here
            del piece
            return kept
        kept += piece
    return None


def column_names(line: bytes) -> list[str]:
    text = line.decode("utf-8-sig", "replace").rstrip("\r")
    names = next(csv.reader([text]), [])
    return ["".join(ch for ch in n if ch.isprintable())[:MAX_NAME_CHARS] for n in names][:MAX_COLUMNS]


def main() -> int:
    if OUT.exists():
        print(f"refused: {OUT} exists; the header check runs once", file=sys.stderr)
        return 2
    opener = urllib.request.build_opener(_HttpsOnly())
    req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{RANGE_END}", "User-Agent": "sentinel-m3-header-check",
                                               "Accept-Encoding": "identity"})
    try:
        with opener.open(req, timeout=60) as resp:
            line = first_line(resp)
    except (urllib.error.URLError, OSError) as e:
        print(f"network or HTTP error: {e}", file=sys.stderr)
        return 3
    if line is None:
        print("no newline within the requested range; nothing stored", file=sys.stderr)
        return 2
    names = column_names(line)
    del line
    if not names:
        print("empty header; nothing stored", file=sys.stderr)
        return 2
    with OUT.open("x") as fh:
        json.dump({"zenodo_record": RECORD, "file": FILE, "columns": names}, fh, indent=1)
        fh.write("\n")
    for n in names:
        print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
