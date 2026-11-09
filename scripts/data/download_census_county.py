#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the 2010 Census county gazetteer -> processed('census_county')/counties_2010.parquet.

    python -B scripts/data/download_census_county.py
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import census_county as CC  # noqa: E402


def main() -> int:
    raw = paths.raw("census_county")
    zp = raw / "Gaz_counties_national_2010.zip"
    if not zp.exists():
        r = requests.get(CC.URL, headers={"User-Agent": CC.USER_AGENT}, timeout=120)
        r.raise_for_status()
        zp.write_bytes(r.content)
    df = CC.parse_gazetteer(zp.read_bytes())
    out = paths.processed("census_county") / "counties_2010.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    write_manifest("census_county", files=[out, zp], sources=[CC.URL], license="US Government work (public domain)",
                   description="2010 Census county gazetteer: internal point, land area, 2010 population/housing",
                   synthetic=False, stats={"n_counties": int(len(df)), "pop_total": int(df["pop2010"].sum()),
                                           "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    print(len(df), int(df["pop2010"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
