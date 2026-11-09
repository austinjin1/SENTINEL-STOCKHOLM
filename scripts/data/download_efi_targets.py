#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download EFI forecasting-challenge target files (neon4cast aquatics, VERA, usgsrc4cast)."""

from __future__ import annotations

import datetime as dt

import requests

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.data.sources.efi import TARGETS, read_targets


def main() -> int:
    out = paths.raw("efi")
    files, stats = [], {}
    stamp = dt.date.today().isoformat()
    for name, url in TARGETS.items():
        dest = out / f"{name}_{stamp}.csv.gz"
        if not dest.exists():
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
        df = read_targets(dest)
        stats[name] = {"rows": int(len(df)), "sites": int(df["site_id"].nunique()),
                       "start": str(df["datetime"].min()), "end": str(df["datetime"].max()),
                       "variables": sorted(df["variable"].unique().tolist())[:60]}
        files.append(dest)
    write_manifest("efi_targets", files=files, sources=list(TARGETS.values()),
                   description="EFI challenge daily targets (neon4cast aquatics, VERA, usgsrc4cast) "
                               f"retrieved {stamp}.", license="see challenge terms (unverified)",
                   stats=stats)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
