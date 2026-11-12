#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D7-ONLY dedicated fetch of EPA StreamCat modelled stream temperature (MAST / MSST / MWST, 2008, 2009, 2013, 2014;
Hill et al. 2013) for NRSA site COMIDs (lead decision 2026-11-09: comparator score only).

The shared StreamCat client excludes these metrics on purpose and is NOT modified: this script builds the POST form
itself and only reuses the client's transport/cache class. Output: processed('d7_mast')/mast_comid.parquet; manifest
``d7_mast``. **No workstream other than D7 TroutRefuge may read ``d7_mast``** (docs/datasheets/d7_mast.md).

    python -B scripts/data/fetch_mast_d7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import streamcat as SC  # noqa: E402

NAMES = [f"{m}{y}" for m in ("mast", "msst", "mwst") for y in (2008, 2009, 2013, 2014)]
OUT = paths.PROCESSED / "d7_mast"


def main() -> int:
    links = pd.read_parquet(paths.PROCESSED / "downstream_catchment" / "unit_links.parquet")
    comids = sorted(links.loc[links["unit_source"] == "nrsa", "comid"].dropna().astype(str).unique(), key=int)
    client = SC.StreamCatClient(paths.RAW / "d7_mast" / "streamcat_cache")
    frames = []
    for b in range(0, len(comids), 500):
        data = {"name": ",".join(NAMES), "aoi": "other", "comid": ",".join(comids[b:b + 500])}
        frames.append(SC.parse_metrics(client.post(SC.METRICS_URL, data)["body"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("comid")
    df = pd.DataFrame({"comid": comids}).merge(df[["comid"] + [c for c in NAMES if c in df.columns]], on="comid", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "mast_comid.parquet", index=False)
    stats = {"n_comids": int(len(df)), "non_null": {c: int(df[c].notna().sum()) for c in df.columns if c != "comid"},
             "restricted_use": "D7 TroutRefuge comparator score only (lead decision 2026-11-09)"}
    write_manifest("d7_mast", files=[OUT / "mast_comid.parquet"], sources=[SC.METRICS_URL + " (POST, aoi=other)"],
                   description="EPA StreamCat modelled stream temperature MAST/MSST/MWST (Hill et al. 2013) for NRSA COMIDs; "
                               "D7-only comparator score, never a model feature",
                   license="US public domain (EPA StreamCat)", synthetic=False, stats=stats)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
