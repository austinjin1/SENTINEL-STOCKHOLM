#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download Delaware DNREC non-tidal fish-kill reports (data.delaware.gov 8pfi-yck7) -> events.

Two requests (metadata + full table, $limit 50000), generic User-Agent. Raw JSON kept under
raw('delaware_fishkill')/<YYYYMMDD>/; processed('labels_delaware_fishkill')/{rows,events}.parquet;
manifest `labels_delaware_fishkill`.

    python -B scripts/data/download_events_delaware_fishkill.py
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import events_delaware_fishkill as DE  # noqa: E402


def main() -> int:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    raw = paths.ensure(paths.raw("delaware_fishkill") / stamp)
    out = paths.processed("labels_delaware_fishkill")
    s = requests.Session()
    s.headers["User-Agent"] = DE.USER_AGENT
    meta = s.get(DE.METADATA, timeout=120)
    meta.raise_for_status()
    (raw / "metadata.json").write_text(meta.text)
    r = s.get(DE.RESOURCE, params={"$limit": 50000, "$order": ":id"}, timeout=300)
    r.raise_for_status()
    rows = r.json()
    (raw / "rows.json").write_text(json.dumps(rows))
    df = DE.parse_rows(rows)
    ev = DE.to_events(df)
    df.drop(columns=["lat", "long"]).to_parquet(out / "rows.parquet", index=False, compression="zstd")
    ev.to_parquet(out / "events.parquet", index=False, compression="zstd")
    md = meta.json()
    stats = {"retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "n_rows": len(df),
             "n_events": len(ev), "n_events_geocoded": int(ev["latitude"].notna().sum()),
             "years": [int(ev["year"].min()), int(ev["year"].max())],
             "cause_rule": ev["cause_rule"].value_counts().to_dict(), "cause": ev["cause"].value_counts().to_dict(),
             "waterbody_type": ev["waterbody_type"].value_counts().to_dict(),
             "source_rows_updated_at": md.get("rowsUpdatedAt"), "licence_field": md.get("license")}
    write_manifest("labels_delaware_fishkill", files=[out / "rows.parquet", out / "events.parquet", raw / "rows.json",
                                                      raw / "metadata.json"],
                   sources=[DE.RESOURCE, DE.METADATA], license="Public Domain (data.delaware.gov licence field)",
                   description="Delaware DNREC non-tidal fish-kill reports, de-duplicated to events, causes "
                               "harmonised to Iowa classes by keyword rules", synthetic=False, stats=stats)
    print(json.dumps(stats, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
