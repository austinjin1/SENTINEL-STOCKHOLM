#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Overlap of the reserved USGS new-site block with other workstreams' stations (METADATA ONLY; no data value).

Checks, for every site of ``docs/preregistration/sonde_newsites_list.json``:
  1. exact identifier matches (``USGS-<no>``, ``WQP:USGS-<no>`` and bare 8–15-digit tokens) in every
     ``docs/preregistration/*.json`` (incl. ``hydromet_track_n_stations.json`` role_of keys);
  2. spatial co-location with M6 track-N WQP stations (``processed/hydromet_wqp_national/stations.parquet``
     coordinates, roles from ``hydromet_track_n_stations.json``) within 100 m / 500 m / 1 km;
  3. presence of the site id in other workstreams' processed station / label tables (catalogue exposure, not splits);
  4. sites actually LOADED by any recorded run (``sentinel_runs/**/data_info.json`` site lists).
Writes ``docs/preregistration/sonde_newsites_overlap.json`` (refuses to overwrite). The pre-declared sensitivity
exclusion set of EW-v3-N / M1-N = sites with ANY exact match (1) OR a track-N station (any role) within 1 km (2).

    python -B scripts/data/newsite_overlap.py
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import re

import numpy as np
import pandas as pd

from sentinel.core import paths

PREREG = paths.REPO_ROOT / "docs" / "preregistration"
OUT = PREREG / "sonde_newsites_overlap.json"
TABLES = ("hydromet_wqp_national/stations.parquet", "s2_wqp_matchups/sites.parquet", "s2_wqp_matchups_hist/sites.parquet",
          "labels_powell_metabolism/sites.parquet", "wqp_discrete/_stations.parquet")
ID_COLS = ("site_id", "site_no", "site_key", "station_key", "MonitoringLocationIdentifier", "monitoring_location_id",
           "nwis_id")
LABEL_TABLES = ("labels_powell_metabolism/sites.parquet",)   # DO-derived labels of another workstream -> exclusion


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    lst = json.loads((PREREG / "sonde_newsites_list.json").read_text())
    new = pd.DataFrame(lst["sites"]).set_index("site_no")
    ids = set(new.index)
    exact = {}
    for f in sorted(glob.glob(str(PREREG / "*.json"))):
        n = f.split("/")[-1]
        if n in ("sonde_newsites_list.json", OUT.name):
            continue
        txt = open(f).read()
        toks = set(re.findall(r"USGS-(\d{7,15})", txt)) | set(re.findall(r"(?<!\d)(\d{8,15})(?!\d)", txt))
        hit = sorted(ids & toks)
        if hit:
            exact[n] = hit
    tn = json.loads((PREREG / "hydromet_track_n_stations.json").read_text())["role_of"]
    st = pd.read_parquet(paths.PROCESSED / "hydromet_wqp_national" / "stations.parquet")
    st["role"] = st["station_key"].map(tn)
    st = st[st["role"].notna() & st["lat"].notna() & st["lon"].notna()]
    from sklearn.neighbors import BallTree

    bt = BallTree(np.radians(st[["lat", "lon"]].astype(float).to_numpy()), metric="haversine")
    a = np.radians(new[["lat", "lon"]].astype(float).to_numpy())
    near = {}
    for r in (100, 500, 1000):
        ind = bt.query_radius(a, r=r / 6371000.0)
        near[f"{r}m"] = {new.index[i]: sorted(set(st["role"].iloc[j])) for i, j in enumerate(ind) if len(j)}
    cat = {}
    for t in TABLES:
        p = paths.PROCESSED / t
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        vals = set()
        for c in [c for c in df.columns if c in ID_COLS]:
            vals |= set(df[c].astype(str).str.extract(r"USGS-(\d{7,15})")[0].dropna())
            vals |= set(df[c].astype(str).str.fullmatch(r"\d{8,15}").pipe(lambda m: df.loc[m, c].astype(str)))
        cat[t] = sorted(ids & vals)
    loaded = set()
    for f in glob.glob(str(paths.RUNS_ROOT / "**" / "data_info.json"), recursive=True):
        try:
            s_ = json.loads(open(f).read()).get("sites")
        except Exception:  # noqa: BLE001
            continue
        if s_:
            loaded |= set(map(str, s_)) & ids
    lab = set(s for t in LABEL_TABLES for s in cat.get(t, []))
    excl = sorted(set(s for v in exact.values() for s in v) | set(near["1000m"]) | lab | loaded)
    rec = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "site_list_sha256": hashlib.sha256((PREREG / "sonde_newsites_list.json").read_bytes()).hexdigest(),
           "exact_id_matches_in_preregistration_files": {k: (len(v), v) for k, v in exact.items()},
           "track_n_colocated": {k: {"n_sites": len(v), "n_with_test_role_station": sum("test" in x for x in v.values()),
                                     "sites": v} for k, v in near.items()},
           "catalogue_exposure_not_splits": {k: (len(v), v) for k, v in cat.items()},
           "loaded_by_any_recorded_run": sorted(loaded),
           "exclusion_set": {"rule": "exact match in any preregistration file (incl. the DEV split) OR track-N station "
                                     "(any role) within 1 km OR in a DO-derived label table of another workstream OR "
                                     "loaded by any recorded run", "n_sites": len(excl), "sites": excl}}
    OUT.write_text(json.dumps(rec, indent=1))
    OUT.chmod(0o444)
    print(json.dumps({"sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      "exact": {k: v[0] for k, v in rec["exact_id_matches_in_preregistration_files"].items()},
                      "track_n_colocated": {k: (v["n_sites"], v["n_with_test_role_station"]) for k, v in rec["track_n_colocated"].items()},
                      "catalogue": {k: v[0] for k, v in rec["catalogue_exposure_not_splits"].items()},
                      "loaded": len(loaded), "exclusion_n": len(excl)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
