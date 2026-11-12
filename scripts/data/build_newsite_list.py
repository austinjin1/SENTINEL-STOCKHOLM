#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Frozen list of the USGS "new-site block" (EW-v3-N / M1-N unseen test population) — METADATA ONLY.

Rule (docs/preregistration/ew_v3_protocol.md §2, m1n_protocol.md §2; lead approval 2026-11-09):
  1. USGS sites with a daily (00001/00002/00003) or instantaneous (00011) series of DO (00300), specific
     conductance (00095) or NO3+NO2 (99133) covering ≥ 1 year inside WY2008–WY2025 (series begin/end metadata;
     USGS Water Data OGC API ``time-series-metadata``, retrieved at build time).
  2. NOT in ``docs/preregistration/sonde_split.json`` (any role) and NOT named in any other
     ``docs/preregistration/*.json`` file of any workstream (conservative: any 8–15-digit token match excludes).
  3. Site type (OGC ``monitoring-locations``) in streams {ST, ST-TS, ST-CA, ST-DCH}, lakes/reservoirs {LK}, estuaries
     {ES}, wetlands {WE}; groundwater, springs, atmosphere, facilities and unknown types are excluded.
No data value is read. Output: ``docs/preregistration/sonde_newsites_list.json`` (+ its sha256 printed); refuses to
overwrite.

    python -B scripts/data/build_newsite_list.py
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import re
import time

import pandas as pd
import requests

from sentinel.core import paths
from sentinel.platform.config import USER_AGENT, USGS_OGC_BASE
from sentinel.platform.live import read_api_key

PREREG = paths.REPO_ROOT / "docs" / "preregistration"
OUT = PREREG / "sonde_newsites_list.json"
PARAMS = {"00300": "do", "00095": "sc", "99133": "no3"}
STATS = {"00001", "00002", "00003", "00011"}
WY0, WY1 = pd.Timestamp("2007-10-01"), pd.Timestamp("2025-09-30")
TYPES = {"ST": "stream", "ST-TS": "stream", "ST-CA": "stream", "ST-DCH": "stream", "LK": "lake", "ES": "estuary",
         "WE": "wetland"}


# Files skipped when collecting site IDs named by other workstreams: ONLY the final EW/M1 split (handled separately)
# and this list itself. Every other file, including the DEV smoke-test split, is scanned (fix 2026-11-09: the first
# build skipped every ``sonde_split*`` file and so missed 113 DEV-split sites; disclosed in ew_v3_protocol.md §2a).
SKIP_FILES = ("sonde_split.json", "sonde_newsites_list.json", "sonde_newsites_overlap.json")
ID_PATTERNS = (r"(?<!\d)(\d{8,15})(?!\d)", r"USGS-(\d{7,15})")


def other_preregistration_ids(prereg_dir) -> tuple[set, dict]:
    """IDs named in every ``*.json`` of ``prereg_dir`` except :data:`SKIP_FILES` -> (ids, {id: [files]})."""
    used, used_by = set(), {}
    for f in sorted(glob.glob(str(prereg_dir / "*.json"))):
        name = f.split("/")[-1]
        if name in SKIP_FILES:
            continue
        txt = open(f).read()
        ids = set().union(*(re.findall(p, txt) for p in ID_PATTERNS))
        used |= ids
        for i in ids:
            used_by.setdefault(i, []).append(name)
    return used, used_by


def _get_all(sess, headers, coll, q):
    url, out, params = f"{USGS_OGC_BASE}/collections/{coll}/items", [], {"f": "json", **q}
    while url:
        for k in range(6):
            r = sess.get(url, params=params, headers=headers, timeout=300)
            if r.status_code == 200:
                break
            time.sleep(int(r.headers.get("Retry-After", "30") or 30) if r.status_code == 429 else 10 * (k + 1))
        else:
            raise RuntimeError(f"{coll}: HTTP {r.status_code}")
        d = r.json()
        out += d.get("features") or []
        url = next((ln["href"] for ln in d.get("links", []) if ln.get("rel") == "next"), None)
        if url and not url.startswith(USGS_OGC_BASE):
            raise RuntimeError("foreign next link")
        params = None
        time.sleep(0.5)
    return out


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists (frozen)")
    key = read_api_key()
    headers = {"User-Agent": USER_AGENT, **({"X-Api-Key": key} if key else {})}
    sess = requests.Session()
    rows = []
    for pc in PARAMS:
        feats = _get_all(sess, headers, "time-series-metadata",
                         {"skipGeometry": "true", "limit": "10000", "parameter_code": pc,
                          "properties": "monitoring_location_id,parameter_code,statistic_id,begin,end"})
        rows += [f["properties"] for f in feats]
    ts = pd.DataFrame(rows)
    ts = ts[ts["statistic_id"].isin(STATS)].copy()
    ts["site"] = ts["monitoring_location_id"].str.replace("USGS-", "", regex=False)
    ts["b"] = pd.to_datetime(ts["begin"], utc=True, errors="coerce").dt.tz_localize(None)
    ts["e"] = pd.to_datetime(ts["end"], utc=True, errors="coerce").dt.tz_localize(None)
    ts["yrs"] = ((ts["e"].clip(upper=WY1) - ts["b"].clip(lower=WY0)).dt.days / 365.25).clip(lower=0)
    per = ts.groupby(["site", "parameter_code"]).agg(yrs=("yrs", "max"), b=("b", "min"), e=("e", "max")).reset_index()
    per = per[per["yrs"] >= 1.0]
    split = json.loads((PREREG / "sonde_split.json").read_text())["site_role"]
    used, used_by = other_preregistration_ids(PREREG)
    cand = sorted(set(per["site"]))
    excl = {"in_sonde_split": sorted(s for s in cand if s in split)}
    cand = [s for s in cand if s not in split]
    excl["in_other_preregistration_files"] = sorted(s for s in cand if s in used)
    cand = [s for s in cand if s not in used]
    meta = []
    for i in range(0, len(cand), 50):
        ids = ",".join(f"USGS-{s}" for s in cand[i:i + 50])
        meta += _get_all(sess, headers, "monitoring-locations", {"limit": "100", "id": ids})
    md = {}
    for f in meta:
        p, g = f.get("properties") or {}, f.get("geometry") or {}
        c = (g.get("coordinates") or [None, None])
        md[str(p.get("monitoring_location_number"))] = {
            "site_type": p.get("site_type_code"), "lat": c[1], "lon": c[0], "huc": p.get("hydrologic_unit_code"),
            "state": p.get("state_name"), "tz": p.get("time_zone_abbreviation"),
            "drainage_area_mi2": p.get("drainage_area")}
    sites, excl["type_excluded_or_unknown"] = [], []
    for s in cand:
        m = md.get(s, {})
        if m.get("site_type") not in TYPES:
            excl["type_excluded_or_unknown"].append({"site": s, "site_type": m.get("site_type")})
            continue
        g = per[per["site"] == s]
        sites.append({"site_no": s, **m, "class": TYPES[m["site_type"]],
                      "variables": {PARAMS[r.parameter_code]: {"years_wy2008_25": round(float(r.yrs), 2),
                                                               "begin": str(r.b.date()), "end": str(r.e.date())}
                                    for r in g.itertuples()}})
    rec = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "rule": __doc__.split("Rule")[1].split("No data value")[0].strip(),
           "source": f"{USGS_OGC_BASE} time-series-metadata + monitoring-locations (metadata only)",
           "test_years": ["2007-10-01", "2025-09-30"],
           "n_sites": len(sites),
           "n_by_class": pd.Series([x["class"] for x in sites]).value_counts().to_dict(),
           "n_by_variable": {v: sum(v in x["variables"] for x in sites) for v in PARAMS.values()},
           "exclusions": {k: (len(v), v) for k, v in excl.items()},
           "other_file_overlap_detail": {s: used_by[s] for s in excl["in_other_preregistration_files"]},
           "sites": sites}
    OUT.write_text(json.dumps(rec, indent=1, default=str))
    OUT.chmod(0o444)
    print(json.dumps({"file": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      "n_sites": rec["n_sites"], "by_class": rec["n_by_class"], "by_var": rec["n_by_variable"],
                      "excluded": {k: v[0] for k, v in rec["exclusions"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
