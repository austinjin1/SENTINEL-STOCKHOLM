# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Track P2 (lite_arch_v2_amendment.md §8): rivers and sondes, FNU turbidity at unseen water bodies.

This module builds the P2 **pool** and the **lock list** from identifiers, coordinates, site types and dates only.
No value column of any source is loaded here (:data:`WQP_META_COLS` / :data:`SONDE_INDEX_COLS` are the only columns
read), so the test block is fixed before any label value is read (LEAD_DECISIONS 2026-11-10, reserved blocks).

Pool exclusions (§8.2): the 729 reserved ``sonde_newsites``; M1 ``sonde_split`` test-role sites; M6 Track-N
test-role stations; sites inside a lake with the test role in M2's orbit split v3; every site of any Lite matchup
build. A P2 water body is a group of the frozen ``lite.data.water_body_groups`` (run over all Lite sites ∪ candidate
sites) that holds no Lite site. Roles come from a salted hash of the smallest site id of the group (§8.3)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data.reserved import NEWSITE_LIST_SHA256, _bare, assert_no_reserved, exclude_reserved, is_reserved

SALT = "SENTINEL-Lite-P2|2026-11-10"          # never changed (§8.3)
ROLE_CUTS = (("test_p2", 0.40), ("val_p2", 0.60), ("train_p2", 1.0))
DATE_MIN, DATE_MAX = "2010-01-01", "2023-09-30"   # M1's time block starts 2023-10-01
CLASSES = ("river_stream", "canal", "lake_reservoir", "estuary_coastal")
SONDE_VARS = ("turb_fnu_mean", "turb_fnu_median")  # mean; median only where the site has no mean series
SONDE_INDEX_COLS = ["site_no", "var", "first", "last"]
WQP_META_COLS = ["site_id", "state", "huc8", "lat", "lon", "site_type", "activity_start_date"]
LITE_BUILDS = ("matchups.parquet", "national/matchups.parquet", "national_pm7/matchups.parquet")
NWIS_CLASS = {"ST": "river_stream", "ST-TS": "estuary_coastal", "ST-CA": "canal", "ST-DCH": "canal",
              "LK": "lake_reservoir", "ES": "estuary_coastal", "OC-CO": "estuary_coastal", "OC": "estuary_coastal",
              "WE": "wetland"}
CONUS_BOX = (24.3, 49.5, -125.0, -66.5)          # lat min, lat max, lon min, lon max
NON_CONUS_STATES = ("AK", "HI", "PR", "VI", "GU", "AS", "MP")
LOCK_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "lite_p2_lock.json"
LOCK_SHA256 = "f3d4c63ebe503ac1774959bfcf943b45bc140883fe5b30933ee03f0cacca52be"   # written 2026-11-10 17:57 UTC; registry row 03323db99ad9
WB_PARAMS = {"radius_km": 2.0, "name_km": 40.0}  # national-2 parameters (lite.national.PARAMS)


def role_u(key: str) -> float:
    return int(hashlib.sha256((SALT + "|" + str(key)).encode()).hexdigest()[:12], 16) / 16 ** 12


def role_of(key: str) -> str:
    u = role_u(key)
    for role, cut in ROLE_CUTS:
        if u < cut:
            return role
    return ROLE_CUTS[-1][0]


def thin_order(site_id: str, date: str) -> str:
    """Sort key used when site-dates must be thinned for the disk cap (§8.5); the same in every role."""
    return hashlib.sha256((SALT + "|" + str(site_id) + "|" + str(date)).encode()).hexdigest()


# --------------------------------------------------------------------------- pool (metadata only)
def lite_sites() -> pd.DataFrame:
    """Every site of any Lite matchup build (4-state, national, national ±7 d): id, coordinates, name, class, HUC8."""
    cols = ["site_id", "lat", "lon", "name", "site_class", "huc8"]
    root = paths.processed("naip_matchups")
    out = [pd.read_parquet(root / f, columns=cols).drop_duplicates("site_id") for f in LITE_BUILDS]
    return pd.concat(out, ignore_index=True).drop_duplicates("site_id").reset_index(drop=True)


def _in_conus(lat, lon) -> np.ndarray:
    a, b, c, d = CONUS_BOX
    return (np.asarray(lat, float) >= a) & (np.asarray(lat, float) <= b) & (np.asarray(lon, float) >= c) \
        & (np.asarray(lon, float) <= d)


def sonde_candidates(counts: dict) -> pd.DataFrame:
    ix = pd.read_parquet(paths.processed("sonde_daily") / "index.parquet", columns=SONDE_INDEX_COLS)
    fn = ix[ix["var"].isin(SONDE_VARS)]
    fn = fn[(fn["first"].astype(str) <= DATE_MAX) & (fn["last"].astype(str) >= DATE_MIN)]
    sites = sorted(set(fn["site_no"].astype(str)))
    counts["sonde_sites_with_daily_fnu_in_window"] = len(sites)
    sites = exclude_reserved(sites)
    counts["sonde_after_reserved"] = len(sites)
    st = pd.read_parquet(paths.processed("sites") / "sites.parquet",
                         columns=["network", "site_id", "name", "lat", "lon", "site_type", "huc8", "in_conus"])
    st = st[st["network"] == "USGS"].drop_duplicates("site_id").set_index("site_id")
    have = [s for s in sites if s in st.index]
    counts["sonde_missing_from_sites_table"] = len(sites) - len(have)
    so = st.loc[have].reset_index()
    so["site_class"] = so["site_type"].map(NWIS_CLASS).fillna("other")
    so = so[so["in_conus"].fillna(False).astype(bool) & _in_conus(so["lat"], so["lon"])]
    so["site_id"] = "USGS-" + so["site_id"].astype(str)
    so["src"] = "sonde"
    return so[["site_id", "lat", "lon", "name", "site_class", "huc8", "src"]]


def wqp_candidates(counts: dict) -> pd.DataFrame:
    from ...data.matchups import site_class

    w = pd.read_parquet(paths.processed("wqp_discrete") / "variable=turbidity_fnu" / "data.parquet",
                        columns=WQP_META_COLS)
    d = pd.to_datetime(w["activity_start_date"], errors="coerce")
    w = w[(d >= DATE_MIN) & (d <= DATE_MAX)]
    ws = w.groupby("site_id", sort=True).agg(lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                                             site_type=("site_type", "first"), huc8=("huc8", "first")).reset_index()
    counts["wqp_fnu_sites_in_window"] = len(ws)
    ws["site_class"] = site_class(ws["site_type"]).to_numpy()
    ws = ws[_in_conus(ws["lat"], ws["lon"]) & ~ws["state"].fillna("").astype(str).str.upper().isin(NON_CONUS_STATES)]
    names = pd.read_parquet(paths.processed("wqp_discrete") / "_stations.parquet", columns=["site_id", "name"])
    ws["name"] = ws["site_id"].map(names.drop_duplicates("site_id").set_index("site_id")["name"])
    ws["src"] = "wqp"
    return ws[["site_id", "lat", "lon", "name", "site_class", "huc8", "src"]]


def _orbit_test_lake(lat, lon) -> np.ndarray:
    """True where the point lies in a HydroLAKES lake whose orbit-v3 block role is test (M2's own rule; no cache
    of M2 is written)."""
    from ..orbit import split as OS

    lk = OS.assign_lakes(np.asarray(lon, float), np.asarray(lat, float), use_cache=False, log=lambda *_: None)
    a = OS.assign(np.asarray(lon, float), np.asarray(lat, float), np.full(len(lk), "2020-01-01"), lakes=lk,
                  log=lambda *_: None)
    return (a["in_lake"].to_numpy(bool)) & (a["role"].to_numpy() == "test")


def candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(candidates, lite sites, counts). Candidates: one row per site after every §8.2 pool exclusion."""
    counts: dict = {}
    lite = lite_sites()
    counts["lite_sites_any_build"] = len(lite)
    so, ws = sonde_candidates(counts), wqp_candidates(counts)
    both = set(so["site_id"]) & set(ws["site_id"])
    cand = pd.concat([so, ws[~ws["site_id"].isin(both)]], ignore_index=True)
    cand.loc[cand["site_id"].isin(both), "src"] = "sonde+wqp"
    counts["candidates_conus_before_exclusions"] = len(cand)
    cand = cand[cand["site_class"].isin(CLASSES)]
    counts["after_class_filter"] = len(cand)
    cand = cand[~cand["site_id"].map(is_reserved)]
    counts["after_reserved_729"] = len(cand)
    m1 = {_bare(k): v for k, v in json.loads(
        (paths.REPO_ROOT / "docs/preregistration/sonde_split.json").read_text())["site_role"].items()}
    cand = cand[[not (s.startswith("USGS-") and m1.get(_bare(s)) == "test") for s in cand["site_id"]]]
    counts["after_m1_test_sites"] = len(cand)
    m6 = json.loads((paths.REPO_ROOT / "docs/preregistration/hydromet_track_n_stations.json").read_text())["role_of"]
    m6_test = {k.split(":", 1)[-1] for k, v in m6.items() if v == "test"}
    cand = cand[~cand["site_id"].isin(m6_test)]
    counts["after_m6_track_n_test"] = len(cand)
    cand = cand[~cand["site_id"].isin(set(lite["site_id"]))]
    counts["after_lite_build_sites"] = len(cand)
    cand = cand[~_orbit_test_lake(cand["lat"], cand["lon"])]
    counts["after_orbit_v3_test_lakes"] = len(cand)
    cand = cand.sort_values("site_id").reset_index(drop=True)
    assert_no_reserved(list(cand["site_id"]), "lite P2 pool")
    return cand, lite, counts


# --------------------------------------------------------------------------- lock
def build_lock() -> dict:
    """The lock record (not written). Uses site identifiers, coordinates, names, classes and HUC8 only."""
    from ..lite import data as D

    cand, lite, counts = candidate_pool()
    cols = ["site_id", "lat", "lon", "name", "site_class", "huc8"]
    allsites = pd.concat([lite[cols].assign(src="lite"), cand[cols + ["src"]]], ignore_index=True)
    allsites["name"] = allsites["name"].fillna("")
    hylak = D.hydrolakes_ids(allsites)
    wb = D.water_body_groups(allsites, WB_PARAMS["radius_km"], WB_PARAMS["name_km"], hylak)
    allsites["wb"] = allsites["site_id"].map(wb)
    seen = set(allsites.loc[allsites["src"] == "lite", "wb"])
    c = allsites[(allsites["src"] != "lite") & ~allsites["wb"].isin(seen)].copy()
    counts["candidate_sites_in_unseen_water_bodies"] = len(c)
    from ...eval.splits import nearest_distance_km

    d = nearest_distance_km(c["lat"], c["lon"], lite["lat"], lite["lon"])
    if not (np.asarray(d) > WB_PARAMS["radius_km"]).all():
        raise RuntimeError("a P2 site lies within the water-body radius of a Lite site")
    c["hylak_id"] = c["site_id"].map(hylak).astype("Int64")
    bodies = []
    for _, g in c.groupby("wb", sort=False):
        g = g.sort_values("site_id")
        key = str(g["site_id"].iloc[0])
        bodies.append({"key": key, "u": round(role_u(key), 9), "role": role_of(key),
                       "main_class": str(g["site_class"].mode().iat[0]),
                       "has_sonde": bool(g["src"].str.startswith("sonde").any()),
                       "sites": [{"site_id": str(r.site_id), "src": str(r.src), "site_class": str(r.site_class),
                                  "lat": round(float(r.lat), 6), "lon": round(float(r.lon), 6),
                                  "huc8": None if pd.isna(r.huc8) else str(r.huc8),
                                  "hylak_id": None if pd.isna(r.hylak_id) else int(r.hylak_id)}
                                 for r in g.itertuples()]})
    bodies.sort(key=lambda b: b["key"])
    by_role = {}
    for role, _ in ROLE_CUTS:
        bb = [b for b in bodies if b["role"] == role]
        by_role[role] = {"water_bodies": len(bb), "sites": sum(len(b["sites"]) for b in bb),
                         "water_bodies_with_sonde": sum(b["has_sonde"] for b in bb),
                         "by_main_class": pd.Series([b["main_class"] for b in bb]).value_counts().sort_index().to_dict()}
    counts["water_bodies"] = len(bodies)
    assert_no_reserved([s["site_id"] for b in bodies for s in b["sites"]], "lite P2 lock list")
    return {"track": "lite_p2", "amendment": "docs/preregistration/lite_arch_v2_amendment.md §8.2–8.3",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "salt": SALT,
            "rule": "u = int(sha256(salt + '|' + key).hexdigest()[:12], 16) / 16**12; key = smallest site id (string "
                    "order) of the water body; u < 0.40 test_p2; u < 0.60 val_p2; else train_p2",
            "role_cuts": [list(x) for x in ROLE_CUTS], "date_window": [DATE_MIN, DATE_MAX], "classes": list(CLASSES),
            "water_body_rule": {**WB_PARAMS, "flowing_huc8": True, "hydrolakes": True,
                                "over": "all Lite sites ∪ candidate sites; P2 water body = group with no Lite site"},
            "columns_read": {"sonde_daily/index.parquet": SONDE_INDEX_COLS,
                             "wqp_discrete/variable=turbidity_fnu/data.parquet": WQP_META_COLS,
                             "note": "no value column of any source was loaded to build this list"},
            "reserved_list_sha256": NEWSITE_LIST_SHA256, "pool_counts": counts, "role_counts": by_role,
            "water_bodies": bodies}


def write_lock(rec: dict, path: Path = LOCK_FILE) -> str:
    """Write the lock list once (exclusive create, read-only). Returns its SHA-256."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lock(path: Path = LOCK_FILE, expect_sha256: str | None = LOCK_SHA256) -> dict:
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expect_sha256 and sha != expect_sha256:
        raise PermissionError("lite_p2_lock.json differs from its pinned SHA-256")
    rec = json.loads(raw)
    for b in rec["water_bodies"]:
        if role_of(b["key"]) != b["role"] or min(s["site_id"] for s in b["sites"]) != b["key"]:
            raise PermissionError(f"lock list role or key of {b['key']} does not follow the salted rule")
    rec["_sha256"] = sha
    return rec


def site_roles(rec: dict) -> pd.DataFrame:
    """One row per P2 site: site_id, water_body (= key), role, src, site_class, lat, lon."""
    rows = [{**s, "water_body": "p2:" + b["key"], "role": b["role"]} for b in rec["water_bodies"] for s in b["sites"]]
    return pd.DataFrame(rows)
