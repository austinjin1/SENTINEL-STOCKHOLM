#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Event-layer label feasibility [S3]: external event catalogues matched to model sites (plan §3.3).

Protected rows are dropped from ids / dates before any label column is read:
* fish kills: KillAlert v2 test events (2018–2026) by ``event_id``; dates outside WY2008–2023-09-30;
* NRC reports: drills and dates outside WY2008–2023-09-30;
* WA microcystin: ToxinWatch test lakes and every year ≥ 2022;
* HAB reports: Orbit test role / test years (frozen Orbit v3 rule), buffer-excluded points, dates after
  2023-09-30.
Matching rules (fixed in the plan): fish kill / NRC → train/val-role stream sonde site within R km whose daily
record spans the event date (``events.match_external``; cluster = nearest matched site's basin group);
HAB report → HydroLAKES lake within 500 m, Orbit v3 role/time block (``orbit.split.assign`` with a
precomputed assignment, so Orbit's cache is never written).

    python -B scripts/eval/events_feasibility_external.py [--register]
"""

from __future__ import annotations

from sentinel.data.reserved import assert_no_reserved  # noqa: E402

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events_feasibility_common as C

from sentinel.core import paths
from sentinel.eval import events as EV

PRE = paths.REPO_ROOT / "docs" / "preregistration"
DATA = paths.PROCESSED
LO = pd.Timestamp("2007-10-01")
TRAIN_END = pd.Timestamp("2021-09-30")
TABLE, ROWS = [], []


def add(catalogue, label, target_model, radius, cell, n_elig, n_match, clusters: dict, note=""):
    r = {"catalogue": catalogue, "label": label, "target_model": target_model, "radius_km": radius, "cell": cell,
         "n_events_eligible": int(n_elig), "n_matched": int(n_match), **{f"n_{k}": int(v) for k, v in clusters.items()},
         "note": note}
    TABLE.append(r)
    ROWS.append({"model": f"external:{catalogue}", "target": f"{label}->{target_model}",
                 "split": f"{cell}/r{radius:g}km", "metric": "n_matched", "value": n_match, "n": n_elig,
                 "extra": {k: v for k, v in r.items() if k not in ("catalogue",)}})


def sonde_sites() -> pd.DataFrame:
    sp = json.loads((PRE / "sonde_split.json").read_text())
    role, basin = pd.Series(sp["site_role"]), pd.Series(sp["basin_group"])
    reg = pd.read_parquet(DATA / "sites" / "sites.parquet")
    reg = reg[reg["network"] == "USGS"].drop_duplicates("site_id")
    reg.index = reg["site_id"].astype(str)
    idx = pd.read_parquet(DATA / "sonde_daily" / "index.parquet")         # metadata: record spans only
    idx = idx[idx["var"].isin(["do_mgl_min", "spcond_uscm_mean", "turb_fnu_mean", "temp_c_mean"])]
    span = idx.groupby("site_no").agg(first=("first", "min"), last=("last", "max"))
    keep = [s for s in role.index if role[s] in ("train", "val") and s in reg.index and s in span.index
            and reg.loc[s, "site_type"] in ("ST", "ST-TS", "ST-CA", "ST-DCH")]
    assert_no_reserved(keep, "events_feasibility_external")   # reserved new-site block
    C.assert_allowed(role[keep], {"train", "val"}, "sonde site roles")
    return pd.DataFrame({"site_id": keep, "lat": pd.to_numeric(reg.loc[keep, "lat"]).to_numpy(),
                         "lon": pd.to_numeric(reg.loc[keep, "lon"]).to_numpy(), "cluster": basin.reindex(keep).to_numpy(),
                         "first": pd.to_datetime(span.loc[keep, "first"]).dt.tz_localize(None).to_numpy(),
                         "last": pd.to_datetime(span.loc[keep, "last"]).dt.tz_localize(None).to_numpy()})


def nearest_match(events: pd.DataFrame, sites: pd.DataFrame, R: float) -> pd.DataFrame:
    m = EV.match_external(events, sites, EV.MatchRule(radius_km=R, require_record_span=True))
    if m.empty:
        return pd.DataFrame(columns=["event_id", "site_id", "distance_km", "cluster", "n_sites"])
    n = m.groupby("event_id").size().rename("n_sites")
    best = m.sort_values(["event_id", "distance_km"]).drop_duplicates("event_id")
    return best.merge(n, on="event_id")


def fishkills(S):
    ka = json.loads((PRE / "downstream_killalert_v2_split.json").read_text())["assignments"]["event_role"]
    for name, src, pre in (("IA", "labels_iowa_fishkill", ""), ("DE", "labels_delaware_fishkill", "DE-")):
        f = DATA / src / "events.parquet"
        ids = pq.read_table(f, columns=["event_id", "date", "latitude", "longitude"]).to_pandas()
        ids["key"] = ids["event_id"].astype(str).map(lambda s, pre=pre: s if (not pre or s.startswith(pre)) else pre + s)
        ids["role"] = ids["key"].map(ka)
        ids["date"] = pd.to_datetime(ids["date"]).dt.tz_localize(None)
        keep = (ids["role"].isin(["train", "val"]) & (ids["date"] >= LO) & (ids["date"] <= C.FUTURE_CUTOFF)).to_numpy()
        ids = ids[keep].reset_index(drop=True)
        C.assert_allowed(ids["role"], {"train", "val"}, f"KillAlert role ({name})")
        lab = C.read_labels(f, ["is_env_wq", "is_anthropogenic"], keep)
        ev = pd.DataFrame({"event_id": ids["key"], "lat": ids["latitude"], "lon": ids["longitude"], "date": ids["date"]})
        for R in (5.0, 10.0, 25.0):
            m = nearest_match(ev, S, R).set_index("event_id")
            for lbl, sel in (("all_cause", np.ones(len(ev), bool)),
                             ("is_env_wq", lab["is_env_wq"].fillna(False).astype(bool).to_numpy()),
                             ("is_anthropogenic", lab["is_anthropogenic"].fillna(False).astype(bool).to_numpy())):
                for blk, bm in (("sonde_trainblock", ev["date"] <= TRAIN_END), ("sonde_valblock", ev["date"] > TRAIN_END)):
                    e = ev[sel & bm.to_numpy()]
                    mm = m.reindex(e["event_id"]).dropna(subset=["site_id"])
                    add(f"fishkill_{name}", lbl, "M1_M6_sonde_sites", R, blk, len(e), len(mm),
                        {"basins": mm["cluster"].nunique(), "sites": mm["n_sites"].sum() if len(mm) else 0},
                        "KillAlert train/val events only")


def nrc(S):
    f = DATA / "nrc_incidents" / "reports.parquet"
    ids = pq.read_table(f, columns=["incident_datetime_local", "date_time_received_local", "lat_decoded", "lon_decoded",
                                    "is_drill", "has_coordinates"]).to_pandas()
    date = pd.to_datetime(ids["incident_datetime_local"], errors="coerce").fillna(
        pd.to_datetime(ids["date_time_received_local"], errors="coerce")).dt.normalize()
    keep = ((date >= LO) & (date <= C.FUTURE_CUTOFF) & ~ids["is_drill"].astype("boolean").fillna(False)
            & ids["has_coordinates"].astype("boolean").fillna(False)
            & np.isfinite(ids["lat_decoded"].astype(float)) & np.isfinite(ids["lon_decoded"].astype(float))).to_numpy()
    C.assert_dates_within(date[keep], lo=LO, what="NRC dates")
    lab = C.read_labels(f, ["any_material_reached_water", "any_cas_with_structure"], keep)
    ev = pd.DataFrame({"event_id": np.arange(int(keep.sum())), "lat": ids["lat_decoded"][keep].astype(float).to_numpy(),
                       "lon": ids["lon_decoded"][keep].astype(float).to_numpy(), "date": date[keep].to_numpy()})
    water = lab["any_material_reached_water"].astype("boolean").fillna(False).to_numpy(bool)
    struct = water & lab["any_cas_with_structure"].astype("boolean").fillna(False).to_numpy(bool)
    for R in (5.0, 10.0):
        m = nearest_match(ev[water], S, R).set_index("event_id")
        for lbl, sel in (("reached_water", water), ("reached_water_structure_M4", struct)):
            for blk, bm in (("sonde_trainblock", ev["date"] <= TRAIN_END), ("sonde_valblock", ev["date"] > TRAIN_END)):
                e = ev[sel & bm.to_numpy()]
                mm = m.reindex(e["event_id"]).dropna(subset=["site_id"])
                add("NRC", lbl, "M1_sonde_sites", R, blk, len(e), len(mm), {"basins": mm["cluster"].nunique()},
                    "not drills; coordinates; WY2008-2023")


def habs():
    import geopandas as gpd

    from sentinel.data.sources import hydrolakes as H
    from sentinel.models.orbit import split as OS

    lakes = gpd.read_parquet(DATA / "hydrolakes" / "hydrolakes_na_eu.parquet",
                             columns=["Hylak_id", "Lake_name", "Lake_area", "Pour_long", "Pour_lat", "region", "geometry"])
    lakes = lakes[lakes["region"] == "north_america"]
    orbit_wb = set(pq.read_table(paths.RUNS_ROOT / "orbit" / "cache" / "dev_v4" / "table.parquet",
                                 columns=["waterbody"]).to_pandas()["waterbody"])

    def assign(lon, lat, dates):
        a = H.assign_points(lon, lat, lakes, max_dist_m=OS.LAKE_MAX_DIST_M)
        b = H.assign_points(lon, lat, lakes, max_dist_m=OS.BUFFER_M)
        a["near_hylak_id"] = b["hylak_id"].to_numpy()
        a["near_dist_m"] = b["lake_dist_m"].to_numpy()
        a["near_pour_lon"] = b["lake_pour_lon"].to_numpy()
        a["near_pour_lat"] = b["lake_pour_lat"].to_numpy()
        return OS.assign(lon, lat, dates, lakes=a.reset_index(drop=True), log=lambda *x: None)

    def run(name, f, lat, lon, date_series, base_keep, label_col, label_fn=None):
        ok = (base_keep & np.isfinite(lat) & np.isfinite(lon) & date_series.notna()).to_numpy()
        a = assign(lon[ok], lat[ok], date_series[ok])
        sub = (a["in_lake"] & ~a["protected"] & ~a["buffer_excluded"] & a["tblock"].isin(["train", "val"])).to_numpy()
        sub &= (date_series[ok] <= C.FUTURE_CUTOFF).to_numpy()
        keep = np.zeros(ok.size, bool)
        keep[np.flatnonzero(ok)[sub]] = True
        a = a[sub].reset_index(drop=True)
        d = date_series[keep].reset_index(drop=True)
        C.assert_allowed(a["role"], {"train", "val"}, f"{name} Orbit role")
        C.assert_allowed(a["tblock"], {"train", "val"}, f"{name} Orbit time block")
        C.assert_dates_within(d, what=f"{name} dates")
        lab = C.read_labels(f, [label_col], keep)[label_col]
        pos = (label_fn(lab) if label_fn else lab.astype("boolean").fillna(False)).to_numpy(bool)
        for era, em in (("all_years", np.ones(len(a), bool)), ("s2_era_2016on", (d >= pd.Timestamp("2016-01-01")).to_numpy())):
            for r_ in ("train", "val"):
                for tb in ("train", "val"):
                    m = (a["role"] == r_).to_numpy() & (a["tblock"] == tb).to_numpy() & em
                    x = a[m & pos]
                    add(name, f"positive_{era}", "M2_M7_orbit_lakes", 0.5, f"{r_}role_x_{tb}years", m.sum(), len(x),
                        {"lakes": x["waterbody"].nunique(), "blocks": x["block"].nunique(),
                         "lakes_in_orbit_dev": len(set(x["waterbody"]) & orbit_wb)}, "in-lake, Orbit v3 roles")

    f = DATA / "labels_ny_habs" / "lake_years.parquet"
    ny = pq.read_table(f, columns=["latitude", "longitude", "year", "first_report_date"]).to_pandas()
    nd = pd.to_datetime(ny["first_report_date"], errors="coerce").fillna(
        pd.to_datetime(ny["year"].astype("Int64").astype(str) + "-07-01", errors="coerce"))
    run("NY_HABs_lake_years", f, ny["latitude"].astype(float), ny["longitude"].astype(float), nd,
        pd.Series(True, index=ny.index), "label_confirmed")
    f = DATA / "labels_ca_fhab" / "bloom_reports.parquet"
    ca = pq.read_table(f, columns=["latitude", "longitude", "date"]).to_pandas()
    run("CA_FHAB_bloom_reports", f, ca["latitude"].astype(float), ca["longitude"].astype(float),
        pd.to_datetime(ca["date"], errors="coerce").dt.tz_localize(None), pd.Series(True, index=ca.index),
        "label_tier_ge_caution")
    tw = json.loads((PRE / "downstream_toxinwatch_split_v2.json").read_text())["assignments"]
    f = DATA / "labels_wa_toxic_algae" / "wa_toxin_siteday.parquet"
    wa = pq.read_table(f, columns=["site_id", "date", "toxin", "year"]).to_pandas()
    wa["site_id"] = wa["site_id"].astype(str)
    wa["tw_role"] = wa["site_id"].map(tw["site_cluster"]).map(tw["lake_role"])
    base = (wa["toxin"].astype(str).str.lower().str.contains("microcystin") & wa["tw_role"].isin(["train", "val"])
            & (wa["year"] < 2022))
    C.assert_allowed(wa.loc[base, "tw_role"], {"train", "val"}, "ToxinWatch lake role")
    ws = pq.read_table(DATA / "labels_wa_toxic_algae" / "wa_sites.parquet", columns=["site_id", "latitude", "longitude"]).to_pandas()
    ws["site_id"] = ws["site_id"].astype(str)
    co = wa[["site_id"]].merge(ws, on="site_id", how="left")
    run("WA_microcystin_sitedays", f, co["latitude"].astype(float), co["longitude"].astype(float),
        pd.to_datetime(wa["date"]).dt.tz_localize(None), base, "exceed_epa_8ugL_any")


def main() -> int:
    a = C.parser(__doc__).parse_args()
    out = C.out_dir(a, "external")
    S = sonde_sites()
    print(f"[sites] train/val-role stream sites with a daily record: {len(S)}", flush=True)
    fishkills(S)
    nrc(S)
    habs()
    res = pd.DataFrame(TABLE)
    f = C.write(res, out, "external_match_counts", {"events_version": EV.__version__, "created": C.stamp(),
                                                     "n_sonde_sites": len(S)})
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    print(res.drop(columns=["note"]).to_string(index=False))
    if a.register:
        n = C.register(ROWS, script=__file__, artifacts=[f],
                       data_manifests=["labels_iowa_fishkill", "labels_delaware_fishkill", "nrc_incidents",
                                       "labels_ny_habs", "labels_ca_fhab", "labels_wa_toxic_algae", "hydrolakes"])
        print(f"registered {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
