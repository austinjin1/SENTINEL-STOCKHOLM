#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Event-layer label feasibility [S4]: can NRC chemical-release reports serve as E8 outcome labels for M4?
Counts KillAlert train/val fish kills (2000–2017; KillAlert test events 2018–2026 are dropped by id before
any cause column is read) preceded by an NRC report 0–7 days earlier, (a) within R km (coordinates) and
(b) in the same state + county (fallback; NRC coordinates are sparse inland). Plan §3.3.

    python -B scripts/eval/events_feasibility_nrc_fishkill.py [--register]
"""

from __future__ import annotations

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
END = pd.Timestamp("2017-12-31")


def norm_county(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"\s+COUNTY$", "", regex=True).str.replace(r"[^A-Z]", "", regex=True)


def _within(kill_date: pd.Timestamp, report_dates, lo: int = 0, hi: int = 7) -> bool:
    """True if any report date lies lo..hi days before the kill date."""
    if report_dates is None or not len(report_dates):
        return False
    lag = (kill_date.to_datetime64() - report_dates).astype("timedelta64[D]").astype(int)
    return bool(((lag >= lo) & (lag <= hi)).any())


def main() -> int:
    a = C.parser(__doc__).parse_args()
    out = C.out_dir(a, "nrc_fishkill")
    ka = json.loads((PRE / "downstream_killalert_v2_split.json").read_text())["assignments"]["event_role"]
    kills = []
    for src, st, pre in (("labels_iowa_fishkill", "IA", ""), ("labels_delaware_fishkill", "DE", "DE-")):
        f = DATA / src / "events.parquet"
        ids = pq.read_table(f, columns=["event_id", "date", "latitude", "longitude", "county"]).to_pandas()
        ids["key"] = ids["event_id"].astype(str).map(lambda s, pre=pre: s if (not pre or s.startswith(pre)) else pre + s)
        ids["role"] = ids["key"].map(ka)
        keep = ids["role"].isin(["train", "val"]).to_numpy()
        ids = ids[keep].reset_index(drop=True)
        C.assert_allowed(ids["role"], {"train", "val"}, f"KillAlert role ({st})")
        lab = C.read_labels(f, ["is_anthropogenic", "is_env_wq"], keep)
        kills.append(pd.concat([ids.assign(state=st), lab], axis=1))
    k = pd.concat(kills, ignore_index=True)
    k["date"] = pd.to_datetime(k["date"]).dt.tz_localize(None).dt.normalize()
    C.assert_dates_within(k["date"], hi=END, what="fish-kill dates")
    k = k[np.isfinite(k["latitude"].astype(float)) & np.isfinite(k["longitude"].astype(float))].reset_index(drop=True)
    nr = pq.read_table(DATA / "nrc_incidents" / "reports.parquet",
                       columns=["incident_datetime_local", "date_time_received_local", "lat_decoded", "lon_decoded",
                                "is_drill", "any_material_reached_water", "location_state", "location_county"]).to_pandas()
    nr["date"] = pd.to_datetime(nr["incident_datetime_local"], errors="coerce").fillna(
        pd.to_datetime(nr["date_time_received_local"], errors="coerce")).dt.normalize()
    nr = nr[(nr["date"] <= END) & ~nr["is_drill"].astype("boolean").fillna(False)].reset_index(drop=True)
    nr["water"] = nr["any_material_reached_water"].astype("boolean").fillna(False).astype(bool)
    rows, table = [], []
    # (a) coordinates: NRC reached-water report within R km, kill 0..7 d after it
    geo = nr[nr["water"] & np.isfinite(nr["lat_decoded"].astype(float)) & np.isfinite(nr["lon_decoded"].astype(float))]
    sites = pd.DataFrame({"site_id": np.arange(len(geo)), "lat": geo["lat_decoded"].astype(float).to_numpy(),
                          "lon": geo["lon_decoded"].astype(float).to_numpy(), "date": geo["date"].to_numpy()})
    ev = pd.DataFrame({"event_id": np.arange(len(k)), "lat": k["latitude"].astype(float), "lon": k["longitude"].astype(float),
                       "date": k["date"]})
    for R in (10.0, 25.0):
        m = EV.match_external(ev, sites, EV.MatchRule(radius_km=R, window_days=(0, 7)))
        hit = np.isin(ev["event_id"], m["event_id"].unique())
        for lbl in ("all", "is_anthropogenic", "is_env_wq"):
            sel = np.ones(len(k), bool) if lbl == "all" else k[lbl].astype("boolean").fillna(False).to_numpy(bool)
            table.append({"rule": f"coords_{R:g}km_0-7d", "label": lbl, "n_kills": int(sel.sum()),
                          "n_with_nrc_report": int((hit & sel).sum())})
    # (b) same county, kill 0..7 d after any / reached-water NRC report
    nr["ckey"] = nr["location_state"].astype(str) + ":" + norm_county(nr["location_county"])
    k["ckey"] = k["state"] + ":" + norm_county(k["county"])
    for which, sub in (("any_report", nr), ("reached_water", nr[nr["water"]])):
        g = {c: v["date"].to_numpy() for c, v in sub[sub["location_state"].isin(["IA", "DE"])].groupby("ckey")}
        hit = np.array([_within(d, g.get(c)) for c, d in zip(k["ckey"], k["date"])])
        for lbl in ("all", "is_anthropogenic", "is_env_wq"):
            sel = np.ones(len(k), bool) if lbl == "all" else k[lbl].astype("boolean").fillna(False).to_numpy(bool)
            table.append({"rule": f"same_county_0-7d_{which}", "label": lbl, "n_kills": int(sel.sum()),
                          "n_with_nrc_report": int((hit & sel).sum())})
    res = pd.DataFrame(table)
    f = C.write(res, out, "nrc_fishkill_counts", {"created": C.stamp(), "n_kills": len(k),
                                                   "n_nrc_reached_water_with_coords": len(geo)})
    print(res.to_string(index=False))
    for r in table:
        rows.append({"model": "M4_chemtox", "target": f"E8_outcome:{r['label']}", "split": f"killalert_trainval/{r['rule']}",
                     "metric": "n_kills_with_nrc_report", "value": r["n_with_nrc_report"], "n": r["n_kills"], "extra": r})
    if a.register:
        n = C.register(rows, script=__file__, artifacts=[f],
                       data_manifests=["labels_iowa_fishkill", "labels_delaware_fishkill", "nrc_incidents"])
        print(f"registered {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
