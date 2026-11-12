#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Frozen series map for the live trial: which OGC ``time_series_id`` feeds each (site, parameter, statistic).

Several co-located sondes can report the same parameter. The training tier (legacy DV, ``sonde_daily``) used the
longest series as primary; with only weeks of live data ``splice_site`` would pick by window length instead.
Rule (no outcome data; pre-trial window only): fetch the OGC daily values for MAP_WINDOW (2026-10-01..2026-11-07,
WY2027, before the trial), and for every (site, parameter, statistic) with more than one OGC series choose the
series with the highest share of days agreeing (|Δ| ≤ 1e-4) with the 2026-11-08 snapshot; ties → more days.
Single-series keys are mapped trivially. Series appearing later fall back to the frozen ``splice_site`` rule (logged).

    python -B scripts/live/build_series_map.py --roles train,val --out dryrun     # dry run (agent)
    python -B scripts/live/build_series_map.py --roles train,val,test --out frozen  # lead, at freeze time
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time

import numpy as np
import pandas as pd
import requests

import live_common as LC
import live_data as LD
from fetch_usgs import fetch_batch
from sentinel.data.sonde_daily import clean_long
from sentinel.data.sources.usgs_legacy import ALL_PARAMS, STATS
from sentinel.platform.config import USER_AGENT
from sentinel.platform.live import read_api_key

MAP_WINDOW = ("2026-10-01", "2026-11-07")
FROZEN_MAP = LC.REPO / "configs" / "live" / "ew_v2_live_series_map.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="train,val")
    ap.add_argument("--out", choices=("dryrun", "frozen"), default="dryrun")
    a = ap.parse_args()
    roles = set(a.roles.split(","))
    if a.out == "frozen" and roles != {"train", "val", "test"}:
        raise SystemExit("the frozen map covers all roles")
    dest = FROZEN_MAP if a.out == "frozen" else LC.DRYRUN_ROOT / "series_map_trainval.json"
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    cfg, cfg_sha = LC.load_config()
    sites = sorted({r["site_no"] for v in cfg["sites"].values() for r in v if r["role"] in roles})
    key = read_api_key()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}
    if key:
        headers["X-Api-Key"] = key
    sess, recs = requests.Session(), []
    for i in range(0, len(sites), 20):
        recs += fetch_batch(sess, headers, sites[i:i + 20], MAP_WINDOW[0], MAP_WINDOW[1], print)
        time.sleep(1.0)
    long, _ = LD.ogc_long(recs)
    clean, _ = clean_long(long)
    clean["var"] = clean["parm_cd"].map(ALL_PARAMS) + "_" + clean["stat_cd"].map(STATS)
    snapdir = __import__("pathlib").Path(cfg["data_pins"]["sonde_daily_dir"])
    idx = pd.date_range(*MAP_WINDOW)
    smap, stats = {}, {"n_keys": 0, "n_multi": 0, "multi": []}
    for s, g in clean.groupby("site_no"):
        p = snapdir / "sites" / f"{s}.parquet"
        sn = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        sn.index = pd.to_datetime(sn.index)
        for (pc, st, v), gg in g.groupby(["parm_cd", "stat_cd", "var"]):
            k = f"{s}|{pc}|{st}"
            ts_ids = sorted(gg["ts_id"].unique())
            stats["n_keys"] += 1
            if len(ts_ids) == 1:
                smap[k] = ts_ids[0]
                continue
            stats["n_multi"] += 1
            ref = sn[v].reindex(idx) if v in sn else pd.Series(np.nan, index=idx)
            best = None
            for t in ts_ids:
                x = gg[gg["ts_id"] == t].set_index("date")["value"].reindex(idx)
                m = x.notna() & ref.notna()
                agree = float(((x[m] - ref[m]).abs() <= 1e-4).mean()) if m.any() else -1.0
                cand = (agree, int(x.notna().sum()), t)
                best = cand if best is None or cand[:2] > best[:2] else best
            smap[k] = best[2]
            stats["multi"].append({"key": k, "chosen": best[2], "agreement": best[0], "n_days": best[1],
                                   "n_series": len(ts_ids)})
    rec = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "window": MAP_WINDOW,
           "roles": sorted(roles), "config_sha256": cfg_sha, "rule": __doc__.split("Rule")[1].split("\n\n")[0].strip(),
           "map": smap, "stats": stats}
    dest.parent.mkdir(parents=True, exist_ok=True)
    LC.excl_write(dest, rec)
    print(json.dumps({"dest": str(dest), "sha256": LC.sha256_file(dest), "n_keys": stats["n_keys"],
                      "n_multi": stats["n_multi"],
                      "low_agreement": [m for m in stats["multi"] if m["agreement"] < 0.9]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
