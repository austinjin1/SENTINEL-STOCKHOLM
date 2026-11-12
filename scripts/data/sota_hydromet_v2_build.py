#!/usr/bin/env python
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Label and input tables of the M6 architecture-v2 round (``docs/preregistration/hydromet_arch_v2_amendment.md`` §2–§3).

Steps (``--steps``): ``grab`` (Tier 1 instantaneous grab rows + L-logger days, train block, fit and inner stations
only), ``vaa`` (3 NHDPlus VAA statics by COMID), ``hourly`` (dense local-solar-hour sonde table, train-role sites ×
train block). Outputs: ``$SENTINEL_DATA/processed/hydromet_sota_v2/``. Reports hold counts only; no value of a
protected row is read (lock-at-extraction, §2.5). Refuses to start with < 60 GB free on ``/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.data.reserved import exclude_reserved as shared_exclude_reserved  # noqa: E402
from sentinel.hydromet_sota import sites as LS  # noqa: E402
from sentinel.hydromet_sota.gso_mt import grab as G  # noqa: E402
from sentinel.hydromet_sota.gso_mt import runner as VR  # noqa: E402
from sentinel.models.hydromet import data as D  # noqa: E402

NAME = "hydromet_sota_v2"
MIN_FREE_GB = 60.0
VAA_COLS = ("pathtimema", "streamorde", "lengthkm")
log = logging.getLogger(NAME)


def check_disk() -> float:
    free = shutil.disk_usage("/").free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(f"only {free:.0f} GB free on / (< {MIN_FREE_GB:.0f}); stop and report")
    return free


def _write_report(out: Path, key: str, body: dict) -> None:
    f = out / "build_report.json"
    rep = json.loads(f.read_text()) if f.is_file() else {}
    rep[key] = body
    f.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))


def step_grab(out: Path) -> None:
    st = G.tier1_stations()
    by_rule = st.groupby("rule").size().to_dict()
    samples = G.read_samples(st, log=log.info)
    inst, daily, counts = G.collapse(samples)
    for df in (inst, daily):
        df.insert(0, "station_key", "WQP:" + df["site_id"].astype(str))
        if len(df) and (df["date"].max() > pd.Timestamp(LS.TRAIN_END) or df["date"].min() < pd.Timestamp(LS.TRAIN_START)):
            raise RuntimeError("grab rows outside the train block")
    ok = st[st["rule"].isin(G.VALUE_RULES)]
    if not set(inst["station_key"]) <= set(ok["key"]):
        raise RuntimeError("a grab row belongs to a station whose values may not be read")
    rule_of = ok.set_index("key")["rule"]
    n = inst.groupby(["station_key", "var"]).size().unstack(fill_value=0)
    stn = ok[["key", "site_no", "rule", "basin_group", "comid", "lat", "lon", "provider_class"]].copy()
    stn["n_inst_T"] = n.get("T", pd.Series(dtype=int)).reindex(stn["key"]).fillna(0).astype(int).to_numpy()
    stn["n_inst_DO"] = n.get("DO", pd.Series(dtype=int)).reindex(stn["key"]).fillna(0).astype(int).to_numpy()
    for df, col in ((inst, "station_key"), (daily, "station_key"), (stn, "key")):      # nothing reserved is written
        VR.assert_pool_clean(df[col].unique(), where=f"M6 arch-v2 grab tables ({col})")
    inst.to_parquet(out / "grab_inst.parquet", index=False)
    daily.to_parquet(out / "grab_logger_daily.parquet", index=False)
    stn.to_parquet(out / "grab_stations.parquet", index=False)
    r = inst["station_key"].map(rule_of)
    hours = np.floor(inst.loc[(r == "fit") & inst["hour"].notna(), "hour"]).astype(int)
    _write_report(out, "grab", {
        "stations_by_rule_metadata_only": {k: int(v) for k, v in by_rule.items()},
        "collapse": counts,
        "inst_rows_by_rule_var": {f"{a}/{b}": int(c) for (a, b), c in inst.groupby([r, "var"]).size().items()},
        "stations_with_rows_by_rule": {k: int(v) for k, v in
                                       stn[(stn["n_inst_T"] + stn["n_inst_DO"]) > 0].groupby("rule").size().items()},
        "provider_class_fit_stations": {k: int(v) for k, v in stn[stn["rule"] == "fit"].groupby("provider_class").size().items()},
        "fit_rows_by_local_solar_hour": {int(h): int(c) for h, c in hours.value_counts().sort_index().items()},
        "row_rules": {"logger_min_hours": G.LOGGER_MIN_HOURS, "fixed_hours": list(G.FIXED_HOURS),
                      "same_visit_span_h": G.SAME_VISIT_SPAN_H, "max_depth_m": G.MAX_DEPTH_M},
        "block": [LS.TRAIN_START, LS.TRAIN_END]})
    log.info("grab: %s", counts)


def step_vaa(out: Path) -> None:
    _, _, final, elig, _, _ = LS.final_frame()
    tr = final[(final["role"] == "train") & final["site_no"].isin(set(elig))]
    pool = pd.read_parquet(G.LEVER_SITES)
    pool = pool[pool["rule"].isin(G.VALUE_RULES)]
    loc = pd.concat([pd.DataFrame({"key": tr["site_no"].astype(str), "comid": tr["comid"].astype("string")}),
                     pd.DataFrame({"key": np.where(pool["source"] == "usgs_dv", pool["site_no"], pool["key"]),
                                   "comid": pool["comid"].astype("string")})], ignore_index=True).drop_duplicates("key")
    VR.assert_pool_clean(loc["key"], where="M6 arch-v2 VAA table")
    vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet", columns=["comid", *VAA_COLS])
    vaa["comid"] = vaa["comid"].astype("int64").astype(str)
    vaa = vaa.drop_duplicates("comid").set_index("comid")
    v = vaa.reindex(loc["comid"].astype(str))[list(VAA_COLS)].astype("float64")
    v = v.where(v > -9990)
    v.index = loc["key"].to_numpy()
    v.index.name = "key"
    v.columns = [f"vaa_{c}" for c in VAA_COLS]
    v.reset_index().to_parquet(out / "vaa_static.parquet", index=False)
    _write_report(out, "vaa", {"locations": int(len(v)), "with_comid_match": int(v.notna().any(axis=1).sum()),
                               "finite_by_column": {c: int(v[c].notna().sum()) for c in v.columns}})
    log.info("vaa: %d locations, %d matched", len(v), int(v.notna().any(axis=1).sum()))


def step_hourly(out: Path) -> None:
    """Dense (site, day, local solar hour bin, {T, DO}) table of sonde hourly means: FINAL train-role eligible sites
    and lever ``fit`` USGS sites × train block. Local solar time = UTC bin centre + lon/15 h."""
    import pyarrow.parquet as pq

    sp, _, final, elig, _, _ = LS.final_frame()
    pool = pd.read_parquet(G.LEVER_SITES)
    extra = set(pool.loc[(pool["source"] == "usgs_dv") & (pool["rule"] == "fit"), "site_no"].astype(str))
    cand = sorted({s for s in elig if sp.site_role[s] == "train"} | extra)
    n_cand = len(cand)
    cand = shared_exclude_reserved(cand)               # candidate list: drop (count logged); final list: refuse
    log.info("hourly: %d of %d candidate sites dropped as reserved", n_cand - len(cand), n_cand)
    hdir = paths.PROCESSED / "sonde_hourly" / "sites"
    sites = [s for s in cand if (hdir / f"{s}.parquet").is_file()]
    VR.assert_pool_clean(sites, where="M6 arch-v2 hourly table")
    stt = D.site_table()
    dates = pd.date_range(D.DATA_START, LS.TRAIN_END, freq="D")
    lo, hi = pd.Timestamp(LS.TRAIN_START), pd.Timestamp(LS.TRAIN_END)
    lo_utc, hi_utc = lo.tz_localize("UTC"), (hi + pd.Timedelta(days=1)).tz_localize("UTC")
    arr = np.full((len(sites), len(dates), 24, 2), np.nan, dtype="float32")
    frac = np.zeros(len(sites), dtype="float32")
    n_rows = 0
    for i, s in enumerate(sites):
        lon = float(stt.loc[s, "lon"])
        have = set(pq.read_schema(hdir / f"{s}.parquet").names)
        # Date filter inside the parquet scan (review finding HOURLY-BUILD-READS-PROTECTED-DATES, 2026-11-10): no row
        # with a UTC time after the train block reaches Python. US longitudes are west, so local solar time is
        # behind UTC and this bound also keeps every local solar day inside the block (re-checked by ``keep``).
        df = pd.read_parquet(hdir / f"{s}.parquet", columns=[c for c in ("temp_c_mean", "do_mgl_mean") if c in have],
                             filters=[("time_utc", ">=", lo_utc), ("time_utc", "<", hi_utc)])
        if len(df) and (df.index.max() >= hi_utc or df.index.min() < lo_utc):
            raise RuntimeError("hourly rows outside the train block were returned by the filtered read")
        df = df.reindex(columns=["temp_c_mean", "do_mgl_mean"])
        t = df.index.tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=0.5) + pd.Timedelta(hours=lon / 15.0)
        day = t.floor("D")
        keep = np.asarray((day >= lo) & (day <= hi))
        hod = (t - day) / pd.Timedelta(hours=1)
        j = np.floor(np.asarray(hod, dtype="float64")).astype(int)[keep]
        ti = np.asarray((day - dates[0]).days)[keep]
        frac[i] = float(np.mod(0.5 + lon / 15.0, 1.0))
        for k, c in enumerate(("temp_c_mean", "do_mgl_mean")):
            arr[i, ti, j, k] = df[c].to_numpy(dtype="float32")[keep]
        n_rows += int(np.isfinite(arr[i]).sum())
        if i % 100 == 0:
            log.info("hourly %d / %d", i, len(sites))
    np.savez_compressed(out / "hourly_train.npz", sites=np.array(sites), hour_frac=frac, values=arr,
                        date_start=str(dates[0].date()), date_end=str(dates[-1].date()))
    _write_report(out, "hourly", {"sites": len(sites), "reserved_candidates_dropped": n_cand - len(cand),
                                  "finite_cells": n_rows, "read_filter_utc": [str(lo_utc), str(hi_utc)],
                                  "site_days_T": int(np.isfinite(arr[..., 0]).any(-1).sum()),
                                  "site_days_DO": int(np.isfinite(arr[..., 1]).any(-1).sum())})
    log.info("hourly: %d sites, %d finite cells", len(sites), n_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default="grab,vaa,hourly")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    free = check_disk()
    out = paths.processed(NAME)
    log.info("free on /: %.0f GB; output %s", free, out)
    for s in a.steps.split(","):
        {"grab": step_grab, "vaa": step_vaa, "hourly": step_hourly}[s.strip()](out)
        check_disk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
