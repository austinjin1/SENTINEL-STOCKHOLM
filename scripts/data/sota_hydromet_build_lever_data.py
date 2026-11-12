#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build ``processed/hydromet_sota``: added training locations for the M6 SOTA levers 1–2
(``docs/preregistration/hydromet_sota_protocol.md`` §3; rules in ``sentinel/hydromet_sota/sites.py``).

    python -B scripts/data/sota_hydromet_build_lever_data.py sites      # USGS + WQP pools, rules, WQP train-block obs
    python -B scripts/data/sota_hydromet_build_lever_data.py upstream   # basin-mean gridMET for the new USGS fit sites
    python -B scripts/data/sota_hydromet_build_lever_data.py manifest

* USGS pool (levers 1a and 2): every ``processed/sonde_daily`` site NOT in the FINAL split with an eligible stream
  site (site type ST/ST-TS/ST-CA/ST-DCH, gridMET forcing, StreamCat) and ≥ 365 train-block days of temperature/DO
  (lever 1a) or of discharge (lever 2). M1-reserved sites are removed first (``exclude_reserved``, logged).
* WQP pool (lever 1b): track-N **train**-role stations (frozen roles ``ffb41471…``) with ≥ 4 accepted grab-sample
  days of water temperature or DO (mg/L) in the train block WY2008–2021 (same filters as track N); only train-block
  dates are read from ``processed/wqp_discrete``.
* Only rows with ``rule == "fit"`` may ever be training rows; ``assert_no_reserved`` runs on them before writing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import logging
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.reserved import assert_no_reserved as shared_assert_no_reserved  # noqa: E402
from sentinel.data.reserved import exclude_reserved as shared_exclude_reserved  # noqa: E402
from sentinel.hydromet_sota import reserved as R  # noqa: E402
from sentinel.hydromet_sota import sites as LS  # noqa: E402

NAME = "hydromet_sota"
OUT = paths.processed(NAME)
INT = paths.interim(NAME)
WQP = paths.PROCESSED / "wqp_discrete"
ND = paths.PROCESSED / "hydromet_wqp_national"
WQP_VARS = {"temp": ("temp_c_mean", "deg C"), "do": ("do_mgl_mean", "mg/L")}
MAX_DEPTH_M = 2.0
NON_STATES = {"AK", "HI", "PR", "GU", "VI", "AS", "MP"}
log = logging.getLogger(NAME)


def _usgs_pool(final, owner, inner_groups, res, vaa, protected) -> pd.DataFrame:
    from sentinel.models.hydromet import data as D

    idx = pd.read_parquet(D.SONDE_INDEX, columns=["site_no", "var"])   # index metadata only (site, variable)
    cand = sorted(set(idx.loc[idx["var"].isin(list(LS.WQ_VARS) + [LS.Q_VAR]), "site_no"].astype(str)))
    split_sites = set(final["site_no"])
    new0 = [s for s in cand if s not in split_sites]
    new = shared_exclude_reserved(new0)                  # M1-reserved block: shared helper (pinned 729-site list)
    log.info(f"[reserved] excluded {len(new0) - len(new)} of {len(new0)} sites listed in {R.NEWSITES_FILE.name}")
    if new != R.exclude_reserved(new0, log=lambda *_: None):   # own guard must agree (same list, same pin)
        raise RuntimeError("shared and M6 reserved helpers disagree")
    st = D.site_table()
    st = st.reindex(new)
    elig = st["site_type"].isin(D.STREAM_TYPES) & (st["forcing_source"] == "gridmet") & st["streamcat_found"].fillna(False)
    info = {"candidates_not_in_split_after_reserved": len(new), "not_in_site_table": int(st["site_key"].isna().sum()),
            "not_stream": int((~st["site_type"].isin(D.STREAM_TYPES)).sum()),
            "no_gridmet": int((st["forcing_source"] != "gridmet").sum()),
            "no_streamcat": int((~st["streamcat_found"].fillna(False).astype(bool)).sum())}
    sites = sorted(st.index[elig.to_numpy()])
    info["eligible"] = len(sites)
    cnt = LS.train_block_days(sites, list(LS.WQ_VARS) + [LS.Q_VAR], log=log.info)
    wq_days = cnt[list(LS.WQ_VARS)].max(axis=1)
    reg = pd.read_parquet(D.SITES_FILE)
    reg = reg[reg["network"] == "USGS"].drop_duplicates("site_id").set_index("site_id")
    sc = pd.read_parquet(D.STREAMCAT_FILE, columns=["site_key", "comid"]).drop_duplicates("site_key").set_index("site_key")
    pool = pd.DataFrame({"key": st.loc[sites, "site_key"].to_numpy(), "site_no": sites,
                         "huc4": reg["huc8"].reindex(sites).astype(str).str[:4].to_numpy(),
                         "comid": sc["comid"].reindex(st.loc[sites, "site_key"]).astype("string").to_numpy()})
    pool["source"] = "usgs_dv"
    pool["wq_train_days"] = wq_days.reindex(sites).to_numpy()
    pool["q_train_days"] = cnt[LS.Q_VAR].reindex(sites).to_numpy()
    for v in LS.WQ_VARS:
        pool[f"days_{v}"] = cnt[v].reindex(sites).to_numpy()
    pool["lever1a"] = pool["wq_train_days"] >= LS.MIN_DV_DAYS
    pool["lever2_q"] = pool["q_train_days"] >= LS.MIN_Q_DAYS
    pool = pool[pool["lever1a"] | pool["lever2_q"]].reset_index(drop=True)
    info["with_365_train_days_wq_or_q"] = int(len(pool))
    conn = LS.flow_connected(pool["comid"].dropna().astype(str), protected, vaa)
    pool = LS.apply_rules(pool, owner=owner, inner_groups=inner_groups, connected=conn, res=res, final_sites=split_sites,
                          bad_comids=LS.reserved_comids())
    return pool, info


def _wqp_train_obs(station_ids: set[str]) -> pd.DataFrame:
    """Daily means of accepted grab samples (track-N filters) in the TRAIN block only."""
    import pyarrow.dataset as ds

    obs = []
    lo, hi = pd.Timestamp(LS.TRAIN_START).date(), pd.Timestamp(LS.TRAIN_END).date()
    for v, (target, unit) in WQP_VARS.items():
        d = ds.dataset(WQP / f"variable={v}" / "data.parquet")
        flt = ((ds.field("site_class") == "stream") & (ds.field("activity_start_date") >= lo)
               & (ds.field("activity_start_date") <= hi))
        t = d.to_table(columns=["site_id", "state", "activity_start_date", "value", "unit", "censored", "depth_m",
                                "activity_type", "status_class"], filter=flt).to_pandas()
        t = t[t["site_id"].isin(station_ids)]
        n0 = len(t)
        t = t[(t["unit"] == unit) & (t["status_class"] == "accepted") & ~t["censored"].fillna(False).astype(bool)
              & np.isfinite(t["value"]) & ~t["activity_type"].fillna("").str.contains("Time Series|Quality Control")
              & (t["depth_m"].isna() | (t["depth_m"] <= MAX_DEPTH_M)) & ~t["state"].isin(NON_STATES)]
        t["date"] = pd.to_datetime(t["activity_start_date"])
        if t["date"].max() > pd.Timestamp(LS.TRAIN_END) or t["date"].min() < pd.Timestamp(LS.TRAIN_START):
            raise RuntimeError("WQP obs outside the train block")
        day = t.groupby(["site_id", "date"], as_index=False).agg(value=("value", "mean"), n_samples=("value", "size"))
        day["target"] = target
        obs.append(day)
        log.info("WQP %s: %d rows at pool stations -> %d kept -> %d station-days", v, n0, len(t), len(day))
    return pd.concat(obs, ignore_index=True)


def _wqp_pool(final, owner, inner_groups, res, vaa, protected, trackn) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    st = pd.read_parquet(ND / "stations.parquet")
    tr = {k for k, r in trackn["role_of"].items() if r == "train"}
    st = st[st["station_key"].isin(tr)].copy()
    info = {"trackn_train_stations": int(len(st))}
    ids0 = st["site_id"].tolist()
    ids = shared_exclude_reserved(ids0)
    log.info(f"[reserved] excluded {len(ids0) - len(ids)} of {len(ids0)} sites listed in {R.NEWSITES_FILE.name}")
    if ids != R.exclude_reserved(ids0, log=lambda *_: None):
        raise RuntimeError("shared and M6 reserved helpers disagree")
    st = st[st["site_id"].isin(ids)]
    obs = _wqp_train_obs(set(st["site_id"]))
    n = obs.groupby(["site_id", "target"]).size().unstack(fill_value=0)
    st["days_temp_c_mean"] = n.get("temp_c_mean", pd.Series(dtype=int)).reindex(st["site_id"]).fillna(0).astype(int).to_numpy()
    st["days_do_mgl_mean"] = n.get("do_mgl_mean", pd.Series(dtype=int)).reindex(st["site_id"]).fillna(0).astype(int).to_numpy()
    st = st[(st["days_temp_c_mean"] >= LS.MIN_WEAK_DAYS) | (st["days_do_mgl_mean"] >= LS.MIN_WEAK_DAYS)]
    info["with_min_weak_days"] = int(len(st))
    pool = pd.DataFrame({"key": st["station_key"].to_numpy(), "site_no": st["site_id"].to_numpy(),
                         "huc4": st["huc8"].astype(str).str[:4].to_numpy(), "comid": st["comid"].astype("string").to_numpy(),
                         "days_temp_c_mean": st["days_temp_c_mean"].to_numpy(),
                         "days_do_mgl_mean": st["days_do_mgl_mean"].to_numpy()})
    pool["source"] = "wqp_grab"
    conn = LS.flow_connected(pool["comid"].dropna().astype(str), protected, vaa)
    pool = LS.apply_rules(pool, owner=owner, inner_groups=inner_groups, connected=conn, res=res,
                          bad_comids=LS.reserved_comids(final))
    # track N gives a station the smallest of its candidate groups (HUC4 owner + groups of flow-connected FINAL sites);
    # all candidates are train-role for a train station, but none of them may be an inner group either
    pool["trackn_group"] = pool["key"].map(trackn["basin_group_of"]).astype(str)
    fitm = pool["rule"] == "fit"
    other = fitm & (pool["trackn_group"] != pool["basin_group"].astype(str))
    info["fit_with_other_trackn_group"] = int(other.sum())
    pool.loc[other & pool["trackn_group"].isin(set(inner_groups)), "rule"] = "inner_group"
    fitm = pool["rule"] == "fit"
    keep_ids = set(pool.loc[fitm, "site_no"])
    obs = obs[obs["site_id"].isin(keep_ids)].copy()
    # per-target ≥ MIN_WEAK_DAYS (a station keeps only the targets with enough days)
    n = obs.groupby(["site_id", "target"])["date"].transform("size")
    obs = obs[n >= LS.MIN_WEAK_DAYS]
    obs["station_key"] = "WQP:" + obs["site_id"]
    return pool, obs, info


def step_sites() -> None:
    sp, sinfo, final, elig, inner_sites, inner_groups = LS.final_frame()
    if not LS.check_inner_matches_reference(inner_sites):
        raise RuntimeError("recomputed inner sites differ from the reference run ealstm_ws_v1_s0")
    log.info("FINAL split %s; %d eligible train/val sites; %d inner sites in %d groups (= reference run)",
             sinfo["sha256"][:12], len(elig), len(inner_sites), len(inner_groups))
    owner = LS.huc4_owner(final)
    res = R.reserved_usgs_sites()
    vaa = LS.load_vaa()
    trackn = LS.load_trackn_roles()
    protected = LS.protected_comids(final, trackn)
    usgs, uinfo = _usgs_pool(final, owner, inner_groups, res, vaa, protected)
    wqp, obs, winfo = _wqp_pool(final, owner, inner_groups, res, vaa, protected, trackn)
    pool = pd.concat([usgs, wqp], ignore_index=True, sort=False)
    fit = pool[pool["rule"] == "fit"]
    R.assert_no_reserved(fit["site_no"].tolist())
    R.assert_no_reserved([k.split(":", 1)[-1] for k in fit["key"]])
    shared_assert_no_reserved(fit["site_no"].tolist(), where="M6 SOTA lever pool")   # hygiene rule 1
    shared_assert_no_reserved(fit["key"].tolist(), where="M6 SOTA lever pool keys")
    if set(fit["comid"].astype(str)) & LS.reserved_comids():
        raise RuntimeError("fit location on a reserved-site COMID")
    if any(LS.contains_reserved_id(k, res) for k in fit["key"]):
        raise RuntimeError("reserved id inside a fit key")
    if set(fit["group_role"]) - {"train"} or set(fit["basin_group"]) & set(inner_groups):
        raise RuntimeError("fit location outside the train-role fit groups")
    OUT.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT / "lever_sites.parquet", compression="zstd", index=False)
    obs.to_parquet(OUT / "wqp_train_obs.parquet", compression="zstd", index=False)
    rep = {"built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "split_sha256": sinfo["sha256"], "trackn_roles_sha256": LS.TRACKN_SHA256,
           "reserved_list_sha256": R.NEWSITES_SHA256, "inner_groups": inner_groups, "n_inner_sites": len(inner_sites),
           "usgs_info": uinfo, "wqp_info": winfo,
           "rules_usgs": usgs["rule"].value_counts().to_dict(), "rules_wqp": wqp["rule"].value_counts().to_dict(),
           "fit_usgs_lever1a": int(((usgs["rule"] == "fit") & usgs["lever1a"]).sum()),
           "fit_usgs_lever2_q": int(((usgs["rule"] == "fit") & usgs["lever2_q"]).sum()),
           "fit_usgs_q_only": int(((usgs["rule"] == "fit") & usgs["lever2_q"] & ~usgs["lever1a"]).sum()),
           "fit_usgs_groups": int(usgs.loc[usgs["rule"] == "fit", "basin_group"].nunique()),
           "fit_wqp_stations": int((wqp["rule"] == "fit").sum()),
           "fit_wqp_groups": int(wqp.loc[wqp["rule"] == "fit", "basin_group"].nunique()),
           "wqp_obs_station_days": obs.groupby("target").size().to_dict(),
           "wqp_obs_stations": obs.groupby("target")["site_id"].nunique().to_dict()}
    (OUT / "lever_sites_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log.info("lever sites: %s", json.dumps(rep, default=str))


def _upstream_builder():
    spec = importlib.util.spec_from_file_location("_up", paths.REPO_ROOT / "scripts" / "data" / "build_hydromet_upstream_forcing.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    return B


def lever_usgs_sites() -> pd.DataFrame:
    """Site list (builder schema) of the new USGS fit sites (levers 1a and 2)."""
    from sentinel.models.hydromet import data as D

    pool = pd.read_parquet(OUT / "lever_sites.parquet")
    pool = pool[(pool["source"] == "usgs_dv") & (pool["rule"] == "fit")]
    R.assert_no_reserved(pool["site_no"].tolist())
    shared_assert_no_reserved(pool["site_no"].tolist(), where="M6 SOTA upstream-forcing site list")
    st = D.site_table().loc[pool["site_no"]]
    sc = pd.read_parquet(D.STREAMCAT_FILE, columns=["site_key", "comid", "comid_method", "wsareasqkm"])
    sc = sc.drop_duplicates("site_key").set_index("site_key")
    df = pd.DataFrame({"site_no": pool["site_no"].to_numpy(), "site_key": st["site_key"].to_numpy(),
                       "lat": st["lat"].to_numpy(), "lon": st["lon"].to_numpy()})
    df["comid"] = sc["comid"].reindex(df["site_key"]).astype(str).to_numpy()
    df["comid_method"] = sc["comid_method"].reindex(df["site_key"]).to_numpy()
    df["streamcat_wsareasqkm"] = sc["wsareasqkm"].reindex(df["site_key"]).to_numpy()
    return df.sort_values("site_key").reset_index(drop=True)


def step_upstream(steps) -> None:
    """Same builder, method and QC as ``processed/hydromet_upstream_forcing`` (sonde sites), applied to the new USGS
    fit sites; output ``upstream_forcing_sota.parquet`` in that directory (suffix ``_sota``)."""
    B = _upstream_builder()
    B.SET, B.SFX = "sota", "_sota"
    B.WORK = B.INT / "sota"
    B.WORK.mkdir(parents=True, exist_ok=True)
    B.site_list = lever_usgs_sites
    B.log = log
    if "basins_local" in steps:
        B.step_basins_local()
    if "aggregate" in steps:
        B.step_aggregate()
    if "combine" in steps:
        B.step_combine()


def step_manifest() -> None:
    files = [OUT / "lever_sites.parquet", OUT / "wqp_train_obs.parquet", OUT / "lever_sites_report.json"]
    rep = json.loads((OUT / "lever_sites_report.json").read_text())
    write_manifest(NAME, files=files,
                   sources=["processed/sonde_daily (manifest sonde_daily; train-block day counts only)",
                            "processed/wqp_discrete (WQP; train block WY2008-2021 only)",
                            "docs/preregistration/sonde_split.json (FINAL, fd4bb14f...)",
                            "docs/preregistration/hydromet_track_n_stations.json (ffb41471...)",
                            "docs/preregistration/sonde_newsites_list.json (6200a77c..., excluded)",
                            "raw/nhdplus_vaa (flow connectivity)"],
                   description="M6 SOTA levers 1-2: added training locations (new USGS DV/discharge sites in FINAL "
                               "train-role non-inner basin groups; WQP train-region grab-sample stations, train block "
                               "only) with the exclusion rule per location.",
                   license="public domain (USGS NWIS, WQP/EPA, NHDPlus V2)", synthetic=False, stats=rep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["sites", "upstream", "manifest"])
    ap.add_argument("--upstream-steps", default="basins_local,aggregate,combine")
    a = ap.parse_args()
    INT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(INT / "build.log")])
    if a.step == "sites":
        step_sites()
    elif a.step == "upstream":
        step_upstream(a.upstream_steps.split(","))
    else:
        step_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
