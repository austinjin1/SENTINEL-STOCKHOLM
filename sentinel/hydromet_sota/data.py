# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lever training data for the M6 SOTA campaign (``docs/preregistration/hydromet_sota_protocol.md`` §2–§4).

:func:`load_lever_train` builds one :class:`~sentinel.models.hydromet.data.HydroMetData` over **train-role locations ×
dates up to the end of the train block only**:

* FINAL train-role eligible sites (1,712; fit + the 256 inner sites, recomputed exactly as the reference runs);
* optional new USGS fit sites (``processed/hydromet_sota/lever_sites.parquet``, rule ``fit``): with temperature/DO
  (lever 1a) and/or discharge only (lever 2);
* optional WQP weak-label stations (lever 1b; ``wqp_train_obs.parquet``, train block only).

Inputs follow the M6 methods (site-cell gridMET + upstream-basin gridMET appended, StreamCat statics, DOY); the
optional auxiliary target is log(discharge + 0.01) at USGS sites (never an input). The guard is called with
``training=True`` for (train block, train role); every location is checked against the M1-reserved list.

:func:`load_val_cell` loads the FINAL val-role eligible sites × (train, val) blocks for descriptive val scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import paths
from ..models.hydromet import data as D
from ..models.hydromet import guard
from . import reserved as R
from . import sites as LS

SOTA_DIR = paths.PROCESSED / "hydromet_sota"
UPSTREAM_SOTA = paths.PROCESSED / "hydromet_upstream_forcing" / "upstream_forcing_sota.parquet"
UPSTREAM_TRACKN = paths.PROCESSED / "hydromet_upstream_forcing" / "upstream_forcing_trackn.parquet"
WQP_DIR = paths.PROCESSED / "hydromet_wqp_national"
Q_OFFSET = 0.01
PINS_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_sota_lever_pins.json"


def verify_lever_pins() -> dict:
    """The lever pool and weak-label table must match their pins (data-hygiene rule 4) before any fit."""
    import hashlib
    import json

    pins = json.loads(PINS_FILE.read_text())
    for f in ("lever_sites.parquet", "wqp_train_obs.parquet"):
        h = hashlib.sha256((SOTA_DIR / f).read_bytes()).hexdigest()
        if h != pins["outputs"][f]:
            raise RuntimeError(f"{f} sha256 {h[:12]} != pinned {pins['outputs'][f][:12]} ({PINS_FILE.name})")
    return pins


def _wqp_forcings(keys, dates) -> np.ndarray:
    import pyarrow.dataset as ds

    S, T = len(keys), len(dates)
    pos = {k: i for i, k in enumerate(keys)}
    forc = np.full((S, T, len(D.FORCING_VARS)), np.nan, dtype="float32")
    dset = ds.dataset(WQP_DIR / "forcings.parquet", format="parquet")
    for c in range(0, S, 500):
        kk = keys[c:c + 500]
        tab = dset.to_table(filter=ds.field("station_key").isin(kk) & (ds.field("date") >= dates[0].date())
                            & (ds.field("date") <= dates[-1].date())).to_pandas()
        si = tab["station_key"].map(pos).to_numpy()
        ti = (pd.to_datetime(tab["date"]) - dates[0]).dt.days.to_numpy()
        for j, v in enumerate(D.FORCING_RAW):
            forc[si, ti, j] = tab[v].to_numpy(dtype="float32")
    forc[:, :, D.FORCING_VARS.index("tmean_c")] = 0.5 * (forc[:, :, D.FORCING_VARS.index("tmax_c")]
                                                       + forc[:, :, D.FORCING_VARS.index("tmin_c")])
    return forc


def _wqp_static(keys, columns) -> pd.DataFrame:
    st = pd.read_parquet(WQP_DIR / "stations.parquet").set_index("station_key").loc[keys]
    sc = pd.read_parquet(WQP_DIR / "streamcat_comid.parquet").drop_duplicates("comid").set_index("comid")
    feat = D.streamcat_feature_columns()
    static = sc.reindex(st["comid"].astype(str))[[c for c in feat if c in sc.columns]].reindex(columns=feat).astype("float64")
    static.index = keys
    static.insert(0, "elev_m", np.nan)
    static.insert(0, "lon", st["lon"].to_numpy(dtype="float64"))
    static.insert(0, "lat", st["lat"].to_numpy(dtype="float64"))
    static["elev_m"] = static["elev_m"].fillna(static.get("elevcat"))   # no gauge datum at WQP stations (as track N)
    return static.reindex(columns=columns)


def _append_upstream(forc_site: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, dict]:
    miss = ~np.isfinite(up) & np.isfinite(forc_site)
    no_basin = int((~np.isfinite(up[:, :, D.FORCING_VARS.index("tmax_c")])).all(axis=1).sum())
    up = np.where(miss, forc_site, up).astype("float32")
    return np.concatenate([forc_site, up], axis=-1), {"sites_without_upstream": no_basin,
                                                       "filled_from_site_frac": float(miss.mean())}


def load_lever_train(*, lever1a: bool = False, q_only_sites: bool = False, weak: bool = False, aux_q: bool = False,
                     log=print):
    """Returns ``(data, fit_idx, inner_idx, weak_idx, aux_y, info)``; ``aux_y`` is (S, T, 1) log-discharge or None."""
    sp, sinfo, final, elig, inner_sites, inner_groups = LS.final_frame()
    if not LS.check_inner_matches_reference(inner_sites):
        raise RuntimeError("inner sites differ from the reference run ealstm_ws_v1_s0")
    train_sites = [s for s in elig if sp.site_role[s] == "train"]
    if lever1a or q_only_sites or weak:
        verify_lever_pins()
    pool = pd.read_parquet(SOTA_DIR / "lever_sites.parquet")
    pool = pool[pool["rule"] == "fit"]
    if set(pool["basin_group"]) & set(inner_groups) or set(pool["group_role"]) - {"train"}:
        raise RuntimeError("lever pool contains non-fit groups")
    usgs = pool[pool["source"] == "usgs_dv"]
    add = set()
    if lever1a:
        add |= set(usgs.loc[usgs["lever1a"], "site_no"])
    if q_only_sites:
        add |= set(usgs.loc[usgs["lever2_q"] & ~usgs["lever1a"], "site_no"])
    new_sites = sorted(add)
    wqp = pool[pool["source"] == "wqp_grab"] if weak else pool.iloc[:0]
    wqp_keys = sorted(wqp["key"])
    usgs_sites = train_sites + new_sites
    R.assert_no_reserved(usgs_sites)
    R.assert_no_reserved([k.split(":", 1)[-1] for k in wqp_keys])
    res = R.reserved_usgs_sites()
    if any(LS.contains_reserved_id(k, res) for k in wqp_keys):
        raise R.ReservedSiteError("reserved number inside a weak-label station key")
    from ..data import reserved as shared_reserved   # M1 shared helper (lead notice 2026-11-10)

    shared_reserved.assert_no_reserved(usgs_sites, where="M6 SOTA lever training set")
    blocks = ("train",)
    access = guard.check_access(blocks, {"train"}, split=sp, training=True)
    end = guard.data_end_date(blocks)
    dates = pd.date_range(D.DATA_START, end, freq="D")
    st = D.site_table()
    keys = st.loc[usgs_sites, "site_key"].tolist()
    log(f"[sota.data] {len(train_sites)} FINAL train-role + {len(new_sites)} new USGS sites + {len(wqp_keys)} WQP "
        f"stations; {dates[0].date()}..{dates[-1].date()}")
    forc_u = D.load_forcings(keys, dates)
    up_u = D.load_upstream_forcings(keys, dates)
    if new_sites:
        up_new = D.load_upstream_forcings(keys, dates, path=UPSTREAM_SOTA)
        up_u = np.where(np.isfinite(up_u), up_u, up_new)
    forc_u, up_info = _append_upstream(forc_u, up_u)
    static_u = D.load_static(usgs_sites, st)
    cube = D.load_cube(usgs_sites, D.CUBE_VARS, start=D.DATA_START, end=end, retries=2, wait_s=10.0)
    guard.assert_cube_within(cube.dates, blocks)
    y_u, prov_u = D.derive_targets(cube.y, cube.prov, cube.variables)
    aux_u = None
    if aux_q:
        qc = D.load_cube(usgs_sites, ["discharge_cfs_mean"], start=D.DATA_START, end=end, retries=2, wait_s=10.0)
        q = qc.y[:, :, 0].astype("float64")
        aux_u = np.where(np.isfinite(q) & (q >= 0), np.log(np.maximum(q, 0) + Q_OFFSET), np.nan)[:, :, None]
    groups_u = [str(sp.basin_group[s]) for s in train_sites] + \
        usgs.set_index("site_no").loc[new_sites, "basin_group"].astype(str).tolist()
    sites_all = list(usgs_sites)
    forc, y, prov, static, groups = forc_u, y_u, prov_u, static_u, groups_u
    aux = aux_u
    if wqp_keys:
        forc_w = _wqp_forcings(wqp_keys, dates)
        up_w = D.load_upstream_forcings(wqp_keys, dates, path=UPSTREAM_TRACKN)
        forc_w, up_info_w = _append_upstream(forc_w, up_w)
        up_info["wqp"] = up_info_w
        static_w = _wqp_static(wqp_keys, list(static_u.columns))
        obs = pd.read_parquet(SOTA_DIR / "wqp_train_obs.parquet")
        obs = obs[obs["station_key"].isin(set(wqp_keys))]
        if obs["date"].max() > pd.Timestamp(LS.TRAIN_END):
            raise guard.ProtectedDataError("weak labels outside the train block")
        y_w = np.full((len(wqp_keys), len(dates), len(D.TARGETS)), np.nan, dtype="float32")
        pos = {k: i for i, k in enumerate(wqp_keys)}
        for v in ("temp_c_mean", "do_mgl_mean"):
            o = obs[obs["target"] == v]
            y_w[o["station_key"].map(pos).to_numpy(), (o["date"] - dates[0]).dt.days.to_numpy(), D.TARGETS.index(v)] = \
                o["value"].to_numpy(dtype="float32")
        forc = np.concatenate([forc_u, forc_w], axis=0)
        y = np.concatenate([y_u, y_w], axis=0)
        prov = np.concatenate([prov_u, np.zeros_like(y_w, dtype=bool)], axis=0)
        static = pd.concat([static_u, static_w])
        groups = groups_u + wqp.set_index("key").loc[wqp_keys, "basin_group"].astype(str).tolist()
        sites_all = sites_all + wqp_keys
        if aux is not None:
            aux = np.concatenate([aux, np.full((len(wqp_keys), len(dates), 1), np.nan)], axis=0)
    roles = np.array(["train"] * len(sites_all))
    data = D.HydroMetData(sites_all, roles, np.array(groups), dates, forc, y, prov, static, sp, sinfo, access, blocks,
                          {"n_sites": len(sites_all), "roles": {"train": len(sites_all)},
                           "upstream_forcing": {"mode": "append", **up_info}, "date_start": str(dates[0].date()),
                           "date_end": str(dates[-1].date())})
    pos = {s: i for i, s in enumerate(sites_all)}
    inner_idx = np.array([pos[s] for s in inner_sites], int)
    fit_idx = np.array([pos[s] for s in usgs_sites if s not in set(inner_sites)], int)
    weak_idx = np.array([pos[k] for k in wqp_keys], int)
    if set(data.basin_group[np.r_[fit_idx, weak_idx]]) & set(inner_groups):
        raise RuntimeError("a fit / weak location lies in an inner group")
    data.info["inner_sites"] = list(inner_sites)
    info = {"n_final_train": len(train_sites), "n_new_usgs": len(new_sites), "n_wqp": len(wqp_keys),
            "n_fit": int(len(fit_idx)), "n_inner": int(len(inner_idx)), "n_weak": int(len(weak_idx)),
            "inner_groups": inner_groups, "split_sha256": sinfo["sha256"], "upstream": up_info,
            "aux_q_sites": int(np.isfinite(aux[:, :, 0]).any(axis=1).sum()) if aux is not None else 0,
            "weak_station_days": {v: int(np.isfinite(y[weak_idx][:, :, D.TARGETS.index(v)]).sum())
                                  for v in ("temp_c_mean", "do_mgl_mean")} if len(weak_idx) else {}}
    log(f"[sota.data] fit {info['n_fit']} / inner {info['n_inner']} / weak {info['n_weak']}; aux-Q sites "
        f"{info['aux_q_sites']}; upstream {up_info}")
    return data, fit_idx, inner_idx, weak_idx, aux, info


def load_val_cell(*, log=print):
    """FINAL val-role eligible sites × (train, val) blocks with upstream forcings appended (descriptive scoring)."""
    sp, sinfo = D.load_split()
    val_sites, _ = D.eligible_sites(sp, ("val",))
    data = D.prepare_data(sp, sinfo, val_sites, blocks=("train", "val"), training=True, log=log)
    D.attach_upstream_forcing(data, "append", log=log)
    data.info["inner_sites"] = []
    return data
