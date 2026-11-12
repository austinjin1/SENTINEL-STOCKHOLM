# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Common-input-schema loaders for track T-E (``docs/preregistration/hydromet_ea_protocol.md`` §3–§4).

One schema for every model and both domains:
* forcing: the 11 M6 site-cell channels (:data:`sentinel.models.hydromet.data.FORCING_VARS`) — gridMET for US sites,
  ERA5-Land (Open-Meteo, harmonised) at the station's ERA5-Land grid point for EA stations;
* statics: :data:`COMMON_STATIC` from the nearest RiverATLAS v1.0 reach (both domains);
* calendar: DOY harmonics (added by the model code, as in M6).

:func:`load_ea` builds a :class:`~sentinel.models.hydromet.data.HydroMetData` over EA stations of the frozen
``hydromet_ea_split.json`` roles (both split files verified by ``ea_split``; M1-test groups dropped for all years).
It **refuses** the EA-test block (≥ 2023-10-01) and test-role groups unconditionally
(the single test look has its own guarded loader after the lead's freeze); M1-reserved US sites never occur here.
:func:`load_us_common` gives US FINAL train/val-role sites in the same schema (gridMET + RiverATLAS statics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import paths
from ..models.hydromet import data as D
from . import ea_split as ES

EA = paths.PROCESSED / "ea_wq_sonde"
SPLIT_FILE = ES.SPLIT_FILE
M1_SPLIT_FILE = ES.M1_SPLIT_FILE
US_ATLAS = paths.PROCESSED / "hydromet_sota" / "riveratlas_us_sites.parquet"
RESERVED_FROM = pd.Timestamp("2023-10-01")
EA_BLOCKS = {"ea_train": ("2012-01-01", "2021-09-30"), "ea_val": ("2021-10-01", "2023-09-30")}
COMMON_STATIC = ("lat", "lon", "ele_mt_cav", "ele_mt_uav", "slp_dg_cav", "slp_dg_uav", "sgr_dk_rav", "UPLAND_SKM",
                 "CATCH_SKM", "ORD_STRA", "tmp_dc_cyr", "tmp_dc_uyr", "tmp_dc_cmx", "tmp_dc_cmn", "pre_mm_cyr",
                 "pre_mm_uyr", "pet_mm_uyr", "ari_ix_uav", "cmi_ix_uyr", "snw_pc_uyr", "dis_m3_pyr", "run_mm_cyr",
                 "gwt_cm_cav", "lka_pc_use", "dor_pc_pva", "rev_mc_usu", "for_pc_cse", "for_pc_use", "crp_pc_use",
                 "pst_pc_use", "urb_pc_cse", "urb_pc_use", "wet_pc_ug1", "cly_pc_uav", "snd_pc_uav", "soc_th_uav",
                 "kar_pc_use", "ppd_pk_cav", "ppd_pk_uav", "hft_ix_u09", "rdd_mk_uav", "nli_ix_uav")
DATA_START = "2011-01-01"


class EAProtectedError(PermissionError):
    pass


def load_split() -> dict:
    """The frozen M6 England split, verified by the one shared check (``ea_split.verify``; never loosened)."""
    try:
        return ES.load(SPLIT_FILE)
    except ES.SplitError as e:
        raise EAProtectedError(f"hydromet_ea_split.json refused: {e}") from e


def load_splits() -> tuple:
    """(M6 split, M1-test groups): both ``hydromet_ea_split.json`` and ``sonde_ea_split.json`` must exist with a
    verified sha256 before any England row is loaded (lead ruling Q8)."""
    try:
        return ES.load_pair(SPLIT_FILE, M1_SPLIT_FILE)
    except ES.SplitError as e:
        raise EAProtectedError(f"England split files refused: {e}") from e


def _blocks_end(blocks) -> pd.Timestamp:
    bad = [b for b in blocks if b not in EA_BLOCKS]
    if bad:
        raise EAProtectedError(f"EA blocks {bad} are not loadable here (EA-test is reserved until the T-E look)")
    return max(pd.Timestamp(EA_BLOCKS[b][1]) for b in blocks)


def _7dadm(x: pd.DataFrame) -> pd.DataFrame:
    out = []
    for stn, g in x.groupby("station"):
        s = g.set_index("date")["temp_c_max"].asfreq("D")
        out.append(s.rolling(7, min_periods=7).mean().rename("temp_c_7dadm").dropna().reset_index().assign(station=stn))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "temp_c_7dadm", "station"])


def load_ea(roles=("train", "val"), blocks=("ea_train", "ea_val"), *, log=print) -> D.HydroMetData:
    """EA stations of ``roles`` (never 'test') × dates up to the end of ``blocks`` (never EA-test)."""
    if "test" in roles:
        raise EAProtectedError("EA test-role groups are reserved (allocation rule 2); use the frozen T-E look")
    end = _blocks_end(blocks)
    if end >= RESERVED_FROM:
        raise EAProtectedError("EA values >= 2023-10-01 are reserved")
    bad = [r for r in roles if r not in ("train", "val")]
    if bad:
        raise EAProtectedError(f"EA roles {bad} are not loadable (only eligible train/val-role stations supply M6 rows)")
    split, m1_test = load_splits()
    m1_test = set(m1_test)       # M1-test groups are dropped for all years (lead ruling Q8)
    stations = [s for s in ES.m6_stations(split, roles) if split["station_group"][s] not in m1_test]
    if any(s in set(split["closed_stations"]) for s in stations):
        raise EAProtectedError("a closed (test / reserved) station reached the loader")
    dates = pd.date_range(DATA_START, end, freq="D")
    S, T = len(stations), len(dates)
    pos = {s: i for i, s in enumerate(stations)}
    geo = pd.read_parquet(EA / "stations_geo.parquet").set_index("station").loc[stations]
    # forcing (ERA5-Land grid point of the station)
    from ..data.sources.openmeteo import cell_id

    cells = [cell_id(a, b) for a, b in zip(geo["lat"], geo["lon"])]
    f = pd.read_parquet(EA / "forcings_era5land.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f = f[(f["date"] >= dates[0]) & (f["date"] <= dates[-1])].set_index(["cell", "date"])
    forc = np.full((S, T, len(D.FORCING_VARS)), np.nan, dtype="float32")
    for i, c in enumerate(cells):
        if c in f.index.get_level_values(0):
            g = f.loc[c].reindex(dates)
            for j, v in enumerate(D.FORCING_RAW):
                forc[i, :, j] = g[v].to_numpy(dtype="float32")
    forc[:, :, D.FORCING_VARS.index("tmean_c")] = 0.5 * (forc[:, :, D.FORCING_VARS.index("tmax_c")]
                                                       + forc[:, :, D.FORCING_VARS.index("tmin_c")])
    # statics (RiverATLAS reach of the station)
    at = pd.read_parquet(EA / "riveratlas_gb.parquet").set_index("HYRIV_ID")
    st = at.reindex(geo["hyriv_id"]).reset_index(drop=True)
    st.index = stations
    st["lat"], st["lon"] = geo["lat"].to_numpy(), geo["lon"].to_numpy()
    static = st.reindex(columns=list(COMMON_STATIC)).astype("float64")
    static.insert(2, "elev_m", static["ele_mt_cav"].to_numpy())   # pressure for DO saturation (M6 name)
    # targets (valid days only; EA-test never present in the processed table)
    w = pd.read_parquet(EA / "station_daily.parquet")
    w = w[w["station"].isin(pos) & (w["date"] <= dates[-1]) & (w["date"] >= dates[0])]
    if (w["date"] >= RESERVED_FROM).any():
        raise EAProtectedError("reserved-period row in the EA daily table")
    w = w.merge(_7dadm(w[w["temp_c_max"].notna()][["station", "date", "temp_c_max"]]), on=["station", "date"], how="left")
    y = np.full((S, T, len(D.TARGETS)), np.nan, dtype="float32")
    si = w["station"].map(pos).to_numpy()
    ti = (w["date"] - dates[0]).dt.days.to_numpy()
    for v in ("temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min"):
        ok = w[v].notna().to_numpy()
        y[si[ok], ti[ok], D.TARGETS.index(v)] = w.loc[ok, v].to_numpy(dtype="float32")
    from ..data.sonde_splits import SondeSplit

    sp = SondeSplit({s: split["station_role"][s] for s in stations}, {s: split["station_group"][s] for s in stations},
                    time_blocks={"train": EA_BLOCKS["ea_train"], "val": EA_BLOCKS["ea_val"],
                                 "test": ("2023-10-01", "2026-09-30")}, notes="EA T-E split (hydromet_ea_split.json)")
    data = D.HydroMetData(stations, np.array([split["station_role"][s] for s in stations]),
                          np.array([split["station_group"][s] for s in stations]), dates, forc, y,
                          np.zeros_like(y, dtype=bool), static, sp, {"sha256": split["sha256"], "final": False,
                                                                     "path": str(SPLIT_FILE)},
                          {"protected": [], "blocks": list(blocks), "roles": list(roles), "m1_test_groups_dropped": sorted(m1_test)}, tuple(
                              "train" if b == "ea_train" else "val" for b in blocks),
                          {"n_sites": S, "domain": "EA", "split_sha256": split["sha256"]})
    log(f"[ea] {S} stations {pd.Series(data.roles).value_counts().to_dict()}; {dates[0].date()}..{dates[-1].date()}")
    return data


def common_static_us(site_keys) -> pd.DataFrame:
    """RiverATLAS COMMON_STATIC for US locations (``riveratlas_us_sites.parquet``, nearest reach), index = site_key."""
    at = pd.read_parquet(US_ATLAS).drop_duplicates("site_key").set_index("site_key")
    return at.reindex(list(site_keys))


MAX_SNAP_M = 1000.0   # same rule in both domains (EA eligibility, hydromet_ea_protocol.md §3-§4)


def load_us_common(roles=("train", "val"), blocks=("train", "val"), *, log=print,
                   max_snap_m: float | None = MAX_SNAP_M) -> D.HydroMetData:
    """US FINAL train/val-role eligible sites in the T-E common schema: gridMET site-cell forcings (11 channels), the
    RiverATLAS COMMON_STATIC set (``elev_m`` = ``ele_mt_cav`` as for EA stations), M6 targets. Training-grade guard
    (never test blocks or test-role sites). Sites whose nearest RiverATLAS reach is farther than ``max_snap_m`` (or
    absent) are left out, as EA stations are (metadata-only rule; ``None`` keeps every site)."""
    from ..data.reserved import assert_no_reserved

    sp, info = D.load_split()
    sites, _ = D.eligible_sites(sp, tuple(roles))
    if max_snap_m is not None:
        snap = common_static_us(D.site_table().loc[sites, "site_key"].tolist())["snap_m"].to_numpy(dtype="float64")
        keep = np.isfinite(snap) & (snap <= float(max_snap_m))
        log(f"[us-common] {int(keep.sum())} of {len(sites)} sites have a RiverATLAS reach within {max_snap_m:.0f} m")
        sites = [s for s, k in zip(sites, keep) if k]
    assert_no_reserved(sites, where="T-E US common-schema load")
    data = D.prepare_data(sp, info, sites, blocks=tuple(blocks), training=True, log=log)
    st = D.site_table().loc[sites]
    at = common_static_us(st["site_key"].tolist())
    at.index = sites
    at["lat"], at["lon"] = st["lat"].to_numpy(), st["lon"].to_numpy()
    static = at.reindex(columns=list(COMMON_STATIC)).astype("float64")
    static.insert(2, "elev_m", static["ele_mt_cav"].to_numpy())
    data.static = static
    data.info["static_schema"] = "T-E common (RiverATLAS COMMON_STATIC)"
    return data
