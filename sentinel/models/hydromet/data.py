# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Data access for SENTINEL-HydroMet (M6): estimation of daily water quality at reaches WITHOUT any
in-situ sensor input.

**Inputs (the only three groups allowed; enforced by :func:`assert_allowed_inputs`):**

* ``forcing`` — gridMET daily observed weather at the site cell (``processed/forcings_daily``):
  precipitation, Tmax/Tmin (+ derived Tmean), shortwave radiation, wind, RH max/min, specific
  humidity, VPD, reference ET. In the default *nowcast/reanalysis* setting the model for day t may
  use forcings of days ≤ t (``forcing_lag_days = 0``); ``forcing_lag_days = L > 0`` restricts it to
  days ≤ t − L (a forecast without NWP; an NWP-driven forecast mode is future work and must use
  ``nwp_archive``, never observed future weather).
* ``static`` — lat, lon, elevation (site registry) and the StreamCat catchment/watershed attributes
  (``processed/catchment/streamcat_sites.parquet``; model-predicted stream temperature is already
  excluded by the datasheet). Sensor-network metadata (``has_*``, ``n_wq_series``, ``wq_*``,
  numbers of up/downstream monitoring sites, NWIS drainage area) is NOT used: it does not exist for
  an unsensed reach.
* ``calendar`` — day-of-year harmonics.

Gauged discharge/stage are sensor measurements and are never inputs.

**Targets** (USGS daily statistics, ``processed/sonde_daily``): ``temp_c_mean``, ``temp_c_7dadm``
(trailing 7-day mean of the daily maximum, defined only when all 7 days are present),
``do_mgl_mean``, ``do_mgl_min``, ``spcond_uscm_mean``, ``turb_fnu_mean``, ``no3no2_mgl_mean``; the
last three are modelled as log(x + offset) (offsets from :data:`sentinel.models.sonde.data.LOG_VARS`).

**Split.** The frozen M1 split (``docs/preregistration/sonde_split.json``): models are fitted on
train-role sites × the train block only; the evaluation cell is val-role sites (unseen basin groups)
× the val block (42-day embargo after its start, :meth:`SondeSplit.time_mask`). Every normalisation
statistic is computed on train-role sites × train-block dates only (:class:`Normaliser`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core import paths
from ...data.sonde_splits import TIME_BLOCKS, SondeSplit
from ...eval.splits import assert_disjoint
from ..sonde.data import LOG_VARS, load_cube
from . import guard

TARGETS = ["temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean", "turb_fnu_mean",
           "no3no2_mgl_mean"]
LOG_TARGETS = {v: LOG_VARS[v] for v in ("spcond_uscm_mean", "turb_fnu_mean", "no3no2_mgl_mean")}
CUBE_VARS = ["temp_c_mean", "temp_c_max", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean", "turb_fnu_mean",
             "no3no2_mgl_mean"]
FORCING_RAW = ["precip_mm", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "rhmax_pct", "rhmin_pct", "sph_kgkg",
               "vpd_kpa", "pet_mm"]
FORCING_VARS = FORCING_RAW + ["tmean_c"]
STREAM_TYPES = ("ST", "ST-TS", "ST-CA", "ST-DCH")
DATA_START = "2005-10-01"   # forcing history/spin-up: 2 years before the train block (no targets used there)
FINAL_SPLIT = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json"
FORCINGS_FILE = paths.PROCESSED / "forcings_daily" / "forcings_daily.parquet"
FORCINGS_COVERAGE = paths.PROCESSED / "forcings_daily" / "coverage.parquet"
STREAMCAT_FILE = paths.PROCESSED / "catchment" / "streamcat_sites.parquet"
SITES_FILE = paths.PROCESSED / "sites" / "sites.parquet"
SONDE_INDEX = paths.PROCESSED / "sonde_daily" / "index.parquet"
SITE_STATIC = ["lat", "lon", "elev_m"]
STREAMCAT_NON_FEATURES = {"site_key", "comid", "comid_method", "streamcat_found"}
CALENDAR_COLS = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
# Any input whose name looks like an in-situ measurement or monitoring-network metadata is refused.
SENSOR_PATTERN = re.compile(
    r"(temp_c|water_temp|do_mgl|do_pct|spcond|turb_|turbid|no3|nitrate|^ph_|_ph$|chl_|fdom|pc_rfu|pc_ugl|"
    r"discharge|gage|stage|"
    r"salinity|^has_|n_wq|^wq_|upstream_sites|downstream_sites|drain_area|_obs|target)", re.I)


# ----------------------------------------------------------------------------- transforms

def to_model(var: str, x):
    x = np.asarray(x, dtype="float64")
    return np.log(np.maximum(x, 0) + LOG_TARGETS[var]) if var in LOG_TARGETS else x


def from_model(var: str, z):
    z = np.asarray(z, dtype="float64")
    return np.maximum(np.exp(z) - LOG_TARGETS[var], 0.0) if var in LOG_TARGETS else z


def doy_harmonics(dates) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    w = 2 * np.pi * (d.dayofyear.to_numpy() - 1) / 365.25
    return pd.DataFrame({"doy_sin1": np.sin(w), "doy_cos1": np.cos(w), "doy_sin2": np.sin(2 * w),
                         "doy_cos2": np.cos(2 * w)}, index=d)


# ----------------------------------------------------------------------------- input whitelist

def streamcat_feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    if df is None:
        import pyarrow.parquet as pq

        names = pq.read_schema(STREAMCAT_FILE).names
        df = pd.DataFrame(columns=names)
    return [c for c in df.columns if c not in STREAMCAT_NON_FEATURES]


def allowed_input_columns(static_cols, forcing_feature_cols) -> dict[str, set]:
    return {"static": set(SITE_STATIC) | set(static_cols), "forcing": set(forcing_feature_cols),
            "calendar": set(CALENDAR_COLS)}


def assert_allowed_inputs(columns, *, static_cols, forcing_feature_cols) -> dict[str, list[str]]:
    """Every model input column must be a forcing feature, a static attribute or a calendar term,
    and none may look like a sensor measurement. Returns the column -> group assignment."""
    groups = allowed_input_columns(static_cols, forcing_feature_cols)
    out: dict[str, list[str]] = {"static": [], "forcing": [], "calendar": []}
    bad, sensorish = [], []
    for c in columns:
        g = next((k for k, s in groups.items() if c in s), None)
        if g is None:
            bad.append(c)
        else:
            out[g].append(c)
        if SENSOR_PATTERN.search(str(c)):
            sensorish.append(c)
    if bad:
        raise ValueError(f"input columns outside forcing/static/calendar groups: {bad[:20]}")
    if sensorish:
        raise ValueError(f"input columns look sensor-derived: {sensorish[:20]}")
    return out


# ----------------------------------------------------------------------------- loaders

def load_split(path=None, *, require_final: bool = True) -> tuple[SondeSplit, dict]:
    from pathlib import Path

    p = Path(path) if path else FINAL_SPLIT
    if not p.is_absolute():
        p = paths.REPO_ROOT / p
    sp = SondeSplit.load(p)
    assert_disjoint(train=sp.sites("train"), val=sp.sites("val"), test=sp.sites("test"))
    final = p.resolve() == FINAL_SPLIT.resolve() and "FINAL" in (sp.notes or "")
    if not final and require_final:
        raise guard.ProtectedDataError(f"split {p} is not the FINAL split; M6 never loads sites under another "
                                       "split (its roles could expose FINAL test-role sites)")
    return sp, {"path": str(p), "sha256": sp.sha256(), "final": bool(final), "notes": sp.notes,
                "n_train": len(sp.sites("train")), "n_val": len(sp.sites("val")), "n_test": len(sp.sites("test"))}


def site_table() -> pd.DataFrame:
    """Eligible-site metadata for USGS sites: lat/lon/elev, type, forcing source, StreamCat flag."""
    s = pd.read_parquet(SITES_FILE)
    s = s[s["network"] == "USGS"].drop_duplicates("site_id").copy()
    s["site_no"] = s["site_id"].astype(str)
    s = s.set_index("site_no")
    cov = pd.read_parquet(FORCINGS_COVERAGE).set_index("site_key")
    sc = pd.read_parquet(STREAMCAT_FILE, columns=["site_key", "streamcat_found"]).set_index("site_key")
    out = pd.DataFrame(index=s.index)
    out["site_key"] = s["site_key"].astype(str)
    out["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(s["lon"], errors="coerce")
    out["elev_m"] = pd.to_numeric(s["elev_m"], errors="coerce")
    out["site_type"] = s["site_type"].astype(str)
    out["forcing_source"] = cov["source"].reindex(out["site_key"]).to_numpy()
    out["streamcat_found"] = sc["streamcat_found"].reindex(out["site_key"]).fillna(False).astype(bool).to_numpy()
    return out


def eligible_sites(split: SondeSplit, roles=("train", "val")) -> tuple[list[str], dict]:
    """Stream sites in the split with gridMET forcings and StreamCat attributes (CONUS)."""
    st = site_table()
    cand = [s for s in split.site_role if split.site_role[s] in roles]
    info = {"candidates": len(cand)}
    st = st.reindex(cand)
    keep = st["site_type"].isin(STREAM_TYPES) & (st["forcing_source"] == "gridmet") & st["streamcat_found"]
    info["excluded_not_stream"] = int((~st["site_type"].isin(STREAM_TYPES)).sum())
    info["excluded_no_gridmet"] = int((st["forcing_source"] != "gridmet").sum())
    info["excluded_no_streamcat"] = int((~st["streamcat_found"]).sum())
    sites = sorted(st.index[keep.to_numpy()])
    info["eligible"] = len(sites)
    return sites, info


def load_forcings(site_keys: list[str], dates: pd.DatetimeIndex) -> np.ndarray:
    """(S, T, F) float32 forcings in :data:`FORCING_VARS` order (raw units), NaN where missing."""
    import pyarrow.dataset as ds

    S, T = len(site_keys), len(dates)
    out = np.full((S, T, len(FORCING_VARS)), np.nan, dtype="float32")
    pos = {k: i for i, k in enumerate(site_keys)}
    dset = ds.dataset(FORCINGS_FILE, format="parquet")
    lo, hi = dates[0].date(), dates[-1].date()
    for chunk in range(0, S, 400):
        keys = site_keys[chunk:chunk + 400]
        flt = (ds.field("site_key").isin(keys)) & (ds.field("date") >= lo) & (ds.field("date") <= hi)
        tab = dset.to_table(columns=["site_key", "date"] + FORCING_RAW, filter=flt).to_pandas()
        if not len(tab):
            continue
        si = tab["site_key"].map(pos).to_numpy()
        ti = (pd.to_datetime(tab["date"]) - dates[0]).dt.days.to_numpy()
        for j, v in enumerate(FORCING_RAW):
            out[si, ti, j] = tab[v].to_numpy(dtype="float32")
    out[:, :, FORCING_VARS.index("tmean_c")] = 0.5 * (out[:, :, FORCING_VARS.index("tmax_c")]
                                                     + out[:, :, FORCING_VARS.index("tmin_c")])
    return out


def load_static(sites: list[str], st: pd.DataFrame | None = None) -> pd.DataFrame:
    st = site_table() if st is None else st
    sc = pd.read_parquet(STREAMCAT_FILE)
    sc = sc.drop_duplicates("site_key").set_index("site_key")
    keys = st.loc[sites, "site_key"].to_numpy()
    feat = streamcat_feature_columns(sc.reset_index())
    out = sc.reindex(keys)[[c for c in feat if c in sc.columns]].astype("float64")
    out.index = sites
    for c in SITE_STATIC[::-1]:
        out.insert(0, c, st.loc[sites, c].to_numpy(dtype="float64"))
    # NWIS elevation is missing at some gauges: fall back to the StreamCat catchment mean elevation
    out["elev_m"] = out["elev_m"].fillna(out.get("elevcat"))
    return out


def derive_targets(cube_y: np.ndarray, cube_prov: np.ndarray, cube_vars: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(S, T, len(TARGETS)) targets (original units) and provisional flags; 7-DADMax derived."""
    S, T, _ = cube_y.shape
    y = np.full((S, T, len(TARGETS)), np.nan, dtype="float32")
    prov = np.zeros((S, T, len(TARGETS)), dtype=bool)
    for j, v in enumerate(TARGETS):
        if v == "temp_c_7dadm":
            k = cube_vars.index("temp_c_max")
            y[:, :, j] = trailing_mean_strict(cube_y[:, :, k], 7)
            p = cube_prov[:, :, k].astype(np.int32)
            cs = np.concatenate([np.zeros((S, 1), np.int32), np.cumsum(p, axis=1)], axis=1)
            win = cs[:, 1:] - cs[:, np.maximum(np.arange(1, T + 1) - 7, 0)]
            prov[:, :, j] = win > 0
        else:
            k = cube_vars.index(v)
            y[:, :, j] = cube_y[:, :, k]
            prov[:, :, j] = cube_prov[:, :, k]
    return y, prov


def trailing_mean_strict(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing w-day mean along axis 1 (days t−w+1..t), NaN unless all w days are finite."""
    S, T = x.shape
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0).astype("float64")
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(v, axis=1)], axis=1)
    cn = np.concatenate([np.zeros((S, 1)), np.cumsum(ok, axis=1)], axis=1)
    out = np.full((S, T), np.nan)
    if T >= w:
        s = cs[:, w:] - cs[:, :-w]
        n = cn[:, w:] - cn[:, :-w]
        out[:, w - 1:] = np.where(n == w, s / w, np.nan)
    return out.astype("float32")


# ----------------------------------------------------------------------------- container

@dataclass
class HydroMetData:
    sites: list[str]
    roles: np.ndarray            # (S,) spatial role
    basin_group: np.ndarray      # (S,) basin group id (split)
    dates: pd.DatetimeIndex
    forc: np.ndarray             # (S, T, F) raw forcing units
    y: np.ndarray                # (S, T, V) targets, original units
    prov: np.ndarray             # (S, T, V) provisional flags
    static: pd.DataFrame         # (S, n_static) raw static attributes (index = sites)
    split: SondeSplit
    split_info: dict
    access: dict
    blocks: tuple
    info: dict = field(default_factory=dict)

    @property
    def targets(self) -> list[str]:
        return list(TARGETS)

    def vi(self, v: str) -> int:
        return TARGETS.index(v)

    def site_idx(self, roles) -> np.ndarray:
        """Indices of sites with the given split roles; the pseudo-roles ``inner`` / ``fit`` select the
        train-role early-stopping groups / the remaining train-role sites (``info['inner_sites']``)."""
        roles = list(roles)
        inner = set(self.info.get("inner_sites", []))
        out = np.zeros(len(self.sites), bool)
        for r in roles:
            if r == "inner":
                out |= np.array([s in inner for s in self.sites])
            elif r == "fit":
                out |= (self.roles == "train") & ~np.array([s in inner for s in self.sites])
            else:
                out |= self.roles == r
        return np.flatnonzero(out)

    def role_label(self, i: int) -> str:
        return "inner" if self.sites[i] in set(self.info.get("inner_sites", [])) else str(self.roles[i])

    def block_mask(self, block: str) -> np.ndarray:
        """(T,) dates usable as targets of ``block`` (non-train blocks: 42-day embargo after start)."""
        if block not in self.blocks:
            raise guard.ProtectedDataError(f"block {block!r} not authorised ({self.blocks})")
        return self.split.time_mask(self.dates, block)

    def target_mask(self, v: str, block: str, site_idx=None, *, approved_only: bool = False) -> np.ndarray:
        """(S, T) bool: observed target ``v`` in ``block`` at ``site_idx`` (others False)."""
        j = self.vi(v)
        m = np.isfinite(self.y[:, :, j]) & self.block_mask(block)[None, :]
        if approved_only:
            m &= ~self.prov[:, :, j]
        if site_idx is not None:
            keep = np.zeros(len(self.sites), bool)
            keep[np.asarray(site_idx, int)] = True
            m &= keep[:, None]
        return m

    def subset(self, idx) -> "HydroMetData":
        idx = np.asarray(idx, int)
        return HydroMetData([self.sites[i] for i in idx], self.roles[idx], self.basin_group[idx], self.dates,
                            self.forc[idx], self.y[idx], self.prov[idx], self.static.iloc[idx], self.split,
                            self.split_info, self.access, self.blocks, dict(self.info))


def prepare_data(split: SondeSplit, split_info: dict, sites: list[str], *, blocks=("train", "val"),
                 allow_test: bool = False, freeze_file=None, training: bool = True, log=print) -> HydroMetData:
    """Authorise blocks/roles (guard), then load forcings, statics and targets up to the end of the
    last authorised block only."""
    roles = np.array([split.site_role.get(s, "none") for s in sites])
    if (roles == "none").any():
        raise ValueError(f"{int((roles == 'none').sum())} sites are not in the split")
    access = guard.check_access(blocks, set(roles.tolist()), allow_test=allow_test, freeze_file=freeze_file,
                                split=split, training=training)
    end = guard.data_end_date(blocks)
    dates = pd.date_range(DATA_START, end, freq="D")
    st = site_table()
    log(f"[hydromet.data] loading forcings for {len(sites)} sites, {dates[0].date()}..{dates[-1].date()}")
    forc = load_forcings(st.loc[sites, "site_key"].tolist(), dates)
    static = load_static(sites, st)
    log("[hydromet.data] loading targets")
    cube = load_cube(sites, CUBE_VARS, start=DATA_START, end=end, retries=2, wait_s=10.0)
    guard.assert_cube_within(cube.dates, blocks)
    y, prov = derive_targets(cube.y, cube.prov, cube.variables)
    bg = np.array([str(split.basin_group.get(s, "")) for s in sites])
    miss = np.isnan(forc[:, :, FORCING_VARS.index("tmax_c")]).mean(axis=1)
    info = {"n_sites": len(sites), "roles": pd.Series(roles).value_counts().to_dict(),
            "forcing_missing_frac_max": float(miss.max()) if len(sites) else None,
            "date_start": str(dates[0].date()), "date_end": str(dates[-1].date())}
    return HydroMetData(list(sites), roles, bg, dates, forc, y, prov, static, split, split_info, access,
                        tuple(blocks), info)


UPSTREAM_FILE = paths.PROCESSED / "hydromet_upstream_forcing" / "upstream_forcing.parquet"
UPSTREAM_PREFIX = "ws_"
UPSTREAM_MODES = ("append", "replace")


def forcing_names(n_channels: int) -> list[str]:
    """Names of the forcing channels of ``HydroMetData.forc``: the site-cell :data:`FORCING_VARS`, followed
    (``upstream_forcing: append``, protocol v0.10) by the same variables averaged over the upstream basin
    (prefix ``ws_``)."""
    if n_channels == len(FORCING_VARS):
        return list(FORCING_VARS)
    if n_channels == 2 * len(FORCING_VARS):
        return list(FORCING_VARS) + [UPSTREAM_PREFIX + v for v in FORCING_VARS]
    raise ValueError(f"unexpected number of forcing channels {n_channels}")


def load_upstream_forcings(site_keys: list[str], dates: pd.DatetimeIndex, path=None) -> np.ndarray:
    """(S, T, F) upstream-basin-averaged gridMET forcings (``processed/hydromet_upstream_forcing``) in
    :data:`FORCING_VARS` order; NaN where the site has no basin or the day is missing."""
    import pyarrow.dataset as ds

    S, T = len(site_keys), len(dates)
    out = np.full((S, T, len(FORCING_VARS)), np.nan, dtype="float32")
    pos = {k: i for i, k in enumerate(site_keys)}
    dset = ds.dataset(path or UPSTREAM_FILE, format="parquet")
    lo, hi = dates[0].date(), dates[-1].date()
    for chunk in range(0, S, 400):
        keys = site_keys[chunk:chunk + 400]
        flt = (ds.field("site_key").isin(keys)) & (ds.field("date") >= lo) & (ds.field("date") <= hi)
        tab = dset.to_table(columns=["site_key", "date"] + FORCING_RAW, filter=flt).to_pandas()
        if not len(tab):
            continue
        si = tab["site_key"].map(pos).to_numpy()
        ti = (pd.to_datetime(tab["date"]) - dates[0]).dt.days.to_numpy()
        for j, v in enumerate(FORCING_RAW):
            out[si, ti, j] = tab[v].to_numpy(dtype="float32")
    out[:, :, FORCING_VARS.index("tmean_c")] = 0.5 * (out[:, :, FORCING_VARS.index("tmax_c")]
                                                     + out[:, :, FORCING_VARS.index("tmin_c")])
    return out


def attach_upstream_forcing(data: "HydroMetData", mode: str, *, path=None, log=print) -> "HydroMetData":
    """Protocol v0.10 (upstream-area forcings, Siddik et al. 2025-style). ``append``: forcing channels become
    site cell + upstream-basin mean (22 channels, :func:`forcing_names`); ``replace``: the upstream-basin
    means replace the site-cell forcings (physics-family parity variant). Where a site has no upstream value
    (no NLDI basin, or a missing day) the site-cell value is used; counts go to ``data.info``. Observed
    weather only — no in-situ input."""
    if mode not in UPSTREAM_MODES:
        raise ValueError(f"upstream_forcing must be one of {UPSTREAM_MODES}, got {mode!r}")
    st = site_table()
    up = load_upstream_forcings(st.loc[data.sites, "site_key"].tolist(), data.dates, path=path)
    site_f = data.forc[:, :, :len(FORCING_VARS)]
    miss = ~np.isfinite(up) & np.isfinite(site_f)
    no_basin = int((~np.isfinite(up[:, :, FORCING_VARS.index("tmax_c")])).all(axis=1).sum())
    up = np.where(miss, site_f, up).astype("float32")
    data.info["upstream_forcing"] = {"mode": mode, "sites_without_upstream": no_basin,
                                     "filled_from_site_frac": float(miss.mean())}
    data.forc = np.concatenate([site_f, up], axis=-1) if mode == "append" else up
    log(f"[hydromet.data] upstream forcing ({mode}): {data.forc.shape[-1]} channels; "
        f"{no_basin} sites without basin, {miss.mean():.4%} cells filled from the site cell")
    return data


GAUGE_DERIVED_STATIC = ("bficat", "bfiws", "runoffcat", "runoffws")


# Protocol v0.9 (cycle 4): a-priori compact static set (location/topography, climate normals, groundwater/hydrology,
# watershed + riparian land cover, dams, soils, channel geometry, N sources); chosen without looking at any val data.
COMPACT_STATIC = (
    "lat", "lon", "elev_m", "elevws", "nhd_slope", "nhd_streamorde", "wsareasqkm",
    "precip9120ws", "tmean9120ws", "tmax9120ws", "tmin9120ws", "precip_minus_evtws",
    "bfiws", "runoffws", "wtdepws", "permws", "rckdepws", "wetindexws",
    "pctimp2019ws", "pctow2019ws", "pctdecid2019ws", "pctconif2019ws", "pctmxfst2019ws", "pctcrop2019ws",
    "pcthay2019ws", "pctgrs2019ws", "pctshrb2019ws", "pctwdwet2019ws", "pcthbwet2019ws", "pcturbhi2019ws",
    "pctdecid2019wsrp100", "pctconif2019wsrp100", "pctmxfst2019wsrp100",
    "damnidstorws", "damdensws", "clayws", "sandws", "omws",
    "bankfullwidth", "wettedwidth", "bankfulldepth",
    "fertws", "manurews", "nsurpws", "wwtpalldensws",
)


def apply_static_ablation(static: pd.DataFrame, ablation: str | None) -> pd.DataFrame:
    """Pre-registered static-input variants (protocol §7, v0.9). ``compact``: keep only
    :data:`COMPACT_STATIC` (cycle 4). ``no_gauge_derived``: drop StreamCat
    attributes interpolated from USGS gauge records (Wolock BFI; McCabe & Wolock runoff) and replace the
    NWIS gauge-datum elevation by the StreamCat (NHDPlus DEM) catchment mean elevation ``elevcat``."""
    if not ablation:
        return static
    if ablation == "compact":
        missing = [c for c in COMPACT_STATIC if c not in static.columns]
        if missing:
            raise ValueError(f"compact static set: columns missing {missing}")
        return static[list(COMPACT_STATIC)].copy()
    if ablation != "no_gauge_derived":
        raise ValueError(f"unknown static ablation {ablation!r}")
    out = static.drop(columns=[c for c in GAUGE_DERIVED_STATIC if c in static.columns]).copy()
    out["elev_m"] = static["elevcat"].to_numpy(dtype="float64") if "elevcat" in static else np.nan
    return out


def inner_split(data: HydroMetData, frac: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split train-role sites by BASIN GROUP into (fit, inner-validation) site indices. The inner
    sites are used for early stopping so that val-role sites stay a clean held-out evaluation."""
    tr = data.site_idx(["train"])
    groups = np.unique(data.basin_group[tr])
    rs = np.random.default_rng(seed)
    order = groups[rs.permutation(len(groups))]
    sizes = pd.Series(data.basin_group[tr]).value_counts()
    # greedy fill in random group order: add a whole group whenever it keeps the inner share <= frac
    # (a prefix rule stops at the first large group and can leave almost no inner sites)
    inner_groups, tot = set(), 0
    for g in order:
        if tot + sizes[g] <= frac * len(tr):
            inner_groups.add(g)
            tot += sizes[g]
    inner_groups = inner_groups or {order[int(np.argmin(sizes.reindex(order).to_numpy()))]}
    is_inner = np.isin(data.basin_group[tr], list(inner_groups))
    fit, inner = tr[~is_inner], tr[is_inner]
    assert_disjoint(fit=[data.sites[i] for i in fit], inner=[data.sites[i] for i in inner],
                    val=[data.sites[i] for i in data.site_idx(["val", "test"])])
    return fit, inner


def smoke_sites(split: SondeSplit, *, n_train: int = 50, n_val: int | None = 0, seed: int = 0) -> list[str]:
    """Small smoke-test site set from the FINAL split: TRAIN-ROLE sites only (review C3a), prioritising
    rarer targets (DO, turbidity, nitrate) so every code path is exercised; smoke runs are scored on
    the inner train-role groups. ``n_val`` must be 0."""
    if n_val:
        raise ValueError("smoke runs use train-role sites only (n_val must be 0)")
    sites, _ = eligible_sites(split, ("train",))
    idx = pd.read_parquet(SONDE_INDEX)
    idx = idx[idx["var"].isin(CUBE_VARS) & (idx["n_days"] >= 365)]
    w = {"no3no2_mgl_mean": 40, "turb_fnu_mean": 15, "do_mgl_mean": 6, "spcond_uscm_mean": 2}
    score = idx.assign(w=idx["var"].map(w).fillna(1)).groupby("site_no")["w"].sum()
    rs = np.random.default_rng(seed)
    out = []
    cand = [s for s in sites if split.site_role[s] == "train" and s in score.index]
    sc = score.reindex(cand) + rs.uniform(0, 3, len(cand))
    out = sc.sort_values(ascending=False).index[:n_train].tolist()
    assert all(split.site_role[s] == "train" for s in out)
    return sorted(out)


# ----------------------------------------------------------------------------- normalisation

@dataclass
class Normaliser:
    """Train-only statistics: forcing mean/std over (fit sites × train-block dates), static
    transform (log10 for heavy-tailed non-negative columns) + mean/std over fit sites, target
    mean/std (model space) over (fit sites × train-block observed values)."""

    forc_mean: np.ndarray
    forc_std: np.ndarray
    static_cols: list
    static_log: list
    static_mean: np.ndarray
    static_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    fit_sites: list

    @classmethod
    def fit(cls, data: HydroMetData, fit_idx, *, max_static_missing: float = 0.5) -> "Normaliser":
        fit_idx = np.asarray(fit_idx, int)
        if not np.isin(data.roles[fit_idx], ["train"]).all():
            raise ValueError("normalisation statistics may only be fitted on train-role sites")
        tm = data.block_mask("train")
        f = data.forc[fit_idx][:, tm].reshape(-1, data.forc.shape[-1]).astype("float64")
        fm, fs = np.nanmean(f, axis=0), np.nanstd(f, axis=0)
        st = data.static.iloc[fit_idx]
        cols = [c for c in data.static.columns if st[c].isna().mean() <= max_static_missing and st[c].nunique() > 1]
        logc = []
        for c in cols:
            x = st[c].dropna()
            if len(x) and x.min() >= 0 and x.skew() > 2.0:
                logc.append(c)
        z = cls._static_transform(st[cols], logc)
        tmn, tsd = [], []
        for j, v in enumerate(TARGETS):
            yy = to_model(v, data.y[fit_idx][:, tm, j])
            yy = yy[np.isfinite(yy)]
            tmn.append(yy.mean() if yy.size else 0.0)
            tsd.append(yy.std() if yy.size > 1 else 1.0)
        return cls(fm, np.maximum(fs, 1e-6), cols, logc, z.mean().to_numpy(), np.maximum(z.std().to_numpy(), 1e-6),
                   np.array(tmn), np.maximum(np.array(tsd), 1e-6), [data.sites[i] for i in fit_idx])

    @staticmethod
    def _static_transform(df: pd.DataFrame, logc) -> pd.DataFrame:
        """Fixed (data-independent) transform: log10(max(x, 0) + 1e-3) on the ``logc`` columns."""
        df = df.astype("float64").copy()
        for c in logc:
            df[c] = np.log10(df[c].clip(lower=0) + 1e-3)
        return df

    def forcing(self, forc: np.ndarray) -> np.ndarray:
        z = (forc - self.forc_mean) / self.forc_std
        return np.nan_to_num(np.clip(z, -10, 10), nan=0.0).astype("float32")

    def static(self, static: pd.DataFrame) -> np.ndarray:
        df = self._static_transform(static.reindex(columns=self.static_cols), self.static_log)
        z = (df.to_numpy() - self.static_mean) / self.static_std
        return np.nan_to_num(np.clip(z, -10, 10), nan=0.0).astype("float32")

    def target(self, v: str, x) -> np.ndarray:
        j = TARGETS.index(v)
        return (to_model(v, x) - self.target_mean[j]) / self.target_std[j]

    def target_inv(self, v: str, z) -> np.ndarray:
        j = TARGETS.index(v)
        return from_model(v, np.asarray(z) * self.target_std[j] + self.target_mean[j])
