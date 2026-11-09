# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Site × day arrays for SENTINEL-Sim (protocol docs/preregistration/sim_protocol.md §2–3).

* **Split**: frozen sonde split (sha256 fd4bb14f…), roles train / val / test from basin groups. Inside
  the train role, 15% of basin groups (seeded) form the **cal** role: never used for gradient updates;
  used for early stopping, pooled error-correction coefficients and pooled residual quantiles.
* **Guard** (:mod:`.guard`): validation mode loads only train/val-role sites and dates ≤ end of the
  validation block (2023-09-30).
* **Inputs** (``x_dyn``, computed from RAW forcings by :func:`dynamic_features`, so scenarios can
  perturb raw values): gridMET Tmax/Tmin/Tmean, shortwave, log1p(precip), wind, RH max/min, VPD,
  log discharge (observed USGS daily mean; gaps ≤ 7 d interpolated, longer gaps → site train-block
  DOY mean of log Q, flagged), discharge-missing flag, DOY harmonics, astronomical day length. Static:
  lat/lon/elevation, NHDPlus log drainage area and log slope, curated StreamCat watershed traits.
  Water-quality observations are NEVER inputs (they are only targets / error-correction anchors).
* **Targets** (USGS daily statistics): temp_c_mean/min/max, do_mgl_mean/min/max, spcond_uscm_mean,
  turb_fnu_mean (last two in log(x + offset) model space).
* **Normalisation**: every mean/std from train-role (non-cal) sites × train-block dates only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core import paths
from ...data.sonde_splits import TIME_BLOCKS, SondeSplit
from ...eval.splits import assert_disjoint
from ..sonde.data import LOG_VARS, load_cube
from . import guard, physics

DATA_START = "2005-10-01"           # forcing/discharge spin-up before the train block (no targets used)
TARGETS = ["temp_c_mean", "temp_c_min", "temp_c_max", "do_mgl_mean", "do_mgl_min", "do_mgl_max",
           "spcond_uscm_mean", "turb_fnu_mean"]
LOG_TARGETS = {"spcond_uscm_mean": LOG_VARS["spcond_uscm_mean"], "turb_fnu_mean": LOG_VARS["turb_fnu_mean"]}
Q_VAR = "discharge_cfs_mean"
FORCING_RAW = ["tmax_c", "tmin_c", "srad_wm2", "precip_mm", "wind_ms", "rhmax_pct", "rhmin_pct", "vpd_kpa"]
DYN_NAMES = ["tmax_c", "tmin_c", "tmean_c", "srad_wm2", "log1p_precip", "wind_ms", "rhmax_pct", "rhmin_pct",
             "vpd_kpa", "log_q", "q_missing", "doy_sin", "doy_cos", "day_length_h"]
STREAMCAT_STATIC = ["pctforest_ws", "pcturban_ws", "pctcrop2019ws", "pctwet_ws", "pctimp2019ws", "pctow2019ws",
                    "bfiws", "runoffws", "precip9120ws", "tmean9120ws", "elevws", "clayws", "sandws", "permws",
                    "wtdepws", "rckdepws", "wetindexws", "log1p_damnidstorws", "log1p_damdensws",
                    "log1p_wwtpalldensws", "log1p_popden2010ws", "pctcarbresidws"]
STATIC_NAMES = ["lat", "lon", "elev_m", "log_da_km2", "log_slope"] + STREAMCAT_STATIC
STREAM_TYPES = ("ST", "ST-TS", "ST-CA", "ST-DCH")
MIN_Q_DAYS = 365
MIN_TARGET_DAYS = 180
CAL_FRAC = 0.15
CAL_SEED = 20261025
Q_FLOOR = 1e-3
SITES_FILE = paths.PROCESSED / "sites" / "sites.parquet"
FORCINGS_FILE = paths.PROCESSED / "forcings_daily" / "forcings_daily.parquet"
STREAMCAT_FILE = paths.PROCESSED / "catchment" / "streamcat_sites.parquet"
NODES_FILE = paths.PROCESSED / "site_network" / "nodes.parquet"
DATA_MANIFESTS = ("sonde_daily", "forcings_daily", "streamcat", "site_network", "sites")


def to_model(var: str, x):
    x = np.asarray(x, dtype="float64")
    return np.log(np.maximum(x, 0) + LOG_TARGETS[var]) if var in LOG_TARGETS else x


def from_model(var: str, z):
    z = np.asarray(z, dtype="float64")
    return np.maximum(np.exp(z) - LOG_TARGETS[var], 0.0) if var in LOG_TARGETS else z


def block_mask(dates: pd.DatetimeIndex, block: str) -> np.ndarray:
    a, b = TIME_BLOCKS[block]
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


# ----------------------------------------------------------------------------- container

@dataclass
class SimData:
    sites: list[str]
    role: np.ndarray                 # (S,) 'train' | 'cal' | 'val' | 'test'
    basin_group: np.ndarray          # (S,)
    dates: pd.DatetimeIndex          # (T,)
    forcing: np.ndarray              # (S, T, len(FORCING_RAW)) raw units
    q_m3s: np.ndarray                # (S, T) filled discharge
    q_missing: np.ndarray            # (S, T) bool
    y: np.ndarray                    # (S, T, len(TARGETS)) original units, NaN missing
    prov: np.ndarray                 # (S, T, len(TARGETS)) provisional flag
    static_raw: np.ndarray           # (S, len(STATIC_NAMES)) raw (NaN allowed)
    lat: np.ndarray
    lon: np.ndarray
    elev: np.ndarray
    slope: np.ndarray                # m/m (NaN → 1e-3 default flagged in static)
    pressure: np.ndarray             # atm
    authorised: dict
    norm: dict = field(default_factory=dict)

    def idx(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.role == role)

    def target_index(self, v: str) -> int:
        return TARGETS.index(v)

    def provenance(self) -> dict:
        h = hashlib.sha256(json.dumps({"sites": self.sites, "roles": self.role.tolist(),
                                       "start": str(self.dates[0]), "end": str(self.dates[-1])}).encode())
        return {"n_sites": len(self.sites), "roles": pd.Series(self.role).value_counts().to_dict(),
                "dates": [str(self.dates[0].date()), str(self.dates[-1].date())],
                "site_set_sha256": h.hexdigest()[:16], "split_sha256": guard.SPLIT_SHA256,
                "mode": self.authorised["mode"]}


# ----------------------------------------------------------------------------- features

def dynamic_features(forcing: np.ndarray, q_m3s: np.ndarray, q_missing: np.ndarray, dates: pd.DatetimeIndex,
                     lat: np.ndarray) -> np.ndarray:
    """Raw forcing (S, T, 8) + discharge → (S, T, len(DYN_NAMES)) un-normalised dynamic features."""
    f = {n: forcing[..., i] for i, n in enumerate(FORCING_RAW)}
    doy = dates.dayofyear.to_numpy()
    w = 2 * np.pi * (doy - 1) / 365.25
    S, T = q_m3s.shape
    out = np.empty((S, T, len(DYN_NAMES)), dtype="float32")
    out[..., 0] = f["tmax_c"]
    out[..., 1] = f["tmin_c"]
    out[..., 2] = 0.5 * (f["tmax_c"] + f["tmin_c"])
    out[..., 3] = f["srad_wm2"]
    out[..., 4] = np.log1p(np.maximum(f["precip_mm"], 0))
    out[..., 5] = f["wind_ms"]
    out[..., 6] = f["rhmax_pct"]
    out[..., 7] = f["rhmin_pct"]
    out[..., 8] = f["vpd_kpa"]
    out[..., 9] = np.log(np.maximum(q_m3s, 0) + Q_FLOOR)
    out[..., 10] = q_missing.astype("float32")
    out[..., 11] = np.sin(w)[None, :]
    out[..., 12] = np.cos(w)[None, :]
    out[..., 13] = physics.day_length_h(lat, doy)
    return out


def _streamcat_static(keys: list[str]) -> pd.DataFrame:
    sc = pd.read_parquet(STREAMCAT_FILE).set_index("site_key").reindex(keys)
    out = pd.DataFrame(index=keys)
    out["pctforest_ws"] = sc[["pctdecid2019ws", "pctconif2019ws", "pctmxfst2019ws"]].sum(axis=1, min_count=1)
    out["pcturban_ws"] = sc[["pcturbop2019ws", "pcturblo2019ws", "pcturbmd2019ws", "pcturbhi2019ws"]].sum(
        axis=1, min_count=1)
    out["pctwet_ws"] = sc[["pctwdwet2019ws", "pcthbwet2019ws"]].sum(axis=1, min_count=1)
    for c in ["pctcrop2019ws", "pctimp2019ws", "pctow2019ws", "bfiws", "runoffws", "precip9120ws", "tmean9120ws",
              "elevws", "clayws", "sandws", "permws", "wtdepws", "rckdepws", "wetindexws", "pctcarbresidws"]:
        out[c] = sc[c]
    for c in ["damnidstorws", "damdensws", "wwtpalldensws", "popden2010ws"]:
        out[f"log1p_{c}"] = np.log1p(np.maximum(sc[c], 0))
    out["nhd_totdasqkm"] = sc["nhd_totdasqkm"]
    out["nhd_slope"] = sc["nhd_slope"]
    return out


def load_forcings(keys: list[str], dates: pd.DatetimeIndex) -> np.ndarray:
    import pyarrow.parquet as pq

    tab = pq.read_table(FORCINGS_FILE, columns=["site_key", "date"] + FORCING_RAW,
                        filters=[("site_key", "in", list(keys))])
    df = tab.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    out = np.full((len(keys), len(dates), len(FORCING_RAW)), np.nan, dtype="float32")
    pos = {k: i for i, k in enumerate(keys)}
    for k, g in df.groupby("site_key"):
        g = g.set_index("date").reindex(dates)
        out[pos[k]] = g[FORCING_RAW].to_numpy(dtype="float32")
    return out


def _fill_q(q_cfs: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Gaps ≤ 7 d linearly interpolated (in log space); longer gaps → the site's train-block DOY
    mean of log Q (± 15 d window; fallback train-block median, then the site's overall median). Flagged."""
    S, T = q_cfs.shape
    q = q_cfs.astype("float64") * physics.CFS_TO_M3S
    lq = np.log(np.maximum(q, 0) + Q_FLOOR)
    miss = ~np.isfinite(lq)
    train = block_mask(dates, "train") | (dates < pd.Timestamp(TIME_BLOCKS["train"][0]))
    doy = dates.dayofyear.to_numpy() - 1
    out = lq.copy()
    for i in range(S):
        s = pd.Series(lq[i])
        interp = s.interpolate(limit_area="inside").to_numpy()
        isna = ~np.isfinite(lq[i])
        run_id = np.cumsum(~isna)                      # NaN runs share the id of the last valid day
        run_len = pd.Series(isna.astype(int)).groupby(run_id).transform("sum").to_numpy()
        short = isna & (run_len <= 7)                  # only gaps of ≤ 7 days are interpolated, fully
        out[i] = np.where(short, interp, lq[i])
        rem = ~np.isfinite(out[i])
        if not rem.any():
            continue
        ok = np.isfinite(lq[i]) & train
        if ok.sum() >= 365:
            k = np.ones(31)
            s_sum = np.bincount(doy[ok], weights=lq[i, ok], minlength=366)
            s_cnt = np.bincount(doy[ok], minlength=366).astype(float)
            circ = lambda a: np.convolve(np.concatenate([a[-15:], a, a[:15]]), k, "valid")  # noqa: E731
            cnt = circ(s_cnt)
            clim = np.where(cnt >= 10, circ(s_sum) / np.maximum(cnt, 1), np.nan)   # log-space DOY mean
            fill = clim[doy]
            fill = np.where(np.isfinite(fill), fill, np.median(lq[i, ok]))
        else:
            # < 365 train-period discharge days: train-period median if ≥ 30 days, else the median of the
            # available record (discharge is an observed INPUT, never a target; documented, review W2)
            allok = np.isfinite(lq[i])
            med = np.median(lq[i, ok]) if ok.sum() >= 30 else (np.median(lq[i, allok]) if allok.any() else 0.0)
            fill = np.full(T, med)
        out[i, rem] = fill[rem]
    return (np.exp(out) - Q_FLOOR).clip(min=0).astype("float32"), miss


def assign_cal(split: SondeSplit, sites: list[str], seed: int = CAL_SEED, frac: float = CAL_FRAC) -> set[str]:
    """Inner calibration sites: ~frac of train-role basin groups (by eligible-site count)."""
    tr = [s for s in sites if split.site_role.get(s) == "train"]
    grp = pd.Series({s: split.basin_group[s] for s in tr})
    size = grp.value_counts()
    rs = np.random.default_rng(seed)
    order = size.index.to_numpy()[rs.permutation(len(size))]
    cum = np.cumsum(size.loc[order].to_numpy()) / size.sum()
    cal_groups = set(order[cum <= frac])
    return {s for s, g in grp.items() if g in cal_groups}


CAL_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "sim_cal_sites.json"


def frozen_cal_sites() -> set[str] | None:
    """The frozen inner-calibration site list (written once by :func:`freeze_cal`; review W7)."""
    if not CAL_FILE.exists():
        return None
    return set(json.loads(CAL_FILE.read_text())["cal_sites"])


def freeze_cal(d: "SimData") -> None:
    if CAL_FILE.exists():
        raise FileExistsError(CAL_FILE)
    rec = {"created": pd.Timestamp.now("UTC").isoformat(), "rule": "data.assign_cal: 15% of train-role basin groups "
           "by eligible-site count, seed 20261025, validation-mode eligibility", "split_sha256": guard.SPLIT_SHA256,
           "cal_sites": sorted(d.sites[i] for i in d.idx("cal")),
           "cal_basin_groups": sorted(set(d.basin_group[d.idx("cal")].tolist()))}
    CAL_FILE.write_text(json.dumps(rec, indent=1))


def build(mode: str = "validation", *, allow_test: bool = False, max_sites: int | None = None,
          site_subset: list[str] | None = None, verbose: bool = True) -> SimData:
    auth = guard.authorise(mode, allow_test=allow_test)
    split = SondeSplit.load(guard.SPLIT_FILE)
    assert_disjoint(train=split.sites("train"), val=split.sites("val"), test=split.sites("test"))
    reg = pd.read_parquet(SITES_FILE)
    reg = reg[(reg["network"] == "USGS") & reg["site_type"].isin(STREAM_TYPES) & reg["in_conus"].astype(bool)]
    reg = reg.assign(site_no=reg["site_id"].astype(str)).set_index("site_no")
    cand = [s for s in sorted(split.site_role) if split.site_role[s] in auth["roles"] and s in reg.index]
    if site_subset is not None:
        keep = set(site_subset)
        cand = [s for s in cand if s in keep]
    dates = pd.date_range(DATA_START, auth["data_end"], freq="D")
    # quick pre-filter from the daily index (full-record counts; exact block counts below)
    ix = pd.read_parquet(paths.PROCESSED / "sonde_daily" / "index.parquet")
    nd = ix.pivot_table(index="site_no", columns="var", values="n_days", aggfunc="sum")
    nd = nd.reindex(cand).fillna(0)
    has_t = nd[[v for v in TARGETS if v in nd.columns]].max(axis=1) >= MIN_TARGET_DAYS
    has_q = nd.get(Q_VAR, pd.Series(0, index=nd.index)) >= MIN_Q_DAYS
    cand = [s for s in cand if has_t.get(s, False) and has_q.get(s, False)]
    if max_sites:
        cand = cand[:max_sites]
    if verbose:
        print(f"[sim.data] {len(cand)} candidate sites after index pre-filter", flush=True)
    cube = load_cube(cand, variables=TARGETS + [Q_VAR], start=DATA_START, end=auth["data_end"])
    y = cube.y[:, :, : len(TARGETS)]
    prov = cube.prov[:, :, : len(TARGETS)]
    qcfs = cube.y[:, :, len(TARGETS)]
    # exact eligibility in each site's own role block
    keep = []
    for i, s in enumerate(cand):
        blk = "train" if split.site_role[s] == "train" else split.site_role[s]
        m = block_mask(dates, blk)
        if np.isfinite(qcfs[i, m]).sum() >= MIN_Q_DAYS and \
                (np.isfinite(y[i, m]).sum(axis=0) >= MIN_TARGET_DAYS).any():
            keep.append(i)
    keep = np.asarray(keep, int)
    sites = [cand[i] for i in keep]
    y, prov, qcfs = y[keep], prov[keep], qcfs[keep]
    keys = [f"USGS-{s}" for s in sites]
    forcing = load_forcings(keys, dates)
    fok = np.isfinite(forcing[..., 0]).mean(axis=1) >= 0.95
    sites = [s for s, k in zip(sites, fok) if k]
    keys = [f"USGS-{s}" for s in sites]
    y, prov, qcfs, forcing = y[fok], prov[fok], qcfs[fok], forcing[fok]
    # forcing gaps (rare): forward fill then back fill along time
    for j in range(forcing.shape[-1]):
        f = pd.DataFrame(forcing[..., j].T).ffill().bfill().to_numpy().T
        forcing[..., j] = f
    q_m3s, q_miss = _fill_q(qcfs, dates)
    r = reg.loc[sites]
    lat = r["lat"].to_numpy(float)
    lon = r["lon"].to_numpy(float)
    elev = r["elev_m"].to_numpy(float)
    scs = _streamcat_static(keys)
    da = scs["nhd_totdasqkm"].to_numpy(float)
    da = np.where(np.isfinite(da), da, r["drain_area_km2"].to_numpy(float))
    slope = scs["nhd_slope"].to_numpy(float)
    slope_f = np.where(np.isfinite(slope) & (slope > 0), slope, 1e-3)
    st = np.column_stack([lat, lon, elev, np.log(np.maximum(da, 0.1)), np.log(np.maximum(slope_f, 1e-5))]
                         + [scs[c].to_numpy(float) for c in STREAMCAT_STATIC]).astype("float32")
    cal = frozen_cal_sites()
    if cal is None:
        cal = assign_cal(split, sites)
    cal = cal & set(sites)
    role = np.array(["cal" if s in cal else split.site_role[s] for s in sites])
    bg = np.array([split.basin_group[s] for s in sites])
    d = SimData(sites=sites, role=role, basin_group=bg, dates=dates, forcing=forcing, q_m3s=q_m3s,
                q_missing=q_miss, y=y, prov=prov, static_raw=st, lat=lat, lon=lon,
                elev=np.where(np.isfinite(elev), elev, 0.0), slope=slope_f,
                pressure=physics.pressure_atm_from_elevation(elev).astype("float32"), authorised=auth)
    assert_disjoint(train=d.idx("train"), cal=d.idx("cal"), val=d.idx("val"))
    assert not np.any(d.role == "test") or auth["mode"] == "test"
    fit_norm(d)
    if verbose:
        print(f"[sim.data] built {d.provenance()}", flush=True)
    return d


# ----------------------------------------------------------------------------- normalisation

def fit_norm(d: SimData, days: np.ndarray | None = None) -> dict:
    """Means/stds from train-role (non-cal) sites × train-block dates only (or the subset ``days``,
    e.g. the cool years of the DSST experiment)."""
    tr = d.idx("train")
    tm = block_mask(d.dates, "train") if days is None else (np.asarray(days, bool) & block_mask(d.dates, "train"))
    dyn = dynamic_features(d.forcing[tr][:, tm], d.q_m3s[tr][:, tm], d.q_missing[tr][:, tm], d.dates[tm], d.lat[tr])
    dm = np.nanmean(dyn.reshape(-1, dyn.shape[-1]), axis=0)
    ds = np.nanstd(dyn.reshape(-1, dyn.shape[-1]), axis=0)
    dm[DYN_NAMES.index("q_missing")], ds[DYN_NAMES.index("q_missing")] = 0.0, 1.0
    sm = np.nanmean(d.static_raw[tr], axis=0)
    ss = np.nanstd(d.static_raw[tr], axis=0)
    ym, ys = [], []
    for j, v in enumerate(TARGETS):
        z = to_model(v, d.y[tr][:, tm, j])
        ym.append(float(np.nanmean(z)))
        ys.append(float(np.nanstd(z)))
    d.norm = {"dyn_mean": dm.astype("float32"), "dyn_std": np.maximum(ds, 1e-6).astype("float32"),
              "st_mean": sm.astype("float32"), "st_std": np.maximum(np.nan_to_num(ss, nan=1.0), 1e-6).astype("float32"),
              "y_mean": np.array(ym, "float32"), "y_std": np.maximum(np.array(ys, "float32"), 1e-6),
              "fit_on": "train-role non-cal sites x train block" + ("" if days is None else " (subset of days)")}
    return d.norm


def normalised_static(d: SimData) -> np.ndarray:
    z = (d.static_raw - d.norm["st_mean"]) / d.norm["st_std"]
    return np.nan_to_num(z, nan=0.0).astype("float32")


def normalised_dynamic(d: SimData, idx, t0: int, t1: int, *, forcing=None, q_m3s=None) -> np.ndarray:
    f = d.forcing[idx, t0:t1] if forcing is None else forcing
    q = d.q_m3s[idx, t0:t1] if q_m3s is None else q_m3s
    dyn = dynamic_features(f, q, d.q_missing[idx, t0:t1], d.dates[t0:t1], d.lat[idx])
    return np.nan_to_num((dyn - d.norm["dyn_mean"]) / d.norm["dyn_std"], nan=0.0).astype("float32")
