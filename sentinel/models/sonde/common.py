# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared experiment setup for every SENTINEL-Sonde model (GBM, Chronos-2, LSTM, event head).

* **Split.** The frozen DEV split ``docs/preregistration/sonde_split_dev.json`` (written by the
  coordinator). :func:`make_temp_split` builds a *temporary* split for smoke tests only; it is
  written under ``$SENTINEL_RUNS/sonde/smoke/`` and its notes say so — never use it for results.
* **Data.** :func:`prepare_data` authorises the requested time blocks / site roles with
  :mod:`.guard` and loads the cube only up to the end of the last authorised block (so WY2024+
  values are not even in memory during development). Climatologies come from
  :func:`.data.climatology` (training years only).
* **Common rows.** Every model forecasts the *same* issue rows: for variable v and block b the
  candidate rows are sites × issue dates (``SondeSplit.time_mask`` for h = 1, i.e. 42-day embargo
  after the block start) where the baselines' issue condition holds (recent observation ≤ 3 d,
  finite climatology). The scoring mask for horizon h is exactly
  ``SondeBaselines.scorable(issue_dates(h), h)`` mapped onto those rows (minus the rare rows where
  a naive baseline is itself undefined, which are dropped for every model). Models may read any
  history *before* the issue date as input (including earlier blocks); only issue dates and
  targets are constrained to the block.
* **Forecast files.** ``q`` of shape (n_rows, n_horizons, 19) in model space, saved as .npz with
  the row keys, so scoring/ensembling/event heads never re-run a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data.sonde_splits import TIME_BLOCKS, SondeSplit, make_split
from ...eval.splits import assert_disjoint
from ..baselines.sonde_baselines import QUANTILES, SondeBaselines
from . import guard
from .data import SITE_DIR, TARGETS, Cube, climatology, issue_dates, load_cube

HORIZONS = (1, 2, 3, 5, 7, 10, 14)
FINAL_SPLIT = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json"   # default
DEV_SPLIT = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split_dev.json"   # smoke tests only
SITE_REGISTRY = paths.PROCESSED / "sites" / "sites.parquet"
SONDE_INDEX = paths.PROCESSED / "sonde_daily" / "index.parquet"
TEMP_SPLIT_NOTE = "TEMPORARY smoke-test split (partial download, no NLDI edges) - NOT FOR RESULTS"
HAZARD_VARS = ("do_mgl_min", "no3no2_mgl_mean", "chl_ugl_mean")


# ----------------------------------------------------------------------------- sites & split

def site_registry() -> pd.DataFrame:
    """USGS sites from the SENTINEL site registry, indexed by site number (string)."""
    s = pd.read_parquet(SITE_REGISTRY)
    s = s[s["network"] == "USGS"].copy()
    s["site_no"] = s["site_id"].astype(str)
    s = s.drop_duplicates("site_no").set_index("site_no")
    out = pd.DataFrame(index=s.index)
    out["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(s["lon"], errors="coerce")
    out["huc2"] = s["huc2"].astype("string").fillna("")
    out["huc_cd"] = s["huc8"].astype("string").fillna("")
    out["elev_m"] = pd.to_numeric(s["elev_m"], errors="coerce")
    out["log_drain_area_km2"] = np.log10(pd.to_numeric(s["drain_area_km2"], errors="coerce").clip(lower=1e-3))
    out["site_type"] = s["site_type"].astype(str)
    return out


def _retry(fn, retries: int = 8, wait_s: float = 15.0):
    """Call ``fn`` with retries (the daily builder rewrites its outputs in place)."""
    import time

    for k in range(retries + 1):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if k == retries:
                raise
            time.sleep(wait_s)


def built_sites() -> list[str]:
    return sorted(p.stem for p in SITE_DIR.glob("*.parquet"))


def wait_for_sites(sites, *, retries: int = 8, wait_s: float = 15.0) -> list[str]:
    """Sites whose daily table is still missing after waiting (the builder may be rewriting the
    directory); callers record them explicitly instead of dropping them silently."""
    import time

    missing = [s for s in sites if not (SITE_DIR / f"{s}.parquet").exists()]
    for _ in range(retries):
        if not missing:
            break
        time.sleep(wait_s)
        missing = [s for s in missing if not (SITE_DIR / f"{s}.parquet").exists()]
    return missing


def load_split(path: str | Path | None = None) -> tuple[SondeSplit, dict]:
    """Load and check a split. Only ``docs/preregistration/sonde_split.json`` (the FINAL split) is
    ``final``; the DEV split and temporary splits are for smoke tests and are never registrable."""
    p = Path(path) if path else FINAL_SPLIT
    if not p.is_file():
        raise FileNotFoundError(f"split file {p} not found (FINAL split pending; for smoke tests pass "
                                f"--split {DEV_SPLIT.relative_to(paths.REPO_ROOT)} or --temp-split)")
    sp = SondeSplit.load(p)
    assert_disjoint(train=sp.sites("train"), val=sp.sites("val"), test=sp.sites("test"))
    notes = sp.notes or ""
    temporary = (TEMP_SPLIT_NOTE in notes or "DEV" in notes or "smoke" in notes.lower()
                 or p.resolve() != FINAL_SPLIT.resolve())
    info = {"path": str(p), "sha256": sp.sha256(), "temporary": bool(temporary),
            "final": not temporary, "notes": notes,
            "n_train": len(sp.sites("train")), "n_val": len(sp.sites("val")), "n_test": len(sp.sites("test"))}
    return sp, info


def make_temp_split(out_path: str | Path, *, site_types=("ST", "ST-TS", "ST-CA", "ST-DCH"),
                    seed: int = 20261022) -> tuple[SondeSplit, dict]:
    """Temporary HUC4-grouped split over the sites built so far (smoke tests only)."""
    reg = site_registry()
    sites = [s for s in built_sites() if s in reg.index]
    meta = reg.loc[sites]
    if site_types:
        meta = meta[meta["site_type"].isin(site_types)]
    df = meta.reset_index()[["site_no", "huc_cd"]]
    sp = make_split(df, None, seed=seed, notes=TEMP_SPLIT_NOTE)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sp.to_json())
    return load_split(out)


def smoke_sites(split: SondeSplit, *, n_train: int = 30, n_val: int = 10, seed: int = 0) -> dict[str, list[str]]:
    """Pick a small smoke-test site set, prioritising sites with the EW hazard variables so that
    every code path (DO min, nitrate, chlorophyll) is exercised."""
    idx = _retry(lambda: pd.read_parquet(SONDE_INDEX))
    idx = idx[idx["var"].isin(TARGETS)]
    first = pd.to_datetime(idx["first"])
    idx = idx[(first <= pd.Timestamp(TIME_BLOCKS["train"][1]) - pd.Timedelta(days=730)) & (idx["n_days"] >= 730)]
    score = idx.assign(w=idx["var"].map({"chl_ugl_mean": 100, "no3no2_mgl_mean": 30, "do_mgl_min": 10})
                       .fillna(1)).groupby("site_no")["w"].sum()
    rs = np.random.default_rng(seed)
    out = {}
    for role, n in (("train", n_train), ("val", n_val)):
        cand = [s for s in split.sites(role) if s in score.index]
        sc = score.loc[cand] + rs.uniform(0, 0.5, len(cand))
        out[role] = sorted(sc.sort_values(ascending=False).index[:n].tolist())
    return out


# ----------------------------------------------------------------------------- data

@dataclass
class Rows:
    """Common forecast rows for one (variable, block): every model predicts these."""

    var: str
    block: str
    si: np.ndarray           # site index into the cube (int32)
    ti: np.ndarray           # issue-date index into cube.dates (int32)
    site_no: np.ndarray      # str
    issue_date: np.ndarray   # datetime64[ns]
    role: np.ndarray         # spatial role of the site
    score_mask: dict = field(default_factory=dict)  # h -> bool (n,)
    obs: dict = field(default_factory=dict)         # h -> float64 (n,) model space
    n_dropped_baseline_undefined: dict = field(default_factory=dict)  # h -> int
    n_dropped_provisional: dict = field(default_factory=dict)         # h -> int (approved-only targets)

    def __len__(self) -> int:
        return int(self.si.size)


@dataclass
class SondeData:
    cube: Cube
    clim_mean: np.ndarray
    clim_std: np.ndarray
    roles: np.ndarray            # (S,) spatial role per site
    static: pd.DataFrame         # indexed like cube.sites
    split: SondeSplit
    split_info: dict
    access: dict
    blocks: tuple
    horizons: tuple = HORIZONS
    weather: np.ndarray | None = None   # (S, T, W) lagged observed weather (load_weather), optional
    _bl: dict = field(default_factory=dict, repr=False)
    _rows: dict = field(default_factory=dict, repr=False)

    @property
    def sites(self) -> list[str]:
        return self.cube.sites

    def site_idx(self, roles) -> np.ndarray:
        return np.flatnonzero(np.isin(self.roles, list(roles)))

    def issue_idx(self, block: str, h: int) -> np.ndarray:
        if block not in self.blocks:
            raise guard.ProtectedDataError(f"block {block!r} not authorised for this dataset ({self.blocks})")
        return issue_dates(self.split, block, horizon=h, context=0, dates=self.cube.dates)

    def baselines(self, var: str) -> SondeBaselines:
        if var not in self._bl:
            self._bl[var] = SondeBaselines(self.cube, self.clim_mean, self.clim_std, var, self.horizons)
        return self._bl[var]

    def rows(self, var: str, block: str) -> Rows:
        key = (var, block)
        if key not in self._rows:
            self._rows[key] = common_rows(self, var, block)
        return self._rows[key]


def prepare_data(split: SondeSplit, split_info: dict, sites: list[str], *, blocks=("train", "val"),
                 allow_test: bool = False, freeze_file=None, variables=TARGETS,
                 horizons=HORIZONS) -> SondeData:
    """Authorise, load and pre-process the cube for ``sites`` (roles taken from the split)."""
    roles = np.array([split.site_role.get(s, "none") for s in sites])
    if (roles == "none").any():
        raise ValueError(f"{int((roles == 'none').sum())} sites are not in the split")
    access = guard.check_access(blocks, set(roles.tolist()), allow_test=allow_test,
                                freeze_file=freeze_file, split=split)
    end = guard.data_end_date(blocks)
    cube = load_cube(sites, list(variables), start=TIME_BLOCKS["train"][0], end=end, retries=8, wait_s=15.0)
    guard.assert_cube_within(cube.dates, blocks)
    cm, cs = climatology(cube)
    usable = np.isfinite(cm).any(axis=(1, 2))
    unscorable = [s for s, u in zip(cube.sites, usable) if not u]
    if not usable.all():  # sites with < 365 training days for every variable cannot be forecast
        keep = np.flatnonzero(usable)
        cube = Cube([cube.sites[i] for i in keep], cube.dates, cube.variables, cube.y[keep], cube.prov[keep])
        cm, cs, roles = cm[keep], cs[keep], roles[keep]
    reg = site_registry()
    static = reg.reindex(cube.sites)[["lat", "lon", "huc2", "elev_m", "log_drain_area_km2"]]
    access = {**access, "unscorable_sites_no_climatology": unscorable}
    return SondeData(cube, cm, cs, roles, static, split, split_info, access, tuple(blocks), tuple(horizons))


def common_rows(data: SondeData, var: str, block: str) -> Rows:
    b = data.baselines(var)
    grid = data.issue_idx(block, 1)
    S, T = b.z.shape
    ok = np.isfinite(b.last[:, grid]) & np.isfinite(b.cm[:, b.doy[grid]])
    si, gi = np.nonzero(ok)
    ti = grid[gi]
    rows = Rows(var, block, si.astype(np.int32), ti.astype(np.int32), np.asarray(data.sites)[si],
                data.cube.dates.to_numpy()[ti], data.roles[si])
    for h in data.horizons:
        t_h = data.issue_idx(block, h)
        dense = np.zeros((S, T), dtype=bool)
        dense[:, t_h] = b.scorable(t_h, h)
        # rows where a naive baseline is undefined (site with < 60 training pairs at this h) are
        # dropped for EVERY model, so paired comparisons always use identical rows
        base_ok = np.isfinite(b.res_q[h]).all(axis=1) & np.isfinite(b.chg_q[h]).all(axis=1)
        sc = dense[si, ti]
        rows.score_mask[h] = sc & base_ok[si]
        rows.n_dropped_baseline_undefined[h] = int((sc & ~base_ok[si]).sum())
        # Approved values only as targets (protocol §6): provisional target observations are not scored
        tg = np.minimum(ti.astype(np.int64) + h, T - 1)
        prov = data.cube.prov[si, tg, data.cube.var_index(var)] & (ti.astype(np.int64) + h < T)
        rows.n_dropped_provisional[h] = int((rows.score_mask[h] & prov).sum())
        rows.score_mask[h] &= ~prov
        tgt = ti.astype(np.int64) + h
        obs = np.full(si.size, np.nan)
        inb = tgt < T
        obs[inb] = b.z[si[inb], tgt[inb]]
        rows.obs[h] = obs
    return rows


def baseline_forecast(data: SondeData, var: str, rows: Rows, model: str) -> np.ndarray:
    """Naive-baseline quantiles on the common rows, shape (n, H, 19), model space."""
    b = data.baselines(var)
    uniq, inv = np.unique(rows.ti, return_inverse=True)
    out = np.full((len(rows), len(data.horizons), len(QUANTILES)), np.nan, dtype=np.float32)
    for k, h in enumerate(data.horizons):
        f = b.forecast(model, uniq, h)  # (S, n_uniq, Q)
        out[:, k] = f[rows.si, inv]
    return out


# ----------------------------------------------------------------------------- forecast files

def save_forecast(path: str | Path, *, model: str, rows: Rows, q: np.ndarray, horizons=HORIZONS,
                  meta: dict | None = None) -> Path:
    from .evaluate import clip_plausible

    q = clip_plausible(rows.var, q).astype(np.float32)   # identical clip for every model (protocol §6)
    if q.shape != (len(rows), len(horizons), len(QUANTILES)):
        raise ValueError(f"forecast shape {q.shape} != {(len(rows), len(horizons), len(QUANTILES))}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, q=q, site_no=rows.site_no.astype(str), issue_date=rows.issue_date.astype("datetime64[D]"),
                        horizons=np.asarray(horizons), quantiles=QUANTILES,
                        header=json.dumps({"model": model, "var": rows.var, "block": rows.block, **(meta or {})},
                                          default=str))
    return path


def load_forecast(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["header"] = json.loads(str(out["header"]))
    return out


def align_forecast(fc: dict, rows: Rows, horizons=HORIZONS) -> np.ndarray:
    """Reorder a loaded forecast onto ``rows`` (raises if any common row is missing)."""
    if tuple(int(h) for h in fc["horizons"]) != tuple(horizons):
        raise ValueError("forecast horizons differ from the requested horizons")
    key_fc = pd.MultiIndex.from_arrays([fc["site_no"].astype(str), fc["issue_date"].astype("datetime64[ns]")])
    key_rows = pd.MultiIndex.from_arrays([rows.site_no.astype(str), rows.issue_date.astype("datetime64[ns]")])
    pos = key_fc.get_indexer(key_rows)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} common rows missing from forecast {fc['header'].get('model')}")
    return fc["q"][pos]


def scorable_finite_check(q: np.ndarray, rows: Rows, horizons=HORIZONS) -> dict:
    """Count non-finite forecasts on scorable rows (must be 0: rows are identical for all models)."""
    bad = {}
    for k, h in enumerate(horizons):
        m = rows.score_mask[h]
        bad[h] = int((~np.isfinite(q[m, k]).all(axis=-1)).sum())
    return bad


# ----------------------------------------------------------------------------- weather (past observed only)

FORCINGS = paths.PROCESSED / "forcings_daily" / "forcings_daily.parquet"
WEATHER_VARS = ("precip_mm", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "vpd_kpa")
WEATHER_LAG_DAYS = 1   # gridMET day d is published after day d ends; known at the end of day d+1


def load_weather(cube: Cube, *, lag_days: int = WEATHER_LAG_DAYS) -> np.ndarray:
    """(S, T, W) OBSERVED daily weather aligned so that index t holds weather of day t − lag_days,
    i.e. only values already published at the end of issue day t (never same-day or future weather;
    observed weather is never used for target days). NaN where a site has no forcings."""
    import datetime as dt

    import pyarrow.dataset as pds

    keys = [f"USGS-{s}" for s in cube.sites]
    d0 = (cube.dates[0] - pd.Timedelta(days=lag_days)).date()
    d1 = cube.dates[-1].date()
    tb = pds.dataset(FORCINGS).to_table(
        filter=pds.field("site_key").isin(keys) & (pds.field("date") >= d0) & (pds.field("date") <= d1),
        columns=["site_key", "date", *WEATHER_VARS]).to_pandas()
    del dt
    out = np.full((len(cube.sites), len(cube.dates), len(WEATHER_VARS)), np.nan, dtype=np.float32)
    pos = {k: i for i, k in enumerate(keys)}
    tb["date"] = pd.to_datetime(tb["date"]) + pd.Timedelta(days=lag_days)     # day the value becomes known
    tb = tb[tb["date"] <= cube.dates[-1]]
    si = tb["site_key"].map(pos).to_numpy()
    ti = cube.dates.get_indexer(pd.DatetimeIndex(tb["date"]))
    ok = ti >= 0
    for j, v in enumerate(WEATHER_VARS):
        out[si[ok], ti[ok], j] = tb[v].to_numpy(np.float32)[ok]
    return out


# ----------------------------------------------------------------------------- catchment attributes (static)

STREAMCAT = paths.PROCESSED / "catchment" / "streamcat_sites.parquet"
# Pre-specified (domain grounds, 2026-10-25; not selected on validation) watershed-level StreamCat set.
CATCHMENT_ATTRS = ("pctcrop2019ws", "pcthay2019ws", "pctimp2019ws", "pcturbhi2019ws", "pctow2019ws",
                   "fertws", "manurews", "naniws", "pctagdrainagews", "popden2010ws", "wwtpalldensws", "rddensws",
                   "pctcarbresidws", "caows", "na2ows", "clayws", "sandws", "permws", "kffactws", "rckdepws",
                   "bfiws", "runoffws", "precip9120ws", "tmean9120ws", "wetindexws", "elevws", "wsareasqkm")
CATCHMENT_SUMS = {"pctforest2019ws": ("pctdecid2019ws", "pctconif2019ws", "pctmxfst2019ws"),
                  "pctwetland2019ws": ("pctwdwet2019ws", "pcthbwet2019ws")}
CATCHMENT_LOG = ("fertws", "manurews", "naniws", "popden2010ws", "wwtpalldensws", "rddensws", "wsareasqkm")


def load_catchment(sites: list[str]) -> pd.DataFrame:
    """Static StreamCat watershed attributes (columns 'sc_<name>'), log1p for skewed densities/areas;
    NaN where a site has no StreamCat match (tree models handle it; NN statics get a mask)."""
    d = pd.read_parquet(STREAMCAT)
    d = d[d["site_key"].astype(str).str.startswith("USGS-")].copy()
    d["site_no"] = d["site_key"].astype(str).str[5:]
    d = d.drop_duplicates("site_no").set_index("site_no")
    out = pd.DataFrame(index=pd.Index(sites, name="site_no"))
    for c in CATCHMENT_ATTRS:
        x = pd.to_numeric(d[c], errors="coerce").reindex(sites)
        out[f"sc_{c}"] = np.log1p(x.clip(lower=0)) if c in CATCHMENT_LOG else x
    for name, parts in CATCHMENT_SUMS.items():
        out[f"sc_{name}"] = sum(pd.to_numeric(d[p], errors="coerce").reindex(sites) for p in parts)
    return out.astype("float64")


# ----------------------------------------------------------------------------- H2 chloride (derived series)

CHLORIDE_DIR = paths.PROCESSED / "chloride_derived"
CHLORIDE_VARS = ("cl_mgl_mean", "cl_mgl_4dmean", "cl_mgl_max")


def chloride_regressions() -> pd.DataFrame:
    r = pd.read_parquet(CHLORIDE_DIR / "regressions.parquet")
    r["site_no"] = r["site_no"].astype(str)
    return r[r["eligible"].astype(bool)].set_index("site_no")[["a", "b", "r2", "n_pairs"]]


def attach_chloride(data: SondeData) -> SondeData:
    """Append the lead-built derived chloride series (sentinel/eval/chloride.py; fixed training-block
    site regressions Cl = a + b·SC) to the cube as variables CHLORIDE_VARS, with provisional flags
    inherited from SC (daily mean; the 4-day mean is provisional if any of its 4 SC days is) and
    training-years climatology. Sites without an eligible regression stay NaN."""
    from .data import climatology as _clim

    cube = data.cube
    S, T = len(cube.sites), len(cube.dates)
    y = np.full((S, T, len(CHLORIDE_VARS)), np.nan, np.float32)
    for i, s in enumerate(cube.sites):
        p = CHLORIDE_DIR / "sites" / f"{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index)
            d = d.reindex(cube.dates)
            for k, v in enumerate(CHLORIDE_VARS):
                if v in d.columns:
                    y[i, :, k] = d[v].to_numpy(np.float32)
    jsc = cube.var_index("spcond_uscm_mean")
    psc = cube.prov[:, :, jsc]
    p4 = psc.copy()
    for k in range(1, 4):
        p4[:, k:] |= psc[:, :-k]
    prov = np.stack([psc, p4, psc], axis=-1)
    newcube = Cube(cube.sites, cube.dates, cube.variables + list(CHLORIDE_VARS),
                   np.concatenate([cube.y, y], axis=-1), np.concatenate([cube.prov, prov], axis=-1))
    cm_n, cs_n = _clim(Cube(cube.sites, cube.dates, list(CHLORIDE_VARS), y, prov))
    data.cube = newcube
    data.clim_mean = np.concatenate([data.clim_mean, cm_n], axis=-1)
    data.clim_std = np.concatenate([data.clim_std, cs_n], axis=-1)
    data._bl.clear()
    data._rows.clear()
    return data
