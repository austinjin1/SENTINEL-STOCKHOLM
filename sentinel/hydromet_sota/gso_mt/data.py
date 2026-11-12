# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Training data of the M6 architecture-v2 round (amendment §2–§3): one pinned pool for every arm and reference.

:func:`load_v2_train` wraps the lever engineer's :func:`sentinel.hydromet_sota.data.load_lever_train` (FINAL
train-role sonde sites + lever ``fit`` USGS sites + Tier 1 grab stations; **train-role locations × train block only**)
and adds what the arch-v2 arms need:

* instantaneous grab rows with their local solar hour (replacing the lever's daily-mean weak labels) and the inner
  grab set (stations with rule ``inner_group``; never fitted in an early-stopped run);
* auxiliary daily labels (T max, T min, DO max, log discharge) and the dense hourly sonde table;
* the 4 solar-geometry channels and the 3 NHDPlus VAA statics (US input set v2).

No validation- or test-block row and no val/test-role location can be loaded through this module.

Lead rulings of the stamp (``docs/preregistration/hydromet_arch_v2_lead_rulings.md`` §2.3) implemented here:

* **Pins.** ``configs/hydromet_sota/v2/pins.json`` (written once by ``scripts/data/sota_hydromet_v2_pins.py``, 0444)
  holds the sha256 of every file of :func:`pin_targets`, a digest of the daily site tables of the pool and the
  fingerprint of the pool as loaded; :func:`verify_pins` re-checks all of it and the loader compares the fingerprint.
* **Date filter inside the file scan** for every reader of a US label table: :func:`scan_cube` replaces the frozen
  ``sonde.data.load_cube`` (which reads a site table whole and then re-indexes) for the duration of the call into the
  lever loader, with identical output; the grab table is read with a row-group / row filter on ``date``; the hourly
  table was built with the filter in its scan and is the pinned table. The lever's daily-mean weak-label table
  (``wqp_train_obs.parquet``) is not read at all.
* **StreamPULSE**: M6's registration is the empty set — :func:`assert_no_streampulse`.
* **Forcing-only keys** of ``upstream_forcing_sota.parquet`` (lead relay 2026-11-10: 445 of its 447 keys are fit
  sites; 2 rule-1b ``reserved_comid`` sites carry forcing rows only): :func:`assert_no_forcing_only_site` on every
  fit, validation and scoring frame."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import stat

import numpy as np
import pandas as pd

from ...core import paths
from ...data.reserved import assert_no_reserved as shared_assert_no_reserved
from ...models.hydromet import data as D
from ...models.hydromet import guard
from ...models.hydromet.lstm import build_inputs
from .. import data as LD
from .. import reserved as R
from .. import sites as LS
from .grab import PROVIDER_CLASSES
from .solar import SOLAR_CHANNELS, solar_channels

V2_DIR = paths.PROCESSED / "hydromet_sota_v2"
PINS_FILE = paths.REPO_ROOT / "configs" / "hydromet_sota" / "v2" / "pins.json"
PINNED = ("grab_inst.parquet", "grab_logger_daily.parquet", "grab_stations.parquet", "vaa_static.parquet",
          "hourly_train.npz")
DAILY = ("T_mean", "T_max", "T_min", "T_7dadm", "DO_mean", "DO_min", "DO_max", "logq", "sc", "turb", "no3")
PRIMARY = {"temp_c_mean": "T_mean", "temp_c_7dadm": "T_7dadm", "do_mgl_mean": "DO_mean", "do_mgl_min": "DO_min"}
AUX_WQ = {"sc": "spcond_uscm_mean", "turb": "turb_fnu_mean", "no3": "no3no2_mgl_mean"}
VAA_COLS = ("vaa_pathtimema", "vaa_streamorde", "vaa_lengthkm")
VAA_LOG = ("vaa_pathtimema", "vaa_lengthkm")
INST = ("T", "DO")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


STREAMPULSE_TOKEN = "streampulse"
POOL_SOURCES = ("usgs_dv", "wqp_grab")        # the only label sources of the arch-v2 pool (Tier 1 only)
N_FORCING_ONLY = 2                            # keys of upstream_forcing_sota.parquet that are not fit sites
CODE_PINS = ("sentinel/hydromet_sota/sites.py", "sentinel/hydromet_sota/reserved.py", "sentinel/data/reserved.py",
             "sentinel/hydromet_sota/gso_mt/grab.py", "sentinel/hydromet_sota/gso_mt/data.py",
             "sentinel/hydromet_sota/gso_mt/solar.py", "sentinel/hydromet_sota/gso_mt/constants.py",
             "sentinel/hydromet_sota/thermox/physics.py", "scripts/data/sota_hydromet_v2_build.py")
CONFIG_REQUIRED = ("common.yaml", "d1.yaml", "d2.yaml", "rtd.yaml", "rtd_w.yaml", "gbm.yaml", "physics.yaml",
                   "stack.yaml", "schema_variant.yaml")


def pin_targets() -> dict:
    """Label → path of every single file under a pin (lead rulings §2.3 PIN): every pool table (labels, stations,
    forcings, statics), ``lever_sites.parquet``, the FINAL split, the track-N role table, the sonde-tier site list
    (the registry of sonde sites and the reserved new-site list), ``sites.py``, both reserved modules, ``grab.py``,
    this module, the v2 build script and every file of ``configs/hydromet_sota/v2/`` (``pins.json`` itself excepted).
    The daily site tables of the pool (one file per site) are covered by :func:`site_tables_digest`."""
    P, RR = paths.PROCESSED, paths.REPO_ROOT
    t = {f"processed/hydromet_sota_v2/{f}": V2_DIR / f for f in PINNED + ("build_report.json",)}
    t.update({
        "processed/hydromet_sota/lever_sites.parquet": LD.SOTA_DIR / "lever_sites.parquet",
        "processed/hydromet_upstream_forcing/upstream_forcing_sota.parquet": LD.UPSTREAM_SOTA,
        "processed/hydromet_upstream_forcing/upstream_forcing_trackn.parquet": LD.UPSTREAM_TRACKN,
        "processed/hydromet_upstream_forcing/upstream_forcing.parquet": D.UPSTREAM_FILE,
        "processed/hydromet_wqp_national/forcings.parquet": LD.WQP_DIR / "forcings.parquet",
        "processed/hydromet_wqp_national/stations.parquet": LD.WQP_DIR / "stations.parquet",
        "processed/hydromet_wqp_national/streamcat_comid.parquet": LD.WQP_DIR / "streamcat_comid.parquet",
        "processed/forcings_daily/forcings_daily.parquet": D.FORCINGS_FILE,
        "processed/catchment/streamcat_sites.parquet": D.STREAMCAT_FILE,
        "processed/sites/sites.parquet": D.SITES_FILE,
        "docs/preregistration/sonde_split.json": D.FINAL_SPLIT,
        "docs/preregistration/hydromet_track_n_stations.json": LS.TRACKN_FILE,
        "docs/preregistration/sonde_newsites_list.json": R.NEWSITES_FILE,
        "runs/hydromet/ealstm_ws_v1_s0/data_info.json": LS.REF_DATA_INFO,
    })
    t.update({c: RR / c for c in CODE_PINS})
    cfg = RR / "configs" / "hydromet_sota" / "v2"
    names = sorted(f.name for f in cfg.iterdir() if f.is_file() and f.name != PINS_FILE.name)
    missing = [c for c in CONFIG_REQUIRED if c not in names]
    if missing:
        raise RuntimeError(f"configs/hydromet_sota/v2 lacks {missing}: every arm's configuration is pinned together")
    t.update({f"configs/hydromet_sota/v2/{n}": cfg / n for n in names})
    return t


def pin_values() -> dict:
    """Values pinned by content rather than by file. The T-E static set is ``ea_data.COMMON_STATIC`` (lead rulings
    §2.4): the tuple is pinned here; the file that holds it is the lever engineer's and still changes while the
    England build runs (T-E is on hold), so its file sha256 is recorded in ``pins.json`` for information only."""
    from .. import ea_data as E

    cs = [str(c) for c in E.COMMON_STATIC]
    return {"ea_common_static_sha256": hashlib.sha256("\n".join(cs).encode()).hexdigest(), "n_common_static": len(cs)}


def site_tables_digest(usgs_sites) -> dict:
    """One digest for the daily site tables (``processed/sonde_daily/sites/<site>.parquet``) of the pool's USGS
    sites: sha256 over the per-file sha256 in the given site order. Counts and hashes only."""
    from ...models.sonde.data import SITE_DIR

    shared_assert_no_reserved([str(x) for x in usgs_sites], where="M6 arch-v2 site-table digest")
    h, n, nbytes = hashlib.sha256(), 0, 0
    for s_ in usgs_sites:
        f = SITE_DIR / f"{s_}.parquet"
        if f.is_file():
            h.update(sha256_file(f).encode())
            n += 1
            nbytes += f.stat().st_size
        else:
            h.update(b"missing")
    return {"sha256": h.hexdigest(), "n_sites": len(list(usgs_sites)), "n_files": n, "bytes": nbytes,
            "tier": os.path.realpath(SITE_DIR.parent).rsplit("/", 1)[-1]}


def pool_fingerprint(v2: "V2Train") -> dict:
    """sha256 of the pool as every arm sees it: location list and index sets, labels, forcings, statics, solar
    channels and VAA statics. Compared with ``pins.json`` at every non-smoke load."""
    def hb(*arrs):
        h = hashlib.sha256()
        for a_ in arrs:
            h.update(np.ascontiguousarray(a_).tobytes())
        return h.hexdigest()

    d = v2.data
    return {"sites": hashlib.sha256("\n".join(map(str, d.sites)).encode()).hexdigest(),
            "groups": hashlib.sha256("\n".join(map(str, d.basin_group)).encode()).hexdigest(),
            "index_sets": hb(*(np.asarray(x, "int64") for x in (v2.fit_idx, v2.inner_idx, v2.final_fit_idx,
                                                               v2.grab_fit_idx, v2.grab_inner_idx, v2.hourly_row,
                                                               v2.prov_class))),
            "daily": hb(v2.daily), "inst": hb(v2.inst_val, v2.inst_hour), "hourly": hb(v2.hourly, v2.hour_frac),
            "forc": hb(d.forc), "static": hb(d.static.to_numpy(dtype="float64")),
            "static_columns": hashlib.sha256("\n".join(map(str, d.static.columns)).encode()).hexdigest(),
            "solar": hb(v2.solar), "vaa": hb(v2.vaa.to_numpy(dtype="float64")),
            "shape": {"locations": len(d.sites), "days": len(d.dates), "forcing_channels": int(d.forc.shape[-1]),
                      "statics": int(d.static.shape[1]) + int(v2.vaa.shape[1])}}


def verify_pins() -> dict:
    """Every file of :func:`pin_targets` must match ``pins.json`` (which must be read-only and must cover every
    target); a missing pin file means the pool is not pinned yet. Called by every non-smoke entry point through
    ``runner.assert_released`` and by :func:`load_v2_train`."""
    if not PINS_FILE.is_file():
        raise RuntimeError(f"{PINS_FILE} is missing: the arch-v2 pool is not pinned (lead rulings §2.3)")
    if stat.S_IMODE(os.stat(PINS_FILE).st_mode) & 0o222:
        raise RuntimeError(f"{PINS_FILE.name} must be read-only (0444): it is written once")
    pins = json.loads(PINS_FILE.read_text())
    files = pins["files"]
    targets = pin_targets()
    lost = sorted(set(targets) - set(files)) + sorted(f"unknown pin {k}" for k in set(files) - set(targets))
    if lost:
        raise RuntimeError(f"{PINS_FILE.name} does not cover {lost[:5]} ({len(lost)} in all)")
    changed = {}
    for label, path in targets.items():
        h = sha256_file(path)
        if h != files[label]:
            if not pin_change_stamped(label, h):
                raise RuntimeError(f"{label}: sha256 {h[:12]} differs from its pin ({PINS_FILE.name}); a change to a "
                                   "pinned file needs a stamped deviation")
            changed[label] = h
    if pins.get("values") != pin_values():
        vs = hashlib.sha256(json.dumps(pin_values(), sort_keys=True).encode()).hexdigest()
        if not pin_change_stamped("configs/hydromet_sota/v2/pins.json#values", vs):
            raise RuntimeError(f"a value pinned by content differs from {PINS_FILE.name}: {sorted(pin_values())}")
        changed["values"] = vs
    return {**pins, "changed_under_stamp": changed}


PIN_CHANGE_PREFIXES = ("sentinel/", "scripts/", "configs/")


def pin_change_line(label: str, sha: str) -> str:
    return f"PIN CHANGE: {label} sha256 {sha}"


def pin_change_stamped(label: str, sha: str) -> bool:
    """``pins.json`` is written once. "Any later change to a pinned value is a counted configuration and needs a
    stamped deviation" (lead rulings §2.3): a pinned **code or config** file may differ from its pin only if the
    deviations file, in the state the lead has stamped (current sha256 in a stamp row, read-only), carries the line
    ``PIN CHANGE: <label> sha256 <new sha256>``. Tables, the split, role tables and the pool fingerprint have no such
    route: they never change."""
    if not label.startswith(PIN_CHANGE_PREFIXES):
        return False
    from . import runner as VR

    return bool(VR.deviations_status()["released"]) and pin_change_line(label, sha) in VR.DEVIATIONS.read_text().splitlines()


def assert_no_streampulse(pool: pd.DataFrame | None = None) -> int:
    """StreamPULSE registration of M6 in this round: the **empty set** (lead rulings §2.3). No pinned file lies in a
    StreamPULSE directory and no pool location has a source other than :data:`POOL_SOURCES`. Returns the number of
    StreamPULSE rows the pool can read: 0."""
    bad = [k for k, v in pin_targets().items() if STREAMPULSE_TOKEN in (k + str(v)).lower()]
    if bad:
        raise RuntimeError(f"a StreamPULSE path among the pool files: {bad}")
    if pool is not None:
        other = set(pool["source"].astype(str)) - set(POOL_SOURCES)
        if other:
            raise RuntimeError(f"pool sources other than {POOL_SOURCES}: {sorted(other)}")
    return 0


def forcing_only_sites() -> set:
    """Site numbers that have rows in ``upstream_forcing_sota.parquet`` without being lever fit sites. Only the key
    column of the table is scanned."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    keys = set(pc.unique(ds.dataset(LD.UPSTREAM_SOTA, format="parquet").to_table(columns=["site_key"])["site_key"])
               .to_pylist())
    pool = pd.read_parquet(LD.SOTA_DIR / "lever_sites.parquet", columns=["key", "site_no", "source", "rule"])
    fit = set(pool.loc[(pool["source"] == "usgs_dv") & (pool["rule"] == "fit"), "key"])
    if not fit <= keys:
        raise RuntimeError(f"{len(fit - keys)} lever fit sites have no row in {LD.UPSTREAM_SOTA.name}")
    extra = keys - fit
    if len(extra) != N_FORCING_ONLY:
        raise RuntimeError(f"{len(extra)} forcing-only keys in {LD.UPSTREAM_SOTA.name}; the lever engineer's statement "
                           f"of 2026-11-10 names {N_FORCING_ONLY}")
    return {k.split("-", 1)[-1] for k in extra}


def assert_no_forcing_only_site(sites, forcing_only=None, where: str = "M6 arch-v2 frame") -> int:
    """Hard failure if a forcing-only site of the lever's upstream table reaches a fit, validation or scoring frame
    (as a site number, a ``USGS-`` key or inside a ``WQP:`` key)."""
    bad = forcing_only_sites() if forcing_only is None else {str(x).split("-", 1)[-1] for x in forcing_only}
    n = sum(any(LS.contains_reserved_id(str(s_), {b}) for b in bad) for s_ in sites)
    if n:
        raise R.ReservedSiteError(f"{where}: {n} forcing-only sites of the lever upstream table in the frame")
    return len(bad)


def assert_dates_within(dates, end, where: str = "M6 arch-v2 frame") -> str:
    """The last date of a frame must not lie after its stamped date filter (lead relay 2026-11-10, caveat b: values
    of the lever upstream table after 2026-09-09 are provisional gridMET — asserted, not assumed, to be unused)."""
    last = pd.Timestamp(pd.DatetimeIndex(dates).max())
    if last > pd.Timestamp(end):
        raise guard.ProtectedDataError(f"{where}: last date {last.date()} lies after the date filter {end}")
    return str(last.date())


def scan_cube(sites, variables, start, end, *, retries: int = 0, wait_s: float = 15.0):
    """The frozen ``sentinel.models.sonde.data.load_cube`` with the **date filter inside the parquet scan**: no row
    dated after ``end`` (or before ``start``) is materialised. Same arguments, same output (tested for equality)."""
    import time

    import pyarrow.parquet as pq

    from ...models.sonde.data import SITE_DIR, Cube

    shared_assert_no_reserved([str(x) for x in sites], where="M6 arch-v2 daily site tables")
    dates = pd.date_range(start, end, freq="D")
    lo, hi = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    S, T, V = len(sites), len(dates), len(variables)
    y = np.full((S, T, V), np.nan, dtype="float32")
    prov = np.zeros((S, T, V), dtype=bool)
    for i, s_ in enumerate(sites):
        f = SITE_DIR / f"{s_}.parquet"
        df = None
        for k in range(retries + 1):
            try:
                if f.exists():
                    names = set(pq.read_schema(f).names)
                    cols = [c for v in variables for c in (v, f"{v}_prov") if c in names]
                    df = pd.read_parquet(f, columns=cols, filters=[("date", ">=", lo), ("date", "<=", hi)])
                    break
            except Exception:  # noqa: BLE001 - partially written file during a rebuild (as the frozen reader)
                if k == retries:
                    raise
            if k < retries:
                time.sleep(wait_s)
        if df is None:
            continue
        df.index = pd.to_datetime(df.index)
        if len(df) and (df.index.max() > hi or df.index.min() < lo):
            raise guard.ProtectedDataError("a row outside the requested dates passed the scan filter")
        df = df.reindex(dates)
        for j, v in enumerate(variables):
            if v in df.columns:
                y[i, :, j] = df[v].to_numpy(dtype="float32")
                if f"{v}_prov" in df.columns:
                    prov[i, :, j] = df[f"{v}_prov"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return Cube(list(sites), dates, list(variables), y, prov)


@contextlib.contextmanager
def cube_scan_filter():
    """While active, ``sentinel.models.hydromet.data.load_cube`` — the name through which the lever loader reads the
    daily site tables — is :func:`scan_cube`. No file of the frozen or lever code is changed; the substitution lives
    for one call and is undone on exit."""
    orig = D.load_cube
    D.load_cube = scan_cube
    try:
        yield
    finally:
        D.load_cube = orig


def read_grab_inst(path=None) -> pd.DataFrame:
    """The instantaneous grab table with the train-block filter inside the scan. The table was built from train-block
    rows only, so the filter must not drop a row: a difference from the file's row count is an error."""
    import pyarrow.parquet as pq

    f = path or (V2_DIR / "grab_inst.parquet")
    lo, hi = pd.Timestamp(LS.TRAIN_START), pd.Timestamp(LS.TRAIN_END)
    inst = pd.read_parquet(f, filters=[("date", ">=", lo), ("date", "<=", hi)])
    if len(inst) != pq.ParquetFile(f).metadata.num_rows:
        raise guard.ProtectedDataError(f"{getattr(f, 'name', f)} holds rows outside the train block")
    if len(inst) and (inst["date"].max() > hi or inst["date"].min() < lo):
        raise guard.ProtectedDataError("a grab row outside the train block passed the scan filter")
    if inst.duplicated(["station_key", "date", "var"]).any():
        raise RuntimeError("duplicated grab rows")
    return inst


@dataclasses.dataclass
class V2Train:
    data: D.HydroMetData          # every location of the pool (roles all "train"), dates DATA_START..train-block end
    fit_idx: np.ndarray           # dense fit sites (FINAL train-role minus inner, + lever fit USGS sites)
    inner_idx: np.ndarray         # the 256 inner sonde sites
    final_fit_idx: np.ndarray     # FINAL train-role fit sites (normalisation statistics; as the reference runs)
    grab_fit_idx: np.ndarray
    grab_inner_idx: np.ndarray
    daily: np.ndarray             # (S, T, len(DAILY)); raw units, log space for logq / sc / turb / no3; NaN missing
    inst_val: np.ndarray          # (S, T, 2) instantaneous T / DO (grab rows)
    inst_hour: np.ndarray         # (S, T, 2) local solar hour; NaN = no time
    hourly: np.ndarray            # (Sh, T, 24, 2) sonde hourly means by local-solar-hour bin
    hourly_row: np.ndarray        # (S,) row into ``hourly`` or -1
    hour_frac: np.ndarray         # (Sh,) fractional bin centre
    prov_class: np.ndarray        # (S,) index into PROVIDER_CLASSES or -1
    solar: np.ndarray             # (S, T, 4)
    vaa: pd.DataFrame             # (S, 3)
    info: dict

    def subset(self, keep) -> "V2Train":
        """A copy restricted to the locations ``keep`` (indices); used by smoke runs and tests."""
        keep = np.asarray(keep, int)
        new = {int(o): i for i, o in enumerate(keep)}
        m = lambda idx: np.array([new[int(i)] for i in idx if int(i) in new], int)   # noqa: E731
        d = self.data
        hr = self.hourly_row[keep]
        used = np.unique(hr[hr >= 0])
        hmap = {int(o): i for i, o in enumerate(used)}
        data = dataclasses.replace(d, sites=[d.sites[i] for i in keep], roles=d.roles[keep],
                                   basin_group=d.basin_group[keep], forc=d.forc[keep], y=d.y[keep], prov=d.prov[keep],
                                   static=d.static.iloc[keep], info={**d.info, "n_sites": len(keep)})
        return V2Train(data, m(self.fit_idx), m(self.inner_idx), m(self.final_fit_idx), m(self.grab_fit_idx),
                       m(self.grab_inner_idx), self.daily[keep], self.inst_val[keep], self.inst_hour[keep],
                       self.hourly[used], np.array([hmap[int(h)] if h >= 0 else -1 for h in hr], int),
                       self.hour_frac[used], self.prov_class[keep], self.solar[keep], self.vaa.iloc[keep],
                       {**self.info, "subset_of": int(len(d.sites))})


def load_v2_train(*, grab: bool = True, require_pins: bool = True, new_usgs: bool = True, log=print) -> V2Train:
    """``new_usgs=False`` (smoke runs only; refused with ``require_pins``) leaves out the lever pool's new USGS fit
    sites, whose basin-mean forcing table is the lever engineer's build and may not exist yet."""
    if require_pins and not new_usgs:
        raise ValueError("the pinned pool always includes the new USGS fit sites")
    pins = verify_pins() if require_pins else None
    lever_pool = pd.read_parquet(LD.SOTA_DIR / "lever_sites.parquet")
    assert_no_streampulse(lever_pool)
    with cube_scan_filter():                       # daily site tables: date filter inside the scan
        # weak=False: the lever's daily-mean weak-label table is never read; Tier 1 stations are attached below
        data, fit_idx, inner_idx, _, aux, linfo = LD.load_lever_train(lever1a=new_usgs, q_only_sites=new_usgs,
                                                                      weak=False, aux_q=True, log=log)
    dates, blocks = data.dates, ("train",)
    assert_dates_within(dates, LS.TRAIN_END, where="M6 arch-v2 training frame")
    if dates[-1] != pd.Timestamp(guard.data_end_date(blocks)):
        raise guard.ProtectedDataError("the training frame does not end with the train block")
    n_final, n_usgs = linfo["n_final_train"], linfo["n_final_train"] + linfo["n_new_usgs"]
    if n_usgs != len(data.sites):
        raise RuntimeError("the lever loader returned locations other than USGS sites")
    usgs_sites = list(data.sites[:n_usgs])
    n_forcing_only = assert_no_forcing_only_site(usgs_sites, where="M6 arch-v2 training pool") if new_usgs else None
    final_fit_idx = fit_idx[fit_idx < n_final]
    y = data.y
    weak_idx = grab_inner_idx = np.array([], int)
    stn = None
    if grab:
        stn = pd.read_parquet(V2_DIR / "grab_stations.parquet").set_index("key")
        lp = lever_pool[lever_pool["source"] == "wqp_grab"].set_index("key")
        fk, ik = sorted(stn.index[stn["rule"] == "fit"]), sorted(stn.index[stn["rule"] == "inner_group"])
        if set(fk) != set(lp.index[lp["rule"] == "fit"]) or not set(ik) <= set(lp.index[lp["rule"] == "inner_group"]):
            raise RuntimeError("Tier 1 stations differ from the lever pool's fit / inner-group stations")
        if set(stn["rule"]) - {"fit", "inner_group"}:
            raise RuntimeError("the grab station table holds a rule other than fit / inner_group (Tier 1 only)")
        wk = fk + ik
        R.assert_no_reserved([k.split(":", 1)[-1] for k in wk])
        shared_assert_no_reserved([k.split(":", 1)[-1] for k in wk], where="M6 arch-v2 grab stations")
        res = R.reserved_usgs_sites()
        if any(LS.contains_reserved_id(k, res) for k in wk):
            raise R.ReservedSiteError("reserved number inside a grab station key")
        role_of = LS.load_trackn_roles()["role_of"]
        if any(role_of.get(k) != "train" for k in ik):
            raise RuntimeError("an inner grab station is not track-N train-role")
        ig = set(linfo["inner_groups"])
        if set(lp.loc[fk, "basin_group"].astype(str)) & ig or set(lp.loc[fk, "group_role"]) - {"train"}:
            raise RuntimeError("a Tier 1 fit station lies in an inner or non-train group")
        forc_w = LD._wqp_forcings(wk, dates)
        forc_w, up_w = LD._append_upstream(forc_w, D.load_upstream_forcings(wk, dates, path=LD.UPSTREAM_TRACKN))
        static_w = LD._wqp_static(wk, list(data.static.columns))
        pi = lp.loc[ik]
        grp = np.concatenate([lp.loc[fk, "basin_group"].astype(str).to_numpy(),
                              np.where(pi["basin_group"].astype(str).isin(ig), pi["basin_group"].astype(str),
                                       pi["trackn_group"].astype(str))])
        S0 = len(data.sites)
        data = dataclasses.replace(
            data, sites=list(data.sites) + wk, roles=np.concatenate([data.roles, np.array(["train"] * len(wk))]),
            basin_group=np.concatenate([data.basin_group, grp]), forc=np.concatenate([data.forc, forc_w], axis=0),
            y=np.concatenate([y, np.full((len(wk),) + y.shape[1:], np.nan, dtype=y.dtype)], axis=0),
            prov=np.concatenate([data.prov, np.zeros((len(wk),) + data.prov.shape[1:], dtype=bool)], axis=0),
            static=pd.concat([data.static, static_w]),
            info={**data.info, "n_sites": S0 + len(wk), "grab_upstream": up_w})
        weak_idx = np.arange(S0, S0 + len(fk))
        grab_inner_idx = np.arange(S0 + len(fk), S0 + len(wk))
        y = data.y
    S, T = len(data.sites), len(dates)
    from .runner import assert_pool_clean

    assert_pool_clean(data.sites, where="M6 arch-v2 training pool")      # every location, before any label is attached
    pos = {s: i for i, s in enumerate(data.sites)}
    tmask = data.block_mask("train")
    # ---- daily labels (train block only)
    daily = np.full((S, T, len(DAILY)), np.nan, dtype="float32")
    for name, v in (("T_mean", "temp_c_mean"), ("T_7dadm", "temp_c_7dadm"), ("DO_mean", "do_mgl_mean"),
                    ("DO_min", "do_mgl_min")):
        daily[:, :, DAILY.index(name)] = y[:, :, data.vi(v)]
    for name, v in AUX_WQ.items():
        daily[:, :, DAILY.index(name)] = D.to_model(v, y[:, :, data.vi(v)].astype("float64"))
    end = guard.data_end_date(blocks)
    cube = scan_cube(usgs_sites, ["temp_c_max", "temp_c_min", "do_mgl_max"], start=D.DATA_START, end=end, retries=2,
                     wait_s=10.0)
    guard.assert_cube_within(cube.dates, blocks)
    if len(cube.dates) != T:
        raise RuntimeError("auxiliary cube dates differ from the training dates")
    for name, v in (("T_max", "temp_c_max"), ("T_min", "temp_c_min"), ("DO_max", "do_mgl_max")):
        daily[:n_usgs, :, DAILY.index(name)] = cube.y[:, :, list(cube.variables).index(v)]
    daily[:aux.shape[0], :, DAILY.index("logq")] = aux[:, :, 0]
    daily[:, ~tmask] = np.nan
    # ---- instantaneous grab rows
    inst_val = np.full((S, T, 2), np.nan, dtype="float32")
    inst_hour = np.full((S, T, 2), np.nan, dtype="float32")
    prov_class = np.full(S, -1, int)
    n_inst = {}
    if grab:
        if len(pd.read_parquet(V2_DIR / "grab_logger_daily.parquet")):
            raise NotImplementedError("logger days present: they need the hourly store (none exist in Tier 1)")
        inst = read_grab_inst()
        si = inst["station_key"].map(pos)
        if si.isna().any():
            raise RuntimeError("a grab row belongs to a station outside the pool")
        si, ti = si.to_numpy(int), (inst["date"] - dates[0]).dt.days.to_numpy()
        vi = (inst["var"] == "DO").to_numpy(int)
        inst_val[si, ti, vi] = inst["value"].to_numpy(dtype="float32")
        inst_hour[si, ti, vi] = inst["hour"].to_numpy(dtype="float32")
        inst_val[:, ~tmask] = np.nan
        pc = stn["provider_class"].map({c: i for i, c in enumerate(PROVIDER_CLASSES)})
        prov_class[[pos[k] for k in pc.index]] = pc.to_numpy(int)
        for nm, idx in (("fit", weak_idx), ("inner", grab_inner_idx)):
            n_inst[nm] = {v: int(np.isfinite(inst_val[idx][:, :, j]).sum()) for j, v in enumerate(INST)}
    # ---- hourly sonde table
    hz = np.load(V2_DIR / "hourly_train.npz")
    if str(hz["date_start"]) != str(dates[0].date()) or str(hz["date_end"]) != str(dates[-1].date()) \
            or pd.Timestamp(str(hz["date_end"])) > pd.Timestamp(LS.TRAIN_END):
        raise RuntimeError("hourly table dates differ from the training dates")
    hourly_row = np.full(S, -1, int)
    assert_pool_clean(hz["sites"].tolist(), where="M6 arch-v2 hourly table")
    for i, s in enumerate(hz["sites"].tolist()):
        if s in pos:
            hourly_row[pos[s]] = i
    if (hourly_row[np.r_[weak_idx, grab_inner_idx].astype(int)] >= 0).any():
        raise RuntimeError("an hourly table is attached to a grab station")
    # ---- deterministic extra inputs
    srad = data.forc[:, :, D.FORCING_VARS.index("srad_wm2")]
    solar = solar_channels(data.static["lat"].to_numpy(), data.static["elev_m"].to_numpy(), dates.dayofyear, srad)
    vaa = pd.read_parquet(V2_DIR / "vaa_static.parquet").set_index("key")
    assert_pool_clean(vaa.index, where="M6 arch-v2 VAA table")
    vaa = vaa.reindex(data.sites)[list(VAA_COLS)]
    data.info["inner_sites"] = [data.sites[i] for i in inner_idx]
    if not (data.roles == "train").all() or set(data.basin_group[np.r_[fit_idx, weak_idx].astype(int)]) & set(linfo["inner_groups"]):
        raise RuntimeError("a fit location is not train-role or lies in an inner group")
    info = {"n_locations": S, "n_fit": int(len(fit_idx)), "n_inner": int(len(inner_idx)),
            "n_final_fit": int(len(final_fit_idx)), "n_grab_fit": int(len(weak_idx)),
            "n_grab_inner": int(len(grab_inner_idx)), "inst_rows": n_inst,
            "n_hourly_sites": int((hourly_row >= 0).sum()),
            "daily_cells": {k: int(np.isfinite(daily[:, :, j]).sum()) for j, k in enumerate(DAILY)},
            "split_sha256": linfo["split_sha256"], "pins_sha256": sha256_file(PINS_FILE) if pins else None,
            "grab": bool(grab), "date_start": str(dates[0].date()), "date_end": str(dates[-1].date()),
            "n_usgs_sites": int(n_usgs), "n_new_usgs": int(linfo["n_new_usgs"]), "streampulse_rows": 0,
            "forcing_only_keys_excluded": n_forcing_only, "date_filter": "inside the file scan (daily, grab, hourly)"}
    v2 = V2Train(data, np.asarray(fit_idx, int), np.asarray(inner_idx, int), np.asarray(final_fit_idx, int),
                 np.asarray(weak_idx, int), grab_inner_idx, daily, inst_val, inst_hour, hz["values"], hourly_row,
                 hz["hour_frac"], prov_class, solar, vaa, info)
    if pins is not None:
        key = "pool" if grab else "pool_nograb"
        fp = pool_fingerprint(v2)
        if fp != pins[key]:
            diff = sorted(k for k in fp if fp[k] != pins[key].get(k))
            raise RuntimeError(f"the pool as loaded differs from its pinned fingerprint in {diff}")
        info["pool_fingerprint"] = "matches pins.json"
    log(f"[v2.data] {info}")
    return v2


def smoke_subset(v2, rs: np.random.Generator):
    """≤ 50 train-role locations: 20 fit sites (hourly tables first), 6 inner sites, 20 grab stations, 4 inner grab."""
    def pick(idx, n, prefer=None):
        idx = np.asarray(idx, int)
        if prefer is not None:
            idx = np.r_[idx[prefer[idx]], idx[~prefer[idx]]]
            return idx[:n]
        return np.sort(rs.choice(idx, min(n, len(idx)), replace=False)) if len(idx) else idx
    has_h = v2.hourly_row >= 0
    keep = np.r_[pick(v2.fit_idx, 20, has_h), pick(v2.inner_idx, 6, has_h), pick(v2.grab_fit_idx, 20),
                 pick(v2.grab_inner_idx, 4)]
    return v2.subset(np.sort(keep))


# ---------------------------------------------------------------------------------------------- model-space arrays
@dataclasses.dataclass
class Scales:
    """Train-only constants (FINAL train-role fit sites × train block): the frozen :class:`Normaliser`, the extra
    input statistics and the label scales. Temperature-type labels share one affine map, DO-type labels another, so
    that mean, max, min, 7-DADMax and instantaneous values live on one axis (needed by the diel operator)."""

    norm: D.Normaliser
    solar_mean: np.ndarray
    solar_std: np.ndarray
    vaa_mean: np.ndarray
    vaa_std: np.ndarray
    label_mean: np.ndarray        # (len(DAILY),)
    label_std: np.ndarray

    def crit_factor(self) -> dict:
        """Per primary target: (label std / frozen target std)², turning a model-space MSE into the standardised
        MSE of the lever protocol (``inner_primary_mse``)."""
        return {v: float((self.label_std[DAILY.index(k)] / self.norm.target_std[D.TARGETS.index(v)]) ** 2)
                for v, k in PRIMARY.items()}

    def state(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()
             if k != "norm"}
        d["norm"] = dataclasses.asdict(self.norm)
        return d

    @classmethod
    def from_state(cls, d: dict) -> "Scales":
        return cls(D.Normaliser(**d["norm"]), *(np.asarray(d[k], dtype="float64") for k in
                                               ("solar_mean", "solar_std", "vaa_mean", "vaa_std", "label_mean",
                                                "label_std")))


def _vaa_transform(vaa: pd.DataFrame) -> np.ndarray:
    v = vaa.reindex(columns=list(VAA_COLS)).astype("float64").copy()
    for c in VAA_LOG:
        v[c] = np.log10(v[c].clip(lower=0) + 1e-3)
    return v.to_numpy()


def fit_scales(v2: V2Train) -> Scales:
    d, idx = v2.data, v2.final_fit_idx
    norm = D.Normaliser.fit(d, idx)
    tm = d.block_mask("train")
    so = v2.solar[idx][:, tm].reshape(-1, len(SOLAR_CHANNELS)).astype("float64")
    va = _vaa_transform(v2.vaa.iloc[idx])
    lm, ls = np.zeros(len(DAILY)), np.ones(len(DAILY))
    for j, k in enumerate(DAILY):
        ref = {"T": "T_mean", "DO": "DO_mean"}.get(k.split("_")[0], k)     # T_* share T_mean's map, DO_* DO_mean's
        x = v2.daily[idx][:, :, DAILY.index(ref)]
        x = x[np.isfinite(x)].astype("float64")
        if x.size > 1:
            lm[j], ls[j] = x.mean(), max(x.std(), 1e-6)
    return Scales(norm, np.nanmean(so, axis=0), np.maximum(np.nanstd(so, axis=0), 1e-6), np.nanmean(va, axis=0),
                  np.maximum(np.nanstd(va, axis=0), 1e-6), lm, ls)


def make_inputs(data: D.HydroMetData, solar: np.ndarray, vaa: pd.DataFrame, sc: Scales, *, lag: int = 0):
    """US input set v2: the frozen ``ealstm_ws`` inputs (22 forcing channels + 4 calendar harmonics; frozen statics)
    + 4 solar channels + 3 VAA statics. Returns ``(dyn, st, dyn_names, st_names)``."""
    dyn, st, dn, sn = build_inputs(data, sc.norm, lag)
    z = np.nan_to_num(np.clip((solar - sc.solar_mean) / sc.solar_std, -10, 10), nan=0.0).astype("float32")
    v = np.nan_to_num(np.clip((_vaa_transform(vaa) - sc.vaa_mean) / sc.vaa_std, -10, 10), nan=0.0).astype("float32")
    return (np.concatenate([dyn, z], axis=-1), np.concatenate([st, v], axis=-1),
            list(dn) + [f"solar_{c}" for c in SOLAR_CHANNELS], list(sn) + list(VAA_COLS))


def standardise_labels(v2: V2Train, sc: Scales):
    """Model-space labels: ``(Yd, inst, hourly)`` with T-type values on T_mean's scale and DO-type on DO_mean's."""
    Yd = ((v2.daily - sc.label_mean) / sc.label_std).astype("float32")
    j = [DAILY.index("T_mean"), DAILY.index("DO_mean")]
    inst = ((v2.inst_val - sc.label_mean[j]) / sc.label_std[j]).astype("float32")
    hourly = ((v2.hourly - sc.label_mean[j]) / sc.label_std[j]).astype("float32")
    return Yd, inst, hourly
