# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Guarded data preparation for SENTINEL-Sonde v2 runs.

Mirrors ``sentinel.models.sonde.runner.prepare`` (frozen; not edited) but (i) loads the auxiliary daily targets of
amendment §2d (T max/min, DO max, SC max, turbidity max) next to the frozen TARGETS, (ii) keeps every site type
(M1-N's primary cell is all classes), and (iii) authorises only the blocks the caller names. Training callers pass
``blocks=("train",)``: the val block is then not authorised at all (``SondeData.issue_idx`` refuses it).
Auxiliary SC/turbidity maxima use the same log offsets as their means (registered in-process in ``LOG_VARS``; the
frozen file is unchanged and no existing variable is affected; asserted below).

Reference window (m1n_protocol §1, ew_v3 W-N2; lead ruling F): per site and variable the first ``REF_DAYS`` calendar
days from the first finite value in the cube (cube start 2007-10-01), ≥ ``REF_MIN_OBS`` observed days. The
climatology of every model is the reference-window climatology, and an issue day is USED (training, early stopping,
baseline rows, scoring; every model) only if issue index ≥ first index + ``REF_DAYS`` + ``REF_EMBARGO``
(:class:`SondeDataV2`). Hindcast inputs before that day are not blanked. A site-variable whose 730-day window is
not wholly inside the dates loaded by a build has NO climatology and no usable issue day in that build (never a
truncated window; ``sonde_arch_v2_dev_lead_rulings.md`` F(3)).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from sentinel.core import paths
from sentinel.data.reserved import assert_no_reserved
from sentinel.models.sonde import common
from sentinel.models.sonde import data as sdata

AUX = ["temp_c_max", "temp_c_min", "do_mgl_max", "spcond_uscm_max", "turb_fnu_max"]
GAGE = "gage_ft_mean"                                   # §2d gage height: input for every model, SONDE-HF output
V2_VARS = list(sdata.TARGETS) + AUX + [GAGE]
DERIVED = ["do_amp_mgl", "do_pctsat", "temp_c_7dadm"]   # amendment §2d, derived from the daily tier; do_pctsat is
# an OBSERVED series for scoring/baselines only: no model reads it and no head is trained on it (lead ruling H)
NOT_MODELLED = ("do_pctsat",)
sdata.LOG_VARS.setdefault("spcond_uscm_max", 1.0)
sdata.LOG_VARS.setdefault("turb_fnu_max", 0.1)
assert sdata.LOG_VARS["spcond_uscm_max"] == sdata.LOG_VARS["spcond_uscm_mean"], "SC max/mean log offsets differ"
assert sdata.LOG_VARS["turb_fnu_max"] == sdata.LOG_VARS["turb_fnu_mean"], "turbidity max/mean log offsets differ"
assert GAGE not in sdata.LOG_VARS and GAGE not in sdata.TARGETS

REF_DAYS = 730          # reference window length (calendar days from the first observation of the variable)
REF_MIN_OBS = 365       # observed days required inside the window
REF_EMBARGO = 42        # days between the window end and the first usable issue day
NEVER = np.iinfo(np.int32).max // 2          # issue start of a site-variable without a reference climatology
GEFS_REQUIRED = {       # month files that must exist (YYYYMM, inclusive) per use
    "step1": {"reforecast": ("200710", "201912")},
    "cv": {"reforecast": ("200710", "201912"), "operational": ("202009", "202309")},
}


@dataclass
class SondeDataV2(common.SondeData):
    """Frozen ``SondeData`` plus the reference-window issue rule. ``rows`` (the common rows every model predicts
    and is scored on) keeps only issue days allowed by :meth:`issue_ok`; training code applies the same mask."""

    ref_issue_start: np.ndarray | None = None          # (S, V) first usable issue index per site and variable
    n_dropped_ref_window: dict = field(default_factory=dict, repr=False)   # (var, block) -> counts

    def issue_ok(self, var: str, si: np.ndarray, ti: np.ndarray) -> np.ndarray:
        if self.ref_issue_start is None:
            raise RuntimeError("reference-window issue rule not built (apply_reference_window_climatology)")
        return np.asarray(ti) >= self.ref_issue_start[np.asarray(si), self.cube.var_index(var)]

    def rows(self, var: str, block: str):
        key = (var, block)
        if key not in self._rows:
            r = common.common_rows(self, var, block)
            keep = self.issue_ok(var, r.si, r.ti)
            self.n_dropped_ref_window[key] = {
                "rows": int((~keep).sum()),
                "scored": {int(h): int((r.score_mask[h] & ~keep).sum()) for h in self.horizons}}
            self._rows[key] = subset_rows(r, keep)
        return self._rows[key]


def subset_rows(rows, keep: np.ndarray):
    """Row subset of a frozen ``Rows`` object (every per-row array and per-horizon mask)."""
    from copy import copy

    r = copy(rows)
    for a in ("si", "ti", "site_no", "issue_date", "role"):
        setattr(r, a, getattr(rows, a)[keep])
    r.score_mask = {h: m[keep] for h, m in rows.score_mask.items()}
    r.obs = {h: o[keep] for h, o in rows.obs.items()}
    r.n_dropped_baseline_undefined = dict(rows.n_dropped_baseline_undefined)
    r.n_dropped_provisional = dict(rows.n_dropped_provisional)
    return r


def prepare_v2(*, blocks=("train",), roles=("train", "val"), site_types=None, use_weather: bool = True,
               use_catchment: bool = True, horizons=(1, 2, 3, 5, 7, 10, 14), log=print) -> SondeDataV2:
    if set(blocks) & {"test", "test_b"} or "test" in roles:
        raise PermissionError("v2 training/CV never loads test blocks or test-role sites")
    split, info = common.load_split(paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json")
    if not info.get("final"):
        raise RuntimeError("FINAL split required")
    sites = sorted(s for r in roles for s in split.sites(r))
    assert_no_reserved(sites)
    missing = set(common.wait_for_sites(sites, retries=0))
    sites = [s for s in sites if s not in missing]
    if site_types:
        reg = common.site_registry()
        sites = [s for s in sites if s in reg.index and reg.loc[s, "site_type"] in set(site_types)]
    d0 = common.prepare_data(split, info, sites, blocks=tuple(blocks), variables=V2_VARS,
                             horizons=tuple(horizons))
    data = SondeDataV2(**{f.name: getattr(d0, f.name) for f in dataclasses.fields(d0)})
    add_derived_targets(data)
    apply_reference_window_climatology(data)     # M1-N §3: reference-window climatology + issue rule, every model
    add_slope_static(data)
    if use_catchment:
        import pandas as pd
        sc = common.load_catchment(data.sites)
        data.static = pd.concat([data.static, sc.set_index(data.static.index)], axis=1)
    if use_weather:
        data.weather = common.load_weather(data.cube)
    log(f"[prep_v2] {len(data.sites)} sites, roles {dict(zip(*np.unique(data.roles, return_counts=True)))}, "
        f"blocks {blocks}, {len(V2_VARS)} variables")
    return data


def observed_weather_fields(data) -> np.ndarray:
    """UNLAGGED observed daily weather (valid day t at index t) in nwp.FIELDS order — perfect-prog TRAINING input
    and bias-correction / error-model reference only (never an input at issue time)."""
    from . import nwp as NWP

    w = common.load_weather(data.cube, lag_days=0)                 # precip, tmax, tmin, srad, wind, vpd
    out = np.full(w.shape[:2] + (len(NWP.FIELDS),), np.nan, np.float32)
    out[..., NWP.F_IDX["tmax_c"]] = w[..., 1]
    out[..., NWP.F_IDX["tmin_c"]] = w[..., 2]
    out[..., NWP.F_IDX["tmean_c"]] = (w[..., 1] + w[..., 2]) / 2
    out[..., NWP.F_IDX["precip_mm"]] = w[..., 0]
    out[..., NWP.F_IDX["srad_wm2"]] = w[..., 3]
    return out


def month_range(first: str, last: str) -> list[str]:
    import pandas as pd

    return [f"{t.year}{t.month:02d}" for t in pd.period_range(first[:4] + "-" + first[4:], last[:4] + "-" + last[4:],
                                                             freq="M")]


def load_gefs(data, product: str = "reforecast", *, months: tuple[str, str] | None = None, root=None) -> np.ndarray:
    """GEFS c00 point extraction (``scripts/data/extract_gefs.py``) → (S, T, 7, F) aligned to issue dates; split
    sites only (the new-site extraction lives elsewhere and is never read here). ``months`` = (first, last) YYYYMM:
    every month file in that range must exist (hard error otherwise) and only those files are read."""
    import pandas as pd

    from . import nwp as NWP

    d = (root or paths.PROCESSED / "gefs_points") / product
    if months is not None:
        need = month_range(*months)
        miss = [m for m in need if not (d / f"{m}.parquet").exists()]
        if miss:
            raise FileNotFoundError(f"GEFS {product}: {len(miss)} of {len(need)} required month files missing "
                                    f"({miss[0]}..{miss[-1]}); refusing to run on an incomplete product")
        files = [d / f"{m}.parquet" for m in need]
    else:
        files = sorted(d.glob("*.parquet"))
    S, T = len(data.sites), len(data.cube.dates)
    x = np.full((S, T, 7, len(NWP.FIELDS)), np.nan, np.float32)
    pos = {s: i for i, s in enumerate(data.sites)}
    for f in files:
        df = pd.read_parquet(f)
        df = df[df["site"].isin(pos)]
        if df.empty:
            continue
        si = df["site"].map(pos).to_numpy()
        ti = data.cube.dates.get_indexer(pd.DatetimeIndex(df["issue"]))
        ok = ti >= 0
        for k in range(1, 8):
            for name, fld, sc in (("tmax", "tmax_c", 10), ("tmin", "tmin_c", 10), ("p", "precip_mm", 10),
                                  ("sw", "srad_wm2", 1)):
                v = df[f"{name}_l{k}"].to_numpy()
                val = np.where(v == -32768, np.nan, v / sc).astype(np.float32)
                x[si[ok], ti[ok], k - 1, NWP.F_IDX[fld]] = val[ok]
            x[si[ok], ti[ok], k - 1, NWP.F_IDX["tmean_c"]] = (x[si[ok], ti[ok], k - 1, NWP.F_IDX["tmax_c"]]
                                                              + x[si[ok], ti[ok], k - 1, NWP.F_IDX["tmin_c"]]) / 2
    return x


def gefs_coverage(x: np.ndarray, dates, first, last) -> float:
    """Share of site × issue-day cells in [first, last] whose lead-1 temperature is finite."""
    import pandas as pd

    sel = np.asarray((dates >= pd.Timestamp(first)) & (dates <= pd.Timestamp(last)))
    if not sel.any():
        return 0.0
    return float(np.isfinite(x[:, sel, 0, 2]).mean())


def reference_window(cube, *, days: int = REF_DAYS, min_obs: int = REF_MIN_OBS, embargo: int = REF_EMBARGO,
                     window: int = 15, min_years: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M1-N §3 ("all models are trained with 2-year-window climatologies"; reference window = the first ``days``
    calendar days from the site's first observation of the variable, ≥ ``min_obs`` observed days; m1n §1/§2b).
    Same estimator as the frozen ``data.climatology`` (circular ±``window``-day smoothing, fallback to the window's
    overall mean/variance where fewer than ``min_years`` distinct years support a day of year), applied to the
    reference window only. Returns (S, 366, V) mean and std in model space (NaN where < ``min_obs`` days or where
    the window runs past the last loaded date) and
    ``issue_start`` (S, V): the first usable issue index = first finite index + ``days`` + ``embargo`` (the window
    covers indices first .. first + days − 1; ``NEVER`` where there is no reference climatology)."""
    import pandas as pd
    from scipy.ndimage import maximum_filter1d

    S, T, V = cube.y.shape
    doy = cube.dates.dayofyear.to_numpy() - 1
    yidx = cube.dates.year.to_numpy() - cube.dates.year.min()
    n_years = int(yidx.max()) + 1
    k = np.ones(2 * window + 1)

    def circ(a):
        return np.convolve(np.concatenate([a[-window:], a, a[:window]]), k, "valid")

    mean = np.full((S, 366, V), np.nan)
    std = np.full((S, 366, V), np.nan)
    start = np.full((S, V), NEVER, dtype=np.int64)
    for j, v in enumerate(cube.variables):
        z = sdata.to_model_space(v, cube.y[:, :, j])
        for i in range(S):
            zi = z[i]
            fin = np.flatnonzero(np.isfinite(zi))
            if fin.size == 0:
                continue
            if int(fin[0]) + days > T:                 # window not wholly inside the loaded dates: no climatology
                continue
            t0 = cube.dates[fin[0]]
            inwin = np.asarray((cube.dates >= t0) & (cube.dates < t0 + pd.Timedelta(days=days)))
            ok = np.isfinite(zi) & inwin
            if ok.sum() < min_obs:
                continue
            start[i, j] = int(fin[0]) + days + embargo
            s_sum = circ(np.bincount(doy[ok], weights=zi[ok], minlength=366))
            s_sq = circ(np.bincount(doy[ok], weights=zi[ok] ** 2, minlength=366))
            s_cnt = circ(np.bincount(doy[ok], minlength=366).astype(float))
            pres = np.zeros((366, n_years), dtype=bool)
            pres[doy[ok], yidx[ok]] = True
            support = maximum_filter1d(pres.astype(np.uint8), size=2 * window + 1, axis=0, mode="wrap").sum(axis=1)
            m = s_sum / np.maximum(s_cnt, 1)
            var = np.maximum(s_sq / np.maximum(s_cnt, 1) - m ** 2, 1e-8)
            fb = support < min_years
            m[fb] = np.mean(zi[ok])
            var[fb] = np.var(zi[ok]) + 1e-8
            mean[i, :, j] = m
            std[i, :, j] = np.sqrt(var)
    return mean, std, start


def reference_window_climatology(cube, **kw) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) of :func:`reference_window`."""
    return reference_window(cube, **kw)[:2]


def apply_reference_window_climatology(data) -> None:
    """Replace the training-years climatology by the reference-window climatology for every model and reference
    (regime parity with the new sites) and set the issue rule. Call before any baseline or tensor is built."""
    if data._bl or data._rows:
        raise RuntimeError("apply before baselines/rows are built")
    data.clim_mean, data.clim_std, data.ref_issue_start = reference_window(data.cube)


def do_pctsat_from_quantiles(q_do_mgl: np.ndarray, t_median_c: np.ndarray, elev_m: np.ndarray) -> np.ndarray:
    """§2d DO % saturation as a DETERMINISTIC transform of a model's DO-mean quantiles (n, Q) at its own T-mean
    median (n,): 100 · q / DO_sat(T median, P(elevation)) (Benson–Krause, same routine as the observed series;
    elevation NaN → 0 m). The uncertainty of T is ignored (disclosed)."""
    from sentinel.models.hydromet.physics import do_sat_mgl, pressure_atm_from_elevation

    pres = np.asarray(pressure_atm_from_elevation(np.nan_to_num(np.asarray(elev_m, float), nan=0.0)), float)
    sat = np.asarray(do_sat_mgl(np.asarray(t_median_c, float), pres), float)
    return 100.0 * np.asarray(q_do_mgl, float) / sat[:, None]


def add_derived_targets(data) -> None:
    """Append the §2d derived daily targets to the cube (before climatologies/baselines are built):
    do_amp_mgl = DO max − DO min (same day; both Approved); do_pctsat = 100 · DO mean / DO_sat(T mean, P(elev))
    (Benson–Krause, M6 routine); temp_c_7dadm = mean of T max over days t−6..t (all 7 present). Provisional if any
    component is provisional."""
    import pandas as pd
    from sentinel.models.hydromet.physics import do_sat_mgl, pressure_atm_from_elevation

    c = data.cube
    if "do_amp_mgl" in c.variables:
        return
    def get(v):
        j = c.var_index(v)
        return c.y[:, :, j].astype(np.float64), c.prov[:, :, j]
    domax, pmax = get("do_mgl_max")
    domin, pmin = get("do_mgl_min")
    domean, pmean = get("do_mgl_mean")
    tmean, ptm = get("temp_c_mean")
    tmax, ptx = get("temp_c_max")
    amp = domax - domin
    elev = np.nan_to_num(data.static["elev_m"].to_numpy(float), nan=0.0)
    pres = np.asarray(pressure_atm_from_elevation(elev), float)[:, None]
    sat = np.asarray(do_sat_mgl(tmean, np.broadcast_to(pres, tmean.shape)), float)
    pct = 100.0 * domean / sat
    roll = pd.DataFrame(tmax.T).rolling(7, min_periods=7).mean().to_numpy().T
    rollp = pd.DataFrame(ptx.T.astype(float)).rolling(7, min_periods=1).max().to_numpy().T > 0
    new_y = np.stack([amp, pct, roll], -1).astype(np.float32)
    new_p = np.stack([pmax | pmin, pmean | ptm, rollp], -1)
    c.y = np.concatenate([c.y, new_y], -1)
    c.prov = np.concatenate([c.prov, new_p], -1)
    c.variables = list(c.variables) + DERIVED
    for arr in ("clim_mean", "clim_std"):        # placeholder slices; replaced by the reference-window climatology
        a = getattr(data, arr)
        setattr(data, arr, np.concatenate([a, np.full(a.shape[:2] + (len(DERIVED),), np.nan)], -1))


def add_slope_static(data) -> None:
    """NHDPlus channel slope (StreamCat site table, column nhd_slope) as a static for EVERY model alike:
    ``sc_log10_nhd_slope`` (picked up by the frozen feature table and tensors via the ``sc_`` prefix) and the raw
    ``slope`` column used only by the PF-GBM Raymond K600 feature."""
    import pandas as pd

    sc = pd.read_parquet(paths.PROCESSED / "catchment" / "streamcat_sites.parquet", columns=["site_key", "nhd_slope"])
    m = sc.set_index("site_key")["nhd_slope"]
    v = np.array([m.get(f"USGS-{s}", np.nan) for s in data.sites], dtype=float)
    data.static["slope"] = v
    data.static["sc_log10_nhd_slope"] = np.log10(np.clip(v, 1e-5, None))
