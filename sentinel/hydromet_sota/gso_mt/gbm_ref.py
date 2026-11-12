# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GBM references G1 and G2 (PF-GBM) with the grab-sample formulation (amendment §5).

* **Instantaneous model** per variable v ∈ {T, DO}: one LightGBM on rows (location, date, hour): every grab row D1
  uses, sonde hourly rows at the fixed hours {1, 5, 9, 13, 17, 21} and every daily-mean row (``kind = daily mean``,
  hour features missing). Inputs = the daily columns + first and second hour harmonics, their products with same-day
  tmax, tmin and srad, and the kind flag. Grab rows share one weight so that their total weight is the grab share s
  (lever ``weak_weight``); a grab row without a time has missing hour features.
* **Derived daily outputs**: daily mean = mean of the 24 hourly predictions; daily max = mean + amplitude model
  (label: sonde daily max − mean, clipped at 0); 7-DADMax = strict trailing 7-day mean of the daily max; DO min = mean −
  amplitude model (label: sonde daily mean − min).
* **Direct models**: v1-style LightGBM per primary daily target on the enlarged daily rows.
* Grid (``max_rows_per_site``, grab share), every member early-stopped on the same inner rows; **per primary target
  the reference takes the lowest inner MSE over {derived, direct} × grid**; the hour-resolved output takes the grid
  point with the lowest capped inner grab MSE. Selected members are refitted on fit + inner rows with their
  iteration count.
* **G2**: physics inputs (prior-parameter equilibrium temperature and trailing means, DO saturation at it, cloud
  ratio, photoperiod, VAA statics) and stacked inputs (cross-fitted by basin group, lever :class:`QStack`) of log Q,
  daily T min, T max and DO max.

Everything here sees train-role locations × train block only (the arrays of :class:`~.data.V2Train`)."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...models.hydromet import data as D
from ...models.hydromet.baselines import LGBMConfig, check_fit_sites
from ...models.hydromet.features import FeatureSet, trailing
from ...models.hydromet.physics import do_sat_mgl, pressure_atm_from_elevation
from ..gbm import QStack, _lgbm, weak_weight
from . import criterion as C
from .data import DAILY, INST, PRIMARY, Scales, V2Train
from .grab import FIXED_HOURS
from .solar import MIX_HOURS, SOLAR_CHANNELS

HOUR_COLS = ("hr_sin1", "hr_cos1", "hr_sin2", "hr_cos2")
INTERACT = ("f_tmax_c_m1", "f_tmin_c_m1", "f_srad_wm2_m1")
KIND_COL = "kind_daily_mean"
N_EXTRA = len(HOUR_COLS) * (1 + len(INTERACT)) + 1
HOURS24 = np.arange(24) + 0.5
STACKED = {"logq": "f_stk_logq", "T_min": "f_stk_tmin", "T_max": "f_stk_tmax", "DO_max": "f_stk_domax"}
TE_WINDOWS = (1, 3, 7, 14, 30)
GRID = ((3000, 0.5), (3000, 0.25), (1000, 0.5))
AMP = {"T": ("T_max", "T_mean", +1.0), "DO": ("DO_mean", "DO_min", -1.0)}      # (upper, lower, sign of the offset)
_J = {k: DAILY.index(k) for k in DAILY}


class ResumeNeeded(RuntimeError):
    """The job chunk is over; the next job continues from the member cache."""


class MemberCache:
    """One file per fitted member under ``root``; a job that passes ``deadline`` (epoch seconds) stops before the
    next member with :class:`ResumeNeeded`. Files are this run's own outputs (joblib)."""

    def __init__(self, root, deadline: float | None = None):
        self.root, self.deadline = Path(root), deadline
        self.root.mkdir(parents=True, exist_ok=True)
        self.n_loaded = self.n_built = 0

    def get(self, key: str, build):
        import joblib

        p = self.root / f"{key}.joblib"
        if p.is_file():
            self.n_loaded += 1
            return joblib.load(p)
        if self.deadline is not None and time.time() > self.deadline:
            raise ResumeNeeded(key)
        obj = build()
        tmp = p.with_suffix(".tmp")
        joblib.dump(obj, tmp, compress=3)
        tmp.replace(p)
        self.n_built += 1
        return obj


def extra_names() -> list[str]:
    return list(HOUR_COLS) + [f"{h}_x_{c}" for h in HOUR_COLS for c in INTERACT] + [KIND_COL]


def hour_block(hour: np.ndarray, x3: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """(n, N_EXTRA) float32: hour harmonics, their products with the three same-day forcings, the kind flag. A missing
    hour gives missing harmonics and products."""
    w = 2.0 * np.pi * np.asarray(hour, "float64") / 24.0
    h = np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], axis=1)
    prod = (h[:, :, None] * np.asarray(x3, "float64")[:, None, :]).reshape(len(h), -1)
    return np.concatenate([h, prod, np.asarray(kind, "float64")[:, None]], axis=1).astype("float32")


# ------------------------------------------------------------------------------------------------ feature sets
def v2_inputs(fs: FeatureSet, solar: np.ndarray, vaa: pd.DataFrame | None) -> FeatureSet:
    """The additions of the US input set v2 that **every** reference gets as features (amendment §3): the 4 solar
    channels and the 3 VAA statics (``vaa=None`` in the T-E schema). G1 = frozen causal features + these."""
    if fs.lag != 0:
        raise NotImplementedError("the arch-v2 references are nowcasts (lag 0)")
    dyn = np.concatenate([fs.dyn, np.asarray(solar, "float32")], axis=-1)
    static = fs.static if vaa is None else pd.concat([fs.static, vaa.set_axis(fs.static.index)], axis=1)
    return FeatureSet(dyn, list(fs.dyn_names) + [f"f_solar_{c}" for c in SOLAR_CHANNELS], static, fs.cal, fs.lag)


def physics_features(fs: FeatureSet, *, te: np.ndarray, elev_m: np.ndarray, solar: np.ndarray | None = None,
                     vaa: pd.DataFrame | None = None) -> FeatureSet:
    """G2 physics inputs: prior-parameter equilibrium temperature (trailing means), DO saturation at it and — when
    ``fs`` does not already hold the :func:`v2_inputs` additions — cloud ratio, photoperiod and the VAA statics."""
    if fs.lag != 0:
        raise NotImplementedError("physics inputs are built for the nowcast (lag 0)")
    te = np.asarray(te, "float64")
    cols = [trailing(te, w, "mean") for w in TE_WINDOWS]
    names = [f"f_te_m{w}" for w in TE_WINDOWS]
    p = pressure_atm_from_elevation(elev_m)[:, None]
    cols.append(np.where(np.isfinite(te), do_sat_mgl(np.nan_to_num(te), p), np.nan))
    names.append("f_dosat_te")
    if solar is not None:
        cols += [solar[:, :, SOLAR_CHANNELS.index("cloud_ratio")], solar[:, :, SOLAR_CHANNELS.index("photoperiod_h")]]
        names += ["f_cloud_ratio", "f_photoperiod_h"]
    dyn = np.concatenate([fs.dyn, np.stack(cols, axis=-1).astype("float32")], axis=-1)
    static = fs.static if vaa is None else pd.concat([fs.static, vaa.set_axis(fs.static.index)], axis=1)
    return FeatureSet(dyn, list(fs.dyn_names) + names, static, fs.cal, fs.lag)


def append_stack(fs: FeatureSet, pred: np.ndarray, name: str) -> FeatureSet:
    q = np.asarray(pred, "float64")
    add = np.stack([q, trailing(q, 7, "mean"), trailing(q, 30, "mean")], axis=-1).astype("float32")
    return FeatureSet(np.concatenate([fs.dyn, add], axis=-1), list(fs.dyn_names) + [name, name + "_m7", name + "_m30"],
                      fs.static, fs.cal, fs.lag)


def add_stacks(v2: V2Train, fs: FeatureSet, cfg: LGBMConfig, seed: int, *, log=print, n_folds: int = 5,
               max_rows_per_site: int = 1000, cache: MemberCache | None = None):
    """Stacked inputs of G2. Each stack model sees the columns of ``fs`` (not the other stacks): out-of-fold
    predictions at the fit sites (``n_folds`` folds by basin group), full-fit predictions at every other location.
    Returns ``(fs with the stacked columns, {label: QStack}, info)``."""
    cols, out, models, info = fs.columns(), fs, {}, {}
    for k, name in STACKED.items():
        lab = v2.daily[:, :, _J[k]].astype("float64")
        def build(lab=lab, k=k):
            q = QStack(cols, cfg, seed, n_folds=n_folds, max_rows_per_site=max_rows_per_site)
            return q, q.fit_oof(v2.data, fs, lab, v2.fit_idx, v2.inner_idx, log=lambda m: log(f"[stack {k}] {m}"))

        qs, pred = build() if cache is None else cache.get(f"stack_{k}", build)
        out, models[k], info[k] = append_stack(out, pred, name), qs, dict(qs.info)
    return out, models, info


def apply_stacks(fs: FeatureSet, models: dict, site_idx=None) -> FeatureSet:
    """Stacked inputs from the full-fit stack models (locations outside the fit sites, e.g. a scoring cell)."""
    idx = np.arange(fs.dyn.shape[0]) if site_idx is None else np.asarray(site_idx, int)
    out = fs
    for k, name in STACKED.items():
        pred = np.full(fs.dyn.shape[:2], np.nan, dtype="float32")
        pred[idx] = models[k].predict(fs, idx)
        out = append_stack(out, pred, name)
    return out


# ------------------------------------------------------------------------------------------------------- rows
def _cap(si: np.ndarray, cap, seed: int) -> np.ndarray:
    """Positions of ≤ ``cap`` rows per location (seeded), in ascending order."""
    if not cap or not len(si):
        return np.arange(len(si))
    o = np.random.default_rng(seed).permutation(len(si))
    keep = pd.Series(si[o]).groupby(si[o]).cumcount().to_numpy() < cap
    return np.sort(o[keep])


def inst_rows(v2: V2Train, j: int, dense_idx, grab_idx, *, cap=None, seed: int = 0, grab: bool = True) -> dict:
    """Rows of the instantaneous model of variable ``INST[j]``: ``src`` 0 = daily mean (kind 1, no hour), 1 = sonde
    hourly row at a fixed hour, 2 = grab row. ``cap`` applies per location to the daily rows and to the hourly rows
    separately; grab rows are never capped."""
    dense_idx = np.asarray(dense_idx, int)
    lab = v2.daily[:, :, _J[("T_mean", "DO_mean")[j]]]
    a, t = np.nonzero(np.isfinite(lab[dense_idx]))
    s = dense_idx[a]
    k = _cap(s, cap, seed)
    parts = [(s[k], t[k], np.full(len(k), np.nan), np.ones(len(k)), lab[s[k], t[k]], np.zeros(len(k), int))]
    hs = dense_idx[v2.hourly_row[dense_idx] >= 0]
    if len(hs):
        fixed = np.asarray(FIXED_HOURS)
        hv = v2.hourly[v2.hourly_row[hs]][:, :, fixed, j]
        a, t, h = np.nonzero(np.isfinite(hv))
        s = hs[a]
        k = _cap(s, cap, seed + 1)
        parts.append((s[k], t[k], fixed[h[k]] + v2.hour_frac[v2.hourly_row[s[k]]], np.zeros(len(k)), hv[a[k], t[k], h[k]],
                      np.ones(len(k), int)))
    grab_idx = np.asarray(grab_idx, int)
    if grab and len(grab_idx):
        a, t = np.nonzero(np.isfinite(v2.inst_val[grab_idx][:, :, j]))
        s = grab_idx[a]
        parts.append((s, t, v2.inst_hour[s, t, j], np.zeros(len(s)), v2.inst_val[s, t, j], np.full(len(s), 2)))
    cat = lambda i, dt: np.concatenate([np.asarray(p[i], dt) for p in parts])   # noqa: E731
    return {"si": cat(0, int), "ti": cat(1, int), "hour": cat(2, "float64"), "kind": cat(3, "float64"),
            "y": cat(4, "float64"), "src": cat(5, int)}


# ----------------------------------------------------------------------------------------------------- models
class DailyGBM:
    """One LightGBM on the daily rows of a label array (S, T) in natural units (NaN = no row)."""

    def __init__(self, name: str, columns, cfg: LGBMConfig, seed: int = 0):
        self.name, self.columns, self.cfg, self.seed = name, list(columns), cfg, seed
        self.model, self.it, self.cap, self.info = None, None, None, {}

    def _xy(self, fs, label, idx, cap):
        idx = np.asarray(idx, int)
        a, t = np.nonzero(np.isfinite(label[idx]))
        s = idx[a]
        k = _cap(s, cap, self.seed)
        return fs.design_matrix(s[k], t[k], self.columns), label[s[k], t[k]].astype("float64")

    def fit(self, data, fs, label, fit_idx, inner_idx, *, cap, log=print):
        check_fit_sites(data, fit_idx, inner_idx)
        self.cap, t0 = cap, time.time()
        X, y = self._xy(fs, label, fit_idx, cap)
        Xv, yv = self._xy(fs, label, inner_idx, None)
        self.model, self.it, rmse = _lgbm(X, y, None, Xv if len(yv) else None, yv, self.cfg, self.seed)
        self.info = {"n_train": int(len(y)), "n_inner": int(len(yv)), "best_iteration": self.it, "inner_rmse": rmse,
                     "max_rows_per_site": cap, "n_columns": int(X.shape[1]), "fit_s": round(time.time() - t0, 1)}
        log(f"[{self.name}] {self.info}")
        return self

    def refit(self, data, fs, label, fit_idx, inner_idx, log=print):
        check_fit_sites(data, fit_idx, inner_idx)
        X, y = self._xy(fs, label, np.r_[np.asarray(fit_idx, int), np.asarray(inner_idx, int)], self.cap)
        self.model, _, _ = _lgbm(X, y, None, None, None, self.cfg, self.seed, n_estimators=max(self.it, 10))
        self.info["refit_rows"] = int(len(y))
        log(f"[{self.name}] refit on fit + inner: {len(y)} rows, {max(self.it, 10)} trees")
        return self

    def predict(self, fs, si, ti, chunk: int = 500_000) -> np.ndarray:
        out = np.empty(len(si))
        for c in range(0, len(si), chunk):
            out[c:c + chunk] = self.model.predict(fs.design_matrix(si[c:c + chunk], ti[c:c + chunk], self.columns))
        return out


class InstGBM:
    """Instantaneous model of one variable (module docstring)."""

    def __init__(self, var: str, columns, cfg: LGBMConfig, seed: int = 0):
        self.var, self.j, self.columns, self.cfg, self.seed = var, INST.index(var), list(columns), cfg, seed
        self.ix = [self.columns.index(c) for c in INTERACT]
        self.model, self.it, self.cap, self.share, self.info = None, None, None, 0.0, {}

    def design(self, fs, si, ti, hour, kind) -> np.ndarray:
        base = fs.design_matrix(si, ti, self.columns)
        return np.concatenate([base, hour_block(hour, base[:, self.ix], kind)], axis=1)

    def _xyw(self, v2, fs, dense, grabs, cap):
        r = inst_rows(v2, self.j, dense, grabs, cap=cap, seed=self.seed, grab=self.share > 0)
        g = r["src"] == 2
        w = np.where(g, weak_weight(int((~g).sum()), int(g.sum()), self.share), 1.0)
        return self.design(fs, r["si"], r["ti"], r["hour"], r["kind"]), r["y"], w, r

    def fit(self, v2: V2Train, fs, *, cap, share: float, log=print):
        check_fit_sites(v2.data, v2.fit_idx, v2.inner_idx, v2.grab_fit_idx, v2.grab_inner_idx)
        self.cap, self.share, t0 = cap, float(share), time.time()
        X, y, w, r = self._xyw(v2, fs, v2.fit_idx, v2.grab_fit_idx, cap)
        Xv, yv, _, _ = self._xyw(v2, fs, v2.inner_idx, v2.grab_inner_idx, None)
        self.model, self.it, rmse = _lgbm(X, y, w, Xv if len(yv) else None, yv, self.cfg, self.seed)
        self.info = {"n_train": int(len(y)), "rows_by_src": np.bincount(r["src"], minlength=3).tolist(),
                     "grab_row_weight": float(w[r["src"] == 2][0]) if (r["src"] == 2).any() else 0.0,
                     "n_inner": int(len(yv)), "best_iteration": self.it, "inner_rmse": rmse,
                     "max_rows_per_site": cap, "grab_share": self.share, "n_columns": int(X.shape[1]),
                     "fit_s": round(time.time() - t0, 1)}
        log(f"[inst {self.var}] {self.info}")
        return self

    def refit(self, v2: V2Train, fs, log=print):
        X, y, w, _ = self._xyw(v2, fs, np.r_[v2.fit_idx, v2.inner_idx], np.r_[v2.grab_fit_idx, v2.grab_inner_idx],
                               self.cap)
        self.model, _, _ = _lgbm(X, y, w, None, None, self.cfg, self.seed, n_estimators=max(self.it, 10))
        self.info["refit_rows"] = int(len(y))
        log(f"[inst {self.var}] refit on fit + inner: {len(y)} rows, {max(self.it, 10)} trees")
        return self

    def mean_over(self, fs, si, ti, hours, chunk: int = 250_000) -> np.ndarray:
        """Mean of the instantaneous predictions over ``hours`` (24 half-hour-centred hours = the daily mean)."""
        out = np.empty(len(si))
        for c in range(0, len(si), chunk):
            base = fs.design_matrix(si[c:c + chunk], ti[c:c + chunk], self.columns)
            n, K = base.shape
            X = np.concatenate([base, np.zeros((n, N_EXTRA), dtype="float32")], axis=1)
            acc = np.zeros(n)
            for h in hours:
                X[:, K:] = hour_block(np.full(n, float(h)), base[:, self.ix], np.zeros(n))
                acc += self.model.predict(X)
            out[c:c + n] = acc / len(hours)
        return out

    def at(self, fs, si, ti, hour) -> np.ndarray:
        """Hour-resolved prediction; a row without a time gets the mean over the 08–17 h mixture hours (as D1)."""
        si, ti, hour = np.asarray(si, int), np.asarray(ti, int), np.asarray(hour, "float64")
        out, t = np.empty(len(si)), np.isfinite(hour)
        if t.any():
            out[t] = self.model.predict(self.design(fs, si[t], ti[t], hour[t], np.zeros(int(t.sum()))))
        if (~t).any():
            out[~t] = self.mean_over(fs, si[~t], ti[~t], MIX_HOURS.astype("float64"))
        return out


# -------------------------------------------------------------------------------------------------- reference
def need_days(daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Days on which the derived outputs are needed at locations with labels ``daily`` (n, T, len(DAILY)): label days
    of a primary target and, for 7-DADMax, the 6 days before each of its label days."""
    f = lambda k: np.isfinite(daily[:, :, _J[k]])   # noqa: E731
    n7 = f("T_7dadm")
    nt = f("T_mean") | n7
    for k in range(1, 7):
        nt[:, :-k] |= n7[:, k:]
    return nt, f("DO_mean") | f("DO_min")


def _mse(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else float("nan")


class GBMRef:
    """One seed of G1 (``fs`` = frozen causal features) or G2 (``fs`` with physics and stacked inputs)."""

    def __init__(self, name: str, cfg: LGBMConfig, *, grid=GRID, seed: int = 0, fixed: dict | None = None):
        """``fixed`` = ``{"selected": {target: [form, cap, share]}, "selected_inst": {var: [cap, share]}}`` (the
        ``info`` of the seed-0 grid run): only those members are fitted and nothing is selected again (seeds 1–2 and
        variants, amendment §4 and §7)."""
        self.name, self.cfg, self.grid, self.seed = name, cfg, [(c, float(s)) for c, s in grid], seed
        self.fixed = None
        if fixed is not None:
            self.fixed = ({v: (s[0], s[1], None if s[2] is None else float(s[2])) for v, s in fixed["selected"].items()},
                          {v: (s[0], float(s[1])) for v, s in fixed["selected_inst"].items()})
        self.inst: dict = {}        # (var, cap, share) -> InstGBM
        self.amp: dict = {}         # (var, cap) -> DailyGBM
        self.direct: dict = {}      # (target, cap) -> DailyGBM
        self.sel: dict = {}         # primary target -> ("derived", cap, share) | ("direct", cap, None)
        self.sel_inst: dict = {}    # var -> (cap, share)
        self.info: dict = {}

    # -- labels
    @staticmethod
    def _amp_label(v2, var):
        up, lo, _ = AMP[var]
        return np.maximum(v2.daily[:, :, _J[up]].astype("float64") - v2.daily[:, :, _J[lo]], 0.0)

    def _derived(self, fs, idx, nt, nd, cap, share, targets=("T", "DO")) -> np.ndarray:
        """(n, T, 7) raw-unit derived outputs at ``idx`` on the needed days (NaN elsewhere)."""
        idx = np.asarray(idx, int)
        out = np.full(nt.shape + (7,), np.nan, dtype="float32")
        for var, need in (("T", nt), ("DO", nd)):
            a, t = np.nonzero(need)
            if var not in targets or not len(a) or (var, cap, share) not in self.inst or (var, cap) not in self.amp:
                continue
            mean = self.inst[(var, cap, share)].mean_over(fs, idx[a], t, HOURS24)
            amp = np.maximum(self.amp[(var, cap)].predict(fs, idx[a], t), 0.0)
            if var == "T":
                out[a, t, _J["T_mean"]], out[a, t, _J["T_max"]] = mean, mean + amp
                out[:, :, _J["T_7dadm"]] = D.trailing_mean_strict(out[:, :, _J["T_max"]].astype("float64"), 7)
            else:
                out[a, t, _J["DO_mean"]], out[a, t, _J["DO_min"]] = mean, mean - amp
        return out

    def _direct(self, fs, idx, daily, cap) -> np.ndarray:
        idx = np.asarray(idx, int)
        out = np.full(daily.shape[:2] + (7,), np.nan, dtype="float32")
        for k in PRIMARY.values():
            a, t = np.nonzero(np.isfinite(daily[:, :, _J[k]]))
            if len(a) and (k, cap) in self.direct:
                out[a, t, _J[k]] = self.direct[(k, cap)].predict(fs, idx[a], t)
        return out

    def _inst_pred(self, fs, v2, idx, cap, share) -> np.ndarray:
        idx = np.asarray(idx, int)
        out = np.full(v2.inst_val[idx].shape, np.nan, dtype="float32")
        for j, var in enumerate(INST):
            a, t = np.nonzero(np.isfinite(v2.inst_val[idx][:, :, j]))
            if len(a) and (var, cap, share) in self.inst:
                out[a, t, j] = self.inst[(var, cap, share)].at(fs, idx[a], t, v2.inst_hour[idx[a], t, j])
        return out

    # -- fit
    def fit(self, v2: V2Train, fs: FeatureSet, scales: Scales, *, log=print, cache: MemberCache | None = None) -> dict:
        """Fits every grid member (early-stopped on inner rows), selects per target, scores the inner criterion with
        the early-stopped members, then refits the selected members on fit + inner rows."""
        cols, d = fs.columns(), v2.data
        if self.fixed is None:
            w_inst = [(var, cap, share) for cap, share in self.grid for var in INST]
            w_amp = [(var, cap) for cap, _ in self.grid for var in INST]
            w_dir = [(k, cap) for cap, _ in self.grid for k in PRIMARY.values()]
            pairs, dcaps = list(self.grid), list(dict.fromkeys(c for c, _ in self.grid))
        else:
            fsel, finst = self.fixed
            der_sel = {v: s for v, s in fsel.items() if s[0] == "derived"}
            w_inst = [(PRIMARY[v].split("_")[0], s[1], s[2]) for v, s in der_sel.items()] + \
                     [(var, *g) for var, g in finst.items()]
            w_amp = [(PRIMARY[v].split("_")[0], s[1]) for v, s in der_sel.items()]
            w_dir = [(PRIMARY[v], s[1]) for v, s in fsel.items() if s[0] == "direct"]
            pairs = list(dict.fromkeys((s[1], s[2]) for s in der_sel.values()))
            dcaps = list(dict.fromkeys(c for _, c in w_dir))
            self.grid = list(dict.fromkeys(pairs + list(finst.values())))
        got = (lambda key, build: build()) if cache is None else cache.get
        for var, cap, share in dict.fromkeys(w_inst):
            self.inst[(var, cap, share)] = got(f"inst_{var}_{cap}_{share}", lambda: InstGBM(
                var, cols, self.cfg, self.seed).fit(v2, fs, cap=cap, share=share, log=log))
        for var, cap in dict.fromkeys(w_amp):
            self.amp[(var, cap)] = got(f"amp_{var}_{cap}", lambda: DailyGBM(
                f"amp {var} cap {cap}", cols, self.cfg, self.seed).fit(
                d, fs, self._amp_label(v2, var), v2.fit_idx, v2.inner_idx, cap=cap, log=log))
        for k, cap in dict.fromkeys(w_dir):
            self.direct[(k, cap)] = got(f"direct_{k}_{cap}", lambda: DailyGBM(
                f"direct {k} cap {cap}", cols, self.cfg, self.seed).fit(
                d, fs, v2.daily[:, :, _J[k]].astype("float64"), v2.fit_idx, v2.inner_idx, cap=cap, log=log))
        ii, gi = v2.inner_idx, v2.grab_inner_idx
        y = v2.daily[ii]
        nt, nd = need_days(y)
        der = {(cap, share): self._derived(fs, ii, nt, nd, cap, share) for cap, share in pairs}
        dir_ = {cap: self._direct(fs, ii, y, cap) for cap in dcaps}
        table, inner = {}, np.full(y.shape[:2] + (7,), np.nan, dtype="float32")
        for v, k in PRIMARY.items():
            cand = {("derived", cap, share): _mse(p[:, :, _J[k]], y[:, :, _J[k]]) for (cap, share), p in der.items()}
            cand.update({("direct", cap, None): _mse(p[:, :, _J[k]], y[:, :, _J[k]]) for cap, p in dir_.items()})
            ok = {c: m for c, m in cand.items() if np.isfinite(m)}
            table[v] = [{"form": c[0], "max_rows_per_site": c[1], "grab_share": c[2], "inner_mse": m}
                        for c, m in cand.items()]
            if not ok:
                continue
            if self.fixed is None:
                self.sel[v] = min(ok, key=ok.get)                  # ties: the earlier candidate (derived first, grid order)
            elif self.fixed[0].get(v) in ok:
                self.sel[v] = self.fixed[0][v]
            else:
                raise RuntimeError(f"{self.name}: the fixed selection of {v} has no finite inner prediction")
            src = der[self.sel[v][1:]] if self.sel[v][0] == "derived" else dir_[self.sel[v][1]]
            inner[:, :, _J[k]] = src[:, :, _J[k]]
        out = {"daily": inner}
        mean, std = scales.label_mean[:7], scales.label_std[:7]
        js = C.j_sonde((inner - mean) / std, (y[:, :, :7] - mean) / std, scales.crit_factor())
        jg, grab_table = None, {}
        if len(gi):
            jm = [_J["T_mean"], _J["DO_mean"]]
            yi = (v2.inst_val[gi] - scales.label_mean[jm]) / scales.label_std[jm]
            pin = {g: self._inst_pred(fs, v2, gi, *g) for g in dict.fromkeys((c, s) for _, c, s in self.inst)}
            best = np.full(yi.shape, np.nan, dtype="float32")
            for j, var in enumerate(INST):
                sc = {g: C.j_grab((p - scales.label_mean[jm]) / scales.label_std[jm], yi, variables=(var,))["j_grab"]
                      for g, p in pin.items()}
                grab_table[var] = [{"max_rows_per_site": g[0], "grab_share": g[1], "inner_grab_mse_std": s}
                                   for g, s in sc.items()]
                ok = {g: s for g, s in sc.items() if s is not None and (var, *g) in self.inst}
                if ok:
                    self.sel_inst[var] = min(ok, key=ok.get) if self.fixed is None else self.fixed[1][var]
                    best[:, :, j] = pin[self.sel_inst[var]][:, :, j]
            out["inst"] = best
            jg = C.j_grab((best - scales.label_mean[jm]) / scales.label_std[jm], yi)
        self.info = {"name": self.name, "seed": self.seed, "grid": [list(g) for g in self.grid],
                     "fixed_selection": self.fixed is not None,
                     "selected": {v: list(s) for v, s in self.sel.items()},
                     "selected_inst": {v: list(s) for v, s in self.sel_inst.items()}, "candidates": table,
                     "grab_candidates": grab_table, "J": C.combine(js, jg), "j_sonde": js["j_sonde"],
                     "by_target": js["by_target"], "n_cells": js["n_cells"],
                     "j_grab": None if jg is None else jg["j_grab"],
                     "grab_by_variable": None if jg is None else jg["by_variable"],
                     "members": {"inst": {f"{k[0]}|{k[1]}|{k[2]}": m.info for k, m in self.inst.items()},
                                 "amp": {f"{k[0]}|{k[1]}": m.info for k, m in self.amp.items()},
                                 "direct": {f"{k[0]}|{k[1]}": m.info for k, m in self.direct.items()}},
                     "n_columns": len(cols) + N_EXTRA}
        log(f"[{self.name}] selected {self.info['selected']}; hour-resolved {self.info['selected_inst']}; "
            f"J {self.info['J']} (sonde {self.info['j_sonde']}, grab {self.info['j_grab']})")
        self._refit(v2, fs, log, got)
        return out

    def _refit(self, v2, fs, log, got=lambda key, build: build()) -> None:
        """Keeps and refits only the selected members (fit + inner rows, selected iteration count)."""
        keep_i, keep_a, keep_d = set(), set(), set()
        for v, s in self.sel.items():
            var = PRIMARY[v].split("_")[0]
            if s[0] == "derived":
                keep_i.add((var, s[1], s[2]))
                keep_a.add((var, s[1]))
            else:
                keep_d.add((PRIMARY[v], s[1]))
        keep_i |= {(var, *g) for var, g in self.sel_inst.items()}
        self.inst = {k: got(f"refit_inst_{k[0]}_{k[1]}_{k[2]}", lambda m=m: m.refit(v2, fs, log))
                     for k, m in self.inst.items() if k in keep_i}
        self.amp = {k: got(f"refit_amp_{k[0]}_{k[1]}", lambda k=k, m=m: m.refit(
            v2.data, fs, self._amp_label(v2, k[0]), v2.fit_idx, v2.inner_idx, log))
            for k, m in self.amp.items() if k in keep_a}
        self.direct = {k: got(f"refit_direct_{k[0]}_{k[1]}", lambda k=k, m=m: m.refit(
            v2.data, fs, v2.daily[:, :, _J[k[0]]].astype("float64"), v2.fit_idx, v2.inner_idx, log))
            for k, m in self.direct.items() if k in keep_d}

    # -- prediction with the refitted members
    def predict_daily(self, fs: FeatureSet, idx, need_t: np.ndarray, need_do: np.ndarray) -> np.ndarray:
        """(n, T, 7) raw-unit predictions of the four primary targets at ``idx``. ``need_t`` / ``need_do`` (n, T) are
        the days to predict (7-DADMax needs its 6 preceding days in ``need_t``)."""
        idx = np.asarray(idx, int)
        out = np.full(need_t.shape + (7,), np.nan, dtype="float32")
        cache: dict = {}
        for v, s in self.sel.items():
            k, var = PRIMARY[v], PRIMARY[v].split("_")[0]
            if s[0] == "derived":
                key = (var, s[1], s[2])
                if key not in cache:
                    cache[key] = self._derived(fs, idx, need_t, need_do, s[1], s[2], targets=(var,))
                out[:, :, _J[k]] = cache[key][:, :, _J[k]]
            else:
                a, t = np.nonzero(need_t if var == "T" else need_do)
                out[a, t, _J[k]] = self.direct[(k, s[1])].predict(fs, idx[a], t)
        return out

    def predict_inst(self, fs: FeatureSet, var: str, si, ti, hour) -> np.ndarray:
        return self.inst[(var, *self.sel_inst[var])].at(fs, si, ti, hour)


SCORE_KEYS = ("inner_rmse", "inner_mse", "inner_grab_mse_std", "inner_rmse_logq", "J", "j_sonde", "j_grab", "by_target",
              "grab_by_variable", "candidates", "grab_candidates")


def strip_scores(obj):
    """A copy of a run record without any inner score (real-data smoke runs keep timing and row counts only)."""
    if isinstance(obj, dict):
        return {k: strip_scores(v) for k, v in obj.items() if k not in SCORE_KEYS}
    return [strip_scores(v) for v in obj] if isinstance(obj, (list, tuple)) else obj


def columns_sha256(columns) -> str:
    import hashlib

    return hashlib.sha256("\n".join(columns).encode()).hexdigest()


def lgbm_config(d: dict, n_jobs: int = 6) -> LGBMConfig:
    f = {x.name for x in dataclasses.fields(LGBMConfig)}
    return LGBMConfig(**{**{k: v for k, v in (d or {}).items() if k in f}, "n_jobs": n_jobs})


def no_grab_selection(sel: dict) -> dict:
    """The no-grab variant of a selection (gate G-D1a): same form and ``max_rows_per_site`` per target, grab share 0.
    Nothing is selected again."""
    return {"selected": {v: [s[0], s[1], None if s[2] is None else 0.0] for v, s in sel["selected"].items()},
            "selected_inst": {v: [s[0], 0.0] for v, s in sel["selected_inst"].items()}}
