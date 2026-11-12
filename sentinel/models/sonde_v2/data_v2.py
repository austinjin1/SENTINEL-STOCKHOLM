# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tensors for SONDE-HF / R-NH (``sonde_arch_v2_amendment.md`` §2, lead rulings L3).

* Daily hindcast inputs, statics and standardised-anomaly targets come from the frozen-candidate
  :class:`sentinel.models.sonde.lstm.SondeTensors` (imported, not edited), with ``horizons = 1..L`` so every lead has
  a target and the split's embargo keeps every target inside the issue block.
* Per-lead targets ``y[b, k, v]`` = standardised anomaly of variable v on day t + k + 1 (k = 0..L−1).
* NWP: exactly ONE declared product (GEFS c00; lead ruling G(b)), one tensor (S, T, L_p, F) per use ("train" =
  perfect-prog observed weather + error model; "real" = archived forecasts for stage 2; "eval" = archived issued
  forecasts). Standardised with the finite TRAIN-block values of the fit sites of the first "train" registration;
  missing → masked. Registering a second product is a hard error.
* Cross-site constants (scale floors, weather / static standardisation) come from the fit sites (``stat_sites``).
* Reference-window issue rule (lead ruling F): a target of variable v at issue day t is a label only if
  t ≥ ``data.ref_issue_start[s, v]``; :meth:`V2Tensors.labelled` gives the usable training rows per target set.
* Deterministic future inputs per lead: clear-sky shortwave (0.75 × extraterrestrial), day length, DOY sin/cos.
* Optional hourly cube (hourly-tier sites only; Approved values; local standard time; last 14 d × 24 h before the
  end of the issue day): per variable a site-standardised level and an observed mask, plus hour-of-day sin/cos.
  Reserved (new-site) sites are excluded through ``sentinel.data.reserved``.
* Inner splits: :func:`basin_folds` (5-fold basin-group split of a site set, fixed seed), used for the step-1 test,
  inner early stopping (15 % of train-role basin groups) and CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.reserved import exclude_reserved
from sentinel.models.sonde.lstm import SondeTensors

from . import nwp as NWP

HOURLY_VARS = ("temp_c_mean", "do_mgl_mean", "spcond_uscm_mean", "ph_median", "turb_fnu_mean")
HOURLY_COLS = {"temp_c_mean": ("temp_c_mean", "temp_c_prov"), "do_mgl_mean": ("do_mgl_mean", "do_mgl_prov"),
               "spcond_uscm_mean": ("spcond_uscm_mean", "spcond_uscm_prov"), "ph_median": ("ph_mean", "ph_prov"),
               "turb_fnu_mean": ("turb_fnu_mean", "turb_fnu_prov")}
H_HOURS = 14 * 24


def basin_folds(sites_idx: np.ndarray, basin: np.ndarray, k: int = 5, seed: int = 20261110) -> np.ndarray:
    """fold id (0..k−1) per site in ``sites_idx``: whole basin groups assigned round-robin after a seeded shuffle."""
    groups = np.unique(basin[sites_idx])
    rs = np.random.default_rng(seed)
    order = rs.permutation(groups)
    gf = {g: i % k for i, g in enumerate(order)}
    return np.array([gf[b] for b in basin[sites_idx]], dtype=np.int64)


def clear_sky_sw(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Daily mean clear-sky shortwave (W/m²) ≈ 0.75 × extraterrestrial (FAO-56 Ra), lat (n,), doy (n, L)."""
    phi = np.deg2rad(lat_deg)[:, None]
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.75 * ra * 1e6 / 86400                       # MJ m-2 d-1 → W m-2


def day_length(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(lat_deg)[:, None]
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    return 24 / np.pi * np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))


class V2Tensors:
    def __init__(self, data, *, variables, L: int = 14, use_weather: bool = True, frozen: dict | None = None,
                 hourly: bool = False, utc_offset: dict | None = None, stat_sites: np.ndarray | None = None):
        """``stat_sites``: the fit sites. The cross-site constants of the frozen tensors (variable scale floors,
        weather and static standardisation) are then computed from these sites only (lead ruling L3), through a
        role proxy handed to the unedited ``SondeTensors``; without it they come from the train-role sites."""
        self.data = data
        self.L = L
        src, roles = data, ("train",)
        if stat_sites is not None and frozen is None:
            from copy import copy
            src = copy(data)
            src.roles = np.where(np.isin(np.arange(len(data.sites)), np.asarray(stat_sites)), "__fit__", "__other__")
            roles = ("__fit__",)
        self.st = SondeTensors(src, variables=variables, horizons=tuple(range(1, L + 1)), context=365,
                               frozen=frozen, use_weather=use_weather, train_roles=roles)
        self.stat_sites = None if stat_sites is None else np.asarray(stat_sites)
        self.variables = self.st.variables
        lat = data.static["lat"].to_numpy(float)
        self.lat = np.where(np.isfinite(lat), lat, 40.0)
        self.products: dict[str, np.ndarray] = {}
        self.nwp_stats: tuple[np.ndarray, np.ndarray] | None = None
        cube = data.cube
        self.issue_start = np.stack([data.ref_issue_start[:, cube.var_index(v)] for v in self.variables], axis=1)
        self._lab: dict[str, np.ndarray] = {}
        self.hourly = None
        if hourly:
            self.hourly, self.hourly_row = load_hourly_cube(data, self.st, utc_offset or {})

    # ------------------------------------------------------------------ NWP
    def set_products(self, use: str, product: np.ndarray, *, stat_sites: np.ndarray | None = None) -> None:
        """Register THE product tensor (S, T, L_p, F) for a use ("train" | "real" | "eval"). Standardisation
        statistics come from the finite train-block values of ``stat_sites`` (the fit sites) in the first "train"
        registration and are reused for every use."""
        if not isinstance(product, np.ndarray) or product.ndim != 4 or product.shape[-1] != len(NWP.FIELDS):
            raise ValueError("exactly one declared NWP product (S, T, L, F) per use; a list or a second product "
                             "is not allowed in this round")
        pi = NWP.F_IDX["precip_mm"]
        if self.nwp_stats is None:
            if use != "train" or stat_sites is None:
                raise ValueError("register the train product first, with stat_sites (standardisation from the "
                                 "train block of the fit sites)")
            te = int(np.searchsorted(self.data.cube.dates, pd.Timestamp("2021-09-30"), side="right"))
            x = product[np.asarray(stat_sites)][:, :te].reshape(-1, product.shape[-1]).astype(np.float64)
            x[:, pi] = np.log1p(np.maximum(x[:, pi], 0))
            n = np.isfinite(x).sum(0)
            if (n < 1000).any():
                raise RuntimeError(f"NWP product has < 1000 finite train-block values for a field: {n.tolist()}")
            self.nwp_stats = (np.nanmean(x, 0), np.nanstd(x, 0) + 1e-6)
            self.nwp_stat_n = n.tolist()
        mu, sd = self.nwp_stats
        q = product.astype(np.float32).copy()
        q[..., pi] = np.log1p(np.maximum(q[..., pi], 0))
        self.products[use] = ((q - mu) / sd).astype(np.float16)

    # ------------------------------------------------------------------ rows
    def labelled(self, variables=None) -> np.ndarray:
        """(S, T) bool: issue day t at site s has ≥ 1 finite target on days t+1..t+L for one of ``variables``
        (default: all tensor variables) AND t is allowed by the reference-window issue rule for that variable."""
        out = None
        for v in (variables or self.variables):
            if v not in self._lab:
                k = self.variables.index(v)
                T = len(self.data.cube.dates)
                fin = np.isfinite(self.st.targets_padded[:, :, k])
                c = np.concatenate([np.zeros((fin.shape[0], 1), np.int32), np.cumsum(fin, axis=1, dtype=np.int32)], 1)
                t = np.arange(T)
                has = (c[:, t + self.L + 1] - c[:, t + 1]) > 0                     # targets on t+1..t+L
                self._lab[v] = has & (t[None, :] >= self.issue_start[:, k][:, None])
            out = self._lab[v].copy() if out is None else out | self._lab[v]
        return out

    def train_rows(self, sites: np.ndarray, block: str = "train", variables=None):
        """(si, ti) training / early-stopping rows of ``sites``: issue days of ``block`` with every lead inside
        the block (frozen ``issue_idx(block, L)``), a recent observation (frozen eligibility) and a label."""
        t_idx = self.data.issue_idx(block, self.L)
        sites = np.asarray(sites)
        m = (self.st.eligible & self.labelled(variables))[np.ix_(sites, t_idx)]
        a, b = np.nonzero(m)
        return sites[a].astype(np.int64), t_idx[b].astype(np.int64)

    # ------------------------------------------------------------------ batches
    def batch(self, si: np.ndarray, ti: np.ndarray, use: str = "train") -> dict:
        st, L, C = self.st, self.L, self.st.context
        win = ti[:, None] + np.arange(-C + 1, 1)[None, :]
        x_dyn = st.dyn[si[:, None], np.clip(win, 0, None)].astype(np.float32)
        x_dyn[win < 0] = 0.0
        lead_t = ti[:, None] + np.arange(1, L + 1)[None, :]
        y = st.targets_padded[si[:, None], lead_t]                                   # (B, L, V)
        y = np.where((ti[:, None] < self.issue_start[si])[:, None, :], np.nan, y)    # reference-window issue rule
        doy_t = (pd.DatetimeIndex(self.data.cube.dates[ti]).dayofyear.to_numpy()[:, None] - 1
                 + np.arange(1, L + 1)[None, :]) % 365
        csw = clear_sky_sw(self.lat[si], doy_t)
        dl = day_length(self.lat[si], doy_t)
        ang = 2 * np.pi * doy_t / 365.25
        fdet = np.stack([csw / 300.0, dl / 12.0, np.sin(ang), np.cos(ang)], -1).astype(np.float32)
        cm = st.cm[si[:, None], doy_t]                                               # (B, L, V)
        cs = st.cs_eff[si[:, None], doy_t]
        # last observed anomaly within 4 days (same rule as the frozen LSTM anchor)
        V = len(self.variables)
        a = st.dyn[si[:, None], np.clip(ti[:, None] + np.arange(-3, 1)[None, :], 0, None)].astype(np.float32)
        an, mk = a[..., :V], a[..., 2 * V:3 * V] > 0.5
        pos = np.where(mk, np.arange(4)[None, :, None], -1).max(1)
        a_last = np.where(pos >= 0, np.take_along_axis(an, np.clip(pos, 0, None)[:, None, :], 1)[:, 0], 0.0)
        if use not in self.products:
            raise KeyError(f"no NWP product registered for use {use!r}")
        p = self.products[use]
        Lp = p.shape[2]
        x = np.full((si.size, L, p.shape[-1]), np.nan, np.float32)
        x[:, :min(L, Lp)] = p[si, ti, :min(L, Lp)]
        fin = np.isfinite(x)
        # a lead is available when its temperature is; other missing fields enter as 0 (= train mean) with a
        # per-field availability flag. Lists of ONE entry: the network interface keeps its product axis.
        nwp_x = [np.concatenate([np.where(fin, x, 0.0), fin.astype(np.float32)], -1)]
        nwp_m = [fin[..., NWP.F_IDX["tmean_c"]]]
        out = {"x_dyn": x_dyn, "x_static": st.static[si], "huc": st.huc[si], "y": y.astype(np.float32),
               "x_fdet": fdet, "a_last": np.nan_to_num(a_last).astype(np.float32),
               "clim_mean": np.nan_to_num(cm, nan=0.0).astype(np.float32),
               "clim_sd": np.nan_to_num(cs, nan=1.0).astype(np.float32), "nwp_x": nwp_x, "nwp_m": nwp_m,
               "clim_ok": np.isfinite(cm) & np.isfinite(cs)}
        if self.hourly is not None:
            r = self.hourly_row[si]
            has = r >= 0
            xh = np.zeros((si.size, H_HOURS, self.hourly.shape[-1]), np.float32)
            if has.any():
                hw = (ti[has, None] + 1) * 24 + np.arange(-H_HOURS, 0)[None, :]
                xh[has] = self.hourly[r[has, None], np.clip(hw, 0, None)].astype(np.float32)
            out["x_hourly"] = xh
            out["has_hourly"] = has & (np.abs(xh).sum(axis=(1, 2)) > 0)
        return out


def load_hourly_cube(data, st: SondeTensors, utc_offset: dict) -> tuple[np.ndarray, np.ndarray]:
    """(n_hourly_sites, T·24, 2·V_h + 2) float16 and a (S,) row map (−1 = no hourly table)."""
    from sentinel.models.sonde.subdaily import HOURLY_DIR

    sites = list(data.cube.sites)
    keep = set(exclude_reserved([s for s in sites if (HOURLY_DIR / f"{s}.parquet").exists()]))
    rows = np.full(len(sites), -1, dtype=np.int64)
    T = len(data.cube.dates)
    d0 = data.cube.dates[0]
    Vh = len(HOURLY_VARS)
    hrs = np.arange(T * 24)
    hod = hrs % 24
    base = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], -1).astype(np.float16)
    cubes = []
    for i, s in enumerate(sites):
        if s not in keep:
            continue
        df = pd.read_parquet(HOURLY_DIR / f"{s}.parquet")
        t = pd.DatetimeIndex(df.index).tz_convert("UTC").tz_localize(None)
        local = t + pd.Timedelta(hours=float(utc_offset.get(s, -6.0)))
        hi = ((local - d0) / pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
        ok = (hi >= 0) & (hi < T * 24)
        x = np.zeros((T * 24, 2 * Vh + 2), np.float16)
        x[:, 2 * Vh:] = base
        for j, v in enumerate(HOURLY_VARS):
            col, prov = HOURLY_COLS[v]
            if col not in df or v not in st.variables:
                continue
            k = st.variables.index(v)
            val = df[col].to_numpy(float)
            good = ok & np.isfinite(val) & ~(df[prov].to_numpy(bool) if prov in df else np.zeros(len(df), bool))
            from sentinel.models.sonde.data import to_model_space
            z = to_model_space(v, val[good])
            lvl = np.clip((z - st.m_site[i, k]) / st.s_site[i, k], -8, 8)
            x[hi[good], j] = np.nan_to_num(lvl)
            x[hi[good], Vh + j] = 1.0
        rows[i] = len(cubes)
        cubes.append(x)
    cube = np.stack(cubes) if cubes else np.zeros((0, T * 24, 2 * Vh + 2), np.float16)
    return cube, rows
