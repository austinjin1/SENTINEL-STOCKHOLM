# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Physics-family parity comparators for the M6 SOTA levers (``hydromet_sota_protocol.md`` §5).

* :func:`extend_air2stream` — ``air2stream_regional_ws`` on the lever data: the reference donors
  (``basews_v1``) plus every added fit site with ≥ 365 train-block daily temperatures, calibrated with the identical
  settings. Per-site calibration is separable (Adam is per parameter and the loss is a mean of per-site losses), so
  calibrating only the added sites equals recalibrating all; donor-attribute standardisation is recomputed.
* :class:`WeightedDOSatRatio` — ``dosat_ratio[...]``: pooled monthly median of DO/DO_sat(T_obs) over fit + inner
  rows; weak-label rows (same-day grab temperature and DO) enter with the configuration's weak share via a weighted
  median.
* :class:`Air2StreamQ` — lever 2: the air2stream update divided by θ^a4, θ = Q̂/mean(Q̂) from the discharge GBM
  (Toffolon & Piccolroaz 2015 flow scaling; a4 ∈ [0, 2]); calibrated per donor with the out-of-fold Q̂.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from ..models.hydromet.baselines import Air2StreamRegional, DOSatRatio, check_fit_sites
from ..models.hydromet.data import HydroMetData
from ..models.hydromet.physics import A2S_BOUNDS, A2S_PARAMS, _ffill, do_sat_mgl, pressure_atm_from_elevation

Q_BOUNDS = {**A2S_BOUNDS, "a4": (0.0, 2.0)}
Q_PARAMS = A2S_PARAMS + ("a4",)


def extend_air2stream(ref: Air2StreamRegional, data: HydroMetData, fs, new_idx, *, name: str, log=print):
    m = copy.deepcopy(ref)
    m.name = name
    new_idx = np.asarray(new_idx, int)
    cand = np.asarray([i for i in new_idx if data.target_mask("temp_c_mean", "train", [i]).sum() >= m.min_obs], int)
    added = 0
    if len(cand):
        cal = m.calibrate(data, fs, cand, log=log)
        good = np.isfinite(cal["nse"]) & (cal["nse"] >= m.min_nse)
        X_new = m._attr_matrix(data, cand[good])
        # raw donor attributes of the reference donors (inverse of the stored standardisation)
        X_ref = m.donor_x * m.attr_std + m.attr_mean
        X = np.vstack([X_ref, np.where(np.isfinite(X_new), X_new, np.nan)])
        m.donor_sites = list(m.donor_sites) + [data.sites[i] for i in cand[good]]
        m.donor_params = {k: np.r_[m.donor_params[k], cal["params"][k][good]] for k in m.donor_params}
        with np.errstate(invalid="ignore"):
            m.attr_mean = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
        m.attr_std = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        m.donor_x = np.nan_to_num((X - m.attr_mean) / m.attr_std, nan=0.0)
        added = int(good.sum())
    m.info = {**ref.info, "n_added_candidates": int(len(cand)), "n_added_donors": added,
              "n_donors": len(m.donor_sites), "note": "reference donors (basews_v1) + added lever sites"}
    log(f"[{name}] {added} of {len(cand)} added sites became donors; {len(m.donor_sites)} donors")
    return m


def _wmedian(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


class WeightedDOSatRatio(DOSatRatio):
    def fit(self, data, fs, fit_idx, inner_idx, *, weak_idx=None, weak_share: float = 0.0, log=print):
        weak_idx = np.asarray([] if weak_idx is None else weak_idx, int)
        check_fit_sites(data, fit_idx, inner_idx, weak_idx)
        month = data.dates.month.to_numpy()
        tm = data.block_mask("train")

        def ratios(idx, v):
            p = pressure_atm_from_elevation(data.static["elev_m"].to_numpy())[idx][:, None]
            tw = data.y[idx, :, data.vi("temp_c_mean")].astype("float64")
            sat = np.where(np.isfinite(tw), do_sat_mgl(np.nan_to_num(tw), p), np.nan)
            r = data.y[idx, :, data.vi(v)] / sat
            r[:, ~tm] = np.nan
            return r

        dense = np.r_[fit_idx, inner_idx].astype(int)
        for v in self.targets:
            rd = ratios(dense, v)
            rw = ratios(weak_idx, v) if (len(weak_idx) and weak_share > 0 and v == "do_mgl_mean") else None
            out = np.full(13, np.nan)
            for mo in range(1, 13):
                xd = rd[:, month == mo]
                xd = xd[np.isfinite(xd) & (xd > 0) & (xd < 3)]
                xw = np.zeros(0)
                if rw is not None:
                    xw = rw[:, month == mo]
                    xw = xw[np.isfinite(xw) & (xw > 0) & (xw < 3)]
                if xd.size + xw.size < 30:
                    continue
                if xw.size and xd.size:
                    ww = weak_share / (1 - weak_share) * xd.size / xw.size
                    out[mo] = _wmedian(np.r_[xd, xw], np.r_[np.ones(xd.size), np.full(xw.size, ww)])
                else:
                    out[mo] = float(np.median(xd if xd.size else xw))
            out[0] = np.nanmedian(out[1:]) if np.isfinite(out[1:]).any() else 1.0
            self.ratio[v] = np.where(np.isfinite(out), out, out[0])
            self.info[v] = {"ratio_by_month": np.round(self.ratio[v][1:], 4).tolist(),
                            "weak_share": weak_share if rw is not None else 0.0}
            log(f"[{self.name}] {v}: monthly DO/DO_sat ratio {np.round(self.ratio[v][1:], 3).tolist()}")
        return self


def air2stream_q_simulate(ta, theta, doy, params, tw0=None):
    """air2stream with flow scaling: Tw[t] = max(Tw[t−1] + (a1 + a2·Ta − a3·Tw[t−1] + a5·cos(2π(doy/365.25 − a6)))
    / θ[t]^a4, 0). numpy or torch; causal."""
    try:
        import torch

        is_t = isinstance(ta, torch.Tensor)
    except ImportError:   # pragma: no cover
        is_t = False
    S, T = ta.shape
    a1, a2, a3, a5, a6, a4 = (params[k] for k in Q_PARAMS)
    if not is_t:
        ta = _ffill(ta)
        season = np.cos(2 * np.pi * (np.asarray(doy, float)[None, :] / 365.25 - np.asarray(a6)[:, None]))
        scale = np.power(np.clip(theta, 1e-3, 1e3), -np.asarray(a4)[:, None])
        tw = np.empty((S, T))
        prev = np.maximum(ta[:, :30].mean(axis=1), 0.0) if tw0 is None else np.asarray(tw0, float)
        for t in range(T):
            prev = np.maximum(prev + (a1 + a2 * ta[:, t] - a3 * prev + a5 * season[:, t]) * scale[:, t], 0.0)
            tw[:, t] = prev
        return tw
    doy_t = torch.as_tensor(np.asarray(doy, float), dtype=ta.dtype, device=ta.device)
    season = torch.cos(2 * math.pi * (doy_t[None, :] / 365.25 - a6[:, None]))
    scale = torch.pow(theta.clamp(1e-3, 1e3), -a4[:, None])
    prev = ta[:, :30].mean(dim=1).clamp(min=0.0) if tw0 is None else tw0
    out = []
    for t in range(T):
        prev = torch.relu(prev + (a1 + a2 * ta[:, t] - a3 * prev + a5 * season[:, t]) * scale[:, t])
        out.append(prev)
    return torch.stack(out, dim=1)


class Air2StreamQ(Air2StreamRegional):
    """``air2stream_regional`` with the θ^a4 flow scaling driven by the stacked discharge prediction Q̂."""

    name = "air2stream_q_regional"

    def __init__(self, qhat_log: np.ndarray, **kw):
        super().__init__(**kw)
        self.set_qhat(qhat_log)

    def set_qhat(self, qhat_log):
        q = np.exp(np.asarray(qhat_log, "float64")) - 0.01
        q = np.maximum(q, 1e-3)
        self._theta = q / np.nanmean(q, axis=1, keepdims=True)
        self._theta = np.where(np.isfinite(self._theta), self._theta, 1.0)

    def calibrate(self, data, fs, site_idx, log=print) -> dict:
        import torch

        from ..models.hydromet.baselines import _air_temp

        site_idx = np.asarray(site_idx, int)
        train_end = int(np.flatnonzero(data.block_mask("train"))[-1]) + 1
        ta = _ffill(_air_temp(fs, site_idx)[:, :train_end])
        th = self._theta[site_idx, :train_end]
        mask = data.target_mask("temp_c_mean", "train", None)[site_idx, :train_end]
        yobs = np.where(mask, data.y[site_idx, :train_end, data.vi("temp_c_mean")], 0.0)
        dev = torch.device(self.device)
        T_ta = torch.tensor(ta, dtype=torch.float32, device=dev)
        T_th = torch.tensor(th, dtype=torch.float32, device=dev)
        T_y = torch.tensor(yobs, dtype=torch.float32, device=dev)
        T_m = torch.tensor(mask, dtype=torch.float32, device=dev)
        init = {"a1": 0.3, "a2": 0.1, "a3": 0.1, "a5": 0.2, "a6": 0.6, "a4": 0.5}
        raw = {}
        for k in Q_PARAMS:
            lo, hi = Q_BOUNDS[k]
            p = (init[k] - lo) / (hi - lo)
            raw[k] = torch.full((len(site_idx),), float(np.log(p / (1 - p))), device=dev, requires_grad=True)
        opt = torch.optim.Adam(list(raw.values()), lr=self.lr)
        doy = data.dates.dayofyear.to_numpy()[:train_end]
        n_obs = T_m.sum(dim=1).clamp(min=1.0)
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            for it in range(self.iters):
                opt.zero_grad()
                params = {k: Q_BOUNDS[k][0] + (Q_BOUNDS[k][1] - Q_BOUNDS[k][0]) * torch.sigmoid(raw[k]) for k in Q_PARAMS}
                sim = air2stream_q_simulate(T_ta, T_th, doy, params)
                loss = ((((sim - T_y) ** 2) * T_m).sum(dim=1) / n_obs).mean()
                loss.backward()
                opt.step()
                if it % 100 == 0 or it == self.iters - 1:
                    log(f"[{self.name}] calibration iter {it}: mean site MSE {float(loss):.3f}")
        finally:
            torch.set_num_threads(prev_threads)
        with torch.no_grad():
            params = {k: (Q_BOUNDS[k][0] + (Q_BOUNDS[k][1] - Q_BOUNDS[k][0]) * torch.sigmoid(raw[k])).cpu().numpy()
                      .astype("float64") for k in Q_PARAMS}
        sim = air2stream_q_simulate(ta, th, doy, params)
        nse = np.full(len(site_idx), np.nan)
        for i in range(len(site_idx)):
            mm = mask[i]
            if mm.sum() >= 30:
                yy = yobs[i, mm]
                nse[i] = 1 - np.sum((sim[i, mm] - yy) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-9)
        return {"params": params, "nse": nse, "n_obs": mask.sum(axis=1)}

    def predict_full(self, data, fs, site_idx) -> np.ndarray:
        from ..models.hydromet.baselines import _air_temp

        site_idx = np.asarray(site_idx, int)
        Xq = np.nan_to_num((self._attr_matrix(data, site_idx) - self.attr_mean) / self.attr_std, nan=0.0)
        d = np.sqrt(((Xq[:, None, :] - self.donor_x[None, :, :]) ** 2).sum(-1))
        own = np.array([[s == t for t in self.donor_sites] for s in (data.sites[i] for i in site_idx)],
                       dtype=bool).reshape(len(site_idx), len(self.donor_sites))
        d[own] = np.inf
        k = min(self.k, self.donor_x.shape[0])
        nn = np.argsort(d, axis=1)[:, :k]
        w = 1.0 / (np.take_along_axis(d, nn, axis=1) + 0.1)
        w = w / w.sum(axis=1, keepdims=True)
        ta = np.repeat(_air_temp(fs, site_idx), k, axis=0)
        th = np.repeat(self._theta[site_idx], k, axis=0)
        params = {p: self.donor_params[p][nn.ravel()] for p in Q_PARAMS}
        sim = air2stream_q_simulate(ta, th, data.dates.dayofyear.to_numpy(), params).reshape(len(site_idx), k, -1)
        return (sim * w[:, :, None]).sum(axis=1)
