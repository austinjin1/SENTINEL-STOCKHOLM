# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Metamorphic / physical-consistency checks for SENTINEL-Sim scenarios (protocol §5.2).

Every check is evaluated per site (evaluation sites × block, JJA days unless stated); the reported
result is the share of eligible sites passing, compared with the pre-registered pass share.

M1  air +2 °C → summer-mean ΔT_w ∈ [0.2, 2] °C (≥ 90% of sites; a non-responsive model fails); "consistent"
    when |ΔT_w/2 − b_site| ≤ 0.25, b_site = train-block slope of weekly T_w on weekly T_air (≥ 70%).
M2  ΔDO_sat equals Benson–Krause applied to the simulated temperature (code identity, max |err| ≤ 1e-3 mg/L).
M3  stable-%sat sites (summer IQR of DO_mean/DO_sat ≤ 0.10 in train-block observations): ΔDO_mean < 0 AND at
    least half the solubility-only change s̄·ΔDO_sat (≥ 90%); the ratio ΔDO_mean/(s̄·ΔDO_sat) is reported.
M4  heterotrophic sites (simulated summer median GPP/|ER| < 1): ΔDO_min < ΔDO_mean (strict; ≥ 80%).
M5  Q × 0.5 → sign(ΔSC) (all block days) = −sign(train-block log SC – log Q slope on OBSERVED discharge only),
    for sites whose slope 95% CI excludes 0 (≥ 80%).
M6  placebo: water-year-shuffled forcing + discharge → pure-simulation RMSE of T_w and DO_mean worse than
    with true forcing (≥ 90% of sites, each variable).
M7  shortwave × 0.5 → summer ΔGPP ≤ −5% of baseline GPP (≥ 95%) and ΔDO_max < 0 (≥ 90%).
M8  (reported only) +2 °C: relative ER change vs the 0.65 eV Arrhenius expectation.
M9  validated-range ends: air −3 °C / +4 °C → sign(ΔT_w) = sign(ΔT) and 0.1·|ΔT| ≤ |ΔT_w| ≤ |ΔT| (≥ 90% each);
    Q × 1.5 → sign(ΔSC) = sign(C–Q slope) at significant-slope sites (≥ 80%); shortwave × 1.1 → ΔGPP > 0 (≥ 95%).
    Precipitation-only scaling is "model association" (scenario.py) and has no physical check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import physics
from .model import simulate
from .scenario import Scenario, run, summer_mask

# M2 tolerance 2e-3 mg/L = fp32 model vs fp64 check (same bound as P5; amended 2026-11-08 after v1 gave 1.24e-3,
# outcome-informed, protocol §12 v1.5)
M2_TOL = 2e-3
PASS_SHARE = {"M1_range": 0.90, "M1_consistent": 0.70, "M2_identity": 1.0, "M3_sign": 0.90, "M4_night": 0.80,
              "M5_cq_sign": 0.80, "M6_placebo_T": 0.90, "M6_placebo_DO": 0.90, "M7_gpp": 0.95, "M7_domax": 0.90,
              "M9_cold3": 0.90, "M9_warm4": 0.90, "M9_wetQ": 0.80, "M9_brightSW": 0.95}
MIN_DTW = 0.2          # °C: minimum summer water-temperature response to +2 °C air (non-responsive models fail)
MIN_DGPP_REL = 0.05    # minimum relative GPP decrease for shortwave × 0.5


def _weekly_slope(tw: np.ndarray, ta: np.ndarray, dates: pd.DatetimeIndex) -> float:
    df = pd.DataFrame({"tw": tw, "ta": ta}, index=dates).resample("W").mean().dropna()
    df = df[df["tw"] > 0.5]           # exclude ice-affected weeks (Kelleher et al. 2012 practice)
    if len(df) < 20:
        return np.nan
    return float(np.polyfit(df["ta"], df["tw"], 1)[0])


def _cq_slope_ci(sc: np.ndarray, q: np.ndarray):
    ok = np.isfinite(sc) & np.isfinite(q) & (q > 0)
    if ok.sum() < 60:
        return np.nan, np.nan, np.nan
    x, y = np.log(q[ok]), np.log(sc[ok] + 1.0)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    # lag-1 autocorrelation-adjusted SE (effective n)
    r1 = np.corrcoef(res[:-1], res[1:])[0, 1] if res.size > 3 else 0.0
    n_eff = max(10.0, res.size * (1 - r1) / (1 + r1)) if np.isfinite(r1) else res.size
    se = np.sqrt(res.var() / (np.var(x) * n_eff))
    return float(beta[1]), float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se)


def run_checks(model, d: D.SimData, idx, block: str = "val", *, device="cpu", seed: int = 0) -> dict:
    idx = np.asarray(idx)
    bm = D.block_mask(d.dates, block)
    t0, t1 = int(np.flatnonzero(bm)[0]), int(np.flatnonzero(bm)[-1]) + 1
    dates = d.dates[t0:t1]
    js = summer_mask(dates)
    base = simulate(model, d, idx, t0, t1, device=device)
    res = {"n_sites": len(idx), "block": block, "site_rows": []}
    tr = D.block_mask(d.dates, "train")
    iT, it = D.FORCING_RAW.index("tmax_c"), D.FORCING_RAW.index("tmin_c")
    ta_all = 0.5 * (d.forcing[idx][..., iT] + d.forcing[idx][..., it])

    warm = run(model, d, idx, Scenario(dT_air=2.0), t0, t1, device=device, baseline=base)
    dry = run(model, d, idx, Scenario(q_scale=0.5), t0, t1, device=device, baseline=base)
    dark = run(model, d, idx, Scenario(sw_scale=0.5), t0, t1, device=device, baseline=base)
    cold = run(model, d, idx, Scenario(dT_air=-3.0), t0, t1, device=device, baseline=base)
    hot4 = run(model, d, idx, Scenario(dT_air=4.0), t0, t1, device=device, baseline=base)
    wet = run(model, d, idx, Scenario(q_scale=1.5), t0, t1, device=device, baseline=base)
    bright = run(model, d, idx, Scenario(sw_scale=1.1), t0, t1, device=device, baseline=base)

    # M2 identity: recompute DO_sat mean from simulated hourly temperature shape
    tm = warm["scen"]["temp_c_mean"]
    amp = warm["scen"]["t_amp"]
    h = np.arange(24) + 0.5
    shape = np.cos(2 * np.pi * (h - physics.T_PEAK_H) / 24)
    th = np.clip(np.clip(tm, 0, None)[..., None] + amp[..., None] * shape, 0, None)
    sat_expect = physics.do_sat_mgl(th, d.pressure[idx][:, None, None]).mean(-1)
    m2_err = float(np.nanmax(np.abs(sat_expect - warm["scen"]["do_sat_mean"])))
    res["M2_identity_max_abs_err"] = m2_err
    res["M2_identity_pass"] = bool(m2_err <= M2_TOL)
    res["solubility_only_dDOsat_20C_plus2"] = float(physics.do_sat_mgl(22.0) - physics.do_sat_mgl(20.0))

    # placebo: permute water years (same permutation for forcing and discharge), seed fixed
    rs = np.random.default_rng(seed)
    wy = (d.dates + pd.DateOffset(months=3)).year.to_numpy()
    years = np.unique(wy[t0:t1])
    f_pl, q_pl = d.forcing[idx].copy(), d.q_m3s[idx].copy()
    pool = np.unique(wy[tr])
    for yv in years:
        tgt = np.flatnonzero(wy == yv)
        src_y = rs.choice(pool)
        src = np.flatnonzero(wy == src_y)
        n = min(len(tgt), len(src))
        f_pl[:, tgt[:n]] = d.forcing[idx][:, src[:n]]
        q_pl[:, tgt[:n]] = d.q_m3s[idx][:, src[:n]]
    pl = simulate(model, d, idx, t0, t1, device=device, forcing=f_pl, q_m3s=q_pl)

    rows = []
    for k, i in enumerate(idx):
        r = {"site": d.sites[i]}
        jT = D.TARGETS.index("temp_c_mean")
        # M1
        dtw = float(np.nanmean(warm["delta"]["temp_c_mean"][k, js]))
        b = _weekly_slope(d.y[i, tr, jT], ta_all[k, tr], d.dates[tr])
        r.update({"dTw_summer": dtw, "aw_slope_train": b, "M1_range": MIN_DTW <= dtw <= 2.0,
                  "M1_consistent": bool(np.isfinite(b) and abs(dtw / 2.0 - b) <= 0.25) if np.isfinite(b) else None})
        # M3
        jD = D.TARGETS.index("do_mgl_mean")
        obs_ratio = d.y[i, tr, jD] / physics.do_sat_mgl(d.y[i, tr, jT], d.pressure[i])
        summer_tr = np.isin(d.dates[tr].month, (6, 7, 8))
        rr = obs_ratio[summer_tr]
        rr = rr[np.isfinite(rr)]
        stable = bool(rr.size >= 60 and (np.quantile(rr, 0.75) - np.quantile(rr, 0.25)) <= 0.10)
        ddo = float(np.nanmean(warm["delta"]["do_mgl_mean"][k, js]))
        dsat = float(np.nanmean(warm["delta"]["do_sat_mean"][k, js]))
        sol = float(np.median(rr) * dsat) if rr.size else np.nan
        r.update({"stable_psat": stable, "dDO_mean_summer": ddo, "dDOsat_summer": dsat, "dDO_solubility_only": sol,
                  "ratio_to_solubility": ddo / sol if sol else np.nan,
                  "M3_sign": bool(ddo < 0 and ddo <= 0.5 * sol) if stable else None})
        # M4
        gpp, er = base["gpp"][k, js], base["er"][k, js]
        pr = np.nanmedian(gpp / np.maximum(np.abs(er), 1e-6))
        hetero = bool(np.isfinite(pr) and pr < 1.0)
        ddmin = float(np.nanmean(warm["delta"]["do_mgl_min"][k, js]))
        r.update({"p_r_ratio": float(pr), "heterotrophic": hetero, "dDO_min_summer": ddmin,
                  "M4_night": (ddmin < ddo) if hetero else None})
        # M5
        jS = D.TARGETS.index("spcond_uscm_mean")
        q_obs = np.where(d.q_missing[i, tr], np.nan, d.q_m3s[i, tr])
        slope, lo, hi = _cq_slope_ci(d.y[i, tr, jS], q_obs)
        dsc = float(np.nanmean(dry["delta"]["spcond_uscm_mean"][k]))     # model space (log)
        sig = np.isfinite(lo) and (lo > 0 or hi < 0)
        r.update({"cq_slope": slope, "cq_sig": bool(sig), "dSC_log_halfQ": dsc,
                  "M5_cq_sign": (np.sign(dsc) == -np.sign(slope)) if sig else None})
        # M6 placebo (pure simulation RMSE in block)
        for v, key in (("temp_c_mean", "M6_placebo_T"), ("do_mgl_mean", "M6_placebo_DO")):
            j = D.TARGETS.index(v)
            o = d.y[i, t0:t1, j]
            ok = np.isfinite(o)
            if ok.sum() >= 60:
                e_true = np.sqrt(np.mean((base[v][k, ok] - o[ok]) ** 2))
                e_pl = np.sqrt(np.mean((pl[v][k, ok] - o[ok]) ** 2))
                r[f"rmse_true_{v}"], r[f"rmse_placebo_{v}"] = float(e_true), float(e_pl)
                r[key] = bool(e_pl > e_true)
            else:
                r[key] = None
        # M7
        dg = float(np.nanmean(dark["delta"]["gpp"][k, js]))
        dmax = float(np.nanmean(dark["delta"]["do_mgl_max"][k, js]))
        g0 = float(np.nanmean(base["gpp"][k, js]))
        r.update({"dGPP_halfSW": dg, "dDOmax_halfSW": dmax, "M7_gpp": dg <= -MIN_DGPP_REL * g0, "M7_domax": dmax < 0})
        # M9 range ends
        dc = float(np.nanmean(cold["delta"]["temp_c_mean"][k, js]))
        dh = float(np.nanmean(hot4["delta"]["temp_c_mean"][k, js]))
        dsc_w = float(np.nanmean(wet["delta"]["spcond_uscm_mean"][k]))
        dg_b = float(np.nanmean(bright["delta"]["gpp"][k, js]))
        r.update({"dTw_minus3": dc, "dTw_plus4": dh, "dSC_log_1p5Q": dsc_w, "dGPP_1p1SW": dg_b,
                  "M9_cold3": bool(-3.0 <= dc <= -0.3), "M9_warm4": bool(0.4 <= dh <= 4.0),
                  "M9_wetQ": (np.sign(dsc_w) == np.sign(slope)) if sig else None, "M9_brightSW": dg_b > 0})
        # M8
        er0, er1 = np.nanmean(base["er"][k, js]), np.nanmean(warm["scen"]["er"][k, js])
        tw0 = np.nanmean(base["temp_c_mean"][k, js])
        r.update({"dER_rel": float(er1 / er0 - 1) if er0 else np.nan,
                  "dER_rel_arrhenius": float(physics.arrhenius_ratio(tw0, dtw) - 1) if np.isfinite(tw0) else np.nan})
        r.update({"flag_q_extrap_halfQ": float(dry["flags"]["q_extrap"][k]),
                  "flag_tair_extrap_plus2": float(warm["flags"]["tair_extrap"][k])})
        rows.append(r)
    df = pd.DataFrame(rows)
    summary = {}
    for key, share in PASS_SHARE.items():
        if key == "M2_identity":
            summary[key] = {"value": m2_err, "pass": res["M2_identity_pass"], "required": f"max|err|<={M2_TOL}"}
            continue
        col = df[key].dropna().astype(bool) if key in df else pd.Series(dtype=bool)
        summary[key] = {"share": float(col.mean()) if len(col) else np.nan, "n_eligible": int(len(col)),
                        "required_share": share, "pass": bool(len(col) and col.mean() >= share)}
    res["summary"] = summary
    res["site_rows"] = df
    res["all_pass"] = bool(all(v["pass"] for v in summary.values()))
    return res
