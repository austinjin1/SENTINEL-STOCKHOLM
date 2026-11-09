# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the shared event layer v1 (sentinel/eval/events.py; docs/plans/event_layer_plan.md).

Synthetic arrays only, except ``test_e1a_reproduces_registered_ew_onset_counts``, which re-counts the
registered EW-v2 H1 onsets on TRAIN/VAL blocks × train/val-role sites through the guarded sonde loader
(skipped when the processed data are not on this machine)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss

from sentinel.eval import events as E
from sentinel.eval import hazards as HZ

REPO = Path(__file__).resolve().parents[1]
HAZ_CODES = {"E1a": "H1_hypoxia", "E1a_t4": "H1_hypoxia_t4", "E1a_t2": "H1_hypoxia_t2", "E2a": "H2_chloride",
             "E2b": "H2_chloride_acute", "E4a": "H5b_chl_bloom", "E5a": "H4_nitrate"}


def _daily(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D")


# ------------------------------------------------------------------------------------------ taxonomy

def test_taxonomy_codes_unique_and_hazard_backed_codes_mirror_hazards():
    assert len(E.EVENT_CODES) == len(set(E.EVENT_CODES)) == len(E.EVENTS)
    for code, key in HAZ_CODES.items():
        ev, h = E.EVENTS[code], HZ.HAZARDS[key]
        assert ev.hazard_key == key
        assert (ev.variables, ev.direction, ev.threshold) == ((h.variable,), h.direction, h.threshold)
        r = ev.onset
        assert (r.method, r.pre_days, r.pre_margin, r.pre_gap, r.refractory_days) == \
               ("hazard", h.pre_days, h.pre_margin, h.pre_gap, h.refractory_days)
        assert E._hazard_for(ev) is h          # the hazard object itself, not a copy
    for ev in E.EVENTS.values():
        for t in ev.tiers:
            assert E.EVENTS[t].parent == ev.code
        if ev.parent:
            assert ev.parent in E.EVENTS
        assert ev.anchor, ev.code
    assert E.__version__ in ("0.dev", "1.0")


def test_not_exported_from_eval_package_and_ew_freeze_code_unchanged():
    assert "events" not in (REPO / "sentinel/eval/__init__.py").read_text()
    from sentinel.models.sonde import guard

    fz = json.loads((REPO / "docs/preregistration/FREEZE_ew.json").read_text())
    assert guard.scoring_code_sha256(files=fz["scoring_code_files"]) == fz["scoring_code_sha256"]


def test_report_events_refuse_measurement_labelling():
    for code in ("E4c", "E8", "E9", "E11", "E5b_nrsa"):
        with pytest.raises(ValueError):
            E.event_state(np.array([1.0]), code)
        with pytest.raises(ValueError):
            E.exceedance_probability(E.Gaussian(np.array([1.0]), np.array([1.0])), code)
    with pytest.raises(KeyError):
        E.get_event("E99")


# ------------------------------------------------------------------------------------------ labelling

@pytest.mark.parametrize("code", sorted(HAZ_CODES))
def test_hazard_backed_onsets_and_issue_labels_bit_identical(code):
    h = HZ.HAZARDS[HAZ_CODES[code]]
    rng = np.random.default_rng(sum(map(ord, code)))
    sign = 1.0 if h.direction == "above" else -1.0
    parts = []
    while sum(map(len, parts)) < 1500:       # safe spells (some marginal) alternating with short excursions
        safe_off = h.pre_margin * rng.choice([0.5, 1.5, 2.0])
        parts.append(h.threshold - sign * (safe_off + rng.random(rng.integers(3, 40))))
        parts.append(h.threshold + sign * (0.1 + rng.random(rng.integers(1, 6))))
    x = np.concatenate(parts)
    x = np.where(rng.random(x.size) < 0.03, np.nan, x)
    ev_ref, on_ref = HZ.daily_onsets(x, h)
    ev, on = E.event_onsets(x, code)
    assert np.array_equal(ev, ev_ref) and np.array_equal(on, on_ref)
    assert on.sum() > 0
    y_ref, v_ref = HZ.issue_labels(x, ev_ref, on_ref, 3, refractory_days=h.refractory_days)
    y, v = E.issue_labels(x, code, 3)
    assert np.array_equal(y, y_ref) and np.array_equal(v, v_ref)
    st = E.event_state(x, code)
    assert np.array_equal(st[np.isfinite(x)] == 1, ev_ref[np.isfinite(x)])
    assert np.isnan(st[~np.isfinite(x)]).all()


def test_event_state_all_nan_and_outside_and_at_or_above():
    assert np.isnan(E.event_state(np.full(5, np.nan), "E1a")).all()
    ev, on = E.event_onsets(np.full(5, np.nan), "E1a")
    assert not ev.any() and not on.any()
    assert E.event_state(np.array([6.0, 6.5, 9.0, 9.1, np.nan]), "E6_snap")[:4].tolist() == [1, 0, 0, 1]
    assert E.event_state(np.array([49.9, 50.0, 50.1]), "E4a_t50").tolist() == [0, 1, 1]
    assert E.event_state(np.array([19.9, 20.0, 20.1]), "E4a").tolist() == [0, 0, 1]


def test_snapshot_events_have_no_onset():
    with pytest.raises(ValueError):
        E.event_onsets(np.array([1.0, 2.0]), "E3a_tss")


def test_derive_series_aggregations():
    n = 60
    d = _daily(n)
    tmax = np.arange(n, dtype=float)
    f = pd.DataFrame({"temp_c_max": tmax, "do_mgl_mean": np.r_[np.full(30, 8.0), np.full(30, 4.0)],
                      "turb_fnu_mean": np.r_[np.full(40, 10.0), [80.0], np.full(19, 10.0)],
                      "ph_min": np.full(n, 7.0), "ph_max": np.full(n, 8.0)}, index=d)
    s7 = E.derive_series(f, "E7")
    assert np.isnan(s7.iloc[:6]).all() and s7.iloc[6] == pytest.approx(3.0)
    s30 = E.derive_series(f, "E1b")
    assert np.isnan(s30.iloc[25]) and s30.iloc[29] == pytest.approx(8.0) and s30.iloc[59] == pytest.approx(4.0)
    ex = E.derive_series(f, "E3a")
    assert np.isnan(ex.iloc[19]) and ex.iloc[40] == pytest.approx(70.0) and ex.iloc[41] == pytest.approx(0.0)
    f.loc[d[10], "ph_min"] = 6.0
    f.loc[d[11], "ph_max"] = 9.5
    f.loc[d[12], "ph_max"] = np.nan
    ph = E.derive_series(f, "E6")
    assert ph.iloc[10] == pytest.approx(0.5) and ph.iloc[11] == pytest.approx(0.5) and np.isnan(ph.iloc[12])
    assert ph.iloc[0] == pytest.approx(-0.5)
    with pytest.raises(ValueError):          # gap in the daily index
        E.derive_series(f.drop(d[5]), "E7")
    with pytest.raises(KeyError):
        E.derive_series(f[["temp_c_max"]], "E6")


def test_e1b_and_e7_onsets_use_clean_window_before_the_averaging_window():
    # E1b: 30-d mean below 5.5; clean pre-window (>= 6.0) must END before the averaging window
    x = np.r_[np.full(60, 8.0), np.full(40, 3.0)]
    s = pd.Series(x).rolling(30, min_periods=27).mean().to_numpy()
    ev, on = E.event_onsets(s, "E1b")
    first = int(np.flatnonzero(ev)[0])
    assert on.sum() == 1 and np.flatnonzero(on)[0] == first
    # E7 (refractory 30 d): two crossings 20 days apart -> second is not an onset
    y = np.r_[np.full(20, 15.0), np.full(5, 25.0), np.full(15, 15.0), np.full(5, 25.0)]
    ev, on = E.event_onsets(y, "E7")
    assert np.flatnonzero(on).tolist() == [20]


def test_e3a_clean_window_and_e3b_runs():
    x = np.r_[np.zeros(5), [60.0], np.zeros(3), [60.0], np.zeros(10), [60.0], [np.nan], np.zeros(3), [60.0]]
    ev, on = E.event_onsets(x, "E3a")
    assert np.flatnonzero(ev).tolist() == [5, 9, 20, 25]
    assert np.flatnonzero(on).tolist() == [5, 20]       # 9: refractory; 25: pre-window has a NaN
    z = np.r_[np.zeros(3), np.full(12, 30.0), np.zeros(2), np.full(9, 30.0), np.zeros(2), np.full(10, 30.0)]
    st = E.event_state(z, "E3b")
    ev, on = E.event_onsets(z, "E3b")
    assert np.flatnonzero(on).tolist() == [3, 28]       # the 9-day run does not qualify
    assert st[3:15].all() and not st[17:26].any() and st[28:38].all()
    assert np.array_equal(ev, st == 1)
    y, v = E.issue_labels(z, "E3b", 2)
    assert y[1] == 1 and y[2] == 1 and not v[5]


def test_coldwater_eligible():
    d = pd.date_range("2010-01-01", "2012-12-31", freq="D")
    cold = 15 + 3 * np.sin(2 * np.pi * (d.dayofyear - 110) / 365)
    warm = cold + 10
    assert E.coldwater_eligible(d, cold) and not E.coldwater_eligible(d, warm)
    assert not E.coldwater_eligible(d[:30], cold[:30])           # too few Jul-Aug days


# ------------------------------------------------------------------------------------------ derived route

def test_gaussian_exceedance_matches_scipy_sigma0_nan_and_spaces():
    mu = np.array([10.0, 20.0, 30.0, 20.0, np.nan, 25.0])
    sd = np.array([5.0, 0.0, 5.0, 2.0, 1.0, np.nan])
    p = E.exceedance_probability(E.Gaussian(mu, sd), "E4a")
    assert p[0] == pytest.approx(norm.sf(20, 10, 5)) and p[2] == pytest.approx(norm.sf(20, 30, 5))
    assert p[1] == 0.0                           # point mass exactly at the threshold, strict '>'
    assert p[3] == pytest.approx(0.5) and np.isnan(p[4]) and np.isnan(p[5])
    assert E.exceedance_probability(E.Gaussian(np.array([50.0]), np.array([0.0])), "E4a_t50")[0] == 1.0
    assert E.exceedance_probability(E.Gaussian(np.array([3.0]), np.array([0.0])), "E1a")[0] == 0.0
    assert E.exceedance_probability(E.Gaussian(np.array([2.0]), np.array([1.0])), "E1a")[0] == \
        pytest.approx(norm.cdf(3, 2, 1))
    lg = E.exceedance_probability(E.Gaussian(np.array([np.log10(20.0)]), np.array([0.3]), space="log10"), "E4a")
    assert lg[0] == pytest.approx(0.5)
    ph = E.exceedance_probability(E.Gaussian(np.array([7.75]), np.array([1.0])), "E6_snap")[0]
    assert ph == pytest.approx(norm.cdf(6.5, 7.75, 1) + norm.sf(9.0, 7.75, 1))
    with pytest.raises(ValueError):
        E.exceedance_probability(E.Gaussian(np.array([1.0]), np.array([-1.0])), "E4a")
    with pytest.raises(ValueError):
        E.exceedance_probability(E.Gaussian(np.array([1.0]), np.array([1.0]), space="sqrt"), "E4a")


def test_quantile_exceedance_close_to_gaussian_interior_and_tails():
    lv = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    mu = np.array([12.0, 20.0, 26.0, 40.0, 5.0])
    q = norm.ppf(lv)[None, :] * 4.0 + mu[:, None]
    p = E.exceedance_probability(E.Quantiles(lv, q), "E4a")
    exact = norm.sf(20, mu, 4.0)
    assert np.allclose(p, exact, atol=0.02)
    assert p[1] == pytest.approx(0.5)
    # tails beyond the outermost quantiles are Gaussian (exact for Gaussian quantiles)
    assert p[3] == pytest.approx(exact[3], rel=1e-6) and p[4] == pytest.approx(exact[4], rel=1e-6)
    # 1-D values, crossing quantiles, NaN row, degenerate (all equal) rows
    assert E.exceedance_probability(E.Quantiles(lv, q[1]), "E4a").shape == (1,)
    cross = np.array([[10, 30, 25, 40, 45, 50, 55.0]])
    pc = E.exceedance_probability(E.Quantiles(lv, cross), "E4a")
    assert 0 < pc[0] < 1
    assert np.isnan(E.exceedance_probability(E.Quantiles(lv, np.r_[np.nan, q[1, 1:]]), "E4a")[0])
    deg = np.full((1, lv.size), 25.0)
    assert E.exceedance_probability(E.Quantiles(lv, deg), "E4a")[0] == 1.0
    assert E.exceedance_probability(E.Quantiles(lv, deg - 10), "E4a")[0] == 0.0
    for bad in (np.array([0.5]), np.array([0.5, 0.1]), np.array([0.0, 0.5]), np.array([0.5, 1.0])):
        with pytest.raises(ValueError):
            E.exceedance_probability(E.Quantiles(bad, np.zeros((1, bad.size))), "E4a")


def test_quantile_exceedance_monotone_in_threshold():
    lv = np.array([0.1, 0.5, 0.9])
    q = E.Quantiles(lv, np.array([[12.0, 18.0, 24.0]]))
    p16, p18, p20 = (E.exceedance_probability(q, c)[0] for c in ("E7_t16", "E7_t18", "E7"))
    assert p16 > p18 > p20 > 0


def test_mixture_and_samples():
    g = E.exceedance_probability(E.Gaussian(np.array([18.0]), np.array([3.0])), "E4a")
    m1 = E.exceedance_probability(E.GaussianMixture(np.ones((1, 1)), np.array([[18.0]]), np.array([[3.0]])), "E4a")
    assert m1[0] == pytest.approx(g[0])
    w = np.array([[2.0, 6.0], [0.0, 0.0], [1.0, 1.0]])
    mu = np.array([[10.0, 30.0], [10.0, 30.0], [20.0, 25.0]])
    sd = np.array([[2.0, 5.0], [1.0, 1.0], [0.0, 0.0]])
    pm = E.exceedance_probability(E.GaussianMixture(w, mu, sd), "E4a")
    assert pm[0] == pytest.approx(0.25 * norm.sf(20, 10, 2) + 0.75 * norm.sf(20, 30, 5))
    assert np.isnan(pm[1])                       # zero total weight
    assert pm[2] == pytest.approx(0.5)           # two point masses: 20 (not > 20) and 25
    assert E.exceedance_probability(E.GaussianMixture(w[2:], mu[2:], sd[2:]), "E4a_t50")[0] == 0.0
    with pytest.raises(ValueError):
        E.exceedance_probability(E.GaussianMixture(-w[:1], mu[:1], sd[:1]), "E4a")
    with pytest.raises(ValueError):
        E.exceedance_probability(E.GaussianMixture(w[:, :1], mu, sd), "E4a")
    s = E.exceedance_probability(E.Samples(np.array([[10, 20, 30, 40.0], [np.nan] * 4])), "E4a")
    assert s[0] == 0.5 and np.isnan(s[1])
    with pytest.raises(TypeError):
        E.exceedance_probability(np.array([1.0]), "E4a")


def test_recalibrate_platt_isotonic_and_guards():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, 20000)
    p_raw = 1 / (1 + np.exp(-z))
    y = (rng.random(z.size) < 1 / (1 + np.exp(-(0.5 * z - 1.0)))).astype(float)
    c = E.recalibrate(p_raw, y)
    assert c.params["a"] == pytest.approx(0.5, abs=0.05) and c.params["b"] == pytest.approx(-1.0, abs=0.06)
    assert c.n_fit == z.size and len(c.fit_sha256) == 64
    out = c(np.array([0.2, np.nan, 0.9]))
    assert np.isnan(out[1]) and 0 < out[0] < out[2] < 1
    iso = E.recalibrate(p_raw, y, method="isotonic")
    grid = np.linspace(0, 1, 101)
    assert np.all(np.diff(iso.transform(grid)) >= -1e-12)
    sep = E.recalibrate(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert np.isfinite([sep.params["a"], sep.params["b"]]).all()
    assert E.recalibrate(np.array([0.1, np.nan, 0.9]), np.array([0, 1, 1])).n_fit == 2
    with pytest.raises(ValueError):
        E.recalibrate(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(ValueError):
        E.recalibrate(np.array([0.1, 1.2]), np.array([0, 1]))
    with pytest.raises(ValueError):
        E.recalibrate(np.array([0.1, 0.2]), np.array([0, 2]))
    with pytest.raises(ValueError):
        E.recalibrate(p_raw, y, method="beta")


# ------------------------------------------------------------------------------------------ metrics

def _toy(n=600, n_groups=30, seed=2):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    y = (rng.random(n) < 0.15 + 0.2 * (g % 3 == 0)).astype(float)
    p = np.clip(0.1 + 0.5 * y + rng.normal(0, 0.15, n), 0, 1)
    return y, p, g


def test_state_metrics_cross_check_sklearn_and_cis():
    y, p, g = _toy()
    r = E.state_metrics(y, p, groups=g, n_boot=300)
    assert r["auprc"]["point"] == pytest.approx(average_precision_score(y, p))
    assert r["brier"]["point"] == pytest.approx(brier_score_loss(y, p))
    for k in ("auprc", "brier"):
        assert r[k]["ci_low"] <= r[k]["point"] <= r[k]["ci_high"]
    assert r["n"] == y.size and r["n_pos"] == int(y.sum()) and r["prevalence"] == pytest.approx(y.mean())
    assert r["n_groups"] == len(np.unique(g)) and r["n_groups_with_pos"] == len(np.unique(g[y == 1]))
    assert r["power_flag"] == "feasible"
    rr = E.state_metrics(y, p, groups=g, p_ref=y.mean(), n_boot=200)
    bs_ref = np.mean((y.mean() - y) ** 2)
    assert rr["bss"]["point"] == pytest.approx(1 - brier_score_loss(y, p) / bs_ref)
    assert np.isfinite(rr["bss"]["ci_low"])
    # NaN rows are dropped (also NaN reference rows)
    y2, p2 = y.copy(), p.copy()
    y2[:10] = np.nan
    p2[10:20] = np.nan
    r2 = E.state_metrics(y2, p2, groups=g, n_boot=50)
    assert r2["n"] == y.size - 20
    assert E.state_metrics(y, p, groups=None, n_boot=50)["unit"] == "row"


def test_state_metrics_edge_cases():
    r = E.state_metrics(np.full(4, np.nan), np.full(4, 0.5), groups=np.arange(4), n_boot=10)
    assert r["n"] == 0 and np.isnan(r["auprc"]["point"]) and np.isnan(r["brier"]["point"])
    r = E.state_metrics(np.zeros(20), np.linspace(0, 1, 20), groups=np.arange(20) % 4, n_boot=50)
    assert np.isnan(r["auprc"]["point"]) and np.isnan(r["auprc"]["ci_low"])
    assert r["brier"]["point"] == pytest.approx(brier_score_loss(np.zeros(20), np.linspace(0, 1, 20)))
    assert r["power_flag"] == "underpowered" and r["n_groups_with_pos"] == 0
    y, p, _ = _toy(200)
    one = E.state_metrics(y, p, groups=np.zeros(200), n_boot=50)
    assert np.isfinite(one["auprc"]["point"]) and np.isnan(one["auprc"]["ci_low"]) and "note" in one["auprc"]
    with pytest.raises(ValueError):
        E.state_metrics(np.array([0, 2.0]), np.array([0.1, 0.2]), groups=None)
    with pytest.raises(ValueError):
        E.state_metrics(np.array([0, 1.0]), np.array([0.1, 1.2]), groups=None)
    with pytest.raises(ValueError):
        E.state_metrics(np.array([0, 1.0]), np.array([0.1, 0.2]), groups=np.array([1, 2, 3]))


def test_compare_routes():
    y, p, g = _toy()
    same = E.compare_routes(y, p, p, groups=g, n_boot=100)
    assert same["delta_brier"]["diff"] == 0 and same["delta_auprc"]["diff"] == 0
    worse = np.clip(p + 0.3, 0, 1)
    r = E.compare_routes(y, p, worse, groups=g, n_boot=300)
    assert r["delta_brier"]["diff"] < 0 and r["delta_brier"]["ci_high"] < 0
    assert r["brier_derived"] == pytest.approx(brier_score_loss(y, p))
    assert r["auprc_direct"] == pytest.approx(average_precision_score(y, worse))
    one = E.compare_routes(y, p, worse, groups=np.zeros(y.size), n_boot=50)
    assert np.isnan(one["delta_brier"]["ci_low"])
    with pytest.raises(ValueError):
        E.compare_routes(y, p, worse + 1.0, groups=g)


def test_power_flag_boundaries():
    assert E.power_flag(30, 10) == "feasible"
    assert E.power_flag(29, 50) == "underpowered"
    assert E.power_flag(500, 9) == "underpowered"
    assert E.power_flag(5, 2, min_pos=5, min_clusters=2) == "feasible"


def _onset_frame():
    rows = []
    # site A: onset day 20; alarm days 16-17 (lead 4) ; false episode at days 50-51; 100 usable days
    for i in range(100):
        rows.append(("A", i, 0.9 if i in (16, 17, 50, 51) else 0.1, True, i == 20))
    # site B: onset day 30 with no alarm; onset day 70 with no usable row in the window (not evaluable)
    for i in range(100):
        usable = not (65 <= i < 70)
        rows.append(("B", i, 0.1, usable, i in (30, 70)))
    f = pd.DataFrame(rows, columns=["site", "day", "score", "usable", "onset"])
    f["date"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(f["day"], "D")
    return f


def test_onset_metrics_hits_lead_false_alarms():
    f = _onset_frame()
    r = E.onset_metrics(f.date, f.score, f.usable, f.onset, threshold=0.5, h_days=5, site=f.site, n_boot=100)
    assert r["n_onsets_evaluable"] == 2 and r["n_onsets_warned"] == 1
    assert r["hit_rate"]["point"] == pytest.approx(0.5)
    assert r["lead_times_days"] == [4.0] and r["lead_time_days_median"] == 4.0
    assert r["n_alarm_episodes"] == 2 and r["n_false_episodes"] == 1
    assert r["site_years"] == pytest.approx(195 / 365.25)
    assert r["false_alarms_per_site_year"]["point"] == pytest.approx(1 / (195 / 365.25))
    # a shorter horizon misses the 4-day lead; the alarm episode then counts as false too
    r3 = E.onset_metrics(f.date, f.score, f.usable, f.onset, threshold=0.5, h_days=3, site=f.site, n_boot=50)
    assert r3["n_onsets_warned"] == 0 and r3["n_false_episodes"] == 2
    # clusters: both sites in one cluster -> CI undefined
    rc = E.onset_metrics(f.date, f.score, f.usable, f.onset, threshold=0.5, h_days=5, site=f.site,
                         clusters={"A": "c", "B": "c"}, n_boot=50)
    assert rc["n_clusters"] == 1 and np.isnan(rc["hit_rate"]["ci_low"])
    # rows with gaps are reindexed daily (episodes do not merge across missing days)
    g = f[~((f.site == "A") & f.day.isin([30, 31]))]
    assert E.onset_metrics(g.date, g.score, g.usable, g.onset, threshold=0.5, h_days=5,
                           site=g.site, n_boot=20)["n_alarm_episodes"] == 2
    with pytest.raises(ValueError):
        E.onset_metrics(pd.concat([f, f.iloc[:1]]).date, pd.concat([f, f.iloc[:1]]).score,
                        pd.concat([f, f.iloc[:1]]).usable, pd.concat([f, f.iloc[:1]]).onset,
                        threshold=0.5, h_days=5, site=pd.concat([f, f.iloc[:1]]).site)


def test_onset_metrics_no_onsets_and_empty():
    f = _onset_frame()
    f["onset"] = False
    r = E.onset_metrics(f.date, f.score, f.usable, f.onset, threshold=0.5, h_days=5, site=f.site, n_boot=20)
    assert r["n_onsets_evaluable"] == 0 and np.isnan(r["hit_rate"]["point"]) and np.isnan(r["lead_time_days_median"])
    assert r["n_false_episodes"] == 2
    e = E.onset_metrics([], [], [], [], threshold=0.5, h_days=5, site=[])
    assert e["n_sites"] == 0 and np.isnan(e["hit_rate"]["point"])


# ------------------------------------------------------------------------------------------ matching

def test_match_external_radius_span_window_and_lake():
    sites = pd.DataFrame({"site_id": ["s1", "s2", "s3"], "lat": [41.0, 41.0, 42.0], "lon": [-93.0, -93.1, -93.0],
                          "first": pd.to_datetime(["2010-01-01", "2015-01-01", "2010-01-01"]),
                          "last": pd.to_datetime(["2020-12-31", "2020-12-31", "2020-12-31"]),
                          "cluster": ["b1", "b1", "b2"]})
    events = pd.DataFrame({"event_id": [1, 2, 3], "lat": [41.0, 41.0, np.nan], "lon": [-93.05, -93.0, -93.0],
                           "date": pd.to_datetime(["2012-06-01", "2016-06-01", "2016-06-01"])})
    m = E.match_external(events, sites, E.MatchRule(radius_km=10.0))
    assert set(map(tuple, m[["event_id", "site_id"]].to_numpy())) == {(1, "s1"), (1, "s2"), (2, "s1"), (2, "s2")}
    d = m[(m.event_id == 2) & (m.site_id == "s2")]["distance_km"].iloc[0]
    assert d == pytest.approx(8.39, abs=0.05)
    m2 = E.match_external(events, sites, E.MatchRule(radius_km=10.0, require_record_span=True))
    assert (1, "s2") not in set(map(tuple, m2[["event_id", "site_id"]].to_numpy()))
    obs = sites.assign(date=pd.to_datetime(["2012-05-30", "2012-06-05", "2012-06-01"]))
    m3 = E.match_external(events, obs, E.MatchRule(radius_km=10.0, window_days=(0, 3)))
    assert m3[["event_id", "site_id"]].values.tolist() == [[1, "s1"]] and m3.lag_days.iloc[0] == 2.0
    lk = E.match_external(events.assign(lake_id=["L1", "L2", "L1"]), sites.assign(lake_id=["L1", "L1", "L1"]),
                          E.MatchRule(radius_km=10.0, same_lake=True))
    assert set(lk.event_id) == {1}
    assert E.match_external(events.iloc[2:], sites, E.MatchRule(radius_km=10.0)).empty


# ------------------------------------------------------------------------------------------ real-data check

def _have_sonde():
    from sentinel.core import paths

    return (paths.PROCESSED / "sonde_daily" / "index.parquet").exists() and \
        (REPO / "docs/preregistration/sonde_split.json").exists()


@pytest.mark.skipif(not _have_sonde(), reason="processed sonde_daily not on this machine")
def test_e1a_reproduces_registered_ew_onset_counts():
    """EW-v2 protocol §9 (registry experiment ew_v2_onset_counts): H1 < 3 mg/L onsets, Approved values,
    train block 566 / 101 and val block 84 / 6 at train- / val-role stream sites. TRAIN/VAL blocks and
    train/val-role sites only, through the guarded loader (test data are never requested)."""
    from sentinel.data.sonde_splits import TIME_BLOCKS
    from sentinel.models.sonde import common

    split, info = common.load_split(common.FINAL_SPLIT)
    reg = common.site_registry()
    have = {p.stem for p in common.SITE_DIR.glob("*.parquet")}
    sites = sorted(s for r in ("train", "val") for s in split.sites(r)
                   if s in have and s in reg.index and reg.loc[s, "site_type"] in {"ST", "ST-TS", "ST-CA", "ST-DCH"})
    data = common.prepare_data(split, info, sites, blocks=("train", "val"))
    assert set(data.roles) <= {"train", "val"}
    j = data.cube.var_index("do_mgl_min")
    x = np.where(data.cube.prov[:, :, j], np.nan, data.cube.y[:, :, j].astype(float))
    on = np.zeros(x.shape, bool)
    for s in range(x.shape[0]):
        if np.isfinite(x[s]).any():
            on[s] = E.event_onsets(x[s], "E1a")[1]
    dates = pd.DatetimeIndex(data.cube.dates)
    roles = np.asarray(data.roles)
    got = {}
    for block in ("train", "val"):
        lo, hi = (pd.Timestamp(t) for t in TIME_BLOCKS[block])
        tm = np.asarray((dates >= lo) & (dates <= hi))
        for role in ("train", "val"):
            got[(block, role)] = int(on[np.ix_(roles == role, tm)].sum())
    assert got == {("train", "train"): 566, ("train", "val"): 101, ("val", "train"): 84, ("val", "val"): 6}


def test_below_direction_log_space_and_point_mass_ties():
    lv = np.array([0.1, 0.5, 0.9])
    z = norm.ppf(lv)
    q = E.Quantiles(lv, (2.5 + z * 1.0)[None, :])                       # median DO 2.5 mg/L
    # linear interpolation between the 0.5 and 0.9 quantiles (3 quantiles only)
    assert E.exceedance_probability(q, "E1a")[0] == pytest.approx(0.5 + 0.4 * 0.5 / z[2])
    ql = E.Quantiles(lv, (np.log10(20.0) + 0.2 * z)[None, :], space="log10")
    assert E.exceedance_probability(ql, "E4a")[0] == pytest.approx(0.5)
    assert E.exceedance_probability(E.Quantiles(lv, np.zeros((1, 3)) - 1.0, space="log10"), "E4a")[0] == 0.0
    s = E.Samples(np.array([[50.0, 50.0, 10.0, 60.0]]))
    assert E.exceedance_probability(s, "E4a_t50")[0] == 0.75               # ties count for '>='
    mix = E.GaussianMixture(np.array([[1.0, 1.0]]), np.array([[3.0, 2.0]]), np.array([[0.0, 0.0]]))
    assert E.exceedance_probability(mix, "E1a")[0] == pytest.approx(0.5)   # mass at 3.0 is not '< 3'
    assert E.exceedance_probability(E.Gaussian(np.array([3.0]), np.array([0.0])), "E1a")[0] == 0.0


def test_state_metrics_nan_reference_rows_dropped_and_cluster_without_positives():
    y, p, g = _toy(300, 10)
    ref = np.full(y.size, 0.2)
    ref[:5] = np.nan
    r = E.state_metrics(y, p, groups=g, p_ref=ref, n_boot=50)
    assert r["n"] == y.size - 5
    g2 = g.copy()
    g2[y == 0] = 99                                    # cluster 99 has negatives only
    r2 = E.state_metrics(y, p, groups=g2, n_boot=50)
    assert r2["n_groups"] == len(np.unique(g2)) and r2["n_groups_with_pos"] == len(np.unique(g2[y == 1]))
    assert 99 not in set(g2[y == 1]) and r2["n_groups"] > r2["n_groups_with_pos"]
