# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for D6 MusselGuard and the WBD HUC12 parser (synthetic arrays only; no network)."""

import numpy as np
import pandas as pd

from sentinel.data.sources import wbd_huc12 as WB
from sentinel.models.downstream import musselguard as MG


def _universe():
    # chain A1 -> A2 -> A3 -> A4 (tohuc), plus isolated B1 far away
    return pd.DataFrame({
        "huc12": ["010000000001", "010000000002", "010000000003", "010000000004", "020000000001"],
        "tohuc": ["010000000002", "010000000003", "010000000004", "OCEAN", "OCEAN"],
        "lat": [45.0, 45.1, 45.2, 45.3, 35.0], "lon": [-90.0, -90.0, -90.0, -90.0, -80.0],
        "areasqkm": [100.0] * 5, "states": ["MN", "MN", "WI", "WI", "NC"],
        "huc8": ["01000000"] * 4 + ["02000000"], "log_area": [2.0] * 5})


def _records(rows):
    r = pd.DataFrame(rows, columns=["latitude", "longitude", "year"])
    r["loc"] = r["latitude"].round(2).astype(str) + "_" + r["longitude"].round(2).astype(str)
    return r


def test_post_T_record_in_same_bin_does_not_move_features():
    """Review C1: a post-T record inside the same 0.01-degree bin must not shift the bin position."""
    u = _universe()
    first = pd.DataFrame({"huc12": ["020000000001"], "first_year_any": [2030]})
    pre = _records([(45.104, -90.004, 2005)])
    both = pd.concat([pre, _records([(45.096, -89.996, 2012)])], ignore_index=True)
    assert pre["loc"].iloc[0] == both["loc"].iloc[1]                    # same bin
    a = MG.build_origin(2006, u, first, pre, horizon=3)
    b = MG.build_origin(2006, u, first, both, horizon=3)
    cols = MG.PRESSURE_FEATURES + MG.RULE_FEATURES
    pd.testing.assert_frame_equal(a[cols], b[cols])
    s = MG.sources_at(both, 2006)
    assert len(s) == 1 and s["lat"].iloc[0] == 45.104


def test_has_us_state_drops_foreign_only_units():
    st = pd.Series(["MN", "CN", "MN,CN", "MX", None, "WI,MI"])
    assert MG.has_us_state(st).tolist() == [True, False, True, False, False, True]


def test_build_origin_uses_only_records_up_to_T():
    u = _universe()
    first = pd.DataFrame({"huc12": ["010000000002", "010000000004", "020000000001"],
                          "first_year_any": [2005, 2010, 2030]})
    src = _records([(45.1, -90.0, 2005), (45.3, -90.0, 2010), (35.0, -80.0, 2030)])
    f = MG.build_origin(2006, u, first, src, horizon=5)
    assert "010000000002" not in set(f["huc12"])                       # infested at T -> not a candidate
    y = dict(zip(f["huc12"], f["y"]))
    assert y["010000000004"] == 1 and y["020000000001"] == 0 and y["010000000001"] == 0
    # moving a post-T record must not change any feature at T
    src2 = src.copy()
    src2.loc[1, ["latitude", "longitude"]] = [45.0, -90.0]
    g = MG.build_origin(2006, u, first, src2, horizon=5)
    cols = [c for c in MG.PRESSURE_FEATURES + MG.LOCATION_FEATURES + MG.RULE_FEATURES]
    pd.testing.assert_frame_equal(f[cols], g[cols])
    hops = dict(zip(f["huc12"], f["hops_from_infested_upstream"]))
    assert hops["010000000003"] == 1 and hops["010000000004"] == 2 and np.isnan(hops["010000000001"])
    dn = dict(zip(f["huc12"], f["hops_to_infested_downstream"]))
    assert dn["010000000001"] == 1 and np.isnan(dn["010000000003"])


def test_label_window_is_exclusive_of_T():
    u = _universe()
    first = pd.DataFrame({"huc12": ["010000000001", "010000000003"], "first_year_any": [2009, 2013]})
    src = _records([(45.0, -90.0, 2009)])
    f = MG.build_origin(2009, u, first, src, horizon=3)
    y = dict(zip(f["huc12"], f["y"]))
    assert "010000000001" not in y and y["010000000003"] == 0          # 2013 > T+3


def test_sc_rule_classes():
    df = pd.DataFrame({"huc8_spcond_median": [50, 150, 250, 400, np.nan]})
    s = MG.sc_rule_score(df)
    assert s[:4].tolist() == [0, 1, 2, 3] and np.isnan(s[4])


def test_wbd_parse_and_tidy():
    payload = {"features": [
        {"attributes": {"huc12": "070900020504", "name": "x", "areasqkm": 10.0, "states": "WI", "tohuc": "070900020505"},
         "geometry": {"rings": [[[-89.0, 43.0], [-88.0, 43.0], [-88.0, 44.0], [-89.0, 44.0], [-89.0, 43.0]]]}},
        {"attributes": {"huc12": "bad", "name": "y", "areasqkm": 1.0, "states": "WI", "tohuc": "CLOSED BASIN"},
         "geometry": {"rings": []}}]}
    d = WB.tidy(WB.parse_page(payload))
    assert len(d) == 1 and d.loc[0, "huc8"] == "07090002"
    assert abs(d.loc[0, "lon"] + 88.5) < 1e-9 and abs(d.loc[0, "lat"] - 43.5) < 1e-9
    assert d.loc[0, "tohuc_kind"] == "huc12"


def test_design_fills_hops_for_linear_models():
    df = pd.DataFrame({"hops_from_infested_upstream": [1.0, np.nan], "hops_to_infested_downstream": [np.nan, 2.0],
                       "lat": [1.0, 2.0]})
    X = MG.design(df, ["hops_from_infested_upstream", "hops_to_infested_downstream", "lat"])
    assert X.isna().sum().sum() == 0 and X.iloc[1, 0] == MG.MAX_HOPS + 1


def test_kanankege_scores_rank_near_infested_higher():
    rs = np.random.default_rng(0)
    n = 400
    u = pd.DataFrame({"huc12": [f"{i:012d}" for i in range(n)], "lat": 40 + rs.uniform(0, 2, n),
                      "lon": -90 + rs.uniform(0, 2, n)})
    near = (u["lat"] < 40.4) & (u["lon"] < -89.6)
    infested = set(u.loc[near, "huc12"].iloc[::2])
    MG._KNN_CACHE.clear()
    sc = MG.kanankege_scores(u, infested, rs.normal(size=n), ranges=(20.0,), k=16, chunk=100)[20.0]
    cand = ~u["huc12"].isin(infested)
    assert sc[(cand & near).to_numpy()].mean() > sc[(cand & ~near).to_numpy()].mean() + 0.05


def test_population_pressure_counts_within_radius():
    u = pd.DataFrame({"lat": [40.0], "lon": [-90.0]})
    counties = pd.DataFrame({"lat": [40.1, 40.8], "lon": [-90.0, -90.0], "pop2010": [999.0, 1e6]})
    f = MG.population_pressure(u, counties)
    assert abs(f["log_pop_25km"].iloc[0] - 3.0) < 1e-9 and f["log_pop_100km"].iloc[0] > 6.0
