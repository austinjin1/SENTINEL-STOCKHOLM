# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the Track N / auxiliary target label rules (sentinel/models/orbit_v2/targets.py; amendment §2.4).

Synthetic result rows only, except one schema test that reads KEY columns of the real tables (no value column, no
detection limit). No protected value is read anywhere in this file."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from sentinel.data import matchups as M
from sentinel.models.orbit import split as SP
from sentinel.models.orbit_v2 import targets as T

SSC_NAME = M.WQP_TARGETS["ssc"][0]


def _rows(recs, *, variable="tp", unit=None, fraction="total"):
    """recs: (site, date, depth, rel, value, censor_type, dl[, characteristic[, status[, fraction]]])"""
    out = []
    for r in recs:
        r = list(r) + [None] * (10 - len(r))
        site, date, depth, rel, value, ctype, dl, char, status, frac = r
        out.append({"site_id": site, "activity_start_date": pd.Timestamp(date).date(), "depth_m": depth,
                    "activity_relative_depth": rel, "status_class": status or "accepted",
                    "fraction_class": frac or fraction, "censored": bool(ctype), "censor_type": ctype or "",
                    "characteristic": char or "Phosphorus", "unit": unit or T.SPECS[variable].unit,
                    "value": np.nan if ctype else value, "detection_limit": dl, "variable": variable})
    df = pd.DataFrame(out)
    df["depth_m"] = df["depth_m"].astype("float32")
    return df


def _lab(df, site, date):
    m = (df.site_id == site) & (df.date == pd.Timestamp(date))
    assert m.sum() == 1, (site, date, int(m.sum()))
    return df[m].iloc[0]


# ------------------------------------------------------------------------------------ specs
def test_specs_match_the_amendment_table():
    assert T.TRACK_N == ("tp", "tn", "doc") and T.AUX == ("turbidity_fnu", "ssc", "acdom440")
    got = {t: (s.unit, s.lo, s.hi, s.fractions) for t, s in T.SPECS.items()}
    assert got == {"tp": ("mg/L as P", 0.001, 10.0, ("total",)), "tn": ("mg/L as N", 0.01, 50.0, ("total",)),
                   "doc": ("mg/L as C", 0.1, 200.0, None), "turbidity_fnu": ("FNU", 0.05, 5000.0, None),
                   "ssc": ("mg/L", 0.1, 5000.0, None), "acdom440": ("1/m", 0.01, 50.0, None)}
    assert T.JOIN_RADIUS_M == 100.0
    assert T.dev_date_max() == dt.date(2023, 12, 31)


# ------------------------------------------------------------------------------------ surface rule
def test_surface_rule_depth_bottom_rejected():
    res = _rows([("A", "2020-06-01", 0.5, "", 0.10, "", None),          # surface
                 ("A", "2020-06-01", 1.0, "", 0.30, "", None),          # depth == 1.0 m is surface
                 ("A", "2020-06-01", 1.01, "", 9.0, "", None),          # deep
                 ("A", "2020-06-01", None, "", 0.20, "", None),         # unknown depth is surface
                 ("A", "2020-06-01", 0.2, "Bottom", 9.0, "", None),     # flagged bottom: never, at any depth
                 ("A", "2020-06-01", None, "near bottom", 9.0, "", None),
                 ("A", "2020-06-01", 0.3, "", 9.0, "", None, None, "rejected"),
                 ("B", "2020-06-01", 12.0, "", 0.5, "", None),          # only a deep result: no label
                 ("C", "2020-06-01", 0.1, "Surface", 0.7, "", None)])
    assert T.surface_mask(res).tolist() == [True, True, False, True, False, False, False, False, True]
    lab, counts = T.site_date_labels(res, "tp")
    assert sorted(lab.site_id) == ["A", "C"]
    a = _lab(lab, "A", "2020-06-01")
    assert a.value == pytest.approx(0.20) and a.n_results == 3 and a.n_quant == 3 and not a.cens
    assert counts["n_site_dates_with_surface_result"] == 2
    with pytest.raises(T.TargetRuleError, match="depth_m"):
        T.surface_mask(res.drop(columns=["depth_m"]))


def test_same_results_and_medians_as_the_shared_function():
    """The predicate here must select what the unchanged ``wqp_surface_samples`` selects (depth handed over as
    ``result_depth`` in metres) and give the same per-(site, date, characteristic) medians, on a target both know."""
    rng = np.random.default_rng(0)
    n = 4000
    depth = np.where(rng.random(n) < 0.4, np.nan, rng.choice([0.0, 0.3, 1.0, 1.0001, 2.5, 30.0], n))
    rel = rng.choice(["", "", "", "Surface", "Bottom", "Near Bottom", "Midwater"], n)
    cens = rng.random(n) < 0.15
    recs = [(f"S{rng.integers(60)}", f"2019-07-{1 + rng.integers(9):02d}", None if np.isnan(d) else float(d), r,
             float(np.round(rng.lognormal(1.0, 1.0), 3)), "<" if c else "", 0.5 if c else None,
             M.WQP_TARGETS["ssc"][int(rng.integers(2))], rng.choice(["accepted", "preliminary", "rejected"]))
            for d, r, c in zip(depth, rel, cens)]
    res = _rows(recs, variable="ssc")
    shared = M.wqp_surface_samples(res.assign(result_depth=res.depth_m.astype(float), result_depth_unit="m"), None)
    mine = T.per_characteristic(res, "ssc")
    k = ["site_id", "date", "characteristic"]
    j = shared.merge(mine, on=k, how="outer", indicator=True)
    assert (j["_merge"] == "both").all() and len(j) == len(shared) == len(mine) > 300
    assert (j.n_results_x == j.n_results_y).all()
    assert (j.n_censored == j.n_left).all()
    np.testing.assert_allclose(j.value.to_numpy(float), j.q_med.to_numpy(float), rtol=0, atol=0, equal_nan=True)
    assert (j.sample_depth_m.isna() | (j.sample_depth_m <= M.SURFACE_MAX_M)).all()


# ------------------------------------------------------------------------------------ fraction, replicates
def test_fraction_rule_and_replicate_medians():
    res = _rows([("A", "2020-06-01", 0.5, "", 0.10, "", None, "Phosphorus"),
                 ("A", "2020-06-01", 0.5, "", 0.20, "", None, "Phosphorus"),
                 ("A", "2020-06-01", 0.5, "", 0.60, "", None, "Phosphorus"),          # median 0.20
                 ("A", "2020-06-01", 0.5, "", 0.40, "", None, "Total Phosphorus, mixed forms"),   # median 0.40
                 ("A", "2020-06-01", 0.5, "", 5.00, "", None, "Phosphorus", None, "unspecified"),  # not total
                 ("A", "2020-06-02", 0.5, "", 0.30, "", None, "Phosphorus", None, "dissolved")])
    lab, _ = T.site_date_labels(res, "tp")
    assert len(lab) == 1
    a = lab.iloc[0]
    assert a.value == pytest.approx(0.30) and a.n_char == 2 and a.n_results == 4     # median of (0.20, 0.40)
    # DOC: every row of the table, whatever its fraction class
    doc = _rows([("A", "2020-06-01", 0.5, "", 3.0, "", None, "Organic carbon", None, "dissolved"),
                 ("A", "2020-06-01", 0.5, "", 5.0, "", None, "Organic carbon", None, "unspecified")], variable="doc")
    lab, _ = T.site_date_labels(doc, "doc")
    assert lab.iloc[0].value == pytest.approx(4.0)


def test_unexpected_unit_is_refused():
    res = _rows([("A", "2020-06-01", 0.5, "", 0.1, "", None)])
    res.loc[0, "unit"] = "ug/L"
    with pytest.raises(T.TargetRuleError, match="unit"):
        T.site_date_labels(res, "tp")


# ------------------------------------------------------------------------------------ censoring, valid range
def test_censoring_rules():
    res = _rows([("Q", "2020-06-01", 0.5, "", 0.05, "", None),           # quantified + censored: quantified median
                 ("Q", "2020-06-01", 0.5, "", None, "<", 0.02),
                 ("L", "2020-06-01", 0.5, "", None, "<", 0.01),          # left-censored only: largest DL
                 ("L", "2020-06-01", 0.5, "", None, "<", 0.02),
                 ("R", "2020-06-01", 0.5, "", None, ">", 5.0),           # right-censored: dropped
                 ("N", "2020-06-01", 0.5, "", None, "<", None),          # censored without a DL: dropped
                 ("D", "2020-06-01", 5.0, "", 0.05, "", None),           # quantified result is deep ...
                 ("D", "2020-06-01", 0.5, "", None, "<", 0.02),          # ... the surface result is censored
                 ("Z", "2020-06-01", 0.5, "", None, "<", 0.0)])          # DL of zero is not a detection limit
    lab, counts = T.site_date_labels(res, "tp")
    assert sorted(lab.site_id) == ["D", "L", "Q"]
    q, l, d = (_lab(lab, s, "2020-06-01") for s in ("Q", "L", "D"))
    assert q.value == pytest.approx(0.05) and not q.cens and np.isnan(q.dl)
    assert np.isnan(l.value) and l.cens and l.dl == pytest.approx(0.02) and l.n_left == 2 and l.n_quant == 0
    assert np.isnan(d.value) and d.cens and d.dl == pytest.approx(0.02)
    assert counts["n_no_usable_result_dropped"] == 3 and counts["n_left_censored_dl_in_range"] == 2
    assert counts["n_quantified_in_range"] == 1


def test_valid_range_sets_missing_and_never_clips():
    lo, hi = T.SPECS["tp"].lo, T.SPECS["tp"].hi
    res = _rows([("LO", "2020-06-01", 0.5, "", lo / 2, "", None),
                 ("HI", "2020-06-01", 0.5, "", hi * 2, "", None),
                 ("HI", "2020-06-01", 0.5, "", None, "<", 0.02),         # not replaced by the censored label
                 ("EL", "2020-06-01", 0.5, "", lo, "", None),            # range limits are inside
                 ("EH", "2020-06-01", 0.5, "", hi, "", None),
                 ("CL", "2020-06-01", 0.5, "", None, "<", lo / 2),       # DL below the range: dropped
                 ("CH", "2020-06-01", 0.5, "", None, "<", hi * 2),
                 ("CI", "2020-06-01", 0.5, "", None, "<", lo)])
    lab, counts = T.site_date_labels(res, "tp")
    assert sorted(lab.site_id) == ["CI", "EH", "EL"]
    assert _lab(lab, "EL", "2020-06-01").value == lo and _lab(lab, "EH", "2020-06-01").value == hi
    assert counts["n_quantified_out_of_range_set_missing"] == 2
    assert counts["n_left_censored_dl_out_of_range_dropped"] == 2
    v = lab.value.dropna()
    assert ((v >= lo) & (v <= hi)).all()


# ------------------------------------------------------------------------------------ readers (leakage rules)
def _table(tmp_path, res, name="variable=tp"):
    d = tmp_path / "wqp_discrete" / name
    d.mkdir(parents=True)
    res.to_parquet(d / "data.parquet", index=False)
    return d / "data.parquet"


def _sites(**roles):
    return pd.DataFrame({"site_id": list(roles), "role": list(roles.values()), "buffer_excluded": False})


def test_read_results_only_development_sites_and_years(tmp_path):
    res = _rows([("TR", "2019-06-01", 0.5, "", 0.11, "", None), ("VA", "2022-06-01", 0.5, "", 0.12, "", None),
                 ("TR", "2023-12-31", 0.5, "", 0.13, "", None), ("TR", "2024-01-01", 0.5, "", 0.14, "", None),
                 ("TE", "2019-06-01", 0.5, "", 0.15, "", None), ("TE", "2024-06-01", 0.5, "", 0.16, "", None),
                 ("XX", "2019-06-01", 0.5, "", 0.17, "", None)])
    f = _table(tmp_path, res)
    got = T.read_results("tp", _sites(TR="train", VA="val"), path=f)
    assert sorted(got.value) == [0.11, 0.12, 0.13]                        # no test site, no site outside the list,
    assert set(got.site_id) == {"TR", "VA"}                               # no 2024+ date
    lab, _ = T.dev_labels("tp", _sites(TR="train", VA="val"), path=f)
    assert len(lab) == 3 and set(SP.time_block(lab.date)) <= {"train", "val"}
    assert len(T.read_results("tp", _sites(TR="train"), date_max="2021-12-31", path=f)) == 1
    for bad in (_sites(TR="train", TE="test"), _sites(TE="test"), _sites(TR="")):
        with pytest.raises(T.TargetRuleError, match="development sites only"):
            T.read_results("tp", bad, path=f)
    buf = _sites(TR="train")
    buf["buffer_excluded"] = True
    with pytest.raises(T.TargetRuleError, match="development sites only"):
        T.read_results("tp", buf, path=f)
    with pytest.raises(T.TargetRuleError, match="after the last development day"):
        T.read_results("tp", _sites(TR="train"), date_max="2024-01-01", path=f)
    with pytest.raises(T.TargetRuleError, match="columns"):
        T.read_results("tp", pd.DataFrame({"site_id": ["TR"]}), path=f)
    with pytest.raises(T.TargetRuleError, match="empty site list"):
        T.read_results("tp", _sites(), path=f)
    two = pd.DataFrame({"site_id": ["TR", "TR"], "role": ["train", "val"], "buffer_excluded": False})
    with pytest.raises(T.TargetRuleError, match="two roles"):
        T.read_results("tp", two, path=f)


def test_sidecar_files_are_never_a_source(tmp_path):
    res = _rows([("TR", "2019-06-01", 0.5, "", 0.11, "", None)])
    for name in ("_qc_rejects.parquet", "_unmapped.parquet"):
        f = tmp_path / name
        res.to_parquet(f, index=False)
        with pytest.raises(T.TargetRuleError, match="not a harmonised"):
            T.read_results("tp", _sites(TR="train"), path=f)
        with pytest.raises(T.TargetRuleError, match="not a harmonised"):
            T.read_keys("tp", path=f)
    assert all(T.source_file(t).parent.name == f"variable={t}" for t in T.SPECS)


def test_key_reader_has_no_value_and_counts_are_value_free(tmp_path):
    res = _rows([("TE", "2019-06-01", 0.5, "", 0.15, "", None), ("TE", "2019-06-01", 0.5, "", None, "<", 0.02),
                 ("TE", "2019-06-02", 0.5, "", None, "<", 0.02), ("TE", "2019-06-03", 7.0, "", 0.2, "", None),
                 ("TE", "2019-06-04", 0.5, "", 0.3, "", None, None, None, "dissolved"),
                 ("TE", "2024-06-01", 0.5, "", 0.16, "", None), ("TR", "2019-06-01", 0.5, "", 0.11, "", None)])
    f = _table(tmp_path, res)
    keys = T.read_keys("tp", ["TE"], date_min="2015-06-23", date_max="2023-12-31", path=f)
    assert not set(T.VALUE_COLS) & set(keys.columns) and set(keys.columns) == set(T.KEY_COLS)
    assert set(keys.site_id) == {"TE"} and len(keys) == 5
    kc = T.key_counts(keys, "tp")
    assert kc.date.dt.strftime("%m-%d").tolist() == ["06-01", "06-02"]    # deep-only and non-total days: no key
    assert kc.n_quant.tolist() == [1, 0] and kc.n_left.tolist() == [1, 1]
    assert not {"value", "dl", "detection_limit"} & set(kc.columns)
    with pytest.raises(T.TargetRuleError, match="value column"):
        T.key_counts(res, "tp")


def test_real_tables_have_the_schema_the_rule_assumes():
    """KEY columns only (schema and vocabulary); skipped when the processed tables are not on this machine."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    for t, sp in T.SPECS.items():
        f = T.source_file(t)
        if not f.exists():
            pytest.skip(f"{f} not present")
        names = set(pq.ParquetFile(f).schema_arrow.names)
        assert {*T.KEY_COLS, *T.VALUE_COLS} <= names and "qc_class" not in names
        tab = pq.read_table(f, columns=["unit", "censor_type", "status_class", "fraction_class"])
        assert set(pc.unique(tab["unit"]).to_pylist()) == {sp.unit}
        assert set(pc.unique(tab["censor_type"]).to_pylist()) <= {"", "<", ">"}
        assert "rejected" not in set(pc.unique(tab["status_class"]).to_pylist())
        if sp.fractions:
            assert set(sp.fractions) <= set(pc.unique(tab["fraction_class"]).to_pylist())


# ------------------------------------------------------------------------------------ joins
def test_join_site_date_keeps_rows_and_order():
    res = _rows([("A", "2020-06-01", 0.5, "", 0.10, "", None), ("B", "2020-06-02", 0.5, "", None, "<", 0.02)])
    lab, _ = T.site_date_labels(res, "tp")
    rows = pd.DataFrame({"site_id": ["B", "A", "A", "C"],
                         "date": pd.to_datetime(["2020-06-02", "2020-06-01", "2020-06-03", "2020-06-01"])},
                        index=[10, 11, 12, 13])
    j = T.join_site_date(rows, lab, "tp")
    assert j.index.tolist() == [10, 11, 12, 13] and list(j.columns) == ["tp", "tp_cens", "tp_dl"]
    np.testing.assert_allclose(j.tp.to_numpy(), [np.nan, 0.10, np.nan, np.nan], equal_nan=True)
    assert j.tp_cens.tolist() == [True, False, False, False]
    np.testing.assert_allclose(j.tp_dl.to_numpy(), [0.02, np.nan, np.nan, np.nan], equal_nan=True)
    with pytest.raises(T.TargetRuleError, match="unique"):
        T.join_site_date(rows, pd.concat([lab, lab]), "tp")


def test_nearest_station_same_waterbody_and_role_only():
    """R-8 / C-9: one station per row, the nearest within 100 m whose water body AND split role equal the row's."""
    m = 1.0 / 111_195.0                                                   # one metre of latitude, in degrees
    st = pd.DataFrame({"site_id": ["S50", "S90", "S150", "T50", "V30"], "lon": [0.0, 0.0, 0.0, 10.0, 0.0],
                       "lat": [50 * m, -90 * m, 150 * m, 45.0 + 50 * m, 30 * m],
                       "waterbody": ["L1", "L2", "L1", "L9", "L2"], "role": ["train", "train", "train", "val", "val"]})
    rows = pd.DataFrame({"lon": [0.0, 0.0, 10.0, np.nan, 20.0, 0.0, 10.0], "lat": [0.0, 60 * m, 45.0, 0.0, 0.0, 0.0, 45.0],
                         "waterbody": ["L2", "L1", "L9", "L1", "L1", "L3", "L9"],
                         "role": ["train", "train", "val", "train", "train", "train", "train"]}, index=list("abcdefg"))
    got = T.nearest_station(rows, st)
    assert got.index.tolist() == list("abcdefg")
    # a: the 30 m station is the row's lake but another role, the 50 m station another lake -> the 90 m station
    # f: no station of the row's water body; g: the only station of the lake has another role -> no value
    assert got.join_site_id.tolist() == ["S90", "S50", "T50", None, None, None, None]
    np.testing.assert_allclose(got.join_dist_m.to_numpy()[:3], [90.0, 10.0, 50.0], rtol=2e-3)
    far = T.nearest_station(rows.iloc[[1]].assign(lat=0.0), st[st.site_id == "S150"])
    assert far.join_site_id.tolist() == [None]                                 # 150 m: outside the radius


def test_nearest_station_ties_go_to_the_lower_station_id():
    m = 1.0 / 111_195.0
    st = pd.DataFrame({"site_id": ["USGS-9", "USGS-10", "USGS-2", "AAA-far"], "lon": [0.0, 0.0, 0.0, 0.0],
                       "lat": [40 * m, -40 * m, 40 * m, 41 * m], "waterbody": "L1", "role": "train"})
    rows = pd.DataFrame({"lon": [0.0], "lat": [0.0], "waterbody": ["L1"], "role": ["train"]})
    for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        got = T.nearest_station(rows, st.iloc[perm].reset_index(drop=True))
        assert got.join_site_id.tolist() == ["USGS-10"]                        # string order of the station id
    assert T.nearest_station(rows, st[st.site_id != "USGS-10"]).join_site_id.tolist() == ["USGS-2"]
    assert T.nearest_station(rows, st[st.site_id == "AAA-far"]).join_site_id.tolist() == ["AAA-far"]


def test_r8_is_a_constant_without_a_switch():
    """C-2: no setting of the join other than R-8; keys only."""
    import inspect

    assert list(inspect.signature(T.nearest_station).parameters) == ["rows", "stations"]
    assert "frozen-rule water body and split role equal the row's" in T.RULE_R8
    assert T.JOIN_RADIUS_M == 100.0
    st = pd.DataFrame({"site_id": ["S"], "lon": [0.0], "lat": [0.0], "waterbody": ["L1"], "role": ["train"]})
    rows = pd.DataFrame({"lon": [0.0], "lat": [0.0], "waterbody": ["L1"], "role": ["train"]})
    with pytest.raises(TypeError):
        T.nearest_station(rows, st, same_waterbody=False)
    for drop in ("waterbody", "role"):
        with pytest.raises(T.TargetRuleError, match="key columns"):
            T.nearest_station(rows.drop(columns=drop), st)
        with pytest.raises(T.TargetRuleError, match="key columns"):
            T.nearest_station(rows, st.drop(columns=drop))
    for c in T.VALUE_COLS:
        with pytest.raises(T.TargetRuleError, match="value column"):
            T.nearest_station(rows, st.assign(**{c: 1.0}))
        with pytest.raises(T.TargetRuleError, match="value column"):
            T.nearest_station(rows.assign(**{c: 1.0}), st)


def test_c9_all_censored_site_date_rule():
    """C-9: censored at the largest DL of the site-date only when every surface result is left-censored and that
    limit is inside the a-priori range; otherwise no value. Each outcome is counted."""
    lo, hi = T.SPECS["tp"].lo, T.SPECS["tp"].hi
    res = _rows([("ALL", "2020-06-01", 0.5, "", None, "<", 0.01),          # all left-censored: largest DL
                 ("ALL", "2020-06-01", 0.5, "", None, "<", 0.03),
                 ("ALL", "2020-06-01", 0.5, "", None, "<", 0.02, "Total Phosphorus, mixed forms"),
                 ("NODL", "2020-06-01", 0.5, "", None, "<", 0.02),         # one "<" result has no DL: still all left
                 ("NODL", "2020-06-01", 0.5, "", None, "<", None),
                 ("MIX", "2020-06-01", 0.5, "", None, "<", 0.02),          # a right-censored result too: not all left
                 ("MIX", "2020-06-01", 0.5, "", None, ">", 5.0),
                 ("OUT", "2020-06-01", 0.5, "", None, "<", 0.02),          # largest DL outside the range: no value
                 ("OUT", "2020-06-01", 0.5, "", None, "<", hi * 2),
                 ("LOW", "2020-06-01", 0.5, "", None, "<", lo / 2)])
    lab, counts = T.site_date_labels(res, "tp")
    assert sorted(lab.site_id) == ["ALL", "NODL"]
    a, n = _lab(lab, "ALL", "2020-06-01"), _lab(lab, "NODL", "2020-06-01")
    assert a.cens and np.isnan(a.value) and a.dl == pytest.approx(0.03) and a.n_results == 3 and a.n_char == 2
    assert n.cens and n.dl == pytest.approx(0.02)
    assert counts["n_left_censored_dl_in_range"] == 2 and counts["n_left_censored_dl_out_of_range_dropped"] == 2
    assert counts["n_not_all_results_left_censored_no_value"] == 1 and counts["n_no_usable_result_dropped"] == 0
    assert sum(v for k, v in counts.items() if k != "n_site_dates_with_surface_result") == 5


def test_replicate_median_key_includes_the_unit():
    """R-3: replicate median per site, local date, target, characteristic and unit."""
    assert T.REPLICATE_KEYS == ("site_id", "date", "characteristic", "unit")
    res = _rows([("A", "2020-06-01", 0.5, "", 0.10, "", None), ("A", "2020-06-01", 0.5, "", 0.30, "", None),
                 ("A", "2020-06-02", 0.5, "", 0.50, "", None)])
    pc = T.per_characteristic(res, "tp")
    assert list(pc.columns[:4]) == list(T.REPLICATE_KEYS) and len(pc) == 2
    assert pc.q_med.tolist() == pytest.approx([0.20, 0.50]) and pc.n_left_any.tolist() == [0, 0]
