# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for SENTINEL-Lite architecture round v2 (sentinel/models/litev2). Synthetic rows only (RESEARCH_PLAN R9)."""

from __future__ import annotations

import ast
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from sentinel.data import matchups as M
from sentinel.models.litev2 import wqp_rule as WR
from sentinel.models.orbit import wqp_surface as WS

CHLA, SECCHI, TURB = "Chlorophyll a", "Depth, Secchi disk depth", "Turbidity"
D1 = "2020-06-01"


def _rows(spec):
    """(site, date, characteristic, unit, value, depth_m, relative_depth) -> processed-schema result rows."""
    df = pd.DataFrame(spec, columns=["site_id", "activity_start_date", "characteristic", "unit", "value", "depth_m",
                                     "activity_relative_depth"])
    df["censored"] = False
    df["detection_limit"] = np.nan
    return df


HARM = [
    ("S1", D1, CHLA, "ug/L", 10.0, 0.2, None), ("S1", D1, CHLA, "ug/L", 20.0, 0.8, None),       # surface only
    ("S2", D1, CHLA, "ug/L", 30.0, 5.0, None),                                                   # deep only
    ("S3", D1, TURB, "NTU", 4.0, 0.5, None), ("S3", D1, TURB, "NTU", 6.0, 0.9, None),          # mixed
    ("S3", D1, TURB, "NTU", 100.0, 8.0, None),
    ("S4", D1, TURB, "FNU", 7.0, np.nan, None),                                                  # unknown depth
    ("S5", D1, SECCHI, "m", 2.5, 3.0, None),                                                     # Secchi exempt
    ("S6", D1, CHLA, "ug/L", 99.0, 0.5, "Bottom"),                                               # bottom-flagged only
]
SIDECAR = [("S7", D1, CHLA, "ug/L", 55.0, np.nan, None),                                        # QC-reject rows only
           ("S1", D1, CHLA, "ug/L", 500.0, np.nan, None)]                                       # next to QC-passed rows


def _side():
    harm = _rows(HARM)
    res_all = pd.concat([harm.drop(columns=["depth_m"]), _rows(SIDECAR).drop(columns=["depth_m"])], ignore_index=True)
    return WS.surface_table(harm, res_all)


def _built(side, chips_per_sample=1):
    """A Lite matchup table as the national builds wrote it: depth-blind values, one row per sample x chip."""
    rows = []
    for _, s in side.iterrows():
        site, date, target, char, unit = s["sample_key"].split("|")
        for c in range(chips_per_sample):
            rows.append({"site_id": site, "date": pd.Timestamp(date), "target": target, "characteristic": char,
                         "unit": unit, "value": s["_value_all"], "n_results": int(s["n_results_all"]),
                         "sample_depth_m": np.nan, "depth_rule": "unspecified", "naip_chip_id": f"{site}-{target}-{c}"})
    return pd.DataFrame(rows)


def _key(site, target, char, unit):
    return f"{site}|{D1}|{target}|{char}|{unit}"


# ------------------------------------------------------------------------------ the reader is M2's, not a copy

def test_reader_is_the_orbit_module_and_holds_no_rule_of_its_own():
    assert WR.WS is WS and WR.RULE is WS.RULE and WR.DROP_REASONS is WS.DROP_REASONS
    tree = ast.parse(inspect.getsource(WR))
    for node in ast.walk(tree):                                                 # drop docstrings (comments go too)
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    code = ast.unparse(tree)
    for banned in ("wqp_surface_samples(", "SURFACE_MAX_M", "load_wqp(", "_qc_rejects", "_unmapped", "<= 1"):
        assert banned not in code, banned
    assert "WS.apply(" in code and "WS.surface_table(" in code and "WS.load_side(" in code


def test_apply_rule_replaces_values_drops_and_records_reasons():
    side = _side()
    built = _built(side, chips_per_sample=2)
    logs = []
    kept, dropped, rec = WR.apply_rule(built, "national", side=side, side_sha256="abc", log=logs.append)
    got = kept.drop_duplicates("sample_key").set_index("sample_key")
    assert got.loc[_key("S1", "chla", CHLA, "ug/L"), "value"] == 15.0           # the sidecar row (500) is out
    assert got.loc[_key("S1", "chla", CHLA, "ug/L"), "n_results"] == 2
    assert got.loc[_key("S3", "turbidity", TURB, "NTU"), "value"] == 5.0        # the deep result (100) is out
    assert got.loc[_key("S3", "turbidity", TURB, "NTU"), "sample_depth_m"] == pytest.approx(0.5)
    assert got.loc[_key("S5", "secchi", SECCHI, "m"), "value"] == 2.5
    assert built.loc[built.site_id == "S3", "value"].iloc[0] == 6.0             # the input is not modified
    why = dropped.drop_duplicates("sample_key").set_index("sample_key")["drop_reason"].to_dict()
    assert why == {_key("S2", "chla", CHLA, "ug/L"): "no_surface_result",
                   _key("S7", "chla", CHLA, "ug/L"): "no_qc_passed_result"}
    # a bottom-flagged result never made a sample, depth-blind or not: S6 is not a key of either table
    assert not side["sample_key"].str.startswith("S6|").any() and not (built.site_id == "S6").any()
    assert len(kept) == 2 * 4 and len(dropped) == 2 * 2 and len(kept) + len(dropped) == len(built)
    assert "_set" not in kept.columns and "value" not in dropped.columns
    assert rec["n_rows"] == 12 and rec["n_rows_kept"] == 8 and rec["n_rows_dropped"] == 4
    assert rec["by_target"]["chla"] == {"n_rows": 6, "dropped_no_qc_passed_result": 2, "dropped_no_surface_result": 2,
                                        "kept_with_fewer_results": 2}
    assert rec["by_target"]["turbidity"]["kept_with_fewer_results"] == 2
    assert rec["side_table_sha256"] == "abc" and rec["rule"] == WS.RULE
    assert rec["built_table_check"] == {"n_rows": 12, "n_keys": 6, "n_key_missing": 0, "n_results_differs": 0}
    assert WR.reason_counts(dropped) == {"chla|no_qc_passed_result": 2, "chla|no_surface_result": 2}
    json.dumps(rec)                                                              # the record is plain JSON


def test_check_built_refuses_a_table_that_is_not_the_depth_blind_pass():
    side = _side()
    built = _built(side)
    assert WR.check_built(built, "national", side)["n_key_missing"] == 0
    extra = built.iloc[[0]].assign(site_id="ZZ")
    with pytest.raises(WR.LabelRuleError, match="'n_key_missing': 1"):
        WR.check_built(pd.concat([built, extra]), "national", side)
    with pytest.raises(WS.SurfaceRuleError, match="not in the side table"):     # M2's reader refuses as well
        WS.apply(WR.keyed(pd.concat([built, extra]), "national"), log=lambda *_: None, side=side)
    wrong = built.copy()
    wrong.loc[0, "n_results"] += 1
    with pytest.raises(WR.LabelRuleError, match="'n_results_differs': 1"):
        WR.apply_rule(wrong, "national", side=side, log=lambda *_: None)


def test_keyed_uses_the_orbit_key_and_refuses_a_foreign_one():
    side = _side()
    built = _built(side)
    k = WR.keyed(built, "national_pm7")
    assert (k["sample_key"] == WS.sample_key(built)).all() and set(k["_set"]) == {"national_pm7"}
    assert sorted(k["sample_key"]) == sorted(side["sample_key"])
    with pytest.raises(WR.LabelRuleError, match="1 stored sample keys differ"):
        WR.keyed(k.assign(sample_key=["x"] + list(k["sample_key"][1:])).drop(columns="_set"), "national_pm7")
    with pytest.raises(WR.LabelRuleError, match="key column"):
        WR.keyed(built.drop(columns=["unit"]), "national")


def test_on_disk_side_table_needs_every_ruling(monkeypatch):
    monkeypatch.setattr(WS, "RULINGS", {"depth_rule": "ruled", "sidecar_exclusion": None})
    with pytest.raises(WS.SurfaceRuleError, match="not ruled yet"):
        WR.apply_rule(_built(_side()), "national", log=lambda *_: None)


def test_apply_rule_inherits_the_depth_guard_of_the_orbit_reader():
    side = _side()
    side["sample_depth_m"] = np.nan                                             # a side table built without depth
    with pytest.raises(WS.SurfaceRuleError, match="sample_depth_m is null for all"):
        WR.apply_rule(_built(_side()), "national", side=side, log=lambda *_: None)


# ------------------------------------------------------------------------------ P2: rows read from one variable

def _fnu(spec):
    w = _rows([(s, d, "Turbidity", "FNU", v, z, rel) for s, d, v, z, rel in spec])
    return w


P2 = [("A", D1, 3.0, 0.3, None), ("A", D1, 5.0, 0.6, None), ("A", D1, 80.0, 6.0, None),      # mixed
      ("B", D1, 40.0, 4.0, None),                                                            # all deeper than 1 m
      ("C", D1, 9.0, np.nan, None),                                                          # unknown depth
      ("D", D1, 9.0, 0.5, "Bottom"),                                                         # bottom-flagged
      ("E", D1, 2.0, 1.0, None), ("E", "2020-06-02", 8.0, 1.01, None)]                       # the 1.0 m boundary


def test_surface_samples_is_the_shared_function_with_depth():
    w = _fnu(P2)
    s = WR.surface_samples(w, target="turbidity")
    assert list(s.columns[:6]) == [*WR.KEY_COLS, "sample_key"]
    got = s.set_index(["site_id", "date"])["value"].to_dict()
    assert got == {("A", pd.Timestamp(D1)): 4.0, ("C", pd.Timestamp(D1)): 9.0, ("E", pd.Timestamp(D1)): 2.0}
    direct = M.wqp_surface_samples(w.drop(columns=["depth_m"]).assign(variable="turbidity", result_depth=w["depth_m"],
                                                                     result_depth_unit="m"), None)
    assert sorted(WS.sample_key(direct)) == sorted(s["sample_key"])
    np.testing.assert_allclose(direct.sort_values("site_id")["value"], s.sort_values("site_id")["value"])
    blind = M.wqp_surface_samples(w.drop(columns=["depth_m"]).assign(variable="turbidity"), None)
    assert len(blind) == 5 and len(s) == 3                                      # what the depth-blind build kept


def test_surface_samples_works_on_a_presence_flag_and_needs_depth():
    w = _fnu(P2)
    flag = w.assign(value=np.where(w["value"].notna(), 1.0, np.nan))
    s = WR.surface_samples(flag, target="turbidity")
    assert set(s["value"]) == {1.0} and sorted(s["site_id"]) == ["A", "C", "E"]
    with pytest.raises(WR.LabelRuleError, match="no depth_m"):
        WR.surface_samples(w.drop(columns=["depth_m"]), target="turbidity")
    with pytest.raises(WR.LabelRuleError, match="unknown target"):
        WR.surface_samples(w, target="turbidity_fnu")
    assert len(WR.surface_samples(w.iloc[:0], target="turbidity")) == 0


# ------------------------------------------------------------------------------ M2 Test-S flag (conflict C2)

from sentinel.models.litev2 import tests_mask as TM  # noqa: E402


def _assign(lon, lat):
    """Synthetic orbit assignment: lon < 0 -> in a test-role lake; lat < 0 -> buffer-excluded; lon >= 10 -> stream."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return pd.DataFrame({"in_lake": lon < 10, "role": np.where(lon < 0, "test", "train"),
                         "buffer_excluded": lat < 0})


def _chips():
    return pd.DataFrame({
        "naip_chip_id": ["a", "b", "c", "d", "e", "f", "g"],
        "lon": [-1.0, -1.0, -1.0, 5.0, -1.0, -1.0, -1.0],
        "lat": [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
        "naip_date": pd.to_datetime(["2019-07-01", "2019-07-01", "2014-07-01", "2019-07-01", "2019-07-01",
                                     "2024-01-01", "2015-06-23"]),
        "sample_date": pd.to_datetime(["2019-07-02", "2019-07-02", "2014-07-01", "2019-07-02", "2019-07-02",
                                       "2023-12-31", "2015-06-22"]),
        "s2_chip_id": ["s1", None, "s3", "s4", "s5", "s6", "s7"]})


def test_test_s_flags():
    f = TM.flags(_chips(), assign=_assign).set_index("naip_chip_id")
    assert f["on_orbit_test_lake"].to_dict() == {"a": True, "b": True, "c": True, "d": False, "e": False, "f": True,
                                                 "g": True}
    assert f["on_test_s"].to_dict() == {"a": True, "b": True, "c": False, "d": False, "e": False, "f": True, "g": True}
    assert f["on_test_s_naip_date"].to_dict()["f"] is np.False_ or not f.loc["f", "on_test_s_naip_date"]
    assert f["kd_masked"].to_dict() == {"a": True, "b": False, "c": False, "d": False, "e": False, "f": True,
                                        "g": True}
    c = TM.counts(f.reset_index(), pd.Series({"a": "train", "b": "train", "c": "val", "d": "val", "e": "val"}))
    assert c["train"] == {"chips": 2, "on_orbit_test_lake": 2, "on_test_s": 2, "on_test_s_naip_date": 2,
                          "kd_masked": 1}
    assert c["no role"]["kd_masked"] == 2
    with pytest.raises(TM.MaskError, match="duplicate"):
        TM.flags(pd.concat([_chips(), _chips()]), assign=_assign)
    with pytest.raises(TM.MaskError, match="missing"):
        TM.flags(_chips().drop(columns=["s2_chip_id"]), assign=_assign)


def test_teacher_values_never_reach_a_test_s_chip():
    chips = _chips()
    f = TM.flags(chips, assign=_assign)
    tab = chips[["naip_chip_id"]].assign(teacher_mu_secchi=1.0, teacher_sd_secchi=0.1, kd_w=1.0, y=2.0)
    cols = ["teacher_mu_secchi", "teacher_sd_secchi"]
    with pytest.raises(TM.MaskError, match="3 teacher values"):
        TM.assert_masked(tab, f, cols, kd_weight_col="kd_w")
    out, rec = TM.mask_teacher(tab, f, cols, kd_weight_col="kd_w")
    TM.assert_masked(out, f, cols, kd_weight_col="kd_w")
    m = f["kd_masked"].to_numpy()
    assert out.loc[m, cols].isna().all().all() and out.loc[~m, cols].notna().all().all()
    assert (out.loc[m, "kd_w"] == 0).all() and (out.loc[~m, "kd_w"] == 1).all()
    assert (out["y"] == 2.0).all() and len(out) == len(tab)                    # labels and rows stay
    assert rec == {"n_rows": 7, "n_kd_masked": 3, "n_teacher_values_removed": {c: 3 for c in cols}}
    assert tab["teacher_mu_secchi"].notna().all()                              # the input is not modified
    with pytest.raises(TM.MaskError, match="carry a KD weight"):
        TM.assert_masked(out.assign(kd_w=1.0), f, cols, kd_weight_col="kd_w")
    with pytest.raises(TM.MaskError, match="1 chips have no Test-S flag"):
        TM.mask_teacher(pd.concat([tab, tab.iloc[[0]].assign(naip_chip_id="zz")]), f, cols)


def test_test_s_window_is_the_reserved_one():
    assert str(TM.ARCHIVE_FROM.date()) == "2015-06-23" and str(TM.ARCHIVE_TO.date()) == "2023-12-31"
    assert list(TM._in_window(["2015-06-22", "2015-06-23", "2023-12-31", "2024-01-01", None])) == \
        [False, True, True, False, False]
