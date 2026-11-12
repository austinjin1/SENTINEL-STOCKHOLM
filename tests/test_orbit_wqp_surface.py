# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the WQP label rule of SENTINEL-Orbit (M2): depth rule + QC sidecar files (wqp_surface.py; lead ruling
2026-11-10). Synthetic rows only (RESEARCH_PLAN R9), except the two on-disk guards at the end, which read counts
and null masks of built tables, never a label value."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.data import matchups as M
from sentinel.models.orbit import data as D
from sentinel.models.orbit import wqp_surface as WS

CHLA, SECCHI, TURB = "Chlorophyll a", "Depth, Secchi disk depth", "Turbidity"


def _rows(spec):
    """spec: (site, date, characteristic, unit, value, depth_m, relative_depth) -> processed-schema result rows."""
    df = pd.DataFrame(spec, columns=["site_id", "activity_start_date", "characteristic", "unit", "value", "depth_m",
                                     "activity_relative_depth"])
    df["censored"] = False
    df["detection_limit"] = np.nan
    return df


def _key(site, date, target, char, unit):
    return f"{site}|{date}|{target}|{char}|{unit}"


SPEC = [
    # surface only
    ("S1", "2020-06-01", CHLA, "ug/L", 10.0, 0.2, None), ("S1", "2020-06-01", CHLA, "ug/L", 20.0, 0.8, None),
    # deep only
    ("S2", "2020-06-01", CHLA, "ug/L", 30.0, 5.0, None), ("S2", "2020-06-01", CHLA, "ug/L", 40.0, 12.0, None),
    # mixed: two surface results and one deep result
    ("S3", "2020-06-01", CHLA, "ug/L", 4.0, 0.5, None), ("S3", "2020-06-01", CHLA, "ug/L", 6.0, 0.9, None),
    ("S3", "2020-06-01", CHLA, "ug/L", 100.0, 8.0, None),
    # unknown depth
    ("S4", "2020-06-01", CHLA, "ug/L", 7.0, np.nan, None),
    # unknown depth next to a deep result
    ("S5", "2020-06-01", CHLA, "ug/L", 7.0, np.nan, None), ("S5", "2020-06-01", CHLA, "ug/L", 50.0, 10.0, None),
    # flagged bottom (never a surface result, whatever the depth) next to a surface result
    ("S6", "2020-06-01", CHLA, "ug/L", 99.0, 0.5, "Bottom"), ("S6", "2020-06-01", CHLA, "ug/L", 3.0, 0.5, "Surface"),
    # flagged bottom only, depth unknown
    ("S7", "2020-06-01", CHLA, "ug/L", 99.0, np.nan, "Near Bottom"),
    # Secchi depth is exempt (its "depth" is the measurement)
    ("S8", "2020-06-01", SECCHI, "m", 2.5, 3.0, None),
    # the boundary: 1.0 m is surface, anything deeper is not
    ("S9", "2020-06-01", TURB, "NTU", 1.5, 1.0, None), ("S9", "2020-06-01", TURB, "NTU", 9.5, 1.01, None),
]


def _side(spec=SPEC, extra_all=None):
    harm = _rows(spec)
    res_all = None
    if extra_all is not None:
        res_all = pd.concat([harm.drop(columns=["depth_m"]), _rows(extra_all).drop(columns=["depth_m"])],
                            ignore_index=True)
    return WS.surface_table(harm, res_all).set_index("sample_key")


# ----------------------------------------------------------------------------------------- the rule

def test_rule_on_synthetic_rows():
    s = _side()
    k = lambda site, t="chla", c=CHLA, u="ug/L": _key(site, "2020-06-01", t, c, u)  # noqa: E731
    # surface only: median of both results
    assert s.loc[k("S1"), "surface"] and s.loc[k("S1"), "value"] == 15.0 and s.loc[k("S1"), "n_results"] == 2
    assert s.loc[k("S1"), "sample_depth_m"] == pytest.approx(0.2)
    # deep only: dropped, and the reason says so
    assert not s.loc[k("S2"), "surface"] and s.loc[k("S2"), "drop_reason"] == "no_surface_result"
    assert np.isnan(s.loc[k("S2"), "value"]) and s.loc[k("S2"), "n_results"] == 0
    assert s.loc[k("S2"), "n_results_all"] == 2 and s.loc[k("S2"), "_value_all"] == 35.0
    # mixed: only the surface results are kept (the depth-blind value was the median of all three)
    assert s.loc[k("S3"), "surface"] and s.loc[k("S3"), "value"] == 5.0
    assert s.loc[k("S3"), "n_results"] == 2 and s.loc[k("S3"), "n_results_all"] == 3
    assert s.loc[k("S3"), "_value_all"] == 6.0 and s.loc[k("S3"), "value_max"] == 6.0
    # unknown depth: surface
    assert s.loc[k("S4"), "surface"] and s.loc[k("S4"), "value"] == 7.0 and np.isnan(s.loc[k("S4"), "sample_depth_m"])
    assert s.loc[k("S4"), "depth_rule"] == "unspecified"
    assert s.loc[k("S5"), "surface"] and s.loc[k("S5"), "value"] == 7.0 and s.loc[k("S5"), "n_results"] == 1
    # flagged bottom: never a surface result, with or without the depth column
    assert s.loc[k("S6"), "value"] == 3.0 and s.loc[k("S6"), "n_results"] == 1 and s.loc[k("S6"), "n_results_all"] == 1
    assert k("S7") not in s.index
    # Secchi: exempt
    ks = _key("S8", "2020-06-01", "secchi", SECCHI, "m")
    assert s.loc[ks, "surface"] and s.loc[ks, "value"] == 2.5 and s.loc[ks, "depth_rule"] == "exempt"
    # boundary
    kt = _key("S9", "2020-06-01", "turbidity", TURB, "NTU")
    assert s.loc[kt, "value"] == 1.5 and s.loc[kt, "n_results"] == 1 and s.loc[kt, "n_results_all"] == 2
    assert M.SURFACE_MAX_M == 1.0 and M.DEPTH_EXEMPT == ("secchi",)


def test_rule_is_the_unchanged_shared_function():
    """The labels are the output of ``matchups.wqp_surface_samples`` given depth_m as result_depth in metres."""
    harm = _rows(SPEC)
    direct = M.wqp_surface_samples(harm.drop(columns=["depth_m"]).assign(result_depth=harm["depth_m"],
                                                                         result_depth_unit="m"), None)
    direct["sample_key"] = WS.sample_key(direct)
    s = WS.surface_table(harm)
    kept = s[s.surface].set_index("sample_key")
    assert sorted(kept.index) == sorted(direct.sample_key)
    d = direct.set_index("sample_key").loc[kept.index]
    for c in ("value", "value_min", "value_max", "sample_depth_m"):
        np.testing.assert_allclose(kept[c].to_numpy(float), d[c].to_numpy(float), equal_nan=True)
    assert (kept["n_results"].to_numpy() == d["n_results"].to_numpy()).all()


def test_sample_key_is_the_matchup_builds_expression():
    df = pd.DataFrame({"site_id": ["USGS-01"], "date": [pd.Timestamp("2021-03-04")], "target": ["tss"],
                       "characteristic": ["Total suspended solids"], "unit": ["mg/L"]})
    assert WS.sample_key(df).tolist() == ["USGS-01|2021-03-04|tss|Total suspended solids|mg/L"]
    src = (paths.REPO_ROOT / "scripts" / "data" / "build_s2_wqp_matchups.py").read_text()
    assert ('smp["sample_key"] = (smp.site_id + "|" + smp.date.dt.strftime("%Y-%m-%d") + "|" + smp.target + "|" + '
            'smp.characteristic + "|" + smp.unit)') in " ".join(src.split())


def test_wqx_depth_fields_next_to_depth_m_are_refused():
    harm = _rows(SPEC).assign(result_depth=0.3)
    with pytest.raises(WS.SurfaceRuleError, match="WQX depth fields"):
        WS.surface_table(harm)
    with pytest.raises(WS.SurfaceRuleError, match="no depth_m"):
        WS.surface_table(_rows(SPEC).drop(columns=["depth_m"]))


# ----------------------------------------------------------------------------------------- sidecar rows

def test_sidecar_rows_never_contribute():
    extra = [
        # a QC blank on the day of a real surface sample (same key): the label ignores it
        ("S1", "2020-06-01", CHLA, "ug/L", 0.0, np.nan, None),
        # a sample made of rejected rows only: no label
        ("X1", "2020-06-02", CHLA, "ug/L", 55.0, np.nan, None),
        # an unmapped characteristic that the shared name list maps to chla
        ("X2", "2020-06-02", "Chlorophyll", "ug/L", 12.0, np.nan, None),
    ]
    s = _side(extra_all=extra)
    k1 = _key("S1", "2020-06-01", "chla", CHLA, "ug/L")
    assert s.loc[k1, "value"] == 15.0 and s.loc[k1, "n_results"] == 2 and s.loc[k1, "n_results_all"] == 3
    assert s.loc[k1, "_value_all"] == 10.0                      # what the matchup build stored (blank included)
    kx = _key("X1", "2020-06-02", "chla", CHLA, "ug/L")
    assert not s.loc[kx, "surface"] and s.loc[kx, "drop_reason"] == "no_qc_passed_result"
    assert s.loc[kx, "n_results_qc"] == 0 and s.loc[kx, "n_results_all"] == 1
    if "Chlorophyll" in M.NAME_TO_TARGET:
        ku = _key("X2", "2020-06-02", "chla", "Chlorophyll", "ug/L")
        assert not s.loc[ku, "surface"] and s.loc[ku, "drop_reason"] == "no_qc_passed_result"
    k2 = _key("S2", "2020-06-01", "chla", CHLA, "ug/L")
    assert s.loc[k2, "drop_reason"] == "no_surface_result" and s.loc[k2, "n_results_qc"] == 2


def test_load_results_leaves_the_sidecar_files_out(tmp_path, monkeypatch):
    root = tmp_path / "wqp_discrete"
    (root / "variable=chla").mkdir(parents=True)
    harm = _rows(SPEC[:7])
    harm.to_parquet(root / "variable=chla" / "data.parquet", index=False)
    rej = _rows([("S1", "2020-06-01", CHLA, "ug/L", 0.0, np.nan, None)]).drop(columns=["depth_m"])
    rej.assign(qc_reason="qc_activity").to_parquet(root / "_qc_rejects.parquet", index=False)
    rej.assign(characteristic="Chlorophyll a", unmapped_reason="x").to_parquet(root / "_unmapped.parquet",
                                                                                index=False)
    pd.DataFrame({"site_id": ["S1"], "lat": [1.0], "lon": [2.0]}).to_parquet(root / "_stations.parquet", index=False)
    other = _rows([("S1", "2020-06-01", "Chloride", "mg/L", 5.0, 0.1, None)])
    (root / "variable=chloride").mkdir()
    other.to_parquet(root / "variable=chloride" / "data.parquet", index=False)
    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    keep_before = M.RESULT_KEEP
    res_all, got, rows = WS.load_results()
    assert M.RESULT_KEEP is keep_before                         # the shared module is not touched
    assert len(res_all) == 9 and "depth_m" not in res_all.columns
    assert len(got) == 7 and got["depth_m"].notna().all()
    assert rows == {"wqp_discrete/_qc_rejects.parquet": 1, "wqp_discrete/_unmapped.parquet": 1,
                    "wqp_discrete/variable=chla/data.parquet": 7}
    s = WS.surface_table(got, res_all).set_index("sample_key")
    k1 = _key("S1", "2020-06-01", "chla", CHLA, "ug/L")
    assert s.loc[k1, "n_results_all"] == 4 and s.loc[k1, "n_results"] == 2 and s.loc[k1, "value"] == 15.0


def test_bottom_flag_counts():
    c = WS.bottom_flag_counts(_rows(SPEC))
    assert c == {"chla": {"n_results_flagged_bottom": 2, "n_flagged_bottom_depth_le_1m": 1,
                          "n_flagged_bottom_depth_unknown": 1}}


# ----------------------------------------------------------------------------------------- apply + guards

def _samples(side, extra_keys=(), set_name="s2_wqp_matchups"):
    """A matchup sample table as the builds wrote it: depth-blind values, sample_depth_m all null."""
    s = side.reset_index()
    smp = pd.DataFrame({"sample_key": list(s.sample_key) + list(extra_keys)})
    smp["target"] = [k.split("|")[2] for k in smp.sample_key]
    smp["value"] = list(s["_value_all"]) + [1.0] * len(extra_keys)
    smp["n_results"] = list(s["n_results_all"]) + [1] * len(extra_keys)
    smp["sample_depth_m"] = np.nan
    smp["depth_rule"] = "unspecified"
    smp["point_key"] = "p"
    smp["_set"] = set_name
    return smp


def test_apply_replaces_values_and_drops_samples_without_a_surface_result():
    side = _side()
    smp = _samples(side)
    logs = []
    out = WS.apply(smp, log=logs.append, side=side.reset_index())
    got = out.set_index("sample_key")
    k = lambda site: _key(site, "2020-06-01", "chla", CHLA, "ug/L")  # noqa: E731
    assert k("S2") not in got.index and len(out) == len(smp) - 1
    assert got.loc[k("S3"), "value"] == 5.0 and got.loc[k("S3"), "n_results"] == 2
    assert got.loc[k("S1"), "sample_depth_m"] == pytest.approx(0.2)
    assert smp.set_index("sample_key").loc[k("S3"), "value"] == 6.0          # the input is not modified
    rec = WS.last_applied()
    c = rec["by_table_target"]["s2_wqp_matchups/chla"]
    assert c == {"n": 6, "dropped_no_qc_passed_result": 0, "dropped_no_surface_result": 1,
                 "kept_with_fewer_results": 2}
    assert rec["n_samples"] == len(smp) and rec["n_with_surface_result"] == len(out)
    assert len(logs) == 1 and "value" not in json.dumps(rec["by_table_target"])


def test_apply_refuses_unmatched_keys_instead_of_defaulting_to_unknown_depth():
    side = _side()
    smp = _samples(side, extra_keys=["Z|2020-01-01|chla|Chlorophyll a|ug/L", "Z|2020-01-02|tss|x|mg/L"])
    with pytest.raises(WS.SurfaceRuleError, match=r"2 sample keys are not in the side table.*s2_wqp_matchups"):
        WS.apply(smp, log=lambda *_: None, side=side.reset_index())


def test_apply_drops_samples_made_of_sidecar_rows_only():
    side = _side(extra_all=[("X1", "2020-06-02", CHLA, "ug/L", 55.0, np.nan, None)])
    out = WS.apply(_samples(side), log=lambda *_: None, side=side.reset_index())
    assert not out.sample_key.str.startswith("X1|").any()
    assert WS.last_applied()["by_table_target"]["s2_wqp_matchups/chla"]["dropped_no_qc_passed_result"] == 1


def test_guard_fails_when_no_non_secchi_row_has_a_depth():
    """The defect this module fixes: a table built without reading depth has sample_depth_m null on every row."""
    side = _side()
    built = _samples(side)                                   # as the matchup builds wrote it
    with pytest.raises(WS.SurfaceRuleError, match="sample_depth_m is null for all"):
        WS.check_depth_read(built)
    fixed = WS.apply(built, log=lambda *_: None, side=side.reset_index())
    WS.check_depth_read(fixed)                               # passes once the rule is applied
    # per table: one table without depths fails even when another has them
    both = pd.concat([fixed, built.assign(_set="s2_wqp_matchups_hist")], ignore_index=True)
    with pytest.raises(WS.SurfaceRuleError, match="s2_wqp_matchups_hist"):
        WS.check_depth_read(both)
    # Secchi rows never count: a Secchi-only table passes, Secchi depths do not rescue a table
    sec = built[built.target == "secchi"]
    WS.check_depth_read(sec)
    with pytest.raises(WS.SurfaceRuleError):
        WS.check_depth_read(pd.concat([built[built.target != "secchi"], sec.assign(sample_depth_m=3.0)]))


def test_apply_runs_the_guard_on_its_output():
    side = _side().reset_index()
    side["sample_depth_m"] = np.nan                          # a side table that was built without depth
    with pytest.raises(WS.SurfaceRuleError, match="sample_depth_m is null for all"):
        WS.apply(_samples(_side()), log=lambda *_: None, side=side)


def test_real_tables_need_every_ruling(monkeypatch):
    monkeypatch.setattr(WS, "RULINGS", {"depth_rule": "ruled", "sidecar_exclusion": None})
    with pytest.raises(WS.SurfaceRuleError, match="not ruled yet"):
        WS.apply(_samples(_side()), log=lambda *_: None)
    monkeypatch.setattr(WS, "RULINGS", {"depth_rule": "ruled", "sidecar_exclusion": "ruled"})
    WS.require_rulings()


def test_load_side_refuses_a_stale_or_foreign_side_table(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    with pytest.raises(WS.SurfaceRuleError, match="missing"):
        WS.load_side()
    d = WS.side_dir()
    d.mkdir(parents=True)
    src = tmp_path / "wqp_discrete" / "variable=chla"
    src.mkdir(parents=True)
    _rows(SPEC).to_parquet(src / "data.parquet", index=False)
    side = _side().reset_index().drop(columns=["_value_all"])
    side.to_parquet(d / WS.SIDE_FILE, index=False)
    meta = {"rule": WS.RULE, "sidecars_left_out": list(WS.SIDECARS),
            "side_table_sha256": WS._sha_file(d / WS.SIDE_FILE),
            "input_files": WS.input_fingerprint([src / "data.parquet"])}
    (d / WS.META_FILE).write_text(json.dumps(meta))
    assert len(WS.load_side()) == len(side)
    (d / WS.META_FILE).write_text(json.dumps({**meta, "rule": "another rule"}))
    with pytest.raises(WS.SurfaceRuleError, match="different rule"):
        WS.load_side()
    (d / WS.META_FILE).write_text(json.dumps({**meta, "side_table_sha256": "0" * 64}))
    with pytest.raises(WS.SurfaceRuleError, match="does not match its meta"):
        WS.load_side()
    (d / WS.META_FILE).write_text(json.dumps(meta))
    _rows(SPEC[:3]).to_parquet(src / "data.parquet", index=False)     # the inputs changed after the build
    with pytest.raises(WS.SurfaceRuleError, match="changed since"):
        WS.load_side()


def test_wqp_rows_goes_through_the_rule(monkeypatch):
    """data.wqp_rows must hand its samples to wqp_surface.apply before any value is used."""
    class Called(Exception):
        pass

    seen = {}

    def fake_apply(smp, log=print, side=None):
        seen["cols"] = list(smp.columns)
        seen["n"] = len(smp)
        raise Called

    smp = _samples(_side()).assign(jrc_water_ok=True, unit="ug/L")
    monkeypatch.setattr(D, "_wqp_tables", lambda log: (smp, pd.DataFrame()))
    monkeypatch.setattr(WS, "apply", fake_apply)
    with pytest.raises(Called):
        D.wqp_rows(log=lambda *_: None)
    assert seen["n"] == int(smp.target.isin(D.TARGETS).sum()) and "sample_key" in seen["cols"]
    src = (paths.REPO_ROOT / "sentinel" / "models" / "orbit" / "data.py").read_text()
    assert src.index("WS.apply(smp, log=log)") < src.index("smp.value * fac")


# ----------------------------------------------------------------------------------------- on-disk guards

def _side_on_disk():
    f = WS.side_dir() / WS.SIDE_FILE
    if not f.exists():
        pytest.skip("side table not built")
    return f


def test_built_side_table_has_depths_and_matches_its_meta():
    f = _side_on_disk()
    meta = json.loads((WS.side_dir() / WS.META_FILE).read_text())
    assert meta["side_table_sha256"] == WS._sha_file(f)
    assert meta["rule"] == WS.RULE and meta["sidecars_left_out"] == list(WS.SIDECARS)
    for name, r in meta["reproduces_built_tables"].items():
        assert r["n_key_missing"] == 0 and r["n_value_differs"] == 0 and r["n_results_differs"] == 0, name
    side = pd.read_parquet(f, columns=["target", "surface", "sample_depth_m"])
    WS.check_depth_read(side[side.surface], "side table")
    assert not side.loc[~side.surface, "sample_depth_m"].notna().any()


def test_every_built_matchup_sample_key_is_in_the_side_table():
    """Key coverage 100 %: every sample of a built matchup table maps to the side table (counts only)."""
    f = _side_on_disk()
    keys = pd.Index(pd.read_parquet(f, columns=["sample_key"])["sample_key"])
    n = 0
    for name in D.WQP_MATCHUP_SETS:
        sf = paths.PROCESSED / name / "samples.parquet"
        if not sf.exists():
            continue
        k = pd.read_parquet(sf, columns=["sample_key"])["sample_key"]
        assert int((~k.isin(keys)).sum()) == 0, name
        n += 1
    if not n:
        pytest.skip("no matchup table built")


def test_label_rule_inputs_are_pinned():
    """The side table and every wqp_discrete file it was built from are in the freeze data pins."""
    from sentinel.models.orbit import testlook as TL

    d, pats = TL.DATA_PIN_SPECS["wqp_surface"]
    assert d == WS.SIDE_NAME and set(pats) == {WS.SIDE_FILE, WS.META_FILE}
    d, pats = TL.DATA_PIN_SPECS["wqp_discrete"]
    assert d == "wqp_discrete" and all(f in pats for f in WS.SIDECARS) and "variable=*/data.parquet" in pats
    _side_on_disk()
    meta = json.loads((WS.side_dir() / WS.META_FILE).read_text())
    pinned = {f.relative_to(paths.PROCESSED).as_posix() for f in TL.pin_files(d, pats)}
    assert set(meta["input_files"]) <= pinned
    assert {f.name for f in TL.pin_files(*TL.DATA_PIN_SPECS["wqp_surface"])} == {WS.SIDE_FILE, WS.META_FILE}
