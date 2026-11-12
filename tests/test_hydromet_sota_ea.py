# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EA England build (M6 T-E): the reserved period (>= 2023-10-01) can never be requested or parsed; QC/daily statistics
and the river-network grouping behave as documented. Synthetic inputs only (code paths)."""

import numpy as np
import pandas as pd
import pytest

from sentinel.hydromet_sota import ea as EA
from sentinel.hydromet_sota import ea_geo as EG


def test_no_request_reaches_the_reserved_period():
    months = EA.month_starts()
    assert months[0] == pd.Timestamp("2012-01-01") and months[-1] == pd.Timestamp("2023-09-01")
    url, p = EA.readings_url("temperature", months[-1])
    assert p["max-date"] == "2023-10-01"          # exclusive upper bound
    with pytest.raises(ValueError):
        EA.readings_url("temperature", pd.Timestamp("2023-10-01"))


def test_parser_refuses_reserved_rows():
    ok = (b'"measure","dateTime","date","value","completeness","quality","qcode"\n'
          b'"http://x/measures/A-temp-i-subdaily-C","2023-09-30T23:45:00","2023-09-30","12.1","","Unchecked",""\n')
    assert len(EA.parse_readings(ok)) == 1
    bad = ok + b'"http://x/measures/A-temp-i-subdaily-C","2023-10-01T00:00:00","2023-10-01","12.0","","Unchecked",""\n'
    with pytest.raises(ValueError, match="reserved"):
        EA.parse_readings(bad)


def test_measure_var():
    assert EA.measure_var("E1-temp-i-subdaily-C") == "temp_c"
    assert EA.measure_var("E1-do-i-subdaily-mgL") == "do_mgl"
    assert EA.measure_var("E1-do-i-subdaily-pct") == "do_pct"
    assert EA.measure_var("E1-level-i-900-m") is None


def _series(n_days=3, step_min=15, start="2015-06-01"):
    t = pd.date_range(start, periods=n_days * 24 * 60 // step_min, freq=f"{step_min}min")
    v = 15 + 2 * np.sin(np.arange(len(t)) / 96 * 2 * np.pi)
    return pd.DataFrame({"measure": "A-temp-i-subdaily-C", "dateTime": t, "date": t.normalize(), "value": v,
                         "completeness": "", "quality": "Unchecked", "qcode": ""})


def test_qc_and_daily_validity():
    g = _series(n_days=5)
    g.loc[10, "value"] = 40.0                       # out of range
    g.loc[50, "value"] = g.loc[50, "value"] + 8.0   # spike
    g.loc[200:400, "value"] = 11.0                  # 50 h flat line over days 3-5 (index 192 = day 3)
    g.loc[300, "quality"] = "Suspect"
    kept, info = EA.clean_measure(g, "temp_c")
    assert info["drop_range"] == 1 and info["drop_spike"] >= 1 and info["drop_flatline"] >= 190
    d = EA.daily_stats(kept, "temp_c")
    assert d["valid"].iloc[0] and not d["valid"].iloc[2]    # day 1 complete; day 3 mostly flat-lined
    assert abs(d["step_s"].iloc[0] - 900) < 1


def test_flow_groups_merge_connected_units():
    nd = {1: 2, 2: 3, 3: 0, 10: 11, 11: 0}
    reach = pd.Series({"a": 1, "b": 3, "c": 10, "d": 11})
    pairs = EG.flow_pairs(reach, nd)
    assert ("a", "b") in pairs and ("c", "d") in pairs and ("a", "c") not in pairs
    unit = pd.Series({"a": "U1", "b": "U2", "c": "U3", "d": "U3"})
    g = EG.basin_groups(unit, pairs)
    assert g["a"] == g["b"] and g["c"] == g["d"] and g["a"] != g["c"]


# ---- the England split: one canonical core for the writer and every reader (synthetic splits only) ----
import copy
import importlib.util
import json
from pathlib import Path

from sentinel.hydromet_sota import ea_data as ED
from sentinel.hydromet_sota import ea_split as ES


def _synthetic_meta(n_units=40, per_unit=3):
    """Synthetic catalogue metadata in the form of ``metadata_eligibility()`` plus the NEXT_DOWN graph.

    Unit i has its own river (reaches 100 i + 1 -> 100 i + 2 -> outlet) with ``per_unit`` eligible stations; the river
    of unit 1 continues into the river of unit 0 (the two units merge). Ineligible stations: ``X_flow`` on the river
    of unit 2 (flow-connected), ``X_unit`` inside unit 3 without a reach, ``X_none`` in a unit without any eligible
    station and without a reach."""
    nd, rows = {}, {}
    for i in range(n_units):
        nd[100 * i + 1], nd[100 * i + 2] = 100 * i + 2, 0
        for k in range(per_unit):
            rows[f"S{i:02d}{k}"] = {"hybas_l7": float(2070000000 + 10 * i), "hyriv_id": float(100 * i + 1 + (k % 2)),
                                    "eligible": True}
    nd[102] = 2                                                  # unit 1's river flows into unit 0's
    rows["X_flow"] = {"hybas_l7": 2079999990.0, "hyriv_id": 201.0, "eligible": False}
    rows["X_unit"] = {"hybas_l7": 2070000030.0, "hyriv_id": np.nan, "eligible": False}
    rows["X_none"] = {"hybas_l7": 2079999990.0, "hyriv_id": np.nan, "eligible": False}
    el = pd.DataFrame.from_dict(rows, orient="index")
    for c in ("test_capable", "val_capable", "has_temp_measure"):
        el[c] = el["eligible"]
    el["has_do_measure"] = el["eligible"] & (np.arange(len(el)) % 2 == 0)
    el["near_discharge"] = False
    return el, nd


def _synthetic_split(**kw):
    el, nd = _synthetic_meta()
    e = el[el["eligible"]]
    grp = EG.basin_groups(e["hybas_l7"].astype("int64").astype(str), EG.flow_pairs(e["hyriv_id"], nd))
    elig, partners = set(e.index), {}
    for a, b in EG.flow_pairs(el["hyriv_id"], nd):
        if (a in elig) != (b in elig):
            s, o = (b, a) if a in elig else (a, b)
            partners.setdefault(s, []).append(o)
    core = ES.build_core(all_stations=list(el.index), station_group=grp.to_dict(),
                         unit_of={s: str(int(u)) for s, u in el["hybas_l7"].items()}, flow_partners=partners,
                         eligibility={"max_snap_m": 1000.0, "min_days": 60}, inputs_sha256={"layer": "0" * 64}, **kw)
    return ES.seal(core, note="synthetic", counts_metadata_only={"n": len(el)})


def test_split_group_id_is_the_smallest_merged_unit():
    b = _synthetic_split()
    assert b["station_group"]["S010"] == b["station_group"]["S000"] == "2070000000"     # units 0 and 1 merged
    assert b["group_units"]["2070000000"] == ["2070000000", "2070000010"]
    assert len(b["role_of_group"]) == 39 and set(b["role_of_group"].values()) == {"train", "val", "test"}


def test_split_round_trip_and_every_core_key_is_hashed(tmp_path):
    b = _synthetic_split()
    assert b["sha256"] == ES.core_sha256(b) and b["core_keys"] == list(ES.CORE_KEYS)
    for k in ("salt", "share_test", "share_val", "big_group_frac", "rule", "version", "eligibility", "inputs_sha256",
              "role_of_group", "group_units", "station_group", "station_role", "stations", "closed_stations"):
        assert k in ES.CORE_KEYS
    f = tmp_path / "split.json"
    f.write_text(json.dumps(b, indent=1))
    assert ES.load(f)["sha256"] == b["sha256"]                    # file round trip, ruled parameters
    b2 = copy.deepcopy(b)
    b2["note"] = "non-core information may differ"
    assert ES.verify(b2)["sha256"] == b["sha256"]
    tamper = {"salt": "other:", "share_test": 0.4, "share_val": 0.2, "big_group_frac": 0.5, "rule": "x", "version": "0",
              "eligibility": {"max_snap_m": 5000.0, "min_days": 60}, "inputs_sha256": {"layer": "1" * 64},
              "closed_stations": []}
    for k, v in tamper.items():
        t = copy.deepcopy(b)
        t[k] = v
        assert ES.core_sha256(t) != b["sha256"], k                # the key is part of the hash
        with pytest.raises(ES.SplitError):
            ES.verify(t)
    for k in ("role_of_group", "group_units", "station_group", "station_role", "stations"):
        t = copy.deepcopy(b)
        first = sorted(t[k])[0]
        t[k][first] = {"role_of_group": "val" if t[k][first] != "val" else "train", "group_units": ["1"],
                       "station_group": "1", "station_role": "val" if t[k][first] != "val" else "train",
                       "stations": {"eligible": False, "status": "reserved", "basis": "none", "groups": []}}[k]
        with pytest.raises(ES.SplitError):
            ES.verify(t)
    t = copy.deepcopy(b)
    del t["salt"]
    with pytest.raises(ES.SplitError):
        ES.verify(t)
    t = copy.deepcopy(b)
    t["core_keys"] = ["rule", "role_of_group", "station_group", "station_role"]       # the old 4-key reader core
    with pytest.raises(ES.SplitError):
        ES.verify(t)


def test_resealed_split_with_other_parameters_is_still_refused():
    """A consistent, correctly hashed split drawn with another S / salt is not the ruled split."""
    for kw in ({"share_test": 0.40}, {"salt": "other:"}, {"share_val": 0.10}, {"big_group_frac": 0.5}):
        b = _synthetic_split(**kw)
        assert ES.verify(b, ruled_parameters=False)["sha256"] == ES.core_sha256(b)
        with pytest.raises(ES.SplitError, match="ruled"):
            ES.verify(b)
    b = _synthetic_split()
    b["role_of_group"] = {g: "train" for g in b["role_of_group"]}      # resealed but not the draw of the rule
    b["station_role"] = {s: "train" for s in b["station_role"]}
    for v in b["stations"].values():
        v["status"] = "train" if v["groups"] else v["status"]
    b["closed_stations"] = sorted(s for s, v in b["stations"].items() if v["status"] in ES.CLOSED)
    b["sha256"] = ES.core_sha256(b)
    with pytest.raises(ES.SplitError, match="draw"):
        ES.verify(b)


def test_every_station_has_a_status():
    b = _synthetic_split()
    st, rg = b["stations"], b["role_of_group"]
    assert len(st) == 40 * 3 + 3
    assert st["X_flow"] == {"eligible": False, "status": rg["2070000020"], "basis": "flow_connected",
                            "groups": ["2070000020"]}
    assert st["X_unit"] == {"eligible": False, "status": rg["2070000030"], "basis": "l7_unit", "groups": ["2070000030"]}
    assert st["X_none"] == {"eligible": False, "status": "reserved", "basis": "none", "groups": []}
    assert "X_none" in b["closed_stations"] and not any(s.startswith("X_") for s in b["station_role"])
    assert set(ES.m6_stations(b, ("train", "val"))).isdisjoint(b["closed_stations"])
    # several flow-connected roles: the most closed one (test before val before train)
    t = ES.station_table(["e1", "e2", "x"], {"e1": "g1", "e2": "g2"}, {"g1": "train", "g2": "test"},
                         {"e1": "1", "e2": "2", "x": "1"}, {"x": ["e1", "e2"]})
    assert t["x"]["status"] == "test" and t["x"]["groups"] == ["g1", "g2"]


def test_min_groups_thresholds_are_mechanical():
    assert ES.min_groups({i: 4 for i in range(15)}) == {"n_stations": 60, "G": 15, "G_eff": 15.0, "status": "claim"}
    assert ES.min_groups({i: 4 for i in range(14)})["status"] == "descriptive"            # G < 15
    lop = {0: 60, **{i: 1 for i in range(1, 20)}}                                           # G = 20, G_eff = 1.7
    r = ES.min_groups(lop)
    assert r["G"] == 20 and r["G_eff"] < ES.G_EFF_MIN and r["status"] == "descriptive"
    assert ES.min_groups({"a": 0, "b": 3})["G"] == 1 and ES.min_groups({})["status"] == "descriptive"
    assert (ES.G_MIN, ES.G_EFF_MIN, ES.SHARE_TEST) == (15, 8.0, 0.50) and len(ES.ENDPOINTS) == 6


def _build_script():
    spec = importlib.util.spec_from_file_location(
        "_ea_build", Path(__file__).resolve().parents[1] / "scripts" / "data" / "sota_hydromet_ea_build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_writer_and_reader_share_the_core(tmp_path, monkeypatch):
    """The split writer of the build script and ``ea_data`` (the reader) accept exactly the same file."""
    B = _build_script()
    el, nd = _synthetic_meta()
    f, f1 = tmp_path / "hydromet_ea_split.json", tmp_path / "sonde_ea_split.json"
    monkeypatch.setattr(B, "metadata_eligibility", lambda: el)
    monkeypatch.setattr(B, "_next_down", lambda: nd)
    monkeypatch.setattr(B, "layer_pins", lambda: {"synthetic": "0" * 64})
    monkeypatch.setattr(B, "SPLIT_FILE", f)
    monkeypatch.setattr(ED, "SPLIT_FILE", f)
    monkeypatch.setattr(ED, "M1_SPLIT_FILE", f1)
    with pytest.raises(SystemExit):
        B.step_split(None, False)                      # not without the lead's flag
    with pytest.raises(SystemExit):
        B.step_split(0.40, True)                       # only the ruled S
    assert not f.exists()
    B.step_split(None, True)
    body = json.loads(f.read_text())
    assert body["sha256"] == ES.core_sha256(body) and body["share_test"] == 0.50
    assert ED.load_split()["sha256"] == body["sha256"]              # reader: same function, same hash
    B.step_split(None, True)                                         # idempotent
    cells = body["min_groups_metadata"]["cells"]
    assert len(cells) == 3 and all(set(c) == set(ES.ENDPOINTS) for c in cells.values())
    # Q8: no England row before M1's file exists and verifies
    with pytest.raises(ED.EAProtectedError, match="sonde_ea_split.json"):
        ED.load_ea()
    open_groups = sorted(g for g, r in body["role_of_group"].items() if r != "test")
    f1.write_text(json.dumps(ES.seal_m1(body, open_groups[:2], salt="M1-EA:", share=0.2)))
    m6, m1 = ED.load_splits()
    assert m1 == open_groups[:2] and m6["sha256"] == body["sha256"]
    test_group = next(g for g, r in body["role_of_group"].items() if r == "test")
    for bad in (ES.seal_m1(body, [test_group]), ES.seal_m1(body, open_groups),             # test-role group; > 20 %
                ES.seal_m1({"sha256": "f" * 64}, []), {**ES.seal_m1(body, []), "m1_test_groups": open_groups[:1]}):
        f1.write_text(json.dumps(bad))
        with pytest.raises(ED.EAProtectedError):
            ED.load_splits()
    f1.write_text(json.dumps(ES.seal_m1(body, [])))                  # the empty draw is a valid file
    assert ED.load_splits()[1] == []
    # a tampered frozen file is refused by the reader and by the writer (never loosened)
    f.chmod(0o644)
    t = dict(body, share_test=0.40)
    f.write_text(json.dumps(t))
    with pytest.raises(ED.EAProtectedError):
        ED.load_split()
    with pytest.raises(ES.SplitError):
        B.step_split(None, True)
    for roles in (("test",), ("reserved",), ("train", "test")):
        with pytest.raises(ED.EAProtectedError):
            ED.load_ea(roles=roles)
