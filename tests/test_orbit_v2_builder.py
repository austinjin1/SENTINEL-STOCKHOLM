# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Key-level rules of the dev_v5m builder (scripts/data/orbitv2_build_dev_v5m.py) under the arch-v2 lead rulings:
C-2 (no switches), C-1 (M7 keys list first), R-8 / R-21 (join sites from keys), R-9 (AquaCorr unit check), R-25.

Synthetic key tables only; one test reads AquaCorr's metadata (schema names and the metadata json, no row). No value
column, no chip and nothing under the runs root is read or written."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.models.orbit_v2 import devm as DM
from sentinel.models.orbit_v2 import m7keys as M7
from sentinel.models.orbit_v2 import targets as T

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "orbitv2_build_dev_v5m.py"
M = 1.0 / 111_195.0                      # one metre of latitude, in degrees


@pytest.fixture(scope="module")
def B():
    spec = importlib.util.spec_from_file_location("orbitv2_build_dev_v5m", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.log = lambda *a: None
    return mod


# ------------------------------------------------------------------------------------ C-2, C-1, R-25
def test_the_two_switches_do_not_exist(B, capsys):
    src = SCRIPT.read_text()
    for gone in ("--context-same-role", "--join-same-waterbody", "context_same_role", "join_same_waterbody",
                 "same_role=", "same_waterbody="):
        assert gone not in src
    for argv in (["--mode", "keys", "--context-same-role"], ["--mode", "keys", "--join-same-waterbody"],
                 ["--mode", "keys", "--stores", "naip_teacher"]):
        with pytest.raises(SystemExit) as e:
            B.main(argv)
        assert e.value.code == 2                         # argparse: unrecognised argument
    capsys.readouterr()
    assert B.MIN_FREE_GB_AFTER == 60.0 and B.MAX_ARTEFACT_GB == 10.0
    assert B.DM.CONTEXT_STORES == ("aquacorr", "wqp")


def test_builder_refuses_before_any_read_when_the_m7_list_differs(B, tmp_path, monkeypatch):
    p = tmp_path / "keys.csv"
    p.write_text(",".join(M7.COLUMNS) + "\n")

    def never(*a, **k):
        raise AssertionError("the builder read something before checking the M7 keys list")

    monkeypatch.setattr(M7, "LIST_PATH", p)
    for name in ("load_dev", "_index"):
        monkeypatch.setattr(B.D, name, never)
    monkeypatch.setattr(B, "aquacorr_protected_keys", never)
    monkeypatch.setattr(B, "added_rows", never)
    for mode in ("keys", "build"):
        with pytest.raises(M7.M7ListError, match="differs from the ruled"):
            B.main(["--mode", mode, "--out", str(tmp_path / "out")])
    monkeypatch.setattr(M7, "LIST_PATH", tmp_path / "absent.csv")
    with pytest.raises(M7.M7ListError, match="does not exist"):
        B.main(["--mode", "keys"])
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------ C-1 counts
def _m7_list(tmp_path, rows):
    base = {"m2_waterbody": "L1", "s2_sensing_date": "2022-07-10", "source": "wqp", "site_id": "S1",
            "sample_date": "2022-07-09", "s2_chip_id": "c", "lite_role": "test", "m2_buffer_excluded": "False",
            "in_m2_test_s_window": "True"}
    p = tmp_path / "keys.csv"
    pd.DataFrame([{**base, **r} for r in rows])[list(M7.COLUMNS)].to_csv(p, index=False)
    return M7.load_list(p, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), n_rows=len(rows))


def test_c1_key_counts_are_counts_of_test_s_key_rows(B, tmp_path):
    lst = _m7_list(tmp_path, [{}, {"m2_waterbody": "L2", "s2_sensing_date": "2021-05-05", "site_id": "Q",
                                   "sample_date": "2021-05-05"}])
    rec = [  # source, waterbody, in_lake, role, buffer, tblock, sensing time, site, sample date, block
        ("wqp", "L1", True, "test", False, "val", "2022-07-11T10:00:00Z", "S9", "2022-07-11", "b1"),      # rule (i)
        ("wqp", "L1", True, "test", False, "val", "2022-07-20T10:00:00Z", "S1", "2022-07-09", "b1"),      # rule (ii)
        ("wqp", "L1", True, "test", False, "val", "2022-08-20T10:00:00Z", "S1", "2022-08-20", "b1"),      # stays
        ("aquacorr", "L2", True, "test", False, "train", "2021-05-04T10:00:00Z", "", "2021-05-04", "b2"),  # rule (i)
        ("aquacorr", "L3", True, "test", False, "train", "2021-05-04T10:00:00Z", "", "2021-05-04", "b2"),  # stays
        ("wqp", "L1", True, "test", True, "val", "2022-07-10T10:00:00Z", "S1", "2022-07-09", "b1"),       # buffer
        ("wqp", "L1", True, "test", False, "test", "2024-07-10T10:00:00Z", "S1", "2024-07-09", "b1"),     # test years
        ("wqp", "L1", True, "train", False, "test", "2024-07-10T10:00:00Z", "S1", "2024-07-09", "b1"),    # not test role
        ("wqp", "Pb|c", False, "test", False, "val", "2022-07-10T10:00:00Z", "S1", "2022-07-09", "Nb"),   # non-lake
    ]
    prot = pd.DataFrame(rec, columns=["source", "waterbody", "in_lake", "role", "buffer_excluded", "tblock",
                                      "sensing_time_utc", "site_id", "date", "block"])
    prot["date"] = pd.to_datetime(prot["date"])
    c = B.c1_key_counts(prot, lst)
    assert c["before"]["all"] == {"n_key_rows": 5, "n_site_dates": 5, "n_waterbodies": 3, "n_role_blocks": 2}
    assert c["before"]["wqp"]["n_key_rows"] == 3 and c["before"]["aquacorr"]["n_key_rows"] == 2
    assert c["matched_and_removed"] == {"n_rows": 5, "n_matched": 3, "n_rule_i_waterbody_sensing_date": 2,
                                        "n_rule_ii_site_sample_date": 1, "n_both": 0}
    assert c["removed_by_source"] == {"aquacorr": 1, "wqp": 2}
    assert c["after"]["all"] == {"n_key_rows": 2, "n_site_dates": 2, "n_waterbodies": 2, "n_role_blocks": 2}
    assert c["sha256"] == M7.LIST_SHA256 and c["n_list_rows"] == 110

    def leaves(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k not in ("path", "sha256", "rule", "definition"):
                    yield from leaves(v)
        else:
            yield x
    assert all(isinstance(v, int) for v in leaves(c))                       # counts only


# ------------------------------------------------------------------------------------ R-8 / R-21 join sites
TRUTH = {  # station: (waterbody, role, buffer_excluded)
    "W-train": ("L1", "train", False), "W-test": ("L1", "test", False), "W-buf": ("L1", "train", True),
    "W-val": ("L4", "val", False), "A-near": ("L2", "train", False), "A-otherlake": ("L3", "train", False),
    "A-val": ("L2", "val", False), "A-test": ("L5", "test", False), "A-tie-b": ("L6", "val", False),
    "A-tie-a": ("L6", "val", False),
}
ST_LAT = {"W-train": 10.0, "W-test": 11.0, "W-buf": 12.0, "W-val": 13.0, "A-near": 20.0 + 60 * M,
          "A-otherlake": 20.0 + 20 * M, "A-val": 20.0 - 30 * M, "A-test": 30.0 + 10 * M, "A-tie-b": 40.0 + 50 * M,
          "A-tie-a": 40.0 - 50 * M}


def _stations():
    ids = list(TRUTH)
    return pd.DataFrame({"site_id": ids, "lat": [ST_LAT[i] for i in ids], "lon": 5.0, "lat_std": np.nan,
                         "lon_std": np.nan})


def _fake_site_roles(stations, log=print):
    assert not set(stations.columns) & {"value", "detection_limit"}         # keys only
    st = stations.drop_duplicates("site_id").reset_index(drop=True)
    t = [TRUTH[i] for i in st.site_id]
    return pd.DataFrame({"site_id": st.site_id.astype(str), "lon": st.lon.to_numpy(float),
                         "lat": st.lat.to_numpy(float), "waterbody": [x[0] for x in t], "in_lake": True,
                         "block": "b", "role": [x[1] for x in t], "buffer_excluded": [x[2] for x in t]})


def test_join_sites_decides_roles_from_station_keys(B, monkeypatch):
    monkeypatch.setattr(B.T, "site_roles", _fake_site_roles)
    rec = [  # source, site_id, lon, lat, waterbody, role, buffer_excluded -> expected site
        ("wqp", "W-train", 5.0, 10.0, "L1", "train", False, "W-train"),
        ("wqp", "W-test", 5.0, 11.0, "L1", "train", False, None),          # the station's own role is test
        ("wqp", "W-buf", 5.0, 12.0, "L1", "train", False, None),           # the station is buffer-excluded
        ("wqp", "W-val", 5.0, 13.0, "L4", "train", False, None),           # station role differs from the row's
        ("wqp", "W-train", 5.0, 10.0, "L8", "train", False, None),         # station water body differs from the row's
        ("wqp", "W-gone", 5.0, 14.0, "L1", "train", False, None),          # station not in the station table
        ("wqp", "W-train", 5.0, 10.0, "L1", "train", True, None),          # buffer-excluded row
        ("wqp", "W-train", 5.0, 10.0, "L1", "test", False, None),          # not a development row
        ("aquacorr", "", 5.0, 20.0, "L2", "train", False, "A-near"),       # 20 m: other lake; 30 m: other role
        ("aquacorr", "", 5.0, 20.0, "L2", "val", False, "A-val"),
        ("aquacorr", "", 5.0, 20.0, "L9", "train", False, None),           # no station of the row's water body
        ("aquacorr", "", 5.0, 30.0, "L5", "train", False, None),           # the only station has a test role
        ("aquacorr", "", 5.0, 40.0, "L6", "val", False, "A-tie-a"),        # tie at 50 m: lower station id
        ("aquacorr", "", 5.0, 50.0, "L2", "train", False, None),           # nothing within 100 m
        ("aquacorr", "", 5.0, 20.0, "L2", "train", True, None),            # buffer-excluded row
    ]
    rows = pd.DataFrame([r[:7] for r in rec], columns=["source", "site_id", "lon", "lat", "waterbody", "role",
                                                       "buffer_excluded"])
    site, sites, counts = B.join_sites(rows, _stations())
    assert site.tolist() == [r[7] for r in rec]
    assert counts["wqp_rows_development_non_buffer"] == 6 and counts["wqp_rows_with_site"] == 1
    assert counts["wqp_rows_station_role_or_waterbody_differs_unknown_or_buffer_no_site"] == 5
    assert counts["aquacorr_rows_with_station"] == 3 and counts["rule"] == T.RULE_R8
    assert sorted(zip(sites.site_id, sites.role)) == [("A-near", "train"), ("A-tie-a", "val"), ("A-val", "val"),
                                                      ("W-train", "train")]
    assert T.check_dev_sites(sites) == ["A-near", "A-tie-a", "A-val", "W-train"]      # readable: development only
    assert not sites.buffer_excluded.astype(bool).any()


def test_join_targets_reads_nothing_in_keys_mode_without_sites(B):
    rows = pd.DataFrame({"date": pd.to_datetime(["2020-06-01", "2020-06-02"])})
    new, counts = B.join_targets(rows, np.array([None, None], dtype=object),
                                 pd.DataFrame(columns=["site_id", "role", "buffer_excluded"]), "keys")
    assert len(new) == 2 and all(new[t].isna().all() and not new[f"{t}_cens"].any() for t in T.SPECS)


# ------------------------------------------------------------------------------------ R-21 ids
def test_protected_ids_are_dropped_or_counted_before_anything_is_written(B):
    rows = pd.DataFrame({"row_id": ["b0", "b1", "a0", "a1", "a2"], "chip_store": ["wqp", "wqp", "wqp", "aquacorr", "wqp"],
                         "chip_id": ["k0", "k1", "k2", "k1", "k4"], "added": [False, False, True, True, True]})
    cand = pd.DataFrame({"chip_id": ["wqp:c0", "wqp:c1", "aquacorr:c2", "wqp:k1"],
                         "protected": [False, True, False, False]})
    r2, c2, counts = B.drop_protected_ids(rows, cand, ["a2", "zz"], ["wqp:c1", "wqp:k1", "wqp:k2", "naip_teacher:k0"])
    assert r2.row_id.tolist() == ["b0", "b1", "a1"]            # a0: protected chip; a2: protected row id
    assert c2.chip_id.tolist() == ["wqp:c0", "aquacorr:c2"]
    assert counts == {"n_base_rows_on_a_protected_id": 1, "n_added_rows_on_a_protected_id_dropped": 2,
                      "n_context_candidates_protected_dropped": 2, "of_which_not_protected_by_their_own_keys": 1}
    cmeta = pd.DataFrame({"chip_id": ["wqp:c0", "aquacorr:c2"]})
    B.check_no_protected_ids(r2[r2.row_id != "b1"], cmeta, ["a2"], ["wqp:c1", "wqp:k1"])
    for rr, cm, msg in ((r2, cmeta, "0 row ids, 1 row chips, 0 context chips"),
                        (rows[rows.row_id == "a2"], cmeta, "1 row ids, 0 row chips, 0 context chips"),
                        (r2[r2.row_id == "b0"], cmeta.assign(chip_id=["wqp:c1", "x"]), "0 row chips, 1 context chips")):
        with pytest.raises(DM.DevmError, match=msg):
            B.check_no_protected_ids(rr, cm, ["a2"], ["wqp:c1", "wqp:k1"])


def test_added_rows_never_ask_a_matchup_table_for_a_value_column(B):
    names = ["sample_key", "point_key", "date", "target", "value", "request_key", "win_same_day", "chip_ok", "aot_c3",
             "detection_limit", "scl_c3_n_water_chip", "unit"]
    want = {*B.MATCHUP_KEYS, "aot_c3", "scl_c3_n_water_chip", "absent"}
    assert B.matchup_columns(names, want) == ["point_key", "date", "request_key", "win_same_day", "chip_ok", "aot_c3",
                                              "scl_c3_n_water_chip"]
    for v in B.VALUE_LIKE:
        with pytest.raises(DM.DevmError, match="value columns"):
            B.matchup_columns(names, {*want, v})
    with pytest.raises(DM.DevmError, match="lacks the key columns"):
        B.matchup_columns([n for n in names if n != "chip_ok"], want)
    assert not set(B.SAMPLE_KEYS) & set(B.VALUE_LIKE) and not set(B.PROT_KEY_COLS) & set(B.VALUE_LIKE)
    assert not set(B.D.IMG_COLS) & set(B.VALUE_LIKE)


def test_row_ids_follow_the_v1_formula(B):
    want = hashlib.sha1(b"wqp|s17|c9").hexdigest()[:16]
    assert B._row_ids(["wqp"], ["s17"], ["c9"]) == [want]


# ------------------------------------------------------------------------------------ R-9
def test_aquacorr_states_no_unit_for_doc_or_acdom(B):
    """R-9 on the metadata only (schema names and the metadata json; no row is read)."""
    raw = B.paths.RAW / "aquacorr_global"
    if not (raw / "aquacorr_matchups.parquet").exists() or not (raw / "aquacorr_metadata.json").exists():
        pytest.skip("AquaCorr raw files not present")
    chk = B.aquacorr_unit_check()
    assert set(B.AQUACORR_UNIT_TARGETS) == {"doc", "acdom440"}
    for t in B.AQUACORR_UNIT_TARGETS:
        assert chk[t]["unit_stated"] is False and chk[t]["unit_column_present"] is False
        assert chk[t]["metadata_unit_keys"] == []
    assert "used neither for fitting nor for scoring" in chk["outcome"]
