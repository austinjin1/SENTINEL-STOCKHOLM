# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""C-1 / R-1 of the arch-v2 lead rulings: the M7 keys list and its match rule (sentinel/models/orbit_v2/m7keys.py).

Synthetic lists only, except one test that loads the ruled list itself (key columns; the file holds no label and no
prediction) and checks its hash and row count. Nothing else of M7 is read."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from sentinel.models.orbit_v2 import m7keys as M7


def _list(tmp_path, rows, columns=M7.COLUMNS, name="keys.csv"):
    base = {"m2_waterbody": "L1", "s2_sensing_date": "2022-07-10", "source": "wqp", "site_id": "S1",
            "sample_date": "2022-07-09", "s2_chip_id": "c", "lite_role": "test", "m2_buffer_excluded": "False",
            "in_m2_test_s_window": "True"}
    df = pd.DataFrame([{**base, **r} for r in rows])[list(columns)]
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _rows(recs):
    return pd.DataFrame(recs, columns=list(M7.ROW_COLS))


def test_the_ruled_constants():
    assert M7.LIST_SHA256 == "4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699"
    assert M7.N_ROWS == 110 and M7.SENSING_TOL_D == 1
    assert M7.LIST_PATH.name == "c2_teacher_scored_m2_test_lake_keys.csv"
    assert M7.LIST_PATH.parts[-3:-1] == ("lite", "counts_arch_v2")
    assert M7.record() == {"path": str(M7.LIST_PATH), "sha256": M7.LIST_SHA256, "n_list_rows": 110, "rule": M7.RULE}


def test_loader_refuses_another_file(tmp_path):
    p, sha = _list(tmp_path, [{}, {"site_id": "S2"}])
    lst = M7.load_list(p, sha256=sha, n_rows=2)
    assert len(lst) == 2 and {"_s2_day", "_sample_day"} <= set(lst.columns)
    with pytest.raises(M7.M7ListError, match="differs from the ruled"):
        M7.load_list(p)                                        # the default is the ruled sha256
    with pytest.raises(M7.M7ListError, match="differs from the ruled"):
        M7.load_list(p, sha256=sha[:-1] + ("0" if sha[-1] != "0" else "1"), n_rows=2)
    with pytest.raises(M7.M7ListError, match="2 data rows, the ruling says 110"):
        M7.load_list(p, sha256=sha)
    with pytest.raises(M7.M7ListError, match="does not exist"):
        M7.load_list(tmp_path / "absent.csv")
    p2, sha2 = _list(tmp_path, [{}], columns=M7.COLUMNS[:-1], name="short.csv")
    with pytest.raises(M7.M7ListError, match="not the ruled key columns"):
        M7.load_list(p2, sha256=sha2, n_rows=1)
    p3, sha3 = _list(tmp_path, [{}], columns=(*M7.COLUMNS[1:], M7.COLUMNS[0]), name="order.csv")
    with pytest.raises(M7.M7ListError, match="not the ruled key columns"):
        M7.load_list(p3, sha256=sha3, n_rows=1)


def test_match_rule_i_and_ii(tmp_path):
    p, sha = _list(tmp_path, [{"m2_waterbody": "L1", "s2_sensing_date": "2022-07-10", "site_id": "S1",
                               "sample_date": "2022-07-09"},
                              {"m2_waterbody": "L2", "s2_sensing_date": "2023-01-01", "site_id": "S7",
                               "sample_date": "2023-01-03"}])
    lst = M7.load_list(p, sha256=sha, n_rows=2)
    rows = _rows([
        ("L1", "2022-07-10", "X", "2022-07-01"),      # (i) same water body, same sensing date
        ("L1", "2022-07-11", "X", "2022-07-01"),      # (i) +1 day
        ("L1", "2022-07-09", "X", "2022-07-01"),      # (i) -1 day
        ("L1", "2022-07-12", "X", "2022-07-01"),      # 2 days: no
        ("L1", "2022-07-08", "X", "2022-07-01"),      # 2 days: no
        ("L2", "2022-07-10", "X", "2022-07-01"),      # another water body's date: no
        ("L9", "2020-01-01", "S1", "2022-07-09"),     # (ii) site and sample date
        ("L9", "2020-01-01", "S1", "2022-07-10"),     # (ii) has no tolerance: no
        ("L9", "2020-01-01", "S7", "2022-07-09"),     # site of one list row, date of another: no
        ("L1", "2022-07-10", "S1", "2022-07-09"),     # both
        ("L2", "2022-12-31", "S7", "2023-01-03"),     # both (year boundary)
    ])
    m, counts = M7.match(rows, lst)
    assert m.tolist() == [True, True, True, False, False, False, True, False, False, True, True]
    assert counts == {"n_rows": 11, "n_matched": 6, "n_rule_i_waterbody_sensing_date": 5,
                      "n_rule_ii_site_sample_date": 3, "n_both": 2}


def test_missing_keys_never_match_and_dates_are_whole_days(tmp_path):
    p, sha = _list(tmp_path, [{"m2_waterbody": "", "site_id": "", "s2_sensing_date": "2022-07-10",
                               "sample_date": "2022-07-09"},
                              {"m2_waterbody": "L1", "site_id": "S1", "s2_sensing_date": "", "sample_date": ""},
                              {"m2_waterbody": "L5", "site_id": "S5", "s2_sensing_date": "2022-08-01",
                               "sample_date": "2022-08-01"}])
    lst = M7.load_list(p, sha256=sha, n_rows=3)
    rows = pd.DataFrame({
        "waterbody": ["", None, "nan", "L1", "L5", "L5", np.nan],
        "sensing_date": ["2022-07-10", "2022-07-10", "2022-07-10", None, pd.Timestamp("2022-08-02 23:59", tz="UTC"),
                         "2022-08-03T00:00:01", "2022-08-01"],
        "site_id": ["", None, "nan", "S1", "none", "none", "S5"],
        "date": ["2022-07-09", "2022-07-09", "2022-07-09", None, None, None, pd.NaT]})
    m, counts = M7.match(rows, lst)
    assert m.tolist() == [False, False, False, False, True, False, False] and counts["n_matched"] == 1
    with pytest.raises(M7.M7ListError, match="lack the key columns"):
        M7.match(rows.drop(columns="site_id"), lst)
    with pytest.raises(M7.M7ListError, match="1 date entries cannot be read"):
        M7.match(rows.assign(sensing_date=["2022-07-10"] * 6 + ["10/07/2022"]), lst)
    np.testing.assert_array_equal(M7._days([np.datetime64("1970-01-03"), "1970-01-02", pd.Timestamp("1970-01-02 23:00"),
                                            None, "", pd.NaT]), [2.0, 1.0, 1.0, np.nan, np.nan, np.nan])
    with pytest.raises(M7.M7ListError, match="did not come from load_list"):
        M7.match(rows, lst.drop(columns="_s2_day"))


def test_the_ruled_list_itself():
    """Key columns of the ruled file only; skipped when the file is not on this machine."""
    if not M7.LIST_PATH.is_file():
        pytest.skip(f"{M7.LIST_PATH} not present")
    assert M7.file_sha256(M7.LIST_PATH) == M7.LIST_SHA256
    lst = M7.load_list()
    assert len(lst) == 110 and tuple(lst.columns[:9]) == M7.COLUMNS
    assert lst["_s2_day"].notna().all()
    assert int((lst["in_m2_test_s_window"] == "True").sum()) == 106
    assert int((lst["m2_buffer_excluded"] == "True").sum()) == 4
    # the list matches itself under both clauses
    m, counts = M7.match(pd.DataFrame({"waterbody": lst["m2_waterbody"], "sensing_date": lst["s2_sensing_date"],
                                       "site_id": lst["site_id"], "date": lst["sample_date"]}), lst)
    assert m.all() and counts["n_rule_i_waterbody_sensing_date"] == 110
