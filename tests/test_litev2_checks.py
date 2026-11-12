# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Row checks of the M7 v2 round (M7_verify ruling Q5 (i), (ii); lead order of 2026-11-10 20:20 EST, A3 and A6).

Synthetic frames, plus two checks on the keys of the real tables: the v2 training rows (training roles only) against
the chip ids, water-body ids, positions and HydroLAKES ids of the validation and test roles. No label column of a
validation or test role is read, nothing is fitted, and only counts are compared."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models.litev2 import checks as C
from sentinel.models.litev2 import labels2b as L
from sentinel.models.litev2 import p2_index as PI
from sentinel.models.litev2 import stamp


def _frame(rows):
    return pd.DataFrame(rows, columns=["uid", "track", "role", "water_body", "lat", "lon", "hylak_id"])


def _train():
    return _frame([("a1", "national", "train", "w1", 40.00, -90.00, 11),
                   ("a2", "national", "train", "w1", 40.01, -90.00, 11),
                   ("a3", "national", "train", "w2", 41.00, -91.00, np.nan),
                   ("p2:b1", "p2", "train_p2", "q6", 45.00, -100.00, -1),
                   ("p2:b2", "p2", "train_p2", "q7", 46.00, -101.00, 31)])


def _keys():
    return _frame([("v1", "national", "val", "w5", 42.00, -92.00, 51),
                   ("v2", "national", "val_n", "w6", 43.00, -93.00, np.nan),
                   ("t1", "national", "test", "w7", 44.00, -94.00, 71),
                   ("t2", "national", "test_time", "w1", 40.00, -90.00, 11),       # a later date of a training lake
                   ("p2:v1", "p2", "val_p2", "q8", 47.00, -102.00, 81),
                   ("p2:t1", "p2", "test_p2", "q9", 48.00, -103.00, np.nan)])


# ------------------------------------------------------------------------------------------------ Q5 (i)
def test_refit_table_holds_no_validation_or_test_chip_or_water_body():
    o = C.assert_refit_clean(_train(), _keys())
    assert o["n_rows"] == 5 and o["roles_not_train"] == [] and o["n_chip"] == 0 and o["n_water_body"] == 0
    assert o["n_cross_track_near"] == 0 and o["n_cross_track_hylak"] == 0
    assert o["n_water_body_time"] == 2 and o["n_water_bodies_time"] == 1     # counted, reported, not a failure

    def broken(tr=None, keys=None, **want):
        got = C.refit_overlap(_train() if tr is None else tr, _keys() if keys is None else keys)
        assert {k: got[k] for k in want} == want
        with pytest.raises(C.CheckError):
            C.assert_refit_clean(_train() if tr is None else tr, _keys() if keys is None else keys)

    for role, uid, wb in (("val", "v1", "w5"), ("val_n", "v2", "w6"), ("test", "t1", "w7")):
        t = pd.concat([_train(), _frame([(uid, "national", role, wb, 42.0, -92.0, np.nan)])], ignore_index=True)
        broken(tr=t, roles_not_train=[role], n_chip=1, n_water_body=1)       # an evaluation row in the table
        t = pd.concat([_train(), _frame([("a9", "national", "train", wb, 30.0, -80.0, np.nan)])], ignore_index=True)
        broken(tr=t, roles_not_train=[], n_chip=0, n_water_body=1)           # another chip of a held-out water body
    for role, uid, wb in (("val_p2", "p2:v1", "q8"), ("test_p2", "p2:t1", "q9")):
        t = pd.concat([_train(), _frame([(uid, "p2", role, wb, 47.0, -102.0, np.nan)])], ignore_index=True)
        broken(tr=t, roles_not_train=[role], n_chip=1, n_water_body=1)
        t = pd.concat([_train(), _frame([("p2:b9", "p2", "train_p2", wb, 30.0, -80.0, np.nan)])], ignore_index=True)
        broken(tr=t, n_chip=0, n_water_body=1)
    t = pd.concat([_train(), _frame([("t2", "national", "train", "w1", 40.0, -90.0, 11)])], ignore_index=True)
    broken(tr=t, n_chip=1, n_water_body=0)                                   # a time-held-out chip itself
    k = pd.concat([_keys(), _frame([("v9", "national", "val", "q7", 60.0, -120.0, np.nan)])], ignore_index=True)
    assert C.assert_refit_clean(_train(), k)["n_water_body"] == 0            # ids are compared inside a track only
    k = pd.concat([_keys(), _frame([("p2:v9", "p2", "val_p2", "z1", 40.005, -90.0, np.nan)])], ignore_index=True)
    broken(keys=k, n_water_body=0, n_cross_track_near=2, n_cross_track_hylak=0)     # the same water, other track
    k = pd.concat([_keys(), _frame([("v9", "national", "val", "z2", 60.0, -120.0, 31)])], ignore_index=True)
    broken(keys=k, n_cross_track_near=0, n_cross_track_hylak=1)              # the same HydroLAKES lake, other track
    assert set(C.CHIP_ROLES) == {"val", "val_n", "val_p2", "test", "test_n", "test_p2", "test_time", "test_time_n"}


# ------------------------------------------------------------------------------------------------ Q5 (ii)
def test_no_train_p2_water_body_is_a_national_training_water_body_under_another_id():
    c = C.assert_one_grouping(_train())
    assert c["n_national"] == 3 and c["n_p2"] == 2 and c["n_near"] == 0 and c["n_hylak"] == 0 and c["min_km"] > 400
    assert C.RADIUS_KM == 2.0
    near = pd.concat([_train(), _frame([("p2:b3", "p2", "train_p2", "q9", 40.01, -90.01, np.nan)])], ignore_index=True)
    assert C.cross_track_conflicts(near)["n_near"] == 1 and C.cross_track_conflicts(near)["min_km"] < 2.0
    with pytest.raises(C.CheckError):
        C.assert_one_grouping(near)
    assert C.cross_track_conflicts(near, radius_km=0.5)["n_near"] == 0
    hy = pd.concat([_train(), _frame([("p2:b3", "p2", "train_p2", "q9", 50.0, -110.0, 11)])], ignore_index=True)
    assert C.cross_track_conflicts(hy)["n_hylak"] == 1                       # HydroLAKES lake 11 is national w1
    with pytest.raises(C.CheckError):
        C.assert_one_grouping(hy)
    same = _train()
    same.loc[same["uid"] == "p2:b2", "water_body"] = "w2"                    # an id used by both tracks
    assert C.cross_track_conflicts(same)["n_shared_id"] == 1
    with pytest.raises(C.CheckError):
        C.assert_one_grouping(same)
    only = _train()[lambda d: d["track"] == "national"]
    assert C.assert_one_grouping(only)["min_km"] is None


# ------------------------------------------------------------------------------------------------ the real keys
REAL = L.labels_path().is_file() and PI.index_path().is_file()


@pytest.mark.skipif(not REAL, reason="label table or P2 index not on this machine")
def test_real_refit_table_is_clean():
    tr = C.train_keys()                                                      # training roles only
    keys = C.protected_keys()                                                # key columns only, no label column
    assert set(keys.columns) == {*C.KEY_COLS, "role", "uid", "track"}
    assert set(tr["role"]) == {"train", "train_p2"} and tr["uid"].is_unique
    o = C.assert_refit_clean(tr, keys)
    assert o["n_rows"] == len(tr) and o["n_chip"] == 0 and o["n_water_body"] == 0
    assert o["n_cross_track_near"] == 0 and o["n_cross_track_hylak"] == 0
    assert o["n_water_body_time"] > 0                                        # test_time: later dates of training lakes
    assert set(keys["role"]) == set(C.CHIP_ROLES)


@pytest.mark.skipif(not REAL, reason="label table or P2 index not on this machine")
def test_real_train_p2_rows_are_not_national_training_water_bodies():
    c = C.assert_one_grouping(C.train_keys())
    assert c["n_p2"] > 0 and c["n_national"] > 0 and c["n_shared_id"] == 0 and c["n_hylak"] == 0
    assert c["n_near"] == 0 and c["min_km"] > C.RADIUS_KM                    # no train_p2 site within 2 km


# ------------------------------------------------------------------------------------------------ A6
def _val_n(n_pos, n_wb_pos, n_neg=200, role="val_n"):
    y = np.r_[np.full(n_pos, np.log10(80.0)), np.full(n_neg, np.log10(5.0)), [np.nan] * 7]
    wb = [f"p{i % max(n_wb_pos, 1)}" for i in range(n_pos)] + [f"n{i % 40}" for i in range(n_neg)] + ["m"] * 7
    return pd.DataFrame({"y_turbidity": y, "water_body": wb, "role": role})


def test_e3a_counts_use_the_frozen_event_definition():
    c = C.e3a_counts(_val_n(48, 18))
    assert c == {"n": 248, "n_pos": 48, "n_wb_pos": 18, "n_wb": 58, "feasible": True}
    assert not C.e3a_counts(_val_n(29, 18))["feasible"] and C.e3a_counts(_val_n(30, 10))["feasible"]
    assert not C.e3a_counts(_val_n(60, 9))["feasible"]
    edge = pd.DataFrame({"y_turbidity": np.log10([50.0, 50.01, 49.9]), "water_body": ["a", "b", "c"]})
    assert C.e3a_counts(edge)["n_pos"] == 1                                  # > 50 NTU, not >= 50
    assert (C.E3A_MIN_POS, C.E3A_MIN_WB) == (30, 10)


def test_e3a_count_step_is_stamp_guarded_one_shot_and_registered(monkeypatch):
    rows, frames = [], {"rerun": _val_n(48, 18), "v2": _val_n(50, 19)}
    log = lambda **kw: rows.append(kw)   # noqa: E731

    def refuse(what, *a, **k):
        raise stamp.StampError(what)

    monkeypatch.setattr(stamp, "require_stamp", refuse)
    with pytest.raises(stamp.StampError):                                    # nothing is counted before the stamp
        C.e3a_turb_count(registry=[], log_row=log, frames=frames)
    assert rows == []
    monkeypatch.setattr(stamp, "require_stamp", lambda *a, **k: None)
    out = C.e3a_turb_count(registry=[], log_row=log, frames=frames)
    assert out["feasible"] and out["status"] == "confirmatory"
    assert out["counts"]["rerun"]["n_pos"] == 48 and out["counts"]["v2"]["n_wb_pos"] == 19
    assert len(rows) == 1 and rows[0]["experiment"] == C.E3A_EXPERIMENT and rows[0]["workstream"] == "M7-lite"
    assert rows[0]["value"] == 48.0 and rows[0]["extra"]["status"] == "confirmatory"
    reg = [{"workstream": rows[0]["workstream"], "experiment": rows[0]["experiment"], "extra": rows[0]["extra"]}]
    with pytest.raises(C.CheckError, match="counted once"):                  # one shot
        C.e3a_turb_count(registry=reg, log_row=log, frames=frames)
    assert len(rows) == 1
    for bad in ({"rerun": _val_n(29, 18), "v2": _val_n(50, 19)}, {"rerun": _val_n(48, 18), "v2": _val_n(40, 9)}):
        got = C.e3a_turb_count(registry=[], log_row=log, frames=bad)
        assert not got["feasible"] and got["status"] == "descriptive_only"   # < 30 positives or < 10 water bodies
        assert rows[-1]["extra"]["status"] == "descriptive_only"
    n = len(rows)
    with pytest.raises(C.CheckError, match="val_n rows only"):               # no other role is counted
        C.e3a_turb_count(registry=[], log_row=log, frames={"rerun": _val_n(48, 18, role="val")})
    assert len(rows) == n
