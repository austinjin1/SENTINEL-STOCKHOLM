# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 harder tracks (protocol §10–§11): region folds, frozen fold file, track-R roles, strata. Synthetic arrays only."""

import numpy as np
import pandas as pd

from sentinel.models.hydromet import tracks as TR


def test_region_folds_cover_all_huc2_once():
    h = [x for v in TR.REGION_FOLDS.values() for x in v]
    assert sorted(h) == [f"{i:02d}" for i in range(1, 19)] and len(h) == len(set(h))


def test_basin_group_goes_whole_to_majority_region():
    sites = ["a", "b", "c", "d"]
    huc8 = {"a": "17010101", "b": "17020202", "c": "16010101", "d": "01010101"}
    bg = {"a": "g1", "b": "g1", "c": "g1", "d": "g2"}
    f = TR.assign_folds(sites, huc8, bg)
    assert f["a"] == f["b"] == f["c"] == "R6_pacific" and f["d"] == "R1_northeast"


def test_fold_roles_never_fit_test_sites():
    fold_of = {"a": "R1_northeast", "b": "R2_southeast", "c": "R1_northeast"}
    role_of = {"a": "train", "b": "val", "c": "test"}
    r = TR.fold_roles(fold_of, role_of, "R1_northeast")
    assert r == {"a": "val", "b": "train", "c": "test"}


def test_frozen_fold_file_matches_protocol_hash():
    body = TR.load_folds()
    assert body["sha256"] == TR.TRACK_R_SHA256
    assert set(body["fold_of"].values()) <= set(TR.REGION_FOLDS)
    # basin groups never straddle regions
    from sentinel.models.hydromet import data as D

    sp, _ = D.load_split()
    df = pd.DataFrame({"s": list(body["fold_of"]), "f": list(body["fold_of"].values())})
    df["g"] = df["s"].map(sp.basin_group)
    assert (df.groupby("g")["f"].nunique() == 1).all()


def test_heatwave_mask_runs_and_tail():
    T = 40
    tmax = np.zeros((1, T))
    train = np.zeros(T, bool)
    train[:20] = True
    tmax[0, :20] = np.arange(20)          # train P95 = 18.05
    tmax[0, 25:28] = 30                   # 3-day run -> flagged + 3 following days
    tmax[0, 32:34] = 30                   # 2-day run -> not flagged
    m = TR.heatwave_mask(tmax, train)
    assert m[0, 25:31].all() and not m[0, 31:].any() and not m[0, :25].any()


def test_csi():
    o = np.array([1, 1, 0, 0, 1], bool)
    p = np.array([1, 0, 1, 0, 1], bool)
    assert np.isclose(TR.csi(o, p), 2 / 4)


def test_track_n_roles_follow_split_and_flow():
    from sentinel.models.hydromet import track_n as TN

    # network: a(1) -> b(2) -> c(3) -> d(4); e isolated in a new HUC4
    vaa = pd.DataFrame({"comid": ["a", "b", "c", "d", "e"], "hydroseq": [50, 40, 30, 20, 99],
                        "dnhydroseq": [40, 30, 20, 0, 0], "pathlength": [30.0, 20.0, 10.0, 0.0, 0.0]})
    split = pd.DataFrame({"site_no": ["s1", "s2"], "comid": ["c", np.nan], "huc8": ["01010001", "02020002"],
                          "role": ["test", "val"], "basin_group": ["G1", "G2"]})
    st = pd.DataFrame({"station_key": ["W1", "W2", "W3", "W4"], "comid": ["a", "d", "b", "e"],
                       "huc8": ["02020002", "03030003", "01010001", "09090009"]})
    r = TN.assign_roles(st, split, vaa).set_index("station_key")
    assert r.loc["W1", "role"] == "excluded"             # HUC4 of a val group but drains into a test site
    assert r.loc["W2", "role"] == "test"                 # downstream of the test site only
    assert r.loc["W3", "role"] == "test" and r.loc["W3", "basin_group"] == "G1"
    assert r.loc["W4", "rule"] == "new_region" and r.loc["W4", "role"] in ("val", "test")
    assert r.loc["W4", "basin_group"] == "N:0909"
