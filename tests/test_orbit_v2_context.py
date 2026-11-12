# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lake-context set of arm A1 (arch-v2 amendment §3.4 item 3) and the development context cache (§4 item 2).
Synthetic chip tables only: no chip store, no label and no model is read."""

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.models.orbit_v2 import context as CX

LON, LAT = 10.0, 50.0
DEG_PER_M = 1.0 / 111_194.9          # degrees of latitude per metre


def _cand(rows):
    """rows: dicts with any of the candidate columns; defaults give a usable in-lake train chip at the point."""
    base = {"waterbody": "L1", "in_lake": True, "block": "b1", "role": "train", "buffer_excluded": False,
            "qc_ok": True, "lon": LON, "lat": LAT, "date": "2020-06-01", "aot": 0.1}
    df = pd.DataFrame([{**base, **r} for r in rows])
    if "chip_id" not in df.columns:
        df["chip_id"] = [f"c{i:03d}" for i in range(len(df))]
    df["date"] = pd.to_datetime(df["date"])
    return df


def _row(**kw):
    base = {"waterbody": "L1", "in_lake": True, "block": "b1", "role": "train", "lon": LON, "lat": LAT,
            "date": "2020-06-01"}
    df = pd.DataFrame([{**base, **kw}])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _ids(cand, cidx):
    return [cand["chip_id"].iloc[j] for j in cidx[0] if j >= 0]


def test_time_window_and_same_date():
    days = {"same": 0, "d2": 2, "d3": 3, "m3": -3, "d183": 183, "d184": 184, "m183": -183, "m184": -184}
    t0 = pd.Timestamp("2020-06-01")
    cand = _cand([{"chip_id": k, "date": t0 + pd.Timedelta(days=v), "aot": 0.1 + 0.001 * i}
                  for i, (k, v) in enumerate(days.items())])
    cidx, cdt, counts = CX.select(_row(), cand)
    assert sorted(_ids(cand, cidx)) == ["d183", "d3", "m183", "m3"]
    got = dict(zip(_ids(cand, cidx), cdt[0][: 4].tolist()))
    assert got == {"d3": 3.0, "m3": -3.0, "d183": 183.0, "m183": -183.0}
    assert (cidx[0, 4:] == -1).all() and np.isnan(cdt[0, 4:]).all()
    assert counts["rows_with_1_to_5_context_dates"] == 1 and counts["rows_with_0_context_dates"] == 0


def test_one_chip_per_date_nearest_then_lowest_id():
    cand = _cand([
        {"chip_id": "far", "date": "2020-06-10", "lat": LAT + 300 * DEG_PER_M},
        {"chip_id": "near", "date": "2020-06-10", "lat": LAT + 20 * DEG_PER_M},
        {"chip_id": "tie_b", "date": "2020-06-20", "lat": LAT + 50 * DEG_PER_M},
        {"chip_id": "tie_a", "date": "2020-06-20", "lat": LAT + 50 * DEG_PER_M},
    ])
    cidx, _, _ = CX.select(_row(), cand)
    assert sorted(_ids(cand, cidx)) == ["near", "tie_a"]


def test_choice_lowest_aot_then_smaller_dt_then_earlier_date():
    t0 = pd.Timestamp("2020-06-01")
    spec = [("a", 40, 0.05), ("b", 10, 0.20), ("c", -10, 0.20), ("d", 5, 0.20), ("e", 90, 0.10), ("f", 30, np.nan),
            ("g", 60, 0.30), ("h", -60, 0.30), ("i", 4, 0.90)]
    cand = _cand([{"chip_id": k, "date": t0 + pd.Timedelta(days=d), "aot": a} for k, d, a in spec])
    cidx, cdt, counts = CX.select(_row(), cand)
    # AOT 0.05, 0.10, then 0.20 by |dt| 5, 10 (earlier date first: -10 before +10), then 0.30 (-60 before +60)
    assert _ids(cand, cidx) == ["a", "e", "d", "c", "b", "h"]
    assert cdt[0].tolist() == [40.0, 90.0, 5.0, -10.0, 10.0, -60.0]
    assert counts[f"rows_with_{CX.K}_context_dates"] == 1
    # unknown AOT is used only after every chip with a known AOT
    sub = cand[cand["chip_id"].isin(["f", "i", "g"])].reset_index(drop=True)
    cidx2, _, _ = CX.select(_row(), sub)
    assert _ids(sub, cidx2) == ["g", "i", "f"]


def test_qc_buffer_and_date_limit():
    cand = _cand([
        {"chip_id": "ok", "date": "2023-12-31"},
        {"chip_id": "qc", "date": "2023-12-20", "qc_ok": False},
        {"chip_id": "buf", "date": "2023-12-21", "buffer_excluded": True},
        {"chip_id": "late", "date": "2024-01-02"},
        {"chip_id": "nocoord", "date": "2023-12-22", "lon": np.nan},
    ])
    cidx, _, counts = CX.select(_row(date="2023-12-10"), cand)
    assert _ids(cand, cidx) == ["ok"]
    assert counts["n_candidates"] == 5 and counts["n_candidates_usable"] == 1
    with pytest.raises(CX.ContextError, match="dated after 2023-12-31"):
        CX.select(_row(date="2024-01-05"), cand)


def test_in_lake_members_are_the_same_lake_only():
    cand = _cand([
        {"chip_id": "same", "date": "2020-06-10", "lat": LAT + 5000 * DEG_PER_M},
        {"chip_id": "other", "date": "2020-06-11", "waterbody": "L2"},
        {"chip_id": "nonlake", "date": "2020-06-12", "waterbody": "p1", "in_lake": False},
    ])
    cidx, _, counts = CX.select(_row(), cand)
    assert _ids(cand, cidx) == ["same"] and counts["rule"] == CX.RULE_R7
    bad = _cand([{"chip_id": "x", "date": "2020-06-10", "role": "test"}])
    with pytest.raises(CX.ContextError, match="another split role"):
        CX.select(_row(), bad)


def test_non_lake_members_within_100_m_same_role_and_block():
    cand = _cand([
        {"chip_id": "in90", "date": "2020-06-10", "lat": LAT + 90 * DEG_PER_M},
        {"chip_id": "out110", "date": "2020-06-11", "lat": LAT + 110 * DEG_PER_M},
        {"chip_id": "role", "date": "2020-06-12", "role": "val"},
        {"chip_id": "block", "date": "2020-06-13", "block": "b2"},
        {"chip_id": "lake", "date": "2020-06-14", "in_lake": True, "waterbody": "L9"},
    ])
    cand.loc[cand["chip_id"] != "lake", ["waterbody", "in_lake"]] = ["p1", False]
    row = _row(waterbody="p0", in_lake=False)
    cidx, _, counts = CX.select(row, cand)
    assert _ids(cand, cidx) == ["in90"]
    assert counts["n_nonlake_members_dropped_other_role_block_or_lake"] == 3
    # every chip here is a test-role chip of the same place: none is a member of a train-role row
    cidx, _, counts = CX.select(row, cand.assign(role="test"))
    assert _ids(cand, cidx) == [] and counts["n_nonlake_members_dropped_other_role_block_or_lake"] == 4


def test_r7_is_a_constant_without_a_switch():
    """C-2: the strict reading of R-7 has no other setting."""
    import inspect

    par = inspect.signature(CX.select).parameters
    assert list(par) == ["rows", "cand", "k"] and "same_role" not in par
    with pytest.raises(TypeError):
        CX.select(_row(), _cand([{"date": "2020-06-10"}]), same_role=False)
    assert "same frozen-rule water body and the same split role" in CX.RULE_R7
    assert CX.NONLAKE_RADIUS_M == 100.0 and CX.STORES == ("aquacorr", "wqp")
    with pytest.raises(CX.ContextError, match="not a context store"):
        CX.store_dir("naip_teacher")
    with pytest.raises(CX.ContextError, match="not a context store"):
        CX.store_dir("mc_benchmark")


def test_label_columns_are_refused():
    cand = _cand([{"date": "2020-06-10"}])
    for col in ("chla", "tp_cens", "value", "secchi_value", "doc_dl", "censored"):
        with pytest.raises(CX.ContextError, match="label columns"):
            CX.select(_row(), cand.assign(**{col: 1.0}))
        with pytest.raises(CX.ContextError, match="label columns"):
            CX.select(_row().assign(**{col: 1.0}), cand)
    with pytest.raises(CX.ContextError, match="missing columns"):
        CX.select(_row(), cand.drop(columns="aot"))
    with pytest.raises(CX.ContextError, match="not unique"):
        CX.select(_row(), pd.concat([cand, cand], ignore_index=True))


def test_selection_does_not_depend_on_row_or_candidate_order():
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2020-06-01")
    cand = _cand([{"chip_id": f"k{i:03d}", "waterbody": f"L{i % 3}", "date": t0 + pd.Timedelta(days=int(d)),
                   "aot": float(a), "lat": LAT + float(m) * DEG_PER_M}
                  for i, (d, a, m) in enumerate(zip(rng.integers(-250, 250, 120), rng.choice([0.05, 0.1, 0.2], 120),
                                                    rng.uniform(0, 2000, 120)))])
    rows = pd.concat([_row(waterbody=f"L{i % 3}", date=t0 + pd.Timedelta(days=7 * i)) for i in range(9)],
                     ignore_index=True)
    cidx, cdt, _ = CX.select(rows, cand)
    ref = [[cand["chip_id"].iloc[j] if j >= 0 else None for j in r] for r in cidx]
    perm, rperm = rng.permutation(len(cand)), rng.permutation(len(rows))
    cand2, rows2 = cand.iloc[perm].reset_index(drop=True), rows.iloc[rperm].reset_index(drop=True)
    cidx2, cdt2, _ = CX.select(rows2, cand2)
    got = [[cand2["chip_id"].iloc[j] if j >= 0 else None for j in r] for r in cidx2]
    assert got == [ref[i] for i in rperm]
    assert np.array_equal(cdt2, cdt[rperm], equal_nan=True)
    # every chosen chip is of the row's lake, inside the window, one per date
    for i, r in enumerate(cidx):
        ch = cand.iloc[r[r >= 0]]
        assert (ch["waterbody"] == rows["waterbody"].iloc[i]).all()
        dt = (ch["date"] - rows["date"].iloc[i]).dt.days.abs()
        assert dt.between(CX.DT_MIN_D, CX.DT_MAX_D).all() and ch["date"].is_unique


# ------------------------------------------------------------------------------------ development cache

def _meta(**kw):
    m = pd.DataFrame({"chip_id": ["a", "b", "c"], "waterbody": ["L1", "L1", "L2"], "role": ["train", "train", "val"],
                      "buffer_excluded": [False, False, False], "protected": [False, False, False],
                      "date": pd.to_datetime(["2019-05-01", "2023-12-31", "2022-07-01"])})
    for k, v in kw.items():
        m[k] = v
    return m


def test_cache_round_trip(tmp_path):
    meta = _meta()
    arr = {"px": np.arange(3 * 8 * 12, dtype=np.float32).reshape(3, 8, 12), "q": np.ones((3, 8), np.float32)}
    rec = CX.write_cache(tmp_path / "ctx", meta.drop(columns="protected"), arr, allow=CX.dev_allow_list(meta),
                         info={"k": CX.K})
    assert rec["n_chips"] == 3 and rec["roles"] == ["train", "val"] and rec["date_limit"] == "2023-12-31"
    m2, a2, info = CX.load_cache(tmp_path / "ctx")
    assert m2["chip_id"].tolist() == ["a", "b", "c"] and np.array_equal(a2["px"], arr["px"]) and info["k"] == CX.K


@pytest.mark.parametrize("change, msg", [
    ({"role": ["train", "train", "test"]}, "1 chips of a non-development role, 0 buffer-excluded, 0 dated"),
    ({"role": ["train", "train", "test_s"]}, "1 chips of a non-development role"),
    ({"buffer_excluded": [False, True, False]}, "0 chips of a non-development role, 1 buffer-excluded, 0 dated"),
    ({"date": pd.to_datetime(["2019-05-01", "2024-01-01", "2022-07-01"])}, "0 buffer-excluded, 1 dated after 2023-12-31"),
    ({"chla": [1.0, 2.0, 3.0]}, "label columns"),
    ({"chip_id": ["a", "a", "c"]}, "duplicate chip ids"),
    ({"role": ["train", "val", "val"]}, "two roles"),
])
def test_cache_writer_refuses(tmp_path, change, msg):
    meta = _meta(**change).drop(columns="protected")
    with pytest.raises(CX.ContextError, match=msg):
        CX.write_cache(tmp_path / "ctx", meta, {"px": np.zeros((3, 2))}, allow=["a", "b", "c"])
    assert not (tmp_path / "ctx").exists()            # nothing is written when the check fails


def test_cache_writer_needs_the_allow_list(tmp_path):
    meta = _meta()
    idx = _meta(protected=[False, True, False])
    assert CX.dev_allow_list(idx).tolist() == ["a", "c"]
    assert CX.dev_allow_list(_meta(role=["test", "train", "val"])).tolist() == ["b", "c"]
    assert CX.dev_allow_list(_meta(date=pd.to_datetime(["2024-03-01", "2020-01-01", "2020-01-01"]))).tolist() == ["b", "c"]
    with pytest.raises(CX.ContextError, match="1 chips are not in the development allow-list"):
        CX.write_cache(tmp_path / "ctx", meta.drop(columns="protected"), {"px": np.zeros((3, 2))},
                       allow=CX.dev_allow_list(idx))
    with pytest.raises(TypeError):
        CX.write_cache(tmp_path / "ctx", meta.drop(columns="protected"), {"px": np.zeros((3, 2))})
    with pytest.raises(CX.ContextError, match="array 'px' has 2 entries"):
        CX.write_cache(tmp_path / "ctx", meta.drop(columns="protected"), {"px": np.zeros((2, 2))}, allow=["a", "b", "c"])
    assert not (tmp_path / "ctx").exists()


def test_cache_loader_refuses_changed_or_foreign_content(tmp_path):
    d = tmp_path / "ctx"
    meta = _meta().drop(columns="protected")
    CX.write_cache(d, meta, {"px": np.zeros((3, 2), np.float32)}, allow=["a", "b", "c"])
    # 1. file changed after writing: hash mismatch
    bad = meta.copy()
    bad.loc[2, "role"] = "test"
    bad.to_parquet(d / CX.META_FILE, index=False)
    with pytest.raises(CX.ContextError, match="content differs"):
        CX.load_cache(d)
    # 2. hash record rewritten as well: the chip check still refuses the cache
    info = json.loads((d / CX.INFO_FILE).read_text())
    info["sha256"][CX.META_FILE] = CX._sha(d / CX.META_FILE)
    (d / CX.INFO_FILE).write_text(json.dumps(info))
    with pytest.raises(CX.ContextError, match="1 chips of a non-development role"):
        CX.load_cache(d)


# ------------------------------------------------------------------------------------ store pin (R-7)

def _pin_fixture(tmp_path):
    files = tmp_path / "stores"
    idx = {"aquacorr": pd.DataFrame({"chip_id": ["a0", "a1", "a2"], "shard": ["s0.npz", "s0.npz", "s1.npz"]}),
           "wqp": pd.DataFrame({"chip_id": ["w0", "w1"], "shard": ["s0.npz", "s0.npz"]})}
    for st, df in idx.items():
        (files / st).mkdir(parents=True)
        for sh in df["shard"].unique():
            (files / st / sh).write_bytes(f"{st}/{sh}".encode())
    return idx, (lambda store, shard: files / store / shard), files


def test_pin_covers_exactly_the_two_stores(tmp_path):
    idx, shard_file, _ = _pin_fixture(tmp_path)
    chips = CX.pin_table(idx)
    assert list(chips.columns) == list(CX.PIN_COLS) and len(chips) == 5 and set(chips["store"]) == set(CX.STORES)
    for bad in ({"wqp": idx["wqp"]}, {**idx, "naip_teacher": idx["wqp"]}, {"aquacorr": idx["aquacorr"], "x": idx["wqp"]}):
        with pytest.raises(CX.ContextError, match="exactly the stores"):
            CX.pin_table(bad)
    rec = CX.write_pin(tmp_path / "pin", chips, shard_file=shard_file, created="t", log=lambda *a: None)
    assert rec["n_chips"] == {"aquacorr": 3, "wqp": 2} and rec["n_shards"] == 3
    assert sorted(rec["shard_sha256"]) == ["aquacorr/s0.npz", "aquacorr/s1.npz", "wqp/s0.npz"]
    c2, r2 = CX.load_pin(tmp_path / "pin")
    assert c2.equals(chips) and r2 == rec
    with pytest.raises(CX.ContextError, match="made once and never replaced"):
        CX.write_pin(tmp_path / "pin", chips, shard_file=shard_file)
    with pytest.raises(CX.ContextError, match="exactly the columns"):
        CX.write_pin(tmp_path / "pin2", chips.assign(extra=1), shard_file=shard_file)
    assert not (tmp_path / "pin2").exists()


def test_pin_excludes_later_chips_and_other_stores(tmp_path):
    idx, shard_file, _ = _pin_fixture(tmp_path)
    chips = CX.pin_table(idx)
    cand = pd.DataFrame({"store": ["aquacorr", "wqp", "wqp", "naip_teacher", "aquacorr"],
                         "store_chip_id": ["a0", "w1", "w9", "a0", "a1"],
                         "shard": ["s0.npz", "s0.npz", "s7.npz", "s0.npz", "s1.npz"]})   # last: pinned chip, other shard
    kept, counts = CX.apply_pin(cand, chips)
    assert kept["store_chip_id"].tolist() == ["a0", "w1"]
    assert counts == {"n_candidates": 5, "n_in_pin": 2, "n_not_in_pin_dropped": 3}
    forged = pd.concat([chips, pd.DataFrame({"store": ["naip_teacher"], "store_chip_id": ["a0"], "shard": ["s0.npz"]})])
    assert CX.apply_pin(cand, forged)[0]["store"].tolist() == ["aquacorr", "wqp"]       # never a third store


def test_pin_refuses_changed_content(tmp_path):
    idx, shard_file, files = _pin_fixture(tmp_path)
    d = tmp_path / "pin"
    rec = CX.write_pin(d, CX.pin_table(idx), shard_file=shard_file, log=lambda *a: None)
    used = [("aquacorr", "s0.npz"), ("wqp", "s0.npz"), ("aquacorr", "s0.npz")]
    assert CX.check_pinned_shards(rec, used, shard_file=shard_file) == 2
    with pytest.raises(CX.ContextError, match="not in the context store pin"):
        CX.check_pinned_shards(rec, [("wqp", "s9.npz")], shard_file=shard_file)
    (files / "wqp" / "s0.npz").write_bytes(b"rewritten after the pin")
    with pytest.raises(CX.ContextError, match="content differs from the context store pin"):
        CX.check_pinned_shards(rec, used, shard_file=shard_file)
    # the chip table of the pin itself
    chips, _ = CX.load_pin(d)
    pd.concat([chips, chips.iloc[[0]].assign(store_chip_id="later")]).to_parquet(d / CX.PIN_CHIPS, index=False)
    with pytest.raises(CX.ContextError, match="content differs from the hash recorded"):
        CX.load_pin(d)
