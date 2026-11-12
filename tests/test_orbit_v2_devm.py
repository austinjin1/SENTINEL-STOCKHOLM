# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""dev_v5m rules and cache (arch-v2 amendment §4). Synthetic tables and chips, with one exception: once the real
dev_v5m cache exists, ``test_real_cache_holds_no_protected_id`` re-derives the protected key sets (keys and counts
only, no label value) and audits the written cache against them (R-21). Everything else reads no chip store, no
label table and no real cache; nothing is written outside pytest's temporary directory."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data import chips as C
from sentinel.models.orbit import data as D
from sentinel.models.orbit import guard
from sentinel.models.orbit_v2 import chipfeat as CF
from sentinel.models.orbit_v2 import context as CX
from sentinel.models.orbit_v2 import devm as DM
from sentinel.models.orbit_v2 import targets as T

SPLIT_COLS = ["role", "block", "tblock", "waterbody", "buffer_excluded", "protected"]


def _base(n=6):
    d = pd.to_datetime(["2019-06-01", "2019-07-01", "2020-06-01", "2022-06-01", "2023-06-01", "2021-08-01"][:n])
    return pd.DataFrame({
        "row_id": [f"r{i}" for i in range(n)], "chip_id": [f"k{i}" for i in range(n)],
        "lon": 10.0, "lat": 50.0 + 0.01 * np.arange(n), "date": d, "chip_date": d,
        "chla": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0][:n], "tss": [np.nan, 3.0, 3.5, np.nan, np.nan, 1.0][:n],
        "turbidity": np.nan, "secchi": [2.0, np.nan, 1.0, 1.5, np.nan, np.nan][:n],
        "waterbody": ["L1", "L1", "L1", "L2", "L2", "Pb|c"][:n], "in_lake": [True, True, True, True, True, False][:n],
        "block": ["b1", "b1", "b1", "b2", "b2", "Nb3"][:n], "role": ["train", "train", "train", "val", "val", "train"][:n],
        "tblock": ["train", "train", "train", "val", "val", "train"][:n], "buffer_excluded": False, "protected": False,
    })


def _table():
    base = _base()
    add = pd.DataFrame({
        "row_id": ["a0"], "chip_id": ["k9"], "lon": [10.0], "lat": [50.2], "date": pd.to_datetime(["2020-07-01"]),
        "chip_date": pd.to_datetime(["2020-07-01"]), "chla": [np.nan], "tss": [np.nan], "turbidity": [np.nan],
        "secchi": [np.nan], "waterbody": ["L1"], "in_lake": [True], "block": ["b1"], "role": ["train"],
        "tblock": ["train"], "buffer_excluded": [False], "protected": [False]})
    df = pd.concat([base.assign(added=False), add.assign(added=True)], ignore_index=True)
    for t in T.SPECS:
        df[t], df[f"{t}_cens"], df[f"{t}_dl"] = np.nan, False, np.nan
    df.loc[0, "tp"] = 0.05
    df.loc[1, ["tp_cens", "tp_dl"]] = [True, 0.01]
    df.loc[6, "doc"] = 4.0
    return base, df


def _ctx():
    cmeta = pd.DataFrame({
        "chip_id": ["wqp:c0", "wqp:c1", "aquacorr:c2"], "waterbody": ["L1", "L1", "L2"], "in_lake": True,
        "block": ["b1", "b1", "b2"], "role": ["train", "train", "val"], "buffer_excluded": False,
        "date": pd.to_datetime(["2019-06-11", "2019-08-01", "2022-06-21"]), "platform": "S2A",
        "lon": 10.0, "lat": [50.0, 50.3, 50.03]})
    return cmeta, _carr(3)


def _carr(m):
    return {"refl": np.zeros((m, DM.P_CONTEXT, 12), np.float16), "dp_m": np.ones((m, DM.P_CONTEXT), np.float32),
            "q": np.zeros((m, 8), np.float32), "offsets": np.zeros((m, DM.P_CONTEXT, 2), np.int8),
            "n_pix": np.full(m, DM.P_CONTEXT, np.int32), "rw3": np.full((m, 12), 0.03, np.float32)}


def _feats(df):
    n = len(df)
    cidx = np.full((n, CX.K), -1, np.int64)
    cdt = np.full((n, CX.K), np.nan, np.float32)
    cidx[0, :2], cdt[0, :2] = [0, 1], [10.0, 61.0]          # row 0 (L1, 2019-06-01)
    cidx[1, 0], cdt[1, 0] = 0, -20.0                        # row 1 (L1, 2019-07-01)
    cidx[3, 0], cdt[3, 0] = 2, 20.0                         # row 3 (L2, 2022-06-01)
    return {"dp_m": np.ones((n, 64), np.float32), "ring": np.zeros((n, CF.RING_DIM), np.float32),
            "glint": np.zeros((n, 3), np.float32), "cidx": cidx, "cdt": cdt}


def _pix(df):
    return {"refl": np.zeros((len(df), 64, 12), np.float16), "n_pix": np.full(len(df), 9, np.int32)}


def _write(d, df=None, base=None, feats=None, cmeta=None, carr=None, allow=None):
    b0, d0 = _table()
    df = d0 if df is None else df
    cm0, ca0 = _ctx()
    cmeta = cm0 if cmeta is None else cmeta
    return DM.write(d, df, _pix(df), _feats(df) if feats is None else feats, {"note": "synthetic"},
                    base=b0 if base is None else base, cmeta=cmeta, carrays=ca0 if carr is None else carr,
                    allow=cmeta["chip_id"].tolist() if allow is None else allow)


CHIP_SPLIT = ["role", "block", "waterbody", "in_lake", "buffer_excluded"]


def _assigner(chips=None, rows=None):
    """Stand-in for the frozen rule: rows keep their stored split columns (after ``rows``); context chips (handed
    over as lon / lat / date only) get the columns of ``chips`` (default: the fixture), matched by date."""
    truth = (_ctx()[0] if chips is None else chips).copy()
    if "protected" not in truth.columns:
        truth["protected"] = False

    def f(df):
        if "row_id" in df.columns:
            return df[SPLIT_COLS] if rows is None else rows(df[SPLIT_COLS].copy())
        assert list(df.columns) == ["lon", "lat", "date"]                 # chips: coordinates and date, nothing else
        return truth.set_index("date").loc[df["date"], [*CHIP_SPLIT, "protected"]].reset_index(drop=True)
    return f


_same = _assigner()


def test_round_trip(tmp_path):
    base, df = _table()
    rec = _write(tmp_path / "c")
    assert rec["counts"]["n_base_rows"] == 6 and rec["counts"]["n_added_rows"] == 1
    assert rec["counts"]["context"] == {"rows_with_0_context_dates": 4, "rows_with_1_to_5_context_dates": 3,
                                        f"rows_with_{CX.K}_context_dates": 0}
    t2, pix, feats, (cmeta, carr), info = DM.load(tmp_path / "c", base=base, assign=_same)
    assert t2["row_id"].tolist() == df["row_id"].tolist() and info["context_rule"] == CX.RULE_R7
    assert info["join_rule"] == T.RULE_R8 and info["context_stores"] == ["aquacorr", "wqp"] and "same_role" not in info
    assert np.array_equal(feats["cidx"], _feats(df)["cidx"]) and len(cmeta) == 3 and carr["refl"].shape[1] == 8
    assert t2.loc[1, "tp_cens"] and np.isnan(t2.loc[1, "tp"]) and t2.loc[1, "tp_dl"] == 0.01


def _mod(col, row, val):
    def f(df):
        df = df.copy()
        df.loc[row, col] = val
        return df
    return f


@pytest.mark.parametrize("change, err, msg", [
    (_mod("protected", 2, True), guard.ProtectedDataError, ""),
    (_mod("role", 2, "test"), DM.DevmError, "1 rows of a non-development role, 0 of a"),
    (_mod("tblock", 2, "test"), DM.DevmError, "0 rows of a non-development role, 1 of a non-development time block, 0 dated"),
    (_mod("date", 2, pd.Timestamp("2024-02-01")), DM.DevmError, "1 dated after 2023-12-31"),
    (_mod("buffer_excluded", 0, True), DM.DevmError, "buffer-excluded row carries a new-target value"),
    (_mod("tp", 1, 0.02), DM.DevmError, "both quantified and censored"),
    (_mod("tp_dl", 1, np.nan), DM.DevmError, "no detection limit"),
    (_mod("tp_dl", 1, 50.0), DM.DevmError, "no detection limit inside the valid range"),
    (_mod("tn", 3, 500.0), DM.DevmError, "outside the valid range"),
    (_mod("chla", 0, 1.5), DM.DevmError, "column 'chla' differs from dev_v5"),
    (_mod("chla", 2, 7.0), DM.DevmError, "column 'chla' differs from dev_v5"),
    (_mod("chip_id", 4, "other"), DM.DevmError, "column 'chip_id' differs"),
    (_mod("added", 5, True), DM.DevmError, "not kept one to one"),
    (_mod("secchi", 6, 2.0), DM.DevmError, "added row carries a base-target value"),
    (_mod("doc", 6, np.nan), DM.DevmError, "added rows carry no new-target value"),
    (_mod("row_id", 6, "r0"), DM.DevmError, "duplicate row ids"),
])
def test_writer_refuses_rows(tmp_path, change, err, msg):
    _, df = _table()
    with pytest.raises(err, match=msg):
        _write(tmp_path / "c", df=change(df))
    assert not (tmp_path / "c").exists()


def test_writer_refuses_a_dropped_base_row(tmp_path):
    _, df = _table()
    with pytest.raises(DM.DevmError, match="1 missing"):
        _write(tmp_path / "c", df=df.drop(index=2).reset_index(drop=True),
               feats={k: np.delete(v, 2, 0) for k, v in _feats(df).items()})


def _feat_mod(fn):
    _, df = _table()
    f = _feats(df)
    fn(f)
    return f


@pytest.mark.parametrize("fn, msg", [
    (lambda f: f["cidx"].__setitem__((2, 0), 7), "outside the context cache"),
    (lambda f: (f["cidx"].__setitem__((3, 1), 0), f["cdt"].__setitem__((3, 1), -1086.0)), "outside the 3-183 day window"),
    (lambda f: f["cdt"].__setitem__((0, 0), 11.0), "do not match the chip dates"),
    (lambda f: (f["cidx"].__setitem__((4, 0), 1), f["cdt"].__setitem__((4, 0), -1400.0)), "outside the 3-183 day window"),
    (lambda f: f.__setitem__("cidx", f["cidx"][:, :3]), "context index shape"),
    (lambda f: f.pop("ring"), "row array 'ring' is missing"),
])
def test_writer_refuses_context_index(tmp_path, fn, msg):
    with pytest.raises(DM.DevmError, match=msg):
        _write(tmp_path / "c", feats=_feat_mod(fn))
    assert not (tmp_path / "c").exists()


def test_context_membership_is_rechecked(tmp_path):
    _, df = _table()
    cmeta, _ = _ctx()
    # an in-lake row with a chip of another lake inside the time window
    other = cmeta.copy()
    other.loc[0, "waterbody"] = "L7"
    with pytest.raises(DM.DevmError, match="another water body or role"):
        _write(tmp_path / "c", cmeta=other)
    with pytest.raises(DM.DevmError, match="another water body or role"):
        _write(tmp_path / "c", cmeta=cmeta.assign(role="val"))
    # non-lake row 5 (2021-08-01, train, block Nb3, lon 10.0, lat 50.05): R-7 has one reading only
    def nonlake(**kw):
        chip = {"chip_id": "wqp:c3", "waterbody": "Pb|c2", "in_lake": False, "block": "Nb3", "role": "train",
                "buffer_excluded": False, "date": pd.Timestamp("2021-08-11"), "platform": "S2A", "lon": 10.0,
                "lat": 50.05 + 60.0 / 111_195.0, **kw}
        return pd.concat([cmeta, pd.DataFrame([chip])], ignore_index=True)

    nl = nonlake()
    f = _feats(df)
    f["cidx"][5, 0], f["cdt"][5, 0] = 3, 10.0
    carr = _carr(4)
    for col in ("platform", "lon", "lat", "block", "in_lake"):
        with pytest.raises(DM.DevmError, match="missing chip columns or arrays"):
            _write(tmp_path / "c", cmeta=nl.drop(columns=col), carr=carr, feats=f)
    with pytest.raises(DM.DevmError, match="missing chip columns or arrays"):
        _write(tmp_path / "c", cmeta=nl, carr={k: v for k, v in carr.items() if k != "rw3"}, feats=f)
    for kw, msg in (({"role": "val"}, "non-lake row has a context chip of another split role"),
                    ({"block": "Nb4"}, "another 200-km block or an in-lake chip"),
                    ({"in_lake": True, "waterbody": "L5"}, "another 200-km block or an in-lake chip"),
                    ({"buffer_excluded": True}, "buffer-excluded"),
                    ({"lat": 50.05 + 140.0 / 111_195.0}, "centred more than 100 m away")):
        with pytest.raises((DM.DevmError, CX.ContextError), match=msg):
            _write(tmp_path / "c", cmeta=nonlake(**kw), carr=carr, feats=f)
    with pytest.raises(TypeError):
        DM.write(tmp_path / "c", df, _pix(df), f, {}, base=_base(), cmeta=nl, carrays=carr, allow=nl["chip_id"].tolist(),
                 same_role=False)                                           # C-2: no switch
    assert not (tmp_path / "c").exists()
    _write(tmp_path / "c", cmeta=nl, carr=carr, feats=f)
    DM.load(tmp_path / "c", assign=_assigner(nl))
    # two chips of one date for one row
    dup = cmeta.copy()
    dup.loc[1, "date"] = dup.loc[0, "date"]
    f2 = _feats(df)
    f2["cdt"][0, 1] = 10.0
    with pytest.raises(DM.DevmError, match="two context chips of the same date"):
        _write(tmp_path / "c2", cmeta=dup, feats=f2)
    # a test-role or late context chip never enters the cache
    with pytest.raises(CX.ContextError, match="1 chips of a non-development role"):
        _write(tmp_path / "c3", cmeta=cmeta.assign(role=["train", "train", "test"]))
    with pytest.raises(CX.ContextError, match="not in the development allow-list"):
        _write(tmp_path / "c3", allow=["wqp:c0", "wqp:c1"])
    assert not (tmp_path / "c3").exists()


def test_loader_refuses_changed_files_and_split_drift(tmp_path):
    base, df = _table()
    d = tmp_path / "c"
    _write(d)
    with pytest.raises(DM.DevmError, match="differs from the frozen rule"):
        DM.load(d, assign=lambda t: t[SPLIT_COLS].assign(role=["val"] + t["role"].tolist()[1:]))
    with pytest.raises(DM.DevmError, match="column 'chla' differs"):
        DM.load(d, base=base.assign(chla=base["chla"] + 1.0), assign=_same)
    # table replaced by one with a protected row, hash record rewritten as well: still refused
    bad = df.copy()
    bad.loc[2, "protected"] = True
    bad.to_parquet(d / DM.TABLE, index=False)
    with pytest.raises(DM.DevmError, match="content differs"):
        DM.load(d, assign=_same)
    info = json.loads((d / DM.INFO).read_text())
    info["sha256"][DM.TABLE] = CX._sha(d / DM.TABLE)
    (d / DM.INFO).write_text(json.dumps(info))
    with pytest.raises(guard.ProtectedDataError):
        DM.load(d, assign=_same)
    with pytest.raises(guard.ProtectedDataError):
        DM.load(d, assign=_same, training=False)       # no freeze file: evaluation code is refused as well


@pytest.mark.parametrize("col, val, msg", [
    ("role", ["train", "train", "train"], "context chip column 'role' differs from the frozen rule"),
    ("waterbody", ["L1", "L9", "L2"], "context chip column 'waterbody' differs"),
    ("block", ["b1", "b1", "b9"], "context chip column 'block' differs"),
    ("in_lake", [True, False, True], "context chip column 'in_lake' differs"),
    ("buffer_excluded", [False, False, True], "context chip column 'buffer_excluded' differs"),
    ("protected", [False, True, False], "holds a chip the frozen rule protects"),
])
def test_loader_rederives_the_context_chips_from_the_frozen_rule(tmp_path, col, val, msg):
    """R-21: the chips' split columns are not trusted from the cache; a chip the rule now protects is refused."""
    d = tmp_path / "c"
    _write(d)
    DM.load(d, assign=_same)
    with pytest.raises(DM.DevmError, match=msg):
        DM.load(d, assign=_assigner(_ctx()[0].assign(**{col: val})))


# ------------------------------------------------------------------------------------ artefact audit (R-21)

def test_no_protected_row_or_chip_id_in_any_artefact(tmp_path):
    """R-21: no protected row id and no protected chip id is in any dev_v5m artefact."""
    d = tmp_path / "c"
    base, df = _table()
    df = df.assign(chip_store=["wqp"] * 6 + ["aquacorr"])
    _write(d, df=df)
    n = DM.audit_artefacts(d, ["p0", "p1"], ["wqp:zz", "aquacorr:k0", "naip_teacher:k9"])
    assert n["files"] == 7 and n["n_protected_row_ids"] == 2 and n["n_protected_chip_ids"] == 3
    assert n["string_columns"] >= 6 and n["json_tokens"] > 20
    assert sorted(p.name for p in d.rglob("*") if p.is_file()) == sorted(
        [DM.TABLE, DM.PIXELS, DM.FEATURES, DM.INFO, CX.META_FILE, CX.ARRAY_FILE, CX.INFO_FILE])
    for rows, chips, msg in ((["r3"], [], f"{DM.TABLE}:row_id: 1 entries"),               # a row id of the table
                             ([], ["wqp:k0"], "chip_store\\+chip_id: 1 entries"),         # the chip of a row
                             ([], ["aquacorr:k9"], "chip_store\\+chip_id: 1 entries"),
                             ([], ["wqp:c1"], f"{CX.META_FILE}:chip_id: 1 entries"),      # a context chip
                             ([], ["aquacorr:c2", "wqp:c0"], f"{CX.META_FILE}:chip_id: 2 entries")):
        with pytest.raises(DM.DevmError, match=msg):
            DM.audit_artefacts(d, rows, chips)
    assert DM.full_chip_ids(["wqp", "aquacorr"], ["a", "b"]).tolist() == ["wqp:a", "aquacorr:b"]


def test_audit_reads_every_file_type_or_refuses(tmp_path):
    d = tmp_path / "c"
    _write(d)
    info = json.loads((d / DM.INFO).read_text())
    (d / "extra.json").write_text(json.dumps({"note": "see chip wqp:secret and row", "ids": ["rowsecret"]}))
    with pytest.raises(DM.DevmError, match="extra.json: 1 tokens"):
        DM.audit_artefacts(d, [], ["wqp:secret"])
    with pytest.raises(DM.DevmError, match="extra.json: 1 tokens"):
        DM.audit_artefacts(d, ["rowsecret"], [])
    (d / "extra.json").unlink()
    np.savez(d / "ids.npz", ids=np.array(["x", "rowsecret"]), v=np.zeros(2))
    with pytest.raises(DM.DevmError, match="ids.npz:ids: 1 entries"):
        DM.audit_artefacts(d, ["rowsecret"], [])
    (d / "ids.npz").unlink()
    (d / "sub").mkdir()
    pd.DataFrame({"anything": ["a", "wqp:secret"]}).to_parquet(d / "sub" / "t.parquet")
    with pytest.raises(DM.DevmError, match="t.parquet:anything: 1 entries"):
        DM.audit_artefacts(d, [], ["wqp:secret"])
    (d / "sub" / "t.parquet").unlink()
    (d / "sub" / "notes.csv").write_text("row_id\nrowsecret\n")
    with pytest.raises(DM.DevmError, match="not an auditable artefact type"):
        DM.audit_artefacts(d, [], [])
    (d / "sub" / "notes.csv").unlink()
    assert DM.audit_artefacts(d, ["rowsecret"], ["wqp:secret"])["files"] == 7 and info["version"] == DM.VERSION


def _builder():
    """The build script, loaded by path and not run: the test re-derives the key sets with the build's own code."""
    f = Path(__file__).resolve().parents[1] / "scripts" / "data" / "orbitv2_build_dev_v5m.py"
    spec = importlib.util.spec_from_file_location("orbitv2_build_dev_v5m_under_test", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_cache_holds_no_protected_id():
    """R-21 on the real cache, skipped before the first build: the protected row and chip key sets are recomputed
    from the keys with the builder's own functions (no label value is read, and an empty set — which the audit would
    pass vacuously — fails the test), every written artefact is audited against them, and the loader's own
    re-derivation passes."""
    if not (DM.cache_dir() / DM.INFO).exists():
        pytest.skip("dev_v5m cache not built")
    B = _builder()
    built = json.loads((DM.cache_dir() / DM.INFO).read_text())["build_counts"]
    prot = B.aquacorr_protected_keys()
    if built.get("added_rows"):                      # the build added WQP rows: their protected keys count as well
        base, _, _ = D.load_dev(built["base_version"])
        prot = pd.concat([prot, B.added_rows(base, None)[1]], ignore_index=True)
    cand = DM.store_candidates(log=lambda *a: None)
    prot_row_ids = set(prot["row_id"])
    prot_chip_ids = set(cand.chip_id[cand.protected.to_numpy(bool)]) | set(
        DM.full_chip_ids(prot.chip_store, prot.chip_id))
    assert len(prot_row_ids) >= 100 and len(prot_chip_ids) >= 100 and len(prot) == built["protected_key_rows"]
    n = DM.audit_artefacts(DM.cache_dir(), prot_row_ids, prot_chip_ids)
    assert n["files"] >= 7 and n["string_columns"] >= 6 and n["json_tokens"] > 20
    assert n["n_protected_row_ids"] == len(prot_row_ids) and n["n_protected_chip_ids"] == len(prot_chip_ids)
    df, _, _, (cmeta, _), info = DM.load()
    assert not df["protected"].any() and set(df["role"]) <= {"train", "val"}
    assert set(cmeta["role"]) <= {"train", "val"} and info["context_stores"] == ["aquacorr", "wqp"]
    assert not set(df["row_id"]) & prot_row_ids
    assert not set(DM.full_chip_ids(df["chip_store"], df["chip_id"])) & prot_chip_ids
    assert not set(DM.full_chip_ids(cmeta["store"], cmeta["store_chip_id"])) & prot_chip_ids


# ------------------------------------------------------------------------------------ candidates and chips

def _index(n=5):
    idx = pd.DataFrame({
        "chip_id": [f"c{i}" for i in range(n)], "shard": "s0.npz", "item_id": [f"S2A_{i}" for i in range(n)],
        "lon": 10.0, "lat": 50.0, "sensing_time_utc": [f"2020-06-{10 + i:02d}T10:30:00Z" for i in range(n)],
        "aot_c3": [0.1, 0.2, np.nan, 0.05, 0.3], "sun_zenith": 30.0, "view_zenith": 5.0, "sun_azimuth": 150.0,
        "view_azimuth": 100.0, "processing_baseline": "04.00", "platform": "S2A", "scl_c3_n_water": [9, 9, 9, 3, 9], "jrc_water_ok": True,
        "offset_suspect": "", "scl_c3_cloud_medium": [0, 1, 0, 0, 0], "scl_c3_cloud_high": 0, "scl_c3_thin_cirrus": 0,
        "scl_c3_cloud_shadow": 0, **{c: 0.03 for c in DM.RW3_COLS}})
    idx["rw3_B11"] = [0.01, 0.01, 0.01, 0.01, 0.2]
    sp = pd.DataFrame({"waterbody": "L1", "in_lake": True, "block": "b1", "role": "train", "tblock": "train",
                       "buffer_excluded": False, "protected": False}, index=range(n))
    return idx, sp


def test_context_candidates_from_image_metadata_only():
    idx, sp = _index()
    cand = DM.context_candidates(idx, "wqp", sp)
    assert set(CX.CAND_COLS) <= set(cand.columns)
    assert cand["chip_id"].tolist() == [f"wqp:c{i}" for i in range(5)]
    assert cand["qc_ok"].tolist() == [True, False, True, False, False]     # cloud, too few water pixels, bright SWIR
    assert cand["date"].iloc[0] == pd.Timestamp("2020-06-10") and np.isnan(cand["aot"].iloc[2])
    assert cand["platform"].tolist() == ["S2A"] * 5
    assert cand[list(DM.RW3_COLS)].shape == (5, 12) and cand["rw3_B11"].iloc[4] == np.float32(0.2)
    with pytest.raises(DM.DevmError, match="3 x 3 water-median"):
        DM.context_candidates(idx.drop(columns="rw3_B05"), "wqp", sp)
    CX.assert_no_labels(cand)
    row = pd.DataFrame({"waterbody": ["L1"], "in_lake": [True], "block": ["b1"], "role": ["train"], "lon": [10.0],
                        "lat": [50.0], "date": pd.to_datetime(["2020-06-01"])})
    cidx, _, _ = CX.select(row, cand)
    assert [cand["chip_id"].iloc[j] for j in cidx[0] if j >= 0] == ["wqp:c0", "wqp:c2"]   # unknown AOT last
    with pytest.raises(CX.ContextError, match="label columns"):
        DM.context_candidates(idx.assign(chla=1.0), "wqp", sp)
    for store in ("naip_teacher", "mc_benchmark", "new_store"):
        with pytest.raises(DM.DevmError, match="not a context store"):
            DM.context_candidates(idx, store, sp)
    with pytest.raises(DM.DevmError, match="differ in length"):
        DM.context_candidates(idx, "wqp", sp.iloc[:3])


def test_candidates_come_from_the_two_context_stores_only(monkeypatch):
    """R-7 / C-2: ``store_candidates`` reads s2_chips/aquacorr and s2_chips/wqp, has no store argument."""
    import inspect

    assert DM.CONTEXT_STORES == ("aquacorr", "wqp") and DM.CONTEXT_STORES is CX.STORES
    assert "stores" not in inspect.signature(DM.store_candidates).parameters
    asked = []
    idx, sp = _index()

    def index(store):
        asked.append(store)
        return idx

    monkeypatch.setattr(DM.T, "lakes_readonly", lambda lon, lat, log=print: None)
    monkeypatch.setattr(DM.SP, "assign", lambda lon, lat, dates, lakes=None, log=print: sp)
    cand = DM.store_candidates(log=lambda *a: None, index=index)
    assert asked == ["aquacorr", "wqp"] and len(cand) == 10
    assert set(cand["chip_id"].str.split(":").str[0]) == {"aquacorr", "wqp"}


def test_context_chip_arrays_use_the_v1_pixel_rule_with_8_pixels():
    layers = list(C.S2_STORE_LAYERS)
    water = np.zeros((64, 64), bool)
    water[:, 30:] = True
    chip = np.zeros((len(layers), 64, 64), np.int16)
    for i, name in enumerate(layers):
        if name == "SCL":
            chip[i] = np.where(water, CF.SCL_WATER, 4)
        elif name in D.BANDS:
            chip[i] = np.where(water, 200, 500)
    refl, off, n, dp = DM.context_chip_arrays(chip, layers)
    r64, o64, n64, _ = D.chip_pixels(chip, layers)
    assert refl.shape == (8, 12) and off.shape == (8, 2) and n == 8 and dp.shape == (8,)
    assert np.array_equal(refl, r64[:8]) and np.array_equal(off, o64[:8]) and n64 > 8
    # d_p: distance of each chosen pixel to the shore (column 29 is the nearest non-water column)
    assert np.allclose(dp, (32 + off[:, 1] - 29) * CF.PIXEL_M)
