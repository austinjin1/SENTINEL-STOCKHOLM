# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""The §4.2 anchor is one artefact (M7_verify ruling Q5 (iii); lead order of 2026-11-10 20:20 EST, A3 (iii)): the
evaluation anchors and comparator S6 are predictions of one registered file with one sha256. Synthetic rows in a
temporary directory; no real row or registry row is read."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.models.litev2 import anchor_store as AS
from sentinel.models.litev2 import checks as C
from sentinel.models.litev2 import labels2b as L
from sentinel.models.litev2 import tabular as T

FAST = (0,)


def _synthetic(n=300, n_wb=40, seed=0):
    rng = np.random.default_rng(seed)
    cols = T.parity_columns()
    X = pd.DataFrame(rng.uniform(0.1, 1.0, size=(n, len(cols))), columns=cols)
    sig = 2.0 * X["c9s_R"].to_numpy() - X["c9s_B"].to_numpy()
    rows = pd.DataFrame({f"y_{q}": np.nan for q in T.QUANTITIES}, index=X.index)
    rows["y_secchi"] = -sig + 0.05 * rng.normal(size=n)
    unit = rng.integers(0, 3, n)
    for k, u in enumerate(("turbidity", "turb_fnu", "turb_ntru")):
        rows.loc[unit == k, f"y_{u}"] = (sig + 0.5 * (k - 1) + 0.05 * rng.normal(size=n))[unit == k]
    rows.loc[::2, "y_tss"] = (sig + 0.05 * rng.normal(size=n))[::2]
    rows.loc[::5, "y_ssc"] = (sig + 0.05 * rng.normal(size=n))[::5]
    rows.loc[::3, "y_chla"] = (X["c9s_G"].to_numpy() + 0.05 * rng.normal(size=n))[::3]
    rows["uid"] = [f"c{i:04d}" for i in range(n)]
    rows["water_body"] = [f"wb{int(i)}" for i in rng.integers(0, n_wb, n)]
    rows["track"], rows["role"] = "national", "train"
    return X, rows


def _keys(uid="v0", wb="held"):
    return pd.DataFrame({"uid": [uid], "track": ["national"], "role": ["val"], "water_body": [wb]})


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("anchors")
    X, rows = _synthetic()
    tr, Xtr, ev, Xev = rows.iloc[:240].reset_index(drop=True), X.iloc[:240].reset_index(drop=True), \
        rows.iloc[240:].reset_index(drop=True), X.iloc[240:].reset_index(drop=True)
    logged = []
    for bad in (_keys(uid="c0003"), _keys(wb=tr["water_body"].iloc[0])):     # Q5 (i) is checked before the fit
        with pytest.raises(C.CheckError):
            AS.build(tr, Xtr, keys=bad, root=root, registry=[], log_row=lambda **kw: logged.append(kw), seeds=FAST)
    assert logged == [] and not (root / AS.MODEL_FILE).exists()
    anc = AS.build(tr, Xtr, keys=_keys(), root=root, registry=[], log_row=lambda **kw: logged.append(kw), seeds=FAST)
    assert len(logged) == 1
    reg = [{"workstream": logged[0]["workstream"], "experiment": logged[0]["experiment"], "extra": logged[0]["extra"]}]
    return {"root": root, "tr": tr, "Xtr": Xtr, "ev": ev, "Xev": Xev, "anc": anc, "reg": reg, "row": logged[0]}


def _sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def test_one_registered_artefact(store):
    root, reg, row = store["root"], store["reg"], store["row"]
    ex = row["extra"]
    assert row["experiment"] == AS.EXPERIMENT and row["split"] == "train" and row["n"] == 240
    assert ex["anchor_sha256"] == ex["sha256"][AS.MODEL_FILE] == _sha(root / AS.MODEL_FILE)
    assert ex["sha256"][AS.TRAIN_FILE] == _sha(root / AS.TRAIN_FILE)
    assert ex["refit_table"]["n_chip"] == 0 and ex["refit_table"]["n_water_body"] == 0
    assert AS.registered(reg)["anchor_sha256"] == ex["anchor_sha256"] and AS.registered([]) == {}
    with pytest.raises(AS.AnchorStoreError, match="one artefact"):
        AS.registered(reg + reg)
    for registry in (reg, []):                                               # never rebuilt, registered or not
        with pytest.raises(AS.AnchorStoreError, match="never rebuilt"):
            AS.build(store["tr"], store["Xtr"], keys=_keys(), root=root, registry=registry,
                     log_row=lambda **kw: pytest.fail("second row"), seeds=FAST)
    with pytest.raises(AS.AnchorStoreError, match="not registered"):
        AS.load_model([], root)
    fs = AS.first_stage(store["tr"], reg, root)
    assert fs["info"]["anchor_sha256"] == ex["anchor_sha256"]
    assert np.allclose(fs["train"].to_numpy(), store["anc"]["train"][list(T.QUANTITIES)].to_numpy(), equal_nan=True)
    assert np.allclose(AS.load_table("train", store["tr"]["uid"], reg, root)[list(T.QUANTITIES)].to_numpy(),
                       fs["train"].to_numpy(), equal_nan=True)               # training rows: out-of-fold anchors
    with pytest.raises(AS.AnchorStoreError, match="differ from the rows"):
        AS.first_stage(store["tr"].iloc[:-1], reg, root)
    with pytest.raises(AS.AnchorStoreError, match="out-of-fold"):
        AS.eval_anchors("train", store["tr"]["uid"], store["Xtr"], reg, root)


def test_evaluation_anchor_and_s6_are_the_same_numbers_of_one_file(store, monkeypatch):
    root, reg, ev, Xev = store["root"], store["reg"], store["ev"], store["Xev"]
    table = AS.eval_anchors("national_val", ev["uid"], Xev, reg, root)
    f = root / "anchors_national_val.parquet"
    meta = json.loads((root / AS.META_FILE).read_text())
    assert meta["tables"]["national_val"]["sha256"] == _sha(f) == AS.table_sha256("national_val", root)
    assert meta["tables"]["national_val"]["anchor_sha256"] == AS.registered(reg)["anchor_sha256"] == meta["anchor_sha256"]
    direct = T.predict_parity(store["anc"]["model"], Xev)                    # the in-memory model of the build
    assert np.array_equal(table[list(T.QUANTITIES)].to_numpy(), direct[list(T.QUANTITIES)].to_numpy())
    s6 = AS.s6_predictions(table)                                            # comparator S6
    anchor = AS.load_table("national_val", ev["uid"], reg, root)             # what a v2 network reads as its anchor
    assert set(s6) == set(L.TARGETS)
    for t in L.TARGETS:
        assert np.array_equal(s6[t], anchor[t].to_numpy(float))              # the same numbers, bit for bit

    def boom(*a, **k):
        raise AssertionError("the block was predicted a second time")

    monkeypatch.setattr(T, "predict_parity", boom)
    again = AS.eval_anchors("national_val", ev["uid"], Xev, reg, root)       # read back, not predicted again
    assert np.array_equal(again[list(T.QUANTITIES)].to_numpy(), table[list(T.QUANTITIES)].to_numpy())
    shuffled = ev["uid"].iloc[::-1].reset_index(drop=True)
    back = AS.load_table("national_val", shuffled, reg, root)
    assert back["uid"].tolist() == shuffled.tolist()
    assert np.array_equal(back["secchi"].to_numpy(), table["secchi"].to_numpy()[::-1])
    for uid in (ev["uid"].iloc[:-1], pd.concat([ev["uid"].iloc[:-1], pd.Series(["zz"])])):
        with pytest.raises(AS.AnchorStoreError, match="exactly the rows"):
            AS.load_table("national_val", uid, reg, root)
    with pytest.raises(AS.AnchorStoreError, match="is missing"):
        AS.load_table("val_p2", ev["uid"], reg, root)


def test_scripts_take_anchor_and_s6_from_the_store_only():
    comp = (paths.REPO_ROOT / "scripts/train/litev2_comparators.py").read_text()
    assert 'preds["S6_gbm_parity"] = AS.s6_predictions(a_ev)' in comp and "AS.eval_anchors(" in comp
    assert comp.count('preds["S6_gbm_parity"]') == 1
    train = (paths.REPO_ROOT / "scripts/train/litev2_train.py").read_text()
    assert "AS.load_table(" in train and "anchors_national_val" not in train
    assert train.count("T.predict_parity(") == 1                             # ablation A4 only: its own reduced
    assert 'T.predict_parity(a4_model(run_dir), fev["X"])' in train          # anchor, hash-checked in its run dir


def test_a_changed_file_is_refused(store):
    """Runs last in this module: it overwrites the artefact of the module fixture."""
    root, reg, ev = store["root"], store["reg"], store["ev"]
    AS.eval_anchors("national_val", ev["uid"], store["Xev"], reg, root)
    f = root / "anchors_national_val.parquet"
    keep = f.read_bytes()
    pd.read_parquet(f).assign(secchi=0.0).to_parquet(f, index=False)         # a second S6 / anchor table
    for call in (lambda: AS.load_table("national_val", ev["uid"], reg, root),
                 lambda: AS.eval_anchors("national_val", ev["uid"], store["Xev"], reg, root)):
        with pytest.raises(AS.AnchorStoreError, match="not the table written from the registered"):
            call()
    f.write_bytes(keep)
    assert len(AS.load_table("national_val", ev["uid"], reg, root)) == len(ev)
    f.unlink()
    with pytest.raises(AS.AnchorStoreError, match="written before and is missing"):
        AS.eval_anchors("national_val", ev["uid"], store["Xev"], reg, root)  # not silently predicted again
    f.write_bytes(keep)
    other = [dict(reg[0], extra={**reg[0]["extra"], "anchor_sha256": "0" * 64})]
    with pytest.raises(AS.AnchorStoreError):
        AS.load_table("national_val", ev["uid"], other, root)                # a table of another anchor model
    m = root / AS.MODEL_FILE
    m.write_bytes(m.read_bytes() + b"\0")                                    # a second model under the same name
    for call in (lambda: AS.load_model(reg, root), lambda: AS.first_stage(store["tr"], reg, root),
                 lambda: AS.load_table("national_val", ev["uid"], reg, root),
                 lambda: AS.eval_anchors("val_p2", ev["uid"], store["Xev"], reg, root)):
        with pytest.raises(AS.AnchorStoreError, match="differs from its registered sha256"):
            call()
    assert not (root / "anchors_val_p2.parquet").exists()


def test_a4_anchor_model_is_hash_checked_in_its_run_directory(tmp_path):
    import importlib.util
    import pickle

    spec = importlib.util.spec_from_file_location("litev2_train_a4m", paths.REPO_ROOT / "scripts/train/litev2_train.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    f = tmp_path / "a4_anchor_model.pkl"
    f.write_bytes(pickle.dumps({"k": 1}))
    (tmp_path / tr.A4_SHA_FILE).write_text(json.dumps({"a4_anchor_model.pkl": _sha(f)}))
    assert tr.a4_model(tmp_path) == {"k": 1}
    f.write_bytes(pickle.dumps({"k": 2}))
    with pytest.raises(RuntimeError, match="differs from the sha256"):
        tr.a4_model(tmp_path)
