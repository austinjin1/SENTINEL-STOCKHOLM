# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing of the M7 v2 runs — synthetic data only (no real row, flag table or registry row is loaded)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models.lite import runner as R
from sentinel.models.litev2 import harness as H
from sentinel.models.litev2 import labels2b as L
from sentinel.models.litev2 import rows as RW
from sentinel.models.litev2 import stamp
from sentinel.models.litev2 import tabular as T
from sentinel.models.litev2 import tests_mask as TM


def _rows(n=60, role="val", track="national", seed=0):
    rng = np.random.default_rng(seed)
    d = {c: [None] * n for c in RW.KEY_COLS}
    df = pd.DataFrame(d)
    df["naip_chip_id"] = [f"c{i:03d}" for i in range(n)]
    df["uid"] = df["naip_chip_id"] if track == "national" else RW.P2_PREFIX + df["naip_chip_id"]
    df["track"], df["role"] = track, role
    df["water_body"] = [f"wb{i % 15}" for i in range(n)]
    df["site_id"] = [f"s{i % 20}" for i in range(n)]
    df["site_class"] = np.where(np.arange(n) % 3 == 0, "river_stream", "lake_reservoir")
    df["naip_year"] = 2021
    for c in (*RW.Y_COLS, *RW.LIM_COLS):
        df[c] = np.nan
    for c in RW.CENS_COLS:
        df[c] = 0
    for t in L.TARGETS:
        df[f"y_{t}"] = rng.normal(size=n)
    df.loc[::4, "y_tss"] = np.nan
    return df


def test_names_and_required_comparators():
    assert len(H.COMPARATORS) == len(set(H.COMPARATORS)) == 25 and len(H.DEEP) == 4
    nat, p2 = H.required_comparators("national_val"), H.required_comparators("val_p2")
    assert set(nat) == set(H.COMPARATORS)
    assert nat["S9_gbm_parity_aft"] == ["secchi", "tss", "chla"] and "S9_gbm_parity_aft" not in p2
    assert nat["P1_kloiber_log"] == ["secchi"] and nat["P1_oc_ndvi"] == ["chla"]
    assert all(ts == ["turb_fnu"] for ts in p2.values())
    assert {c for c in p2 if c.startswith("P1_")} == {"P1_red_nir_log", "P1_red_nir_log_water", "P1_red_green_log_water"}
    assert not any(t in ("turb_fnu", "ssc") for ts in nat.values() for t in ts)   # descriptive on national val


def _registry(block, drop=()):
    return [{"workstream": R.WORKSTREAM, "experiment": H.EXPERIMENT[block], "split": "val",
             "model": H.MODEL_PREFIX + c, "target": t}
            for c, ts in H.required_comparators(block).items() for t in ts if f"{c}/{t}" not in drop]


def test_gate_refuses_until_every_comparator_row_exists():
    for block in H.EXPERIMENT:
        with pytest.raises(H.GateError):
            H.require_comparators(block, [])
        H.require_comparators(block, _registry(block))
    miss = H.missing_comparators("national_val", _registry("national_val", drop=("S10_tabpfn/chla",)))
    assert miss == ["S10_tabpfn/chla"]
    with pytest.raises(H.GateError):
        H.require_comparators("national_val", _registry("national_val", drop=("P3b_densenet_parity_lr3e-4/tss",)))
    wrong = [dict(r, workstream="M2-orbit") for r in _registry("val_p2")]            # another workstream's rows
    assert len(H.missing_comparators("val_p2", wrong)) == len(_registry("val_p2"))
    other = [dict(r, experiment=H.EXPERIMENT["national_val"]) for r in _registry("val_p2")]   # the other block
    assert len(H.missing_comparators("val_p2", other)) == len(_registry("val_p2"))
    notval = [dict(r, split="test") for r in _registry("val_p2")]
    assert len(H.missing_comparators("val_p2", notval)) == len(_registry("val_p2"))


def test_score_block_populations_and_descriptive_flags():
    rows = _rows(90)
    preds = {t: rows[f"y_{t}"].fillna(0.0).to_numpy() + 0.1 for t in L.TARGETS}
    sc = H.score_block("national_val", rows, preds)
    assert set(sc) == set(L.TARGETS)
    lake = (rows["site_class"] == "lake_reservoir").to_numpy()
    assert sc["secchi"]["population"] == "lake" and sc["secchi"]["n"] == int(lake.sum())
    assert sc["turbidity"]["population"] == "all" and sc["turbidity"]["n"] == len(rows)
    assert sc["tss"]["n"] == int(rows["y_tss"].notna().sum())
    assert sc["turb_fnu"]["descriptive_only"] and sc["ssc"]["descriptive_only"] and not sc["tss"]["descriptive_only"]
    assert sc["turb_fnu"]["population"] == "all" and sc["turb_fnu"]["primary"] == sc["turbidity"]["primary"]
    assert all(s["interpretable_n"] for s in sc.values())
    p2 = _rows(8, role="val_p2", track="p2")
    s2 = H.score_block("val_p2", p2, {"turb_fnu": p2["y_turb_fnu"].to_numpy() + 0.2, "secchi": np.zeros(8)})
    assert set(s2) == {"turb_fnu"} and not s2["turb_fnu"]["interpretable_n"] and not s2["turb_fnu"]["descriptive_only"]
    rivers = _rows(30)
    rivers["site_class"] = "river_stream"                      # no lake row: Secchi has no primary population
    assert "secchi" not in H.score_block("national_val", rivers, {"secchi": np.zeros(30), "tss": np.zeros(30)})
    none = _rows(12)
    none["y_chla"] = np.nan
    assert H.score_block("national_val", none, {"chla": np.zeros(12)}) == {}


def _flags(ids, masked):
    return pd.DataFrame({"naip_chip_id": list(ids), "kd_masked": [i in masked for i in ids]})


def test_kd_mask_never_looks_up_a_p2_chip_in_the_national_table():
    nat = _rows(6, role="train")
    p2 = _rows(4, role="train_p2", track="p2")                  # same chip-id strings as the national rows
    rows = pd.concat([nat, p2], ignore_index=True)
    flags = _flags(nat["naip_chip_id"], {"c001", "c003"})
    m = H.kd_masked(rows, flags)
    assert m.tolist() == [False, True, False, True, False, False] + [False] * 4
    with pytest.raises(TM.MaskError):
        H.kd_masked(rows, _flags(["c000", "c001"], set()))      # a national chip without a flag


def test_teacher_means_mask_and_training_rows_only():
    nat = _rows(6, role="train")
    p2 = _rows(2, role="train_p2", track="p2")
    rows = pd.concat([nat, p2], ignore_index=True)
    flags = _flags(nat["naip_chip_id"], {"c001", "c003"})
    n, k = len(rows), 5
    t_w = np.ones((n, k, 4), np.float32)
    t_w[6:] = 0.0                                               # P2 rows: no pairing
    kd = {"t_mu": np.full((n, k, 4), 0.5, np.float32), "t_w": t_w}
    tm, info = H.teacher_means(rows, None, flags, kd)
    assert list(tm.columns) == list(T.TEACHER_COLS)
    assert tm.iloc[[1, 3]].isna().all().all() and tm.iloc[[0, 2, 4, 5]].notna().all().all()
    assert tm.iloc[6:].isna().all().all()
    assert info["n_kd_masked"] == 2 and all(v == 2 for v in info["n_teacher_values_removed"].values())
    for role in ("val", "val_n", "val_p2", "test"):
        bad = rows.copy()
        bad.loc[0, "role"] = role
        with pytest.raises(TM.MaskError):
            H.teacher_means(bad, None, flags, kd)
    leak = dict(kd, t_w=np.ones((n, k, 4), np.float32))         # a teacher value on a P2 row is refused
    with pytest.raises(TM.MaskError):
        H.teacher_means(rows, None, flags, leak)


def test_save_predictions_keys_on_uid(tmp_path):
    rows = pd.concat([_rows(3), _rows(3, role="val_p2", track="p2")], ignore_index=True)
    H.save_predictions(tmp_path / "p.parquet", rows, {"turb_fnu": np.arange(6.0)}, {"turb_fnu": np.ones(6)})
    got = pd.read_parquet(tmp_path / "p.parquet")
    assert got["uid"].is_unique and got["naip_chip_id"].duplicated().any()
    assert got["p_turb_fnu"].tolist() == list(np.arange(6.0)) and "s_turb_fnu" in got


def test_registry_rows_need_the_stamp(monkeypatch):
    def refuse(what):
        raise stamp.StampError(what)

    monkeypatch.setattr(stamp, "require_stamp", refuse)
    called = []
    monkeypatch.setattr(R, "log_scores", lambda *a, **k: called.append(k) or [])
    with pytest.raises(stamp.StampError):
        H.log_block("national_val", "S6_gbm_parity", {}, cfg={}, artifacts=[], is_baseline=True)
    assert not called
    monkeypatch.setattr(stamp, "require_stamp", lambda what: None)
    H.log_block("val_p2", "S6_gbm_parity", {}, cfg={}, artifacts=[], is_baseline=True)
    assert called[0]["model"] == "litev2_S6_gbm_parity" and called[0]["experiment"] == "litev2_val_p2"
    assert called[0]["split"] == "val" and called[0]["extra"]["block"] == "val_p2"
