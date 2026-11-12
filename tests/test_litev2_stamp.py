# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the stamp guard, the national-2b loader, the P2 index loader (F7, exclusions) and the teacher scoring
block of SENTINEL-Lite architecture round v2. Synthetic rows only (RESEARCH_PLAN R9); the pin files on disk are
read for their role mappings and hashes only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.models.lite.guard import ProtectedDataError
from sentinel.models.litev2 import labels2b as L
from sentinel.models.litev2 import p2_index as PI
from sentinel.models.litev2 import stamp as S
from sentinel.models.litev2 import tests_mask as TM


# --------------------------------------------------------------------------- stamp guard (F9)
def _repo(tmp_path, monkeypatch, *, read_only=True, skip=()):
    """A synthetic repo root with the four files and an empty registry; the amendment pin follows the synthetic file."""
    shas = {}
    for i, rel in enumerate(S.STAMPED_FILES):
        if rel in skip:
            continue
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"synthetic file {i}\n")
        shas[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        if read_only:
            os.chmod(f, 0o444)
    if S.AMENDMENT in shas:
        monkeypatch.setattr(S, "AMENDMENT_SHA256", shas[S.AMENDMENT])
    return tmp_path, tmp_path / "registry.jsonl", shas


def _row(reg: Path, shas, *, workstream=S.WORKSTREAM, written_by=S.WRITTEN_BY, experiment=S.STAMP_EXPERIMENT, rid="r1"):
    with reg.open("a") as fh:
        fh.write(json.dumps({"id": rid, "timestamp": "2026-11-10T00:00:00+00:00", "workstream": workstream,
                             "experiment": experiment, "extra": {"written_by": written_by, "sha256": shas}}) + "\n")


def test_guard_is_closed_without_the_four_files_or_a_lead_row(tmp_path, monkeypatch):
    root, reg, shas = _repo(tmp_path, monkeypatch, skip=(S.ADDENDUM, S.DEVIATIONS, S.RULINGS))
    st = S.stamp_status(reg, root)
    assert not st["stamped"] and "3 of the four files do not exist" in st["reason"]
    root2 = tmp_path / "b"
    root2.mkdir()
    root2, reg2, shas2 = _repo(root2, monkeypatch)
    st = S.stamp_status(reg2, root2)                                   # files there, no registry at all
    assert not st["stamped"] and st["n_rows"] == 0 and "no lite/prereg_snapshot row" in st["reason"]
    with pytest.raises(S.StampError, match="refused"):
        S.require_stamp("val rows", reg2, root2)
    with pytest.raises(S.StampError):
        S.guard_roles(["train", "val_n"], "a v2 arm", reg2, root2)
    S.guard_roles(["train", "train_p2"], "a v2 arm", reg2, root2)     # training roles pass before the stamp


def test_guard_opens_only_for_one_lead_row_with_all_four_current_hashes(tmp_path, monkeypatch):
    root, reg, shas = _repo(tmp_path, monkeypatch)
    _row(reg, shas, written_by="M7", rid="a")                          # the workstream cannot stamp itself
    _row(reg, shas, workstream="lite", rid="b")                        # wrong workstream name
    _row(reg, shas, experiment="lite/prereg_snapshot_draft", rid="c")  # another experiment
    _row(reg, {k: v for k, v in shas.items() if k != S.RULINGS}, rid="d")          # three files only
    _row(reg, {**shas, S.ADDENDUM: "0" * 64}, rid="e")                 # a stale hash
    st = S.stamp_status(reg, root)
    assert not st["stamped"] and st["n_rows"] == 3 and st["n_lead_rows"] == 2
    _row(reg, shas, rid="good")
    st = S.stamp_status(reg, root)
    assert st["stamped"] and st["row_id"] == "good" and st["reason"] == "stamped"
    assert S.require_stamp("val rows", reg, root)["row_id"] == "good"
    S.guard_roles(["val", "val_n", "val_p2"], "a v2 arm", reg, root)


def test_guard_closes_on_a_writable_file_an_edit_or_a_changed_amendment(tmp_path, monkeypatch):
    root, reg, shas = _repo(tmp_path, monkeypatch)
    _row(reg, shas)
    assert S.is_stamped(reg, root)
    dev = root / S.DEVIATIONS
    os.chmod(dev, 0o644)
    assert "not read-only" in S.stamp_status(reg, root)["reason"]
    dev.write_text(dev.read_text() + "entry 2\n")                      # an append after the stamp
    os.chmod(dev, 0o444)
    st = S.stamp_status(reg, root)
    assert not st["stamped"] and "current hashes" in st["reason"]
    _row(reg, {**shas, S.DEVIATIONS: hashlib.sha256(dev.read_bytes()).hexdigest()}, rid="r2")
    assert S.stamp_status(reg, root)["row_id"] == "r2"                 # the lead stamps the new content
    monkeypatch.setattr(S, "AMENDMENT_SHA256", "f" * 64)               # the amendment is not the reviewed v0.3
    assert "differs from the reviewed" in S.stamp_status(reg, root)["reason"]


def test_guard_constants_are_the_ruled_ones():
    assert S.WORKSTREAM == "M7-lite" and S.STAMP_EXPERIMENT == "lite/prereg_snapshot" and S.WRITTEN_BY == "lead"
    assert S.AMENDMENT_SHA256 == "2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56"
    assert set(S.VAL_ROLES) == {"val", "val_n", "val_p2"}
    assert [Path(f).name for f in S.STAMPED_FILES] == [
        "lite_arch_v2_amendment.md", "lite_arch_v2_label_addendum.md", "lite_arch_v2_deviations.md",
        "lite_arch_v2_lead_rulings.md"]


# --------------------------------------------------------------------------- national-2b loader
def _labels():
    n = 8
    lab = pd.DataFrame({"naip_chip_id": [f"c{i}" for i in range(n)],
                        "site_id": ["A", "A", "B", "RES-1", "C", "D", "E", "F"],
                        "role": ["train", "train", "train", "train", "val", "val_n", "test_n", "excluded"],
                        "version_source": ["national_pm7", "added", "frozen", "national", "frozen", "national",
                                           "national", "added"]})
    for c in L.Y_COLS:
        lab[c] = np.nan
    for c in (*L.CENS_COLS,):
        lab[c] = 0
    for c in (*L.LIM_COLS,):
        lab[c] = np.nan
    for t in ("turb_fnu", "ssc", "turb_ntru"):
        lab[f"n_{t}"] = 0
    lab["label_drop_reason"], lab["national2b_wb"] = "", "w"
    lab.loc[[0, 3, 4, 5, 6], "y_secchi"] = 0.1                         # original-target labels
    lab.loc[1, "y_turb_fnu"] = 0.5                                     # an added chip: recovered label only
    lab.loc[2, "cens_tss"], lab.loc[2, "lim_tss"] = -1, 0.0            # the frozen chip kept by a censored row only
    return lab


@pytest.fixture()
def synth_labels(monkeypatch):
    monkeypatch.setattr(L, "load_labels", lambda split: _labels())
    monkeypatch.setattr(L, "exclude_reserved", lambda ids: [i for i in ids if not str(i).startswith("RES")])
    monkeypatch.setattr(L, "assert_no_reserved", lambda ids, where="": None)
    return {"chip_role": {}}


def test_role_frame_never_returns_a_test_role_and_needs_the_stamp_for_validation(synth_labels, monkeypatch):
    def closed(what, *a, **k):
        raise S.StampError(what)

    monkeypatch.setattr(S, "require_stamp", closed)
    for r in L.TEST_ROLES:
        with pytest.raises(ProtectedDataError):
            L.role_frame(synth_labels, ["train", r])
    for r in L.VAL_ROLES:
        with pytest.raises(S.StampError):
            L.role_frame(synth_labels, ["train", r])
        with pytest.raises(S.StampError):
            L.role_frame(synth_labels, [r], view="rerun")
    assert set(L.role_frame(synth_labels, ["train"])["role"]) == {"train"}
    monkeypatch.setattr(S, "require_stamp", lambda what, *a, **k: {"stamped": True})
    assert sorted(L.role_frame(synth_labels, ["val", "val_n"])["naip_chip_id"]) == ["c4", "c5"]


def test_role_frame_views_and_the_reserved_block(synth_labels):
    v2 = L.role_frame(synth_labels, ["train"])
    assert sorted(v2["naip_chip_id"]) == ["c0", "c1", "c2"]            # c3 sits at a reserved site: never returned
    rr = L.role_frame(synth_labels, ["train"], view="rerun")
    assert list(rr["naip_chip_id"]) == ["c0"]                          # no added chip, no censored-only chip
    assert not [c for c in rr.columns if "turb_fnu" in c or "ssc" in c or c.startswith(("cens_", "lim_"))]
    with pytest.raises(ValueError):
        L.role_frame(synth_labels, ["train"], view="other")


def test_rerun_activation_goes_through_the_guarded_loader(synth_labels, monkeypatch):
    from sentinel.models.lite import national as N

    for name in ("load_split", "role_frame", "labels_path", "national_scene_stats_path", "train_val_frames"):
        monkeypatch.setattr(N, name, getattr(N, name))                 # restored after the test
    monkeypatch.setattr(L, "load_split", lambda *a, **k: synth_labels)
    monkeypatch.setattr(S, "require_stamp", lambda what, *a, **k: (_ for _ in ()).throw(S.StampError(what)))
    L.activate_rerun()
    with pytest.raises(S.StampError):
        N.role_frame(synth_labels, ("train", "val", "val_n"))
    assert list(N.role_frame(synth_labels, ("train",))["naip_chip_id"]) == ["c0"]


def test_roles_unchanged_is_checked_on_the_role_mapping_alone():
    s2 = {"chip_role": {"a": "train", "b": "val", "c": "test_n"}}
    assert L.assert_roles_unchanged({"chip_role": {**s2["chip_role"], "x": "train", "y": "excluded"}}, s2) == 3
    with pytest.raises(L.LabelVersionError, match="changed role"):
        L.assert_roles_unchanged({"chip_role": {"a": "train", "b": "train", "c": "test_n"}}, s2)
    with pytest.raises(L.LabelVersionError, match="no national-2b role"):
        L.assert_roles_unchanged({"chip_role": {"a": "train", "b": "val"}}, s2)
    with pytest.raises(L.LabelVersionError, match="added chips"):
        L.assert_roles_unchanged({"chip_role": {**s2["chip_role"], "x": "val"}}, s2)


def test_pin_on_disk_is_the_registered_label_version():
    from sentinel.models.lite import national2 as N2

    split = L.load_split()                                             # raises unless the canonical hash matches
    assert split["_sha256"] == L.FROZEN_SHA256 and split["national2_split_sha256"] == N2.FROZEN_SHA256
    assert L.assert_roles_unchanged(split) == 32820
    tampered = {**split, "chip_role": {**split["chip_role"], next(iter(split["chip_role"])): "tampered"}}
    assert L.split_sha256(tampered) != L.FROZEN_SHA256
    assert N2.FROZEN_SHA256.startswith("976b5be7")                     # the as-built pin is not replaced


# --------------------------------------------------------------------------- P2 index: F7, exclusions, loader
def _blocks(lat, lon):
    from sentinel.eval.splits import spatial_blocks
    from sentinel.models.lite import national as N

    return [str(b) for b in spatial_blocks(np.asarray(lat, float), np.asarray(lon, float), N.PARAMS["block_km"])]


def _index():
    """train_p2 water bodies W1 (two chips, one in a val-role block), W2 (train-role block), W3 (site only in a
    test-role block), U* (blocks without a Lite role); one val_p2 and one test_p2 row in the val-role block."""
    rows = [("t1", "s1", "W1", "train_p2", 40.0, -100.0), ("t2", "s2", "W1", "train_p2", 41.0, -100.0),
            ("t3", "s3", "W2", "train_p2", 42.0, -100.0), ("t4", "s4", "W3", "train_p2", 42.0, -100.0),
            ("v1", "s5", "V1", "val_p2", 40.0, -100.0), ("x1", "s6", "X1", "test_p2", 40.0, -100.0)]
    rows += [(f"u{i}", f"su{i}", f"U{i}", "train_p2", 30.0 + i, -90.0) for i in range(12)]
    ix = pd.DataFrame(rows, columns=["naip_chip_id", "site_id", "water_body", "role", "lat", "lon"])
    ix["eligible"], ix["has_label"], ix["cens"] = True, True, 0
    ix["y_turb_fnu"] = np.where(ix["role"] == "test_p2", np.nan, 1.0)
    ix["cens_log10_limit"] = np.nan
    known = {_blocks([40.0], [-100.0])[0]: "val", _blocks([41.0], [-100.0])[0]: "train",
             _blocks([42.0], [-100.0])[0]: "train", _blocks([43.0], [-100.0])[0]: "test"}
    sites = ix[["site_id", "water_body", "role", "lat", "lon"]].copy()
    sites = pd.concat([sites, pd.DataFrame([{"site_id": "s7", "water_body": "W3", "role": "train_p2", "lat": 43.0,
                                             "lon": -100.0}])], ignore_index=True)      # a lock site without a chip
    return ix, known, sites


def test_f7_removes_whole_water_bodies_and_draws_the_unseen_blocks():
    ix, known, sites = _index()
    out, rep = PI.f7_removals(ix, known, sites)
    by = dict(zip(ix["naip_chip_id"], out))
    assert by["t1"] and by["t2"]                                       # W1: one chip in a val-role block, removed whole
    assert not by["t3"]                                                # W2: train-role block
    assert by["t4"]                                                    # W3: a lock site (no chip) in a test-role block
    assert not by["v1"] and not by["x1"]                               # other roles are never removed
    assert rep["water_bodies_removed_eval_block"] == 2 and rep["rows_removed_eval_block_water_body"] == 3
    assert rep["rows_in_blocks_without_a_lite_role"] == 12 and rep["blocks_without_a_lite_role"] == 12
    drawn = rep["drawn_block_roles"]
    assert sum(drawn.values()) == 12 and drawn.get("train", 0) < 12   # the salted rule gives some val/test blocks
    n_u = sum(by[f"u{i}"] for i in range(12))
    assert n_u == 12 - drawn["train"] == rep["rows_removed_by_draw"]
    assert rep["usable_rows_left"] == 16 - 3 - n_u
    out2, rep2 = PI.f7_removals(ix, known, sites)                      # deterministic (fixed seed, sorted blocks)
    assert (out == out2).all() and rep == rep2
    out3, _ = PI.f7_removals(ix, known, sites.iloc[:0])                # without the lock sites W3 would stay
    assert not dict(zip(ix["naip_chip_id"], out3))["t4"]


def test_excluded_lock_sites_are_a_hard_failure():
    assert set(PI.EXCLUDED_AT_LOAD) == {"MDNR-3256/1.8", "USGS-383848122215501", "USGS-374955122313301",
                                        "CBP_WQX-01608500"}
    PI.assert_no_excluded(["USGS-01234567"], "test")
    for s in PI.EXCLUDED_AT_LOAD:
        with pytest.raises(PI.P2IndexError):
            PI.assert_no_excluded(["USGS-01234567", s], "test")


@pytest.fixture()
def synth_index(tmp_path, monkeypatch):
    ix, known, sites = _index()
    f = tmp_path / "index.parquet"
    ix.to_parquet(f, index=False)
    monkeypatch.setattr(PI, "index_path", lambda: f)
    monkeypatch.setattr(PI, "INDEX_SHA256", hashlib.sha256(f.read_bytes()).hexdigest())
    monkeypatch.setattr(PI, "known_block_roles", lambda: known)
    monkeypatch.setattr(PI, "assert_no_reserved", lambda ids, where="": None)
    orig = PI.f7_removals
    monkeypatch.setattr(PI, "f7_removals", lambda ix_, known_=None, sites_=None: orig(ix_, known, sites))
    return ix, f


def test_load_index_applies_f7_the_stamp_and_the_pin(synth_index, monkeypatch):
    ix, f = synth_index
    monkeypatch.setattr(S, "stamp_status", lambda *a, **k: {"stamped": False, "reason": "not stamped: test"})
    tr = PI.load_index(["train_p2"])
    assert "t3" in set(tr["naip_chip_id"]) and not {"t1", "t2", "t4"} & set(tr["naip_chip_id"])
    assert set(tr["role"]) == {"train_p2"}
    with pytest.raises(S.StampError):
        PI.load_index(["train_p2", "val_p2"])
    with pytest.raises(PI.P2IndexError, match="unknown P2 roles"):
        PI.load_index(["val"])
    te = PI.load_index(["test_p2"])                                    # keys and the present flag only
    assert list(te["naip_chip_id"]) == ["x1"] and te["y_turb_fnu"].isna().all()
    monkeypatch.setattr(S, "stamp_status", lambda *a, **k: {"stamped": True, "reason": "stamped"})
    assert list(PI.load_index(["val_p2"])["naip_chip_id"]) == ["v1"]
    monkeypatch.setattr(PI, "INDEX_SHA256", "0" * 64)                  # another file than the pinned index
    with pytest.raises(PI.P2IndexError, match="differs from the pinned"):
        PI.load_index(["train_p2"])


def test_load_index_refuses_a_test_value_and_an_excluded_site(synth_index, monkeypatch):
    ix, f = synth_index
    bad = ix.copy()
    bad.loc[bad["role"] == "test_p2", "y_turb_fnu"] = 2.0
    bad.to_parquet(f, index=False)
    monkeypatch.setattr(PI, "INDEX_SHA256", hashlib.sha256(f.read_bytes()).hexdigest())
    with pytest.raises(PI.P2IndexError, match="test_p2 label value"):
        PI.load_index(["train_p2"])
    bad = ix.copy()
    bad.loc[0, "site_id"] = "CBP_WQX-01608500"
    bad.to_parquet(f, index=False)
    monkeypatch.setattr(PI, "INDEX_SHA256", hashlib.sha256(f.read_bytes()).hexdigest())
    with pytest.raises(PI.P2IndexError):
        PI.load_index(["train_p2"])


def test_p2_labels_use_the_project_label_rule_and_the_index_is_pinned():
    import inspect

    src = inspect.getsource(PI.wqp_days)
    assert "WR.surface_samples(" in src and "M.wqp_surface_samples" not in inspect.getsource(PI)
    assert len(PI.INDEX_SHA256) == 64 and PI.INDEX_SHA256 != "PENDING"
    assert PI.F7_EVAL_BLOCK_ROLES == ("val", "test")


# --------------------------------------------------------------------------- scoring an S2-input model (C2)
def _assign(lon, lat):
    """lon < 0 -> point in a test-role water body; lat < 0 -> its buffer; lon >= 10 -> not in a lake."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return pd.DataFrame({"in_lake": lon < 10, "role": np.where(lon < 0, "test", "train"), "buffer_excluded": lat < 0})


def _ev():
    return pd.DataFrame({
        "naip_chip_id": list("abcdefgh"),
        "lon": [-1.0, -1.0, -1.0, -1.0, 5.0, -1.0, -1.0, 20.0],
        "lat": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "naip_date": pd.to_datetime(["2019-07-01", "2019-07-01", "2024-06-01", "2024-01-03", "2019-07-01", None,
                                     "2024-06-01", "2019-07-01"]),
        "sample_date": pd.to_datetime(["2019-07-02", "2019-07-02", "2024-06-02", "2024-01-04", "2019-07-02", None,
                                       "2023-12-31", "2019-07-01"]),
        "dt_days_s2_naip": [1.0, 1.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0],
        "s2_chip_id": ["s1", "s2", "s3", "s4", "s5", "s6", "s3", None]})


def test_no_s2_input_model_is_scored_on_an_m2_test_water_body_before_2024():
    ev = _ev()
    b = TM.teacher_score_blocked(ev, assign=_assign)
    assert dict(zip(ev["naip_chip_id"], b)) == {
        "a": True,      # test-role lake, 2019
        "b": True,      # its buffer rows are rows of the water body too
        "c": False,     # 2024 on every date
        "d": True,      # the teacher scene can be up to 5 days before the NAIP date: 2023-12-29
        "e": False,     # a train-role lake
        "f": True,      # no date at all: blocked
        "g": True,      # sampled on 2023-12-31
        "h": False}     # not in a lake
    ids = TM.teacher_chip_ids(ev, b)
    assert ids == ["s5"]                                               # s3 is shared with a blocked row: not predicted
    TM.assert_teacher_scorable(ev, ids, b)
    with pytest.raises(TM.MaskError, match="2 rows of M2 test-role water bodies"):
        TM.assert_teacher_scorable(ev, ["s1", "s3", "s5"], b)
    with pytest.raises(TM.MaskError, match="date columns"):
        TM.teacher_score_blocked(ev.drop(columns=["naip_date", "sample_date"]), assign=_assign)
    with pytest.raises(TM.MaskError, match="does not match"):
        TM.assert_teacher_scorable(ev, ids, b[:3])


def test_national_look_script_blocks_the_teacher_step():
    src = (paths.REPO_ROOT / "scripts" / "eval" / "national_lite_score_test.py").read_text()
    i_block, i_assert = src.index("TM.teacher_score_blocked(ev)"), src.index("TM.assert_teacher_scorable(ev, ids, blocked)")
    i_pred = src.index("tch.predict_store(")
    assert i_block < i_assert < i_pred                                 # the hard failure sits in front of the call
    assert "chip_ids=ids" in src and 'ev.loc[has, "s2_chip_id"]' not in src
    assert src.index("except TM.MaskError:") < src.index("except Exception as e:   # descriptive only")
    assert 's2_ok = ev["s2_chip_id"].where(~blocked)' in src
    from sentinel.models.lite import national_testlook as T

    assert "sentinel/models/litev2/tests_mask.py" in T.national_frozen_files()   # covered by the national code hash


def test_v2_code_reads_validation_rows_through_the_guarded_loaders_only():
    """No litev2 module or litev2 script calls a ``lite`` role loader directly (they would bypass the stamp)."""
    files = list((paths.REPO_ROOT / "sentinel" / "models" / "litev2").glob("*.py")) + \
        list((paths.REPO_ROOT / "scripts").glob("*/litev2_*.py"))
    assert len(files) >= 8
    for f in files:
        src = f.read_text()
        for call in ("N.role_frame(", "N2.role_frame(", "D.role_frame(", "national.role_frame(", "national2.role_frame(",
                     "N.train_val_frames(", "N2.train_val_frames("):
            assert call not in src, f"{f.name} calls {call}"


# --------------------------------------------------------------------------- the re-run wrapper (order 2(c))
def _rerun_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("national2b_rerun",
                                                  paths.REPO_ROOT / "scripts/train/national2b_rerun.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rerun_wrapper_fixes_name_and_refuses_changed_settings(tmp_path):
    rr = _rerun_module()
    cfg = tmp_path / "x_national2.yaml"
    cfg.write_text("name: x_national2\nkd_weight: 0.3\n")
    out = rr.prepare_argv(["train", "--config", str(cfg), "--track", "national"])
    assert out[-2:] == ["--name", "x_national2b"]
    for extra in (["--seeds", "0"], ["--smoke"], ["--name", "y"]):
        with pytest.raises(SystemExit):
            rr.prepare_argv(["train", "--config", str(cfg), "--track", "national", *extra])
    with pytest.raises(SystemExit):
        rr.prepare_argv(["train", "--config", str(cfg), "--track", "original"])
    with pytest.raises(SystemExit):
        rr.prepare_argv(["targets", "--manifest", "m"])                 # a re-run computes no teacher target
    other = tmp_path / "x.yaml"
    other.write_text("name: x\n")
    with pytest.raises(SystemExit):
        rr.prepare_argv(["train", "--config", str(other), "--track", "national"])


def test_rerun_wrapper_needs_the_stamp(tmp_path, monkeypatch):
    rr = _rerun_module()
    cfg = tmp_path / "x_national2.yaml"
    cfg.write_text("name: x_national2\n")
    monkeypatch.setattr(S, "require_stamp", lambda what, *a, **k: (_ for _ in ()).throw(S.StampError(what)))
    with pytest.raises(S.StampError):
        rr.main(["train", "--config", str(cfg), "--track", "national"])
