# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Release guard and data-hygiene rules of the M6 architecture-v2 round
(``docs/preregistration/hydromet_arch_v2_lead_rulings.md`` §2.2, §2.3, L10; lead relay of 2026-11-10 15:00 EST).

* every non-smoke entry point passes ``runner.assert_released``: stamped amendment, stamped deviations file with its
  current sha256 and read-only, ``data.verify_pins`` (with the pins sha256 written in the deviations file) and the
  code-pin record; a stamp row is a lead row (``extra.written_by == "lead"``, the exact experiment name); a non-smoke
  run reads only a pinned config file; every non-smoke run record carries the release fields; runs of another pin
  state or code-pin sha256 are not combined (entry-1 lead rulings: F1, F4, F6, decisions 2–4);
* the date filter is inside the file scan of every US label table;
* StreamPULSE is the empty set; the two forcing-only keys of the lever upstream table never reach a frame;
* R-TD: one configuration, grab share 0.5, the published form as its registered variant, no fit before the published
  recipe is transcribed.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from sentinel.hydromet_sota import data as LD  # noqa: E402
from sentinel.hydromet_sota import reserved as R  # noqa: E402
from sentinel.hydromet_sota import sites as LS  # noqa: E402
from sentinel.hydromet_sota.gso_mt import data as VD  # noqa: E402
from sentinel.hydromet_sota.gso_mt import runner as VR  # noqa: E402
from sentinel.models.hydromet import data as D  # noqa: E402
from sentinel.models.hydromet import guard  # noqa: E402

ROOT = Path(VR.__file__).resolve().parents[3]
EXP = "hydromet_arch_v2/prereg_snapshot"


def _v2_files():
    return (sorted((ROOT / "sentinel" / "hydromet_sota").glob("*/*.py"))
            + sorted((ROOT / "scripts").glob("*/sota_hydromet_v2_*.py")))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _row(shas, rid="r1", ws=VR.WORKSTREAM, exp=EXP, script=None, by="lead"):
    r = {"id": rid, "workstream": ws, "experiment": exp, "timestamp": "t", "extra": {"sha256": shas}}
    if by is not None:
        r["extra"]["written_by"] = by
    if script:
        r["code"] = {"script": script}
    return json.dumps(r) + "\n"


REL = {"amendment_sha256": "a" * 64, "amendment_registry_id": "row-a", "deviations_sha256": "d" * 64,
       "deviations_registry_id": "row-d", "pins_sha256": "b" * 64, "changed_under_stamp": {},
       "code_pins_sha256": "c" * 64, "code_changed_under_stamp": {}, "reader": VR.POOL_READER}


def _code_pins(tmp_path, monkeypatch, names=("runner.py", "nets.py")):
    """A code-pin record over temporary files (the real record and the real code are never touched)."""
    d = tmp_path / "code"
    d.mkdir()
    files = {f"sentinel/x/{n}": d / n for n in names}
    for p in files.values():
        p.write_text(f"# {p.name}\n")
    rec = tmp_path / "code_pins.json"
    monkeypatch.setattr(VR, "CODE_PINS", rec)
    monkeypatch.setattr(VR, "code_pin_targets", lambda: dict(files))
    return files, rec


def _write_code_pins(rec, files):
    if rec.exists():
        os.chmod(rec, 0o644)
    rec.write_text(json.dumps({"files": {k: _sha(v) for k, v in files.items()}}))
    os.chmod(rec, 0o444)


# ------------------------------------------------------------------------------------------------ L10 release guard
def test_release_guard_needs_stamp_deviations_row_and_pins(tmp_path, monkeypatch):
    am, dv, reg, pins = (tmp_path / n for n in ("amendment.md", "deviations.md", "registry.jsonl", "pins.json"))
    am.write_text("amendment\n")
    os.chmod(am, 0o444)
    a_sha = _sha(am)
    monkeypatch.setattr(VR, "AMENDMENT", am)
    monkeypatch.setattr(VR, "DEVIATIONS", dv)
    monkeypatch.setattr(VD, "PINS_FILE", pins)
    pins.write_text(json.dumps({"files": {"a": "0" * 64}}))
    p_sha = _sha(pins)
    calls = {"n": 0, "fail": False}

    def fake_verify():
        calls["n"] += 1
        if calls["fail"]:
            raise RuntimeError("sha256 differs from its pin")
        return {**json.loads(pins.read_text()), "changed_under_stamp": {}}
    monkeypatch.setattr(VD, "verify_pins", fake_verify)

    with pytest.raises(VR.NotStampedError):                                   # (i) the amendment stamp itself
        VR.assert_released(registry_path=reg)
    reg.write_text(_row({VR.AMENDMENT_REL: a_sha}))
    assert VR.assert_stamped(registry_path=reg)["stamped"]
    with pytest.raises(VR.NotStampedError, match="does not exist"):           # (ii) no deviations file
        VR.assert_released(registry_path=reg)
    dv.write_text(f"## Entry 1\npins.json sha256 {'f' * 64}\n")
    with pytest.raises(VR.NotStampedError, match="no lead stamp row"):
        VR.assert_released(registry_path=reg)
    d_sha = _sha(dv)
    with reg.open("a") as fh:                                                 # rows that are not a stamp of this file
        fh.write(_row({VR.AMENDMENT_REL: a_sha, VR.DEVIATIONS_REL: "1" * 64}, rid="stale"))
        fh.write(_row({VR.AMENDMENT_REL: "2" * 64, VR.DEVIATIONS_REL: d_sha}, rid="other-amendment"))
        fh.write(_row({VR.DEVIATIONS_REL: d_sha}, rid="no-amendment"))
        fh.write(_row({VR.AMENDMENT_REL: a_sha, VR.DEVIATIONS_REL: d_sha}, rid="self",
                      script="scripts/data/sota_hydromet_v2_pins.py"))        # written by an arch-v2 script
        fh.write(_row({VR.AMENDMENT_REL: a_sha, VR.DEVIATIONS_REL: d_sha}, rid="ws", ws="M1-sonde"))
        fh.write(_row({VR.AMENDMENT_REL: a_sha, VR.DEVIATIONS_REL: d_sha}, rid="exp", exp="M6V2/pins_v1"))
        for i, kw in enumerate(({"by": None}, {"by": "M6 lever engineer"}, {"by": "Lead"},   # review F1: not lead rows
                                {"exp": "hydromet_sota/prereg_snapshot"}, {"exp": "hydromet/prereg_snapshot"},
                                {"exp": "x/" + EXP}, {"exp": EXP + "/prereg_snapshot"})):
            fh.write(_row({VR.AMENDMENT_REL: a_sha, VR.DEVIATIONS_REL: d_sha}, rid=f"not-lead-{i}", **kw))
    with pytest.raises(VR.NotStampedError, match="no lead stamp row"):
        VR.assert_released(registry_path=reg)
    assert VR.deviations_status(reg)["n_rows_naming_file"] == 3
    with reg.open("a") as fh:
        fh.write(_row({VR.AMENDMENT_REL: a_sha, VR.DEVIATIONS_REL: d_sha}, rid="lead1"))
    with pytest.raises(VR.NotStampedError, match="read-only"):                # stamped, still writable
        VR.assert_released(registry_path=reg)
    os.chmod(dv, 0o444)
    assert calls["n"] == 0                                                    # pins are checked last
    calls["fail"] = True
    with pytest.raises(RuntimeError, match="differs from its pin"):           # (iii) a pinned file changed
        VR.assert_released(registry_path=reg)
    calls["fail"] = False
    with pytest.raises(VR.NotStampedError, match="pins.json"):                # pins sha not under the lead's stamp
        VR.assert_released(registry_path=reg)
    os.chmod(dv, 0o644)
    dv.write_text(f"## Entry 1\npins.json sha256 {p_sha}\n")
    os.chmod(dv, 0o444)
    with pytest.raises(VR.NotStampedError, match="no lead stamp row"):        # the edit closed the gate
        VR.assert_released(registry_path=reg)
    with reg.open("a") as fh:
        fh.write(_row({VR.AMENDMENT_REL: a_sha, VR.DEVIATIONS_REL: _sha(dv)}, rid="lead2"))
    files, rec = _code_pins(tmp_path, monkeypatch)
    with pytest.raises(VR.NotStampedError, match="code-pin record"):          # (v) no code-pin record
        VR.assert_released(registry_path=reg)
    assert VR.assert_entry_released(registry_path=reg)["deviations_registry_id"] == "lead2"
    _write_code_pins(rec, files)
    out = VR.assert_released(registry_path=reg)
    assert out["deviations_registry_id"] == "lead2" and out["pins_sha256"] == p_sha and out["n_pinned"] == 1
    assert out["amendment_sha256"] == a_sha and out["deviations_sha256"] == _sha(dv)
    assert out["changed_under_stamp"] == {} and out["code_changed_under_stamp"] == {}
    assert out["code_pins_sha256"] == _sha(rec) and out["n_code_pinned"] == 2
    rel = VR.release_fields(out, reader=VR.POOL_READER)                       # what a run record stores
    assert rel["deviations_registry_id"] == "lead2" and rel["amendment_registry_id"] == "lead2"   # last lead row
    assert set(rel) == set(REL)
    os.chmod(dv, 0o644)
    with dv.open("a") as fh:                                                  # entry 2 appended: closed until stamped
        fh.write("## Entry 2\n")
    os.chmod(dv, 0o444)
    with pytest.raises(VR.NotStampedError, match="no lead stamp row"):
        VR.assert_released(registry_path=reg)


def test_every_non_smoke_entry_point_calls_the_release_guard():
    """Static rules: each arch-v2 training / evaluation script calls ``assert_released`` (never ``assert_stamped``
    or the guard without the code pin), builds the release record from its result, reads the pool only through
    ``load_v2_train`` with the pins required, reads a config only through ``load_config`` / ``pinned_config_path``
    and imports the synthetic module only under ``--synthetic``; the pins script, which wrote the pin file before
    the stamp of entry 1, calls ``assert_stamped``."""
    scripts = sorted((ROOT / "scripts").glob("train/sota_hydromet_v2_*.py")) \
        + sorted((ROOT / "scripts").glob("eval/sota_hydromet_v2_*.py"))
    assert len(scripts) >= 5
    for f in scripts:
        t = f.read_text()
        assert "assert_released(" in t, f.name
        assert not re.search(r"VR\.assert_stamped\(", t), f.name
        assert "assert_entry_released(" not in t, f.name
        assert re.search(r"VR\.(with_release|release_fields)\(", t), f.name
        assert "HR.public_config(" not in t and "verify_code_pins(" not in t, f.name
        loads = [ln for ln in t.splitlines() if "load_v2_train(" in ln and "VD." in ln]
        assert all(re.search(r"require_pins=(True|not a\.smoke)\b", x) for x in loads), f.name
        assert not re.search(r"\b(load_cube|scan_cube|read_grab_inst|load_lever_train|read_parquet)\(", t), f.name
        lines = t.splitlines()
        for i, ln in enumerate(lines):
            if "synthetic import" in ln:
                assert lines[i - 1].strip() == "if a.synthetic:", f.name
        if "/train/" in str(f):
            assert "assert_pool_read(" in t and re.search(r"VR\.(load_config|pinned_config_path)\(", t), f.name
            assert "safe_load(open(" not in t or "pinned_config_path(" in t, f.name
    pins = (ROOT / "scripts" / "data" / "sota_hydromet_v2_pins.py").read_text()
    assert "VR.assert_stamped(" in pins and "assert_entry_released(" in pins
    for f in sorted((ROOT / "sentinel" / "hydromet_sota").glob("*/*.py")):
        if f.name != "synthetic.py":
            assert "synthetic" not in re.sub(r"synthetic (and|runs|members)", "", f.read_text()) \
                or f.name == "runner.py", f.name
    assert not any("inherit_from_d1" in f.read_text() for f in _v2_files())
    assert not any("inherit_from_d1" in f.read_text() for f in VR.CFG_DIR.glob("*.yaml"))


def test_verify_pins_covers_every_target_and_detects_a_change(tmp_path, monkeypatch):
    files = {f"t/{n}": tmp_path / n for n in ("a.parquet", "b.yaml")}
    for f in files.values():
        f.write_text(f.name)
    pins = tmp_path / "pins.json"
    monkeypatch.setattr(VD, "PINS_FILE", pins)
    monkeypatch.setattr(VD, "pin_targets", lambda: dict(files))
    with pytest.raises(RuntimeError, match="not pinned"):
        VD.verify_pins()
    good = {k: _sha(v) for k, v in files.items()}
    vals = VD.pin_values()
    assert vals["n_common_static"] == 42 and len(vals["ea_common_static_sha256"]) == 64
    pins.write_text(json.dumps({"files": good, "values": vals}))
    with pytest.raises(RuntimeError, match="read-only"):
        VD.verify_pins()
    os.chmod(pins, 0o444)
    assert VD.verify_pins()["files"] == good

    def rewrite(obj):
        os.chmod(pins, 0o644)
        pins.write_text(json.dumps(obj))
        os.chmod(pins, 0o444)
    rewrite({"files": {"t/a.parquet": good["t/a.parquet"]}, "values": vals})                  # a target without a pin
    with pytest.raises(RuntimeError, match="does not cover"):
        VD.verify_pins()
    rewrite({"files": {**good, "t/gone": "0" * 64}, "values": vals})                          # a pin without a target
    with pytest.raises(RuntimeError, match="does not cover"):
        VD.verify_pins()
    rewrite({"files": good})                                                  # the T-E static set is pinned by content
    with pytest.raises(RuntimeError, match="pinned by content"):
        VD.verify_pins()
    rewrite({"files": good, "values": {**vals, "n_common_static": 39}})
    with pytest.raises(RuntimeError, match="pinned by content"):
        VD.verify_pins()
    rewrite({"files": good, "values": vals})
    files["t/b.yaml"].write_text("dropout: 0.3\n")                            # a pinned config value changed
    with pytest.raises(RuntimeError, match="differs from its pin"):
        VD.verify_pins()


def test_a_pinned_code_or_config_file_changes_only_through_a_stamped_deviations_line(tmp_path, monkeypatch):
    am, dv, reg, pins = (tmp_path / n for n in ("amendment.md", "deviations.md", "registry.jsonl", "pins.json"))
    am.write_text("amendment\n")
    cfg, tab = tmp_path / "d2.yaml", tmp_path / "x.parquet"
    cfg.write_text("a: 1\n")
    tab.write_text("table")
    files = {"configs/hydromet_sota/v2/d2.yaml": cfg, "processed/x.parquet": tab}
    monkeypatch.setattr(VR, "AMENDMENT", am)
    monkeypatch.setattr(VR, "DEVIATIONS", dv)
    monkeypatch.setattr(VD, "PINS_FILE", pins)
    monkeypatch.setattr(VD, "pin_targets", lambda: dict(files))
    monkeypatch.setattr(VR.paths, "REGISTRY_PATH", reg)
    pins.write_text(json.dumps({"files": {k: _sha(v) for k, v in files.items()}, "values": VD.pin_values()}))
    os.chmod(pins, 0o444)
    assert VD.verify_pins()["changed_under_stamp"] == {}
    cfg.write_text("a: 2\n")
    with pytest.raises(RuntimeError, match="differs from its pin"):           # no deviations file
        VD.verify_pins()
    line = VD.pin_change_line("configs/hydromet_sota/v2/d2.yaml", _sha(cfg))
    dv.write_text("entry 2\n" + line + "\n")
    with pytest.raises(RuntimeError, match="differs from its pin"):           # the line alone, without the lead's stamp
        VD.verify_pins()
    os.chmod(dv, 0o444)
    reg.write_text(_row({VR.AMENDMENT_REL: _sha(am), VR.DEVIATIONS_REL: _sha(dv)}, script="scripts/x/sota_hydromet_v2_y.py"))
    with pytest.raises(RuntimeError, match="differs from its pin"):           # a row written by an arch-v2 script
        VD.verify_pins()
    reg.write_text(_row({VR.AMENDMENT_REL: _sha(am), VR.DEVIATIONS_REL: _sha(dv)}))
    assert VD.verify_pins()["changed_under_stamp"] == {"configs/hydromet_sota/v2/d2.yaml": _sha(cfg)}
    cfg.write_text("a: 3\n")                                                  # a further change is not covered
    with pytest.raises(RuntimeError, match="differs from its pin"):
        VD.verify_pins()
    cfg.write_text("a: 2\n")
    tab.write_text("table 2")                                                 # a table has no change route at all
    os.chmod(dv, 0o644)
    dv.write_text("entry 2\n" + line + "\n" + VD.pin_change_line("processed/x.parquet", _sha(tab)) + "\n")
    os.chmod(dv, 0o444)
    reg.write_text(_row({VR.AMENDMENT_REL: _sha(am), VR.DEVIATIONS_REL: _sha(dv)}))
    with pytest.raises(RuntimeError, match="processed/x.parquet"):
        VD.verify_pins()


def test_pin_targets_cover_the_ruling_list():
    """Rulings §2.3 PIN, item by item (labels only; nothing is hashed here)."""
    try:
        t = VD.pin_targets()
    except RuntimeError as e:                                                 # a required config file is missing
        pytest.fail(str(e))
    need = ["processed/hydromet_sota_v2/" + f for f in VD.PINNED] + [
        "processed/hydromet_sota/lever_sites.parquet", "docs/preregistration/sonde_split.json",
        "docs/preregistration/hydromet_track_n_stations.json", "docs/preregistration/sonde_newsites_list.json",
        "processed/sites/sites.parquet", "sentinel/hydromet_sota/sites.py", "sentinel/hydromet_sota/reserved.py",
        "sentinel/data/reserved.py", "sentinel/hydromet_sota/gso_mt/grab.py", "sentinel/hydromet_sota/gso_mt/data.py",
        "scripts/data/sota_hydromet_v2_build.py", "processed/hydromet_upstream_forcing/upstream_forcing_sota.parquet"]
    need += [f"configs/hydromet_sota/v2/{c}" for c in VD.CONFIG_REQUIRED]
    assert not [k for k in need if k not in t]
    assert "configs/hydromet_sota/v2/pins.json" not in t
    assert all(Path(v).is_file() for v in t.values()), [k for k, v in t.items() if not Path(v).is_file()]
    cfg = {f.name for f in VR.CFG_DIR.iterdir() if f.is_file() and f.name != "pins.json"}
    assert cfg == {k.rsplit("/", 1)[-1] for k in t if k.startswith("configs/hydromet_sota/v2/")}


# ------------------------------------------------------------- entry-1 lead rulings: F1, F4, code pin, run records
def test_a_stamp_row_is_a_lead_row_of_the_exact_experiment(tmp_path, monkeypatch):
    """Review F1: ``extra.written_by == "lead"`` and experiment ``hydromet_arch_v2/prereg_snapshot`` exactly."""
    am, reg = tmp_path / "amendment.md", tmp_path / "registry.jsonl"
    am.write_text("amendment\n")
    os.chmod(am, 0o444)
    monkeypatch.setattr(VR, "AMENDMENT", am)
    shas = {VR.AMENDMENT_REL: _sha(am)}
    bad = [{"by": None}, {"by": "M6 lever engineer"}, {"by": "lead "}, {"by": ["lead"]},
           {"exp": "hydromet_sota/prereg_snapshot"}, {"exp": "hydromet/prereg_snapshot"}, {"exp": "a" + EXP},
           {"ws": "M1-sonde"}, {"script": "scripts/eval/sota_hydromet_v2_inner.py"}, {"rid": ""}]
    reg.write_text("".join(_row(shas, **{"rid": f"bad{i}", **kw}) for i, kw in enumerate(bad))
                   + json.dumps({"id": "noextra", "workstream": VR.WORKSTREAM, "experiment": EXP}) + "\n"
                   + json.dumps({"id": "strextra", "workstream": VR.WORKSTREAM, "experiment": EXP, "extra": "lead"})
                   + "\n" + json.dumps(["hydromet_arch_v2/prereg_snapshot"]) + "\nnot json /prereg_snapshot\n")
    assert not list(VR._stamp_rows(reg)) and VR.stamp_status(reg)["n_stamp_rows"] == 0
    with pytest.raises(VR.NotStampedError):
        VR.assert_stamped(registry_path=reg)
    with reg.open("a") as fh:
        fh.write(_row(shas, rid="lead"))
    assert VR.assert_stamped(registry_path=reg)["registry_id"] == "lead"
    assert [r["id"] for r, _ in VR._stamp_rows(reg)] == ["lead"]


def test_the_real_stamp_rows_are_lead_rows():
    """The registry as it is: the guard is open only through rows that carry the lead's marker."""
    for r, _ in VR._stamp_rows():
        assert r["experiment"] == VR.STAMP_EXPERIMENT == EXP and r["extra"]["written_by"] == VR.STAMP_WRITER == "lead"
    st, dv = VR.stamp_status(), VR.deviations_status()
    assert st["stamped"] == (bool(st["registry_id"]) and st["read_only"])
    assert dv["released"] == (bool(dv["registry_id"]) and dv["read_only"])


def test_a_non_smoke_run_reads_only_a_pinned_config_file(tmp_path):
    """Review F4: a copy of a config outside ``configs/hydromet_sota/v2/`` is refused, whatever its content."""
    real = VR.CFG_DIR / "d1.yaml"
    assert VR.pinned_config_path(real) == real and VR.pinned_config_path(str(real)) == real
    assert VR.load_config(real, "g1")["_config_file"] == str(real)
    copy = tmp_path / "d1.yaml"
    copy.write_text(real.read_text())
    link = tmp_path / "link" / "d1.yaml"
    link.parent.mkdir()
    link.symlink_to(real)
    for p in (copy, link, VR.CFG_DIR / "nope.yaml", VR.CFG_DIR / "pins.json", tmp_path / "common.yaml",
              VR.CFG_DIR.parent / "C1.yaml"):
        with pytest.raises(VR.NotStampedError, match="pinned file"):
            VR.pinned_config_path(p)
    with pytest.raises(VR.NotStampedError, match="pinned file"):
        VR.load_config(copy, "g1")
    with pytest.raises(VR.NotStampedError, match="pinned file"):
        VR.load_config(link, "g1")
    assert VR.load_config(copy, "g1", smoke=True)["_smoke"] is True              # smoke runs register nothing
    assert VR.load_config(VR.CFG_DIR / ".." / "v2" / "d1.yaml", "g1")["_config_file"] == str(real)
    gbm = (ROOT / "scripts" / "train" / "sota_hydromet_v2_gbm.py").read_text()   # the one driver that reads YAML itself
    i, j = gbm.index("pinned_config_path("), gbm.index("yaml.safe_load(open(cpath)")
    assert gbm.index("if not a.smoke:") < i < j


def test_code_pin_record_and_its_stamped_change_route(tmp_path, monkeypatch):
    """Decision 4: the code-pin record is read-only, covers every code target, and a code file changes (or is
    added) only through a line of the stamped deviations file; the last line of a label is its only valid state."""
    am, dv, reg = (tmp_path / n for n in ("amendment.md", "deviations.md", "registry.jsonl"))
    am.write_text("amendment\n")
    monkeypatch.setattr(VR, "AMENDMENT", am)
    monkeypatch.setattr(VR, "DEVIATIONS", dv)
    files, rec = _code_pins(tmp_path, monkeypatch)
    lab = "sentinel/x/nets.py"
    with pytest.raises(VR.NotStampedError, match="is missing"):
        VR.verify_code_pins(reg)
    rec.write_text(json.dumps({"files": {k: _sha(v) for k, v in files.items()}}))
    with pytest.raises(VR.NotStampedError, match="read-only"):
        VR.verify_code_pins(reg)
    os.chmod(rec, 0o444)
    out = VR.verify_code_pins(reg)
    assert out == {"code_pins_sha256": _sha(rec), "n_code_pinned": 2, "code_changed_under_stamp": {}}
    orig = files[lab].read_text()
    files[lab].write_text("# nets v2\n")

    def stamp(text):
        if dv.exists():
            os.chmod(dv, 0o644)
        dv.write_text(text)
        os.chmod(dv, 0o444)
        reg.write_text(_row({VR.AMENDMENT_REL: _sha(am), VR.DEVIATIONS_REL: _sha(dv)}))
    with pytest.raises(VR.NotStampedError, match="code-pinned state"):        # changed, no deviations file
        VR.verify_code_pins(reg)
    v2 = VR.code_pin_change_line(lab, _sha(files[lab]))
    assert v2 == f"CODE PIN CHANGE: {lab} sha256 {_sha(files[lab])}"
    dv.write_text("entry 2\n" + v2 + "\n")
    with pytest.raises(VR.NotStampedError, match="code-pinned state"):        # the line without the lead's stamp
        VR.verify_code_pins(reg)
    os.chmod(dv, 0o444)
    reg.write_text(_row({VR.AMENDMENT_REL: _sha(am), VR.DEVIATIONS_REL: _sha(dv)}, by="M6 arch-v2 engineer"))
    with pytest.raises(VR.NotStampedError, match="code-pinned state"):        # a row that is not the lead's
        VR.verify_code_pins(reg)
    stamp("entry 2\nsee `" + v2 + "`\n " + v2 + "\n")                        # quoted or indented: not a whole line
    with pytest.raises(VR.NotStampedError, match="code-pinned state"):
        VR.verify_code_pins(reg)
    stamp("entry 2\n" + v2 + "\n")
    assert VR.verify_code_pins(reg)["code_changed_under_stamp"] == {lab: _sha(files[lab])}
    files[lab].write_text(orig)                                               # the original pin is no longer a state
    with pytest.raises(VR.NotStampedError, match="code-pinned state"):
        VR.verify_code_pins(reg)
    files[lab].write_text("# nets v3\n")
    v3 = VR.code_pin_change_line(lab, _sha(files[lab]))
    stamp("entry 2\n" + v2 + "\nentry 3\n" + v3 + "\n")                      # the last line wins
    assert VR.verify_code_pins(reg)["code_changed_under_stamp"] == {lab: _sha(files[lab])}
    files[lab].write_text("# nets v2\n")
    with pytest.raises(VR.NotStampedError, match="code-pinned state"):        # the earlier stamped state is void
        VR.verify_code_pins(reg)
    files[lab].write_text("# nets v3\n")
    new = tmp_path / "code" / "do_stage.py"                                   # a new driver or module
    new.write_text("# new\n")
    files["scripts/train/sota_hydromet_v2_do.py"] = new
    with pytest.raises(VR.NotStampedError, match="sota_hydromet_v2_do.py"):
        VR.verify_code_pins(reg)
    stamp(dv.read_text() + VR.code_pin_change_line("scripts/train/sota_hydromet_v2_do.py", _sha(new)) + "\n")
    assert len(VR.verify_code_pins(reg)["code_changed_under_stamp"]) == 2
    del files["sentinel/x/runner.py"]                                         # a pinned code file went away
    with pytest.raises(VR.NotStampedError, match="missing"):
        VR.verify_code_pins(reg)


def test_code_pin_targets_are_the_ruling_list():
    """Entry-1 lead rulings, decision 4, item by item; no file of pins.json is pinned twice."""
    t = VR.code_pin_targets()
    need = [f"sentinel/hydromet_sota/gso_mt/{n}.py" for n in ("runner", "nets", "trainer", "criterion", "gbm_ref",
                                                             "gates", "__init__")]
    need += [f"sentinel/hydromet_sota/stack/{p.name}" for p in (ROOT / "sentinel/hydromet_sota/stack").glob("*.py")]
    need += ["sentinel/hydromet_sota/thermox/__init__.py", "sentinel/hydromet_sota/thermox/nets.py"]
    need += [f"scripts/train/sota_hydromet_v2_{n}.py" for n in ("nn", "gbm", "physics", "stack")]
    need += ["scripts/eval/sota_hydromet_v2_inner.py", "scripts/data/sota_hydromet_v2_pins.py"]
    assert sorted(t) == sorted(need) and all(p.is_file() for p in t.values())
    assert not set(t) & set(VD.pin_targets())
    both = set(t) | set(VD.pin_targets())                  # every arch-v2 code file is under one of the two records
    left = [str(p.relative_to(ROOT)) for p in _v2_files() if str(p.relative_to(ROOT)) not in both]
    assert left == ["sentinel/hydromet_sota/gso_mt/synthetic.py"]
    assert VR.CODE_PINS.parent.name == "preregistration" and VR.CFG_DIR not in VR.CODE_PINS.parents


def test_the_code_pin_record_matches_the_code_once_written():
    """Before the record exists nothing non-smoke can run; once written it is read-only and verifies."""
    if not VR.CODE_PINS.is_file():
        with pytest.raises(VR.NotStampedError, match="code-pin record"):
            VR.verify_code_pins()
        return
    assert VR._read_only(VR.CODE_PINS)
    out = VR.verify_code_pins()
    assert out["n_code_pinned"] == len(VR.code_pin_targets()) - len(out["code_changed_under_stamp"]) \
        or out["n_code_pinned"] == len(VR.code_pin_targets())


def test_non_smoke_run_records_carry_the_release_fields(tmp_path, monkeypatch):
    """F1, decision 2 condition 3, decision 3, decision 4: deviations sha256 and row id, pins sha256,
    ``changed_under_stamp``, the code-pin sha256 and the reader are in the run record, ``release.json`` and every
    registry row; a non-smoke record without them is refused."""
    import sentinel.core.registry as REG

    ok = ["01646500", "WQP:USGS-01646500"]
    cfg = {"arm": "d1", "point": "g1", "variant_of": None, "_smoke": False, "_config_file": "x"}
    for bad in (cfg, {**cfg, "_release": None}, {**cfg, "_release": {**REL, "pins_sha256": None}},
                {**cfg, "_release": {k: v for k, v in REL.items() if k != "changed_under_stamp"}},
                {**cfg, "_release": {**REL, "reader": ""}}, {**cfg, "_release": {**REL, "deviations_registry_id": None}}):
        with pytest.raises(VR.NotStampedError, match="release fields"):
            VR.training_run(bad, tmp_path, seed=0, refit=False, sites=ok)
        with pytest.raises(VR.NotStampedError, match="release fields"):
            VR.log_inner(bad, {"best_J": 1.0}, seed=0, rdir=tmp_path)
    assert not list(tmp_path.iterdir())
    with pytest.raises(VR.NotStampedError):
        VR.with_release(cfg, None)                                            # non-smoke without the guard's result
    smoke = VR.with_release({**cfg, "_smoke": True}, None)
    assert smoke["_release"] is None and VR.run_release(smoke) is None and "release" not in VR.run_config(smoke)
    assert VR.write_release(tmp_path, smoke) is None and not list(tmp_path.iterdir())
    stamp = {**{k: REL[k] for k in REL if k not in ("reader", "amendment_registry_id")}, "registry_id": "row-a"}
    run = VR.with_release(cfg, stamp)
    assert run["_release"] == REL and "_release" not in VR.HR.public_config(run)   # not in the counted config hash
    assert VR.run_config(run, seed=3)["release"] == REL and VR.run_config(run, seed=3)["seed"] == 3
    tr = VR.training_run(run, tmp_path, seed=0, refit=False, sites=ok)
    assert tr.config["release"] == REL
    rec = json.loads((tmp_path / "release.json").read_text())
    assert {k: rec[k] for k in REL} == REL and len(rec["jobs"]) == 1
    later = {**run, "_release": {**REL, "deviations_sha256": "e" * 64, "deviations_registry_id": "row-e"}}
    VR.write_release(tmp_path, later)                                         # a resume after a newer stamped entry
    rec = json.loads((tmp_path / "release.json").read_text())
    assert [j["deviations_registry_id"] for j in rec["jobs"]] == ["row-d", "row-e"]
    for k, v in (("pins_sha256", "0" * 64), ("code_pins_sha256", "0" * 64), ("changed_under_stamp", {"a": "b"}),
                 ("code_changed_under_stamp", {"a": "b"})):
        with pytest.raises(VR.NotStampedError, match="another state"):        # a run never spans two states
            VR.write_release(tmp_path, {**run, "_release": {**REL, k: v}})
    seen = []
    monkeypatch.setattr(REG, "log_result", lambda **kw: seen.append(kw) or {"id": "x"})
    VR.log_inner(run, {"best_J": 1.0, "best_j_sonde": 0.9, "best_j_grab": None}, seed=0, rdir=tmp_path)
    assert len(seen) == 2 and all(r["extra"]["release"] == REL and "release" not in r["config"] for r in seen)
    info = type("V", (), {"info": {"pins_sha256": REL["pins_sha256"]}})()
    VR.assert_pool_read(run, info)
    VR.assert_pool_read(smoke, type("V", (), {"info": {}})())
    for other in ({"pins_sha256": None}, {"pins_sha256": "0" * 64}, {}):      # a pool read without the pin check
        with pytest.raises(VR.NotStampedError, match="require_pins=True"):
            VR.assert_pool_read(run, type("V", (), {"info": other})())


def test_runs_of_another_state_are_not_combined(tmp_path):
    """Decision 4 (code-pin sha256) and decision 2 condition 4 (a superseded pin state is void)."""
    dirs = [tmp_path / f"d1_g1_s{s}" for s in range(3)]
    for d in dirs:
        d.mkdir()
    with pytest.raises(VR.NotStampedError, match="no release record"):
        VR.assert_combinable(dirs, REL, "t")
    for d in dirs:
        VR.write_json(d / "release.json", {**REL, "deviations_sha256": "f" * 64, "jobs": []})
    out = VR.assert_combinable(dirs, REL, "t")                                 # a newer deviations entry is no state
    assert out["n_runs"] == 3 and out["code_pins_sha256"] == REL["code_pins_sha256"]
    assert out["code_changed_under_stamp_by_run"] == {d.name: {} for d in dirs}
    for k, v in (("code_pins_sha256", "0" * 64), ("pins_sha256", "0" * 64), ("changed_under_stamp", {"x": "y"})):
        VR.write_json(dirs[1] / "release.json", {**REL, k: v})
        with pytest.raises(VR.NotStampedError, match="another state"):
            VR.assert_combinable(dirs, REL, "t")
    VR.write_json(dirs[1] / "release.json", {k: v for k, v in REL.items() if k != "code_pins_sha256"})
    with pytest.raises(VR.NotStampedError, match="release fields"):
        VR.assert_combinable(dirs, REL, "t")
    VR.write_json(dirs[1] / "release.json", REL)
    with pytest.raises(VR.NotStampedError, match="release fields"):
        VR.assert_combinable(dirs, {**REL, "code_pins_sha256": None}, "t")


# ------------------------------------------------------------------------------------ date filter inside the scan
@pytest.fixture
def site_dir(tmp_path, monkeypatch):
    from sentinel.models.sonde import data as SD

    d = tmp_path / "sites"
    d.mkdir()
    rs = np.random.default_rng(1)
    idx = pd.date_range("2019-06-01", "2022-03-31", freq="D", name="date")
    for s, cols in (("S0001", ("temp_c_mean", "temp_c_max", "do_mgl_mean")), ("S0002", ("temp_c_mean",))):
        df = pd.DataFrame(index=idx[rs.random(len(idx)) > 0.2])
        for c in cols:
            df[c] = rs.normal(12, 5, len(df)).astype("float64")
            df[f"{c}_prov"] = rs.random(len(df)) > 0.7
            df[f"{c}_ts"] = "x"
        df.to_parquet(d / f"{s}.parquet")
    monkeypatch.setattr(SD, "SITE_DIR", d)
    return d


def test_scan_cube_equals_the_frozen_reader_and_never_materialises_a_later_row(site_dir, monkeypatch):
    from sentinel.models.sonde import data as SD

    sites, variables = ["S0001", "S0002", "S0404"], ["temp_c_mean", "temp_c_max", "do_mgl_mean", "do_mgl_min"]
    start, end = "2019-10-01", "2021-09-30"
    ref = SD.load_cube(sites, variables, start=start, end=end)
    seen = []
    orig = pd.read_parquet

    def spy(*a, **k):
        df = orig(*a, **k)
        seen.append((k.get("filters"), df.index.min(), df.index.max(), list(df.columns)))
        return df
    monkeypatch.setattr(pd, "read_parquet", spy)
    got = VD.scan_cube(sites, variables, start=start, end=end)
    monkeypatch.setattr(pd, "read_parquet", orig)
    assert got.sites == ref.sites and got.variables == ref.variables and got.dates.equals(ref.dates)
    np.testing.assert_array_equal(got.y, ref.y)
    np.testing.assert_array_equal(got.prov, ref.prov)
    assert np.isfinite(got.y).sum() > 500 and got.prov.sum() > 50 and np.isnan(got.y[2]).all()
    assert len(seen) == 2
    for filters, lo, hi, cols in seen:                                        # what was read from the file
        assert filters and lo >= pd.Timestamp(start) and hi <= pd.Timestamp(end)
        assert not [c for c in cols if c.endswith("_ts")]
    whole = orig(site_dir / "S0001.parquet")
    assert whole.index.max() > pd.Timestamp(end) and whole.index.min() < pd.Timestamp(start)


def test_cube_scan_filter_replaces_the_frozen_name_for_one_call_only(site_dir):
    orig = D.load_cube
    with VD.cube_scan_filter():
        assert D.load_cube is VD.scan_cube
        c = D.load_cube(["S0001"], ["temp_c_mean"], start="2020-01-01", end="2020-12-31")
        assert c.y.shape == (1, 366, 1)
    assert D.load_cube is orig
    with pytest.raises(KeyError):
        with VD.cube_scan_filter():
            raise KeyError("x")
    assert D.load_cube is orig


def test_scan_cube_refuses_reserved_sites(site_dir):
    from sentinel.data import reserved as shared

    some = sorted(shared.reserved_ids())[0]
    with pytest.raises(shared.ReservedSiteError):
        VD.scan_cube(["S0001", some], ["temp_c_mean"], start="2020-01-01", end="2020-12-31")


def test_grab_reader_filters_in_the_scan_and_refuses_a_table_with_later_rows(tmp_path):
    rows = pd.DataFrame({"station_key": ["WQP:A-1", "WQP:A-1", "WQP:B-2"], "site_id": ["A-1", "A-1", "B-2"],
                         "date": pd.to_datetime(["2010-05-01", "2015-07-02", LS.TRAIN_END]),
                         "var": ["T", "DO", "T"], "value": [10.0, 8.0, 12.0], "hour": [10.5, np.nan, 14.0],
                         "kind": ["grab"] * 3})
    f = tmp_path / "grab_inst.parquet"
    rows.to_parquet(f, index=False)
    assert len(VD.read_grab_inst(f)) == 3
    late = pd.concat([rows, rows.iloc[[0]].assign(date=pd.Timestamp(LS.TRAIN_END) + pd.Timedelta(days=1))])
    late.to_parquet(f, index=False)
    with pytest.raises(guard.ProtectedDataError, match="outside the train block"):
        VD.read_grab_inst(f)
    pd.concat([rows, rows.iloc[[0]]]).to_parquet(f, index=False)
    with pytest.raises(RuntimeError, match="duplicated"):
        VD.read_grab_inst(f)


def test_frame_dates_are_asserted_against_the_date_filter():
    d = pd.date_range("2007-10-01", LS.TRAIN_END, freq="D")
    assert VD.assert_dates_within(d, LS.TRAIN_END) == LS.TRAIN_END
    with pytest.raises(guard.ProtectedDataError, match="after the date filter"):
        VD.assert_dates_within(d.append(pd.DatetimeIndex(["2021-10-01"])), LS.TRAIN_END)
    with pytest.raises(guard.ProtectedDataError):                             # provisional gridMET range of the lever table
        VD.assert_dates_within(pd.DatetimeIndex(["2026-09-10"]), "2026-09-09")


# ----------------------------------------------------------------------------------- StreamPULSE and forcing-only keys
def test_streampulse_is_the_empty_set(monkeypatch):
    ok = pd.DataFrame({"source": ["usgs_dv", "wqp_grab"]})
    monkeypatch.setattr(VD, "pin_targets", lambda: {"processed/x.parquet": Path("/data/x.parquet")})
    assert VD.assert_no_streampulse(ok) == 0
    with pytest.raises(RuntimeError, match="pool sources"):
        VD.assert_no_streampulse(pd.DataFrame({"source": ["usgs_dv", "streampulse"]}))
    monkeypatch.setattr(VD, "pin_targets", lambda: {"x": Path("/data/processed/labels_streampulse/x.parquet")})
    with pytest.raises(RuntimeError, match="StreamPULSE path"):
        VD.assert_no_streampulse(ok)


def test_no_arch_v2_file_reads_a_streampulse_or_powell_table():
    """No arch-v2 code names a StreamPULSE table (empty set) and none imports the downstream metabolism reader
    (Powell rows wait for the shared role table). The guard in ``gso_mt/data.py`` is the only place the word occurs."""
    files = _v2_files()
    assert len(files) > 8
    for f in files:
        t = f.read_text()
        assert not re.search(r"downstream_metabolism|labels_streampulse|raw/streampulse|labels_powell|powell_estimates",
                             t), f.name
        if f.name == "data.py" and f.parent.name == "gso_mt":
            code = [ln for ln in t.splitlines() if "streampulse" in ln.lower() and "read_" in ln]
            assert not code
        else:
            assert "streampulse" not in t.lower(), f.name


def test_forcing_only_keys_never_reach_a_frame():
    ok = ["01646500", "USGS-01646500", "WQP:USGS-01646500"]
    assert VD.assert_no_forcing_only_site(ok, forcing_only={"USGS-02298123999"}) == 1
    for bad in ("02298123999", "USGS-02298123999", "WQP:USGS-02298123999", "WQP:ORG-02298123999-A"):
        with pytest.raises(R.ReservedSiteError, match="forcing-only"):
            VD.assert_no_forcing_only_site(ok + [bad], forcing_only={"USGS-02298123999"})


@pytest.mark.skipif(not LD.UPSTREAM_SOTA.is_file(), reason="lever upstream table not on this host")
def test_lever_upstream_table_has_exactly_two_forcing_only_keys_and_they_are_not_fit_sites():
    """Lead relay 2026-11-10: 445 of the 447 keys are fit sites; the other two carry forcing rows only. The keys are
    derived from the tables (never written into code) and must be absent from the lever fit list."""
    extra = VD.forcing_only_sites()
    assert len(extra) == VD.N_FORCING_ONLY == 2
    pool = pd.read_parquet(LD.SOTA_DIR / "lever_sites.parquet", columns=["site_no", "source", "rule"])
    usgs = pool[pool["source"] == "usgs_dv"]
    fit = usgs.loc[usgs["rule"] == "fit", "site_no"].astype(str).tolist()
    assert len(fit) == 445 and VD.assert_no_forcing_only_site(fit, forcing_only=extra) == 2
    assert set(usgs.loc[usgs["site_no"].astype(str).isin(extra), "rule"]) == {"reserved_comid"}
    with pytest.raises(R.ReservedSiteError):
        VD.assert_no_forcing_only_site(fit + sorted(extra)[:1], forcing_only=extra)


# -------------------------------------------------------------------------------------------------------- R-TD
def test_rtd_has_one_configuration_and_no_fit_before_the_recipe_is_transcribed():
    import yaml

    arm = yaml.safe_load((VR.CFG_DIR / "rtd.yaml").read_text())
    assert list(arm["grid"]) == ["fixed"] and arm["grid"]["fixed"] == {"grab_share": 0.5}
    assert list(arm["ablations"]) == ["published"]
    assert arm["ablations"]["published"] == {"data": {"grab": False}, "grab_share": 0.0}
    cfg = VR.load_config(VR.CFG_DIR / "rtd.yaml", "fixed")
    for verified, source in ((False, None), (False, "entry 1"), (True, None), (True, "")):
        with pytest.raises(VR.NotStampedError, match="recipe"):
            VR.assert_recipe(dict(cfg, recipe_verified=verified, recipe_source=source))
    VR.assert_recipe(dict(cfg, recipe_verified=True, recipe_source="deviations entry 1; paper, table"))
    VR.assert_recipe(dict(cfg, recipe_verified=False, _smoke=True))           # smoke runs register nothing
    VR.assert_recipe(VR.load_config(VR.CFG_DIR / "d1.yaml", "g1"))            # arms without a published recipe
    if arm["recipe_verified"] is not True:                                    # as long as the file says "not verified"
        with pytest.raises(VR.NotStampedError):
            VR.assert_recipe(cfg)


def test_rtd_w_is_one_fixed_configuration_from_the_first_row_of_willard_table_s6():
    """Lead ruling L11: hidden 570 / 4 layers / dropout 2.216e-05 and nothing else from Willard et al. 2024; every
    other value is that of ``rtd.yaml`` (common recipe, grab share 0.5, the no-grab published form)."""
    import yaml

    w, r = (yaml.safe_load((VR.CFG_DIR / f).read_text()) for f in ("rtd_w.yaml", "rtd.yaml"))
    assert list(w["grid"]) == ["fixed"] and w["grid"] == r["grid"] and w["ablations"] == r["ablations"]
    assert w["recipe_verified"] is True and w["recipe_source"]
    cw, cr = (VR.load_config(VR.CFG_DIR / f, "fixed") for f in ("rtd_w.yaml", "rtd.yaml"))
    assert (cw["net"]["hidden"], cw["net"]["layers"], cw["net"]["dropout"]) == (570, 4, 2.216e-05)
    assert (cr["net"]["hidden"], cr["net"].get("layers", 1), cr["net"]["dropout"]) == (100, 1, 0.5)
    assert cw["train"] == cr["train"] and cw["grab_share"] == cr["grab_share"] == 0.5
    assert {k: v for k, v in cw["net"].items() if k not in ("hidden", "layers", "dropout")} == \
        {k: v for k, v in cr["net"].items() if k not in ("hidden", "layers", "dropout")}
    pub = VR.load_config(VR.CFG_DIR / "rtd_w.yaml", "published")
    assert pub["variant_of"] == "fixed" and pub["data"]["grab"] is False and pub["net"] == cw["net"]


# ---------------------------------------------------- B-6 (lead batch confirmation 2026-11-11): dedicated tests for
# the five protections the mutation set killed only through the record's own self-check (M05, M12, M20, M25, M29).
def test_verify_code_pins_compares_the_whole_sha256_not_a_prefix(tmp_path, monkeypatch):
    """M05: a recorded hash that agrees with the file on its first 12 (or 60) characters and differs after is not
    the pinned state; only the full 64-character sha256 passes."""
    files, rec = _code_pins(tmp_path, monkeypatch)
    monkeypatch.setattr(VR, "stamped_code_states", lambda registry_path=None: {})
    lab = next(iter(files))
    true = _sha(files[lab])
    for n in (12, 32, 60, 63):
        tail = "".join("f" if c != "f" else "0" for c in true[n:])      # every character after the prefix differs
        _write_code_pins(rec, files)
        doc = json.loads(rec.read_text())
        doc["files"][lab] = true[:n] + tail
        os.chmod(rec, 0o644)
        rec.write_text(json.dumps(doc))
        os.chmod(rec, 0o444)
        with pytest.raises(VR.NotStampedError, match="not the code-pinned state"):
            VR.verify_code_pins()
    _write_code_pins(rec, files)
    assert VR.verify_code_pins()["n_code_pinned"] == len(files)


def test_check_release_requires_full_length_sha256_fields():
    """M12: every sha256 field of a release record is a 64-character string; a prefix, an over-long string or a
    non-string is refused, for each of the four sha keys."""
    assert VR.check_release(REL) is REL
    for k in VR.RELEASE_SHA_KEYS:
        for bad in ("c" * 12, "c" * 63, "c" * 65, "", 1, None, ["c" * 64]):
            with pytest.raises(VR.NotStampedError, match="release fields"):
                VR.check_release({**REL, k: bad})


def test_assert_combinable_compares_whole_state_fields_not_prefixes(tmp_path):
    """M20: two runs whose pin record, code pin or ``changed_under_stamp`` agree on a prefix (8, 32 or 63
    characters) but not in full are not combined; identical fields combine."""
    def run(name, rel):
        d = tmp_path / name
        d.mkdir()
        (d / VR.RELEASE_FILE).write_text(json.dumps(rel))
        return d
    same = run("same", REL)
    assert VR.assert_combinable([same], REL, "t")["n_runs"] == 1
    for k in ("pins_sha256", "code_pins_sha256"):
        for n in (8, 32, 63):
            other = run(f"{k}_{n}", {**REL, k: REL[k][:n] + "e" * (64 - n)})
            with pytest.raises(VR.NotStampedError, match="another state"):
                VR.assert_combinable([same, other], REL, "t")
    other = run("changed", {**REL, "changed_under_stamp": {"configs/hydromet_sota/v2/d1.yaml": "e" * 64}})
    with pytest.raises(VR.NotStampedError, match="another state"):
        VR.assert_combinable([same, other], REL, "t")
    with pytest.raises(VR.NotStampedError, match="another state"):
        VR.assert_combinable([other], REL, "t")


def test_log_inner_refuses_a_smoke_run_and_writes_no_row(tmp_path, monkeypatch):
    """M25: a smoke run never reaches the registry: ``log_inner`` raises before ``log_result`` is called, with and
    without a release record on the configuration."""
    import sentinel.core.registry as REG

    calls = []
    monkeypatch.setattr(REG, "log_result", lambda **kw: calls.append(kw) or {"id": "x"})
    for cfg in ({"arm": "d1", "point": "g1", "_smoke": True, "_release": None},
                {"arm": "d1", "point": "g1", "_smoke": True, "_release": dict(REL)}):
        with pytest.raises(RuntimeError, match="smoke"):
            VR.log_inner(cfg, {"best_J": 1.0, "best_j_sonde": 1.0, "best_j_grab": 1.0}, seed=0, rdir=tmp_path)
    assert calls == []
    ok = {"arm": "d1", "point": "g1", "_smoke": False, "_release": dict(REL), "variant_of": None,
          "_config_file": "x"}
    rows = VR.log_inner(ok, {"best_J": 1.0}, seed=0, rdir=tmp_path)
    assert len(rows) == 1 and len(calls) == 1 and calls[0]["experiment"] == VR.INNER_EXPERIMENT


def test_code_pin_record_is_written_once(tmp_path, monkeypatch):
    """M29: the pin script refuses to write the code-pin record when one exists, before it reads or hashes
    anything; the existing file keeps its bytes and its mode."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("v2_pins_script",
                                                  ROOT / "scripts" / "data" / "sota_hydromet_v2_pins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rec = tmp_path / "code_pins.json"
    rec.write_text('{"files": {}}\n')
    os.chmod(rec, 0o444)
    before = rec.read_bytes()
    monkeypatch.setattr(VR, "CODE_PINS", rec)
    monkeypatch.setattr(VR, "assert_entry_released", lambda *a, **k: pytest.fail("the record was re-written"))
    monkeypatch.setattr(VR, "code_pin_targets", lambda: pytest.fail("code hashed although the record exists"))
    with pytest.raises(SystemExit, match="written once"):
        mod.write_code_pins(VR, VD, guard, __import__("sentinel.core.paths", fromlist=["x"]))
    assert rec.read_bytes() == before and VR._read_only(rec)
