# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Launch rules of the M7 v2 round (lead order of 2026-11-10 20:20 EST: A1 Test-S KD mask on the re-run path,
A2 warm start from the re-run checkpoints only, A4 comparators before any candidate fit, A7 GPU-hour ceiling).
Synthetic tensors, temporary directories and in-memory registry lists only: no real row, checkpoint or registry row
is read and nothing is fitted."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.core import paths
from sentinel.models.lite import distill as KD
from sentinel.models.lite import losses as LS
from sentinel.models.lite import runner as R
from sentinel.models.litev2 import harness as H
from sentinel.models.litev2 import labels2b
from sentinel.models.litev2 import launch as LA
from sentinel.models.litev2 import rows as RW
from sentinel.models.litev2 import stamp
from sentinel.models.litev2 import tests_mask as TM

N_T = 3


def _script(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ A1
def _kd(n=3, k=4, hw=4, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    cell = np.arange(k).reshape(2, 2).repeat(hw // 2, 0).repeat(hw // 2, 1)
    return {"cell_map": np.tile(cell, (n, 1, 1)).astype(np.int64), "water_w": np.ones((n, hw, hw), np.float32),
            "t_mu": rng.normal(size=(n, k, N_T)).astype(np.float32),
            "t_var": rng.uniform(0.05, 0.2, (n, k, N_T)).astype(np.float32),
            "t_w": np.ones((n, k, N_T), np.float32)}


def _nll(out, kd, idx=slice(None)):
    t = lambda key: torch.as_tensor(kd[key][idx])   # noqa: E731
    return LS.distill_nll(out[idx], t("cell_map"), t("water_w"), t("t_mu"), t("t_var"), t("t_w"), N_T)


def _flags(ids, masked):
    return pd.DataFrame({"naip_chip_id": list(ids), "kd_masked": [i in masked for i in ids]})


def test_masked_pairing_contributes_zero_to_the_kd_term():
    kd = _kd()
    out = torch.randn(3, 2 * N_T, 4, 4, generator=torch.Generator().manual_seed(1), requires_grad=True)
    masked, info = LA.mask_kd_tensors(kd, np.array([True, False, False]))
    assert info == {"n_chips": 3, "n_kd_masked": 1, "n_chips_with_teacher_before_mask": 3,
                    "n_masked_with_teacher": 1, "n_chips_with_teacher": 2}
    assert (kd["t_w"] == 1).all() and np.isfinite(kd["t_mu"]).all()          # the loader's tensors are not changed
    loss = _nll(out, masked)
    assert torch.allclose(loss, _nll(out, kd, slice(1, None)))               # = the two unmasked chips alone
    assert not torch.allclose(loss, _nll(out, kd))
    loss.backward()
    assert float(out.grad[0].abs().max()) == 0.0 and float(out.grad[1:].abs().max()) > 0.0
    every, _ = LA.mask_kd_tensors(kd, np.ones(3, bool))
    assert float(_nll(out, every)) == 0.0                                    # nothing left: the KD term is 0
    with pytest.raises(TM.MaskError):
        LA.mask_kd_tensors(kd, np.ones(2, bool))


def test_rerun_path_wraps_the_frozen_kd_loader(monkeypatch):
    kd, calls, msgs = _kd(), [], []

    def frozen(lab, targets_path, **kw):
        calls.append((len(lab), targets_path, kw))
        return kd

    monkeypatch.setattr(KD, "load_kd_tensors", frozen)
    lab = pd.DataFrame({"naip_chip_id": ["c0", "c1", "c2"]})
    LA.activate_rerun_kd_mask(flags=_flags(["c0", "c1", "c2", "c9"], {"c0", "c9"}), log=msgs.append)
    wrapped = KD.load_kd_tensors
    assert wrapped is not frozen and wrapped.tests_masked
    got = KD.load_kd_tensors(lab, "targets.npz", stride=4)                   # the call of lite_train.py line 85
    assert calls == [(3, "targets.npz", {"stride": 4})]
    assert (got["t_w"][0] == 0).all() and np.isnan(got["t_mu"][0]).all() and np.isnan(got["t_var"][0]).all()
    assert (got["t_w"][1:] == 1).all() and np.array_equal(got["t_mu"][1:], kd["t_mu"][1:])
    assert got["cell_map"] is kd["cell_map"] and '"n_kd_masked": 1' in msgs[0]
    out = torch.randn(3, 2 * N_T, 4, 4, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(_nll(out, got), _nll(out, kd, slice(1, None)))
    LA.activate_rerun_kd_mask(flags=_flags(["c0"], set()))                   # idempotent: not wrapped twice
    assert KD.load_kd_tensors is wrapped
    with pytest.raises(TM.MaskError):                                        # a chip without a flag is refused
        KD.load_kd_tensors(pd.DataFrame({"naip_chip_id": ["c0", "zz"]}), "targets.npz")


def test_frozen_training_script_reads_the_loader_through_the_module():
    src = (paths.REPO_ROOT / "scripts/train/lite_train.py").read_text()
    assert "from sentinel.models.lite import distill as KD" in src and "KD.load_kd_tensors(" in src
    assert "from sentinel.models.lite.distill import load_kd_tensors" not in src


def test_rerun_wrapper_masks_train_and_logs_it_in_the_ledger(monkeypatch, tmp_path):
    rr = _script("scripts/train/national2b_rerun.py", "national2b_rerun")
    seen = []
    n2run = types.SimpleNamespace(_ssl_init_hook=lambda: seen.append("ssl"),
                                  _context_film_hook=lambda argv: seen.append("film"), TARGETS={"train": "entry.py",
                                                                                                "baselines": "b.py"})
    monkeypatch.setattr(labels2b, "activate_rerun", lambda: seen.append("labels"))
    monkeypatch.setattr(LA, "activate_rerun_kd_mask", lambda *a, **k: seen.append("kd_mask"))
    rr.activate(["train", "--config", "x"], n2run)
    assert seen == ["labels", "ssl", "film", "kd_mask"]
    seen.clear()
    rr.activate(["baselines", "--track", "national"], n2run)
    assert seen == ["labels", "ssl", "film"]                                 # no KD in the baselines
    assert rr.pop_budget(["train", "--gpu-budget-h", "1.25", "--x"]) == (["train", "--x"], 1.25)
    assert rr.pop_budget(["train"]) == (["train"], rr.RERUN_BUDGET_H) and rr.RERUN_BUDGET_H <= LA.JOB_CAP_H

    cfg = tmp_path / "x_national2.yaml"
    cfg.write_text("name: x_national2\n")
    jobs = []

    class Job:
        def __init__(self, name, block, budget):
            jobs.append((name, block, budget))

        def __enter__(self):
            seen.append("job_start")

        def __exit__(self, *exc):
            seen.append("job_end")

    entry = types.SimpleNamespace(main=lambda argv: seen.append("train") or 0)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(stamp, "require_stamp", lambda *a, **k: None)
    monkeypatch.setattr(LA, "gpu_job", Job)
    monkeypatch.setattr(rr, "_load", lambda rel, name: n2run if name == "national2_run" else entry)
    seen.clear()
    assert rr.main(["train", "--config", str(cfg), "--track", "national", "--gpu-budget-h", "1.0"]) == 0
    assert seen == ["labels", "ssl", "film", "kd_mask", "job_start", "train", "job_end"]
    assert jobs == [("rerun_x_national2b", "reruns", 1.0)]


# ------------------------------------------------------------------------------------------------ A2
STAMP = "20261111-010203"


def _run(root, name, stamp_=STAMP, seeds=LA.SEEDS):
    d = root / f"natstudent_{name}_{stamp_}"
    d.mkdir(parents=True)
    for s in seeds:
        (d / f"model_seed{s}.pt").write_bytes(f"{name} {s}".encode())
    return d


def _registered(root):
    got = []
    shas = LA.register_init_checkpoints(root, registry=[], log_row=lambda **kw: got.append(kw))
    assert len(got) == 1 and got[0]["experiment"] == LA.INIT_EXPERIMENT and got[0]["workstream"] == "M7-lite"
    assert got[0]["extra"]["sha256"] == shas and len(shas) == 6
    return [{"workstream": got[0]["workstream"], "experiment": got[0]["experiment"], "extra": got[0]["extra"]}]


def test_no_fallback_to_an_as_built_checkpoint(tmp_path):
    with pytest.raises(LA.LaunchError):
        LA.init_run_dir(True, tmp_path)                                      # nothing there
    as_built = _run(tmp_path, "mnv4s_raw_crop256_kd03_national2", "20261110-101010")
    _run(tmp_path, "mnv4s_raw_crop256_nokd_national2", "20261110-111111")
    for kd in (True, False):
        with pytest.raises(LA.LaunchError, match="No as-built checkpoint"):
            LA.init_checkpoint(kd, 0, runs_root=tmp_path, registry=[])
    _run(tmp_path, LA.INIT_NAME[True], seeds=(0, 1))                         # the re-run has not finished
    with pytest.raises(LA.LaunchError, match="found 0"):
        LA.init_checkpoint(True, 0, runs_root=tmp_path, registry=[])
    with pytest.raises(LA.LaunchError):
        LA.register_init_checkpoints(tmp_path, registry=[], log_row=lambda **kw: pytest.fail("no row"))
    (tmp_path / f"natstudent_{LA.INIT_NAME[True]}_{STAMP}" / "model_seed2.pt").write_bytes(b"x")
    assert LA.init_run_dir(True, tmp_path).name.endswith("national2b_" + STAMP)
    with pytest.raises(LA.LaunchError):                                      # the twin's re-run is still missing
        LA.init_run_dir(False, tmp_path)
    _run(tmp_path, LA.INIT_NAME[True], "20261112-000000")                    # two finished re-runs: not chosen between
    with pytest.raises(LA.LaunchError, match="found 2"):
        LA.init_run_dir(True, tmp_path)
    assert as_built.is_dir()


def test_warm_start_checkpoint_is_registered_and_hash_checked(tmp_path):
    d = _run(tmp_path, LA.INIT_NAME[True])
    dn = _run(tmp_path, LA.INIT_NAME[False])
    with pytest.raises(LA.LaunchError, match="not registered"):
        LA.init_checkpoint(True, 0, runs_root=tmp_path, registry=[])
    reg = _registered(tmp_path)
    for s in LA.SEEDS:
        assert LA.init_checkpoint(True, s, runs_root=tmp_path, registry=reg) == d / f"model_seed{s}.pt"
        assert LA.init_checkpoint(False, s, runs_root=tmp_path, registry=reg) == dn / f"model_seed{s}.pt"
    assert LA.init_checkpoint(True, 1, init_run=str(d), runs_root=tmp_path, registry=reg).parent == d
    for bad in ("natstudent_mnv4s_raw_crop256_kd03_national2_20261110-101010", dn.name):
        with pytest.raises(LA.LaunchError, match="loads no other checkpoint"):
            LA.init_checkpoint(True, 1, init_run=bad, runs_root=tmp_path, registry=reg)
    with pytest.raises(LA.LaunchError):
        LA.init_checkpoint(True, 3, runs_root=tmp_path, registry=reg)        # not a registered seed
    assert LA.register_init_checkpoints(tmp_path, registry=reg, log_row=lambda **kw: pytest.fail("second row")) \
        == reg[0]["extra"]["sha256"]
    (d / "model_seed1.pt").write_bytes(b"other weights")
    with pytest.raises(LA.LaunchError, match="differs from its registered sha256"):
        LA.init_checkpoint(True, 1, runs_root=tmp_path, registry=reg)
    assert LA.init_checkpoint(True, 0, runs_root=tmp_path, registry=reg).is_file()
    with pytest.raises(LA.LaunchError, match="already registered"):
        LA.register_init_checkpoints(tmp_path, registry=reg, log_row=lambda **kw: pytest.fail("second row"))


def test_v2_jobs_get_their_leader_from_the_launch_rule_only(monkeypatch, tmp_path):
    tr = _script("scripts/train/litev2_train.py", "litev2_train_a2")
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    _run(tmp_path / "lite", "mnv4s_raw_crop256_kd03_national2", "20261110-101010")       # the as-built run only
    for call in (lambda: tr.leader_state(True, 0), lambda: tr.leader_state(False, 0),
                 lambda: tr.init_name(True, [0, 1, 2]),
                 lambda: tr.leader_state(True, 0, "natstudent_mnv4s_raw_crop256_kd03_national2_20261110-101010")):
        with pytest.raises(LA.LaunchError):
            call()
    for rel in ("scripts/train/litev2_train.py", "scripts/train/litev2_comparators.py"):
        src = (paths.REPO_ROOT / rel).read_text()
        assert "_national2_" not in src and "natstudent_" not in src         # no run directory is named in a script
    src = (paths.REPO_ROOT / "scripts/train/litev2_train.py").read_text()
    assert src.count("torch.load(") == src.count("weights_only=True")
    assert "LA.init_checkpoint(" in src


# ------------------------------------------------------------------------------------------------ A4
def _registry(block):
    return [{"workstream": R.WORKSTREAM, "experiment": H.EXPERIMENT[block], "split": "val",
             "model": H.MODEL_PREFIX + c, "target": t} for c, ts in H.required_comparators(block).items() for t in ts]


def test_gate_covers_fit_and_needs_both_blocks():
    both = _registry("national_val") + _registry("val_p2")
    for c in H.COMPARATORS:
        assert not H.is_candidate(c)
        H.require_gate(c, [])                                               # a comparator is never held back
    for c in (*H.CANDIDATES, "anchor_lite_nokd", "A1_nokd", "anything_else"):
        assert H.is_candidate(c)
        H.require_gate(c, both)
        for part in ([], _registry("national_val"), _registry("val_p2"), both[:-1]):
            with pytest.raises(H.GateError):
                H.require_gate(c, part)


def test_no_candidate_fit_starts_before_the_comparators(monkeypatch, tmp_path):
    from sentinel.core import registry as REG

    tr = _script("scripts/train/litev2_train.py", "litev2_train_a4")

    class Reached(Exception):
        pass

    def boom(*a, **k):
        raise AssertionError("a row, a run directory or a checkpoint was touched before the gate")

    def reached(*a, **k):
        raise Reached

    monkeypatch.setattr(REG, "read_registry", lambda *a, **k: [])
    monkeypatch.setattr(stamp, "require_stamp", lambda *a, **k: None)
    monkeypatch.setattr(RW, "train_rows", boom)
    monkeypatch.setattr(RW, "eval_rows", boom)
    monkeypatch.setattr(LA, "init_checkpoint", boom)
    monkeypatch.setattr(LA, "gpu_job", boom)
    monkeypatch.setattr(paths, "run_dir", boom)
    for arm in H.CANDIDATES:
        for stage in ("fit", "eval", "both"):
            with pytest.raises(H.GateError):
                tr.main(["--arm", arm, "--stage", stage, "--device", "cpu"])
    with pytest.raises(H.GateError):
        tr.main(["--arm", "anchor_lite", "--no-kd", "--stage", "fit", "--device", "cuda"])
    a = argparse.Namespace(arm="anchor_lite", seeds=[0], no_kd=False, init_run=None, device="cpu", stage="fit")
    with pytest.raises(H.GateError):                                         # the stage itself, not only main()
        tr.stage_fit(a, {"net": "anchor"}, "anchor_lite", tmp_path)
    monkeypatch.setattr(paths, "run_dir", reached)                           # a comparator passes the gate
    with pytest.raises(Reached):
        tr.main(["--arm", "P3a_densenet", "--backbone-lr", "1e-4", "--stage", "fit", "--device", "cpu"])
    monkeypatch.setattr(REG, "read_registry", lambda *a, **k: _registry("national_val") + _registry("val_p2"))
    with pytest.raises(Reached):                                             # and a candidate once the rows exist
        tr.main(["--arm", "anchor_lite", "--stage", "fit", "--device", "cpu"])


# ------------------------------------------------------------------------------------------------ A7
class Clock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


def _lines(root):
    return [json.loads(x) for x in LA.ledger_path(root).read_text().splitlines()]


def test_gpu_job_logs_hours_and_refuses_bad_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assert LA.ledger_summary(tmp_path)["total_h"] == 0.0 and LA.CEILING_H == 45.0 and LA.JOB_CAP_H == 2.0
    with LA.gpu_job("rerun_a", "reruns", 1.5, runs_root=tmp_path, clock=Clock(1000.0, 1000.0 + 1800), alarm=False):
        mid = LA.ledger_summary(tmp_path)
        assert mid["total_h"] == 1.5 and mid["n_open"] == 1                  # a running job counts at its budget
    start, end = _lines(tmp_path)
    assert start["event"] == "start" and start["gpu"] == "5" and start["block"] == "reruns" and start["budget_h"] == 1.5
    assert end["event"] == "end" and end["status"] == "ok" and end["hours"] == pytest.approx(0.5)
    assert end["job_id"] == start["job_id"]
    with pytest.raises(ZeroDivisionError):
        with LA.gpu_job("p3a", "densenet", 2.0, runs_root=tmp_path, clock=Clock(0.0, 3600.0), alarm=False):
            1 / 0
    assert _lines(tmp_path)[-1]["status"] == "failed"                        # a failed job's hours are logged too
    s = LA.ledger_summary(tmp_path)
    assert s["finished_h"] == pytest.approx(1.5) and s["per_block_h"] == {"reruns": 0.5, "densenet": 1.0}
    assert s["n_jobs"] == 2 and s["n_open"] == 0
    n = len(_lines(tmp_path))
    for block, budget in (("ssl_pretrain", 1.0), ("reruns", 2.5), ("reruns", 0.0), ("reruns", -1.0)):
        with pytest.raises(LA.LaunchError):
            with LA.gpu_job("x", block, budget, runs_root=tmp_path, alarm=False):
                pytest.fail("the job started")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")                          # another user's GPU
    with pytest.raises(LA.LaunchError, match="another user"):
        with LA.gpu_job("x", "reruns", 1.0, runs_root=tmp_path, alarm=False):
            pytest.fail("the job started")
    for cvd in ("", "0,1", "GPU-1a2b"):                                      # not started through the lock helper
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
        with pytest.raises(LA.LaunchError, match="lock helper"):
            with LA.gpu_job("x", "reruns", 1.0, runs_root=tmp_path, alarm=False):
                pytest.fail("the job started")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(LA.LaunchError, match="lock helper"):
        with LA.gpu_job("x", "reruns", 1.0, runs_root=tmp_path, alarm=False):
            pytest.fail("the job started")
    assert len(_lines(tmp_path)) == n                                        # a refused job writes nothing


def test_gpu_job_refuses_to_pass_the_ceiling(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with open(LA.ledger_path(tmp_path), "w") as fh:
        for i in range(21):                                                  # 42 h finished
            fh.write(json.dumps({"event": "start", "job_id": f"j{i}", "name": "d", "block": "densenet",
                                 "budget_h": 2.0}) + "\n")
            fh.write(json.dumps({"event": "end", "job_id": f"j{i}", "name": "d", "hours": 2.0, "status": "ok"}) + "\n")
        fh.write(json.dumps({"event": "start", "job_id": "open", "name": "r", "block": "reruns",
                             "budget_h": 1.5}) + "\n")                       # + 1.5 h of a job without an end record
    assert LA.ledger_summary(tmp_path)["total_h"] == 43.5
    n = len(_lines(tmp_path))
    with pytest.raises(LA.CeilingError, match="does not start"):
        with LA.gpu_job("late", "v2_networks", 2.0, runs_root=tmp_path, alarm=False):
            pytest.fail("the job started")
    assert len(_lines(tmp_path)) == n
    with LA.gpu_job("fits", "v2_networks", 1.5, runs_root=tmp_path, clock=Clock(0.0, 3600.0), alarm=False):
        with pytest.raises(LA.CeilingError):                                 # 45 h are spoken for while it runs
            with LA.gpu_job("next", "tabpfn", 0.25, runs_root=tmp_path, alarm=False):
                pytest.fail("the job started")
    assert LA.ledger_summary(tmp_path)["total_h"] == 44.5                    # measured 1 h replaces the 1.5 h budget
    with LA.gpu_job("next", "tabpfn", 0.5, runs_root=tmp_path, clock=Clock(0.0, 60.0), alarm=False):
        pass
    assert issubclass(LA.CeilingError, LA.LaunchError)


def test_gpu_job_is_stopped_at_its_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(LA.CeilingError, match="budget"):
        with LA.gpu_job("slow", "s11_embed", 1.0 / 3600.0, runs_root=tmp_path):
            time.sleep(5)
    assert _lines(tmp_path)[-1]["status"] == "failed"


def test_every_gpu_entry_point_of_the_round_goes_through_the_ledger():
    root = paths.REPO_ROOT / "scripts/train"
    for f, n in (("litev2_train.py", 1), ("litev2_comparators.py", 1), ("litev2_tabpfn_memtest.py", 1),
                 ("national2b_rerun.py", 1)):
        assert (root / f).read_text().count("gpu_job(") >= n, f
    tr = _script("scripts/train/litev2_train.py", "litev2_train_a7")
    ns = lambda **k: argparse.Namespace(**{"arm": "anchor_lite", "no_kd": False, "seeds": [0, 1, 2], "stage": "both",   # noqa: E731
                                           "gpu_budget_h": None, **k})
    assert tr.gpu_block(ns()) == "v2_networks" and tr.gpu_block(ns(no_kd=True)) == "slot_twins"
    assert tr.gpu_block(ns(arm="P3a_densenet")) == "densenet" and tr.gpu_block(ns(arm="embed")) == "s11_embed"
    assert {tr.gpu_block(ns(arm=a)) for a in ("A1", "A3", "A4", "A5", "anchor_set")} == {"v2_networks"}
    for a in (ns(), ns(no_kd=True), ns(arm="P3a_densenet"), ns(arm="P3b_densenet_parity", seeds=[0], stage="fit"),
              ns(arm="embed"), ns(stage="eval")):
        assert 0 < tr.gpu_budget(a) <= LA.JOB_CAP_H
    assert tr.gpu_budget(ns(gpu_budget_h=1.25)) == 1.25


# ------------------------------------------------------------------------------------------------ one-shot steps
def test_steps_script_guards_the_registered_steps(monkeypatch, tmp_path, capsys):
    from sentinel.models.litev2 import checks as C

    st = _script("scripts/train/litev2_steps.py", "litev2_steps")
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    assert st.main(["ledger"]) == 0
    assert json.loads(capsys.readouterr().out)["total_h"] == 0.0 and not LA.ledger_path().exists()

    def refuse(what, *a, **k):
        raise stamp.StampError(what)

    def boom(*a, **k):
        raise AssertionError("the step ran before the stamp")

    monkeypatch.setattr(stamp, "require_stamp", refuse)
    monkeypatch.setattr(C, "e3a_turb_count", boom)
    monkeypatch.setattr(LA, "register_init_checkpoints", boom)
    for step in ("register-init", "e3a-count"):
        with pytest.raises(stamp.StampError):
            st.main([step])
    assert set(st.STEPS) == {"row-checks", "ledger", "register-init", "e3a-count"}
