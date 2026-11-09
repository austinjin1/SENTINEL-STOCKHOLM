# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 pre-test review fixes: primary statistics (C1), frozen invocation (C2), single pass (C3),
controls (C4/W8); re-review 2026-11-08 fixes (H2 chloride consistency C1, shared evaluable-onset
definition C2, power rule before Holm C3, fixed single-pass marker + registry fallback C4, content
data pins W1, --split refusal W5). Synthetic arrays only (tests)."""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.eval import ew_controls as C
from sentinel.models.sonde import guard

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sonde_events", ROOT / "scripts" / "train" / "sonde_events.py")
SE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SE)


def _rows(n_sites=20, n_days=400, seed=0, skill=True):
    rs = np.random.default_rng(seed)
    site = np.repeat([f"s{i}" for i in range(n_sites)], n_days)
    dates = np.tile(pd.date_range("2024-01-01", periods=n_days).to_numpy(), n_sites)
    z = rs.normal(size=site.size)
    y = (rs.random(site.size) < 1 / (1 + np.exp(4 - 1.5 * z))).astype(float)
    p = 1 / (1 + np.exp(4 - 1.5 * z)) if skill else np.full(site.size, y.mean())
    q = np.full(site.size, y.mean())
    basin = {f"s{i}": f"b{i % 7}" for i in range(n_sites)}
    return y, p, q, site, dates, basin


def test_bss_pvalue_and_holm():
    y, p, q, site, dates, basin = _rows()
    r = C.bss_pvalue(y, p, q, site=site, dates=dates, basin=basin, n_boot=300)
    assert r["bss"] > 0 and r["p_one_sided"] < 0.05
    y2, p2, q2, *_ = _rows(skill=False, seed=1)
    r2 = C.bss_pvalue(y2, p2 + 0.02, q2, site=site, dates=dates, basin=basin, n_boot=300)
    assert r2["p_one_sided"] > 0.5
    h = C.holm({"H1": 0.01, "H2": 0.04, "H4": np.nan})
    assert h["H1"]["reject"] and not h["H2"]["reject"] and not h["H4"]["reject"]   # 0.04 > 0.05/2
    assert abs(h["H1"]["p_holm"] - 0.03) < 1e-12
    with pytest.raises(ValueError):
        C.bss_pvalue(y, np.where(np.arange(y.size) == 0, np.nan, p), q, site=site, dates=dates, basin=basin)


def test_probability_controls():
    y, p, q, site, dates, _ = _rows()
    lp = C.label_block_permutation(y, p, q, site=site, dates=dates, n_perm=50)
    assert lp["bss_obs"] > lp["bss_null_p95"] and lp["p_one_sided"] <= 1 / 51 + 1e-12
    m = C.molchan(y, p)
    assert 0 < m["molchan_skill"] <= 1
    assert abs(C.molchan(y, np.random.default_rng(3).random(y.size))["molchan_skill"]) < 0.1
    t, ece = C.reliability(y, p)
    assert ece < 0.02 and t["n"].sum() == y.size
    rev = C.relative_economic_value(y, p)
    assert max(v for v in rev.values() if np.isfinite(v)) > 0
    dm = C.diebold_mariano((p - y) ** 2, (q - y) ** 2, dates)
    assert dm["dm_stat"] < 0 and dm["dm_p_one_sided"] < 0.05


def test_alarm_controls_and_placebo_cluster():
    from sentinel.eval.early_warning import Event

    idx = pd.date_range("2024-01-01", periods=300)
    frames, events = {}, {}
    for i in range(10):
        on = idx[50 + 20 * i]
        sc = np.full(300, -1.0)
        sc[50 + 20 * i - 2:50 + 20 * i] = 1.0            # alarms just before each real onset
        frames[f"s{i}"] = pd.DataFrame({"score": sc, "usable": True}, index=idx)
        events[f"s{i}"] = [Event(f"s{i}", on, on, on, 0.0, 24.0, False)]
    ra = C.random_alarm_pod(frames, events, 0.0, 3)
    assert ra["pod_obs"] == 1.0 and ra["pod_random_expected"] < 0.1
    cs = C.circular_shift(frames, events, 0.0, 3, n_surr=200)
    assert cs["p_one_sided"] < 0.05
    det = pd.DataFrame({"site": [f"s{i}" for i in range(10)], "evaluable": True, "detected": True})
    plc = pd.DataFrame({"site": np.repeat([f"s{i}" for i in range(10)], 5), "evaluable": True,
                        "detected": np.tile([True, False, False, False, False], 10)})
    r = C.placebo_cluster(det, plc, {f"s{i}": f"b{i % 4}" for i in range(10)}, n_boot=300)
    assert abs(r["diff"] - 0.8) < 1e-12 and r["diff_ci_low"] > 0 and r["p_one_sided"] < 0.01


def _args(**kw):
    base = dict(set=[], seed=None, smoke=False, temp_split=False, config=str(ROOT / "configs/sonde/events.yaml"),
                include_test_sites=True, eval_blocks="test", split=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_frozen_invocation_lock():
    cfg = SE.runner.load_config(ROOT / "configs/sonde/events.yaml")
    h = SE.frozen_config_hash(cfg)
    cal = {"config_hash": h}
    fr = {"invocation": {"config_hash": h, "quantile_models": {}, "lstm_checkpoint": None}}
    assert SE.check_frozen_invocation(_args(), cfg, cal, fr, {}, None) == "test"
    assert SE.check_frozen_invocation(_args(eval_blocks="test,test_b"), cfg, cal, fr, {}, None) == "test_ab"
    for bad in (dict(set=["ew.n_boot=10"]), dict(seed=3), dict(smoke=True), dict(include_test_sites=False),
                dict(eval_blocks="test_b"), dict(config=str(ROOT / "configs/sonde/common.yaml"))):
        with pytest.raises(guard.ProtectedDataError):
            SE.check_frozen_invocation(_args(**bad), cfg, cal, fr, {}, None)
    with pytest.raises(guard.ProtectedDataError):
        SE.check_frozen_invocation(_args(), cfg, cal, fr, {"gbm": "/x"}, None)
    cfg2 = {**cfg, "ew": {**cfg["ew"], "k_placebo": 5}}
    with pytest.raises(guard.ProtectedDataError):
        SE.check_frozen_invocation(_args(), cfg2, cal, fr, {}, None)


def test_single_pass_marker(tmp_path):
    reg = tmp_path / "registry.jsonl"
    m = SE.claim_single_pass("test", "a" * 64, ["x"], marker_dir=tmp_path, registry_path=reg)
    assert m.name == ".ew_scored_test.json" and json.loads(m.read_text())["pass"] == "test"
    with pytest.raises(guard.ProtectedDataError):
        SE.claim_single_pass("test", "a" * 64, ["x"], marker_dir=tmp_path, registry_path=reg)
    # review C4: a RE-WRITTEN freeze record (new sha) must not re-open the pass
    with pytest.raises(guard.ProtectedDataError):
        SE.claim_single_pass("test", "b" * 64, ["x"], marker_dir=tmp_path, registry_path=reg)
    SE.claim_single_pass("test_ab", "a" * 64, ["x"], marker_dir=tmp_path, registry_path=reg)   # own marker


def test_single_pass_legacy_marker_and_registry_fallback(tmp_path):
    d1, d2 = tmp_path / "m1", tmp_path / "m2"
    d1.mkdir()
    (d1 / ".ew_scored_test_0123456789ab.json").write_text("{}")          # legacy freeze-sha-keyed marker
    with pytest.raises(guard.ProtectedDataError):
        SE.claim_single_pass("test", "c" * 64, ["x"], marker_dir=d1, registry_path=tmp_path / "none.jsonl")
    # the legacy test marker must not block the pooled pass (regex, not prefix match)
    SE.claim_single_pass("test_ab", "c" * 64, ["x"], marker_dir=d1, registry_path=tmp_path / "none.jsonl")
    # registry fallback: marker deleted, but the pass_claimed row logged at the start of scoring refuses it
    reg = tmp_path / "registry.jsonl"
    m = SE.claim_single_pass("test", "a" * 64, ["x"], marker_dir=d2, registry_path=reg)
    n = SE.log_pass_rows("test", "test/sonde_split_x", d2, stage="claimed", freeze={"_sha256": "a" * 64},
                         registry_path=reg)
    assert n == 1 and SE.registry_has_pass("test", reg) and not SE.registry_has_pass("test_ab", reg)
    m.chmod(0o644)
    m.unlink()
    with pytest.raises(guard.ProtectedDataError, match="registry"):
        SE.claim_single_pass("test", "d" * 64, ["x"], marker_dir=d2, registry_path=reg)


def test_split_refused_before_marker():
    cfg = SE.runner.load_config(ROOT / "configs/sonde/events.yaml")
    h = SE.frozen_config_hash(cfg)
    cal = {"config_hash": h}
    fr = {"invocation": {"config_hash": h, "quantile_models": {}, "lstm_checkpoint": None}}
    with pytest.raises(guard.ProtectedDataError, match="--split"):
        SE.check_frozen_invocation(_args(split="docs/preregistration/sonde_split_dev.json"), cfg, cal, fr, {}, None)
    assert SE.check_frozen_invocation(_args(split=str(ROOT / SE.DEFAULT_SPLIT)), cfg, cal, fr, {}, None) == "test"
    with pytest.raises(guard.ProtectedDataError, match="split_sha256"):
        SE.check_frozen_invocation(_args(), cfg, cal, {**fr, "split_sha256": "0" * 64}, {}, None)


def test_evaluable_onsets_one_definition_for_gate_pod_and_controls():
    """Review C2: onset at the block start, after the last issue day + L, with no usable rows, censored
    -> not evaluable; the gate, POD, placebo and alarm controls all use the same set."""
    from sentinel.eval import ew_scoring as S
    from sentinel.eval.early_warning import Event

    idx = pd.date_range("2024-01-01", "2024-01-31")
    usable = np.ones(idx.size, bool)
    usable[11:14] = False                                   # 2024-01-12 .. 01-14 unusable
    f = pd.DataFrame({"score": np.where(np.arange(idx.size) % 2 == 0, 1.0, -1.0), "usable": usable}, index=idx)

    def ev(d, cens=False):
        t = pd.Timestamp(d)
        return Event("s0", t, t, t, 0.0, 24.0, cens)

    evs = [ev("2024-01-01"),            # block start: no issue row before it
           ev("2024-01-15"),            # window [01-12, 01-15) has no usable row
           ev("2024-01-20"),            # evaluable
           ev("2024-01-25", cens=True),  # censored
           ev("2024-02-02"),            # last issue + 2 d (<= L = 3): evaluable
           ev("2024-02-05")]            # last issue + 5 d: outside (first, last + L]
    frames, events = {"s0": f}, {"s0": evs}
    got = S.evaluable_onsets(frames, events, 3)
    assert [str(e.onset.date()) for e in got["s0"]] == ["2024-01-20", "2024-02-02"]
    gate = S.count_evaluable_onsets(frames, events, 3, {"s0": "b0"})
    assert gate == {"n_evaluable_onsets": 2, "n_sites_with_onsets": 1, "n_basin_groups_with_onsets": 1}
    plc = S.placebo_comparison(frames, events, 0.0, 3, k_per_onset=2, seed=0)
    assert plc["events_evaluable"] == 2
    assert C.random_alarm_pod(frames, events, 0.0, 3)["n_events"] == 2
    assert C.circular_shift(frames, events, 0.0, 3, n_surr=20, min_shift=5)["n_events"] == 2


def test_underpowered_hazards_enter_holm_with_p_one():
    """Review C3: underpowered members cannot step down H1's Holm threshold."""
    res = {"H1_hypoxia": {"p_one_sided": 0.02, "n_evaluable_onsets": 60, "n_basin_groups_with_onsets": 12},
           "H2_chloride": {"p_one_sided": 0.0055, "n_evaluable_onsets": 1, "n_basin_groups_with_onsets": 1},
           "H4_nitrate": {"p_one_sided": 0.01, "n_evaluable_onsets": 5, "n_basin_groups_with_onsets": 2}}
    fam = ("H1_hypoxia", "H2_chloride", "H4_nitrate")
    raw = C.holm({h: r["p_one_sided"] for h, r in res.items()})
    assert all(raw[h]["reject"] for h in fam)                 # the flaw: raw p-values reject everything
    d = C.primary_decision(res, fam)
    assert d["H2_chloride"]["underpowered"] and d["H4_nitrate"]["underpowered"]
    assert d["H2_chloride"]["p_holm_input"] == 1.0 and d["H2_chloride"]["p_one_sided_descriptive"] == 0.0055
    assert not any(d[h]["reject_holm"] or d[h]["claim_allowed"] for h in fam)   # 0.02 > 0.05 / 3
    res["H1_hypoxia"]["p_one_sided"] = 0.01
    d = C.primary_decision(res, fam)
    assert d["H1_hypoxia"]["claim_allowed"] and abs(d["H1_hypoxia"]["p_holm"] - 0.03) < 1e-12
    # >= 50 onsets but a single basin group: degenerate outer bootstrap -> underpowered
    one = {"H1_hypoxia": {"p_one_sided": 0.001, "n_evaluable_onsets": 80, "n_basin_groups_with_onsets": 1}}
    d = C.primary_decision(one, fam)
    assert d["H1_hypoxia"]["underpowered"] and not d["H1_hypoxia"]["claim_allowed"]
    assert d["H2_chloride"]["p_holm_input"] == 1.0 and not d["H2_chloride"]["claim_allowed"]   # missing result


def test_snapshot_content_pin(tmp_path, monkeypatch):
    d = tmp_path / "chloride_derived"
    (d / "sites").mkdir(parents=True)
    (d / "regressions.parquet").write_bytes(b"r")
    (d / "sites" / "a.parquet").write_bytes(b"1")
    (d / "notes.txt").write_text("not read by the loader")
    h0 = SE.snapshot_content_sha256("chloride_derived", d)
    assert h0["n_content_files"] == 2
    (d / "notes.txt").write_text("changed")
    assert SE.snapshot_content_sha256("chloride_derived", d)["content_sha256"] == h0["content_sha256"]
    (d / "sites" / "a.parquet").write_bytes(b"2")             # same size, new content
    assert SE.snapshot_content_sha256("chloride_derived", d)["content_sha256"] != h0["content_sha256"]
    with pytest.raises(guard.ProtectedDataError, match="no pin"):
        SE.verify_snapshots({}, strict=True, log=print, pins={})


def test_chloride_consistency_check(monkeypatch):
    """Review C1: derived chloride built from another SC snapshot is refused."""
    sites = ["s1", "s2"]
    variables = ["spcond_uscm_mean", "spcond_uscm_max", "cl_mgl_mean", "cl_mgl_4dmean", "cl_mgl_max"]
    sc = np.array([[100.0, 200.0, np.nan, 300.0], [50.0, 60.0, 70.0, 80.0]])
    y = np.full((2, 4, 5), np.nan, np.float32)
    y[:, :, 0], y[:, :, 1] = sc, sc + 10
    y[0, :, 2], y[0, :, 4] = 2 + 0.5 * sc[0], 2 + 0.5 * (sc[0] + 10)    # s1 eligible; s2 has no regression
    cube = types.SimpleNamespace(sites=sites, variables=variables, y=y, var_index=variables.index)
    data = types.SimpleNamespace(cube=cube)
    regs = pd.DataFrame({"a": [2.0], "b": [0.5], "r2": [0.95], "n_pairs": [30]}, index=pd.Index(["s1"], name="site_no"))
    monkeypatch.setattr(SE.common, "chloride_regressions", lambda: regs)
    assert SE.check_chloride_consistency(data)["chloride_sites_checked"] == 1
    y[0, 3, 2] = 2 + 0.5 * 290.0                              # stale value from a superseded SC pull
    with pytest.raises(guard.ProtectedDataError, match="inconsistent"):
        SE.check_chloride_consistency(data)
    y[0, 3, 2] = 2 + 0.5 * 300.0
    y[0, 2, 2] = 5.0                                          # value where SC is now missing
    with pytest.raises(guard.ProtectedDataError):
        SE.check_chloride_consistency(data)


def test_placebo_cluster_empty_cell():
    empty = pd.DataFrame(columns=["evaluable", "detected"])
    assert np.isnan(C.placebo_cluster(empty, empty, {})["diff"])
