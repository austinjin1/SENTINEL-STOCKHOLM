# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Seed units, arm mixtures, validation scoring and the selection / kill rules of architecture round v2
(orbit_arch_v2_amendment.md §3.2, §3.6, §3.7). Synthetic networks and rows only; writes under tmp_path only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.models.orbit import data as D
from sentinel.models.orbit import evaluate as E
from sentinel.models.orbit_v2 import train as TR
from sentinel.models.orbit_v2 import units as U

N, K = 240, 5


def _base(seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"row_id": [f"r{i}" for i in range(N)], "block": [f"b{i % 24}" for i in range(N)]})
    for t in D.TARGETS:
        df[t] = 10 ** r.normal(0.6, 0.4, N)
    df.loc[: N // 2, "tss"] = np.nan
    df.loc[5:, "turbidity"] = np.nan                         # fewer than 20 labelled rows in every cell
    masks = {"valxval_lake": np.arange(N) < 150, "valxany_lake": np.arange(N) < 200,
             "fit": np.arange(N) >= 220}
    return df, masks


def _write_arm(root, arm, df, *, noise=0.1, seed=0, targets=None, rows=None, **meta):
    targets = list(targets or D.TARGETS)
    rows = np.arange(200) if rows is None else rows
    y = np.stack([np.log10(df[t].to_numpy(float)) if t in df else np.zeros(len(df)) for t in targets], 1)[rows]
    y = np.nan_to_num(y, nan=0.5)
    for u in TR.UNITS:
        for i in TR.INITS:
            r = np.random.default_rng(1000 * seed + TR.network_seed(u, i))
            d = U.network_dir(root, arm, u, i)
            d.mkdir(parents=True)
            perm = r.permutation(len(rows))                  # every network stores its rows in another order
            mu = y[:, :, None] + noise * r.standard_normal((len(rows), len(targets), K))
            w = r.dirichlet(np.ones(K), (len(rows), len(targets)))
            sg = r.uniform(0.05, 0.3, mu.shape)
            np.savez_compressed(d / "preds.npz", row_id=df.row_id.to_numpy()[rows][perm].astype(str),
                                w=w[perm].astype(np.float32), mu=mu[perm].astype(np.float32),
                                sg=sg[perm].astype(np.float32))
            m = {"arm": arm, "unit": u, "init": i, "seed": TR.network_seed(u, i), "targets": targets,
                 "arm_config_hash": "h" * 64, "arm_file_sha256": "f" * 64, "data": {"version": "dev_v5m", "sha256": {"table.parquet": "d" * 64}},
                 "prediction_rows": "development rows outside the fit cell", "smoke": False, **meta}
            (d / "network.json").write_text(json.dumps(m))
            (d / "DONE").write_text("x\n")
    return root


def test_assemble_puts_the_nine_networks_on_common_rows(tmp_path):
    df, _ = _base()
    asm = U.assemble(_write_arm(tmp_path, "a0", df), "a0")
    assert len(asm["nets"]) == 9 and asm["targets"] == list(D.TARGETS) and len(asm["row_id"]) == 200
    y = np.log10(df["chla"].to_numpy())
    pos = pd.Series(np.arange(200), index=asm["row_id"])
    for mix in asm["nets"].values():                         # rows were re-ordered to the common order
        assert mix[0].shape == (200, 4, K)
        assert np.abs((mix[0] * mix[1]).sum(-1)[pos.loc[df.row_id[:200]].to_numpy(), 0] - y[:200]).max() < 0.4


@pytest.mark.parametrize("damage, msg", [
    (lambda d: (d / "DONE").unlink(), "not complete"),
    (lambda d: (d / "network.json").write_text(json.dumps({**json.loads((d / "network.json").read_text()),
                                                            "smoke": True})), "smoke-test network"),
    (lambda d: (d / "network.json").write_text(json.dumps({**json.loads((d / "network.json").read_text()),
                                                            "seed": 7})), "does not describe"),
    (lambda d: (d / "network.json").write_text(json.dumps({**json.loads((d / "network.json").read_text()),
                                                            "arm_config_hash": "x"})), "differs from the first"),
    (lambda d: (d / "network.json").write_text(json.dumps({**json.loads((d / "network.json").read_text()),
                                                            "data": {"version": "dev_v5"}})), "differs from the first"),
])
def test_assemble_refuses_an_incomplete_or_inconsistent_arm(tmp_path, damage, msg):
    df, _ = _base()
    _write_arm(tmp_path, "a1_c2", df)
    damage(U.network_dir(tmp_path, "a1_c2", 1, 2))
    with pytest.raises(U.UnitError, match=msg):
        U.assemble(tmp_path, "a1_c2")


def test_assemble_refuses_other_rows_and_a_wrong_folder(tmp_path):
    df, _ = _base()
    _write_arm(tmp_path, "a0", df)
    d = U.network_dir(tmp_path, "a0", 2, 0)
    z = dict(np.load(d / "preds.npz"))
    z["row_id"] = np.where(z["row_id"] == "r3", "zz", z["row_id"])
    np.savez_compressed(d / "preds.npz", **z)
    with pytest.raises(U.UnitError, match="prediction rows differ"):
        U.assemble(tmp_path, "a0")
    with pytest.raises(U.UnitError, match="not complete"):
        U.assemble(tmp_path, "a1_c1")                        # no such arm under the root
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a1_c1").symlink_to(tmp_path / "a0")   # networks of another arm under this arm's name
    with pytest.raises(U.UnitError, match="does not describe arm a1_c1"):
        U.assemble(tmp_path / "x", "a1_c1")


def test_unit_and_arm_mixtures_are_equal_weight_averages(tmp_path):
    df, _ = _base()
    asm = U.assemble(_write_arm(tmp_path, "a0", df), "a0")
    mem = U.members(asm)
    assert list(mem)[:4] == ["arm", "unit0", "unit1", "unit2"] and len(mem) == 13
    assert [lev for lev, _ in mem.values()].count("network") == 9
    w, mu, sg = mem["unit1"][1]
    assert w.shape == (200, 4, 3 * K) and np.allclose(w.sum(-1), 1.0)
    assert np.allclose(w[..., K:2 * K], asm["nets"][(1, 1)][0] / 3) and np.allclose(mu[..., :K], asm["nets"][(1, 0)][1])
    wa, mua, sga = mem["arm"][1]
    assert wa.shape == (200, 4, 9 * K) and np.allclose(wa.sum(-1), 1.0)
    assert np.allclose(wa[..., :K], asm["nets"][(0, 0)][0] / 9)                 # every network has weight 1/9
    # the point prediction is the median of the averaged mixture, not the average of the medians
    x = U.medians(mem["arm"][1])
    assert x.shape == (200, 4) and np.allclose(TR.mixture_cdf(mem["arm"][1], x), 0.5, atol=1e-4)
    same = {k: asm["nets"][(0, 0)] for k in asm["nets"]}
    assert np.allclose(U.medians(U.arm_mixture(same)), U.medians(asm["nets"][(0, 0)]), atol=1e-5)


def test_score_uses_the_v1_cells_and_common_rows(tmp_path):
    df, masks = _base()
    asm = U.assemble(_write_arm(tmp_path, "a1_c3", df, targets=list(D.TARGETS) + ["tp"]), "a1_c3")
    v1 = {"chla": pd.Series(np.log10(df["chla"].to_numpy()) + 0.2, index=df.row_id)}
    v1["chla"].iloc[:10] = np.nan                            # v1 lacks 10 rows: they leave the common set
    tab, rows = U.score(asm, df, masks, other={"v1": v1}, n_boot=50)
    assert set(tab["cell"]) == {"valxval_lake", "valxany_lake"} and "turbidity" not in set(tab["target"])
    assert "tp" not in set(tab["target"])                    # base targets only
    c = tab[(tab.cell == "valxval_lake") & (tab.target == "chla")].set_index("model")
    assert len(c) == 14 and (c["n"] == 140).all() and c.loc["v1", "level"] == "other"
    assert c.loc["a1_c3:arm", "level"] == "arm" and (c["level"] == "unit").sum() == 3 and (c["level"] == "network").sum() == 9
    assert c.loc["v1", "mdsa"] == pytest.approx(100 * (10 ** 0.2 - 1), rel=1e-6)
    assert c.loc["a1_c3:arm", "mdsa"] < c.loc["v1", "mdsa"]
    assert c.loc["a1_c3:arm", "mdsa_ci_low"] <= c.loc["a1_c3:arm", "mdsa"] <= c.loc["a1_c3:arm", "mdsa_ci_high"]
    s = tab[(tab.cell == "valxval_lake") & (tab.target == "secchi")]
    assert (s["n"] == 150).all() and "v1" not in set(s["model"])
    # the stored rows reproduce the table, and the paired difference uses them
    r = rows[(rows.cell == "valxval_lake") & (rows.target == "chla")]
    assert len(r) == 140 and E.mdsa_log10(r["a1_c3:arm"] - r["y"]) == pytest.approx(c.loc["a1_c3:arm", "mdsa"])
    p = U.paired(rows, "valxval_lake", "chla", "a1_c3:arm", "v1", n_boot=50)
    assert p["diff"] == pytest.approx(c.loc["a1_c3:arm", "mdsa"] - c.loc["v1", "mdsa"]) and p["diff"] < 0 and p["n"] == 140
    only_arm, _ = U.score(asm, df, masks, levels=("arm",), cells=("valxval_lake",), n_boot=20)
    assert set(only_arm["model"]) == {"a1_c3:arm"}
    # a cell row without a prediction is an error, never a silently smaller cell
    short = U.assemble(_write_arm(tmp_path / "s", "a0", df, rows=np.arange(100)), "a0")
    with pytest.raises(U.UnitError, match="have no prediction"):
        U.score(short, df, masks, n_boot=20)


def test_evaluate_arms_applies_the_rules_only_with_complete_inputs(tmp_path):
    df, masks = _base()
    noise = {"a0": 0.5, "a1_c3": 0.45, "a1_c2": 0.05, "a1_c1": 0.3}
    asms = {a: U.assemble(_write_arm(tmp_path, a, df, noise=n, seed=k), a) for k, (a, n) in enumerate(noise.items())}
    bias = lambda b: {t: pd.Series(np.log10(df[t].to_numpy()) + b, index=df.row_id) for t in ("chla", "secchi")}  # noqa: E731
    res = U.evaluate_arms(asms, df, masks, bias(0.3), n_boot=30)
    assert set(res["scores"]["arm"]) == set(noise) and not res["notes"]
    assert res["chla"]["a1_c2"] == min(res["chla"].values()) and res["kill"]["a1_configuration"] == "a1_c2"
    assert res["kill"]["passed"] and res["kill"]["v1"]["chla"] == pytest.approx(100 * (10 ** 0.3 - 1), rel=1e-6)
    assert res["kill"]["paired"]["secchi"]["n"] == 150 and res["selection"]["v2"] == "a1_c2"
    assert set(res["rows"]) == set(noise) | {"kill"}
    weak = U.evaluate_arms(asms, df, masks, bias(0.001), n_boot=30)          # v1 almost exact: A1 fails the gate
    assert not weak["kill"]["passed"] and weak["selection"]["v2"] == "a0" and weak["selection"]["candidates"] == ["a0"]
    part = U.evaluate_arms({a: asms[a] for a in ("a0", "a1_c2")}, df, masks, bias(0.3), n_boot=30)
    assert part["kill"] is None and part["selection"] is None and len(part["notes"]) == 2
    nov1 = U.evaluate_arms(asms, df, masks, None, n_boot=30)
    assert nov1["kill"] is None and nov1["selection"] is None
    with pytest.raises(U.UnitError, match="v1 has no predictions"):
        U.evaluate_arms(asms, df, masks, {"chla": bias(0.3)["chla"]}, n_boot=30)
    with pytest.raises(U.UnitError, match="unknown arm"):
        U.evaluate_arms({"a2": asms["a0"]}, df, masks, None, n_boot=30)


def test_kill_criterion_needs_two_points_on_both_targets():
    v1 = {"chla": 60.0, "secchi": 40.0}
    assert U.kill_criterion({"chla": 58.0, "secchi": 38.0}, v1)["passed"]                # exactly 2 points: passes
    assert not U.kill_criterion({"chla": 58.01, "secchi": 30.0}, v1)["passed"]
    assert not U.kill_criterion({"chla": 50.0, "secchi": 38.5}, v1)["passed"]
    assert not U.kill_criterion({"chla": 61.0, "secchi": 41.0}, v1)["passed"]
    k = U.kill_criterion({"chla": 55.0, "secchi": 36.0, "tss": 1.0}, v1)
    assert k["gain_points"] == {"chla": 5.0, "secchi": 4.0} and k["margin_points"] == 2.0
    for bad in ({"chla": 55.0}, {"chla": 55.0, "secchi": float("nan")}):
        with pytest.raises(U.UnitError, match="kill criterion"):
            U.kill_criterion(bad, v1)


def test_a1_best_and_selection_rule():
    assert U.a1_best({"a0": 1.0, "a1_c1": 50.0, "a1_c2": 51.0, "a1_c3": 52.0}) == "a1_c1"
    assert U.a1_best({"a1_c1": 50.0, "a1_c2": 50.0, "a1_c3": 50.01}) == "a1_c2"         # exact tie: the simpler one
    with pytest.raises(U.UnitError, match="without a validation score"):
        U.a1_best({"a1_c1": 50.0, "a1_c2": 51.0})
    sc = {"a0": 55.0, "a1_c3": 54.0, "a1_c2": 52.0, "a1_c1": 51.97}
    assert U.select_v2(sc, True)["v2"] == "a1_c2" and U.select_v2(sc, True)["tied"] == ["a1_c2", "a1_c1"]
    assert U.select_v2(sc, False)["v2"] == "a0" and U.select_v2(sc, False)["candidates"] == ["a0"]
    assert U.select_v2({**sc, "a0": 52.01}, True)["v2"] == "a0"                          # within 0.05 of the lowest
    assert U.select_v2({**sc, "a1_c1": 51.9}, True)["v2"] == "a1_c1"
    assert U.select_v2({"a0": 55.0}, False)["v2"] == "a0"
    with pytest.raises(U.UnitError, match="needs the A0"):
        U.select_v2({"a1_c1": 1.0, "a1_c2": 1.0, "a1_c3": 1.0}, True)
    with pytest.raises(U.UnitError, match="have no validation score"):
        U.select_v2({"a0": 55.0, "a1_c1": 50.0}, True)
    with pytest.raises(U.UnitError, match="unknown arm"):
        U.select_v2({"a0": 55.0, "a2": 50.0}, False)
    with pytest.raises(U.UnitError, match="not finite"):
        U.select_v2({"a0": float("nan")}, False)


def test_per_seed_rule_applies_to_seed_units():
    assert U.units_all_better([50.0, 51.0, 52.0], 52.5)
    assert not U.units_all_better([50.0, 51.0, 52.5], 52.5)                              # equal is not better
    with pytest.raises(U.UnitError):
        U.units_all_better([50.0, 51.0], 52.5)
    with pytest.raises(U.UnitError):
        U.units_all_better([50.0, 51.0, float("nan")], 52.5)


def test_r18a_the_stricter_kill_outcome_stands():
    a1 = {"chla": 50.0, "secchi": 40.0}
    cand = dict(U.kill_criterion(a1, {"chla": 53.0, "secchi": 42.5}), a1_configuration="a1_c2")       # passes
    froz = dict(U.kill_criterion(a1, {"chla": 51.5, "secchi": 42.5}), a1_configuration="a1_c2")       # fails
    assert cand["passed"] and not froz["passed"]
    k = U.kill_stricter(cand, froz)
    assert not k["passed"] and k["v1"] == froz["v1"] and k["gain_points"] == froz["gain_points"]
    assert k["r18a"] == {"passed_vs_candidate": True, "passed_vs_frozen": False,
                         "gain_points_vs_candidate": cand["gain_points"],
                         "gain_points_vs_frozen": froz["gain_points"]}
    assert not U.kill_stricter(froz, cand)["passed"]                  # a pass against the frozen v1 does not undo it
    assert U.kill_stricter(cand, cand)["passed"] and not U.kill_stricter(froz, froz)["passed"]
    assert froz["passed"] is False and "r18a" not in froz             # inputs are not changed
    with pytest.raises(U.UnitError, match="different A1 configurations"):
        U.kill_stricter(cand, dict(froz, a1_configuration="a1_c3"))
    with pytest.raises(U.UnitError, match="two kill-criterion records"):
        U.kill_stricter(cand, None)


def test_r17_per_seed_rule_and_label_sentence():
    units = [50.0, 51.0, 52.0]
    nets = [49.0, 52.4, 53.0, 51.0, 52.0, 54.0, 52.5, 50.5, 55.0]
    r = U.per_seed_rule(units, nets, {"gbm": 52.5, "mdn_b+gbm": 53.5, "knn": 60.0})
    assert r["passed"] and r["k_networks"] == 5 and r["n_networks"] == 9 and r["best_comparator"] == 52.5
    assert r["sentence"] == ("the per-seed rule was applied to 3-network seed units; 5 of 9 single networks "
                             "individually beat every comparator")
    r = U.per_seed_rule(units, nets, {"gbm": 52.5, "mdn_b+gbm": 51.9})         # one unit is not lower than one
    assert not r["passed"] and r["k_networks"] == 3 and "3 of 9" in r["sentence"]
    assert U.per_seed_rule([49.0] * 3, [49.0] * 9, {"gbm": 49.0})["k_networks"] == 0       # equal is not better
    for bad in ((units, nets[:8], {"gbm": 52.5}), (units, nets, {}), (units, nets, {"gbm": float("nan")}),
                (units[:2], nets, {"gbm": 52.5}), (units, [float("nan")] * 9, {"gbm": 52.5})):
        with pytest.raises(U.UnitError):
            U.per_seed_rule(*bad)


def test_attribution_names_every_change_between_two_models():
    at = lambda a, b, off=True, amp=False: U.attribution(a, b, v1_ctx_off=off, v1_amp=amp)
    assert at("a1_c3", "a0") == ["land ring", "multi-target heads", "added rows", "day of year"]
    assert at("a1_c3", "a0", off=False) == ["land ring", "multi-target heads", "added rows"]
    assert at("a1_c2", "a0", off=False) == ["context set", "land ring", "multi-target heads", "added rows"]
    assert at("a1_c1", "a0") == ["context set", "land ring", "multi-target heads", "added rows",
                                 "latitude and longitude", "day of year"]
    assert all("added rows" in at(a, "a0") for a in U.A1_CONFIGS)                            # C-6
    assert at("a0", "a1_c1") == at("a1_c1", "a0")
    assert at("a0", "v1") == ["seed units with EMA weights", "fp16 AMP"]                     # C-7
    assert at("v1", "a0", amp=True) == ["seed units with EMA weights"]
    assert at("a1_c2", "v1") == at("a1_c2", "a0") + at("a0", "v1")
    assert at("a1_c2", "a1_c3") == ["context set"] and at("a1_c1", "a1_c2") == ["latitude and longitude"]
    assert at("a1_c1", "a1_c3") == ["context set", "latitude and longitude"]
    for bad in (("a0", "a0"), ("a2", "a0"), ("v1", "v2")):
        with pytest.raises(U.UnitError):
            at(*bad)


# ----------------------------------------------------------------------------------------- script
def _script():
    import importlib.util
    from pathlib import Path

    f = Path(__file__).resolve().parents[1] / "scripts" / "train" / "orbitv2_units.py"
    spec = importlib.util.spec_from_file_location("orbitv2_units_script", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_is_refused_before_the_stamp_and_reads_nothing(tmp_path, monkeypatch):
    from sentinel.core import paths
    from sentinel.models.orbit_v2 import stamp as ST

    mod = _script()
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(ST, "stamp_status", lambda *a, **k: {"stamped": False, "amendment_sha256": "0" * 64,
                                                             "row_id": None, "timestamp": None, "n_rows": 0,
                                                             "rulings_sha256": "0" * 64, "reason": "test"})
    monkeypatch.setattr(mod.PL, "load_dev", lambda *a, **k: pytest.fail("development rows were loaded"))
    monkeypatch.setattr(mod.U, "assemble", lambda *a, **k: pytest.fail("predictions were read"))
    with pytest.raises(ST.StampError, match="refused before the lead's stamp"):
        mod.main(["--arms", "a0", "--register"])
    assert not (tmp_path / "runs").exists()


def test_script_registers_scores_kill_and_selection(tmp_path, monkeypatch):
    import sentinel.core.registry as R

    mod = _script()
    df, masks = _base()
    noise = {"a0": 0.5, "a1_c3": 0.45, "a1_c2": 0.05, "a1_c1": 0.3}
    asms = {a: U.assemble(_write_arm(tmp_path, a, df, noise=n, seed=k), a) for k, (a, n) in enumerate(noise.items())}
    v1 = {t: pd.Series(np.log10(df[t].to_numpy()) + 0.3, index=df.row_id) for t in ("chla", "secchi")}
    res = U.evaluate_arms(asms, df, masks, v1, n_boot=20)
    logged = []
    monkeypatch.setattr(R, "log_result", lambda **kw: logged.append(kw))
    n = mod.register(res, {a: {"arm": a} for a in noise}, tmp_path / "out", "dev_v5m", 20, "stamp123")
    assert n == len(logged) and {r["experiment"] for r in logged} == {"orbit_v2_val_dev_v5m"}
    assert all("stamp123" in r["notes"] and "NOT a claim" in r["notes"] for r in logged)
    names = {r["model"] for r in logged if r["metric"] == "mdsa"}
    assert {"a0", "a0@unit0", "a0@u2i1", "a1_c2", "a1_c2@unit2"} <= names
    nets = [r for r in logged if r["metric"] == "mdsa" and r["extra"]["level"] == "network"]
    assert nets and all(r["split"] == "valxval_lake@dev_v5m" for r in nets)       # networks: primary cell only
    kill = [r for r in logged if r["metric"] == "a1_kill_criterion_passed"]
    assert len(kill) == 1 and kill[0]["model"] == "a1_c2" and kill[0]["value"] == 1.0
    assert kill[0]["extra"]["v1_reference"] is None and kill[0]["extra"]["r18a"] is None
    assert {r["target"] for r in logged if r["metric"] == "mdsa_diff_vs_v1"} == {"chla", "secchi"}
    sel = [r for r in logged if r["metric"] == "v2_selected"]
    assert len(sel) == 1 and sel[0]["model"] == "a1_c2" and sel[0]["extra"]["a1_passed"] is True
    assert not (tmp_path / "out").exists()                   # register() writes registry rows only


def test_script_names_the_v1_reference_by_sha_and_kind(tmp_path, monkeypatch):
    import hashlib

    from sentinel.models.orbit import guard as G1

    mod = _script()
    cand = tmp_path / "orbit_freeze_candidate.json"
    cand.write_text("{}")
    r = mod.v1_reference(str(cand))
    assert r == {"file": str(cand.resolve()), "sha256": hashlib.sha256(b"{}").hexdigest(), "kind": "candidate"}
    frozen = tmp_path / "FREEZE_orbit.json"
    frozen.write_text('{"x": 1}')
    assert mod.v1_reference(str(frozen))["kind"] == "candidate"        # only the lead's file is the frozen reference
    monkeypatch.setattr(G1, "FREEZE_FILE", frozen)
    assert mod.v1_reference(str(frozen))["kind"] == "frozen"
    with pytest.raises(SystemExit, match="no such file"):
        mod.v1_reference(str(tmp_path / "missing.json"))


def test_script_r18a_arguments_are_checked_before_any_row_is_read(tmp_path, monkeypatch):
    from sentinel.core import paths
    from sentinel.models.orbit import guard as G1
    from sentinel.models.orbit_v2 import stamp as ST

    mod = _script()
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(ST, "require_stamp", lambda *a, **k: {"row_id": "stamp123"})
    monkeypatch.setattr(mod.PL, "load_dev", lambda *a, **k: pytest.fail("development rows were loaded"))
    monkeypatch.setattr(mod.U, "assemble", lambda *a, **k: pytest.fail("predictions were read"))
    monkeypatch.setattr(mod.ARM, "resolve_config", lambda *a, **k: pytest.fail("configurations were resolved"))
    cand = tmp_path / "cand.json"
    cand.write_text("{}")
    frozen = tmp_path / "FREEZE_orbit.json"
    frozen.write_text("{}")
    monkeypatch.setattr(G1, "FREEZE_FILE", frozen)
    with pytest.raises(SystemExit, match="go together"):
        mod.main(["--arms", "a0", "--v1-compare", "run_x"])
    with pytest.raises(SystemExit, match="go together"):
        mod.main(["--arms", "a0", "--v1-reference", str(cand)])
    first = tmp_path / "units.json"
    first.write_text(json.dumps({"v1_reference": {"kind": "candidate"}, "kill": {"passed": True}}))
    with pytest.raises(SystemExit, match="recomputation against FREEZE_orbit.json"):
        mod.main(["--arms", "a0", "--v1-compare", "run_x", "--v1-reference", str(cand), "--kill-first", str(first)])
    with pytest.raises(SystemExit, match="recomputation against FREEZE_orbit.json"):
        mod.main(["--arms", "a0", "--kill-first", str(first)])
    for rec in ({"v1_reference": {"kind": "frozen"}, "kill": {"passed": True}},
                {"v1_reference": {"kind": "candidate"}, "kill": None}, {"kill": {"passed": True}}):
        first.write_text(json.dumps(rec))
        with pytest.raises(SystemExit, match="not the record of a kill criterion"):
            mod.main(["--arms", "a0", "--v1-compare", "run_x", "--v1-reference", str(frozen),
                      "--kill-first", str(first)])
    assert not (tmp_path / "runs").exists()

