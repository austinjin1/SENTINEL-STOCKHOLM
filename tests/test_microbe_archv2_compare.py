# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests of scripts/eval/microbe_arch_v2_compare.py on synthetic tables (no real data, no fit)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.models.microbe import archv2 as AV
from sentinel.models.microbe import stamp as ST

REPO = paths.REPO_ROOT
P, G = list(AV.FAMILIES["P"]), list(AV.FAMILIES["G"])


def _load():
    spec = importlib.util.spec_from_file_location("microbe_arch_v2_compare_under_test",
                                                  REPO / "scripts/eval/microbe_arch_v2_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def box(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(paths, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    paths.REGISTRY_PATH.write_text("")
    rng = np.random.default_rng(0)
    rows = [(f"s{d}_{k}_{i}", f"D{d:02d}", f"D{d:02d}S{k}") for d in range(6) for k in range(3) for i in range(8)]
    S = pd.DataFrame(rows, columns=["sample_id", "domain_id", "site_id"]).set_index("sample_id")
    for t in P + G:
        S[t] = rng.normal(size=len(S)) + S["domain_id"].str[1:].astype(int).to_numpy() * 0.5
    S.loc[S.index[:5], P[0]] = np.nan                      # a few missing labels
    return _load(), S, tmp_path, rng


def _table(S, targets, rng, noise) -> pd.DataFrame:
    return pd.DataFrame({t: S[t].fillna(0.0).to_numpy() + rng.normal(scale=noise, size=len(S)) for t in targets},
                        index=S.index)


def _write(df, name, fam, old_track=False):
    d = paths.RUNS_ROOT / "microbe" / ("cv" if old_track else "cv_archv2") / "lodo"
    d = d if old_track else d / fam
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{name}.parquet")


def _rows():
    return [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines() if x.strip()]


def test_candidate_is_compared_with_every_comparator_that_has_a_table(box):
    C, S, tmp, rng = box
    _write(_table(S, G, rng, 0.2), "xgb_v4clr", "G")                       # the candidate: good, one table
    weak = [_table(S, G, rng, 2.5) for _ in C.SEEDS]
    for s, df in zip(C.SEEDS, weak):
        _write(df, f"xgb_clr_seed{s}", "G")
    _write(sum(weak) / len(weak), "xgb_clr", "G")                           # stochastic comparator, 3 seeds
    _write(_table(S, G, rng, 0.8), "wa_tol", "G")                          # deterministic comparator
    _write(_table(S, G, rng, 0.05), "xgb_nichez_context", "G")             # not a comparator of this candidate
    out = C.run(["xgb_v4clr", "rf_v4clr"], ["G"], S, n_boot=400, stamp_row="teststamp")["G"]
    assert out["rf_v4clr"] == {"not_run": True}
    rec = out["xgb_v4clr"]
    assert rec["comparators_run"] == ["xgb_clr", "wa_tol"] and "xgb_v4clr" not in rec["vs"]
    assert set(rec["comparators_not_run"]) == set(C.PARITY) - {"xgb_v4clr", "xgb_clr", "wa_tol"}
    assert rec["has_seed_variation"] is False
    for m in ("xgb_clr", "wa_tol"):
        for t in G:
            o = rec["vs"][m][t]
            assert o["diff"] > 0 and o["beats"] and len(o["diff_per_seed"]) == 3 and o["positive_in_every_seed"]
            assert o["ci_site"][0] > 0 and o["ci_domain"][0] > 0
    assert rec["targets_beating_every_comparator_run"] == G
    assert rec["targets_beating_every_comparator_run_in_every_seed"] == G
    # the per-seed differences against the stochastic comparator differ by seed; against the single table they do not
    assert len(set(np.round(rec["vs"]["xgb_clr"][G[0]]["diff_per_seed"], 12))) == 3
    assert len(set(np.round(rec["vs"]["wa_tol"][G[0]]["diff_per_seed"], 12))) == 1
    for t in G:                                             # strongest simple baseline: the better comparator
        b = rec["strongest_simple_baseline"][t]
        assert b["model"] == "wa_tol"
        assert b["point_diff"] == pytest.approx(rec["r2"][t] - b["r2"])
    rows = _rows()
    assert len(rows) == 2 * len(G) + len(G)
    assert {r["experiment"] for r in rows} == {C.EXPERIMENT}
    assert {r["config"]["stamp_row"] for r in rows} == {"teststamp"}
    assert sum(r["metric"] == "r2_point_diff_vs_strongest_simple_baseline" for r in rows) == len(G)
    one = next(r for r in rows if r["metric"] == "r2_diff_vs_xgb_clr")
    assert one["extra"]["reference_has_seed_variation"] is True and one["extra"]["beats"] is True


def test_a_worse_candidate_beats_nothing_and_the_extras_join_only_for_d1(box):
    C, S, tmp, rng = box
    _write(_table(S, P, rng, 1.5), "niche_s", "P")
    _write(_table(S, P, rng, 0.3), "xgb_v4clr", "P", old_track=True)       # family P falls back to the old track
    _write(_table(S, P, rng, 0.6), "xgb_nichez_context", "P")              # ruling B3: joins for D1, if run
    rec = C.run(["niche_s"], ["P"], S, n_boot=400, log=False)["P"]["niche_s"]
    assert rec["comparators_run"] == ["xgb_v4clr", "xgb_nichez_context"]
    assert "xgb_v4mr" not in rec["comparators_not_run"]                    # an extra that was not run is not owed
    assert rec["targets_beating_every_comparator_run"] == []
    assert all(rec["vs"]["xgb_v4clr"][t]["diff"] < 0 and not rec["vs"]["xgb_v4clr"][t]["beats"] for t in P)
    assert all(v["model"] == "xgb_v4clr" and v["point_diff"] < 0 for v in rec["strongest_simple_baseline"].values())
    assert rec["vs"]["xgb_v4clr"][P[0]]["n"] == len(S) - 5                 # rows without a label are left out
    assert _rows() == []                                                   # log=False writes nothing
    other = C.run(["xgb_v4clr"], ["P"], S, n_boot=400, log=False)["P"]["xgb_v4clr"]
    assert other["comparators_run"] == []                                  # the control is no comparator of a V4 table
    assert other["targets_beating_every_comparator_run"] == []             # nothing run, nothing beaten


def test_main_refuses_without_the_stamp(box, monkeypatch):
    C, S, tmp, rng = box
    with pytest.raises(ST.StampError):
        C.main(["--candidates", "niche_s"])
    assert _rows() == []
